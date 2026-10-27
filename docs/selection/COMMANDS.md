# Commands — protocol-selection study (1.7.0)

Every command runs offline on emulated fake devices; none contacts a quantum provider, and none submits hardware jobs.
On Linux set `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048` first (README, Linux note) and one thread per math
library (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`) when running several workers. All text files
are read and written as UTF-8 regardless of the platform's locale.

## The corrected pilot, end to end (the sequence executed for 1.7.0)

```bash
R=runs/pilot17
# 0. plan: tasks, written base circuits, independent logical circuits, variants, designs per half, trials per design,
#    shots per deployment, overlap with earlier studies (ancestry registry), CPU/disk estimate and caps
python -m qemx.selection plan configs/selection/pilot17.yaml
python -m qemx.selection plan configs/selection/full17.yaml --measured $R/dev

# 1. labels (resumable; bound to their run: a changed configuration, manifest, split, portfolio or device snapshot is
#    refused, a changed code or environment needs --accept-changes and is recorded)
python -m qemx.selection label configs/selection/pilot17.yaml --out $R --workers 2
python -m qemx.selection label configs/selection/pilot17.yaml --out $R --workers 2 --final-set   # historical final
python -m qemx.selection label configs/selection/regimes17.yaml --out $R --workers 2 --subdir regimes
python -m qemx.selection label configs/selection/noisecomp17.yaml --out $R --workers 2 --subdir noisecomp
python -m qemx.selection mismatch configs/selection/mismatch17.yaml --out $R --workers 2           # labels only

# 2. development analysis (every split; the extras that do not depend on the frozen choice), from saved trials only
python -m qemx.selection analyze configs/selection/pilot17.yaml --out $R --extras ablation,learning_curve,budget_generalization,abstention,interval_coverage,certificates,floor_sensitivity,design_diagnostics

# 3. freeze the configuration (lowest pooled out-of-fold mean log10 risk), evaluate it as deployed and by nested CV,
#    then add the deployment threshold (deploy_tau) and the digest of the nested results
python -m qemx.selection freeze configs/selection/pilot17.yaml --out $R
python -m qemx.selection analyze configs/selection/pilot17.yaml --out $R --splits "" --extras deployed,nested_selective
python -m qemx.selection freeze configs/selection/pilot17.yaml --out $R --complete

# 4. historical re-evaluation of the 1.6.0 final test with the frozen configuration (once; not confirmatory)
python -m qemx.selection final configs/selection/pilot17.yaml --out $R

# 5. controlled studies with the frozen recommender (trained on dev + final without the studied circuits)
python -m qemx.selection regimes configs/selection/regimes17.yaml --out $R --skip-label
python -m qemx.selection regimes configs/selection/noisecomp17.yaml --out $R --skip-label --subdir noisecomp
python -m qemx.selection mismatch configs/selection/mismatch17.yaml --out $R --dev $R/dev

# 6. confirmation on new logical circuits: labeled after the freeze, evaluated once (refuses any ancestry overlap)
python -m qemx.selection confirm configs/selection/confirm17.yaml --out $R --dev-config configs/selection/pilot17.yaml --workers 2

# 7. the model bundles shipped with the application (frozen configuration; deployable-feature variant)
python -m qemx.selection bundle configs/selection/pilot17.yaml --out $R --version qemx-rec-1.7.0
python -m qemx.selection bundle configs/selection/pilot17.yaml --out $R --version qemx-rec-1.7.0-deployable --feature-set deployable

# 8. focused experiment B (learning-design variability and training allocations; exploratory)
python scripts/design_study.py --out $R/design_study --designs 20 --trials 10

# 9. runtime of a recommendation by component, figures and tables (from saved tables only)
python -m qemx.selection profile configs/selection/pilot17.yaml --out $R --widths 3,4,5,6,7
python -m qemx.selection figures configs/selection/pilot17.yaml --out $R
python -m qemx.selection report configs/selection/pilot17.yaml --out $R
```

`scripts/reproduce_pilot_17.sh` runs the same sequence. Figures can be redrawn one at a time
(`figures CONFIG --out R --only fig01_budget_efficiency,supp_runtime`); a figure whose inputs are missing is listed in
`figures/PENDING.md` instead of being drawn (`supp_bias_variance` and `supp_representative` need the per-trial arrays).
After drawing, `python scripts/validate_figures.py R` parses every SVG and PDF, checks the PNGs and the notes, and
compares plotted numbers with the analysis tables.

The shipped run is `reports/selection_v170/` (records, tables, figures; the per-trial arrays are in the separate
trial-data archive, verified by `reports/selection_v170/TRIALS_SHA256SUMS`).

## One recommendation (the Playground's recommender, from the command line)

```bash
# a Playground state (Export → JSON) or an experiment document; the device snapshot and the total budget
python -m qemx.selection recommend --state my_state.json --budget 100000 --device fake_fez
python -m qemx.selection recommend --experiment examples/experiments/qft_known_answer.yaml --budget 100000 --device fake_fez --allow-simulation
```

The output is the same dictionary the Playground shows: the choice, whether the model's choice was accepted (margin
against the frozen threshold) or replaced by the budget-only fallback SBS-B, out-of-distribution reasons, structural
feasibility per protocol, the predicted log10 risks (model-conditional), and every feature with its information class
and cost. Simulated features are computed only with `--allow-simulation`.

## Hardware validation (prepared, never submitted automatically)

```bash
python -m qemx.selection hardware plan --hardware-config configs/selection/hardware17.yaml --bundle qemx/selection/bundles/qemx-rec-1.7.0 --out runs/hw
python -m qemx.selection hardware check --plan runs/hw/hardware_plan.json
```

Execution needs the owner's authorization of that exact plan (`QEMX_HW_AUTHORIZE=<digest>`), a campaign ledger with one
shared ceiling (`python -m qemx.campaign init --ceiling 600 --reserve 60`) and credentials from `QISKIT_IBM_TOKEN`, a
saved account or a hidden prompt — never a command-line argument. See `docs/selection/HARDWARE.md`.

## Other commands

```bash
python -m qemx.selection validate --out runs/validation --tasks 3 --B 2000 --R 300   # emulator vs shot-by-shot execution
python -m qemx.selection run configs/selection/smoke.yaml --out runs/smoke           # smoke: label + analyze + figures
python -m qemx.selection analyze configs/selection/pilot17.yaml --out runs/pilot17 --splits family,device --extras ""
python -m qemx.selection analyze configs/selection/pilot17.yaml --out runs/pilot17 --splits "" --extras nested_selective
python scripts/check_text_io.py                                                      # static UTF-8 I/O check
```

## Workflows

| configuration | purpose | tasks / independent base circuits | executed in 1.7.0 |
|---|---|---|---|
| `smoke.yaml` | pipeline check (fresh designs, 3 per half; 3 budgets, r = 24) | small | yes (CI and locally) |
| `pilot17.yaml` | corrected pilot: the 1.6.0 tasks relabeled with fresh designs (10 per half), 10 budgets, r = 100 + 100 | 120 dev / 40 + 36 historical final / 12 | see RESULTS |
| `regimes17.yaml` | controlled regime study (noise scale s, observables, layouts) | 21 / 3 | see RESULTS |
| `noisecomp17.yaml` | noise composition: one component at a time | 30 / 3 | see RESULTS |
| `mismatch17.yaml` | hidden discrepancies, equal-budget reactions, held-out discrepancy types | 10 base circuits × scenarios | see RESULTS |
| `confirm17.yaml` | once-only confirmation on new logical circuits | 50 / 25 | see RESULTS |
| `full17.yaml` | larger study (242 logical circuits, 726 tasks, caps 80 CPU-h, 10 GB) | planned | **no** (runnable plan) |
| `hardware17.yaml` | hardware transfer check (2 tasks, 4 blocks, B = 20 000; 600 s shared ceiling, 60 s reserve) | planned | **no** (owner authorization) |
| `pilot.yaml`, `regimes.yaml`, `mismatch.yaml`, `full.yaml` | the 1.6.0 configurations (fixed shared designs), kept for the historical data | — | 1.6.0 only |

## Outputs

```
runs/<name>/
  manifest.json            configuration and task list (development, final)
  <part>/run.json          the run's binding: digests of the label configuration, manifest, split, portfolio, device
                           snapshots; code and environment per session; imports (part = dev, final, regimes, …)
  <part>/provenance.json   one entry per labeling session (versions, digests, what changed on resume)
  <part>/tasks/<id>.json   label record: status, allocations (ok / infeasible / method-error), ledgers, costs,
                           features, design records (seed, identity, diagnostics, fallback), digest of the arrays
  <part>/tasks/<id>.npz    per-trial estimates, flags and design indices of both halves
  analysis/*.csv           source data of every figure and table (docs/selection/DATA_DICTIONARY.md)
  analysis/label_failures.csv  label tasks without labels (error, timeout, memory), never silently dropped
  freeze.json              the frozen configuration (schema freeze/2; with --complete: deploy_tau, nested digest)
  final/, confirm/         the once-only evaluations (summary, choices, labels, metrics)
  regimes/, noisecomp/     controlled studies and their analysis/
  mismatch/                scenario labels, diagnostics.csv, policies.csv, analysis/ (equal-budget reactions)
  design_study/            focused experiment B (designs.csv, summary.csv)
  figures/*.{pdf,svg,png,md} and figures/*__*.csv, figures/PENDING.md (absent when every figure was drawn)
  tables/*.{csv,md}
```

## Resource limits and checkpoints

Each task runs in its own supervised child process (spawn) on every platform, also with one worker. The parent
enforces the wall-clock limit (`limits.task_seconds`) and a resident-memory watchdog over the child's process tree
(`limits.memory_gb`); on POSIX the child also limits its own address space. A task that exceeds a limit, raises or
crashes is recorded as `timeout`, `memory` or `error` and the run continues; `--retry-failed` runs those tasks
again. Writes are atomic (temporary file, then rename). A run can be interrupted at any time and restarted with the
same command. `plan` compares the estimated CPU-hours and disk with the configuration's caps (`limits.cpu_hours`,
`limits.disk_gb`).
