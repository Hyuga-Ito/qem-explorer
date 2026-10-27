# Commands — protocol-selection study (1.9.0)

Every command runs offline on emulated fake devices; none contacts a quantum provider, and none submits hardware jobs.
On Linux set `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048` first (README, Linux note). Worker processes get one
thread per math library automatically (1.9.0); set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for
the parent too when you run `analyze` beside other jobs. All text files are read and written as UTF-8 regardless of the
platform's locale. Windows (PowerShell) versions of the commands below: `docs/WINDOWS_FIRST_RUN.md`.

## Exit codes (1.9.0)

| code | meaning | commands |
|---|---|---|
| 0 | complete; or deliberately partial with `--limit` and nothing failed; or a once-only step already done and verified with `--reuse` (`freeze --reuse`, `freeze --complete --reuse`, `final --reuse`, `confirm --reuse`) | all |
| 1 | incomplete: failed tasks (timeout, memory, crash, error), not started (ceiling, interruption) or missing; or, without `--reuse`, a once-only step that was already done (verified, unchanged) | `label`, `run`, `freeze`, `final`, `confirm`, `regimes`, `mismatch`, `status`, `pipeline`, `scripts/root_audit.py label` |
| 2 | refused: invalid configuration, binding or freeze mismatch, admission (a cap), corrupted or stale inputs, a locked run, an incomplete population for analysis or replay | all |
| 3 | `plan`: a configured cap would refuse the run; `recommend`: unreadable or unsupported input | `plan`, `recommend` |

## A study with the pipeline runner (recommended)

```bash
python -m qemx.selection check-config configs/selection/pilot19.yaml configs/pipelines/pilot19.yaml
python -m qemx.selection plan configs/selection/pilot19.yaml --workers auto --tasks-csv runs/pilot19_tasks.csv
python -m qemx.selection pipeline configs/pipelines/pilot19.yaml --out runs/pilot19 --workers auto
python -m qemx.selection status --out runs/pilot19
```

`pipeline FILE` runs the stages listed in the file in order — `plan`, `label`, `studies-label`, `scripts-label`,
`analyze`, `freeze`, `nested`, `freeze-complete`, `bundles`, `confirm`, `replay`, `studies-analyze`,
`scripts-analyze`, `recommend-check`, `profile`, `figures`, `report`, `tables`, `status` — each as an argument array
(no shell). On a rerun it verifies and keeps what is done: complete labels bound to the configuration are not
relabeled, an existing freeze or once-only evaluation is verified (`--reuse`) and kept, and any failure stops every
dependent stage with that stage's exit code. Options: `--until STAGE`, `--only A,B`, `--dry-run` (print the commands),
`--python EXE`, and the pass-through flags `--retry-failed`, `--override-limits "reason"`, `--reset-analysis`,
`--accept-changes`. Files named in a pipeline are resolved next to the pipeline file, then in the repository. Shipped
pipelines: `configs/pipelines/smoke19.yaml` (minutes), `pilot19.yaml` (about an hour), `fulldev19.yaml` (planned;
3–9 CPU-hours of labels, no confirmation stage), `reproduce_v18.yaml` (the 1.8.0 study into a new directory).

## Single commands

```bash
python -m qemx.selection check-config FILE [FILE ...]          # any YAML format, by its own parser (YAML_FORMATS.md)
python -m qemx.selection plan CONFIG [--out RUN] [--measured LABELS_DIR] [--workers N|auto] [--tasks-csv F] [--json F]
                                                               # + consumed CPU (1.9.1) and the analysis workload (fits)
python -m qemx.selection label CONFIG --out RUN [--workers N|auto] [--limit K] [--subdir NAME]
        [--retry-failed] [--quarantine-corrupted] [--override-limits "reason"] [--accept-changes]
python -m qemx.selection status --out RUN [--stage dev|final|confirm|NAME] [--check exists|digest|load] [--json F]
                                                               # expected / completed / failed / missing / corrupted,
                                                               # labeling time consumed (every attempt; 1.9.1)
python -m qemx.selection analyze CONFIG --out RUN [--splits ...] [--extras ...]      # '--extras=' for none (any shell)
        [--allow-partial] [--reset-analysis] [--accept-changes]
        # 1.9.1: deployed/selective/nested_selective are deferred until the freeze unless analysis.deployed_family
        # and deployed_set are declared; after the freeze: --splits= --extras deployed,nested_selective
python -m qemx.selection freeze CONFIG --out RUN [--complete] [--reuse] [--accept-changes]
        # first freeze (1.9.1): the analysis must match this configuration, labels, data and code, cover every
        # declared split and grid configuration; the choice stays within the grid; another code's analysis needs
        # --accept-changes (recorded). Frozen run: verified → 1 (0 with --reuse); inconsistent → 2
python -m qemx.selection confirm CONF --out RUN --dev-config DEV --train dev [--reuse]
python -m qemx.selection final CONFIG --out RUN [--reuse]
python -m qemx.selection bundle CONFIG --out RUN --version NAME [--feature-set FS] [--train dev] [--bundle-out DIR]
                                                               # default RUN/bundles/NAME; never overwrites
python -m qemx.selection recommend --experiment FILE --budget B [--device D] [--bundle DIR] [--allow-simulation]
python -m qemx.selection profile CONFIG --out RUN [--bundle DIR]   # recommendation runtime and memory
python -m qemx.selection figures CONFIG --out RUN [--plot-config FILE] [--figures-out DIR]
python scripts/replay_confirmation.py --labels RUN/confirm --out RUN/replay [--bundle-dir RUN/bundles]
        [--bundles NAME:sim,NAME:nosim] [--subset FILE --subset-reason "why"]
python scripts/execution_audit.py --out DIR [--roots 5] [--trials 600]      # recommend → design → execution (R3)
python scripts/verify_trials.py reports/selection_v180      # or: --attachments ARCHIVE.zip ... (studies by reports/TRIAL_ATTACHMENTS.json)
python -m qemx.selection.tables19 [--v18 DIR] [--audit DIR] [--pilot DIR] [--out DIR] [--no-pdf]
```

* **Failures** are kept as records and reported on every rerun; `--retry-failed` relabels them after you changed what
  caused them (e.g. `limits.task_seconds`, operational: a freeze stays valid). Each attempt is in `attempts.jsonl`;
  replaced failure records in `failed_attempts/`. 1.9.1: tasks stopped by Ctrl+C/SIGTERM are logged as `interrupted`
  and every attempt stays in the consumed-time ledger that `limits.cpu_hours` is checked against.
* **Corrupted or missing arrays** stop labeling; `--quarantine-corrupted` moves the records to `quarantine/` (listed in
  `run.json`) and relabels them.
* **Caps** (`limits.cpu_hours`, `disk_gb`, `min_free_disk_gb`) are checked before labeling against the estimate from
  measured rates (the run's own, else `--measured`); unknown estimates are reported as unknown. `--override-limits`
  records a deliberate start; the monitored ceiling still stops at the cap (running tasks finish).
* **Partial analyses** (`--allow-partial`) are marked and cannot be frozen. An analysis made with another
  configuration or other data is refused (`--reset-analysis` archives it, never for a frozen run); other code needs
  `--accept-changes` (recorded).


## The 1.8.0 study, end to end (historical; the sequence executed for 1.8.0)

`scripts/reproduce_v18.sh` (1.9.0: `python -m qemx.selection pipeline configs/pipelines/reproduce_v18.yaml`) runs the
same sequence resumably into a new directory. Every configuration below was declared (and the development and
confirmation sets registered in `configs/selection/ancestry_registry.json`) before any of its labels existed.

```bash
R=runs/v18
# 0. smoke test of the 17-candidate pipeline (label, analyze, freeze, bundle, recommend, figures)
python -m qemx.selection label configs/selection/smoke18.yaml --out runs/smoke18 --workers 2
python -m qemx.selection plan configs/selection/dev18.yaml

# 1. development labels (88 tasks, 44 new logical circuits, seed scheme 2, 17 candidates)
python -m qemx.selection label configs/selection/dev18.yaml --out $R --workers 2

# 2. stage-2 crossed circuit × preparation-root audit (8 circuits × 4 roots × 3 budgets; exploratory)
python scripts/root_audit.py label --out $R/audit18 --workers 1

# 3. controlled studies (labels only; analyzed with the frozen recommender after step 5)
python -m qemx.selection label configs/selection/noisecomp18.yaml --out $R --workers 1 --subdir noisecomp
python -m qemx.selection label configs/selection/regimes18.yaml --out $R --workers 1 --subdir regimes

# 4. development analysis: every split and the extras that do not depend on the frozen choice
python -m qemx.selection analyze configs/selection/dev18.yaml --out $R --extras ablation,learning_curve,budget_generalization,abstention,interval_coverage,certificates,floor_sensitivity,design_diagnostics

# 5. freeze (lowest pooled out-of-fold mean log10 risk), nested selective evaluation, deployment threshold
python -m qemx.selection freeze configs/selection/dev18.yaml --out $R
python -m qemx.selection analyze configs/selection/dev18.yaml --out $R --splits "" --extras deployed,nested_selective
python -m qemx.selection freeze configs/selection/dev18.yaml --out $R --complete

# 6. confirmation on 29 new logical circuits: labeled after the freeze, evaluated once
python -m qemx.selection confirm configs/selection/conf18.yaml --out $R --dev-config configs/selection/dev18.yaml --train dev --workers 1

# 7. bundles: the frozen configuration and the default deployable-feature variant (as executed for the release, into
#    qemx/selection/bundles/; scripts/reproduce_v18.sh writes them into $R/bundles with --bundle-out instead, so a
#    reproduction never overwrites the shipped bundles, and passes --bundle-dir $R/bundles to the replay)
python -m qemx.selection bundle configs/selection/dev18.yaml --out $R --version qemx-rec-1.8.0 --train dev
python -m qemx.selection bundle configs/selection/dev18.yaml --out $R --version qemx-rec-1.8.0-deployable --feature-set deployable18 --train dev

# 8. the full deployed policy on the confirmation set (deploy.recommend itself, both bundles; finding F1)
python scripts/replay_confirmation.py --labels $R/confirm --out $R/replay

# 9. original (8) against expanded (17) portfolio on identical development labels
python scripts/portfolio_comparison.py --labels $R/dev --out $R/portfolio

# 10. controlled studies and the audit, analyzed with the frozen recommender / the deployable bundle
python -m qemx.selection regimes configs/selection/noisecomp18.yaml --out $R --skip-label --subdir noisecomp
python -m qemx.selection regimes configs/selection/regimes18.yaml --out $R --skip-label --subdir regimes
python scripts/root_audit.py analyze --out $R/audit18 --bundle qemx/selection/bundles/qemx-rec-1.8.0-deployable

# 11. classical cost by component, early rejection; figures, thesis tables, validation
python -m qemx.selection profile configs/selection/dev18.yaml --out $R --widths 3,4,5,6,7
python -m qemx.selection figures configs/selection/dev18.yaml --out $R --plot-config configs/selection/figures18.yaml
python -m qemx.selection report configs/selection/dev18.yaml --out $R
python -m qemx.selection.thesis_tables --run $R --out $R/thesis_tables
python scripts/validate_figures.py $R

# 12. hardware validation plan of the default bundle: a dry run (nothing is submitted)
python -m qemx.selection hardware plan --hardware-config configs/selection/hardware17.yaml --bundle qemx/selection/bundles/qemx-rec-1.8.0-deployable --out $R/hardware_plan
```

Independent check of the headline numbers: a separate agent recomputed the confirmation, full-policy, model-component
and portfolio numbers from `confirm/labels.csv`, `confirm/choices_confirm.csv`, `replay/*.csv` and
`analysis/labels.csv` with its own code (private audit archive, `evidence/independent_repro/repro.py`); the metric is
defined in `docs/selection/METHODS.md` §16.6 and needs only pandas and NumPy.

The shipped run is `reports/selection_v180/` (records, tables, figures, thesis tables; the per-trial arrays are in the
separate trial-data archives, verified by `reports/selection_v180/TRIALS_SHA256SUMS`). The exploratory full-policy
replay of the 1.8.0 policy on the **1.7.0** confirmation (post-inspection, not confirmatory) is
`scripts/replay_v170_confirmation.py` (evidence in the private audit archive).

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

The output is the same dictionary the Playground shows (policy `qemx.selection.policy/2`, 1.8.0): the choice,
whether the model's choice was accepted (margin against the frozen threshold) or replaced by the budget-only fallback
SBS-B, out-of-distribution reasons, the **preflight** status of every candidate (verified-feasible, infeasible or
unverified, with its level: structural for every candidate; the candidates reached in the model's order are resolved at
the design level — a learning protocol's training design is built with its deployment seed — or, with
`--allow-simulation`, at the allocation level on the supplied model), the candidates skipped because they were
unverified or infeasible, the execution note (the method seed that executes the verified design), the predicted log10
risks (model-conditional), and every feature with its information class and cost. Unverified candidates are never
accepted. Inputs beyond the bundle's bounded-cost limits are rejected before compilation (`Unsupported`); simulated
features are computed only with `--allow-simulation`.

```bash
python -m qemx.selection registry --markdown          # the 17 candidates' capability records (CANDIDATES.md)
python -m qemx.selection registry --candidate SV       # one record as JSON
```

## Hardware validation (prepared, never submitted automatically)

```bash
python -m qemx.selection hardware plan --hardware-config configs/selection/hardware17.yaml --bundle qemx/selection/bundles/qemx-rec-1.8.0-deployable --out runs/hw
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

| configuration | purpose | tasks / independent base circuits | executed |
|---|---|---|---|
| `smoke18.yaml` | 1.8.0 pipeline check with the 17 candidates (3 budgets, r = 12, 2 designs per half) | 10 / 5 | 1.8.0 |
| `dev18.yaml` | 1.8.0 development set: new circuits, 17 candidates, seed scheme 2, 6 designs and r = 60 per half, 10 budgets | 88 / 44 | 1.8.0 |
| `conf18.yaml` | 1.8.0 once-only confirmation on new circuits | 58 / 29 | 1.8.0 |
| `regimes18.yaml` | 1.8.0 controlled regime study (noise scale s, observables, layouts; the regimes17 slices relabeled) | 21 / 3 | 1.8.0 |
| `noisecomp18.yaml` | 1.8.0 noise composition at matched light-cone exposure | 18 / 3 | 1.8.0 |
| `scripts/root_audit.py` | 1.8.0 crossed circuit × preparation-root audit | 8 × 4 roots | 1.8.0 |
| `smoke.yaml` | pipeline check (fresh designs, 3 per half; 3 budgets, r = 24) | small | CI and locally |
| `pilot17.yaml` | corrected pilot: the 1.6.0 tasks relabeled with fresh designs (10 per half), 10 budgets, r = 100 + 100 | 120 dev / 40 + 36 historical final / 12 | 1.7.0 |
| `regimes17.yaml` | controlled regime study (noise scale s, observables, layouts) | 21 / 3 | 1.7.0 |
| `noisecomp17.yaml` | noise composition: one component at a time | 30 / 3 | 1.7.0 |
| `mismatch17.yaml` | hidden discrepancies, equal-budget reactions, held-out discrepancy types | 10 base circuits × scenarios | 1.7.0 (not repeated in 1.8.0) |
| `confirm17.yaml` | once-only confirmation on new logical circuits | 50 / 25 | 1.7.0 |
| `full17.yaml` | larger study (242 logical circuits, 726 tasks, caps 80 CPU-h, 10 GB) | planned | **no** (runnable plan) |
| `hardware17.yaml` | hardware transfer check (2 tasks, 4 blocks, B = 20 000; 600 s shared ceiling, 60 s reserve) | planned | **no** (owner authorization; dry-run plan only) |
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
  design_study/            focused experiment B (designs.csv, summary.csv; 1.7.0)
  replay/                  1.8.0: the deployed policy on the confirmation, per bundle (rows, summaries)
  portfolio/               1.8.0: original against expanded portfolio (contrasts, policies)
  audit18/                 1.8.0: root audit labels (root_<seed>/) and analysis/
  thesis_tables/           1.8.0: CSV, LaTeX fragments, Markdown, standalone PDF, provenance
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
