# Corrected pilot of the protocol-selection study (QEM Explorer 1.7.0)

The run directory of `configs/selection/pilot17.yaml` and its companion studies, labeled with fresh, independent
learning designs (review R1) and analyzed with the 1.7.0 protocol. **Simulated observations** (emulated trials on IBM
fake devices, exact density-matrix engine); no hardware data. Interpretation: `docs/selection/RESULTS.md`; figures:
`docs/selection/FIGURES.md`; column definitions: `docs/selection/DATA_DICTIONARY.md`; commands:
`docs/selection/COMMANDS.md`. The 1.6.0 pilot (`reports/selection_pilot/`) is the historical dataset with the
fixed-shared design estimand.

| path | content |
|---|---|
| `manifest.json` | the configuration and the task lists (development; historical final test) |
| `dev/` | 120 development tasks (40 independent base circuits): `tasks/<task_id>.json` (record: specification, budgets, allocations, design plan, failure flags, provenance), `run.json` (binding: configuration, manifest, split, portfolio, device snapshots, code, environment), `provenance.json`, `label_summary.json` |
| `freeze.json` | the frozen configuration, chosen on development data only, with the nested-CV threshold τ and the digests of the analyses it was chosen from |
| `final/` | the 36 tasks of the 1.6.0 final test, relabeled with fresh designs and evaluated once with the frozen configuration — **historical, not confirmatory** (its circuits were seen in 1.6.0) |
| `confirm/` | the confirmation: 50 tasks on 25 new logical circuits (registered before labeling, no ancestry overlap), labeled after the freeze and evaluated once (`EVALUATED`, `metrics_confirm.csv`) |
| `regimes/`, `noisecomp/` | controlled studies on `fake_fez`: the error scale s, observable and layout blocks (21 tasks); one noise component changed at a time (30 tasks); `analysis/` holds the cells, per-protocol risks and features |
| `mismatch/` | 90 tasks (10 circuits × 9 execution laws), diagnostics, equal-budget reactions with held-out discrepancy types (`analysis/equal_budget.csv`, `reactions.csv`, `discrimination.csv`) |
| `analysis/` | every evaluation table of the development data (splits, pooled results, optimism, ablations, learning curve, budget generalization, nested selective evaluation, interval coverage, certificates, design variance and diagnostics, failure profile, runtime) |
| `analysis_estimands*/` | the 1.6.0 labels against the corrected labels of the same tasks and seeds (`scripts/compare_design_estimands.py`): identity of the protocols without a training design, and the CDR/NIL estimand change |
| `design_study/` | focused experiment B: CDR/NIL training allocations on five tasks, 20 designs per variant (`scripts/design_study.py --designs 20 --trials 10`) |
| `hardware/` | the hardware validation plan of the shipped bundle — **planned, not executed**; nothing was submitted |
| `figures/` | six main and nine supplementary figures (PDF, SVG, PNG), each with its notes (`.md`: question, methods box, caption, interpretation, inputs, command) and plotted data (`__*.csv`) |
| `tables/`, `RESULTS_TABLES.md` | Tables 1, 1b, 2, 3, 3h and S1, and the results tables quoted in `docs/selection/RESULTS.md` |
| `TRIALS_SHA256SUMS` | SHA-256 of every per-trial array (`*.npz`) of this run |

## Per-trial arrays

The per-trial arrays (`<part>/tasks/<task_id>.npz`: estimates, flags and learning-design indices of both label halves;
347 files, about 33 MB) are not in the source archive, to keep it small. They are published as two trial-data
archives, `qem-explorer-1.7.0-trials-1.zip` (development and historical final test) and
`qem-explorer-1.7.0-trials-2.zip` (confirmation, regimes, noise composition, mismatch), whose paths are relative to the
repository root. Extract both there and verify:

```bash
cd reports/selection_v170 && sha256sum -c TRIALS_SHA256SUMS
```

Without them, everything except the per-trial drill-down in the app, the two figures drawn from individual trials
(`supp_bias_variance`, `supp_representative`: reported as pending, the shipped versions stay) and the re-analysis
from trials works from the tables and records in this directory.

## Regenerate

Figures and tables from the saved tables (no simulation, a few minutes):

```bash
python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170
python -m qemx.selection report  configs/selection/pilot17.yaml --out reports/selection_v170
python scripts/validate_figures.py reports/selection_v170
```

Re-analysis from the trials (needs the trial-data archive; no simulation; about two hours on two cores): copy this
directory and run steps 2–9 of `docs/selection/COMMANDS.md` on the copy. `final` and `confirm` are evaluated once per
run directory (`EVALUATED`) and refuse to run again here. Relabeling from scratch: `scripts/reproduce_pilot_17.sh`
(measured: about 10 CPU-hours of labeling for the 347 tasks — dev 3.5, historical final 2.0, confirmation 1.8,
mismatch 1.9, regimes 0.4, noise composition 0.4 — plus about 2 hours of analysis on two cores).
