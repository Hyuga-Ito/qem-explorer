# Protocol-selection study of QEM Explorer 1.8.0

The run directory of `configs/selection/dev18.yaml` (development), `conf18.yaml` (confirmation) and the companion
studies of 1.8.0, with the 17-candidate portfolio, seed scheme 2 (namespaced preparation streams) and the
execution/preflight contract (`qemx/selection/preflight.py`). **Simulated observations** (emulated trials on IBM fake
devices, exact density-matrix engine); no hardware data. Interpretation: `docs/selection/RESULTS.md` (1.8.0 part) and
`docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md`; figures: `docs/selection/FIGURES.md`; column definitions:
`docs/selection/DATA_DICTIONARY.md`; commands: `docs/selection/COMMANDS.md`. The earlier studies stay where they were
(`reports/selection_v170/`, `reports/selection_pilot/`).

| path | content |
|---|---|
| `manifest.json` | the configuration and the task lists |
| `dev/` | 88 development tasks on 44 new logical (base) circuits: `tasks/<task_id>.json` (record: specification, budgets, allocations, design plan, preflight status, failure flags, provenance), `run.json` (binding: configuration, manifest, split, portfolio, device snapshots, seed scheme, code, environment), `provenance.json`, `label_summary.json` |
| `freeze.json` | the frozen configuration `MODEL[hgb\|deployable18+simulated]`, chosen on development data only, with the nested-CV threshold τ and the digests of the analyses it was chosen from |
| `confirm/` | the confirmation: 58 tasks on 29 new logical circuits (registered before labeling, no ancestry overlap), labeled after the freeze and evaluated once (`EVALUATED`, `metrics_confirm.csv`) |
| `replay/` | the full deployed policy (`deploy.recommend` itself, policy `qemx.selection.policy/2`) of both shipped bundles on every confirmation row, with the model-component benchmark (`replay_<bundle>.csv/.json`, `replay_summary.json`) |
| `portfolio/` | the original 8 against the expanded 17 candidates on identical development labels (`portfolio_comparison.json`, `portfolio_contrasts.csv`) — exploratory |
| `audit18/` | the crossed circuit × preparation-root audit: 8 circuits × 4 independent roots × 3 budgets (`root_<seed>/`), and `analysis/` (label stability against the design-level SE, ranking stability, the execution contract against each root's allocation, the deployed policy per root) — exploratory |
| `regimes/`, `noisecomp/` | controlled studies on `fake_fez`: the error scale s with observable and layout blocks (21 tasks); one noise component changed at a time at matched exposure (18 tasks); `analysis/` holds the cells, per-candidate risks and features — exploratory |
| `analysis/` | every evaluation table of the development data (splits, pooled results, optimism, ablations, learning curve, budget generalization, nested selective evaluation, interval coverage, certificates, design variance and diagnostics, failure profile, runtime and early rejection) |
| `hardware_plan/` | the hardware validation plan of the shipped bundle as a dry run — **planned, not executed**; nothing was submitted |
| `figures/` | the six main figures (Figure 6 pending: the mismatch study was not repeated, `figures/PENDING.md`) and the supplementary figures, each with its notes (`.md`: question, methods box, caption, interpretation, inputs, command) and plotted data (`__*.csv`) |
| `thesis_tables/` | the thesis tables as CSV, LaTeX fragments, Markdown and a standalone PDF from one generator, with `provenance.json` (input digests) |
| `tables/`, `RESULTS_TABLES.md` | the results tables of the development analysis and the confirmation |
| `TRIALS_SHA256SUMS` | SHA-256 of every per-trial array (`*.npz`) of this run |

The two shipped bundles are in `qemx/selection/bundles/`: `qemx-rec-1.8.0` (the frozen configuration, simulator-only
features) and `qemx-rec-1.8.0-deployable` (the default: deployable features, no exact simulation at recommendation
time).

## Per-trial arrays

The per-trial arrays (`<part>/tasks/<task_id>.npz` and `audit18/root_*/tasks/<task_id>.npz`: estimates, flags and
learning-design indices of both label halves; 217 files, about 24 MB) are not in the source archive. They are
published as two trial-data archives, `qem-explorer-1.8.0-trials-1.zip` (development) and
`qem-explorer-1.8.0-trials-2.zip` (confirmation, regimes, noise composition, root audit), whose paths are relative to
the repository root. Extract both there and verify:

```bash
cd reports/selection_v180 && sha256sum -c TRIALS_SHA256SUMS
```

Without them, everything except the per-trial drill-down in the app, the two figures drawn from individual trials
(`supp_bias_variance`, `supp_representative`: reported as pending, the shipped versions stay) and the re-analysis
from trials works from the tables and records in this directory.

## Regenerate

Figures and tables from the saved tables (no simulation, a few minutes):

```bash
python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml
python -m qemx.selection report  configs/selection/dev18.yaml --out reports/selection_v180
python -m qemx.selection.thesis_tables --run reports/selection_v180 --out reports/selection_v180/thesis_tables
python scripts/validate_figures.py reports/selection_v180
```

Re-analysis from the trials (needs the trial-data archives; no simulation): copy this directory and run steps 4–11 of
the 1.8.0 sequence in `docs/selection/COMMANDS.md` on the copy. `confirm` is evaluated once per run directory
(`EVALUATED`) and refuses to run again there. Relabeling from scratch: `scripts/reproduce_v18.sh` (measured on the
build machine: about 3.5 CPU-hours of labeling for the 217 tasks — development 1.3, confirmation 1.2, regimes 0.3,
noise composition 0.3, root audit 0.4 — plus the analyses, about two hours on two cores).
