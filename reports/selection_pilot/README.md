# Bundled pilot of the protocol-selection study (QEM Explorer 1.6.0)

> **1.7.0 note.** This is the historical 1.6.0 dataset: its CDR and NIL labels reuse six learning designs in both
> label halves (the fixed-shared estimand, review R1). The corrected pilot is `reports/selection_v170/`. To keep the
> 1.7.0 source archive small, its copy of this folder omits the per-trial arrays (`*.npz`) and `figures/`; both are
> unchanged in the 1.6.0 release archive. Nothing else in this folder was changed.

A complete run directory of `configs/selection/pilot.yaml`, `configs/selection/mismatch.yaml` and
`configs/selection/regimes.yaml`: every label with its per-trial data, the provenance of every labeling session, the
analysis tables, the figures and the tables. **Simulated observations** (emulated trials on IBM fake devices); no
hardware data. Interpretation: `docs/selection/RESULTS.md`; figures: `docs/selection/FIGURES.md`; column
definitions: `docs/selection/DATA_DICTIONARY.md`.

| path | content |
|---|---|
| `manifest.json` | the configuration and the task lists (development; final test, chosen by keyed hash before labeling) |
| `dev/` | 120 development tasks: `tasks/<task_id>.json` (record) and `.npz` (per-trial estimates and flags of both halves), `provenance.json` (two sessions: the labeling and the resume after decision D16), `label_summary.json` |
| `freeze.json` | the development choice frozen before the final test was labeled |
| `final/` | 36 final-test tasks, labeled and evaluated once (`EVALUATED`, `summary_final.csv`, `choices_final.csv`); `metrics_final.csv` and `labels.csv` recomputed from them without refitting |
| `mismatch/` | 90 tasks (10 circuits × 9 execution laws), `scenarios.json`, `diagnostics.csv`, `policies.csv`, `null_check.csv`, `model_training.json` |
| `regimes/` | the controlled regime study (decision D23): 21 tasks on `fake_fez` with the device error scale s, an observable block and a layout block; `regimes/analysis/` holds the regime cells, the per-protocol risks, the features, the labels and the model's training record |
| `analysis/` | every evaluation table (`summary_*`, `choices_*`, `pooled_*`, `optimism.csv`, `family_benefit.csv`, the frozen configuration's rows with selective recommendation and calibration, ablation, learning curve, budget generalization, certificates, runtime) |
| `figures/` | six main and eight supplementary figures (PDF, SVG, PNG), each with its notes (`.md`: question, caption, interpretation, inputs, command, plotting configuration) and plotted data (`__*.csv`) |
| `tables/`, `RESULTS_TABLES.md` | the three tables, the rejection table and the results tables quoted in `docs/selection/RESULTS.md` |

The development labels were analyzed twice: once for the freeze and the final test (the frozen choice and the
final-test choices in this directory come from that analysis), and again, after the final test, to the re-sent
specification (decision D25); the tables here are the second analysis. Nothing frozen was changed.

Regenerate figures and tables from the saved tables (no simulation, a few minutes):

```
python -m qemx.selection figures configs/selection/pilot.yaml --out reports/selection_pilot
python -m qemx.selection report  configs/selection/pilot.yaml --out reports/selection_pilot
```

Recompute the analyses from the per-trial labels (no simulation; about an hour on two cores):

```
python -m qemx.selection analyze  configs/selection/pilot.yaml    --out <copy of this directory>
python -m qemx.selection mismatch configs/selection/mismatch.yaml --out <copy> --dev <copy>/dev --skip-label
python -m qemx.selection regimes  configs/selection/regimes.yaml  --out <copy> --skip-label
python -m qemx.selection profile  configs/selection/pilot.yaml    --out <copy>
```

The final test is evaluated once per run directory (`final/EVALUATED`); `final` refuses to run again here.
