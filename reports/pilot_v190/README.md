# pilot_v190 — the 1.9.0 pilot study (exploratory; for planning)

**What it is.** The executed bounded pilot of the next research stage: 27 new logical base circuits (disjoint from every
earlier study; `configs/selection/ancestry_registry.json`), 54 tasks, five families (brickwork, HEA, Ising, QAOA:
12 tasks each; GHZ: 6), widths 3–5, budgets 10³, 3·10⁴, 10⁶, 3·10⁷ shots, 17 candidates, R = 24 trials per label half,
3 fresh training designs per half (seed scheme 2), and **noise-diverse training**: six calibration-supplied noise
settings on three base devices (native 18 tasks, two-qubit errors ×2 10, overall error ×2 9, relaxation ×2 8, overall
error ×0.5 5, readout ×2 4). Everything is simulated (emulated fake-device trials); nothing ran on hardware.

**Status.** Exploratory. It measures costs and checks every stage of the pipeline on real data for the planned
development study (`configs/selection/fulldev19.yaml`); its numbers are a pilot's, not results, and it is never
confirmatory evidence. Interpretation: `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §I.2 B–D.

**How it was produced** (1.9.0, two Linux cores):

```
python -m qemx.selection pipeline configs/pipelines/pilot19.yaml --out runs/pilot19 --workers 2
```

The labels were made with the 1.9.0 development code at commit `052f8e4`, before the version string was set to
1.9.0, so `dev/provenance.json` records `app_version: 1.8.0` with that commit; the later stages ran on the 1.9.0 code
at commit `abd39d3` (after the first independent-verification fixes). Between the two commits the scientific code
changed only in the recommender's input validation (`qemx/selection/deploy.py`), not in labeling, features, models or
evaluation.

Stages: plan → label (0.86 CPU-hours, 27 minutes with 2 workers) → analyze (71 minutes) → freeze → nested evaluation
(20 minutes) → freeze completion → bundles → portfolio comparison → recommend check → profile → figures → report →
status. Recorded command lines in the figure notes and the portfolio comparison are given with repository-relative
paths (`--out reports/pilot_v190` regenerates the figures from the saved tables).

**Contents.**

| path | what |
|---|---|
| `plan/expanded_tasks.csv` | every task as expanded before labeling (identities, split, family, width, input, observable, layout, device, noise modifiers) |
| `dev/` | per-task records (`tasks/*.json`), the run identity (`run.json`), provenance, attempts; the per-trial arrays are **not** here |
| `analysis/` | out-of-fold evaluations (`pooled_*.csv`: same-distribution, noise, device-base, joint), choices, nested selective evaluation, optimism (BBC-CV), learning curve, design variance, runtime profile, labels table |
| `freeze.json` | the configuration chosen on the same-distribution split (`MODEL[hgb|deployable18+simulated]`), bound to the analysis, the development data digest and the scientific configuration digest; τ = 0.124 |
| `bundles/` | `pilot19-sim` (the frozen configuration) and `pilot19-deployable` — exploratory bundles for checking the pipeline, **not** for recommendations |
| `portfolio/` | 8 vs 17 candidates on identical labels |
| `figures/`, `tables/`, `RESULTS_TABLES.md` | the standard figures and tables of the run (with their notes) |
| `completion.json` | `python -m qemx.selection status` of the run: 54/54 complete, data digest |
| `TRIALS_SHA256SUMS` | SHA-256 of the 54 per-trial arrays (`dev/tasks/*.npz`) |

**Per-trial arrays.** In the release attachment `qem-explorer-1.9.0-trials-pilot.zip` (`reports/TRIAL_ATTACHMENTS.json`).
Extract it into the repository root, then verify:

```
python scripts/verify_trials.py reports/pilot_v190
```
