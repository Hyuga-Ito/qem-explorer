# Final holdout for the main study (frozen, not yet run)

**Status (2026-10-18): frozen, unrun.** `manifests/direction_A_v2_final_holdout.yaml`, SHA-256
`60fc000c95025825f139d7b21b68d8676b4a0989fec96a2bbdb36da6b6be08c3` (with LF line endings; `tests/test_final_holdout.py`
fails if it changes).

## Why it exists

The pilot's confirmatory subset was fixed before any outcome existed, but its results were then read while the
analysis, the features and the recommender were being developed. It is therefore a development benchmark, not an
untouched test. The final holdout is a set of base circuits that no development step has seen.

## What it contains

23 base circuits, 69 cells (three noise scales each) on the `fake_fez` device model, engine E2, readout arm on:

| family | base circuits | how they differ from development |
|---|---|---|
| QFT | n = 4, 5; input s = 3, 6 | development uses s = 1 |
| GHZ (X basis) | n = 4, 5, 6; θ = π/8, π/4 | development uses π/6, π/3, 5π/12 |
| Ising Trotter | n = 4, 5, 6; 2 steps; h = 0.75 | development uses h = 0.5, 1, 2 |
| Brickwork | n = 4, 5, 6; 3 layers; seeds 10, 11 | development uses seeds 0–2 and other depths |
| Bernstein–Vazirani | n = 4: s = 6, 12; n = 5: s = 13, 25 | development uses the default secrets |

`python -m qemx.phase.run holdout-check manifests/direction_A_v2_final_holdout.yaml --against <development
manifests>` checks that no base circuit (family, width, declared parameters) is shared. The protocols, Monte-Carlo
settings, budget grid and device model equal the pilot's, so a model trained on development data describes these
cells; `python -m qemx.study holdout` refuses the evaluation if the resolved protocols or the noise context differ.

## Rules

1. **Freeze first.** Before the holdout is run, write down in `reports/final_holdout/FROZEN.md`: the development
   run(s) used for training, the feature set, the ridge penalty, the abstention thresholds (`z`, coverage), the
   practical-equivalence margin δ, the fallback of the full policy, and the source digest
   (`python -c "from qemx.provenance import source_digest; print(source_digest())"`).
2. **Run once.** `python -m qemx.phase.run run manifests/direction_A_v2_final_holdout.yaml --out data/final_holdout`,
   then `collect`, then `python -m qemx.study holdout <development run> data/final_holdout --out
   reports/final_holdout --feature-set <frozen set>`.
3. **Report everything.** Every endpoint below, favorable or not, with its denominators. A change of the model
   after the evaluation turns the holdout into development data; say so if it happens.

## Endpoints

* **Primary:** the paired mean of log10(RMSE of the full policy's choice / RMSE of the best fixed method chosen on the
  development data) over the holdout rows (cell × budget), with a 95 % bootstrap interval over base circuits.
  Negative means the policy is better. The policy is considered better only if the whole interval is below 0.
* **Secondary:** coverage (share of rows with a recommendation); median and 90th-percentile regret of the decided
  rows; the probability of a harmful choice (worse than raw by more than δ) for the selective model, the full policy
  and the fixed method; the coverage of the nominal 90 % intervals; the same paired comparison per family.
* **Weights:** every (cell, budget) row counts once; uncertainty resamples whole base circuits. More budget points
  are not more independent circuits.

## What the pilot already suggests

On the development data the baseline recommender does **not** beat the best fixed method (`reports/pilot/report.md`,
Figure 6: the paired log10 RMSE ratio of its choices against the best fixed method is positive under grouped
cross-validation for every feature tier). Adding calibrated-noise or exact-state features did not change that. A
holdout evaluation of this model would most likely confirm a negative result; it becomes informative once a
candidate model beats the fixed baseline on development data.
