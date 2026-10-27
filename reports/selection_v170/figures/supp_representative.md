# supp_representative

**Question.** What do successful, typical, failed and abstained recommendations look like trial by trial?

**Caption.** Representative outcomes of the deployed recommender at B = 10⁵ on held-out base circuits, chosen by documented rules: the rows with the lowest, the median and the highest log10 regret ratio, and among the rows where the nested evaluation (outer-fold threshold chosen on inner folds) abstains (preferring rows where abstaining changes the deployed protocol) the one with the median margin z. Protocols without a training design: boxes of per-trial errors (evaluation half; quartiles, whiskers 1.5 IQR, dots beyond them — the rare large errors that dominate an MSE). CDR and NIL: every trial as a point, one column per fresh training design (alternating grey), the design's mean error as a coloured tick — separated clusters are designs, not shot noise. Filled triangle: the recommender's choice (its RMSE in the title); open triangle: the fallback used on abstention. Illustrative cases selected by rule, not a random sample. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Spread is variance, offset from zero is bias, and for the learning protocols the spread between design means is the design component of the variance; all are part of each protocol's risk.

**Data.** `supp_representative__trials.csv`, `supp_representative__rows.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | four rows chosen by documented rules |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 120 development tasks on 40 independent base circuits; 10 fresh designs per half for CDR and NIL; r = 100 trials per half |
| Uncertainty and its scope | none (individual trials) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | illustrative |
| Infeasible, failed, unresolved, missing | — |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/deployed_rows.csv`: task_id, B, log_regret, margin, choice, fallback
- `analysis/selective_rows.csv`: task_id, B, accepted_crossfit
- `dev/tasks/<task_id>.json, .npz`: spec, budgets, protocols, ideal, est (trial estimates)

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
