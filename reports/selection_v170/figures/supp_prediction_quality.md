# supp_prediction_quality

**Question.** How accurate are the held-out risk predictions per candidate, given the labels' own uncertainty?

**Caption.** Prediction quality of the deployed recommender on held-out base circuits (group 5-fold; 120 tasks). Left: predicted against evaluated log10(MSE + 10⁻⁶) per candidate, with ±1.96 standard errors of the label (delta method on the SE matching the estimand — over training designs for CDR and NIL under fresh designs; capped at 3 decades for display). Points beyond the axis (evaluated log10 MSE > 1: divergent exponential-ZNE labels) are counted in the titles and kept in the source data; the RMSE includes them. Right: per row, the predicted log-risk difference between the predicted best and runner-up protocols against the evaluated difference of the same two protocols (evaluation half; ±1.96 SE, the two labels treated as independent): a selection is right when the evaluated difference is positive. Grey: the evaluated difference is within 1.96 SE of zero (the labels cannot order the pair). Pointwise intervals; descriptive. The 10⁻⁶ floor compresses differences between very small risks. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Good choices need correct orderings within a task, not accurate values; errors shared across candidates cancel. The predicted best is evaluated better than the predicted runner-up in 62 % of rows (65 % of the 756 rows whose evaluated difference is resolved); correlation of predicted and evaluated differences 0.23.

**Data.** `supp_prediction_quality__points.csv`, `supp_prediction_quality__stats.csv`, `supp_prediction_quality__runner_up.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | held-out rows |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 120 development tasks on 40 independent base circuits; 10 fresh designs per half for CDR and NIL; r = 100 trials per half |
| Uncertainty and its scope | ±1.96 label SE (delta method); best–runner-up differences with independent-label SE |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | descriptive (held-out predictions) |
| Infeasible, failed, unresolved, missing | labels beyond the axis counted in titles |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/prediction.csv`: protocol, observed, predicted, observed_se

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
