# supp_target_precision

**Question.** What budget does each protocol need to reach an MSE target, by the point estimate and with uncertainty?

**Caption.** Budget needed to meet an MSE target (120 development tasks, evaluation half): per task and protocol the smallest evaluated budget whose MSE point estimate (filled) or approximate pointwise upper limit MSE + 1.96 SE (open; SE matching the estimand, over training designs for CDR and NIL under fresh designs) is at most the target; the median over tasks is drawn with the share of tasks that attain it. The upper limit is pointwise at each budget: the first grid budget that passes it is not a guaranteed minimum budget (the grid is searched, and the intervals are not simultaneous). Infeasible protocol–budget pairs count as not attained at that budget. Unattained targets are censored at the grid's largest budget (3·10⁷): 'not reached' means the median task did not attain it; nothing is extrapolated. Descriptive (development data). Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Uncertainty-qualified attainment needs more budget than point attainment; the gap is the price of the (pointwise, approximate) confidence about the target.

**Data.** `supp_target_precision__budgets.csv`, `supp_target_precision__summary.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the MSE target and budget |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 120 development tasks on 40 independent base circuits; 10 fresh designs per half for CDR and NIL; r = 100 trials per half |
| Uncertainty and its scope | pointwise approximate upper limit MSE + 1.96 SE (not a minimum-budget guarantee after a grid search) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | descriptive (development data) |
| Infeasible, failed, unresolved, missing | unattained targets censored at 3·10⁷ |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/labels.csv`: task_id, protocol, B, feasible, mse_eval, se_mse_eval

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
