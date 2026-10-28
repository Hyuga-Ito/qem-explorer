# supp_design_variability

**Question.** How much do CDR's and NIL's errors depend on the training design drawn, and does another training allocation change that?

**Caption.** Learning-design variability of the learning candidates (CDR, NIL, FR). (a) Development labels, evaluation half: for every task, budget and fresh training design (6 designs per half), the design's mean loss relative to the median design of the same task and budget (log10 of the ratio, both + 10⁻⁶); violins over all designs, black: median, markers: the three largest; dotted line: a design ten times worse than the task's median design. (b) Median over tasks of the between-design share of the variance of the evaluation-half MSE, Var_D(E[L|D])/K ÷ (that + E Var(L|D)/(Km)), from a one-way analysis of variance. (c) Ratio of the design-level (cluster-robust) to the trial-level standard error of the label, median and 90 % quantile over tasks: a trial-level SE understates the fresh-design uncertainty by this factor (1 = no understatement). (d) The focused design study (experiment B, scripts/design_study.py): on the review's severe-NIL tasks and controls, design-level MSEs (ticks; open circle: mean over designs) of CDR, NIL and alternative training allocations (more training circuits, another share of B, the bare circuit added) at the largest budget; dotted: MSE 10⁻² (RMSE 0.1, the study's severe threshold). Exploratory; designs are the independent unit within a task, tasks are few. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Share of designs at least ten times worse than their task's median design: CDR B=10⁴ 0 %, CDR B=10⁶ 0 %, CDR B=3·10⁷ 0 %, NIL B=10⁶ 7 %, NIL B=3·10⁷ 7 %, FR B=10⁴ 6 %, FR B=10⁶ 7 %, FR B=3·10⁷ 7 %. Median between-design share of the label variance at the largest budget: CDR 0.90, NIL 0.82, FR 0.47.

**Data.** `supp_design_variability__design_losses.csv`, `supp_design_variability__variance.csv`, `supp_design_variability__design_study.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the training design (fresh draws) and, in the design study, the training allocation |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 88 development tasks on 44 independent base circuits; 6 fresh designs per half for CDR and NIL; r = 60 trials per half; design study: n/a designs per variant |
| Uncertainty and its scope | designs as the independent unit within a task |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory |
| Infeasible, failed, unresolved, missing | infeasible variants listed in design_study/designs.csv |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/design_diagnostics.csv`: task_id, protocol, B, half, design, mean_loss
- `analysis/design_variance.csv`: protocol, B, designs, median_se_ratio, median_design_share_of_label_variance
- `design_study/designs.csv`: task, variant, B, status, design, mean_loss

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
