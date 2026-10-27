# supp_portfolio_expansion

**Question.** Do more candidates help, and is the benefit from the candidates or from learning to select them?

**Caption.** Original (portfolio-1.6, the eight candidates of 1.6/1.7) against expanded (portfolio-1.8, 17 candidates) portfolio on identical development tasks and labels (216 rows, 27 base circuits; out-of-fold, 5 folds grouped by base circuit; feature set deployable18). Every policy is refitted for each portfolio. (A) Paired contrasts of the mean over base circuits of log10(MSE + 10⁻⁶) with 95 % bootstrap intervals over base circuits (mean over base circuits of log10(MSE + 1e-6), evaluation half; paired bootstrap 1000): grey — what more candidates give a fixed policy and an oracle-like reference (the candidate best on the independent selection half, scored on the evaluation half); blue — what learned selection adds within each portfolio, and the total. (B) The policies' means with each portfolio. (C) Share of rows on which each 1.8 policy chose a candidate (%). (D) Declared tail: rows whose chosen risk exceeds ten times RAW's on the same row.

**Interpretation.** Observation (exploratory, development data): more candidates change SBS-B by -0.018 [-0.042, +0.009] and the attainable reference by -0.030 [-0.052, -0.007]; learned selection (HGB) changes the 1.8 portfolio's SBS-B by +0.086 [+0.014, +0.158]. A larger portfolio helps a policy only if the policy can identify where the new candidates win.

**Data.** `supp_portfolio_expansion__contrasts.csv`, `supp_portfolio_expansion__policies.csv`, `supp_portfolio_expansion__chosen.csv`, `supp_portfolio_expansion__tails.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the portfolio (8 or 17 candidates) and the policy; held fixed: tasks, labels, folds, feature set |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 54 development tasks on 27 independent base circuits; r = 24 trials per half; 3 fresh designs per half |
| Uncertainty and its scope | 95 % paired bootstrap over base circuits |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory (development, out-of-fold) |
| Infeasible, failed, unresolved, missing | infeasible candidates are not candidates of a row; fallbacks stay in the risk |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `portfolio/portfolio_contrasts.csv`: contrast, diff, lo, hi
- `portfolio/portfolio_policies.csv`: policy, mean_log10, mse, p_severe_vs_raw, q95_risk, worst5_loss_share
- `portfolio/portfolio_comparison.json`: policies.<policy>.chosen, rows, base_circuits, feature_set

**Command.** `python -m qemx.selection figures configs/selection/pilot19.yaml --out reports/pilot_v190 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
