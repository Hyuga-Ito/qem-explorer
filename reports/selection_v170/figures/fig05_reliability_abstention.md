# fig05_reliability_abstention

**Question.** Can the system recognize unreliable recommendations (nested evaluation)?

**Caption.** Reliability of the recommender MODEL[hgb|deployable+simulated], evaluated by nested cross-validation over base circuits (outer group 5-fold; in each outer training set an inner group 4-fold chooses the threshold τ; the frozen rule is applied once to the outer test fold; every base circuit is tested once; 40 base circuits, 1200 rows). The rule: use the model's choice when the margin z = (P₂ − P₁)/√(s₁² + s₂²) between its two lowest predicted log10 risks is at least τ, otherwise SBS-B (no shots). Outer folds' τ: 0.02, 0.00, inf, 0.18, 0.03. (A) Mean log10 regret of the accepted rows against coverage (descriptive sweep over τ on the outer rows), random rejection, each fold's nested operating point and the pooled one. (B) Whole-policy arithmetic MSE over all rows with the fallback's loss included (base circuits weighted equally); random rejection at the operating point's coverage per fold (expected value; bar: 2.5–97.5 % of random acceptance sets). (C) Paired difference to SBS-B in mean log10 MSE with 95 % base-circuit bootstrap intervals; arithmetic MSE annotated. (D) Measured coverage of log10-risk intervals on held-out circuits: inner-CV residual intervals (statement kind cv-residual: no finite-sample guarantee; bars: range over folds) per candidate, for the selected candidate and Bonferroni-simultaneous; group split-conformal intervals (calibration circuits held out of the fit; finite-sample coverage ≥ 1 − α for all rows of a new circuit drawn from the same distribution — an assumption that fails for new families, devices or budget ranges). Measured coverage is a property of these circuits, not a guarantee for others. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** At the nested operating point (coverage 0.65) the whole-policy mean MSE is 0.000688 [0.000359, 0.00111], against 0.00071 for the model alone, 0.000697 for SBS-B alone and 0.000723 for random rejection at the same coverage; Δ mean log10 MSE vs SBS-B -0.003 [-0.043, +0.040], vs random rejection -0.000 [-0.008, +0.008]. Abstention is useful only if the selective policy beats both the model alone and random rejection at matched coverage on the whole-policy scale; a lower accepted-subset regret (A) alone does not establish reliability.

**Data.** `fig05_reliability_abstention__curve.csv`, `fig05_reliability_abstention__summary.csv`, `fig05_reliability_abstention__folds.csv`, `fig05_reliability_abstention__calibration.csv`, `fig05_reliability_abstention__intervals.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the abstention threshold (coverage); held fixed: the frozen model and its fallback SBS-B |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 120 development tasks on 40 independent base circuits; 10 fresh designs per half for CDR and NIL; r = 100 trials per half |
| Uncertainty and its scope | nested group cross-validation (outer 5 folds, threshold chosen on inner folds); 95 % bootstrap over base circuits; random rejection at matched coverage over repeated draws |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory; measured coverage is not a guarantee; interval panels labelled by statement kind |
| Infeasible, failed, unresolved, missing | abstained rows use the fallback and stay in the whole-policy MSE |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/nested_curve.csv`: coverage, accepted_log_regret, policy_mse, random_policy_mse
- `analysis/nested_summary.csv`: policy, coverage, mse, mse_lo, mse_hi, diff_log_vs_sbsb(_lo/_hi), diff_log_vs_random(_lo/_hi), random_draws_mse_lo/hi, accepted_log_regret
- `analysis/nested_rows.csv`: fold, accepted, log_regret_model
- `analysis/nested_folds.csv`: fold, tau
- `analysis/calibration.csv`: kind, nominal, coverage
- `analysis/interval_coverage.csv`: kind, nominal, coverage, finite, statement

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
