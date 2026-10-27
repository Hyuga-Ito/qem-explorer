# supp_coverage_certificates

**Question.** Do the label intervals and finite-sample certificates hold empirically?

**Caption.** Interval coverage and finite-sample bounds; each panel says what kind of statement it supports. (a) Measured coverage of log10-risk intervals per held-out fold (points; bar: mean over 5 folds, about 8 base circuits each), against the nominal level (dashed): group split-conformal intervals (the model refitted on part of the training circuits, calibrated on the rest; per protocol and Bonferroni-simultaneous; a ≥ 1 − α guarantee per circuit under exchangeability of base circuits — measured here, not assumed), cross-validated residual quantiles (marginal, simultaneous and joint over the feasible protocols of a row; calibrated in cross-validation, no finite-sample guarantee) and the residual-Gaussian interval (no guarantee). Every interval targets the evaluated label log10(R̂ + 10⁻⁶), which contains label noise. Median full widths in log10: split-conformal 8.47, cv-residual 2.55, residual-gaussian 2.43. (b) Median radius t of the Hoeffding and empirical-Bernstein bounds (α = 0.05 over the K feasible protocols) on the clipped squared error (estimates projected onto [−1, 1], H = 4) — a finite-sample bound for the clipped estimator, not for the unclipped MSE that the study reports; units: independent trials, or for CDR and NIL under fresh designs their design means (median units of the chosen protocol: B=1000: 100, B=3000: 100, B=10000: 100, B=30000: 100, B=100000: 100, B=300000: 100, B=1000000: 100, B=3000000: 100, B=10000000: 100, B=30000000: 100); share of informative bounds (upper bound of the chosen protocol below RAW's lower bound): 0.000. (c) Realized regret against 2ε with ε the largest selection-half error: a post-hoc check computed with the evaluation labels, not a certificate; it holds on 1.000 of rows (regrets below 10⁻⁹ drawn at 10⁻⁹). Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Bounded-loss certificates are orders of magnitude wider than the risks they would certify at these trial counts; conformal intervals are calibrated marginally over base circuits, not per task; the 2ε check is descriptive.

**Data.** `supp_coverage_certificates__coverage.csv`, `supp_coverage_certificates__certificates.csv`, `supp_coverage_certificates__two_eps.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | held-out folds and budgets |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 120 development tasks on 40 independent base circuits; 10 fresh designs per half for CDR and NIL; r = 100 trials per half |
| Uncertainty and its scope | per-fold coverage; bounds at α = 0.05 |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | split-conformal / cv-residual / finite-sample-bound (clipped) / post-hoc-check, each labelled |
| Infeasible, failed, unresolved, missing | vacuous bounds reported as such |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/interval_coverage.csv`: see DATA_DICTIONARY
- `analysis/certificates.csv`: radius, holds

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
