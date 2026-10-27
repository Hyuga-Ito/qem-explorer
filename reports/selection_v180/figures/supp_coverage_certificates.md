# supp_coverage_certificates

**Question.** Do the label intervals and finite-sample certificates hold empirically?

**Caption.** Interval coverage and finite-sample bounds; each panel says what kind of statement it supports. (a) Measured coverage of log10-risk intervals per held-out fold (points; bar: mean over 5 folds, about 9 base circuits each), against the nominal level (dashed): group split-conformal intervals (the model refitted on part of the training circuits, calibrated on the rest; per protocol and Bonferroni-simultaneous; a ≥ 1 − α guarantee per circuit under exchangeability of base circuits — measured here, not assumed), cross-validated residual quantiles (marginal, simultaneous and joint over the feasible protocols of a row; calibrated in cross-validation, no finite-sample guarantee) and the residual-Gaussian interval (no guarantee). Every interval targets the evaluated label log10(R̂ + 10⁻⁶), which contains label noise. Median full widths in log10: split-conformal 5.41, residual-gaussian 2.41. (b) Median radius t of the Hoeffding and empirical-Bernstein bounds (α = 0.05 over the K feasible protocols) on the clipped squared error (estimates projected onto [−1, 1], H = 4) — a finite-sample bound for the clipped estimator, not for the unclipped MSE that the study reports; units: independent trials, or for the learning candidates (CDR, NIL; FR in 1.8.0) under fresh designs their design means (median units of the chosen protocol: B=1000: 60, B=3000: 60, B=10000: 60, B=30000: 60, B=100000: 60, B=300000: 60, B=1000000: 60, B=3000000: 60, B=10000000: 60, B=30000000: 60); share of informative bounds (upper bound of the chosen protocol below RAW's lower bound): 0.000. (c) Realized regret against 2ε with ε the largest selection-half error: a post-hoc check computed with the evaluation labels, not a certificate; it holds on 1.000 of rows (regrets below 10⁻⁹ drawn at 10⁻⁹). Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Bounded-loss certificates are orders of magnitude wider than the risks they would certify at these trial counts; conformal intervals are calibrated marginally over base circuits, not per task; the 2ε check is descriptive.

**Data.** `supp_coverage_certificates__coverage.csv`, `supp_coverage_certificates__certificates.csv`, `supp_coverage_certificates__two_eps.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | held-out folds and budgets |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 88 development tasks on 44 independent base circuits; 6 fresh designs per half for CDR and NIL; r = 60 trials per half |
| Uncertainty and its scope | per-fold coverage; bounds at α = 0.05 |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | split-conformal / cv-residual / finite-sample-bound (clipped) / post-hoc-check, each labelled |
| Infeasible, failed, unresolved, missing | vacuous bounds reported as such |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/interval_coverage.csv`: see DATA_DICTIONARY
- `analysis/certificates.csv`: radius, holds

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
