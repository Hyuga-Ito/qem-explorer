# Main-study figure report — direction-A-v2-pilot

Simulation study on the device model named in the manifest (no hardware data in figures 1–4 and 6; figure 5 is an EMULATION rehearsal). 75 cells from 25 base circuits (18 explore / 7 confirm base circuits, split by hash before any outcome). Resource: the total shot budget B of one estimate (training, calibration and every extra circuit included), allocated by the equal-budget ledger (policy *fixed*). Outcome: RMSE against the exact ideal value. Intervals: 95 % bootstrap over base circuits. Practical equivalence δ = 10 % in RMSE.

† Upper bounds, never candidates: VD (flag `optimistic`: simulated with ideal resources, e.g. virtual distillation with ideal copies, an ideal derangement and ideal readout). They appear in Figures 3–4 for reference and are excluded from the regime maps and the recommender.

Approximation flags present in the records: `calibration-mean-first-order`, `degenerate`, `gaussian-sampler`, `nonlinear`, `optimistic`.

## Figure 1 — Feature coverage

![](fig1_coverage.png)

(a) Compiled two-qubit count against the light-cone fault rate λ_LC per cell (○ explore, □ confirmatory). (b) Base circuits per family and split. Regions without points are not covered: no figure below extrapolates into them.

## Figure 2 — Empirical performance-regime map

![](fig2_regime_map.png)

Each point is one cell at one budget of the grid (no interpolation), colored by the best method; gray: a tie within δ; white: raw within δ of the best (no demonstrated benefit); black: no method can run at that budget. (a) exploration cells, (b) confirmatory cells. Regime shares:

| regime | (cell, B) points | share |
|---|---|---|
| PEC | 1218 | 0.396 |
| raw ≈ best | 722 | 0.235 |
| ZNE-L | 560 | 0.182 |
| tie | 293 | 0.095 |
| ZNE-R | 180 | 0.059 |
| NIL-G2 | 59 | 0.019 |
| CDR | 43 | 0.014 |

Points are (cell, budget) pairs on a grid: 3075 points from 25 base circuits. They are not independent observations — inference in this report resamples whole base circuits.

**Monte-Carlo stability of the labels.** The RMSEs of the learning methods (and of target-nonlinear estimators) come from Monte-Carlo moments with their own errors. Redrawing every method's RMSE from its Monte-Carlo error (independent draws), 0 of 2060 single-method labels have P(label is the winner) < 0.90 (`regimes.csv`: `p_winner`, `p_in_tie`). Such a label is *MC-unresolved*: more Monte-Carlo precision, not a different method, could change it. This is not the same as a practical tie (difference below δ).

## Figure 3 — Paired change against raw

![](fig3_paired_improvement.png)

(a) Median over the cells where the method runs (conditional on feasibility) of log10(RMSE_method / RMSE_raw) at B = 100,000 with 95 % base-circuit bootstrap intervals; the gray band is ±δ. (b) Bars: share of *all planned* cells in which the method is better than raw by more than δ — a case that cannot run at this budget counts as not better; ticks: the same share among the cells where it runs. The labels give better/planned and the number of cells where it cannot run.

| method | planned | feasible (fraction) | median log10 ratio, feasible cells [95 % CI] | median absolute RMSE gain, feasible cells [95 % CI] | better / planned | better / feasible |
|---|---|---|---|---|---|---|
| ZNE-R | 75 | 75 (1) | -0.068 [-0.235, 0.0795] | 0.00149 [-0.00179, 0.00591] | 38/75 = 0.51 | 38/75 = 0.51 |
| ZNE-L | 75 | 75 (1) | -0.296 [-0.45, -0.151] | 0.00484 [0.00227, 0.00948] | 55/75 = 0.73 | 55/75 = 0.73 |
| PEC | 75 | 75 (1) | -0.412 [-0.539, -0.252] | 0.00541 [0.00299, 0.00933] | 67/75 = 0.89 | 67/75 = 0.89 |
| CDR | 75 | 75 (1) | -0.161 [-0.324, 0] | 0.00266 [0, 0.00659] | 45/75 = 0.6 | 45/75 = 0.6 |
| FR-R | 75 | 75 (1) | 0.443 [0, 0.942] | -0.0252 [-0.0726, 0] | 21/75 = 0.28 | 21/75 = 0.28 |
| NIL-G2 | 75 | 9 (0.12) | 0 [0, 0] | 0 [0, 0] | 0/75 = 0 | 0/9 = 0 |
| VD | 75 | 75 (1) | -0.591 [-0.732, -0.358] | 0.00726 [0.0046, 0.0102] | 74/75 = 0.99 | 74/75 = 0.99 |

## Figure 4 — Cost to reach a stated precision

![](fig4_cost_to_precision.png)

Smallest total budget at which RMSE ≤ 0.02, per cell (points) and its median over reachable cells (bar). 'Unreachable' cells have a bias floor above 0.02 at any budget: more shots cannot fix them.

## Figure 5 — Discrepancy diagnostics and method rankings (EMULATION, not hardware)

![](fig5_discrepancy_rehearsal.png)

End-to-end rehearsal of the hardware path on emulated observations of the divergence suite: (a) the discrepancy statistic z when the truth is the predictor's own model (null) and under controlled unmodeled effects; (b) per circuit, Kendall τ between the ordering of the four methods by predicted and by observed classical fidelity F_c (Chen et al. Eq. 1, unnormalized). Under the null, Holm-adjusted flags: 0 of 1120 rows; unadjusted |z| > 3: 2.

## Figure 6 — Baseline recommender on held-out data

![](fig6_recommender.png)

Ridge models of log10 RMSE per method from recommendation-time features (n, two_qubit_logical, non_clifford, two_qubit, depth, obs_weight, lam_lc, lam_ro, scale, coherent_fraction, log10_budget), fitted inside each training fold. Folds group every row of a base circuit; leave-family-out and leave-width-out test transfer; the confirmatory subset was fixed before any outcome (it has since been inspected during development, so it is a development benchmark, not an untouched holdout). (a) Median regret (RMSE_chosen / RMSE_best − 1) of the recommender on the rows where it decided, and of the baselines on all rows. (b) Selective risk: regret as rows are admitted from the most to the least confident.

Denominators: *rows* are (cell, budget) pairs, *decided* the rows with a recommendation; intervals resample base circuits. The **full policy** falls back on abstention to the training fold's best fixed method (raw if it cannot run), so it is defined on every row. *Harmful*: the choice is worse than raw by more than δ. *Paired log10 RMSE ratio*: the mean over the same rows of log10(RMSE_choice / RMSE_best fixed method); negative means the recommender's choice is better.

| evaluation | feature set | rows (base circuits) | decided | coverage | accuracy (in δ set) | median regret, decided [p90] | harmful, decided | full policy: median regret [p90] | full policy harmful | paired log10 RMSE ratio vs fixed, decided [95 % CI] | paired log10 RMSE ratio, full policy vs fixed [95 % CI] | interval coverage (nominal 90 %, n) | fixed: median regret, harmful | rule regret | raw regret |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grouped cv | circuit_observable | 270 (18) | 165 | 0.61 | 0.33 | 0.788 [8.39] | 0 | 1.42 [9.38] | 0.026 | 0.266 [0.167, 0.374] | 0.162 [0.0921, 0.24] | 0.99 (165) | 0.138, 0.052 | 1.09 | 2.45 |
| leave family out | circuit_observable | 270 (18) | 72 | 0.27 | 0.29 | 0.736 [5.51] | 0.042 | 0.0885 [4.08] | 0.048 | 0.0827 [0.0295, 0.131] | 0.0221 [0.00341, 0.0455] | 0.61 (72) | 0.0339, 0.041 | 1.09 | 2.45 |
| leave width out | circuit_observable | 90 (6) | 63 | 0.7 | 0.27 | 0.845 [9.81] | 0 | 0.831 [8.1] | 0 | 0.221 [0.095, 0.335] | 0.155 [0.0648, 0.256] | 1 (63) | 0.165, 0.044 | 1.52 | 2.89 |
| confirmatory | circuit_observable | 105 (7) | 66 | 0.63 | 0.58 | 0.0413 [2.5] | 0 | 0.201 [2.94] | 0.048 | 0.107 [0.0305, 0.203] | 0.0671 [0.0183, 0.135] | 0.97 (66) | 0.0375, 0.086 | 1.17 | 0.524 |
| grouped cv | structural | 270 (18) | 161 | 0.6 | 0.33 | 0.788 [7.85] | 0 | 1.13 [8.71] | 0.026 | 0.248 [0.132, 0.365] | 0.148 [0.0706, 0.23] | 0.99 (161) | 0.138, 0.052 | 1.09 | 2.45 |
| leave family out | structural | 270 (18) | 78 | 0.29 | 0.37 | 0.449 [7.29] | 0.026 | 0.107 [4.71] | 0.044 | 0.145 [0.0255, 0.26] | 0.0419 [0.00472, 0.0955] | 0.72 (78) | 0.0339, 0.041 | 1.09 | 2.45 |
| leave width out | structural | 90 (6) | 65 | 0.72 | 0.26 | 0.871 [8.92] | 0 | 0.901 [8.1] | 0 | 0.24 [0.125, 0.352] | 0.174 [0.0748, 0.265] | 1 (65) | 0.165, 0.044 | 1.52 | 2.89 |
| confirmatory | structural | 105 (7) | 53 | 0.5 | 0.72 | 0 [0.728] | 0 | 0.0516 [1.8] | 0.048 | 0.0467 [-0.00169, 0.0981] | 0.0236 [-0.000903, 0.05] | 0.87 (53) | 0.0375, 0.086 | 1.17 | 0.524 |
| grouped cv | normalized | 270 (18) | 160 | 0.59 | 0.34 | 0.687 [7.51] | 0 | 1.2 [8.92] | 0.03 | 0.247 [0.159, 0.336] | 0.146 [0.0897, 0.207] | 0.99 (160) | 0.138, 0.052 | 1.09 | 2.45 |
| leave family out | normalized | 270 (18) | 110 | 0.41 | 0.32 | 0.618 [4.95] | 0.045 | 0.49 [5.48] | 0.048 | 0.237 [0.152, 0.325] | 0.0965 [0.0474, 0.149] | 0.98 (110) | 0.0339, 0.041 | 1.09 | 2.45 |
| leave width out | normalized | 90 (6) | 51 | 0.57 | 0.31 | 0.788 [7.17] | 0.02 | 0.835 [6.61] | 0.011 | 0.224 [0.089, 0.386] | 0.127 [0.0514, 0.203] | 1 (51) | 0.165, 0.044 | 1.52 | 2.89 |
| confirmatory | normalized | 105 (7) | 65 | 0.62 | 0.6 | 0.014 [1.8] | 0 | 0.21 [2.44] | 0.038 | 0.0961 [0.0224, 0.168] | 0.0595 [0.0137, 0.101] | 1 (65) | 0.0375, 0.086 | 1.17 | 0.524 |
| grouped cv | with_state | 270 (18) | 157 | 0.58 | 0.36 | 0.5 [6.73] | 0.0064 | 1.01 [8.1] | 0.033 | 0.222 [0.101, 0.332] | 0.129 [0.0526, 0.208] | 0.97 (157) | 0.138, 0.052 | 1.09 | 2.45 |
| leave family out | with_state | 270 (18) | 107 | 0.4 | 0.44 | 0.288 [5.83] | 0 | 0.266 [5.01] | 0.026 | 0.168 [0.0554, 0.268] | 0.0666 [0.0177, 0.128] | 0.88 (107) | 0.0339, 0.041 | 1.09 | 2.45 |
| leave width out | with_state | 90 (6) | 54 | 0.6 | 0.33 | 0.422 [6.23] | 0.056 | 0.617 [7.08] | 0.033 | 0.197 [0.0528, 0.317] | 0.118 [0.0272, 0.217] | 1 (54) | 0.165, 0.044 | 1.52 | 2.89 |
| confirmatory | with_state | 105 (7) | 57 | 0.54 | 0.7 | 0 [0.694] | 0 | 0.0478 [1.79] | 0.038 | 0.0355 [0.00293, 0.0787] | 0.0193 [0.00167, 0.041] | 0.84 (57) | 0.0375, 0.086 | 1.17 | 0.524 |

Feature sets are acquisition tiers: *circuit_observable* uses only the circuit, the observable and the budget; *structural* adds the calibrated noise model (λ_LC, λ_ro, scale, coherent fraction); *with_state* adds exact ideal-state entropies (exponential in n). A tier earns its place only if it lowers held-out regret beyond the uncertainty shown. Intervals: group-CV conformal (12 base circuits); group-CV conformal (13 base circuits); group-CV conformal (14 base circuits); group-CV conformal (18 base circuits). Feature weights are associations within the sampled design, not causes. A recommendation outside the training coverage is withheld ('abstain-coverage').

**Monte-Carlo sensitivity of this evaluation** (grouped CV, structural features, 40 redraws of the ledger's RMSEs from their Monte-Carlo errors; median relative SE 0.039, largest 0.053): regret median 0.775–0.788; accuracy 0.327–0.329; coverage 0.596–0.6; policy regret median 1.13–1.2; harmful 0–0; interval coverage 0.988–0.988.

## Limits of this report

* Simulation on one device model; the hardware transfer test is a separate, budgeted campaign (docs/hardware_preflight.md). No hardware result is shown here.
* A pilot: few base circuits per family. Intervals are wide by design; read them, not the point estimates.
* The base-circuit bootstrap intervals do not contain the Monte-Carlo error of the estimated RMSEs; its effect on the labels (`p_winner`) and on the recommender evaluation (redraw ranges) is reported separately above.
