# Results tables — pilot19

Generated from `pilot19/analysis/` by `python -m qemx.selection report`. Simulated observations (emulated fake-device trials); no hardware data. Risks are evaluation-half MSEs; log values are log10(MSE + 1e-6); intervals are 95 % bootstrap intervals over test base circuits.

## Data

54 tasks from 27 base circuits; 4 budgets; 17 protocols; 3672 (task, protocol, budget) labels, of which 3105 feasible.

### Median log10(MSE + 10⁻⁶) by protocol and budget (evaluation half, feasible tasks)

| protocol | 1e+03 | 3e+04 | 1e+06 | 3e+07 |
|---|---|---|---|---|
| CDR |  | -3.71 | -5.007 | -5.517 |
| FR |  | -1.507 | -1.493 | -1.455 |
| NIL |  | -2.313 | -2.523 | -2.429 |
| PEC | -2.845 | -4.212 | -5.305 | -5.443 |
| PEC-TW | -2.878 | -4.248 | -5.388 | -5.585 |
| RAW | -2.897 | -3.712 | -3.795 | -3.775 |
| RO | -2.869 | -4.005 | -4.155 | -4.151 |
| RO-DD | -2.843 | -3.971 | -4.235 | -4.234 |
| RO-TREX | -2.921 | -3.973 | -4.138 | -4.129 |
| RO-TW | -2.868 | -3.983 | -4.234 | -4.245 |
| SV | -3.242 | -4.564 | -5.002 | -5.078 |
| ZNE-E | -2.286 | -3.792 | -5.143 | -5.763 |
| ZNE-E-B | -2.297 | -3.792 | -5.143 | -5.763 |
| ZNE-L | -2.393 | -3.791 | -5.165 | -5.769 |
| ZNE-L-TW | -2.451 | -3.831 | -5.191 | -5.758 |
| ZNE-L∘SV | -2.608 | -4.118 | -5.485 | -5.933 |
| ZNE-R | -1.909 | -3.401 | -4.796 | -5.756 |


### How often each protocol is the measured best (tasks per budget)

| protocol | 1e+03 | 3e+04 | 1e+06 | 3e+07 |
|---|---|---|---|---|
| CDR | 0 | 2 | 4 | 11 |
| PEC | 7 | 13 | 13 | 7 |
| PEC-TW | 6 | 10 | 15 | 6 |
| RAW | 12 | 7 | 1 | 1 |
| RO | 5 | 3 | 0 | 0 |
| RO-DD | 1 | 3 | 2 | 2 |
| RO-TREX | 8 | 5 | 3 | 2 |
| RO-TW | 10 | 2 | 3 | 2 |
| SV | 2 | 5 | 1 | 1 |
| ZNE-E | 0 | 2 | 7 | 11 |
| ZNE-L | 3 | 2 | 4 | 4 |
| ZNE-L-TW | 0 | 0 | 0 | 4 |
| ZNE-L∘SV | 0 | 0 | 1 | 0 |
| ZNE-R | 0 | 0 | 0 | 3 |


### Share of tasks on which each protocol is feasible

| protocol | 1e+03 | 3e+04 | 1e+06 | 3e+07 |
|---|---|---|---|---|
| CDR | 0 | 1 | 1 | 1 |
| FR | 0 | 1 | 1 | 1 |
| NIL | 0 | 0.8704 | 0.8704 | 0.8704 |
| PEC | 1 | 1 | 1 | 1 |
| PEC-TW | 1 | 1 | 1 | 1 |
| RAW | 1 | 1 | 1 | 1 |
| RO | 1 | 1 | 1 | 1 |
| RO-DD | 1 | 1 | 1 | 1 |
| RO-TREX | 1 | 1 | 1 | 1 |
| RO-TW | 1 | 1 | 1 | 1 |
| SV | 0.1111 | 0.1111 | 0.1111 | 0.1111 |
| ZNE-E | 1 | 1 | 1 | 1 |
| ZNE-E-B | 1 | 1 | 1 | 1 |
| ZNE-L | 1 | 1 | 1 | 1 |
| ZNE-L-TW | 1 | 1 | 1 | 1 |
| ZNE-L∘SV | 0.1111 | 0.1111 | 0.1111 | 0.1111 |
| ZNE-R | 1 | 1 | 1 | 1 |


## Policies on held-out base circuits (group 5-fold over base circuits, all budgets; out-of-fold pooled)

| policy | mean_log10_mse | diff_vs_SBS | diff_vs_SBS_B | gap_closed | policy_MSE | P_regret_gt_2x | coverage | mean_log10_regret | median_regret | p90_regret | share_negative_regret | within_10pct | worse_than_RAW | fallback_trials |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL-CANDIDATE | -4.583 [-4.668, -4.457] | -0.251 [-0.347, -0.166] | -0.232 [-0.311, -0.165] | 1.00 [1.00, 1.00] | 0.0003775 | 0 | 1 | 0.000 [0.000, 0.000] | 0 | 0 | 0 | 1 | 0.088 | 0.0033 |
| MODEL+ABSTAIN[hgb\|deployable18] | -4.351 [-4.494, -4.165] | -0.019 [-0.072, 0.027] | 0.000 [0.000, 0.000] | 0.08 [-0.14, 0.25] | 0.0006252 | 0.2824 | 0.3148 | 0.232 [0.165, 0.311] | 5.6e-07 | 0.00036 | 0.185 | 0.44 | 0.162 | 0 |
| SBS-B | -4.351 [-4.494, -4.165] | -0.019 [-0.072, 0.027] | 0.000 [0.000, 0.000] | 0.08 [-0.14, 0.25] | 0.0006252 | 0.2824 | 1 | 0.232 [0.165, 0.311] | 5.6e-07 | 0.00036 | 0.185 | 0.44 | 0.162 | 0 |
| MODEL[hgb\|deployable18+simulated] | -4.351 [-4.487, -4.192] | -0.019 [-0.088, 0.043] | 0.000 [-0.054, 0.056] | 0.08 [-0.20, 0.31] | 0.0005237 | 0.2593 | 1 | 0.232 [0.158, 0.313] | 9.9e-07 | 0.00028 | 0.204 | 0.444 | 0.167 | 0.0017 |
| SBS | -4.332 [-4.475, -4.144] | 0.000 [0.000, 0.000] | 0.019 [-0.027, 0.072] | 0.00 [0.00, 0.00] | 0.0005284 | 0.2639 | 1 | 0.251 [0.166, 0.347] | 9.9e-07 | 0.00039 | 0.199 | 0.458 | 0.167 | 0 |
| MODEL[enet\|deployable18+simulated] | -4.322 [-4.452, -4.147] | 0.011 [-0.068, 0.084] | 0.030 [-0.027, 0.083] | -0.04 [-0.44, 0.22] | 0.0007192 | 0.2917 | 1 | 0.262 [0.182, 0.334] | 2.1e-06 | 0.00027 | 0.176 | 0.435 | 0.19 | 0.0042 |
| MODEL[hgb\|size_budget] | -4.317 [-4.444, -4.159] | 0.015 [-0.055, 0.081] | 0.034 [-0.032, 0.097] | -0.06 [-0.40, 0.18] | 0.0006513 | 0.3148 | 1 | 0.266 [0.190, 0.352] | 1.8e-06 | 0.00042 | 0.176 | 0.398 | 0.185 | 0.0008 |
| SIZE-B | -4.314 [-4.455, -4.134] | 0.018 [-0.028, 0.061] | 0.037 [0.000, 0.080] | -0.07 [-0.33, 0.10] | 0.0005437 | 0.3241 | 1 | 0.269 [0.194, 0.356] | 1.1e-06 | 0.00042 | 0.208 | 0.44 | 0.176 | 0.0039 |
| MODEL[ridge\|deployable18+simulated] | -4.304 [-4.448, -4.104] | 0.029 [-0.062, 0.114] | 0.048 [-0.011, 0.105] | -0.11 [-0.60, 0.21] | 0.0007826 | 0.3333 | 1 | 0.280 [0.206, 0.371] | 1.8e-06 | 0.00031 | 0.176 | 0.431 | 0.204 | 0.0075 |
| MODEL[enet\|size_budget] | -4.294 [-4.423, -4.130] | 0.038 [-0.032, 0.115] | 0.057 [-0.003, 0.119] | -0.15 [-0.58, 0.10] | 0.0005694 | 0.3472 | 1 | 0.289 [0.216, 0.372] | 2e-06 | 0.00036 | 0.185 | 0.417 | 0.208 | 0.0042 |
| MODEL[ridge\|size_budget] | -4.283 [-4.406, -4.120] | 0.050 [-0.020, 0.121] | 0.069 [0.007, 0.130] | -0.20 [-0.64, 0.06] | 0.0005811 | 0.3704 | 1 | 0.301 [0.236, 0.388] | 2.9e-06 | 0.00037 | 0.162 | 0.384 | 0.227 | 0.0037 |
| MODEL[hgb\|deployable18] | -4.266 [-4.413, -4.078] | 0.066 [-0.022, 0.161] | 0.086 [0.017, 0.162] | -0.26 [-0.80, 0.08] | 0.000701 | 0.338 | 1 | 0.318 [0.230, 0.425] | 2e-06 | 0.00068 | 0.148 | 0.384 | 0.194 | 0.0039 |
| MODEL[ridge\|deployable18] | -4.255 [-4.399, -4.067] | 0.077 [-0.010, 0.167] | 0.096 [0.027, 0.164] | -0.31 [-0.85, 0.04] | 0.0008102 | 0.3796 | 1 | 0.328 [0.249, 0.416] | 3.8e-06 | 0.00042 | 0.153 | 0.37 | 0.222 | 0.01 |
| MODEL[enet\|deployable18] | -4.238 [-4.375, -4.066] | 0.095 [0.028, 0.159] | 0.114 [0.056, 0.173] | -0.38 [-0.81, -0.10] | 0.0007702 | 0.3704 | 1 | 0.346 [0.260, 0.441] | 4.9e-06 | 0.00032 | 0.134 | 0.384 | 0.213 | 0.0041 |
| MODEL[struct\|size_budget] | -4.214 [-4.360, -4.016] | 0.119 [0.045, 0.210] | 0.138 [0.075, 0.220] | -0.47 [-1.01, -0.15] | 0.0008383 | 0.3981 | 1 | 0.370 [0.280, 0.470] | 6.2e-06 | 0.00043 | 0.134 | 0.338 | 0.213 | 0.0156 |
| MODEL[struct\|deployable18] | -4.207 [-4.345, -4.042] | 0.125 [0.055, 0.189] | 0.145 [0.092, 0.203] | -0.50 [-1.04, -0.17] | 0.0008412 | 0.4398 | 1 | 0.377 [0.307, 0.458] | 5.5e-06 | 0.00044 | 0.125 | 0.356 | 0.241 | 0.0104 |
| MODEL[struct\|deployable18+simulated] | -4.164 [-4.342, -3.933] | 0.168 [0.074, 0.270] | 0.187 [0.099, 0.279] | -0.67 [-1.20, -0.27] | 0.002145 | 0.4352 | 1 | 0.419 [0.291, 0.561] | 5.3e-06 | 0.001 | 0.157 | 0.37 | 0.194 | 0.0093 |
| RAW | -3.481 [-3.728, -3.231] | 0.851 [0.676, 1.019] | 0.870 [0.692, 1.037] | -3.39 [-5.10, -2.40] | 0.001844 | 0.6759 | 1 | 1.102 [0.897, 1.303] | 0.00012 | 0.0025 | 0.116 | 0.218 | 0 | 0 |


### Paired difference to SBS by budget (Δ mean log10(MSE + 10⁻⁶); < 0 is better). Best model: `MODEL[hgb|deployable18+simulated]` (selected on these scores)

| B | best_model | SBS_B | SIZE_B | oracle | best_model_vs_SBS_B |
|---|---|---|---|---|---|
| 1e+03 | 0.003 [-0.041, 0.045] | 0.008 [-0.048, 0.062] | -0.021 [-0.072, 0.031] | -0.048 [-0.111, 0.013] | -0.005 [-0.069, 0.051] |
| 3e+04 | 0.033 [-0.044, 0.115] | 0.000 [0.000, 0.000] | -0.009 [-0.021, 0.001] | -0.125 [-0.175, -0.072] | 0.033 [-0.044, 0.115] |
| 1e+06 | 0.015 [-0.103, 0.123] | 0.000 [0.000, 0.000] | 0.031 [-0.008, 0.084] | -0.314 [-0.455, -0.183] | 0.015 [-0.103, 0.123] |
| 3e+07 | -0.129 [-0.283, -0.004] | -0.085 [-0.296, 0.105] | 0.070 [-0.105, 0.233] | -0.518 [-0.711, -0.335] | -0.044 [-0.234, 0.152] |


## Optimism of the selected configuration (BBC-CV, Tsamardinos et al. 2018)

The best model configuration is chosen on the same pooled scores it is reported with. BBC-CV selects on bootstrap samples of base circuits and scores the selection on the out-of-bag circuits; `optimism` is the difference to the naive winner's score (log10 units, > 0: the naive score is too good). Interval: 2.5–97.5 % of the out-of-bag values.

| split | reference | candidates | naive_best | naive_diff | bias_corrected_diff | P_diff_below_0 | optimism |
|---|---|---|---|---|---|---|---|
| device-base | SBS | 4 | MODEL[hgb\|deployable18] | -0.015 | -0.005 [-0.101, 0.107] | 0.542 | 0.014 |
| device-base | SBS-B | 4 | MODEL[hgb\|deployable18] | -0.003 | 0.005 [-0.069, 0.109] | 0.464 | 0.014 |
| joint | SBS | 4 | MODEL[ridge\|size_budget] | -0.071 | -0.059 [-0.159, 0.100] | 0.642 | -0 |
| joint | SBS-B | 4 | MODEL[ridge\|size_budget] | -0.096 | -0.092 [-0.143, 0.015] | 0.952 | -0 |
| noise | SBS | 4 | MODEL[hgb\|size_budget] | 0.037 | 0.065 [-0.030, 0.198] | 0.11 | 0.032 |
| noise | SBS-B | 4 | MODEL[hgb\|size_budget] | 0.047 | 0.073 [0.000, 0.212] | 0.026 | 0.032 |
| same-distribution | SBS | 12 | MODEL[hgb\|deployable18+simulated] | -0.019 | 0.007 [-0.096, 0.127] | 0.474 | 0.029 |
| same-distribution | SBS-B | 12 | MODEL[hgb\|deployable18+simulated] | 0 | 0.024 [-0.060, 0.117] | 0.306 | 0.029 |


## Generalization (all budgets; test rows of all parts of a split pooled)

| split | policy | parts | base_circuits | mean_log10_mse | diff_vs_SBS | diff_vs_SBS_B | gap_closed | mean_log10_regret |
|---|---|---|---|---|---|---|---|---|
| same-distribution | ALL-CANDIDATE | 5 | 27 | -4.583 [-4.668, -4.457] | -0.251 [-0.347, -0.166] | -0.232 [-0.311, -0.165] | 1.00 [1.00, 1.00] | 0.000 [0.000, 0.000] |
| same-distribution | MODEL[enet\|deployable18+simulated] | 5 | 27 | -4.322 [-4.452, -4.147] | 0.011 [-0.068, 0.084] | 0.030 [-0.027, 0.083] | -0.04 [-0.44, 0.22] | 0.262 [0.182, 0.334] |
| same-distribution | MODEL[enet\|deployable18] | 5 | 27 | -4.238 [-4.375, -4.066] | 0.095 [0.028, 0.159] | 0.114 [0.056, 0.173] | -0.38 [-0.81, -0.10] | 0.346 [0.260, 0.441] |
| same-distribution | MODEL[enet\|size_budget] | 5 | 27 | -4.294 [-4.423, -4.130] | 0.038 [-0.032, 0.115] | 0.057 [-0.003, 0.119] | -0.15 [-0.58, 0.10] | 0.289 [0.216, 0.372] |
| same-distribution | MODEL[hgb\|deployable18+simulated] | 5 | 27 | -4.351 [-4.487, -4.192] | -0.019 [-0.088, 0.043] | 0.000 [-0.054, 0.056] | 0.08 [-0.20, 0.31] | 0.232 [0.158, 0.313] |
| same-distribution | MODEL[hgb\|deployable18] | 5 | 27 | -4.266 [-4.413, -4.078] | 0.066 [-0.022, 0.161] | 0.086 [0.017, 0.162] | -0.26 [-0.80, 0.08] | 0.318 [0.230, 0.425] |
| same-distribution | MODEL[hgb\|size_budget] | 5 | 27 | -4.317 [-4.444, -4.159] | 0.015 [-0.055, 0.081] | 0.034 [-0.032, 0.097] | -0.06 [-0.40, 0.18] | 0.266 [0.190, 0.352] |
| same-distribution | MODEL[ridge\|deployable18+simulated] | 5 | 27 | -4.304 [-4.448, -4.104] | 0.029 [-0.062, 0.114] | 0.048 [-0.011, 0.105] | -0.11 [-0.60, 0.21] | 0.280 [0.206, 0.371] |
| same-distribution | MODEL[ridge\|deployable18] | 5 | 27 | -4.255 [-4.399, -4.067] | 0.077 [-0.010, 0.167] | 0.096 [0.027, 0.164] | -0.31 [-0.85, 0.04] | 0.328 [0.249, 0.416] |
| same-distribution | MODEL[ridge\|size_budget] | 5 | 27 | -4.283 [-4.406, -4.120] | 0.050 [-0.020, 0.121] | 0.069 [0.007, 0.130] | -0.20 [-0.64, 0.06] | 0.301 [0.236, 0.388] |
| same-distribution | MODEL[struct\|deployable18+simulated] | 5 | 27 | -4.164 [-4.342, -3.933] | 0.168 [0.074, 0.270] | 0.187 [0.099, 0.279] | -0.67 [-1.20, -0.27] | 0.419 [0.291, 0.561] |
| same-distribution | MODEL[struct\|deployable18] | 5 | 27 | -4.207 [-4.345, -4.042] | 0.125 [0.055, 0.189] | 0.145 [0.092, 0.203] | -0.50 [-1.04, -0.17] | 0.377 [0.307, 0.458] |
| same-distribution | MODEL[struct\|size_budget] | 5 | 27 | -4.214 [-4.360, -4.016] | 0.119 [0.045, 0.210] | 0.138 [0.075, 0.220] | -0.47 [-1.01, -0.15] | 0.370 [0.280, 0.470] |
| same-distribution | SBS-B | 5 | 27 | -4.351 [-4.494, -4.165] | -0.019 [-0.072, 0.027] | 0.000 [0.000, 0.000] | 0.08 [-0.14, 0.25] | 0.232 [0.165, 0.311] |
| same-distribution | SIZE-B | 5 | 27 | -4.314 [-4.455, -4.134] | 0.018 [-0.028, 0.061] | 0.037 [0.000, 0.080] | -0.07 [-0.33, 0.10] | 0.269 [0.194, 0.356] |


## Paired benefit of the frozen configuration over SBS-B by family (all budgets; < 0 favors the frozen configuration)

| family | base_circuits | diff_log10_mse | diff_mse |
|---|---|---|---|
| brickwork | 6 | 0.010 [-0.071, 0.092] | 9.5e-05 [3.3e-05, 0.00016] |
| ghz | 3 | 0.122 [-0.141, 0.329] | 6.7e-05 [1.1e-05, 0.00015] |
| hea | 6 | 0.042 [0.004, 0.081] | -1.5e-05 [-3.9e-05, 1.4e-05] |
| ising | 6 | -0.083 [-0.174, -0.014] | -5.4e-05 [-0.00021, 8.7e-05] |
| qaoa | 6 | -0.030 [-0.182, 0.139] | -0.00052 [-0.0012, -6.4e-05] |


## Selective recommendation of the frozen configuration (margin threshold, fallback SBS-B)

| policy | mean_log10_mse | mean_log10_regret | mean_mse |
|---|---|---|---|
| model only (coverage 1) | -4.351 | 0.2319 | 0.00052 |
| selective, cross-fitted threshold (coverage 0.597) | -4.335 | 0.2479 | 0.00063 |
| SBS-B only (coverage 0) | -4.351 | 0.2319 | 0.00063 |


Thresholds per test fold (chosen on the other folds): fold 1 0.0135, fold 2 1.1, fold 3 0.0666, fold 4 0.0666, fold 5 0.0666; accepted-row mean log10 regret at the operating point 0.235; deployment threshold on all development rows τ = 0.06656.

## Calibration of the frozen configuration's log-risk intervals (held out; inner-CV residuals)

| kind | nominal | coverage | fold_min | fold_max | mean_width |
|---|---|---|---|---|---|
| per candidate | 0.5 | 0.544 | 0.384 | 0.639 | 0.645 |
| per candidate | 0.8 | 0.834 | 0.691 | 0.922 | 1.467 |
| per candidate | 0.9 | 0.934 | 0.833 | 0.975 | 2.329 |
| per candidate | 0.95 | 0.99 | 0.954 | 1 | 4.266 |
| selected candidate | 0.5 | 0.613 | 0.375 | 0.8 | 0.527 |
| selected candidate | 0.8 | 0.855 | 0.729 | 0.95 | 1.254 |
| selected candidate | 0.9 | 0.939 | 0.875 | 1 | 2.104 |
| selected candidate | 0.95 | 0.988 | 0.938 | 1 | 3.862 |
| simultaneous | 0.5 | 1 | 1 | 1 |  |
| simultaneous | 0.8 | 1 | 1 | 1 |  |
| simultaneous | 0.9 | 1 | 1 | 1 |  |
| simultaneous | 0.95 | 1 | 1 | 1 |  |


## Learning curve (mean log10 regret on held-out folds)

| train_base_circuits | model | q10 | q90 | SBS | draws |
|---|---|---|---|---|---|
| 5 | 0.396 | 0.358 | 0.439 | 0.326 | 4 |
| 6 | 0.332 | 0.256 | 0.432 | 0.212 | 6 |
| 10 | 0.358 | 0.259 | 0.478 | 0.305 | 4 |
| 11 | 0.28 | 0.138 | 0.449 | 0.208 | 6 |
| 16 | 0.333 | 0.164 | 0.504 | 0.247 | 10 |
| 21 | 0.396 | 0.351 | 0.441 | 0.305 | 2 |
| 22 | 0.242 | 0.154 | 0.348 | 0.208 | 3 |


## Classical cost of one recommendation (median of repeats, seconds; label_s: one development task's label generation, for comparison)

| n | compile_s | deployable_s | simulated_s | inference_s | label_s | peak_rss_mb |
|---|---|---|---|---|---|---|
| 3 | 0.0114 | 0.017 | 0.2914 | 0.0395 | 38.59 | 308.9 |
| 4 | 0.0118 | 0.0226 | 0.5291 | 0.0448 | 50.61 | 315.3 |
| 5 | 0.0154 | 0.042 | 0.7086 | 0.042 | 66.63 | 315.3 |
| 6 | 0.0131 | 0.0356 | 1.045 | 0.0471 |  | 319.6 |
| 12 |  |  |  |  |  | 321.9 |
| 24 |  |  |  |  |  | 338.6 |
| 40 |  |  |  |  |  | 356 |


## Learning-design variance of the labels (evaluation half)

| protocol | B | tasks | designs | median_se_ratio | p90_se_ratio | median_between_share | median_design_share_of_label_variance | share_tasks_between_dominant |
|---|---|---|---|---|---|---|---|---|
| CDR | 30000 | 54 | 3 | 0.951 | 1.634 | 0 | 0 | 0.019 |
| CDR | 1000000 | 54 | 3 | 1.099 | 2.671 | 0.028 | 0.189 | 0.167 |
| CDR | 30000000 | 54 | 3 | 2.526 | 3.369 | 0.602 | 0.924 | 0.556 |
| FR | 30000 | 54 | 3 | 1.198 | 1.94 | 0.058 | 0.332 | 0.056 |
| FR | 1000000 | 54 | 3 | 1.303 | 2.28 | 0.093 | 0.45 | 0.093 |
| FR | 30000000 | 54 | 3 | 1.387 | 2.445 | 0.122 | 0.526 | 0.148 |
| NIL | 30000 | 47 | 3 | 1.764 | 2.451 | 0.266 | 0.743 | 0.191 |
| NIL | 1000000 | 47 | 3 | 1.682 | 2.354 | 0.233 | 0.708 | 0.128 |
| NIL | 30000000 | 47 | 3 | 1.821 | 2.651 | 0.289 | 0.765 | 0.255 |


## Failure profile (all budgets; same-distribution held-out rows)

| policy | B | rows | base_circuits | mse | worst1_share | worst5_share | severe_rate | max_log_regret | worse_than_raw_2x |
|---|---|---|---|---|---|---|---|---|---|
| RAW | all | 216 | 27 | 0.0018 | 0.0962 | 0.4499 | 0.4583 | 4.155 | 0 |
| SBS | all | 216 | 27 | 0.0005 | 0.1401 | 0.4378 | 0.0787 | 2.491 | 0.0556 |
| SBS-B | all | 216 | 27 | 0.0006 | 0.2196 | 0.549 | 0.0556 | 2.357 | 0.0556 |
| SIZE-B | all | 216 | 27 | 0.0005 | 0.1502 | 0.4802 | 0.0833 | 2.479 | 0.0463 |
| MODEL[ridge\|size_budget] | all | 216 | 27 | 0.0006 | 0.2455 | 0.4868 | 0.0648 | 2.409 | 0.0972 |
| MODEL[ridge\|deployable18] | all | 216 | 27 | 0.0008 | 0.2019 | 0.5892 | 0.0787 | 2.409 | 0.0972 |
| MODEL[ridge\|deployable18+simulated] | all | 216 | 27 | 0.0008 | 0.209 | 0.5786 | 0.0648 | 2.209 | 0.0787 |
| MODEL[enet\|size_budget] | all | 216 | 27 | 0.0006 | 0.2506 | 0.4982 | 0.0694 | 2.409 | 0.1019 |
| MODEL[enet\|deployable18] | all | 216 | 27 | 0.0008 | 0.2288 | 0.5791 | 0.1019 | 2.409 | 0.1019 |
| MODEL[enet\|deployable18+simulated] | all | 216 | 27 | 0.0007 | 0.2274 | 0.593 | 0.0741 | 2.123 | 0.0694 |
| MODEL[hgb\|size_budget] | all | 216 | 27 | 0.0007 | 0.2033 | 0.5379 | 0.0648 | 1.998 | 0.0556 |
| MODEL[hgb\|deployable18] | all | 216 | 27 | 0.0007 | 0.2058 | 0.4942 | 0.0972 | 2.902 | 0.0602 |
| MODEL[hgb\|deployable18+simulated] | all | 216 | 27 | 0.0005 | 0.1494 | 0.4451 | 0.0694 | 1.857 | 0.0556 |
| MODEL[struct\|size_budget] | all | 216 | 27 | 0.0008 | 0.1828 | 0.5741 | 0.1065 | 2.54 | 0.0833 |
| MODEL[struct\|deployable18] | all | 216 | 27 | 0.0008 | 0.225 | 0.5969 | 0.0833 | 2.507 | 0.1157 |
| MODEL[struct\|deployable18+simulated] | all | 216 | 27 | 0.0021 | 0.4127 | 0.745 | 0.125 | 3.894 | 0.1019 |
| MODEL+ABSTAIN[hgb\|deployable18] | all | 216 | 27 | 0.0006 | 0.2196 | 0.549 | 0.0556 | 2.357 | 0.0556 |
| ALL-CANDIDATE | all | 216 | 27 | 0.0004 | 0.1002 | 0.3576 | 0 | 0 | 0.0278 |


## Selective recommendation, nested evaluation (`MODEL[hgb|deployable18+simulated]`)

| policy | coverage | mse | mean_log10_mse | diff_log_vs_SBS_B | diff_log_vs_random |
|---|---|---|---|---|---|
| policy | 0.296 | 0.000604 [0.000355, 0.00102] | -4.324 [-4.468, -4.157] | 0.027 [-0.011, 0.068] | -0.004 [-0.026, 0.010] |
| model | 1 | 0.000524 [0.000344, 0.000809] | -4.351 [-4.490, -4.196] | 0.000 [-0.051, 0.054] | -0.031 [-0.067, 0.002] |
| fallback (SBS-B) | 0 | 0.000625 [0.000348, 0.00107] | -4.351 [-4.497, -4.180] | 0.000 [0.000, 0.000] | -0.031 [-0.069, -0.003] |
| SIZE-B |  | 0.000544 [0.000333, 0.000861] | -4.314 [-4.468, -4.158] | 0.037 [-0.002, 0.078] | 0.006 [-0.050, 0.054] |
| SBS |  | 0.000528 [0.000347, 0.000836] | -4.332 [-4.480, -4.163] | 0.019 [-0.027, 0.069] | -0.012 [-0.073, 0.039] |
| random rejection | 0.296 | 0.000608 [0.000353, 0.00103] | -4.320 [-4.458, -4.153] | 0.031 [0.003, 0.069] | 0.000 [0.000, 0.000] |


Outer folds' τ: 0.000148, 0.303, 0.419, 0.447, 0.169; deployment τ = 0.1239.

