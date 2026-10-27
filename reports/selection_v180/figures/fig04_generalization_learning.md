# fig04_generalization_learning

**Question.** Does the model generalize beyond its training distribution, and how much data does it need?

**Caption.** Generalization and data needs. (A) Mean log10 regret ratio against the per-task reference (protocol identified on the selection half, scored on the evaluation half — the same reference definition in every cell), all budgets, on held-out base circuits: same distribution (group 5-fold), leave one family out, held-out device (fake_kyiv, crossed with the folds), larger width (train n ≤ 5, test n = 6); columns: the budget-dependent fixed policy as reference, the size-and-budget tree, Ridge, the structured risk model (b² + v/((1−f)B) per protocol with ridge-regularized task effects) and boosted trees, all on deployable features; 95 % base-circuit bootstrap intervals and the number n of independent test base circuits in each cell. (B) Learning curve: the model with a fixed setting (deployable features) and SBS trained on random subsets of each training fold's base circuits, scored on the fixed held-out fold; mean over folds and repeats, band 10–90 % of fold × repeat draws (not a confidence interval). (C) SBS-B, Ridge, the structured model and boosted trees trained only on budgets up to a maximum (dotted line; one facet per maximum) and scored at every budget of held-out circuits (group 5-fold, 44 base circuits); filled: in range, open on grey: extrapolation; at an unseen budget SBS-B uses the nearest trained budget. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Observation: paired difference of the nonlinear column to SBS-B in mean log10 MSE (< 0 favors the model): same-distribution -0.011 [-0.094, +0.068]; family +0.038 [-0.013, +0.091]; device -0.027 [-0.172, +0.120]; width -0.065 [-0.178, +0.041]; learning curve: mean log10 regret 0.393 at 9 and 0.289 at 35 training base circuits (mean over folds and repeats; SBS 0.421 and 0.459). Mean log10 regret over the extrapolated budgets: train B ≤ 300000: SBS-B (budget-only) 0.463, Ridge | deployable18 0.384, structured b²+v/B | deployable18 0.603, boosted trees | deployable18 0.375; train B ≤ 10000: SBS-B (budget-only) 0.481, Ridge | deployable18 0.404, structured b²+v/B | deployable18 1.059, boosted trees | deployable18 0.585. Matrix rows are different claims (interpolation in the same distribution, extrapolation in family, device and width); panel C separates budget interpolation from extrapolation.

**Data.** `fig04_generalization_learning__matrix.csv`, `fig04_generalization_learning__learning_curve.csv`, `fig04_generalization_learning__budget.csv`

**Configuration.** `{"budget_policies": ["SBS-B", "MODEL[ridge|deployable18]", "MODEL[struct|deployable18]", "MODEL[hgb|deployable18]"], "columns": {"linear": "MODEL[ridge|deployable18]", "nonlinear": "MODEL[hgb|deployable18]", "reference": "SBS-B", "simple": "SIZE-B", "structured": "MODEL[struct|deployable18]"}, "rows": ["same-distribution", "family", "device", "width"]}`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the split (family, device, width, budget range held out) and the training-set size; held fixed: features and model settings per column |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 88 development tasks on 44 independent base circuits; 6 fresh designs per half for CDR and NIL; r = 60 trials per half |
| Uncertainty and its scope | 95 % bootstrap over held-out base circuits; learning-curve band 10–90 % of fold × repeat draws (not a confidence interval) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory (development data; held-out splits) |
| Infeasible, failed, unresolved, missing | cells without a held-out circuit are blank; extrapolated budgets marked |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/pooled_<split>.csv`: policy, B (= all), mean_log_regret, log_regret_lo, log_regret_hi, base_circuits, rows
- `analysis/learning_curve.csv`: train_base_circuits, mean_log_regret, sbs_mean_log_regret
- `analysis/budget_generalization.csv`: max_train_B, policy, B, regime, mean_log_regret

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
