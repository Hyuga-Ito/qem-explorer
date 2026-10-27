# fig01_budget_efficiency

**Question.** Does recommendation improve budget efficiency over raw execution and fixed protocols?

**Caption.** Budget efficiency on held-out base circuits (development set, group 5-fold over 44 base circuits, out-of-fold; every base circuit weighs the same). (A) Mean evaluation-half MSE of each policy's chosen protocol against the total shot budget (all calibration, training and execution shots charged), with 95 % bootstrap bands over base circuits for SBS-B and the nonlinear recommender (the other policies' intervals are in the source data); the nonlinear recommender is the configuration frozen for the final test (MODEL[hgb|deployable18+simulated]); 'frozen + selective rule' accepts its choice when the confidence margin reaches τ and otherwise uses SBS-B, evaluated by nested group cross-validation (τ chosen on the inner folds only). (B) Regret performance profiles at three budgets fixed in the plotting configuration: the weighted fraction of tasks whose regret (policy MSE minus the MSE of the protocol identified on the independent selection half) is at most the tolerance; budgets are not merged. The reference is estimated, so some regrets are negative (share for the selected model: low 27 %, medium 14 %, high 5 %); they count as within every tolerance and are not clipped. (C) Paired difference of log10 MSE between the proposed policy and SBS-B by circuit family, all budgets, 95 % bootstrap intervals over the family's base circuits (count in parentheses; a family with fewer than three base circuits is drawn open, without an interval); negative values favor the proposed policy; beside it the same paired difference in absolute mean MSE (symmetric-log axis), which the few large-MSE tasks dominate. (D) Share of the total squared error contributed by the worst 5 % of held-out rows, all budgets, and the rate of severe misselection (risk more than 10× the reference's). Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Observation: over all budgets the frozen recommender's paired difference to SBS-B is -0.088 [-0.145, -0.027] in mean log10 MSE and +2.3e-05 [-5.7e-05, +0.00016] in mean MSE (negative favors the recommender); the point estimates on the two scales differ in sign: the typical (log) task and the large-MSE tasks that dominate the arithmetic mean need not favor the same policy; families (≥ 3 base circuits) whose interval excludes 0 in favor of the recommender: ghz, ising, brickwork; against it: none. Differences inside the bootstrap intervals are not evidence of a difference. Hypothesis, not tested here: per-family differences reflect which families have ranking reversals inside the budget grid.

**Data.** `fig01_budget_efficiency__curves.csv`, `fig01_budget_efficiency__profiles.csv`, `fig01_budget_efficiency__families.csv`, `fig01_budget_efficiency__catastrophic.csv`

**Configuration.** `{"policies": {"budget_fixed": "SBS-B", "fixed": "SBS", "linear": "MODEL[ridge|deployable18+simulated]", "nonlinear": "MODEL[hgb|deployable18+simulated]", "nonlinear_abstain": "MODEL[hgb|deployable18+simulated]+abstain", "raw": "RAW"}, "regret_profile": {"high": 10000000, "low": 1000, "medium": 100000}, "weights": "base_circuit"}`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | total budget B and the selection policy; held fixed: the development tasks and their labels |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 88 development tasks on 44 independent base circuits; 6 fresh designs per half for CDR and NIL; r = 60 trials per half |
| Uncertainty and its scope | 95 % bootstrap over base circuits (each base circuit weighted equally; its variants averaged first); pointwise per budget |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory (development data, group 5-fold held-out predictions); the frozen policy is evaluated once on the confirmation set elsewhere |
| Infeasible, failed, unresolved, missing | infeasible protocols are not candidates; fallback trials stay in the risk; failed label tasks are listed in label_failures.csv (0 here) and excluded |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/choices_same-distribution.csv`: task_id, base_id, family, n, device, B, <policy>|choice
- `analysis/labels.csv`: task_id, B, protocol, mse_eval, cal/fit/learn_fallback_eval, nonresponse_eval
- `analysis/nested_rows.csv`: task_id, B, base_id, accepted, choice, fallback, risk_policy
- `analysis/family_benefit.csv`: family, diff_log(_lo/_hi), diff_mse(_lo/_hi), base_circuits
- `analysis/failure_profile.csv`: policy, B, mse, worst1_share, worst5_share, severe_rate
- `freeze.json`: policy

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
