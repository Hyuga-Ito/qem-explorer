# supp_deployment_policy

**Question.** What does the deployed application policy do on new circuits, and how does its full-policy result compare with the model-component benchmark?

**Caption.** Once-only confirmation of the frozen 1.8.0 policy (580 rows, 29 base circuits). Primary contrast (D41, fixed after the development results and before any confirmation result): the full deployed policy of the frozen configuration (simulator features, preflight at the allocation level) against SBS-B; secondary: default (deployable features). Every row is a call of the deployment API (deploy.recommend, policy qemx.selection.policy/2) on a confirmation task at one total budget, scored with the labels' evaluation half; nothing is approximated by benchmark-only feasibility. (A) Outcome of each call: the model's top-ranked candidate accepted; accepted after the preflight skipped higher-ranked candidates that were unverified or infeasible; the SBS-B fallback because the margin was below the frozen τ, because an out-of-distribution check fired, or because no prediction existed; unsupported inputs. (B) Paired difference to SBS-B of the mean over base circuits of log10(MSE + 10⁻⁶) with 95 % bootstrap intervals over base circuits: the full policy of each bundle, the model component of the same configuration (the model's best label-feasible candidate, without OOD, preflight or margin rules), and the fixed baselines and the empirical reference of the confirmation evaluation. A difference of Δ means the geometric mean of (MSE + 10⁻⁶) changes by 10^Δ — not the MSE itself. (C) Per-row log10 ratio of the policy's risk to SBS-B's for the primary bundle, by outcome (bar = median). (D) Candidates chosen by the full policy and the model component (primary bundle; * = new in 1.8.0). (E) Share of the final candidate's evaluation trials carrying a method fallback or failure flag (part of the risk; never removed), per outcome; solid = primary bundle, light = secondary. Risk: the fresh-design average risk of the returned candidate (the labels' evaluation half averages independently drawn training designs), not the risk conditional on the one design the API returns for execution. Seconds: median classical wall time of a call that recomputed its features (the first budget of each task; later budgets reuse the cached budget-independent features). Unscored rows (a returned candidate without feasible labels): simulator features 0, default 0.

**Interpretation.** Observation (primary contrast D41, specified after the development results and before the confirmation results; evaluated once): the full deployed policy of the frozen simulator features configuration differs from SBS-B by -0.144 [-0.231, -0.076] in the base-circuit mean of log10(MSE + 10⁻⁶) (a factor 0.72 on the geometric mean of MSE + 10⁻⁶, not on the MSE itself); 60 of 580 rows used the fallback and 233 accepted a candidate after the preflight skipped a higher-ranked one. The model-component benchmark is not the application's performance; panel B shows both. Secondary (declared before the confirmation): the default (deployable features) bundle's full policy -0.084 [-0.179, -0.002]. Default bundle: on 47 rows (4 base circuits) the model component's candidate (SV 39, ZNE-L∘SV 8) was left unverified by the deployed preflight, which never accepts an unverified candidate: on 33 of them the policy accepted another candidate, on 14 the request was also out of distribution and fell back to SBS-B.

**Data.** `supp_deployment_policy__outcomes.csv`, `supp_deployment_policy__comparison.csv`, `supp_deployment_policy__log_ratio.csv`, `supp_deployment_policy__chosen.csv`, `supp_deployment_policy__execution.csv`, `supp_deployment_policy__rows.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the bundle (default deployable-feature bundle, simulator-feature bundle); held fixed: the frozen configuration and the confirmation labels |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 58 confirmation tasks on 29 base circuits (none in development); r = 60 trials per half |
| Uncertainty and its scope | 95 % bootstrap over base circuits of the paired difference to SBS-B (each circuit's rows averaged first) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | confirmatory, once only: the primary contrast (D41) is the frozen simulator-feature configuration at allocation-level preflight against SBS-B, specified after the development results and before the confirmation results (not a pre-development registration); the default deployable-feature bundle is a secondary analysis |
| Infeasible, failed, unresolved, missing | every row kept: fallbacks, skipped candidates and execution fallbacks are part of the policy; a returned candidate without feasible labels is counted as unscored (no imputed risk) |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `replay/replay_<bundle>.csv`: task_id, base_id, B, status, accepted, choice, model_choice, skipped, risk, scored, seconds
- `replay/replay_<bundle>.json, replay/replay_summary.json`: diff, diff_lo, diff_hi, unscored, model_component_benchmark
- `confirm/choices_confirm.csv`: task_id, B, SBS-B|choice, <configuration>|choice
- `confirm/labels.csv`: task_id, protocol, B, mse_eval, fail_any_eval
- `confirm/metrics_confirm.csv`: policy, B (= all), diff_vs_sbsb(_lo/_hi)

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
