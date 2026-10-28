# supp_root_audit

**Question.** Are the labels, the rankings and the execution contract stable across independent preparation roots?

**Caption.** Crossed audit: 8 base circuits (the circuits of the ten 1.7.0 deployment regression cases, earlier failure types and controls) labeled under 4 independently namespaced preparation roots (seed scheme 2) at budgets 10,000, 300,000, 10,000,000 with the 17 candidates, r = 40 trials and 6 fresh designs per half. (A) For every (task, candidate, budget) feasible under ≥ 2 roots: the between-root variance of the evaluation-half MSE divided by the mean squared label standard error (design-level for learning candidates). Under honest standard errors the ratio scatters around 1; ratios ≫ 1 mean the label depends on rare designs or preparation draws the SE does not see. (B) Number of distinct winning candidates over the four roots per (task, budget) — with the 17 candidates and with the original eight held fixed (a subset of the same labels) — and the mean log10 regret of one root's winner judged by another root's labels. (C) Agreement of the design-level preflight (qemx.selection.preflight, each root's allocation seed) with the feasibility the labels recorded.

**Interpretation.** Observation (exploratory): median variance ratios by candidate CDR 1, FR 1, NIL 0.78, PEC 0.85, PEC-TW 0.44, RAW 0.81, RO 0.53, RO-DD 0.88, RO-TREX 0.69, RO-TW 1.1, SV 1, ZNE-E 0.82, ZNE-E-B 0.77, ZNE-L 0.63, ZNE-L-TW 0.93, ZNE-L∘SV 0.96, ZNE-R 0.71; 10 of 24 (task, budget) cells have one winner under all roots; mean cross-root log regret -0.006; contract agreement 1.0000. Ratios well above 1 identify candidates whose labels — and hence any model trained on them — are less certain than their standard errors state.

**Data.** `supp_root_audit__var_ratio.csv`, `supp_root_audit__stability.csv`, `supp_root_audit__ranking.csv`, `supp_root_audit__cells.csv`, `supp_root_audit__contract.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the preparation root (4 independently namespaced master seeds) at 3 budgets; held fixed: circuits, candidates, R, designs per half |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 8 base circuits × 4 roots; r = 40 trials and 6 fresh designs per half |
| Uncertainty and its scope | variance ratios per (task, candidate, budget) over 4 roots (few roots: a ratio is itself uncertain by a factor of about 2–3) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory (stage-2 audit) |
| Infeasible, failed, unresolved, missing | cells feasible under fewer than 2 roots have no ratio (listed in the source data) |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `audit18/analysis/label_stability.csv`: task_id, protocol, B, var_ratio
- `audit18/analysis/ranking_stability.csv`: task_id, B, distinct_winners, mean_cross_root_log_regret
- `audit18/analysis/contract.csv`: task_id, root, protocol, B, preflight, labels_feasible, agree
- `audit18/analysis/audit_summary.json`: roots, budgets, tasks, policy_by_root

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
