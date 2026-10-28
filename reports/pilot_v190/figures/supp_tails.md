# supp_tails

**Question.** How heavy are each candidate's loss tails, and do single training designs carry them?

**Caption.** Complete loss distributions on the development set (54 tasks, 27 base circuits; feasible rows only, each candidate's own feasible subset — the panels describe each candidate's tail, they do not rank candidates on a common subset). (A) Per candidate and budget: median, 90 % quantile and maximum over feasible rows of the log10 ratio of its evaluation-half risk to RAW's on the same row. (B) Base-circuit-weighted share of feasible rows with risk above 10 × RAW's (the declared severe degradation, finding F7), every budget; – = no feasible row. (C) Share of the candidate's total circuit-averaged loss carried by the worst 5 % of base circuits (dotted line: the share if all circuits lost equally; right: feasible base circuits — with few circuits the share is necessarily high). (D) For the learning candidates: the largest single training design's share of the summed design losses of a label half (K designs per half; dotted 1/K = equal designs); values near 1 mean one design carried the label's loss. No estimate is clipped or removed; failures and fallbacks are part of every risk.

**Interpretation.** Observation (descriptive): the highest severe-degradation rates are FR at B = 30,000: 70 %; NIL at B = 30,000: 67 %; FR at B = 30,000,000: 67 %. Heavy tails are a property of a candidate at a budget, not of its median: compare panel A's maxima with its medians before relying on a candidate whose typical loss is low.

**Data.** `supp_tails__relative_to_raw.csv`, `supp_tails__severe.csv`, `supp_tails__concentration.csv`, `supp_tails__designs.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | candidate and budget |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 54 development tasks on 27 independent base circuits; 3 fresh designs per half |
| Uncertainty and its scope | none (distributions and rates of the labels) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | descriptive (development data) |
| Infeasible, failed, unresolved, missing | each candidate on its own feasible subset (– where none); nothing clipped or removed |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/labels.csv`: task_id, base_id, protocol, B, feasible, mse_eval
- `analysis/design_diagnostics.csv`: task_id, protocol, B, half, design, mean_loss

**Command.** `python -m qemx.selection figures configs/selection/pilot19.yaml --out reports/pilot_v190 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
