# supp_runtime

**Question.** What does one recommendation cost classically as the circuit grows?

**Caption.** Classical cost of one recommendation against width for a 2-step ising circuit on fake_fez at B = 10⁵, through the deployment path of the Playground and the recommend command: compilation, the deployable features (structure, observable, light cone, calibration-derived noise sums and the closed-form protocol overheads — no simulation of the circuit), the simulator-only augmentation (per-shot variances and signal of RO, ZNE-R and PEC on the snapshot by exact density-matrix simulation and the ideal state's entropies — exponential in width; only for a model that uses them) and model inference; medians of 3 repeats in one process. Dashed: for comparison, the median wall time of generating the benchmark labels of one development task of that width (all protocols, all budgets, both label halves; 39/27/33/21/0 tasks per width; missing where the development set has no task of that width) — a cost of building the study, not of a recommendation. Memory: peak (resource), in the source data. Measured on one machine; not a benchmark of other hardware.

**Interpretation.** Observation: at n = 7 the largest component of a recommendation is 'simulated'; the simulator-only augmentation grows by a factor 3.5 from n = 3 to n = 7, while a recommendation from deployable features needs no simulation. Label generation, the study's own cost, is orders of magnitude above a recommendation.

**Data.** `supp_runtime__runtime.csv`, `supp_runtime__medians.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | circuit width |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | one circuit family; repeats in one process |
| Uncertainty and its scope | median of repeats |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | measurement on one machine |
| Infeasible, failed, unresolved, missing | label time missing where the development set has no task of that width |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/runtime.csv`: n, compile_s, deployable_s, simulated_s, inference_s, label_s, label_tasks, peak_rss_mb, rss_kind

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
