# supp_runtime

**Question.** What does one recommendation cost classically as the circuit grows?

**Caption.** Classical cost of one recommendation against width for a 2-step ising circuit on fake_fez at B = 10⁵, through the deployment path of the Playground and the recommend command: compilation, the deployable features (structure, observable, light cone, calibration-derived noise sums and the closed-form protocol overheads — no simulation of the circuit), the simulator-only augmentation (per-shot variances and signal of RO, ZNE-R and PEC on the snapshot by exact density-matrix simulation and the ideal state's entropies — exponential in width; only for a model that uses them) and model inference; medians of 3 repeats in one process. Dashed: for comparison, the median wall time of generating the benchmark labels of one development task of that width (all protocols, all budgets, both label halves; 18.0/18.0/18.0/0.0 tasks per width; missing where the development set has no task of that width) — a cost of building the study, not of a recommendation. Memory: peak (resource), in the source data. Measured on one machine; not a benchmark of other hardware. 1.8.0 components: the structural preflight of every candidate of the portfolio, the design-level preflight of the learning candidates (training design with its deployment seed, ideal training values by state vector), and the whole recommend() call with the deployable bundle (guard, compilation, features without cache, out-of-distribution checks, preflight in the model's order, margin rule, fallback). No component uses quantum shots. (B) Wide GHZ chains (12, 24, 40 qubits; observable z_mid) outside every training width: the whole call and the process's peak memory; status as returned (an out-of-distribution answer uses the fallback; no dense state of the circuit is built).

**Interpretation.** Observation: at n = 6 the largest component of a recommendation is 'simulated'; the simulator-only augmentation grows by a factor 3.6 from n = 3 to n = 6, while a recommendation from deployable features needs no simulation. Label generation, the study's own cost, is orders of magnitude above a recommendation.

**Data.** `supp_runtime__runtime.csv`, `supp_runtime__medians.csv`, `supp_runtime__rejection.csv`

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

- `analysis/runtime.csv`: kind, n, compile_s, deployable_s, simulated_s, inference_s, preflight_structural_s, preflight_design_s, recommend_s, recommend_status, label_s, label_tasks, peak_rss_mb, rss_kind (1.8.0 columns optional)

**Command.** `python -m qemx.selection figures configs/selection/pilot19.yaml --out reports/pilot_v190 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
