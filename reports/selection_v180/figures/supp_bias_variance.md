# supp_bias_variance

**Question.** Which part of each protocol's error is a floor that more shots do not remove?

**Caption.** Bias–variance decomposition and error floors for the three regime slices at s = 1 (evaluation half, r = 60 trials per cell) and 7 protocols. Top: squared bias corrected for finite samples, ē² − SE(ē)², where SE(ē) matches the estimand — over trials for protocols without a training design, over the 6 fresh training designs of a half for the learning candidates (CDR, NIL; FR in 1.8.0) (cluster-robust); bands: squares of the 95 % interval of the mean error, lower end clipped at 0; estimates ≤ 0 (a bias below the resolution of the data) are drawn as open markers at 10⁻¹⁰ and the line is broken there. Bottom: variance of one estimate (solid) and, for the learning protocols, its between-design component Var_D(E[X̂|D]) from a one-way analysis of variance (dashed; omitted where estimated as 0). MSE = bias² + variance of one estimate; the between-design part does not fall with more target shots. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Observation: at B = 3·10⁷ the corrected squared bias exceeds the variance of one estimate for — ising: PEC, RAW, RO, RO-TW, ZNE-E-B, ZNE-L; hea: PEC, RAW, RO, RO-TW, ZNE-E-B, ZNE-L; qaoa: PEC, RAW, RO, RO-TW, ZNE-L. Between-design share of the variance at that budget: hea CDR 94 %, ising CDR 91 %, qaoa CDR 16 %. A bias² that stays flat while the variance falls is an error floor: more shots do not remove it; a between-design share that grows with B is variance that only more (or better) training designs remove.

**Data.** `supp_bias_variance__curves.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | budget per protocol at s = 1 on the three regime slices |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 21 regime tasks; r = 60 trials per half; 6 designs per half |
| Uncertainty and its scope | 95 % interval of the mean error with the SE matching the estimand (design-level for CDR, NIL) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | descriptive |
| Infeasible, failed, unresolved, missing | bias² estimates ≤ 0 drawn as open markers at 10⁻¹⁰ |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `regimes/tasks/<task_id>.json, .npz`: spec, budgets, protocols, ideal, allocations, est, design
- `regimes/analysis/regime_cells.csv`: task_id, noise_scale, observable, layout

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
