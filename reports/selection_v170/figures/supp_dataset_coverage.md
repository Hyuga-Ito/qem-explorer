# supp_dataset_coverage

**Question.** Which part of the feature space do the development and final-test sets cover?

**Caption.** Dataset coverage: distributions of deployable features for the development, historical final-test and confirmation tasks (step histograms: share of tasks for the integer-valued width and observable weight, density over quantile bins otherwise), and — separately, in red, because it is oracle information used only for labels — the magnitude of the ideal value |μ|; 20 % of these tasks have |μ| < 0.1 (dotted line). Noise components: the gate share of the light-cone fault weight against the readout weight for every task, with the controlled settings of the noise-composition study (diamonds) for comparison — the development devices vary the components only jointly, so the controlled study reaches compositions the development set does not. Counts: tasks (failed label tasks listed separately), independent base circuits (the unit of every bootstrap), fresh learning designs per label half for CDR and NIL, and trials per half. Support by family and width is in the source data. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Observation: confirmation: median n 5, two-qubit gates 10, depth 42, λ_LC 0.050, weight 1.5; development: median n 4, two-qubit gates 8, depth 25, λ_LC 0.049, weight 2; final test (historical): median n 4.5, two-qubit gates 8, depth 34, λ_LC 0.055, weight 2; base circuits: confirmation 25, development 40, final test (historical) 12, noise-composition study 3. Similar medians do not establish that the final test covers the same regions of feature space; the histograms and the support table show where it is thin.

**Data.** `supp_dataset_coverage__tasks.csv`, `supp_dataset_coverage__support.csv`, `supp_dataset_coverage__counts.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | — |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 120 development tasks on 40 independent base circuits; 10 fresh designs per half for CDR and NIL; r = 100 trials per half; final test 36 tasks on 12 base circuits; confirmation 50 tasks on 25 base circuits |
| Uncertainty and its scope | none (distributions) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | descriptive |
| Infeasible, failed, unresolved, missing | failed label tasks counted separately |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `dev/tasks/*.json, final/tasks/*.json`: spec (family, n), base_id, ideal, features

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
