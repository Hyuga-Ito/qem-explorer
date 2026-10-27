# supp_dataset_coverage

**Question.** Which part of the feature space do the development and final-test sets cover?

**Caption.** Dataset coverage: distributions of deployable features for the development, historical final-test and confirmation tasks (step histograms: share of tasks for the integer-valued width and observable weight, density over quantile bins otherwise), and — separately, in red, because it is oracle information used only for labels — the magnitude of the ideal value |μ|; 24 % of these tasks have |μ| < 0.1 (dotted line). Noise components: the gate share of the light-cone fault weight against the readout weight for every task, with the controlled settings of the noise-composition study (diamonds) for comparison — the development devices vary the components only jointly, so the controlled study reaches compositions the development set does not. Counts: tasks (failed label tasks listed separately), independent base circuits (the unit of every bootstrap), fresh learning designs per label half for the learning candidates (CDR, NIL; FR in 1.8.0), and trials per half. Support by family and width is in the source data. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Observation: development: median n 4, two-qubit gates 6, depth 33, λ_LC 0.048, weight 1; base circuits: development 27. Similar medians do not establish that the final test covers the same regions of feature space; the histograms and the support table show where it is thin.

**Data.** `supp_dataset_coverage__tasks.csv`, `supp_dataset_coverage__support.csv`, `supp_dataset_coverage__counts.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | — |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 54 development tasks on 27 independent base circuits; 3 fresh designs per half for CDR and NIL; r = 24 trials per half; final test 0 tasks on 0 base circuits; confirmation 0 tasks on 0 base circuits |
| Uncertainty and its scope | none (distributions) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | descriptive |
| Infeasible, failed, unresolved, missing | failed label tasks counted separately |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `dev/tasks/*.json, final/tasks/*.json`: spec (family, n), base_id, ideal, features

**Command.** `python -m qemx.selection figures configs/selection/pilot19.yaml --out reports/pilot_v190 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
