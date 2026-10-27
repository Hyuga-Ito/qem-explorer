# fig06_mismatch_diagnostics

**Question.** Does model mismatch predict recommendation failure, and does reacting to a diagnostic pay for its shots?

**Caption.** Model mismatch, diagnostics and reactions at equal total budget (10 base circuits × 9 execution laws; the recommender, PEC's model and every allocation see only the calibration snapshot; injected parameters are hidden). (A) Mean log10 regret of the recommender's choice with all B shots on the target, against the injected strength in physical units, at three budgets; 95 % bootstrap intervals over base circuits. (B) What one deployment observes: single draws of the echo statistic |z| (d(B) = 200 … 200000 shots charged to B) against the recommender's log10 regret on that task and budget; AUC: how well one deployment's |z| ranks failures (log10 regret > 0.3) above non-failures. (C) Paired difference of mean log10 MSE to spending all B shots on the target without a diagnostic, every grid budget pooled, by discrepancy type: withholding the diagnostic's shots without reacting; reacting to a flag with ZNE-L or SBS-B (declared fallbacks); a learned reselection (per budget, the reaction that minimized the risk on the other discrepancy types — evaluated only on the held-out type); the learned reaction triggered at random with the same mean flag rate. Under 'none' the flag rate is the false-positive rate (4.8 % at α = 0.05). Detecting a discrepancy does not identify its physical mechanism. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Observation: false-positive rate 4.8 %; single-deployment AUC 10³: 0.55, 10⁴: 0.64, 10⁵: 0.59, 10⁶: 0.70, 10⁷: 0.74; learned reselection (type held out) helps (interval < 0) under: no scenario; hurts under: drift, readout, temporal; same reaction, random trigger helps (interval < 0) under: no scenario; hurts under: drift, readout, temporal; flag → ZNE-L helps (interval < 0) under: no scenario; hurts under: crosstalk, drift, no mismatch, readout, temporal; flag → SBS-B helps (interval < 0) under: no scenario; hurts under: coherent, crosstalk, readout, temporal; diagnostic shots withheld helps (interval < 0) under: no scenario; hurts under: readout. A diagnostic earns its cost only where reacting to it lowers the risk after its shots are charged and beats the same reaction triggered at random. Simulated robustness is not hardware transfer.

**Data.** `fig06_mismatch_diagnostics__response.csv`, `fig06_mismatch_diagnostics__draws.csv`, `fig06_mismatch_diagnostics__discrimination.csv`, `fig06_mismatch_diagnostics__reactions.csv`, `fig06_mismatch_diagnostics__equal_budget.csv`

**Configuration.** `{"mismatch": {"budgets": [1000, 100000, 10000000], "facets": {"coherent": {"label": "coherent fraction \u03c7", "points": {"coherent-0.25": 0.25, "coherent-0.5": 0.5, "none": 0.0}}, "crosstalk": {"label": "ZZ crosstalk angle \u03b8 (rad)", "points": {"crosstalk-0.02": 0.02, "crosstalk-0.05": 0.05, "none": 0.0}}, "drift": {"label": "error-rate drift factor s", "points": {"drift-1.5": 1.5, "drift-2": 2.0, "none": 1.0}}}}}`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | the injected discrepancy (type and strength) between the snapshot and the execution law |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 90 mismatch tasks on 10 base circuits; r = 60 trials per half |
| Uncertainty and its scope | per-deployment diagnostic draws (one observation per deployment) vs repetition summaries, kept apart; equal-budget comparisons charge the diagnostic shots |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory; reactions trained and tested on different discrepancy types (leave one type out) |
| Infeasible, failed, unresolved, missing | infeasible diagnostics at small B are reported as such |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `mismatch/diagnostics.csv`: scenario, type, task_id, B, d, threshold, alpha, flag_rate
- `mismatch/policies.csv`: scenario, type, task_id, base_id, B, policy, risk, mean_log10_risk
- `mismatch/analysis/deployment_draws.csv`: B, abs_z, model_log_regret
- `mismatch/analysis/discrimination.csv`: B, auc_abs_z
- `mismatch/analysis/reactions.csv`: held_out_type, B, reaction

**Command.** `python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170`

**Plotting configuration.** `configs/selection/figures.yaml` (version 2, sha256 cafda9040bf74d20…)
