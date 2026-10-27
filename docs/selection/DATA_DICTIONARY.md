# Data dictionary — protocol-selection study (1.7.0)

## Configuration (`configs/selection/*.yaml`, schema `qemx.selection.config/1`)

| field | meaning |
|---|---|
| `name`, `master_seed` | workflow name; seed of every random stream (with task, budget, protocol and half) |
| `R` | trials per half per (task, protocol, budget); 2R in total |
| `budgets` | total shot budgets B |
| `protocols` | portfolio keys (`qemx.selection.protocols.PORTFOLIO`) |
| `final_test_fraction` | share of base circuits held back (keyed hash of `base_id`) |
| `tasks.widths`, `tasks.widths_by_family` | widths n (qubits of the logical circuit; BV counts its ancilla) |
| `tasks.families` | family → list of parameter sets |
| `tasks.inputs`, `observables`, `layouts`, `devices`, `variants_per_base` | variant pools and how many non-default variants each base circuit gets |
| `status` | `executed`, `planned` (documentation; `plan` reports it) |
| `learning_designs` | 1.7.0: `{mode: fresh, per_half: K}` — K independent training designs per label half for CDR/NIL; an integer is the 1.6.0 `fixed-shared` plan (K designs shared by both halves; digests unchanged) |
| `limits.task_seconds`, `limits.memory_gb` | per-task wall-clock and resident-memory limits, enforced by the parent of each supervised child process |
| `limits.cpu_hours`, `limits.disk_gb` | caps compared with `plan`'s estimate (a configuration over its caps is reported, not started unknowingly) |
| `regimes.noise_axis` | `scale` (regime study) or `component` (noise-composition study) |
| `analysis.*` | splits and their options, feature sets, model families, abstention model, bootstrap size |
| `mismatch.*` | scenarios, diagnostic fraction and minimum, α, robust protocol, model family and set |

## Task (`TaskSpec`, schema `qemx.task/1`)

| field | meaning |
|---|---|
| `family`, `n`, `params` | the logical circuit |
| `inputs` | `default` / `plus` / `product` |
| `observable` | `default` / `z_mid` / `zz_mid` / `z_all` |
| `device`, `layout` | calibration snapshot and chain (`best` / `alt`) |
| `opt_level`, `seed` | transpiler settings |
| `truth` | execution-law modifiers (`chi=`, `s=`, `ro=`, `xt=`, `mix=`); empty = the snapshot |
| device modifiers | `fake_fez@s=2,g2=3,…`: `s` every error rate × s; `g1`, `g2` one-/two-qubit gate errors ×; `t` T1 and T2 ÷ t; `ro` readout ×; `chi` coherent fraction of every gate error (exact engine); `hom` homogenized rates; `eng` engine — applied to the snapshot and the execution law alike |
| `task_id`, `base_id` | identity (every field) and ancestry (logical circuit only) |

## Labels (`<run>/<part>/tasks/<task_id>.json` and `.npz`, schema `qemx.selection.labels/2`; 1.6.0 records `/1` are read)

JSON record: `task_id`, `base_id`, `spec`, `config_digest`, `run_id`, `budgets`, `protocols`, `R`, `master_seed`,
`status` (`ok`, `error`, `timeout`, `memory`; non-ok records carry `error_kind`, `error`, `seconds` and no arrays),
`ideal` (oracle), `task` (reproducibility record: inputs, observable, basis, relabel, layout, physical qubits,
measurement map, compiled two-qubit count and depth, circuit digest), `features` (oracle entries included for research
only), `learning_designs` and `design_plan` (mode `fresh` or `fixed-shared`, `per_half`, `count`), `estimand` (text),
`allocations` (one per protocol × budget: `status` ok / infeasible / method-error, feasibility, reason, N, s, c, s_tr,
training circuits, ZNE split, ledger by role, information used), `costs` (T_exec, circuits sent), `method_status`,
`design_records` (per learning protocol, one per drawn design: half, index, seed, `design_id`, `content_digest`,
status, fallback, `diagnostics` — n_train, n_features, rank, spread and spread_needed, l1, nonzero, solver_ok, twins,
y_min, y_max, sv_max, sv_min, condition, exact_feature_estimate; research only: oracle_error,
oracle_ideal_in_label_range), `design_status`, `prepare_seconds`, `seconds`, `flags`, `arrays` (sha256 of the NPZ;
a mismatch is refused as corrupted).

NPZ arrays: `est` (2, P, nB, R) unclipped estimates of the selection (0) and evaluation (1) halves, NaN where
infeasible; `flags` (2, P, nB, 4, R) booleans in the order `cal_fallback`, `fit_fallback`, `learn_fallback`,
`nonresponse`; `design` (2, P, nB, R) the trial's design index in its half's pool (−1: no design); `trial_seconds`
(P, nB). 1.6.0 arrays without `design` get the round-robin indices of their six shared designs when read.

`run.json` (schema `qemx.selection.run/1`): `identity` (stage; digests of `labels_config`, `tasks` with `n_tasks`,
`split`, `portfolio`, `devices`), `identity_digest`, `config`, `manifest` (task ids), `sessions` (started, execution:
code digest and environment, supervision), `imports` (explicitly reused tasks with their reason).

Beside the task files: `provenance.json` (schema `qemx.selection.provenance/2`; `stage`; `sessions`, one per
labeling session: `started`, `provenance` — app version, source/lock/inventory digests, git commit and whether the
tree was dirty, Python, platform, key package versions — `engine_sha256` per engine file, and for a resumed run
`changed_since_first`) and `label_summary.json` (last session: tasks, ok, skipped, errors, seconds,
per_task_seconds of the tasks labeled in it).

## Analysis tables (`<run>/analysis/`)

| table | columns |
|---|---|
| `labels.csv` | task_id, base_id, family, n, device, truth, protocol, B, feasible, status (ok / infeasible / method-error), reason, ideal, design_mode, r, shots_{calibration, training, diagnostics, selection, execution}, t_exec, circuits_sent, and per half (sel, eval): mse, bias, var, se_mse (matching the estimand: design-level for CDR/NIL under fresh designs), se_mse_trial, se_mse_design, var_between_design, var_within_design, designs, mse_clip, the four flag rates, fail_any (share of trials with ≥ 1 flag; the union), flags_per_trial |
| `label_failures.csv` | task_id, status (error / timeout / memory), error_kind, error — label tasks without labels, excluded from the dataset |
| `collection.json` | legacy (no run.json), manifest size, missing and unexpected task ids, failures, imports, rows, design_mode |
| `design_diagnostics.csv` | per task, learning protocol, budget (10⁴, 10⁶, 3·10⁷), half and design: task_id, base_id, family, n, device, protocol, B, half, design, design_id, seed, status, fallback, trials, mean_loss, mean_estimate, ideal, fallback_rate, and every `diag_*` diagnostic of the design record |
| `design_variance.csv` | per learning protocol and budget: tasks, designs, median_se_ratio and p90_se_ratio (design-level / trial-level SE), median_between_share (Var_between / (Var_between + Var_within)), median_design_share_of_label_variance (Var_between / (Var_between + Var_within/m)), share_tasks_between_dominant |
| `failure_profile.csv` | per policy and budget (and `all`): rows, base_circuits, mse (base-circuit weighted), worst1_share and worst5_share (share of the total squared error from the worst 1 % / 5 % of rows), severe_rate (log10 regret > 1), max_log_regret, worse_than_raw_2x |
| `nested_summary.csv` | per policy of the nested evaluation (policy = the selective rule at its nested τ; model; fallback (SBS-B); random rejection; SIZE-B; SBS): base_circuits, rows, coverage, mse with mse_lo/hi, mean_log10_risk with log_lo/hi, diff_mse_vs_sbsb, diff_log_vs_sbsb, diff_log_vs_random (each with _lo/_hi), random_draws_mse_lo/hi and random_draws_log_lo/hi (2.5–97.5 % over random acceptance sets), accepted_log_regret, configuration, evaluation |
| `nested_folds.csv` | fold, tau (chosen on the inner folds of the outer training circuits), inner_rows, test_rows, test_base_circuits, inner_base_circuits, accepted |
| `nested_rows.csv` | per outer test row: fold, task_id, base_id, family, device, n, B, margin, tau, accepted, choice, fallback, risk_/log_risk_ of model, fallback, SIZE-B, SBS and the policy, risk_ref, log_regret_model, log_regret_fallback |
| `nested_curve.csv` | descriptive sweep of τ on the outer rows: tau, coverage, policy_mse, policy_log10_risk, random_policy_mse, random_policy_log10_risk, accepted_log_regret |
| `nested_operating.csv` | deploy_tau (frozen by `freeze --complete`), rule, chosen_on, performance, configuration |
| `nested_groups.csv` | Table S1: group (family / device), value, rows, base_circuits, rejection_rate (base-circuit weighted), statement |
| `summary_<split>.csv` | split, part, policy, B (or `all`), rows, base_circuits; risk: mean_log10_risk with ci_lo/ci_hi, policy_mse (base-circuit-weighted arithmetic mean MSE); regret against the per-task reference (the protocol identified on the selection half, scored on the evaluation half): mean_log_regret with log_regret_lo/log_regret_hi, median_log_ratio, p90_log_regret, mean_regret, median_regret, p90_regret (absolute, MSE units), p_negative_regret, p_regret_gt_10pct, p_regret_gt_2x, tol_success_10pct and tol_success_2x (share within 10 % / a factor 2 of the reference); raw_degradation (share of rows worse than RAW); fail_share (mean share of trials on a fallback path or without response); coverage, accepted_log10_risk; gap_closed with gap_lo/gap_hi; paired differences diff_vs_sbs (diff_lo/diff_hi), diff_vs_sbsb and diff_vs_sizeb with their `_lo`/`_hi` |
| `choices_<split>.csv` | per test row: split, part, task_id, base_id, family, n, device, B, and per policy `<policy>|choice`, `<policy>|log_ratio`, `<policy>|accepted` |
| `pooled_<split>.csv` | the columns of `summary_<split>.csv` (without part) for all test rows of the split scored together, one bootstrap over all test base circuits — the reported numbers (computed from `choices_<split>.csv` and `labels.csv`) |
| `optimism.csv` | per split and reference policy (`ref`: SBS, SBS-B): candidates, base_circuits, naive_best, naive_mean_log10_risk, naive_diff_vs_ref, bbc_mean_log10_risk with bbc_lo/bbc_hi, bbc_diff_vs_ref with bbc_diff_lo/bbc_diff_hi, p_diff_below_0, optimism, ref, samples, selection_frequency (top 5) |
| `family_benefit.csv` | per family: policy (the frozen configuration), reference (SBS-B), base_circuits, rows, diff_log (paired mean difference of log10 MSE, policy − reference) with diff_log_lo/hi, diff_mse (absolute) with diff_mse_lo/hi |
| `ablation.csv` | variant (`all`, `-<group>`, or a semantic removal such as `-lightcone (all derived)`), family, base_set, tuned, kind (full / group / semantic), removed (the features removed), base_circuits, uncertainty, mean_log_regret with ci_lo/ci_hi, mean_regret, diff_vs_all with paired diff_lo/diff_hi |
| `learning_curve.csv` | fold, fraction, repeat, train_base_circuits, test_base_circuits, gap_closed, mean_log10_risk, diff_vs_sbs, mean_log_regret, sbs_mean_log_regret (SBS fitted on the same subset) |
| `budget_generalization.csv` | max_train_B (`full` or the largest training budget), policy, B (test budget), regime (`in range` / `extrapolation`), mean_log_regret with ci_lo/ci_hi, base_circuits |
| `deployed_rows.csv` | the frozen configuration's out-of-fold rows: fold, task_id, base_id, family, device, n, B, choice, fallback (SBS-B of the same training fold), margin z, predicted_log_risk, log_regret, log_regret_fallback, log_risk, log_risk_fallback, regret, regret_fallback, risk, risk_fallback, configuration |
| `prediction.csv` | fold, task_id, family, B, protocol, predicted (held-out log10 risk), observed (evaluation half), observed_se (delta-method SE of the observed log10 MSE), observed_sel, chosen |
| `calibration.csv` | fold, nominal, kind (`per candidate`, `selected candidate`, `simultaneous`), coverage, mean_width (decades), rows |
| `selective_curve.csv` | per threshold τ: coverage with coverage_lo/hi, accepted_rows, accepted_log_regret with accepted_lo/hi (accepted rows pooled), policy_log_risk with policy_lo/hi, policy_log_regret, policy_mse with policy_mse_lo/hi (whole policy after the fallback), and the random-rejection baselines random_accepted_log_regret, random_policy_log_risk, random_policy_mse |
| `selective_folds.csv` | fold, tau (chosen on the other folds), validation_policy_log_risk, test_rows, test_accepted |
| `selective_operating.csv` | the cross-fitted test operating point: coverage, accepted_log_regret, policy_log_risk, policy_log_regret, policy_mse, the model-only and fallback-only values (model_*, fallback_*), and deploy_tau (chosen on all development rows; used by later studies) |
| `selective_groups.csv` | group (family, device), value, rows, base_circuits, rejection_rate |
| `selective_rows.csv` | fold, task_id, base_id, B, margin, accepted_crossfit |
| `abstention.csv` | alpha, coverage, accepted_log10_risk, all_log10_risk, coverage by family and device (group-conformal abstention; see METHODS) |
| `interval_coverage.csv` | fold, kind (cv-residual, cv-residual-simultaneous, joint (simultaneous), residual-gaussian, split-conformal, split-conformal-simultaneous), coverage, nominal, width_log10, n, groups, statement (the claims vocabulary), finite (split conformal: false when G calibration circuits are too few for the level, the interval is then unbounded), calibration_circuits, proper_training_circuits, unit |
| `floor_sensitivity.csv` | variant (floor of the log transform), mean_log10_risk with ci, diff_vs_default with paired diff_lo/diff_hi |
| `certificates.csv` | task_id, family, B, method (hoeffding, bernstein, two-eps), r, chosen, U_chosen, excess_bound, improves_on_raw, informative (= improves_on_raw), radius (the deviation term t of the chosen protocol: U = min(H, mean + t)), halfwidth ((U − L)/2 with L clipped at 0), units_chosen (independent units of the chosen protocol: trials, or design means for CDR/NIL under fresh designs), mse_clipped, mse_unclipped, statement (finite-sample-bound for the clipped estimator; post-hoc-check for two-eps), scope, eps, regret, bound, holds |
| `runtime.csv` | `profile`: family, n, repeat, compile_s, deployable_s (features without simulation), simulated_s (simulator-only augmentation), inference_s, label_s (median wall time of labeling one development task of that width, for comparison), label_tasks, peak_rss_mb, rss_kind, configuration |
| `task_features.csv` | one row per task: ancestry, variant fields and every feature |
| `results.json` | schema `qemx.selection.results/1`: protocols, budgets, row/task/base-circuit counts, the evaluations run (parts, table names) and the extras with their seconds |

The analyses `deployed`, `selective`, `ablation`, `learning_curve`, `budget_generalization`, `abstention`,
`interval_coverage`, `certificates` and `floor_sensitivity` are run with `analyze --extras …`; `selective` recomputes
the `selective_*` tables from `deployed_rows.csv` without refitting a model.

## Historical final test (`<run>/final/`) and confirmation (`<run>/confirm/`)

`final/` holds the 1.6.0 final test relabeled with fresh designs and re-evaluated once with the frozen 1.7.0
configuration (historical, not confirmatory); `confirm/` the confirmation set (`confirm17.yaml`), labeled after the
freeze and evaluated once (`summary_confirm.csv`, `choices_confirm.csv`, `labels.csv`, `metrics_confirm.csv`,
`EVALUATED` with the training parts, rows and base circuits and the freeze it used). Policies in both: the frozen
configuration, its selective variant at `deploy_tau`, SBS-B, SIZE-B, SBS, RAW, the all-candidate reference, and the
declared secondary comparators Ridge, the structured model and boosted trees on deployable features (D39).

| file | content |
|---|---|
| `EVALUATED` | written once by `final`; a second evaluation is refused |
| `summary_final.csv`, `choices_final.csv` | the once-only evaluation of every policy (development-trained, frozen choice included) |
| `metrics_final.csv` | the pooled columns of `summary_<split>.csv` recomputed from `choices_final.csv` and `labels.csv` (no refit, no new labels) |
| `labels.csv` | the final-test labels in the format of `analysis/labels.csv` |

## Controlled regime study (`<run>/regimes/`)

Labels as in `dev/` (tasks of `configs/selection/regimes.yaml`; devices `fake_fez@s=X` scale every calibrated error
rate of the snapshot and of the execution law by X). `regimes/analysis/` below; the noise-composition study
(`noisecomp17.yaml`) has the same layout under `<run>/noisecomp/`.

| file | columns |
|---|---|
| `regime_cells.csv` | per task × budget: task_id, base_id, family, device, noise_scale, noise_change (the non-default device modifiers, e.g. `g2=3`; `none`), lam_total and lam_lc (snapshot fault weights of that setting), observable, layout, B, n_feasible, trials_per_candidate (both halves), reference (lowest MSE over both halves), runner_up, unresolved (runner-up within 1.96 SE of the difference; design-level SE for CDR/NIL under fresh designs; pointwise), model_choice, margin, abstained (margin < deploy_tau), fallback_choice (SBS-B), policy_choice, log_regret_model, regret_model, log_regret_policy, log_regret_fallback, deploy_tau, tau_source, configuration, statement, resolution |
| `regime_protocols.csv` | per task × budget × feasible protocol: mse_pooled and se_pooled (both halves), mse_eval, mse_sel, predicted_log_risk, chosen |
| `regime_features.csv` | the features of every regime task (as `task_features.csv`; oracle columns `ideal`, `raw_bias_true` are never model inputs) |
| `labels.csv` | the regime labels in the format of `analysis/labels.csv` |
| `model_training.json` | policy, train_rows, dev_rows, train_parts, tau_source, excluded_base_circuits (regime circuits removed from training), deploy_tau |

## Figures and tables (`<run>/figures/`, `<run>/tables/`)

Every figure `<name>` is written as `<name>.pdf`, `.svg` and `.png` (300 dpi), with `<name>.md` (question, caption,
interpretation, data files, plotting settings, inputs with columns, command, plotting-configuration version and hash)
and its plotted data as `<name>__<key>.csv` — the exact numbers drawn, computed from the analysis tables above, so a
figure can be checked or redrawn without the code (`docs/selection/FIGURES.md`). Tables: `table1_design`,
`table1b_studies`, `table2_protocols`, `table3_policy_results` (confirmation), `table3h_historical_final`,
`table_s1_rejection` as CSV and Markdown. The notes of every figure carry its methods box. `figures/PENDING.md` lists
figures not drawn for missing inputs. `RESULTS_TABLES.md` (written by `report`) collects
the result tables quoted in `docs/selection/RESULTS.md`.

## Mismatch (`<run>/mismatch/`)

Every policy spends exactly B shots in total (1.7.0, equal-budget comparison; `qemx/selection/run.py`
`mismatch_analyze`). Policies: `TARGET@B` (no diagnostic, the model's choice with all B shots), `NO-DIAG@B-d` (the
model's choice at B − d, the diagnostic's shots withheld), `FIXED→ZNE-L` and `FIXED→SBS-B` (on a flag, that protocol
at B − d), `DIAG-RESELECT` (on a flag, the learned reaction of `analysis/reactions.csv`, trained without the held-out
discrepancy type), `RANDOM→reaction` (the same reaction triggered at random with the matched mean flag rate), `BEST`
(the per-task reference: identified on the selection half, scored on the evaluation half). Flag-triggered risks are
exact mixtures P(flag)·R_react(B − d) + (1 − P(flag))·R_model(B − d).

| file | columns |
|---|---|
| `scenarios.json` | scenario, task_id (with the perturbation), truth modifiers, base_task |
| `diagnostics.csv` | scenario, type (discrepancy type: none, coherent, drift, readout, crosstalk, temporal), task_id, base_id, B, d (echo shots charged to B), p_true and p_sup (echo return probability under the execution law and under the snapshot), threshold, alpha, flag_rate (exact binomial probability that one deployment flags), z_q10, z_median, z_q90 (2000 simulated tests), model_log_regret (the recommender's log10 regret on that task and budget with all B shots), model_choice |
| `policies.csv` | scenario, type, task_id, base_id, family, B, policy (above), risk (evaluation-half MSE, or the exact mixture), mean_log10_risk, model_choice, p_flag (flag probability used in the mixture), diag_shots (d, 0 for TARGET@B), reaction (the reaction protocol of the flag-triggered policies) |
| `analysis/equal_budget.csv` | type, B, policy, reference (TARGET@B or NO-DIAG@B-d), diff_log10 (paired difference of mean log10 risk, < 0 favours the policy), lo, hi (95 % bootstrap over base circuits), base_circuits, rows |
| `analysis/reactions.csv` | held_out_type, B, reaction (the learned reaction used when that type is held out; for type none, trained on every mismatch type) |
| `analysis/deployment_draws.csv` | scenario, type, task_id, B, draw, abs_z (one simulated deployment's echo statistic), flagged, model_log_regret — single-deployment draws, kept apart from repetition summaries |
| `analysis/discrimination.csv` | B, deployments, failures (draws with model log10 regret > 0.3), auc_abs_z (how well one deployment's |z| ranks failures above non-failures), flag_rate_failures, flag_rate_ok |
| `analysis/labels.csv` | per (task, protocol, budget) label summary of the mismatch tasks (columns as in `analysis/labels.csv` above) |
| `model_training.json` | policy, training rows, excluded base circuits (the mismatch circuits are never in the training set) |
| `null_check.csv` | p_sup, d, simulated_false_positive (4000 tests under the finite-shot null, scenario `none`) |

## Estimand comparison, design study, hardware plan (`reports/selection_v170/`)

| file | columns |
|---|---|
| `analysis_estimands*/identity.csv` | protocol, tasks, identical (tasks whose trials are identical in the 1.6.0 and the corrected labels; `_final`, `_regimes`: the historical final test and the regime study) |
| `analysis_estimands*/learning.csv` | task_id, family, n, protocol (CDR, NIL), B, mse_old, se_old_reported (trial-level, as 1.6.0 reported it), se_old_design, designs_old, mse_new, se_new (design-level), designs_new, plan_old, plan_new, se_ratio_new_over_old, log10_mse_ratio, old_outside_new_interval |
| `analysis_estimands*/summary.csv` | protocol, B, tasks, median/mean MSE under both estimands, median_se_ratio, median_abs_log10_mse_ratio, share_old_outside_new_interval |
| `design_study/designs.csv` | task, task_id, variant (CDR, CDR-T32, CDR-f50, NIL, NIL-T32, NIL-f25, NIL-f75, NIL-bare), B, status, reason, design, seed, fallback, trials, mean_loss (the design's MSE over its trials), mean_estimate, ideal, fallback_rate, s_tr, diagnostics `diag_*` (training circuits and features, rank resolved above shot noise, label spread and the spread needed, coefficient ℓ1 norm and non-zeros, solver status, duplicate training circuits, label range, extreme singular values and condition number, exact-feature estimate; research only: the oracle error and whether the ideal value lies in the training labels' range) |
| `design_study/summary.csv` | task, variant, B, designs, mse, se_design, median_design_loss, max_design_loss, severe_share (designs with MSE > 10⁻²), severe_contribution (their share of the variant's MSE), fallback_share, per diagnostic its mean over severe and over other designs (`*_severe`, `*_other`) |
| `hardware/hardware_plan.json` | schema, budget, device_model, backend, blocks, ceiling_s, reserve_s, runs (task, protocol, roles, supported, reason, circuits, shots per run), jobs (job, executions, circuits, low/central/high usage estimate in s, reserved_s, admitted), bundle and its digest, digest of the content, remaining_s, reserved_s, admitted_jobs, total_jobs, fits, executed (always false) |
| `TRIALS_SHA256SUMS` | SHA-256 and path of every per-trial array (`*.npz`) of the run, shipped in the separate trial-data archive |

## Protocols

| key | estimator | readout | training share | description |
|---|---|---|---|---|
| RAW | raw | none | 0.0 | Unmitigated: all B shots on the target circuit; sample mean of the observable. |
| RO | raw | tensored | 0.0 | Readout correction only: tensored inverse of per-qubit assignment matrices estimated from two calibration circuits (|0…0⟩, |1…1⟩); calibration/target split by the supplied model. |
| ZNE-L | zne | tensored | 0.0 | Zero-noise extrapolation, global folding at λ = 1, 3, linear fit, tensored readout correction; shots per scale ∝ |η_j|σ_j of the supplied model (integer optimal split). |
| ZNE-R | zne | tensored | 0.0 | ZNE, global folding at λ = 1, 3, 5, Richardson (quadratic) extrapolation, tensored readout correction; supplied-model allocation. |
| ZNE-E | zne | tensored | 0.0 | ZNE, global folding at λ = 1, 3, 5, exponential fit with a declared linear fallback when the fit fails (counted per trial); tensored readout correction; supplied-model allocation. |
| PEC | pec | tensored | 0.0 | Probabilistic error cancellation of the device's median-rate depolarizing model (from the calibration record), one shot per sampled circuit (exact signed law), tensored readout correction. |
| CDR | learning | tensored | 0.25 | Clifford data regression: 16 near-Clifford training circuits with classically computed ideal values, OLS fit, 25 % of B on training (equal per circuit); fallback decided from each trial's own training data; tensored readout correction. |
| NIL | nil | tensored | 0.5 | Neighbor-informed learning (Chen et al.): weight-1 Pauli neighbors at every position, 2-design training ensemble, ℓ1-bounded (≤ 2) regression without intercept, equal target split; 50 % of B on training; tensored readout correction. |

## Features

Class: available (free), acquired (charged), simulated (supplied-model simulation; exponential in width), oracle (never an input).

| feature | class | group | definition |
|---|---|---|---|
| `n` | available | structure | qubits of the logical circuit |
| `two_qubit_logical` | available | structure | two-qubit gates of the logical circuit as written |
| `non_clifford` | available | structure | non-Clifford gates of the logical circuit as written |
| `two_qubit` | available | structure | two-qubit gates of the compiled circuit |
| `depth` | available | structure | ASAP layers of the compiled circuit (virtual RZ included) |
| `pulses_1q` | available | structure | non-virtual one-qubit gates of the compiled circuit |
| `non_clifford_slots` | available | structure | compiled RZ angles that are not multiples of π/2 (the slots CDR/NIL training replaces) |
| `idle_fraction` | available | structure | share of (qubit, layer) slots after a qubit's first gate without a gate |
| `duration_us` | available | structure | scheduled duration of the compiled circuit (ALAP, µs) |
| `obs_weight` | available | observable | largest Pauli weight of the observable's Z-string terms |
| `obs_terms` | available | observable | non-identity terms of the observable |
| `obs_support` | available | observable | classical bits any term acts on |
| `lc_qubits` | available | lightcone | qubits in the backward structural light cone (see module text) |
| `lc_gates` | available | lightcone | compiled gates inside the light cone |
| `lc_2q` | available | lightcone | compiled two-qubit gates inside the light cone |
| `lc_locations` | available | lightcone | fault locations with p > 0 inside the light cone |
| `lam_lc` | available | noise | Σ Pauli-twirled fault probability inside the light cone (snapshot model) |
| `lam_lc_exact` | available | noise | −Σ ln(1 − p) over the same locations |
| `lam_lc_gate` | available | noise | lam_lc from gate errors |
| `lam_lc_idle` | available | noise | lam_lc from idle windows (relaxation, dephasing) |
| `lam_total` | available | noise | Σ fault probability over every location of the circuit |
| `lc_share` | available | noise | lam_lc / lam_total |
| `lam_ro` | available | noise | Σ over the measured support of the mean readout flip probability |
| `log10_budget` | available | budget | log10 of the total shot budget B |
| `pec_log_gamma` | available | protocol | ln γ of PEC's quasi-probability decomposition of the snapshot's median-rate model (readout γ included): its standard deviation grows as γ |
| `zne_eta_l1_linear` | available | protocol | Σ|η_j| of linear extrapolation at the achieved fold λ (gate counts): ZNE-L's noise amplification of the per-scale means |
| `zne_eta_l1_richardson` | available | protocol | Σ|η_j| of Richardson extrapolation at λ = 1, 3, 5 |
| `log10_v_raw_sup` | simulated | protocol | log10 per-shot variance of the readout-corrected estimator under the supplied model |
| `log10_v_zne_sup` | simulated | protocol | log10 per-shot variance of ZNE-R under the supplied model (optimal split) |
| `log10_v_pec_sup` | simulated | protocol | log10 per-shot variance of PEC under the supplied model |
| `abs_signal_sup` | simulated | protocol | |⟨O⟩| of the noisy circuit under the supplied model (raw, no readout error) |
| `entropy_half` | simulated | state | half-cut entanglement entropy (bits) of the ideal final state |
| `output_entropy_bits` | simulated | state | Shannon entropy of the ideal output distribution |
| `ideal` | oracle | oracle | exact ideal value μ |
| `raw_bias_true` | oracle | oracle | exact bias of the unmitigated estimator under the execution law |

Feature sets: `size_budget` (4); `structure` (13); `lightcone` (9); `deployable` (27); `deployable+simulated` (33).
