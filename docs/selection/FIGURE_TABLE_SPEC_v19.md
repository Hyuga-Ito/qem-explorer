# Figure and table specification of the next research stage (1.9.0)

Human-readable form of `configs/selection/figure_spec19.yaml` (version 19.1; the YAML is authoritative and is checked by
`tests/test_tables19.py`: every generator exists, every status is one of confirmatory / exploratory / descriptive /
pending, Figure 6 is pending). A figure or table that is not listed here is not part of the thesis contract.

**Metric.** Δ = mean over base circuits g of the mean over g's rows (task variants × budgets of the panel) of log10(R_policy + 1e-6) − log10(R_SBS-B + 1e-6); R = fresh-design average risk on the evaluation half unless the panel says "executed" (risk conditional on the returned verified design); 10^Δ is a factor on the geometric mean of R + 1e-6, not on the MSE. Intervals: 95 % percentile bootstrap over base circuits (1000 resamples, seed 0), the circuit's variants, budgets and roots resampled together.

**Evidence status.** *confirmatory*: a once-only evaluation of a design declared before its labels (only the 1.8.0
confirmation qualifies today); *exploratory*: development, pilot or inspected data — hypothesis-generating;
*descriptive*: counts, costs and definitions; *pending*: specified, no data yet. A panel never upgrades its status
because its interval excludes zero.

## F1 — Actual policy benefit

**F1A.** Does the full deployed policy beat SBS-B, overall and per budget?

* source: reports/selection_v180/replay/replay_<bundle>.csv + confirm labels (arrays)
* x: total shot budget B (log)
* y: Δ vs SBS-B
* unit: base circuit
* uncertainty: bootstrap over base circuits
* comparison: primary (D41) and secondary bundle
* costs: classical seconds per call in s6_costs
* status: confirmatory (1.8.0 once-only confirmation; per-budget rows descriptive)
* generator: qemx.selection.tables19.figure1
* data: available

**F1B.** Is the benefit visible in arithmetic MSE too?

* source: same as F1A
* x: budget
* y: arithmetic MSE (log)
* unit: base circuit
* uncertainty: none drawn (descriptive)
* comparison: policy vs SBS-B
* status: descriptive
* generator: qemx.selection.tables19.figure1
* data: available

**F1C.** Does scoring the returned verified design instead of the fresh-design label change the contrast?

* source: reports/execution_audit_v190/execution_audit_summary.json
* x: Δ vs SBS-B
* y: policy
* unit: base circuit (8)
* uncertainty: bootstrap over base circuits
* comparison: executed vs fresh-design target
* status: exploratory (inspected circuits)
* generator: qemx.selection.tables19.figure1
* data: available

**F1D.** Is the gain from more candidates or from learned selection? (fixed 8 vs 17 on the same tasks)

* source: <run>/portfolio/portfolio_contrasts.csv
* x: Δ
* y: contrast
* unit: base circuit
* uncertainty: bootstrap
* comparison: expansion with a fixed policy; learning within each portfolio
* status: exploratory (pilot19; fulldev19 when run)
* generator: qemx.selection.tables19.figure1
* data: available for the pilot

## F2 — Performance regimes

**F2A.** Which candidate is preferred across budget × declared noise strength/composition?

* source: <run>/regimes/analysis/regime_cells.csv
* x: budget
* y: noise setting
* facets: a few circuits/observables
* unit: task cell
* uncertainty: unresolved when the top two lie within 1.96 SE
* comparison: empirical reference vs recommendation; regret
* status: exploratory (1.8.0 regime study exists; a noise-diverse version needs fulldev19's regime companion)
* rule: no sharp phase boundaries drawn from noisy winners; infeasible regions shown
* generator: qemx.selection.plots.fig02_performance_regimes
* data: 1.8.0 only (native-strength training)

## F3 — Information value

**F3A.** Which grouped features (retuned) change policy risk, at what acquisition cost?

* source: <run>/analysis/ablation.csv (retuned per group)
* x: Δ when the group is removed
* y: feature group
* unit: base circuit
* uncertainty: paired bootstrap
* comparison: deployable vs simulator-only vs paid diagnostics (equal total budget)
* costs: seconds and shots of acquisition
* status: pending for 1.9 (fulldev19 extras: ablation)
* rule: importance is predictive, correlated features complicate attribution; not a causal law
* generator: qemx.selection.plots.fig03_feature_mechanisms
* data: 1.8.0 only

## F4 — Generalization and sample efficiency

**F4A.** Does the policy transfer to unseen circuits, devices, noise compositions and joint shifts?

* source: <run>/analysis/pooled_{same-distribution,noise,device-base,joint}.csv
* x: split
* y: Δ vs SBS-B
* unit: base circuit
* uncertainty: bootstrap
* status: exploratory in the pilot (s5_pilot); fulldev19 planned
* generator: qemx.selection.tables19.s5_pilot
* data: available for the pilot (table)

**F4B.** How does performance scale with independent circuits and design roots?

* source: <run>/analysis/learning_curve.csv
* x: independent base circuits
* y: Δ vs SBS-B
* unit: fold
* uncertainty: spread over repeats
* status: exploratory
* generator: qemx.selection.plots.fig04_generalization_learning
* data: pilot (2 repeats) and 1.8.0

**F4C.** Runtime and memory of a recommendation against width

* source: <run>/analysis/runtime.csv
* x: width
* y: seconds / MB
* unit: call
* status: descriptive
* generator: qemx.selection.plots.supp_runtime
* data: pilot profile, 1.8.0

## F5 — Reliability

**F5A.** Coverage against whole-policy risk/regret, vs random rejection at matched coverage

* source: <run>/analysis/nested_curve.csv, nested_summary.csv
* unit: base circuit
* uncertainty: nested CV + bootstrap
* status: exploratory (pilot, 1.8.0)
* rule: invalid or vacuous certificates stay visible; empirical intervals labelled empirical
* generator: qemx.selection.plots.fig05_reliability_abstention

## F6 — Discrepancy usefulness

**F6A.** False-positive rate and power of a diagnostic vs its budget; does a discrepancy predict a ranking reversal/regret; net equal-budget benefit of reactions?

* source: <run>/mismatch/policies.csv, diagnostics.csv (17-candidate study)
* unit: base circuit
* uncertainty: bootstrap; exact flag probabilities
* status: PENDING — the 17-candidate mismatch study is not executed; the 1.7.0 eight-candidate result is not evidence for it
* generator: qemx.selection.plots.fig06_mismatch_diagnostics
* data: none for 1.9

## Tables

| id | question | source | status | generator |
|---|---|---|---|---|
| s1_datasets | dataset and split counts | configs/selection/*.yaml (expansion) | descriptive | `qemx.selection.tables19.s1_datasets` |
| s2_candidates | candidate applicability and resource definitions | qemx/selection/registry.py | descriptive | `qemx.selection.tables19.s2_candidates` |
| s3_primary | primary policy comparison by budget | replay rows + derived labels | confirmatory (1.8.0) | `qemx.selection.tables19.s3_primary` |
| s4_execution | executed vs fresh-design risk | execution_audit_summary.json | exploratory | `qemx.selection.tables19.s4_execution` |
| s5_pilot | pilot transfer splits and portfolio | pilot analysis/pooled_*.csv, portfolio_contrasts.csv | exploratory | `qemx.selection.tables19.s5_pilot` |
| s6_costs | measured labeling and recommendation costs | pilot task records, replay summary | descriptive | `qemx.selection.tables19.s6_costs` |

Build: `python -m qemx.selection.tables19` (defaults: `reports/selection_v180`, `reports/execution_audit_v190`,
`reports/pilot_v190` → `reports/tables_v190/`; CSV, LaTeX, Markdown, `tables_v19.tex` and, with pdflatex, a PDF).
Optional sources that are absent are skipped and listed, never filled with placeholders.
