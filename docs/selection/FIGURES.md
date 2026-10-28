# Figures and tables — protocol-selection study (1.7.0 specifications; 1.8.0 additions)

The six main figures and nine supplementary figures below keep their 1.7.0 panel specifications in 1.8.0; the 1.8.0
additions (four supplementary figures, the extended runtime figure, the thesis tables) are described in
[1.8.0 additions](#180-additions) at the end. Every figure and table of the study is drawn by
`python -m qemx.selection figures CONFIG --out RUN` from the saved tables of a run directory only: nothing is simulated, fitted or relabeled while plotting. The figure code is
`qemx/selection/plots.py` (the six main figures, the supplementary figures, the tables) and
`qemx/selection/figures.py` (palettes, saving convention, certificates figure).

Per figure the command writes

| file | content |
|---|---|
| `<name>.pdf`, `<name>.svg` | vector figure (fonts embedded as Type 42 in PDF; text kept as text in SVG) |
| `<name>.png` | 300 dpi raster |
| `<name>__<key>.csv` | source data: exactly the numbers plotted, one CSV per panel or layer |
| `<name>.md` | the question, the caption, the interpretation note, the source-data files, the plotting settings used, the **methods box** (below), the input tables with the columns read, the command, and the plotting configuration's version and hash |

Figures whose input tables are missing are not drawn; they are listed with the missing files in
`figures/PENDING.md` (with any figure that failed). A complete run leaves no `PENDING.md`.

## The methods box of every figure (1.7.0)

Each figure's notes carry a methods box (`plots.FIG_METHODS`, counts filled from the run's records by
`plots.run_counts`) stating: what was varied and what was held fixed; units, budget definition and estimator; the
independent counts (tasks, base circuits, designs per half, trials per half); the uncertainty construction and its
scope (pointwise or simultaneous, over which unit); simulated or hardware observations; confirmatory, exploratory or
descriptive status; how infeasible, failed, unresolved and missing cases are shown; and the log-floor and clipping
conventions.

## Conventions shared by all figures

* **Unit of analysis.** A test row is one task (base circuit × input × observable × layout × device) at one total
  budget B. Every aggregate weighs each independent **base circuit** equally (its variants are averaged first);
  intervals are 95 % percentile intervals of a bootstrap over base circuits (1 000 resamples unless stated).
* **Risk.** The evaluation-half MSE of the protocol a policy chose (observables normalized to ‖O‖∞ ≤ 1; failures and
  fallbacks included; nonresponse scored as estimate 0). log10 values use log10(MSE + 10⁻⁶).
* **Regret.** The chosen protocol's evaluation-half MSE minus that of the protocol identified on the independent
  selection half (absolute regret, MSE units), or the log10 ratio of the two (log10 regret). The reference is an
  estimate, so regret can be negative; negative values are kept, never clipped, and their share is reported. Signed
  regret is drawn on a zero-centred diverging scale (Figure 2, 1.7.0).
* **Budgets.** Total shot budget B with every calibration, training, diagnostic and execution shot charged. Budgets
  are never merged in a panel that is stratified by budget; a budget that is not in the run's grid is reported as
  such and left empty.
* **Held-out evaluation.** Development-set figures use out-of-fold rows (group 5-fold over base circuits, or the
  named split; nested for the abstention threshold); the confirmation set appears in Table 3, the historical 1.6.0
  final test in Table 3h.
* **Learning designs.** Standard errors of CDR and NIL labels are over their fresh training designs (cluster-robust);
  1.6.0's fixed-shared designs are not mixed with fresh ones in any figure.
* **Typography.** No text below 6 pt at the printed size (7.4 in wide; tick labels 7 pt, axis labels and titles
  7–8 pt); legends placed away from the data or outside the axes; model labels shortened with the full configuration
  names in the notes. Every figure of the release was viewed after drawing.
* **Files.** Every SVG and PDF is parsed after drawing (XML parse; PDF page count and extracted text) and the plotted
  numbers are compared with the tables (`scripts/validate_figures.py`). PDFs are not claimed to be byte-identical
  across environments (font subsetting and library versions differ); within one environment the PNG, SVG and CSV
  outputs are compared.
* **Labels of data kind.** Every figure carries "Simulated observations (emulated fake-device trials); no hardware
  data." Model predictions are labelled as such in the axis titles.

## Palettes

Fixed by role, never by rank; validated with `figures.palette_report` (smallest pairwise OKLab ΔE × 100 under normal
vision and simulated protan/deutan/tritan vision, Machado et al. 2009), and kept by a test.

* **Protocols** (categorical, Okabe–Ito based; same colour in every figure): RAW #C8C8C8, RO #E69F00,
  ZNE-L #56B4E9, ZNE-R #0072B2, ZNE-E #003F63, PEC #D55E00, CDR #009E73, NIL #CC79A7, each with its own marker.
  Smallest ΔE: 15.3 normal, 9.6 protan, 7.6 deutan (CDR–NIL; always with distinct markers), 8.5 tritan.
* **1.8.0 candidates** (composite encoding: colour **and** a unique marker; names on the axes wherever possible):
  RO-TREX #7B2004, RO-TW #96BC79, RO-DD #4C6339, ZNE-L-TW #A974F7, ZNE-E-B #6325A5, PEC-TW #BA1D2A, FR #4C323A,
  SV #8F4D70, ZNE-L∘SV #B1389B, from a max–min search over OKLCh (lightness 0.35–0.79) given the eight colours above,
  which are unchanged. Seventeen colours cannot keep the ≥ 15 normal-vision separation: over all 17 the smallest
  ΔE × 100 is 10.4 normal, 7.1 protan, 6.4 deutan, 7.0 tritan, hence the markers; the eight 1.6/1.7 colours keep
  their own check (≥ 15 normal). The first 1.8.0 assignment (family hues with a lightness step) failed the check
  (4.1, ZNE-R against ZNE-E-B) and was replaced before any 1.8.0 figure was drawn. In the regime maps, whose cells
  are coloured by candidate, the legend lists only the candidates that occur.
* **Policies in Figure 1**: RAW gray dotted, SBS orange dashed, SBS-B vermilion dashed, linear recommender sky
  blue, nonlinear recommender blue, nonlinear + abstention green dash-dot (ΔE ≥ 15.3 normal, ≥ 8.5 under every
  simulated deficiency); line style and marker differ for every policy.
* **Budgets in Figure 6**: three steps of one blue ramp (light → dark = low → high budget).
* **Sequential** (regret, support): one blue ramp #F7FBFF → #08306B. **Diverging** (signed differences): #B2182B ↔
  #2166AC around the gray #E6E6E6.

## Plotting configuration

`configs/selection/figures.yaml` (schema `qemx.selection.figures/1`, versioned) fixes every plotting choice before the
figures are drawn: the policy drawn for each role in Figure 1 (`frozen` = the configuration in `freeze.json`), the
three budgets of the regret profiles, the regime slices, the observable and noise panels of Figure 3, the rows and
columns of the generalization matrix, the mismatch facets with their physical strengths, the MSE targets, the
bias–variance protocols, and the budget of the representative outcomes. An alternative file is passed with
`--plot-config`; its version and hash are written into each figure's notes.

## Main figures

| figure | question | panels | inputs (run directory) |
|---|---|---|---|
| **1** `fig01_budget_efficiency` | Does recommendation improve budget efficiency over raw execution and fixed protocols? | **A** budget–risk curves: base-circuit-weighted arithmetic MSE (log y) against B for RAW, SBS, SBS-B, Ridge, the frozen recommender and its abstaining variant from the nested evaluation, bootstrap bands, matched held-out rows. **B** regret performance profiles at three preselected budgets. **C** paired benefit by family against SBS-B: log10 MSE difference and, beside it, the absolute MSE difference (symmetric-log axis). **D** catastrophic-error contributions: share of the total squared error from the worst 1 % and 5 % of rows and the rate of severe misselection (≥ 10× the reference) per policy. | `analysis/choices_same-distribution.csv`, `analysis/labels.csv`, `analysis/nested_rows.csv`, `analysis/family_benefit.csv`, `analysis/failure_profile.csv`, `freeze.json` |
| **2** `fig02_performance_regimes` | Where should each protocol be used along budget and noise strength, and does the recommender follow it? | Per slice (ising-4, hea-4, qaoa-4; decision D23) matched panels over total budget × device error scale s: **reference preference** (lowest MSE over both halves; hatched = unresolved, the runner-up within 1.96 design-aware SE of the difference — pointwise, not simultaneous), **recommendation** of the frozen recommender (ring = abstained to SBS-B), **signed empirical regret** of the deployed policy on a zero-centred diverging scale (negative values kept), **support / feasibility** (feasible candidates; marks for infeasible candidates and for cells below `regime.min_trials`). Descriptive empirical maps, not phase diagrams or simultaneous guarantees. | `regimes/analysis/regime_cells.csv` |
| **3** `fig03_feature_mechanisms` | Which information changes the decision? | **A** retrained, retuned ablations: feature groups and semantic removals (all light-cone-derived quantities, all simulated quantities; the calibration-derived removal equals the noise group), Δ mean log10 regret with base-circuit intervals and the number of base circuits. **B** controlled observable dependence (one task, four observables). **C** controlled overall noise strength s. **D** controlled noise composition (`noisecomp17`): three circuits × one component changed at a time; the measured best per cell with λ_total, the model's choice and the resolution of the top two. | `analysis/ablation.csv`, `regimes/analysis/regime_protocols.csv`, `regimes/analysis/regime_features.csv`, `noisecomp/analysis/regime_cells.csv` |
| **4** `fig04_generalization_learning` | Does the model generalize beyond its training distribution, and how much data does it need? | **A** generalization matrix (same distribution, family, device, width) with columns SBS-B, SIZE-B, Ridge, the structured risk model and boosted trees; interval and independent test base circuits per cell. **B** learning curve against independent training base circuits. **C** budget extrapolation, one facet per training range (full, ≤ 3·10⁵, ≤ 10⁴): SBS-B, Ridge, the structured model and boosted trees; filled = interpolation, open on grey = extrapolation. | `analysis/pooled_<split>.csv`, `analysis/learning_curve.csv`, `analysis/budget_generalization.csv` |
| **5** `fig05_reliability_abstention` | Can the system recognize unreliable recommendations (nested evaluation)? | **A** coverage against the accepted rows' regret with random rejection and each outer fold's operating point. **B** whole-policy arithmetic MSE after the fallback against coverage, with random rejection at matched coverage. **C** the nested operating point against the model alone, SBS-B, random rejection (with its draw range), SIZE-B and SBS. **D** measured coverage of the log-risk intervals, each labelled by the statement it supports (split-conformal, CV residual, Gaussian). | `analysis/nested_curve.csv`, `analysis/nested_summary.csv`, `analysis/nested_rows.csv`, `analysis/nested_folds.csv`, `analysis/calibration.csv`, `analysis/interval_coverage.csv` |
| **6** `fig06_mismatch_diagnostics` | Does model mismatch predict recommendation failure, and does reacting to a diagnostic pay for its shots? | **A** controlled mismatch response of the recommender's regret (physical units, zero-mismatch control). **B** a single deployment's echo statistic |z| against the subsequent regret (simulated deployments, one observation each), with its discrimination (AUC) per budget. **C** equal-budget gains and losses against TARGET@B: no diagnostic (B − d), fixed reactions (ZNE-L, SBS-B), the learned reaction evaluated on held-out discrepancy types, and random reaction at the matched flag rate. | `mismatch/diagnostics.csv`, `mismatch/policies.csv`, `mismatch/analysis/deployment_draws.csv`, `mismatch/analysis/discrimination.csv`, `mismatch/analysis/reactions.csv` |

## Supplementary figures

| figure | question | content | inputs |
|---|---|---|---|
| `supp_target_precision` | What budget does a protocol need for an MSE target? | smallest evaluated budget meeting the target by the point estimate and by the approximate pointwise upper limit MSE + 1.96 SE (design-aware); not a minimum-budget guarantee; unattained targets censored | `analysis/labels.csv` |
| `supp_bias_variance` | Which error is a floor? | squared bias corrected as ē² − SE(ē)² with a design-aware SE; variance of one estimate and, for CDR and NIL, its between-design component | `regimes/tasks/*`, `regimes/analysis/regime_cells.csv` |
| `supp_prediction_quality` | How good are the held-out risk predictions where it matters? | predicted against evaluated log10 MSE per candidate; and the predicted against evaluated difference between the predicted best and runner-up (order right / resolved) | `analysis/prediction.csv` |
| `supp_dataset_coverage` | Which feature space and noise composition is covered, by how many independent circuits? | development and final-test feature distributions, the oracle |μ| separately, noise components (gate vs idle share of λ_LC against readout weight, with the noise-composition settings), counts of tasks, failed tasks, base circuits, designs and trials | `dev/tasks`, `final/tasks`, `noisecomp/tasks` |
| `supp_coverage_certificates` | Do intervals and certificates hold, and what do they guarantee? | measured interval coverage by statement kind; finite-sample bounds on the clipped estimator with their units; the 2ε relation as a post-hoc check | `analysis/interval_coverage.csv`, `analysis/certificates.csv` |
| `supp_failures_costs` | How often do protocols fail, and where do shots go? | infeasibility and method errors, union failure rate of trials (no-estimate part separate), software failures of label tasks listed apart, shot ledger at B = 10⁵ | `analysis/labels.csv`, `analysis/label_failures.csv` |
| `supp_representative` | What do outcomes look like trial by trial? | four rows chosen by fixed rules; CDR and NIL trials drawn by training design | `analysis/deployed_rows.csv`, `analysis/nested_rows.csv`, `dev/tasks/<id>` |
| `supp_runtime` | What does one recommendation cost classically? | compilation, deployable features, simulator-only augmentation, inference, and benchmark-label generation for comparison, against width | `analysis/runtime.csv` |
| `supp_design_variability` (new) | How much do CDR's and NIL's errors depend on the training design? | design-level loss distributions relative to the task's risk; between-design share of the label variance; the design study's alternative training allocations | `analysis/design_diagnostics.csv`, `analysis/design_variance.csv`, `design_study/designs.csv` |

## Tables (`<run>/tables/`, CSV and Markdown)

| table | content | inputs |
|---|---|---|
| `table1_design` | per family: widths, compiled depth range, observables, devices, **tasks and base circuits counted separately** for development, historical final and confirmation, budget grid, trials per label, learning designs, split rule | `manifest.json`, `<part>/tasks` |
| `table1b_studies` | every study of the ancestry registry: role (development, historical-final, exploration, confirmation, planned), logical circuits, executed or not, tasks and base circuits labeled, failed tasks | `configs/selection/ancestry_registry.json`, run records |
| `table2_protocols` | the protocol configurations compared (estimator, readout, settings, noise information, training/calibration cost, allocation, failure policy, approximation status) — "the configurations compared, not every implementation of the method" | `PORTFOLIO`, `plots.PROTOCOL_INFO` |
| `table3_policy_results` | the **confirmation set** (confirmatory, evaluated once): per policy, base circuits and rows, mean log10 MSE, arithmetic MSE, mean log10 regret, median and 90 % regret, 90 % log10 regret, share of regret > 2×, share within 10 %, share worse than RAW, paired difference to SBS-B, coverage, union failure share; base-circuit weighted, 95 % intervals | `confirm/metrics_confirm.csv` |
| `table3h_historical_final` | the same columns for the historical 1.6.0 final test re-evaluated with the frozen 1.7.0 configuration (not confirmatory) | `final/metrics_final.csv` |
| `table_s1_rejection` | rejection rates of the nested threshold by family and device with base-circuit counts | `analysis/nested_groups.csv` (or the 1.6.0 `selective_groups.csv`, labelled) |

## Regenerating and checking

```bash
python -m qemx.selection figures configs/selection/pilot17.yaml --out reports/selection_v170
python scripts/validate_figures.py reports/selection_v170
```

`reports/selection_v170/` holds the analysis tables and the label records of the corrected pilot, so the commands
regenerate every figure and table without the per-trial arrays except `supp_representative` and `supp_bias_variance`,
which read individual trials (they are listed in `PENDING.md` when the arrays are absent and the shipped versions stay;
the arrays are in the trial-data archive, `reports/selection_v170/README.md`). One figure at a time:
`--only NAME[,NAME…]`.
The 1.6.0 figures are historical material: they stay in the repository under `reports/selection_pilot/figures/` and in the 1.6.0 release archive, and are omitted from the 1.7.0 source archive (size); drawing them with the 1.7.0 code is not supported, because several 1.7.0 figures need tables that the 1.6.0 analysis did not produce.

Consistency tests (`tests/test_selection_eval.py`, `tests/test_selection_cli.py`, `tests/test_selection_v170.py`) and
the validator check that the plotted numbers equal those recomputed from the tables, that every SVG parses and every
PDF has its pages and text, and that the palettes keep their colour-vision separations.

## 1.8.0 additions

The 1.8.0 study (`reports/selection_v180/`, run `runs/v18`) is drawn with the same code and the plotting
configuration `configs/selection/figures18.yaml` (version 3: only the configuration names follow the 1.8.0 feature
sets `deployable18` and `deployable18+simulated`, and the bias–variance panel adds RO-TW and ZNE-E-B). The 1.7.0
configuration `configs/selection/figures.yaml` stays the configuration of `reports/selection_v170/`.

```bash
python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 \
    --plot-config configs/selection/figures18.yaml
python -m qemx.selection.thesis_tables --run reports/selection_v180 --out reports/selection_v180/thesis_tables
python scripts/validate_figures.py reports/selection_v180
```

### New supplementary figures (`qemx/selection/plots18.py`)

| figure | question | content | inputs |
|---|---|---|---|
| `supp_deployment_policy` | What does the deployed application do on new circuits, and how does its full-policy result compare with the model component? | **A** outcome of every `deploy.recommend` call on the confirmation set, per bundle: top-ranked candidate accepted, accepted after the preflight skipped unverified/infeasible candidates, SBS-B fallback (margin below τ, out of distribution, no prediction), unsupported input; rows, base circuits and median classical seconds per call. **B** paired difference to SBS-B (mean over base circuits of log10(MSE + 10⁻⁶), 95 % bootstrap over base circuits) of the full policy of each bundle beside the model component of the same configuration and the fixed baselines. **C** per-row log10 risk ratio to SBS-B by outcome. **D** candidates chosen by the full policy and the model component. **E** share of the final candidate's evaluation trials with a method fallback or failure flag, by outcome | `replay/replay_<bundle>.csv/.json`, `replay/replay_summary.json`, `confirm/choices_confirm.csv`, `confirm/labels.csv`, `confirm/metrics_confirm.csv` |
| `supp_portfolio_expansion` | Do more candidates help, and is the benefit from the candidates or from learning to select them? | **A** paired contrasts (out-of-fold development): SBS-B and the selection-half reference with 17 against 8 candidates; learned selection (Ridge, HGB) against SBS-B within each portfolio; the total. **B** slope chart of each policy's mean with the two portfolios. **C** share of rows on which each 1.8 policy chose each candidate. **D** severe degradation against RAW per policy and portfolio | `portfolio/portfolio_contrasts.csv`, `portfolio/portfolio_policies.csv`, `portfolio/portfolio_comparison.json` |
| `supp_root_audit` | Are the labels, the rankings and the execution contract stable across independent preparation roots? | **A** between-root variance of each label over its mean squared label SE (1 = honest SE), per candidate and budget. **B** distinct winners of a (task, budget) over four roots, and the log regret of one root's winner under another root's labels. **C** agreement of the design-level preflight with each root's allocation | `audit18/analysis/label_stability.csv`, `ranking_stability.csv`, `contract.csv`, `audit_summary.json` |
| `supp_tails` | How heavy are each candidate's loss tails, and do single training designs carry them? | **A** median, 90 % quantile and maximum of log10(R_k/R_RAW) per candidate at three budgets. **B** base-circuit-weighted rate of severe degradation (R_k > 10 R_RAW; declared before the 1.8.0 results) per candidate and budget ("–" = not feasible). **C** share of a candidate's loss carried by its worst 5 % of base circuits. **D** the largest single training design's share of a learning label's summed design loss (dotted: 1/K) | `analysis/labels.csv`, `analysis/design_diagnostics.csv` |

Each candidate in `supp_tails` and in the thesis table of tails is shown on its **own feasible subset**: these are
descriptions of a candidate's tail, not a ranking (rankings use paired comparisons on common rows).

`supp_runtime` (1.7.0 figure) gains, when `analysis/runtime.csv` has the 1.8.0 columns, the structural preflight of
every candidate, the design-level preflight of the learning candidates and the whole `recommend()` call (panel A),
and a panel B with the early-rejection probes (GHZ-type chains of 12, 24 and 40 qubits outside every training
width: seconds of the whole call, status, peak memory). The features are timed without the 1.8.0 feature cache.

Outcome colours (`plots18.OUTCOMES`, fixed order, never by rank) are Okabe–Ito hues and a dark neutral; smallest
pairwise OKLab ΔE × 100: 15.6 normal, 9.6 protan and deutan, 8.5 tritan (`tests/test_plots18.py`). Candidates keep
the protocol palette; the nine 1.8.0 candidates have the colours chosen by the max–min search described under
"Palettes" above, always with their own marker, and are marked `*` in candidate axes.

### Legibility with 17 candidates (1.8.0 review of every figure)

Every figure of `reports/selection_v180/` was inspected after drawing; the specifications are unchanged, the layouts
adapt to the larger portfolio:

* Figure 2 counts the candidates of each slice from its labels and marks where a candidate of the slice is
  infeasible; Figure 3 B puts the protocol legend outside the axes when there are more than eight candidates, and the
  text of hatched (unresolved) cells in panel D sits on a light box; Figure 4 C styles the budget-extrapolation lines
  by model family, so the 1.7.0 and 1.8.0 configuration names draw alike; Figure 5 C centres the MSE annotations.
* `supp_prediction_quality` draws a facet for **every** candidate (6 columns for 17; 1.7.0 kept 4 × 2) with shared
  axes; `supp_target_precision` grows with the number of candidates and writes one "not reached (point | upper)" line
  when neither bound is reached; `supp_representative` sets its axis to the candidates whose 99th-percentile error is
  within 10× the panel's median (and the choice) and counts the trials beyond it with arrows;
  `supp_design_variability` includes FR (1.8.0's third learning candidate; `analysis/design_variance.csv` now covers
  every learning candidate); `supp_coverage_certificates`, `supp_runtime` and `supp_root_audit` keep their labels inside
  the figure.
* `supp_deployment_policy` D shows the model component's choices also for a bundle whose configuration was not a
  confirmation comparator (computed from the bundle's own predictions over the label-feasible candidates, as the
  replay scores it); its interpretation counts the rows where the component's candidate was left unverified by the
  deployed preflight.

### Thesis tables (`qemx/selection/thesis_tables.py`)

One generator writes, for each table, the numbers (`<table>.csv`), an includeable LaTeX fragment (`<table>.tex`,
`booktabs`, label `tab:qem180-*`), the same strings in Markdown (`thesis_tables.md`), a standalone wrapper
(`qem180_thesis_tables.tex`, compiled to `qem180_thesis_tables.pdf` when `pdflatex` exists) and `provenance.json`
(generator version, the source tables with their sha256). Tables: executed studies; candidates with feasible
denominators and the shot ledger; the once-only confirmation with the full deployed policy of both bundles beside the
model components; development out-of-fold; representative per-budget MSEs with feasible denominators (descriptive);
tails, fallbacks and failures; classical costs and early rejection; original against expanded portfolio. A table whose
source is missing is not written and is listed under "Not generated".

### Validation

`scripts/validate_figures.py` additionally checks, when present: the full-policy differences of
`supp_deployment_policy` against `replay/replay_<bundle>.json`; the contrasts of `supp_portfolio_expansion` against
`portfolio/portfolio_contrasts.csv`; the severe-degradation rates of `supp_tails` recomputed from
`analysis/labels.csv`; thesis table 3 against `confirm/metrics_confirm.csv` and table 7 against the portfolio
contrasts.

Figure 6 (mismatch diagnostics) needs a mismatch study of the run; 1.8.0 did not repeat the 1.7.0 mismatch study
(its question and code path are unchanged; see `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md`), so in
`reports/selection_v180/` it is listed in `figures/PENDING.md` and the 1.7.0 figure in `reports/selection_v170/figures/`
remains the evidence, with the 1.6/1.7 portfolio.
