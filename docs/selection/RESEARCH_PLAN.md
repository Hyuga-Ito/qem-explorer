# Research plan and decision log — budget-aware QEM protocol selection (1.7.0)

## Plan (1.7.0)

**Central question.** Which information improves QEM selection beyond a strong budget-only baseline (SBS-B), and
when does that improvement fail? The study is not optimized to give a positive answer; "not resolved" is a result.

**Stages executed in 1.7.0** (every count distinguishes total shots per deployment B, trials r, independent learning
designs K and independent base circuits):

1. Smoke workflow (`smoke.yaml`) and the CI matrix.
2. Corrected pilot (`pilot17.yaml`): the 1.6.0 pilot's tasks and seeds relabeled with fresh, independent learning
   designs (K = 10 per half, r = 100 per half, 10 budgets 10³–3·10⁷): 120 development tasks on 40 base circuits and the
   historical final test (36 tasks, 12 base circuits). Same tasks as 1.6.0, so old and new labels can be compared
   directly.
3. Controlled studies: the regime study relabeled (`regimes17.yaml`), the noise-composition study (`noisecomp17.yaml`,
   new) and the mismatch study at equal total budget (`mismatch17.yaml`).
4. Focused experiment B (`scripts/design_study.py`): learning-design variability and alternative training
   allocations of CDR and NIL on the review's severe-NIL tasks.
5. Freeze (`freeze`, then `freeze --complete` with the nested threshold), historical re-evaluation of the 1.6.0 final
   test, and the once-only confirmation on new logical circuits (`confirm17.yaml`, 25 base circuits, 50 tasks).

**Planned, not executed:** `full17.yaml` (242 logical circuits, 726 tasks; `plan` gives ≈ 19.1 CPU-hours from the
corrected pilot's measured rates; the 15.6 first written here came from the labels then available — against caps of
80 CPU-hours and 10 GB; resumable, bound to its run) and the hardware
validation (`hardware17.yaml`; owner authorization required).

**Primary endpoint** (unchanged): mean over held-out base circuits of log10 of the evaluation-half MSE of the chosen
protocol, paired difference to SBS-B with a base-circuit bootstrap interval. Secondary: arithmetic policy MSE
(base-circuit weighted), regret tails, severe misselection (≥ 10× the reference), the share of total error from the
worst 1 %/5 % of rows, whole-policy MSE after abstention against random rejection at matched coverage.

**What would change the conclusion.** A model policy whose paired interval against SBS-B excludes 0 in its favour on
held-out base circuits *and* on the confirmation set supports the claim within this design; an interval containing 0
is "not resolved". Results on emulated fake devices do not establish hardware performance.

## Plan (1.6.0, historical)

**Claim to test.** At equal, fully charged shot budgets, choosing a complete QEM protocol per task from deployable
descriptors lowers the estimation error relative to fixed choices — and the gain, where it exists, depends on the
budget and survives a shift of family, device, width or budget only partly.

**Primary endpoint.** Mean over held-out base circuits of log10 of the evaluation-half MSE of the chosen protocol, per
budget and over all budgets; paired difference to SBS with a base-circuit bootstrap interval. Secondary: gap closed,
arithmetic policy MSE, regret tails (P(ratio > 1.1), P(ratio > 2)), coverage and accepted-task risk for the abstaining
policy.

**Design.** Families × widths 3–6 × parameters give base circuits; each gets its default variant and up to two
further variants (input, observable, layout, device). Stage 1 (smoke) checks the pipeline; stage 2 (pilot, this
release) labels the development set with the full budget grid and r = 100 + 100 trials; the final test (30 % of base
circuits, chosen by a keyed hash before labeling) is labeled and evaluated once after the development choices are
frozen. Stage 3 (full, not executed) raises r to 500 + 500, adds widths and devices, and refines budgets near the
observed ranking reversals (boundary workflow). The mismatch study re-labels a subset under hidden perturbations.

**What would change the conclusion.** A learned policy whose paired interval against SBS-B excludes 0 in its favour
on held-out base circuits *and* on the final test supports the claim within this design; an interval containing 0
is "not resolved", not "equivalent". Results on emulated fake devices do not establish hardware performance.

## Decision log

Decisions are dated (2026-10-19) and were made before the pilot's results were seen unless marked otherwise.

| # | decision | reason |
|---|---|---|
| D1 | Candidates are complete protocols (§3 of METHODS), 8 of them, fixed before labeling | "ZNE" is not a candidate; the comparator is a declared set (companion §1.5.3) |
| D2 | Allocations from the *supplied* snapshot model only; PEC inverts the snapshot's median-rate model | deployable information; the truth enters only through the data |
| D3 | Learning protocols spend a declared fraction of B on training (CDR 25 %, NIL 50 %) | a fixed training cost makes them infeasible at most budgets; a budget-dependent rule is part of the protocol |
| D4 | Two independent halves per label; score on the evaluation half; reference identified on the selection half | removes the optimizer's curse from reported regret |
| D5 | Primary aggregate: base-circuit-weighted mean log10 MSE; ε0 = 10⁻⁶ | robust to the few tasks whose MSE dominates an arithmetic mean; ε0 caps differences below an RMSE of 10⁻³ |
| D6 | Gap closed in the ASlib orientation (1 = oracle, 0 = SBS) | stated convention (Lindauer et al. use the opposite) |
| D7 | Final test = 30 % of base circuits by keyed hash (salt `qemx-1.6-final`) | 20 % gave only 5 base circuits on the pilot grid; 30 % gives 12 across 7 families. Changed before any label existed |
| D8 | Nonresponse loss μ² (estimate 0); no trial dropped | companion Eq. 1.7 |
| D9 | Certificates on the clipped variant (H = 4), unclipped retained | squared error of extrapolated estimators is unbounded |
| D10 | No graph neural network unless tabular models leave a resolved gap to the oracle that structure features could explain | complexity must be earned (companion §1.5.3) |
| D11 | The diagnostic is the echo circuit with 2 % of B (≥ 200 shots), robust fallback ZNE-L, α = 0.05 | cheap, matched conventions, a model-free fallback estimator |
| D12 | Pilot r = 100 per half (not 500) | compute: the pilot fits the session; the full workflow states its cost |
| D13 | Faithful input tokens (1.5.1 fix) merged into 1.6.0 | checked first: no compiled circuit of the pilot changes (156 of 156 identical) |
| D14 | Learning protocols redraw their random training design per deployment: 6 designs per task, trials spread round-robin; the first pilot labeling (one fixed design per task, 2026-10-19) was discarded and every task relabeled | found while checking NIL's largest errors (before any model or policy was evaluated): with one design per task, a task's NIL label reflected one random design (bias 0.18 on one task, 0.0001 with another seed) instead of the protocol's risk |
| D15 | NIL keeps 16 training circuits (not the paper's ~10³) | compute; its design variance at this size is part of the result, stated with it |
| D16 | Mid-run fix (2026-10-20): the early-return samplers of CDR and NIL (Clifford circuit; training labels without spread) accept the emulator's keywords | a learning-protocol design that fell back crashed its task (TypeError, two tasks); the fix touches only that path, both tasks were relabeled from scratch; every task that reached the path had crashed, so no stored label was made by it (`dev/provenance.json`, second session) |
| D17 | A learning protocol on a circuit with no non-Clifford slot is infeasible (nothing to train), not silently RO | declared before results; such a deployment would not run the protocol |
| D18 | Split results are reported out-of-fold pooled (one bootstrap over all test base circuits), not as fold-averaged intervals; the paired difference to SBS is the plotted transfer measure, gap closed stays in the tables | decided 2026-10-20 before any pilot analysis existed (after the smoke run only): an average of per-fold interval bounds is not an interval, and the gap closed is a ratio that is unstable where the oracle's margin over SBS is small |
| D19 | The best model configuration's score is reported with its BBC-CV bias-corrected estimate | same date and state; the winner of 20 configurations is chosen on the scores it is reported with |
| D20 | A supplementary figure for coverage and certificates (`supp_coverage_certificates`) | the plan named coverage among the supplementary figures; it was missing from the first figure set |
| D21 | The device and budget splits are crossed with the group 5-fold over base circuits | **found after their first results were computed** (2026-10-20): the first version trained on the other variants and budgets of the tested circuits, which share their ancestry — a violation of the plan's rule that no base circuit is on both sides of a split. Rerun with the fix; the first results are kept in the audit material, not reported |
| D22 | The mismatch study's model is trained on the development set without the mismatch study's base circuits | found before the mismatch analysis was run: 8 of its 10 base circuits are development circuits (in-sample choices otherwise); the other 2 belong to the final test set — the mismatch analysis runs after the freeze, so no development choice can depend on them |
| D23 | Controlled regime study (`configs/selection/regimes.yaml`) for the regime maps and the controlled panels of Figure 3: three new base circuits (Ising 4 qubits 2 steps, HEA 4 qubits 2 layers, QAOA 4 qubits p = 1) on `fake_fez`, device error scale s ∈ {0.5, 1, 1.5, 2, 3} (every calibrated error rate × s) applied to snapshot and execution law alike, all ten budgets, r = 100 + 100; the Ising slice also under four observables and both layouts at s = 1; the recommender is the frozen pilot policy trained without these circuits. Hypotheses: (B) the measured ranking at B = 10⁵ changes with the observable's support; (C) as s grows, PEC's sampling overhead γ² grows exponentially and its preference at B = 10⁵ gives way to extrapolation or learning | declared 2026-10-20 before any of these tasks was labeled, after the user re-sent the figure specification (the pilot's aggregated regime map does not hold a noise parameter fixed) |
| D24 | A family parameter `seed` in a configuration is the task's transpiler seed (`TaskSpec.seed`); the instance seed of random families is `angle_seed` | found 2026-10-20 while designing D23: the executed pilot and mismatch configurations meant `seed` as an instance seed, but it set the transpiler seed, so all their HEA, QAOA and brickwork circuits use instance seed 0. Their base circuits still differ (layers or p), every task records its transpiler seed, and the labels are what the records say; the executed configurations keep their keys (ids unchanged) with a note, the unexecuted ones now use `angle_seed` |
| D25 | Re-analysis of the development labels to the re-sent specification (2026-10-20, from 03:41 UTC): regret metrics (absolute and log10, tails, tolerance success, degradation against RAW, failure share), per-variant retuned ablation, learning curve in regret, budget generalization with restricted training ranges, the frozen configuration's out-of-fold rows with selective recommendation (cross-fitted threshold, random-rejection baseline, whole-policy MSE) and three kinds of interval calibration, paired family benefit against SBS-B. The reliability, family-benefit, regime and representative-outcome analyses use the configuration frozen in `freeze.json` | **post hoc: after the final test was evaluated (02:27)**. Nothing frozen changes: the frozen configuration, the final-test choices and labels are reused as saved; the final-test metrics are recomputed from them (`final/metrics_final.csv`). Development numbers present in both analyses are compared in the audit material (`evidence/reanalysis/`) |
| D26 | Selective recommendation: margin z = (P₂ − P₁)/√(s₁² + s₂²) with s the inner-CV RMSE per protocol; abstain to SBS-B (no shots) below τ; τ chosen per test fold on the other folds' rows by the lowest whole-policy mean log10 risk (cross-fitting), and once on all development rows for later studies (`deploy_tau`) | declared with D25, before its results; the conformal abstention of the first analysis stays reported (it was inert on the pilot: its simultaneous level is unattainable with 32 base circuits per fold) |
| D27 | Figures implemented to the re-sent panel specification with a versioned plotting configuration (`configs/selection/figures.yaml`) fixed before the figures were drawn from the re-analysis | the first figure set aggregated over different noise conditions and lacked several required panels; plotting choices are recorded per figure |
| D28 | Learning designs are sampled fresh and independently: K = 10 designs per label half from separate seed pools, the allocation computed from a separate allocation design, designs paired across budgets within a half; SEs over designs | review R1, declared 2026-10-21 before relabeling; the 1.6.0 labels (6 fixed designs shared by both halves) are kept as a historical dataset with their own estimand |
| D29 | The corrected pilot relabels the 1.6.0 pilot's tasks with the same seeds (only the design plan changes) | declared 2026-10-21; makes the effect of the design correction directly comparable task by task |
| D30 | The 1.6.0 final test is historical (it overlaps earlier studies by logical circuit); a confirmation set of 25 new logical circuits (`confirm17.yaml`) is registered in the ancestry registry and evaluated once after the freeze | review R5, declared 2026-10-21 before any confirmation task was labeled |
| D31 | Selective recommendation evaluated by nested group CV; `deploy_tau` chosen by the same inner procedure on all development rows and frozen (`freeze --complete`) | review R2, declared 2026-10-21 |
| D32 | Arithmetic policy MSE weighted by base circuit; SBS ranked on one population with a declared fallback; failure rates as the union of flags | review S5/S2, declared 2026-10-21 |
| D33 | Model comparison adds the structured risk model b² + v/((1 − f)B) (stage-1 NNLS per task, stage-2 ridge) beside Ridge, elastic net, boosted trees and the MLP; Ridge, boosted trees and the structured model in every split and in budget extrapolation | prompt §8, declared 2026-10-21 before the corrected pilot was analyzed |
| D34 | Semantic ablations (all light-cone-derived, all calibration-derived, all simulated quantities) beside the group ablations | review of Figure 3, declared 2026-10-21 |
| D35 | Noise-composition study `noisecomp17.yaml`: the three regime circuits under one component changed at a time (g2 × 2, × 3; g1 × 3; T1, T2 ÷ 2, ÷ 3; readout × 2, × 0.5; coherent fraction 25 %, 50 %). Hypotheses at B = 10⁵, written 2026-10-22 00:40 UTC before any of its tasks was labeled: (H1) raising two-qubit errors moves the preference away from PEC as raising the overall scale s does; (H2) readout changes move RAW against the readout-corrected protocols but leave the ranking among the readout-corrected protocols unchanged; (H3) a coherent fraction, which the snapshot's Pauli model does not describe, raises the risk of PEC more than that of the extrapolation protocols | prompt §8 C; exploratory (three circuits, one task per cell) |
| D36 | Mismatch reactions compared at equal total budget with TARGET@B, NO-DIAG@B−d, fixed fallbacks, a learned reaction trained without the held-out discrepancy type, and random triggering at the matched flag rate; single-deployment diagnostic draws kept apart from repetition summaries | review of Figure 6 and prompt §8 F, declared 2026-10-21 |
| D37 | Focused experiment B (design study): K = 20 fresh designs per variant (30 were planned; reduced to 20 before the run after timing 3 designs per variant: ≈ 2 CPU-hours at 20) on five tasks (the review's two severe-NIL tasks and three controls) at B = 10⁴, 10⁶, 3·10⁷, variants NIL, NIL-T32, NIL-f25, NIL-f75, NIL-bare, CDR, CDR-T32, CDR-f50; severe = design MSE > 10⁻². Question: is a severe NIL loss a property of the protocol or of particular designs, and does another training allocation change it? The variants are not added to the portfolio | prompt §8 B, declared 2026-10-22 before it was run |
| D38 | Figures: plotting configuration version 2 (noise-composition panel, structured model in Figure 4, Ridge in budget extrapolation, design-variability supplementary figure); every figure carries a methods box | review §7 and prompt §9, declared 2026-10-22 before the corrected pilot's figures were drawn |
| D39 | The once-only evaluations (historical final, confirmation) report the frozen configuration as the primary policy and, as declared secondary comparators, SBS-B, SIZE-B, SBS, RAW and Ridge, the structured model and boosted trees on deployable features (each tuned by inner group-CV on the training rows only) | prompt §8, declared 2026-10-22 01:33 UTC before any final-test or confirmation evaluation and before any confirmation task was labeled |
| D40 | 1.8.0 study design, declared before any 1.8.0 label (commit `2fdf9a5`, 2026-10-24 10:14 UTC): portfolio-1.8 (the eight 1.6/1.7 candidates plus RO-TREX, RO-TW, RO-DD, ZNE-L-TW, ZNE-E-B, PEC-TW, FR, SV, ZNE-L∘SV); seed scheme 2 with CRN group `task` (development, confirmation) or `base` (controlled studies); 6 fresh designs and r = 60 trials per half; development set `dev18.yaml` (44 new logical circuits) and confirmation set `conf18.yaml` (29 new logical circuits) registered in the ancestry registry; declared tails (risk > 10 × RAW's, risk > 0.1, q95, worst-5 % loss share) reported beside, never instead of, the primary metric; ZNE-E-B as the separately named bounded estimator | review F7, N1 and section E |
| D41 | 1.8.0 confirmatory comparison: the **full deployed policy** (`deploy.recommend`, policy/2, scored on the evaluation half) of the bundle of the frozen configuration (`qemx-rec-1.8.0`, configuration chosen by the unchanged freeze rule — `MODEL[hgb\|deployable18+simulated]` — evaluated at the allocation preflight level its simulated features require) against SBS-B on the mean over base circuits of log10(MSE + 10⁻⁶), paired percentile bootstrap over base circuits (1 000 resamples). The claim "the deployed recommender improves on SBS-B" is established only if this difference is resolved (< 0, interval excluding 0) on the confirmation **and** the frozen configuration's out-of-fold development difference is resolved (the 1.7.0 rule). Reported beside it, not claimed: the model component of the same configuration (the 1.7.0 primary), the full policy of the default deployable bundle (`qemx-rec-1.8.0-deployable`, design-level preflight), and the D39 comparators. Written 2026-10-24 ≈ 23:58 UTC, after the development analysis and the freeze (development results seen), while the confirmation tasks were being labeled and before any confirmation result was computed or inspected | review F1 ("compare complete policies against SBS-B primarily"); D39 |
| D42 | Controlled regime study `regimes18.yaml` (the regimes17 slices relabeled with portfolio-1.8 and scheme 2) and its registration as exploration, declared 2026-10-24 ≈ 20:10 UTC before its labels; the crossed root audit (8 circuits × 4 roots × 3 budgets) declared with its script before its labels; its ranking stability is also reported with the original eight candidates held fixed (added before its analysis) | review section E ("keep the primary portfolio fixed initially") |
| D43 | The 1.7.0 mismatch study is not repeated in 1.8.0 (its question and code path are unchanged; its result stays the evidence, with the 1.6/1.7 portfolio); Figure 6 of the 1.8.0 report is listed as pending | compute budget; ranked as a next experiment |

Decisions taken after results are marked "post hoc" in `RESULTS.md` and are not used for the primary claims.
