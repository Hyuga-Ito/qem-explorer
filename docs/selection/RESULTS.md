# Results — budget-aware QEM protocol selection (QEM Explorer 1.8.0; the 1.7.0 corrected pilot in Part II)

**Every number below is a simulated observation**: emulated trials on IBM fake devices (calibration snapshots
executed by the exact density-matrix engine). No hardware data exist in this release and no hardware job was
submitted; nothing here establishes hardware performance. Methods: `METHODS.md` (1.8.0: §16); plan, hypotheses and
decisions (D1–D43): `RESEARCH_PLAN.md`; figures: `FIGURES.md`; data: `reports/selection_v180/` (Part I) and
`reports/selection_v170/` (Part II). Every table and figure is regenerated from the saved tables without simulation
(`python -m qemx.selection figures|report`, `python -m qemx.selection.thesis_tables`). What the numbers suggest, the
physical explanations and the ranked next experiments are in `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md`.

**Kinds of statement** (the vocabulary of `qemx/selection/claims.py`). *Empirical*: measured on held-out base
circuits of this study — a finite sample of one task distribution. *Confirmatory*: the one comparison declared before
the confirmation set was evaluated (D41), evaluated once. *Exploratory*: controlled single-task comparisons, the
portfolio comparison, the root audit, per-budget and per-family breakdowns, secondary comparators, anything decided
after seeing results (marked *post hoc*) — descriptive, no multiplicity correction. *Model-conditional*: statements
about a fitted model's predictions. Intervals are 95 % percentile bootstrap intervals over independent base circuits
(1 000 resamples) unless stated otherwise; Δ is a paired difference of the mean over base circuits of
log10(MSE + 10⁻⁶) (< 0 is better). A difference Δ changes the *geometric mean of (MSE + 10⁻⁶)* by the factor 10^Δ
(−0.1 ≈ factor 0.79); it is not a percentage change of the MSE. "Not resolved" means the interval contains 0; it is not
"equivalent".

# Part I — QEM Explorer 1.8.0

**In one paragraph.** 1.8.0 evaluates what the application actually does — the full deployed policy with its
execution preflight, out-of-distribution checks, margin rule and fallback — instead of only the model's best
label-feasible candidate, on a 17-candidate portfolio (nine new candidates) with independently namespaced preparation
seeds. On 29 new logical circuits, evaluated once after the freeze, the full deployed policy of the frozen configuration
(boosted trees on deployable and simulator-only features) improves on choosing the best protocol per budget (SBS-B) by
**Δ = −0.144 [−0.231, −0.076]** (geometric-mean MSE × 0.72), with every row scored; its development out-of-fold
difference is −0.088 [−0.145, −0.027]. The rule declared before the confirmation was evaluated (D41) is therefore
met: the claim "the deployed recommender improves on SBS-B" is **established in the sense of that rule, for this
simulated task distribution** — with three qualifications that matter: the optimism-corrected development difference (BBC-CV) is not resolved
(−0.064 [−0.143, +0.013]); D41 was written after the development results were seen; and the established
configuration needs exact simulation at recommendation time (about 0.65 s per new circuit at 3–6 qubits, exponential in
width). The default bundle without simulation reaches −0.084 [−0.179, −0.002], an interval whose exclusion of zero is
within bootstrap Monte-Carlo noise; its distance to its own model component comes from 47 rows on four GHZ circuits,
about half from symmetry verification that its preflight cannot verify without a simulation and half from
out-of-distribution fallbacks (post hoc). On identical development labels, adding candidates gives a resolved gain —
SBS-B over 17 candidates beats SBS-B over the original 8 by −0.070 [−0.133, −0.012] — while learned policies on
deployable features are not resolvedly better than SBS-B within either portfolio. The bounded estimator ZNE-E-B removes
ZNE-E's divergent estimates (largest MSE 10⁹ → 0.04); feature regression (FR) is severe on 61 % of its rows
(circuit-weighted), with a failure peak at five qubits, where its 17 regressors exceed its 16 training circuits (an
interpolation-threshold hypothesis, untested); the crossed root audit finds the labels' standard errors about honest
(median between-root variance ÷ SE² 0.44–1.08 per candidate) and the execution contract exact (288/288). An
independent re-implementation reproduced the confirmation, full-policy, model-component and attainable-portfolio point
estimates to 10⁻¹⁴.

## I.1 What was executed

| workflow | what | status | measured cost |
|---|---|---|---|
| smoke (`smoke18.yaml`) | the 17-candidate pipeline end to end (label, analyze, freeze, bundle, recommend, figures) | executed | minutes |
| development (`dev18.yaml`) | 88 tasks on 44 **new** logical circuits (8 families: GHZ 20, Ising 16, HEA 16, QAOA 16, brickwork 8, BV 4, Grover 4, QFT 4 tasks; 3–6 qubits; `fake_fez` 52, `fake_torino` 20, `fake_kyiv` 16), 10 budgets 10³–3·10⁷, 17 candidates, r = 60 + 60 trials, K = 6 fresh designs per half for the learning candidates, seed scheme 2 | executed, 88/88 ok | 1.3 CPU-h labels |
| development analysis | six splits, 30 model configurations (ridge, elastic net, boosted trees, MLP, structured b² + v/B × six feature sets incl. `deployable18`), BBC-CV, ablations, learning curve, budget generalization, nested selective evaluation, interval coverage, certificates, design variance, failure profile | executed | ≈ 2 h on 2 cores |
| freeze | unchanged rule (lowest pooled out-of-fold mean log10 risk): `MODEL[hgb\|deployable18+simulated]`, nested-CV margin threshold τ = 3.8·10⁻⁵ | executed | — |
| confirmation (`conf18.yaml`) | 58 tasks on 29 new logical circuits (registered before labeling, no ancestry overlap), labeled after the freeze, evaluated once | executed, 58/58 ok | 1.2 CPU-h labels |
| full-policy replay (`scripts/replay_confirmation.py`) | `deploy.recommend` itself on every confirmation row, both shipped bundles, scored on the evaluation half | executed, 0 unscored rows | minutes |
| portfolio comparison (`scripts/portfolio_comparison.py`) | the original 8 against the 17 candidates on identical development labels; every policy refitted per portfolio | executed (exploratory) | minutes |
| crossed root audit (`scripts/root_audit.py`) | 8 circuits × 4 independent preparation roots × 3 budgets × 17 candidates | executed (exploratory) | 0.4 CPU-h |
| regime study (`regimes18.yaml`) | the 1.7.0 regime slices with portfolio-1.8: three 4-qubit circuits on `fake_fez`, error scale s ∈ {0.5, 1, 1.5, 2, 3}, observable and layout blocks (21 tasks) | executed (exploratory) | 0.3 CPU-h |
| noise composition (`noisecomp18.yaml`) | the three circuits under one noise component changed at a time at matched exposure (18 tasks) | executed (exploratory) | 0.3 CPU-h |
| recommendation cost (`profile`) | widths 3–7 by component; early rejection at 12, 24, 40 qubits | executed | minutes |
| hardware plan (`hardware plan`, dry run) | the shipped bundle's validation plan: 28 jobs, 524 s reserved of the 540 s window | **planned, not executed**; nothing submitted | — |
| replay on the 1.7.0 confirmation (`scripts/replay_v170_confirmation.py`) | the 1.7.0 bundles under policy/2 on the inspected 1.7.0 confirmation | executed (exploratory, post-inspection; private evidence) | minutes |
| independent reproduction | a separate agent recomputed, from the saved tables with its own code (without the project's code), seven confirmation comparisons, both full deployed policies, the default bundle's model component and the attainable portfolio contrast | executed | — |
| mismatch study | not repeated (D43); the 1.7.0 result (Part II §10) stays the evidence; Figure 6 of the 1.8.0 report is pending | **not executed** | — |

Labeling used about 3.5 CPU-hours on a 2-core machine shared with the analyses; one container restart interrupted the
development labeling, which resumed from its records (`--accept-changes`, recorded in `dev/provenance.json`).

## I.2 The confirmatory comparison (D41; Table 3 of the thesis tables, `supp_deployment_policy`)

| policy on the 29 confirmation circuits (580 rows) | mean log10(MSE + 10⁻⁶) | Δ vs SBS-B [95 %] | arithmetic MSE | kind |
|---|---|---|---|---|
| **full deployed policy, frozen configuration** (`qemx-rec-1.8.0`; simulator-only features; allocation-level preflight) | −4.496 | **−0.144 [−0.231, −0.076]** | 5.4·10⁻⁴ | confirmatory (D41) |
| model component of the same configuration (its best label-feasible candidate) | −4.511 | −0.159 [−0.253, −0.086] | 6.1·10⁻⁴ | reported beside (the 1.7.0 primary) |
| full deployed policy, default bundle (`qemx-rec-1.8.0-deployable`; deployable features; design-level preflight) | −4.436 | −0.084 [−0.179, −0.002] | 5.7·10⁻⁴ | reported beside; interval fragile (below) |
| model component of the default bundle (`MODEL[hgb\|deployable18]`, from its own predictions) | −4.459 | −0.107 [−0.224, −0.006] | 9.5·10⁻⁴ | reported beside |
| ridge, 1.7.0 deployable features (D39 comparator) | −4.482 | −0.130 [−0.226, −0.037] | 6.6·10⁻⁴ | exploratory |
| boosted trees, 1.7.0 deployable features | −4.444 | −0.092 [−0.204, +0.010] | 9.1·10⁻⁴ | exploratory |
| structured b² + v/B, 1.7.0 deployable features | −4.367 | −0.014 [−0.125, +0.085] | 1.0·10⁻³ | exploratory |
| SIZE-B | −4.366 | −0.013 [−0.130, +0.103] | 7.8·10⁻⁴ | baseline |
| SBS-B | −4.352 | 0 | 8.6·10⁻⁴ | reference |
| SBS (single best protocol) | −4.144 | +0.208 [+0.067, +0.380] | 1.4·10⁻³ | baseline |
| RAW | −3.178 | +1.175 [+0.968, +1.411] | 8.0·10⁻³ | baseline |
| empirical reference (selection-half best) | −4.653 | −0.301 [−0.406, −0.210] | 3.1·10⁻⁴ | attainable bound |

**What the application did** (`replay/`, panel A of `supp_deployment_policy`). Frozen bundle: 520 rows accepted, 60
out-of-distribution rows (6 tasks) fell back to SBS-B's first verified candidate, 233 rows accepted a candidate after
the preflight skipped higher-ranked ones (mostly SV and ZNE-L∘SV, infeasible outside tasks with a declared symmetry;
277 rows list skipped candidates, 44 of them out-of-distribution fallbacks), 0 unscored rows, no margin fallback
(τ accepts ≈ 99 % of development rows). Default bundle: 530 accepted, 50 out-of-distribution (5 tasks), 282 accepted
after skipping (326 rows list skipped candidates), 0 unscored. No full-policy row exceeded ten times RAW's risk; the
95 % quantile of the chosen risk is 2.6·10⁻³ for both bundles against 4.6·10⁻³ for SBS-B. Classical cost of a call that
computes a task's features: median 0.65 s (frozen bundle, exact simulation) and 0.05 s (default bundle); later budgets of
the same task reuse the cached features.

**Statement (confirmatory in the sense of D41).** The full deployed policy of the frozen configuration is resolvedly
better than SBS-B on the confirmation (−0.144 [−0.231, −0.076]) and the frozen configuration's out-of-fold development
difference is resolved (−0.088 [−0.145, −0.027]): the declared rule is met and the claim is **established in its
sense** — in simulation, for this task
distribution (3–6 qubits, eight circuit families, three fake devices, budgets 10³–3·10⁷). Qualifications, all reported
with the result:

1. *Selection optimism.* The development difference is that of the configuration chosen as the best of 30; corrected by
   bootstrap bias-corrected CV it is −0.064 [−0.143, +0.013] (probability of a negative difference 0.94, optimism
   0.020). The 1.7.0 rule uses the uncorrected out-of-fold difference; under a BBC-corrected rule the development
   criterion would not be met.
2. *Timing of D41.* D41 (the full deployed policy as primary) was written after the development analysis and the freeze,
   while the confirmation tasks were being labeled and before any confirmation result was computed. The freeze rule,
   the configuration and the confirmation set were fixed earlier (D40); the choice of primary *estimand* was not
   pre-registered before development results existed.
3. *What is established.* The established recommender computes simulator-only features by exact simulation of the
   supplied noise model at recommendation time (cost exponential in width; allocation-level preflight limited to 11
   qubits). The deployable default bundle is a separate, secondary result.
4. *Simulated observations only.* Hardware transfer is not addressed (§I.11).

**The default bundle's interval is fragile.** Its upper bound (−0.002) sits on zero within bootstrap Monte-Carlo noise:
over 200 bootstrap seeds of 1 000 resamples the upper bound is below 0 in 59 % (range −0.009 to +0.006), and with
200 000 resamples it is −0.0007 (independent reproduction, private evidence). Its model component behaves the same way
(77 %; −0.0023). These are *not resolved robustly* and are not claimed. The frozen bundle's interval excludes 0 in
every seed.

**Independent reproduction.** A separate agent re-implemented the metric and the bootstrap from the saved tables
(`confirm/labels.csv`, `choices_confirm.csv`, `replay/*.csv`, development labels) without the project's code. Every
point estimate of the table above except the structured comparator, both full-policy differences, the default bundle's
model component and the portfolio "attainable" contrast agree to ≤ 1.8·10⁻¹⁴; each replayed risk equals the label of
the returned candidate on all 580 rows of both bundles (0 mismatches, 0 infeasible choices); intervals agree within
Monte-Carlo spread. Not independently recomputed: the development out-of-fold and BBC-CV numbers, the fixed-policy
portfolio contrast, the tails, the audit, the controlled studies and the costs.

## I.3 Why the default bundle trails its own model component (post hoc, exploratory)

On 47 confirmation rows (4 GHZ circuits, 8 tasks) the default bundle's model component chose SV or ZNE-L∘SV, which the
labels show feasible there (the component's choice is the labelled best on 25 of the 47 rows). On 33 of these rows the
design-level preflight of the default bundle (no simulation) returned *unverified* for them — SV's minimum target
shots ⌈5/P_accept⌉ need the noisy acceptance probability, which only a simulation of the supplied model provides — and
policy/2, which never accepts an unverified candidate, chose another one; the other 14 rows were out of distribution
and fell back to SBS-B. On the 47 rows the full policy is 0.30 decades worse than its model component and only 0.03
better than SBS-B. Replacing all 47 rows by the component's choice would move the default bundle from −0.084 to
−0.109 (its component: −0.107); the 33 preflight rows alone account for −0.096 and the 14 out-of-distribution rows
alone for −0.097, i.e. about half each. The frozen bundle, which verifies at the allocation level, chose SV on 30 rows
and ZNE-L∘SV on 11. The preflight half is the measured price of the conservative contract; a cheap acceptance bound
for declared symmetries is ranked as a next step (`RESEARCH_FINDINGS_AND_NEXT_STEPS.md`).

## I.4 More candidates or better selection? (`supp_portfolio_expansion`, thesis Table 8; exploratory)

Development data, out of fold (5 folds grouped by base circuit), every policy refitted per portfolio, feature set
`deployable18`, identical tasks and labels:

| contrast | Δ [95 %] | factor 10^Δ |
|---|---|---|
| more candidates, fixed policy (SBS-B over 17 − SBS-B over 8) | **−0.070 [−0.133, −0.012]** | 0.85 |
| more candidates, attainable (reference over 17 − reference over 8) | −0.092 [−0.165, −0.037] | 0.81 |
| learning within portfolio-1.6 (ridge − SBS-B; HGB − SBS-B) | +0.015 [−0.033, +0.066]; +0.002 [−0.057, +0.063] | 1.04; 1.01 |
| learning within portfolio-1.8 (ridge − SBS-B; HGB − SBS-B) | −0.017 [−0.064, +0.030]; −0.011 [−0.094, +0.068] | 0.96; 0.98 |
| total (ridge 1.8 − SBS-B 1.6; HGB 1.8 − SBS-B 1.6) | −0.087 [−0.175, −0.009]; −0.081 [−0.197, +0.019] | 0.82; 0.83 |

On development data the expansion of the portfolio gives the resolved gain; learning with the `deployable18`
features adds nothing resolved in either portfolio (the two contrasts were not compared formally). SBS-B over 17 candidates chooses ZNE-E-B (48 % of rows) and PEC-TW (46 %); the empirical
reference over 17 uses PEC-TW and PEC (18 % each), ZNE-E-B (9 %), RO-TREX and CDR (8 %), SV (7 %). Seven learned
configurations are resolved against SBS-B on development data (`analysis/pooled_same-distribution.csv`): three with
the simulator-only features (HGB −0.088, ridge −0.066, elastic net −0.062 with `deployable18+simulated`, against HGB
−0.011, ridge −0.017, elastic net −0.036 with `deployable18` alone) and four linear models on deployable subsets
(ridge −0.077 and elastic net −0.068 on the light-cone features, ridge −0.056 and elastic net −0.061 on budget and size
alone). Simulator-only features give the largest development gains; the full `deployable18` set does not help the
learners it was meant for. On the confirmation, models on the 1.7.0 deployable features did better than on development data (ridge
−0.130 resolved, HGB −0.092 not resolved) — one confirmation set, 29 circuits; SBS-B is also weaker there (its gap to
the reference is 0.30 against 0.28 on development data).

## I.5 Which candidate wins where (Figure 2, `supp_target_precision`, thesis Table 5)

Measured best on the evaluation half (an oracle on the scored data; the reference policy of I.4 chooses on the
selection half), development tasks (88 per budget): at B = 10³ RAW (21), RO-TREX (17), PEC (13) and PEC-TW (10); from 10⁴ to 10⁶ PEC and PEC-TW together win 31–43 tasks per budget; at B ≥ 3·10⁶ CDR rises to 13–21,
and at 3·10⁷ ZNE-L-TW, ZNE-R and ZNE-L take 6–8 each. SV, feasible on the 16 GHZ tasks only, is best on 8–10 of them up to
B = 10⁵ and on 2 at 3·10⁷. FR is best once in 880 rows. Median evaluation-half MSE at B = 10⁵ (feasible tasks): SV
1.3·10⁻⁵, PEC-TW 1.7·10⁻⁵, PEC 1.7·10⁻⁵, ZNE-L∘SV 2.0·10⁻⁵, ZNE-E-B 4.1·10⁻⁵, ZNE-L 4.4·10⁻⁵, CDR 5.4·10⁻⁵, RO-TREX
1.6·10⁻⁴, RAW 4.6·10⁻⁴, NIL 3.2·10⁻³, FR 1.0·10⁻². Descriptive: candidates have different feasible subsets.

## I.6 Severe tails (finding F7; `supp_tails`, thesis Table 6)

Declared before the 1.8.0 results (D40): severe = risk > 10 × RAW's on the same row; also risk > 0.1, the 95 % quantile
and the share of loss carried by the worst 5 % of base circuits. Development rows, each candidate on its feasible set:

| candidate | severe (> 10 × RAW; circuit-weighted share of feasible rows) | largest MSE | remark |
|---|---|---|---|
| ZNE-E | 2.4 % | 1.0·10⁹ (B = 10³) | 5 rows above MSE 1, all at B ≤ 3 000 |
| ZNE-E-B | 2.4 % | 0.043 | projection onto the observable's range; the same evaluation-half MSE as ZNE-E on 788 of 880 (task, B) cells, never larger on the other 92 |
| ZNE-R | 11.4 % | 0.021 | 41 % of rows at B = 10³, 2.3 % at 10⁶, ≤ 1.1 % from 3·10⁶ on |
| NIL | 47.9 % (47.3 % of rows) | 0.31 | ≈ 45–49 % at every budget; infeasible below 3·10⁴ |
| FR | 61.0 % (62.0 % of rows) | 2.0·10⁴ | 102 rows above MSE 1; by width 36 / 58 / 85 / 70 % severe at n = 3 / 4 / 5 / 6, median MSE 0.001 / 0.010 / **1.87** / 0.009 |
| CDR | 2.6 % | 0.034 | infeasible at 10³ |
| PEC, PEC-TW, RO family, ZNE-L, ZNE-L-TW, SV, ZNE-L∘SV | 0–2.5 % | ≤ 0.027 | RAW's own largest MSE is 0.032 |

The bounded estimator removes the catastrophic magnitudes without changing how often ZNE-E is worse than RAW by a
factor 10: the projection caps an estimate at the observable's range, which bounds the error by 2 but not by a factor
of RAW's small error. At the policy level the declared tails are small: on the confirmation no policy except the
structured comparator (1.7 %) has a row above 10 × RAW; on development data the frozen configuration has 0.45 % of rows
against SBS-B's 0.34 %, but concentrates its losses more (worst-5 % share 0.37 against 0.23) — its improvements are
broad, its failures concentrated.

## I.7 Development analysis (Figures 1, 3, 4, 5; thesis Table 4)

* **Frozen configuration**, out of fold: −0.088 [−0.145, −0.027] vs SBS-B; ridge with the light-cone features −0.077
  [−0.134, −0.022]; ridge on budget and size alone −0.056 [−0.117, −0.003]; HGB `deployable18` −0.011 [−0.094, +0.068].
  The MLP fails catastrophically on two feature sets (arithmetic MSE 2.5·10⁵ from single divergent predictions; its
  log metric is merely worse). By family the frozen configuration is resolvedly better than SBS-B on brickwork (4
  circuits), GHZ (10) and Ising (8) — exploratory; for the two-circuit families (BV, Grover, QFT) a bootstrap over
  circuits only spans the two circuits' values and resolves nothing.
* **Generalization splits** (all budgets): no learned configuration is resolvedly better than SBS-B when whole families,
  widths or (except boosted trees on budget and size, −0.122 [−0.204, −0.047]) devices are held out. Budget
  interpolation: only ridge on budget and size is resolved (−0.076 [−0.146, −0.019]); boosted trees and the structured
  model on `deployable18` are resolvedly worse (+0.18, +0.20). Budget extrapolation: trained on B ≤ 10⁴, the mean log10
  regret beyond the training range is ridge 0.40, SBS-B 0.48, HGB 0.58, structured 1.06; trained on B ≤ 3·10⁵ it is
  HGB 0.38, ridge 0.38, SBS-B 0.46, structured 0.60; no learned model is resolved against SBS-B in the pooled
  extrapolation split.
* **Ablation** (frozen configuration, every variant retrained): removing budget costs +0.116 [+0.032, +0.209] decades of
  regret, the light cone +0.032 [+0.002, +0.077], the observable +0.035 [+0.006, +0.070]; the device-snapshot features
  (+0.039 [−0.012, +0.092]), protocol overheads and basic structure are not resolved.
* **Learning curve** (mean over folds and repeats): mean log10 regret 0.39 with 9 training base circuits, 0.31 with
  18, 0.30 with 26 and 0.29 with 35, while SBS stays at 0.42–0.46 — the gain over 9 circuits is clear, the curve is
  nearly flat from 18 circuits on.
* **Reliability**: nested selective evaluation accepts 99.1 % of rows and is indistinguishable from random rejection at
  the same coverage (+0.0001 [−0.0035, +0.0045]); cv-residual intervals cover 0.911 and split-conformal 0.913 at nominal
  0.90, but the simultaneous versions are infinite; finite-sample certificates are uninformative (Hoeffding radius
  0.92, Bernstein 1.12, far above the risks); the 2ε post-hoc bound holds on 880/880 rows.
* **Learning designs** (`supp_design_variability`): the between-design share of the label variance at the largest
  budget is 0.90 for CDR, 0.82 for NIL and 0.47 for FR; a trial-level SE would understate the design-level SE by a
  median factor up to 2.4 (CDR at 3·10⁷), 2.0 (NIL) and 1.4 (FR). Labels use the design-level SE.

## I.8 Are the labels honest across preparation roots? (finding N1; `supp_root_audit`; exploratory)

Eight circuits (the circuits of the ten 1.7.0 deployment regression cases, earlier failure types and controls),
labeled under four independently namespaced roots at B = 10⁴, 3·10⁵, 10⁷:

* **Standard errors.** For 338 (task, candidate, budget) cells feasible under ≥ 2 roots, the between-root variance of
  the evaluation-half MSE divided by the mean squared design-level SE has medians 0.44 (PEC-TW) to 1.08 (RO-TW) per
  candidate (above 1 for RO-TW, FR, SV and CDR) — about honest, not understated on average; 4 cells exceed 4, and
  NIL's 90 % quantile is 6.1 (its rare large failures depend on the root).
* **Rankings.** The winner is the same under all four roots in 10 of 24 (task, budget) cells with 17 candidates and in
  11 of 24 with the original 8; using another root's winner costs on average −0.006 decades (17) and −0.001 (8) — the
  disagreements are between near-equal candidates (largest cross-root regret 0.33). No root's winner was infeasible
  under another root.
* **Execution contract.** The design-level preflight agrees with each root's actual allocation on 288/288 learning-
  candidate rows (CDR, FR, NIL; feasible and infeasible).
* **Deployed policy per root**: mean log10 risk −4.77, −4.78, −4.76, −4.80 — a spread of 0.04 decades.

## I.9 Controlled regimes and noise composition (finding F10; Figure 3 B–D; exploratory)

* **Error scale** (`regimes18`, four-qubit HEA, Ising and QAOA on `fake_fez`, all budgets): the frozen recommender's
  mean log10 regret is −0.04 to 0.09 for s ≤ 1.5 but rises at s = 2 (Ising 0.31, HEA 0.13) and s = 3 (HEA 0.60, Ising 0.49);
  QAOA stays ≤ 0.08, and its top candidates are unresolved in every cell. The training data contain only the
  devices' native error rates (s = 1). The selective rule abstained in none of these cells.
* **Noise composition** (`noisecomp18`, one component changed at a time at matched exposure): with half of the gate
  error coherent (χ = 0.5, s = 2) the recommender's regret is 1.04 (HEA), 0.44 (Ising) and 1.15 (QAOA) decades over all
  budgets — the largest of the noise-composition settings for HEA and QAOA. At B = 10⁵ the measured best is ZNE-E
  (HEA), RO-DD (Ising) and ZNE-L (QAOA), the first two unresolved against their runner-up; the recommender, which sees
  calibration error rates and the light-cone fault weight, chooses PEC-TW, ZNE-E and ZNE-L. QAOA's large regret comes
  from B ≥ 3·10⁵, where the recommender picks CDR or ZNE-E-B instead of ZNE-L. A longer idle time (t = 5.5) costs 0.64
  (HEA), 0.19 (Ising) and 0.41 (QAOA). The unchanged setting itself has a regret of 0.15 for HEA here against 0.06 in
  the regime study, so differences of about 0.1 between cells are within their noise; most cells' top two are
  unresolved. The direction of the coherent-error result agrees with 1.7.0's (Part II §5), with the larger
  portfolio.

## I.10 Cost of a recommendation and early rejection (finding F2; `supp_runtime`, thesis Table 7)

Median of repeats on the build machine (2 shared cores): the whole `recommend()` call of the default bundle takes
0.039–0.059 s at n = 3–7 (the n = 7 probe is out of distribution); separately timed components: features
0.014–0.036 s, inference ≈ 0.027 s, structural preflight of all 17 candidates < 0.001 s, and the design-level preflight
of *all* learning candidates 0.07–0.32 s — more than a whole call, because the call verifies lazily, only down the
model's order until the first verified candidate. The simulator-only features of the frozen bundle cost 0.23–0.90 s and
grow exponentially; generating one development task's benchmark labels takes 35–59 s. Wide requests are rejected
before any dense engine: GHZ chains of 12, 24 and 40 qubits return *out of distribution* in 2.0, 4.6 and 16.5 s (the
process's peak resident memory stays at 330–354 MB).

## I.11 What was not executed

* **Hardware.** The validation plan of the shipped bundle (`hardware_plan/`) was produced as a dry run — 28 jobs, 524 s
  of the 540 s window, plan digest `6015610c…` — and nothing was submitted (owner authorization required).
* **Mismatch study** under the 1.8.0 policy (D43): not repeated; Figure 6 is pending in `reports/selection_v180`.
* **Larger study** (more circuits, wider circuits, more devices): not run; the learning curve is nearly flat from 18
  to 35 circuits, so whether learning on deployable features is limited by data or by information is open.
* **Exploratory replay on the 1.7.0 confirmation** (post-inspection, not confirmatory; rerun with the reviewed replay
  code, private `runs/replay_v170b`): the 1.7.0 deployable bundle under policy/2 −0.041 [−0.069, −0.009] vs SBS-B, the
  1.7.0 simulator bundle −0.026 [−0.048, −0.004] (222 of 460 in-distribution rows accepted by its margin rule; the
  first run, before the review fixes, gave −0.025 and 220); 0 of 500 rows unscored in each.

## I.12 Limitations and threats to validity (1.8.0)

Simulated fake devices only, with Pauli-twirled or Markovian noise models except where the noise-composition study adds
coherent errors; 3–6 qubits; 73 new logical circuits in total (44 + 29), eight families; one confirmation set; D41
written after development results; selection optimism (BBC) not resolved; the established configuration needs exact
simulation; the default bundle's advantage is not robust; the 10⁻⁶ floor compresses differences between very small
risks (sensitivity in `analysis/floor_sensitivity.csv`); tails are declared but not part of the primary rule; the
mismatch question was not re-examined with 17 candidates.

---

# Part II — the corrected pilot of QEM Explorer 1.7.0 (historical)

The text below is the 1.7.0 results document, unchanged except for this heading and its title line. Its section numbers (§1–§17) refer to
Part II.

*Original title: Results — budget-aware QEM protocol selection, corrected pilot (QEM Explorer 1.7.0).*

**Every number below is a simulated observation**: emulated trials on IBM fake devices (calibration snapshots
executed by the exact density-matrix engine). No hardware data exist in this release and no hardware job was
submitted; nothing here establishes hardware performance. Methods: `METHODS.md`; plan, hypotheses and decisions
(D1–D39): `RESEARCH_PLAN.md`; figures: `FIGURES.md`; data: `reports/selection_v170/` (its `RESULTS_TABLES.md` is
regenerated by `python -m qemx.selection report`, every figure by `python -m qemx.selection figures`, both from the
saved tables without simulation). The 1.6.0 results are superseded by this document; §15 says which of their claims
survive.

**Kinds of statement** (the vocabulary of `qemx/selection/claims.py`). *Empirical*: measured on held-out base
circuits of this study — a finite sample of one task distribution. *Confirmatory*: the one comparison declared before
the confirmation set was labeled and evaluated once (§12). *Historical*: the 1.6.0 final test, whose circuits earlier
studies had seen (§11). *Exploratory*: controlled single-task comparisons, per-budget and per-family breakdowns,
secondary comparators — descriptive, no multiplicity correction. *Model-conditional*: statements about a fitted
model's predictions. Intervals are 95 % bootstrap intervals over independent base circuits (1 000 resamples) unless
stated otherwise; Δ is a paired difference of mean log10 MSE (< 0 is better; −0.1 is a factor 1.26 in
geometric-mean MSE). "Not resolved" means the interval contains 0; it is not "equivalent".

**In one paragraph.** With CDR's and NIL's labels corrected to the risk of a freshly drawn training design (review
R1), every conclusion of 1.6.0 that does not involve those two protocols' reliability survives, and the trial-level
standard errors of 1.6.0 understated their uncertainty by a factor 1.1–2.5. Choosing the protocol per budget (SBS-B)
remains most of the available gain: −0.141 [−0.205, −0.073] decades against the single best protocol on held-out
development circuits. The frozen learned recommender (boosted trees on deployable and simulated features, chosen
again on the corrected labels) is −0.043 [−0.105, 0.026] against SBS-B on held-out development circuits — not
resolved — and −0.087 [−0.161, −0.024] on the 25 new logical circuits of the once-only confirmation; by the rule
declared before the study (a resolved advantage on *both*), the claim is **not established**, although both point
estimates favour the recommender and the per-task oracle shows 0.20–0.34 decades still available. Budget is the only
information whose removal resolvedly hurts. Models win clearly only when extrapolating beyond the training budgets,
where ridge regression extrapolates best; the physics-informed b² + v/B model is between ridge and boosted trees there
and worse in range. Abstention evaluated by nested cross-validation does not beat random rejection at matched
coverage. In controlled single-task studies, the best protocol moves with the observable, the error scale and the
noise *composition* — a coherent fraction of the gate errors displaces PEC in all six cells tested — while the
recommender, which sees only Pauli calibration data, keeps choosing PEC. A cheap echo diagnostic detects mismatch at
its stated false-positive rate, but at equal total budget no reaction to it beats spending every shot on the target,
and the learned reaction does as well triggered at random as by the diagnostic. NIL's severe failures at 16 training
circuits are spread over designs on two of five tasks (a protocol-level problem at this size) and concentrated in a
single design on one; more training circuits halve NIL's error on two tasks without removing the failures.

## 1. What was executed

| workflow | what | status | measured cost |
|---|---|---|---|
| CI matrix and smoke workflow | tests on Python 3.10, 3.12, 3.13 (three dependency sets), static UTF-8 check, `smoke.yaml` | executed (QA_SUMMARY) | — |
| corrected pilot, development (`pilot17.yaml`) | the 120 tasks of the 1.6.0 pilot (40 base circuits, 8 families, 3–6 qubits, `fake_fez`, `fake_torino`, `fake_kyiv`) relabeled with the same seeds and fresh learning designs: 10 budgets 10³–3·10⁷, 8 protocols, r = 100 + 100 trials per label, K = 10 independent designs per half for CDR and NIL | executed, 0 failed tasks | 3.5 CPU-h labels |
| historical final test | the 36 tasks (12 base circuits) of the 1.6.0 final test, relabeled the same way, evaluated once with the frozen configuration | executed | 2.0 CPU-h labels |
| development analysis | six splits, 25 model configurations (ridge, elastic net, boosted trees, MLP, structured model × five feature sets), BBC-CV, group and semantic ablations, learning curve, budget generalization, nested selective evaluation, interval coverage, certificates, design variance and diagnostics, failure profile | executed | ≈ 2 h on 2 cores (restart-safe, checkpointed) |
| regime study (`regimes17.yaml`) | 21 tasks: three 4-qubit circuits on `fake_fez`, error scale s ∈ {0.5, 1, 1.5, 2, 3}, observable and layout blocks | executed | 0.4 CPU-h |
| noise-composition study (`noisecomp17.yaml`, new) | the three regime circuits under one noise component changed at a time (10 settings), hypotheses H1–H3 declared before labeling | executed | 0.4 CPU-h |
| mismatch study (`mismatch17.yaml`) | 90 tasks (10 circuits × 9 execution laws), 5 budgets, r = 60 + 60; equal-budget reactions with held-out discrepancy types | executed | 1.9 CPU-h |
| design study (experiment B, `scripts/design_study.py`) | 5 tasks × 8 CDR/NIL training allocations × 20 designs × 3 budgets, 10 trials per design | executed | ≈ 2 CPU-h (estimated from timing, not separately logged) |
| confirmation (`confirm17.yaml`, new) | 50 tasks on 25 new logical circuits, registered before labeling, labeled after the freeze, evaluated once | executed | 1.8 CPU-h labels |
| recommendation runtime (`profile`) | widths 3–7 | executed | minutes |
| hardware validation (`hardware17.yaml`) | plan for two tasks, RAW/RO/recommended/SBS-B, four blocks | **planned, not executed** (owner authorization required, §14) | — |
| larger study (`full17.yaml`) | 242 logical circuits, 726 tasks | **planned, not executed** (§14) | estimate 19 CPU-h |

Labeling used about 10 CPU-hours in total on a 2-core machine shared with the analysis; container restarts (about
every two hours) interrupted the pipelines several times, and every step resumed from its last checkpoint (no label or
table was recomputed with different inputs; each resume is recorded in the run's provenance).

## 2. Labels, and what the design correction changed (review R1)

**Same tasks, same seeds.** The corrected labels differ from the 1.6.0 labels only in how CDR and NIL draw their
training designs: 1.6.0 reused six designs in both label halves (the risk of one fixed set of designs, with trial-level
standard errors); 1.7.0 draws K = 10 fresh, independent designs per half from separate seed pools (the risk of a
deployment that draws a new design, with design-level standard errors). As required, the six protocols without a
training design are **identical trial for trial** on all 120 development tasks (36/36 on the final test except one
ZNE-R task whose allocation changed by one shot at a near-tie; 21/21 regime tasks) — `analysis_estimands/identity.csv`.

**The estimand change** (`analysis_estimands/summary.csv`, development, evaluation half):

| | CDR | NIL |
|---|---|---|
| median design-level SE ÷ trial-level SE (corrected labels), over budgets | 1.12 – 2.51 (largest at B ≥ 3·10⁶) | 1.83 – 2.14 |
| median between-design share of the label variance | 0.22 – 0.92 (grows with B) | 0.77 – 0.86 |
| median ratio of the corrected to the 1.6.0 SE as reported | 1.15 – 2.10 | 1.64 – 2.01 |
| share of tasks whose 1.6.0 MSE lies outside the corrected label's ±1.96 SE | 16 – 30 % | 22 – 29 % |
| median \|log10\| change of a task's MSE | 0.04 – 0.12 | 0.16 – 0.21 |
| median MSE over tasks | within about 25 % | 1.3 – 1.9 × higher (e.g. 2.4·10⁻³ → 4.5·10⁻³ at 10⁶) |

So the 1.6.0 labels understated the uncertainty of every CDR and NIL label, and understated NIL's typical risk. The
design SE is the honest one for a protocol that draws a new design per deployment; with K = 10 designs per half it is
itself uncertain (few clusters).

**Failures** (supplementary failure figure): 0 of 347 label tasks failed in software. Protocol infeasibility is
separate (CDR infeasible on 15 of 120 tasks at every budget — no non-Clifford gate to train on, D17; NIL infeasible
below 3·10⁴ and on 27 tasks at every budget). The union rate of trials with at least one fallback is largest for ZNE-E at small budgets
(15 % at 10³, 5 % from 10⁶) and NIL (≈ 9 % at every feasible budget). ZNE-E's rare divergent exponential
extrapolations (single trials with |estimate| ≫ 1) are part of its risk and produce the catastrophic arithmetic
policy MSEs of §4.

## 3. Which protocol wins where (Figure 2, Figure 3 B–C, supplementary target-precision and bias–variance figures)

Median log10 MSE over development tasks (evaluation half, feasible tasks; corrected labels):

| protocol | 10³ | 10⁴ | 10⁵ | 10⁶ | 10⁷ | 3·10⁷ |
|---|---|---|---|---|---|---|
| RAW | −2.86 | −3.29 | −3.38 | −3.38 | −3.38 | −3.38 |
| RO | −2.85 | −3.62 | −3.79 | −3.83 | −3.83 | −3.82 |
| ZNE-L | −2.39 | −3.39 | −4.34 | −5.19 | −5.64 | −5.71 |
| ZNE-R | −1.93 | −2.94 | −3.91 | −4.86 | −5.59 | −5.80 |
| ZNE-E | −2.35 | −3.35 | −4.30 | −5.22 | −5.72 | −5.83 |
| PEC | −2.84 | −3.79 | −4.64 | −5.08 | −5.14 | −5.15 |
| CDR | — | −3.21 | −4.23 | −5.09 | −5.59 | −5.66 |
| NIL | — | — | −2.41 | −2.35 | −2.32 | −2.43 |

The measured best protocol moves with the budget: RAW or PEC at 10³ (50 and 48 of 120 tasks), PEC from 3·10³ to
3·10⁵ (55–69 tasks), then ZNE-E, CDR, PEC and ZNE-L at 10⁶ and above, where the bias floors of PEC (the snapshot's
median-rate model is not the device's gate-by-gate noise), RAW and RO take over. NIL is the best protocol in 1 of
1 200 (task, budget) rows; at 16 training circuits its error does not fall with B (§9).

**Budget for a target precision** (per task the smallest evaluated budget meeting the target; median over tasks;
unattained targets censored, never extrapolated). MSE ≤ 10⁻⁴: PEC at 3·10⁴ (82 % of tasks attain it at some budget),
ZNE-L, ZNE-E and CDR at 10⁵, ZNE-R at 3·10⁵; RAW, RO and NIL are not reached by the median task (28 %, 43 %, 28 %).
MSE ≤ 10⁻⁵: ZNE-L, ZNE-E, PEC and CDR at 10⁶ (57–80 %), ZNE-R at 3·10⁶. MSE ≤ 10⁻⁶: ZNE-E at 10⁷ (61 %; 3·10⁷ with the
upper 95 % bound), ZNE-R and ZNE-L at 3·10⁷; CDR (49 %) and PEC (26 %) are not reached by the median task — CDR
reached it in 1.6.0 by its point estimate; with design-level uncertainty it does not.

**Controlled regime study (Figure 2; D23).** Three 4-qubit circuits, only the device error scale s and the budget
changing: 5 (Ising), 4 (HEA) and 5 (QAOA) distinct reference protocols over the maps; 68 of 150 cells are unresolved
(45 %; 1.6.0: 38 % — the design-level SEs of CDR and NIL are larger). The frozen recommender (trained on the
development and historical-final circuits, not on these) chooses the reference in 63 % of the 82 resolved cells
(66 % after abstention; it abstains in 62 of 150 cells). Its deployed mean log10 regret is −0.01 to 0.15 on QAOA,
0.04 to 0.12 on Ising, and grows with s on HEA (0.07 at s = 0.5 to 0.67 at s = 3; largest cell 1.49).

**Controlled single-task comparisons at B = 10⁵ (Figure 3 B, C; exploratory).** Ising, s = 1: RAW is the measured best
for the two weight-1 observables, PEC for the weight-2 and weight-4 ones (as in 1.6.0); the recommender chooses PEC for
all four. Along s: RAW (s ≤ 1) → PEC (1.5, 2) → ZNE-L (3); the s = 2 and s = 3 cells are unresolved, so the switch is
not established.

**Error floors** (bias–variance figure, regime slices at s = 1): at B = 3·10⁷ the corrected squared bias exceeds the
variance of one estimate for RAW, RO, PEC and ZNE-L on all three circuits and for CDR on Ising — floors that more shots
do not remove. CDR's remaining variance is mostly between designs at that budget (99 % on HEA, 73 % on Ising, 39 % on
QAOA): variance that only more or better training designs remove.

## 4. Policies against strong baselines (Figure 1, Table 3; empirical)

Held-out base circuits, group 5-fold over the 40 development base circuits, all 10 budgets, out-of-fold pooled. The
reference of the regret is the protocol identified on the selection half and scored on the evaluation half, so
regret can be negative. Severe: risk more than 10 × the reference's.

| policy | mean log10 MSE | Δ vs SBS-B | mean log10 regret | P(regret > 2×) | severe | mean MSE |
|---|---|---|---|---|---|---|
| per-task oracle (all-candidate reference) | −4.577 [−4.668, −4.474] | −0.203 [−0.259, −0.147] | 0 | 0 | 0 | 2.7·10⁻⁴ |
| **frozen: HGB, deployable + simulated** | −4.417 [−4.532, −4.284] | −0.043 [−0.105, +0.026] | 0.160 | 0.17 | 2.8 % | 7.1·10⁻⁴ |
| — bias-corrected for selection among 25 (BBC-CV) | −4.386 [−4.565, −4.203] | −0.018 [−0.098, +0.058] | | | | |
| ridge, deployable + simulated | −4.402 [−4.526, −4.270] | −0.028 [−0.069, +0.013] | 0.174 | 0.20 | 3.1 % | 7.1·10⁻⁴ |
| HGB, deployable | −4.381 [−4.510, −4.253] | −0.007 [−0.063, +0.042] | 0.196 | 0.23 | 4.3 % | 8.5·10⁻⁴ |
| ridge, deployable | −4.345 [−4.480, −4.198] | +0.029 [−0.029, +0.088] | 0.231 | 0.27 | 4.7 % | 7.2·10² |
| structured b² + v/B, deployable + simulated | −4.359 [−4.473, −4.236] | +0.015 [−0.033, +0.067] | 0.217 | 0.23 | 5.0 % | 1.2·10⁻³ |
| structured b² + v/B, deployable | −4.301 [−4.435, −4.152] | +0.073 [+0.011, +0.133] | 0.276 | 0.29 | 8.2 % | 1.5·10⁻³ |
| SIZE-B (depth-3 tree on n, two-qubit gates, log B) | −4.367 [−4.510, −4.224] | +0.007 [−0.034, +0.052] | 0.210 | 0.21 | 6.8 % | 1.5·10⁻³ |
| **SBS-B** (best fixed protocol per budget) | −4.374 [−4.492, −4.236] | 0 | 0.203 | 0.20 | 5.2 % | 7.0·10⁻⁴ |
| SBS (single best protocol: PEC) | −4.233 [−4.383, −4.074] | +0.141 [+0.073, +0.205] | 0.344 | 0.31 | 11.7 % | 7.6·10⁻⁴ |
| RAW | −3.151 [−3.396, −2.908] | +1.223 [+1.028, +1.422] | 1.426 | 0.77 | 60.9 % | 6.8·10⁻³ |

* **Choosing per budget is most of the gain.** SBS-B is −0.141 [−0.205, −0.073] against the single best protocol.
  (SBS is now PEC: 1.7.0 ranks every candidate on one population, "this protocol where feasible, else RAW"; 1.6.0
  averaged each protocol over its own feasible rows, which made CDR — feasible only at larger budgets — look best;
  review S5.)
* **The frozen recommender adds −0.043 [−0.105, 0.026] — not resolved.** It was chosen as the best of 25
  configurations on these scores; the BBC-CV correction (optimism 0.031) gives −0.018 [−0.098, 0.058], better than
  SBS-B in 66 % of the bootstrap selections. No other configuration is resolved against SBS-B in its favour; the
  structured model on deployable features is resolvedly worse (+0.073 [0.011, 0.133]).
* **Where the gain is** (per budget; ten descriptive comparisons, not tests): the point estimate favours the
  recommender at 8 of 10 budgets and is resolved only at 3·10⁵ (−0.235 [−0.354, −0.109]), where the reference switches
  from PEC to extrapolation and CDR for many tasks. At every budget the oracle keeps 0.08–0.38 decades over SBS-B that no policy
  captures.
* **Tails.** The recommender's severe-misselection rate is 2.8 % (SBS-B 5.2 %, SBS 11.7 %) and P(regret > 2×) 0.17
  (0.20); the worst 5 % of its rows carry 73 % of its total squared error (SBS-B 69 %).
* **Arithmetic mean MSE (Figure 1 A) tells a different story at large budgets.** Over all budgets the recommender and
  SBS-B are equal (+1.3·10⁻⁵ [−5.8·10⁻⁴, 6.8·10⁻⁴]). At B ≥ 10⁶ one task carries 86–87 % of the recommender's
  arithmetic mean: on QFT-3 (m = 1, `fake_fez`) it chooses NIL, whose risk there is 0.065 — the task on which the
  design study (§9) finds one catastrophic design in twenty. Policies that choose ZNE-E at small budgets inherit its
  divergent trials: 8 of the 25 configurations (all five MLPs among them) have policy MSEs of 10²–10⁷ while their
  log-scale scores are ordinary.
* **By family (Figure 1 C)**: resolved in the recommender's favour only for GHZ; QFT (3 circuits) and Ising point
  toward SBS-B with wide intervals.
* **The 1.6.0 conformal abstention (`MODEL+ABSTAIN`) is inert.** Its simultaneous intervals are unbounded at α ≤ 0.2
  with 32 calibration circuits per fold, so it accepts only rows where the model already agrees with SBS-B (41 %) and
  equals SBS-B on every split. The selective rule of §8 replaces it.

## 5. Which information helps? (Figure 3; empirical and exploratory)

**Ablations (Figure 3 A; predictive importance, not causal).** Each group or semantic source removed, the model
retrained and retuned with the same inner group-4-fold and scored on held-out base circuits; Δ is the increase in mean
log10 regret (> 0: the information helps):

| removed | kind | Δ mean log10 regret |
|---|---|---|
| budget | group | +0.176 [+0.107, +0.281] |
| every simulated quantity | semantic | +0.036 [−0.016, +0.083] |
| protocol overheads | group | +0.025 [−0.025, +0.069] |
| simulator-only state | group | +0.014 [−0.010, +0.052] |
| device snapshot (noise) — equals every calibration-derived quantity | group | +0.008 [−0.014, +0.027] |
| observable | group | −0.002 [−0.009, +0.005] |
| basic structure | group | −0.006 [−0.054, +0.026] |
| light cone | group | −0.011 [−0.053, +0.014] |
| light cone with all its derived sums | semantic | −0.014 [−0.055, +0.018] |

Only the budget is resolved as necessary. Removing all simulated information points toward a loss (+0.036) without
resolving it; removing light-cone information points toward a small gain. Removing every calibration-derived quantity
is the noise group itself (shown once).

**Noise composition (Figure 3 D; D35; exploratory, three circuits, one task per cell).** At B = 10⁵, one component of
the snapshot *and* of the execution law changed at a time. The measured best differs from the circuit's baseline in 16
of 27 changed cells (12 resolved). The frozen recommender chooses PEC in all 30 cells; it differs from the measured
best in 14. Change of log10 MSE against the baseline:

| change | PEC (hea / ising / qaoa) | ZNE-L | ZNE-E | RAW | measured best (hea / ising / qaoa) |
|---|---|---|---|---|---|
| two-qubit errors × 3 | +0.26 / +0.10 / +0.11 | +0.02 / +0.22 / +0.13 | +0.05 / +0.20 / +0.19 | +0.79 / +1.47 / +0.59 | ZNE-L (unres.) / PEC / PEC |
| T1, T2 ÷ 3 | +0.38 / +0.52 / +0.77 | +0.28 / +0.10 / −0.01 | +0.32 / +0.01 / +0.08 | +0.17 / +0.21 / +0.27 | CDR (unres.) / RAW (unres.) / ZNE-L (unres.) |
| readout × 2 | +0.02 / +0.02 / +0.05 | +0.08 / +0.10 / +0.02 | +0.04 / +0.12 / +0.04 | −0.07 / +0.41 / +0.19 | PEC / PEC / PEC |
| readout × 0.5 | +0.01 / −0.02 / −0.01 | −0.01 / +0.01 / −0.03 | −0.04 / +0.04 / −0.01 | +0.07 / +0.43 / −0.12 | PEC / PEC / PEC |
| coherent fraction 50 % | +2.34 / +1.26 / +1.84 | +1.55 / +0.89 / +0.47 | +0.33 / +0.74 / +0.84 | +1.45 / +1.13 / +0.01 | ZNE-E / RO / ZNE-L |

* **H1** (raising two-qubit errors moves the preference away from PEC, as raising s does): **not supported** for
  these circuits. PEC's risk rises by only 0.10–0.26 decades at × 3 and it stays the resolved best on Ising and QAOA;
  on HEA ZNE-L overtakes it in an unresolved cell. What moves PEC most among the Pauli components is relaxation
  (T1, T2 ÷ 3: +0.38 to +0.77) — a hypothesis for the s-effect of §3, not tested here.
* **H2** (readout changes move RAW against the readout-corrected protocols, leaving their order unchanged): the second
  part holds (PEC stays best in all six readout cells); the first does not hold cleanly — on Ising, RAW's MSE rises by
  0.4 decades when the readout error is halved *or* doubled, consistent with an accidental cancellation between
  readout and gate bias at the baseline (RAW is best there only because of it).
* **H3** (a coherent fraction, which the snapshot's Pauli model does not describe, raises PEC's risk more than the
  extrapolation protocols'): **supported for these three circuits** — PEC's MSE rises by 0.8–2.3 decades, more than
  ZNE-L's, ZNE-E's or ZNE-R's in every one of the six coherent cells, and in all six the measured best is not PEC
  (resolved; PEC was the best baseline protocol on HEA and QAOA and the runner-up on Ising).
  The recommender cannot see this: no calibration feature describes coherent errors (§10).

## 6. Linear, nonlinear and structured models (empirical)

With comparable tuning (inner group-4-fold over a declared grid), boosted trees and ridge are within each other's
intervals on every feature set (best HGB −4.417, best ridge −4.402); the MLP and the structured model are the
weakest families, and every MLP configuration has catastrophic arithmetic MSEs. The **structured risk model** b² + v/((1 − f)B) (per-task floor and
sampling coefficients from cheap features; METHODS §7) is worse than SBS-B in range on deployable features (+0.073
[0.011, 0.133]) and not resolved with simulated features (+0.015); its strength is extrapolation (§7).

**Prediction quality** (supplementary prediction figure; held out): RMSE of the predicted log10(MSE + 10⁻⁶) per
protocol 0.33–0.50 decades for ZNE-R, CDR, ZNE-L, RO, PEC and RAW, 1.08 for NIL and 1.27 for ZNE-E (rare, large
failures). What matters for choosing is the order of the best two: the predicted best is evaluated better than the
predicted runner-up in 62 % of rows (65 % of the 756 rows whose evaluated difference is resolved); the correlation of
predicted and evaluated differences is 0.23. The models order protocols that differ by decades, not those that differ
by factors of two, which is where the oracle's remaining margin lies.

## 7. Where does a learned policy fail to generalize? (Figure 4; empirical)

Every split keeps each base circuit on one side. Mean log10 regret per policy (same reference everywhere) and the
paired difference of boosted trees (deployable features) and of the oracle to SBS-B:

| split | test base circuits | SBS-B | SIZE-B | ridge | structured | HGB | HGB Δ vs SBS-B | oracle Δ vs SBS-B |
|---|---|---|---|---|---|---|---|---|
| same distribution (5-fold) | 40 | 0.20 | 0.21 | 0.23 | 0.28 | 0.20 | −0.007 [−0.063, +0.042] | −0.203 [−0.259, −0.147] |
| leave one family out | 40 | 0.21 | 0.23 | 0.21 | 0.28 | 0.22 | +0.006 [−0.062, +0.074] | −0.211 [−0.270, −0.154] |
| held-out device (`fake_kyiv`) | 29 | 0.39 | 0.35 | 0.43 | 0.40 | 0.33 | −0.062 [−0.183, +0.055] | −0.391 [−0.552, −0.237] |
| larger width (test n = 6) | 7 | 0.29 | 0.30 | 0.35 | 0.32 | 0.20 | −0.096 [−0.278, +0.008] | −0.294 [−0.504, −0.157] |
| budget extrapolation (train B ≤ 3·10⁵) | 40 | 0.56 | 0.54 | 0.28 | 0.33 | 0.34 | −0.221 [−0.410, −0.053] | −0.565 [−0.758, −0.396] |
| budget interpolation (alternate grid points) | 40 | 0.21 | 0.21 | 0.21 | 0.35 | 0.31 | +0.108 [+0.034, +0.174] | −0.207 [−0.273, −0.144] |

* **Budget extrapolation is the one place where models clearly beat the fixed rules.** Above the training range,
  SBS-B can only reuse its largest training budget's ranking (mostly PEC). The best configuration (ridge on size and
  budget) is −0.263 [−0.487, −0.037] against SBS-B after BBC-CV correction, better in 98.5 % of the bootstrap
  selections.
* **Budget generalization with restricted training (Figure 4 C).** Mean log10 regret over the extrapolated budgets,
  trained on B ≤ 10⁴: ridge 0.31, structured 0.40, boosted trees 0.44, SBS-B 0.46; trained on B ≤ 3·10⁵: ridge 0.32,
  boosted trees 0.36, structured 0.37, SBS-B 0.56. Trees are constant beyond their training range; ridge's log-B trend
  carries the reversal toward extrapolation and learning protocols. The structured model's v/B term extrapolates in the
  right direction but its floor estimates cost it in range (0.34–0.43 against 0.08–0.21 for the others): the
  decomposition is not the information that makes ridge extrapolate well.
* **Everywhere else the learned policy is not resolved against SBS-B**, while the oracle keeps 0.20–0.57 decades on
  every split. On interpolated budgets boosted trees are resolvedly *worse* (+0.108).
* **More training circuits help a little (Figure 4 B):** mean log10 regret 0.271 with 8 training base circuits, 0.228,
  0.219 and 0.208 with 16, 24 and 32 (SBS on the same subsets 0.449 to 0.344).

## 8. Reliability: abstention, intervals, certificates (Figure 5, supplementary coverage figure; empirical)

**Selective recommendation, nested (review R2).** Accept the model's choice when its confidence margin
z = (P₂ − P₁)/√(s₁² + s₂²) reaches τ, otherwise SBS-B (no shots). τ is chosen by an inner group-4-fold inside each
outer training fold and applied once to the outer test fold; every base circuit is tested once.

| policy (nested, 40 base circuits) | coverage | whole-policy mean MSE | Δ mean log10 MSE vs SBS-B | Δ vs random rejection at matched coverage |
|---|---|---|---|---|
| selective (nested τ) | 0.65 | 6.9·10⁻⁴ [3.6·10⁻⁴, 1.1·10⁻³] | −0.003 [−0.043, +0.040] | −0.000 [−0.008, +0.008] |
| model alone | 1 | 7.1·10⁻⁴ | −0.043 [−0.109, +0.025] | −0.040 [−0.081, −0.007] |
| SBS-B alone | 0 | 7.0·10⁻⁴ | 0 | +0.003 [−0.040, +0.042] |

The outer folds chose τ = 0.016, 0.0006, ∞ (reject everything), 0.18 and 0.025 — the threshold is unstable at this
size. The selective policy is no better than random rejection at the same coverage and does not reach the model alone
on the log scale; the margin carries little information about failure. Rejection rates by family range from 7 % (BV)
to 48 % (GHZ) and 100 % (Grover, one circuit) (Table S1). The frozen deployment threshold `deploy_tau` = 0.131 is
recorded with the bundle; the evidence does not support using abstention to improve risk.

**Intervals (Figure 5 D).** CV-residual intervals of the log risk (no finite-sample guarantee) cover 0.93 per candidate
at nominal 0.90 (median width 2.5 decades); Gaussian residual intervals 0.92; **split-conformal** intervals over
held-out base circuits (≥ 1 − α if circuits are exchangeable) cover 0.89 on average (per fold 0.75–0.98; median width
8.5 decades); the simultaneous versions are unbounded at α = 0.1 with 32 calibration circuits per fold. Calibration is
marginal over base circuits, not per task.

**Certificates.** The Hoeffding and empirical-Bernstein radii of the clipped estimator (units: trials, or design means
of CDR and NIL) are 0.68 and 0.61 against risks of 10⁻⁷–10⁻³: none of the 1 200 rows is informative. The 2ε relation
(selection-half error) holds on all 1 200 rows — a post-hoc check, not a guarantee.

## 9. Learning designs: variability and the design study (supplementary design-variability figure; exploratory)

On the development labels, 7 % (10⁶) and 5 % (3·10⁷) of NIL's designs are at least ten times worse than their task's
median design (CDR: 0–1 %), and the largest NIL design losses are 10⁴–10⁵ × the median design's.

**Experiment B (D37)** asked whether NIL's severe losses (design MSE > 10⁻²) are a property of the protocol or of
particular designs, and whether another training allocation changes them. 20 fresh designs per variant, 10 trials
each, at B = 10⁴, 10⁶, 3·10⁷:

| task | NIL (16 circuits, 50 % of B): MSE at 10⁶ / severe designs | NIL-T32 (32 circuits) | NIL-bare (bare circuit in training) | CDR |
|---|---|---|---|---|
| QFT-3, default obs. (review case) | 9.0·10⁻³ / 1 of 20 carries 99.7 % of it | 9.1·10⁻³ / 1 of 20 | 2.8·10⁻⁵ / 0 | 1.2·10⁻⁶ / 0 |
| QFT-5, all-Z obs. (review case) | 0.24 / 85 % | 0.048 / 70 % | 0.24 / 85 % | 3.2·10⁻³ / 10 % |
| Ising-4 | 0.017 / 60 % | 8.9·10⁻³ / 50 % | 0.013 / 50 % | 5.8·10⁻⁶ / 0 |
| HEA-5 | 7.4·10⁻³ / 35 % | 2.8·10⁻³ / 0 | 7.0·10⁻³ / 30 % | 7.7·10⁻⁶ / 0 |
| QAOA-4 | 8.8·10⁻⁶ / 0 | 6.8·10⁻⁶ / 0 | 7.7·10⁻⁶ / 0 | 8.7·10⁻⁷ / 0 |

* By the interpretation declared before the run: on QFT-5 and Ising-4 the severe losses are **spread over most
  designs** — NIL at 16 training circuits is unreliable on those tasks, not unlucky; on QFT-3 they are
  **concentrated in one design**, but none of the stored diagnostics (rank, condition number, coefficient ℓ1 norm,
  label spread) separates it from the others (its condition number is *lower*), so no design-level safeguard follows
  from these diagnostics.
* Doubling the training circuits halves NIL's MSE on Ising-4, removes the severe designs on HEA-5 and lowers QFT-5
  five-fold, but does not remove the failures (on HEA-5, 15 % of the 32-circuit designs are still severe at 3·10⁷).
  Including the bare circuit in training removes QFT-3's catastrophe and changes little else. Changing NIL's training share of B (25 % or 75 %) has no consistent effect; CDR with 50 % of B on
  training lowers its error at B = 10⁴ on every task.
* CDR has no severe design except on QFT-5 with the all-Z observable (45 % of designs at 10⁴, 10 % from 10⁶).

## 10. Mismatch at equal total budget (Figure 6; review of Figure 6; exploratory)

Ten circuits under nine execution laws; the recommender, PEC's model and every allocation see only the snapshot. The
echo diagnostic spends d(B) = max(200, 2 % of B) shots out of B.

* **Mismatch hurts the snapshot-based choice at large budgets (Figure 6 A).** Mean log10 regret at 10⁵ rises from 0.07
  (none) to 0.80 and 1.01 under coherent fractions 0.25 and 0.5, 0.33 and 0.45 under ZZ crosstalk, 0.13 and 0.22 under
  drift × 1.5 and × 2; at 10⁷ to 0.69–1.00 (coherent) and 0.65–0.87 (crosstalk). At 10³ it stays below 0.16.
* **Detection is calibrated; one deployment discriminates poorly (Figure 6 B).** The flag rate under the snapshot is
  4.8 % (α = 0.05). A single deployment's |z| ranks the failures that follow (log10 regret > 0.3) above non-failures
  with AUC 0.55 at 10³, 0.64 at 10⁴, 0.59 at 10⁵, 0.70 at 10⁶ and 0.74 at 10⁷.
* **No reaction beats spending every shot on the target (Figure 6 C).** Pooled over budgets, against TARGET@B (no
  diagnostic, all B shots on the target): withholding the diagnostic's shots costs 0.004–0.030; reacting to a flag with
  ZNE-L costs 0.03–0.19 and with SBS-B 0.005–0.33 (largest under coherent errors and crosstalk). The learned
  reselection — per budget the reaction that minimized the risk on the other discrepancy types — is −0.050
  [−0.177, 0.085] under coherent errors, +0.007 without mismatch and +0.028 to +0.071 under the other perturbations
  (resolvedly worse under drift, readout drift and temporal mixing). **It is indistinguishable from the same reaction triggered at random** at the matched flag rate
  (differences ≤ 0.016 in every scenario): what helps or hurts is the reaction, not the detection. The learned reaction
  is the same for every held-out type (RO at 10³, PEC at 10⁴–10⁵, CDR at 10⁶, ZNE-R at 10⁷): a budget-only switch.
* Detecting a discrepancy does not identify its mechanism, and a reaction that would earn the diagnostic's cost — one
  robust to the detected perturbation, such as the extrapolation protocols under coherent errors (§5, H3) — was not
  found among the declared ones.

## 11. Historical final test (evaluated once; historical, not confirmatory)

The 36 tasks (12 base circuits) of the 1.6.0 final test, relabeled with fresh designs, every policy trained on the 120
development tasks, evaluated once (`final/EVALUATED`). Its logical circuits were seen by 1.6.0, so it is not an
untouched test (review R5); D39 fixed the comparators before it was evaluated.

| policy | mean log10 MSE | Δ vs SBS-B | mean log10 regret | P(regret > 2×) | mean MSE |
|---|---|---|---|---|---|
| per-task oracle | −4.424 [−4.664, −4.092] | −0.344 [−0.457, −0.239] | 0 | 0 | 7.5·10⁻⁴ |
| **frozen: HGB, deployable + simulated** | −4.197 [−4.532, −3.767] | −0.117 [−0.225, −0.005] | 0.227 | 0.27 | 1.8·10⁻³ |
| HGB, deployable (D39) | −4.174 [−4.474, −3.759] | −0.093 [−0.157, −0.029] | 0.250 | 0.23 | 1.3·10² |
| selective (frozen τ; coverage 0.55) | −4.151 [−4.465, −3.733] | −0.071 [−0.159, +0.002] | 0.273 | 0.30 | 1.9·10⁻³ |
| ridge, deployable (D39) | −4.150 [−4.421, −3.794] | −0.069 [−0.150, +0.001] | 0.274 | 0.32 | 1.7·10⁻³ |
| structured, deployable (D39) | −4.125 [−4.381, −3.782] | −0.045 [−0.143, +0.074] | 0.299 | 0.31 | 2.2·10⁻³ |
| SBS-B | −4.080 [−4.364, −3.690] | 0 | 0.344 | 0.34 | 2.3·10⁻³ |
| SIZE-B | −4.056 [−4.375, −3.639] | +0.024 [−0.028, +0.105] | 0.368 | 0.36 | 2.7·10⁻³ |
| SBS (PEC) | −3.967 [−4.300, −3.556] | +0.114 [+0.018, +0.211] | 0.458 | 0.39 | 2.5·10⁻³ |
| RAW | −3.211 [−3.621, −2.729] | +0.869 [+0.641, +1.088] | 1.213 | 0.74 | 6.6·10⁻³ |

The frozen recommender is resolvedly ahead of SBS-B on these 12 circuits (1.6.0: −0.106 [−0.214, 0.002]); the
deployable-feature trees too, but their arithmetic MSE is dominated by one ZNE-E choice whose evaluation half holds a
divergent trial (MSE 4.8·10⁴ on one row). Because these circuits were seen by 1.6.0, this is a historical
re-evaluation, not evidence for the claim.

## 12. Confirmation (the confirmatory statement of this release)

25 new logical circuits (8 families, 3–6 qubits; 50 tasks, mostly `fake_fez`), registered in the ancestry registry
before labeling (no overlap with any earlier study), labeled after the freeze with the corrected protocol, every
policy trained on the development and historical-final tasks, evaluated once (`confirm/EVALUATED`). Primary: the
frozen configuration against SBS-B (D30, D39).

| policy | mean log10 MSE | Δ vs SBS-B | mean log10 regret | P(regret > 2×) | mean MSE |
|---|---|---|---|---|---|
| per-task oracle | −4.556 [−4.710, −4.332] | −0.220 [−0.317, −0.141] | 0 | 0 | 7.2·10⁻⁴ |
| **frozen: HGB, deployable + simulated (primary)** | −4.423 [−4.604, −4.178] | **−0.087 [−0.161, −0.024]** | 0.133 | 0.16 | 1.4·10⁻³ |
| HGB, deployable | −4.417 [−4.601, −4.169] | −0.081 [−0.150, −0.026] | 0.139 | 0.16 | 1.7·10⁻³ |
| selective (frozen τ; coverage 0.49) | −4.402 [−4.578, −4.160] | −0.066 [−0.141, −0.015] | 0.154 | 0.18 | 1.2·10⁻³ |
| SIZE-B | −4.341 [−4.537, −4.096] | −0.006 [−0.098, +0.065] | 0.214 | 0.23 | 1.1·10⁻³ |
| SBS-B | −4.336 [−4.547, −4.033] | 0 | 0.220 | 0.22 | 2.1·10⁻³ |
| ridge, deployable | −4.332 [−4.542, −4.030] | +0.004 [−0.031, +0.039] | 0.224 | 0.27 | 2.0·10⁻³ |
| structured, deployable | −4.280 [−4.484, −3.981] | +0.056 [−0.004, +0.118] | 0.276 | 0.30 | 2.0·10⁻³ |
| SBS (PEC) | −4.216 [−4.452, −3.909] | +0.120 [+0.046, +0.204] | 0.340 | 0.32 | 1.7·10⁻³ |
| RAW | −3.290 [−3.586, −2.914] | +1.045 [+0.821, +1.303] | 1.265 | 0.72 | 1.2·10⁻² |

**Confirmatory statement.** On these 25 circuits the frozen recommender's geometric-mean MSE is 18 % lower than
SBS-B's (Δ = −0.087 [−0.161, −0.024] decades), it closes 61 % of the gap between SBS and the per-task oracle, and its
arithmetic mean MSE is lower too (1.4·10⁻³ against 2.1·10⁻³). Per budget (descriptive) the advantage is largest from
3·10⁵ on (−0.12 to −0.16). The secondary comparators are exploratory.

**Against the plan's criterion** (RESEARCH_PLAN: a resolved advantage over SBS-B on held-out development circuits
*and* on the confirmation set): the confirmation is resolved, the development estimate is not (−0.043
[−0.105, 0.026]; BBC −0.018 [−0.098, 0.058]). **The claim is therefore not established at pilot scale.** The two
estimates are compatible with a small advantage of about 0.04–0.09 decades (10–20 % in geometric-mean MSE); the
oracle's margin of 0.20–0.34 decades on all three sets shows that per-task information exists that the pilot's models
do not capture. The larger study (§14) is designed to resolve differences of this size.

## 13. Deployment: the recommender and its cost

The application and `python -m qemx.selection recommend` use the same versioned bundles (`qemx/selection/bundles/`):
`qemx-rec-1.7.0` (the frozen configuration; simulated features, computed only when allowed) and
`qemx-rec-1.7.0-deployable` (deployable features only, the default because it never simulates the circuit). Each holds
the preprocessing, feature order, candidate definitions, training support, the fallback SBS-B and the statement kind
of every output, without pickle, and a validation summary of its own configuration: the out-of-fold development
result and the confirmation (−0.087 and −0.081 against SBS-B). Only the frozen bundle has a validated selective
threshold (τ = 0.131, with the nested evaluation of §8); the deployable-only bundle has none, so its recommendation is
the model's choice, and it says so. Out-of-support tasks are reported as such.

**Cost of a recommendation** (supplementary runtime figure; median of repeats on the shared build machine — magnitudes,
not a benchmark): compilation 0.011–0.013 s, deployable features 0.017–0.042 s (n = 3–7), inference ≈ 0.02 s;
the simulated features add 0.33 s at n = 3 to 1.16 s at n = 7 and grow as 4ⁿ in memory. Producing one study label
takes 71–103 s per task (n = 3–6) — the study's cost, not the user's.

## 14. Compute, and what was not executed

* **`full17.yaml`** (242 logical circuits, 726 tasks: 203 development and 39 final logical circuits, 5 devices, widths
  3–7, 3 variants, the corrected design plan): `plan --measured reports/selection_v170/dev` estimates 19.1 CPU-hours of
  labeling and 130 MB of disk (width 7 extrapolated), within the configured caps of 80 CPU-hours and 10 GB. Resumable
  and bound to its run. With 203 development circuits the paired SE against SBS-B would shrink by about √5, enough to
  resolve the 0.04-decade difference of §4 if it is real; and the simultaneous split-conformal level becomes
  attainable (≥ 79 calibration circuits per fold at α = 0.1, K = 8).
* **Hardware validation** (`hardware17.yaml`, `reports/selection_v170/hardware/`): two tasks (Ising-4, GHZ-4 on
  `ibm_fez`), RAW, RO, the bundle's recommendation and SBS-B (both PEC) at B = 20 000, four blocks: 24 jobs, 452 s
  reserved on the conservative usage estimate of the 540 s available under the shared 600 s ceiling minus a 60 s
  reserve. Not executed: it needs the owner's authorization of the plan's digest, the campaign ledger and credentials
  (`docs/selection/HARDWARE.md`). A hardware result would be labelled as hardware observations, never merged with the
  emulated labels.

## 15. Assessment of the 1.6.0 results

| 1.6.0 claim | 1.7.0 evidence | status |
|---|---|---|
| Choosing per budget (SBS-B) beats the single best protocol by −0.165 decades | −0.141 [−0.205, −0.073]; SBS is now PEC (one population, S5) | **survives** |
| The frozen recommender is −0.058 [−0.125, 0.017] (dev) and −0.106 [−0.214, 0.002] (final) against SBS-B, resolved on neither | dev −0.043 [−0.105, 0.026]; historical final −0.117 [−0.225, −0.005]; confirmation −0.087 [−0.161, −0.024] | **direction survives; the conclusion "not established" stands** (now by the development estimate; the final test is historical) |
| The per-task oracle leaves 0.23–0.36 decades | 0.20 (dev), 0.34 (final), 0.22 (confirmation) | survives |
| On arithmetic MSE the recommender is not ahead; one NIL choice dominates | equal on dev, ahead on the confirmation; the same QFT-3 NIL choice dominates at B ≥ 10⁶; experiment B traces it to one catastrophic design | survives, explained |
| NIL never wins; its error is flat in B at 16 training circuits | 1 of 1 200 rows; flat in B; and its typical risk was understated by the shared designs (1.3–1.9 ×) | survives, strengthened |
| CDR reaches MSE 10⁻⁶ at 3·10⁷ (median task, point estimate) | not reached by the median task with design-level labels | **changed** |
| Only the budget is a resolved necessary descriptor | +0.176 [0.107, 0.281]; semantic ablations add no resolved source | survives |
| Gradient-boosted trees are not better than ridge | within intervals on every feature set | survives |
| Learned models clearly win only in budget extrapolation; ridge extrapolates better than trees | BBC −0.263 [−0.487, −0.037]; ridge best on restricted training; the structured model is between | survives |
| The cross-fitted selective rule is no better than random rejection | nested: −0.000 [−0.008, 0.008] vs random; also worse than the model alone | **survives under the stricter nested evaluation** |
| The conformal abstention equals SBS-B | inert on every split | survives |
| Certificates are vacuous at these risks | radii 0.61–0.68 against risks ≤ 10⁻³ | survives |
| The recommender barely reacts to the error scale; regret grows with s on HEA | 63 % agreement on resolved cells; HEA 0.07 → 0.67 | survives |
| The echo diagnostic is calibrated, but neither declared reaction helps | calibrated (4.8 %); at equal budget no reaction beats TARGET@B, and the learned reaction equals random triggering | survives, sharpened |
| Trial-level SEs and the 1.6.0 final test as an untouched test | design-level SEs 1.1–2.5 × larger; the final test is historical | **corrected** (R1, R5) |

## 16. Additional experiments: why, what they cost, how to read them

| experiment | scientific uncertainty it resolves | measured cost | interpretation declared before running → outcome |
|---|---|---|---|
| Corrected pilot (`pilot17`, dev + historical final) | Do the 1.6.0 conclusions survive independent learning designs (R1)? | 5.5 CPU-h labels + ≈ 2 h analysis | Non-learning protocols identical trial for trial (they are); a conclusion that changes with the CDR/NIL estimand was carried by the design reuse → only the CDR precision claim changed (§15) |
| Nested selective evaluation | Does abstention beat random rejection when its threshold never sees the test circuits (R2)? | ≈ 26 min wall | Useful only if the whole-policy MSE beats both the model alone and random rejection at matched coverage → it beats neither: not established |
| Structured risk model | Does a physics-informed b² + v/B model generalize across budgets better than black-box models? | part of the analysis | Better budget extrapolation than ridge and trees would support the decomposition → better than trees when trained on B ≤ 10⁴, worse than ridge everywhere, worse in range: not supported |
| Semantic ablations | Is light-cone, calibration or simulated information *as a whole* needed? | part of the analysis | An interval containing 0 means not shown to matter here → none resolved |
| Noise composition (`noisecomp17`) | Does the composition of the noise, not only its level, change the best protocol? | 0.4 CPU-h labels | H1–H3 supported only where resolved; a mechanism for these circuits, not a law → H3 supported, H1 not, H2 in part (§5) |
| Equal-budget mismatch reactions (`mismatch17`) | Does a diagnostic pay for its shots when reactions are trained without the held-out discrepancy type? | 1.9 CPU-h labels | A reaction earns its cost only if it beats TARGET@B and the same reaction triggered at random → none does |
| Design study (experiment B) | Are NIL's severe failures properties of NIL or of particular designs, and does another allocation fix them? | ≈ 2 CPU-h | Concentrated with distinguishable diagnostics → safeguard possible; spread → protocol unreliable at this size → spread on two tasks, concentrated but indistinguishable on one (§9) |
| Confirmation (`confirm17`) | Does the frozen recommender's advantage (or its absence) hold on new logical circuits, evaluated once? | 1.8 CPU-h labels, 3 min evaluation | The only confirmatory statement → −0.087 [−0.161, −0.024] in the recommender's favour (§12) |

## 17. Limitations and threats to validity

* **Emulation, not hardware.** Fake-device snapshots executed exactly, one realization of each; the mismatch, regime
  and noise-composition laws are the only departures. Nothing here transfers to hardware without a hardware study.
* **Small pilot.** 40 development base circuits, 12 historical-final and 25 confirmation circuits; per-budget, per-family
  and per-split statements are descriptive; the width split has 7 test circuits; controlled studies have one circuit
  per family and one task per cell.
* **Few design clusters.** K = 10 designs per half makes the design-level SE itself uncertain; NIL with 16 training
  circuits (the paper uses 256–5 000).
* **Narrow noise in training.** Three real snapshots; the recommender has no feature that describes coherent errors
  or crosstalk and does not respond to the error scale or composition; its maps (Figure 2) describe this training set.
* **Selection among configurations.** The best of 25 configurations is optimistic on development scores; BBC-CV and
  the confirmation address it, the per-budget and per-split columns do not.
* **Multiplicity.** Per-budget, per-family and per-cell intervals are pointwise; with 30 noise-composition cells about
  one or two "resolved" changes are expected by chance (the Ising readout cells may be such a case).
* **Decisions after results.** D37 reduced the design study from 30 to 20 designs per variant after timing three
  designs (before any result); D39 added the comparators of the once-only evaluations before either was evaluated.
  Every figure-layout change after the first drawing changed presentation only (the figure validator compares the
  plotted numbers with the tables).
