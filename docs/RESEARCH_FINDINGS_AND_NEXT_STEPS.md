# Research findings and next steps — QEM Explorer 1.9.0

This is my interpretation of the evidence after 1.9.0, for the thesis question: **which attainable information makes
budget-aware selection of complete QEM protocols reliably better than the best fixed protocol per budget (SBS-B), and
when do missing context or unstable protocols defeat it?** Part I covers what 1.9.0 adds and how I now read the whole
body of evidence; Part II is the 1.8.0 interpretation, kept as written (its ranked experiments are superseded by §I.6).

**Everything is simulated**: emulated trials on IBM fake-device snapshots (exact density-matrix engine), 3–6 qubits.
Δ is the base-circuit mean of log10(R + 10⁻⁶) for a policy minus SBS-B; a Δ multiplies the geometric mean of
(R + 10⁻⁶) by 10^Δ (−0.144 ≈ a factor 0.72). Intervals are 95 % bootstrap intervals over base circuits.

## Part I — after 1.9.0

### I.1 What changed in the evidence, and what did not

* **No 1.8.0 number moved.** The confirmation's per-trial arrays reproduce the saved labels; the two primary contrasts
  are recomputed from the arrays (−0.14406 frozen, −0.08397 default bundle; equal to the saved replays to 10⁻¹³, the
  differences being summation order); the SBS-B
  column of the confirmation is reproduced on all 580 rows both from the bundles' ranks and by refitting the SBS-B rule
  on the development labels. The defects found by the reviews (completion, caps, resumption, replay binding, YAML
  semantics, state/study parity) did not touch the shipped, complete, single-sequence run.
* **One reporting error, corrected.** The supplementary deployment-policy note described the default (deployable)
  bundle as the confirmed one. The confirmed configuration (D41) is the frozen bundle with simulator-only features and
  simulation allowed; the deployable bundle is secondary and not robustly resolved (`docs/selection/ERRATA_v180.md`).
  The first review's other criticism of that note (the 47 rows) was not a defect: 33 accepted alternatives and 14
  out-of-distribution fallbacks, mechanisms that are not exclusive, were already stated.
* **The estimand is now explicit.** The labels — and therefore every replay — score the *fresh-design average risk*
  R^fresh of the returned candidate: for a learning candidate, the risk averaged over new training designs, not the
  risk of the one design the recommender returns. This is a defined, legitimate benchmark (the second review is right
  that it is not a defect), but it is not what a user executes. 1.9.0 measures the gap directly (§I.2 A).

### I.2 New observations

**A. Execution audit (exploratory; `reports/execution_audit_v190/`).** For the eight circuits of the 1.8.0 root
audit — 1.7.0 regression cases, earlier failure types and controls; inspected, but in neither the bundles' training
set nor the confirmation — at three budgets (10⁴, 3·10⁵, 10⁷) and five deployment roots, the whole chain was executed:
`recommend` → the returned verified design → the study's allocation → 600 executions; SBS-B and RAW the same way.

| policy | executed − SBS-B | fresh-design − SBS-B | executed − fresh (contrast) |
|---|---|---|---|
| frozen bundle (primary) | −0.248 [−0.313, −0.178] | −0.223 [−0.289, −0.156] | −0.024 [−0.047, +0.000] |
| default deployable bundle | −0.117 [−0.224, +0.013] | −0.122 [−0.199, −0.041] | +0.005 [−0.036, +0.061] |
| RAW | +1.029 | +1.040 | −0.011 [−0.030, +0.005] |

Budget ledgers: 480 rows, none over budget, none with unspent shots. Learning candidates were returned on only 10 of
the 480 rows (median log10 ratio executed/fresh −0.009, 10–90 % range −0.109 to +0.037).

*Reading.* Scoring R^fresh instead of the returned design's risk did not flatter the policies on these circuits: the
contrast moves by at most a few hundredths of a decade, in the policies' favour for the frozen bundle. The reason is
mechanical: the policies mostly return candidates without training designs (PEC/PEC-TW, ZNE variants, readout
variants, SV), whose fresh-design and returned-design risks coincide up to sampling. The audit is therefore **weak
evidence about learning candidates** (CDR, NIL, FR), exactly where the two risks can differ most — their between-design
share of the label variance is 0.5–0.9 at large budgets (1.8.0 Part II obs. 6; pilot design_variance.csv: CDR 0.92 at
3·10⁷). Note also that the default bundle's *executed* interval includes 0 on these eight circuits although its fresh
interval does not; with eight circuits this is consistent with its known fragility, not a new effect.

**B. The 1.9.0 pilot (exploratory, for planning; `reports/pilot_v190/`).** 27 new base circuits, 54 tasks, five
families, widths 3–5, four budgets (10³–3·10⁷), R = 24 trials per half, 3 fresh designs per half, **noise-diverse
training**: six calibration-supplied noise settings (native, overall scale s = 0.5 and 2, two-qubit errors g2 = 2,
relaxation t = 2, readout ro = 2) on three base devices. Out-of-fold results against SBS-B (overall row):

| split | best learned policy | other learned policies | ALL-CANDIDATE (reference) |
|---|---|---|---|
| same distribution (5 folds) | hgb, deployable + simulated (the freeze): +0.000 [−0.054, +0.056] | deployable only: hgb +0.086 [+0.017, +0.162], ridge +0.096 [+0.027, +0.164], enet +0.114 [+0.056, +0.173]; struct +0.14 to +0.19 | −0.232 [−0.311, −0.165] |
| held-out noise setting | none better than SBS-B; hgb on size/budget +0.047 [+0.014, +0.085] | deployable: ridge +0.073, hgb +0.103 (both resolved worse) | −0.248 |
| held-out base device | hgb deployable −0.003 [−0.053, +0.053] | others +0.04 to +0.07 | −0.240 |
| joint (HEA under g2 = 2; 3 circuits) | ridge size/budget −0.096 [−0.134, −0.063] | hgb deployable +0.186 | −0.071 |

Optimism (BBC-CV) of choosing the best configuration out of fold: 0.03 decades on the same-distribution split; the
corrected difference of the frozen configuration against SBS-B is +0.024 [−0.060, +0.117]. Learning curve (the
learned policy minus the single best protocol, SBS): +0.100, +0.065, +0.087, +0.057 at ≈ 6, 11, 16 and 22 training
circuits (two repeats, one at the last point) — positive (worse) throughout, its shape not resolved.
Nested selective evaluation of the frozen configuration (outer 5, inner 4 folds; τ = 0.124): at 30 % coverage the
selective policy is +0.027 [−0.011, +0.068] against SBS-B and indistinguishable from random rejection at the same
coverage (−0.004 [−0.026, +0.010]). Portfolio comparison on identical pilot labels: SBS-B over 17 instead of 8
candidates −0.018 [−0.042, +0.009] (1.8.0: −0.070, resolved); the attainable reference over 17 instead of 8 −0.030
[−0.052, −0.007]; learned deployable-feature policies against SBS-B **within** the 8-candidate portfolio +0.066 (ridge)
and +0.039 (hgb), neither resolved, but **within** the 17-candidate portfolio +0.096 [+0.026, +0.163] and +0.086
[+0.014, +0.158], both resolved worse — the same models lose more when they must choose among more candidates.

**C. Who wins depends on the noise setting — but not robustly enough to use at this size.** In the pilot's evaluation
half the most frequent winner at 3·10⁷ shots is ZNE-E under native noise (7 of 18 tasks), CDR with doubled two-qubit
errors (4 of 10) and PEC at doubled overall error (5 of 9); at 10⁶, PEC under g2 = 2 but ZNE-E under native noise. A
budget rule refitted *within* each noise setting (a noise-aware SBS-B, leave-one-circuit-out; my own exploratory
computation on `analysis/labels.csv`, not a saved table) is **worse** than the pooled SBS-B: +0.069 [+0.032, +0.108],
and worse in every noise setting (+0.04 to +0.14). The winners differ, but 4–18 tasks per setting cannot rank 17
candidates reliably.

**D. Costs (measured; two shared Linux cores).** Labeling: 0.86 CPU-hours for the pilot's 54 tasks (27 minutes with two workers; median 39 / 51 /
67 s per task at 3 / 4 / 5 qubits). The pilot's analysis (four model families × three feature sets on the
same-distribution split, two × two on three further splits, design diagnostics and learning curve): 71 minutes
single-threaded, plus 20 minutes for the nested evaluation. A recommendation (pilot profile, medians): deployable
features 0.02–0.04 s, inference 0.04–0.05 s, simulator-only features 0.29 / 0.53 / 0.71 / 1.05 s at 3 / 4 / 5 / 6
qubits — the simulated features are almost the whole cost, growing by a factor 1.3–1.8 per added qubit here.

### I.3 How much each observation can bear

* **The 1.8.0 confirmation remains the only confirmatory evidence**, with its three qualifications (D41 fixed after
  development results were seen; the BBC-corrected development difference not resolved; the established configuration
  needs exact simulation). 1.9.0 adds no confirmatory evidence and should not be read as adding any.
* **Execution audit:** eight inspected circuits, five roots. It can show that the estimand difference is *small for the
  candidates the policies actually return*; it cannot bound it for learning candidates.
* **Pilot:** 27 base circuits, exploratory, used for planning. Its negative learned-policy results are resolved against
  SBS-B on the same-distribution and held-out-noise splits, but the pilot has fewer circuits, fewer trials (R = 24 vs
  60) and fewer designs (3 vs 6 per half) than 1.8.0's development set, so its labels are noisier; the joint split
  rests on three circuits and says nothing.
* **Noise-stratified rule (I.2 C):** one exploratory computation, no multiplicity correction, leave-one-circuit-out on
  pilot labels; it is a *diagnosis* of why learning fails at this size, not a result about noise-aware selection.

### I.4 Interpretation

**1. The attainable gain is large; most of what is captured comes from the budget and the portfolio.** On the 1.8.0
confirmation the per-task reference (ALL-CANDIDATE: best on the selection half, scored on the evaluation half) is
−0.301 [−0.406, −0.210] below SBS-B; SBS-B itself closes 41 % of the gap between the single best protocol and that
reference; the confirmed full policy closes about another half of the remainder (−0.144, i.e. ≈ 48 % of the SBS-B →
reference gap), the deployable one 28 % (−0.084). In the pilot the reference is −0.23 below SBS-B and no
learned policy closes any of it. In the vocabulary of algorithm selection (Rice 1976; Kerschke et al. 2019), the
single-best-solver-per-budget is a strong baseline and the virtual-best gap is real but hard to capture.

**2. Why learning on deployable features fails at this sample size — an estimation problem before an information
problem.** Three observations point the same way. (i) The winners *do* change with calibration-visible noise (I.2 C),
so the information is not absent. (ii) A rule that uses that information in the simplest possible way (a separate
budget ranking per noise setting) loses against the pooled ranking, because each ranking is estimated from a handful
of tasks with noisy labels; the learned models face the same variance with more parameters. (iii) The pilot's learning
curve gives no evidence of a plateau at 22 training circuits; neither it nor 1.8.0's (fixed native noise; −0.128,
−0.124, −0.170 against SBS at 18, 26 and 35 circuits) has a resolved shape — four points, one or two repeats each.
Selecting among K = 17 candidates multiplies the chance of picking a candidate whose estimated advantage is noise: the
expected optimism of a maximum over K noisy estimates grows roughly like σ·√(2 ln K), so with K = 17 a per-cell ranking
needs several times the circuits a two-candidate comparison does. (iv) The pilot's portfolio comparison shows exactly
this: the same learners lose +0.04–0.07 against SBS-B when choosing among 8 candidates and +0.09–0.10 among 17. The near-ties of the root audit (one winner under all
four roots in 10 of 24 cells, at negligible regret) say the same from the label side. My reading is that the
deployable-feature result is **data-limited under noise diversity** and possibly **information-limited under native
noise** (where the device features hardly vary); fulldev19 (82 base circuits, 6 designs, R = 60) is designed to
separate the two.

**3. Why simulator-only features help.** The features that need a simulation of the supplied noise model (noisy
expectation values, acceptance probabilities, the size of the ideal signal relative to its noisy version) predict the
regime in which extrapolation or inversion amplifies noise into large relative error — ZNE-E's divergences, PEC's
sampling overhead relative to |⟨O⟩| — per circuit, which calibration-level descriptors cannot. This remains a
hypothesis (Part II §4); the permutation importance of those features on 1.8.0 development data is still the cheapest
test. It is also the practical bottleneck: the configuration that works costs an exact simulation per circuit
(≈ 0.65 s at 3–6 qubits, exponential in width), so its *deployable* value at scale is unknown.

**4. Physical regimes visible in the labels.** The budget dependence follows the bias–variance balance of each
protocol: at small B the variance amplification of extrapolation and inversion dominates (at 10³ mostly RAW and
readout variants win, with PEC/PEC-TW on some circuits); at intermediate B, PEC/PEC-TW's unbiasedness under a well-matched Pauli model pays for its
sampling overhead γ²/B (PEC's overhead grows exponentially in the circuit's total error rate, so doubling g2 or s moves
the crossover budget, as the pilot's per-setting winners show); at the largest budgets the residual model bias of PEC
(a depolarizing inverse at a median rate, not the snapshot's actual channels; van den Berg et al. 2023 show how a
learned sparse Pauli–Lindblad model reduces exactly this) and the curvature bias of ZNE decide, and learning-based CDR
wins where its training design is stable. General lower bounds on the sampling cost of any error mitigation that grows
exponentially with depth and error rate (Takagi et al. 2022; Quek et al. 2024) imply that at larger widths the
question shifts from *which* protocol to *whether any* beats RAW at the given budget — a regime the 3–6-qubit
emulation cannot reach and the thesis should state as a limit of scope. Cai et al. (2023) review the protocols;
Bultrini et al. (2023) found, as here, that the best method depends strongly on the shot budget and the noise.

**5. Relation to machine-learned QEM.** Liao et al. (2024) use machine learning to *perform* mitigation (learning the
map from noisy to mitigated values, including emulating ZNE) on hardware. This study learns to *choose* among complete
protocols under a fully charged shot budget; the two are complementary, and FR here is a small instance of the former
(its failure near p ≈ N — 17 unknowns, 16 training circuits at n = 5 — is the interpolation-threshold behaviour of
least squares; Hastie et al. 2022). A ridge-regularized FR is the obvious repair (§I.6, experiment 6).

### I.5 What is still open

* Whether learning on deployable features beats SBS-B with enough circuits under noise diversity (fulldev19).
* How large the fresh-design vs returned-design difference is for learning candidates (execution audit on fulldev19).
* Coherent errors: invisible to the Pauli descriptors (1.8.0 F10); fulldev19 adds a few coherent-fraction tasks, but no
  coherent-error feature exists yet.
* Mismatch reactions with 17 candidates (Figure 6 pending); VD/M3 (no faithful finite-shot path); hardware transfer.

### I.6 Ranked next experiments

Ranked by expected information for the thesis question per CPU-hour. Each is declared (configuration committed and
registered) before its labels exist.

| rank | experiment | design and decision rule to declare | cost | what it decides |
|---|---|---|---|---|
| 1 | **fulldev19** (designed, not run) | 82 base circuits, 246 tasks, seven noise settings (native and six modifiers, incl. a coherent fraction), R = 60, 6 designs/half, 10 budgets; splits same-distribution, noise, device-base, joint, width, budget extrapolation. Declare: deployable learners vs SBS-B, BBC-corrected, on the same-distribution **and** held-out-noise splits; the learning-curve slope at 40–65 circuits | 3–9 CPU-h labels (measured bases; conditional: the coherent tasks are unmeasured); analysis **not estimated in time** — 4× the pilot's model fits and 51× its fits × training rows (1.9.1, `plan`); time the same-distribution split first | data-limited vs information-limited (I.4 2) |
| 2 | **Resize conf19 before the freeze** | between-circuit SD of per-circuit Δ on the 1.8.0 confirmation: 0.21 (frozen) and 0.25 (deployable). Circuits for 80 % power at two-sided 5 %: Δ = −0.10 → 35 / 48; −0.08 → 55 / 75; −0.05 → 140 / 191. conf19 as designed (32 circuits, five families — no QFT/BV/Grover) has ≈ 45–58 % power at −0.08. State the meaningful effect and the population (options A–C in `docs/selection/RESEARCH_PLAN_v19.md`), then size conf19 (≈ 75 circuits ≈ 2–6 CPU-h) | minutes to decide | a confirmation that can fail informatively |
| 3 | **Execution audit on fulldev19 circuits** | the joint audit on circuits and budgets where CDR/NIL/FR are returned (from 10⁵ upward: half of the 1.9.0 audit's learning rows were at 3·10⁵); rule: executed − fresh contrast within ±0.05 | ≈ 1 CPU-h | whether R^fresh is an adequate target for learning candidates |
| 4 | **Noise-aware budget rules with shrinkage** | SBS-B per noise setting shrunk toward the pooled ranking (empirical Bayes), and a two-feature rule (budget × estimated PEC overhead `pec_log_gamma`); on fulldev19 out of fold | minutes | whether the heterogeneity of I.2 C is usable with simple, low-variance rules |
| 5 | **Permutation importance of the simulated features** (1.8.0 dev) | grouped, retuned | minutes | which simulated quantities carry the gain; candidates for cheap surrogates |
| 6 | **Ridge FR at the interpolation threshold** | FR at n = 5 with 16 / 32 / 64 training circuits, OLS vs ridge | ≈ 0.5 CPU-h | confirms or refutes the mechanism; a usable FR |
| 7 | **Cheap SV verification** | analytic lower bound on SV's acceptance from light-cone fault weights in the design-level preflight | hours of coding | recovers part of the default bundle's preflight loss on symmetric circuits |
| 8 | **17-candidate mismatch study** | `mismatch17` with portfolio-1.8 and policy/2 | ≈ 2 CPU-h | Figure 6 |
| 9 | **Hardware validation** (owner authorization) | the existing dry-run plan (28 jobs, 524 s of 540 s) | 9 QPU-minutes | first transfer check (two tasks; descriptive) |

### I.7 Threats to validity (additions to Part II §8)

Noise diversity by calibration-supplied modifiers of three base snapshots is still a narrow model of real device
variation (no drift, no crosstalk model, coherent errors only as a fixed fraction); the pilot's labels are noisier than
the full design's; the noise-stratified computation (I.2 C) is post hoc; the execution audit uses inspected circuits.

### References (checked for this note)

* J. R. Rice, "The algorithm selection problem", *Advances in Computers* 15, 65–118 (1976).
* P. Kerschke, H. H. Hoos, F. Neumann, H. Trautmann, "Automated algorithm selection: survey and perspectives",
  *Evolutionary Computation* 27(1), 3–45 (2019), doi:10.1162/evco_a_00242.
* I. Tsamardinos, E. Greasidou, G. Borboudakis, "Bootstrapping the out-of-sample predictions for efficient and accurate
  cross-validation", *Machine Learning* 107, 1895–1922 (2018), doi:10.1007/s10994-018-5714-4 (BBC-CV).
* Z. Cai et al., "Quantum error mitigation", *Rev. Mod. Phys.* 95, 045005 (2023), doi:10.1103/RevModPhys.95.045005.
* R. Takagi, S. Endo, S. Minagawa, M. Gu, "Fundamental limits of quantum error mitigation", *npj Quantum Information*
  8 (2022), doi:10.1038/s41534-022-00618-z.
* Y. Quek, D. Stilck França, S. Khatri, J. J. Meyer, J. Eisert, "Exponentially tighter bounds on limitations of quantum
  error mitigation", *Nature Physics* 20, 1648–1658 (2024).
* E. van den Berg, Z. K. Minev, A. Kandala, K. Temme, "Probabilistic error cancellation with sparse Pauli–Lindblad
  models on noisy quantum processors", *Nature Physics* (2023), doi:10.1038/s41567-023-02042-2.
* D. Bultrini et al., "Unifying and benchmarking state-of-the-art quantum error mitigation techniques", *Quantum* 7,
  1034 (2023), doi:10.22331/q-2023-06-06-1034.
* H. Liao et al., "Machine learning for practical quantum error mitigation", *Nature Machine Intelligence* (2024),
  doi:10.1038/s42256-024-00927-2.
* T. Hastie, A. Montanari, S. Rosset, R. J. Tibshirani, "Surprises in high-dimensional ridgeless least squares
  interpolation", *Annals of Statistics* 50(2) (2022).

## Part II — the 1.8.0 study (historical interpretation, kept as written)

This note interprets the 1.8.0 study for the thesis: what was observed, how strong the statistical evidence is, which
physical or statistical mechanisms explain the observations (and how well the data support each explanation), what
remains speculation, which results were negative, and which experiments should come next, ranked. The numbers, their
definitions and their tables are in `docs/selection/RESULTS.md` (Part I); the decisions D40–D43 in
`docs/selection/RESEARCH_PLAN.md`; the figures in `docs/selection/FIGURES.md`.

**All observations are simulations**: emulated trials on IBM fake devices (calibration snapshots executed by an exact
density-matrix engine), 3–6 qubits, eight circuit families, budgets 10³–3·10⁷ shots. No hardware result exists. Δ is a
paired difference of the mean over base circuits of log10(MSE + 10⁻⁶); a difference Δ multiplies the geometric mean of
(MSE + 10⁻⁶) by 10^Δ.

### 1. Status of the evidence

| level | what it covers in 1.8.0 |
|---|---|
| **confirmatory** (declared, evaluated once) | D41: the full deployed policy of the frozen configuration against SBS-B on 29 new logical circuits — Δ = −0.144 [−0.231, −0.076]; together with the resolved development difference (−0.088 [−0.145, −0.027]) the declared rule is met |
| **empirical** (held-out development circuits, out of fold) | model comparisons, generalization splits, ablations, learning curve, reliability, tails |
| **exploratory** | portfolio comparison, crossed root audit, regime and noise-composition studies, per-family and per-budget breakdowns, the default bundle's full policy, everything marked *post hoc* |
| **independently reproduced** | seven confirmation comparisons, both full deployed policies, the default bundle's model component and the attainable portfolio contrast (point estimates to ≤ 1.8·10⁻¹⁴), by separate code that did not use the project's; not the development, tails, audit, controlled-study or cost numbers |

Three facts limit how far the confirmatory result reaches. (i) D41 fixed the *estimand* (full deployed policy rather
than the model component) after the development results were seen, although before any confirmation result; the
configuration, the freeze rule and the confirmation set were fixed before. (ii) The optimism-corrected development
difference (BBC-CV) is −0.064 [−0.143, +0.013]: a rule based on it would not be met. (iii) The established
configuration uses simulator-only features, i.e. an exact simulation of the supplied noise model at recommendation
time; it says nothing about a recommender that cannot simulate.

### 2. Observations

1. **Deployed policy against SBS-B (confirmation).** Frozen bundle −0.144 [−0.231, −0.076]; its model component
   −0.159 [−0.253, −0.086]; default bundle (no simulation) −0.084 [−0.179, −0.002], whose upper bound is below 0 in
   only 59 % of bootstrap seeds. No row of either full policy was worse than 10 × RAW; 0 rows were unscored.
2. **Expansion versus learning (development, identical labels).** SBS-B over 17 candidates against SBS-B over 8:
   −0.070 [−0.133, −0.012]; attainable (reference over 17 against over 8): −0.092 [−0.165, −0.037]; learned policies on
   deployable features against SBS-B within either portfolio: between −0.017 and +0.015, none resolved. The largest
   resolved development gains of learned policies use simulator-only features (HGB −0.088, ridge −0.066, elastic net
   −0.062); linear models on the light-cone features or on budget and size alone are also resolved (−0.056 to
   −0.077).
3. **Winners by budget** (evaluation half). RAW, RO-TREX, PEC and PEC-TW at 10³; PEC and PEC-TW from 10⁴ to 10⁶ (31–43 of 88 tasks per
   budget); CDR at ≥ 3·10⁶ (up to 21 of 88); SV on 8–10 of its 16 GHZ tasks up to 10⁵.
4. **Bias and variance at the largest budget** (3·10⁷; medians over feasible development tasks, evaluation half):
   squared bias is about 98 % or more of the MSE for RAW, the readout-only family, PEC, PEC-TW and SV, about 90 % for linear and
   exponential ZNE, 72 % for Richardson ZNE, 55 % for ZNE-L∘SV, 14 % for CDR, 19 % for NIL and 3 % for FR. Median
   |bias|: RAW 2.1·10⁻², readout family 1.2–1.3·10⁻², SV 1.8·10⁻³, PEC 1.4·10⁻³, PEC-TW 1.3·10⁻³, ZNE-L 1.1·10⁻³,
   ZNE-R 9.5·10⁻⁴, ZNE-E-B 8.8·10⁻⁴, CDR 6.1·10⁻⁴, ZNE-L∘SV 2.9·10⁻⁴ (on the 16 GHZ tasks only); NIL 2.4·10⁻² (above
   RAW's) and FR 1.4·10⁻².
5. **Tails.** ZNE-E's largest MSE 1.0·10⁹ (5 rows, all at B ≤ 3 000) against 0.043 for ZNE-E-B, with an identical
   severe-vs-RAW rate (2.4 %); ZNE-R severe on 41 % of rows at 10³, 2.3 % at 10⁶, ≤ 1.1 % from 3·10⁶; NIL ≈ 45–49 % at
   every budget; FR 61 % (circuit-weighted), by width 36 / 58 / 85 / 70 % at n = 3 / 4 / 5 / 6 with median MSE 0.001 / 0.010 / 1.87 / 0.009.
6. **Learning designs.** Between-design share of the label variance at 3·10⁷: CDR 0.90, NIL 0.82, FR 0.47.
7. **Root audit.** Between-root variance ÷ design-level SE²: medians 0.44–1.08 per candidate (NIL's 90 % quantile 6.1);
   one winner under all four roots in 10/24 cells; mean cross-root regret −0.006; contract agreement 288/288; deployed
   policy per root −4.76 to −4.80.
8. **Regimes and composition.** Recommender regret ≤ 0.09 for error scales s ≤ 1.5, up to 0.31 at s = 2 and up to 0.60
   at s = 3; 0.44–1.15 with half of the gate error coherent; 0.19–0.64 with longer idle times.
9. **Preflight cost of conservatism (post hoc).** On 47 confirmation rows (4 GHZ circuits) the default bundle's model
   component chose SV / ZNE-L∘SV but the full policy did not: on 33 rows the preflight could not verify them without a
   simulation, 14 rows were out of distribution. With the component's choice on all 47 its Δ would be −0.109 instead of
   −0.084; the 33 preflight rows account for about half (−0.096), the 14 fallback rows for the other half (−0.097).
10. **Generalization and reliability.** No learned configuration is resolvedly better than SBS-B on held-out families or
    widths; trained on budgets ≤ 10⁴, ridge extrapolates beyond them best (regret 0.40 against SBS-B's 0.48; not
    resolved in the pooled extrapolation split); the learning curve (mean over folds) falls from 0.39 at 9 to 0.31 at
    18 training circuits and stays near 0.29–0.30 up to 35; nested abstention equals random rejection; simultaneous
    intervals are infinite and finite-sample certificates uninformative.

### 3. Statistical evidence: how much each observation can bear

* **Confirmatory comparison.** Resolved with margin (upper bound −0.076), stable across bootstrap seeds, reproduced
  independently. One confirmation set of 29 circuits: the interval is about ±0.08 decades; the point estimate is
  larger in magnitude than the development estimate (−0.144 vs −0.088); SBS-B being weaker on the confirmation (its gap
  to the reference 0.30 vs 0.28) accounts for at most about a third of the difference, the rest is consistent with
  sampling variation — not evidence that the recommender is better than the development estimate says.
* **Selection optimism.** The frozen configuration is the best of 30 by out-of-fold score; BBC-CV estimates the optimism
  at 0.020 decades and leaves the corrected difference unresolved. The confirmation is the protection against this
  optimism, and it holds; the development criterion of the rule is the weak link.
* **Default bundle.** Δ = −0.084 is a point estimate with an interval touching 0 (seed-dependent at 1 000 resamples;
  −0.0007 with 200 000). It should be reported as "favourable, not robustly resolved".
* **Portfolio comparison.** Out-of-fold development data that also chose the frozen configuration; exploratory, eight
  contrasts, no multiplicity correction. The fixed-policy expansion contrast (−0.070, upper bound −0.012) is the
  simplest of them: it needs no model at all, only per-budget selection on more candidates. "More candidates help more
  than learning" compares point estimates whose intervals overlap; it was not tested.
* **Root audit.** Eight circuits, four roots: enough to see that the SEs are not *under*stated on average (median
  ratios 0.44–1.08 per candidate) and that rankings flip mostly between near-equal candidates; not enough to estimate
  tail behaviour of the ratio.
* **Controlled studies.** Single circuits per family, many cells with unresolved top-two candidates; they show
  direction and mechanism, not effect sizes for a population.
* **Mechanism claims below** combine these observations with the estimators' known statistics; they are hypotheses
  consistent with the data unless a test is named.

### 4. Physical and statistical explanations

**Bias floors decide large budgets.** At 3·10⁷ shots almost every protocol's MSE is its squared bias, so the ranking is
the ranking of residual biases. RAW's bias (2·10⁻²) is the full effect of gate and readout noise; readout correction
removes the readout part and leaves ≈ 1.2·10⁻² of gate error. PEC inverts a *depolarizing* model at the median gate
error rate; the snapshots' actual channels (gate-specific rates, thermal relaxation with amplitude and phase damping)
differ, and the mismatch leaves ≈ 1.4·10⁻³. Pauli twirling turns the executed two-qubit errors into Pauli
channels, closer to the cancelled model, and lowers the floor only slightly (1.3·10⁻³) because the model's single median
rate remains wrong for individual gates. ZNE removes the leading order in the noise strength and leaves the curvature
of the decay (≈ 1·10⁻³). CDR's regression learns the actual noisy-to-ideal map on near-Clifford circuits and leaves the
smallest bias of the general-purpose candidates (6·10⁻⁴), so it wins once its training variance is paid for (≥ 3·10⁶).
*Support*: observation 4 directly; consistent with 1.7.0.

**Symmetry verification removes detectable errors only.** SV discards shots whose declared Z-parity is violated; errors
that preserve every declared parity (two bit flips within one check, or a fault before an entangling gate whose
propagated pattern keeps the parity) remain, so SV's bias floor (1.8·10⁻³) is close to PEC's although its variance at moderate budgets is lower —
hence SV wins up to 10⁵ and loses at large B. Applying ZNE to symmetry-verified folds removes part of the remaining
error and gives the lowest floor of the non-learning candidates on those 16 tasks (2.9·10⁻⁴; CDR's median there is
3.4·10⁻⁵). *Support*: observations 3–4 on the 16 GHZ tasks; a direct test
would count parity-preserving faults in the light cone.

**Variance amplification explains Richardson ZNE's low-budget failures.** Quadratic Richardson extrapolation from scale
factors 1, 3, 5 has coefficients (15/8, −5/4, 3/8); the estimate's variance is Σ c_i² σ_i² ≈ 5.2 σ² per shot-split
before the budget split over three circuits, so at 10³ shots its MSE is variance-dominated (bias share 0.9 %) and often
ten times RAW's; the factor does not change with B, the variance does, and the severe rate falls to 2.3 % at 10⁶ and ≤ 1.1 % beyond.
*Support*: observation 5 and the bias shares at 10³ and 3·10⁷.

**Exponential ZNE diverges where the signal is small; the projection only bounds the damage.** The exponential fit
divides differences of noisy expectation values; when the folded signals approach zero or change sign, the fitted decay
rate and the extrapolated value become arbitrarily large (10⁹ at 10³ shots). Projecting each trial onto the observable's
range [−1, 1] bounds every error by 2, which removes the catastrophic magnitudes, but a trial clipped to the boundary can
still be ten times worse than RAW's small error — so the *frequency* of severe rows is unchanged (2.4 %), only their
size. *Support*: observation 5; the two have the same evaluation-half MSE on 788 of 880 cells, and ZNE-E-B is never
worse on the others.

**Feature regression fails near the interpolation threshold of least squares (hypothesis; consistent with four widths,
untested).** FR
regresses the ideal value on ⟨O⟩, every ⟨Z_i⟩ and every ⟨Z_iZ_j⟩ with an intercept by ordinary least squares on 16
training circuits: 8, 12, 17 and 23 unknowns at n = 3, 4, 5, 6. The variance of an OLS prediction grows like
p/(N − p − 1) as the number of unknowns p approaches the number of training points N and is controlled again by the
minimum-norm solution when p > N (the "double descent" peak). The observed median MSE by width — 0.001, 0.010, **1.87**,
0.009 — peaks at n = 5, where p (17) just exceeds N (16), in every family at that width; FR's MSE is variance-dominated (bias share 2–3 %) and 47 % of its label
variance is between training designs. *Test* (ranked below): FR at n = 5 with 32 and 64 training circuits, and with
ridge regularization.

**NIL's failures come from its training designs, not from the shot budget.** NIL's severe rate is flat in B (45–49 %)
and its label variance is 82 % between designs: with a fixed number of training circuits, the spread of the fitted
neighbour model across designs does not shrink when more target shots are spent. At the largest budget its median |bias| exceeds RAW's.
*Support*: observations 5–6; consistent with the 1.7.0 design study (more training circuits halve NIL's error without
removing the failures).

**Why more candidates help, and learning on deployable features does not (yet).** The policy value of a portfolio is the expected minimum over its
candidates' risks at each budget. The nine new candidates add low-bias options in exactly the regime where the
original eight were weakest — PEC-TW and ZNE-E-B at medium budgets, RO-TREX at small budgets, SV and ZNE-L∘SV on
symmetric tasks — so even a budget-only rule (SBS-B) gains −0.070. Learning gains only when its features predict *which*
candidate is best on a given circuit; with deployable features (structure, light-cone fault weights, observable,
device snapshot) the within-circuit ordering of near-equal candidates is mostly below the labels' resolution (in the
root audit 14 of 24 cells change winner between roots, at negligible cost). *Support*: observation 2 and the audit.

**Why simulator-only features help (hypothesis).** The features that need a simulation of the supplied noise model
include the noisy expectation values and the acceptance probabilities that set some candidates' cost; the size of the
ideal signal relative to the noise determines whether extrapolation and inversion amplify noise into large relative
errors (ZNE-E's divergences, PEC's sampling overhead relative to |μ|). A model that sees these can rank candidates
within a circuit; a model that sees only structure and calibration cannot. *Not tested*: a permutation importance of
the simulated features on development data would test it.

**Coherent errors are invisible to Pauli features (F10; hypothesis supported by three circuits and the descriptor's
unit tests).** The light-cone fault weight and every 1.8.0 noise descriptor
(two-qubit light-cone weight, readout asymmetry, spread of two-qubit error rates, duration over T2) are computed from
the snapshot's error rates and coherence times, which do not distinguish a coherent from a stochastic error of the same
infidelity. A coherent over-rotation of the same average infidelity adds errors
that interfere — growing quadratically in depth instead of linearly — and that PEC's depolarizing inverse cannot
cancel. At B = 10⁵ the measured best is ZNE-E, RO-DD and ZNE-L on the three circuits (two of them unresolved against
their runner-up) and the recommender chooses PEC-TW, ZNE-E and ZNE-L; over all budgets its regret reaches 1.15 (QAOA,
mostly at B ≥ 3·10⁵, where it picks CDR or ZNE-E-B instead of ZNE-L). *Support*: observation 8; the unit tests of the light-cone descriptor show the same feature value for reinforcing
and cancelling coherent errors.

**Error scale is out of the training distribution.** All training devices run at their native error rates (s = 1). At
s = 2–3 the exponential growth of PEC's sampling overhead and the curvature of ZNE's decay change the ranking; the
recommender was never shown this regime, and its selective rule abstained in none of these cells.
*Support*: observation 8.

### 5. Speculation (not supported by a test in this release)

* The gap between the confirmation (−0.144) and development (−0.088) estimates of the frozen configuration may partly
  reflect the confirmation's family mix (more QAOA, fewer GHZ tasks) rather than chance alone; the per-family breakdown
  of the confirmation has too few circuits per family to say.
* Of the 0.06 decades between the default and the frozen bundle's full policies, about 0.012 is the default bundle's
  preflight loss (observation 9); most of the value of simulation therefore seems to lie in *predicting* risk, a
  smaller part in *verifying* candidates.
* The audit's SE ratios below 1 (PEC-TW 0.44) suggest the design-level SEs are conservative for candidates with
  re-randomized compilations, whose trial-to-trial variation already contains the twirl's randomness.
* On hardware, coherent and non-Markovian errors would likely enlarge the gap between Pauli-feature recommenders and the
  per-circuit best (observation 8), so a hardware validation could look worse than the simulated confirmation.

### 6. Negative results

* Learning on the `deployable18` features did not beat SBS-B on development data in either portfolio, and no learned
  configuration did on held-out families or widths; on the confirmation, boosted trees on deployable features did not
  robustly (the default bundle) — only the exploratory ridge comparator on the 1.7.0 deployable features is resolved
  there (−0.130 [−0.226, −0.037]).
* Nested abstention is indistinguishable from random rejection at the same coverage; simultaneous prediction intervals
  are infinite; finite-sample certificates never certify.
* The physics-informed b² + v/B model extrapolates worst (regret 1.06 from training budgets ≤ 10⁴) and is worse in
  range; the MLP fails catastrophically on two feature sets.
* The device-snapshot features do not resolvedly help (ablation +0.039 [−0.012, +0.092]).
* FR and NIL are severe-failure candidates as implemented (61 % and 48 % of feasible rows worse than 10 × RAW,
  circuit-weighted).
* The bounded estimator ZNE-E-B does not reduce how often ZNE-E is badly wrong, only how wrong.
* The recommender neither adapts to nor flags error scales above the training range or coherent noise.
* The default deployable bundle's advantage is not robustly resolved.

### 7. Ranked next experiments

Ranked by expected information for the thesis question (does budget-aware *learned* selection beat a budget-only rule,
and when?) per unit of cost. Each would be declared in `RESEARCH_PLAN.md` before its data exist.

| rank | experiment | design | decision rule to declare | cost (2 cores) | what it would change |
|---|---|---|---|---|---|
| 1 | **Cheap verification of declared symmetries** | an analytic lower bound on SV's acceptance probability from the light-cone fault weights (no simulation) in the design-level preflight; evaluated on dev18 rows (never on conf18) | the default bundle's development Δ improves and no row returns a candidate that is infeasible in the labels | hours of coding, minutes to evaluate | the default bundle could recover the ≈ 0.012-decade preflight part of its 0.025-decade loss on GHZ circuits without simulation (the other half is out-of-distribution fallback) |
| 2 | **New confirmation of a deployable recommender with a BBC-robust rule** | ≥ 40 new logical circuits; primary: the default bundle's full policy; rule: resolved on the confirmation with 10 000 bootstrap resamples **and** the BBC-corrected development difference resolved | as stated | ≈ 1.5 CPU-h labels | turns the fragile −0.084 into a statement either way |
| 3 | **Larger development set** | 120–150 new logical circuits (the learning curve is nearly flat from 18 to 35 circuits; more circuits test whether that is a plateau), same families and devices, with 7–8 qubit circuits where the exact engine allows | out-of-fold Δ of deployable learners vs SBS-B; learning-curve slope | 4–6 CPU-h | tests whether learning on deployable features is data-limited or information-limited |
| 4 | **FR at the interpolation threshold** | FR at n = 5 with 16, 32, 64 training circuits, OLS vs ridge, on the GHZ/Ising/HEA circuits of the audit | MSE peak at p ≈ N disappears with N ≥ 2p or with ridge | ≈ 0.5 CPU-h | confirms or refutes the mechanism; a ridge FR could become a useful candidate |
| 5 | **Hardware validation** (owner authorization) | the dry-run plan of the shipped bundle on `ibm_fez`: 28 jobs, 524 s of the 540 s window, RAW / RO / recommended / SBS-B | pre-declared in `configs/selection/hardware17.yaml`; descriptive (two tasks) | 9 minutes of QPU time | the first test of transfer; limited by two tasks |
| 6 | **Coherent-error descriptors** | add a coherent-error proxy (e.g. an echo or purity diagnostic of the target, or calibration-derived over-rotation) and relabel the noise-composition study with a larger χ grid | regret in coherent cells falls without raising it in Pauli cells | ≈ 1 CPU-h | addresses F10, the largest controlled failure |
| 7 | **Error-scale coverage in training** | include devices at s ∈ {0.5, 2} in development; test on s = 3 | regret at s = 2–3 vs the s = 1-only model; the OOD check flags s = 3 | ≈ 2 CPU-h | tells whether noise strength is learnable from the snapshot features |
| 8 | **Mismatch study with 17 candidates** (D43) | rerun `mismatch17` with portfolio-1.8 and policy/2 | as in 1.7.0 | ≈ 2 CPU-h | updates Figure 6 |
| 9 | **Permutation importance of the simulated features** | on dev18 out of fold | which simulated features carry the development gain | minutes | tests the hypothesis of §4 and points to cheaper surrogates |

### 8. Threats to validity

Simulation of fake-device snapshots with largely stochastic noise; small widths; one confirmation set; the confirmatory
estimand fixed after development results; selection optimism; the 10⁻⁶ floor of the log metric (differences between
risks below about 10⁻⁶ are compressed); exploratory analyses without multiplicity correction; the mismatch question not
re-examined; labels that use the evaluation half once per study but have been inspected for the development set many
times.
