# Budget-aware selection of QEM protocols — methods (QEM Explorer 1.8.0)

**1.8.0.** Sections 1–15 describe the method as established in 1.7.0 and still in force; §16 lists every change made
in 1.8.0 (the deployed policy and its preflight contract, bounded-cost deployment, provenance and seeds, variance,
tails, the expanded candidate portfolio, independent shot-level references) with the review finding it answers. Where
§16 supersedes a statement of §1–15 it says so.

This document states what the 1.7.0 study computes, under which assumptions, and where the implementation departs
from the theory companion (`Automated_QEM_Protocol_Recommendation.pdf`, cited as "the companion" with its equation
numbers). Code: `qemx/selection/`. Commands: `docs/selection/COMMANDS.md`. Results: `docs/selection/RESULTS.md`.
Literature: `docs/selection/LITERATURE.md`. What changed from 1.6.0 and why: `docs/REVIEW_RESPONSE_1.7.md`; every
change of an estimator, a convention or a reported quantity is listed in §14 below, none was made silently.

Everything reported comes from **emulated fake-device trials** (calibration snapshots of IBM fake backends executed by
the exact density-matrix engine). No hardware data exist in this release.

## 1. Questions

Central question (1.7.0): **which information improves QEM selection beyond a strong budget-only baseline (SBS-B),
and when does that improvement fail?** The sub-questions below are kept from 1.6.0; the study is not tuned to give a
positive answer.

1. Under equal, fully charged budgets, does a learned selection policy beat fixed choices (RAW, the single best
   protocol, a budget-dependent fixed protocol, a size-and-budget rule)?
2. Which descriptors are necessary — circuit size and budget, structure, light cones, the calibration snapshot,
   simulator-only quantities?
3. Do nonlinear models (gradient-boosted trees, a small neural network) beat regularized linear ones?
4. Which protocol wins where, across budgets 10³–3·10⁷, and where do rankings reverse?
5. Where does a learned policy fail to generalize (families, devices, widths, budgets), and does abstention help?

Secondary: when the execution law departs from the calibration snapshot, can a cheap diagnostic detect it, and does
a diagnostic-triggered fallback lower the risk after its shots are charged?

## 2. Tasks (`task.py`)

A task is 𝒯 = (ρ_in, C, O, D_t, 𝓘, B) (companion Def. 1.2):

| part | implementation |
|---|---|
| ρ_in | per-qubit input tokens (`0 1 + - +i -i`, `ph:t`, `bloch:θ,φ`), prepared once by the compiler; families with a known answer keep their own input; others vary it (`default`, `plus`, a seeded random product state) |
| C | a family instance (QFT/AQFT, GHZ, Trotterized Ising, brickwork, Grover, Bernstein–Vazirani, hardware-efficient ansatz, QAOA on a ring) of 3–6 qubits, compiled for the device with a fixed layout: the calibration-best chain (`best`) or the best chain disjoint from it (`alt`) |
| O | a diagonal observable with values in [−1, 1]: the family's own, Z on the middle qubit, ZZ on the middle pair, or the parity of all bits — so μ ∈ [−1, 1] |
| D_t | the device's calibration snapshot (`fake_fez`, `fake_torino`: Heron, CZ; `fake_kyiv`: Eagle, ECR) — the *supplied* model — and, separately, the execution law (equal to the snapshot except in mismatch scenarios) |
| 𝓘 | information classes (below) |
| B | the total shot budget, grid 10³, 3·10³, …, 3·10⁷ |

Identity: `task_id` hashes every field that changes the numbers; `base_id` hashes only the logical circuit (family,
width, parameters). Every variant of one logical circuit — input, observable, layout, device — shares its `base_id`,
and every split keeps a `base_id` on one side. The reproducibility record (compiled facts, layout, measurement map,
circuit digest; provenance of the run) is stored beside the features and never used as one.

**Information classes** (companion Table 1.1):

| class | meaning | examples |
|---|---|---|
| available | known before any shot, free | structure, observable, light cones, the snapshot's error rates, B |
| acquired | measured at deployment, shots charged to B | the echo diagnostic, calibration counts |
| simulated | classical simulation of the *supplied* model: deployable, but exponential in width | supplied-model variances, ideal-state entropies, the supplied noisy signal |
| oracle | labels and research evaluation only | the ideal value, the execution law, injected mismatch parameters |

Oracle features are refused as model inputs (`features.matrix`).

## 3. Candidates: complete protocols and their ledgers (`protocols.py`)

A candidate fixes every choice before data are seen (companion Def. 1.3). The portfolio:

| key | protocol |
|---|---|
| RAW | all shots on the target; sample mean |
| RO | tensored readout correction from two calibration circuits |
| ZNE-L | global folding λ = 1, 3; linear fit; RO |
| ZNE-R | global folding λ = 1, 3, 5; Richardson; RO |
| ZNE-E | λ = 1, 3, 5; exponential fit with a declared linear fallback; RO |
| PEC | inverse of the snapshot's median-rate depolarizing model; one shot per sampled circuit; RO |
| CDR | 16 near-Clifford training circuits; OLS; 25 % of B on training; per-trial fallback; RO |
| NIL | weight-1 Pauli neighbors at every position (12 sampled), 16 2-design training circuits, ℓ1 ≤ 2 without intercept, equal target split (Chen et al.'s estimator at a much smaller training set than their ~10³ circuits); 50 % of B on training; RO |

Excluded and why: virtual distillation and chained protocols (not executable as implemented: exact copies or
oracle splits); symmetry verification (only defined for tasks with a symmetry; the pilot's families mostly have
none).

**Ledger** (companion Eq. 1.8): B = B_calibration + B_training + B_diagnostics + B_selection + B_execution, in
integers, conserved exactly (`Allocation.check`; tested for every protocol and budget). Allocation uses only the
*supplied* model:

* training (CDR, NIL): ⌊f·(B − B_diag − B_sel)/T⌋ shots on each of T training circuits; infeasible below 32;
* calibration vs target: the variance-optimal continuous split of the supplied cost model (v per target shot, w per
  calibration shot; companion Prop. 1.4 with c = 1), rounded down per calibration circuit (≥ 1), the rest to the
  target;
* ZNE folds: the exact integer minimizer of Σ a_j²/N_j with a_j = |η_j|σ_j of the supplied model (Prop. 1.4,
  integer version), every fold ≥ 1 shot;
* infeasible when any minimum fails, with the reason recorded; a protocol is never silently given more shots.

**Failure handling** (companion Eq. 1.7), applied per trial and counted: a singular finite calibration → no readout
correction in that trial; a failed exponential fit → the linear fit of the same data; CDR training data that resolve
no feature direction (decided from that trial's own data) → the readout-corrected value; no estimate at all
(nonresponse; none occurred at feasible budgets) → the declared estimate 0, i.e. loss μ². Nothing is dropped.

### Method inventory: what is executable

Every estimator of the app was audited for what a deployment could run, with which information, before the
portfolio was fixed (decision D1):

| method (app) | executable at deployment | in the portfolio | note |
|---|---|---|---|
| raw, readout (tensored) | yes | RAW, RO | readout calibration drawn per trial (θ̂) |
| readout M3, TREX | yes | no | M3 needs a per-trial solve; TREX's finite-mask hardware variant is not emulated |
| ZNE (linear, Richardson, exponential) | yes, with a deployable fold allocation | ZNE-L, ZNE-R, ZNE-E | the app's default allocation uses exact per-fold variances (oracle); the portfolio allocates from the snapshot |
| PEC, one shot per sampled circuit | yes | PEC | the app's default model is the oracle noise; the portfolio inverts the snapshot's median-rate model |
| PEC, several shots per sampled circuit | yes | no | its sampler is Gaussian (instance variance estimated), not an exact law |
| CDR, FR | yes | CDR | the per-trial fallback is decided on that trial's sampled features |
| NIL | yes, with the equal target split | NIL | the app's default split uses exact variances (oracle) |
| symmetry verification | yes, for tasks with a symmetry | no | defined for few of the pilot's families |
| virtual distillation | no | no | needs ideal copies and exact states |
| chains (outer ∘ inner) | no | no | split by the inner method's exact variances |

**NIL is the published estimator, at a reduced training size.** Compared with Chen et al. (arXiv:2512.12578, v1 of
14 Dec 2025, the only version found; no journal version): it follows the paper in the linear combination of noisy
weight-1 Pauli neighbors (a random subset, the same neighbor map on training circuits), no intercept, 2-design
training circuits with angles in {0, π/2, π, 3π/2}, ideal labels of the unmodified training circuits, and ℓ1-bounded
least squares with γ = 2. The portfolio's configuration is the paper-faithful one (`l1`, bound 2, no intercept,
neighbors after every noisy operation, no bare target column, equal split). It departs in the training-set size (16
circuits against the paper's 256–5 000; decision D15), in exact density-matrix ideal values instead of stabilizer
simulation (equal values at these widths, not scalable as coded), and in redrawing the rotations of the compiled
native circuit rather than of the logical ansatz; CPTP-basis and weight-m > 1 neighbors are not implemented.

## 4. Trial-level emulation (`emulate.py`)

One trial is one complete deployment at budget B. Its risk R_a = E[(μ̂_a − μ)²] = Bias² + Var (companion Eq. 1.5)
includes calibration and training randomness by the law of total variance (Eq. 1.6): every trial draws its own
calibration θ̂ (shared by all circuits of that trial), its own training data (refit per trial) and its own target
counts. Nonlinear estimators are never evaluated on exact probabilities in place of finite data.

Training designs (1.7.0: fresh designs; review R1). CDR and NIL draw their training design at random (which
non-Clifford slots become Clifford; which 2-design training circuits and neighbors). A deployment draws a fresh design,
so the design is part of the protocol's randomness. The label's estimand is the **fresh-design risk**
R = E_D E[L | D], and its estimator must sample designs independently:

* each label half has its own pool of K = `learning_designs.per_half` (10 in the pilot) designs drawn from independent
  seeds — SeedSequence([master seed, "DESG", half, k]) hashed with the task — so no design is in both halves, and
  the trials of a half are spread over its pool (m = r/K trials per design);
* a protocol's *allocation* (how many shots go to training and calibration) is computed from a separate allocation
  design (the base seed), so allocations do not depend on the sampled designs;
* the same pool serves every budget of a half: a declared pairing across budgets (common random numbers), which makes
  budget-to-budget differences of one protocol less noisy and is never used across halves;
* every trial records its design index (`design` array in the labels; −1 for protocols without a design), and every
  design's identity, seed, diagnostics and fallback are stored in the record (`design_records`).

The variance of the half's MSE estimate is Var(R̂) = Var_D(E[L|D])/K + E_D Var(L|D)/(Km): more trials per design shrink
only the second term. Its standard error is therefore computed over designs (cluster-robust over the K designs,
√(K/(K−1) Σ_k (S_k − n_k R̂)²)/n), and the between- and within-design components are estimated by a one-way analysis
of variance (`labels.design_stats`). With K = 10 designs per half these are few clusters: the SE is approximate
(Cameron & Miller 2015). 1.6.0 shared K = 6 fixed designs between both halves; its labels estimate the risk
*conditional on those designs* and its trial-level SE understates the fresh-design uncertainty. The 1.6.0 labels are
kept unchanged as a historical dataset (`reports/selection_pilot/`) and read with that estimand
(`learning_designs: 6` is parsed as `fixed-shared`; their configuration digests are unchanged).

When a learning protocol has no usable training direction (a degenerate design) its declared structural fallback is
used and flagged for every trial of that design (`learn_fallback`). When the label run does not need the supplied
model's training moments (`protocol_moments` off), NIL and the learning estimators skip them and flag
"moments-not-estimated" instead of returning placeholders.

Exactness. Every circuit is a fixed circuit with i.i.d. shots under a stationary law, so its N-shot counts are
multinomial in its exact output distribution, computed once per circuit by the exact engine (cached by the runner).
PEC with one shot per sampled circuit draws each shot from the exact sign-resolved law P(±, x) = (p_abs(x) ±
p_net(x))/2 — a shot is an instance draw followed by a measurement, so the two-stage law is exact. Large budgets
(3·10⁷) therefore cost no more than small ones.

Validation against direct execution (`direct_trials`): an independent re-implementation draws every shot
separately — calibration shot by shot, ZNE with its own `numpy.polyfit`, PEC by sampling an explicit instance
circuit per shot (one correction per gate location with probability |q|/γ), simulating it under the execution law and
drawing one outcome. RAW, RO, ZNE-L, ZNE-R and PEC agree with the emulator (means within 4 SE, variances not
rejected at p = 10⁻³; `tests/test_selection_core.py`, `python -m qemx.selection validate`).

Not covered by a static law: drift within a run and shot-to-shot correlation. The temporal-drift scenario is a
static mixture of two laws (exact for randomly interleaved shots, not for a monotone drift).

## 5. Labels (`labels.py`)

For every task, protocol and budget, 2r trials in two halves with independent random streams (`rng_for`: a
SeedSequence of the master seed and a hash of task, budget, protocol and half):

* R̂_sel (r trials): the label models learn from (companion Eq. 1.16);
* R̂_eval (r trials): an independent estimate, used only to score decisions.

Per-trial estimates and flags are stored (`tasks/<task_id>.npz`) with a JSON record (allocations, ledgers, costs,
features, the reproducibility record, the configuration digest). A task with a record of the same configuration is
skipped (resume); one of another configuration is refused. Costs are kept apart: shots by role, circuit instances
sent, a duration-weighted execution time T_exec ≈ Σ_j N_j (t_j + 250 µs) from the device's ALAP schedule (companion
Eq. 1.9), and the emulation's own CPU seconds (not a deployment cost).

Statuses (1.7.0; review R4). An allocation is `ok`, `infeasible` (a declared property of the protocol at that
budget, with the reason) or `method-error` (the protocol's preparation raised an exception, recorded with its message).
A label *task* is `ok`, `error`, `timeout` or `memory`; failed tasks are written as error records without arrays,
collected in `label_failures.csv` and excluded from the dataset, never silently dropped; a record whose arrays do not
match their digest is refused as corrupted. Flags overlap (a trial can fall back in calibration and in the fit), so
failure rates are reported as the union `fail_any` (share of trials with at least one flag), never as a sum.

Execution (1.7.0; review R4). Each label task runs in a supervised child process (spawn) with a wall-time and a
memory limit enforced by the parent (psutil watchdog of the process tree; RLIMIT_AS inside the child on POSIX), on
every platform and also with one worker (the parent process itself is never limited). A run directory is bound to its
inputs (`run.json`): the digests of the label configuration, the task manifest, the split, the protocol portfolio and
the device snapshots, plus the code and environment. Resuming with changed inputs is refused; a changed code or
environment needs `--accept-changes` and is recorded. The analysis reads only the manifest's tasks; the final and
confirmation evaluations require a complete set; reuse of tasks from another run is an explicit, recorded import.

## 6. Features (`features.py`)

Groups: structure, observable, light cone, noise (snapshot), budget, protocol overheads, state (simulated). Light-cone
features use the backward *structural* light cone of the observable on the compiled circuit (from the measured
qubits backward; a two-qubit gate touching the set adds both qubits): `lc_qubits`, `lc_gates`, `lc_2q`,
`lc_locations` (fault locations with p > 0 inside it), `lam_lc` (Σ Pauli-twirled fault probability inside it under
the snapshot), `lam_lc_exact` (−Σ ln(1 − p)), `lam_total` and `lc_share`. A sum of error indicators is a feature, not
a fidelity (companion §1.5.1). Feature sets for question 2: `size_budget` ⊂ `structure`, `lightcone`, `deployable`
(all available), `deployable+simulated`. Dictionary: `docs/selection/DATA_DICTIONARY.md`.

The light-cone group removes only the structural light-cone counts; the noise group still holds `lam_lc`,
`lam_lc_gate`, `lam_lc_idle` and `lc_share`, which are computed from the light cone. 1.7.0 therefore adds *semantic*
ablations that remove everything derived from one source (`features.SEMANTIC_ABLATIONS`): every light-cone-derived
quantity (the structural counts and their noise-weighted sums) and every simulated quantity. The third declared
semantic removal, every calibration-derived noise sum, is exactly the noise group of the deployed feature set, so it is
reported once (as the group).

## 7. Models (`models.py`)

One regressor per protocol predicts y = log10(R̂_sel + ε0), ε0 = 10⁻⁶ (an RMSE of 10⁻³; sensitivity 10⁻⁷, 10⁻⁵).
Families with comparable tuning (a grid of 6 settings each, the same inner group-4-fold over base circuits):
ridge (α), elastic net (α × ℓ1 ratio), histogram gradient-boosted trees (depth × learning rate), and a small MLP
(one hidden layer of 16 or 32 units × weight decay). Inputs are standardized inside each fit; missing values take
the training medians. The log target is not a mean-risk prediction (exponentiating it is biased), so models are
compared by decision performance.

Structured risk model (1.7.0, `struct`; prompt §8). A physics-informed baseline, R_a(x, B) ≈ b_a(x)² +
v_a(x)/((1 − f_a)B): a bias floor that more shots do not remove and a sampling term that falls with the shots left for
the target after the training share f_a. Stage 1 fits (b², v) ≥ 0 per training task and protocol by non-negative least
squares of the relative error over the task's budgets; stage 2 maps the budget-free features to log10 b² and log10 v
(standardized ridge, α = 1). It is a testable approximation, not a QEM law: the fixed calibration overhead F_a(x)
and any interaction u_a(x, B) (nonlinear estimators, allocation rules that change with B, fallbacks at small B) are
not modelled, and its fit is reported per protocol (`tests/test_selection_v170.py` checks it recovers b² and v from
synthetic risks). Graph neural networks were not built (decision D10): in the corrected pilot, removing the
structure group changes the held-out regret by −0.006 [−0.054, 0.026] decades and the light-cone group (or everything
derived from the light cone) by −0.011 to −0.014 with intervals containing 0, ridge on structure features is within
the boosted trees' interval, and the learned policies leave no resolved gap to SBS-B on held-out development circuits
that circuit structure could explain (`RESULTS.md` Part II §§4–5). A graph model would have to earn its complexity against that.

## 8. Policies and evaluation (`policies.py`, `evaluate.py`)

Policies: RAW; SBS (single best on training rows: the protocol with the lowest base-circuit-weighted mean log10
risk over the training rows where *every* candidate it is compared with is feasible — one population for all
candidates, 1.7.0 — with a declared fallback order where it is infeasible); SBS-B (the same per budget); SIZE-B (depth-3 tree on n,
two-qubit gates, log10 B); MODEL (argmin of predicted risk over feasible protocols, Eq. 1.18); MODEL+ABSTAIN (accept
only if the chosen protocol's simultaneous conformal upper bound is below the fallback's lower bound; fallback
SBS-B; with G calibration base circuits the simultaneous level α/K is attainable only if G ≥ (1 − α/K)/(α/K), 79 for
α = 0.1 and K = 8 — below that the bound is infinite and the policy is its fallback); ALL-CANDIDATE (argmin of the test task's own R̂_sel — a benchmarking construction, never deployable).

Conventions (fixed before results):

* scoring by R̂_eval; the per-task reference a* = argmin R̂_sel (identified on the other half), so regret is free of
  the optimizer's curse (a test shows the selection-half minimum is biased low and the evaluation half is not);
* primary aggregate: mean over test base circuits (each weighted equally) of log10(R̂_eval(π) + ε0); arithmetic
  policy MSE beside it, **also weighted by base circuit** (1.7.0; review S5 — 1.6.0 averaged rows, so circuits with
  more variants weighed more; the row mean is kept as `policy_mse_rows` for comparison);
* gap closed = (m_SBS − m_π)/(m_SBS − m_VBS): 1 = per-task oracle, 0 = single best, < 0 worse;
* 95 % intervals from a paired bootstrap over test base circuits; label noise separately (redraws of R̂_eval);
* a split's results are reported out-of-fold pooled: each base circuit is tested once (K-fold, leave one group
  out), all test rows are scored together and resampled in one bootstrap (`pooled_<split>.csv`); per-part
  summaries are kept (`summary_<split>.csv`). The intervals describe the sampling of test circuits, not the
  variability of the training sets;
* the best of several model configurations is picked on the same scores it is reported with, which is optimistic;
  its bootstrap bias-corrected estimate (BBC-CV, Tsamardinos, Greasidou & Borboudakis 2018: select on each
  bootstrap sample of base circuits, score the selection on the out-of-bag circuits) is reported beside it
  (`optimism.csv`), and the final test evaluates the frozen choice once;
* splits: group 5-fold (same distribution); leave one family out; held-out device (`fake_kyiv`); width (train
  n < 6, test n = 6); budget extrapolation (train B ≤ 3·10⁵) and interpolation (alternate grid points). Family and
  width are properties of a base circuit; the device and budget splits are crossed with the group 5-fold, so a
  circuit tested on `fake_kyiv` or at a held-out budget never contributes training rows (its other variants and
  budgets share its ancestry). No base circuit is on both sides of any split;
* the final test of 1.6.0 (30 % of base circuits by a keyed hash) was evaluated once in 1.6.0. Comparing logical
  circuits (gate sequences, parameters, default inputs and measurement conventions — `ancestry.logical_id`) showed
  that it overlaps earlier studies, so in 1.7.0 it is a **historical** final test: it is relabeled with fresh designs,
  re-evaluated with the frozen 1.7.0 configuration (`final`, reported as historical, not confirmatory) and then used
  as development data;
* the **confirmation set** (`confirm17.yaml`): 25 new logical circuits (50 tasks), registered in the ancestry
  registry (`configs/selection/ancestry_registry.json`) before labeling, checked to share no logical circuit with any
  earlier study, labeled after the freeze and evaluated once (`confirm`). Only this evaluation is confirmatory.

Statement kinds (1.7.0; `claims.py`). Every reported number carries the kind of statement it supports: `empirical`
(measured on these data), `cv-residual` (calibrated in cross-validation, no finite-sample guarantee),
`split-conformal` (finite-sample marginal coverage under exchangeability of base circuits), `finite-sample-bound`
(Hoeffding/Bernstein on the clipped estimator), `model-conditional` (true if the model's assumptions hold),
`post-hoc-check` (computed with the evaluation labels, e.g. the 2ε check), `ood-warning` and `descriptive`. A
measured coverage is never reported as a guarantee.

Regret and the reported metrics (decision D25). For a test row with chosen protocol π(x) and reference
a* = argmin R̂_sel: absolute regret R̂_eval(π) − R̂_eval(a*) (MSE units) and log10 regret
log10((R̂_eval(π) + ε0)/(R̂_eval(a*) + ε0)). Both can be negative, because a* is an estimate (it is identified on the
other half); negative values are kept in every mean and tail and their share is reported (`p_negative_regret`), so
the empirical regret is not the theoretical (non-negative) regret against the true best protocol. Reported per
policy and budget: mean, median and 90 % quantile of both regrets; the share within 10 % and within a factor 2 of
the reference (practical-tolerance success); the share of rows worse than RAW (degradation); the failure share (mean
share of trials on a fallback path or without response); coverage (share receiving the policy's own choice);
arithmetic and log10 MSE. Every aggregate weighs base circuits equally.

Analyses of the frozen configuration (D25, D26; `analysis.py`):

* **deployed rows** — the frozen configuration refitted on each training fold of the group 5-fold, one model per
  fold reused for every analysis below: per held-out row its choice, its predicted log risks, SBS-B of the same
  fold as fallback, and the margin z = (P₂ − P₁)/√(s₁² + s₂²) between its two lowest predictions (s: the inner-CV
  RMSE of each protocol's prediction);
* **selective recommendation, nested** (1.7.0; review R2) — accept the choice when z ≥ τ, else SBS-B (no shots, no
  measurement). 1.6.0 chose each outer fold's τ on the other folds' out-of-fold rows, but the models behind those
  rows had been trained on the outer fold's labels. 1.7.0 evaluates the rule by nested group cross-validation
  (`analysis.nested_selective`): for every outer fold, an inner group-4-fold on the outer training circuits only
  gives margins and log risks of models fitted without the inner held-out circuits; τ is chosen on those inner rows
  (lowest whole-policy mean log10 risk, base circuits weighted equally); the model and SBS-B are refitted on the
  outer training rows and the rule is applied once to the outer test rows. Reported: coverage, the whole-policy
  arithmetic MSE and mean log10 risk *including the fallback's loss*, the model alone, SBS-B, SIZE-B and SBS, and
  random rejection at the same coverage per fold (its exact expectation and the 2.5–97.5 % range over 200 random
  acceptance sets). `deploy_tau` — the threshold the frozen configuration deploys — is chosen once on all
  development rows by the same inner procedure and recorded by `freeze --complete` with the digest of the nested
  results; the confirmation set and the controlled studies use it;
* **calibration** — intervals of the log10 risk at nominal 50, 80, 90, 95 %: the training fold's model ± the
  base-circuit-weighted (1 − α)(1 + 1/G) quantile of the absolute residuals of its own inner group cross-validation;
  per candidate (marginal), for the selected candidate only (post-selection; no guarantee) and Bonferroni-
  simultaneous over the feasible candidates. Residuals of inner-CV models placed around the model refitted on the
  whole training fold are the "CV" construction, not split conformal: it has no finite-sample 1 − α guarantee (the
  CV+ construction guarantees 1 − 2α under exchangeability of base circuits; Barber, Candès, Ramdas & Tibshirani
  2021, Ann. Statist. 49, 486–507), so empirical coverage on held-out circuits is reported rather than assumed.
  Coverage is measured against the noisy evaluation-half labels; it concerns those labels, not the true risks;
* **ablation** — each feature group removed, and each semantic source removed (§6), the model retrained and its
  hyperparameters retuned with the same inner group-4-fold; increase in mean log10 regret with a paired bootstrap
  interval over base circuits. Predictive importance, not causal; the controlled interventions (observable, noise
  strength, noise composition) are the mechanistic evidence;
* **learning curve** — model and SBS trained on random subsets of each training fold's base circuits, scored on the
  fixed held-out fold;
* **budget generalization** — Ridge, boosted trees, the structured model and SBS-B trained only on rows with
  B ≤ 10⁴ or B ≤ 3·10⁵ (or all), scored at every budget of held-out base circuits; budgets inside the training range
  are interpolation, above it extrapolation;
* **design diagnostics and variance** (1.7.0) — per task, learning protocol, half, design and budget: the design's
  mean loss and its diagnostics (training-label spread and range, rank resolved above shot noise, singular values and
  condition of the exact training features, coefficient ℓ1 norm, fallback; research-only oracle diagnostics such as
  whether μ lies inside the training labels' range) (`design_diagnostics.csv`); per protocol and budget the median
  ratio of design-level to trial-level SE and the between-design share of the label variance
  (`design_variance.csv`);
* **failure profile** (1.7.0) — per policy and budget: arithmetic MSE, the share of the total squared error from the
  worst 1 % and 5 % of rows, and the rate of severe misselection (a choice ≥ 10× worse than the reference);
* **family benefit** — paired difference of log10 MSE and of MSE between the frozen configuration and SBS-B per
  family, with intervals over the family's base circuits.

Controlled regime study (D23; `run.regime_analyze`). Three new base circuits on `fake_fez`, each with only the
device error scale s changing (every calibrated error rate of the snapshot and of the execution law × s), plus an
observable block and a layout block for the Ising circuit at s = 1. Per task and budget: the reference preference is
the protocol with the lowest MSE over both halves; it is *unresolved* when the runner-up's MSE lies within 1.96
standard errors of the difference (independent trials; this is not an equivalence test). The recommender is the
frozen configuration trained on the development set without these circuits; it abstains to SBS-B when its margin is
below `deploy_tau`. The regret is computed as in the main evaluation (choice scored on the evaluation half against the
protocol identified on the selection half). The maps are evaluated cells only — empirical performance regime maps,
not phase diagrams; boundaries are known to the grid's resolution. 1.7.0 computes the standard errors at the design
level for CDR and NIL and states that the unresolved rule is pointwise per cell, not simultaneous over cells or
candidates.

Noise-composition study (1.7.0; prompt §8 C; `noisecomp17.yaml`, analyzed by `regimes --subdir noisecomp`). The three
regime circuits (n = 4, `fake_fez`, calibration-best chain) under one noise component changed at a time, snapshot and
execution law together: two-qubit gate errors × 2, × 3 (`g2`), one-qubit gate errors × 3 (`g1`), relaxation T1, T2 ÷ 2,
÷ 3 (`t`), readout × 2, × 0.5 (`ro`), and a coherent fraction of every gate error of 25 % and 50 % (`chi`, exact
engine). Each setting's total fault weight λ_total is recorded so settings of similar aggregate error but different
composition can be compared. Exploratory: one task per cell, three circuits.

## 9. Certification and intervals (`certify.py`)

Bounded-loss certificates need a bounded loss. Extrapolated and signed estimators can leave [−1, 1], so their
squared error is not bounded by 4; certificates use a declared clipped variant (projection onto [−1, 1], which cannot
increase the error for μ ∈ [−1, 1] but can change the bias), with H = 4. **They are finite-sample bounds for the
clipped estimator only**; the unclipped MSE the study reports is not covered and is kept beside them. Implemented:
Hoeffding simultaneous intervals (companion Thm. 1.7), empirical Bernstein (Maurer & Pontil 2009) with a union bound,
and the selection rule of Thm. 1.6 (argmin U; U_â < L_RAW certifies an improvement on the same event). The bounds need
independent units (1.7.0; review R3): for protocols without a training design the units are the trials; for CDR and NIL
under fresh designs the units are the K design means (trials sharing a design are not independent); for the 1.6.0
fixed designs the bound is conditional on those designs, and says so (`scope`). The 2ε relation of Prop. 1.5 is
evaluated with the evaluation labels and is reported as a **post-hoc check**, not a certificate.

Predictive intervals of the log10 risk, each with its statement kind (§8):

* `split-conformal` (1.7.0): the outer training circuits are split at random into a proper part (70 %) and a
  calibration part; the model is fitted on the proper part; each calibration circuit gives one score per protocol
  (its maximum absolute residual over its rows); the half-width is the ⌈(1 − α)(G + 1)⌉-th smallest of the G scores
  (+∞ when that exceeds G), per protocol and Bonferroni-simultaneous over the protocols. Under exchangeability of base
  circuits every row of a new circuit is covered with probability ≥ 1 − α (Lei et al. 2018). Coverage is measured per
  held-out circuit;
* `cv-residual`: inner-CV residual quantiles placed around the refitted model (marginal, simultaneous, joint over the
  feasible protocols of a row) — calibrated in cross-validation, no finite-sample guarantee;
* `empirical`: the residual-Gaussian interval, no guarantee.

All intervals target the noisy evaluation label log10(R̂_eval + ε0), not the true risk.

## 10. Mismatch (`mismatch.py`, `run.mismatch_analyze`)

Hidden execution-law perturbations: coherent fraction χ (0.25, 0.5), static drift s (1.5, 2), readout drift (×2),
ZZ crosstalk after every two-qubit gate on chain neighbors (0.02, 0.05 rad), temporal mixture with s = 2. Diagnostic:
the echo circuit (compiled circuit then its inverse) with d(B) = max(200, 2 % of B) shots, z against the snapshot's
prediction with the same measurement map and readout; flag at |z| > 1.96 (α = 0.05). Its false-positive rate is
checked against a finite-shot binomial null by simulation. No divergence is computed on quasi-probabilities.

Reactions at **equal total budget** (1.7.0; review of Figure 6 and prompt §8 F). A mismatched task has the same
features as the matched one (features read the snapshot only), so the model's choice does not change; what changes is
the risk. Every policy spends exactly B shots: `TARGET@B` (no diagnostic, all B shots on the model's choice — "the same
shots spent on the target"), `NO-DIAG@B−d` (the diagnostic's shots withheld, isolating its information value),
`FIXED→ZNE-L` and `FIXED→SBS-B` (the 1.6.0 declared reactions, run at B − d when flagged), `DIAG-RESELECT` (a learned
reaction: when flagged, the protocol or SBS-B that minimized the flagged policy's mean log10 risk on the *training*
scenarios at that budget, trained without the held-out discrepancy type — leave one type out), `RANDOM→reaction` (the
same reaction triggered at random with the diagnostic's mean flag rate: matched rate, no information), and `BEST` (the
per-task reference). Expected risks of flag-triggered policies are exact mixtures P(flag)·R_react(B − d) +
(1 − P(flag))·R_model(B − d). Two kinds of evidence are kept apart: summaries over many repetitions (flag rates, mean
z) and what a single deployment observes (one z per deployment; `deployment_draws.csv` simulates 200 deployments per
task and budget and relates the observed |z| to the model's regret on that task, `discrimination.csv`). Detecting a
discrepancy does not identify its physical mechanism.

## 11. Review of the theory companion

Checked against primary sources and the code. Correct as stated: the crossover of Example 1.1 (B = 8/b²; 3 200 shots
at b = 0.05) and the boundary of Fig. 1.3; the bias–variance identity (1.5) and the law of total variance (1.6); the
failure decomposition (1.7); Prop. 1.4 and its Cauchy–Schwarz proof; Prop. 1.5; Thm. 1.6; Thm. 1.7 (Hoeffding with
a union bound); the projection argument for clipping; (1.23)–(1.24) by Minkowski's inequality in L²; Prop. 1.8 for
losses in [0, H]; the range [0, ln 2] of the Jensen–Shannon divergence.

Corrections and practical limits, with what the implementation does:

1. **Eq. (1.14) mixes a single-shot and a sample-mean statement.** For one signed outcome Y, Var(Y) ≤ γ²; the mean
   of N independent outcomes has Var ≤ γ²/N — and only when each shot is its own sampled circuit. With s shots per
   sampled circuit (K = N/s circuits) the variance is [γ²E_j σ_j²/s + Var_j(γ sgn(q_j) μ_j)]/K ≤ γ² s/N, which
   exceeds γ²/N whenever the instance means differ. The portfolio's PEC uses s = 1; the emulator uses the exact
   sign-resolved law, not the bound.
2. **Hoeffding certificates are uninformative at QEM risk scales.** With K = 8, α = 0.05, H = 4 the half-width is
   0.68, 0.22, 0.068, 0.022 for r = 10², 10³, 10⁴, 10⁵ trials — above every risk of interest (10⁻⁶–10⁻²);
   reaching 10⁻⁴ needs r ≈ 4.6·10⁹. Empirical Bernstein is limited by its range term ≈ 60/r. The implementation
   computes both and reports how often they certify anything (`RESULTS.md` Part II §8: never, in the pilot; Part I §I.7: never, in 1.8.0);
   fine-grained statements use the independent-half estimates and bootstrap intervals, labeled as such.
3. **Multinomial emulation assumes fixed circuits with i.i.d. shots.** True for RAW, RO, ZNE and the training and
   calibration circuits; PEC and any per-shot randomization need the two-stage law (implemented); drift within a run
   and correlated shots are outside it (§ 4).
4. **Prop. 1.8 needs the complete transcript and a bounded loss.** A diagnostic marginal does not bound the total
   variation of the transcript, so the study relates diagnostics to risk empirically (§ 10), as the companion advises.
5. **Allocation needs variances that are unknown at deployment.** The study allocates from the supplied snapshot
   model (deployable, but exponential-cost simulation at large width) and states that in the non-mismatch scenarios
   the supplied model equals the execution law, so allocations there coincide with oracle allocations.
6. Conformal coverage of *noisy* risk labels concerns those labels, not the true risks (companion §1.6.4); reported
   as such. The implemented intervals place inner-CV residual quantiles around the refitted model (the "CV"
   construction), which carries no finite-sample guarantee even for exchangeable base circuits; split conformal
   (1 − α) or CV+ (1 − 2α; Barber, Candès, Ramdas & Tibshirani 2021) would. Their empirical coverage on held-out
   circuits is therefore measured and reported (§8), not assumed.

## 12. Deployment: the recommender in the application (`deploy.py`; prompt §7)

One task representation serves the study, the command line and the Playground: `DeploymentTask` (circuit, input,
measurement bases and bit order, observable, device snapshot, budget; `from_state` for a Playground state,
`from_spec` for a study task, both compiling to the same experiment). Features are computed on request only, with
their class and cost reported per feature: deployable features need no simulation of the circuit (structure,
observable, light cone, calibration-derived sums and closed-form protocol overheads, e.g. PEC's γ from the snapshot's
calibration); simulated features run only when allowed. Feasibility is structural (minimum shots of each protocol at
B, no simulation) — **superseded in 1.8.0** by the shared preflight contract (§16.1), under which a structural check
alone never makes a learning protocol verified-feasible. A **model bundle** (`qemx/selection/bundles/<version>/`, JSON + NPZ, no pickle; sha256 of the
arrays checked on load) holds the preprocessing and model as arrays, the feature definitions and order, the
candidate protocols with their versions, the training support (devices, budgets, widths, feature ranges), the
validation summary of its own configuration (out-of-fold development result, confirmation; the nested selective
evaluation only for the frozen configuration), the selective threshold where one was validated (the frozen
configuration; a bundle without one recommends the model's choice) and the fallback, and the guarantee status of every
statement. A
recommendation outside the training support is reported as out-of-distribution and falls back to SBS-B; unsupported
experiments (non-device backends without a chosen snapshot, unsupported observables) raise an explicit error. The UI
("Recommend for my circuit") and `python -m qemx.selection recommend` return the same dictionary. The 1.5 recommender
(a ridge on the old benchmark tables) remains available, labelled as the legacy pathway. Benchmark labels are never
computed at deployment.

## 13. Hardware validation (prepared, not executed; `hardware.py`, `docs/selection/HARDWARE.md`)

A small plan (two tasks; RAW, RO, the bundle's recommendation and its SBS-B fallback; four blocks at B = 20 000)
allocated by the approved hardware path, with QPU usage estimated per job (high estimate for admission) against one
shared ceiling (600 s by default) minus a reserve (60 s), or against the campaign ledger. Execution requires the
owner's authorization of that exact plan (`QEMX_HW_AUTHORIZE` = the plan's digest), a campaign ledger and credentials
from the environment, a saved account or a hidden prompt. No job was submitted for 1.7.0.

## 14. Changes of estimators, conventions and reported quantities in 1.7.0

| what | 1.6.0 | 1.7.0 | why |
|---|---|---|---|
| CDR/NIL label estimand | risk conditional on 6 fixed designs shared by both halves | fresh-design risk, independent pools per half | R1: designs were reused, SEs understated |
| SE of CDR/NIL labels | over trials | over designs (cluster-robust) | R1 |
| policy MSE | mean over rows | mean over base circuits (row mean kept) | S5 |
| SBS ranking | each candidate on its own feasible rows | one population, declared fallback | S5 |
| selective threshold | cross-fitted on out-of-fold rows of models trained with the test fold | nested group CV | R2 |
| certificates | trials as units for every protocol | design means for learning protocols | R3 |
| 2ε relation | listed among certificates | post-hoc check | R3 |
| final test | confirmatory | historical (overlaps earlier studies); new confirmation set | R5 |
| failure rate | sum of flag rates | union of flags (`fail_any`) | S2 |
| figure 2 regret | clipped at 0 | signed (negative kept) | review §7 |

Unchanged: the protocol portfolio and its allocation rules, the budget definition (every shot charged), the
measurement conventions, the log floor ε0 = 10⁻⁶, the reference definition (identified on the selection half, scored
on the evaluation half) and the trial counts (r = 100 per half).

## 15. Limitations

Fake-device emulation only (Aer-style models without leakage or non-Markovian noise beyond the scenarios); 3–6
qubits; one realization of each calibration snapshot; finite trials per label (r = 100 per half in the pilot) and
10 fresh designs per half for CDR and NIL (few clusters for a cluster-robust SE);
the supplied-model allocation and the simulated features are exponential in width; no hardware transfer. Costs are
charged per task: calibration shared across tasks (amortized) is not analyzed; memory per task is bounded and
enforced (`limits.memory_gb`). The development set has 52 logical circuits; the confirmation set 25; claims about
"circuits in general" are limited by that diversity. The larger study (`full17.yaml`, 242 logical circuits) is planned,
not executed. Ensemble-spread intervals (across refits) were not implemented; like the
residual-Gaussian interval they would carry no coverage guarantee. The results are in `RESULTS.md`, their threats to
validity in its §17.

## 16. Changes in 1.8.0

Each change answers a finding of the 1.7.0 review (`QEM_Explorer_1.7.0_Independent_Review.md` and its follow-up;
finding codes F1–F10, N1). The issue-to-evidence matrix is `docs/ISSUE_EVIDENCE_MATRIX.md`; results are in
`docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` and `docs/selection/RESULTS.md`.

### 16.1 One execution/preflight contract (F1; `preflight.py`)

The ten 1.7.0 deployment regression cases (two circuits: GHZ in the X basis with observable Z on the middle qubit)
were accepted CDR/NIL choices whose training design was degenerate: every training circuit has ⟨X_mid⟩ = 0, the labels
have no spread, the method has no training circuits, and the study's allocation marks the protocol infeasible — but
the deployment API used a structural check that called it feasible. 1.8.0 replaces both checks by one contract used by
study labeling (as a check), `deploy.recommend` (UI, CLI, batch), the replay and the hardware plan:

| status | meaning |
|---|---|
| `verified-feasible` | the protocol can be executed as defined at this budget (at the stated level) |
| `infeasible` | it cannot (with the reason) |
| `unverified` | the cheaper level cannot decide; it is never accepted |

Levels, cheapest first: **structural** (minimum shots of every role at B; decides every non-learning candidate and
rejects learning candidates whose minima cannot be met), **design** (a learning protocol's training design is built
with its deployment seed and its ideal training values computed by state vector — information genuinely available
before execution, never the target's ideal value; up to 20 qubits), **allocation** (the study's own rule on the
supplied model; needs simulation, `--allow-simulation`). The recommender walks the candidates in the model's
predicted order, resolves an unverified candidate at the requested level, skips infeasible and unresolved ones (each
recorded with its status) and returns the first verified candidate; the margin is taken to the next candidate not
known to be infeasible; the frozen τ and the fallback (SBS-B's first **verified** candidate; RAW if none) follow.
Preflight uses no quantum shots; its classical seconds are reported. Agreement of the design level with the 1.7.0
labels' allocations: 27 760 rows of 334 tasks, 0 disagreements (`evidence/preflight/agree_v170_sv.json`, private
archive). Every selection, OOD, threshold, feasibility, execution-failure and fallback rule is part of the evaluated
policy (`replay.py`, §16.9), and the policy has a version (`qemx.selection.policy/2`).

### 16.2 Bounded-cost deployment (F2; `deploy.guard`, `device_exact.product_calibration`)

The budget, the device (a fake-device snapshot), the width against the device, and the observable (a dense vector only
up to 20 qubits, otherwise valid compact Z-string terms with Σ|c| ≤ 1) are checked before compilation (`deploy.guard`);
inputs beyond them raise `Unsupported` with the reason. Calibration marginals are computed by a factorized local-channel
calculation when the device model's readout and preparation channels are products over qubits (all fake-device
snapshots: X-gate preparation error, relaxation, readout asymmetry, layout and measurement maps, relabeling; maximum
|dense − factorized| 1.8 × 10⁻¹⁵ on five configurations); a correlated or context-dependent model raises
`NotFactorizable` and uses the dense path only up to 11 qubits, otherwise the feature is reported missing (imputed, and
flagged as OOD). Observables of up to 20 qubits may be dense vectors; wider tasks pass compact Pauli terms
(`obs_terms`). A sentinel test replaces every dense engine by a tripwire and recommends a 24-qubit task. Features are
cached by content (the experiment without its budget, the feature names and definitions, the simulation switch, the
policy version); timings in `supp_runtime` and thesis table 6 are measured
without the cache. The studied allocation of a learning protocol (and PEC's γ) still needs supplied-model information
whose cost grows with width; the fast inference does not make the complete protocol scalable.

### 16.3 Provenance and compatibility (F3, F4, F9; `binding.py`, `deploy.compatibility`, `textio.py`)

Mismatch labels carry their run id. Collection checks each record's task id, label configuration, run id (or an
auditable import with its JSON digest) and NPZ digest; a wrong JSON digest is rejected even when the arrays match.
Records of 1.6/1.7 without a run id and without a seed scheme are `legacy_unbound` evidence, never mixed into a
scheme-2 run. Bundles record a feature fingerprint (definitions, order, normalization), the protocol definitions, the
policy version and the device-snapshot fingerprints: a changed definition is `BundleIncompatible` (refused); a changed
calibration of a known device is a transfer, reported as out-of-distribution. Current JSON is strict (`null` plus a
`"$nonfinite"` map of JSON pointers); older `NaN` tokens remain readable.

### 16.4 Preparation seeds (N1; `emulate.stream_seed`)

Seed scheme 2 derives every preparation and trial stream from stable identifiers — (root, CRN key, protocol, half,
index, role) — instead of one per-study seed pool reused across tasks. Common random numbers are explicit groups
(`crn: task` or `crn: base`); a design has a seed token and a unique `instance_id`. The scheme is part of the run and
record identity; a resume cannot mix schemes. Scheme 1 reproduces the 1.7.0 labels bitwise
(`evidence/seeds/v1_reproduces_v170.json`); scheme-2 labels are unchanged by task order, resume and worker count
(`tests/test_provenance_v180.py`). The 1.7.0 trials are not altered; every 1.8.0 study uses scheme 2.

### 16.5 Variance of one deployment (F5; `labels.deployment_variance`)

For fresh designs the pooled trial variance s² has expectation σ²_W + m(K − 1)/(Km − 1) σ²_D (balanced groups), not
σ²_W + σ²_D. Labels now also report `var_deploy` = σ̂²_W + σ̂²_D from the one-way random-effects ANOVA (method of
moments; unbalanced groups through m₀ = (n − Σn_k²/n)/(K − 1); σ̂²_D truncated at 0, flagged). Mean squared losses and
the MSE labels are unchanged; `var` keeps its pooled meaning.

### 16.6 Primary metric wording and declared tails (F6, F7; `evaluate.tail_metrics`)

The primary objective stays the mean over base circuits of log10(MSE + 10⁻⁶). A difference Δ is a factor 10^Δ in the
geometric mean of (MSE + 10⁻⁶), not a percentage of the MSE; reports and figures say so. Declared before any 1.8.0
result: severe degradation p_severe_vs_raw (risk > 10 × RAW's on the same row), p_severe_abs (risk > 0.1), the 95 %
quantile of the risk and the worst-5 % loss share — reported beside, never instead of, the primary metric. Severe ZNE
and learning failures stay in every risk. A bounded estimator is a **separately named candidate**: ZNE-E-B projects
ZNE-E's estimate onto the observable's eigenvalue range (deterministic, never increases a trial's error; paired with
ZNE-E by common random numbers); ZNE-E is unchanged and historical estimates are never clipped.

### 16.7 Independent shot-level references (F8; `reference.py`)

An independently written reference draws counts from the exact output laws and re-implements the estimators of
ZNE-E and ZNE-E-B, CDR, FR, NIL (with its own ℓ1 solver), symmetry verification and ZNE∘SV, TREX and RO, including
training designs and labels, calibration, target shots, degeneracy, fit/solver failure, per-replicate fallback flags,
zero acceptance and the budget ledger. Emulated and reference trials are compared in distribution (mean z score,
variance-ratio F test); the linear and transform candidates are compared with direct shot-by-shot execution
(`tests/test_emulator_independent.py`). The registry's `validation` field states which evidence each candidate has.

### 16.8 Candidate portfolio-1.8 (`protocols.py`, `registry.py`, `docs/selection/CANDIDATES.md`)

The eight 1.6/1.7 candidates (portfolio-1.6, immutable identities) plus RO-TREX (twirled-readout correction), RO-TW
(Pauli-twirled two-qubit gates), RO-DD (XY4 dynamical decoupling), ZNE-L-TW, ZNE-E-B, PEC-TW, FR (feature regression,
this project's CDR extension), SV (symmetry verification of declared Z parities) and ZNE-L∘SV (every fold
post-selected). Each has a capability record: applicability, prerequisites, resources, ledger, failure handling,
allocation information and validation. Not candidates: virtual distillation (the implementation has no finite-shot
multi-copy protocol; blocked with the reason) and M3 (the emulator applies the tensored inverse; a separate identity
would duplicate RO). SV applies only when the task's construction declares its symmetry (never inferred from the
ideal output); an observable that is a function of the checks is trivially exact after post-selection, which is why
the GHZ-Z tasks use Z on the middle qubit.

### 16.9 Full-policy evaluation (F1; `replay.py`, `scripts/replay_confirmation.py`)

The confirmation is evaluated twice, and both are reported: the **model component** (the model's best label-feasible
candidate, as in 1.7.0) and the **full policy** (`deploy.recommend` itself on every confirmation task and budget,
scored with the evaluation half). Every row is kept; a returned candidate without feasible labels is unscored and makes
the primary difference undefined rather than imputed. The default bundle uses deployable features only; the
simulator-feature bundle is evaluated at the allocation level.

### 16.10 Features and models

Four calibration-derived descriptors join the deployable set (`deployable18`): λ_LC of two-qubit gate errors only
(`lam_lc_2q`), readout asymmetry (`ro_asym`), the spread of the used two-qubit gates' reported errors
(`err2q_spread`) and the scheduled duration over the median T2 (`duration_t2`). The 1.7.0 sets keep their names and
fingerprints. Model families: Ridge, elastic net, histogram gradient boosting, a small MLP and the structured risk
model, each per feature set; the frozen configuration is chosen as before (lowest pooled out-of-fold mean log10 risk)
and validated by nested cross-validation. A scalar Pauli-twirled light-cone exposure cannot distinguish coherent errors
that reinforce from ones that cancel (`tests/test_lightcone_v180.py`, F10); features are descriptors, not
mechanisms.
