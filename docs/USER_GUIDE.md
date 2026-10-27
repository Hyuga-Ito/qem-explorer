# QEM Explorer 1.9.0 — user and developer guide to the research parts

This guide covers the experiment model, the statistics, the costs, the studies and the hardware path. For the app's
pages see `README.md`; for the equations and sources see the Theory page (`docs/theory.md`). Conventions:

- bitstrings are written c(n−1)…c0;
- integers are little-endian (q0 = 1s bit);
- angles are in radians.

## 1. Circuits: input, output and presets

**The editable experiment** is defined by four things:

- the **input**: one single-qubit state per qubit, the state the circuit starts from;
- the **circuit**;
- its **parameters**;
- the **measurement**: the basis per qubit and the classical bit order.

**One place for the input (1.5.1).** The Playground's *Inputs and measurement* panel is the only place where inputs
are set, and what it shows is exactly what is simulated:

- **Per qubit**: `|0⟩ |1⟩ |+⟩ |−⟩ |+i⟩ |−i⟩`, a phase state `|φt⟩ = (|0⟩ + e^{2πit}|1⟩)/√2` (t in turns) or a general
  state `|θ,φ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩` (radians). The register-wide box takes one state for every qubit
  (`plus`, `ph:0.25`, `bloch:1.2,0`), `ramp:k` (the Fourier state F†|k⟩), `bits:0101`, or one state per qubit
  separated by spaces, q0 first (`0 + 1`).
- **A template's input settings** — the QFT's *input index s* and *input preparation*, the eigenstate of the Ising
  QPE, the logical θ of the error-correcting codes — appear in this panel, not in *Start from a template*, and apply
  at once: they write the per-qubit states. They never change the circuit. Templates with a fixed input (QPE's
  eigenstate |1⟩, Bernstein–Vazirani's ancilla |−⟩, Shor's work register |1⟩) load it into the per-qubit states too.
  *Start from a template* keeps only genuine circuit parameters (size, bit-order convention, AQFT degree, oracle,
  code, error, …).
- **Prepared once.** The compiler prepares the input before the circuit; the circuit holds no preparation gates.
  The circuit drawing shows each input and its preparation gates at the start of the wire.
- **Edits are never "corrected".** If you change a qubit's state, that is the input; the panel says that it differs
  from the template's own input and offers *Use the template's input*. When the inputs equal the template's input for
  another setting (for example `ramp:3` is the QFT's Fourier input with s = 3), the template's controls show that
  setting. The template's own observable (for example P(s) of the QFT) follows its input until you choose another
  observable.

**The Output panel is derived and read-only.** It shows the *ideal prediction*: the noiseless output distribution
and the observable's ideal value, computed from the statevector of the actual input, the circuit, the measurement
bases and the bit order (listed in the panel). It is recomputed whenever one of them changes; it has no controls, and
nothing in it can change an input. Above 20 qubits it states that the preview is unavailable, and why. That is a limit
of the preview only; experiments have no hard qubit or depth limit. The noisy simulation prediction ("noisy
(unmitigated)") and the methods' estimates are labeled separately in *Mitigate & compare*; after any change of
input, circuit, measurement or noise they are marked out of date (header badge, Output panel, *Mitigate & compare*)
until you press *Run simulation*.

**Input, output state, measurement and observable (1.7.0).** The Output panel keeps four things apart, because a
change of measurement basis changes what a bitstring *means*, not the state:

- **A. Input state** — the per-qubit states that are prepared (Bloch description per qubit);
- **B. Output state before measurement** — the ideal state after the circuit, independent of the measurement bases
  (a product state is named per qubit; an entangled or superposed state is described honestly, e.g. "superposition of
  4 basis states", never forced into one bitstring);
- **C. Measured distribution** — the distribution of the classical bits in the chosen bases and bit order (with
  X bases, P(0000) is the probability of four "+" outcomes, and the panel says so);
- **D. Observable** — its ideal value and physical meaning (e.g. P(s) = probability of reading s in the chosen
  bases).

A template's identity ("F F†|0⟩ = |0⟩", "no SWAPs: outputs relabeled classically") is shown as a *prediction* only
while its assumptions hold (the template's own input and Z bases); otherwise it is shown as a property of the
template with the reason it does not apply to the current experiment. The screenshot case of the review — QFT with
s = 0, Fourier preparation and every input set to |+⟩ — is physically consistent: F†|0…0⟩ = |+⟩^⊗n, so "plus" is
the same input, the output is |0…0⟩ and P(0000) = 1 in Z bases. *Apply to all* reports whether it changed anything
and which qubits; changing inputs or bases afterwards clears that message. Inputs and bases are independent: a basis
change never edits an input.

**QFT templates.** `input preparation = fourier` prepares F†|s⟩, whose ideal output is |s⟩ (a known-answer
benchmark); `basis` prepares |s⟩, whose ideal output F|s⟩ is uniform in the Z basis. The convention is
F|y⟩ = N^(−1/2) Σ_k e^(2πi·yk/N)|k⟩ (tested against the analytic formula with two simulators). Bit order: the SWAP
network is part of the circuit; *relabel outputs* reads the bits in reverse order (the measurement); *reversed input*
is a documented input convention: it loads F†|rev(s)⟩ so that the circuit without SWAPs outputs |s⟩, and the Inputs
panel says so.

**Cited presets.** The *Cited preset* selector (`qemx/presets_cited.py`) lists, per preset, the source with arXiv
version and locator, units and conventions, the instance and sizes, purpose and limitations, and whether it is an
**exact reproduction**, **adapted** (with its differences) or **educational**.

## 2. Experiments as files (UI ↔ YAML ↔ CLI)

```yaml
schema: qemx.experiment/1
name: my-experiment
circuit: {template: qft, params: {n: 4, s: 5, mode: swap, prep: fourier}}   # or {n: 3, text: "h 0; cx 0 1"}
observable: {bitstring: "0101"}          # or {z: 0}, {parity: [0, 3]}, {pauli: "Z0*Z1 + 0.5*X0"}
noise: {preset: "…"}                     # and/or fields: p1, p2, ro01, ro10, amp_damp, …
backend: numpy                           # or a fake device, e.g. fake_fez
methods:
  - {preset: "ZNE · Richardson"}
  - {estimator: zne, readout: tensored, params: {rule: exponential, on_fail: linear}}
budget: {shots: 100000}
run: {seed: 0, repeats: 120, check_budget: true}
```

`python -m qemx.experiment validate | plan | run FILE --out DIR`. `validate` and `plan` never simulate; `run` writes
`result.json` and `rows.csv` and refuses real hardware. The Playground's *Download experiment (.yaml)* and the CLI
resolve to the same state. Credentials anywhere in a document are refused; YAML is read with `safe_load`.

**Inputs in documents (1.5.1).** A template's input becomes the resolved state's `inputs` and its circuit holds the
rest, exactly as in the Playground (loading a template there and resolving the short form give the same state and
hash). Set a template's input with its input parameters in `circuit.params`. A short-form `inputs` is accepted with a
template that loads its own input only when it equals that input (or is all `"0"`); anything else is refused, since it
would prepare twice. In a batch sweep, `input:` replaces the input, the template's own included.

**Provenance (1.5.1).** Every result document, phase record and run file carries `source_sha256` (a digest of the
application source as it is on disk — it works on an unpacked ZIP without git), `lock_sha256` (of
`requirements-lock.txt`), `git_commit` when there is one, the versions of the scientific packages and
`inventory_sha256` (all installed distributions; the full list in a study's `run.json`). A configuration hash says
what was asked; these digests say which code and environment answered. Run files replace your home folder by `~`.

## 3. Statistics of an estimate (`qemx/stats.py`)

A result carries `estimate_kind`:

| kind | meaning |
|---|---|
| `exact` | no finite training and linear in the target data: `mean` is the expected estimate at infinite target shots, `var1` the per-shot variance |
| `protocol` | learning methods (CDR, FR, NIL) with finite training data D: `mean` = E_D[m(D)], Var = Var_D[m(D)] + E_D[v(D)]/N_t, by Monte Carlo over training draws; the exact-training fit is `reference_mean` |
| `reference` | estimators nonlinear in their *target* data (exponential ZNE, symmetry verification, chains with an exponential outer ZNE): `mean` is the fit to exact probabilities, kept as a reference; tables and curves use **budget-dependent protocol moments** (below) |
| `measured` | one hardware experiment (a replay of measured counts) |

**Target-nonlinear estimators (1.5.1).** A log-fit or a ratio of sampled quantities has a finite-shot mean that
differs from its value at exact probabilities, and the fit can fail (a sampled E(λ) ≤ 0, no accepted shot). Such a
method declares a **failure policy** before it runs:

- exponential ZNE: `on_fail: linear` (default; the least-squares line through the same data, no extra shots) or
  `abstain` (no estimate; its moments are conditional on success and the abstention rate is reported);
- symmetry verification: abstain when no shot is accepted.

At the evaluated budget the comparison table uses Monte-Carlo moments of the complete target experiment (columns
`fit failures`, `failure policy`, `MC experiments`; `sd from` = "protocol MC"). Cost–precision curves use a moment
grid up to the budget where the first-order (delta-method) values are validated, then the first-order formula; each
point records its source. On the review's case (one qubit, ry(1.2), p1 = 0.15) first order holds only from about
2 400 target shots; below that the delta-method SD understates the spread by up to about 39 %. At small budgets the
moments also jump between neighboring shot counts (integer counts); curves average four adjacent counts.

**Measured rows.** A replay of one hardware run is one experiment: its signed deviation e = estimate − ideal is
shown as **signed error to reference**, and |e| as **absolute error**, not as "bias" and "RMSE". e² is an unbiased
observation of the MSE; |e| is not an unbiased RMSE. Statistical bias and RMSE need repeated complete experiments
(the hardware transfer test, section 8).

Validation: `python scripts/validate_learning_moments.py` (learning methods) and
`python scripts/validate_protocol_moments.py` (nonlinear target estimators, an independent sampling path, 20 000
experiments per case, |z| ≤ 4).

## 4. Costs and budgets (`qemx/budget.py`, `qemx/phase/ledger.py`, `qemx/hardware.py`)

- **Equal total shots.** Every contender spends the same total B per estimate: target, training, calibration,
  folded, neighbor and sampled circuits included. Integer allocations add up to B exactly in the simulator, the phase
  ledger and the hardware planner; a protocol that cannot spend all of B records an `effective_budget` and why.
- **One feasibility rule.** The minimum budget of a protocol is the same everywhere (`budget.Feasibility`): one shot
  per circuit (PEC: its shots per sampled circuit), ⌈5/P_accept⌉ target shots with postselection, the calibration and
  training minima of the policy. The study pipeline uses at least 32 shots per calibration circuit; the Playground
  allows 1.
- **PEC on hardware.** The configured shots per sampled circuit s are kept and K = ⌊N_t/s⌋ circuits are sampled. A
  hardware cap on K raises s; that is recorded as requested vs effective protocol (`Plan.protocol_table()`), and the
  transfer predictions simulate the effective protocol (`Plan.effective_recipes()`).
- **Information regimes.** Results say what they were allowed to know: `oracle` (simulator noise parameters or
  exact-probability allocations), `supplied-calibration` (typed-in rates, a device calibration snapshot) or
  `learned-with-cost`. ZNE's optimal shot split uses model variances (an oracle allocation in simulation).
- **Equal shots are not equal QPU time.** The hardware planner keeps shots, executions, distinct circuits, circuit
  instances, Sampler calls and a QPU-seconds estimate apart (`Plan.resources()`).

## 5. Discrepancy (`qemx/discrepancy.py`)

Distances between probability distributions (validated first): TVD, Hellinger, the classical fidelity of Chen et al.
(Eq. 1, unnormalized) and Lubinski et al.'s normalized fidelity. A signed vector raises `QuasiProbabilityError`;
`project_to_simplex` is explicit and reports the mass it moved, and the divergence tables carry `moved_mass_obs` and
`moved_mass_pred_*` for projected RO/ZNE distributions.

`distance_test` compares the observed plug-in distance with its matched-shot sampling distribution. With an
uncertain model (a finitely sampled calibration, a fitted noise model), pass draws of the model distribution
(`p_draws`): the null is then a **mixture of multinomials** (one parameter value per experiment). Averaging the
distributions into one multinomial is not equivalent and is anti-conservative when the parameter uncertainty is not
negligible. `adjust_pvalues` gives Holm, Hochberg, Benjamini–Hochberg and Bonferroni. A large residual alone does not
attribute an error to a physical mechanism.

## 6. Features and the main study

**Features** (`docs/feature_dictionary.md`, schema `qemx.features/2`) carry a level and an acquisition tier:

1. circuit, observable and budget (logical and compiled counts, observable weight/terms/settings);
2. the calibrated noise model (light-cone fault rates λ_LC, readout, scale, coherent fraction) — model-dependent
   descriptors, not sufficient statistics;
3. exact ideal-state entropies (exponential in n; from the smaller side of each cut);
0. oracle features, never recommendation inputs.

Cheap features never depend on exact state work. Pauli-sum observables have setting-aware descriptors (terms,
qubit-wise-commuting settings, weights, support). `two_qubit_compile_overhead` (schema /1: `routing_overhead`) mixes
decomposition, routing and optimization; `non_clifford_slots` counts the compiled slots a training generator
replaces. Old tables are read through an alias map.

**Manifests.** `design: {kind: maximin, per_stratum: k}` chooses space-filling cells per (family, n). The
explore/confirm split is **one policy for the whole study** (a top-level `confirmatory: {fraction, seed}`, or
identical stage policies), by base circuit; conflicting policies are refused and disjointness is checked before
running and before evaluating.

**Analysis** (`python -m qemx.study report RUN --manifest M --out DIR`), from stored records only:

- regime maps without interpolation; practical ties (within δ = 10 %) are separate from *MC-unresolved* labels
  (`p_winner` < 0.9 when the RMSEs are redrawn from their Monte-Carlo errors);
- paired changes against raw with base-circuit bootstrap intervals, and the improvement share **among all planned
  cells** (a cell that cannot run counts as not better) next to the share **among feasible cells**, with counts;
- cost to reach ε;
- the recommender evaluation (below) and the discrepancy rehearsal.

**Recommender** (`qemx/recommend.py`): ridge models of log10 RMSE per method on recommendation-time features, with
abstention ("abstain-coverage", "abstain-uncertain") and "use raw" when no method is predicted to help by more than
δ. Decision thresholds and intervals use **out-of-fold residuals by base circuit**; the interval is a
group-cross-validated conformal interval (nominal 90 %; marginal, assumes exchangeable base circuits). Evaluated by
grouped folds, leave-family-out, leave-width-out and the confirmatory subset, against always-raw, the best fixed
method and a rule baseline, with: coverage, accuracy, regret median and tails, the probability of a harmful choice,
the **full policy** (fallback to the training fold's best fixed method on abstention) and paired log10 RMSE ratios
against the best fixed method on the same rows. Feature tiers are compared as ablations. A live recommendation
(page *QEM Performance & Selection*) reproduces the study's protocols and noise context or abstains.

**The bundled pilot** (`data/pilot_v2/`, `reports/pilot/`): 75 cells, 25 base circuits, fake_fez. Its baseline
recommender does not beat the best fixed method, and no feature tier changes that. `python
scripts/reproduce_pilot_report.py --out DIR --compare reports/pilot` regenerates the report from the records.

**Final holdout.** `docs/FINAL_HOLDOUT.md`: 23 new base circuits, frozen, evaluated once with
`python -m qemx.study holdout DEV_RUN HOLDOUT_RUN` after the method is frozen.

## 7. Supporting study: discrepancy as a decision tool (`qemx/decision.py`)

Does a cheap diagnostic, paid for from the same budget, make a model-based choice of protocol safer? A D-shot TVD
check of the unmitigated target circuit against the model's prediction switches to a fallback protocol when it
flags; the policy is evaluated under the truth with the diagnostic's cost inside, on held-out circuits and mismatch
types. `reports/diagnostic_policy/report.md` (EMULATION): with the pre-declared fallback (readout correction only) the
diagnostic made every held-out scenario worse; with a model-free ZNE fallback (a post hoc sensitivity analysis) it
helped only under amplitude damping. The check is most sensitive to readout errors, which the protocols' own
calibration already corrects — it is not aimed at what the decision depends on.

## 8. Hardware

`docs/hardware_preflight.md`: one shared 600 s ceiling, the transactional campaign ledger, and the main study's
transfer test (`qemx/transfer.py`, `scripts/hardware_transfer.py`): QFT-4 and BV-4, four effective protocols, four
paired blocks, signed-error endpoint, 331 s reserved. The emulation of the whole path on the fake device reproduces
its predictions (`reports/hardware_transfer/rehearsal_emulation`). No hardware result exists in this release.

## 9. Performance and concurrency

`scripts/profile_workflows.py` profiles representative workflows in fresh processes. Compiled circuits and
time-travel traces are shared between sessions as values that are not mutated; comparisons (mutable runners and
outcomes) are cached per session. The export renderer runs one bounded process at a time; its wait counts toward
its time limit, and its whole process tree is stopped on a timeout. A single long native call (a large eigensolve,
a transpilation) cannot be interrupted inside the process: public hosting needs isolated workers and memory limits.

## 10. Known limitations

- **One device model.** All study numbers are simulations on fake_fez (or the injector's noise). Transfer to hardware
  is untested until the campaign runs.
- **Small pilot.** 25 base circuits; the recommender evaluation is correspondingly uncertain, and its negative result
  is the current finding.
- **Monte-Carlo moments are estimates.** Their errors are recorded and propagated into the label stability and the
  recommender's sensitivity, not into every interval.
- **Sources not fully read.** See `docs/literature_matrix.md`.

## 11. Protocol selection (`qemx/selection/`, `docs/selection/`; since 1.6.0)

The study asks which complete QEM protocol to run on a task at a fully charged shot budget, and when a learned
choice beats a fixed one. 1.8.0 expands the candidates to 17 (`docs/selection/CANDIDATES.md`), evaluates the deployed
application policy itself (not only its model component) and adds the documents
`docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` (interpretation, open questions, ranked next experiments) and
`docs/ISSUE_EVIDENCE_MATRIX.md` (every review finding, its fix and its evidence). It is independent of the Direction A pipeline (§6) and has its own documents:

| document | content |
|---|---|
| `docs/selection/METHODS.md` | task representation, candidates and ledgers, emulation, labels, features, models, policies, evaluation conventions, certificates, mismatch; the review of the theory companion |
| `docs/selection/RESEARCH_PLAN.md` | the plan and the dated decision log |
| `docs/selection/DATA_DICTIONARY.md` | every column of every saved table |
| `docs/selection/COMMANDS.md` | commands, workflows, outputs, resource limits |
| `docs/selection/RESULTS.md` | what was executed, the results and their limits, compute and estimates |
| `docs/selection/FIGURES.md` | every figure and table: its question, panels, inputs, plotting configuration and checks; the thesis tables |
| `docs/selection/CANDIDATES.md` | the 17 candidates' capability records (generated: `python -m qemx.selection registry --markdown`) |

A typical session: `python -m qemx.selection plan CONFIG` (task count and a resource estimate), `label`
(resumable; interrupt and restart at will), `analyze`, `figures`, `report`; `freeze` and `final` once, at the end;
`regimes` (the controlled regime study) and `profile` (classical cost against width) after the freeze.

**Since 1.9.0** a whole study is better run with the pipeline runner, which knows the order of these steps and what
each has produced: `python -m qemx.selection pipeline configs/pipelines/pilot19.yaml --out runs/pilot19 --workers auto`.
Interrupt it at any moment and rerun the same command: complete labels are verified and kept, a freeze or once-only
evaluation that exists is verified (never redone), and a failed stage stops everything that depends on it. Before a
large run: `check-config` (the file's format and semantics), `plan --tasks-csv` (the expanded task table — which
variants, inputs and devices the configuration actually produces) and `plan --measured LABELS_DIR` (CPU, disk and
memory from measured rates against the configuration's `limits`). While and after it runs, `status --out RUN` gives
the completion of every labels directory. Exit codes: 0 complete, 1 incomplete (failed or missing tasks, reported on
every rerun), 2 refused, 3 a cap would refuse the run (`plan`). Failed tasks are retried only with `--retry-failed`;
corrupted arrays are moved aside only with `--quarantine-corrupted`. YAML formats and expansion rules:
`docs/selection/YAML_FORMATS.md`; the next study designs and their costs: `docs/selection/RESEARCH_PLAN_v19.md`. The bundled
studies are shown on the QEM Performance page, tab *Protocol selection study*: the 1.8.0 study
(`reports/selection_v180/`) first, then the corrected pilot of 1.7.0 (`reports/selection_v170/`) and the 1.6.0 pilot
(`reports/selection_pilot/`), marked historical. For each:
the policy table; a task drill-down — every protocol's risk per budget and each policy's choice, then the task and its circuit
(with a text drawing of the logical circuit on request), the protocols' settings and allocations at a chosen budget,
the shot ledger (calibration, training, diagnostics, selection, execution), the distribution of the repeated trials
of both label halves against the ideal value (needs the per-trial arrays: for the corrected pilot, extract the
trial-data archive, `reports/selection_v170/README.md`), and the provenance of the labels (code commit, lock,
packages, seeds); and the figures with their notes.

Three conventions to keep in mind when reading its numbers: risks are mean squared errors over repeated *emulated*
executions on fake devices (not hardware); "SBS" is the single best fixed protocol ranked on the training base
circuits (the next one in its ranking where it is infeasible); and every interval is a bootstrap over held-out *base circuits*, so variants of one circuit never count as
independent evidence.


## 12. The recommender (`qemx/selection/deploy.py`, QEM Performance → *Recommend for my circuit*)

The recommendation tab turns the Playground's experiment — circuit, inputs, measurement bases and bit order,
observable — into the same `DeploymentTask` the command line uses (`python -m qemx.selection recommend --state
STATE.json --budget B --device D`). Choose a model bundle, the device snapshot the protocols would use (the Playground's
backend when it is a fake device) and the total shot budget B, then *Recommend*. 1.8.0 runs policy
`qemx.selection.policy/2`, the same policy the study evaluated on its confirmation set:

1. a **guard** checks the request before compilation (width, depth, observable form) — a wide circuit is rejected
   early, never sent to a dense engine;
2. the **features** the bundle needs are computed (deployable features only unless simulation is allowed), and
   **out-of-distribution** checks run (device, budget, width, features outside the training support);
3. a **preflight** gives every candidate a status — *verified-feasible*, *infeasible* or *unverified* — at the
   structural level (ledger and budget arithmetic), the design level (a learning candidate's training design is built
   and its ideal training values computed) or the allocation level (the study's own allocation on the supplied noise
   model; needs simulation, at most 11 qubits);
4. the candidates are walked in the order of their predicted risk: the first **verified** one is the model's choice;
   infeasible and unverified candidates are skipped and listed (*skipped candidates*) — an unverified candidate is
   never returned;
5. the frozen margin rule (bundles that have one) and the **fallback**: SBS-B's first verified candidate when the
   request is out of distribution or the margin is below τ.

The result shows the choice and why, the skipped candidates with their reasons, each candidate's predicted log10 risk
(model-conditional) with its preflight status and level, every feature with its information class (available /
simulated), value, cost and method, the execution note, and each candidate's capability record (what it can mitigate,
what it needs, how it spends B). Unsupported experiments (no device snapshot, an observable the study does not cover)
are reported, not guessed.

Four bundles are shipped:

| bundle | features | preflight level | evidence (1.8.0 confirmation, full deployed policy vs SBS-B) |
|---|---|---|---|
| `qemx-rec-1.8.0-deployable` (preselected) | deployable (`deployable18`), no simulation | design | −0.084 [−0.179, −0.002] — favourable, not robustly resolved; on GHZ tasks it cannot verify symmetry verification without a simulation |
| `qemx-rec-1.8.0` | deployable + simulator-only (`deployable18+simulated`); needs *Allow simulated features* | allocation | −0.144 [−0.231, −0.076] — the configuration the study's declared rule established (simulation, 3–6 qubits) |
| `qemx-rec-1.7.0-deployable`, `qemx-rec-1.7.0` | the 1.7.0 sets | as above | the 1.7.0 study; they still load (recognized by their recorded fingerprints) and run under policy/2 |

Only deployable features are computed by default — no simulation of the circuit. A bundle whose model uses simulated
features asks for permission (*Allow simulated features*); their cost grows exponentially with width (about 0.2–0.9 s
at 3–7 qubits on the build machine). A bundle whose candidate or feature definitions differ from the code is refused
with the reason; a device whose calibration snapshot changed since training is treated as a transfer (out of
distribution). A recommendation is marked out of date when anything it depends on changes — since 1.9.0 including
the *Allow simulated features* switch. Since 1.9.0 the result also shows the **effective device** (an explicitly chosen
device overrides the experiment's backend), the requested and physical **layout**, the **declared symmetry** used and
where it came from (a study construction, or the experiment's declaration after it was checked to hold on the ideal
measured distribution — a declaration that does not hold is not used, with a note), and any conversion notes. The
Playground state, a YAML experiment and the command line therefore give the same recommendation for the same
experiment. Benchmark labels (the
study's emulated trials) are never computed here. The 1.5 recommender (a ridge on the old benchmark tables) is in the
expander labelled *Legacy 1.5 recommender* (another model, data, candidate set and budget definition).
