# QEM Explorer 1.5 — user and developer guide to the new parts

This guide covers what changed in 1.5. For everything else, see `README.md`, the Theory page (`docs/theory.md`) and the per-version notes under `docs/v13`, `docs/v14`. Spelling and conventions:

- bitstrings are written c(n−1)…c0;
- integers are little-endian (q0 = 1s bit);
- angles are in radians.

## 1. Circuits: input, output and presets

**The editable experiment** is defined by four things:

- the **input**: the per-qubit input states, plus any preparation gates a template puts at the start (tagged `prep`, drawn as preparation);
- the **circuit**;
- its **parameters**;
- the **measurement**: the basis per qubit and the classical bit order.

**The Output panel is derived and read-only.** It shows the *ideal prediction*: the noiseless output distribution and the observable's ideal value, computed from the statevector. Above 20 qubits it states that the preview is unavailable, and why. That is a limit of the preview only. Experiments keep their own memory checks and have no hard qubit or depth limit.

An output is in general a **distribution**. The panel says whether the prediction is deterministic, uniform or spread out. Two other outputs are labeled separately in *Mitigate & compare*:

- the noisy simulation prediction ("noisy (unmitigated)");
- the methods' estimates.

**QFT templates.**

- `input preparation = fourier` prepares F†|s⟩ (one H and one phase per qubit; F†|0⟩ = |+⟩^n). The ideal output is |s⟩: a known-answer benchmark.
- `basis` prepares |s⟩. The ideal output F|s⟩ is uniform over all 2^n outcomes, and its input-dependent phases are invisible in the Z basis.
- The convention is F|y⟩ = N^(−1/2) Σ_k e^(2πi·yk/N)|k⟩. Tests check it against this analytic formula with two independent simulators (`tests/test_io_semantics.py`).

**Cited presets.** The template card has a searchable *Cited preset* selector (`qemx/presets_cited.py`). Each preset lists:

- the source with arXiv version, and the locator (equation, section or figure);
- units and the gate/angle convention;
- the problem instance and the sizes it is meant for;
- its purpose and limitations, and the resolved values.

Each preset is labeled **exact reproduction**, **adapted** (with its differences) or **educational**. None is "recommended by the literature": there is no universal optimal QAOA angle or depth. The QAOA ring preset reproduces the p = 1 ratio 3/4 in the app's own convention (tested for n = 4, 6, 8).

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
  - {estimator: learning, params: {features: fr, model: ridge, alpha: 0.001}}
budget: {shots: 100000}
run: {seed: 0, repeats: 120, check_budget: true}
```

`python -m qemx.experiment validate | plan | run FILE --out DIR`:

- `validate` and `plan` never simulate. `plan` prints the per-method budget preflight and the configuration hash.
- `run` writes `result.json` and `rows.csv`, with method ids, the configuration hash and provenance: versions, git commit, Python and platform.
- `run` refuses real hardware.

The app's *Download experiment (.yaml)* and the CLI resolve to the same state. `tests/test_experiment.py` checks that the Playground's numbers are reproduced exactly from the YAML through the CLI.

## 3. Statistics of an estimate (`qemx/stats.py`)

A result carries `estimate_kind`:

- **exact**: no finite training. `mean` is the expected estimate at infinite target shots, and `var1` is the per-shot variance.
- **protocol**: learning methods with finite training data D. `mean` = E_D[m(D)] and `var1` follows Var(Ô) = Var_D[m(D)] + E_D[v(D)]/N_t, by Monte Carlo over training draws until the declared tolerance is met (`moments`). The exact-training fit is kept as `reference_mean`.
  - If some refits fail, the moments are conditional on success (flag `conditional-on-success`).
  - If more than half fail, the method reports a failure instead of numbers.
- **measured**: one hardware experiment. Its uncertainty comes from a bootstrap of the measured counts.

Shared calibration (tensored or TREX readout used for both training and target data) enters through a jointly sampled calibration term w. Its effect on the mean is first order (flag `calibration-mean-first-order`).

## 4. Costs and budgets

- **Equal total budget.** Every comparison spends the same total number of shots B per estimate: target, training, calibration, folded, neighbor and sampled circuits included. The ledger (`qemx.phase.ledger`) allocates B optimally inside each method.
- **Budget preflight.** Before anything runs, each contender's fixed cost is checked. An unaffordable one shows "insufficient_budget: needs at least N shots".
- **Two resources, kept apart.** Equal *shots* and equal *QPU usage* are different resources. The hardware planner (`qemx.campaign`) works in QPU seconds; classical cost is reported separately.

## 5. Discrepancy (`qemx/discrepancy.py`)

**Distances between probability distributions** (validated before use):

- TVD;
- Hellinger distance;
- the classical fidelity F_c = (Σ√pq)² of Chen et al., Eq. 1, without uniform normalization;
- Lubinski et al.'s normalized fidelity.

**Signed quasi-probabilities.** A signed vector (readout-inverted or ZNE-extrapolated) raises `QuasiProbabilityError`. The app then shows "ℓ1 … (not a TVD)" and clips nothing. `project_to_simplex` is explicit and reports the mass it moved.

**Finite shots.** `distance_test` compares the observed plug-in distance with its sampling distribution under the model at the same shots. A nonzero distance alone is not a discrepancy.

**Many tests.** `adjust_pvalues` provides Holm, Hochberg, Benjamini–Hochberg and Bonferroni. Divergence tables flag with Holm.

**Conventions.** `convert_error_rate` converts between the depolarizing p, the entanglement infidelity ε and the average gate infidelity r: ε = p(1 − 1/d²) and r = p(d − 1)/d. An RB error estimates r, not p.

**Rehearsal.** `scripts/rehearse_divergence.py` runs the hardware path (plan → counts → analysis) on emulated truths. It measures the false-positive rate when the model is true, and the power against controlled alternatives. Every output is labeled EMULATION.

## 6. Features and the main study

- **Feature dictionary:** `docs/feature_dictionary.md`, from `qemx/features.py`, version `qemx.features/1`. It separates:
  - logical features (device-independent);
  - compiled features (device- and layout-specific);
  - observable features;
  - exact state features (O(2^n), not scalable);
  - noise features (light-cone fault rates);
  - oracle features, which are never recommendation inputs.

  Two-qubit count is not entanglement. The classical output entropy is labeled "not an entanglement measure".
- **Study design (manifests):**
  - `design: {kind: maximin, per_stratum: k}` chooses k space-filling cells per (family, n) instead of the full grid.
  - `confirmatory: {fraction, seed}` fixes a confirmatory subset by base circuit before any outcome exists.
- **Analysis:** `python -m qemx.study report RUN_DIR --manifest M --out DIR` regenerates every figure and table from stored records, without re-simulating. It produces:
  - empirical performance-regime maps, with ties, "raw ≈ best" and "no feasible method"; there is no interpolation;
  - paired changes against raw, with base-circuit bootstrap intervals;
  - cost to reach ε;
  - the recommender evaluation;
  - the discrepancy rehearsal.
- **Recommender** (`qemx/recommend.py`): ridge models of log RMSE per method, fitted inside training folds. It abstains outside its coverage and says **use raw** when nothing is predicted to help by more than 10 %.
  - Its evaluation uses grouped folds by base circuit, leave-family-out, leave-width-out and the confirmatory subset, against always-raw, best-fixed and rule baselines.
  - Feature weights are associations, not causes.
- **Upper-bound methods.** Methods flagged `optimistic` (virtual distillation simulated with ideal copies, an ideal derangement and ideal readout) are shown as upper bounds and are never candidates.

## 7. Hardware

`docs/hardware_preflight.md` gives the shared 600-second plan, the usage model and ledger, the stages, the commands and the stopping rules. No hardware result exists in this release.

## 8. Performance and concurrency

`scripts/profile_workflows.py` profiles representative workflows in fresh processes and works on any checkout. Results for 1.4.2 and 1.5 are in `docs/TRACEABILITY_1.5.md`.

Concurrency:

- Cached results are shared between sessions only as immutable data, keyed by content.
- Two sessions with different settings keep their own results (`tests/test_concurrency.py`).
- Concurrent engine calls give exactly the serial results.

Hard per-call memory and time limits for native code are not provided. Public hosting still needs isolated workers.

## 9. Known limitations

- **One device model.** All study numbers are simulations on one device model (fake_fez) or on the injector's noise. Transfer to hardware is untested until the approved campaign runs.
- **Small pilot.** The pilot study has 25 base circuits, and its recommender evaluation is correspondingly uncertain. On grouped folds, the baseline recommender does not beat the best fixed method.
- **Sources not fully read.** NIL's sample-complexity appendix, Cai et al.'s Sec. IV and Proctor et al.'s GP details could not be read (`docs/literature_matrix.md`). Nothing in the app depends on them.
- **No browser timings.** Browser QA covered rendering, overflow and keyboard focus at 1440 px and 390 px (`docs/TRACEABILITY_1.5.md`). It did not measure human-perceived latency, and it did not cover touch interaction.
