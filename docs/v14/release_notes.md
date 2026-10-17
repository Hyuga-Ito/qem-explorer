# QEM Explorer 1.4 — release notes

1.4 implements neighbor-informed learning (Chen et al., arXiv:2512.12578) as its own estimator. It adds a composable
pipeline that chains mitigation methods, encoded-algorithm templates (code × benchmark), and analysis scripts with
input and output kept separate from the computations. It also fixes every confirmed finding of a line-by-line audit
of the methods, the Direction A pipeline and the app. Every number below was checked by a test. The number of tests
is in the last section.

## 1. Neighbor-informed learning (`qemx/methods/nil.py`, estimator `nil`)

The feature-regression method of 1.3 (`learning`, `features="nil"`) was not Chen et al.'s protocol. It is kept,
correctly named, as **FR** (see §2). `nil` follows the paper:

* **Neighbor circuits:** the target with one Pauli after one gate (U_j → PU_j, weight 1), s drawn at random from all
  insertion points (after two-qubit gates, or after every gate). The draw is nested, so the first s of a larger draw
  are the smaller draw. ZNE neighbors (λ = 1, 3, 5, global folding) are also available.
* **Training circuits:** every non-Clifford rotation gets an angle drawn uniformly from {0, π/2, π, 3π/2} (Algorithm 2,
  a rotational 2-design). They carry the same insertions as the target.
* **Regression:**
  * Chen et al.'s ℓ1-constrained least squares, ‖c‖₁ ≤ γ, with no intercept (γ = 2 for Pauli neighbors, 5 for ZNE
    neighbors).
  * Also: Lasso, Ridge, OLS, elastic net and NNLS, each with or without an intercept.
* **Solvers are certified, not assumed:**
  * OLS (minimum norm) and Ridge are solved exactly.
  * Lasso and elastic net: a primal active-set method.
  * ℓ1-constrained: a safeguarded Newton iteration on the multiplier over exact Lasso points, accepted only at a face
    that satisfies the KKT conditions to rounding.
  * A duality gap or Frank–Wolfe gap bounds the excess objective by 10⁻¹⁰‖y‖².
  * On 400 stored problems per rule (2 400 fits: collinear, rank-deficient and wide designs), every fit is certified,
    and its objective exceeds CVXPY's by at most 1.4·10⁻¹²‖y‖².
* **Stabilization:** the rules with an ℓ1 term or a sign constraint are solved with ε‖c‖² added,
  ε = (10⁻⁶‖X‖₂)². The problem is then strictly convex, so the solution is unique and does not depend on column order.
  The cost is at most ε‖c*‖² in objective; the tests bound it.
* **Duplicate neighbors:** neighbor circuits that coincide on every training circuit (a Pauli that commutes to the
  measurement) are merged and share their weight equally under every rule. This is the point an interior-point solver
  converges to.
* **Cost:**
  * Training uses T × k circuits.
  * Target shots are split N_j ∝ |c_j|σ_j, so v = (Σ|c_j|σ_j)² and circuits with c_j = 0 are not run. Alternatively
    the split is equal, N/k each, as in Chen et al.'s experiments.
  * Only the equal split can be planned on IBM hardware, because a plan is sent as one batch. The optimal split is
    refused there with a reason.
* **Monte-Carlo accounting:**
  * The training-shot variance u comes from refits on resampled training data. The first-order (delta-method) value
    is meaningless for collinear features.
  * The calibration coefficient w comes from paired refits with and without a sampled calibration. Training and
    target data are mitigated with the same θ̂ ("joint calibration").
  * Samplers accept per-replicate calibration draws. A wrong number of draws is refused, not truncated.
* **App:**
  * A fit plot (training circuits and target) and a coefficient chart.
  * Every parameter carries help text.
  * Short titles such as "NIL (ℓ1≤2)" and "NIL (Ridge α=0.001, T=32)".
  * New manifest `manifests/direction_A_nil_v1.yaml` (1 280 cells). Stages A1, A2 and B reuse the cells of run 2.
    Stage E-nil-sensitivity covers γ, α, T, s, the intercept, the bare circuit, the split and the training ensemble.

## 2. Literature consistency

* **FR is not Strikis et al.** The 1.3 docstring, README and handbook cited Strikis et al. (PRX Quantum 2, 040330
  (2021)) for feature regression on ⟨O⟩, ⟨Z_i⟩, ⟨Z_iZ_j⟩. Strikis et al. learn combinations of *Pauli-inserted
  variants* of the circuit, and that is what NIL scales up. FR is now described as this project's heuristic. The
  closest published work is ML-QEM (Liao et al., Nat. Mach. Intell. 6, 1478 (2024)). Method keys are FR-L and FR-R;
  stores of runs 1 and 2 keep NIL-L and NIL-R and are renamed on loading.
* **Lasso scale:** sklearn minimizes (1/2N)Σ(…)² + α‖c‖₁. FR now passes α/(2N), so α means what the formula says.
* **Handbook:** §6.2 (FR) and §6.3 (NIL) were rewritten with Chen et al.'s protocol and parameters. vnCDR (Lowe et al.)
  and UNITED (Bultrini et al.) are placed as the published combinations of ZNE, CDR and VD. References were added for
  Chen 2025, Liao 2024 and Bultrini 2023.

## 3. Composable pipeline (`qemx/methods/compose.py`, estimator `chain`)

An outer method that combines circuits (ZNE, CDR or NIL) runs each of its circuits through an inner per-circuit
mitigation (raw, symmetry verification, VD or PEC). The inner method's result is the interface: mean, per-shot
variance, cost fields, sampler and calibration function. The accounting therefore composes exactly:

* ZNE's split uses the inner per-shot σ_j.
* Training shots are spent on the inner method.
* Post-selection needs Σ_j ⌈5/P_j⌉ target shots.
* u is measured by refits with the inner samplers.
* w comes from Monte Carlo at exact features.

Checks:

* `chain(zne, raw)` reproduces `zne` exactly: mean, variance, calibration gradient and the sampler's draws.
* `chain(nil, raw)` reproduces `nil` exactly: mean, variance, coefficients and training-shot variance.
* With depolarizing noise and PEC told the true rates, `chain(zne, pec)` gives a flat E(λ) at the ideal value.

The chain's parameters are copied from its components, prefixed and shown only when relevant (`Param.when`), so the UI,
the state validation and the manifest checks treat them like any method's. The Direction A pipeline supports chains,
including ZNE's integer split and training levels. Chains run in simulation only.

## 4. Encoded algorithms (`qemx/qec.py`, template `qec_algorithm`)

There are four codes and nine benchmarks; every pair up to 12 qubits builds:

* Codes: bit-flip, phase-flip, [[4,2,2]] and [[5,1,3]].
* Benchmarks: state preparation, Bell, GHZ-3, Deutsch–Jozsa, Bernstein–Vazirani, RX, RZ Ramsey, Grover-2 and QFT-2.

How it works:

* Every logical gate is labeled fault-tolerant or not, with the reason.
* The decoder is the observable, and error detection is symmetry verification with every check.
* The rotation benchmarks are non-Clifford, so CDR and NIL can train on them.
* Eight presets (for example "Grover-2 on [[4,2,2]]: every logical gate transversal") sit in a selectbox on the
  Playground's template card.
* The QEC view shows p_L against the intended logical value.

## 5. Direction A pipeline (`qemx/phase`)

* **Ledger:**
  * Integer allocations are now exact. The variance is discretely convex in the calibration shots, so a unit-step
    descent from floor/ceil of s* reaches the optimum. Rounding alone did not: with several target circuits it missed
    by up to 5 %, and RMSE(B) could rise with B.
  * The first-order opt path searches the integer s_tr with the exact best s for each.
  * Brute force over 12 000 random allocations (fixed and opt, one to three target circuits) finds none
    suboptimal, and RMSE(B) never increases in the audit's counterexamples.
  * `Cost` rejects NaN or negative inputs, which would otherwise yield silent wrong budgets.
  * Training levels carry their own v and w.
  * `crossovers` finds changes of winner that pass through exact ties.
* **Training levels:**
  * A failed or non-finite level is recorded with `ok: false` and left out of the ledger, instead of removing the
    method from the cell.
  * The calibration coefficient w is measured at every level, on the same refits as u.
  * Rules on the same circuits share their random streams (common random numbers).
* **Failures:**
  * A time limit, out-of-memory error or interrupt inside a method or a check aborts the cell, which is then recorded
    as `timeout` or `oom`. Before, it was swallowed and the cell was recorded "ok" with part of its data missing.
  * The cooperative time limit is now checked in the refit and validation loops.
  * A cell with any failed method or check is "partial", and `--retry-errors` re-runs it.
* **Validation:**
  * The learning methods' calibration is simulated jointly inside the sampler.
  * A nonzero difference with zero standard error gives z = ±∞. Before, it gave 0, which passed.
  * The ledger's standard error includes w's.
  * Re-runs at other training shots are cached.
  * Bootstrap standard errors are recorded beside the pre-registered plug-in ones as a diagnostic for heavy tails.
* **Records and tables:**
  * The record schema is `qemx.phase.cell/2`; /1 stays readable. Records are checked against the schema when written.
  * `collect` uses only the cells of the run's manifest. Stale records from an earlier manifest are reported, not
    double-counted.
  * Every table is rewritten: a stale table, or a stale parquet beside a fresh CSV, can no longer survive.
  * One bad record becomes an error row and never aborts the collect.
  * The budget grid comes from the manifest, and the policies use each cell's own s_min.
  * The errors table also lists failed levels and checks.
* **Runner:**
  * An output directory is bound to one manifest by SHA-256 (`--force` rebinds it).
  * Every invocation is logged in `run_history.jsonl`.
  * Temporary files of interrupted writes are removed.
* **Manifest checks:** duplicate method keys, stage-level training settings, a validate block without a fraction and
  the budget grid are now checked.
* **Smaller fixes:**
  * λ_LC's gate count no longer counts DD pulses.
  * `plots.winner_map` filters the candidate set and names every method.
  * `fit.py` and the M3 calibration resampling are documented.

## 6. Analysis scripts: input and output separated

* `scripts/dirA_io.py` provides the input and output objects:
  * Sources: a run directory or memory.
  * A `Report` that holds sections, verdicts, text and tables, rendered as text or Markdown.
  * Report and figure sinks for a directory or memory.
* `dirA_checks.run_checks(R)`, `dirA_figures.make_figures(T, sink, rep)` and `dirA_nil.analyse(T)` compute without
  opening a file. The `main()` functions only wire source → computation → sink.
* The checks give the same report from disk and from memory, and leave the run directory untouched.
* Winners restricted to a figure's methods are computed vectorized, and the result equals qemx's rules.
* `family_map` breaks exact ties by method order, not by data order.
* Cross-stage matching uses the full cell coordinates and reports ambiguous matches.
* Circuits are grouped by their parameters, not their labels.
* `B*(ε)` uses qemx's own cost reconstruction.
* New: `scripts/dirA_nil.py`. It compares the NIL rules on the same circuits (|b|, v, u, γ, certificates), runs NIL-G2
  head to head against FR, CDR, ZNE and PEC per budget, and computes B*(ε) ratios.

## 7. App

* Fixes 1–16 of the app audit:
  * Time-travel cache, large registers and atomic model updates.
  * IBM submission and collection.
  * Recipe identity and the hardware plan keys.
  * Rerun performance: 4–7× faster for circuits of 10⁵–2·10⁵ operations.
  * Strict parameter validation.
  * Guarded computations.
  * Results tied to the state they were run with.
* Method parameters can depend on other parameters (`Param.when`).
* A method decides whether it runs on hardware (`hardware_supported`).
* Engine speed-ups at n = 10 (results identical to 10⁻¹²):
  * Kronecker mitigation matrices: 0.93 s → 0.008 s.
  * Cached per-shot values: 0.21 s → 0.0005 s.
  * Vectorized M3: 3.3 s → 0.06 s.
* The IBM token is unchanged from 1.3: a password field, never stored in the state, never exported.

## 8. Compatibility

* Runs 1 and 2 (`direction_A_v1`) load unchanged. Their FR keys are renamed on loading, and their tables gain no new
  columns until they are collected again.
* `direction_A_nil_v1` reproduces run 2's reference methods **statistically, not bit for bit**:
  * The exact b and v are the same.
  * u and w come from Monte-Carlo refits at every training level.
  * The learning fallback is decided per level.
  * The random streams are keyed differently.
* New table columns:
  * methods: `s_min`, `s_tr_min`, `x_model`, `x_solver_ok`, …
  * levels: `ok`, `w`, `se_w`, …
  * validation: `z_rmse_boot`, `calibration_mc`.

## 9. End-to-end check with the production settings

One validated cell of `direction_A_nil_v1` was run as the pipeline runs it: QFT-4 at noise scale 0.25 on
fake_fez, 15 methods, 2 000 Monte-Carlo refits at the protocol's training level and 500 at each of the others, and
validation at six budgets with 2 000 experiments each, under both policies.

* Every method row is "ok", every training level is usable, and every NIL fit is certified.
* All 102 independent Monte-Carlo checks pass. The z-scores have mean −0.00 and sd 0.84 (RMSE), and mean 0.06 and
  sd 0.95 (bias). KS p = 0.70 and 0.24.
* It took 10.4 minutes on one core.
* NIL-OLS, now exact, shows what ordinary least squares does with near-collinear neighbors: v ≈ 6·10⁸. It never
  wins. The stabilized rules and Ridge stay at v ≈ 0.03–0.06.

## 10. Tests

The full suite has 678 tests (32 of them marked slow), all passing. New in 1.4:

* `test_nil.py`: solvers against brute force and certificates, twins, order invariance, joint calibration, Monte-Carlo
  w against first order, and pipeline cells with validation.
* `test_compose.py`: exact reproductions, PEC and VD inners, a pipeline cell, and the Playground running and building
  chains (AppTest).
* `test_pipeline_v14.py`: exact allocations, failures, collect, manifests, schema, sinks.
* `test_scripts_io.py`: a real run through every script, disk against memory.
* `test_qec_templates.py`: 120 tests, including mutation checks and the presets.
* `test_audit_methods.py` and `test_audit_app.py`.
