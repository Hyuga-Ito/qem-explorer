# QEM Explorer 1.4.1 — release notes

1.4.1 adds two studies to the Direction A pipeline:

* a head-to-head comparison of **global and local folding** for zero-noise extrapolation at equal total budget,
  including **fractional ZNE** (λ = 1, 4/3, 5/3, 2) with chosen and averaged fold subsets (§6);
* a **hardware against fake-device divergence** analysis: the same application circuits on an IBM device and
  predicted from the device's own component-level calibration, drawn as a difference volumetric plot — and, in its
  second version, the diagnostics that say *why* they differ: χ² ladder, divergence budget, folding fingerprint,
  spectator probes, layout variants and a heavy-hex map, validated on planted errors (§7; research plan
  `docs/v14/research_plan_divergence.md`).

Both use the I/O-decoupled analysis scripts of 1.4: computations take tables or objects and return tables, reports
and figures; sources and sinks do the reading and writing. Every number below comes from a run of the shipped
manifests in this sandbox (fake_fez, Qiskit 2.5.2, Aer 0.17.2, qiskit-ibm-runtime 0.50.0). No real device was
reachable from here: the hardware path was exercised against a mocked service and in Qiskit's local mode, and the
divergence figures below are from an **emulated** device with a known injected error, labeled as such everywhere.

## 1. ZNE folding: what changed in the methods

**Effective noise scale.** ZNE now records, for every folded circuit:

* the nominal scale λ (two-qubit gates of the folded circuit ÷ the original's);
* the effective scale λ_eff = λ_LC(folded) / λ_LC(original): the factor by which folding raises the light-cone fault
  rate of the noise model (`qemx.phase.lightcone`);
* the depth and the two-qubit count of the folded circuit.

A new parameter `lam_est` chooses the λ of the fit: `gates` (the nominal λ, the default and the behavior of 1.4.0)
or `fault-rate` (λ_eff). The fault-rate fit needs a noise model: it works in simulation (fake devices and NoiseSpec
models) and raises a clear error otherwise. In simulation the model is the one that generates the data, so λ_eff is
exact there — an optimistic reference, not a hardware-ready estimator.

**Local folding as a pipeline component.** Local folding already ran as a ZNE method (`folding: local`). 1.4.1 adds:

* as an **outer** method: `chain` with `outer: zne, zne_folding: local` over any inner method (raw, PEC, symmetry, VD);
  the chain's ZNE now also takes `zne_lam_est` (it raised a TypeError on the new parameter before this fix);
* as a **feature generator**: NIL with `neighbors: fold, folding: local` (also `nil_folding` inside a chain). The
  subset folded once more for a fractional λ depends only on the seed and the gate positions, so the target and every
  training circuit fold the same gates. The regression learns its coefficients, so it needs no value of λ.

Checked: `chain(zne local, raw)` and `chain(nil local fold, raw)` equal the plain methods to 1e-12; the cell runner's
fold guard also covers local fold neighbors.

**Families.** `grover` and `bv` (1.4) are joined by `ghz` with `basis: Z` (peaked output distribution, for the
divergence suite). The X-basis GHZ cells keep their parameters, so their cell ids do not change.

## 2. The folding study (`manifests/direction_A_zne_folding_v1.yaml`, `scripts/dirA_zne_folding.py`)

**Design.** 744 cells in stages F1–F4 (the fractional stages F5, F6 add 170: §6):

* 28 circuits: Grover (3–4 qubits, k = 1–3), QFT and AQFT, random brickwork (4 and 6 qubits, 2–16 layers, two
  seeds), BV and GHZ; 3 to 189 two-qubit gates after compilation to a fixed chain of fake_fez;
* noise scales s = 0.25 … 4;
* stages:
  * F1: readout on;
  * F2: readout off;
  * F3: homogenized device with a coherent fraction χ = 0, 0.25, 0.5 of every gate's error;
  * F4: folding inside composed methods.
* methods:
  * ZNE-G / ZNE-L: Richardson on λ = 1, 3, 5;
  * ZNE-Gf / ZNE-Lf: the same fits against λ_eff;
  * ZNE-G-lin / ZNE-L-lin: linear on 1, 3;
  * F4 adds NIL with global or local fold neighbors, and ZNE (global or local) over PEC.

Every pair is compared at equal total budget by the exact ledger (integer allocations, readout calibration included).

**The question's "λ = 1, ?.??, ?.??".** With the same nominal scales 1, 3, 5 (the same two-qubit gate count), the
effective scales on fake_fez are (median over 196 cells of F1):

| folding | λ_eff | range at λ = 5 |
|---|---|---|
| global | 1, 3.26, 5.53 | 5.03 – 10.93 |
| local | **1, 2.62, 4.23** | 4.02 – 4.79 |

Local folding repeats only the two-qubit gates. One-qubit gates and most idle time stay unfolded, so its noise grows
more slowly than its gate count.

**Local folding does not break down because of depth.** At equal two-qubit gate count (checked in every circuit),
local folding's most folded circuit is 0.31–0.52 × as deep as global folding's. It breaks down because it
**under-amplifies** the noise: Richardson extrapolation in the nominal λ lands at the residual fraction
ρ = Σ η_j λ_eff,j ≈ 0.19 of the unmitigated circuit's noise. At low noise the measured bias of ZNE-L is
0.21 × b_RAW (median; first-order prediction 0.19, median absolute error 0.05).

For global folding λ_LC is not a good proxy. The folded circuit's light cone is wider than the original's, so λ_LC
counts faults that rarely reach a local observable (λ_eff up to 10.9 on shallow brickwork). The nominal λ is nearly
unbiased at first order (measured b/b_RAW ≈ −0.01). Fitting global folding against λ_eff (ZNE-Gf) makes it *worse*
(0.19 × b_RAW), so `lam_est: fault-rate` is meant for local folding.

**Where local folding breaks down.** ZNE-L against ZNE-G, stage F1 (196 cells):

| outcome | share of cells |
|---|---|
| global better at every budget (1e3 … 1e8 shots) | 47 % |
| local better at small budgets, then breaks down | 43 % (median B_break = 5.1·10³ shots) |
| local not worse at the largest budget | 10 % (6 % better at every budget) |

The local method's folded circuits are less noisy, so its variance is smaller. Its bias is larger. They cross at
N× = (v_G − v_L)/(b_L² − b_G²), and to first order N× ∝ λ_LC^−1. Measured:

* B_break falls with λ_LC: Spearman −0.70, against −0.49 for the noise scale alone and −0.20 for the two-qubit count;
* the first-order N× ranks B_break almost perfectly: Spearman 0.97 in F1, 1.00 in F2;
* fitted slope of log B_break against log λ_LC: −0.74 in F1, −1.03 in F2 (readout off).

So local folding breaks down earlier as noise × depth grows (λ_LC combines both). Global folding wins at every
budget when the observable's ideal value is near 0, because then the per-shot variance does not depend on the noise.

**Fault-rate calibration.** ZNE-Lf removes most of local folding's bias: 0.06 × b_RAW at low noise, against 0.21.
It pays in variance, because λ_eff = 1, 2.62, 4.23 are closer together, so the Richardson weights are larger. It is
better than ZNE-G at the largest budget in 38 % of the F1 cells (at every budget in 10 %), and worse at every budget
in 62 %.

**Coherent errors change the picture (F3).** Coherent errors add in amplitude when the same gate is repeated, so local
folding of self-inverse gates (CZ·CZ·CZ) amplifies them faster than their infidelity suggests.

| χ | local not worse than global at the largest budget | median b/b_RAW of ZNE-L | median b/b_RAW of ZNE-Lf |
|---|---|---|---|
| 0 | 7 % | 0.24 | 0.08 |
| 0.25 | 63 % | 0.04 | −0.15 |
| 0.5 | 54 % | −0.15 | −0.41 |

λ_LC is blind to coherence, so the fault-rate calibration over-corrects as χ grows. On hardware with coherent
errors, local folding is a serious competitor, and its λ calibration must come from data, not from error rates.

**Folding inside composed methods (F4).**

* NIL learns its own λ: global and local fold neighbors tie (median RMSE ratio 1.00 at the largest budget).
* ZNE over PEC: local folding is better at every budget in 8 of 16 cells, and breaks down in the other 8.

**Outputs of the script.**

* Tables:
  * `folding`, `lambda_summary`, `depth`;
  * `head_to_head`: exact crossovers, regime, B_break, N×;
  * `breakdown` and `breakdown_coherent`;
  * `bias_check` and `bias_coherent`.
* Figures:
  * λ_eff against the nominal λ, and ρ against the circuit size;
  * the depth comparison;
  * crossover maps: one panel per circuit ordered by two-qubit count, budget × noise scale, color = log₂ of the RMSE
    ratio, black ticks = B_break;
  * B_break against λ_LC, with the fit and the first-order slope;
  * the breakdown map: two-qubit gates × noise scale, color = B_break;
  * measured against predicted bias.

## 3. The divergence study (`qemx/divergence.py`, `scripts/dirA_divergence.py`, `manifests/direction_A_divergence_v1.yaml`)

**Suite.** 59 application circuits, 2–12 qubits and up to 144 two-qubit gates after compilation:

* GHZ (Z basis);
* Bernstein–Vazirani;
* QFT;
* Grover;
* random brickwork;
* Ising Trotter circuits.

Each is compiled once for the device on the best chain of its current calibration. Routing is confined to the chain,
and every as-executed circuit is identified by a SHA-256 digest. Each circuit runs under four methods (RAW, tensored
readout mitigation, ZNE global, ZNE local), each with its own 6 000 shots, calibration included. Circuits are never
shared between methods.

**Predictions** come from the calibration on each circuit's qubits, taken at execution time when the service keeps
the history, else at submission. Three models are simulated on the as-executed circuits:

* **depolarizing** (primary): the "fake device from component-level metrics". Every gate is followed by the
  depolarizing channel whose average gate infidelity is the reported error, plus each qubit's asymmetric readout error.
  There is no relaxation and no idle error. Checked: the installed channel's infidelity equals the reported error to
  1e-6.
* **chen**: Chen et al.'s model as published. One median rate for all one-qubit gates and one for all two-qubit
  gates, no readout. Their ε (entanglement infidelity) and IBM's r (average infidelity) describe the same channel,
  with r = ε·d/(d+1).
* **device**: the full fake-device model (relaxation over gates and idle windows, readout).

**Comparison at the same shots.**

* Fc = (Σ√(p_i q_i))² and Lubinski et al.'s normalized fidelity.
* Each prediction is evaluated as the observation is: 400 Monte-Carlo replicates of the method from the predicted
  probabilities, at the same shots per circuit, through the same estimator (readout inversion, ZNE on the
  distribution, projection onto the simplex).
* Δ = mean predicted Fc − observed Fc carries the same finite-shot bias on both sides.
* z = Δ / pooled SD, √((replicate SD² + bootstrap SE²)/2): two estimates of the same shot noise when the model is
  right; |z| > 3 is a divergence that the shot noise of the budget does not explain. (The first version divided by
  the replicate SD alone, which explodes for estimators the model predicts to be nearly deterministic — a plurality
  vote on a clean peak.)
* Observations carry bootstrap standard errors.

**The difference volumetric plot** (`dirA_figures.difference_volumetric`, `_grid`; `predicted_vs_observed`):

* axes: circuit width × post-compilation two-qubit gates in log₂ bins (Chen et al. Fig. 11). Depth or Lubinski's
  normalized depth are alternative x axes;
* tiles: the mean Δ of the tile;
* markers: one per circuit (shape = family, fill = its own Δ), heavy outline when |z| > 3;
* the most divergent circuits are numbered on the plot and named above it;
* color: one diverging blue–gray–red scale for all methods; red means the device is worse than its model.

**Null test.** When the emulated device is exactly the depolarizing model (the whole 59-circuit suite, all four
methods), Δ is pure shot noise:

* the 236 z-scores (pooled SD, recomputed for this release) have mean −0.06 and SD 0.90, consistent with N(0, 1)
  (Kolmogorov–Smirnov p = 0.13); with the first version's replicate-SD z they had mean −0.03, SD 0.96, p = 0.41;
* none has |z| > 3 (largest 2.36); the χ² ladder reads "as modeled".

So a significant Δ on hardware is not an artefact of the estimators, the projection or the finite budget. Against
the same data, the device model (which adds relaxation) predicts lower fidelities (Δ < 0 in 12–20 % of the RAW and
RO circuits). Chen et al.'s model, without readout, over-predicts the raw fidelity (z > 3 in 37 %).

**Demonstration with a known unmodeled error (emulated, not hardware).** The truth model is fake_fez's full model
plus a ZZ crosstalk rotation of 0.05 rad on every spectator pair after every CZ. The analysis isolates it:

* the effective two-qubit error fitted to Fc is 4.2·10⁻³ observed against 1.5·10⁻³ predicted, 2.8 × (the same kind of
  gap as Chen et al.'s 100 against 46.4·10⁻⁴);
* the divergence concentrates in the structures with many CZ gates next to spectators. RAW, depolarizing model, mean
  Δ by family:

  | family | mean Δ |
  |---|---|
  | QFT | 0.11 |
  | Ising | 0.09 |
  | BV | 0.04 |
  | Grover | 0.03 |
  | brickwork | 0.015 |
  | GHZ | 0.008 |

* Δ grows with the two-qubit count within every family (Spearman 0.89–1.0);
* the largest raw divergences are Ising-10 (r = 8) Δ = 0.31 (z = 89), Ising-8 (r = 8) Δ = 0.26 and QFT-8 Δ = 0.25;
* ZNE magnifies the unmodeled error: folding repeats the crosstalk too, up to Δ = 0.54 on Ising-8 (r = 8). Yet only
  14–17 % of ZNE's circuits are significant at this budget (against 78 % raw, pooled z): its variance hides the
  smaller divergences;
* the device model explains part of the gap (idle relaxation); the rest is the injected crosstalk;
* read with the v2 diagnostics (§7; features refreshed on this 1.4.0-era plan): χ² ladder "context + non-linear"
  (χ²/dof 149 → 90 with a rescaled calibration → 38 with context), folding fingerprint "repetition-coherent"
  (A_global 2.01 ± 0.04, A_local 2.65 ± 0.04) — a coherent error that grows when the same gate is repeated, which
  is what a ZZ rotation after every CZ is.

**Hardware path and credentials.**

* `run-ibm` submits all circuits in a Batch (job mode when the plan does not allow batches), at most 300 circuits per
  job, after an ISA and fold check of every circuit. `jobs.json` holds job ids only.
* If submission stops part-way, the submitted job ids are kept (`partial`). `run-ibm --resume` submits only the runs
  they do not cover, and nothing is ever resubmitted silently. `--force` submits everything again and keeps the old
  ids in `jobs.replaced-*.json`.
* `collect-ibm` retrieves counts and the calibration at execution time. A job that ends in error loses only its own
  runs.
* The token is read from `$QISKIT_IBM_TOKEN`, a saved account or a hidden `getpass` prompt. It is never a
  command-line argument, never written to a file and never printed, and it is masked in connection errors. A test
  with a mocked service scans every file written and all output for it.
* `run-local` runs the same ISA circuits through `SamplerV2` in local mode on the fake twin. That mode applies gate
  and readout errors but no idle relaxation, so the depolarizing model, not the device model, is its closest
  predictor.
* No hard limits on width or depth:
  * predictions are exact (NumPy) up to `exact_max_qubits`, use the Aer density matrix up to `aer_dm_max_qubits`, and
    Aer trajectories above;
  * every instance runs inside try/except, and out-of-memory errors are recorded per instance;
  * `--workers` runs instances in spawned processes. When the workers cannot start, the work falls back to serial.
    A worker killed during the run (e.g. out of memory) is reported for its unfinished instances, which are never
    re-run in the main process.

**Cost.** The default suite is 826 circuits and 1.42 M shots; `plan` prints the totals. On two cores the
predictions of the three models took 12 min and the emulation 16 min.

## 4. Fixes found on the way

* `chain(outer=zne)` passed every ZNE parameter to its outer ZNE, which did not accept `lam_est`: every chained ZNE
  would have failed after the new parameter. Fixed, and tested for equality with the plain method.
* `scripts/dirA_checks.py` labels: a GHZ label read a variable from the wrong scope (introduced and fixed in this
  release before shipping).
* The analysis scripts now find `qemx` in a source checkout (`dirA_io._find_qemx`, appended to `sys.path`; an
  installed qemx wins). Before, a script run from the repository root silently fell back to grid interpolation of
  B*(ε) instead of the exact ledger.
* Fixed after an independent code review of this release (each one now has a test):
  * resuming a partial submission;
  * keeping the counts of finished jobs when another job fails;
  * a worker that dies during the run is never re-run in the main process;
  * significance outlines of the normalized-fidelity plots use z of the normalized fidelity;
  * the first-order crossover follows B_break's convention when local folding has the smaller bias;
  * duplicate ZNE scales are rejected when the manifest is read, instead of every circuit being dropped at planning
    time.

Figures and tables of the runs above are in `docs/v14/results_1.4.1/`:

* `folding/`;
* `divergence_emulated_crosstalk/`;
* `divergence_emulated_null/`.

## 5. Tests

* `tests/test_zne_folding.py`:
  * λ_eff below nominal for local folding and above for global;
  * equal two-qubit counts and local circuits never deeper;
  * the fault-rate fit and its zero residual;
  * chain and NIL equalities;
  * fold neighbors in training circuits;
  * the families;
  * the manifest (914 cells);
  * the script on a pipeline run, with B_break checked against the exact ledger one shot before and after, the grid
    fallback, figures and main.
* `tests/test_divergence.py`:
  * metrics against brute force (simplex projection against SLSQP, readout inversion);
  * suite validation;
  * chains are connected paths;
  * ideal distributions against the density-matrix engine;
  * equal budgets, digests and ISA circuits;
  * predictor channels;
  * exact against Aer;
  * the null test and the crosstalk injection test;
  * worker independence and the serial fallback;
  * Sampler local mode;
  * the figures;
  * the command line end to end;
  * token hygiene with a mocked IBM service.

## 6. Fractional ZNE (`qemx.transforms.fold_order`, `qemx.methods.zne`; stages F5, F6)

**Methods.** A local fold at λ = 1 + 2(k + m/n) repeats every two-qubit gate k times and m of the n gates once more.

* `subset` chooses the m gates: `random` (a seeded subset per λ, the behavior of 1.4.0), or a *nested* order —
  the gates folded at a smaller λ are folded at every larger one, so the noise grows monotonically along the scales:
  * `nested`: a seeded random permutation;
  * `stratified`: the bit-reversal (van der Corput) order of the gate positions — every prefix spread evenly over
    the circuit (half the gates = every other gate), deterministic;
  * `balanced`: greedy, every prefix with the circuit's mean calibrated two-qubit error (λ in gates ≈ λ in noise;
    stratified without a device model).
* `fold_draws: K` averages K independent nested subsets at every fractional λ. The shots are split over the K
  circuits, so the per-shot variance does not change; the estimator stays linear (Σ_jk η_j/K · E_jk) and the ledger
  receives one weight per circuit. Not available inside a chain (one fold per scale there).
* `rule: poly2` — the least-squares parabola (Σ|η| = 19 on λ = 1, 4/3, 5/3, 2, against 111 for Richardson through
  the same points and 3.6 for the line).
* The divergence plan accepts the same folds (`subset` in a ZNE method; run roles `fold:local.stratified:1.333`), and
  skips — with a recorded reason — circuits with too few two-qubit gates for distinct fractional scales.
* The hardware page's distribution view and the exported script fold the same subsets the method ran (both folded
  the 1.4.0 random subset before, and the exported script took λ from a *global* fold for local folding).

**Study.** F5: 22 circuits with ≥ 6 two-qubit gates × 7 noise scales (154 cells): ZNE-G (Richardson on 1, 3, 5,
the reference), global partial folding with poly2 (ZNE-Gq), local stratified folding with poly2, exponential, linear
and Richardson fits. F6: 4 circuits × 4 scales: the subsets, and averages over 4 and 16 subsets. The manifest now
has 914 cells.

| method (λ = 1, 4/3, 5/3, 2) | Σ\|η\| | median \|b\|/\|b_RAW\| | RMSE ratio to ZNE-G at 10⁴ / 10⁶ shots | cells better than ZNE-G at 10⁶ |
|---|---|---|---|---|
| ZNE-G (1, 3, 5) | 3.8 | 0.07 | 1 / 1 | — |
| ZNE-Gq, global, poly2 | 20.6 | 0.36 | 5.1 / 5.3 | 15 % |
| ZNE-Lq, local stratified, poly2 | 20.6 | 0.32 | 5.1 / 5.1 | 16 % |
| ZNE-Lq-exp, exponential | 4.5 | 0.23 | 1.1 / 1.3 | 21 % |
| ZNE-Lq-lin, linear | 3.9 | 0.32 | 1.07 / 1.74 | 4 % |
| ZNE-Lq-R, Richardson | 120 | 0.71 | 29 / 27 | 12 % |

On average fractional scales lose: λ ≤ 2 is a long lever arm to λ = 0, local folding reaches only λ_eff = 1, 1.27,
1.56, 1.82, and the fits that bend multiply the shot noise. **They win where integer folding fails** — share of
cells ZNE-Lq wins at the largest budget, by light-cone fault count:

| λ_LC | ≤ 0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | 0.4–0.8 | > 0.8 |
|---|---|---|---|---|---|---|
| ZNE-Lq | 3 % | 7 % | 9 % | 14 % | 42 % | 94 % |
| ZNE-Lq-exp | 3 % | 4 % | 0 % | 14 % | 53 % | 94 % |
| ZNE-Gq | 0 % | 0 % | 0 % | 7 % | 47 % | 88 % |

Logistic 50 % points: λ*_LC = 0.57 (ZNE-Lq), 0.58 (exponential), 0.60 (global partial). Against fractional global
folding, fractional local folding wins or ties in 62 % of the cells.

**Which gates (F6).** The subset moves E(λ) by more than shot noise: over 8 seeds the bias of a single nested subset
has an SD of 0.016–0.029 — as large as the bias itself. Median |b|/|b_RAW|: stratified 0.28, nested 0.31, balanced
0.45, random 0.54; **averaged over K = 4 subsets 0.135, K = 16 0.142**; the RMSE ratio to ZNE-G at 10⁶ shots drops
from 2.33 (one nested subset) to 1.44 (K = 4). The subset noise falls as 1/√K (0.0155, 0.0118, 0.0048 for K = 1, 4,
16 on Brick-4, L = 8). The script adds the tables `fractional`, `fractional_regime`, `fractional_crossover`,
`subset_noise` and the figure `folding_fractional_regime_<stage>`; the λ_eff and depth tables group by the requested
scale set (a fractional fold reaches only nearby scales on few gates: 1, 1.33, 1.67, 2 becomes 1, 1.29, 1.67, 1.95
on 21 gates).

## 7. Divergence v2: explaining the gap (`qemx/context_noise.py`, `manifests/direction_A_divergence_v2.yaml`)

Research plan, literature review and the adaptation of a published benchmark design to IBM heavy hex: `docs/v14/research_plan_divergence.md`.

**New in the plan.**

* *Variants* (`variants: {count, mode, threshold}`): `layouts` — disjoint chains of the chip (needs `layout: chain`),
  `twirl` — Pauli-twirled copies, `seeds` — transpiler seeds. The benchmark's budget is shared by its variants.
* *Spectator-context probes* (`family: context`): a random two-qubit core on (c0, c1) of the chain a–s0–c0–c1–s1–b
  with spectators idle in |0⟩, idle in |+⟩, or driven by CZ gates; alignment barriers survive compilation and every
  fold; the ideal output is the same in all three modes.
* *Context features* of every as-executed circuit: two- and one-qubit gates, active, passive and CR-control
  spectators, parallel adjacent two-qubit gates, idle µs on the ALAP schedule, measured qubits, and the calibrated
  gate and readout error.
* *Injections* for emulation: `crosstalk`, `scale`, `coherent`, `t2` (every T2 lowered to so many µs) and `hotspot`
  (every gate on one physical qubit 5× its reported error).

**New in the analysis** (each validated on planted errors — the diagnostic matrix below):

* aggregates of the variants: simple aggregation `RAW·agg` and Chen et al.'s plurality vote `RAW·PV`, with Cochran's
  heterogeneity of the variants' Δ (het_Q, het_p, het_I2);
* **excess error per two-qubit gate**, (ln F_pred − ln F_obs)/n₂, and its volumetric plot;
* the **χ² ladder** (`model_ladder`): nested weighted fits of the excess log-infidelity — none, a rescaled
  calibration c·(−ln F_pred), context features, both — with the verdict *as modeled*, *rescaled calibration ×(1 + c)*,
  *context*, and *+ non-linear* when no linear model fits;
* the **divergence budget** (`attribution`): weighted non-negative least squares over the mechanisms, bootstrap
  intervals, VIF, a stacked budget per family (figure `divergence_budget`);
* the **folding fingerprint** (`fold_response`, `fingerprint_summary`): A = κ_obs/κ_pred under global and local
  folding, pooled per family by a weighted fit; readings gate-attached, time-attached, repetition-coherent (figure
  `divergence_fingerprint_<predictor>`);
* the **probe reading** (`context_table`): the excess contrast of superposed and driven spectators against idle ones,
  combined over depths and variants (figure `divergence_context_probes`);
* the **heavy-hex map** (`qubit_table`, `heavy_hex_coordinates`): the excess per two-qubit-gate participation of every
  physical qubit on the device's layout (figure `divergence_heavy_hex`);
* `recovery LABEL=DIR …`: the **diagnostic matrix** of several analyses (figure `divergence_recovery`).

**Validation: the pilot** (`manifests/direction_A_divergence_pilot.yaml`; 29 benchmarks × 3 layouts = 87 circuits,
6 000 shots per benchmark and method; truth = the depolarizing model of fake_fez with one planted error):

| planted | χ² ladder | budget | fingerprint A_global, A_local | probes (Stouffer z) | layouts | ε_eff ratio |
|---|---|---|---|---|---|---|
| none | as modeled (χ²/dof 1.22) | — | 0.98, 0.96: as modeled | none (−1.3, −0.5) | 0 of 29 differ | 1.03 |
| gate errors ×1.6 | rescaled calibration ×1.54 | rescaled calibration 90 % | 1.51, 1.48: gate-attached | driven only (10.6) | 0 of 29 | 1.60 |
| T2 = 30 µs | context + non-linear | idle time 100 % | 1.25, 1.01: time-attached | superposed only (13.5) | 1 of 29 | 1.35 |
| ZZ 0.05 rad after every CZ | context + non-linear | parallel 38 %, idle 31 %, active spectators 18 % | 2.02, 2.44: repetition-coherent | both (4.2, 7.6) | 0 of 29 | 1.77 |
| defect on qubit 110 | context + non-linear | rescaled calibration 60 %, parallel 40 % | 1.03, 1.08: as modeled | driven only (8.6) | **13 of 29**, worst qubits 111, 110 | 1.29 |

Each planted cause is named by the diagnostic built for it, and only one diagnostic is misleading: the budget's split
of the crosstalk (spectator exposure, parallel gates and idle time grow together in application circuits — VIF up
to 23 — and a coherent error is not additive); the designed probes and the fingerprint separate what the regression
cannot. A defect on one qubit is seen only by the position column: every other diagnostic pools over the chip, so it
reads the defect as a partly rescaled calibration plus something non-linear — which is why the layout variants are
part of the design. Null calibration: the 405 per-variant
z-scores of all methods have mean 0.02, SD 0.95, KS p = 0.31, one |z| > 3; no variant group has heterogeneity
p < 0.01 (smallest 0.052).

**The v2 manifest** (52 benchmarks × 3 layouts = 156 circuits, five methods, 2 868 circuits, 2.21 M shots; the
header lists the validation commands), validated the same way before any hardware time
(`results_1.4.1/divergence_v2_emulated/`):

| planted | χ² ladder | budget | fingerprint | probes (z) | layouts | ε_eff ratio |
|---|---|---|---|---|---|---|
| none | as modeled (χ²/dof 0.93) | — | 0.96, 0.99: as modeled | none (−0.8, −0.3) | 0 of 52 | 1.03 |
| gate errors ×1.6 | rescaled calibration ×1.57 (share 99 %) | rescaled calibration 87 % | 1.31, 1.27: gate-attached | driven only (22.7) | 2 of 52 | 1.60 |
| T2 = 30 µs | context + non-linear (share 31 %) | idle time 100 % | 1.12, 0.97: time-attached | superposed only (34.9) | 0 of 52 | 1.28 |
| ZZ 0.05 rad | context + non-linear (share 74 %) | parallel 45 %, active spectators 28 %, idle 27 % | 2.11, 2.75: repetition-coherent | both (7.6, 17.8) | 2 of 52 | 1.66 |

With 156 precise circuits the first-order rescaling c·(−ln F_pred) is rejected even for a pure ×1.6 (χ²/dof 1.86),
because ln F is linear in the error scale only to first order; the ladder therefore calls a gap a rescaled calibration
when context is not significant *or* immaterial (the calibration alone removes ≥ 95 % of the χ² all mechanisms
remove), and flags "non-linear" above χ²/dof = 2. The fingerprint of the ×1.6 truth is 1.3 rather than 1.6 on this
deeper suite: the fits of ln F against λ = 1, 3, 5 flatten where F reaches the floor; the reading (equal, above 1) is
what matters, not the value.

## 8. Fixes in this round

* `compare` crashed when a method had no runs on an instance (a fractional ZNE skipped on a short circuit):
  it now skips the method there; `fold_response` likewise.
* The folding fingerprint used the model's *exact* probabilities for κ_pred while κ_obs comes from finite shots;
  ZNE gives its high-λ folds the fewest shots, so the observed decay was steeper by a sampling bias (A ≈ 4 on
  brickwork under the null). κ_pred is now the mean over replicates at the same shots; the interval of A is the
  basic bootstrap (the plain percentile interval inherits the bootstrap's own bias).
* The z of Δ pools the model's replicate SD with the observation's bootstrap SE (a plurality vote on a clean peak has
  a replicate SD near 0, which made z explode).
* The effective error-rate table fitted a decay law to ZNE-extrapolated distributions; it now covers unmitigated and
  readout-mitigated estimators only.
* `heavy_hex_coordinates` drew ibm_torino's pendant qubits (129–132) on top of their neighbors.
* `submit` sets dynamical decoupling and service-side twirling off explicitly (`_plain_sampler`): the analysis
  measures idle windows and plans its own twirls. Note: `SamplerV2` is deprecated in qiskit-ibm-runtime 0.50 in
  favor of the client-side `executor_sampler.Sampler` (research plan §8).
* The exported script's IBM path computed λ of a *local* fold from a global fold, and folded a different subset
  than the app.
* The plan prints its skipped methods grouped (`error_summary`), not one line per circuit.
* Plans made before this round carry no context features; `dirA_divergence.py refresh-features RUN_DIR`
  recomputes them (circuits and digests unchanged; the old plan is kept), and the χ² ladder says when they are
  missing instead of concluding that context adds nothing.

## 9. Tests added

* `tests/test_fractional_zne.py` (23): fold orders (permutations, seeds, bit reversal, balanced prefixes), achieved
  λ, nestedness of every nested order, stratified spread, the 1.4.0 random subset, alignment barriers, poly weights
  and Σ|η|, fractional ZNE in a cell and inside the chain, `fold_draws` (draw 0 = the single-draw circuit, equal
  per-shot variance, 1 + 3K circuits, per-circuit ledger weights, validation), the divergence roles and plan skips.
* `tests/test_context_noise.py` (26): Poisson-binomial tail and plurality vote against enumeration, layout-specific
  errors voted out, fallback without consensus; ALAP schedule and every feature on hand-built circuits; heavy-hex
  coordinates of fake_fez and fake_torino; NNLS recovering a planted mechanism and a bad qubit; the context family;
  keep barriers through compilation and folds; layout and twirl variants; heterogeneity, z and excess; aggregates and
  votes under the null; the fingerprint under the null and with doubled gate errors; the t2 injection; the χ² ladder
  and the budget on planted data; the diagnostic matrix.
* `tests/test_zne_folding.py`: the manifest's six stages (914 cells). `tests/test_divergence.py`: circuits run as
  written (no DD, no twirling).
