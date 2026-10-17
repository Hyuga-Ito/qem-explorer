# What average gate fidelity cannot see — research and architecture plan

QEM Explorer 1.4.2 · Direction A · thesis sub-theme: *detecting, visualizing and explaining the noise of a real IBM
device that a fake device built from average gate fidelities does not capture*.

Every number in this document comes from runs of the shipped code and manifests in the development sandbox (fake_fez,
Qiskit 2.x, Aer 0.17). **No IBM device was reachable from the sandbox.** The "observed" data are *emulated*: a fake
device with a known, deliberately injected error. They test whether each tool names a cause we planted; the hardware
run that answers the thesis question is §6.

---

## 0. Summary

| question | tool (module) | result on emulated ground truth |
|---|---|---|
| Is the device worse than its fake twin, beyond shot noise? | Δ and pooled z per circuit; difference volumetric plot (`qemx.divergence.compare`, `dirA_figures.difference_volumetric`) | null: 405 per-variant z-scores, mean 0.02, SD 0.95, KS p = 0.31 |
| Is the gap a mis-scaled calibration or context? | **χ² ladder** (`dirA_divergence.model_ladder`) | gate errors ×1.6 → "rescaled calibration ×1.54"; T2, ZZ crosstalk → "context" |
| Which context? | **divergence budget**: weighted NNLS over circuit features (`qemx.context_noise`) | T2 → idle time 100 %; ×1.6 → rescaled calibration 90 %; crosstalk → spectators + parallel gates (+31 % misassigned to idle) |
| Is it attached to gates, to time, or coherent? | **folding fingerprint** A_global vs A_local (`qemx.divergence.fold_response`) | ×1.6 → gate-attached (1.51, 1.48); T2 → time-attached (1.25, 1.01); ZZ → repetition-coherent (2.02, 2.44) |
| Do spectators matter? | **context probes** zero / plus / driven spectators (`phase.families.context`) | T2 → superposed only (z = 13.5); ×1.6 → driven only (z = 10.6); ZZ → both (z = 4.2, 7.6) |
| Where on the chip? Does it depend on the layout? | layout variants, Cochran heterogeneity, **heavy-hex map** | a defect on qubit 110 → 13 of 29 benchmarks' layouts disagree (none under the null), worst qubits 111, 110 |
| Does fractional ZNE (λ = 1, 4/3, 5/3, 2) help at equal budget? | F5/F6 of the folding study (`qemx.methods.zne`, `qemx.transforms.fold_order`) | only past a light-cone fault count λ_LC ≈ 0.6 (wins 94 % of cells above 0.8, 3–14 % below 0.4); averaging the fold subset halves its bias |

The **diagnostic matrix** (`dirA_divergence.py recovery`, figure `divergence_recovery`) puts the six answers side by
side for every data set. On the five emulated ground truths of the pilot and the four of the v2 suite, each planted
cause is named by the diagnostic built for it, and the only misleading answer is the budget's split of the crosstalk
(a single-qubit defect is seen only by the position column, by design); the hardware run adds one row.

---

## 1. The question and the hypotheses

A fake device (`FakeFez`, `FakeTorino`, …) is built from component-level metrics: average gate infidelities from
randomized benchmarking, readout assignment errors, T1, T2, gate durations. Chen et al. found the same for IonQ Forte:
the per-gate depolarizing model predicts the application benchmarks well for small circuits and too optimistically for
large ones — a fitted effective two-qubit error about twice the benchmarked one. On an IBM heavy-hex device the
candidate mechanisms differ (spectators, cross-resonance or CZ crosstalk, simultaneous neighboring gates, idle
dephasing, local defects), and so do the experiments that can separate them.

* **H0** the fake device predicts every application circuit within shot noise.
* **H1** the gap is a uniform rescaling of the calibrated errors (the benchmarks under-report every error by one
  factor): no context needed.
* **H2** the gap needs context — which neighbors are spectators, which gates run in parallel, how long qubits idle.
* **H3** part of the gap is coherent or non-additive (it does not grow linearly in the number of events).
* **H4** the gap depends on *where* on the chip a circuit runs, beyond what the calibration of those qubits says.

Each tool below tests one of them, with a null distribution derived from the shot noise of the actual budget.

---

## 2. Literature review

### 2.1 Chen et al., *Benchmarking a trapped-ion quantum computer with 30 qubits*, Quantum 8, 1516 (2024), arXiv:2308.05071v2

What they do, and what this project takes from it:

* **Application benchmarks with variants.** Every benchmark circuit runs in 25 *variants*, 100 shots each; in their
  design a variant changes both the local decomposition of the entangling gates (the gate frame) and the qubit
  assignment, never the ideal output. Here: `variants: {count, mode}` with `layouts` (disjoint heavy-hex chains, then
  the same chains reversed, then overlapping chains — never one assignment twice), `twirl` (independent Pauli twirls
  of every two-qubit gate), `layouts+twirl` (both at once) and `seeds`. The budget is shared by the variants.
* **Aggregation and plurality voting** (App. B, Eqs. 4–6, after Maksymov et al., arXiv:2301.07233). A bit string
  survives if it appears in at least t variants; its vote is the Poisson-binomial probability that it is drawn in ≥ t
  of the variants. Following the authors' reference implementation, t = min(7, the largest occurrence) is set in one
  step (t ≤ 2: the average), the survivors keep their *raw* frequencies (no renormalization over the survivors) and
  variants without a survivor are dropped. `qemx.context_noise.plurality_vote` follows it step by step and is tested
  against brute-force enumeration of every joint draw. For V ≠ 25 the start
  threshold scales as round(0.28 V) with a floor of 3.
  Observation from the pilot: the vote sharpens peaked outputs (BV, QFT on a basis state: Fc → 1.00) and *lowers* Fc of
  broad outputs (context probes, brickwork: 0.83–0.88 against 0.99 aggregated) — it is a decoder for peaked
  distributions, not a general estimator. Both aggregations are reported (`RAW·agg`, `RAW·PV`).
* **Hellinger-type fidelity** Fc = (Σ √(p_i q_i))² against the ideal distribution, and the **predicted − observed**
  difference on the **volumetric plane** (width × post-compilation two-qubit gates in log₂ bins, their Fig. 11).
  Adopted as is; Δ is evaluated *at the same shots* (Monte-Carlo replicates of the prediction through the same
  estimator), and every circuit carries a z-score.
* **Component-level simulation**: depolarizing gates with ε₁Q = 2.0·10⁻⁴, ε₂Q = 46.4·10⁻⁴ (entanglement
  infidelities), no SPAM; every variant simulated and *sampled* at the same shots as the experiment. Adopted as the `chen` predictor (one median rate per gate class, no readout), beside the
  per-gate `depolarizing` predictor (every gate its own reported error, with readout) and the full `device` model.
  IBM reports average gate infidelity r; Chen's ε is the entanglement infidelity, r = ε·d/(d + 1).
* **Effective error rate**: log Fc linear in the two-qubit count, fitted to observation and prediction. Adopted
  (`effective_rate`, the `rates` table), for unmitigated and readout-mitigated estimators only — an extrapolated
  distribution does not decay exponentially.

What the paper does not do and this project adds: separating mechanisms (§4.2–4.5), position dependence (§4.6), and
the shot-noise null for every statement.

### 2.2 What this thesis takes from Chen et al. — and what it does not

The authors also published their circuits, histograms and scripts. **This thesis uses none of their data.** Two
reasons, besides the wish to stand on its own measurements: their circuits are compiled for all-to-all MS gates (on
heavy hex they would be routed with SWAPs and become different circuits), and the question here is about the context
noise of an IBM device, which only IBM runs can answer. What is taken over is methodological — the variant design,
the plurality vote, F_c, Δ on the volumetric plane and ε_eff — re-implemented and tested in QEM Explorer. The suite is
regenerated per IBM device (GHZ, BV, QFT, Grover, brickwork, Ising Trotter, context probes), compiled on a chain,
with every as-executed circuit pinned by a SHA-256 digest.

### 2.3 Lubinski et al., *Application-oriented performance benchmarks*, IEEE TQE 4 (2023), arXiv:2110.03137

Normalized fidelity max((Fc − Fc,uniform)/(1 − Fc,uniform), 0) — the share of the fidelity above the uniform
distribution's — and normalized depth (transpiled to {rx, ry, rz, cx}, all-to-all). Adopted: `normalized_fidelity`,
`lubinski_depth`; the volumetric plot can use either depth as its x axis. The **logarithm of the normalized fidelity**
is the response of every regression here (it is additive in independent error events; Fc is not).

### 2.4 Context-dependent noise on superconducting processors

* Rudinger et al., *Probing context-dependent errors in quantum processors*, PRX 9, 021045 (2019),
  arXiv:1810.05651 — the same gate behaving differently in different circuit contexts; the template for the probes.
* Sarovar et al., *Detecting crosstalk errors in quantum information processors*, Quantum 4, 321 (2020) — crosstalk
  as a violation of locality/independence assumptions, detectable by designed experiments.
* Rudinger et al., *Experimental characterization of crosstalk errors with simultaneous gate set tomography*, PRX
  Quantum 2, 040338 (2021), arXiv:2103.09890 — simultaneous operation of neighboring gates.
* Zhao et al., *Quantum crosstalk analysis for simultaneous gate operations on superconducting qubits*, PRX Quantum 3,
  020301 (2022) — static ZZ and simultaneous-drive crosstalk on fixed-frequency transmons.
* Murali et al., *Software mitigation of crosstalk on NISQ computers*, ASPLOS 2020 — scheduling around parallel
  adjacent gates (the `parallel_adjacent` feature).
* Sundaresan et al., *Reducing unitary and spectator errors in cross resonance with optimized rotary echoes*, PRX
  Quantum 1, 020318 (2020) — spectator errors of CR gates (the `control_spectators` feature, for ECR/CX devices).
* McKay et al., *Benchmarking quantum processor performance at scale*, arXiv:2311.05933 — layer fidelity: errors
  measured on full simultaneous layers, which is what an isolated-gate RB number misses.
* Proctor et al., *Measuring the capabilities of quantum computers*, Nature Physics 18, 75 (2022) — volumetric
  benchmarks and the gap between component metrics and circuit performance.
* Wallman & Emerson, PRA 94, 052325 (2016); Hashim et al., PRX 11, 041039 (2021) — randomized compiling turns coherent
  errors into stochastic ones (relevant to H3 and to the `twirl` variants).
* van den Berg et al., Nature Physics 19, 1116 (2023); Kim et al., Nature 618, 500 (2023) — learned sparse
  Pauli–Lindblad layer noise on IBM devices: the per-layer, context-aware model that a fake device is not.

### 2.5 Zero-noise extrapolation and folding

Temme, Bravyi & Gambetta, PRL 119, 180509 (2017); Li & Benjamin, PRX 7, 021050 (2017); Endo, Benjamin & Li, PRX 8,
031027 (2018); Giurgica-Tiron et al., *Digital zero noise extrapolation for quantum error mitigation*, IEEE QCE 2020,
arXiv:2005.10921 — unitary folding, global (U(U†U)^k) and local (G(G†G)^k), and *partial* folding for non-odd scale
factors (fold a subset of gates once more, chosen from the left, the right or at random). Fractional folding is
therefore not new; what is new here is (§5) the choice of *which* subset (nested, stratified, error-balanced orders),
the averaging over subsets, and an equal-budget map of where fractional scales pay.

---

## 3. Adapting the benchmark design to IBM heavy hex

| element | Chen et al. (IonQ Forte) | here (IBM Heron, heavy hex) | where |
|---|---|---|---|
| connectivity | all-to-all (no routing) | heavy hex, degree ≤ 3; circuits laid on a **chain** (best calibrated path, beam search) so routing is confined and comparable across widths | `chain_for_backend` |
| variants | 25, each with its own gate decomposition and qubit assignment | `layouts`: disjoint chains, then reversed, then overlapping (never one assignment twice); `twirl`: Pauli twirls; `layouts+twirl`: both | `make_plan`, `disjoint_chains` |
| native two-qubit gate | MS (XX) | CZ (Heron) or ECR (Eagle); directional ECR adds the CR control spectator | `circuit_features` |
| component error | DRB ε (entanglement infidelity) | RB r (average infidelity), per gate *and* per qubit pair | `predictor_model` |
| dominant context noise | (motional modes, spin-phase drift, crosstalk via beams) | static ZZ to spectators, simultaneous neighboring gates, idle dephasing, TLS defects on single qubits | §4 |
| simulation | depolarizing, no SPAM | depolarizing per gate **with readout**; Chen's median model; the full fake device (T1/T2, idle windows) | `predictors` |
| statistics | point estimates | replicates of the prediction at the same shots, bootstrap of the observation, pooled z | `compare` |
| plot | predicted − observed on width × 2Q gates (√2 bins) | the same (log₂ or √2 bins) on a symmetric diverging scale with a significance dot per tile, plus the fidelity gap with ε_eff, excess error per gate and numbered worst circuits on request | `difference_volumetric`, `fidelity_gap` |

Idle time is the largest qualitative difference: Forte's qubits idle in a long-coherence regime, a heavy-hex qubit
idles for ~0.1–1 µs per layer next to always-on ZZ couplings. The ALAP schedule IBM backends use is reproduced
(`alap_schedule`) so the idle feature counts the windows the device actually has (a qubit waits in |0⟩ before its
first gate, so that lead time is not idle).

---

## 4. Original contributions: metrics, estimators and figures

Each tool is defined on the *as-executed* circuits of a plan, uses only the shots the methods already spend, and comes
with its null distribution. Pictures are produced by `dirA_divergence.py analyze` and `recovery`.

### 4.1 Excess error per two-qubit gate (and its volumetric plot)

ex_c = (ln F_pred,c − ln F_obs,c) / n₂,c (F: normalized fidelity; NaN below F = 0.02). Δ = Fc_pred − Fc_obs grows with
circuit size and vanishes again when both decay to the uniform floor, so a big and a small circuit are not comparable
on Δ; ex is the unmodeled error *per gate*, comparable across the whole volumetric plane
(`divergence_volumetric_excess_per_gate`). Significance outlines use the z of the normalized fidelity.

### 4.2 The χ² ladder — calibration or context? (H0, H1, H2, H3)

Every circuit's excess log-infidelity y_c = ln F_pred − ln F_obs (readout-mitigated, so readout drops out) with its
standard error SE_c (delta method; the observation's bootstrap pooled with the model's replicate SD). Nested weighted
fits, coefficients ≥ 0:

| model | y_c = | tests |
|---|---|---|
| none | 0 | H0 |
| calibration | c · (−ln F_pred,c) | H1: every calibrated error (1 + c)× too low, propagated through the circuit by the model itself |
| context | Σ_k β_k x_ck (spectators, parallel adjacent gates, idle µs) | H2 alone |
| calibration + context | both | H2 beyond H1, by Δχ² |

With correct SEs, a sufficient model has χ²/dof ≈ 1 — checked under the null: 1.22 (p = 0.08) on 87 circuits.
Verdicts: *as modeled*; *rescaled calibration ×(1 + c)* when context adds nothing significant (Δχ² p > 0.01) or
nothing material (the calibration alone removes ≥ 95 % of the χ² that all mechanisms together remove — with 156
precise circuits the first-order rescaling is always rejected somewhere, since ln F is linear in the error scale only
to first order); *context* otherwise; *+ non-linear* when even the full model leaves χ²/dof > 2 (H3). The regressor
−ln F_pred is the key choice: the simple count of gates (or Σ of reported errors) misses how an error propagates to
the output distribution, and fitted the ×1.6 injection with χ²/dof 2.3–4.1 instead of 1.25.

| emulated truth (pilot, 87 circuits) | χ²/dof none → calibration → context → both | calibration share | verdict |
|---|---|---|---|
| no injection | 1.22 → 1.22 → 1.25 → 1.25 | — | as modeled |
| gate errors ×1.6 | 4.68 → **1.25** → 2.09 → 1.25 | 99 % | rescaled calibration ×1.54 (Δχ² p = 0.20) |
| T2 = 30 µs | 3.57 → 2.95 → **2.32** → 2.32 | 51 % | context + non-linear |
| ZZ 0.05 rad | 12.6 → 7.73 → 6.45 → **6.51** | 78 % | context + non-linear |
| defect on qubit 110 | 6.91 → 6.38 → 6.42 → 6.36 | 87 % | context + non-linear |

On the v2 suite (156 circuits) the same truths give: none 0.93 (as modeled); ×1.6: 10.1 → 1.86 → 3.52 → 1.82,
calibration share 99 % (rescaled calibration ×1.57); T2: 11.3 → 9.73 → 6.13 → 6.13, share 31 % (context +
non-linear); ZZ: 20.2 → 11.6 → 8.59 → 8.59, share 74 % (context + non-linear).

Non-linear is the right answer for both T2 and ZZ: dephasing hurts only qubits idling *in superposition* (a qubit in
|0⟩ is immune), and a coherent ZZ rotation adds in amplitude. For the defect it is the position dependence that no
circuit-level feature describes — the layout variants catch it (§4.6).

### 4.3 The divergence budget — which context? (H2)

The same weighted non-negative least squares with every mechanism of `qemx.context_noise.BUDGET` (priced error,
active/passive/control spectators, parallel adjacent pairs, idle µs), bootstrap intervals over circuits, VIF for
collinearity, and a per-family stacked budget with the unexplained remainder (`divergence_budget`). Recovered: T2 →
idle time 100 % (0.0059 per µs, 95 % CI 0.0025–0.0089); ×1.6 → rescaled calibration 90 %. Not recovered cleanly: ZZ
crosstalk → parallel adjacent 38 %, idle 31 %, active spectators 18 %. The reason is structural: in application
circuits, spectator exposure, parallel gates and idle time grow together (VIF of active spectators 23), and a coherent
error is not additive. **The budget is therefore reported only together with the designed probes and the
fingerprint**, which do separate these mechanisms; the recovery matrix makes that limitation visible instead of hiding
it.

### 4.4 The folding fingerprint — gate, time, or coherent? (H2, H3)

ZNE already runs each circuit folded at λ = 1, 3, 5, globally (U(U†U)^k: every gate *and every idle window* repeated)
and locally (G(G†G)^k: only the two-qubit gates). From those runs, with no extra shots: κ = −d ln F/dλ of the
readout-mitigated normalized fidelity, for the observation and for the model's replicates *at the same shots* (finite
shots bias κ — the high-λ folds get the fewest shots — and the bias must be the same on both sides; using the model's
exact probabilities inflated A to 4 in the null). A = κ_obs/κ_pred; pooled per family by the weighted fit
κ_obs = A·κ_pred with the bootstrap SEs (χ²/dof says whether the circuits share one A).

* A_global ≈ A_local > 1 — **gate-attached** (calibration under-reports gate errors; incoherent crosstalk during gates);
* A_global > A_local ≈ 1 — **time-attached** (idle dephasing, drift: only global folding stretches idle windows);
* A_local > A_global — **repetition-coherent** (the same gate repeated back to back builds up a coherent error).

| emulated truth | A_global | A_local | reading (3 SE) |
|---|---|---|---|
| none | 0.98 ± 0.04 | 0.96 ± 0.04 | as modeled |
| gate errors ×1.6 | 1.51 ± 0.05 | 1.48 ± 0.05 | gate-attached |
| T2 = 30 µs | 1.25 ± 0.04 | 1.01 ± 0.04 | time-attached |
| ZZ 0.05 rad | 2.02 ± 0.06 | 2.44 ± 0.07 | repetition-coherent |

Figure `divergence_fingerprint_<predictor>`: every circuit faint at (A_global, A_local) on log axes with the three
regions shaded, every family's pooled point bold.

### 4.5 Spectator-context probes (H2)

A 6-qubit chain a–s0–c0–c1–s1–b; a random two-qubit core (RY, RZ, CZ layers) on (c0, c1); the spectators s0, s1
either **zero** (idle in |0⟩), **plus** (idle in |+⟩, H … H) or **driven** (CZ(a, s0), CZ(s1, b) in every layer);
alignment barriers (`keep`) survive compilation and every fold. The ideal output is the same in all three modes
(tested), so the component model predicts almost the same fidelity for zero and plus. The excess contrast
[Fc(zero) − Fc(mode)]_obs − [·]_pred, combined over core depths and layout variants (Stouffer z), reads:

| significant excess in | reading | emulated truth that produced it |
|---|---|---|
| plus only | idle dephasing (superposed spectators idle) | T2 = 30 µs: plus z = 13.5, driven 0.9 |
| driven only | gate error (the driven mode adds CZ gates) | gate errors ×1.6: driven z = 10.6, plus 1.2 |
| plus and driven | spectator crosstalk | ZZ 0.05 rad: plus 4.2, driven 7.6 |
| neither | no spectator effect | none: −1.3, −0.5 |

Figure `divergence_context_probes`: Fc against core depth per mode (observed and predicted), and the excess contrast.

### 4.6 Position: layout variants, heterogeneity and the heavy-hex map (H4)

Three variants of every benchmark run on disjoint chains. Cochran's Q of the variants' Δ (weights 1/SE²) tests whether
they share one divergence (heterogeneity p and I²); under the null no group of 29 has p < 0.01 (smallest 0.052).
The per-qubit attribution regresses the excess on the two-qubit-gate load of every physical qubit (weighted,
non-negative, ridge) and draws it on the device's heavy-hex coordinates (`heavy_hex_coordinates`, unit-length
couplers for fake_fez, fake_torino and fake_brisbane, including Torino's pendant qubits). A local defect — the
`hotspot` injection, every gate on one physical qubit 5× its reported error — is the test case (release notes §7): 13 of 29 benchmarks' layouts disagree (none under the null), and
the map's worst qubits are 111 and 110 — the defect and the neighbor that every chain through it also uses.

### 4.7 The diagnostic matrix and the recovery protocol

`dirA_divergence.py recovery LABEL=DIR …` reads the tables of several analyses and draws one row per data set and one
column per diagnostic — a red dot where the diagnostic finds an anomaly, an open ring where it reads "as modeled",
the key numbers underneath (`diagnostic_matrix`). It is the validation protocol of the
thesis: before trusting a reading on hardware, show that the same pipeline, on the same circuits and budget, names each
planted cause. It also states the limits honestly (the budget column for crosstalk).

---

## 5. Fractional ZNE

### 5.1 Design

λ = 1 + 2(k + m/n): every two-qubit gate folded k times, m of the n gates once more. The new part is *which* m
(`qemx.transforms.fold_order`):

* `random` — a seeded random subset per λ (the behavior of 1.4.0; not nested across scales);
* `nested` — a seeded random permutation; λ₁ < λ₂ fold a growing set (monotone noise);
* `stratified` — bit-reversal (van der Corput) order of the gate positions: every prefix spread evenly over the
  circuit (half the gates = every other gate), deterministic;
* `balanced` — greedy: every prefix has the circuit's mean calibrated two-qubit error, so λ in gates ≈ λ in noise;
* `fold_draws: K` — K independent nested subsets per fractional λ, averaged, the shots split over them (same budget,
  K× more distinct circuits). The estimator stays linear: Ê(0) = Σ_jk (η_j/K) E_jk with optimal shots ∝ |η_j|σ_jk/K.

Fits: `linear`, `poly2` (least-squares parabola), `exponential`, `richardson`. Σ|η| (the factor on the shot noise)
on λ = 1, 4/3, 5/3, 2: Richardson 111, poly2 19, linear 3.6 — against 3.5 for Richardson on 1, 3, 5. Methods,
chain, NIL and the divergence plan all accept the fractional roles (`fold:local.stratified:1.333`).

### 5.2 Results (folding study stages F5, F6; fake_fez, equal total budget, exact ledger)

| method (λ = 1, 4/3, 5/3, 2 unless noted) | Σ|η| | median \|b\|/\|b_RAW\| | RMSE / RMSE(ZNE-G) at 10⁴ / 10⁶ shots | cells won at 10⁶ |
|---|---|---|---|---|
| ZNE-G: global, Richardson 1, 3, 5 (reference) | 3.8 | 0.07 | 1 / 1 | — |
| ZNE-Gq: global partial fold, poly2 | 20.6 | 0.36 | 5.1 / 5.3 | 15 % |
| ZNE-Lq: local stratified, poly2 | 20.6 | 0.32 | 5.1 / 5.1 | 16 % |
| ZNE-Lq-exp: local stratified, exponential | 4.5 | 0.23 | 1.1 / 1.3 | 21 % |
| ZNE-Lq-lin: local stratified, linear | 3.9 | 0.32 | 1.07 / 1.74 | 4 % |
| ZNE-Lq-R: local stratified, Richardson | 120 | 0.71 | 29 / 27 | 12 % |

Fractional scales lose on average: the lever arm from λ ≤ 2 to λ = 0 is long, local folding under-amplifies the noise
(λ_eff = 1, 1.27, 1.56, 1.82; global partial folding 1, 1.37, 1.78, 2.12), and the high-weight fits multiply the shot noise. **They win where integer folding
fails**: the share of cells ZNE-Lq wins against ZNE-G at the largest budget, by the light-cone fault count of the
circuit,

| λ_LC | ≤ 0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | 0.4–0.8 | > 0.8 |
|---|---|---|---|---|---|---|
| ZNE-Lq | 3 % | 7 % | 9 % | 14 % | 42 % | **94 %** |
| ZNE-Lq-exp | 3 % | 4 % | 0 % | 14 % | 53 % | **94 %** |

with a logistic 50 % point at λ*_LC = 0.57 (ZNE-Lq), 0.58 (exponential), 0.60 (global partial). Rule: **use
fractional local folding only when the calibration predicts λ_LC ≳ 0.6** — where E(λ = 3, 5) has decayed into the
noise floor. Against fractional *global* folding at the same scales, local folding wins or ties in 62 % of cells.

**Which gates (F6, 16 cells).** The subset moves E(λ) by more than shot noise: across 8 nested seeds the bias of the
single-subset quadratic fit has an SD of 0.016–0.029 (comparable to the bias itself). Median |b|/|b_RAW|: stratified
0.28, nested 0.31, balanced 0.45, random (1.4.0) 0.54; averaging the subset, **K = 4: 0.135, K = 16: 0.142**, with
the RMSE ratio against ZNE-G at 10⁶ shots falling from 2.33 (one nested subset) to 1.44 (K = 4). The 1/√K law of the
subset noise is checked directly (0.0155 → 0.0118 → 0.0048 for K = 1, 4, 16 on Brick-4, L = 8).

---

## 6. The hardware experiment (`manifests/direction_A_divergence_v2.yaml`)

* **Suite**: 52 benchmarks × 3 layout variants = 156 instances: context probes (L = 2, 4, 8, 16 × 3 modes × 2 cores),
  GHZ (Z basis, 2–8), QFT (2–6), BV (3–8), brickwork (4, 6; L = 4, 8, 16), Ising Trotter (4–8, 2 and 4 steps), Grover.
* **Methods**: RAW, RO, ZNE-G, ZNE-L (fingerprint), ZNE-Lq (fractional; skipped on 14 benchmarks with < 6 two-qubit
  gates). 9 000 shots per benchmark and method (3 000 per variant, calibration included).
* **Cost**: 2 868 circuits, 2.21 M shots (`plan` prints the totals). At the 0.57 ms per shot measured on ibm_fez
  this is about 21 QPU minutes plus per-circuit overhead, which is more than one 10-minute window.
* **The 10-minute version** (`manifests/direction_A_divergence_v2_qpu10.yaml`) makes three cuts, in this order:
  * drop ZNE-Lq;
  * keep the probes at L = 4, 16 (both seeds);
  * use `budget: 4500` (1 500 shots per variant).

  The result is 40 benchmarks × 3 variants, 1 680 circuits and 720 000 shots: about 410 s, or about 530 s with a 1.3×
  margin. **Validated in emulation:** the same four planted truths on fake_fez are all named correctly
  (`results_1.4.2/qpu10_recovery`):

  | planted | ladder | fingerprint (A_g, A_l) | probes | ε ratio |
  |---|---|---|---|---|
  | none | as modeled | 0.97, 0.96 | none | 1.00 |
  | ×1.6 | rescaled calibration ×1.45 | gate-attached (1.42, 1.43) | gate error | 1.61 |
  | T₂ = 30 µs | context + non-linear | time-attached (1.38, 0.91) | idle dephasing | 1.30 |
  | ZZ 0.05 rad | context + non-linear | repetition-coherent (1.94, 2.56) | crosstalk | 1.70 |

  Two cautions:
  * the ZZ budget spreads over parallel gates 33 %, calibration 31 % and active spectators 22 %, which is the
    collinearity limit of §4.3;
  * no position cause was planted, so the layout column was tested only for false alarms (none).
* **Session**: plan with `--backend ibm_fez` the same day (chains on today's calibration), submit everything in one
  Batch, collect the calibration *at execution time* (`collect-ibm` asks the service for the properties of the run's
  date). Repeat on a second day for drift: H4 across days is the variant heterogeneity in time.
* **Before submitting**: the four emulations of the header of the manifest and `recovery` — the diagnostic matrix of
  the exact suite and budget to be run. (The pilot's matrix is in `results_1.4.1/divergence_recovery/`.)
* **Reporting**: the hardware row of the diagnostic matrix, the volumetric plots of Δ and of excess per gate, the
  budget with its VIFs, the fingerprint figure, the heavy-hex map, the probe figure. Every figure says
  "emulated" or the backend name and the date.

---

## 7. Threats to validity

* **Emulation is not hardware.** The injected errors are models; the claim is only that the tools name *these*
  causes. Real devices mix several (the ladder's "non-linear" flag and the budget's unexplained share say how much is
  left).
* **Coherent and non-additive errors** break linear attribution (§4.3). The fingerprint and the probes do not assume
  additivity. `variant_mode: twirl` makes coherent errors stochastic (then additive) — a second hardware run with
  twirl variants is the test of H3.
* **Collinearity** of features in application circuits: VIF is reported; the probes are the designed experiment that
  breaks it.
* **The calibration snapshot**: properties are reported once per calibration cycle; between calibration and
  execution the device drifts. Collect the execution-time properties, and read a uniform "rescaled calibration" on
  hardware with that in mind.
* **Readout**: RO uses tensored (per-qubit) mitigation; correlated readout errors stay in the RO excess.
* **Folding assumptions**: the fingerprint assumes the folded circuits are executed as written (checked: ISA and
  two-qubit counts of every folded circuit, `fold_guard_transpiled`); dynamical decoupling inserted by the service
  would change idle windows — `submit` switches DD and service-side twirling off explicitly (`_plain_sampler`).
* **Fractional results** are from simulation of fake_fez's noise (E2 engine): the λ_LC rule is a model statement; on
  hardware it is a hypothesis to test with ZNE-Lq in the v2 manifest.

---

## 8. Architecture

```
manifests/*.yaml ──► qemx.divergence.suite_from_manifest ─► Suite
                     make_plan(suite, backend) ─► Plan (instances: compiled circuits, layouts, ideal outputs,
                     │                                 context features; runs: role, shots, digest)
                     ├─ run-ibm / collect-ibm (qiskit_ibm_runtime SamplerV2 in a Batch) ─► counts.json
                     ├─ run-local (SamplerV2 local mode on the fake twin)                ─► counts.json
                     └─ emulate (truth model + injections: crosstalk, scale, coherent, t2, hotspot) ─► counts.json
                     snapshot ─► per-instance DeviceModel JSON ─► predict (exact / Aer DM / Aer MC) ─► predictions
compare / group_rows / fold_response ─► tables ─► scripts/dirA_divergence.analyse ─► Report (tables + text)
                                                  model_ladder, attribution, fingerprint_summary, context_table,
                                                  qubit_table, rates_table, error_summary
scripts/dirA_figures.py (tables in, figures out; one style system, setup_matplotlib(preset)): fidelity_gap,
difference_volumetric(_grid), predicted_vs_observed, divergence_budget, ladder_chart, folding_fingerprint,
heavy_hex_map, context_probe, diagnostic_matrix, attribute
qemx.context_noise (NumPy/SciPy only): circuit_features, alap_schedule, plurality_vote, nnls_attribution,
qubit_attribution, heavy_hex_coordinates
qemx.transforms.fold_order / fold_local; qemx.methods.zne (poly2, subsets, fold_draws); scripts/dirA_zne_folding.py
(fractional_summary, fractional_regime, subset_noise, fig_fractional_regime)
```

**Contracts.** `qemx.divergence` and `qemx.context_noise` never read or write files; the scripts do I/O through
sources and sinks (`dirA_io`). Every table is a pandas DataFrame with documented columns; every figure function takes
tables and returns a Matplotlib figure. The token never leaves `connect()` (environment variable, saved account or a
hidden prompt; masked in errors; tested by scanning every written file).

**Debugging guide (API drift).** The places that touch external APIs, and what to check when a Qiskit release
changes them:

| symptom | where | what changed, typically |
|---|---|---|
| `submit` fails on `SamplerV2(mode=…)` or `Batch` | `qemx.divergence.submit` | runtime primitive / session signature |
| `DeprecationWarning: The SamplerV2 class is deprecated` (qiskit-ibm-runtime 0.50) | `submit`, `run_local`, `_plain_sampler` | the client-side `qiskit_ibm_runtime.executor_sampler.Sampler` replaces it; port these three call sites, keep DD and twirling off |
| counts missing after `collect` | `counts_from_pub` | the result's classical-register field name (`data.<creg>.get_counts()`) |
| features or durations empty | `device_neighbors`, `model_for` | `backend.coupling_map` / `backend.target[...]` property names |
| `fake_backend("fake_fez")` fails | `qemx.device.fake_backend` | the fake provider's class names (`FakeFez`, `FakeTorino`) |
| ISA check fails after a transpiler update | `physical_circuits`, `fold_guard_transpiled` | optimization at level 0 cancelling folds (it must not) |
| Aer engine mismatch | `measured_probs` | `save_density_matrix` / method names |

Every one of them has a test (`tests/test_divergence.py`, `tests/test_context_noise.py`,
`tests/test_fractional_zne.py`); run them first.

---

## 9. Reproduce

```
python -m qemx.phase.run run manifests/direction_A_zne_folding_v1.yaml --out data/zf --stage F5-fractional --workers 4
python -m qemx.phase.run run manifests/direction_A_zne_folding_v1.yaml --out data/zf --stage F6-subsets --workers 4
python -m qemx.phase.run collect data/zf
python scripts/dirA_zne_folding.py data/zf --out figs_zf

python scripts/dirA_divergence.py plan manifests/direction_A_divergence_pilot.yaml --out data/pilot
for T in null xtalk t2 scale hot; do …emulate (see the manifest header)…; done
python scripts/dirA_divergence.py analyze data/pilot --tag xtalk --out figs_pilot_xtalk --workers 4
python scripts/dirA_divergence.py recovery "no injection=figs_pilot_null" … --out figs_pilot_recovery
python scripts/dirA_divergence.py refresh-features data/old_plan     # plans made before 1.4.1 (no context features)
python scripts/dirA_divergence.py figures figs_pilot_xtalk --run-dir data/pilot --out redrawn --font lm

```

## References

Chen et al., Quantum 8, 1516 (2024), arXiv:2308.05071 ·
Lubinski et al., IEEE TQE 4, 3100332 (2023), arXiv:2110.03137 · Maksymov et al., arXiv:2301.07233 · Rudinger et al.,
PRX 9, 021045 (2019), arXiv:1810.05651 · Sarovar et al., Quantum 4, 321 (2020) · Rudinger et al., PRX Quantum 2,
040338 (2021), arXiv:2103.09890 · Zhao et al., PRX Quantum 3, 020301 (2022) · Murali et al., ASPLOS 2020 ·
Sundaresan et al., PRX Quantum 1, 020318 (2020) · McKay et al., arXiv:2311.05933 · Proctor et al., Nat. Phys. 18, 75
(2022) · Wallman & Emerson, PRA 94, 052325 (2016) · Hashim et al., PRX 11, 041039 (2021) · van den Berg et al., Nat.
Phys. 19, 1116 (2023) · Kim et al., Nature 618, 500 (2023) · Temme, Bravyi & Gambetta, PRL 119, 180509 (2017) ·
Li & Benjamin, PRX 7, 021050 (2017) · Endo, Benjamin & Li, PRX 8, 031027 (2018) · Giurgica-Tiron et al., IEEE QCE
2020, arXiv:2005.10921.
