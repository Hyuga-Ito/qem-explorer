# Literature matrix: QEM Explorer primary sources

**Search and access date:** 2026-10-17 (all sources below were accessed on this date).
**Method:** WebSearch and WebFetch only, on arxiv.org (abs, html, pdf), ora.ox.ac.uk (University of Oxford repository, publisher version of record), quantum-journal.org, research.ibm.com, ibm.com and quantum.cloud.ibm.com. No curl or python fetching.
**Scope limit:** WebFetch reads at most about 100,000 characters of a PDF and about 80,000 to 87,000 characters of an arXiv HTML page. Where a section sat beyond that limit, this file says so, and the gap was filled from another official copy if one existed.

## Versions actually read

| # | Source | arXiv id + version read | Status / venue | Format read |
|---|---|---|---|---|
| 1 | J.-S. Chen et al., *Benchmarking a trapped-ion quantum computer with 30 qubits* | 2308.05071**v2** (2 Nov 2024; v1 9 Aug 2023) | Published: Quantum 8, 1516 (2024-11-07), DOI 10.22331/q-2024-11-07-1516 (confirmed on quantum-journal.org) | HTML v2 (truncated in Sec. 4) + PDF v2 (full, incl. App. A–C) |
| 2 | T. Proctor et al., *Featuremetric benchmarking: Quantum computer benchmarks based on circuit features* | 2504.12575**v1** (17 Apr 2025), the only version listed | **Preprint** (no journal ref on arXiv) | HTML v1 |
| 3 | Zhenyu Chen et al., *Scalable Quantum Error Mitigation with Neighbor-Informed Learning* | 2512.12578**v1** (14 Dec 2025), the only version listed | **Preprint** | HTML v1 (ends in App. D); PDF v1 (ends in App. B.2.2). **App. C (sample-complexity proof) and the reference list could not be read.** |
| 4 | Zhenyu Cai et al., *Quantum error mitigation* | 2210.00921**v3** (28 Dec 2023) + publisher version of record (ORA Oxford) | Published: Rev. Mod. Phys. 95, 045005 (2023), DOI 10.1103/RevModPhys.95.045005 | PDF v3 (Secs. I–III.G), ORA publisher PDF (to mid Sec. III.H). **Sec. III.H second half and all of Sec. IV ("Comparisons and Combinations") could not be read.** |
| 5 | K. Rudinger et al., *Probing context-dependent errors in quantum processors* | 1810.05651**v1** (12 Oct 2018), the only version | Published: Phys. Rev. X 9, 021045 (2019), DOI 10.1103/PhysRevX.9.021045 | PDF v1 |
| 6 | J. Harris, K. Lively, P. Schuhmacher, *Reducing Quantum Error Mitigation Bias Using Verifiable Benchmark Circuits* | 2603.10224**v1** (10 Mar 2026), the only version | **Preprint** | HTML v1 (to Eq. 17) + PDF v1 (Secs. III–V, App. A, C–E) |
| 7 | IBM Quantum docs, "Workload usage" (URL `.../guides/estimate-job-run-time`) | n/a (live page, no last-updated date shown) | Official documentation | Web page |
| 8 | IBM Quantum docs, "Maximum execution time…" (`.../guides/max-execution-time`), plus "Plans overview", "Execution modes", "Choose the right execution mode", "Run jobs in a session", "Run jobs in a batch", EstimatorOptions API page, IBM blog "Doubling down on open-access quantum computing" (16 Mar 2026) | n/a | Official documentation / IBM blog | Web pages |
| 9 | T. Lubinski et al. (QED-C), *Application-Oriented Performance Benchmarks for Quantum Computing* | 2110.03137**v3** (10 Jan 2023) | Published: IEEE Trans. Quantum Eng. 4 (2023), DOI 10.1109/TQE.2023.3253761 (journal confirmed by DOAJ listing and by Chen et al.'s ref. [4]: "IEEE Trans. Quan. Eng. 4, 1–32 (2023)") | PDF v3 (Sec. III.G read) |
| 10 | P. Czarnik, A. Arrasmith, P. J. Coles, L. Cincio, *Error mitigation with Clifford quantum-circuit data* | 2005.10189**v3** (16 Nov 2021) | Published: Quantum 5, 592 (2021), DOI 10.22331/q-2021-11-26-592 | PDF v3 |
| 11 | H. Liao et al., *Machine Learning for Practical Quantum Error Mitigation* | 2309.17368**v2** (22 Nov 2024) | Published: Nature Machine Intelligence (DOI 10.1038/s42256-024-00927-2; IBM Research page dates it 22 Nov 2024). **Volume and page (cited in project docs as 6, 1478) not verified: nature.com fetch failed.** | PDF v2 |

---

## Matrix

| Source | Relevant result / assumptions | Intended use in the app | Expected differences from a typical implementation | Validation required |
|---|---|---|---|---|
| **Chen et al. 2024** (Quantum 8, 1516) | F_c(p,q) = (Σ_i √(p_i q_i))², Eq. 1, **explicitly without** Lubinski's normalization to uniform. 25 variants × 100 shots. Each variant has a different local gate decomposition **and** a different circuit-qubit to physical-qubit map. Two aggregations: simple sum (Figs. 6, 9–11) and plurality vote (App. B, Eqs. 4–6, used only for #AQ, Fig. 7). Component model (Sec. 4, Eq. 2): Pauli-form depolarizing after every gate with ε₁Q = 2.0×10⁻⁴ and ε₂Q = 46.4×10⁻⁴ (median DRB), no SPAM. Simulated with qsim/cuStateVec, every variant, simple aggregation. The model over-predicts fidelity. The observed data suggest a rough rate of 100×10⁻⁴ against the model's ≈ 46.4×10⁻⁴. | `chen` predictor, F_c, plurality vote, Δ on the volumetric plane, effective error rate | (a) Their ε is the **entanglement infidelity / total Pauli-error probability**, not the "depolarizing parameter" of a ρ→(1−λ)ρ+λI/d channel, and not IBM's average gate infidelity r. (b) The plurality vote is a non-linear decoder for peaked outputs. (c) Variants on all-to-all MS hardware ≠ heavy-hex layouts. (d) Ambiguity in App. B over whether surviving frequencies are renormalized. | Unit test: the converted rate reproduces the entanglement infidelity of the simulated channel. Brute-force check of App. B Eqs. 4–5 for t ≥ N_v/2, where they are exact. Report which aggregation produced each figure. Do not mix F_c with the normalized F. |
| **Proctor et al. 2025** (preprint) | Performance s(c) modelled as a function of circuit features (w, d, ξ₂Q, ξ₁Q, mid-circuit-measurement density, qubit indicators, plus "pseudo-features"). GP regression (also a monotonic GP) of the **mean** success probability or process fidelity. About 3 % MAE with about 75 training points (ibmq_montreal). Interpolation only. The authors say the feature sets are likely not rich enough for circuits from another distribution, and that extrapolation is harder. | Justification for predicting performance from circuit features (regression of log normalized fidelity on features, volumetric plane) | They use mirror-circuit success probability and SR-DFE process fidelity, not F_c of application circuits. Their model is a GP, not a linear or log-linear one. Drift is ignored. | Held-out test split from the **same** circuit distribution, plus a separate out-of-family test. Report interpolation and extrapolation errors separately. |
| **Z. Chen et al. 2025, NIL** (preprint) | Neighbors are the target with one X/Y/Z inserted after one noisy gate (weight-1, U_j→PU_j), or ZNE-folded copies. The estimator is ŷ = Σ_j c_j x_j with **no intercept**. Fit by least squares (Eq. 6) or ℓ1-constrained least squares, ‖c‖₁ ≤ γ (Eqs. 1/7), with γ = 2 (Pauli neighbors) or γ = 5 (ZNE neighbors), using CVXPY+MOSEK. Training uses Algorithm 2: every rotation angle is drawn from {0, π/2, π, 3π/2} (a rotation 2-design, not a unitary 2-design). Theorem 2: T = O(ln(N/δ)/ε²). Assumes gate noise unchanged when angles change. **Simulation only**, with local depolarizing noise of 10⁻³ (1Q) and 10⁻² (2Q). | `nil` estimator, and the theory reference for training-size and design claims | The paper does not say that the bare circuit is a neighbor, how 2Q gates take Paulis, or how the γ-dependence enters the Theorem 2 constant (App. C unread). It is internally inconsistent on the weight-1 count: O(m²) in Sec. 6 versus "the number of gates" in Sec. 4. | Mark the app's additions (bare circuit, insertion only after 2Q gates, intercept options) as deviations. Reproduce one of their small TFI-VQE settings under their noise model before citing agreement. |
| **Cai et al. 2023** (RMP review) | Sampling overhead C_em = Var[Ô_em]/Var[Ô_ρ] = N_shot ratio (Eq. 3), and C_em ~ R²/R² by range (Eq. 5). Fault rate λ = Σ_f p_f (Eq. 7), with P₀ = e^{−λ} under a Poisson model (Eq. 8). Post-selection costs e^{λ} (Sec. II.D). PEC: C_em ~ Q² = (Σ|α_n|)² (Eq. 14). Single gate: ((1+p)/(1−p))² (Eq. 16). Full circuit: e^{4λ} for uniform Pauli noise with M large and λ = Mp finite (Eq. 18). MSE = Bias² + Var (Sec. II.B). ZNE needs amplifiable, time-invariant noise, and Richardson extrapolation is valid only for small λ. PEC needs tomographic noise knowledge. Learning-based methods (Sec. III.H, Eqs. 47–48) need training circuits that are similar in structure (similar faults) and classically simulable. The linear map E₀ ≈ θ₀ + θ₁E is motivated by global depolarizing noise, but the review says that assumption is "not necessary". | Cost ledger (λ, β, C_em), equal-budget comparison, method assumptions | In Eq. 18, λ uses p = **Pauli-error probability per gate** (the entanglement infidelity), not the average gate infidelity r. The project's λ_LC must use the same convention or state its conversion. | Check the ledger: for uniform Pauli noise, the fitted PEC exponent should give β ≈ 4 when λ is defined as in Eq. 7. |
| **Rudinger et al. 2019** (PRX 9, 021045) | Context dependence is tested by running the **same circuits** under C contexts. Per circuit, an LLR statistic (Eq. 4) is compared against χ² with k = (C−1)(M−1) d.o.f. (Eqs. 5–6, Wilks). A Hochberg step-up correction is applied across circuits (Eq. 9), plus an aggregate test (Eqs. 11–13), at α = 5 % with Bonferroni splitting across context comparisons. Effect size is measured by JSD (Eq. 15) and by SSTVD, the TVD reported only if significant (Eqs. 19–21). On ibmqx3: no drift between the two idle contexts, crosstalk from a CNOT on another rung that decays with distance, max SSTVD ≈ 28 % (Q3–Q14 CNOT). | Template for the context probes and their null distribution | They compare **outcome distributions of identical circuits**. A regression of fidelity on features is a different test. Their χ² asymptotics need N_c ≫ 1. | Use the same LLR/χ² test (or an exact or permutation alternative for small counts) on probe pairs. Control nuisance contexts such as circuit duration and time order. |
| **Harris et al. 2026** (preprint) | A verifiable benchmark circuit B(A) has the same native-gate sequence as the application circuit A (C1), a single known output bitstring (C2), and ideal ⟨O⟩ = ±1 (C3). On IBM: "simply replace √X gates with X gates" (zero gate overhead, not necessarily Clifford). Hardware-agnostic version: random Clifford Pauli rotations plus a final layer. Two uses: b-mit rescaling ⟨O⟩_A^QEM / ⟨O⟩_B^QEM (Eq. 5), and bnZNE, which extrapolates against ε(r) measured on B (Eq. 18). Cost: "simply 2N". The analytic argument assumes uniform 2Q depolarizing noise. The benchmarks may miss crosstalk because they create no entanglement. | Candidate bias-calibration control, and a context-probe design for heavy hex | Bias reduction is argued, not proven, beyond uniform depolarizing noise. Eq. 9 is explicitly left unjustified. The fit type (linear or exponential) is chosen on the benchmark circuits. | Simulate under coherent and crosstalk noise, where the authors expect it to be weakest, before using it on hardware. Count the 2× shots in the equal-budget ledger. |
| **IBM "Workload usage"** | Usage is the time "a QPU is locked to execute workloads". Session: elapsed active time. Batch: cumulative lock time of all jobs. Job: its lock time. Estimate: ⟨per sub-job overhead ≈ 2 s⟩ + (rep_delay + circuit length) × executions. Quick formula: 2 + 0.00035 × executions [s]. User-cancelled or failed jobs: usage up to that point is reported. System errors: zero. | QPU budget planner | The quick formula assumes the default rep_delay of 250 µs and about 100 µs circuits. The project measured about 0.57 ms per shot on ibm_fez, so the quick formula underestimates usage for these circuits. | Calibrate the per-shot cost from `job.usage()` of a pilot job. Keep the 2 s per sub-job overhead. |
| **IBM "Maximum execution time"** + plan pages | `max_execution_time` is per job, counts QPU usage only, and is capped by a 3 h service limit. The API default is `Unset`. A job that exceeds it is cancelled with RuntimeJobMaxTimeoutError. Session and batch have a max TTL (`max_time`) and a non-configurable interactive TTL of 1 min. Batch default max TTL: 8 h on paid plans, **10 min on the Open plan**. Its timer "starts when the batch starts", so it is elapsed time, not usage. Open Plan: 10 min of QPU per 28-day rolling window, **no sessions**. Since 16 Mar 2026, an opt-in 180 min over 12 months is available after 20 min of use within 12 months (non-trial accounts). | Budget guards | **No option bounds total usage across a campaign** except the Open Plan allowance itself and (on paid plans) instance cost limits. The page does not say whether a job cancelled for exceeding max_execution_time is billed. | Set `max_execution_time` per job **and** keep an app-side ledger of `job.usage()`. Plan a batch to finish well within the Open-plan 10-min TTL, or split it. |
| **Lubinski et al. 2023** (IEEE TQE 4) | F_s = (Σ_x √(P_out P_ideal))² (Eq. 1). F_raw = (F_s − F_s(P_ideal, P_uni)) / (1 − F_s(P_ideal, P_uni)) (Eq. 3). F = max{F_raw, 0} (Eq. 2). Sec. III.G gives the rationale: "F_s is non-zero even when the output of the circuit is completely random." | `normalized_fidelity`, the regression response | They call F_s the "classical fidelity", related to the Hellinger distance, not "Hellinger fidelity". | Test: F(P_ideal, uniform) = 0 and F(P, P) = 1. Clip at 0 before taking a log. |
| **Czarnik et al. 2021** (Quantum 5, 592) | Linear ansatz f(X^noisy, a) = a₁X^noisy + a₂ (Eq. 2), fitted by least squares (Eq. 3). Training circuits are near-Clifford, with N non-Clifford gates kept and the rest replaced by nearby Clifford gates (powers of S), sampled by MCMC with weights e^{−d²/σ²}, σ = 1/2 (App. B, Eq. 14). "60−80 training circuits and 8192 shots" were enough. The linear form is exact for global depolarizing noise (App. A) but is "not expect[ed] … to correct for all types of noise". | CDR estimator | Many implementations replace gates uniformly at random, without MCMC or closeness weighting. The paper keeps N non-Clifford gates as a refinement parameter. | Record N, the replacement rule and the number of training circuits. Compare with the paper's QAOA setting if CDR performance is claimed to match. |
| **Liao et al. 2024** (Nat. Mach. Intell.) | ML-QEM (OLS, random forest, MLP, GNN) maps features to the ideal value. OLS/RF/MLP features (Sec. IV A 1–3): native-gate counts (parameterized gates in binned angles), the Pauli observable in sparse representation, optional device noise parameters, **and the noisy expectation values**. The GNN also encodes the gate graph, gate errors, T1/T2 and readout errors. Training circuits are random or Trotter circuits with simulated ideal labels, **not Clifford training circuits**. The random forest performed best. Under coherent noise, extrapolation failed. | Closest published relative of the project's FR (regression on noisy ⟨O⟩, ⟨Z_i⟩, ⟨Z_iZ_j⟩ from near-Clifford training) | ML-QEM is trained across a class of circuits and observables. FR trains per target circuit on near-Clifford variants, which is closer to CDR with extra features. It is unclear in the paper whether the noisy values of several observables enter one model jointly. | Describe FR as the project's own heuristic. Cite Liao et al. as related, not as its source. Test FR under coherent noise, where ML-QEM was reported to fail. |

---

## Per-source notes

### 1. Chen et al., "Benchmarking a trapped-ion quantum computer with 30 qubits"

- **Bibliographic:** Jwo-Sy Chen, Erik Nielsen, Matthew Ebert, Volkan Inlek, Kenneth Wright, Vandiver Chaplin, Andrii Maksymov, Eduardo Páez, Amrit Poudel, Peter Maunz, John Gamble. arXiv:2308.05071v2 (v1 9 Aug 2023, v2 2 Nov 2024). Quantum 8, 1516 (2024-11-07). URLs: https://arxiv.org/abs/2308.05071, https://arxiv.org/html/2308.05071v2, https://arxiv.org/pdf/2308.05071v2, https://quantum-journal.org/papers/q-2024-11-07-1516/.
- **Sections:** 1 Introduction; 2 Experimental setup; 3 Benchmarking (3.1 Component-level, 3.2 Application-oriented); 4 Simulation of application circuits; 5 Conclusion; App. A Direct randomized benchmarking; App. B Plurality voting procedure; App. C Mølmer–Sørensen gate properties. The PDF is 18 pages.

**Fidelity (Sec. 3.2, Eq. 1).**
- The definition is F_c(p,q) = (Σ_i √(p_i q_i))², with p the measured and q the ideal outcome distribution. The text notes it equals (1 − H²)², with H the Hellinger distance.
- Quote: "Eq. 1 differs from the final fidelity suggested in [4] by omitting a normalization to the uniform distribution." [4] is Lubinski et al.
- Quote: "Such normalization is insignificant for wide circuits with concentrated output distributions."
- **So Chen et al. do not normalize against the uniform distribution.**
- #AQ pass criterion: all application circuits with w_c ≤ n and d_c ≤ n² have fidelity greater than 1/e. Forte achieves #AQ 29.

**Variants (Sec. 2, Sec. 3.2).**
- There are 25 variants per circuit, each run 100 times, giving 2,500 counts per algorithm instance.
- Variants "differ by local gate decomposition yet result in the same measurement statistics in the absence of noise", and "Each variant also has a different circuit-qubit to physical-qubit assignment."
- Sec. 3.2 says: "the circuit-qubit to physical-qubit mapping may change between variants or gate bases may be randomized."
- Barriers in the QED-C circuits are respected at every compilation level up to local gate optimization.
- Purpose: when the variants are aggregated, "errors will be randomized" instead of accumulating systematically.

**Simple aggregation (Sec. 3.2).**
- "The outcomes of the 25 variant circuits are simply added together (after permuting the qubit labels if needed) to form an overall outcome histogram containing 2500 counts".
- It is used in Fig. 6 (application benchmarks without error mitigation) and in Figs. 9–11 (the comparison with simulation).

**Plurality voting (Sec. 3.2, App. B; introduced in their Ref. [23], Maksymov, Nguyen, Nam, Markov, arXiv:2301.07233).**
- The paper calls it error mitigation: "a classical post-processing step that attempts to amplify the signal and minimize the noise". It is used for #AQ (Fig. 7).
- Procedure (App. B), with "25 variants, 100 shots each, and a starting threshold of t = 7 for all experiments", values "chosen heuristically":
  1. Shuffle the shots within each variant and arrange them in a shots × variants array.
  2. In each row, take the plurality bitstring and add it to the aggregate histogram if its count exceeds t.
  3. Re-randomize and repeat until the aggregate converges.
  4. "If no bit strings have been accepted, decrease t by one."
  5. "If t = 2, then stop and instead average the counts across shots".
- Efficient computation:
  - "if a bit string does not appear within at least t variants, it will never be chosen in the vote. Hence, we omit all such bit strings". Variants left with no surviving bitstrings are dropped, and N_v is the number of surviving variants.
  - f_vb is "the frequency of bit string b in the variant-circuit distribution indexed by v".
  - Eq. 4: p_b^(m) = Σ_{S∈P_m} Π_{v∈S} f_vb Π_{v∉S}(1 − f_vb).
  - Eq. 5: p_b = Σ_{m=t}^{N_v} p_b^(m), then "We then normalize the set of probabilities {p_b}".
  - Eq. 6: for t < N_v/2, p_b = 1 − Σ_{m<t}(…). The symbol G_b^(m) in Eq. 6 is not defined, an apparent notational slip for p_b^(m).
  - The result is exact for t ≥ N_v/2 and approximate for t < N_v/2.
- **Ambiguity:** whether f_vb is renormalized over the surviving bitstrings. The only statement is "All subsequent discussions of probability are relative to this reduced collection of bit strings."

**Component-level characterization (Sec. 3.1, App. A).**
- DRB on all 435 pairs. The fit is a + b·p^d (2Q fits fixed at 0.25 + b·p^d). Two layer mixes, p₂Q = 0.25 and 0.75, are separated into r₁Q and r₂Q by a 2×2 linear system (Eq. 3; garbled in PDF extraction).
- **No explicit r(p) conversion formula appears in the text.**
- "In situations where the noise is known or restricted to be purely stochastic, the DRB error rate is equal to the average entanglement infidelity of a circuit layer."
- Medians: 1Q 2.0×10⁻⁴ (Fig. 3) and 2Q 46.4×10⁻⁴ (Fig. 4), with the worst pair at 885×10⁻⁴. SPAM is 0.5 % per qubit (Sec. 2).

**System-level model (Sec. 4).**
- Eq. 2: Λ_ε(ρ) = (1 − ε)ρ + ε/(4ⁿ − 1) Σ_{P≠I} PρP, applied after every ideal gate. ε₁Q = 2.0×10⁻⁴ and ε₂Q = 46.4×10⁻⁴ are the median DRB rates. "This definition implies that ε1Q equals the entanglement infidelity".
- No SPAM, idle or crosstalk terms. The authors justify the simple model as sufficient for the question asked, and say that "we ignore structure in the errors even at the component level".
- Simulation: "Circuits are simulated using Google's Qsim simulator on a cluster of GPUs" (cuStateVec). Every variant was simulated, and "We combine variant circuit results using simple aggregation rather than plurality-voting."
- Results: "significant discrepancy between the predicted and observed performance". Also: "the observed fidelity values suggest a rough error rate of 100 × 10−4 whereas the predictions (unsurprisingly) show a rate close to the component-level 2-qubit gate error rate, ϵ2Q = 46.4×10−4". Fig. 10 shows "a characteristic arc above the x = y line", and the ratio is "approximately 1/2".
- Fig. 11 plots predicted minus observed fidelity on the width × post-compilation 2Q-gate plane. Its color scale is "strictly positive since … the predicted fidelity is always higher". **The caption does not state the bin factor.** The tick labels look like powers of two, but this is not stated.
- Conclusion (Sec. 5): "these simplistic models were unable to reproduce the experimentally-measured application-level benchmark". Their "current hypothesis is that the noise we observe is stochastic spin-phase noise" with "a non-trivial power spectral density" that "cannot be decomposed into Markovian gate-level noise". They attribute it to mechanical or optical-path fluctuations, which cause context dependence.

### 2. Proctor et al., "Featuremetric benchmarking…" (preprint)

- **Bibliographic:** Timothy Proctor, Anh Tran, Xingxin Liu, Aditya Dhumuntarao, Stefan Seritan, Alaina Green, Norbert M. Linke. arXiv:2504.12575v1 (17 Apr 2025), the only version on 2026-10-17, quant-ph and cs.LG. https://arxiv.org/abs/2504.12575, https://arxiv.org/html/2504.12575v1.
- **Capability function (Sec. II A):** "a performance metric s: ℂ → ℝ, which we call a capability function".
  - Success probability is used for definite-outcome (mirror) circuits (II A 1).
  - Process fidelity is estimated by SR-DFE for general Clifford circuits (II A 2, App. A).
  - Hellinger fidelity is mentioned only as an example. **No uniform-baseline normalization is discussed.**
- **Features (Sec. III):**
  - Depth d, width w, 2Q-gate density ξ₂Q, 1Q-gate density ξ₁Q, mid-circuit-measurement density, and qubit-membership indicators A_q(c).
  - "Pseudo-features" are generator parameters, such as the QAOA graph and p, Trotter step size, or algorithmic depth.
  - Quote: "we do not attempt to provide a comprehensive list."
  - Featuremetric benchmarking generalizes volumetric benchmarking (χ = 2 features) to χ features. A grid of measurements "fails for χ ≥ 3".
- **Model (Sec. V, App. B–C):**
  - GP regression of the **mean** capability s̄(v) (Eqs. 10–12), plus a monotonic GP following Riihimäki and Vehtari. The full distribution of s is not modelled.
  - The parametric form p^{wd} was considered and rejected as too restrictive.
  - The kernel and hyperparameters were not visible in the HTML text read; App. B and C were not in the excerpt.
- **Data (Sec. IV):**
  - ibmq_montreal: mirror circuits over (w,d), K = 20 per shape, N = 1024 shots.
  - ibmq_algiers: (w, d, ξ₂Q), 531 feature values, 5,310 circuits.
  - IonQ Forte1 (20 qubits): random Clifford circuits with SR-DFE, 256 Sobol-sampled feature values, 15,360 circuits.
  - **No QFT or Bernstein–Vazirani application circuits were run.**
- **Claims:**
  - "converging to δ_abs ≈ 3% once around 75 training data are used" (ibmq_montreal, Sec. V B).
  - GP models "can reproduce a volumetric benchmarking plot with high accuracy using much less data".
  - "Our GP models enable the prediction of any circuit's performance from featuremetric benchmarking data" (Sec. VI), immediately qualified.
- **Non-claims and limitations (Sec. VI):**
  - "it [is] unlikely that our current feature sets are rich enough" for circuits from another distribution.
  - Extrapolation "is typically a much more difficult problem than interpolation".
  - "Model training is not currently automated and stable".
  - Drift is ignored (Sec. II A).
  - The 3-D Forte data are "not statistically consistent with a monotonic decay" (Sec. IV C).
  - Sec. III A: it is unclear "whether it is possible to reliably predict s(c) from a small set of features, or what good features are".

### 3. Zhenyu Chen et al., "Scalable Quantum Error Mitigation with Neighbor-Informed Learning" (NIL, preprint)

- **Bibliographic:** Zhenyu Chen, Bin Cheng, Minbo Gao, Xiaodie Lin, Ruiqi Zhang, Zhaohui Wei, Zhengfeng Ji. arXiv:2512.12578v1 (14 Dec 2025). https://arxiv.org/abs/2512.12578, https://arxiv.org/html/2512.12578v1, https://arxiv.org/pdf/2512.12578v1.
- **Neighbor circuits (Sec. 2, Sec. 6):**
  - Neighbors are a set of modified circuits {C₁,…,C_N}, formed by inserting single-qubit gates after noisy operations.
  - Pauli neighbors use G = {X, Y, Z}: "when inserting a gate P ∈ G after the single gate U_j, it will be replaced by PU_j". "weight-m Pauli neighbors refer to circuits where a Pauli gate is inserted after at most m noisy gates", "the same as the learning-based PEC approach [16]".
  - CPTP-basis neighbors and ZNE neighbors are also allowed. In Sec. 6 the ZNE neighbors use α ∈ {1, 1.1, 1.34, 1.58}.
  - Selection: "we randomly select s circuits from the weight-1 neighbor circuits".
  - Count: Sec. 6 says the weight-1 set "is of size O(m²)" for m noise positions. Sec. 4 says its number "equals the number of gates in the target circuit". **These two statements are inconsistent.**
  - **Not stated:** whether the unmodified circuit is itself a neighbor, and how Paulis are inserted at 2Q gates (on one qubit or on both).
- **Estimator:** ŷ = Σ_{j=1}^N c_j x_j, with x_j the noisy ⟨C̃_j⟩ (Eq. 3). **There is no intercept.** In ZNE the combine step is an extrapolation, and in PEC it is a quasi-probability-weighted sum. NIL generalizes both.
- **Training circuits:** Target circuits are Pauli rotations R_P(θ) plus fixed Cliffords.
  - Algorithm 2, "Strategy for constructing training circuits with quantum rotation 2-design", draws each angle uniformly from {0, π/2, π, 3π/2}. This is a "quantum rotation 2-design" (Def. 3, Thm. 4, App. B.1), **not a unitary 2-design**.
  - Algorithm 1, the baseline, replaces each rotation with a random element of all 24 single-qubit Cliffords.
  - The training circuits remain Clifford and are simulated with stim.
  - Assumption (App. A.1): "the noise models of all quantum gates remain unchanged when θ_i's are varied."
- **Regression:**
  - Loss: MSE (Eq. 5). OLS (Eq. 6) via sklearn.
  - Lasso as the ℓ1-**constrained** form Σ_j|c_j| ≤ γ (Eqs. 1/7), with γ = 5 for ZNE neighbors and γ = 2 otherwise, solved with CVXPY and MOSEK.
  - Finite shots add a regularizer Λ = diag(X₁,…,X_N)/N_s (Thm. 7, App. B.2.2).
- **Guarantees:**
  - Theorem 1 (informal; formal Thms. 5–6): as T → ∞, the 2-design-trained linear (or Lasso) model converges to the optimal linear estimator for the target circuits.
  - Theorem 2 / Sec. 7: "O(ln(N/δ)/ε²) training circuits" suffice for MSE within ε of the optimum with confidence 1 − δ, where N is the number of neighbors.
  - Shots per neighbor: O(1/ε²).
  - **How γ enters the constant, and whether |x_j| ≤ 1 is assumed, are in App. C, which could not be read.**
  - No explicit Pauli or Markovian noise assumption is stated beyond the angle-independence above.
- **Settings used (no general recommendation is given):**
  - VQE benchmarks: 5,000 training and 1,000 test circuits.
  - 20-qubit and 100-qubit cases: 1,000 training circuits and 10,000 shots per neighbor. The 20-qubit case used all weight-1 neighbors and the 100-qubit case at most 200 random ones. F₂ used half of the weight-1 neighbors.
  - LiH: 4⁴ = 256 2-design training circuits, against 24⁴ for Clifford training.
  - Noise: local depolarizing at 0.001 (1Q) and 0.01 (2Q).
  - **The results are simulations only.** Superconducting hardware appears only in a cost estimate.
  - The Table 1 caption claims "two orders of magnitude" over ZNE. The ratios shown are about 60× to 600×.

### 4. Cai et al., "Quantum error mitigation" (RMP review)

- **Bibliographic:** Zhenyu Cai, Ryan Babbush, Simon C. Benjamin, Suguru Endo, William J. Huggins, Ying Li, Jarrod R. McClean, Thomas E. O'Brien. arXiv:2210.00921v3 (28 Dec 2023; first submitted 3 Oct 2022). Rev. Mod. Phys. 95, 045005 (published 13 Dec 2023 per ORA). https://arxiv.org/abs/2210.00921, https://arxiv.org/pdf/2210.00921v3, https://ora.ox.ac.uk/objects/uuid:c20b81bd-e170-42d8-aaab-0be1967df398.
- **Bias, variance and MSE (Sec. II.B):**
  - Eq. 1: MSE[Ô] = E[(Ô − Tr[Oρ₀])²] = Bias² + Var.
  - Eq. 2 gives the variance of the noisy sample mean, (Tr[O²ρ] − Tr[Oρ]²)/N_cir.
  - "different ways of performing error mitigation often lead to different trade-offs between bias and variance".
- **Sampling overhead (Sec. II.B):**
  - Eq. 3: C_em = N_shot^ε(Ô_em)/N_shot^ε(Ô_ρ) = Var[Ô_em]/Var[Ô_ρ].
  - Eq. 5: C_em ~ R[Ô_em]²/R[Ô_ρ]², by estimator range.
  - Sec. II.A: "One could increase N_cir to compensate but this cost should be acknowledged". Also: "the cost is the sampling overhead of that error-mitigation method versus the ideal noiseless case."
- **Fault rate (Sec. II.C):**
  - Eq. 7: λ = Σ_f p_f, "the circuit fault rate".
  - Eq. 8: P₀ = e^{−λ}, from modelling the number of faults as Poisson. This applies with many fault locations and λ ~ 1.
  - Sec. II.D: post-selecting fault-free runs costs "C_em = e^λ", for example about 150 at λ = 5 and about 10⁴ at λ = 9.
  - Sec. III.E: Tr[ρ²] ≲ e^{−2λ} (purity-based methods).
- **ZNE (Sec. III.A):**
  - Noise must be amplifiable. Pulse stretching works "under the assumption that the noise was time-invariant".
  - "theoretically Richardson extrapolation will only be valid for small λ".
  - Gate insertion "may change the gate error model … such that the factor of increase … is no longer well-defined".
- **PEC (Sec. III.B):**
  - Eq. 14: C_em ~ Q² = (Σ_n|α_n|)².
  - Eq. 15: U = U_p/(1−p) − pN/(1−p).
  - Eq. 16: ((1+p)/(1−p))².
  - Eq. 18: "if we assume all the gates suffer from Pauli noise with the same error rate p", "the circuit size M is large and the circuit fault rate λ = Mp is finite", then C_em ≈ e^{4Mp} = e^{4λ}.
  - Noise must be "learned … through some form of tomography", which is efficient only for local gates. Pauli noise can be obtained via twirling.
- **Learning-based methods (Sec. III.H, read in the ORA publisher PDF up to mid-section):**
  - The training circuit should be "similar to P, usually in terms of circuit structures, such that it contains circuit faults similar to P", and "classically simulable".
  - The learned mitigation then "also works well for the primary circuit P due to their similarity".
  - Loss (Eq. 47): mean squared error over the training set. Linear least squares applies when the estimate is linear in θ.
  - Eq. 48: E₀(A) ≈ θ₀ + θ₁E(A) (Czarnik et al.), "obtained by training using the Clifford variants of the primary circuit". Linearity "can naturally arise when we assume that all noise sources in the circuit are globally depolarizing", which is "not a necessary assumption".
- **Not read:** the rest of Sec. III.H and all of Sec. IV ("Comparisons and Combinations": IV.A.1 noise calibration overhead, IV.A.2 mean square errors). Both lie beyond the fetch limit in every official copy tried (arXiv PDF and HTML, ar5iv, ORA). **Nothing here should be attributed to Sec. IV without reading it.**

### 5. Rudinger et al., "Probing context-dependent errors in quantum processors"

- **Bibliographic:** Kenneth Rudinger, Timothy Proctor, Dylan Langharst, Mohan Sarovar, Kevin Young, Robin Blume-Kohout. arXiv:1810.05651v1 (12 Oct 2018). Phys. Rev. X 9, 021045 (2019). https://arxiv.org/abs/1810.05651, https://arxiv.org/pdf/1810.05651v1.
- **Null hypothesis (Sec. II.A):** "context independence holds iff all of the data were drawn from the same underlying probability distribution".
  - A context is a different pool of counts from the **same** circuit set, for example a time period or a neighbor qubit idle versus driven.
  - Each context is a coarse-graining over unrecorded contexts.
- **Test:**
  - Statistic: the per-circuit log-likelihood ratio λ = −2Σ_m[x_m log(x_m/N) − Σ_c x_{c,m} log(x_{c,m}/N_c)] (Eq. 4).
  - Reference distribution: χ² with k = (C−1)(M−1) degrees of freedom by Wilks' theorem (Eq. 5), so p ≈ 1 − F_k(λ) (Eq. 6). This requires N_c ≫ 1.
  - Multiple circuits: "we can implement the ICTs with a Hochberg correction", with pseudo-threshold α/(Q − l_max + 1) (Eq. 9).
  - Aggregate test: λ_agg = Σ_q λ_q with k_agg = Q(C−1)(M−1) (Eqs. 11–12), reported as N_σ (Eq. 13). It is combined with the per-circuit tests by the closed-test principle at α = 5 %. A Bonferroni split covers multiple context comparisons ("a Bonferroni correction was used to split this 5% evenly over the comparisons for the seven rungs").
- **Effect size:**
  - JSD_q = λ_q/(2N_q) (Eq. 15).
  - TVD (Eq. 19) and SSTVD (Eq. 20), which is reported only when significant, and the max SSTVD (Eq. 21).
  - "Failure to detect context dependence does not imply that this circuit is probably context independent."
- **Results:**
  - Simulated drift: 1,405 LSGST circuits, N_σ ≈ 21 against a threshold of about 2.9.
  - ibmqx3, GST circuits on a pair, comparing a neighbor rung idle (a), driven by a CNOT (b), and idle again (c): no drift between (a) and (c). Crosstalk appeared in (a) versus (b) and decreased with distance from Q15.
  - "we observed a max SSTVD of around 28%" for the Q3–Q14 CNOT (sequence HSSSSH).
  - Confound: context (b) circuits were longer.
  - The method "can't be applied directly to SRB data", because SRB sequences are independently sampled. The tools are in pyGSTi.

### 6. Harris, Lively, Schuhmacher, "Reducing Quantum Error Mitigation Bias Using Verifiable Benchmark Circuits" (preprint)

- **Bibliographic:** arXiv:2603.10224v1 (10 Mar 2026). https://arxiv.org/abs/2603.10224, https://arxiv.org/html/2603.10224v1, https://arxiv.org/pdf/2603.10224v1.
- **Definition (Sec. II):** B(A) satisfies three conditions.
  - C1: A and B transpile to "the same sequence of native gates and measurements, up to replacing existing gates with new equivalent-error gates".
  - C2: B's outcome is "a single bitstring which can be efficiently classically determined".
  - C3: ⟨O⟩_B^exact = 1 (or −1, fixed by a classical bit flip).
  - Inputs: Pauli observables, start in |0…0⟩, no mid-circuit measurement.
- **Generation (Sec. III):**
  - III A, hardware-agnostic: each Pauli rotation is replaced by a random Pauli rotation (angles π/2, π or 3π/2 for weight 1, and π for weight 2). A final layer of 1Q rotations maps the known product state to ⟨O⟩ = 1. B is Clifford, and the 1Q-gate overhead is about 4 R_X per 1Q rotation and 14 per 2Q rotation (App. C). This assumes gates are "subjected to very similar errors with weak angle-dependence".
  - III B, IBM Heron basis {CZ, R_Z, X, √X}: "simply replace √X gates with X gates". There is zero gate overhead, and B need not be Clifford.
- **Bias reduction (Sec. II):**
  - b-mit: ⟨O⟩_A^b-mit ≡ ⟨O⟩_A^QEM / ⟨O⟩_B^QEM (Eq. 5). This leaves bias (b_A − b_B) (Eq. 7) and requires |b_A − b_B| ≤ |b_A| (Eq. 9), which the authors say "still needs to be justified".
  - For ZNE under uniform 2Q depolarizing noise (Eq. 10), the bias terms are argued to be "similar in magnitude".
  - bnZNE procedure:
    1. Build B.
    2. Run A and B at each noise level r.
    3. Compute ε(r) ≡ Π_q (1 − p(b_q; q, r)) (Eq. 18), "the product of the probabilities of obtaining the wrong measurement outcome across each of the measured qubits".
    4. Extrapolate (ε(r), ⟨O(r)⟩_A) to ε = 0.
  - Linear or exponential fit, "the best-performing option with respect to the benchmark circuits" (App. A).
- **Cost:** "for a given error mitigation method requiring N circuit runs, the cost of the bias-mitigated method is simply 2N". The experiments used "precisely twice as many circuit runs as for regular ZNE".
- **Experiments (Sec. IV):**
  - Simulation: 10-qubit kicked Ising and Heisenberg models with depolarizing noise, CZ folding at r = 1, 3, 5, and 5 twirls per level.
  - Hardware: ibm_fez, a 100-qubit 1D kicked Ising chain with 5, 10 and 20 Trotter layers (495, 990 and 1,980 CZ gates). 2,048 shots per instance, 32 Pauli twirls per circuit per ZNE level, DD and TREX. MPS references from ITensors.jl.
  - Abstract: "up to 15%" fidelity improvement over standard QEM. App. E: bnZNE at 20 layers reaches equivalent fidelity above 0.9 against about 0.65 for IC-ZNE.
  - Sec. IV B: bias mitigation "tends to reduce the accuracy of bnZNE despite often improving" RMSE.
- **Limitations (Sec. V):**
  - Unentangling benchmarks "do not introduce entanglement", so they may not "fully capture the effect of crosstalk noise".
  - "the bias is largely reduced but not entirely eliminated".
  - The analytic result covers uniform depolarizing noise only.

### 7. IBM Quantum documentation: "Workload usage" (URL `https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time`)

- **Page title** is "Workload usage | IBM Quantum Documentation". No last-updated date is shown. Accessed 2026-10-17.
- **Definition of usage:**
  - Usage "is determined by the amount of time a QPU is locked to execute workloads."
  - "Session usage is measured as the elapsed time while the session remains active". QPU capacity is reserved even when idle.
  - "Batch usage is measured as the cumulative time the QPU is locked to execute all jobs in the batch."
  - "Single job usage is measured as the time the QPU is locked to execute the job."
- **Failed and cancelled jobs:**
  - Job or batch mode: "If the failure or cancellation occurred because of a system error, the reported usage is zero". For user errors or cancellations, "the reported usage is any consumption that has occurred up to that point" (includes QPU preparation overhead).
  - Session mode: "The reported usage is the wall-clock time the session is active".
- **Formula:**
  - `<per sub-job overhead> + (rep_delay + <circuit length>) * <num executions>`. The overhead is "approximately 2s per sub-job", and large jobs may be split into sub-jobs.
  - rep_delay "is 250 microseconds on most IBM Quantum backends" (backend.default_rep_delay).
  - num executions = circuits × shots after PUB broadcasting. Mitigation adds circuits, and PEA and PEC add noise-learning overhead. Estimator groups qubit-wise commuting observables.
  - Quick version: "you can use `2+0.00035*<num executions>` as a quick formula", without advanced mitigation or a custom rep_delay.
  - It is "an approximation". The `scheduler_timing` total "is NOT the time used for billing".
- **Classical time:** not separately addressed. Usage is defined as QPU lock time, and the estimate includes the payload-loading overhead.
- **Querying usage:**
  - qiskit-ibm-runtime ≥ 0.30: `job.usage()`, `batch.usage()`, `session.usage()`. REST: `GET /jobs/{id}` and `GET /sessions/{id}`.
  - The Instances page shows a rolling 28-day window in real time. The Analytics page recalculates hourly over the last 28 full days, and the two can differ.

### 8. IBM Quantum documentation: "Maximum execution time" and plan/mode pages

- **Max-execution-time page** (`https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time`), no last-updated date shown:
  - "The service enforces a maximum execution time limit of three hours to ensure fair device usage." The page does not say whether this varies by plan.
  - The effective limit is the smaller of `max_execution_time` and the three hours.
  - "The `max_execution_time` value is based on QPU usage (quantum usage only), not wall clock time." The example sets `estimator.options.max_execution_time = 2500`.
  - "If a job exceeds this time limit, it is forcibly canceled and a `RuntimeJobMaxTimeoutError` exception is raised."
  - **The page does not say whether the consumed usage of such a job is billed.** The usage page's rule for cancellations not caused by a system error would report consumption up to cancellation, but this case is not explicitly classified.
  - Batch and session: a max TTL is assigned at start. When it ends, running jobs finish and queued jobs fail. The interactive TTL "cannot be configured".
  - "On the Open Plan, you can use up to 10 minutes on a QPU per 28-day rolling window."
  - "The session or batch is immediately terminated when the cost limit is reached." Cost limits apply to instances; the usage page mentions them for Pay-As-You-Go.
- **EstimatorOptions API** (`https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/options-estimator-options`; source links point to the stable/0.50 branch): `max_execution_time`, "Type: `UnsetType` | `int`", "Default value: `Unset`".
- **Run jobs in a batch** (`.../guides/run-jobs-batch`):
  - Default maximum TTL: "All paid plans | 8 hours" and "Open | 10 minutes".
  - Interactive TTL: "1 minute for all plans".
  - Set with `Batch(backend=…, max_time="25m")`, which "should exceed the longest job's execution time". "This timer starts when the batch starts", and the batch "starts when its first job begins execution".
  - **The page does not state whether the TTL can be raised on the Open plan, or explicitly that it is wall-clock.** It is described as a timer, not as usage.
- **Run jobs in a session** (`.../guides/run-jobs-session`):
  - "Open Plan users cannot submit session jobs".
  - The Premium default max TTL is 8 h; some instances may differ. Interactive TTL is 60 s. Set with `Session(max_time=…)`.
- **Execution modes** (`.../guides/execution-modes`): "There are three execution modes: job, session, and batch". "Open Plan users cannot submit session jobs." So **Open Plan has job and batch mode**; batch availability is inferred from the batch TTL table, which lists the Open plan.
- **Choose the right execution mode:**
  - "Generally, use batch mode unless you have workloads that don't have all inputs ready at the outset."
  - Batch is "generally less expensive than sessions".
- **Plans overview** (`.../guides/plans-overview`):
  - Open Plan: "up to 10 minutes per 28-day rolling window"; "Open Plan instances can only be created in the us-east region"; "As of 16 March 2026, active Open Plan users can opt in to an additional 180 minutes over the next 12 months."
  - Pay-As-You-Go is billed per QPU time. Flex starts at ≥ 400 min per year. Premium and On-Prem are by contract.
- **IBM blog** "Doubling down on open-access quantum computing" (16 Mar 2026, https://www.ibm.com/quantum/blog/open-plan-updates):
  - Base: "up to 10 minutes of quantum runtime every 28 days". The same post elsewhere also says "10 minutes/month".
  - Promotion: "once you use 20 minutes of compute within any 12 month period, you'll be able to opt in to the new 180-minute promotion". Non-trial account required. It reverts to 10 min after 12 months.
- **Campaign-wide cap:**
  - Nothing in these pages provides a user-set cap on total usage across many jobs, other than instance cost limits on paid plans and the plan allowance itself.
  - `max_execution_time` is per job.
  - Batch and session `max_time` is a TTL timer that fails queued jobs; it is not a usage budget.

### 9. Lubinski et al., "Application-Oriented Performance Benchmarks for Quantum Computing"

- **Bibliographic:** Thomas Lubinski, Sonika Johri, Paul Varosy, Jeremiah Coleman, Luning Zhao, Jason Necaise, Charles H. Baldwin, Karl Mayer, Timothy Proctor. arXiv:2110.03137v3 (10 Jan 2023; first submitted 7 Oct 2021). IEEE Trans. Quantum Eng. 4 (2023), DOI 10.1109/TQE.2023.3253761. https://arxiv.org/abs/2110.03137, https://arxiv.org/pdf/2110.03137v3.
- **Sec. III.G** "Quantifying Result Quality with Circuit Fidelity":
  - Eq. 1: F_s(P_ideal, P_output) = (Σ_x √(P_output(x) P_ideal(x)))².
  - Eq. 2: F(P_ideal, P_output) = max{F_raw(P_ideal, P_output), 0}.
  - Eq. 3: F_raw = (F_s(P_ideal, P_output) − F_s(P_ideal, P_uni)) / (1 − F_s(P_ideal, P_uni)).
  - Rationale: "F_s is non-zero even when the output of the circuit is completely random". The normalization makes F(P_ideal, P_uni) = 0.
  - They call F_s the classical fidelity, "related to the classical Hellinger distance".

### 10. Czarnik et al., "Error mitigation with Clifford quantum-circuit data" (CDR)

- **Bibliographic:** Piotr Czarnik, Andrew Arrasmith, Patrick J. Coles, Lukasz Cincio. arXiv:2005.10189v3 (16 Nov 2021; first submitted 20 May 2020). Quantum 5, 592 (2021). https://arxiv.org/abs/2005.10189, https://arxiv.org/pdf/2005.10189v3.
- **Training set:** T_ψ = {X^noisy_φi, X^exact_φi} (unnumbered).
- **Ansatz and fit:**
  - Eq. 2: f(X^noisy_ψ, a) = a₁X^noisy_ψ + a₂.
  - Eq. 3: least squares C = Σ_{φi∈S_ψ}(X^exact_φi − (a₁X^noisy_φi + a₂))².
  - The fitted ansatz is then applied to X^noisy_ψ; there is no separate numbered "X^cor" equation.
- **Training circuits:**
  - "replacing a subset of the gates in the circuit that prepares |ψ⟩ with Clifford gates that are close in distance", "a near-Clifford circuit with N non-Clifford gates" (App. B).
  - Replacements are powers of S = R_Z(π/2), weighted w(n) = e^{−d²/σ²} with σ = 1/2 (Eq. 14).
  - Sampling uses MCMC. For QAOA, the likelihood (Eq. 15) favors exact energies near a guess; a monotonic alternative is Eq. 16. For phase estimation, Eq. 17 favors noisy values near the target.
  - "60−80 training circuits and 8192 shots per expectation value estimation are enough". The figures use 70 circuits, and 63 on hardware.
  - "Increasing the number of nonClifford gates N in the training data monotonically reduces the error". Classical cost grows exponentially with N, and N ≤ 30 was used.
- **Noise:**
  - "global depolarizing noise leads precisely to a linear relationship" (App. A). "we do not expect the linear ansatz to correct for all types of noise".
  - The simulations use an Ourense GST noise model, which "does not capture spatial and time inhomogeneous aspects", and measurement noise is omitted.
- **Results:** On IBM Almaden, a 16-qubit QAOA, the mean relative error was "0.409 for the noisy estimates and 0.028 for the corrected ones". ZNE gave no "meaningful correction" in that setting.

### 11. Liao et al., "Machine Learning for Practical Quantum Error Mitigation" (ML-QEM)

- **Bibliographic:** Haoran Liao, Derek S. Wang, Iskandar Sitdikov, Ciro Salcedo, Alireza Seif, Zlatko K. Minev. arXiv:2309.17368v2 (22 Nov 2024; first submitted 29 Sep 2023). Nature Machine Intelligence, DOI 10.1038/s42256-024-00927-2. https://arxiv.org/abs/2309.17368, https://arxiv.org/pdf/2309.17368v2, https://research.ibm.com/publications/machine-learning-for-practical-quantum-error-mitigation--1.
- **Features:**
  - OLS (Sec. IV A 1): "counts of each native gate on the backend (native parameterized gates are counted in binned angles), the Pauli observable in sparse Pauli operator representation, and optional device-specific noise parameters". The model "takes in these circuit-level features as well as the noisy expectation values" and "predicts the ideal expectation values".
  - RF and MLP: "identical" features.
  - GNN: the transpiled-circuit DAG, with node vectors for gate type and gate errors plus the qubits' T1/T2 and readout errors. The noisy expectation values are concatenated before the dense layers.
  - Crosstalk and idle duration were "found not helpful".
  - **Not stated:** whether the noisy values of several observables (for example all weight-one Z_i) enter one model jointly. App. A 2 has "100 tree estimators for each observable", which suggests per-observable models. Bitstring counts are not features.
- **Training data:**
  - 4-qubit random circuits (500 train and 200 test per depth).
  - Trotterized 1D TFIM, measured "in a randomly chosen Pauli basis for all the weight-one observables" (300 + 300 per step).
  - ibm_algiers (50 train and 250 test per step).
  - VQE on H₂.
  - 100-qubit "mimicry" on ibm_brisbane, with ZNE + TREX values as labels.
  - Labels come from ideal simulation, except in mimicry. **Clifford circuits are not training data.** CDR "focuses on a single circuit and a single expectation value", whereas ML-QEM targets "a large class of circuits".
- **Results:**
  - The random forest is best ("RF consistently outperforms the other ML-QEM models").
  - On ibm_algiers, it has about 40 % lower overall and 50 % lower runtime overhead than ZNE.
  - Under coherent noise, "none of the models demonstrate effective mitigation" in extrapolation.
  - Mimicry "does not allow the ML-QEM model to outperform the mimicked QEM method".
  - No predictive uncertainty is given. Training cost is accounted for separately (App. B).

---

## Conflicts and caveats

1. **Chen et al. do not normalize fidelity.** Their F_c (Eq. 1) is the plain classical (Bhattacharyya-squared, = (1−H²)²) fidelity. The text explicitly omits Lubinski's uniform-baseline normalization (Lubinski Eqs. 2–3), arguing it is "insignificant for wide circuits with concentrated output distributions". For **narrow circuits or broad ideal distributions** (context probes, brickwork, Ising) the two metrics differ substantially. Any figure must state which one it shows, and the effective rate fitted from log F_c is not the one fitted from log F_normalized.
2. **"RB error = depolarizing parameter" is false as stated.** Chen et al.'s ε (Eq. 2) is the total Pauli-error probability, which equals the **entanglement infidelity**. Their DRB rate equals the entanglement infidelity only "where the noise is known or restricted to be purely stochastic". For a d-dimensional gate with entanglement infidelity ε, the following are standard identities, not taken from the papers:
   - average gate infidelity r = ε·d/(d+1);
   - "replace-with-I/d" depolarizing parameter λ = ε·d²/(d²−1) = r·d/(d−1).
   - For 1Q (d = 2): ε = 1.5r and λ = 2r. For 2Q (d = 4): ε = 1.25r and λ = 4r/3.
   - IBM's reported gate errors are RB-type average infidelities r, so feeding r directly as ε or λ underestimates the modelled noise. The project's stated conversions (r = ε·d/(d+1); PEC p = r·d/(d−1)) are consistent with these identities.
   - Cai et al.'s e^{4λ} (Eq. 18) uses the **Pauli-error probability** p per gate, so λ_LC built from r must be rescaled before comparing β with 4.
3. **The Chen et al. component model includes no SPAM.** SPAM is 0.5 % per qubit (Sec. 2), but it is not in the Sec. 4 model. The over-prediction (observed rate ≈ 100×10⁻⁴ against the model's ≈ 46.4×10⁻⁴) is attributed to out-of-model, non-Markovian spin-phase noise, not to SPAM. A "chen" predictor with readout added is no longer Chen's model.
4. **Plurality voting is error mitigation, applied only to #AQ.** Its App. B formula (Eqs. 4–5) is exact only for t ≥ N_v/2, and with t = 7 of 25 it is an approximation.
   - It is unclear whether the frequencies f_vb are renormalized after rare bitstrings are dropped ("relative to this reduced collection"). The final p_b **are** normalized.
   - The project's choice (raw frequencies, follows the authors' reference implementation) is not contradicted by the paper, but it is not stated there either. Cite the code, not the paper, for that detail.
   - Eq. 6's G_b^(m) is undefined, a typo for p_b^(m).
   - The Fig. 11 bin factor is not given in the caption, so do not cite "log₂ bins" or "√2 bins" as Chen's.
5. **NIL as published versus the app.**
   - No intercept. ℓ1-constrained (not penalized) least squares with γ = 2 or 5.
   - Rotation 2-design: angles uniform in {0, π/2, π, 3π/2}, which is not the 24-Clifford substitution or a unitary 2-design.
   - Simulation only, under local depolarizing noise.
   - The paper does not state that the bare circuit is a neighbor, or how Paulis are placed at 2Q gates. These are app choices.
   - The weight-1 count is inconsistent in the paper (O(m²) versus the number of gates).
   - Theorem 2 is an **upper bound** O(ln(N/δ)/ε²) on training size for being within ε of the *optimal linear estimator*. It says nothing about the optimal linear estimator's own bias floor. Plateaus in error versus T are compatible with it but not predicted by it.
6. **The project's FR is neither NIL nor Strikis et al. nor ML-QEM.**
   - ML-QEM features are gate counts, the observable encoding, optional device noise parameters and noisy expectation values, trained across a circuit class on non-Clifford random or Trotter circuits.
   - FR (noisy ⟨O⟩, ⟨Z_i⟩, ⟨Z_iZ_j⟩ on per-target near-Clifford variants) is closer to "CDR with extra regressors".
   - The release notes for v1.4 already say this. The literature check supports that naming.
7. **max_execution_time does not bound total usage.**
   - It is a per-job QPU-usage limit (capped at 3 h). Its API default is `Unset`.
   - Batch and session `max_time` are TTL timers that fail queued jobs; they are not usage budgets.
   - The only caps across many jobs are instance cost limits (paid plans) and the Open Plan allowance (10 min per 28-day rolling window).
   - Whether a job cancelled by max_execution_time is billed is **not stated**. Assume it is, since user-side cancellations report usage up to that point, and confirm with `job.usage()`.
8. **Open Plan specifics that affect the run plan.**
   - No sessions. Batch is available, with a **default batch max TTL of 10 minutes** on the Open plan and an interactive TTL of 1 minute.
   - The batch timer starts when its first job begins execution and runs as a timer, not as usage.
   - A batch planned at about 410 to 530 s of QPU usage can therefore run into the 10-min TTL if jobs queue behind other users' work or idle more than 1 min between jobs. Queued jobs then fail.
   - The docs read here do not show whether the Open plan can raise the TTL. Splitting into smaller batches, or keeping each job self-contained, is the safe reading.
   - The quick usage formula (0.35 ms per execution + 2 s) underestimates the project's measured ≈ 0.57 ms per shot on ibm_fez.
9. **Featuremetric predictions are in-distribution.** Proctor et al. explicitly do not claim that their features suffice outside the training circuit distribution, or for extrapolation. Use them to justify feature-based interpolation, not cross-family prediction.
10. **Harris et al. bias reduction** rests on uniform 2Q-depolarizing analysis plus an unproven inequality (Eq. 9). The IBM variant (√X → X) yields product-state benchmarks that may miss crosstalk. The cost is 2× runs.
11. **Rudinger et al. compare outcome distributions of identical circuits across contexts** (LLR/χ², Hochberg, SSTVD). Feature-regression residuals are a different and weaker kind of evidence for context dependence. A null result is not evidence of independence.

## Sources of the cited circuit presets (`qemx/presets_cited.py`)

Read on 2026-10-17 with WebFetch (arXiv PDFs and abstract pages). Some equation glyphs came through a PDF-to-text
summarizer garbled. Those places are flagged below.

| Preset | Source and locator | What the source supports | What the app adds or changes |
|---|---|---|---|
| `qft-known-answer` (adapted) | Lubinski et al., arXiv:2110.03137v3, App. A.2 and Fig. 26 (benchmark **QFT(2)**). The paper's own labels are QFT(1)/QFT(2), not "method 1/2". | QFT(2) prepares the Fourier-basis state of x with "a series of Hadamard gates and Z rotations", then runs "only a standard circuit for the inverse QFT". The expected result is the bit string x. | The mirror image: F†\|s⟩ is prepared and F applied. QFT(1) (add 1 in the Fourier basis) is not implemented. |
| `aqft-coppersmith` (adapted) | Coppersmith, IBM RC 19642 (1994), arXiv:quant-ph/0201067v1. The sections are unnumbered; see "Approximate Fourier Transform". | "we simply delete those matrices Q_JK with K ≥ J + m". With m = 1 this is the Hadamard transform. Each entry carries a phase error factor exp(iε), \|ε\| ≤ 2πL·2^(−m). | Index shift: the app's degree m keeps distances d ≤ m, so app m = m_C − 1. Coppersmith gives no "m ≈ log₂ n" rule. The exponent glyph of Q_JK was garbled in extraction; the reading exp(iπ/2^(K−J)) is consistent with the text. |
| `qaoa-ring-p1` (exact reproduction) | Farhi, Goldstone, Gutmann, arXiv:1411.4028v1, Sec. IV ("The Ring of Disagrees"). Wang, Hadfield, Jiang, Rieffel, arXiv:1706.02998v2 (PRA 97, 022304), Sec. III. | Farhi: for the ring at p = 1 the maximum is 3/4, found by numerical maximization; **no angles are stated**. Wang et al.: "Equation (16) yields the approximation ratio 0.75 at (β,γ)=(π/8,π/4)". Their conventions: C_uv = ½(I − Z_uZ_v), U_C = e^{−iγH_C}, U_B = e^{−iβH_B}, H_B = ΣX. | The app uses the same convention, checked by `tests/test_presets_cited.py`: the ratio is 0.75 for even rings n = 4, 6, 8. Wang et al.'s Sec. IV uses a rescaled convention; it is not used here. |
| `qaoa-3reg-p1` (adapted) | Farhi et al., arXiv:1411.4028v1, Sec. V. | "for p = 1, the worst case approximation ratio that the quantum algorithm produces is 0.6924" over 3-regular graphs. | One graph family only (the Möbius ladder), with optimized angles. The test checks ratio ≥ 0.6924. |
| `hea-depth-2` (educational) | Kandala et al., arXiv:1704.05018v2 (Nature 549, 242), Eq. (2) and Fig. 1c–d. | "applying d entanglers U_ENT that alternate with N Euler rotations". Each Euler rotation is Z·X·Z, and the entanglers come from cross-resonance drift. p = N(3d + 2) angles in the main text; the Supplement says N(3d − 1), an inconsistency. | The app's ansatz uses RY·RZ layers and CX entanglers. Only the layered structure is shared. |
| `ising-trotter-1st` (educational) | Childs & Su, arXiv:1901.00564v2 (PRL 123, 050503), main-text Eq. (6). Childs, Su, Tran, Wiebe, Zhu, arXiv:1912.08854v3 (PRX 11, 011020), Sec. 3.1. | Eq. (6): ‖S₁(t) − e^{−itH}‖ ≤ ∫₀ᵗ∫₀^{τ₁} ‖[H_even, H_odd]‖. Evaluating the integral gives (t²/2)‖[·,·]‖; that evaluation is ours. Sec. 3.1: first-order error O(‖[B, A]‖t²). | The app splits H into its ZZ and X parts. The commutator norm is bounded with the triangle inequality (4(n − 1)\|Jh\|), so the bound is loose. The test checks that the actual error stays below it. Lloyd (1996) was not read. |

Not verified for the presets: the QED-C repository's "method" naming; any explicit t²/2 constant in Childs et al.
1912.08854 Sec. 5 or its appendices; journal volume/page details beyond the DOIs.

## Could not verify

- Cai et al. Sec. III.H (second half) and **all of Sec. IV** ("Comparison among QEM methods": noise calibration overhead, MSE comparison). The fetch was truncated in every official copy (arXiv PDF v3 and HTML v3, ar5iv, ORA publisher PDF). Whether the review contains a summary comparison table is unknown.
- NIL App. C: the explicit constant of Theorem 2, how γ enters it, and boundedness assumptions. Also NIL's reference list, so citation [16] is unconfirmed (the text calls it "learning-based PEC").
- Proctor et al. App. B–C: the GP kernel and hyperparameters.
- Liao et al.'s Nat. Mach. Intell. volume and page (6, 1478 in project docs). nature.com could not be fetched; only the DOI and the IBM Research page (22 Nov 2024) were confirmed.
- IBM: whether the Open-plan batch max TTL can be set above 10 min; whether a max_execution_time cancellation is billed; and whether the three-hour job cap varies by plan.
