# QEM Explorer 1.2 — release notes

QEM Explorer 1.2 makes every comparison an exact control experiment: all methods spend the same total shot budget,
every cost is counted the way it is paid, and the simulator and IBM hardware produce the same results suite. It adds
Grover's search and the hidden subgroup problem, diagrams for every algorithm, and a re-verified implementation of
every algorithm and estimator. All presets and example values remain illustrative.

## Equal budgets and honest costs

* **One total budget per method.** Every contender spends the sidebar's budget B on target, training and
  calibration circuits together, B = N_t + F + c·s. The calibration shots are chosen optimally,
  s* = Rκ/(1 + cκ) with κ = √(w/(cv)), and the finite calibration's error enters the variance by the delta method
  (w/s). New module `qemx/budget.py`; the cost table, the RMSE curves, the repeated experiments, Cost vs precision,
  Head-to-head, Batch run and the bullseye all use it.
* **Training noise is counted.** CDR/NIL's training shots are part of the budget, and their shot noise propagates
  through the fit to a constant u (first order, delta method). When the simulated repeats show a clearly larger
  spread (NIL's near-collinear features), the excess is booked as u. A target precision below √(b² + u) is reported
  as unreachable.
* **Circuits are what is sent.** PEC counts its sampled circuits: one shot per circuit (the textbook protocol),
  so N_t circuits, or ⌈N_t/s⌉ with s shots per circuit (new parameter; the instance-to-instance variance
  (s − 1)·Var_inst is then added). Twirling and randomized compiling count one random circuit per shot; ZNE one per
  noise scale; CDR/NIL their training circuits; readout mitigation its calibration circuits.
* New table columns: *target shots*, *training shots*, *calibration shots*; *QPU s* = B·t_shot, the same for every
  method. A method whose training alone exceeds B says so instead of showing a number.

## Hardware parity

* **The same allocation on hardware.** The IBM Quantum page records each contender's circuits with their roles,
  predicts their outcomes with the device's calibrated error rates and allocates the budget exactly as the
  simulator's accounting does: ZNE's N_j ∝ |η_j|σ_j, a Hamiltonian's N_g ∝ √v_g, s* calibration shots, PEC's K
  sampled circuits with N_t/K shots each, the training shots. Contenders never share counts.
* **The same results suite.** The replay returns the simulator's result structure, so the page shows the
  Playground's analysis views (measured distributions, readout-corrected and ZNE-extrapolated), estimates, spread of
  repeated experiments, the equal-budget cost table, every method's internals, and the same ZIP plus the plan, the
  circuits sent and the job IDs. A *shot allocation* table shows how each contender spends its budget.

## Exports

* **No HTML files** in any ZIP. Figures are PNG (3×) and SVG; without a renderer each figure is saved as Plotly JSON.
* **Global export.** The Playground's (and the IBM page's) *Download all* computes every cross-page result of the
  comparison on the fly — `cost_precision/` (RMSE against budget, budget and QPU time for ε, crossovers),
  `pec_wall/`, `bias_variance/` — and adds the other pages' stored results (depth sweep, bullseye, head-to-head,
  model mismatch, batch run), the state and the Qiskit script. Tables carry CSV and LaTeX.

## Algorithms and diagrams

* **Grover's search** (any n ≥ 2, any marked item, optimal k* = ⌊π/(4θ)⌋ or custom iterations; the multi-controlled
  Z as CZ, a 6-CX Clifford+T circuit, or Qiskit's synthesis) and the **hidden subgroup problem** (Z₂ⁿ with a hidden
  subgroup of dimension 1 or 2; the cyclic group Z_N with N = 2ⁿ) in *FTQC · deep algorithms*, with Learn
  walkthroughs and Theory sections.
* **Diagrams** for every algorithm in Learn and Theory (Bell, GHZ, QFT, QPE, Ising, BV, HSP, Grover, VQE, QAOA, QML,
  the five codes, Shor, HHL, ZNE folding), typeset with quantikz (`scripts/make_figures.py`) and shown on a light
  card in both themes. Diagrams are sized by their text — labels render at about body-text size (×1.9, never below
  ×1.35; a diagram wider than the column scrolls instead of shrinking further) — and their captions are typeset by the
  app under the card (`docs/figures/captions.json`).
* **Theory & sources** redesigned for reading: five parts with a contents column, strict section hierarchy, wider
  line spacing and a narrower measure; new references (Grover 1996; Bennett et al. 1997; Boyer et al. 1998;
  Childs & van Dam 2010; Jozsa 2001), 100 in total.

## Interface

* **One Run button.** The sidebar's *Run simulation* runs the Playground and Cost vs precision; duplicate run
  buttons are gone. Pages with their own experiment (sweeps, mismatch curves) name what they run.
* **Inputs and bases in the composer.** Applying `ramp:1`, `plus`, `bits:0101` or a basis such as `XZY` to all
  qubits draws the preparation gates (e.g. H·P(7π/4)) and the basis changes (H, S†·H) as locked boxes on the wires.
* **π in angles.** Angle fields accept `pi`, `π`, `\pi`, `Pi`, `2pi`, `3π/4`, `\frac{\pi}{8}`, `2·π`, `−π/2`; angles
  are shown as fractions of π up to π/1024 (`π/128`).
* **Readable tooltips.** Math in every `help` popup (widgets, metrics, table columns) is rendered as Unicode
  (√(b² + v/N), γ = ∑ᵢ|qᵢ|) instead of raw TeX.

## Verification

Two independent code-level audits (Sonnet) compared every template and estimator with its formal definition.

* Confirmed correct, with numerical evidence: the QFT equals the DFT matrix; Grover's success probability equals
  sin²((2k+1)θ) for n = 2–5; HSP outcomes are uniform on H^⊥ (Z₂ⁿ) and on the multiples of N/2^j (Z_N); QPE, BV,
  Simon, Shor, HHL, VQE, QAOA and all five codes; Richardson and linear ZNE weights; PEC's quasi-probabilities
  invert the channel and γ matches its closed form; the readout inverses, M3 and TREX; the calibration gradient;
  symmetry verification's and virtual distillation's variances against brute-force sampling; the Hamiltonian split;
  the optimal calibration split.
* Fixed: M3 dropped negative entries of signed (PEC) data; PEC's variance on Aer ignored the readout-mitigated
  observable (up to +39 %) and estimated the instance variance from at most 32 instances; ZNE counted noiseless
  virtual RZ gates when a circuit has no two-qubit gates; the Lasso penalty was 2N times the documented one; the
  learning variance omitted the training data's shot noise; the training-set rank check ignored shot noise;
  symmetry verification read classical bits instead of qubits when the readout is relabeled; the budget model
  treated calibration as free when its gradient vanished; documentation drift (AQFT modes, QAOA ratio for odd
  rings, the QML sign convention, the repetition code's r + 1 exposures, the d = 2 surface-code checks).
* 327 tests, including 34 new theory-conformance, parity and UI tests (`tests/test_v12.py`).

---

# QEM Explorer 1.1 — release notes (previous)

QEM Explorer 1.1 turns the 1.0 laboratory into a research platform: it reads each result the way its mathematics
asks, shows what the compiler does to a circuit on real connectivity, runs whole studies with one button, and
ships a handbook and a learning path. All presets and example values remain illustrative.

## Analysis and visualization

* **Analysis panel with automatic views** (*Mitigate & compare*): an output-distribution histogram for probability
  observables (ideal, noisy, readout-corrected and ZNE-extrapolated, exact and marginalized onto the bits the
  observable reads; TVD and Hellinger fidelity), estimates ± σ with the spread of repeated experiments for expectation
  values, effective logical error and syndrome histograms for codes, and a grid of each method's diagnostic curve
  (ZNE fit, PEC quasi-probabilities and γ, learning fit, VD spectrum, acceptance, readout assignment errors). A
  **View** selector overrides the automatic choice; custom circuits say so.
* Collision-free circuit drawings (ZNE fold bands, PEC insertions, grouped operations) at any width.
* Bloch spheres wrap into rows for 5+ qubits; two-line titles from three spheres per row.
* LaTeX tooltips on every cost-table column and metric.

## Compilation and hardware realism

* **Logical ↔ physical switch** in the composer: each logical gate above the native gates that implement it
  (H → RZ·√X·RZ), routing SWAPs framed, and a rewrite table.
* **Topology & layout** tab: coupling graphs (heavy-hex devices drawn row by row), why each SWAP was inserted, and a
  mapomatic-style VF2 search for the lowest-error physical qubits with one-click recompilation.
* ISA compilation on **2D-grid** and **heavy-hex** lattices.
* **Crosstalk** noise: phase kicks and residual ZZ on idle neighbors of a driven gate; DD refocuses them.

## Observables and interactivity

* **Hamiltonian observables** `0.5*Z0*Z1 + 0.2*X0 - 0.1*Y1` (strict parser, qubit-wise commuting settings, exact
  results on every backend and in exported scripts).
* **Live θ sliders** for VQE and QAOA in *Time travel*, and 1D / γ × β energy-landscape scans.

## Studies, learning and theory

* **Sweeps in Batch run** — one line per swept quantity, every combination a scenario:
  `ramp: 0, 1, 3` (the phase-ramp input $\lvert{\rm ramp}{:}k\rangle=F^\dagger\lvert k\rangle$ on the whole register),
  `input: plus, bits:0101`, `basis: Z, X, ZZXY`, any template parameter (`m: 1, 2, 3`, `s: 0, 1`) and any noise
  parameter (`p2: 0.005, 0.01`). Each axis applies only where it means something (template parameters to templates
  that have them, noise parameters not to fake devices); a new chart plots the error against the swept value with
  everything else held fixed. Strict parser, safe expression reader, size guards.
* **Register-wide inputs and bases** in the Playground (*All qubits: input / basis → Apply to all*): `plus`,
  `ramp:k`, `bits:0101`, `X`, `ZZXY`. Phase inputs are stored as `ph:<turns>` per qubit, saved and exported like the
  other inputs.

* **Batch run** page: algorithms × methods × noise models × noise scales × simulators with one *Run all*; per-scenario
  failure isolation; heat map, winners, error against noise scale; included in the Download-all ZIP.
* **Learn** page: seven hands-on guides with one-click Playground setups, QC fundamentals for QEM, and walkthroughs of
  VQE, QAOA, Shor, HHL and QEC primitives.
* **QEC templates**: the five-qubit code [[5,1,3]] (Clifford encoder, lookup decoder, corrects any single-qubit error)
  and the d = 2 surface code (one plaquette; transversal or ancilla syndrome extraction).
* Symmetry verification checks several parities at once (e.g. every Z check of a code).
* **Theory handbook**: new sections on advanced mitigation (symmetry expansion, pulse-level ZNE and PEA, quantum
  subspace expansion, classical shadows, M3) and on the road to fault tolerance (stabilizer formalism, code taxonomy,
  decoding, threshold theorem, logical error mitigation); about 95 references with DOIs.

## Quality

* **Stability:** one process-wide lock around every entry into Qiskit's native code (compilation, circuit
  construction, Aer, drawing), and `runner.fastReruns = false`, so two runs of one session never overlap. A native
  crash cannot be caught by `try/except`; this prevents the concurrent use that can cause one.
* US English throughout the documentation, comments and interface; option values renamed in 1.1 (for
  example `VQE-optimized` → `VQE-optimized`) are still accepted from 1.0 state files.
* Two independent QA passes (Sonnet) on the batch runner, the responsive layout, the text parsers, the analysis
  engine, the new codes and the handbook; every confirmed finding fixed with a regression test.
* 293 tests (pytest, including AppTest end-to-end runs of every page and the exported script in a terminal and in a
  Jupyter kernel).
