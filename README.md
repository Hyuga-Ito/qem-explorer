# QEM Explorer 1.5.0

An interactive laboratory for quantum error mitigation (QEM): build a circuit, inject exactly the noise you
want (or a real IBM device's calibration), look at the circuits each method really runs, scrub through the
state gate by gate, compare any set of methods on bias, variance, cost and robustness, run the same
experiment on IBM Quantum hardware, and export a standalone Qiskit script that reproduces every number.

> Intuitive UI makes intuitive conceptual understanding.

All presets and example values are **illustrative**; none of them are measured results.

## Quick start

```
python -m venv .venv                      # Python 3.10–3.13
.venv\Scripts\activate                    # Windows   (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py            # then open http://localhost:8501
```

**VS Code**: open the folder, run *Python: Create Environment* (venv, tick `requirements-dev.txt`), then press
**F5** with "QEM Explorer (Streamlit app)" selected. `.vscode/` also holds test configurations and tasks.
If a newer package release breaks something, `requirements-lock.txt` pins the tested versions.

## What is new in 1.5

Read `docs/MIGRATION_1.5.md` first: some 1.4 numbers for learning methods change meaning. The details are in `docs/USER_GUIDE_1.5.md`, and what was changed, why, how it was tested and what remains open is in `docs/TRACEABILITY_1.5.md`.

* **Statistics you can rely on.** Learning methods report the moments of the protocol actually run, and the exact-training fit is kept as a reference (`qemx/stats.py`). An independent check is `python scripts/validate_learning_moments.py --out validation.json`.
* **Fixes.**
  * Correct readout overhead, including readouts that mostly flip.
  * Clear errors for singular or invalid channels.
  * Exports that never hang: images render in a separate, time-limited process.
  * A budget preflight ("insufficient_budget") before any fit.
  * Guides that replace or merge contenders, atomically.
* **Input and output.** The QFT's input preparation is an explicit input. The Playground's **Output** panel is read-only and derived. Parameter presets carry source, version and locator, and are searchable.
* **One experiment model.** The app's state, YAML files and the CLI share one validated schema:

  ```
  python -m qemx.experiment validate|plan|run examples/experiments/qft_known_answer.yaml --out results/qft
  ```

* **Research areas.** *QEM Performance & Selection* covers regime maps, paired comparisons and a baseline recommender with abstention. *Noise Models & Discrepancy* covers calibrated discrepancy tests and the emulated rehearsal. The four analysis pages are their drill-downs.
* **Main study.**
  * Feature dictionary: `docs/feature_dictionary.md`.
  * Space-filling designs and a confirmatory subset in the manifests.
  * Reports regenerated from stored data:

    ```
    python -m qemx.phase.run run manifests/direction_A_v2_pilot.yaml --out data/dirA_v2_pilot --workers 2
    python -m qemx.phase.run collect data/dirA_v2_pilot
    python -m qemx.study report data/dirA_v2_pilot --manifest manifests/direction_A_v2_pilot.yaml \
        --rehearsal reports/rehearsal_v3 --out reports/pilot
    ```

  The scalable study is `manifests/direction_A_v2_study.yaml`.
* **Discrepancy rehearsal without hardware** (labeled EMULATION):

  ```
  python scripts/rehearse_divergence.py manifests/direction_A_divergence_v3_qpu600.yaml --out data/reh_v3
  ```

* **Hardware within one shared 600-second campaign.**
  * A conservative usage model and ledger (`python -m qemx.campaign`) admit every job before it is submitted.
  * The plan is in `docs/hardware_preflight.md`.
  * Nothing is submitted without a ledger and an explicit `run-ibm`.

## Research backend: equal-budget phase diagrams (`qemx.phase`; v1.3, extended in v1.4)

A Streamlit-free pipeline for Direction A of the thesis plan: for every QEM method on every grid coordinate
(circuit family × n × noise scale × engine), it measures the primitives of the equal-budget ledger
B = N_t + T·s_tr + c·s, RMSE = √(b² + v/N_t + w/s + u). It also computes λ_LC, Monte-Carlo-validates the ledger
and writes versioned records. The design, the self-audit and the pre-registered manifest are in
`docs/v13/direction_A_design.md` (PDF alongside).

```
python -m qemx.phase.run plan       manifests/direction_A_v1.yaml
python -m qemx.phase.run run        manifests/direction_A_v1.yaml --out data/dirA_v1 --workers 8
python -m qemx.phase.run collect    data/dirA_v1
python scripts/dirA_checks.py       data/dirA_v1                  # acceptance checks → data/dirA_v1/checks
python scripts/dirA_figures.py      data/dirA_v1 --out figs        # phase maps, family maps, RMSE and B*(ε)
```

Neighbor-informed learning (v1.4) has its own study, `manifests/direction_A_nil_v1.yaml` (1 280 cells: the
coordinates of run 2 plus a NIL sensitivity stage), analyzed with
`python scripts/dirA_nil.py data/dirA_nil_v1 --out figs_nil` and `dirA_figures.py --methods nil`. An output
directory is bound to its manifest (SHA-256); `collect` uses only that manifest's cells. The analysis scripts read
through a *source* and write through *sinks* (`scripts/dirA_io.py`), so every computation also runs on in-memory
tables. Measured on one core: one validated 4-qubit cell of the NIL study (15 methods, 150 Monte-Carlo checks)
takes about 10 minutes. Release notes: `docs/v14/release_notes.md`.

Two further studies (v1.4.1, release notes `docs/v14/release_notes_1.4.1.md`):

* **Global against local folding for ZNE** — `manifests/direction_A_zne_folding_v1.yaml` (914 cells: Grover, QFT,
  random brickwork, BV and GHZ circuits with 3–189 two-qubit gates; readout on and off; coherent errors; folding
  inside composed methods). ZNE records for every folded circuit its effective noise scale λ_eff (the light-cone fault
  rate of the folded circuit ÷ the original's) and can fit against it (`lam_est: fault-rate`); local folding is also
  an outer method of the chain (`zne_folding: local`) and a feature generator of NIL (`neighbors: fold, folding:
  local`). `python scripts/dirA_zne_folding.py data/dirA_zne_folding_v1 --out figs_zne_folding` finds the exact
  crossover budgets (where local folding breaks down) and draws crossover maps. **Fractional ZNE** (stages F5, F6):
  local folding at λ = 1, 4/3, 5/3, 2 with a chosen fold subset (`subset: random | nested | stratified |
  balanced`), averaged over `fold_draws` independent subsets, fitted by `rule: poly2 | linear | exponential`; it pays
  only on circuits whose light-cone fault count exceeds about 0.6.
* **Hardware against fake-device divergence** — `manifests/direction_A_divergence_v1.yaml`: the same application
  circuits on an IBM device and predicted from the device's own calibration of the day (per-gate depolarizing model
  with readout; Chen et al.'s median model; the full fake-device model), raw, readout-mitigated and with ZNE at equal
  budget. `scripts/dirA_divergence.py` plans, submits (`run-ibm`, token from `$QISKIT_IBM_TOKEN`, a saved account or
  a hidden prompt — never a command-line argument or a file), collects with the calibration at execution time, runs
  dry runs (`run-local`, `emulate` with known injected noise) and draws the difference volumetric plot
  (Predicted Fc − Observed Fc over width × post-compilation two-qubit gates, |z| > 3 outlined).
  `manifests/direction_A_divergence_v2.yaml` adds what explains the gap: layout variants on disjoint chains (plurality
  vote, heterogeneity, a heavy-hex map of the excess per qubit), spectator-context probes, the folding fingerprint,
  the χ² ladder (rescaled calibration or context?) and the divergence budget; `dirA_divergence.py recovery` draws
  the diagnostic matrix of several analyses (emulations with planted errors beside the hardware run). Research plan:
  `docs/v14/research_plan_divergence.md`.

**1.4.2** (release notes `docs/v14/release_notes_1.4.2.md`): the plurality vote follows the authors' reference
implementation (threshold cap, raw survivor frequencies); a variant mode `layouts+twirl` changes the qubit assignment
and the gate frame together, and chains are never reused in the same orientation; `scripts/dirA_figures.py` is
redesigned for the thesis (one style system with `times`/`lm`/`sans` presets, colorblind-safe diverging and
sequential scales, the fidelity-gap and χ²-ladder figures, a clean diagnostic matrix), and every figure of an
analysis can be redrawn from its tables:

```
python scripts/dirA_divergence.py figures figs_div --run-dir data/dirA_div_v2 --out figs_redrawn --font lm
```

```
python scripts/dirA_divergence.py plan manifests/direction_A_divergence_v1.yaml --out data/dirA_div_v1
python scripts/dirA_divergence.py run-ibm data/dirA_div_v1 && python scripts/dirA_divergence.py collect-ibm data/dirA_div_v1 --wait
python scripts/dirA_divergence.py analyze data/dirA_div_v1 --tag ibm --out figs_div --workers 8
```

Fake devices accept modifiers: `fake_fez@s=2,ro=0,eng=exact` scales every gate error and relaxation rate by 2,
switches readout errors off, and runs the exact superoperator engine. That engine reproduces Aer's device model and
also applies signed PEC maps, so PEC's mean is exact.

## Pages

| Page | What it answers |
|---|---|
| **Home** (`app.py`) | The bias–variance trade-off in one equation, a ten-second live demo, the method map |
| **Learn** | Eleven hands-on guides (seven on mitigation basics, four on research methods) that set up the Playground or an analysis page in one click; QC fundamentals for QEM; walkthroughs of VQE, QAOA, Shor, HHL, QEC primitives, Grover's search and the hidden subgroup problem, each with a circuit diagram; **Research topics**: equal-budget comparisons and phase diagrams, global/local/fractional ZNE folding, context noise on heavy hex, device-versus-model divergence and running on IBM hardware within ten minutes |
| **Playground** | Build → Circuits → Topology & layout → Time travel → Mitigate & compare → Export (below) |
| **Batch run** | Tick algorithms × methods × noise models × noise scales × simulators and **Run all**: one table, a scenario × method heat map, winners per scenario, error against the noise scale; everything in the ZIP. **Sweeps** add register inputs (`ramp: 0, 1`, `input: plus, bits:0101`), measurement bases (`basis: Z, X`), any template parameter (`m: 1, 2, 3`) and any noise parameter (`p2: 0.005, 0.01`), every combination as its own scenario, with an *error against the swept value* chart |
| **Cost vs precision** | RMSE(B) = √(b² + v/N_t + w/s + u) for every contender at a total shot budget B (target, training and calibration shots together); budget and QPU time to reach a target ε; the crossover B\* where a mitigated method beats the raw one; the PEC exponential wall Γ = Πγ as a curve and a depth × error-rate map; a depth sweep with real simulations |
| **Bias–variance bullseye** | Every dart is one simulated experiment, placed by its error in two observables |
| **Model mismatch** | PEC with an assumed rate ≠ true rate (curve and heat map), readout calibration drift, unmodeled coherent noise |
| **Head-to-head sweeps** | Any number of contenders over p₁, p₂, damping, over-rotation, readout error or a noise scale; a "who wins where" map over (parameter, shots) |
| **Theory & sources** | A research handbook in six parts with a contents column and diagrams for every algorithm: every equation behind the app, including the equal-budget accounting; advanced mitigation (symmetry expansion, pulse-level ZNE and PEA, QSE, classical shadows, M3, chained mitigation); the road to fault tolerance (stabilizer formalism, code taxonomy, decoding, threshold, logical QEM); **Part VI, research methods**: the light-cone fault rate, equal-budget phase diagrams, a device against its component model, context-dependent noise on heavy hex and the diagnostics that explain the gap; 123 references with DOIs; calculators |
| **Run on IBM Quantum** | The Playground's experiment as sampler jobs — a local rehearsal on a fake device, then real hardware — with every contender's budget allocated exactly as in simulation and the Playground's full results view and ZIP |

### Playground

| Tab | What you can do |
|---|---|
| Build | Pick a template by family (below). Drag gates and noise channels from the palette onto the wires (drop on the bin to delete, drag the small handle of a two-qubit gate to retarget it, double-click to edit an angle, Undo/Redo), or edit as a table or as text (`h 0; cx 0 1; depol(0.05) 0 1`). Choose input states, measurement bases, the observable — including any **Hamiltonian** `0.5*Z0*Z1 + 0.2*X0 - 0.1*Y1`, measured in qubit-wise commuting settings — and the compilation (abstract gates, or an ISA with CX/CZ on all-to-all, line, ring, **2D grid** or **heavy-hex** connectivity; or a fake device's own target). The **Show the compiled (physical) circuit** switch aligns every logical gate above its native gates (H → RZ·√X·RZ) with routing SWAPs framed |
| Topology & layout | The coupling graph (heavy-hex devices drawn row by row) with the initial layout, the couplers where SWAPs were inserted and why each one was needed; calibrated qubit and coupler errors; a mapomatic-style **search for the lowest-error physical qubits** and one click to recompile on them |
| Circuits | The logical circuit next to what a method actually runs: ZNE folds (U†U blocks shaded), a PEC instance with its sign, a twirl instance, DD pulses in idle windows, a CDR training circuit, VD copies; composer view or Qiskit's IBM-style drawing |
| Time travel | Scrub step by step through the implemented, folded, DD or twirled circuit. The executed operation is boxed — only that gate and the wires it acts on. Per-qubit Bloch spheres that wrap into rows for 5+ qubits (noisy vs ideal, tilt and length, trails), flat projections, the density matrix (or its error), fidelity, purity and ⟨O⟩ step by step. For VQE and QAOA, **live θ sliders** re-simulate on every move, and an energy-landscape scan (1D, or γ × β) shows ideal against noisy |
| Mitigate & compare | Any number of preset or custom contenders (estimator + twirling + DD + readout mitigation), all run by the sidebar's single **Run simulation** button. An **Analysis** panel picks the view from the result's mathematics — an output-distribution histogram (ideal, noisy, readout-corrected, ZNE-extrapolated; exact) for probabilities, estimates ± σ for expectation values, logical error and syndromes for codes, or each method's diagnostic curve — with a manual override. Then an **equal-budget** cost table (bias, sd, RMSE, γ, circuits actually sent, acceptance, target / training / calibration shots, QPU time, status) whose tooltips render their formulas as Unicode math, and a look inside each method |
| Export | A standalone Python script that runs in a terminal **and** as a pasted Jupyter cell; state JSON download and upload |

Every major figure has a **Why this happens?** expander with the physics in TeX. Heavy work runs only behind a
Run button; results show a stale badge or warning when the inputs change. Rotation angles accept `pi`, `π`, `\pi`,
`2pi`, `3π/4` or `\frac{\pi}{8}` and are shown as fractions of π (`π/128`). Applying an input (`ramp:1`, `plus`,
`bits:0101`) or a basis to all qubits draws the preparation and basis-change gates in the composer. Every
"Download all (ZIP)" button bundles tables (CSV and LaTeX), repeated-experiment samples, figures (PNG at 3× and
SVG; Plotly JSON when no Chrome/Chromium is available — no HTML files) and the state JSON. The Playground's and the
IBM Quantum page's bundles also compute every cross-page result of the comparison — cost and precision (RMSE
against budget, budget for ε, crossovers), the PEC wall, the bias–variance split — and add the other pages' stored
results and the Qiskit script.

## Templates

| Family | Templates |
|---|---|
| Basics | Bell pair, GHZ, Grover (2 qubits) |
| Algorithms | QFT and approximate QFT (SWAP network, reversed input or relabeled output), phase estimation, QPE of an Ising Hamiltonian, Trotterized Ising dynamics, Bernstein–Vazirani, Simon |
| NISQ · variational | VQE with a hardware-efficient ansatz (angles optimized on load or random), QAOA for MaxCut (ring, path, 3-regular, complete, random graphs; depth p; optimized or fixed angles), QML ZZ-feature-map classifier, data re-uploading classifier |
| FTQC · error correction | 3-qubit bit-/phase-flip codes (syndrome ancillas with coherent Toffoli correction, syndrome only, or majority-vote readout; single errors or error channels), repetition-code memory (distance d, r syndrome rounds), **surface code d = 2** (one plaquette, [[4,1,2]]; transversal or ancilla syndrome extraction), **five-qubit code [[5,1,3]]** (Clifford encoder, syndrome lookup decoder, corrects any single-qubit error), Steane [[7,1,3]] encoder with raw or syndrome-decoded logical readout |
| FTQC · encoded algorithms | **code × benchmark** (v1.4): bit-flip, phase-flip, [[4,2,2]] or [[5,1,3]] blocks running state preparation, Bell, GHZ-3, Deutsch–Jozsa, Bernstein–Vazirani, RX, RZ Ramsey, Grover-2 or QFT-2; every logical gate labeled fault-tolerant or not (with the reason); decoder as the observable, error detection as symmetry verification with every check; injectable single errors; eight presets on the template card |
| FTQC · deep algorithms | Shor order finding for N = 15 (precomputed or repeated powers; permutation or synthesized multipliers — up to thousands of two-qubit gates), **Grover's search** (n ≥ 2 qubits, any marked item, optimal or custom iterations; multi-controlled Z), the **hidden subgroup problem** (Z₂ⁿ with a hidden subgroup of dimension 1–2, and the cyclic group Z_N), HHL for a 2×2 system |

## Methods

| Family | Implemented | Source |
|---|---|---|
| Zero-noise extrapolation | global and local folding; Richardson, linear, exponential; optimal shot allocation | Temme, Bravyi & Gambetta, PRL 119, 180509 (2017); Giurgica-Tiron et al., IEEE QCE 2020 |
| Probabilistic error cancellation | inverse and compensation (depolarizing model with assumed rates), learned Pauli model; exact means, sampled instances on Aer; one shot per sampled circuit by default (shots per circuit adjustable, with the instance-variance term) | Temme et al. (2017); Endo, Benjamin & Li, PRX 8, 031027 (2018); van den Berg et al., Nat. Phys. 19, 1116 (2023) |
| Learning-based | CDR (noisy ⟨O⟩) and feature regression FR (⟨O⟩, ⟨Zᵢ⟩, ⟨ZᵢZⱼ⟩; this project's heuristic) with OLS, Ridge, Lasso; tunable number of kept non-Clifford gates; **adaptive training set** that guarantees varying training values or reports the fit as not applicable; training-shot and calibration variances by Monte-Carlo refits | Czarnik et al., Quantum 5, 592 (2021); for FR the closest published work is Liao et al., Nat. Mach. Intell. 6, 1478 (2024) |
| Neighbor-informed learning | weight-1 Pauli neighbors (or ZNE neighbors) learned on 2-design training circuits; regression rule selectable: ℓ1-constrained (Chen et al.), Lasso, Ridge, OLS, elastic net, NNLS; certified solvers; optimal or equal target split | Chen et al., arXiv:2512.12578 (2025); Strikis et al., PRX Quantum 2, 040330 (2021) |
| Chains | an outer estimator (ZNE, CDR, NIL) applied to an inner mitigation of every circuit it runs (raw, symmetry verification, virtual distillation, PEC), with exact cost accounting | Lowe et al., PRR 3, 033098 (2021); Bultrini et al., Quantum 7, 1034 (2023) |
| Readout | tensored inverse, M3 (subspace solve), TREX (twirled readout) | Bravyi et al., PRA 103, 042605 (2021); Nation et al., PRX Quantum 2, 040326 (2021); van den Berg, Minev & Temme, PRA 105, 032620 (2022) |
| Symmetry verification | post-selection on one or several Z-parities (e.g. every Z check of a code) | Bonet-Monroig et al., PRA 98, 062339 (2018); McArdle, Yuan & Benjamin, PRL 122, 180501 (2019) |
| Virtual distillation | M copies (ideal copies and derangement) | Huggins et al., PRX 11, 041036 (2021); Koczor, PRX 11, 031057 (2021) |
| Pauli twirling / RC | exact twirl of two-qubit or all gate errors | Wallman & Emerson, PRA 94, 052325 (2016) |
| Dynamical decoupling | X2 and XY4 in idle windows | Viola & Lloyd, PRA 58, 2733 (1998); Ezzell et al., PR Applied 20, 064027 (2023) |

A method is a class with a parameter schema registered with `@register` (`qemx/methods/`); the UI builds its
controls from the schema. See `docs/ARCHITECTURE.md` and the Theory page (`docs/theory.md`).

## Simulators

* **Exact density matrix (NumPy)** — instructions fused into one superoperator per step; exact PEC means.
* **Qiskit Aer** (`density_matrix`) — the same instruction list; PEC samples instances.
* **Fake IBM devices** (FakeFez, FakeTorino, FakeBrisbane, …) — the circuit is compiled for the device's coupling
  map and native gates (broken couplers excluded), scheduled ALAP with real durations; gate errors, thermal
  relaxation in idle windows and per-qubit readout come from the calibration snapshot.

The noise injector also models **crosstalk**: a driven gate kicks the phase of every idle neighbor
($R_Z(\delta)$ and a residual $ZZ$), which dynamical decoupling refocuses.

Every number field accepts exact typed values next to its slider. There are **no hard limits** on qubits or depth:
the app estimates memory (16·4ⁿ bytes for an n-qubit density matrix) and warns, a per-run time limit can be set in
the sidebar, and out-of-memory errors or time-outs are caught and explained instead of crashing the page.

## Equal budgets

Every contender spends the same total budget B per estimate: B = N_t + F + c·s (target shots, training shots of
CDR/NIL, c readout-calibration circuits with s shots each, s chosen optimally). The table's *sd* is
√(v/N_t + w/s + u) — target shots, the finite calibration (delta method) and the training data's shot noise — and
*circuits* counts what is actually sent: one per ZNE scale, ⌈N_t/s⌉ sampled PEC circuits (one shot each by
default), one random circuit per shot with twirling, 8 measurement masks per circuit with TREX, the training and the
calibration circuits (`qemx/budget.py`).

## Running on IBM Quantum

The *Run on IBM Quantum* page rehearses the job locally on a fake device (SamplerV2 in local testing mode) and
then submits it to real hardware through `qiskit-ibm-runtime` with the channel `ibm_quantum_platform`. Each
contender's budget is allocated exactly as in simulation (ZNE's optimal split, s* calibration shots, PEC's K
sampled circuits, training shots), contenders never share counts, and raw, ZNE, PEC, CDR/NIL, symmetry
verification, readout mitigation (including TREX) and twirling/DD options are replayed on the measured counts with
exactly the estimator code used in simulation. The results page and the ZIP are the Playground's.

## Security

* **IBM token** — entered only in a password field (`st.text_input(type="password")`), used inside the submit
  callback and deleted from the session immediately afterwards; it is never written to exported `.py` scripts,
  `.json` state files, ZIP bundles or logs, and error messages are scrubbed of it. *A token is a password. Before
  this notebook goes into a repository or an appendix, clear it and use one of the other two.* Exported scripts read
  `IBM_QUANTUM_TOKEN` (or `QISKIT_IBM_TOKEN`) from the environment or use a saved account, and never print a token.
* **Input sanitization** — gate parameters are evaluated by a whitelist AST interpreter (numbers, + − × ÷ ^,
  π, e, τ, sqrt, sin, cos, asin, acos, atan; length and exponent limits), never by `eval`; the text-to-gate
  parser accepts only known gate and noise names; uploaded state files are size-limited and validated field by
  field; labels are stripped to a safe character set and every user- or exception-derived text is Markdown-escaped
  before it is displayed.
* **Robust execution** — every simulation runs inside `try/except`: memory errors, time-outs and unexpected
  exceptions become an `st.error` with an explanation, never a crash or a stack trace (`showErrorDetails = "none"`).

## Tests

```
pytest -m "not slow"   # engine, templates, noise, backends, methods, analysis, security, UI state
pytest                 # + every page through AppTest, the exported script in a terminal and in a Jupyter kernel
python scripts/validate_learning_moments.py --out validation.json    # statistical check (independent experiments)
python scripts/profile_workflows.py --root . --out profile.json       # timing and memory of representative workflows
```

The number of tests changes with every release: `pytest --collect-only -q | tail -1` prints the current count.

## Deployment

Streamlit Community Cloud or any host with Python 3.10–3.13: `requirements.txt` for Python packages and
`packages.txt` (Chromium) for high-resolution PNG/SVG figure export. `.streamlit/config.toml` hides stack traces,
enables XSRF protection and limits uploads to 20 MB.

## Layout

```
app.py                      landing page
pages/                      Playground, Cost vs precision, Bias–variance, Model mismatch, Head-to-head,
                            Theory & sources, Run on IBM Quantum, Batch run, Learn
qemx/                       engine (no Streamlit): circuit, templates, observables, hamiltonian, noise, compile,
                            stages, layout_search, device, backends/, transforms, methods/, recipes, analysis,
                            insight, variational, batch, sweep, learn, timeline, hardware, budget, report, export,
                            bundle, resources, explain, viz/
ui/                         Streamlit glue: navigation, session state, widgets, sidebar, drag-and-drop composer,
                            results (the shared results view), tooltips (math in help texts), figures (diagrams)
tests/                      pytest
docs/                       ARCHITECTURE.md, theory.md (the handbook), learn/ (Learn chapters), figures/ (diagrams)
scripts/make_figures.py     rebuilds docs/figures/*.svg from quantikz sources (pdflatex + pdftocairo)
```
