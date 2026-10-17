# QEM Explorer 1.4.2 — release notes

1.4.2 refines the divergence study's estimators and redesigns its figures for the thesis. QEM Explorer and IBM devices
remain the only subject.

* **Plurality vote.** It now follows the authors' published reference implementation exactly (§1).
* **Variant mode `layouts+twirl`.** It changes the qubit assignment and the gate frame together, and the chain
  assignment never repeats (§2).
* **Figures.** `scripts/dirA_figures.py` is redesigned around one message: *the gap between a component-level noise
  model and the device's context-dependent noise* (§3).
* **The 10-minute hardware manifest.** `direction_A_divergence_v2_qpu10` fits the diagnostics into 720,000 shots and
  is validated in emulation (§4).
* **Theory and Learn.** The handbook gains Part VI on the research methods, and the Learn page a *Research topics*
  tab, four research guides and one-click setups for the analysis pages (§5).

Every number below comes from runs of the shipped code in the development sandbox (fake_fez, Qiskit 2.x, Aer 0.17).
**No IBM device was reachable from here.** Every divergence figure therefore shows an emulated device — fake_fez with
a known, planted error — and is labeled "emulated".

## 1. Plurality vote (`qemx.context_noise.plurality_vote`)

Chen et al.'s approximate plurality vote (App. B, Eqs. 4–6) now follows their published reference implementation step
by step. It differs from 1.4.1 in two places:

* **Threshold.** t = min(t_start, the largest number of variants that share one outcome). It is set in one step
  rather than lowered one at a time. t ≤ 2 falls back to the average, and a single surviving outcome gets probability
  1.
* **Frequencies.** The survivors keep their *raw* measured frequencies. Variants without a survivor are dropped, but
  the frequencies are not renormalized over the survivors.

`renormalize=True` restores the 1.4.1 reading of the paper, and it gives visibly different votes when the variants
put different mass outside the survivors (`test_plurality_vote_raw_and_renormalized_differ`). The vote probability
is computed by the Poisson-binomial recursion, which is exact; the reference instead sums over subsets. It is checked
against brute-force enumeration of every joint draw, in both modes.

## 2. Variants on heavy hex (`qemx.divergence`)

| mode | what varies between the variants of a benchmark |
|---|---|
| `layouts` | the chain the circuit is laid on: disjoint chains (best calibrated first), then the **same chains reversed** (another qubit-to-position assignment), then overlapping chains — never one assignment twice |
| `twirl` | the gate frame: an independent Pauli twirl of every two-qubit gate, same chain |
| `layouts+twirl` | **new:** both — another chain *and* another twirl per variant |
| `seeds` | the transpiler seed, same chain |

`disjoint_chains(backend, n, count)` was checked on fake_fez with 40 variants of 6 qubits: 17 disjoint chains, then 17
reversed, then 6 overlapping, and no duplicates. A frame or a chain never changes the ideal output. On heavy hex,
though, another chain also changes the spectators and the couplers. That is exactly the context noise an average gate
fidelity cannot see, and the variant heterogeneity test isolates it.

Tests: `test_disjoint_chains_reverse_then_overlap_but_never_repeat`, `test_layouts_plus_twirl_variants`.

## 3. Figures for the thesis (`scripts/dirA_figures.py`)

**Style system.**

* `setup_matplotlib(preset, base=8)` selects a type preset, also settable through `QEMX_FIG_FONT`:
  * `times` (the default): TeX Gyre Termes, with STIX maths;
  * `lm`: Latin Modern, for a LaTeX thesis set in lmodern;
  * `sans`: TeX Gyre Heros, for slides.
* Japanese text falls back to Noto CJK JP.
* Sizes come from the page: `TEXT_WIDTH = 6.3 in`, `COLUMN_WIDTH = 3.1 in`, 8 pt text.
* PDFs are written with embedded TrueType fonts. Axes and legends are kept recessive (no frames).
* `dirA_figures.py` and `dirA_divergence.py figures` take `--font`.

**One color grammar for the gap** (validated with the dataviz method):

* **Diverging Δ.** Blue ↔ gray ↔ red, five steps per arm at mirrored OKLab lightness:
  0.34, 0.43, 0.57, 0.72, 0.86 | 0.95 | 0.86, 0.72, 0.57, 0.43, 0.35.
  * Red means the device is worse than its model, blue means better, gray means as modeled.
  * The scale is symmetric, capped at a rounded 98th percentile; an arrow on the colorbar marks values beyond it.
* **Sequential.** The red arm alone, for the per-qubit excess.
* **Families are shapes.** Circuit families get unique marker shapes and are never colored, so color stays free for
  the value.
* **Significance is drawn, not colored.** A filled marker is beyond the shot noise (|z| > 3); an open marker is
  within it.
* **Mechanisms and probe modes** use the reference categorical palette in a fixed order. It is CVD-checked on
  adjacent pairs, and on all pairs of the first three colors.

| figure | what it shows | change |
|---|---|---|
| `fidelity_gap` | **headline**: observed and predicted F_c against two-qubit gates, joined per circuit, with the fitted decays and their ε_eff written at the lines | new |
| `difference_volumetric` | tiles with surface gaps, log₂ or √2 bins, a significance dot per tile. `circuits=True` draws every circuit inside its tile (shape = family; "k/n" when crowded). `annotate=k` numbers the worst k and names them in the legend | redesigned; annotation off by default |
| `difference_volumetric_grid` | the same rows and bins in every panel, one horizontal colorbar ("device better ← → device worse"), one key | redesigned |
| `predicted_vs_observed` | color = significance class, shape = family, a one-row key | redesigned |
| `divergence_budget` | (a) each mechanism's mean contribution per circuit with its 95 % interval; (b) per family, stacked, with the unexplained part hatched; one unit across both panels | redesigned |
| `ladder_chart` | χ²/dof of the nested explanations (none, rescaled calibration, context, both), the 99 % point of each, and the verdict underneath | new |
| `folding_fingerprint` | faint circuits, pooled family points with ±1 SE, non-overlapping direct labels (`place_labels`), subtle regions | redesigned |
| `heavy_hex_map` | warm sequential scale, rings where significant; the worst qubits and every significant qubit labeled | redesigned |
| `context_probe` | the three spectator modes, observed solid, predicted dashed, direct labels; excess ±2 SE | redesigned |
| `diagnostic_matrix` | a real table: a red dot for an anomaly, an open ring for "as modeled", key numbers in gray; cells wrap and rows grow | redesigned |
| `attribute(fig, text)` | a source line under any figure (device and date, or "emulated"). The layout is frozen first, so a tight save cannot move a legend onto it | new |

**Drivers.**

* `dirA_divergence.py analyze` also writes `divergence_gap_*` and `divergence_ladder`.
* `dirA_divergence.py figures DIR --run-dir RUN` redraws *every* figure of an analysis from its tables, with no counts
  and no simulation needed. `--no-per-method` skips the single-method volumetric plots.

**Migration.**

* `difference_volumetric(…, annotate=4)` reproduces the numbered labels of 1.4.1.
* `DIV_COLORS` now has 11 anchors (it had 7).
* Every function name and positional signature that other scripts use is kept.

## 4. Results, tests

* `docs/v14/results_1.4.2/` holds the redrawn figures of the pilot and v2 analyses and their diagnostic matrices
  (`*_recovery`, with `recovery.csv`). All of them are emulated on fake_fez with planted errors, and say so.
* **The 10-minute manifest.** `manifests/direction_A_divergence_v2_qpu10.yaml` trims v2 to one QPU window: fractional
  ZNE dropped, probes at core depths 4 and 16, 4,500 shots per benchmark and method. That gives 720,000 shots in 1,680
  runs (v2: 2,214,000 in 2,868), about 410 s at 0.57 ms per shot. The same four planted causes were emulated on it, and
  the `qpu10_*` folders hold the figures. Every cause is still named:

  | planted | ladder | budget | fingerprint $(A_g, A_l)$ | probes | ε ratio |
  |---|---|---|---|---|---|
  | none | as modeled | — | 0.97, 0.96 | none | 1.00 |
  | gate errors × 1.6 | rescaled calibration × 1.45 | calibration 74 % | 1.42, 1.43: gate-attached | gate error | 1.61 |
  | $T_2$ = 30 µs | context + non-linear | idle time 100 % | 1.38, 0.91: time-attached | idle dephasing | 1.30 |
  | ZZ 0.05 rad | context + non-linear | parallel 33 %, calibration 31 %, active spectators 22 % | 1.94, 2.56: repetition-coherent | spectator crosstalk | 1.70 |

  No position-dependent cause was planted in this check: the layout column was tested for false alarms only (none).
* **New tests.**
  * `tests/test_figures_v142.py` covers the presets, the palette's mirrored lightness, the recovered ε_eff rates,
    non-overlapping labels, the ladder and the source line.
  * The variant and vote tests are in `tests/test_context_noise.py`.
  * `tests/test_learn_v142.py` covers the Learn overhaul (§5).

## 5. Theory and Learn

**Theory (`docs/theory.md`, the Theory & sources page).**

* **Part VI, "Research methods"** (§18–21) is new:
  * the light-cone fault rate and exact noise scaling;
  * equal-budget regimes, crossovers and unreachable precision;
  * the two-constant cost law and the Monte-Carlo validation;
  * a device against its component model ($F_c$, $F_N$, three predictors, same-shot comparison, $\varepsilon_{\rm eff}$,
    the volumetric plots, emulated causes);
  * context-dependent noise on heavy hex (features, layout variants, the plurality vote, spectator probes, per-qubit
    attribution);
  * the diagnostics (χ² ladder, divergence budget, folding fingerprint, diagnostic matrix, the hardware protocol).
* **Elsewhere in the handbook:**
  * §4.8 reads folding as a measurement of the noise;
  * §16.7 covers chained mitigation (an outer ZNE, CDR or NIL over an inner symmetry verification, VD or PEC) and its
    accounting.
* **References.** 20 new references, 123 in all.
* **Fix.** A control character that had replaced `\theta` in §3.1 is removed.

**Learn (`docs/learn/`, `qemx/learn.py`, the Learn page).**

* **A *Research topics* tab** with five new chapters, each with one-click setups:

  | chapter | what it covers |
  |---|---|
  | `09_equal_budget` | the ledger, the three regimes and $B^*$, reading Cost vs precision and Head-to-head, λ_LC, the research maps |
  | `10_zne_folding` | global, local and fractional folding, $\lambda_{\rm eff}$ and the residual ρ, coherent errors and twirling, the fingerprint |
  | `11_context_noise` | the mechanisms on heavy hex, the context features, layout variants, the spectator probes, the vote |
  | `12_divergence` | the divergence pipeline, how to read every figure, the diagnostics, emulated causes, the commands |
  | `13_hardware` | the two routes to hardware, the 10-minute budget, token handling, the run-day checklist |

  The probe chapter includes a four-qubit spectator probe that runs in the Playground. Its response to DD and ZNE
  names the mechanism.
* **Four research guides** under *Research methods*: *Who wins depends on the budget*, *Global versus local folding*,
  *Spectators: find the context noise* and *When the model misses an error*.
* **The fundamentals** gain a note on chained mitigation and §7, a bridge to the research topics.
* **Setups can open the analysis pages.** `ui.state.apply_setup` now opens Cost vs precision, Head-to-head, Model
  mismatch or the IBM Quantum page:
  * with an opening tab (`tabs`);
  * with whitelisted controls preset (`widgets`, prefixes `w_hh_`, `w_cvp_`, `w_mm_`, `w_hw_`, `w_b_`);
  * with custom contenders (`custom`), validated exactly like the recipe builder's.

  A setup is static data. Nothing in it is executed, and an invalid one is reported, never applied.
* **Every number the new chapters and guides quote** is reproduced from the engine in `tests/test_learn_v142.py`, so
  the text cannot drift from the code. The test also covers the setups, the button labels, spelling (American), and
  an AppTest of every topic setup.
