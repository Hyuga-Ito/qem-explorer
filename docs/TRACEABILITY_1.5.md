# QEM Explorer 1.5.0: traceability and implementation report

## Revision and environment

- **Starting point.** The audited archive `qem-explorer-1.4.2-code.zip`, SHA-256 `26da74f9f310b72e95ecab01632f9a4569a8643ba9938f6ffca66aa6bb04af5b`, the same revision the QA report examined. It is imported as baseline commit `628f72b`. The QA findings were reproduced on it before any change, with matching numbers:
  - the learning moments;
  - the readout γ;
  - the NIL test failure (`0.23523180844059890 − 0.23523180844059893 ≈ −2.8e−17`);
  - the dead-renderer export.

  The baseline suite gave 788 passed, 1 failed (that NIL assertion) and 1 deselected (the test that contacts IBM).
- **Commits.**

  | commit | content |
  |---|---|
  | `38107ca` | priority-1 repairs |
  | `c53eff7` | I/O semantics, presets, experiment model, discrepancy |
  | `8f0a715` | study, recommender, campaign |
  | final commit | navigation, docs, tests |
- **Environment.** Linux x86-64, 2 CPUs, Python 3.13.16 with:
  - NumPy 2.5.3, SciPy 1.18.1, Qiskit 2.6.0, Qiskit Aer 0.17.2, qiskit-ibm-runtime 0.50.0;
  - Streamlit 1.66.0, Plotly 7.1.0, scikit-learn 1.9.1, pandas 3.0.5;
  - Chromium for Kaleido and for Playwright.
- **Sandbox-only workaround.** The sandbox needed an `LD_PRELOAD` of the two bundled libgomp copies, because the scikit-learn and Qiskit Aer wheels conflict over static TLS on Python 3.13. This is a host workaround and is **not** shipped or required by the app.
- **No hardware.** No QPU job was submitted and no credits were used.

## Evidence classes used below

- **R**: reproduced defect, fixed and covered by a test.
- **S**: source-level risk, addressed in code.
- **H**: hypothesis, not established.
- **E**: environment restriction.
- **U**: unexecuted scenario.

## Traceability table

| # | requirement / finding | evidence | affected code | change | test | outcome | remaining limitation |
|---|---|---|---|---|---|---|---|
| 1 | Learning summaries mixed the exact-fit mean with finite-shot spread (QA §4.2) | R: audit cases reproduced exactly | `methods/learning.py`, `methods/nil.py`, `methods/compose.py`, `hamiltonian.py`, `recipes.summary` | `qemx/stats.py` contract: `estimate_kind`, protocol moments μ = E_D[m], Var = Var_D[m] + E_D[v]/N_t, adaptive MC with tolerance, failure rate and the `conditional-on-success` flag (>50 % failures → no numbers), exact fit kept as `reference_mean`; hardware replay keeps the measured mean | `scripts/validate_learning_moments.py` (6 000 independent end-to-end experiments per case); `tests/test_qa_v15.py`; `tests/test_nil.py`; `tests/test_compose.py` | reported vs empirical RMSE: CDR 0.1328 vs 0.1354 (z −0.90); FR 0.3986 vs 0.3973 (z 0.47); NIL 0.1825 vs 0.1818 (z 1.44); audit: 0.0880/0.1354, 0.2704/0.3973, 0.1754/0.1818 | shared-calibration effect on the mean is first order (flagged); the MC error of learning RMSE is not propagated into the study's bootstrap intervals |
| 2 | Readout γ wrong above 50 % flips (QA §4.3) | R: γ = 1e24 for e = h = 0.6 | `methods/base.py` | `assignment_inverse_gamma` = (1 + \|e − h\|)/\|1 − e − h\|; invertibility, conditioning and domain separated; singular → explanatory error; TREX uses the same rule | `test_qa_v15.py` (closed form against the explicit inverse, 25 for two qubits, singular in all three modes, negative determinant accepted) | γ = 25 | near-singular calibrations are accepted, with a condition number; first-order calibration error is unreliable there (documented) |
| 3 | PEC/NIL input validation (QA §4.4) | R: `depol_quasi(−0.01)`, NaN, learned PEC at p = 1, `nil_fit` with γ < 0 | `methods/pec.py`, `methods/nil.py` | domain checks with physical explanations; singular transfer eigenvalue detected before any reciprocal; γ = 0 defined (c = 0) | `test_qa_v15.py` | invalid inputs fail clearly | — |
| 4 | NIL objective assertion exact at 1e−17 (QA §4.5) | R | `tests/test_nil.py` | scale-aware round-off tolerance; feasibility, gap and certificate checks kept | `test_nil.py` | passes; nothing loosened beyond round-off | — |
| 5 | Export hangs on a dead renderer (QA §4.6) | R: the 1.4.2 queue wait never returns | `qemx/bundle.py`, `qemx/render_worker.py` | rendering in a separate, time-limited process; partial images kept; data written regardless; data-only toggle | `test_qa_v15.py` (chrome / crash / hang faults, data-only) | missing Chrome 0.23 s, crash 0.16 s, hang bounded by the limit; normal 6-figure export 7.0 s (1.4.2: 7.6 s) | Kaleido's own server failure modes beyond these three were not enumerated |
| 6 | Budget preflight (QA §3.1, §5.5) | R: default NIL needed ≥ 212 997 shots at a 32 768 budget and showed 120 failed repeats | `recipes.py`, `methods/*.min_cost`, `ui/results.py` | `insufficient_budget` before any fit, "needs at least N shots", unaffordable row shown; the "NIL · small budget" preset is labeled budget-scaled | `test_qa_v15.py` (inclusive minimum, detection, unrun row) | no fit or MC for unaffordable protocols in the Playground | studies deliberately cost every recipe (`check_budget=False`) |
| 7 | Guide state (QA §3.1, §5.5) | R: guides merged into prior contenders; partial updates possible | `ui/state.py`, `pages/9_Learn.py`, `pages/1_Playground.py` | saved vs active recipes; atomic validate-then-apply with rollback; replace (default) or merge | `test_qa_v15.py` (replace, merge, atomic, legacy state) | a guide always defines its own experiment | — |
| 8 | FR named `features: nil` (QA, prompt §3.3) | R | `methods/base.py`, `methods/learning.py`, `phase/manifest.py` | `PARAM_ALIASES` migration with reported notes; manifests not rewritten (cell ids stable); new manifests use FR-* / NIL-* keys | `test_qa_v15.py` | unambiguous ids | old manifest keys `NIL-L`/`NIL-R` still mean FR (documented) |
| 9 | Record schema and mixed semantics | S | `phase/store.py`, `phase/run.py` | schema `/3`; `run` refuses to finish an older-schema directory without `--force` | `test_qa_v15.py` | old data preserved, never mixed silently | — |
| 10 | Output control mutated the input (prompt §4) | R: QFT "output s" inserted preparation gates | `templates.py`, `pages/1_Playground.py`, `qemx/io_semantics.py` | the input parameter is named as input; `prep` = fourier / basis; read-only derived Output panel (ideal prediction, deterministic/uniform/spread, unavailable above 20 qubits with a reason) | `tests/test_io_semantics.py` (analytic QFT on basis inputs with two independent simulators, uniform output with input phases, \|+⟩ → \|0⟩, inverse round trip, SWAP/relabel order, measurement basis, no mutation, AppTest: input unchanged after refresh and parameter edits) | 30 tests pass | the bit-reversed load is defined for the Fourier input only (refused otherwise) |
| 11 | Cited presets (prompt §4) | literature read 2026-10-17 | `qemx/presets_cited.py`, template card | 7 presets with source, version, locator, convention, units, sizes, purpose, limits, kind; searchable selector; resolved values shown | `tests/test_presets_cited.py` (QAOA ring p = 1 gives 3/4 for n = 4, 6, 8; 3-regular ≥ 0.6924; the Trotter bound holds; AQFT index shift) | exact reproduction verified in the app's own convention | the Coppersmith exponent glyph was garbled in extraction (consistent reading used); Lloyd 1996 not read |
| 12 | One experiment model (prompt §9) | S: separate UI-state and CLI paths | `qemx/state_schema.py` (moved from `ui/state.py`), `qemx/experiment.py`, Export tab | `qemx.experiment/1`: short and state forms; field-path errors; secret scan; `safe_load`; config hash; stable method ids; validate / plan / run / from-state; YAML download and upload in the app | `tests/test_experiment.py` (Playground state → YAML → CLI reproduces the app's estimates to 1e−12; refuses to overwrite another configuration) | UI = YAML = CLI | the phase manifests remain the study-level format; they share recipe validation but are not `qemx.experiment` documents |
| 13 | Discrepancy measures (prompt §7.2) | S: UI computed TVD on signed vectors; Hellinger clipped silently | `qemx/discrepancy.py`, `ui/results.py`, `scripts/dirA_divergence.py` | validated TVD / Hellinger / F_c / normalized F; quasi-probabilities refused (ℓ1 labeled "not a TVD"); explicit projection with moved mass; parametric-bootstrap distance test; Holm / Hochberg / BH / Bonferroni; rate conventions | `tests/test_discrepancy.py` (matched-model false-positive rate within 4σ of 5 %, power > 80 %, Holm FWER, adjusted p known values, conventions against the channel's Choi state) | divergence tables flag with Holm | JSD not added (no distinct question identified) |
| 14 | qpu10 rehearsal (prompt §7.3) | U → executed as emulation | `scripts/rehearse_divergence.py` | null (truth = predictor) and controlled alternatives through plan → counts → analyze; rates with SEs; planned usage from the same preflight | run on v3 (`reports/rehearsal_v3`) | null: Holm 0 / 1 120, p < 0.05 4.4 % ± 0.6 %, z sd 0.94; crosstalk 0.05: raw 28 %, aggregated 62 %, RO 17 %, ZNE 0 % flagged | emulation only; ZNE rows have no power at this budget |
| 15 | 600-second campaign (prompt §8) | S: per-job limits only; batch TTL risk; qpu10 needed 596 s high | `qemx/campaign.py`, `divergence.submit/collect`, `hardware.submit`, `scripts/dirA_divergence.py`, IBM page | conservative usage model; ledger with admission before each job, per-job id persistence, reconciliation from `job.usage()`, overrun stop, outstanding bound, orphan resolution; required for `run-ibm`; job mode default; `max_execution_time` per job; v3 manifest fits | `tests/test_campaign.py` (mock provider: ceiling, failures, cancellations, crash, resume without duplicates, overrun); `tests/test_divergence.py` | plan: 525 s high of 540 s admissible (`docs/hardware_preflight.md`) | the guarantee holds while real usage ≤ high estimate; sub-job split and billing of limit-cancelled jobs are undocumented by IBM |
| 16 | Feature dictionary (prompt §6.2) | — | `qemx/features.py` | versioned; levels; costs; availability; NaN with reasons; intermediate entanglement profile; oracle features refused as inputs | `tests/test_study_recommend.py` | `docs/feature_dictionary.md` | exact state features cost O(2^n) and stop at 14 qubits |
| 17 | Study framework (prompt §6.1) | — | `phase/manifest.py`, `manifests/direction_A_v2_*.yaml` | maximin space-filling design per stratum; confirmatory split by base circuit; actual NIL beside FR | `test_study_recommend.py` | offline pilot: 75 cells / 25 base circuits in 7.5 min | the pilot is small; the scalable manifest (300 cells) was planned, not run |
| 18 | Regime maps and figures (prompt §6.4) | — | `qemx/study.py` | ≤ 6 core figures from stored records; ties, "raw ≈ best", "no feasible"; no interpolation; base-circuit bootstrap; captions state protocol, resource, unit, intervals and flags; `optimistic` methods (VD) shown as upper bounds, never candidates | report regenerated from tables (`reports/pilot`) | see "Findings" below | one device model |
| 19 | Recommender (prompt §6.5) | — | `qemx/recommend.py`, QEM Performance page | ridge per method; abstention (coverage, uncertain); "use raw"; baselines; grouped / leave-family / leave-width / confirmatory evaluation; leakage check | `test_study_recommend.py` (known rule learned, row split refused, abstention) | pilot numbers in "Findings" | not better than the best fixed method on grouped folds |
| 20 | Navigation (prompt §5) | — | `ui/nav.py`, `pages/10_*`, `pages/11_*`, Home | two research areas; the four analysis pages are drill-downs at unchanged addresses; lazy tabs | `tests/test_research_pages.py`; browser QA | consolidated without removing analyses | Batch and the analysis pages keep their own controls; not merged into one page |
| 21 | Performance (prompt §10) | — | `scripts/profile_workflows.py` | profile both trees in fresh processes | table below | no regression; learning +10 % (protocol moments) | no browser-perceived latency measured |
| 22 | Concurrency (QA §3.3) | H: no corruption reproduced | `tests/test_concurrency.py` | concurrent engine calls; two sessions | concurrent results equal serial bitwise; sessions keep their own results | — | no hard per-native-call memory/time limit; public hosting needs isolated workers (U) |
| 23 | Regression convergence hidden (QA §5.6) | S | `methods/learning.py` | ConvergenceWarning recorded (`fit_diagnostics`: converged, n_iter, coefficient ℓ1, condition) plus a note | suite | diagnostics in details | MC refits do not record per-refit diagnostics |
| 24 | Open-source files (QA §3.2) | R: missing | repo root | `SECURITY.md`, `CONTRIBUTING.md`, `CITATION.cff` (placeholder author) | — | added | **License: owner decision required**; Windows/macOS CI not run (U) |
| 25 | Literature grounding (prompt §12) | — | `docs/literature_matrix.md` | matrix of 11 sources plus the preset sources, versions and dates, conflicts, unread parts | — | — | Cai §IV, NIL App. C, Proctor GP details unread |

## Tests run

- **Full suite** (`python -m pytest tests/`, slow tests included, `-p no:randomly`) on the final tree: **904 passed, 0 failed** in 8 min 12 s (`docs/evidence_1.5/pytest_full.log`). The 1.4.2 baseline gave 788 passed and 1 failed.
- **New 1.5 test modules:**
  - `test_qa_v15.py`, `test_io_semantics.py`, `test_presets_cited.py`, `test_experiment.py`;
  - `test_discrepancy.py`, `test_campaign.py`, `test_study_recommend.py`;
  - `test_research_pages.py`, `test_concurrency.py`.
- **Updated test modules:** `test_nil.py`, `test_compose.py`, `test_divergence.py` (the mandatory ledger).
- **Statistical validation:** `docs/evidence_1.5/validation_learning.json`, 3 cases × 6 000 repeats, all passed.
- **Browser QA** (Chromium via Playwright, `docs/evidence_1.5/browser_qa.json` and screenshots). It covered 10 pages at 1440 × 900 and 390 × 844, with these results:
  - 0 app exceptions;
  - no horizontal overflow;
  - Tab moves focus through the navigation links;
  - the Playground Output panel is present.

  The only console errors came from Streamlit's own health probes on sub-paths (404) and from Google fonts blocked by the sandbox (E). Human-perceived latency and touch interaction were not tested (U).

## Performance (fresh process per run, 3 runs, median [min–max] s; peak RSS MB)

| workflow | 1.4.2 | peak | 1.5.0 | peak |
|---|---|---|---|---|
| import app modules | 0.83 [0.82–0.86] | 150 | 0.79 [0.78–0.87] | 150 |
| Home first render (AppTest) | 0.76 [0.75–0.78] | 272 | 0.74 [0.72–0.86] | 272 |
| Playground first / warm rerun (AppTest) | 0.19 / 0.15 | 289 | 0.19 / 0.16 | 290 |
| Playground "Run simulation", defaults | 3.52 [3.46–3.70] | 319 | 3.37 [3.35–3.51] | 319 |
| compare: QFT(4), 6 presets, 100 000 shots | 0.54 [0.50–0.58] | 105 | 0.53 [0.53–0.54] | 105 |
| CDR on QFT(3), 8 × 1 024 training | 1.58 [1.57–1.61] | 224 | 1.74 [1.51–1.83] | 224 |
| ZIP export, 6 figures, PNG + SVG | 7.64 [6.59–8.42] | 129 | 7.01 [6.90–7.94] | 116 |
| ZIP export, renderer dead | never returns (QA: killed at 20 s) | — | 0.16–0.23 (crash / no Chrome); ≤ the time limit (hang) | — |

What changed:

- **Speed.** The main gain is the bounded export. The CDR cost rose by about 10 % because it now computes the protocol moments; that is the price of correct numbers.
- **Coverage.** No content was removed to gain speed, and no shot count or repetition count was lowered.

## Findings of the offline pilot (simulation, fake_fez model, 25 base circuits; `reports/pilot`)

Observed in this design. They are associations, not general laws.

- **Regime map.** Over (cell, budget) points:

  | regime | share |
  |---|---|
  | PEC best | 40 % |
  | raw within 10 % of the best | 24 % |
  | linear ZNE best | 18 % |
  | ties | 10 % |
  | Richardson ZNE best | 6 % |
  | NIL and CDR best | 2 % and 1 % |
  | no feasible method | none in this grid |

  Raw is competitive mostly at small budgets. Richardson ZNE wins mainly at large λ_LC and large budgets.
- **At B = 10⁵ against raw** (median log10 RMSE ratio, 95 % base-circuit bootstrap):

  | method | median ratio | 95 % CI | note |
  |---|---|---|---|
  | PEC | −0.41 | [−0.54, −0.25] | better than raw by > 10 % in 89 % of cells |
  | ZNE-L | −0.30 | [−0.45, −0.15] | |
  | CDR | −0.16 | [−0.32, 0] | |
  | ZNE-R | −0.07 | [−0.24, 0.08] | |
  | FR-R | +0.44 | [0, 0.94] | worse than raw in most cells at this budget |
  | NIL-G2 | — | — | cannot run in 66 of 75 cells at 10⁵: its training needs 147 456 shots |

  VD looks best (−0.59), but it is an ideal-resource upper bound and is excluded from the regime maps and the recommender.
- **Recommender (structural features):**

  | evaluation | median regret on decided rows | best fixed method | coverage |
  |---|---|---|---|
  | grouped CV | 0.10 | 0.14 (on all rows); ≈ 0 on the same decided rows | 40 % |
  | confirmatory subset | 0 | 0.04 | 40 % |

  - Its "use raw" decisions were right (raw truly within 10 %) 52 % of the time in grouped CV.
  - The 90 % prediction intervals covered 84–88 % in grouped CV and on the confirmatory subset, but only 23 % under leave-family-out. Intervals are not trustworthy across families.
  - Normalized features cut leave-width-out regret from 0.46 to 0.18. That is weak evidence, from one held-out width, that normalization helps across widths.
  - **Conclusion:** the baseline recommender does not yet beat the best fixed method on grouped folds. Its value in this pilot is its abstention and its honest "use raw".
- **Discrepancy and rankings (emulation).** Kendall τ between the predicted and the observed method orderings per circuit had the same distribution under the null and under both alternatives (median ≈ 0.67). At this budget, the injected mismatch did not reorder RAW/RO/ZNE rankings in a detectable way. It is evidence about the emulated effect sizes only.

## Claims and their support

- **Supported:**
  - the corrected learning moments agree with independent experiments within MC uncertainty in the three audit cases;
  - the readout γ, validation, export, budget and guide fixes;
  - the I/O semantics;
  - the cited-preset reproductions in the app's convention;
  - UI = YAML = CLI;
  - calibrated discrepancy tests under matched-model emulation;
  - campaign admission logic under a mock provider.
- **Not claimed:**
  - any hardware transfer (no hardware data);
  - novelty of FR (it is this project's heuristic; Liao et al.'s ML-QEM is the closest published relative);
  - causal identification of error mechanisms;
  - a general optimal-QEM recommender;
  - publication readiness;
  - validity of the pilot's regime shares beyond its design.

## Unresolved

1. **License:** choose one before any public release (owner decision).
2. **Hardware campaign:** not executed. The plan is in `docs/hardware_preflight.md`.
3. **Production study:** the scalable study (`direction_A_v2_study.yaml`, 300 cells) is planned but not run here.
4. **Hosting:** no hard per-call limits for native code; multi-user hosting needs process isolation.
5. **CI:** Windows and macOS were not run.
6. **Recommender:** the uncertainty of its intervals across families is poor; better models or more data are future work.
