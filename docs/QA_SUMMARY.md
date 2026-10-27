# QEM Explorer 1.7.0 — verification summary

What was run on the release, with the numbers it produced, and what was not verified. All runs are local (Ubuntu
24.04, glibc 2.39, headless, 2 CPU cores shared between jobs). GitHub Actions itself (including its Windows and
legacy-locale jobs) was not run for this release, and no circuit was sent to quantum hardware.

Release tree: branch `v1.7.0`. The test suites of section 1 ran on the *packaged archives* of commit `cd07b53`,
extracted into empty directories — once with the two trial-data archives extracted as well (the complete release)
and once without them (the source archive alone). The later commits change documents, one introductory sentence of
the QEM Performance page, and the validation metadata the `bundle` command writes into the two shipped bundles (their
model arrays are byte-identical; each bundle's validation now describes its own configuration), with a test of it;
the selection command-line, selection-UI, `test_selection_v170.py` and page tests ran again on the final tree in
three Python versions (section 1).

Every Linux run sets `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048` and one thread per math library, as
`.github/workflows/ci.yml` does (README, Linux note).

## 1. Test suites

| environment | tree | Python | Streamlit | scikit-learn | NumPy | `pytest -m "not slow"` | `pytest -m slow` (end-to-end) |
|---|---|---|---|---|---|---|---|
| release lock (`requirements-lock.txt`) | source + trial data | 3.12.3 | 1.66.0 | 1.8.0 | 2.4.4 | **1604 passed** | **54 passed** |
| release lock | source archive alone | 3.12.3 | 1.66.0 | 1.8.0 | 2.4.4 | **1604 passed** | **48 passed**, 6 skipped (they need the per-trial arrays; the reason is printed) |
| latest releases (`requirements-dev.txt`) | source + trial data | 3.12.13 | 1.66.0 | 1.9.1 | 2.5.4 | **1604 passed** | **54 passed** |
| latest releases | source + trial data | 3.13.16 | 1.66.0 | 1.9.1 | 2.5.4 | **1604 passed** | **54 passed** |
| latest releases | source + trial data | 3.10.20 | 1.65.0 | 1.7.2 | 2.2.6 | **1604 passed** | **54 passed** |
| 1.5.0 lock with Streamlit 1.64.0 (page, UI, I/O, selection-tab and text-I/O tests) | source + trial data | 3.12.3 | 1.64.0 | 1.8.0 | — | **63 passed** | |
| final tree (`5134d2d`): selection command line, selection UI, `test_selection_v170.py` and page tests, both partitions | source + trial data | 3.12.3 / 3.10.20 / 3.13.16 | | | | **46 passed** in each | |

Python 3.10 now passes completely (1.6.0 had two test-only failures there). 1.7.0 adds 87 test cases (1658 in
all), among them the selection invariants of `tests/test_selection_v170.py`, the UTF-8 tests of
`tests/test_text_io.py` and the Playground I/O semantics of `tests/test_io_semantics_v170.py`.

Failures found by the first matrix run (on commit `4a5ecbd`) and fixed before the runs above:

- **A defect**: the supervisor's "spawn" children re-ran the parent's `__main__`; after a Streamlit `AppTest` (whose
  generated script becomes `__main__`) the children re-ran that script and failed. Children now never import the
  parent's main script (`supervise._isolated_main`; test `test_supervised_children_never_rerun_the_parents_main_script`,
  which fails without the fix).
- Two tests written for 1.6.0 expected its tables: the device-specification test (now with the 1.7.0 component
  modifiers `g1`, `g2`, `t`) and the Figure 6 consistency test (now on the 1.7.0 equal-budget tables, with the same
  check that plotted differences equal those recomputed from the policy table).
- Two UI tests assumed the per-trial arrays in the shipped study; without them the drill-down test now checks the
  explanatory message and the recommendation-tab test is skipped with its reason.

## 2. UTF-8 text I/O and platforms

| check | result |
|---|---|
| static checker (`scripts/check_text_io.py`: every text `open`, `read_text`/`write_text`, `csv`, `json`, `yaml`, subprocess text pipes and `pandas` text I/O name `encoding="utf-8"`; binary I/O untouched; no `errors="ignore"`) | 0 findings on the release tree |
| runtime tests (`tests/test_text_io.py`): Japanese, Greek and mathematical text in configurations, experiment documents, state files, CSV/JSON/YAML/Markdown outputs and figure notes; non-ASCII and space-containing paths; a UTF-8 BOM in a user-edited configuration; undecodable input raises an error naming the byte position | pass in every environment of section 1 |
| child processes with UTF-8 mode off (`PYTHONUTF8=0`) and `EncodingWarning` as an error (`-X warn_default_encoding`): project text I/O and the command line's printed UTF-8 | pass (part of the suites) |
| the text-I/O and Playground I/O tests with a real ISO-8859-1 locale generated locally (`QEMX_LEGACY_LOCALE=en_US.ISO-8859-1`; the child's preferred encoding checked to be ISO-8859-1), on the extracted release | 46 passed |
| Windows | **not run**: the CI job (`windows-latest`, supervised labeling, the recommender, a Unicode path) is configured, not executed |
| Python 3.10 | the two 1.6.0 test-only failures are fixed (`tomli` fallback; the recomputed-feature tolerance) — see the 3.10 row above |

## 3. Scientific and data checks

| check | result |
|---|---|
| Corrected labels against the 1.6.0 labels of the same tasks and seeds (`scripts/compare_design_estimands.py`) | the six protocols without a training design are identical trial for trial on 120/120 development, 21/21 regime and 35/36 historical-final tasks (one ZNE-R allocation differs by one shot at a near-tie); CDR/NIL changes are reported in `RESULTS.md` §2 |
| Design-level standard error | a simulation test recovers the true between-design SD; the trial-level SE is less than half of it there (`test_design_level_standard_error_contains_the_between_design_variance`) |
| Nested selective evaluation | perturbing a fold's labels leaves its threshold unchanged (`test_nested_threshold_of_a_fold_never_depends_on_that_folds_labels`) |
| Ancestry | the confirmation set shares no logical circuit with any earlier study; `confirm` refuses an overlap (tests) |
| Run binding | incompatible resumes, stale files and a wrong freeze are refused (tests); every resume after a container restart during the pilot was recorded (`run.json`, `provenance.json`) |
| Ledgers | every allocation spends exactly B in integers (asserted at labeling, tested per protocol and budget); the hardware plan charges every run exactly B (a test, after a correction: section 6 of `REVIEW_RESPONSE_1.7.md`, N7) |
| Figures | all 15 figures and 6 tables of `reports/selection_v170` drawn from saved tables; `scripts/validate_figures.py`: every SVG parses, every PDF has its pages and text (`pdfinfo`, `pdftotext`), every PNG verifies, every note has its methods box, 31 plotted numbers equal the analysis tables — all checks passed; every figure was viewed after drawing |
| Regeneration from saved tables | from a fresh extraction of the archives, the shipped corrected pilot's figures and tables were deleted and regenerated from its saved tables: 73 source-data, note and table files compared with the shipped ones, 0 differences (numbers to 1e-9 relative; notes except the recorded command line); the validator passes on the regenerated copy |
| Emulator against an independent shot-by-shot execution | rerun on the release (fresh extraction): 2 tasks × RAW, RO, ZNE-L, ZNE-R, PEC at B = 2 000, 200 emulated against 200 directly executed experiments each — all 10 comparisons within |z| ≤ 4 for the mean (largest 2.95), variance ratios 0.85–1.21 (smallest p = 0.18). The six protocols without a training design are trial-identical to 1.6.0, whose labels were validated the same way on the snapshot and on a snapshot with every error rate × 2 |
| Echo diagnostic's null | false-positive rate 4.8 % at α = 0.05 (exact binomial law; `mismatch/null_check.csv`) |
| Smoke workflow (`smoke.yaml`) | 20 tasks labeled (0 errors), analysis, 11 figures drawn, the 5 that need the regime, mismatch or profile studies listed as pending; 23 min on the shared machine |

## 4. Interface

- Streamlit `AppTest`: the *Protocol selection study* tab lists the corrected study first, renders its policy view,
  drill-down (with the per-trial arrays: task, circuit drawing, settings and allocation, shot ledger, trial
  distribution, provenance; without them: an explanatory message) and figures; the *Recommend for my circuit* tab
  recommends with a bundle exactly as `python -m qemx.selection recommend` does and marks a result out of date when
  the budget changes (`tests/test_selection_ui.py`); the Playground's Output panel separates input state, output state
  before measurement, measured distribution and observable (`tests/test_io_semantics_v170.py`).
- Headless Chromium against the app served from the extracted archives: every page renders without an exception and the
  Playground shows the 1.7.0 Output panel; the *Protocol selection study* tab selects the corrected study first and
  renders its policy view, drill-down (shot ledger and trial distribution, the trial data being extracted) and
  figures; *Recommend for my circuit* returns a recommendation for the default experiment (the deployable bundle:
  PEC at B = 10⁵, with the protocol definition, the predicted risks and the feature costs). The first run of the two
  browser scripts failed on the scripts themselves (the Playground check still looked for the 1.6.0 wording, and the
  recommendation was read before it had finished); the corrected scripts passed against the same extracted tree
  (screenshots in the audit material).

## 5. Packaged archives

Three public archives and one private one, each with a `.sha256` file:

| archive | content | size |
|---|---|---|
| `qem-explorer-1.7.0.zip` | the source tree (one root folder `qem-explorer-1.7.0/`, from `git archive` of the release commit): code, configurations, tests, examples, documentation, the shipped bundles, the corrected pilot's records, tables and figures; without the per-trial arrays and without the 1.6.0 pilot's arrays and figures | ≈ 26 MB, ≈ 1 400 files |
| `qem-explorer-1.7.0-trials-1.zip`, `-trials-2.zip` | the corrected pilot's 347 per-trial arrays (development and historical final; confirmation, regimes, noise composition, mismatch), paths relative to the repository root, verified by `reports/selection_v170/TRIALS_SHA256SUMS` | ≈ 16 MB each |
| `qem-explorer-1.7.0-audit.zip` (private) | the review as received, patches 1.6.0 → 1.7.0 per commit, test logs, fresh-extraction logs and screenshots, run logs, the design study's halves, the 1.6.0 figures and results document, the tools used to build and check the release | — |

The source archive's staged tree is scanned before zipping (no symlinks, nested archives, office files, manuals,
review material, pickles, ledgers, environment or key files, files over 5 MB, caches, absolute sandbox paths,
personal names or e-mail addresses, or secret-like strings; the only matches are digests in provenance records and a
test's fake token); file times are fixed and the file list is sorted, so the archive is reproducible from the commit.

**Fresh extraction** (`tools/fresh_check17.sh` in the audit material), all steps on the archives of commit
`cd07b53` in an empty directory with a new virtual environment from the lock file: archive and inventory checksums;
extraction of both trial-data archives with every array's checksum verified; installation from the lock (Python
3.12.3); import (`qemx` 1.7.0, Streamlit 1.66.0); the static UTF-8 check (0 findings); app start-up (health check) and
the headless browser checks of section 4; the four example documents validated and one run; the 1.5 study commands;
the bundled 1.5 pilot report reproduced (`regimes.csv` identical); `plan` of the larger study; the shipped corrected
pilot's figures and tables regenerated and compared (0 differences in 73 files) and validated; `recommend` with both
bundles (with and without simulated features); `hardware plan` (same digest as the shipped plan); the emulator
validation of section 3; the smoke workflow; the link check (0 broken). Every step passed except the first run of the
two browser scripts, explained in section 4.

## 6. Not verified

- GitHub Actions on GitHub's runners, the Windows job, macOS.
- Any IBM Quantum service call and any hardware result; the hardware plan is prepared and refused without the owner's
  authorization; the study is emulation on fake devices only.
- The larger study `full17.yaml` (resource estimate only, `RESULTS.md` §14).
- CDR, NIL and exponential ZNE against an independent shot-by-shot implementation: the emulator validation covers
  RAW, RO, ZNE-L, ZNE-R and PEC; the learning protocols reuse the methods' own finite-training samplers, which 1.5.1
  validated against complete end-to-end experiments (`scripts/validate_learning_moments.py`, not rerun).
- The finite-sample coverage of the CV-residual intervals (no guarantee); only their empirical coverage on held-out
  circuits is reported. The split-conformal intervals have a marginal guarantee under exchangeability of circuits,
  which the pilot's circuit sample does not establish.
- Byte identity of PDFs across environments (not claimed).
