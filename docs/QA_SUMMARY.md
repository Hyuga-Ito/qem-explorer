# QEM Explorer 1.9.1 — verification summary

What was run for the corrective release 1.9.1, with the numbers it produced, and what was not verified. All runs
are local: Ubuntu 24.04, glibc 2.39, headless, two CPU cores shared between jobs, `GLIBC_TUNABLES=glibc.rtld.
optional_static_tls=2048` and one thread per math library (as the CI does). The 1.9.0 summary follows below.

* **GitHub Actions was not run** (nothing was pushed, tagged or published).
* **Native Windows was not run** (no Windows computer was available). The Windows guide's Python commands were run on
  Linux in paths with spaces and Japanese characters; its PowerShell-only lines were not.
* **No expensive study was run:** not fulldev19, not the reserved confirmation conf19, not the full pilot again; no
  circuit was sent to quantum hardware.

The release candidate is commit `1871049`; the release commit differs from it only in documents (this file, the
Windows guide's status notes, the release notes); the code files of the two source archives are byte-identical
(checked from their inventories).

## A. The findings: reproduced on 1.9.0, absent from 1.9.1

Probes in the private evidence archive (`probes/`), run on a worktree of the 1.9.0 release commit `615fd5d` and on
1.9.1 with copied pilot labels or tiny generated studies:

| probe | 1.9.0 | 1.9.1 |
|---|---|---|
| F1 shipped fulldev order, grid without the old default (copied pilot labels) | exit 2 at `freeze-complete`; nested of `hgb\|deployable+simulated`, frozen `ridge\|size_budget` | exit 0; nested of the frozen configuration; τ = 0.1079260780570268 (the review's corrected-order value) |
| F1 the review's order (analyze → freeze → nested → freeze-complete) | exit 2 | exit 0, same τ |
| F1 recovery of a run 1.9.0 stopped (tiny study) | — | plain rerun exit 2 ("--accept-changes", nothing moved); with `--accept-changes` exit 0; reruns exit 0; frozen policy unchanged |
| F2 grid changed to ridge/size_budget without re-analysis | exit 0, chose `hgb\|deployable18+simulated` | exit 2 with the problems listed |
| F3 cap between current-record seconds (1.69 s) and attempt seconds (4.81 s), with `--override-limits` | relabeled the quarantined task, exit 0 | ceiling stops before any start, exit 1 |
| F4 `s6_costs` wall time | 4.6609 h = 9.3218 / 2 under "7 workers" | 1.3317 h = 9.3218 / 7 for a declared 8-core/7-worker computer, on any computer |
| F5 supervised task inside `unshare --pid --fork` | worker exits 70 at once ("closed its pipe without a result") | task completes; psutil inspection disabled and reported |

## B. Test suites (release-candidate code)

| run | result |
|---|---|
| fast partition (`-m "not slow"`, `QEMX_REQUIRE_TRIALS=selection_v180,pilot_v190`, Python 3.12.3, the lock) | **1788 passed**, 0 failed, 0 skipped (12 min 48 s) |
| slow partition (`-m slow`, same) | **64 passed**, 0 failed, 0 skipped (7 min 22 s) |
| `tests/test_fixes_v191.py` (19 regressions, incl. slow) | passed in every run above and below |
| Python 3.10.20 and 3.13.16: `test_fixes_v191` (incl. slow), `test_readiness_v190`, `test_pipeline_v190`, `test_text_io` | 87 passed each |
| process tests inside a simulated PID namespace (`unshare --pid --fork`): parent kill, Ctrl+C/SIGTERM with lock release, live/dead locks, pipeline termination | passed (and by both independent agents) |
| legacy locale `LC_ALL=en_US.ISO-8859-1` (no UTF-8 mode): `test_text_io`, `test_fixes_v191`, `test_pipeline_v190` (fast) | 27 passed, **1 failed**: `test_dry_run_uses_argument_arrays_with_unicode_and_spaces` — the operating system cannot create a directory named with Japanese characters under a Latin-1 file-system encoding (`UnicodeEncodeError` in `os.makedirs`); the same failure occurs on 1.9.0; text *contents* are read and written as UTF-8 in every test. Windows file names are Unicode, so this does not apply there |

## C. Fresh extraction (release-candidate archive) into `…/展開 テスト/qem-explorer-1.9.1`

New virtual environment from `requirements-lock.txt` (Python 3.12.13; install 76 s), then:

| step (commands of `docs/WINDOWS_FIRST_RUN.md`) | exit | time |
|---|---|---|
| focused fast tests (`test_text_io`, `test_readiness_v190`, `test_pipeline_v190`, `test_fixes_v191`) | 0 (76 passed, 2 skipped: need the pilot's trial arrays, not extracted here) | 72 s |
| `test_fixes_v191.py -m slow` | 0 (6 passed) | 89 s |
| smoke19 `check-config`, `plan --tasks-csv`, `pipeline … --workers 2`, unchanged rerun, `status` | 0, 0, 0, 0, 0 | pipeline 274 s, rerun 30 s |
| Ctrl+C (SIGINT) 8 s into `label pilot19 --limit 4 --workers 2`, then the same command | first: KeyboardInterrupt (exit −2), two `interrupted` attempts (4.56 s, 4.57 s), lock released; resume exit 0; ledger 6 attempts, 0.025 h | 9 s + 41 s |
| `plan pilot19 --out` that partial run | 0; "already consumed 0.03 h … projected 0.30 h" | 1 s |
| full-study command pattern on smoke labels: `pipeline --until label`, `analyze --splits same-distribution --extras=`, `pipeline`, rerun | 0, 0, 0, 0 | — |
| fulldev19 `check-config`, `plan --tasks-csv`, `plan --measured reports/pilot_v190/dev`, `pipeline --dry-run` | 0, 0, 0, 0 | seconds |
| pilot19 `plan`, `pipeline --dry-run` | 0, 0 | seconds |

(A first, scripted attempt to send SIGINT to a background job was ignored by design — POSIX shells start background
jobs of a script with SIGINT ignored — so the labels finished normally; its logs are kept as
`scripted_sigint_ignored_*`. The Ctrl+C result above comes from a driver that sends SIGINT to a foreground-style
child, as a terminal does.)

## D. Runs made with 1.9.0, continued with 1.9.1

* smoke19 completed by 1.9.0, then the 1.9.1 pipeline twice without options: exit 0, 0; every stage verified and kept
  (`[nested] kept: the freeze is completed …`); `status` 0.
* A full-order run stopped by 1.9.0 (exit 2): see table A.

## E. Historical data and figures

`python -m qemx.selection.tables19 --v18 reports/selection_v180 --audit reports/execution_audit_v190 --pilot
reports/pilot_v190`: tables s1–s5 (CSV and TeX), Figure 1's PNG and CSV, `tables_v19.tex` are byte-identical to the
shipped files; the figure's PDF/SVG differ only in their embedded creation dates (the shipped files are kept);
`s6_costs.*`, `tables_v19.md/.pdf` and `provenance.json` change (F4). The trial-data tests (`test_trial_data_v180`,
`test_trial_data_pilot19`, required, not skipped) pass in the partitions above: the 1.8.0 and pilot arrays and their
recomputed contrasts are unchanged.

## F. Independent debugging

Two sub-agents on separate copies — Sonnet 5.5 (`claude-sonnet-5-5`) and Fable 5.1 (`claude-fable-5-1`), identities
as each reported — reproduced F1–F5 on 1.9.0, confirmed the fixes, recomputed 9.3218 CPU-h, 1.3317 h, the split parts
(5/35/15/43/1/5) and fits (334 611; 51.25× the pilot's fits × rows) and the pilot ledger (3114.249 s), and found ten
defects in the candidate (one medium in accounting, two medium in orchestration, the rest minor), all fixed with
regressions and confirmed fixed on their re-check (`docs/ISSUE_EVIDENCE_MATRIX.md`, part 1.9.1).

## G. Not verified

Native Windows (Ctrl+C, Ctrl+Break, closing the window, parent death, file locking by other programs) and the CI's
`windows-smoke` job; macOS (the PID-view check was changed for its framework builds without a Mac); the full pilot
(re-run not needed: its stage order is the one tested on small studies, and its data are unchanged); fulldev19's
analysis time; the streamlit application (no UI code changed in 1.9.1).

# QEM Explorer 1.9.0 — verification summary

What was run for 1.9.0, with the numbers it produced, and what was not verified. All runs are local: Ubuntu 24.04,
glibc 2.39, headless, two CPU cores shared between jobs. Every Linux run sets
`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048` and one thread per math library, as the CI does.

* **GitHub Actions was not run** for 1.9.0 (nothing was pushed; the remote `main` is still 1.7.0).
* **Native Windows was not run.** The Windows commands were executed on Linux, in paths with spaces and Japanese
  characters.
* **No circuit was sent to quantum hardware.**

The 1.8.0 verification summary is in the 1.8.0 archive and in the Git history of this file.

The release candidate is commit `8f4b0a0`. The release commit differs from it in two ways only (section 7):
* this file;
* a docstring escape in `qemx/selection/tables19.py`.

## 1. The review findings: reproduced on 1.8.0, absent from 1.9.0

`tools/repro_findings.py` (private audit archive) runs the same probes on the extracted 1.8.0 release and on 1.9.0:

| probe | 1.8.0 (`evidence/repro_baseline.json`) | 1.9.0 (`evidence/repro_v190_final.json`) |
|---|---|---|
| N1 timeout, then an unchanged resume | resume exit 0, record still `timeout` | first and resume exit 1, failure reported |
| N1b `run` after a labeling failure | exit 0, analysis ran | exit 1, analysis not run |
| N2 caps `disk_gb`/`cpu_hours` = 10⁻¹² | labeled, exit 0 | `plan` exit 3, `label` exit 2, nothing written |
| N3 once-only steps on a complete copy of the 1.8.0 run | `freeze` → 1 (`set -e` stops the script); `|| true` masking | `freeze --reuse`, `freeze --complete --reuse` → 0 after verification; the script is the pipeline; no masking |
| N4 replay of a run with a deleted task | 9 rows, nothing excluded | refused (`IncompleteRun`) |
| N4b fabricated `labels.csv` alone | accepted | refused (no validated records) |
| N6 typo, duplicates, empty lists, negative seed, `per_half > R`, no development tasks | all accepted | all refused (schema or `design.check`) |
| N8 explicit device vs state backend | state backend used | explicit device used |
| R1 GHZ-6: study vs state | SV vs PEC-TW, different keys | SV and SV, same key |
| R8 staleness and the simulation switch | not in the comparison | in `recommendation_signature`; the UI compares it |

The v1.7 trial manifest failure (N5: 347 missing files) was reproduced by running the 1.8.0 job's step. Its 1.9.0
counterpart is `test_trial_data_job_unpacks_only_the_requested_studys_attachments`, which executes the workflow's
selection script.

## 2. Test suites

| tree | environment | Python | tests | result |
|---|---|---|---|---|
| fresh extraction of the release-candidate archive with all three trial-data archives, in `…/fresh テスト/qem-explorer-1.9.0` | new virtual environment installed from `requirements-lock.txt` (NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0, Qiskit 2.5.2, Aer 0.17.2, Streamlit 1.66.0, pandas 3.0.2) | 3.12.13 | `pytest -m "not slow"` | **1 773 passed**, 0 failed, 0 skipped (26 min) |
| same | same | 3.12.13 | `pytest -m slow` | **53 passed**, 5 skipped (the 1.7.0 study tests need the 1.7.0 trial-data archive, which belongs to that release) (10 min) |
| source archive alone (no trial data) | release lock | 3.12.3 | trial-data, tables, metadata and 1.7.0-study tests | 45 passed, 25 skipped (each needs per-trial arrays; the reason is printed); with `QEMX_REQUIRE_TRIALS=selection_v180,pilot_v190` they **fail** (19 errors), as the trial-data CI job requires; `verify_trials.py` exits 1 |
| development tree at the release candidate | latest releases of `requirements-dev.txt` (NumPy 2.2.6, scikit-learn 1.7.2, Streamlit 1.65.0) | 3.10.20 | 1.9.0 tests + text I/O, selection core and CLI, release metadata (8 files, not slow) | 104 passed, 5 skipped (trial data not in that tree) |
| same | latest releases (NumPy 2.5.4, scikit-learn 1.9.1, Streamlit 1.66.0) | 3.13.16 | same | 104 passed, 5 skipped |
| development tree with trial data | release lock | 3.12.3 | `pytest -m slow` (before the second-round fixes) and the 1.9.0 slow tests after them | 58 passed; the 5 pipeline tests passed again after the fixes (3 min 44 s) |

The 1.9.0 tests added by this release:
* `tests/test_readiness_v190.py` (48 cases);
* `tests/test_pipeline_v190.py` (5 cases, slow: interruption and resume at every stage in a Japanese path, failure propagation, SIGTERM);
* `tests/test_trial_data_v180.py` (16);
* `tests/test_trial_data_pilot19.py` (3);
* `tests/test_tables19.py` (7).

`scripts/check_text_io.py`: 0 findings.

**Legacy locale** (`en_US.ISO-8859-1`, generated locale: file names `iso8859-1`, text default ISO-8859-1):
* `tests/test_text_io.py`: 13 passed;
* `plan` and `check-config` under `-X warn_default_encoding` with the strict wrapper: exit 0;
* `test_selection_core.py`, `test_experiment.py`, `test_tables19.py` under the wrapper: 49 passed.

**Failures found by these runs and fixed before the release candidate:**
* The 1.8.0 configuration test read the new figure specification as a study configuration. The specification is now excluded by its format.
* The pipeline overwrote its own completion report (second independent round, F1).
* Five rounding or wording errors were found by the fact-check (section 5).

## 3. End-to-end runs (bounded; the full study was not run)

| run | where | result |
|---|---|---|
| smoke19 pipeline, first run | `runs/smoke 19 テスト` (development tree) | all stages, 7.5 minutes on two cores; rerun with every stage verified and kept: 53 s |
| smoke19 pipeline from the fresh extraction | `runs/スモーク テスト` (fresh environment) | `check-config` 0; pipeline exit 0 (9.5 minutes while the test suite shared the cores); rerun exit 0 in 53 s with every stage verified and kept; `status` 10/10 complete |
| pilot19 pipeline | `runs/pilot19` | 54/54 tasks; labels 0.86 CPU-h (27 minutes, two workers), analysis 71 minutes, nested evaluation 20 minutes, bundles 57 s, portfolio 5.5 minutes, profile 1.9 minutes, figures 54 s (stage logs in the private archive); resumed after the label stage with newer code — the complete labels were verified and kept, not relabeled |
| interruption and resume at every stage | `tests/test_pipeline_v190.py` (slow) | freeze and confirmation byte-identical after resume; an edited source table stops the pipeline (exit 2); confirm `--reuse` with another configuration or other training parts exits 2 |
| SIGTERM of the pipeline during labeling | slow test | label command, workers and lock gone within seconds (the previous code left them running) |
| parent killed (SIGKILL) during labeling | fast test; second independent round | the worker ends (test bound 8 s; measured 0.91 s by the independent round); an immediate resume exits 0 |
| execution audit | `runs/execution_audit` | 480 rows, ledgers exact (0 over, 0 unspent); results in `reports/execution_audit_v190/` |
| tables and Figure 1 | `python -m qemx.selection.tables19` | six tables (CSV, LaTeX, Markdown), a 6-page PDF without overfull boxes, Figure 1 (4 panels; viewed); SBS-B column checked against the rule refitted on the development labels: 580/580 |

## 4. Scientific and data checks

* **1.8.0 per-trial data** (`tests/test_trial_data_v180.py`, 16 passed with the arrays):
  * 217 arrays match `TRIALS_SHA256SUMS`;
  * all eight labels directories are complete and bound;
  * the cached confirmation labels equal the tables derived from the arrays;
  * the saved replay rows are scored with the derived labels, and the primary contrasts are recomputed to 10⁻¹²;
  * development and confirmation circuits are disjoint;
  * an edited SBS-B column is refused by the replay script and by the tables;
  * an unverifiable baseline is refused.
* **Pilot per-trial data** (`tests/test_trial_data_pilot19.py`, 3 passed):
  * 54 arrays match `TRIALS_SHA256SUMS`;
  * the run is complete and bound, and its data digest is the one in `freeze.json` and `completion.json`;
  * the shipped labels table equals the table derived from the arrays.
* **Historical identities:** all 1 989 task identities of the 19 earlier configurations are unchanged (`tests/data/expansion_ids_v180.json`). After the LaTeX escaping fix, the 1.8.0 thesis tables regenerate with identical content: all 14 table files (CSV and LaTeX) are byte-identical, and only the release label in the Markdown header and the provenance differs.
* **Ancestry:** smoke19, pilot19, fulldev19 and conf19 are registered (146 new logical circuits). They share no circuit with any other study, and conf19 passes the confirmation check.
* **The second review's corrections**, checked against the code and data:
  * the symmetry loss and the inverted caption are real defects, both fixed;
  * the 47-row caption was correct;
  * fresh-design risk is a defined benchmark, now named as the estimand and measured against executed risk;
  * the historical data are valid;
  * the CI trial-data failure was confined to the optional job.

## 5. Independent verification (what was actually used)

* **Sonnet 5.5** (sub-agent; model `claude-sonnet-5-5` per its transcript), two debugging rounds on separate copies of the code.
  * Round 1 reported NEW-1 … NEW-9, NEW-A and N1/N2 gaps. All are fixed, except NEW-A, which was clarified (an override admits a start; the cap still holds).
  * Round 2 re-checked these fixes and reported 15 further items (F1–F15) and 3 suspicions:
    * fixed with regression tests: F1–F13 and F15;
    * F14 (the `.cmd` join script): fixed, reviewed statically only;
    * S1 (≈ 1 s watchdog window): documented as a limitation.
  * Both reports are in the private audit archive (`evidence/review_agents/`).
* **Fable 5.1** (sub-agent; model `claude-fable-5-1`), an independent fact-check of the research documents against the saved data. It recomputed:
  * every interval;
  * the power calculation;
  * the noise-stratified rule, reproduced exactly at +0.069 [+0.032, +0.108].

  It found 10 items:
  * a missing test file named in the matrix;
  * an 11× (not 3×) analysis-row ratio;
  * seven (not six) fulldev19 noise settings;
  * a disk estimate that ignored missing arrays — a code defect, fixed with a test;
  * an incomplete range;
  * exactness and rounding wording;
  * over-read learning curves;
  * "a quarter" instead of 28 %;
  * a stale YAML count;
  * durations that are verifiable only from the private logs.

  All were corrected; the durations are now attributed to the logs in the private archive.

## 6. Interface

Headless Chromium against the app started from the fresh extraction (fresh environment), *QEM Performance &
Selection → Recommend for my circuit* (`tools/browser_recommend19.py`, screenshots in the private archive):

* with the default deployable bundle, the result shows the effective device, the layouts and the declared symmetry;
* with the frozen bundle, the result appears with no exception and no out-of-date mark;
* switching *Allow simulated features* on marks it out of date, and switching it back removes the mark.

**PASSED.** The rest of the app was not re-checked in a browser for 1.9.0, because only the recommendation view
changed; the page-rendering tests are part of the fast suite.

## 7. Packaged archives

* **`qem-explorer-1.9.0.zip`** — the GitHub-ready source.
  * One root folder `qem-explorer-1.9.0/`, built by `git archive` from the release commit (tracked files only), so run directories, locks, caches and per-trial arrays cannot enter it.
  * Left out, as in 1.8.0: `archive/` (superseded notes and manuals) and the 1.6.0 pilot's arrays and figures.
  * About 2 180 files, ≈ 53 MB.
  * Scan (`packaging/` in the private archive) finds none of the following: symlinks, nested archives, office files, manuals, review inputs, credential, environment or ledger files, caches, machine paths.
  * The only `.npz` files are the six model bundles' arrays.
  * The personal-name scan matches only the repository URL; the secret scan matches only a fake token in a UI test.
* **Trial-data archives.** Each study's arrays are verified against that study's `TRIALS_SHA256SUMS` before packing; extract each archive into the source root.
  * `qem-explorer-1.9.0-trials-1.zip`: 88 development arrays of `selection_v180`.
  * `-trials-2.zip`: 129 arrays (confirmation, regimes, noise composition, root audit).
  * `-trials-pilot.zip`: the 54 pilot arrays.
  * The 1.8.0 arrays are byte-identical to the 1.8.0 attachments.
* **`qem-explorer-1.9.0-private-audit.zip`** — owner only (`README_AUDIT.md` inside).
* **Delivery folder.**
  * Archives over 25 MB are split into `.001`, `.002`, …;
  * `SHA256SUMS.txt` lists every archive and part;
  * `join_qem-explorer-1.9.0.cmd` uses Windows built-ins only;
  * `join_verify.py` is a cross-platform alternative that was executed here;
  * `README_DELIVERY.txt`.
* **Release candidate → release.** Only `qemx/selection/tables19.py` changed: a `\D` in the docstring of `unicode_preamble` raised a `SyntaxWarning`; there is no behaviour change. The archive of that code was checked after extraction into a path with a space and Japanese characters, with the RC's trial data:
  * `scripts/check_text_io.py`: 0 findings;
  * `test_release_metadata.py`, `test_tables19.py`, `test_readiness_v190.py` (not slow), `test_trial_data_v180.py` and `test_trial_data_pilot19.py`: **78 passed**;
  * `verify_trials.py`: 217/217 and 54/54;
  * apart from that file, the tree is identical to the tested release candidate.
* Every Python file of the public tree compiles without `SyntaxWarning`.

## 8. Not verified

* GitHub Actions for 1.9.0. It has not run because nothing was pushed, so all CI statements are local emulations of the workflow's steps.
* Native Windows:
  * PowerShell commands;
  * `CTRL_BREAK_EVENT` stopping of stage commands;
  * the `.cmd` join script, which was reviewed twice statically; the Python join script was executed with complete, missing-part, corrupted-part and existing-wrong-output deliveries.
* The full development study fulldev19, the reserved confirmation conf19 and the historical reproduction of 1.8.0 (days of CPU). None was run, as instructed.
* Hardware. Nothing was submitted.
* The coherent-fraction (`chi`) task cost of fulldev19, which is unmeasured.
* Recommendation runtime above 6 qubits.
