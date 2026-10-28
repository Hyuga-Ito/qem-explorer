# GitHub Actions failures of 1.7.0 — analysis, reproduction and repair (1.8.0)

This document covers the failed `tests` workflow runs of commit `e68ea55` ("QEM Explorer 1.7.0", the owner's upload of
the 1.7.0 archive). It also covers the repairs made in 1.8.0 and what remains unverified.

## Inputs actually examined

There were two workflow runs on 2026-10-22, both for commit `e68ea5597f15889a416cdcd64605ad980802e3d8`. The owner
supplied the complete job logs of both as `logs_107753736389.zip` and `logs_107753868476.zip`. The repository's
`git ls-remote` shows `main` and `refs/tags/v1.7.0^{}` both at `e68ea55`, so the two runs belong to the same commit.

| Run | Trigger | Jobs (7) |
|---|---|---|
| 107753736389 | push to `refs/heads/main` | locked, latest (3.10, 3.12, 3.13), oldest-streamlit, text-encoding, windows-smoke |
| 107753868476 | push of `refs/tags/v1.7.0` | the same seven jobs |

The logs contain the runner image versions (`ubuntu-24.04` 20261012.334.1 and 20261018.339.1; `windows-latest`),
the resolved packages, every step's commands and tracebacks, and the exit codes. They do not contain the runners'
CPU models, so where a cause depends on the CPU this is stated as an inference, backed by a local reproduction.

## Result per job

| Job | Run 107753736389 (main) | Run 107753868476 (tag) | First substantive failure |
|---|---|---|---|
| locked (Python 3.12.15, release lock, NumPy 2.4.4) | **failed**: 1 failed, 1603 passed | **failed**: 1 failed, 1603 passed | `test_rec_context.py::test_live_noise_features_come_from_the_selected_device_not_the_injector` |
| latest 3.12 (NumPy 2.5.4) | passed (1604 + 48 passed, 6 skipped) | **failed**: 1 failed, 1603 passed | the same test |
| latest 3.10 (NumPy 2.2.6) | passed | passed | — |
| latest 3.13 (NumPy 2.5.4) | passed | passed | — |
| oldest-streamlit (1.64.0) | passed (15) | passed (15) | — |
| text-encoding (ISO-8859-1) | **failed**: 4 passed, 5 errors | **failed**: 4 passed, 5 errors | fixture `unicode_dir` of `tests/test_text_io.py`: `UnicodeEncodeError: 'latin-1' codec can't encode characters` in `os.mkdir` |
| windows-smoke | passed (100 passed, 3 deselected) | passed | — |

There were no installation, collection, timeout or runner failures. When a step fails, the steps after it in the
same job are not run. So the locked job's slow tests and figure validation were not executed on GitHub, and neither
was the text-encoding job's last step (selection and experiment workflows with `EncodingWarning` as an error).

## Failure 1: an exact comparison of a recomputed `lam_lc`

**What the log shows.** The comparison was `{'lam_lc': 0.02828271497755281} != {'lam_lc': 0.028282714977555273}`.
That is a relative difference of 8.7e-14. `lam_ro`, `scale` and `coherent_fraction` were identical. The second value
is stored in `tests/data/pilot_cells/` and was produced on the development machine. The first was recomputed on the
runner.

**Cause (reproduced exactly).**

- `lam_lc` sums the Pauli-twirled fault probabilities of the light-cone locations.
- Each probability is an inverse Walsh–Hadamard combination of PTM eigenvalues that are 1 − O(10⁻⁴)
  (`qemx/noise.py::pauli_probs_from_eigs`). The subtraction turns rounding in the eigenvalues into a relative error
  about 700 times larger in each small probability.
- NumPy dispatches different SIMD kernels at run time: AVX-512 on the development machine (an Intel Xeon with
  AVX512F/ICL/SPR), AVX2 on runners without AVX-512.
- With `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` the local release lock gives **exactly
  0.02828271497755281**, the CI value.
- OpenBLAS kernel selection alone (`OPENBLAS_CORETYPE` = Haswell, Zen, SkylakeX) does not change the value.
  SandyBridge changes it by 2.0e-15 and Prescott by 5.1e-14.
- The same emulation over the whole fast suite of unmodified 1.7.0 gives **1 failed, 1603 passed**, the same test as
  on GitHub (`evidence/ci/fast_noavx512_v170.log` in the audit archive).

**Why the result differed between the two runs.** The `latest 3.12` job passed on main and failed on the tag,
although both runs used NumPy 2.5.4 and the same image version. GitHub assigns hosted runners of one label to
different CPUs. This is consistent with an AVX-512 runner in one run and an AVX2 runner in the other. It is an
inference: the logs do not record the CPU, which is why every 1.8.0 job now prints it.

**Is the difference physical?** No.

- Random one-ulp perturbations of every PTM eigenvalue move `lam_lc` by up to 7.9e-14 relative. That is the size of
  the CI difference.
- Sixteen-ulp perturbations move it by up to 1.2e-12.
- A 1e-6 relative change of the device's error rates moves `lam_lc` by 1.0e-6, a million times more.
- The value was correct on both machines. Only the test's exact equality was wrong.

**Fix** (`tests/test_rec_context.py`, smallest sound change):

- The test's real invariant is that "the injector is ignored". It is now asserted *bitwise* among the three injectors
  on the same machine (same inputs, same process).
- Agreement with the stored record is asserted within `LAM_LC_REL_TOL = 1e-12`. `lam_ro`, `scale` and
  `coherent_fraction` stay exact.
- A new test, `test_lam_lc_tolerance_is_rounding_level_not_physical`, measures the justification each time: one-ulp
  eigenvalue perturbations stay below a tenth of the tolerance, and a 1e-6 change of the error rates exceeds it by
  more than 1e5.
- An earlier comment attributed the effect to pairwise summation order. It is corrected.
- The computation itself is unchanged. Scientific outputs were not altered to make equality hold.

## Failure 2: Unicode directory names under a Latin-1 locale

**What the log shows.**

- The job sets `PYTHONUTF8=0` and installs `en_US.ISO-8859-1`.
- It runs the test file with `LC_ALL=en_US.ISO-8859-1`.
- The fixture creates `データ αβγ ∑ résumé` and `os.mkdir` raises `UnicodeEncodeError: 'latin-1' codec can't encode
  characters in position 63-65`.
- Five tests that use the fixture error at setup. The four that do not, pass.

**Cause.** File *names* and file *contents* are encoded independently.

- On Linux, Python's file-system encoding follows the locale. Under ISO-8859-1 it is `iso8859-1`, and no program can
  create a Japanese or Greek file name there.
- On Windows, file names are always UTF-16 and Python's file-system encoding is UTF-8, whatever the ANSI code page.
- The fixture therefore tested something no Linux Latin-1 system supports. The project's contract, explicit UTF-8 for
  file contents, was never at fault.
- 1.7.0's local legacy-locale check had not reproduced the CI's parent-process locale (its 46 passes did not cover
  this case). Under `LOCPATH` with the locale compiled, the local run now gives **4 passed, 5 errors**, identical to
  CI.

**A latent failure found while reproducing.**

- The next step of that job (never reached on GitHub) ran `python -X warn_default_encoding -W error::EncodingWarning`.
- `-W error::EncodingWarning` also turns third-party warnings into errors. For example `dill` (imported by Qiskit)
  opens `os.devnull` without an encoding at import time.
- Locally the step fails with an EncodingWarning raised inside `dill`. The command-line `-W` form only matches exact
  module names, so it cannot be restricted to the project.

**Fix.** No test was skipped and no check was disabled.

- **Test fixture** (`tests/test_text_io.py`):
  - The fixture uses the richest non-ASCII name the interpreter's file-system encoding can represent: the full
    Japanese/Greek/math names everywhere they are representable, Latin-1 accented names (`données àéî résumé`, …)
    under a Latin-1 locale. It never falls back to ASCII; if even Latin-1 is unavailable it fails with a diagnosis.
  - `test_path_names_are_not_degraded_where_representable` guarantees the full names on every UTF-8 system and on
    Windows.
- **New test `test_unicode_file_names_with_a_legacy_text_encoding`** covers the actual Windows situation: UTF-8 file
  names with a legacy *text* default. On Linux the child process starts under `C.UTF-8`, which fixes the file-system
  encoding, then switches only `LC_CTYPE` to ISO-8859-1. Project configuration, labels and experiment documents are
  then written and read in Japanese/Greek paths with `EncodingWarning` as an error. On the Windows job it runs as it
  is (cp1252 text default, UTF-16 names).
- **`qemx/textio.py`** raises `PathEncodingError` for a file name the locale cannot represent. The error names the
  file-system encoding and the remedy (a UTF-8 locale), and is distinct from `TextEncodingError` (contents that are
  not UTF-8).
- **`scripts/strict_encoding.py`** turns `EncodingWarning` into an error for *project* modules only. Third-party
  warnings stay visible and are never silenced. Its own test checks both directions. Under the real Latin-1 locale the
  workflow step's three commands now pass locally (plan, validate, and 42 tests).

## Workflow changes (`.github/workflows/ci.yml`)

| Change | Reason |
|---|---|
| Every job prints CPU, NumPy SIMD dispatch, BLAS core (`scripts/ci_runner_info.py`); Windows also prints file-name and text encodings | the 1.7.0 logs could not attribute a CPU-dependent failure |
| `locked`: the CPU-sensitive tests are rerun with NumPy's AVX-512 dispatch disabled when the runner has it | both SIMD paths are exercised on every run, whichever runner GitHub assigns |
| Later test steps run after an earlier failure (`if: ${{ !cancelled() }}`); the job still fails | one failure no longer hides the next (the slow tests and the strict-encoding step never ran in 1.7.0) |
| JUnit XML uploaded on failure (`actions/upload-artifact@v4`) | failure evidence without re-running |
| Windows steps set `$PSNativeCommandUseErrorActionPreference = $true` | PowerShell otherwise ignores a failing native command that is not the last of a multi-line step |
| `text-encoding` checks that the legacy locale is really active, and uses `scripts/strict_encoding.py` | the locale is verified, and project code is strict without failing on dependencies |
| `actions/checkout@v5`, `actions/setup-python@v6`, `cache-dependency-path` | Node 20 deprecation warnings in every 1.7.0 job; meaningful cache keys |
| `timeout-minutes` per job | bounded runs |
| Optional manual job `trial-data` (workflow_dispatch input `trials_tag`) | runs the six tests that need a release's per-trial arrays, downloaded from that release with the workflow's own read-only token; skipped by default |

The workflow uses no credentials, no QPU and no network beyond package installation (and the optional download). The
push and the tag run both remain: they validate the branch and the release tag. No test was deleted, no assertion
weakened beyond the documented `lam_lc` tolerance, no `continue-on-error` and no hidden exit codes.

## Local verification

These results were produced in the development container, not on GitHub; see `docs/QA_SUMMARY.md` for the full
matrix and the counts.

- The release lock (Python 3.12.3) passes both affected files natively and with
  `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`.
- `tests/test_text_io.py` passes under the default UTF-8 locale and under a real ISO-8859-1 locale
  (`LOCPATH` + `LC_ALL=en_US.ISO-8859-1`, `PYTHONUTF8=0`).

## A third failure found locally before it reached CI

Running the suite under the CI's other Python versions (`docs/QA_SUMMARY.md`) found one more failure, in the new
test of this release: the child process of `test_unicode_file_names_with_a_legacy_text_encoding` called
`locale.getencoding()`, which exists only from Python 3.11, so the `latest (3.10)` job would have failed. The child now
falls back to `locale.getpreferredencoding(False)` (the same value with UTF-8 mode off) on Python 3.10; the test
passes on 3.10, 3.12 and 3.13, and under the ISO-8859-1 locale.

## What remains unverified

- **The GitHub Actions status of 1.8.0 is pending.** No workflow has run on this code; a local pass is not a green
  remote run.
- The runner CPU models of the two 1.7.0 runs are unknown (inferred from the reproduction, not observed).
- The `windows-smoke` job passed in 1.7.0; the 1.8.0 Windows steps have not run on a Windows machine.

### After uploading, check

1. Actions → *tests* → the run for the commit: all six standard jobs green. `trial-data` is skipped unless dispatched.
2. In `locked`, the step *Runner (CPU, …)* names the CPU and the dispatch targets. The step *Floating-point
   portability* either reruns three test files with AVX-512 disabled or says the runner has none.
3. In `text-encoding`, the first line of *Text I/O tests under ISO-8859-1* reads `file names: iso8859-1  text default:
   ISO-8859-1`, and `test_unicode_file_names_with_a_legacy_text_encoding` passes.
4. In `windows-smoke`, *Runner* prints `file names: utf-8  text default: cp1252`.
5. Optionally, run the workflow manually with `trials_tag` set to a release whose attachments include the trial-data
   ZIPs.
