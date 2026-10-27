# Release consistency — QEM Explorer 1.9.0

## 1.9.0

* **Version.** `qemx/__init__.py` (`__version__ = "1.9.0"`) remains the only source; `pyproject.toml` reads it;
  `CITATION.cff` (version 1.9.0, date 2026-10-25), the first lines of `README.md` and `docs/RELEASE_NOTES.md` and the
  header of `requirements-lock.txt` agree (`tests/test_release_metadata.py`). Pins are unchanged from 1.8.0; a full
  archival snapshot of the tested Linux environment is in `docs/environment/freeze_py312_linux.txt`.
* **GitHub state (read, not changed).** `main` is 1.7.0 (`e68ea55`), tags `v1.5.0` and `v1.7.0`. 1.8.0 was never
  pushed, so updating GitHub with 1.9.0 also brings every 1.8.0 change; the 1.8.0 checklist below applies with 1.9.0
  names. Neither release has a GitHub Actions run yet.
* **New in the tree.** `configs/pipelines/`, `configs/selection/{smoke19,pilot19,fulldev19,conf19,figure_spec19}.yaml`,
  `reports/{pilot_v190,execution_audit_v190,tables_v190,corrections_v190}/`, `reports/TRIAL_ATTACHMENTS.json`,
  `qemx/selection/{integrity,resources,pipeline,formats,execution_audit,tables19}.py`, the 1.9.0 tests and documents.
  No tracked file of 1.8.0 is removed; `reports/selection_v180/` is unchanged byte for byte (its figure note with the
  inverted wording is kept; the correction is beside it).
* **Not in the public archive.** Run directories (`runs/`), locks (`.qemx-run.lock`), caches, per-trial arrays (in the
  trial-data archives), review inputs, the independent debugging report, logs and reproduction evidence (in the private
  audit archive).
* **Attachments.** `qem-explorer-1.9.0-trials-1.zip`, `-trials-2.zip` (the 1.8.0 study's arrays, unchanged) and
  `-trials-pilot.zip` (pilot19); `reports/TRIAL_ATTACHMENTS.json` maps each to its study for the optional CI job.

## 1.8.0 audit (historical)

This note records the release-consistency audit of 1.8.0: what was checked across the repository (application
configuration, editor configuration, README and installation instructions, dependency declarations, workflows, package
metadata, examples and study configurations), what was updated, what was checked and deliberately left unchanged, and
how to update the GitHub repository without leaving stale files behind.

## The "QEM Explorer 1.5.0" labels in the GitHub file listing

GitHub shows, beside every file and folder, the message of the **last commit that changed it**. `.streamlit/` showed
"QEM Explorer 1.5.0" because its content has not changed since that commit; it is not a statement about the content.
The audit therefore inspected contents, not labels:

* `.streamlit/config.toml` is valid and current: every key (`theme.*`, `client.showSidebarNavigation`,
  `client.toolbarMode`, `client.showErrorDetails = "none"`, `server.headless`, `server.runOnSave`,
  `server.maxUploadSize = 20`, `server.enableXsrfProtection`, `server.enableCORS`, `runner.fastReruns`,
  `browser.gatherUsageStats`) exists, undeprecated, in Streamlit 1.66.0 (the lock) and the supported range
  1.64–1.66 (checked against Streamlit's option registry); XSRF protection with CORS enabled is the consistent
  combination. **Left unchanged** — so its listing will keep showing the 1.5.0 commit, which is correct.
* `.vscode/` (launch, tasks, settings, extensions) was valid, but its launch configurations did not set the Linux
  `GLIBC_TUNABLES` reserve the README asks for, so a Linux user pressing F5 could hit the documented OpenMP TLS
  error. 1.8.0 adds it (ignored on Windows and macOS); its listing will show the 1.8.0 commit.

Both folders are part of the source archive.

## One source for the version

`qemx/__init__.py` (`__version__ = "1.8.0"`) is the only place the version is written for code and packaging:
`pyproject.toml` reads it (`dynamic = ["version"]`, `[tool.setuptools.dynamic] version = {attr =
"qemx.__version__"}`; a wheel built from the tree is `qem_explorer-1.8.0`), and `qemx.provenance` records it in every
run. `CITATION.cff`, the README title and `docs/RELEASE_NOTES.md` must state the same version;
`tests/test_release_metadata.py` fails if they do not.

## Updated

| file | change | reason |
|---|---|---|
| `qemx/__init__.py`, `pyproject.toml` | 1.8.0; `pyproject` reads the version from the package; `[build-system]`; `project.urls` (repository); the `dev` extra equals `requirements-dev.txt` (pytest, nbclient, ipykernel, tomli on Python < 3.11; the unused `playwright` removed) | one version source; the extra had drifted |
| `CITATION.cff` | version 1.8.0, release date, `repository-code`; the TODO points to `docs/OWNER_DECISIONS.md` and no longer suggests a specific license | active version; no invented owner decisions |
| `README.md` | title, "What is new in 1.8.0" (1.7.0 section renamed "What was new"), PowerShell activation and the Windows guide, the two missing pages (QEM Performance & Selection, Noise Models & Discrepancy), layout with `qemx/selection`, `configs/`, `examples/`, the new documents, `.streamlit/`, `.vscode/`, the workflow | outdated or missing instructions |
| `requirements-lock.txt` | header only: "1.8.0 was tested with", unchanged since 1.5.1, Python 3.12 only | the header named 1.5.1; pins unchanged |
| `.github/workflows/ci.yml` | the two observed CI failures (`docs/CI_FAILURE_ANALYSIS.md`): runner diagnostics, an AVX-512-disabled rerun of the numerically sensitive tests, the legacy-locale job through the strict-encoding wrapper, current action versions, test reports as artifacts, figure validation of the shipped studies, Windows steps that fail on a failing command | genuine failures and stale steps |
| `.vscode/launch.json`, `.vscode/tasks.json` | `GLIBC_TUNABLES` for the Linux OpenMP TLS reserve | consistency with the README |
| `.gitignore` | the 1.8.0 study's per-trial arrays (`reports/selection_v180/**/*.npz`) | they ship in separate trial-data archives |
| `.gitattributes` | `reports/** -text`: study records keep their bytes on Windows checkouts | label JSON and analysis tables are hashed in run bindings and provenance files; a CRLF conversion would change their digests |
| `examples/experiments/qft_known_answer.yaml`, `ghz_parity_hamiltonian.yaml` | header "QEM Explorer 1.5 · one experiment" → version-neutral, keeping the historical fact "introduced in 1.5" | an active label that read as the current version |
| `docs/USER_GUIDE.md`, `docs/MIGRATION.md`, `docs/ARCHITECTURE.md`, `docs/selection/*.md` | titles, current modules, the 1.8.0 commands, figures, data dictionary, methods §16 | outdated paths, module lists and tab names |
| `docs/selection/schemas/*.schema.json` | regenerated from `qemx/selection/schemas.py`; a test keeps them equal and validates every shipped configuration | five of fourteen configurations failed the shipped 1.7.0 schema |
| `pages/10_QEM_Performance.py`, `ui/selection_view.py` | the active texts name the selection recommender without a stale "1.7"; the newest study is listed first | active labels |
| `python -m qemx.selection hardware plan` | default output `runs/hardware_plan` | 1.7.0 wrote into the shipped `reports/selection_v170/hardware` |

## Checked and left unchanged

| item | why it stays |
|---|---|
| `.streamlit/config.toml` | valid for Streamlit 1.64–1.66 (above); `theme.font = "sans serif"` is an accepted spelling |
| `requirements.txt`, `requirements-dev.txt`, `packages.txt` | the supported ranges and the Chromium package for figure export are current; CI tests the lock, the latest releases and Streamlit 1.64 |
| pins of `requirements-lock.txt` | the release was tested on them (Python 3.12) |
| `configs/selection/pilot17.yaml`, `confirm17.yaml`, `regimes17.yaml`, `noisecomp17.yaml`, `mismatch17.yaml`, `hardware17.yaml` and the 1.6.0 configurations | their parsed content is bound into the digests of shipped runs; their headers name the release that executed them (historical facts) |
| `qemx/selection/bundles/qemx-rec-1.7.0*`, `reports/selection_v170/`, `reports/selection_pilot/`, `manifests/` | frozen evidence and identities; 1.8.0 adds new bundles and a new study folder beside them |
| version numbers inside documents and code comments that describe history ("1.5.1: …", "introduced in 1.6.0", schema names such as `qemx.selection.labels/2`, `qemx.features/2`) | provenance and scientific schema versions, not release labels |
| `examples/experiments/hardware_transfer_*.yaml` | "not part of the 1.5.1 hardware plan" is a historical statement that is still true |
| `SECURITY.md`, `CONTRIBUTING.md` | current; the security contact is the owner's decision (`docs/OWNER_DECISIONS.md`) |
| license, authors, security contact | owner decisions; nothing invented |

## Files to remove when updating GitHub

The 1.8.0 source archive was compared, path by path and byte by byte, with the GitHub tree of `v1.7.0` (commit
`e68ea55`, 1 398 tracked files): every file tracked there is still present in 1.8.0, so **no tracked file becomes
obsolete**; 62 of them changed in content, the others are byte-identical, and 523 files are new (most of them the
1.8.0 study in `reports/selection_v180/`). One file differed in its line endings only:
`docs/research/direction_A_v1_cells.csv` was written with CRLF by Python's `csv` module and normalized to LF by the
owner's Git on GitHub; 1.8.0 ships the LF bytes, identical to GitHub's, so the update does not touch it. Files that
exist only locally and are deliberately *not* in the public archive — `archive/` (superseded notes and manuals), the
per-trial arrays (`*.npz`, in the trial-data archives), the 1.6.0 pilot's figures — are not tracked on GitHub either.
The procedure below removes any stale file anyway, so it stays correct if the archive changes.

## Update checklist (GitHub)

1. In a clean clone of `https://github.com/Hyuga-Ito/qem-explorer` on `main`, create a branch:
   `git switch -c release-1.9.0`.
2. Remove the tracked tree, keeping Git's metadata: `git rm -r -q .` (this deletes the working files, not the
   history).
3. Extract `qem-explorer-1.9.0.zip` and copy the **contents** of its single root folder `qem-explorer-1.9.0/`
   (including `.github/`, `.streamlit/`, `.vscode/`, `.gitignore`, `.gitattributes`) into the clone's root.
4. `git add -A`, then review `git status`: deletions are files that 1.9.0 no longer has (none are expected), and
   `reports/selection_v180/`, `reports/pilot_v190/` and the other 1.8.0/1.9.0 additions are new.
5. Run the checks of `docs/WINDOWS_FIRST_RUN.md` step 2 (or `pytest -m "not slow"`), commit
   ("QEM Explorer 1.9.0"), push the branch, open a pull request and let CI run; merge when it is green. To exercise the
   optional trial-data job, upload the trial archives to a draft release and start the workflow by hand
   (`trials_study: selection_v180`).
6. Tag the merge commit `v1.9.0` (existing tags are not moved; whether to tag 1.8.0 separately is your decision,
   `docs/OWNER_DECISIONS.md`), and attach the trial-data archives to the GitHub release; never attach the private
   audit archive.

Nothing in this release was pushed, tagged or published automatically.
