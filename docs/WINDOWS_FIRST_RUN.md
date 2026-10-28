# First run and local experiments on Windows (QEM Explorer 1.9.1)

This guide covers a fresh copy of the release on Windows 10 or 11: installation, the application, a smoke run of
the research pipeline, the bounded pilot, how to resume, retry and check a run, and how to plan the full development
study. **Status of these commands:** the Python commands were executed for 1.9.1 **on Linux** (Python 3.12, the release
lock, a fresh extraction into a path with spaces and Japanese characters; `docs/QA_SUMMARY.md`), as marked in each
section. **None was executed on native Windows** (no Windows computer was available), and the CI's `windows-smoke` job
has **not run on GitHub Actions** (nothing was pushed). The PowerShell-specific lines (`tar`, `Activate.ps1`,
`$env:`, `Remove-Item`) are untested. Everything runs offline on emulated fake devices; nothing contacts IBM Quantum.

## 1. Python 3.12 and the release

Use Python **3.12** (the release lock was tested with 3.12.3; several pins need ≥ 3.11).

| Option | Commands |
|---|---|
| python.org installer | Install 3.12 (tick *Add python.exe to PATH*), then open **PowerShell** |
| Anaconda / Miniconda | **Anaconda Prompt**: `conda create -n qem python=3.12` then `conda activate qem` |

Any folder works, including one with spaces or Japanese characters (`C:\研究\qem explorer`). Prefer a short path if
*Long paths* are disabled.

```powershell
$D = "$env:USERPROFILE\Downloads\QEM_Explorer_1.9.1_delivery"   # the folder that holds the downloaded archives
New-Item -ItemType Directory -Force C:\work | Out-Null
cd C:\work
tar -xf "$D\qem-explorer-1.9.1.zip"
cd qem-explorer-1.9.1
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

*The environment was created from `requirements-lock.txt` in a fresh extraction on Linux (Python 3.12.13) for 1.9.1;
the PowerShell lines above are untested.* Use a new folder for 1.9.1 (do not unpack over 1.9.0); an existing `runs\` folder can be moved or pointed to with
`--out`. (If `qem-explorer-1.9.1.zip` arrived in parts — `qem-explorer-1.9.1.zip.001`, `.002`, … — double-click
`join_qem-explorer-1.9.1.cmd` in that folder first: it checks that every part listed in `SHA256SUMS.txt` is present,
verifies every part, joins them, verifies the joined archive and every archive delivered in one piece, and stops with
"FAILED" on any missing or wrong file without overwriting anything. `python join_verify.py` does the same on any
platform.)

## 2. Check the installation (about a minute, then tens of minutes)

```powershell
python -m pytest -q -m "not slow" tests\test_text_io.py tests\test_readiness_v190.py tests\test_pipeline_v190.py tests\test_fixes_v191.py
python -m pytest -q tests\test_fixes_v191.py -m slow      # the full-development order and the CPU ledger (minutes)
python -m pytest -q -m "not slow"
```

The first line includes the readiness regressions with real supervised child processes (timeouts, resume, corrupted
arrays, ceilings, parent death, Ctrl+C) and the 1.9.1 corrections; the second runs the shipped full-development stage
order on a small generated study. *Executed on Linux for 1.9.1 (results in `docs/QA_SUMMARY.md`); not on Windows.*

## 3. The application

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501 (headless server; no browser opens by itself). *QEM Performance & Selection →
Recommend for my circuit* shows the effective device, the physical layout and the declared symmetry used, and marks a
shown recommendation out of date when the circuit, input, bases, bit order, observable, symmetry, device, budget,
bundle **or the "allow simulated features" switch** change.

## 4. Smoke run of the research pipeline (minutes)

```powershell
python -m qemx.selection check-config configs\selection\smoke19.yaml configs\pipelines\smoke19.yaml
python -m qemx.selection plan configs\selection\smoke19.yaml --tasks-csv "runs\smoke19 テスト\plan\tasks.csv"
python -m qemx.selection pipeline configs\pipelines\smoke19.yaml --out "runs\smoke19 テスト" --workers 2
python -m qemx.selection status --out "runs\smoke19 テスト"
```

The pipeline runs plan → label → analyze → freeze → nested evaluation → freeze completion → bundles (written into the
run folder) → a recommendation with the run's own bundle → figures → report → status, every command as an argument
array (paths with spaces and non-ASCII characters need no extra quoting inside the pipeline). Measured on two Linux
cores: about 7 minutes for the first run, under a minute for a complete rerun (every stage verified and kept).
*Executed on Linux for 1.9.1 from a fresh extraction (first run and unchanged rerun, exit 0); not on Windows.*

## 5. The bounded pilot (about 1 CPU-hour of labels plus analysis)

```powershell
python -m qemx.selection plan configs\selection\pilot19.yaml --workers auto
python -m qemx.selection pipeline configs\pipelines\pilot19.yaml --out runs\pilot19 --workers auto
```

`--workers auto` uses CPUs − 1, limited so that the per-task memory limit (3 GB) times the workers stays within 70 %
of RAM (e.g. 3 workers on a 16 GB computer, whatever the core count). Measured: 54 tasks, 0.86 CPU-hours of labels
(27 minutes with 2 workers; median 39 / 51 / 67 s per task at 3 / 4 / 5 qubits on two shared Linux cores), plus the
analysis (≈ 70 minutes single-threaded) and the nested evaluation (≈ 20 minutes): about two hours in all on two cores
(1.9.0 measurement; the 1.9.1 corrections do not change the pilot's work). *For 1.9.1, `plan` and `pipeline
--dry-run` of the pilot and four of its tasks (with a Ctrl+C and a resume) were executed on Linux; the whole pilot was
not re-run (its stage order is the one the 1.9.1 regressions run end to end on small studies).*

## 6. Interrupting, resuming, retrying

* **Stop** with Ctrl+C (or close the window): running tasks are stopped, the run's lock is released; a child whose
  parent disappeared ends itself within about a second. Python prints a `KeyboardInterrupt` traceback; that is
  expected. Since 1.9.1 the stopped tasks' time is kept as `interrupted` attempts. *Ctrl+C, a resume and parent death
  were executed on Linux for 1.9.1; not on Windows (Ctrl+Break and closing the window untested).*
* **Resume**: run the same command again. Labeled tasks are kept (arrays verified), missing ones are labeled.
* **Failed tasks** (timeout, memory, error) are kept as records and are *not* retried automatically; the command exits
  1 and says so on every rerun. After raising the limit (`limits.task_seconds` / `memory_gb`, operational — a freeze
  stays valid), run the same command with `--retry-failed`; the failed attempt stays in `attempts.jsonl` and
  `failed_attempts/`.
* **Corrupted records** (missing or changed arrays) stop labeling (exit 2); `--quarantine-corrupted` moves them to
  `quarantine\` (listed in `run.json`) and relabels them.
* **Resource caps** (`limits.cpu_hours`, `disk_gb`, `min_free_disk_gb`): a known estimate above a cap is refused
  before anything runs (exit 2); `--override-limits "reason"` records a deliberate start, but the monitored ceiling still
  stops labeling at the cap (checkpoint-safe: running tasks finish). To go past a cap, raise it. Since 1.9.1 the CPU
  cap counts every attempt of every invocation (failed, retried, quarantined, interrupted): `status` prints "labeling
  time consumed".
* **Exit codes**: 0 complete (or deliberately partial with `--limit` and nothing failed); 1 incomplete (or a
  once-only step such as `freeze` that was already done, unless `--reuse`); 2 refused
  (configuration, binding, admission, corrupted or stale inputs, an inconsistent once-only step); 3 `plan`: a cap
  would refuse the run. `analyze` refuses incomplete labels unless `--allow-partial` (marked partial; a freeze refuses
  it).

## 7. Trial data of the shipped studies (optional)

The per-trial arrays are release attachments of 1.9.0 and apply unchanged to 1.9.1 (the data did not change). Unpack
them **into the repository folder** and verify the study they belong to:

```powershell
tar -xf "$D\qem-explorer-1.9.0-trials-1.zip"
tar -xf "$D\qem-explorer-1.9.0-trials-2.zip"
tar -xf "$D\qem-explorer-1.9.0-trials-pilot.zip"
python scripts\verify_trials.py reports\selection_v180 reports\pilot_v190
$env:QEMX_REQUIRE_TRIALS = "selection_v180,pilot_v190"
python -m pytest -q tests\test_trial_data_v180.py tests\test_trial_data_pilot19.py
Remove-Item Env:QEMX_REQUIRE_TRIALS
```

## 8. A run started with 1.9.0

* **smoke19 / pilot19 runs completed with 1.9.0** stay valid: their labels, analysis and freeze are verified and kept by
  the 1.9.1 pipeline (an analysis made by 1.9.0 code is accepted for *adding* analyses only with `--accept-changes`,
  recorded).
* **A fulldev19 run that 1.9.0 stopped at `freeze-complete`** (exit 2, "the nested evaluation is of
  MODEL[hgb|deployable+simulated] …"): rerun the same pipeline with 1.9.1 and `--accept-changes`. The labels, the
  analysis and the frozen configuration are kept; the premature nested tables are moved to
  `analysis\stale_nested_<time>\` and recomputed for the frozen configuration; then τ is recorded
  (`docs/MIGRATION.md`, Part 00000). *Tested on Linux on a small study run with 1.9.0, not on Windows.*

```powershell
python -m qemx.selection pipeline configs\pipelines\fulldev19.yaml --out runs\fulldev19 --workers auto --accept-changes
```

## 9. Planning the full development study (fulldev19) — decide before running

Labels: 3.0–9.3 CPU-hours estimated from the 1.8.0 and pilot rates (conditional: the coherent `chi` tasks are
unmeasured). Analysis: **time unknown** — 4× the pilot's model fits and 51× its fits × training rows, so it can be far
longer than the labels; measure it as below before committing the computer. Run in this order, checking each output:

```powershell
python -m qemx.selection check-config configs\selection\fulldev19.yaml configs\pipelines\fulldev19.yaml
python -m qemx.selection plan configs\selection\fulldev19.yaml --measured runs\pilot19\dev --workers auto --tasks-csv runs\fulldev19_tasks.csv
python -m qemx.selection label configs\selection\fulldev19.yaml --out runs\fulldev19 --workers auto --limit 6
python -m qemx.selection plan configs\selection\fulldev19.yaml --out runs\fulldev19 --workers auto
python -m qemx.selection pipeline configs\pipelines\fulldev19.yaml --out runs\fulldev19 --workers auto --until label
python -m qemx.selection status --out runs\fulldev19
python -m qemx.selection analyze configs\selection\fulldev19.yaml --out runs\fulldev19 --splits same-distribution --extras=
python -m qemx.selection pipeline configs\pipelines\fulldev19.yaml --out runs\fulldev19 --workers auto
```

1. `plan --measured` prints the labeling estimate for *your* computer and the analysis workload (split parts, model
   fits, the same-distribution split's share of the fits × rows).
2. `label --limit 6` labels six tasks to measure this computer's rates; the second `plan` uses them.
3. `pipeline --until label` labels everything (resumable: rerun the same line after an interruption).
4. Time the `analyze … --splits same-distribution --extras=` line (write `--extras=` exactly like this: an empty `""`
   is dropped by Windows PowerShell 5.1). Divide its time by the share `plan` printed (0.186) for a rough time of the
   main splits; the extras come on top. If that is too long for your computer, stop here and decide (fewer splits or
   families in a new, declared configuration).
5. The last line runs the analysis (the same-distribution split again, with the others), the freeze, the nested
   evaluation of the frozen configuration, τ, bundles and figures. It never labels or evaluates conf19.

*Status: for fulldev19, `check-config`, `plan` (with and without `--measured`) and `pipeline --dry-run` were executed
on Linux for 1.9.1 (exit 0). The same sequence with smaller studies was executed too: `label --limit 4` followed by
`plan --out` on pilot19, and `pipeline --until label` → `analyze --splits same-distribution --extras=` → `pipeline` →
rerun on smoke19's labels (all exit 0); the shipped fulldev19 stage order ran end to end on a small generated study.
fulldev19 itself was not run (not requested, and expensive). Nothing here was run on Windows.*

## 10. What not to start without a decision

* The reserved confirmation (`configs\selection\conf19.yaml`): once only, after the development policy is frozen **and**
  after you have chosen its population and size (32 circuits of five families give ≈ 45–58 % power at Δ = 0.08; options
  in `docs\selection\RESEARCH_PLAN_v19.md`, "conf19: estimand, population and size").
* Hardware: nothing is submitted by any command here; the hardware plan needs your explicit authorization.

## Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `py` not found | Use `python` from the activated environment, or reinstall Python with the launcher |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use cmd and `.venv\Scripts\activate.bat` |
| `pip` resolves different versions | Not Python 3.12: check `python --version` and recreate the environment |
| "is being written by another process" | Another command writes the same run; wait. A lock of a process that no longer runs is replaced automatically |
| "was labeled with other code or another environment" | Upgrading the software mid-run: add `--accept-changes` to resume (recorded); complete labels are reused without it |
| "holds tables of another code" | The analysis was made by other code: `--accept-changes` adds to it (recorded); never re-analyze a frozen run in place |
| "the analysis cannot be frozen with this configuration" (first `freeze`) | The analysis does not match the configuration, labels or code, or lacks a declared split or grid configuration: re-run `analyze` (or `--accept-changes` for a code change only; recorded) |
| `[analyze] deployed, nested_selective deferred …` | Expected (1.9.1): they evaluate the frozen configuration and run in the `nested` stage after `freeze` |
| `--extras ""` gives "expected one argument" | Windows PowerShell 5.1 drops empty arguments: write `--extras=` (or `--splits=`) |
| A file "is not valid UTF-8 (byte …)" | Re-save it as UTF-8; QEM Explorer never guesses an encoding |
| Figure export fails | Install Chrome or Chromium for Kaleido; interactive figures are unaffected |
