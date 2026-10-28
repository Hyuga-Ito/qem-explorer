# Migrating to 1.9.1

Existing data are never rewritten: old run directories, records, ledgers, manifests and state files stay as they are
and still load; every migration is reported. A run directory refuses to be finished by a version that writes
records of another statistical meaning.

# Part 00000 — from 1.9.0 to 1.9.1

No statistical meaning changes: labels, task identities, estimators, policies, the metric, the 1.8.0 confirmation and
the 1.9.0 pilot are the same. Every item is in `docs/ISSUE_EVIDENCE_MATRIX.md` (part 1.9.1).

**Full development order (F1).** `analyze` no longer computes the analyses that evaluate the frozen configuration
(`deployed`, `selective`, `nested_selective`) before `freeze`: when they come from the configuration's extras (or the
default extras of a configuration without an `extras` key) they are recorded as *deferred* in `results.json`; when
they are named explicitly (`analyze --extras deployed` before a freeze) the command refuses (exit 2) instead of
evaluating a configuration that may never be frozen. After the freeze the pipeline's `nested` stage computes them (by
hand: `analyze CONFIG --out RUN --splits= --extras deployed,nested_selective`). A direct `analyze` of a configuration
that declares both `analysis.deployed_family` and `analysis.deployed_set` still evaluates that declared configuration
at once (one of the two alone counts as undeclared, and the note says so); the pipeline always computes these tables
in its `nested` stage (also when the pipeline file's `analyze.extras` names them). Each such table now records a binding (configuration, data, evaluation settings `k`,
`nested_inner_k`, `bootstrap`, code). The `nested` stage refuses before a freeze and moves nothing when the
re-analysis would be refused (another analysis section, labels or data; other code without `--accept-changes`); it
keeps tables whose binding matches the freeze, and tables made before 1.9.1 (no binding) that evaluate the frozen
configuration; other tables are moved to `analysis/stale_nested_<time>/` (with a README; `results.json` keeps a
`stale_nested` record) and recomputed. `freeze --complete` needs the `nested_selective` tables (as before) and checks
the binding of every frozen-configuration table present; tables made before 1.9.1 are accepted only when they evaluate
the frozen configuration, and the freeze records that only the configuration was checked. *Quick runs:* `run
configs/selection/smoke.yaml` (label, analyze, figures; no freeze) now leaves the figures that need these tables
(fig05 and two supplementary figures) pending, as reported by `figures`; `freeze`, then `analyze … --splits= --extras
deployed,nested_selective`, then `figures` draws them for the frozen configuration.

*A `fulldev19` run started with 1.9.0* (stopped with exit 2 at `freeze-complete`, "the nested evaluation is of
MODEL[hgb|deployable+simulated] …"): run the same pipeline again with 1.9.1 and `--accept-changes`:

```
python -m qemx.selection pipeline configs/pipelines/fulldev19.yaml --out runs/fulldev19 --workers auto --accept-changes
```

The labels and the analysis tables are kept and verified, **the frozen configuration stays as it is** (the first
freeze is never redone), the premature nested tables are set aside in `analysis/stale_nested_<time>/` and recomputed
for the frozen configuration, and `freeze --complete` then records τ. `--accept-changes` is needed because the
analysis tables were made by 1.9.0 code; it is recorded in `results.json` (`code_changes`). Without it the nested stage
stops with exit 2 and says so. Tested on a copy: the shipped stage order run with 1.9.0 to its exit 2, then 1.9.1
without (exit 2) and with `--accept-changes` (exit 0), and an unchanged rerun (exit 0; `[nested] kept`). An
extras-only analysis (the nested stage) no longer re-pools the split tables, so the table the freeze was chosen on is
never rewritten.

**First freeze (F2).** Before the first `freeze`, the analysis must match the configuration and labels given now
(analysis section, label-determining configuration, data digest) and have been made by this code; every declared split
must be present (incomplete comparisons are refused), and the same-distribution comparison must cover every
configuration of the declared grid; the choice is made only within the declared grid. Exit 2 with the list of
problems otherwise. An analysis made by other code (for example by 1.9.0) needs a fresh `analyze` or `freeze
--accept-changes`, which is recorded in `freeze.json` (`accepted`); an analysis from before 1.9.0 (no identity) needs
`--accept-changes` too. An existing freeze is verified as in 1.9.0 (exit 1, or 0 with `--reuse`; 2 when inconsistent).

**CPU accounting (F3).** `limits.cpu_hours` is checked against the labels directory's *consumed* seconds: every attempt
of every invocation in `attempts.jsonl` (failed, replaced by `--retry-failed`, quarantined, and — new —
`interrupted` entries for tasks stopped by Ctrl+C or SIGTERM), plus the recorded seconds of records that have no attempt
entry (labeled before 1.9.0 or imported; matched oldest first; a `failed_attempts/` copy identical to its record is
the same attempt, so it is never counted twice, and repeated `--retry-failed` runs that relabel nothing no longer add
copies). In 1.9.0 the ceiling restarted
from the current records only, so a retry hid the failed attempts' time. Admission uses consumed + estimated remaining.
`integrity.completion` reports it as `consumed`; `status` and `plan` show it. The unit is still summed worker
wall-clock seconds (≈ CPU seconds with one numerical thread per worker), and a task already running when the cap is
reached still finishes (checkpoint-safe overshoot). Quantum shot ledgers are unchanged.

**Planning output (F4).** `plan` prints a structural *analysis workload* (split parts, model fits including inner
tuning, fits × training rows) — counts, not a time. `reports/tables_v190/s6_costs` states the wall time for a declared
hypothetical computer (8 cores, 7 workers): 9.3 CPU-h / 7 = 1.3 h (1.9.0 printed 4.7 h = 9.3 / 2 from the generating
computer under a "7 workers" caption); the row is renamed `fulldev19_wall_hours_scenario`, and a row
`fulldev19_analysis_fits` is added.

**Summaries.** Per-split summary rows have a new column `inference`: strata over one base circuit are "descriptive"
(their bootstrap over circuits is degenerate), fewer than five "few base circuits … descriptive".

**Process supervision (F5).** Workers end when their parent ends through the multiprocessing parent handle instead
of comparing process IDs with psutil (which can look at another PID namespace in containers); psutil-based memory
sampling and process-tree stopping are used only when psutil's PID view is this process's own (Linux: `/proc/self`
names this PID; macOS: the parent PIDs agree; Windows: always); lock liveness uses the process's own PID view. The
regime study and `profile` keep 1.9.0's model before a freeze (`hgb|deployable+simulated` unless declared). No user
action.

# Part 0000 — from 1.8.0 to 1.9.0

No statistical meaning changes: labels, task identities (all 1989 of the 19 shipped configurations, pinned in
`tests/data/expansion_ids_v180.json`), estimators, policies, the metric and the 1.8.0 results are the same. What
changes is what commands **accept, report and return**, so scripts written around 1.8.0 can see different exit codes.
Every item below is in `docs/ISSUE_EVIDENCE_MATRIX.md` with its reason.

**Exit codes and completion**

- `label`, `run`, `confirm`, `final`, `regimes`, `mismatch`, `scripts/root_audit.py label`: 0 = the run is complete
  (or deliberately partial with `--limit` and nothing failed), 1 = incomplete, 2 = refused (configuration, binding,
  admission, corrupted or stale inputs, an inconsistent once-only step). In 1.8.0 an unchanged rerun of a run with a
  timed-out task printed `tasks=0, errors=0` and exited 0; it now reports the failed task and exits 1 on every rerun.
- Failed tasks are kept and not retried automatically (as in 1.8.0, which however counted them as done and could only
  retry them after the record was deleted by hand). Use `--retry-failed`; the earlier record is kept in
  `failed_attempts/` and every attempt in `attempts.jsonl` (new files in the labels directory).
- A record whose arrays are missing or changed stops labeling (exit 2) instead of being skipped;
  `--quarantine-corrupted` moves it to `quarantine/` and relabels it.
- `run` returns the labeling result and does not analyze an incomplete run (1.8.0 returned the figures' code).
- `analyze` refuses incomplete labels (exit 2) unless `--allow-partial`; such results are marked `partial` and
  `freeze` refuses them. Zero completed tasks are always refused.
- `freeze` and `freeze --complete` on an already frozen run now **verify** the existing freeze against the
  configuration, the analysis and the development data instead of only refusing: consistent → exit 1 as before
  ("already done", unchanged), or 0 with the new `--reuse`; inconsistent → 2. `final` and `confirm` accept `--reuse`
  with the same meaning for their once-only evaluation. The pipeline runner always passes `--reuse`, so a resumed
  pipeline continues past these steps.
- `plan` exits 3 when a configured cap would refuse the run; `recommend` exits 3 for an unreadable or unsupported input.
- New: `status --out DIR` (any run), `pipeline FILE --out DIR` (stage-aware runner), `check-config` for every YAML
  format, `plan --tasks-csv`, `plan --json`.

**Resources**

- `limits.cpu_hours`, `disk_gb` and the new `limits.min_free_disk_gb` are enforced: a known estimate above a cap is
  refused before labeling unless `--override-limits "reason"` (recorded in `run.json`); while labeling, monitored
  ceilings stop at the cap even with an override (checkpoint-safe). Without measured rates the CPU estimate is
  reported as unknown rather than guessed. `task_seconds` and `memory_gb` are unchanged (per task).
- `--workers auto` = CPUs − 1, bounded so that workers × `memory_gb` stays within 70 % of RAM.
- Each labels directory and analysis holds a lock (`.qemx-run.lock`) while a command writes it; a second writer is
  refused; a lock of a process that no longer runs is replaced.

**Analyses, freezes and replays**

- An analysis records its identity (digests of the `analysis` section, the label set, the data and the code). Tables
  made with another configuration or data are refused (`--reset-analysis` archives them first; a frozen run's analysis
  is never archived); tables made by other code need `--accept-changes`, recorded in `results.json` (`code_changes`).
- A freeze records the development data digest and the scientific configuration digest (`limits` and `status` are
  operational and excluded); operational changes keep a freeze valid.
- `confirm` checks the confirmation's logical circuits against the **training labels actually used**, not only the
  ancestry registry. `EVALUATED` records the configuration digest, training parts and output digests.
- `replay.replay` scores the manifest's population and refuses an incomplete one (`require_complete`); an exploratory
  subset needs `subset` and `subset_reason` and is recorded. A cached `labels.csv` that differs from the arrays raises
  `StaleTable`. `scripts/replay_confirmation.py` checks the SBS-B column against the rule and returns 2 on errors
  (1.8.0 printed a traceback).
- `bundle` writes into `RUN/bundles/<version>` by default (1.8.0: into `qemx/selection/bundles/`, the shipped folder)
  and never overwrites an existing bundle. Use `--bundle-out` for another place.

**Recommendations**

- A state converted for the recommender keeps a declared symmetry when it holds on the ideal measured distribution
  (≤ 20 qubits); otherwise it is dropped with a note. A state that declared a valid symmetry can therefore get a
  different — now study-consistent — recommendation (GHZ-6 at B = 10³ on `fake_kyiv` with the frozen bundle: SV, as for the study task, instead of PEC-TW).
- An explicit device (`--device`, the app's selector) overrides the state's backend; the layout is validated against it
  (distinct, non-negative, in range). 1.8.0 used the state's backend.
- The result adds `effective_device`, `requested_layout`, `physical_layout`, `symmetry`, `conversion_notes`,
  `simulation` and `signature`. Invalid devices, layouts or symmetries are `Unsupported` (no traceback).
- The app marks a shown recommendation out of date when the "allow simulated features" switch changes.

**Configurations**

- Selection configurations: `analysis`, `split_options` (per split), `mismatch`, `regimes` and `limits` are closed key
  sets; lists must be non-empty and unique; `master_seed ≥ 0`. A configuration with a typo such as
  `analysis.feature_set` was silently run with defaults in 1.8.0 and is now refused. All shipped configurations remain
  valid. Descriptive keys the code does not read (`mismatch.robust`, `regimes.*`) give warnings.
- New optional `expansion_policy: explicit` (every block states `include_default` and `variants_per_base`); without it
  the historical expansion applies unchanged. New splits `noise`, `device-base` and `joint`.
- Plotting configurations: an explicitly named file that is missing, or an unknown key, is an error (1.8.0 fell back to
  defaults).

**Scripts and CI**

- `scripts/reproduce_v18.sh` calls the pipeline (`configs/pipelines/reproduce_v18.yaml`); `reproduce_pilot_17.sh` no
  longer masks failures. Both can be resumed after an interruption.
- `scripts/verify_trials.py` verifies the named study's manifest only (`--attachments` maps archives to studies).
- Tests that need trial arrays fail instead of skipping when `QEMX_REQUIRE_TRIALS` names their study.
- `figures --figures-out DIR` writes figures elsewhere than the run.


# Part 000 — from 1.7.0 to 1.8.0

What changes meaning (each change with its reason in `docs/selection/METHODS.md` §16 and
`docs/ISSUE_EVIDENCE_MATRIX.md`):

- **The deployed recommendation** follows policy `qemx.selection.policy/2`: candidates are walked in the model's
  order and only a **verified-feasible** candidate is returned (shared preflight contract,
  `qemx/selection/preflight.py`); unverified candidates are skipped and listed; the fallback is SBS-B's first verified
  candidate. The result dictionary gains `policy`, `preflight_level`, `guard`, `preflight` (status, level, reason,
  minima per candidate), `skipped_candidates`, `design_seeds` and `execution`; `feasibility` keeps its keys and now
  carries the preflight status. The same experiment can therefore receive a different (verified) recommendation than
  in 1.7.0 — by design, for the ten regression cases of the review.
- **The 1.7.0 bundles still load** (`qemx-rec-1.7.0`, `qemx-rec-1.7.0-deployable`): their feature definitions and
  snapshots are recognized by the recorded 1.7.0 fingerprints. A bundle whose feature definitions, protocol
  definitions or normalization differ from the code is refused (`BundleIncompatible`); a different calibration of a
  known device is a transfer, reported as out-of-distribution. New bundles: `qemx-rec-1.8.0` (the frozen 1.8.0
  configuration) and `qemx-rec-1.8.0-deployable` (deployable features; the default in the UI and the CLI).
- **Labels of new studies** use seed scheme 2 (`learning_designs: {…, seed_scheme: 2, crn: task|base}`): preparation
  and trial streams come from stable identifiers instead of a per-study seed pool. A configuration without
  `seed_scheme` keeps scheme 1 and reproduces 1.7.0 bitwise; its digests are unchanged. A run refuses to mix schemes.
- **Label summaries** add `var_deploy_{sel,eval}` (the random-effects variance of one fresh deployment; finding F5);
  `var_*` keeps its pooled meaning. Summaries of policies add `p_severe_vs_raw`, `p_severe_abs`, `q95_risk` and
  `worst5_loss_share`. Re-run `analyze` on an existing run to obtain them; labels are not affected.
- **JSON written by 1.8.0 is strict**: unavailable or unbounded numbers are `null` with a `"$nonfinite"` map of JSON
  pointers; `qemx.textio.read_json` restores them. 1.7.0 files with `NaN` tokens remain readable. External tools that
  read 1.8.0 JSON directly see `null`, never a bare `NaN`.
- **Collection** checks every record's task id, label configuration, run id (or a recorded import) and arrays
  digest; mismatch labels carry their run id. 1.6/1.7 records without a run id are accepted only as
  `legacy_unbound` evidence in their own (legacy) runs.

What is new:

- 17 candidates (`portfolio-1.8`; `python -m qemx.selection registry --markdown`); the 1.6/1.7 eight remain
  `portfolio-1.6` with unchanged identities. A configuration without `protocols` still uses portfolio-1.6.
- `python -m qemx.selection registry`, `scripts/replay_confirmation.py`, `scripts/portfolio_comparison.py`,
  `scripts/root_audit.py`, `scripts/replay_v170_confirmation.py`, `python -m qemx.selection.thesis_tables`,
  `scripts/reproduce_v18.sh`; the four supplementary figures of `qemx/selection/plots18.py`; feature sets
  `deployable18` and `deployable18+simulated` (the 1.7.0 sets keep their names and meaning).
- `python -m qemx.selection hardware plan` writes to `runs/hardware_plan` by default (1.7.0 wrote into the shipped
  `reports/selection_v170/hardware`); it still never submits.
- `python -m qemx.selection profile` writes the 1.8.0 runtime columns and the early-rejection rows; 1.7.0
  `runtime.csv` files still draw.
- `analysis/design_variance.csv` covers every learning candidate of the run's portfolio (with portfolio-1.8: CDR, NIL
  and FR); for a 1.6/1.7 run (CDR, NIL only) its content is unchanged.
- `scripts/replay_confirmation.py --bundle-dir DIR` replays bundles from another directory (`scripts/reproduce_v18.sh`
  builds its bundles inside the run directory and never overwrites the shipped ones).
- The QEM Performance page lists the 1.8.0 study first; the recommender tab shows the preflight status and level of
  every candidate, the skipped candidates and the execution note, and the capability record of each candidate.
- Version metadata has one source: `qemx/__init__.py` (`__version__`); `pyproject.toml` reads it
  (`[tool.setuptools.dynamic]`), and a test keeps `CITATION.cff`, the README title and the release notes in step.

Nothing to do for an existing installation beyond installing the 1.8.0 lock (`requirements-lock.txt` unchanged in
content for Python 3.12). Old run directories, bundles, notebooks and exported scripts keep working.

# Part 00 — from 1.6.0 to 1.7.0

What changes meaning (each change is listed with its reason in `docs/selection/METHODS.md` §14 and
`docs/REVIEW_RESPONSE_1.7.md`):

- **CDR and NIL labels** now estimate the *fresh-design* risk: each label half has its own pool of independently drawn
  training designs (`learning_designs: {mode: fresh, per_half: K}`), and their standard errors are over designs. A
  1.6.0 configuration's integer `learning_designs: 6` is read as `fixed-shared` (the 1.6.0 estimand: the risk
  conditional on six designs shared by both halves); its digests are unchanged, so 1.6.0 run directories still load,
  resume and analyze as before, and they are reported with that estimand. A run directory refuses to mix designs
  plans. To compare with 1.7.0, relabel with a fresh-design configuration (`pilot17.yaml` relabels the 1.6.0 pilot's
  tasks with the same seeds).
- **Aggregates**: the arithmetic `policy_mse` is weighted by base circuit (the 1.6.0 row mean is kept as
  `policy_mse_rows`); SBS/SBS-B rank every candidate on one population; failure shares are the union of flags
  (`fail_any`), not sums. Re-run `analyze` on an existing run directory to get the corrected tables; labels are not
  affected.
- **Selective recommendation** is evaluated by nested cross-validation (`analyze --extras nested_selective`,
  `nested_*.csv`); `freeze.json` has schema `qemx.selection.freeze/2` and is completed with `freeze --complete`
  (deployment threshold and nested digest). 1.6.0's cross-fitted `selective_*.csv` tables are still produced by the
  `deployed` extra and labelled as the 1.6.0 procedure.
- **Statement vocabulary**: interval kinds are `cv-residual`, `cv-residual-simultaneous`, `joint (simultaneous)`,
  `residual-gaussian` and the new `split-conformal(-simultaneous)`; 1.6.0's "conformal" interval is the
  `cv-residual` one. Certificates name their units and say "clipped estimator"; the 2ε relation is a post-hoc check.
- **The 1.6.0 final test is historical** (it overlaps earlier studies by logical circuit); confirmatory claims come
  from the new confirmation set (`confirm` command, `confirm17.yaml`).

What is new or behaves differently:

- **Run binding** (`<part>/run.json`): resuming a run with a different label configuration, task manifest, split,
  portfolio or device snapshot is refused; a code or environment change needs `--accept-changes` (recorded). Stale
  task files are never collected. Old directories without `run.json` are read as legacy and reported so.
- **Supervised labeling**: every task runs in a child process with parent-enforced time and memory limits on all
  platforms (Windows included); error, timeout and memory records are written without arrays and listed in
  `label_failures.csv`; `--retry-failed` reruns them. Records store an arrays digest; a mismatch is refused.
- **Configurations** are validated against the schema on load: unknown keys and family parameters are errors, invalid
  variants are errors (1.6.0 silently skipped them). New keys: `status`, `learning_designs` as a mapping, `limits.
  cpu_hours`/`disk_gb`, `study`; device modifiers `g1`, `g2`, `t` (one-, two-qubit gate errors and relaxation).
- **New commands**: `confirm`, `bundle`, `recommend`, `hardware plan|check`, `freeze --complete`, `regimes
  --subdir`, `label --final-set/--subdir/--accept-changes/--retry-failed`. `profile` reports compilation, deployable
  features, simulated features, inference and label generation separately (columns renamed).
- **Recommender in the app**: the QEM Performance page's "Recommend for my circuit" tab uses the 1.7 model bundles
  (`qemx/selection/bundles/`); the 1.5 recommender is kept in an expander labelled legacy.
- **Playground**: the Output panel separates the input state, the output state before measurement, the measured
  distribution in the chosen bases and the observable; a template's identity is shown as a prediction only while its
  assumptions hold; "Apply to all" reports what changed. Saved states load unchanged.
- **Text files** are read and written as UTF-8 everywhere (no locale dependence); a user-edited configuration may
  start with a UTF-8 BOM; undecodable files raise an error naming the byte position.
- **Hardware plan**: a run's executions are the sum of its circuits' shots (a merged PEC circuit was charged once
  per occurrence); plans made with 1.6.0 that included PEC overstated its shots and should be remade.
- **Figures**: `figures --only NAME[,NAME…]`; a figure whose input tables are missing is listed in `PENDING.md`
  (now also the two figures that need per-trial arrays, when those are absent).
- **Shipped data**: the corrected pilot is `reports/selection_v170/` (records, tables, figures); its per-trial arrays
  are in two trial-data archives to be extracted into the repository root. The 1.6.0 pilot stays as historical data;
  the 1.7.0 source archive omits its per-trial arrays and figures.
- **Version** 1.7.0 in `pyproject.toml`, `qemx.__version__` and `CITATION.cff`; Python ≥ 3.10; `tomli` is a
  development dependency on Python < 3.11.

# Part 0 — from 1.5.1 to 1.6.0

Nothing that 1.5.1 computes changes its value or meaning. 1.6.0 adds the protocol-selection study.

- **New package `qemx.selection`** and its command line `python -m qemx.selection`, configurations under
  `configs/selection/`, documentation under `docs/selection/`, the bundled pilot under `reports/selection_pilot/` (a
  complete run directory: labels, provenance, analysis tables, figures, tables).
- **Method samplers** (`MethodResult.sampler`) accept new optional keywords — `theta` (RAW/readout, PEC with one
  shot per sampled circuit, symmetry verification), `split` and `with_failures` (ZNE), `deployable` and `s_tr`
  (CDR/FR), `s_tr` (NIL). Calls without them behave as before. A PEC sampler without an exact per-shot law (several
  shots per sampled circuit, or the Aer engine) refuses `theta` with an error instead of ignoring it.
- **QEM Performance page**: a fifth tab, *Protocol selection (1.6)*; the four existing tabs are unchanged and keep
  their labels (saved tab selections still apply).
- **Within 1.6.0 development** (relevant only to run directories made with pre-release 1.6.0 code): analysis tables
  gained regret columns, `ablation.csv` is retuned per variant, the `prediction` extra became `deployed` (with
  `selective`), and figures are drawn by `qemx.selection.plots` with `configs/selection/figures.yaml`; rerun
  `analyze` (or `analyze --splits "" --extras deployed,…`) and `figures` on such a directory. Labels are unchanged
  and need no relabeling. A configuration key `seed` under a family is the transpiler seed; use `angle_seed` for a
  random family's instance seed (decision D24).
- **Version** 1.6.0 in `pyproject.toml`, `qemx.__version__` and `CITATION.cff`; Python ≥ 3.10 as before; the lock
  file is unchanged.

# Part A — from 1.5.0 to 1.5.1

## A1. What changes meaning, and what to recompute

| artifact (1.5.0) | in 1.5.1 | action |
|---|---|---|
| exponential-ZNE and symmetry-verification rows (Playground, Batch, exports, phase studies) | `estimate`, `bias`, `sd`, `RMSE` are the protocol's Monte-Carlo moments at the budget; the exact-probability fit is `exact-fit value` (estimate kind `reference`); a declared failure policy (`on_fail`) | recompute any such number you quote; 1.5.0 reported the exact fit as an exact finite-shot mean and left failed fits undefined |
| phase records | schema `qemx.phase.cell/4`: target-nonlinear rows add `target_moments` and `failure_policy`; provenance adds source/lock/inventory digests | `/3` records still read; `run` refuses to mix `/3` and `/4` in one directory (use a new `--out`) |
| hardware/replay rows | a single run shows `signed error to reference` and `absolute error` | read 1.5.0 "bias"/"RMSE" of a measured row as these |
| PEC hardware plans and local rehearsals | PEC keeps its shots per sampled circuit; a cap is an explicit effective protocol; shot totals equal B; replay variance without double-counted shot noise | re-plan before any transfer comparison; 1.5.0's "shots per PEC circuit" was N_t/K and its PEC sd up to ~17 % too large |
| phase-ledger feasibility with postselection or PEC s > 1 | one feasibility rule | derived tables change only for those rows at small budgets; the bundled pilot has none (its tables are byte-identical) |
| `tables/methods.json` of a study report | schema `qemx.methods/2` (resolved protocols and noise context) | `python -m qemx.study context RUN --manifest M --out REPORT` regenerates it (the old file is kept as `methods.v1.json`); a `/1` file is refused for live recommendation |
| feature tables | schema `qemx.features/2`: `routing_overhead` → `two_qubit_compile_overhead` (now signed), `routing_ratio` → `two_qubit_compile_ratio`, `two_qubit_density` → `two_qubit_per_qubit_layer`; new `non_clifford_slots`, observable-setting features | old columns load through aliases, except `routing_overhead`, which 1.5.0 clamped at 0 and is refused |
| `paired_improvement.csv` | adds `planned`, `feasible`, `infeasible`, `better`, `share_better_all`, `share_better_feasible`; `share_better` keeps its conditional meaning | quote the share with its denominator |
| recommender evaluation | intervals are group-CV conformal (1.5.0: ±1.645 × in-sample residual SD); decisions use out-of-fold residual SDs; adds full policy, harm, tails, paired comparisons | regenerate the report (`python -m qemx.study report …`); conclusions of the pilot do not change (it still does not beat the best fixed method) |
| manifests with stage-level `confirmatory` policies | one study-wide policy; conflicting policies are refused | declare a top-level `confirmatory`; a frozen 1.5.0 split stays as it was (same hash input) |
| campaign ledgers | schema `qemx.campaign.ledger/2`, transactional; reservations are max(high estimate, per-job limit); unknown outcomes stay committed | a `/1` ledger migrates on its next change; its `reserved` entries become `submission_unknown` and must be resolved once (`python -m qemx.campaign resolve`) |

## A2. Behavior changes

- **Hardware submission requires a ledger** at every real-provider entry point (app, `hardware.submit`,
  `divergence.submit`, the exported script's `--ibm` path); a configured path that does not exist is an error. Job
  mode is the default. `resolve` needs `--job-id` or `--not-submitted`. `collect-ibm` needs the ledger (or an
  explicit `--no-ledger`).
- **Per-job limits.** `max_execution_time` defaults to ⌈high estimate⌉ s (1.5.0: ⌈1.1 × high⌉ + 1 s).
- **Large hardware jobs** are split by the backend's circuit limit (`max_experiments`, 300 for fake backends).
- **Live recommendation** abstains (`abstain-context`) when the experiment's protocols or noise context differ from the
  study's; in device mode the noise features come from the selected device, never from the noise injector.
- **Tabs** of the research and analysis pages keep their selection in app state (robust on Streamlit 1.64 and 1.66).
- **Comparisons are cached per session.**
- **Local hardware emulation** (`hardware.run_local`) now includes idle relaxation and asymmetric readout, like the
  predictions; 1.5.0's emulation was about 20 % quieter on QFT-4.
- **Head-to-Head noise sweeps** cost exponential ZNE and symmetry verification with their budget-dependent moments
  (rows gain `target moments`); their curves and winner maps change only for those methods.
- **Live recommendation** costs PEC with its shots per sampled circuit and post-selected protocols with their measured
  acceptance (the study ledger's rule).
- **Final holdout.** `python -m qemx.study holdout` refuses a holdout run whose protocols or noise context differ from
  the development run (`--allow-context-mismatch` for a labeled diagnostic evaluation).
- **Transfer test.** `run-ibm` writes `frozen_predictions.json` (the design's model on the day's calibration) and
  `collect-ibm` reads it; a 1.5.1-preview `predictions.json` in a run directory is not used. Interrupted runs resume
  without re-sending jobs.
- **Python ≥ 3.10** (as 1.5.0). The lock is tested on Python 3.12 with Streamlit 1.66.0; Streamlit ≥ 1.64 is
  supported. On Linux, set `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048` when scikit-learn ≥ 1.8 is installed
  (README, Linux note); CI does.

## A3. New

`qemx/provenance.py`, `qemx/decision.py`, `qemx/transfer.py`, `qemx/rec_context.py`; `scripts/hardware_transfer.py`,
`scripts/diagnostic_policy_study.py`, `scripts/reproduce_pilot_report.py`, `scripts/validate_protocol_moments.py`;
`data/pilot_v2/` (primary records); `manifests/direction_A_v2_final_holdout.yaml` (frozen); `python -m qemx.study
holdout`, `python -m qemx.phase.run holdout-check`.

## A4. One input model in the Playground and in documents

**1.5.0.** A template's input (the QFT's F†|s⟩, BV's ancilla |−⟩, QPE's eigenstate) was written into the circuit as
gates tagged `prep`, while the per-qubit inputs still showed |0⟩; the QFT's *input index s* and *input preparation*
were template-panel parameters that took effect only on *Load template*; a per-qubit input was applied *before* the
template's preparation (both at once).

**1.5.1.**

- The input is one state per qubit, set only in *Inputs and measurement* and prepared once by the compiler.
  Loading a template moves its preparation gates into the per-qubit inputs; the circuit keeps the rest. Each input
  token is written with the template's own gates, so the compiled circuit — and every noisy result — is unchanged
  gate for gate (the QFT's Fourier input therefore shows as phase inputs `ph:t`, e.g. `ph:0.5` for H·P(π)|0⟩, not
  the canonical |−⟩ = H·X|0⟩, which an abstract noise model would charge for one more gate).
- **Config hashes of template experiments change** (`qemx.experiment`): the hashed state now holds the input tokens
  instead of preparation gates. The numbers do not change (the seeded `qft_known_answer.yaml` reproduces the 1.5.0
  rows to within 1e-11), but `run` into a 1.5.0 result directory asks for a new `--out` or `--force`.
- Input-only template parameters (`role="input"`: QFT/AQFT `s` and `prep`, QPE-Ising `state`, the logical `theta`
  of `qec_3q`, `repetition`, `steane`, `five_qubit`, `surface_d2`) are shown in that panel and apply at once; they
  never change the circuit. Their names and values are unchanged, so parameter sets, cited presets, Learn setups,
  sweeps and YAML files keep working.
- New input token `bloch:θ,φ` (radians) for general single-qubit states; the register-wide box also takes one state
  per qubit (`0 + 1`).
- Encoded algorithms (`qec_algorithm`) tag their logical injection and encoder `encode`: they are circuit, and their θ
  and s stay circuit parameters of the logical benchmark.
- The QPE-Ising and Shor templates write their input X gates first (before the Qiskit-decomposed body): the same
  gates in another order, preparing the same states (checked for every template and a parameter grid: fidelity 1 to
  1e-9). Under a device model the scheduled idle times of those two templates can differ slightly.

**Compatibility.**

- *States saved by 1.5.0* load unchanged in meaning: leading `prep` gates are folded into the per-qubit inputs (the
  same state, prepared once), and the Playground says so when it loads such a file. The CLI runs a saved state as
  written.
- *Short-form YAML*: the resolved state now holds the template's input as `inputs` and the circuit without its
  preparation (as the Playground), so the configuration hash of such a document changes once; the experiment is the
  same. An `inputs` field next to a template that loads its own input is refused unless it equals that input or is all
  `"0"` (1.5.0 silently applied both).
- *Batch sweeps*: `input:` now replaces a template's input (1.5.0 applied it before the template's preparation).
- The phase-study families keep the preparation in their circuits (same state; the identity of the frozen pilot and
  holdout circuits is unchanged).

# Part B — from 1.4.x to 1.5.0

1.5.0 changed some reported numbers, because they then described the protocol that was actually run. It renamed
one method id, and it added an experiment file format.

## B1. Learning methods (CDR, FR, NIL): what `estimate` and `RMSE` mean

**1.4.2.** A learning method's row showed two different things:

- the **exact-training reference fit**: the regression fitted to exact training expectation values, an oracle;
- the spread of **finite-shot refits**.

So bias and RMSE described a protocol nobody runs. In the QA report's diagnostic cases, the reported and the empirical RMSE differed considerably:

| case | 1.4.2 reported | 1.4.2 empirical | 1.5 reported | 1.5 empirical (6 000 independent experiments) |
|---|---|---|---|---|
| CDR-OLS | 0.0880 | 0.1354 | 0.1328 | 0.1354 (z = −0.90) |
| FR-Ridge | 0.2704 | 0.3973 | 0.3986 | 0.3973 (z = 0.47) |
| NIL-ℓ1 | 0.1754 | 0.1818 | 0.1825 | 0.1818 (z = 1.44) |

Reproduce the comparison with `python scripts/validate_learning_moments.py --out validation_learning.json`. The pass criterion is |z| ≤ 4 against the declared Monte Carlo errors.

**1.5.** The fields are now separate (`qemx/stats.py`):

| field | meaning |
|---|---|
| `estimate_kind` | `exact` (no finite training), `protocol` (expected over the finite training data), or `measured` (one hardware experiment) |
| `mean`, `var1` | the protocol's moments: μ = E_D[m(D)] and Var = Var_D[m(D)] + E_D[v(D)]/N_t |
| `reference_mean`, `reference_var1` | the exact-training fit, kept for comparison (the table column *exact-fit value*) |
| `moments` | the Monte Carlo record behind `mean`/`var1`: repeats, finite draws, MC standard errors, failure rate, tolerance |
| `flags` | e.g. `conditional-on-success`, which means some refits failed and the moments are conditional on success; `calibration-mean-first-order` |

**What to do.**

- Recompute any 1.4.x Playground, Batch or export numbers for learning methods that you want to quote.
- Phase-study records written by 1.4 used their own, already protocol-level, ledger. They are still readable. They use record schema `qemx.phase.cell/2`; 1.5 writes `/3`.
- To re-run a study, use a new `--out` directory. `run` refuses to mix the two schemas unless `--force` is given.

## B2. Method id `features: nil` → `features: fr`

The learning estimator's feature-regression variant (FR: the noisy ⟨O⟩, ⟨Z_i⟩ and ⟨Z_iZ_j⟩) was selected with `features: "nil"`. That name reads as neighbor-informed learning (NIL), which is a different method: `estimator: nil`.

| where | 1.4.x | 1.5 |
|---|---|---|
| recipe parameter | `{"estimator": "learning", "features": "nil"}` | `{"estimator": "learning", "features": "fr"}` |
| presets | "FR · Ridge", "FR · Lasso" (already so named) | unchanged, now `features: fr` |
| manifests | method keys such as `NIL-L`, `NIL-R` with `features: nil` | still valid; the key names are kept so cell ids do not change, and a note is printed: `learning: features='nil' is the legacy name of features='fr' (migrated)` |

The legacy value is translated when a recipe is built (`Recipe.make`). The manifest file itself is never rewritten. New manifests (`direction_A_v2_*.yaml`) use the keys `FR-R` and `FR-L` for feature regression and `NIL-G2` for neighbor-informed learning.

## B3. Circuit input and output

**1.4.2.** The QFT templates had a parameter labeled **"output s"**. Setting it changed the *input* preparation: Hadamard and phase gates that prepare F†|s⟩. Choosing output 0 inserted |+⟩ preparation.

**1.5.**

- The parameter is **"input index s"**.
- A new parameter, **"input preparation"**, takes one of two values:
  - `fourier`: the known-answer benchmark F†|s⟩, whose ideal output is |s⟩;
  - `basis`: |s⟩ prepared with X gates, whose ideal output is uniform.
- The Playground shows a read-only **Output** panel: the ideal prediction derived from the input, the circuit and the measurement. The noisy prediction and the sampled outcomes stay in *Mitigate & compare*.

**Compatibility.**

- A parameter set without `prep` means `fourier`, which is exactly the 1.4.2 circuit (tested).
- State files store the circuit's operations explicitly, so a saved 1.4.2 state reproduces the same circuit.
- The phase-study QFT family is unchanged.

## B4. Readout mitigation

`Context.readout_gamma` now returns the inverse 1-norm product ∏_i (1 + |e_i − h_i|) / |1 − e_i − h_i|. For two bits with e = h = 0.6 it gives **25**; 1.4.2 displayed 1e24. TREX uses the same formula with e = h = p̄.

Two consequences:

- An assignment matrix with a negative determinant (a readout that mostly flips) is invertible and accepted.
- A singular matrix (P(1|0) + P(0|1) = 1) is an error with an explanation. It is no longer clamped.

## B5. Budget preflight

The Playground checks every contender's fixed cost (training and calibration shots) before running anything. A contender that cannot run at the total budget is shown as **"insufficient_budget: needs at least N shots"**: it is not run, and it does not count as 120 failed repeats.

Studies (`compare(..., check_budget=False)`, the default) still run every recipe, because their cost curves are evaluated at many budgets.

A budget-scaled teaching preset, **"NIL · small budget"** (T = 6, 3 neighbors + the bare circuit, 128 shots each, so 3 072 training shots), runs within 4 096 shots. It is not the configuration of the NIL paper.

## B6. Guides (Learn page) and contenders

- **Saved custom recipes are now separate from active contenders.** Older states without `custom_active` treat every saved recipe as active, as before.
- **Applying a guide** *replaces* the active contenders by default, so every guide starts from its own experiment. Saved recipes are never deleted. The toggle *Keep my current contenders* merges them instead.
- **Setups are atomic.** Every field is validated first. If anything fails, nothing changes.

## B7. Exports

PNG and SVG rendering runs in a separate, time-limited process. A missing Chrome, a crashed or a hung renderer returns the ZIP promptly, with every table, the state and each figure's Plotly JSON, plus any images finished before the failure. The README inside the ZIP says what is missing. Turning off the sidebar toggle *PNG/SVG images in ZIP downloads* skips rendering.

## B8. One experiment file for the app, the CLI and YAML

Experiment documents use `schema: qemx.experiment/1` and are validated by the same schema as the app's state files (`qemx/state_schema.py`).

- **Getting one:** the Playground's **Export → Download experiment (.yaml)** writes the current experiment. **Load a state or experiment** accepts `.json` and `.yaml`.
- **CLI:**

  ```bash
  python -m qemx.experiment validate|plan|run FILE [--out DIR]
  python -m qemx.experiment from-state qem_explorer_state.json --out experiment.yaml   # 1.4 state → YAML
  ```

- **Guarantees:**
  - Errors name the field, for example `methods[1].params: …`.
  - Credentials anywhere in a document are refused.
  - YAML is read with `safe_load`.
  - The configuration hash identifies the resolved experiment. `run` never overwrites a result of another configuration.

## B9. Divergence tables

`divergence.csv` gains the columns `p_value` (two-sided normal), `p_holm` and `q_bh`. The latter two are adjusted within each predictor × method × source family. The meaning of other columns has changed:

- `significant` is now **Holm-adjusted p < 0.05**. In 1.4 it was the unadjusted |z| > 3.
- The unadjusted flag is kept as `abs_z_gt_3`.
- `structures.csv` adds the columns `holm_worse` and `holm_better`.

## B10. Hardware submission

`scripts/dirA_divergence.py run-ibm` now requires a campaign ledger (`python -m qemx.campaign init`). It admits jobs one by one against the shared 600 s ceiling, defaults to **job mode** (`--mode batch` is still available), writes `jobs.json` after every job, and sets each job's `max_execution_time` from its own high usage estimate.

The IBM page uses the ledger when `QEMX_CAMPAIGN_LEDGER` points to one (in 1.5.1 it is required). See `docs/hardware_preflight.md`.

## B11. Navigation

| 1.4.2 sidebar | 1.5 |
|---|---|
| Cost vs precision, Bias–variance bullseye, Head-to-head sweeps | drill-downs of **QEM Performance & Selection** |
| Model mismatch | drill-down of **Noise Models & Discrepancy** |

Page addresses, Learn setups and bookmarks are unchanged.
