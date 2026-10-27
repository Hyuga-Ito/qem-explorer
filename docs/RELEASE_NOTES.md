# QEM Explorer 1.9.1 — release notes (2026-10-26)

1.9.1 is a focused corrective release of 1.9.0, made before local experiments start. It answers the independent
review of 1.9.0 (F1–F5 and its research-design notes); each finding was reproduced on the 1.9.0 release first and
fixed with a regression test (`docs/ISSUE_EVIDENCE_MATRIX.md`, part 1.9.1). No new research feature; **no scientific
result changes** — the 1.8.0 confirmation, the 1.9.0 pilot, the execution audit and Figure 1 are unchanged
(regenerated: tables s1–s5 and Figure 1's PNG and CSV byte-identical; its PDF/SVG differ only in embedded creation
dates, so the shipped files were kept); only the planning table s6 changes. Migration: `docs/MIGRATION.md` (Part 00000). Verification: `docs/QA_SUMMARY.md` (part 1.9.1).
Windows: `docs/WINDOWS_FIRST_RUN.md` (commands marked tested on Linux / not tested on Windows).

## Fixes

- **F1 — the shipped full-development order could not finish.** `analyze` evaluated the deployed and nested tables
  before `freeze`, with a fallback configuration outside fulldev19's grid; the `nested` stage reused them because the
  file existed, and `freeze --complete` refused (exit 2). Frozen-configuration analyses are now deferred until the
  freeze, carry a binding (configuration, data, evaluation settings, code) that the `nested` stage and
  `freeze --complete` check, and stale tables are set aside (`analysis/stale_nested_<time>/`) and recomputed. A run
  stopped by 1.9.0 recovers with `pipeline … --accept-changes` without changing its frozen configuration; an
  extras-only analysis no longer re-pools the table the freeze was chosen on (found while testing that recovery).
- **F2 — `freeze` accepted a stale analysis.** The first freeze now checks the analysis against the configuration,
  labels, data and code, requires every declared split and grid configuration, and chooses only within the declared
  grid; a code change needs `--accept-changes` (recorded).
- **F3 — CPU ceilings forgot earlier attempts.** A persistent consumed-resource ledger (every attempt of every
  invocation, including failed, retried, quarantined and interrupted ones; legacy and imported records without double
  counting) is what `limits.cpu_hours`, admission and `status` use. Still summed worker wall time, not a CPU meter.
- **F4 — the cost table's wall time.** Computed for a declared hypothetical computer (8 cores, 7 workers: 1.3 h, not
  the generating computer's 2 cores under a "7 workers" caption); the 9.3 CPU-hour labeling figure is a conditional
  estimate, not an upper bound; the analysis is sized by model fits (`plan` prints them) instead of a row-scaled time.
- **F5 — process supervision in containers.** Workers watch their parent through the multiprocessing parent handle;
  psutil is used only where its PID view matches the process's own; lock liveness uses the process's own PID view.
- **Descriptive strata.** Summary rows carry `inference`: one test circuit → descriptive.
- **Independent debugging.** Two sub-agents (Sonnet 5.5 and Fable 5.1) on separate copies found ten further defects
  in the candidate — e.g. a retry copy counted as a second attempt, `regimes`/`profile` failing before a freeze, the
  nested stage moving tables it could not recompute — all fixed with regressions and re-checked by them.

## Research planning (documents only)

conf19 stays unexecuted. Its population (five families; no QFT/BV/Grover) is stated, and the estimand, a power table
and three sizing/composition options are written out for your decision (`docs/selection/RESEARCH_PLAN_v19.md`).
Controlled learning-curve/ablation readings within fulldev19 are specified, simulator-assisted and simulation-free
recommendations are kept distinct, and the "10–15 CPU-hour" analysis estimate is withdrawn.

## Not changed (nonblocking; listed in the issue matrix)

Execution-audit trial-level outcomes are still not saved; Figure 1's caption wording and summary inset; hatching
one-circuit strata in figures; label-precision/design-count downsampling. Native Windows and GitHub Actions: not run.

# QEM Explorer 1.9.0 — release notes (2026-10-25)

1.9.0 prepares the research pipeline for large local experiments. It answers the independent review of 1.8.0 and the
second review that supplements and partly corrects it; each finding was first reproduced on the extracted 1.8.0
release (or found not to be a defect), then fixed with a regression test (`docs/ISSUE_EVIDENCE_MATRIX.md`). An
independent debugging pass on a separate copy found further defects in the new code, all fixed before release.
Migration: `docs/MIGRATION.md` (Part 0000). Verification, with what was and was not executed: `docs/QA_SUMMARY.md`.
Windows: `docs/WINDOWS_FIRST_RUN.md`. Owner decisions: `docs/OWNER_DECISIONS.md`.

**No result of 1.8.0 changes.** The defects below did not affect the complete, single-sequence 1.8.0 run: its 217
per-trial arrays verify, its primary contrasts are recomputed from the arrays (−0.14406 and −0.08397, equal to the
saved replays to 10⁻¹³), and the 580 SBS-B choices of its replay follow the SBS-B rule. Two statements in the text of one
supplementary figure were wrong and are corrected beside the historical file (`docs/selection/ERRATA_v180.md`).

## Running large studies safely (second review N1–N3, N5–N7)

- **Truthful completion (N1).** Run-wide completion — expected, completed, failed by reason, missing, corrupted,
  unexpected — reported by every labeling command and `status`; exit 0 only when complete (1 incomplete, 2 refused,
  3 `plan` over a cap). Failed tasks are retained and reported on each rerun; `--retry-failed` keeps the history
  (`attempts.jsonl`, `failed_attempts/`); missing or changed arrays stop labeling until `--quarantine-corrupted`; `run`
  stops on a labeling failure; `analyze --allow-partial` marks partial results, which `freeze` refuses.
- **Resource limits (N2).** Preflight from measured rates (the run's own or `--measured`; unknown stays unknown; disk
  calibrated on 1.8.0), admission against `limits.cpu_hours`, `disk_gb` and the new `min_free_disk_gb`
  (`--override-limits "reason"` recorded), monitored ceilings during labeling (checkpoint-safe), `--workers auto`
  bounded by memory; `plan --tasks-csv`/`--json`.
- **Interrupt and resume (N3).** `python -m qemx.selection pipeline FILE --out DIR`: stages as argument arrays (no
  shell, so paths with spaces and Japanese characters work on every platform), verified reuse of complete labels and of
  once-only steps (`freeze`, `final` and `confirm` verify an existing result; with `--reuse` they then exit 0), every
  failure stops its dependants, logs kept as bytes. One writer per labels directory and analysis (lock; stale and zombie locks replaced); Ctrl+C,
  SIGTERM and Ctrl+Break stop the workers and release the lock; workers whose parent was killed end themselves.
  `scripts/reproduce_v18.sh` is now a wrapper of `configs/pipelines/reproduce_v18.yaml`; no step masks failures.
- **Bound replays and derived tables (N4).** Replay scores the manifest's population and refuses an incomplete one;
  an exploratory subset needs a stated reason and is recorded; a cached `labels.csv` must equal the table derived from
  the arrays (`StaleTable`); the SBS-B column of a choices file is checked against the rule; provenance (run identity,
  data digest, bundle digests, estimand) in every replay. Freezes record the development data digest and the
  scientific configuration digest; analyses record their identity and refuse silent merges; `confirm` checks the
  logical circuits against the training labels actually used; bundles go into the run (`RUN/bundles`) and never
  overwrite.
- **Trial-data CI (N5).** The optional job verifies only the study whose attachments it receives
  (`reports/TRIAL_ATTACHMENTS.json`) and fails, never skips, when they are missing (`QEMX_REQUIRE_TRIALS`). New
  `tests/test_trial_data_v180.py` (manifest, completion, derived labels, recomputed contrasts, disjointness, an edited
  baseline refused). The Windows job adds partial labeling, a partial analysis, `status` and, in a path with spaces and Japanese characters, the pipeline's dry run, its plan stage and two labels.
- **YAML semantics (N6).** Closed key sets inside `analysis`, `split_options`, `mismatch`, `regimes` and `limits`;
  unique, non-empty lists; `master_seed ≥ 0`; semantic checks before labeling (`design.check`); `check-config` routes
  every YAML format to its own parser (`docs/selection/YAML_FORMATS.md`); strict plotting configurations;
  `expansion_policy: explicit`; the expansion report shows inserted default variants and family-forced inputs.
  Historical task identities are unchanged (1989 identities of 19 configurations pinned).
- **Current presets (N7).** `smoke19`, `pilot19` (both executed), `fulldev19` and `conf19` (planned) — new logical
  circuits disjoint from every earlier study, noise-diverse training, held-out `noise`, `device-base` and `joint`
  splits; the historical configurations are unchanged.

## One recommendation everywhere (N8, R1, R8)

- A state's declared symmetry is kept when it holds on the ideal measured distribution (inputs, circuit, bases, bit
  order; up to 20 qubits) and otherwise dropped with a note; duplicate qubits are rejected. Study, state, YAML
  experiment, app and CLI now give the same task key, device, layout, preflight evidence and choice.
- An explicit device overrides the state's backend; layouts are validated; invalid inputs are `Unsupported`.
- `recommendation_signature` (task, bundle, policy, simulation permission, preflight level) drives the app's
  staleness mark; results show the effective device, physical layout, symmetry and conversion notes.
- Experiment documents keep a template's further parity checks (`symmetry_more`).

## Research and reporting (R2, R3, R5; prompt §§3–5)

- **Corrected captions (R2).** The supplementary deployment-policy figure names the primary (frozen configuration
  with simulation; D41) and the secondary (deployable) bundle in the right order, states when D41 was written, and
  what the labels estimate (fresh-design risk). The first review's criticism of the 47-row explanation was not a
  defect: the caption already separated 33 accepted alternatives and 14 out-of-distribution fallbacks.
- **Execution audit (R3).** `scripts/execution_audit.py` scores what a user would actually run — recommendation →
  returned verified design → allocation → execution — for the policy, SBS-B and RAW, over five deployment roots
  (the eight circuits of the 1.8.0 root audit — 1.7.0 regression cases, earlier failure types and controls; inspected, but
  in neither the bundles' training set nor the confirmation — three budgets, 600 trials per row). Executed minus fresh-design
  contrast: primary −0.024 [−0.047, +0.000], default bundle +0.005 [−0.036, +0.061]; budget ledgers: none over, none
  unspent. Only 10 of 480 rows involved a learning candidate, so this is weak evidence for those candidates.
- **Pilot (R5).** pilot19 executed (27 new base circuits, 54 tasks, six calibration-supplied noise settings; 0.86
  CPU-hours of labels, 71 minutes of analysis): no learned policy beats SBS-B out of fold (deployable-feature learners
  are worse by 0.09–0.14 decades on the same-distribution split; only a three-circuit joint split favours one model);
  exploratory and for planning (`reports/pilot_v190/`, `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §I.2).
- **Figures and tables.** A versioned specification of Figures 1–6 and Tables s1–s6
  (`configs/selection/figure_spec19.yaml`, `docs/selection/FIGURE_TABLE_SPEC_v19.md`; Figure 6 pending — no
  17-candidate mismatch study), Figure 1 (policy benefit, four panels) and the tables as CSV, LaTeX and PDF in
  `reports/tables_v190/`, with tests that a table cannot take the wrong source column.
- **Plans.** `docs/selection/RESEARCH_PLAN_v19.md` (designs, measured costs, the freeze checklist before the reserved
  confirmation); interpretation and ranked next experiments in `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md`.

## Not done in 1.9.0

- The full development study and the reserved confirmation were not run (as instructed); no hardware job was
  submitted; nothing was pushed, tagged or published, so no GitHub Actions run exists for 1.8.0 or 1.9.0.
- Native Windows was not executed (the Windows CI job will run it after a push); the commands were executed on Linux.
- Learning burdens as model inputs (R6), a 17-candidate mismatch study (R7) and VD/M3 (R4) remain open.

# QEM Explorer 1.8.0 — release notes (2026-10-25)

1.8.0 answers the review of 1.7.0 and its follow-up, fixes the two failures of the 1.7.0 GitHub Actions run, and runs a
new, pre-declared selection study with an expanded portfolio of 17 candidates. Every finding, its change, its evidence
and what is not claimed: `docs/ISSUE_EVIDENCE_MATRIX.md`. Migration: `docs/MIGRATION.md` (Part 000). Methods changes:
`docs/selection/METHODS.md` §16. Verification: `docs/QA_SUMMARY.md`. Release consistency and the GitHub update
checklist: `docs/RELEASE_CONSISTENCY.md`. Decisions only the owner can make: `docs/OWNER_DECISIONS.md`.

## The two CI failures of 1.7.0 (`docs/CI_FAILURE_ANALYSIS.md`)

- **`lam_lc` exact comparison.** The runner's NumPy dispatched AVX-512 kernels whose last-bit differences the inverse
  Walsh–Hadamard step of `noise.pauli_probs_from_eigs` amplifies (≈ 700× at p ≈ 10⁻⁴). The test now compares that one
  feature at relative tolerance 10⁻¹² (10⁵× below a physical change, shown by a test); scientific code is unchanged;
  CI reruns the numerically sensitive tests with AVX-512 disabled. Reproduced locally by disabling AVX-512 dispatch.
- **Unicode directories under a Latin-1 locale.** File-*name* encoding (the locale on Linux, UTF-16 on Windows) is now
  separated from text-*file* encoding (always UTF-8); fixtures use names the locale can represent, a dedicated test
  checks a clear `PathEncodingError`, and a split-locale child process emulates Windows. A third latent failure on
  Python 3.10 (`locale.getencoding` is 3.11+) was found and fixed in the same test.
- Both pass locally under the lock, Python 3.10/3.12/3.13 and the legacy locale; the GitHub run of 1.8.0 is pending
  (nothing was pushed).

## Foundations (findings F1–F10, N1)

- **One execution/preflight contract (F1)** — verified-feasible / infeasible / unverified at the structural, design and
  allocation level, shared by labeling, `recommend`, the app, the replay and the hardware plan. Policy
  `qemx.selection.policy/2` never returns an unverified candidate, lists the skipped ones and falls back to SBS-B's
  first verified candidate; the margin runner-up is the next verified candidate. The ten 1.7.0 deployment regressions
  are resolved. The confirmation now scores the **full deployed policy** (`deploy.recommend` itself).
- **Bounded cost (F2)**: a guard before compilation, factorized calibration marginals, compact Pauli observables; a
  24-qubit request never reaches a dense engine (sentinel test).
- **Provenance (F3, F4, F9)**: records bound to their run (mismatch labels included), bundles checked for scientific
  compatibility (feature, protocol and snapshot fingerprints), strict JSON with a `"$nonfinite"` map, a study index in
  the ancestry registry (with declared post-hoc reuse).
- **Statistics (F5, F6, F7)**: the random-effects variance of one fresh deployment beside the pooled variance; the
  metric described as a factor of the geometric mean of (MSE + 10⁻⁶); declared tail metrics (severe vs RAW, risk > 0.1,
  q95, worst-5 % loss share) in every summary, with missing risks masked.
- **Independent shot-level references (F8)** for every estimator whose labels drive decisions (ZNE-E, ZNE-E-B, CDR, FR,
  NIL with its own ℓ1 solver, SV, ZNE∘SV, TREX, RO) and direct execution for the linear and transform candidates.
- **Light-cone descriptor (F10)**: its assumptions stated; tests show it cannot see coherent or correlated noise.
- **Preparation seeds (N1)**: seed scheme 2 — streams from stable identifiers, explicit common-random-number groups;
  scheme 1 reproduces 1.7.0 bitwise.

## Candidates

17 candidates (`docs/selection/CANDIDATES.md`, `python -m qemx.selection registry`): the eight 1.6/1.7 protocols plus
RO-TREX, RO-TW, RO-DD, ZNE-L-TW, ZNE-E-B (exponential ZNE projected onto the observable's range), PEC-TW, FR (feature
regression), SV (symmetry verification of declared checks) and ZNE-L∘SV, each with a capability record. The 1.6/1.7
candidates keep their identities (`portfolio-1.6`).

## Research (`docs/selection/RESULTS.md` Part I; `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md`)

All results are simulated (fake-device emulation); nothing was submitted to hardware.

- **Confirmatory (D41).** On 29 new logical circuits, evaluated once, the full deployed policy of the frozen
  configuration (boosted trees, deployable + simulator-only features) is −0.144 [−0.231, −0.076] against SBS-B in mean
  log10(MSE + 10⁻⁶), and its development out-of-fold difference is −0.088 [−0.145, −0.027]: the declared rule is met.
  Qualifications: the optimism-corrected development difference is not resolved (−0.064 [−0.143, +0.013]); D41 (the
  full policy as the primary estimand) was written after development results were seen; the configuration needs exact
  simulation at recommendation time.
- **Default bundle** (no simulation): −0.084 [−0.179, −0.002], not robustly resolved (bootstrap-seed sensitive); about
  half of its gap to its model component comes from symmetry verification that the preflight cannot verify without a
  simulation, half from out-of-distribution fallbacks on the same GHZ circuits.
- **Portfolio**: on identical development labels, SBS-B over 17 candidates beats SBS-B over 8 by −0.070
  [−0.133, −0.012]; learned policies on deployable features are not resolvedly better than SBS-B in either portfolio.
- **Tails**: ZNE-E-B caps ZNE-E's divergent estimates (largest MSE 10⁹ → 0.04) at the same severe rate; FR and NIL are
  severe on 61 % and 48 % of feasible rows (circuit-weighted); FR's failures peak at five qubits, where its 17
  regressors exceed its 16 training circuits (an untested interpolation-threshold hypothesis).
- **Root audit**: label standard errors about honest across four preparation roots (not understated on average);
  contract agreement 288/288; deployed policy stable across roots.
- An independent re-implementation reproduced the confirmation, full-policy, model-component and attainable-portfolio
  point estimates.
- Not executed: the mismatch study under the 1.8.0 policy (Figure 6 pending), hardware validation (dry-run plan only),
  a larger study.

## Outputs

- `reports/selection_v180/`: the study's records, tables, figures and thesis tables; the per-trial arrays ship in
  `qem-explorer-1.8.0-trials-1.zip` and `-trials-2.zip`.
- Four new supplementary figures (deployment policy, portfolio expansion, root audit, tails); the 17-candidate palette
  uses colour plus a unique marker per candidate (validated for normal and colour-blind vision); every figure was
  reviewed for legibility with 17 candidates.
- Thesis tables from one generator as CSV, LaTeX fragments, Markdown and a standalone PDF
  (`python -m qemx.selection.thesis_tables`).
- `scripts/reproduce_v18.sh` reproduces the study (bundles are written into the run directory, never over the shipped
  ones).

## Application and command line

- *Recommend for my circuit* runs policy/2 and shows the guard, every candidate's preflight status and level, the
  skipped candidates, the execution note and each candidate's capability record; the 1.8.0 deployable bundle is
  preselected, the frozen 1.8.0 bundle asks for permission to simulate. The QEM Performance page lists the 1.8.0 study
  first.
- `python -m qemx.selection hardware plan` writes to `runs/hardware_plan` by default and never submits; the plan's
  instructions now name the existing `hardware check` command and the approved submission path.
- Unchanged (and still tested): IBM Quantum tokens are entered in password fields, kept only as long as needed and
  never written to exported scripts or state files; the command line reads them from `QISKIT_IBM_TOKEN`, a saved
  account or a hidden prompt.

## Windows and release consistency

- `docs/WINDOWS_FIRST_RUN.md` (PowerShell, a check of the installation); `.vscode/` launch configurations set the Linux
  OpenMP reserve the README describes.
- One version source (`qemx/__init__.py`); `pyproject.toml`, `CITATION.cff`, the README title and these notes follow it
  (test). `.streamlit/` was checked and left unchanged. The obsolete-file check against `v1.7.0` and the GitHub update
  checklist are in `docs/RELEASE_CONSISTENCY.md`.

---

# QEM Explorer 1.7.0 — release notes (2026-10-22)

1.7.0 answers an independent review of 1.6.0. Every finding was checked against the code and data before it was
changed; the issue-to-evidence matrix is `docs/REVIEW_RESPONSE_1.7.md`. The foundations were repaired first (how CDR
and NIL labels sample their training designs, how abstention is evaluated, how runs are bound to their inputs, how
failures are collected), the pilot was relabeled with the corrected sampling, and the study was extended with focused
experiments. Migration: `docs/MIGRATION.md` (Part 00). Verification: `docs/QA_SUMMARY.md`. Decisions only the owner can
make: `docs/OWNER_DECISIONS.md`.

## Scientific corrections (each listed with its reason in `docs/selection/METHODS.md` §14)

- **Learning designs** (review R1). CDR and NIL labels now estimate the risk of a protocol that draws a fresh training
  design at deployment: each label half has its own pool of independent designs (10 in the pilot), every trial records
  its design, and standard errors are over designs (cluster-robust) with between/within-design variance components.
  The 1.6.0 labels (six designs shared by both halves) keep their own estimand and are kept as historical data.
- **Abstention** (R2) is evaluated by nested group cross-validation; the deployment threshold is frozen with the
  configuration (`freeze --complete`).
- **Independent circuits and ancestry** (R3, R4). `plan` counts tasks, written base circuits, independent logical
  circuits, variants, designs and trials separately; a canonical logical identity and a study-wide ancestry registry
  show which circuits earlier studies used; the 1.6.0 final test is historical; a new confirmation set of 25 logical
  circuits was registered before labeling and evaluated once.
- **Statements** (R5). One vocabulary (`qemx/selection/claims.py`); a group split-conformal interval beside the
  CV-residual ones; certificates name their units and say "clipped estimator"; the 2ε relation is a post-hoc check.
- **Aggregation** (S4, S5). Failure shares are the union of flags; policy MSE is weighted by base circuit; the single
  best protocol is ranked on one population with a declared fallback.

## Software corrections

- **Execution** (S1): supervised child processes on every platform (also with one worker), parent-enforced time and
  memory limits, outcomes ok / error / timeout / memory; the profiler no longer needs POSIX `resource`.
- **Failure records** (S2) are collected, never crash collection; corrupted records (array digest) are refused.
- **Run binding** (S3): a run directory is bound to its configuration, manifest, split, portfolio, device snapshots,
  code and environment; incompatible resumes and stale files are refused; reuse is an explicit import; `final` and
  `confirm` check the freeze before labeling.
- **Schemas** (S6) are enforced on load; unknown keys and family parameters are errors; invalid variants are errors.
- **Python 3.10** (S7): `tomli` fallback for the TOML test and a justified tolerance; the tests pass on 3.10.
- **Hardware plan**: merged PEC circuits were charged their shots once per occurrence (about 3·10⁸ shots for a run at
  B = 20 000), which declared a feasible plan infeasible; every run is now charged exactly B (with a test).
- **UTF-8 everywhere**: every project text file is read and written as UTF-8 (`qemx/textio.py`), a static checker
  (`scripts/check_text_io.py`) and runtime tests with Japanese, Greek and mathematical text and Unicode paths, also
  under a non-UTF-8 locale; CI jobs for a legacy locale and Windows.

## Application

- **Playground** — the Output panel separates the input state, the output state before measurement, the measured
  distribution in the chosen bases and the observable with its meaning; template identities are predictions only
  while their assumptions hold; *Apply to all* reports what it changed; inputs and bases stay independent.
- **Recommend for my circuit** uses the 1.7 recommender: the Playground state becomes the same `DeploymentTask` as
  `python -m qemx.selection recommend`; versioned model bundles without pickle (preprocessing, feature order,
  candidate definitions, training support, validation summary, threshold and fallback, statement kinds); only the
  requested features, without simulating the circuit unless allowed; out-of-distribution and unsupported results are
  reported. The 1.5 recommender is kept as a labelled legacy expander.
- The *Protocol selection study* tab lists the corrected study first and marks the 1.6.0 study as historical.

## Research (`docs/selection/RESULTS.md`)

All results are simulated (fake-device emulation). With the corrected labels the six protocols without a training
design are identical trial for trial to 1.6.0; CDR's and NIL's trial-level standard errors had understated their
uncertainty by a factor 1.1–2.5 and NIL's typical risk by 1.3–1.9. Choosing the protocol per budget (SBS-B) remains
most of the gain (−0.141 [−0.205, −0.073] decades against the single best protocol). The frozen recommender (boosted
trees, deployable + simulated features) is −0.043 [−0.105, 0.026] against SBS-B on held-out development circuits and
−0.087 [−0.161, −0.024] on the once-only confirmation (25 new logical circuits): favourable on both, resolved on the
confirmation only, so by the rule declared before the study the claim is **not established**. Budget is the only
information whose removal resolvedly hurts; models win clearly only when extrapolating beyond the training budgets
(ridge best; the structured b² + v/B model in between). Nested abstention does not beat random rejection. A coherent
fraction of the gate errors displaces PEC in all six cells tested while the recommender, which sees only Pauli
calibration data, keeps choosing it. At equal total budget no reaction to the echo diagnostic beats spending every
shot on the target, and the learned reaction does as well triggered at random. NIL's severe failures at 16 training
circuits are spread over designs on two of five tasks and concentrated in one undetectable design on another.
`docs/selection/RESULTS.md` §15 assesses every 1.6.0 claim: the CDR precision claim changed; the others survive,
some sharpened.

New experiments: the corrected pilot (same tasks and seeds as 1.6.0), the noise-composition study (one noise
component at a time), equal-budget mismatch reactions with held-out discrepancy types, a design study of CDR/NIL
training allocations, the structured risk model b² + v/((1 − f)B), semantic ablations, and the confirmation. Planned
and not executed: the larger study `full17.yaml` (runnable plan with estimates and caps) and the hardware validation
(`docs/selection/HARDWARE.md`, owner authorization required). Everything reported is simulated (fake-device
emulation); no hardware job was submitted.

## Figures and tables

All six main and eight supplementary figures revised to the review (signed regret on a diverging scale, semantic
ablations and noise composition, Ridge and the structured model in budget extrapolation, nested reliability panels,
single-deployment diagnostics and equal-budget reactions, design-aware bias–variance, best-vs-runner-up prediction,
counts and noise components in dataset coverage, statement kinds in certificates, failure semantics, trials by
design, runtime by component), a new design-variability figure, a methods box in every figure's notes, Table 1 with
tasks and base circuits apart, a studies table, Table 3 for the confirmation and Table 3h for the historical final
test, and a figure validator that parses every SVG/PDF and compares plotted numbers with the tables
(`scripts/validate_figures.py`).

## Distribution

- `qem-explorer-1.7.0.zip` — the source archive: code, documentation, configurations, the shipped recommender bundles,
  and the corrected pilot's records, tables and figures (`reports/selection_v170/`).
- `qem-explorer-1.7.0-trials-1.zip` and `-trials-2.zip` — the corrected pilot's per-trial arrays (`*.npz`, 347 files:
  development and historical final test; confirmation, regimes, noise composition and mismatch), to be extracted into
  the repository root; `reports/selection_v170/TRIALS_SHA256SUMS` verifies them. Needed only for re-analysis from
  trials, the per-trial drill-down in the app and two supplementary figures.
- The 1.6.0 pilot (`reports/selection_pilot/`) is kept as the historical dataset; its per-trial arrays and figures are
  omitted from the source archive (unchanged in the 1.6.0 release archive).
- Review material, run logs and intermediate evidence are not part of either archive.

## Known limitations

- Fake-device emulation only; 3–6 qubits; one realization of each calibration snapshot.
- 52 development logical circuits (40 + the 12 historical-final ones) and 25 confirmation circuits: claims about
  circuits in general are limited by that diversity; 10 learning designs per half are few clusters.
- NIL with 16 training circuits (the paper uses 256–5 000).
- Bounded-loss certificates are uninformative at these risks; split-conformal intervals need more calibration
  circuits than the pilot has for simultaneous levels.
- GitHub Actions (including the Windows job) was not run for this release; macOS untested.
