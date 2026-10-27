# QEM Explorer 1.9.1

An interactive laboratory for quantum error mitigation (QEM): build a circuit, inject exactly the noise you
want (or a real IBM device's calibration), look at the circuits each method really runs, scrub through the
state gate by gate, compare any set of methods on bias, variance, cost and robustness, run the same
experiment on IBM Quantum hardware within a guarded usage budget, and export a standalone Qiskit script that
reproduces every number. It is also the experimental codebase of a graduation thesis on finite-budget selection of
QEM protocols.

> Intuitive UI makes intuitive conceptual understanding.

All presets and example values are **illustrative**; none of them are measured results. Every study result in this
repository is a **simulation** (or an emulation, labeled so); no hardware result is included.

## Quick start

```
python -m venv .venv                      # Python 3.10–3.13
.venv\Scripts\activate                    # Windows   (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py            # then open http://localhost:8501
```

In **PowerShell** the activation is `.venv\Scripts\Activate.ps1` (if scripts are blocked:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`). A step-by-step first run on Windows, including a check of the
installation, is in `docs/WINDOWS_FIRST_RUN.md`.
With Anaconda: `conda create -n qemx python=3.12`, `conda activate qemx`, then the `pip install` line above.
`requirements-lock.txt` pins the exact versions the release was tested with (Python 3.12); use it for a reproducible
environment. **VS Code**: open the folder, run *Python: Create Environment* (venv, tick `requirements-dev.txt`), then
press **F5** with "QEM Explorer (Streamlit app)" selected.

**Linux: `ImportError: … libgomp….so.1.0.0: cannot allocate memory in static TLS block`.** scikit-learn ≥ 1.8 and
qiskit-aer each ship their own build of the OpenMP runtime; on glibc, the second one loaded into a process can exceed
the small static-TLS reserve, so the app's learning methods or the tests fail on import. Reserve more before starting
Python — `export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048` (add it to `.venv/bin/activate` to make it
permanent) — as the CI does. Windows and macOS are not affected; no package or code change is needed.

## What is new in 1.9.1

1.9.1 is a focused corrective release of 1.9.0, answering an independent review of 1.9.0 before local experiments
start. Every finding was reproduced on 1.9.0 first and has a regression test (`docs/ISSUE_EVIDENCE_MATRIX.md`, part
1.9.1); no scientific result changes, and the historical data, confirmation and figure are untouched.

* **Full development pipeline (F1).** The shipped `fulldev19` order failed at `freeze-complete`: the deployed and
  nested tables were computed before the freeze for a configuration outside the declared grid and then reused. They
  are now deferred until the freeze; reused nested tables are validated (configuration, data, evaluation settings,
  code), and stale ones are set aside and recomputed. A run started with 1.9.0: `docs/MIGRATION.md` (Part 00000).
  A quick `run` (no freeze) now leaves the figures of the frozen configuration pending until `freeze`.
* **Stale analyses are not frozen (F2).** The first `freeze` checks the analysis against the configuration, labels and
  code, requires every declared split and grid configuration, and chooses only within the declared grid; a code
  change needs `--accept-changes` (recorded).
* **CPU accounting that persists (F3).** A consumed-resource ledger over every attempt of every invocation (failed,
  retried, quarantined, interrupted); CPU ceilings and admission start from it.
* **Cost table (F4).** The wall time is computed for a declared hypothetical computer; the labeling estimate is
  conditional; the analysis is sized by its model fits (`plan` prints them), not by rows.
* **Process supervision (F5).** Workers watch their parent through the multiprocessing parent handle (works in
  containers whose PID views differ); lock liveness is checked in the process's own PID view.
* **Planning notes.** One-circuit test strata are marked descriptive; conf19's estimand, population and size options
  are written out (`docs/selection/RESEARCH_PLAN_v19.md`); conf19 stays unexecuted.

## What was new in 1.9.0

1.9.0 makes the research pipeline dependable for large local experiments. It answers the independent review of 1.8.0
and the second review that supplements and partly corrects it; every finding was reproduced (or not) on 1.8.0 before
anything changed (`docs/ISSUE_EVIDENCE_MATRIX.md`: reproduction, root cause, fix, regression test, result, what
remains). The 1.8.0 study's numbers do not change; two errors in the text of one supplementary figure are corrected
beside the historical file (`docs/selection/ERRATA_v180.md`). Behaviour changes: `docs/MIGRATION.md` (Part 0000);
verification: `docs/QA_SUMMARY.md`; Windows: `docs/WINDOWS_FIRST_RUN.md`.

* **Truthful completion.** Every labeling command reports the whole run — expected, completed, failed (by reason),
  missing, corrupted — and exits 0 only when it is complete (1 incomplete, 2 refused). Failed tasks are kept and
  reported on every rerun until `--retry-failed`; corrupted arrays stop labeling until `--quarantine-corrupted`;
  `status` reports any run; `analyze` and `freeze` refuse incomplete labels unless explicitly marked partial.
* **Resource limits that hold.** A preflight from measured rates, admission against `limits.cpu_hours`, `disk_gb` and
  `min_free_disk_gb` (refused unless `--override-limits "reason"`, recorded), monitored ceilings while labeling, and
  `--workers auto` sized by memory as well as cores.
* **Interrupt and resume.** `python -m qemx.selection pipeline FILE --out DIR` runs a study stage by stage (argument
  arrays, no shell; Windows, Linux and macOS alike), verifies and keeps completed once-only steps, and stops every
  dependent stage on a failure. One writer per run; Ctrl+C or SIGTERM stops the workers; orphaned workers end
  themselves.
* **Replay and tables bound to their data.** Replay uses the manifest's population (incomplete → refused; a subset only
  with a stated reason), cached tables are re-derived from the arrays, the SBS-B baseline is checked against its rule;
  a freeze is bound to its development data; analyses of another configuration, data or code are never merged silently.
* **One recommendation everywhere.** A saved state, a YAML experiment, the app and the command line give the same task,
  device, layout, symmetry and choice. A declared symmetry is used only when it holds; an explicit device overrides the
  state's backend; a shown recommendation goes out of date when the simulation switch changes.
* **YAML you can check before running.** Each YAML format is validated by its own parser (`check-config`); typos are
  refused inside nested sections too; semantic checks run before labeling; `plan --tasks-csv` exports the expanded task
  table; new studies use `expansion_policy: explicit` (`docs/selection/YAML_FORMATS.md`).
* **CI.** The optional trial-data job verifies only the study whose arrays it receives and fails (never skips) without
  them; the Windows job labels, analyzes partially and runs the pipeline's first stages in a Japanese path. (Not yet run on GitHub: nothing was pushed.)
* **Next research stage.** Smoke and pilot designs (both executed), a full development design and a reserved
  confirmation (planned, with measured costs), all with noise-diverse training and held-out noise, device and joint
  shifts; a joint execution audit (recommend → returned design → allocation → execution); a versioned figure/table
  specification and tables as CSV + LaTeX (`docs/selection/RESEARCH_PLAN_v19.md`, `reports/tables_v190/`).

**1.9.0 research in one paragraph** (emulation only; `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` Part I): the 1.8.0
confirmation stands unchanged (frozen full policy −0.144 [−0.231, −0.076] against SBS-B, recomputed from the arrays).
Executing the design the recommender actually returns, instead of scoring the fresh-design average risk, moves that
contrast by −0.024 [−0.047, +0.000] on eight audit circuits (weak evidence for learning candidates, which are rarely
returned there). An exploratory pilot with noise-diverse training (27 new circuits, six calibration-visible noise
settings) finds no learned policy better than SBS-B out of fold: deployable-feature learners are worse by 0.09–0.14
decades, more so among 17 than among 8 candidates, and a budget rule refitted per noise setting is worse too (+0.07),
although the winning protocol does change with the noise setting — the signal exists but is not yet estimable from this
few circuits. The planned development study (fulldev19, 82 circuits) is designed to separate data-limited from
information-limited learning; the reserved confirmation should be resized for power before it is frozen.

```
python -m qemx.selection check-config configs/selection/smoke19.yaml configs/pipelines/smoke19.yaml
python -m qemx.selection pipeline configs/pipelines/smoke19.yaml --out runs/smoke19 --workers 2    # minutes
python -m qemx.selection status --out runs/smoke19
python -m qemx.selection plan configs/selection/pilot19.yaml --workers auto --tasks-csv runs/pilot19_tasks.csv
python -m qemx.selection pipeline configs/pipelines/pilot19.yaml --out runs/pilot19 --workers auto  # about an hour
python -m qemx.selection.tables19                                                                   # the 1.9.0 tables
```

## What was new in 1.8.0

1.8.0 answers the review of 1.7.0 and its follow-up (`docs/ISSUE_EVIDENCE_MATRIX.md`: every finding, the change and
its evidence), fixes the two failures of the 1.7.0 CI run (`docs/CI_FAILURE_ANALYSIS.md`), and runs a new,
pre-declared study with an expanded candidate portfolio. Every change of meaning is listed in `docs/MIGRATION.md`
(Part 000) and `docs/selection/METHODS.md` §16; the interpretation, open questions and next experiments are in
`docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md`.

* **The deployed policy is what is evaluated.** One execution/preflight contract (verified-feasible, infeasible,
  unverified; structural, design and allocation levels) is shared by labeling, the app, the command line and the
  hardware plan; the recommender never returns an unverified candidate, records the ones it skipped and falls back to
  SBS-B's first verified candidate. The ten 1.7.0 deployment regressions (degenerate CDR/NIL training designs) are
  resolved, and the confirmation now reports the full deployed policy beside its model component.
* **Bounded cost.** Inputs are checked before compilation; calibration marginals are factorized where the noise model
  permits; wide observables are compact Pauli terms; a sentinel test proves a 24-qubit request never reaches a dense
  engine.
* **Provenance and statistics.** Records bound to their run (mismatch labels included); bundles checked for
  scientific compatibility; namespaced preparation seeds (seed scheme 2) with explicit common-random-number groups;
  the variance of one deployment separated from the pooled label variance; declared tail metrics beside the primary
  log metric; strict JSON; independent shot-level references for the estimators whose labels drive decisions.
* **17 candidates.** The eight 1.6/1.7 protocols plus TREX readout, Pauli-twirled and dynamical-decoupling variants,
  twirled ZNE and PEC, a bounded exponential ZNE, feature regression, symmetry verification and ZNE∘SV — each with a
  capability record (`docs/selection/CANDIDATES.md`). Virtual distillation and M3 are documented as not emulable as
  separate candidates here.
* **Study and outputs.** A development study on 44 new logical circuits and a once-only confirmation on 29 more; a
  crossed circuit × preparation-root audit; controlled regime and noise-composition studies; four new supplementary
  figures; thesis tables as CSV, LaTeX and PDF.
* **Windows and CI.** A Windows first-run guide (`docs/WINDOWS_FIRST_RUN.md`); the CI separates file-name encoding
  from text encoding and reruns the numerically sensitive tests without AVX-512.

**1.8.0 study in one paragraph** (emulation only, `docs/selection/RESULTS.md` Part I): on 29 new logical circuits,
evaluated once, the full deployed policy of the frozen recommender (boosted trees on deployable and simulator-only
features, with its preflight, out-of-distribution checks and fallback) improves on choosing the best protocol per budget
by −0.14 [−0.23, −0.08] decades of geometric-mean MSE, and its development out-of-fold difference is resolved too, so
the rule declared before the confirmation was evaluated is met for this simulated task distribution — with the
optimism-corrected development difference not resolved, a primary estimand fixed after the development results were
seen, and a recommender that needs exact simulation (about 0.65 s per circuit at 3–6 qubits). The default bundle without simulation reaches −0.08
[−0.18, −0.002], not robustly resolved. On identical development labels the larger portfolio gives a resolved gain
(−0.07 [−0.13, −0.01] for SBS-B over 17 instead of 8 candidates) while learning on deployable features does not. The
confirmation and full-policy numbers were reproduced by independent code.
The run is shipped in `reports/selection_v180/`; its per-trial arrays are in two separate trial-data archives
(`reports/selection_v180/README.md`).

```
python -m qemx.selection recommend --experiment examples/experiments/qft_known_answer.yaml --budget 100000 --device fake_fez
python -m qemx.selection recommend --experiment examples/experiments/qft_known_answer.yaml --budget 100000 --device fake_fez \
    --bundle qemx/selection/bundles/qemx-rec-1.8.0 --allow-simulation          # the frozen (confirmed) configuration
python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml
python -m qemx.selection.thesis_tables --run reports/selection_v180 --out reports/selection_v180/thesis_tables
bash scripts/reproduce_v18.sh                                                                              # the whole 1.8.0 study
```

## What was new in 1.7.0

1.7.0 answers an independent review of 1.6.0 (`docs/REVIEW_RESPONSE_1.7.md`): it repairs the foundations of the
protocol-selection study, relabels its pilot with the corrected sampling, extends the research with focused
experiments, and connects the new recommender to the Playground. Every change of meaning is listed in
`docs/MIGRATION.md` (Part 00) and `docs/selection/METHODS.md` §14.

* **Statistics.** CDR and NIL labels sample fresh, independent training designs per label half, with design-level
  standard errors; abstention is evaluated by nested cross-validation; independent circuits are counted apart from
  variants, designs and trials, with a study-wide ancestry registry and a new once-only confirmation set; one
  vocabulary for what a number guarantees (empirical, CV-residual, split-conformal, finite-sample bound, post-hoc).
* **Execution.** Supervised child processes with parent-enforced limits on every platform, failure records that never
  crash collection, run directories bound to their inputs, enforced configuration schemas, UTF-8 text I/O
  everywhere, Python 3.10 fixes, CI jobs for Windows and a legacy locale.
* **Recommender in the app.** *Recommend for my circuit* (QEM Performance page) runs the 1.7 model bundle on the
  Playground's experiment through the same task representation as `python -m qemx.selection recommend`, computing
  only the requested features (no simulation unless allowed) and reporting out-of-distribution and unsupported
  cases.
* **Playground.** The Output panel separates input state, output state before measurement, measured distribution and
  observable; template identities are shown as predictions only when their assumptions hold.
* **Research.** The corrected pilot, a noise-composition study, equal-budget mismatch reactions with held-out
  discrepancy types, a learning-design study, a physics-informed risk model and the confirmation; revised figures with
  a methods box each; a hardware-validation plan under one shared 600 s ceiling (prepared, never submitted).

**Corrected pilot in one line** (emulation only): choosing the protocol per budget remains most of the gain (−0.14 decades against the single best protocol); the frozen learned recommender is −0.04 [−0.11, 0.03] decades against that budget-only rule on held-out development circuits and −0.09 [−0.16, −0.02] on 25 new confirmation circuits — favourable on both, resolved only on the confirmation, so by the declared rule the claim is not yet established (`docs/selection/RESULTS.md` Part II). The run is shipped in `reports/selection_v170/`; its per-trial arrays are in two separate trial-data archives (`reports/selection_v170/README.md`).

```
python -m qemx.selection plan      configs/selection/full17.yaml --measured reports/selection_v170/dev   # size, cost, caps
python -m qemx.selection run       configs/selection/smoke.yaml --out runs/smoke                          # minutes
python -m qemx.selection recommend --experiment examples/experiments/qft_known_answer.yaml --budget 100000 --device fake_fez
python -m qemx.selection figures   configs/selection/pilot17.yaml --out reports/selection_v170            # no simulation
bash scripts/reproduce_pilot_17.sh runs/pilot17 2                                                         # the whole pilot
```

## What was new in 1.6.0

1.6.0 adds a research framework for **budget-aware selection of complete, executable QEM protocols** and runs its
pilot study on emulated fake devices (`docs/selection/`; results in `docs/selection/RESULTS.md`). Nothing that 1.5.1
computes changes its value or meaning (`docs/MIGRATION.md`, Part 0; every change in `docs/RELEASE_NOTES.md`).

* **Tasks and candidates.** A task is 𝒯 = (ρ_in, C, O, D_t, 𝓘, B) with versioned identity and ancestry; the
  candidates are eight complete protocols (RAW, readout correction, three ZNE variants, PEC, CDR, NIL) whose integer
  ledgers spend exactly the budget B — calibration, training, execution — allocated from the calibration snapshot.
* **Labels you can audit.** Trial-level emulation (validated against shot-by-shot execution), two independent label
  halves (choose on one, score on the other), every trial stored, resumable labeling with per-task limits and a
  provenance record per session.
* **Policies and evaluation.** RAW, the single best protocol, a budget-dependent fixed protocol, a size-and-budget
  tree, risk models (ridge, elastic net, boosted trees, a small MLP) on five feature sets, abstention, and the
  per-task oracle; held-out base circuits, out-of-fold pooled with base-circuit bootstrap intervals, a bias-corrected
  estimate for the selected configuration, five generalization splits and a once-only final test.
* **Regret, reliability and mismatch.** Regret against an independently identified reference (tails, tolerance
  success, degradation against RAW), selective recommendation with a cross-fitted threshold, interval calibration,
  Hoeffding and empirical-Bernstein certificates; hidden coherent, drift, readout, crosstalk and temporal
  perturbations with an echo diagnostic whose shots are charged.
* **Controlled regime study.** Three circuits on a fake device whose error rates are scaled, mapped over budget × noise
  strength (reference preference, recommendation, regret, support), plus observable and layout blocks.
* **Figures and tables** (six main figures to a written panel specification, eight supplementary figures, three
  tables and a supplementary one) regenerate from the saved analysis tables with a versioned plotting configuration;
  the QEM Performance page has a *Protocol selection (1.6)* tab whose task drill-down shows the circuit, protocol
  settings, shot ledger, trial distribution and provenance.

**Pilot result in one line** (emulation only): choosing the protocol per budget beats the single best protocol by a
factor 1.5 in geometric-mean MSE; a learned per-task choice adds a further factor 1.14 (held-out circuits) to 1.28
(final test), which the pilot cannot distinguish from no gain, and its choices do not follow the noise strength that
changes the best protocol within a circuit — see `docs/selection/RESULTS.md`.

```
python -m qemx.selection plan    configs/selection/full.yaml --measured reports/selection_pilot/dev   # size and cost
python -m qemx.selection run     configs/selection/smoke.yaml --out runs/smoke                        # minutes
python -m qemx.selection figures configs/selection/pilot.yaml --out reports/selection_pilot           # no simulation
python -m qemx.selection report  configs/selection/pilot.yaml --out reports/selection_pilot           # no simulation
```

## What was new in 1.5.1

1.5.1 repaired the defects found by an independent review of 1.5.0 and made the research workflow end to end
reproducible (what changed in meaning: `docs/MIGRATION.md`, Part A).

* **Statistics.** Estimators that are nonlinear in their target data (exponential ZNE, symmetry verification) report
  Monte-Carlo moments at the evaluated budget, with a declared failure policy; the exact-probability fit is kept as
  a reference. A single hardware run reports a *signed error* and an *absolute error*, not a bias or an RMSE.
* **One context for study and app.** The live recommendation uses exactly the protocols (readout correction,
  calibration, training) and the device noise context the study used, and abstains before predicting when they
  differ. Features are schema `qemx.features/2`: cheap structural and observable features never depend on exact
  state simulation, and exact entropies are computed from the smaller side of each cut.
* **Spending you can trust.** The campaign ledger is transactional across sessions and processes, a submission whose
  outcome is unknown stays committed until resolved, every real-provider path requires the ledger, and per-job
  provider limits fit inside the admitted allowance.
* **Budgets and protocols.** Hardware plans spend exactly the declared shot budget; PEC keeps its configured shots
  per sampled circuit (a hardware cap is an explicit, recorded and simulated effective protocol); postselection
  feasibility follows one rule everywhere.
* **Study analysis.** Improvement shares carry their denominators; one explore/confirm split per base circuit across
  the whole study; recommender evaluation with group-cross-validated intervals, a full policy with fallback,
  harmful-choice rates, regret tails and paired comparisons; the Monte-Carlo error of the estimated RMSEs.
* **Reproducibility.** Results and records carry digests of the source tree, the lock file and the installed
  packages. The pilot's 75 primary records are bundled (`data/pilot_v2/`) and regenerate the pilot report without
  any simulation.

## Research workflow

**Main study: finite-budget QEM protocol selection** (simulation; `docs/USER_GUIDE.md`).

```
# reproduce the bundled pilot report from its primary records (no simulation)
python scripts/reproduce_pilot_report.py --out pilot_report --compare reports/pilot

# run a study and report it (the pilot, or the scalable design direction_A_v2_study.yaml)
python -m qemx.phase.run run manifests/direction_A_v2_pilot.yaml --out data/dirA_v2_pilot --workers 2
python -m qemx.phase.run collect data/dirA_v2_pilot
python -m qemx.study report data/dirA_v2_pilot --manifest manifests/direction_A_v2_pilot.yaml --out reports/my_pilot
```

The pilot's baseline recommender does **not** beat the best fixed method on held-out base circuits
(`reports/pilot/report.md`, Figure 6). A frozen final holdout for a future confirmation is defined in
`manifests/direction_A_v2_final_holdout.yaml` with its rules in `docs/FINAL_HOLDOUT.md`.

**Supporting study: does a discrepancy diagnostic make the decision safer?** (`reports/diagnostic_policy/`,
EMULATION): `python scripts/diagnostic_policy_study.py --out reports/diagnostic_policy`. The discrepancy rehearsal of
the divergence suite: `python scripts/rehearse_divergence.py manifests/direction_A_divergence_v3_qpu600.yaml --out
data/reh_v3`.

**Hardware within one shared 600-second campaign** (`docs/hardware_preflight.md`): a transfer test of the main
study's simulated ranking on two circuits with repeated, paired blocks, planned and emulated offline:

```
python scripts/hardware_transfer.py plan     --out reports/hardware_transfer/plan
python scripts/hardware_transfer.py rehearse --out reports/hardware_transfer/rehearsal_emulation --blocks 24
```

Nothing is submitted without a campaign ledger (`python -m qemx.campaign init …`) and an explicit `run-ibm`.

## Research backend: equal-budget phase diagrams (`qemx.phase`)

A Streamlit-free pipeline: for every QEM method on every grid coordinate (circuit family × n × noise scale ×
engine), it measures the primitives of the equal-budget ledger B = N_t + T·s_tr + c·s,
RMSE = √(b² + v/N_t + w/s + u), with budget-dependent Monte-Carlo moments for target-nonlinear estimators. It also
computes λ_LC, Monte-Carlo-validates the ledger and writes versioned records (schema `qemx.phase.cell/4`). The
pre-registered v1 design is in `docs/research/direction_A_v1_design.md`.

```
python -m qemx.phase.run plan       manifests/direction_A_v1.yaml
python -m qemx.phase.run run        manifests/direction_A_v1.yaml --out data/dirA_v1 --workers 8
python -m qemx.phase.run collect    data/dirA_v1
python scripts/dirA_checks.py       data/dirA_v1                  # acceptance checks → data/dirA_v1/checks
python scripts/dirA_figures.py      data/dirA_v1 --out figs        # phase maps, family maps, RMSE and B*(ε)
```

Further studies with their own manifests and analysis scripts:

* **Neighbor-informed learning** — `manifests/direction_A_nil_v1.yaml`, analyzed with
  `python scripts/dirA_nil.py data/dirA_nil_v1 --out figs_nil`.
* **Global against local folding for ZNE** — `manifests/direction_A_zne_folding_v1.yaml`, analyzed with
  `python scripts/dirA_zne_folding.py data/dirA_zne_folding_v1 --out figs_zne_folding` (exact crossover budgets,
  fractional ZNE with fold subsets and draws).
* **Hardware against fake-device divergence** — `manifests/direction_A_divergence_v*.yaml` with
  `scripts/dirA_divergence.py` (plan, `run-ibm`, `collect-ibm`, local dry runs, emulation with injected noise,
  analysis and the diagnostic matrix). Research plan: `docs/research/divergence_research_plan.md`.

An output directory is bound to its manifest (SHA-256); `collect` uses only that manifest's cells, and a run
directory refuses to mix record schemas. Fake devices accept modifiers: `fake_fez@s=2,ro=0,eng=exact` scales every
gate error and relaxation rate by 2, switches readout errors off, and runs the exact superoperator engine.

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
| **QEM Performance & Selection** | The protocol-selection studies (1.8.0 first; policy tables, task drill-down, figures with their notes), *Recommend for my circuit* (the shipped model bundle on the Playground's experiment: choice, fallback and why, the preflight status of every candidate, out-of-distribution reasons, feature costs), the drill-down analyses and the feature dictionary |
| **Noise Models & Discrepancy** | Where a specified noise model fails to predict circuit behavior: stored divergence and rehearsal results with matched protocols and calibrated statistics, and calculators on numbers you type (nothing executed) |

### Playground

| Tab | What you can do |
|---|---|
| Build | Pick a template by family (below). Drag gates and noise channels from the palette onto the wires (drop on the bin to delete, drag the small handle of a two-qubit gate to retarget it, double-click to edit an angle, Undo/Redo), or edit as a table or as text (`h 0; cx 0 1; depol(0.05) 0 1`). Set the input in one place, **Inputs and measurement** — per qubit, for the whole register, or through a template's input settings (QFT input index and preparation, the Ising QPE's eigenstate, a code's logical θ), prepared once before the circuit — and read the **Output** panel: the ideal output derived from input, circuit, bases and bit order, never edited. Choose measurement bases, the observable — including any **Hamiltonian** `0.5*Z0*Z1 + 0.2*X0 - 0.1*Y1`, measured in qubit-wise commuting settings — and the compilation (abstract gates, or an ISA with CX/CZ on all-to-all, line, ring, **2D grid** or **heavy-hex** connectivity; or a fake device's own target). The **Show the compiled (physical) circuit** switch aligns every logical gate above its native gates (H → RZ·√X·RZ) with routing SWAPs framed |
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
*circuits* counts what is actually sent: one per ZNE scale, ⌊N_t/s⌋ sampled PEC circuits (one shot each by
default), one random circuit per shot with twirling, 8 measurement masks per circuit with TREX, the training and the
calibration circuits (`qemx/budget.py`). Integer shot allocations add up to B exactly; when a protocol cannot spend
all of B (PEC with s > 1 shots per sampled circuit), the plan records the effective budget and why. Equal *shots* are
not equal *QPU time*: the hardware planner estimates QPU seconds separately.

## Running on IBM Quantum

The *Run on IBM Quantum* page rehearses the job locally on a fake device (SamplerV2 in local testing mode) and
then submits it to real hardware through `qiskit-ibm-runtime` with the channel `ibm_quantum_platform`, only through
the campaign ledger named by `QEMX_CAMPAIGN_LEDGER` (job mode by default). Each contender's budget is allocated
exactly as in simulation (ZNE's optimal split, s* calibration shots, PEC's sampled circuits at its configured shots
per circuit — a cap on sampled circuits is recorded as an effective protocol — and training shots), and raw, ZNE,
PEC, CDR/NIL, symmetry verification, readout mitigation (including TREX) and twirling/DD options are replayed on the
measured counts with exactly the estimator code used in simulation. A replay of one hardware run reports a signed
error and an absolute error against the ideal value. The results page and the ZIP are the Playground's.

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
  A kernel out-of-memory kill or a long native call cannot be caught this way; see `SECURITY.md` for hosting.
* **Hardware spending** — every real-provider submission goes through the campaign ledger (`SECURITY.md`,
  `docs/hardware_preflight.md`).

## License and citation

This release does not ship a license file: choosing one is the owner's decision (`docs/OWNER_DECISIONS.md`).
Citation metadata is in `CITATION.cff`.

## Tests

```
pytest -m "not slow"   # engine, templates, noise, backends, methods, analysis, security, UI state
pytest                 # + every page through AppTest, the exported script in a terminal and in a Jupyter kernel
python scripts/validate_learning_moments.py --out validation.json     # learning methods (independent experiments)
python scripts/validate_protocol_moments.py --out validation_nl.json  # nonlinear ZNE and symmetry verification
python scripts/profile_workflows.py --root . --out profile.json        # timing and memory of representative workflows
```

The number of tests changes with every release: `pytest --collect-only -q | tail -1` prints the current count.

## Deployment

Streamlit Community Cloud or any host with Python 3.10–3.13: `requirements.txt` for Python packages and
`packages.txt` (Chromium) for high-resolution PNG/SVG figure export. `.streamlit/config.toml` hides stack traces,
enables XSRF protection and limits uploads to 20 MB. On a Linux host, start the service with the `GLIBC_TUNABLES`
setting of the quick start (verified on Ubuntu 24.04; not verified on Streamlit Community Cloud, whose start-up
environment this project cannot set).

## Layout

```
app.py                      landing page
pages/                      Playground, Cost vs precision, Bias–variance, Model mismatch, Head-to-head,
                            Theory & sources, Run on IBM Quantum, Batch run, Learn, QEM Performance & Selection,
                            Noise Models & Discrepancy
qemx/                       engine (no Streamlit): circuit, templates, observables, hamiltonian, noise, compile,
                            stages, layout_search, device, backends/, transforms, methods/, recipes, analysis,
                            insight, variational, batch, sweep, learn, timeline, hardware, budget, report, export,
                            bundle, resources, explain, viz/, phase/ (Direction A), experiment (experiment
                            documents), textio (UTF-8 and strict JSON)
qemx/selection/             the protocol-selection study and the deployed recommender (no Streamlit); model bundles
                            in qemx/selection/bundles/
ui/                         Streamlit glue: navigation, session state, widgets, sidebar, drag-and-drop composer,
                            results (the shared results view), selection_view and recommend_view (QEM Performance),
                            tooltips (math in help texts), figures (diagrams)
configs/selection/          study configurations (YAML; schema in docs/selection/schemas/), ancestry registry,
                            plotting configurations, the 1.9.0 figure/table specification
configs/pipelines/          study pipelines for `python -m qemx.selection pipeline` (smoke19, pilot19, fulldev19,
                            reproduce_v18)
examples/experiments/       experiment documents for `python -m qemx.experiment` and `qemx.selection recommend`
tests/                      pytest
manifests/                  study manifests (pre-registered ones are never edited; the final holdout is frozen)
data/pilot_v2/              primary records of the main-study pilot (with SHA-256 sums)
reports/                    the protocol-selection studies (selection_v180, selection_v170, selection_pilot), the
                            1.9.0 pilot, execution audit, tables and corrections (pilot_v190, execution_audit_v190,
                            tables_v190, corrections_v190), the main-study pilot report, the discrepancy rehearsal, the diagnostic-policy study, the hardware plan
docs/                       USER_GUIDE.md, MIGRATION.md, RELEASE_NOTES.md, QA_SUMMARY.md, ARCHITECTURE.md,
                            RESEARCH_FINDINGS_AND_NEXT_STEPS.md, ISSUE_EVIDENCE_MATRIX.md, CI_FAILURE_ANALYSIS.md,
                            WINDOWS_FIRST_RUN.md, RELEASE_CONSISTENCY.md, OWNER_DECISIONS.md, theory.md (the
                            handbook), hardware_preflight.md, FINAL_HOLDOUT.md, feature_dictionary.md,
                            literature_matrix.md, selection/ (the study's methods, commands, data dictionary,
                            figures, candidates, results), research/ (study designs), learn/ (Learn chapters), figures/
scripts/                    study analysis, hardware planning, validations, reproduction, profiling, CI helpers
scripts/make_figures.py     rebuilds docs/figures/*.svg from quantikz sources (pdflatex + pdftocairo)
.streamlit/config.toml      server and theme settings (headless, XSRF protection, 20 MB uploads, no stack traces)
.vscode/                    launch and task definitions for VS Code (Streamlit app, tests)
.github/workflows/ci.yml    CI: locked, latest (3.10/3.12/3.13), oldest Streamlit, legacy-locale text encoding, Windows
```
