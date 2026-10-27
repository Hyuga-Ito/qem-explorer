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
