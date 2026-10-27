# Research plan of the next stage (1.9.0; corrected in 1.9.1): smoke, pilot, full development, reserved confirmation

**Thesis question.** Which attainable information enables reliable budget-aware QEM selection beyond SBS-B, and when
do missing context or unstable protocols defeat it? Fake-device emulation is primary; hardware is on hold.

**Two kinds of runs, kept apart.**

| kind | what it is | pipeline | evidence status |
|---|---|---|---|
| historical reproduction | re-running a study whose results were inspected (1.8.0: `dev18`, `conf18`, the controlled studies) into a new directory | `configs/pipelines/reproduce_v18.yaml` | reproduces published numbers; **never** new confirmation evidence |
| prospective study | a design declared before its labels: `smoke19` (checks the pipeline), `pilot19` (costs, feasibility), `fulldev19` (development), `conf19` (reserved confirmation) | `configs/pipelines/{smoke19,pilot19,fulldev19}.yaml`; `conf19` by its own command after the freeze | smoke: none; pilot: exploratory (planning); fulldev: development; conf19: confirmatory once, after everything below is frozen |

All four prospective configurations are new logical circuits, disjoint from every earlier study and from each other;
they are registered in `configs/selection/ancestry_registry.json` (smoke19 exploration, pilot19 development, fulldev19
and conf19 planned), checked by `tests/test_readiness_v190.py::test_the_1_9_studies_are_registered_and_disjoint_from_every_other_study`.

## Design (all four)

* **Candidates:** the expanded registry (17; `docs/selection/CANDIDATES.md`). VD and M3 stay excluded from equal-budget
  claims (no faithful finite-shot implementation with a resource ledger).
* **Labels:** fresh training designs, seed scheme 2 (namespaced streams), independent selection/evaluation halves;
  common-random-number group = task. Budgets 10³–3·10⁷ shots (smoke/pilot fewer). More shots never add independent
  circuits: the unit of inference is the base circuit.
* **Noise diversity in training (new):** calibration-supplied snapshot modifiers on three base devices — overall error
  scale `s` (0.5, 2), two-qubit errors `g2` (2), relaxation `t` (2), readout `ro` (2), a coherent fraction `chi`
  (0.25; fulldev/conf only, exact engine). The modifiers change what the user's calibration shows *and* the execution
  law together: the deployment model never receives a hidden execution-law parameter. (Readout *asymmetry* is not a
  supported modifier; it varies only between native snapshots.)
* **Expansion:** `expansion_policy: explicit` — every block states `include_default` and `variants_per_base`; variants
  are sampled from the whole grid (no implicit default variant). `plan --tasks-csv` exports the expanded table before
  any label.
* **Held-out analyses:** same-distribution group folds (circuits), held-out noise setting (`noise`), base device
  (`device-base`), joint family × noise (`joint`), width and budget extrapolation (fulldev). Matched-exposure
  composition comparisons are **not** part of these designs; a composition study must solve the matching per circuit or
  report the residual exposure mismatch (the 1.8.0 QAOA case was +11.8 %, beyond its ±10 % comment).
* **Models:** Ridge, ElasticNet, HGB, MLP (fulldev), StructuredRisk, each tuned in inner folds; feature sets
  size/budget, structure, light cone, deployable (1.8.0 noise descriptors included), deployable + simulator-only.
  Feature ablations are retuned per group (fulldev extras). No deeper network or GNN before the bottleneck is located
  (missing information vs. too few independent circuits vs. representation vs. tuning vs. unstable candidates).

## Expanded designs and measured costs

| study | tasks | base circuits | noise settings (tasks) | families (tasks) | widths | R/half, designs/half, budgets | labeling cost (measured basis) |
|---|---|---|---|---|---|---|---|
| smoke19 | 10 | 5 | native 5, g2=2 4, t=2 1 | ising 4, hea 4, ghz 2 | 3–4 | 12, 2, 3 | executed: 7.5 min end to end on two cores (first run), < 1 min for a verified rerun |
| pilot19 | 54 | 27 | native 18, g2=2 10, s=2 9, t=2 8, s=0.5 5, ro=2 4 | brickwork, hea, ising, qaoa 12 each; ghz 6 | 3–5 | 24, 3, 4 | executed: 0.86 CPU-h (median 39/51/67 s per task at n = 3/4/5, two shared cores) |
| fulldev19 | 246 | 82 | native 80, s=2 38, g2=2 34, t=2 34, ro=2 23, s=0.5 21, chi=0.25 16 | brickwork, hea, ising, qaoa 48 each; ghz 36; bv, grover, qft 6 each | 3–6 | 60, 6, 10 | **3.0 CPU-h** from the 1.8.0 dev18 rates (same R, budgets and designs; widths 3–6 measured) to **9.3 CPU-h** from the pilot's rates (measured on two shared cores, so contention-inflated; preparation scaled ×1.86 for twice the designs) — a *conditional* estimate, conservative for the measured kinds of tasks under these assumptions but **not a proven upper bound**: the coherent (`chi`, exact engine) tasks are unmeasured — label a few first; labels ≈ 0.04–0.06 GB |
| conf19 | 64 | 32 | native 28, s=2 9, t=2 8, other 4–5 each | hea, ising, qaoa 16 each; brickwork, ghz 8 each | 3–6 | 60, 6, 10 | 0.8–2.4 CPU-h (same bases) |

Wall time ≈ CPU-hours / workers with one numerical thread per worker (`plan` reports it for the computer it runs on;
`reports/tables_v190/s6_costs` states it for a declared hypothetical 8-core computer with 7 workers: 9.3 / 7 = 1.3 h);
analysis, model search and figures are **not** included, and labeling caps (`limits.cpu_hours`) do not limit them.

**Analysis cost (1.9.1; independent review of 1.9.0).** Measured on the pilot: analysis 71 minutes single-threaded (4
model families × 3 feature sets on the same-distribution split, 2 × 2 on three further splits, design diagnostics,
learning curve), nested evaluation 20 minutes, bundles, portfolio comparison, profile and figures ≈ 9 minutes (stage
logs in the private audit archive). 1.9.0 extrapolated "10–15 CPU-hours" for fulldev19 from its 11.4 times as many
rows; that ignores how the number of model fits grows. `plan` now prints the structural workload
(`resources.analysis_workload`: split parts from the expanded tasks, model policies per part, fits including the
inner 4-fold tuning of every grid point for each of the 17 candidates, and fits × training rows):

| | pilot19 | fulldev19 | ratio |
|---|---:|---:|---:|
| analysis rows (tasks × budgets) | 216 | 2 460 | 11.4 |
| split parts (same-distribution, noise, device-base, joint, width, budget-extrap) | 5, 17, 9, 1, –, – | 5, 35, 15, 43, 1, 5 | 3.2 |
| model policies per same-distribution part | 13 | 26 | 2 |
| model fits, main splits | 83 980 | 334 611 | 4.0 |
| fits × training rows | 8.4 × 10⁶ | 4.3 × 10⁸ | **51** |
| extras declared | 2 | 10 (8 that the pilot did not run, incl. ablation and nested selection) | — |

These are counts, **not a time**: an MLP or the structured model costs far more per fit than a ridge regression, and
the extras are not counted. If the cost per fit grows about linearly with the training rows, the main splits alone
could take tens of times the pilot's analysis, not 10–15 CPU-hours — this is a hypothesis to measure, not an
estimate. Before committing a computer to the whole analysis, label fulldev19, then time the same-distribution split
alone (`analyze … --splits same-distribution --extras=`; 18.6 % of the main splits' fits × rows) and extrapolate with
the shares that `plan` prints; run the extras after that measurement. Run `analyze` on a computer that can stay on (it
checkpoints after every split and extra).

```powershell
python -m qemx.selection plan configs\selection\fulldev19.yaml --measured runs\pilot19\dev --workers auto --tasks-csv runs\fulldev19_tasks.csv
python -m qemx.selection label configs\selection\fulldev19.yaml --out runs\fulldev19 --workers auto --limit 6    # measure here
python -m qemx.selection plan configs\selection\fulldev19.yaml --out runs\fulldev19 --workers auto              # rates of this run
python -m qemx.selection pipeline configs\pipelines\fulldev19.yaml --out runs\fulldev19 --workers auto --until label
python -m qemx.selection analyze configs\selection\fulldev19.yaml --out runs\fulldev19 --splits same-distribution --extras=   # time this
python -m qemx.selection pipeline configs\pipelines\fulldev19.yaml --out runs\fulldev19 --workers auto
```

(The last command re-evaluates the same-distribution split together with the others; the deployed and nested
evaluations wait for the freeze — `analyze` records them as deferred — and are computed by the `nested` stage.)


## Controlled analyses within fulldev19 (data-limited or information-limited?)

The pilot does **not** show that too few circuits, rather than missing information in the features, is why learned
selection fails: fulldev19 changes the circuits (27 → 82), widths (3–5 → 3–6), families (5 → 8), budgets (4 → 10),
trials (24 → 60 per half), designs (3 → 6), noise settings (6 → 7) and the model grid at once, so "fulldev19 beats the
pilot" would not identify a cause. Within fulldev19, with the test population held fixed:

1. **Learning curve at fixed test folds** (`learning_curve` extra, declared): the same outer same-distribution folds;
   the training base circuits subsampled to 25/50/75/100 % (3 repeats), R, designs, budgets, feature set and model
   fixed. A curve still falling at 100 % supports *data-limited*; a flat curve at a gap to SBS-B supports
   *information- or representation-limited*.
2. **Feature ablation at fixed folds** (`ablation` extra): feature groups removed one at a time and retuned, on the
   same folds and labels; a group whose removal costs nothing carries no usable signal at this size.
3. **Model capacity at fixed features**: the same-distribution comparison already crosses the five model families with
   the five feature sets on the same folds (`summary_same-distribution.csv`); compare within a feature set.
4. **Per-noise-setting and joint shifts** read per part with the number of test circuits (below).
5. **Not implemented (would be new code, not part of 1.9.1):** label-precision downsampling (fewer trials per half on
   the same circuits) and design-count downsampling; run them only as a declared follow-up.

Declare before looking at fulldev19's results which pattern of (1)–(3) would count as data-limited and which as
information-limited, and keep conf19 closed until then.

## Reading per-part results

* A held-out-noise or joint part with **one** independent test circuit has a degenerate bootstrap over circuits: its
  interval says nothing about the population. Since 1.9.1 every summary row has an `inference` column ("descriptive:
  one base circuit"; "few base circuits (n) … descriptive" below five). fulldev19 has such parts (min test groups 1 in
  `noise` and `joint`); report them as counts and descriptive contrasts, never as a resolved region. The pooled
  out-of-fold summary (`pooled_<split>.csv`, one bootstrap over all test circuits of the split) is the inferential
  quantity.
* Per-budget intervals are pointwise; an all-budget statement needs simultaneous coverage, which is not computed.
* **Simulator-assisted and simulation-free recommendations stay distinct**: the bundle with simulator-only features
  (`fulldev19-sim`, like 1.8.0's "primary"/frozen bundle) and the deployable bundle without simulation
  (`fulldev19-deployable`, like 1.8.0's "secondary"/default bundle) are separate policies with separate numbers; a
  result of one is never reported as the other's.

## conf19: estimand, population and size (a decision before labeling)

conf19 as drafted is **32 base circuits (64 tasks) of five families** — hea 8, qaoa 8, ising 8, brickwork 4, ghz 4 —
at widths 3–6 with fulldev19's noise settings. It has none of fulldev19's QFT, BV and Grover circuits (6 of fulldev19's
82 base circuits), so it cannot confirm anything about them; its header said "same task distribution as fulldev19",
which 1.9.1 corrects (comment only: the configuration and its digest are unchanged). conf19 stays unexecuted.

**Estimand (to state in the freeze).** Δ = the mean over the population's base circuits c of the mean over c's task
variants and the 10 budgets of log10(R_policy + 10⁻⁶) − log10(R_SBS-B + 10⁻⁶), with R the fresh-design average risk on
the evaluation half, the frozen policy's choice (with its τ and fallback) against SBS-B refitted on the development
labels; unit: base circuit; 95 % percentile bootstrap over base circuits (variants, budgets and design roots resampled
with their circuit); decision: confirmed if the interval's upper bound is below 0 (and, if a meaningful effect δ is
stated, report whether Δ ≤ −δ is supported — a separate, stronger claim). The population is part of the estimand:
which families, widths, noise settings and in which proportions.

**Power (normal approximation, two-sided 5 %; planning only).** Between-circuit SD of the per-circuit Δ on the 1.8.0
confirmation (29 circuits, recomputed from the replay rows): 0.210 (frozen bundle) and 0.247 (deployable bundle). The
1.9 policy and population may have another SD; re-estimate it from fulldev19's out-of-fold per-circuit contrasts
(development data, not confirmation) before fixing N.

| circuits N | power at δ = 0.05 | δ = 0.08 | δ = 0.10 |
|---:|---|---|---|
| 32 (as drafted) | 27 % / 21 % | **58 % / 45 %** | 77 % / 63 % |
| 48 | 38 % / 29 % | 75 % / 61 % | 91 % / 80 % |
| 64 | 48 % / 37 % | 86 % / 74 % | 97 % / 90 % |
| 75 | 54 % / 42 % | 91 % / 80 % | 98 % / 94 % |
| 96 | 64 % / 51 % | 96 % / 89 % | > 99 % / 98 % |

(SD 0.210 / 0.247.) 80 % power needs 35 / 48 circuits at δ = 0.10, 55 / 75 at 0.08, 140 / 191 at 0.05.

**Options (choose one, register the new logical circuits before any label, then freeze).**

| option | population (estimand) | size | labeling cost (fulldev19 rates scaled by tasks; chi unmeasured) | what it can confirm |
|---|---|---|---|---|
| A. five-family target, resized | hea, qaoa, ising, brickwork, ghz at widths 3–6, conf19's proportions and noise settings | 48 circuits (δ = 0.10 at ≥ 80 % for both SDs) or 75 (δ = 0.08) | ≈ 1.2–3.6 CPU-h (96 tasks) / 1.9–5.6 CPU-h (150 tasks) | the frozen policy on that five-family population; nothing about QFT/BV/Grover |
| B. fulldev19's population | all eight families in fulldev19's proportions (≈ 7 % structured algorithms) | 75–96 circuits, of which ≈ 6 structured | ≈ 2–7 CPU-h | the policy on fulldev19's mixture; per-family statements for QFT/BV/Grover remain descriptive (≈ 2 circuits each) |
| C. family-balanced | equal weight per family (the estimand averages family means) | ≥ 8 per family × 8 = 64–96 circuits | ≈ 2–7 CPU-h; needs new structured-algorithm instances (secrets, marked states, QFT inputs) | a family-averaged effect; power depends on the between-family spread as well |

A keeps the drafted generator and is the smallest change; B matches what fulldev19 trains on; C answers a different
question (does it help the typical family, not the typical circuit). Whichever is chosen, the freeze records the
population, the estimand, δ, N and the decision rule before conf19 is labeled; the conf18 and earlier confirmations
remain inspected historical data.

## What must be frozen before conf19 is labeled

1. the policy configuration and its deployment threshold τ (`freeze`, `freeze --complete` on fulldev19);
2. the primary estimand and metric (the exact Δ of `configs/selection/figure_spec19.yaml`), the comparison (SBS-B,
   and RAW), the aggregation unit (base circuit) and the resampling hierarchy (circuits, with variants, budgets and
   design roots resampled together);
3. the eligibility/abstention rules and any paid diagnostic with its budget (charged to B);
4. the split design and stopping rules (no peeking; a technically interrupted labeling resumes, it is not restarted
   with other settings);
5. the population and the number of independent circuits, from the between-circuit variability and a stated
   meaningful effect (section "conf19: estimand, population and size" above: options A–C and the power table). **conf19
   as drafted (32 circuits, five families) has only ≈ 45–58 % power at Δ = −0.08**: state the meaningful effect and the
   population, then resize conf19 (new logical circuits, registered before any label) —
   `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §I.6, experiment 2;
6. the figure/table contracts (`configs/selection/figure_spec19.yaml`).

Then: `python -m qemx.selection confirm configs/selection/conf19.yaml --out runs/fulldev19 --dev-config
configs/selection/fulldev19.yaml --train dev` — once. Earlier confirmations (conf18, confirm17) are inspected data:
they can be reproduced and explored, not used to confirm a new policy.

## Execution-risk target

Every replay so far scores the returned candidate's fresh-design average risk. Before conf19, the joint audit
(`scripts/execution_audit.py`) should be repeated on fulldev19 circuits with more learning-candidate rows: the 1.9.0
audit on eight inspected circuits found no material difference (−0.024 [−0.047, +0.000] for the primary policy), but
only 10 of its 480 rows involved a learning candidate.
