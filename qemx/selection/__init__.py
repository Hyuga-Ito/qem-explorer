"""Budget-aware selection of complete, executable QEM protocols (QEM Explorer 1.7.0).

The study asks which complete mitigation protocol to run for a given task and total shot budget, and whether a
learned recommender beats fixed choices once every shot is charged. Modules:

=================  ==================================================================================================
``task``           task representation 𝒯 = (ρ_in, C, O, D_t, 𝓘, B), ancestry ids, information classes
``protocols``      the candidate portfolio and integer budget ledgers (allocations from the supplied model only)
``emulate``        trial-level emulation of complete protocols from cached exact distributions; direct-execution
                   validation
``labels``         per-trial labels in two independent halves, checkpoint/resume, costs
``features``       deployable / simulated / oracle features, light-cone definitions, feature sets
``dataset``        the decision dataset built from saved trials
``models``         per-protocol risk models (ridge, elastic net, gradient-boosted trees, small MLP, the structured
                   b² + v/((1 − f)B) model), grouped tuning
``policies``       RAW, single best, budget-dependent fixed, size-and-budget tree, models, abstention, references
``evaluate``       regret, gap closed, base-circuit bootstrap, generalization splits
``certify``        Hoeffding / empirical-Bernstein bounds (clipped estimator), the 2ε post-hoc check, CV-residual and
                   group split-conformal intervals
``claims``         the vocabulary of statement kinds (empirical, cv-residual, split-conformal, …)
``supervise``      supervised child processes for labeling (parent-enforced time and memory limits)
``binding``        run binding: what a run directory was made from; refused incompatible resumes; freeze checks
``ancestry``       canonical logical identity of circuits and the study-wide ancestry registry
``deploy``         the shared deployment task, cheap features, structural feasibility, model bundles, recommend
``hardware``       hardware validation plan and authorization gate (never submits by itself)
``mismatch``       hidden execution-law perturbations, echo and readout diagnostics, finite-shot nulls
``analysis``       source-data tables of every figure
``figures``        figures (PDF/SVG/PNG + CSV + caption) and tables
``design``         configurations, task grids, staged acquisition, the untouched final test
``run``            workflows (smoke, pilot, full, ablation, generalization, boundary, mismatch)
=================  ==================================================================================================

Command line: ``python -m qemx.selection --help``. Methods: ``docs/selection/METHODS.md``.
"""
SCHEMA_VERSION = "1.7.0"
