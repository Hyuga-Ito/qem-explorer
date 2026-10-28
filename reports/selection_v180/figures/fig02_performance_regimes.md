# fig02_performance_regimes

**Question.** Where should each protocol be used along budget and noise strength, and does the recommender follow it?

**Caption.** Empirical performance regime maps of the controlled regime study (decision D23): three preregistered slices (4-qubit Ising, HEA and QAOA circuits; default input and observable; calibration-best layout on fake_fez). Within a slice only the device error scale s changes (every reported gate error and relaxation rate of the snapshot × s, applied to the snapshot the protocols use and to the execution law alike). Cells are the evaluated (budget, s) points; nothing is interpolated. Reference preference: the candidate with the lowest MSE over both label halves (120 trials); hatched where the runner-up is within 1.96 standard errors of the difference (unresolved — overlap is not equivalence, and no equivalence procedure is implemented). Recommendation: the frozen recommender's choice (trained on the development set without these circuits) as deployed; a ring marks abstention to SBS-B under the threshold chosen on development rows, and the cell then shows SBS-B's choice (the model's own choice is in the source data). Regret: log10 regret ratio of the deployed policy (with abstention) on a shared, zero-centred scale; the 27 negative estimates (the reference, identified on the selection half, scored worse on the evaluation half: estimation noise, not negative true regret) are drawn as they are. A triangle marks cells where a candidate is infeasible (counts in the support map); no cell is short of data (a cross would mark one). The maps are descriptive and pointwise: the 'unresolved' rule compares the top two candidates of one cell with an approximate standard error (design-level for learning protocols under fresh designs); it is not a simultaneous statement over candidates or cells. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Observation: 44 of 150 cells are resolved; among them the model's own choice equals the reference in 34 % and the deployed choice (after abstention) in 34 %; it abstained in 0 of 150 cells; distinct reference protocols per slice: ising 4, hea 6, qaoa 7; largest deployed log10 regret 1.45. These are empirical regime maps, not phase diagrams: boundaries are known only to the grid's resolution and were not refined; an unresolved cell is not a tie.

**Data.** `fig02_performance_regimes__cells.csv`

**Configuration.** `{"min_trials": 30, "regret_vmax": 0.9617157252391807, "slices": ["ising", "hea", "qaoa"]}`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | budget and overall noise scale s along controlled slices (one circuit each); held fixed: circuit, input, layout and observable within a slice |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 21 regime tasks on 3 base circuits; r = 60 trials per half |
| Uncertainty and its scope | unresolved cells: runner-up within 1.96 approximate SE of the difference (pointwise, not simultaneous over cells or candidates; design-level SE for the learning candidates (CDR, NIL; FR in 1.8.0)) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory, descriptive empirical map — not a phase diagram and not a simultaneous guarantee |
| Infeasible, failed, unresolved, missing | hatched/white: unresolved or fewer trials than the plotting minimum; infeasible candidates shown in the support panel; signed regret keeps negative empirical values |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `regimes/analysis/regime_cells.csv`: family, noise_scale, B, observable, layout, reference, unresolved, model_choice, abstained, policy_choice, log_regret_policy, n_feasible, trials_per_candidate

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
