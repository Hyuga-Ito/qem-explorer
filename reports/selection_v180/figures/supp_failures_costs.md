# supp_failures_costs

**Question.** How often do protocols fail or fall back, and where do the shots go?

**Caption.** Failures and costs (development labels, 88 tasks). (a) Share of tasks on which a protocol is infeasible at B (its minimum calibration or training shots do not fit, or it has nothing to train on; a declared property of the protocol at that budget) and, crosses, the share whose preparation raised an error (method error, recorded with its message). (b) Share of trials with at least one declared fallback (singular calibration, failed exponential fit, CDR or NIL training fallback) or no estimate — the union over the overlapping flags, never their sum; dotted, open: the no-estimate part. Every such trial is in the protocol's risk. Software failures of whole label tasks (error, timeout, memory) are not protocol outcomes; they are listed in label_failures.csv (none) and excluded from the dataset. (c) Ledger of each protocol at B = 10⁵ by role (integers summing to B in every allocation; shares are medians over tasks). Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Infeasibility, fallback and software failure are different events: the first is a property of the protocol at a budget, the second part of its risk, the third an execution problem of the study (0 label tasks).

**Data.** `supp_failures_costs__status.csv`, `supp_failures_costs__fallbacks.csv`, `supp_failures_costs__software_failures.csv`, `supp_failures_costs__ledger.csv`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | protocol and budget |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 88 development tasks on 44 independent base circuits; 6 fresh designs per half for CDR and NIL; r = 60 trials per half |
| Uncertainty and its scope | none (rates) |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | descriptive |
| Infeasible, failed, unresolved, missing | infeasible, method error, fallback (union of flags), no estimate and software failures (0 label tasks) reported separately |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/labels.csv`: protocol, B, feasible, *_fallback_eval, nonresponse_eval, shots_*

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
