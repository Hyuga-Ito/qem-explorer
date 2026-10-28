# fig03_feature_mechanisms

**Question.** Which information changes the decision?

**Caption.** Which information changes the decision? (A) Ablations of the deployed configuration's feature set, each variant retrained and retuned with the same inner group-4-fold protocol and scored on held-out base circuits (group 5-fold; 44 independent base circuits): feature groups (circles) and semantic removals (diamonds: every quantity derived from one source — the light cone with all its noise-weighted sums; every simulated quantity; removing every calibration-derived noise sum is the noise group itself and is shown once). Paired difference in mean log10 regret to the full model with 95 % bootstrap intervals over base circuits; predictive, not causal, importance. (B) One circuit, input, device and budget with only the observable changing (support weight w, light-cone qubits LC and light-cone fault weight λ under each label): candidate MSEs over both label halves with ±1.96 SE (SE over trials; over learning designs for the learning candidates under fresh designs); ring: the frozen model's own choice (before abstention); triangle: the measured best. (C) The same task with every calibrated error rate of the device (snapshot and truth alike) multiplied by s at a fixed budget; marks as in (B). (D) The noise-composition study: 3 circuits (one task per cell; n = 4 on the calibration-best chain of fake_fez) under one noise component changed at a time — one- or two-qubit gate errors, relaxation (T1, T2 divided), readout, or a coherent fraction of every gate error (exact engine) — the snapshot and the execution law changing together. Each cell names the measured best protocol (both halves), the total fault weight λ of the setting and the frozen model's choice (✗: differs); hatching: the runner-up within 1.96 SE of the difference (pointwise, not an equivalence test). Panels B–D are single-task controlled comparisons: they show whether the ranking moves with the controlled variable for these tasks, not how often it does in general. Simulated observations (emulated fake-device trials); no hardware data.

**Interpretation.** Semantic ablations (Δ mean log10 regret): light cone + all its derived sums +0.036 [+0.003, +0.073]. Measured best per observable (B): z_mid: RAW, zz_mid: PEC, default: RAW, z_all: PEC-TW. Measured best per error scale (C): s=0.5: PEC-TW, s=1: RAW, s=1.5: PEC-TW, s=2: PEC-TW, s=3: PEC-TW. Noise composition (D): the measured best differs from the circuit's baseline in 10 of 15 changed cells (4 of them resolved); the model's choice differs from the measured best in 12 of 18 cells. A change of the measured best between settings whose top two are unresolved is not evidence of a regime change.

**Data.** `fig03_feature_mechanisms__ablation.csv`, `fig03_feature_mechanisms__observable.csv`, `fig03_feature_mechanisms__noise.csv`, `fig03_feature_mechanisms__composition.csv`

**Configuration.** `{"noise_composition_panel": {"budget": 100000, "families": ["ising", "hea", "qaoa"]}, "noise_panel": {"budget": 100000, "family": "ising", "layout": "best", "observable": "default"}, "observable_panel": {"budget": 100000, "device": "fake_fez", "family": "ising", "layout": "best"}}`

**Methods box.**

| | |
|---|---|
| Varied / held fixed | (A) the information available to the model; (B) the observable; (C) the overall noise scale; (D) one noise component at a time — everything else held fixed within a panel |
| Units, budget, estimator | MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment (calibration, training, diagnostics and target shots all charged) |
| Independent counts | 88 development tasks on 44 independent base circuits; 6 fresh designs per half for CDR and NIL; r = 60 trials per half; panels B–D: single tasks per point (18 noise-composition tasks on 3 base circuits) |
| Uncertainty and its scope | (A) 95 % bootstrap over base circuits of the paired difference; (B, C) ±1.96 SE over trials (over designs for the learning candidates (CDR, NIL; FR in 1.8.0)); (D) pointwise resolution of the top two |
| Observations | simulated (emulated fake-device trials); no hardware observations |
| Status | exploratory; (B–D) controlled single-task comparisons of hypotheses declared before labeling |
| Infeasible, failed, unresolved, missing | infeasible candidates omitted per cell; unresolved top-two hatched in (D) |
| Log floor and clipping | log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping only inside the bounded-loss certificates, so marked) |

**Inputs.**

- `analysis/ablation.csv`: variant, diff_vs_all, diff_lo, diff_hi, tuned
- `regimes/analysis/regime_protocols.csv`: family, device, observable, layout, B, protocol, mse_pooled, se_pooled, chosen
- `regimes/analysis/regime_features.csv`: family, device, observable, layout, obs_weight, lc_qubits, lam_lc

**Command.** `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml`

**Plotting configuration.** `configs/selection/figures18.yaml` (version 3, sha256 49c6b66b8a7a83be…)
