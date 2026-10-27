# Candidate registry (qemx.selection.registry/1)

Generated from `qemx/selection/registry.py`; do not edit by hand.

Portfolios: `portfolio-1.6` (8 candidates); `portfolio-1.8` (17 candidates).

| Candidate | Family | Components | Applies to | Resources | Ledger | Failure handling | Validation |
|---|---|---|---|---|---|---|---|
| RAW | unmitigated / readout only | raw | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 1 target circuit | B = N | none needed | direct |
| RO | unmitigated / readout only | raw; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 1 target + 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient | a singular calibration draw → no readout correction in that trial (cal_fallback) | direct |
| ZNE-L | zero-noise extrapolation | zne linear 1,3 global folding; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 2 folded target circuits + 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient; N split over folds ∝ |η_j|σ_j | cal_fallback | direct |
| ZNE-R | zero-noise extrapolation | zne Richardson 1,3,5 global folding; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 3 folded target circuits + 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient; N split over folds | cal_fallback | direct |
| ZNE-E | zero-noise extrapolation | zne exponential 1,3,5, linear fallback; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 3 folded target circuits + 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient; N split over folds | exponential fit failure → linear fit of the same data (fit_fallback); cal_fallback | direct+reference |
| PEC | probabilistic error cancellation | pec inverse of median-rate depolarizing model; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | one sampled circuit per shot (instances drawn from the quasi-probability), 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient | cal_fallback; noninvertible model → method error (recorded) | direct |
| CDR | Clifford data regression / feature regression | CDR OLS, 16 near-Clifford training circuits; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 16 training circuits, 1 target, 2 calibration circuits | B = 25 % training (equal per circuit) + calibration + target | degenerate training data → readout-corrected target value (learn_fallback) | direct+reference |
| NIL | neighbor-informed learning | NIL ℓ1 ≤ 2, weight-1 Pauli neighbors at every position; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 16 training circuits × k neighbors, k target neighbors, 2 calibration circuits | B = 50 % training + calibration + target (equal split) | degenerate training data → readout-corrected target value (learn_fallback) | direct+reference |
| RO-TREX | unmitigated / readout only | raw; TREX readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots; Z-string terms | 1 target circuit with a random X layer before measurement per shot, 1 calibration circuit | B = s (calibration) + N | singular symmetric calibration → no correction (cal_fallback) | reference |
| RO-TW | unmitigated / readout only | raw; Pauli twirl of 2q gates; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 1 target circuit re-randomized per shot (N random compilations), 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient | cal_fallback | direct |
| RO-DD | unmitigated / readout only | raw; XY4 dynamical decoupling; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 1 target circuit with DD pulses, 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient | cal_fallback | direct |
| ZNE-L-TW | zero-noise extrapolation | zne linear 1,3; Pauli twirl of 2q gates; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | 2 folded circuits re-randomized per shot, 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient; N split over folds | cal_fallback | direct |
| ZNE-E-B | zero-noise extrapolation | zne exponential 1,3,5, linear fallback; tensored readout; projection onto the observable's range | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | as ZNE-E | as ZNE-E | as ZNE-E; then projection (deterministic, never increases a trial's error) | reference |
| PEC-TW | probabilistic error cancellation | pec inverse depolarizing; Pauli twirl of 2q gates; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | one sampled, twirled circuit per shot; 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient | cal_fallback | direct |
| FR | Clifford data regression / feature regression | feature regression OLS on ⟨O⟩, ⟨Z_i⟩, ⟨Z_iZ_j⟩; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots | as CDR (16 training circuits) | as CDR | as CDR | reference |
| SV | symmetry verification | symmetry verification of declared Z-parities; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots; tasks with a declared symmetry (GHZ in the Z basis from its own input) | 1 target circuit; rejected shots are spent and counted; 2 calibration circuits | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient; minimum target shots ⌈5/P_accept⌉ (needs the noisy model: preflight 'allocation') | a repeat with no accepted shot → nonresponse (declared loss μ²) | reference |
| ZNE-L∘SV | chained mitigation (outer ∘ inner) | zne linear 1,3; symmetry verification of every fold; tensored readout | diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots; tasks with a declared symmetry | 2 folded circuits, each post-selected; 2 calibration circuits shared by both folds | B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient; N split over folds; ⌈5/P_j⌉ per fold | a fold with no accepted shot → NaN → nonresponse | reference |

## Definitions (the identity bundles are checked against)

* **RAW** — Unmitigated: all B shots on the target circuit; sample mean of the observable.
* **RO** — Readout correction only: tensored inverse of per-qubit assignment matrices estimated from two calibration circuits (|0…0⟩, |1…1⟩); calibration/target split by the supplied model.
* **ZNE-L** — Zero-noise extrapolation, global folding at λ = 1, 3, linear fit, tensored readout correction; shots per scale ∝ |η_j|σ_j of the supplied model (integer optimal split).
* **ZNE-R** — ZNE, global folding at λ = 1, 3, 5, Richardson (quadratic) extrapolation, tensored readout correction; supplied-model allocation.
* **ZNE-E** — ZNE, global folding at λ = 1, 3, 5, exponential fit with a declared linear fallback when the fit fails (counted per trial); tensored readout correction; supplied-model allocation.
* **PEC** — Probabilistic error cancellation of the device's median-rate depolarizing model (from the calibration record), one shot per sampled circuit (exact signed law), tensored readout correction.
* **CDR** — Clifford data regression: 16 near-Clifford training circuits with classically computed ideal values, OLS fit, 25 % of B on training (equal per circuit); fallback decided from each trial's own training data; tensored readout correction.
* **NIL** — Neighbor-informed learning (Chen et al.): weight-1 Pauli neighbors at every position, 2-design training ensemble, ℓ1-bounded (≤ 2) regression without intercept, equal target split; 50 % of B on training; tensored readout correction.
* **RO-TREX** — Readout correction by TREX: the measurement is twirled (random X before it, classical flip after; one randomization per shot), each Z string divided by its value on one twirled |0…0⟩ calibration circuit; calibration/target split by the supplied model.
* **RO-TW** — Pauli twirling of the two-qubit gates (one random compilation per shot; gate errors become their Pauli twirl), no estimator, tensored readout correction.
* **RO-DD** — Dynamical decoupling (XY4 in idle windows; the pulses carry their gate errors), no estimator, tensored readout correction.
* **ZNE-L-TW** — ZNE-L on Pauli-twirled two-qubit gates (one random compilation per shot): global folding at λ = 1, 3, linear fit, tensored readout correction, supplied-model split.
* **ZNE-E-B** — ZNE-E (λ = 1, 3, 5, exponential fit, linear fallback) followed by a declared projection of the estimate onto the observable's eigenvalue range — a separately named bounded estimator (review finding F7); ZNE-E itself is unchanged. Labels use ZNE-E's random streams (common random numbers), so each ZNE-E-B trial is the projection of the paired ZNE-E trial.
* **PEC-TW** — PEC of the median-rate depolarizing model on Pauli-twirled two-qubit gates (twirling makes the executed gate errors Pauli channels, closer to the cancelled model); one shot per sampled circuit; tensored readout correction.
* **FR** — Feature regression (this project's CDR extension, not NIL): OLS on ⟨O⟩, every ⟨Z_i⟩ and ⟨Z_iZ_j⟩ of 16 near-Clifford training circuits, 25 % of B on training, tensored readout correction; CDR's adaptive training set and fallback.
* **SV** — Symmetry verification: post-selection on the Z-parity checks the task's construction declares (never inferred from its ideal output), rejected shots counted in B, a repeat with no accepted shot is a nonresponse; tensored readout correction. Applicable only with a declared symmetry.
* **ZNE-L∘SV** — Chained: linear ZNE (λ = 1, 3, global folding) whose every folded circuit is symmetry-verified with the task's declared checks (the inner post-selection applies to each fold; its rejected shots are paid per fold); one calibration shared by every fold; tensored readout correction. Applicable only with a declared symmetry.

## Implemented but not in the selection portfolios

| Item | Status | Reason |
|---|---|---|
| vd (virtual distillation, M = 2, 3, 4) | blocked | the implementation evaluates Tr(ρ^M O)/Tr(ρ^M) with ideal copies and an ideal derangement and samples a Gaussian with the delta-method variance; a finite-shot multi-copy protocol (M·n qubits, controlled derangement, numerator/denominator estimation and covariance, noisy copies) is not implemented, so an equal-budget comparison would be invented. Kept as a simulator-only upper bound in the app. |
| readout m3 (M3 subspace solve) | not in portfolio | supported in the app; in the trial emulator a calibration draw enters through the tensored inverse (Context._mitigator), so emulated M3 labels would equal RO's law rather than finite-shot M3 — a faithful label needs per-repeat subspace solves |
| twirl 'all' (randomized compiling of every gate) | not in portfolio | bounded grid: two-qubit twirling represents the transform; 'all' adds one-qubit frames |
| dd 'X2' (CPMG) | not in portfolio | bounded grid: XY4 represents DD |
| zne folding 'local', fractional scales, fold_draws, lam_est 'fault-rate', rule 'poly2', on_fail 'abstain' | not in portfolio | bounded grid: three global-folding rules (linear, Richardson, exponential) represent ZNE |
| pec model 'learned', construction 'compensation', scope '2q', shots_per_circuit > 1, instances | not in portfolio | 'learned' assumes the true channel is known without a characterization budget (an idealized information access); the others are settings of the represented construction |
| learning model ridge/lasso, strategy 'fixed', keep > 0 | not in portfolio | bounded grid: CDR (OLS) and FR represent the learning estimator |
| nil neighbors 'fold', positions '2q', other regression models/ensembles/target_split 'optimal' | not in portfolio | bounded grid; 'optimal' target split needs fitted coefficients before the target runs (not deployable on hardware) |
| chain outer cdr/nil, inner pec/vd; chain outer zne with inner raw/pec | not in portfolio | ZNE∘SV represents chained mitigation; cdr/nil outer multiplies the T×k training circuits by the inner cost (compute), inner vd inherits VD's blocker, zne∘raw equals ZNE |

## Inventory

| Kind | Implemented | Candidates |
|---|---|---|
| estimator | raw | RAW, RO, RO-TREX, RO-TW, RO-DD |
| estimator | zne | ZNE-L, ZNE-R, ZNE-E, ZNE-L-TW, ZNE-E-B |
| estimator | pec | PEC, PEC-TW |
| estimator | learning | CDR, FR |
| estimator | nil | NIL |
| estimator | symmetry | SV |
| estimator | vd | see UNSUPPORTED |
| estimator | chain | ZNE-L∘SV |
| readout | none | RAW |
| readout | tensored | RO, ZNE-L, ZNE-R, ZNE-E, PEC, CDR, NIL, RO-TW, RO-DD, ZNE-L-TW, ZNE-E-B, PEC-TW, FR, SV, ZNE-L∘SV |
| readout | m3 | see UNSUPPORTED |
| readout | trex | RO-TREX |
| twirl | none | RAW, RO, ZNE-L, ZNE-R, ZNE-E, PEC, CDR, NIL, RO-TREX, RO-DD, ZNE-E-B, FR, SV, ZNE-L∘SV |
| twirl | 2q | RO-TW, ZNE-L-TW, PEC-TW |
| twirl | all | see UNSUPPORTED |
| dd | none | RAW, RO, ZNE-L, ZNE-R, ZNE-E, PEC, CDR, NIL, RO-TREX, RO-TW, ZNE-L-TW, ZNE-E-B, PEC-TW, FR, SV, ZNE-L∘SV |
| dd | X2 | see UNSUPPORTED |
| dd | XY4 | RO-DD |
