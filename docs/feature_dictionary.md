# Feature dictionary (qemx.features/1)

Level: logical (device-independent), compiled (device/layout-specific), observable, state (exact ideal pure state: O(2^n), not scalable), noise (light-cone fault rates from the noise model or calibration), context (study settings), oracle (needs the exact simulation: never a recommendation input).

| feature | level | definition | normalization | cost | scalable | unit | source |
|---|---|---|---|---|---|---|---|
| `n` | logical | number of qubits of the logical circuit (width) | none | O(gates) | yes | qubits | cell.n |
| `two_qubit_logical` | logical | two-qubit gates of the logical circuit | none | O(gates) | yes |  | descriptors |
| `two_qubit_logical_per_qubit` | logical | two_qubit_logical / n | per qubit | O(gates) | yes |  |  |
| `non_clifford` | logical | non-Clifford gates of the logical circuit (what CDR/FR/NIL training replaces) | none | O(gates) | yes |  | descriptors |
| `non_clifford_per_qubit` | logical | non_clifford / n | per qubit | O(gates) | yes |  |  |
| `two_qubit` | compiled | two-qubit gates after compilation (routing included) | none | O(gates) | yes |  | circuit |
| `depth` | compiled | depth of the compiled circuit (layers of physical gates) | none | O(gates) | yes |  | circuit |
| `pulses_1q` | compiled | one-qubit physical gates after compilation | none | O(gates) | yes |  | descriptors |
| `routing_overhead` | compiled | two-qubit gates added by routing (compiled − logical) | none | O(gates) | yes |  | circuit |
| `routing_ratio` | compiled | two_qubit / max(two_qubit_logical, 1) | ratio | O(gates) | yes |  |  |
| `two_qubit_density` | compiled | two_qubit / (n · depth): share of qubit-layer slots in two-qubit gates | per qubit-layer | O(gates) | yes |  |  |
| `idle_fraction` | compiled | share of (qubit, layer) slots after a qubit's first gate in which it idles | fraction | O(gates) | yes |  | descriptors |
| `obs_weight` | observable | largest Z-string weight in the observable's decomposition | none | O(gates) | yes |  | descriptors |
| `obs_terms` | observable | number of Z strings in the observable's decomposition | none | O(gates) | yes |  | descriptors |
| `obs_locality` | observable | obs_weight / n | per qubit | O(gates) | yes |  |  |
| `meyer_wallach` | state | Meyer–Wallach Q = 2(1 − mean_k Tr ρ_k²) of the ideal pure state | none | O(2^n) | no |  | descriptors |
| `entropy_half` | state | von Neumann entropy (bits) of the first ⌊n/2⌋ qubits of the ideal pure state | none | O(2^n) | no | bits | descriptors |
| `entropy_half_per_qubit` | state | entropy_half / ⌊n/2⌋ (fraction of the maximum) | per qubit | O(2^n) | no |  |  |
| `entropy_max_cut` | state | largest entropy over contiguous bipartitions of the logical line | none | O(2^n) | no | bits | descriptors |
| `entropy_profile_max` | state | largest half-cut entropy (bits) of the ideal pure state after any layer of the logical circuit: entanglement the final state may conceal (the known-answer QFT ends in a basis state) | none | O(layers · 2^n) | no | bits | qemx.features.entanglement_profile |
| `output_entropy_bits` | state | Shannon entropy of the ideal computational-basis output distribution — a classical quantity, NOT an entanglement measure | none | O(2^n) | no | bits | descriptors |
| `lam_lc` | noise | light-cone fault rate λ_LC: summed error probability of the operations in the observable's backward light cone (gates and idles) | none | O(gates) | yes |  | lambda |
| `lam_lc_gate` | noise | λ_LC from gate errors | none | O(gates) | yes |  | lambda |
| `lam_lc_idle` | noise | λ_LC from idle (relaxation/dephasing) windows | none | O(gates) | yes |  | lambda |
| `lam_ro` | noise | readout fault rate of the measured qubits | none | O(gates) | yes |  | lambda |
| `idle_share` | noise | lam_lc_idle / lam_lc | fraction | O(gates) | yes |  |  |
| `lam_per_2q` | noise | lam_lc / max(light-cone two-qubit gates, 1) | per two-qubit gate | O(gates) | yes |  |  |
| `scale` | context | noise scale s applied to the device model (1 = as calibrated) | none | O(gates) | yes |  | cell |
| `coherent_fraction` | context | share χ of each gate error that is coherent (engine E1) | none | O(gates) | yes |  | cell |
| `log10_budget` | context | log10 of the total shot budget B of the estimate | none | O(gates) | yes | log10 shots |  |
| `ideal` | oracle | exact ideal value of the observable | none | O(2^n) | no |  |  |
| `noisy_fidelity` | oracle | fidelity of the exact noisy state to the ideal state | none | O(4^n) | no |  |  |
| `noisy_purity` | oracle | purity of the exact noisy state | none | O(4^n) | no |  |  |

Missing values are NaN with a reason; a model that cannot handle a missing input abstains.
