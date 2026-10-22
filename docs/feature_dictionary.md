# Feature dictionary (qemx.features/2)

Generated from `qemx/features.py` (`python -m qemx.study features --out docs/feature_dictionary.md`).

Level: logical (the circuit as written), compiled (device/layout/compiler-specific), observable, state (exact ideal pure state: exponential, not scalable), noise (light-cone fault rates under the experiment's noise model — model-dependent descriptors, not sufficient statistics), context (study settings and budget), oracle (needs the exact simulation: never a recommendation input).

Acquisition tier: 1 = circuit, observable and budget (no noise model, no simulation); 2 = calibrated noise model (light-cone fault rates, device scale, coherent fraction); 3 = exact ideal state (exponential in n; small systems only); 0 = oracle (needs the exact ideal or noisy simulation; never a recommendation input).

| feature | level | tier | definition | normalization | cost | scalable | unit | source |
|---|---|---|---|---|---|---|---|---|
| `n` | logical | 1 | number of qubits of the logical circuit (width) | none | O(gates) | yes | qubits | cell.n |
| `two_qubit_logical` | logical | 1 | two-qubit gates of the logical circuit as written (input preparation and basis change add none) | none | O(gates) | yes |  | descriptors |
| `two_qubit_logical_per_qubit` | logical | 1 | two_qubit_logical / n | per qubit | O(gates) | yes |  |  |
| `non_clifford` | logical | 1 | non-Clifford gates of the logical circuit as written (T, T† and rotations by non-Clifford angles): logical content before decomposition into native gates — not the number of slots a training generator replaces (see non_clifford_slots) | none | O(gates) | yes |  | descriptors |
| `non_clifford_per_qubit` | logical | 1 | non_clifford / n | per qubit | O(gates) | yes |  |  |
| `two_qubit` | compiled | 1 | two-qubit gates of the compiled circuit (native gates after layout, routing and optimization) | none | O(gates) | yes |  | circuit |
| `depth` | compiled | 1 | ASAP layers of the compiled circuit (qemx.circuit.layers): every gate, virtual RZ included, takes one layer on its qubits whatever its duration; noise operations and barriers take none | none | O(gates) | yes |  | circuit |
| `pulses_1q` | compiled | 1 | non-virtual one-qubit gates of the compiled circuit (√X, X, …; RZ is virtual) | none | O(gates) | yes |  | descriptors |
| `non_clifford_slots` | compiled | 1 | non-Clifford gates of the compiled circuit (native basis: RZ angles that are not multiples of π/2): exactly the slots the CDR/FR/NIL training generators replace by Clifford gates (qemx.transforms.non_clifford_slots) | none | O(gates) | yes |  | descriptors.non_clifford_isa |
| `two_qubit_compile_overhead` | compiled | 1 | two_qubit − two_qubit_logical (signed). It mixes the decomposition of logical gates into native two-qubit gates (a controlled phase becomes two CZ), SWAPs inserted by routing and gates removed by optimization; it is not a routing count (schema /1: routing_overhead) | none | O(gates) | yes |  | circuit |
| `two_qubit_compile_ratio` | compiled | 1 | two_qubit / max(two_qubit_logical, 1): the same mixture of decomposition, routing and optimization as a ratio (schema /1: routing_ratio) | ratio | O(gates) | yes |  |  |
| `two_qubit_per_qubit_layer` | compiled | 1 | two_qubit / (n · depth): compiled two-qubit gates per (qubit, layer) slot, a rate with depth as defined above. A two-qubit gate occupies two slots of one layer, so the share of slots occupied by two-qubit gates is twice this value (schema /1: two_qubit_density) | per qubit-layer | O(gates) | yes |  |  |
| `idle_fraction` | compiled | 1 | share of (qubit, layer) slots after a qubit's first gate in which it has no gate (layers as in depth: a scheduling proxy, not durations) | fraction | O(gates) | yes |  | descriptors |
| `obs_weight` | observable | 1 | largest Pauli weight among the observable's non-identity terms: the Z strings of a diagonal observable's Walsh–Hadamard decomposition, or the Pauli strings of a Pauli sum | none | O(n·2^n) diagonal (2^n stored values); O(terms·n) Pauli sum | yes |  | descriptors |
| `obs_terms` | observable | 1 | number of non-identity terms (\|coefficient\| > 1e-12) | none | O(n·2^n) diagonal; O(terms·n) Pauli sum | yes |  | descriptors |
| `obs_weight_mean` | observable | 1 | mean weight of the non-identity terms (unweighted) | none | O(n·2^n) diagonal; O(terms·n) Pauli sum | yes |  | qemx.phase.descriptors.observable_descriptors |
| `obs_support` | observable | 1 | number of qubits (classical bits for a diagonal observable) that any term acts on | none | O(n·2^n) diagonal; O(terms·n) Pauli sum | yes | qubits | qemx.phase.descriptors.observable_descriptors |
| `obs_settings` | observable | 1 | measurement settings per estimate: 1 for a diagonal observable (one basis); for a Pauli sum the qubit-wise-commuting groups of the app's greedy grouping (qemx.hamiltonian.qwc_groups), each a separately executed circuit | none | O(terms²·n) | yes |  | qemx.phase.descriptors.observable_descriptors |
| `obs_locality` | observable | 1 | obs_weight / n | per qubit | O(gates) | yes |  |  |
| `meyer_wallach` | state | 3 | Meyer–Wallach Q = 2(1 − mean_k Tr ρ_k²) of the ideal pure state | none | storage 16·2^n B; time O(gates·2^n) + O(n·2^n) | no |  | descriptors |
| `entropy_half` | state | 3 | von Neumann entropy (bits) of the first ⌊n/2⌋ qubits of the ideal pure state (Schmidt spectrum of the smaller side) | none | storage 16·2^n B; time O(gates·2^n) + O(2^(3n/2)) | no | bits | descriptors |
| `entropy_half_per_qubit` | state | 3 | entropy_half / ⌊n/2⌋ (fraction of the maximum) | per qubit | storage 16·2^n B; time O(gates·2^n) + O(2^(3n/2)) | no |  |  |
| `entropy_max_cut` | state | 3 | largest entropy over the n − 1 contiguous bipartitions (first k qubits \| rest) of the logical qubit order, each from its smaller side | none | storage 16·2^n B; time O(gates·2^n) + O(2^(3n/2)) | no | bits | descriptors |
| `entropy_profile_max` | state | 3 | largest half-cut entropy (bits) of the ideal pure state after any layer of the logical circuit: entanglement the final state may conceal (the known-answer QFT ends in a basis state) | none | storage 16·2^n B; time O(gates·2^n + layers·2^(3n/2)) | no | bits | qemx.phase.descriptors.entanglement_profile |
| `output_entropy_bits` | state | 3 | Shannon entropy of the ideal computational-basis output distribution — a classical quantity, NOT an entanglement measure | none | storage 16·2^n B; time O(gates·2^n) | no | bits | descriptors |
| `lam_lc` | noise | 2 | light-cone fault rate λ_LC: summed Pauli-twirled fault probability of the fault locations (gate errors, idle windows) in the observable's backward structural light cone, under the experiment's noise model (device mode: the selected device model with its scale and coherent fraction; injector mode: the injector). Pauli sums: the \|coefficient\|-weighted mean over terms of each term's λ_LC (its qubits mapped through the measurement map; the settings' basis rotations are not included). A model-dependent descriptor under an independent-location surrogate (Pauli-twirled fault probabilities: coherent orientation and accumulation are discarded; structural light cones can exceed the exact Heisenberg support; quasi-static noise enters through its per-segment marginal), not a sufficient statistic for mitigation performance | none | O(locations) | yes |  | lambda |
| `lam_lc_gate` | noise | 2 | λ_LC from gate errors (same rule) | none | O(locations) | yes |  | lambda |
| `lam_lc_idle` | noise | 2 | λ_LC from idle (relaxation/dephasing) windows (same rule) | none | O(locations) | yes |  | lambda |
| `lam_ro` | noise | 2 | readout fault rate: Σ over the measured support of the mean flip probability (P(1\|0) + P(0\|1))/2; Pauli sums: \|coefficient\|-weighted mean over terms | none | O(gates) | yes |  | lambda |
| `idle_share` | noise | 2 | lam_lc_idle / lam_lc | fraction | O(gates) | yes |  |  |
| `lam_per_2q` | noise | 2 | lam_lc / max(light-cone two-qubit gates, 1) | per two-qubit gate | O(gates) | yes |  |  |
| `scale` | context | 2 | noise scale s of the device model (device-specification modifier s; 1 = as calibrated). Not defined for the noise injector (NaN) | none | O(gates) | yes |  | cell / backend specification |
| `coherent_fraction` | context | 2 | coherent fraction χ of the homogenized device model (modifier chi; engine E1); 0 for the device's own model. Not defined for the noise injector (NaN) | none | O(gates) | yes |  | cell / backend specification |
| `log10_budget` | context | 1 | log10 of the total shot budget B of the estimate | none | O(gates) | yes | log10 shots |  |
| `ideal` | oracle | 0 | exact ideal value of the observable | none | O(2^n) | no |  |  |
| `noisy_fidelity` | oracle | 0 | fidelity of the exact noisy state to the ideal state | none | O(4^n) | no |  |  |
| `noisy_purity` | oracle | 0 | purity of the exact noisy state | none | O(4^n) | no |  |  |

Missing values are NaN with a reason; a model that cannot handle a missing input abstains.

## Names of earlier schemas

| schema /1 name | /2 name | values identical | note |
|---|---|---|---|
| `routing_overhead` | `two_qubit_compile_overhead` | no | renamed: it is not a routing count. /1 records stored max(compiled − logical, 0) and the /1 live path the signed difference; /2 is the signed difference everywhere (equal when ≥ 0) |
| `routing_ratio` | `two_qubit_compile_ratio` | yes | renamed (same formula): not a routing ratio |
| `two_qubit_density` | `two_qubit_per_qubit_layer` | yes | renamed (same formula): a rate per (qubit, layer) slot, not the share of occupied slots |

## Changes

* /2 (1.5.1): routing_overhead → two_qubit_compile_overhead (signed; decomposition + routing − optimization), routing_ratio → two_qubit_compile_ratio, two_qubit_density → two_qubit_per_qubit_layer (a rate; the occupied-slot share is twice it). New: non_clifford_slots (compiled slots replaced by training), obs_weight_mean, obs_support, obs_settings (setting-aware Pauli sums). Pauli-sum observables get observable and λ features (|c|-weighted rule) instead of NaN. Noise features in device mode come from the selected device model with its scale and coherent fraction (1.5.0's live path used the inactive injector and scale 1). Structural and observable features no longer depend on the exact-state cutoff. State features use the smaller side of every cut and one incrementally advanced statevector (values unchanged to 1e-10).

## Feature sets of the study

* **circuit_observable**: `n`, `two_qubit_logical`, `non_clifford`, `two_qubit`, `depth`, `obs_weight`, `log10_budget` (tiers 1)
* **structural**: `n`, `two_qubit_logical`, `non_clifford`, `two_qubit`, `depth`, `obs_weight`, `lam_lc`, `lam_ro`, `scale`, `coherent_fraction`, `log10_budget` (tiers 1, 2)
* **normalized**: `two_qubit_logical_per_qubit`, `non_clifford_per_qubit`, `two_qubit_per_qubit_layer`, `two_qubit_compile_ratio`, `obs_locality`, `lam_lc`, `lam_per_2q`, `idle_share`, `lam_ro`, `coherent_fraction`, `log10_budget` (tiers 1, 2)
* **with_state**: `n`, `two_qubit_logical`, `non_clifford`, `two_qubit`, `depth`, `obs_weight`, `lam_lc`, `lam_ro`, `scale`, `coherent_fraction`, `log10_budget`, `entropy_half`, `entropy_profile_max`, `meyer_wallach` (tiers 1, 2, 3)
