# QEM Explorer v1.3 — release notes (5 October 2026)

v1.3 adds a research backend for Direction A (equal-budget phase diagrams) and fixes five v1.2 errors. The full
design, self-audit and manifest are in `docs/v13/direction_A_design.pdf`.

**Tests:** 368 pass (327 from v1.2 + 41 new).

## New

### `qemx.phase` — a Streamlit-free pipeline

Modules:

- `lightcone`
- `scaling`
- `families`
- `descriptors`
- `ledger`
- `cell`
- `validate`
- `store`
- `manifest`
- `run`
- `fit`
- `plots`

Command-line interface: `python -m qemx.phase.run plan|masterlist|run|collect|status`.

### Device specifications

The form is `fake_fez@s=2,ro=0,hom=1,chi=0.25,eng=exact`:

- **`s`** scales every gate error and relaxation rate.
- **`ro`** scales readout.
- **`hom=1`** homogenizes to the medians.
- **`chi`** sets the coherent fraction (exact engine only).
- **`eng=exact`** selects the exact superoperator engine.

Compilation always uses the base device, so a noise sweep runs one identical circuit.

### Exact device engine (`qemx/device_exact.py`)

- Replays Aer's device model (schedule, relaxation, gate errors). Agreement with Aer is ≤ 5·10⁻⁹, and the remaining difference is Aer's.
- Applies signed PEC maps, so PEC's mean on a fake device is exact.

### Equal-budget ledger (`qemx.phase.ledger`)

- Integer allocations satisfy N_t + T·s_tr + c·s = B exactly (asserted).
- Two training policies: `fixed` and `opt`.
- B*(ε), crossovers and winners.
- Learning methods use bias, variance and target variance measured by Monte Carlo at five training levels.

### Validation (`qemx.phase.validate`)

- Monte Carlo of complete experiments, including resampled calibration, at the ledger's exact integer allocation.
- Smoke run: 128 of 128 checks pass, and the z-scores are consistent with N(0, 1).

### Manifest

`manifests/direction_A_v1.yaml` is pre-registered: 1,705 cells in 7 stages. The master list is `docs/v13/direction_A_v1_cells.csv`, with λ_LC computed in advance for every cell.

### Hardware path

`fold_guard_transpiled` refuses to submit a job whose two-qubit gate count changes during transpilation.

## Fixed (these affect v1.2 results)

### V1 — readout calibration mapped to the wrong bit

Tensored and M3 readout mitigation, and on hardware also TREX, measured the calibration circuits through the identity map. When routing or relabeling permutes the bits, this applies qubit i's confusion matrix to bit i.

- **Size:** a bias of 2.5–3.7·10⁻³, independent of λ, on n = 6 fake_fez chains.
- **Action:** re-check any app result that combined readout mitigation with a permuted measurement map.

### V2 — NIL-Lasso training variance was meaningless

The delta-method training variance of NIL-Lasso differentiated a rank-deficient active set. In a test cell it was 10³× too large. `qemx.phase` measures it by Monte Carlo instead.

### V3 — ZNE sampler shot rounding

ZNE's sampler rounded the per-circuit shots, so they did not add up to N. It now uses the exact integer optimum (`budget.optimal_integer_split`).

### V4 — DD pulses before the first gate

On the device engine, DD pulses were placed before a qubit's first gate. They are now skipped, which matches IBM's `skip_reset_qubits` default.

### V5 — identity-Hamiltonian result

A Pauli-sum Hamiltonian that is a multiple of the identity returned `train_circuits = 1.0`.

## Changed

- **Version:** 1.3.0.
- **New requirements:** `pyyaml`, `jsonschema`, `threadpoolctl`; `pyarrow` is optional (Parquet tables).
- **New field:** `MethodResult` now has `cal_fun` and `train_resampler`.
- **Learning sampler:** readout mitigation is vectorized (linear modes).
