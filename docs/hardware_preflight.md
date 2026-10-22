# Hardware preflight (1.5.1): one shared 600-second campaign

**Status: an offline plan. Nothing has been submitted, and nothing in this release submits a job unless a person
with credentials runs a hardware command against a campaign ledger.** Every number below comes from offline planning
on the `fake_fez` device model and from an emulation of the hardware path on it.

This plan replaces the 1.5.0 plan, which spent most of the allowance on a distribution-fidelity discrepancy campaign
(`direction_A_divergence_v3_qpu600.yaml`, 418 s high) whose endpoint differs from the main study's. Version 1.5.1
puts the main study first: a small, well-powered test of whether the simulated ranking of complete QEM protocols holds
on the device, with the discrepancy diagnostics computed on the same circuits.

## 1. Policy

- **One ceiling.** The main study and the supporting discrepancy study share **600 s of QPU usage**. The campaign
  ledger keeps a **60 s reserve**, so at most **540 s** can be admitted; a smaller remaining account allowance governs
  if it is lower (`--account-remaining`).
- **A limit, not a target.** The plan below reserves **331 s**. The rest stays unspent unless a later, separately
  justified plan is made.
- **What is counted.** QPU usage is the time the QPU is locked for our jobs (IBM "Workload usage"). The ledger records
  QPU seconds, shots, jobs and job ids. Waiting time, classical computation and cost are reported separately.
- **Frozen before data.** Circuits, protocols, budgets, the number of blocks, the endpoint, the contrasts and the
  analysis are fixed here (design `transfer-v1`, `qemx/transfer.py`). Predictions are frozen with the day's
  calibration before the first job (`frozen_predictions.json`, written once by `run-ibm` with
  `transfer.predictions_on`: the same device model and simulation path as the predictions below, on the day's
  calibration — on the fake device it reproduces them exactly).

## 2. Usage model and the ledger (`qemx/campaign.py`)

| quantity | value | source |
|---|---|---|
| low | 2 s per sub-job + 0.35 ms × executions | IBM quick formula ("Workload usage", read 2026-10-17) |
| central | 2 s per sub-job + **0.57 ms** × executions | this project's earlier ibm_fez measurement; replaced by Stage 1 |
| high | 1.3 × central | margin |
| sub-jobs | one per 100 circuits (assumption) | IBM does not document the split; calibrated in Stage 1 |
| per-job limit | `max_execution_time` = ⌈high⌉ s | set on every job |
| reservation | max(high, per-job limit) | committed by the ledger until the job is reconciled |

"Executions" are circuits × shots. One Sampler call per shot count is one job (qiskit-ibm-runtime ≥ 0.48 deprecates
different shots within one call), each with at least one sub-job, so the job count drives usage more than the shots
do for these small circuits.

Admission and bookkeeping:

- **One transaction per change.** The ledger is a JSON file (schema `qemx.campaign.ledger/2`) shared by the app and
  the CLIs. Every change takes an exclusive lock on `<ledger>.lock`, re-reads the file, checks, changes it and writes
  it through a uniquely named temporary file. Stale objects and concurrent processes cannot overwrite each other.
  Keep the ledger on a local disk (not a network share or a cloud-synced folder).
- **Admission.** A job is admitted only if its reservation fits in what remains: ceiling (or the smaller account
  allowance) − reserve − reconciled usage − the reservations of every unreconciled job, including submissions whose
  outcome is unknown.
- **Per-job limits inside the allowance.** Each job carries `max_execution_time` = ⌈high⌉ s and the ledger reserves at
  least that much, so the provider-enforced limits of all jobs in flight add up to no more than the admitted
  allowance. (1.5.0 sent ⌈1.1 × high⌉ + 1 s while reserving only the high estimate.) A job that would need more than
  its limit is stopped by the provider; its usage up to then is still billed.
- **Unknown outcomes.** Only a definite "not submitted" releases a reservation. A timeout, a lost connection or any
  other error of the provider call leaves it committed as `submission_unknown`, and no submitter starts until it is
  resolved by the job's client tag (`qemx-ledger-<id>`): automatically by the collectors, or by hand with
  `python -m qemx.campaign resolve`.
- **Mandatory ledger.** Real-provider submission without a valid ledger is refused at the submission boundary; a
  configured path that does not exist is an error. Only fake backends and local simulators run without one.
- **Reconciliation.** Collection reconciles each ended job with `job.usage()`. If a job's reported usage exceeds its
  high estimate, admission stops until the usage model is recalibrated.
- **No automatic retries, job mode by default.** A resumed run never re-sends a recorded block.

**What remains conditional.** The ceiling holds if the provider stops each job at its `max_execution_time` and
reports usage within it; billing granularity or cancellation latency beyond a limit must fit in the 60 s reserve. The
account allowance is the value entered at `init`; usage outside this ledger is not seen. These rules are tested with
mock providers (`tests/test_campaign.py`, `tests/test_spending.py`, `tests/test_ibm_page_ledger.py`,
`tests/test_v151_consolidation.py`, `tests/test_transfer.py`).

## 3. The design (`qemx/transfer.py`, design `transfer-v1`)

**Question.** For circuits whose simulated regimes are clearly separated, does the simulated ranking of complete QEM
protocols at a fixed total budget per estimate hold on the device?

**Circuits** (compiled for the day's backend on its good qubits):

| circuit | role | ideal | why |
|---|---|---|---|
| QFT-4, known answer (input F†\|5⟩, observable P(\|5⟩)) | primary | 1 | strong predicted separation (RO 0.145 vs ZNE-L 0.032) |
| Bernstein–Vazirani, 4 data qubits, s = 11 (observable P(\|1⟩\|s⟩)) | primary | 1 | moderate separation (RO 0.038 vs ZNE-L 0.016) |

A predicted near-tie (Ising-4, 2 Trotter steps: RO 0.0154 vs ZNE-L 0.0171) is left out: about 30 blocks would be
needed for 0.8 power, and at 4 blocks its power would be about 0.1.

**Protocols** (the hardware planner's *effective* protocols; total budget B = 4 000 shots per complete estimate,
every circuit of the protocol included, integer allocation conserved exactly):

| id | protocol | requested → effective |
|---|---|---|
| RAW | unmitigated | — |
| RO | tensored readout-error inversion | — |
| ZNE-L | linear ZNE, global folding λ = 1, 3, with readout inversion | — |
| PEC | PEC (inverse, depolarizing model from the day's calibration), with readout inversion | at most 64 sampled circuits: shots per sampled circuit 1 → 43 (QFT-4) and 32 (BV-4); recorded in `frozen_predictions.json` and simulated as such |

**Blocks.** One block runs every protocol on every circuit once, back to back: a complete estimate each, with its
own calibration circuits, PEC samples and seeds. **K = 4 blocks** (the smallest K with ≥ 0.8 predicted power for
both primary contrasts). Contrasts are paired within blocks.

**Endpoint and analysis.** Signed error e = estimate − exact ideal value of one complete estimate (the main study's
scalar endpoint). Per protocol: MSE = mean(e²) over blocks (bootstrap interval over blocks).

- *Primary contrasts:* the predicted best protocol against readout correction only — ZNE-L vs RO on both circuits.
  For each, d_b = e²_RO,b − e²_ZNE-L,b; one-sided paired t test of mean(d) > 0; Holm over the two contrasts at 0.05.
- *Secondary:* ZNE-L vs PEC on both circuits (reported, not confirmatory; predicted power ≈ 0.25–0.3 at K = 4).
- *Discrepancy on the same circuits:* for every protocol, z = (mean error − predicted bias)/SE and the ratio of the
  observed to the predicted SD with its χ² interval; the unmitigated counts also give the distribution-level
  distance tests of `qemx.discrepancy` against the day's device-model prediction.

**Predictions** (simulations of the effective protocols on the fake_fez model; on the day they are recomputed with the
backend's calibration and frozen before the first job):

| circuit | protocol | predicted bias | predicted SD | predicted RMSE |
|---|---|---|---|---|
| QFT-4 | RAW | −0.179 | 0.006 | 0.179 |
| QFT-4 | RO | −0.145 | 0.010 | 0.145 |
| QFT-4 | ZNE-L | −0.026 | 0.018 | 0.032 |
| QFT-4 | PEC (s = 43) | −0.028 | 0.048 | 0.055 |
| BV-4 | RAW | −0.096 | 0.005 | 0.096 |
| BV-4 | RO | −0.036 | 0.011 | 0.038 |
| BV-4 | ZNE-L | 0.000 | 0.016 | 0.016 |
| BV-4 | PEC (s = 32) | −0.015 | 0.032 | 0.035 |

Predicted power at K = 4: QFT-4 primary 1.00, BV-4 primary 0.85; secondary 0.24 and 0.29.

## 4. Usage of the plan (`reports/hardware_transfer/plan/preflight.json`)

| stage | what | jobs | executions | high s | reserved s |
|---|---|---|---|---|---|
| 1 | usage pilot: 4 small circuits, 2 500 shots each, one job | 1 | 10 000 | 10.0 | 11 |
| 2 | transfer test, per block (2 circuits × 4 protocols) | 18 | 32 000 | 70.5 | 80 |
| 2 | 4 blocks | 72 | 128 000 | 282 | 320 |
| | **total** | 73 | 138 000 | 292 | **331 of 540** |

Each block's 18 jobs (one per circuit and shot count) are listed with their low/central/high estimates and limits in
`preflight.json`. At most 6 blocks would fit under the uncalibrated model.

## 5. What the emulation shows (EMULATION, not hardware)

The whole hardware path — planning, transpiling, twirl/mask draws, sampling, replay and analysis — was run on the
fake device in local testing mode for 24 blocks (`reports/hardware_transfer/rehearsal_emulation`). The emulator now
has the predicting model's physics (idle relaxation in the ALAP schedule, each qubit's asymmetric readout; 1.5.0's
emulator lacked both and was 20 % quieter than the predictions, z ≈ 25 — finding N44).

- Every protocol's mean error agrees with its prediction (|z| ≤ 2.1); SD ratios are near 1, except PEC on QFT-4
  (1.4, 95 % interval 1.1–1.9): with at most 64 sampled circuits the predicted instance variance is approximate.
- Both primary contrasts are confirmed with 24 blocks. Split into six disjoint sets of 4 blocks, the QFT-4 contrast is
  confirmed 6/6 times and the BV-4 contrast 5/6 times, in line with the predicted power (1.00 and 0.85).

An emulation that agrees with its own model checks the pipeline, not the device.

## 6. Commands (after explicit approval only)

```bash
# once: the campaign ledger (one ceiling for both themes)
python -m qemx.campaign init --ledger campaign_ledger.json --ceiling 600 --reserve 60 --account-remaining <s left>

# offline, any time: the plan, the predictions and the emulation
python scripts/hardware_transfer.py plan     --out reports/hardware_transfer/plan
python scripts/hardware_transfer.py rehearse --out reports/hardware_transfer/rehearsal_emulation --blocks 24

# Stage 1: calibrate the usage model (one small job), then reconcile and read the per-shot usage
python scripts/dirA_divergence.py plan manifests/direction_A_divergence_hwpilot.yaml --out data/hwpilot --backend ibm_fez
python scripts/dirA_divergence.py run-ibm data/hwpilot --ledger campaign_ledger.json --mode job
python scripts/dirA_divergence.py collect-ibm data/hwpilot --ledger campaign_ledger.json --wait
python -m qemx.campaign status --ledger campaign_ledger.json

# re-plan with the measured per-shot usage; K is fixed here, by usage only, before any Stage 2 outcome exists
python scripts/hardware_transfer.py plan --out data/transfer_plan --per-shot-ms <measured>

# Stage 2: freeze the day's predictions, submit K blocks (each job admitted by the ledger), collect, analyze
python scripts/hardware_transfer.py run-ibm     --out data/transfer --ledger campaign_ledger.json --blocks 4
python scripts/hardware_transfer.py collect-ibm --out data/transfer --ledger campaign_ledger.json --wait
```

**Interrupted runs.** Every provider call is written to `jobs.json` before it is made, and every job id right after.
After an interruption, `run-ibm` refuses to continue while the ledger holds an unresolved call; once it is resolved
(`python -m qemx.campaign resolve … --job-id <JOB>` or `--not-submitted`), `run-ibm` finishes the interrupted block
with its own saved plan and sends only the jobs not yet sent — a job resolved to an existing id is never sent again.
`collect-ibm` analyzes complete blocks only, and drops a request whose measurement-twirl masks lost a failed job
(never averaged over the surviving masks).

The token comes from `QISKIT_IBM_TOKEN`, a saved account, or a hidden prompt (`--ask-token`). It is never a
command-line argument and is never written to disk. `data/transfer/plans/*.pkl` are the submitted plans (local working
files, SHA-256 recorded in `jobs.json`); never load plan files from someone else.

## 7. Stopping rules

1. Before every job the ledger admits its reservation, or nothing is submitted.
2. After Stage 1: re-run the plan with the measured per-shot usage. If 4 blocks no longer fit under the ledger,
   reduce K to the largest that fits and report the reduced power; do not change shots or protocols to fit.
3. K is decided before Stage 2 from usage alone and is not changed after any Stage 2 outcome is seen (no optional
   stopping or extension).
4. If a job fails or is cancelled, do not resubmit automatically: reconcile, inspect, decide, and record the decision.
5. The reserve is not used to finish a stage.

## 8. What the hardware data can and cannot show

- A confirmed primary contrast supports the simulated ranking for that circuit, at B = 4 000, on that device and in
  that time window. It does not validate the simulated regime maps in general, other circuits, other budgets or other
  devices.
- An unconfirmed contrast is not evidence that the protocols are equivalent.
- A large discrepancy z for a protocol says that its mean error differs from the prediction; it does not identify the
  physical cause.
- Learning methods (CDR, FR, NIL) are not in this hardware test: their training budgets would need far more jobs per
  block. Their transfer remains untested.
