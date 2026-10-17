# Hardware preflight plan: one shared 600-second campaign

**Status: plan for review. Nothing has been submitted.** No command in this release submits a job unless both of these are true: the user has created a campaign ledger, and the user has run a `run-ibm` step (or pressed **Run on IBM Quantum**) with credentials. Every number below comes from offline planning and emulation.

## 1. Policy

- **One ceiling.** Both thesis themes share one ceiling of **600 s of QPU usage** in total: the main study (QEM performance and transfer) and the supporting study (model discrepancy). The ledger keeps a **60 s reserve**, so at most **540 s** can be admitted. If the account's remaining allowance is smaller, the smaller number applies (`--account-remaining`).
- **Not a target.** The ceiling bounds the plan; nothing is added to use it up. The confirmatory circuits and their shot allocations below are fixed *before* any hardware result exists.
- **What is counted.** QPU usage is the time the QPU is locked for our jobs (IBM "Workload usage"). The ledger records:
  - QPU seconds;
  - shots, number of jobs and job ids.

  Waiting time, classical computation and monetary cost are reported separately and are not limited by the ledger.
- **What cannot be bounded per job.** `max_execution_time` limits each job's QPU usage only. A batch or session `max_time` is an elapsed-time TTL that fails queued jobs. No provider setting caps a campaign, so the app keeps its own ledger. A job cancelled by the user, or stopped by its limit, still reports the usage it consumed up to that point.

## 2. Usage model and admission (`qemx/campaign.py`)

| quantity | value | source |
|---|---|---|
| low | 2 s per sub-job + 0.35 ms × executions | IBM quick formula ("Workload usage", read 2026-10-17) |
| central | 2 s per sub-job + **0.57 ms** × executions | this project's earlier ibm_fez measurement (replaced by Stage 1's pilot) |
| high (admission) | 1.3 × central | margin |
| sub-jobs | one per 100 circuits (assumption) | IBM does not document the split; to be calibrated in Stage 1 |

"Executions" means the number of circuits times the number of shots.

How admission works:

- **Before a job is submitted**, the ledger admits it only if its *high* estimate fits within the remaining allowance. That remainder is the ceiling, minus the reserve, minus reconciled usage, minus the high estimates of all unreconciled jobs.
- **When it is submitted**, each job's `max_execution_time` is set to 1.1 × its high estimate.
- **Recording.** Job ids are written to `jobs.json` and to the ledger right after each job, so a crash cannot lose one.
- **Collecting.** `collect-ibm --ledger` reconciles each job with the usage the provider reports through `job.usage()`.
- **Overruns.** If any job's reported usage exceeds its high estimate, admission stops until the usage model is recalibrated.
- **No automatic retries.** After an interruption, a reservation without a job id blocks further submission until it is resolved by hand: `python -m qemx.campaign resolve`, after checking the provider's job list.

These rules are tested with a mock provider in `tests/test_campaign.py`. The tests cover failures, cancellations, a crash between submission and bookkeeping, resuming without duplicate submissions, overruns and the bound on outstanding work.

## 3. Stages

| stage | what | circuits / methods | executions | planned usage (low / central / **high**) | runs only if |
|---|---|---|---|---|---|
| 0 | Offline validation (done) | `plan` on the fake twin, emulated null and alternatives, preflight | — | 0 s | — |
| 1 | Pilot: calibrate the usage model | `direction_A_divergence_hwpilot.yaml`: 1 job, GHZ-2/3 and QFT-2/3, raw, 2 500 shots each | 10 000 | 6 / 8 / **10 s** | ledger initialized; approval given |
| 2 | Confirmatory core: model discrepancy + transfer of the raw/RO/ZNE ranking | `direction_A_divergence_v3_qpu600.yaml`: 35 benchmarks × 3 layout variants × {RAW, RO, ZNE-G, ZNE-L}, 3 000 shots per benchmark and method | 480 000 (8 jobs) | 216 / 322 / **418 s** | Stage 1 reconciled; the re-run preflight still admits every job |
| 3 | Transfer of learning methods (optional) | `hardware_transfer_qft4.yaml`, `hardware_transfer_ising4.yaml`: {raw, ZNE-R, CDR (8 × 512), NIL small budget}, 12 000 shots per method | ≈ 96 000 | ≈ 47 / 75 / **97 s** | after Stage 2 is reconciled, the ledger admits it in full; otherwise it is dropped, not shrunk |
| | **Total (high)** | | | **≈ 525 s of the 540 s admissible** | |

In Stage 2, the RAW, RO and ZNE data serve both themes. They measure model discrepancy, and they test whether the simulated ranking of raw, readout-mitigated and ZNE estimates holds on the device. They are counted once.

### Stage 2 job table (conservative model, `python -m qemx.campaign preflight`)

| job | executions | circuits | low s | central s | high s | max_execution_time s |
|---|---|---|---|---|---|---|
| 240 circuits × 65 shots | 15,600 | 240 | 11.5 | 14.9 | 19.4 | 23 |
| 300 circuits × 200 shots | 60,000 | 300 | 27.0 | 40.2 | 52.3 | 59 |
| 300 circuits × 200 shots | 60,000 | 300 | 27.0 | 40.2 | 52.3 | 59 |
| 120 circuits × 200 shots | 24,000 | 120 | 12.4 | 17.7 | 23.0 | 27 |
| 240 circuits × 214 shots | 51,360 | 240 | 24.0 | 35.3 | 45.9 | 52 |
| 240 circuits × 321 shots | 77,040 | 240 | 33.0 | 49.9 | 64.9 | 73 |
| 120 circuits × 600 shots | 72,000 | 120 | 29.2 | 45.0 | 58.6 | 66 |
| 120 circuits × 1000 shots | 120,000 | 120 | 46.0 | 72.4 | 94.1 | 105 |

**Why not the 1.4.2 `qpu10` manifest.** At 4 500 shots per benchmark and method it needs 596 s under the high estimate, which is more than the 540 s the ledger admits. The 1.4.2 notes put it at "≈ 530 s with a 1.3× margin", but that figure left out the per-sub-job overhead. The v3 manifest is a new version (v2 is unchanged). It keeps the design and cuts shots to 3 000. As a consequence, shot noise per circuit grows by about 1.22×, and so does the smallest detectable discrepancy.

## 4. Backend, mode and settings

- **Backend.** The study's backend is `ibm_fez`, as in the manifest. On the day of the run, `plan --backend <name>` compiles against that day's calibration. A different backend is an explicit choice: the app never switches devices silently, and `_check_backend` refuses a backend that does not match the plan.
- **Mode.** Use **job mode** (`--mode job`, the default). On the Open plan, a batch's 10-minute TTL counts elapsed time, including queue waits, so a ~420 s batch could fail queued jobs. Sessions are not available on the Open plan.
- **Unchanged settings.** Repetition delay, shots per circuit and mitigation settings are not changed to fit the budget. Any change would be a new manifest version with its scientific consequence stated.

## 5. What the hardware data can and cannot show (from offline predictions)

- **Discrepancy, emulated rehearsal of Stage 2** (`reports/rehearsal_v3`, EMULATION):
  - *Null case* (the predictor's own model): Holm flags 0 / 1 120 rows. Unadjusted p < 0.05 in 4.4 % ± 0.6 % of rows. z has mean −0.10 and SD 0.94, so the test is calibrated, slightly conservative.
  - *Power*: injected ZZ crosstalk of 0.05 rad is flagged in 28 % of raw rows, 62 % of aggregated raw rows, 17 % of readout-mitigated rows and **0 % of ZNE rows**. A 1.5× under-reported error is flagged in 15 %, 45 %, 5 % and 0 %.
  - *Reading*: at this budget the campaign can detect discrepancies of roughly that size in raw and aggregated data. It cannot detect them through ZNE, whose variance hides them. A null result does not establish that the model is adequate.
- **Transfer of learning methods, simulated predictions** on fake_fez at 12 000 shots per method:

  | circuit | raw RMSE | ZNE-R RMSE | CDR RMSE | NIL (small) RMSE |
  |---|---|---|---|---|
  | QFT-4, known answer | 0.179 | 0.041 | 0.029 | 0.169 |
  | Ising-4, 2 steps | 0.031 | 0.022 | 0.026 | 0.063 |

  - **QFT-4.** The predicted ranking is well separated: per-method SDs are ≤ 0.03, except NIL's 0.15. One run can therefore test it.
  - **Ising-4.** The predicted differences (≈ 0.005–0.009) are smaller than the per-run SDs (≈ 0.012). Ising-4 is a control: the simulation predicts *no* resolvable ranking, so a large difference on hardware would be the surprise.
  - **Scope.** These two circuits cannot validate the simulated regime maps in general. They are a bounded transfer check.

## 6. Commands (after explicit approval only)

```bash
# once: the campaign ledger (the ceiling covers both themes)
python -m qemx.campaign init --ledger campaign_ledger.json --ceiling 600 --reserve 60 --account-remaining <s left on the account>

# Stage 1 (pilot): calibrate the per-shot usage, then reconcile
python scripts/dirA_divergence.py plan manifests/direction_A_divergence_hwpilot.yaml --out data/hwpilot --backend ibm_fez
python -m qemx.campaign preflight data/hwpilot --ledger campaign_ledger.json
python scripts/dirA_divergence.py run-ibm data/hwpilot --ledger campaign_ledger.json --mode job
python scripts/dirA_divergence.py collect-ibm data/hwpilot --ledger campaign_ledger.json --wait
python -m qemx.campaign status --ledger campaign_ledger.json     # measured per-shot = reported usage / 10 000

# Stage 2: re-plan on the day, re-run the preflight with the calibrated per-shot cost, submit, collect, analyze
python scripts/dirA_divergence.py plan manifests/direction_A_divergence_v3_qpu600.yaml --out data/dirA_div_v3 --backend ibm_fez
python -m qemx.campaign preflight data/dirA_div_v3 --ledger campaign_ledger.json --per-shot-ms <measured>
python scripts/dirA_divergence.py run-ibm data/dirA_div_v3 --ledger campaign_ledger.json --mode job --per-shot-ms <measured>
python scripts/dirA_divergence.py collect-ibm data/dirA_div_v3 --ledger campaign_ledger.json --wait
python scripts/dirA_divergence.py analyze data/dirA_div_v3 --tag ibm --out reports/dirA_div_v3_ibm --workers 4

# Stage 3 (only if the ledger admits it): QEMX_CAMPAIGN_LEDGER=campaign_ledger.json streamlit run app.py
#   Playground → Export → load examples/experiments/hardware_transfer_qft4.yaml → Run on IBM Quantum
```

The token comes from `QISKIT_IBM_TOKEN`, from a saved account, or from a hidden prompt (`--ask-token`). It is never a command-line argument and is never written to disk.

## 7. Stopping rules

1. Before every job: the ledger admits its high estimate, or nothing is submitted.
2. After Stage 1: if the measured per-shot usage exceeds 0.74 ms (1.3 × 0.57), re-run the Stage 2 preflight at the measured value. If it no longer fits, cut Stage 3 first, then reduce Stage 2's shots per benchmark through a new manifest version. Do not raise the margin silently.
3. If any job ends in an error or is cancelled, do not resubmit automatically. Reconcile it, inspect it, and decide.
4. If the account's remaining allowance is below the plan, the smaller value governs (`--account-remaining`).
5. The reserve is not used up to "finish" a stage.
