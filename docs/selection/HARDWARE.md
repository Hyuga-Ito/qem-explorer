# Hardware validation of the selection recommender (prepared, not executed)

The protocol-selection study is a fake-device emulation study. Hardware transfer is an optional, secondary stage
(review roadmap, stage F). QEM Explorer 1.7.0 prepares it and **submits nothing**: no job was sent for this release.

## What is prepared

`python -m qemx.selection hardware plan --hardware-config configs/selection/hardware17.yaml --bundle <bundle>`

* a plan of a few tasks, each run with RAW, RO, the bundle's recommendation and its budget-only fallback (SBS-B) at
  one total budget B (every calibration, training and target shot charged), repeated in independent blocks (the unit
  of a hardware risk estimate);
* every protocol recorded and allocated by the approved hardware path (`qemx.hardware.make_plan`: the simulator's
  integer ledgers, the backend's calibration as the supplied model); protocols the hardware path cannot run are listed
  with the reason;
* QPU usage per job from the campaign's conservative usage model (`qemx.campaign.UsageModel`: low = IBM's quick
  formula; central = sub-job overhead + measured per-shot cost; **high = central × 1.3, used for admission**) and
  job-by-job admission against **one shared ceiling, 600 s by default, minus a reserve (60 s)** — or against the campaign
  ledger's remaining allowance when `--ledger` is given;
* `hardware_plan.json` (with a digest of its content) and `hardware_plan.md`.

With the shipped plan and the 1.7.0 bundle (two tasks, four blocks, B = 20 000; the bundle recommends PEC for both
tasks, which is also SBS-B's choice, so each task has three distinct protocol runs) every job fits: 24 jobs,
452 s reserved on the high estimate of the 540 s available. Three tasks at four blocks do not fit (28 of 36 jobs
admitted) — the preflight says so instead of trimming silently. The plan of this release is
`reports/selection_v170/hardware/hardware_plan.{json,md}` (digest `a0f34288f20a049b…`).

Correction in 1.7.0: the plan used to multiply the shots of a merged PEC circuit by the number of times the recipe
asked for it, although the merged request already carries the shares of every occurrence; it reported PEC runs of
about 3·10⁸ shots at B = 20 000 and declared a feasible plan infeasible. Every run is now charged exactly B (test
`test_hardware_plan_charges_each_protocol_run_exactly_its_budget`).

## What execution needs (owner only)

1. The owner's authorization of *this* plan: `QEMX_HW_AUTHORIZE=<the plan's digest>` (a changed plan has another
   digest; `hardware check --plan hardware_plan.json` verifies it).
2. The campaign ledger (`QEMX_CAMPAIGN_LEDGER`, created with `python -m qemx.campaign init --ceiling 600 --reserve
   60`): one shared ceiling for every hardware run of the thesis; reservations on the high estimate; nothing retried
   automatically; a submission whose outcome is unknown blocks further ones until resolved.
3. Credentials from the environment (`QISKIT_IBM_TOKEN`), a saved account or a hidden prompt — never a command-line
   argument, never written or printed.

Submission itself goes through the approved, tested IBM workflow with the ledger (the IBM Quantum page, or
`scripts/hardware_transfer.py` for the transfer test); `qemx.selection.hardware.execute` is the gate (authorization,
ledger, unresolved submissions) and is exercised only against `campaign.MockProvider` in the tests.

## What a hardware result could and could not show

Four blocks per protocol estimate a hardware risk only coarsely; the plan is a transfer check (does the simulated
ranking of the recommended protocol against RAW, RO and SBS-B hold on the device?), not a re-estimation of the study.
Its numbers would be labelled *hardware observations*, never merged with the emulated labels, and reported with the
device's calibration of the day.
