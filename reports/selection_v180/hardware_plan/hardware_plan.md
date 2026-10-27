# Hardware validation plan (not executed)

Digest `6015610c337d89a3`. Budget B = 20,000 shots per protocol run, 4 blocks (independent deployments) per run; device model fake_fez (stand-in for ibm_fez).

Shared QPU ceiling 600 s, reserve 60 s: 540 s available; the plan reserves 524 s (high estimates); 28 of 28 jobs fit.

| task | protocol | roles | supported | circuits | shots per run | blocks |
|---|---|---|---|---|---|---|
| ising-4(steps=2)|in=default|O=default|fake_fez/best | RAW | RAW | yes | 1 | 20,000 | 4 |
| ising-4(steps=2)|in=default|O=default|fake_fez/best | RO | RO | yes | 3 | 20,000 | 4 |
| ising-4(steps=2)|in=default|O=default|fake_fez/best | PEC-TW | recommended, SBS-B | yes | 285 | 20,000 | 4 |
| ghz-4(theta=1.0472)|in=default|O=default|fake_fez/best | RAW | RAW | yes | 1 | 20,000 | 4 |
| ghz-4(theta=1.0472)|in=default|O=default|fake_fez/best | RO | RO | yes | 3 | 20,000 | 4 |
| ghz-4(theta=1.0472)|in=default|O=default|fake_fez/best | PEC | recommended | yes | 72 | 20,000 | 4 |
| ghz-4(theta=1.0472)|in=default|O=default|fake_fez/best | PEC-TW | SBS-B | yes | 72 | 20,000 | 4 |

To execute (owner only): `python -m qemx.selection hardware check --plan hardware_plan.json` verifies the plan; set `QEMX_HW_AUTHORIZE=6015610c337d89a3fb6421200b8816446b5f861c9d1831d38dcac20a72e7d96c`, a campaign ledger (`QEMX_CAMPAIGN_LEDGER`) and credentials (environment, saved account or hidden prompt), then submit through the approved IBM workflow with the ledger (`docs/selection/HARDWARE.md`). Neither `plan` nor `check` submits anything.
