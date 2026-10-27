# Hardware validation plan (not executed)

Digest `a0f34288f20a049b`. Budget B = 20,000 shots per protocol run, 4 blocks (independent deployments) per run; device model fake_fez (stand-in for ibm_fez).

Shared QPU ceiling 600 s, reserve 60 s: 540 s available; the plan reserves 452 s (high estimates); 24 of 24 jobs fit.

| task | protocol | roles | supported | circuits | shots per run | blocks |
|---|---|---|---|---|---|---|
| ising-4(steps=2)|in=default|O=default|fake_fez/best | RAW | RAW | yes | 1 | 20,000 | 4 |
| ising-4(steps=2)|in=default|O=default|fake_fez/best | RO | RO | yes | 3 | 20,000 | 4 |
| ising-4(steps=2)|in=default|O=default|fake_fez/best | PEC | recommended, SBS-B | yes | 285 | 20,000 | 4 |
| ghz-4(theta=1.0472)|in=default|O=default|fake_fez/best | RAW | RAW | yes | 1 | 20,000 | 4 |
| ghz-4(theta=1.0472)|in=default|O=default|fake_fez/best | RO | RO | yes | 3 | 20,000 | 4 |
| ghz-4(theta=1.0472)|in=default|O=default|fake_fez/best | PEC | recommended, SBS-B | yes | 72 | 20,000 | 4 |

To execute (owner only): set `QEMX_HW_AUTHORIZE=a0f34288f20a049b4e3dec3dbfa119262c2b74d0283fdd9415f76ed6d2e89c0e`, a campaign ledger (`QEMX_CAMPAIGN_LEDGER`) and credentials, then `python -m qemx.selection hardware execute --plan hardware_plan.json`.
