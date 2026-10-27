# Divergence rehearsal — EMULATION (not hardware evidence)

Manifest `direction_A_divergence_v3_qpu600.yaml`; primary predictor `depolarizing`; 2 null seed(s); alternatives: crosstalk=0.05, scale=1.5.

## False positives (truth = the predictor's own model)

| flag | flagged / tests | rate ± SE |
|---|---|---|
| abs_z_gt_3 | 2 / 1120 | 0.0018 ± 0.0013 |
| p_lt_alpha | 49 / 1120 | 0.0437 ± 0.0061 |
| holm | 0 / 1120 | 0.0000 ± 0.0000 |
| families_with_any_holm_flag | 0 / 12 | 0.0000 ± 0.0000 |

z under the null: mean -0.103, sd 0.944 (calibrated: 0 and 1; sd < 1 is conservative).

## Detection power (Holm, predicted > observed)

| alternative | method | flagged / circuits | rate ± SE |
|---|---|---|---|
| alternative: crosstalk=0.05 | RAW | 34 / 120 | 0.283 ± 0.041 |
| alternative: crosstalk=0.05 | RAW·PV | 9 / 40 | 0.225 ± 0.066 |
| alternative: crosstalk=0.05 | RAW·agg | 25 / 40 | 0.625 ± 0.077 |
| alternative: crosstalk=0.05 | RO | 20 / 120 | 0.167 ± 0.034 |
| alternative: crosstalk=0.05 | ZNE-G | 0 / 120 | 0.000 ± 0.000 |
| alternative: crosstalk=0.05 | ZNE-L | 0 / 120 | 0.000 ± 0.000 |
| alternative: scale=1.5 | RAW | 18 / 120 | 0.150 ± 0.033 |
| alternative: scale=1.5 | RAW·PV | 4 / 40 | 0.100 ± 0.047 |
| alternative: scale=1.5 | RAW·agg | 18 / 40 | 0.450 ± 0.079 |
| alternative: scale=1.5 | RO | 6 / 120 | 0.050 ± 0.020 |
| alternative: scale=1.5 | ZNE-G | 0 / 120 | 0.000 ± 0.000 |
| alternative: scale=1.5 | ZNE-L | 0 / 120 | 0.000 ± 0.000 |

Planned QPU usage of this manifest (conservative planner, qemx.campaign; same job split as the hardware preflight): 216–418 s (central 322 s); fits the 600 s policy with a 60 s reserve: yes.
