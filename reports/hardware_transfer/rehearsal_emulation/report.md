# Hardware transfer test — EMULATION on fake_fez (local testing mode, 24 blocks)

Design `transfer-v1`: repeated blocks of complete protocols, paired within blocks; endpoint = signed error of one complete estimate against the exact ideal value.

## Contrasts

| circuit | kind | better (predicted) | worse | predicted RMSE | blocks | mean d = e²_worse − e²_better | one-sided p | Holm p | confirmed |
|---|---|---|---|---|---|---|---|---|---|
| qft4 | primary | ZNE-L | RO | 0.032 vs 0.145 | 24 | 0.0208 | 0 | 0 | yes |
| qft4 | secondary | ZNE-L | PEC | 0.032 vs 0.055 | 24 | 0.00567 | 0.006 | — | — |
| bv4 | primary | ZNE-L | RO | 0.0164 vs 0.0377 | 24 | 0.00121 | 2.4e-07 | 2.4e-07 | yes |
| bv4 | secondary | ZNE-L | PEC | 0.0164 vs 0.0353 | 24 | 0.000762 | 0.0069 | — | — |

## Per protocol (discrepancy against the prediction on the same circuits)

| circuit | protocol | blocks | RMSE [95 % CI of MSE] | predicted RMSE | mean error ± SE | predicted bias | z | SD observed / predicted [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| bv4 | PEC | 24 | 0.0331 [0.000692, 0.00166] | 0.0353 | -0.000773 ± 0.00691 | -0.0151 | 2.1 | 1.1 [0.82, 1.5] |
| bv4 | RAW | 24 | 0.0947 [0.00865, 0.0093] | 0.0961 | -0.0946 ± 0.000885 | -0.096 | 1.6 | 0.93 [0.72, 1.3] |
| bv4 | RO | 24 | 0.0394 [0.00128, 0.00182] | 0.0377 | -0.0383 ± 0.00192 | -0.0362 | -1.1 | 0.89 [0.7, 1.3] |
| bv4 | ZNE-L | 24 | 0.0183 [0.000166, 0.000531] | 0.0164 | 0.00422 ± 0.00372 | 0.000158 | 1.1 | 1.1 [0.87, 1.6] |
| qft4 | PEC | 24 | 0.0812 [0.00318, 0.0109] | 0.055 | -0.0506 ± 0.0132 | -0.0276 | -1.7 | 1.4 [1.1, 1.9] |
| qft4 | RAW | 24 | 0.181 [0.032, 0.0335] | 0.179 | -0.181 ± 0.00109 | -0.179 | -1.6 | 0.88 [0.68, 1.2] |
| qft4 | RO | 24 | 0.148 [0.0206, 0.023] | 0.145 | -0.147 ± 0.0022 | -0.145 | -0.93 | 1 [0.81, 1.5] |
| qft4 | ZNE-L | 24 | 0.0304 [0.000626, 0.00127] | 0.032 | -0.0269 ± 0.00297 | -0.0263 | -0.17 | 0.8 [0.63, 1.1] |

MSE intervals: bootstrap over blocks. z: (mean error − predicted bias) / SE. The SD ratio interval is the χ² interval of a normal sample's SD.

A confirmed contrast supports the simulated ranking for that circuit at this budget on this device and time window only. An unconfirmed one is not evidence that the protocols are equivalent.
