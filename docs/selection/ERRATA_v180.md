# Errata and qualifications of the 1.8.0 study (written in 1.9.0)

The 1.8.0 study (`reports/selection_v180`) is kept **unchanged** as the historical record: its trial records, labels,
freeze, confirmation and the files generated from them are byte-identical to the 1.8.0 release. The corrections below
are made in the generators (so every later regeneration is correct) and, where a corrected file is useful, written
**beside** the historical one (`reports/corrections_v190/`). No number of the study changed: the saved labels and
arrays were re-verified (`tests/test_trial_data_v180.py`: 217 arrays against the study's checksum manifest; cached
tables equal the tables derived from the arrays; the published primary contrasts recomputed from the arrays to 10⁻¹³).

| item | what the 1.8.0 file says | correct statement | where corrected |
|---|---|---|---|
| E1 primary/secondary (review R2, confirmed) | `figures/supp_deployment_policy.md`: the methods box calls the simulator-feature bundle at allocation level "a secondary analysis"; the interpretation calls the *default* bundle's result confirmatory | Primary contrast (D41): the frozen simulator-feature configuration at allocation-level preflight against SBS-B, −0.144 [−0.231, −0.076]; secondary: the default deployable bundle, −0.084 [−0.179, −0.002] | generator `qemx/selection/plots18.py` (`order_bundles`, caption, methods box); regenerated note `reports/corrections_v190/figures/supp_deployment_policy.md` |
| E2 timing of D41 | "confirmatory for the frozen configuration" without the timing | D41's primary estimand was specified **after** the development results and **before** any confirmation result; it is not a pre-development registration. The confirmation was evaluated once | same generator; `docs/selection/RESULTS.md` already stated the timing |
| E3 metric label | axes and some captions say "log10 MSE" | the metric is log10(MSE + 10⁻⁶) averaged per base circuit; 10^Δ is a factor on the geometric mean of MSE + 10⁻⁶, not on the MSE | generators `plots.py`, `plots18.py`, `report.py` |
| E4 risk target of the replay | "risk of the returned candidate" | the replay scores the returned candidate's **fresh-design average risk** R^fresh (the evaluation half averages independently drawn training designs), not the risk conditional on the one design the API returns for execution. The 1.9.0 joint audit (`reports/execution_audit_v190`) measures the latter directly on eight inspected circuits: the difference of the policy contrasts is −0.024 [−0.047, +0.000] (primary) and +0.005 [−0.036, +0.061] (default bundle) — no material difference there, on few learning rows | `qemx/selection/replay.py` (provenance `estimand`), captions, `RESEARCH_FINDINGS_AND_NEXT_STEPS.md` |
| E5 the 47 unverified-component rows | — (correct) | the caption already separates 33 accepted alternatives and 14 out-of-distribution fallbacks; the 14 may also have an unverified component candidate (not mutually exclusive). Not an error; the 1.9.0 generator names the bundle these counts belong to | — |
| E6 bootstrap-seed sensitivity of the default bundle | "fragile across bootstrap seeds (59 % of seeds)" | the interval's upper end is ≈ 0 (≈ +0.00014 with 200 000 resamples); seed sensitivity is a property of the interval estimate, **not** a probability that the effect exists | `RESEARCH_FINDINGS_AND_NEXT_STEPS.md` |

Software defects found in 1.9.0 that did **not** affect the 1.8.0 numbers (the complete study was labeled in one
sequence, no task failed, and its replay used a complete bound population): the incomplete-run exit codes (N1),
unenforced whole-run caps (N2), the reproduction script (N3), the replay's file enumeration and cached tables (N4), the
trial-data CI job (N5), unvalidated nested YAML keys (N6) and the state/UI recommendation path (R1, N8, R8). They matter
for new local runs and for the Playground; see `docs/ISSUE_EVIDENCE_MATRIX.md`.
