# Evidence for the 1.5.0 report (docs/TRACEABILITY_1.5.md)

| file | what |
|---|---|
| `baseline/pytest_1.4.2_baseline.log` | full suite on the audited 1.4.2 tree (788 passed, 1 failed: the NIL round-off assertion) |
| `baseline_focused.json`, `reproduce_audit.py`, `reproduce_export.py`, `reproduce_ui.py`, `validate_readout_fix.py` | the QA evidence's reproduction scripts (from `QEM_Explorer_1.4.2_QA_Evidence.zip`) and their output on the audited tree, before any change |
| `validation_learning.json` | `scripts/validate_learning_moments.py` on 1.5: three audit cases × 6 000 independent experiments (all within the declared criteria) |
| `profile_before.json`, `profile_after.json` | `scripts/profile_workflows.py` on 1.4.2 and 1.5 (same machine, fresh processes, 3 runs) |
| `browser_qa.json`, `browser_qa.py`, `screenshots/` | Chromium (Playwright) review of 10 pages at 1440 × 900 and 390 × 844 |
| `transfer_predictions/` | simulated predictions (fake_fez) for the hardware-transfer subset of `docs/hardware_preflight.md` |
| `pytest_full.log` | the full test suite on the final 1.5.0 tree: 904 passed, 0 failed (slow tests included) |

The emulated discrepancy rehearsal is in `reports/rehearsal_v3/`, labeled EMULATION. The offline pilot figure report is in `reports/pilot/`. No hardware data exist in this release.
