# Main-study pilot — primary records

These are the **primary records** of the offline main-study pilot (manifest `manifests/direction_A_v2_pilot.yaml`,
SHA-256 `9aaaf73b…1d293`): one JSON record per cell, 75 cells from 25 base circuits on the `fake_fez` device model at
three noise scales. They are simulation results, not hardware data.

| item | value |
|---|---|
| produced | 2026-10-17 (UTC), offline, with the 1.5.0 development tree |
| record schema | `qemx.phase.cell/3` (the 1.5.1 code reads it; it writes `/4`) |
| environment | Python 3.13.16, Qiskit 2.6.0, Qiskit Aer 0.17.2, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 (see `run.json`) |
| version string in the records | `qemx 1.4.2` — the development tree had not yet bumped its version number when the pilot ran; the records are the ones the 1.5.0 report was built from |
| integrity | `SHA256SUMS` (75 records and the manifest) |

`run.json` is the run file with two fields sanitized for publication (`manifest`, `argv` held absolute paths of the
development workspace; see its `redacted` field). Nothing else was changed. The derived tables are not stored here:
they are recomputed from the records.

## Reproducing the bundled report

```bash
python scripts/reproduce_pilot_report.py --out pilot_report --compare reports/pilot
```

This verifies the checksums, copies the records to a temporary directory, derives the tables
(`qemx.phase.run collect`) and writes the report (`qemx.study report`), then checks that the regenerated Figure 3
table and the regime table equal those in `reports/pilot/`. No simulation runs. Figure 5 (the discrepancy
rehearsal) needs that rehearsal's raw analysis directory, which is not bundled (`--rehearsal DIR`).

## What changed with 1.5.1

The records are unchanged. The 1.5.1 analysis reads them with the corrected reporting:

- improvement shares with explicit denominators (all planned cells and feasible cells);
- feature schema `qemx.features/2` (descriptive names; values identical);
- recommender evaluation with group-cross-validated conformal intervals, the full policy, harmful choices, regret tails
  and paired comparisons;
- the Monte-Carlo error of the estimated RMSEs and the stability of the regime labels.

None of the pilot's methods is a target-nonlinear estimator (exponential ZNE, symmetry verification), so the 1.5.1
budget-dependent moments do not change any of its numbers.
