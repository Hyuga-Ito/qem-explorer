# execution_audit_v190 — joint recommendation → design → execution audit (1.9.0; exploratory)

What a user would actually run, scored directly: for the eight circuits of the 1.8.0 root audit (1.7.0 regression
cases, earlier failure types and controls; inspected, but in neither the 1.8.0 bundles' training set nor the
confirmation) at budgets 10⁴, 3·10⁵ and 10⁷, under five deployment roots (the API's own design seeds and four
independent redraws), `recommend` returns a candidate and its verified design, the study's allocation spends the
budget, and the candidate is executed 600 times; SBS-B and RAW are executed the same way. The *fresh-design* columns
are the labels' target — the average risk over new training designs, the mean over the four preparation roots of
`reports/selection_v180/audit18`. Everything is simulated (emulated fake-device trials).

| file | content |
|---|---|
| `execution_audit_rows.csv` | one row per (policy, task, budget, root): returned candidate, design seed, executed risk, fresh-design risk, ledger |
| `execution_audit_summary.json` | contrasts against SBS-B (executed, fresh) and executed − fresh, base-circuit bootstrap; ledger check |

Reproduce: `python scripts/execution_audit.py --out runs/execution_audit --trials 600 --roots api,1,2,3,4` (needs the
1.8.0 trial-data archives for the fresh-design labels). Interpretation: `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §I.2 A.
