# Results of QEM Explorer 1.4.1 (sandbox runs)

Every folder holds the tables (`tables/*.csv`, `report.txt`, `report.md`) and figures that the shipped scripts write.
**All divergence data are emulated** — a fake device with a known, deliberately injected error — because no IBM device
was reachable from the sandbox. Each figure says so in its title.

| folder | what | made by |
|---|---|---|
| `folding/` | global against local folding, stages F1–F4 (fake_fez, equal budget) | `scripts/dirA_zne_folding.py` |
| `fractional/` | fractional ZNE, stages F5 (λ = 1, 4/3, 5/3, 2; five rules) and F6 (fold subsets, subset averaging) | `scripts/dirA_zne_folding.py` |
| `divergence_emulated_null/` | v1 suite (59 circuits), truth = the depolarizing model: the null test | `dirA_divergence.py analyze` |
| `divergence_emulated_crosstalk/` | v1 suite, truth = fake_fez + ZZ crosstalk 0.05 rad | `dirA_divergence.py analyze` |
| `divergence_pilot_emulated/` | pilot suite (29 benchmarks × 3 layouts) with five ground truths: none, gate errors ×1.6, T2 = 30 µs, ZZ crosstalk, a defect on qubit 110; the diagnostic matrix | `analyze`, `recovery` |
| `divergence_v2_emulated/` | the v2 manifest (52 benchmarks × 3 layouts) with four ground truths and its diagnostic matrix | `analyze`, `recovery` |

Read with the release notes (`docs/v14/release_notes_1.4.1.md`, §2, §3, §6, §7) and the research plan
(`docs/v14/research_plan_divergence.md`).
