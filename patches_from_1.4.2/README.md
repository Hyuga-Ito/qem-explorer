# Reviewable patches: 1.4.2 (audited) → 1.5.0

These four commits turn the audited archive `qem-explorer-1.4.2-code.zip` (SHA-256 `26da74f9…5b`) into this tree:

1. `0001` repairs the statistics, validation, export and guide-state defects of the QA report.
2. `0002` covers input/output semantics, cited presets, one experiment model (UI/YAML/CLI) and the discrepancy measures.
3. `0003` adds the main-study framework, the recommender, the discrepancy rehearsal and the 600-second campaign ledger.
4. `0004` adds the research-area pages, experiment examples, the hardware preflight plan, docs and evidence.

To review a patch, read it as text. To apply them to a clean copy of 1.4.2:

```
git init qem-142 && cd qem-142 && <unzip qem-explorer-1.4.2-code.zip here> && git add -A && git commit -m base
git am --keep-cr ../patches_from_1.4.2/*.patch
```
