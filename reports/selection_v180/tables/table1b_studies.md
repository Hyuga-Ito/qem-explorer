| study | role | logical_circuits | executed | tasks_labeled | base_circuits_labeled | failed_tasks | note |
|---|---|---|---|---|---|---|---|
| audit18 | exploration | 8 | not in this run |  |  |  | 1.8.0 crossed circuit x preparation-root audit (scripts/root_audit.py): the circuits of the ten 1.7.0 deployment regression cases (two confirm17 circuits, reused after confirm17's once-only evaluation), earlier failure types and controls; declared before labeling, registered after its labeling had started |
| conf18 | confirmation | 29 | not in this run |  |  |  | 1.8.0 once-only confirmation, declared 2026-10-24 before any dev18 analysis |
| confirm17 | confirmation | 25 | yes | 58 | 29 | 0 | 1.7.0 confirmation set (declared before labeling) |
| dev18 | development | 44 | not in this run |  |  |  | 1.8.0 development study (expanded portfolio), declared 2026-10-24 before labeling |
| full | planned | 63 | no (planned) |  |  |  | 1.6.0 full configuration, never executed; superseded by full17 |
| full17 | planned | 242 | no (planned) |  |  |  | 1.7.0 larger study, planned (not executed) |
| mismatch | exploration | 10 | not in this run | 0 | 0 | 0 | mismatch study (base circuits of the pilot) |
| noisecomp17 | exploration | 3 | yes | 18 | 3 | 0 | noise-composition study (1.7.0), the regime circuits |
| noisecomp18 | exploration | 3 | not in this run |  |  |  | 1.8.0 noise-composition study (exposure-matched), configuration declared and committed 2026-10-24 before labeling; registered after its labeling had started (the registry entry lists circuits only) |
| pilot/dev | development | 40 | yes | 88 | 44 | 0 | 1.6.0 pilot development set; relabeled with fresh designs as pilot17/dev (1.7.0) |
| pilot/final | historical-final | 12 | not in this run | 0 | 0 | 0 | 1.6.0 final test, evaluated once in 1.6.0; part of the 1.7.0 development data (pilot17/final) |
| regimes | exploration | 3 | yes | 21 | 3 | 0 | controlled regime study (D23); relabeled as regimes17 |
| regimes18 | exploration | 3 | not in this run |  |  |  | 1.8.0 controlled regime study (the regimes17 slices relabeled with portfolio-1.8 and seed scheme 2), declared and registered 2026-10-24 before labeling |
