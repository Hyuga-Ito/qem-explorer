#!/usr/bin/env bash
# Reproduce the corrected pilot of QEM Explorer 1.7.0 (fake-device emulation only; nothing is submitted to hardware).
#
#   bash scripts/reproduce_pilot_17.sh [RUN_DIR] [WORKERS]
#
# Every step is resumable: rerunning the script skips finished label tasks and refuses a run whose inputs changed.
# The once-only steps (final, confirm) refuse a second evaluation. Measured on 2 cores for 1.7.0: see
# docs/selection/RESULTS.md (runtime). Requires the locked environment (requirements-lock.txt).
set -euo pipefail
R="${1:-runs/pilot17}"
W="${2:-2}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$(uname -s)" = "Linux" ]; then export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048; fi
PY="${PYTHON:-python}"
S="$PY -m qemx.selection"
C=configs/selection

$S plan $C/pilot17.yaml
$S label $C/pilot17.yaml --out "$R" --workers "$W"
$S label $C/pilot17.yaml --out "$R" --workers "$W" --final-set
$S label $C/regimes17.yaml --out "$R" --workers "$W" --subdir regimes
$S label $C/noisecomp17.yaml --out "$R" --workers "$W" --subdir noisecomp
$S mismatch $C/mismatch17.yaml --out "$R" --workers "$W"

# development analysis, freeze, then the deployed rows and the nested evaluation of the frozen configuration
$S analyze $C/pilot17.yaml --out "$R" --extras ablation,learning_curve,budget_generalization,abstention,interval_coverage,certificates,floor_sensitivity,design_diagnostics
$S freeze $C/pilot17.yaml --out "$R"
$S analyze $C/pilot17.yaml --out "$R" --splits "" --extras deployed,nested_selective
$S freeze $C/pilot17.yaml --out "$R" --complete
$S final $C/pilot17.yaml --out "$R" || true            # once only: a second run reports "already evaluated"

$S regimes $C/regimes17.yaml --out "$R" --skip-label
$S regimes $C/noisecomp17.yaml --out "$R" --skip-label --subdir noisecomp
$S mismatch $C/mismatch17.yaml --out "$R" --dev "$R/dev"
$S confirm $C/confirm17.yaml --out "$R" --dev-config $C/pilot17.yaml --workers "$W" || true   # once only

$PY scripts/design_study.py --out "$R/design_study" --designs 20 --trials 10
$S profile $C/pilot17.yaml --out "$R" --widths 3,4,5,6,7
$S figures $C/pilot17.yaml --out "$R"
$S report $C/pilot17.yaml --out "$R"
