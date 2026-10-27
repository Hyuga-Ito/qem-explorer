#!/usr/bin/env bash
# Reproduce the 1.8.0 study into a NEW run directory (fake-device emulation only; nothing is submitted to hardware).
#
#   bash scripts/reproduce_v18.sh [RUN_DIR] [WORKERS]
#
# 1.9.0 (second review N3): a thin wrapper around the stage-aware pipeline (configs/pipelines/reproduce_v18.yaml).
# 1.8.0's command list called `freeze` unconditionally (an existing freeze exits 1, so a restarted run stopped there)
# and hid every confirmation failure behind `|| true`. The pipeline verifies once-only steps instead of redoing them,
# resumes labeling, stops on any failure and passes the run's own bundles to later stages. On Windows, run the same
# pipeline directly: python -m qemx.selection pipeline configs/pipelines/reproduce_v18.yaml --out runs\v18_repro
# Reproducing an inspected study reproduces it; it is not new confirmation evidence.
set -euo pipefail
R="${1:-runs/v18}"
W="${2:-2}"
PY="${PYTHON:-python}"
exec "$PY" -m qemx.selection pipeline configs/pipelines/reproduce_v18.yaml --out "$R" --workers "$W"
