"""Exploratory, post-inspection replay of the 1.8.0 application policy on the **1.7.0** confirmation set (1.8.0;
review finding F1). Not confirmatory: the 1.7.0 confirmation was inspected (and its ten deployment regression cases
were derived from it) before this policy was written. It answers one question — does the 1.8.0 policy (shared
preflight contract, lazy resolution in the model's order, verified fallback) remove the ten regression cases and
keep every row scored — using the 1.7.0 bundles, whose configurations were frozen on the 1.7.0 development set.

    python scripts/replay_v170_confirmation.py --confirm reports/selection_v170/confirm --out runs/replay_v170

Three replays: the deployable bundle with each experiment's own design seed (as the application) and with the
labels' allocation seed (the design the benchmark's feasibility was decided on), and the simulator-feature bundle at
the allocation preflight level.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RUNS = (("deployable_deployseed", "qemx-rec-1.7.0-deployable", False, "deployment"),
        ("deployable_studyseed", "qemx-rec-1.7.0-deployable", False, "study"),
        ("simulated_alloc", "qemx-rec-1.7.0", True, "study"))


def main(argv=None) -> int:
    from qemx import textio as TX
    from qemx.selection import deploy as DP
    from qemx.selection import replay as RP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--confirm", default=os.path.join(ROOT, "reports", "selection_v170", "confirm"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ch = list(csv.DictReader(open(os.path.join(a.confirm, "choices_confirm.csv"), encoding="utf-8")))
    tab = RP._labels(a.confirm)
    risk = {(r.task_id, r.protocol, int(r.B)): float(r.mse_eval) for r in tab.itertuples()}
    key = {(r["task_id"], int(r["B"])): r for r in ch}
    sbsb = {k: risk.get((k[0], r["SBS-B|choice"], k[1]), float("nan")) for k, r in key.items()}
    out = {}
    for tag, bundle, sim, seeds in RUNS:
        t0 = time.time()
        path = os.path.join(DP.BUNDLE_DIR, bundle)
        excluded = []
        rows = RP.replay(a.confirm, path, allow_simulation=sim, design_seeds=seeds, excluded=excluded)
        s = RP.summarize(rows, sbsb)
        col = "%s|choice" % DP.load_bundle(path).meta["configuration"]
        s["model_component_benchmark"] = RP.summarize(RP.component_rows(rows, key, col, risk), sbsb)
        s["model_component_configuration"] = DP.load_bundle(path).meta["configuration"]
        s["excluded_label_tasks"] = excluded
        s.update(bundle=bundle, allow_simulation=sim, design_seeds=seeds, policy=DP.POLICY,
                 status_note="exploratory, post-inspection replay on the 1.7.0 confirmation (not confirmatory)",
                 seconds_total=time.time() - t0)
        RP.write(rows, s, a.out, tag)
        out[tag] = {k: s.get(k) for k in ("rows", "unscored", "status", "accepted", "with_skipped_candidates",
                                          "diff", "diff_lo", "diff_hi", "policy_mse", "baseline_mse")}
    TX.write_json(os.path.join(a.out, "replay_summary.json"), out)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
