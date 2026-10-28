"""Joint recommendation → verified design → allocation → execution audit of the 1.8.0 bundles (1.9.0; review R3).

The eight tasks of the root audit (scripts/root_audit.py) at its three budgets; deployment roots 'api' (the design
seeds the shipped API returns) and four independent redraws; both 1.8.0 bundles (simulator features with simulation
allowed — the D41 primary configuration — and the default deployable bundle), the SBS-B candidate the API verifies,
and RAW, each executed with its verified design (qemx.selection.execution_audit). Fresh-design labels: the mean over
the four preparation roots of reports/selection_v180/audit18 (when extracted). Exploratory: these tasks were inspected
in 1.8.0; the audit compares two risk targets, it is not a new confirmation.

    python scripts/execution_audit.py --out runs/execution_audit [--trials 600] [--roots api,1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def fresh_tables(study: str) -> dict:
    """{'any': {(task, protocol, B): (feasible, mean mse over roots, se of that mean)}} from the four audit roots."""
    import numpy as np
    from qemx.selection import replay as RP
    per = []
    for r in (18001, 18002, 18003, 18004):
        d = os.path.join(study, "audit18", "root_%d" % r)
        if not os.path.isdir(os.path.join(d, "tasks")):
            return {}
        per.append(RP.bound_population(d)["table"])
    keys = {}
    for t in per:
        for row in t.itertuples():
            keys.setdefault((row.task_id, row.protocol, int(row.B)), []).append(
                (RP._feasible(row.feasible), float(row.mse_eval)))
    out = {}
    for k, v in keys.items():
        ok = len(v) == 4 and all(f for f, _ in v) and all(math.isfinite(m) for _, m in v)
        m = np.array([x for _, x in v], float)
        out[k] = (ok, float(m.mean()) if ok else math.nan, float(m.std(ddof=1) / 2.0) if ok else math.nan)
    return {"any": out}


def main(argv=None) -> int:
    from qemx import textio as TX
    from qemx.selection import deploy as DP
    from qemx.selection import execution_audit as EA
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import root_audit as RA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--roots", default="api,1,2,3,4")
    ap.add_argument("--tasks", type=int, default=None, help="only the first K tasks (a quick check)")
    ap.add_argument("--study", default=os.path.join(ROOT, "reports", "selection_v180"))
    a = ap.parse_args(argv)
    roots = [r if r == "api" else int(r) for r in a.roots.split(",")]
    specs = RA.specs()[: a.tasks] if a.tasks else RA.specs()
    bundles = {"qemx-rec-1.8.0": (os.path.join(DP.BUNDLE_DIR, "qemx-rec-1.8.0"), True),
               "qemx-rec-1.8.0-deployable": (os.path.join(DP.BUNDLE_DIR, "qemx-rec-1.8.0-deployable"), False)}
    tabs = fresh_tables(a.study)
    rows = EA.run(specs, bundles, RA.BUDGETS, roots=roots, trials=a.trials, label_tables=tabs)
    s = EA.summarize(rows)
    s.update(fresh_labels="mean over the four preparation roots of %s/audit18" % os.path.relpath(a.study, ROOT)
             if tabs else "not available (trial arrays not extracted)", status="exploratory audit (inspected tasks)",
             command="python scripts/execution_audit.py --out %s --trials %d --roots %s" % (a.out, a.trials, a.roots))
    EA.write(rows, s, a.out)
    print(json.dumps(s, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
