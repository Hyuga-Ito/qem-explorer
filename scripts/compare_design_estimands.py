"""Old against corrected labels of the same tasks (1.7.0, review R1): the 1.6.0 pilot (six learning designs shared by
both label halves) and the corrected pilot (fresh, independent designs per half) were labeled from the same tasks and
seeds, so their protocols without a training design must agree trial for trial and CDR/NIL show the effect of the
correction.

    python scripts/compare_design_estimands.py OLD_LABELS_DIR NEW_LABELS_DIR --out OUT_DIR

Writes ``identity.csv`` (per protocol: tasks whose trials are identical), ``learning.csv`` (per task, learning
protocol and budget: evaluation-half MSE and its standard error under both estimands — trial-level for the 1.6.0
labels as reported then, design-level for the corrected labels — and the ratio of the two SEs) and ``summary.csv``
(per protocol and budget: medians and the share of tasks whose 1.6.0 MSE lies outside the corrected label's
±1.96 SE interval). Simulated observations only.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from qemx import textio as TX  # noqa: E402
from qemx.selection import labels as LB  # noqa: E402
from qemx.selection import protocols as PR  # noqa: E402


def compare(old: str, new: str):
    import pandas as pd
    ids = sorted(set(LB.task_ids(old)) & set(LB.task_ids(new)))
    ident, rows = {}, []
    for t in ids:
        ro, ao = LB.load_task(old, t)
        rn, an = LB.load_task(new, t)
        if ao is None or an is None:
            continue
        if ro["protocols"] != rn["protocols"] or ro["budgets"] != rn["budgets"]:
            raise ValueError("%s: different protocols or budgets" % t)
        mu = rn["ideal"]
        plan_o, plan_n = LB.plan_of(ro), LB.plan_of(rn)
        for i, p in enumerate(ro["protocols"]):
            a, b = ao["est"][:, i], an["est"][:, i]
            ident.setdefault(p, []).append(bool(np.array_equal(np.nan_to_num(a, nan=9e9), np.nan_to_num(b, nan=9e9))))
            if not PR.PORTFOLIO[p].learning:
                continue
            for j, B in enumerate(rn["budgets"]):
                al = rn["allocations"][i * len(rn["budgets"]) + j]
                if not al["feasible"]:
                    continue
                lo = (np.where(np.isfinite(a[1, j]), a[1, j], 0.0) - mu) ** 2
                ln = (np.where(np.isfinite(b[1, j]), b[1, j], 0.0) - mu) ** 2
                so = LB.design_stats(lo, ao["design"][1, i, j] if "design" in ao else np.full(len(lo), -1))
                sn = LB.design_stats(ln, an["design"][1, i, j])
                se_new = LB.risk_se(sn, plan_n, True)
                rows.append(dict(task_id=t, family=rn["spec"]["family"], n=rn["spec"]["n"], protocol=p, B=int(B),
                                 mse_old=float(lo.mean()), se_old_reported=float(so["se_trial"]),
                                 se_old_design=float(so["se_design"]), designs_old=int(so["designs"]),
                                 mse_new=float(ln.mean()), se_new=float(se_new), designs_new=int(sn["designs"]),
                                 plan_old=plan_o.get("mode", "fixed-shared"), plan_new=plan_n.get("mode")))
    d = pd.DataFrame(rows)
    d["se_ratio_new_over_old"] = d.se_new / d.se_old_reported
    d["log10_mse_ratio"] = np.log10((d.mse_new + 1e-6) / (d.mse_old + 1e-6))
    d["old_outside_new_interval"] = (d.mse_old - d.mse_new).abs() > 1.96 * d.se_new
    summ = d.groupby(["protocol", "B"]).agg(tasks=("task_id", "nunique"), median_mse_old=("mse_old", "median"),
                                            median_mse_new=("mse_new", "median"),
                                            mean_mse_old=("mse_old", "mean"), mean_mse_new=("mse_new", "mean"),
                                            median_se_ratio=("se_ratio_new_over_old", "median"),
                                            median_abs_log10_mse_ratio=("log10_mse_ratio",
                                                                        lambda x: float(np.median(np.abs(x)))),
                                            share_old_outside_new_interval=("old_outside_new_interval", "mean"))
    idt = pd.DataFrame([dict(protocol=p, tasks=len(v), identical=int(sum(v))) for p, v in ident.items()])
    return idt, d, summ.reset_index()


def main(argv=None) -> int:
    TX.utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    idt, d, s = compare(a.old, a.new)
    idt.to_csv(os.path.join(a.out, "identity.csv"), index=False, encoding="utf-8")
    d.to_csv(os.path.join(a.out, "learning.csv"), index=False, encoding="utf-8")
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False, encoding="utf-8")
    print(idt.to_string(index=False))
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
