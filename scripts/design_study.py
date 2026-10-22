"""Focused experiment B (1.7.0): learning-design variability and alternative training allocations of CDR and NIL.

Question: when a learning protocol fails badly on a task, is it the protocol in general, or particular training
designs, and does a different training allocation (more training circuits, a different share of B) change that?
The review's severe NIL cases (one of six fixed designs with mean loss ≈ 0.72 at B = 3·10⁷) motivate it.

For each task and protocol variant, K independent designs are drawn (fresh seeds, one pool), each run for m complete
trials at a few budgets; every design's diagnostics (training-label spread and range, rank resolved above shot noise,
singular values and condition of the exact training features, coefficient ℓ1 norm, fallback; and, research only,
whether the ideal value lies inside the training labels' range) are stored with its mean loss. Variants are complete
protocols of their own (registered under their own keys for this script only; the study's portfolio is unchanged).

    python scripts/design_study.py --out reports/selection_v170/design_study [--designs 30] [--trials 10]

Resumable: (task, variant) pairs already in OUT/designs.csv are kept and not rerun. Writes designs.csv (one row per
task, variant, design and budget) and summary.csv (per task, variant and budget:
mean loss, design-level standard error, between-design variance share, share of designs with a severe loss and the
diagnostics that separate them). Simulated observations only (fake-device emulation).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from qemx.selection import emulate as EM  # noqa: E402
from qemx.selection import labels as LB  # noqa: E402
from qemx.selection import protocols as PR  # noqa: E402
from qemx.selection import task as TK  # noqa: E402

TASKS = [  # (family, n, params, inputs, observable, device, layout): the review's NIL cases and two controls
    ("qft", 3, dict(m=1), "default", "default", "fake_fez", "best"),          # review: one design with loss 0.72
    ("qft", 5, {}, "default", "z_all", "fake_torino", "alt"),                # review: NIL MSE 0.38–0.65
    ("ising", 4, dict(steps=2), "default", "default", "fake_fez", "best"),
    ("hea", 5, dict(layers=3), "default", "default", "fake_fez", "best"),
    ("qaoa", 4, dict(p=1), "plus", "z_mid", "fake_fez", "alt"),
]
BUDGETS = (10000, 1000000, 30000000)
SEVERE = 0.01                  # a design whose mean squared error exceeds 0.01 (RMSE 0.1) is a severe failure


def variants() -> dict:
    """The protocol variants: the study's CDR and NIL and alternative training allocations of each."""
    base_nil = PR.PORTFOLIO["NIL"]
    base_cdr = PR.PORTFOLIO["CDR"]

    def nil(key, n_train=16, f=0.5, bare=False):
        p = dict(base_nil.params)
        p.update(n_train=n_train, include_bare=bare)
        return PR.Protocol(key, "nil", "tensored", tuple(sorted(p.items())), f, "NIL variant: %d training circuits, "
                           "%.0f %% of B on training%s" % (n_train, 100 * f, ", bare circuit included" if bare else ""))

    def cdr(key, n_train=16, f=0.25):
        p = dict(base_cdr.params)
        p.update(n_train=n_train)
        return PR.Protocol(key, "learning", "tensored", tuple(sorted(p.items())), f,
                           "CDR variant: %d training circuits, %.0f %% of B on training" % (n_train, 100 * f))
    return {"NIL": base_nil, "NIL-T32": nil("NIL-T32", n_train=32), "NIL-f25": nil("NIL-f25", f=0.25),
            "NIL-f75": nil("NIL-f75", f=0.75), "NIL-bare": nil("NIL-bare", bare=True),
            "CDR": base_cdr, "CDR-T32": cdr("CDR-T32", n_train=32), "CDR-f50": cdr("CDR-f50", f=0.5)}


def run(out: str, designs: int, trials: int, seed: int = 20261105, only=None) -> None:
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    vs = variants()
    PR.PORTFOLIO.update({k: v for k, v in vs.items() if k not in PR.PORTFOLIO})
    rows, done = [], set()
    prev = os.path.join(out, "designs.csv")
    if os.path.exists(prev):                     # resume: (task, variant) pairs already written are kept, not rerun
        old = pd.read_csv(prev, encoding="utf-8")
        rows = old.to_dict("records")
        done = set(zip(old.task_id, old.variant))
    for fam, n, prm, inp, obs, dev, lay in TASKS:
        spec = TK.make_spec(fam, n, inputs=inp, observable=obs, device=dev, layout=lay, **prm)
        todo = [k for k in vs if (not only or k in only) and (spec.task_id, k) not in done]
        if not todo:
            continue
        built = TK.build(spec)
        mu = float(built.ideal)
        for key in vs:
            if key not in todo:
                continue
            t0 = time.perf_counter()
            prep = EM.prepare(built, [key], seed=seed, designs={"mode": "fresh", "per_half": designs}, halves=(0,))
            recs = {r["index"]: r for r in prep.design_records.get(key, []) if r["half"] == 0}
            for B in BUDGETS:
                rng = LB.rng_for(seed, spec.task_id, B, key, 0)
                tr = EM.trials(prep, key, B, designs * trials, rng, half=0)
                if tr.status != "ok":
                    rows.append(dict(task=spec.label, task_id=spec.task_id, variant=key, B=B, status=tr.status,
                                     reason=tr.allocation.get("reason", "")))
                    continue
                x = np.where(np.isfinite(tr.estimates), tr.estimates, 0.0)
                for k in range(designs):
                    m = tr.design == k
                    r = recs.get(k, {})
                    dg = r.get("diagnostics") or {}
                    rows.append(dict(task=spec.label, task_id=spec.task_id, variant=key, B=B, status="ok", design=k,
                                     seed=r.get("seed"), fallback=r.get("fallback", ""), trials=int(m.sum()),
                                     mean_loss=float(np.mean((x[m] - mu) ** 2)), mean_estimate=float(np.mean(x[m])),
                                     ideal=mu, fallback_rate=float(tr.flags["learn_fallback"][m].mean()),
                                     s_tr=int(tr.allocation.get("s_tr", 0)),
                                     **{"diag_" + kk: vv for kk, vv in dg.items() if kk != "warning"}))
            print("%s %s %.0fs" % (spec.label[:40], key, time.perf_counter() - t0), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, "designs.csv"), index=False, encoding="utf-8")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "designs.csv"), index=False, encoding="utf-8")
    summarize(d, out)


def summarize(d, out: str) -> None:
    """summary.csv of a designs table (see module text)."""
    import pandas as pd
    summ = []
    ok = d[d.status == "ok"]
    for (task, v, B), e in ok.groupby(["task", "variant", "B"]):
        L = e.mean_loss.to_numpy()
        K = len(L)
        sev = L > SEVERE
        row = dict(task=task, variant=v, B=int(B), designs=K, mse=float(L.mean()),
                   se_design=float(L.std(ddof=1) / math.sqrt(K)) if K > 1 else math.nan,
                   median_design_loss=float(np.median(L)), max_design_loss=float(L.max()),
                   severe_share=float(sev.mean()), severe_contribution=float(L[sev].sum() / L.sum()) if L.sum() > 0
                   else math.nan, fallback_share=float((e.fallback.fillna("").astype(str) != "").mean()))
        for col in ("diag_rank", "diag_condition", "diag_l1", "diag_spread", "diag_oracle_ideal_in_label_range"):
            if col in e:
                a, b = e[col][sev], e[col][~sev]
                row[col + "_severe"] = float(np.nanmean(a.astype(float))) if len(a) else math.nan
                row[col + "_other"] = float(np.nanmean(b.astype(float))) if len(b) else math.nan
        summ.append(row)
    pd.DataFrame(summ).to_csv(os.path.join(out, "summary.csv"), index=False, encoding="utf-8")


def main(argv=None) -> int:
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--only", default=None, help="comma-separated variants")
    ap.add_argument("--merge", nargs="+", default=None, help="merge the designs.csv of these output directories "
                    "(parallel runs over disjoint variants) into --out and summarize")
    a = ap.parse_args(argv)
    if a.merge:
        import pandas as pd
        d = pd.concat([pd.read_csv(os.path.join(m, "designs.csv"), encoding="utf-8") for m in a.merge],
                      ignore_index=True)
        os.makedirs(a.out, exist_ok=True)
        d.to_csv(os.path.join(a.out, "designs.csv"), index=False, encoding="utf-8")
        summarize(d, a.out)
        return 0
    run(a.out, a.designs, a.trials, only=a.only.split(",") if a.only else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
