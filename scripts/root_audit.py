"""Crossed circuit × preparation-root audit (1.8.0; prompt sections 2 and 7, stage 2; exploratory).

Eight base circuits — the two circuits of the ten 1.7.0 deployment regression cases, earlier failure types (deep
learning-protocol designs, a small-signal observable) and controls — are labeled under four independently namespaced
preparation roots (master seeds; seed scheme 2: every root draws its own learning designs and trial streams) at three
budgets with the expanded portfolio. The roots separate three mechanisms the recommender's errors can come from:

* **candidate training instability** — the between-root spread of a candidate's risk label against the spread its
  design-level standard error predicts (ratio ≈ 1: the SE is honest; ≫ 1: rare designs dominate, the SE is not);
* **model discrimination** — whether the best candidate of a (task, budget) is the same under every root (if it
  changes, no model can be "right" at the resolution of these labels); computed with the 17 candidates and, with the
  primary portfolio held fixed, with the original eight (portfolio-1.6, a subset of the same labels);
* **execution-contract mismatch** — whether the design-level preflight (qemx.selection.preflight) agrees with each
  root's own allocation design.

Usage::

    python scripts/root_audit.py label   --out runs/v18/audit18 --workers 2
    python scripts/root_audit.py analyze --out runs/v18/audit18 [--bundle qemx/selection/bundles/qemx-rec-1.8.0-deployable]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ROOTS = (18001, 18002, 18003, 18004)
BUDGETS = (10000, 300000, 10000000)
R = 40
DESIGNS = {"mode": "fresh", "per_half": 6, "seed_scheme": 2, "crn": "task"}
TASKS = (  # (family, n, params, observable, device, layout, why)
    ("ghz", 6, dict(theta=1.3), "z_mid", "fake_fez", "alt", "regression case 2d8e84ea (degenerate CDR/NIL design)"),
    ("ghz", 5, dict(theta=0.8), "z_mid", "fake_fez", "alt", "regression case f975bae8 (degenerate CDR/NIL design)"),
    ("hea", 4, dict(layers=3, angle_seed=1), "default", "fake_fez", "best", "deep learning designs (1.7 pilot)"),
    ("qaoa", 5, dict(p=2, angle_seed=1), "default", "fake_torino", "best", "ZNE-E tail failures (1.7 pilot)"),
    ("ising", 5, dict(steps=3), "zz_mid", "fake_kyiv", "best", "deep Trotter, two-point observable"),
    ("brickwork", 4, dict(layers=4, angle_seed=1), "z_all", "fake_fez", "best", "small signal (z_all)"),
    ("ghz", 4, dict(theta=0.65, basis="Z"), "z_mid", "fake_fez", "best", "declared symmetry (SV applicable)"),
    ("qft", 4, dict(), "default", "fake_fez", "best", "control: known answer, Clifford-free"),
)


def specs():
    from qemx.selection import task as TK
    return [TK.make_spec(f, n, observable=o, device=d, layout=lay, **p) for f, n, p, o, d, lay, _ in TASKS]


def cfg_of(root: int) -> dict:
    from qemx.selection import protocols as PR
    return dict(name="audit18", master_seed=int(root), R=R, budgets=list(BUDGETS), protocols=list(PR.PORTFOLIO_18),
                learning_designs=dict(DESIGNS), final_test_fraction=0.0)


def label(out: str, workers: int, retry_failed: bool = False) -> int:
    """Label every root (resumable). 1.9.0 (second review N1): labeled tasks are skipped in this process, failed ones
    are kept and retried only with ``retry_failed``, and the exit code is 1 unless every root is complete."""
    from qemx.selection import binding as BD
    from qemx.selection import integrity as IN
    from qemx.selection import run as RN
    sp = specs()
    rc = 0
    for root in ROOTS:
        target = os.path.join(out, "root_%d" % root)
        cfg = cfg_of(root)
        with IN.RunLock(target, "root audit label"):
            run = BD.bind(target, cfg, sp, sp, [], "audit")
            before = IN.completion(target)
            if before["corrupted"]:
                print("root %d: %s — nothing labeled" % (root, IN.describe(before)))
                rc = 1
                continue
            failed = {t for v in before["failed"].values() for t in v}
            work = [s for s in sp if s.task_id not in set(before["completed_ids"])
                    and (retry_failed or s.task_id not in failed)]
            RN._supervised_label(work, target, list(BUDGETS), cfg["protocols"], R, root, DESIGNS,
                                 {"task_seconds": 3600, "memory_gb": 3}, workers, print, run_id=run["identity_digest"],
                                 session=run["sessions"][-1]["started"])
            after = IN.completion(target)
            print("root %d: %s" % (root, IN.describe(after)))
            if not after["complete"]:
                rc = 1
    return rc


def analyze(out: str, bundle: str | None) -> dict:
    import pandas as pd
    from qemx import textio as TX
    from qemx.selection import emulate as EM
    from qemx.selection import labels as LB
    from qemx.selection import preflight as PF
    from qemx.selection import protocols as PR
    from qemx.selection import task as TK
    from qemx.selection import integrity as IN
    tabs = []
    for root in ROOTS:
        c = IN.require_complete(os.path.join(out, "root_%d" % root), "the root audit")
        t = LB.summarize(os.path.join(out, "root_%d" % root), task_ids=c["completed_ids"])
        t["root"] = root
        tabs.append(t)
    T = pd.concat(tabs, ignore_index=True)
    rows = []
    for (tid, k, B), g in T.groupby(["task_id", "protocol", "B"]):
        f = g[g.feasible.astype(bool)]
        if len(f) < 2:
            rows.append(dict(task_id=tid, protocol=k, B=int(B), roots_feasible=len(f)))
            continue
        m = f.mse_eval.to_numpy(float)
        se2 = np.nanmean(f.se_mse_eval.to_numpy(float) ** 2)
        lm = np.log10(m + 1e-6)
        rows.append(dict(task_id=tid, protocol=k, B=int(B), roots_feasible=len(f), mean_mse=float(m.mean()),
                         between_root_var=float(m.var(ddof=1)), mean_label_se2=float(se2),
                         var_ratio=float(m.var(ddof=1) / se2) if se2 > 0 else math.nan,
                         log10_spread=float(lm.max() - lm.min())))
    per = pd.DataFrame(rows)
    rank = []
    # ranking stability with the expanded portfolio and, as the review asked for the first audit, with the original
    # portfolio held fixed (its eight candidates are a subset of the same labels)
    # a root's winner is identified on its selection half and scored on evaluation halves (the study's convention,
    # evaluate.scores); root a's winner judged by root b: log10 of b's evaluation risk of a's winner over b's
    # evaluation risk of b's own winner. A winner infeasible under another root is counted, not dropped.
    feas = T.feasible.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    for pname, keys in (("1.8", PR.PORTFOLIO_18), ("1.6", PR.PORTFOLIO_16)):
        Tp = T[T.protocol.isin(keys) & feas & np.isfinite(T.mse_eval.astype(float))
               & np.isfinite(T.mse_sel.astype(float))]
        ev = {(r.task_id, int(r.B), r.root, r.protocol): float(r.mse_eval) for r in Tp.itertuples()}
        best = (Tp.sort_values("mse_sel").groupby(["task_id", "B", "root"]).head(1)
                [["task_id", "B", "root", "protocol"]])
        for (tid, B), g in best.groupby(["task_id", "B"]):
            winners = sorted(set(g.protocol))
            cross, infeasible = [], 0
            for a in g.itertuples():
                for b_ in g.itertuples():
                    if a.root == b_.root:
                        continue
                    r_ab = ev.get((tid, int(B), b_.root, a.protocol))
                    r_bb = ev.get((tid, int(B), b_.root, b_.protocol))
                    if r_ab is None:
                        infeasible += 1
                        continue
                    cross.append(math.log10((r_ab + 1e-6) / (r_bb + 1e-6)))
            rank.append(dict(portfolio=pname, task_id=tid, B=int(B), roots=len(g), distinct_winners=len(winners),
                             winners=";".join(winners),
                             mean_cross_root_log_regret=float(np.mean(cross)) if cross else math.nan,
                             max_cross_root_log_regret=float(np.max(cross)) if cross else math.nan,
                             pairs=len(cross), pairs_winner_infeasible_under_other_root=infeasible))
    rank = pd.DataFrame(rank)
    # execution contract: design-level preflight with each root's allocation seed vs the labels' allocation
    contract = []
    for s in specs():
        b = TK.build(s)
        for root in ROOTS:
            rec = LB.load_record(os.path.join(out, "root_%d" % root), s.task_id)
            if rec.get("status") != "ok":
                continue
            for a in rec["allocations"]:
                pr = PR.PORTFOLIO[a["protocol"]]
                if not pr.learning:
                    continue
                seed = EM.allocation_seed(rec, pr.key)        # the seed the labels' allocation was computed with
                p = PF.design(b.compiled, b.observable, pr.key, a["B"], supplied=s.supplied_backend, seed=seed)
                contract.append(dict(task_id=s.task_id, root=root, protocol=pr.key, B=a["B"], preflight=p.status,
                                     labels_feasible=bool(a["feasible"]),
                                     agree=(p.status == PF.VERIFIED) == bool(a["feasible"])))
    contract = pd.DataFrame(contract)
    lab = {s.task_id: s.label for s in specs()}
    for df in (per, rank):
        df["task"] = df.task_id.map(lab)
    ad = os.path.join(out, "analysis")
    os.makedirs(ad, exist_ok=True)
    per.to_csv(os.path.join(ad, "label_stability.csv"), index=False, encoding="utf-8")
    rank.to_csv(os.path.join(ad, "ranking_stability.csv"), index=False, encoding="utf-8")
    contract.to_csv(os.path.join(ad, "contract.csv"), index=False, encoding="utf-8")
    ok = per.dropna(subset=["var_ratio"]) if "var_ratio" in per else per
    summary = dict(
        status="exploratory (stage-2 audit; not confirmatory)", roots=list(ROOTS), budgets=list(BUDGETS),
        tasks=len(TASKS), rows=len(per),
        var_ratio_median_by_protocol={k: float(g.var_ratio.median()) for k, g in ok.groupby("protocol")},
        var_ratio_q90_by_protocol={k: float(g.var_ratio.quantile(0.9)) for k, g in ok.groupby("protocol")},
        rows_var_ratio_gt_4=int((ok.var_ratio > 4).sum()), rows_with_ratio=int(len(ok)),
        ranking_cells=int((rank.portfolio == "1.8").sum()),
        cells_with_one_winner=int(((rank.portfolio == "1.8") & (rank.distinct_winners == 1)).sum()),
        mean_cross_root_log_regret=float(rank[rank.portfolio == "1.8"].mean_cross_root_log_regret.mean()),
        pairs_winner_infeasible_under_other_root=int(rank[rank.portfolio == "1.8"]
                                                     .pairs_winner_infeasible_under_other_root.sum()),
        portfolio16=dict(ranking_cells=int((rank.portfolio == "1.6").sum()),
                         cells_with_one_winner=int(((rank.portfolio == "1.6") & (rank.distinct_winners == 1)).sum()),
                         mean_cross_root_log_regret=float(rank[rank.portfolio == "1.6"]
                                                          .mean_cross_root_log_regret.mean())),
        contract_rows=len(contract), contract_agreement=float(contract.agree.mean()) if len(contract) else math.nan)
    if bundle:
        from qemx.selection import replay as RP
        pol = {}
        for root in ROOTS:
            rows_ = RP.replay(os.path.join(out, "root_%d" % root), bundle)
            pol[root] = dict(mean_log10=float(np.mean([math.log10(r["risk"] + 1e-6) for r in rows_ if r["scored"]])),
                             unscored=sum(1 for r in rows_ if not r["scored"]),
                             choices=sorted({r["choice"] for r in rows_}))
        summary["policy_by_root"] = pol
    TX.write_json(os.path.join(ad, "audit_summary.json"), summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["label", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args(argv)
    if a.action == "label":
        return label(a.out, a.workers, a.retry_failed)
    else:
        print(json.dumps(analyze(a.out, a.bundle), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
