"""Original versus expanded portfolio on identical tasks and labels (1.8.0; prompt section 3, last paragraph).

Out-of-fold (5 folds grouped by base circuit) on a portfolio-1.8 study: every policy is fitted and scored twice, once
choosing among the eight 1.6/1.7 candidates (portfolio-1.6, the labels' column subset) and once among all 17. The
differences separate

* the benefit of **having more candidates** — SBS-B(1.8) − SBS-B(1.6), and the attainable one, ALL-CANDIDATE(1.8) −
  ALL-CANDIDATE(1.6) (selection-half oracle, scored on the evaluation half);
* the benefit of **learning to select** them — MODEL(p) − SBS-B(p) within each portfolio p.

Primary metric: mean over base circuits of log10(MSE + 1e-6) on the evaluation half (each circuit's rows averaged
first); paired percentile bootstrap over base circuits. Arithmetic MSE and the declared tails beside it. Development
evidence (out-of-fold), not confirmation.

    python scripts/portfolio_comparison.py --labels runs/v18/dev --out runs/v18/portfolio
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _portable(path: str) -> str:
    """A path for the record: relative to the repository when inside it, else as given (never the machine's
    absolute layout of a directory outside the repository)."""
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run(labels: str, out: str, families=("ridge", "hgb"), set_name: str = "deployable18", k: int = 5,
        reps: int = 1000, seed: int = 0) -> dict:
    import pandas as pd
    from qemx import textio as TX
    from qemx.selection import dataset as DT
    from qemx.selection import evaluate as EV
    from qemx.selection import features as FE
    from qemx.selection import models as MD
    from qemx.selection import policies as PO
    from qemx.selection import protocols as PR
    full = DT.build(labels)
    groups = full.groups
    folds = MD.group_kfold(groups, k, seed)
    risk, chosen, portfolios = {}, {}, {}
    for pname, keys in (("1.6", PR.PORTFOLIO_16), ("1.8", tuple(p for p in PR.PORTFOLIO_18 if p in full.protocols))):
        portfolios[pname] = list(keys)
        ds = DT.with_protocols(full, list(keys))
        pols = [("SBS-B", lambda: PO.SBSBudget())] + [
            ("MODEL[%s]" % f, (lambda f=f: PO.Model(f, FE.SETS[set_name], set_name))) for f in families]
        for name, make in pols:
            r = np.full(len(ds.rows), np.nan)
            c = np.full(len(ds.rows), "", dtype=object)
            for tr, te in folds:
                pol = make().fit(ds, tr)
                ch, _ = pol.choose(ds, te)
                r[te] = [ds.Y_eval[i, j] if j >= 0 else np.nan for i, j in zip(te, ch)]   # −1: no choice
                c[te] = [ds.protocols[j] if j >= 0 else "" for j in ch]
            risk["%s|%s" % (name, pname)] = r
            chosen["%s|%s" % (name, pname)] = c
        ref = np.array([int(np.argmin(np.where(ds.F[i], ds.Y_sel[i], np.inf))) for i in range(len(ds.rows))])
        risk["ALL-CANDIDATE|%s" % pname] = np.array([ds.Y_eval[i, j] for i, j in enumerate(ref)])
        chosen["ALL-CANDIDATE|%s" % pname] = np.array([ds.protocols[j] for j in ref], dtype=object)
    raw = full.Y_eval[:, list(full.protocols).index("RAW")]      # RAW's evaluation-half risk on the same rows
    u = sorted(set(groups.tolist()))
    lg = {p: np.array([np.nanmean(np.log10(v[groups == g] + 1e-6)) for g in u]) for p, v in risk.items()}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(u), size=(int(reps), len(u)))

    def diff(a, b):
        d = lg[a] - lg[b]
        bs = d[picks].mean(1)
        return dict(a=a, b=b, diff=float(d.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                    ratio_geomean_mse_plus_floor=float(10 ** d.mean()))
    contrasts = dict(
        expansion_fixed_policy=diff("SBS-B|1.8", "SBS-B|1.6"),
        expansion_attainable=diff("ALL-CANDIDATE|1.8", "ALL-CANDIDATE|1.6"))
    for f in families:
        contrasts["learning_within_1.6_%s" % f] = diff("MODEL[%s]|1.6" % f, "SBS-B|1.6")
        contrasts["learning_within_1.8_%s" % f] = diff("MODEL[%s]|1.8" % f, "SBS-B|1.8")
        contrasts["total_%s" % f] = diff("MODEL[%s]|1.8" % f, "SBS-B|1.6")
    policies = {}
    for p, v in risk.items():
        policies[p] = dict(mean_log10=float(lg[p].mean()), mse=EV.wmean(v, groups), **EV.tail_metrics(v, groups),
                           p_severe_vs_raw=EV.wmean((v > EV.SEVERE_VS_RAW * raw).astype(float), groups),
                           chosen={kk: int(n) for kk, n in pd.Series(chosen[p]).value_counts().items()})
    res = dict(status="development, out-of-fold (%d folds grouped by base circuit); not confirmation" % k,
               labels=_portable(labels), rows=len(full.rows), base_circuits=len(u), feature_set=set_name,
               portfolios=portfolios,
               metric="mean over base circuits of log10(MSE + 1e-6), evaluation half; paired bootstrap %d" % reps,
               contrasts=contrasts, policies=policies)
    os.makedirs(out, exist_ok=True)
    TX.write_json(os.path.join(out, "portfolio_comparison.json"), res)
    pd.DataFrame([dict(policy=p, **{k_: v for k_, v in d.items() if k_ != "chosen"}) for p, d in policies.items()]
                 ).to_csv(os.path.join(out, "portfolio_policies.csv"), index=False, encoding="utf-8")
    pd.DataFrame([dict(contrast=c, **d) for c, d in contrasts.items()]).to_csv(
        os.path.join(out, "portfolio_contrasts.csv"), index=False, encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="deployable18")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.labels, a.out, set_name=a.set)["contrasts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
