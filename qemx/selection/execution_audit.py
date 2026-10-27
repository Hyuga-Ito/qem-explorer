r"""Joint recommendation → verified design → allocation → execution audit (1.9.0; independent review R3, prompt §3).

The study's labels estimate, for a learning candidate a, the *fresh-design* risk

    R_a^fresh(T, B) = E_{D,ξ}[(μ̂_a(T, B; D, ξ) − μ_T)²]

(the evaluation half averages independently drawn training designs D). The deployed API returns *one* verified design
D* (its method seed) and execution with it has the conditional risk

    R_a(T, B | D*) = E_ξ[(μ̂_a(T, B; D*, ξ) − μ_T)²].

A policy that recommends, verifies the design by preflight and executes it has an end-to-end risk that conditions on
the preflight's outcome; substituting R^fresh for it is a defined benchmark, not automatically an execution estimate.
This audit measures the end-to-end quantity directly on a small set of tasks:

* for each task and each **deployment root** (``api``: the design seeds the shipped API returns; integers: independent
  deployments that redraw every learning candidate's design, ``deployment_seed``), the recommendation is computed with
  those seeds (so the preflight verifies exactly the designs that are executed) for the given bundles, together with
  the SBS-B fallback candidate the API verifies and RAW;
* each returned candidate is **executed** with its verified design and the allocation the API's rule gives at B
  (``emulate.prepare(..., seed=design seed, designs=1)``, ``emulate.trials``): ``trials`` fresh executions, the loss
  (μ̂ − μ)² with the declared zero-estimate nonresponse rule, μ the exact ideal value (emulation only);
* the same (task, candidate, B) fresh-design label of a labels directory is attached when one exists (the root audit's
  four preparation roots label the same tasks).

Summaries (``summarize``): per policy the base-circuit mean of log10(R + 10⁻⁶) under the executed (returned-design)
risk and under the fresh-design label, the policy − SBS-B contrast under both, and their difference, with a percentile
bootstrap over base circuits (roots and budgets of a circuit resampled together). With few circuits the intervals are
wide; the audit tests whether the two targets differ materially, it does not re-estimate the headline.

Ledgers: every executed allocation's shot ledger (calibration, training, diagnostics, selection, execution) is kept;
``unspent`` = B − Σ ledger is reported as it is (never padded). Nothing here submits anything to hardware.
"""
from __future__ import annotations

import hashlib
import math
import os
import time

import numpy as np

from .. import textio as TX
from .models import EPS0


def deployment_seed(root, task, key: str) -> int:
    """Design seed of protocol ``key`` for deployment ``root`` of ``task``: the API's own seed for root 'api', else a
    stable digest namespaced by the root (an independent redraw of every learning design)."""
    from . import deploy as DP
    if root in (None, "api"):
        return DP.design_seed(task, key)
    import dataclasses
    exp = dataclasses.replace(task, budget=1).key()
    h = hashlib.sha256(("qemx.design-audit/1|%s|%s|%s" % (root, exp, key)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def _rng(root, task_id: str, key: str, B: int) -> np.random.Generator:
    h = hashlib.sha256(("qemx.exec-audit/1|%s|%s|%s|%d" % (root, task_id, key, int(B))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def run(specs, bundles: dict, budgets, roots=("api", 1, 2, 3, 4), trials: int = 600, labels=None, progress=print,
        label_tables=None) -> list:
    """Rows of the audit. ``bundles``: {name: (bundle path, allow_simulation)}. ``labels``: optional {root: labels
    dir} whose fresh-design labels are attached (a root key may also be 'any': one directory for every root);
    ``label_tables``: the same as already derived tables."""
    from . import deploy as DP
    from . import emulate as EM
    from . import protocols as PR
    from . import task as TK
    from . import replay as RP
    B_ = {n: DP.load_bundle(p) for n, (p, _sim) in bundles.items()}
    tabs = dict(label_tables or {})
    for r, d in (labels or {}).items():
        if r not in tabs:
            pop = RP.bound_population(d, require_complete=False)
            tabs[r] = {(t.task_id, t.protocol, int(t.B)): (RP._feasible(t.feasible), float(t.mse_eval),
                                                            float(getattr(t, "se_mse_eval", math.nan)))
                       for t in pop["table"].itertuples()}
    rows = []
    for spec in specs:
        built = TK.build(spec)
        mu = float(built.ideal)
        for root in roots:
            cache = {}

            def prepared(key, task):
                sd = deployment_seed(root, task, key)
                if key not in cache:
                    t0 = time.perf_counter()
                    cache[key] = (EM.prepare(built, [key], seed=int(sd), designs=1), sd, time.perf_counter() - t0)
                return cache[key]
            for B in budgets:
                task = DP.from_spec(spec, int(B))
                chosen = {}
                for name, (path, sim) in bundles.items():
                    seeds = {k: deployment_seed(root, task, k) for k in B_[name].protocols}
                    r = DP.recommend(task, B_[name], allow_simulation=sim, design_seeds=seeds)
                    chosen["POLICY|" + name] = (r["choice"], r["status"], r.get("accepted"))
                    chosen.setdefault("SBS-B", (r["fallback"], "fallback candidate verified by the API", None))
                chosen["RAW"] = ("RAW", "fixed", None)
                for pol, (key, status, accepted) in chosen.items():
                    prep, sd, prep_s = prepared(key, task)
                    tr = EM.trials(prep, key, int(B), int(trials), _rng(root, spec.task_id, key, B))
                    led = (tr.allocation or {}).get("ledger") or {}
                    if tr.status == "ok":
                        loss = (np.where(np.isfinite(tr.estimates), tr.estimates, 0.0) - mu) ** 2
                        mse, se = float(loss.mean()), float(loss.std(ddof=1) / math.sqrt(len(loss)))
                    else:
                        mse, se = math.nan, math.nan
                    lab = None
                    for lk in (root, "any"):
                        if lk in tabs:
                            lab = tabs[lk].get((spec.task_id, key, int(B)))
                            break
                    rows.append(dict(
                        root=str(root), task_id=spec.task_id, base_id=spec.base_id, family=spec.family, n=spec.n,
                        B=int(B), policy=pol, protocol=key, learning=bool(PR.PORTFOLIO[key].learning),
                        recommend_status=status, accepted=accepted,
                        design_seed=int(sd) if PR.PORTFOLIO[key].learning else None, execution_status=tr.status,
                        executed_mse=mse, executed_se=se, trials=int(trials),
                        fail_share=float(np.mean(np.any(np.array([v for v in tr.flags.values()]), axis=0)))
                        if tr.status == "ok" else math.nan,
                        fresh_label_mse=(lab[1] if lab and lab[0] else math.nan),
                        fresh_label_se=(lab[2] if lab and lab[0] else math.nan),
                        ledger_total=int(sum(int(v) for v in led.values())) if led else None,
                        unspent=(int(B) - int(sum(int(v) for v in led.values()))) if led else None,
                        **{"ledger_" + k: int(v) for k, v in led.items()},
                        prepare_seconds=round(prep_s, 3)))
                    if progress:
                        progress("%s root %s B=%d %s %s mse=%.3g fresh=%.3g" % (
                            spec.task_id, root, B, pol, key, mse, rows[-1]["fresh_label_mse"]))
    return rows


def _boot(d_by_circuit: dict, reps: int, seed: int) -> tuple:
    u = sorted(d_by_circuit)
    v = np.array([d_by_circuit[k] for k in u], float)
    m = float(v.mean())
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(int(reps), len(v)))].mean(1)
    return m, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def summarize(rows, reps: int = 2000, seed: int = 0) -> dict:
    """Per policy: base-circuit means of log10(R + 10⁻⁶) under the executed risk and the fresh-design label, the
    contrast to SBS-B under both targets and the difference of the two contrasts (bootstrap over base circuits,
    roots and budgets of a circuit kept together). Rows without a fresh-design label are left out of the comparisons
    that need it (counted)."""
    import pandas as pd
    t = pd.DataFrame(rows)
    out = dict(rows=len(t), base_circuits=int(t.base_id.nunique()), roots=sorted(t.root.unique().tolist()),
               budgets=sorted(int(b) for b in t.B.unique()), trials_per_row=int(t.trials.max()) if len(t) else 0,
               metric="base-circuit mean of log10(R + 1e-6); R executed = risk of the returned verified design "
                      "(conditional on it), R fresh = the labels' fresh-design average risk", policies={})
    key = ["root", "task_id", "base_id", "B"]
    base = t[t.policy == "SBS-B"].set_index(key)
    for pol in sorted(set(t.policy) - {"SBS-B"}):
        p = t[t.policy == pol].set_index(key)
        j = p.join(base, rsuffix="_sbsb", how="inner")
        res = dict(rows=int(len(j)))
        for tgt, col in (("executed", "executed_mse"), ("fresh", "fresh_label_mse")):
            ok = np.isfinite(j[col].astype(float)) & np.isfinite(j[col + "_sbsb"].astype(float))
            jj = j[ok].reset_index()
            if not len(jj):
                res[tgt] = dict(rows=0)
                continue
            d = (np.log10(jj[col].astype(float) + EPS0) - np.log10(jj[col + "_sbsb"].astype(float) + EPS0))
            byc = d.groupby(jj.base_id).mean().to_dict()
            m, lo, hi = _boot(byc, reps, seed)
            lv = np.log10(jj[col].astype(float) + EPS0).groupby(jj.base_id).mean()
            res[tgt] = dict(rows=int(len(jj)), mean_log10=float(lv.mean()), diff_vs_sbsb=m, lo=lo, hi=hi,
                            arithmetic_mse=float(jj[col].astype(float).groupby(jj.base_id).mean().mean()))
        both = j[np.isfinite(j.executed_mse.astype(float)) & np.isfinite(j.fresh_label_mse.astype(float)) &
                 np.isfinite(j.executed_mse_sbsb.astype(float)) & np.isfinite(j.fresh_label_mse_sbsb.astype(float))]
        both = both.reset_index()
        if len(both):
            dd = ((np.log10(both.executed_mse + EPS0) - np.log10(both.executed_mse_sbsb + EPS0)) -
                  (np.log10(both.fresh_label_mse + EPS0) - np.log10(both.fresh_label_mse_sbsb + EPS0)))
            m, lo, hi = _boot(dd.groupby(both.base_id).mean().to_dict(), reps, seed + 1)
            res["executed_minus_fresh_contrast"] = dict(rows=int(len(both)), diff=m, lo=lo, hi=hi)
        out["policies"][pol] = res
    lr = t[np.isfinite(t.executed_mse.astype(float)) & np.isfinite(t.fresh_label_mse.astype(float)) & t.learning]
    if len(lr):
        r = np.log10(lr.executed_mse + EPS0) - np.log10(lr.fresh_label_mse + EPS0)
        out["learning_rows"] = dict(rows=int(len(lr)), median_log10_ratio=float(np.median(r)),
                                    q10=float(np.quantile(r, 0.1)), q90=float(np.quantile(r, 0.9)),
                                    note="log10 of executed (returned-design) over fresh-design risk, learning "
                                         "candidates only")
    led = t.dropna(subset=["ledger_total"]) if "ledger_total" in t else t.iloc[0:0]
    out["ledger"] = dict(rows=int(len(led)), over_budget=int((led.unspent < 0).sum()) if len(led) else 0,
                         unspent_rows=int((led.unspent > 0).sum()) if len(led) else 0,
                         max_unspent_share=float((led.unspent / led.B).max()) if len(led) else math.nan)
    return out


def write(rows, summary: dict, out_dir: str) -> None:
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "execution_audit_rows.csv"), index=False, encoding="utf-8")
    TX.write_json(os.path.join(out_dir, "execution_audit_summary.json"), summary, indent=1, default=str)
