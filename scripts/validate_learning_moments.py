#!/usr/bin/env python3
r"""Independent check of the reported protocol moments of the learning methods (QA report §4.2).

For each case the reported row (``recipes.summary``: estimate, sd, RMSE at the total budget B) is compared with R
independent complete experiments drawn by the method's own end-to-end sampler (``budget.draw``): fresh training data
at the protocol's training shots, a refit, and fresh target shots at the budget's target allocation — a different
code path from the Monte-Carlo moments behind the reported row (``train_resampler`` with the *exact* target).

Pass criteria (declared before running, Monte-Carlo uncertainty only):

* mean:  |reported − empirical| ≤ 4·sqrt(mcse_reported² + sd_emp²/R)
* sd:    |reported − empirical| ≤ 4·SE(sd_emp) + 4·SE(sd_reported), SE(sd) ≈ sd·sqrt(1/(2(R−1))) (normal approximation;
         the reported sd's own MC error from u and var1 is propagated to first order)
* RMSE:  reported within 4 combined standard errors of the empirical RMSE

The three default cases are the QA report's diagnostic cases (QFT(3, s=5), ideal 1, B = 100 000, 8 training circuits,
seed 13; p1 = p2/10). They are diagnostics, not a ranking of methods.

    python scripts/validate_learning_moments.py --out validation_learning.json            # 6 000 repeats per case
    python scripts/validate_learning_moments.py --reps 1500 --out quick.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES = [
    # (label, p2, train shots, recipe kwargs)
    ("CDR-OLS", 0.03, 64, dict(estimator="learning", features="cdr", model="ols", n_train=8, train_shots=64)),
    ("FR-Ridge", 0.03, 1024, dict(estimator="learning", features="fr", model="ridge", alpha=1e-3, n_train=8,
                                  train_shots=1024)),
    ("NIL-L1", 0.20, 1024, dict(estimator="nil", model="l1", l1_bound=2.0, n_train=8, n_neighbors=4,
                                train_shots=1024)),
]


def run_case(label, p2, train_shots, kw, B=100_000, reps=6000, seed=13, draw_seed=5101):
    from qemx import budget as BG
    from qemx import compile as CP
    from qemx import templates as T
    from qemx.methods import Recipe
    from qemx.noise import NoiseSpec
    from qemx.recipes import compare, summary
    kw = dict(kw)
    est = kw.pop("estimator")
    rc = Recipe.make(est, **kw)
    tpl = T.build("qft", {"n": 3})
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    t0 = time.time()
    ideal, outs, _ = compare(comp, NoiseSpec(p1=p2 / 10, p2=p2), tpl.observable, [rc], shots=B, seed=seed)
    o = outs[0]
    if o.result is None:
        raise RuntimeError(o.error)
    r = o.result
    t_fit = time.time() - t0
    x = np.asarray(BG.draw(r, B, reps, np.random.default_rng(draw_seed)), float)
    t_draw = time.time() - t0 - t_fit
    ok = x[np.isfinite(x)]
    row = summary(o, ideal, B, 250.0, x)
    R = len(ok)
    m_emp, sd_emp = float(np.mean(ok)), float(np.std(ok, ddof=1))
    rmse_emp = float(np.sqrt(np.mean((ok - ideal) ** 2)))
    mom = r.moments or {}
    mcse = float(mom.get("mean_mcse", 0.0))
    acc = BG.account(r, B)
    se_sd_emp = sd_emp / math.sqrt(2 * (R - 1))
    # MC error of the reported sd: Var(sd²) from u (u_se) and var1/N_t (var1_mcse/N_t), first order in sd
    sd_rep = float(row["sd"])
    var_rep_se = math.sqrt(float(mom.get("u_se", 0.0) or 0.0) ** 2 + (float(mom.get("var1_mcse", 0.0) or 0.0)
                                                                     / max(acc["target_shots"], 1.0)) ** 2)
    se_sd_rep = var_rep_se / (2 * sd_rep) if sd_rep > 0 else 0.0
    # empirical RMSE's SE by the delta method on mean((x − ideal)²)
    sq = (ok - ideal) ** 2
    se_rmse_emp = float(np.std(sq, ddof=1) / math.sqrt(R) / (2 * rmse_emp))
    rmse_rep = float(row["RMSE"])
    se_rmse_rep = (abs(row["bias"]) * mcse + sd_rep * se_sd_rep) / rmse_rep if rmse_rep > 0 else 0.0
    z_mean = (r.mean - m_emp) / math.sqrt(mcse ** 2 + sd_emp ** 2 / R)
    z_sd = (sd_rep - sd_emp) / math.sqrt(se_sd_emp ** 2 + se_sd_rep ** 2)
    z_rmse = (rmse_rep - rmse_emp) / math.sqrt(se_rmse_emp ** 2 + se_rmse_rep ** 2)
    return dict(
        case=label, method=o.title, ideal=ideal, p1=p2 / 10, p2=p2, train_shots=train_shots, budget=B, reps=reps,
        finite_reps=R, target_shots=acc["target_shots"], seconds_method=round(t_fit, 2), seconds_draws=round(t_draw, 2),
        estimate_kind=r.estimate_kind, reported_mean=r.mean, reported_mean_mcse=mcse,
        reference_mean=r.reference_mean, reported_sd=sd_rep, reported_rmse=rmse_rep, protocol_repeats=mom.get("repeats"),
        empirical_mean=m_emp, empirical_sd=sd_emp, empirical_rmse=rmse_emp,
        z_mean=z_mean, z_sd=z_sd, z_rmse=z_rmse,
        passed=bool(abs(z_mean) <= 4 and abs(z_sd) <= 4 and abs(z_rmse) <= 4))


def main(argv=None):
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", default=None, help="subset of: " + ", ".join(c[0] for c in CASES))
    a = ap.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    rows = []
    for label, p2, s_tr, kw in CASES:
        if a.cases and label not in a.cases:
            continue
        row = run_case(label, p2, s_tr, kw, reps=a.reps)
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(dict(script="validate_learning_moments.py", criteria="|z| <= 4 for mean, sd and RMSE",
                           cases=rows), f, indent=1)
    bad = [r["case"] for r in rows if not r["passed"]]
    print("FAILED: %s" % ", ".join(bad) if bad else "PASSED: %d cases" % len(rows))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
