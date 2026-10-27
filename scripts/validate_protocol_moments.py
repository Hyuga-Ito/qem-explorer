"""Independent validation of the budget-dependent protocol moments of a target-nonlinear estimator (1.5.1, F01).

Case: one qubit, ry(1.2), observable Z, depolarizing p1 = 0.15 after every gate, exponential ZNE with global folding at
λ = 1, 3, 5 — for each declared failure policy (fallback to the linear fit; abstain) and total budgets B = 100, 300,
1000 (no readout error, no calibration, so all B shots are target shots).

What the app reports (``qemx.recipes.summary``: estimate, bias, sd, RMSE, fit-failure rate, abstention rate, from
``qemx.stats.target_moments``) is compared with complete experiments simulated along an independent path:

* exact E(λ_k) = cos(1.2)·(1 − p1)^k for k = 1, 3, 5 gates — the analytic Bloch-vector contraction of the channel
  (1 − p)ρ + p·I/2 (not the app's simulator);
* the target shots split by an exhaustive search of the integer allocation that minimizes Σ a_j²/N_j, with a_j the
  analytic derivative of the exponential fit times the per-shot SD √(1 − E_j²) (not the app's allocator);
* binomial counts per circuit from this script's own generator, the fit by numpy.linalg.lstsq of log|E| (and of E for
  the linear fallback), and the declared policy.

Pass criterion: |z| ≤ 4 for the mean, the SD, the RMSE and the fit-failure (or abstention) rate, with z built from
both sides' Monte-Carlo standard errors. Usage:

    python scripts/validate_protocol_moments.py --out validation_protocol.json [--repeats 20000]
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

THETA, P1, KS = 1.2, 0.15, (1, 3, 5)
BUDGETS = (100, 300, 1000)
Z_MAX = 4.0


def exact_E() -> np.ndarray:
    return math.cos(THETA) * (1 - P1) ** np.asarray(KS, float)


def exp_derivative(lams, E) -> np.ndarray:
    """∂/∂E_j of s·exp(c0), c0 the intercept of the least-squares line of log|E| on λ."""
    lams = np.asarray(lams, float)
    E = np.asarray(E, float)
    y = np.log(np.abs(E))
    xb = lams.mean()
    sxx = float(((lams - xb) ** 2).sum())
    c1 = float(((lams - xb) * (y - y.mean())).sum() / sxx)
    est = math.copysign(math.exp(y.mean() - c1 * xb), E[0])
    dc0 = 1.0 / len(lams) - xb * (lams - xb) / sxx
    return est * dc0 / E


def best_split(a, N: int) -> tuple:
    """Exhaustive minimum of Σ a_j²/N_j over integers N_j ≥ 1 with Σ N_j = N (three circuits)."""
    a2 = np.asarray(a, float) ** 2
    n1 = np.arange(1, N - 1)[:, None]
    n2 = np.arange(1, N - 1)[None, :]
    n3 = N - n1 - n2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n3 >= 1, a2[0] / n1 + a2[1] / n2 + a2[2] / np.maximum(n3, 1), np.inf)
    i, j = np.unravel_index(int(np.argmin(f)), f.shape)
    return int(n1[i, 0]), int(n2[0, j]), int(N - n1[i, 0] - n2[0, j])


def lstsq_intercept(x, Y) -> np.ndarray:
    A = np.stack([np.ones(len(x)), np.asarray(x, float)], 1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(Y, float).T, rcond=None)
    return coef[0]


def independent(B: int, policy: str, R: int, rng) -> dict:
    lams = np.asarray(KS, float)
    E = exact_E()
    a = np.abs(exp_derivative(lams, E)) * np.sqrt(1 - E ** 2)
    Nj = best_split(a, B)
    p0 = (1 + E) / 2
    Ehat = np.stack([2 * rng.binomial(n, p, size=R) / n - 1 for n, p in zip(Nj, p0)], 1)
    s = np.where(Ehat[:, 0] != 0, np.sign(Ehat[:, 0]), 1.0)
    ok = np.all(s[:, None] * Ehat > 0, axis=1)
    est = np.full(R, np.nan)
    if ok.any():
        est[ok] = s[ok] * np.exp(lstsq_intercept(lams, np.log(s[ok, None] * Ehat[ok])))
    if policy == "fallback" and (~ok).any():
        est[~ok] = lstsq_intercept(lams, Ehat[~ok])
    fin = est[np.isfinite(est)]
    n = len(fin)
    mu, var = float(fin.mean()), float(fin.var(ddof=1))
    m4 = float(np.mean((fin - mu) ** 4))
    var_se = math.sqrt(max(m4 - var * var * (n - 3) / (n - 1), 0.0) / n)
    ideal = math.cos(THETA)
    sq = (fin - ideal) ** 2
    return dict(split=list(Nj), repeats=R, finite=n, mean=mu, mean_se=math.sqrt(var / n), var=var, var_se=var_se,
                rmse=math.sqrt(float(sq.mean())), mse_se=float(sq.std(ddof=1) / math.sqrt(n)),
                fit_failure_rate=float((~ok).mean()), abstention_rate=float(1 - n / R))


def z(a, b, sa, sb) -> float:
    s = math.sqrt(sa * sa + sb * sb)
    return (a - b) / s if s > 0 else (0.0 if a == b else math.inf)


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="validation_protocol.json")
    ap.add_argument("--repeats", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args(argv)
    from qemx import budget as BG
    from qemx.circuit import parse
    from qemx.compile import build
    from qemx.methods import Recipe
    from qemx.noise import NoiseSpec
    from qemx.observables import z as Zobs
    from qemx.recipes import compare, summary, target_protocol
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    cases, ok_all = [], True
    for policy, on_fail in (("fallback", "linear"), ("abstain", "abstain")):
        for B in BUDGETS:
            ideal, outs, _ = compare(build(parse("ry(%r) 0" % THETA, 1)), NoiseSpec(p1=P1), Zobs(0, 1),
                                     [Recipe.make("zne", rule="exponential", on_fail=on_fail)], shots=B, seed=13)
            o = outs[0]
            r = o.result
            row = summary(o, ideal, B, 1.0)
            tm = target_protocol(r, BG.account(r, B))
            app_E = list(r.details["E"])
            ind = independent(B, policy, a.repeats, rng)
            rep_sd = math.sqrt(tm.var)
            emp_sd = math.sqrt(ind["var"])
            rep_rmse = math.sqrt(row["bias"] ** 2 + rep_sd ** 2)
            rep_rmse_se = (abs(row["bias"]) * tm.mean_mcse + tm.var_se / 2) / max(rep_rmse, 1e-300)
            emp_rmse_se = ind["mse_se"] / (2 * max(ind["rmse"], 1e-300))
            rate_rep = tm.fit_failure_rate if policy == "fallback" else tm.abstention_rate
            rate_emp = ind["fit_failure_rate"] if policy == "fallback" else ind["abstention_rate"]
            rate_se = math.sqrt(max(rate_emp * (1 - rate_emp), 1e-12) * (1 / tm.repeats + 1 / ind["repeats"]))
            zs = dict(mean=z(tm.mean, ind["mean"], tm.mean_mcse, ind["mean_se"]),
                      sd=z(rep_sd, emp_sd, tm.var_se / (2 * rep_sd), ind["var_se"] / (2 * emp_sd)),
                      rmse=z(rep_rmse, ind["rmse"], rep_rmse_se, emp_rmse_se),
                      rate=z(rate_rep, rate_emp, 0.0, rate_se))
            # the 1.5.0 report (exact-probability fit and first-order sd) against the same experiments
            sd1 = math.sqrt(r.reference_var1 / B)
            legacy = dict(mean=z(r.reference_mean, ind["mean"], 0.0, ind["mean_se"]),
                          sd=z(sd1, emp_sd, 0.0, ind["var_se"] / (2 * emp_sd)))
            ok = all(abs(v) <= Z_MAX for v in zs.values())
            ok_all &= ok
            cases.append(dict(policy=policy, budget=B, passed=ok, z=zs, legacy_first_order_z=legacy,
                              app=dict(estimate=row["estimate"], estimate_kind=row["estimate kind"],
                                       bias=row["bias"], bias_mcse=row["bias MC ±"], sd=rep_sd, rmse=rep_rmse,
                                       rate=rate_rep, mc_experiments=tm.repeats, exact_fit=r.reference_mean,
                                       first_order_sd=sd1, status=row["status"], flags=row["flags"]),
                              independent=ind, exact_E_app=app_E, exact_E_analytic=list(exact_E()),
                              ideal=ideal))
            print("%-8s B=%5d  pass=%s  z(mean)=%+.2f z(sd)=%+.2f z(rmse)=%+.2f z(rate)=%+.2f | app mean %.4f sd %.4f "
                  "RMSE %.4f rate %.4f | indep mean %.4f sd %.4f RMSE %.4f rate %.4f | 1.5.0 z(mean)=%+.1f z(sd)=%+.1f"
                  % (policy, B, ok, zs["mean"], zs["sd"], zs["rmse"], zs["rate"], tm.mean, rep_sd, rep_rmse, rate_rep,
                     ind["mean"], emp_sd, ind["rmse"], rate_emp, legacy["mean"], legacy["sd"]), flush=True)
    out = dict(script="scripts/validate_protocol_moments.py", criterion="|z| <= %g" % Z_MAX, repeats=a.repeats,
               seed=a.seed, seconds=time.time() - t0, passed=bool(ok_all), cases=cases,
               numpy=np.__version__, python=sys.version.split()[0])
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("ALL PASSED" if ok_all else "SOME CASES FAILED", "->", a.out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
