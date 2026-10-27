r"""Certification and uncertainty statements (1.6.0; 1.7.0 units and vocabulary): what can be guaranteed, under which
assumptions, and how informative it is at QEM risk scales. Every output names its statement kind
(``claims.KINDS``).

Benchmark certificates (theory companion §1.6.1–1.6.2; statement kind ``finite-sample-bound``). For one task and
budget, independent *units* of each of K protocols give losses ℓ = (μ̃ − μ)². Squared error of an extrapolated or signed
estimator is **not** bounded by 4: μ̂ can leave [−1, 1]. Bounded-loss certificates therefore use a *declared clipped
variant* of every protocol, μ̃ = Π_[−1,1](μ̂) — a projection that cannot increase the error for μ ∈ [−1, 1] but can
change the bias — with H = 4; the unclipped results are reported beside them, without a bound.

Units (1.7.0, review R1): for protocols without a training design the units are the r independent trials; for
learning protocols under fresh designs they are the K independent *design means* (the trials of one design share it,
so they are not independent deployments); for 1.6.0's fixed designs shared by both halves the trials are independent
only given those designs, and the bound is a bound on the risk *conditional on them* (stated in ``scope``).

* ``hoeffding_t`` — t = H √(ln(2K/α) / (2u)) for u units (Theorem 1.7; union bound over K, any dependence across
  protocols);
* ``bernstein_t`` — Maurer & Pontil (2009, Thm. 4) two-sided with a union bound, using the sample variance:
  t = √(2 V̂ ln(4K/α)/u) + 7H ln(4K/α)/(3(u − 1));
* with simultaneous intervals, choosing â = argmin U gives R_â − min R ≤ U_â − min L (Theorem 1.6), and U_â < L_0
  certifies a smaller expected squared error of the *clipped* estimator than the included baseline 0 (RAW) on the
  same event.

At the risks of interest (10⁻⁶–10⁻²) and u in the tens or hundreds these bounds are wide; ``informative`` reports
whether a certificate says anything (U_â < L_RAW) and the units needed for a given half-width.

The 2ε relation (``regret_two_eps``; statement kind ``post-hoc-check``) is checked with the benchmark's measured
risks, which a deployment does not have: it describes the study, it does not certify a deployment.

Predictive uncertainty is a different object (§1.6.4). ``interval_coverage`` measures, on held-out base circuits,
three kinds of interval for a model's predicted log risk: the inner-CV residual interval (``cv-residual``: no
finite-sample guarantee), its Bonferroni-simultaneous version, a residual-Gaussian interval (no guarantee at all), and
a group split-conformal interval (``split-conformal``: finite-sample coverage over exchangeable base circuits, with the
calibration circuits held out of the fit).
"""
from __future__ import annotations

import math

import numpy as np

H_CLIPPED = 4.0


def hoeffding_t(r: int, K: int, alpha: float = 0.05, H: float = H_CLIPPED) -> float:
    return float(H * math.sqrt(math.log(2 * K / alpha) / (2 * r)))


def bernstein_t(var: float, r: int, K: int, alpha: float = 0.05, H: float = H_CLIPPED) -> float:
    lg = math.log(4 * K / alpha)
    return float(math.sqrt(2 * max(var, 0.0) * lg / r) + 7 * H * lg / (3 * (r - 1)))


def r_needed_hoeffding(halfwidth: float, K: int, alpha: float = 0.05, H: float = H_CLIPPED) -> float:
    return float(H ** 2 * math.log(2 * K / alpha) / (2 * halfwidth ** 2))


def simultaneous(losses, alpha: float = 0.05, method: str = "hoeffding", H: float = H_CLIPPED) -> tuple:
    """(L, U) arrays for K candidates from (K, r) bounded losses (NaN rows: infeasible → L = U = NaN)."""
    losses = np.asarray(losses, float)
    K = int(np.sum(np.all(np.isfinite(losses), axis=1)))
    L = np.full(len(losses), np.nan)
    U = np.full(len(losses), np.nan)
    for a, l in enumerate(losses):
        if not np.all(np.isfinite(l)):
            continue
        r = len(l)
        m = float(l.mean())
        t = hoeffding_t(r, K, alpha, H) if method == "hoeffding" else bernstein_t(float(l.var(ddof=1)), r, K, alpha, H)
        L[a], U[a] = max(0.0, m - t), min(H, m + t)
    return L, U


def unit_losses(loss_row, design_row=None, fresh: bool = False) -> np.ndarray:
    """The independent units of one protocol's losses: the trials, or (learning protocols under fresh designs) the
    per-design mean losses."""
    loss_row = np.asarray(loss_row, float)
    if not fresh or design_row is None:
        return loss_row
    d = np.asarray(design_row)
    if np.all(d < 0):
        return loss_row
    return np.array([loss_row[d == k].mean() for k in np.unique(d[d >= 0])])


def _bounds(units: list, alpha: float, method: str, H: float = H_CLIPPED) -> tuple:
    K = sum(1 for u in units if u is not None)
    L = np.full(len(units), np.nan)
    U = np.full(len(units), np.nan)
    rad = np.full(len(units), np.nan)
    for a, l in enumerate(units):
        if l is None or len(l) < 2:
            continue
        m = float(np.mean(l))
        t = hoeffding_t(len(l), K, alpha, H) if method == "hoeffding" else bernstein_t(float(np.var(l, ddof=1)),
                                                                                       len(l), K, alpha, H)
        L[a], U[a], rad[a] = max(0.0, m - t), min(H, m + t), t
    return L, U, rad


def certify(est, mu: float, raw_index: int, alpha: float = 0.05, method: str = "hoeffding", design=None,
            fresh: bool = False) -> dict:
    """Certificate for one task and budget from (K, r) unclipped trial estimates of the evaluation half. ``design``:
    (K, r) design indices (``labels``); with ``fresh`` the units of learning protocols are their design means."""
    est = np.asarray(est, float)
    clipped = np.clip(np.where(np.isfinite(est), est, 0.0), -1, 1)
    feas = np.any(np.isfinite(est), axis=1)            # infeasible: no trial at all; NaN inside: nonresponse → 0
    losses = np.where(feas[:, None], (clipped - mu) ** 2, np.nan)
    units = [unit_losses(losses[a], None if design is None else design[a], fresh) if feas[a] else None
             for a in range(len(losses))]
    L, U, rad = _bounds(units, alpha, method)
    scope = ("clipped estimator; units: trials (protocols without a design) and design means (learning protocols, "
             "fresh designs)" if fresh else "clipped estimator; units: trials (for learning protocols with fixed "
             "shared designs: the risk conditional on those designs)")
    if not np.isfinite(U).any():
        return dict(chosen=-1, informative=False, statement="finite-sample-bound", scope=scope)
    a = int(np.nanargmin(U))
    excess = float(U[a] - np.nanmin(L))
    improves = bool(np.isfinite(L[raw_index]) and U[a] < L[raw_index])
    unclipped = np.where(feas[:, None], (np.where(np.isfinite(est), est, 0.0) - mu) ** 2, np.nan)
    return dict(chosen=a, U_chosen=float(U[a]), excess_bound=excess, improves_on_raw=improves,
                informative=improves, halfwidth=float((U[a] - L[a]) / 2), radius=float(rad[a]),
                units_chosen=int(len(units[a])), mse_clipped=float(np.nanmean(losses[a])),
                mse_unclipped=float(np.nanmean(unclipped[a])), statement="finite-sample-bound", scope=scope)


def regret_two_eps(pred_risk, true_risk) -> dict:
    """Proposition 1.5 on one task: with ε = max over feasible candidates of |R̂_a − R_a|, the argmin of R̂ has
    regret ≤ 2ε. Returns ε, the actual regret and whether the bound holds (it always does; the point is how loose)."""
    p, t = np.asarray(pred_risk, float), np.asarray(true_risk, float)
    ok = np.isfinite(p) & np.isfinite(t)
    if not ok.any():
        return dict(eps=math.nan, regret=math.nan, bound=math.nan, holds=True, statement="post-hoc-check")
    eps = float(np.max(np.abs(p[ok] - t[ok])))
    j = int(np.flatnonzero(ok)[np.argmin(p[ok])])
    reg = float(t[j] - np.min(t[ok]))
    return dict(eps=eps, regret=reg, bound=2 * eps, holds=bool(reg <= 2 * eps + 1e-15), statement="post-hoc-check")


def reference_transform(L: float, U: float, delta: float) -> tuple:
    """Eq. (1.24): an interval [L̃, Ũ] for the risk against a reference μ̃ with |μ̃ − μ| ≤ δ becomes
    [max(0, √L̃ − δ)², (√Ũ + δ)²] for the risk against μ (on the joint event; allocate α to both sources)."""
    lo = max(0.0, math.sqrt(max(L, 0.0)) - delta) ** 2
    hi = (math.sqrt(max(U, 0.0)) + delta) ** 2
    return lo, hi


def interval_coverage(policy_model, ds, test_idx, alpha: float = 0.1) -> dict:
    """Empirical coverage, on held-out base circuits, of three kinds of predictive interval for log10(R_eval + ε0):
    group-conformal per protocol (marginal), the same with a Bonferroni simultaneous level, and a residual-Gaussian
    interval (prediction ± z·inner-CV RMSE; no guarantee). Coverage is reported per base circuit (equal weights)."""
    from .models import EPS0
    from scipy.stats import norm
    rm = policy_model.rm
    test_idx = np.asarray(test_idx)
    P = rm.predict(ds.X(policy_model.features)[test_idx])
    hm = rm.conformal_halfwidth(alpha, simultaneous=False)
    hs = rm.conformal_halfwidth(alpha, simultaneous=True)
    z = norm.ppf(1 - alpha / 2)
    g = ds.groups[test_idx]
    res = {}
    for kind, hw in (("cv-residual", hm), ("cv-residual-simultaneous", hs),
                     ("residual-gaussian", {p: z * v for p, v in rm.cv_rmse.items()})):
        hits, grp, widths = [], [], []
        for r, i in enumerate(test_idx):
            for j, p in enumerate(ds.protocols):
                if not ds.F[i, j] or p not in hw or not np.isfinite(P[r, j]):
                    continue
                y = math.log10(ds.Y_eval[i, j] + EPS0)
                hits.append(abs(y - P[r, j]) <= hw[p])
                grp.append(g[r])
                widths.append(2 * hw[p])
        if not hits:
            continue
        hits, grp = np.array(hits, float), np.array(grp)
        per = [hits[grp == k].mean() for k in sorted(set(grp.tolist()))]
        res[kind] = dict(coverage=float(np.mean(per)), nominal=1 - alpha, width_log10=float(np.mean(widths)),
                         n=int(len(hits)), groups=int(len(per)),
                         statement="cv-residual" if kind.startswith("cv-residual") else "empirical")
    # simultaneous: all feasible protocols of a row covered together
    hits_all, grp_all = [], []
    for r, i in enumerate(test_idx):
        ok = [abs(math.log10(ds.Y_eval[i, j] + EPS0) - P[r, j]) <= hs.get(p, -1)
              for j, p in enumerate(ds.protocols) if ds.F[i, j] and p in hs and np.isfinite(P[r, j])]
        if ok:
            hits_all.append(all(ok)), grp_all.append(g[r])
    if hits_all:
        hits_all, grp_all = np.array(hits_all, float), np.array(grp_all)
        per = [hits_all[grp_all == k].mean() for k in sorted(set(grp_all.tolist()))]
        res["joint (simultaneous)"] = dict(coverage=float(np.mean(per)), nominal=1 - alpha, n=int(len(hits_all)),
                                          groups=int(len(per)), statement="cv-residual")
    return res


def split_conformal(family: str, features, set_name: str, ds, train_idx, test_idx, alpha: float = 0.1,
                    cal_fraction: float = 0.3, seed: int = 0, fixed: dict | None = None) -> dict:
    """Group split-conformal intervals for the predicted log10 risk (statement kind ``split-conformal``).

    The outer training base circuits are split at random into a proper training part and a calibration part
    (``cal_fraction`` of the circuits, at least 2); the model is fitted on the proper part only; each calibration
    circuit g gets one score per protocol, s_g = max over its rows of |log10(R_eval + ε0) − prediction|; the half-width
    is the ⌈(1 − α)(G + 1)⌉-th smallest of the G scores (+∞ when that exceeds G). Under exchangeability of base
    circuits every row of a new circuit is then covered with probability ≥ 1 − α (per protocol; ``simultaneous``:
    Bonferroni over the protocols). Coverage on ``test_idx`` is measured per base circuit (all its rows covered)."""
    from . import policies as PO
    from .models import EPS0
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    g = ds.groups[train_idx]
    u = np.array(sorted(set(g.tolist())))
    rng = np.random.default_rng(seed)
    n_cal = max(2, int(round(cal_fraction * len(u))))
    cal_g = set(rng.choice(u, size=n_cal, replace=False).tolist())
    proper = train_idx[[x not in cal_g for x in g]]
    cal = train_idx[[x in cal_g for x in g]]
    pol = PO.Model(family, features, set_name, fixed=fixed).fit(ds, proper)
    Pc = pol.predict(ds, cal)
    G = len(cal_g)
    K = len(ds.protocols)
    out = {}
    for kind, a in (("split-conformal", alpha), ("split-conformal-simultaneous", alpha / K)):
        k = int(math.ceil((1 - a) * (G + 1)))
        hw = {}
        for j, p in enumerate(ds.protocols):
            sc = {}
            for r, i in enumerate(cal):
                if ds.F[i, j] and np.isfinite(Pc[r, j]):
                    e = abs(math.log10(ds.Y_eval[i, j] + EPS0) - Pc[r, j])
                    sc[ds.groups[i]] = max(sc.get(ds.groups[i], 0.0), e)
            vals = sorted(sc.values())
            if len(vals) < G or not vals:
                continue
            hw[p] = float(vals[k - 1]) if k <= len(vals) else math.inf
        Pt = pol.predict(ds, test_idx)
        cov_g, widths = {}, []
        for r, i in enumerate(test_idx):
            for j, p in enumerate(ds.protocols):
                if not ds.F[i, j] or p not in hw or not np.isfinite(Pt[r, j]):
                    continue
                hit = abs(math.log10(ds.Y_eval[i, j] + EPS0) - Pt[r, j]) <= hw[p]
                cov_g[(ds.groups[i], p)] = cov_g.get((ds.groups[i], p), True) and hit
                widths.append(2 * hw[p])
        per_circ = {}
        for (gg, p), ok in cov_g.items():
            per_circ.setdefault(gg, []).append(ok)
        if kind.endswith("simultaneous"):
            cov = [float(all(v)) for v in per_circ.values()]
        else:
            cov = [float(np.mean(v)) for v in per_circ.values()]
        out[kind] = dict(coverage=float(np.mean(cov)) if cov else math.nan, nominal=1 - alpha,
                         width_log10=float(np.mean(widths)) if widths else math.nan,
                         finite=bool(widths and np.all(np.isfinite(widths))), calibration_circuits=G,
                         proper_training_circuits=int(len(set(ds.groups[proper].tolist()))),
                         groups=int(len(per_circ)), statement="split-conformal",
                         unit="all rows of a base circuit, per protocol" if kind == "split-conformal" else
                         "all rows and protocols of a base circuit")
    return out
