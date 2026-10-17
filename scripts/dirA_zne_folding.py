#!/usr/bin/env python3
r"""Global against local folding for zero-noise extrapolation at equal total budget (direction_A_zne_folding_v1).

    python scripts/dirA_zne_folding.py data/dirA_zne_folding_v1 --out figs_zne_folding       # report, tables, figures
    python scripts/dirA_zne_folding.py data/dirA_zne_folding_v1 --out figs_zne_folding --no-figures

From a notebook (no file is read or written by the computations)::

    import dirA_zne_folding as Z, dirA_figures as F
    T = F.load("data/dirA_zne_folding_v1")
    Fo = Z.folding_table(T)               # one row per (cell, ZNE method): nominal λ, effective λ, depths, ρ
    L = Z.lambda_summary(Fo)              # "λ = 1, 3, 5 (global) against λ = 1, 2.66, 4.33 (local)"
    H = Z.head_to_head(T)                 # one row per (cell, pair): exact crossover budgets of the ledger
    Z.fig_crossover_maps(T, H, "F1-folding")

What is compared
----------------
Global folding repeats the whole circuit, U(U†U)^k; local folding repeats every two-qubit gate, G(G†G)^k. With the same
nominal scales (λ = 1, 3, 5, counted in two-qubit gates) both run the same number of two-qubit gates, so at equal shots
they cost the same — and local folding's circuits are *shallower* (global folding also repeats every one-qubit layer and
idle window; the depth table checks both facts per circuit). They do not amplify the noise by the same factor. The
pipeline records for every folded circuit the effective scale λ_eff = λ_LC(folded)/λ_LC(original), the factor by
which folding raises the light-cone fault rate of the noise model. Richardson extrapolation in the nominal λ lands at

    ρ = Σ_j η_j(λ_nominal) · λ_eff,j            (the noise left over, as a fraction of the unmitigated circuit's).

Local folding repeats the original gates in their original places, so λ_eff measures the amplification of the noise
the observable sees, and to first order the bias is ρ·b_RAW with ρ > 0: local folding leaves the one-qubit gates and
most idle time unfolded and under-corrects (bias_check compares ρ with the measured b/b_RAW). For global folding the
light cone of the folded circuit is wider than the original's (U†U spreads it), so λ_LC also counts faults far from the
observable that rarely reach it: λ_eff overstates the amplification of a local observable's noise and ρ is not a
prediction there. The fault-rate variants (ZNE-Lf, ZNE-Gf) fit against λ_eff itself (ρ = 0 by construction); in
simulation λ_eff is exact for the model, so they are an optimistic reference.

Where local folding breaks down. With RMSE²(N) = b² + v/N two methods cross at N× = (v_G − v_L)/(b_L² − b_G²) target
shots. Local folding's folded circuits are less noisy, so v_L ≤ v_G by a term ∝ λ_LC, while its bias is larger by a
term ∝ λ_LC: to first order N× ∝ λ_LC^-1 — local folding breaks down at a budget that falls as noise × depth grows. When
the per-shot variance does not depend on the noise (an observable whose ideal value is near 0), v_L ≈ v_G and global
folding is better at every budget. B_break, the exact crossover of the ledger (integer allocations, readout
calibration included), is compared with N× (``N_cross_first_order``).

Fractional ZNE (stages F5, F6). Deep circuits decay to the noise floor long before λ = 5: an extrapolation through
λ = 1, 3, 5 fits points that carry no signal. Folding a third of the two-qubit gates once more per step gives
λ = 1, 4/3, 5/3, 2 — every point keeps signal — at the price of a longer lever arm: Richardson through four close
points has Σ|η| = 111 (against 3.5 for 1, 3, 5), a least-squares parabola 19, a line 3.6. So the fractional schedule
trades variance for bias; the ledger says at which budget the trade pays. Local fractional folds choose their gates
by ``subset`` (qemx.transforms.fold_order): stratified over the circuit's duration, nested random, error-balanced
with the calibrated gate errors, or the random subset of 1.4.0; global fractional folds repeat a tail of the circuit.

Tables
------
folding         per cell and folding method: nominal λ, λ_eff, depths and two-qubit counts of the folded circuits, the
                residual fraction ρ, and the measured b/b_RAW
lambda_summary  per stage, folding and nominal scales: median (min–max) λ_eff — the "?.??" of "λ = 1, ?.??, ?.??"
depth           per circuit: depth of the folded circuits, global against local (equal two-qubit counts checked)
head_to_head    per cell and pair (local method, global method): |b|, v, RMSE at the smallest and largest budget, the
                exact budgets at which the better method changes (qemx ledger, integer allocations), the regime,
                B_break (the budget from which the global method is better at every larger budget) and N×
breakdown       per stage, pair and family: how often local folding breaks down inside the budget range, median B_break,
                and the rank correlation of B_break with λ_LC, the noise scale s and the two-qubit count
bias_check      measured b/b_RAW against the first-order prediction ρ, per method

Figures: λ_eff against the nominal λ; the depth of the folded circuits; crossover maps (budget × noise scale per
circuit, ordered by two-qubit count; colour: log₂ of the RMSE ratio; black ticks: B_break); B_break against λ_LC;
the breakdown map (two-qubit gates × noise scale, colour: B_break); measured against predicted bias.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dirA_figures as F  # noqa: E402
from dirA_io import DirectoryFigureSink, DirectoryReportSink, Report  # noqa: E402

# (challenger, reference): the columns of head_to_head are named local/glob after the first study's pairs (local
# against global folding); for the fractional pairs read them as challenger/reference. A pair is skipped where either
# method is missing.
PAIRS = (("ZNE-L", "ZNE-G"), ("ZNE-Lf", "ZNE-G"), ("ZNE-Lf", "ZNE-Gf"), ("ZNE-L-lin", "ZNE-G-lin"),
         ("NIL-ZNE-L", "NIL-ZNE-G"), ("ZNE-L-PEC", "ZNE-G-PEC"),
         ("ZNE-Lq", "ZNE-G"), ("ZNE-Gq", "ZNE-G"), ("ZNE-Lq-exp", "ZNE-G"), ("ZNE-Lq", "ZNE-Gq"),
         ("ZNE-Lq-nest", "ZNE-Lq"), ("ZNE-Lq-bal", "ZNE-Lq"), ("ZNE-Lq-rand", "ZNE-Lq"),
         ("ZNE-Lq-avg4", "ZNE-Lq-nest"), ("ZNE-Lq-avg16", "ZNE-Lq-nest"), ("ZNE-Lq-avg16", "ZNE-Lq"))
PRIMARY = ("ZNE-L", "ZNE-G")
MAPPED = {("ZNE-L", "ZNE-G"), ("ZNE-Lf", "ZNE-G"), ("NIL-ZNE-L", "NIL-ZNE-G"), ("ZNE-L-PEC", "ZNE-G-PEC"),
          ("ZNE-Lq", "ZNE-G"), ("ZNE-Gq", "ZNE-G"), ("ZNE-Lq-exp", "ZNE-G"), ("ZNE-Lq-bal", "ZNE-Lq")}
TIE = 0.05                    # |log RMSE ratio| below log(1.05) is a near-tie (as in the phase maps)
GLOBAL_COL, LOCAL_COL, NEUTRAL = "#2a78d6", "#eb6834", "#f0efec"   # palette slots 1 and 2 of dirA_figures (CVD-safe)
FAMILY_MARKER = {"grover": "D", "qft": "s", "brickwork": "o", "bv": "^", "ghz": "v", "ising": "P"}


# ------------------------------------------------------------------ helpers
def _js(x):
    """A JSON list column value as a list of floats (None for missing)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (list, tuple)):
        return [float(v) if v is not None else math.nan for v in x]
    try:
        v = json.loads(x)
    except (TypeError, ValueError):
        return None
    return None if v is None else [float(t) if t is not None else math.nan for t in v]


def richardson_weights(lams) -> np.ndarray:
    """η_j = Π_{m≠j} λ_m/(λ_m − λ_j) (as qemx.methods.zne.richardson_weights)."""
    lams = np.asarray(lams, float)
    w = np.ones(len(lams))
    for j in range(len(lams)):
        for m in range(len(lams)):
            if m != j:
                w[j] *= lams[m] / (lams[m] - lams[j])
    return w


def linear_weights(lams) -> np.ndarray:
    """Weights of the least-squares line's intercept (as qemx.methods.zne.linear_weights)."""
    V = np.vander(np.asarray(lams, float), 2, increasing=True)
    return np.linalg.pinv(V)[0]


def poly_weights(lams, degree: int) -> np.ndarray:
    """Weights of the value at λ = 0 of the least-squares polynomial of the given degree (as
    qemx.methods.zne.poly_weights)."""
    return np.linalg.pinv(np.vander(np.asarray(lams, float), degree + 1, increasing=True))[0]


def residual_fraction(rule: str, lams_fit, lams_eff) -> float:
    """ρ = Σ_j η_j(λ of the fit)·λ_eff,j: the effective noise scale at which the extrapolation lands (0: none left).
    For a fit that is linear in the data (Richardson, linear, poly2); NaN otherwise or when λ_eff is unknown."""
    if lams_fit is None or lams_eff is None or len(lams_fit) != len(lams_eff) or not np.all(np.isfinite(lams_eff)):
        return math.nan
    if rule == "richardson":
        eta = richardson_weights(lams_fit)
    elif rule == "linear":
        eta = linear_weights(lams_fit)
    elif rule == "poly2" and len(lams_fit) >= 3:
        eta = poly_weights(lams_fit, 2)
    else:
        return math.nan
    return float(eta @ np.asarray(lams_eff, float))


def _cells(T) -> pd.DataFrame:
    keep = ["stage", "family", "n", "circuit", "label", "engine", "scale", "readout", "coherent_fraction", "lam_lc",
            "two_qubit", "depth", "ideal"]
    return T.cells[[c for c in keep if c in T.cells.columns]]


def _budget_range(T, policy="fixed") -> tuple:
    if T.ledger is not None and len(T.ledger):
        B = pd.to_numeric(T.ledger.loc[T.ledger["policy"] == policy, "B"], errors="coerce").dropna()
        if len(B):
            return float(B.min()), float(B.max())
    return 1e3, 1e8


# ------------------------------------------------------------------ computations (tables in, tables out)
def folding_table(T) -> pd.DataFrame:
    """One row per (cell, method) with a folding record (ZNE, or a chain with a ZNE outer method): nominal and
    effective λ, the folded circuits' depths and two-qubit counts, ρ, and the measured bias relative to RAW's."""
    m = T.methods
    if "x_folding" not in m.columns:
        return pd.DataFrame()
    m = m[(m["status"] == "ok") & m["x_folding"].notna()].copy()
    if m.empty:
        return pd.DataFrame()
    raw = T.methods[(T.methods["method"] == "RAW") & (T.methods["status"] == "ok")].set_index("cell_id")["b"]
    rows = []
    for r in m.to_dict("records"):
        lf, lg, lu = _js(r.get("x_lams_fault")), _js(r.get("x_lams_gates")), _js(r.get("x_lams"))
        dep, tq = _js(r.get("x_depths")), _js(r.get("x_twoq"))
        rule = r.get("x_rule")
        b, b_raw = F._num(r.get("b")), F._num(raw.get(r["cell_id"], math.nan))
        dr = _js(r.get("x_draws"))
        rows.append(dict(
            cell_id=r["cell_id"], method=r["method"], folding=r["x_folding"],
            subset=r.get("x_subset") if (r["x_folding"] == "local" and isinstance(r.get("x_subset"), str)) else "",
            draws=int(max(dr)) if dr else 1,
            lam_est=r.get("x_lam_est") if isinstance(r.get("x_lam_est"), str) else "gates", rule=rule,
            lams_nominal=lg, lams_fit=lu, lams_eff=lf,
            lam_eff_2=lf[1] if lf and len(lf) > 1 else math.nan, lam_eff_max=lf[-1] if lf else math.nan,
            depth_1=dep[0] if dep else math.nan, depth_max=dep[-1] if dep else math.nan,
            twoq_1=tq[0] if tq else math.nan, twoq_max=tq[-1] if tq else math.nan,
            rho=residual_fraction(rule, lu, lf), b=b, b_raw=b_raw,
            b_over_raw=(b / b_raw) if (math.isfinite(b) and math.isfinite(b_raw) and b_raw != 0) else math.nan,
            v=F._num(r.get("v"))))
    out = pd.DataFrame(rows)
    out["scales"] = scale_sets(out)
    return out.join(_cells(T), on="cell_id")


def scale_sets(Fo: pd.DataFrame) -> pd.Series:
    """The scale set every method asked for, as a label: the most common achieved nominal set of the method, rounded
    to two decimals (a partial fold adds whole gates, so a circuit with few two-qubit gates achieves only nearby
    scales — 1, 1.33, 1.67, 2 becomes 1, 1.29, 1.67, 1.95 on 21 gates)."""
    if Fo.empty:
        return pd.Series(dtype=object)
    lab = Fo["lams_nominal"].map(lambda v: ", ".join("%.3g" % round(x, 2) for x in v) if v else "")
    mode = lab.groupby(Fo["method"]).agg(lambda x: x.value_counts().index[0])
    return Fo["method"].map(mode)


def lambda_summary(Fo: pd.DataFrame, by=("stage", "folding", "subset")) -> pd.DataFrame:
    """Median (min, max) λ_eff at every scale, one row per group and requested scale set (``scales``; the achieved
    nominal λ, median, beside it). Each folded circuit counts once (methods that share the folded circuits — ZNE-L,
    ZNE-Lf; ZNE-Lq and its other rules — are not double counted)."""
    if Fo.empty or Fo["lams_eff"].isna().all():
        return pd.DataFrame()
    d = Fo[Fo["lams_eff"].notna()].copy()
    by = [b for b in by if b in d.columns]
    if "scales" not in d.columns:
        d["scales"] = d["lams_nominal"].map(lambda v: ", ".join("%g" % x for x in v))
    d = d.drop_duplicates(["cell_id"] + by + ["scales", "draws"] if "draws" in d.columns else ["cell_id"] + by
                          + ["scales"])
    out = []
    for key, g in d.groupby(by + ["scales"]):
        if len({len(v) for v in g["lams_eff"]}) != 1:
            continue
        A = np.array(g["lams_eff"].tolist(), float)
        Nm = np.array(g["lams_nominal"].tolist(), float)
        med, lo, hi = np.median(A, 0), A.min(0), A.max(0)
        row = dict(zip(by + ["scales"], key if isinstance(key, tuple) else (key,)))
        row.update(cells=len(g), nominal_median=", ".join("%.3g" % x for x in np.median(Nm, 0)),
                   lam_eff_median=", ".join("%.2f" % x for x in med),
                   lam_eff_min=", ".join("%.2f" % x for x in lo), lam_eff_max=", ".join("%.2f" % x for x in hi),
                   rho_median=float(np.nanmedian(g["rho"])) if g["rho"].notna().any() else math.nan)
        out.append(row)
    return pd.DataFrame(out)


def depth_table(Fo: pd.DataFrame) -> pd.DataFrame:
    """Per circuit and requested scale set run with both foldings: two-qubit gates and depth of the most folded
    circuit, global against local. Equal two-qubit counts are what makes the comparison equal-cost per shot; they are
    checked here (exactly for odd-integer scales; a fractional fold of the two kinds can differ by a gate or two)."""
    if Fo.empty:
        return pd.DataFrame()
    d = Fo[Fo["lam_est"] == "gates"].copy()
    d["nominal"] = d["scales"] if "scales" in d.columns else d["lams_nominal"].map(
        lambda v: ", ".join("%g" % x for x in v) if v else "")
    d = d.drop_duplicates(["circuit", "folding", "nominal"])
    both = d.groupby(["circuit", "nominal"])["folding"].nunique()
    d = d.join(both.rename("_n"), on=["circuit", "nominal"])
    d = d[d["_n"] == 2]
    if d.empty:
        return pd.DataFrame()
    P = d.pivot_table(index=["circuit", "label", "nominal"], columns="folding",
                      values=["depth_1", "depth_max", "twoq_1", "twoq_max"], aggfunc="first")
    if ("depth_max", "global") not in P.columns or ("depth_max", "local") not in P.columns:
        return pd.DataFrame()
    out = pd.DataFrame(dict(depth=P[("depth_1", "global")], two_qubit=P[("twoq_1", "global")],
                            depth_global=P[("depth_max", "global")], depth_local=P[("depth_max", "local")],
                            twoq_global=P[("twoq_max", "global")], twoq_local=P[("twoq_max", "local")])).reset_index()
    out["depth_ratio_local_global"] = out["depth_local"] / out["depth_global"]
    out["twoq_diff"] = out["twoq_local"] - out["twoq_global"]
    integer = out["nominal"].map(lambda s: all(abs(float(x) - round(float(x))) < 1e-9 and round(float(x)) % 2 == 1
                                               for x in str(s).split(",")))
    out["fractional"] = ~integer
    out["equal_two_qubit"] = np.where(integer, out["twoq_diff"] == 0, out["twoq_diff"].abs() <= 2)
    return out.sort_values(["two_qubit", "circuit"]).reset_index(drop=True)


def _grid_crossovers(ra: np.ndarray, rg: np.ndarray, B: np.ndarray, a: str, g: str) -> list:
    """Crossovers from RMSE curves on a budget grid (no qemx): [(first grid budget of the new winner, winner)]."""
    s = np.where(ra == rg, 0, np.where(ra < rg, -1, 1))
    out, last = [], None
    for Bi, si in zip(B, s):
        if si == 0:
            continue
        if last is not None and si != last:
            out.append((float(Bi), a if si < 0 else g))
        last = si
    return out


def _regime(r_lo: tuple, r_hi: tuple, cross: list, a: str, g: str) -> tuple:
    """(regime, B_break, B_first) from the RMSEs at the ends of the budget range and the crossovers in between."""
    def win(ra, rg):
        if not (math.isfinite(ra) or math.isfinite(rg)):
            return None
        if ra == rg:
            return "tie"
        return a if ra < rg else g
    w_lo, w_hi = win(*r_lo), win(*r_hi)
    if w_lo is None and w_hi is None and not cross:
        return "neither feasible", math.nan, math.nan
    first = cross[0][0] if cross else math.nan
    if w_hi == g:
        to_g = [B for B, w in cross if w == g]
        b_break = to_g[-1] if to_g else -math.inf          # global better throughout: breaks down from the start
    else:
        b_break = math.inf                                 # local is not worse at the largest budget
    winners = [w for _, w in cross]
    if not cross:
        regime = {a: "local throughout", g: "global throughout", "tie": "tie"}.get(w_hi or w_lo, "neither feasible")
    elif len(cross) == 1:
        regime = "local, then global" if winners[0] == g else "global, then local"
    else:
        regime = "multiple"
    return regime, b_break, first


def head_to_head(T, pairs=PAIRS, policy: str = "fixed", points: int = 101) -> pd.DataFrame:
    """One row per (cell, pair): |b|, v, RMSE at the ends of the budget range, every crossover budget and the regime.
    Exact (qemx ledger: integer allocations, integer bisection) when qemx is importable; otherwise from the ledger
    grid (``exact`` = False: crossovers at grid resolution)."""
    present = set(T.methods.loc[T.methods["status"] == "ok", "method"])
    pairs = [(a, g) for a, g in pairs if a in present and g in present]
    if not pairs:
        return pd.DataFrame()
    lo, hi = _budget_range(T, policy)
    M = T.methods[T.methods["status"] == "ok"].set_index(["cell_id", "method"])
    ids = sorted({c for c, k in M.index if k in {x for p in pairs for x in p}})
    costs = F.table_costs(T, ids)
    rows = []
    if costs:
        from qemx.phase.ledger import Policy, crossovers, rmse
        for a, g in pairs:
            for cid in ids:
                if (cid, a) not in costs or (cid, g) not in costs:
                    continue
                (ca, s_min, s_tr_min), (cg, _, _) = costs[(cid, a)], costs[(cid, g)]
                if ca is None or cg is None:
                    continue
                pol = Policy(policy, True, s_min, s_tr_min)
                r_lo = (rmse(ca, int(lo), pol), rmse(cg, int(lo), pol))
                r_hi = (rmse(ca, int(hi), pol), rmse(cg, int(hi), pol))
                cross = crossovers(ca, cg, pol, lo, hi, points)
                rows.append(_h2h_row(cid, a, g, M, r_lo, r_hi, cross, True))
    else:
        L = T.ledger[(T.ledger["policy"] == policy)] if T.ledger is not None else pd.DataFrame()
        if len(L):
            L = L.assign(r=pd.to_numeric(L["rmse"], errors="coerce").fillna(np.inf))
            P = L.pivot_table(index=["cell_id", "B"], columns="method", values="r", aggfunc="first")
            for a, g in pairs:
                if a not in P.columns or g not in P.columns:
                    continue
                for cid, d in P[[a, g]].groupby(level=0):
                    d = d.droplevel(0).sort_index().dropna()
                    if d.empty or (cid, a) not in M.index or (cid, g) not in M.index:
                        continue                                  # the pair is not in this cell
                    B, ra, rg = d.index.to_numpy(float), d[a].to_numpy(float), d[g].to_numpy(float)
                    cross = _grid_crossovers(ra, rg, B, a, g)
                    rows.append(_h2h_row(cid, a, g, M, (ra[0], rg[0]), (ra[-1], rg[-1]), cross, False))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).join(_cells(T), on="cell_id")
    out["B_lo"], out["B_hi"] = lo, hi
    return out


def _h2h_row(cid, a, g, M, r_lo, r_hi, cross, exact) -> dict:
    regime, b_break, first = _regime(r_lo, r_hi, cross, a, g)
    ma, mg = M.loc[(cid, a)], M.loc[(cid, g)]
    ba, bg = F._num(ma.get("b")), F._num(mg.get("b"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_lo = r_lo[0] / r_lo[1] if r_lo[1] > 0 else math.nan
        ratio_hi = r_hi[0] / r_hi[1] if r_hi[1] > 0 else math.nan
    vl, vg = F._num(ma.get("v")), F._num(mg.get("v"))
    return dict(cell_id=cid, pair="%s vs %s" % (a, g), local=a, glob=g, abs_b_local=abs(ba), abs_b_global=abs(bg),
                N_cross_first_order=first_order_crossover(ba, vl, bg, vg),
                v_local=F._num(ma.get("v")), v_global=F._num(mg.get("v")),
                rmse_lo_local=r_lo[0], rmse_lo_global=r_lo[1], rmse_hi_local=r_hi[0], rmse_hi_global=r_hi[1],
                ratio_lo=ratio_lo, ratio_hi=ratio_hi, crossings=len(cross),
                crossovers=json.dumps([[float(B), w] for B, w in cross]), B_first=first, B_break=b_break,
                regime=regime, exact=exact)


def first_order_crossover(b_l: float, v_l: float, b_g: float, v_g: float) -> float:
    """First-order counterpart of B_break, in target shots, from RMSE² = b² + v/N:

    * local folding has the larger bias and the smaller variance: N× = (v_G − v_L)/(b_L² − b_G²), the shots from
      which global folding is better (local folding breaks down there);
    * global folding is not worse in either: −inf (better at every budget);
    * local folding has the smaller bias (whatever the variances): inf — it is the better one at large budgets, so it
      never breaks down (with a larger variance it loses only *below* (v_L − v_G)/(b_G² − b_L²));
    * NaN for missing values, or when the two are the same."""
    if not all(math.isfinite(x) for x in (b_l, v_l, b_g, v_g)):
        return math.nan
    db, dv = b_l * b_l - b_g * b_g, v_g - v_l
    if db == 0 and dv == 0:
        return math.nan
    if db < 0 or (db == 0 and dv > 0):
        return math.inf
    if dv <= 0:
        return -math.inf
    return dv / db


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def breakdown_summary(H: pd.DataFrame, by=("stage", "pair", "family")) -> pd.DataFrame:
    """Per group: cells, how often local folding is worse at the largest budget (breaks down inside the range), how
    often it is worse from the smallest budget on, the median finite B_break, and the rank correlation of log B_break
    with λ_LC, the noise scale, the two-qubit count and the first-order crossover N× (cells that break down inside the
    range)."""
    if H.empty:
        return pd.DataFrame()
    out = []
    by = [b for b in by if b in H.columns]
    for key, g in H.groupby(by):
        bb = g["B_break"].astype(float)
        inside = bb[np.isfinite(bb)]
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(cells=len(g), local_breaks_down=float(np.mean(bb < np.inf)),
                   global_throughout=float(np.mean(bb == -np.inf)),
                   local_holds=float(np.mean(bb == np.inf)),
                   median_B_break=float(np.median(inside)) if len(inside) else math.nan,
                   rho_lam_lc=_spearman(np.log(g.loc[np.isfinite(bb), "lam_lc"].astype(float)), np.log(inside)),
                   rho_scale=_spearman(np.log(g.loc[np.isfinite(bb), "scale"].astype(float)), np.log(inside)),
                   rho_two_qubit=_spearman(g.loc[np.isfinite(bb), "two_qubit"].astype(float), np.log(inside)),
                   rho_first_order=_spearman(np.log(g.loc[np.isfinite(bb), "N_cross_first_order"].astype(float)
                                                    .where(lambda x: x > 0)), np.log(inside))
                   if "N_cross_first_order" in g else math.nan,
                   median_ratio_hi=float(np.nanmedian(g["ratio_hi"])) if g["ratio_hi"].notna().any() else math.nan)
        out.append(row)
    return pd.DataFrame(out)


def power_law(H: pd.DataFrame) -> tuple:
    """(slope, intercept, cells) of log10 B_break = slope·log10 λ_LC + intercept over the cells that break down
    inside the budget range (NaN with fewer than 3 cells). First order in the noise: B_break ∝ λ_LC^-1 (the bias
    difference grows ∝ λ_LC, so its square ∝ λ_LC², and the variance difference ∝ λ_LC; module docstring)."""
    d = H[np.isfinite(H["B_break"].astype(float)) & (H["lam_lc"].astype(float) > 0)]
    if len(d) < 3:
        return math.nan, math.nan, len(d)
    x, y = np.log10(d["lam_lc"].astype(float)), np.log10(d["B_break"].astype(float))
    if np.ptp(x) == 0:
        return math.nan, math.nan, len(d)
    s, c = np.polyfit(x, y, 1)
    return float(s), float(c), len(d)


def bias_check(Fo: pd.DataFrame, min_raw: float = 2e-3) -> pd.DataFrame:
    """Measured b/b_RAW against the first-order prediction ρ, per stage and method (cells with |b_RAW| ≥ min_raw).
    The prediction is first order in the noise: expect agreement where λ_LC is small and departures where it is not."""
    if Fo.empty:
        return pd.DataFrame()
    d = Fo[np.isfinite(Fo["rho"]) & np.isfinite(Fo["b_over_raw"]) & (Fo["b_raw"].abs() >= min_raw)]
    out = []
    for (st, k), g in d.groupby(["stage", "method"]):
        small = g[g["lam_lc"] <= g["lam_lc"].quantile(0.25)] if len(g) >= 4 else g
        out.append(dict(stage=st, method=k, folding=g["folding"].iloc[0], lam_est=g["lam_est"].iloc[0], cells=len(g),
                        rho_median=float(g["rho"].median()), measured_median=float(g["b_over_raw"].median()),
                        measured_median_low_noise=float(small["b_over_raw"].median()),
                        median_abs_error_low_noise=float((small["b_over_raw"] - small["rho"]).abs().median())))
    return pd.DataFrame(out)


def _weights(rule: str, lams) -> np.ndarray:
    lams = np.asarray(lams, float)
    if rule == "richardson":
        return richardson_weights(lams)
    if rule == "linear":
        return linear_weights(lams)
    if rule == "poly2" and len(lams) >= 3:
        return poly_weights(lams, 2)
    return np.full(len(lams), np.nan)


def fractional_summary(T, Fo: pd.DataFrame, reference: str = "ZNE-G", budgets=(1e4, 1e6), policy: str = "fixed",
                       min_raw: float = 2e-3) -> pd.DataFrame:
    """Does fractional ZNE pay at equal budget? One row per stage and ZNE method of the stages that run fractional
    scales: rule, folding, subset, fold draws, scale set, Σ|η| (the shot-noise amplification, median), the median
    |b| / |b_RAW| (cells with |b_RAW| ≥ min_raw), the median per-shot variance relative to RAW's, and at each budget
    the median RMSE ratio against ``reference`` (< 1: better) and the share of cells where the method is better."""
    if Fo.empty or T.ledger is None or not len(T.ledger):
        return pd.DataFrame()
    frac_stages = sorted({st for st, g in Fo.groupby("stage") if any(
        abs((x - 1) / 2 - round((x - 1) / 2)) > 1e-9 for v in g["lams_nominal"].dropna() for x in v)})
    if not frac_stages:
        return pd.DataFrame()
    m = T.methods[T.methods["status"] == "ok"]
    v_raw = m[m["method"] == "RAW"].set_index("cell_id")["v"]
    L = T.ledger[T.ledger["policy"] == policy].copy()
    L["B"] = pd.to_numeric(L["B"], errors="coerce")
    grid = np.unique(L["B"].dropna().to_numpy())
    out = []
    for st in frac_stages:
        f = Fo[Fo["stage"] == st]
        cells = set(f["cell_id"])
        for k, g in f.groupby("method", sort=False):
            gamma = m[(m["method"] == k) & m["cell_id"].isin(cells)]["gamma"]
            low = g[g["b_raw"].abs() >= min_raw]
            vv = g.join(v_raw.rename("v_raw"), on="cell_id")
            row = dict(stage=st, method=k, rule=g["rule"].iloc[0], folding=g["folding"].iloc[0],
                       subset=g["subset"].iloc[0] if "subset" in g else "", draws=int(g["draws"].max())
                       if "draws" in g else 1, scales=g["scales"].iloc[0] if "scales" in g else "", cells=len(g),
                       sum_abs_eta=float(np.nanmedian(pd.to_numeric(gamma, errors="coerce"))) if len(gamma)
                       else math.nan,
                       abs_b_over_raw=float(low["b_over_raw"].abs().median()) if len(low) else math.nan,
                       v_over_raw=float((vv["v"] / vv["v_raw"]).median()))
            for B in budgets:
                Bg = grid[np.argmin(np.abs(np.log(grid / B)))] if len(grid) else math.nan
                P = L[(L["B"] == Bg) & L["cell_id"].isin(cells)].pivot_table(index="cell_id", columns="method",
                                                                              values="rmse", aggfunc="first")
                if k in P.columns and reference in P.columns and k != reference:
                    r = (P[k] / P[reference]).replace([np.inf, -np.inf], np.nan).dropna()
                    row["rmse_ratio_%s" % _bk(B)] = float(r.median()) if len(r) else math.nan
                    row["better_%s" % _bk(B)] = float((r < 1).mean()) if len(r) else math.nan
            out.append(row)
    return pd.DataFrame(out)


def _bk(B: float) -> str:
    e = int(round(math.log10(B)))
    return "1e%d" % e if abs(B - 10 ** e) < 1e-6 * B else "%g" % B


LC_BANDS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, math.inf)


def fractional_regime(H: pd.DataFrame, reference: str = "ZNE-G", bands=LC_BANDS) -> tuple:
    """Where does a fractional method beat the reference at the largest budget? (table by λ_LC band, crossover
    table). Per challenger of ``reference`` and λ_LC band: cells, share of cells where the challenger has the lower
    RMSE at the largest budget, median RMSE ratio there. The crossover λ*_LC is where a logistic fit of "challenger
    wins" against log λ_LC passes 1/2 (NaN when the fit does not cross inside the data)."""
    if H is None or H.empty:
        return pd.DataFrame(), pd.DataFrame()
    h = H[(H["glob"] == reference) & np.isfinite(H["lam_lc"].astype(float)) & np.isfinite(H["ratio_hi"])].copy()
    h = h[h["local"].map(lambda k: "q" in str(k).split("-")[1] if "-" in str(k) else False)]
    if h.empty:
        return pd.DataFrame(), pd.DataFrame()
    h["band"] = pd.cut(h["lam_lc"].astype(float), list(bands), right=True)
    h["wins"] = h["ratio_hi"] < 1
    T = h.groupby(["stage", "local", "band"], observed=True).agg(
        cells=("cell_id", "count"), wins=("wins", "mean"), ratio_hi=("ratio_hi", "median")).reset_index()
    T["band"] = T["band"].astype(str)
    X = []
    for (st, k), g in h.groupby(["stage", "local"]):
        a, b = _logistic(np.log(g["lam_lc"].astype(float).to_numpy()), g["wins"].to_numpy(float))
        lo, hi = float(g["lam_lc"].min()), float(g["lam_lc"].max())
        x = math.exp(-a / b) if (math.isfinite(a) and math.isfinite(b) and b > 0) else math.nan
        X.append(dict(stage=st, method=k, cells=len(g), wins=float(g["wins"].mean()),
                      lam_lc_star=x if (math.isfinite(x) and lo <= x <= hi) else math.nan, slope=b))
    return T, pd.DataFrame(X)


def _logistic(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple:
    """(a, b) of P(y = 1) = 1 / (1 + exp(−(a + b x))) by Newton's method (NaN when the classes separate)."""
    if len(x) < 4 or y.min() == y.max():
        return math.nan, math.nan
    A = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ w)))
        g = A.T @ (y - p)
        Hm = (A * (p * (1 - p))[:, None]).T @ A + 1e-9 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        w = w + step
        if np.abs(step).max() < 1e-10 or np.abs(w).max() > 1e6:
            break
    return (float(w[0]), float(w[1])) if np.abs(w).max() < 1e6 else (math.nan, math.nan)


def subset_noise(T, Fo: pd.DataFrame) -> pd.DataFrame:
    """The subset's own effect on fractional ZNE, from the methods that average several fold subsets (fold_draws >
    1): per cell, the SD of E(λ) across the subsets at every fractional λ (exact values, no shot noise) and the SD
    it gives a *single-subset* estimate, √(Σ_j η_j² SD_j²) — a bias that changes sign with the subset, against the
    bias left after averaging K subsets (≈ 1/√K of it) and the per-shot SD."""
    m = T.methods
    if "x_E_draws" not in m.columns:
        return pd.DataFrame()
    m = m[(m["status"] == "ok") & m["x_E_draws"].notna()]
    rows = []
    for r in m.to_dict("records"):
        try:
            Ed = json.loads(r["x_E_draws"]) if isinstance(r["x_E_draws"], str) else r["x_E_draws"]
        except (TypeError, ValueError):
            continue
        if not Ed or max(len(e) for e in Ed) < 2:
            continue
        lams = _js(r.get("x_lams"))
        sd = np.array([np.std(e, ddof=1) if len(e) > 1 else 0.0 for e in Ed], float)
        eta = _weights(r.get("x_rule"), lams)
        rows.append(dict(cell_id=r["cell_id"], method=r["method"], draws=max(len(e) for e in Ed),
                         subset_sd_max=float(sd.max()), single_subset_sd=float(math.sqrt(np.sum(eta ** 2 * sd ** 2))),
                         b=F._num(r.get("b")), sigma=math.sqrt(F._num(r.get("v")))))
    out = pd.DataFrame(rows)
    return out.join(_cells(T), on="cell_id") if len(out) else out


def analyse(T, pairs=PAIRS, policy: str = "fixed") -> Report:
    """Every table of this module, as a report (text and tables)."""
    rep = Report()
    rep.section("EFFECTIVE NOISE SCALE · λ_eff = λ_LC(folded) / λ_LC(original), per folding (each folded circuit once)")
    Fo = folding_table(T)
    rep.table("folding", Fo.drop(columns=["lams_nominal", "lams_fit", "lams_eff"], errors="ignore").assign(
        lams_nominal=Fo["lams_nominal"].map(json.dumps) if len(Fo) else None,
        lams_eff=Fo["lams_eff"].map(json.dumps) if len(Fo) else None) if len(Fo) else Fo)
    Ls = lambda_summary(Fo)
    if Ls.empty:
        rep.verdict("lambda-eff", None, "no folded circuit with an effective λ in this store (stores before v1.4.1 do "
                                        "not record it)")
    else:
        rep.table("lambda_summary", Ls, show=True)
        rep.text("ρ = Σ η_j(λ_nominal)·λ_eff,j: the share of the unmitigated circuit's noise that the nominal-λ "
                 "extrapolation leaves (> 0: under-corrects, < 0: over-corrects).")
    D = depth_table(Fo)
    if not D.empty:
        rep.section("DEPTH · the most folded circuit, global against local (same two-qubit gate count)")
        rep.table("depth", D)
        rep.frame(D[["label", "nominal", "two_qubit", "depth", "depth_global", "depth_local",
                     "depth_ratio_local_global", "equal_two_qubit"]].round(3))
        rep.verdict("equal-two-qubit-count", bool(D["equal_two_qubit"].all()),
                    "the folded circuits of the two foldings have the same two-qubit gate count (fractional scales: "
                    "within two gates) in %d of %d circuit × scale-set pairs" % (int(D["equal_two_qubit"].sum()), len(D)))
        rep.verdict("local-not-deeper", None if not len(D) else bool((D["depth_local"] <= D["depth_global"]).all()),
                    "local folding's most folded circuit is %.2f–%.2f × as deep as global folding's: a breakdown of "
                    "local folding is not caused by added depth" % (D["depth_ratio_local_global"].min(),
                                                                     D["depth_ratio_local_global"].max()))
    rep.section("HEAD TO HEAD · local against global folding at equal total budget (%s policy)" % policy)
    H = head_to_head(T, pairs, policy)
    rep.table("head_to_head", H)
    if H.empty:
        rep.verdict("head-to-head", None, "no pair of %s in this store" % ", ".join("/".join(p) for p in pairs))
        return rep
    lo, hi = float(H["B_lo"].iloc[0]), float(H["B_hi"].iloc[0])
    rep.text("budget range %.0e … %.0e; B_break: the budget from which the global method has the lower RMSE at every "
             "larger budget (−inf: from the smallest, inf: local is not worse at the largest)%s"
             % (lo, hi, "" if H["exact"].all() else " — crossovers at ledger-grid resolution (qemx not importable)"))
    S = breakdown_summary(H)
    rep.table("breakdown", S)
    S2 = breakdown_summary(H, by=("stage", "pair"))
    rep.frame(S2.round(3))
    chi = H.groupby("stage")["coherent_fraction"].nunique() if "coherent_fraction" in H.columns else pd.Series()
    if (chi > 1).any():
        Hc = H[H["stage"].isin(chi[chi > 1].index)]
        Sc = breakdown_summary(Hc, by=("stage", "pair", "coherent_fraction"))
        rep.table("breakdown_coherent", Sc)
        rep.text("by coherent fraction χ (coherent errors add in amplitude when a gate is repeated, so local folding "
                 "of self-inverse gates amplifies them faster than their infidelity suggests):")
        rep.frame(Sc[["stage", "pair", "coherent_fraction", "cells", "local_holds", "global_throughout",
                      "median_B_break", "median_ratio_hi"]].round(3))
    for (st, pair), g in H.groupby(["stage", "pair"]):
        s, c, k = power_law(g)
        if math.isfinite(s):
            rep.text("%s · %s: log10 B_break = %.2f·log10 λ_LC %+.2f over %d cells (first-order expectation: −1)"
                     % (st, pair, s, c, k))
    reg = H.groupby(["stage", "pair", "regime"]).size().unstack(fill_value=0)
    rep.text("regimes (cells):")
    rep.frame(reg)
    rep.section("BIAS · measured b/b_RAW against the first-order prediction ρ (cells with |b_RAW| ≥ 0.002)")
    Bc = bias_check(Fo)
    rep.table("bias_check", Bc, show=True)
    Fs = fractional_summary(T, Fo, policy=policy)
    if len(Fs):
        rep.section("FRACTIONAL ZNE · bias, shot-noise amplification and RMSE at equal budget against ZNE-G "
                    "(Richardson on 1, 3, 5, global)")
        rep.table("fractional", Fs)
        rep.frame(Fs.round(3))
        rep.text("sum_abs_eta: Σ|η| (× the readout amplification), the factor on the shot noise; abs_b_over_raw: "
                 "median |b|/|b_RAW|; rmse_ratio_B: median RMSE / RMSE of ZNE-G at total budget B; better_B: share of "
                 "cells where the method wins")
    Rg, Rx = fractional_regime(H)
    if len(Rg):
        rep.table("fractional_regime", Rg)
        rep.table("fractional_crossover", Rx)
        rep.text("where fractional ZNE pays: share of cells won against ZNE-G at the largest budget, by light-cone "
                 "fault count λ_LC; λ*_LC: logistic 50 % point")
        rep.frame(Rg.pivot_table(index=["stage", "local"], columns="band", values="wins", observed=True,
                                 sort=False).round(2).reset_index())
        rep.frame(Rx.round(3))
    Sn = subset_noise(T, Fo)
    if len(Sn):
        rep.table("subset_noise", Sn)
        g = Sn.groupby("method").agg(cells=("cell_id", "count"), draws=("draws", "max"),
                                     single_subset_sd=("single_subset_sd", "median"),
                                     abs_b=("b", lambda b: float(np.median(np.abs(b)))),
                                     sigma_per_shot=("sigma", "median")).reset_index()
        rep.text("subset noise: the SD that the choice of fold subset gives a single-subset estimate (median over "
                 "cells), against the bias left after averaging the draws:")
        rep.frame(g.round(4))
    if not Fo.empty and "coherent_fraction" in Fo.columns and Fo.groupby("stage")["coherent_fraction"].nunique().max() > 1:
        d = Fo[np.isfinite(Fo["b_over_raw"]) & (Fo["b_raw"].abs() >= 2e-3)]
        d = d[d["stage"].isin([st for st, g in d.groupby("stage") if g["coherent_fraction"].nunique() > 1])]
        Bx = d.groupby(["stage", "coherent_fraction", "method"])["b_over_raw"].median().unstack("method")
        rep.table("bias_coherent", Bx.reset_index())
        rep.text("median b/b_RAW by coherent fraction χ (λ_LC, hence ρ, is blind to coherence):")
        rep.frame(Bx.round(3).reset_index())
    return rep


# ------------------------------------------------------------------ figures (tables in, figures out)
def _fold_style(folding: str, lam_est: str = "gates"):
    col = GLOBAL_COL if folding == "global" else LOCAL_COL
    mk = "s" if folding == "global" else "o"
    return col, mk, ("-" if lam_est == "gates" else "--")


def _scale_set_for_figure(d: pd.DataFrame) -> str:
    """The requested scale set the λ_eff figure shows: one run with both foldings if there is one (most rows),
    else the most common one."""
    if "scales" not in d.columns or d.empty:
        return ""
    both = d.groupby("scales")["folding"].nunique()
    pool = both[both == 2].index if (both == 2).any() else both.index
    return str(d[d["scales"].isin(pool)]["scales"].value_counts().index[0])


def fig_lambda_eff(Fo: pd.DataFrame, stage: str | None = None):
    """(a) λ_eff against the nominal λ, every folded circuit (points) and the median (line), identity dashed;
    (b) ρ against the circuit's two-qubit gate count, for one rule. One requested scale set (both foldings when the
    stage runs them on the same set)."""
    import matplotlib.pyplot as plt
    d = Fo[(Fo["lam_est"] == "gates") & Fo["lams_eff"].notna()]
    if stage is not None:
        d = d[d["stage"] == stage]
    sset = _scale_set_for_figure(d)
    if sset:
        d = d[d["scales"] == sset]
    if "subset" in d.columns:                         # one local subset (the first in the preference order)
        subs = [x for x in ("stratified", "", "random", "nested", "balanced") if x in set(d["subset"])]
        if subs:
            d = d[(d["folding"] == "global") | (d["subset"] == subs[0])]
    if "draws" in d.columns:
        d = d[d["draws"] <= 1]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axs[0]
    top, xtop = 1.0, 1.0
    xmax = max((max(v) for v in d["lams_nominal"] if v), default=5.0)
    for fo in ("global", "local"):
        g = d[d["folding"] == fo].drop_duplicates(["cell_id"]).copy()
        if g.empty or len({len(v) for v in g["lams_eff"]}) != 1:
            continue
        col, mk, _ = _fold_style(fo)
        A = np.array(g["lams_eff"].tolist(), float)
        x = np.median(np.array(g["lams_nominal"].tolist(), float), 0)
        jit = 0.02 * xmax * (1 if fo == "global" else -1)
        ax.scatter(np.tile(x + jit, (len(A), 1)).ravel(), A.ravel(), s=5, color=col, alpha=0.25, linewidths=0)
        med = np.median(A, 0)
        ax.plot(x + jit, med, color=col, marker=mk, ms=4, lw=1.4,
                label="%s: λ_eff = %s" % (fo, ", ".join("%.2f" % v for v in med)))
        top, xtop = max(top, float(A.max())), max(xtop, float(x.max()))
    pad = 0.1 * xtop
    ax.plot([0, top + pad], [0, top + pad], color=F.INK2, lw=0.8, ls=":", zorder=0)
    ax.set_xlim(1 - pad, xtop + pad)
    ax.set_ylim(1 - pad, top + pad)
    ax.set_xlabel("nominal λ (two-qubit gates ÷ original)")
    ax.set_ylabel(r"effective $\lambda$ ($\lambda_{LC}$ ÷ original)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.set_title("(a) noise amplification reached%s" % ("" if stage is None else " · " + stage), loc="left",
                 fontsize=9)
    ax = axs[1]
    rules = [r for r in ("richardson", "poly2", "linear") if (d["rule"] == r).any()
             and np.isfinite(d.loc[d["rule"] == r, "rho"].astype(float)).any()]
    integer = bool(len(d)) and all(abs(x - round(x)) < 1e-9 and round(x) % 2 == 1 for x in d["lams_nominal"].iloc[0])
    rule = ("richardson" if (integer and "richardson" in rules) else
            next((r for r in ("poly2", "linear", "richardson") if r in rules), None))
    for fo in ("global", "local"):
        g = d[(d["folding"] == fo) & (d["rule"] == rule)].drop_duplicates(["cell_id"])
        if g.empty:
            continue
        col, mk, _ = _fold_style(fo)
        ax.scatter(g["two_qubit"].astype(float), g["rho"], s=10, color=col, marker=mk, alpha=0.6, linewidths=0,
                   label=fo)
    ax.axhline(0, color=F.INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    F.plain_log_axis(ax)
    ax.set_xlabel("two-qubit gates of the circuit")
    ax.set_ylabel(r"$\rho=\sum_j\eta_j\lambda_{{\rm eff},j}$ (%s)" % {"richardson": "Richardson", "poly2": "quadratic",
                                                                     "linear": "linear"}.get(rule, "—"))
    ax.set_title("(b) noise left by the extrapolation", loc="left", fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def fig_depth(D: pd.DataFrame):
    """Depth of the most folded circuit against the original depth: global (squares) and local (circles)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    if len(D):
        longest = max(D["nominal"].unique(), key=lambda t: (len(t.split(",")), float(t.split(",")[-1])))
        d = D[D["nominal"] == longest]
    else:
        d = D
    ax.scatter(d["depth"], d["depth_global"], color=GLOBAL_COL, marker="s", s=14, label="global folding")
    ax.scatter(d["depth"], d["depth_local"], color=LOCAL_COL, marker="o", s=14, label="local folding")
    lam_max = float(str(d["nominal"].iloc[0]).split(",")[-1]) if len(d) else 5.0
    x = np.geomspace(max(d["depth"].min(), 1), d["depth"].max(), 10) if len(d) else np.array([1, 10])
    ax.plot(x, lam_max * x, color=F.INK2, lw=0.8, ls=":", label="λ × depth")
    ax.set_xscale("log")
    ax.set_yscale("log")
    F.plain_log_axis(ax, "x")
    F.plain_log_axis(ax, "y")
    ax.set_xlabel("depth of the compiled circuit")
    ax.set_ylabel("depth at nominal λ = %g" % lam_max)
    ax.set_title("Equal two-qubit gates, unequal depth", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def _ratio_grid(T, cell_ids, a, g, budgets, policy):
    """log2(RMSE_a / RMSE_g) of each cell over ``budgets`` (exact ledger when available, else the ledger grid)."""
    costs = F.table_costs(T, cell_ids)
    out = {}
    if costs:
        from qemx.phase.ledger import Policy, rmse
        for cid in cell_ids:
            if (cid, a) in costs and (cid, g) in costs and costs[(cid, a)][0] is not None \
                    and costs[(cid, g)][0] is not None:
                pol = Policy(policy, True, costs[(cid, a)][1], costs[(cid, a)][2])
                ra = np.array([rmse(costs[(cid, a)][0], int(B), pol) for B in budgets])
                rg = np.array([rmse(costs[(cid, g)][0], int(B), pol) for B in budgets])
                out[cid] = (ra, rg)
    else:
        L = T.ledger[(T.ledger["policy"] == policy) & T.ledger["cell_id"].isin(cell_ids)]
        for cid, d in L.groupby("cell_id"):
            P = d.assign(r=pd.to_numeric(d["rmse"], errors="coerce").fillna(np.inf)).pivot_table(
                index="B", columns="method", values="r", aggfunc="first")
            if a in P.columns and g in P.columns:
                out[cid] = (np.interp(np.log(budgets), np.log(P.index.to_numpy(float)), P[a].to_numpy(float)),
                            np.interp(np.log(budgets), np.log(P.index.to_numpy(float)), P[g].to_numpy(float)))
    grid = {}
    for cid, (ra, rg) in out.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log2(ra / rg)
        r[~np.isfinite(ra) & ~np.isfinite(rg)] = np.nan                  # neither feasible
        r[~np.isfinite(ra) & np.isfinite(rg)] = 2.0                       # only the global method is feasible
        r[np.isfinite(ra) & ~np.isfinite(rg)] = -2.0
        grid[cid] = r
    return grid


def fold_cmap():
    """Diverging map: global better (blue) ← neutral grey → local better (orange); two hues, grey midpoint."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("fold", [LOCAL_COL, NEUTRAL, GLOBAL_COL])


def fig_crossover_maps(T, H: pd.DataFrame, stage: str, pair=PRIMARY, *, readout=None, coherent_fraction=None,
                       circuits=None, ncols: int = 4, policy: str = "fixed", points: int = 41):
    """Small multiples, one per circuit (ordered by two-qubit gate count): x = total budget B, y = noise scale s,
    colour = log₂(RMSE_local / RMSE_global) clipped to ±2 (orange: local better, blue: global better), black ticks =
    B_break (from there on the global method is better at every larger budget), hatched = neither method feasible."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    a, g = pair
    h = H[(H["stage"] == stage) & (H["local"] == a) & (H["glob"] == g)]
    if readout is not None:
        h = h[h["readout"] == readout]
    if coherent_fraction is not None:
        h = h[np.isclose(h["coherent_fraction"].astype(float), coherent_fraction)]
    if circuits is not None:
        h = h[h["circuit"].isin(circuits)]
    if h.empty:
        raise ValueError("no %s vs %s cells in stage %s" % (a, g, stage))
    lo, hi = float(h["B_lo"].iloc[0]), float(h["B_hi"].iloc[0])
    budgets = np.unique(np.round(np.geomspace(lo, hi, points)).astype(np.int64))
    grid = _ratio_grid(T, list(h["cell_id"]), a, g, budgets, policy)
    order = h.groupby("circuit").agg(q=("two_qubit", "first"), label=("label", "first")).sort_values(["q", "label"])
    k = len(order)
    ncols = min(ncols, k)
    nrows = int(math.ceil(k / ncols))
    row_in, top_in, bottom_in = 1.55, 0.55, 1.15              # inches: panel row, title band, axis + colour bar
    H_in = top_in + nrows * row_in + bottom_in
    fig, axs = plt.subplots(nrows, ncols, figsize=(1.8 * ncols + 0.9, H_in), sharex=True, sharey=True,
                            squeeze=False)
    fig.subplots_adjust(left=0.6 / (1.8 * ncols + 0.9), right=0.98, top=1 - top_in / H_in, bottom=bottom_in / H_in,
                        hspace=0.5, wspace=0.12)
    norm, cmap = Normalize(-2, 2), fold_cmap()
    edges_B = np.sqrt(budgets[:-1] * budgets[1:])
    xedges = np.concatenate([[budgets[0]], edges_B, [budgets[-1]]]).astype(float)
    scales = np.sort(h["scale"].astype(float).unique())
    ys = np.log(scales)
    mid = (ys[:-1] + ys[1:]) / 2 if len(ys) > 1 else np.array([])
    yedges = np.exp(np.concatenate([[ys[0] - (mid[0] - ys[0] if len(mid) else 0.5)], mid,
                                    [ys[-1] + (ys[-1] - mid[-1] if len(mid) else 0.5)]]))
    im = None
    for i, (circ, info) in enumerate(order.iterrows()):
        ax = axs[i // ncols][i % ncols]
        hc = h[h["circuit"] == circ].set_index("scale")
        Z = np.full((len(scales), len(budgets)), np.nan)
        for j, s in enumerate(scales):
            if s in hc.index:
                cid = hc.loc[s, "cell_id"]
                cid = cid.iloc[0] if isinstance(cid, pd.Series) else cid
                if cid in grid:
                    Z[j] = np.clip(grid[cid], -2, 2)
        ax.set_facecolor("#ffffff")
        ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(Z), cmap=cmap, norm=norm, shading="flat",
                      edgecolors="none", rasterized=True)
        nan = np.isnan(Z)
        for j in range(len(scales)):                     # neither method feasible: hatched runs of tiles
            c = 0
            while c < len(budgets):
                if not nan[j, c]:
                    c += 1
                    continue
                e = c
                while e + 1 < len(budgets) and nan[j, e + 1]:
                    e += 1
                ax.add_patch(plt.Rectangle((xedges[c], yedges[j]), xedges[e + 1] - xedges[c], yedges[j + 1] - yedges[j],
                                           facecolor="#ffffff", edgecolor=F.GRID, hatch="////", lw=0))
                c = e + 1
        for j, s in enumerate(scales):
            if s not in hc.index:
                continue
            bb = float(hc.loc[s, "B_break"] if not isinstance(hc.loc[s, "B_break"], pd.Series)
                       else hc.loc[s, "B_break"].iloc[0])
            if math.isfinite(bb):
                ax.plot([bb, bb], [yedges[j], yedges[j + 1]], color=F.INK, lw=1.6, solid_capstyle="butt")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        ax.set_title("%s · $n_2$=%d" % (info["label"], int(info["q"])), loc="left", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.minorticks_off()
        ax.set_yticks(scales)
        ax.set_yticklabels(["%g" % s for s in scales])
        ax.set_xticks([10 ** e for e in range(int(math.ceil(math.log10(lo))), int(math.floor(math.log10(hi))) + 1)])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "10$^{%d}$" % round(math.log10(v))))
    for j in range(k, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    for r in range(nrows):
        axs[r][0].set_ylabel("noise scale s", fontsize=7)
    for c in range(ncols):
        axs[nrows - 1][c].set_xlabel("total shots B", fontsize=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.2, 0.42 / H_in, 0.6, 0.11 / H_in])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(["%s 4× better" % a, "2×", "equal", "2×", "%s 4× better" % g])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("RMSE ratio %s / %s (log scale) · black tick: B_break, from which %s is better at every larger "
                 "budget · hatched: neither reaches a finite RMSE · $n_2$: two-qubit gates" % (a, g, g), fontsize=7)
    sub = [stage] + (["readout %s" % readout] if readout else []) + (
        ["χ = %g" % coherent_fraction] if coherent_fraction is not None else [])
    fig.suptitle("%s against %s at equal total budget · %s" % (a, g, " · ".join(sub)), x=0.01,
                 y=1 - 0.12 / H_in, ha="left", va="top", fontsize=9)
    return fig


def fig_breakdown(H: pd.DataFrame, stage: str, pairs=None):
    """B_break against λ_LC (log-log), one panel per pair: family marker shapes; cells where local folding is not worse
    at the largest budget sit on the top edge (open, ▲), cells where global folding is better from the smallest budget
    on the bottom edge (open, ▼); fitted power law and the first-order slope −1 (dotted)."""
    import matplotlib.pyplot as plt
    h = H[H["stage"] == stage]
    if pairs is None:                                   # pairs with a crossover to show (≥ 3 finite B_break)
        pairs = [p for p, g in h.groupby("pair", sort=False) if np.isfinite(g["B_break"].astype(float)).sum() >= 3]
    pairs = [p for p in pairs if p in set(h["pair"])]
    if not pairs:
        raise ValueError("no pairs with a crossover inside the budget range in stage %s" % stage)
    fig, axs = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 3.0), sharey=True, squeeze=False)
    lo, hi = float(h["B_lo"].iloc[0]), float(h["B_hi"].iloc[0])
    for ax, p in zip(axs[0], pairs):
        d = h[h["pair"] == p]
        for fam, gf in d.groupby("family"):
            mk = FAMILY_MARKER.get(fam, "o")
            x = gf["lam_lc"].astype(float)
            bb = gf["B_break"].astype(float)
            fin = np.isfinite(bb)
            ax.scatter(x[fin], bb[fin], marker=mk, s=16, color=F.INK2, alpha=0.75, linewidths=0, label=fam)
            ax.scatter(x[bb == np.inf], np.full(int((bb == np.inf).sum()), hi), marker="^", s=16, facecolors="none",
                       edgecolors=F.INK2, linewidths=0.6)
            ax.scatter(x[bb == -np.inf], np.full(int((bb == -np.inf).sum()), lo), marker="v", s=16,
                       facecolors="none", edgecolors=F.INK2, linewidths=0.6)
        s, c, k = power_law(d)
        if math.isfinite(s):
            xx = np.geomspace(d["lam_lc"].astype(float).min(), d["lam_lc"].astype(float).max(), 20)
            ax.plot(xx, 10 ** (c + s * np.log10(xx)), color=F.INK, lw=1.2,
                    label=r"fit: B $\propto$ $\lambda_{LC}^{%.2f}$" % s)
            xm = float(np.median(np.log10(xx)))
            ax.plot(xx, 10 ** (c + s * xm - 1 * (np.log10(xx) - xm)), color=F.INK, lw=0.8, ls=":",
                    label="first order: slope −1")
        ax.axhline(hi, color=F.GRID, lw=0.8, zorder=0)
        ax.axhline(lo, color=F.GRID, lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        F.plain_log_axis(ax, "x")
        ax.set_ylim(lo / 2, hi * 2)
        ax.set_xlabel(r"$\lambda_{LC}$ (light-cone fault rate of the circuit)")
        ax.set_title(p, loc="left", fontsize=9)
    axs[0][0].set_ylabel("B_break (total shots)")
    hd, lb = axs[0][0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=min(len(lb), 7), frameon=False, bbox_to_anchor=(0.5, -0.06),
               fontsize=7)
    fig.suptitle("Where the first method breaks down against the second · %s · open triangles: on the top edge, "
                 "never inside the budget range; on the bottom edge, from the smallest budget" % stage,
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    return fig


def fig_breakdown_map(H: pd.DataFrame, stage: str, pair: str = "%s vs %s" % PRIMARY):
    """One marker per cell at (two-qubit gates, noise scale s); fill = log10 B_break on a sequential scale (dark: local
    folding breaks down at a small budget); open ○: local folding is not worse at the largest budget; ×: global is
    better from the smallest budget. Marker shape = family."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    h = H[(H["stage"] == stage) & (H["pair"] == pair)]
    if h.empty:
        raise ValueError("no %s cells in stage %s" % (pair, stage))
    lo, hi = float(h["B_lo"].iloc[0]), float(h["B_hi"].iloc[0])
    cmap = LinearSegmentedColormap.from_list("seq", ["#0b2a4a", GLOBAL_COL, "#cfe0f5"])
    norm = Normalize(math.log10(lo), math.log10(hi))
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    rng = np.random.default_rng(0)
    for fam, gf in h.groupby("family"):
        mk = FAMILY_MARKER.get(fam, "o")
        x = gf["two_qubit"].astype(float) * np.exp((rng.random(len(gf)) - 0.5) * 0.12)
        y = gf["scale"].astype(float)
        bb = gf["B_break"].astype(float)
        fin = np.isfinite(bb)
        ax.scatter(x[fin], y[fin], c=np.log10(bb[fin]), cmap=cmap, norm=norm, marker=mk, s=30, edgecolors=F.INK,
                   linewidths=0.3, label=fam)
        ax.scatter(x[bb == np.inf], y[bb == np.inf], marker=mk, s=30, facecolors="#ffffff", edgecolors=F.INK2,
                   linewidths=0.6)
        ax.scatter(x[bb == -np.inf], y[bb == -np.inf], marker="x", s=22, color=F.INK, linewidths=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    F.plain_log_axis(ax, "x")
    sc = np.sort(h["scale"].astype(float).unique())
    ax.set_yticks(sc)
    ax.set_yticklabels(["%g" % v for v in sc])
    ax.minorticks_off()
    ax.set_xlabel("two-qubit gates of the circuit")
    ax.set_ylabel("noise scale s")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$ B_break (open: never inside the range; ×: from the smallest budget)", fontsize=7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=6, handletextpad=0.2,
              columnspacing=0.8)
    ax.set_title("%s · %s" % (pair, stage), loc="left", fontsize=9)
    fig.tight_layout()
    return fig


def fig_bias(Fo: pd.DataFrame, stage: str, methods=("ZNE-G", "ZNE-L", "ZNE-Lf"), min_raw: float = 2e-3):
    """Measured b/b_RAW against λ_LC per method; the first-order predictions ρ as horizontal guides (median per
    method). The guide is a first-order prediction for local folding; for global folding λ_LC over-counts the faults
    that reach a local observable (module docstring), and its guide is drawn for comparison only. Points far from
    their guide at large λ_LC are the higher-order terms of E(λ)."""
    import matplotlib.pyplot as plt
    d = Fo[(Fo["stage"] == stage) & Fo["method"].isin(methods) & np.isfinite(Fo["b_over_raw"])
           & (Fo["b_raw"].abs() >= min_raw)]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for k in methods:
        g = d[d["method"] == k]
        if g.empty:
            continue
        col, mk, ls = _fold_style(g["folding"].iloc[0], g["lam_est"].iloc[0])
        col, mk = METHOD_STYLE.get(k, (col, mk))
        face = col if g["lam_est"].iloc[0] == "gates" else "none"
        rho = float(np.nanmedian(g["rho"]))
        note = ", from λ_LC: not predictive" if g["folding"].iloc[0] == "global" and g["lam_est"].iloc[0] == "gates" \
            else ""
        ax.scatter(g["lam_lc"].astype(float), g["b_over_raw"], marker=mk, s=12, facecolors=face, edgecolors=col,
                   linewidths=0.6, alpha=0.7, label="%s (ρ ≈ %.2f%s)" % (k, rho, note))
        if np.isfinite(g["rho"]).any():
            ax.axhline(float(np.nanmedian(g["rho"])), color=col, lw=0.9, ls=ls)
    ax.axhline(0, color=F.INK2, lw=0.6)
    ax.set_xscale("log")
    F.plain_log_axis(ax, "x")
    ax.set_xlabel(r"$\lambda_{LC}$")
    ax.set_ylabel(r"$b\,/\,b_{\rm RAW}$ (share of the raw bias left)")
    ax.set_title("Bias of the extrapolation · %s" % stage, loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    return fig


# fixed colour and marker per method where one figure shows several local variants (never cycled)
METHOD_STYLE = {"ZNE-Lq": ("#eb6834", "o"), "ZNE-Lq-nest": ("#1baf7a", "^"), "ZNE-Lq-avg4": ("#4a3aa7", "D"),
                "ZNE-Lq-avg16": ("#4a3aa7", "d"), "ZNE-Lq-bal": ("#eda100", "v"), "ZNE-Lq-rand": ("#9a9994", "x")}
FRACTIONAL_COLORS = {"ZNE-Lq": "#eb6834", "ZNE-Lq-exp": "#4a3aa7", "ZNE-Lq-lin": "#1baf7a", "ZNE-Gq": "#2a78d6",
                     "ZNE-Lq-R": "#9a9994"}         # fixed slots (never cycled); others fall back to grey


def fig_fractional_regime(H: pd.DataFrame, stage: str, reference: str = "ZNE-G"):
    """Where fractional ZNE pays: (a) the RMSE ratio against the reference at the largest budget, every cell
    against its light-cone fault count λ_LC (log–log; below 1: the fractional method wins), binned medians as
    lines; (b) the share of cells won per λ_LC band."""
    import matplotlib.pyplot as plt
    h = H[(H["stage"] == stage) & (H["glob"] == reference) & H["local"].isin(list(FRACTIONAL_COLORS))
          & np.isfinite(H["ratio_hi"]) & (H["ratio_hi"] > 0)]
    if h.empty:
        raise ValueError("no fractional method against %s in %s" % (reference, stage))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw=dict(width_ratios=(1.35, 1)))
    edges = np.array([b for b in LC_BANDS if math.isfinite(b) and b > 0] + [float(h["lam_lc"].max()) * 1.01])
    edges = np.concatenate([[float(h["lam_lc"].min()) * 0.99], edges[edges > float(h["lam_lc"].min())]])
    mids = np.sqrt(edges[:-1] * edges[1:])
    keys = [k for k in FRACTIONAL_COLORS if k in set(h["local"]) and k != "ZNE-Lq-R"]
    width = 0.8 / max(len(keys), 1)
    for i, k in enumerate(keys):
        g = h[h["local"] == k]
        col = FRACTIONAL_COLORS[k]
        ax.scatter(g["lam_lc"].astype(float), g["ratio_hi"], s=9, color=col, alpha=0.35, linewidths=0)
        idx = np.digitize(g["lam_lc"].astype(float), edges[1:-1])
        med = [np.median(g["ratio_hi"][idx == j]) if (idx == j).any() else np.nan for j in range(len(mids))]
        ax.plot(mids, med, color=col, lw=2, label=k)
        win = [float((g["ratio_hi"][idx == j] < 1).mean()) if (idx == j).any() else np.nan for j in range(len(mids))]
        bx.bar(np.arange(len(mids)) + (i - (len(keys) - 1) / 2) * width, win, width=width * 0.92, color=col,
               label=k)
    ax.axhline(1, color=F.INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    F.plain_log_axis(ax, "x")
    F.plain_log_axis(ax, "y")
    ax.set_xlabel(r"light-cone fault count $\lambda_{LC}$")
    ax.set_ylabel("RMSE / RMSE of %s at the largest budget" % reference)
    ax.set_title("Fractional ZNE (λ ≤ 2) against integer global folding · %s" % stage, loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    bx.set_xticks(np.arange(len(mids)))
    bx.set_xticklabels(["%.2g–%.2g" % (a, b) for a, b in zip(edges[:-1], edges[1:])], rotation=30, ha="right",
                       fontsize=7)
    bx.axhline(0.5, color=F.INK2, lw=0.8, ls=":")
    bx.set_ylim(0, 1)
    bx.set_xlabel(r"$\lambda_{LC}$ band")
    bx.set_ylabel("share of cells won")
    fig.tight_layout()
    return fig


def make_figures(T, rep: Report, sink, stages=None) -> list:
    """Every figure of the study from the tables of ``rep`` (tables in, figures out)."""
    Fo, H, D = folding_table(T), rep.tables.get("head_to_head"), rep.tables.get("depth")
    saved = []

    def put(fig, name):
        saved.append(sink.save(fig, name))
    if Fo is not None and not Fo.empty and Fo["lams_eff"].notna().any():
        st0 = sorted(Fo.loc[Fo["lams_eff"].notna(), "stage"].unique())[0]       # one stage: each circuit once
        put(fig_lambda_eff(Fo, st0), "folding_lambda_eff")
    if D is not None and not D.empty:
        put(fig_depth(D), "folding_depth")
    if H is None or H.empty:
        return saved
    for st in stages or sorted(H["stage"].unique()):
        hs = H[H["stage"] == st]
        if hs.empty:
            continue
        chis = sorted(hs["coherent_fraction"].astype(float).unique()) if "coherent_fraction" in hs else [None]
        for chi in chis:
            sub = None if len(chis) == 1 else chi
            for a, g in [p for p in PAIRS if p in MAPPED]:
                if ((hs["local"] == a) & (hs["glob"] == g)).any():
                    try:
                        put(fig_crossover_maps(T, hs, st, (a, g), coherent_fraction=sub),
                            "folding_map_%s_%s_vs_%s%s" % (_slug(st), a, g, "" if sub is None else "_chi%g" % sub))
                    except ValueError:
                        pass
        try:
            put(fig_breakdown(H, st), "folding_breakdown_%s" % _slug(st))
        except ValueError:                               # no pair crosses inside the budget range
            pass
        if (hs["pair"] == "%s vs %s" % PRIMARY).any():
            put(fig_breakdown_map(H, st), "folding_breakdown_map_%s" % _slug(st))
        if Fo is not None and not Fo.empty and (Fo["stage"] == st).any():
            have = set(Fo.loc[Fo["stage"] == st, "method"])
            ms = tuple(k for k in ("ZNE-G", "ZNE-L", "ZNE-Lf", "ZNE-Lq", "ZNE-Lq-nest", "ZNE-Lq-avg4") if k in have)[:4]
            put(fig_bias(Fo, st, methods=ms), "folding_bias_%s" % _slug(st))
        try:
            put(fig_fractional_regime(H, st), "folding_fractional_regime_%s" % _slug(st))
        except ValueError:                               # no fractional method in this stage
            pass
    return saved


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))


# ------------------------------------------------------------------ driver: input → computation → output
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="the run directory (contains tables/)")
    ap.add_argument("--out", default="figs_zne_folding")
    ap.add_argument("--policy", default="fixed", choices=("fixed", "opt"))
    ap.add_argument("--stage", action="append", help="figures for these stages only (default: all)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    T = F.load(a.data)                                                     # input
    rep = analyse(T, policy=a.policy)                                      # computation
    DirectoryReportSink(os.path.join(a.out, "tables")).write(rep)         # output
    if not a.no_figures:
        F.setup_matplotlib()
        make_figures(T, rep, DirectoryFigureSink(a.out, a.formats), a.stage)
    sys.stdout.write(rep.render_text())
    print("tables and report written to %s" % os.path.join(a.out, "tables"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
