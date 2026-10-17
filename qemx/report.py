"""Cross-page results derived from one comparison, without Streamlit: what the Cost-vs-precision page shows for a
comparison, the PEC wall and the bias–variance split. The Playground's and the IBM Quantum page's "Download all"
compute these on the fly, so one ZIP holds every table and figure a result supports, whichever tab it came from.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import analysis as A

DEFAULTS = dict(eps=0.01, budget=600.0, nmin=100, nmax=10 ** 8, nq=10, p1=5e-4, p2s="0.002, 0.005, 0.01, 0.02",
                eps_w=0.01, L=60, con="inverse", bud=3600.0, t=250.0)


def cost_rows(res: dict) -> list:
    """The comparison's rows reduced to the cost model of each method (bias, per-shot v, training/calibration)."""
    N0 = int(res["shots"])
    return [dict(method=r["method"], family=r["family"], bias=r["bias"], v=A.per_shot_variance(r, N0),
                 fixed_shots=r.get("fixed_shots", 0), cal_circuits=r.get("cal_circuits", 0),
                 w_cal=r.get("w_cal", 0.0), u_train=r.get("u_train", 0.0),
                 n_circuits=r.get("n_circuits", 1), accepted=r.get("accepted", 1.0))   # CostModel.of_row: n_min
            for r in res["rows"] if np.isfinite(r.get("estimate", np.nan))]


def _range(nmin, nmax):
    nmin = max(int(nmin), 1)
    nmax = int(nmax) if int(nmax) > nmin else nmin * 10
    return nmin, nmax


def cost_precision(res: dict, eps: float, budget_s: float, nmin: int, nmax: int, t_shot_us: float,
                   dark: bool = False) -> tuple:
    """({figure}, {table}) of the Cost-vs-precision page's "Your circuit" tab for a comparison result."""
    from .viz import analysis_charts as AC
    figs, tables = {}, {}
    rows = cost_rows(res)
    if not rows:
        return figs, tables
    nmin, nmax = _range(nmin, nmax)
    shots = np.logspace(math.log10(nmin), math.log10(nmax), 160)
    n_budget = budget_s / (t_shot_us * 1e-6) if budget_s and t_shot_us else None
    raw = next((r for r in rows if r["method"] == "Unmitigated"), None)
    cross = []
    if raw:
        for r in rows:
            if r is raw:
                continue
            x = A.crossover_rows(r, raw)
            if x and nmin <= x <= nmax:
                cross.append(dict(method=r["method"], N=x, rmse=float(A.rmse_row(r, [x])[0])))
    figs["rmse_vs_shots"] = AC.rmse_vs_shots(rows, shots, eps, n_budget, dark=dark, t_shot_us=t_shot_us,
                                             crossings=cross)
    eps_grid = np.logspace(math.log10(eps) - 1.5, math.log10(eps) + 1.5, 120)
    figs["time_needed"] = AC.shots_needed(rows, eps_grid, t_shot_us, dark=dark, budget_s=budget_s or None)
    tab = []
    for r in rows:
        Ne = A.budget_row(r, eps)
        x = A.crossover_rows(r, raw) if raw and r is not raw else None
        t_need = Ne * t_shot_us * 1e-6 if math.isfinite(Ne) else math.inf
        tab.append({"method": r["method"], "bias": r["bias"], "per-shot σ": math.sqrt(r["v"]),
                    "budget for ε": Ne, "QPU time for ε (s)": t_need,
                    "within budget": "yes" if (budget_s and t_need <= budget_s) else ("no" if budget_s else "–"),
                    "beats unmitigated above B": x})
    tables["cost_for_target"] = pd.DataFrame(tab)
    curve = {"budget B (shots)": shots}
    for r in rows:
        curve["RMSE · " + r["method"]] = A.rmse_row(r, shots)
    tables["rmse_vs_budget"] = pd.DataFrame(curve)
    return figs, tables


def parse_p2s(text: str, default=(0.01,)):
    """Comma-separated two-qubit error rates in (0, 1); ``default`` when the text is not such a list."""
    try:
        p2s = sorted({float(x) for x in str(text).replace(";", ",").split(",") if x.strip()})
        if not p2s or any(not (0 < v < 1) for v in p2s):
            raise ValueError
        return p2s
    except ValueError:
        return list(default) if default is not None else None


def pec_wall(nq: int, p1: float, p2s, eps: float, L: int, con: str, budget_s: float, t_shot_us: float,
             dark: bool = False) -> tuple:
    """({figure}, {table}) of the PEC wall: Γ, shots and QPU time against depth for each two-qubit error rate."""
    from .viz import analysis_charts as AC
    p2s = parse_p2s(p2s) if isinstance(p2s, str) else list(p2s)
    L = max(int(L), 1)
    layers = np.unique(np.linspace(0, L, min(L, 60) + 1).round().astype(int))
    wall = A.pec_wall(layers, p2s, int(nq), float(p1), float(eps), t_shot_us, con)
    figs = {"pec_wall": AC.pec_wall_chart(layers, wall, budget_s or None, dark=dark)}
    p2grid = np.logspace(math.log10(min(p2s) / 3), math.log10(max(p2s) * 2), 40)
    figs["pec_wall_map"] = AC.wall_heatmap(layers, p2grid, int(nq), float(p1), float(eps), t_shot_us, budget_s, con,
                                           dark=dark)
    rows = []
    for p2, d in sorted(wall.items()):
        ok = np.where(d["seconds"] <= (budget_s or np.inf))[0]
        rows.append({"p₂": p2, "γ per layer": float(d["gamma"][1]) ** (1 / max(int(layers[1]), 1))
                     if len(layers) > 1 else 1.0,
                     "Γ at max depth": float(d["gamma"][-1]), "shots at max depth": float(d["shots"][-1]),
                     "deepest affordable (layers)": int(layers[ok[-1]]) if len(ok) else 0})
    tables = {"pec_wall": pd.DataFrame(rows),
              "pec_wall_curves": pd.DataFrame([{"p2": p2, "layers": int(l_), "Gamma": float(g), "shots": float(n_),
                                                "seconds": float(s)} for p2, d in wall.items()
                                               for l_, g, n_, s in zip(layers, d["gamma"], d["shots"], d["seconds"])])}
    return figs, tables, wall, layers


def bias_variance(res: dict) -> pd.DataFrame:
    """MSE = bias² + variance at the comparison's budget, per method (the one-observable bullseye)."""
    out = []
    for r in res["rows"]:
        sd = float(r.get("sd", np.nan))
        if not np.isfinite(r.get("estimate", np.nan)):
            continue
        b2, var = float(r["bias"]) ** 2, sd * sd if np.isfinite(sd) else math.inf
        mse = b2 + var
        out.append({"method": r["method"], "bias": r["bias"], "sd": sd, "bias²": b2, "variance": var, "MSE": mse,
                    "bias share": b2 / mse if (np.isfinite(mse) and mse > 0) else np.nan})
    return pd.DataFrame(out)


def cross_tab(res: dict, params: dict | None = None, extras: dict | None = None) -> tuple:
    """(figures, tables) for a ZIP: the cost-vs-precision analysis of ``res``, the PEC wall for the page's settings,
    the bias–variance split, and the stored results of other pages (``extras``: depth sweep rows, bullseye,
    head-to-head, mismatch, batch tables) when they exist. Never raises: a part that fails is skipped."""
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None})
    figs, tables = {}, {}
    parts = [("cost_precision", lambda: cost_precision(res, float(p["eps"]), float(p["budget"]), int(p["nmin"]),
                                                       int(p["nmax"]), float(p["t"]))),
             ("pec_wall", lambda: pec_wall(int(p["nq"]), float(p["p1"]), p["p2s"], float(p["eps_w"]), int(p["L"]),
                                           p["con"], float(p["bud"]), float(p["t"]))[:2]),
             ("bias_variance", lambda: ({}, {"bias_variance": bias_variance(res)}))]
    for name, fn in parts:
        try:
            f, t = fn()
        except Exception:                                    # noqa: BLE001 - the rest of the ZIP still ships
            continue
        figs.update({"%s/%s" % (name, k): v for k, v in f.items()})
        tables.update({"%s/%s" % (name, k): v for k, v in t.items()})
    for name, (f, t) in (extras or {}).items():
        figs.update({"%s/%s" % (name, k): v for k, v in (f or {}).items()})
        tables.update({"%s/%s" % (name, k): v for k, v in (t or {}).items()})
    return figs, tables
