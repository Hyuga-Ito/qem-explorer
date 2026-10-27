r"""Fits of the two-constant ledger, :math:`\mathrm{RMSE}_i(\lambda,B)=\sqrt{b_i(\lambda)^2+A_i e^{\beta_i\lambda}/B}`
(hypothesis A1 of the research plan), with cell-bootstrap confidence intervals.

* **Variance law.** For every method the *effective variance coefficient* of a cell is
  :math:`V=\lim_{B\to\infty}B\,[\mathrm{Var}(B)-u]=(\sqrt v+\sqrt{c\,w})^2` (fixed policy: training shots are a
  constant offset), so :math:`\ln V=\ln A+\beta\lambda` is fitted by least squares on the cells of a group. For the
  learning methods :math:`u` is reported separately (it does not scale with B), and :math:`v` and :math:`w` are
  those of the method row: the reference fit (exact training features) for v, the protocol's training level for w
  — not the averages over the training noise that the ledger's levels carry.
* The bias used is the row's ``b``: for the learning methods the bias at *infinite* training shots (``b_inf``); the
  finite-shot bias of each training level is in the levels table.
* **Bias law.** :math:`b(\lambda)=\beta_1\lambda+\beta_2\lambda^2` through the origin (no noise, no bias), by least
  squares on the signed bias.

Confidence intervals resample cells with replacement (``boot`` replicates, percentile 2.5–97.5 %); the seed is fixed
so the numbers are reproducible. Groups with fewer than ``min_cells`` cells or a λ range narrower than ``min_span``
are reported with ``ok = False`` instead of a fit.
"""
from __future__ import annotations

import math

import numpy as np


def v_eff(v: float, c: int, w: float) -> float:
    return (math.sqrt(max(v, 0.0)) + math.sqrt(max(c * w, 0.0))) ** 2


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_variance_law(lam, V, boot: int = 2000, seed: int = 0, min_cells: int = 6, min_span: float = 0.2) -> dict:
    """ln V = ln A + β λ. Returns A, β, their bootstrap CIs, R², and the residual SD of ln V."""
    lam, V = np.asarray(lam, float), np.asarray(V, float)
    ok = np.isfinite(lam) & np.isfinite(V) & (V > 0)
    lam, y = lam[ok], np.log(V[ok])
    if len(lam) < min_cells or np.ptp(lam) < min_span:
        return dict(ok=False, cells=int(len(lam)), span=float(np.ptp(lam)) if len(lam) else 0.0)
    X = np.column_stack([np.ones_like(lam), lam])
    coef = _ols(X, y)
    res = y - X @ coef
    r2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum()) if np.ptp(y) > 0 else 1.0
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(int(boot)):
        i = rng.integers(0, len(lam), len(lam))
        if np.ptp(lam[i]) <= 0:
            continue
        bs.append(_ols(X[i], y[i]))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(ok=True, cells=int(len(lam)), A=float(math.exp(coef[0])), beta=float(coef[1]),
                A_ci=[float(math.exp(lo[0])), float(math.exp(hi[0]))], beta_ci=[float(lo[1]), float(hi[1])],
                r2=float(r2), resid_sd=float(np.std(res, ddof=2)) if len(lam) > 2 else 0.0)


def fit_bias_law(lam, b, boot: int = 2000, seed: int = 0, min_cells: int = 6) -> dict:
    """b = β1 λ + β2 λ² through the origin."""
    lam, b = np.asarray(lam, float), np.asarray(b, float)
    ok = np.isfinite(lam) & np.isfinite(b)
    lam, b = lam[ok], b[ok]
    if len(lam) < min_cells:
        return dict(ok=False, cells=int(len(lam)))
    X = np.column_stack([lam, lam ** 2])
    coef = _ols(X, b)
    res = b - X @ coef
    rng = np.random.default_rng(seed)
    bs = np.array([_ols(X[i], b[i]) for i in (rng.integers(0, len(lam), len(lam)) for _ in range(int(boot)))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(ok=True, cells=int(len(lam)), b1=float(coef[0]), b2=float(coef[1]), b1_ci=[float(lo[0]), float(hi[0])],
                b2_ci=[float(lo[1]), float(hi[1])], resid_sd=float(np.std(res, ddof=2)) if len(lam) > 2 else 0.0)


GROUP = ("stage", "device", "opt_level", "engine", "readout", "coherent_fraction", "family", "n", "method")


def fit_table(methods_df, group=GROUP, boot: int = 2000):
    """One row per group: the variance law and the bias law. ``methods_df`` is tables/methods.csv. The default
    grouping keeps stages, devices and optimization levels apart: cells of different stages may describe the same
    physics (e.g. the sensitivity stage repeats primary cells) and must not be pooled (pseudo-replication)."""
    import pandas as pd
    df = methods_df[methods_df.status == "ok"].copy()
    df["V"] = [v_eff(v, c, w) for v, c, w in zip(df.v, df.c.fillna(0), df.w.fillna(0))]
    rows = []
    for key, g in df.groupby(list(group), dropna=False):
        fv = fit_variance_law(g.lam_lc.values, g.V.values, boot=boot)
        fb = fit_bias_law(g.lam_lc.values, g.b.values, boot=boot)
        row = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        row.update({"var_" + k: v for k, v in fv.items()})
        row.update({"bias_" + k: v for k, v in fb.items()})
        rows.append(row)
    return pd.DataFrame(rows)
