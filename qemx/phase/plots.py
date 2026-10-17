r"""Static figures of Direction A data (matplotlib; for reports and the thesis, not the app).

* :func:`rmse_curves` — RMSE(B) of every method of one cell (integer ledger), with the B*(ε) markers;
* :func:`winner_map` — the phase diagram of one (stage, family, n, engine, readout, χ) slice: the winning method on the
  (λ_LC, B) plane, one column per cell.

Colors follow the app's method-family hues (``qemx.viz.theme.FAMILY``, validated for color-vision deficiency);
variants within a family differ in lightness and are always named in the legend, so color is never the only cue.
"""
from __future__ import annotations

import math

import numpy as np

STYLE = {  # method key -> (color, marker, label); the palette of scripts/dirA_figures.py (CVD-checked adjacent pairs)
    "ZNE-R": ("#2a78d6", "s", "ZNE Richardson"),
    "PEC": ("#eb6834", "D", "PEC inverse"),
    "CDR": ("#1baf7a", "^", "CDR (OLS)"),
    "FR-R": ("#eda100", "v", "FR Ridge"),
    "VD": ("#e87ba4", "h", "VD M=2 (optimistic)"),
    "FR-L": ("#008300", "P", "FR Lasso"),
    "ZNE-E": ("#4a3aa7", "o", "ZNE exponential"),
    "RAW": ("#9a9994", "x", "Unmitigated"),
    "NIL-G2": ("#e87ba4", "*", "NIL ℓ1≤2 (Chen et al.)"),   # VD's slot: the NIL study's maps leave VD out
    # runs 1-2 keyed feature regression NIL-L / NIL-R (estimator "learning", not Chen et al.'s NIL)
    "NIL-L": ("#008300", "P", "FR Lasso (key NIL-L)"),
    "NIL-R": ("#eda100", "v", "FR Ridge (key NIL-R)"),
}
_EXTRA_MARKERS = "<>pd1234+|_"


def _style(k):
    """A method's (color, marker, label); a key without a slot gets dark grey, a marker of its own and its key as the
    label (it is still named in the legend)."""
    if k in STYLE:
        return STYLE[k]
    return ("#52514e", _EXTRA_MARKERS[sum(map(ord, str(k))) % len(_EXTRA_MARKERS)], str(k))


def rmse_curves(rec: dict, policy="fixed", budgets=None, ax=None, eps=(0.01,)):
    import matplotlib.pyplot as plt
    from .cell import cost_of
    from .ledger import budget_for, rmse
    if isinstance(policy, str):
        from .store import cell_policies
        pol = cell_policies(rec, [policy])[0]           # the cell's own lower bounds, as in collect
    else:
        pol = policy
    budgets = np.unique(np.round(np.logspace(3, 8, 121)).astype(np.int64)) if budgets is None else budgets
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.0))
    for r in rec["methods"]:
        c = cost_of(r)
        if c is None:
            continue
        col, mk, lab = _style(r["key"])
        y = np.array([rmse(c, int(B), pol) for B in budgets])
        ax.plot(budgets, np.where(np.isfinite(y), y, np.nan), color=col, lw=2, label=lab)
        for e in eps:
            Bs = budget_for(c, e, pol)
            if math.isfinite(Bs) and Bs <= budgets[-1]:
                ax.plot([Bs], [e], marker=mk, color=col, ms=7, mec="white", mew=1)
    for e in eps:
        ax.axhline(e, color="#5E5E5E", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total shots B")
    ax.set_ylabel("RMSE")
    c = rec["cell"]
    ax.set_title("%s n=%d · %s · s=%g · λ_LC=%.3f · %s policy" % (c["family"], c["n"], c["engine"], c["scale"],
                                                                 rec["lambda"]["lam_lc"], pol.name), fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, which="major", color="#E6E6E6", lw=0.6)
    return ax


def winner_map(winners_df, methods_df, *, stage, family, n, policy="fixed", candidates="all", ax=None,
               fparams=None):
    """Tiles: x = λ_LC of each cell (column widths to the midpoints in log λ), y = budget; color = winner.
    ``candidates``: "all" or "no_optimistic" (the winners table has one row per candidate set)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    cells = methods_df[(methods_df.stage == stage) & (methods_df.family == family) & (methods_df.n == n)]
    if fparams is not None:
        cells = cells[cells.fparams == fparams]
    lam = cells.groupby("cell_id").lam_lc.first().sort_values()
    w = winners_df[(winners_df.cell_id.isin(lam.index)) & (winners_df.policy == policy)
                   & (winners_df.candidates == candidates)]
    if w.duplicated(["cell_id", "B"]).any():
        raise ValueError("several winner rows per (cell, budget): the cells of this slice are not unique")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.0))
    if lam.empty or w.empty:
        return ax
    lx = np.log10(lam.values)
    edges = np.concatenate([[lx[0] - 0.05], (lx[1:] + lx[:-1]) / 2, [lx[-1] + 0.05]])
    Bs = np.sort(w.B.unique())
    ly = np.log10(Bs)
    yedges = np.concatenate([[ly[0] - (ly[1] - ly[0]) / 2], (ly[1:] + ly[:-1]) / 2, [ly[-1] + (ly[-1] - ly[-2]) / 2]])
    seen = set()
    for i, cid in enumerate(lam.index):
        wc = w[w.cell_id == cid].set_index("B")
        for j, B in enumerate(Bs):
            if B not in wc.index:
                continue
            k = wc.loc[B, "winner"]
            seen.add(k)
            ax.add_patch(Rectangle((10 ** edges[i], 10 ** yedges[j]), 10 ** edges[i + 1] - 10 ** edges[i],
                                   10 ** yedges[j + 1] - 10 ** yedges[j], color=_style(k)[0], lw=0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** edges[0], 10 ** edges[-1])
    ax.set_ylim(10 ** yedges[0], 10 ** yedges[-1])
    ax.set_xlabel("λ_LC (light-cone fault rate)")
    ax.set_ylabel("total shots B")
    ax.set_title("winner · %s · %s n=%d · %s policy" % (stage, family, n, policy), fontsize=10)
    order = [k for k in STYLE if k in seen] + sorted(k for k in seen if k not in STYLE)
    ax.legend(handles=[Patch(color=_style(k)[0], label=_style(k)[2]) for k in order], fontsize=8,
              frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ax
