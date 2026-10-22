#!/usr/bin/env python3
r"""Thesis figures: Direction A (equal-budget phase diagrams) and the hardware divergence study.

    python scripts/dirA_figures.py data/dirA_v1 --out figs            # health report + the default figure set
    python scripts/dirA_figures.py data/dirA_v1 --out figs --only health
    python scripts/dirA_figures.py data/dirA_v1 --out figs --only phase --eps 0.01   # with the target overlay
    python scripts/dirA_figures.py data/dirA_v1 --out figs --font lm  # Latin Modern (a LaTeX thesis in lmodern)

The divergence figures are drawn by ``scripts/dirA_divergence.py analyze`` and ``figures``; from a notebook every
figure function returns a matplotlib figure and writes nothing::

    import dirA_figures as F, dirA_io as IO
    F.setup_matplotlib("times")                     # once: fonts, sizes, line weights of the thesis
    T = F.load("data/dirA_v1")                      # or F.load(IO.MemorySource({...tables...}))
    fig = F.phase_grid(T, "A1-primary", n=6)
    IO.DirectoryFigureSink("figs").save(fig, "phase_qft6")      # PDF (vector, fonts embedded) and PNG

Direction A figures
-------------------
phase_map       winner of the equal-budget comparison on the (noise scale s, total shots B) plane for one circuit
phase_grid      the same for the representative circuits of one size, side by side, one shared legend
family_map      one family pooled over its circuits on the (λ_LC, B) plane; hatched where < 2/3 of them agree
rmse_curves     RMSE(B) of every method in one cell, fixed and opt training policies side by side
budget_curves   B*(ε): the smallest budget reaching RMSE ≤ ε, against λ_LC
bias_fraction   signed residual bias b/|b_RAW| of each method against λ_LC
transfer_grid   stage C: the same circuits on fake_fez and the three transfer devices
validation      z-scores of the Monte-Carlo validation against N(0, 1)

Divergence figures — "the gap between a component noise model and the device"
------------------------------------------------------------------------------
fidelity_gap                the headline: observed and predicted F_c against two-qubit gates with the fitted decays
                            and their effective error rates ε_eff (after Chen et al. 2024, Fig. 9)
difference_volumetric       predicted − observed F_c on the width × two-qubit-gate plane (Chen et al. 2024, Fig. 11),
                            log₂ or √2 bins, a significance dot per tile; circuits and labels on request
difference_volumetric_grid  every mitigation method on one colour scale (equal budget)
predicted_vs_observed       predicted against observed F_c, coloured by significance, shaped by family
divergence_budget           which mechanism carries the unmodeled error: overall and per family
ladder_chart                the χ² ladder: does the gap need context, or only a rescaled calibration?
folding_fingerprint         global against local folding anomaly: gate-, time- or repetition-attached
heavy_hex_map               where on the chip the model under-predicts
context_probe               the same two-qubit core with idle / superposed / driven spectators
diagnostic_matrix           which diagnostic names the planted (emulation) or real (device) cause

Design (one grammar for every figure)
-------------------------------------
* Sizes are the thesis's: full text width 6.3 in, one column 3.1 in, 8 pt text, so a figure is placed at 100 % and
  its text matches the caption. Fonts: ``times`` (TeX Gyre Termes + STIX maths — Times-like, default), ``lm`` (Latin
  Modern, for lmodern theses), ``sans`` (TeX Gyre Heros / Liberation Sans, for slides). Japanese text falls back to
  Noto CJK JP. PDF output embeds TrueType fonts (Type 42).
* Colour does one job per figure, and the job fixes the scale: identity of a mitigation method → the categorical
  palette of the dataviz method (adjacent pairs validated for colour-vision deficiency; fixed order, never cycled;
  on maps every region also carries its abbreviation); polarity of a divergence → a blue ↔ grey ↔ red diverging scale
  with mirrored OKLab lightness (red = device worse than its model, blue = better, grey = as modeled); magnitude of an
  unmodeled error → the red arm alone. Circuit families are marker shapes, never colours.
* Significance is drawn, not coloured: a filled mark is beyond the shot noise (|z| > 3), an open one within it.
* Recessive axes (no top/right spines, thin grey ticks), short left-aligned panel titles; the explanation belongs in
  the caption. Print figures are light-only.
Requires pandas, numpy, matplotlib, scipy; qemx (optional) gives exact B*(ε) instead of grid interpolation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dirA_io import DirectoryFigureSink, Report, source  # noqa: E402

# ------------------------------------------------------------------ style: methods
METHODS = ["ZNE-R", "PEC", "CDR", "FR-R", "VD", "FR-L", "ZNE-E", "RAW"]            # fixed order = palette slots
STYLE = {   # key: (colour, marker, linestyle, legend label)
    "ZNE-R": ("#2a78d6", "s", "-", "ZNE · Richardson"),
    "PEC":   ("#eb6834", "D", "-", "PEC · inverse"),
    "CDR":   ("#1baf7a", "^", "-", "CDR · OLS"),
    "FR-R":  ("#eda100", "v", "-", "FR · Ridge"),
    "VD":    ("#e87ba4", "h", "-", "VD (M=2, upper bound)"),
    "FR-L":  ("#008300", "P", "-", "FR · Lasso"),
    "ZNE-E": ("#4a3aa7", "o", "-", "ZNE · exponential"),
    "RAW":   ("#9a9994", "x", "--", "Unmitigated"),
    # neighbor-informed learning (Chen et al.) takes VD's slot in the NIL study's maps, where VD — an optimistic
    # upper bound — is left out (use_methods(NIL_MAP_METHODS)); the two never share a figure
    "NIL-G2": ("#e87ba4", "*", "-", "NIL · ℓ1 ≤ 2 (Chen)"),
}
NIL_MAP_METHODS = ["ZNE-R", "PEC", "CDR", "FR-R", "NIL-G2", "FR-L", "ZNE-E", "RAW"]
# Runs 1 and 2 (direction_A_v1) keyed feature regression as NIL-L / NIL-R; the thesis calls it FR-L / FR-R.
# Keys are renamed on loading wherever the method's estimator is "learning" (never Chen et al.'s "nil").
LEGACY_LEARNING_KEYS = ("NIL-L", "NIL-R", "NIL-L-a3", "NIL-L-a2", "NIL-R-a2", "NIL-R-32")


def use_methods(keys):
    """Set the palette slots (≤ 8, RAW last) of every figure that follows. Colours follow the method, so two methods
    that share a colour may not be used together."""
    keys = list(keys)
    cols = [STYLE[k][0] for k in keys]
    if len(set(cols)) != len(cols):
        raise ValueError("methods %s share a colour" % keys)
    if len(keys) > 8 or keys[-1] != "RAW":
        raise ValueError("at most 8 methods, RAW last")
    METHODS[:] = keys
    return METHODS


def _canonical_keys(T) -> dict:
    """{legacy key: thesis key} for this store: feature-regression keys of runs 1-2 become FR-*."""
    m = T.methods
    est = m.groupby("method")["estimator"].first() if "estimator" in m.columns else {}
    return {k: "FR-" + k[4:] for k in LEGACY_LEARNING_KEYS if k in getattr(est, "index", ()) and est[k] == "learning"}


def style(k):
    return STYLE.get(k, ("#52514e", ".", ":", k))


# ------------------------------------------------------------------ style: page, type, ink
TEXT_WIDTH = 6.3          # in: the thesis's text block (A4, 2.5 cm margins ≈ 16 cm); full-width figures
COLUMN_WIDTH = 3.1        # in: half the text block; side-by-side figures and single-panel plots
INK = "#0b0b0b"           # primary text, data emphasis
INK2 = "#52514e"          # secondary text: tick labels, annotations, legends
MUTED = "#8a8984"         # reference lines (diagonals, fits that are not the message)
BASELINE = "#b5b4af"      # zero / unity lines
GRID = "#e4e3df"          # grid, chip couplers, table rules
SURFACE = "#ffffff"
SERIF_CJK = "Noto Serif CJK JP"
FONT_PRESETS = {
    # name: (rcParams of the family, mathtext fontset)
    "times": ({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "STIXGeneral",
                                                       SERIF_CJK, "DejaVu Serif"]}, "stix"),
    "lm": ({"font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", SERIF_CJK, "DejaVu Serif"]},
           "cm"),
    "sans": ({"font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "Helvetica",
                                                               "Arial", "Noto Sans CJK JP", "DejaVu Sans"]},
             "stixsans"),
}
PRESET = {"name": "times"}


def setup_matplotlib(preset: str | None = None, base: float = 8.0):
    """The thesis style for every figure that follows: ``preset`` (``times`` default, ``lm``, ``sans``; also the
    QEMX_FIG_FONT environment variable) and the base text size in points (8 pt: the size of a caption's small text,
    so a full-width figure needs no scaling). Recessive axes: no top/right spines, thin grey ticks, no grid unless a
    figure asks for one; frameless legends; PDF with embedded TrueType fonts; 300 dpi PNG."""
    import matplotlib as mpl
    name = (preset or os.environ.get("QEMX_FIG_FONT") or "times").lower()
    if name not in FONT_PRESETS:
        raise ValueError("font preset %r: one of %s" % (name, ", ".join(FONT_PRESETS)))
    fam, math_set = FONT_PRESETS[name]
    PRESET["name"] = name
    small = base - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mpl.rcParams.update(fam)
        mpl.rcParams.update({
            "mathtext.fontset": math_set, "axes.unicode_minus": True,
            "font.size": base, "axes.titlesize": base, "axes.labelsize": base, "axes.titleweight": "normal",
            "axes.titlepad": 4.0, "axes.labelpad": 2.5, "xtick.labelsize": small, "ytick.labelsize": small,
            "legend.fontsize": small, "legend.title_fontsize": small, "figure.titlesize": base + 0.5,
            "axes.edgecolor": INK2, "axes.linewidth": 0.5, "axes.labelcolor": INK, "axes.titlecolor": INK,
            "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
            "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.4,
            "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
            "lines.linewidth": 1.1, "lines.markersize": 3.6, "lines.markeredgewidth": 0.6, "patch.linewidth": 0.5,
            "errorbar.capsize": 0, "hatch.linewidth": 0.5,
            "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.3, "legend.columnspacing": 1.0, "legend.labelspacing": 0.3,
            "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
            "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        })
    return name


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02):
    """A bold (a), (b), … in the top-left corner outside the axes (for figures whose panels are cited separately)."""
    ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold",
            fontsize=8, color=INK)


def _ink_on(hexcol: str) -> str:
    """Dark or white label text on a fill (WCAG relative luminance)."""
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if (1.05 / (L + 0.05)) > ((L + 0.05) / (0.0 + 0.05)) else INK


def save(fig, out_dir, name):
    """Kept for notebooks of earlier versions: DirectoryFigureSink(out_dir).save(fig, name)."""
    return DirectoryFigureSink(out_dir).save(fig, name)


# ------------------------------------------------------------------ loading
class Tables:
    """The collected tables plus a per-cell table (one row per cell, with a readable circuit label)."""

    def __init__(self, src, load_ledger: bool = True):
        self.src = source(src)
        self.dir = getattr(self.src, "dir", None)
        self.methods = self.src.table("methods")
        self.winners = self.src.table("winners")
        self.levels = self.src.table("levels", required=False)
        self.validation = self.src.table("validation", required=False)
        self.errors = self.src.table("errors", required=False)
        self.ledger = self.src.table("ledger", columns=["cell_id", "method", "policy", "B", "feasible", "rmse"]) \
            if load_ledger else None
        ren = _canonical_keys(self)
        if ren:
            for df, cols in ((self.methods, ["method"]), (self.winners, ["winner", "runner_up"]),
                             (self.levels, ["method"]), (self.validation, ["method"]), (self.errors, ["method"]),
                             (self.ledger, ["method"])):
                if df is not None and len(df):
                    for c in cols:
                        if c in df.columns:
                            df[c] = df[c].replace(ren)
            if "missing" in self.winners.columns:
                self.winners["missing"] = self.winners["missing"].fillna("").astype(str).map(
                    lambda s: ",".join(ren.get(x, x) for x in s.split(",")) if s else s)
        self.renamed = ren
        m = self.methods
        m["fp"] = m["fparams"].map(json.loads)
        m["circuit"] = [circuit_id(f, n, **p) for f, n, p in zip(m.family, m.n, m.fp)]
        m["label"] = [circuit_label(f, n, p) for f, n, p in zip(m.family, m.n, m.fp)]
        cols = ["cell_id", "stage", "family", "n", "circuit", "label", "device", "engine", "scale", "readout",
                "coherent_fraction", "opt_level", "lam_lc", "lam_total", "two_qubit", "depth", "ideal"]
        self.cells = m[[c for c in cols if c in m.columns]].drop_duplicates("cell_id").set_index("cell_id")


def load(data, load_ledger: bool = True) -> Tables:
    """The tables of a run: ``data`` is a run directory or a dirA_io source."""
    return Tables(data, load_ledger)


def circuit_id(family: str, n: int, **p) -> str:
    """A stable key of a circuit: family, n and its parameters (the cells of one circuit differ only in noise)."""
    return "%s|%d|%s" % (family, int(n), json.dumps({k: p[k] for k in sorted(p)}, sort_keys=True))


def circuit_label(family: str, n: int, p: dict) -> str:
    if family == "qft":
        return "QFT-%d" % n if p.get("m") is None else "AQFT-%d (m=%d)" % (n, p["m"])
    if family == "ghz":
        return "GHZ-%d%s" % (n, " (Z)" if p.get("basis", "X") == "Z" else "")
    if family == "ising":
        return "Ising-%d (r=%d)" % (n, p["steps"])
    if family == "brickwork":
        return "Brick-%d (L=%d, seed %d)" % (n, p["layers"], p["seed"])
    if family == "grover":
        return "Grover-%d (k=%d)" % (n, p.get("k", 1))
    if family == "bv":
        return "BV-%d" % n
    if family == "context":
        return "Context-%d (L=%d, %s)" % (n, p.get("layers", 4), p.get("spectators", "zero"))
    return "%s-%d" % (family, n)


REPRESENTATIVE = {   # one circuit per family and size, for the overview grids
    4: [("qft", dict(m=None, s=1)), ("ghz", dict(theta=1.0471975511965976)),
        ("ising", dict(steps=4, T=1.0, J=1.0, h=1.0)), ("brickwork", dict(layers=6, seed=0))],
    6: [("qft", dict(m=None, s=1)), ("qft", dict(m=2, s=1)), ("ghz", dict(theta=1.0471975511965976)),
        ("ising", dict(steps=4, T=1.0, J=1.0, h=1.0)), ("brickwork", dict(layers=6, seed=0))],
}


# ------------------------------------------------------------------ health
def health(T: Tables, expected_cells: int | None = None, rep: Report | None = None) -> Report:
    """What is in the data and what failed — read this before trusting any figure. Returns the report (text and
    the method-status table ``rep.tables["status"]``); nothing is printed."""
    rep = rep if rep is not None else Report()
    rep.section("HEALTH · what the tables hold")
    ok = T.cells
    if expected_cells is None:
        expected_cells = T.src.expected_cells()
    rep.text("cells in the tables: %d%s" % (len(ok), "" if expected_cells is None else
                                            " of %d in the manifest" % expected_cells))
    rep.frame(ok.groupby("stage").size().rename("cells").reset_index())
    st = T.methods.groupby(["stage", "status"]).size().unstack(fill_value=0)
    rep.text("method rows by status:")
    rep.table("status", st, show=True)
    E = T.errors
    if E is not None and len(E):
        E = E.copy()
        E["kind"] = np.where(E["method"].isna(), "whole cell", "one method")
        E["message"] = E["error"].fillna("").astype(str).str.split("\n").str[0].str.slice(0, 110)
        rep.text("errors by stage, kind and status:")
        rep.frame(E.groupby(["stage", "kind", "status"]).size().rename("entries").reset_index())
        rep.text("most frequent error messages:")
        rep.frame(E.groupby(["kind", "message"]).size().sort_values(ascending=False).head(12).rename(
            "entries").reset_index())
    w = T.winners
    rep.text("winner rows: %d (%d cells × policies × budgets × candidate sets)" % (len(w), w.cell_id.nunique()))
    inc = w[w.n_feasible < w.n_methods]
    if len(inc):
        rep.text("note: %d winner rows where some method was infeasible at that budget (expected at small B for "
                 "the learning methods)" % len(inc))
    miss = w[w["missing"].fillna("").astype(str) != ""]
    if len(miss):
        rep.verdict("missing", False, "%d winner rows decided without a method that failed in that cell (column "
                    "'missing')" % len(miss))
    return rep


# ------------------------------------------------------------------ phase diagram of one circuit
def winners(T: Tables) -> pd.DataFrame:
    """The winners among the figure's methods (METHODS). When the store holds no other method this is the collected
    winners table; otherwise (e.g. the NIL study's 15 methods) the winners are recomputed from the ledger table,
    restricted to METHODS, with the same rules as ``qemx.phase.store.ledger_rows``: lowest RMSE, ties to the method
    listed first in the cell, "no_optimistic" without methods flagged optimistic (VD)."""
    keys = tuple(METHODS)
    present = set(T.methods["method"].unique())
    if present <= set(keys):
        return T.winners
    cache = getattr(T, "_winners_cache", {})
    if keys in cache:
        return cache[keys]
    if T.ledger is None or not len(T.ledger):
        raise ValueError("this store holds methods outside %s; load it with the ledger table to restrict the "
                         "winners" % list(keys))
    w = _restricted_winners(T.methods, T.ledger, keys)
    cache[keys] = w
    T._winners_cache = cache
    return w


def _restricted_winners(methods: pd.DataFrame, ledger: pd.DataFrame, keys) -> pd.DataFrame:
    """Winner rows over the methods ``keys`` only, vectorized; the rules of qemx.phase.store.ledger_rows: lowest RMSE
    (infeasible = inf), ties to the method listed first in the cell, candidate sets "all" and "no_optimistic"."""
    keys = list(keys)
    m = methods
    order = m.groupby("cell_id", sort=False)["method"].agg(list)
    rank = pd.DataFrame([(cid, k, i) for cid, ks in order.items() for i, k in enumerate(ks)],
                        columns=["cell_id", "method", "rank"])
    opt = set(m.loc[m["flags"].fillna("").astype(str).str.contains("optimistic"), "method"]) \
        if "flags" in m.columns else set()
    L = ledger[ledger["method"].isin(keys)][["cell_id", "method", "policy", "B", "rmse"]].copy()
    L["r"] = pd.to_numeric(L["rmse"], errors="coerce").fillna(np.inf)
    L = L.merge(rank, on=["cell_id", "method"], how="left")
    L["rank"] = L["rank"].fillna(99).astype(int)
    present = L.groupby("cell_id")["method"].agg(lambda s: set(s))
    miss = {cid: ",".join(k for k in order.get(cid, []) if k in keys and k not in present.get(cid, set()))
            for cid in order.index}
    out = []
    g = ["cell_id", "policy", "B"]
    for field, pool in (("all", L), ("no_optimistic", L[~L["method"].isin(opt)])):
        if pool.empty:
            continue
        P = pool.sort_values(g + ["r", "rank"], kind="mergesort")
        P["pos"] = P.groupby(g, sort=False).cumcount()
        first = P[P["pos"] == 0].set_index(g)
        second = P[P["pos"] == 1].set_index(g)
        stats = P.groupby(g, sort=False).agg(n_methods=("method", "size"),
                                             n_feasible=("r", lambda x: int(np.isfinite(x).sum())))
        W = stats.join(first[["method", "r"]].rename(columns={"method": "winner", "r": "rmse"})) \
            .join(second[["method", "r"]].rename(columns={"method": "runner_up", "r": "rmse_runner_up"}))
        W = W[np.isfinite(W["rmse"])].reset_index()
        ru = W["rmse_runner_up"].astype(float)
        W["margin"] = np.where(np.isfinite(ru) & (W["rmse"] > 0), ru / W["rmse"] - 1, np.nan)
        W["rmse_runner_up"] = ru.where(np.isfinite(ru))
        W["candidates"] = field
        W["missing"] = W["cell_id"].map(miss).fillna("")
        W["B"] = W["B"].astype(int)
        out.append(W)
    cols = ["cell_id", "policy", "B", "candidates", "n_methods", "n_feasible", "missing", "winner", "rmse",
            "runner_up", "rmse_runner_up", "margin"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def _winner_grid(T: Tables, cells: pd.DataFrame, policy: str, candidates: str):
    w = winners(T)
    w = w[(w.policy == policy) & (w.candidates == candidates) & (w.cell_id.isin(cells.index))]
    Bs = np.sort(w.B.unique())
    order = list(cells.index)
    idx = {k: i for i, k in enumerate(METHODS)}
    grid = np.full((len(Bs), len(order)), -1, int)
    tie = np.zeros_like(grid, bool)
    rmse = np.full(grid.shape, np.nan)
    for j, cid in enumerate(order):
        sub = w[w.cell_id == cid].set_index("B")
        for i, B in enumerate(Bs):
            if B in sub.index:
                r = sub.loc[B]
                if r["winner"] not in idx:
                    raise ValueError("winner %r is not among the figure's methods %s (restrict_winners)" %
                                     (r["winner"], METHODS))
                grid[i, j] = idx[r["winner"]]
                tie[i, j] = bool(pd.notna(r["margin"]) and r["margin"] < TIE)
                rmse[i, j] = float(r["rmse"]) if pd.notna(r["rmse"]) else np.nan
    return Bs, grid, tie, rmse


VEIL = 0.62     # opacity of the white veil over tiles where even the winner misses the target ε


def _draw_eps(ax, xedges, yedges, rmse, eps):
    """Target overlay: a white veil over the tiles where the winner's RMSE exceeds ε (no method reaches ε at that
    budget), and a black step line along the smallest budget that reaches ε in each column (B_min(ε)). A column
    without the line is veiled to the top: there no method reaches ε at any budget on the grid."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    miss = ~(rmse <= eps)                                   # NaN (no winner) counts as a miss
    V = np.ma.masked_where(~miss, np.ones_like(rmse))
    ax.pcolormesh(xedges, yedges, V, cmap=ListedColormap([(1, 1, 1, VEIL)]), vmin=0, vmax=1, shading="flat",
                  linewidth=0, zorder=1.5)
    ny, nx = rmse.shape
    first = np.full(nx, -1)
    for j in range(nx):
        ok = np.nonzero(~miss[:, j])[0]
        if len(ok):
            first[j] = ok[0]
    segs = []
    for j in range(nx):
        if first[j] >= 0:
            segs.append([(xedges[j], yedges[first[j]]), (xedges[j + 1], yedges[first[j]])])
        if j + 1 < nx and first[j] >= 0 and first[j + 1] >= 0 and first[j] != first[j + 1]:
            segs.append([(xedges[j + 1], yedges[first[j]]), (xedges[j + 1], yedges[first[j + 1]])])
    if segs:
        ax.add_collection(LineCollection(segs, colors=INK, linewidths=1.1, zorder=4.5, capstyle="projecting"))
    return first


TIE = 0.05      # runner-up within 5 % of the winner's RMSE: marked as a near-tie


def _draw_tiles(ax, xedges, yedges, grid, tie, label_min_tiles=4, veil=None):
    """Filled tiles (winner colours), white boundaries where the winner changes, abbreviations on regions.
    veil: boolean mask of the tiles that _draw_eps will fade (label ink is then chosen for the faded colour)."""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([style(k)[0] for k in METHODS])
    G = np.ma.masked_less(grid, 0)
    ax.pcolormesh(xedges, yedges, G, cmap=cmap, vmin=-0.5, vmax=len(METHODS) - 0.5, shading="flat",
                  linewidth=0, rasterized=False)
    ny, nx = grid.shape
    segs = []
    for i in range(ny):
        for j in range(nx):
            if j + 1 < nx and grid[i, j] != grid[i, j + 1]:
                segs.append([(xedges[j + 1], yedges[i]), (xedges[j + 1], yedges[i + 1])])
            if i + 1 < ny and grid[i, j] != grid[i + 1, j]:
                segs.append([(xedges[j], yedges[i + 1]), (xedges[j + 1], yedges[i + 1])])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=SURFACE, linewidths=1.4, capstyle="projecting"))
    # near-ties: a small dot at the tile centre
    ys, xs = np.nonzero(tie & (grid >= 0))
    if len(xs):
        cx = np.sqrt(xedges[xs] * xedges[xs + 1]) if xedges[0] > 0 and _is_log(ax, "x") else (xedges[xs] + xedges[xs + 1]) / 2
        cy = np.sqrt(yedges[ys] * yedges[ys + 1])
        ax.scatter(cx, cy, s=2, c=SURFACE, linewidths=0, alpha=0.85, zorder=4)
    _label_regions(ax, xedges, yedges, grid, veil)


LABEL_PT = 6.5


def _faded(hexcol: str) -> str:
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(255 * (VEIL + (1 - VEIL) * c))) for c in (r, g, b))


def _label_regions(ax, xedges, yedges, grid, veil=None):
    """Write each region's method abbreviation where it fits: horizontally on the widest run of the region's columns
    that leaves room above and below, else vertically in a single column tall enough, else not at all (the legend
    names it). Placement nearest the region's centre among the positions that fit."""
    fig = ax.figure
    bb = ax.get_position()
    W, H = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()

    def frac(edges, log):
        e = np.log10(edges) if log else np.asarray(edges, float)
        return (e - e[0]) / (e[-1] - e[0])
    fx, fy = frac(xedges, _is_log(ax, "x")), frac(yedges, _is_log(ax, "y"))
    cw, rh = np.diff(fx) * W, np.diff(fy) * H                     # tile sizes in inches
    cx_in, cy_in = (fx[:-1] + fx[1:]) / 2 * W, (fy[:-1] + fy[1:]) / 2 * H
    from scipy import ndimage
    ny, nx = grid.shape
    for k in np.unique(grid[grid >= 0]):
        name = METHODS[k]
        tw = 0.62 * LABEL_PT / 72 * len(name) + 0.04             # text box in inches (bold serif, small caps-ish)
        th = LABEL_PT / 72 * 1.15
        lab, nreg = ndimage.label(grid == k)
        for r in range(1, nreg + 1):
            mask = lab == r
            ii, jj = np.nonzero(mask)
            gx, gy = np.mean(cx_in[jj]), np.mean(cy_in[ii])       # region centre (inches)
            best = None
            for i in range(ny):                                  # horizontal candidates
                j = 0
                while j < nx:
                    if not mask[i, j]:
                        j += 1
                        continue
                    a = j
                    while j + 1 < nx and mask[i, j + 1]:
                        j += 1
                    b = j
                    j += 1
                    if cw[a:b + 1].sum() < tw:
                        continue
                    # rows around i (within the run's columns) must belong to the region for the text height
                    lo = hi = i
                    while lo > 0 and mask[lo - 1, a:b + 1].all() and (cy_in[i] - cy_in[lo - 1]) < th / 2 + rh[i]:
                        lo -= 1
                    while hi < ny - 1 and mask[hi + 1, a:b + 1].all() and (cy_in[hi + 1] - cy_in[i]) < th / 2 + rh[i]:
                        hi += 1
                    if (fy[hi + 1] - fy[lo]) * H < th:
                        continue
                    x = (fx[a] + fx[b + 1]) / 2 * W
                    d = (x - gx) ** 2 + (cy_in[i] - gy) ** 2
                    if best is None or d < best[0]:
                        best = (d, (fx[a] + fx[b + 1]) / 2, (fy[lo] + fy[hi + 1]) / 2, 0)
            if best is None:                                     # vertical candidates (one column)
                for j in range(nx):
                    if cw[j] < th:
                        continue
                    rows = np.nonzero(mask[:, j])[0]
                    if not len(rows):
                        continue
                    runs = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
                    for run in runs:
                        if (fy[run[-1] + 1] - fy[run[0]]) * H < tw:
                            continue
                        y = (fy[run[0]] + fy[run[-1] + 1]) / 2
                        d = (cx_in[j] - gx) ** 2 + (y * H - gy) ** 2
                        if best is None or d < best[0]:
                            best = (d, (fx[j] + fx[j + 1]) / 2, y, 90)
            if best is not None:
                fill = style(name)[0]
                if veil is not None:
                    jj0 = int(np.clip(np.searchsorted(fx, best[1], side="right") - 1, 0, nx - 1))
                    ii0 = int(np.clip(np.searchsorted(fy, best[2], side="right") - 1, 0, ny - 1))
                    if veil[ii0, jj0]:
                        fill = _faded(fill)
                ax.text(best[1], best[2], name, transform=ax.transAxes, ha="center", va="center",
                        fontsize=LABEL_PT, color=_ink_on(fill), zorder=5, fontweight="bold",
                        rotation=best[3])


def plain_log_axis(ax, which="x"):
    """Log axis with plain tick labels (0.01, 0.03, 0.1, …) instead of 6×10⁻²."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v)))
    axis.set_minor_formatter(NullFormatter())


def _is_log(ax, which):
    return (ax.get_xscale() if which == "x" else ax.get_yscale()) == "log"


def _budget_edges(Bs):
    ly = np.log10(Bs.astype(float))
    d = np.diff(ly)
    e = np.concatenate([[ly[0] - d[0] / 2], (ly[1:] + ly[:-1]) / 2, [ly[-1] + d[-1] / 2]])
    return 10 ** e


def phase_map(T: Tables, stage: str, circuit: str, *, policy="fixed", candidates="all", ax=None,
              coherent_fraction=None, device=None, legend=True, title=None, eps=None, scales=None):
    """Winner on the (noise scale s, budget B) plane for one circuit. Columns are the cells (one per scale), equally
    wide; each tick shows s and the cell's λ_LC. With eps: tiles where even the winner misses RMSE ≤ eps are veiled
    and a black step line marks the smallest budget that reaches eps (see _draw_eps)."""
    import matplotlib.pyplot as plt
    c = T.cells[(T.cells.stage == stage) & (T.cells.circuit == circuit)]
    if coherent_fraction is not None:
        c = c[np.isclose(c.coherent_fraction, coherent_fraction)]
    if device is not None:
        c = c[c.device == device]
    if scales is not None:
        c = c[np.isin(np.round(c.scale.astype(float), 9), np.round(np.asarray(scales, float), 9))]
    if c.empty:
        raise ValueError("no cells for %s in %s" % (circuit, stage))
    if c.scale.duplicated().any():
        raise ValueError("several cells per scale (filter by coherent_fraction / device): %s" %
                         sorted(c.drop_duplicates(["device", "coherent_fraction"])[["device", "coherent_fraction"]]
                                .itertuples(index=False)))
    c = c.sort_values("scale")
    Bs, grid, tie, rmse = _winner_grid(T, c, policy, candidates)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.set_yscale("log")
    xedges = np.arange(len(c) + 1, dtype=float)
    _draw_tiles(ax, xedges, _budget_edges(Bs), grid, tie, veil=None if eps is None else ~(rmse <= eps))
    if eps is not None:
        _draw_eps(ax, xedges, _budget_edges(Bs), rmse, eps)
    ax.set_xlim(0, len(c))
    ax.set_xticks(np.arange(len(c)) + 0.5)
    ax.set_xticklabels(["%g\n%.2g" % (s, l) for s, l in zip(c.scale, c.lam_lc)],
                       fontsize=7 if len(c) <= 6 else 6.2)
    ax.set_xlabel("noise scale s (top) · λ$_{\\rm LC}$ (bottom)", fontsize=8)
    ax.set_ylabel("total shots B")
    ax.set_ylim(_budget_edges(Bs)[0], _budget_edges(Bs)[-1])
    ax.tick_params(axis="x", length=0)
    ax.set_title(title or "%s · %s policy%s" % (c.label.iloc[0], policy, "" if candidates == "all" else
                                                  " · VD excluded"), loc="left")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    if legend:
        _legend(ax.figure, grid, eps=eps)
    return ax, grid


def _legend(fig, *grids, loc="lower center", ncol=4, extra="ties", eps=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    used = sorted({int(k) for g in grids for k in np.unique(g) if k >= 0})
    handles = [Patch(facecolor=style(METHODS[k])[0], edgecolor="none",
                     label="%s (%s)" % (METHODS[k], style(METHODS[k])[3])) for k in used]
    if extra == "ties":
        handles.append(plt_dot())
    elif extra == "hatch":
        handles.append(Patch(facecolor="#9a9994", edgecolor="white", hatch="////", label="< 2/3 of circuits agree"))
    if eps is not None:
        mix = tuple(VEIL * 1.0 + (1 - VEIL) * c for c in (0x9a / 255, 0x99 / 255, 0x94 / 255))
        handles.append(Patch(facecolor=mix, edgecolor="none", label="faded: no method reaches RMSE ≤ %g" % eps))
        handles.append(Line2D([], [], color=INK, lw=1.1, label="smallest B reaching RMSE ≤ %g" % eps))
    fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.2, columnspacing=1.2)


def plt_dot():
    from matplotlib.lines import Line2D
    return Line2D([], [], marker="o", color="none", markerfacecolor="#9a9994", markeredgecolor="none", markersize=3,
                  label="runner-up within %d %%" % round(TIE * 100))


def phase_grid(T: Tables, stage: str, n: int, *, policy="fixed", candidates="all", circuits=None,
               coherent_fraction=None, ncols=3, panel=(2.25, 2.35), eps=None):
    """Phase maps of the representative circuits of size n (or `circuits`), ncols per row, one shared legend.
    The default panel size puts three panels across a 6.6-inch text width. eps: the target overlay of phase_map."""
    import matplotlib.pyplot as plt
    if circuits is None:
        circuits = [circuit_id(f, n, **p) for f, p in REPRESENTATIVE.get(n, [])]
    have = [cid for cid in circuits if ((T.cells.stage == stage) & (T.cells.circuit == cid)).any()]
    if not have:
        raise ValueError("none of the representative circuits are in %s" % stage)
    nc = min(ncols, len(have))
    nr = int(math.ceil(len(have) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(panel[0] * nc, panel[1] * nr + 0.55), sharey=True, squeeze=False)
    grids = []
    for i, cid in enumerate(have):
        ax = axs[i // nc][i % nc]
        _, g = phase_map(T, stage, cid, policy=policy, candidates=candidates, ax=ax, legend=False,
                         coherent_fraction=coherent_fraction, title=T.cells[T.cells.circuit == cid].label.iloc[0],
                         eps=eps)
        grids.append(g)
        if i % nc:
            ax.set_ylabel("")
        if i // nc < nr - 1 and (i + nc) < len(have):
            ax.set_xlabel("")
    for j in range(len(have), nr * nc):
        axs[j // nc][j % nc].set_visible(False)
    chi = "" if coherent_fraction is None else " · χ = %g" % coherent_fraction
    fig.suptitle("%s · winner at equal total budget · %s training policy%s%s" % (
        stage, policy, "" if candidates == "all" else " · VD excluded", chi), x=0.01, ha="left", fontsize=10)
    leg_h = 0.55 if eps is None else 0.8
    fig.set_figheight(panel[1] * nr + leg_h)
    fig.tight_layout(rect=(0, leg_h / (panel[1] * nr + leg_h), 1, 1))
    _legend(fig, *grids, ncol=3, eps=eps)
    return fig


# ------------------------------------------------------------------ transfer across calibration snapshots (stage C)
C_DEVICES = ("fake_torino", "fake_kingston", "fake_brisbane")


def transfer_grid(T: Tables, n: int, *, policy="fixed", candidates="no_optimistic", circuits=None,
                  devices=C_DEVICES, ref_stage="A1-primary", stage="C-transfer", panel=(1.62, 1.75)):
    """Rows: the representative circuits of size n. Columns: fake_fez (the primary stage, restricted to the transfer
    stage's noise scales) and the transfer devices. Each panel's ticks give s and that device's λ_LC, so the same s on
    different devices can be read against λ_LC (does the boundary follow λ_LC or s?)."""
    import matplotlib.pyplot as plt
    if circuits is None:
        circuits = [circuit_id(f, n, **p) for f, p in REPRESENTATIVE.get(n, [])]
    C = T.cells[T.cells.stage == stage]
    if C.empty:
        raise ValueError("no %s cells" % stage)
    scales = sorted(C.scale.astype(float).unique())
    have = [cid for cid in circuits if (C.circuit == cid).any()]
    if not have:
        raise ValueError("none of the representative circuits are in %s" % stage)
    cols = [("fake_fez", ref_stage)] + [(d, stage) for d in devices if (C.device == d).any()]
    nr, nc = len(have), len(cols)
    fig, axs = plt.subplots(nr, nc, figsize=(panel[0] * nc + 0.4, panel[1] * nr + 0.6), sharey=True, squeeze=False)
    grids = []
    for i, cid in enumerate(have):
        for j, (dev, st) in enumerate(cols):
            ax = axs[i][j]
            try:
                _, g = phase_map(T, st, cid, policy=policy, candidates=candidates, ax=ax, legend=False,
                                 device=dev, scales=scales, title="%s · %s" % (
                                     T.cells[T.cells.circuit == cid].label.iloc[0] if j == 0 else "",
                                     dev.replace("fake_", "")))
                grids.append(g)
            except ValueError:
                ax.set_visible(False)
                continue
            ax.title.set_fontsize(7.5)
            if j:
                ax.set_ylabel("")
            if i < nr - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("s · λ$_{\\rm LC}$", fontsize=7)
    fig.suptitle("Transfer · winner at equal total budget on four calibration snapshots · %s policy%s" % (
        policy, "" if candidates == "all" else " · VD excluded"), x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.6 / (panel[1] * nr + 0.6), 1, 1))
    _legend(fig, *grids, ncol=3)
    return fig


# ------------------------------------------------------------------ one family pooled on the λ_LC axis
def family_map(T: Tables, stage: str, family: str, n: int | None = None, *, policy="fixed", candidates="all",
               bins=10, agree=2 / 3, ax=None, coherent_fraction=None, device=None):
    """Majority winner per (λ_LC bin, B) over all cells of a family (every depth, seed and scale). Hatched tiles:
    fewer than `agree` of the cells in the tile share the majority winner (an exact tie shows the method listed
    first in METHODS). Blank: no cell in that λ bin. The numbers on top count the cells per bin."""
    import matplotlib.pyplot as plt
    c = T.cells[(T.cells.stage == stage) & (T.cells.family == family)]
    if n is not None:
        c = c[c.n == n]
    if coherent_fraction is not None:
        c = c[np.isclose(c.coherent_fraction, coherent_fraction)]
    if device is not None:
        c = c[c.device == device]
    if c.empty:
        raise ValueError("no cells")
    lo, hi = np.log10(c.lam_lc.min()), np.log10(c.lam_lc.max())
    xedges = 10 ** np.linspace(lo - 1e-9, hi + 1e-9, bins + 1)
    c = c.assign(bin=np.clip(np.searchsorted(xedges, c.lam_lc.values, side="right") - 1, 0, bins - 1))
    w = winners(T)
    w = w[(w.policy == policy) & (w.candidates == candidates) & w.cell_id.isin(c.index)]
    w = w.join(c["bin"], on="cell_id")
    Bs = np.sort(w.B.unique())
    idx = {k: i for i, k in enumerate(METHODS)}
    grid = np.full((len(Bs), bins), -1, int)
    weak = np.zeros_like(grid, bool)
    counts = np.zeros(bins, int)
    for b, gb in w.groupby("bin"):
        counts[b] = gb.cell_id.nunique()
        for i, B in enumerate(Bs):
            s = gb[gb.B == B].winner.value_counts()
            if len(s):
                bad = [k for k in s.index if k not in idx]
                if bad:
                    raise ValueError("winner %r is not among the figure's methods %s" % (bad[0], METHODS))
                # majority; an exact tie goes to the method listed first in METHODS (not to the data order)
                top = sorted(s.items(), key=lambda kv: (-kv[1], idx[kv[0]]))[0]
                grid[i, b] = idx[top[0]]
                weak[i, b] = top[1] / s.sum() < agree
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    yedges = _budget_edges(Bs)
    _draw_tiles(ax, xedges, yedges, grid, np.zeros_like(weak), label_min_tiles=5)
    ys, xs = np.nonzero(weak & (grid >= 0))
    from matplotlib.patches import Rectangle
    for i, j in zip(ys, xs):
        ax.add_patch(Rectangle((xedges[j], yedges[i]), xedges[j + 1] - xedges[j], yedges[i + 1] - yedges[i],
                               fill=False, hatch="////", edgecolor=(1, 1, 1, 0.75), linewidth=0))
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    plain_log_axis(ax, "x")
    ax.set_xlabel("λ$_{\\rm LC}$ (light-cone fault rate)")
    ax.set_ylabel("total shots B")
    for j in range(bins):
        if counts[j]:
            ax.text(np.sqrt(xedges[j] * xedges[j + 1]), yedges[-1] * 1.08, str(counts[j]), ha="center", va="bottom",
                    fontsize=6, color=INK2)
    ax.set_title("%s · %s%s · majority winner (cells per bin on top)" % (
        stage, family, "" if n is None else "-%d" % n), loc="left", pad=12)
    return ax, grid


# ------------------------------------------------------------------ RMSE curves of one cell
def rmse_curves(T: Tables, cell_id: str, eps=(0.01,), axs=None):
    import matplotlib.pyplot as plt
    L = T.ledger[T.ledger.cell_id == cell_id]
    if L.empty:
        raise ValueError("no ledger rows for %s" % cell_id)
    info = T.cells.loc[cell_id]
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, pol in zip(axs, ("fixed", "opt")):
        for k in METHODS:
            d = L[(L.policy == pol) & (L.method == k)].sort_values("B")
            if d.empty:
                continue
            col, mk, ls, lab = style(k)
            y = d.rmse.where(d.feasible.astype(bool))
            ax.plot(d.B, y, color=col, ls=ls, lw=1.5, marker=mk, markevery=8, ms=3.5, label=lab)
        for e in eps:
            ax.axhline(e, color=INK2, lw=0.6, ls=":")
            ax.text(L.B.min(), e * 1.06, "ε = %g" % e, va="bottom", ha="left", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total shots B")
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_title("%s training policy" % pol, loc="left")
    axs[0].set_ylabel("RMSE")
    fig = axs[0].figure
    fig.suptitle("%s · %s · s=%g · λ$_{\\rm LC}$=%.3g" % (info.label, info.stage, info.scale, info.lam_lc), x=0.01,
                 ha="left", fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


# ------------------------------------------------------------------ signed bias against λ_LC
BIAS_METHODS = ["ZNE-R", "ZNE-E", "PEC", "CDR", "FR-L", "FR-R"]


def bias_fraction(T: Tables, stage: str, n: int, *, circuits=None, methods=BIAS_METHODS, ylim=1.25, ncols=3,
                  panel=(2.25, 2.1), coherent_fraction=None):
    """Signed residual bias of each method as a fraction of the unmitigated bias, b_i / |b_RAW|, against λ_LC, one
    panel per representative circuit. 0 = bias removed; −1 = no better than RAW (when RAW's bias is negative);
    a sign change marks a zero crossing, near which a method can win at large budgets by cancellation. Points beyond
    ±ylim are drawn as open markers on the edge. The learning methods' b is their infinite-training-shot bias."""
    import matplotlib.pyplot as plt
    if circuits is None:
        circuits = [circuit_id(f, n, **p) for f, p in REPRESENTATIVE.get(n, [])]
    M = T.methods[(T.methods.stage == stage) & (T.methods.status == "ok")]
    if coherent_fraction is not None:
        M = M[np.isclose(M.coherent_fraction, coherent_fraction)]
    have = [cid for cid in circuits if (M.circuit == cid).any()]
    if not have:
        raise ValueError("none of the circuits are in %s" % stage)
    nc = min(ncols, len(have))
    nr = int(math.ceil(len(have) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(panel[0] * nc, panel[1] * nr + 0.5), sharey=True, squeeze=False)
    used = []
    for i, cid in enumerate(have):
        ax = axs[i // nc][i % nc]
        d = M[M.circuit == cid]
        raw = d[d.method == "RAW"].set_index("cell_id")["b"]
        lam = d.drop_duplicates("cell_id").set_index("cell_id")["lam_lc"]
        ax.axhline(0, color=INK2, lw=0.6)
        for yref in (-1, 1):
            ax.axhline(yref, color=INK2, lw=0.5, ls=(0, (2, 2)))
        for k in methods:
            e = d[d.method == k].set_index("cell_id")
            if e.empty:
                continue
            x = lam.reindex(e.index).values
            y = (e["b"] / raw.reindex(e.index).abs()).values
            o = np.argsort(x)
            x, y = x[o], y[o]
            col, mk, ls, lab = style(k)
            yc = np.clip(y, -ylim, ylim)
            inside = np.abs(y) <= ylim
            ax.plot(x, yc, color=col, lw=1.3, ls=ls, zorder=3)
            ax.scatter(x[inside], yc[inside], marker=mk, s=14, color=col, zorder=4, linewidths=0.8)
            if (~inside).any():
                ax.scatter(x[~inside], yc[~inside], marker=mk, s=16, facecolors="none", edgecolors=col, zorder=4,
                           linewidths=0.9)
            used.append(k)
        ax.set_xscale("log")
        plain_log_axis(ax, "x")
        ax.set_ylim(-ylim * 1.06, ylim * 1.06)
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_title(T.cells[T.cells.circuit == cid].label.iloc[0], loc="left")
        ax.set_xlabel("λ$_{\\rm LC}$")
        if i % nc == 0:
            ax.set_ylabel("b / |b$_{\\rm RAW}$|")
    for j in range(len(have), nr * nc):
        axs[j // nc][j % nc].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=style(k)[0], marker=style(k)[1], lw=1.3, ms=4, label=style(k)[3])
               for k in methods if k in used]
    handles.append(Line2D([], [], color="none", marker="o", markerfacecolor="none", markeredgecolor=INK2, ms=4,
                          label="beyond ±%g (clipped)" % ylim))
    fig.suptitle("%s · residual bias as a fraction of the unmitigated bias%s" % (
        stage, "" if coherent_fraction is None else " · χ = %g" % coherent_fraction), x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.5 / (panel[1] * nr + 0.5), 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------ B*(ε) against λ_LC
def table_costs(T: Tables, cell_ids=None) -> dict:
    """{(cell_id, method): (qemx ledger Cost or None, cell's s_min, cell's s_tr_min)} rebuilt from the flat tables
    (method rows and training levels) through qemx's own ``cost_of``; {} when qemx is not importable."""
    try:
        from qemx.phase.cell import cost_of
    except ImportError:
        return {}
    M = T.methods[T.methods.status == "ok"]
    if cell_ids is not None:
        M = M[M.cell_id.isin(list(cell_ids))]
    lv = T.levels if T.levels is not None and len(T.levels) else pd.DataFrame(columns=["cell_id", "method"])
    lv_by = {k: g for k, g in lv.groupby(["cell_id", "method"])} if len(lv) else {}
    out = {}
    for r in M.to_dict("records"):
        row = dict(key=r["method"], status="ok", b=_num(r.get("b")), v=_num(r.get("v")), w=_num(r.get("w")),
                   c=int(_f(r.get("c"))), T=int(_f(r.get("T"))), s_tr=int(_f(r.get("s_tr"))),
                   u1=_f(r.get("u1")), m=int(_f(r.get("m")) or 1), a=json.loads(r.get("a") or "[]"),
                   levels=[{k: _num(v) if k not in ("ok", "error") else v for k, v in l.items()}
                           for l in lv_by.get((r["cell_id"], r["method"]), pd.DataFrame()).to_dict("records")])
        for l in row["levels"]:
            l["ok"] = not (l.get("ok") is False or str(l.get("ok")).lower() == "false")
        out[(r["cell_id"], r["method"])] = (cost_of(row), int(_f(r.get("s_min")) or 32),
                                             int(_f(r.get("s_tr_min")) or 32))
    return out


def budget_to_target(T: Tables, cell_ids, eps: float, policy="fixed") -> pd.DataFrame:
    """B*(ε) per (cell, method): exact (qemx ledger, integer allocations) when qemx is importable, otherwise
    log-log interpolation of the 51-point ledger grid. inf: the method's floor (bias, training noise) is above ε."""
    costs = table_costs(T, cell_ids)
    if costs:
        from qemx.phase.ledger import Policy, budget_for
        out = []
        for (cid, k), (c, s_min, s_tr_min) in costs.items():
            pol = Policy(policy, True, s_min, s_tr_min)           # the cell's own lower bounds, as in collect
            out.append(dict(cell_id=cid, method=k, Bstar=budget_for(c, eps, pol) if c is not None else math.inf,
                            exact=True))
        return pd.DataFrame(out)
    L = T.ledger[T.ledger.cell_id.isin(cell_ids) & (T.ledger.policy == policy)]
    out = []
    for (cid, k), d in L.groupby(["cell_id", "method"]):
        d = d.sort_values("B")
        ok = d.feasible.astype(bool) & d.rmse.notna()
        x, y = np.log(d.B[ok].values.astype(float)), np.log(d.rmse[ok].values)
        below = np.nonzero(y <= math.log(eps))[0]
        if not len(below):
            B = math.inf
        elif below[0] == 0:
            B = float(np.exp(x[0]))
        else:
            i = below[0]
            B = float(np.exp(np.interp(math.log(eps), [y[i], y[i - 1]], [x[i], x[i - 1]])))
        out.append(dict(cell_id=cid, method=k, Bstar=B, exact=False))
    return pd.DataFrame(out)


def _f(x):
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _num(x):
    """A table value as a float, or None for a missing one (NaN in pandas)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def budget_curves(T: Tables, stage: str, circuit: str, eps=0.01, policy="fixed", ax=None, coherent_fraction=None):
    """B*(ε) against λ_LC for one circuit (its cells across noise scales). A method whose RMSE floor (bias,
    training noise) stays above ε at every budget is drawn as an open marker in the 'unreachable' band on top."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    c = T.cells[(T.cells.stage == stage) & (T.cells.circuit == circuit)]
    if coherent_fraction is not None:
        c = c[np.isclose(c.coherent_fraction, coherent_fraction)]
    c = c.sort_values("lam_lc")
    D = budget_to_target(T, list(c.index), eps, policy).join(c[["lam_lc"]], on="cell_id")
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.1))
    fin_all = D.Bstar[np.isfinite(D.Bstar)]
    lo = 10 ** math.floor(math.log10(max(fin_all.min(), 1.0))) if len(fin_all) else 1e3
    band = 10 ** (math.ceil(math.log10(max(fin_all.max(), 1e4))) + 0.3) if len(fin_all) else 1e9
    for slot, k in enumerate(METHODS):
        d = D[D.method == k].sort_values("lam_lc")
        if d.empty:
            continue
        col, mk, ls, lab = style(k)
        fin = np.isfinite(d.Bstar)
        ax.plot(d.lam_lc[fin], d.Bstar[fin], color=col, ls=ls, lw=1.4, marker=mk, ms=4, label=lab)
        if (~fin).any():                                    # one row per method inside the band
            yy = np.full((~fin).sum(), band * 1.22 ** slot)
            if mk in ("x", "+", "1", "2", "3", "4", "|", "_"):  # line-art markers have no face
                ax.scatter(d.lam_lc[~fin], yy, marker=mk, s=16, c=col, linewidths=1.0, zorder=5)
            else:
                ax.scatter(d.lam_lc[~fin], yy, marker=mk, s=16, facecolors="none", edgecolors=col,
                           linewidths=1.0, zorder=5)
    ax.axhspan(band / 1.15, band * 1.22 ** len(METHODS), color="#f0efec", zorder=0, lw=0)
    ax.text(0.01, band * 1.22 ** (len(METHODS) - 0.3), "unreachable: floor > ε", transform=ax.get_yaxis_transform(),
            fontsize=6.5, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(lo / 2, band * 1.22 ** len(METHODS))
    plain_log_axis(ax, "x")
    ax.set_xlabel("λ$_{\\rm LC}$")
    ax.set_ylabel("B*(ε): shots to reach RMSE ≤ %g" % eps)
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_title("%s · %s · %s policy" % (c.label.iloc[0], stage, policy), loc="left")
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=INK2, markersize=4))
    l.append("open marker: unreachable")
    ax.legend(h, l, frameon=False, fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ax


# ------------------------------------------------------------------ validation
def validation(T: Tables):
    import matplotlib.pyplot as plt
    from scipy import stats
    V = T.validation
    if V is None or V.empty:
        raise ValueError("no validation table")
    V = V[V.feasible.astype(bool) & V.z_rmse.notna() & V.independent.astype(bool)]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    xs = np.linspace(-5, 5, 400)
    for ax, col, name in zip(axs, ("z_rmse", "z_bias"), ("RMSE", "bias")):
        z = V[col].values
        # density over ALL checks (weights), so the bars integrate to the in-range share and outliers are not hidden
        inr = np.abs(z) <= 5
        ax.hist(z[inr], bins=np.arange(-5, 5.01, 0.5), weights=np.full(inr.sum(), 1.0 / (0.5 * len(z))),
                color="#86b6ef", edgecolor=SURFACE, linewidth=0.8)
        ax.plot(xs, stats.norm.pdf(xs), color=INK, lw=1.0, label="N(0, 1)")
        p = stats.kstest(z, "norm").pvalue
        ax.set_title("%s: n=%d, mean %.2f, sd %.2f, KS p=%.2f\n|z|>4: %d (%.2f %%), outside ±5 (not drawn): %d" % (
            name, len(z), z.mean(), z.std(ddof=1), p, int(np.sum(np.abs(z) > 4)), 100 * np.mean(np.abs(z) > 4),
            int(np.sum(~inr))), loc="left", fontsize=8)
        ax.set_xlabel("z = (Monte Carlo − ledger) / SE")
        ax.set_xlim(-5, 5)
    axs[0].set_ylabel("density")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    return fig


# ====================================================================== the gap between a component model and a device
# Figures of scripts/dirA_divergence.py. One message runs through all of them — how far an
# average-gate-fidelity noise model is from the device, and which context it is missing — so they share one colour
# grammar (validated: dataviz method, OKLab lightness of every ramp monotone per arm, CVD-safe categorical slots):
#
#   warm (red)   the device is worse than its model: unmodeled error          (diverging warm arm; sequential ramp)
#   cool (blue)  the device is better than its model: the model is pessimistic (diverging cool arm)
#   grey         as modeled / within shot noise                                (neutral midpoint, open markers)
#
# Circuit families are told apart by marker shape, never by colour (colour stays free for the value). Panel titles are
# short; the explanation belongs in the thesis caption.

# diverging scale: blue ↔ red with a grey midpoint, five steps per arm at mirrored OKLab lightness
# (0.34, 0.43, 0.57, 0.72, 0.86 | 0.95 | 0.86, 0.72, 0.57, 0.43, 0.35)
DIV_COLORS = ["#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#b7d3f6", "#f0efec",
              "#f6c3be", "#e9837f", "#cf3a39", "#8f1d1d", "#6b1414"]
EXCESS_COLORS = ["#f0efec", "#f6c3be", "#e9837f", "#cf3a39", "#8f1d1d", "#6b1414"]   # the warm arm: magnitude only
WORSE, BETTER, NEUTRAL_MARK = "#cf3a39", "#2a78d6", "#a9a8a3"
MODEL_COLOR, DEVICE_COLOR = "#2a78d6", "#cf3a39"                  # model (prediction) cool, device (observation) warm
FAMILY_MARKERS = {"qft": "s", "ghz": "o", "bv": "^", "grover": "D", "brickwork": "h", "ising": "P", "context": "*"}
FAMILY_LABEL = {"qft": "QFT", "ghz": "GHZ", "bv": "BV", "grover": "Grover", "brickwork": "brickwork",
                "ising": "Ising", "context": "context probe"}
DEPTH_AXES = {"two_qubit": "post-compilation two-qubit gates", "depth": "post-compilation depth",
              "norm_depth": "normalized depth (Lubinski et al.)",
              "two_qubit_executed": "two-qubit gates executed (most folded circuit)"}
VALUE_LABEL = {"delta": r"$F_c^{\rm pred} - F_c^{\rm obs}$",
               "delta_nf": r"$F_N^{\rm pred} - F_N^{\rm obs}$ (normalized fidelity)",
               "ex_nf_per_2q": r"unmodeled error per two-qubit gate, $(\ln F_N^{\rm pred} - \ln F_N^{\rm obs})/n_2$",
               "ex_per_2q": r"unmodeled error per two-qubit gate, $(\ln F_c^{\rm pred} - \ln F_c^{\rm obs})/n_2$"}
VALUE_SHORT = {"delta": "Δ", "delta_nf": "ΔF_N", "ex_nf_per_2q": "excess/gate", "ex_per_2q": "excess/gate"}
Z_OF_VALUE = {"delta": "z", "delta_nf": "z_nf", "ex_nf_per_2q": "z_nf", "ex_per_2q": "z"}
BINS = {"log2": 1, "sqrt2": 2}                    # bins per octave of the x axis (sqrt2: Chen et al.'s volumetric plot)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("divergence", DIV_COLORS)


def excess_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("excess", EXCESS_COLORS)


def _div_rows(D: pd.DataFrame, method: str, predictor: str, x: str, value: str) -> pd.DataFrame:
    d = D[(D["method"] == method) & (D["predictor"] == predictor)]
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    d = d[np.isfinite(pd.to_numeric(d[value], errors="coerce")) & np.isfinite(pd.to_numeric(d[x], errors="coerce"))]
    return d.copy()


def _vmax(D: pd.DataFrame, value: str) -> float:
    """A round, symmetric colour limit at the 98th percentile of |value| (a few short or badly mitigated circuits
    would set it otherwise; the colourbar then shows an arrow for the values beyond it): per-gate rates rounded up
    to 0.001, Δ to 0.02."""
    v = pd.to_numeric(D[value], errors="coerce").abs()
    m = float(np.nanpercentile(v, 98)) if v.notna().any() else 0.0
    if value.startswith("ex_"):
        return max(1e-3, float(np.ceil(m * 1e3) / 1e3))
    return max(0.02, float(np.ceil(m * 50) / 50))


def _family_handles(families, color=None, size=4.2):
    from matplotlib.lines import Line2D
    fams = [f for f in FAMILY_MARKERS if f in set(families)]
    return [Line2D([], [], ls="", marker=FAMILY_MARKERS[f], ms=size, mfc=color or "none", mec=INK2, mew=0.7,
                   label=FAMILY_LABEL.get(f, f)) for f in fams]


def _signif_class(z, z_sig: float):
    z = np.asarray(z, float)
    return np.where(z > z_sig, 1, np.where(z < -z_sig, -1, 0))


def _no_data(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK2)
    ax.set_xticks([])
    ax.set_yticks([])


def _horizontal_colorbar(fig, mappable, ax, label, ends=("device better", "device worse"), extend="neither",
                         **kw):
    """A slim colourbar under ``ax`` (an axes or a list of them), the meaning of each end written above its end."""
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal", location="bottom", extend=extend,
                      extendfrac=0.03, fraction=kw.get("fraction", 0.06), pad=kw.get("pad", 0.02),
                      aspect=kw.get("aspect", 30), shrink=kw.get("shrink", 0.85))
    cb.outline.set_linewidth(0.4)
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=bool(ends and ends[0] == "device better")))
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
    cb.set_label(label, fontsize=7, labelpad=2)
    if ends:
        off = 0.035 if extend in ("min", "both") else 0.012
        cb.ax.text(-off, 0.5, ends[0] + " \u2190", transform=cb.ax.transAxes, ha="right", va="center", fontsize=6.5,
                   color=INK2)
        off = 0.035 if extend in ("max", "both") else 0.012
        cb.ax.text(1 + off, 0.5, "\u2192 " + ends[1], transform=cb.ax.transAxes, ha="left", va="center",
                   fontsize=6.5, color=INK2)
    return cb


def _tile_layout(d: pd.DataFrame, x: str, k: int):
    xv = pd.to_numeric(d[x], errors="coerce").astype(float)
    return np.where(xv >= 1, np.floor(np.log2(np.maximum(xv, 1)) * k + 1e-9), -1).astype(int)


TILE_CAPACITY = 12      # circuits drawn inside one tile (4 per row); beyond that the tile shows "significant/total"


def difference_volumetric(D: pd.DataFrame, method: str, predictor: str = "depolarizing", *, x: str = "two_qubit",
                          value: str = "delta", z_sig: float = 3.0, annotate: int = 0, vmax: float | None = None,
                          ax=None, title: str = "", colorbar: bool = True, legend: bool = True, families=None,
                          circuits: bool = False, bins: str = "log2", widths=None, bin_range=None):
    """The difference volumetric plot (Chen et al. 2024, Fig. 11): circuit width against ``x`` in logarithmic bins
    (``bins="log2"``, or Chen et al.'s half-octave ``"sqrt2"``). Every tile is the mean ``value`` of its circuits on
    the symmetric diverging scale — red: the device is worse than its component model, blue: better, grey: as
    modeled. A dot in a tile's corner: most of its circuits lie beyond the shot noise (z > z_sig; open ring:
    z < −z_sig).

    ``circuits=True`` draws every circuit inside its tile (shape = family; filled beyond the shot noise; a tile with
    more than TILE_CAPACITY circuits shows "significant/total" instead); ``annotate=k`` numbers the k circuits with
    the largest significant value and names them in the legend (implies ``circuits``). ``x``: two_qubit (default),
    depth, norm_depth (Lubinski et al.) or two_qubit_executed. ``widths`` and ``bin_range`` fix the rows and columns
    (difference_volumetric_grid passes the union over its panels so they align)."""
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    own = ax is None
    if own:
        nw = len(widths) if widths is not None else D["width"].nunique() if "width" in D.columns else 6
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.4, max(2.9, 0.09 * nw + 1.9)), constrained_layout=True)
    else:
        fig = ax.figure
    d = _div_rows(D, method, predictor, x, value)
    ax.set_title(title or method, loc="left")
    if d.empty:
        _no_data(ax, "no comparable rows")
        return fig
    zcol = Z_OF_VALUE.get(value, "z")
    d["zv"] = pd.to_numeric(d[zcol], errors="coerce") if zcol in d.columns else np.nan
    d["vv"] = pd.to_numeric(d[value], errors="coerce").astype(float)
    k = BINS.get(bins, 1)
    d["bin"] = _tile_layout(d, x, k)
    vmax = float(vmax) if vmax else _vmax(d, value)
    norm, cmap = Normalize(-vmax, vmax), diverging_cmap()
    widths = sorted(int(w) for w in (widths if widths is not None else d["width"].unique()))
    ypos = {w: i for i, w in enumerate(widths)}          # one row per width that occurs (no empty rows)
    b0, b1 = (int(bin_range[0]), int(bin_range[1])) if bin_range is not None else (int(d["bin"].min()),
                                                                                   int(d["bin"].max()))
    gap = 0.07                                           # the surface gap between tiles (fraction of a tile)
    if annotate:
        circuits = True
    top = d[(d["zv"] > z_sig) & (d["vv"] > 0)].sort_values("vv", ascending=False).head(int(annotate)) \
        if annotate else d.head(0)
    rank = {idx: i + 1 for i, idx in enumerate(top.index)}
    stroke = [pe.withStroke(linewidth=1.6, foreground="white")]
    for (b, w), g in d.groupby(["bin", "width"]):
        v = float(g["vv"].mean())
        xl, y = b / k, ypos[int(w)]
        ax.add_patch(Rectangle((xl + gap / (2 * k), y - 0.5 + gap / 2), (1 - gap) / k, 1 - gap,
                               facecolor=cmap(norm(np.clip(v, -vmax, vmax))), edgecolor="none", zorder=1))
        cls = _signif_class(g["zv"], z_sig)
        if not circuits:
            if (cls == 1).mean() >= 0.5:
                ax.plot([xl + 0.84 / k], [y + 0.27], marker="o", ms=2.6, color=INK, mew=0, zorder=4)
            elif (cls == -1).mean() >= 0.5:
                ax.plot([xl + 0.84 / k], [y + 0.27], marker="o", ms=2.8, mfc="none", mec=INK, mew=0.6, zorder=4)
            continue
        g = g.sort_values(["family", x, "label"])
        m = len(g)
        if m > TILE_CAPACITY:                            # too many to draw: the count of significant circuits
            ax.text(xl + 0.5 / k, y, "%d/%d" % (int((cls != 0).sum()), m), ha="center", va="center", fontsize=5.8,
                    color=INK, zorder=4, path_effects=stroke)
            continue
        per_row = min(m, 4)
        nrow = int(math.ceil(m / per_row))
        for j, (idx, r) in enumerate(g.iterrows()):
            cx = xl + (j % per_row + 0.5) / (per_row * k)
            cy = y + (0.0 if nrow == 1 else ((j // per_row) - (nrow - 1) / 2) * 0.27)
            sig = _signif_class([r["zv"]], z_sig)[0] != 0
            ax.plot([cx], [cy], ls="", marker=FAMILY_MARKERS.get(r["family"], "o"), ms=3.0 if nrow < 3 else 2.4,
                    mfc=INK if sig else "white", mec=INK, mew=0.45, zorder=3)
            if idx in rank:
                ax.text(cx, cy + 0.17, str(rank[idx]), fontsize=5.8, ha="center", va="bottom", color=INK, zorder=5,
                        path_effects=stroke)
    ax.set_xlim(b0 / k - 0.04, (b1 + 1) / k + 0.04)
    ax.set_ylim(-0.56, len(widths) - 0.44)
    ticks = list(range(int(np.floor(b0 / k)), int(np.ceil((b1 + 1) / k)) + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%d" % 2 ** t if t >= 0 else "0" for t in ticks])
    every = int(math.ceil(len(widths) / 14))           # at most ~14 row labels
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([str(w) if (len(widths) - 1 - i) % every == 0 else "" for i, w in enumerate(widths)])
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(DEPTH_AXES.get(x, x) + (" (half-octave bins)" if k == 2 else ""))
    ax.set_ylabel("circuit width (qubits)")
    if colorbar:
        from matplotlib.cm import ScalarMappable
        _horizontal_colorbar(fig, ScalarMappable(norm=norm, cmap=cmap), ax, VALUE_LABEL.get(value, value),
                             extend=_extend(d["vv"], vmax), pad=0.02, shrink=0.55, aspect=24)
    if legend:
        hs = _volumetric_key(circuits, z_sig, families if families is not None else d["family"])
        for idx, r in top.iterrows():
            hs.append(Line2D([], [], ls="", marker="$%d$" % rank[idx], ms=4.5, color=INK,
                             label="%s (%s %.2g)" % (r["label"], VALUE_SHORT.get(value, value), float(r["vv"]))))
        if own:
            fig.legend(handles=hs, loc="outside lower center", ncol=min(len(hs), 3 if annotate else 4),
                       handletextpad=0.3, columnspacing=1.0)
        else:
            ax.legend(handles=hs, loc="upper left", fontsize=6.3)
    return fig


def _volumetric_key(circuits: bool, z_sig: float, families) -> list:
    from matplotlib.lines import Line2D
    if circuits:
        return _family_handles(families) + [
            Line2D([], [], ls="", marker="o", ms=3.0, mfc=INK, mec=INK, label="|z| > %g (filled)" % z_sig)]
    return [Line2D([], [], ls="", marker="o", ms=2.6, color=INK, mew=0, label="most circuits z > %g" % z_sig),
            Line2D([], [], ls="", marker="o", ms=2.8, mfc="none", mec=INK, mew=0.6,
                   label="most circuits z < −%g" % z_sig)]


def _extend(v, vmax: float) -> str:
    v = pd.to_numeric(pd.Series(v), errors="coerce")
    lo, hi = bool((v < -vmax).any()), bool((v > vmax).any())
    return "both" if lo and hi else "min" if lo else "max" if hi else "neither"


def difference_volumetric_grid(D: pd.DataFrame, methods, predictor: str = "depolarizing", *, x: str = "two_qubit",
                               value: str = "delta", ncols: int = 3, title: str = "", bins: str = "log2", **kw):
    """The difference volumetric plot of several methods (equal budget) on one colour scale, with the same rows
    (widths) and columns (bins) in every panel: how mitigation moves the gap. One colourbar and one key for all."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    methods = list(methods)
    sub = D[(D["predictor"] == predictor) & D["method"].isin(methods)]
    if "status" in sub.columns:
        sub = sub[sub["status"] == "ok"]
    sub = sub[np.isfinite(pd.to_numeric(sub[value], errors="coerce")) & np.isfinite(pd.to_numeric(sub[x],
                                                                                                errors="coerce"))]
    vmax = kw.pop("vmax", None) or (_vmax(sub, value) if len(sub) else 0.1)
    k = BINS.get(bins, 1)
    widths = sorted(int(w) for w in sub["width"].unique()) if len(sub) else None
    bb = _tile_layout(sub, x, k) if len(sub) else None
    bin_range = (int(bb.min()), int(bb.max())) if bb is not None else None
    ncols = max(1, min(ncols, len(methods)))
    nrows = int(math.ceil(len(methods) / ncols))
    panel_h = max(1.75, 0.075 * len(widths or []) + 0.6)
    fig, axs = plt.subplots(nrows, ncols, figsize=(TEXT_WIDTH, panel_h * nrows + 0.85), squeeze=False,
                            constrained_layout=True, sharex=True, sharey=True)
    fig.get_layout_engine().set(wspace=0.07)
    circuits = bool(kw.get("circuits") or kw.get("annotate"))
    for i, m in enumerate(methods):
        a = axs[i // ncols][i % ncols]
        difference_volumetric(D, m, predictor, x=x, value=value, vmax=vmax, ax=a, colorbar=False, legend=False,
                              bins=bins, widths=widths, bin_range=bin_range, **kw)
        if i % ncols:
            a.set_ylabel("")
        if (i + ncols) < len(methods):
            a.set_xlabel("")
            a.tick_params(labelbottom=False)
        else:
            a.tick_params(labelbottom=True)
    for j in range(len(methods), nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    used = [axs[i // ncols][i % ncols] for i in range(len(methods))]
    _horizontal_colorbar(fig, ScalarMappable(norm=Normalize(-vmax, vmax), cmap=diverging_cmap()), used,
                         VALUE_LABEL.get(value, value), extend=_extend(sub[value], vmax), fraction=0.04, pad=0.02,
                         aspect=46, shrink=0.55)
    fig.legend(handles=_volumetric_key(circuits, kw.get("z_sig", 3.0), sub["family"]), loc="outside lower center",
               ncol=8)
    return fig


def fidelity_gap(D: pd.DataFrame, method: str = "RAW", predictor: str = "depolarizing", *, x: str = "two_qubit",
                 floor: float = 0.05, title: str = "", ax=None, connect: bool = True, families_legend: bool = True,
                 fit: str = "free"):
    """The gap itself (after Chen et al.'s Fig. 9): F_c of every circuit against its post-compilation two-qubit gate
    count, observed (filled, warm) and predicted by the component model (open, cool), joined by a thin segment — the
    unmodeled error of that circuit; with the fitted decays F_c ≈ a (1 − ε)^{n₂} of both and their effective
    two-qubit error rates ε_eff written at the lines (the circuits with F_c ≥ ``floor``). Semilog: an exponential
    decay is a straight line, and the gap between the two lines is the error the model does not have.
    ``fit="free"`` fits the intercept a (it absorbs state preparation and readout, as qemx.divergence.effective_rate
    does); ``fit="origin"`` fixes a = 1 (Chen et al.'s guide lines; right when F_c(N₂ → 0) ≈ 1)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.6, 3.0), constrained_layout=True)
    else:
        fig = ax.figure
    d = D[(D["method"] == method) & (D["predictor"] == predictor)]
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    d = d[np.isfinite(pd.to_numeric(d["fc_obs"], errors="coerce")) & np.isfinite(pd.to_numeric(d["fc_pred"],
                                                                                              errors="coerce"))]
    if title:
        ax.set_title(title, loc="left")
    if d.empty:
        _no_data(ax, "no comparable rows")
        return fig
    n2 = pd.to_numeric(d[x], errors="coerce").astype(float).to_numpy()
    fo = d["fc_obs"].astype(float).to_numpy()
    fp = d["fc_pred"].astype(float).to_numpy()
    if connect:
        for xi, a, b in zip(n2, fp, fo):
            ax.plot([xi, xi], [a, b], color=WORSE if b < a else BETTER, lw=0.4, alpha=0.35, zorder=1,
                    solid_capstyle="butt")
    for fam, g in d.groupby("family"):
        mk = FAMILY_MARKERS.get(fam, "o")
        xx = pd.to_numeric(g[x], errors="coerce").astype(float)
        ax.plot(xx, g["fc_pred"].astype(float), ls="", marker=mk, ms=3.4, mfc="white", mec=MODEL_COLOR, mew=0.6,
                zorder=2)
        ax.plot(xx, g["fc_obs"].astype(float), ls="", marker=mk, ms=3.4, mfc=DEVICE_COLOR, mec="white", mew=0.35,
                zorder=3)
    xmax = float(np.nanmax(n2)) * 1.04
    xs = np.linspace(0, xmax, 200)
    lo = max(float(np.nanmin(np.concatenate([fo, fp]))), floor) * 0.85
    rates, lines = {}, []
    for who, f, col in (("model", fp, MODEL_COLOR), ("device", fo, DEVICE_COLOR)):
        ok = np.isfinite(f) & (f >= floor) & np.isfinite(n2)
        if ok.sum() >= 3 and np.ptp(n2[ok]) > 0:
            if fit == "origin":
                s_, a_ = float(np.sum(n2[ok] * np.log(f[ok])) / np.sum(n2[ok] ** 2)), 0.0
            else:
                s_, a_ = np.polyfit(n2[ok], np.log(f[ok]), 1)
            eps = 1 - math.exp(s_)
            rates[who] = eps
            ax.plot(xs, np.exp(a_ + s_ * xs), color=col, lw=1.1, zorder=0, ls="-" if who == "device" else (0, (4, 2)))
            lines.append((who, s_, a_, eps))
    for who, s_, a_, eps in lines:                         # the rate at the end of its line, inside the axes
        xe = xmax
        if s_ < 0 and a_ + s_ * xe < math.log(lo * 1.3):
            xe = (math.log(lo * 1.3) - a_) / s_
        ax.annotate(r"$\varepsilon_{\rm eff}^{\rm %s} = %s$" % (who, _sci(eps)), (xe, math.exp(a_ + s_ * xe)),
                    xytext=(5, 2) if xe < xmax else (-2, 6), textcoords="offset points", fontsize=6.5, color=INK,
                    ha="left" if xe < xmax else "right", va="bottom",
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["x"]).withStroke(
                        linewidth=1.8, foreground="white")])
    ax.set_yscale("log")
    ax.set_ylim(lo, 1.08)
    _plain_log_y(ax)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(DEPTH_AXES.get(x, x))
    ax.set_ylabel(r"$F_c$")
    hs = [Line2D([], [], ls="-", marker="o", ms=3.4, mfc=DEVICE_COLOR, mec="white", color=DEVICE_COLOR, lw=1.1,
                 label="device (observed)"),
          Line2D([], [], ls=(0, (4, 2)), marker="o", ms=3.4, mfc="white", mec=MODEL_COLOR, color=MODEL_COLOR,
                 lw=1.1, label="component model (predicted)")]
    ax.legend(handles=hs, loc="upper right", fontsize=6.5, handlelength=2.2)
    if families_legend:
        fh = _family_handles(d["family"], size=3.6)
        if own:
            fig.legend(handles=fh, loc="outside lower center", ncol=min(len(fh), 7), fontsize=6.3,
                       handletextpad=0.15, columnspacing=0.8)
        else:
            ax.legend(handles=hs + fh, loc="upper right", fontsize=6.3, ncol=2)
    fig.gap_rates = rates                                  # for captions and tests
    return fig


def _sci(x: float) -> str:
    """3.5×10⁻³ in mathtext."""
    if not (isinstance(x, float) and math.isfinite(x)) or x <= 0:
        return "—"
    e = int(math.floor(math.log10(x)))
    return r"%.2g\times10^{%d}" % (x / 10 ** e, e)


def _plain_log_y(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(0.2, 0.5, 1.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def predicted_vs_observed(D: pd.DataFrame, methods, predictor: str = "depolarizing", *, z_sig: float = 3.0,
                          title: str = "", ncols: int = 3):
    """Predicted against observed F_c, one panel per method: above the diagonal the device is worse than its model.
    Markers: shape = family; filled red / blue beyond the shot noise of the budget (z > z_sig / z < −z_sig), open grey
    within it. Equal axes over the range of the data."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    methods = list(methods)
    ncols = max(1, min(ncols, len(methods)))
    nrows = int(math.ceil(len(methods) / ncols))
    w = TEXT_WIDTH if ncols > 1 else COLUMN_WIDTH + 0.3
    fig, axs = plt.subplots(nrows, ncols, figsize=(w, w / ncols * nrows + 0.55), squeeze=False,
                            constrained_layout=True)
    sub = D[(D["predictor"] == predictor) & D["method"].isin(methods)]
    if "status" in sub.columns:
        sub = sub[sub["status"] == "ok"]
    vals = pd.concat([pd.to_numeric(sub["fc_obs"], errors="coerce"), pd.to_numeric(sub["fc_pred"], errors="coerce")])
    lo = max(0.0, float(vals.min()) - 0.03) if vals.notna().any() else 0.0
    lo = math.floor(lo * 20) / 20
    for i, m in enumerate(methods):
        ax = axs[i // ncols][i % ncols]
        d = sub[sub["method"] == m]
        ax.plot([lo, 1.0], [lo, 1.0], color=MUTED, lw=0.6, zorder=0)
        for fam, g in d.groupby("family"):
            cls = _signif_class(pd.to_numeric(g.get("z"), errors="coerce"), z_sig)
            for c, face, edge in ((0, "white", NEUTRAL_MARK), (1, WORSE, WORSE), (-1, BETTER, BETTER)):
                gg = g[cls == c]
                if len(gg):
                    ax.plot(gg["fc_obs"].astype(float), gg["fc_pred"].astype(float), ls="",
                            marker=FAMILY_MARKERS.get(fam, "o"), ms=3.4, mfc=face, mec=edge if c else NEUTRAL_MARK,
                            mew=0.6, zorder=2 + abs(c))
        ax.set_xlim(lo, 1.015)
        ax.set_ylim(lo, 1.015)
        ax.set_aspect("equal")
        ax.set_title(m, loc="left")
        ax.set_xlabel(r"observed $F_c$")
        if i % ncols == 0:
            ax.set_ylabel(r"predicted $F_c$ (%s model)" % predictor)
        ax.text(0.04, 0.96, "device worse", transform=ax.transAxes, fontsize=6.3, color=INK2, va="top")
        ax.text(0.96, 0.04, "device better", transform=ax.transAxes, fontsize=6.3, color=INK2, ha="right")
    for j in range(len(methods), nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    hs = [Line2D([], [], ls="", marker="o", ms=3.4, mfc=WORSE, mec=WORSE, label="z > %g" % z_sig),
          Line2D([], [], ls="", marker="o", ms=3.4, mfc=BETTER, mec=BETTER, label="z < −%g" % z_sig),
          Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=NEUTRAL_MARK, label="within shot noise")]
    hs += _family_handles(sub["family"])
    fig.legend(handles=hs, loc="outside lower center", ncol=min(len(hs), 10), fontsize=6.3, handletextpad=0.15,
               columnspacing=0.7)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8.5)
    return fig


# ------------------------------------------------------------------ why: context mechanisms (dirA_divergence.analyse)
MECHANISM_ORDER = ["priced", "gate_error", "n2q", "active_spectators", "passive_spectators", "control_spectators",
                   "parallel_adjacent", "idle_us", "readout_error", "measured", "n1q"]
MECHANISM_COLORS = {   # categorical slots of the reference palette in a fixed order; never cycled
    "priced": "#2a78d6", "gate_error": "#2a78d6", "n2q": "#2a78d6",          # what the calibration prices
    "active_spectators": "#eb6834", "passive_spectators": "#eda100", "control_spectators": "#e87ba4",
    "parallel_adjacent": "#6250d6", "idle_us": "#1baf7a", "readout_error": "#9a9994", "measured": "#9a9994",
    "n1q": "#008300", "unexplained": "#e4e3df"}
MECHANISM_LABEL = {"priced": "rescaled calibration", "gate_error": "calibrated gate error",
                   "n2q": "two-qubit gates (count)", "n1q": "one-qubit gates", "active_spectators": "active spectators",
                   "passive_spectators": "passive spectators", "control_spectators": "CR control spectators",
                   "parallel_adjacent": "parallel adjacent gates", "idle_us": "idle time",
                   "readout_error": "calibrated readout error", "measured": "readout", "unexplained": "unexplained"}


def divergence_budget(coef: pd.DataFrame, budget: pd.DataFrame, title: str = ""):
    """Where the unmodeled error comes from. (a) Each mechanism's share of the excess log-infidelity, as the mean
    contribution per circuit with the bootstrap 95 % interval of its coefficient (hollow: the circuits cannot tell it
    from the others, VIF > 10). (b) The same per family, stacked, with what the mechanisms leave unexplained
    (hatched; it may be negative). All in one unit — excess log-infidelity per circuit — so the bars compare."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    c = coef[np.isfinite(pd.to_numeric(coef["coef"], errors="coerce"))].copy()
    fams = []
    if budget is not None and len(budget):
        tot = budget[budget["mechanism"] != "unexplained"].groupby("family")["contribution"].sum()
        fams = list(tot.sort_values(ascending=False).index)
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.26 * max(len(c), len(fams), 4) + 1.35),
                            gridspec_kw=dict(width_ratios=[1, 1.35]), constrained_layout=True)
    ax = axs[0]
    if budget is not None and len(budget) and "circuits" in budget.columns:
        per = budget.drop_duplicates(["family", "mechanism"])
        wts = per.groupby("family")["circuits"].first()
        mean = (per.assign(w=per["family"].map(wts)).assign(cw=lambda t: t["contribution"] * t["w"])
                .groupby("mechanism")["cw"].sum() / max(float(wts.sum()), 1.0))
    else:
        mean = pd.Series(dtype=float)
    order = [m for m in MECHANISM_ORDER if m in set(c["mechanism"])] + \
        [m for m in c["mechanism"] if m not in MECHANISM_ORDER]
    c = c.set_index("mechanism").loc[order].reset_index()
    for i, r in c.iterrows():
        m = r["mechanism"]
        v = float(mean.get(m, 0.0))
        b = float(r["coef"])
        lo = v * float(r["lo"]) / b if b > 0 and np.isfinite(r["lo"]) else np.nan
        hi = v * float(r["hi"]) / b if b > 0 and np.isfinite(r["hi"]) else np.nan
        col = MECHANISM_COLORS.get(m, INK2)
        poor = np.isfinite(r.get("vif", np.nan)) and r["vif"] > 10
        ax.barh(i, v, height=0.56, color="white" if poor else col, edgecolor=col, linewidth=0.8)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=INK, lw=0.7, solid_capstyle="butt")
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels([MECHANISM_LABEL.get(m, m) for m in c["mechanism"]])
    ax.invert_yaxis()
    ax.axvline(0, color=BASELINE, lw=0.6)
    ax.set_xlabel("excess log-infidelity per circuit")
    ax.set_title("(a) by mechanism", loc="left")
    ax.tick_params(axis="y", length=0)
    ax = axs[1]
    mechs = []
    if budget is not None and len(budget):
        used = budget.groupby("mechanism")["contribution"].apply(lambda v: float(np.abs(v).sum()))
        mechs = [m for m in order + ["unexplained"] if used.get(m, 0) > 0]
        for j, fam in enumerate(fams):
            g = budget[budget["family"] == fam].set_index("mechanism")["contribution"]
            pos, neg = 0.0, 0.0
            for m in mechs:
                v = float(g.get(m, 0.0))
                if v == 0 or not np.isfinite(v):
                    continue
                start = pos if v > 0 else neg + v
                un = m == "unexplained"
                ax.barh(j, abs(v), left=start, height=0.56, color=MECHANISM_COLORS.get(m, INK2),
                        edgecolor=MUTED if un else SURFACE, linewidth=0.4 if un else 0.8,
                        hatch="//////" if un else None)
                if v > 0:
                    pos += v
                else:
                    neg += v
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([FAMILY_LABEL.get(f, f) for f in fams])
    ax.invert_yaxis()
    ax.axvline(0, color=BASELINE, lw=0.6)
    ax.set_xlabel("excess log-infidelity per circuit")
    ax.set_title("(b) by circuit family", loc="left")
    ax.tick_params(axis="y", length=0)
    if mechs:
        hs = [Patch(facecolor=MECHANISM_COLORS.get(m, INK2), edgecolor=MUTED if m == "unexplained" else SURFACE,
                    linewidth=0.4, hatch="//////" if m == "unexplained" else None, label=MECHANISM_LABEL.get(m, m))
              for m in mechs]
        if any(np.isfinite(c["vif"].astype(float)) & (c["vif"].astype(float) > 10)):
            hs.append(Patch(facecolor=SURFACE, edgecolor=INK2, linewidth=0.8, label="hollow: VIF > 10 (not separable)"))
        fig.legend(handles=hs, loc="outside lower center", ncol=min(len(hs), 4), fontsize=6.5, handlelength=1.2)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8.5)
    return fig


def place_labels(ax, xs, ys, texts, fontsize: float = 6.5, pad: float = 4.0, **kw):
    """Direct labels next to points without overlaps: each label takes the first of eight positions around its
    point (right, left, above, below, then the diagonals, at growing distance) whose box overlaps neither a placed
    label nor another labelled point; a thin leader line joins a label that had to move away. Greedy, in order of
    decreasing y; boxes are estimated from the font size (no renderer needed)."""
    fig = ax.figure
    T = ax.transData
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    pts = np.array([T.transform((float(a), float(b))) for a, b in zip(xs, ys)]) * 72.0 / fig.dpi   # in points
    hgt = fontsize * 1.15
    placed, out = [], []
    order = np.argsort(-pts[:, 1]) if len(pts) else []
    for i in order:
        px, py = pts[i]
        w = 0.5 * fontsize * len(str(texts[i])) + 1.0
        cands = []
        for dist in (pad, pad + 6, pad + 13, pad + 22):
            cands += [(dist, -hgt / 2), (-dist - w, -hgt / 2), (-w / 2, dist), (-w / 2, -dist - hgt),
                      (dist, dist * 0.5), (dist, -dist * 0.5 - hgt), (-dist - w, dist * 0.5),
                      (-dist - w, -dist * 0.5 - hgt)]
        choice = cands[0]
        for dx, dy in cands:
            box = (px + dx, py + dy, px + dx + w, py + dy + hgt)
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
            hit = hit or any(j != i and box[0] - 2 < q[0] < box[2] + 2 and box[1] - 2 < q[1] < box[3] + 2
                             for j, q in enumerate(pts))
            if not hit:
                choice = (dx, dy)
                break
        dx, dy = choice
        placed.append((px + dx, py + dy, px + dx + w, py + dy + hgt))
        far = math.hypot(dx + (w / 2 if dx < 0 else 0), dy + hgt / 2) > pad + 8
        out.append(ax.annotate(str(texts[i]), (float(xs[i]), float(ys[i])),
                               xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=INK,
                               ha="left", va="bottom", zorder=6,
                               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=0, shrinkB=2.5)
                               if far else None, **kw))
    return out


FINGERPRINT_TINT = {"time": "#e6f4ec", "coherent": "#fbe9e2"}


def folding_fingerprint(pairs: pd.DataFrame, title: str = "", summary: pd.DataFrame | None = None):
    """The folding fingerprint: how the device answers noise amplification compared with its model. Each axis is an
    amplification anomaly A = κ_obs/κ_pred (κ: decay rate of ln F against the fold factor λ) — horizontally for
    global folding (gates and idle windows repeated), vertically for local folding (only the two-qubit gates). Every
    circuit is a small grey marker; every family's pooled anomaly (weighted fit, ±1 SE; ``summary`` from
    dirA_divergence.fingerprint_summary) is labelled. Along the diagonal above (1, 1): unmodeled error on the gates;
    below it (shaded green): attached to time; above it (shaded red): coherent build-up under repetition."""
    import matplotlib.pyplot as plt
    A_g, A_l = pairs["A_global"].astype(float), pairs["A_local"].astype(float)
    d = pairs[np.isfinite(A_g) & np.isfinite(A_l) & (A_g > 0) & (A_l > 0)]
    hidden = int((np.isfinite(A_g) & np.isfinite(A_l)).sum()) - len(d)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.35, COLUMN_WIDTH + 0.25))
    if title:
        ax.set_title(title, loc="left")
    if d.empty:
        _no_data(ax, "no circuit with both foldings above the fidelity floor")
        return fig
    S = None
    if summary is not None and len(summary):
        S = summary[(summary["family"] != "all") & np.isfinite(summary["A_global"].astype(float))
                    & np.isfinite(summary["A_local"].astype(float))]
    pts = [d["A_global"], d["A_local"]] + ([S["A_global"], S["A_local"]] if S is not None else [])
    allv = pd.concat(pts).astype(float)
    lo = max(min(float(allv.quantile(0.02)), 0.8) * 0.85, 0.05)
    hi = max(float(allv.quantile(0.98)), 1.25) * 1.15
    xs = np.geomspace(lo, hi, 50)
    ax.fill_between(xs, lo, xs, color=FINGERPRINT_TINT["time"], lw=0, zorder=0)
    ax.fill_between(xs, xs, hi, color=FINGERPRINT_TINT["coherent"], lw=0, zorder=0)
    ax.plot(xs, xs, color=MUTED, lw=0.6, zorder=1)
    ax.axvline(1, color=BASELINE, lw=0.6, zorder=1)
    ax.axhline(1, color=BASELINE, lw=0.6, zorder=1)
    dd = d.assign(A_global=d["A_global"].clip(lo, hi), A_local=d["A_local"].clip(lo, hi))
    for fam, g in dd.groupby("family"):
        ax.plot(g["A_global"], g["A_local"], ls="", marker=FAMILY_MARKERS.get(fam, "o"), ms=3.0, mfc="none",
                mec=NEUTRAL_MARK, mew=0.5, zorder=2)
    if S is not None:
        for _, r in S.iterrows():
            x, y = float(np.clip(r["A_global"], lo, hi)), float(np.clip(r["A_local"], lo, hi))
            sx = float(r.get("A_global_se", np.nan))
            sy = float(r.get("A_local_se", np.nan))
            if np.isfinite(sx) and np.isfinite(sy):
                ax.errorbar([x], [y], xerr=[[min(sx, x - lo)], [sx]], yerr=[[min(sy, y - lo)], [sy]], fmt="none",
                            ecolor=INK, elinewidth=0.6, capsize=0, zorder=3)
            ax.plot([x], [y], ls="", marker=FAMILY_MARKERS.get(r["family"], "o"), ms=5.2, mfc=INK, mec="white",
                    mew=0.6, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    plain_log_axis(ax, "x")
    plain_log_axis(ax, "y")
    ax.set_xlabel(r"$A_{\rm global}=\kappa_{\rm obs}/\kappa_{\rm pred}$ (global folding)")
    ax.set_ylabel(r"$A_{\rm local}$ (local folding)")
    kw = dict(transform=ax.transAxes, fontsize=6.5, color=INK2)
    ax.text(0.97, 0.04, "time-attached\n(idle, drift)", ha="right", va="bottom", **kw)
    ax.text(0.04, 0.96, "coherent build-up\nunder repetition", ha="left", va="top", **kw)
    ax.text(0.97, 0.97, "gate-attached", ha="right", va="top", **kw)
    corner = ["+  as modeled (1, 1)"] if lo < 1 < hi else []
    if hidden:
        corner.append("%d circuit%s with A ≤ 0 not shown" % (hidden, "s" if hidden > 1 else ""))
    if corner:
        ax.text(0.04, 0.04, "\n".join(corner), ha="left", va="bottom", **kw)
    if lo < 1 < hi:
        ax.plot([1], [1], marker="+", ms=7, color=INK, mew=0.9, zorder=5)
    fig.tight_layout()
    if S is not None and len(S):
        import matplotlib.patheffects as pe
        place_labels(ax, np.clip(S["A_global"].astype(float).to_numpy(), lo, hi),
                     np.clip(S["A_local"].astype(float).to_numpy(), lo, hi),
                     [FAMILY_LABEL.get(f, f) for f in S["family"]], fontsize=6.5,
                     path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    return fig


def heavy_hex_map(Q: pd.DataFrame, edges, num_qubits: int, title: str = "", label_top: int = 5):
    """Where on the chip the model under-predicts: every qubit of the device at its heavy-hex position; the qubits
    the circuits used are filled by their unmodeled error per two-qubit-gate participation (warm sequential scale,
    0 = as modeled), ringed in ink where the bootstrap interval excludes 0; the ``label_top`` worst carry their
    index; unused qubits are small grey dots."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from qemx.context_noise import heavy_hex_coordinates
    pos = heavy_hex_coordinates([tuple(e) for e in edges], int(num_qubits))
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    span_x, span_y = float(np.ptp(xs)) + 1, float(np.ptp(ys)) + 1
    w = TEXT_WIDTH * 0.78
    fig, ax = plt.subplots(figsize=(w, min(w * span_y / span_x + 0.55, 6.0)), constrained_layout=True)
    ax.add_collection(LineCollection([[pos[a], pos[b]] for a, b in edges], colors="#dcdbd5", linewidths=0.8,
                                     zorder=1))
    q = Q.set_index("qubit") if len(Q) else Q
    used = set(int(k) for k in q.index) if len(q) else set()
    free = [k for k in range(int(num_qubits)) if k not in used]
    ax.scatter([pos[k][0] for k in free], [pos[k][1] for k in free], s=6, color="#e4e3df", lw=0, zorder=2)
    vmax = float(q["coef"].max()) if len(q) and np.isfinite(q["coef"]).any() and q["coef"].max() > 0 else 1e-3
    norm, cmap = Normalize(0, vmax), excess_cmap()
    if len(q):
        ks = sorted(used)
        vals = [max(float(q.loc[k, "coef"]), 0.0) for k in ks]
        sig = [np.isfinite(q.loc[k, "lo"]) and q.loc[k, "lo"] > 0 for k in ks]
        ax.scatter([pos[k][0] for k in ks], [pos[k][1] for k in ks], s=34, c=[cmap(norm(v)) for v in vals],
                   edgecolors=[INK if s_ else "#a9a8a3" for s_ in sig], linewidths=[1.0 if s_ else 0.4 for s_ in sig],
                   zorder=3)
        sigq = q[np.isfinite(q["lo"]) & (q["lo"] > 0)].index
        top = q[q["coef"] > 0.1 * vmax].sort_values("coef", ascending=False).head(int(label_top)).index
        for k in list(dict.fromkeys(list(top) + list(sigq)))[:int(label_top) + 3]:
            ax.annotate(str(int(k)), pos[int(k)], xytext=(0, 5.5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=6.3, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.02)
    _horizontal_colorbar(fig, ScalarMappable(norm=norm, cmap=cmap), ax,
                         "unmodeled error per two-qubit gate on the qubit (0: as modeled)", ends=None,
                         fraction=0.035, pad=0.01, aspect=40, shrink=0.6)
    if title:
        ax.set_title(title, loc="left")
    return fig


PROBE_COLORS = {"zero": "#2a78d6", "plus": "#eb6834", "driven": "#1baf7a"}     # slots 1–3 (validated all-pairs)
PROBE_LABEL = {"zero": r"idle $|0\rangle$", "plus": r"idle $|{+}\rangle$", "driven": "driven"}


def context_probe(C: pd.DataFrame, title: str = ""):
    """The spectator-context probes: the same two-qubit core with its two spectators idle in |0⟩, idle in |+⟩, or
    driven by their own CZ gates. (a) F_c against the core depth L, observed (markers, solid) and predicted
    (dashed), mean over layout variants; (b) the excess contrast [F_c(idle |0⟩) − F_c(mode)]_obs − [·]_pred with
    ±2 SE — what the spectators cost on the device beyond what the model says. Superposed spectators alone: idle
    dephasing; driven alone: gate error; both: spectator crosstalk."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35), constrained_layout=True)
    if C is None or C.empty:
        _no_data(axs[0], "no context probes")
        return fig
    g = C.groupby(["spectators", "layers"])
    m = g[["fc_obs", "fc_pred", "contrast_excess"]].mean().reset_index()
    if "contrast_excess_se" in C.columns:
        se = g["contrast_excess_se"].apply(lambda v: float(np.sqrt(np.nansum(np.square(v)))) / max(len(v), 1))
        m = m.merge(se.rename("se").reset_index(), on=["spectators", "layers"])
    else:
        m["se"] = g["contrast_excess"].std().reset_index(drop=True) / np.sqrt(g.size().reset_index(drop=True))
    ax = axs[0]
    for mode in ("zero", "plus", "driven"):
        s = m[m["spectators"] == mode].sort_values("layers")
        if s.empty:
            continue
        col = PROBE_COLORS[mode]
        ax.plot(s["layers"], s["fc_pred"], color=col, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.plot(s["layers"], s["fc_obs"], color=col, lw=1.1, marker="o", ms=3.4, mec="white", mew=0.5, zorder=2)
        ax.annotate(PROBE_LABEL[mode], (float(s["layers"].iloc[-1]), float(s["fc_obs"].iloc[-1])), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.5, color=INK, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(m["layers"].unique()))
    ax.set_xticklabels([str(int(v)) for v in sorted(m["layers"].unique())])
    ax.set_xlabel("core layers $L$")
    ax.set_ylabel(r"$F_c$")
    ax.set_title("(a) same core, three spectator contexts", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.1, marker="o", ms=3, label="observed"),
                       Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 2)), label="predicted")], loc="lower left",
              fontsize=6.5)
    ax.margins(x=0.25)
    ax = axs[1]
    for k, mode in enumerate(("plus", "driven")):
        s = m[m["spectators"] == mode].sort_values("layers")
        if s.empty:
            continue
        col = PROBE_COLORS[mode]
        xx = s["layers"].astype(float) * (1.04 if k else 0.96)
        ax.errorbar(xx, s["contrast_excess"], yerr=2 * s["se"], color=col, lw=1.1, marker="o", ms=3.4, mec="white",
                    mew=0.5, elinewidth=0.7, capsize=0, label=PROBE_LABEL[mode])
    ax.axhline(0, color=BASELINE, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(m["layers"].unique()))
    ax.set_xticklabels([str(int(v)) for v in sorted(m["layers"].unique())])
    ax.set_xlabel("core layers $L$")
    ax.set_ylabel("excess contrast (obs − pred)")
    ax.set_title("(b) what the spectators cost beyond the model", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8.5)
    return fig


def attribute(fig, text: str):
    """A source line under everything else in the figure (right-aligned, muted): the device and date of a run, or
    "emulated: fake_fez + planted error". The layout is drawn and frozen first, so the line can be placed below its
    lowest artist."""
    fig.canvas.draw()
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    for L in fig.legends:          # pin figure legends where the layout put them (a tight save re-anchors them)
        e = L.get_window_extent(r)
        L.set_loc("lower left")
        L.set_bbox_to_anchor((e.x0 / fig.bbox.width, e.y0 / fig.bbox.height), transform=fig.transFigure)
        L.borderaxespad = 0
    bb = fig.get_tightbbox(r)
    W, H = fig.get_figwidth(), fig.get_figheight()
    fig.text(min(bb.x1, W) / W, (bb.y0 - 0.04) / H, text, ha="right", va="top", fontsize=5.5, color=MUTED)
    return fig


def ladder_chart(L: pd.DataFrame, title: str = "", verdict: str = ""):
    """The χ² ladder of dirA_divergence.model_ladder: χ²/dof of the nested explanations of the excess log-infidelity
    (none; a rescaled calibration; context mechanisms; both). 1 is what a sufficient model reaches; the grey tick is
    the 99 % point of χ²/dof for that model's degrees of freedom."""
    import matplotlib.pyplot as plt
    from scipy.stats import chi2 as CHI2
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.4, 1.75))
    if L is None or L.empty:
        _no_data(ax, "no χ² ladder")
        return fig
    names = {"none": "none (model right)", "calibration (priced)": "rescaled calibration", "context": "context",
             "calibration + context": "calibration + context", "per-family rescaling": "per-family rescaling"}
    y = np.arange(len(L))
    v = L["chi2_per_dof"].astype(float).to_numpy()
    crit = np.array([CHI2.ppf(0.99, max(int(d), 1)) / max(int(d), 1) for d in L["dof"]])
    cols = [WORSE if vv > c else NEUTRAL_MARK for vv, c in zip(v, crit)]
    ax.barh(y, v, height=0.55, color=cols)
    for yi, c in zip(y, crit):
        ax.plot([c, c], [yi - 0.36, yi + 0.36], color=INK2, lw=0.8)
    for yi, vv in zip(y, v):
        ax.text(vv, yi, "  %.2f" % vv, va="center", fontsize=6.5, color=INK)
    ax.axvline(1, color=INK2, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([names.get(m, m) for m in L["model"]])
    ax.invert_yaxis()
    ax.set_xscale("log")
    plain_log_axis(ax, "x")
    ax.set_xlabel(r"$\chi^2/{\rm dof}$ of the weighted fit")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(0.7, float(np.nanmin(v)) * 0.8), float(np.nanmax(v)) * 2.2)
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    if verdict:
        import textwrap
        ax.annotate("\n".join(textwrap.wrap("Verdict — " + verdict, 92)), xy=(0.01, 0),
                    xycoords=("figure fraction", "axes fraction"), xytext=(0, -27), textcoords="offset points",
                    fontsize=6.3, color=INK2, va="top", ha="left", annotation_clip=False)
    return fig


# ------------------------------------------------------------------ the diagnostic matrix (dirA_divergence recovery)
def _rnum(r, k):
    """Row r's value of k as a float (NaN when missing or not a number)."""
    v = r.get(k, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def position_flag(r) -> bool:
    """Layout dependence: more benchmarks with disagreeing variants (Cochran p < 0.01) than chance allows — at
    least 3 and over 10 % of them (the null expectation is 1 %)."""
    k, n = _rnum(r, "het_sig"), _rnum(r, "het_groups")
    return bool(np.isfinite(k) and np.isfinite(n) and n > 0 and k >= 3 and k / n > 0.1)


def position(r) -> str:
    k, n = _rnum(r, "het_sig"), _rnum(r, "het_groups")
    if not (np.isfinite(k) and np.isfinite(n)):
        return ""
    txt = "%s\n%d of %d benchmarks differ" % ("layout-dependent" if position_flag(r) else "layout-independent", k, n)
    wq = r.get("worst_qubits", "")
    if position_flag(r) and isinstance(wq, str) and wq:
        txt += "\nworst qubits %s" % wq
    return txt


def probe_reading(significant) -> str:
    """What the spectator probes say from which contexts show a significant excess contrast: superposed spectators
    alone → idle dephasing (they wait in superposition); driven spectators alone → gate error (the driven mode adds
    two-qubit gates next to the core); both → spectator crosstalk (ZZ / drive); none → no spectator effect."""
    s = set(significant)
    if not s:
        return "no spectator effect"
    if s == {"plus"}:
        return "idle dephasing"
    if s == {"driven"}:
        return "gate error"
    return "spectator crosstalk"


NULL_READINGS = {"as modeled", "no spectator effect", "layout-independent", "— (nothing to attribute)", ""}


def _split_share(item: str) -> tuple:
    """'parallel adjacent gates 45%' → ('parallel adjacent gates', '45 %')."""
    name, _, pct = item.strip().rpartition(" ")
    return (name, pct.replace("%", " %")) if pct.endswith("%") and name else (item.strip(), "")


def diagnostic_matrix(R: pd.DataFrame, title: str = "", z: float = 3.0):
    """The diagnostic matrix: one row per data set (emulations with a known planted error, then the device), one
    column per diagnostic — the χ² ladder (does the gap need context?), the divergence budget (which context), the
    folding fingerprint (gate, time or coherent), the spectator probes, the layout variants (position), and the
    effective error-rate ratio. Each cell: the reading — after a filled red dot when the diagnostic finds an
    anomaly, an open grey ring when it reads "as modeled" — and its key numbers underneath in grey. A table drawn as
    one (rows grow with their text; nothing overflows its column)."""
    import textwrap
    import matplotlib.pyplot as plt
    cols = [("ladder", "χ² ladder", 1.0), ("budget", "divergence budget", 1.12), ("fingerprint",
            "folding fingerprint", 1.12), ("probes", "spectator probes", 1.0), ("position", "layout variants", 1.0),
            ("eps", r"$\varepsilon_{\rm eff}^{\rm obs}/\varepsilon_{\rm eff}^{\rm pred}$", 0.55)]
    fig_w = TEXT_WIDTH
    lab_w = 0.95                                          # inches
    unit = (fig_w - lab_w) / sum(c[2] for c in cols)
    widths = [c[2] * unit for c in cols]
    x0 = lab_w + np.concatenate([[0], np.cumsum(widths)[:-1]])
    fs_h, fs_d, fs_l = 6.5, 5.8, 7.0
    dot = 0.13                                            # inches taken by the status dot

    def wrap(text, width_in, fs):
        n = max(8, int((width_in - dot - 0.04) * 72 / (0.43 * fs)))
        out = []
        for part in str(text).split("\n"):
            out += textwrap.wrap(part, n, break_on_hyphens=False) or [""]
        return [s_ for s_ in out if s_ != ""]

    rows = []
    for _, r in R.iterrows():
        sig = [m for m in ("plus", "driven") if np.isfinite(_rnum(r, "probe_%s_z" % m))
               and _rnum(r, "probe_%s_z" % m) > z]
        fp = str(r.get("fingerprint", "") or "")
        fp_head, _, fp_rest = fp.partition(" (")
        lad = str(r.get("ladder", "") or "")
        lad_head, _, lad_k = lad.replace(" + non-linear", "").partition(" ×")
        items = [b for b in str(r.get("budget", "") or "").split(",") if b.strip()]
        b_head, b_pct = _split_share(items[0]) if items else ("", "")
        pos_txt = position(r).split("\n")
        cells = {
            "ladder": (lad_head, "\n".join(x for x in (("×" + lad_k) if lad_k else "",
                                                       "+ non-linear" if "non-linear" in lad else "") if x)),
            "budget": (b_head, "\n".join([b_pct] + ["%s %s" % _split_share(b) for b in items[1:3]]).strip()),
            "fingerprint": (fp_head, fp_rest.rstrip(")").replace("A_g", "$A_g$").replace("A_l", "$A_l$")
                            .replace(", ", "\n")),
            "probes": (probe_reading(sig), "\n".join("%s z = %.1f" % (m, _rnum(r, "probe_%s_z" % m))
                                                     for m in ("plus", "driven")
                                                     if np.isfinite(_rnum(r, "probe_%s_z" % m)))),
            "position": (pos_txt[0] if pos_txt else "", "\n".join(pos_txt[1:])),
            "eps": ("%.2f" % _rnum(r, "eps_ratio") if np.isfinite(_rnum(r, "eps_ratio")) else "", ""),
        }
        lines = {k: (wrap(h, w, fs_h), wrap(d_, w, fs_d), h in NULL_READINGS or h.startswith("—"))
                 for (k, _, _), w, (h, d_) in zip(cols, widths, [cells[c[0]] for c in cols])}
        label = wrap(r["data"], lab_w + dot, fs_l)
        need = max([len(h) * fs_h * 1.2 + len(d_) * fs_d * 1.2 for h, d_, _ in lines.values()]
                   + [len(label) * fs_l * 1.2]) / 72 + 0.12
        rows.append((label, lines, max(need, 0.36)))
    head_h = 0.3
    fig_h = head_h + sum(r_[2] for r_ in rows) + (0.2 if title else 0.04)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")
    for j, (_, head, _) in enumerate(cols):
        ax.text(x0[j] + 0.03, head_h / 2, head, fontsize=fs_h + 0.5, fontweight="bold", color=INK, va="center")
    ax.plot([0.03, fig_w - 0.03], [head_h, head_h], color=INK2, lw=0.6)
    y = head_h
    for label, lines, h in rows:
        top = y + 0.07
        ax.text(0.05, top, "\n".join(label), fontsize=fs_l, color=INK, va="top", fontweight="bold",
                linespacing=1.15)
        for j, (key, _, _) in enumerate(cols):
            hl, dl, null = lines[key]
            if not hl:
                continue
            tx = x0[j] + 0.03
            if key != "eps":
                ax.plot([tx + 0.045], [top + fs_h * 0.6 / 72], marker="o", ms=3.4, mfc="none" if null else WORSE,
                        mec=NEUTRAL_MARK if null else WORSE, mew=0.7)
                tx += dot
            ax.text(tx, top, "\n".join(hl), fontsize=fs_h, color=INK, va="top", linespacing=1.15)
            if dl:
                ax.text(tx, top + len(hl) * fs_h * 1.2 / 72, "\n".join(dl), fontsize=fs_d, color=INK2, va="top",
                        linespacing=1.15)
        y += h
        ax.plot([0.03, fig_w - 0.03], [y, y], color=GRID, lw=0.5)
    if title:
        ax.text(0.05, y + 0.06, title, fontsize=fs_d, color=INK2, va="top")
    return fig


# ------------------------------------------------------------------ driver
def make_figures(T: Tables, sink, rep: Report, *, only=None, eps=None, methods="run2") -> list:
    """The default figure set of a run, saved through ``sink`` (computation and drawing only; the sink decides where
    the figures go). Skipped figures are noted in ``rep``. Returns the saved names."""
    import matplotlib.pyplot as plt
    bias_methods = BIAS_METHODS
    if methods == "nil":
        use_methods(NIL_MAP_METHODS)
        bias_methods = BIAS_METHODS + ["NIL-G2"]
    want = set(only) if only is not None else {"phase", "family", "curves", "budget", "bias", "transfer",
                                                "validation"}
    made = []
    sfx = ("" if eps is None else "_eps%g" % eps) + ("_nil" if methods == "nil" else "")
    stages = [s for s in ("A1-primary", "A2-gate-only") if (T.cells.stage == s).any()]
    rep.section("FIGURES")

    def skip(what, e):
        rep.text("skip %s: %s" % (what, e))
    if "phase" in want:
        for st in stages:
            for n in sorted(T.cells[T.cells.stage == st].n.unique()):
                for pol in ("fixed", "opt"):
                    for cand in ("all", "no_optimistic"):
                        try:
                            fig = phase_grid(T, st, int(n), policy=pol, candidates=cand, eps=eps)
                            made.append(sink.save(fig, "phase_%s_n%d_%s_%s%s" % (st, n, pol, cand, sfx)))
                        except ValueError as e:
                            skip("phase grid %s n=%d %s %s" % (st, n, pol, cand), e)
        if (T.cells.stage == "B-coherent").any():
            for chi in sorted(T.cells[T.cells.stage == "B-coherent"].coherent_fraction.unique()):
                for n in sorted(T.cells[T.cells.stage == "B-coherent"].n.unique()):
                    try:
                        fig = phase_grid(T, "B-coherent", int(n), candidates="no_optimistic", coherent_fraction=chi,
                                         eps=eps)
                        made.append(sink.save(fig, "phase_B-coherent_n%d_chi%g%s" % (n, chi, sfx)))
                    except ValueError as e:
                        skip("phase grid B-coherent n=%d χ=%g" % (n, chi), e)
    if "family" in want:
        for st in stages:
            fams = sorted(T.cells[T.cells.stage == st].family.unique())
            nc = 2
            nr = int(math.ceil(len(fams) / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.7 * nr + 0.6), squeeze=False)
            grids = []
            for i, fam in enumerate(fams):
                ax = axs[i // nc][i % nc]
                try:
                    _, g = family_map(T, st, fam, candidates="no_optimistic", ax=ax)
                except ValueError as e:
                    skip("family map %s %s" % (st, fam), e)
                    ax.set_visible(False)
                    continue
                ax.set_title(fam, loc="left", pad=12)
                grids.append(g)
            for j in range(len(fams), nr * nc):
                axs[j // nc][j % nc].set_visible(False)
            fig.suptitle("%s · majority winner over all circuits of a family (VD excluded)" % st, x=0.01, ha="left",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0.6 / (2.7 * nr + 0.6), 1, 1))
            _legend(fig, *grids, ncol=3, extra="hatch")
            made.append(sink.save(fig, "family_%s%s" % (st, "_nil" if methods == "nil" else "")))
    if "curves" in want or "budget" in want:
        for st in stages[:1]:
            reps_ = [(f, int(n), p) for n in sorted(T.cells[T.cells.stage == st].n.unique())
                     for f, p in REPRESENTATIVE.get(int(n), [])]
            for f, n, p in reps_:
                cid = circuit_id(f, n, **p)
                cs = T.cells[(T.cells.stage == st) & (T.cells.circuit == cid)].sort_values("scale")
                if cs.empty:
                    continue
                if "curves" in want and T.ledger is not None:
                    one = cs.iloc[(cs.scale - 1.0).abs().argmin()].name
                    made.append(sink.save(rmse_curves(T, one), "rmse_%s_%s%s" % (st, _slug(cs.label.iloc[0]), sfx)))
                if "budget" in want:
                    fig, ax = plt.subplots(figsize=(4.0, 3.1))
                    budget_curves(T, st, cid, 0.01, ax=ax)
                    made.append(sink.save(fig, "bstar_%s_%s%s" % (st, _slug(cs.label.iloc[0]), sfx)))
    if "transfer" in want and (T.cells.stage == "C-transfer").any():
        for n in sorted(T.cells[T.cells.stage == "C-transfer"].n.unique()):
            for pol in ("fixed", "opt"):
                try:
                    made.append(sink.save(transfer_grid(T, int(n), policy=pol), "transfer_n%d_%s%s" % (n, pol, sfx)))
                except ValueError as e:
                    skip("transfer n=%d %s" % (n, pol), e)
    if "bias" in want:
        for st in stages:
            for n in sorted(T.cells[T.cells.stage == st].n.unique()):
                try:
                    made.append(sink.save(bias_fraction(T, st, int(n), methods=bias_methods),
                                          "bias_%s_n%d%s" % (st, n, sfx)))
                except ValueError as e:
                    skip("bias %s n=%d" % (st, n), e)
    if "validation" in want and T.validation is not None and len(T.validation):
        try:
            made.append(sink.save(validation(T), "validation_z"))
        except ValueError as e:
            skip("validation", e)
    rep.text("%d figures saved" % len(made))
    return made


def main(argv=None):
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="the run directory (contains tables/)")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--only", choices=["health", "phase", "family", "curves", "budget", "bias", "transfer",
                                       "validation"], nargs="*")
    ap.add_argument("--eps", type=float, default=None,
                    help="target RMSE: veil the phase-map tiles where no method reaches it and draw B_min(eps); "
                         "files get the suffix _eps<value>")
    ap.add_argument("--methods", choices=["run2", "nil"], default="run2",
                    help="run2: the eight methods of direction_A_v1 (default); nil: VD replaced by neighbor-informed "
                         "learning NIL-G2 (direction_A_nil_v1 stores; files get the suffix _nil)")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--font", choices=sorted(FONT_PRESETS), default=None,
                    help="type preset: times (TeX Gyre Termes, default), lm (Latin Modern), sans (slides)")
    a = ap.parse_args(argv)
    setup_matplotlib(a.font)
    warnings.simplefilter("ignore")
    T = load(a.data)                                                       # input
    rep = Report()
    want = set(a.only or ["health", "phase", "family", "curves", "budget", "bias", "transfer", "validation"])
    if "health" in want:
        health(T, rep=rep)
    sink = DirectoryFigureSink(a.out, a.formats)                          # output
    make_figures(T, sink, rep, only=sorted(want - {"health"}), eps=a.eps, methods=a.methods)
    sys.stdout.write(rep.render_text())
    print("%d figures written to %s (%s)" % (len(sink.saved), a.out, " + ".join(f.upper() for f in a.formats)))
    return 0


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


if __name__ == "__main__":
    sys.exit(main())
