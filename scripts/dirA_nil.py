#!/usr/bin/env python3
r"""Neighbor-informed learning in the Direction A data (direction_A_nil_v1): how the regression rules compare, and
how NIL compares with the other methods at the same total budget.

    python scripts/dirA_nil.py data/dirA_nil_v1 --out figs_nil            # report, CSV tables and figures
    python scripts/dirA_nil.py data/dirA_nil_v1 --out figs_nil --no-figures

From a notebook (no file is read or written by the computations)::

    import dirA_nil as N, dirA_figures as F, dirA_io as IO
    T = F.load("data/dirA_nil_v1")
    rules = N.rule_table(T)                     # one row per (cell, NIL rule): |b|, v, u, w, γ, ratios to NIL-G2
    h2h = N.head_to_head(T, "NIL-G2", ["FR-L", "FR-R", "CDR", "ZNE-R"])

Tables
------
rules          per cell and NIL rule (estimator ``nil``): bias, target variance v, training variance u, calibration
               coefficient w, γ = ‖c‖₁, nonzero coefficients, solver certificate; ratios to the reference rule
               (Chen et al.'s ℓ1 ≤ 2, NIL-G2) in the same cell — same neighbor and training circuits (common random
               numbers), so the ratios isolate the regression rule
rules_summary  medians of those ratios per rule and stage; the share of fits without an optimality certificate
head_to_head   per stage, family, policy and budget: the share of cells where NIL-G2 has the lower RMSE than each
               reference method, and the median RMSE ratio
bstar          B*(ε) of NIL-G2 against each reference per cell (exact integer ledger, qemx), and their ratio

Figures: the rule comparison (bias and variance ratios) and the head-to-head shares against the budget. Phase maps
with NIL in the palette come from ``dirA_figures.py --methods nil``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dirA_figures as F  # noqa: E402
from dirA_io import DirectoryFigureSink, DirectoryReportSink, Report  # noqa: E402

REFERENCE_RULE = "NIL-G2"
REFERENCES = ("FR-L", "FR-R", "CDR", "ZNE-R", "PEC")


# ------------------------------------------------------------------ computations (tables in, tables out)
def rule_table(T, reference: str = REFERENCE_RULE) -> pd.DataFrame:
    """One row per (cell, NIL rule) with the rule's primitives and their ratios to ``reference`` in the same cell."""
    m = T.methods
    m = m[(m["estimator"] == "nil") & (m["status"] == "ok")].copy()
    if m.empty:
        return pd.DataFrame()
    keep = ["cell_id", "stage", "family", "n", "label", "scale", "lam_lc", "method", "b", "v", "u", "w", "c",
            "x_model", "x_l1", "x_nonzero", "x_solver_ok", "x_target_split"]
    m = m[[k for k in keep if k in m.columns]].copy()
    m["abs_b"] = m["b"].astype(float).abs()
    ref = m[m["method"] == reference].set_index("cell_id")[["abs_b", "v", "u", "w"]]
    m = m.join(ref, on="cell_id", rsuffix="_ref")
    with np.errstate(divide="ignore", invalid="ignore"):
        m["bias_ratio"] = m["abs_b"] / m["abs_b_ref"].where(m["abs_b_ref"] > 0)
        m["v_ratio"] = m["v"].astype(float) / m["v_ref"].astype(float).where(m["v_ref"] > 0)
        m["u_ratio"] = m["u"].astype(float) / m["u_ref"].astype(float).where(m["u_ref"] > 0)
    return m


def rules_summary(R: pd.DataFrame) -> pd.DataFrame:
    """Median ratios to the reference rule per stage and rule, and the share of uncertified fits."""
    if R.empty:
        return pd.DataFrame()
    agg = dict(cells=("cell_id", "nunique"), median_bias_ratio=("bias_ratio", "median"),
               median_v_ratio=("v_ratio", "median"), median_u_ratio=("u_ratio", "median"),
               median_gamma=("x_l1", "median"), median_nonzero=("x_nonzero", "median"))
    agg = {k: v for k, v in agg.items() if v[0] in R.columns}
    S = R.groupby(["stage", "method"]).agg(**agg)
    if "x_solver_ok" in R.columns:
        ok = R["x_solver_ok"].map(lambda x: str(x).lower() in ("true", "1", "1.0"))
        S["uncertified_pct"] = (~ok).groupby([R["stage"], R["method"]]).mean() * 100
    return S.reset_index()


def head_to_head(T, a: str, refs, policy: str = "fixed") -> pd.DataFrame:
    """Per stage, family and budget: share of cells where ``a`` has the lower RMSE than each reference (ties count
    half), and the median RMSE ratio a/ref (infeasible = inf)."""
    if T.ledger is None or T.ledger.empty:
        return pd.DataFrame()
    L = T.ledger[(T.ledger["policy"] == policy) & T.ledger["method"].isin([a] + list(refs))].copy()
    L["r"] = pd.to_numeric(L["rmse"], errors="coerce").fillna(np.inf)
    P = L.pivot_table(index=["cell_id", "B"], columns="method", values="r", aggfunc="first")
    if a not in P.columns:
        return pd.DataFrame()
    cells = T.cells[["stage", "family"]]
    out = []
    for ref in refs:
        if ref not in P.columns:
            continue
        x = P[[a, ref]].dropna().reset_index().join(cells, on="cell_id")
        both_inf = ~np.isfinite(x[a]) & ~np.isfinite(x[ref])
        x = x[~both_inf]
        win = np.where(x[a] < x[ref], 1.0, np.where(x[a] == x[ref], 0.5, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = x[a] / x[ref]
        x = x.assign(win=win, ratio=ratio)
        g = x.groupby(["stage", "family", "B"]).agg(cells=("cell_id", "nunique"), share_a_wins=("win", "mean"),
                                                   median_ratio=("ratio", "median")).reset_index()
        g.insert(0, "reference", ref)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def bstar_table(T, a: str, refs, eps: float = 0.01, policy: str = "fixed") -> pd.DataFrame:
    """B*(ε) of ``a`` and of each reference per cell (qemx's exact ledger), with their ratio."""
    ids = list(T.cells.index)
    D = F.budget_to_target(T, ids, eps, policy)
    if D.empty:
        return D
    P = D.pivot_table(index="cell_id", columns="method", values="Bstar", aggfunc="first")
    rows = []
    for ref in refs:
        if a not in P.columns or ref not in P.columns:
            continue
        x = P[[a, ref]].dropna()
        for cid, r in x.iterrows():
            ba, br = float(r[a]), float(r[ref])
            rows.append(dict(cell_id=cid, reference=ref, Bstar_a=ba, Bstar_ref=br,
                             ratio=(ba / br) if math.isfinite(ba) and math.isfinite(br) and br > 0 else np.nan,
                             a_reaches=math.isfinite(ba), ref_reaches=math.isfinite(br)))
    out = pd.DataFrame(rows)
    return out.join(T.cells[["stage", "family", "label", "scale", "lam_lc"]], on="cell_id") if len(out) else out


def analyse(T, eps: float = 0.01, reference: str = REFERENCE_RULE, refs=REFERENCES) -> Report:
    """Every table of this module, as a report (text and tables)."""
    rep = Report()
    rep.section("RULES · NIL regression rules against %s in the same cell (same neighbor and training circuits)"
                % reference)
    R = rule_table(T, reference)
    rep.table("rules", R)
    S = rules_summary(R)
    if S.empty:
        rep.verdict("nil-rules", None, "no NIL rows (estimator nil) in this store")
    else:
        rep.text("median ratios to %s (below 1: smaller than the reference rule's):" % reference)
        rep.table("rules_summary", S, show=True)
        if "uncertified_pct" in S.columns:
            worst = float(S["uncertified_pct"].max())
            rep.verdict("nil-certificates", worst == 0.0, "fits without an optimality certificate: at most %.2f %% "
                        "of the cells of any rule" % worst)
    rep.section("HEAD TO HEAD · %s against %s at equal total budget (fixed policy)" % (reference, ", ".join(refs)))
    H = head_to_head(T, reference, refs)
    rep.table("head_to_head", H)
    if not H.empty:
        top = H[H["B"] == H["B"].max()]
        rep.text("at the largest budget (share of cells where %s has the lower RMSE):" % reference)
        rep.frame(top.pivot_table(index=["stage", "family"], columns="reference", values="share_a_wins").round(2)
                  .reset_index())
    rep.section("B*(ε = %g) · budget to reach the target, %s against each reference" % (eps, reference))
    B = bstar_table(T, reference, refs, eps)
    rep.table("bstar", B)
    if not B.empty:
        g = B.groupby(["stage", "reference"]).agg(cells=("cell_id", "size"), a_reaches=("a_reaches", "mean"),
                                                 ref_reaches=("ref_reaches", "mean"),
                                                 median_ratio=("ratio", "median")).reset_index()
        rep.frame(g.round(3))
    return rep


# ------------------------------------------------------------------ figures (tables in, figures out)
def fig_rules(R: pd.DataFrame, reference: str = REFERENCE_RULE):
    """|b| and v of every NIL rule relative to the reference rule, one point per cell (log scale)."""
    import matplotlib.pyplot as plt
    rules = [k for k in sorted(R["method"].unique()) if k != reference]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 0.38 * max(len(rules), 3) + 1.2), sharey=True)
    for ax, col, name in zip(axs, ("bias_ratio", "v_ratio"), ("|b| / |b$_{\\rm ref}$|", "v / v$_{\\rm ref}$")):
        for i, k in enumerate(rules):
            x = R.loc[R["method"] == k, col].astype(float)
            x = x[np.isfinite(x) & (x > 0)]
            if len(x):
                jitter = (np.random.default_rng(i).random(len(x)) - 0.5) * 0.35
                ax.scatter(x, i + jitter, s=6, color=F.INK2, alpha=0.45, linewidths=0)
                ax.plot([x.median()] * 2, [i - 0.3, i + 0.3], color=F.INK, lw=1.6)
        ax.axvline(1.0, color=F.GRID, lw=1.0, zorder=0)
        ax.set_xscale("log")
        _plain_log(ax)
        ax.set_xlabel(name)
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(rules)
        ax.grid(True, axis="x", color=F.GRID, lw=0.5)
    axs[0].set_title("NIL rules against %s, same cell (bar: median)" % reference, loc="left", fontsize=9)
    fig.tight_layout()
    return fig


def _plain_log(ax):
    """Plain tick labels (0.7, 1, 1.05) on a log x axis; minor ticks are labelled too when the axis spans less than a
    decade (otherwise a narrow axis would show no label at all)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    lo, hi = ax.get_xlim()
    fmt = FuncFormatter(lambda v, _: "%g" % float("%.3g" % v))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(fmt if hi / max(lo, 1e-300) < 10 else NullFormatter())
    ax.tick_params(axis="x", which="minor", labelsize=7)


def fig_head_to_head(H: pd.DataFrame, reference: str = REFERENCE_RULE):
    """Share of cells where the reference rule beats each method, against the budget (one panel per stage)."""
    import matplotlib.pyplot as plt
    stages = sorted(H["stage"].unique())
    fig, axs = plt.subplots(1, len(stages), figsize=(3.4 * len(stages), 2.8), sharey=True, squeeze=False)
    for ax, st in zip(axs[0], stages):
        d = H[H["stage"] == st].assign(wsum=lambda x: x["share_a_wins"] * x["cells"]) \
            .groupby(["reference", "B"]).agg(wsum=("wsum", "sum"), cells=("cells", "sum")).reset_index()
        d["share"] = d["wsum"] / d["cells"]                 # cell-weighted over the families
        for ref in d["reference"].unique():
            x = d[d["reference"] == ref].sort_values("B")
            col, mk, ls, lab = F.style(ref)
            ax.plot(x["B"], x["share"], color=col, marker=mk, ms=3, lw=1.3, ls=ls, label=lab)
        ax.axhline(0.5, color=F.GRID, lw=1.0, zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("total shots B")
        ax.set_title(st, loc="left", fontsize=9)
    axs[0][0].set_ylabel("share of cells where %s wins" % reference)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


# ------------------------------------------------------------------ driver: input → computation → output
def main(argv=None):
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="the run directory (contains tables/)")
    ap.add_argument("--out", default="figs_nil")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    T = F.load(a.data)                                                     # input
    rep = analyse(T, eps=a.eps)                                            # computation
    DirectoryReportSink(os.path.join(a.out, "tables")).write(rep)         # output
    if not a.no_figures:
        F.setup_matplotlib()
        sink = DirectoryFigureSink(a.out, a.formats)
        R, H = rep.tables.get("rules"), rep.tables.get("head_to_head")
        if R is not None and not R.empty:
            sink.save(fig_rules(R), "nil_rules")
        if H is not None and not H.empty:
            sink.save(fig_head_to_head(H), "nil_head_to_head")
    sys.stdout.write(rep.render_text())
    print("tables and report written to %s" % os.path.join(a.out, "tables"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
