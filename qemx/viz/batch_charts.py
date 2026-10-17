"""Charts for batch runs: a scenario × method heatmap, method wins, and error against the noise scale."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T
from .analysis_charts import style_map
from ..batch import rmse_at, scenario_key


def _values(rows, N, metric):
    if metric == "RMSE":
        return rmse_at(rows, N)
    if metric == "|bias|":
        return np.abs(np.array([r.get("bias", math.nan) for r in rows], float))
    if metric == "γ":
        return np.array([r.get("γ", math.nan) for r in rows], float)
    return np.array([r.get("estimate", math.nan) for r in rows], float)


def _wrap_method(m: str) -> str:
    """Long method titles over two or three short lines: 'ZNE Richardson<br>(global 1·3·5)<br>+ twirl + TREX'."""
    return m.replace(" (", "<br>(").replace(") + ", ")<br>+ ")


def _wrap_scenario(s: str) -> str:
    alg, _, rest = s.partition(" · ")
    return "<b>%s</b><br>%s" % (alg, rest) if rest else s


def _order(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def heatmap(rows, N: int, metric: str = "RMSE", dark: bool = False) -> go.Figure:
    """Scenarios (rows) × methods (columns); color = log10 of the metric (more contrast with the page = larger
    error); the best method of each scenario is starred. Failed cells stay empty."""
    scen = _order(scenario_key(r) for r in rows)
    meth = _order(r["method"] for r in rows)
    vals = _values(rows, N, metric)
    Z = np.full((len(scen), len(meth)), np.nan)
    for r, v in zip(rows, vals):
        Z[scen.index(scenario_key(r)), meth.index(r["method"])] = v
    logged = metric in ("RMSE", "|bias|")
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.log10(np.where(Z > 0, Z, np.nan)) if logged else Z
    best = {}
    if logged:                                          # the best method of each scenario: a star in its cell
        for i in range(len(scen)):
            if np.isfinite(Z[i]).any():
                best[i] = int(np.nanargmin(Z[i]))
    text = [[("" if not np.isfinite(z) else ("★ " if best.get(i) == j else "") +
              ("%.2g" % z if logged else "%.3f" % z)) for j, z in enumerate(row)] for i, row in enumerate(Z)]
    fig = go.Figure(go.Heatmap(
        z=C, x=[_wrap_method(m) for m in meth], y=[_wrap_scenario(s) for s in scen], text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale=([[0, "#0E2A38"], [0.5, "#3F7FA0"], [1, "#D6E8F2"]] if dark else
                    [[0, "#F3F7FA"], [0.5, "#6FA3C0"], [1, "#0B3D56"]]), showscale=True, xgap=2, ygap=2,
        colorbar=dict(title=dict(text=("log₁₀ " if logged else "") + metric, side="top"), thickness=10, len=0.8),
        customdata=Z, hovertemplate="%{y}<br>%{x}<br>" + metric + " %{customdata:.4g}<extra></extra>"))
    fig.update_traces(hoverlabel=dict(namelength=-1))
    h = int(min(1800, 170 + 40 * len(scen)))
    fig.update_layout(template=T.tpl(dark), height=h, margin=dict(l=10, r=20, t=30, b=10),
                      xaxis=dict(side="top", tickangle=0, automargin=True, tickfont=dict(size=11)),
                      yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(size=11)),
                      showlegend=False)
    return fig


def wins(rows, N: int, metric: str = "RMSE", dark: bool = False) -> go.Figure:
    """How many scenarios each method wins (lowest metric)."""
    from ..batch import winners
    w = winners(rows, N, metric)
    meth = _order(r["method"] for r in rows)
    fam = {r["method"]: r["family"] for r in rows}
    counts = {m: sum(1 for x in w if x["best method"] == m) for m in meth}
    sty = style_map([(m, fam[m]) for m in meth])
    order = sorted(meth, key=lambda m: -counts[m])
    fig = go.Figure(go.Bar(x=[counts[m] for m in order], y=order, orientation="h",
                           marker=dict(color=[sty[m][0] for m in order], cornerradius=4),
                           text=[str(counts[m]) for m in order], textposition="outside",
                           hovertemplate="%{y}: best in %{x} scenario(s)<extra></extra>"))
    top = max(counts.values()) if counts else 1
    fig.update_layout(template=T.tpl(dark), height=int(120 + 34 * len(order)), margin=dict(l=10, r=30, t=30, b=40),
                      xaxis=dict(title="scenarios won (lowest %s)" % metric, dtick=1 if top <= 8 else None,
                                 range=[0, max(top, 1) * 1.18], tickformat="d"),
                      yaxis=dict(autorange="reversed", automargin=True), bargap=0.35, showlegend=False)
    return fig


def log_ticks(values, per_decade=(1, 2, 5), max_ticks: int = 8) -> list:
    """Tick values 1-2-5 × 10^k spanning the data, labeled in full (0.005, 0.01, 0.02 …) instead of Plotly's bare
    '2' and '5' minor labels. Falls back to powers of ten when the range is wide."""
    v = [x for x in values if x > 0]
    if not v:
        return []
    lo, hi = math.floor(math.log10(min(v))), math.ceil(math.log10(max(v)))
    for steps in (per_decade, (1, 3), (1,)):
        ticks = [m * 10.0 ** k for k in range(lo, hi + 1) for m in steps]
        ticks = [t for t in ticks if min(v) / 1.6 <= t <= max(v) * 1.6]
        if len(ticks) < 2:
            return []                                   # a very narrow range: Plotly's own ticks read better
        if len(ticks) <= max_ticks:
            return [float("%.3g" % t) for t in ticks]
    return [10.0 ** k for k in range(lo, hi + 1)]


def versus_scale(rows, N: int, noise: str, metric: str = "RMSE", dark: bool = False) -> go.Figure:
    """Small multiples (one per algorithm × simulator): the metric against the noise scale λ for each method."""
    sel = [r for r in rows if r["noise"] == noise]
    panels = _order((r["algorithm"], r["simulator"]) for r in sel)
    meth = _order(r["method"] for r in sel)
    fam = {r["method"]: r["family"] for r in sel}
    sty = style_map([(m, fam[m]) for m in meth])
    cols = min(2, max(1, len(panels)))
    nrows = int(math.ceil(len(panels) / cols)) or 1
    one_sim = len({p[1] for p in panels}) == 1
    fig = make_subplots(rows=nrows, cols=cols, subplot_titles=[p[0] if one_sim else "%s · %s" % p for p in panels],
                        horizontal_spacing=0.07, vertical_spacing=0.16 if nrows > 1 else 0.1)
    vals = _values(sel, N, metric)
    for k, p in enumerate(panels):
        rr, cc = k // cols + 1, k % cols + 1
        for m in meth:
            pts = sorted([(r["λ"], v) for r, v in zip(sel, vals) if (r["algorithm"], r["simulator"]) == p and
                          r["method"] == m and np.isfinite(v)])
            if not pts:
                continue
            col, dash, sym = sty[m]
            fig.add_trace(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="lines+markers", name=m,
                                     legendgroup=m, showlegend=k == 0, line=dict(color=col, dash=dash, width=2),
                                     marker=dict(symbol=sym, size=8, color=col),
                                     hovertemplate=m + "<br>λ %{x}<br>" + metric + " %{y:.4g}<extra></extra>"),
                          row=rr, col=cc)
    fig.update_xaxes(title_text="noise scale λ")
    if metric in ("RMSE", "|bias|"):
        tv = log_ticks([v for v in vals if np.isfinite(v) and v > 0])
        fig.update_yaxes(type="log", tickvals=tv or None, ticktext=["%g" % t for t in tv] or None)
    fig.update_yaxes(title_text=metric if metric != "RMSE" else "RMSE at N = %s" % f"{N:,}", col=1)
    fig.update_layout(template=T.tpl(dark), height=int(150 + 290 * nrows), margin=dict(l=60, r=20, t=40, b=60),
                      legend=dict(orientation="h", x=0, y=-0.06 - 0.1 / nrows, yanchor="top"))
    return fig


def versus_sweep(rows, N: int, axis: str, metric: str = "RMSE", dark: bool = False, max_panels: int = 12):
    """Small multiples: the metric against one swept quantity for each method. A panel is everything else held
    fixed — algorithm, the other swept values, noise model and scale, simulator. Returns (figure, panels shown,
    panels in all)."""
    sel = [r for r in rows if axis in r]
    others = lambda r: " · ".join(p for p in str(r.get("sweep", "")).split(" · ")
                                  if not (p.startswith(axis + "=") or (axis == "input" and "=" not in p)))
    key = lambda r: " · ".join(x for x in (r.get("algorithm_base", r["algorithm"]), others(r),
                                            r["noise"] if r["noise"] == "device calibration" else
                                            "%s ×%g" % (r["noise"], r["λ"]), r["simulator"]) if x)
    panels_all = _order(key(r) for r in sel)
    panels = panels_all[:max_panels]
    xs = _order(r[axis] for r in sel)
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs)
    meth = _order(r["method"] for r in sel)
    fam = {r["method"]: r["family"] for r in sel}
    sty = style_map([(m, fam[m]) for m in meth])
    cols = min(2, max(1, len(panels)))
    nrows = int(math.ceil(len(panels) / cols)) or 1
    fig = make_subplots(rows=nrows, cols=cols, subplot_titles=[_wrap_title(p) for p in panels],
                        horizontal_spacing=0.08, vertical_spacing=min(0.16, 0.5 / max(nrows, 1)) if nrows > 1 else 0.1)
    vals = _values(sel, N, metric)
    for k, p in enumerate(panels):
        rr, cc = k // cols + 1, k % cols + 1
        for m in meth:
            pts = [(r[axis], v) for r, v in zip(sel, vals) if key(r) == p and r["method"] == m and np.isfinite(v)]
            if numeric:
                pts.sort()
            if not pts:
                continue
            col, dash, sym = sty[m]
            fig.add_trace(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="lines+markers", name=m,
                                     legendgroup=m, showlegend=k == 0, line=dict(color=col, dash=dash, width=2),
                                     marker=dict(symbol=sym, size=8, color=col),
                                     hovertemplate=m + "<br>" + axis + " %{x}<br>" + metric +
                                     " %{y:.4g}<extra></extra>"), row=rr, col=cc)
    fig.update_xaxes(title_text=axis, type=None if numeric else "category")
    if metric in ("RMSE", "|bias|"):
        tv = log_ticks([v for v in vals if np.isfinite(v) and v > 0])
        fig.update_yaxes(type="log", tickvals=tv or None, ticktext=["%g" % t for t in tv] or None)
    fig.update_yaxes(title_text=metric if metric != "RMSE" else "RMSE at N = %s" % f"{N:,}", col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(template=T.tpl(dark), height=int(170 + 290 * nrows), margin=dict(l=60, r=20, t=50, b=60),
                      legend=dict(orientation="h", x=0, y=-0.06 - 0.1 / nrows, yanchor="top"))
    return fig, len(panels), len(panels_all)


def _wrap_title(t: str, width: int = 48) -> str:
    """Panel titles over at most two lines, broken at ' · '."""
    if len(t) <= width:
        return t
    parts, lines, cur = t.split(" · "), [], ""
    for p in parts:
        if cur and len(cur) + 3 + len(p) > width:
            lines.append(cur)
            cur = p
        else:
            cur = p if not cur else cur + " · " + p
    lines.append(cur)
    return "<br>".join(lines[:2]) + (" …" if len(lines) > 2 else "")
