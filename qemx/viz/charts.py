"""Comparison charts: estimates with error bars, the distribution map of repeated experiments,
and per-method detail plots."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import theme as T


def estimates(rows, ideal: float, raw: float | None, N: int, dark: bool = False) -> go.Figure:
    """rows: dicts(title, family, mean, sd). One row per recipe; ±1σ at N shots."""
    ink = T.ink(dark)
    fig = go.Figure()
    ys = [r["title"] for r in rows]
    for r in rows:
        fam = r["family"]
        fig.add_trace(go.Scatter(
            x=[r["mean"]], y=[r["title"]], mode="markers", showlegend=False,
            error_x=dict(type="data", array=[r["sd"]], color=T.FAMILY[fam], thickness=2, width=6),
            marker=dict(size=13, color=T.FAMILY[fam], symbol=T.SYMBOL[fam],
                        line=dict(color=T.FAMILY[fam] if fam == "raw" else ("#0E1117" if dark else "white"), width=1.5)),
            hovertemplate="%s<br>mean %.4f<br>±1σ %.4f at N = %s<extra></extra>" % (r["title"], r["mean"], r["sd"],
                                                                                 f"{N:,}")))
    fig.add_vline(x=ideal, line=dict(color=ink, dash="dash", width=1.4))
    fig.add_annotation(x=ideal, y=1, yref="paper", yanchor="bottom", text="ideal %.4f" % ideal, showarrow=False,
                       font=dict(size=12, color=ink))
    if raw is not None:
        fig.add_vline(x=raw, line=dict(color=T.GREY, dash="dot", width=1.4))
    xs = [r["mean"] for r in rows if np.isfinite(r["mean"])] + [ideal]
    sds = [r["sd"] for r in rows if np.isfinite(r["sd"])] or [0]
    pad = max(max(sds) * 1.5, 0.02)
    fig.update_layout(template=T.tpl(dark), height=90 + 44 * len(rows), margin=dict(l=10, r=20, t=34, b=45),
                      xaxis=dict(title="estimate of ⟨O⟩ (±1σ at %s shots)" % f"{N:,}", range=[min(xs) - pad, max(xs) + pad]),
                      yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=ys, automargin=True))
    return fig


def distribution_map(samples: dict, families: dict, ideal: float, N: int, dark: bool = False) -> go.Figure:
    """Violin + points of repeated experiments: where each method's estimates actually land."""
    ink = T.ink(dark)
    fig = go.Figure()
    for title, s in samples.items():
        s = np.asarray(s, float)
        s = s[np.isfinite(s)]
        fam = families[title]
        fig.add_trace(go.Violin(x=s, y=[title] * len(s), orientation="h", name=title, side="both", width=0.8,
                                line=dict(color=T.FAMILY[fam], width=1.2), fillcolor=_rgba(T.FAMILY[fam], 0.25),
                                points="all", pointpos=0, jitter=0.35, marker=dict(size=3, color=T.FAMILY[fam], opacity=0.6),
                                meanline=dict(visible=True, color=T.FAMILY[fam]), spanmode="hard", showlegend=False,
                                hoveron="violins+points",
                                hovertemplate="%{x:.4f}<extra>" + title + "</extra>"))
    fig.add_vline(x=ideal, line=dict(color=ink, dash="dash", width=1.4))
    fig.update_layout(template=T.tpl(dark), height=90 + 52 * len(samples), margin=dict(l=10, r=20, t=20, b=45),
                      xaxis=dict(title="estimates from repeated experiments at %s shots each" % f"{N:,}"),
                      yaxis=dict(autorange="reversed", automargin=True), violingap=0.25)
    return fig


def _rgba(hex_, a):
    h = hex_.lstrip("#")
    return "rgba(%d,%d,%d,%.2f)" % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def zne_fit(details: dict, ideal: float, dark: bool = False) -> go.Figure:
    from ..methods.zne import curve
    ink = T.ink(dark)
    lams, E, sig = details["lams"], details["E"], details["sigma"]
    grid = np.linspace(0, max(lams) * 1.05, 120)
    fig = go.Figure()
    for rule, dash in (("linear", "dot"), ("richardson", "solid"), ("exponential", "dash")):
        y = curve(rule, lams, E, grid)
        fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", name=rule + (" (used)" if rule == details["rule"] else ""),
                                 line=dict(color=T.FAMILY["zne"], dash=dash, width=2.6 if rule == details["rule"] else 1.4),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lams, y=E, mode="markers", name="measured E(λ)", marker=dict(color=ink, size=9)))
    fig.add_trace(go.Scatter(x=[0], y=[ideal], mode="markers", name="ideal",
                             marker=dict(color=ink, size=14, symbol="circle-open", line=dict(width=2))))
    lo = min(min(E), ideal) - 0.1 * abs(max(E) - min(E) + 1e-3)
    fig.update_layout(template=T.tpl(dark), height=320, xaxis=dict(title="noise scale λ", range=[-0.2, max(lams) * 1.08]),
                      yaxis=dict(title="⟨O⟩"), margin=dict(t=40))
    return fig


def gamma_growth(gpath: np.ndarray, dark: bool = False) -> go.Figure:
    G = np.arange(len(gpath))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=G, y=(gpath + 1) / 2, mode="lines", fill="tozeroy", name="positive mass (Γ+1)/2",
                             line=dict(color=T.BLUE, width=1.5), fillcolor="rgba(21,96,130,0.22)"))
    fig.add_trace(go.Scatter(x=G, y=-(gpath - 1) / 2, mode="lines", fill="tozeroy", name="negative mass (Γ−1)/2",
                             line=dict(color=T.RED, width=1.5), fillcolor="rgba(185,28,28,0.22)"))
    fig.add_trace(go.Scatter(x=G, y=np.ones(len(G)), mode="lines", name="net = 1",
                             line=dict(color=T.ink(dark), dash="dash", width=1.4)))
    fig.update_layout(template=T.tpl(dark), height=300, xaxis=dict(title="corrected gates so far", rangemode="tozero"),
                      yaxis=dict(title="quasi-probability mass"), margin=dict(t=40))
    return fig


def quasi_bars(quasi, dark: bool = False) -> go.Figure:
    labels = [l for _, l in quasi]
    vals = [v for v, _ in quasi]
    fig = go.Figure(go.Bar(x=list(range(len(vals))), y=vals, marker=dict(color=[T.BLUE if v > 0 else T.RED for v in vals]),
                           text=labels, hovertemplate="%{text}: q = %{y:+.5f}<extra></extra>", showlegend=False))
    fig.add_hline(y=0, line=dict(color=T.ink(dark), width=1))
    fig.update_layout(template=T.tpl(dark), height=280, margin=dict(t=20),
                      xaxis=dict(title="operation after the gate", tickmode="array", tickvals=list(range(len(vals))),
                                 ticktext=labels, tickangle=-60 if len(vals) > 8 else 0),
                      yaxis=dict(title="quasi-probability q"))
    return fig


def learning_scatter(details: dict, ideal: float, mean: float, dark: bool = False) -> go.Figure:
    ink = T.ink(dark)
    x, y = details["x_train"], details["y_train"]
    xt = details["x_target"]
    lo = min(np.min(x), np.min(y), xt, ideal) - 0.05
    hi = max(np.max(x), np.max(y), xt, ideal, mean) + 0.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="no mitigation (ideal = noisy)",
                             line=dict(color=T.GREY, dash="dot", width=1.2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="training circuits",
                             marker=dict(color=T.FAMILY["learning"], size=9, line=dict(color="white", width=1)),
                             hovertemplate="noisy %{x:.4f} → ideal %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xt, xt], y=[xt, mean], mode="lines+markers", name="target: noisy → learned",
                             line=dict(color=ink, width=1.6), marker=dict(size=[7, 11], color=ink,
                                                                          symbol=["circle", "arrow-up"], angleref="previous")))
    fig.add_trace(go.Scatter(x=[xt], y=[ideal], mode="markers", name="ideal target",
                             marker=dict(size=15, color=ink, symbol="circle-open", line=dict(width=2))))
    fig.update_layout(template=T.tpl(dark), height=360, margin=dict(t=40),
                      xaxis=dict(title="noisy ⟨O⟩ (feature 1)", range=[lo, hi]),
                      yaxis=dict(title="ideal ⟨O⟩", range=[lo, hi], scaleanchor="x", scaleratio=1))
    return fig


def nil_fit(details: dict, ideal: float, mean: float, dark: bool = False) -> go.Figure:
    """Neighbor-informed learning: the fitted value Σ c_j x_j (+ b) of every training circuit against its ideal value
    (on the diagonal = fitted exactly), and the target's estimate against its ideal value."""
    ink = T.ink(dark)
    fx, y = np.asarray(details["fitted"], float), np.asarray(details["y_train"], float)
    lo = float(min(np.min(fx), np.min(y), ideal, mean)) - 0.05
    hi = float(max(np.max(fx), np.max(y), ideal, mean)) + 0.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="fitted = ideal",
                             line=dict(color=T.GREY, dash="dot", width=1.2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=y, mode="markers", name="training circuits",
                             marker=dict(color=T.FAMILY["learning"], size=9, symbol="triangle-up",
                                         line=dict(color="white", width=1)),
                             hovertemplate="fitted %{x:.4f} · ideal %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[mean], y=[ideal], mode="markers", name="target: estimate · ideal",
                             marker=dict(size=15, color=ink, symbol="circle-open", line=dict(width=2)),
                             hovertemplate="estimate %{x:.4f} · ideal %{y:.4f}<extra></extra>"))
    fig.update_layout(template=T.tpl(dark), height=360, margin=dict(t=40),
                      xaxis=dict(title="fitted Σ c<sub>j</sub> x<sub>j</sub>", range=[lo, hi]),
                      yaxis=dict(title="ideal ⟨O⟩", range=[lo, hi], scaleanchor="x", scaleratio=1))
    return fig


def nil_coefficients(details: dict, dark: bool = False) -> go.Figure:
    """The learned coefficient of every neighbor circuit (blue positive, red negative; zero: not run on the target).
    γ = Σ|c_j| is the method's sampling overhead."""
    names = list(details["names"])
    c = np.asarray(details["coef"], float)
    col = [T.BLUE if v > 0 else (T.RED if v < 0 else T.GREY) for v in c]
    fig = go.Figure(go.Bar(x=names, y=c, marker=dict(color=col, line=dict(width=0)),
                           hovertemplate="%{x}: c = %{y:+.4f}<extra></extra>", showlegend=False))
    fig.add_hline(y=0, line=dict(color=T.ink(dark), width=1))
    fig.update_layout(template=T.tpl(dark), height=320, margin=dict(t=40, b=90),
                      title=dict(text="coefficients · γ = Σ|c<sub>j</sub>| = %.3f" % float(np.abs(c).sum()),
                                 font=dict(size=14)),
                      xaxis=dict(title="neighbor circuit", tickangle=-45), yaxis=dict(title="c<sub>j</sub>"))
    return fig


def spectrum(eigs: np.ndarray, M: int, dark: bool = False) -> go.Figure:
    w = np.clip(np.asarray(eigs, float), 0, None)
    k = np.arange(1, min(len(w), 12) + 1)
    fig = go.Figure()
    for m, dash, sym in ((1, "dot", "circle"), (M, "solid", "square")):
        v = w[:len(k)] ** m / np.sum(w ** m)
        fig.add_trace(go.Scatter(x=k, y=np.clip(v, 1e-12, None), mode="lines+markers",
                                 name="ρ" if m == 1 else "ρ<sup>%d</sup>/Tr ρ<sup>%d</sup>" % (m, m),
                                 line=dict(color=T.GREY if m == 1 else T.FAMILY["vd"], dash=dash), marker=dict(symbol=sym, size=8)))
    fig.update_layout(template=T.tpl(dark), height=300, margin=dict(t=40),
                      xaxis=dict(title="eigenvalue rank", dtick=1), yaxis=dict(type="log", title="weight", exponentformat="power"))
    return fig


# ------------------------------------------------------------------ analysis views (see qemx.insight)
SERIES_STYLE = {"ideal": ("ideal", None), "noisy (unmitigated)": ("raw", None), "readout-corrected": ("readout", None),
                "ZNE-extrapolated": ("zne", None)}


def _bitstrings(n: int, idx) -> list:
    return [format(int(i), "0%db" % n) for i in idx]


def output_histogram(series: dict, n: int, support=None, top: int = 24, dark: bool = False,
                     bits: list | None = None) -> go.Figure:
    """Grouped bars of the output distribution over bit strings (q_{n-1} … q_0), ideal as an outline, noisy and
    mitigated filled. Only the ``top`` most probable outcomes (over all series) are drawn; the rest are summed into
    'other'. ``support``: outcomes the observable counts (values 1), marked by a pale band."""
    ink = T.ink(dark)
    names = [k for k in ("ideal", "noisy (unmitigated)", "readout-corrected", "ZNE-extrapolated") if k in series]
    M = np.stack([np.asarray(series[k], float) for k in names])
    weight = np.max(np.abs(M), axis=0)
    idx = np.argsort(-weight, kind="stable")[:top]
    idx = np.sort(idx[weight[idx] > 1e-9]) if np.any(weight[idx] > 1e-9) else np.sort(idx[:1])
    rest = np.setdiff1d(np.arange(M.shape[1]), idx)
    labels = _bitstrings(n, idx)
    show_other = rest.size and np.max(np.abs(M[:, rest].sum(axis=1))) > 1e-6
    if show_other:
        labels = labels + ["other (%d)" % rest.size]
    fig = go.Figure()
    sup = set(int(i) for i in (support if support is not None else []))
    for j, i in enumerate(idx):
        if int(i) in sup:
            fig.add_shape(type="rect", xref="x", yref="paper", x0=j - 0.5, x1=j + 0.5, y0=0, y1=1, layer="below",
                          fillcolor="rgba(161,98,7,0.10)" if not dark else "rgba(234,179,8,0.14)", line=dict(width=0))
    for k, M_row in zip(names, M):
        y = list(M_row[idx]) + ([float(M_row[rest].sum())] if show_other else [])
        fam = SERIES_STYLE[k][0]
        if fam == "ideal":
            marker = dict(color="rgba(0,0,0,0)", line=dict(color=ink, width=1.6), cornerradius=3)
        else:
            marker = dict(color=T.FAMILY[fam], line=dict(color="#0E1117" if dark else "white", width=1),
                          cornerradius=3)
        fig.add_trace(go.Bar(x=labels, y=y, name=k, marker=marker,
                             hovertemplate="%{x}<br>" + k + " %{y:.4f}<extra></extra>"))
    if sup:
        fig.add_trace(go.Bar(x=[None], y=[None], name="counted by ⟨O⟩", marker=dict(
            color="rgba(161,98,7,0.25)"), hoverinfo="skip"))
    fig.update_layout(template=T.tpl(dark), height=400, barmode="group", bargap=0.25, bargroupgap=0.06,
                      margin=dict(l=60, r=20, t=40, b=70), hovermode="x unified",
                      xaxis=dict(title="outcome (%s)" % (" ".join("q%d" % q for q in reversed(bits)) if bits and
                                                          len(bits) <= 8 else "q%d … q0" % (n - 1)),
                                 type="category", tickangle=-45 if len(labels) > 12
                                 else 0, tickfont=dict(family="IBM Plex Mono, Menlo, monospace", size=11)),
                      yaxis=dict(title="probability", zeroline=True, zerolinecolor=T.GREY),
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"))
    return fig


def syndrome_histogram(dists: dict, check_names: list, dark: bool = False) -> go.Figure:
    """P(syndrome) for the ideal and the noisy run: '0…0' means no check fired."""
    ink = T.ink(dark)
    k = len(check_names)
    labels = [format(s, "0%db" % k) for s in range(2 ** k)]
    fig = go.Figure()
    for name, p in dists.items():
        fam = "ideal" if name == "ideal" else "raw"
        marker = (dict(color="rgba(0,0,0,0)", line=dict(color=ink, width=1.6), cornerradius=3) if fam == "ideal"
                  else dict(color=T.FAMILY["symmetry"], cornerradius=3, line=dict(color="#0E1117" if dark else "white",
                                                                                    width=1)))
        hover = ["%s<br>%s" % (lab, ", ".join(nm for nm, b in zip(check_names, lab) if b == "1") or "no check fired")
                 for lab in labels]
        fig.add_trace(go.Bar(x=labels, y=p, name=name, marker=marker, customdata=hover,
                             hovertemplate="%{customdata}<br>" + name + " %{y:.4f}<extra></extra>"))
    fig.add_annotation(x=labels[0], y=1, yref="paper", yanchor="bottom", showarrow=False, text="no error detected",
                       font=dict(size=11, color=T.MUTED))
    fig.update_layout(template=T.tpl(dark), height=330, barmode="group", bargap=0.3, margin=dict(l=60, r=20, t=40, b=60),
                      xaxis=dict(title="syndrome (%s)" % " · ".join(check_names), type="category",
                                 tickfont=dict(family="IBM Plex Mono, Menlo, monospace")),
                      yaxis=dict(title="probability", range=[0, 1.05]),
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.1))
    return fig


def logical_error_bars(rows: list, dark: bool = False) -> go.Figure:
    """Effective logical error p_L = (1 − ⟨Z_L⟩/⟨Z_L⟩_ideal)/2 per method (lower is better)."""
    rows = [r for r in rows if np.isfinite(r["pL"])]
    fig = go.Figure(go.Bar(
        x=[r["pL"] for r in rows], y=[r["method"] for r in rows], orientation="h",
        marker=dict(color=[T.FAMILY.get(r["family"], T.GREY) for r in rows], cornerradius=4),
        text=["%.2e" % r["pL"] if abs(r["pL"]) < 1e-2 else "%.3f" % r["pL"] for r in rows], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>p_L = %{x:.4g}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=T.ink(dark), width=1))
    xmax = max([abs(r["pL"]) for r in rows] + [1e-3])
    xmin = min([r["pL"] for r in rows] + [0.0])
    fig.update_layout(template=T.tpl(dark), height=100 + 40 * len(rows), margin=dict(l=10, r=40, t=30, b=45),
                      xaxis=dict(title="effective logical error p_L (0 is ideal; < 0 overshoots)",
                                 range=[xmin - 0.35 * xmax if xmin < 0 else 0, xmax * 1.35]),
                      yaxis=dict(autorange="reversed", automargin=True), showlegend=False, bargap=0.35)
    return fig


def readout_bars(mats: list, dark: bool = False) -> go.Figure:
    """Calibrated readout error of every classical bit: P(1|0) and P(0|1)."""
    c = list(range(len(mats)))
    p10 = [float(A[1][0]) for A in mats]
    p01 = [float(A[0][1]) for A in mats]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["c%d" % i for i in c], y=p10, name="P(read 1 | prepared 0)",
                         marker=dict(color=T.FAMILY["readout"], cornerradius=3)))
    fig.add_trace(go.Bar(x=["c%d" % i for i in c], y=p01, name="P(read 0 | prepared 1)",
                         marker=dict(color="rgba(91,107,0,0.45)", cornerradius=3)))
    fig.update_layout(template=T.tpl(dark), height=300, barmode="group", margin=dict(l=60, r=20, t=40, b=45),
                      yaxis=dict(title="assignment error", tickformat=".1%"),
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.12))
    return fig
