"""Charts for the analysis pages: cost vs precision, the PEC wall, the bias–variance bullseye, model mismatch and
head-to-head sweeps. Color follows the method *family* (fixed palette in ``theme``); several methods of one
family are told apart by marker and dash, never by a generated hue."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T

DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
SYMBOLS2 = ["circle", "square", "diamond", "triangle-up", "x", "star", "hexagon", "cross"]


def _rgba(hex_, a):
    h = hex_.lstrip("#")
    return "rgba(%d,%d,%d,%.2f)" % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def style_map(methods_families: list) -> dict:
    """method -> (color, dash, symbol): family color, then dash/symbol by order within the family."""
    seen, out = {}, {}
    for m, fam in methods_families:
        k = seen.get(fam, 0)
        seen[fam] = k + 1
        out[m] = (T.FAMILY.get(fam, T.GREY), DASHES[k % len(DASHES)], SYMBOLS2[k % len(SYMBOLS2)])
    return out


def _base(fig, dark, height, **kw):
    fig.update_layout(template=T.tpl(dark), height=height, margin=dict(l=60, r=20, t=40, b=50), **kw)
    return fig


# ------------------------------------------------------------------ cost vs precision
def rmse_vs_shots(rows: list, shots, eps: float | None, n_budget: float | None, dark=False, t_shot_us: float = 250.0,
                  crossings: list | None = None) -> go.Figure:
    """RMSE(N) = sqrt(b² + v/N) per method on log–log axes; the floor |b| is where more shots stop helping."""
    from ..analysis import curve_sources, rmse_row, target_grid
    sty = style_map([(r["method"], r["family"]) for r in rows])
    fig = go.Figure()
    shots = np.asarray(shots, float)
    for r in rows:
        col, dash, sym = sty[r["method"]]
        y = rmse_row(r, shots)
        nl = target_grid(r) is not None
        # the bias floor: the large-budget limit (for target-nonlinear rows the reference bias, not this budget's)
        floor = abs(float(r.get("reference bias", r["bias"]))) if nl else abs(r["bias"])
        src = curve_sources(r, shots) if nl else ["analytic"] * len(shots)
        fig.add_trace(go.Scatter(x=shots, y=y, mode="lines", name=r["method"], line=dict(color=col, width=2.4, dash=dash),
                                 customdata=np.stack([shots * t_shot_us * 1e-6, np.full_like(shots, floor)], 1),
                                 text=src,
                                 hovertemplate="%s<br>N = %%{x:,.0f} shots (%%{customdata[0]:.3g} s QPU)"
                                               "<br>RMSE = %%{y:.4g} (%%{text})<br>bias floor %%{customdata[1]:.3g}"
                                               "<extra></extra>" % r["method"]))
        if floor > 0 and np.isfinite(floor):
            fig.add_trace(go.Scatter(x=[shots[0], shots[-1]], y=[floor] * 2, mode="lines", showlegend=False,
                                     line=dict(color=col, width=1, dash="dot"), hoverinfo="skip", opacity=0.6))
    if eps:
        fig.add_hline(y=eps, line=dict(color=T.ink(dark), width=1.4, dash="dash"))
        fig.add_annotation(x=1, xref="paper", y=math.log10(eps), yref="y", text="target ε = %.3g" % eps,
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12))
    if n_budget:
        fig.add_vline(x=n_budget, line=dict(color=T.YELLOW, width=1.6, dash="dash"))
        fig.add_annotation(x=math.log10(n_budget), xref="x", y=1, yref="paper", text="QPU budget", showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=12, color=T.YELLOW))
    for c in crossings or []:
        fig.add_trace(go.Scatter(x=[c["N"]], y=[c["rmse"]], mode="markers", showlegend=False,
                                 marker=dict(size=11, symbol="circle-open", color=T.ink(dark), line=dict(width=2)),
                                 hovertemplate="%s overtakes unmitigated<br>at N = %%{x:,.0f}<extra></extra>" % c["method"]))
    return _base(fig, dark, 470, xaxis=dict(type="log", title="total shot budget B per method (log)"),
                 yaxis=dict(type="log", title="RMSE = √(bias² + σ²/N)"),
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def shots_needed(rows: list, eps_grid, t_shot_us: float, dark=False, budget_s: float | None = None) -> go.Figure:
    """QPU time each method needs to reach RMSE ε (infinite once ε is below its bias)."""
    from ..analysis import budget_row
    sty = style_map([(r["method"], r["family"]) for r in rows])
    fig = go.Figure()
    for r in rows:
        col, dash, sym = sty[r["method"]]
        N = np.array([budget_row(r, e) for e in eps_grid], float)
        t = N * t_shot_us * 1e-6
        t[~np.isfinite(t)] = np.nan
        fig.add_trace(go.Scatter(x=eps_grid, y=t, mode="lines", name=r["method"], line=dict(color=col, width=2.4, dash=dash),
                                 hovertemplate="%s<br>ε = %%{x:.3g}<br>QPU time %%{y:.3g} s<extra></extra>" % r["method"]))
    if budget_s:
        fig.add_hline(y=budget_s, line=dict(color=T.YELLOW, width=1.6, dash="dash"))
    return _base(fig, dark, 420, xaxis=dict(type="log", title="target RMSE ε (log)", autorange="reversed"),
                 yaxis=dict(type="log", title="QPU time needed (s, log)"),
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def pec_wall_chart(layers, wall: dict, budget_s: float | None, dark=False) -> go.Figure:
    """QPU time PEC needs vs depth for several error rates: straight lines on a log axis = exponential cost."""
    fig = go.Figure()
    shades = ["#E0B35A", "#C98E26", "#B7791F", "#8E5B12", "#5E3B08"]
    for k, (p2, d) in enumerate(sorted(wall.items())):
        col = shades[min(k, len(shades) - 1)] if len(wall) <= 5 else T.FAMILY["pec"]
        fig.add_trace(go.Scatter(x=layers, y=d["seconds"], mode="lines+markers", name="p₂ = %.3g" % p2,
                                 line=dict(color=col, width=2.6), marker=dict(size=6),
                                 customdata=np.stack([d["gamma"], d["shots"]], 1),
                                 hovertemplate="p₂ = %g<br>%%{x} layers<br>Γ = %%{customdata[0]:.3g}<br>"
                                               "shots %%{customdata[1]:.3g}<br>QPU time %%{y:.3g} s<extra></extra>" % p2))
    if budget_s:
        fig.add_hline(y=budget_s, line=dict(color=T.ink(dark), width=1.5, dash="dash"))
        fig.add_annotation(x=0, xref="paper", y=math.log10(budget_s), yref="y", text="QPU budget %.3g s" % budget_s,
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12))
        for s, lab in ((3600.0, "1 hour"), (86400.0, "1 day"), (3.15e7, "1 year")):
            fig.add_hline(y=s, line=dict(color=T.GREY, width=0.8, dash="dot"))
            fig.add_annotation(x=1, xref="paper", y=math.log10(s), yref="y", text=lab, showarrow=False,
                               xanchor="right", yanchor="bottom", font=dict(size=11, color=T.MUTED))
    return _base(fig, dark, 460, xaxis=dict(title="circuit depth (layers)"),
                 yaxis=dict(type="log", title="QPU time for the target precision (s, log)"),
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def wall_heatmap(layers, p2s, n_qubits, p1, eps, t_shot_us, budget_s, construction="inverse", dark=False) -> go.Figure:
    from ..analysis import pec_wall
    w = pec_wall(layers, p2s, n_qubits, p1, eps, t_shot_us, construction)
    Z = np.array([np.log10(np.maximum(w[float(p)]["seconds"], 1e-12)) for p in p2s])
    fig = go.Figure(go.Heatmap(x=layers, y=p2s, z=Z, colorscale=[[0, "#F7F3EA"], [0.5, "#D9A441"], [1, "#6B3E00"]],
                               colorbar=dict(title=dict(text="log₁₀ s", side="right"), thickness=10),
                               hovertemplate="%{x} layers, p₂ = %{y:.3g}<br>10^%{z:.2f} s<extra></extra>"))
    if budget_s:
        fig.add_trace(go.Contour(x=layers, y=p2s, z=Z, contours=dict(start=math.log10(budget_s), end=math.log10(budget_s),
                                                                    size=1, coloring="none"),
                                 line=dict(color=T.ink(dark), width=2.5, dash="dash"), showscale=False,
                                 hoverinfo="skip", name="budget"))
    return _base(fig, dark, 380, xaxis=dict(title="layers"), yaxis=dict(title="two-qubit error p₂", type="log"))


def depth_chart(rows: list, N: int, dark=False) -> go.Figure:
    """RMSE at N shots vs depth, with the bias part drawn thin: where each method breaks as circuits grow."""
    from ..analysis import rmse_row
    methods = []
    for r in rows:
        if (r["method"], r["family"]) not in methods:
            methods.append((r["method"], r["family"]))
    sty = style_map(methods)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("RMSE at N = %s shots" % f"{N:,}", "sampling overhead γ²"),
                        horizontal_spacing=0.12)
    for m, fam in methods:
        rs = [r for r in rows if r["method"] == m]
        d = [r["depth"] for r in rs]
        col, dash, sym = sty[m]
        fig.add_trace(go.Scatter(x=d, y=[float(rmse_row(r, [N])[0]) for r in rs], mode="lines+markers",
                                 name=m, legendgroup=m, line=dict(color=col, dash=dash, width=2.4),
                                 marker=dict(symbol=sym, size=8, color=col)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d, y=[abs(r["bias"]) for r in rs], mode="lines", showlegend=False, legendgroup=m,
                                 line=dict(color=col, dash="dot", width=1), opacity=0.7, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d, y=[r["gamma"] ** 2 for r in rs], mode="lines+markers", showlegend=False,
                                 legendgroup=m, line=dict(color=col, dash=dash, width=2), marker=dict(symbol=sym, size=7,
                                                                                                     color=col)),
                      row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1, title="RMSE (dotted: |bias|)")
    fig.update_yaxes(type="log", row=1, col=2, title="γ²")
    fig.update_xaxes(title="depth", row=1, col=1)
    fig.update_xaxes(title="depth", row=1, col=2)
    return _base(fig, dark, 440, legend=dict(orientation="h", y=1.1, yanchor="bottom", x=0))


# ------------------------------------------------------------------ bias–variance bullseye
def bullseye(darts: list, N: int, labels=("O_A", "O_B"), dark=False, ring: float | None = None) -> go.Figure:
    """Each dart is one repeated experiment: (estimate − ideal) for two observables. The center is the truth.
    Bias = how far the cloud's center sits from the bullseye; variance = the size of the cloud (2σ ellipse)."""
    ink = T.ink(dark)
    sty = style_map([(d["method"], d["family"]) for d in darts])
    fig = go.Figure()
    allpts = np.concatenate([np.abs(np.concatenate([d["x"], d["y"]])) for d in darts if len(d["x"])] or [np.array([0.05])])
    R = float(np.quantile(allpts, 0.98)) * 1.15 if allpts.size else 0.1
    R = max(R, 1e-4)
    ring = ring or _nice(R / 4)
    t = np.linspace(0, 2 * np.pi, 121)
    for k in range(1, int(math.ceil(R / ring)) + 1):
        fig.add_trace(go.Scatter(x=k * ring * np.cos(t), y=k * ring * np.sin(t), mode="lines", showlegend=False,
                                 line=dict(color="rgba(150,150,150,%.2f)" % (0.55 if k == 1 else 0.3), width=1),
                                 hoverinfo="skip"))
        fig.add_annotation(x=k * ring * 0.7071, y=k * ring * 0.7071, text="%.3g" % (k * ring), showarrow=False,
                           font=dict(size=10, color=T.MUTED))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=14, color=T.RED, symbol="circle", line=dict(color="white", width=2))))
    for d in darts:
        col, dash, sym = sty[d["method"]]
        x, y = d["x"], d["y"]
        if len(x) == 0:
            continue
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=d["method"], legendgroup=d["method"],
                                 marker=dict(size=6, color=_rgba(col, 0.45), symbol=sym, line=dict(width=0)),
                                 hovertemplate=d["method"] + "<br>Δ%s = %%{x:.4f}<br>Δ%s = %%{y:.4f}<extra></extra>" % labels))
        mx, my = float(np.mean(x)), float(np.mean(y))
        if len(x) > 2:
            C = np.cov(np.vstack([x, y]))
            w, V = np.linalg.eigh(C + 1e-18 * np.eye(2))
            ell = (V @ np.diag(2 * np.sqrt(np.maximum(w, 0))) @ np.vstack([np.cos(t), np.sin(t)]))
            fig.add_trace(go.Scatter(x=mx + ell[0], y=my + ell[1], mode="lines", showlegend=False, legendgroup=d["method"],
                                     line=dict(color=col, width=2, dash=dash), hoverinfo="skip"))
        fig.add_annotation(x=mx, y=my, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor=col, text="")
        fig.add_trace(go.Scatter(x=[mx], y=[my], mode="markers", showlegend=False, legendgroup=d["method"],
                                 marker=dict(size=12, color=col, symbol=sym, line=dict(color=ink, width=1.2)),
                                 hovertemplate="%s center<br>bias (%.4f, %.4f)<br>|bias| = %.4f<extra></extra>"
                                               % (d["method"], mx, my, math.hypot(mx, my))))
    lim = R
    return _base(fig, dark, 560, xaxis=dict(title="error in %s" % labels[0], range=[-lim, lim], zeroline=False),
                 yaxis=dict(title="error in %s" % labels[1], range=[-lim, lim], scaleanchor="x", scaleratio=1,
                            zeroline=False),
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def _nice(x: float) -> float:
    if x <= 0:
        return 0.01
    e = math.floor(math.log10(x))
    f = x / 10 ** e
    return (1 if f < 1.5 else 2 if f < 3.5 else 5 if f < 7.5 else 10) * 10 ** e


def mse_bars(darts: list, N: int, dark=False) -> go.Figure:
    """MSE = bias² + variance for each method at N shots (sum over both observables)."""
    from ..analysis import mse_parts
    names = [d["method"] for d in darts]
    parts = [mse_parts(d, N) for d in darts]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[p["bias2"] for p in parts], orientation="h", name="bias²",
                         marker=dict(color=T.RED, line=dict(color="white", width=2)),
                         hovertemplate="%{y}<br>bias² = %{x:.3g}<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=[p["var"] for p in parts], orientation="h", name="variance σ²/N",
                         marker=dict(color=T.BLUE, line=dict(color="white", width=2)),
                         hovertemplate="%{y}<br>σ²/N = %{x:.3g}<extra></extra>"))
    return _base(fig, dark, 90 + 46 * len(names), barmode="stack",
                 xaxis=dict(title="mean squared error at N = %s (both observables)" % f"{N:,}"),
                 yaxis=dict(autorange="reversed", automargin=True), legend=dict(orientation="h", y=1.02, yanchor="bottom"))


# ------------------------------------------------------------------ model mismatch
def mismatch_curve(rows: list, dark=False, what="assumed / true error rate") -> go.Figure:
    ink = T.ink(dark)
    raw = next(r for r in rows if r["ratio"] is None)
    pts = [r for r in rows if r["ratio"] is not None]
    x = np.array([r["ratio"] for r in pts])
    y = np.array([r["estimate"] for r in pts])
    sd = np.array([r["sd"] for r in pts])
    col = T.FAMILY["pec"]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], horizontal_spacing=0.1,
                        subplot_titles=("PEC estimate vs what it assumes", "cost γ² it pays"))
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[y + sd, (y - sd)[::-1]], fill="toself",
                             fillcolor=_rgba(col, 0.18), line=dict(width=0), hoverinfo="skip", showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="PEC", line=dict(color=col, width=2.6),
                             marker=dict(symbol="diamond", size=8),
                             hovertemplate="ratio %{x:.2f}<br>estimate %{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=raw["ideal"], line=dict(color=ink, dash="dash", width=1.4), row=1, col=1)
    fig.add_hline(y=raw["estimate"], line=dict(color=T.GREY, dash="dot", width=1.4), row=1, col=1)
    fig.add_vline(x=1.0, line=dict(color=T.GREY, width=1), row=1, col=1)
    fig.add_annotation(x=x.max(), y=raw["ideal"], text="ideal", showarrow=False, xanchor="right", yanchor="bottom",
                       row=1, col=1)
    fig.add_annotation(x=x.max(), y=raw["estimate"], text="unmitigated", showarrow=False, xanchor="right",
                       yanchor="top", font=dict(color=T.MUTED), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["gamma"] ** 2 for r in pts], mode="lines+markers", showlegend=False,
                             line=dict(color=col, width=2), marker=dict(size=6)), row=1, col=2)
    fig.update_xaxes(title=what, row=1, col=1)
    fig.update_xaxes(title=what, row=1, col=2)
    fig.update_yaxes(title="⟨O⟩", row=1, col=1)
    fig.update_yaxes(title="γ² (shot overhead)", type="log", row=1, col=2)
    return _base(fig, dark, 430, showlegend=False)


def mismatch_heatmap(B, assumed, true, dark=False) -> go.Figure:
    """Diverging map of PEC's bias: red = under-corrected (too low), blue = over-corrected; white diagonal = matched."""
    B = np.asarray(B, float)
    m = float(np.nanmax(np.abs(B))) or 1e-3
    fig = go.Figure(go.Heatmap(x=assumed, y=true, z=B, zmin=-m, zmax=m, zmid=0,
                               colorscale=[[0, T.RED], [0.5, "#F2F2F2"], [1, T.BLUE]],
                               colorbar=dict(title=dict(text="bias", side="right"), thickness=10),
                               hovertemplate="assumed p₂ %{x:.3g}<br>true p₂ %{y:.3g}<br>bias %{z:+.4f}<extra></extra>"))
    lo, hi = min(min(assumed), min(true)), max(max(assumed), max(true))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=T.ink(dark), width=1.5, dash="dash"),
                             hoverinfo="skip", showlegend=False))
    return _base(fig, dark, 430, xaxis=dict(title="assumed error rate p₂ (the model PEC is given)"),
                 yaxis=dict(title="true device error rate p₂"))


def drift_curve(rows: list, dark=False) -> go.Figure:
    ink = T.ink(dark)
    x = [r["drift"] * 100 for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[r["estimate"] for r in rows], mode="lines+markers", name="readout-mitigated",
                             line=dict(color=T.FAMILY["readout"], width=2.6), marker=dict(size=7)))
    fig.add_hline(y=rows[0]["ideal"], line=dict(color=ink, dash="dash", width=1.3))
    fig.add_hline(y=rows[0]["raw"], line=dict(color=T.GREY, dash="dot", width=1.3))
    fig.add_vline(x=0, line=dict(color=T.GREY, width=1))
    return _base(fig, dark, 360, xaxis=dict(title="calibration error of the readout rates (%)"),
                 yaxis=dict(title="⟨O⟩ (dashed: ideal, dotted: unmitigated)"), showlegend=False)


# ------------------------------------------------------------------ head to head
def _sd_row(r: dict, B: int) -> float:
    """Standard error of a row's method at total budget B."""
    from ..budget import CostModel, variance
    var = variance(CostModel.of_row(r), B)
    return math.sqrt(var) if math.isfinite(var) else float("nan")


def sweep_lines(rows: list, N: int, xlabel: str, metric: str = "RMSE", dark=False) -> go.Figure:
    from ..analysis import rmse_row
    methods = []
    for r in rows:
        if (r["method"], r["family"]) not in methods:
            methods.append((r["method"], r["family"]))
    sty = style_map(methods)
    fig = go.Figure()
    for m, fam in methods:
        rs = sorted([r for r in rows if r["method"] == m and np.isfinite(r["bias"])], key=lambda r: r["value"])
        if not rs:
            continue
        col, dash, sym = sty[m]
        xs = [r["value"] for r in rs]
        if metric == "RMSE":
            ys = [float(rmse_row(r, [N])[0]) for r in rs]
        elif metric == "|bias|":
            ys = [abs(r["bias"]) for r in rs]
        elif metric == "estimate":
            ys = [r["estimate"] for r in rs]
        elif metric == "sd":
            ys = [_sd_row(r, N) for r in rs]
        else:
            ys = [r["gamma"] ** 2 for r in rs]
        if metric == "estimate":
            sd = np.array([_sd_row(r, N) for r in rs])
            fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[np.array(ys) + sd, (np.array(ys) - sd)[::-1]],
                                     fill="toself", fillcolor=_rgba(col, 0.14), line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, legendgroup=m))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=m, legendgroup=m,
                                 line=dict(color=col, dash=dash, width=2.4), marker=dict(symbol=sym, size=8, color=col)))
    if metric == "estimate" and rows:
        fig.add_trace(go.Scatter(x=sorted({r["value"] for r in rows}),
                                 y=[next(r["ideal"] for r in rows if r["value"] == v) for v in sorted({r["value"] for r in rows})],
                                 mode="lines", name="ideal", line=dict(color=T.ink(dark), dash="dash", width=1.4)))
    ylab = {"RMSE": "RMSE at N = %s" % f"{N:,}", "|bias|": "|bias|", "estimate": "⟨O⟩ (±1σ)", "sd": "σ at N = %s" % f"{N:,}",
            "γ²": "γ² (shot overhead)"}[metric]
    return _base(fig, dark, 440, xaxis=dict(title=xlabel),
                 yaxis=dict(title=ylab, type="log" if metric in ("RMSE", "|bias|", "sd", "γ²") else "linear"),
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def winner_heatmap(vals, methods, families, win, best, second, shots, xlabel, dark=False) -> go.Figure:
    """Which method has the lowest RMSE for each (noise setting, shot budget)."""
    sty = style_map(list(zip(methods, families)))
    k = len(methods)
    colors = [sty[m][0] for m in methods]
    # discrete color scale: one band per method
    cs = []
    for i, c in enumerate(colors):
        cs += [[i / k, c], [(i + 1) / k, c]]
    Z = np.asarray(win, float).T
    text = [["%s<br>RMSE %.3g (next best %.3g)" % (methods[int(win[i, j])], best[i, j], second[i, j])
             for i in range(len(vals))] for j in range(len(shots))]
    fig = go.Figure(go.Heatmap(x=vals, y=shots, z=Z, zmin=-0.5, zmax=k - 0.5, colorscale=cs, text=text,
                               hovertemplate="%{text}<extra></extra>", showscale=False, xgap=1, ygap=1))
    for i, m in enumerate(methods):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=m,
                                 marker=dict(size=12, symbol="square", color=colors[i])))
    # label the regions: center of each method's cells
    for i, m in enumerate(methods):
        pts = np.argwhere(np.asarray(win).T == i)
        if len(pts) >= 3:
            jy, ix = np.median(pts[:, 0]), np.median(pts[:, 1])
            fig.add_annotation(x=vals[int(round(ix))], y=shots[int(round(jy))], text=m.split(" (")[0], showarrow=False,
                               font=dict(size=11, color="white"), bgcolor="rgba(0,0,0,0.35)")
    return _base(fig, dark, 460, xaxis=dict(title=xlabel), yaxis=dict(title="total shot budget per method", type="log"),
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def duel_heatmap(vals, shots, ratio, a: str, b: str, xlabel, dark=False) -> go.Figure:
    """log10 of RMSE(a)/RMSE(b): blue where a wins, red where b wins."""
    L = np.log10(np.asarray(ratio, float)).T
    m = float(np.nanmax(np.abs(L[np.isfinite(L)]))) if np.isfinite(L).any() else 1.0
    fig = go.Figure(go.Heatmap(x=vals, y=shots, z=L, zmid=0, zmin=-m, zmax=m,
                               colorscale=[[0, T.BLUE], [0.5, "#F2F2F2"], [1, T.RED]],
                               colorbar=dict(title=dict(text="log₁₀ RMSE ratio", side="right"), thickness=10),
                               hovertemplate="%s / %s = 10^%%{z:.2f}<extra></extra>" % (a, b)))
    return _base(fig, dark, 420, xaxis=dict(title=xlabel), yaxis=dict(title="total shot budget per method", type="log"),
                 title=dict(text="blue: %s wins · red: %s wins" % (a, b), font=dict(size=13)))
