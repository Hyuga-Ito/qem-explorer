"""State views for the time-travel inspector: Bloch spheres, the density matrix, and the
fidelity / purity trace along the circuit."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T


AXIS_LABELS = [((0, 0, 1.32), "|0⟩"), ((0, 0, -1.32), "|1⟩"), ((1.34, 0, 0), "|+⟩"), ((-1.34, 0, 0), "|−⟩"),
               ((0, 1.34, 0), "|+i⟩"), ((0, -1.34, 0), "|−i⟩")]
NOISY = "#1D6FA3"
TILT = "#B7791F"


def tilt_deg(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def angles(r) -> tuple:
    """(θ, φ) in degrees of a Bloch vector (θ from |0⟩, φ from +x toward +y)."""
    x, y, z = r
    L = np.linalg.norm(r)
    if L < 1e-9:
        return float("nan"), float("nan")
    th = np.degrees(np.arccos(np.clip(z / L, -1, 1)))
    ph = np.degrees(np.arctan2(y, x)) if np.hypot(x, y) > 1e-9 else float("nan")
    return float(th), float(ph)


def _slerp(a, b, t):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return np.outer(1 - t, a) + np.outer(t, b)
    return (np.sin((1 - t) * om)[:, None] * a + np.sin(t * om)[:, None] * b) / np.sin(om)


def _sphere_traces(scene: str, dark: bool) -> list:
    out = []
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    out.append(go.Surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v), scene=scene,
                          colorscale=[[0, "#9FB7C9"], [1, "#9FB7C9"]], showscale=False, opacity=0.10 if not dark else 0.14,
                          hoverinfo="skip", lighting=dict(ambient=1, diffuse=0, specular=0), contours=dict(
                              x=dict(highlight=False), y=dict(highlight=False), z=dict(highlight=False))))
    t = np.linspace(0, 2 * np.pi, 91)
    ring = "rgba(110,110,110,0.55)" if not dark else "rgba(200,200,200,0.5)"
    faint = "rgba(140,140,140,0.22)"
    for (xs, ys, zs), col, w in (((np.cos(t), np.sin(t), 0 * t), ring, 2.5),       # equator
                                 ((np.cos(t), 0 * t, np.sin(t)), faint, 1.5),       # x-z meridian
                                 ((0 * t, np.cos(t), np.sin(t)), faint, 1.5)):      # y-z meridian
        out.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=w), hoverinfo="skip",
                                showlegend=False, scene=scene))
    axc = "#9A9A9A" if not dark else "#AAAAAA"
    for v3 in ((1.15, 0, 0), (0, 1.15, 0), (0, 0, 1.15)):
        out.append(go.Scatter3d(x=[-v3[0], v3[0]], y=[-v3[1], v3[1]], z=[-v3[2], v3[2]], mode="lines",
                                line=dict(color=axc, width=1.5), hoverinfo="skip", showlegend=False, scene=scene))
    out.append(go.Scatter3d(x=[p[0] for p, _ in AXIS_LABELS], y=[p[1] for p, _ in AXIS_LABELS],
                            z=[p[2] for p, _ in AXIS_LABELS], mode="text", text=[l for _, l in AXIS_LABELS],
                            textfont=dict(size=12, color=T.ink(dark)), hoverinfo="skip", showlegend=False, scene=scene))
    return out


def _vector_traces(scene, r, color, name, dash, show, cone=True, width=8) -> list:
    x, y, z = r
    L = float(np.linalg.norm(r))
    th, ph = angles(r)
    hov = "%s<br>r = (%.3f, %.3f, %.3f)<br>|r| = %.3f<br>θ = %.1f°, φ = %s<extra></extra>" % (
        name, x, y, z, L, th, "–" if not np.isfinite(ph) else "%.1f°" % ph)
    out = [go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines", line=dict(color=color, width=width, dash=dash),
                        name=name, legendgroup=name, showlegend=show, scene=scene, hovertemplate=hov)]
    if cone and L > 0.05:
        d = np.array(r) / L
        out.append(go.Cone(x=[x], y=[y], z=[z], u=[d[0]], v=[d[1]], w=[d[2]], sizemode="absolute", sizeref=0.22,
                           anchor="tip", colorscale=[[0, color], [1, color]], showscale=False, scene=scene,
                           hoverinfo="skip", legendgroup=name))
    else:
        out.append(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers", legendgroup=name, showlegend=False, scene=scene,
                                marker=dict(size=5, color="rgba(0,0,0,0)", line=dict(color=color, width=3)),
                                hoverinfo="skip"))
    return out


def grid_shape(n: int, per_row: int | None = None) -> tuple:
    """(rows, columns) of the responsive grid: one row up to 4 qubits, then rows of 4 (5 from 17 qubits)."""
    if per_row is None:
        per_row = n if n <= 4 else (4 if n <= 16 else 5)
    per_row = max(1, min(per_row, n))
    return -(-n // per_row), per_row


def bloch_row(noisy: np.ndarray, ideal: np.ndarray, dark: bool = False, color: str = NOISY, labels=None,
              height: int = 360, trail_noisy=None, trail_ideal=None, show_shadow: bool = True,
              per_row: int | None = None) -> go.Figure:
    """One Bloch sphere per qubit, designed to be read at a glance:

    * the **ideal** state is a dashed ink arrow with an open tip; the **noisy** state a solid blue arrow with a cone;
    * the **tilt** between them is an amber arc labeled in degrees (coherent errors rotate the arrow);
    * the **length** |r| is printed in the title (incoherent errors and entanglement shorten the arrow — if the
      ideal arrow is short too, the qubit is entangled, not noisy);
    * the **trail** shows where the noisy tip has been during the circuit (optional);
    * a dotted **shadow** drops the noisy tip onto the equator, making the phase φ easy to read.

    From 5 qubits the spheres wrap into a grid of 4 per row (``height`` is then per row), so a 7-qubit code or a
    16-qubit register stays readable at a glance without scrolling sideways.
    """
    n = noisy.shape[0]
    R_, C_ = grid_shape(n, per_row)
    titles = []
    for q in range(n):
        tl = tilt_deg(noisy[q], ideal[q])
        if C_ <= 2:
            titles.append("<b>q%d</b>  |r| = %.3f <span style='color:#8A8A8A'>(ideal %.2f)</span>%s" % (
                q, np.linalg.norm(noisy[q]), np.linalg.norm(ideal[q]), "" if not np.isfinite(tl) else "  ·  tilt %.1f°" % tl))
        else:                                    # 3+ per row: two short lines so neighboring titles never collide
            titles.append("<b>q%d</b> |r| %.2f <span style='color:#8A8A8A'>(%.2f)</span><br>%s" % (
                q, np.linalg.norm(noisy[q]), np.linalg.norm(ideal[q]), "" if not np.isfinite(tl) else "tilt %.1f°" % tl))
    specs = [[({"type": "scene"} if r * C_ + c < n else None) for c in range(C_)] for r in range(R_)]
    fig = make_subplots(rows=R_, cols=C_, specs=specs, subplot_titles=titles, horizontal_spacing=0.01,
                        vertical_spacing=0.06 if R_ > 1 else 0.0)
    ink = T.ink(dark)
    for q in range(n):
        sc = "scene" if q == 0 else "scene%d" % (q + 1)
        for tr in _sphere_traces(sc, dark):
            fig.add_trace(tr)
        if trail_noisy is not None and len(trail_noisy) > 1:
            tn = np.asarray(trail_noisy)[:, q]
            fig.add_trace(go.Scatter3d(x=tn[:, 0], y=tn[:, 1], z=tn[:, 2], mode="lines+markers", scene=sc,
                                       line=dict(color=color, width=2.5), marker=dict(size=2.2, color=color),
                                       opacity=0.55, name="noisy path", legendgroup="trail", showlegend=q == 0,
                                       hoverinfo="skip"))
        if trail_ideal is not None and len(trail_ideal) > 1:
            ti = np.asarray(trail_ideal)[:, q]
            fig.add_trace(go.Scatter3d(x=ti[:, 0], y=ti[:, 1], z=ti[:, 2], mode="lines", scene=sc,
                                       line=dict(color=T.GREY, width=2, dash="dot"), opacity=0.7, name="ideal path",
                                       legendgroup="trailI", showlegend=q == 0, hoverinfo="skip"))
        for tr in _vector_traces(sc, ideal[q], ink, "ideal", "dash", q == 0, cone=False, width=6):
            fig.add_trace(tr)
        for tr in _vector_traces(sc, noisy[q], color, "noisy", "solid", q == 0):
            fig.add_trace(tr)
        a, b = np.asarray(ideal[q]), np.asarray(noisy[q])
        tl = tilt_deg(b, a)
        if np.isfinite(tl) and tl > 0.5:
            arc = _slerp(a, b, np.linspace(0, 1, 30)) * 0.42
            mid = arc[len(arc) // 2] * 1.25
            fig.add_trace(go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines", scene=sc,
                                       line=dict(color=TILT, width=6), name="tilt", legendgroup="tilt",
                                       showlegend=q == 0, hoverinfo="skip"))
            fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode="text", text=["%.1f°" % tl], scene=sc,
                                       textfont=dict(size=12, color=TILT), showlegend=False, hoverinfo="skip"))
        if show_shadow and np.hypot(b[0], b[1]) > 0.05:
            fig.add_trace(go.Scatter3d(x=[b[0], b[0], 0], y=[b[1], b[1], 0], z=[b[2], 0, 0], mode="lines", scene=sc,
                                       line=dict(color=color, width=2, dash="dot"), opacity=0.6, showlegend=False,
                                       hoverinfo="skip"))
    lim = 1.5 if C_ <= 2 else 1.62                      # keep the |0⟩ label inside the scene, below the title
    cfg = dict(xaxis=dict(visible=False, range=[-lim, lim]), yaxis=dict(visible=False, range=[-lim, lim]),
               zaxis=dict(visible=False, range=[-lim, lim]), aspectmode="cube", dragmode="orbit",
               camera=dict(eye=dict(x=1.12, y=0.56, z=0.42), up=dict(x=0, y=0, z=1)), uirevision="bloch")
    fig.update_layout(**{("scene" if q == 0 else "scene%d" % (q + 1)): cfg for q in range(n)})
    fig.update_layout(template=T.tpl(dark), height=int(height * (1 if R_ == 1 else 0.92 * R_)),
                      margin=dict(l=0, r=0, t=46 if C_ <= 2 else 60, b=0), uirevision="bloch",
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.0 + 0.1 / R_, yanchor="bottom"))
    for ann in fig.layout.annotations:                  # (legend on the left: the toolbar appears on the right)
        ann.font = dict(size=12)
    return fig


def bloch_projections(noisy: np.ndarray, ideal: np.ndarray, dark: bool = False, color: str = NOISY,
                      trail_noisy=None, height: int | None = None, per_row: int | None = None) -> go.Figure:
    """Flat views that need no 3D reading: for each qubit the side view (x–z plane: how far from |0⟩/|1⟩)
    and the top view (x–y plane: the phase φ). Arrows, the tilt arc and the unit circle are exact.
    Up to 4 qubits: side views in the first row, top views in the second. From 5 qubits each qubit gets a
    (side | top) pair and the pairs wrap into rows of 4 qubits, so the page grows downwards, never sideways."""
    n = noisy.shape[0]
    paired = n > 4
    if paired:
        P = per_row or 4
        R_, C_ = -(-n // P), 2 * P
        cell = {(q, v): (q // P + 1, 2 * (q % P) + v + 1) for q in range(n) for v in (0, 1)}
    else:
        R_, C_ = 2, n
        cell = {(q, v): (v + 1, q + 1) for q in range(n) for v in (0, 1)}
    inv = {rc: qv for qv, rc in cell.items()}
    titles = []
    for r in range(1, R_ + 1):
        for c in range(1, C_ + 1):
            qv = inv.get((r, c))
            titles.append("" if qv is None else ("q%d · %s" % (qv[0], ("side (x–z)", "top (x–y)")[qv[1]]) if paired
                                                  else "q%d · %s" % (qv[0], ("side view (x–z)", "top view (x–y)")[qv[1]])))
    fig = make_subplots(rows=R_, cols=C_, subplot_titles=titles, horizontal_spacing=0.03 if paired else 0.04,
                        vertical_spacing=0.12 if R_ <= 2 else 0.25 / R_)
    where = {(r, c): (r - 1) * C_ + c for r in range(1, R_ + 1) for c in range(1, C_ + 1)}
    ink = T.ink(dark)
    t = np.linspace(0, 2 * np.pi, 121)
    for q in range(n):
        for rr, (i, j, lx, ly) in enumerate(((0, 2, ("|−⟩", "|+⟩"), ("|1⟩", "|0⟩")),
                                            (0, 1, ("|−⟩", "|+⟩"), ("|−i⟩", "|+i⟩")))):
            row, col = cell[(q, rr)]
            fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#A2A9B0", width=1.5),
                                     hoverinfo="skip", showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=0.5 * np.cos(t), y=0.5 * np.sin(t), mode="lines",
                                     line=dict(color="rgba(160,160,160,0.35)", width=1, dash="dot"), hoverinfo="skip",
                                     showlegend=False), row=row, col=col)
            for xs, ys in (([-1.1, 1.1], [0, 0]), ([0, 0], [-1.1, 1.1])):
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(150,150,150,0.5)", width=1),
                                         hoverinfo="skip", showlegend=False), row=row, col=col)
            if trail_noisy is not None and len(trail_noisy) > 1:
                tn = np.asarray(trail_noisy)[:, q]
                fig.add_trace(go.Scatter(x=tn[:, i], y=tn[:, j], mode="lines", line=dict(color=color, width=1.5),
                                         opacity=0.45, hoverinfo="skip", showlegend=False), row=row, col=col)
            a, b = np.asarray(ideal[q]), np.asarray(noisy[q])
            fig.add_trace(go.Scatter(x=[0, a[i]], y=[0, a[j]], mode="lines+markers", line=dict(color=ink, width=2, dash="dash"),
                                     marker=dict(size=[0, 9], symbol="circle-open", color=ink), name="ideal",
                                     showlegend=(q == 0 and rr == 0), legendgroup="ideal",
                                     hovertemplate="ideal (%.3f, %.3f)<extra></extra>" % (a[i], a[j])),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=[0, b[i]], y=[0, b[j]], mode="lines", line=dict(color=color, width=3.5),
                                     name="noisy", showlegend=(q == 0 and rr == 0), legendgroup="noisy",
                                     hovertemplate="noisy (%.3f, %.3f)<extra></extra>" % (b[i], b[j])),
                          row=row, col=col)
            ax_i = where[(row, col)]
            xr = "x%s" % ("" if ax_i == 1 else ax_i)
            if np.hypot(b[i], b[j]) > 0.04:
                fig.add_annotation(x=b[i], y=b[j], ax=0, ay=0, xref=xr, yref=xr.replace("x", "y"), axref=xr,
                                   ayref=xr.replace("x", "y"), showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2.5,
                                   arrowcolor=color, text="")
            ta, tb_ = np.arctan2(a[j], a[i]), np.arctan2(b[j], b[i])
            if np.hypot(a[i], a[j]) > 0.05 and np.hypot(b[i], b[j]) > 0.05:
                d = (tb_ - ta + np.pi) % (2 * np.pi) - np.pi
                if abs(np.degrees(d)) > 0.5:
                    s = np.linspace(ta, ta + d, 30)
                    fig.add_trace(go.Scatter(x=0.33 * np.cos(s), y=0.33 * np.sin(s), mode="lines",
                                             line=dict(color=TILT, width=3), hoverinfo="skip", showlegend=False),
                                  row=row, col=col)
                    fig.add_annotation(x=0.5 * np.cos(ta + d / 2), y=0.5 * np.sin(ta + d / 2), xref=xr,
                                       yref=xr.replace("x", "y"), text="%.1f°" % abs(np.degrees(d)), showarrow=False,
                                       font=dict(size=11, color=TILT))
            for (px, py, txt) in ((-1.28, 0, lx[0]), (1.28, 0, lx[1]), (0, -1.22, ly[0]), (0, 1.22, ly[1])):
                fig.add_annotation(x=px, y=py, xref=xr, yref=xr.replace("x", "y"), text=txt, showarrow=False,
                                   font=dict(size=11, color=T.MUTED))
            fig.update_xaxes(range=[-1.45, 1.45], visible=False, row=row, col=col, fixedrange=True)
            fig.update_yaxes(range=[-1.35, 1.35], visible=False, row=row, col=col, scaleanchor=xr, scaleratio=1,
                             fixedrange=True)
    for (r, c) in where:                               # hide the axes of empty cells in the last row
        if (r, c) not in inv:
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)
    h = (height or 520) if not paired else int(40 + 235 * R_)
    fig.update_layout(template=T.tpl(dark), height=h, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.0 + 0.08 / max(R_ - 1, 1),
                                  yanchor="bottom"))
    for ann in fig.layout.annotations:
        if ann.text and ann.text.startswith("q"):
            ann.font = dict(size=12)
    return fig


def density_hinton(rho: np.ndarray, n: int, dark: bool = False, title: str = "", ref: np.ndarray | None = None,
                   mode: str = "state") -> go.Figure:
    """Hinton-style density matrix: square area ∝ |ρ_ij|, color = phase arg ρ_ij (cyclic scale).
    mode 'error' draws ρ − ρ_ideal."""
    M = rho if mode == "state" or ref is None else rho - ref
    d = 2 ** n
    lab = [format(i, "0%db" % n) for i in range(d)]
    mag = np.abs(M)
    ph = np.angle(M)
    I, J = np.meshgrid(range(d), range(d), indexing="ij")
    scale = max(mag.max(), 1e-12)
    keep = mag.ravel() > 1e-4 * scale
    size = (np.sqrt(mag.ravel() / scale) * (360 / d))[keep]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=J.ravel()[keep], y=I.ravel()[keep], mode="markers",
        marker=dict(symbol="square", size=size, color=ph.ravel()[keep], cmin=-np.pi, cmax=np.pi, colorscale="Twilight",
                    line=dict(width=0),
                    colorbar=dict(title=dict(text="arg ρ<sub>ij</sub>", side="right"), tickvals=[-np.pi, 0, np.pi],
                                  ticktext=["−π", "0", "π"], thickness=10, len=0.7)),
        customdata=np.stack([mag.ravel()[keep], ph.ravel()[keep]], 1),
        text=["⟨%s|ρ|%s⟩" % (lab[i], lab[j]) for i, j in zip(I.ravel()[keep], J.ravel()[keep])],
        hovertemplate="%{text}<br>|ρ| = %{customdata[0]:.4f}<br>arg = %{customdata[1]:.3f} rad<extra></extra>",
        showlegend=False))
    fig.update_layout(template=T.tpl(dark), height=380, title=title or None, margin=dict(l=50, r=10, t=36 if title else 10, b=40),
                      xaxis=dict(tickmode="array", tickvals=list(range(d)), ticktext=lab, range=[-0.6, d - 0.4],
                                 tickangle=-90 if n > 2 else 0, tickfont=dict(size=9 if n > 3 else 11), showgrid=False,
                                 showline=False, ticks=""),
                      yaxis=dict(tickmode="array", tickvals=list(range(d)), ticktext=lab, range=[d - 0.4, -0.6],
                                 tickfont=dict(size=9 if n > 3 else 11), showgrid=False, showline=False, ticks="",
                                 scaleanchor="x", scaleratio=1))
    return fig


def observable_trace(steps, noisy, ideal, cursor: int | None, labels, label: str = "⟨O⟩", dark: bool = False,
                     color: str = T.FAMILY["zne"]) -> go.Figure:
    """⟨O⟩ = Tr(ρ O) after every step for the noisy and the ideal state ("if measured now", before readout)."""
    ink = T.ink(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=ideal, mode="lines", name="ideal", line=dict(color=ink, width=1.6, dash="dash"),
                             text=labels, hovertemplate="step %{x}: %{text}<br>ideal %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=steps, y=noisy, mode="lines+markers", name="noisy", line=dict(color=color, width=2.2),
                             marker=dict(size=5), text=labels,
                             hovertemplate="step %{x}: %{text}<br>noisy %{y:.4f}<extra></extra>"))
    if cursor is not None:
        fig.add_vline(x=cursor, line=dict(color=T.BLUE, width=1.5))
    fig.update_layout(template=T.tpl(dark), height=240, margin=dict(l=55, r=15, t=30, b=45),
                      xaxis=dict(title="circuit step"), yaxis=dict(title=label[:40]))
    return fig


def trace(steps, fid, pur, cursor: int | None, labels, dark: bool = False, color: str = T.FAMILY["zne"]) -> go.Figure:
    """Fidelity to the ideal state and purity after every step, with the inspected step marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=fid, mode="lines+markers", name="fidelity F(ρ, ρ<sub>ideal</sub>)",
                             line=dict(color=color, width=2.2), marker=dict(size=5), text=labels,
                             hovertemplate="step %{x}: %{text}<br>F = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=steps, y=pur, mode="lines", name="purity Tr ρ²",
                             line=dict(color=T.GREY, width=1.6, dash="dash"), text=labels,
                             hovertemplate="step %{x}: %{text}<br>Tr ρ² = %{y:.4f}<extra></extra>"))
    if cursor is not None:
        fig.add_vline(x=cursor, line=dict(color=T.BLUE, width=1.5))
    lo = min(min(fid), min(pur))
    fig.update_layout(template=T.tpl(dark), height=260, margin=dict(l=55, r=15, t=30, b=45),
                      xaxis=dict(title="circuit step"), yaxis=dict(title="", range=[max(0, lo - 0.05), 1.01]))
    return fig


# ------------------------------------------------------------------ variational landscapes
def landscape_1d(x, ideal, noisy, current: float, xlabel: str, ylabel: str, dark: bool = False) -> go.Figure:
    """E(θ) along one parameter, ideal (dashed ink) and noisy (blue), the current value marked."""
    ink = T.ink(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ideal, mode="lines", name="ideal", line=dict(color=ink, width=1.6, dash="dash"),
                             hovertemplate=xlabel + " = %{x:.3f}<br>ideal %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=noisy, mode="lines", name="noisy (unmitigated)",
                             line=dict(color=T.FAMILY["zne"], width=2.2),
                             hovertemplate=xlabel + " = %{x:.3f}<br>noisy %{y:.4f}<extra></extra>"))
    fig.add_vline(x=current, line=dict(color=T.BLUE, width=1.5), annotation_text="now", annotation_position="top")
    fig.update_layout(template=T.tpl(dark), height=300, margin=dict(l=60, r=15, t=30, b=45),
                      xaxis=dict(title=xlabel), yaxis=dict(title=ylabel[:40]), hovermode="x unified",
                      legend=dict(orientation="h", x=0, y=1.12))
    return fig


def landscape_2d(gx, gy, ideal, noisy, current: tuple, xlabel: str, ylabel: str, zlabel: str,
                 maximise: bool, dark: bool = False) -> go.Figure:
    """Two heatmaps (ideal | noisy) of E over a 2D grid with a shared single-hue scale, the current point (ring) and
    each map's best point (star)."""
    from plotly.subplots import make_subplots
    ideal, noisy = np.asarray(ideal, float), np.asarray(noisy, float)
    lo = float(np.nanmin([np.nanmin(ideal), np.nanmin(noisy)]))
    hi = float(np.nanmax([np.nanmax(ideal), np.nanmax(noisy)]))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ideal", "noisy (unmitigated)"), horizontal_spacing=0.09,
                        shared_yaxes=True)
    scale = [[0, "#F3F7FA"], [0.5, "#6FA3C0"], [1, "#0B3D56"]]
    if not maximise:
        scale = [[0, "#0B3D56"], [0.5, "#6FA3C0"], [1, "#F3F7FA"]]           # darker = better either way
    for c, Z in ((1, ideal), (2, noisy)):
        fig.add_trace(go.Heatmap(x=gx, y=gy, z=Z.T, zmin=lo, zmax=hi, colorscale=scale, showscale=c == 2,
                                 colorbar=dict(title=dict(text=zlabel[:24], side="top"), thickness=10, len=0.8),
                                 hovertemplate=xlabel + " %{x:.3f}<br>" + ylabel + " %{y:.3f}<br>%{z:.4f}<extra></extra>"),
                      row=1, col=c)
        j = np.unravel_index(np.nanargmax(Z) if maximise else np.nanargmin(Z), Z.shape)
        fig.add_trace(go.Scatter(x=[gx[j[0]]], y=[gy[j[1]]], mode="markers", showlegend=c == 1, name="best on grid",
                                 marker=dict(symbol="star", size=14, color=T.YELLOW, line=dict(color="white", width=1)),
                                 hovertemplate="best: %{x:.3f}, %{y:.3f}<extra></extra>"), row=1, col=c)
        fig.add_trace(go.Scatter(x=[current[0]], y=[current[1]], mode="markers", showlegend=c == 1, name="now",
                                 marker=dict(symbol="circle-open", size=14, color=T.RED, line=dict(width=2.5)),
                                 hovertemplate="now: %{x:.3f}, %{y:.3f}<extra></extra>"), row=1, col=c)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    fig.update_layout(template=T.tpl(dark), height=410, margin=dict(l=60, r=20, t=40, b=95),
                      legend=dict(orientation="h", x=0, y=-0.3, yanchor="top"))
    return fig
