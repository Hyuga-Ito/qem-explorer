"""An IBM-Quantum-Composer-style circuit drawing in Plotly, collision-free by construction.

Geometry lives in data units: one column per time step, one row per qubit, and every gate is a box of at most
0.72 × 0.68 of its cell, so neighboring gates, outlines and group frames always keep a gap however the figure
is resized, zoomed or scrolled. Only text and small markers have pixel sizes; they are scaled from the pixels a
column is expected to get (``width_px`` / visible columns), and parameter labels move to the hover text when a
cell is too narrow to hold them.

Gates are colored like the Composer (H red, Paulis and two-qubit gates navy, phase gates cyan, pulsed rotations
magenta); noise operations are cream boxes. What a QEM method added is shown without covering the gates: folded
circuits get labeled bands behind their U and U† blocks, and PEC/twirl/DD/training insertions get an outline
inside their own cell. ``groups`` draws frames behind sets of columns (the compile diff uses them), and a cursor
outlines one operation (only that gate and the wires it acts on) for the time-travel inspector. Very long circuits
are drawn in a window of columns around the cursor so the figure stays light in the browser.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..circuit import GATES, CircuitSpec, fmt_angle_tex
from . import theme as T

BOX_H = 0.34                 # half height of a gate box (qubit pitch = 1)
BOX_W = 0.36                 # maximal half width (column pitch = 1)
MAX_OPS = 2500               # beyond this many operations only a window of columns is drawn
FOLD_FILL = {"fold-inv": "rgba(105,41,196,0.10)", "fold": "rgba(105,41,196,0.045)"}
FOLD_LABEL = {"fold-inv": "U†", "fold": "U"}


def _segment(o) -> str:
    return o.tag if o.tag in ("fold", "fold-inv") else ""


def columns(spec: CircuitSpec) -> tuple:
    """(list of (column, op index)), number of columns). Each op occupies the span of its qubits, so a gate on
    distant qubits blocks the wires its connector crosses. Where a folded circuit switches between U and U†
    blocks, all wires are aligned first, so each block gets columns of its own (drawing only; the simulation's
    timing is unchanged)."""
    n = spec.n
    level = [0] * n
    out = []
    seg = None
    for i, o in enumerate(spec.ops):
        s = _segment(o)
        if seg is not None and s != seg and (s or seg):
            m = max(level)
            level = [m] * n
        seg = s
        if o.kind == "barrier":
            qs = o.qubits or tuple(range(n))
            m = max(level[q] for q in qs)
            for q in qs:
                level[q] = m
            out.append((m, i))
            continue
        lo, hi = min(o.qubits), max(o.qubits)
        span = range(lo, hi + 1)
        c = max(level[q] for q in span)
        out.append((c, i))
        for q in span:
            level[q] = c + 1
    return out, (max(level) if level else 0)


def column_of(spec: CircuitSpec) -> dict:
    return {i: c for c, i in columns(spec)[0]}


def _fill(o):
    if o.kind == "noise":
        return T.GATE_COLOR["noise"]
    if o.kind == "box":
        return T.GATE_COLOR["box"]
    fam = GATES[o.name][3]
    if o.name in ("x", "y", "z", "id"):
        fam = "pauli"
    return T.GATE_COLOR.get(fam, T.GATE_COLOR["pauli"])


class _Batch:
    """Collects shapes and annotations and hands them to Plotly in one call (``Figure.add_shape`` copies the
    whole shape tuple each time, which is quadratic for circuits with thousands of gates)."""

    def __init__(self, fig):
        self.fig, self.shapes, self.notes = fig, [], []

    def add_shape(self, **kw):
        self.shapes.append(kw)

    def add_annotation(self, **kw):
        self.notes.append(kw)

    def __getattr__(self, name):
        return getattr(self.fig, name)

    def done(self):
        self.fig.update_layout(shapes=self.shapes, annotations=self.notes)
        return self.fig


def _runs(qs) -> list:
    """Sorted qubits split into runs of neighbors: [0, 1, 3] -> [[0, 1], [3]]."""
    qs = sorted(qs)
    runs = [[qs[0]]]
    for q in qs[1:]:
        if q == runs[-1][-1] + 1:
            runs[-1].append(q)
        else:
            runs.append([q])
    return runs


def _col_runs(cols) -> list:
    cols = sorted(set(cols))
    if not cols:
        return []
    runs, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c > prev + 1:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    return runs


def geometry(n: int, x_span: float, width_px: float, height_per_qubit: int) -> dict:
    """Pixel-aware sizes: the box half width keeps gates roughly square, fonts follow the box size."""
    px_col = max((width_px - 56) / max(x_span, 1.0), 6.0)
    px_row = float(height_per_qubit)
    bw = min(BOX_W, BOX_H * px_row / px_col)
    box_px = 2 * bw * px_col
    font = float(np.clip(0.42 * box_px, 7.0, 12.0))
    return dict(px_col=px_col, bw=bw, bh=BOX_H, box_px=box_px, font=font,
                show_param=box_px >= 29, sub_font=max(6.5, font - 3.5),
                dot=float(np.clip(0.30 * box_px, 5.5, 11.0)), target=float(np.clip(0.80 * box_px, 11.0, 24.0)),
                xmark=float(np.clip(0.46 * box_px, 7.0, 14.0)))


def _lighter(color: str, t: float = 0.45) -> str:
    if not (isinstance(color, str) and color.startswith("#") and len(color) == 7):
        return color
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(v + (255 - v) * t) for v in (r, g, b))


def draw(spec: CircuitSpec, *, title: str = "", dark: bool = False, window: int | None = None,
         width_px: float = 1150, cursor_op: int | None = None, measure_map: dict | None = None,
         height_per_qubit: int = 46, show_measure: bool = True, highlight_tags: bool = True,
         max_ops: int = MAX_OPS, positions: list | None = None, n_columns: int | None = None,
         groups: list | None = None, qubit_labels: list | None = None, sections: list | None = None) -> go.Figure:
    """Draw ``spec``.

    window     : the most columns shown at once (default: as many as fit ``width_px`` at about square cells);
                 wider circuits scroll sideways.
    width_px   : the width the figure is expected to get in the page (a hint for text and marker sizes).
    positions  : optional [(column, op index)] replacing the automatic layout (the compile diff aligns two
                 circuits this way); ``n_columns`` is then the total number of columns.
    groups     : optional frames behind columns: dicts with x0, x1 (columns), q0, q1 (rows), label, kind.
    sections   : optional [(first row, title)]: a separator above each section after the first (the compile diff
                 stacks the logical circuit over the physical one in one figure, so both scroll together).
    """
    import re as _re
    n = spec.n

    def rn(q):
        return _re.sub(r"<[^>]+>", "", qubit_labels[q]) if qubit_labels and q < len(qubit_labels) else "q%d" % q
    if positions is None:
        cols_all, ncol = columns(spec)
    else:
        cols_all, ncol = list(positions), int(n_columns if n_columns is not None else
                                                (max((c for c, _ in positions), default=-1) + 1))
    auto_window = max(6, int((width_px - 56) / (0.95 * height_per_qubit)))
    window = auto_window if window is None else max(4, min(int(window), auto_window))
    col_of = dict((i, c) for c, i in cols_all)
    cursor_col = col_of.get(cursor_op) if cursor_op is not None else None
    clip = None                                         # (first, last) column drawn when the circuit is very long
    if max_ops and len(cols_all) > max_ops:
        per_col = max(len(cols_all) / max(ncol, 1), 1.0)
        span = max(int(max_ops / per_col), window + 6)
        centre = cursor_col if cursor_col is not None else span // 2
        first = max(0, min(centre - span // 2, ncol - span))
        clip = (first, first + span - 1)
        cols = [(c, i) for c, i in cols_all if clip[0] <= c <= clip[1]]
    else:
        cols = cols_all
    show_measure = show_measure and (clip is None or clip[1] >= ncol - 1)
    xs_end = ncol + (1 if show_measure else 0)
    lo = -1 if clip is None else clip[0] - 1
    hi = (min(xs_end, window) if xs_end > window else xs_end) if clip is None else min(xs_end, lo + 1 + window)
    if xs_end <= window and clip is None:
        hi = max(xs_end, 3)
    G = geometry(n, hi - lo, width_px, height_per_qubit)
    bw, bh = G["bw"], G["bh"]
    muted = "#9AA3AB" if dark else T.MUTED
    fig = _Batch(go.Figure())
    w0 = -0.7 if clip is None else clip[0] - 0.7
    w1 = xs_end - 0.2 if clip is None else min(xs_end - 0.2, clip[1] + 0.7)
    wire = "#5B6770" if dark else "#A2A9B0"
    for q in range(n):
        fig.add_shape(type="line", x0=w0, x1=w1, y0=q, y1=q, line=dict(color=wire, width=1.3), layer="below")

    # ---- frames behind groups of columns (compile diff), then fold bands; both stay inside their columns
    top_pad = 0.0
    for gi, g in enumerate(groups or ()):
        if clip is not None and (g["x1"] < clip[0] or g["x0"] > clip[1]):
            continue
        kind = g.get("kind", "")
        fill = {"route": "rgba(183,121,31,0.13)", "prep": "rgba(130,130,130,0.07)"}.get(
            kind, "rgba(21,96,130,0.07)" if gi % 2 == 0 else "rgba(21,96,130,0.13)")
        edge = {"route": "rgba(183,121,31,0.85)"}.get(kind, "rgba(21,96,130,0.35)")
        fig.add_shape(type="rect", x0=g["x0"] - 0.46, x1=g["x1"] + 0.46, y0=g["q0"] - 0.47, y1=g["q1"] + 0.47,
                      fillcolor=fill, line=dict(color=edge, width=1, dash="dot" if kind != "route" else None),
                      layer="below")
        if g.get("label") and g.get("label_y") is not None:      # labels only where the caller made room
            fig.add_annotation(x=(g["x0"] + g["x1"]) / 2, y=g["label_y"], text=g["label"], showarrow=False,
                               yanchor="middle", borderpad=1, font=dict(size=max(8.0, G["font"] - 2),
                                                                         color=T.YELLOW if kind == "route" else muted),
                               bgcolor="rgba(14,17,23,0.9)" if dark else "rgba(255,255,255,0.92)")
    if highlight_tags:
        for tag in ("fold", "fold-inv"):
            runs = _col_runs(c for c, i in cols if spec.ops[i].tag == tag)
            for a, b in runs:
                fig.add_shape(type="rect", x0=a - 0.47, x1=b + 0.47, y0=-0.62, y1=n - 0.38, layer="below",
                              fillcolor=FOLD_FILL[tag], line=dict(width=0))
                if b - a >= 0 and G["px_col"] * (b - a + 1) >= 18:
                    fig.add_annotation(x=(a + b) / 2, y=-0.64, text=FOLD_LABEL[tag], showarrow=False, yanchor="bottom",
                                       font=dict(size=max(8.0, G["font"] - 1), color="#6929C4"))
                    top_pad = max(top_pad, 0.22)

    # ---- gates
    hx, hy, htxt = [], [], []                          # invisible hover targets
    tx, ty, tt, tcol, tsize = [], [], [], [], []       # labels
    mx, my, msym, msize, mfill, mline, mlw = [], [], [], [], [], [], []   # small markers (dots, ⊕, ×)

    def box(c, q0, q1, fill, edge, ew, dash=None):
        fig.add_shape(type="rect", x0=c - bw, x1=c + bw, y0=q0 - bh, y1=q1 + bh, fillcolor=fill,
                      line=dict(color=edge, width=ew, dash=dash), layer="below")     # below the label trace

    def outline(c, q0, q1, color):
        fig.add_shape(type="rect", x0=c - bw - 0.05, x1=c + bw + 0.05, y0=q0 - bh - 0.05, y1=q1 + bh + 0.05,
                      fillcolor="rgba(0,0,0,0)", line=dict(color=color, width=2), layer="above")

    def label(c, y, text, color, size):
        tx.append(c); ty.append(y); tt.append(text); tcol.append(color); tsize.append(size)

    for c, i in cols:
        o = spec.ops[i]
        if o.kind == "barrier":
            qs = o.qubits or tuple(range(n))
            fig.add_shape(type="rect", x0=c - 0.58, x1=c - 0.42, y0=min(qs) - 0.4, y1=max(qs) + 0.4,
                          fillcolor="rgba(162,169,176,0.35)", line=dict(width=0), layer="below")
            continue
        fc, tc = _fill(o)
        tag_col = T.TAG_OUTLINE.get(o.tag) if (highlight_tags and o.tag not in ("fold", "fold-inv")) else None
        par = ", ".join(fmt_angle_tex(p) if o.kind == "gate" else "%.3g" % p for p in o.params)
        hpar = ", ".join(fmt_angle_tex(p, 6) if o.kind == "gate" else "%.6g" % p for p in o.params)
        hover = "<b>%s</b>%s on %s%s" % (o.name, "(%s)" % hpar if hpar else "", ", ".join(rn(q) for q in o.qubits),
                                         ("<br>" + ("added by routing (a SWAP)" if o.tag == "route" else
                                                    o.tag if o.tag.startswith("part of") else "added by: " + o.tag))
                                         if o.tag else "")
        qs = sorted(o.qubits)
        if o.kind in ("noise", "box") or o.k == 1:
            edge = T.YELLOW if o.kind == "noise" else ("rgba(0,0,0,0.25)" if not dark else "rgba(255,255,255,0.3)")
            runs = _runs(qs)
            if len(runs) > 1:                           # one box per run of neighboring qubits; idle wires stay clear
                fig.add_shape(type="line", x0=c, x1=c, y0=runs[0][-1] + bh, y1=runs[-1][0] - bh, layer="below",
                              line=dict(color=edge, width=1.2, dash="dot"))
            for r in runs:
                box(c, r[0], r[-1], fc, edge, 1.1, "dot" if o.kind == "noise" else None)
                if tag_col:
                    outline(c, r[0], r[-1], tag_col)
                ym = (r[0] + r[-1]) / 2
                lab = o.label()
                if par and G["show_param"]:
                    lab = "%s<br><span style='font-size:%dpx'>%s</span>" % (lab, G["sub_font"], par)
                label(c, ym, lab, tc, G["font"] if len(o.label()) <= 3 else G["font"] - 1.5)
                hx.append(c); hy.append(ym); htxt.append(hover)
            continue
        # ---- two-qubit gates
        col = fc if not tag_col else tag_col
        if dark and not tag_col:                        # dots and lines on a dark page: a lighter shade
            col = _lighter(col)
        fig.add_shape(type="line", x0=c, x1=c, y0=qs[0], y1=qs[-1], line=dict(color=col, width=2), layer="below")
        if o.name == "cx":
            ctrl, tgt = o.qubits
            mx += [c, c]; my += [ctrl, tgt]; msym += ["circle", "circle-cross"]
            msize += [G["dot"], G["target"]]; mfill += [col, col]; mline += [col, "#FFFFFF"]; mlw += [1, 2]
        elif o.name == "swap":
            for q in qs:
                mx.append(c); my.append(q); msym.append("x-thin"); msize.append(G["xmark"]); mfill.append(col)
                mline.append(col); mlw.append(2.2)
        elif o.name == "ecr":
            for q in qs:
                box(c, q, q, fc, fc, 1)
                label(c, q, "ECR", tc, max(7.0, G["font"] - 2.5))
        else:                                           # cz, cp, rzz: dots
            for q in qs:
                mx.append(c); my.append(q); msym.append("circle"); msize.append(G["dot"]); mfill.append(col)
                mline.append(col); mlw.append(1)
        if tag_col:
            for r in _runs(qs):
                outline(c, r[0], r[-1], tag_col)
        if par and G["show_param"] and o.name in ("cp", "rzz"):
            fig.add_annotation(x=c, y=(qs[0] + qs[-1]) / 2 if qs[-1] - qs[0] > 0 else qs[0] - 0.5,
                               text=("ZZ " if o.name == "rzz" else "") + par, showarrow=False,
                               bgcolor="rgba(14,17,23,0.85)" if dark else "rgba(255,255,255,0.9)",
                               font=dict(size=G["sub_font"], color=muted), borderpad=1)
        for q in qs:
            hx.append(c); hy.append(q); htxt.append(hover)

    # ---- measurements
    if show_measure:
        inv = {v: k for k, v in (measure_map or {q: q for q in range(n)}).items()}
        for q in range(n):
            if sections and measure_map is not None and q not in inv:     # stacked drawings: unmeasured extras
                continue
            fig.add_shape(type="rect", x0=ncol - bw, x1=ncol + bw, y0=q - bh, y1=q + bh, fillcolor="#A2A9B0",
                          line=dict(color="#697077", width=1), layer="below")
            label(ncol, q, "M", "#161616", G["font"])
            hx.append(ncol); hy.append(q)
            htxt.append("measure %s → classical bit %s" % (rn(q), inv.get(q, "–")))

    if mx:
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(symbol=msym, size=msize, color=mfill, line=dict(color=mline, width=mlw))))
    if tx:
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="text", text=tt, hoverinfo="skip", showlegend=False,
                                 textfont=dict(color=tcol, size=tsize,
                                               family="IBM Plex Sans, Helvetica, Arial, sans-serif")))
    if hx:
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", hovertext=htxt, hoverinfo="text", showlegend=False,
                                 marker=dict(size=max(G["box_px"], 8), color="rgba(0,0,0,0)", symbol="square")))
    if cursor_col is not None and (clip is None or clip[0] <= cursor_col <= clip[1]):
        _cursor(fig, spec.ops[cursor_op], cursor_col, n, dark, bw, bh)
    ticks = qubit_labels or ["q<sub>%d</sub>" % q for q in range(n)]
    fig.update_layout(template=T.tpl(dark), height=max(150, 60 + height_per_qubit * n + int(40 * top_pad)),
                      title=title or None,
                      margin=dict(l=44 if not qubit_labels else 78, r=10, t=36 if title else 8,
                                  b=8 + (16 if xs_end > window or clip is not None else 0)),
                      xaxis=dict(visible=False, range=[lo, hi]),
                      yaxis=dict(range=[n - 0.4, -0.75 - top_pad], tickmode="array", tickvals=list(range(n)),
                                 ticktext=ticks, showline=False, ticks="", showgrid=False, fixedrange=True),
                      dragmode="pan", hovermode="closest")
    for k, (row, name) in enumerate(sections or ()):
        if k > 0:
            fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=row - 0.5, y1=row - 0.5, layer="above",
                          line=dict(color=muted, width=1.5, dash="dash"))
    if sections:                                        # titles in the bottom margin, clear of every gate
        fig.add_annotation(xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="top", showarrow=False,
                           yshift=-2 - (15 if xs_end > window or clip is not None else 0), text="  ·  ".join("%s %s" % ("▲" if k == 0 else "▼", nm)
                                                        for k, (_, nm) in enumerate(sections)),
                           font=dict(size=10, color=muted))
        fig.update_layout(margin=dict(b=fig.layout.margin.b + 18 + (14 if xs_end > window or clip is not None
                                                                       else 0)))
    if clip is not None:
        fig.add_annotation(xref="paper", yref="paper", x=1, y=0, yanchor="top", xanchor="right", showarrow=False,
                           text="%d operations: drawing columns %d–%d of %d around the cursor · drag to scroll ⇆"
                                % (len(cols_all), clip[0], clip[1], ncol), font=dict(size=10, color=muted))
    elif xs_end > window:
        fig.add_annotation(xref="paper", yref="paper", x=1, y=0, yanchor="top", xanchor="right", showarrow=False,
                           text="%d of %d columns shown · drag sideways to scroll ⇆" % (window, xs_end),
                           font=dict(size=10, color=muted))
    if cursor_col is not None and (cursor_col >= lo + window - 2 or cursor_col < lo + 1):
        fig.update_xaxes(range=[cursor_col - window + 3, cursor_col + 3])
    return fig.done()


def _cursor(fig, o, c: int, n: int, dark: bool, bw: float = BOX_W, bh: float = BOX_H):
    """Outline the executed operation only: a box around the gate on the qubits it acts on (one box per run of
    neighboring qubits) and, for a gate on distant qubits, a band along its connector. Wires the gate does not
    touch stay unmarked. The box sits just outside any method outline and inside the gate's own column."""
    qs = sorted(o.qubits) if o.qubits else list(range(n))
    fill = "rgba(21,96,130,0.16)" if not dark else "rgba(120,190,230,0.22)"
    band = "rgba(21,96,130,0.40)" if not dark else "rgba(120,190,230,0.45)"
    edge = T.BLUE if not dark else "#78BEE6"
    runs = _runs(qs)
    if len(runs) > 1:                                  # the connector of this gate (not the idle wires it crosses)
        fig.add_shape(type="line", x0=c, x1=c, y0=qs[0], y1=qs[-1], layer="below", line=dict(color=band, width=10))
    dx, dy = min(bw + 0.10, 0.47), bh + 0.08
    for r in runs:
        fig.add_shape(type="rect", x0=c - dx, x1=c + dx, y0=r[0] - dy, y1=r[-1] + dy, layer="below",
                      fillcolor=fill, line=dict(width=0))
        for q in r:                                    # the active wire segments
            fig.add_shape(type="line", x0=c - dx, x1=c + dx, y0=q, y1=q, layer="below", line=dict(color=edge, width=3))
        fig.add_shape(type="rect", x0=c - dx, x1=c + dx, y0=r[0] - dy, y1=r[-1] + dy, layer="above",
                      line=dict(color=edge, width=2.5), fillcolor="rgba(0,0,0,0)")
    fig.add_annotation(x=c, y=qs[0] - dy - 0.04, text="▼", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=edge))
