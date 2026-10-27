"""Device and coupling-map graphs: where the circuit's qubits sit, which couplers the routing used, how good each
qubit and coupler is, and which physical qubits a noise-aware layout would pick.

Coordinates: IBM's heavy-hex processors (127/133/156 qubits) are laid out row by row — long chains of consecutive
qubit numbers, joined by short bridge chains — which reproduces the familiar lattice picture. Smaller devices use
Qiskit's own coordinate tables when it has one; everything else (line, ring, grid, all-to-all) gets a simple
geometric layout or a seeded spring layout.
"""
from __future__ import annotations

import ast
import inspect
import math
import re
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

from . import theme as T
from ..resources import qiskit_serial


@lru_cache(maxsize=1)
@qiskit_serial
def _qiskit_maps() -> dict:
    """Qiskit's hand-made gate-map coordinates (read from its source; empty if the format changes)."""
    try:
        import qiskit.visualization.gate_map as gm
        src = inspect.getsource(gm.plot_gate_map)
        out = {}
        for m in re.finditer(r"qubit_coordinates_map\[(\d+)\]\s*=\s*(\[.*?\]\])", src, re.S):
            out[int(m.group(1))] = [tuple(p) for p in ast.literal_eval(m.group(2))]
        return out
    except Exception:
        return {}


def _adjacency(n: int, edges) -> dict:
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        if a != b and 0 <= a < n and 0 <= b < n:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _rows_layout(n: int, adj: dict, min_len: int = 6):
    """Heavy-hex style: long runs i, i+1, … of connected qubits are rows; bridges sit between them."""
    rows, cur = [], [0]
    for i in range(1, n):
        if i in adj[i - 1]:
            cur.append(i)
        else:
            rows.append(cur)
            cur = [i]
    rows.append(cur)
    long = [r for r in rows if len(r) >= min_len]
    if len(long) < 2 or sum(len(r) for r in long) < 0.6 * n:
        return None
    pos = {}
    for q in long[0]:
        pos[q] = (q - long[0][0], 0.0)
    for k in range(1, len(long)):
        prev, row = long[k - 1], long[k]
        yp, y = 2.0 * (k - 1), 2.0 * k
        offset = None
        for b in range(prev[-1] + 1, row[0]):              # bridge qubits between the two rows
            up = [q for q in adj[b] if q in prev]
            dn = [q for q in adj[b] if q in row]
            if up and dn:
                x = pos[up[0]][0]
                pos[b] = (x, yp + 1)
                if offset is None:
                    offset = x - (dn[0] - row[0])
        offset = 0.0 if offset is None else offset
        for q in row:
            pos[q] = (offset + q - row[0], y)
        for b in range(prev[-1] + 1, row[0]):               # bridges placed after the offset is known
            if b not in pos:
                dn = [q for q in adj[b] if q in row]
                pos[b] = (pos[dn[0]][0], y - 1) if dn else (0.0, y - 1)
    for q in range(n):                                      # anything left (edge bridges): next to a neighbor
        if q not in pos:
            nb = [x for x in adj[q] if x in pos]
            pos[q] = (pos[nb[0]][0], pos[nb[0]][1] + 1) if nb else (float(q), -2.0)
    return pos


def coordinates(n: int, edges, kind: str = "") -> dict:
    """{qubit: (x, y)} with y growing downwards (row order)."""
    adj = _adjacency(n, edges)
    if kind == "line" or (edges and all(abs(a - b) == 1 for a, b in edges) and len(edges) == n - 1):
        return {q: (float(q), 0.0) for q in range(n)}
    if kind in ("ring", "all"):
        return {q: (math.cos(2 * math.pi * q / n - math.pi / 2), math.sin(2 * math.pi * q / n - math.pi / 2))
                for q in range(n)}
    if kind == "grid":
        w = max((abs(b - a) for a, b in edges), default=int(math.ceil(math.sqrt(n)))) or 1
        return {q: (float(q % w), float(q // w)) for q in range(n)}
    rows = _rows_layout(n, adj, 5 if kind == "heavy-hex" else 6)
    if rows is not None:
        return rows
    maps = _qiskit_maps()
    if n in maps and len(maps[n]) == n:
        return {q: (float(maps[n][q][1]), float(maps[n][q][0])) for q in range(n)}
    try:
        import rustworkx as rx
        g = rx.PyGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from_no_data([(a, b) for a, b in edges])
        p = rx.spring_layout(g, seed=3, num_iter=400)
        return {q: (float(p[q][0]), float(p[q][1])) for q in range(n)}
    except Exception:
        return {q: (math.cos(2 * math.pi * q / n), math.sin(2 * math.pi * q / n)) for q in range(n)}


def _label_side(q: int, pos: dict, adj: dict) -> str:
    """Put a label on the side of the node with no coupler (above, below, right or left)."""
    x, y = pos[q]
    dirs = {"top center": 0, "bottom center": 0, "middle right": 0, "middle left": 0}
    for b in adj.get(q, ()):
        dx, dy = pos[b][0] - x, pos[b][1] - y
        if abs(dy) >= abs(dx):
            dirs["top center" if dy < 0 else "bottom center"] += 1      # y grows downwards
        else:
            dirs["middle right" if dx > 0 else "middle left"] += 1
    return min(dirs, key=lambda k: (dirs[k], list(dirs).index(k)))


def _seq_color(v: float, vmax: float) -> str:
    """One hue, light → dark (the house blue): low error light, high error dark."""
    t = 0.0 if not vmax else float(np.clip(v / vmax, 0, 1))
    lo, hi = np.array([214, 232, 242]), np.array([11, 61, 86])
    c = lo + (hi - lo) * t
    return "rgb(%d,%d,%d)" % tuple(c)


def figure(n: int, edges, *, kind: str = "", used: dict | None = None, swap_edges=(), node_err: dict | None = None,
           edge_err: dict | None = None, broken_nodes=(), broken_edges=(), candidate: list | None = None,
           node_hover: dict | None = None, title: str = "", dark: bool = False, height: int | None = None,
           node_metric: tuple = ("readout<br>error", ".1%")) -> go.Figure:
    """The coupling graph.

    used        {physical qubit: label} for the circuit's qubits (e.g. "q0"), drawn with a ring and the label
    swap_edges  couplers where routing inserted SWAPs (amber)
    node_err    {physical qubit: error} colors nodes (e.g. readout error); edge_err colors couplers
    candidate   physical qubits of a proposed layout (dashed ring)
    """
    pos = coordinates(n, edges, kind)
    adj = _adjacency(n, edges)
    ink = T.ink(dark)
    used = used or {}
    fig = go.Figure()
    emax = max(edge_err.values()) if edge_err else 0.0
    broken_e = {tuple(sorted(e)) for e in broken_edges}
    swap_e = {tuple(sorted(e)) for e in swap_edges}
    used_set = set(used)
    # couplers: batch by style
    styles = {}
    for a, b in {tuple(sorted(e)) for e in edges}:
        if a not in pos or b not in pos:
            continue
        if (a, b) in swap_e:
            key = ("swap", T.YELLOW, 6)
        elif (a, b) in broken_e:
            key = ("broken", "rgba(160,160,160,0.6)", 1.5)
        elif a in used_set and b in used_set:
            key = ("used", T.BLUE, 4)
        elif edge_err:
            e = edge_err.get((a, b), edge_err.get((b, a), 0.0))
            key = ("err%d" % int(8 * (e / emax if emax else 0)), _seq_color(e, emax), 2.2)
        else:
            key = ("plain", "#A2A9B0" if not dark else "#5B6770", 2)
        xs, ys = styles.setdefault(key, ([], []))
        xs += [pos[a][0], pos[b][0], None]
        ys += [pos[a][1], pos[b][1], None]
    names = {"swap": "routing SWAP on this coupler", "used": "coupler used by the circuit", "broken": "broken coupler"}
    for (kname, color, w), (xs, ys) in sorted(styles.items(), key=lambda kv: kv[0][0] in ("used", "swap")):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=w, dash="dot" if kname ==
                                                                     "broken" else None),
                                 hoverinfo="skip", showlegend=kname in names, name=names.get(kname, "")))
    # qubits
    nmax = max(node_err.values()) if node_err else 0.0
    xs = [pos[q][0] for q in range(n)]
    ys = [pos[q][1] for q in range(n)]
    cols = [("rgba(200,200,200,0.5)" if q in set(broken_nodes) else
             (_seq_color(node_err.get(q, 0.0), nmax) if node_err else ("#DDE1E6" if not dark else "#3A4148")))
            for q in range(n)]
    hover = [node_hover.get(q, "p%d" % q) if node_hover else "physical qubit %d" % q for q in range(n)]
    size = 11 if n > 60 else 16 if n > 20 else 26
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=size, color=cols,
                                                                      line=dict(color="#697077", width=0.8)),
                             hovertext=hover, hoverinfo="text", showlegend=False))
    if candidate:
        cx = [pos[q][0] for q in candidate if q in pos]
        cy = [pos[q][1] for q in candidate if q in pos]
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", name="lowest-error layout found",
                                 marker=dict(size=size + 12, color="rgba(0,0,0,0)", symbol="circle",
                                             line=dict(color=T.FAMILY["learning"], width=2.5, dash="dot")),
                                 hoverinfo="skip"))
    if used:
        ux = [pos[q][0] for q in used if q in pos]
        uy = [pos[q][1] for q in used if q in pos]
        fig.add_trace(go.Scatter(x=ux, y=uy, mode="markers+text", name="the circuit's qubits",
                                 text=[used[q] for q in used if q in pos],
                                 textposition=[_label_side(q, pos, adj) for q in used if q in pos],
                                 textfont=dict(color=ink, size=11 if n > 60 else 12),
                                 marker=dict(size=size + 6, color="rgba(0,0,0,0)",
                                             line=dict(color=T.BLUE if not dark else "#78BEE6", width=3)),
                                 hovertext=["p%d holds %s" % (q, used[q]) for q in used if q in pos],
                                 hoverinfo="text"))
    if n <= 60:                                              # physical numbers on small graphs
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="text", text=[str(q) for q in range(n)], hoverinfo="skip",
                                 textfont=dict(size=9 if n > 20 else 11, color="#161616"), showlegend=False))
    if node_err:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale=[[0, _seq_color(0, 1)], [1, _seq_color(1, 1)]], cmin=0,
                                             cmax=nmax or 1, color=[0], showscale=True,
                                             colorbar=dict(title=dict(text=node_metric[0], side="top"),
                                                           thickness=10, len=0.6, tickformat=node_metric[1]))))
    span_x = (max(xs) - min(xs)) if xs else 1
    span_y = (max(ys) - min(ys)) if ys else 1
    h = height or int(min(620, max(260, 110 + 520 * (span_y + 1) / max(span_x + 1, 1))))
    fig.update_layout(template=T.tpl(dark), height=h, title=title or None, margin=dict(l=10, r=10, t=36 if title
                                                                                       else 10, b=10),
                      xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                      yaxis=dict(visible=False, autorange="reversed"),
                      legend=dict(orientation="h", x=0, y=-0.02, yanchor="top"), hovermode="closest", dragmode="pan")
    return fig
