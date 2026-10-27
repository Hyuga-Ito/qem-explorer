"""One visual system for every figure.

Method families have fixed hues, checked with a color-vision-deficiency validator on both the
light (#ffffff) and the dark (#0e1117) surface; every family also has its own marker symbol, and
every chart labels methods in text, so color is never the only cue. Red is reserved for a
negative sign, house blue for a positive one; gray is the unmitigated reference and ink the ideal.
Gate colors in circuit drawings follow IBM Quantum Composer's conventions.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

INK = "#000000"
INK_DARK = "#F2F2F2"
RED = "#B91C1C"
BLUE = "#156082"
YELLOW = "#A16207"
GREY = "#8A8A8A"
MUTED = "#5E5E5E"
GRID = "#E6E6E6"

FAMILY = {"raw": GREY, "zne": "#1D6FA3", "pec": "#B7791F", "learning": "#0F9D8A", "vd": "#8B5CF6",
          "symmetry": "#C2185B", "readout": "#5B6B00"}
SYMBOL = {"raw": "circle-open", "zne": "square", "pec": "diamond", "learning": "triangle-up", "vd": "hexagon",
          "symmetry": "star", "readout": "cross"}

# IBM-Quantum-Composer-like gate colors (fill, text)
GATE_COLOR = {
    "h": ("#FA4D56", "#FFFFFF"),
    "pauli": ("#002D9C", "#FFFFFF"),
    "phase": ("#33B1FF", "#000000"),
    "pulse": ("#9F1853", "#FFFFFF"),
    "rot": ("#9F1853", "#FFFFFF"),
    "two": ("#002D9C", "#FFFFFF"),
    "noise": ("#FCF4D6", "#7A4E00"),
    "box": ("#DDE1E6", "#161616"),
}
TAG_OUTLINE = {"pec": FAMILY["pec"], "twirl": FAMILY["learning"], "dd": FAMILY["vd"], "fold-inv": "#6929C4",
               "train": FAMILY["learning"], "nil": FAMILY["learning"], "check": FAMILY["symmetry"]}

FONT = "STIX Two Text, Times New Roman, Times, serif"


def template(dark: bool = False) -> go.layout.Template:
    bg = "#0E1117" if dark else "#FFFFFF"
    ink = INK_DARK if dark else INK
    grid = "#262730" if dark else GRID
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=14, color=ink), paper_bgcolor=bg, plot_bgcolor=bg,
        colorway=[FAMILY[k] for k in ("zne", "pec", "learning", "vd", "symmetry", "readout")],
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(showline=True, linecolor=ink, ticks="outside", tickcolor=ink, gridcolor=grid, zeroline=False),
        yaxis=dict(showline=True, linecolor=ink, ticks="outside", tickcolor=ink, gridcolor=grid, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
    )
    return t


pio.templates["qx"] = template(False)
pio.templates["qx_dark"] = template(True)


def tpl(dark: bool) -> str:
    return "qx_dark" if dark else "qx"


def ink(dark: bool) -> str:
    return INK_DARK if dark else INK


PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "svg", "scale": 1},
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d"]}
