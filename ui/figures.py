"""Algorithm diagrams (quantikz → SVG, built by scripts/make_figures.py) inside Markdown documents.

A document marks a figure with an HTML comment on its own line, ``<!-- fig:grover -->``. ``markdown(md)`` renders
the text around it with st.markdown and each figure as a centered image on a light card, so the black-on-white
diagram reads the same in the light and the dark theme. A figure's caption (docs/figures/captions.json, written by
the build script) is typeset by the app under the card, at text size.
"""
from __future__ import annotations

import base64
import json
import os
import re

import streamlit as st

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
MARK = re.compile(r"(?m)^<!--[ \t]*fig:([a-z0-9_]+)[ \t]*-->[ \t]*$")
CSS = """<style>
.qxfig{background:#ffffff;border:1px solid rgba(11,26,34,.10);border-radius:12px;padding:16px 14px;
  margin:.6rem 0 .35rem 0;text-align:center;overflow-x:auto}
.qxfig img{height:auto;display:inline-block;object-fit:contain !important}
</style>"""
# Diagrams are sized by their text, not by the column: the TeX source uses 10 pt type, and SCALE makes it render at
# about the size of the body text (≈ 18–19 px). A diagram wider than the column shrinks to fit, but never below
# MIN_SCALE — beyond that the card scrolls sideways instead of making the labels unreadably small. (Streamlit's
# Markdown styles give images object-fit: scale-down, which never enlarges a picture; the card overrides it.)
SCALE = 1.9
MIN_SCALE = 1.35


@st.cache_data(show_spinner=False)
def _captions(mtime: float) -> dict:
    try:
        return json.load(open(os.path.join(FIG_DIR, "captions.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def caption(name: str) -> str:
    """The diagram's caption (Markdown with $…$ math), typeset by the app under the picture; '' if none."""
    path = os.path.join(FIG_DIR, "captions.json")
    return _captions(os.path.getmtime(path) if os.path.exists(path) else 0.0).get(name, "")


@st.cache_data(show_spinner=False)
def _svg(name: str, mtime: float) -> tuple:
    """(data URI, natural width in px) of a figure; ('', 0) if it is missing."""
    path = os.path.join(FIG_DIR, name + ".svg")
    try:
        raw = open(path, "rb").read()
    except OSError:
        return "", 0
    m = re.search(rb'width="([\d.]+)(pt|px)?"', raw[:600])
    w = float(m.group(1)) * (4 / 3 if (m and m.group(2) == b"pt") else 1) if m else 600.0
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode(), w


def names() -> list:
    try:
        return sorted(f[:-4] for f in os.listdir(FIG_DIR) if f.endswith(".svg"))
    except OSError:
        return []


def figure(name: str, scale: float = SCALE) -> None:
    path = os.path.join(FIG_DIR, name + ".svg")
    uri, w = _svg(name, os.path.getmtime(path) if os.path.exists(path) else 0.0)
    if not uri:
        return
    width, floor = int(round(w * scale)), int(round(w * MIN_SCALE))
    st.markdown(CSS + '<div class="qxfig"><img src="%s" alt="%s diagram" style="width:%dpx;'
                'max-width:max(100%%, %dpx);object-fit:contain"/></div>' % (uri, name, width, floor),
                unsafe_allow_html=True)
    cap = caption(name)
    if cap:
        st.caption(cap)


def split(md: str) -> list:
    """[('text', str) | ('fig', name)] in document order."""
    out, pos = [], 0
    for m in MARK.finditer(md):
        if md[pos:m.start()].strip():
            out.append(("text", md[pos:m.start()]))
        out.append(("fig", m.group(1)))
        pos = m.end()
    if md[pos:].strip():
        out.append(("text", md[pos:]))
    return out


def markdown(md: str) -> None:
    """st.markdown with the document's figures in place."""
    for kind, x in split(md):
        if kind == "fig":
            figure(x)
        else:
            st.markdown(x)


def strip(md: str) -> str:
    """The text without figure markers (for search and plain exports)."""
    return MARK.sub("", md)
