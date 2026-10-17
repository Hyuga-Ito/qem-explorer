"""Theory & sources: the physics and mathematics behind every part of the app, with small calculators."""
import math
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Theory & sources · QEM Explorer", page_icon=":material/menu_book:", layout="wide")

from qemx.methods.zne import linear_weights, richardson_weights  # noqa: E402
from ui import state as S  # noqa: E402
from ui.nav import sidebar_nav  # noqa: E402
from ui import figures as FG  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar_nav()
DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "theory.md")


@st.cache_data(show_spinner=False)
def sections(mtime: float) -> list:
    text = open(DOC, encoding="utf-8").read()
    parts = re.split(r"(?m)^## ", text)
    out = []
    for p in parts[1:]:
        title, _, body = p.partition("\n")
        out.append((title.strip(), body.strip()))
    return out


@st.cache_data(show_spinner=False)
def preface(mtime: float) -> str:
    """The handbook's introduction (the text before the first section, without its title)."""
    head = re.split(r"(?m)^## ", open(DOC, encoding="utf-8").read())[0]
    return re.sub(r"(?m)^# .*$", "", head).strip()


PARTS = [("Part I · Foundations", (1, 2, 3)), ("Part II · The methods in the app", (4, 5, 6, 7, 8, 9, 10, 11)),
         ("Part III · The machinery", (12, 13, 14, 15)), ("Part IV · Advanced mitigation", (16,)),
         ("Part V · Toward fault tolerance", (17,)),
         ("Part VI · Research methods: budgets, phase diagrams, device divergence", (18, 19, 20, 21)),
         ("Sources", (None,))]
READING_CSS = """<style>
.st-key-th_body{max-width:54rem}
.st-key-th_body p,.st-key-th_body li{font-size:1.06rem;line-height:1.8}
.st-key-th_body p{margin:0 0 1.1rem 0}
.st-key-th_body li{margin-bottom:.35rem}
.st-key-th_body h2{margin-top:3rem;padding-bottom:.35rem;letter-spacing:-.01em}
.st-key-th_body h3{margin-top:2.3rem;margin-bottom:.6rem;font-size:1.42rem}
.st-key-th_body h4{margin-top:1.7rem;margin-bottom:.45rem;font-size:1.16rem}
.st-key-th_body .katex-display{margin:1.2rem 0 1.35rem 0;overflow-x:auto;overflow-y:hidden}
.st-key-th_body table{font-size:.97rem;margin:.6rem 0 1.4rem 0}
.st-key-th_body th,.st-key-th_body td{padding:.55rem .85rem !important;line-height:1.55}
.st-key-th_body .qx-part{font-size:.78rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  opacity:.62;margin:3.2rem 0 -1.6rem 0}
.st-key-th_toc p{font-size:.95rem;line-height:1.55;margin-bottom:.25rem}
.st-key-th_toc .qx-toc-part{font-size:.72rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  opacity:.6;margin:1.1rem 0 .3rem 0}
</style>"""


def section_number(title: str):
    m = re.match(r"(\d+)\.", title)
    return int(m.group(1)) if m else None


def part_of(title: str) -> str:
    n = section_number(title)
    return next((name for name, nums in PARTS if n in nums), PARTS[-1][0])


def anchor(title: str) -> str:
    a = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    return re.sub(r"[\s]+", "-", a)


secs = sections(os.path.getmtime(DOC))
st.title("Theory & sources")
st.caption("A research handbook: foundations, every method in the app, advanced mitigation, the road to fault "
           "tolerance, and the research methods behind the thesis studies — equal-budget phase diagrams and how an "
           "IBM device departs from its component-level model — with verified references.")
with st.expander("How this handbook is organized", icon=":material/menu_book:", expanded=True, key="th_preface"):
    st.markdown(preface(os.path.getmtime(DOC)))

with st.expander("Calculators", icon=":material/calculate:", expanded=False, key="th_calc"):
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("**PEC overhead of one gate**")
        S.wdefault("w_th_p", 0.01)
        S.wdefault("w_th_k", 2)
        p = st.number_input("depolarizing p", min_value=0.0, max_value=0.999, step=0.001, format="%.4f", key="w_th_p",
                            persist_state=P)
        k = st.number_input("qubits k", min_value=1, step=1, key="w_th_k", persist_state=P)
        f = 1 - p
        if int(k) <= 26:                              # exact while d² fits comfortably in a float
            d2 = float(4 ** int(k))
            g = (2 * (d2 - 1) / f - (d2 - 2)) / d2
            st.latex(r"\gamma=\frac{2(d^2-1)/f-(d^2-2)}{d^2}=%.5f" % g)
        else:                                         # d → ∞ limit
            g = 2 / f - 1
            st.latex(r"\gamma\;\xrightarrow{d\to\infty}\;\frac{2}{f}-1=%.5f" % g)
        log10_G2 = 200 * math.log10(g) if g > 0 else 0.0
        if log10_G2 < 15:
            st.caption("Γ² for 100 such gates: %.3g× the shots." % 10 ** log10_G2)
        elif log10_G2 < 1e6:
            st.caption("Γ² for 100 such gates: about 10^%.1f× the shots." % log10_G2)
        else:
            st.caption("Γ² for 100 such gates: effectively ∞.")
    with c2:
        st.markdown("**ZNE weights**")
        S.wdefault("w_th_l", "1, 3, 5")
        txt = st.text_input("noise scales λ", key="w_th_l", persist_state=P, max_chars=100)
        try:
            lams = W.parse_floats(txt)                      # finite decimal numbers only
            if len(lams) < 2 or len(set(lams)) < len(lams) or not all(math.isfinite(x) and abs(x) < 1e6 for x in lams):
                raise ValueError
            eta_r = richardson_weights(lams)
            eta_l = linear_weights(lams)
            st.dataframe(pd.DataFrame({"λ": lams, "η Richardson": eta_r, "η linear": eta_l}), hide_index=True,
                         width="stretch", column_config={"η Richardson": st.column_config.NumberColumn(format="%+.4f"),
                                                         "η linear": st.column_config.NumberColumn(format="%+.4f")})
            st.caption("Variance factor (Σ|η|)²: Richardson %.3f, linear %.3f." % (np.abs(eta_r).sum() ** 2,
                                                                                    np.abs(eta_l).sum() ** 2))
        except ValueError:
            st.error("Enter two or more distinct numbers separated by commas.")
    with c3:
        st.markdown("**Shots for a target precision**")
        for kk, vv in (("w_th_b", 0.01), ("w_th_s", 1.0), ("w_th_e", 0.02)):
            S.wdefault(kk, vv)
        b = st.number_input("bias b", step=0.001, format="%.4f", key="w_th_b", persist_state=P)
        s = st.number_input("per-shot σ (incl. overhead γ)", min_value=0.0, step=0.1, key="w_th_s", persist_state=P)
        e = st.number_input("target RMSE ε", min_value=1e-9, step=0.001, format="%.4f", key="w_th_e", persist_state=P)
        if e <= abs(b):
            st.latex(r"N_\varepsilon=\infty\quad(\varepsilon\le|b|)")
        else:
            try:
                ne = s ** 2 / (e ** 2 - b ** 2)
                txt_n = f"{math.ceil(ne):,}" if ne < 1e15 else "%.3g" % ne
            except (OverflowError, ValueError, ZeroDivisionError):
                txt_n = r"\infty"
            st.latex(r"N_\varepsilon=\frac{\sigma^2}{\varepsilon^2-b^2}=%s" % txt_n)

st.html(READING_CSS)
left, right = st.columns([1, 3.4], gap="large")
with left:
    S.wdefault("w_th_q", "")
    q = st.text_input("Search the theory", key="w_th_q", persist_state=P, max_chars=80, placeholder="e.g. Richardson")
    with st.container(key="th_toc"):
        st.markdown("**Contents**")
        last = None
        for title, body in secs:
            if part_of(title) != last:
                last = part_of(title)
                st.markdown("<div class='qx-toc-part'>%s</div>" % W.md_escape(last), unsafe_allow_html=True)
            hit = not q or q.lower() in (title + FG.strip(body)).lower()
            st.markdown(("[%s](#%s)" if hit else "<span style='opacity:0.4'>%s</span>") % (
                (title, anchor(title)) if hit else (title,)), unsafe_allow_html=not hit)
with right:
    with st.container(key="th_body"):
        shown, last = 0, None
        for title, body in secs:
            if q and q.lower() not in (title + FG.strip(body)).lower():
                continue
            shown += 1
            if part_of(title) != last:
                last = part_of(title)
                st.markdown("<div class='qx-part'>%s</div>" % W.md_escape(last), unsafe_allow_html=True)
            st.header(title, anchor=anchor(title), divider="gray")
            FG.markdown(body)
        if not shown:
            st.info("No section mentions “%s”." % W.md_escape(q))
