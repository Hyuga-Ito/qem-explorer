"""Head-to-head sweeps: any two or more methods across a noise parameter, and who wins where."""
import math

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Head-to-head · QEM Explorer", page_icon=":material/compare_arrows:", layout="wide")

from qemx import analysis as A  # noqa: E402
from qemx import device  # noqa: E402
from qemx.bundle import make_zip  # noqa: E402
from qemx.recipes import PRESETS as RECIPE_PRESETS, preset  # noqa: E402
from qemx.viz import analysis_charts as AC  # noqa: E402
from ui import common as C  # noqa: E402
from ui import state as S  # noqa: E402
from ui import metrics as MX  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar(run_label=None)
st.title("Head-to-head sweeps")
st.markdown("Pick two or more contenders and a noise knob. One simulation per setting gives each method's bias and "
            "variance, so the winner can be mapped over every shot budget at once.")
compiled, err = C.compiled_now()
C.context_line(compiled)
if err:
    st.error("The circuit does not compile: %s" % err)
backend = C.sim_backend_for_sweeps()
if device.is_device(ss.backend):
    st.info("Sweeps vary the noise injector's parameters, so they run on the exact NumPy simulator with the injector's "
            "model (a fake device's calibration is fixed).", icon=":material/info:")

custom = W.custom_options(ss.custom_recipes)             # {"custom:<Recipe.key()>": Recipe}: identity, not title
custom_label = W.custom_labels(custom)
options = list(RECIPE_PRESETS) + list(custom)
for k, v in (("w_hh_rc", ["Unmitigated", "ZNE · Richardson", "PEC · inverse", "CDR · OLS"]), ("w_hh_param", "p2"),
             ("w_hh_min", 0.0), ("w_hh_max", 0.04), ("w_hh_pts", 7), ("w_hh_metric", "RMSE"), ("w_hh_N", 10000)):
    S.wdefault(k, v)
ss.w_hh_rc = [x for x in ss.w_hh_rc if x in options]
c = st.columns([2.4, 1.4])
c[0].multiselect("Contenders (two or more)", options, key="w_hh_rc", persist_state=P,
                 format_func=lambda k: custom_label.get(k, k),
                 help="Presets, plus any custom contenders built in the Playground.")
c[1].selectbox("Sweep", list(A.SWEEPS), key="w_hh_param", format_func=lambda k: A.SWEEPS[k][0], persist_state=P)
c = st.columns(4, vertical_alignment="bottom")
lo = c[0].number_input("from", step=0.001, format="%.4f", key="w_hh_min", persist_state=P)
hi = c[1].number_input("to", step=0.001, format="%.4f", key="w_hh_max", persist_state=P)
pts = c[2].number_input("points", min_value=2, step=1, key="w_hh_pts", persist_state=P)


def req():
    ss.hh_run = True


c[3].button("Run sweep", type="primary", icon=":material/play_arrow:", key="hh_btn", on_click=req,
            disabled=compiled is None or len(ss.w_hh_rc) < 2)
if len(ss.w_hh_rc) < 2:
    st.caption("Choose at least two contenders.")
def hh_sig():
    return S.signature(extra=("hh", tuple(ss.w_hh_rc), ss.w_hh_param, float(lo), float(hi), int(pts)))


if ss.pop("hh_run", False) and compiled is not None and len(ss.w_hh_rc) >= 2:
    recipes = [custom[t] if t in custom else preset(t, ss.symmetry) for t in ss.w_hh_rc]
    bar = st.progress(0.0, text="Simulating each setting…")

    def sweep_():                    # the grid and the NumPy compilation (device backends) run inside the guard
        vals_ = np.linspace(float(lo), float(hi), int(pts))
        if ss.w_hh_param != "scale":
            vals_ = np.clip(vals_, 0.0, None)
        comp = compiled
        if device.is_device(ss.backend):
            import dataclasses
            from qemx import compile as CP
            comp = CP.build(S.spec(), ss.inputs, ss.basis, dataclasses.replace(S.compile_options(), device=None,
                                                                                initial_layout=None), ss.relabel)
        return A.sweep(comp, ss.noise, ss.observable, recipes, ss.w_hh_param, vals_, backend, int(ss.shots),
                       ss.symmetry, lambda f: bar.progress(f, text="Simulating each setting…"))
    ok, rows = W.guarded("The sweep", sweep_, n=int(ss.n))
    bar.empty()
    if ok:
        ss.hh_result = dict(rows=rows, param=ss.w_hh_param, sig=hh_sig())

res = ss.get("hh_result")
figs, tables = {}, {}
if res is None:
    st.info("Press **Run sweep**.")
else:
    rows = res["rows"]
    if res.get("sig") != hh_sig():
        C.stale_note("Run sweep")
    xlabel = A.SWEEPS[res["param"]][0]
    fails = sorted({(r["method"], r["error"]) for r in rows if r.get("error")})
    for m, e in fails[:5]:
        st.warning("%s failed at some settings: %s" % (W.md_escape(m), W.md_escape(e[:200])))
    c = st.columns([1.2, 1.2, 3], vertical_alignment="bottom")
    metric = c[0].selectbox("show", ["RMSE", "|bias|", "estimate", "sd", "γ²"], key="w_hh_metric", persist_state=P)
    N = int(c[1].number_input("shot budget B", min_value=1, step=1000, key="w_hh_N", persist_state=P))
    figs["sweep"] = AC.sweep_lines(rows, N, xlabel, metric, dark=C.dark())
    W.show_fig(figs["sweep"], "sweep")
    W.why("sweep_lines")
    shots = np.logspace(1, 8, 57)
    vals, methods, win, best, second = A.winner_map(rows, shots)
    fam = {r["method"]: r["family"] for r in rows}
    st.subheader("Who wins where")
    figs["winner_map"] = AC.winner_heatmap(vals, methods, [fam[m] for m in methods], win, best, second, shots, xlabel,
                                           dark=C.dark())
    W.show_fig(figs["winner_map"], "winner_map")
    W.why("winner_map")
    if len(methods) >= 2:
        st.subheader("Duel")
        S.wdefault("w_hh_a", methods[0])
        S.wdefault("w_hh_b", methods[1])
        if ss.w_hh_a not in methods:
            ss.w_hh_a = methods[0]
        if ss.w_hh_b not in methods:
            ss.w_hh_b = methods[1]
        c = st.columns(2)
        a = c[0].selectbox("method a", methods, key="w_hh_a", persist_state=P)
        b = c[1].selectbox("method b", methods, key="w_hh_b", persist_state=P)
        from qemx.analysis import rmse_row
        Ra = np.array([[rmse_row(r, shots) for r in rows if r["method"] == a and r["value"] == v][0]
                       if any(r["method"] == a and r["value"] == v and np.isfinite(r["bias"]) for r in rows)
                       else np.full(len(shots), np.nan) for v in vals])
        Rb = np.array([[rmse_row(r, shots) for r in rows if r["method"] == b and r["value"] == v][0]
                       if any(r["method"] == b and r["value"] == v and np.isfinite(r["bias"]) for r in rows)
                       else np.full(len(shots), np.nan) for v in vals])
        figs["duel"] = AC.duel_heatmap(vals, shots, Ra / Rb, a, b, xlabel, dark=C.dark())
        W.show_fig(figs["duel"], "duel")
        W.why("duel")
    tables["sweep"] = pd.DataFrame(rows)
    with st.expander("Data", key="exp_hh_data"):
        st.dataframe(tables["sweep"], hide_index=True, width="stretch",
                     column_config=MX.column_config(tables["sweep"].columns))
    state = S.to_json()
    ff, tt = dict(figs), dict(tables)
    _img = bool(ss.get("w_zip_images", True))

    def zip_bytes():
        return make_zip("Head-to-head sweep", tables=tt, figures=ff, state=state, images=_img,
                        notes="Sweep of %s; RMSE at N = %d shots in the line chart." % (xlabel, N))
    st.divider()
    st.download_button("Download all (ZIP): data and figures (PNG/SVG)", zip_bytes,
                       file_name="qem_explorer_head_to_head.zip", mime="application/zip", icon=":material/folder_zip:",
                       key="dl_all_hh")
