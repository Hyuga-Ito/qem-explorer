"""Model mismatch: what happens when a mitigation method's noise model is wrong."""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model mismatch · QEM Explorer", page_icon=":material/tune:", layout="wide")

from qemx import analysis as A  # noqa: E402
from qemx import device  # noqa: E402
from qemx.bundle import make_zip  # noqa: E402
from qemx.recipes import preset  # noqa: E402
from qemx.methods import Recipe  # noqa: E402
from qemx.viz import analysis_charts as AC  # noqa: E402
from ui import common as C  # noqa: E402
from ui import state as S  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar(run_label=None)
st.title("Model mismatch")
st.markdown("PEC and readout mitigation undo the noise **they are told about**. Here the assumed model and the true "
            "noise are set separately, so you can watch a wrong model bias the answer.")
compiled, err = C.compiled_now()
C.context_line(compiled)
if err:
    st.error("The circuit does not compile: %s" % err)
dev = device.is_device(ss.backend)
if dev and compiled is not None:
    ok_dm, dm = W.guarded("Reading the device calibration", device.get_model, ss.backend, tuple(compiled.info["phys"]))
    true_ref = dm.equivalent_noise() if ok_dm else ss.noise
    st.info("Device model selected: the truth is %s; PEC is told the device's median rates × the ratio below "
            "(p₁ = %.2e, p₂ = %.2e at ratio 1). The heat map and drift tabs use the injector's model on NumPy."
            % (device.device_label(ss.backend), true_ref.p1, true_ref.p2), icon=":material/memory:")
else:
    true_ref = ss.noise

for k, v in (("w_mm_rmin", 0.0), ("w_mm_rmax", 2.0), ("w_mm_pts", 11), ("w_mm_con", "inverse"),
             ("w_mm_pure", True), ("w_mm_amin", 0.002), ("w_mm_amax", 0.04), ("w_mm_gpts", 9), ("w_mm_dmin", -0.5),
             ("w_mm_dmax", 0.5), ("w_mm_eps", "0, 0.02, 0.04, 0.06, 0.08, 0.1")):
    S.wdefault(k, v)
TABS = ["Assumed vs true rate", "Mismatch map", "Readout drift", "Unmodeled noise"]
S.wdefault("w_mm_tabs", ss.page_tab.get("mm", TABS[0]))
tabs = st.tabs(TABS, key="w_mm_tabs", on_change=S.cb_tab, args=("mm", "w_mm_tabs"))
figs, tables = {}, {}

with tabs[0]:
    if tabs[0].open is not False:
        c = st.columns(5, vertical_alignment="bottom")
        rmin = c[0].number_input("ratio from", min_value=0.0, step=0.1, key="w_mm_rmin", persist_state=P)
        rmax = c[1].number_input("to", min_value=0.01, step=0.1, key="w_mm_rmax", persist_state=P)
        pts = c[2].number_input("points", min_value=2, step=1, key="w_mm_pts", persist_state=P)
        con = c[3].selectbox("construction", ["inverse", "compensation"], key="w_mm_con", persist_state=P)
        pure = c[4].toggle("true noise = depolarizing only", key="w_mm_pure", persist_state=P, disabled=dev,
                           help="On: the truth has only p₁ and p₂, so ratio 1 is exactly unbiased. Off: the injector's "
                                "full noise (damping, coherent, idle, readout) is the truth.")

        def req():
            ss.mm_run = True
        st.button("Run the mismatch curve", type="primary", icon=":material/play_arrow:", key="mm_btn", on_click=req,
                  disabled=compiled is None)
        if ss.pop("mm_run", False) and compiled is not None:
            truth = true_ref if dev else (type(ss.noise)(p1=ss.noise.p1, p2=ss.noise.p2) if pure else ss.noise)
            if not dev and truth.p1 == 0 and truth.p2 == 0:
                st.warning("The injector has no depolarizing noise (p₁ = p₂ = 0): set p₂ in the sidebar first.")
            else:
                bar = st.progress(0.0, text="PEC with each assumed rate…")
                def sweep_():                           # the grid is built inside the guard (any number of points)
                    ratios = np.linspace(float(rmin), float(max(rmax, rmin + 1e-6)), int(pts))
                    return A.pec_mismatch(compiled, truth, ss.observable, ratios, con, "all",
                                          ss.backend if dev else "numpy", int(ss.shots),
                                          compiled.info.get("phys") if dev else None,
                                          lambda f: bar.progress(min(f, 1.0), text="PEC with each assumed rate…"))
                ok, rows = W.guarded("The mismatch sweep", sweep_, n=int(ss.n))
                bar.empty()
                if ok:
                    ss.mm_result = dict(rows=rows, sig=S.signature(extra=("mm", pure, con, float(rmin), float(rmax),
                                                                          int(pts))))
        r = ss.get("mm_result")
        if r:
            if r.get("sig") != S.signature(extra=("mm", pure, con, float(rmin), float(rmax), int(pts))):
                C.stale_note("Run")
            figs["mismatch_curve"] = AC.mismatch_curve(r["rows"], dark=C.dark())
            W.show_fig(figs["mismatch_curve"], "mismatch_curve")
            W.why("mismatch")
            tables["mismatch"] = pd.DataFrame(r["rows"])
            with st.expander("Data", key="exp_mm_data"):
                st.dataframe(tables["mismatch"], hide_index=True, width="stretch")

with tabs[1]:
    if tabs[1].open is not False:
        st.markdown("Bias of PEC over **assumed** (columns) and **true** (rows) two-qubit error rates; p₁ keeps the "
                    "injector's ratio p₁/p₂. Exact NumPy simulation.")
        c = st.columns(4, vertical_alignment="bottom")
        amin = c[0].number_input("p₂ from", min_value=0.0, max_value=0.9, step=0.001, format="%.4f", key="w_mm_amin",
                                 persist_state=P)
        amax = c[1].number_input("to", min_value=0.0001, max_value=0.9, step=0.001, format="%.4f", key="w_mm_amax",
                                 persist_state=P)
        gp = c[2].number_input("grid points", min_value=2, step=1, key="w_mm_gpts", persist_state=P,
                               help="n² PEC runs; large grids take longer (there is no cap).")

        def req2():
            ss.mm_map_run = True
        c[3].button("Run map", type="primary", icon=":material/grid_on:", key="mm_map_btn", on_click=req2,
                    disabled=compiled is None)
        if ss.pop("mm_map_run", False) and compiled is not None:
            import dataclasses
            opts = dataclasses.replace(S.compile_options(), device=None, initial_layout=None)
            bar = st.progress(0.0, text="PEC over the grid…")

            def map_():                                 # the grid is built inside the guard
                grid_ = np.linspace(float(amin), float(max(amax, amin + 1e-6)), int(gp))
                base = ss.noise if not dev else true_ref
                if base.p2 == 0:
                    base = dataclasses.replace(base, p2=float(grid_[len(grid_) // 2]))
                return grid_, A.pec_mismatch_map(S.spec(), ss.inputs, ss.basis, opts, ss.relabel,
                                                 type(base)(p1=base.p1, p2=base.p2), ss.observable, grid_, grid_, True,
                                                 "inverse", lambda f: bar.progress(f, text="PEC over the grid…"))
            ok, gB = W.guarded("The mismatch map", map_, n=int(ss.n))
            grid, B = gB if ok else (None, None)
            bar.empty()
            if ok:
                ss.mm_map = dict(B=B, grid=grid, sig=S.signature(extra=("mm_map", float(amin), float(amax), int(gp))))
        m = ss.get("mm_map")
        if m:
            if m.get("sig") != S.signature(extra=("mm_map", float(amin), float(amax), int(gp))):
                C.stale_note("Run map")
            figs["mismatch_map"] = AC.mismatch_heatmap(m["B"], m["grid"], m["grid"], dark=C.dark())
            W.show_fig(figs["mismatch_map"], "mismatch_map")
            W.why("mismatch_map")
            tables["mismatch_map"] = pd.DataFrame(m["B"], index=["true %.4g" % g for g in m["grid"]],
                                                  columns=["assumed %.4g" % g for g in m["grid"]]).reset_index()

with tabs[2]:
    if tabs[2].open is not False:
        st.markdown("Tensored readout mitigation calibrated with flip rates that are off by a fraction δ.")
        c = st.columns(3, vertical_alignment="bottom")
        dmin = c[0].number_input("δ from", step=0.1, key="w_mm_dmin", persist_state=P)
        dmax = c[1].number_input("to", step=0.1, key="w_mm_dmax", persist_state=P)
        rn = ss.noise if not dev else true_ref
        drift_sig = S.signature(extra=("drift", float(dmin), float(dmax), dev))

        def req4():
            ss.mm_drift_run = True
        if rn.ro01 == 0 and rn.ro10 == 0:
            st.warning("The noise has no readout error; set ε₀₁ / ε₁₀ in the sidebar to see drift.")
        else:
            st.button("Run the drift curve", type="primary", icon=":material/play_arrow:", key="mm_drift_btn",
                      on_click=req4, disabled=compiled is None)
        if ss.pop("mm_drift_run", False) and compiled is not None and (rn.ro01 or rn.ro10):
            def drift_():                               # compile (device: on NumPy) and simulate inside the guard
                import dataclasses
                comp_np = compiled
                if dev:
                    from qemx import compile as CP
                    comp_np = CP.build(S.spec(), ss.inputs, ss.basis,
                                       dataclasses.replace(S.compile_options(), device=None, initial_layout=None),
                                       ss.relabel)
                return A.readout_mismatch(comp_np, rn, ss.observable,
                                          np.linspace(float(dmin), float(max(dmax, dmin + 1e-6)), 21))
            ok, rows = W.guarded("The drift curve", drift_, n=int(ss.n))
            if ok:
                ss.mm_drift = dict(rows=rows, sig=drift_sig)
        dr = ss.get("mm_drift")
        if dr:
            if dr.get("sig") != drift_sig:
                C.stale_note("Run the drift curve")
            figs["readout_drift"] = AC.drift_curve(dr["rows"], dark=C.dark())
            W.show_fig(figs["readout_drift"], "readout_drift")
            W.why("readout_drift")
            tables["readout_drift"] = pd.DataFrame(dr["rows"])

with tabs[3]:
    if tabs[3].open is not False:
        st.markdown("PEC with the **correct** depolarizing rates, while the truth also has a coherent over-rotation "
                    "ε that the model does not contain.")
        eps_txt = st.text_input("over-rotation ε values", key="w_mm_eps", persist_state=P, max_chars=200)

        def req3():
            ss.mm_un_run = True
        st.button("Run the coherent-error sweep", type="primary", icon=":material/play_arrow:", key="mm_un_btn", on_click=req3,
                  disabled=compiled is None or dev)
        if dev:
            st.caption("This tab varies the injector's model; choose the NumPy or Aer simulator in the sidebar.")
        if ss.pop("mm_un_run", False) and compiled is not None and not dev:
            try:
                vals = sorted(set(W.parse_floats(eps_txt)))     # finite decimal numbers only (no nan / inf)
            except ValueError:
                vals = []
                st.error("Enter finite numbers separated by commas.")
            if vals:
                import dataclasses
                base = dataclasses.replace(type(ss.noise)(p1=ss.noise.p1, p2=ss.noise.p2 or 0.01))
                rec = [Recipe.make("raw"), Recipe.make("pec", construction="inverse"),
                       preset("PEC · learned model + RC")]
                bar = st.progress(0.0, text="Sweeping ε…")
                ok, rows = W.guarded("The unmodeled-noise sweep", A.sweep, compiled, base, ss.observable, rec,
                                     "over_rot", vals, "numpy", int(ss.shots), ss.symmetry,
                                     lambda f: bar.progress(f, text="Sweeping ε…"), n=int(ss.n))
                bar.empty()
                if ok:
                    ss.mm_un = rows
                    ss.mm_un_sig = S.signature(extra=("mm_un", eps_txt))
        rows = ss.get("mm_un")
        if rows:
            if ss.get("mm_un_sig") != S.signature(extra=("mm_un", eps_txt)):
                C.stale_note("Run")
            figs["unmodelled_noise"] = AC.sweep_lines(rows, int(ss.shots), "coherent over-rotation ε (not in the model)",
                                                      "estimate", dark=C.dark())
            W.show_fig(figs["unmodelled_noise"], "unmodelled_noise")
            W.why("unmodeled")
            tables["unmodelled_noise"] = pd.DataFrame(rows)

# everything this page has computed, captured now so that the ZIP (built in a download thread) never reads
# session_state; figures of tabs that are closed in this run are rebuilt from the stored results
_cap = dict(figs=dict(figs), tables=dict(tables), mm=ss.get("mm_result"), mm_map=ss.get("mm_map"), mm_un=ss.get("mm_un"),
            shots=int(ss.shots), state=S.to_json(), drift=ss.get("mm_drift"))


def _collect(cap):
    fg, tb = dict(cap["figs"]), dict(cap["tables"])
    if cap["mm"] and "mismatch_curve" not in fg:
        fg["mismatch_curve"] = AC.mismatch_curve(cap["mm"]["rows"])
        tb["mismatch"] = pd.DataFrame(cap["mm"]["rows"])
    if cap["mm_map"]:
        m = cap["mm_map"]
        fg.setdefault("mismatch_map", AC.mismatch_heatmap(m["B"], m["grid"], m["grid"]))
        tb.setdefault("mismatch_map", pd.DataFrame(m["B"], index=["true %.4g" % g for g in m["grid"]],
                                                   columns=["assumed %.4g" % g for g in m["grid"]]).reset_index())
    if cap["mm_un"] and "unmodelled_noise" not in fg:
        fg["unmodelled_noise"] = AC.sweep_lines(cap["mm_un"], cap["shots"], "coherent over-rotation ε (not in the model)",
                                                "estimate")
        tb["unmodelled_noise"] = pd.DataFrame(cap["mm_un"])
    if cap.get("drift") and "readout_drift" not in fg:
        try:
            fg["readout_drift"] = AC.drift_curve(cap["drift"]["rows"])
            tb["readout_drift"] = pd.DataFrame(cap["drift"]["rows"])
        except Exception:                           # the rest of the ZIP is still useful
            pass
    return fg, tb


_has_any = bool(figs or _cap["mm"] or _cap["mm_map"] or _cap["mm_un"] or _cap.get("drift"))


_IMG = bool(ss.get("w_zip_images", True))          # captured now: the ZIP is built in a download thread


def zip_bytes():
    fg, tb = _collect(_cap)
    return make_zip("Model mismatch", tables=tb, figures=fg, state=_cap["state"], images=_IMG)


st.divider()
st.download_button("Download all (ZIP): every table and figure of this page (PNG/SVG)", zip_bytes,
                   file_name="qem_explorer_model_mismatch.zip", mime="application/zip", icon=":material/folder_zip:",
                   key="dl_all_mm", disabled=not _has_any)
