"""Cost vs precision: RMSE against shots and QPU time for your circuit, the PEC exponential wall, and a depth
sweep that shows where each method breaks as circuits grow."""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cost vs precision · QEM Explorer", page_icon=":material/trending_up:", layout="wide")

from qemx import analysis as A  # noqa: E402
from qemx import compile as CP  # noqa: E402
from qemx import report as RP  # noqa: E402
from qemx import templates as T  # noqa: E402
from qemx.bundle import make_zip  # noqa: E402
from qemx.viz import analysis_charts as AC  # noqa: E402
from ui import common as C  # noqa: E402
from ui import state as S  # noqa: E402
from ui import metrics as MX  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar("Run simulation", "cvp_run")
st.title("Cost vs precision")
st.markdown("How many shots — and how much QPU time — does each method need for a given precision? Where does "
            "mitigation start to pay off, and where does PEC hit its exponential wall? Every method is costed at the "
            "same **total** shot budget B: target, training and calibration circuits together.")
compiled, err = C.compiled_now()
C.context_line(compiled)
if err:
    st.error("The circuit does not compile: %s" % err)

for _k, _v in (("w_cvp_eps", 0.01), ("w_cvp_budget", 600.0), ("w_cvp_nmin", 100), ("w_cvp_nmax", 10 ** 8),
               ("w_wall_n", 10), ("w_wall_p1", 5e-4), ("w_wall_p2s", "0.002, 0.005, 0.01, 0.02"), ("w_wall_eps", 0.01),
               ("w_wall_L", 60), ("w_wall_con", "inverse"), ("w_wall_budget", 3600.0), ("w_ds_family", "mirror"),
               ("w_ds_depths", "0, 1, 2, 3, 4"), ("w_ds_N", 10000)):
    S.wdefault(_k, _v)
if ss.pop("cvp_run", False) and compiled is not None:
    with st.spinner("Simulating every contender…"):
        C.run_comparison(compiled)
TABS = ["Your circuit", "The PEC wall", "Depth sweep"]
tabs = S.page_tabs("cvp", "w_cvp_tabs", TABS)          # the selected tab is app state (survives page switches)
figs, tables = {}, {}

# ------------------------------------------------------------------ 1. RMSE vs shots for the current comparison
with tabs[0]:
    if tabs[0].open is not False:
        c = st.columns([1, 1, 1, 1.2], vertical_alignment="bottom")
        for k, v in (("w_cvp_eps", 0.01), ("w_cvp_budget", 600.0), ("w_cvp_nmin", 100), ("w_cvp_nmax", 10 ** 8)):
            S.wdefault(k, v)
        eps = c[0].number_input("target precision ε (RMSE)", min_value=1e-6, format="%.4g", step=0.001, key="w_cvp_eps",
                                persist_state=P)
        budget = c[1].number_input("QPU time budget (s)", min_value=0.0, step=60.0, format="%.1f", key="w_cvp_budget",
                                   persist_state=P, help="Drawn as a line; converted to shots with the sidebar's time "
                                                         "per shot.")
        nmin = c[2].number_input("budget B from (shots)", min_value=1, step=10, key="w_cvp_nmin", persist_state=P)
        nmax = c[3].number_input("to", min_value=2, step=1000, key="w_cvp_nmax", persist_state=P)
        if nmin >= nmax:
            st.warning("‘shots from’ (%s) is not below ‘to’ (%s); the charts use %s to %s." % (
                f"{int(nmin):,}", f"{int(nmax):,}", f"{int(nmin):,}", f"{int(nmin) * 10:,}"))
            nmax = int(nmin) * 10

        res = ss.get("results")
        if res is None:
            st.info("No comparison yet: press **Run simulation** in the sidebar (it runs the Playground's contenders).",
                    icon=":material/play_arrow:")
        else:
            if not C.results_current():
                st.warning("These results are from different settings; press **Run simulation** to refresh.")
            f_, t_ = RP.cost_precision(res, float(eps), float(budget), int(nmin), int(nmax), float(ss.t_shot_us),
                                       dark=C.dark())
            figs.update(f_)
            tables.update(t_)
            if "rmse_vs_shots" in figs:
                W.show_fig(figs["rmse_vs_shots"], "rmse_vs_shots", t_.get("rmse_vs_budget"))
                W.why("rmse_shots")
                W.show_fig(figs["time_needed"], "time_needed")
                W.why("time_needed")
                df = tables["cost_for_target"].copy()
                for col in ("budget for ε", "QPU time for ε (s)", "beats unmitigated above B"):
                    df[col] = [("∞" if v == float("inf") else "–") if (v is None or not np.isfinite(v)) else
                               "%.3g" % v for v in df[col]]
                cfg = MX.column_config(df.columns)
                for col in ("budget for ε", "QPU time for ε (s)", "beats unmitigated above B"):
                    cfg[col] = st.column_config.TextColumn(col, help=MX.help_for(col))
                st.dataframe(df, hide_index=True, width="stretch", column_config=cfg)
            st.caption("‘∞’ means no budget reaches ε: the method's bias — or, for CDR / FR / NIL, the shot noise of its fixed "
                       "training data — is larger than ε. ‘–’ in the last column: never better than the unmitigated "
                       "estimate (or it is the reference). Budgets include each method's training and calibration "
                       "shots.")

# ------------------------------------------------------------------ 2. the PEC wall
with tabs[1]:
    if tabs[1].open is not False:
        st.markdown("PEC is unbiased for a correct noise model, but its shot cost multiplies gate by gate. Pick a "
                    "register and error rates to see the wall.")
        for k, v in (("w_wall_n", 10), ("w_wall_p1", 5e-4), ("w_wall_p2s", "0.002, 0.005, 0.01, 0.02"),
                     ("w_wall_eps", 0.01), ("w_wall_L", 60), ("w_wall_con", "inverse"), ("w_wall_budget", 3600.0)):
            S.wdefault(k, v)
        c = st.columns(4)
        nq = c[0].number_input("qubits", min_value=2, step=1, key="w_wall_n", persist_state=P)
        p1 = c[1].number_input("one-qubit p₁", min_value=0.0, max_value=1.0, step=1e-4, format="%.5f", key="w_wall_p1",
                               persist_state=P)
        p2txt = c[2].text_input("two-qubit p₂ values", key="w_wall_p2s", persist_state=P, max_chars=200,
                                help="comma-separated, e.g. 0.002, 0.01")
        eps_w = c[3].number_input("target precision ε", min_value=1e-6, step=0.001, format="%.4g", key="w_wall_eps",
                                  persist_state=P)
        c = st.columns(4)
        Lmax = c[0].number_input("max layers", min_value=1, step=5, key="w_wall_L", persist_state=P)
        con = c[1].selectbox("construction", ["inverse", "compensation"], key="w_wall_con", persist_state=P)
        bud = c[2].number_input("QPU budget (s)", min_value=0.0, step=600.0, key="w_wall_budget", persist_state=P)
        p2s = RP.parse_p2s(p2txt, default=None)
        if p2s is None:
            st.error("Enter two-qubit error rates between 0 and 1, separated by commas.")
            p2s = [0.01]
        f_, t_, wall, layers = RP.pec_wall(int(nq), float(p1), p2s, float(eps_w), int(Lmax), con, float(bud),
                                           float(ss.t_shot_us), dark=C.dark())
        figs.update(f_)
        tables.update(t_)
        W.show_fig(figs["pec_wall"], "pec_wall", t_["pec_wall_curves"])
        st.markdown("**Where the budget runs out** (dashed contour) over depth and two-qubit error rate")
        W.show_fig(figs["pec_wall_map"], "pec_wall_map")
        W.why("pec_wall")
        st.dataframe(tables["pec_wall"], hide_index=True, width="stretch", column_config={
            "γ per layer": st.column_config.NumberColumn(format="%.4f"),
            "Γ at max depth": st.column_config.NumberColumn(format="%.3g"),
            "shots at max depth": st.column_config.NumberColumn(format="%.3g")})

# ------------------------------------------------------------------ 3. depth sweep with real simulations
with tabs[2]:
    if tabs[2].open is not False:
        st.markdown("Simulate circuits of growing depth that share one ideal answer, with your contenders and your "
                    "noise (or device).")
        for k, v in (("w_ds_family", "mirror"), ("w_ds_depths", "0, 1, 2, 3, 4"), ("w_ds_N", 10000)):
            S.wdefault(k, v)
        c = st.columns([1.4, 1.4, 1])
        fam = c[0].selectbox("circuit family", ["mirror", "ising"], key="w_ds_family", persist_state=P,
                             format_func={"mirror": "your circuit, echoed: U (U†U)^k",
                                          "ising": "Ising model, k Trotter steps"}.get)
        dtxt = c[1].text_input("depths k", key="w_ds_depths", persist_state=P, max_chars=200)
        Nd = c[2].number_input("shot budget B for the RMSE", min_value=1, step=1000, key="w_ds_N", persist_state=P)
        try:
            depths = sorted(set(W.parse_ints(dtxt)))       # ASCII whole numbers only
            if not depths or min(depths) < 0:
                raise ValueError
            if fam == "ising":
                depths = [max(d, 1) for d in depths]
        except ValueError:
            st.error("Enter non-negative whole numbers separated by commas.")
            depths = []

        def req_ds():
            ss.ds_run = True
        st.button("Run depth sweep", type="primary", icon=":material/play_arrow:", key="ds_btn", on_click=req_ds,
                  disabled=compiled is None or not depths)
        if ss.pop("ds_run", False) and compiled is not None and depths:
            recipes = S.recipes()
            opts = S.compile_options()
            if fam == "mirror":
                def mk(k):
                    base = S.spec()
                    return CP.build(A.mirror_family(base, k), ss.inputs, ss.basis, opts, ss.relabel)

                def ob(k):
                    return ss.observable
            else:
                n_is = max(int(ss.n), 2)

                def mk(k):
                    return CP.build(T.build("ising", dict(n=n_is, steps=k)).spec, None, None, opts, False)

                def ob(k):
                    return T.build("ising", dict(n=n_is, steps=k)).observable
            bar = st.progress(0.0, text="Simulating depths…")
            ok, rows = W.guarded("The depth sweep", A.depth_sweep, mk, depths, ss.noise, ob, recipes, ss.backend,
                                 int(ss.shots), ss.symmetry, lambda f: bar.progress(f, text="Simulating depths…"),
                                 n=int(ss.n))
            bar.empty()
            if ok:
                ss.ds_result = dict(rows=rows, family=fam, sig=S.signature(extra=("ds", fam, tuple(depths))))
        dsr = ss.get("ds_result")
        if dsr:
            if dsr.get("sig") != S.signature(extra=("ds", fam, tuple(depths))):
                st.warning("This sweep was computed for other settings (circuit, noise, contenders, family or depths); "
                           "press Run depth sweep to refresh.")
            figs["depth_sweep"] = AC.depth_chart(dsr["rows"], int(Nd), dark=C.dark())
            W.show_fig(figs["depth_sweep"], "depth_sweep")
            W.why("depth_sweep")
            tables["depth_sweep"] = pd.DataFrame(dsr["rows"])
            with st.expander("Data", key="exp_ds_data"):
                st.dataframe(tables["depth_sweep"], hide_index=True, width="stretch")


# ------------------------------------------------------------------ download everything this page has produced
_res = ss.get("results")
_ds = ss.get("ds_result")
_state = S.to_json()
_IMG = bool(ss.get("w_zip_images", True))          # captured now: the ZIP is built in a download thread
_params = dict(eps=float(ss.w_cvp_eps), budget=float(ss.w_cvp_budget), nmin=int(ss.w_cvp_nmin), nmax=int(ss.w_cvp_nmax),
               nq=int(ss.w_wall_n), p1=float(ss.w_wall_p1), p2s=str(ss.w_wall_p2s), eps_w=float(ss.w_wall_eps),
               L=int(ss.w_wall_L), con=ss.w_wall_con, bud=float(ss.w_wall_budget), N=int(ss.w_ds_N), t=float(ss.t_shot_us))


def zip_bytes():
    """Runs in a download thread: uses only the values captured above, never session_state."""
    pr, figs_all, tabs_all = _params, {}, {}
    if _res is not None:
        f_, t_ = RP.cost_precision(_res, pr["eps"], pr["budget"], pr["nmin"], pr["nmax"], pr["t"])
        figs_all.update(f_)
        tabs_all.update(t_)
        tabs_all["comparison_rows"] = pd.DataFrame(_res["rows"])
    f_, t_, _, _ = RP.pec_wall(pr["nq"], pr["p1"], pr["p2s"], pr["eps_w"], pr["L"], pr["con"], pr["bud"], pr["t"])
    figs_all.update(f_)
    tabs_all.update(t_)
    if _ds:
        figs_all["depth_sweep"] = AC.depth_chart(_ds["rows"], pr["N"])
        tabs_all["depth_sweep"] = pd.DataFrame(_ds["rows"])
    return make_zip("Cost vs precision", tables=tabs_all, figures=figs_all, state=_state, images=_IMG,
                    notes="Time per shot %.1f µs; target ε = %g; QPU budget %g s." % (pr["t"], pr["eps"], pr["budget"]))


st.divider()
st.download_button("Download all (ZIP): every table and figure of this page (PNG/SVG)", zip_bytes,
                   file_name="qem_explorer_cost_vs_precision.zip", mime="application/zip", icon=":material/folder_zip:",
                   key="dl_all_cvp")
