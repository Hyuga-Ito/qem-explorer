"""QEM Performance & Selection: study results (regime maps, paired comparisons, costs), the baseline recommender for
the current circuit, and the drill-down analyses. Everything here reads stored study tables or the current circuit;
nothing heavy runs until a button is pressed."""
import json
import math
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="QEM Performance & Selection · QEM Explorer", page_icon=":material/insights:",
                   layout="wide")

from qemx import features as FT  # noqa: E402
from qemx import recommend as RC  # noqa: E402
from ui import common as C  # noqa: E402
from ui import state as S  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar(run_label=None)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")

st.title("QEM Performance & Selection")
st.markdown("Which mitigation method works for which circuit, observable, noise and budget — measured at **equal total "
            "shot budget** (training, calibration and every extra circuit included) against the exact ideal value. "
            "Results come from pre-registered study runs (`python -m qemx.phase.run`, then `python -m qemx.study "
            "report`); this page only reads their tables.")


def study_dirs() -> list:
    """Report directories under reports/ that hold the study tables (no free-text paths: nothing else is read)."""
    if not os.path.isdir(REPORTS):
        return []
    out = []
    for d in sorted(os.listdir(REPORTS)):
        t = os.path.join(REPORTS, d, "tables")
        if os.path.exists(os.path.join(t, "rmse_wide.csv")) and os.path.exists(os.path.join(t, "cells_features.csv")):
            out.append(d)
    return out


@st.cache_data(max_entries=4, show_spinner=False)
def load_tables(name: str, mtime: float):
    t = os.path.join(REPORTS, name, "tables")
    W_ = pd.read_csv(os.path.join(t, "rmse_wide.csv"))
    R_ = pd.read_csv(os.path.join(t, "regimes.csv")) if os.path.exists(os.path.join(t, "regimes.csv")) else None
    I_ = pd.read_csv(os.path.join(t, "paired_improvement.csv")) if os.path.exists(
        os.path.join(t, "paired_improvement.csv")) else None
    ev = None
    if os.path.exists(os.path.join(t, "recommender_evaluation.json")):
        with open(os.path.join(t, "recommender_evaluation.json"), encoding="utf-8") as f:
            ev = json.load(f)
    return W_, R_, I_, ev


def method_feasibility(name: str, methods: list, comp, noise, observable, B: int) -> list:
    """Whether each study method can run at budget B on this circuit (the budget preflight; True when a method has
    no fixed-cost model)."""
    from qemx.methods import Context, Recipe, Runner
    from qemx.recipes import preflight
    path = os.path.join(REPORTS, name, "tables", "methods.json")
    specs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            specs = json.load(fh)
    out = []
    for m in methods:
        d = dict(specs.get(m) or {})
        est = d.pop("estimator", None)
        if est is None:
            out.append(True)
            continue
        tw, dd = d.pop("twirl", "none"), d.pop("dd", "none")
        rc = Recipe.make(est, twirl=tw, dd=dd, **d)
        pf = preflight(rc, lambda c, o, rc=rc: Context(c, noise, o, rc, Runner("numpy"), shots=B), comp, observable, B)
        out.append(True if pf is None else bool(pf["feasible"]))
    return out


dirs = study_dirs()
TABS = ["Study results", "Recommend for my circuit", "Drill-down analyses", "Feature dictionary"]
S.wdefault("w_qp_tabs", ss.page_tab.get("qp", TABS[0]))
tabs = st.tabs(TABS, key="w_qp_tabs", on_change=S.cb_tab, args=("qp", "w_qp_tabs"))

with tabs[0]:
    if tabs[0].open is not False:
        if not dirs:
            st.info("No study report found under `reports/`. Run a study and its report, e.g. `python -m qemx.study "
                    "report data/dirA_v2_pilot --manifest manifests/direction_A_v2_pilot.yaml --out reports/pilot`.",
                    icon=":material/folder_open:")
        else:
            S.wdefault("w_qp_dir", dirs[0])
            if ss.w_qp_dir not in dirs:
                ss.w_qp_dir = dirs[0]
            name = st.selectbox("Study report", dirs, key="w_qp_dir", persist_state=P)
            mt = os.path.getmtime(os.path.join(REPORTS, name, "tables", "rmse_wide.csv"))
            ok, tabs_ = W.guarded("Reading the study tables", load_tables, name, mt)
            if ok:
                W_, R_, I_, ev = tabs_
                cells = W_["cell_id"].nunique()
                groups = W_["group"].nunique() if "group" in W_ else cells
                st.caption("%d cells · %d base circuits · budgets %s … %s shots · simulation on the study's device "
                           "model (not hardware)." % (cells, groups, f"{int(W_['B'].min()):,}", f"{int(W_['B'].max()):,}"))
                figs = [f for f in ("fig2_regime_map", "fig3_paired_improvement", "fig4_cost_to_precision",
                                    "fig6_recommender", "fig1_coverage", "fig5_discrepancy_rehearsal")
                        if os.path.exists(os.path.join(REPORTS, name, f + ".png"))]
                S.wdefault("w_qp_fig", figs[0] if figs else "")
                if figs:
                    if ss.w_qp_fig not in figs:
                        ss.w_qp_fig = figs[0]
                    labels = {"fig1_coverage": "Coverage", "fig2_regime_map": "Regime map",
                              "fig3_paired_improvement": "Paired change vs raw",
                              "fig4_cost_to_precision": "Cost to precision", "fig5_discrepancy_rehearsal":
                              "Discrepancy rehearsal (emulation)", "fig6_recommender": "Recommender evaluation"}
                    f = st.segmented_control("Figure", figs, key="w_qp_fig", format_func=lambda k: labels.get(k, k),
                                             persist_state=P)
                    if f:
                        st.image(os.path.join(REPORTS, name, f + ".png"), width="stretch")
                with st.expander("Report text (captions, tables, limits)", icon=":material/description:"):
                    rp = os.path.join(REPORTS, name, "report.md")
                    if os.path.exists(rp):
                        with open(rp, encoding="utf-8") as fh:
                            st.markdown(fh.read().replace("![](", "<!-- ").replace(".png)", " -->"))
                if I_ is not None:
                    st.subheader("Paired change against raw")
                    st.dataframe(I_, hide_index=True, width="stretch")
                if R_ is not None:
                    with st.expander("Regime labels of every (cell, budget)", icon=":material/table_rows:"):
                        st.dataframe(R_, hide_index=True, width="stretch", height=320)

with tabs[1]:
    if tabs[1].open is not False:
        st.markdown("A **transparent baseline**: one ridge model per method predicts log₁₀ RMSE from features that are "
                    "known before running anything (circuit, observable, noise model, budget). It recommends a method "
                    "only when the prediction separates it from the alternatives; it says **use raw** when no method "
                    "is predicted to beat unmitigated results by more than 10 %, and it **abstains** outside the "
                    "training data. It is not an optimal-QEM oracle: see its held-out evaluation in the study report.")
        if not dirs:
            st.info("Needs a study report under `reports/` to learn from.", icon=":material/folder_open:")
        else:
            S.wdefault("w_qp_rdir", dirs[0])
            if ss.w_qp_rdir not in dirs:
                ss.w_qp_rdir = dirs[0]
            c1, c2, c3 = st.columns(3)
            name = c1.selectbox("Learn from", dirs, key="w_qp_rdir", persist_state=P)
            S.wdefault("w_qp_fs", "structural")
            fs = c2.selectbox("Features", list(FT.SETS), key="w_qp_fs", persist_state=P,
                              help="structural: counts and fault rates; normalized: per-qubit/per-gate versions; "
                                   "with_state: adds exact entanglement features (cost 2^n).")
            S.wdefault("w_qp_B", int(ss.get("shots", 4096)))
            B = c3.number_input("Total shot budget B", min_value=100, max_value=10 ** 9, step=1000, key="w_qp_B",
                                persist_state=P)

            def do_recommend():
                ss.qp_rec = None
                ss.qp_err = ""
                try:
                    mt_ = os.path.getmtime(os.path.join(REPORTS, name, "tables", "rmse_wide.csv"))
                    W_, _, _, _ = load_tables(name, mt_)
                    names = list(FT.SETS[fs])
                    methods = [c[5:] for c in W_.columns if c.startswith("rmse_") and "feas_" + c[5:] in W_]
                    mj = os.path.join(REPORTS, name, "tables", "methods.json")
                    if os.path.exists(mj):                 # the study's candidates (upper-bound methods excluded)
                        with open(mj, encoding="utf-8") as fh:
                            cand = set(json.load(fh))
                        methods = [m for m in methods if m in cand]
                    W_ = W_.assign(x_log10_budget=np.log10(W_["B"].astype(float)))
                    X = W_[["x_" + n for n in names]].to_numpy(float)
                    Y = np.column_stack([np.log10(W_["rmse_" + m].astype(float).clip(lower=1e-12)) for m in methods])
                    F = np.column_stack([W_["feas_" + m].astype(bool) & np.isfinite(W_["rmse_" + m].astype(float))
                                         for m in methods])
                    model = RC.RidgeRecommender(methods, alpha=1.0).fit(X, Y, F)
                    comp, err = C.compiled_now()
                    if comp is None:
                        raise ValueError("the circuit does not compile: %s" % err)
                    from qemx import device as DV
                    noise = ss.noise
                    v, miss = FT.from_live(comp, noise, ss.observable, float(B))
                    x = np.array([v.get(n, math.nan) for n in names], float)
                    # feasibility at B from each protocol's fixed cost (training + calibration), before any run
                    feas = method_feasibility(name, methods, comp, noise, ss.observable, int(B))
                    r = model.recommend(x, feas)
                    ctx_note = ""
                    if DV.is_device(ss.backend) is False:
                        ctx_note = ("The study was simulated on a fake device model; your noise model is the "
                                    "injector's. Treat any recommendation as an extrapolation across noise models.")
                    ss.qp_rec = dict(kind=r.kind, method=r.method, reason=r.reason, predicted=r.predicted,
                                     interval=list(r.interval), alternatives=list(r.alternatives),
                                     features={n: v.get(n) for n in names},
                                     missing={n: miss[n] for n in names if n in miss}, note=ctx_note,
                                     sig=S.signature(), budget=int(B), study=name, feature_set=fs)
                except Exception as e:                   # noqa: BLE001 - never crash the page from a callback
                    ss.qp_err = "Could not compute a recommendation: %s: %s" % (type(e).__name__, str(e)[:300])
            st.button("Recommend", type="primary", icon=":material/recommend:", key="qp_go", on_click=do_recommend)
            if ss.get("qp_err"):
                st.error(W.md_escape(ss.qp_err), icon=":material/error:")
            rec = ss.get("qp_rec")
            if rec:
                if rec["sig"] != S.signature() or rec["budget"] != int(B):
                    st.warning("The circuit, noise or budget changed since this recommendation; press Recommend again.",
                               icon=":material/update:")
                if rec["kind"] in ("method",):
                    lo, hi = rec["interval"]
                    st.success("**%s** — predicted RMSE %.3g (90 %% interval %.3g – %.3g). Alternatives: %s." % (
                        rec["method"], 10 ** rec["predicted"][rec["method"]], 10 ** lo, 10 ** hi,
                        ", ".join(rec["alternatives"]) or "none"), icon=":material/recommend:")
                elif rec["kind"] == "raw":
                    st.info("**Use raw results** — %s." % rec["reason"], icon=":material/do_not_disturb_on:")
                else:
                    st.warning("**No recommendation** (%s): %s." % (rec["kind"], rec["reason"]), icon=":material/help:")
                if rec.get("note"):
                    st.caption(rec["note"])
                if rec["predicted"]:
                    st.dataframe(pd.DataFrame(sorted(rec["predicted"].items(), key=lambda kv: kv[1]),
                                              columns=["method", "predicted log10 RMSE"]), hide_index=True)
                st.dataframe(pd.DataFrame([dict(feature=k, value=val, missing=rec["missing"].get(k, ""))
                                           for k, val in rec["features"].items()]), hide_index=True, width="stretch")
                st.caption("Feature weights are associations within the study's design, not causes. Feasibility here "
                           "assumes every method can run at B; the Playground's budget preflight decides that exactly.")

with tabs[2]:
    if tabs[2].open is not False:
        st.markdown("Focused analyses of the current circuit (each runs only when you press its button):")
        for path, label, icon, what in (
                ("pages/2_Cost_vs_Precision.py", "Cost vs precision", ":material/trending_up:",
                 "RMSE against total budget, budget to reach a precision, crossovers between methods."),
                ("pages/5_Head_to_Head.py", "Head-to-head sweeps", ":material/compare_arrows:",
                 "Two or more methods along one parameter (noise scale, depth, shots…)."),
                ("pages/3_Bias_Variance.py", "Bias–variance bullseye", ":material/adjust:",
                 "Where each method's repeated estimates land: bias against spread."),
                ("pages/8_Batch_Run.py", "Batch run", ":material/playlist_play:",
                 "Many algorithms × methods × noise settings in one table.")):
            c = st.container(border=True)
            c.page_link(path, label=label, icon=icon)
            c.caption(what)

with tabs[3]:
    if tabs[3].open is not False:
        st.markdown(FT.dictionary_markdown())
