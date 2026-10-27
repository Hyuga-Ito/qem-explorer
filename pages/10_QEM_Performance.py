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
    W_ = pd.read_csv(os.path.join(t, "rmse_wide.csv"), encoding="utf-8")
    R_ = pd.read_csv(os.path.join(t, "regimes.csv"), encoding="utf-8") if os.path.exists(os.path.join(t, "regimes.csv")) else None
    I_ = pd.read_csv(os.path.join(t, "paired_improvement.csv"), encoding="utf-8") if os.path.exists(
        os.path.join(t, "paired_improvement.csv")) else None
    ev = None
    if os.path.exists(os.path.join(t, "recommender_evaluation.json")):
        with open(os.path.join(t, "recommender_evaluation.json"), encoding="utf-8") as f:
            ev = json.load(f)
    return W_, R_, I_, ev


def study_context(name: str):
    """The study's resolved protocols and context (``tables/methods.json``, schema qemx.methods/2)."""
    from qemx import rec_context as RX
    return RX.load_study(os.path.join(REPORTS, name))


dirs = study_dirs()
TABS = ["Study results", "Recommend for my circuit", "Drill-down analyses", "Feature dictionary",
        "Protocol selection study"]
tabs = S.page_tabs("qp", "w_qp_tabs", TABS)            # the selected tab is app state (survives page switches)

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

def legacy_recommender():
    """The 1.5 Direction-A baseline recommender (a different model, data set, candidates and budget definition),
    kept as a clearly identified legacy pathway during the migration to the 1.7 selection recommender."""
    if True:
        st.markdown("A **transparent baseline**: one ridge model per method predicts log₁₀ RMSE from features that are "
                    "known before running anything (circuit, observable, noise model, budget). It recommends a "
                    "protocol exactly as the study ran it (estimator, readout mitigation, calibration, training) only "
                    "when the prediction separates it from the alternatives; it recommends the study's **baseline "
                    "protocol** when no method is predicted to beat it by more than 10 %, and it **abstains** outside "
                    "the training data or when this experiment's noise context (device model, scale, readout, "
                    "observable, budget) is not the study's. It is not an optimal-QEM oracle: see its held-out "
                    "evaluation in the study report.")
        st.caption("In the bundled pilot this baseline does **not** beat simply using the best fixed method: on "
                   "held-out base circuits its choices have a larger RMSE than the best fixed method's (study report, "
                   "Figure 6), and neither calibrated-noise nor exact-state features change that. Treat a "
                   "recommendation as a hypothesis to check, not as advice.")
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
                              help="circuit_observable: circuit, observable and budget only (no noise model); "
                                   "structural: adds light-cone fault rates, readout and scale under the selected "
                                   "noise model; normalized: per-qubit/per-gate versions; with_state: adds exact "
                                   "entanglement features (statevector 16·2^n bytes, time O(2^(3n/2))).")
            S.wdefault("w_qp_B", int(ss.get("shots", 4096)))
            B = c3.number_input("Total shot budget B", min_value=100, max_value=10 ** 9, step=1000, key="w_qp_B",
                                persist_state=P)

            def do_recommend():
                """Resolve the study's protocols and this experiment's noise context, check that they are compatible
                (before any prediction), compute the features under the experiment's own noise model, cost every
                protocol under the study's cost policy, then predict."""
                from qemx import rec_context as RX
                ss.qp_rec = None
                ss.qp_err = ""
                try:
                    mt_ = os.path.getmtime(os.path.join(REPORTS, name, "tables", "rmse_wide.csv"))
                    W_, _, _, _ = load_tables(name, mt_)
                    names = list(FT.SETS[fs])
                    study = study_context(name)
                    comp, err = C.compiled_now()
                    if comp is None:
                        raise ValueError("the circuit does not compile: %s" % err)
                    live = RX.live_context(comp, ss.backend, ss.noise, ss.observable)
                    base = dict(sig=S.signature(), budget=int(B), study=name, feature_set=fs, context=live.summary(),
                                predicted={}, interval=[math.nan, math.nan], alternatives=[], method=None,
                                protocol="", baseline=study.baseline, baseline_protocol=(
                                    RX.describe_protocol(study, study.baseline) if study.baseline in study.protocols
                                    else ""), features={}, missing={}, notes=[], costs={})
                    problems, notes = RX.compatibility(study, live, float(B), names, list(W_.columns))
                    base["notes"] = notes
                    if problems:                         # the model does not describe this protocol / noise context
                        ss.qp_rec = dict(base, kind="abstain-context", reason="; ".join(problems),
                                         problems=problems)
                        return
                    cols, _, _ = FT.resolve_columns(W_.columns, names, study.feature_schema)
                    methods = [m for m in study.candidates if "rmse_" + m in W_ and "feas_" + m in W_]
                    W_ = W_.assign(x_log10_budget=np.log10(W_["B"].astype(float)))
                    cols["log10_budget"] = "x_log10_budget"
                    X = W_[[cols[n] for n in names]].to_numpy(float)
                    Y = np.column_stack([np.log10(W_["rmse_" + m].astype(float).clip(lower=1e-12)) for m in methods])
                    F = np.column_stack([W_["feas_" + m].astype(bool) & np.isfinite(W_["rmse_" + m].astype(float))
                                         for m in methods])
                    # out-of-fold residuals by base circuit: group-CV conformal intervals (qemx.recommend)
                    model = RC.RidgeRecommender(methods, alpha=1.0).fit(
                        X, Y, F, W_["group"].to_numpy() if "group" in W_ else None)
                    v, miss = RX.live_features(live, float(B), names)
                    x = np.array([v.get(n, math.nan) for n in names], float)
                    # feasibility of each resolved protocol on this circuit, under the study's cost policy
                    costs = RX.protocol_costs(study, live, int(B), methods, symmetry=ss.get("symmetry"))
                    r = model.recommend(x, [bool(costs[m]["feasible"]) for m in methods])
                    ss.qp_rec = dict(base, kind=r.kind, method=r.method, reason=r.reason, predicted=r.predicted,
                                     interval=list(r.interval), alternatives=list(r.alternatives),
                                     protocol=RX.describe_protocol(study, r.method, costs.get(r.method))
                                     if r.method else "",
                                     features={n: v.get(n) for n in names},
                                     missing={n: miss[n] for n in names if n in miss}, costs=costs)
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
                    st.success("**%s** — predicted RMSE %.3g (nominal 90 %% interval %.3g – %.3g, from held-out "
                               "residuals of whole base circuits; it can undercover for an unlike circuit). "
                               "Alternatives: %s." % (
                        W.md_escape(rec["method"]), 10 ** rec["predicted"][rec["method"]], 10 ** lo, 10 ** hi,
                        ", ".join(rec["alternatives"]) or "none"), icon=":material/recommend:")
                    st.caption("Protocol, exactly as the study ran it: %s." % W.md_escape(rec["protocol"]))
                elif rec["kind"] == "raw":
                    st.info("**Use the study's baseline protocol** — %s — %s." % (
                        W.md_escape(rec["protocol"] or rec.get("baseline_protocol") or "RAW"), rec["reason"]),
                        icon=":material/do_not_disturb_on:")
                elif rec["kind"] == "abstain-context":
                    st.warning("**No recommendation** — this experiment is outside the study's protocol or noise "
                               "context:\n\n" + "\n".join("- " + W.md_escape(p) for p in rec.get("problems", [])),
                               icon=":material/block:")
                else:
                    st.warning("**No recommendation** (%s): %s." % (rec["kind"], rec["reason"]), icon=":material/help:")
                ctxs = rec.get("context") or {}
                st.caption("Noise context of the features: %s%s." % (
                    W.md_escape(ctxs.get("backend", "")),
                    (" · device %s, scale s = %g, coherent fraction %g, readout %s" % (
                        ctxs.get("device"), ctxs.get("scale"), ctxs.get("coherent_fraction"), ctxs.get("readout")))
                    if ctxs.get("kind") == "device" else " · noise injector (no device scale)"))
                for nt in rec.get("notes") or []:
                    st.caption(W.md_escape(nt))
                if rec["predicted"]:
                    st.dataframe(pd.DataFrame(sorted(rec["predicted"].items(), key=lambda kv: kv[1]),
                                              columns=["method", "predicted log10 RMSE"]), hide_index=True)
                if rec.get("costs"):
                    st.dataframe(pd.DataFrame([dict(method=k, protocol=c.get("title", ""),
                                                    readout=c.get("readout", ""),
                                                    calibration_circuits=c.get("calibration_circuits"),
                                                    training_shots=c.get("training_shots"),
                                                    min_budget=c.get("min_budget"), feasible=c.get("feasible"),
                                                    note=c.get("error", ""))
                                               for k, c in rec["costs"].items()]), hide_index=True, width="stretch")
                if rec["features"]:
                    st.dataframe(pd.DataFrame([dict(feature=k, value=val, missing=rec["missing"].get(k, ""))
                                               for k, val in rec["features"].items()]), hide_index=True,
                                 width="stretch")
                st.caption("Feature weights are associations within the study's design, not causes. Feasibility uses "
                           "each protocol's fixed cost on this circuit under the study's cost policy (minimum budget "
                           "m + T·s_tr + c·s_min); target circuits counted before running are a lower bound for ZNE "
                           "and NIL.")

with tabs[1]:
    if tabs[1].open is not False:
        from ui import recommend_view as RV
        RV.render()
        with st.expander("Legacy 1.5 recommender (Direction A study: another model, data, candidate set and budget "
                         "definition)", icon=":material/history:"):
            st.caption("Kept during the migration. It is **not** the 1.7 selection recommender above: it predicts "
                       "RMSE of the 1.5 study's method presets from the Direction A pilot, under that study's cost "
                       "policy.")
            legacy_recommender()

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

with tabs[4]:
    if tabs[4].open is not False:
        st.markdown("The study of budget-aware **protocol selection**: complete protocols (calibration, training, "
                    "allocation and failure handling included) compared at equal total budget on emulated "
                    "fake-device trials, with learned and fixed selection policies. The corrected pilot of 1.7 is "
                    "listed first; the 1.6 pilot is kept as a historical study.")
        from ui import selection_view as SV
        ok, _ = W.guarded("Reading the selection study", SV.render, REPORTS)
