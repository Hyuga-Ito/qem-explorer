"""Noise Models & Discrepancy: where a specified noise model fails to predict circuit behavior, with matched
protocols, calibrated statistics and the link to QEM selection. Reads stored divergence/rehearsal results; the
calculators work on numbers you type (validated, nothing executed)."""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Noise Models & Discrepancy · QEM Explorer", page_icon=":material/difference:",
                   layout="wide")

from qemx import discrepancy as DS  # noqa: E402
from ui import state as S  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar(run_label=None)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.title("Noise Models & Discrepancy")
st.markdown("A noise model built from component-level characterization predicts every circuit's output; the device "
            "(or an emulator) produces observations of **the same compiled circuits**, processed by **the same** "
            "estimator. A discrepancy rejects the claim that the model fits *for those circuits*; it does not by "
            "itself identify a physical mechanism. Distances are computed only between genuine probability "
            "distributions — never on signed quasi-probabilities — and many simultaneous tests are corrected "
            "(Holm for the family-wise error rate).")

TABS = ["Rehearsal results", "Conventions", "Distance test", "Drill-down: model mismatch"]
tabs = S.page_tabs("nd", "w_nd_tabs", TABS)            # the selected tab is app state (survives page switches)


def rehearsal_dirs() -> list:
    base = os.path.join(ROOT, "reports")
    out = []
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if os.path.exists(os.path.join(base, d, "rehearsal.json")):
                out.append(d)
    return out


with tabs[0]:
    if tabs[0].open is not False:
        ds = rehearsal_dirs()
        if not ds:
            st.info("No rehearsal under `reports/`. Run `python scripts/rehearse_divergence.py "
                    "manifests/direction_A_divergence_v3_qpu600.yaml --out reports/rehearsal_v3` (offline; it labels "
                    "every result EMULATION).", icon=":material/folder_open:")
        else:
            S.wdefault("w_nd_dir", ds[0])
            if ss.w_nd_dir not in ds:
                ss.w_nd_dir = ds[0]
            name = st.selectbox("Rehearsal", ds, key="w_nd_dir", persist_state=P)
            with open(os.path.join(ROOT, "reports", name, "rehearsal.json"), encoding="utf-8") as f:
                rj = json.load(f)
            st.warning("**%s.** Synthetic truth through the hardware path (plan → counts → analysis)." % rj.get(
                "label", "EMULATION"), icon=":material/science:")
            fp = rj.get("false_positive", {})
            c = st.columns(4)
            c[0].metric("Holm flags under the null", "%d / %d" % (fp.get("holm", {}).get("count", 0),
                                                                 fp.get("holm", {}).get("n", 0)), border=True,
                        help="Rows flagged when the truth is the predictor's own model: false positives.")
            c[1].metric("Unadjusted p < 0.05", "%.3f" % fp.get("p_lt_alpha", {}).get("rate", float("nan")),
                        border=True, help="Should be ≈ 0.05 for a calibrated test.")
            c[2].metric("z mean / sd (null)", "%.2f / %.2f" % (rj.get("z_calibration", {}).get("mean", np.nan),
                                                              rj.get("z_calibration", {}).get("sd", np.nan)),
                        border=True, help="Calibrated: 0 / 1. An sd below 1 is conservative.")
            u = rj.get("planned_usage") or {}
            c[3].metric("Planned QPU usage (high)", "%.0f s" % u.get("high_s", np.nan), border=True,
                        help="The conservative planner's high estimate for this manifest (qemx.campaign).")
            rows = []
            for sc, ms in (rj.get("power") or {}).items():
                for m, v in ms.items():
                    rows.append(dict(alternative=sc.replace("alternative: ", ""), method=m,
                                     flagged=v["holm_worse"]["count"], circuits=v["holm_worse"]["n"],
                                     power=v["holm_worse"]["rate"], se=v["holm_worse"]["se"]))
            if rows:
                st.subheader("Detection power at this budget (Holm, predicted fidelity above observed)")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            md = os.path.join(ROOT, "reports", name, "rehearsal.md")
            if os.path.exists(md):
                with st.expander("Rehearsal report", icon=":material/description:"):
                    with open(md, encoding="utf-8") as f:
                        st.markdown(f.read())

with tabs[1]:
    if tabs[1].open is not False:
        st.markdown("An RB error per gate is an **average gate infidelity** r (under gate-independent noise); Chen et "
                    "al. report the **entanglement infidelity** ε; a simulator's depolarizing channel "
                    "ρ ↦ (1−p)ρ + p·I/d takes **p**. For d = 2ᵏ: ε = p(1 − 1/d²), r = p(d − 1)/d = ε·d/(d + 1).")
        S.wdefault("w_nd_val", 0.0075)
        S.wdefault("w_nd_from", "average_gate_infidelity")
        S.wdefault("w_nd_k", 2)
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("value", min_value=0.0, max_value=1.0, step=0.0005, format="%.6f", key="w_nd_val",
                              persist_state=P)
        frm = c2.selectbox("convention", DS.RATE_KINDS, key="w_nd_from", persist_state=P)
        k = c3.selectbox("gate qubits", [1, 2], key="w_nd_k", persist_state=P)
        try:
            st.dataframe(pd.DataFrame([dict(convention=t, value=DS.convert_error_rate(val, frm, t, int(k)))
                                       for t in DS.RATE_KINDS]), hide_index=True)
        except ValueError as e:
            st.error(W.md_escape(str(e)), icon=":material/error:")

with tabs[2]:
    if tabs[2].open is not False:
        st.markdown("Is an observed histogram consistent with a model's distribution at this many shots? The plug-in "
                    "distance is compared with its sampling distribution **under the model** (parametric bootstrap): "
                    "a nonzero distance alone is not a discrepancy. Enter probabilities and counts over the same "
                    "outcomes (comma-separated).")
        S.wdefault("w_nd_p", "0.5, 0.0, 0.0, 0.5")
        S.wdefault("w_nd_c", "480, 12, 9, 499")
        S.wdefault("w_nd_metric", "tvd")
        c1, c2 = st.columns(2)
        pt = c1.text_input("model probabilities", key="w_nd_p", max_chars=2000, persist_state=P)
        ct = c2.text_input("observed counts", key="w_nd_c", max_chars=2000, persist_state=P)
        metric = st.segmented_control("metric", ["tvd", "hellinger", "classical_fidelity"], key="w_nd_metric",
                                      persist_state=P)

        def parse(txt, integer=False):
            vals = [x.strip() for x in str(txt).split(",") if x.strip()]
            if not vals or len(vals) > 4096:
                raise ValueError("give between 1 and 4096 comma-separated numbers")
            out = [int(x) if integer else float(x) for x in vals]
            if integer and any(v < 0 for v in out):
                raise ValueError("counts must be ≥ 0")
            return np.asarray(out, float)
        try:
            p = parse(pt)
            cnt = parse(ct, integer=True)
            if p.shape != cnt.shape:
                raise ValueError("the model and the counts must list the same number of outcomes")
            N = int(cnt.sum())
            if N < 1:
                raise ValueError("no shots")
            res = DS.distance_test(p, cnt / N, N, metric or "tvd", reps=2000, rng=0)
            m = st.columns(3)
            m[0].metric(metric or "tvd", "%.4f" % res["observed"], border=True)
            m[1].metric("under the model: mean / 95 %%", "%.4f / %.4f" % (res["null_mean"], res["null_q95"]),
                        border=True)
            m[2].metric("p-value", "%.4f" % res["p_value"], border=True)
            if res["sparse"]:
                st.warning("Fewer than 5 shots per outcome on average: the full histogram is a weak test. Compare a "
                           "marginal or an observable instead.", icon=":material/warning:")
        except (ValueError, OverflowError) as e:
            st.error(W.md_escape(str(e)[:300]), icon=":material/error:")

with tabs[3]:
    if tabs[3].open is not False:
        st.markdown("What a wrong noise model does to PEC and readout mitigation on the current circuit:")
        st.page_link("pages/4_Model_Mismatch.py", label="Model mismatch", icon=":material/tune:")
        st.caption("For hardware divergence studies use `scripts/dirA_divergence.py` (plan → run-ibm/emulate → "
                   "analyze); the campaign ledger (`python -m qemx.campaign`) bounds QPU usage.")
