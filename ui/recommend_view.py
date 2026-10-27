"""The 1.7 selection recommender for the Playground's circuit (QEM Performance page, "Recommend for my circuit").

The Playground state becomes a ``qemx.selection.deploy.DeploymentTask`` — the same representation the command line
uses (``python -m qemx.selection recommend --state …``), so the UI and the CLI resolve identical experiments,
features and recommendations. The recommendation is cached under the task's key and the bundle's digest: any change
of circuit, input, bases, bit order, observable, device or budget marks it out of date."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from ui import state as S
from ui import widgets as W
from ui.widgets import P


def _bundle(name: str):
    from qemx.selection import deploy as DP
    import os
    return DP.load_bundle(os.path.join(DP.BUNDLE_DIR, name))


def current_task(budget: int, device: str):
    """The deployment task of the Playground's current state (raises deploy.Unsupported outside the task class)."""
    from qemx.selection import deploy as DP
    return DP.from_state(S.to_json(), int(budget), device)


def render():
    from qemx import device as DV
    from qemx.selection import deploy as DP
    ss = st.session_state
    names = DP.bundles()
    st.markdown("**Selection recommender (1.7)** — a versioned model bundle trained in the protocol-selection study "
                "(fake-device emulation): for this circuit, input, measurement, observable, device and total budget B "
                "it predicts each complete protocol's log risk from features computed *without simulating the "
                "circuit*, recommends the lowest only when the frozen selective rule accepts it, and otherwise "
                "falls back to the study's budget-only policy SBS-B.")
    if not names:
        st.info("No model bundle is installed (qemx/selection/bundles). Build one with `python -m qemx.selection "
                "bundle CONFIG --out RUN`.", icon=":material/inventory_2:")
        return
    S.wdefault("w_sr_bundle", names[0])
    if ss.w_sr_bundle not in names:
        ss.w_sr_bundle = names[0]
    c1, c2, c3 = st.columns(3)
    bname = c1.selectbox("Model bundle", names, key="w_sr_bundle", persist_state=P)
    ok, b = W.guarded("Reading the model bundle", _bundle, bname)
    if not ok:
        return
    devs = list(b.meta["support"]["devices"])
    play_dev = ss.backend if DV.is_device(ss.backend) else None
    opts = ([play_dev] if play_dev and play_dev not in devs else []) + devs
    S.wdefault("w_sr_dev", play_dev or devs[0])
    if ss.w_sr_dev not in opts:
        ss.w_sr_dev = opts[0]
    dev = c2.selectbox("Device snapshot", opts, key="w_sr_dev", persist_state=P,
                       help="The calibration snapshot the protocols would use. The Playground's backend is used when "
                            "it is a fake device; the training devices are listed.")
    S.wdefault("w_sr_B", 100000)
    B = c3.number_input("Total shot budget B", min_value=100, max_value=10 ** 9, step=1000, key="w_sr_B",
                        persist_state=P)
    S.wdefault("w_sr_sim", False)
    needs_sim = [n for n in b.features if b.meta["feature_classes"].get(n) == "simulated"]
    st.toggle("Allow simulated features (exact simulation of the snapshot; cost grows exponentially with width)",
              key="w_sr_sim", persist_state=P, disabled=not needs_sim,
              help="Only needed when the bundle's model uses features of class 'simulated'. %s" % (
                  "This bundle uses: " + ", ".join(needs_sim) if needs_sim else "This bundle uses none."))
    if play_dev is None:
        st.caption("The Playground runs on a noise injector; the recommendation is for the same circuit on the chosen "
                   "device snapshot, compiled for it.")

    def go():
        ss.sr_rec, ss.sr_err = None, ""
        try:
            task = current_task(int(B), dev)
            rec = DP.recommend(task, b, allow_simulation=bool(ss.get("w_sr_sim")))
            ss.sr_rec = dict(rec, key=task.key(), bundle_digest=b.meta.get("arrays_sha256"))
        except DP.Unsupported as e:
            ss.sr_rec = dict(status="unsupported", reason=str(e), key=None, bundle_digest=b.meta.get("arrays_sha256"))
        except MemoryError:
            ss.sr_err = "Not enough memory to compute the requested features for this circuit."
        except Exception as e:                       # noqa: BLE001 - never crash the page from a callback
            ss.sr_err = "Could not compute a recommendation: %s: %s" % (type(e).__name__, str(e)[:300])
    st.button("Recommend", type="primary", icon=":material/recommend:", key="sr_go", on_click=go)
    if ss.get("sr_err"):
        st.error(W.md_escape(ss.sr_err), icon=":material/error:")
    rec = ss.get("sr_rec")
    if not rec:
        return
    try:
        key_now = current_task(int(B), dev).key()
    except Exception:                                # noqa: BLE001 - an unsupported state has no key
        key_now = None
    if rec.get("key") != key_now or rec.get("bundle_digest") != b.meta.get("arrays_sha256"):
        st.warning("The circuit, input, measurement, observable, device, budget or bundle changed since this "
                   "recommendation; press Recommend again.", icon=":material/update:")
    if rec["status"] == "unsupported":
        st.warning("**Unsupported** — " + W.md_escape(rec["reason"]), icon=":material/block:")
        return
    if rec.get("accepted"):
        st.success("**%s** — %s" % (rec["choice"], W.md_escape(rec["reason"])), icon=":material/recommend:")
    elif rec["status"] == "out-of-distribution":
        st.warning("**%s (fallback SBS-B)** — %s" % (rec["choice"], W.md_escape(rec["reason"])) + "\n\n" +
                   "\n".join("- " + W.md_escape(x) for x in rec.get("ood", [])), icon=":material/travel_explore:")
    else:
        st.info("**%s (fallback SBS-B)** — %s" % (rec["choice"], W.md_escape(rec["reason"])),
                icon=":material/do_not_disturb_on:")
    for w in rec.get("warnings", []):
        st.caption(":material/warning: " + W.md_escape(w))
    pd_ = rec.get("protocol_definition") or {}
    if pd_:
        st.caption("Protocol, exactly as the study defines it: %s" % W.md_escape(pd_.get("description", "")))
    st.caption("Model %s · bundle %s · the model's choice: %s (margin %s, threshold τ = %s). Predicted risks are "
               "model-conditional (they hold only as far as the model is right); the nested cross-validation "
               "numbers in the bundle are empirical, for the study's task distribution." % (
                   W.md_escape(rec["configuration"]), W.md_escape(rec["bundle"]), rec.get("model_choice"),
                   "%.2f" % rec["margin"] if isinstance(rec.get("margin"), float) and math.isfinite(rec["margin"])
                   else rec.get("margin"), rec.get("tau")))
    if rec.get("predicted_log10_risk"):
        rows = [dict(protocol=p, predicted_log10_risk=v, feasible=rec["feasibility"][p]["feasible"],
                     note=rec["feasibility"][p]["note"]) for p, v in sorted(rec["predicted_log10_risk"].items(),
                                                                          key=lambda kv: kv[1])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Features, their information class and cost", icon=":material/tune:"):
        fr = pd.DataFrame(rec["feature_report"]["rows"])
        fr["value"] = fr.feature.map(rec["features"])
        st.dataframe(fr[["feature", "value", "info", "seconds", "computed", "method"]], hide_index=True,
                     width="stretch")
        st.caption("Compilation %.2f s; the recommendation %.2f s in total. Feasibility: %s." % (
            rec["feature_report"]["compile_seconds"], rec["seconds"], W.md_escape(rec["feasibility_method"])))
