"""The selection recommender for the Playground's circuit (QEM Performance page, "Recommend for my circuit"; 1.7.0,
preflight contract and capability records 1.8.0).

The Playground state becomes a ``qemx.selection.deploy.DeploymentTask`` — the same representation the command line
uses (``python -m qemx.selection recommend --state …``), so the UI and the CLI resolve identical experiments,
features and recommendations. The recommendation is cached under the task's key and the bundle's digest: any change
of circuit, input, bases, bit order, observable, declared symmetry, device, budget, bundle or simulation permission
marks it out of date (1.9.0: ``deploy.recommendation_signature``). The selected "Device snapshot" is the effective
device (the same rule as ``recommend --device``); the result shows it with the physical layout and the symmetry used."""
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


def signature_now(budget: int, device: str, bundle, allow_simulation: bool) -> str:
    """The signature a recommendation made now would have (``deploy.recommendation_signature``)."""
    from qemx.selection import deploy as DP
    return DP.recommendation_signature(current_task(budget, device), bundle, allow_simulation=allow_simulation)


def render():
    from qemx import device as DV
    from qemx.selection import deploy as DP
    ss = st.session_state
    names = DP.bundles()
    st.markdown("**Selection recommender** — a versioned model bundle trained in the protocol-selection study "
                "(fake-device emulation): for this circuit, input, measurement, observable, device and total budget B "
                "it predicts each complete protocol's log risk from features computed *without simulating the "
                "circuit*, checks that the candidate can actually run as defined (preflight: *verified*, "
                "*infeasible* or *unverified* — never accepted while unverified), recommends the lowest verified one "
                "only when the frozen selective rule accepts it, and otherwise falls back to the study's budget-only "
                "policy SBS-B.")
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
                       help="The calibration snapshot the protocols would use (it replaces the Playground's backend "
                            "for the recommendation; the Playground's backend is preselected when it is a fake "
                            "device). The physical layout of the state is checked on the chosen device.")
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
        task = None
        try:
            task = current_task(int(B), dev)
            rec = DP.recommend(task, b, allow_simulation=bool(ss.get("w_sr_sim")))
            ss.sr_rec = dict(rec, key=task.key(), bundle_digest=b.meta.get("arrays_sha256"))
        except DP.Unsupported as e:                  # 1.9.0: keep the signature when the task itself was built
            sig = DP.recommendation_signature(task, b, allow_simulation=bool(ss.get("w_sr_sim"))) if task else None
            ss.sr_rec = dict(status="unsupported", reason=str(e), key=None, bundle_digest=b.meta.get("arrays_sha256"),
                             signature=sig)
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
    try:                                             # 1.9.0: the signature includes the simulation permission
        sig_now = signature_now(int(B), dev, b, bool(ss.get("w_sr_sim")))
    except Exception:                                # noqa: BLE001 - an unsupported state has no signature
        sig_now = None
    if rec.get("signature") != sig_now or rec.get("bundle_digest") != b.meta.get("arrays_sha256"):
        st.warning("The circuit, input, measurement, observable, declared symmetry, device, budget, bundle or the "
                   "simulation permission changed since this recommendation; press Recommend again.",
                   icon=":material/update:")
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
    sym = rec.get("symmetry") or {}
    st.caption(":material/memory: Effective device **%s** · requested layout %s · physical qubits used %s · declared "
               "symmetry: %s" % (W.md_escape(str(rec.get("effective_device"))), rec.get("requested_layout") or "none",
                                 rec.get("physical_layout") or "—",
                                 W.md_escape(sym.get("declared") or "none") + (" (%s)" % sym["source"]
                                                                               if sym.get("source") else "")))
    for x in rec.get("conversion_notes") or []:
        st.caption(":material/info: " + W.md_escape(x))
    pd_ = rec.get("protocol_definition") or {}
    if pd_:
        st.caption("Protocol, exactly as the study defines it: %s" % W.md_escape(pd_.get("description", "")))
    ex = rec.get("execution") or {}
    if ex.get("note"):
        st.caption(":material/fact_check: " + W.md_escape(ex["note"]) + " (the design whose training labels the "
                   "preflight verified).")
    for x in rec.get("skipped_candidates") or []:
        st.caption(":material/skip_next: The model ranked **%s** higher, but its preflight is *%s*: %s" % (
            x["protocol"], x["status"], W.md_escape(x["reason"])))
    try:
        from qemx.selection import registry as RG
        cap = RG.describe(rec["choice"])
        with st.expander("Capability record of %s (what it needs, costs and does on failure)" % rec["choice"],
                         icon=":material/list_alt:"):
            st.dataframe(pd.DataFrame([dict(item=k, value=", ".join(v) if isinstance(v, list) else str(v))
                                       for k, v in cap.items() if k not in ("definition",)]),
                         hide_index=True, width="stretch")
    except KeyError:
        pass
    st.caption("Model %s · bundle %s · the model's choice: %s (margin %s, threshold τ = %s). Predicted risks are "
               "model-conditional (they hold only as far as the model is right); the nested cross-validation "
               "numbers in the bundle are empirical, for the study's task distribution." % (
                   W.md_escape(rec["configuration"]), W.md_escape(rec["bundle"]), rec.get("model_choice"),
                   "%.2f" % rec["margin"] if isinstance(rec.get("margin"), float) and math.isfinite(rec["margin"])
                   else rec.get("margin"), rec.get("tau")))
    if rec.get("predicted_log10_risk"):
        rows = [dict(protocol=p, predicted_log10_risk=v, preflight=rec["feasibility"][p].get("status"),
                     level=rec["feasibility"][p].get("level"), note=rec["feasibility"][p]["note"])
                for p, v in sorted(rec["predicted_log10_risk"].items(), key=lambda kv: kv[1])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Features, their information class and cost", icon=":material/tune:"):
        fr = pd.DataFrame(rec["feature_report"]["rows"])
        fr["value"] = fr.feature.map(rec["features"])
        st.dataframe(fr[["feature", "value", "info", "seconds", "computed", "method"]], hide_index=True,
                     width="stretch")
        st.caption("Compilation %.2f s; the recommendation %.2f s in total. Feasibility: %s." % (
            rec["feature_report"]["compile_seconds"], rec["seconds"], W.md_escape(rec["feasibility_method"])))
