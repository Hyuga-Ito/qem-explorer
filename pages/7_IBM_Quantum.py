"""Run the Playground's recipes on IBM Quantum hardware, or test the whole pipeline locally on a fake backend.

Security: the API token is typed into a password field inside a form. It is read once, in the submit callback,
used to open the connection, and the field is cleared before the page reruns; it is never written to the model,
the state file, the exported script, logs or messages. Only job IDs are kept."""
import logging

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Run on IBM Quantum · QEM Explorer", page_icon=":material/cloud_upload:", layout="wide")

from qemx import device  # noqa: E402
from qemx import hardware as H  # noqa: E402
from qemx.compile import Compiled, with_io  # noqa: E402
from qemx.methods import REGISTRY  # noqa: E402
from qemx.resources import RunTimeout, deadline, explain, scrub  # noqa: E402
from ui import common as C  # noqa: E402
from ui import results as RES  # noqa: E402
from ui import state as S  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

LOG = logging.getLogger("qem_explorer")
S.init()
ss = st.session_state
sidebar(run_label=None)
st.title("Run on IBM Quantum")
st.markdown("Send the Playground's contenders to a real IBM Quantum processor — or rehearse the complete pipeline "
            "locally on a fake backend first. The app records the circuits each estimator needs, gives every "
            "contender the same shot budget split exactly as in simulation, runs the circuits with the Sampler "
            "primitive and replays the same estimator code on the measured counts. The results page is the "
            "Playground's: the same figures, tables, analysis views and ZIP.")
compiled_sim, err = C.compiled_now()
C.context_line(compiled_sim)
W.why("hardware")

recipes = S.recipes()
support = [(r, *H.supported(r)) for r in recipes]
with st.expander("Contenders and what runs on hardware", expanded=False, key="exp_hw_rc"):
    st.dataframe(pd.DataFrame([{"contender": r.title(), "runs on hardware": "yes" if ok else "no",
                                "note": why} for r, ok, why in support]), hide_index=True, width="stretch")
    st.caption("Change the contenders on the Playground's Mitigate & compare tab.")

_K = next(p for p in REGISTRY["pec"].params if p.name == "instances")      # PEC's own bounds for K
K_LO, K_HI = int(_K.lo or 1), int(_K.hi or 1024)
for k, v in (("w_hw_mode", "local"), ("w_hw_fake", device.base_name(ss.backend) if device.is_device(ss.backend)
                                       else "fake_fez"), ("w_hw_pec_inst", 64), ("w_hw_opt", 1)):
    S.wdefault(k, v)
if ss.w_hw_fake not in device.FAKE_DEVICES:
    ss.w_hw_fake = "fake_fez"
try:
    ss.w_hw_pec_inst = min(max(int(ss.w_hw_pec_inst), K_LO), K_HI)
except (TypeError, ValueError):
    ss.w_hw_pec_inst = 64
st.segmented_control("Where", ["local", "ibm"], key="w_hw_mode", persist_state=P,
                     format_func={"local": "Local test on a fake backend (no account)",
                                  "ibm": "IBM Quantum hardware"}.get)
mode = ss.get("w_hw_mode") or "local"
c = st.columns(3)
c[0].number_input("PEC sampled circuits K", min_value=K_LO, max_value=K_HI, step=8, key="w_hw_pec_inst",
                  persist_state=P,
                  help="PEC's target shots $N_t$ are split evenly over $K$ sampled circuits: $s=N_t/K$ shots each. "
                       "The estimator's variance includes the instance-to-instance term $(s-1)\\,\\mathrm{Var}_{\\rm inst}$, "
                       "so fewer circuits cost precision, not correctness.")
c[1].select_slider("transpiler optimization level", [0, 1, 2, 3], key="w_hw_opt", persist_state=P)
c[2].metric("shot budget per contender", f"{int(ss.shots):,}",
            help="The sidebar's budget $B$: each contender's target, training and calibration circuits share it, "
                 "split as in simulation (qemx/budget.py).")


def hw_recipes():
    """The contenders with the page's K for PEC. Contenders that become identical (PEC recipes that differed only
    in K) are sent once: each would otherwise be measured, and billed, separately."""
    from qemx.methods import Recipe
    out, seen = [], set()
    for r, ok, _ in support:
        if r.estimator == "pec":
            p = dict(r.p)
            p["instances"] = int(ss.w_hw_pec_inst)
            r = Recipe.make("pec", r.twirl, r.dd, r.readout, **p)
        if r.key() not in seen:
            seen.add(r.key())
            out.append(r)
    return out


def layout_for(backend_name: str, num_qubits: int) -> tuple:
    """(forced initial layout or None, note). The Playground's physical qubits apply only to the device they were
    chosen for (its fake twin for real hardware) and only if they fit; otherwise the compiler chooses."""
    try:
        init = S.parse_layout(ss.compile.get("initial_layout", ""))
    except ValueError:
        return None, ""
    if not init:
        return None, ""
    if not device.is_device(ss.backend):
        return None, "The Playground's forced layout belongs to a lattice, not to %s; the compiler chose the qubits." \
            % backend_name
    base = device.base_name(ss.backend)
    if backend_name not in (base, device.fake_class(ss.backend), "ibm_" + base[len("fake_"):]):
        return None, ("The Playground's forced layout was chosen for %s, not %s; the compiler chose the qubits."
                      % (device.device_label(ss.backend), backend_name))
    if len(init) != int(ss.n) or len(set(init)) != len(init) or max(init) >= num_qubits:
        return None, "The Playground's forced layout does not fit this circuit on %s; the compiler chose the qubits." \
            % backend_name
    return tuple(init), ""


def compile_for(backend_obj, tag, name=None):
    logical = with_io(S.spec(), ss.inputs, ss.basis)
    init, note = layout_for(name or getattr(backend_obj, "name", tag), int(backend_obj.num_qubits))
    spec, mm, phys, info = device.compile_for_target(logical, backend_obj, device.good_target_for(backend_obj), tag,
                                                     int(ss.w_hw_opt), int(ss.compile.get("seed", 11)),
                                                     ss.compile.get("layout", "sabre"), init)
    if ss.relabel:
        n = logical.n
        mm = {i: mm[n - 1 - i] for i in range(n)}
    info["layout_note"] = note
    return Compiled(logical, spec, mm, info)


def with_simulation(res, sig):
    """Add the Playground's simulated estimate of each contender — only when that simulation was run for the very
    state that was sent (``sig``), never for settings edited since."""
    sim = ss.get("results")
    if sim is None or sim.get("sig") != sig:
        return res
    simv = {r["method"]: r["estimate"] for r in sim["rows"]}
    for r in res["rows"]:
        if r["method"] in simv:
            r["simulated (Playground)"] = simv[r["method"]]
    return res


def plan_view(plan, key):
    with st.expander("Shot allocation: how each contender spends the budget", icon=":material/account_balance:",
                     key="exp_plan_" + key):
        st.dataframe(pd.DataFrame(plan.table()), hide_index=True, width="stretch")
        st.caption("Planned from the device's calibrated error rates: ZNE splits its target shots N_j ∝ |η_j|σ_j, "
                   "readout mitigation gives each calibration circuit s* shots, PEC runs K circuits with N_t/K "
                   "shots each (identical sampled circuits are merged into one with their shots added, so "
                   "*circuits* counts what is actually sent), CDR / FR / NIL spend their training shots. Contenders never "
                   "share counts. "
                   + " ".join(plan.notes))


def results_view(res, plan, title, key, extra=None, state=None, sig=None):
    st.markdown("**%s**" % W.md_escape(title))
    if sig is not None and sig != S.signature():
        st.caption(":material/history: Sent before the latest changes in the Playground; the ZIP describes the state "
                   "that was sent.")
    plan_view(plan, key)
    tables = {"hardware_plan": pd.DataFrame(plan.table()), "hardware_requests": pd.DataFrame(plan.request_table())}
    tables.update(extra or {})
    RES.render(res, kp=key + "_", bundle=RES.bundle_factory(res, state or S.to_json(), title, title=title,
                                                            extra_tables=tables),
               zip_name="qem_explorer_%s.zip" % key)


# ================================================================== local rehearsal
if mode == "local":
    st.selectbox("Fake backend", list(device.FAKE_DEVICES), key="w_hw_fake", format_func=device.device_label,
                 persist_state=P)
    st.caption("Uses qiskit-ibm-runtime's local testing mode: the same transpiled circuits and Sampler calls as on "
               "hardware, executed by Qiskit Aer with the device's noise snapshot (gate and readout errors of the "
               "qubits the circuits use). Gate twirling and dynamical "
               "decoupling are primitive options that only real hardware applies; locally, twirling is done with "
               "explicit random instances and DD is skipped.")

    def run_local():
        ss.hw_local_run = True
    st.button("Run locally", type="primary", icon=":material/play_arrow:", key="hw_local_btn", on_click=run_local,
              disabled=compiled_sim is None)
    if ss.pop("hw_local_run", False) and compiled_sim is not None:
        sent_sig, sent_state = S.signature(), S.to_json()
        okb, b = W.guarded("Loading %s" % ss.w_hw_fake, device.fake_backend, ss.w_hw_fake)
        ok, comp = W.guarded("Compiling for %s" % b.name, compile_for, b, device.fake_class(ss.w_hw_fake),
                             ss.w_hw_fake) if okb else (False, None)
        if ok:
            ok, plan = W.guarded("Planning", H.make_plan, comp, ss.observable, hw_recipes(), int(ss.shots), ss.symmetry,
                                 0, b)
            if ok:
                bar = st.progress(0.0, text="Running the Sampler locally…")
                ok, freqs = W.guarded("The local run", H.run_local, plan, ss.w_hw_fake, 0,
                                      lambda f: bar.progress(f, text="Running the Sampler locally…"))
                bar.empty()
                if ok:
                    ok, res = W.guarded("The analysis", H.analyze, plan, freqs, float(ss.t_shot_us),
                                        label="local Sampler on %s" % ss.w_hw_fake)
                    if ok:
                        ss.hw_local = dict(res=with_simulation(res, sent_sig), plan=plan, backend=ss.w_hw_fake,
                                           phys=comp.info["phys"], sig=sent_sig, state=sent_state,
                                           note=comp.info.get("layout_note", ""))
    r = ss.get("hw_local")
    if r and "res" in r:
        sm = r["plan"].summary()
        st.caption("%s · physical qubits %s · %d circuits, %s shots in total. %s" % (
            device.device_label(r["backend"]), ", ".join(map(str, r["phys"])), sm["circuits"], f"{sm['shots']:,}",
            W.md_escape(r.get("note", ""))))
        results_view(r["res"], r["plan"], "Local run on %s" % r["backend"], "hw_local", state=r.get("state"),
                     sig=r.get("sig"))

# ================================================================== real hardware
else:
    st.warning("Real hardware uses your IBM Quantum account and QPU time. Start small (few contenders, few PEC "
               "instances) and check the plan below.", icon=":material/warning:")
    st.markdown("**Credentials.** Paste an API key from quantum.cloud.ibm.com. It is sent to this app's server only to "
                "open the connection, cleared from the page immediately, and never stored, logged or included in any "
                "download. Use this only on a deployment you trust; locally, running the exported script with the "
                "`IBM_QUANTUM_TOKEN` environment variable is equally secure.")
    st.caption("A token is a password. Before this notebook goes into a repository or an appendix, clear it and use "
               "one of the other two.")

    def _connect(token: str, channel: str, instance: str):
        from qiskit_ibm_runtime import QiskitRuntimeService
        return QiskitRuntimeService(channel=channel, token=token, instance=instance or None)

    def _campaign_ledger():
        """The thesis campaign's QPU ledger when one is configured (QEMX_CAMPAIGN_LEDGER), else None."""
        import os
        path = os.environ.get("QEMX_CAMPAIGN_LEDGER", "")
        if not path or not os.path.exists(path):
            return None
        from qemx import campaign as CG
        return CG.Ledger.load(path)

    def submit_cb():
        """Runs before the rerun: read the token, clear the field, connect, compile, submit. Nothing secret is kept.
        The sidebar's time limit bounds the planning; jobs submitted before a failure are always kept."""
        token = (ss.get("hw_token") or "").strip()
        ss["hw_token"] = ""
        ss.hw_msg, ss.hw_err = "", ""
        if not token:
            ss.hw_err = "Enter an API token."
            return
        channel = ss.get("hw_channel") or "ibm_quantum_platform"
        sent_sig, sent_state = S.signature(), S.to_json()
        ctx = {}
        try:
            with deadline(ss.get("time_limit", 0)):
                service = _connect(token, channel, (ss.get("hw_instance") or "").strip())
                name = (ss.get("hw_backend") or "").strip()
                n = int(ss.n)
                b = service.backend(name) if name else service.least_busy(operational=True, simulator=False,
                                                                          min_num_qubits=n)
                comp = compile_for(b, b.name)
                plan = H.make_plan(comp, ss.observable, hw_recipes(), int(ss.shots), ss.symmetry, 0, b)
                if not plan.requests:
                    ss.hw_err = "Nothing to run: " + "; ".join("%s: %s" % kv for kv in plan.skipped.items())
                    return
                ctx.update(plan=plan, backend=b.name, phys=comp.info["phys"])
            ledger = _campaign_ledger()
            if ledger is not None and ledger.unsubmitted_reservations():
                ss.hw_err = ("The campaign ledger holds a reservation without a job id (an interrupted submission). "
                             "Resolve it first: python -m qemx.campaign status/resolve.")
                return
            recs = H.submit(plan, b, max_execution_time=int(ss.get("hw_maxtime") or 0) or None, ledger=ledger)
            ss.hw_jobs = dict(records=recs, plan=plan, backend=b.name, phys=comp.info["phys"], channel=channel,
                              sig=sent_sig, state=sent_state, note=comp.info.get("layout_note", ""))
            ss.hw_res, ss.hw_cache = None, {}
            ss.hw_msg = "Submitted %d job(s) to %s: %s" % (len(recs), b.name, ", ".join(r["job_id"] for r in recs))
        except H.SubmitError as e:                       # some jobs went through: keep them, say which
            ss.hw_jobs = dict(records=e.records, channel=channel, sig=sent_sig, state=sent_state, note="", **ctx)
            ss.hw_res, ss.hw_cache = None, {}
            ss.hw_err = scrub("Submission stopped after %d job(s) (%s). The submitted jobs below still run and can be "
                              "collected; the contenders whose jobs were not sent will report missing counts." % (
                                  len(e.records), str(e)[:300]), token)
            LOG.warning("hardware submission stopped part-way: %s", scrub(str(e), token))
        except Exception as e:                           # noqa: BLE001 - the campaign ledger refused the first job
            from qemx.campaign import CampaignLimit
            if not isinstance(e, CampaignLimit):
                title, msg = explain(e, "the submission")
                ss.hw_err = scrub("%s. %s" % (title, msg), token)
                LOG.warning("hardware submission failed: %s", scrub(type(e).__name__ + ": " + str(e), token))
            else:
                ss.hw_err = "Not submitted — campaign QPU ledger: %s" % str(e)[:300]
        except RunTimeout as e:                           # never let the token reach a message
            title, msg = explain(e, "the submission")
            ss.hw_err = scrub("%s. %s" % (title, msg), token)
            LOG.warning("hardware submission failed: %s", scrub(type(e).__name__ + ": " + str(e), token))
        finally:
            token = None                                 # noqa: F841 - drop the last reference

    def fetch_cb():
        token = (ss.get("hw_token2") or "").strip()
        ss["hw_token2"] = ""
        ss.hw_err = ""
        jobs = ss.get("hw_jobs")
        if not jobs:
            ss.hw_err = "No submitted jobs in this session."
            return
        if not token:
            ss.hw_err = "Enter the API token again to read the results (it is not kept between steps)."
            return
        try:
            with deadline(ss.get("time_limit", 0)):
                service = _connect(token, jobs["channel"], (ss.get("hw_instance2") or "").strip())
                cache = ss.setdefault("hw_cache", {})
                freqs, status = H.collect(service, jobs["records"], jobs["plan"].compiled.implemented.n, cache=cache)
                ss.hw_status = status
                failed = [x for x in status if x.get("failed")]
                if freqs is None:
                    if failed and len(failed) == len(status):
                        ss.hw_err = "Every job ended without results: " + ", ".join(
                            "%s %s" % (x["job_id"], x["status"]) for x in failed)
                    else:
                        ss.hw_msg = "Not finished yet: " + ", ".join("%s %s" % (x["job_id"], x["status"])
                                                                     for x in status)
                    return
                ss.hw_res = with_simulation(H.analyze(jobs["plan"], freqs, float(ss.t_shot_us),
                                                      label="IBM Quantum %s" % jobs["backend"]), jobs.get("sig"))
                ss.hw_msg = "Results received."
                if failed:
                    ss.hw_err = ("%d job(s) ended without results (%s); the contenders they measured report missing "
                                 "counts." % (len(failed), ", ".join("%s %s" % (x["job_id"], x["status"])
                                                                     for x in failed)))
        except (Exception, RunTimeout) as e:
            title, msg = explain(e, "reading the results")
            ss.hw_err = scrub("%s. %s" % (title, msg), token)
        finally:
            token = None                                 # noqa: F841

    st.markdown("**1 · Submit**")
    for k, v in (("hw_channel", "ibm_quantum_platform"), ("hw_instance", ""), ("hw_backend", ""), ("hw_maxtime", 0),
                 ("hw_instance2", "")):
        S.wdefault(k, v)
    c = st.columns([1.2, 2, 1.4, 1.4])                  # settings (not secret) live outside the form and persist
    c[0].selectbox("channel", ["ibm_quantum_platform", "ibm_cloud"], key="hw_channel", persist_state=P)
    c[1].text_input("instance CRN (optional)", key="hw_instance", max_chars=300, persist_state=P)
    c[2].text_input("backend (blank = least busy)", key="hw_backend", max_chars=64, placeholder="e.g. ibm_torino",
                    persist_state=P)
    c[3].number_input("max execution time per job (s, 0 = default)", min_value=0, step=60, key="hw_maxtime",
                      persist_state=P)
    with st.form("hw_submit", clear_on_submit=True, border=True):   # the token only: cleared on submit, not kept
        st.text_input("IBM Quantum API token", type="password", key="hw_token", autocomplete="off",
                      placeholder="paste your API key (it is cleared right after use)")
        st.form_submit_button("Submit to IBM Quantum", type="primary", icon=":material/cloud_upload:",
                              on_click=submit_cb)
    jobs = ss.get("hw_jobs")
    if jobs:
        st.markdown("**Submitted:** %s on %s (physical qubits %s)" % (
            ", ".join("`%s`" % r["job_id"] for r in jobs["records"]), jobs["backend"],
            ", ".join(map(str, jobs["phys"]))))
        sm = jobs["plan"].summary()
        st.caption("%d circuits, %s shots in total. Skipped: %s" % (
            sm["circuits"], f"{sm['shots']:,}", "; ".join("%s (%s)" % kv for kv in jobs["plan"].skipped.items())
            or "none"))
        if jobs.get("note"):
            st.caption(W.md_escape(jobs["note"]))
        plan_view(jobs["plan"], "hw_sub")
        st.markdown("**2 · Collect results**")
        st.text_input("instance CRN for collecting (optional)", key="hw_instance2", max_chars=300, persist_state=P)
        with st.form("hw_fetch", clear_on_submit=True, border=True):
            st.text_input("IBM Quantum API token", type="password", key="hw_token2", autocomplete="off")
            st.form_submit_button("Check status and fetch results", icon=":material/download:", on_click=fetch_cb)
    if ss.get("hw_err"):
        st.error(W.md_escape(ss.hw_err))                 # exception text: shown literally, never as Markdown
    if ss.get("hw_msg"):
        st.success(W.md_escape(ss.hw_msg))
    if ss.get("hw_res") and jobs:
        results_view(ss.hw_res, jobs["plan"], "IBM Quantum %s" % jobs["backend"], "hw_real",
                     state=jobs.get("state"), sig=jobs.get("sig"), extra={"hardware_jobs": pd.DataFrame([dict(job_id=r["job_id"], backend=r["backend"],
                                                                group=" / ".join(r["group"]), shots=r.get("shots"),
                                                                pubs=len(r["pubs"]))
                                                           for r in jobs["records"]])})
