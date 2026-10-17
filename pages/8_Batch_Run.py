"""Batch run: tick several algorithms, mitigation methods, noise models, noise scales and simulators, run them all
with one button, and compare everything in one table and a few charts."""
import math
import time

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Batch run · QEM Explorer", page_icon=":material/playlist_play:", layout="wide")

from qemx import batch as B  # noqa: E402
from qemx import device  # noqa: E402
from qemx import sweep as SW  # noqa: E402
from qemx import templates as T  # noqa: E402
from qemx.bundle import make_zip  # noqa: E402
from qemx.noise import PRESETS as NOISE_PRESETS  # noqa: E402
from qemx.recipes import PRESETS as RECIPE_PRESETS, preset  # noqa: E402
from qemx.viz import batch_charts as BC  # noqa: E402
from ui import common as C  # noqa: E402
from ui import metrics as MX  # noqa: E402
from ui import state as S  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar(run_label=None)
st.title("Batch run")
st.markdown("Tick the algorithms, methods, noise models, noise scales and simulators to compare, then press **Run "
            "all**. Every combination is simulated exactly (bias and variance, no shot noise), one after the other; "
            "a combination that fails is reported and the batch carries on.")
compiled, err = C.compiled_now()
names = B.algorithm_names(ss.get("template_name", "current circuit"))
custom = W.custom_options(ss.custom_recipes)             # {"custom:<Recipe.key()>": Recipe}: identity, not title
custom_label = W.custom_labels(custom)
methods_all = list(RECIPE_PRESETS) + list(custom)
algos_all = [B.CURRENT] + [k for cat in T.CATEGORIES.values() for k in cat]
noises_all = [k for k in NOISE_PRESETS if k != "Ideal (no noise)"]
SCALES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
sims_all = ["numpy", "aer"] + list(device.FAKE_DEVICES)
for k, v in (("w_b_algos", [B.CURRENT, "ghz", "qaoa"]), ("w_b_methods", ["Unmitigated", "ZNE · Richardson",
                                                                         "PEC · inverse", "Readout · M3"]),
             ("w_b_noises", ["Depolarizing", "Hardware-realistic (illustrative)"]), ("w_b_scales", [1.0, 2.0]),
             ("w_b_sims", ["numpy"]), ("w_b_N", 10000), ("w_b_metric", "RMSE"), ("w_b_sweep", "")):
    S.wdefault(k, v)
for k, allowed in (("w_b_algos", algos_all), ("w_b_methods", methods_all), ("w_b_noises", noises_all),
                   ("w_b_scales", SCALES), ("w_b_sims", sims_all)):
    ss[k] = [x for x in ss[k] if x in allowed]

with st.container(border=True):
    st.markdown("**What to run**")
    c = st.columns([1.3, 1.3], gap="medium")
    c[0].multiselect("Algorithms", algos_all, key="w_b_algos", persist_state=P,
                     format_func=lambda k: names.get(k, k),
                     help="The current Playground circuit (with its inputs, observable and compilation), and templates "
                          "at their default parameters with their own observables.")
    c[1].multiselect("Mitigation methods", methods_all, key="w_b_methods", persist_state=P,
                     format_func=lambda k: custom_label.get(k, k),
                     help="Presets, plus any custom contenders built in the Playground.")
    c = st.columns([1.6, 1, 1.2], gap="medium")
    c[0].multiselect("Noise models", noises_all, key="w_b_noises", persist_state=P,
                     help="The sidebar's presets. A fake device brings its own calibrated noise instead.")
    c[1].multiselect("Noise scales λ", SCALES, key="w_b_scales", persist_state=P, format_func=lambda s: "×%g" % s,
                     help="Every rate of the noise model times λ (probabilities capped at 1).")
    c[2].multiselect("Simulators", sims_all, key="w_b_sims", persist_state=P,
                     format_func=lambda k: device.device_label(k) if device.is_device(k) else
                     {"numpy": "exact density matrix (NumPy)", "aer": "Qiskit Aer"}.get(k, k),
                     help="Fake IBM devices compile for the device and use its calibration (one run per algorithm).")

SWEEP_HELP = r"""
| line | sweeps |
|---|---|
| `ramp: 0, 1, 3` | the phase-ramp input $\lvert{\rm ramp}{:}k\rangle=F^\dagger\lvert k\rangle$ on the whole register |
| `input: plus, 0, bits:0101` | register inputs: `0 1 + - +i -i plus minus`, `ramp:k`, `bits:…` |
| `basis: Z, X, ZZXY` | measurement bases: one letter for all qubits, or one per qubit (q0 first) |
| `m: 1, 2, 3` | any parameter of the ticked templates (here the AQFT degree) |
| `p2: 0.005, 0.01` | any noise parameter; replaces that rate in every noise model |
"""
with st.container(border=True):
    st.markdown("**Sweep inputs, bases and parameters** (optional)")
    c = st.columns([1.5, 2], gap="medium")
    c[0].text_area("One swept quantity per line: name: value, value, …", key="w_b_sweep", height=132,
                   max_chars=SW.MAX_TEXT, persist_state=P, placeholder="ramp: 0, 1\nbasis: Z, X\nm: 1, 2, 3",
                   help="Every combination of the lines becomes its own scenario, for every algorithm it applies to: "
                        "template parameters only to templates that have them, noise parameters not to fake devices. "
                        "Inputs are prepared before the circuit (templates that prepare their own input, such as "
                        "the QFT's output s, are swept with their own parameter). Values are read by the app's safe "
                        "expression reader; nothing is executed.")
    with c[1]:
        st.markdown(SWEEP_HELP)
axes, sweep_err = [], None
try:
    axes = SW.parse(ss.w_b_sweep, [a for a in ss.w_b_algos if a != B.CURRENT])
    scen = B.plan(ss.w_b_algos, ss.w_b_noises, ss.w_b_scales, ss.w_b_sims, axes)
except SW.SweepError as e:
    sweep_err, axes = str(e), []
    scen = B.plan(ss.w_b_algos, ss.w_b_noises, ss.w_b_scales, ss.w_b_sims)
if sweep_err:
    st.error("Sweep: %s" % W.md_escape(sweep_err), icon=":material/error:")
elif axes:
    st.caption("Sweeping %s." % "; ".join("**%s** over %s" % (W.md_escape(a.name), W.md_escape(
        ", ".join(SW._fmt(v) for v in a.values))) for a in axes))
n_est = len(scen) * len(ss.w_b_methods)
missing = [nm for nm, v in (("an algorithm", ss.w_b_algos), ("a method", ss.w_b_methods),
                            ("a simulator", ss.w_b_sims)) if not v]
if sweep_err:
    missing.append("a valid sweep")
if not any(device.is_device(s) for s in ss.w_b_sims) and (not ss.w_b_noises or not ss.w_b_scales):
    missing.append("a noise model and a scale")
if B.CURRENT in ss.w_b_algos and compiled is None:
    st.warning("The current Playground circuit does not compile for the sidebar's simulator (%s); scenarios on "
               "other simulators still try it, failures are reported per row." % W.md_escape(err or "unknown error"),
               icon=":material/warning:")
if any(s in ("numpy", "aer") for s in ss.w_b_sims) and any(device.is_device(s) for s in ss.w_b_sims) and \
        (not ss.w_b_noises or not ss.w_b_scales):
    st.caption("NumPy / Aer scenarios need a noise model and a scale; with none chosen only the fake devices run.")


def request():
    ss.batch_run = True


row = st.columns([3, 1.2], vertical_alignment="center")
row[0].markdown("**%s × %s = %s estimates.** %s" % (
    W.plural(len(scen), "scenario"), W.plural(len(ss.w_b_methods), "method"), f"{n_est:,}",
    "Choose %s." % ", ".join(missing) if missing else
    "Each scenario is one exact simulation of every method; large circuits, PEC and learning methods take longest."))
row[1].button("Run all (%d)" % len(scen), type="primary", icon=":material/playlist_play:", key="b_run",
              on_click=request, disabled=bool(missing), width="stretch")
if len(scen) > 60 and not missing:
    st.caption("%d scenarios is a long batch; the sidebar's time limit applies to each scenario, and the page stays "
               "responsive between them." % len(scen))


def batch_sig():
    return S.signature(extra=("batch", tuple(ss.w_b_algos), tuple(ss.w_b_methods), tuple(ss.w_b_noises),
                              tuple(ss.w_b_scales), tuple(ss.w_b_sims), ss.w_b_sweep))


if ss.pop("batch_run", False) and not missing:
    # preset names: each algorithm's symmetry is filled in per scenario; custom contenders run as built
    recipes = [custom[t] if t in custom else t for t in ss.w_b_methods]
    current = None
    try:                                  # each scenario compiles the Playground circuit for its own simulator
        current = dict(spec=S.spec(), inputs=list(ss.inputs), basis=ss.basis, observable=ss.observable,
                       symmetry=ss.symmetry, relabel=ss.relabel, name=ss.get("template_name", "circuit"))
    except Exception:                                         # noqa: BLE001 - reported as failed rows
        current = None
    bar = st.progress(0.0, text="Starting…")
    t0 = time.time()

    def progress(done, total, label):
        el = time.time() - t0
        bar.progress(done / total, text="%d / %d · %s · %.0f s left" % (done, total, label[:90],
                                                                          el / done * (total - done)))
    try:
        rows = B.run(scen, recipes, current=current, opts=S.compile_options(), shots=int(ss.shots),
                     t_shot_us=float(ss.t_shot_us), time_limit=ss.get("time_limit"), progress=progress, names=names,
                     axes=axes)
    except Exception as e:                                    # noqa: BLE001 - B.run already isolates scenarios
        rows = []
        st.error("The batch stopped: %s: %s" % (type(e).__name__, W.md_escape(str(e)[:300])))
    bar.empty()
    if rows:
        ss.batch_result = dict(rows=rows, sig=batch_sig(), seconds=time.time() - t0, shots=int(ss.shots),
                               config=dict(algorithms=[names.get(a, a) for a in ss.w_b_algos],
                                           methods=[custom_label.get(m, m) for m in ss.w_b_methods],
                                           noises=list(ss.w_b_noises),
                                           scales=list(ss.w_b_scales), simulators=list(ss.w_b_sims),
                                           sweep=ss.w_b_sweep.strip(), axes=[a.name for a in axes]))

res = ss.get("batch_result")
if res is None:
    st.info("Press **Run all**.", icon=":material/playlist_play:")
    st.stop()
rows = res["rows"]
if res.get("sig") != batch_sig():
    C.stale_note("Run all")
failed = [r for r in rows if str(r.get("status", "")).startswith("failed")]
m = st.columns(4)
m[0].metric("scenarios", len({B.scenario_key(r) for r in rows}), border=True)
m[1].metric("estimates", len(rows), border=True)
m[2].metric("failed", len(failed), border=True, help="Rows whose status starts with 'failed' (see the table).")
m[3].metric("time", "%.1f s" % res["seconds"], border=True)
if failed:
    with st.expander("%s failed" % W.plural(len(failed), "estimate"), icon=":material/error:", key="exp_b_failed"):
        st.dataframe(pd.DataFrame([{k: r.get(k) for k in ("algorithm", "noise", "λ", "simulator", "method", "status")}
                                   for r in failed]), hide_index=True, width="stretch")

c = st.columns([1.2, 1.2, 3], vertical_alignment="bottom")
metric = c[0].selectbox("color by", ["RMSE", "|bias|", "estimate", "γ"], key="w_b_metric", persist_state=P,
                        help=MX.help_for("RMSE"))
N = int(c[1].number_input("shot budget N (for RMSE)", min_value=1, step=1000, key="w_b_N", persist_state=P,
                          help="RMSE(N) = √(b² + v/N_t + w/s + u) at a total budget of N shots per method (training and calibration included) is recomputed for any N without rerunning."))
figs, tables = {}, {}
st.subheader("Every scenario × every method")
figs["batch_heatmap"] = BC.heatmap(rows, N, metric, dark=C.dark())
W.show_fig(figs["batch_heatmap"], "batch_heatmap")
W.why("batch")
l, r = st.columns([1.7, 1], gap="large")
win = B.winners(rows, N, "RMSE" if metric not in ("|bias|",) else "|bias|")
with l:
    st.markdown("**Best method per scenario**")
    tables["winners"] = pd.DataFrame(win, columns=["scenario", "best method",
                                                   "RMSE" if metric != "|bias|" else "|bias|", "runner-up",
                                                   "gain vs unmitigated"])
    st.dataframe(tables["winners"], hide_index=True, width="stretch", column_config=MX.column_config(
        tables["winners"].columns, {"gain vs unmitigated": st.column_config.NumberColumn(
            format="%.2f×", help="Unmitigated error ÷ best error in this scenario (higher is better)."),
            "RMSE": st.column_config.NumberColumn(format="%.3g", help=MX.help_for("RMSE")),
            "|bias|": st.column_config.NumberColumn(format="%.3g", help=MX.help_for("|bias|"))}))
with r:
    figs["batch_wins"] = BC.wins(rows, N, "RMSE" if metric != "|bias|" else "|bias|", dark=C.dark())
    W.show_fig(figs["batch_wins"], "batch_wins")
scales_run = sorted({r["λ"] for r in rows if r["noise"] != B.DEVICE_NOISE})
if len(scales_run) > 1:
    st.subheader("Error against the noise scale")
    nz_run = [x for x in dict.fromkeys(r["noise"] for r in rows) if x != B.DEVICE_NOISE]
    S.wdefault("w_b_scale_noise", nz_run[0])
    if ss.w_b_scale_noise not in nz_run:
        ss.w_b_scale_noise = nz_run[0]
    nz = st.segmented_control("Noise model", nz_run, key="w_b_scale_noise", persist_state=P) or nz_run[0]
    figs["batch_vs_scale"] = BC.versus_scale(rows, N, nz, metric if metric in ("RMSE", "|bias|") else "RMSE",
                                             dark=C.dark())
    W.show_fig(figs["batch_vs_scale"], "batch_vs_scale")
    st.caption("A method whose curve stays flat as λ grows removes the noise's effect; the unmitigated curve grows "
               "with λ. Crossings show where a cheap method stops being enough.")
swept = [a for a in res["config"].get("axes", []) if len({r.get(a) for r in rows if a in r}) > 1]
if swept:
    st.subheader("Error against the swept values")
    S.wdefault("w_b_sweep_axis", swept[0])
    if ss.w_b_sweep_axis not in swept:
        ss.w_b_sweep_axis = swept[0]
    ax_ = st.segmented_control("Swept quantity", swept, key="w_b_sweep_axis", persist_state=P) or swept[0]
    figs["batch_vs_sweep"], shown, total = BC.versus_sweep(rows, N, ax_, metric if metric in ("RMSE", "|bias|")
                                                           else "RMSE", dark=C.dark())
    W.show_fig(figs["batch_vs_sweep"], "batch_vs_sweep")
    W.why("batch_sweep")
    st.caption("Each panel holds everything else fixed (algorithm, the other swept values, noise, simulator)%s." % (
        "; the first %d of %d panels are shown, the table below has all" % (shown, total) if shown < total else ""))
tables["batch_results"] = pd.DataFrame(rows)
tables["batch_results"]["RMSE at N"] = B.rmse_at(rows, N)
with st.expander("All results (%d rows)" % len(rows), icon=":material/table:", key="exp_b_all"):
    cols = ["algorithm", "sweep", "noise", "λ", "simulator", "method", "family", "estimate", "ideal", "bias", "sd", "RMSE",
            "RMSE at N", "γ", "circuits", "QPU s", "qubits", "2q gates", "status", "seconds"]
    df = tables["batch_results"][[c_ for c_ in cols if c_ in tables["batch_results"].columns]]
    st.dataframe(df, hide_index=True, width="stretch", column_config=MX.column_config(df.columns, {
        "RMSE at N": st.column_config.NumberColumn(format="%.4f", help=MX.help_for("RMSE")),
        "seconds": st.column_config.NumberColumn(format="%.2f", help="Wall-clock time of the scenario.")}))
    st.caption("RMSE uses the shot budget per method set in the sidebar (%s); *RMSE at N* uses the budget N above." %
               f"{res['shots']:,}")
cfg = res["config"]
state = S.to_json()
ff, tt = dict(figs), dict(tables)
note = ("Batch run: algorithms %s; methods %s; noise models %s; scales %s; simulators %s; sweep %s. RMSE at N = %d "
        "shots. %d scenarios, %d estimates, %d failed, %.1f s." % (
            ", ".join(cfg["algorithms"]), ", ".join(cfg["methods"]), ", ".join(cfg["noises"]),
            ", ".join("×%g" % s for s in cfg["scales"]), ", ".join(cfg["simulators"]),
            (cfg.get("sweep") or "none").replace("\n", "; "), N,
            len({B.scenario_key(r) for r in rows}), len(rows), len(failed), res["seconds"]))


_IMG = bool(ss.get("w_zip_images", True))          # captured now: the ZIP is built in a download thread


def zip_bytes():
    return make_zip("Batch run", tables=tt, figures=ff, state=state, notes=note, images=_IMG)


st.divider()
st.download_button("Download all (ZIP): results, winners and figures (PNG/SVG)", zip_bytes,
                   file_name="qem_explorer_batch.zip", mime="application/zip", icon=":material/folder_zip:",
                   key="dl_all_batch")
st.caption("The Playground's *Download all (ZIP)* also includes the latest batch results.")
