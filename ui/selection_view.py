"""Drill-down view of a protocol-selection study (1.6.0; 1.7.0 fresh-design studies listed first) for the QEM Performance page: policy results by budget,
per-task protocol risks and every policy's choice — with the task, its circuit, the protocols' settings, the shot
ledger, the trial distribution and the provenance of the labels — and the study's figures with their captions.
Reads only saved tables and label files under ``reports/<study>/`` (no free-text paths; task ids are validated before
a file name is built from them; nothing is simulated here)."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from qemx.selection.figures import POLICY_STYLE, PROTOCOL_COLORS, _policy_kind
from qemx.viz import theme as TH

KEY = "w_sel_"
_TASK_ID = re.compile(r"^[0-9a-f]{20}$")


def studies(reports: str) -> list:
    """Study directories under ``reports`` (those with ``analysis/results.json``): fresh-design studies (1.7.0) first,
    then historical ones (1.6.0 fixed-shared designs), each group by name."""
    if not os.path.isdir(reports):
        return []
    names = [d for d in sorted(os.listdir(reports))
             if os.path.exists(os.path.join(reports, d, "analysis", "results.json"))]
    return sorted(names, key=lambda d: (design_mode(os.path.join(reports, d)) != "fresh", d))


def design_mode(path: str) -> str:
    """The learning-design plan of a study's labels (``fresh`` from 1.7.0 on; ``fixed-shared`` for 1.6.0)."""
    try:
        with open(os.path.join(path, "analysis", "results.json"), encoding="utf-8") as f:
            return json.load(f).get("design_mode") or "fixed-shared"
    except (OSError, ValueError):
        return "unknown"


def fingerprint(path: str) -> tuple:
    """Cache key of a study directory's tables (1.7.0): the size and modification time of every analysis table and
    run binding, so an updated file under the same name is read again (1.6.0 keyed on one file only)."""
    out = []
    for sub in ("analysis", "dev", "final"):
        d = os.path.join(path, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith((".csv", ".json")):
                st_ = os.stat(os.path.join(d, f))
                out.append((sub, f, st_.st_size, st_.st_mtime_ns))
    return tuple(out)


def _task_files_key(path: str, task_id: str) -> tuple:
    out = []
    for part in ("dev", "final"):
        for ext in (".json", ".npz"):
            p = os.path.join(path, part, "tasks", task_id + ext)
            if os.path.exists(p):
                st_ = os.stat(p)
                out.append((part, ext, st_.st_size, st_.st_mtime_ns))
    return tuple(out)


@st.cache_data(max_entries=4, show_spinner=False)
def _load(path: str, key) -> dict:
    a = os.path.join(path, "analysis")
    pooled = os.path.join(a, "pooled_same-distribution.csv")      # out-of-fold pooled (1.6.0); else per fold
    out = dict(summary=pd.read_csv(pooled if os.path.exists(pooled) else
                                   os.path.join(a, "summary_same-distribution.csv"), encoding="utf-8"),
               labels=pd.read_csv(os.path.join(a, "labels.csv"), encoding="utf-8"))
    for name in ("choices_same-distribution", "task_features"):
        p = os.path.join(a, name + ".csv")
        out[name] = pd.read_csv(p, encoding="utf-8") if os.path.exists(p) else None
    return out


@st.cache_data(max_entries=8, show_spinner=False)
def _record(path: str, task_id: str, key=()):
    """A task's label record and trial estimates (``dev`` or ``final``), or None. The id is validated first."""
    if not _TASK_ID.match(task_id or ""):
        return None
    for part in ("dev", "final"):
        j = os.path.join(path, part, "tasks", task_id + ".json")
        z = os.path.join(path, part, "tasks", task_id + ".npz")
        if os.path.exists(j) and os.path.exists(z):
            rec = json.load(open(j, encoding="utf-8"))
            with np.load(z, allow_pickle=False) as f:
                arr = {k: f[k] for k in ("est", "flags", "design") if k in f.files}
            prov = os.path.join(path, part, "provenance.json")
            pv = json.load(open(prov, encoding="utf-8")) if os.path.exists(prov) else {}
            return dict(part=part, rec=rec, est=arr.get("est"), flags=arr.get("flags"), provenance=pv)
    return None


def render(reports: str) -> None:
    names = studies(reports)
    if not names:
        st.info("No protocol-selection study under `reports/` (run `python -m qemx.selection run CONFIG --out "
                "reports/<name>`).", icon=":material/info:")
        return
    name = st.selectbox("Study", names, key=KEY + "study")
    path = os.path.join(reports, name)
    try:
        d = _load(path, fingerprint(path))
    except Exception as e:                           # noqa: BLE001 - a broken study is reported, not a crash
        st.error("The study tables could not be read: %s" % e, icon=":material/error:")
        return
    if design_mode(path) != "fresh":
        st.warning("Historical study (1.6.0): its CDR and NIL labels are conditional on six training designs shared by "
                   "both label halves, and their standard errors omit the between-design variance (review R1). The "
                   "corrected study draws fresh, independent designs per half; prefer it for any conclusion.",
                   icon=":material/history:")
    st.caption("Simulated observations (emulated fake-device trials, exact density-matrix engine); no hardware data. "
               "Risks are evaluation-half MSEs; choices of measured-risk policies use the independent selection "
               "half. Policies are scored on held-out base circuits (group 5-fold, out-of-fold pooled); intervals "
               "are 95 % bootstrap intervals over base circuits.")
    view = st.segmented_control("View", ["Policies by budget", "Task drill-down", "Figures"],
                                default="Policies by budget", key=KEY + "view")
    if view == "Policies by budget":
        _policies(d["summary"])
    elif view == "Task drill-down":
        _tasks(d, path)
    else:
        _figures(os.path.join(path, "figures"))


def _policies(s: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    sb = s[s.B.astype(str) != "all"].copy()
    sb["B"] = sb.B.astype(float)
    agg = sb.groupby(["policy", "B"])[["mean_log10_risk", "ci_lo", "ci_hi"]].mean().reset_index()
    pols = sorted(agg.policy.unique(), key=lambda p: (_policy_kind(p), p))
    default = [p for p in pols if not p.startswith("MODEL[") or "deployable]" in p][:8]
    pick = st.multiselect("Policies", pols, default=default, key=KEY + "pols")
    fig = go.Figure()
    for p in pick:
        e = agg[agg.policy == p].sort_values("B")
        c, m, _ = POLICY_STYLE.get(_policy_kind(p), ("#444444", "o", "-"))
        fig.add_trace(go.Scatter(x=e.B, y=e.mean_log10_risk, mode="lines+markers", name=p, line=dict(color=c),
                                 error_y=dict(type="data", symmetric=False, array=e.ci_hi - e.mean_log10_risk,
                                              arrayminus=e.mean_log10_risk - e.ci_lo, thickness=1)))
    fig.update_layout(xaxis=dict(type="log", title="total shot budget B"), yaxis_title="mean log10 MSE",
                      height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key=KEY + "fig")
    tab = s[s.B.astype(str) == "all"].groupby("policy")[["mean_log10_risk", "ci_lo", "ci_hi", "diff_vs_sbs",
                                                         "diff_lo", "diff_hi", "gap_closed", "p_regret_gt_2x",
                                                         "coverage"]].mean()
    st.dataframe(tab.sort_values("mean_log10_risk").round(4), width="stretch")


def _tasks(d: dict, path: str = "") -> None:
    import plotly.graph_objects as go
    lab = d["labels"]
    fams = sorted(lab.family.unique())
    fam = st.selectbox("Family", fams, key=KEY + "fam")
    ids = sorted(lab[lab.family == fam].task_id.unique())
    tf = d.get("task_features")
    label = {}
    if tf is not None:
        for _, r in tf[tf.task_id.isin(ids)].iterrows():
            label[r.task_id] = "%s · n=%d · %s · O=%s · in=%s · %s" % (r.task_id[:8], r.n, r.device, r.observable,
                                                                       r.inputs, r.layout)
    tid = st.selectbox("Task", ids, format_func=lambda t: label.get(t, t), key=KEY + "task")
    t = lab[(lab.task_id == tid) & lab.feasible]
    fig = go.Figure()
    for p, e in t.groupby("protocol"):
        e = e.sort_values("B")
        fig.add_trace(go.Scatter(x=e.B, y=e.mse_eval, mode="lines+markers", name=p,
                                 line=dict(color=PROTOCOL_COLORS.get(p, "#444444")),
                                 error_y=dict(type="data", array=2 * e.se_mse_eval, thickness=1)))
    fig.update_layout(xaxis=dict(type="log", title="total shot budget B"),
                      yaxis=dict(type="log", title="MSE (evaluation half, ± 2 SE)"), height=380,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key=KEY + "taskfig")
    ch = d.get("choices_same-distribution")
    if ch is not None:
        c = ch[ch.task_id == tid]
        cols = [x for x in c.columns if x.endswith("|choice")]
        show = c[["B"] + cols].rename(columns={x: x[:-len("|choice")] for x in cols}).set_index("B")
        st.markdown("**Choices on held-out folds** (this task was in the test fold of its group split)")
        st.dataframe(show, width="stretch")
    with st.expander("Labels of this task"):
        keep = ["protocol", "B", "mse_eval", "se_mse_eval", "bias_eval", "var_eval", "mse_sel", "shots_calibration",
                "shots_training", "shots_execution", "t_exec"]
        st.dataframe(t[[k for k in keep if k in t.columns]], width="stretch")
    infeas = lab[(lab.task_id == tid) & ~lab.feasible]
    if len(infeas):
        st.caption("Infeasible: " + "; ".join("%s at B=%d (%s)" % (r.protocol, r.B, r.reason)
                                            for _, r in infeas.head(12).iterrows()))
    try:
        rec = _record(path, tid, _task_files_key(path, tid) if _TASK_ID.match(tid or "") else ()) if path else None
    except Exception as e:                           # noqa: BLE001 - a broken label file is reported, not a crash
        st.error("The label file of this task could not be read: %s" % e, icon=":material/error:")
        return
    if rec is None:
        st.info("The trial files of this task are not in the study folder; the drill-down below needs "
                "`dev/tasks/<task_id>.json` and `.npz`. The source archive ships the per-trial arrays (`.npz`) of "
                "the corrected pilot in a separate trial-data archive (see `reports/selection_v170/README.md`).",
                icon=":material/info:")
        return
    _drill(rec, lab[lab.task_id == tid])


def _drill(rec: dict, lab: pd.DataFrame) -> None:
    """Task and circuit, protocol settings, shot ledger, trial distribution and provenance of one labeled task."""
    import plotly.graph_objects as go
    r = rec["rec"]
    budgets = [int(b) for b in r["budgets"]]
    protos = list(r["protocols"])
    st.markdown("#### Task %s (%s set)" % (r["task_id"], rec["part"]))
    task = r.get("task", {})
    spec = r.get("spec", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Task**")
        st.dataframe(pd.DataFrame(dict(
            field=["family", "width n", "parameters", "inputs", "observable", "device", "layout", "transpiler seed",
                   "base circuit", "ideal ⟨O⟩ (oracle)"],
            value=[spec.get("family"), spec.get("n"), json.dumps(dict(spec.get("params", []))),
                   "%s → %s" % (spec.get("inputs"), ", ".join(task.get("inputs", []))),
                   "%s = %s" % (spec.get("observable"), task.get("observable", {}).get("label", "")),
                   spec.get("device"), "%s → physical %s" % (spec.get("layout"), task.get("phys", "")),
                   spec.get("seed"), r.get("base_id"), "%.6g" % float(r.get("ideal", np.nan))])).astype(str),
            hide_index=True, width="stretch")
    with c2:
        st.markdown("**Compiled circuit**")
        f = r.get("features", {})
        st.dataframe(pd.DataFrame(dict(
            field=["two-qubit gates", "depth", "logical / implemented operations", "duration (µs)",
                   "light-cone qubits / gates", "light-cone fault rate λ", "readout fault rate", "circuit digest"],
            value=[task.get("two_qubit"), task.get("depth"),
                   "%s / %s" % (task.get("logical_ops"), task.get("implemented_ops")),
                   "%.3g" % float(f.get("duration_us", np.nan)),
                   "%s / %s" % (f.get("lc_qubits"), f.get("lc_gates")), "%.4f" % float(f.get("lam_lc", np.nan)),
                   "%.4f" % float(f.get("lam_ro", np.nan)), task.get("circuit_digest")])).astype(str),
            hide_index=True, width="stretch")
    if st.toggle("Show the logical circuit", key=KEY + "circ"):
        try:
            from qemx.circuit import to_qiskit
            from qemx.selection import task as TK
            body = TK.family_circuit(TK.TaskSpec.from_json(spec))[0]
            st.caption("Logical circuit of the family (without input preparation and measurement basis); the "
                       "executed circuit is this one transpiled to the device (figures in the table above).")
            st.code(str(to_qiskit(body).draw(output="text", fold=110)), language=None)
        except Exception as e:                       # noqa: BLE001 - drawing is optional
            st.error("The circuit could not be rebuilt: %s" % e, icon=":material/error:")
    B = st.select_slider("Budget for the ledger and the trials", budgets,
                         value=min(budgets, key=lambda b: abs(np.log10(b) - 5)),
                         format_func=lambda b: "%g" % b, key=KEY + "B")
    j_b = budgets.index(int(B))
    # protocol settings and allocation at this budget
    from qemx.selection import protocols as PR
    from qemx.selection.plots import PROTOCOL_INFO
    alloc = {a["protocol"]: a for a in r.get("allocations", []) if int(a["B"]) == int(B)}
    rows = []
    for p in protos:
        P = PR.PORTFOLIO.get(p)
        a = alloc.get(p, {})
        info = PROTOCOL_INFO.get(p, ("",) * 5)
        rows.append(dict(protocol=p, settings=", ".join("%s=%s" % kv for kv in P.params) if P else "",
                         feasible=a.get("feasible"), reason=a.get("reason", ""), execution_shots=a.get("N"),
                         calibration_shots_per_circuit=a.get("s"), calibration_circuits=a.get("c"),
                         training_circuits=a.get("train_circuits"), shot_split=str(a.get("split", [])),
                         noise_information=info[0], approximation=info[4],
                         status=r.get("method_status", {}).get(p, "")))
    st.markdown("**Protocol settings and allocation at B = %g**" % B)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    # cost breakdown (ledger)
    parts = ["calibration", "training", "diagnostics", "selection", "execution"]
    colors = {"calibration": "#E69F00", "training": "#CC79A7", "diagnostics": "#56B4E9", "selection": "#999999",
              "execution": "#0072B2"}
    fig = go.Figure()
    for k in parts:
        fig.add_trace(go.Bar(x=protos, y=[alloc.get(p, {}).get("ledger", {}).get(k, 0) for p in protos], name=k,
                             marker_color=colors[k]))
    fig.update_layout(barmode="stack", yaxis_title="shots (sum = B)", height=300,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.markdown("**Shot ledger at B = %g** (every shot charged to the same budget)" % B)
    st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key=KEY + "ledger")
    # trial distribution
    est = rec.get("est")
    if est is not None:
        half = st.segmented_control("Label half", ["evaluation", "selection"], default="evaluation",
                                    key=KEY + "half") or "evaluation"
        h = 1 if half == "evaluation" else 0
        mu = float(r.get("ideal", np.nan))
        fig = go.Figure()
        summ = []
        for i, p in enumerate(protos):
            x = est[h, i, j_b]
            if not np.isfinite(x).any() and not alloc.get(p, {}).get("feasible", True):
                continue
            ok = np.isfinite(x)
            fig.add_trace(go.Box(y=x[ok], name=p, boxpoints="all", jitter=0.4, pointpos=0, marker=dict(size=3),
                                 line=dict(color=PROTOCOL_COLORS.get(p, "#444444")), showlegend=False))
            e = np.where(ok, x, 0.0) - mu
            summ.append(dict(protocol=p, trials=int(x.size), nonresponse=int((~ok).sum()),
                             mean=float(np.nanmean(x)) if ok.any() else np.nan, bias=float(np.mean(e)),
                             std=float(np.std(e, ddof=1)) if x.size > 1 else np.nan, mse=float(np.mean(e ** 2))))
        fig.add_hline(y=mu, line=dict(color="#333333", dash="dot", width=1), annotation_text="ideal ⟨O⟩")
        fig.update_layout(yaxis_title="estimate of ⟨O⟩ per trial", height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.markdown("**Trial distribution at B = %g** (%s half; one point per independent repetition of the whole "
                    "protocol)" % (B, half))
        st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key=KEY + "trials")
        st.dataframe(pd.DataFrame(summ).round(6), hide_index=True, width="stretch")
    with st.expander("Provenance of these labels"):
        pv = rec.get("provenance", {})
        sess = pv.get("sessions", [])
        prov = sess[-1].get("provenance", {}) if sess else {}
        st.dataframe(pd.DataFrame(dict(
            field=["configuration digest", "master seed", "trials per half (R)", "training designs",
                   "labeling seconds", "label sessions", "app version", "git commit", "uncommitted changes",
                   "lock sha256", "Python", "packages"],
            value=[r.get("config_digest"), r.get("master_seed"), r.get("R"), r.get("learning_designs"),
                   "%.1f" % float(r.get("seconds", np.nan)), len(sess), prov.get("app_version"),
                   prov.get("git_commit"), prov.get("git_dirty"), prov.get("lock_sha256"), prov.get("python"),
                   ", ".join("%s %s" % kv for kv in prov.get("packages", {}).items())])).astype(str),
            hide_index=True, width="stretch")
        ds = r.get("design_status", {})
        if ds:
            st.caption("Training-design status: " + "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in ds.items()))


def _figures(fdir: str) -> None:
    if not os.path.isdir(fdir):
        st.info("No figures in this study (python -m qemx.selection figures …).")
        return
    for f in sorted(x for x in os.listdir(fdir) if x.endswith(".png")):
        name = f[:-4]
        st.image(os.path.join(fdir, f), caption=name, width="stretch")
        md = os.path.join(fdir, name + ".md")
        if os.path.exists(md):
            with st.expander("Caption and data of %s" % name):
                st.markdown(open(md, encoding="utf-8").read())
