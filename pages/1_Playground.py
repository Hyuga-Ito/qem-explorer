"""Playground: build a circuit (drag and drop, table or text), inject noise or pick a fake IBM device, look inside
every method's circuits, scrub through the state gate by gate, compare mitigation methods, export everything."""
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Playground · QEM Explorer", page_icon=":material/science:", layout="wide")

from qemx import device  # noqa: E402
from qemx import hamiltonian as HM  # noqa: E402
from qemx import export as EX  # noqa: E402
from qemx import insight as IN  # noqa: E402
from qemx import layout_search as LS  # noqa: E402
from qemx import resources as RS  # noqa: E402
from qemx import stages as STG  # noqa: E402
from qemx import variational as VAR  # noqa: E402
from qemx import compile as CP  # noqa: E402
from qemx.compile import coupling_map  # noqa: E402
from qemx.explain import WHY  # noqa: E402
from qemx import templates as T  # noqa: E402
from qemx.backends.base import BACKENDS  # noqa: E402
from qemx.circuit import (GATES, NOISE_OPS, CircuitSpec, Op, ParseError, fmt_angle, fmt_param, parse,  # noqa: E402
                          parse_line)
from qemx.compile import INPUT_STATES, input_label  # noqa: E402
from qemx.methods import Recipe  # noqa: E402
from qemx.recipes import PRESETS as RECIPE_PRESETS, table_rows  # noqa: E402
from qemx.timeline import INSPECT, PREVIEWS, preview  # noqa: E402
from qemx.viz import composer as CV  # noqa: E402
from qemx.viz import qiskit_draw as QD  # noqa: E402
from qemx.viz import states as SV  # noqa: E402
from qemx.viz import theme as TH  # noqa: E402
from qemx.viz import topology as TOPO  # noqa: E402
from ui import cache as K  # noqa: E402
from ui import composer_dnd as DND  # noqa: E402
from ui import state as S  # noqa: E402
from ui import metrics as MX  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.common import dark  # noqa: E402
from ui import results as RES  # noqa: E402
from ui.sidebar import sidebar, sim_label  # noqa: E402
from ui.widgets import P  # noqa: E402


S.init()
ss = st.session_state
sidebar("Run simulation", "run_requested")

# ================================================================== circuit + compilation (cheap, every rerun)
spec_error = compile_error = None
compiled = None
try:
    spec = S.spec()
except (ValueError, ParseError) as e:
    spec, spec_error = None, str(e)
if spec is not None:
    try:
        compiled = K.get_compiled(spec, ss.inputs, ss.basis, S.compile_options(), ss.relabel)
        ss.device_info = compiled.info if device.is_device(ss.backend) else None
    except Exception as e:                                   # compile errors are shown, never crash the page
        compile_error = "%s: %s" % (type(e).__name__, e)


def run_all():
    sig = S.signature()
    recipes = tuple(S.recipes())
    if not recipes:
        st.warning("Choose at least one contender in **Mitigate & compare** before running.",
                   icon=":material/playlist_add:")
        return
    n = compiled.implemented.n
    nj = K.noise_json(ss.noise)
    with st.status("Simulating: time travel, then %d method%s…" % (len(recipes), "" if len(recipes) == 1 else "s"),
                   expanded=True) as status:
        ok1, tl = W.guarded("Time travel", K.timeline, sig, compiled, nj, "implemented", 3.0, "global", "X2",
                            ss.backend, n=n)
        ok2, cmp_ = W.guarded("The comparison", K.comparison, sig, compiled, nj, ss.observable, recipes, ss.backend,
                              int(ss.shots), ss.symmetry, n=n)
        if not ok2:
            status.update(label="Stopped: see the message inside.", state="error", expanded=True)
            return
        ideal, outs, runner = cmp_
        ok3, tr = W.guarded("The cost table", table_rows, outs, ideal, int(ss.shots), ss.t_shot_us, n=n)
        if not ok3:
            status.update(label="Stopped: see the message inside.", state="error", expanded=True)
            return
        samples, rows = tr
        # the state this result belongs to travels with it (ZIP, script), whatever is edited afterwards
        ss.results = dict(sig=sig, ideal=ideal, outs=outs, samples=samples, rows=rows, calls=runner.calls,
                          shots=int(ss.shots), backend=ss.backend, dist=analysis_data(outs, runner),
                          observable=ss.observable, state=S.to_json())
        if ok1:
            ss.timeline = dict(sig=sig, key=("implemented", 3.0, "global", "X2"), tl=tl)
        failed = [o.title for o in outs if o.result is None]
        if not ok1 or failed:
            status.update(label="Done with problems: %s. Details inside." % "; ".join(
                (["time travel failed"] if not ok1 else []) + (["%d method(s) failed" % len(failed)] if failed else [])),
                state="error", expanded=True)
        else:
            status.update(label="Done: %d simulations." % runner.calls, state="complete", expanded=False)


def analysis_data(outs, runner) -> dict | None:
    """Exact output distributions for the analysis views (reuses the comparison's memoized simulations)."""
    n = compiled.logical.n
    noise = runner.device_model.equivalent_noise() if runner.device_model is not None else ss.noise
    obs = ss.observable
    try:
        d = IN.distributions(compiled, noise, outs, runner, ss.backend)
    except Exception as e:                               # noqa: BLE001 - the views say what is missing
        return dict(series={}, notes=["Distributions unavailable: %s" % W.plain(str(e), 160)], readout=[], n=n)
    custom = S.is_custom()
    logical = (not custom and getattr(obs, "key", None) == ss.get("template_obs_key")
               and bool(ss.relabel) == bool(ss.get("template_relabel")))
    tpl, prm = ss.get("template"), dict(ss.get("tpl_params") or {})
    d["series"]["ideal"] = np.clip(d["series"]["ideal"], 0.0, None) if "ideal" in d["series"] else None
    d.update(n=n, values=None if hasattr(obs, "terms") else [float(v) for v in obs.values],
             hamiltonian=hasattr(obs, "terms"),
             checks=IN.qec_checks(tpl, prm, n) if logical else [],
             intended=IN.intended_logical(tpl, prm) if logical else None,
             snapshot=dict(template=tpl, params=prm, custom=custom, logical=logical))
    if d["series"].get("ideal") is None:
        d["series"].pop("ideal", None)
    return d


# ================================================================== header
st.title("Playground")
header = st.container()          # filled at the end of the script, after every callback has updated the state
if ss.pop("run_requested", False):
    if compiled is None:
        st.error("Fix the circuit first: %s" % (spec_error or compile_error or "it did not compile"))
    else:
        run_all()
stale = ss.results is None or ss.results["sig"] != S.signature()
if spec_error:
    st.error("The circuit has an error: %s" % spec_error)
if compile_error:
    st.error("Compilation failed: %s" % compile_error)

HALF_PX, FULL_PX = 560, 1150              # expected widths of a half-page and a full-page figure (text/marker scale)
TABS = ["Build", "Circuits", "Topology & layout", "Time travel", "Mitigate & compare", "Export"]
S.wdefault("w_pg_tabs", ss.page_tab.get("playground", "Build"))
tabs = st.tabs(TABS, key="w_pg_tabs", on_change=S.cb_tab, args=("playground", "w_pg_tabs"))


# ================================================================== Build
def template_card():
    with st.container(border=True):
        st.markdown("**Start from a template**")
        try:
            _template_card_body()
        except Exception as e:                       # noqa: BLE001 - e.g. parameters out of a widget's range
            st.error("The template controls could not be shown (%s: %s)." % (type(e).__name__,
                                                                          W.md_escape(str(e)[:200])),
                     icon=":material/error:")


def _template_card_body():
    S.wdefault("w_tpl", ss.template if ss.template in T.REGISTRY else "bell")
    S.wdefault("w_tpl_cat", T.category_of(ss.w_tpl))

    def cat_changed():
        cat_ = ss.get("w_tpl_cat")
        if cat_ in T.CATEGORIES and ss.get("w_tpl") not in T.CATEGORIES[cat_]:
            ss.w_tpl = T.CATEGORIES[cat_][0]
    if ss.w_tpl_cat not in T.CATEGORIES or ss.w_tpl not in T.CATEGORIES[ss.w_tpl_cat]:
        ss.w_tpl_cat = T.category_of(ss.w_tpl)          # a template set elsewhere (state file, test) wins
    c = st.columns([1.2, 1.6], vertical_alignment="bottom")
    cat = c[0].selectbox("Family", list(T.CATEGORIES), key="w_tpl_cat", on_change=cat_changed, persist_state=P,
                         help="Basics and algorithms; NISQ variational circuits (VQE, QAOA, QML); fault-tolerance "
                              "primitives (error-correcting codes) and deep algorithms (Shor, HHL).")
    keys = list(T.CATEGORIES[cat])
    if ss.w_tpl not in keys:
        ss.w_tpl = keys[0]
    tkey = c[1].selectbox("Template", keys, format_func=lambda k: T.REGISTRY[k][0], key="w_tpl", persist_state=P)
    schema = T.REGISTRY[tkey][2]
    from qemx import presets_cited as PC
    S.wdefault("w_cited_preset", "—")
    if ss.w_cited_preset != "—" and ss.w_cited_preset not in PC.BY_KEY:
        ss.w_cited_preset = "—"
    st.selectbox("Cited preset (type to search)", ["—"] + [p.key for p in PC.PRESETS], key="w_cited_preset",
                 on_change=S.cb_cited_preset, persist_state=P,
                 format_func=lambda k: "—" if k == "—" else "%s · %s" % (PC.BY_KEY[k].title, PC.BY_KEY[k].kind),
                 help="Fills the template's controls with a documented setting: source and locator, convention, "
                      "units, sizes, purpose and limits. Exact reproductions, adapted settings and teaching examples "
                      "are labeled as such. Every value stays editable; press Load template to build it.")
    cp_ = PC.BY_KEY.get(ss.w_cited_preset)
    if cp_ is not None and cp_.template == tkey:
        with st.expander("Provenance: %s" % cp_.title, icon=":material/menu_book:", key="exp_cited"):
            st.markdown(PC.describe(cp_.key))
    if tkey in getattr(T, "QEC_ALGORITHM_KEYS", ()):
        names = [nm for nm, k_, _ in T.QEC_PRESETS if k_ == tkey]
        st.selectbox("Preset (code × benchmark)", ["—"] + names, key="w_tpl_preset_%s" % tkey,
                     on_change=S.cb_template_preset, args=(tkey,), persist_state=P,
                     help="Fills the parameters below with a ready-made pairing of an error-correcting code and a "
                          "benchmark algorithm (what is fault-tolerant in it is stated in the notes after loading). "
                          "Press Load template to build it.")
    cur = dict(ss.tpl_params) if tkey == ss.template else {}
    vals = {}
    ncol = 4
    grid = None
    for i, p in enumerate(schema):
        if i % ncol == 0:
            grid = st.columns(ncol, vertical_alignment="bottom")
        box = grid[i % ncol]
        k = "w_tp_%s_%s" % (tkey, p.name)
        S.wdefault(k, cur.get(p.name, p.default))
        if p.kind == "choice":
            if ss[k] not in p.options:
                ss[k] = p.default
            fmt = str
            if tkey in ("qft", "aqft") and p.name == "mode":
                fmt = lambda v: T._BITS.get(v, str(v))        # noqa: E731
            elif tkey in ("qft", "aqft") and p.name == "prep":
                fmt = lambda v: T.QFT_PREP_LABELS.get(v, str(v))   # noqa: E731
            vals[p.name] = box.selectbox(p.label, p.options, key=k, persist_state=P, help=p.help or None,
                                         format_func=fmt)
        else:
            lo, hi = p.bound(p.lo, {**T.defaults(tkey), **vals}), p.bound(p.hi, {**T.defaults(tkey), **vals})
            if p.kind == "int":                 # bounds such as s < 2^n exceed what a number widget can show
                lo = None if lo is None else max(int(lo), -S.JS_SAFE_INT)
                hi = None if hi is None else min(int(hi), S.JS_SAFE_INT)
            v = ss[k]
            v = int(v) if p.kind == "int" else float(v)
            if lo is not None and v < lo:
                v = lo
            if hi is not None and v > hi:
                v = hi
            if p.kind == "int":
                v = min(max(int(v), -S.JS_SAFE_INT), S.JS_SAFE_INT)
            ss[k] = int(v) if p.kind == "int" else float(v)
            if p.kind == "int":
                vals[p.name] = int(box.number_input(p.label, min_value=int(lo) if lo is not None else None,
                                                    max_value=int(hi) if hi is not None else None,
                                                    step=int(p.step or 1), key=k, persist_state=P,
                                                    help=p.help or None))
            else:
                vals[p.name] = float(box.number_input(p.label, min_value=float(lo) if lo is not None else None,
                                                      max_value=float(hi) if hi is not None else None,
                                                      step=float(p.step or 0.01), format="%.4f", key=k,
                                                      persist_state=P, help=p.help or None))
    st.button("Load template", icon=":material/download:", key="tpl_load", type="primary",
              on_click=S.load_template, args=(tkey,))
    if ss.get("tpl_error"):
        st.error(W.md_escape(ss.tpl_error), icon=":material/error:")
    if ss.get("tpl_tex"):
        st.latex(ss.tpl_tex)
    if ss.get("tpl_notes"):
        st.caption(W.md_escape(ss.tpl_notes))
    n_new = T.clamp_params(tkey, vals)
    if tkey in ("shor", "hea_vqe", "qaoa") or n_new.get("n", 0) >= 10 or n_new.get("t", 0) >= 8:
        st.caption("Loading replaces the circuit (Undo in the composer brings it back)."
                   + (" Variational angles are optimized when you press Load; this takes longer for many qubits."
                      if tkey in ("hea_vqe", "qaoa") else ""))


def editors():
    with st.expander("Edit as a table", icon=":material/table_rows:", key="exp_table"):
        def table():
            d = pd.DataFrame([dict(operation=o.name, qubits=" ".join(map(str, o.qubits)),
                                   parameter=", ".join(fmt_angle(p) if o.kind == "gate" else fmt_param(p)
                                                       for p in o.params)) for o in S.spec().ops])
            return d if not d.empty else pd.DataFrame(columns=["operation", "qubits", "parameter"])
        df = S.memo(S.spec(), "table", table)              # rebuilt only when the circuit changes
        edited = st.data_editor(
            df, num_rows="dynamic", hide_index=False, key="w_ed_%d" % ss.ops_version, width="stretch",
            column_config={
                "operation": st.column_config.SelectboxColumn("operation", options=list(GATES) + list(NOISE_OPS)
                                                              + ["barrier"], required=True),
                "qubits": st.column_config.TextColumn("qubits", help="space-separated, e.g. 0 1", max_chars=200),
                "parameter": st.column_config.TextColumn("parameter", help="angle or probability, e.g. pi/4",
                                                         max_chars=200)})
        if st.button("Apply table", key="ed_apply", icon=":material/check:"):
            new, errs = [], []
            for i, r in edited.iterrows():
                if not isinstance(r["operation"], str) or not r["operation"]:
                    continue
                par = str(r["parameter"]).strip() if isinstance(r["parameter"], str) and r["parameter"].strip() else ""
                line = "%s%s %s" % (r["operation"], "(%s)" % par if par else "", r["qubits"] or "")
                try:
                    o = parse_line(line, int(ss.n))
                    if o is not None:
                        old = ss.ops[i] if isinstance(i, int) and 0 <= i < len(ss.ops) else None
                        same = old is not None and old["name"] == o.name and list(old["qubits"]) == list(o.qubits) \
                            and np.allclose(old["params"], o.params, rtol=1e-10, atol=1e-12)
                        new.append(dict(old) if same else dict(name=o.name, qubits=list(o.qubits),
                                                               params=list(o.params), kind=o.kind, tag=""))
                except (ParseError, ValueError) as e:
                    errs.append("row %s: %s" % (i, e))
            if errs:
                st.error("  \n".join(W.md_escape(x[:300]) for x in errs[:20]))
            else:
                S.set_circuit(CircuitSpec(int(ss.n), tuple(Op(d["name"], tuple(d["qubits"]), tuple(d["params"]),
                                                               d["kind"], d.get("tag", "")) for d in new)))
                st.rerun()
    with st.expander("Type the circuit as text", icon=":material/code:", key="exp_text"):
        st.caption("One operation per line or separated by ';'. Gates: " + ", ".join(GATES) +
                   ". Noise: " + ", ".join("%s(p)" % k for k in NOISE_OPS) + ". Example: `h 0; cx 0 1; depol(0.05) 0 1`. "
                   "Parameters accept numbers and expressions with pi, e, tau, sqrt, sin, cos — nothing is executed.")
        # one box per circuit version; "page" scope keeps typed-but-unapplied text across tab switches
        txt = st.text_area("circuit", S.memo(S.spec(), "text", lambda: S.spec().text()), height=180,
                           key="w_txt_%d" % ss.ops_version, label_visibility="collapsed", persist_state="page") or ""
        if st.button("Apply text", key="txt_apply", icon=":material/check:"):
            if not txt.strip():
                st.error("The text is empty. To remove every operation use **Clear** in the composer.")
            else:
                try:
                    new_spec = parse(txt, int(ss.n))
                    if not new_spec.ops:
                        raise ParseError("no operations found (only blank or comment lines). To remove every "
                                         "operation use Clear in the composer.")
                    S.set_circuit(new_spec)
                    st.rerun()
                except (ParseError, ValueError) as e:
                    st.error(W.md_escape(str(e)[:1000]))


def register_io():
    """Batch-configure every qubit at once: a register input (plus, ramp:k, bits:…) and/or a basis (X, ZZXY)."""
    S.wdefault("w_io_all_in", "")
    S.wdefault("w_io_all_bs", "")
    c = st.columns(2, vertical_alignment="bottom")
    c[0].text_input("All qubits: input", key="w_io_all_in", max_chars=60, persist_state=P,
                    placeholder="plus · ramp:1 · bits:0101",
                    help=r"`0 1 + - +i -i` (also `plus`, `minus`) on every qubit; `ramp:k` for the phase ramp "
                         r"$\lvert{\rm ramp}{:}k\rangle=F^\dagger\lvert k\rangle=\tfrac1{\sqrt N}\sum_j "
                         r"e^{-2\pi ijk/N}\lvert j\rangle$ (a product state: qubit q gets the phase "
                         r"$-2\pi k2^q/N$); `bits:0101` for a basis state written $q_{n-1}\cdots q_0$. "
                         r"Leave empty to keep the inputs.")
    c[1].text_input("All qubits: basis", key="w_io_all_bs", max_chars=64, persist_state=P, placeholder="X · ZZXY",
                    help="One letter for every qubit, or one letter per qubit (q0 first). Leave empty to keep the "
                         "bases.")
    st.button("Apply to all qubits", key="io_all_apply", on_click=S.cb_register_io, icon=":material/done_all:",
              width="stretch")
    if ss.get("io_error"):
        st.error(W.md_escape(ss.io_error), icon=":material/error:")
    if any(x not in INPUT_STATES for x in ss.inputs):
        st.caption(r"$\lvert\varphi t\rangle=(\lvert0\rangle+e^{2\pi it}\lvert1\rangle)/\sqrt2$, the phase $t$ in "
                   r"turns (φ.875: $t=0.875$).")


def io_card():
    with st.container(border=True):
        st.markdown("**Inputs and measurement**")
        n = int(ss.n)
        per = min(n, 4)                      # four per row keeps "|+i⟩" readable next to the arrow
        nonstd = sum(1 for q in range(n) if ss.inputs[q] != "0" or ss.basis[q] != "Z")
        box = st.expander("Input state and measurement basis of each qubit (%d not |0⟩ / Z)" % nonstd,
                          icon=":material/tune:", expanded=nonstd > 0, key="exp_io") if n > 8 else st.container()
        with box:
            register_io()
            for start in range(0, n, per):
                cols = st.columns(per)
                for q in range(start, min(n, start + per)):
                    S.wdefault("w_in_%d" % q, ss.inputs[q])
                    if ss["w_in_%d" % q] != ss.inputs[q]:
                        ss["w_in_%d" % q] = ss.inputs[q]
                    opts = list(INPUT_STATES) + ([ss.inputs[q]] if ss.inputs[q] not in INPUT_STATES else [])
                    cols[q - start].selectbox("q%d in" % q, opts, key="w_in_%d" % q,
                                              format_func=input_label, on_change=S.cb_input, args=(q,),
                                              persist_state=P)
            for start in range(0, n, per):
                cols = st.columns(per)
                for q in range(start, min(n, start + per)):
                    S.wdefault("w_bs_%d" % q, ss.basis[q])
                    if ss["w_bs_%d" % q] != ss.basis[q]:
                        ss["w_bs_%d" % q] = ss.basis[q]
                    cols[q - start].selectbox("q%d basis" % q, ["Z", "X", "Y"], key="w_bs_%d" % q,
                                              on_change=S.cb_basis, args=(q,), persist_state=P)
        try:
            obs = S.observable_choices()                  # names only: nothing of size 2^n is built here
        except Exception as e:                            # noqa: BLE001 - never crash the page on a huge register
            st.error("The observable list could not be built for %d qubits (%s)." % (n, W.md_escape(str(e)[:200])),
                     icon=":material/error:")
            obs = {ss.observable.key: ss.observable.label}
        keys = list(obs) + ["custom", "hamiltonian"]
        if ss.get("w_obs") not in keys:
            ss.w_obs = ss.observable.key if ss.observable.key in obs else keys[0]
        st.selectbox("Observable ⟨O⟩", keys, key="w_obs", on_change=S.cb_observable, persist_state=P,
                     format_func=lambda k: obs[k] if k in obs else
                     ("P(bitstring)…" if k == "custom" else "Hamiltonian: a sum of Pauli strings…"),
                     help="Diagonal observables are measured in one setting; a Hamiltonian with X or Y terms is "
                          "measured in several (one per group of qubit-wise commuting terms).")
        if ss.get("obs_error"):
            st.error(W.md_escape(ss.obs_error), icon=":material/error:")
        if ss.w_obs == "hamiltonian":
            hamiltonian_editor(n)
        if ss.w_obs == "custom":
            st.text_input("bitstring q%d…q0" % (n - 1), key="w_obs_bits", on_change=S.cb_obs_bits, max_chars=64,
                          placeholder="e.g. " + "10" * (n // 2) + "1" * (n % 2), persist_state=P)
            if ss.get("obs_bits_error"):
                st.error(ss.obs_bits_error, icon=":material/error:")
            elif not ss.get("w_obs_bits"):
                st.caption("Type a bitstring; the observable stays %s until then." % W.md_escape(ss.observable.label))
        st.latex(ss.observable.tex)
        S.wdefault("w_relabel", bool(ss.relabel))
        st.toggle("Read the output bits in reverse order", key="w_relabel", on_change=S.cb_relabel, persist_state=P,
                  help="Classical relabeling instead of a SWAP network (QFT without SWAPs).")
        if RS.dm_bytes(n) > 64 * 2 ** 20:
            est = RS.Estimate(n, 4 * RS.dm_bytes(n), 0, RS.available_bytes())
            W.memory_notice(est, "A density matrix of %d qubits" % n)


def output_card():
    """The experiment's output: read-only, derived from the input, the circuit and the measurement (never edited)."""
    from qemx import io_semantics as IO
    with st.container(border=True):
        st.markdown("**Output** · derived, read-only")
        if compiled is None:
            st.caption("No output: the circuit does not compile yet.")
            return
        try:
            obs = ss.observable
            okey = getattr(obs, "key", "?")
            d = S.memo(compiled, "ideal_output_%s" % okey,
                       lambda: IO.derived_output(compiled.logical, bool(ss.relabel), obs))
            inp = S.memo(compiled, "input_description", lambda: IO.describe_input(S.spec(), ss.inputs))
        except Exception as e:                                # noqa: BLE001 - a display never breaks the page
            st.caption("The ideal output could not be derived (%s)." % W.md_escape(str(e)[:200]))
            return
        parts = []
        if inp["nonzero"]:
            parts.append("input states %s" % ", ".join("q%d %s" % (q, inp["per_qubit"][q]) for q in inp["nonzero"][:6])
                         + (" …" if len(inp["nonzero"]) > 6 else ""))
        if inp["prep_gates"]:
            parts.append("template preparation %s" % inp["prep_summary"])
        st.caption("Input: %s. Change it with the template's *input* parameters or the per-qubit inputs above; this "
                   "panel only reads it." % ("; ".join(parts) if parts else "|0…0⟩"))
        st.markdown("**Ideal prediction** (noiseless): " + W.md_escape(IO.output_summary(d)))
        if d.get("available") and not d.get("deterministic") and d["support"] <= 64 and len(d["top"]) > 1:
            st.dataframe(pd.DataFrame(d["top"], columns=["outcome c(n-1)…c0", "ideal probability"]), hide_index=True,
                         width="stretch", height=min(35 * (len(d["top"]) + 1) + 3, 320))
        st.caption("The *noisy simulation prediction* and *sampled outcomes* of each method are in **Mitigate & "
                   "compare** (labeled there). Bitstrings are written c(n−1)…c0, little-endian integers.")


def hamiltonian_editor(n: int):
    """Text input for H = Σ c P (safe parser), examples, and the measurement settings it needs."""
    cur = ss.observable if isinstance(ss.observable, HM.PauliSum) else None
    ex = HM.examples(n)
    S.wdefault("w_obs_ham", cur.text if cur is not None else next(iter(ex.values())))

    def use_example(txt):
        ss.w_obs_ham = txt
        S.cb_hamiltonian()
    st.text_area("H = Σ c·P", key="w_obs_ham", on_change=S.cb_hamiltonian, height=76, max_chars=4000,
                 persist_state=P, placeholder="0.5*Z0*Z1 + 0.2*X0 - 0.1*Y1",
                 help="Real coefficients (numbers or pi/4, sqrt(2)/2 …) times Pauli factors X0, Y1, Z2 joined by "
                      "`*`, a space or nothing; `+`/`−` between terms; a bare number is a constant. Press "
                      "Ctrl+Enter (or leave the box) to apply. Nothing you type is executed.")
    row = st.container(horizontal=True, gap="small")
    for i, (name, txt) in enumerate(ex.items()):
        row.button(name, key="ham_ex_%d" % i, on_click=use_example, args=(txt,), type="tertiary",
                   icon=":material/function:")
    if ss.get("ham_error"):
        st.error(W.md_escape(ss.ham_error), icon=":material/error:")
    elif cur is None:
        st.caption("Apply a Hamiltonian to use it; the observable stays %s until then." % W.md_escape(
            ss.observable.label))
    else:
        g = cur.groups()
        st.caption("%s · measured in %s: %s%s" % (
            W.plural(len(cur.terms), "term"), W.plural(len(g), "setting"),
            ", ".join("`%s`" % HM.basis_string(b, n) for b, _ in g) or "none (a constant)",
            " · constant %+g" % cur.constant if cur.constant else ""))


def compile_card():
    with st.container(border=True):
        st.markdown("**Compilation**")
        cm = ss.compile
        if device.is_device(ss.backend):
            st.caption("Compiled for **%s**: its coupling map and native gates; broken couplers in the calibration "
                       "are avoided." % device.device_label(ss.backend))
            c1, c2 = st.columns(2)
            c1.select_slider("optimization level", [0, 1, 2, 3], key="w_cm_opt", on_change=S.cb_compile,
                             args=("opt_level", "w_cm_opt"), persist_state=P)
            if ss.get("w_cm_lay") not in ("sabre", "dense", "trivial"):
                ss.w_cm_lay = "sabre"
            c2.selectbox("qubit selection", ["sabre", "dense", "trivial"], key="w_cm_lay", on_change=S.cb_compile,
                         args=("layout", "w_cm_lay"), persist_state=P,
                         format_func={"sabre": "SABRE (+ best-error layout)", "dense": "dense subgraph",
                                      "trivial": "trivial (qi → physical i)"}.get)
            c1, c2 = st.columns(2)
            c1.number_input("transpiler seed", min_value=0, step=1, key="w_cm_seed", on_change=S.cb_compile,
                            args=("seed", "w_cm_seed"), persist_state=P)
            c2.text_input("physical qubits (optional)", key="w_cm_init", on_change=S.cb_compile,
                          args=("initial_layout", "w_cm_init"), persist_state=P, max_chars=400,
                          placeholder="e.g. 10, 11, 12", help="Force the layout: physical qubit for q0, q1, …")
            if ss.get("layout_error"):
                st.error(ss.layout_error)
            elif S.layout_problem():
                st.warning("The forced layout is ignored: %s. Clear it or list one physical qubit per circuit qubit." %
                           S.layout_problem(), icon=":material/warning:")
        else:
            if ss.get("w_cm_mode") not in ("abstract", "isa"):
                ss.w_cm_mode = cm["mode"]
            st.segmented_control("Gate set", ["abstract", "isa"], key="w_cm_mode", on_change=S.cb_compile,
                                 args=("mode", "w_cm_mode"), persist_state=P,
                                 format_func={"abstract": "Abstract (as drawn)", "isa": "ISA: RZ, √X, X, CX|CZ"}.get)
            if cm["mode"] == "isa":
                c1, c2 = st.columns(2)
                c1.selectbox("two-qubit gate", ["cx", "cz"], key="w_cm_b2", on_change=S.cb_compile,
                             args=("basis_2q", "w_cm_b2"), persist_state=P)
                c2.selectbox("coupling map", list(S.COUPLINGS), key="w_cm_cp", on_change=S.cb_compile,
                             args=("coupling", "w_cm_cp"), persist_state=P, format_func=S.COUPLINGS.get,
                             help="Which qubit pairs can interact. Grid and heavy-hex use the smallest lattice that "
                                  "holds the circuit; the compiled circuit keeps only the qubits it uses. See the "
                                  "**Topology & layout** tab for the graph and the SWAPs.")
                c1, c2 = st.columns(2)
                if ss.get("w_cm_lay") not in ("sabre", "trivial"):
                    ss.w_cm_lay = "sabre"
                c1.selectbox("qubit selection / routing", ["sabre", "trivial"], key="w_cm_lay", on_change=S.cb_compile,
                             args=("layout", "w_cm_lay"), persist_state=P,
                             format_func={"sabre": "optimized (SABRE)", "trivial": "trivial (qi → qi)"}.get)
                c2.select_slider("optimization level", [0, 1, 2, 3], key="w_cm_opt", on_change=S.cb_compile,
                                 args=("opt_level", "w_cm_opt"), persist_state=P)
                if cm["coupling"] != "all":
                    st.text_input("physical qubits (optional)", key="w_cm_init", on_change=S.cb_compile,
                                  args=("initial_layout", "w_cm_init"), persist_state=P, max_chars=400,
                                  placeholder="e.g. 0, 1, 2", help="Force the layout: lattice qubit for q0, q1, …")
                    if ss.get("layout_error"):
                        st.error(ss.layout_error)
                    elif S.layout_problem():
                        st.warning("The forced layout is ignored: %s. Clear it or list one physical qubit per "
                                   "circuit qubit." % S.layout_problem(), icon=":material/warning:")
        if compiled is not None:
            m = st.columns(3)
            m[0].metric("2q gates", W.count_str(compiled.implemented.count("2q")),
                        delta=compiled.implemented.count("2q") - compiled.logical.count("2q") or None,
                        delta_color="inverse", border=True)
            m[1].metric("depth", W.count_str(compiled.implemented.depth()), border=True)
            m[2].metric("1q pulses", W.count_str(compiled.implemented.count("1q")), border=True)
            if cm["mode"] == "isa" or device.is_device(ss.backend):
                st.caption("Measurement map (classical bit ← qubit): " +
                           ", ".join("c%d←q%d" % (k, v) for k, v in sorted(compiled.measure_map.items())))
            if device.is_device(ss.backend):
                phys = compiled.info.get("phys", [])
                st.caption("Physical qubits: " + ", ".join("q%d → %d" % (i, p) for i, p in enumerate(phys)))
                with st.expander("Calibration of these qubits", icon=":material/memory:", key="exp_cal"):
                    dm = device.get_model(ss.backend, tuple(phys))
                    st.dataframe(pd.DataFrame(dm.table()), hide_index=True, width="stretch", column_config={
                        "T1 (µs)": st.column_config.NumberColumn(format="%.1f"),
                        "T2 (µs)": st.column_config.NumberColumn(format="%.1f"),
                        "√X error": st.column_config.NumberColumn(format="%.2e"),
                        "P(1|0)": st.column_config.NumberColumn(format="%.4f"),
                        "P(0|1)": st.column_config.NumberColumn(format="%.4f")})
                    edges = sorted({(k[1], v) for k, v in dm.errors.items() if len(k[1]) == 2 and v is not None})
                    if edges:
                        st.caption("Two-qubit gates (%s) on these qubits: " % dm.two_qubit + "; ".join(
                            "q%d–q%d error %.2e" % (a[0], a[1], e) for a, e in edges if a[0] < a[1]))
                    med = dm.median_errors()
                    st.caption("Medians: 1q error %.1e, 2q error %.1e, readout %.2e, T1 %.0f µs, T2 %.0f µs." % (
                        med["r1"], med["r2"], med["ro"], med["t1"] * 1e6, med["t2"] * 1e6))


DIFF_VIEWS = {"aligned": "Logical over physical (aligned)", "routed": "Routed (layout + SWAPs)",
              "native": "Native gates", "executed": "Executed (optimized)"}


def lattice_buttons(prefix: str):
    """One click to compile the circuit onto a lattice (ISA gate set)."""
    row = st.container(horizontal=True, gap="small")
    for kind in ("line", "ring", "grid", "heavy-hex"):
        row.button("Compile on a %s" % S.COUPLINGS[kind], key="%s_try_%s" % (prefix, kind), on_click=S.cb_try_coupling,
                   args=(kind,), icon=":material/hub:", type="secondary")


def diff_toggle():
    """The composer's logical ↔ physical switch (the view itself is drawn full width below the Build columns)."""
    if compiled is None:
        return
    S.wdefault("w_diff", False)
    st.toggle("Show the compiled (physical) circuit", key="w_diff", persist_state=P,
              help="Overlay the logical circuit on the circuit the device runs: layout, routing SWAPs and every gate "
                   "rewritten in native gates (RZ, √X, X, CX/CZ/ECR). Shown below.")


def physical_view():
    """What the compiler did to the drawn circuit: logical over native, aligned, with SWAPs and a rewrite table."""
    if compiled is None or not ss.get("w_diff"):
        return
    st.markdown("**Compiled (physical) circuit** — each logical gate above the native gates that implement it")
    if not device.is_device(ss.backend) and ss.compile.get("mode", "abstract") == "abstract":
        st.info("**Abstract mode** runs the gates as drawn, so there is nothing to compare. Choose *ISA* under "
                "Compilation, a fake IBM device in the sidebar, or compile onto a lattice:", icon=":material/info:")
        lattice_buttons("diff")
        return
    ok, stg = W.guarded("The compile stages", K.get_stages, spec, ss.inputs, ss.basis, S.compile_options(),
                        ss.relabel, n=compiled.implemented.n)
    if not ok:
        return
    S.wdefault("w_diff_view", "aligned")
    st.segmented_control("View", list(DIFF_VIEWS), key="w_diff_view", format_func=DIFF_VIEWS.get, persist_state=P)
    view = ss.get("w_diff_view") or "aligned"
    m = st.columns(4)
    m[0].metric("SWAPs", stg.n_swaps, border=True,
                help="SWAPs the router inserted because two interacting qubits were not neighbors.")
    m[1].metric("2q gates", "%d → %d" % (compiled.logical.count("2q"), compiled.implemented.count("2q")),
                border=True, help="Two-qubit gates: logical → executed (after translation and optimization).")
    m[2].metric("qubits used", len(stg.phys), border=True,
                help="Physical qubits touched, including extra ones that routing passed states through.")
    m[3].metric("depth", "%d → %d" % (compiled.logical.depth(), compiled.implemented.depth()), border=True,
                help="Circuit depth: logical → executed.")
    labels = ["p%d" % p for p in stg.phys]
    if view == "aligned":
        if len(stg.native.ops) > STG.MAX_DIFF_OPS:
            st.caption("The aligned view is drawn for up to %d native operations; this circuit has %s. Showing the "
                       "routed circuit instead." % (STG.MAX_DIFF_OPS, f"{len(stg.native.ops):,}"))
            view = "routed"
        else:
            kw = STG.combined(stg)
            sp = kw.pop("spec")
            fig = CV.draw(sp, dark=dark(), width_px=FULL_PX, height_per_qubit=40, **kw)
            fig.update_layout(uirevision="diff_aligned")
            st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key="diff_aligned")
    if view in ("routed", "native"):
        sp = stg.routed if view == "routed" else stg.native
        mm = {c: stg.phys.index(p) for c, p in stg.final_layout.items() if p in stg.phys}
        fig = CV.draw(sp, dark=dark(), width_px=FULL_PX, measure_map=mm, qubit_labels=[
            "p%d (q%s)" % (p, q) if q is not None else "p%d (extra)" % p
            for p, q in ((p, {v: k for k, v in stg.initial_layout.items()}.get(p)) for p in stg.phys)])
        fig.update_layout(uirevision="diff_" + view)
        st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key="diff_" + view)
    elif view == "executed":
        phys = compiled.info.get("phys")
        fig = CV.draw(compiled.implemented, dark=dark(), width_px=FULL_PX, measure_map=compiled.measure_map,
                      qubit_labels=["p%d" % p for p in phys] if phys else None)
        st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key="diff_exec")
        st.caption("The circuit the simulator runs: the native circuit after optimization (neighboring RZ merged, "
                   "canceling pairs removed), on the physical qubits it uses.")
    if stg.initial_layout:
        st.caption("Initial layout: %s · measured on: %s" % (
            ", ".join("q%d→p%d" % (q, p) for q, p in sorted(stg.initial_layout.items())),
            ", ".join("q%d@p%s" % (q, p) for q, p in sorted(stg.final_layout.items()))))
    with st.expander("How each logical gate was rewritten", icon=":material/call_split:", key="exp_decomp"):
        st.dataframe(pd.DataFrame([{"logical gate": k, "native gates (a, b = its two qubits)": v}
                                   for k, v in stg.decomposition.items()]), hide_index=True, width="stretch")
        st.caption("RZ is virtual on IBM hardware (a frame change: no pulse, no error); √X and X are pulses. "
                   "Gates are read left to right in time.")
    W.why("compile_diff")


with tabs[0]:
    if tabs[0].open is not False:
        W.responsive_css()
        left, right = st.container(key="pg_build").columns([1.75, 1], gap="large")
        with left:
            template_card()
            with st.container(border=True):
                st.markdown("**Composer** — drag gates and noise channels onto the wires")
                DND.render(dark())
                diff_toggle()
            editors()
        with right:
            io_card()
            output_card()
            compile_card()
        if ss.get("w_diff"):
            with st.container(border=True):
                physical_view()


# ================================================================== Circuits
QISKIT_DRAW_MAX = 300                      # Matplotlib drawings of longer circuits are slow and unreadable
with tabs[1]:
    if tabs[1].open is not False:
        if compiled is None:
            st.info("Fix the circuit first.")
        else:
            S.wdefault("w_pv_kind", "zne")
            st.segmented_control("Implemented circuit of", list(PREVIEWS), key="w_pv_kind", format_func=PREVIEWS.get,
                                 persist_state=P)
            kind = ss.get("w_pv_kind") or "implemented"
            c = st.columns([1.2, 1, 0.9, 1.3], vertical_alignment="bottom")
            for k, v in (("w_pv_lam", 3.0), ("w_pv_fold", "global"), ("w_pv_dd", "X2"), ("w_pv_min1", True),
                         ("w_pv_seed", 0), ("w_pv_style", "composer")):
                S.wdefault(k, v)
            lam, folding, dd, seed, min1 = 3.0, "global", "X2", 0, True
            if kind == "zne":
                lam = float(c[0].number_input("noise scale λ", min_value=1.0, step=0.5, format="%.2f", key="w_pv_lam",
                                              persist_state=P))
                folding = c[1].selectbox("folding", ["global", "local"], key="w_pv_fold", persist_state=P)
            if kind == "dd":
                dd = c[0].selectbox("sequence", ["X2", "XY4"], key="w_pv_dd", persist_state=P)
            if kind == "pec":
                min1 = c[0].toggle("show an instance with insertions", key="w_pv_min1", persist_state=P,
                                   help="At low error rates most PEC instances insert nothing; this redraws until one does.")
            if kind in ("pec", "twirl", "cdr", "zne"):
                seed = int(c[2].number_input("random seed", min_value=0, step=1, key="w_pv_seed", persist_state=P))
            c[3].segmented_control("Drawing", ["composer", "qiskit"], key="w_pv_style", persist_state=P,
                                   format_func={"composer": "Composer", "qiskit": "Qiskit (IBM)"}.get)
            style = ss.get("w_pv_style") or "composer"
            ok, pv = W.guarded("The preview", preview, kind, compiled, ss.noise, lam=lam, folding=folding, dd=dd,
                               seed=seed, pec_min1=min1, n=compiled.implemented.n)
            if ok:
                title, impl = pv
                title, _, sub = title.partition(" (")
                l, r = st.columns(2, gap="medium")
                for col, sp, head in ((l, compiled.logical, "Logical circuit"), (r, impl, title)):
                    with col:
                        st.markdown("**%s**" % head)
                        if style == "qiskit" and len(sp.ops) > QISKIT_DRAW_MAX:
                            st.caption("%s operations: the Qiskit drawing would be a very large image, so this shows "
                                       "the composer view (drag to scroll)." % f"{len(sp.ops):,}")
                            st.plotly_chart(CV.draw(sp, dark=dark(), width_px=HALF_PX), width="stretch", config=TH.PLOT_CONFIG,
                                            key="pv_big_" + ("l" if sp is compiled.logical else "r"))
                        elif style == "qiskit":
                            try:
                                st.image(QD.png(sp, dark=dark()), width="stretch")
                            except Exception as e:
                                st.warning("Qiskit drawer unavailable (%s); showing the composer view." % type(e).__name__)
                                st.plotly_chart(CV.draw(sp, dark=dark(), width_px=HALF_PX), width="stretch",
                                                key="pv_fb_" + ("l" if sp is compiled.logical else "r"))
                        else:
                            fig = CV.draw(sp, dark=dark(), measure_map=compiled.measure_map if sp is not compiled.logical
                                          else None, width_px=HALF_PX)
                            fig.update_layout(uirevision="pv_" + head)
                            st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG,
                                            key="pv_" + ("l" if sp is compiled.logical else "r"))
                        added = sum(1 for o in sp.ops if o.tag in ("fold", "fold-inv", "pec", "twirl", "dd", "train"))
                        m = st.columns(4)
                        m[0].metric("operations", W.count_str(len([o for o in sp.ops if o.kind != "barrier"])))
                        m[1].metric("2q gates", W.count_str(sp.count("2q")))
                        m[2].metric("depth", W.count_str(sp.depth()))
                        m[3].metric("added", W.count_str(added) if sp is not compiled.logical else "–",
                                    help="operations the method inserted (folds, Paulis, pulses, changed angles)")
                if sub:
                    st.caption(sub.rstrip(")").capitalize() + ".")
            st.caption("Outlines mark what a method adds: shaded blocks are folded inverses U†, amber insertions are "
                       "PEC Paulis, teal ones twirl Paulis or changed training angles, violet ones DD pulses; cream boxes "
                       "are noise channels you placed.")
            W.why("composer")


# ================================================================== Topology & layout
NODE_METRICS = {"readout": ("readout error", "readout<br>error", ".1%"), "sx": ("√X error", "√X<br>error", ".1e")}


def swap_story(stg) -> list:
    """Each routing SWAP: the coupler, the logical qubits it exchanged, and the logical gate it made possible."""
    where = {p: q for q, p in stg.initial_layout.items()}           # physical -> logical, updated by every SWAP
    logical_ops = [o for o in stg.logical.ops if o.kind != "barrier"]
    rows = []
    for j, (ro, s) in enumerate(zip(stg.routed.ops, stg.src_routed)):
        if s >= 0:
            continue
        pa, pb = (stg.phys[q] for q in ro.qubits)
        qa, qb = where.get(pa), where.get(pb)
        need = ""
        for ro2, s2 in zip(stg.routed.ops[j + 1:], stg.src_routed[j + 1:]):
            if s2 >= 0 and len(ro2.qubits) == 2 and ({stg.phys[q] for q in ro2.qubits} & {pa, pb}):
                lo = logical_ops[s2]
                need = "%s on %s" % (lo.label(), ", ".join("q%d" % q for q in lo.qubits))
                break
        rows.append({"#": len(rows) + 1, "coupler": "p%d–p%d" % (pa, pb),
                     "exchanges": "%s ↔ %s" % ("q%d" % qa if qa is not None else "idle",
                                               "q%d" % qb if qb is not None else "idle"),
                     "enables": need or "(final placement)"})
        where[pa], where[pb] = qb, qa
        where = {k: v for k, v in where.items() if v is not None}
    return rows


def topology_tab():
    if compiled is None:
        st.info("Fix the circuit first.")
        return
    is_dev = device.is_device(ss.backend)
    cm = dict(S.COMPILE_DEFAULT, **ss.compile)
    if not is_dev and (cm["mode"] == "abstract" or cm["coupling"] == "all"):
        st.info("This circuit is compiled **%s**: every qubit can interact with every other, so nothing is routed. "
                "Put it on a real connectivity to see layout and SWAPs — or choose a fake IBM device in the sidebar "
                "(*Simulator*) for its calibrated heavy-hex processor." %
                ("abstract (as drawn)" if cm["mode"] == "abstract" else "on an all-to-all coupling map"),
                icon=":material/hub:")
        lattice_buttons("topo")
        W.why("topology")
        return
    n = compiled.logical.n
    opts = S.compile_options()
    ok, stg = W.guarded("The compile stages", K.get_stages, spec, ss.inputs, ss.basis, opts, ss.relabel, n=n)
    if not ok:
        return
    used = {p: "q%d" % q for q, p in stg.initial_layout.items()}
    if is_dev:
        di = LS.device_info(ss.backend)
        N, edges, kind, title = di.n, di.edges, "", device.device_label(ss.backend)
    else:
        cmap = coupling_map(cm["coupling"], n)
        N = cmap.size() if cmap is not None else n
        edges = sorted({tuple(sorted(e)) for e in cmap.get_edges()}) if cmap is not None else []
        kind = cm["coupling"] if cm["coupling"] in ("line", "ring", "grid", "heavy-hex") else ""
        title = "%s, %d qubits" % (S.COUPLINGS[cm["coupling"]], N)
    left, right = st.columns([1.6, 1], gap="large")
    with right:
        m = st.columns(2)
        m[0].metric("routing SWAPs", stg.n_swaps, border=True)
        m[1].metric("physical qubits used", len(stg.phys), border=True)
        m = st.columns(2)
        m[0].metric("2q gates added", compiled.implemented.count("2q") - compiled.logical.count("2q"), border=True,
                    help="Native two-qubit gates in the executed circuit minus the logical ones (SWAPs are 3 each; "
                         "optimization can remove some).")
        if is_dev:
            try:
                F = "%.3f" % LS.score(compiled, ss.backend)
            except Exception:                            # noqa: BLE001 - informational only
                F = "–"
            m[1].metric("success F", F, border=True, help=MX.LAYOUT_HELP["F"])
        rows = swap_story(stg)
        if rows:
            st.markdown("**Why each SWAP was inserted**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(38 + 35 * len(rows), 320),
                         column_config={"#": st.column_config.NumberColumn(width="small"),
                                        "enables": st.column_config.TextColumn(
                                            help="The next logical two-qubit gate that needed this SWAP: its "
                                                 "qubits were not neighbors before.")})
        else:
            st.success("No SWAPs: every two-qubit gate already acts on neighboring qubits.",
                       icon=":material/check_circle:")
    candidate, rank = None, None
    key = (S.spec_digest(spec), tuple(ss.inputs), ss.basis, repr(opts), ss.relabel, ss.backend)
    if is_dev and ss.get("layout_rank") and ss.layout_rank.get("key") == key:
        rank = ss.layout_rank["result"]
    with right:
        if is_dev:
            st.markdown("**Qubit optimization (noise-aware layout)**")
            st.caption("Places the compiled circuit on every equivalent set of qubits (same connections, VF2 subgraph "
                       "search) and ranks them by calibrated error, like *mapomatic*.")
            if st.button("Find the lowest-error physical qubits", key="topo_rank", icon=":material/travel_explore:",
                         type="primary"):
                ok_r, res = W.guarded("The layout search", K.layout_rank, S.spec_digest(spec), tuple(ss.inputs), ss.basis,
                                      repr(opts), ss.relabel, ss.backend, compiled, 12, n=n)
                if ok_r:
                    ss.layout_rank = dict(key=key, result=res)
                    rank = res
        elif cm["mode"] == "isa":
            st.caption("A lattice has no calibration: every qubit and coupler is equally good, so only the SWAP count "
                       "matters. Choose a fake IBM device in the sidebar to rank qubits by their measured errors.")
    if rank is not None:
        with right:
            if not rank["rows"]:
                st.warning("No placement found (the device graph has no free copy of this circuit's connections).")
            else:
                best = rank["rows"][0]
                df = pd.DataFrame([{"#": r["rank"], "physical qubits (q0, q1, …)": ", ".join(map(str,
                                    r["initial_layout"] or r["phys"])), "F": r["F"], "1 − F": r["error"],
                                    "vs now": r["F"] - rank["current"]} for r in rank["rows"]])
                st.dataframe(df, hide_index=True, width="stretch", height=min(38 + 35 * len(df), 280), column_config={
                    "F": st.column_config.NumberColumn(format="%.4f", help=MX.LAYOUT_HELP["F"]),
                    "1 − F": st.column_config.NumberColumn(format="%.4f", help=MX.LAYOUT_HELP["1 − F"]),
                    "vs now": st.column_config.NumberColumn(format="%+.4f", help=MX.LAYOUT_HELP["vs now"])})
                st.caption("%s examined%s · now F = %.4f, best F = %.4f." % (
                    W.plural(rank["examined"], "placement"), "" if rank["complete"] else " (search capped)",
                    rank["current"], best["F"]))
                S.wdefault("w_topo_pick", 1)
                pick = st.selectbox("Highlight layout", [r["rank"] for r in rank["rows"]], key="w_topo_pick",
                                    format_func=lambda k: "#%d" % k)
                chosen = next((r for r in rank["rows"] if r["rank"] == pick), best)
                candidate = chosen["phys"]
                txt = ", ".join(map(str, chosen["initial_layout"]))
                st.button("Use layout #%d (recompile on these qubits)" % pick, key="topo_use", on_click=S.cb_use_layout,
                          args=(txt,), icon=":material/check:", disabled=not chosen["initial_layout"],
                          help="Sets *physical qubits* under Compilation to %s; the same gates and SWAPs then run on "
                               "these qubits." % (txt or "–"))
        if cm.get("initial_layout"):
            right.button("Back to automatic qubit selection", key="topo_clear", on_click=S.cb_use_layout, args=("",),
                         icon=":material/undo:", type="tertiary")
    with left:
        c = st.columns([1, 1.2])
        metric = "readout"
        if is_dev:
            S.wdefault("w_topo_metric", "readout")
            c[0].segmented_control("Color qubits by", list(NODE_METRICS), key="w_topo_metric",
                                   format_func=lambda k: NODE_METRICS[k][0], persist_state=P)
            metric = ss.get("w_topo_metric") or "readout"
        node_err = edge_err = hover = None
        broken_n = broken_e = ()
        if is_dev:
            node_err = di.readout if metric == "readout" else di.e1
            edge_err, hover, broken_n, broken_e = di.e2, dict(di.hover), di.broken_nodes, di.broken_edges
            for q, p in stg.initial_layout.items():
                hover[p] = hover.get(p, "") + "<br>holds q%d at the start" % q
        fig = TOPO.figure(N, edges, kind=kind, used=used, swap_edges=stg.swap_edges, node_err=node_err,
                          edge_err=edge_err, broken_nodes=broken_n, broken_edges=broken_e, candidate=candidate,
                          node_hover=hover, title=title, dark=dark(), node_metric=NODE_METRICS[metric][1:])
        W.show_fig(fig, "topology")
        st.caption("Rings: where the circuit's qubits start (initial layout). Amber couplers: where routing inserted "
                   "SWAPs. %s" % ("Qubit color: %s (darker is worse); coupler color: two-qubit gate error; dotted "
                                  "gray: broken in this calibration (never used). Dashed teal rings: the highlighted "
                                  "candidate layout. Scroll to zoom, drag to pan." % NODE_METRICS[metric][0]
                                  if is_dev else "Scroll to zoom, drag to pan."))
    W.why("topology", extra=WHY["layout_search"] if is_dev else "")


with tabs[2]:
    if tabs[2].open is not False:
        topology_tab()


# ================================================================== Time travel
LIVE_MAX_QUBITS = 8            # live parameter updates re-simulate the density matrix on every slider move


def variational_sliders(params):
    """Sliders for the template's angles, grouped by layer."""
    groups = {}
    for k, p in enumerate(params):
        groups.setdefault(p.get("group", ""), []).append(k)
    per_row = 4 if len(params) > 4 else len(params)
    for g, ks in groups.items():
        if len(groups) > 1:
            st.caption(g)
        cols = st.columns(per_row)
        for j, k in enumerate(ks):
            p = params[k]
            S.wdefault("w_var_%d" % k, float(p["value"]))
            cols[j % per_row].slider("%s · %s" % (p["name"], p.get("what", "")), float(p["lo"]), float(p["hi"]),
                                     key="w_var_%d" % k, step=0.01, format="%.2f", persist_state=P,
                                     help="Template value %.4f. Range: one period of the gate." % p["value"])


def landscape(params, vals, comp_live, obs_label):
    """Energy landscape along one parameter, or over (γ, β) for QAOA with p = 1 (behind a button)."""
    two_d = len(params) == 2 and params[0]["name"].startswith("γ") and params[1]["name"].startswith("β")
    modes = (["2d"] if two_d else []) + ["1d"]
    S.wdefault("w_var_scan_mode", modes[0])
    if ss.get("w_var_scan_mode") not in modes:
        ss.w_var_scan_mode = modes[0]
    c = st.columns([1.3, 1.4, 1, 1], vertical_alignment="bottom")
    mode = c[0].segmented_control("Scan", modes, key="w_var_scan_mode", persist_state=P,
                                  format_func={"2d": "2D: γ × β", "1d": "1D: one angle"}.get) or modes[0]
    k = 0
    if mode == "1d":
        S.wdefault("w_var_scan_k", 0)
        if not isinstance(ss.get("w_var_scan_k"), int) or ss.w_var_scan_k >= len(params):
            ss.w_var_scan_k = 0
        k = c[1].selectbox("angle", list(range(len(params))), key="w_var_scan_k", persist_state=P,
                           format_func=lambda i: "%s · %s" % (params[i]["name"], params[i].get("what", "")))
    S.wdefault("w_var_scan_pts", 15 if mode == "2d" else 41)
    pts = int(c[2].number_input("points per axis", min_value=5, max_value=41 if mode == "2d" else 401, step=2,
                                key="w_var_scan_pts", persist_state=P))
    base_key = (S.signature(), mode, k, pts)
    go_ = c[3].button("Scan", key="var_scan_btn", icon=":material/insights:", type="primary", width="stretch",
                      help="One exact simulation (ideal and noisy, unmitigated) per grid point; the other angles stay "
                           "at their slider values.")
    if go_:
        opts = S.compile_options()
        fixed = comp_live.info.get("initial_layout")
        if fixed and not opts.initial_layout and (opts.device or opts.mode == "isa"):
            import dataclasses as _dc                   # the whole scan runs on the same physical qubits
            opts = _dc.replace(opts, initial_layout=tuple(fixed))    # (and skips the layout search per point)

        def make(sp):
            return CP.build(sp, ss.inputs, ss.basis, opts, ss.relabel)
        axes = [(0, np.linspace(params[0]["lo"], params[0]["hi"], pts)),
                (1, np.linspace(params[1]["lo"], params[1]["hi"], pts))] if mode == "2d" else \
            [(k, np.linspace(params[k]["lo"], params[k]["hi"], pts))]
        total = pts ** len(axes)
        bar = st.progress(0.0, text="Scanning %s points…" % f"{total:,}")
        t0 = time.time()

        def progress(f):
            el = time.time() - t0
            bar.progress(min(f, 1.0), text="Scanning %s points… %.0f s left" % (f"{total:,}", el * (1 - f) / max(f, 1e-9)))
        ok, res = W.guarded("The landscape scan", VAR.scan, make, spec, params, vals, ss.noise, ss.observable, axes,
                            ss.backend, progress=progress, n=comp_live.implemented.n)
        bar.empty()
        if ok:
            ss.var_scan = dict(key=base_key, res=res, vals=list(vals))
    sc = ss.get("var_scan")
    if not sc or sc["key"][1:] != base_key[1:]:
        st.caption("Press **Scan** to map ⟨O⟩ over %s." % ("the (γ, β) plane" if mode == "2d" else "this angle"))
        return
    res = sc["res"]
    maximise = ss.observable.key == "cut_ratio"
    if sc["key"][0] != base_key[0]:
        st.caption("Scanned before the last change (noise, circuit or other angles); press Scan to refresh.")
    if mode == "2d":
        gx, gy = res["grids"]
        fig = SV.landscape_2d(gx, gy, res["ideal"], res["noisy"], (vals[0], vals[1]), "γ₁", "β₁",
                              obs_label, maximise, dark=dark())
        W.show_fig(fig, "var_landscape_2d")
        Z = np.asarray(res["noisy"])
        j = np.unravel_index(np.nanargmax(Z) if maximise else np.nanargmin(Z), Z.shape)
        best = list(vals)
        best[0], best[1] = gx[j[0]], gy[j[1]]
    else:
        x = res["grids"][0]
        kk = res["axes"][0]
        fig = SV.landscape_1d(x, res["ideal"], res["noisy"], vals[kk], params[kk]["name"], obs_label, dark=dark())
        W.show_fig(fig, "var_landscape_1d")
        Z = np.asarray(res["noisy"])
        j = int(np.nanargmax(Z) if maximise else np.nanargmin(Z))
        best = list(vals)
        best[kk] = x[j]
    zi, zn = np.asarray(res["ideal"]), np.asarray(res["noisy"])
    st.caption("Grid optimum (%s): ideal %.4f, noisy %.4f. Noise flattens the landscape toward Tr(O)/2ⁿ; the "
               "noisy optimum is usually at almost the same angles, just shallower." % (
                   "max" if maximise else "min", np.nanmax(zi) if maximise else np.nanmin(zi),
                   np.nanmax(zn) if maximise else np.nanmin(zn)))
    st.button("Move the sliders to the noisy optimum", key="var_to_best", on_click=S.cb_var_set, args=(best,),
              icon=":material/my_location:")


def variational_panel(params):
    """Live θ sliders; returns the current values."""
    with st.container(border=True):
        st.markdown("**Live parameters θ** — move a slider: the circuit, the Bloch spheres and ⟨O⟩ below follow at "
                    "once (exact simulation of the density matrix).")
        variational_sliders(params)
        b = st.container(horizontal=True, gap="small")
        b.button("Reset to the template's angles", key="var_reset", on_click=S.cb_var_reset, icon=":material/undo:",
                 type="tertiary")
        if b.button("Use these angles in the circuit", key="var_apply", icon=":material/check:", type="secondary",
                    help="Writes the angles into the circuit, so Run simulation and Mitigate & compare use them."):
            S.cb_var_apply()
            st.rerun()                                 # the whole page: every tab shows the new circuit
        slot = st.container()
    return S.var_values(params), slot


def live_readout(tl, comp_live, params, vals):
    """⟨O⟩ of the final state at the slider angles (ideal and noisy, before readout), and the landscape scan."""
    label = ss.observable.label
    try:
        M = HM.observable_matrix(comp_live, ss.observable, tl.spec.n) if (
            tl.rho_noisy is not None and tl.spec.n == comp_live.implemented.n) else None
    except Exception:                                   # noqa: BLE001 - an unusual layout: skip the numbers
        M = None
    m = st.columns(3)
    if M is not None:
        e_n = float(np.real(np.trace(tl.rho_noisy[-1] @ M)))
        e_i = float(np.real(np.trace(tl.rho_ideal[-1] @ M)))
        base = ss.var_spec["params"]
        at_template = all(abs(v - p["value"]) < 1e-9 for v, p in zip(vals, base))
        m[0].metric("⟨O⟩ ideal", "%.4f" % e_i, border=True, help="Tr(ρ_ideal O) at these angles.")
        m[1].metric("⟨O⟩ noisy", "%.4f" % e_n, delta="%+.4f vs ideal" % (e_n - e_i), delta_color="off", border=True,
                    help="Tr(ρ O) with the sidebar's noise, before readout error (unmitigated).")
        m[2].metric("angles", "template" if at_template else "changed", border=True,
                    help="Whether the sliders are at the template's (optimized) angles.")
    st.caption("Observable: %s. The noisy value moves with the angles *and* shrinks toward Tr(O)/2ⁿ with the noise." %
               W.md_escape(label[:80]))
    with st.expander("Energy landscape ⟨O⟩(θ)", icon=":material/landscape:", key="exp_var_landscape"):
        landscape(params, vals, comp_live, label)
        W.why("variational")


@st.fragment
def time_travel():
    if compiled is None:
        st.info("Fix the circuit first.")
        return
    params = S.var_params()
    live = bool(params) and compiled.implemented.n <= LIVE_MAX_QUBITS
    comp_tt, live_sig, vals, slot = compiled, None, None, None
    if params:
        vals, slot = variational_panel(params)
        if not live:
            st.caption("%d qubits: live updates are off above %d qubits. Use *Use these angles in the circuit*, then "
                       "Run simulation." % (compiled.implemented.n, LIVE_MAX_QUBITS))
        else:
            live_spec = VAR.apply(spec, params, vals)
            ok, comp_tt = W.guarded("Compiling the live circuit", K.get_compiled, live_spec, ss.inputs, ss.basis,
                                    S.compile_options(), ss.relabel, n=compiled.implemented.n)
            if not ok:
                return
            live_sig = S.signature(("live", tuple(round(v, 12) for v in vals)))
    if live_sig is None and ss.results is None and ss.timeline is None:
        st.info("Press **Run simulation** in the sidebar to compute the states along the circuit.")
        return
    for k, v in (("w_tt_kind", "implemented"), ("w_tt_lam", 3.0), ("w_tt_fold", "global"), ("w_tt_dd", "X2"),
                 ("w_tt_view", "both"), ("w_tt_trail", True), ("w_tt_rho", "noisy")):
        S.wdefault(k, v)
    c = st.columns([2, 0.9, 0.9, 1.9], vertical_alignment="bottom")
    kind = c[0].selectbox("Inspect", list(INSPECT), key="w_tt_kind", format_func=INSPECT.get, persist_state=P)
    lam, folding, dd = 3.0, "global", "X2"
    if kind == "zne":
        lam = float(c[1].number_input("λ", min_value=1.0, step=0.5, format="%.2f", key="w_tt_lam", persist_state=P))
        folding = c[2].selectbox("folding", ["global", "local"], key="w_tt_fold", persist_state=P)
    if kind == "dd":
        dd = c[1].selectbox("sequence", ["X2", "XY4"], key="w_tt_dd", persist_state=P)
    c[3].segmented_control("Bloch view", ["3d", "flat", "both"], key="w_tt_view", persist_state=P,
                           format_func={"3d": "3D spheres", "flat": "Flat views", "both": "Both"}.get)
    # the signature of the state comp_tt was compiled from: the trace cache is shared by every session, so it is
    # never keyed by an older run's signature while the circuit has changed since
    sig = live_sig or S.signature()
    if not live_sig and ss.timeline and ss.timeline["key"] == (kind, lam, folding, dd) and ss.timeline["sig"] == sig:
        tl = ss.timeline["tl"]
    else:
        with st.spinner("Tracing the state gate by gate…"):
            ok, tl = W.guarded("Time travel", K.timeline, sig, comp_tt, K.noise_json(ss.noise), kind, lam, folding,
                               dd, ss.backend, n=comp_tt.implemented.n)
        if not ok:
            return
    if live_sig and slot is not None:
        with slot:
            live_readout(tl, comp_tt, params, vals)
    if stale and not live_sig:
        st.info("Time travel follows the current circuit and noise; the comparison results are from an earlier run. "
                "Press Run simulation to refresh them.", icon=":material/history:")
    n_steps = len(tl.ops)
    skey = "w_tt_step_" + kind                       # each implemented circuit keeps its own position
    S.wdefault(skey, n_steps)                        # and starts at the end (the final state)
    if not isinstance(ss[skey], int) or ss[skey] > n_steps or ss[skey] < 0:
        ss[skey] = n_steps

    def nudge(d):
        ss[skey] = int(min(max(ss[skey] + d, 0), n_steps))
    b = st.columns([0.6, 6, 0.6], vertical_alignment="center")
    b[0].button("◀", key="tt_prev", on_click=nudge, args=(-1,), help="previous step", width="stretch")
    b[1].slider("Step (0 = initial state)", 0, max(n_steps, 1), key=skey, persist_state=P,
                help="Drag to scrub through the circuit; the operation outlined in blue (only its gate and wires) is the "
                         "last executed step.")
    b[2].button("▶", key="tt_next", on_click=nudge, args=(1,), help="next step", width="stretch")
    step = int(min(ss[skey], n_steps))
    st.markdown("**Step %d / %d** — %s" % (step, n_steps, tl.labels[step]) + (("  \n" + tl.note) if tl.note else ""))
    exec_spec = CircuitSpec(tl.spec.n, tuple(tl.ops), tl.spec.name)
    fig = CV.draw(exec_spec, dark=dark(), cursor_op=step - 1 if step > 0 else None, width_px=FULL_PX,
                  measure_map=comp_tt.measure_map)
    fig.update_layout(uirevision="tt_comp")
    st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key="tt_comp")
    view = ss.get("w_tt_view") or "both"
    tc = st.columns([1.4, 1, 2], vertical_alignment="center")
    trail = tc[0].toggle("Show where each Bloch vector came from", key="w_tt_trail", persist_state=P)
    S.wdefault("w_tt_trail_n", 8)
    back = int(tc[1].number_input("steps back", min_value=1, step=1, key="w_tt_trail_n", persist_state=P,
                                  disabled=not trail, help="Length of the path drawn behind each arrow."))
    lo = max(0, step - back)
    tn = tl.bloch_noisy[lo: step + 1] if trail else None
    ti = tl.bloch_ideal[lo: step + 1] if trail else None
    nq = tl.spec.n
    if nq > 16 and view != "flat":
        st.caption("%d qubits: showing flat views (browsers limit the number of 3D scenes on one page)." % nq)
        view = "flat"
    elif nq > 4:
        st.caption("%d qubits: the spheres wrap into rows of %d so every qubit stays in view." % (nq, 4))
    if view in ("3d", "both"):
        st.plotly_chart(SV.bloch_row(tl.bloch_noisy[step], tl.bloch_ideal[step], dark=dark(), height=360,
                                     trail_noisy=tn, trail_ideal=ti), width="stretch", config=TH.PLOT_CONFIG,
                        key="tt_bloch")
    if view in ("flat", "both"):
        st.plotly_chart(SV.bloch_projections(tl.bloch_noisy[step], tl.bloch_ideal[step], dark=dark(), trail_noisy=tn,
                                             height=500 if nq <= 4 else 470), width="stretch", config=TH.PLOT_CONFIG,
                        key="tt_flat")
    rows = []
    for q in range(nq):
        a, b_ = tl.bloch_noisy[step][q], tl.bloch_ideal[step][q]
        th_a, ph_a = SV.angles(a)
        th_b, ph_b = SV.angles(b_)
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b_))
        why = "pure, as ideal" if la > 0.995 and lb > 0.995 else (
            "entangled (the ideal vector is short too)" if lb < 0.98 and la >= lb - 0.02 else
            "noise made it mixed" if la < lb - 0.005 else "–")
        rows.append({"qubit": "q%d" % q, "|r| noisy": la, "|r| ideal": lb, "θ noisy (°)": th_a, "θ ideal (°)": th_b,
                     "φ noisy (°)": ph_a, "φ ideal (°)": ph_b, "tilt (°)": SV.tilt_deg(a, b_), "length": why})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(
                     format="%.3f" if "|r|" in k else "%.1f",
                     help="azimuth; undefined (None) when the vector points along ±z" if k.startswith("φ") else None)
                                for k in ("|r| noisy", "|r| ideal", "θ noisy (°)", "θ ideal (°)", "φ noisy (°)",
                                          "φ ideal (°)", "tilt (°)")})
    W.why("time_travel")
    l, r = st.columns([1, 1.15], gap="medium")
    with l:
        if tl.rho_noisy is None:
            st.info("The density matrix of %d qubits has %s entries per step, so time travel keeps per-step "
                    "summaries only (Bloch vectors, fidelity, purity)." % (nq, f"{4 ** nq:,}"))
        elif nq > 7:
            st.info("A %d×%d density matrix is too large to draw usefully; the fidelity and purity on the right "
                    "summarize it." % (2 ** nq, 2 ** nq))
        else:
            st.segmented_control("Density matrix", ["noisy", "ideal", "error"], key="w_tt_rho", persist_state=P,
                                 format_func={"noisy": "noisy ρ", "ideal": "ideal ρ", "error": "ρ − ρ_ideal"}.get)
            mode = ss.get("w_tt_rho") or "noisy"
            rho = tl.rho_ideal[step] if mode == "ideal" else tl.rho_noisy[step]
            st.plotly_chart(SV.density_hinton(rho, nq, dark=dark(), ref=tl.rho_ideal[step],
                                              mode="error" if mode == "error" else "state"),
                            width="stretch", config=TH.PLOT_CONFIG, key="tt_dm")
            W.why("density")
    with r:
        st.plotly_chart(SV.trace(list(range(n_steps + 1)), tl.fid, tl.pur, step, tl.labels, dark=dark()),
                        width="stretch", config=TH.PLOT_CONFIG, key="tt_trace")
        W.why("fidelity")
        if tl.rho_noisy is not None and tl.spec.n == comp_tt.implemented.n and tl.spec.n <= 10:
            try:
                M = HM.observable_matrix(comp_tt, ss.observable, tl.spec.n)
                ev_n = [float(np.real(np.trace(r_ @ M))) for r_ in tl.rho_noisy]
                ev_i = [float(np.real(np.trace(r_ @ M))) for r_ in tl.rho_ideal]
                st.plotly_chart(SV.observable_trace(list(range(n_steps + 1)), ev_n, ev_i, step, tl.labels,
                                                    ss.observable.label, dark=dark()),
                                width="stretch", config=TH.PLOT_CONFIG, key="tt_obs")
                st.caption("%s after step %d: noisy %.4f, ideal %.4f (as if measured now, before readout error)."
                           % (W.md_escape(ss.observable.label[:60]), step, ev_n[step], ev_i[step]))
            except Exception:                          # a stale timeline or an unusual layout: skip the extra chart
                pass


with tabs[3]:
    if tabs[3].open is not False:
        time_travel()


# ================================================================== Mitigate & compare
with tabs[4]:
    if tabs[4].open is not False:
        st.markdown("Pick any number of contenders, or build your own recipe. Everything runs when you press "
                    "**Run simulation** in the sidebar.")
        S.wdefault("w_rc_pick", list(ss.recipe_names))
        st.multiselect("Contenders", list(RECIPE_PRESETS), key="w_rc_pick", on_change=S.cb_recipes, persist_state=P,
                       help="Run them with **Run simulation** in the sidebar — the one button that runs the "
                            "Playground (time travel, every contender, the analysis).")
        with st.expander("Build a custom contender", icon=":material/tune:", key="rc_builder"):
            W.recipe_builder(ss.noise, int(ss.n), ss.symmetry)
        if ss.custom_recipes:
            st.markdown("**Saved custom recipes**")
            st.caption("Ticked recipes are contenders in the next run. A guide that sets up the Playground replaces "
                       "the contenders (or adds to them, if chosen on the Learn page) but never deletes a saved recipe.")

            def rm(i):
                key = Recipe.from_json(ss.custom_recipes[i]).key()
                ss.custom_recipes = [x for j, x in enumerate(ss.custom_recipes) if j != i]
                ss.custom_active = [k for k in S.custom_active_keys() if k != key]

            def toggle(key, wkey):
                act = [k for k in S.custom_active_keys() if k != key]
                ss.custom_active = act + ([key] if ss.get(wkey) else [])
            active = set(S.custom_active_keys())
            for i, d in enumerate(list(ss.custom_recipes)):
                r_ = Recipe.from_json(d)
                cc = st.columns([6, 1], vertical_alignment="center")
                wk = "rc_on_%d" % i
                ss[wk] = r_.key() in active                    # mirrors the model (set before the widget renders)
                cc[0].checkbox(r_.title(), key=wk, on_change=toggle, args=(r_.key(), wk))
                cc[1].button("Remove", key="rc_rm_%d" % i, icon=":material/close:", on_click=rm, args=(i,),
                             help="Delete this saved recipe.")
        res = ss.results
        if res is None:
            st.info("No results yet: press **Run simulation** in the sidebar.", icon=":material/play_arrow:")
        else:
            # the ZIP describes the run: its own state snapshot (and script), not the settings edited since
            RES.render(res, stale=stale, bundle=RES.bundle_factory(res, res.get("state") or S.to_json(),
                                                                   "simulator: " + sim_label(res.get("backend", "numpy"))),
                       zip_name="qem_explorer_playground.zip")


# ================================================================== Export
with tabs[5]:
    if tabs[5].open is not False:
        st.markdown("A standalone Python script that rebuilds this exact state — circuit, compilation, noise (or the "
                    "fake device) and every contender — with Qiskit and Qiskit Aer, and prints the same numbers as the "
                    "app. For real hardware from the browser, use **Run on IBM Quantum** in the sidebar.")
        state = S.to_json()

        def gen():
            """Callback: the script for the state at the click; the app's values only if they belong to it."""
            ss.export_error = ""
            try:
                res_ = ss.get("results")
                current = res_ is not None and res_["sig"] == S.signature()
                expected = ([o.result.mean if o.result is not None else None for o in res_["outs"]]
                            if current else None)            # one value per contender, in the state's order
                ss.export_code = EX.script(S.to_json(), expected)
            except Exception as e:                          # noqa: BLE001 - never crash the page from a callback
                ss.export_code = ""
                ss.export_error = "The script could not be generated (%s: %s)." % (type(e).__name__, str(e)[:200])
        c1, c2, c3 = st.columns(3, vertical_alignment="center")
        c1.button("Generate Qiskit script", type="primary", icon=":material/code:", key="ex_gen", on_click=gen)
        c2.download_button("Download state (.json)", lambda st_=state: json.dumps(st_, indent=1),
                           "qem_explorer_state.json", "application/json", key="ex_json", icon=":material/save:")

        def exp_yaml(st_=state):
            from qemx import experiment as XP
            return XP.dump_yaml(XP.from_state(st_, "playground"))
        c2.download_button("Download experiment (.yaml)", exp_yaml, "qem_explorer_experiment.yaml",
                           "application/x-yaml", key="ex_yaml", icon=":material/description:",
                           help="The same experiment for the command line: `python -m qemx.experiment run "
                                "qem_explorer_experiment.yaml --out results` reproduces this page's table. YAML and "
                                "the state file are validated by the same schema.")
        if ss.get("export_error"):
            st.error(W.md_escape(ss.export_error), icon=":material/error:")
        up = c3.file_uploader("Load a state (.json) or experiment (.yaml)", type=["json", "yaml", "yml"], key="ex_up",
                              label_visibility="collapsed", max_upload_size=20)

        def load_state():
            f = ss.get("ex_up")
            ss.import_error = ""
            if f is None:
                return
            try:
                raw = f.getvalue()
                if len(raw) > S.MAX_STATE_BYTES:
                    raise ValueError("the file is larger than %d MB" % (S.MAX_STATE_BYTES // 2 ** 20))
                if f.name.lower().endswith((".yaml", ".yml")):
                    from qemx import experiment as XP
                    S.from_json(XP.to_state(XP.loads(raw.decode("utf-8"))))     # validated like the CLI
                else:
                    S.from_json(json.loads(raw.decode("utf-8")))
                ss.import_ok = True
            except (ValueError, UnicodeDecodeError) as e:
                ss.import_error = "Could not load this file: %s" % W.md_escape(str(e)[:300])
            except Exception as e:                     # anything else is still just a bad file, never a crash
                ss.import_error = "Could not load this file (%s)." % type(e).__name__
        if up is not None:
            st.button("Load this state", key="ex_load", on_click=load_state, icon=":material/upload:")
        if ss.get("import_error"):
            st.error(ss.import_error)
        elif ss.pop("import_ok", False):
            st.success("State loaded.")
        if ss.get("export_code"):
            st.download_button("Download script (.py)", ss.export_code, "qem_explorer_export.py", "text/x-python",
                               key="ex_py", icon=":material/download:")
            st.code(ss.export_code, language="python", line_numbers=True, height=520)
        st.caption("A token is a password. Before this notebook goes into a repository or an appendix, clear it and use "
                   "one of the other two. Neither the script nor the state file ever contains a token: the script reads "
                   "IBM Quantum credentials only from the IBM_QUANTUM_TOKEN environment variable or a saved account, "
                   "and never prints them.")


# ================================================================== header (last: reflects every change above)
stale = ss.results is None or ss.results["sig"] != S.signature()
with header:
    chips = st.container(horizontal=True, gap="small")
    with chips:
        st.badge(W.plural(int(ss.n), "qubit"), color="blue")
        if compiled is not None:
            st.badge("%s · depth %d" % (W.plural(compiled.implemented.count("2q"), "two-qubit gate"),
                                        compiled.implemented.depth()),
                     color="gray")
        if device.is_device(ss.backend):
            st.badge("device: %s" % device.fake_class(ss.backend), color="orange")
        else:
            st.badge("noise: %s" % ss.noise_preset, color="orange" if not ss.noise.is_ideal else "green")
            st.badge(BACKENDS[ss.backend], color="gray")
        st.badge("observable %s" % W.plain(ss.observable.label), color="violet")
        if ss.results is None:
            st.badge("not run yet: press Run simulation", icon=":material/play_arrow:", color="yellow")
        elif stale:
            st.badge("results are out of date: press Run simulation", icon=":material/refresh:", color="red")
        else:
            st.badge("results up to date", icon=":material/check:", color="green")
