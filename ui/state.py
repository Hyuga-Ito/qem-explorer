"""Session state for every page.

Two kinds of keys live in ``st.session_state``:

* **Model keys** (``n``, ``ops``, ``inputs``, ``noise``, ``compile``, ``backend``, ``shots``, …) hold the
  truth. Pages read them; nothing but callbacks writes them.
* **Widget keys** (prefix ``w_``) belong to widgets. Every widget is created with a fixed key, no
  ``value``/``index`` argument and ``persist_state="session"``, so its value survives reruns, hidden
  tabs and page switches. A widget writes the model in its ``on_change`` callback.

When the model changes as a whole (loading a template or a saved state, resizing the register, an edit
in the drag-and-drop composer) the change runs in a callback — before any widget exists in that
rerun — and ``push_widgets()`` copies the model into the widget keys. There are no epoch keys and no
widget whose key encodes a value, so a widget is never silently recreated with a default.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np

import streamlit as st

from qemx import observables as OBS
from qemx import templates as T
from qemx.circuit import CircuitSpec, Op
from qemx.compile import CompileOptions
from qemx.methods import Recipe
from qemx.noise import PRESETS, NoiseSpec
from qemx.recipes import DEFAULT_SET, PRESETS as RECIPE_PRESETS, preset
from ui.tooltips import install as _plain_tooltips

_plain_tooltips()          # every help tooltip shows Unicode math (ui/tooltips.py)

DEFAULT_NOISE = "Hardware-realistic (illustrative)"
try:
    from qemx import __version__ as VERSION              # state files carry the app's version
except Exception:                                         # pragma: no cover
    VERSION = "unknown"
HISTORY = 200                           # undo steps kept (memory only; not a circuit-size limit)
NOISE_FIELDS = [f.name for f in dataclasses.fields(NoiseSpec) if f.name != "field_samples"]
from qemx.state_schema import COMPILE_DEFAULT, COUPLINGS, parse_layout  # noqa: E402,F401


def ss():
    return st.session_state


# ------------------------------------------------------------------ init
def init():
    s = ss()
    if s.get("qx_init"):
        return
    from qemx.resources import default_time_limit
    s.ops_version = 0
    s.history, s.future = [], []
    s.compile = dict(COMPILE_DEFAULT)
    s.noise = PRESETS[DEFAULT_NOISE]
    s.noise_preset = DEFAULT_NOISE
    s.backend = "numpy"
    s.shots = 32768                     # total shot budget per method and estimate (qemx/budget.py)
    s.t_shot_us = 250.0
    s.time_limit = default_time_limit()
    s.recipe_names = list(DEFAULT_SET)
    s.custom_recipes = []            # the saved library of custom recipes (state files, recipe builder, guides)
    s.custom_active = []             # keys of the saved custom recipes that are active contenders
    s.results = None
    s.timeline = None
    s.page_tab = {}
    _load_template_model("qft", T.defaults("qft"))
    s.qx_init = True
    push_widgets()


def wdefault(key: str, value):
    """Give a widget key its first value (only if the key does not exist yet; safe before the widget renders)."""
    if key not in ss():
        ss()[key] = value


# ------------------------------------------------------------------ model -> widgets
def push_widgets():
    """Copy the model into every mirrored widget key. Call only from callbacks or before widgets render."""
    s = ss()
    s.w_backend = s.backend
    s.w_shots_num = int(s.shots)
    s.w_shots_sl = nearest_shots(int(s.shots))
    s.w_tshot = float(s.t_shot_us)
    s.w_tlimit = float(s.time_limit)
    s.w_nz_preset = s.noise_preset if s.noise_preset in list(PRESETS) + ["Custom"] else "Custom"
    for f in NOISE_FIELDS:
        v = float(getattr(s.noise, f))
        s["w_nz_%s_num" % f] = v
        s["w_nz_%s_sl" % f] = min(max(v, 0.0), NOISE_SLIDER_MAX.get(f, 1.0))
    for q in range(int(s.n)):
        s["w_in_%d" % q] = s.inputs[q]
        s["w_bs_%d" % q] = s.basis[q]
    s.w_obs = s.observable.key
    s.w_relabel = bool(s.relabel)
    cm = s.compile
    s.w_cm_mode = cm["mode"]
    s.w_cm_b2 = cm["basis_2q"]
    s.w_cm_cp = cm["coupling"]
    s.w_cm_lay = cm["layout"]
    s.w_cm_opt = int(cm["opt_level"])
    s.w_cm_seed = int(cm.get("seed", 11))
    s.w_cm_init = str(cm.get("initial_layout", "") or "")
    s.w_rc_pick = list(s.recipe_names)
    s.w_tpl = s.template


SHOT_STEPS = [2 ** k for k in range(6, 21)]          # 64 … 1,048,576 for the slider; typing allows any count


def nearest_shots(n: int) -> int:
    return min(SHOT_STEPS, key=lambda v: abs(math.log(max(n, 1)) - math.log(v)))


# ------------------------------------------------------------------ circuit state
def _ops_dicts(spec: CircuitSpec) -> list:
    return [dict(name=o.name, qubits=list(o.qubits), params=list(o.params), kind=o.kind, tag=o.tag) for o in spec.ops]


def spec() -> CircuitSpec:
    """The circuit as a CircuitSpec. Rebuilt only when the operations change (every writer of ``ops`` replaces the
    list and bumps ``ops_version``), so the many calls per rerun cost nothing for deep circuits."""
    s = ss()
    ident = (id(s.ops), len(s.ops), s.get("ops_version"), int(s.n), s.get("template_name", "circuit"))
    hit = s.get("_qx_spec_cache")
    if hit is not None and hit[0] == ident:
        return hit[1]
    sp = CircuitSpec(int(s.n), tuple(Op(d["name"], tuple(d["qubits"]), tuple(d["params"]), d["kind"],
                                        d.get("tag", "")) for d in s.ops), s.get("template_name", "circuit"))
    s["_qx_spec_cache"] = (ident, sp)
    return sp


def spec_digest(sp: CircuitSpec) -> str:
    """A content hash of a circuit, computed once per CircuitSpec object (cache keys, signatures)."""
    d = getattr(sp, "_qx_digest", None)
    if d is None:
        d = hashlib.sha1(sp.key().encode()).hexdigest()
        object.__setattr__(sp, "_qx_digest", d)              # CircuitSpec is frozen; this is a private memo
    return d


def memo(obj, name: str, fn):
    """fn() computed once per (immutable) object, e.g. the text or the table rows of the cached CircuitSpec."""
    attr = "_qx_memo_" + name
    v = getattr(obj, attr, None)
    if v is None:
        v = fn()
        object.__setattr__(obj, attr, v)
    return v


def _remember():
    s = ss()
    s.history = (s.get("history", []) + [(int(s.n), [dict(d) for d in s.ops])])[-HISTORY:]
    s.future = []


def set_circuit(new: CircuitSpec, remember: bool = True):
    """Replace the circuit (callback context). Inputs, bases and the observable follow the register size.
    Atomic: everything that depends on the register size is computed first, so a failure (e.g. a register too
    large for its observable) raises before the model changes."""
    s = ss()
    resized = _resized_model(new.n) if int(s.get("n", new.n)) != new.n else None
    if remember and "ops" in s:
        _remember()
    if resized is not None:
        _apply_model(resized)
    s.n = new.n
    s.ops = _ops_dicts(new)
    s.ops_version = s.get("ops_version", 0) + 1


def _resized_model(n: int) -> dict:
    """The size-dependent model for n qubits (inputs, bases, observable, symmetry), computed without assigning."""
    s = ss()
    n = int(n)
    out = dict(inputs=(list(s.get("inputs", [])) + ["0"] * n)[:n], basis=(s.get("basis", "") + "Z" * n)[:n])
    o = s.get("observable")
    if o is None or o.n != n:
        out["observable"] = OBS.default_observable(n)    # dense when it fits, a Pauli sum otherwise (no 2^n crash)
    sym = s.get("symmetry")
    if sym and max(sym[0]) >= n:
        out["symmetry"] = None
    return out


def _apply_model(m: dict):
    s = ss()
    for k, v in m.items():
        s[k] = v


def _resize_model(n: int):
    _apply_model(_resized_model(n))


def _restore(n: int, ops: list):
    """Undo / redo: put back (n, ops), atomically."""
    s = ss()
    resized = _resized_model(n) if n != int(s.n) else None
    if resized is not None:
        _apply_model(resized)
    s.n, s.ops = n, ops
    s.ops_version += 1
    push_widgets()


def undo():
    s = ss()
    if not s.get("history"):
        return
    n, ops = s.history[-1]
    cur = (int(s.n), [dict(d) for d in s.ops])
    _restore(n, ops)                                     # raises before touching the history if it cannot
    s.history.pop()
    s.future = (s.get("future", []) + [cur])[-HISTORY:]


def redo():
    s = ss()
    if not s.get("future"):
        return
    n, ops = s.future[-1]
    cur = (int(s.n), [dict(d) for d in s.ops])
    _restore(n, ops)
    s.future.pop()
    s.history = (s.get("history", []) + [cur])[-HISTORY:]


def _load_template_model(key: str, params: dict):
    s = ss()
    params = T.clamp_params(key, params)
    tpl = T.REGISTRY[key][1](**params)
    if "ops" in s:
        _remember()
    s.n = tpl.spec.n
    s.ops = _ops_dicts(tpl.spec)
    s.ops_version = s.get("ops_version", 0) + 1
    s.template = key
    s.template_ops_version = s.ops_version            # edits after this make the circuit "custom"
    s.template_obs_key = tpl.observable.key           # the analysis reads codes through their own logical observable
    s.template_relabel = bool(tpl.relabel)
    s.w_an_view = "auto"                              # a new experiment starts with the automatic analysis view
    s.template_name = tpl.spec.name
    s.tpl_params = dict(params)
    s.tpl_tex = tpl.tex
    s.tpl_notes = tpl.notes
    s.observable = tpl.observable
    s.symmetry = tpl.symmetry
    s.relabel = tpl.relabel
    s.inputs = ["0"] * tpl.spec.n
    s.basis = "Z" * tpl.spec.n
    s.var_spec = dict(version=s.ops_version, params=[dict(p) for p in tpl.variational]) if tpl.variational else None
    s.var_scan = None
    for p in T.REGISTRY[key][2]:                      # the template card shows what was loaded (Learn setups too)
        s["w_tp_%s_%s" % (key, p.name)] = params[p.name]
    for k, p in enumerate(tpl.variational):
        s["w_var_%d" % k] = float(p["value"])


# ------------------------------------------------------------------ live variational parameters
def var_params():
    """The template's live parameters while the circuit is unchanged since loading (else None)."""
    s = ss()
    vs = s.get("var_spec")
    if not vs or vs.get("version") != s.get("ops_version"):
        return None
    from qemx import variational as V
    try:
        return vs["params"] if V.valid(spec(), vs["params"]) else None
    except Exception:                                   # noqa: BLE001 - an invalid circuit has no live parameters
        return None


def is_custom() -> bool:
    """True once the circuit was edited after loading a template (or never came from one)."""
    s = ss()
    return s.get("template") not in T.REGISTRY or s.get("template_ops_version") != s.get("ops_version")


def var_values(params) -> list:
    s = ss()
    out = []
    for k, p in enumerate(params):
        try:
            v = float(s.get("w_var_%d" % k, p["value"]))
        except (TypeError, ValueError):
            v = float(p["value"])
        out.append(min(max(v, p["lo"]), p["hi"]) if np.isfinite(v) else float(p["value"]))
    return out


def cb_var_reset():
    params = var_params() or []
    for k, p in enumerate(params):
        ss()["w_var_%d" % k] = float(p["value"])


def cb_var_set(values):
    """Move the sliders (e.g. to the best point of a scan)."""
    params = var_params() or []
    for k, (p, v) in enumerate(zip(params, values)):
        ss()["w_var_%d" % k] = float(min(max(v, p["lo"]), p["hi"]))


def cb_var_apply():
    """Write the live angles into the circuit; they become the template's new base values."""
    from qemx import variational as V
    s = ss()
    params = var_params()
    if not params:
        return
    vals = var_values(params)
    new = V.apply(spec(), params, vals)
    _remember()
    s.ops = _ops_dicts(new)
    s.ops_version = s.get("ops_version", 0) + 1
    s.var_spec = dict(version=s.ops_version, params=[dict(p, value=float(v)) for p, v in zip(params, vals)])
    if s.get("template_ops_version") == s.ops_version - 1:
        s.template_ops_version = s.ops_version        # new angles, same template
    s.var_scan = None


def load_template(key: str, params: dict | None = None):
    """Callback: load a template with its parameters (or the current widget values)."""
    if params is None:
        params = {p.name: ss().get("w_tp_%s_%s" % (key, p.name), p.default) for p in T.REGISTRY[key][2]}
    ss().tpl_error = ""
    from qemx.resources import RunTimeout, deadline
    try:
        with deadline(ss().get("time_limit", 0)):
            _load_template_model(key, params)
    except RunTimeout:
        ss().tpl_error = ("Building this template passed the time limit set in the sidebar (optimizing variational "
                          "angles costs ~2^n per step). Raise the limit, use fewer qubits, or choose random angles.")
        return
    except MemoryError:
        ss().tpl_error = "Building this template needs more memory than is available (the state has 2^n amplitudes)."
        return
    except (ValueError, ArithmeticError) as e:     # a template that cannot be built keeps the old circuit
        ss().tpl_error = "Could not build this template: %s" % (str(e)[:300] or type(e).__name__)
        return
    except Exception as e:                                       # noqa: BLE001 - never crash the page from a callback
        ss().tpl_error = "Could not build this template (%s: %s)" % (type(e).__name__, str(e)[:200])
        return
    push_widgets()


def template_preset_values(key: str, name: str) -> dict:
    """Widget values {widget key: value} of a named preset of template ``key`` (the encoded-algorithm presets of
    qemx/qec.py: a code × benchmark pair with its parameters); every other parameter of the template at its default.
    Empty for an unknown preset."""
    hit = next((prm for nm, k, prm in getattr(T, "QEC_PRESETS", ()) if nm == name and k == key), None)
    if hit is None:
        return {}
    out = {}
    for p in T.REGISTRY[key][2]:
        v = hit.get(p.name, p.default)
        if p.kind == "int":
            v = int(v)
        elif p.kind == "float":
            v = float(v)
        out["w_tp_%s_%s" % (key, p.name)] = v
    return out


def cb_cited_preset():
    """Callback: fill the template controls from a cited preset (qemx.presets_cited). Nothing is built until Load."""
    from qemx import presets_cited as PC
    s = ss()
    key = s.get("w_cited_preset")
    if key not in PC.BY_KEY:
        return
    p = PC.BY_KEY[key]
    s.w_tpl_cat = T.category_of(p.template)
    s.w_tpl = p.template
    for name, v in PC.resolve(key)["params"].items():
        s["w_tp_%s_%s" % (p.template, name)] = v


def cb_template_preset(key: str):
    """Callback: set the template's parameter widgets to the chosen preset (the user then presses Load)."""
    for k, v in template_preset_values(key, ss().get("w_tpl_preset_%s" % key, "")).items():
        ss()[k] = v


# ------------------------------------------------------------------ derived
def compile_options() -> CompileOptions:
    from qemx import device
    s = ss()
    cm = dict(COMPILE_DEFAULT, **s.compile)
    init = parse_layout(cm.get("initial_layout", ""))
    dev = device.is_device(s.backend)
    layout = cm["layout"] if dev or cm["layout"] in ("sabre", "trivial") else "sabre"   # "dense": devices only
    return CompileOptions(mode=cm["mode"], basis_2q=cm["basis_2q"], coupling=cm["coupling"], layout=layout,
                          opt_level=int(cm["opt_level"]), seed=int(cm.get("seed", 11)),
                          device=s.backend if dev else None,
                          initial_layout=tuple(init) if init and layout_allowed() and not layout_problem(init)
                          else None)


def layout_problem(init=None) -> str:
    """Why a forced initial layout cannot be used for the current circuit and target ('' if it can)."""
    from qemx import device
    s = ss()
    if init is None:
        try:
            init = parse_layout(dict(COMPILE_DEFAULT, **s.compile).get("initial_layout", ""))
        except ValueError as e:
            return str(e)
    if not init or not layout_allowed():
        return ""
    from qemx.compile import layout_problem as _problem
    cm = dict(COMPILE_DEFAULT, **s.compile)
    return _problem(init, int(s.n), cm["coupling"], s.backend if device.is_device(s.backend) else None)


def layout_allowed() -> bool:
    """A forced initial layout applies to devices and to ISA compilation on a coupling map (not all-to-all)."""
    from qemx import device
    s = ss()
    cm = dict(COMPILE_DEFAULT, **s.compile)
    return device.is_device(s.backend) or (cm["mode"] == "isa" and cm["coupling"] != "all")


def cb_use_layout(text: str):
    """Topology tab: force the initial layout (physical qubit of q0, q1, …)."""
    s = ss()
    s.compile = {**COMPILE_DEFAULT, **s.compile, "initial_layout": text}
    s.w_cm_init = text
    s.layout_error = ""


def cb_try_coupling(kind: str):
    """Topology tab: compile to the ISA gate set on a lattice (line, ring, grid, heavy-hex)."""
    s = ss()
    s.compile = {**COMPILE_DEFAULT, **s.compile, "mode": "isa", "coupling": kind, "initial_layout": ""}
    s.w_cm_mode, s.w_cm_cp, s.w_cm_init = "isa", kind, ""


def custom_active_keys() -> list:
    """Keys of the saved custom recipes that are active contenders. A session without the key (older state) treats
    every saved custom recipe as active, as before 1.5."""
    s = ss()
    if s.get("custom_active") is None:
        return [Recipe.from_json(d).key() for d in (s.get("custom_recipes") or [])]
    return list(s.custom_active)


def recipes() -> list:
    s = ss()
    out, seen = [], set()
    active = set(custom_active_keys())
    for r in [preset(n, s.get("symmetry")) for n in s.recipe_names if n in RECIPE_PRESETS] + \
             [Recipe.from_json(d) for d in s.custom_recipes if Recipe.from_json(d).key() in active]:
        if r.key() not in seen:                    # the same recipe twice would share one result row
            seen.add(r.key())
            out.append(r)
    return out


def observable_choices() -> dict:
    """{key: label} for the observable selectors: the current observable first, then the catalogue. Names only —
    nothing of size 2^n is built on a rerun."""
    s = ss()
    cat = OBS.catalogue_labels(int(s.n))
    o = s.get("observable")
    if o is not None and o.n == int(s.n):
        cat = {o.key: o.label, **cat}
    return cat


def observable_for(key: str):
    """The observable behind a key of observable_choices() (built on demand; ValueError if it cannot be)."""
    s = ss()
    o = s.get("observable")
    if o is not None and o.n == int(s.n) and o.key == key:
        return o
    try:
        return OBS.catalogue_item(int(s.n), key)
    except MemoryError:
        raise ValueError("the observable needs more memory than is available for %d qubits; use a Hamiltonian "
                         "(Pauli sum) instead" % int(s.n)) from None


def observable_json(o) -> dict:
    """An observable as plain data: a Hamiltonian keeps its text, a diagonal observable its values."""
    from qemx.hamiltonian import PauliSum
    if isinstance(o, PauliSum):
        return dict(key=o.key[:40], label=o.label, tex=o.tex, pauli=o.text)
    return dict(key=o.key, label=o.label, tex=o.tex, values=list(o.values))


def cb_hamiltonian():
    """Callback: parse the Hamiltonian text (safe reader); keep the old observable on errors."""
    from qemx import hamiltonian as HM
    s = ss()
    txt = (s.get("w_obs_ham") or "").strip()
    s.ham_error = ""
    if not txt:
        return
    try:
        s.observable = HM.parse_pauli_sum(txt, int(s.n))
    except HM.HamiltonianError as e:
        s.ham_error = str(e)


def observable_digest(o) -> str:
    """Content hash of an observable, computed once per (immutable) observable object."""
    def mk():
        return hashlib.sha1(json.dumps(observable_json(o), sort_keys=True, default=str).encode()).hexdigest()
    try:
        return memo(o, "digest", mk)
    except (AttributeError, TypeError):                   # an object that cannot carry the memo
        return mk()


def signature(extra=None) -> str:
    """Everything a run depends on; results with another signature are stale. The circuit and the observable
    enter through content hashes cached per object, so a rerun of a deep circuit stays cheap."""
    s = ss()
    o = s.get("observable")
    d = dict(spec=spec_digest(spec()), inputs=list(s.inputs), basis=s.basis, compile=s.compile, relabel=s.relabel,
             noise=dataclasses.asdict(s.noise), backend=s.backend, obs=o.key if o is not None else "",
             obs_values=observable_digest(o) if o is not None else "",
             recipes=[r.to_json() for r in recipes()], shots=int(s.shots), extra=extra)
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------ widget -> model callbacks
def cb_backend():
    s = ss()
    from qemx import device
    if s.w_backend != s.backend and dict(COMPILE_DEFAULT, **s.compile).get("initial_layout"):
        s.compile = {**COMPILE_DEFAULT, **s.compile, "initial_layout": ""}    # physical qubits of another target
        s.w_cm_init = ""
    if not device.is_device(s.w_backend) and dict(COMPILE_DEFAULT, **s.compile)["layout"] not in ("sabre", "trivial"):
        s.compile = {**COMPILE_DEFAULT, **s.compile, "layout": "sabre"}      # "dense" exists for devices only
        s.w_cm_lay = "sabre"
    s.backend = s.w_backend


def cb_shots(src: str):
    s = ss()
    if src == "sl":
        v = int(s.w_shots_sl)
        s.w_shots_num = v
    else:
        v = max(int(s.w_shots_num or 1), 1)
        s.w_shots_sl = nearest_shots(v)
    s.shots = v


def cb_tshot():
    ss().t_shot_us = float(ss().w_tshot)


def cb_tlimit():
    ss().time_limit = max(float(ss().w_tlimit or 0.0), 0.0)


NOISE_SLIDER_MAX = {"p1": 0.02, "p2": 0.10, "amp_damp": 0.10, "phase_damp": 0.10, "kick": 0.50, "over_rot": 0.20,
                    "zz_phase": 0.50, "z_phase": 0.30, "field_sigma": 0.50, "idle_dephase": 0.10, "idle_damp": 0.10,
                    "ro01": 0.20, "ro10": 0.20}


def cb_noise_field(f: str, src: str):
    s = ss()
    hi = NOISE_SLIDER_MAX.get(f, 1.0)
    if src == "sl":
        v = float(s["w_nz_%s_sl" % f])
        s["w_nz_%s_num" % f] = v
    else:
        v = float(s["w_nz_%s_num" % f] or 0.0)
        s["w_nz_%s_sl" % f] = min(max(v, 0.0), hi)
    s.noise = dataclasses.replace(s.noise, **{f: v})
    match = [k for k, spec_ in PRESETS.items() if spec_ == s.noise]
    s.noise_preset = match[0] if match else "Custom"
    s.w_nz_preset = s.noise_preset


def cb_noise_preset():
    s = ss()
    name = s.w_nz_preset
    if name in PRESETS:
        s.noise = PRESETS[name]
        s.noise_preset = name
        for f in NOISE_FIELDS:
            v = float(getattr(s.noise, f))
            s["w_nz_%s_num" % f] = v
            s["w_nz_%s_sl" % f] = min(max(v, 0.0), NOISE_SLIDER_MAX.get(f, 1.0))
    else:
        s.noise_preset = "Custom"


def cb_input(q: int):
    s = ss()
    inputs = list(s.inputs)
    inputs[q] = s["w_in_%d" % q]
    s.inputs = inputs


def cb_register_io():
    """Set every qubit's input and/or measurement basis from one register-wide value (qemx.sweep syntax)."""
    from qemx import sweep as SW
    s = ss()
    n = int(s.n)
    s.io_error = ""
    inp, bs = str(s.get("w_io_all_in") or "").strip(), str(s.get("w_io_all_bs") or "").strip()
    if not inp and not bs:
        s.io_error = "Type an input (e.g. plus, ramp:1) and/or a basis (e.g. X) first."
        return
    try:
        new_in = SW.input_tokens(inp, n) if inp else list(s.inputs)
        new_bs = SW.basis_string(bs, n) if bs else s.basis
    except SW.SweepError as e:
        s.io_error = str(e)
        return
    s.inputs, s.basis = new_in, new_bs
    for q in range(n):
        s["w_in_%d" % q] = new_in[q]
        s["w_bs_%d" % q] = new_bs[q]


def cb_basis(q: int):
    s = ss()
    b = list(s.basis)
    b[q] = s["w_bs_%d" % q]
    s.basis = "".join(b)


def cb_observable():
    s = ss()
    k = s.w_obs
    s.obs_error = ""
    if k in observable_choices():
        try:
            s.observable = observable_for(k)
        except (ValueError, MemoryError) as e:           # the old observable stays; the page shows why
            s.obs_error = str(e)[:300] or type(e).__name__
    elif k == "hamiltonian":
        if not s.get("w_obs_ham"):                    # start from the first example, applied at once
            from qemx import hamiltonian as HM
            s.w_obs_ham = next(iter(HM.examples(int(s.n)).values()))
        cb_hamiltonian()


def cb_obs_bits():
    s = ss()
    bs = (s.get("w_obs_bits") or "").strip()
    n = int(s.n)
    if len(bs) == n and set(bs) <= {"0", "1"}:
        if not OBS.dense_ok(n):
            s.obs_bits_error = ("A projector on %d qubits needs 2^%d stored values, more than this machine can hold; "
                                "the observable was not changed." % (n, n))
            return
        try:
            s.observable = OBS.projector(int(bs, 2), n)
            s.obs_bits_error = ""
        except (MemoryError, ValueError) as e:
            s.obs_bits_error = "The projector could not be built (%s); the observable was not changed." % (
                str(e)[:200] or type(e).__name__)
    elif bs:
        s.obs_bits_error = "Type %d characters, each 0 or 1 (q%d first). The observable was not changed." % (n, n - 1)


def cb_relabel():
    ss().relabel = bool(ss().w_relabel)


def cb_compile(field_: str, wkey: str):
    s = ss()
    cm = dict(COMPILE_DEFAULT, **s.compile)
    v = s[wkey]
    if field_ == "mode" and v is None:
        v = cm["mode"]
        s[wkey] = v
    if field_ == "initial_layout":
        try:
            parse_layout(v)
            s.layout_error = ""
        except ValueError as e:
            s.layout_error = str(e)
            return
    if field_ in ("mode", "coupling") and cm.get(field_) != v and cm.get("initial_layout"):
        cm["initial_layout"] = ""                  # a forced layout belongs to the old coupling map
        s.w_cm_init = ""
    cm[field_] = v
    s.compile = cm


def cb_recipes():
    s = ss()
    s.recipe_names = [n for n in (s.w_rc_pick or []) if n in RECIPE_PRESETS]


def cb_tab(page: str, key: str):
    ss().page_tab = dict(ss().get("page_tab", {}), **{page: ss()[key]})


# ------------------------------------------------------------------ JSON state
def to_json() -> dict:
    """The app state as plain data. It never contains credentials: tokens are not part of the model."""
    s = ss()
    sp = spec()
    return dict(app="QEM Explorer", version=VERSION, circuit=memo(sp, "json", sp.to_json),   # read-only, cached
                inputs=list(s.inputs), basis=s.basis,
                relabel=bool(s.relabel), compile=dict(s.compile), noise=dataclasses.asdict(s.noise),
                observable=observable_json(s.observable),
                symmetry=list(s.symmetry[0]) + [s.symmetry[1]] if s.get("symmetry") else None,
                symmetry_more=(s.symmetry[2] if s.get("symmetry") and len(s.symmetry) > 2 else ""),
                recipes=[r.to_json() for r in recipes()], backend=s.backend, shots=int(s.shots),
                t_shot_us=float(s.t_shot_us))


from qemx.state_schema import (  # noqa: E402,F401 - the shared schema (moved in 1.5); re-exported for pages/tests
    JS_SAFE_INT, MAX_STATE_BYTES, check_param, param_visible, validate_params, validate_state)


def from_json(d: dict):
    """Callback: load a validated state."""
    v = validate_state(d)
    s = ss()
    _remember()
    s.n = v["n"]
    s.ops = v["ops"]
    s.ops_version = s.get("ops_version", 0) + 1
    s.template_name = v["name"]
    s.tpl_tex, s.tpl_notes, s.var_spec = "", "", None      # the loaded circuit is not the template shown before
    s.inputs, s.basis, s.relabel = v["inputs"], v["basis"], v["relabel"]
    s.compile = v["compile"]
    s.noise = NoiseSpec(**v["noise"])
    match = [k for k, spec_ in PRESETS.items() if spec_ == s.noise]
    s.noise_preset = match[0] if match else "Custom"
    s.observable = v["observable"] if not isinstance(v["observable"], tuple) else OBS.Observable(*v["observable"])
    s.symmetry = v["symmetry"]
    s.recipe_names = []
    s.custom_recipes = v["recipes"]
    s.custom_active = [Recipe.from_json(d).key() for d in v["recipes"]]     # a loaded state's recipes are all active
    s.backend = v["backend"]
    s.shots = v["shots"]
    s.t_shot_us = v["t_shot_us"]
    s.results = None
    s.timeline = None
    push_widgets()


# ------------------------------------------------------------------ Learn: one-click setups
SETUP_PAGE_TABS = {"playground": "w_pg_tabs", "cvp": "w_cvp_tabs", "mm": "w_mm_tabs"}     # page key → tab widget
SETUP_WIDGET_PREFIXES = ("w_hh_", "w_cvp_", "w_mm_", "w_hw_", "w_b_")   # analysis-page controls a setup may preset


SETUP_MODES = ("replace", "merge")


def _validate_setup(setup: dict) -> list:
    """Check every field of a setup before anything changes; returns the custom contenders as Recipe objects.
    ValueError with the field's name on anything invalid (a setup is static data of qemx.learn, never executed)."""
    from qemx import hamiltonian as HM
    from qemx.circuit import parse
    if "template" in setup:
        key, params = setup["template"]
        if key not in T.REGISTRY:
            raise ValueError("template: unknown template %r" % (key,))
        n = T.build(key, T.clamp_params(key, dict(T.defaults(key), **params))).spec.n
    elif "circuit" in setup:
        name, n, text = setup["circuit"]
        parse(text, int(n), name)
        n = int(n)
    else:
        n = int(ss().get("n", 1))
    ob = setup.get("observable")
    if ob:
        if ob[0] == "z":
            if not 0 <= int(ob[1]) < n:
                raise ValueError("observable: qubit %s is outside the circuit's %d qubits" % (ob[1], n))
        elif ob[0] == "hamiltonian":
            HM.parse_pauli_sum(ob[1], n)
        else:
            raise ValueError("observable: unknown kind %r" % (ob[0],))
    if "noise" in setup and setup["noise"] not in PRESETS:
        raise ValueError("noise: unknown preset %r" % (setup["noise"],))
    bad = [r for r in setup.get("recipes", []) if r not in RECIPE_PRESETS]
    if bad:
        raise ValueError("recipes: unknown preset%s %s" % ("" if len(bad) == 1 else "s", ", ".join(map(repr, bad))))
    custom = []
    for d in setup.get("custom") or []:
        validate_params(d["estimator"], d.get("params", {}))
        custom.append(Recipe.from_json(d))
    return custom


def apply_setup(setup: dict, mode: str = "replace"):
    """Callback for the Learn page: put the Playground (or an analysis page) into a guide's starting state, then
    ask the Learn page to switch pages (``goto``).

    Keys: ``template`` or ``circuit``, ``observable``, ``noise``, ``recipes`` (preset names), ``custom`` (recipe
    dicts, validated like the recipe builder's), ``backend``, ``compile``, ``batch``, ``tab`` (the Playground's),
    ``tabs`` ({page key: tab} for other pages), ``widgets`` (initial values of whitelisted analysis-page controls)
    and ``page``. Setups are static data of ``qemx.learn``; nothing in them is executed.

    Contenders. ``mode="replace"`` (default) makes the guide's contenders the *only* active ones, so a guide always
    starts from the same experiment whatever was explored before; the user's saved custom recipes stay saved (inactive,
    one click to reactivate). ``mode="merge"`` adds the guide's contenders to the active ones.

    Atomic. Every field is validated first, and the change is applied as one transaction: if anything fails, the
    session is restored exactly and the error is shown on the Learn page."""
    from qemx import hamiltonian as HM
    from qemx.circuit import parse
    from qemx.resources import deadline
    s = ss()
    s.learn_error = ""
    if mode not in SETUP_MODES:
        s.learn_error = "Could not set this up: unknown mode %r" % (mode,)
        return
    try:
        custom = _validate_setup(setup)
    except Exception as e:                                   # noqa: BLE001 - nothing has changed yet
        s.learn_error = "Could not set this up: %s: %s" % (type(e).__name__, str(e)[:200])
        return
    # transaction: a shallow snapshot (lists and dicts copied one level, since some helpers append in place)
    snap = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in s.items()}
    try:
        if "template" in setup:
            key, params = setup["template"]
            with deadline(s.get("time_limit", 0)):
                _load_template_model(key, T.clamp_params(key, dict(T.defaults(key), **params)))
        elif "circuit" in setup:
            name, n, text = setup["circuit"]
            set_circuit(parse(text, int(n), name))
            s.template_name = name
            s.inputs, s.basis = ["0"] * int(n), "Z" * int(n)
            s.symmetry, s.relabel = None, False
            s.var_spec, s.tpl_tex, s.tpl_notes = None, "", ""
        ob = setup.get("observable")
        if ob and ob[0] == "z":
            s.observable = OBS.z(int(ob[1]), int(s.n))
        elif ob and ob[0] == "hamiltonian":
            s.observable = HM.parse_pauli_sum(ob[1], int(s.n))
            s.w_obs_ham = ob[1]
        if "noise" in setup:
            s.noise, s.noise_preset = PRESETS[setup["noise"]], setup["noise"]
        if "recipes" in setup or custom:
            names = [r for r in setup.get("recipes", [])]
            library = list(s.get("custom_recipes") or [])
            have = {Recipe.from_json(d).key() for d in library}
            for r in custom:                                 # the guide's custom recipes join the saved library
                if r.key() not in have:
                    have.add(r.key())
                    library.append(r.to_json())
            keys = [r.key() for r in custom]
            if mode == "replace":
                s.recipe_names = names
                s.custom_active = keys
            else:
                s.recipe_names = list(dict.fromkeys(list(s.get("recipe_names") or []) + names))
                s.custom_active = list(dict.fromkeys(custom_active_keys() + keys))
            s.custom_recipes = library
        s.backend = setup.get("backend", "numpy")
        s.compile = {**COMPILE_DEFAULT, **setup.get("compile", {"mode": "abstract"}), "initial_layout": ""}
        b = setup.get("batch")
        if b:
            s.w_b_algos, s.w_b_methods = list(b["algos"]), list(b["methods"])
            s.w_b_noises, s.w_b_scales, s.w_b_sims = list(b["noises"]), list(b["scales"]), list(b["sims"])
        tab = setup.get("tab")
        if tab:
            s.page_tab = dict(s.get("page_tab", {}), playground=tab)
            s.w_pg_tabs = tab
        for page, tab_ in setup.get("tabs", {}).items():       # the opening tab of another page
            if page in SETUP_PAGE_TABS:
                s.page_tab = dict(s.get("page_tab", {}), **{page: tab_})
                s[SETUP_PAGE_TABS[page]] = tab_
        for k, v in setup.get("widgets", {}).items():          # controls of the analysis pages (a fixed whitelist)
            if k.startswith(SETUP_WIDGET_PREFIXES):
                s[k] = list(v) if isinstance(v, (list, tuple)) else v
        push_widgets()
    except Exception as e:                                   # noqa: BLE001 - roll back; a setup never crashes the page
        for k in [k for k in list(s.keys()) if k not in snap]:
            try:
                del s[k]
            except Exception:                                # noqa: BLE001
                pass
        for k, v in snap.items():
            try:
                s[k] = v
            except Exception:                                # noqa: BLE001 - a widget key that cannot be set now
                pass
        s.learn_error = "Could not set this up (nothing was changed): %s: %s" % (type(e).__name__, str(e)[:200])
        return
    s.goto = setup.get("page", "pages/1_Playground.py")
