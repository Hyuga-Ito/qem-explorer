"""The validated experiment-state schema shared by the app (ui/state.py), the CLI and YAML experiments
(qemx/experiment.py): field-by-field validation of a state document, method-parameter checks and the compile
defaults. Pure functions: nothing here reads the Streamlit session, and nothing from a document is executed.

Moved from ui/state.py in 1.5 so that the CLI validates exactly as the app does (ui/state re-exports every name).
"""
from __future__ import annotations

import dataclasses
import math
import re

from .circuit import GATES, NOISE_OPS
from .compile import INPUT_STATES, valid_input
from .methods import REGISTRY, Recipe
from .noise import NoiseSpec

OP_TAGS = ("", "prep", "encode", "basis", "fold", "fold-inv", "pec", "twirl", "dd", "check", "train")

COUPLINGS = {"all": "all-to-all", "line": "line", "ring": "ring", "grid": "2D grid", "heavy-hex": "heavy-hex (IBM)"}
COMPILE_DEFAULT = dict(mode="abstract", basis_2q="cx", coupling="line", layout="sabre", opt_level=1, seed=11,
                       initial_layout="")


def parse_layout(text) -> list:
    if not text:
        return []
    out = []
    if len(str(text)) > 2000:
        raise ValueError("the layout is too long")
    for t in re.split(r"[,\s]+", str(text).strip()):
        if t:
            if not (t.isascii() and t.isdecimal()) or len(t) > 6:
                raise ValueError("physical qubits must be non-negative integers, e.g. 3, 4, 5")
            out.append(int(t))
    return out


MAX_STATE_BYTES = 20 * 2 ** 20
_PROBABILITY_FIELDS = {"p1", "p2", "amp_damp", "phase_damp", "idle_dephase", "idle_damp", "ro01", "ro10"}
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9\s⟨⟩()|,.+\-=·…'^/Σ⊗≈₀₁₂₃₄₅₆₇₈₉ᵢ₊@]", re.UNICODE)
_TEX_OK = {"langle", "rangle", "otimes", "in", "sum", "cdot", "mathrm", "text", "rm", "left", "right", "frac", "tfrac",
           "sqrt", "pi", "dots", "ldots", "quad", "Sigma", "sigma", "mid", "!", ",", ";", "{", "}", "_", "|"}


def _finite(v, name, lo=None, hi=None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("%s must be a number" % name)
    try:
        v = float(v)
    except OverflowError:
        raise ValueError("%s is out of range" % name) from None
    if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError("%s is out of range" % name)
    return v


def _int(v, name, lo=None, hi=None) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("%s must be an integer" % name)
    if isinstance(v, float) and (not math.isfinite(v) or not v.is_integer()):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError("%s is out of range" % name)
    return v


def _safe_tex(tex: str, fallback: str) -> str:
    if not isinstance(tex, str) or len(tex) > 300:
        return fallback
    for cmd in re.findall(r"\\([A-Za-z]+|.)", tex):
        if cmd not in _TEX_OK:
            return fallback
    return tex


JS_SAFE_INT = (1 << 53) - 1                 # Streamlit's number widgets reject larger integer bounds
MAX_LIST = 64                               # list parameters (noise scales, qubits)
MAX_TEXT_PARAM = 200
_TEXT_PARAM_OK = re.compile(r"^[A-Za-z0-9 ,;:._+\-*/()]*$")


def param_visible(p, chosen: dict) -> bool:
    """A parameter is shown (and used) only if every (other parameter, allowed values) condition of ``p.when``
    holds for the values chosen so far (controlling parameters come first in a method's list)."""
    for cond in getattr(p, "when", ()) or ():
        other, allowed = cond
        if chosen.get(other, None) not in tuple(allowed):
            return False
    return True


def _same(v, d) -> bool:
    if isinstance(v, (list, tuple)) and isinstance(d, (list, tuple)):
        return list(v) == list(d)
    return v == d and type(v) is type(d) or (v is None and d is None)


def check_param(p, v, where: str = ""):
    """Validate one method parameter value against its schema (kind, options, lo/hi); ValueError if invalid."""
    name = (where + "." if where else "") + p.name
    if p.kind == "choice":
        if not any(_same(v, o) or v == o for o in p.options):
            raise ValueError("invalid value for %s" % name)
    elif p.kind == "float":
        if v is None and p.default is None:
            return
        _finite(v, name, p.lo, p.hi)
    elif p.kind == "int":
        if v is None and p.default is None:
            return
        _int(v, name, p.lo, p.hi if p.hi is not None else 10 ** 9)
    elif p.kind in ("floats", "ints"):
        if not isinstance(v, list) or len(v) > MAX_LIST:
            raise ValueError("invalid list for %s" % name)
        for x in v:
            if p.kind == "floats":
                _finite(x, name, p.lo, p.hi)
            else:
                _int(x, name, p.lo if p.lo is not None else 0, p.hi)
    elif p.kind == "bool":
        if not isinstance(v, bool):
            raise ValueError("%s must be true or false" % name)
    elif p.kind == "text":
        if not isinstance(v, str) or len(v) > MAX_TEXT_PARAM:
            raise ValueError("invalid text for %s" % name)
        if p.name in ("more", "symmetry_more"):              # further parity checks of symmetry verification
            from qemx.methods.symmetry import parse_checks
            parse_checks(v)
        elif not _TEXT_PARAM_OK.match(v):
            raise ValueError("%s contains characters that are not allowed" % name)
    else:
        raise ValueError("unknown kind of parameter %s" % name)


def validate_params(estimator: str, params) -> dict:
    """Check a recipe's parameters against the method's schema. A parameter hidden by its ``when`` conditions is
    accepted at its default (as the UI leaves it); every other value must satisfy kind, options and lo/hi."""
    if estimator not in REGISTRY:
        raise ValueError("unknown estimator %r" % str(estimator)[:40])
    schema = list(REGISTRY[estimator].params)
    if not isinstance(params, dict) or not set(params) <= {p.name for p in schema}:
        raise ValueError("unknown parameter in recipe %s" % estimator)
    from qemx.methods.base import migrate_params
    params = migrate_params(estimator, params)[0]           # legacy names (e.g. FR's features="nil") → current ids
    chosen = {}
    for p in schema:
        v = params.get(p.name, p.default)
        if not param_visible(p, chosen) and _same(v, p.default):
            chosen[p.name] = p.default
            continue
        if p.name in params:
            check_param(p, v, estimator)
        chosen[p.name] = v
    return chosen


def validate_state(d) -> dict:
    """Check an uploaded state field by field; raise ValueError with a readable message on anything unexpected.
    Nothing from the file is executed or rendered as HTML."""
    try:
        return _validate_state(d)
    except ValueError:
        raise
    except Exception as e:                       # any other shape problem is still just an invalid file
        raise ValueError("the file does not describe a valid state (%s)" % type(e).__name__) from None


def _dict(v, name) -> dict:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError("%s must be an object" % name)
    return v


def _validate_state(d) -> dict:
    if not isinstance(d, dict):
        raise ValueError("the file is not a QEM Explorer state")
    c = d.get("circuit")
    if not isinstance(c, dict):
        raise ValueError("missing circuit")
    ob = _dict(d.get("observable"), "observable")
    pauli = ob.get("pauli")
    if pauli is not None:                                  # a Hamiltonian: text, parsed by the safe Pauli-sum reader
        if not isinstance(pauli, str) or len(pauli) > 4000:
            raise ValueError("observable.pauli must be a short text")
        n = _int(c.get("n"), "circuit.n", 1, 64)
        vals_raw = []
    else:
        vals_raw = ob.get("values")
        if not isinstance(vals_raw, list) or len(vals_raw) < 2 or len(vals_raw) & (len(vals_raw) - 1):
            raise ValueError("observable must list 2^n values")
        n_obs = len(vals_raw).bit_length() - 1             # the file size bounds n before anything is allocated
        n = _int(c.get("n"), "circuit.n", 1, n_obs)
        if n != n_obs:
            raise ValueError("observable has %d values but the circuit has %d qubits" % (len(vals_raw), n))
    if not isinstance(c.get("ops", []), list):
        raise ValueError("circuit.ops must be a list")
    ops = []
    for i, o in enumerate(c.get("ops", [])):
        if not isinstance(o, dict):
            raise ValueError("operation %d is malformed" % i)
        name, kind = o.get("name"), o.get("kind", "gate")
        qs = [_int(q, "qubit of operation %d" % i, 0) for q in o.get("qubits", [])]
        if any(q >= n for q in qs) or len(set(qs)) != len(qs):
            raise ValueError("operation %d uses an invalid qubit" % i)
        ps = [_finite(p, "parameter of operation %d" % i) for p in o.get("params", [])]
        if kind == "gate":
            if name not in GATES or GATES[name][0] != len(qs) or GATES[name][1] != len(ps):
                raise ValueError("operation %d: unknown gate or wrong arity" % i)
        elif kind == "noise":
            if name not in NOISE_OPS or len(ps) != 1 or not qs or (NOISE_OPS[name][0] and
                                                                    len(qs) != NOISE_OPS[name][0]):
                raise ValueError("operation %d: unknown noise channel or wrong number of qubits" % i)
            if name not in ("overrot", "zphase") and not 0 <= ps[0] <= 1:
                raise ValueError("operation %d: probability out of range" % i)
        elif kind == "barrier":
            name, ps = "barrier", []
        else:
            raise ValueError("operation %d: unknown kind" % i)
        tag = o.get("tag", "")
        tag = tag if tag in OP_TAGS else ""
        ops.append(dict(name=name, qubits=qs, params=ps, kind=kind, tag=tag))
    inputs = d.get("inputs", ["0"] * n)
    if not (isinstance(inputs, list) and len(inputs) == n and all(valid_input(x) for x in inputs)):
        raise ValueError("inputs must list one of %s (or ph:<turns>, bloch:<θ>,<φ>) per qubit" % ", ".join(INPUT_STATES))
    basis = d.get("basis", "Z" * n)
    if not (isinstance(basis, str) and len(basis) == n and set(basis) <= set("XYZ")):
        raise ValueError("basis must be %d letters from X, Y, Z" % n)
    cm = dict(COMPILE_DEFAULT)
    for k, v in _dict(d.get("compile"), "compile").items():
        if k == "mode" and v in ("abstract", "isa"):
            cm[k] = v
        elif k == "basis_2q" and v in ("cx", "cz"):
            cm[k] = v
        elif k == "coupling" and v in COUPLINGS:
            cm[k] = v
        elif k == "layout" and v in ("sabre", "trivial", "dense"):
            cm[k] = v
        elif k == "opt_level":
            cm[k] = _int(v, "compile.opt_level", 0, 3)
        elif k == "seed":
            cm[k] = _int(v, "compile.seed", 0, 2 ** 32 - 1)
        elif k == "initial_layout":
            cm[k] = ",".join(map(str, parse_layout(v)))
    names = {f.name for f in dataclasses.fields(NoiseSpec)}
    nz = {}
    for k, v in _dict(d.get("noise"), "noise").items():
        if k not in names:
            continue
        if k == "field_samples":
            nz[k] = _int(v, "noise." + k, 1, 4096)
        elif k in _PROBABILITY_FIELDS:
            nz[k] = _finite(v, "noise." + k, 0.0, 1.0)
        else:
            nz[k] = _finite(v, "noise." + k, 0.0, 100.0)
    vals = tuple(_finite(v, "observable value") for v in vals_raw)
    ham = None
    if pauli is not None:
        from qemx import hamiltonian as HM
        try:
            ham = HM.parse_pauli_sum(pauli, n)
        except HM.HamiltonianError as e:
            raise ValueError("observable.pauli: %s" % e) from None
    key = str(ob.get("key", "custom"))[:40]
    key = re.sub(r"[^\w@.-]", "", key) or "custom"
    label = _SAFE_LABEL.sub("", str(ob.get("label", "⟨O⟩")))[:60] or "⟨O⟩"
    tex = _safe_tex(ob.get("tex", ""), r"\langle O\rangle")
    sym = d.get("symmetry")
    if sym is not None:
        if not (isinstance(sym, list) and len(sym) >= 2):
            raise ValueError("symmetry must be [qubits..., eigenvalue]")
        qs = [_int(q, "symmetry qubit", 0) for q in sym[:-1]]
        ev = _int(sym[-1], "symmetry eigenvalue")
        if any(q >= n for q in qs) or ev not in (1, -1):
            raise ValueError("symmetry out of range")
        more = d.get("symmetry_more") or ""
        if more:
            from qemx.methods.symmetry import parse_checks
            groups = parse_checks(more)
            if any(q >= n for g in groups for q in g):
                raise ValueError("symmetry out of range")
            more = "; ".join(" ".join(map(str, g)) for g in groups)
        sym = (tuple(qs), ev, more) if more else (tuple(qs), ev)
    recs = []
    recs_raw = d.get("recipes", []) or []
    if not isinstance(recs_raw, list):
        raise ValueError("recipes must be a list")
    for r in recs_raw:
        if not isinstance(r, dict) or r.get("estimator") not in REGISTRY:
            raise ValueError("unknown estimator in recipes")
        validate_params(r["estimator"], r.get("params", {}) or {})
        if r.get("twirl", "none") not in ("none", "2q", "all") or r.get("dd", "none") not in ("none", "X2", "XY4") \
                or r.get("readout", "none") not in ("none", "tensored", "m3", "trex"):
            raise ValueError("invalid transform in recipe")
        recs.append(Recipe.from_json(r).to_json())
    from qemx.backends.base import BACKENDS
    from qemx import device
    backend = d.get("backend", "numpy")
    if backend not in BACKENDS and not device.is_device(backend):
        backend = "numpy"
    return dict(n=n, ops=ops, name=_SAFE_LABEL.sub("", str(c.get("name", "circuit")))[:60] or "circuit", inputs=inputs, basis=basis, compile=cm,
                noise=nz, observable=ham if ham is not None else (key, label, tex, vals), symmetry=sym, recipes=recs,
                backend=backend,
                shots=_int(d.get("shots", 4096), "shots", 1, 10 ** 12),
                t_shot_us=_finite(d.get("t_shot_us", 250.0), "t_shot_us", 0.0, 1e9), relabel=bool(d.get("relabel", False)))
