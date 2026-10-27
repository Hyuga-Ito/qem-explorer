r"""One validated experiment model for the app, the CLI and YAML files (1.5).

An *experiment document* (YAML or JSON) resolves to the app's state document (``qemx.state_schema``, the same
validation as "Load a state" in the Playground) plus run settings, so a configuration means the same thing in every
entry point. Two equivalent forms are accepted:

* ``state:`` — exactly the app's exported state (Playground → Export → Download state, or ``from-state`` below);
* a short form — ``circuit`` (a template with parameters, or gate text), optional ``inputs``/``basis``/``relabel``,
  ``observable``, ``noise``, ``compile``, ``backend``, ``methods``, ``budget``.

Every document has ``schema: qemx.experiment/1`` and may have ``run`` (seed, repeats, budget preflight). Validation
errors name the field (``methods[2].params: …``). The configuration hash is the SHA-256 of the canonical resolved
state plus run settings, so two documents that resolve to the same experiment have the same hash. Method ids are
stable: a short hash of the method's canonical recipe (``ZNE-R@1a2b3c4d``).

YAML is read with ``yaml.safe_load`` (no tags, nothing executed). Credentials never belong in a document: a field
whose name looks like a secret (token, password, api key) is rejected.

Stages (validate → plan → run → collect → render). ``validate`` and ``plan`` never simulate; ``run`` simulates on the
app's simulators only (real hardware has its own approved workflow, ``qemx.hardware``)::

    python -m qemx.experiment validate examples/experiments/qft_known_answer.yaml
    python -m qemx.experiment plan     examples/experiments/qft_known_answer.yaml
    python -m qemx.experiment run      examples/experiments/qft_known_answer.yaml --out results/qft_pilot
    python -m qemx.experiment from-state qem_explorer_state.json --out my_experiment.yaml
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import sys

SCHEMA = "qemx.experiment/1"
RESULT_SCHEMA = "qemx.experiment.result/1"
TOP_KEYS = {"schema", "name", "description", "state", "circuit", "inputs", "basis", "relabel", "observable", "noise",
            "compile", "backend", "methods", "budget", "run", "symmetry"}
RUN_KEYS = {"seed", "repeats", "check_budget", "table_seed"}
SECRET = re.compile(r"token|password|passwd|secret|api[_-]?key|credential", re.I)
MAX_DOC_BYTES = 20 * 2 ** 20


class ExperimentError(ValueError):
    """A document that does not describe a valid experiment; ``path`` names the field."""

    def __init__(self, path: str, msg: str):
        self.path = path
        super().__init__("%s: %s" % (path, msg) if path else msg)


# ------------------------------------------------------------------ loading
def load(path: str) -> dict:
    """Read a YAML or JSON experiment document safely."""
    if os.path.getsize(path) > MAX_DOC_BYTES:
        raise ExperimentError("", "the file is larger than %d MB" % (MAX_DOC_BYTES // 2 ** 20))
    from .textio import TextEncodingError, read_config_text
    try:
        text = read_config_text(path)                 # UTF-8; a byte-order mark is accepted (textio)
    except TextEncodingError as e:
        raise ExperimentError("", str(e)) from None
    return loads(text, json_hint=path.lower().endswith(".json"))


def loads(text: str, json_hint: bool = False) -> dict:
    if json_hint:
        doc = json.loads(text)
    else:
        import yaml
        doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ExperimentError("", "an experiment document is a mapping")
    return doc


def _scan_secrets(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = "%s.%s" % (path, k) if path else str(k)
            if SECRET.search(str(k)):
                raise ExperimentError(p, "credentials never belong in an experiment document (use the environment "
                                         "variable or a saved account; see SECURITY.md)")
            _scan_secrets(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _scan_secrets(v, "%s[%d]" % (path, i))


# ------------------------------------------------------------------ short form → state document
def _circuit_state(c, path="circuit") -> tuple:
    """(circuit dict, observable dict, relabel, symmetry, input tokens) from the short ``circuit`` field. A template's
    input preparation becomes the input tokens and the circuit holds the rest (``io_semantics.split_preparation``),
    as in the Playground, so the input is prepared once."""
    from . import templates as T
    from .circuit import parse
    if not isinstance(c, dict):
        raise ExperimentError(path, "must be a mapping with 'template' (and 'params') or 'n' and 'text'")
    if "template" in c:
        key = c["template"]
        if key not in T.REGISTRY:
            raise ExperimentError(path + ".template", "unknown template %r (one of %s)" % (
                key, ", ".join(sorted(T.REGISTRY))))
        params = c.get("params") or {}
        if not isinstance(params, dict):
            raise ExperimentError(path + ".params", "must be a mapping")
        names = {p.name for p in T.REGISTRY[key][2]}
        bad = set(params) - names
        if bad:
            raise ExperimentError(path + ".params", "unknown parameter(s) %s of template %s (allowed: %s)" % (
                sorted(bad), key, sorted(names)))
        clamped = T.clamp_params(key, params)
        changed = {k: (params[k], clamped[k]) for k in params if params[k] != clamped[k]}
        if changed:                                     # never silently change a configured value
            raise ExperimentError(path + ".params", "values outside the template's domain: %s" % ", ".join(
                "%s=%r (nearest valid %r)" % (k, a, b) for k, (a, b) in changed.items()))
        try:
            from .io_semantics import template_inputs
            tokens, body, tpl = template_inputs(key, clamped)
        except ValueError as e:
            raise ExperimentError(path + ".params", str(e)) from None
        from .observables import Observable
        obs = tpl.observable
        ob = (dict(key=obs.key, label=obs.label, tex=obs.tex, values=list(obs.values)) if isinstance(obs, Observable)
              else dict(key=obs.key[:40], label=obs.label, tex=obs.tex, pauli=obs.text))
        sym = None
        if tpl.symmetry:
            sym = list(tpl.symmetry[0]) + [tpl.symmetry[1]]
        circ = body.to_json()
        circ["template"] = dict(key=key, params=clamped)
        return circ, ob, bool(tpl.relabel), sym, tokens
    if "text" in c:
        try:
            n = int(c["n"])
            spec = parse(str(c["text"]), n, str(c.get("name", "circuit"))[:60])
        except KeyError:
            raise ExperimentError(path + ".n", "the number of qubits is required with 'text'") from None
        except Exception as e:                          # noqa: BLE001 - parser errors are user errors
            raise ExperimentError(path + ".text", str(e)[:300]) from None
        from . import observables as OBS
        o = OBS.z(0, n)
        return spec.to_json(), dict(key=o.key, label=o.label, tex=o.tex, values=list(o.values)), False, None, \
            ["0"] * n
    raise ExperimentError(path, "give 'template' (with 'params') or 'n' and 'text'")


def _inputs_state(doc: dict, tokens: list, n: int) -> list:
    """The per-qubit inputs of a short-form document: the template's own input (its input parameters), or ``inputs``.
    A template that loads its own input (the QFT's s and prep, BV's ancilla |−⟩, …) and a different ``inputs`` would
    prepare twice or leave unclear which one is meant, so that is an error; ``inputs`` all |0⟩ means the template's
    input, as before 1.5.1."""
    if "inputs" not in doc:
        return list(tokens)
    given = doc["inputs"]
    if not (isinstance(given, list) and len(given) == n):
        return given                                     # validate_state reports the shape with its own message
    if all(t == "0" for t in tokens) or all(t == "0" for t in given):
        return list(given) if all(t == "0" for t in tokens) else list(tokens)
    from .io_semantics import same_inputs
    try:
        if same_inputs(given, tokens):
            return list(given)
    except Exception:                                   # noqa: BLE001 - an invalid token: validate_state names it
        return given
    raise ExperimentError("inputs", "the template loads its own input (%s); set it with the template's input "
                          "parameters in circuit.params, or describe the circuit without the template (text or the "
                          "state form) to choose other inputs" % ", ".join("q%d %s" % (q, t) for q, t in
                                                                           enumerate(tokens) if t != "0")[:200])


def _observable_state(ob, n: int, path="observable") -> dict:
    from . import observables as OBS
    if not isinstance(ob, dict) or len(ob) != 1:
        raise ExperimentError(path, "one of {z: qubit}, {parity: [qubits]}, {bitstring: 'c(n-1)…c0'}, {pauli: text}")
    (k, v), = ob.items()
    try:
        if k == "z":
            o = OBS.z(int(v), n)
        elif k == "parity":
            o = OBS.parity([int(q) for q in v], n)
        elif k == "bitstring":
            s = str(v)
            if len(s) != n or set(s) - {"0", "1"}:
                raise ValueError("a bitstring of %d characters 0/1, written c(n-1)…c0" % n)
            o = OBS.projector(int(s, 2), n, label="P(%s)" % s)
        elif k == "pauli":
            from . import hamiltonian as HM
            h = HM.parse_pauli_sum(str(v), n)
            return dict(key=h.key[:40], label=h.label, tex=h.tex, pauli=h.text)
        else:
            raise ValueError("unknown observable kind %r" % k)
    except ExperimentError:
        raise
    except Exception as e:                              # noqa: BLE001
        raise ExperimentError("%s.%s" % (path, k), str(e)[:300]) from None
    return dict(key=o.key, label=o.label, tex=o.tex, values=list(o.values))


def _noise_state(nz, path="noise") -> dict:
    from .noise import PRESETS, NoiseSpec
    if nz is None:
        return dataclasses.asdict(NoiseSpec())
    if not isinstance(nz, dict):
        raise ExperimentError(path, "must be a mapping ({preset: name} and/or noise fields)")
    nz = dict(nz)
    base = NoiseSpec()
    if "preset" in nz:
        name = nz.pop("preset")
        if name not in PRESETS:
            raise ExperimentError(path + ".preset", "unknown preset %r (one of %s)" % (name, ", ".join(PRESETS)))
        base = PRESETS[name]
    names = {f.name for f in dataclasses.fields(NoiseSpec)}
    bad = set(nz) - names
    if bad:
        raise ExperimentError(path, "unknown field(s) %s" % sorted(bad))
    return dataclasses.asdict(base.replace(**nz))


def _methods_state(ms, path="methods") -> list:
    from .methods import REGISTRY, Recipe
    from .recipes import PRESETS as RP, preset
    from .state_schema import validate_params
    if not isinstance(ms, list) or not ms:
        raise ExperimentError(path, "a non-empty list of methods")
    out = []
    for i, m in enumerate(ms):
        p = "%s[%d]" % (path, i)
        if not isinstance(m, dict):
            raise ExperimentError(p, "must be a mapping")
        if "preset" in m:
            if m["preset"] not in RP:
                raise ExperimentError(p + ".preset", "unknown preset %r" % (m["preset"],))
            out.append(preset(m["preset"]).to_json())
            continue
        est = m.get("estimator")
        if est not in REGISTRY:
            raise ExperimentError(p + ".estimator", "unknown estimator %r (one of %s)" % (est, ", ".join(REGISTRY)))
        try:
            validate_params(est, m.get("params") or {})
        except ValueError as e:
            raise ExperimentError(p + ".params", str(e)) from None
        for k, allowed in (("twirl", ("none", "2q", "all")), ("dd", ("none", "X2", "XY4")),
                           ("readout", ("none", "tensored", "m3", "trex"))):
            if m.get(k, "none") not in allowed:
                raise ExperimentError("%s.%s" % (p, k), "one of %s" % ", ".join(allowed))
        bad = set(m) - {"id", "estimator", "params", "twirl", "dd", "readout"}
        if bad:
            raise ExperimentError(p, "unknown field(s) %s" % sorted(bad))
        out.append(Recipe.make(est, twirl=m.get("twirl", "none"), dd=m.get("dd", "none"),
                               readout=m.get("readout", "none"), **(m.get("params") or {})).to_json())
    return out


def to_state(doc: dict) -> dict:
    """The app state document an experiment document describes (validated)."""
    from . import __version__
    from .state_schema import validate_state
    if not isinstance(doc, dict):
        raise ExperimentError("", "an experiment document is a mapping")
    if doc.get("schema") != SCHEMA:
        raise ExperimentError("schema", "must be %r (got %r)" % (SCHEMA, doc.get("schema")))
    bad = set(doc) - TOP_KEYS
    if bad:
        raise ExperimentError("", "unknown top-level field(s) %s (allowed: %s)" % (sorted(bad), sorted(TOP_KEYS)))
    _scan_secrets(doc)
    if "state" in doc:
        short = set(doc) & (TOP_KEYS - {"schema", "name", "description", "state", "run"})
        if short:
            raise ExperimentError("", "give either 'state' or the short form, not both (also found %s)" % sorted(short))
        st = copy.deepcopy(doc["state"])
        if not isinstance(st, dict):
            raise ExperimentError("state", "must be a mapping (the app's exported state)")
    else:
        if "circuit" not in doc:
            raise ExperimentError("circuit", "required (or give 'state')")
        circ, ob, relabel, sym, tokens = _circuit_state(doc["circuit"])
        n = int(circ["n"])
        inputs = _inputs_state(doc, tokens, n)
        if "observable" in doc:
            ob = _observable_state(doc["observable"], n)
        budget = doc.get("budget") or {}
        if not isinstance(budget, dict) or set(budget) - {"shots", "t_shot_us"}:
            raise ExperimentError("budget", "a mapping with 'shots' and optionally 't_shot_us'")
        from .recipes import DEFAULT_SET, preset
        st = dict(app="QEM Explorer", version=__version__, circuit=circ, inputs=inputs,
                  basis=doc.get("basis", "Z" * n), relabel=bool(doc.get("relabel", relabel)),
                  compile=doc.get("compile") or {}, noise=_noise_state(doc.get("noise")), observable=ob,
                  symmetry=doc.get("symmetry", sym), symmetry_more="",
                  recipes=_methods_state(doc["methods"]) if "methods" in doc
                  else [preset(nm).to_json() for nm in DEFAULT_SET],
                  backend=doc.get("backend", "numpy"), shots=budget.get("shots", 4096),
                  t_shot_us=budget.get("t_shot_us", 250.0))
    try:
        st = plain(st)
    except (TypeError, ValueError) as e:
        raise ExperimentError("state", "not plain data: %s" % e) from None
    try:
        v = validate_state(st)
    except ValueError as e:
        raise ExperimentError("state", str(e)) from None
    if st.get("backend", "numpy") != v["backend"]:
        raise ExperimentError("backend", "unknown backend %r" % (st.get("backend"),))
    return st


def run_settings(doc: dict) -> dict:
    r = doc.get("run") or {}
    if not isinstance(r, dict) or set(r) - RUN_KEYS:
        raise ExperimentError("run", "a mapping with keys from %s" % sorted(RUN_KEYS))
    from .state_schema import _int
    try:
        return dict(seed=_int(r.get("seed", 0), "seed", 0, 2 ** 63 - 1),
                    repeats=_int(r.get("repeats", 120), "repeats", 1, 10 ** 7),
                    table_seed=_int(r.get("table_seed", 1), "table_seed", 0, 2 ** 63 - 1),
                    check_budget=bool(r.get("check_budget", True)))
    except ValueError as e:
        raise ExperimentError("run", str(e)) from None


def _np_default(x):
    if hasattr(x, "item"):                              # NumPy scalars
        return x.item()
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError("not JSON serializable: %s" % type(x).__name__)


def plain(obj):
    """Plain JSON data (NumPy scalars and arrays converted), so YAML, JSON and hashes agree."""
    return json.loads(json.dumps(obj, default=_np_default))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def config_hash(state: dict, run: dict) -> str:
    """SHA-256 of what determines the numbers: the state (without app name/version) and the run settings."""
    st = {k: v for k, v in state.items() if k not in ("app", "version")}
    return hashlib.sha256(canonical(dict(state=st, run=run)).encode()).hexdigest()


def method_id(recipe_json: dict) -> str:
    from .methods import Recipe
    r = Recipe.from_json(recipe_json)
    short = re.sub(r"[^A-Za-z0-9]+", "-", r.title()).strip("-")[:32] or r.estimator
    return "%s@%s" % (short, hashlib.sha256(canonical(r.to_json()).encode()).hexdigest()[:8])


# ------------------------------------------------------------------ resolution
@dataclasses.dataclass
class Resolved:
    state: dict
    run: dict
    config_hash: str
    compiled: object
    noise: object
    observable: object
    recipes: list
    method_ids: list
    symmetry: object
    backend: str
    shots: int
    t_shot_us: float


def compile_options(v: dict):
    """CompileOptions of a validated state; a forced layout that cannot be used is an error, never dropped."""
    from . import device
    from .compile import CompileOptions
    from .state_schema import COMPILE_DEFAULT, parse_layout
    cm = dict(COMPILE_DEFAULT, **v["compile"])
    dev = device.is_device(v["backend"])
    layout = cm["layout"] if dev or cm["layout"] in ("sabre", "trivial") else "sabre"
    init = parse_layout(cm.get("initial_layout", ""))
    if init and len(init) != int(v["n"]):
        raise ExperimentError("state.compile.initial_layout", "lists %d qubits for a %d-qubit circuit" % (
            len(init), int(v["n"])))
    return CompileOptions(mode=cm["mode"], basis_2q=cm["basis_2q"], coupling=cm["coupling"], layout=layout,
                          opt_level=int(cm["opt_level"]), seed=int(cm.get("seed", 11)),
                          device=v["backend"] if dev else None, initial_layout=tuple(init) if init else None)


def resolve(doc: dict) -> Resolved:
    """Validate a document and build the objects the simulators use (no simulation yet)."""
    from . import compile as CP
    from . import observables as OBS
    from .circuit import CircuitSpec, Op
    from .methods import Recipe
    from .noise import NoiseSpec
    from .state_schema import validate_state
    st = to_state(doc)
    run = run_settings(doc)
    v = validate_state(st)
    spec = CircuitSpec(int(v["n"]), tuple(Op(d["name"], tuple(d["qubits"]), tuple(d["params"]), d["kind"],
                                            d.get("tag", "")) for d in v["ops"]), v["name"])
    try:
        compiled = CP.build(spec, v["inputs"], v["basis"], compile_options(v), relabel=v["relabel"])
    except ExperimentError:
        raise
    except Exception as e:                              # noqa: BLE001
        raise ExperimentError("state.compile", "%s: %s" % (type(e).__name__, str(e)[:300])) from None
    obs = v["observable"] if not isinstance(v["observable"], tuple) else OBS.Observable(*v["observable"])
    recipes = [Recipe.from_json(d) for d in v["recipes"]]
    if not recipes:
        raise ExperimentError("methods", "at least one method is needed")
    return Resolved(state=st, run=run, config_hash=config_hash(st, run), compiled=compiled,
                    noise=NoiseSpec(**v["noise"]), observable=obs, recipes=recipes,
                    method_ids=[method_id(d) for d in v["recipes"]], symmetry=v["symmetry"], backend=v["backend"],
                    shots=int(v["shots"]), t_shot_us=float(v["t_shot_us"]))


def plan(r: Resolved) -> dict:
    """What a run would do, without simulating: sizes, per-method budget preflight, configuration hash."""
    from .methods import Context, Runner
    from .recipes import preflight
    runner = Runner(r.backend)
    rows = []
    for rc, mid in zip(r.recipes, r.method_ids):
        def make_ctx(c, o, rc=rc):
            return Context(c, r.noise, o, rc, runner, shots=r.shots, symmetry=r.symmetry, seed=r.run["seed"])
        pf = preflight(rc, make_ctx, r.compiled, r.observable, r.shots)
        rows.append(dict(method_id=mid, method=rc.title(), preflight="n/a (no fixed-cost model)" if pf is None else (
            "ok" if pf["feasible"] else "insufficient_budget"),
            min_budget=None if pf is None else float(pf["min_budget"]),
            training_shots=None if pf is None else int(pf["training"])))
    impl = r.compiled.implemented
    return dict(schema=SCHEMA, config_hash=r.config_hash, qubits=int(impl.n), two_qubit_gates=int(impl.count("2q")),
                depth=int(impl.depth()), backend=r.backend, shots=r.shots, run=r.run, methods=rows)


def provenance() -> dict:
    """Source, dependency and environment identity of a run (qemx.provenance: source-tree and lock digests, git
    commit when available, package versions and an inventory digest). Keys of 1.5.0 are kept."""
    from .provenance import provenance as _prov
    p = _prov()
    out = dict(p, created=p["created"])
    for mod, dist in (("numpy", "numpy"), ("scipy", "scipy"), ("qiskit", "qiskit"), ("qiskit_aer", "qiskit-aer"),
                      ("pandas", "pandas")):
        out[mod] = p["packages"].get(dist)
    return out


def execute(r: Resolved) -> dict:
    """Run the experiment on the app's simulators; returns the result document (rows as in the Playground table)."""
    import math
    from . import device
    from .recipes import compare, table_rows
    if r.backend.startswith("ibm_") and not device.is_device(r.backend):
        raise ExperimentError("backend", "real hardware is not run from this command; use the approved hardware "
                                         "workflow (docs/hardware_preflight.md)")
    ideal, outs, runner = compare(r.compiled, r.noise, r.observable, r.recipes, r.backend, r.shots, r.symmetry,
                                  seed=r.run["seed"], check_budget=r.run["check_budget"])
    samples, rows = table_rows(outs, ideal, r.shots, r.t_shot_us, repeats=r.run["repeats"], seed=r.run["table_seed"])
    by_title = {o.title: (o, mid) for o, mid in zip(outs, r.method_ids)}

    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
        if hasattr(x, "item"):
            return clean(x.item())
        return x
    out_rows = []
    for row in rows:
        o, mid = by_title.get(row["method"], (None, None))
        d = {k: clean(v) for k, v in row.items()}
        d["method_id"] = mid
        out_rows.append(d)
    failures = [dict(method_id=mid, method=o.title, status=getattr(o, "status", "") or "error", error=o.error)
                for o, mid in zip(outs, r.method_ids) if o.result is None]
    return dict(schema=RESULT_SCHEMA, config_hash=r.config_hash, ideal=float(ideal), rows=out_rows,
                failures=failures, simulations=int(runner.calls), provenance=provenance(), run=r.run,
                state=r.state)


def from_state(state: dict, name: str = "experiment", run: dict | None = None) -> dict:
    """An experiment document holding the app's state as is (the UI → YAML direction)."""
    doc = dict(schema=SCHEMA, name=str(name)[:80], state=plain(state))
    if run:
        doc["run"] = dict(run)
    to_state(doc)                                       # validate before handing it out
    return doc


def dump_yaml(doc: dict) -> str:
    import yaml
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120)


# ------------------------------------------------------------------ CLI
def _write_json_atomic(path: str, obj) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(prog="python -m qemx.experiment", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("validate", "plan", "run"):
        q = sub.add_parser(c)
        q.add_argument("file")
        if c == "run":
            q.add_argument("--out", required=True, help="output directory (created; an existing result with a "
                                                        "different configuration hash is never overwritten)")
            q.add_argument("--force", action="store_true", help="overwrite a result of another configuration")
    q = sub.add_parser("from-state", help="convert an app state (.json) into an experiment document (.yaml)")
    q.add_argument("file")
    q.add_argument("--out", required=True)
    q.add_argument("--name", default="experiment")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "from-state":
            with open(a.file, encoding="utf-8") as f:
                st = json.load(f)
            doc = from_state(st, a.name)
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(dump_yaml(doc))
            print("wrote %s (config %s)" % (a.out, config_hash(to_state(doc), run_settings(doc))[:12]))
            return 0
        doc = load(a.file)
        r = resolve(doc)
        if a.cmd == "validate":
            print("valid: %s · %d qubits · %d methods · config %s" % (
                doc.get("name", os.path.basename(a.file)), r.compiled.logical.n, len(r.recipes), r.config_hash[:12]))
            return 0
        if a.cmd == "plan":
            print(json.dumps(plan(r), indent=1))
            return 0
        os.makedirs(a.out, exist_ok=True)
        res_path = os.path.join(a.out, "result.json")
        if os.path.exists(res_path) and not a.force:
            with open(res_path, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("config_hash") != r.config_hash:
                raise ExperimentError("", "%s holds a result of another configuration (%s…); use another --out or "
                                          "--force" % (res_path, str(old.get("config_hash"))[:12]))
        res = execute(r)
        _write_json_atomic(res_path, res)
        import csv
        cols = sorted({k for row in res["rows"] for k in row})
        with open(os.path.join(a.out, "rows.csv.part"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(res["rows"])
        os.replace(os.path.join(a.out, "rows.csv.part"), os.path.join(a.out, "rows.csv"))
        print("ideal %.6g · %d methods · %d failures · config %s → %s" % (
            res["ideal"], len(res["rows"]), len(res["failures"]), r.config_hash[:12], a.out))
        return 0
    except ExperimentError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
