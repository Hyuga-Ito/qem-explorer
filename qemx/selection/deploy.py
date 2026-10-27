r"""Deployment path of the protocol-selection recommender (1.7.0; review §8.2–8.3, prompt section 7).

The research study labels *benchmark tasks*: exact ideal values, every candidate executed many times, oracle
comparisons. A *deployment* has none of that: a circuit, its input preparation, a diagonal observable, a device
snapshot, a compilation and a total budget B. This module keeps the two apart.

* ``DeploymentTask`` — the shared task representation of the Playground, the command line and batch runs. It is built
  from the app's state document (``from_state``: the Playground's ``Export`` and ``qemx.experiment`` documents resolve
  to the same state) or from a study task (``from_spec``), and identified by ``key()``: the same experiment gives the
  same key, features and recommendation from the UI and from the CLI.
* ``features`` — only the features a model needs, each with its information class, its cost (seconds) and how it
  was computed. Structural, observable, light-cone and calibration features (class ``available``) use the compiled
  circuit and the snapshot's calibration only — no state is simulated. Features of class ``simulated`` need an
  exact simulation of the snapshot model (exponential in width); they are computed only when explicitly allowed and
  reported as such.
* ``ModelBundle`` — a versioned, self-contained recommender: fitted preprocessing (imputation medians,
  standardization) and per-protocol models serialized as plain arrays (JSON and NPZ, loaded without pickle, evaluated
  by NumPy code here and verified against scikit-learn when the bundle is built), the feature names and order, the
  candidate protocols with their definitions and digest, the training support (feature ranges, families, widths,
  devices, budgets), a validation summary (nested cross-validation, confirmation if run), the selective rule and its
  frozen threshold, the fallback (SBS-B ranks per budget) and the guarantee status of every number it reports.
* ``recommend`` — features → OOD checks → structural feasibility at B (no simulation) → predicted log risks
  (model-conditional) → the choice, its margin and the frozen selective rule (accept or fall back to SBS-B), with
  every assumption and cost in the result. Outside the declared capabilities the result is ``unsupported``; outside
  the training support it is flagged ``out-of-distribution`` and falls back.

1.8.0 (review findings F1, F2, F4; ``POLICY``):

* a bounded-cost **guard** runs before compilation: budget, device, width against the device and against the
  declared dense limits, observable form. Unsupported inputs return at once, never after an exponential allocation;
* the readout-calibration quantities of the cheap features are computed by a **factorized** product-state
  calculation (``device_exact.product_calibration``: the same instruction list as the dense engine, exact for
  calibration circuits whose instructions are all single-qubit, which the code checks), and observables may be given
  **compactly** as Z-string terms (``DeploymentTask.obs_terms``), so the deployable path has no 2^n object;
* candidate feasibility follows the shared **preflight contract** (``qemx.selection.preflight``): verified-feasible,
  infeasible or unverified, with its level. A learning candidate's training design is constructed (its ideal
  training values computed classically, no shots) before it can be accepted; a degenerate design makes it
  infeasible; candidates that stay unverified are never accepted or used as the fallback. The verified design's seed
  is returned, so execution replays it;
* a bundle is checked for **scientific compatibility** at load: candidate definitions, feature definitions and the
  calibration snapshots it was trained on. Changed definitions are refused; a changed snapshot under the same device
  name is a transfer (out-of-distribution), not an error.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .. import textio as TX
from . import features as FE
from . import protocols as PR

BUNDLE_SCHEMA = "qemx.selection.bundle/1"
BUNDLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundles")
POLICY = "qemx.selection.policy/2"          # 1.8.0: guard, factorized calibration, preflight contract (module text)
DENSE_OBS_MAX_WIDTH = 20                    # a dense observable vector (2^n values) is accepted up to this width
SIMULATION_MAX_WIDTH = 11                   # simulated features: dense density matrices (16·4^n bytes ≈ 64 MiB at 11)


class Unsupported(ValueError):
    """An experiment outside the recommender's declared capabilities."""


# ====================================================================== the shared task representation
@dataclass(frozen=True)
class DeploymentTask:
    n: int
    ops: tuple                         # ((name, qubits, params, kind, tag), …) of the circuit body (no preparation)
    inputs: tuple                      # per-qubit input tokens (prepared once, before the circuit)
    basis: str                         # measurement basis per qubit (q0 first)
    relabel: bool                      # classical bits read in reverse order
    observable: tuple                  # (key, label, values) of a diagonal observable over the measured bits
    device: str                        # calibration snapshot (fake device specification)
    budget: int                        # total shot budget B
    opt_level: int = 1
    seed: int = 11
    initial_layout: tuple | None = None
    name: str = "circuit"
    obs_terms: tuple | None = None     # 1.8.0: compact form ((bits, coefficient), …) of the diagonal observable as Z
    #                                    strings over classical bits; with it, ``observable[2]`` may be empty
    symmetry: tuple | None = None      # 1.8.0: declared Z-parity checks (qubits, eigenvalue, "more"), from the task's
    #                                    construction (a study family or an app template) — never from its output
    symmetry_source: str = ""          # 1.9.0: "construction" (study family), "state, validated" (declared in the
    #                                    state and checked on the ideal measured distribution); not part of key()
    notes: tuple = ()                  # 1.9.0: conversion notes (device override, dropped declaration); not in key()

    def key(self) -> str:
        """Identity of the experiment. A dense observable is identified by its values (as in 1.7.0, so keys are
        unchanged); a compact one by its terms."""
        if self.observable[2]:
            ob = [self.observable[0], hashlib.sha256(np.asarray(self.observable[2], float).tobytes()).hexdigest()[:16]]
        else:
            ob = [self.observable[0], "terms", hashlib.sha256(json.dumps(
                [[sorted(int(b) for b in bits), float(c)] for bits, c in (self.obs_terms or ())]).encode()
            ).hexdigest()[:16]]
        d = dict(n=self.n, ops=[list(map(_plain, o)) for o in self.ops], inputs=list(self.inputs), basis=self.basis,
                 relabel=self.relabel, observable=ob, device=self.device, budget=self.budget,
                 opt_level=self.opt_level, seed=self.seed, initial_layout=list(self.initial_layout or ()))
        if self.symmetry:
            d["symmetry"] = [list(self.symmetry[0]), int(self.symmetry[1]), str(self.symmetry[2])]
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:20]

    def terms(self) -> list:
        """[(|coefficient|, frozenset of classical bits)] of the observable's non-identity Z strings (compact form, or
        the Walsh–Hadamard transform of the dense values)."""
        if self.obs_terms is not None:
            return [(abs(float(c)), frozenset(int(b) for b in bits)) for bits, c in self.obs_terms
                    if bits and abs(float(c)) > 1e-12]
        from ..phase.descriptors import observable_terms
        return [(abs(c), frozenset(bits)) for c, bits, _ in observable_terms(self.obs())]

    def body(self):
        from ..circuit import CircuitSpec, Op
        return CircuitSpec(int(self.n), tuple(Op(o[0], tuple(o[1]), tuple(o[2]), o[3], o[4]) for o in self.ops),
                           self.name)

    def obs(self):
        """The observable as a dense diagonal vector (2^n values). From the compact form only up to
        DENSE_OBS_MAX_WIDTH; beyond it raises Unsupported (the deployable path does not need it)."""
        from ..observables import Observable
        k, label, vals = self.observable
        if not vals and self.obs_terms is not None:
            if self.n > DENSE_OBS_MAX_WIDTH:
                raise Unsupported("a dense observable at width %d (2^%d values) is not built; the deployable path "
                                  "uses the compact Z-string terms" % (self.n, self.n))
            v = np.zeros(2 ** int(self.n))
            idx = np.arange(2 ** int(self.n))
            for bits, c in self.obs_terms:
                par = np.zeros_like(idx)
                for b in bits:
                    par ^= (idx >> int(b)) & 1
                v += float(c) * (1 - 2 * par)
            vals = tuple(float(x) for x in v)
        return Observable(k, label, "", tuple(float(x) for x in vals))

    @property
    def supplied_backend(self) -> str:
        from .task import with_modifiers
        return self.device if "eng=" in self.device else with_modifiers(self.device, "eng=exact")

    def compile(self):
        from .. import compile as CP
        return CP.build(self.body(), list(self.inputs), self.basis,
                        CP.CompileOptions(device=self.device, opt_level=int(self.opt_level), seed=int(self.seed),
                                          initial_layout=tuple(self.initial_layout) if self.initial_layout else None),
                        relabel=bool(self.relabel))

    def to_json(self) -> dict:
        return dict(n=self.n, ops=[list(map(_plain, o)) for o in self.ops], inputs=list(self.inputs),
                    basis=self.basis, relabel=self.relabel, observable=dict(key=self.observable[0],
                                                                           label=self.observable[1]),
                    device=self.device, budget=self.budget, opt_level=self.opt_level, seed=self.seed,
                    initial_layout=list(self.initial_layout) if self.initial_layout else None, key=self.key(),
                    symmetry=None if not self.symmetry else [list(self.symmetry[0]), int(self.symmetry[1]),
                                                             str(self.symmetry[2])],
                    symmetry_source=self.symmetry_source or None, notes=list(self.notes),
                    **({} if self.obs_terms is None else dict(obs_terms=[[sorted(int(b) for b in bits), float(c)]
                                                                         for bits, c in self.obs_terms])))


def _plain(x):
    if isinstance(x, (list, tuple)):
        return [_plain(y) for y in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    return x


def _ops_of(spec) -> tuple:
    return tuple((o.name, tuple(int(q) for q in o.qubits), tuple(float(p) for p in o.params), o.kind, o.tag or "")
                 for o in spec.ops)


SYMMETRY_CHECK_MAX_WIDTH = 20              # a state's declared symmetry is checked on the ideal distribution up to here


def normalize_symmetry(sym) -> tuple | None:
    """(qubits, eigenvalue, more) of a symmetry declaration in any of its spellings — the state schema's
    (qubits, eigenvalue) or (qubits, eigenvalue, more), a ``protocols.declared_symmetry`` dict, or None. ``more`` is
    normalized to "a b; c d"."""
    if not sym:
        return None
    if isinstance(sym, dict):
        q, ev, more = sym.get("qubits"), sym.get("eigenvalue"), sym.get("more", "")
    else:
        q, ev = sym[0], sym[1]
        more = sym[2] if len(sym) > 2 else ""
    if more:
        from ..methods.symmetry import parse_checks
        groups = parse_checks(more)
        if any(len(set(g)) != len(g) for g in groups):
            raise ValueError("a parity check names a qubit twice (%s)" % more)
        more = "; ".join(" ".join(str(int(x)) for x in g) for g in groups)
    q = tuple(int(x) for x in q)
    if len(set(q)) != len(q):                        # 1.9.0: Z0Z0Z1 is not a two-qubit check (vacuous parity)
        raise ValueError("the symmetry names a qubit twice (%s)" % list(q))
    return q, int(ev), str(more or "")


def check_symmetry(n: int, ops, inputs, basis: str, relabel: bool, sym: tuple) -> dict:
    """Whether a declared symmetry holds for *this* experiment (1.9.0; review R1): the probability that the ideal
    measured distribution — the inputs prepared once, the circuit, the measurement bases, the bit order — violates the
    declared parity checks (each over classical bits, as symmetry verification evaluates them). ``valid`` when it is
    below 1e-9. An ideal state-vector calculation: exponential in width, run up to SYMMETRY_CHECK_MAX_WIDTH qubits
    (``unchecked`` above). This checks a declaration; it never searches the output for symmetries."""
    t0 = time.perf_counter()
    if n > SYMMETRY_CHECK_MAX_WIDTH:
        return dict(status="unchecked", reason="the ideal check is not run above %d qubits" % SYMMETRY_CHECK_MAX_WIDTH,
                    seconds=0.0)
    from ..circuit import CircuitSpec, Op
    from ..compile import with_io
    from ..engine import permute_probs
    from .. import io_semantics as IO
    from ..methods.symmetry import parity_mask, parse_checks
    body = CircuitSpec(int(n), tuple(Op(o[0], tuple(o[1]), tuple(o[2]), o[3], o[4]) for o in ops), "check")
    psi = IO.ideal_statevector(with_io(body, list(inputs), str(basis)))
    p = np.abs(psi) ** 2
    p = permute_probs(p / p.sum(), IO.measure_map(n, relabel), n)
    ok = parity_mask(n, sym[0], sym[1])
    for g in parse_checks(sym[2]):
        ok &= parity_mask(n, g, 1)
    viol = float(p[~ok].sum())
    return dict(status="valid" if viol < 1e-9 else "invalid", violation_probability=viol,
                seconds=time.perf_counter() - t0,
                reason="" if viol < 1e-9 else "the ideal measured distribution violates the declared parity checks "
                                               "with probability %.3g" % viol)


def _validate_layout(init, n: int, dev: str):
    from .. import device as DV
    if not init:
        return None
    q = [int(x) for x in init]
    if len(q) != int(n) or len(set(q)) != len(q) or min(q) < 0:
        raise Unsupported("the state's physical layout %s must name %d distinct non-negative qubits" % (q, n))
    try:
        nq = int(DV.fake_backend(DV.base_name(dev)).num_qubits)
    except Exception:                                # noqa: BLE001 - a device without a qubit count is not checked
        nq = None
    if nq is not None and max(q) >= nq:
        raise Unsupported("the physical layout %s is not valid on %s (%d qubits); clear the layout or choose another "
                          "device" % (q, dev, nq))
    return tuple(q)


def from_state(state: dict, budget: int, device: str | None = None) -> DeploymentTask:
    """The deployment task of an app state document (the Playground's state, an exported state or a
    ``qemx.experiment`` document's state). The input preparation is the per-qubit inputs, prepared once (a preparation
    saved in the circuit is folded into them, as the Playground does).

    1.9.0 (second review N8, review R1), one rule for the UI, the command line and experiment documents:

    * **device** — an explicit ``device`` (the Playground's "Device snapshot", ``recommend --device``) is the
      calibration snapshot; without it the state's backend when it is a fake device; otherwise Unsupported. The
      state's physical layout is checked on the effective device (distinct, in range) and kept; a layout chosen for
      another device is noted;
    * **symmetry** — the state's declared Z-parity checks (``symmetry``, ``symmetry_more``) are kept when they hold
      for this experiment: checked on the ideal measured distribution (inputs, circuit, bases, bit order;
      ``check_symmetry``). A declaration that does not hold (e.g. after an edit, another input or another basis), or
      that cannot be checked (above SYMMETRY_CHECK_MAX_WIDTH qubits), is not used and the reason is noted; symmetry
      verification is then not applicable, exactly as for a study task without a construction declaration."""
    from .. import device as DV
    from .. import io_semantics as IO
    from ..circuit import CircuitSpec, Op
    from ..state_schema import parse_layout, validate_state
    try:
        v = validate_state(state)
    except ValueError as e:                          # 1.9.0: an invalid state is unsupported input, not a crash
        raise Unsupported("the state is not valid: %s" % e) from None
    sp = CircuitSpec(int(v["n"]), tuple(Op(o["name"], tuple(o["qubits"]), tuple(o["params"]), o["kind"],
                                          o.get("tag", "")) for o in v["ops"]), v["name"])
    inputs, body = IO.split_preparation(sp, v["inputs"])
    ob = v["observable"]
    if hasattr(ob, "terms") or isinstance(ob, dict) and "pauli" in ob:
        raise Unsupported("the recommender supports diagonal observables (Z strings, projectors); a Hamiltonian with "
                          "X or Y terms is measured in several settings and is outside the study's task class")
    if isinstance(ob, tuple):
        key, label, _tex, vals = ob
    else:
        key, label, vals = ob.key, ob.label, ob.values
    vals = np.asarray(vals, float)
    if vals.size != 2 ** int(v["n"]) or np.max(np.abs(vals)) > 1 + 1e-12:
        raise Unsupported("the observable must be diagonal over the measured bits with |O| ≤ 1 (the study's task "
                          "class)")
    backend = v["backend"]
    notes = []
    if device:
        try:
            ok_dev = bool(DV.is_device(device))
            if ok_dev and "@" in device:
                from .task import with_modifiers
                DV.parse_device(device if "eng=" in device else with_modifiers(device, "eng=exact"))
        except ValueError as e:
            raise Unsupported("%r is not a valid device specification: %s" % (device, e)) from None
        if not ok_dev:
            raise Unsupported("%r is not a fake-device calibration snapshot" % device)
        dev = device
        if DV.is_device(backend) and backend != device:
            notes.append("device %s (explicitly selected) replaces the state's backend %s" % (device, backend))
    elif DV.is_device(backend):
        dev = backend
    else:
        raise Unsupported("the recommender is trained on fake-device calibration snapshots; choose a fake device "
                          "(the state's backend %r is a noise injector)" % backend)
    cm = dict(v.get("compile") or {})
    try:
        lay = parse_layout(cm.get("initial_layout", "")) or None
    except ValueError as e:
        raise Unsupported("the state's physical layout is not valid: %s" % e) from None
    init = _validate_layout(lay, int(v["n"]), dev)
    if init and DV.is_device(backend) and backend != dev:
        notes.append("the state's physical layout %s (chosen for %s) is used on %s; it is valid there but may not be "
                     "its best layout" % (list(init), backend, dev))
    ops = _ops_of(body)
    try:
        sym = normalize_symmetry(v.get("symmetry"))
    except ValueError as e:
        notes.append("the state's declared symmetry was not used: %s" % e)
        sym = None
    src = ""
    if sym:
        chk = check_symmetry(int(v["n"]), ops, inputs, str(v["basis"]), bool(v["relabel"]), sym)
        if chk["status"] == "valid":
            src = "state, validated"
        else:
            notes.append("the state's declared symmetry %s was not used: %s" % (
                _sym_text(sym), chk["reason"]))
            sym = None
    return DeploymentTask(int(v["n"]), ops, tuple(inputs), str(v["basis"]), bool(v["relabel"]),
                          (str(key), str(label), tuple(float(x) for x in vals)), str(dev), int(budget),
                          int(cm.get("opt_level", 1)), int(cm.get("seed", 11)), init, str(v["name"]),
                          symmetry=sym, symmetry_source=src, notes=tuple(notes))


def _sym_text(sym) -> str:
    t = "Z%s = %+d" % ("Z".join(str(q) for q in sym[0]), sym[1])
    return t + (" and %s" % sym[2] if sym[2] else "")


def recommendation_signature(task: DeploymentTask, bundle, *, allow_simulation: bool,
                             preflight_level: str | None = None) -> str:
    """What a shown recommendation depends on (1.9.0; review R8): the task (key: circuit, inputs, bases, bit order,
    observable, device, budget, layout, declared symmetry), the bundle's arrays, the policy, whether simulation was
    allowed and the preflight level. Any change marks a shown recommendation out of date."""
    d = dict(task=task.key(), bundle=bundle.meta.get("arrays_sha256"), version=bundle.meta.get("version"),
             policy=POLICY, allow_simulation=bool(allow_simulation),
             level=preflight_level or ("allocation" if allow_simulation else "design"))
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:20]


def from_spec(spec, budget: int) -> DeploymentTask:
    """The deployment task of a study task (``task.TaskSpec``): the same physical experiment, without its ideal
    value. Compiling it reproduces the study's compiled circuit (layout, seed and optimization level as the study)."""
    from .task import family_circuit, input_tokens, layout_for, observable_of
    body, basis, default_obs, relabel, fam_tokens = family_circuit(spec)
    tokens = input_tokens(spec, fam_tokens)
    obs = observable_of(spec, default_obs, int(body.n))
    lay = layout_for(spec, int(body.n))
    sym = PR.declared_symmetry(spec)
    return DeploymentTask(int(body.n), _ops_of(body), tuple(tokens), str(basis), bool(relabel),
                          (obs.key, obs.label, tuple(float(x) for x in obs.o)), spec.device, int(budget),
                          int(spec.opt_level), int(spec.seed), tuple(int(q) for q in lay), spec.label,
                          symmetry=normalize_symmetry(sym), symmetry_source="construction" if sym else "")


def to_state(spec, budget: int = 4096) -> dict:
    """The app state document of a study task (1.9.0): the same physical experiment — circuit body, per-qubit inputs,
    measurement bases, bit order, diagonal observable, device snapshot and layout, and the construction's declared
    symmetry — so that ``from_state(to_state(spec), B)`` and ``from_spec(spec, B)`` describe one experiment (parity
    tests). The state carries no ideal value."""
    from .. import __version__
    from .task import family_circuit, input_tokens, layout_for, observable_of
    body, basis, default_obs, relabel, fam_tokens = family_circuit(spec)
    tokens = input_tokens(spec, fam_tokens)
    obs = observable_of(spec, default_obs, int(body.n))
    lay = layout_for(spec, int(body.n))
    sym = normalize_symmetry(PR.declared_symmetry(spec))
    return dict(app="QEM Explorer", version=__version__, circuit=body.to_json(), inputs=list(tokens), basis=str(basis),
                relabel=bool(relabel), compile=dict(opt_level=int(spec.opt_level), seed=int(spec.seed),
                                                    initial_layout=",".join(str(int(q)) for q in lay)),
                noise={}, observable=dict(key=obs.key, label=obs.label, values=[float(x) for x in obs.o]),
                symmetry=(list(sym[0]) + [int(sym[1])]) if sym else None, symmetry_more=(sym[2] if sym else ""),
                recipes=[], backend=spec.device, shots=int(budget), t_shot_us=250.0)


# ====================================================================== features at deployment
STRUCTURAL = ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "pulses_1q", "non_clifford_slots",
              "idle_fraction", "obs_weight", "obs_terms", "obs_support")
LIGHTCONE = ("lc_qubits", "lc_gates", "lc_2q", "lc_locations", "lam_lc", "lam_lc_exact", "lam_lc_gate", "lam_lc_idle",
             "lam_total", "lc_share", "lam_ro")
SIMULATED = tuple(f.name for f in FE.DICTIONARY if f.info == "simulated")
METHOD = {**{k: "compiled circuit and observable (no simulation)" for k in STRUCTURAL},
          **{k: "compiled circuit, light cone and the snapshot's calibration (no simulation)" for k in FE.NOISE18},
          **{k: "backward light cone with the snapshot's calibrated error rates (no simulation)" for k in LIGHTCONE},
          "duration_us": "ALAP schedule with the snapshot's gate durations (no simulation)",
          "pec_log_gamma": "product of the depolarizing quasi-probability norms of the snapshot's median rates and "
                           "the readout inverse's norm from the two one-layer calibration circuits (no simulation of "
                           "the target circuit)",
          "zne_eta_l1_linear": "fold scales achieved by global folding (gate counts) (no simulation)",
          "zne_eta_l1_richardson": "fold scales achieved by global folding (gate counts) (no simulation)",
          "log10_budget": "the budget",
          **{k: "exact simulation of the snapshot model (exponential in width)" for k in SIMULATED}}


def _achieved(spec, lam):
    from ..methods.zne import achieved_lambda
    from ..transforms import fold_global
    return achieved_lambda(spec, fold_global(spec, float(lam)))


def cheap_protocol_features(comp, supplied: str, *, report: dict | None = None) -> dict:
    """pec_log_gamma and the ZNE noise-amplification norms without simulating the target state (the same quantities
    the study reads from the methods' results; equality is tested on the study's tasks).

    The readout part of pec_log_gamma needs the assignment matrices of the two one-layer calibration circuits
    (|0…0⟩ and X on every qubit; the X gates carry their gate error). 1.7.0 obtained them from the dense
    density-matrix engine (4^n memory). 1.8.0 computes them per qubit (``device_exact.product_calibration``), which is
    the same model whenever every resolved instruction of the calibration circuits acts on one qubit — checked, not
    assumed. If not (a correlated channel), the dense engine is used up to SIMULATION_MAX_WIDTH and the method is
    reported; beyond it the feature is not computed (NaN, reported) rather than allocating an impractical state."""
    from .. import device
    from .. import device_exact as DE
    from ..methods.base import assignment_inverse_gamma
    from ..methods.pec import depol_quasi, gamma_of
    phys = tuple(comp.info["phys"])
    dm = device.get_model(supplied, phys)
    noise = dm.equivalent_noise()
    lg = 0.0
    for o in comp.implemented.ops:
        if o.kind != "gate" or o.is_virtual:
            continue
        p = noise.p1 if o.k == 1 else noise.p2
        lg += math.log(gamma_of(depol_quasi(float(p), o.k, "inverse")))
    n = int(comp.implemented.n)
    method = "factorized"
    try:
        A = DE.product_calibration(n, dm, comp.measure_map)
    except DE.NotFactorizable as e:
        if n > SIMULATION_MAX_WIDTH:
            A, method = None, "not computed: %s, and width %d exceeds the dense limit %d" % (e, n, SIMULATION_MAX_WIDTH)
        else:
            from ..methods import Context, Runner
            ctx = Context(comp, noise, None, PR.PORTFOLIO["PEC"].recipe(), Runner(backend=supplied, layout=phys))
            A, method = ctx.calibration()["A"], "dense engine (%s)" % e
    if A is None:
        lg = math.nan
    else:
        for a in A:
            lg += math.log(assignment_inverse_gamma(a[1, 0], a[0, 1]))
    if report is not None:
        report["calibration_method"] = method
    out = dict(pec_log_gamma=lg)
    for key, deg in (("ZNE-L", 1), ("ZNE-R", 2)):
        lams = [_achieved(comp.implemented, l) for l in dict(PR.PORTFOLIO[key].params)["scales"]]
        out["zne_eta_l1_%s" % ("linear" if deg == 1 else "richardson")] = FE._eta_l1(lams, deg)
    return out


def guard(task: DeploymentTask, bundle=None) -> dict:
    """Bounded-cost checks before compilation (1.8.0; review finding F2): raises Unsupported with the reason when a
    request is outside what the deployment path can evaluate at bounded cost, else returns notes. Never compiles,
    simulates or builds a 2^n object."""
    from .. import device as DV
    notes = {}
    try:
        B = int(task.budget)
    except (TypeError, ValueError):
        raise Unsupported("the budget must be a positive integer number of shots, got %r" % (task.budget,)) from None
    if B < 1 or B != task.budget:
        raise Unsupported("the budget must be a positive integer number of shots, got %r" % (task.budget,))
    if not DV.is_device(task.device):
        raise Unsupported("the recommender is trained on fake-device calibration snapshots; %r is not one"
                          % (task.device,))
    n = int(task.n)
    if n < 1:
        raise Unsupported("the circuit has no qubits")
    nq = DV.num_qubits(task.device)
    if n > nq:
        raise Unsupported("%d qubits do not fit on %s (%d qubits)" % (n, DV.base_name(task.device), nq))
    vals = task.observable[2]
    if vals:
        if n > DENSE_OBS_MAX_WIDTH:
            raise Unsupported("a dense observable (2^%d values) above width %d: give it as Z-string terms "
                              "(DeploymentTask.obs_terms)" % (n, DENSE_OBS_MAX_WIDTH))
        if len(vals) != 2 ** n:
            raise Unsupported("the observable has %d values, not 2^%d" % (len(vals), n))
    elif task.obs_terms is None:
        raise Unsupported("the task has no observable")
    else:
        for bits, c in task.obs_terms:
            if any(not 0 <= int(b) < n for b in bits):
                raise Unsupported("an observable term acts on a bit outside 0…%d" % (n - 1))
        if sum(abs(float(c)) for _, c in task.obs_terms) > 1 + 1e-9:
            raise Unsupported("the observable's coefficients exceed |O| ≤ 1 (the study's task class)")
    if bundle is not None:
        sup = bundle.meta.get("support", {})
        widths = sup.get("widths") or []
        if widths and not (min(widths) <= n <= max(widths)):
            notes["width"] = ("width %d is outside the training widths %s: out-of-distribution (fallback); features "
                              "are computed at bounded cost" % (n, widths))
        if n > SIMULATION_MAX_WIDTH and any(FE.BY_NAME[f].info == "simulated" for f in bundle.features):
            notes["simulation"] = ("width %d exceeds the simulated-feature limit %d: simulated features are not "
                                   "computed (≈%.3g GiB per density matrix)" % (n, SIMULATION_MAX_WIDTH,
                                                                              16 * 4.0 ** n / 2 ** 30))
    return notes


_FEATURE_CACHE: "dict" = {}
FEATURE_CACHE_SIZE = 256


def features(task: DeploymentTask, names, *, allow_simulation: bool = False, compiled=None,
             use_cache: bool = True) -> tuple:
    """(values, report) of the requested features only, with a content-keyed cache (1.8.0): every feature but
    log10_budget is budget-independent, so a recomputation for the same experiment at another budget reuses them. The
    key is the experiment without its budget, the feature names and their definitions' fingerprint, the simulation
    switch and the policy version; a hit is reported (``cache``) and its original seconds are kept."""
    import dataclasses
    key = None
    if use_cache:
        key = (dataclasses.replace(task, budget=1).key(), tuple(names), feature_fingerprint(
            [n for n in names if n in FE.BY_NAME]), bool(allow_simulation), POLICY)
        hit = _FEATURE_CACHE.get(key)
        if hit is not None:
            vals, rep = TX.loads(hit[0]), TX.loads(hit[1])
            if "log10_budget" in vals:
                vals["log10_budget"] = math.log10(int(task.budget))
            rep["cache"] = "hit (budget-independent features of this experiment)"
            return vals, rep
    vals, rep = _features(task, names, allow_simulation=allow_simulation, compiled=compiled)
    if key is not None:
        if len(_FEATURE_CACHE) >= FEATURE_CACHE_SIZE:
            _FEATURE_CACHE.pop(next(iter(_FEATURE_CACHE)))
        _FEATURE_CACHE[key] = (TX.dumps(vals), TX.dumps(rep))
        vals = TX.loads(_FEATURE_CACHE[key][0])
        rep = dict(TX.loads(_FEATURE_CACHE[key][1]), cache="miss")
    return vals, rep


def _features(task: DeploymentTask, names, *, allow_simulation: bool = False, compiled=None) -> tuple:
    """(values, report) of the requested features only. ``report``: one row per feature with its information class,
    the seconds it took (shared work is charged to the first feature that needs it) and the method. Simulated
    features are computed only with ``allow_simulation`` (else NaN, reported as not computed)."""
    from .. import features as FT
    from ..context_noise import alap_schedule
    from ..phase.lightcone import calculate_lambda_lc
    names = list(names)
    bad = [n for n in names if n not in FE.BY_NAME or FE.BY_NAME[n].info == "oracle"]
    if bad:
        raise ValueError("not deployable features: %s" % ", ".join(bad))
    vals, rep = {}, []
    t0 = time.perf_counter()
    comp = compiled if compiled is not None else task.compile()
    t_compile = time.perf_counter() - t0
    sup = task.supplied_backend
    extra = {}

    def record(keys, seconds, computed=True):
        for i, k in enumerate(keys):
            rep.append(dict(feature=k, info=FE.BY_NAME[k].info, seconds=float(seconds if i == 0 else 0.0),
                            method=METHOD.get(k, ""), computed=bool(computed and np.isfinite(vals.get(k, math.nan)))))
    want = set(names)
    terms = None
    if want & set(STRUCTURAL):
        t = time.perf_counter()
        ks = [k for k in STRUCTURAL if k in want and not k.startswith("obs_")]
        if ks:
            v, _ = FT.compute(comp, None, sup, None, names=ks)
            vals.update({k: float(v.get(k, math.nan)) for k in ks})
        ko = [k for k in STRUCTURAL if k in want and k.startswith("obs_")]
        if ko:                     # 1.8.0: from the Z-string terms (compact or transformed), as observable_descriptors
            terms = task.terms()
            w = [len(b) for _, b in terms]
            od = dict(obs_weight=float(max(w) if w else 0), obs_terms=float(len(w)),
                      obs_support=float(len(frozenset().union(*[b for _, b in terms])) if terms else 0))
            vals.update({k: od[k] for k in ko})
        record(ks + ko, time.perf_counter() - t)
    if "duration_us" in want:
        from .. import device
        t = time.perf_counter()
        dm = device.get_model(sup, tuple(comp.info["phys"]))
        vals["duration_us"] = float(alap_schedule(comp.implemented, dm.durations)[3]) * 1e6
        record(["duration_us"], time.perf_counter() - t)
    r = None
    if want & (set(LIGHTCONE) | set(FE.NOISE18)):
        t = time.perf_counter()
        if terms is None:
            terms = task.terms()
        bits = frozenset().union(*[b for _, b in terms]) if terms else frozenset()
        mm = comp.measure_map or {}
        r = calculate_lambda_lc(comp, sup, support=sorted({int(mm.get(b, b)) for b in bits}))
        lc = dict(lc_qubits=float(len(r.lc_qubits)), lc_gates=float(r.lc_gates), lc_2q=float(r.lc_2q_gates),
                  lc_locations=float(r.n_loc_lc), lam_lc=float(r.lam_lc), lam_lc_exact=float(r.lam_lc_exact),
                  lam_lc_gate=float(r.lam_lc_gate), lam_lc_idle=float(r.lam_lc_idle), lam_total=float(r.lam_total),
                  lc_share=float(r.lam_lc / r.lam_total) if r.lam_total > 0 else 0.0, lam_ro=float(r.lam_ro))
        ks = [k for k in LIGHTCONE if k in want]
        vals.update({k: lc[k] for k in ks})
        record(ks, time.perf_counter() - t)
    if want & set(FE.NOISE18):                       # 1.8.0 noise-composition descriptors (snapshot only)
        from .. import device as DV
        t = time.perf_counter()
        nz = FE.noise18(comp, DV.get_model(sup, tuple(comp.info["phys"])), rep=r, support_bits=bits)
        ks = [k for k in FE.NOISE18 if k in want]
        vals.update({k: nz[k] for k in ks})
        record(ks, time.perf_counter() - t)
    cheap = [k for k in ("pec_log_gamma", "zne_eta_l1_linear", "zne_eta_l1_richardson") if k in want]
    if cheap:
        t = time.perf_counter()
        c = cheap_protocol_features(comp, sup, report=extra)
        vals.update({k: c[k] for k in cheap})
        record(cheap, time.perf_counter() - t)
    if "log10_budget" in want:
        vals["log10_budget"] = math.log10(int(task.budget))
        record(["log10_budget"], 0.0)
    sim = [k for k in names if FE.BY_NAME[k].info == "simulated"]
    if sim:
        if allow_simulation and int(task.n) > SIMULATION_MAX_WIDTH:
            extra["simulation"] = "not computed: width %d above SIMULATION_MAX_WIDTH = %d" % (task.n,
                                                                                          SIMULATION_MAX_WIDTH)
            for k in sim:
                vals[k] = math.nan
            record(sim, 0.0, computed=False)
        elif allow_simulation:
            t = time.perf_counter()
            vals.update({k: v for k, v in simulated_features(task, comp).items() if k in sim})
            record(sim, time.perf_counter() - t)
        else:
            for k in sim:
                vals[k] = math.nan
            record(sim, 0.0, computed=False)
    return {k: vals.get(k, math.nan) for k in names}, dict(rows=rep, compile_seconds=float(t_compile), **extra)


def simulated_features(task: DeploymentTask, comp) -> dict:
    """The ``simulated`` features: the supplied-model methods run exactly on the snapshot (RO, ZNE-R, PEC per-shot
    variances, the noisy signal) and the ideal state's entropies. Exponential in width."""
    from .. import features as FT
    from . import emulate as EM
    from .run import FEATURE_PROTOCOLS

    class _Spec:
        supplied_backend = task.supplied_backend
        truth = ""
        device = task.device
    built = type("Deployment", (), dict(spec=_Spec(), compiled=comp, observable=task.obs(), ideal=None))()
    prep = EM.prepare(built, FEATURE_PROTOCOLS, seed=0, designs=1)
    s = prep.supplied
    out = {}

    def ok(r):
        return r is not None and not isinstance(r, Exception)
    ro, zr, pec, raw = s.get("RO"), s.get("ZNE-R"), s.get("PEC"), s.get("RAW")
    out["log10_v_raw_sup"] = FE._log10(ro.var1) if ok(ro) else math.nan
    out["log10_v_zne_sup"] = FE._log10(zr.var1) if ok(zr) else math.nan
    out["log10_v_pec_sup"] = FE._log10(pec.var1) if ok(pec) else math.nan
    if ok(raw):
        ctx = prep.ctx_truth["RAW"]
        run = ctx.runner.run(comp.implemented, ctx.noise, comp.measure_map)
        out["abs_signal_sup"] = float(abs(task.obs().o @ run.probs))
    else:
        out["abs_signal_sup"] = math.nan
    v, _ = FT.compute(comp, task.obs(), task.supplied_backend, None, names=["entropy_half", "output_entropy_bits"])
    out.update({k: float(v.get(k, math.nan)) for k in ("entropy_half", "output_entropy_bits")})
    return out


# ====================================================================== structural feasibility (no simulation)
def structural_feasibility(task: DeploymentTask, comp, key: str, B: int) -> tuple:
    """(feasible, reason) at the structural preflight level (``qemx.selection.preflight.structural``). 1.8.0: a
    learning protocol whose training design has not been constructed is *unverified* — returned as (False, reason),
    never as feasible (1.7.0 returned True here; review finding F1)."""
    from . import preflight as PF
    r = PF.structural(comp, key, B)
    return r.status == PF.VERIFIED, r.reason


def design_seed(task: DeploymentTask, key: str) -> int:
    """The method seed of a deployment's training design for protocol ``key``: a stable digest of the experiment
    (without its budget, so one experiment keeps its design across budgets, as the study's pools do) and the protocol
    (31 bits; never Python's randomized hash). Execution with this seed replays the verified design."""
    import dataclasses
    exp = dataclasses.replace(task, budget=1).key()      # the experiment without its budget: one design for every B
    h = hashlib.sha256(("qemx.design/1|%s|%s" % (exp, key)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


# ====================================================================== model serialization (no pickle)
def _ser_pipeline(est) -> tuple:
    """(meta, arrays) of a fitted estimator: Pipeline(StandardScaler, Ridge|ElasticNet|MLP), HGB, or the structured
    model (two ridge pipelines)."""
    from sklearn.pipeline import Pipeline
    from .models import StructuredRisk
    if isinstance(est, StructuredRisk):
        m1, a1 = _ser_pipeline(est.m1)
        m2, a2 = _ser_pipeline(est.m2)
        return (dict(kind="struct", f=est.f, budget_col=est.bc, eps0=est.eps0, b2=m1, v=m2),
                {**{"b2_" + k: v for k, v in a1.items()}, **{"v_" + k: v for k, v in a2.items()}})
    if isinstance(est, Pipeline):
        sc, reg = est.steps[0][1], est.steps[-1][1]
        arrays = dict(mean=np.asarray(sc.mean_, float), scale=np.asarray(sc.scale_, float))
        name = type(reg).__name__
        if name in ("Ridge", "ElasticNet"):
            arrays.update(coef=np.asarray(reg.coef_, float).ravel(), intercept=np.atleast_1d(float(reg.intercept_)))
            return dict(kind="linear", estimator=name), arrays
        if name == "MLPRegressor":
            if reg.activation != "relu" or reg.out_activation_ != "identity":
                raise ValueError("unsupported MLP activation")
            for i, (W, b) in enumerate(zip(reg.coefs_, reg.intercepts_)):
                arrays["W%d" % i], arrays["b%d" % i] = np.asarray(W, float), np.asarray(b, float)
            return dict(kind="mlp", layers=len(reg.coefs_)), arrays
        raise ValueError("unsupported estimator %s" % name)
    if type(est).__name__ == "HistGradientBoostingRegressor":
        trees = [p[0].nodes for p in est._predictors]
        offs = np.cumsum([0] + [len(t) for t in trees])
        nodes = np.concatenate(trees) if trees else np.zeros(0)
        cols = {k: np.asarray(nodes[k]) for k in ("value", "feature_idx", "num_threshold", "missing_go_to_left",
                                                  "left", "right", "is_leaf")}
        return (dict(kind="hgb", trees=len(trees), baseline=float(np.ravel(est._baseline_prediction)[0])),
                {**{"node_" + k: v for k, v in cols.items()}, "offsets": offs.astype(np.int64)})
    raise ValueError("unsupported estimator %s" % type(est).__name__)


def _std(X, a, prefix=""):
    sc = a[prefix + "scale"]
    return (np.asarray(X, float) - a[prefix + "mean"]) / np.where(sc == 0, 1.0, sc)


def _predict_ser(meta, a, X, prefix="") -> np.ndarray:
    X = np.asarray(X, float)
    k = meta["kind"]
    if k == "linear":
        return _std(X, a, prefix) @ a[prefix + "coef"] + a[prefix + "intercept"][0]
    if k == "mlp":
        h = _std(X, a, prefix)
        for i in range(meta["layers"]):
            h = h @ a[prefix + "W%d" % i] + a[prefix + "b%d" % i]
            if i < meta["layers"] - 1:
                h = np.maximum(h, 0.0)
        return h.ravel()
    if k == "hgb":
        out = np.full(len(X), meta["baseline"])
        offs = a[prefix + "offsets"]
        val, fi, th = a[prefix + "node_value"], a[prefix + "node_feature_idx"], a[prefix + "node_num_threshold"]
        mgl, lt, rt, leaf = (a[prefix + "node_missing_go_to_left"], a[prefix + "node_left"], a[prefix + "node_right"],
                             a[prefix + "node_is_leaf"])
        for t in range(meta["trees"]):
            o = int(offs[t])
            for r in range(len(X)):
                node = 0
                while not leaf[o + node]:
                    x = X[r, fi[o + node]]
                    if np.isnan(x):
                        node = int(lt[o + node] if mgl[o + node] else rt[o + node])
                    else:
                        node = int(lt[o + node] if x <= th[o + node] else rt[o + node])
                out[r] += val[o + node]
        return out
    if k == "struct":
        Xn = np.delete(X, meta["budget_col"], axis=1)
        Be = (1.0 - meta["f"]) * 10.0 ** X[:, meta["budget_col"]]
        b2 = 10.0 ** _predict_ser(meta["b2"], a, Xn, prefix + "b2_")
        v = 10.0 ** _predict_ser(meta["v"], a, Xn, prefix + "v_")
        from .models import STRUCT_FLOOR
        return np.log10(np.maximum(b2 - STRUCT_FLOOR, 0.0) + np.maximum(v - STRUCT_FLOOR, 0.0) / Be + meta["eps0"])
    raise ValueError(k)


# ====================================================================== the bundle
@dataclass
class ModelBundle:
    meta: dict
    arrays: dict = field(default_factory=dict)
    path: str = ""

    @property
    def features(self) -> list:
        return list(self.meta["features"])

    @property
    def protocols(self) -> list:
        return list(self.meta["protocols"])

    def predict(self, X) -> np.ndarray:
        """(rows, protocols) predicted log10(R + ε0) (NaN where a protocol has no model)."""
        X = np.asarray(X, float).copy()
        med = np.asarray(self.meta["impute"], float)
        bad = ~np.isfinite(X)
        X[bad] = np.take(med, np.nonzero(bad)[1])
        out = np.full((len(X), len(self.protocols)), np.nan)
        for j, p in enumerate(self.protocols):
            m = self.meta["models"].get(p)
            if m is not None:
                out[:, j] = _predict_ser(m, self.arrays, X, prefix="%s__" % p)
        return out


def build_bundle(ds, train_idx, family: str, set_name: str, out_dir: str, *, version: str, tau: float | None,
                 validation: dict | None = None, training: dict | None = None) -> ModelBundle:
    """Fit the configuration on ``train_idx`` and write a bundle to ``out_dir`` (bundle.json + arrays.npz). The
    serialized models are evaluated by this module's NumPy code and compared with scikit-learn's predictions on the
    training rows (the largest difference is stored; the build fails above 1e-9)."""
    from . import policies as PO
    from .claims import KINDS
    train_idx = np.asarray(train_idx)
    names = FE.SETS[set_name]
    pol = PO.Model(family, names, set_name).fit(ds, train_idx)
    fb = PO.SBSBudget().fit(ds, train_idx)
    rm = pol.rm
    models, arrays = {}, {}
    for p, est in rm.models.items():
        meta, arr = _ser_pipeline(est)
        models[p] = meta
        arrays.update({"%s__%s" % (p, k): v for k, v in arr.items()})
    X = ds.X(names)[train_idx]
    support = {}
    for j, n in enumerate(names):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            support[n] = dict(min=float(col.min()), max=float(col.max()), q01=float(np.quantile(col, 0.01)),
                              q99=float(np.quantile(col, 0.99)))
    rows = [ds.rows[i] for i in train_idx]
    meta = dict(schema=BUNDLE_SCHEMA, version=version, created=time.strftime("%Y-%m-%d"),
                configuration="MODEL[%s|%s]" % (family, set_name), family=family, feature_set=set_name,
                features=list(names), feature_classes={n: FE.BY_NAME[n].info for n in names},
                protocols=list(ds.protocols),
                protocol_definitions={k: PR.PORTFOLIO[k].to_json() for k in ds.protocols},
                portfolio_digest=hashlib.sha256(json.dumps([PR.PORTFOLIO[k].to_json() for k in ds.protocols],
                                                           sort_keys=True).encode()).hexdigest()[:16],
                eps0=float(rm.eps0), impute=[float(x) for x in rm.impute], models=models,
                chosen=rm.chosen, cv_rmse={k: float(v) for k, v in rm.cv_rmse.items()},
                selective=dict(rule="accept the model's choice when its margin z = (P2 − P1)/sqrt(s1² + s2²) ≥ τ, "
                                    "else SBS-B", tau=None if tau is None else float(tau)),
                fallback=dict(policy="SBS-B", ranks={str(b): [ds.protocols[j] for j in r] for b, r in fb.rank.items()},
                              global_rank=[ds.protocols[j] for j in fb.global_rank],
                              rule="first feasible protocol in the training ranking of the nearest budget (log scale)"),
                support=dict(features=support, budgets=[int(min(r["B"] for r in rows)), int(max(r["B"] for r in rows))],
                             families=sorted({r["family"] for r in rows}), widths=sorted({int(r["n"]) for r in rows}),
                             devices=sorted({r["device"] for r in rows}), base_circuits=len({r["base_id"] for r in rows}),
                             rows=len(rows)),
                validation=validation or {}, training=training or {},
                feature_fingerprint=feature_fingerprint(names), policy=POLICY,
                snapshots={d: _snapshot(d) for d in sorted({r["device"] for r in rows})},
                guarantees={"predicted log risks": "model-conditional", "margin and selective rule": "model-conditional",
                            "nested-CV performance": "empirical", "training-support check": "ood-warning"},
                statement_kinds={k: KINDS[k] for k in ("model-conditional", "empirical", "ood-warning")},
                capabilities=dict(task_class="fake-device calibration snapshots (exact engine); diagonal observables "
                                             "with |O| ≤ 1; product-state inputs; circuits the study's compiler maps",
                                  cheap_path=all(FE.BY_NAME[n].info == "available" for n in names)))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "arrays.npz"), **arrays)
    b = ModelBundle(meta, {k: np.asarray(v) for k, v in arrays.items()}, out_dir)
    sk = rm.predict(X)
    mine = b.predict(X)
    ok = np.isfinite(sk) & np.isfinite(mine)
    diff = float(np.max(np.abs(sk[ok] - mine[ok]))) if ok.any() else 0.0
    if diff > 1e-9:
        raise AssertionError("the serialized models differ from scikit-learn by %.3g" % diff)
    meta["verification"] = dict(max_abs_difference_vs_sklearn=diff, rows=int(len(X)))
    meta["arrays_sha256"] = _sha(os.path.join(out_dir, "arrays.npz"))
    TX.write_json(os.path.join(out_dir, "bundle.json"), meta, indent=1, sort_keys=True)
    return b


def _snapshot(device: str) -> str:
    from .. import device as DV
    return DV.snapshot_fingerprint(device)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class BundleIncompatible(ValueError):
    """A bundle whose scientific definitions (candidates, features) differ from this code's (1.8.0; finding F4)."""


def feature_fingerprint(names) -> str:
    """Digest of the definitions (name, information class, group, definition text) of the features ``names``, in
    order. A changed definition, class or order changes it."""
    rows = [[n, FE.BY_NAME[n].info, FE.BY_NAME[n].group, FE.BY_NAME[n].definition] for n in names]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# Bundles built before 1.8.0 do not store the fingerprints. These are the values of the 1.7.0 code that built the two
# shipped bundles (computed by 1.8.0 from that code's definitions and its release lock's fake-device snapshots, whose
# definitions and snapshots 1.8.0 keeps unchanged; tests/test_preflight.py rebuilds them). A future change of a
# definition then refuses the old bundles instead of silently reinterpreting their features.
LEGACY_FINGERPRINTS = {
    "features": {"qemx-rec-1.7.0": "efc3c92fc7f82098", "qemx-rec-1.7.0-deployable": "2d5017917d8e1398"},
    "snapshots": {"fake_fez": "73950e082405bc1f", "fake_torino": "e8c52e86d4be590d", "fake_kyiv": "c17d476dfb3ae6a2"},
}


def compatibility(meta: dict) -> dict:
    """Scientific compatibility of a bundle's metadata with this code: ``errors`` (refuse to load: a candidate's
    definition or a feature's definition differs), ``transfer`` (load, but every recommendation is
    out-of-distribution: a training device's calibration snapshot differs under the same name) and the checks done."""
    from .. import device as DV
    errors, transfer, checks = [], [], []
    for k in meta.get("protocols", []):
        mine = PR.PORTFOLIO.get(k)
        theirs = meta.get("protocol_definitions", {}).get(k)
        if mine is None:
            errors.append("candidate %s is not defined in this version" % k)
        elif theirs != json.loads(json.dumps(mine.to_json())):
            errors.append("candidate %s is defined differently here (bundle %s; this code %s)"
                          % (k, json.dumps(theirs, sort_keys=True), json.dumps(mine.to_json(), sort_keys=True)))
    checks.append("candidate definitions (%d)" % len(meta.get("protocols", [])))
    names = list(meta.get("features", []))
    missing = [n for n in names if n not in FE.BY_NAME]
    if missing:
        errors.append("features unknown to this version: %s" % ", ".join(missing))
    else:
        bad = [n for n in names if meta.get("feature_classes", {}).get(n, FE.BY_NAME[n].info) != FE.BY_NAME[n].info]
        if bad:
            errors.append("features with another information class here: %s" % ", ".join(bad))
        want = meta.get("feature_fingerprint") or LEGACY_FINGERPRINTS["features"].get(meta.get("version"), "")
        if want and want != feature_fingerprint(names):
            errors.append("feature definitions differ from the bundle's (fingerprint %s ≠ %s)"
                          % (feature_fingerprint(names), want))
        checks.append("feature definitions (%s)" % ("fingerprint" if want else "names and classes only"))
    snaps = meta.get("snapshots") or {d: LEGACY_FINGERPRINTS["snapshots"].get(DV.base_name(d))
                                      for d in meta.get("support", {}).get("devices", [])}
    for d, fp in snaps.items():
        if not fp:
            transfer.append("calibration snapshot of %s not recorded by the bundle" % d)
            continue
        try:
            now = DV.snapshot_fingerprint(d)
        except Exception as e:                          # noqa: BLE001 - a device this installation does not have
            transfer.append("device %s unavailable here (%s)" % (d, str(e)[:80]))
            continue
        if now != fp:
            transfer.append("calibration snapshot of %s differs from training (%s ≠ %s): a new calibration is a "
                            "transfer, evaluated separately" % (d, now, fp))
    checks.append("calibration snapshots (%d)" % len(snaps))
    return dict(errors=errors, transfer=transfer, checks=checks)


def load_bundle(path: str, *, check: bool = True) -> ModelBundle:
    """A bundle directory (bundle.json + arrays.npz): its digest is checked, the arrays are loaded without pickle, and
    (1.8.0) its scientific compatibility with this code is checked (``compatibility``): different candidate or
    feature definitions raise BundleIncompatible; a changed calibration snapshot is recorded in
    ``meta['compatibility']`` and makes recommendations out-of-distribution."""
    meta = TX.read_json(os.path.join(path, "bundle.json"))
    if meta.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("%s is not a model bundle (%s)" % (path, meta.get("schema")))
    ap = os.path.join(path, "arrays.npz")
    if _sha(ap) != meta.get("arrays_sha256"):
        raise ValueError("%s: arrays.npz does not match its digest in bundle.json" % path)
    if check:
        comp = compatibility(meta)
        if comp["errors"]:
            raise BundleIncompatible("%s is not compatible with this version: %s" % (path, "; ".join(comp["errors"])))
        meta["compatibility"] = comp
    with np.load(ap, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return ModelBundle(meta, arrays, path)


def bundles() -> list:
    """The bundles shipped with the application (newest first)."""
    if not os.path.isdir(BUNDLE_DIR):
        return []
    return sorted([d for d in os.listdir(BUNDLE_DIR) if os.path.exists(os.path.join(BUNDLE_DIR, d, "bundle.json"))],
                  reverse=True)


# ====================================================================== recommendation
def recommend(task: DeploymentTask, bundle: ModelBundle, *, allow_simulation: bool = False,
              preflight_level: str | None = None, design_seeds: dict | None = None) -> dict:
    """The recommendation for one deployment task (plain data; the UI and the CLI show the same dictionary).

    Policy ``POLICY`` (1.8.0), every step part of what is evaluated:
    guard (bounded cost, before compilation) → compile → features → out-of-distribution checks → structural preflight
    of every candidate → predicted log risks → walk the candidates from the lowest predicted risk: a candidate the
    preflight left *unverified* is resolved at ``preflight_level`` (default ``design``: a learning protocol's
    training design is constructed with its deployment seed and its ideal training values computed; ``allocation``
    with ``allow_simulation``: the study's own rule on the supplied model), an infeasible one is skipped (recorded) →
    the first verified candidate is the model's choice; the margin to the next candidate not known to be infeasible
    → frozen selective rule (τ) → the result, or SBS-B's first *verified* candidate as the fallback (OOD, margin
    below τ, no prediction). Unverified candidates are never accepted. ``design_seeds``: per-protocol design seeds
    (default ``design_seed(task, key)``; a study replay passes the study's allocation seed)."""
    from .. import device as DV
    from . import preflight as PF
    t0 = time.perf_counter()
    level = preflight_level or ("allocation" if allow_simulation else "design")
    if level not in PF.LEVELS:
        raise ValueError("preflight_level must be one of %s" % ", ".join(PF.LEVELS))
    out = dict(task=task.to_json(), bundle=bundle.meta["version"], configuration=bundle.meta["configuration"],
               policy=POLICY, preflight_level=level, status="ok", warnings=[],
               statement_kinds=bundle.meta.get("guarantees", {}))
    notes = guard(task, bundle)                      # raises Unsupported before any compilation
    out["guard"] = notes
    names = bundle.features
    needs_sim = [n for n in names if FE.BY_NAME[n].info == "simulated"]
    if needs_sim and not allow_simulation:
        out["warnings"].append("the model uses simulated features (%s); they were not computed (enable simulation to "
                               "compute them; cost grows exponentially with width)" % ", ".join(needs_sim))
    try:
        comp = task.compile()
    except Exception as e:                           # noqa: BLE001
        raise Unsupported("the circuit does not compile for %s: %s" % (task.device, str(e)[:200])) from None
    out["effective_device"] = task.device
    out["requested_layout"] = list(task.initial_layout) if task.initial_layout else None
    out["physical_layout"] = [int(q) for q in (comp.info.get("phys") or [])]
    out["symmetry"] = dict(declared=None if not task.symmetry else _sym_text(task.symmetry),
                           source=task.symmetry_source or None)
    out["conversion_notes"] = list(task.notes)
    out["simulation"] = dict(
        target_state_features=("computed (exact simulation of the noisy snapshot model of this circuit)"
                               if allow_simulation and needs_sim else "not computed"),
        design_preflight="learning candidates reached in the model's order: their training circuits' ideal values "
                         "by state vector (classical, no shots; not a simulation of this circuit's noisy state)",
        symmetry_check=("the declared symmetry was checked on this circuit's ideal measured distribution"
                        if task.symmetry_source == "state, validated" else "not run"))
    out["signature"] = recommendation_signature(task, bundle, allow_simulation=allow_simulation,
                                                preflight_level=preflight_level)
    vals, rep = features(task, names, allow_simulation=allow_simulation, compiled=comp)
    out["features"] = vals
    out["feature_report"] = rep
    sup = bundle.meta["support"]
    ood = []
    base = DV.base_name(task.device)
    if base not in {DV.base_name(d) for d in sup["devices"]}:
        ood.append("device %s was not in the training data (%s)" % (base, ", ".join(sup["devices"])))
    for t in (bundle.meta.get("compatibility") or {}).get("transfer", []):
        if base in t:
            ood.append(t)
    if not (sup["budgets"][0] <= int(task.budget) <= sup["budgets"][1]):
        ood.append("budget %s outside the training range %s–%s" % (f"{task.budget:,}", f"{sup['budgets'][0]:,}",
                                                                    f"{sup['budgets'][1]:,}"))
    if int(task.n) not in sup["widths"]:
        ood.append("width %d outside the training widths %s" % (task.n, sup["widths"]))
    for n_, v in vals.items():
        s = sup["features"].get(n_)
        if s is not None and np.isfinite(v):
            span = max(s["max"] - s["min"], 1e-12)
            if v < s["min"] - 0.05 * span or v > s["max"] + 0.05 * span:
                ood.append("feature %s = %.4g outside the training range [%.4g, %.4g]" % (n_, v, s["min"], s["max"]))
    missing = [n for n, v in vals.items() if not np.isfinite(v)]
    if missing:
        ood.append("features not computed (imputed with training medians): %s" % ", ".join(missing))
    out["ood"] = ood
    # ---- preflight: structural for every candidate; resolved lazily below
    seeds = {k: int((design_seeds or {}).get(k, design_seed(task, k))) for k in bundle.protocols}
    pf = {k: PF.structural(comp, k, task.budget, symmetry=task.symmetry) for k in bundle.protocols}
    supplied_cache = {}

    def resolve(k):
        r = pf[k]
        if r.status != PF.UNVERIFIED or level == "structural":
            return r
        if level == "design" or task.n > SIMULATION_MAX_WIDTH:
            # the allocation level simulates the supplied model (dense, 4^n): above SIMULATION_MAX_WIDTH it is not
            # run and the candidate is resolved at the design level instead (1.8.0, independent review)
            r = PF.design(comp, task.obs() if task.n <= DENSE_OBS_MAX_WIDTH else None, k, task.budget,
                          supplied=task.supplied_backend, seed=seeds[k], symmetry=task.symmetry)
            if level == "allocation":
                r = dataclasses.replace(r, reason=(r.reason + "; " if r.reason else "") +
                                        "allocation level not run above %d qubits (supplied-model simulation)"
                                        % SIMULATION_MAX_WIDTH)
        else:
            if k not in supplied_cache:
                supplied_cache[k] = _supplied_result(task, comp, k, seeds[k])
            r = PF.allocation(supplied_cache[k], k, task.budget)
        pf[k] = r
        return r
    x = np.array([[vals[n] for n in names]], float)
    P = bundle.predict(x)[0]
    pred = {p: float(P[j]) for j, p in enumerate(bundle.protocols) if np.isfinite(P[j])}
    out["predicted_log10_risk"] = pred
    order = sorted(pred, key=lambda p: pred[p])
    skipped, best = [], None
    for k in order:
        r = resolve(k)
        if r.status == PF.VERIFIED:
            best = k
            break
        skipped.append(dict(protocol=k, status=r.status, reason=r.reason))
    # the runner-up of the margin: the next candidate in the model's order that is verified at the same level — the
    # deployment counterpart of the label-feasible runner-up the selective threshold τ was calibrated with
    # (policies.SelectiveMargin). Resolved before the fallback, so the margin never depends on which candidates the
    # fallback's walk happened to resolve (1.8.0, independent review).
    runner = None
    if best is not None:
        for k in order[order.index(best) + 1:]:
            if resolve(k).status == PF.VERIFIED:
                runner = k
                break
    fallback = None
    for k in _fallback_rank(bundle, task.budget):
        if k in pf and resolve(k).status == PF.VERIFIED:
            fallback = k
            break
    if fallback is None:
        fallback = "RAW"
        out["warnings"].append("no SBS-B candidate is verified feasible at this budget; RAW is used")
    out["fallback"] = fallback
    out["skipped_candidates"] = skipped
    out["preflight"] = {k: r.to_json() for k, r in pf.items()}
    out["feasibility"] = {k: dict(feasible=r.status == PF.VERIFIED, status=r.status, level=r.level, note=r.reason)
                          for k, r in pf.items()}
    out["feasibility_method"] = ("preflight contract (qemx.selection.preflight): structural for every candidate; "
                                 "candidates reached in the model's order resolved at level %r" % level)
    out["design_seeds"] = {k: seeds[k] for k in bundle.protocols if PR.PORTFOLIO[k].learning}
    if best is None:
        out.update(status="no-model", choice=fallback, accepted=False,
                   reason="no verified-feasible protocol has a model prediction; the fallback is used")
        return _finish_choice(out, bundle, t0)
    if runner is not None:
        s1 = bundle.meta["cv_rmse"].get(best, math.nan)
        s2 = bundle.meta["cv_rmse"].get(runner, math.nan)
        margin = float((pred[runner] - pred[best]) / math.sqrt(s1 ** 2 + s2 ** 2))
    else:
        margin = math.inf
    tau = bundle.meta["selective"]["tau"]
    out.update(model_choice=best, runner_up=runner, margin=margin, tau=tau)
    if ood:
        out.update(status="out-of-distribution", choice=fallback, accepted=False,
                   reason="outside the training support: the model's prediction is not used (fallback SBS-B)")
    elif tau is not None and not (margin >= tau):
        out.update(choice=fallback, accepted=False,
                   reason="margin %.2f below the frozen threshold τ = %.2f: fallback SBS-B" % (margin, tau))
    else:
        out.update(choice=best, accepted=True, reason="the model's choice (margin %.2f ≥ τ)" % margin
                   if tau is not None else "the model's choice (no selective threshold in this bundle)")
    return _finish_choice(out, bundle, t0)


def _supplied_result(task, comp, key, seed):
    """The supplied model's method result of one protocol (the study's allocation design), for level 'allocation'."""
    from . import emulate as EM

    class _Spec:
        supplied_backend = task.supplied_backend
        truth = ""
        device = task.device
        symmetry = task.symmetry
        task_id = task.key()
        base_id = task.key()
    built = type("Deployment", (), dict(spec=_Spec(), compiled=comp, observable=task.obs(), ideal=None))()
    prep = EM.prepare(built, [key], seed=int(seed), designs=1)
    return prep.supplied.get(key)


def _finish_choice(out, bundle, t0):
    k = out["choice"]
    out["protocol_definition"] = bundle.meta["protocol_definitions"].get(k, {})
    out["execution"] = dict(protocol=k, preflight=out["preflight"].get(k, {}).get("status"),
                            design_seed=out.get("design_seeds", {}).get(k),
                            note=("run this protocol with method seed %d to execute the verified training design"
                                  % out["design_seeds"][k]) if k in out.get("design_seeds", {}) else "")
    return _finish(out, t0)


def _fallback_rank(bundle, B) -> list:
    ranks = bundle.meta["fallback"]["ranks"]
    if not ranks:
        return list(bundle.meta["fallback"]["global_rank"])
    bs = np.array(sorted(int(b) for b in ranks))
    b = int(bs[np.argmin(np.abs(np.log10(bs) - math.log10(max(int(B), 1))))])
    return list(ranks[str(b)])


def _finish(out, t0):
    out["seconds"] = float(time.perf_counter() - t0)
    return out
