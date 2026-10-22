r"""Simulated IBM devices: Qiskit fake backends as a third kind of simulator.

A fake backend carries a calibration snapshot of a real processor: its coupling map, native gates
(CZ on Heron, ECR on Eagle, CX on Falcon), gate durations, gate errors, T1/T2 and readout errors.
Selecting one changes two things.

1. **Compilation** targets the device: the logical circuit is transpiled with the backend's
   target (layout on the real coupling map, native gates, SWAPs where needed) and then *compacted*
   to the physical qubits it uses, so the density matrix stays 2^n × 2^n for n active qubits.

2. **Execution** replaces the NoiseSpec model with the device model on those qubits:

   * every native gate is followed by the error Qiskit Aer derives from the calibration
     (thermal relaxation over the gate's duration plus depolarizing to match the reported error;
     ``basic_device_gate_errors``),
   * the circuit is scheduled as late as possible with the real durations and every idle window
     becomes thermal relaxation exp(−t/T1), exp(−t/T2) (so dynamical decoupling has something to fix),
   * readout is each qubit's own assignment matrix, P(1|0) ≠ P(0|1).

Pauli twirling is applied exactly: each gate error is replaced by its Pauli-twirled channel
(the diagonal of its Pauli transfer matrix). Dynamical decoupling places X–X or X–Y–X–Y pulses in the
scheduled idle windows. PEC corrections are applied as ideal Paulis. Noise channels the user placed
in the circuit are applied as written.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .circuit import CircuitSpec, Op, layer_order
from .resources import check
from .resources import qiskit_serial

# name -> (class in qiskit_ibm_runtime.fake_provider, description)
FAKE_DEVICES = {
    "fake_fez": ("FakeFez", "156 qubits · Heron r2 · CZ"),
    "fake_torino": ("FakeTorino", "133 qubits · Heron r1 · CZ"),
    "fake_marrakesh": ("FakeMarrakesh", "156 qubits · Heron r2 · CZ"),
    "fake_kingston": ("FakeKingston", "156 qubits · Heron r2 · CZ"),
    "fake_aachen": ("FakeAachen", "156 qubits · Heron r2 · CZ"),
    "fake_brisbane": ("FakeBrisbane", "127 qubits · Eagle r3 · ECR"),
    "fake_sherbrooke": ("FakeSherbrooke", "127 qubits · Eagle r3 · ECR"),
    "fake_kyiv": ("FakeKyiv", "127 qubits · Eagle r3 · ECR"),
    "fake_osaka": ("FakeOsaka", "127 qubits · Eagle r3 · ECR"),
    "fake_prague": ("FakePrague", "33 qubits · CZ"),
    "fake_algiers": ("FakeAlgiers", "27 qubits · Falcon · CX"),
    "fake_perth": ("FakePerth", "7 qubits · Falcon · CX"),
}
_LOCK = threading.RLock()


# ------------------------------------------------------------------ device specifications
# A device backend is named by a *device specification*: a fake device, optionally followed by modifiers,
#     fake_fez                         the calibration snapshot as is (Aer density-matrix execution)
#     fake_fez@s=2                     every gate error × 2 and every relaxation rate × 2 (T1, T2 ÷ 2)
#     fake_fez@s=2,ro=0                … and readout errors × 0 (switched off)
#     fake_fez@eng=exact               the same model executed by the exact superoperator engine (device_exact.py),
#                                      which also applies signed quasi-probability maps (exact PEC means)
#     fake_fez@hom=1                   homogenized: every qubit gets the medians of the selected qubits (T1, T2,
#                                      readout) and every gate the median error and duration of its kind
#     fake_fez@hom=1,chi=0.25,eng=exact  … with a coherent fraction χ of each gate's non-relaxation error turned into a
#                                      unitary over-rotation (one-qubit pulses) or ZZ phase (two-qubit gates), the
#                                      total gate infidelity unchanged (exact engine only; see device_exact.py)
# The compiled circuit (layout, routing, native gates) depends only on the base device, never on the modifiers, so
# a noise sweep runs the identical circuit at every scale.
_SPEC = re.compile(r"^([a-z][a-z0-9_]{0,40})(?:@([a-z][a-z0-9]{0,3}=[A-Za-z0-9.+\-]{1,24}(?:,[a-z][a-z0-9]{0,3}=[A-Za-z0-9.+\-]{1,24}){0,9}))?$")
DEVICE_MODS = {"s": 1.0, "ro": 1.0, "eng": "aer", "hom": 0.0, "chi": 0.0, "g1": 1.0, "g2": 1.0, "t": 1.0}
# Component scales (1.7.0, noise-composition study): g1 multiplies the reported one-qubit gate errors, g2 the
# two-qubit gate errors, t the relaxation rates (T1 → T1/t, T2 → T2/t); they combine multiplicatively with s, which
# scales all three together (s = 2 ≡ g1 = g2 = t = 2). ro scales readout, chi makes part of each gate's error coherent.
ENGINES = ("aer", "exact")


# Calibration snapshots (1.5.1): a DeviceModel built from any backend's calibration (``model_for``, e.g. a real
# device's calibration of the day) registered under a simulator name "cal:<label>", so that ``compare``,
# ``simulate`` and the methods run on exactly that model through the same device path as the fake devices.
_REGISTERED: dict = {}
_CAL_NAME = re.compile(r"^cal:[A-Za-z0-9_.\-]{1,80}$")


def register_model(dm: "DeviceModel", label: str) -> str:
    """The simulator name ("cal:<label>") of a calibration snapshot; registering the same label again replaces it,
    so a label should identify the calibration (e.g. backend name and ``transfer.calibration_hash``)."""
    import dataclasses as _dc
    name = "cal:" + re.sub(r"[^A-Za-z0-9_.\-]", "_", str(label))[:80]
    _REGISTERED[name] = _dc.replace(dm, name=name, noise={}, relax_cache={})   # its own caches, under its own name
    return name


def parse_device(name: str) -> tuple:
    """(base device, modifiers with defaults filled in). Raises ValueError for anything malformed: unknown keys,
    non-finite or out-of-range values, an unknown engine. The base name is not checked here (see is_device)."""
    if isinstance(name, str) and name in _REGISTERED:      # a calibration snapshot: the model as measured
        return name, dict(DEVICE_MODS)
    m = _SPEC.match(name or "") if isinstance(name, str) else None
    if not m:
        raise ValueError("not a device specification: %r" % (str(name)[:60],))
    mods = dict(DEVICE_MODS)
    seen = set()
    for item in (m.group(2).split(",") if m.group(2) else ()):
        k, v = item.split("=", 1)
        if k not in DEVICE_MODS:
            raise ValueError("unknown device modifier %r (allowed: %s)" % (k, ", ".join(DEVICE_MODS)))
        if k in seen:
            raise ValueError("device modifier %r given twice" % k)
        seen.add(k)
        if k == "eng":
            if v not in ENGINES:
                raise ValueError("unknown engine %r (allowed: %s)" % (v, ", ".join(ENGINES)))
            mods[k] = v
            continue
        x = float(v)
        if not np.isfinite(x) or x < 0 or (k in ("s", "g1", "g2", "t") and x <= 0):
            raise ValueError("device modifier %s=%r is out of range (s, g1, g2, t > 0; ro ≥ 0)" % (k, v))
        if k == "hom" and x not in (0.0, 1.0):
            raise ValueError("hom must be 0 or 1")
        if k == "chi" and x > 1.0:
            raise ValueError("chi (coherent fraction) must be in [0, 1]")
        mods[k] = x
    if mods["chi"] > 0 and mods["eng"] != "exact":
        raise ValueError("a coherent fraction (chi > 0) needs the exact engine: add eng=exact")
    return m.group(1), mods


def base_name(name: str) -> str:
    """The fake device a specification refers to (``fake_fez@s=2`` -> ``fake_fez``)."""
    try:
        return parse_device(name)[0]
    except ValueError:
        # a well-formed specification whose modifiers need another engine (``chi`` without ``eng=exact``) still
        # names its base device (1.7.0: a coherent-fraction task could not find its device)
        m = _SPEC.match(name or "") if isinstance(name, str) else None
        return m.group(1) if m else name


def device_spec(base: str, s: float = 1.0, ro: float = 1.0, eng: str = "aer", hom: bool = False,
                chi: float = 0.0) -> str:
    """Canonical specification string (defaults omitted, numbers in shortest round-trip form)."""
    if base not in FAKE_DEVICES:
        raise ValueError("unknown fake device %r" % base)
    mods = []
    if hom:
        mods.append("hom=1")
    if float(chi) != 0.0:
        mods.append("chi=%r" % float(chi))
    if float(s) != 1.0:
        mods.append("s=%r" % float(s))
    if float(ro) != 1.0:
        mods.append("ro=%r" % float(ro))
    if eng != "aer":
        mods.append("eng=%s" % eng)
    spec = base + ("@" + ",".join(mods) if mods else "")
    parse_device(spec)                                     # validates the values
    return spec


def is_device(backend: str | None) -> bool:
    if not backend or not isinstance(backend, str):
        return False
    if backend in _REGISTERED:
        return True
    m = _SPEC.match(backend)
    return bool(m) and m.group(1) in FAKE_DEVICES


def supports_negative(backend: str | None) -> bool:
    """Whether this backend applies signed (non-CP) quasi-probability maps exactly: numpy and eng=exact devices."""
    if backend == "numpy":
        return True
    return is_device(backend) and parse_device(backend)[1]["eng"] == "exact"


def device_label(backend: str) -> str:
    base, mods = parse_device(backend)
    cls, desc = FAKE_DEVICES[base]
    extra = []
    if mods["s"] != 1.0:
        extra.append("errors × %g" % mods["s"])
    for k, what in (("g1", "1q gate errors"), ("g2", "2q gate errors"), ("t", "relaxation rates")):
        if mods.get(k, 1.0) != 1.0:
            extra.append("%s × %g" % (what, mods[k]))
    if mods["ro"] != 1.0:
        extra.append("readout × %g" % mods["ro"])
    if mods["hom"]:
        extra.append("homogenized")
    if mods["chi"]:
        extra.append("coherent fraction %g" % mods["chi"])
    if mods["eng"] != "aer":
        extra.append("exact engine")
    return "%s (%s)%s" % (cls, desc, (" · " + ", ".join(extra)) if extra else "")


def fake_class(backend: str) -> str:
    """The qiskit_ibm_runtime class name of a device specification's base device."""
    return FAKE_DEVICES[base_name(backend)][0]


@lru_cache(maxsize=16)
@qiskit_serial
def fake_backend(name: str):
    import warnings
    import qiskit_ibm_runtime.fake_provider as fp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(fp, FAKE_DEVICES[base_name(name)][0])()


# ------------------------------------------------------------------ device properties
@dataclass
class QubitInfo:
    t1: float
    t2: float
    p01: float                 # P(read 1 | prepared 0)
    p10: float                 # P(read 0 | prepared 1)


@dataclass
class DeviceModel:
    name: str
    phys: tuple                          # physical qubit of compact index i
    dt: float
    qubits: list                         # QubitInfo per compact qubit
    two_qubit: str                       # cz | ecr | cx
    durations: dict                      # (name, compact qargs) -> seconds
    errors: dict                         # (name, compact qargs) -> reported error
    noise: dict = field(default_factory=dict)       # twirl mode -> NoiseModel
    relax_cache: dict = field(default_factory=dict)
    coherent: float = 0.0                # χ: coherent fraction of each gate's non-relaxation error (exact engine)

    @property
    def n(self) -> int:
        return len(self.phys)

    def readout_matrices(self) -> list:
        return [np.array([[1 - q.p01, q.p10], [q.p01, 1 - q.p10]]) for q in self.qubits]

    def gate_error(self, name: str, qargs: tuple) -> float | None:
        return self.errors.get((name, tuple(qargs)))

    def median_errors(self) -> dict:
        e1 = [v for (nm, qa), v in self.errors.items() if len(qa) == 1 and nm in ("sx", "x") and v is not None]
        e2 = [v for (nm, qa), v in self.errors.items() if len(qa) == 2 and v is not None]
        ro = [(q.p01 + q.p10) / 2 for q in self.qubits]
        return dict(r1=float(np.median(e1)) if e1 else 0.0, r2=float(np.median(e2)) if e2 else 0.0,
                    ro=float(np.mean(ro)) if ro else 0.0,
                    t1=float(np.median([q.t1 for q in self.qubits])), t2=float(np.median([q.t2 for q in self.qubits])))

    def equivalent_noise(self):
        """A NoiseSpec with the device's median rates, used where a method needs a *model* (PEC's assumed rates,
        TREX's symmetric flip probability). Depolarizing p = r·d/(d−1) from the average gate infidelity r."""
        from .noise import NoiseSpec
        m = self.median_errors()
        p01 = float(np.mean([q.p01 for q in self.qubits])) if self.qubits else 0.0
        p10 = float(np.mean([q.p10 for q in self.qubits])) if self.qubits else 0.0
        return NoiseSpec(p1=min(1.0, m["r1"] * 2), p2=min(1.0, m["r2"] * 4 / 3), ro01=p01, ro10=p10)

    def table(self) -> list:
        rows = []
        for i, (p, q) in enumerate(zip(self.phys, self.qubits)):
            sx = self.errors.get(("sx", (i,)))
            rows.append({"qubit": "q%d" % i, "physical": int(p), "T1 (µs)": q.t1 * 1e6, "T2 (µs)": q.t2 * 1e6,
                         "√X error": sx, "P(1|0)": q.p01, "P(0|1)": q.p10})
        return rows


def _props(backend):
    try:
        return backend.properties()
    except Exception:
        return None


def get_model(name: str, phys: tuple) -> DeviceModel:
    """The fake device restricted to the physical qubits `phys` (compact index i = phys[i]); a specification with
    modifiers (``fake_fez@s=2,ro=0``) gives the scaled model (see scale_model); a registered calibration snapshot
    (``register_model``) is returned as registered, for its own physical qubits only."""
    dm = _REGISTERED.get(name) if isinstance(name, str) else None
    if dm is not None:
        if tuple(int(p) for p in phys) != tuple(int(p) for p in dm.phys):
            raise ValueError("calibration snapshot %s covers physical qubits %s, not %s" % (name, list(dm.phys),
                                                                                          list(phys)))
        return dm
    return _get_model(name, tuple(phys))


get_model.cache_clear = lambda: _get_model.cache_clear()     # noqa: E731 - the cache of the fake-device models


@lru_cache(maxsize=256)
@qiskit_serial
def _get_model(name: str, phys: tuple) -> DeviceModel:
    base, mods = parse_device(name)
    if base not in FAKE_DEVICES:
        raise ValueError("unknown fake device %r" % base)
    plain = model_for(fake_backend(base), phys, base)
    if mods["hom"]:
        plain = homogenize(plain, name)
    if any(mods[k] != 1.0 for k in ("s", "ro", "g1", "g2", "t")):
        plain = scale_model(plain, mods["s"], mods["ro"], name, g1=mods["g1"], g2=mods["g2"], t=mods["t"])
    plain.name = name                           # the name keeps caches of different models and engines apart
    plain.coherent = float(mods["chi"])
    return plain


def homogenize(dm: DeviceModel, name: str | None = None) -> DeviceModel:
    """Every qubit gets the median T1 and T2 (T2 ≤ 2 T1) and the mean P(1|0), P(0|1) of the model's qubits; every gate
    the median reported error and the median duration of its kind (over the gates of that name the model covers).
    The couplings and the schedule's structure are unchanged: only the heterogeneity of the calibration is removed."""
    t1 = float(np.median([q.t1 for q in dm.qubits]))
    t2 = min(float(np.median([q.t2 for q in dm.qubits])), 2 * t1)
    p01 = float(np.mean([q.p01 for q in dm.qubits]))
    p10 = float(np.mean([q.p10 for q in dm.qubits]))
    by_name = {}
    for (nm, qa), e in dm.errors.items():
        by_name.setdefault(nm, []).append((e, dm.durations.get((nm, qa), 0.0)))
    med = {nm: (None if any(e is None for e, _ in v) else float(np.median([e for e, _ in v])),
                float(np.median([d for _, d in v]))) for nm, v in by_name.items()}
    errors = {k: med[k[0]][0] for k in dm.errors}
    durations = {k: med[k[0]][1] if k[0] in med else d for k, d in dm.durations.items()}
    return DeviceModel(name or dm.name + "@hom", dm.phys, dm.dt, [QubitInfo(t1, t2, p01, p10) for _ in dm.qubits],
                       dm.two_qubit, durations, errors)


def scale_model(dm: DeviceModel, s: float, ro: float = 1.0, name: str | None = None, *, g1: float = 1.0,
                g2: float = 1.0, t: float = 1.0) -> DeviceModel:
    """The device model with every reported gate error × s and every relaxation rate × s (T1 → T1/s, T2 → T2/s),
    and both readout flip probabilities × ro. Gate durations, couplings and the schedule are unchanged.

    Aer derives each gate's error from (duration, T1, T2, reported error) as thermal relaxation over the gate
    followed by depolarizing that makes up the rest of the reported error. Scaling the reported error and the
    relaxation rates together keeps that split, so every error source grows with s (linearly to first order; the
    exact fault rate of the scaled circuit is measured by qemx.phase.lightcone, not assumed). Out-of-range values
    raise: no gate error may exceed that of the completely depolarizing channel, 1 − 1/d, and no readout flip
    probability may reach 1/2.

    Component scales (1.7.0): ``g1`` and ``g2`` multiply only the one- and two-qubit gates' reported errors, ``t``
    only the relaxation rates; each combines with ``s``. Scaling relaxation without the reported gate errors moves
    part of each gate's error from the depolarizing remainder to relaxation (the reported error stays as long as the
    relaxation part does not exceed it) and lengthens nothing but the idle decoherence."""
    s, ro, g1, g2, t = float(s), float(ro), float(g1), float(g2), float(t)
    if not (np.isfinite(s) and s > 0 and np.isfinite(ro) and ro >= 0):
        raise ValueError("scale factors must be finite, s > 0 and ro >= 0")
    if not all(np.isfinite(x) and x > 0 for x in (g1, g2, t)):
        raise ValueError("component scales g1, g2 and t must be finite and > 0")
    errors = {}
    for (nm, qa), e in dm.errors.items():
        if e is None:
            errors[(nm, qa)] = None
            continue
        d = 2 ** len(qa)
        es = float(e) * s * (g1 if len(qa) == 1 else g2 if len(qa) == 2 else 1.0)
        if es > 1 - 1 / d:
            raise ValueError("gate %s on %s: error %.4g × %g = %.4g exceeds the completely depolarizing value %.4g"
                             % (nm, qa, e, s, es, 1 - 1 / d))
        errors[(nm, qa)] = es
    qubits = []
    for i, q in enumerate(dm.qubits):
        p01, p10 = q.p01 * ro, q.p10 * ro
        if p01 >= 0.5 or p10 >= 0.5:
            raise ValueError("qubit %d: readout flip probability × %g reaches 1/2" % (i, ro))
        qubits.append(QubitInfo(q.t1 / (s * t), q.t2 / (s * t), p01, p10))
    return DeviceModel(name or "%s@s=%r,ro=%r" % (dm.name, s, ro), dm.phys, dm.dt, qubits, dm.two_qubit,
                       dict(dm.durations), errors)


def model_for(b, phys: tuple, name: str | None = None) -> DeviceModel:
    """DeviceModel of any BackendV2 (fake or real) on the physical qubits `phys`."""
    name = name or getattr(b, "name", "backend")
    phys = tuple(int(p) for p in phys)
    tg = b.target
    idx = {p: i for i, p in enumerate(phys)}
    props = _props(b)
    qinfo = []
    for p in phys:
        qp = b.qubit_properties(p) if hasattr(b, "qubit_properties") else None
        t1 = float(getattr(qp, "t1", None) or 1e-3)
        t2 = float(getattr(qp, "t2", None) or t1)
        t2 = min(t2, 2 * t1)
        p01 = p10 = None
        if props is not None:
            try:
                d = props.qubit_property(p)
                p01 = float(d["prob_meas1_prep0"][0])
                p10 = float(d["prob_meas0_prep1"][0])
            except Exception:
                p01 = p10 = None
        if p01 is None:
            try:
                e = float(tg["measure"][(p,)].error or 0.0)
            except Exception:
                e = 0.0
            p01 = p10 = e
        qinfo.append(QubitInfo(t1, t2, p01, p10))
    durations, errors = {}, {}
    two = next((g for g in ("cz", "ecr", "cx") if g in tg.operation_names), "cx")
    for nm in tg.operation_names:
        if nm in ("measure", "delay", "reset") or nm.startswith(("if_", "for_", "while_", "switch")):
            continue
        try:
            items = tg[nm].items()
        except Exception:
            continue
        for qargs, ip in items:
            if qargs is None or not all(q in idx for q in qargs):
                continue
            c = tuple(idx[q] for q in qargs)
            durations[(nm, c)] = float(ip.duration) if ip is not None and ip.duration is not None else 0.0
            errors[(nm, c)] = float(ip.error) if ip is not None and ip.error is not None else None
    return DeviceModel(name, tuple(phys), float(b.dt or 1e-9), qinfo, two, durations, errors)


@qiskit_serial
def _sub_target(dm: DeviceModel):
    """A Target on the compact qubits with the device's gates, durations, errors and T1/T2."""
    from qiskit.circuit.library import CZGate, ECRGate, CXGate, RZGate, SXGate, XGate, IGate
    from qiskit.circuit import Parameter, Measure
    from qiskit.providers import QubitProperties
    from qiskit.transpiler import InstructionProperties, Target
    tg = Target(num_qubits=dm.n, dt=dm.dt,
                qubit_properties=[QubitProperties(t1=q.t1, t2=q.t2) for q in dm.qubits])
    gates = {"cz": CZGate(), "ecr": ECRGate(), "cx": CXGate(), "rz": RZGate(Parameter("θ")), "sx": SXGate(),
             "x": XGate(), "id": IGate()}
    by = {}
    for (nm, qa), dur in dm.durations.items():
        if nm in gates:
            by.setdefault(nm, {})[qa] = InstructionProperties(duration=dur, error=dm.errors.get((nm, qa)))
    for nm, d in by.items():
        tg.add_instruction(gates[nm], d)
    tg.add_instruction(Measure(), {(i,): InstructionProperties(duration=1e-6, error=(q.p01 + q.p10) / 2)
                                   for i, q in enumerate(dm.qubits)})
    return tg


@qiskit_serial
def _twirl_error(err):
    """Pauli twirl of a QuantumError: the Pauli channel with the same Pauli fidelities (PTM diagonal)."""
    from qiskit.quantum_info import Kraus, Pauli, pauli_basis
    from qiskit_aer.noise import pauli_error
    k = err.num_qubits
    K = Kraus(err.to_quantumchannel()).data
    labels = pauli_basis(k).to_labels()
    mats = [Pauli(l).to_matrix() for l in labels]
    d = 2 ** k
    f = np.array([np.real(np.trace(P @ sum(A @ P @ A.conj().T for A in K))) / d for P in mats])
    paulis = [Pauli(l) for l in labels]
    probs = []
    for Q in paulis:
        s = np.array([1.0 if Q.commutes(P) else -1.0 for P in paulis])
        probs.append(float(s @ f) / d ** 2)
    probs = np.clip(np.array(probs), 0, None)
    probs = probs / probs.sum()
    return pauli_error([(l, p) for l, p in zip(labels, probs) if p > 1e-15])


@qiskit_serial
def device_noise_model(dm: DeviceModel, twirl: str = "none"):
    """Aer NoiseModel on the compact qubits (gate errors only; idle relaxation is added explicitly)."""
    key = twirl
    if key in dm.noise:
        return dm.noise[key]
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device.models import basic_device_gate_errors
    with _LOCK:
        errs = basic_device_gate_errors(target=_sub_target(dm), thermal_relaxation=True)
        nm = NoiseModel(basis_gates=sorted({"rz", "sx", "x", dm.two_qubit, "id"}))
        for name, qargs, err in errs:
            if name in ("measure", "delay", "reset"):
                continue
            k = len(qargs)
            tw = (twirl in ("2q", "all") and k == 2) or (twirl == "all" and k == 1 and name in ("sx", "x"))
            nm.add_quantum_error(_twirl_error(err) if tw else err, name, list(qargs))
    dm.noise[key] = nm
    return nm


@qiskit_serial
def relax_instruction(dm: DeviceModel, q: int, seconds: float):
    key = (q, round(seconds * 1e12))
    ins = dm.relax_cache.get(key)
    if ins is None:
        from qiskit_aer.noise import thermal_relaxation_error
        qi = dm.qubits[q]
        ins = thermal_relaxation_error(qi.t1, qi.t2, max(seconds, 0.0)).to_instruction()
        dm.relax_cache[key] = ins
    return ins


# ------------------------------------------------------------------ compilation
BAD_ERROR = 0.5            # calibration snapshots flag broken couplers / qubits with error 1.0


@lru_cache(maxsize=16)
@qiskit_serial
def good_target(name: str):
    return good_target_for(fake_backend(base_name(name)))


@qiskit_serial
def good_target_for(b):
    """The device target without broken parts (gate error ≥ 0.5, readout error ≥ 0.4), so layout and routing
    never use a coupler or qubit the calibration marks as failed — as the IBM service does."""
    from qiskit.circuit import Measure, Parameter
    from qiskit.circuit.library import CXGate, CZGate, ECRGate, IGate, RZGate, SXGate, XGate
    from qiskit.transpiler import InstructionProperties, Target
    tg = b.target
    bad = set()
    for q in range(b.num_qubits):
        try:
            e = tg["measure"][(q,)].error
            if e is not None and e >= 0.4:
                bad.add(q)
        except Exception:
            pass
        try:
            e = tg["sx"][(q,)].error
            if e is not None and e >= BAD_ERROR:
                bad.add(q)
        except Exception:
            pass
    qprops = [b.qubit_properties(q) for q in range(b.num_qubits)] if hasattr(b, "qubit_properties") else None
    new = Target(num_qubits=b.num_qubits, dt=b.dt, qubit_properties=qprops)
    gates = {"cz": CZGate(), "ecr": ECRGate(), "cx": CXGate(), "rz": RZGate(Parameter("θ")), "sx": SXGate(),
             "x": XGate(), "id": IGate(), "measure": Measure()}
    for nm, g in gates.items():
        if nm not in tg.operation_names:
            continue
        props = {}
        for qa, ip in tg[nm].items():
            if qa is None or any(q in bad for q in qa):
                continue
            if nm != "measure" and ip is not None and ip.error is not None and ip.error >= BAD_ERROR:
                continue
            props[qa] = InstructionProperties(duration=getattr(ip, "duration", None), error=getattr(ip, "error", None))
        new.add_instruction(g, props)
    return new


def compile_for_device(logical: CircuitSpec, name: str, opt_level: int = 1, seed: int = 11,
                       layout: str = "sabre", initial_layout=None):
    """(implemented compact spec, measure_map, phys, info). Qubit i of the result is physical qubit phys[i]."""
    base = base_name(name)                    # the compiled circuit never depends on noise modifiers
    spec, mm, phys, info = compile_for_target(logical, fake_backend(base), good_target(base), FAKE_DEVICES[base][0],
                                              opt_level, seed, layout, initial_layout)
    info["device"] = name
    return spec, mm, phys, info


@qiskit_serial
def compile_for_target(logical: CircuitSpec, backend, target, tag: str, opt_level: int = 1, seed: int = 11,
                       layout: str = "sabre", initial_layout=None):
    """Transpile for a backend's target (without broken parts) and compact to the physical qubits used."""
    from qiskit import transpile
    from .circuit import to_qiskit
    from .compile import QISKIT_LOCK
    b = backend
    n = logical.n
    if n > b.num_qubits:
        raise ValueError("%s has %d qubits; the circuit needs %d" % (tag, b.num_qubits, n))
    with QISKIT_LOCK:
        qc = to_qiskit(logical, measure=True)
        kw = dict(target=target, optimization_level=int(opt_level), seed_transpiler=int(seed))
        if initial_layout:
            kw["initial_layout"] = list(initial_layout)
        elif layout in ("sabre", "dense", "trivial"):
            kw["layout_method"] = layout
        tq = transpile(qc, **kw)
    used = set()
    for ins in tq.data:
        if ins.operation.name in ("delay",):
            continue
        if ins.operation.name == "barrier" and not (getattr(ins.operation, "label", "") or "").startswith("qx|"):
            continue
        for qq in ins.qubits:
            used.add(tq.find_bit(qq).index)
    phys = tuple(sorted(used))
    if len(phys) > n:
        extra = sorted(set(phys) - {tq.find_bit(ins.qubits[0]).index for ins in tq.data
                                    if ins.operation.name == "measure"})
        raise ValueError("routing on %s passed the circuit's states through physical qubit%s %s, which are not part "
                         "of the %d-qubit circuit; the app measures every qubit it simulates, so choose physical "
                         "qubits that are connected (or clear the forced layout and let the compiler choose)"
                         % (tag, "" if len(extra) == 1 else "s", ", ".join(map(str, extra)) or "?", n))
    idx = {p: i for i, p in enumerate(phys)}
    ops, mm = [], {}
    from .circuit import GATES
    for ins in tq.data:
        nm = ins.operation.name
        qs = tuple(idx[tq.find_bit(q).index] for q in ins.qubits if tq.find_bit(q).index in idx)
        if nm == "measure":
            mm[tq.find_bit(ins.clbits[0]).index] = qs[0]
        elif nm in ("delay", "reset"):
            continue
        elif nm == "barrier":
            lab = getattr(ins.operation, "label", None) or ""
            if lab.startswith("qx|"):
                _, kind, val = lab.split("|")
                ops.append(Op(kind, qs, (float(val),), "noise"))
            elif lab == "keep" and qs:                   # an alignment barrier of the circuit itself: kept
                ops.append(Op("barrier", qs, (), "barrier", "keep"))
        elif nm in GATES:
            ops.append(Op(nm, qs, tuple(float(p) for p in ins.operation.params), "gate"))
        else:
            raise ValueError("the device compiler produced '%s', which the app cannot simulate" % nm)
    edges = set()
    for qargs in (target.qargs or ()):
        if qargs is not None and len(qargs) == 2 and qargs[0] in idx and qargs[1] in idx:
            edges.add(tuple(sorted((idx[qargs[0]], idx[qargs[1]]))))
    spec = CircuitSpec(len(phys), tuple(ops), logical.name + " (%s)" % tag, tuple(sorted(edges)))
    swaps = max(spec.count("2q") - logical.count("2q"), 0)
    try:                                       # logical qubit -> physical qubit before routing
        il = tq.layout.initial_index_layout(filter_ancillas=True) if tq.layout is not None else list(range(n))
        init = [int(il[q]) for q in range(n)]
    except Exception:                          # noqa: BLE001 - informational only
        init = []
    info = dict(phys=list(phys), two_qubit=spec.count("2q"), depth=spec.depth(),
                logical_two_qubit=logical.count("2q"), routing_overhead=swaps, backend_name=getattr(b, "name", tag),
                initial_layout=init)
    return spec, mm, phys, info


# ------------------------------------------------------------------ execution
_TRANSLATE: dict = {}


@qiskit_serial
def _translate(dm: DeviceModel, o: Op) -> list:
    """Native (name, params, compact qubits) sequence for one app gate on the device qubits."""
    key = (dm.name, dm.phys, o.name, tuple(round(float(p), 12) for p in o.params), o.qubits)
    hit = _TRANSLATE.get(key)
    if hit is not None:
        return hit
    native = {"rz", "sx", "x", dm.two_qubit, "id"}
    if o.name in native and (o.k == 1 or (o.name, o.qubits) in {(k[0], k[1]) for k in dm.durations}):
        out = [(o.name, tuple(float(p) for p in o.params), o.qubits)]
    else:
        from qiskit import QuantumCircuit, transpile
        from .compile import QISKIT_LOCK
        qc = QuantumCircuit(dm.n)
        getattr(qc, o.name)(*o.params, *o.qubits)
        with QISKIT_LOCK:
            tq = transpile(qc, target=_sub_target(dm), optimization_level=1, initial_layout=list(range(dm.n)),
                           routing_method="none" if o.k == 1 or _coupled(dm, o.qubits) else None, seed_transpiler=1)
        out = []
        for ins in tq.data:
            if ins.operation.name in ("barrier", "delay"):
                continue
            out.append((ins.operation.name, tuple(float(p) for p in ins.operation.params),
                        tuple(tq.find_bit(q).index for q in ins.qubits)))
    if len(_TRANSLATE) > 50000:
        _TRANSLATE.clear()
    _TRANSLATE[key] = out
    return out


def _coupled(dm: DeviceModel, qs) -> bool:
    return any((nm == dm.two_qubit and set(qa) == set(qs)) for (nm, qa) in dm.durations)


DD_PULSES = {"X2": ("x", "x"), "XY4": ("x", "y", "x", "y")}


@qiskit_serial
def device_circuit(spec: CircuitSpec, dm: DeviceModel, *, corrections=None, dd: str = "none", snapshots: bool = False):
    """The executable Qiskit circuit (compact qubits): native gates, user noise, PEC corrections,
    ALAP schedule with idle relaxation, DD pulses and optional per-step density-matrix saves.
    Returns (circuit, executed ops, step labels)."""
    import qiskit_aer  # noqa: F401  (registers save_density_matrix)
    from qiskit import QuantumCircuit
    from qiskit.circuit import Delay
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Kraus
    from qiskit.transpiler import InstructionDurations, PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
    from .noise import explicit_noise
    n = dm.n
    keep = any(o.kind == "barrier" and o.tag == "keep" for o in spec.ops)
    # alignment barriers ("keep") constrain the schedule, so the circuit is translated in its own order; otherwise
    # in layer order (the steps of time travel)
    ops_all = list(spec.ops) if keep else layer_order(spec)[0]
    exec_ops = [o for o in ops_all if o.kind != "barrier"]
    qc = QuantumCircuit(n)
    step = -1
    for o in ops_all:
        if o.kind == "barrier":
            if o.tag == "keep":
                qc.barrier(*(o.qubits or range(n)))
            continue
        step += 1
        check()
        if o.kind == "gate":
            for nm, ps, qs in _translate(dm, o):
                getattr(qc, nm)(*ps, *qs)
            if corrections is not None:
                for ins in corrections(o, step):
                    if (ins.label or "").startswith("ref#"):
                        # a placeholder for the exact engine (device_exact.py), which substitutes the referenced
                        # instruction -- possibly a signed map; it takes no time, like the PEC Paulis below
                        qc.append(UnitaryGate(np.eye(2 ** len(ins.qubits)), label=ins.label), list(ins.qubits))
                    elif ins.kind == "unitary":
                        qc.append(UnitaryGate(ins.data, check_input=False, label="pec"), list(ins.qubits))
                    else:
                        raise ValueError("the Aer device engine cannot apply a %s correction (signed maps need the "
                                         "exact engine: append @eng=exact to the device)" % ins.kind)
        elif o.kind == "noise":
            for ins in explicit_noise(o, step):
                if ins.kind == "unitary":
                    qc.append(UnitaryGate(ins.data, check_input=False, label="noise"), list(ins.qubits))
                elif ins.kind == "kraus":
                    qc.append(Kraus(list(ins.data)).to_instruction(), list(ins.qubits))
                else:
                    qc.append(Kraus([np.sqrt(max(w, 0)) * U for w, U in ins.data]).to_instruction(), list(ins.qubits))
        if snapshots:
            qc.barrier(*o.qubits, label="step%d" % step)
    # schedule with the device durations (in units of dt)
    durs = [(nm, list(qa), int(round(d / dm.dt))) for (nm, qa), d in dm.durations.items()]
    durs += [("unitary", None, 0), ("kraus", None, 0), ("barrier", None, 0)]
    idur = InstructionDurations(durs, dt=dm.dt)
    pm = PassManager([ALAPScheduleAnalysis(durations=idur), PadDelay(durations=idur)])
    sq = pm.run(qc)
    data = list(sq.data)
    if snapshots:
        data = _order_by_step(sq, data, len(exec_ops))
    # delays -> relaxation (with DD pulses in long windows); step markers -> saves
    out = QuantumCircuit(n)
    xdur = [dm.durations.get(("x", (q,)), 0.0) for q in range(n)]
    started = set()                         # qubits that have had a gate (DD is not applied before the first one)
    for ins in data:
        nm = ins.operation.name
        qs = [sq.find_bit(q).index for q in ins.qubits]
        if nm not in ("delay", "barrier", "save_step"):
            started.update(qs)
        if nm == "delay":
            T = float(ins.operation.duration) * (dm.dt if ins.operation.unit == "dt" else 1.0)
            q = qs[0]
            # a qubit still in |0⟩ needs no decoupling (Qiskit's PadDynamicalDecoupling, skip_reset_qubits=True)
            pulses = DD_PULSES.get(dd, ()) if q in started else ()
            if pulses and T >= len(pulses) * xdur[q] * 2 and xdur[q] > 0:
                free = T - len(pulses) * xdur[q]
                k = len(pulses)
                gaps = [free / (2 * k)] + [free / k] * (k - 1) + [free / (2 * k)]
                out.append(relax_instruction(dm, q, gaps[0]), [q])
                for j, g in enumerate(pulses):
                    if g == "y":                    # Y ∝ X·Z: a virtual RZ(π) then X
                        out.rz(np.pi, q)
                    out.x(q)
                    out.append(relax_instruction(dm, q, gaps[j + 1]), [q])
            elif T > 0:
                out.append(relax_instruction(dm, q, T), [q])
        elif nm == "save_step":
            out.save_density_matrix(label="s%d" % ins.step)
        elif nm == "barrier":
            continue
        else:
            out.append(ins.operation, qs)
    out.save_density_matrix(label="final")
    return out, exec_ops


class _Save:
    """Marker: save the state after `step` (inserted by _order_by_step)."""
    def __init__(self, step):
        self.step = step
        self.operation = type("SaveOp", (), {"name": "save_step"})()
        self.qubits = ()


def _order_by_step(sq, data, nsteps):
    """Reorder a scheduled circuit so that all instructions of step k (and the idle windows just before them)
    come before anything of step k+1, then insert a save marker after each step. The order on every wire is
    unchanged, so the circuit is the same channel; only the positions of the saves become well defined."""
    wires = [[sq.find_bit(q).index for q in ins.qubits] for ins in data]
    step_of = [None] * len(data)
    # a step marker is the barrier labeled 'stepK'; every other item belongs to the next marker on its wire
    marker = {}
    for i, ins in enumerate(data):
        lab = getattr(ins.operation, "label", "") or ""
        if ins.operation.name == "barrier" and lab.startswith("step"):
            marker[i] = int(lab[4:])
    nxt = {}
    for i in range(len(data) - 1, -1, -1):
        if i in marker:
            for w in wires[i]:
                nxt[w] = marker[i]
            continue
        cands = [nxt.get(w, nsteps) for w in wires[i]]
        step_of[i] = min(cands) if cands else nsteps
    groups = {}
    for i, ins in enumerate(data):
        if i in marker:
            continue
        groups.setdefault(step_of[i], []).append(ins)
    out = []
    for k in range(nsteps + 1):
        out += groups.get(k, [])
        if k < nsteps:
            out.append(_Save(k))
    return out


@qiskit_serial
def run_device(spec: CircuitSpec, backend: str, phys: tuple, measure_map: dict | None = None, *, ideal: bool = False,
             twirl: str = "none", dd: str = "none", corrections=None, snapshots: bool = False):
    """RunResult of `spec` (compact qubits) on the device model of `backend` restricted to `phys`."""
    from .engine import RunResult, permute_probs, simulate
    from .noise import NoiseSpec
    n = spec.n
    mm = measure_map or {i: i for i in range(n)}
    if ideal:
        return simulate(spec, NoiseSpec(), "numpy", measure_map=mm, ideal=True, snapshots=snapshots)
    if len(phys) != n:
        raise ValueError("device layout has %d qubits, circuit %d" % (len(phys), n))
    dm = get_model(backend, tuple(int(p) for p in phys))
    if parse_device(backend)[1]["eng"] == "exact":
        if snapshots:
            raise ValueError("per-step snapshots are not available with the exact device engine (eng=exact); "
                             "use the Aer device engine for time travel")
        from . import device_exact
        rho, exec_ops = device_exact.final_state(spec, dm, twirl=twirl, dd=dd, corrections=corrections)
        return finish(rho, dm, mm, n, exec_ops, None, exact=True)
    qc, exec_ops = device_circuit(spec, dm, corrections=corrections, dd=dd, snapshots=bool(snapshots))
    from qiskit_aer import AerSimulator
    check()
    sim = AerSimulator(method="density_matrix", noise_model=device_noise_model(dm, twirl))
    res = sim.run(qc, shots=1).result()
    if not res.success:
        raise RuntimeError("Aer (device model): %s" % getattr(res, "status", "failed"))
    data = res.data(0)
    rho = np.asarray(data["final"])
    snaps = None
    if snapshots:
        snaps = {int(k[1:]): np.asarray(v) for k, v in data.items() if k.startswith("s") and k[1:].isdigit()}
    return finish(rho, dm, mm, n, exec_ops, snaps)


def finish(rho, dm: DeviceModel, mm: dict, n: int, exec_ops, snaps, exact: bool = False):
    """RunResult from the final state: distribution over classical bits, then each qubit's own readout error.

    ``exact`` (the superoperator engine) keeps the diagonal as it is, like the NumPy engine: a signed
    quasi-probability map (PEC's net run) yields negative entries, and its |q|-weighted run a trace of Γ — clipping or
    renormalizing would change the estimator. Aer's diagonal is clipped and renormalized against round-off."""
    from .engine import RunResult, permute_probs
    if exact:
        p = np.real(np.diag(rho)).copy()
    else:
        p = np.clip(np.real(np.diag(rho)), 0, None)
        p = p / p.sum() if p.sum() > 0 else p
    p = permute_probs(p, mm, n)
    # readout: bit i is read from qubit mm[i] with that qubit's own confusion matrix
    A = dm.readout_matrices()
    t = p.reshape([2] * n)
    for i in range(n):
        ax = n - 1 - i
        t = np.moveaxis(np.tensordot(A[mm.get(i, i)], t, axes=([1], [ax])), 0, ax)
    pm = t.reshape(-1)
    return RunResult(n, p, pm, rho, exec_ops, list(range(len(exec_ops))), snaps, 1)
