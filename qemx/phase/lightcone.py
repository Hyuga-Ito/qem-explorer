r"""The circuit fault rate λ and its light-cone restriction λ_LC.

Definitions (the ones every phase diagram is plotted against)
-------------------------------------------------------------
A **fault location** f is one elementary noise instruction of the executed circuit: the error after a gate, an idle
segment's relaxation or dephasing, a crosstalk kick, a noise operation placed in the circuit. Its **fault
probability** is the probability that its Pauli twirl applies a non-identity Pauli,

.. math:: p_f = 1 - p_I,\qquad p_I = \frac{1}{4^k}\sum_P \lambda_P,

with :math:`\lambda_P` the diagonal of the channel's Pauli transfer matrix (exact for any CPTP map: unitary,
Kraus or mixture; a coherent rotation by θ has :math:`p_f=\sin^2(\theta/2)`). Then

* :math:`\lambda=\sum_f p_f` over all locations (Cai et al., RMP 95, 045005 (2023)); idle windows before a qubit's
  first gate are not locations (a T = 0 relaxation leaves |0⟩ unchanged), so both engines start counting a qubit's
  idling at its first gate;
* :math:`\lambda_{\rm LC}=\sum_{f\in{\rm LC}} p_f^{\rm LC}` over the locations in the **backward light cone** of the
  observable: walking the instruction list backwards from the measured qubits the observable depends on, a
  multi-qubit instruction (gate or noise) that touches the current set adds its qubits to it. A location counts
  with the probability of the Pauli components that act non-trivially on light-cone qubits,
  :math:`p_f^{\rm LC}=\sum_{P:\,P|_{\rm LC}\neq I}p_P` (an error on a qubit outside the light cone cannot change
  the observable);
* :math:`\lambda^{\rm exact}_{\rm LC}=-\sum_{f\in{\rm LC}}\ln(1-p_f^{\rm LC})`, so that
  :math:`e^{-\lambda^{\rm exact}}` is the probability that no light-cone fault occurs (independent locations);
* :math:`\lambda_{\rm ro}=\sum_{q\in\,\rm support}(P(1|0)+P(0|1))/2`, reported separately: readout is mitigated
  by its own calibration and is not part of λ.

The observable's support is the set of classical bits its value o(x) depends on (the union of the masks of its
Walsh–Hadamard (Z-string) decomposition), mapped to physical qubits with the measurement map.

Engines: a NoiseSpec (the instruction list of :func:`qemx.noise.resolve`) or a device specification / DeviceModel
(the instruction list of :func:`qemx.device_exact.instructions`, which includes the relaxation of every scheduled
idle window). A quasi-static field enters through its per-segment marginal, :math:`p_f=(1-e^{-(\sigma t)^2/2})/2`,
without credit for refocusing by dynamical decoupling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from ..circuit import CircuitSpec
from ..noise import NoiseSpec, instr_superop, pauli_labels, pauli_probs_from_eigs, ptm

STRUCTURAL = ("gate", "dd")                  # instruction sources that are operations, not faults
EXCLUDED = ("pec",)                          # mitigation corrections are not part of the circuit's noise


@dataclass
class LambdaReport:
    lam_lc: float                    # Σ p_f^LC over light-cone locations
    lam_lc_exact: float              # −Σ ln(1 − p_f^LC)
    lam_total: float                 # Σ p_f over every location
    lam_ro: float                    # readout, Σ over the support of the mean flip probability
    lam_lc_gate: float               # light-cone part from gate errors (incl. twirled, crosstalk, explicit)
    lam_lc_idle: float               # light-cone part from idle windows
    n_loc_lc: int                    # locations with p_f^LC > 0
    n_loc_total: int
    support: tuple                   # physical (compact) qubits of the observable's support
    lc_qubits: tuple                 # qubits in the light cone at the start of the circuit
    lc_gates: int                    # circuit gates inside the light cone (DD pulses not counted)
    lc_2q_gates: int
    engine: str
    by_source: dict = field(default_factory=dict)
    lam_lc_2q: float = 0.0           # 1.8.0: light-cone part from errors of multi-qubit instructions (2q gate errors)

    def to_json(self) -> dict:
        d = asdict(self)
        d["support"] = list(self.support)
        d["lc_qubits"] = list(self.lc_qubits)
        return d


_PROBS: dict = {}


def pauli_probs(ins) -> np.ndarray:
    """Pauli probabilities of the twirl of one instruction's channel, over pauli_labels(k) (label[i] acts on the
    instruction's i-th qubit). Cached by content."""
    k = len(ins.qubits)
    if ins.kind == "unitary":
        key = ("u", k, np.round(np.asarray(ins.data), 14).tobytes())
    elif ins.kind == "kraus":
        key = ("k", k, b"".join(np.round(np.asarray(K), 14).tobytes() for K in ins.data))
    else:
        key = ("m", k, b"".join(np.round(np.asarray([w]), 14).tobytes() + np.round(np.asarray(U), 14).tobytes()
                                for w, U in ins.data))
    hit = _PROBS.get(key)
    if hit is None:
        S = instr_superop(ins, tuple(ins.qubits))
        eigs = np.diag(ptm(S))
        hit = np.real(pauli_probs_from_eigs(eigs, k))
        if len(_PROBS) > 50000:
            _PROBS.clear()
        _PROBS[key] = hit
    return hit


def fault_probability(ins, on=None) -> float:
    """p_f of one instruction; with ``on`` (a set of qubits) only Pauli components acting non-trivially there."""
    p = pauli_probs(ins)
    labels = pauli_labels(len(ins.qubits))
    if on is None:
        mask = [l != "I" * len(l) for l in labels]
    else:
        idx = [i for i, q in enumerate(ins.qubits) if q in on]
        mask = [any(l[i] != "I" for i in idx) for l in labels]
    return float(max(np.sum(p[np.asarray(mask)]), 0.0))


def observable_support(o, measure_map: dict | None, n: int) -> tuple:
    """Physical (compact) qubits whose measured bits the diagonal observable o(x) depends on."""
    from ..observables import Observable
    if isinstance(o, Observable):
        dec = o.z_decomposition()
    else:
        vals = np.asarray(o, float)
        dec = Observable("o", "o", "o", tuple(vals)).z_decomposition()
    mask = 0
    for m, c in dec.items():
        if abs(c) > 1e-12:
            mask |= int(m)
    mm = measure_map or {i: i for i in range(n)}
    return tuple(sorted({int(mm.get(b, b)) for b in range(n) if (mask >> b) & 1}))


def _noise_instructions(spec: CircuitSpec, noise, *, layout=None, twirl="none", dd="none") -> tuple:
    """(instructions, readout flip mean per qubit, engine label)."""
    if isinstance(noise, NoiseSpec):
        from ..noise import resolve
        # δ = 1 rad per layer on every qubit marks each idle segment's field rotation (angle = segment duration in
        # layers); its fault probability is replaced by the field's marginal below
        delta = np.ones(spec.n) if noise.field_sigma else None
        instr, _, _ = resolve(spec, noise, delta=delta, twirl=twirl, dd=dd)
        ro = [(noise.ro01 + noise.ro10) / 2.0] * spec.n
        return instr, ro, "noisespec"
    from .. import device, device_exact
    if isinstance(noise, str):
        if layout is None:
            raise ValueError("a device specification needs the compiled circuit's physical layout")
        dm = device.get_model(noise, tuple(int(p) for p in layout))
        label = noise
    else:
        dm, label = noise, getattr(noise, "name", "device")
    instr, _ = device_exact.instructions(spec, dm, twirl=twirl, dd=dd)
    ro = [(q.p01 + q.p10) / 2.0 for q in dm.qubits]
    return instr, ro, "device:" + label


def calculate_lambda_lc(circuit, noise_model, observable=None, measure_map: dict | None = None, *, layout=None,
                        twirl: str = "none", dd: str = "none", support=None) -> LambdaReport:
    """λ_LC (and λ, λ_ro) of a compiled circuit under a noise model.

    circuit       a :class:`~qemx.compile.Compiled` (its implemented circuit, measurement map and physical layout
                  are used) or an implemented :class:`~qemx.circuit.CircuitSpec`
    noise_model   a :class:`~qemx.noise.NoiseSpec`, a device specification such as ``"fake_fez@s=2"``, or a
                  :class:`~qemx.device.DeviceModel`
    observable    an Observable or o-vector over classical bits; None = every measured qubit
    twirl, dd     the recipe's transforms (twirling replaces gate errors by their Pauli twirl; DD adds pulses)
    support       (1.5.1) the observable's support as physical (compact) qubits, given directly — e.g. one term of
                  a Pauli sum (``qemx.features.noise_features``); overrides ``observable``
    """
    from ..compile import Compiled
    if isinstance(circuit, Compiled):
        spec, mm = circuit.implemented, (measure_map or circuit.measure_map)
        layout = layout if layout is not None else circuit.info.get("phys")
    else:
        spec, mm = circuit, measure_map
    n = spec.n
    if support is not None:
        support = tuple(sorted({int(q) for q in support}))
    elif observable is None:
        support = tuple(sorted({int(v) for v in (mm or {i: i for i in range(n)}).values()}))
    else:
        support = observable_support(observable, mm, n)
    instr, ro, engine = _noise_instructions(spec, noise_model, layout=layout, twirl=twirl, dd=dd)
    sigma = float(noise_model.field_sigma) if isinstance(noise_model, NoiseSpec) else 0.0
    # idle windows before a qubit's first gate act on |0⟩, a fixed point of thermal relaxation at T = 0: they are not
    # fault locations. (The NoiseSpec engine opens idle windows at the first gate; the device schedule pads the
    # start of every wire, so this keeps λ defined identically on both engines.)
    started, leading = set(), set()
    for i, ins in enumerate(instr):
        if ins.source in STRUCTURAL:
            started.update(ins.qubits)
        elif ins.source == "idle" and not started.issuperset(ins.qubits):
            leading.add(i)
    S = set(support)
    lam_lc = lam_ex = lam_tot = lam_gate = lam_idle = lam_2q = 0.0
    n_lc = n_tot = lc_g = lc_2q = 0
    by = {}
    for i in range(len(instr) - 1, -1, -1):
        ins = instr[i]
        if ins.source in EXCLUDED or i in leading:
            continue
        qs = tuple(ins.qubits)
        touches = bool(S.intersection(qs))
        if ins.source in STRUCTURAL:
            if touches and ins.source == "gate":           # circuit gates only (v1.3 also counted DD pulses)
                lc_g += 1
                if len(qs) >= 2:
                    lc_2q += 1
                    S.update(qs)
            continue
        if sigma and ins.source == "idle" and ins.label == "field":
            # RZ(δ·t) with δ ~ N(0, σ²): marginal dephasing, p_f = (1 − e^{−(σt)²/2})/2; t = this segment's angle at δ=1
            t = abs(float(np.angle(ins.data[1, 1] / ins.data[0, 0])))
            pf = (1.0 - math.exp(-(sigma * t) ** 2 / 2.0)) / 2.0
            pf_lc = pf if touches else 0.0
        else:
            pf = fault_probability(ins)
            pf_lc = fault_probability(ins, on=S) if touches else 0.0
        lam_tot += pf
        if pf > 0:
            n_tot += 1
        if pf_lc > 0:
            n_lc += 1
            lam_lc += pf_lc
            if pf_lc >= 1.0:
                raise ValueError("a fault location has probability 1")
            lam_ex += -math.log1p(-pf_lc)
            by[ins.source] = by.get(ins.source, 0.0) + pf_lc
            if ins.source == "idle":
                lam_idle += pf_lc
            else:
                lam_gate += pf_lc
                if len(qs) >= 2:
                    lam_2q += pf_lc
        if touches and len(qs) >= 2:
            S.update(qs)
    lam_ro = float(sum(ro[q] for q in support))
    return LambdaReport(lam_lc, lam_ex, lam_tot, lam_ro, lam_gate, lam_idle, n_lc, n_tot, tuple(support),
                        tuple(sorted(S)), lc_g, lc_2q, engine, by, lam_2q)
