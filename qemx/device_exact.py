r"""Exact execution of the fake-device model with superoperators (the ``@eng=exact`` engine).

The Aer device engine (``device.run_device``) builds one Qiskit circuit per run (``device.device_circuit``: native
gates, ALAP schedule with the device durations, thermal relaxation in every idle window, dynamical-decoupling
pulses, PEC corrections) and lets Aer's density-matrix simulator apply the gate errors of an Aer ``NoiseModel``
(``device.device_noise_model``). This module executes *the same circuit* with *the same errors* in the NumPy
density-matrix executor:

* every instruction of ``device_circuit``'s output is converted, in order, to an :class:`~qemx.noise.Instruction`
  (gates → unitaries, relaxation and explicit noise → Kraus maps);
* after each gate it applies the error Aer would apply: the ``QuantumError`` that
  ``basic_device_gate_errors`` derives for that gate name on exactly those qubits (Pauli-twirled for the twirled
  gates, with the same rule as ``device_noise_model``), as its Kraus operators. Aer applies a gate's error after the
  gate, on the gate's qubits; a gate without an entry gets none, in both engines;
* corrections may be **signed** quasi-probability maps (PEC): they enter ``device_circuit`` as zero-duration
  placeholders (so the schedule is the one the Aer engine builds for sampled Pauli corrections) and are replaced
  here by the signed map. The NumPy executor applies negative weights exactly, so the PEC mean on a device model is
  exact instead of a Monte Carlo estimate over sampled instances.

Equivalence with Aer is a tested property (``tests/test_v13_phase.py``): the final density matrices agree to better
than 1e-8 (≤ 5e-9 observed; tested at 1e-7) for every twirl and DD mode, on CZ (Heron) and ECR (Eagle) devices. The residual is
Aer's: its internal representation of a composite gate error (depolarizing composed with relaxation) differs from
the declared QuantumError's superoperator by up to ~1e-9 per gate, while this engine reproduces the declared channel
to ~1e-15 (``exact_kraus`` checks its reconstruction at 1e-12).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .circuit import CircuitSpec
from .noise import Instruction
from .resources import qiskit_serial

NATIVE = ("rz", "sx", "x", "cz", "ecr", "cx", "id")


def exact_kraus(channel) -> list:
    """Kraus operators of a CPTP channel (QuantumError, QuantumChannelInstruction, SuperOp, …) without truncation.

    ``qiskit.quantum_info.Kraus`` drops Choi eigenvalues of magnitude <= 1e-8 (and shifts ill-conditioned Choi
    matrices by 1e-10 before the decomposition). For the device errors used here that is harmless (both conversions
    reproduce the superoperator to ~1e-15), but a weak channel with small Choi eigenvalues would be truncated. This
    conversion keeps every positive eigenvalue and checks the reconstruction against the channel's superoperator
    (1e-12), so a silent truncation is impossible."""
    from qiskit.quantum_info import Choi, Kraus, SuperOp
    if hasattr(channel, "to_quantumchannel"):
        channel = channel.to_quantumchannel()
    S = SuperOp(channel)
    din, dout = int(np.prod(S.input_dims())), int(np.prod(S.output_dims()))
    C = Choi(S).data
    C = (C + C.conj().T) / 2
    w, V = np.linalg.eigh(C)
    scale = max(1.0, float(np.max(np.abs(w))))
    if float(np.min(w)) < -1e-10 * scale:
        raise ValueError("channel is not completely positive (Choi eigenvalue %.3g)" % float(np.min(w)))
    Ks = [np.sqrt(float(x)) * V[:, i].reshape((dout, din), order="F") for i, x in enumerate(w) if x > 0]
    err = float(np.max(np.abs(SuperOp(Kraus(Ks)).data - S.data)))
    if err > 1e-12:
        raise RuntimeError("Kraus reconstruction error %.3g" % err)
    return [np.asarray(K, complex) for K in Ks]


@qiskit_serial
def gate_error_table(dm, twirl: str = "none") -> dict:
    """{(gate name, compact qubits): [Kraus matrices]} — the errors ``device_noise_model(dm, twirl)`` installs."""
    key = "exact-table|" + twirl
    hit = dm.noise.get(key)
    if hit is not None:
        return hit
    from qiskit_aer.noise.device.models import basic_device_gate_errors
    from .device import _LOCK, _sub_target, _twirl_error
    with _LOCK:
        errs = basic_device_gate_errors(target=_sub_target(dm), thermal_relaxation=True)
    table = {}
    chi = float(getattr(dm, "coherent", 0.0) or 0.0)
    for name, qargs, err in errs:
        if name in ("measure", "delay", "reset"):
            continue
        k = len(qargs)
        qa = tuple(int(q) for q in qargs)
        tw = (twirl in ("2q", "all") and k == 2) or (twirl == "all" and k == 1 and name in ("sx", "x"))
        if chi > 0 and name in ("sx", "x", "cz", "ecr", "cx"):
            chan = coherent_gate_channel(dm, name, qa, chi)
            table[(name, qa)] = exact_kraus(pauli_twirl_channel(chan) if tw else chan)
            continue
        e = _twirl_error(err) if tw else err
        table[(name, qa)] = exact_kraus(e)
    dm.noise[key] = table
    return table


def _relaxation(dm, qa: tuple, seconds: float):
    """Thermal relaxation over a gate, built as Aer's basic_device_gate_errors builds it (first qubit least
    significant: error_q0.expand(error_q1))."""
    from qiskit_aer.noise import thermal_relaxation_error
    err = None
    for q in qa:
        e = thermal_relaxation_error(dm.qubits[q].t1, dm.qubits[q].t2, seconds)
        err = e if err is None else err.expand(e)
    return err


def coherent_gate_channel(dm, name: str, qa: tuple, chi: float):
    """The error channel of one gate with a coherent fraction χ (a qiskit SuperOp, applied after the gate):

        E = U_c ∘ R ∘ D(p),   R = thermal relaxation over the gate duration (as Aer),
                              U_c = RX(a) (one-qubit pulses) or exp(−iθ Z⊗Z/2) (two-qubit gates),
                              D(p) = depolarizing with parameter p.

    The reported error r is split as Aer splits it — relaxation r_R plus the rest r − r_R — and a fraction χ of the
    rest is made coherent: the unitary's own infidelity is χ (r − r_R), i.e. (2/3) sin²(a/2) or (4/5) sin²(θ/2).
    The depolarizing parameter then makes the *total* average gate infidelity exactly r: the average fidelity is
    linear in the channel, F(E) = (1 − p) F(U_c∘R) + p/d, so p = (F(U_c∘R) − (1 − r)) / (F(U_c∘R) − 1/d) (checked to
    1e-12). With χ = 0 this is Aer's model. If relaxation alone reaches r there is no room for a coherent part and
    the channel is R (as in Aer)."""
    import math
    from qiskit.quantum_info import Operator, SuperOp, average_gate_fidelity
    from qiskit_aer.noise import depolarizing_error
    k = len(qa)
    d = 2 ** k
    r = float(dm.errors[(name, qa)] or 0.0)
    R = SuperOp(_relaxation(dm, qa, float(dm.durations.get((name, qa), 0.0))).to_quantumchannel())
    r_R = 1.0 - float(average_gate_fidelity(R))
    if r <= r_R + 1e-15:
        return R
    r_c = chi * (r - r_R)
    if k == 1:
        a = 2.0 * math.asin(math.sqrt(1.5 * r_c))
        U = np.array([[math.cos(a / 2), -1j * math.sin(a / 2)], [-1j * math.sin(a / 2), math.cos(a / 2)]])
    else:
        th = 2.0 * math.asin(math.sqrt(1.25 * r_c))
        U = np.diag(np.exp(-0.5j * th * np.array([1, -1, -1, 1])))
    UR = R.compose(SuperOp(Operator(U)))                # first R, then U
    F0 = float(average_gate_fidelity(UR))
    p = (F0 - (1.0 - r)) / (F0 - 1.0 / d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("gate %s on %s: no depolarizing parameter reproduces r = %.4g with χ = %g" % (name, qa, r, chi))
    E = SuperOp(depolarizing_error(p, k).to_quantumchannel()).compose(UR)    # D(p), then R, then U
    got = 1.0 - float(average_gate_fidelity(E))
    if abs(got - r) > 1e-12:
        raise AssertionError("coherent channel infidelity %.6g != %.6g" % (got, r))
    return E


def pauli_twirl_channel(chan):
    """The Pauli twirl of a channel: the Pauli channel with the same Pauli fidelities (PTM diagonal)."""
    from qiskit.quantum_info import Pauli, SuperOp, pauli_basis
    from qiskit_aer.noise import pauli_error
    k = int(round(np.log2(chan.input_dims()[0] if len(chan.input_dims()) == 1 else np.prod(chan.input_dims()))))
    labels = pauli_basis(k).to_labels()
    d = 2 ** k
    S = SuperOp(chan)
    f = []
    for l in labels:
        P = Pauli(l).to_matrix()
        out = SuperOp(S).data @ P.reshape(-1, order="F")          # SuperOp acts on column-stacked vectors
        f.append(float(np.real(np.trace(P @ out.reshape(d, d, order="F")))) / d)
    f = np.array(f)
    paulis = [Pauli(l) for l in labels]
    probs = np.array([sum((1.0 if Q.commutes(P) else -1.0) * fv for P, fv in zip(paulis, f)) / d ** 2 for Q in paulis])
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    return pauli_error([(l, float(pr)) for l, pr in zip(labels, probs) if pr > 1e-15])


def instructions(spec: CircuitSpec, dm, *, twirl: str = "none", dd: str = "none", corrections=None) -> tuple:
    """(instruction list, executed logical ops) equivalent to the Aer device run of ``spec`` on ``dm``.

    Instruction sources: ``gate`` (native gates, DD pulses included), ``noise`` (the gate errors), ``idle``
    (thermal relaxation in scheduled idle windows), ``explicit`` (noise operations placed in the circuit), ``pec``
    (corrections). Steps number the converted circuit's instructions; a gate and its error share a step."""
    from .device import device_circuit
    refs = {}
    wrapped = None
    if corrections is not None:
        def wrapped(o, step):
            out = []
            for ins in corrections(o, step):
                lab = "ref#%d" % len(refs)
                refs[lab] = ins
                out.append(dataclasses.replace(ins, label=lab))
            return out
    qc, exec_ops = device_circuit(spec, dm, corrections=wrapped, dd=dd, snapshots=False)
    table = gate_error_table(dm, twirl)
    out = []
    for k, ci in enumerate(qc.data):
        op = ci.operation
        nm = op.name
        qs = tuple(qc.find_bit(q).index for q in ci.qubits)
        if nm in ("save_density_matrix", "barrier"):
            continue
        if nm == "unitary":
            lab = op.label or ""
            if lab.startswith("ref#"):
                ins = refs[lab]
                if tuple(ins.qubits) != qs:
                    raise RuntimeError("correction placeholder moved from %s to %s" % (ins.qubits, qs))
                out.append(dataclasses.replace(ins, step=k, label="PEC correction", source="pec"))
            else:
                out.append(Instruction("unitary", qs, np.asarray(op.to_matrix(), complex), k,
                                       "pec" if lab == "pec" else "explicit", lab or "unitary"))
        elif nm == "quantum_channel":             # thermal relaxation of an idle window (device.relax_instruction)
            out.append(Instruction("kraus", qs, _kraus_cached(op), k, "idle", "relaxation"))
        elif nm == "kraus":                       # a noise operation the user placed in the circuit
            out.append(Instruction("kraus", qs, exact_kraus(op), k, "explicit", "explicit noise"))
        elif nm in NATIVE:
            out.append(Instruction("unitary", qs, np.asarray(op.to_matrix(), complex), k, "gate", nm))
            err = table.get((nm, qs))
            if err is not None:
                out.append(Instruction("kraus", qs, err, k, "noise", "%s error" % nm))
        else:
            raise ValueError("the exact device engine cannot execute '%s'" % nm)
    return out, exec_ops


_KRAUS: dict = {}


def _kraus_cached(op) -> list:
    """exact_kraus of a relaxation instruction; device.relax_instruction caches the instruction objects, so their
    identity is a valid key (the cache holds a reference, so an id is never reused while cached)."""
    hit = _KRAUS.get(id(op))
    if hit is None or hit[0] is not op:
        if len(_KRAUS) > 20000:
            _KRAUS.clear()
        hit = (op, exact_kraus(op))
        _KRAUS[id(op)] = hit
    return hit[1]


def final_state(spec: CircuitSpec, dm, *, twirl: str = "none", dd: str = "none", corrections=None) -> tuple:
    """(final density matrix of the compact qubits, executed logical ops)."""
    from .backends.numpy_dm import NumpyBackend
    instr, exec_ops = instructions(spec, dm, twirl=twirl, dd=dd, corrections=corrections)
    rho, _ = NumpyBackend().execute(instr, spec.n)
    return rho, exec_ops


# ---------------------------------------------------------------------------------------------- factorized product runs
class NotFactorizable(ValueError):
    """The circuit's resolved noise instructions are not all single-qubit, so its output does not factorize."""


def _apply_1q(rho: np.ndarray, ins) -> np.ndarray:
    d = ins.data
    if ins.kind == "unitary":
        U = np.asarray(d, complex)
        return U @ rho @ U.conj().T
    if ins.kind == "kraus":
        return sum(np.asarray(K, complex) @ rho @ np.asarray(K, complex).conj().T for K in d)
    if ins.kind == "mixture":
        return sum(float(w) * (np.asarray(U, complex) @ rho @ np.asarray(U, complex).conj().T) for w, U in d)
    raise ValueError("unknown instruction kind %r" % (ins.kind,))


def product_marginals(spec: CircuitSpec, dm, *, twirl: str = "none", dd: str = "none") -> np.ndarray:
    """P(qubit q ends in |1⟩) for every compact qubit of a circuit whose *every* resolved instruction (gates, gate
    errors, idle relaxation, DD pulses) acts on one qubit — e.g. the readout-calibration circuits |0…0⟩ and X on every
    qubit (1.8.0; review finding F2). Each qubit then evolves by its own sequence of single-qubit channels from |0⟩,
    so the n-qubit output state is the product of n 2×2 density matrices: O(n) memory instead of the dense engine's
    4^n. The same instruction list as ``final_state`` is used, so this is the same model, not an approximation.
    Raises NotFactorizable if any instruction acts on two or more qubits (a two-qubit gate, crosstalk or a
    correlated channel): the product calculation would then be wrong, and the caller must decide (dense engine within
    a declared width limit, or an explicit unsupported/cost status)."""
    instr, _ = instructions(spec, dm, twirl=twirl, dd=dd)
    multi = sorted({tuple(i.qubits) for i in instr if len(i.qubits) > 1})
    if multi:
        raise NotFactorizable("%d instruction(s) act on several qubits (first on %s): the output state is not a "
                              "product" % (sum(1 for i in instr if len(i.qubits) > 1), multi[0]))
    rhos = [np.array([[1.0, 0.0], [0.0, 0.0]], complex) for _ in range(spec.n)]
    for ins in instr:
        q = int(ins.qubits[0])
        rhos[q] = _apply_1q(rhos[q], ins)
    return np.array([float(np.real(r[1, 1])) for r in rhos])


def product_calibration(n: int, dm, measure_map: dict) -> list:
    """Per-classical-bit assignment matrices A[i] = [[P(0|0), P(0|1)], [P(1|0), P(1|1)]] of the two readout-calibration
    circuits (|0…0⟩; X on every qubit) on the device model, computed with ``product_marginals`` and each qubit's own
    readout matrix — the quantities ``methods.base.Context.calibration`` obtains from the dense engine. Classical bit i
    reads compact qubit measure_map[i]. Raises NotFactorizable when the calibration circuits do not factorize."""
    from .circuit import Op
    zero = CircuitSpec(n, (), "cal-0")
    one = CircuitSpec(n, tuple(Op("x", (q,), (), "gate", "prep") for q in range(n)), "cal-1")
    m0, m1 = product_marginals(zero, dm), product_marginals(one, dm)
    ro = dm.readout_matrices()
    A = []
    for i in range(n):
        q = int(measure_map.get(i, i)) if measure_map else i
        R = ro[q]                                  # [[1 − p01, p10], [p01, 1 − p10]]
        read1_given_zero = (1 - m0[q]) * R[1, 0] + m0[q] * R[1, 1]
        read0_given_one = (1 - m1[q]) * R[0, 0] + m1[q] * R[0, 1]
        A.append(np.array([[1 - read1_given_zero, read0_given_one], [read1_given_zero, 1 - read0_given_one]]))
    return A
