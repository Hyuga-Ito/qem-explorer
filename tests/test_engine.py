"""Circuits, templates, noise channels and the two backends."""
import math

import numpy as np
import pytest

from qemx import circuit as C
from qemx import compile as CP
from qemx import templates as T
from qemx.backends.base import NotSupported
from qemx.engine import bloch_vectors, simulate
from qemx.noise import (Instruction, NoiseSpec, PRESETS, depol_mixture, field_deltas, gate_noise_superop, pauli_matrix,
                        ptm, resolve, superop_kraus, kraus_amp, kraus_phase)


# ------------------------------------------------------------------ circuits
def test_parse_and_text_round_trip():
    s = C.parse("h 0; cx 0 1\nrz(pi/4) 1 # comment\ncp(pi/2) 0 2; depol(0.02) 0 1; barrier; ampdamp(0.1) 2", 3)
    assert C.parse(s.text(), 3).ops == s.ops
    assert s.ops[2].params[0] == pytest.approx(math.pi / 4)


@pytest.mark.parametrize("bad", ["cx 0", "rz 0", "foo 1", "h 0 0", "cx 1 1", "depol(1.5) 0", "rx(pi/2 0"])
def test_parse_errors(bad):
    with pytest.raises((C.ParseError, ValueError)):
        C.parse(bad, 3)


def test_eval_expr():
    assert C.eval_expr("2*pi/3") == pytest.approx(2 * math.pi / 3)
    assert C.eval_expr("sqrt(2)/2") == pytest.approx(math.sqrt(2) / 2)
    with pytest.raises(ValueError):
        C.eval_expr("__import__('os')")


def test_qiskit_round_trip_keeps_noise_ops():
    s = C.parse("h 0; cx 0 1; depol(0.1) 0 1; rz(0.3) 1; ampdamp(0.2) 0", 2)
    back, mm = C.from_qiskit(C.to_qiskit(s, measure=True))
    assert [o for o in back.ops if o.kind != "barrier"] == [o for o in s.ops]
    assert mm == {0: 0, 1: 1}


def test_gate_matrices_are_qiskit_matrices():
    from qiskit.quantum_info import Operator
    s = C.parse("h 0; cx 0 1; cp(0.7) 1 0; rzz(0.4) 0 1; sx 1; ry(0.3) 0", 2)
    U = np.eye(4, dtype=complex)
    for o in s.ops:
        M = C.op_matrix(o)
        if o.qubits == (0, 1) or o.k == 2:
            full = M if o.qubits == (0, 1) else np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ M @ \
                np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        else:
            full = np.kron(np.eye(2), M) if o.qubits[0] == 0 else np.kron(M, np.eye(2))
        U = full @ U
    assert np.allclose(U, Operator(C.to_qiskit(s)).data)


# ------------------------------------------------------------------ templates
def test_qft_body_plus_swaps_is_qiskit_qft():
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import Operator
    for n in (2, 3, 4):
        spec = C.CircuitSpec(n, tuple(T.qft_body(list(range(n))) + T.swap_network(list(range(n)))))
        assert np.allclose(Operator(C.to_qiskit(spec)).data, Operator(QFTGate(n)).data)


@pytest.mark.parametrize("mode", ["swap", "bitreverse", "relabel"])
@pytest.mark.parametrize("s", [0, 1, 5, 6])
def test_qft_modes_reach_target(mode, s):
    tpl = T.qft(3, s, mode)
    for opts in (CP.CompileOptions(), CP.CompileOptions(mode="isa", coupling="line")):
        comp = CP.build(tpl.spec, relabel=tpl.relabel, opts=opts)
        r = simulate(comp.implemented, NoiseSpec(), measure_map=comp.measure_map, ideal=True)
        assert tpl.observable.o @ r.probs == pytest.approx(1.0, abs=1e-9)


def test_aqft_matches_product_of_cosines():
    # dropped phases at distance d contribute cos^2(pi/2^(d+1)) for the Fourier state of s = 1
    n, m = 4, 1
    tpl = T.qft(n, 1, "swap", m)
    r = simulate(CP.build(tpl.spec).implemented, NoiseSpec(), ideal=True)
    expect = np.prod([math.cos(math.pi / 2 ** (d + 1)) ** 2 for d in range(m + 1, n)])
    assert tpl.observable.o @ r.probs == pytest.approx(expect, abs=1e-9)


def test_qpe_fejer_kernel():
    t, phi = 3, 0.3
    tpl = T.qpe(t, phi)
    r = simulate(CP.build(tpl.spec).implemented, NoiseSpec(), ideal=True)
    p = r.probs.reshape(2, 2 ** t)[1]           # target qubit t is |1>
    N = 2 ** t
    for j in range(N):
        d = phi - j / N
        a = abs(np.sum(np.exp(2j * np.pi * np.arange(N) * d))) ** 2 / N ** 2
        assert p[j] == pytest.approx(a, abs=1e-9)


def test_ising_trotter_matches_exact_product_formula():
    from scipy.linalg import expm
    n, J, h, dt, steps = 3, 1.0, 0.7, 0.2, 3
    tpl = T.ising(n, J, h, dt, steps)
    r = simulate(tpl.spec, NoiseSpec(), ideal=True)
    Z = np.diag([1, -1]); X = np.array([[0, 1], [1, 0]]); I = np.eye(2)
    def op(P, q):
        m = np.array([[1.0]])
        for k in reversed(range(n)):
            m = np.kron(m, P if k == q else I)
        return m
    HZ = -J * sum(op(Z, i) @ op(Z, i + 1) for i in range(n - 1))
    HX = -h * sum(op(X, i) for i in range(n))
    U = np.linalg.matrix_power(expm(-1j * HX * dt) @ expm(-1j * HZ * dt), steps)
    psi = U[:, 0]
    assert np.allclose(r.probs, np.abs(psi) ** 2, atol=1e-9)


# ------------------------------------------------------------------ noise
def test_channels_are_cptp_and_depolarizing_infidelity():
    for S in (superop_kraus(kraus_amp(0.3)), superop_kraus(kraus_phase(0.2))):
        R = ptm(S)
        assert R[0, 0] == pytest.approx(1) and np.allclose(R[0, 1:], 0)
    for k in (1, 2):
        p = 0.07
        assert sum(w for w, _ in depol_mixture(p, k)) == pytest.approx(1)
        assert NoiseSpec(p1=p, p2=p).gate_infidelity(k) == pytest.approx((2 ** k - 1) / 2 ** k * p)


def test_twirl_keeps_average_fidelity():
    from qemx.noise import twirl_instructions
    noise = NoiseSpec(p2=0.02, zz_phase=0.2, amp_damp=0.03)
    o = C.Op("cx", (0, 1), (), "gate")
    from qemx.noise import gate_noise_instructions, instr_superop
    ins = gate_noise_instructions(noise, o, 0)
    S = np.eye(16, dtype=complex)
    for i in ins:
        S = instr_superop(i, (0, 1)) @ S
    T_ = twirl_instructions(ins, (0, 1), 0)[0]
    S2 = instr_superop(T_, (0, 1))
    assert np.trace(ptm(S)) == pytest.approx(np.trace(ptm(S2)))
    R2 = ptm(S2)
    assert np.allclose(R2, np.diag(np.diag(R2)), atol=1e-12)


def test_field_samples_are_antithetic():
    d = field_deltas(NoiseSpec(field_sigma=0.1, field_samples=8), 3)
    assert len(d) == 8 and np.allclose(np.sum(d, axis=0), 0)


def test_dd_refocuses_a_quasi_static_field_but_not_markovian_dephasing():
    spec = C.parse("h 0; h 1; cx 1 2; cx 1 2; cx 1 2; cx 1 2; barrier; h 0", 3)     # qubit 0 idles for 4 layers
    ideal = simulate(spec, NoiseSpec(), ideal=True).probs
    field = NoiseSpec(field_sigma=0.6, field_samples=16)
    off = simulate(spec, field).probs
    on = simulate(spec, field, dd="X2").probs
    assert abs(off[0] - ideal[0]) > 1e-3
    assert np.allclose(on, ideal, atol=1e-12)
    deph = NoiseSpec(idle_dephase=0.1)
    assert abs(simulate(spec, deph, dd="X2").probs[0] - ideal[0]) > 1e-3


# ------------------------------------------------------------------ backends
@pytest.mark.parametrize("preset", list(PRESETS))
def test_numpy_and_aer_agree(preset):
    tpl = T.qft(3, 3, "swap")
    comp = CP.build(tpl.spec, opts=CP.CompileOptions(mode="isa", coupling="line"))
    noise = PRESETS[preset].replace(field_samples=4)
    for kw in (dict(), dict(dd="XY4"), dict(twirl="all")):
        a = simulate(comp.implemented, noise, "numpy", measure_map=comp.measure_map, snapshots=True, **kw)
        b = simulate(comp.implemented, noise, "aer", measure_map=comp.measure_map, snapshots=True, **kw)
        assert np.allclose(a.probs, b.probs, atol=1e-10)
        assert np.allclose(a.probs_meas, b.probs_meas, atol=1e-10)
        for k in a.snapshots:
            assert np.allclose(a.snapshots[k], b.snapshots[k], atol=1e-10)


def test_aer_refuses_negative_weights():
    from qemx.backends.aer import AerBackend
    ins = [Instruction("mixture", (0,), [(1.2, np.eye(2)), (-0.2, pauli_matrix("X"))], 0, "pec")]
    with pytest.raises(NotSupported):
        AerBackend().execute(ins, 1)


def test_bloch_vectors_of_plus_and_plus_i():
    r = simulate(C.parse("h 0; h 1; s 1", 2), NoiseSpec(), ideal=True)
    b = bloch_vectors(r.rho, 2)
    assert np.allclose(b[0], [1, 0, 0]) and np.allclose(b[1], [0, 1, 0])
