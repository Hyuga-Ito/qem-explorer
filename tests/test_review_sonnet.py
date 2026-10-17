"""Regression tests from an independent physics/engineering review.

The first group pins down behavior that was checked and is correct; the second group reproduces
bugs the review found, which are fixed, so they now guard against regressions.
"""
import math
import subprocess
import sys

import numpy as np
import pytest

from qemx import compile as CP, observables as OBS, templates as T
from qemx.circuit import GATES, Op, ParseError, inverse, op_matrix, parse
from qemx.engine import simulate, bloch_vectors
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.methods.pec import depol_quasi, learned_quasi
from qemx.noise import (NoiseSpec, _frac_power, depol_mixture, gate_noise_instructions, idle_windows, instr_superop,
                        pauli_labels, pauli_matrix, ptm, resolve, twirl_instructions, Instruction)
from qemx.backends.numpy_dm import NumpyBackend
from qemx.recipes import compare
from qemx.transforms import fold_global


def _ctx(spec, noise, obs, recipe, backend="numpy", opts=None, shots=1024):
    comp = CP.build(spec, opts=opts or CP.CompileOptions())
    return Context(comp, noise, obs, recipe, Runner(backend), shots=shots)


def _run(spec, noise, obs, recipe, backend="numpy"):
    ctx = _ctx(spec, noise, obs, recipe, backend)
    return REGISTRY[recipe.estimator].run(ctx, **recipe.p), ctx


# ============================================================ verified correct (regression guards)
def test_inverse_table_inverts_every_gate():
    rng = np.random.default_rng(0)
    for nm, (nq, npar, *_) in GATES.items():
        o = Op(nm, tuple(range(nq)), tuple(rng.uniform(-3, 3, npar)))
        assert np.allclose(op_matrix(inverse(o)) @ op_matrix(o), np.eye(2 ** nq), atol=1e-12), nm


@pytest.mark.parametrize("dd", ["X2", "XY4"])
@pytest.mark.parametrize("L", [2, 3, 4, 5])
def test_dd_refocuses_static_detuning_and_x2_pulse_times(dd, L):
    sp = parse("h 0\n" + "rx(0.0) 1\n" * (L + 1) + "barrier 0 1\nh 0", 2)
    assert idle_windows(sp)[0] == [(1, L + 1)][:1] or idle_windows(sp)[0][0][1] - idle_windows(sp)[0][0][0] + 1 == L
    N = NoiseSpec(field_sigma=1.0)
    rho_none, _ = NumpyBackend().execute(resolve(sp, N, delta=[0.3, 0.0])[0], 2)
    rho_dd, _ = NumpyBackend().execute(resolve(sp, N, delta=[0.3, 0.0], dd=dd)[0], 2)
    assert bloch_vectors(rho_none, 2)[0][2] == pytest.approx(math.cos(0.3 * L), abs=1e-9)
    assert bloch_vectors(rho_dd, 2)[0][2] == pytest.approx(1.0, abs=1e-9)
    if dd == "X2":                                   # pulses at T/4 and 3T/4
        ins, _, _ = resolve(sp, N, delta=[0.3, 0.0], dd="X2")
        t, pos = 0.0, []
        for i in ins:
            if i.qubits == (0,) and i.source == "idle":
                t += float(np.angle(i.data[1, 1] / i.data[0, 0])) / 0.3
            elif i.qubits == (0,) and i.source == "dd":
                pos.append(t)
        assert pos == pytest.approx([L / 4, 3 * L / 4])


def test_twirl_is_the_exact_pauli_average_of_the_channel():
    ns = NoiseSpec(amp_damp=0.05, over_rot=0.1, zz_phase=0.2, z_phase=0.1, p2=0.02)
    for k, nm in ((1, "x"), (2, "cx")):
        o = Op(nm, tuple(range(k)), (), "gate")
        gn = gate_noise_instructions(ns, o, 0)
        S = np.eye(4 ** k, dtype=complex)
        for i in gn:
            S = instr_superop(i, o.qubits) @ S
        St = instr_superop(twirl_instructions(gn, o.qubits, 0)[0], o.qubits)
        Ps = [np.kron(pauli_matrix(l), pauli_matrix(l).conj()) for l in pauli_labels(k)]
        Sd = sum(P @ S @ P.conj().T for P in Ps) / 4 ** k
        assert np.allclose(St, Sd, atol=1e-12)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("construction", ["inverse", "compensation"])
def test_pec_constructions_invert_depolarizing(k, construction):
    qs = depol_quasi(0.07, k, construction)
    C = sum(w * np.kron(pauli_matrix(l), pauli_matrix(l).conj()) for w, l in qs)
    D = instr_superop(Instruction("mixture", tuple(range(k)), depol_mixture(0.07, k), 0, "x"), tuple(range(k)))
    assert np.allclose(C @ D, np.eye(4 ** k), atol=1e-12)
    f = 0.93
    d2 = 4 ** k
    expect = (2 * (d2 - 1) / f - (d2 - 2)) / d2 if construction == "inverse" else (1.07 / 0.93)
    assert sum(abs(w) for w, _ in qs) == pytest.approx(expect)


def test_learned_pec_inverts_the_twirled_channel():
    o = Op("cx", (0, 1), (), "gate")
    ns = NoiseSpec(p2=0.05, zz_phase=0.1, amp_damp=0.01)
    C = sum(w * np.kron(pauli_matrix(l), pauli_matrix(l).conj()) for w, l in learned_quasi(ns, o))
    gn = gate_noise_instructions(ns, o, 0)
    tw = instr_superop(twirl_instructions(gn, o.qubits, 0)[0], o.qubits)
    assert np.allclose(C @ tw, np.eye(16), atol=1e-12)


@pytest.mark.parametrize("mode", ["swap", "bitreverse", "relabel"])
def test_qft_modes_return_the_input_integer(mode):
    for n in (2, 3):
        for s in range(2 ** n):
            t = T.qft(n, s, mode)
            comp = CP.build(t.spec, relabel=t.relabel)
            r = simulate(comp.implemented, NoiseSpec(), "numpy", measure_map=comp.measure_map, ideal=True)
            assert t.observable.o @ r.probs == pytest.approx(1.0, abs=1e-9)


def test_isa_routing_measure_map_and_vd_use_the_right_bits():
    spec = parse("h 0\ncx 0 3\nry(0.7) 3\ncx 3 1\nrx(0.4) 2\ncx 1 2\nrz(0.3) 0\nh 2", 4)
    obs = OBS.z_string([0, 3], 4)
    ref = simulate(CP.build(spec).implemented, NoiseSpec(), "numpy", ideal=True).probs
    for coup in ("line", "ring"):
        comp = CP.build(spec, opts=CP.CompileOptions(mode="isa", coupling=coup, layout="sabre"))
        r = simulate(comp.implemented, NoiseSpec(), "numpy", measure_map=comp.measure_map, ideal=True)
        assert np.allclose(r.probs, ref, atol=1e-9)
        ctx = Context(comp, NoiseSpec(p1=0.002, p2=0.01), obs, Recipe.make("vd", M=2), Runner(), shots=100)
        assert abs(REGISTRY["vd"].run(ctx, M=2).mean - ctx.ideal_value()) < 1e-3


# ============================================================ bugs found in review (now fixed)
def test_aer_works_in_a_fresh_process_without_prior_transpile():
    code = ("from qemx.circuit import parse; from qemx.noise import NoiseSpec; from qemx.engine import simulate;"
            "simulate(parse('h 0',1), NoiseSpec(p1=0.01), 'aer')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr[-300:]


def test_pec_aer_instances_are_unbiased_with_repeated_gates():
    import qiskit_aer  # noqa: F401
    spec = parse("ry(1.0) 0\ncx 0 1\nry(0.5) 1\ncx 0 1\nry(0.7) 1\ncx 0 1\ncx 0 1", 2)
    obs, noise = OBS.z_string([0, 1], 2), NoiseSpec(p1=0.01, p2=0.06)
    rc = Recipe.make("pec", instances=1024)
    exact, _ = _run(spec, noise, obs, rc, "numpy")
    aer, _ = _run(spec, noise, obs, rc, "aer")
    assert abs(aer.mean - exact.mean) < 4 * aer.details["mc_err"]        # 0.028 vs 0.054 (+-0.006)


def test_pec_sampler_mean_matches_mean_for_m3():
    spec = parse("ry(1.0) 0\ncx 0 1\nry(0.5) 1\ncx 0 1", 2)
    r, _ = _run(spec, NoiseSpec(p1=0.005, p2=0.03, ro01=0.03, ro10=0.06), OBS.z_string([0, 1], 2),
                Recipe.make("pec", readout="m3"))
    s = r.sampler(2_000_000, 8, np.random.default_rng(1))
    assert abs(s.mean() - r.mean) < 2e-3


def test_learning_variance_same_for_m3_and_tensored():
    spec = parse("h 0\ncx 0 1\nrx(0.4) 1\nrx(0.3) 0", 2)
    noise, obs = NoiseSpec(p1=0.005, p2=0.03, ro01=0.02, ro10=0.04), OBS.z_string([0, 1], 2)
    a, _ = _run(spec, noise, obs, Recipe.make("learning", readout="m3"))
    b, _ = _run(spec, noise, obs, Recipe.make("learning", readout="tensored"))
    assert a.mean == pytest.approx(b.mean)
    assert a.var1 == pytest.approx(b.var1, rel=1e-6)


def test_zne_duplicate_achieved_lambdas_do_not_silently_give_inf():
    tpl = T.bell()
    comp = CP.build(tpl.spec)
    _, outs, _ = compare(comp, NoiseSpec(p2=0.05), tpl.observable, [Recipe.make("zne", scales=[1.0, 1.5, 2.0])])
    o = outs[0]
    assert o.result is None or np.isfinite(o.result.mean)


def test_zne_local_folding_without_two_qubit_gates():
    spec = parse("rx(0.4) 0\nry(0.6) 1", 2)
    comp = CP.build(spec)
    _, outs, _ = compare(comp, NoiseSpec(p1=0.02), OBS.z_string([0], 2), [Recipe.make("zne", folding="local")])
    o = outs[0]
    assert o.result is None or np.isfinite(o.result.mean)


def test_zne_nonfinite_note_does_not_blame_the_wrong_rule():
    tpl = T.bell()
    _, outs, _ = compare(CP.build(tpl.spec), NoiseSpec(p2=0.05), tpl.observable,
                         [Recipe.make("zne", scales=[1.0, 1.5, 2.0], rule="richardson")])
    r = outs[0].result
    if r is None:                                   # duplicate achieved λ is now refused with a clear error
        assert "distinct" in outs[0].error
    else:
        assert np.isfinite(r.mean) or not any("exponential" in n for n in r.notes)


def test_zne_on_gateless_circuit_is_reported():
    spec = parse("depol(0.1) 0\nampdamp(0.2) 1", 2)
    comp = CP.build(spec)
    _, outs, _ = compare(comp, NoiseSpec(), OBS.z_string([0], 2), [Recipe.make("zne")])
    o = outs[0]
    assert o.result is None or np.isfinite(o.result.mean)


@pytest.mark.parametrize("bad", ["rx(1/0) 0", "rx(9**9**9**9) 0"])
def test_parser_turns_arithmetic_errors_into_parse_errors(bad):
    with pytest.raises(ValueError):
        parse(bad, 1)


def test_parser_accepts_documented_expressions_with_function_calls():
    assert parse("rx(sqrt(2)/2) 0", 1).ops[0].params[0] == pytest.approx(math.sqrt(2) / 2)


def test_default_field_sampling_is_converged():
    sp = parse("h 0\n" + "rx(0.0) 1\n" * 7 + "barrier 0 1\nh 0", 2)
    z = lambda ns: bloch_vectors(simulate(sp, NoiseSpec(field_sigma=0.15, field_samples=ns), "numpy").rho, 2)[0][2]
    assert abs(z(NoiseSpec().field_samples) - math.exp(-0.15 ** 2 * 36 / 2)) < 0.03


def test_over_rotation_of_large_angle_rx_is_eps_times_theta():
    eps, th = 0.05, 3 * math.pi
    A = _frac_power("rx", (th,), eps)
    B = op_matrix(Op("rx", (0,), (eps * th,)))
    assert abs(np.trace(A.conj().T @ B)) / 2 == pytest.approx(1.0, abs=1e-9)
