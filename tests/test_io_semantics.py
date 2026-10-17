"""1.5: input/output semantics. The input (per-qubit states + template preparation) is the editable source of truth;
the output is derived and read-only. QFT conventions are checked against the analytic transform
F|y⟩ = N^{-1/2} Σ_k e^{2πi yk/N}|k⟩ (little-endian integers), with two independent simulators (the app's NumPy
statevector and Qiskit's Statevector)."""
import itertools
import math

import numpy as np
import pytest

from qemx import compile as CP
from qemx import io_semantics as IO
from qemx import templates as T
from qemx.circuit import CircuitSpec


def _F(n):
    N = 2 ** n
    k = np.arange(N)
    return np.exp(2j * np.pi * np.outer(k, k) / N) / math.sqrt(N)


def _same_up_to_phase(a, b, tol=1e-10):
    i = int(np.argmax(np.abs(b)))
    ph = a[i] / b[i]
    return abs(abs(ph) - 1) < tol and np.allclose(a, ph * b, atol=tol)


def _qiskit_sv(spec):
    from qiskit.quantum_info import Statevector
    from qemx.circuit import to_qiskit
    return np.asarray(Statevector(to_qiskit(spec)).data)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_qft_of_computational_basis_inputs_matches_the_analytic_transform(n):
    F = _F(n)
    for x in range(2 ** n):
        tpl = T.qft(n, x, "swap", None, "basis")
        psi = T.statevector(n, tpl.spec.ops)
        assert _same_up_to_phase(psi, F[:, x]), (n, x)
        assert _same_up_to_phase(_qiskit_sv(tpl.spec), F[:, x]), (n, x)      # independent simulator


@pytest.mark.parametrize("n", [2, 3, 5])
def test_basis_inputs_give_uniform_output_with_input_dependent_phases(n):
    N = 2 ** n
    phases = []
    for x in (0, 1, N - 1):
        tpl = T.qft(n, x, "swap", None, "basis")
        comp = CP.build(tpl.spec, relabel=tpl.relabel)
        d = IO.derived_output(comp.logical, tpl.relabel, tpl.observable, top=N)
        assert d["available"] and d["support"] == N and not d["deterministic"]
        assert all(pr == pytest.approx(1 / N, abs=1e-12) for _, pr in d["top"])
        assert d["entropy_bits"] == pytest.approx(n, abs=1e-9)
        assert d["observable_value"] == pytest.approx(1 / N, abs=1e-12)
        psi = T.statevector(n, tpl.spec.ops)
        phases.append(psi / psi[0])
    assert not np.allclose(phases[0], phases[1]) and not np.allclose(phases[1], phases[2])
    assert "Uniform" in IO.output_summary(d)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_plus_state_maps_to_all_zeros(n):
    tpl = T.qft(n, 0, "swap")                      # F†|0⟩ = |+⟩^n: the preparation is one H per qubit
    prep = IO.preparation_ops(tpl.spec)
    assert [o.name for o in prep] == ["h"] * n
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    d = IO.derived_output(comp.logical, tpl.relabel, tpl.observable)
    assert d["deterministic"] and d["top"][0][0] == "0" * n and d["observable_value"] == pytest.approx(1.0)
    # the same through the per-qubit input selector on the bare QFT (no template preparation)
    body = CircuitSpec(n, tuple(T.qft_body(list(range(n)))) + tuple(T.swap_network(list(range(n)))), "F")
    comp2 = CP.build(body, ["+"] * n, "Z" * n)
    d2 = IO.derived_output(comp2.logical)
    assert d2["deterministic"] and d2["top"][0][0] == "0" * n


@pytest.mark.parametrize("n,mode", list(itertools.product([2, 3, 4], ["swap", "bitreverse", "relabel"])))
def test_fourier_prep_known_answer_in_every_bit_order(n, mode):
    for s in range(2 ** n):
        tpl = T.qft(n, s, mode)
        comp = CP.build(tpl.spec, relabel=tpl.relabel)
        d = IO.derived_output(comp.logical, tpl.relabel, tpl.observable)
        assert d["deterministic"], (n, mode, s)
        assert int(d["top"][0][0], 2) == s, (n, mode, s, d["top"][0])      # bitstring c(n-1)…c0 = s
        assert d["observable_value"] == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_inverse_qft_round_trip_is_identity(n):
    q = list(range(n))
    rng = np.random.default_rng(n)
    ops = tuple(T.qft_body(q)) + tuple(T.qft_body(q, inverse=True))
    U = np.stack([T.statevector(n, (tuple(T.basis_prep(n, x)) + ops)) for x in range(2 ** n)], axis=1)
    assert np.allclose(U, np.eye(2 ** n), atol=1e-10)
    # F† F|ψ⟩ for a random state built from rotations
    rot = tuple(T._g("ry", k, params=(float(a),)) for k, a in zip(q, rng.uniform(0, 3, n)))
    psi = T.statevector(n, rot)
    assert np.allclose(T.statevector(n, rot + ops), psi, atol=1e-10)


def test_swap_network_reverses_bits():
    n = 3
    for x in range(8):
        spec = CircuitSpec(n, tuple(T.basis_prep(n, x)) + tuple(T.swap_network(list(range(n)))), "R")
        d = IO.derived_output(spec)
        assert d["top"][0][0] == format(x, "03b")[::-1]
        d_rel = IO.derived_output(CircuitSpec(n, tuple(T.basis_prep(n, x)), "id"), relabel=True)
        assert d_rel["top"][0][0] == format(x, "03b")[::-1]      # classical relabeling reads the same reversed bits


def test_measurement_basis_changes_the_derived_output_not_the_input():
    n = 3
    tpl = T.qft(n, 0, "swap", None, "basis")            # F|0⟩ = |+⟩^n
    z = IO.derived_output(CP.build(tpl.spec, None, "Z" * n).logical)
    x = IO.derived_output(CP.build(tpl.spec, None, "X" * n).logical)
    assert not z["deterministic"] and z["support"] == 8
    assert x["deterministic"] and x["top"][0][0] == "000"


def test_deriving_the_output_never_mutates_the_input():
    n = 3
    tpl = T.qft(n, 5, "swap")
    inputs = ["0", "+", "1"]
    comp = CP.build(tpl.spec, inputs, "ZXZ")
    before = (list(inputs), comp.logical, tpl.spec)
    for _ in range(3):
        IO.derived_output(comp.logical, False, tpl.observable)
        IO.describe_input(tpl.spec, inputs)
    assert (list(inputs), comp.logical, tpl.spec) == before


def test_bitreverse_is_refused_for_basis_inputs():
    with pytest.raises(ValueError, match="bit-reversed"):
        T.qft(3, 1, "bitreverse", None, "basis")
    with pytest.raises(ValueError):
        T.qft(3, 1, "swap", None, "sideways")


def test_legacy_qft_parameters_mean_the_fourier_preparation():
    old = T.clamp_params("qft", dict(n=3, s=5, mode="swap"))         # a 1.4.2 parameter set has no "prep"
    assert old["prep"] == "fourier"
    assert T.build("qft", old).spec == T.qft(3, 5, "swap").spec      # same circuit as before 1.5
    from qemx.phase.families import qft as fam_qft
    assert fam_qft(4).spec == T.qft(4, 1, "relabel").spec            # the phase-study family is unchanged


def test_preview_reports_unavailable_instead_of_inventing_an_answer():
    n = 5
    tpl = T.qft(n, 3, "swap")
    d = IO.derived_output(CP.build(tpl.spec).logical, max_qubits=4)
    assert not d["available"] and "amplitudes" in d["reason"] and d["top"] == []
    assert IO.output_summary(d).startswith("Ideal output not computed")


def test_aqft_index_convention_against_coppersmith():
    """The app's AQFT degree m keeps controlled phases π/2^d with d ≤ m; Coppersmith deletes d ≥ m_C (m = m_C − 1)."""
    n = 5
    ops = T.qft_body(list(range(n)), m=2)
    d_kept = {abs(o.qubits[0] - o.qubits[1]) for o in ops if o.name == "cp"}
    assert d_kept == {1, 2}
    assert all(o.name != "cp" for o in T.qft_body(list(range(n)), m=0))   # m_C = 1: the Hadamard transform


# ------------------------------------------------------------------ the Playground: Output is read-only and derived
@pytest.mark.slow
def test_playground_output_panel_is_derived_and_never_changes_the_input():
    import os
    try:
        import qiskit_ibm_runtime  # noqa: F401  (see tests/test_ui.py)
    except Exception:                                        # pragma: no cover
        pass
    from streamlit.testing.v1 import AppTest
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=900).run()
    at.switch_page("pages/1_Playground.py").run()

    def w(kind, key):
        return next(x for x in getattr(at, kind) if x.key == key)

    def text():
        return " ".join(m.value for m in at.markdown).replace("\\", "")      # undo Markdown escaping
    w("selectbox", "w_tpl_cat").set_value("Algorithms").run()
    w("selectbox", "w_tpl").set_value("qft").run()
    w("number_input", "w_tp_qft_s").set_value(0).run()
    w("button", "tpl_load").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Deterministic: |000⟩" in text()
    ops0, inputs0 = [dict(d) for d in at.session_state["ops"]], list(at.session_state["inputs"])
    # editing the input parameter without loading, rerunning and switching tabs change nothing
    w("number_input", "w_tp_qft_s").set_value(3).run()
    at.run()
    assert [dict(d) for d in at.session_state["ops"]] == ops0 and list(at.session_state["inputs"]) == inputs0
    assert not any(x.key and x.key.startswith("w_out") for x in list(at.selectbox) + list(at.number_input)
               + list(at.text_input))
    # a basis-state input: the derived output becomes a uniform distribution
    w("selectbox", "w_tp_qft_prep").set_value("basis").run()
    w("button", "tpl_load").click().run()
    assert "Uniform over all 8 outcomes" in text()
    assert list(at.session_state["inputs"]) == ["0", "0", "0"]
    # the cited preset selector fills the controls, Load builds it
    w("selectbox", "w_cited_preset").set_value("qaoa-ring-p1").run()
    assert at.session_state["w_tpl"] == "qaoa" and at.session_state["w_tp_qaoa_angles"] == "fixed"
    w("button", "tpl_load").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Ideal ⟨O⟩ = 0.75" in text()
