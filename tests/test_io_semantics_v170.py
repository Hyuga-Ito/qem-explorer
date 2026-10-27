"""1.7.0: the Playground keeps four things apart — A the input state, B the ideal output state before measurement
(independent of the measurement basis), C the distribution of the measured bits in the selected bases, D the
observable on those bits — and says when a template's identity is a prediction for the current experiment.

Mathematical checks use an independent statevector implementation (numpy only); the Streamlit checks drive the page."""
import math

import numpy as np
import pytest

from qemx import io_semantics as IO
from qemx import observables as OBS
from qemx import sweep as SW
from qemx.circuit import CircuitSpec, Op
from qemx.compile import input_state

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
SDG = np.diag([1, -1j])
ROT = {"Z": np.eye(2), "X": H, "Y": H @ SDG}               # R_b: rotates basis b to the computational basis


def _g(name, *q, p=()):
    return Op(name, tuple(q), tuple(p), "gate", "")


def _apply1(psi, U, q, n):
    t = psi.reshape([2] * n)
    ax = n - 1 - q
    t = np.moveaxis(np.tensordot(U, np.moveaxis(t, ax, 0), axes=(1, 0)), 0, ax)
    return t.reshape(-1)


def _ref_state(body, inputs):
    """Independent: product input, then the body's gates (h, x, cx, rz, ry only)."""
    n = body.n
    psi = np.array([1.0 + 0j])
    for q in reversed(range(n)):                             # little endian: q0 is the last tensor factor
        psi = np.kron(psi, input_state(inputs[q]))
    for o in body.ops:
        if o.name == "h":
            psi = _apply1(psi, H, o.qubits[0], n)
        elif o.name == "x":
            psi = _apply1(psi, np.array([[0, 1], [1, 0]]), o.qubits[0], n)
        elif o.name == "ry":
            th = o.params[0]
            psi = _apply1(psi, np.array([[math.cos(th / 2), -math.sin(th / 2)], [math.sin(th / 2), math.cos(th / 2)]]),
                          o.qubits[0], n)
        elif o.name == "rz":
            th = o.params[0]
            psi = _apply1(psi, np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)]), o.qubits[0], n)
        elif o.name == "cx":
            c, t = o.qubits
            idx = np.arange(2 ** n)
            flip = ((idx >> c) & 1) == 1
            out = psi.copy()
            out[idx[flip]] = psi[idx[flip] ^ (1 << t)]
            psi = out
        else:
            raise NotImplementedError(o.name)
    return psi


def _ref_distribution(body, inputs, basis, relabel):
    n = body.n
    psi = _ref_state(body, inputs)
    for q in range(n):
        psi = _apply1(psi, ROT[basis[q]], q, n)
    p = np.abs(psi) ** 2
    out = np.zeros(2 ** n)
    for x in range(2 ** n):                                   # bit i ← qubit mm[i]
        z = 0
        for i in range(n):
            q = n - 1 - i if relabel else i
            z |= ((x >> q) & 1) << i
        out[z] += p[x]
    return out


def _dist(d, n):
    v = np.zeros(2 ** n)
    for b, pr in d["top"]:
        v[int(b, 2)] = pr
    return v


# ------------------------------------------------------------------ bulk input syntax
@pytest.mark.parametrize("text,expected", [
    ("plus", ["+"] * 3), ("+", ["+"] * 3), ("minus", ["-"] * 3), ("-", ["-"] * 3), ("0", ["0"] * 3),
    ("1", ["1"] * 3), ("+i", ["+i"] * 3), ("-i", ["-i"] * 3), ("ph:0.25", ["ph:0.25"] * 3),
    ("0 + 1", ["0", "+", "1"]), ("bits:011", ["1", "1", "0"]), ("bits:100", ["0", "0", "1"])])
def test_every_bulk_input_syntax_and_alias_resolves_to_per_qubit_tokens(text, expected):
    assert SW.input_tokens(text, 3) == expected


@pytest.mark.parametrize("text", ["pluss", "0 +", "bits:01", "bits:0121", "ramp:x", "bloch:1", "+ + + +", ""])
def test_invalid_bulk_inputs_are_refused_never_guessed(text):
    with pytest.raises(SW.SweepError):
        SW.input_tokens(text, 3)


def test_ramp_input_is_the_fourier_state_of_k():
    toks = SW.input_tokens("ramp:3", 3)
    psi = np.array([1.0 + 0j])
    for q in reversed(range(3)):
        psi = np.kron(psi, input_state(toks[q]))
    N = 8
    want = np.array([np.exp(-2j * math.pi * 3 * j / N) for j in range(N)]) / math.sqrt(N)
    assert abs(abs(np.vdot(want, psi)) - 1) < 1e-12


# ------------------------------------------------------------------ B does not depend on the basis; C does
CASES = [
    ("identity", CircuitSpec(2, (), "id"), ["+", "+"]),
    ("hadamard", CircuitSpec(2, (_g("h", 0), _g("h", 1)), "h"), ["0", "1"]),
    ("bell", CircuitSpec(2, (_g("h", 0), _g("cx", 0, 1)), "bell"), ["0", "0"]),
    ("ghz3", CircuitSpec(3, (_g("h", 0), _g("cx", 0, 1), _g("cx", 1, 2)), "ghz"), ["0", "0", "0"]),
    ("rot", CircuitSpec(3, (_g("ry", 0, p=(0.7,)), _g("cx", 0, 2), _g("rz", 2, p=(1.1,)), _g("h", 1)), "rot"),
     ["+i", "1", "ph:0.125"]),
]


@pytest.mark.parametrize("name,body,inputs", CASES, ids=[c[0] for c in CASES])
def test_output_state_is_basis_independent_and_measurement_matches_the_definition(name, body, inputs):
    n = body.n
    B = IO.output_state(body, inputs)
    ref = _ref_state(body, inputs)
    assert B["status"] == "exact"
    for q, d in enumerate(B["qubits"]):
        r = np.asarray(d["bloch"])
        t = ref.reshape([2] * n)
        t = np.moveaxis(t, n - 1 - q, 0).reshape(2, -1)
        rho = t @ t.conj().T
        assert np.allclose(r, [2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real], atol=1e-10)
    for basis in ("Z" * n, "X" * n, "Y" * n, ("XZY" * 3)[:n]):
        for relabel in (False, True):
            C = IO.measurement_distribution(body, inputs, basis, relabel, top=2 ** n)
            assert C["status"] == "exact"
            assert np.allclose(_dist(C, n), _ref_distribution(body, inputs, basis, relabel), atol=1e-12), (basis,
                                                                                                        relabel)
    assert IO.output_state(body, inputs) == B                                  # no hidden state; pure function


def test_plus_inputs_on_the_identity_are_uniform_in_z_and_deterministic_in_x():
    body = CircuitSpec(3, (), "id")
    B = IO.output_state(body, ["+"] * 3)
    assert B["product"] and B["text"] == "Product state: |+⟩ on every qubit"
    cz = IO.measurement_distribution(body, ["+"] * 3, "ZZZ", False)
    cx = IO.measurement_distribution(body, ["+"] * 3, "XXX", False)
    assert cz["uniform"] and not cz["deterministic"]
    assert cx["deterministic"] and cx["top"][0][0] == "000"


def test_entangled_outputs_are_reported_as_distributions_not_one_bitstring():
    bell = CASES[2][1]
    B = IO.output_state(bell, ["0", "0"])
    assert B["product"] is False and "Entangled" in B["text"]
    C = IO.measurement_distribution(bell, ["0", "0"], "ZZ", False)
    assert not C["deterministic"] and sorted(b for b, _ in C["top"]) == ["00", "11"]
    assert "A distribution over 2 outcomes" in IO.output_summary(C)


def test_bit_reversal_changes_the_bits_not_the_state():
    body = CircuitSpec(3, (_g("x", 0),), "x0")                 # |q2 q1 q0⟩ = |001⟩
    assert IO.output_state(body, ["0"] * 3)["text"] == "Product state |001⟩ (q2 … q0)"
    assert IO.measurement_distribution(body, ["0"] * 3, "ZZZ", False)["top"][0][0] == "001"
    assert IO.measurement_distribution(body, ["0"] * 3, "ZZZ", True)["top"][0][0] == "100"


# ------------------------------------------------------------------ the QFT case of the report
def test_qft_fourier_input_with_s0_already_is_plus_and_applying_plus_changes_nothing():
    tokens, body, tpl = IO.template_inputs("qft", dict(n=4, s=0, mode="relabel", prep="fourier"))
    plus = SW.input_tokens("plus", 4)
    assert IO.same_inputs(tokens, plus)                        # F†|0⟩ = |+⟩^⊗4: physically the same input
    msg = IO.apply_summary(tokens, plus, "ZZZZ", "ZZZZ", True, False)
    assert msg.startswith("inputs unchanged") and "output does not change" in msg
    assert IO.output_state(body, plus)["text"] == "Product state: |0⟩ on every qubit"
    assert IO.measurement_distribution(body, plus, "ZZZZ", True, tpl.observable)["observable_value"] == \
        pytest.approx(1.0)
    ok, why = IO.template_claim(True, True, "ZZZZ", True, True)
    assert ok and not why


def test_qft_computational_input_is_a_product_state_with_phases():
    tokens, body, tpl = IO.template_inputs("qft", dict(n=3, s=3, mode="swap", prep="basis"))
    B = IO.output_state(body, tokens)
    assert B["product"] and all(abs(d["bloch"][2]) < 1e-9 for d in B["qubits"])     # every qubit on the equator
    C = IO.measurement_distribution(body, tokens, "ZZZ", False)
    assert C["uniform"]


def test_basis_change_is_declared_as_a_change_of_the_physical_observable():
    tokens, body, tpl = IO.template_inputs("qft", dict(n=4, s=0, mode="relabel", prep="fourier"))
    z = IO.observable_meaning(tpl.observable, "ZZZZ", True, 4)
    x = IO.observable_meaning(tpl.observable, "XXXX", True, 4)
    assert "computational-basis projector onto the qubit state |0000⟩" in z
    assert "projector onto |++++⟩" in x and "changed the physical observable" in x
    ok, why = IO.template_claim(True, True, "XXXX", True, True)
    assert not ok and "measurement bases" in why[0]
    C = IO.measurement_distribution(body, tokens, "XXXX", True, tpl.observable)
    assert C["uniform"] and C["observable_value"] == pytest.approx(1 / 16)
    assert IO.output_state(body, tokens) == IO.output_state(body, tokens)             # B unchanged by the basis


def test_template_claim_lists_every_broken_assumption():
    ok, why = IO.template_claim(False, False, "XZ", False, True)
    assert not ok and len(why) == 3
    ok, why = IO.template_claim(True, False, "ZZ", True, True)
    assert not ok and why == ["the inputs differ from the template's input"]


def test_experiment_key_changes_with_every_part_of_the_experiment():
    body = CASES[3][1]
    obs = OBS.projector(0, 3)
    base = IO.experiment_key(body, ["0"] * 3, "ZZZ", False, obs)
    assert IO.experiment_key(body, ["0"] * 3, "ZZZ", False, obs) == base
    assert len({base, IO.experiment_key(body, ["+", "0", "0"], "ZZZ", False, obs),
                IO.experiment_key(body, ["0"] * 3, "XZZ", False, obs),
                IO.experiment_key(body, ["0"] * 3, "ZZZ", True, obs),
                IO.experiment_key(body, ["0"] * 3, "ZZZ", False, OBS.projector(1, 3)),
                IO.experiment_key(CASES[4][1], ["0"] * 3, "ZZZ", False, obs)}) == 6


def test_large_registers_report_not_computed_instead_of_an_answer():
    body = CircuitSpec(4, (), "id")
    B = IO.output_state(body, ["0"] * 4, max_qubits=3)
    C = IO.measurement_distribution(body, ["0"] * 4, "ZZZZ", False, max_qubits=3)
    assert B["status"] == "not computed" and B["reason"] and C["status"] == "not computed"


# ------------------------------------------------------------------ the page
def _playground():
    import os
    try:
        import qiskit_ibm_runtime  # noqa: F401  (see tests/test_ui.py)
    except Exception:                                        # pragma: no cover
        pass
    from streamlit.testing.v1 import AppTest
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=900).run()
    at.switch_page("pages/1_Playground.py").run()
    return at


def _w(at, kind, key):
    return next(x for x in getattr(at, kind) if x.key == key)


def _text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]).replace("\\", "")


def _load_qft(at, **params):
    _w(at, "selectbox", "w_tpl_cat").set_value("Algorithms").run()
    _w(at, "selectbox", "w_tpl").set_value("qft").run()
    for k, v in params.items():
        w = [x for x in list(at.number_input) + list(at.selectbox) if x.key == "w_tp_qft_%s" % k]
        if w:
            w[0].set_value(v).run()
    _w(at, "button", "tpl_load").click().run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.slow
def test_playground_bulk_plus_on_the_qft_fourier_input_is_reported_as_unchanged_and_basis_change_as_such():
    at = _playground()
    _load_qft(at, n=4, s=0, mode="relabel")
    ss = at.session_state
    tokens0 = list(ss["inputs"])
    assert "Prediction for this experiment" in _text(at)
    assert "B · Output state before measurement" in _text(at) and "|0⟩ on every qubit" in _text(at)
    # typed but not applied: the page says so, nothing changes
    _w(at, "text_input", "w_io_all_in").set_value("plus").run()
    assert "Not applied yet" in _text(at) and list(ss["inputs"]) == tokens0
    _w(at, "button", "io_all_apply").click().run()
    assert "inputs unchanged" in _text(at) and "Not applied yet" not in _text(at)
    assert ss["basis"] == "ZZZZ" and IO.same_inputs(ss["inputs"], tokens0)
    assert "Prediction for this experiment" in _text(at)
    # a basis change: the output state stays, the measured bits and the observable's meaning change
    _w(at, "text_input", "w_io_all_in").set_value("").run()
    _w(at, "text_input", "w_io_all_bs").set_value("X").run()
    _w(at, "button", "io_all_apply").click().run()
    t = _text(at)
    assert ss["basis"] == "XXXX" and IO.same_inputs(ss["inputs"], tokens0)            # the input is untouched
    assert "|0⟩ on every qubit" in t and "Uniform over all 16 outcomes" in t
    assert "Template property — not a prediction" in t and "changed the physical observable" in t
    # a per-qubit input edit leaves the bases alone
    _w(at, "selectbox", "w_in_0").set_value("1").run()
    assert ss["basis"] == "XXXX" and ss["inputs"][0] == "1"
    assert "Template property — not a prediction" in _text(at)
    # results computed now are marked out of date by a basis change
    at.session_state["run_requested"] = True
    at.run()
    assert at.session_state["results"] is not None and "marked out of date" not in _text(at)
    _w(at, "selectbox", "w_bs_1").set_value("Z").run()
    assert "marked out of date" in _text(at) and ss["inputs"][0] == "1"
    # reloading the template restores its input, bases and claim; switching pages keeps the experiment
    _load_qft(at, n=4, s=0, mode="relabel")
    assert ss["basis"] == "ZZZZ" and "Prediction for this experiment" in _text(at)
    at.switch_page("pages/6_Theory.py").run()
    at.switch_page("pages/1_Playground.py").run()
    assert not at.exception and ss["basis"] == "ZZZZ" and "Prediction for this experiment" in _text(at)


@pytest.mark.slow
def test_playground_custom_entangled_circuit_has_a_distribution_and_no_template_claim():
    at = _playground()
    _load_qft(at, n=3, s=1, mode="swap")
    exp = [x for x in at.expander if "Type the circuit as text" in str(getattr(x, "label", ""))]
    txt = [x for x in at.text_area if x.key and x.key.startswith("w_txt_")][0]
    txt.set_value("h 0; cx 0 1; cx 1 2").run()
    _w(at, "button", "txt_apply").click().run()
    assert not at.exception, [e.value for e in at.exception]
    t = _text(at)
    assert at.session_state["inputs"] == IO.template_inputs("qft", dict(n=3, s=1, mode="swap"))[0]   # kept
    assert "Entangled state" in t and "A distribution over 8 outcomes" in t
    assert "Template property — not a prediction" in t and "circuit was edited" in t
    _w(at, "text_input", "w_io_all_in").set_value("0").run()
    _w(at, "button", "io_all_apply").click().run()
    t = _text(at)
    assert "inputs: 3 of 3 qubits changed" in t and "Entangled state" in t
    assert "A distribution over 2 outcomes" in t and "|000⟩ 0.500" in t
    del exp
