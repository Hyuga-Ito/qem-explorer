"""1.5.1: one input model for every template. The input is one single-qubit state per qubit, set in one place
(Inputs and measurement), prepared exactly once by the compiler; template parameters with role "input" only choose
those states; the output is derived from input, circuit, basis and bit order and never edited."""
import math

import numpy as np
import pytest

from qemx import compile as CP
from qemx import io_semantics as IO
from qemx import templates as T

KEYS = sorted(T.REGISTRY)
VARIANTS = {
    "qft": [dict(n=n, s=s, mode=m, prep=p) for n in (2, 3, 4) for s in (0, 1, 2 ** n - 1, 2 ** n // 3)
            for m in ("swap", "bitreverse", "relabel") for p in ("fourier", "basis") if not (m == "bitreverse" and p == "basis")],
    "aqft": [dict(n=4, s=s, mode=m, m=mm) for s in (0, 5, 11) for m in ("swap", "bitreverse", "relabel") for mm in (1, 2)],
    "qpe": [dict(t=t, phi=ph) for t in (1, 3) for ph in (0.25, 0.3)],
    "qpe_ising": [dict(t=t, state=s) for t in (1, 2) for s in ("00", "01", "10", "11")],
    "bv": [dict(n=n, s=s) for n in (1, 3) for s in (0, 1)],
    "shor": [dict(a=a, t=2) for a in (2, 7, 11, 14)],
    "qec_3q": [dict(code=c, theta=th) for c in ("bit-flip", "phase-flip") for th in (0.0, 1.0472, 4.0)],
    "repetition": [dict(d=3, rounds=r, theta=th) for r in (0, 1) for th in (0.0, 1.2, 5.0)],
    "steane": [dict(theta=th) for th in (0.0, 0.7, 4.0)],
    "five_qubit": [dict(theta=th) for th in (0.0, 2.0)],
    "surface_d2": [dict(theta=th, syndrome=sy) for th in (0.0, 1.0472)
                   for sy in ("transversal readout", "ancilla extraction (3 ancillas)")],
}


def _cases():
    for k in KEYS:
        yield k, T.defaults(k)
        for p in VARIANTS.get(k, []):
            yield k, p


def _sv(spec):
    return T.statevector(spec.n, spec.ops)


def _same_state(a, b, tol=1e-9):
    return abs(abs(np.vdot(a, b)) - 1) < tol


# ------------------------------------------------------------------ tokens
def test_every_pure_state_has_one_canonical_token_that_prepares_it():
    rng = np.random.default_rng(5)
    for _ in range(200):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        tok = CP.state_token(v)
        assert CP.valid_input(tok) and _same_state(CP.input_state(tok), v / np.linalg.norm(v))
        assert CP.state_token(CP.input_state(tok)) == tok                      # canonical: a fixed point
    for tok in CP.INPUT_STATES:
        assert CP.state_token(CP.input_state(tok)) == tok
    assert CP.state_token(CP.input_state("ph:0.25")) == "+i" and CP.bloch_token(math.pi) == "1"
    assert CP.same_input(CP.bloch_token(4.0), "bloch:2.28318530718,3.14159265359")


@pytest.mark.parametrize("bad", ["bloch:1", "bloch:1,0,0", "bloch:1;0", "bloch:(1),0", "bloch:1,0 ", "ph:x",
                                 "bloch:__import__('os')", "BLOCH:1,0", "bloch:1e999999,0", "", None, 3])
def test_input_tokens_are_strictly_validated(bad):
    assert not CP.valid_input(bad)
    if isinstance(bad, str):
        with pytest.raises(ValueError):
            CP.input_ops(bad)


# ------------------------------------------------------------------ templates: inputs prepared exactly once
@pytest.mark.parametrize("key,params", list(_cases()))
def test_template_input_is_prepared_once_and_the_circuit_keeps_the_rest(key, params):
    tokens, body, tpl = IO.template_inputs(key, params)
    assert len(tokens) == body.n == tpl.spec.n and all(CP.valid_input(t) for t in tokens)
    assert not IO.preparation_ops(body)                                         # nothing left to prepare twice
    assert IO.split_preparation(body, tokens) == (tokens, body)                  # idempotent
    assert tuple(body.ops) == tuple(tpl.spec.ops[len(tpl.spec.ops) - len(body.ops):])
    if tpl.spec.n <= 12:
        assert _same_state(_sv(CP.with_io(body, tokens)), _sv(tpl.spec))         # the template's own state


def _gates(spec):
    """Gate list with angles to 1e-9; a phase gate's angle modulo 2π (P(−π) = P(π))."""
    def ang(g, x):
        if g != "p":
            return round(float(x), 9)
        a = float(x) % (2 * math.pi)
        return 0.0 if abs(a - 2 * math.pi) < 1e-9 else round(a, 9)
    return [(o.name, tuple(o.qubits), tuple(ang(o.name, x) for x in o.params), o.kind) for o in spec.ops]


@pytest.mark.parametrize("key,params", list(_cases()))
def test_moving_the_preparation_into_the_inputs_keeps_every_compiled_gate(key, params):
    """Not only the same state: the same gates, angle for angle, so a noise model sees the circuit it saw when the
    preparation was written into the template (the QFT's H·P(π) stays ph:0.5, not the canonical |−⟩ = H·X)."""
    tokens, body, tpl = IO.template_inputs(key, params)
    assert _gates(CP.with_io(body, tokens)) == _gates(CP.with_io(tpl.spec, ["0"] * tpl.spec.n))


def test_faithful_tokens_reproduce_the_written_gates_or_fall_back_to_the_canonical_state():
    def tok(seq):
        v = np.array([1, 0], complex)
        from qemx.circuit import gate_matrix
        for g, p in seq:
            v = np.asarray(gate_matrix(g, p), complex) @ v
        return IO.faithful_token(seq, v)
    assert tok([("h", ()), ("p", (math.pi,))]) == "ph:0.5"                     # not "-" (= X·H)
    assert tok([("x", ()), ("h", ())]) == "-"
    assert tok([("h", ()), ("p", (-math.pi / 2,))]) == "ph:0.75"               # a phase angle modulo 2π
    assert tok([("h", ())]) == "+" and tok([("x", ())]) == "1"
    assert tok([("ry", (1.0,))]) == "bloch:1,0"
    assert tok([("h", ()), ("h", ()), ("x", ())]) == "1"                       # no token writes these gates
    assert tok([("ry", (0.4,)), ("p", (0.3,))]) == "bloch:0.4,0.3"


@pytest.mark.parametrize("key", [k for k in KEYS if IO.input_params(k)])
def test_input_parameters_never_change_the_circuit(key):
    base = T.defaults(key)
    bodies, inputs = set(), set()
    grid = {"s": [0, 1, 3], "prep": ["fourier", "basis"], "state": ["00", "01", "10", "11"],
            "theta": [0.0, 0.5, 2.0, 4.5]}
    names = [p.name for p in IO.input_params(key)]
    for combo in __import__("itertools").product(*[grid[n] for n in names]):
        p = dict(base, **dict(zip(names, combo)))
        try:
            tokens, body, _ = IO.template_inputs(key, p)
        except ValueError:
            continue
        bodies.add(body.key())
        inputs.add(tuple(tokens))
    assert len(bodies) == 1 and len(inputs) > 1


@pytest.mark.parametrize("key,params", [(k, p) for k, p in _cases() if IO.input_params(k)])
def test_input_parameters_are_recovered_from_the_inputs(key, params):
    tokens, _, _ = IO.template_inputs(key, params)
    found = IO.infer_template_input(key, params, tokens)
    assert found is not None
    want = T.clamp_params(key, params)
    for k, v in found.items():
        assert (v == pytest.approx(want[k], abs=1e-9)) if isinstance(v, float) else v == want[k], (k, v, want[k])
    assert IO.same_inputs(IO.template_inputs(key, dict(want, **found))[0], tokens)


def test_edited_inputs_match_no_preset_and_a_ramp_matches_the_fourier_input():
    from qemx import sweep as SW
    assert IO.infer_template_input("qft", dict(n=3, s=5), ["0", "+", "1"]) is None
    assert IO.infer_template_input("qft", dict(n=3, s=5), SW.input_tokens("ramp:3", 3)) == dict(s=3, prep="fourier")
    assert IO.infer_template_input("qft", dict(n=3, mode="bitreverse"), SW.input_tokens("ramp:3", 3)) == \
        dict(s=T.rev(3, 3), prep="fourier")
    assert IO.infer_template_input("steane", {}, ["+"] + ["0"] * 6) is None          # θ sits on q2
    assert IO.infer_template_input("qec_3q", {}, ["+i"] + ["0"] * 4) is None          # not a real RY state
    assert IO.infer_template_input("bv", {}, ["0"] * 5) is None                       # no input parameters


def test_qft_output_is_derived_from_the_actual_input_in_every_bit_order():
    for n in (2, 3, 4):
        for mode in ("swap", "bitreverse", "relabel"):
            for s in range(2 ** n):
                tokens, body, tpl = IO.template_inputs("qft", dict(n=n, s=s, mode=mode))
                d = IO.derived_output(CP.with_io(body, tokens), tpl.relabel, tpl.observable)
                assert d["deterministic"] and int(d["top"][0][0], 2) == s
                assert d["observable_value"] == pytest.approx(1.0, abs=1e-10)
                # another input through the same circuit: the output follows the input, nothing else changes
                other = ["0"] * n
                d2 = IO.derived_output(CP.with_io(body, other), tpl.relabel)
                assert d2["deterministic"] is False or n == 1
    tokens, body, tpl = IO.template_inputs("qft", dict(n=3, s=6, prep="basis"))
    d = IO.derived_output(CP.with_io(body, tokens), tpl.relabel, tpl.observable, top=8)
    assert d["support"] == 8 and "Uniform" in IO.output_summary(d)


@pytest.mark.parametrize("key", KEYS)
def test_derived_output_of_every_family_equals_the_template_circuit(key):
    tokens, body, tpl = IO.template_inputs(key, T.defaults(key))
    if tpl.spec.n > 12:
        pytest.skip("large register")
    a = IO.derived_output(CP.with_io(body, tokens, "Z" * body.n), tpl.relabel, tpl.observable, top=16)
    b = IO.derived_output(CP.with_io(tpl.spec, None, "Z" * body.n), tpl.relabel, tpl.observable, top=16)
    assert [x for x, _ in a["top"]] == [x for x, _ in b["top"]]
    assert np.allclose([p for _, p in a["top"]], [p for _, p in b["top"]], atol=1e-10)
    if a["observable_value"] is not None:
        assert a["observable_value"] == pytest.approx(b["observable_value"], abs=1e-10)


def test_encoded_algorithms_keep_their_encoder_in_the_circuit():
    tokens, body, tpl = IO.template_inputs("qec_algorithm", T.defaults("qec_algorithm"))
    assert all(t == "0" for t in tokens) and body is tpl.spec and not IO.input_params("qec_algorithm")
    assert any(o.tag == "encode" for o in body.ops)


# ------------------------------------------------------------------ documents, batches, legacy states
def test_experiment_short_form_uses_the_template_input_and_refuses_a_second_one():
    from qemx import experiment as E
    doc = dict(schema=E.SCHEMA, circuit=dict(template="qft", params=dict(n=3, s=5)), methods=[dict(preset="Unmitigated")])
    st = E.to_state(doc)
    tokens, body, _ = IO.template_inputs("qft", dict(n=3, s=5))
    assert st["inputs"] == tokens and not any(o.get("tag") == "prep" for o in st["circuit"]["ops"])
    assert E.to_state(dict(doc, inputs=["0"] * 3))["inputs"] == tokens              # |0⟩: the template's input
    assert E.to_state(dict(doc, inputs=list(tokens)))["inputs"] == tokens
    with pytest.raises(E.ExperimentError) as ei:
        E.to_state(dict(doc, inputs=["+", "+", "+"]))
    assert ei.value.path == "inputs"
    plain = dict(schema=E.SCHEMA, circuit=dict(template="ghz", params=dict(n=3)), inputs=["+", "0", "1"])
    assert E.to_state(plain)["inputs"] == ["+", "0", "1"]                         # no template input: as given


def test_batch_register_input_replaces_the_template_input():
    from qemx import batch as B
    a = B._algorithm("qft", None, {}, (), {})
    tokens, body, _ = IO.template_inputs("qft", T.defaults("qft"))
    assert a["inputs"] == tokens and a["spec"] == body
    b = B._algorithm("qft", None, {}, (("input", "ramp:2"),), {"input": "input"})
    from qemx import sweep as SW
    assert b["inputs"] == SW.input_tokens("ramp:2", body.n) and b["spec"] == body


def test_a_state_saved_before_1_5_1_keeps_its_state_with_the_preparation_moved_into_the_inputs():
    tpl = T.build("qft", dict(n=3, s=5))                                  # the 1.5.0 Playground saved this circuit
    for saved_inputs in (["0"] * 3, ["0", "1", "+"]):                       # and these inputs (applied first)
        tokens, body = IO.split_preparation(tpl.spec, saved_inputs)
        assert _same_state(_sv(CP.with_io(body, tokens)), _sv(CP.with_io(tpl.spec, saved_inputs)))
        assert not IO.preparation_ops(body)


# ------------------------------------------------------------------ the Playground's session model (no browser)
class _SS(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


@pytest.fixture
def session(monkeypatch):
    import streamlit as st
    from ui import state as S
    fake = _SS()
    monkeypatch.setattr(st, "session_state", fake, raising=False)
    S.init()
    return fake


def test_loading_a_template_puts_its_input_in_the_inputs_and_the_rest_in_the_circuit(session):
    from ui import state as S
    S.load_template("qft", dict(n=3, s=5, mode="swap", prep="fourier"))
    tokens, body, _ = IO.template_inputs("qft", dict(n=3, s=5))
    assert session.inputs == tokens and S.spec().ops == body.ops
    assert [session["w_in_%d" % q] for q in range(3)] == tokens                 # the widgets show the model
    assert S.input_matches_template()


def test_changing_the_template_input_changes_only_the_inputs(session):
    from ui import state as S
    S.load_template("qft", dict(n=3, s=5))
    ops0, sig0, obs0 = list(S.spec().ops), S.signature(), session.observable.key
    session["w_tp_qft_s"] = 3
    S.cb_template_input()
    assert list(S.spec().ops) == ops0 and session.inputs == IO.template_inputs("qft", dict(n=3, s=3))[0]
    assert [session["w_in_%d" % q] for q in range(3)] == session.inputs
    assert session.tpl_params["s"] == 3 and session.observable.label == "P(011)" and obs0 != session.observable.key
    assert S.signature() != sig0                                                  # earlier results are now stale
    d = IO.derived_output(CP.with_io(S.spec(), session.inputs), session.relabel, session.observable)
    assert d["deterministic"] and d["top"][0][0] == "011"
    session["w_tp_qft_prep"] = "basis"
    S.cb_template_input()
    assert session.inputs == ["1", "1", "0"] and list(S.spec().ops) == ops0


def test_per_qubit_edits_are_the_input_and_are_reported_against_the_template(session):
    from ui import state as S
    S.load_template("qft", dict(n=3, s=5))
    session["w_in_0"] = "0"
    S.cb_input(0)
    assert session.inputs[0] == "0" and not S.input_matches_template()
    assert session.tpl_params["s"] == 5                                         # nothing was changed to "fit"
    session["w_io_all_in"], session["w_io_all_bs"] = "ramp:6", ""
    S.cb_register_io()
    assert S.input_matches_template() and session.tpl_params["s"] == 6 and session["w_tp_qft_s"] == 6
    S.cb_template_input()                                                       # "Use the template's input"
    assert session.inputs == IO.template_inputs("qft", dict(n=3, s=6))[0]


def test_a_user_chosen_observable_does_not_follow_the_template_input(session):
    from ui import state as S
    from qemx import observables as OBS
    S.load_template("qft", dict(n=3, s=5))
    session.observable = OBS.z(0, 3)
    session["w_tp_qft_s"] = 2
    S.cb_template_input()
    assert session.observable.key == OBS.z(0, 3).key


@pytest.mark.parametrize("key", ["qpe_ising", "steane", "qec_3q", "bv", "shor", "qpe", "ghz", "qaoa"])
def test_every_family_loads_with_inputs_that_match_its_state(session, key):
    from ui import state as S
    S.load_template(key, T.defaults(key))
    tokens, body, tpl = IO.template_inputs(key, T.defaults(key))
    assert session.inputs == tokens and S.spec().ops == body.ops and S.input_matches_template()
    if tpl.spec.n <= 12:
        assert _same_state(_sv(CP.with_io(S.spec(), session.inputs)), _sv(tpl.spec))
    for p in IO.input_params(key):
        assert session["w_tp_%s_%s" % (key, p.name)] == session.tpl_params[p.name]


def test_theta_of_a_code_is_an_input_and_the_intended_logical_value_follows_it(session):
    from ui import state as S
    from qemx import insight as IN
    S.load_template("steane", T.defaults("steane"))
    session["w_tp_steane_theta"] = 2.0
    S.cb_template_input()
    assert session.inputs[2] == CP.bloch_token(2.0) and all(t == "0" for q, t in enumerate(session.inputs) if q != 2)
    assert IN.intended_logical("steane", session.tpl_params) == pytest.approx(math.cos(2.0))
    assert r"\cos\theta=%.4f" % math.cos(2.0) in session.tpl_tex                 # the description follows too


def test_loading_an_old_state_moves_its_preparation_into_the_inputs(session):
    from ui import state as S
    from qemx import observables as OBS
    tpl = T.build("qft", dict(n=3, s=5))
    st = S.to_json()
    st.update(circuit=dict(n=3, name="QFT-3", ops=[dict(name=o.name, qubits=list(o.qubits), params=list(o.params),
                                                        kind=o.kind, tag=o.tag) for o in tpl.spec.ops]),
              inputs=["0"] * 3, observable=S.observable_json(OBS.projector(5, 3)))
    S.from_json(st)
    assert not IO.preparation_ops(S.spec()) and session.inputs == IO.template_inputs("qft", dict(n=3, s=5))[0]
    assert "input-preparation gate" in session.import_note
    d = IO.derived_output(CP.with_io(S.spec(), session.inputs))
    assert d["deterministic"] and d["top"][0][0] == "101"
