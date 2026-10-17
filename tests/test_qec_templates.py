"""Encoded benchmark algorithms on small codes (qemx/qec.py, template "qec_algorithm"): the codes as data, exact
logical values, correction and detection of single errors, fault tolerance as claimed (and its absence where the
notes say so), the app's pipelines (symmetry verification, learning-based mitigation, the QEC analysis helpers), and
the 1.0 option spellings."""
import math

import numpy as np
import pytest

from qemx import compile as CP
from qemx import insight as IN
from qemx import qec as Q
from qemx import templates as T
from qemx.circuit import Op
from qemx.methods import Context, Recipe, Runner
from qemx.noise import NoiseSpec, PRESETS
from qemx.recipes import compare, preset

KEY = "qec_algorithm"
SUPPORTED = [(c, a) for c in Q.CODE_KEYS for a in Q.ALGORITHMS if Q.num_qubits(c, a) <= Q.MAX_QUBITS]
# errors each code handles: corrected (repetition codes, [[5,1,3]]) or detected ([[4,2,2]])
CORRECTED = {"bit-flip": "X", "phase-flip": "Z", "5-1-3": "XYZ"}


def _tpl(code, alg, **kw):
    return T.build(KEY, dict(code=code, algorithm=alg, **kw))


def _probs(spec):
    return np.abs(T.statevector(spec.n, spec.ops)) ** 2


def _accept_mask(tpl):
    x = np.arange(2 ** tpl.spec.n)
    ok = np.ones(2 ** tpl.spec.n, bool)
    for _, qs in tpl.checks:
        par = np.zeros_like(x)
        for q in qs:
            par ^= (x >> q) & 1
        ok &= par == 0
    return ok


def _decoded(tpl, ops=None):
    """(decoded value, P(all checks pass), value among the accepted shots) from an exact statevector."""
    p = _probs(tpl.spec if ops is None else tpl.spec.replace(ops=ops))
    o = np.asarray(tpl.observable.values)
    m = _accept_mask(tpl)
    acc = float(p[m].sum())
    return float(p @ o), acc, (float(p[m] @ o[m]) / acc if acc > 1e-12 else float("nan"))


def _expected(alg, theta=1.0472, basis="Z"):
    """Closed forms, independent of the logical simulation in qec.py."""
    if alg == "state prep":
        return math.cos(theta) if basis == "Z" else math.sin(theta)
    if alg in ("RX rotation", "RZ Ramsey"):
        return math.cos(theta)
    return 1.0                       # Bell, GHZ (both bases), DJ, BV, Grover and QFT answer with certainty


# ------------------------------------------------------------------ 1. the codes as data
@pytest.mark.parametrize("key", Q.CODE_KEYS)
def test_code_tables_are_consistent(key):
    c = Q.code(key)
    S = [Q.Pauli.parse(s) for s in c.stabilizers]
    X = [c.x_l(j) for j in range(c.k)]
    Z = [c.z_l(j) for j in range(c.k)]
    assert len(S) == c.n - c.k and all(a.commutes(b) for a in S for b in S)
    assert len({(g.x, g.z) for g in c.stabilizer_group()}) == 2 ** (c.n - c.k)          # independent generators
    for j in range(c.k):
        assert all(X[j].commutes(s) and Z[j].commutes(s) for s in S)
        for i in range(c.k):
            assert X[i].commutes(Z[j]) == (i != j) and X[i].commutes(X[j]) and Z[i].commutes(Z[j])
    # distance: the lowest weight of a logical operator (an element of N(S) \ S); 1 for the repetition codes
    group = c.stabilizer_group()
    logical = [Q.Pauli(x, z) for x in range(2 ** c.n) for z in range(2 ** c.n)
               if all(Q.Pauli(x, z).commutes(s) for s in S) and (x, z) not in {(g.x, g.z) for g in group}]
    assert min(p.weight for p in logical) == c.d


@pytest.mark.parametrize("key", Q.CODE_KEYS)
def test_encoder_prepares_the_code_space_with_injected_inputs(key):
    c = Q.code(key)
    th = [0.7, 2.1][:c.k]
    ops = [Op("ry", (c.inputs[j],), (th[j],)) for j in range(c.k)]
    ops += [Op(g[0], tuple(g[1:])) for g in c.encoder]
    a = T.statevector(c.n, ops)
    ev = lambda P: float(np.vdot(a, P.matrix(c.n) @ a).real)                     # noqa: E731
    assert all(ev(Q.Pauli.parse(s)) == pytest.approx(1.0, abs=1e-9) for s in c.stabilizers)
    for j in range(c.k):
        assert ev(c.z_l(j)) == pytest.approx(math.cos(th[j]), abs=1e-9)
        assert ev(c.x_l(j)) == pytest.approx(math.sin(th[j]), abs=1e-9)


def test_decoders_majority_detection_and_the_five_qubit_table():
    rep = Q.code("bit-flip").readouts["Z"]
    for x in range(8):
        bits = [(x >> q) & 1 for q in range(3)]
        syn = tuple(sum(bits[q] for q in chk) % 2 for chk in rep.checks)
        raw = sum(bits) % 2
        assert raw ^ rep.table[syn][0] == int(sum(bits) >= 2)                   # majority vote
    assert Q.code("4-2-2").readouts["Z"].table == {(0,): (0, 0)}                 # detection only: no correction
    five = Q.code("5-1-3").readouts["Z"].table
    assert len(five) == 16 and five == {s: (int(v),) for s, v in T.five_qubit_decoder_table().items()}
    assert Q.code("bit-flip").readouts["X"].checks == () and not Q.code("bit-flip").readouts["X"].fault_tolerant


def test_pauli_algebra():
    X, Z, Y = (Q.Pauli.parse(s) for s in "XZY")
    assert Y == Q.Pauli(0, 0, 1) * X * Z and Y.sign() == 1 and not X.commutes(Z)
    assert (Q.Pauli.parse("XX") * Q.Pauli.parse("ZZ")).sign() == -1                      # XX·ZZ = −YY
    assert np.allclose(Q.Pauli.parse("-XZ").matrix(2), -np.kron(np.diag([1, -1]), [[0, 1], [1, 0]]))
    assert Q.propagate(0b01, 0, [("cx", 0, 1)]) == (0b11, 0) and Q.propagate(0, 0b10, [("cx", 0, 1)]) == (0, 0b11)
    assert Q.propagate(0b1, 0, [("h", 0)]) == (0, 0b1)


# ------------------------------------------------------------------ 2. exact logical values
@pytest.mark.parametrize("code,alg", SUPPORTED)
def test_every_template_gives_its_logical_value_in_exact_simulation(code, alg):
    """The app's engine path (compile.build + Context/Runner, density matrix): the decoded observable equals the
    closed-form logical value, and every check reads +1."""
    tpl = _tpl(code, alg)
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    ctx = Context(comp, NoiseSpec(), tpl.observable, Recipe.make("raw"), Runner("numpy"))
    assert ctx.ideal_value() == pytest.approx(_expected(alg), abs=1e-9)
    assert tpl.logical == pytest.approx(_expected(alg), abs=1e-9)
    assert ctx.ideal_value(o=_accept_mask(tpl).astype(float)) == pytest.approx(1.0, abs=1e-9)
    assert tpl.spec.n == tpl.observable.n == Q.num_qubits(code, alg) <= Q.MAX_QUBITS
    assert IN.value_kind(tpl.observable.values) == "parity"                     # ±1: p_L = (1 − ⟨O⟩/⟨O⟩_ideal)/2


@pytest.mark.parametrize("code", Q.CODE_KEYS)
def test_parameters_angles_secrets_oracles_bases_and_decoders(code):
    cases = [("state prep", dict(theta=0.4)), ("state prep", dict(theta=2.2, logical_basis="X")),
             ("RX rotation", dict(theta=0.3)), ("RZ Ramsey", dict(theta=2.5)), ("Bell", dict(logical_basis="X"))]
    cases += [(a, dict(s=s)) for a in ("Bernstein-Vazirani", "Grover-2", "QFT-2") for s in range(4)]
    cases += [("Deutsch-Jozsa", dict(oracle=o)) for o in Q.ORACLES]
    if Q.num_qubits(code, "GHZ-3") <= Q.MAX_QUBITS:
        cases.append(("GHZ-3", dict(logical_basis="X")))
    for alg, kw in cases:
        for dec in Q.DECODERS:
            tpl = _tpl(code, alg, decoder=dec, **kw)
            v, acc, _ = _decoded(tpl)
            exp = _expected(alg, kw.get("theta", 1.0472), kw.get("logical_basis", "Z"))
            assert v == pytest.approx(exp, abs=1e-9) and tpl.logical == pytest.approx(exp, abs=1e-9), (alg, kw, dec)
            assert acc == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("code", Q.CODE_KEYS)
def test_random_logical_circuits_compile_exactly(code):
    """Random 2-logical-qubit circuits over the whole logical gate set: the decoded distribution of the physical
    circuit equals the distribution of the logical circuit, outcome by outcome."""
    rng = np.random.default_rng(5)
    c = Q.code(code)
    one = ["h", "s", "sdg", "t", "x", "y", "z", "rz", "rx", "ry"]
    for trial in range(6):
        body = []
        for _ in range(7):
            r = rng.integers(6)
            q = int(rng.integers(2))
            if r < 3:
                g = one[int(rng.integers(len(one)))]
                body.append((g, (q,), (float(rng.uniform(-3, 3)),) if g.startswith("r") else ()))
            elif r == 3:
                body.append(("cx", (q, 1 - q), ()))
            elif r == 4:
                g = ["cz", "cp", "swap"][int(rng.integers(3))]
                body.append((g, (0, 1), (float(rng.uniform(-3, 3)),) if g == "cp" else ()))
            else:
                body.append(("h", (0, 1), ()))
        prep = {0: [("ry", (float(rng.uniform(0, 3)),))], 1: [("rx", (float(rng.uniform(0, 3)),))]}
        basis = "XZ"[trial % 2]
        for y in range(4):
            alg = Q.Algorithm("random", 2, prep, body, basis,
                              lambda b, y=y: ((b[0] == (y & 1)) & (b[1] == (y >> 1))).astype(float), "P", "")
            built = Q.compile_algorithm(c, alg)
            p = _probs(built.spec)
            assert float(p @ np.asarray(built.observable.values)) == pytest.approx(built.logical, abs=1e-9), \
                (code, trial, y, body)


# ------------------------------------------------------------------ 3. single errors: corrected or detected
@pytest.mark.parametrize("code,alg", SUPPORTED)
def test_a_single_error_before_readout_is_corrected_or_detected(code, alg):
    n = Q.num_qubits(code, alg)
    ideal = _expected(alg)
    for k in range(n):
        for P in "XYZ":
            tpl = _tpl(code, alg, error="%s on qubit k" % P, k=k)
            v, acc, v_acc = _decoded(tpl)
            if code == "4-2-2":                      # distance 2: an error that flips a Z_L readout bit is detected
                if P in "XY":
                    assert acc == pytest.approx(0.0, abs=1e-9), (alg, k, P)
                else:                                # Z right before a Z-basis measurement changes nothing
                    assert acc == pytest.approx(1.0, abs=1e-9) and v == pytest.approx(ideal, abs=1e-9)
            elif P in CORRECTED[code]:
                assert v == pytest.approx(ideal, abs=1e-9), (code, alg, k, P)


def test_raw_parity_does_not_correct_and_x_basis_readouts_detect_the_conjugate_errors():
    th = 0.9
    for code, P in (("bit-flip", "X"), ("phase-flip", "Z"), ("5-1-3", "Y")):
        tpl = _tpl(code, "RX rotation", theta=th, decoder="raw logical parity", error="%s on qubit k" % P, k=0)
        assert _decoded(tpl)[0] == pytest.approx(-math.cos(th), abs=1e-9), code
    for k in range(4):                               # [[4,2,2]] read in X: the X0X1X2X3 check catches Z and Y
        for P, detected in (("Z", True), ("Y", True), ("X", False)):
            _, acc, _ = _decoded(_tpl("4-2-2", "Bell", logical_basis="X", error="%s on qubit k" % P, k=k))
            assert acc == pytest.approx(0.0 if detected else 1.0, abs=1e-9)
    tpl = _tpl("bit-flip", "Bell", logical_basis="X")
    assert tpl.checks == () and tpl.symmetry is None and "unprotected" in T.REGISTRY[KEY][2][5].help


def test_symmetry_verification_is_error_detection_with_the_templates_checks():
    tpl = _tpl("4-2-2", "Grover-2", error="X on qubit k", k=2)
    rc = preset("Symmetry verification", tpl.symmetry)
    assert rc.p["qubits"] == [0, 1, 2, 3] and rc.p["more"] == ""
    _, outs, _ = compare(CP.build(tpl.spec), PRESETS["Ideal (no noise)"], tpl.observable, [preset("Unmitigated"), rc],
                         symmetry=tpl.symmetry)
    assert outs[1].result.accept == pytest.approx(0.0, abs=1e-12)
    tpl = _tpl("bit-flip", "GHZ-3")                  # several blocks: every check goes into "more"
    assert tpl.symmetry == ((0, 1), 1, "1 2; 3 4; 4 5; 6 7; 7 8")
    assert [lab for lab, _ in tpl.checks] == ["Z0Z1", "Z1Z2", "Z3Z4", "Z4Z5", "Z6Z7", "Z7Z8"]


# ------------------------------------------------------------------ 4. fault tolerance as claimed
FT_CASES = [("4-2-2", "Grover-2", dict(s=s)) for s in range(4)] + \
           [("4-2-2", "Bernstein-Vazirani", dict(s=s)) for s in range(4)] + \
           [("4-2-2", "Deutsch-Jozsa", dict(oracle=o)) for o in Q.ORACLES] + \
           [("4-2-2", "Bell", dict(logical_basis=b)) for b in "ZX"] + \
           [(c, a, {}) for c in ("bit-flip", "phase-flip") for a in ("Bell", "GHZ-3")]


@pytest.mark.parametrize("code,alg,kw", FT_CASES)
def test_every_single_fault_after_encoding_is_harmless_when_all_logical_gates_are_fault_tolerant(code, alg, kw):
    """Insert one Pauli fault at every position between the end of the encoder and the measurement: on [[4,2,2]] it is
    detected (P_accept = 0) or harmless; on a repetition code its own error type is corrected."""
    tpl = _tpl(code, alg, **kw)
    assert tpl.qec["all_gates_fault_tolerant"] and "Every logical gate of this benchmark is fault-tolerant" in tpl.notes
    ideal = _expected(alg, basis=kw.get("logical_basis", "Z"))
    ops = list(tpl.spec.ops)
    start = tpl.qec["body"][0]
    paulis = {"4-2-2": "XYZ", "bit-flip": "X", "phase-flip": "Z"}[code]
    for i in range(start, len(ops) + 1):
        for q in range(tpl.spec.n):
            for P in paulis:
                v, acc, v_acc = _decoded(tpl, ops[:i] + [Op(P.lower(), (q,))] + ops[i:])
                if code == "4-2-2":
                    assert acc < 1e-9 or (acc > 1 - 1e-9 and v_acc == pytest.approx(ideal, abs=1e-9)), (i, q, P)
                else:
                    assert v == pytest.approx(ideal, abs=1e-9), (i, q, P)


@pytest.mark.parametrize("code", ["4-2-2", "phase-flip"])
def test_a_fault_inside_a_parity_ladder_is_an_undetected_logical_error(code):
    """The notes' claim for the non-fault-tolerant rotations: a Z fault on the ladder's last qubit, right after its
    RZ, spreads to the whole rotation axis X_L — accepted by every check and not corrected."""
    th = 1.0472
    tpl = _tpl(code, "RX rotation", theta=th)
    assert not tpl.qec["all_gates_fault_tolerant"] and "NOT fault-tolerant: RX_L(L0)" in tpl.notes
    ops = list(tpl.spec.ops)
    b0, b1 = tpl.qec["body"]
    i = next(j for j in range(b0, b1) if ops[j].name == "rz")
    v, acc, v_acc = _decoded(tpl, ops[:i + 1] + [Op("z", ops[i].qubits)] + ops[i + 1:])
    assert acc == pytest.approx(1.0, abs=1e-9) and v == pytest.approx(-math.cos(th), abs=1e-9)


def test_fault_tolerance_labels():
    def ft(code, alg, **kw):
        return _tpl(code, alg, **kw).qec["all_gates_fault_tolerant"]
    assert ft("4-2-2", "Grover-2") and ft("bit-flip", "GHZ-3") and ft("phase-flip", "Bell")
    assert not ft("4-2-2", "QFT-2") and not ft("bit-flip", "Deutsch-Jozsa") and not ft("5-1-3", "Bell")
    for code, alg in SUPPORTED:
        tpl = _tpl(code, alg)
        assert "Encoding — NOT fault-tolerant" in tpl.notes and "no syndrome-extraction rounds" in tpl.notes
        assert ("Readout — NOT fault-tolerant" in tpl.notes) == (code == "5-1-3")
        for g in tpl.qec["gates"]:
            relabel_or_transversal = any(w in g["implementation"] for w in ("relabel", "transversal", "Paulis"))
            assert g["fault_tolerant"] == relabel_or_transversal, g
    g = {d["gate"]: d for d in _tpl("4-2-2", "Grover-2").qec["gates"]}
    assert g["CZ_L(L0,L1)"]["implementation"].startswith("S·S†·S†·S")
    assert "relabel" in _tpl("4-2-2", "Deutsch-Jozsa").qec["gates"][0]["implementation"]


# ------------------------------------------------------------------ 5. the app's pipelines
@pytest.mark.parametrize("code,alg", [("4-2-2", "Grover-2"), ("bit-flip", "Bell"), ("phase-flip", "RX rotation"),
                                      ("5-1-3", "state prep")])
def test_symmetry_verification_on_a_noisy_simulation(code, alg):
    tpl = _tpl(code, alg)
    ideal, outs, _ = compare(CP.build(tpl.spec), PRESETS["Depolarizing"], tpl.observable,
                             [preset("Unmitigated"), preset("Symmetry verification", tpl.symmetry)],
                             symmetry=tpl.symmetry)
    raw, sv = (o.result for o in outs)
    assert ideal == pytest.approx(tpl.logical, abs=1e-9)
    assert 0.3 < sv.accept < 1.0
    assert abs(sv.mean - ideal) < abs(raw.mean - ideal)                     # detection removes part of the error


def test_learning_based_mitigation_trains_on_the_non_clifford_logical_rotation():
    from qemx.transforms import non_clifford_slots
    for alg in ("RX rotation", "RZ Ramsey", "QFT-2"):
        assert non_clifford_slots(_tpl("4-2-2", alg).spec), alg
    assert not non_clifford_slots(_tpl("4-2-2", "Grover-2").spec)
    tpl = _tpl("4-2-2", "RX rotation")
    ideal, outs, _ = compare(CP.build(tpl.spec), PRESETS["Depolarizing"], tpl.observable,
                             [preset("Unmitigated"), preset("CDR · OLS")])
    raw, cdr = (o.result for o in outs)
    assert cdr is not None and not any("Clifford:" in n for n in cdr.notes)
    assert abs(cdr.mean - ideal) < 0.5 * abs(raw.mean - ideal)


def test_checks_feed_the_qec_analysis_helpers():
    tpl = _tpl("bit-flip", "Bell", error="X on qubit k", k=4)
    p = _probs(tpl.spec)
    syn = IN.syndrome_distribution(p, list(tpl.checks), tpl.spec.n)
    assert syn[0b0011] == pytest.approx(1.0)                    # checks Z3Z4 and Z4Z5 fire: bits of checks 2 and 3
    clean = _tpl("bit-flip", "Bell")
    assert IN.syndrome_distribution(_probs(clean.spec), list(clean.checks), clean.spec.n)[0] == pytest.approx(1.0)
    assert IN.logical_error(0.8, tpl.logical) == pytest.approx(0.1)
    assert T.qec_checks(KEY, dict(code="bit-flip", algorithm="Bell")) == list(clean.checks)
    assert T.intended_logical(KEY, dict(algorithm="RX rotation", theta=0.5)) == pytest.approx(math.cos(0.5))
    assert T.qec_checks("five_qubit", {}) == [] and T.intended_logical("ghz", {}) is None
    assert T.qec_checks(KEY, dict(code="5-1-3", algorithm="GHZ-3")) == []          # does not build: no checks


def test_qec_analysis_view_once_insight_lists_the_template():
    """Active once qemx/insight.py lists qec_algorithm (its integration patch): the analysis then opens on the QEC view,
    reads the template's own checks and measures p_L against the error-free logical value."""
    if KEY not in IN.QEC_TEMPLATES:
        pytest.skip("qemx/insight.py does not list qec_algorithm yet")
    prm = dict(code="4-2-2", algorithm="Grover-2")
    tpl = T.build(KEY, prm)
    prof = IN.profile(KEY, prm, False, tpl.observable, ["raw"], tpl.spec.n)
    assert prof.primary == "qec" and prof.checks == list(tpl.checks)
    assert IN.qec_checks(KEY, prm, tpl.spec.n) == list(tpl.checks)
    assert IN.intended_logical(KEY, dict(prm, error="X on qubit k")) == pytest.approx(tpl.logical)
    prm = dict(code="bit-flip", algorithm="Bell", logical_basis="X")          # no checks in that basis
    assert IN.profile(KEY, prm, False, T.build(KEY, prm).observable, ["raw"], 6).primary == "expectation"


# ------------------------------------------------------------------ 6. registry, schema, presets
def test_registry_schema_and_presets():
    assert KEY in T.CATEGORIES["FTQC · encoded algorithms"] and T.category_of(KEY) == "FTQC · encoded algorithms"
    tpl = T.build(KEY)
    assert tpl.spec.n == 4 and tpl.logical == pytest.approx(1.0) and tpl.symmetry == ((0, 1, 2, 3), 1, "")
    names = [p.name for p in T.REGISTRY[KEY][2]]
    assert names[:2] == ["code", "algorithm"] and {"theta", "decoder", "error", "k", "where"} <= set(names)
    assert T.clamp_params(KEY, dict(code="bit-flip", algorithm="GHZ-3", k=99))["k"] == 8
    assert T.clamp_params(KEY, dict(code="nonsense"))["code"] == "4-2-2"
    with pytest.raises(ValueError, match="needs 15 qubits"):
        T.build(KEY, dict(code="5-1-3", algorithm="GHZ-3"))
    for name, key, prm in T.QEC_PRESETS:
        t = T.build(key, prm)
        assert t.notes and t.spec.n <= Q.MAX_QUBITS, name
    from qemx import sweep as SW                                   # every option can be swept by name
    axes = SW.parse("code: %s\nalgorithm: %s" % (", ".join(Q.CODE_KEYS), ", ".join(Q.ALGORITHMS)), [KEY])
    assert [a.values for a in axes] == [Q.CODE_KEYS, Q.ALGORITHMS]


def test_legacy_spellings_map_to_current_values():
    assert T.clamp_params("hea_vqe", dict(angles="VQE-optimised"))["angles"] == "VQE-optimized"
    assert T.clamp_params("qaoa", dict(angles="optimised"))["angles"] == "optimized"
    assert T.clamp_params("shor", dict(multiplier="synthesised"))["multiplier"] == "synthesized"
    assert T.clamp_params("qaoa", dict(angles="optimized"))["angles"] == "optimized"           # current values stay
    options = {o for _, _, schema in T.REGISTRY.values() for p in schema if p.kind == "choice" for o in p.options}
    assert all(k not in options and v in options for k, v in T.LEGACY_CHOICES.items())
    from qemx import sweep as SW                                   # a 1.0 spelling typed into a sweep
    assert SW.parse("angles: optimised, fixed", ["qaoa"])[0].values == ("optimized", "fixed")
    assert SW.parse("angles: VQE-optimised", ["hea_vqe"])[0].values == ("VQE-optimized",)
    assert SW.parse("multiplier: synthesised", ["shor"])[0].values == ("synthesized",)


def test_playground_presets_fill_every_parameter_and_build():
    import sys
    sys.path.insert(0, ".")
    from ui import state as S
    from qemx import templates as T
    assert T.QEC_PRESETS
    for name, key, prm in T.QEC_PRESETS:
        vals = S.template_preset_values(key, name)
        assert set(vals) == {"w_tp_%s_%s" % (key, p.name) for p in T.REGISTRY[key][2]}
        params = {p.name: vals["w_tp_%s_%s" % (key, p.name)] for p in T.REGISTRY[key][2]}
        assert all(params[k] == v for k, v in prm.items())
        assert T.build(key, params).spec.n >= 3
    assert S.template_preset_values("qec_algorithm", "no such preset") == {}
