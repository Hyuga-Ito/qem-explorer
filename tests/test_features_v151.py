"""1.5.1: feature extraction (findings F06, F07 and review §5.7 of the independent 1.5.0 review).

F06  exact state features: every contiguous-cut entropy from the smaller side of the cut (no 2^(n−1)-square matrix),
     one statevector advanced layer by layer for the entanglement profile, values unchanged (checked against the
     1.5.0 dense algorithm, kept here verbatim as the reference, and against an SVD of the reshaped statevector), and
     cached descriptors that cannot be mutated.
F07  structural and observable features do not depend on the exact-state cutoff; Pauli-sum observables get
     setting-aware observable features and λ_LC by the documented |coefficient|-weighted rule.
§5.7 descriptive names (with aliases for schema /1 tables) and logical vs compiled non-Clifford content.
"""
import math

import numpy as np
import pytest

from qemx import compile as CP
from qemx import features as FT
from qemx.circuit import parse
from qemx.noise import NoiseSpec
from qemx.observables import z
from qemx.phase import descriptors as DS


def _random(n, layers, seed):
    rng = np.random.default_rng(seed)
    lines = []
    for L in range(layers):
        for q in range(n):
            lines.append("%s(%.15f) %d" % (["rx", "ry", "rz"][rng.integers(3)], rng.uniform(0, 2 * np.pi), q))
        for q in range(L % 2, n - 1, 2):
            lines.append("%s %d %d" % (["cx", "cz"][rng.integers(2)], q, q + 1))
    return parse("\n".join(lines), n, "r%d-%d-%d" % (n, layers, seed))


# ---- the 1.5.0 algorithm, verbatim (reference for the values)
def _ref_reduced(psi, keep, n):
    t = psi.reshape([2] * n)
    axes_keep = [n - 1 - q for q in sorted(keep, reverse=True)]
    axes_tr = [a for a in range(n) if a not in axes_keep]
    t = np.transpose(t, axes_keep + axes_tr).reshape(2 ** len(keep), -1)
    return t @ t.conj().T


def _ref_entropy(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log2(w)))


def _ref_profile(spec):
    from qemx.circuit import layer_order
    from qemx.templates import statevector
    n = spec.n
    ops, lay, _ = layer_order(spec)
    gates = [(o, l) for o, l in zip(ops, lay) if o.kind == "gate"]
    out, done = [], []
    for L in sorted(set(l for _, l in gates)):
        done += [o for o, l in gates if l == L]
        psi = statevector(n, done)
        out.append(_ref_entropy(_ref_reduced(psi, list(range(n // 2)), n)) if n >= 2 else 0.0)
    return out


@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6), (9, 7)])
def test_cut_entropies_and_state_descriptors_match_the_dense_reference(n, seed):
    from qemx.templates import statevector
    spec = _random(n, 3 + seed % 3, seed)
    psi = statevector(n, [o for o in spec.ops if o.kind == "gate"])
    for k in range(1, n):
        ref = _ref_entropy(_ref_reduced(psi, list(range(k)), n))
        s = np.linalg.svd(psi.reshape(2 ** (n - k), 2 ** k), compute_uv=False) ** 2   # independent: Schmidt values
        s = s[s > 1e-14]
        svd = float(-np.sum(s * np.log2(s)))
        got = DS.cut_entropy(psi, k, n)
        assert got == pytest.approx(ref, abs=1e-10) and got == pytest.approx(svd, abs=1e-10)
    d = DS.describe(spec, spec, z(0, n))
    pur = [float(np.real(np.trace(r @ r))) for r in (_ref_reduced(psi, [q], n) for q in range(n))]
    assert d["meyer_wallach"] == pytest.approx(2 * (1 - np.mean(pur)), abs=1e-10)
    assert d["entropy_half"] == pytest.approx(_ref_entropy(_ref_reduced(psi, list(range(n // 2)), n)), abs=1e-10)
    assert d["entropy_max_cut"] == pytest.approx(max(_ref_entropy(_ref_reduced(psi, list(range(k)), n))
                                                     for k in range(1, n)), abs=1e-10)
    assert FT.entanglement_profile(spec) == pytest.approx(_ref_profile(spec), abs=1e-10)


def test_contiguous_cuts_use_little_endian_qubit_order():
    """Analytic: a Bell pair on qubits (0, 1) and |+⟩ on qubit 2 has S(q0) = 1, S(q0 q1) = 0; a Bell pair on (1, 2)
    with |+⟩ on q0 has S(q0) = 0, S(q0 q1) = 1. A big-endian slip swaps them."""
    from qemx.templates import statevector
    for text, want in (("h 0\ncx 0 1\nh 2", (1.0, 0.0)), ("h 1\ncx 1 2\nh 0", (0.0, 1.0))):
        spec = parse(text, 3, "bell")
        psi = statevector(3, list(spec.ops))
        assert [DS.cut_entropy(psi, k, 3) for k in (1, 2)] == pytest.approx(list(want), abs=1e-12)
    d = DS.describe(parse("h 1\ncx 1 2\nh 0", 3, "b"), parse("h 1\ncx 1 2\nh 0", 3, "b"), z(0, 3))
    assert d["entropy_half"] == pytest.approx(0.0, abs=1e-12) and d["entropy_max_cut"] == pytest.approx(1.0)


def _record_decompositions(monkeypatch):
    shapes = []
    eig, svd = np.linalg.eigvalsh, np.linalg.svd

    def e(m, *a, **k):
        shapes.append(np.shape(m))
        return eig(m, *a, **k)

    def s(m, *a, **k):
        shapes.append(np.shape(m))
        return svd(m, *a, **k)
    monkeypatch.setattr(np.linalg, "eigvalsh", e)
    monkeypatch.setattr(np.linalg, "svd", s)
    return shapes


@pytest.mark.parametrize("n", [8, 12])
def test_no_matrix_larger_than_the_smaller_side_of_a_cut(n, monkeypatch):
    """1.5.0 kept the first k qubits for every cut: a 2^(n−1)-square matrix for the last one (128 × 128 at n = 8,
    8192 × 8192 = 1 GiB at n = 14). The largest decomposed matrix must be 2^⌊n/2⌋ square."""
    getattr(DS, "clear_cache", lambda: None)()          # (absent in 1.5.0: lets this test run there)
    shapes = _record_decompositions(monkeypatch)
    DS.describe(_random(n, 2, 11), _random(n, 2, 11), z(0, n))
    FT.entanglement_profile(_random(n, 2, 12))
    assert shapes and max(s[0] for s in shapes) <= 2 ** (n // 2)


def _count_gate_applications(monkeypatch):
    import qemx.templates as T
    box = {"n": 0}
    orig = T._mat

    def counted(name, params):
        box["n"] += 1
        return orig(name, params)
    monkeypatch.setattr(T, "_mat", counted)
    return box


def test_profile_applies_each_gate_once(monkeypatch):
    """The review's probe: 20 layers of one gate each took 210 gate applications (every prefix re-simulated)."""
    getattr(DS, "clear_cache", lambda: None)()          # (absent in 1.5.0: lets this test run there)
    box = _count_gate_applications(monkeypatch)
    prof = FT.entanglement_profile(parse("\n".join("rx(0.1) 0" for _ in range(20)), 2, "chain"))
    assert len(prof) == 20 and box["n"] == 20
    spec = _random(5, 6, 3)
    box["n"] = 0
    FT.entanglement_profile(spec)
    assert box["n"] == sum(1 for o in spec.ops if o.kind == "gate")


def test_descriptor_cache_returns_immutable_values_keyed_by_content_and_definition(monkeypatch):
    getattr(DS, "clear_cache", lambda: None)()          # (absent in 1.5.0: lets this test run there)
    spec = _random(4, 3, 5)
    a = DS.structural_descriptors(spec, spec)
    assert DS.structural_descriptors(spec.replace(name="same content"), spec) is a     # content, not object
    with pytest.raises(TypeError):
        a["two_qubit"] = 0                                                              # shared entries are read-only
    p = FT.entanglement_profile(spec)
    assert isinstance(p, tuple)
    box = _count_gate_applications(monkeypatch)
    FT.entanglement_profile(spec)
    assert box["n"] == 0                                                                # cache hit, no simulation
    monkeypatch.setattr(DS, "DEFINITIONS", "a changed definition")
    FT.entanglement_profile(spec)
    assert box["n"] == sum(1 for o in spec.ops if o.kind == "gate")                     # new definition: recomputed
    assert DS.state_descriptors(spec, ["entropy_half"]).keys() == {"entropy_half"}     # only what was requested


# ------------------------------------------------------------------ F07
def _isa(text, n):
    return CP.build(parse(text, n, "probe"), None, None, CP.CompileOptions(mode="isa", coupling="line"))


def test_cheap_features_survive_a_lowered_state_cutoff():
    """Review probe: 3 qubits, state cutoff 2 → 1.5.0 lost non_clifford and obs_weight (and other counts)."""
    comp = _isa("h 0\ncx 0 1\nt 1\ncx 1 2\nrz(0.3) 2", 3)
    noise = NoiseSpec(p1=1e-3, p2=1e-2)
    lo, mlo = FT.from_live(comp, noise, z(0, 3), 1e4, state_max_qubits=2)
    hi, _ = FT.from_live(comp, noise, z(0, 3), 1e4)
    for k in ("n", "two_qubit_logical", "non_clifford", "non_clifford_slots", "two_qubit", "depth", "pulses_1q",
              "idle_fraction", "obs_weight", "obs_terms", "obs_locality", "obs_settings", "lam_lc", "lam_ro"):
        assert math.isfinite(lo[k]) and lo[k] == hi[k], k
    assert lo["non_clifford"] == 2 and lo["obs_weight"] == 1
    assert math.isnan(lo["entropy_half"]) and "skipped above 2 qubits" in mlo["entropy_half"]
    assert math.isfinite(hi["entropy_half"])
    only, _ = FT.compute(comp, z(0, 3), noise, 1e4, names=["two_qubit", "obs_weight"])
    assert set(only) == {"two_qubit", "obs_weight"}                                     # nothing else computed


def test_pauli_sum_observables_get_setting_aware_features_and_a_documented_lambda():
    from qemx.hamiltonian import clbit_of, parse_pauli_sum, qwc_groups
    from qemx.observables import z_string
    from qemx.phase.lightcone import calculate_lambda_lc
    comp = CP.build(parse("h 0\ncx 0 1\nt 1\ncx 1 2\nrz(0.3) 2", 3, "probe"), None, None,
                    CP.CompileOptions(mode="isa", coupling="line"), relabel=True)
    noise = NoiseSpec(p1=1e-3, p2=1e-2, idle_damp=2e-3)
    H = parse_pauli_sum("0.5*Z0*Z1 + 0.25*X1*X2 - 0.3*Y0 + 0.2*Z2 + 1.5", 3)
    v, miss = FT.from_live(comp, noise, H, 1e4)
    terms = [(c, f) for c, f in H.terms if f]
    assert v["obs_terms"] == 4 and v["obs_weight"] == 2 and v["obs_weight_mean"] == pytest.approx(1.5)
    assert v["obs_support"] == 3 and v["obs_settings"] == len(qwc_groups(H.terms)) == 2
    # independent: each term as a diagonal Z string on its classical bits, its own λ_LC, |c|-weighted mean
    lam = [calculate_lambda_lc(comp, noise, z_string(sorted(clbit_of(comp, q) for q, _ in f), 3)).lam_lc
           for c, f in terms]
    want = sum(abs(c) * l for (c, _), l in zip(terms, lam)) / sum(abs(c) for c, _ in terms)
    assert v["lam_lc"] == pytest.approx(want, rel=1e-12) and v["lam_lc"] > 0
    assert not {"lam_lc", "obs_weight", "non_clifford"} & set(miss)


def test_names_say_what_the_counts_contain_and_old_tables_still_load():
    assert FT.VERSION == "qemx.features/2"
    names = {f.name for f in FT.DICTIONARY}
    assert {"routing_overhead", "routing_ratio", "two_qubit_density"}.isdisjoint(names)
    assert {"two_qubit_compile_overhead", "two_qubit_compile_ratio", "two_qubit_per_qubit_layer",
            "non_clifford_slots", "obs_settings"} <= names
    comp = _isa("cx 0 1\ncx 0 1\nh 2\ncx 1 2", 3)                    # the optimizer cancels the CX pair
    v, _ = FT.from_live(comp, NoiseSpec(), z(2, 3), 1e4)
    assert v["two_qubit_compile_overhead"] == v["two_qubit"] - v["two_qubit_logical"] < 0   # signed, not "routing"
    assert v["two_qubit_per_qubit_layer"] == pytest.approx(v["two_qubit"] / (3 * v["depth"]))
    cols = ["x_two_qubit_density", "x_routing_ratio", "x_routing_overhead", "x_lam_lc"]
    m, notes, prob = FT.resolve_columns(cols, ["two_qubit_per_qubit_layer", "two_qubit_compile_ratio", "lam_lc"],
                                        "qemx.features/1")
    assert m == {"two_qubit_per_qubit_layer": "x_two_qubit_density", "two_qubit_compile_ratio": "x_routing_ratio",
                 "lam_lc": "x_lam_lc"} and len(notes) == 2 and not prob
    _, _, prob = FT.resolve_columns(cols, ["two_qubit_compile_overhead"], "qemx.features/1")
    assert prob and "different definition" in prob[0]                   # clamped in /1 records: not mixed silently
    _, _, prob = FT.resolve_columns(cols, ["lam_lc"], "qemx.features/9")
    assert prob


def test_logical_non_clifford_content_differs_from_compiled_training_slots():
    from qemx.transforms import non_clifford_slots
    comp = _isa("h 0\ncp(0.7853981633974483) 0 1", 2)
    v, _ = FT.from_live(comp, NoiseSpec(), z(1, 2), 1e4)
    assert v["non_clifford"] == 1                                        # one controlled phase as written
    assert v["non_clifford_slots"] == len(non_clifford_slots(comp.implemented)) > 1   # what CDR/FR/NIL replace


def test_lambda_wording_does_not_overstate_it():
    d = FT.BY_NAME["lam_lc"].definition
    assert "not a sufficient statistic" in d and "independent-location surrogate" in d
    md = FT.dictionary_markdown()
    assert "qemx.features/2" in md and "routing_overhead" in md and "two_qubit_compile_overhead" in md
