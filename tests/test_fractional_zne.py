"""Fractional ZNE (v1.4.1): local folding at fractional noise scales λ = 1 + 2(k + m/n), the nested fold orders
(qemx.transforms.fold_order), the quadratic least-squares rule (poly2), the ``subset`` parameter of ZNE and of the
chain, and the fractional run roles of the divergence plan."""
from __future__ import annotations

import math

import numpy as np
import pytest

from qemx.circuit import CircuitSpec, Op
from qemx.methods.zne import extrapolate, linear_weights, poly_weights, richardson_weights
from qemx.transforms import FOLD_SUBSETS, _bit_reversal_order, achieved_lambda, fold_local, fold_order

FRACTIONAL = (1.0, 4 / 3, 5 / 3, 2.0)


def _line(n2: int, n: int = 4) -> CircuitSpec:
    """n2 CZ gates walking along a line of n qubits, with RY rotations between them (non-Clifford, foldable)."""
    ops = []
    for j in range(n2):
        a = j % (n - 1)
        ops += [Op("ry", (a,), (0.3 + 0.1 * j,)), Op("cz", (a, a + 1))]
    return CircuitSpec(n, tuple(ops), "line-%d" % n2)


def _fold_counts(spec: CircuitSpec, folded: CircuitSpec) -> list:
    """How many times each two-qubit gate of ``spec`` (circuit order) is folded in ``folded``: the fold copies of
    a gate follow it directly, as (inverse, copy) pairs tagged fold-inv / fold."""
    out, cur = [], None
    for o in folded.ops:
        if o.tag in ("fold", "fold-inv"):
            if cur is not None and o.tag == "fold":
                out[cur] += 1
            continue
        if o.kind == "gate" and o.k == 2:
            out.append(0)
            cur = len(out) - 1
        else:
            cur = None
    assert len(out) == spec.count("2q")
    return out


# ------------------------------------------------------------------ fold orders
def test_bit_reversal_order_is_a_spread_permutation():
    assert _bit_reversal_order(6) == [0, 4, 2, 1, 5, 3]
    assert _bit_reversal_order(1) == [0] and _bit_reversal_order(0) == []
    o = _bit_reversal_order(12)
    assert sorted(o) == list(range(12))
    assert sorted(o[:6]) == [0, 2, 4, 6, 8, 10]                 # half the gates: every other one
    assert sorted(o[:3]) == [0, 4, 8]                           # a quarter: evenly spaced


@pytest.mark.parametrize("subset", ["nested", "stratified", "balanced"])
def test_fold_orders_are_permutations(subset):
    w = np.random.default_rng(1).lognormal(-5, 0.6, 17) if subset == "balanced" else None
    o = fold_order(17, subset, seed=3, weights=w)
    assert sorted(o) == list(range(17))


def test_fold_order_seeds_and_errors():
    assert fold_order(9, "nested", seed=1) == fold_order(9, "nested", seed=1)
    assert fold_order(9, "nested", seed=1) != fold_order(9, "nested", seed=2)
    assert fold_order(9, "stratified", seed=1) == fold_order(9, "stratified", seed=2)     # deterministic
    assert fold_order(9, "balanced") == fold_order(9, "stratified")                      # no weights: stratified
    assert fold_order(5, "balanced", weights=np.full(5, 0.01)) == fold_order(5, "stratified")   # uniform: the same
    with pytest.raises(ValueError):
        fold_order(5, "sideways")
    with pytest.raises(ValueError):
        fold_order(5, "balanced", weights=[0.1, 0.2])
    with pytest.raises(ValueError):
        fold_order(3, "balanced", weights=[0.1, np.nan, 0.2])
    assert set(FOLD_SUBSETS) == {"random", "nested", "stratified", "balanced"}


def test_balanced_order_folds_noise_in_proportion_to_gates():
    """Every prefix of the balanced order carries the mean error per gate of the whole circuit (to within one
    gate's spread): the folded error grows like the folded gate count, so λ in gates ≈ λ in fault rate."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        w = rng.lognormal(-5, 0.8, 24)
        dev = lambda order: max(abs(w[order[:j]].sum() - j * w.mean()) for j in range(1, 25))   # noqa: E731
        bal = fold_order(24, "balanced", weights=w)
        assert dev(np.array(bal)) <= w.max() - w.min() + 1e-15
        assert dev(np.array(bal)) <= dev(np.array(fold_order(24, "stratified"))) + 1e-15


# ------------------------------------------------------------------ fractional local folds
def test_fractional_local_fold_reaches_the_requested_scale():
    spec = _line(12)
    for lam, want in ((4 / 3, 1 + 2 * 2 / 12), (5 / 3, 1 + 2 * 4 / 12), (2.0, 2.0), (3.0, 3.0), (3.5, 1 + 2 * 1.25)):
        for subset in FOLD_SUBSETS:
            f = fold_local(spec, lam, seed=5, subset=subset)
            assert achieved_lambda(spec, f) == pytest.approx(want), (lam, subset)
    assert fold_local(spec, 1.0) is spec


@pytest.mark.parametrize("subset", ["nested", "stratified", "balanced"])
def test_nested_orders_fold_a_growing_set(subset):
    """The gates folded at a smaller λ are folded at every larger one: the noise grows monotonically along the
    scales (the 1.4.0 random subset does not guarantee this)."""
    spec = _line(15)
    w = list(np.random.default_rng(2).lognormal(-5, 0.5, 15)) if subset == "balanced" else None
    prev = None
    for lam in (1 + 2 / 15, 1 + 6 / 15, 1 + 10 / 15, 1 + 14 / 15, 3.0, 3 + 8 / 15):
        c = np.array(_fold_counts(spec, fold_local(spec, lam, seed=4, subset=subset, weights=w)))
        if prev is not None:
            assert np.all(c >= prev), (subset, lam)
        prev = c


def test_stratified_subset_spreads_over_the_circuit():
    spec = _line(12)
    c = _fold_counts(spec, fold_local(spec, 2.0, subset="stratified"))
    assert [i for i, x in enumerate(c) if x] == [0, 2, 4, 6, 8, 10]


def test_random_subset_is_the_140_behaviour():
    spec = _line(12)
    a = fold_local(spec, 5 / 3, seed=9)                         # default subset: random, seeded
    assert a.ops == fold_local(spec, 5 / 3, seed=9, subset="random").ops
    assert sum(_fold_counts(spec, a)) == 4


def test_fold_local_keeps_alignment_barriers():
    from qemx.phase.families import build
    spec = build("context", 6, layers=3, spectators="driven").spec
    keep = sum(1 for o in spec.ops if o.kind == "barrier" and o.tag == "keep")
    for subset in ("random", "stratified"):
        f = fold_local(spec, 5 / 3, seed=1, subset=subset)
        assert sum(1 for o in f.ops if o.kind == "barrier" and o.tag == "keep") == keep


# ------------------------------------------------------------------ extrapolation rules
def test_poly_weights():
    lams = np.array(FRACTIONAL)
    for deg in (1, 2, 3):
        eta = poly_weights(lams, deg)
        assert eta.sum() == pytest.approx(1.0)                  # a constant is extrapolated exactly
        c = np.arange(1, deg + 2) * 0.1
        assert eta @ np.polyval(c[::-1], lams) == pytest.approx(c[0])    # … and any polynomial of the degree
    assert np.allclose(poly_weights(lams, 3), richardson_weights(lams))
    assert np.allclose(poly_weights(lams, 1), linear_weights(lams))
    with pytest.raises(ValueError):
        poly_weights([1.0, 2.0], 2)
    E = 0.9 - 0.2 * lams + 0.03 * lams ** 2
    assert extrapolate("poly2", lams, E) == pytest.approx(0.9)


def test_variance_amplification_of_the_rules():
    """Σ|η| (the standard deviation of the estimate per unit shot noise) of the rules on the integer and the
    fractional scales: Richardson through closely spaced points is useless; a low-degree fit is not."""
    g = lambda eta: float(np.abs(eta).sum())                    # noqa: E731
    assert g(richardson_weights([1, 3, 5])) == pytest.approx(3.5)
    assert g(richardson_weights(FRACTIONAL)) == pytest.approx(111.0, rel=1e-6)
    assert g(linear_weights(FRACTIONAL)) == pytest.approx(3.6, abs=1e-9)
    assert g(poly_weights(FRACTIONAL, 2)) == pytest.approx(19.0, rel=1e-6)


# ------------------------------------------------------------------ the method and the chain
@pytest.fixture(scope="module")
def cell():
    from qemx.phase.cell import CellSpec, MethodSpec, cell_runner
    q = dict(scales=list(FRACTIONAL), folding="local")
    M = (MethodSpec.make("RAW", "raw"),
         MethodSpec.make("ZNE-Lq", "zne", subset="stratified", rule="poly2", **q),
         MethodSpec.make("ZNE-Lq-bal", "zne", subset="balanced", rule="poly2", **q),
         MethodSpec.make("ZNE-Lq-lin", "zne", subset="nested", rule="linear", **q),
         MethodSpec.make("ZNE-Lq-n1", "zne", subset="nested", rule="poly2", **q),
         MethodSpec.make("ZNE-Lq-n4", "zne", subset="nested", rule="poly2", fold_draws=4, **q),
         MethodSpec.make("CH-ZNE-Lq", "chain", outer="zne", inner="raw", zne_scales=list(FRACTIONAL),
                         zne_folding="local", zne_rule="poly2", zne_subset="stratified"),
         MethodSpec.make("ZNE-bad", "zne", scales=[1.0, 2.0], folding="local", rule="poly2"))
    spec = CellSpec("T", "brickwork", 4, (("layers", 8), ("seed", 0)), layout=(136, 143, 142, 141), scale=2.0,
                    methods=M, seed=5, train_levels=(512,), train_reps=30, train_reps_other=20)
    rec = cell_runner(spec)
    return {r["key"]: r for r in rec["methods"]}


def test_fractional_zne_in_a_cell(cell):
    d = cell["ZNE-Lq"]["details"]
    assert d["subset"] == "stratified" and d["folding"] == "local"
    lam = np.array(d["lams"])
    assert lam[0] == 1 and np.all(np.diff(lam) > 0) and lam[-1] == pytest.approx(2.0)
    assert np.all(np.abs(lam - np.array(FRACTIONAL)) < 2 / d["twoq"][0] + 1e-9)    # within one gate of nominal
    assert cell["ZNE-Lq-bal"]["details"]["subset"] == "balanced"
    assert cell["ZNE-Lq-lin"]["details"]["subset"] == "nested"
    # b is the exact bias of each estimator: the quadratic fit removes most of the raw bias, at a variance cost
    # (Σ|η| = 19 against 1). Which gates a fractional fold repeats moves the bias by about as much as is left of it
    # (here the error-balanced subset keeps almost all of it): see test_fold_draws_average_the_subset_out.
    assert abs(cell["ZNE-Lq"]["b"]) < 0.6 * abs(cell["RAW"]["b"])
    assert cell["ZNE-Lq"]["v"] > 10 * cell["RAW"]["v"]
    assert any("λ_eff" in n for n in cell["ZNE-Lq"]["notes"])


def test_fold_draws_average_the_subset_out(cell):
    """fold_draws = K runs K independent nested subsets at every fractional λ and averages them: E(λ) is the mean
    of the draws, the per-shot variance is unchanged (the shots are split over the draws), the circuit count grows
    to 1 + 3K and the ledger gets one weight per circuit."""
    one, four = cell["ZNE-Lq-n1"], cell["ZNE-Lq-n4"]
    d = four["details"]
    assert d["draws"] == [1, 4, 4, 4] and four["circuits"] == 13 and one["circuits"] == 4
    assert d["E_draws"][1][0] == pytest.approx(one["details"]["E"][1])          # draw 0 is the K = 1 circuit
    assert [float(np.mean(g)) for g in d["E_draws"]] == pytest.approx(d["E"])
    assert max(np.ptp(g) for g in d["E_draws"][1:]) > 1e-3                      # the subsets differ in E(λ) …
    assert four["v"] == pytest.approx(one["v"], rel=0.05)                         # … at the same shot noise
    assert len(four["a"]) == four["m"] == 13 and sum(four["a"]) ** 2 == pytest.approx(four["v"], rel=1e-9)
    assert "×4" in four["title"]


def test_fractional_zne_inside_the_chain_equals_the_method(cell):
    a, b = cell["ZNE-Lq"], cell["CH-ZNE-Lq"]
    assert b["mean"] == pytest.approx(a["mean"], abs=1e-12)
    assert b["details"]["subset"] == "stratified"
    assert b["details"]["lams"] == pytest.approx(a["details"]["lams"])


def test_fold_draws_validation():
    from types import SimpleNamespace

    from qemx.methods.base import REGISTRY
    ctx = SimpleNamespace(spec=_line(6), seed=0)
    for kw in (dict(folding="global", fold_draws=2), dict(folding="local", subset="stratified", fold_draws=2),
               dict(folding="local", fold_draws=0)):
        with pytest.raises(ValueError):
            REGISTRY["zne"].run(ctx, scales=list(FRACTIONAL), rule="linear", **kw)
    from qemx.methods.compose import _chain_params
    assert "zne_fold_draws" not in {p.name for p in _chain_params()}         # one fold per scale inside a chain


def test_quadratic_rule_needs_three_scales(cell):
    r = cell["ZNE-bad"]
    assert r["status"] == "error" and "three scales" in r["error"]


# ------------------------------------------------------------------ the divergence plan
def test_divergence_fractional_roles_and_validation():
    from qemx import divergence as D
    m = D.Method("ZNE-Lq", "zne", "local", FRACTIONAL, "poly2", "stratified")
    assert m.fold_kind == "local.stratified"
    assert D.Method("ZNE-L", "zne", "local").fold_kind == "local"
    assert D.Method("ZNE-G", "zne", "global", subset="stratified").fold_kind == "global"
    base = dict(study="t", kind="divergence", device="ibm_fez", fake="fake_fez", budget=2000, cal_shots=100,
                circuits=[{"family": "qft", "n": 3, "m": None, "s": 1}])
    s = D.suite_from_manifest(dict(base, methods={"Q": {"kind": "zne", "folding": "local", "subset": "nested",
                                                        "scales": [1, 1.333, 1.667, 2], "rule": "poly2"}}))
    assert s.methods[0].subset == "nested" and s.methods[0].rule == "poly2"
    for bad in ({"kind": "zne", "folding": "local", "subset": "diagonal", "scales": [1, 2, 3]},
                {"kind": "zne", "folding": "local", "scales": [1, 2], "rule": "poly2"},
                {"kind": "zne", "folding": "local", "scales": [1, 2, 2]}):
        with pytest.raises(ValueError):
            D.suite_from_manifest(dict(base, methods={"Q": bad}))


def test_divergence_run_spec_matches_fold_local():
    from types import SimpleNamespace

    from qemx import divergence as D
    spec = _line(12)
    inst = SimpleNamespace(spec=spec, seed=17, fold_weights=list(np.linspace(0.002, 0.01, 12)))
    for kind, subset in (("local", "random"), ("local.stratified", "stratified"), ("local.nested", "nested"),
                         ("local.balanced", "balanced")):
        got = D.run_spec(inst, "fold:%s:1.667" % kind)
        want = fold_local(spec, 1.667, 17, subset, inst.fold_weights if subset == "balanced" else None)
        assert got.ops == want.ops, kind
    with pytest.raises(ValueError):
        D.run_spec(inst, "fold:diagonal:2")


def test_plan_skips_fractional_scales_on_too_few_gates():
    """GHZ-2 has one two-qubit gate: λ = 4/3 and 5/3 both fold to 1 (or 3) — the method skips that circuit and
    the plan records why, while every other method still runs on it."""
    from qemx import divergence as D
    from qemx.device import fake_backend
    suite = D.suite_from_manifest(dict(
        study="t", kind="divergence", device="ibm_fez", fake="fake_fez", budget=2000, cal_shots=100,
        methods={"RAW": {"kind": "raw"},
                 "ZNE-Lq": {"kind": "zne", "folding": "local", "subset": "stratified",
                            "scales": [1, 1.333, 1.667, 2], "rule": "poly2"}},
        circuits=[{"family": "ghz", "n": 2, "theta": math.pi / 2, "basis": "Z"},
                  {"family": "qft", "n": 4, "m": None, "s": 1}]))
    plan = D.make_plan(suite, fake_backend("fake_fez"))
    assert len(plan.instances) == 2
    err = [e for e in plan.errors if e.get("method") == "ZNE-Lq"]
    assert len(err) == 1 and "same λ" in err[0]["error"]
    ghz = next(i for i in plan.instances if i.family == "ghz")
    qft = next(i for i in plan.instances if i.family == "qft")
    assert plan.runs_of(ghz.key, "RAW") and not plan.runs_of(ghz.key, "ZNE-Lq")
    q = plan.runs_of(qft.key, "ZNE-Lq")
    assert [r.role for r in q if r.role.startswith("fold:")] == ["fold:local.stratified:%g" % s
                                                                for s in (1, 1.333, 1.667, 2)]
    assert sum(r.shots for r in q) == 2000
