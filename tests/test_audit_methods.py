"""Regression tests for the estimator/statistics audit (2026-10-10).

Each test reproduces a finding of the audit and pins the fix:
NaN safety of the equal-budget accounting, the empirical-sd fallback for nonlinear estimators with exact samplers,
ZNE's exact shot spending and per-circuit minimum budget, symmetry verification's small-N·P_accept regime, the
vectorized M3 solve, o_eff without a dense 2ⁿ×2ⁿ matrix, and the folding docstring about idle noise.
"""
import math

import numpy as np
import pytest

from qemx import budget as BG
from qemx import compile as CP
from qemx import observables as OBS
from qemx import templates as T
from qemx.circuit import parse
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.methods.base import MethodResult, m3_solve
from qemx.methods.zne import largest_remainder_split
from qemx.noise import NoiseSpec, idle_windows
from qemx.recipes import Outcome, nonlinear_exact, summary
from qemx.transforms import fold_global


def _ctx(tpl, noise, recipe, relabel=None):
    comp = CP.build(tpl.spec, relabel=tpl.relabel if relabel is None else relabel)
    return Context(comp, noise, tpl.observable, recipe, Runner(), shots=4096)


def _run(tpl, noise, est, readout="none", **p):
    rc = Recipe.make(est, readout=readout, **p)
    ctx = _ctx(tpl, noise, rc)
    return REGISTRY[est].run(ctx, **rc.p), ctx


def _sign_changing_zne(on_fail="abstain"):
    """ZNE, exponential rule, whose E(λ) changes sign: with the 'abstain' failure policy the estimate is NaN (audit
    reproduction; this was the only behavior before 1.5.1). The default policy since 1.5.1 is 'linear'."""
    tpl = T.Template(parse("x 0; cx 0 1; cx 0 1", 2), OBS.z(0, 2), "")
    r, ctx = _run(tpl, NoiseSpec(amp_damp=0.12), "zne", rule="exponential", on_fail=on_fail)
    return r, ctx


# ============================================================ (2) NaN safety of the budget accounting
def test_cost_model_maps_unknown_variances_to_infinity():
    r = MethodResult(mean=float("nan"), var1=float("nan"), var_fixed=float("nan"), fixed_shots=100)
    m = BG.CostModel.of(r)
    assert m.v == math.inf and m.u == math.inf and m.w == 0.0 and m.fixed == 100.0
    r2 = MethodResult(mean=0.0, var1=1.0, cal_w=float("nan"), cal_circuits=2)
    assert BG.CostModel.of(r2).w == math.inf
    r3 = MethodResult(mean=0.0, var1=-1e-18)                       # negative rounding stays a zero variance
    assert BG.CostModel.of(r3).v == 0.0
    row = dict(v=float("nan"), u_train=float("nan"), w_cal=float("nan"))
    mr = BG.CostModel.of_row(row)
    assert mr.v == mr.u == mr.w == math.inf
    assert BG.CostModel.of_row(dict(v=1.0, u_train=None)).u == 0.0   # a missing u is no training noise


def test_nan_estimates_never_get_a_finite_budget():
    good = BG.CostModel(v=1.0)
    for bias, m in ((float("nan"), good), (0.0, BG.CostModel(v=float("inf"))),
                    (0.0, BG.CostModel(v=1.0, c=2, w=float("inf"))), (0.0, BG.CostModel(v=1.0, u=float("inf")))):
        assert BG.variance(m, 1e6) == math.inf or not math.isfinite(bias)
        assert BG.rmse(bias, m, 1e6) == math.inf
        assert BG.budget_for(bias, m, 0.01) == math.inf
        assert BG.crossover(bias, m, 0.1, good) is None
        assert BG.crossover(0.1, good, bias, m) is None
        assert np.all(BG.rmse_curve(bias, m, [10, 1e4, 1e8]) == math.inf)
    assert BG.split(BG.CostModel(v=1.0, fixed=float("nan")), 1e6) == (0.0, 0.0)
    assert BG.variance(BG.CostModel(v=1.0, fixed=float("nan")), 1e6) == math.inf
    # finite models are unchanged
    assert BG.budget_for(0.0, good, 0.01) == pytest.approx(1e4, rel=1e-6)


def test_sign_changing_exponential_zne_reports_infinite_cost():
    r, ctx = _sign_changing_zne()
    assert math.isnan(r.mean)
    m = BG.CostModel.of(r)
    b = r.mean - ctx.ideal_value()
    assert m.v == math.inf
    assert BG.budget_for(b, m, 0.01) == math.inf               # was 1 shot before the fix
    assert BG.rmse(b, m, 1e4) == math.inf
    assert BG.crossover(b, m, 0.0, BG.CostModel(v=1.0)) is None
    row = summary(Outcome(Recipe.make("zne", rule="exponential", on_fail="abstain"), r), ctx.ideal_value(), 10000,
                  1.0, samples=r.sampler(9999, 20, np.random.default_rng(0)))
    assert row["RMSE"] == math.inf and row["v"] == math.inf
    assert BG.budget_for(row["bias"], BG.CostModel.of_row(row), 0.01) == math.inf


def test_sign_changing_exponential_zne_falls_back_to_the_linear_fit_by_default():
    """1.5.1 failure policy 'linear' (the default): the exact E(λ) change sign, so the reference is the least-squares
    line through the same E(λ) — no extra shots — and the result says so."""
    from qemx.methods.zne import extrapolate
    r, ctx = _sign_changing_zne(on_fail="linear")
    E, lams = r.details["E"], r.details["lams"]
    assert math.isnan(extrapolate("exponential", lams, E))
    assert r.mean == pytest.approx(extrapolate("linear", lams, E))
    assert r.reference_mean == pytest.approx(r.mean) and r.estimate_kind == "reference"
    assert r.details["reference_rule"] == "linear" and r.failure_policy == "fallback"
    assert any("Failure policy" in n for n in r.notes)
    assert np.isfinite(r.var1)


# ============================================================ (4) empirical-sd fallback for nonlinear estimators
def test_nonlinear_exact_selects_exponential_zne_and_symmetry_only():
    assert nonlinear_exact(Recipe.make("zne", rule="exponential"))
    assert nonlinear_exact(Recipe.make("symmetry"))
    for rc in (Recipe.make("zne"), Recipe.make("zne", rule="linear"), Recipe.make("vd"), Recipe.make("raw"),
               Recipe.make("pec"), Recipe.make("learning")):
        assert not nonlinear_exact(rc)


@pytest.mark.parametrize("estimator,params,expect", [
    ("zne", dict(rule="exponential"), "repeats"),
    ("symmetry", {}, "repeats"),
    ("zne", dict(rule="richardson"), "analytic"),
    ("vd", {}, "analytic"),
])
def test_empirical_sd_fallback_for_nonlinear_exact_samplers(estimator, params, expect):
    rc = Recipe.make(estimator, **params)
    r = MethodResult(mean=0.0, var1=1.0, circuits=3 if estimator == "zne" else 1)
    B = 10000
    sd_an = math.sqrt(1.0 / B)
    x = np.random.default_rng(0).normal(0.0, 3 * sd_an, 120)          # a clear excess over the analytic sd
    row = summary(Outcome(rc, r), 0.0, B, 1.0, samples=x)
    assert row["sd from"] == expect
    if expect == "repeats":
        assert row["sd"] == pytest.approx(np.std(x, ddof=1))
    else:
        assert row["sd"] == pytest.approx(sd_an)
    assert row["u_train"] == 0.0 and row["v"] == 1.0                 # the excess shrinks with B: not booked as u
    # below the 25 % threshold the analytic value stays
    row2 = summary(Outcome(rc, r), 0.0, B, 1.0, samples=np.random.default_rng(1).normal(0.0, sd_an, 120))
    assert row2["sd from"] == "analytic"


def test_learning_fallback_books_the_excess_as_training_noise_as_before():
    rc = Recipe.make("learning")
    r = MethodResult(mean=0.0, var1=1.0, fixed_shots=1000, train_circuits=4)
    B = 11000
    x = np.random.default_rng(0).normal(0.0, 0.05, 120)
    row = summary(Outcome(rc, r), 0.0, B, 1.0, samples=x)
    emp = float(np.std(x, ddof=1))
    assert row["sd from"] == "repeats" and row["sd"] == pytest.approx(emp)
    assert row["u_train"] == pytest.approx(emp ** 2 - 1.0 / (B - 1000))


# ============================================================ (5) ZNE: exact shot spending, one shot per circuit
def test_largest_remainder_split_sums_exactly():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(1, 6))
        a = rng.uniform(0, 1, k) * rng.integers(0, 2, k)
        N = int(rng.integers(k, 60))
        n = largest_remainder_split(a, N)
        assert sum(n) == N and min(n) >= 1
    assert largest_remainder_split([np.nan, 1.0, 1.0], 10) in ((4, 3, 3), (3, 4, 3), (3, 3, 4))
    with pytest.raises(ValueError):
        largest_remainder_split([1, 1, 1], 2)


def _spy(ctx):
    calls = []
    orig = ctx.sample_expvals

    def spy(rr, N, B, rng, o=None):
        calls.append(int(N))
        return orig(rr, N, B, rng, o)
    ctx.sample_expvals = spy
    return calls


@pytest.mark.parametrize("noise", [NoiseSpec(p1=0.004, p2=0.03), NoiseSpec()])   # optimal split / degenerate weights
def test_zne_sampler_spends_exactly_N_shots(noise):
    r, ctx = _run(T.ghz(3), noise, "zne")
    calls = _spy(ctx)
    for N in (3, 4, 10, 1001):
        calls.clear()
        x = r.sampler(N, 5, np.random.default_rng(0))
        assert sum(calls) == N and min(calls) >= 1 and np.isfinite(x).all()
    for N in (1, 2):                                        # fewer shots than folded circuits: no estimate
        calls.clear()
        x = r.sampler(N, 5, np.random.default_rng(0))
        assert calls == [] and np.isnan(x).all()


def test_zne_min_budget_counts_its_circuits():
    r, ctx = _run(T.ghz(3), NoiseSpec(p1=0.004, p2=0.03), "zne")
    m = BG.CostModel.of(r)
    assert m.circuits == 3 and m.n_min == 3 and BG.min_budget(m) == 3
    assert BG.split(m, 2) == (0.0, 0.0) and BG.variance(m, 2) == math.inf
    assert BG.split(m, 3) == (3.0, 0.0)
    assert np.isnan(BG.draw(r, 2, 4, np.random.default_rng(0))).all()
    acc = BG.account(r, 2)
    assert not acc["feasible"] and acc["min_budget"] == 3
    # with readout calibration: one shot per calibration circuit on top
    r2, _ = _run(T.ghz(3), NoiseSpec(p1=0.004, p2=0.03, ro01=0.02, ro10=0.04), "zne", readout="tensored")
    m2 = BG.CostModel.of(r2)
    assert BG.min_budget(m2) == 2 + 3
    for B in (5, 6, 50, 12345):
        N, s = BG.split(m2, B)
        assert N >= 3 and abs(N + m2.c * s - B) < 1e-9
    # constructor stays backward compatible (positional v, fixed, c, w, u; defaults one circuit, no post-selection)
    m0 = BG.CostModel(1.0, 0.0, 2, 0.5, 0.0)
    assert m0.circuits == 1 and m0.accept == 1.0 and BG.min_budget(m0) == 3


# ============================================================ (6) symmetry verification at small N·P_accept
def _sym_case(readout="none"):
    tpl = T.Template(parse("x 0; h 1; cx 1 2", 3), OBS.z(1, 3), "")
    return _run(tpl, NoiseSpec(p1=0.1, p2=0.3, ro01=0.15, ro10=0.15), "symmetry", readout=readout,
                qubits=(1, 2), eigenvalue=1)


def test_symmetry_min_target_shots_and_note():
    r, ctx = _sym_case()
    P = r.accept
    assert 0.6 < P < 0.7
    need = math.ceil(5 / P)
    assert any("⌈5/P_accept⌉ = %d" % need in nt for nt in r.notes)
    m = BG.CostModel.of(r)
    assert m.n_min == need and BG.min_budget(m) == need
    assert BG.variance(m, need - 1) == math.inf and math.isfinite(BG.variance(m, need))
    assert np.isnan(BG.draw(r, need - 1, 3, np.random.default_rng(0))).all()
    # the row carries what of_row needs
    row = summary(Outcome(Recipe.make("symmetry", qubits=[1, 2]), r), 0.0, 1000, 1.0)
    assert BG.min_budget(BG.CostModel.of_row(row)) == need


def test_symmetry_accounting_identity_with_calibration():
    r, ctx = _sym_case("tensored")
    m = BG.CostModel.of(r)
    assert m.c == 2 and m.n_min == math.ceil(5 / r.accept)
    assert BG.min_budget(m) == m.c + m.n_min
    for B in (BG.min_budget(m), BG.min_budget(m) + 1, 100, 10 ** 5):
        N, s = BG.split(m, B)
        assert N >= m.n_min and s >= 1 and abs(N + m.c * s - B) < 1e-9
        assert BG.account(r, B)["feasible"]
    assert not BG.account(r, BG.min_budget(m) - 1)["feasible"]


def test_no_constraint_without_post_selection():
    assert BG.CostModel(v=1.0, accept=1.0).n_min == 1
    assert BG.CostModel(v=1.0, accept=0.0).n_min == 1                # var1 is already infinite there
    assert BG.CostModel(v=1.0, accept=1e-320).n_min == 1e18           # vanishing acceptance stays finite
    assert BG.CostModel(v=1.0, accept=0.5).n_min == 10


@pytest.mark.parametrize("readout", ["none", "tensored", "m3", "trex"])
def test_symmetry_results_unchanged_and_sampler_vectorized(readout):
    r, ctx = _sym_case(readout)
    # the delta-method variance with the dense mitigation matrix (pre-fix formula)
    rr = ctx.run()
    o = ctx.obs.o
    mask = np.zeros(2 ** ctx.n, bool)
    x = np.arange(2 ** ctx.n)
    mask[(((x >> 1) ^ (x >> 2)) & 1) == 0] = True
    Mt = ctx.mitigation_matrix().T
    a, b = Mt @ (mask * o), Mt @ mask.astype(float)
    p = np.clip(ctx.measured(rr), 0, None)
    ea, eb = p @ a, p @ b
    R = ea / eb
    v = (p @ a ** 2 - ea ** 2 - 2 * R * (p @ (a * b) - ea * eb) + R * R * (p @ b ** 2 - eb ** 2)) / eb ** 2
    assert r.var1 == pytest.approx(v, rel=1e-12, abs=1e-14)
    # the sampler equals the per-repeat mitigation it replaced
    N, B = 7, 400
    xs = r.sampler(N, B, np.random.default_rng(5))
    ref = []
    for fr in ctx.sample_freqs(rr, N, B, np.random.default_rng(5)):
        g = ctx.mitigate(fr) * mask
        ref.append(g @ o / g.sum() if g.sum() > 0 else np.nan)
    ref = np.array(ref)
    assert np.array_equal(np.isnan(xs), np.isnan(ref))
    ok = ~np.isnan(ref)
    assert np.max(np.abs(xs[ok] - ref[ok])) < 1e-12


# ============================================================ (7) M3: vectorized reduced matrix
def _m3_reference(p_meas, A_list, n, hamming=None, tol=1e-12):
    """The element-wise implementation the vectorized one replaced."""
    obs = np.flatnonzero(np.abs(p_meas) > tol)
    if len(obs) == 0:
        return p_meas
    m = len(obs)
    Ared = np.empty((m, m))
    for a, x in enumerate(obs):
        for b, y in enumerate(obs):
            if hamming is not None and bin(int(x) ^ int(y)).count("1") > hamming:
                Ared[a, b] = 0.0
                continue
            v = 1.0
            for i in range(n):
                v *= A_list[i][(x >> i) & 1, (y >> i) & 1]
            Ared[a, b] = v
    Ared /= Ared.sum(axis=0, keepdims=True)
    out = np.zeros_like(p_meas)
    out[obs] = np.linalg.solve(Ared, p_meas[obs])
    return out


@pytest.mark.parametrize("n", [1, 3, 6])
def test_m3_solve_matches_elementwise_reference(n):
    rng = np.random.default_rng(n)
    A = [np.array([[1 - e, h], [e, 1 - h]]) for e, h in rng.uniform(0, 0.1, size=(n, 2))]
    for _ in range(4):
        p = rng.dirichlet(np.ones(2 ** n))
        p[rng.random(2 ** n) < 0.3] = 0.0
        p[rng.random(2 ** n) < 0.1] *= -1                   # signed quasi-probabilities keep their negatives
        for hm in (None, 0, 1, 2):
            assert np.max(np.abs(m3_solve(p, A, n, hm) - _m3_reference(p, A, n, hm))) < 1e-12
    assert np.array_equal(m3_solve(np.zeros(2 ** n), A, n), np.zeros(2 ** n))


# ============================================================ (8) o_eff without a dense matrix
def _dense_reference(ctx):
    d = 2 ** ctx.n
    if ctx.recipe.readout == "none":
        return np.eye(d)
    if ctx.recipe.readout == "m3":
        A = ctx.calibration()["A"]
        return np.stack([ctx._tensor_apply([np.linalg.inv(a) for a in A], e) for e in np.eye(d)], 1)
    return np.stack([ctx.mitigate(e) for e in np.eye(d)], 1)


@pytest.mark.parametrize("readout", ["none", "tensored", "m3", "trex"])
def test_o_eff_and_mitigation_matrix_match_the_dense_construction(readout):
    noise = NoiseSpec(p1=0.004, p2=0.03, ro01=0.03, ro10=0.06)
    ctx = _ctx(T.ghz(4), noise, Recipe.make("raw", readout=readout))
    M = _dense_reference(ctx)
    assert np.max(np.abs(ctx.mitigation_matrix() - M)) < 1e-12
    rng = np.random.default_rng(0)
    for o in (ctx.obs.o, rng.normal(size=2 ** ctx.n), (ctx.obs.o > 0).astype(float)):
        assert np.max(np.abs(ctx.o_eff(o) - M.T @ o)) < 1e-12
    # cached, and the cache cannot be corrupted through the returned array
    v1 = ctx.o_eff()
    v1[:] = 99.0
    assert np.max(np.abs(ctx.o_eff() - M.T @ ctx.obs.o)) < 1e-12
    rr = ctx.run()
    assert ctx.var_shot(rr) == pytest.approx(
        float(np.clip(ctx.measured(rr), 0, None) @ (M.T @ ctx.obs.o) ** 2
              - (np.clip(ctx.measured(rr), 0, None) @ (M.T @ ctx.obs.o)) ** 2), rel=1e-12, abs=1e-14)
    # sample_expvals: one product for the linear mitigations, the same numbers as per-repeat mitigation
    x = ctx.sample_expvals(rr, 300, 64, np.random.default_rng(3))
    ref = np.array([ctx.obs.o @ ctx.mitigate(f) for f in ctx.sample_freqs(rr, 300, 64, np.random.default_rng(3))])
    assert np.max(np.abs(x - ref)) < 1e-12


def test_o_eff_respects_a_subclass_mitigation():
    class Odd(Context):
        """A non-per-bit (still linear) mitigation: the fast path must not be used."""
        def mitigate(self, f):
            return 0.5 * f + 0.5 * f[::-1]

    class SameTrex(Context):
        """Overrides mitigate with the per-bit TREX map (like the hardware context): the fast path is fine."""
        def mitigate(self, f):
            return self._tensor_apply([np.linalg.inv(np.array([[1 - p, p], [p, 1 - p]]))
                                       for p in self.calibration()["pbar"]], f)

    noise = NoiseSpec(p2=0.02, ro01=0.03, ro10=0.06)
    tpl = T.ghz(3)
    for cls, ro in ((Odd, "tensored"), (SameTrex, "trex")):
        comp = CP.build(tpl.spec)
        ctx = cls(comp, noise, tpl.observable, Recipe.make("raw", readout=ro), Runner())
        M = np.stack([ctx.mitigate(e) for e in np.eye(8)], 1)
        assert np.max(np.abs(ctx.mitigation_matrix() - M)) < 1e-12
        assert np.max(np.abs(ctx.o_eff() - M.T @ ctx.obs.o)) < 1e-12
        rr = ctx.run()
        x = ctx.sample_expvals(rr, 200, 16, np.random.default_rng(2))
        ref = np.array([ctx.obs.o @ ctx.mitigate(f) for f in ctx.sample_freqs(rr, 200, 16, np.random.default_rng(2))])
        assert np.max(np.abs(x - ref)) < 1e-12


@pytest.mark.parametrize("est,params", [("raw", {}), ("zne", {}), ("pec", {}), ("symmetry", {}),
                                        ("learning", dict(features="nil"))])
@pytest.mark.parametrize("readout", ["tensored", "trex"])
def test_estimator_results_unchanged_by_o_eff(est, params, readout):
    """mean, var1 and the calibration gradient computed through o_eff equal those from the dense matrix."""
    noise = NoiseSpec(p1=0.004, p2=0.03, ro01=0.03, ro10=0.06)
    tpl = T.ising(3) if est == "learning" else T.ghz(3)
    rc = Recipe.make(est, readout=readout, **params)
    ctx_new = _ctx(tpl, noise, rc)
    r_new = REGISTRY[est].run(ctx_new, **rc.p)
    ctx_old = _ctx(tpl, noise, rc)
    dense = _dense_reference(ctx_old)
    ctx_old.mitigation_matrix = lambda: dense
    ctx_old.o_eff = lambda o=None: dense.T @ (ctx_old.obs.o if o is None else o)
    r_old = REGISTRY[est].run(ctx_old, **rc.p)
    assert r_new.mean == pytest.approx(r_old.mean, abs=1e-12)
    assert r_new.var1 == pytest.approx(r_old.var1, rel=1e-10, abs=1e-12)
    assert np.allclose(r_new.cal_grad, r_old.cal_grad, rtol=1e-8, atol=1e-10)


# ============================================================ (9) folding does not scale idle noise by λ
def test_global_folding_scales_idle_noise_differently_and_says_so():
    import qemx.transforms as TR
    assert "idle noise does not scale by λ" in TR.__doc__
    tpl = T.ghz(4)
    spec = CP.build(tpl.spec).implemented

    def idle(sp):
        return sum(b - a + 1 for runs in idle_windows(sp).values() for a, b in runs)
    base = idle(spec)
    assert base > 0
    ratios = [idle(fold_global(spec, lam)) / base for lam in (3, 5)]
    assert ratios != [3.0, 5.0]                                   # e.g. 7 and 13 for GHZ-4
