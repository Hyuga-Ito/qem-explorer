"""Every method on a model where its answer is known exactly."""
import numpy as np
import pytest

from qemx import circuit as C
from qemx import compile as CP
from qemx import observables as OBS
from qemx import templates as T
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.methods.base import m3_solve
from qemx.methods.pec import depol_quasi, gamma_of
from qemx.methods.zne import extrapolate, richardson_weights, weights
from qemx.noise import NoiseSpec
from qemx.recipes import PRESETS, compare, preset


def run(tpl, noise, recipe, opts=CP.CompileOptions(), backend="numpy", symmetry=None):
    comp = CP.build(tpl.spec, relabel=tpl.relabel, opts=opts)
    ctx = Context(comp, noise, tpl.observable, recipe, Runner(backend), shots=4096, symmetry=symmetry)
    return ctx.ideal_value(), REGISTRY[recipe.estimator].run(ctx, **recipe.p), ctx


# ------------------------------------------------------------------ ZNE
def test_richardson_reproduces_polynomials():
    lams = [1, 2.5, 4]
    w = richardson_weights(lams)
    assert w.sum() == pytest.approx(1)
    for coef in ([1.0], [0.3, -0.2], [0.9, 0.1, -0.05]):
        E = np.polyval(coef[::-1], lams)
        assert w @ E == pytest.approx(coef[0])


def test_exponential_is_exact_for_exponential_decay_of_either_sign():
    lams = [1, 3, 5]
    for A in (0.8, -0.6):
        E = A * np.exp(-0.2 * np.array(lams))
        assert extrapolate("exponential", lams, E) == pytest.approx(A)


def test_weights_match_finite_differences():
    lams, E = [1, 3, 5], np.array([0.8, 0.6, 0.45])
    w = weights("exponential", lams, E)
    h = 1e-7
    for j in range(3):
        Ep = E.copy(); Ep[j] += h
        assert w[j] == pytest.approx((extrapolate("exponential", lams, Ep) - extrapolate("exponential", lams, E)) / h,
                                     rel=1e-4)


def test_zne_exponential_exact_for_global_depolarizing_like_decay():
    # one qubit, X-basis: <X> decays as f^(number of gates); exponential extrapolation is exact
    spec = C.parse("h 0; x 0; x 0; x 0; x 0", 1)
    tpl = T.Template(spec, OBS.z(0, 1), "")
    comp = CP.build(spec, basis="X")
    noise = NoiseSpec(p1=0.05)
    ctx = Context(comp, noise, OBS.z(0, 1), Recipe.make("zne", rule="exponential"), Runner())
    r = REGISTRY["zne"].run(ctx, **Recipe.make("zne", rule="exponential").p)
    assert r.mean == pytest.approx(ctx.ideal_value(), abs=1e-9)


# ------------------------------------------------------------------ PEC
@pytest.mark.parametrize("p", [0.01, 0.05, 0.2])
def test_pec_gamma_formulas(p):
    f = 1 - p
    assert gamma_of(depol_quasi(p, 2, "inverse")) == pytest.approx((15 / f - 7) / 8)
    assert gamma_of(depol_quasi(p, 1, "inverse")) == pytest.approx((3 / f - 1) / 2)
    assert gamma_of(depol_quasi(p, 2, "compensation")) == pytest.approx((1 + p) / (1 - p))
    assert gamma_of(depol_quasi(p, 2, "compensation")) > gamma_of(depol_quasi(p, 2, "inverse"))


@pytest.mark.parametrize("construction", ["inverse", "compensation"])
def test_pec_is_exact_when_its_model_is_right(construction):
    tpl = T.qft(3, 5, "swap")
    noise = NoiseSpec(p1=0.004, p2=0.03)
    ideal, r, _ = run(tpl, noise, Recipe.make("pec", construction=construction))
    assert r.mean == pytest.approx(ideal, abs=1e-10)


def test_pec_learned_model_with_twirling_is_exact_for_any_gate_noise():
    tpl = T.qft(3, 5, "swap")
    noise = NoiseSpec(p1=0.002, p2=0.02, amp_damp=0.01, over_rot=0.05, zz_phase=0.1)
    ideal, r, _ = run(tpl, noise, Recipe.make("pec", "all", model="learned"),
                      opts=CP.CompileOptions(mode="isa", coupling="line"))
    assert r.mean == pytest.approx(ideal, abs=1e-9)


def test_pec_sampler_matches_variance_model():
    tpl = T.bell()
    ideal, r, _ = run(tpl, NoiseSpec(p2=0.1), Recipe.make("pec"))
    s = r.sampler(20000, 600, np.random.default_rng(3))
    assert s.mean() == pytest.approx(ideal, abs=4 * s.std() / np.sqrt(len(s)))
    assert s.std() == pytest.approx(np.sqrt(r.var1 / 20000), rel=0.12)


def test_pec_on_aer_samples_instances_around_the_exact_mean():
    tpl = T.bell()
    noise = NoiseSpec(p2=0.1)
    ideal, r_np, _ = run(tpl, noise, Recipe.make("pec"))
    _, r_aer, _ = run(tpl, noise, Recipe.make("pec", instances=256), backend="aer")
    assert abs(r_aer.mean - r_np.mean) < 5 * max(r_aer.details["mc_err"], 1e-3)


# ------------------------------------------------------------------ learning
def test_cdr_is_exact_for_global_depolarizing_noise():
    # a depolarizing channel on the whole register maps <O> -> f <O> + (1-f) Tr(O)/d: a line
    spec = C.parse("h 0; rz(0.3) 0; cx 0 1; ry(0.7) 1; depol(0.2) 0 1", 2)
    obs = OBS.z(1, 2)
    tpl = T.Template(spec, obs, "")
    ideal, r, _ = run(tpl, NoiseSpec(), Recipe.make("learning", features="cdr", model="ols", n_train=8))
    assert r.mean == pytest.approx(ideal, abs=1e-9)


@pytest.mark.parametrize("model", ["ols", "ridge", "lasso"])
def test_learning_models_run_and_improve(model):
    tpl = T.qft(3, 5, "swap")
    noise = NoiseSpec(p1=0.002, p2=0.02, amp_damp=0.01)
    isa = CP.CompileOptions(mode="isa", coupling="line")
    ideal, raw, _ = run(tpl, noise, Recipe.make("raw"), opts=isa)
    _, r, _ = run(tpl, noise, Recipe.make("learning", features="nil", model=model, alpha=1e-4, n_train=24), opts=isa)
    assert abs(r.mean - ideal) < 0.1 * abs(raw.mean - ideal)


def test_learning_warns_when_the_target_is_outside_the_training_span():
    tpl = T.qft(3, 5, "swap")                      # abstract: only 4 non-Clifford slots
    _, r, _ = run(tpl, NoiseSpec(p2=0.02, amp_damp=0.01), Recipe.make("learning", features="nil", model="ols"))
    assert any("feature directions" in n for n in r.notes)


# ------------------------------------------------------------------ readout
def test_readout_mitigation_is_exact_for_readout_only_noise():
    tpl = T.qft(3, 5, "swap")
    noise = NoiseSpec(ro01=0.03, ro10=0.07)
    for mode in ("tensored", "m3", "trex"):
        ideal, r, _ = run(tpl, noise, Recipe.make("raw", readout=mode))
        assert r.mean == pytest.approx(ideal, abs=1e-9), mode


def test_m3_equals_full_inverse_on_full_support():
    A = [np.array([[0.97, 0.05], [0.03, 0.95]]), np.array([[0.9, 0.1], [0.1, 0.9]])]
    p = np.array([0.4, 0.3, 0.2, 0.1])
    full = np.kron(np.linalg.inv(A[1]), np.linalg.inv(A[0])) @ p
    assert np.allclose(m3_solve(p, A, 2), full)


# ------------------------------------------------------------------ symmetry, VD
def test_symmetry_verification_removes_single_bit_flips():
    spec = C.parse("h 0; cx 0 1; bitflip(0.1) 1", 2)
    tpl = T.Template(spec, OBS.projector(0, 2), "")
    ideal, raw, _ = run(tpl, NoiseSpec(), Recipe.make("raw"))
    _, r, _ = run(tpl, NoiseSpec(), Recipe.make("symmetry", qubits=[0, 1], eigenvalue=1))
    assert abs(raw.mean - ideal) > 0.01
    assert r.mean == pytest.approx(ideal, abs=1e-12)
    assert r.accept == pytest.approx(0.9)


def test_vd_converges_to_the_dominant_eigenvector():
    tpl = T.ghz(3)
    noise = NoiseSpec(p2=0.05)
    ideal, r2, _ = run(tpl, noise, Recipe.make("vd", M=2))
    _, r4, _ = run(tpl, noise, Recipe.make("vd", M=4))
    _, raw, _ = run(tpl, noise, Recipe.make("raw"))
    assert abs(r4.mean - ideal) <= abs(r2.mean - ideal) + 1e-12 <= abs(raw.mean - ideal) + 1e-12


# ------------------------------------------------------------------ everything together
def test_every_preset_runs():
    tpl = T.qft(3, 5, "swap")
    comp = CP.build(tpl.spec, opts=CP.CompileOptions(mode="isa", coupling="line"))
    from qemx.noise import PRESETS as NP
    ideal, outs, _ = compare(comp, NP["Hardware-realistic (illustrative)"].replace(field_samples=2), tpl.observable,
                             [preset(k) for k in PRESETS])
    for o in outs:
        assert o.result is not None, o.error
        assert np.isfinite(o.result.var1)
        s = o.result.sampler(1024, 8, np.random.default_rng(0))
        assert len(s) == 8
