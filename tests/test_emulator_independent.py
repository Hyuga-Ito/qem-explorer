"""1.8.0 (review finding F8): emulated trials vs an independently implemented shot-level reference
(qemx/selection/reference.py) for the candidates whose labels drive decisions — exponential ZNE and its bounded
variant, CDR, FR, NIL, symmetry verification, ZNE∘SV, TREX — including stress cases: fit failures near a small
signal, per-replicate CDR fallback at few training shots, zero-acceptance post-selection, near-singular finite
calibrations. Distributions are compared (mean z score, variance-ratio F test), never sample by sample.
RAW/RO/ZNE-L/ZNE-R/PEC and the transform variants use emulate.direct_trials (1.6.0's direct execution)."""
import numpy as np
import pytest

Z_MAX, P_MIN = 4.5, 1e-4          # per comparison; ~25 comparisons here, so a false alarm is rare


def _prep(spec, keys, seed=5):
    from qemx.selection import emulate as EM
    from qemx.selection import task as TK
    return EM.prepare(TK.build(spec), keys, seed=seed, designs=1)


def _check(c):
    assert abs(c["z_mean"]) <= Z_MAX and c["p_var"] >= P_MIN, c


@pytest.fixture(scope="module")
def ghz_z():
    from qemx.selection import task as TK
    return _prep(TK.make_spec("ghz", 3, theta=1.1, basis="Z", observable="z_mid"),
                 ["RO", "SV", "ZNE-L∘SV", "RO-TREX", "ZNE-E", "ZNE-E-B"])


@pytest.mark.parametrize("key,B", [("SV", 4000), ("ZNE-L∘SV", 4000), ("RO-TREX", 4000), ("ZNE-E", 4000),
                                   ("ZNE-E-B", 4000), ("RO", 300)])
def test_postselection_trex_and_zne_match_the_reference(ghz_z, key, B):
    from qemx.selection import reference as RF
    _check(RF.compare(ghz_z, key, B, 400, np.random.default_rng(11)))


def test_zero_acceptance_and_fit_failures_are_reproduced():
    """Strong noise (Kyiv × 8, P_accept ≈ 0.68) and two target shots — below the allocation rule's ⌈5/P⌉, on purpose:
    about 10 % of SV repeats keep no shot (nonresponse), at the reference's rate; a small signal makes the exponential
    fit fail on sampled data, so its declared linear fallback is exercised."""
    import dataclasses
    from qemx.selection import emulate as EM
    from qemx.selection import protocols as PR
    from qemx.selection import reference as RF
    from qemx.selection import task as TK
    prep = _prep(TK.make_spec("ghz", 6, theta=1.5, basis="Z", observable="z_mid", device="fake_kyiv@s=8"),
                 ["SV", "ZNE-E", "ZNE-E-B"])
    a = PR.allocate(PR.PORTFOLIO["SV"], prep.supplied["SV"], 200)
    tiny = dataclasses.replace(a, N=2, ledger=dict(a.ledger, execution=2))
    c = RF.compare(prep, "SV", 200, 1500, np.random.default_rng(3), alloc=tiny)
    assert c["nonresponse_emulated"] > 0.03
    assert abs(c["nonresponse_emulated"] - c["nonresponse_reference"]) < 5 * np.sqrt(
        c["nonresponse_emulated"] * (1 - c["nonresponse_emulated"]) / 1500 * 2)
    _check(c)
    alloc = PR.allocate(PR.PORTFOLIO["ZNE-E"], prep.supplied["ZNE-E"], 200)
    _, flags = EM._run(PR.PORTFOLIO["ZNE-E"], prep.truth["ZNE-E"], prep.ctx_truth["ZNE-E"], alloc, 600,
                       np.random.default_rng(4))
    assert flags["fit_fallback"].mean() > 0.05                  # the stress case does exercise the fallback
    _check(RF.compare(prep, "ZNE-E", 200, 600, np.random.default_rng(5)))
    cb = RF.compare(prep, "ZNE-E-B", 200, 600, np.random.default_rng(5))
    _check(cb)
    assert cb["var_emulated"] <= RF.compare(prep, "ZNE-E", 200, 600, np.random.default_rng(5))["var_emulated"] + 1e-12


@pytest.fixture(scope="module")
def hea():
    from qemx.selection import task as TK
    return _prep(TK.make_spec("hea", 3, layers=2, angle_seed=3), ["CDR", "FR", "NIL", "ZNE-E"])


@pytest.mark.parametrize("key,B", [("CDR", 20000), ("FR", 20000), ("ZNE-E", 20000)])
def test_learning_estimators_match_the_reference(hea, key, B):
    from qemx.selection import reference as RF
    _check(RF.compare(hea, key, B, 300, np.random.default_rng(12)))


def test_cdr_per_replicate_fallback_matches_the_reference():
    """B = 2048: 32 shots per training circuit on a deep, strongly damped circuit (Kyiv × 6) — the declared rule
    (no training direction above the shot-noise edge → the mitigated target value) fires in ~17 % of replicates."""
    from qemx.selection import emulate as EM
    from qemx.selection import protocols as PR
    from qemx.selection import reference as RF
    from qemx.selection import task as TK
    prep = _prep(TK.make_spec("hea", 3, layers=4, angle_seed=1, device="fake_kyiv@s=6"), ["CDR"])
    alloc = PR.allocate(PR.PORTFOLIO["CDR"], prep.supplied["CDR"], 2048)
    assert alloc.s_tr == PR.TRAIN_MIN
    _, fl = EM._run(PR.PORTFOLIO["CDR"], prep.truth["CDR"], prep.ctx_truth["CDR"], alloc, 300,
                    np.random.default_rng(8))
    assert 0.05 < fl["learn_fallback"].mean() < 0.5
    _check(RF.compare(prep, "CDR", 2048, 400, np.random.default_rng(14)))


@pytest.mark.slow
def test_nil_matches_the_reference_with_its_own_l1_solver(hea):
    from qemx.selection import reference as RF
    _check(RF.compare(hea, "NIL", 20000, 300, np.random.default_rng(13)))


def test_reference_l1_solver_and_mitigation_map_are_correct():
    from qemx.selection import reference as RF
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 6))
    y = X @ np.array([0.5, -0.3, 0, 0, 0.2, 0]) + 0.01 * rng.normal(size=16)
    c = RF.l1_bounded_ls(X, y, 10.0, 0.0)                    # inactive bound: least squares
    assert np.allclose(c, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-8)
    c2 = RF.l1_bounded_ls(X, y, 0.4, 0.0)                    # active bound: on the ball, KKT stationarity
    assert abs(np.abs(c2).sum() - 0.4) < 1e-9
    g = -2 * X.T @ (X @ c2 - y)
    act = np.abs(c2) > 1e-9
    assert np.allclose(g[act], np.max(np.abs(g)) * np.sign(c2[act]), atol=1e-6)
    # tensored inverse map: (A⁻¹)ᵀo applied to a distribution equals o·A⁻¹p
    n, th = 3, np.array([[0.02, 0.05], [0.01, 0.03], [0.04, 0.02]])
    o = rng.normal(size=2 ** n)
    A = np.array([[1.0]])
    for i in reversed(range(n)):
        e, h = th[i]
        A = np.kron(A, np.array([[1 - e, h], [e, 1 - h]]))
    p = rng.dirichlet(np.ones(2 ** n))
    assert np.isclose(RF.mitigation_values(o, th, "tensored", n) @ p, o @ np.linalg.solve(A, p))


@pytest.mark.parametrize("key", ["RAW", "RO", "ZNE-L", "ZNE-R", "PEC", "RO-TW", "RO-DD", "ZNE-L-TW", "PEC-TW"])
def test_linear_candidates_and_transforms_match_direct_execution(key):
    from qemx.selection import emulate as EM
    from qemx.selection import task as TK
    prep = _prep(TK.make_spec("ising", 3, steps=2), [key])
    rng = np.random.default_rng(21)
    B, R = (400, 120) if key.startswith("PEC") else (3000, 400)    # PEC: one instance circuit simulated per shot
    em = EM.trials(prep, key, B, R, rng)
    di = EM.direct_trials(prep, key, B, R, rng)
    _check(EM.compare_laws(em.estimates, di))
