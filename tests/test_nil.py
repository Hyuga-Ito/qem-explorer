"""Neighbor-informed learning (Chen et al., arXiv:2512.12578): exact regression solvers, neighbor and training
circuits, the estimator's accounting, its samplers, hardware planning and the Direction A pipeline."""
import itertools
import json
import math

import numpy as np
import pytest

from qemx import compile as CP
from qemx import templates as T
from qemx.circuit import CLIFFORD_ANGLE, parse
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.methods import nil as N
from qemx.noise import PRESETS, NoiseSpec
from qemx.transforms import non_clifford_slots

MODELS = ("l1", "lasso", "ridge", "ols", "elasticnet", "nnls")


# ------------------------------------------------------------------ solvers: exact answers on orthonormal designs
def _orthonormal(T_, k, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(T_, k)))
    return Q, rng.normal(size=T_)


def _soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def _project_l1(z, gamma):
    """Euclidean projection on the ℓ1 ball (Duchi et al. 2008): the constrained solution for XᵀX = I."""
    if np.abs(z).sum() <= gamma:
        return z.copy()
    u = np.sort(np.abs(z))[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (css - gamma))[0][-1]
    theta = (css[rho] - gamma) / (rho + 1.0)
    return _soft(z, theta)


@pytest.mark.parametrize("seed", range(6))
def test_solvers_exact_on_orthonormal_designs(seed):
    X, y = _orthonormal(12, 5, seed)
    z = X.T @ y
    lam, l2, gamma = 0.3, 0.7, 0.8
    c, b, info = N.nil_fit(X, y, "lasso", alpha=lam)
    assert info["ok"] and b == 0.0 and np.allclose(c, _soft(z, lam / 2), atol=1e-12)
    c, _, info = N.nil_fit(X, y, "l1", l1_bound=gamma)
    assert info["ok"] and np.allclose(c, _project_l1(z, gamma), atol=1e-12)
    assert np.abs(c).sum() == pytest.approx(min(gamma, np.abs(z).sum()), rel=1e-12)
    c, _, _ = N.nil_fit(X, y, "ridge", alpha=l2)
    assert np.allclose(c, z / (1 + l2), atol=1e-12)
    c, _, _ = N.nil_fit(X, y, "ols")
    assert np.allclose(c, z, atol=1e-12)
    c, _, info = N.nil_fit(X, y, "nnls")
    assert info["ok"] and np.allclose(c, np.maximum(z, 0), atol=1e-12)
    rho = 0.4                                                  # ‖·‖² + α[ρ‖c‖₁ + (1 − ρ)‖c‖²] per coordinate
    c, _, info = N.nil_fit(X, y, "elasticnet", alpha=lam, l1_ratio=rho)
    assert info["ok"] and np.allclose(c, _soft(z, lam * rho / 2) / (1 + lam * (1 - rho)), atol=1e-12)


def test_elastic_net_limits_are_lasso_and_ridge():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(16, 6)), rng.normal(size=16)
    for a in (1e-3, 0.5):
        assert np.allclose(N.nil_fit(X, y, "elasticnet", alpha=a, l1_ratio=1.0)[0], N.nil_fit(X, y, "lasso", alpha=a)[0],
                           atol=1e-11)
        assert np.allclose(N.nil_fit(X, y, "elasticnet", alpha=a, l1_ratio=0.0)[0], N.nil_fit(X, y, "ridge", alpha=a)[0],
                           atol=1e-11)


# ------------------------------------------------------------------ solvers: brute force over active sets and signs
def _brute(X, y, model, lam=0.0, gamma=0.0):
    """Exact optimum by enumerating every (active set, sign pattern): the KKT system of each face, kept if
    sign-consistent; the best objective wins (exponential, tiny k only)."""
    k = X.shape[1]
    best = (math.inf, np.zeros(k))

    def f(c):
        r = y - X @ c
        return float(r @ r + (lam * np.abs(c).sum() if model == "lasso" else 0.0))
    for m in range(0, k + 1):
        for S in itertools.combinations(range(k), m):
            for sg in itertools.product((-1.0, 1.0), repeat=m):
                S_, s_ = np.array(S, int), np.array(sg)
                if model == "lasso":
                    c = N._nil_kkt_lasso(X, y, lam, S_, s_) if m else np.zeros(k)
                else:
                    if m == 0:
                        c = np.zeros(k)
                    else:
                        c, mu = N._nil_kkt_l1(X, y, gamma, S_, s_)
                        if not (math.isfinite(mu) and mu >= -1e-12):
                            c = np.zeros(k)
                            c[S_] = np.linalg.lstsq(X[:, S_], y, rcond=None)[0]
                        if np.abs(c).sum() > gamma * (1 + 1e-9):
                            continue
                if m and np.any(c[np.array(S, int)] * s_ < -1e-12):
                    continue
                if f(c) < best[0]:
                    best = (f(c), c)
    return best


@pytest.mark.parametrize("seed", range(25))
def test_lasso_and_l1_match_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    T_, k = int(rng.choice([5, 8, 16])), int(rng.integers(2, 5))
    att = rng.uniform(0.5, 1.0, k)
    yv = rng.choice([-1.0, 0.0, 0.5, 1.0], size=T_)
    X = yv[:, None] * att + rng.normal(0, 0.05, (T_, k))
    lam, gamma = float(rng.choice([1e-3, 1e-2, 0.1])), float(rng.choice([0.5, 1.0, 2.0]))
    fb, cb = _brute(X, yv, "lasso", lam=lam)
    c, _, info = N.nil_fit(X, yv, "lasso", alpha=lam)
    r = yv - X @ c
    assert info["ok"] and float(r @ r + lam * np.abs(c).sum()) <= fb + 1e-12 * max(1.0, fb)
    fb, cb = _brute(X, yv, "l1", gamma=gamma)
    c, _, info = N.nil_fit(X, yv, "l1", l1_bound=gamma)
    r = yv - X @ c
    assert info["ok"] and np.abs(c).sum() <= gamma * (1 + 1e-12)
    assert float(r @ r) <= fb + 1e-12 * max(1.0, fb)


def test_duplicate_features_are_handled_and_certified():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(16, 4))
    X = np.hstack([base, base[:, :2]])                        # exact duplicates (a Z that commutes to the readout)
    y = base @ np.array([0.7, -0.4, 0.2, 0.1]) + rng.normal(0, 0.01, 16)
    xt = np.concatenate([rng.normal(size=4), [0, 0]])
    xt[4:] = xt[:2]                                           # the target sees the same duplicates
    preds = {}
    for model in MODELS:
        c, b, info = N.nil_fit(X, y, model, alpha=1e-3, l1_bound=2.0)
        assert info["ok"], model
        preds[model] = b + xt @ c
    # duplicates may split their weight arbitrarily, the prediction may not
    c1 = N.nil_fit(X, y, "l1", l1_bound=2.0)[0]
    c2 = N.nil_fit(X[:, :4], y, "l1", l1_bound=2.0)[0]
    assert xt @ c1 == pytest.approx(xt[:4] @ c2, abs=1e-10)


def test_intercept_is_unpenalized_and_shift_equivariant():
    rng = np.random.default_rng(9)
    X, y = rng.normal(size=(16, 5)), rng.normal(size=16)
    for model in MODELS:
        c1, b1, _ = N.nil_fit(X, y, model, alpha=1e-2, l1_bound=1.0, intercept=True)
        c2, b2, _ = N.nil_fit(X, y + 3.0, model, alpha=1e-2, l1_bound=1.0, intercept=True)
        assert np.allclose(c1, c2, atol=1e-10) and b2 == pytest.approx(b1 + 3.0, abs=1e-10), model


def test_warm_start_does_not_change_the_solution():
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(16, 8)), rng.normal(size=16)
    for model in ("l1", "lasso", "elasticnet"):
        ref = N.nil_fit(X, y, model, alpha=1e-2, l1_bound=1.5)
        for _ in range(5):
            Z = X + rng.normal(0, 0.05, X.shape)
            cold = N.nil_fit(Z, y, model, alpha=1e-2, l1_bound=1.5)
            warm = N.nil_fit(Z, y, model, alpha=1e-2, l1_bound=1.5, warm=(ref[0], ref[2]))
            assert np.allclose(cold[0], warm[0], atol=1e-10), model


def _aug(X, y, eps):
    k = X.shape[1]
    return np.vstack([X, np.sqrt(eps) * np.eye(k)]), np.concatenate([y, np.zeros(k)])


def test_certificates_bound_the_excess_objective(monkeypatch):
    """The certificates refer to the stabilized objective (‖y − Xc‖² + ε‖c‖² + penalty); relative to the plain
    objective the stabilized solution is worse by at most ε‖c*‖² (c*: the plain optimum) plus its own gap."""
    rng = np.random.default_rng(8)
    X, y = rng.normal(size=(10, 14)), rng.normal(size=10)    # more features than training circuits
    lasso_f = lambda c: float((y - X @ c) @ (y - X @ c) + 0.05 * np.abs(c).sum())
    l1_f = lambda c: float((y - X @ c) @ (y - X @ c))
    c, _, info = N.nil_fit(X, y, "lasso", alpha=0.05)
    eps = info["eps"]
    assert eps == pytest.approx((N.NIL_STAB * np.linalg.norm(X, 2)) ** 2)
    assert info["ok"] and N._nil_gap_lasso(*_aug(X, y, eps), c, 0.05) <= N.NIL_GAP_REL * (y @ y)
    cl, _, infol = N.nil_fit(X, y, "l1", l1_bound=1.0)
    assert infol["ok"] and N._nil_gap_l1(*_aug(X, y, eps), cl, 1.0) <= N.NIL_GAP_REL * (y @ y)
    assert np.abs(cl).sum() <= 1.0 + 1e-12
    monkeypatch.setattr(N, "NIL_STAB", 0.0)                  # the plain problems (unique here: X in general position)
    c0, _, i0 = N.nil_fit(X, y, "lasso", alpha=0.05)
    assert i0["ok"] and i0["eps"] == 0.0 and N._nil_gap_lasso(X, y, c0, 0.05) <= N.NIL_GAP_REL * (y @ y)
    # c0 is optimal only up to its own certificate (and floating-point roundoff of the objective's evaluation), so the
    # stabilized solution may undercut it by that much: the lower bound is −certificate − roundoff, not exactly 0
    f, f0 = lasso_f(c), lasso_f(c0)
    roundoff = 64 * np.finfo(float).eps * max(1.0, abs(f), abs(f0))
    certificate = N.NIL_GAP_REL * float(y @ y)
    assert -certificate - roundoff <= f - f0
    assert f - f0 <= eps * (c0 @ c0) + 2 * certificate + roundoff
    cl0, _, il0 = N.nil_fit(X, y, "l1", l1_bound=1.0)
    assert il0["ok"] and N._nil_gap_l1(X, y, cl0, 1.0) <= N.NIL_GAP_REL * (y @ y)
    assert -N.NIL_GAP_REL * (y @ y) <= l1_f(cl) - l1_f(cl0) <= eps * (cl0 @ cl0) + 2 * N.NIL_GAP_REL * (y @ y)
    assert np.allclose(c, c0, atol=1e-6) and np.allclose(cl, cl0, atol=1e-6)
    with pytest.raises(ValueError):
        N.nil_fit(X, y, "pls")


# ------------------------------------------------------------------ neighbor and training circuits
def _qft_ctx(recipe, noise=None, seed=3, backend="numpy"):
    tpl = T.build("qft")
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    return Context(comp, noise or PRESETS["Depolarizing"], tpl.observable, recipe, Runner(backend), shots=4096,
                   seed=seed)


def test_pool_insertion_and_nested_draws():
    spec = parse("h 0; cx 0 1; rz(0.3) 1; cx 1 2; t 2; cz 0 2", 3)
    pool2 = N.nil_pool(spec, "2q")
    assert len(pool2) == 3 * 2 * 3                           # 3 two-qubit gates × 2 qubits × X, Y, Z
    assert len(N.nil_pool(spec, "all")) == 3 * (1 + 1 + 1) + 3 * 2 * 3
    nb = N.nil_insert(spec, 1, 1, "y")
    assert len(nb.ops) == len(spec.ops) + 1 and nb.ops[2].name == "y" and nb.ops[2].qubits == (1,)
    assert nb.ops[2].tag == "nil" and nb.ops[:2] == spec.ops[:2] and nb.ops[3:] == spec.ops[2:]
    ctx = _qft_ctx(Recipe.make("nil"))
    m = REGISTRY["nil"]
    small = [f[0] for f in m.neighbors(ctx, n_neighbors=4)]
    big = [f[0] for f in m.neighbors(ctx, n_neighbors=9)]
    assert small[0] == "bare" and set(small) <= set(big) and len(set(big)) == len(big) == 10
    assert [f[0] for f in m.neighbors(ctx, n_neighbors=4, include_bare=False)] == small[1:]
    lab = m.neighbors(ctx, neighbors="fold", scales=(1.0, 3.0))
    assert [f[0] for f in lab] == ["λ=1", "λ=3"] and lab[0][1](ctx.spec) is ctx.spec
    with pytest.raises(ValueError):
        m.neighbors(ctx, neighbors="fold", scales=(0.5, 3.0))


@pytest.mark.parametrize("ensemble", ["design4", "uniform", "near"])
def test_training_circuits(ensemble):
    spec = parse("h 0; rz(0.3) 0; cx 0 1; t 1; ry(1.1) 1; cp(0.7) 0 1; rz(1.5707963267948966) 0", 2)
    slots = non_clifford_slots(spec)
    rng = np.random.default_rng(0)
    for _ in range(20):
        tr = N.nil_training_circuit(spec, rng, ensemble)
        assert len(tr.ops) == len(spec.ops)
        for i, (a, b) in enumerate(zip(spec.ops, tr.ops)):
            if i not in slots:
                assert a == b                                 # Clifford operations untouched (incl. rz(π/2))
            else:
                assert b.qubits == a.qubits and b.tag == "train"
                if ensemble != "uniform":
                    step = CLIFFORD_ANGLE[b.name]
                    assert abs(b.params[0] / step - round(b.params[0] / step)) < 1e-12
        if ensemble != "uniform":
            assert not non_clifford_slots(tr)                 # classically simulable training circuits
    draws = {N.nil_training_circuit(spec, np.random.default_rng(s), "design4").ops[1].params[0] for s in range(200)}
    assert {round(x / (math.pi / 2)) for x in draws} == {0, 1, 2, 3}       # all four angles of the 2-design


# ------------------------------------------------------------------ the estimator
@pytest.mark.parametrize("model", MODELS)
def test_estimator_accounting(model):
    rc = Recipe.make("nil", "none", "none", "tensored", model=model, n_train=8, n_neighbors=5)
    ctx = _qft_ctx(rc)
    r = REGISTRY["nil"].run(ctx, **rc.p)
    d = r.details
    k = d["n_features"]
    assert k == 6 and r.train_circuits == 8 * k and r.fixed_shots == 8 * k * 1024
    c, sig = np.asarray(d["coef"]), np.asarray(d["sigma"])
    # the reference fit (exact training features) obeys the closed forms; mean / var1 are the protocol's moments at
    # the training shots (qemx/stats.py)
    assert r.estimate_kind == "protocol" and r.moments["train_shots"] == 1024
    assert r.mean == pytest.approx(r.moments["mean"], abs=1e-15) and r.var1 == pytest.approx(r.moments["var1"])
    assert r.reference_var1 == pytest.approx(float(np.sum(np.abs(c) * sig)) ** 2, rel=1e-12)
    assert r.circuits == max(int(np.sum(c != 0)), 1)
    assert r.gamma == pytest.approx(np.abs(c).sum() * ctx.readout_gamma(), rel=1e-12)
    assert r.reference_mean == pytest.approx(d["intercept"] + c @ d["xt"], abs=1e-14)
    assert np.allclose(d["fitted"], d["X"] @ c + d["intercept"])
    assert d["solver_ok"] and r.cal_circuits == 2 and len(r.cal_grad) == 2 * ctx.n
    if model == "l1":
        assert np.abs(c).sum() <= 2.0 + 1e-12
    if model == "nnls":
        assert np.all(c >= 0)
    req = Recipe.make("nil", "none", "none", "tensored", model=model, n_train=8, n_neighbors=5, target_split="equal")
    re_ = REGISTRY["nil"].run(_qft_ctx(req), **req.p)
    ce = np.asarray(re_.details["coef"])
    assert re_.circuits == k and re_.reference_var1 == pytest.approx(k * float(np.sum((ce * sig) ** 2)), rel=1e-12)
    assert np.allclose(ce, c)                                 # the split does not change the fit


def test_estimator_is_exact_without_noise():
    rc = Recipe.make("nil", model="l1", l1_bound=2.0, n_train=8, n_neighbors=4)
    ctx = _qft_ctx(rc, noise=NoiseSpec())
    r = REGISTRY["nil"].run(ctx, **rc.p)
    # every training circuit's bare feature equals its label, so c = e_bare fits exactly (‖c‖₁ = 1 ≤ γ); the
    # stabilization ε‖c‖² (nil_fit) shrinks it by a factor 1 − O(ε/‖x_bare‖²), ε ≈ 1e-10·‖X‖₂²: exact up to 10ε
    eps = r.details["eps"]
    assert 0 < eps < 1e-8
    assert np.abs(r.details["X"] @ r.details["coef"] - r.details["y_train"]).max() < 10 * eps
    assert r.reference_mean == pytest.approx(ctx.ideal_value(), abs=10 * eps)
    # the protocol (finite training shots) refits on noisy features: its mean deviates by at most a few training sd
    assert abs(r.mean - ctx.ideal_value()) < 5 * math.sqrt(r.var_fixed) + 5 * r.moments["mean_mcse"] + 1e-12
    # 'bare' may have twins (a Pauli that acts trivially on every training circuit): the group shares weight 1
    names, c = list(r.details["names"]), np.asarray(r.details["coef"])
    grp = next((g for g in r.details["twins"] if "bare" in g), ["bare"])
    on = np.isin(names, grp)
    assert np.allclose(c[on], 1.0 / on.sum(), atol=10 * eps) and np.abs(c[~on]).max() < 10 * eps


def test_clifford_and_degenerate_circuits_fall_back_to_unmitigated():
    tpl = T.build("bell")
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    rc = Recipe.make("nil")
    ctx = Context(comp, PRESETS["Depolarizing"], tpl.observable, rc, Runner(), seed=1)
    r = REGISTRY["nil"].run(ctx, **rc.p)
    assert "Clifford" in r.notes[0] and r.fixed_shots == 0 and r.mean == pytest.approx(ctx.expval(ctx.run()))


def test_law_of_total_variance_and_sampler():
    rc = Recipe.make("nil", "none", "none", "none", model="lasso", alpha=1e-3, n_train=8, n_neighbors=5,
                     train_shots=256)
    ctx = _qft_ctx(rc, noise=NoiseSpec(p1=0.01, p2=0.05))
    r = REGISTRY["nil"].run(ctx, **rc.p)
    est, v1 = r.train_resampler(256, 3000, np.random.default_rng(1))
    u, v = float(np.var(est, ddof=1)), float(np.mean(v1))
    N_t = 2000
    x = r.sampler(N_t, 3000, np.random.default_rng(2))
    assert np.all(np.isfinite(x))
    pred = v / N_t + u
    se = pred * math.sqrt(2.0 / 3000) * 2.0
    assert abs(float(np.var(x, ddof=1)) - pred) < 4 * se
    assert abs(float(np.mean(x)) - float(np.mean(est))) < 4 * math.sqrt(pred / 3000) * 1.5


def test_recalibrated_refit_paths_agree(monkeypatch):
    rc = Recipe.make("nil", "none", "none", "tensored", model="l1", n_train=6, n_neighbors=3)
    out = []
    for vec in (True, False):
        monkeypatch.setattr(N, "_NIL_VECTOR_REFIT", vec)
        ctx = _qft_ctx(rc)
        r = REGISTRY["nil"].run(ctx, **rc.p)
        th = np.asarray(r.cal_theta) * 1.3
        out.append((r.cal_fun(ctx._mitigator(th)), r.cal_fun(ctx._mitigator(np.asarray(r.cal_theta))),
                    r.reference_mean, np.asarray(r.cal_grad)))
    assert out[0][0] == pytest.approx(out[1][0], abs=1e-12)
    assert out[0][1] == pytest.approx(out[0][2], abs=1e-12) and out[1][1] == pytest.approx(out[1][2], abs=1e-12)
    assert np.allclose(out[0][3], out[1][3], atol=1e-6)


def test_sampler_needs_one_shot_per_used_circuit():
    rc = Recipe.make("nil", model="ridge", n_train=4, n_neighbors=5, train_shots=64)
    ctx = _qft_ctx(rc)
    r = REGISTRY["nil"].run(ctx, **rc.p)
    x = r.sampler(3, 50, np.random.default_rng(0))             # 6 circuits with c ≠ 0, 3 shots: no estimate
    assert np.all(np.isnan(x))


def test_hardware_plan_requires_the_equal_split():
    from qemx import hardware as H
    tpl = T.build("qft")
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    opt = Recipe.make("nil", n_train=4, n_neighbors=3, train_shots=64)
    eq = Recipe.make("nil", n_train=4, n_neighbors=3, train_shots=64, target_split="equal")
    plan = H.make_plan(comp, tpl.observable, [opt, eq], 200000, seed=1)
    assert opt.title() in plan.skipped and "equal" in plan.skipped[opt.title()]
    a = plan.alloc[1]
    assert a["training"] == 4 * 4 * 64
    tgt = [r for r in plan.requests.values() if r.tag == 1 and r.role == "target"]
    assert len(tgt) == 4 and max(r.shots for r in tgt) - min(r.shots for r in tgt) <= 1


def test_recipe_titles_and_registry():
    assert REGISTRY["nil"].family == "learning"
    assert Recipe.make("nil").title() == "NIL (ℓ1≤2)"
    assert Recipe.make("nil", model="ridge", alpha=1e-3, n_train=32).title() == "NIL (Ridge α=0.001, T=32)"
    assert Recipe.make("nil", neighbors="fold", l1_bound=5.0).title() == "NIL-ZNE (ℓ1≤5, λ=1,3,5)"
    assert Recipe.make("learning", features="nil", model="lasso").title().startswith("FR (")


# ------------------------------------------------------------------ Direction A
def test_nil_manifest_mirrors_run_two():
    import yaml
    from qemx.phase import manifest as M
    old = M.load("manifests/direction_A_v1.yaml")
    new = M.load("manifests/direction_A_nil_v1.yaml")
    for k_old, k_new in (("RAW", "RAW"), ("ZNE-R", "ZNE-R"), ("ZNE-E", "ZNE-E"), ("PEC", "PEC"), ("CDR", "CDR"),
                         ("VD", "VD"), ("NIL-L", "FR-L"), ("NIL-R", "FR-R")):
        assert old["methods"][k_old] == new["methods"][k_new]
    assert old["defaults"] == new["defaults"] and old["circuits"] == new["circuits"]
    so = {s["name"]: s for s in old["stages"]}
    for s in new["stages"]:
        if s["name"] in so:
            a, b = dict(so[s["name"]]), dict(s)
            a.pop("methods"), b.pop("methods")
            assert a == b, s["name"]
    cells_o = {(c.stage, c.family, c.n, c.fparams, c.scale, c.readout, c.coherent_fraction): c
               for c in M.expand(old, stages=["A1-primary"])}
    cells_n = M.expand(new, stages=["A1-primary"])
    assert len(cells_n) == len(cells_o)
    for c in cells_n:
        o = cells_o[(c.stage, c.family, c.n, c.fparams, c.scale, c.readout, c.coherent_fraction)]
        assert (c.seed, c.layout, c.train_levels, c.opt_level, c.device) == (o.seed, o.layout, o.train_levels,
                                                                             o.opt_level, o.device)
    # the validated 15 % is drawn by the hash of the cell id, which includes the method list: another subset
    assert abs(sum(1 for c in cells_n if c.validate_budgets) - sum(1 for c in cells_o.values() if c.validate_budgets)) == 0
    keys = [m for m in new["methods"] if m.startswith("NIL")]
    assert not {"NIL-L", "NIL-R"} & set(keys)                 # never reuse run 2's feature-regression keys


@pytest.fixture(scope="module")
def nil_cell():
    from qemx.phase.cell import CellSpec, MethodSpec, cell_runner
    methods = (MethodSpec.make("RAW", "raw"),
               MethodSpec.make("NIL-G2", "nil", model="l1", l1_bound=2.0, n_train=8, n_neighbors=5),
               MethodSpec.make("NIL-RIDGE", "nil", model="ridge", alpha=1e-3, n_train=8, n_neighbors=5),
               MethodSpec.make("NIL-ZNE", "nil", neighbors="fold", model="l1", l1_bound=5.0, n_train=8))
    spec = CellSpec(stage="test", family="ghz", n=4, fparams=(("theta", 1.0471975511965976),),
                    layout=(136, 143, 142, 141), engine="E2", scale=1.0, methods=methods, seed=7,
                    train_levels=(256, 1024), train_reps=300, train_reps_other=200,
                    validate_budgets=(10 ** 6,), validate_reps=600, validate_policy=("fixed", "opt"))
    return spec, cell_runner(spec)


def test_nil_cell_rows_levels_and_ledger(nil_cell):
    from qemx.phase.cell import cost_of
    from qemx.phase.ledger import PIPELINE_FIXED, allocate
    from qemx.phase.store import clean, method_rows, validate_record
    spec, rec = nil_cell
    validate_record(json.loads(json.dumps(clean(rec), allow_nan=False)))
    rows = {r["key"]: r for r in rec["methods"]}
    assert all(r["status"] == "ok" for r in rows.values()), [r.get("error") for r in rows.values()]
    for key in ("NIL-G2", "NIL-RIDGE", "NIL-ZNE"):
        r = rows[key]
        k = r["details"]["n_features"]
        assert r["T"] == 8 * k and r["s_tr"] == 1024 and r["F"] == 8 * k * 1024 and r["a"] == []
        assert r["m"] == max(r["details"]["nonzero"], 1) and "nonlinear" in r["flags"]
        assert {l["s_tr"] for l in r["levels"]} == {256, 1024} and r["details"]["solver_ok"]
        c = cost_of(r)
        al = allocate(c, 10 ** 6, PIPELINE_FIXED)
        assert al.feasible and al.N + c.T * al.s_tr + c.c * al.s == 10 ** 6
    assert rows["NIL-ZNE"]["details"]["n_features"] == 3
    assert any(g["scales"] == [1.0, 3.0, 5.0] for g in rec["guards"])
    mr = {r["method"]: r for r in method_rows(rec)}
    assert mr["NIL-G2"]["x_model"] == "l1" and mr["NIL-G2"]["x_solver_ok"] is True


def test_nil_cell_validation(nil_cell):
    _, rec = nil_cell
    vals = [v for v in rec["validation"] if v["method"].startswith("NIL") and v.get("feasible") and "mc" in v]
    assert len(vals) == 6
    for v in vals:
        assert v["identity"] and v["mc"]["frac_nonfinite"] == 0.0
        assert abs(v["z_bias"]) <= 4.0 and abs(v["z_rmse"]) <= 4.0, v


def test_nil_cell_levels_measure_calibration_and_validation_is_joint(nil_cell):
    from qemx.phase.cell import cost_of
    _, rec = nil_cell
    for r in rec["methods"]:
        if r["estimator"] != "nil":
            continue
        assert r["c"] > 0 and r["w_run"] is not None
        for l in r["levels"]:
            assert l["ok"] and l["w"] >= 0 and l["se_w"] >= 0 and l["w_pairs"] == l["reps"]
        own = next(l for l in r["levels"] if l["s_tr"] == r["s_tr"])
        assert r["w"] == own["w"] and r["u"] == own["u"]
        assert all(len(t) == 5 for t in cost_of(r).levels)
    vals = [v for v in rec["validation"] if v["method"].startswith("NIL") and "mc" in v]
    assert vals and all(v["calibration_mc"].startswith("joint") for v in vals)
    assert all("se_w" in v["analytic"] for v in vals)


# ------------------------------------------------------------------ audit fixes (v1.4)
def test_twins_share_weight_and_order_does_not_matter():
    rng = np.random.default_rng(11)
    base = rng.normal(size=(16, 3))
    y = base @ np.array([0.8, -0.3, 0.1]) + rng.normal(0, 0.01, 16)
    X = np.hstack([base, base[:, :1]])                     # column 3 equals column 0 on every training circuit …
    xt = np.array([0.4, 0.2, -0.1, 0.9])                  # … but not on the target
    preds = []
    for perm in ([0, 1, 2, 3], [3, 1, 2, 0], [1, 3, 0, 2]):
        for model in MODELS:
            c, b, info = N.nil_fit(X[:, perm], y, model, alpha=1e-3, l1_bound=2.0)
            assert info["ok"], model
            cc = np.zeros(4)
            cc[perm] = c
            assert cc[0] == pytest.approx(cc[3], rel=1e-6, abs=1e-12), (model, perm)   # equal split
            preds.append((model, b + xt @ cc))
    for model in MODELS:
        v = [p for m, p in preds if m == model]
        assert max(v) - min(v) < 1e-10, model
    assert N._nil_twins(X) == [[0, 3]]


def test_l1_interior_solution_has_zero_multiplier():
    rng = np.random.default_rng(12)
    X, y = rng.normal(size=(20, 4)), rng.normal(size=20)
    c_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    c, _, info = N.nil_fit(X, y, "l1", l1_bound=10 * np.abs(c_ls).sum())
    assert info["lam"] == 0.0 and np.allclose(c, c_ls, atol=1e-10)


@pytest.mark.parametrize("est,params", [("nil", dict(model="l1", n_train=6, n_neighbors=3)),
                                        ("nil", dict(model="ridge", n_train=6, n_neighbors=3, target_split="equal")),
                                        ("learning", dict(features="cdr", model="ols", n_train=8)),
                                        ("learning", dict(features="nil", model="ridge", n_train=8))])
def test_joint_calibration_samplers_pair_up(est, params):
    rc = Recipe.make(est, "none", "none", "tensored", **params)
    ctx = _qft_ctx(rc, noise=PRESETS["Hardware-realistic (illustrative)"])
    r = REGISTRY[est].run(ctx, **rc.p)
    assert r.joint_calibration and r.cal_w is not None and r.cal_w >= 0
    th0 = np.tile(np.asarray(r.cal_theta), (40, 1))
    a = r.sampler(5000, 40, np.random.default_rng(1))
    b = r.sampler(5000, 40, np.random.default_rng(1), theta=th0)
    assert np.allclose(a, b, atol=1e-10)
    e0, v0 = r.train_resampler(256, 30, np.random.default_rng(2))
    e1, v1 = r.train_resampler(256, 30, np.random.default_rng(2), theta=th0[:30])
    assert np.allclose(e0, e1, atol=1e-10) and np.allclose(v0, v1)
    with pytest.raises(ValueError):                         # 40 draws for 30 replicates: refused, not truncated
        r.train_resampler(256, 30, np.random.default_rng(2), theta=th0)
    with pytest.raises(ValueError):
        r.sampler(5000, 40, np.random.default_rng(1), theta=th0[:, :-1])
    th = ctx.sample_cal_theta(2000, 30, np.random.default_rng(3))
    e2, _ = r.train_resampler(256, 30, np.random.default_rng(2), theta=th)
    assert np.all(np.isfinite(e2)) and not np.allclose(e0, e2)


def test_calibration_w_agrees_with_first_order_where_the_fit_is_linear():
    rc = Recipe.make("learning", "none", "none", "tensored", features="cdr", model="ols", n_train=16)
    ctx = _qft_ctx(rc, noise=PRESETS["Hardware-realistic (illustrative)"])
    r = REGISTRY["learning"].run(ctx, **rc.p)
    g, th = np.asarray(r.cal_grad), np.asarray(r.cal_theta)
    w1 = float(np.sum(g ** 2 * th * (1 - th)))
    assert r.cal_w == pytest.approx(w1, rel=0.25)          # 500 Monte-Carlo pairs: ±6 %, plus training-noise effects
    from qemx.budget import w_cal
    assert w_cal(r) == r.cal_w


def test_learning_training_resampler_applies_the_fallback_per_level():
    rc = Recipe.make("learning", features="cdr", model="ols", n_train=8, train_shots=1024)
    ctx = _qft_ctx(rc)
    r = REGISTRY["learning"].run(ctx, **rc.p)
    raw = ctx.expval(ctx.run())
    est, v1 = r.train_resampler(1, 20, np.random.default_rng(0))          # one shot per training circuit
    assert np.allclose(est, raw) and np.allclose(v1, ctx.var_shot(ctx.run()))


def test_nil_declares_hardware_support():
    m = REGISTRY["nil"]
    assert m.hardware_supported(dict(target_split="optimal"))[0] is False
    assert m.hardware_supported(dict(target_split="equal")) == (True, "")


def test_crn_key_ignores_the_rule_only():
    keys = set()
    for model in ("l1", "ridge", "ols"):
        rc = Recipe.make("nil", model=model, n_train=6, n_neighbors=3)
        keys.add(REGISTRY["nil"].run(_qft_ctx(rc), **rc.p).details["crn_key"])
    assert len(keys) == 1
    rc = Recipe.make("nil", model="l1", n_train=6, n_neighbors=4)
    assert REGISTRY["nil"].run(_qft_ctx(rc), **rc.p).details["crn_key"] not in keys


def test_calibration_maps_apply_to_many_observables_at_once():
    rc = Recipe.make("raw", "none", "none", "tensored")
    ctx = _qft_ctx(rc, noise=PRESETS["Hardware-realistic (illustrative)"])
    th = ctx.sample_cal_theta(2000, 3, np.random.default_rng(1))
    O = np.random.default_rng(2).normal(size=(2 ** ctx.n, 5))
    for t in th:
        both = ctx.o_eff_theta(t, O)
        cols = np.stack([ctx.o_eff_theta(t, O[:, j]) for j in range(5)], 1)
        assert both.shape == O.shape and np.allclose(both, cols, rtol=1e-13, atol=1e-13)
    assert np.allclose(ctx.o_eff_theta(ctx.cal_theta()), ctx.o_eff(), atol=1e-12)
