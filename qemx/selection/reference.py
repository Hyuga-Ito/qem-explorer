r"""Independent shot-level reference executions of the selection candidates (1.8.0; review finding F8).

The trial emulator (``emulate``) draws a protocol's complete trials from each circuit's exact output law through the
methods' own samplers. Agreement of two wrappers around the same estimator code would not validate it. This module
re-implements, separately, everything *after* the physics: shot-by-shot sampling (one categorical draw per shot),
the finite readout calibration, the readout-mitigation map, every estimator (ZNE fits, CDR/FR regressions with
their declared fallback rule, NIL's ℓ1-bounded regression by its own solver, symmetry-verified ratios, TREX's
division by calibration values, chained ZNE∘SV) and the declared post-processing. What it shares with the emulator
is only (a) the exact output distribution of every executed circuit (the simulator: the physics under test is the
same), (b) the protocol's design — which circuits are run (CDR's training circuits, NIL's neighbors), constructed by
the method's design code with the same seed — and (c) the integer allocation of the budget. A difference between
the two trial laws beyond Monte Carlo error is then a defect of the emulation or of an estimator implementation.

``compare(prep, key, B, R, rng)`` returns the two-sample comparison of ``emulate`` and reference trials of one
candidate and design (``emulate.compare_laws``: mean z score, variance-ratio F test). Small widths and budgets only
(per-shot draws are slow by design).
"""
from __future__ import annotations

import math

import numpy as np

from . import protocols as PR


# ------------------------------------------------------------------------------------------------ shots and readout
def counts(p, N: int, rng) -> np.ndarray:
    """Counts of N single-shot categorical draws from distribution p."""
    p = np.clip(np.asarray(p, float), 0, None)
    p = p / p.sum()
    return np.bincount(rng.choice(len(p), size=int(N), p=p), minlength=len(p)).astype(float)


def _bits(n):
    return (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1


def calibration_draw(ctx, s: int, rng, mode: str) -> np.ndarray | None:
    """One finite calibration: per bit (P(1|0), P(0|1)) from s shots on |0…0⟩ and on X on every qubit (tensored), or
    the symmetrized flip rate p̄ per bit from s shots of the twirled |0…0⟩ circuit (TREX, drawn by flipping each
    bit with probability ½ before the asymmetric readout and flipping it back). None when there is no readout
    mitigation. A singular draw (1 − e − h = 0, or 1 − 2p̄ = 0) is returned as no correction (the declared fallback)."""
    from ..circuit import CircuitSpec, Op
    if mode == "none":
        return None
    n = ctx.n
    bits = _bits(n)
    zero = CircuitSpec(n, (), "cal-0")
    if mode == "tensored":
        one = CircuitSpec(n, tuple(Op("x", (q,), (), "gate", "prep") for q in range(n)), "cal-1")
        c0 = counts(ctx.runner.run(zero, ctx.noise, ctx.mm, role="calibration").probs_meas, s, rng)
        c1 = counts(ctx.runner.run(one, ctx.noise, ctx.mm, role="calibration").probs_meas, s, rng)
        e = c0 @ bits / float(s)
        h = c1 @ (1 - bits) / float(s)
        th = np.stack([e, h], 1)
        return np.zeros_like(th) if np.any(np.abs(1 - e - h) < 1e-12) else th
    if mode == "trex":
        pre = ctx.runner.run(zero, ctx.noise, ctx.mm, role="calibration").probs      # before readout
        pb = _trex_readout_draws(ctx, pre, s, rng) @ bits / float(s)
        return np.zeros(n) if np.any(np.abs(1 - 2 * pb) < 1e-12) else pb
    raise ValueError(mode)


def _readout_flips(ctx):
    """Per classical bit (P(1|0), P(0|1)) of the device's readout (the qubit behind the bit, via the measure map)."""
    dm = ctx.runner.device_model
    out = []
    for i in range(ctx.n):
        q = dm.qubits[int(ctx.mm.get(i, i))]
        out.append((q.p01, q.p10))
    return out


def _trex_readout_draws(ctx, pre, N: int, rng) -> np.ndarray:
    """Counts of N shots of a TREX-measured circuit: an outcome x from the pre-readout law, a random flip mask r
    (X before measurement), the device's asymmetric readout of x ⊕ r, then the classical flip back by r."""
    n = ctx.n
    flips = _readout_flips(ctx)
    x = rng.choice(2 ** n, size=int(N), p=np.clip(pre, 0, None) / np.clip(pre, 0, None).sum())
    xb = (x[:, None] >> np.arange(n)[None, :]) & 1
    r = rng.integers(0, 2, size=(int(N), n))
    phys = xb ^ r
    u = rng.random((int(N), n))
    p_flip = np.where(phys == 0, [f[0] for f in flips], [f[1] for f in flips])
    read = phys ^ (u < p_flip)
    y = read ^ r
    idx = (y << np.arange(n)[None, :]).sum(1)
    return np.bincount(idx, minlength=2 ** n).astype(float)


def mitigation_values(o, theta, mode: str, n: int) -> np.ndarray:
    """Per-outcome values (A⁻¹)ᵀ o of the readout correction with calibration theta: tensored — the inverse of each
    bit's assignment matrix [[1−e, h], [e, 1−h]]; TREX — of the symmetric [[1−p̄, p̄], [p̄, 1−p̄]]; none — o."""
    o = np.asarray(o, float)
    if mode == "none" or theta is None:
        return o.copy()
    t = np.asarray(theta, float)
    v = o.reshape((2,) * n)            # axis k ↔ bit n−1−k (little-endian index)
    for i in range(n):
        if mode == "tensored":
            e, h = t[i]
            A = np.array([[1 - e, h], [e, 1 - h]])
        else:
            A = np.array([[1 - t[i], t[i]], [t[i], 1 - t[i]]])
        M = np.linalg.inv(A).T
        ax = n - 1 - i
        v = np.moveaxis(np.tensordot(M, np.moveaxis(v, ax, 0), axes=(1, 0)), 0, ax)
    return v.reshape(-1)


# ------------------------------------------------------------------------------------------------ estimators
def exp_fit(lams, E) -> tuple:
    """(estimate, failed): least squares of log|E| on λ with every E of one sign (E = A e^{−bλ}); else failed."""
    E = np.asarray(E, float)
    s = np.sign(E[0]) if E[0] != 0 else 1.0
    if not np.all(s * E > 0):
        return math.nan, True
    b, a = np.polyfit(np.asarray(lams, float), np.log(s * E), 1)
    return float(s * math.exp(a)), False


def line_at_zero(lams, E) -> float:
    return float(np.polyval(np.polyfit(np.asarray(lams, float), np.asarray(E, float), 1), 0.0))


def l1_bounded_ls(X, y, bound: float, eps: float, iters: int = 4000) -> np.ndarray:
    """argmin ‖Xc − y‖² + eps‖c‖² subject to ‖c‖₁ ≤ bound, by accelerated projected gradient (FISTA) with the exact
    Euclidean projection onto the ℓ1 ball (Duchi et al., ICML 2008) — written for this reference, not NIL's
    active-set solver."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    k = X.shape[1]
    L = 2.0 * (np.linalg.norm(X, 2) ** 2 + eps) or 1.0

    def proj(v):
        if np.abs(v).sum() <= bound:
            return v
        u = np.sort(np.abs(v))[::-1]
        css = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, k + 1) > (css - bound))[0][-1]
        th = (css[rho] - bound) / (rho + 1.0)
        return np.sign(v) * np.maximum(np.abs(v) - th, 0.0)
    c = np.zeros(k)
    z, t = c.copy(), 1.0
    for _ in range(iters):
        g = 2 * X.T @ (X @ z - y) + 2 * eps * z
        c_new = proj(z - g / L)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        z = c_new + (t - 1) / t_new * (c_new - c)
        if np.max(np.abs(c_new - c)) < 1e-14:
            c = c_new
            break
        c, t = c_new, t_new
    return c


# ------------------------------------------------------------------------------------------------ trials
def trials(prep, key: str, B: int, R: int, rng, *, res=None, ctx=None, alloc=None) -> np.ndarray:
    """R reference trials of candidate ``key`` at budget B with the allocation design's result (or the given
    ``res``/``ctx`` of one design). ``alloc``: an explicit allocation (e.g. fewer target shots than the allocation
    rule's minimum, to exercise zero-acceptance). NaN: no estimate (a nonresponse)."""
    pr = PR.PORTFOLIO[key]
    res = prep.truth[key] if res is None else res
    ctx = prep.ctx_truth[key] if ctx is None else ctx
    alloc = PR.allocate(pr, prep.supplied[key], int(B)) if alloc is None else alloc
    if not alloc.feasible:
        raise ValueError("infeasible at B = %d: %s" % (B, alloc.reason))
    n, o, mode = ctx.n, np.asarray(ctx.obs.o, float), pr.readout
    out = np.full(int(R), np.nan)
    for r in range(int(R)):
        th = calibration_draw(ctx, alloc.s, rng, mode)
        ov = mitigation_values(o, th, mode, n)
        if pr.estimator == "raw":
            if mode == "trex":
                c = _trex_readout_draws(ctx, ctx.run().probs, alloc.N, rng)
            else:
                c = counts(ctx.run().probs_meas, alloc.N, rng)
            out[r] = c @ ov / alloc.N
        elif pr.estimator == "zne":
            d = res.sampler.direct
            E_flat = np.array([counts(rr.probs_meas, nj, rng) @ ov / nj for rr, nj in zip(d["runs"], alloc.split)])
            E = E_flat @ d["P"]
            rule = dict(pr.params)["rule"]
            if rule == "exponential":
                v, failed = exp_fit(d["lams"], E)
                out[r] = line_at_zero(d["lams"], E) if failed else v
            elif rule == "linear":
                out[r] = line_at_zero(d["lams"], E)
            else:
                out[r] = float(np.polyval(np.polyfit(d["lams"], E, len(E) - 1), 0.0))
            if pr.post == "clip-observable-range":
                out[r] = float(np.clip(out[r], o.min(), o.max()))
        elif pr.estimator == "symmetry":
            out[r] = _sv_value(ctx, res, ctx.run().probs_meas, alloc.N, ov, th, rng)
        elif pr.estimator == "chain":
            E = [_sv_value(ctx, rj, rj_run.probs_meas, nj, ov, th, rng)
                 for rj, rj_run, nj in _chain_folds(res, alloc)]
            lams = res.details["lams"]
            out[r] = line_at_zero(lams, E) if np.all(np.isfinite(E)) else math.nan
        elif pr.estimator == "learning":
            out[r] = _cdr_value(ctx, res, pr, alloc, th, rng)
        elif pr.estimator == "nil":
            out[r] = _nil_value(ctx, res, pr, alloc, th, rng)
        else:
            raise NotImplementedError(pr.estimator)
    return out


def _mask(ctx, res) -> np.ndarray:
    """The post-selection mask over classical bitstrings from the declared checks (recomputed here)."""
    from ..methods.symmetry import parse_checks
    d = res.details
    n = ctx.n
    bits = _bits(n)
    relabel = bool((ctx.compiled.info or {}).get("relabel"))
    bit = (lambda q: n - 1 - q) if relabel else (lambda q: q)
    ok = (bits[:, [bit(q) for q in d["qubits"]]].sum(1) % 2) == (0 if int(d["eigenvalue"]) == 1 else 1)
    for g in d.get("more") or parse_checks(""):
        ok &= (bits[:, [bit(q) for q in g]].sum(1) % 2) == 0
    return ok


def _sv_value(ctx, res, p_meas, N, ov_o, th, rng) -> float:
    """Ratio of mitigated, post-selected sums: Σ f·Aᵀ⁻¹(mask·o) / Σ f·Aᵀ⁻¹(mask) over N shots."""
    mode = ctx.recipe.readout
    m = _mask(ctx, res).astype(float)
    num_v = mitigation_values(m * np.asarray(ctx.obs.o, float), th, mode, ctx.n)
    den_v = mitigation_values(m, th, mode, ctx.n)
    c = counts(p_meas, N, rng)
    den = c @ den_v
    return float(c @ num_v / den) if den > 0 else math.nan


def _chain_folds(res, alloc):
    """(inner result, its executed circuit's law, shots) per fold of a ZNE∘SV chain."""
    inner = getattr(res.sampler, "inner", None)
    if inner is None:
        raise NotImplementedError("the chained result does not expose its inner fold results")
    return [(rj, rj.sampler.direct["run"], nj) for rj, nj in zip(inner, alloc.split)]


def _cdr_value(ctx, res, pr, alloc, th, rng) -> float:
    """CDR / FR: training features from s_tr shots per training circuit, the declared rank rule (singular values of
    the centred training features above σ(√T + √k), σ² the largest per-shot feature variance estimated from the
    sampled training data), OLS with intercept; fallback: the mitigated target value."""
    from ..methods.learning import feature_matrix
    d = res.details or {}
    if "names" not in d or (d.get("degenerate") and "X" not in d):    # Clifford, or degenerate training labels:
        c = counts(ctx.run().probs_meas, alloc.N, rng)                  # the declared fallback for every trial
        return float(c @ mitigation_values(ctx.obs.o, th, pr.readout, ctx.n) / alloc.N)
    specs, y = _cdr_design(ctx, pr)
    F, _ = feature_matrix(ctx.n, np.asarray(ctx.obs.o, float), dict(pr.params).get("features", "cdr"))
    Fv = np.stack([mitigation_values(F[:, j], th, pr.readout, ctx.n) for j in range(F.shape[1])], 1)
    T, k, s = len(specs), F.shape[1], int(alloc.s_tr)
    X = np.empty((T, k))
    vmax = 0.0
    for i, sp in enumerate(specs):
        c = counts(ctx.run(sp, role="training").probs_meas, s, rng)
        f = c / s
        X[i] = f @ Fv
        vmax = max(vmax, float(np.max(f @ Fv ** 2 - (f @ Fv) ** 2)))
    tol = max(1e-9, math.sqrt(max(vmax, 0.0) / s) * (math.sqrt(T) + math.sqrt(k)))
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    ct = counts(ctx.run().probs_meas, alloc.N, rng)
    xt = ct @ Fv / alloc.N
    if T < 2 or not np.any(sv > tol):
        return float(xt[0])                                   # fallback: the mitigated target value (column ⟨O⟩)
    A = np.hstack([np.ones((T, 1)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(beta[0] + xt @ beta[1:])


def _cdr_design(ctx, pr):
    from ..methods.learning import choose_training_set
    from ..transforms import non_clifford_slots
    from .preflight import _IdealContext
    p = dict(pr.params)
    slots = non_clifford_slots(ctx.spec)
    keep_eff = min(int(p.get("keep", 0)), max(len(slots) - 1, 0))
    need = float(p.get("min_spread", 0.05)) * (float(np.ptp(ctx.obs.o)) or 1.0)
    specs, y, _, _ = choose_training_set(ctx, slots, int(p.get("n_train", 16)), keep_eff,
                                         p.get("strategy", "adaptive"), need)
    ideal = _IdealContext(ctx.compiled, ctx.obs.o, ctx.seed)
    return specs, np.array([ideal.ideal_value(sp) for sp in specs])          # labels recomputed independently


def _nil_value(ctx, res, pr, alloc, th, rng) -> float:
    """NIL: features = mitigated values of every neighbor circuit of each training circuit (s_tr shots each), labels
    recomputed by state vector, ℓ1-bounded regression without intercept (own solver, the same ε stabilization),
    equal target split over the target's neighbor circuits."""
    from ..methods import REGISTRY
    from ..methods.nil import _nil_seed, nil_training_circuit
    from .preflight import _IdealContext
    p = dict(pr.params)
    feats = REGISTRY["nil"].neighbors(ctx, p.get("neighbors", "pauli"), p.get("positions", "2q"),
                                      int(p.get("n_neighbors", 12)), bool(p.get("include_bare", True)),
                                      tuple(p.get("scales", (1.0, 3.0, 5.0))), p.get("folding", "global"))
    k = len(feats)
    rng_tr = _nil_seed(ctx.seed, 2)
    T = int(p.get("n_train", 16))
    tr = [nil_training_circuit(ctx.spec, rng_tr, p.get("ensemble", "design4")) for _ in range(T)]
    ideal = _IdealContext(ctx.compiled, ctx.obs.o, ctx.seed)
    y = np.array([ideal.ideal_value(sp) for sp in tr])
    if np.ptp(y) < 1e-9:
        c = counts(ctx.run().probs_meas, alloc.N, rng)
        return float(c @ mitigation_values(ctx.obs.o, th, pr.readout, ctx.n) / alloc.N)
    ov = mitigation_values(ctx.obs.o, th, pr.readout, ctx.n)
    s = int(alloc.s_tr)
    X = np.array([[counts(ctx.run(f[1](sp), role="training").probs_meas, s, rng) @ ov / s for f in feats]
                  for sp in tr])
    eps = (1e-6 * np.linalg.norm(X, 2)) ** 2
    coef = l1_bounded_ls(X, y, float(p.get("l1_bound", 2.0)), eps)
    nj = np.full(k, alloc.N // k)
    nj[: alloc.N - int(nj.sum())] += 1
    xt = np.array([counts(ctx.run(f[1](ctx.spec)).probs_meas, int(m), rng) @ ov / m for f, m in zip(feats, nj)])
    return float(np.where(coef != 0, coef * xt, 0.0).sum())


def compare(prep, key: str, B: int, R: int, rng, *, design=None, alloc=None) -> dict:
    """Emulated (``emulate._run`` on the same method result and allocation) versus reference trials."""
    from . import emulate as EM
    pr = PR.PORTFOLIO[key]
    res = prep.truth[key] if design is None else design[0]
    ctx = prep.ctx_truth[key] if design is None else design[1]
    alloc = PR.allocate(pr, prep.supplied[key], int(B)) if alloc is None else alloc
    em, _ = EM._run(pr, res, ctx, alloc, int(R), rng)
    ref = trials(prep, key, B, R, rng, res=res, ctx=ctx, alloc=alloc)
    c = EM.compare_laws(em, ref)
    c.update(protocol=key, B=int(B), nonresponse_emulated=float(np.mean(~np.isfinite(em))),
             nonresponse_reference=float(np.mean(~np.isfinite(ref))))
    return c
