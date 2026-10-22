r"""The statistical result contract: what a reported number *means*.

A mitigation result can refer to four different quantities, and QEM Explorer keeps them apart:

==============  ==================================================================================================
``exact``       an exact (infinite-shot) property of the simulated protocol, computed from exact probabilities.
                For ZNE, PEC, readout mitigation and the other estimators that are linear in their data this *is*
                the expected value of the finite-shot protocol.
``reference``   the exact-probability **reference fit** of a learning method (CDR, FR, NIL): the regression fitted
                to *exact* training features. It is an oracle quantity — a real experiment never sees exact
                training features — and it is generally **not** the protocol's expected value, because the fit is a
                nonlinear function of the sampled training data:
                :math:`\mathbb E_D[\hat f_D(x)]\neq f_{\mathbb E[D]}(x)`.
``protocol``    the expected moments of the specified finite-shot protocol (finite training shots, exact target
                features), estimated by Monte Carlo over independent training-data draws :math:`D`:

                .. math:: \mu=\mathbb E_D[m(D)],\qquad
                          \operatorname{Var}(\hat O)=\operatorname{Var}_D[m(D)]+\mathbb E_D[v(D)]/N_t,

                with :math:`m(D)` the estimate of the fit on draw :math:`D` applied to the exact target and
                :math:`v(D)` its per-shot target variance. The decomposition is the law of total variance; it is
                exact when, given the fit, the estimate is linear in the target's sample frequencies (readout
                mitigation none / tensored / TREX) and holds to first order in :math:`1/N_t` for M3.
``measured``    one experiment's estimate (hardware replay, or one simulated run): a single draw, not a mean.
==============  ==================================================================================================

Bias and RMSE in tables, budget curves and exports are always computed from the ``protocol`` (or ``exact``) mean of
a *simulated* result. A ``measured`` hardware result has no bias: against a trusted ideal value it has one *signed
error to reference* e = Ô − O₀, and its absolute value |e| is one observation of the squared-error scale (e² is an
unbiased observation of the MSE under the repeated-experiment law; |e| is not an unbiased RMSE estimate).

**Estimators that are nonlinear in their target data** (1.5.1). Exponential ZNE fits log|E(λ)| to the sampled
expectations; symmetry verification divides two sampled sums. Their finite-shot mean depends on the number of target
shots :math:`N_t` (a bias of order :math:`1/N_t`) and their fit can fail on sampled data (a sign change of E(λ), no
accepted shot). For them the result is a ``reference``: ``mean`` = ``reference_mean`` is the estimator applied to
exact probabilities (under the declared failure policy), and the protocol's moments are **budget dependent**:

.. math:: \mu(N_t)=\mathbb E[\hat O_{N_t}],\qquad \sigma^2(N_t)=\operatorname{Var}[\hat O_{N_t}],

estimated by Monte Carlo over complete target experiments (:func:`target_moments`, adaptive to the same tolerance on
the mean's Monte-Carlo standard error as the learning moments) at the evaluated budget, and on a grid of target-shot
counts (:func:`moment_grid`) for budget curves. The failure policy is part of the protocol and is declared before
evaluation: *fallback* (a shot-free fallback estimator on the same data, so every experiment has an estimate and the
moments are unconditional) or *abstain* (no estimate; the moments are conditional on success and the abstention
probability is reported next to them). Beyond the target-shot count from which the Monte-Carlo moments agree with
the first-order (delta-method) ones within the declared criteria (:data:`FIRST_ORDER_CRITERIA`), curves use the
first-order formula and say so; where that agreement was not established, the curve is labeled
"first-order (not validated)".

Readout calibration enters the variance to first order (:math:`w/s`, measured by Monte Carlo for learning methods);
its :math:`O(1/s)` effect on the mean of a nonlinear refit is not included in :math:`\mu` and the result carries the
flag ``calibration-mean-first-order`` when a calibration is present.

Failed refits are never dropped silently: :class:`Moments` records how many of the requested refits produced a
finite estimate, and moments computed from the survivors are marked ``conditional`` (conditional on the protocol
succeeding). A protocol that fails in more than half of the refits raises :class:`ProtocolFailure`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

KINDS = ("exact", "reference", "protocol", "measured")

# Monte-Carlo refits behind protocol moments: start, cap, and the declared tolerance on the mean's Monte-Carlo
# standard error (absolute, in units of the observable). Repeats double until mean_mcse <= tol or the cap is reached;
# the number used is always recorded.
REPEATS_START = 500
REPEATS_MAX = 4000
MEAN_MCSE_TOL = 2.5e-3
MAX_FAILURE_RATE = 0.5


class ProtocolFailure(FloatingPointError):
    """The finite-shot protocol fails (non-finite estimate) in too many training-data draws to report moments."""


@dataclass(frozen=True)
class Moments:
    """Monte-Carlo moments of a finite-training protocol at one number of training shots."""
    mean: float               # E_D[m(D)]
    var1: float               # E_D[v(D)]: per-shot target variance, averaged over training draws
    u: float                  # Var_D[m(D)]: training-data variance (does not shrink with target shots)
    train_shots: int          # shots per training circuit at which the draws were made
    repeats: int              # refits requested
    finite: int               # refits that produced a finite estimate
    mean_mcse: float          # Monte-Carlo standard error of ``mean``: sqrt(u / finite)
    var1_mcse: float          # Monte-Carlo standard error of ``var1``
    u_se: float               # standard error of ``u`` (fourth-moment formula, no Gaussian assumption)
    tol: float                # declared tolerance on mean_mcse that stopped the adaptive repeats (nan: none)
    kind: str = "protocol"

    @property
    def failed(self) -> int:
        return int(self.repeats - self.finite)

    @property
    def failure_rate(self) -> float:
        return self.failed / self.repeats if self.repeats else math.nan

    @property
    def conditional(self) -> bool:
        """True when failed refits were excluded: the moments are conditional on the protocol succeeding."""
        return self.failed > 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(failed=self.failed, failure_rate=self.failure_rate, conditional=self.conditional)
        return d


def _se_var(x: np.ndarray, var: float) -> float:
    n = len(x)
    if n < 4:
        return math.nan
    m4 = float(np.mean((x - x.mean()) ** 4))
    return float(math.sqrt(max(m4 - var * var * (n - 3) / (n - 1), 0.0) / n))


def moments_from_draws(est, var1, train_shots: int, repeats: int | None = None, tol: float = math.nan) -> Moments:
    """Moments from R refits: ``est`` the estimates m(D_r), ``var1`` their per-shot target variances v(D_r).
    Non-finite rows count as failed refits (both arrays must be finite for a row to count)."""
    est = np.asarray(est, float).ravel()
    v1 = np.asarray(var1, float).ravel()
    if est.shape != v1.shape:
        raise ValueError("estimates and variances have different shapes %s, %s" % (est.shape, v1.shape))
    R = int(len(est) if repeats is None else repeats)
    ok = np.isfinite(est) & np.isfinite(v1)
    if np.any(v1[ok] < -1e-12 * max(1.0, float(np.max(np.abs(v1[ok]))) if ok.any() else 1.0)):
        raise FloatingPointError("a refit produced a negative per-shot variance")
    n = int(ok.sum())
    if n < 2:
        raise ProtocolFailure("only %d of %d training-data draws gave a finite estimate" % (n, R))
    if (R - n) / max(R, 1) > MAX_FAILURE_RATE:
        raise ProtocolFailure("the protocol fails in %d of %d training-data draws (> %.0f %%): its moments are not "
                              "reported" % (R - n, R, 100 * MAX_FAILURE_RATE))
    e, v = est[ok], np.maximum(v1[ok], 0.0)
    u = float(np.var(e, ddof=1))
    return Moments(mean=float(np.mean(e)), var1=float(np.mean(v)), u=u, train_shots=int(train_shots), repeats=R,
                   finite=n, mean_mcse=float(math.sqrt(u / n)), var1_mcse=float(np.std(v, ddof=1) / math.sqrt(n)),
                   u_se=_se_var(e, u), tol=float(tol))


def training_moments(train_resampler, train_shots: int, rng, *, repeats: int = REPEATS_START,
                     max_repeats: int = REPEATS_MAX, tol: float = MEAN_MCSE_TOL) -> Moments:
    """Protocol moments of a learning method at ``train_shots`` shots per training circuit.

    ``train_resampler(s_tr, R, rng) -> (estimates, per-shot variances)`` refits the method on R independent draws of
    the training data and applies each fit to the exact target features. Repeats start at ``repeats`` and double (new
    draws from the same generator, so the result is deterministic for a seeded ``rng``) until the Monte-Carlo
    standard error of the mean is at most ``tol`` or ``max_repeats`` is reached; ``tol`` = nan disables adaptation."""
    if int(repeats) < 2:
        raise ValueError("at least two refits are required")
    if int(max_repeats) < int(repeats):
        raise ValueError("max_repeats must be at least repeats")
    ests, vs = [], []
    total, batch = 0, int(repeats)
    while True:
        e, v = train_resampler(int(train_shots), int(batch), rng)
        e, v = np.asarray(e, float).ravel(), np.asarray(v, float).ravel()
        if e.shape != (batch,) or v.shape != (batch,):
            raise ValueError("unexpected training-refit result shape %s, %s (expected %d)" % (e.shape, v.shape, batch))
        ests.append(e)
        vs.append(v)
        total += batch
        m = moments_from_draws(np.concatenate(ests), np.concatenate(vs), train_shots, total, tol)
        if not (math.isfinite(tol) and m.mean_mcse > tol and total < int(max_repeats)):
            return m
        batch = min(total, int(max_repeats) - total)


def moments_wanted(ctx) -> bool:
    """Whether a learning method should estimate its protocol moments by Monte Carlo (the default). The protocol-
    selection emulator turns this off (``Context.protocol_moments = False``) for the extra training designs it draws:
    their trials use only the method's sampler, never the moments, which cost most of a design's preparation time
    (1.7.0). A result made without them carries the flag ``moments-not-estimated``."""
    return bool(getattr(ctx, "protocol_moments", True))


def reference_moments(mean: float, var1: float, train_shots: int) -> Moments:
    """Placeholder moments when Monte-Carlo moments were not requested: the exact-feature fit's mean and per-shot
    variance, no training-data variance estimate (``u`` = nan) and no refits."""
    nan = math.nan
    return Moments(mean=float(mean), var1=float(var1), u=nan, train_shots=int(train_shots), repeats=0, finite=0,
                   mean_mcse=nan, var1_mcse=nan, u_se=nan, tol=nan, kind="reference (moments not estimated)")


def is_measured(ctx) -> bool:
    """True when the context replays measured hardware data (not a simulation and not a hardware *recording*, which
    answers with predictions to plan the allocation)."""
    r = getattr(ctx, "runner", None)
    return getattr(r, "backend", None) == "hardware" and not getattr(r, "recording", True)


def protocol_contract(ctx, mom: Moments, ref_mean: float, ref_var1: float, notes: list, degenerate: bool = False) -> dict:
    """The MethodResult fields of a learning method under the statistical contract.

    Simulation: ``mean``/``var1`` become the protocol moments; the exact-probability fit is kept as the reference.
    Hardware replay: ``mean`` stays the fit to the *measured* training data applied to the measured target (one
    experiment); the resampled refits are then a bootstrap of the measured data and only supply the uncertainty."""
    if is_measured(ctx):
        rec = dict(mom.as_dict(), kind="bootstrap")
        notes.append("Measured estimate: the regression fitted to the measured training data and applied to the measured "
                     "target. The training-data uncertainty u comes from %d bootstrap refits of the measured counts."
                     % mom.finite)
        return dict(mean=float(ref_mean), var1=float(ref_var1), estimate_kind="measured", reference_mean=None,
                    reference_var1=None, moments=rec, flags=("measured",))
    flags = []
    if mom.conditional:
        flags.append("conditional-on-success")
        notes.append("%d of %d training-data draws gave no finite estimate (failure rate %.1f %%); the moments are "
                     "conditional on success." % (mom.failed, mom.repeats, 100 * mom.failure_rate))
    if not degenerate and math.isfinite(ref_mean):
        d = mom.mean - float(ref_mean)
        if abs(d) > 3 * max(mom.mean_mcse, 1e-15):
            notes.append("With %d shots per training circuit the protocol's expected value is %.4g (± %.2g Monte-Carlo), "
                         "not the %.4g of the fit to exact training features: the regression is nonlinear in its "
                         "noisy training data. Bias and RMSE use the protocol's value; the exact fit is kept as a "
                         "reference." % (mom.train_shots, mom.mean, mom.mean_mcse, ref_mean))
    return dict(mean=float(mom.mean), var1=float(mom.var1), estimate_kind="protocol", reference_mean=float(ref_mean),
                reference_var1=float(ref_var1), moments=mom.as_dict(), flags=tuple(flags))


# ------------------------------------------------------------------ budget-dependent (target-nonlinear) protocols
TARGET_REPEATS_START = 2000      # complete target experiments per Monte-Carlo batch (doubles adaptively)
TARGET_REPEATS_MAX = 64000       # cap on experiments at one target-shot count (always recorded)
TARGET_GRID_MAX_SHOTS = 10 ** 7  # the moment grid stops here (or where first order is validated, if earlier)
TARGET_GRID_PER_DECADE = 4       # grid points per decade of target shots
TARGET_BLOCK_BYTES = 32 * 2 ** 20    # memory bound of one sampling block (repeats × categories × 8 bytes)
FIRST_ORDER_CRITERIA = dict(
    sd_rel=0.05,                 # |σ(N)·√N / √var1 − 1| ≤ 0.05 + 3·(its Monte-Carlo SE)
    bias_frac=0.10,              # |μ(N) − reference| ≤ 0.10·σ(N) + 3·mean_mcse
    fit_failure_max=1e-3,        # the primary fit failed in at most 0.1 % of experiments
    consecutive=2,               # at two consecutive grid points
)
POLICIES = ("fallback", "abstain")


@dataclass(frozen=True)
class TargetMoments:
    """Monte-Carlo moments of a complete target experiment of a target-nonlinear estimator at ``target_shots``."""
    target_shots: int
    mean: float               # expected estimate (conditional on an estimate existing when the policy abstains)
    var: float                # variance of one estimate (same conditioning)
    repeats: int              # complete experiments simulated
    finite: int               # experiments that produced an estimate
    fit_failures: int         # experiments whose primary (nonlinear) fit failed: fallback used, or abstained
    mean_mcse: float          # Monte-Carlo SE of ``mean``
    var_se: float             # Monte-Carlo SE of ``var`` (fourth-moment formula)
    policy: str               # "fallback" | "abstain"
    tol: float                # tolerance on mean_mcse that stopped the adaptive repeats
    kind: str = "target-protocol"

    @property
    def fit_failure_rate(self) -> float:
        return self.fit_failures / self.repeats if self.repeats else math.nan

    @property
    def abstention_rate(self) -> float:
        return (self.repeats - self.finite) / self.repeats if self.repeats else math.nan

    @property
    def conditional(self) -> bool:
        return self.finite < self.repeats

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.var, 0.0)) if math.isfinite(self.var) else math.nan

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(fit_failure_rate=self.fit_failure_rate, abstention_rate=self.abstention_rate,
                 conditional=self.conditional, sd=self.sd)
        return d


def target_moments(draw, target_shots: int, rng, policy: str, *, repeats: int = TARGET_REPEATS_START,
                   max_repeats: int = TARGET_REPEATS_MAX, tol: float = MEAN_MCSE_TOL) -> TargetMoments:
    """Moments of ``draw(N, R, rng) -> (estimates, primary_fit_failed)`` at N = ``target_shots``: R complete target
    experiments per batch, doubling (fresh draws from the same generator) until the mean's Monte-Carlo SE is at most
    ``tol`` or ``max_repeats`` experiments were simulated. Non-finite estimates are abstentions; they are counted, never
    averaged. Fewer than two estimates: mean and var are NaN (recorded, not raised)."""
    if policy not in POLICIES:
        raise ValueError("failure policy must be one of %s" % ", ".join(POLICIES))
    N = int(target_shots)
    ests, fails = [], []
    total, batch = 0, int(repeats)
    while True:
        e, f = draw(N, int(batch), rng)
        e, f = np.asarray(e, float).ravel(), np.asarray(f, bool).ravel()
        if e.shape != (batch,) or f.shape != (batch,):
            raise ValueError("unexpected target-sampler shape %s, %s (expected %d)" % (e.shape, f.shape, batch))
        ests.append(e)
        fails.append(f)
        total += batch
        x = np.concatenate(ests)
        ok = x[np.isfinite(x)]
        n = len(ok)
        if n >= 2:
            var = float(np.var(ok, ddof=1))
            mcse = float(math.sqrt(var / n))
        else:
            var, mcse = math.nan, math.inf
        if not (math.isfinite(tol) and mcse > tol and total < int(max_repeats)):
            break
        batch = min(total, int(max_repeats) - total)
    return TargetMoments(target_shots=N, mean=float(np.mean(ok)) if n >= 2 else math.nan, var=var, repeats=total,
                         finite=n, fit_failures=int(np.concatenate(fails).sum()), mean_mcse=mcse,
                         var_se=_se_var(ok, var) if n >= 4 else math.nan, policy=policy, tol=float(tol))


def _first_order_ok(m: TargetMoments, ref_mean: float, var1: float, n_eff: float | None = None) -> bool:
    c = FIRST_ORDER_CRITERIA
    if not (math.isfinite(m.mean) and math.isfinite(m.var) and math.isfinite(ref_mean) and math.isfinite(var1)):
        return False
    if m.fit_failure_rate > c["fit_failure_max"] or m.conditional:
        return False
    sd1 = math.sqrt(max(var1, 0.0) / (n_eff or m.target_shots))
    if sd1 <= 0:
        return m.var <= 1e-24 and abs(m.mean - ref_mean) <= 3 * max(m.mean_mcse, 1e-12)
    sd_rel = m.sd / sd1 - 1.0
    sd_rel_se = (m.var_se / (2 * max(m.var, 1e-300))) if math.isfinite(m.var_se) else math.inf
    return (abs(sd_rel) <= c["sd_rel"] + 3 * sd_rel_se
            and abs(m.mean - ref_mean) <= c["bias_frac"] * m.sd + 3 * m.mean_mcse)


LATTICE_WINDOW = 4               # adjacent target-shot counts averaged at every grid point (see moment_grid)


def moment_grid(draw, n_min: int, ref_mean: float, var1: float, rng, policy: str, *,
                n_max: int = TARGET_GRID_MAX_SHOTS, tol: float = MEAN_MCSE_TOL, window: int = LATTICE_WINDOW) -> dict:
    """Budget-dependent moments of a target-nonlinear estimator on a geometric grid of target-shot counts from
    ``n_min`` (the fewest shots that can give an estimate) up to ``n_max``, stopping once the first-order moments
    (``ref_mean``, ``var1/N``) are validated by :data:`FIRST_ORDER_CRITERIA` at consecutive grid points. The record
    is plain data (lists), so it can be stored in rows, records and exports.

    Lattice effects. At small budgets the moments of such an estimator are not a smooth function of N: counts are
    integers, so whether a sampled E(λ) can be exactly 0 (an even number of shots on a circuit), or a ratio's
    denominator can vanish, changes from one N to the next (exponential ZNE at a few hundred shots: the SD moves by
    ~20 % between N and N + 2). Every grid point therefore averages the ``window`` adjacent counts N … N+window−1 and
    records the spread of their SDs (``sd_range``); curves between grid points are this local average, while a table
    at one budget uses the moments at exactly that budget."""
    n_min = max(int(n_min), 1)
    n_max = max(int(n_max), n_min)
    w = max(int(window), 1)
    k = max(int(math.ceil(TARGET_GRID_PER_DECADE * math.log10(n_max / n_min))) + 1, 2)
    grid = sorted({int(round(x)) for x in np.geomspace(n_min, n_max, k)} | {n_min})
    pts, agree, from_n = [], 0, None
    for N in grid:
        sub = [target_moments(draw, N + i, rng, policy, tol=tol * math.sqrt(w)) for i in range(w)] if w > 1 else None
        if sub is None:
            m = target_moments(draw, N, rng, policy, tol=tol)
            rng_sd = 0.0
        else:
            m = TargetMoments(target_shots=int(N), mean=float(np.mean([s.mean for s in sub])),
                              var=float(np.mean([s.var for s in sub])), repeats=int(sum(s.repeats for s in sub)),
                              finite=int(sum(s.finite for s in sub)), fit_failures=int(sum(s.fit_failures for s in sub)),
                              mean_mcse=float(math.sqrt(sum(s.mean_mcse ** 2 for s in sub)) / w),
                              var_se=float(math.sqrt(sum(s.var_se ** 2 for s in sub)) / w), policy=policy,
                              tol=float(tol))
            sds = [s.sd for s in sub if math.isfinite(s.sd)]
            rng_sd = float(max(sds) - min(sds)) if len(sds) > 1 else math.nan
        n_eff = 1.0 / float(np.mean([1.0 / (N + i) for i in range(w)]))     # var1/n_eff = mean of var1/(N+i)
        ok = _first_order_ok(m, ref_mean, var1, n_eff)
        pts.append(dict(m.as_dict(), first_order_ok=bool(ok), sd_range=rng_sd, n_eff=n_eff))
        agree = agree + 1 if ok else 0
        if agree >= FIRST_ORDER_CRITERIA["consecutive"]:
            from_n = int(pts[-FIRST_ORDER_CRITERIA["consecutive"]]["target_shots"])
            break
    return dict(kind="target-moment-grid", policy=policy, reference_mean=float(ref_mean), var1=float(var1),
                n_min=n_min, n_max=n_max, first_order_from=from_n, criteria=dict(FIRST_ORDER_CRITERIA),
                tol=float(tol), window=w, N=[p["n_eff"] for p in pts], N_start=[p["target_shots"] for p in pts],
                mean=[p["mean"] for p in pts],
                var=[p["var"] for p in pts], mean_mcse=[p["mean_mcse"] for p in pts], var_se=[p["var_se"] for p in pts],
                sd_range=[p["sd_range"] for p in pts], fit_failure_rate=[p["fit_failure_rate"] for p in pts],
                abstention_rate=[p["abstention_rate"] for p in pts], repeats=[p["repeats"] for p in pts],
                first_order_ok=[p["first_order_ok"] for p in pts])


def grid_at(grid: dict, N: float) -> dict:
    """Moments of the protocol at N target shots from a :func:`moment_grid` record: Monte Carlo at a grid point,
    log-linear interpolation of μ and N·σ² between grid points, the first-order formula where the grid validated it
    (``source`` says which), and NaN below the smallest usable N."""
    nan = math.nan
    Ns = np.asarray(grid.get("N") or [], float)
    if not len(Ns) or not (N >= float(grid.get("n_min", Ns[0]))):
        return dict(mean=nan, var=nan, fit_failure_rate=nan, abstention_rate=nan, source="infeasible")
    if N < Ns[0]:                                    # inside the first averaging window: its moments, scaled in N·σ²
        return dict(mean=float(grid["mean"][0]), var=float(grid["var"][0]) * Ns[0] / N,
                    fit_failure_rate=float(grid["fit_failure_rate"][0]),
                    abstention_rate=float(grid["abstention_rate"][0]),
                    sd_range=float((grid.get("sd_range") or [nan])[0]), source="monte-carlo (first window)")
    fo = grid.get("first_order_from")
    ref, v1 = float(grid["reference_mean"]), float(grid["var1"])
    if fo is not None and N >= fo:
        return dict(mean=ref, var=v1 / N, fit_failure_rate=0.0, abstention_rate=0.0, source="first-order (validated)")
    if N > Ns[-1]:
        return dict(mean=ref, var=v1 / N, fit_failure_rate=nan, abstention_rate=nan,
                    source="first-order (not validated)")
    mu, var = np.asarray(grid["mean"], float), np.asarray(grid["var"], float)
    ff, ab = np.asarray(grid["fit_failure_rate"], float), np.asarray(grid["abstention_rate"], float)
    sr = np.asarray(grid.get("sd_range") or [nan] * len(Ns), float)
    i = int(np.searchsorted(Ns, N, side="right") - 1)
    if Ns[i] == N or i == len(Ns) - 1:
        return dict(mean=float(mu[i]), var=float(var[i]), fit_failure_rate=float(ff[i]),
                    abstention_rate=float(ab[i]), sd_range=float(sr[i]), source="monte-carlo")
    t = (math.log(N) - math.log(Ns[i])) / (math.log(Ns[i + 1]) - math.log(Ns[i]))
    nv = (1 - t) * var[i] * Ns[i] + t * var[i + 1] * Ns[i + 1]
    return dict(mean=float((1 - t) * mu[i] + t * mu[i + 1]), var=float(nv / N),
                fit_failure_rate=float((1 - t) * ff[i] + t * ff[i + 1]),
                abstention_rate=float((1 - t) * ab[i] + t * ab[i + 1]),
                sd_range=float(np.nanmax([sr[i], sr[i + 1]])) if np.isfinite([sr[i], sr[i + 1]]).any() else nan,
                source="interpolated")


def compressed_values(values, probs):
    """(distinct per-shot values, their total probabilities): the sum of N per-shot values depends only on how many
    shots took each distinct value, so a multinomial over these few categories has exactly the distribution of the
    N-shot sum (and needs far less memory than one over all 2^n bitstrings)."""
    v = np.asarray(values, float)
    p = np.clip(np.asarray(probs, float), 0, None)
    p = p / p.sum()
    key = np.round(v, 12)
    u, inv = np.unique(key, return_inverse=True)
    vals = np.array([v[inv == j][0] for j in range(len(u))])
    return vals, np.bincount(inv, weights=p, minlength=len(u))


def sample_means(vals, probs, N: int, R: int, rng) -> np.ndarray:
    """R sample means of N i.i.d. shots with per-shot values ``vals`` taken with probabilities ``probs``, in blocks
    of bounded memory (:data:`TARGET_BLOCK_BYTES`)."""
    N, R = max(int(N), 1), int(R)
    block = max(16, int(TARGET_BLOCK_BYTES // (8 * max(len(vals), 1))))
    out = np.empty(R)
    for a in range(0, R, block):
        b = min(R, a + block)
        out[a:b] = rng.multinomial(N, probs, size=b - a) @ vals / N
    return out


def rmse_with_mc(bias: float, sd: float, bias_mcse: float = 0.0) -> tuple:
    """(RMSE, its Monte-Carlo standard error from the bias's MC error, first order)."""
    r = math.sqrt(bias * bias + sd * sd) if (math.isfinite(bias) and math.isfinite(sd)) else math.inf
    if not math.isfinite(r) or r == 0.0:
        return r, math.nan
    return r, abs(bias) * float(bias_mcse) / r
