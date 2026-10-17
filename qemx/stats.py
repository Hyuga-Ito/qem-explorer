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
a *simulated* result. A ``measured`` hardware result has no bias unless a trusted ideal reference exists.

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


def rmse_with_mc(bias: float, sd: float, bias_mcse: float = 0.0) -> tuple:
    """(RMSE, its Monte-Carlo standard error from the bias's MC error, first order)."""
    r = math.sqrt(bias * bias + sd * sd) if (math.isfinite(bias) and math.isfinite(sd)) else math.inf
    if not math.isfinite(r) or r == 0.0:
        return r, math.nan
    return r, abs(bias) * float(bias_mcse) / r
