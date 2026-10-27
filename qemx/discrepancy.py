r"""Discrepancy measures between a noise model's predictions and observations (1.5).

A small set of complementary measures, each with its domain stated and checked:

=====================  =========================================================================================
measure                definition and domain
=====================  =========================================================================================
``tvd``                total variation distance ½ Σ_x |p(x) − q(x)| between two probability distributions on the
                       same outcomes; in [0, 1]
``hellinger``          Hellinger distance √(1 − BC), BC = Σ_x √(p(x) q(x)) the Bhattacharyya coefficient; in [0, 1]
``classical_fidelity`` the squared overlap BC² (Chen et al., arXiv:2308.05071v2, Eq. 1, *without* uniform-baseline
                       normalization). A classical distribution fidelity, not the quantum state fidelity
``normalized``         Lubinski et al.'s normalized fidelity max((F − F_u)/(1 − F_u), 0), F_u the overlap of the
                       ideal with the uniform distribution (arXiv:2110.03137v3, Sec. III G, Eqs. 1–3)
``residual``           observable residual o_obs − o_pred and its standardized form
                       z = (o_obs − o_pred) / √(se_obs² + se_pred² + se_model²) when every term is a valid SE
=====================  =========================================================================================

Probability distributions are validated (finite, entries ≥ −tol, total 1 ± tol). A signed quasi-probability vector —
what readout inversion or ZNE produce before any projection — is **not** a probability distribution: the distance
functions raise :class:`QuasiProbabilityError` for it instead of clipping and renormalizing silently. A physical
projection is available only explicitly (:func:`project_to_simplex`), and it reports the mass it moved so that its
bias can be evaluated separately.

Finite samples. With N shots on K outcomes the plug-in TVD of an empirical distribution is biased upward (about
√(K/(2πN)) when the outcomes are equally likely): a nonzero distance alone is not a discrepancy. ``null_distance``
gives the distance's distribution under the model at the same shots (parametric bootstrap), and ``marginal`` reduces a
high-dimensional comparison to the bits that matter. Many simultaneous tests are corrected with Holm (family-wise
error), Hochberg, or Benjamini–Hochberg (false-discovery rate): see ``adjust_pvalues``.

Noise-rate conventions (``convert_error_rate``). For the depolarizing channel Λ_p(ρ) = (1 − p)ρ + p I/d on d = 2^k:
entanglement infidelity ε = p (1 − 1/d²), average gate infidelity r = p (d − 1)/d = ε d/(d + 1). A randomized-
benchmarking error per gate estimates r (under gate-independent noise), not p; Chen et al. report ε.
"""
from __future__ import annotations

import math

import numpy as np

PROB_TOL = 1e-9


class QuasiProbabilityError(ValueError):
    """A distance between probability distributions was asked of a vector that is not one."""


def check_distribution(p, name: str = "p", tol: float = PROB_TOL) -> np.ndarray:
    """The vector as floats if it is a probability distribution (finite, entries ≥ −tol, total 1 ± tol·K)."""
    a = np.asarray(p, float)
    if a.ndim != 1 or a.size < 1:
        raise ValueError("%s must be a non-empty vector" % name)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s contains non-finite values" % name)
    neg = float(a.min())
    if neg < -tol:
        raise QuasiProbabilityError(
            "%s has negative entries (min %.3g): a signed quasi-probability is not a probability distribution, so "
            "TVD, Hellinger and classical fidelity are undefined for it. Compare observables instead, or project it "
            "explicitly with project_to_simplex and report the projection's effect." % (name, neg))
    tot = float(a.sum())
    if abs(tot - 1.0) > tol * max(1, a.size):
        raise QuasiProbabilityError("%s sums to %.9g, not 1: not a probability distribution" % (name, tot))
    return np.clip(a, 0.0, None)


def _pair(p, q):
    p, q = check_distribution(p, "p"), check_distribution(q, "q")
    if p.shape != q.shape:
        raise ValueError("the distributions have different outcome sets (%d vs %d)" % (p.size, q.size))
    return p, q


def tvd(p, q) -> float:
    p, q = _pair(p, q)
    return float(0.5 * np.abs(p - q).sum())


def bhattacharyya(p, q) -> float:
    p, q = _pair(p, q)
    return float(min(np.sqrt(p * q).sum(), 1.0))


def hellinger(p, q) -> float:
    return math.sqrt(max(0.0, 1.0 - bhattacharyya(p, q)))


def classical_fidelity(p, q) -> float:
    """(Σ √(p q))² — Chen et al. Eq. 1 (no uniform normalization). Not the quantum state fidelity."""
    return bhattacharyya(p, q) ** 2


def normalized_fidelity(p_ideal, q) -> float:
    """max((F(p, q) − F(p, u)) / (1 − F(p, u)), 0) with u uniform (Lubinski et al.). NaN when p is uniform."""
    p, q = _pair(p_ideal, q)
    fu = float(np.sqrt(p).sum() ** 2 / p.size)
    if fu >= 1 - 1e-12:
        return math.nan
    return max((classical_fidelity(p, q) - fu) / (1 - fu), 0.0)


def l1_signed(q, p) -> float:
    """½ Σ |q − p| for a *signed* vector q against a distribution p: an ℓ1 distance, explicitly not a TVD (q is not a
    distribution; the value can exceed 1). Use it only where it is labeled as such."""
    q = np.asarray(q, float)
    p = check_distribution(p, "p")
    if q.shape != p.shape or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite vector over the same outcomes")
    return float(0.5 * np.abs(q - p).sum())


def project_to_simplex(q) -> tuple:
    """(p, moved): the Euclidean projection of a signed vector onto the probability simplex and the ℓ1 mass it moved,
    ½ Σ |p − q|. Explicit only: report ``moved`` with any metric computed on ``p``."""
    from .divergence import nearest_probability
    q = np.asarray(q, float)
    p = nearest_probability(q)
    return p, float(0.5 * np.abs(p - q).sum())


def marginal(p, bits, n: int) -> np.ndarray:
    """Distribution of the classical bits ``bits`` (index bit i = classical bit i; output little-endian in ``bits``)."""
    p = check_distribution(p, "p")
    idx = np.arange(2 ** int(n))
    out = np.zeros(2 ** len(bits))
    sub = np.zeros_like(idx)
    for k, b in enumerate(bits):
        sub |= ((idx >> int(b)) & 1) << k
    np.add.at(out, sub, p)
    return out


def _dist_rows(F, p, metric):
    if metric == "tvd":
        return 0.5 * np.abs(F - p).sum(1)
    if metric == "hellinger":
        return np.sqrt(np.clip(1 - np.sqrt(F * p).sum(1), 0, None))
    if metric == "classical_fidelity":
        return np.sqrt(F * p).sum(1) ** 2
    raise ValueError("unknown metric %r" % metric)


def null_distance(p_model, shots: int, metric: str = "tvd", reps: int = 2000, rng=None, p_draws=None) -> np.ndarray:
    """Distances between the model and empirical distributions of ``shots`` samples drawn under the model: the
    sampling distribution of the plug-in metric when the model is right (parametric bootstrap).

    ``p_draws`` (K × outcomes): draws of the model's distribution under its own parameter uncertainty (a calibration
    estimated from finite shots, a fitted noise model). Each null replicate then first draws one parameter value,
    then ``shots`` samples from that distribution, and compares them with ``p_model`` (the prediction actually
    used): the observed counts come from *one* realization of the uncertain parameter, so their law is a mixture of
    multinomials. Substituting the average distribution into a single multinomial is not equivalent in general — it
    understates the spread when the parameter uncertainty is not negligible against shot noise."""
    p = check_distribution(p_model, "p_model")
    rng = np.random.default_rng(rng)
    if p_draws is None:
        F = rng.multinomial(int(shots), p, size=int(reps)) / float(shots)
    else:
        D = np.atleast_2d(np.asarray(p_draws, float))
        if D.shape[1] != p.size:
            raise ValueError("p_draws must have one column per outcome (%d), not %d" % (p.size, D.shape[1]))
        for k in range(D.shape[0]):
            check_distribution(D[k], "p_draws[%d]" % k)
        pick = rng.integers(0, D.shape[0], int(reps))
        F = np.stack([rng.multinomial(int(shots), D[k] / D[k].sum()) for k in pick]) / float(shots)
    return _dist_rows(F, p, metric)


def distance_test(p_model, counts_freq, shots: int, metric: str = "tvd", reps: int = 2000, rng=None,
                  p_draws=None) -> dict:
    """Observed plug-in distance and its p-value under the model at the same shots (one-sided: larger is worse; for
    the classical fidelity, smaller is worse). By default the model is treated as exact. To include the model's own
    parameter uncertainty, pass ``p_draws`` (draws of its distribution under that uncertainty): the null is then the
    mixture of multinomials described in :func:`null_distance` (``null`` says which was used)."""
    p = check_distribution(p_model, "p_model")
    f = check_distribution(counts_freq, "observed")
    obs = {"tvd": tvd, "hellinger": hellinger, "classical_fidelity": classical_fidelity}[metric](p, f)
    null = null_distance(p, shots, metric, reps, rng, p_draws)
    worse = (null <= obs) if metric == "classical_fidelity" else (null >= obs)
    return dict(metric=metric, observed=float(obs), null_mean=float(null.mean()), null_q95=float(
        np.quantile(null, 0.05 if metric == "classical_fidelity" else 0.95)),
        p_value=float((worse.sum() + 1) / (len(null) + 1)), shots=int(shots), outcomes=int(p.size),
        sparse=bool(int(shots) < 5 * p.size),
        null="exact model (multinomial)" if p_draws is None else "mixture over %d parameter draws" % len(p_draws))


def residual(o_obs: float, o_pred: float, se_obs: float, se_pred: float = 0.0, se_model: float = 0.0) -> dict:
    """Observable residual, its standard error and z. z is NaN when no positive standard error is available."""
    r = float(o_obs) - float(o_pred)
    terms = [float(x) for x in (se_obs, se_pred, se_model) if x is not None and math.isfinite(float(x))]
    se = math.sqrt(sum(t * t for t in terms)) if terms else math.nan
    return dict(residual=r, se=se, z=r / se if se and se > 0 else math.nan)


def two_sided_p(z) -> np.ndarray:
    from scipy.stats import norm
    z = np.asarray(z, float)
    return np.where(np.isfinite(z), 2 * norm.sf(np.abs(z)), np.nan)


def adjust_pvalues(p, method: str = "holm") -> np.ndarray:
    """Adjusted p-values (NaN entries stay NaN and are not counted): ``holm`` (family-wise error, any dependence),
    ``hochberg`` (FWER, independent or positively dependent tests), ``bh`` (Benjamini–Hochberg false-discovery rate,
    independent or positively dependent tests), ``bonferroni``."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    ps = pv[order]
    if method == "bonferroni":
        adj = np.minimum(ps * m, 1.0)
    elif method == "holm":
        adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ps), 1.0)
    elif method == "hochberg":
        adj = np.minimum(np.minimum.accumulate(((m - np.arange(m)) * ps)[::-1])[::-1], 1.0)
    elif method == "bh":
        adj = np.minimum(np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1], 1.0)
    else:
        raise ValueError("unknown method %r" % method)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def simultaneous_z(m: int, alpha: float = 0.05) -> float:
    """|z| threshold of m simultaneous two-sided tests at family-wise level α (Šidák, independent tests)."""
    from scipy.stats import norm
    if m < 1:
        return math.nan
    a1 = 1 - (1 - float(alpha)) ** (1.0 / int(m))
    return float(norm.isf(a1 / 2))


# ------------------------------------------------------------------ noise-rate conventions
RATE_KINDS = ("depolarizing_p", "entanglement_infidelity", "average_gate_infidelity")


def convert_error_rate(value: float, frm: str, to: str, qubits: int = 1) -> float:
    """Convert between the conventions of a depolarizing channel on ``qubits`` qubits (see the module text). An RB
    error per gate is an *average gate infidelity* under gate-independent noise."""
    if frm not in RATE_KINDS or to not in RATE_KINDS:
        raise ValueError("conventions: %s" % ", ".join(RATE_KINDS))
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError("an error rate must be finite and ≥ 0")
    d = 2 ** int(qubits)
    p = {"depolarizing_p": v, "entanglement_infidelity": v / (1 - 1 / d ** 2),
         "average_gate_infidelity": v * d / (d - 1)}[frm]
    return {"depolarizing_p": p, "entanglement_infidelity": p * (1 - 1 / d ** 2),
            "average_gate_infidelity": p * (d - 1) / d}[to]
