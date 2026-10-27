r"""Neighbor-informed learning (NIL): Z. Chen, B. Cheng, M. Gao, X. Lin, R. Zhang, Z. Wei & Z. Ji, *Scalable
quantum error mitigation with neighbor-informed learning*, arXiv:2512.12578 (2025).

Features are the noisy expectation values of **neighbor circuits**: copies of the target with one Pauli
:math:`P\in\{X,Y,Z\}` inserted right after one gate (:math:`U_j\to PU_j`, *weight-1* neighbors), drawn at random from
all such insertion points; or the target at amplified noise (ZNE neighbors, by global or local unitary folding). The
estimate is a linear combination learned on training circuits whose ideal values are classically computable:

.. math::
    \hat O=\sum_j c_j\,\langle\widetilde{C}_j\rangle\;(+\,b),\qquad
    \mathbf c=\arg\min_{\mathbf c}\frac1T\sum_{i=1}^T\Bigl(\sum_j c_j x^{(i)}_j-y^{(i)}\Bigr)^2
    \quad\text{s.t.}\ \|\mathbf c\|_1\le\gamma

with :math:`x^{(i)}_j` the noisy value of neighbor :math:`j` of training circuit :math:`i` and :math:`y^{(i)}` the
ideal value of training circuit :math:`i`. Training circuits replace the angle of every non-Clifford rotation by a
uniform draw from :math:`\{0,\pi/2,\pi,3\pi/2\}` (a rotational 2-design, Chen et al. Algorithm 2) and carry the same
insertions as the target. Chen et al. fit the ℓ1-*constrained* least squares above (``model="l1"``; γ = 2 for Pauli
neighbors, 5 for ZNE neighbors, no intercept). Here the same features can be fitted with other rules:

=============  =====================================================================================================
``l1``         :math:`\min\|y-X\mathbf c\|^2` s.t. :math:`\|\mathbf c\|_1\le\gamma` (Chen et al.)
``lasso``      :math:`\min\|y-X\mathbf c\|^2+\alpha\|\mathbf c\|_1`
``ridge``      :math:`\min\|y-X\mathbf c\|^2+\alpha\|\mathbf c\|_2^2`
``ols``        :math:`\min\|y-X\mathbf c\|^2` (minimum-norm solution when it is not unique)
``elasticnet`` :math:`\min\|y-X\mathbf c\|^2+\alpha\,[\rho\|\mathbf c\|_1+(1-\rho)\|\mathbf c\|_2^2]`
               (ρ = 1: Lasso, ρ = 0: Ridge, with the same α)
``nnls``       :math:`\min\|y-X\mathbf c\|^2` s.t. :math:`\mathbf c\ge0`
=============  =====================================================================================================

The penalties use the same convention as the feature-regression method (``learning.py``): the squared error is
*summed* over the training circuits. With ``intercept=True`` an unpenalized constant b is fitted as well (the data are
centered; the penalty or constraint acts on **c** only).

Solvers. OLS (minimum-norm) and Ridge are solved exactly. The rules with an ℓ1 term or a sign constraint are solved
with a stabilization ε‖c‖₂², ε = (10⁻⁶‖X‖₂)², which makes them strictly convex (a unique, order-independent solution,
also when neighbor circuits are linearly dependent on the training set) at an objective cost of at most ε‖c*‖²; columns
identical on the training set are merged and share their weight equally (see :func:`nil_fit`). The ℓ1 rules
are solved by a primal active-set method (Lasso, elastic net) and by a safeguarded Newton iteration on the multiplier
over exact Lasso points, finished by the Karush–Kuhn–Tucker point of the final face (ℓ1-constrained). Optimality is
*certified*, not assumed: a duality gap (Lasso, elastic net) or Frank–Wolfe gap (ℓ1-constrained) bounds the excess of
the stabilized objective by 10⁻¹⁰‖y‖². A fit without a certificate is flagged (``details.solver_ok``), never silently
presented as optimal.

Cost (equal-budget ledger)
--------------------------
* training: :math:`T\cdot k` circuits (T training circuits × k neighbor circuits each), ``train_shots`` shots each;
* target: the k neighbor circuits of the target. Given the fitted **c**, the target shots are split
  :math:`N_j\propto|c_j|\sigma_j` (``target_split="optimal"``; σ_j the per-shot standard deviation of circuit j,
  the convention of ZNE's split), so :math:`v=(\sum_j|c_j|\sigma_j)^2`; circuits with :math:`c_j=0` are not run.
  ``target_split="equal"`` gives every neighbor circuit N/k shots (the plan of Chen et al.'s experiments, which run
  every neighbor circuit with the same number of shots, 10⁴): :math:`v=k\sum_jc_j^2\sigma_j^2`. With the optimal
  split the sampling overhead is :math:`(\sum_j|c_j|\sigma_j)^2\le\|\mathbf c\|_1^2\max_j\sigma_j^2`.

The split is decided after the training data are in (it needs **c**); on IBM hardware, where every circuit of a plan
is sent in one batch, only the equal split can be planned honestly and the optimal split is refused.

Departures from the paper, stated: Chen et al. use ~10³ training circuits and a classical stabilizer simulator; the
default here is T = 16 (as for CDR) and exact simulation. Whether the unmodified circuit is itself a feature is a
parameter (``include_bare``, default on, as in the ZNE construction whose α = 1 circuit is the target). Inserted
Paulis are physical gates: on a device model X and Y carry the device's single-qubit gate error and Z is a virtual
frame change, as on hardware.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np

from .base import MethodResult, Param, QEMMethod, register
from .raw import cal_fields, display_circuit, readout_fun
from ..backends.base import NotSupported
from ..budget import optimal_integer_split
from ..resources import check
from ..circuit import CLIFFORD_ANGLE, Op
from ..transforms import fold_global, fold_local, non_clifford_slots

NIL_MODELS = ("l1", "lasso", "ridge", "ols", "elasticnet", "nnls")
NIL_MODEL_LABEL = {"l1": "ℓ1-constrained", "lasso": "Lasso", "ridge": "Ridge", "ols": "OLS",
                   "elasticnet": "Elastic net", "nnls": "NNLS"}
NIL_PAULIS = ("x", "y", "z")
NIL_ENSEMBLES = ("design4", "uniform", "near")
_NIL_VECTOR_REFIT = True     # readout-recalibrated refits through Mᵀo (tests switch it off to compare both paths)


# ====================================================================== regression (certified solvers)
NIL_GAP_REL = 1e-10      # certificate: the excess objective is at most NIL_GAP_REL·‖y‖² (duality / Frank–Wolfe gap)


def _nil_tol(X, y, c=None, lam=0.0) -> float:
    """Rounding scale of g = Xᵀ(y − Xc): a violation below it is not acted on (optimality is certified by a gap).
    ‖X‖ is the Frobenius norm (an upper bound of the spectral norm, without an SVD in every solver step)."""
    sx = float(np.sqrt(np.sum(X * X))) if X.size else 0.0
    sc = float(np.linalg.norm(c)) if c is not None and np.size(c) else 0.0
    return 1e-12 * (sx * float(np.linalg.norm(y)) + sx * sx * sc + float(lam)) + 1e-300


def _nil_lstsq(A, b) -> np.ndarray:
    return np.linalg.lstsq(A, b, rcond=None)[0] if A.size else np.zeros(A.shape[1])


def _nil_spd_solve(A, rhs) -> np.ndarray:
    """(AᵀA)⁻¹ rhs through the QR factor of A (condition κ(A), not κ(A)²); minimum-norm when AᵀA is singular."""
    m = A.shape[1]
    if m == 0:
        return np.zeros(0)
    if A.shape[0] >= m:
        R = np.linalg.qr(A, mode="r")
        d = np.abs(np.diag(R))
        if d.size and d.min() > 1e-13 * d.max():
            from scipy.linalg import solve_triangular
            z = solve_triangular(R, rhs, trans="T", lower=False, check_finite=False)      # Rᵀ z = rhs
            return solve_triangular(R, z, lower=False, check_finite=False)                 # R x = z
    return np.linalg.lstsq(A.T @ A, rhs, rcond=None)[0]


def _nil_kkt_lasso(X, y, lam, S, s) -> np.ndarray:
    """The Lasso point with active set S and signs s: X_SᵀX_S c_S = X_Sᵀy − (λ/2) s."""
    c = np.zeros(X.shape[1])
    if len(S):
        A = X[:, S]
        c[S] = _nil_spd_solve(A, A.T @ y - 0.5 * lam * np.asarray(s, float))
    return c


def _nil_kkt_l1(X, y, gamma, S, s) -> tuple:
    """(c, μ) on active set S with signs s for min ‖y − Xc‖² s.t. ‖c‖₁ ≤ γ, constraint active:
    X_SᵀX_S c_S + (μ/2) s = X_Sᵀy and sᵀc_S = γ (μ is the multiplier; the Lasso with λ = μ has the same solution)."""
    c = np.zeros(X.shape[1])
    if not len(S):
        return c, math.nan
    A = X[:, S]
    s = np.asarray(s, float)
    c_ls = _nil_spd_solve(A, A.T @ y)
    d = _nil_spd_solve(A, s)
    den = float(s @ d)
    if not math.isfinite(den) or abs(den) < 1e-300:
        return c, math.nan
    mu = 2.0 * (float(s @ c_ls) - gamma) / den
    c[S] = c_ls - 0.5 * mu * d
    return c, mu


def _nil_gap_lasso(X, y, c, lam) -> float:
    """Duality gap of ‖y − Xc‖² + λ‖c‖₁ (an upper bound on the excess objective): the dual is
    max 2νᵀy − ‖ν‖² over ‖Xᵀν‖∞ ≤ λ/2, evaluated at the residual scaled into that set."""
    if not np.all(np.isfinite(c)):
        return math.inf
    r = y - X @ c
    m = float(np.max(np.abs(X.T @ r))) if X.shape[1] else 0.0
    nu = r if m <= 0.5 * lam else r * (0.5 * lam / m)
    return float(r @ r + lam * np.abs(c).sum() - (2.0 * nu @ y - nu @ nu))


def _nil_gap_l1(X, y, c, gamma) -> float:
    """Frank–Wolfe gap of min ‖y − Xc‖² over ‖c‖₁ ≤ γ at c (an upper bound on the excess objective if c is
    feasible; infinite otherwise): max over the ball of ∇f·(c − s) = 2(γ‖Xᵀr‖∞ − cᵀXᵀr)."""
    if not np.all(np.isfinite(c)) or float(np.abs(c).sum()) > gamma * (1 + 1e-12):
        return math.inf
    g = X.T @ (y - X @ c)
    return float(2.0 * (gamma * np.max(np.abs(g)) - c @ g)) if X.shape[1] else 0.0


def _nil_certified(gap, y) -> bool:
    return bool(gap <= NIL_GAP_REL * max(float(y @ y), 1e-300))


def _nil_lasso_as(X, y, lam, c0=None) -> np.ndarray:
    """Primal active-set method for min ‖y − Xc‖² + λ‖c‖₁ (every step lowers the objective; capped at
    100k + 100 outer iterations — the result is certified by its duality gap afterwards, never assumed).

    On a face (support S, signs s fixed) the objective is the quadratic ‖y − Xc‖² + λ sᵀc; its minimizer h solves
    X_SᵀX_S h_S = X_Sᵀy − (λ/2)s. The iterate moves towards h as far as the signs allow (ratio test); a coordinate
    that reaches zero leaves S. At a face minimizer, the coordinate with the largest |g_j| = |X_jᵀ(y − Xc)| > λ/2
    enters with the sign of g_j (a descent direction); otherwise the point is optimal. Duplicate features never
    enter (their correlation equals that of their active copy)."""
    k = X.shape[1]
    tau = 0.5 * lam
    c = np.zeros(k) if c0 is None else np.array(c0, float)
    c[~np.isfinite(c)] = 0.0
    s = np.sign(c)
    S = set(int(j) for j in np.flatnonzero(c))
    for _ in range(100 * k + 100):
        for _inner in range(6 * k + 10):                    # descend to the minimizer of the current face
            if not S:
                break
            Sa = np.array(sorted(S), dtype=int)
            A = X[:, Sa]
            rank = len(Sa)
            if A.shape[0] < len(Sa):
                rank = -1
            else:
                dR = np.abs(np.diag(np.linalg.qr(A, mode="r")))
                if dR.min() <= 1e-10 * dR.max():
                    rank = -1                               # possibly dependent: decide with the SVD
            if rank < 0:
                sv, vt = np.linalg.svd(A, full_matrices=True)[1:]
                rank = int(np.sum(sv > 1e-12 * max(sv[0], 1e-300) * max(A.shape))) if sv.size else 0
            if rank < len(Sa):
                # dependent columns: the face quadratic has a flat direction v (Av = 0) along which λ sᵀc does not
                # increase; follow it until a coordinate reaches zero (the fit is unchanged, the support shrinks)
                v = vt[-1]
                ss = s[Sa]
                if ss @ v > 0 or (ss @ v == 0 and not np.any(c[Sa] * v < 0)):
                    v = -v
                neg = np.flatnonzero(c[Sa] * v < 0)
                if not len(neg):
                    break
                tj = -c[Sa][neg] / v[neg]
                i = int(np.argmin(tj))
                c[Sa] = c[Sa] + tj[i] * v
                c[Sa[neg[i]]] = 0.0
                S.discard(int(Sa[neg[i]]))
                for j in list(S):
                    if c[j] * s[j] <= 0:
                        c[j] = 0.0
                        S.discard(j)
                continue
            h = _nil_kkt_lasso(X, y, lam, Sa, s[Sa])
            bad = [j for j in Sa if h[j] * s[j] <= 0]
            if not bad:
                c = h
                break
            ts = {j: c[j] / (c[j] - h[j]) for j in bad}      # c_j s_j > 0 ≥ h_j s_j → t_j ∈ (0, 1]
            t = min(ts.values())
            c = c + t * (h - c)
            for j in bad:
                if ts[j] <= t * (1 + 1e-12):
                    c[j] = 0.0
                    S.discard(j)
            for j in list(S):                               # rounding: anything that crossed zero leaves
                if c[j] * s[j] <= 0:
                    c[j] = 0.0
                    S.discard(j)
        g = X.T @ (y - X @ c)
        off = np.array([j for j in range(k) if j not in S], dtype=int)
        if not len(off):
            return c
        w = int(off[np.argmax(np.abs(g[off]))])
        if abs(g[w]) <= tau + _nil_tol(X, y, c, lam):
            return c
        S.add(w)
        s[w] = float(np.sign(g[w]))
    return c


def _nil_lasso_cd(X, y, lam, tol, max_iter) -> np.ndarray:
    from sklearn.linear_model import Lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Lasso(alpha=lam / (2.0 * X.shape[0]), fit_intercept=False, tol=tol, max_iter=max_iter,
                  precompute=True).fit(X, y)
    return np.asarray(m.coef_, float).copy()


def _nil_lasso(X, y, lam, start=None, fallback=True) -> tuple:
    """(c, ok, λ) for min ‖y − Xc‖² + λ‖c‖₁ (λ > 0): the active-set method (warm-started if ``start`` is given),
    certified by its duality gap (≤ NIL_GAP_REL·‖y‖²); fallback: coordinate descent followed by the active-set method.
    An uncertified solution is returned with ok = False, never presented as optimal."""
    k = X.shape[1]
    best = (math.inf, np.zeros(k))
    for c0 in ([start] if start is not None else []) + [None] + (["cd"] if fallback else []):
        if isinstance(c0, str):
            try:
                c0 = _nil_lasso_cd(X, y, lam, 1e-12, 50000)
            except (ValueError, FloatingPointError):
                continue
        c = _nil_lasso_as(X, y, lam, c0)
        gap = _nil_gap_lasso(X, y, c, lam)
        if gap < best[0]:
            best = (gap, c)
        if _nil_certified(gap, y):
            return c, True, lam
    return best[1], False, lam


def _nil_min_l1_ls(X, y) -> np.ndarray | None:
    """The least-squares solution of smallest ℓ1 norm: the unique one for full column rank; otherwise the linear
    program min ‖c‖₁ s.t. Xc = ŷ (ŷ the projection of y on the column space; HiGHS), polished by least squares on
    the support it selects. None if the program fails."""
    T, k = X.shape
    if T >= k:
        R = np.linalg.qr(X, mode="r")
        dg = np.abs(np.diag(R))
        if dg.min() > 1e-13 * dg.max():
            return _nil_lstsq(X, y)                    # orthogonal factorization: condition κ, not κ²
    from scipy.optimize import linprog
    yh = X @ _nil_lstsq(X, y)
    sc = max(float(np.max(np.abs(X))), 1e-300)
    res = linprog(np.ones(2 * k), A_eq=np.hstack([X, -X]) / sc, b_eq=yh / sc, bounds=(0, None), method="highs",
                  options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
    if res.status != 0 or res.x is None:
        return None
    c0 = res.x[:k] - res.x[k:]
    S = np.flatnonzero(np.abs(c0) > 1e-12 * max(float(np.max(np.abs(c0))), 1e-300))
    c = np.zeros(k)
    if len(S):
        c[S] = _nil_lstsq(X[:, S], y)
    return c


def _nil_kkt_ok_l1(X, y, c, gamma, mu) -> bool:
    """Exact optimality of c for min ‖y − Xc‖² s.t. ‖c‖₁ ≤ γ with an active constraint and multiplier μ ≥ 0, up to
    rounding: ‖c‖₁ = γ, g_S = (μ/2)·sign(c_S) and |g_j| ≤ μ/2 off the support (g = Xᵀ(y − Xc))."""
    if not (math.isfinite(mu) and mu >= 0) or not np.all(np.isfinite(c)):
        return False
    if abs(float(np.abs(c).sum()) - gamma) > 1e-12 * gamma:
        return False
    g = X.T @ (y - X @ c)
    tol = 1e3 * _nil_tol(X, y, c, mu)
    S = c != 0
    return bool(np.all(np.abs(g[S] - 0.5 * mu * np.sign(c[S])) <= tol) and np.all(np.abs(g[~S]) <= 0.5 * mu + tol))


def _nil_l1(X, y, gamma, warm=None) -> tuple:
    """(c, ok, μ) for min ‖y − Xc‖² s.t. ‖c‖₁ ≤ γ.

    If the least-squares solution of smallest ℓ1 norm lies inside the ball the constraint is slack (μ = 0) and that
    solution is used (the unique least-squares solution for full column rank). Otherwise the constraint is active:
    ‖c(λ)‖₁ of the Lasso solutions is continuous and non-increasing in λ, and the solution is c(μ) with
    ‖c(μ)‖₁ = γ. A safeguarded Newton iteration finds μ — on a face of fixed signs ‖c(λ)‖₁ = sᵀc_ls − (λ/2) sᵀd is
    linear, so a Newton step lands on the face's crossing — with exact (active-set) Lasso points, finished by the
    exact KKT point of the final face: a face point is returned as soon as it satisfies the KKT conditions to
    rounding (_nil_kkt_ok_l1), which makes the result the unique optimum of the (stabilized, strictly convex) problem
    rather than any point within the gap tolerance — independent of the column order. Certificate (ok): the excess
    objective is at most min(Frank–Wolfe gap, f(c) − f_LS) ≤ NIL_GAP_REL·‖y‖²; if no face point passes the KKT
    check within the iteration cap, the best certified point is returned."""
    T, k = X.shape
    if gamma <= 0 or not np.any(X) or not np.any(y):
        return np.zeros(k), True, 0.0
    r_ls = y - X @ _nil_lstsq(X, y)
    f_ls = float(r_ls @ r_ls)
    best = (math.inf, np.zeros(k), 0.0)

    def consider(c, mu):
        nonlocal best
        c = np.asarray(c, float)
        if not np.all(np.isfinite(c)):
            return False
        n1 = float(np.abs(c).sum())
        if n1 > gamma:                                      # rounding: stay inside the ball
            c = c * (gamma / n1)
        r = y - X @ c
        cert = min(_nil_gap_l1(X, y, c, gamma), max(float(r @ r) - f_ls, 0.0))
        if cert < best[0]:
            best = (cert, c, float(mu))
        return _nil_certified(cert, y)
    c_bp = _nil_min_l1_ls(X, y)
    if c_bp is not None and float(np.abs(c_bp).sum()) <= gamma * (1 + 1e-12):
        ok = consider(c_bp, 0.0)
        return best[1], ok, 0.0
    lam_max = 2.0 * float(np.max(np.abs(X.T @ y)))
    lo, hi, c_hi = 0.0, lam_max, np.zeros(k)
    if warm is not None and math.isfinite(warm[1]) and 0 < warm[1] < lam_max:
        lam = float(warm[1])                                # a nearby problem's multiplier and solution
        c = _nil_lasso(X, y, lam, start=np.asarray(warm[0], float), fallback=False)[0]
    else:
        lam = 0.5 * lam_max
        c = _nil_lasso(X, y, lam, fallback=False)[0]
    tried = set()
    for _ in range(80):
        S = np.flatnonzero(c != 0)
        lam_new = math.nan
        if len(S):
            sg_ = np.sign(c[S])
            key = (tuple(S.tolist()), tuple(sg_.tolist()))
            cs, lam_new = _nil_kkt_l1(X, y, gamma, S, sg_)
            # the exact constrained point of this face, if it is consistent, is the answer once certified
            if key not in tried and math.isfinite(lam_new) and lam_new >= 0 and np.all(cs[S] * sg_ >= 0):
                tried.add(key)
                if consider(cs, lam_new) and _nil_kkt_ok_l1(X, y, cs, gamma, lam_new):
                    return cs, True, float(lam_new)
        n1 = float(np.abs(c).sum())
        if n1 > gamma:
            lo = max(lo, lam)
        else:
            hi, c_hi = min(hi, lam), c
        if lo > 0 and hi / lo < 1 + 1e-14:
            break
        if not (math.isfinite(lam_new) and lo < lam_new < hi):
            lam_new = math.sqrt(lo * hi) if lo > 0 else hi * 1e-2
        if lam_new <= lam_max * 1e-300:
            break
        lam = lam_new
        c = _nil_lasso(X, y, lam, start=c, fallback=False)[0]
    for cand in (c, c_hi):
        if consider(cand, lam):
            break
    c_b, mu_b = best[1], best[2]
    if float(np.abs(c_b).sum()) < gamma * (1 - 1e-9):
        mu_b = 0.0                                          # an interior point: the multiplier is zero
    return c_b, bool(_nil_certified(best[0], y)), mu_b


def _nil_nnls(X, y) -> tuple:
    """(c, ok) for min ‖y − Xc‖² s.t. c ≥ 0 (Lawson–Hanson active set, polished on the passive set and checked)."""
    from scipy.optimize import nnls
    c0 = np.asarray(nnls(X, y, maxiter=50 * max(X.shape[1], 1))[0], float)

    def good(c):
        tol = 1000.0 * _nil_tol(X, y, c)
        g = X.T @ (y - X @ c)
        p = c > 0
        return bool(np.all(c >= 0) and np.all(np.abs(g[p]) <= tol) and np.all(g[~p] <= tol))
    P = np.flatnonzero(c0 > 0)
    c = np.zeros(X.shape[1])
    if len(P):
        c[P] = _nil_lstsq(X[:, P], y)
    if good(c):
        return c, True
    return c0, good(c0)


NIL_STAB = 1e-6          # stabilization of the ℓ1 / sign-constrained rules: ε = (NIL_STAB·‖X‖₂)² (see nil_fit)


def nil_fit(X, y, model: str, alpha: float = 0.0, l1_ratio: float = 0.5, l1_bound: float = 2.0,
            intercept: bool = False, warm=None) -> tuple:
    """(coef, intercept, info) of one regression rule on the training data (rows: training circuits, columns:
    neighbor circuits). ``info``: ok (optimality certified), lam (the effective ℓ1 multiplier), eps (the
    stabilization), model. ``warm``: (coef, info) of a nearby problem, a starting point for the solver only.

    Exact rules. OLS is the minimum-norm least-squares solution (SVD, NumPy's default cutoff) and Ridge (α > 0) the
    exact ridge solution: both are unique, and nothing is added to them.

    Stabilized rules. The rules with an ℓ1 term or a sign constraint (ℓ1-constrained, Lasso, elastic net, NNLS) are
    solved with an added ε‖c‖₂², ε = (1e-6·‖X‖₂)² (centred X with an intercept). Without it their solution is not
    unique whenever neighbor circuits are linearly dependent on the training circuits — common: a Pauli that commutes
    to the measurement of every Clifford training circuit gives a copy of another column — while they differ on the
    target; the estimate would then depend on solver tie-breaks, and the active-set face systems could not be solved.
    With ε the problem is strictly convex (a unique solution, continuous in the data; face systems of condition
    ≤ 1e6), and its optimum exceeds the unstabilized one by at most ε‖c*‖² — about 1e-12 of ‖y‖² for γ-bounded
    coefficients, below the solver certificate. The certificates (``ok``) refer to the stabilized objective.

    Copies. Columns identical on the training data (to 1e-12 of its scale, _nil_twins) are merged before solving:
    for a fixed total C of a group of m copies, the ℓ2 terms (ε, Ridge, minimum norm) are smallest for the equal split
    (C²/m), and the ℓ1 terms (≥ |C|) and the sign constraint of NNLS are met by it, so every rule's solution splits C
    equally — the point an interior-point solver such as Chen et al.'s CVXPY/MOSEK converges to. Solving the merged
    problem (one column per group, ℓ2 weight 1/m) gives that split exactly and makes the fit independent of the column
    order.

    Input contract (checked here, at the lowest level, so every caller gets the same validation): X a finite 2-D
    array (T, k) with T ≥ 1 rows, y a finite vector of length T, α ≥ 0 and finite, 0 ≤ ρ ≤ 1, and for the ℓ1-bounded
    rule a finite bound γ ≥ 0. γ = 0 is accepted and means "no weight on any feature": the only feasible coefficient
    vector is c = 0 (with an intercept, the estimate is the mean training label); it is a legitimate, if useless,
    constraint, not an error."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("NIL training features must be a non-empty 2-D array (training circuits × features), got "
                         "shape %s" % (X.shape,))
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("NIL training labels must be a vector with one value per training circuit (%d), got shape %s"
                         % (X.shape[0], y.shape))
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("NIL training data contain non-finite values")
    try:
        alpha, l1_ratio, l1_bound = float(alpha), float(l1_ratio), float(l1_bound)
    except (TypeError, ValueError):
        raise ValueError("NIL regularization parameters must be numbers") from None
    if not (math.isfinite(alpha) and alpha >= 0):
        raise ValueError("the NIL penalty α must be finite and ≥ 0, got %r" % alpha)
    if not (math.isfinite(l1_ratio) and 0.0 <= l1_ratio <= 1.0):
        raise ValueError("the elastic-net mix ρ must be in [0, 1], got %r" % l1_ratio)
    if model == "l1" and not (math.isfinite(l1_bound) and l1_bound >= 0):
        raise ValueError("the ℓ1 bound γ must be finite and ≥ 0, got %r" % l1_bound)
    if intercept:
        xm, ym = X.mean(0), float(np.mean(y))
        Xc, yc = X - xm, y - ym
    else:
        xm, ym = np.zeros(X.shape[1]), 0.0
        Xc, yc = X, y
    k = X.shape[1]
    eps = (NIL_STAB * float(np.linalg.norm(Xc, 2))) ** 2 if Xc.size else 0.0
    groups = _nil_groups(Xc)                                # copies merged: one column per group (see above)
    m = np.array([len(g) for g in groups], float)
    Xg = np.column_stack([Xc[:, g].mean(1) for g in groups]) if groups else Xc
    kg = len(groups)

    def aug(l2):
        """‖y − X_g C‖² + l2 Σ C_g²/m_g as one least-squares problem on augmented data."""
        if l2 <= 0:
            return Xg, yc
        return np.vstack([Xg, np.diag(np.sqrt(l2 / m))]), np.concatenate([yc, np.zeros(kg)])
    wc = None if warm is None else np.array([float(np.sum(np.asarray(warm[0], float)[g])) for g in groups])
    wl = None if warm is None else float(warm[1].get("lam", math.nan))
    ok, lam = True, 0.0

    def min_norm_ls():
        """argmin ‖y − Xc‖² of least original-space norm: Σ C_g²/m_g, i.e. lstsq in D = C/√m."""
        r = np.sqrt(m)
        return r * _nil_lstsq(Xg * r[None, :], yc)
    if model not in NIL_MODELS:
        raise ValueError("unknown NIL regression model %r (choose from %s)" % (model, ", ".join(NIL_MODELS)))
    rho = float(l1_ratio)
    l1p, l2p = (float(alpha) * rho, float(alpha) * (1.0 - rho)) if model == "elasticnet" else (0.0, 0.0)
    exact = model == "ols" or (model in ("ridge", "lasso") and alpha <= 0) or model == "ridge" or \
        (model == "elasticnet" and l1p <= 0)
    if exact:
        eps = 0.0                                           # least-squares rules: unique without stabilization
    if model == "ols" or (model in ("ridge", "lasso") and alpha <= 0) or (model == "elasticnet" and alpha <= 0):
        c = min_norm_ls()
    elif model == "ridge":
        c = _nil_lstsq(*aug(float(alpha)))
    elif model == "lasso":
        c, ok, lam = _nil_lasso(*aug(eps), float(alpha), start=wc)
    elif model == "elasticnet":
        if l1p <= 0:
            c = _nil_lstsq(*aug(l2p))
        else:
            c, ok, lam = _nil_lasso(*aug(l2p + eps), l1p, start=wc)
    elif model == "l1":
        c, ok, lam = _nil_l1(*aug(eps), float(l1_bound), warm=None if wc is None else (wc, wl))
    else:                                                   # nnls
        c, ok = _nil_nnls(*aug(eps))
    cg = np.asarray(c, float)
    c = np.zeros(k)
    for g, Cg, mg in zip(groups, cg, m):
        c[g] = Cg / mg
    b = float(ym - xm @ c) if intercept else 0.0
    return c, b, dict(ok=bool(ok), lam=float(lam), eps=float(eps), model=model,
                      merged=[list(map(int, g)) for g in groups if len(g) > 1])


def _nil_groups(X) -> list:
    """Every column in exactly one group: the groups of _nil_twins and the remaining columns alone, in the order of
    their first column."""
    twins = _nil_twins(X)
    inside = {j for g in twins for j in g}
    out = twins + [[j] for j in range(X.shape[1]) if j not in inside]
    return sorted(out, key=lambda g: g[0])


# ====================================================================== neighbor and training circuits
def nil_pool(spec, positions: str = "2q") -> list:
    """Every weight-1 insertion point (op index, qubit, Pauli): a Pauli right after a gate, on one of its qubits —
    after every two-qubit gate (``positions="2q"``) or after every gate (``"all"``)."""
    out = []
    for i, o in enumerate(spec.ops):
        if o.kind != "gate" or (positions == "2q" and o.k != 2):
            continue
        for q in o.qubits:
            for P in NIL_PAULIS:
                out.append((i, int(q), P))
    return out


def nil_gate_numbers(spec, positions: str = "2q") -> dict:
    """op index → its number among the gates that can carry an insertion (0-based; the labels of the neighbors)."""
    idx = [i for i, o in enumerate(spec.ops) if o.kind == "gate" and (positions != "2q" or o.k == 2)]
    return {i: g for g, i in enumerate(idx)}


def nil_insert(spec, i: int, q: int, P: str):
    """The neighbor circuit: Pauli P on qubit q right after operation i."""
    ops = list(spec.ops)
    ops.insert(i + 1, Op(P, (q,), (), "gate", "nil"))
    return spec.replace(ops=ops, name="%s ·%s(q%d) after op %d" % (spec.name, P.upper(), q, i))


def nil_training_circuit(spec, rng, ensemble: str = "design4"):
    """A training circuit: every non-Clifford rotation gets a new angle — ``design4``: a uniform multiple of its
    Clifford step (rz, rx, ry, p, rzz: {0, π/2, π, 3π/2}, Chen et al. Algorithm 2; cp: {0, π}); ``uniform``: a uniform
    angle in [0, 2π) (not Clifford: the ideal values need a full classical simulation); ``near``: the Clifford angle
    near the original one, probability ∝ exp(−Δθ²) (CDR's near-Clifford circuits). T and T† count as RZ(±π/4).
    Every other operation, and the position of every operation, is unchanged, so insertions carry over."""
    ops = list(spec.ops)
    for i in non_clifford_slots(spec):
        o = ops[i]
        if o.name in ("t", "tdg"):
            name, step, th0 = "rz", math.pi / 2, (math.pi / 4 if o.name == "t" else -math.pi / 4)
        else:
            name, step, th0 = o.name, CLIFFORD_ANGLE[o.name], float(o.params[0])
        n_c = int(round(2 * math.pi / step))
        if ensemble == "design4":
            th = step * int(rng.integers(n_c))
        elif ensemble == "uniform":
            th = float(rng.uniform(0.0, 2 * math.pi))
        elif ensemble == "near":
            cands = np.arange(n_c) * step
            w = np.exp(-np.angle(np.exp(1j * (cands - th0))) ** 2)
            th = float(rng.choice(cands, p=w / w.sum()))
        else:
            raise ValueError("unknown training ensemble %r (choose from %s)" % (ensemble, ", ".join(NIL_ENSEMBLES)))
        ops[i] = Op(name, o.qubits, (th,), "gate", "train")
    return spec.replace(ops=ops, name=spec.name + " (NIL training)")


def _nil_seed(seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible streams of one method seed (neighbor choice, training circuits)."""
    return np.random.default_rng([int(seed) % (2 ** 63), 0x4E494C, int(stream)])


# ====================================================================== the method
@register
class NeighborInformedLearning(QEMMethod):
    key = "nil"
    label = "Neighbor-informed learning (Chen et al.)"
    family = "learning"
    tex = (r"\hat O=\sum_j c_j\,\langle\widetilde{C}_j\rangle,\qquad \mathbf c=\arg\min_{\|\mathbf c\|_1\le\gamma}"
           r"\sum_{i=1}^{T}\Bigl(\sum_j c_j\,x^{(i)}_j-y^{(i)}\Bigr)^2")
    why = (r"Neighbor circuits are the target with one extra Pauli after one gate. They see almost the same noise, "
           r"so a linear combination of their noisy values can cancel the noise. The combination is learned on "
           r"training circuits that share the insertions but have Clifford angles (ideal values computable). "
           r"Chen et al. bound $\|\mathbf c\|_1\le\gamma$, which bounds the shot cost $\gamma^2$; other regression "
           r"rules trade bias against that cost differently.")
    params = (
        Param("neighbors", "neighbor circuits", "choice", "pauli", ("pauli", "fold"),
              help="pauli: one X, Y or Z inserted after one gate (weight-1 neighbors); fold: the circuit at the "
                   "noise scales below (Chen et al.'s ZNE neighbors, by global or local folding)."),
        Param("positions", "insertion points", "choice", "2q", ("2q", "all"),
              help="Pauli neighbors: after every two-qubit gate (on either of its qubits), or after every gate."),
        Param("n_neighbors", "Pauli neighbors drawn (s)", "int", 12, lo=1, hi=100000, step=1,
              help="Drawn at random from all insertion points (all of them if fewer). The draw depends only on the "
                   "seed and is nested: the first s of a larger draw are the smaller draw."),
        Param("include_bare", "include the unmodified circuit", "bool", True,
              help="Pauli neighbors only: the target itself as one more feature. (Fold neighbors contain the "
                   "unmodified circuit exactly when λ = 1 is among the scales.)"),
        Param("scales", r"fold scales $\lambda$", "floats", (1.0, 3.0, 5.0),
              help="Fold neighbors only: noise scales λ ≥ 1 (nominal: two-qubit gates of the folded circuit ÷ "
                   "those of the original)."),
        Param("folding", "folding of the fold neighbors", "choice", "global", ("global", "local"),
              help="Fold neighbors only. global: U(U†U)^k; local: every two-qubit gate G → G(G†G)^k (a seeded "
                   "random subset once more for a fractional λ, the same subset in the target and in every training "
                   "circuit). The regression learns the coefficients, so it needs no value of λ: local folding's "
                   "smaller effective noise scale is absorbed by the fit.",
              when=(("neighbors", ("fold",)),)),
        Param("model", "regression", "choice", "l1", NIL_MODELS,
              help="l1: ‖c‖₁ ≤ γ (Chen et al.); lasso, ridge, elasticnet: penalty α; ols; nnls: c ≥ 0."),
        Param("l1_bound", r"$\ell_1$ bound $\gamma$", "float", 2.0, lo=0.01, hi=1000.0, step=0.1,
              help="Model l1 only. Chen et al.: γ = 2 for Pauli neighbors, 5 for ZNE neighbors."),
        Param("alpha", r"penalty $\alpha$", "float", 1e-3, lo=0.0, hi=1000.0, step=1e-4,
              help="lasso, ridge, elasticnet: the squared error is summed over the training circuits, plus α·Ω(c)."),
        Param("l1_ratio", r"elastic-net mix $\rho$", "float", 0.5, lo=0.0, hi=1.0, step=0.05,
              help="Ω(c) = ρ‖c‖₁ + (1 − ρ)‖c‖₂² (ρ = 1: Lasso, ρ = 0: Ridge)."),
        Param("intercept", "fit an intercept", "bool", False,
              help="Chen et al. fit no intercept. With one, a constant b is added (unpenalized)."),
        Param("n_train", "training circuits (T)", "int", 16, lo=2, hi=100000, step=1),
        Param("ensemble", "training angles", "choice", "design4", NIL_ENSEMBLES,
              help="design4: {0, π/2, π, 3π/2} (Chen et al.); uniform: any angle (non-Clifford); near: CDR's "
                   "near-Clifford angles."),
        Param("target_split", "target shot split", "choice", "optimal", ("optimal", "equal"),
              help="optimal: N_j ∝ |c_j|σ_j after training (circuits with c_j = 0 are not run); equal: N/k shots "
                   "for each of the k neighbor circuits. On IBM hardware only equal can be planned."),
        Param("train_shots", "shots per training circuit", "int", 1024, lo=16, hi=10 ** 8, step=64,
              help="Every training circuit is run once per neighbor: T × k circuits with this many shots each, "
                   "part of the method's total budget."),
    )

    def short(self, p):
        if p.get("neighbors", "pauli") == "fold":
            core = "NIL-ZNE"
        else:
            core = "NIL"
        m = p.get("model", "l1")
        if m == "l1":
            rule = "ℓ1≤%g" % float(p.get("l1_bound", 2.0))
        elif m == "elasticnet":
            rule = "EN α=%g ρ=%g" % (float(p.get("alpha", 1e-3)), float(p.get("l1_ratio", 0.5)))
        elif m in ("lasso", "ridge"):
            rule = "%s α=%g" % (NIL_MODEL_LABEL[m], float(p.get("alpha", 1e-3)))
        else:
            rule = NIL_MODEL_LABEL[m]
        extra = []
        if p.get("neighbors", "pauli") == "pauli":
            if int(p.get("n_neighbors", 12)) != 12:
                extra.append("s=%d" % int(p["n_neighbors"]))
            if p.get("positions", "2q") != "2q":
                extra.append("all gates")
            if not p.get("include_bare", True):
                extra.append("no bare")
        else:
            extra.append("λ=" + ",".join("%g" % x for x in p.get("scales", (1.0, 3.0, 5.0))))
            if p.get("folding", "global") == "local":
                extra.append("local folding")
        if int(p.get("n_train", 16)) != 16:
            extra.append("T=%d" % int(p["n_train"]))
        if p.get("ensemble", "design4") != "design4":
            extra.append(p["ensemble"])
        if p.get("intercept", False):
            extra.append("+b")
        if p.get("target_split", "optimal") != "optimal":
            extra.append("equal split")
        return "%s (%s)" % (core, ", ".join([rule] + extra))

    # ------------------------------------------------------------------ features
    def neighbors(self, ctx, neighbors="pauli", positions="2q", n_neighbors=12, include_bare=True,
                  scales=(1.0, 3.0, 5.0), folding="global") -> list:
        """[(label, transform, facts)] of the feature circuits; transform(spec) builds the neighbor of any circuit with
        the target's structure (the target itself or one of its training circuits). Local fold neighbors use the
        context's seed, as ZNE does: the subset folded once more for a fractional λ depends only on the seed and the
        positions of the two-qubit gates, which training circuits share with the target."""
        spec = ctx.spec
        if neighbors == "fold":
            lams = [float(x) for x in scales]
            if not lams or any(not math.isfinite(x) or x < 1 for x in lams) or len(set(lams)) != len(lams):
                raise ValueError("fold neighbors need distinct noise scales λ ≥ 1, got %s" % (list(scales),))
            if folding == "global":
                return [("λ=%g" % lam, (lambda s, lam=lam: fold_global(s, lam)), dict(kind="fold", scale=lam))
                        for lam in lams]
            if folding == "local":
                seed = int(ctx.seed)
                return [("λ=%g·local" % lam, (lambda s, lam=lam: fold_local(s, lam, seed)),
                         dict(kind="fold", scale=lam, folding="local")) for lam in lams]
            raise ValueError("folding must be 'global' or 'local'")
        if neighbors != "pauli":
            raise ValueError("neighbors must be 'pauli' or 'fold'")
        pool = nil_pool(spec, positions)
        out = [("bare", (lambda s: s), dict(kind="bare"))] if include_bare else []
        if not pool:
            if not out:
                raise ValueError("the circuit has no %s to insert a Pauli after" %
                                 ("two-qubit gate" if positions == "2q" else "gate"))
            return out
        s_ = min(int(n_neighbors), len(pool))
        pick = sorted(_nil_seed(ctx.seed, 1).permutation(len(pool))[:s_].tolist())
        gnum = nil_gate_numbers(spec, positions)
        for t in pick:
            i, q, P = pool[t]
            g = gnum[i]
            lab = "%s(q%d)·g%d" % (P.upper(), q, g)
            out.append((lab, (lambda s, i=i, q=q, P=P: nil_insert(s, i, q, P)),
                        dict(kind="pauli", op=int(i), gate=int(g), qubit=int(q), pauli=P.upper(),
                             gate_name=spec.ops[i].name, gate_qubits=[int(x) for x in spec.ops[i].qubits])))
        return out

    # ------------------------------------------------------------------ estimator
    def min_cost(self, ctx, neighbors="pauli", positions="2q", n_neighbors=12, include_bare=True,
                 scales=(1.0, 3.0, 5.0), folding="global", n_train=16, train_shots=1024, **_) -> dict | None:
        """T training circuits × k feature circuits (neighbors, plus the bare circuit) × training shots."""
        if not non_clifford_slots(ctx.spec):       # NIL falls back to the unmitigated value: nothing is trained
            return dict(training=0, target_circuits=1, note="Clifford circuit: no training circuits")
        try:
            k = len(self.neighbors(ctx, neighbors, positions, n_neighbors, include_bare, scales, folding))
        except Exception:                          # noqa: BLE001 - an invalid setting is reported by run()
            return None
        return dict(training=int(n_train) * k * int(train_shots), target_circuits=1,
                    note="%d training circuits × %d feature circuits × %d shots" % (int(n_train), k,
                                                                                    int(train_shots)))

    def run(self, ctx, neighbors="pauli", positions="2q", n_neighbors=12, include_bare=True, scales=(1.0, 3.0, 5.0),
            folding="global", model="l1", l1_bound=2.0, alpha=1e-3, l1_ratio=0.5, intercept=False, n_train=16,
            ensemble="design4", target_split="optimal", train_shots=1024) -> MethodResult:
        if model not in NIL_MODELS:
            raise ValueError("unknown regression model %r (choose from %s)" % (model, ", ".join(NIL_MODELS)))
        if ensemble not in NIL_ENSEMBLES:
            raise ValueError("unknown training ensemble %r (choose from %s)" % (ensemble, ", ".join(NIL_ENSEMBLES)))
        if target_split not in ("optimal", "equal"):
            raise ValueError("target_split must be 'optimal' or 'equal'")
        if positions not in ("2q", "all"):
            raise ValueError("positions must be '2q' or 'all'")
        T, s_tr = int(n_train), int(train_shots)
        s_tr_default = s_tr                        # the sampler may override it per experiment (1.6)
        if T < 2 or s_tr < 1 or int(n_neighbors) < 1:
            raise ValueError("NIL needs at least 2 training circuits, 1 neighbor and 1 shot per training circuit")
        if not (math.isfinite(float(l1_bound)) and float(l1_bound) > 0) and model == "l1":
            raise ValueError("the ℓ1 bound γ must be positive")
        if not (0.0 <= float(l1_ratio) <= 1.0) or float(alpha) < 0 or not math.isfinite(float(alpha)):
            raise ValueError("need α ≥ 0 and 0 ≤ ρ ≤ 1")
        if getattr(ctx.runner, "backend", None) == "hardware" and target_split == "optimal":
            raise NotSupported("NIL's optimal target split N_j ∝ |c_j|σ_j needs the fitted coefficients before the "
                               "target circuits run, but a hardware plan sends every circuit in one batch. Use "
                               "“target shot split” = equal on hardware.")
        fit_kw = dict(alpha=float(alpha), l1_ratio=float(l1_ratio), l1_bound=float(l1_bound), intercept=bool(intercept))
        # common random numbers: rules that share neighbor and training circuits share the Monte-Carlo training noise
        # of the pipeline's training levels (qemx.phase.cell.training_levels keys its streams by this), so differences
        # between regression rules are measured on the same sampled data
        crn_key = "nil:" + json.dumps(dict(neighbors=neighbors, positions=positions, n_neighbors=int(n_neighbors),
                                           include_bare=bool(include_bare),
                                           scales=[float(x) for x in scales] if neighbors == "fold" else None,
                                           n_train=int(n_train), ensemble=ensemble), sort_keys=True) + (
            "|local" if neighbors == "fold" and folding == "local" else "")       # global: the keys of 1.4.0
        o = ctx.obs.o
        slots = non_clifford_slots(ctx.spec)

        def unmitigated(note, details=None):
            rr = ctx.run()
            return MethodResult(ctx.expval(rr), ctx.var_shot(rr), 1, gamma=ctx.readout_gamma(),
                                **cal_fields(ctx, readout_fun(ctx, rr)),
                                implemented=[("Implemented circuit", display_circuit(ctx))], details=details or {},
                                notes=[note], sampler=lambda N, B, rng, theta=None, s_tr=None: ctx.sample_expvals_theta(
                                    rr, N, B, rng, theta))
        if not slots:
            return unmitigated("The circuit is Clifford: its ideal value is classically computable, and every "
                               "training circuit would equal the target. NIL is not applied.")
        feats = self.neighbors(ctx, neighbors, positions, n_neighbors, include_bare, scales, folding)
        k = len(feats)
        rng_tr = _nil_seed(ctx.seed, 2)
        tr_specs = [nil_training_circuit(ctx.spec, rng_tr, ensemble) for _ in range(T)]
        y = np.array([ctx.ideal_value(sp) for sp in tr_specs], float)
        span = float(np.ptp(o)) or 1.0
        spread = float(np.std(y))
        names = [f[0] for f in feats]
        base_details = dict(names=names, neighbors=[f[2] for f in feats], n_features=k, n_train=T, model=model,
                            ensemble=ensemble, target_split=target_split, y_train=y, spread=spread,
                            spread_needed=0.05 * span)
        if np.ptp(y) < 1e-9:
            warning = ("The training data cannot train the model: all %d training circuits have the same ideal value "
                       "%.4g. NIL is not applied; the value shown is the unmitigated one, with its own uncertainty "
                       "and cost. Use another training ensemble, more training circuits, or another method."
                       % (T, y[0]))
            return unmitigated(warning, dict(base_details, degenerate=True, warning=warning, coef=np.zeros(k),
                                             intercept=0.0, rank=0, nonzero=0, l1=0.0, solver_ok=True))
        # ---- noisy runs: T training circuits × k neighbors, then the k neighbors of the target
        tr_runs = [[ctx.run(f[1](sp), role="training") for f in feats] for sp in tr_specs]
        tg_runs = [ctx.run(f[1](ctx.spec)) for f in feats]
        tr_meas = [[ctx.measured(r) for r in row] for row in tr_runs]
        tg_meas = [ctx.measured(r) for r in tg_runs]

        def feat(f, mit=None):
            return float(o @ (ctx.mitigate(f) if mit is None else mit(f)))
        X = np.array([[feat(f) for f in row] for row in tr_meas])
        xt = np.array([feat(f) for f in tg_meas])
        coef, b0, info = nil_fit(X, y, model, **fit_kw)
        mean = float(b0 + coef @ xt)
        sig = np.sqrt(np.maximum([ctx.var_shot(r) for r in tg_runs], 0.0))
        tr_var = np.array([[max(ctx.var_shot(r), 0.0) for r in row] for row in tr_runs])

        def target_var(c):
            """Per-shot target variance of the estimator with coefficients c under the split rule."""
            if target_split == "equal":
                return float(k * np.sum((c * sig) ** 2))
            return float(np.sum(np.abs(c) * sig)) ** 2

        var1 = target_var(coef)
        a = np.abs(coef) * sig
        if target_split == "equal":
            alloc = np.full(k, 1.0 / k)
            circuits = k
        else:
            alloc = a / a.sum() if a.sum() > 0 else np.where(coef != 0, 1.0, 0.0) / max(int(np.sum(coef != 0)), 1)
            circuits = max(int(np.sum(coef != 0)), 1)
        # ---- diagnostics: feature directions resolved above the training shot noise
        Xd = X - X.mean(0) if intercept else X
        noise = float(np.sqrt(np.max(tr_var) / s_tr))
        sv = np.linalg.svd(Xd, compute_uv=False) if Xd.size else np.zeros(0)
        rank = int(np.sum(sv > noise * (math.sqrt(T) + math.sqrt(k))))
        twins = _nil_twins(X)
        notes = []
        warning = ""
        if spread < 0.05 * span:
            warning = ("Weak training signal: the ideal training values spread by only %.3g (%.3g would be 5 %% of the "
                       "observable's range), so the coefficients are poorly determined." % (spread, 0.05 * span))
        if not info["ok"]:
            notes.append("The regression's optimality could not be certified to the solver tolerance (strongly "
                         "collinear features); the best solution found is used.")
        if twins:
            notes.append("%d group%s of neighbor circuits give identical values on every training circuit (%s); the "
                         "fit cannot tell them apart and shares their weight equally." % (
                             len(twins), "" if len(twins) == 1 else "s",
                             "; ".join(" = ".join(names[j] for j in g) for g in twins[:4]) + (" …" if len(twins) > 4
                                                                                              else "")))
        if rank < k:
            notes.append("The training data resolve %d of %d feature directions above the shot noise of %d shots "
                         "per training circuit." % (rank, k, s_tr))
        notes.append("Training: %d circuits × %d neighbor circuits × %d shots = %s shots; target: %d neighbor "
                     "circuit%s with c_j ≠ 0, γ = ‖c‖₁ = %.3g." % (T, k, s_tr, f"{T * k * s_tr:,}",
                                                                   int(np.sum(coef != 0)),
                                                                   "" if int(np.sum(coef != 0)) == 1 else "s",
                                                                   float(np.abs(coef).sum())))

        F_tr = np.array([f for row in tr_meas for f in row])            # (T·k, 2ⁿ) measured distributions
        F_tg = np.array(tg_meas)
        dim = F_tr.shape[1]

        def refit(mit):
            """The whole estimate with another readout mitigation (training features refitted too). The
            calibration's mitigation maps are linear in the distribution, so o·M f = (Mᵀo)·f: Mᵀo costs 2ⁿ
            applications of M, cheaper than one per circuit while 2ⁿ ≤ T·k + k."""
            if _NIL_VECTOR_REFIT and dim <= F_tr.shape[0] + F_tg.shape[0]:
                o_m = np.array([float(o @ mit(e)) for e in np.eye(dim)])
                Xm = (F_tr @ o_m).reshape(T, k)
                xm = F_tg @ o_m
            else:
                Xm = np.array([[feat(f, mit) for f in row] for row in tr_meas])
                xm = np.array([feat(f, mit) for f in tg_meas])
            cm, bm, _ = nil_fit(Xm, y, model, warm=(coef, info), **fit_kw)
            return float(bm + xm @ cm)

        linear_ro = ctx.recipe.readout in ("none", "tensored", "trex")
        o_eff = ctx.o_eff() if linear_ro else None

        def probs(rr):
            p = np.clip(ctx.measured(rr), 0, None)
            return p / p.sum()
        tr_p = [[probs(r) for r in row] for row in tr_runs]
        tg_p = [probs(r) for r in tg_runs]
        P_tg = np.array(tg_p)

        def values(counts, n, oe=None):
            """Mitigated ⟨O⟩ of sampled count rows (shots n per row; rows with n = 0 give 0); ``oe``: per-row
            per-shot values (calibration draws), else this recipe's mitigation."""
            n = np.maximum(np.asarray(n, float), 1.0)
            f = np.asarray(counts, float) / (n[:, None] if n.ndim else n)
            if oe is not None:
                return np.einsum("rd,rd->r", f, oe)
            if linear_ro:
                return f @ o_eff
            return np.array([float(o @ ctx.mitigate(fr)) for fr in f])

        def theta_rows(theta, R):
            """(per-row per-shot values Mᵀ(θ̂_r)o or None, rows with a non-invertible θ̂)."""
            th = ctx.theta_rows(theta, R)
            if th is None:
                return None, np.zeros(int(R), bool)
            bad = np.array([not ctx.theta_invertible(t) for t in th])
            oe = np.array([ctx.o_eff_theta(t if not b_ else ctx.cal_theta()) for t, b_ in zip(th, bad)])
            return oe, bad

        def sample_training(shots, R, rng, oe=None):
            """(R, T, k) training features with ``shots`` shots per training circuit."""
            out = np.empty((int(R), T, k))
            for t in range(T):
                for j in range(k):
                    out[:, t, j] = values(rng.multinomial(int(shots), tr_p[t][j], size=int(R)), int(shots), oe)
            return out

        def split(c, N):
            """Integer target shots per neighbor circuit for coefficients c (None: N is below the circuits needed)."""
            if target_split == "equal":
                if N < k:
                    return None
                n = np.full(k, N // k, dtype=np.int64)
                n[: N - int(n.sum())] += 1
                return n
            act = np.flatnonzero(c != 0)
            n = np.zeros(k, dtype=np.int64)
            if not len(act):
                return n
            if N < len(act):
                return None
            n[act] = optimal_integer_split(np.abs(c[act]) * sig[act], int(N))[1]
            return n

        def sampler(N, B, rng, theta=None, s_tr=None):
            """B complete experiments with N target shots: training features sampled (train_shots per training
            circuit, or ``s_tr`` for a budget-dependent training allocation, 1.6), the rule refitted, the N target
            shots split by the fitted coefficients, the target neighbor circuits sampled. With ``theta`` (B
            calibration draws θ̂) every circuit of experiment b is mitigated with θ̂_b."""
            oe, bad = theta_rows(theta, B)
            Xs = sample_training(s_tr_default if s_tr is None else int(s_tr), B, rng, oe)
            C = np.empty((int(B), k))
            b_ = np.empty(int(B))
            for bi in range(int(B)):
                if bi % 16 == 0:
                    check()                                 # cooperative time limit (qemx.resources.deadline)
                C[bi], b_[bi], _ = nil_fit(Xs[bi], y, model, warm=(coef, info), **fit_kw)
            Nj = np.zeros((int(B), k), dtype=np.int64)
            short = np.zeros(int(B), bool)
            for bi in range(int(B)):
                n = split(C[bi], int(N))
                if n is None:
                    short[bi] = True
                else:
                    Nj[bi] = n
            est = b_.copy()
            for j in range(k):
                counts = rng.multinomial(Nj[:, j], tg_p[j])
                xj = values(counts, Nj[:, j], oe)
                est += np.where(C[:, j] != 0, C[:, j] * xj, 0.0)
            est[short | bad] = np.nan
            return est

        def train_resampler(shots, R, rng, theta=None):
            """(estimates, per-shot target variances): R refits on training features sampled with ``shots`` shots
            per training circuit, each applied to the exact target features, with the per-shot target variance of
            that fit under the split rule. With N target shots, Var = E_c[v(c)]/N + Var_c[estimate] (law of total
            variance; exact for linear readout mitigation and the continuous split). With ``theta`` (R calibration
            draws θ̂) the training and target features of replicate r are mitigated with θ̂_r; the random draws are
            those of the call without it, so the two calls pair up."""
            oe, bad = theta_rows(theta, R)
            Xs = sample_training(shots, R, rng, oe)
            xts = None if oe is None else P_tg @ oe.T                       # (k, R) exact target features
            est, v1 = np.empty(int(R)), np.empty(int(R))
            for bi in range(int(R)):
                if bi % 16 == 0:
                    check()
                cb, bb, _ = nil_fit(Xs[bi], y, model, warm=(coef, info), **fit_kw)
                est[bi] = bb + (xt if xts is None else xts[:, bi]) @ cb
                v1[bi] = target_var(cb)
            est[bad] = np.nan
            return est, v1

        # ---- training-shot variance and calibration coefficient of the protocol as run, by Monte Carlo (the fit is
        # piecewise smooth at best; with near-collinear features first-order values are meaningless)
        from .learning import MC_REPS, CAL_REF_SHOTS, calibration_w
        from ..stats import moments_wanted, protocol_contract, reference_moments, training_moments
        seed64 = int(ctx.seed) % (2 ** 63)
        # protocol moments at s_tr (qemx/stats.py): mean of the refitted estimates, their variance u and the mean
        # conditional per-shot target variance; the fit to exact training features is kept as the reference
        want = moments_wanted(ctx)
        mom = training_moments(train_resampler, s_tr, np.random.default_rng([seed64, 0x4E49, 1])) if want \
            else reference_moments(mean, var1, s_tr)
        u_train = mom.u
        contract = protocol_contract(ctx, mom, mean, var1, notes)
        mean, var1 = contract.pop("mean"), contract.pop("var1")
        if want:
            notes.append("Shot noise of the training data adds a variance %.3g that more target shots cannot remove "
                         "(σ floor %.3g; %d Monte-Carlo refits)." % (u_train, math.sqrt(max(u_train, 0.0)),
                                                                      mom.repeats))
        else:
            contract["flags"] = tuple(contract["flags"]) + ("moments-not-estimated",)
        cal = cal_fields(ctx, refit)
        if cal and want:
            cal["cal_w"] = calibration_w(ctx, train_resampler, s_tr, MC_REPS, seed64)
            contract["flags"] = tuple(contract["flags"]) + ("calibration-mean-first-order",)
        ex_nb = next((f for f in feats if f[2]["kind"] != "bare"), feats[0])
        implemented = [("Target circuit", display_circuit(ctx)),
                       ("Neighbor circuit “%s” (1 of %d)" % (ex_nb[0], k), display_circuit(ctx, ex_nb[1](ctx.spec))),
                       ("Training circuit 1 of %d (%s angles), same neighbor" % (T, ensemble),
                        display_circuit(ctx, ex_nb[1](tr_specs[0])))]
        details = dict(base_details, degenerate=False, warning=warning, coef=coef, intercept=b0, X=X, xt=xt,
                       sigma=sig, alloc=[float(x) for x in alloc], rank=rank, nonzero=int(np.sum(coef != 0)),
                       l1=float(np.abs(coef).sum()), lam=float(info["lam"]), eps=float(info["eps"]),
                       solver_ok=bool(info["ok"]), twins=[[names[j] for j in g] for g in twins],
                       fitted=X @ coef + b0, x_train=X[:, 0], x_target=float(xt[0]), crn_key=crn_key,
                       cal_w_ref_shots=CAL_REF_SHOTS)
        return MethodResult(
            **contract,
            mean=mean, var1=var1, circuits=circuits, fixed_shots=T * k * s_tr, train_circuits=T * k,
            var_fixed=float(u_train), gamma=float(np.abs(coef).sum()) * ctx.readout_gamma(),
            **cal, joint_calibration=True, implemented=implemented, details=details, notes=notes,
            sampler=sampler, train_resampler=train_resampler)

    def hardware_supported(self, params) -> tuple:
        """The optimal target split needs the fitted coefficients before the target runs; a hardware plan sends every
        circuit in one batch, so only the equal split can be planned."""
        if dict(params or {}).get("target_split", "optimal") == "optimal":
            return False, ("NIL's optimal target split N_j ∝ |c_j|σ_j needs the fitted coefficients before the target "
                           "circuits run, but a hardware plan sends every circuit in one batch. Use “target shot "
                           "split” = equal on hardware.")
        return True, ""


def _nil_twins(X, rtol: float = 1e-12) -> list:
    """Groups of columns that are identical on every training circuit (to rtol of the design's scale)."""
    X = np.asarray(X, float)
    k = X.shape[1]
    if k < 2 or not X.size:
        return []
    scale = max(float(np.max(np.abs(X))), 1.0)
    close = np.max(np.abs(X[:, :, None] - X[:, None, :]), axis=0) <= rtol * scale     # (k, k), one pass
    np.fill_diagonal(close, False)
    if not close.any():
        return []
    seen, out = set(), []
    for i in range(k):                                     # greedy, in column order (as before)
        if i in seen:
            continue
        g = [i] + [int(j) for j in np.flatnonzero(close[i]) if j > i and int(j) not in seen]
        if len(g) > 1:
            out.append(g)
            seen.update(g)
    return out
