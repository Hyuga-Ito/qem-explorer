r"""The equal-budget ledger: RMSE of every method at the same total number of shots B, with an exact integer split.

Model (one method, one cell)
----------------------------
.. math::
    B = N_t + T\,s_{\rm tr} + c\,s,\qquad
    \mathrm{RMSE}(B) = \sqrt{\,b(s_{\rm tr})^2 + \frac{v}{N_t} + \frac{w}{s} + u(s_{\rm tr})\,}

* :math:`N_t` target shots, split among the method's :math:`m` target circuits (ZNE: :math:`N_j\propto|\eta_j|\sigma_j`,
  so :math:`v=(\sum_j|\eta_j|\sigma_j)^2`); every target circuit needs at least one shot (``per_circuit`` shots: PEC
  with s shots per sampled circuit needs s), and post-selection needs :math:`\lceil 5/P_{\rm accept}\rceil` target
  shots (``accept``): the feasibility contract of :mod:`qemx.budget` (:func:`qemx.budget.min_total_budget`), which
  :func:`min_budget` and every allocation here use, so the phase ledger, the interactive cost model and the hardware
  planner agree on feasibility. NIL's split
  :math:`N_j\propto|c_j|\sigma_j` depends on the fitted coefficients, so its rows carry no per-circuit ``a``: the
  ledger uses the pooled :math:`v(s_{\rm tr})=E_c[(\sum_j|c_j|\sigma_j)^2]` of its training levels (continuous split;
  the integer split that the Monte-Carlo validation draws is larger by a term of second order in :math:`1/N_t`);
* :math:`F=T\,s_{\rm tr}` training shots (CDR/FR: :math:`T` training circuits; NIL: :math:`T` training circuits × k neighbor circuits; :math:`s_{\rm tr}` shots each). Their
  shot noise adds the variance :math:`u(s_{\rm tr})` and, because a regression on noisy features is biased
  (errors-in-variables attenuation), shifts the bias to :math:`b(s_{\rm tr})`; the fitted coefficients being random,
  the target's per-shot variance becomes :math:`v(s_{\rm tr})=E_c[v(c)]`. For the learning methods the pipeline
  measures all three by Monte Carlo at pre-registered **training levels** (``Cost.levels``, e.g. 64 … 16384 shots),
  so that :math:`\mathrm{Var}=v(s_{\rm tr})/N_t+u(s_{\rm tr})` is exact (law of total variance; for linear readout
  mitigation — none, tensored, TREX — and to first order in :math:`1/N_t` for M3).
  Without levels the first-order model :math:`b(s_{\rm tr})=b`, :math:`u=u_1/s_{\rm tr}` is used;
* :math:`c` readout-calibration circuits with :math:`s` shots each: :math:`w/s` (first order). For the methods
  whose calibration acts jointly on training and target data (CDR/FR/NIL: the regression is refitted on features
  mitigated with the same estimated calibration), :math:`w` is measured by Monte Carlo — paired refits with and
  without a sampled calibration — at every training level, :math:`w(s_{\rm tr})`.

Policies (:class:`Policy`)
--------------------------
``fixed``  the protocol as run: :math:`s_{\rm tr}=s_{\rm tr,0}` (the manifest's value). The calibration shots are
           optimal, :math:`s^*=R\kappa/(1+c\kappa)`, :math:`\kappa=\sqrt{w/(cv)}`, :math:`R=B-F`;
``opt``    the training shots are also chosen per budget: the best level (with levels), or the continuous optimum
           :math:`\mathrm{Var}^*=(\sqrt v+\sqrt{Tu_1}+\sqrt{cw})^2/B` under the lower bounds (first-order model).

``s_min`` (shots per calibration circuit) and ``s_tr_min`` are lower bounds; a calibration with very few shots
cannot be inverted reliably, so the pipeline uses ``s_min = 32`` (:data:`PIPELINE_FIXED`, :data:`PIPELINE_OPT`).

Integer split
-------------
With ``integer=True`` every count is an integer and :math:`N_t+Ts_{\rm tr}+cs=B` holds **exactly** (asserted).
ZNE's per-circuit shots come from :func:`qemx.budget.optimal_integer_split` (exact). Given :math:`s_{\rm tr}`, the
variance is a discretely convex function of the integer :math:`s` — the optimal integer split's variance
:math:`V(N)` is discretely convex in :math:`N` (greedy marginal allocation of a separable convex objective), its
restriction to :math:`N=R-cs` stays convex, and :math:`w/s` is convex — so a descent over unit steps from
floor/ceil of the continuous :math:`s^*` reaches the integer optimum (rounding alone does not: with several target
circuits the continuous :math:`s^*` refers to the pooled variance, not to :math:`V`). The first-order ``opt`` path
(no levels) searches the integer :math:`s_{\rm tr}` around the continuous optimum, with the exact best :math:`s`
for each. The optimal integer RMSE is non-increasing in B wherever the feasible allocations of B are contained in
those of B + 1 shifted by one target shot — true for every policy here, since a target shot can always be added — so
:func:`budget_for` bisects over integers.

The model is first order in the target and calibration shot noise; ``qemx.phase.validate`` checks it by Monte Carlo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np

from ..budget import MIN_ACCEPTED, Feasibility, min_target_shots, min_total_budget, optimal_integer_split


@dataclass(frozen=True)
class Policy:
    name: str = "fixed"              # fixed | opt
    integer: bool = True
    s_min: int = 1                   # shots per calibration circuit, lower bound
    s_tr_min: int = 1                # shots per training circuit, lower bound (opt, first-order path)

    def __post_init__(self):
        if self.name not in ("fixed", "opt"):
            raise ValueError("policy must be 'fixed' or 'opt'")
        if int(self.s_min) < 1 or int(self.s_tr_min) < 1:
            raise ValueError("lower bounds must be >= 1")


PIPELINE_FIXED = Policy("fixed", True, 32, 32)
PIPELINE_OPT = Policy("opt", True, 32, 32)


def _pol(policy) -> Policy:
    return policy if isinstance(policy, Policy) else Policy(str(policy))


def _rule(pol: Policy) -> Feasibility:
    """The policy's lower bounds as the shared feasibility rule (qemx.budget.Feasibility)."""
    return Feasibility(MIN_ACCEPTED, int(pol.s_min), int(pol.s_tr_min))


def n_min(cost: "Cost", policy="fixed") -> int:
    """Fewest target shots that yield an estimate (qemx.budget.min_target_shots): ``per_circuit`` per target circuit
    and, with post-selection, ⌈5/P_accept⌉."""
    return min_target_shots(cost.m, cost.accept, cost.per_circuit, _rule(_pol(policy)))


@dataclass(frozen=True)
class Cost:
    method: str
    b: float                         # bias at infinite shots (target, training, calibration): mean − ideal
    v: float                         # per-shot variance on the target circuits (optimal split among them)
    w: float = 0.0                   # calibration: Var_cal = w / s
    c: int = 0                       # calibration circuits
    T: int = 0                       # training circuits
    s_tr: int = 0                    # shots per training circuit of the protocol as run (fixed policy)
    u1: float = 0.0                  # first-order training variance: u = u1 / s_tr
    u_extra: float = 0.0             # constant excess variance (added at every budget)
    m: int = 1                       # target circuits that each need at least one shot
    a: tuple = ()                    # per-target-circuit |η_j| σ_j (ZNE); empty = one pooled target
    levels: tuple = ()               # ((s_tr, b, u[, v[, w]]), …) at s_tr training shots, measured by Monte Carlo
    accept: float = 1.0              # post-selection acceptance P_accept (needs ⌈5/P_accept⌉ target shots)
    per_circuit: int = 1             # shots each target circuit needs at least (PEC: s shots per sampled circuit)
    # budget-dependent moments of a target-nonlinear estimator (qemx.stats.moment_grid; exponential ZNE, symmetry
    # verification): below the grid's validated first-order range the ledger uses its Monte-Carlo mean and variance
    target_grid: dict | None = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        """Every number finite and of the right sign: a NaN would make every comparison False and turn into a wrong
        allocation, budget or winner without any error."""
        def fin(name, x, lo=None):
            if x is None or not math.isfinite(float(x)) or (lo is not None and float(x) < lo):
                raise ValueError("Cost %s: %s = %r is not a finite number%s" % (
                    self.method, name, x, "" if lo is None else " >= %g" % lo))
        fin("b", self.b)
        for nm in ("v", "w", "u1", "u_extra"):
            fin(nm, getattr(self, nm), 0.0)
        fin("accept", self.accept, 0.0)
        if float(self.accept) > 1.0:
            raise ValueError("Cost %s: accept = %r is not a probability" % (self.method, self.accept))
        for nm, lo in (("c", 0), ("T", 0), ("s_tr", 0), ("m", 1), ("per_circuit", 1)):
            x = getattr(self, nm)
            if int(x) != x or x < lo:
                raise ValueError("Cost %s: %s = %r must be an integer >= %d" % (self.method, nm, x, lo))
        for x in self.a:
            fin("a", x, 0.0)
        seen = set()
        for lv in self.levels:
            if not 3 <= len(lv) <= 5 or int(lv[0]) != lv[0] or lv[0] < 1 or int(lv[0]) in seen:
                raise ValueError("Cost %s: bad level %r (s_tr, b, u[, v[, w]] with distinct integer s_tr >= 1)"
                                 % (self.method, lv))
            seen.add(int(lv[0]))
            fin("level b", lv[1])
            for x in lv[2:]:
                fin("level u/v/w", x, 0.0)

    @property
    def F(self) -> int:
        return int(self.T) * int(self.s_tr)

    def at_level(self, s_tr: int) -> tuple:
        """(bias, training variance, per-shot target variance, calibration coefficient) at s_tr shots per training
        circuit. A level may carry its own v (the target variance averaged over the training noise, E_c[var1(c)])
        and w (the calibration coefficient of the refitted protocol); otherwise those of the method are used."""
        for lv in self.levels:
            if int(lv[0]) == int(s_tr):
                return (float(lv[1]), float(lv[2]) + self.u_extra,
                        float(lv[3]) if len(lv) > 3 else _v_pooled(self), float(lv[4]) if len(lv) > 4 else self.w)
        if self.levels:
            raise KeyError("no Monte-Carlo level at s_tr = %d (levels: %s)" % (s_tr, [l[0] for l in self.levels]))
        return (self.b, (self.u1 / s_tr if (self.T > 0 and s_tr > 0) else 0.0) + self.u_extra, _v_pooled(self),
                self.w)

    @property
    def u(self) -> float:
        """Training variance of the protocol as run."""
        return self.at_level(self.s_tr)[1] if self.T > 0 else self.u_extra

    def to_json(self) -> dict:
        d = asdict(self)
        if d.get("target_grid") is None:
            d.pop("target_grid", None)
        d["a"] = list(self.a)
        d["levels"] = [list(l) for l in self.levels]
        return d

    @staticmethod
    def from_json(d: dict) -> "Cost":
        d = dict(d)
        d["a"] = tuple(d.get("a") or ())
        d["levels"] = tuple(tuple(l) for l in (d.get("levels") or ()))
        return Cost(**{k: d[k] for k in Cost.__dataclass_fields__ if k in d})


@dataclass
class Alloc:
    B: float
    feasible: bool
    N: float = 0                     # target shots
    s: float = 0                     # shots per calibration circuit
    s_tr: float = 0                  # shots per training circuit
    N_j: tuple = ()                  # per target circuit (multi-circuit methods)
    b: float = math.nan              # bias at this allocation (depends on s_tr for the learning methods)
    var: float = math.inf            # total variance
    var_target: float = math.inf
    var_cal: float = 0.0
    var_train: float = 0.0
    source: str = "first-order"      # where b and var_target come from (target-nonlinear rows: qemx.stats.grid_at)
    abstention_rate: float = 0.0     # share of experiments without an estimate ("abstain" failure policy)

    @property
    def mse(self) -> float:
        return self.b ** 2 + self.var if self.feasible else math.inf


def _tr_choices(cost: Cost, pol: Policy) -> list:
    """Training shot levels the policy may use ([] = the continuous first-order path)."""
    if cost.T <= 0:
        return [0]
    if pol.name == "fixed":
        return [int(cost.s_tr)]
    if cost.levels:
        return sorted(int(l[0]) for l in cost.levels)
    return []


def min_budget(cost: Cost, policy="fixed") -> int:
    """Smallest B that yields an estimate (every circuit at its minimum number of shots): the shared feasibility
    function qemx.budget.min_total_budget with the policy's lower bounds and the smallest training level it may use."""
    pol = _pol(policy)
    tr = _tr_choices(cost, pol)
    t_min = min(tr) if tr else int(pol.s_tr_min)
    return int(min_total_budget(cost.m, cost.accept, cost.per_circuit, int(cost.T) * t_min, int(cost.c), _rule(pol)))


def _target_var(cost: Cost, N: float, integer: bool, v: float | None = None) -> tuple:
    if N < cost.m or N <= 0 or N < n_min(cost):
        return math.inf, ()
    if not cost.a or len(cost.a) <= 1:
        return (cost.v if v is None else v) / N, ((int(N),) if integer else ())
    a = np.asarray(cost.a, float)
    if not integer:
        return float(np.sum(a)) ** 2 / N, ()
    return optimal_integer_split(a, int(N))


def _v_pooled(cost: Cost) -> float:
    return float(np.sum(cost.a)) ** 2 if cost.a and len(cost.a) > 1 else float(cost.v)


def _best_s(cost: Cost, B, R, s_tr, b, u, v, w, pol: Policy, v_level: bool) -> Alloc:
    """Optimal calibration shots s (and target shots N = R − c·s) when R shots remain after training. Integer: from
    floor/ceil of the continuous optimum, unit steps while the variance falls (it is discretely convex in s, see the
    module docstring), so the result is the integer optimum."""
    vt_arg = v if v_level else None

    def at(s):
        N = R - cost.c * s
        vt, Nj = _target_var(cost, N, pol.integer, vt_arg)
        vc = w / s if cost.c > 0 else 0.0
        return Alloc(B, True, N, s, s_tr, Nj, b, vt + vc + u, vt, vc, u)
    nmin = n_min(cost, pol)
    if cost.c <= 0:
        if R < nmin:
            return Alloc(B, False)
        return at(0)
    smin = float(pol.s_min)
    smax = (R - nmin) / cost.c
    if pol.integer:
        smax = math.floor(smax)
    if smax < smin:
        return Alloc(B, False)
    if w <= 0:
        s_star = smin
    elif v <= 0:
        s_star = smax
    else:
        k = math.sqrt(w / (cost.c * v))
        s_star = R * k / (1 + cost.c * k)
    s_star = min(max(s_star, smin), smax)
    if not pol.integer:
        return at(s_star)
    lo_, hi_ = int(math.ceil(smin)), int(smax)
    cands = {min(max(int(math.floor(s_star)), lo_), hi_), min(max(int(math.ceil(s_star)), lo_), hi_)}
    best = min((at(s_) for s_ in sorted(cands)), key=lambda a_: a_.var)
    for step in (-1, 1):                               # descend; convexity: one direction can improve at most
        while lo_ <= best.s + step <= hi_:
            nxt = at(best.s + step)
            if not nxt.var < best.var:
                break
            best = nxt
    return best


def _given_tr(cost: Cost, B, s_tr: int, pol: Policy) -> Alloc:
    """Optimal (N, s) when the training shots are fixed at s_tr per circuit."""
    if cost.T > 0:
        b, u, v, w = cost.at_level(s_tr)
    else:
        b, u, v, w = cost.b, cost.u_extra, _v_pooled(cost), cost.w
    R = B - (cost.T * s_tr if cost.T > 0 else 0)
    return _best_s(cost, B, R, s_tr, b, u, v, w, pol, v_level=cost.T > 0)


def _continuous_opt(cost: Cost, B, pol: Policy) -> Alloc:
    """First-order opt path (no levels): s_tr continuous with u = u1/s_tr, KKT water-filling over three pools."""
    nmin = n_min(cost, pol)
    pools = [("N", _v_pooled(cost), 1.0, float(nmin))]
    if cost.c > 0:
        pools.append(("s", cost.w, float(cost.c), float(pol.s_min)))
    pools.append(("t", cost.u1, float(cost.T), float(pol.s_tr_min)))
    if B < sum(cw * L for _, _, cw, L in pools):
        return Alloc(B, False)

    def x_of(mu):
        return {nm: (max(L, math.sqrt(a / (mu * cw))) if a > 0 else L) for nm, a, cw, L in pools}

    def spend(mu):
        x = x_of(mu)
        return sum(cw * x[nm] for nm, _, cw, _ in pools)
    if any(a > 0 for _, a, _, _ in pools):
        lo, hi = 1e-300, 1e300                      # spend() decreases in mu
        for _ in range(4000):
            mid = math.sqrt(lo * hi)
            if spend(mid) > B:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1 + 1e-15:
                break
        x = x_of(hi)
    else:
        x = {nm: L for nm, _, _, L in pools}
    s_c, t_c = x.get("s", 0.0), x["t"]
    if not pol.integer:
        N = B - (cost.c * s_c if cost.c > 0 else 0) - cost.T * t_c
        if N < nmin or (cost.c > 0 and s_c < pol.s_min) or t_c < pol.s_tr_min:
            return Alloc(B, False)
        vt, _ = _target_var(cost, N, False)
        vc = cost.w / s_c if cost.c > 0 else 0.0
        vtr = cost.u1 / t_c + cost.u_extra
        return Alloc(B, True, N, s_c if cost.c > 0 else 0, t_c, (), cost.b, vt + vc + vtr, vt, vc, vtr)

    def at_t(t):
        """The exact integer optimum over s with t training shots per circuit (None if infeasible)."""
        if t < pol.s_tr_min:
            return None
        a_ = _best_s(cost, B, B - cost.T * t, t, cost.b, cost.u1 / t + cost.u_extra, _v_pooled(cost), cost.w, pol,
                     v_level=False)
        return a_ if a_.feasible else None
    t_hi = (B - nmin - (cost.c * pol.s_min if cost.c > 0 else 0)) // cost.T
    if t_hi < pol.s_tr_min:
        return Alloc(B, False)
    t0 = min(max(int(round(t_c)), int(pol.s_tr_min)), int(t_hi))
    best = None
    for t in range(max(int(pol.s_tr_min), t0 - 3), min(int(t_hi), t0 + 3) + 1):
        a_ = at_t(t)
        if a_ is not None and (best is None or a_.var < best.var):
            best = a_
    if best is None:
        return Alloc(B, False)
    for step in (-1, 1):                               # min over s of a convex function: unimodal in t
        while True:
            nxt = at_t(best.s_tr + step)
            if nxt is None or not nxt.var < best.var:
                break
            best = nxt
    return best


def allocate(cost: Cost, B, policy="fixed") -> Alloc:
    """The optimal split of B shots under the policy. Integer allocations satisfy the budget identity exactly."""
    pol = _pol(policy)
    B = int(B) if pol.integer else float(B)
    if B < min_budget(cost, pol):
        return Alloc(B, False)
    choices = _tr_choices(cost, pol)
    if choices:
        best = None
        for t in choices:
            a = _given_tr(cost, B, t, pol)
            if a.feasible and (best is None or a.mse < best.mse):
                best = a
        out = best if best is not None else Alloc(B, False)
    else:
        out = _continuous_opt(cost, B, pol)
    if out.feasible and pol.integer:
        total = out.N + (cost.c * out.s if cost.c > 0 else 0) + (cost.T * out.s_tr if cost.T > 0 else 0)
        if total != B or any(int(x) != x for x in (out.N, out.s, out.s_tr)):
            raise AssertionError("budget identity violated: %r + %d·%r + %d·%r != %d" % (out.N, cost.T, out.s_tr,
                                                                                      cost.c, out.s, B))
        if out.N_j and sum(out.N_j) != out.N:
            raise AssertionError("per-circuit target shots do not add up")
        out.N, out.s, out.s_tr = int(out.N), int(out.s), int(out.s_tr)
    return _target_nonlinear(cost, out)


def _target_nonlinear(cost: Cost, a: Alloc) -> Alloc:
    """An allocation of a target-nonlinear estimator with its budget-dependent moments (review F01): at N target
    shots below the grid's validated first-order range, the bias is the protocol's Monte-Carlo mean minus the ideal
    value and the target variance is the protocol's (``source`` says where each number came from). The per-circuit
    split is the method's own (the grid was simulated with it); the ledger does not re-optimize it for these rows.
    No usable moments at N (every experiment abstained, or N below the grid): infeasible."""
    g = cost.target_grid
    if not g or not a.feasible:
        return a
    from ..stats import grid_at
    m = grid_at(g, float(a.N))
    a.source = m.get("source", "")
    if str(a.source).startswith("first-order (validated)"):
        return a
    if not (math.isfinite(m.get("mean", math.nan)) and math.isfinite(m.get("var", math.nan))):
        return Alloc(a.B, False)
    ideal = float(g["reference_mean"]) - float(cost.b)
    a.b = float(m["mean"]) - ideal
    a.var = a.var - a.var_target + float(m["var"])
    a.var_target = float(m["var"])
    a.abstention_rate = float(m.get("abstention_rate", 0.0) or 0.0)
    return a


def rmse(cost: Cost, B, policy="fixed") -> float:
    a = allocate(cost, B, policy)
    return math.sqrt(a.mse) if a.feasible and math.isfinite(a.mse) else math.inf


def rmse_curve(cost: Cost, budgets, policy="fixed") -> np.ndarray:
    return np.array([rmse(cost, B, policy) for B in np.atleast_1d(budgets)])


def floor_mse(cost: Cost, policy="fixed") -> float:
    """lim_{B→∞} MSE: what more target and calibration shots cannot remove."""
    pol = _pol(policy)
    if cost.T <= 0:
        return cost.b ** 2 + cost.u_extra
    ch = _tr_choices(cost, pol)
    if ch:
        return min(cost.at_level(t)[0] ** 2 + cost.at_level(t)[1] for t in ch)     # v/N_t, w/s → 0
    return cost.b ** 2 + cost.u_extra                  # first-order opt: training shots grow with B


def budget_for(cost: Cost, eps: float, policy="fixed", hi_max: float = 1e15) -> float:
    """Smallest integer B with RMSE(B) <= eps; inf when the floor reaches eps."""
    pol = _pol(policy)
    if eps * eps <= floor_mse(cost, pol):
        return math.inf
    lo = min_budget(cost, pol)
    if rmse(cost, lo, pol) <= eps:
        return float(lo)
    hi = max(2 * lo, 2)
    while rmse(cost, hi, pol) > eps:
        hi *= 2
        if hi > hi_max:
            return math.inf
    while hi - lo > 1:                                 # the integer RMSE is non-increasing in B
        mid = (lo + hi) // 2
        if rmse(cost, mid, pol) <= eps:
            hi = mid
        else:
            lo = mid
    return float(hi)


def winners(costs: list, budgets, policy="fixed") -> list:
    """Per budget: (B, winning method, its RMSE, runner-up, its RMSE). Ties go to the method listed first."""
    out = []
    for B in np.atleast_1d(budgets):
        r = sorted(((rmse(c, B, policy), i, c.method) for i, c in enumerate(costs)), key=lambda t: (t[0], t[1]))
        out.append((float(B), r[0][2], r[0][0], r[1][2] if len(r) > 1 else None, r[1][0] if len(r) > 1 else None))
    return out


def crossovers(c1: Cost, c2: Cost, policy="fixed", lo: float = 1e3, hi: float = 1e8, points: int = 401) -> list:
    """Integer budgets in [lo, hi] at which the better of two methods changes: [(B, method strictly better from B
    on)]. Located on a geometric grid, then exactly by integer bisection. An infeasible method (RMSE = inf) is worse
    than a feasible one; equal RMSEs are a tie: a change of sign that passes through ties (on grid points or
    between them) is reported at the first budget where the other method is strictly better. Changes between grid
    points that reverse twice are not resolved (use more points)."""
    def sgn(B):
        r1, r2 = rmse(c1, B, policy), rmse(c2, B, policy)
        if r1 == r2:
            return 0
        return -1 if r1 < r2 else 1
    grid = np.unique(np.round(np.geomspace(lo, hi, points)).astype(np.int64))
    nz = [(int(B), s_) for B, s_ in ((int(B), sgn(int(B))) for B in grid) if s_ != 0]
    out = []
    for (a, sa), (b, sb) in zip(nz[:-1], nz[1:]):
        if sa == sb:
            continue
        while b - a > 1:                                   # invariant: sgn(a) != sb, sgn(b) == sb
            mid = (a + b) // 2
            if sgn(mid) == sb:
                b = mid
            else:
                a = mid
        out.append((float(b), c1.method if sb < 0 else c2.method))
    return out
