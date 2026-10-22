r"""Equal-budget accounting: every method spends the same total number of shots :math:`B` per estimate.

A method's shots go to three kinds of circuits:

* **target circuits** — the circuits that carry the estimate (the folded copies of ZNE, the sampled instances of
  PEC, the post-selected circuit of symmetry verification, …): :math:`N_t` shots, split among them as the method
  prescribes (ZNE optimally, :math:`N_j\propto|\eta_j|\sigma_j`), giving variance :math:`v/N_t`;
* **training circuits** — CDR / FR / NIL: a fixed number :math:`F` of shots set by the method's parameters;
* **calibration circuits** — readout mitigation: :math:`c` circuits with :math:`s` shots each. Their finite-shot
  error propagates to first order (delta method) as :math:`w/s` with
  :math:`w=\sum_k(\partial\hat O/\partial\theta_k)^2\,\theta_k(1-\theta_k)` over the estimated flip probabilities
  :math:`\theta_k`.

With :math:`B=N_t+F+cs` fixed, the calibration shots are chosen to minimize the total variance
:math:`v/(B-F-cs)+w/s`, which gives :math:`s^*=R\kappa/(1+c\kappa)` with :math:`R=B-F` and
:math:`\kappa=\sqrt{w/(cv)}` (every calibration circuit gets at least one shot, also when :math:`w=0`). A method whose
fixed cost exceeds the budget cannot produce an estimate at that budget (infinite variance). The training shots'
own shot noise propagates through the fit to a constant :math:`u` (CDR / FR / NIL; it does not shrink with :math:`B`
because :math:`F` is fixed). The mean squared error at budget :math:`B` is
:math:`\mathrm{MSE}(B)=b^2+v/N_t(B)+w/s^*(B)+u`.

**Randomized circuits.** Pauli twirling and randomized compiling run a fresh random circuit for every target shot
(one shot per instance: the shots are then independent samples of the twirled distribution, so the variance above is
exact). PEC runs :math:`N_t/s_{\rm PEC}` sampled circuits with :math:`s_{\rm PEC}` shots each; the instance-to-instance
spread is part of :math:`v` (see ``methods/pec.py``).

**Smallest usable target budget — one feasibility contract.** An estimate needs at least one shot on every distinct
target circuit (ZNE: one per noise scale; a randomized protocol with :math:`s` shots per sampled circuit, such as PEC
with ``shots_per_circuit`` = s: one complete sampled circuit, :math:`s` shots), and post-selection (symmetry
verification) needs enough shots to expect a few accepted ones:
:math:`N_t\ge n_{\min}=\max(\text{circuits}\cdot s,\lceil k/P_{\rm accept}\rceil)` with the failure policy
:math:`k` = ``MIN_ACCEPTED`` = 5 (the chance that no shot is kept is then about :math:`e^{-5}`). Every readout
calibration circuit needs at least ``cal_min`` shots (1 here; the phase pipeline asks for 32) and the training shots
are spent in full. :func:`min_target_shots` and :func:`min_total_budget` state this contract once (with a
:class:`Feasibility` rule for the minima); the interactive cost model (:class:`CostModel`, :func:`split`), the phase
ledger (``qemx.phase.ledger``) and the hardware planner (``qemx.hardware``) all call them, so the three agree on the
minimum budget and on whether a budget is feasible (``tests/test_feasibility_contract.py``). Below the minimum the
budget is infeasible.

**Small accepted counts.** The post-selected estimate is a ratio; its variance :math:`v/N_t` is first order (delta
method). At the feasibility boundary the conditional variance of the ratio, given that at least one shot was kept,
is larger by the factor :math:`N_tP\,E[1/K\mid K\ge1]` (:math:`K\sim\mathrm{Bin}(N_t,P)` accepted shots; about
1.2–1.3 at :math:`N_tP=5`), exactly for post-selection without readout mitigation: :func:`postselection_check`
computes it, and the comparison table switches to the spread of exact repeats when it is clearly larger.

**Information regimes.** The optimal splits above use variances computed from exact or predicted probabilities: in
simulation that is *oracle* information (``ORACLE``); a hardware plan uses the device's calibration snapshot
(``SUPPLIED``); only quantities estimated from circuits charged to the budget (readout calibration, training data) are
``LEARNED``. :func:`account` labels its allocation accordingly.

**Unknown variances.** A non-finite variance term (a failed fit, a NaN estimate) is not a small one: the cost model
maps it to :math:`+\infty`, and ``variance``, ``rmse``, ``budget_for`` and ``crossover`` then report an infinite RMSE,
an infinite budget or no crossover — never a finite answer for a NaN estimate or bias.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_ACCEPTED = 5          # post-selection: accepted shots an estimate should expect (P(no shot kept) ≈ e⁻⁵)
_N_MIN_CAP = 1e18         # keeps n_min finite (callers format it as an integer) for vanishing acceptance

# information regimes (review §5.3/§5.5): what an allocation or a noise model was allowed to know
ORACLE = "oracle"                         # exact simulator quantities (true channel, exact probabilities): a reference
SUPPLIED = "supplied-calibration"         # a fixed calibration snapshot or assumed rates, not paid for in the budget
LEARNED = "learned-with-cost"             # estimated from circuits whose shots are charged to the budget


def _var_term(x) -> float:
    """A variance term of the cost model: negative rounding → 0, non-finite (NaN, ±inf) → +inf."""
    x = float(x)
    return max(x, 0.0) if math.isfinite(x) else math.inf


def _finite(*xs) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


@dataclass(frozen=True)
class Feasibility:
    """The minima of the feasibility contract (see the module text). One rule object serves every adapter: the
    interactive cost model uses the default, the phase ledger maps its ``Policy`` lower bounds onto it
    (``s_min`` → ``cal_min``, ``s_tr_min`` → ``train_min``), the hardware planner takes one as an argument."""
    min_accepted: float = MIN_ACCEPTED   # failure policy of post-selection: expected accepted target shots
    cal_min: int = 1                     # shots per readout-calibration circuit
    train_min: int = 1                   # shots per training circuit, where a policy chooses them

    def __post_init__(self):
        if not (math.isfinite(float(self.min_accepted)) and float(self.min_accepted) > 0):
            raise ValueError("min_accepted must be finite and > 0")
        for nm in ("cal_min", "train_min"):
            x = getattr(self, nm)
            if int(x) != x or int(x) < 1:
                raise ValueError("%s must be an integer >= 1" % nm)


DEFAULT_FEASIBILITY = Feasibility()


def _accept_of(accept) -> float:
    """Acceptance as the contract reads it: None (not post-selected) or a non-finite value -> 1 (no constraint)."""
    if accept is None:
        return 1.0
    a = float(accept)
    return a if math.isfinite(a) else 1.0


def min_target_shots(circuits=1, accept=1.0, per_circuit=1, rule: Feasibility = DEFAULT_FEASIBILITY) -> int:
    """Fewest target shots that yield an estimate: ``per_circuit`` shots on each distinct target circuit (1; PEC with
    s shots per sampled circuit: s) and, with post-selection, enough shots to expect ``rule.min_accepted`` accepted
    ones, ⌈k/P_accept⌉. P_accept = 0 adds no constraint: such a method already reports an infinite variance."""
    n = max(int(circuits or 0), 1) * max(int(per_circuit or 1), 1)
    a = _accept_of(accept)
    if 0.0 < a < 1.0 - 1e-12:
        n = max(n, int(math.ceil(min(float(rule.min_accepted) / a, _N_MIN_CAP) - 1e-9)))
    return int(n)


def min_total_budget(target_circuits=1, accept=1.0, per_circuit=1, training_shots=0.0, cal_circuits=0,
                     rule: Feasibility = DEFAULT_FEASIBILITY) -> float:
    """The smallest total budget that yields an estimate: the training shots, ``rule.cal_min`` shots per calibration
    circuit and :func:`min_target_shots` target shots. The one feasibility function of every adapter."""
    tr = float(training_shots or 0.0)
    if not math.isfinite(tr):
        return math.inf
    return tr + max(int(cal_circuits or 0), 0) * int(rule.cal_min) + float(
        min_target_shots(target_circuits, accept, per_circuit, rule))


def feasible(B, target_circuits=1, accept=1.0, per_circuit=1, training_shots=0.0, cal_circuits=0,
             rule: Feasibility = DEFAULT_FEASIBILITY) -> bool:
    """Does the total budget B yield an estimate at all (B ≥ :func:`min_total_budget`)?"""
    return float(B) >= min_total_budget(target_circuits, accept, per_circuit, training_shots, cal_circuits, rule)


def postselection_check(N: int, accept: float) -> dict:
    """Small accepted counts: what the first-order (delta-method) variance of a post-selected ratio leaves out at N
    target shots with acceptance P. K ~ Bin(N, P) shots are kept; the estimate exists when K ≥ 1 (probability
    1 − (1 − P)^N) and, given K, is the mean of K independent accepted shots, so (no readout mitigation)
    Var(estimate | K ≥ 1) = σ²_acc · E[1/K | K ≥ 1] exactly, while the delta method gives σ²_acc / (N P).
    Returns {failure_prob, expected_accepted, var_inflation = N P · E[1/K | K ≥ 1]} (≥ 1 − failure_prob by Jensen;
    → 1 as N P → ∞; about 1.29 at the feasibility boundary N P = 5 for small P).
    Readout mitigation adds the calibration's own nonlinearity, which this does not cover."""
    N = int(N)
    P = float(accept)
    if N < 1 or not (0.0 < P <= 1.0):
        return dict(failure_prob=1.0, expected_accepted=0.0, var_inflation=math.inf)
    if P >= 1.0:
        return dict(failure_prob=0.0, expected_accepted=float(N), var_inflation=1.0)
    lam = N * P
    fail = math.exp(N * math.log1p(-P))
    if lam > 1e4:                                   # second order: E[1/K] ≈ (1 + (1 − P)/(N P)) / (N P)
        return dict(failure_prob=fail, expected_accepted=lam, var_inflation=1.0 + (1.0 - P) / lam)
    from scipy.stats import binom
    lo = max(1, int(lam - 40 * math.sqrt(lam * (1 - P)) - 40))
    hi = min(N, int(lam + 40 * math.sqrt(lam * (1 - P)) + 40))
    k = np.arange(lo, hi + 1)
    pmf = binom.pmf(k, N, P)
    p_ok = 1.0 - fail
    if p_ok <= 0:
        return dict(failure_prob=1.0, expected_accepted=lam, var_inflation=math.inf)
    e_inv = float(np.sum(pmf / k)) / p_ok
    return dict(failure_prob=fail, expected_accepted=lam, var_inflation=lam * e_inv)


@dataclass(frozen=True)
class CostModel:
    v: float                 # per-shot variance of the estimator on its target circuits
    fixed: float = 0.0       # training shots (do not depend on the budget)
    c: int = 0               # readout-calibration circuits
    w: float = 0.0           # calibration variance coefficient: Var_cal = w / s
    u: float = 0.0           # variance from the fixed training shots (constant in B)
    circuits: int = 1        # distinct target circuits: each needs at least one target shot (ZNE: one per scale)
    accept: float = 1.0      # post-selection acceptance P_accept (symmetry verification)
    per_circuit: int = 1     # shots one target circuit needs at least (PEC: its shots per sampled circuit s)
    rule: Feasibility = DEFAULT_FEASIBILITY   # minima of the feasibility contract

    @property
    def n_min(self) -> float:
        """Fewest target shots that yield an estimate (:func:`min_target_shots`): one per target circuit (s per
        sampled PEC circuit), and with post-selection enough to expect MIN_ACCEPTED accepted shots, ⌈5/P_accept⌉.
        (P_accept = 0 adds no constraint: such a method already reports an infinite variance.)"""
        return float(min_target_shots(self.circuits, self.accept, self.per_circuit, self.rule))

    @staticmethod
    def of(result, rule: Feasibility = DEFAULT_FEASIBILITY) -> "CostModel":
        acc = getattr(result, "accept", 1.0)
        spc = getattr(result, "shots_per_circuit", None)
        return CostModel(v=_var_term(result.var1), fixed=float(getattr(result, "fixed_shots", 0) or 0),
                         c=int(getattr(result, "cal_circuits", 0) or 0), w=_var_term(w_cal(result)),
                         u=_var_term(getattr(result, "var_fixed", 0.0) or 0.0),
                         circuits=max(int(getattr(result, "circuits", 1) or 0), 1),
                         accept=_accept_of(acc), per_circuit=max(int(spc or 1), 1), rule=rule)

    @staticmethod
    def of_row(row: dict, rule: Feasibility = DEFAULT_FEASIBILITY) -> "CostModel":
        v, u, acc = row.get("v", 0.0), row.get("u_train", 0.0), row.get("accepted", 1.0)
        return CostModel(v=_var_term(0.0 if v is None else v), fixed=float(row.get("fixed_shots", 0.0) or 0.0),
                         c=int(row.get("cal_circuits", 0) or 0), w=_var_term(row.get("w_cal", 0.0) or 0.0),
                         u=_var_term(0.0 if u is None else u), circuits=max(int(row.get("n_circuits", 1) or 0), 1),
                         accept=_accept_of(acc), per_circuit=max(int(row.get("per_circuit", 1) or 1), 1),
                         rule=rule)


def w_cal(result) -> float:
    w = getattr(result, "cal_w", None)
    if w is not None:                               # measured by Monte Carlo (nonlinear refits)
        return float(w) if math.isfinite(float(w)) else math.inf
    g, th = getattr(result, "cal_grad", None), getattr(result, "cal_theta", None)
    if g is None or th is None or len(g) == 0:
        return 0.0
    g, th = np.asarray(g, float), np.clip(np.asarray(th, float), 0.0, 1.0)
    return float(np.sum(g ** 2 * th * (1 - th)))


def split(m: CostModel, B: float) -> tuple:
    """(target shots N_t, shots per calibration circuit s) for total budget B; (0, 0) when infeasible
    (B < min_budget(m), i.e. N_t < n_min or s < cal_min, or a non-finite fixed cost). B = N_t + fixed + c·s holds
    whenever feasible."""
    R = float(B) - m.fixed
    nmin = m.n_min
    smin = float(m.rule.cal_min)
    if not math.isfinite(R):
        return 0.0, 0.0
    if m.c <= 0:
        return (max(R, 0.0), 0.0) if R >= nmin else (0.0, 0.0)
    if R < m.c * smin + nmin:
        return 0.0, 0.0
    if m.w <= 0:                                  # the calibration circuits still run: cal_min shots each
        return R - m.c * smin, smin
    if not _finite(m.v, m.w):                     # the variance is infinite whatever the split: keep it feasible
        s = smin
    elif m.v <= 0:
        s = (R - nmin) / m.c
    else:
        k = math.sqrt(m.w / (m.c * m.v))
        s = R * k / (1 + m.c * k)
    s = min(max(s, smin), (R - nmin) / m.c)
    return R - m.c * s, s


def optimal_integer_split(a, N: int) -> tuple:
    """Integers N_j >= 1 with Σ N_j = N minimizing Σ a_j²/N_j (the variance of a linear combination of circuits
    whose j-th term has standard deviation a_j per shot, e.g. ZNE with a_j = |η_j|σ_j). Exact: start from the floors
    of the continuous optimum N_j ∝ a_j, fill greedily, then apply unit transfers until none improves (for a
    separable convex objective under one sum constraint, no improving transfer ⇔ optimal). Returns (variance, N_j)."""
    a = np.abs(np.asarray(a, float))
    k = len(a)
    N = int(N)
    if k == 0 or N < k:
        return math.inf, ()
    if not np.any(a > 0):
        n = np.full(k, N // k, dtype=np.int64)
        n[: N - int(n.sum())] += 1
        return 0.0, tuple(int(x) for x in n)
    n = np.maximum(np.floor(a / a.sum() * N).astype(np.int64), 1)
    a2 = a ** 2
    while n.sum() > N:                                  # raising tiny shares to 1 overshot: take back the cheapest
        cost_dec = np.where(n > 1, a2 / np.maximum(n - 1, 1) - a2 / n, np.inf)
        n[int(np.argmin(cost_dec))] -= 1
    while n.sum() < N:
        n[int(np.argmax(a2 / n - a2 / (n + 1)))] += 1
    for _ in range(4 * k * k + 4):
        f = a2 / n
        best, move = 0.0, None
        for i in range(k):
            if n[i] <= 1:
                continue
            up = a2[i] / (n[i] - 1) - f[i]
            for j in range(k):
                if i != j:
                    d = up + (a2[j] / (n[j] + 1) - f[j])
                    if d < best - 1e-15 * float(f.sum()):
                        best, move = d, (i, j)
        if move is None:
            break
        n[move[0]] -= 1
        n[move[1]] += 1
    if int(n.sum()) != N or int(n.min()) < 1:
        raise AssertionError("integer split failed")
    return float(np.sum(a2 / n)), tuple(int(x) for x in n)


def integer_split(shares, N: int, minimum: int = 1, exact_max: int = 64) -> tuple:
    """Integers n_j ≥ ``minimum`` with Σ n_j = N exactly (budget-conserving), following the continuous split
    N·shares_j. With ``minimum`` = 1 and at most ``exact_max`` parts: the exact minimizer of Σ shares_j²/n_j
    (:func:`optimal_integer_split` — the first-order variance of a linear combination when shares ∝ a_j, as for
    ZNE's N_j ∝ |η_j|σ_j or a Pauli sum's N_g ∝ √v_g); otherwise the floors of N·shares_j raised to ``minimum``, then
    the remaining shots by largest remainders (Hamilton's method; integral targets N·shares_j are met exactly). Non-finite, negative or all-zero
    shares split evenly. ValueError when N < minimum·len(shares)."""
    a = np.asarray(shares, float).ravel()
    k = len(a)
    N, minimum = int(N), int(minimum)
    if k == 0:
        if N != 0:
            raise ValueError("no parts to receive %d shots" % N)
        return ()
    if N < minimum * k:
        raise ValueError("need at least %d shots per part (N = %d < %d × %d)" % (minimum, N, k, minimum))
    if not np.isfinite(a).all() or np.any(a < 0) or a.sum() <= 0:
        a = np.ones(k)
    if minimum == 1 and k <= exact_max:
        return optimal_integer_split(a, N)[1]
    q = N * a / a.sum()
    n = np.maximum(np.floor(q).astype(np.int64), minimum)
    while int(n.sum()) > N:                       # minimums overshot: take back from the most over-served parts
        over = np.where(n > minimum, n - q, -np.inf)
        n[int(np.argmax(over))] -= 1
    rest = N - int(n.sum())
    if rest > 0:                                  # largest remainders (Hamilton's method)
        n[np.argsort(-(q - n), kind="stable")[:rest]] += 1
    out = tuple(int(x) for x in n)
    if sum(out) != N or min(out) < minimum:
        raise AssertionError("integer split does not conserve the budget")
    return out


def _model_finite(m: CostModel) -> bool:
    return _finite(m.v, m.w, m.u)


def variance(m: CostModel, B: float) -> float:
    """Variance at total budget B; inf when infeasible or when any variance term is not finite."""
    if not _model_finite(m):
        return math.inf
    N, s = split(m, B)
    if not N > 0:
        return math.inf
    return m.v / N + (m.w / s if s > 0 else 0.0) + m.u


def rmse(bias: float, m: CostModel, B: float) -> float:
    if not _finite(bias):
        return math.inf
    var = variance(m, B)
    return math.sqrt(bias * bias + var) if math.isfinite(var) else math.inf


def rmse_curve(bias: float, m: CostModel, budgets) -> np.ndarray:
    return np.array([rmse(bias, m, b) for b in np.atleast_1d(np.asarray(budgets, float))])


def min_budget(m: CostModel) -> float:
    """The smallest budget that yields an estimate at all (:func:`min_total_budget`): the fixed training shots,
    cal_min shots per calibration circuit and n_min target shots (one per target circuit; ⌈5/P_accept⌉ with
    post-selection)."""
    return min_total_budget(m.circuits, m.accept, m.per_circuit, m.fixed, m.c, m.rule)


def budget_for(bias: float, m: CostModel, eps: float) -> float:
    """Smallest total budget B with RMSE(B) ≤ eps (inf when the bias and the training-noise floor reach eps, and
    whenever the bias or a variance term is not finite: a NaN estimate never gets a finite budget)."""
    if not (_finite(bias, eps) and _model_finite(m)):
        return math.inf
    if eps * eps <= bias * bias + m.u:
        return math.inf
    target = eps * eps - bias * bias
    lo = min_budget(m)
    if not math.isfinite(lo):
        return math.inf
    if variance(m, lo) <= target:
        return lo
    hi = max(lo * 2, 2.0)
    while variance(m, hi) > target:
        hi *= 2
        if hi > 1e18:
            return math.inf
    for _ in range(200):                     # variance(B) decreases monotonically in B
        mid = math.sqrt(lo * hi) if lo > 0 else (lo + hi) / 2
        if variance(m, mid) > target:
            lo = mid
        else:
            hi = mid
        if hi / max(lo, 1e-300) < 1 + 1e-9:
            break
    return hi


def crossover(b_m: float, m: CostModel, b_r: float, r: CostModel, lo: float = 1.0, hi: float = 1e15):
    """Budget above which method m has the lower RMSE than reference r and keeps it; None if never (or always), and
    None whenever either bias or any variance term is not finite."""
    if not (_finite(b_m, b_r) and _model_finite(m) and _model_finite(r)):
        return None
    lo = max(lo, min_budget(m), min_budget(r))
    if not (math.isfinite(lo) and lo < hi):
        return None
    grid = np.unique(np.geomspace(lo, hi, 400))
    d = np.array([rmse(b_m, m, B) - rmse(b_r, r, B) for B in grid])
    ok = np.isfinite(d)
    if not ok.any():
        return None
    better = (d < 0) & ok
    if better.all() or not better[-1]:
        return None
    i = int(np.flatnonzero(~better)[-1])            # last budget where m is not better
    a, b = grid[i], grid[i + 1]
    for _ in range(100):
        mid = math.sqrt(a * b)
        dm = rmse(b_m, m, mid) - rmse(b_r, r, mid)
        if dm < 0:
            b = mid
        else:
            a = mid
    return float(b)


def _as_run(m: CostModel, B: float, alloc) -> tuple:
    """(N_t, s): the optimal split at B, or an allocation actually run (hardware: {'N': …, 's': …})."""
    if alloc is None:
        return split(m, B)
    return float(alloc["N"]), float(alloc.get("s", 0.0) or 0.0)


def account(result, B: int, randomized_target: bool = False, alloc: dict | None = None, meas_twirls: int = 1) -> dict:
    """How a method spends a total budget B: target / training / calibration shots, circuits sent, analytic sd.
    ``alloc`` replaces the optimal split by the one actually run (hardware), so sd is that experiment's.
    ``meas_twirls``: measurement-twirled readout (TREX) sends every fixed circuit with that many random X masks."""
    m = CostModel.of(result)
    N, s = _as_run(m, B, alloc)
    feasible = N >= 1
    var = (m.v / N + (m.w / s if s > 0 else 0.0) + m.u) if feasible else math.inf
    spc = getattr(result, "shots_per_circuit", None)
    base = int(getattr(result, "circuits", 1) or 0)
    if not feasible:
        target_c = 0
    elif spc:                                         # randomized protocol: one circuit per spc shots
        target_c = int(math.ceil(N / spc))
    elif randomized_target:                           # twirling / RC: a fresh random circuit for every shot
        target_c = int(N)
    else:
        target_c = base * max(int(meas_twirls), 1)
    train_c = int(getattr(result, "train_circuits", 0) or 0)
    if randomized_target and train_c:
        train_c = int(m.fixed)                       # twirled training circuits: one random instance per shot
    cal_c = (int(m.c) if s > 0 or m.c else 0) * max(int(meas_twirls), 1)
    if alloc is not None and alloc.get("circuits_by_role"):         # hardware: the circuits actually sent
        cr = alloc["circuits_by_role"]
        target_c, train_c, cal_c = int(cr["target"]), int(cr["training"]), int(cr["calibration"])
    # resources kept apart (review §5.2): shots = circuit executions; distinct logical circuits before any
    # randomization; circuit instances actually sent (``circuits``, after PEC sampling / twirling / TREX masks)
    distinct = (base if feasible else 0) + int(getattr(result, "train_circuits", 0) or 0) + int(m.c)
    executions = (float(N) + float(m.fixed) + float(m.c * s)) if feasible else 0.0
    post = postselection_check(int(N), m.accept) if (feasible and m.accept < 1.0 - 1e-12) else None
    return dict(budget=int(B), feasible=bool(feasible), target_shots=float(N), training_shots=float(m.fixed),
                calibration_shots=float(m.c * s), shots_per_cal=float(s), sd=math.sqrt(var) if feasible else math.inf,
                var_target=m.v / N if feasible else math.inf, var_cal=(m.w / s) if (feasible and s > 0) else 0.0,
                var_train=m.u,
                circuits=target_c + train_c + cal_c, target_circuits=target_c, training_circuits=train_c,
                calibration_circuits=cal_c, min_budget=min_budget(m),
                executions=executions, distinct_circuits=distinct, circuit_instances=target_c + train_c + cal_c,
                allocation_information=(ORACLE if alloc is None else str(alloc.get("information_regime", SUPPLIED))),
                postselection_var_inflation=(post["var_inflation"] if post else 1.0))


def draw(result, B: int, n: int, rng, alloc: dict | None = None) -> np.ndarray | None:
    """n simulated experiments at total budget B: the method's own sampler at its target shots, plus the
    first-order Gaussian error of a finitely sampled readout calibration (``alloc``: the split actually run)."""
    if result.sampler is None:
        return None
    m = CostModel.of(result)
    N, s = _as_run(m, B, alloc)
    if N < 1:
        return np.full(int(n), np.nan)
    x = np.asarray(result.sampler(int(max(round(N), 1)), int(n), rng), float)
    if s > 0 and m.w > 0:
        if not math.isfinite(m.w):
            return np.full(x.shape, np.nan)            # an unknown calibration error: no usable estimate
        x = x + rng.normal(0.0, math.sqrt(m.w / s), size=x.shape)
    return x
