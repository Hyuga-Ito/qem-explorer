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

**Smallest usable target budget.** An estimate needs at least one shot on every distinct target circuit (ZNE: one per
noise scale), and post-selection (symmetry verification) needs enough shots to expect a few accepted ones:
:math:`N_t\ge n_{\min}=\max(\text{circuits},\lceil 5/P_{\rm accept}\rceil)`. Below that the budget is infeasible.

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


def _var_term(x) -> float:
    """A variance term of the cost model: negative rounding → 0, non-finite (NaN, ±inf) → +inf."""
    x = float(x)
    return max(x, 0.0) if math.isfinite(x) else math.inf


def _finite(*xs) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


@dataclass(frozen=True)
class CostModel:
    v: float                 # per-shot variance of the estimator on its target circuits
    fixed: float = 0.0       # training shots (do not depend on the budget)
    c: int = 0               # readout-calibration circuits
    w: float = 0.0           # calibration variance coefficient: Var_cal = w / s
    u: float = 0.0           # variance from the fixed training shots (constant in B)
    circuits: int = 1        # distinct target circuits: each needs at least one target shot (ZNE: one per scale)
    accept: float = 1.0      # post-selection acceptance P_accept (symmetry verification)

    @property
    def n_min(self) -> float:
        """Fewest target shots that yield an estimate: one per target circuit, and with post-selection enough to
        expect MIN_ACCEPTED accepted shots, ⌈5/P_accept⌉. (P_accept = 0 adds no constraint: such a method already
        reports an infinite variance.)"""
        n = float(max(int(self.circuits or 0), 1))
        a = float(self.accept)
        if math.isfinite(a) and 0.0 < a < 1.0 - 1e-12:
            n = max(n, float(math.ceil(min(MIN_ACCEPTED / a, _N_MIN_CAP) - 1e-9)))
        return n

    @staticmethod
    def of(result) -> "CostModel":
        acc = getattr(result, "accept", 1.0)
        return CostModel(v=_var_term(result.var1), fixed=float(getattr(result, "fixed_shots", 0) or 0),
                         c=int(getattr(result, "cal_circuits", 0) or 0), w=_var_term(w_cal(result)),
                         u=_var_term(getattr(result, "var_fixed", 0.0) or 0.0),
                         circuits=max(int(getattr(result, "circuits", 1) or 0), 1),
                         accept=float(acc) if acc is not None else 1.0)

    @staticmethod
    def of_row(row: dict) -> "CostModel":
        v, u, acc = row.get("v", 0.0), row.get("u_train", 0.0), row.get("accepted", 1.0)
        return CostModel(v=_var_term(0.0 if v is None else v), fixed=float(row.get("fixed_shots", 0.0) or 0.0),
                         c=int(row.get("cal_circuits", 0) or 0), w=_var_term(row.get("w_cal", 0.0) or 0.0),
                         u=_var_term(0.0 if u is None else u), circuits=max(int(row.get("n_circuits", 1) or 0), 1),
                         accept=float(acc) if acc is not None else 1.0)


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
    (N_t < n_min, or a non-finite fixed cost). B = N_t + fixed + c·s holds whenever feasible."""
    R = float(B) - m.fixed
    nmin = m.n_min
    if not math.isfinite(R):
        return 0.0, 0.0
    if m.c <= 0:
        return (max(R, 0.0), 0.0) if R >= nmin else (0.0, 0.0)
    if R < m.c + nmin:
        return 0.0, 0.0
    if m.w <= 0:                                  # the calibration circuits still run: one shot each
        return R - m.c, 1.0
    if not _finite(m.v, m.w):                     # the variance is infinite whatever the split: keep it feasible
        s = 1.0
    elif m.v <= 0:
        s = (R - nmin) / m.c
    else:
        k = math.sqrt(m.w / (m.c * m.v))
        s = R * k / (1 + m.c * k)
    s = min(max(s, 1.0), (R - nmin) / m.c)
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
    """The smallest budget that yields an estimate at all: the fixed training shots, one shot per calibration circuit
    and n_min target shots (one per target circuit; ⌈5/P_accept⌉ with post-selection)."""
    return m.fixed + m.c + m.n_min if m.c > 0 else m.fixed + m.n_min


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
    return dict(budget=int(B), feasible=bool(feasible), target_shots=float(N), training_shots=float(m.fixed),
                calibration_shots=float(m.c * s), shots_per_cal=float(s), sd=math.sqrt(var) if feasible else math.inf,
                var_target=m.v / N if feasible else math.inf, var_cal=(m.w / s) if (feasible and s > 0) else 0.0,
                var_train=m.u,
                circuits=target_c + train_c + cal_c, target_circuits=target_c, training_circuits=train_c,
                calibration_circuits=cal_c, min_budget=min_budget(m))


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
