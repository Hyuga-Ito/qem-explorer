r"""Candidate-risk models (1.6.0).

For every protocol a, a regressor predicts y = log10(R̂_sel(𝒯, a, B) + ε0) from deployable features x = φ(𝒯, B)
(theory companion §1.5.2–1.5.3). ε0 (default 1e-6, i.e. an RMSE of 1e-3) keeps the logarithm finite and caps how
much the model cares about differences between already-negligible risks; a log prediction exponentiated is not a
conditional-mean risk prediction, so models are selected and scored by decision performance (regret, policy MSE),
and the floor's influence is reported (``FLOORS``).

Model families, tuned the same way (a grid of comparable size, inner group-K-fold over base circuits, the same
folds for every family):

* ``ridge`` — standardized features, unpenalized intercept (Eq. 1.17), α on a log grid;
* ``enet`` — elastic net, α × ℓ1-ratio grid;
* ``hgb`` — histogram gradient-boosted trees (scikit-learn), depth × learning-rate grid, early stopping off;
* ``mlp`` — a small neural network (one hidden layer of 16 or 32 units, L-BFGS, the solver suited to a few thousand
  rows), weight decay × hidden size grid, fixed seeds;
* ``struct`` (1.7.0) — a physics-informed risk model, R_a(x, B) ≈ b_a(x)² + v_a(x)/B_eff with B_eff = (1 − f_a)·B
  and f_a the protocol's declared training share: a residual-bias floor plus a sampling term after the training
  overhead (``StructuredRisk``). A testable approximation, not a law: nonlinear estimators and budget-dependent
  allocations bend it, and u_a(x, B) — what it leaves out — is the residual.

Secondary analyses (ablation, learning curves, abstention sweeps) refit one fixed setting (``fixed``) instead of
re-tuning, so that their differences come from the data or features they vary, not from tuning noise.

Only training rows of feasible (task, protocol, budget) combinations enter; infeasibility is known before execution
(``protocols.allocate``) and handled by the policy, not the regressor.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

EPS0 = 1e-6
FLOORS = (1e-7, 1e-6, 1e-5)
FAMILIES = ("ridge", "enet", "hgb", "mlp", "struct")


def target(mse, eps0: float = EPS0) -> np.ndarray:
    return np.log10(np.asarray(mse, float) + float(eps0))


def _grid(family: str) -> list:
    if family == "ridge":
        return [dict(alpha=a) for a in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)]
    if family == "enet":
        return [dict(alpha=a, l1_ratio=r) for a in (1e-3, 1e-2, 1e-1) for r in (0.2, 0.8)]
    if family == "hgb":
        return [dict(max_depth=d, learning_rate=lr) for d in (2, 3, 5) for lr in (0.05, 0.15)]
    if family == "mlp":
        return [dict(alpha=a, hidden=h) for a in (1e-4, 1e-2, 1.0) for h in (16, 32)]
    if family == "struct":
        return [dict(alpha=a) for a in (1e-2, 1e-1, 1.0, 10.0, 100.0)]
    raise ValueError(family)


def make(family: str, seed: int = 0, **hp):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if family == "ridge":
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=hp["alpha"]))
    if family == "enet":
        from sklearn.linear_model import ElasticNet
        return make_pipeline(StandardScaler(), ElasticNet(alpha=hp["alpha"], l1_ratio=hp["l1_ratio"],
                                                          max_iter=20000))
    if family == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=hp["max_depth"], learning_rate=hp["learning_rate"],
                                             max_iter=200, early_stopping=False, random_state=seed)
    if family == "mlp":
        from sklearn.neural_network import MLPRegressor
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(hp["hidden"],), alpha=hp["alpha"],
                                                            solver="lbfgs", max_iter=800, random_state=seed))
    raise ValueError(family)


def group_kfold(groups, k: int = 5, seed: int = 0) -> list:
    """(train, test) index arrays with whole groups held out; groups assigned by a seeded permutation."""
    g = np.asarray(groups)
    uniq = np.array(sorted(set(g.tolist())))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    k = max(2, min(int(k), len(uniq)))
    out = []
    for f in range(k):
        test_g = set(uniq[perm[f::k]].tolist())
        te = np.flatnonzero([x in test_g for x in g])
        tr = np.flatnonzero([x not in test_g for x in g])
        out.append((tr, te))
    return out


def _fit(est, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y)
    return est


@dataclass
class RiskModel:
    family: str
    features: tuple
    protocols: tuple
    eps0: float = EPS0
    seed: int = 0
    models: dict = field(default_factory=dict)          # protocol -> fitted estimator
    chosen: dict = field(default_factory=dict)          # protocol -> hyper-parameters
    cv_rmse: dict = field(default_factory=dict)         # protocol -> inner-CV RMSE of log10 risk
    oof_resid: dict = field(default_factory=dict)       # protocol -> (groups, residuals) out of fold (conformal)
    impute: np.ndarray | None = None
    fixed: dict | None = None                            # one hyper-parameter setting instead of the grid

    def _prep(self, X):
        X = np.asarray(X, float).copy()
        if self.impute is None:
            self.impute = np.nanmedian(X, 0)
            self.impute[~np.isfinite(self.impute)] = 0.0
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.impute, np.nonzero(bad)[1])
        return X

    def _make(self, p: str, hp: dict):
        if self.family == "struct":
            from .protocols import PORTFOLIO
            if "log10_budget" not in self.features:
                raise ValueError("the structured model needs log10_budget among its features")
            return StructuredRisk(PORTFOLIO[p].train_fraction, list(self.features).index("log10_budget"),
                                  hp["alpha"], self.eps0)
        return make(self.family, self.seed, **hp)

    def _fit_one(self, p, hp, Xa, ya, ta):
        est = self._make(p, hp)
        if self.family == "struct":
            return est.fit(Xa, ya, ta)
        return _fit(est, Xa, ya)

    def fit(self, X, Y, F, groups, inner_k: int = 4, tasks=None) -> "RiskModel":
        """X (rows, features); Y (rows, protocols) label MSEs (NaN where infeasible); F (rows, protocols) feasible;
        groups (rows,) base-circuit ids; ``tasks`` (rows,) task ids (the structured model's stage 1). Tunes per
        protocol by inner group-K-fold RMSE of the log target and keeps the out-of-fold residuals of the chosen
        setting (for the interval constructions)."""
        X = self._prep(X)
        Y, F, g = np.asarray(Y, float), np.asarray(F, bool), np.asarray(groups)
        tk = np.asarray(tasks) if tasks is not None else np.array([hash(tuple(np.round(r_, 10))) for r_ in X])
        for j, p in enumerate(self.protocols):
            use = F[:, j] & np.isfinite(Y[:, j])
            if use.sum() < 8 or len(set(g[use].tolist())) < 3:
                continue
            Xu, yu, gu, tu = X[use], target(Y[use, j], self.eps0), g[use], tk[use]
            folds = group_kfold(gu, inner_k, seed=self.seed + 17)
            best = None
            for hp in ([dict(self.fixed)] if self.fixed else _grid(self.family)):
                pred = np.full(len(yu), np.nan)
                for tr, te in folds:
                    if len(tr) < 4 or len(te) == 0:
                        continue
                    try:
                        pred[te] = self._fit_one(p, hp, Xu[tr], yu[tr], tu[tr]).predict(Xu[te])
                    except ValueError:                      # struct: too few multi-budget tasks in this fold
                        continue
                ok = np.isfinite(pred)
                rm = float(np.sqrt(np.mean((pred[ok] - yu[ok]) ** 2))) if ok.any() else math.inf
                if best is None or rm < best[0] - 1e-12:
                    best = (rm, hp, pred)
            if not math.isfinite(best[0]):
                continue
            self.chosen[p], self.cv_rmse[p] = best[1], best[0]
            self.oof_resid[p] = (gu, yu - best[2])
            try:
                self.models[p] = self._fit_one(p, best[1], Xu, yu, tu)
            except ValueError:
                self.chosen.pop(p), self.cv_rmse.pop(p), self.oof_resid.pop(p)
        return self

    def predict(self, X) -> np.ndarray:
        """(rows, protocols) predicted log10(R + ε0); NaN for protocols without a model."""
        X = self._prep(X)
        out = np.full((len(X), len(self.protocols)), np.nan)
        for j, p in enumerate(self.protocols):
            m = self.models.get(p)
            if m is not None:
                out[:, j] = m.predict(X)
        return out

    def conformal_halfwidth(self, alpha: float = 0.1, simultaneous: bool = False) -> dict:
        """Group-weighted conformal-style half-width of the log10 prediction per protocol from the inner out-of-fold
        residuals (every base circuit weighs the same; (1 − α)(1 + 1/G) quantile). Placed around the model refitted on
        all training rows this is the cross-validation ("CV") construction: no finite-sample 1 − α guarantee (CV+
        guarantees 1 − 2α; Barber et al. 2021), so coverage is measured, not assumed. ``simultaneous``: Bonferroni over
        the K protocols (α/K each), so that all K intervals of a new task hold together — required before reporting
        the interval of a *selected* protocol (optimizer's curse)."""
        from ..recommend import weighted_quantile
        K = max(len(self.oof_resid), 1)
        a = alpha / K if simultaneous else alpha
        out = {}
        for p, (g, r) in self.oof_resid.items():
            ok = np.isfinite(r)
            if not ok.any():
                continue
            size = {k: int(np.sum(g[ok] == k)) for k in set(g[ok].tolist())}
            w = np.array([1.0 / size[k] for k in g[ok]])
            G = len(size)
            out[p] = weighted_quantile(np.abs(r[ok]), w, (1 - a) * (1 + 1.0 / G))
        return out


FIXED = {"ridge": dict(alpha=1.0), "enet": dict(alpha=1e-2, l1_ratio=0.2), "hgb": dict(max_depth=3, learning_rate=0.05),
         "mlp": dict(alpha=1e-2, hidden=32), "struct": dict(alpha=1.0)}

STRUCT_FLOOR = 1e-14


def struct_fit_task(R, B_eff) -> tuple:
    """(b², v) ≥ 0 of one task and protocol from its risks R at the budgets' B_eff: nonnegative least squares of the
    relative error Σ_B ((b² + v/B_eff − R)/R)² (every budget counts by its relative fit, not its magnitude)."""
    from scipy.optimize import nnls
    R = np.maximum(np.asarray(R, float), STRUCT_FLOOR)
    Be = np.asarray(B_eff, float)
    A = np.column_stack([1.0 / R, (1.0 / Be) / R])
    sol, _ = nnls(A, np.ones(len(R)))
    return float(sol[0]), float(sol[1])


class StructuredRisk:
    """Per protocol: stage 1 fits (b², v) to every training task's risks over its budgets (``struct_fit_task``);
    stage 2 maps the budget-free features to log10 b² and log10 v (standardized ridge). Prediction:
    log10(b̂² + v̂/B_eff + ε0)."""

    def __init__(self, f_train: float, budget_col: int, alpha: float = 1.0, eps0: float = EPS0):
        self.f, self.bc, self.alpha, self.eps0 = float(f_train), int(budget_col), float(alpha), float(eps0)

    def _xb(self, X):
        X = np.asarray(X, float)
        B = 10.0 ** X[:, self.bc]
        return np.delete(X, self.bc, axis=1), (1.0 - self.f) * B

    def fit(self, X, y_log, tasks):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Xn, Be = self._xb(X)
        R = 10.0 ** np.asarray(y_log, float) - self.eps0
        tasks = np.asarray(tasks)
        rows, t1, t2 = [], [], []
        for t in sorted(set(tasks.tolist())):
            m = tasks == t
            if m.sum() < 2:
                continue
            b2, v = struct_fit_task(R[m], Be[m])
            rows.append(np.flatnonzero(m)[0])
            t1.append(math.log10(b2 + STRUCT_FLOOR))
            t2.append(math.log10(v + STRUCT_FLOOR))
        if len(rows) < 3:
            raise ValueError("too few tasks with two or more budgets for the structured model")
        Xt = Xn[rows]
        self.m1 = _fit(make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)), Xt, np.array(t1))
        self.m2 = _fit(make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)), Xt, np.array(t2))
        self.tasks_fitted = len(rows)
        return self

    def predict(self, X):
        Xn, Be = self._xb(X)
        b2 = 10.0 ** self.m1.predict(Xn)
        v = 10.0 ** self.m2.predict(Xn)
        return np.log10(np.maximum(b2 - STRUCT_FLOOR, 0.0) + np.maximum(v - STRUCT_FLOOR, 0.0) / Be + self.eps0)

    def components(self, X) -> tuple:
        """(b², v, B_eff) predicted for rows X."""
        Xn, Be = self._xb(X)
        return 10.0 ** self.m1.predict(Xn), 10.0 ** self.m2.predict(Xn), Be
