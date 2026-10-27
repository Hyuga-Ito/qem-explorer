r"""Selection policies (1.6.0). Each is fitted on training rows only and returns, for test rows, the chosen protocol
index (and, for abstaining policies, whether the recommendation was accepted).

* ``RAW`` — always the unmitigated estimator.
* ``SBS`` — the single best protocol on the training rows: the smallest mean log10(R_sel + ε0) over *all* training
  (task, budget) rows of "this protocol where feasible, else RAW" — one population for every candidate, base circuits
  weighted equally (1.7.0; 1.6.0 averaged each protocol over its own feasible rows only) — chosen on *training*
  instances (Kerschke et al., Lindauer et al.; choosing it on the test set would flatter the baseline). Where it is
  infeasible, the next in its training ranking.
* ``SBS-B`` — a budget-dependent fixed policy: the same ranking computed separately at each budget of the grid.
* ``SIZE-B`` — a simple size-and-budget model: a depth-3 classification tree on (n, two-qubit gates, log10 B)
  predicting the training rows' best protocol; an infeasible prediction falls back to SBS-B.
* ``MODEL`` — argmin of a fitted ``RiskModel`` over the feasible protocols (Eq. 1.18).
* ``MODEL+ABSTAIN`` — accept the model's choice only when its simultaneous conformal upper bound is below the
  fallback's lower bound (U_â < L_0 on the predicted log scale, the theory companion's Theorem 1.6 applied to
  predictions); otherwise use the fallback (SBS-B). Coverage κ is the accepted share (Eq. 1.25).
* ``ALL-CANDIDATE`` — argmin of the measured R_sel of the test task itself: a benchmarking construction that would
  need every candidate's risk at deployment (impossible without the ideal value); scored on the independent
  evaluation half, it is the per-task oracle reference (VBS) of the regret.

No policy here acquires shots before deciding, so selection and diagnostic budgets are 0; policies that do
(``qemx.selection.mismatch``) charge them to B through ``protocols.allocate``.
"""
from __future__ import annotations

import math

import numpy as np

from .models import EPS0, RiskModel, target


FALLBACK = 0                    # index of RAW in the portfolio order: always feasible, the declared fallback


def _rank_from_train(Y, F, eps0=EPS0, groups=None, fallback: int = FALLBACK) -> list:
    """Protocol indices sorted by their training score; never-feasible protocols last.

    1.7.0 (review S5): every candidate is scored on the *same* population — all training rows — as the policy
    "this protocol where feasible, else the declared fallback (RAW)", mean log10(R + ε0) with every base circuit
    weighted equally. 1.6.0 averaged each protocol over the rows where it is feasible only, so protocols infeasible
    at small budgets were compared on easier subsets."""
    Yf = np.where(F, Y, np.nan)
    fb = target(Yf[:, fallback], eps0)
    T = np.where(F, target(Yf, eps0), fb[:, None])
    if groups is None:
        groups = np.arange(len(Y))
    g = np.asarray(groups)
    u, inv = np.unique(g, return_inverse=True)
    m = []
    for j in range(T.shape[1]):
        if not F[:, j].any():
            m.append(np.inf)
            continue
        col = T[:, j]
        ok = np.isfinite(col)
        s_ = np.bincount(inv[ok], weights=col[ok], minlength=len(u))
        c_ = np.bincount(inv[ok], minlength=len(u))
        m.append(float(np.mean(s_[c_ > 0] / c_[c_ > 0])) if (c_ > 0).any() else np.inf)
    return [int(j) for j in np.argsort(np.array(m), kind="stable")]


def _first_feasible(rank, f_row) -> int:
    for j in rank:
        if f_row[j]:
            return int(j)
    return -1


class Policy:
    name = "policy"
    abstains = False

    def fit(self, ds, idx) -> "Policy":
        return self

    def choose(self, ds, idx) -> tuple:
        """(chosen protocol index per row, accepted per row)."""
        raise NotImplementedError


class Raw(Policy):
    name = "RAW"

    def choose(self, ds, idx):
        j = ds.protocols.index("RAW")
        return np.full(len(idx), j), np.ones(len(idx), bool)


class SBS(Policy):
    name = "SBS"

    def fit(self, ds, idx):
        idx = np.asarray(idx)
        self.rank = _rank_from_train(ds.Y_sel[idx], ds.F[idx], groups=ds.groups[idx],
                                     fallback=ds.protocols.index("RAW"))
        return self

    def choose(self, ds, idx):
        return np.array([_first_feasible(self.rank, ds.F[i]) for i in idx]), np.ones(len(idx), bool)


class SBSBudget(Policy):
    name = "SBS-B"

    def fit(self, ds, idx):
        idx = np.asarray(idx)
        B = ds.col("B")[idx]
        raw = ds.protocols.index("RAW")
        self.global_rank = _rank_from_train(ds.Y_sel[idx], ds.F[idx], groups=ds.groups[idx], fallback=raw)
        self.rank = {}
        for b in sorted(set(B.tolist())):
            sel = idx[B == b]
            self.rank[int(b)] = _rank_from_train(ds.Y_sel[sel], ds.F[sel], groups=ds.groups[sel], fallback=raw)
        return self

    def rank_for(self, B: int) -> list:
        if int(B) in self.rank:
            return self.rank[int(B)]
        # an unseen budget: the nearest trained budget on the log scale (declared extrapolation rule)
        bs = np.array(sorted(self.rank))
        if not len(bs):
            return self.global_rank
        return self.rank[int(bs[np.argmin(np.abs(np.log10(bs) - math.log10(B)))])]

    def choose(self, ds, idx):
        return np.array([_first_feasible(self.rank_for(ds.rows[i]["B"]), ds.F[i]) for i in idx]), \
            np.ones(len(idx), bool)


class SizeBudget(Policy):
    name = "SIZE-B"
    NAMES = ("n", "two_qubit", "log10_budget")

    def fit(self, ds, idx):
        from sklearn.tree import DecisionTreeClassifier
        idx = np.asarray(idx)
        X = ds.X(self.NAMES)[idx]
        Ys = np.where(ds.F[idx], ds.Y_sel[idx], np.inf)
        y = np.argmin(Ys, 1)
        self.tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(np.nan_to_num(X), y)
        self.fallback = SBSBudget().fit(ds, idx)
        return self

    def choose(self, ds, idx):
        X = np.nan_to_num(ds.X(self.NAMES)[np.asarray(idx)])
        pred = self.tree.predict(X)
        out = []
        for i, j in zip(idx, pred):
            out.append(int(j) if ds.F[i, int(j)] else _first_feasible(self.fallback.rank_for(ds.rows[i]["B"]),
                                                                      ds.F[i]))
        return np.array(out), np.ones(len(idx), bool)


class Model(Policy):
    abstains = False

    def __init__(self, family: str, feature_set: tuple, set_name: str = "", eps0: float = EPS0, seed: int = 0,
                 fixed: dict | None = None):
        self.family, self.features, self.eps0, self.seed, self.fixed = family, tuple(feature_set), eps0, seed, fixed
        self.name = "MODEL[%s|%s]" % (family, set_name or "custom")

    def fit(self, ds, idx):
        idx = np.asarray(idx)
        self.rm = RiskModel(self.family, self.features, tuple(ds.protocols), self.eps0, self.seed,
                            fixed=self.fixed).fit(ds.X(self.features)[idx], ds.Y_sel[idx], ds.F[idx], ds.groups[idx],
                                                  tasks=ds.col("task_id")[idx])
        self.fallback = SBSBudget().fit(ds, idx)
        return self

    def predict(self, ds, idx) -> np.ndarray:
        return self.rm.predict(ds.X(self.features)[np.asarray(idx)])

    def choose(self, ds, idx):
        P = self.predict(ds, idx)
        out = []
        for r, i in enumerate(idx):
            p = np.where(ds.F[i] & np.isfinite(P[r]), P[r], np.inf)
            out.append(int(np.argmin(p)) if np.isfinite(p).any()
                       else _first_feasible(self.fallback.rank_for(ds.rows[i]["B"]), ds.F[i]))
        return np.array(out), np.ones(len(idx), bool)


class ModelAbstain(Model):
    abstains = True

    def __init__(self, family, feature_set, set_name="", alpha: float = 0.1, **kw):
        super().__init__(family, feature_set, set_name, **kw)
        self.alpha = alpha
        self.name = "MODEL+ABSTAIN[%s|%s]" % (family, set_name or "custom")

    def fit(self, ds, idx):
        super().fit(ds, idx)
        self.h = self.rm.conformal_halfwidth(self.alpha, simultaneous=True)
        return self

    def choose(self, ds, idx):
        P = self.predict(ds, idx)
        out, acc = [], []
        for r, i in enumerate(idx):
            fb = _first_feasible(self.fallback.rank_for(ds.rows[i]["B"]), ds.F[i])
            p = np.where(ds.F[i] & np.isfinite(P[r]), P[r], np.inf)
            if not np.isfinite(p).any():
                out.append(fb), acc.append(False)
                continue
            j = int(np.argmin(p))
            if j == fb:
                out.append(j), acc.append(True)
                continue
            hj = self.h.get(ds.protocols[j], math.inf)
            hf = self.h.get(ds.protocols[fb], math.inf)
            ok = bool(np.isfinite(P[r, fb]) and P[r, j] + hj < P[r, fb] - hf)
            out.append(j if ok else fb), acc.append(ok)
        return np.array(out), np.array(acc)


class SelectiveMargin(Model):
    """The selective recommender of 1.7.0: the model's choice when its confidence margin z = (P₂ − P₁)/√(s₁² + s₂²)
    between the two lowest predicted log risks is at least τ (s: each protocol's inner-CV RMSE), otherwise the
    fallback SBS-B (no shots are spent on deciding). τ is chosen by cross-validation on development data and frozen
    (``analysis.nested_selective``); its performance is estimated by the nested procedure, never on the rows it was
    chosen on."""
    abstains = True

    def __init__(self, family, feature_set, set_name="", tau: float = float("inf"), **kw):
        super().__init__(family, feature_set, set_name, **kw)
        self.tau = float(tau)
        self.name = "SELECTIVE[%s|%s]" % (family, set_name or "custom")

    def margins(self, ds, idx) -> np.ndarray:
        P = self.predict(ds, idx)
        z = np.full(len(idx), np.inf)
        for r, i in enumerate(idx):
            feas = [j for j in range(len(ds.protocols)) if ds.F[i, j] and np.isfinite(P[r, j])]
            o = sorted(feas, key=lambda j: P[r, j])
            if len(o) >= 2:
                s1 = self.rm.cv_rmse.get(ds.protocols[o[0]], np.nan)
                s2 = self.rm.cv_rmse.get(ds.protocols[o[1]], np.nan)
                z[r] = float((P[r, o[1]] - P[r, o[0]]) / math.sqrt(s1 ** 2 + s2 ** 2))
        return z

    def choose(self, ds, idx):
        idx = np.asarray(idx)
        ch, _ = super().choose(ds, idx)
        z = self.margins(ds, idx)
        acc = z >= self.tau
        fb = np.array([_first_feasible(self.fallback.rank_for(ds.rows[i]["B"]), ds.F[i]) for i in idx])
        return np.where(acc, ch, fb), acc


class AllCandidate(Policy):
    name = "ALL-CANDIDATE"

    def choose(self, ds, idx):
        out = []
        for i in idx:
            y = np.where(ds.F[i], ds.Y_sel[i], np.inf)
            out.append(int(np.argmin(y)))
        return np.array(out), np.ones(len(idx), bool)


def standard(feature_sets: dict, families=("ridge", "enet", "hgb", "mlp"), abstain_family: str = "hgb",
             abstain_set: str = "deployable", eps0: float = EPS0) -> list:
    """The policy list of the main comparison."""
    pols = [Raw(), SBS(), SBSBudget(), SizeBudget()]
    for fam in families:
        for sname, names in feature_sets.items():
            pols.append(Model(fam, names, sname, eps0=eps0))
    if abstain_set in feature_sets:
        pols.append(ModelAbstain(abstain_family, feature_sets[abstain_set], abstain_set))
    pols.append(AllCandidate())
    return pols
