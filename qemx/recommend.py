r"""A transparent baseline QEM recommender and its honest evaluation (1.5).

Model. For every method m a ridge regression predicts log10 RMSE_m(B) from recommendation-time features (circuit,
observable, noise model, budget; never oracle features — ``qemx.features``). Inputs are standardized and the ridge
penalty is fixed **inside each training fold** (no information from the test fold). A method that cannot run at
the budget (its fixed cost exceeds B, known from the protocol before any execution) is never recommended.

Decision with abstention. Let m̂ be the method with the smallest predicted log RMSE, σ the residual standard
deviation of the methods' predictions (from the training fold) and k = ``z`` (default 1.0):

1. **outside coverage** — some input lies outside the training range by more than ``coverage_z`` training
   standard deviations: abstain;
2. **use raw / no demonstrated benefit** — the predicted gain of m̂ over raw is smaller than the practical-
   equivalence margin δ (log10(1+δ)) plus k·σ_diff: recommend raw (a legitimate outcome, not a failure);
3. **uncertain** — the runner-up is within k·σ_diff of m̂: abstain and report both;
4. otherwise recommend m̂ with a predicted interval for its log RMSE.

Evaluation. Held-out folds group every row of a *base circuit* (family, n, parameters: all noise scales and budgets
of it) together — random row splits would put near-duplicates on both sides. Also leave-one-family-out and
leave-width-out (train on smaller widths, test on the largest: the test of whether normalized features carry across
widths). Reported: regret RMSE_chosen / RMSE_best − 1, accuracy within the practical-equivalence set, coverage
(share not abstained), selective risk versus coverage, and the coverage of the predicted intervals. Baselines:
always raw, the best fixed method (chosen on the training fold) and a literature-motivated rule (PEC's sampling cost
grows as e^{4λ}, Cai et al. RMP 95, 045005, Eq. 18). Feature weights are associations, not causes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DELTA = 0.10                     # practical equivalence: RMSE within 10 % of the best is a tie
RAW = "RAW"


@dataclass
class Fold:
    train: np.ndarray
    test: np.ndarray
    name: str


@dataclass
class Recommendation:
    method: str | None
    kind: str                     # "method" | "raw" | "abstain-coverage" | "abstain-uncertain" | "no-feasible"
    predicted: dict = field(default_factory=dict)      # method -> predicted log10 RMSE
    interval: tuple = (math.nan, math.nan)
    alternatives: tuple = ()
    reason: str = ""


class RidgeRecommender:
    def __init__(self, methods, alpha: float = 1.0, z: float = 1.0, coverage_z: float = 3.0, delta: float = DELTA):
        self.methods, self.alpha, self.z, self.coverage_z, self.delta = list(methods), float(alpha), float(z), \
            float(coverage_z), float(delta)

    def fit(self, X, Y, feasible):
        """X (rows, features), Y (rows, methods) log10 RMSE (NaN where infeasible), feasible (rows, methods)."""
        X = np.asarray(X, float)
        ok = np.all(np.isfinite(X), axis=1)
        X = X[ok]
        Y, F = np.asarray(Y, float)[ok], np.asarray(feasible, bool)[ok]
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        self.lo, self.hi = X.min(0), X.max(0)
        Z = (X - self.mu) / self.sd
        A = np.c_[np.ones(len(Z)), Z]
        P = self.alpha * np.eye(A.shape[1])
        P[0, 0] = 0.0
        self.W, self.res_sd = {}, {}
        for j, m in enumerate(self.methods):
            use = F[:, j] & np.isfinite(Y[:, j])
            if use.sum() < A.shape[1] + 2:
                continue
            w = np.linalg.solve(A[use].T @ A[use] + P, A[use].T @ Y[use, j])
            r = Y[use, j] - A[use] @ w
            dof = max(int(use.sum()) - A.shape[1], 1)
            self.W[m] = w
            self.res_sd[m] = float(math.sqrt(float(r @ r) / dof))
        return self

    def predict(self, X) -> np.ndarray:
        Z = (np.atleast_2d(np.asarray(X, float)) - self.mu) / self.sd
        A = np.c_[np.ones(len(Z)), Z]
        return np.column_stack([A @ self.W[m] if m in self.W else np.full(len(Z), np.nan) for m in self.methods])

    def recommend(self, x, feasible) -> Recommendation:
        x = np.asarray(x, float)
        if not np.all(np.isfinite(x)):
            return Recommendation(None, "abstain-coverage", reason="a required feature is missing")
        span = (self.hi - self.lo)
        far = (x < self.lo - self.coverage_z * self.sd) | (x > self.hi + self.coverage_z * self.sd)
        if far.any() or np.any((span == 0) & (x != self.lo)):
            return Recommendation(None, "abstain-coverage", reason="outside the training data's range")
        pred = self.predict(x)[0]
        cand = [(pred[j], m) for j, m in enumerate(self.methods) if feasible[j] and np.isfinite(pred[j])]
        if not cand:
            return Recommendation(None, "no-feasible", reason="no method can run at this budget")
        cand.sort()
        best, mb = cand[0]
        sig = lambda m: self.res_sd.get(m, math.inf)              # noqa: E731
        d = {m: float(p) for p, m in cand}
        tie = math.log10(1 + self.delta)
        if RAW in d and mb != RAW:
            sd_diff = math.sqrt(sig(mb) ** 2 + sig(RAW) ** 2)
            if d[RAW] - best < tie + self.z * sd_diff:
                return Recommendation(RAW, "raw", d, (d[RAW] - 1.645 * sig(RAW), d[RAW] + 1.645 * sig(RAW)),
                                      (mb,), "no demonstrated benefit over raw at this budget")
        if len(cand) > 1:
            p2, m2 = cand[1]
            sd_diff = math.sqrt(sig(mb) ** 2 + sig(m2) ** 2)
            if p2 - best < self.z * sd_diff and m2 != RAW and mb != RAW:
                return Recommendation(None, "abstain-uncertain", d, alternatives=(mb, m2),
                                      reason="%s and %s are not distinguishable by the model" % (mb, m2))
        return Recommendation(mb, "method" if mb != RAW else "raw", d,
                              (best - 1.645 * sig(mb), best + 1.645 * sig(mb)), tuple(m for _, m in cand[1:3]))


# ------------------------------------------------------------------ baselines
def best_fixed(Y_train, F_train, methods) -> str:
    """The method with the smallest mean log10 RMSE over the training rows where it is feasible."""
    Y = np.where(np.asarray(F_train, bool), np.asarray(Y_train, float), np.nan)
    means = [np.nanmean(Y[:, j]) if np.isfinite(Y[:, j]).any() else np.inf for j in range(len(methods))]
    return methods[int(np.argmin(means))]


def rule_based(lam_lc: float, budget: float, feasible: dict) -> str:
    """Transparent heuristic: negligible noise → raw; PEC when its e^{4λ} sampling overhead (Cai et al. Eq. 18)
    leaves at least 10⁴ effective shots; else Richardson ZNE; raw when nothing else is feasible."""
    if not math.isfinite(lam_lc) or lam_lc < 0.01:
        return RAW
    if feasible.get("PEC") and budget * math.exp(-4 * lam_lc) >= 1e4:
        return "PEC"
    for m in ("ZNE-R", "ZNE-L"):
        if feasible.get(m):
            return m
    return RAW


# ------------------------------------------------------------------ folds
def group_folds(groups, k: int = 5, seed: int = 0) -> list:
    """k folds of whole groups (base circuits), groups assigned by a seeded permutation."""
    g = np.asarray(groups)
    uniq = np.array(sorted(set(g.tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    out = []
    for f in range(min(k, len(uniq))):
        test_g = set(uniq[perm[f::min(k, len(uniq))]].tolist())
        te = np.flatnonzero([x in test_g for x in g])
        tr = np.flatnonzero([x not in test_g for x in g])
        out.append(Fold(tr, te, "group fold %d" % (f + 1)))
    return out


def leave_one_out(labels, name: str) -> list:
    lab = np.asarray(labels)
    return [Fold(np.flatnonzero(lab != v), np.flatnonzero(lab == v), "%s = %s held out" % (name, v))
            for v in sorted(set(lab.tolist()))]


def check_no_leakage(folds, groups) -> None:
    g = np.asarray(groups)
    for f in folds:
        inter = set(g[f.train].tolist()) & set(g[f.test].tolist())
        if inter:
            raise AssertionError("base circuits on both sides of %s: %s" % (f.name, sorted(inter)[:3]))


# ------------------------------------------------------------------ evaluation
def evaluate(X, Y, F, methods, folds, lam=None, budget=None, alpha: float = 1.0, z: float = 1.0,
             delta: float = DELTA) -> dict:
    """Out-of-fold decisions of the recommender and the baselines, with summary metrics."""
    X, Y, F = np.asarray(X, float), np.asarray(Y, float), np.asarray(F, bool)
    Ytrue = np.where(F, Y, np.inf)                       # infeasible: infinite RMSE
    rows = []
    for fold in folds:
        if len(fold.train) == 0 or len(fold.test) == 0:
            continue
        rec = RidgeRecommender(methods, alpha, z, delta=delta).fit(X[fold.train], Y[fold.train], F[fold.train])
        fixed = best_fixed(Y[fold.train], F[fold.train], methods)
        for i in fold.test:
            best = float(np.min(Ytrue[i]))
            if not math.isfinite(best):
                continue
            tie = {m for j, m in enumerate(methods) if Ytrue[i, j] <= best + math.log10(1 + delta)}
            r = rec.recommend(X[i], F[i])
            chosen = r.method
            def regret(m):
                if m is None:
                    return math.nan
                j = methods.index(m)
                return 10 ** (Ytrue[i, j] - best) - 1 if math.isfinite(Ytrue[i, j]) else math.inf
            pred_j = methods.index(chosen) if chosen in methods else None
            inside = (r.interval[0] <= Ytrue[i, pred_j] <= r.interval[1]) if pred_j is not None and \
                math.isfinite(r.interval[0]) else None
            fx = fixed if F[i, methods.index(fixed)] else RAW
            rb = rule_based(lam[i] if lam is not None else math.nan, budget[i] if budget is not None else 1e5,
                            {m: bool(F[i, j]) for j, m in enumerate(methods)})
            rows.append(dict(row=int(i), fold=fold.name, kind=r.kind, chosen=chosen, regret=regret(chosen),
                             in_tie=(chosen in tie) if chosen else None, interval_hit=inside,
                             margin=_margin(r), raw_in_tie=RAW in tie, regret_raw=regret(RAW),
                             regret_fixed=regret(fx), regret_rule=regret(rb), oracle=methods[int(np.argmin(Ytrue[i]))]))
    return dict(rows=rows, summary=summarize(rows))


def _margin(r: Recommendation) -> float:
    if not r.predicted or r.method is None:
        return -math.inf
    vals = sorted(r.predicted.values())
    return float(vals[1] - vals[0]) if len(vals) > 1 else math.inf


def summarize(rows: list) -> dict:
    if not rows:
        return {}
    n = len(rows)
    dec = [r for r in rows if r["chosen"] is not None]
    fin = lambda xs: [x for x in xs if x is not None and math.isfinite(x)]     # noqa: E731
    def med(xs):
        xs = fin(xs)
        return float(np.median(xs)) if xs else math.nan
    def mean(xs):
        xs = fin(xs)
        return float(np.mean(xs)) if xs else math.nan
    hits = [r["interval_hit"] for r in dec if r["interval_hit"] is not None]
    raw_rows = [r for r in rows if r["kind"] == "raw"]
    return dict(rows=n, coverage=len(dec) / n, abstain_coverage=sum(r["kind"] == "abstain-coverage" for r in rows) / n,
                abstain_uncertain=sum(r["kind"] == "abstain-uncertain" for r in rows) / n,
                recommend_raw=len(raw_rows) / n,
                raw_correct=(sum(r["raw_in_tie"] for r in raw_rows) / len(raw_rows)) if raw_rows else math.nan,
                accuracy=(sum(bool(r["in_tie"]) for r in dec) / len(dec)) if dec else math.nan,
                regret_median=med([r["regret"] for r in dec]), regret_mean=mean([r["regret"] for r in dec]),
                interval_coverage=(sum(hits) / len(hits)) if hits else math.nan,
                baseline_raw_regret_median=med([r["regret_raw"] for r in rows]),
                baseline_fixed_regret_median=med([r["regret_fixed"] for r in rows]),
                baseline_rule_regret_median=med([r["regret_rule"] for r in rows]),
                recommender_vs_fixed_on_decided=med([r["regret"] for r in dec]) - med([r["regret_fixed"] for r in dec])
                if dec else math.nan)


def risk_coverage(rows: list) -> list:
    """(coverage, selective median regret) as the decided rows are admitted from most to least confident."""
    dec = sorted([r for r in rows if r["chosen"] is not None and r["regret"] is not None
                  and math.isfinite(r["regret"])], key=lambda r: -r["margin"])
    out = []
    for k in range(1, len(dec) + 1):
        out.append((k / len(rows), float(np.median([r["regret"] for r in dec[:k]]))))
    return out
