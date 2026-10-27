"""Focused checks of the 1.5.1 recommender-evaluation statistics (review 5.8, 6.8): the group-weighted conformal
quantile, group folds, the coverage of group-CV conformal intervals on new base circuits, the base-circuit bootstrap of
paired comparisons, the full policy's fallback, and the Monte-Carlo stability of regime labels."""
import math

import numpy as np
import pandas as pd
import pytest

from qemx import recommend as RC
from qemx import study as SY


def test_weighted_quantile_weighs_every_base_circuit_equally():
    # group A: three rows (weight 1/3 each); group B: four rows (weight 1/4 each) -> each group carries half
    x = [1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0]
    w = [1 / 3] * 3 + [1 / 4] * 4
    assert RC.weighted_quantile(x, w, 0.5) == 3.0
    assert RC.weighted_quantile(x, w, 0.51) == 10.0
    assert RC.weighted_quantile(x, w, 1.0) == 10.0
    assert RC.weighted_quantile(x, w, 1.01) == math.inf          # too few calibration groups for the level
    assert RC.weighted_quantile([math.nan, 2.0], [1, 1], 0.5) == 2.0


def test_group_folds_hold_out_whole_base_circuits_once():
    g = np.repeat(["c%d" % k for k in range(11)], 4)
    folds = RC.group_folds(g, k=5, seed=3)
    seen = np.concatenate([f.test for f in folds])
    assert sorted(seen.tolist()) == list(range(len(g)))           # every row is tested exactly once
    RC.check_no_leakage(folds, g)
    bad = [RC.Fold(np.array([0, 4]), np.array([1]), "bad")]       # rows 0 and 1 share base circuit c0
    with pytest.raises(AssertionError):
        RC.check_no_leakage(bad, g)


def _grouped_data(rng, G, per, slope=(0.5, -0.2), u_sd=0.3, e_sd=0.05):
    g = np.repeat(np.arange(G), per)
    X = rng.normal(size=(G * per, 2))
    u = rng.normal(0, u_sd, G)[g]                                 # base-circuit effect shared by its rows
    ya = X @ np.asarray(slope) + u + rng.normal(0, e_sd, G * per)
    Y = np.c_[ya + 1.0, ya]                                       # RAW, A
    return X, Y, np.ones_like(Y, bool), g


def test_group_cv_conformal_intervals_cover_new_base_circuits_near_the_nominal_level():
    """Marginal coverage over new, exchangeable base circuits: (1 − α)(1 + 1/G) quantile of out-of-fold residuals,
    every base circuit weighted equally -> close to 90 % (finite G makes it slightly conservative)."""
    rng = np.random.default_rng(11)
    cov = []
    for _ in range(30):
        X, Y, F, g = _grouped_data(rng, G=40, per=5)
        rec = RC.RidgeRecommender(["RAW", "A"], alpha=1.0).fit(X, Y, F, groups=g)
        assert rec.interval_kind.startswith("group-CV conformal") and rec.calibration_groups == 40
        Xn, Yn, _, gn = _grouped_data(rng, G=200, per=1)
        pred = rec.predict(Xn)[:, 1]
        cov.append(np.mean(np.abs(Yn[:, 1] - pred) <= rec.q["A"]))
    assert 0.86 <= float(np.mean(cov)) <= 0.97, np.mean(cov)


def test_without_enough_groups_the_interval_is_labeled_in_sample_gaussian():
    rng = np.random.default_rng(2)
    X, Y, F, g = _grouped_data(rng, G=2, per=10)
    rec = RC.RidgeRecommender(["RAW", "A"]).fit(X, Y, F, groups=g)
    assert rec.interval_kind == "gaussian (in-sample)"
    assert rec.q["A"] == pytest.approx(1.645 * rec.res_sd["A"])


def test_paired_interval_resamples_whole_base_circuits_not_rows():
    """Copies of a base circuit's rows add no information: ten identical rows per base circuit give the same
    estimate and interval as one row per base circuit."""
    rng = np.random.default_rng(5)
    d = rng.normal(0.1, 0.05, 20)                                 # per base circuit log10 ratio
    one = [dict(group="c%d" % k, a=10 ** dk - 1, b=0.0) for k, dk in enumerate(d)]
    ten = [dict(r) for r in one for _ in range(10)]
    e1, lo1, hi1, n1 = RC.paired_ci(one, "a", "b", reps=1500, seed=4)
    e2, lo2, hi2, n2 = RC.paired_ci(ten, "a", "b", reps=1500, seed=4)
    assert (n1, n2) == (20, 200)
    assert e1 == pytest.approx(float(np.mean(d))) and e2 == pytest.approx(e1)
    assert (lo2, hi2) == pytest.approx((lo1, hi1), abs=1e-12)
    assert lo1 < 0.1 < hi1 and lo1 > 0                            # the true mean ratio is inside; a decided sign
    nog = [dict(a=r["a"], b=r["b"]) for r in one]                 # rows without a group are their own groups
    assert RC.paired_ci(nog, "a", "b", reps=1500, seed=4)[:3] == pytest.approx((e1, lo1, hi1))


def test_full_policy_has_a_choice_and_a_regret_on_every_evaluated_row():
    rng = np.random.default_rng(8)
    X, Y, F, g = _grouped_data(rng, G=30, per=4)
    Y = np.c_[Y, Y[:, 1] + rng.normal(0, 0.02, len(Y))]          # B: a near-tie of A -> some abstentions
    F = np.ones_like(Y, bool)
    F[::7, 2] = False                                             # B cannot run on some rows
    folds = RC.group_folds(g, k=5, seed=1)
    ev = RC.evaluate(X, Y, F, ["RAW", "A", "B"], folds, groups=g)
    rows = ev["rows"]
    assert len(rows) == len(Y)
    assert any(r["chosen"] is None for r in rows)                 # the model abstains somewhere ...
    for r in rows:                                                # ... the full policy never does
        assert r["policy"] is not None and math.isfinite(r["regret_policy"]) and r["regret_policy"] >= 0
        if r["chosen"] is None:
            assert r["policy"] == r["fixed"]
        j = ["RAW", "A", "B"].index(r["policy"])
        assert F[r["row"], j]                                     # never an infeasible method
    assert all(r["regret"] >= 0 for r in rows if r["chosen"] is not None)


def _W(rmse_a, rmse_b, feas_b=True):
    return pd.DataFrame([dict(cell_id="c", B=1000, rmse_A=rmse_a, rmse_B=rmse_b, feas_A=True, feas_B=feas_b)])


def _E(se_a, se_b):
    return pd.DataFrame([dict(cell_id="c", B=1000, method="A", rmse_mc_se=se_a),
                         dict(cell_id="c", B=1000, method="B", rmse_mc_se=se_b)])


def test_winner_stability_separates_resolved_winners_from_monte_carlo_coin_flips():
    clear = SY.winner_stability(_W(1.0, 2.0), _E(0.01, 0.02), ["A", "B"], draws=4000).iloc[0]
    assert clear.p_winner == pytest.approx(1.0) and clear.p_in_tie == pytest.approx(1.0)
    flip = SY.winner_stability(_W(1.0, 1.0001), _E(0.1, 0.1), ["A", "B"], draws=20000).iloc[0]
    assert flip.p_winner == pytest.approx(0.5, abs=0.02)
    assert flip.p_in_tie > flip.p_winner                          # a practical tie is a weaker statement
    exact = SY.winner_stability(_W(1.0, 1.0001), _E(0.0, 0.0), ["A", "B"]).iloc[0]
    assert exact.p_winner == 1.0 and exact.mc_max_rel_se == 0.0   # no MC error: the label is exact
    alone = SY.winner_stability(_W(1.0, 0.5, feas_b=False), _E(0.1, 0.1), ["A", "B"]).iloc[0]
    assert alone.p_winner == pytest.approx(1.0)                   # an infeasible method never wins
