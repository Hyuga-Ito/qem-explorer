"""1.5.1 regression tests: budget-dependent moments of target-nonlinear estimators (F01), measured-row labels (review
5.1), one explore/confirm split per base circuit across a study (F11) and explicit denominators (F12)."""
import copy
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from qemx import analysis as A
from qemx import budget as BG
from qemx.circuit import parse
from qemx.compile import build
from qemx.methods import Recipe
from qemx.methods.base import MethodResult
from qemx.noise import NoiseSpec
from qemx.observables import z as Zobs
from qemx import recipes as RC
from qemx.recipes import Outcome, compare, summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THETA, P1 = 1.2, 0.15


def _exp_zne(B, on_fail="linear"):
    ideal, outs, _ = compare(build(parse("ry(%r) 0" % THETA, 1)), NoiseSpec(p1=P1), Zobs(0, 1),
                             [Recipe.make("zne", rule="exponential", on_fail=on_fail)], shots=B, seed=13)
    return ideal, outs[0]


def _independent(B, split, R, seed, policy="fallback"):
    """Complete experiments along an independent path: analytic E(λ) = cos θ (1 − p)^k, binomial counts, lstsq fits."""
    lams = np.array([1.0, 3.0, 5.0])
    E = math.cos(THETA) * (1 - P1) ** lams
    rng = np.random.default_rng(seed)
    Eh = np.stack([2 * rng.binomial(n, (1 + e) / 2, size=R) / n - 1 for n, e in zip(split, E)], 1)
    ok = np.all(Eh > 0, axis=1)                                   # sign of E(1) is + here with overwhelming odds
    Amat = np.stack([np.ones(3), lams], 1)
    est = np.full(R, np.nan)
    est[ok] = np.exp(np.linalg.lstsq(Amat, np.log(Eh[ok]).T, rcond=None)[0][0])
    if policy == "fallback":
        est[~ok] = np.linalg.lstsq(Amat, Eh[~ok].T, rcond=None)[0][0]
    return est, ok


# ============================================================ F01: nonlinear ZNE
def test_exponential_zne_reports_budget_dependent_protocol_moments():
    ideal, o = _exp_zne(300)
    r = o.result
    assert r.estimate_kind == "reference" and r.reference_mean == pytest.approx(r.mean)
    assert r.reference_mean == pytest.approx(ideal, abs=1e-9)        # E(λ) is exactly exponential here
    row = summary(o, ideal, 300, 1.0)
    assert row["estimate kind"] == "protocol" and "budget-dependent-moments" in row["flags"]
    assert row["exact-fit value"] == pytest.approx(ideal, abs=1e-9)
    # the finite-shot protocol is biased upward (the log-fit) — 1.5.0 reported bias 0 here
    assert row["bias"] > 3 * row["bias MC ±"] > 0
    tm = RC.target_protocol(r, BG.account(r, 300))
    est, ok = _independent(300, BG.optimal_integer_split(r.details["a_circuits"], 300)[1], 20000, 7)
    se = math.sqrt(tm.mean_mcse ** 2 + np.var(est, ddof=1) / len(est))
    assert abs(tm.mean - est.mean()) <= 4 * se
    assert row["sd"] == pytest.approx(np.std(est, ddof=1), rel=0.06)
    assert row["fit failures"] == pytest.approx(1 - ok.mean(), abs=0.02)
    assert row["failure policy"] == "fallback"


def test_failure_policies_are_complete_and_labeled():
    ideal, o = _exp_zne(100, "linear")
    d = o.result.sampler(100, 4000, np.random.default_rng(0))
    assert np.isfinite(d).all()                                    # fallback: every experiment has an estimate
    row = summary(o, ideal, 100, 1.0, d)
    assert row["fit failures"] > 0.2 and "conditional-on-success" not in row["flags"]
    ideal, o2 = _exp_zne(100, "abstain")
    d2 = o2.result.sampler(100, 4000, np.random.default_rng(0))
    assert (~np.isfinite(d2)).mean() > 0.2                          # abstain: no estimate, counted
    row2 = summary(o2, ideal, 100, 1.0, d2)
    assert "conditional-on-success" in row2["flags"] and "abstains in" in row2["status"]
    assert row2["failure policy"] == "abstain"
    assert "abstain on failure" in o2.recipe.title() and "abstain" not in o.recipe.title()


def test_on_fail_is_a_sparse_parameter():
    """Only the exponential rule carries the new parameter: other ZNE recipes keep their 1.5.0 identity."""
    assert "on_fail" not in Recipe.make("zne").p and "on_fail" not in Recipe.make("zne", rule="linear").p
    assert Recipe.make("zne", rule="exponential").p["on_fail"] == "linear"
    assert "on_fail" not in Recipe.make("zne", rule="richardson", on_fail="abstain").p
    assert "zne_on_fail" not in Recipe.make("chain").p
    assert Recipe.make("chain", zne_rule="exponential").p["zne_on_fail"] == "linear"


def test_vectorized_exponential_rule_matches_the_scalar_one():
    from qemx.methods.zne import extrapolate, extrapolate_many, extrapolate_policy
    rng = np.random.default_rng(3)
    lams = [1.0, 2.0, 3.5]
    Em = rng.normal(0.2, 0.25, size=(400, 3))
    for on_fail in ("linear", "abstain"):
        est, failed = extrapolate_many("exponential", lams, Em, on_fail)
        for e, f, row in zip(est, failed, Em):
            v, fl = extrapolate_policy("exponential", lams, row, on_fail)
            assert fl == f and (math.isnan(v) and math.isnan(e) or v == pytest.approx(e, rel=1e-10, abs=1e-12))
            assert f == (not np.isfinite(extrapolate("exponential", lams, row)))
    assert failed.any() and (~failed).any()


def test_budget_curves_use_budget_dependent_moments():
    ideal, o = _exp_zne(300)
    row = summary(o, ideal, 300, 1.0)
    g = A.target_grid(row)
    assert g is not None and g["first_order_from"] is not None
    p300 = A.row_point(row, 300)
    assert p300["source"] in ("monte-carlo", "interpolated")
    # the curve (averaged over adjacent budgets: integer-count lattice effects) agrees with the table at its B
    # within the recorded lattice spread of the SD plus 5 %
    assert abs(p300["rmse"] - row["RMSE"]) <= 0.5 * p300["sd_range"] + 0.05 * row["RMSE"]
    assert p300["sd_range"] > 0.01                                 # the lattice effect is real and recorded
    first = math.sqrt((o.result.reference_mean - ideal) ** 2 + o.result.reference_var1 / 300)
    assert p300["rmse"] > 1.1 * first                               # the first-order curve was optimistic here
    big = 4 * g["first_order_from"]
    pb = A.row_point(row, big)
    assert pb["source"] == "first-order (validated)"
    assert pb["rmse"] == pytest.approx(math.sqrt(o.result.reference_var1 / big), rel=1e-9)
    assert A.rmse_row(row, [300])[0] == pytest.approx(p300["rmse"])
    assert A.budget_row(row, 0.05) > 0 and math.isfinite(A.budget_row(row, 0.05))


def test_moment_grid_validates_first_order_for_a_linear_estimator():
    from qemx.stats import grid_at, moment_grid

    def draw(N, R, rng):                                           # a linear estimator: unbiased, var 1/N
        return rng.normal(0.5, 1 / math.sqrt(N), R), np.zeros(R, bool)
    g = moment_grid(draw, 2, 0.5, 1.0, np.random.default_rng(1), "fallback")
    assert g["first_order_from"] is not None and g["first_order_from"] <= 10
    assert grid_at(g, 10 ** 6)["source"] == "first-order (validated)"
    assert grid_at(g, 1)["source"] == "infeasible"


def test_symmetry_verification_reports_its_finite_shot_ratio_moments():
    from qemx import templates as T
    from qemx import observables as OBS
    tpl = T.Template(parse("h 0; cx 0 1", 2), OBS.z_string([0, 1], 2), "")
    ideal, outs, _ = compare(build(tpl.spec), NoiseSpec(p2=0.3), tpl.observable,
                             [Recipe.make("symmetry", qubits=[0, 1], eigenvalue=1)], shots=40, seed=2)
    r = outs[0].result
    assert r.estimate_kind == "reference" and r.failure_policy == "abstain" and r.target_sampler is not None
    row = summary(outs[0], ideal, 40, 1.0)
    assert row["estimate kind"] == "protocol" and row["failure policy"] == "abstain"
    assert 0.0 <= row["fit failures"] < 0.05


def test_hamiltonian_settings_split_the_budget_exactly():
    from qemx import hamiltonian as HM
    seen = []

    def mk(v):
        def s(N, B, rng):
            seen.append(N)
            return np.zeros(B)
        return MethodResult(0.0, v, sampler=s)
    ps = HM.parse_pauli_sum("Z0 + X0 + 0.5*Y0", 1)
    parts = [({}, [0], mk(1.0)), ({}, [1], mk(2.0)), ({}, [2], mk(3.0))]
    r = HM.combine(ps, parts, 101)
    for N in (3, 7, 101, 1000):
        seen.clear()
        r.sampler(N, 2, np.random.default_rng(0))
        assert sum(seen) == N
    assert sum(d["shots"] for d in r.details["hamiltonian"]) == 101


# ============================================================ review 5.1: measured rows
def test_measured_rows_are_signed_and_absolute_errors():
    r = MethodResult(0.4, 1.0, estimate_kind="measured", flags=("measured",))
    row = summary(Outcome(Recipe.make("raw"), r), 0.5, 1000, 1.0)
    assert row["signed error to reference"] == pytest.approx(-0.1) and row["absolute error"] == pytest.approx(0.1)
    df = RC.measured_view(pd.DataFrame([row]))
    assert "bias" not in df and "RMSE" not in df and "bootstrap sd" in df
    assert df["signed error to reference"].iloc[0] == pytest.approx(-0.1)
    sim = summary(Outcome(Recipe.make("raw"), MethodResult(0.4, 1.0)), 0.5, 1000, 1.0)
    assert list(RC.measured_view(pd.DataFrame([sim])).columns) == list(pd.DataFrame([sim]).columns)


# ============================================================ F11: one split per base circuit
def _pilot():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "manifests", "direction_A_v2_pilot.yaml"), encoding="utf-8"))


def _sides(m):
    from qemx.phase import manifest as M
    split = M.confirmatory_split(m)
    out = {}
    for c in M.expand(m):
        out.setdefault(M.base_circuit_key(c.family, c.n, c.fparams), set()).add(split[c.cell_id])
    return split, out


def test_a_second_stage_cannot_put_a_base_circuit_on_both_sides():
    m = _pilot()
    st = dict(m["stages"][0], name="P-extra", scales=[0.75, 1.5])
    st.pop("confirmatory")
    m["stages"].append(st)
    _, sides = _sides(m)
    assert len(sides) == 25 and all(len(v) == 1 for v in sides.values())
    assert sum(v == {"confirm"} for v in sides.values()) == 7


def test_conflicting_stage_policies_are_rejected_before_a_run():
    from qemx.phase import manifest as M
    m = _pilot()
    st = dict(m["stages"][0], name="P-extra", scales=[0.75], confirmatory={"fraction": 0.25, "seed": 7})
    m["stages"].append(st)
    with pytest.raises(ValueError, match="conflicting confirmatory policies"):
        M.check(m)
    m2 = copy.deepcopy(_pilot())
    m2["confirmatory"] = {"fraction": 0.3, "seed": 2026}
    with pytest.raises(ValueError, match="conflict with the study-level policy"):
        M.check(m2)


def test_the_frozen_pilot_split_is_unchanged():
    """The 1.5.0 stage-level hash, recomputed here, gives the same split as the study-level policy."""
    import hashlib
    from qemx.phase import manifest as M
    m = _pilot()
    split, _ = _sides(m)
    conf = m["stages"][0]["confirmatory"]
    for c in M.expand(m):
        key = json.dumps([c.family, c.n, list(c.fparams)], sort_keys=True, default=str)
        h = int(hashlib.sha256(("confirm|%s|%s" % (conf.get("seed", 0), key)).encode()).hexdigest()[:8], 16)
        assert split[c.cell_id] == ("confirm" if h / 0xFFFFFFFF < conf["fraction"] else "explore")


def test_check_split_detects_leakage_and_canonical_identity():
    from qemx.phase import manifest as M
    k1 = M.base_circuit_key("ising", 4, (("T", 1.0), ("steps", 2)))
    assert k1 == M.base_circuit_key("ising", 4, [["T", 1], ["steps", 2.0]])
    with pytest.raises(ValueError, match="both the exploration and the confirmatory"):
        M.check_split([("a", k1), ("b", k1)], {"a": "explore", "b": "confirm"})
    M.check_split([("a", k1), ("b", k1)], {"a": "confirm", "b": "confirm"})


# ============================================================ F12: denominators
def test_improvement_shares_state_their_denominators():
    from qemx import study as SD
    W = pd.DataFrame([dict(cell_id="c1", B=1000, group="g1", rmse_RAW=0.10, feas_RAW=True, rmse_X=0.05, feas_X=True),
                      dict(cell_id="c2", B=1000, group="g2", rmse_RAW=0.10, feas_RAW=True, rmse_X=np.inf,
                           feas_X=False)])
    r = SD.improvements(W, ["RAW", "X"], 1000).iloc[0]
    assert (r["planned"], r["feasible"], r["infeasible"], r["better"]) == (2, 1, 1, 1)
    assert r["share_better_all"] == pytest.approx(0.5)
    assert r["share_better_feasible"] == pytest.approx(1.0) and r["feasible_fraction"] == pytest.approx(0.5)
