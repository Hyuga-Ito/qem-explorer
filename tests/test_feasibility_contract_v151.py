"""Execution budgets 1.5.1 — one feasibility contract (review F10).

The interactive cost model (qemx.budget) and the phase ledger (qemx.phase.ledger, fed by phase.cell.cost_of) use one
feasibility function: post-selection needs ⌈5/P_accept⌉ target shots, PEC with s shots per sampled circuit needs s,
calibration circuits need cal_min shots each. Small accepted counts are validated against exact sampling.
(The invariant over the hardware planner as well is in tests/test_hardware_budget_v151.py.)
"""
import math

import numpy as np
import pytest

from qemx import budget as BG
from qemx import observables as OBS
from qemx.circuit import parse
from qemx.compile import build
from qemx.methods import REGISTRY, Recipe, Runner
from qemx.methods.base import MethodResult
from qemx.noise import NoiseSpec


def test_review_postselection_example_agrees_across_cost_systems():
    from qemx.phase import ledger as L
    from qemx.phase.cell import MethodSpec, _method_row, cost_of
    res = MethodResult(mean=0.5, var1=1.0, circuits=1, accept=0.01)
    row = _method_row(MethodSpec.make("SV", "symmetry", qubits=[0, 1]), Recipe.make("symmetry", qubits=(0, 1)),
                      res, 0.4)
    cost = cost_of(row)
    assert cost.accept == 0.01
    assert BG.min_budget(BG.CostModel.of(res)) == L.min_budget(cost) == 500
    assert not L.allocate(cost, 499).feasible and L.allocate(cost, 500).feasible
    # records written before 1.5.1: a Cost without the new fields still loads (no post-selection, one shot each)
    d = cost.to_json()
    d.pop("accept")
    d.pop("per_circuit")
    old = L.Cost.from_json(d)
    assert old.accept == 1.0 and old.per_circuit == 1 and L.min_budget(old) == 1


GHZ6 = "h 0; cx 0 1; cx 1 2; cx 2 3; cx 3 4; cx 4 5"


HEAVY = NoiseSpec(p1=0.02, p2=0.12, ro01=0.15, ro10=0.2)


def test_symmetry_small_accepted_counts_are_validated_against_exact_sampling():
    """At the feasibility boundary N = ⌈5/P⌉ the ratio estimator's failure probability and its conditional variance
    (given at least one kept shot) are what budget.postselection_check computes exactly: the delta-method variance
    is optimistic there by about 25–30 %, which the comparison table's repeats reveal."""
    from qemx.methods.base import Context as Ctx
    tpl_c = build(parse(GHZ6, 6))
    rc = Recipe.make("symmetry", qubits=(0, 1), more="1 2; 2 3; 3 4; 4 5")
    ctx = Ctx(tpl_c, HEAVY, OBS.z(0, 6), rc, Runner(), shots=4096)
    r = REGISTRY["symmetry"].run(ctx, **rc.p)
    P = r.accept
    N = BG.CostModel.of(r).n_min
    assert N == math.ceil(5 / P)
    chk = BG.postselection_check(int(N), P)
    x = r.sampler(int(N), 40000, np.random.default_rng(11))
    ok = np.isfinite(x)
    assert abs((1 - ok.mean()) - chk["failure_prob"]) < 4 * math.sqrt(chk["failure_prob"] / 40000) + 1e-4
    infl = float(np.var(x[ok], ddof=1)) / (r.var1 / N)
    assert 1.15 < chk["var_inflation"] < 1.35
    assert infl == pytest.approx(chk["var_inflation"], rel=0.05)
    assert BG.postselection_check(10 ** 6, 0.3)["var_inflation"] == pytest.approx(1.0, abs=1e-5)


def test_cost_model_and_ledger_share_the_calibration_minimum():
    from qemx.phase import ledger as L
    rule = BG.Feasibility(cal_min=32)
    m = BG.CostModel(v=1.0, fixed=200, c=2, w=0.5, circuits=3, rule=rule)
    assert BG.min_budget(m) == 200 + 64 + 3
    N, s = BG.split(m, 267)
    assert s == 32 and N == 3
    assert BG.split(m, 266) == (0.0, 0.0)
    cost = L.Cost("x", b=0.0, v=1.0, w=0.5, c=2, T=10, s_tr=20, m=3)
    assert L.min_budget(cost, L.Policy("fixed", True, 32, 1)) == 267
    # PEC with s shots per sampled circuit needs one complete circuit
    pec = MethodResult(0.0, 1.0, circuits=1, shots_per_circuit=8)
    assert BG.min_budget(BG.CostModel.of(pec)) == 8 and BG.split(BG.CostModel.of(pec), 7) == (0.0, 0.0)
    assert L.min_budget(L.Cost("pec", b=0.0, v=1.0, per_circuit=8)) == 8
