"""1.8.0: regressions for the findings of the independent review of the 1.8.0 policy-evaluation code (margin runner-up,
study seeds under seed scheme 2, the model-component benchmark's rows, missing risks in tail metrics, the
allocation-level width guard)."""
import json
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = json.load(open(os.path.join(ROOT, "tests", "data", "deployment_regression_cases_1.7.0.json"),
                       encoding="utf-8"))["cases"]
DEPLOYABLE = os.path.join(ROOT, "qemx", "selection", "bundles", "qemx-rec-1.7.0-deployable")


def _task(spec_json, B):
    from qemx.selection import deploy as DP
    from qemx.selection import task as TK
    return DP.from_spec(TK.TaskSpec.from_json(spec_json), int(B))


def test_margin_runner_up_is_a_verified_candidate():
    """The margin compares the choice with the next *verified* candidate in the model's order — the deployment
    counterpart of the label-feasible runner-up the threshold τ was calibrated with — not with an unverified
    learning candidate whose design turns out degenerate."""
    from qemx.selection import deploy as DP
    from qemx.selection import preflight as PF
    b = DP.load_bundle(DEPLOYABLE)
    seen = 0
    for c in CASES[:4]:
        r = DP.recommend(_task(c["task_spec"], c["budget"]), b)
        if r.get("runner_up") is None:
            continue
        seen += 1
        assert r["preflight"][r["runner_up"]]["status"] == PF.VERIFIED, (c["case_id"], r["runner_up"])
        order = sorted(r["predicted_log10_risk"], key=r["predicted_log10_risk"].get)
        between = order[order.index(r["model_choice"]) + 1:order.index(r["runner_up"])]
        assert all(r["preflight"][k]["status"] != PF.VERIFIED for k in between)
    assert seen


def test_study_design_seeds_follow_the_records_seed_scheme():
    """``replay(..., design_seeds='study')`` must reuse the seed each label record's allocation was computed with:
    the master seed under scheme 1, the namespaced allocation stream under scheme 2 (as ``emulate.prepare``)."""
    from qemx.selection import emulate as EM
    from qemx.selection import task as TK
    spec = TK.make_spec("hea", 3, layers=2, angle_seed=7)
    for plan, crn in ((dict(mode="fresh", per_half=1), ""),
                      (dict(mode="fresh", per_half=1, seed_scheme=2, crn="task"), spec.task_id),
                      (dict(mode="fresh", per_half=1, seed_scheme=2, crn="base"), spec.base_id)):
        prep = EM.prepare(TK.build(spec), ["CDR"], seed=1234567, designs=plan)
        rec = dict(master_seed=1234567, seed_scheme=EM.DesignPlan.from_config(plan).seed_scheme, crn_group=crn)
        assert prep.ctx_truth["CDR"].seed == EM.allocation_seed(rec, "CDR")


def test_model_component_rows_describe_the_component_and_missing_risks_are_undefined():
    from qemx.selection import replay as RP
    rows = [dict(task_id="t1", base_id="b1", family="ghz", n=3, B=1000, status="out-of-distribution",
                 accepted=False, choice="RO", skipped="CDR:infeasible", scored=True, risk=1e-3, seconds=0.1,
                 cache="miss"),
            dict(task_id="t2", base_id="b2", family="ghz", n=3, B=1000, status="ok", accepted=True, choice="PEC",
                 skipped="", scored=True, risk=2e-3, seconds=0.02, cache="hit")]
    choices = {("t1", 1000): {"M|choice": "ZNE-L"}, ("t2", 1000): {"M|choice": ""}}
    risk = {("t1", "ZNE-L", 1000): 5e-4}
    comp = RP.component_rows(rows, choices, "M|choice", risk)
    assert [c["choice"] for c in comp] == ["ZNE-L", ""] and {c["status"] for c in comp} == {"model-component"}
    assert comp[0]["scored"] and not comp[1]["scored"] and math.isnan(comp[1]["risk"])
    s = RP.summarize(comp, {("t1", 1000): 1e-3, ("t2", 1000): 1e-3})
    assert s["primary"].startswith("undefined") and "diff" not in s
    s2 = RP.summarize(rows, {("t1", 1000): 1e-3, ("t2", 1000): 1e-3})
    assert s2["seconds_fresh_calls"] == 1 and s2["seconds_median_fresh"] == 0.1


def test_missing_risks_are_not_counted_as_not_severe():
    from qemx.selection import evaluate as EV
    t = EV.tail_metrics(np.array([0.5, np.nan, 0.01]), np.array(["a", "a", "b"]))
    assert t["p_severe_abs"] == pytest.approx(0.5)


def test_allocation_level_above_the_simulation_width_uses_the_design_level(monkeypatch):
    """The allocation level simulates the supplied model densely; above SIMULATION_MAX_WIDTH it is never called."""
    from qemx.phase.families import best_chain
    from qemx.selection import deploy as DP

    def tripwire(*a, **k):
        raise AssertionError("allocation-level simulation called above the width limit")
    monkeypatch.setattr(DP, "_supplied_result", tripwire)
    n = DP.SIMULATION_MAX_WIDTH + 2
    ops = (("ry", (0,), (1.0,), "gate", ""),) + tuple(("cx", (i, i + 1), (), "gate", "") for i in range(n - 1))
    t = DP.DeploymentTask(n, ops, tuple("0" * n), "Z" * n, False, ("z_mid", "Z on the middle bit", ()), "fake_fez",
                          100000, initial_layout=tuple(best_chain("fake_fez", n)), obs_terms=(((n // 2,), 1.0),))
    r = DP.recommend(t, DP.load_bundle(DEPLOYABLE), preflight_level="allocation", allow_simulation=False)
    assert r["preflight_level"] == "allocation" and r["choice"]


def test_a_declared_post_hoc_reuse_is_not_an_overlap_of_that_confirmation_only():
    """1.8.0: the root audit reuses two 1.7.0 confirmation circuits (its regression cases) after their once-only
    evaluation; declared as ``post_hoc_of``, that reuse does not count against confirm17, but it does count for any
    other study."""
    from qemx.selection import ancestry as AY
    from qemx.selection import task as TK
    specs = [TK.make_spec("ghz", 4, theta=0.77), TK.make_spec("qft", 3)]
    reg = AY.register({"schema": AY.load()["schema"]}, "conf_x", "confirmation", specs[:1])
    AY.register(reg, "audit_x", "exploration", specs, post_hoc_of=["conf_x"])
    assert AY.overlap(reg, specs[:1], exclude_study="conf_x") == {}
    AY.check_confirmation(reg, specs[:1], "conf_x")
    assert AY.overlap(reg, specs[:1], exclude_study="later") == {"confirmation": [AY.logical_id(specs[0])],
                                                                 "exploration": [AY.logical_id(specs[0])]}
    assert reg["studies"]["audit_x"]["post_hoc_of"] == ["conf_x"]
