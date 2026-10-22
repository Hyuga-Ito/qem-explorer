"""1.5.1: defects found by the independent re-review of the 1.5.1 tree (N50–N56). Everything provider-side is mocked:
no account, no network, no QPU."""
import math
import os
import sys
import types
import warnings
from types import MappingProxyType

import numpy as np
import pytest

from qemx import campaign as CG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_spending import _RealLike, _est, _hw_plan, _ledger, _mock_runtime  # noqa: E402


# ------------------------------------------------------------------ N50: Head-to-Head sweeps keep F01's moments
def test_noise_sweep_rows_carry_the_budget_dependent_moments_of_nonlinear_estimators():
    from qemx import analysis as A
    from qemx.circuit import parse
    from qemx.compile import build
    from qemx.methods import Recipe
    from qemx.noise import NoiseSpec
    from qemx.observables import z
    from qemx.recipes import compare
    comp = build(parse("ry(1.2) 0", 1))
    rec = [Recipe.make("zne", rule="exponential", on_fail="linear")]
    row = A.sweep(comp, NoiseSpec(p1=0.15), z(0, 1), rec, "p1", [0.15], "numpy", 300)[0]
    assert "target moments" in row and A.target_grid(row) is not None
    ideal, outs, _ = compare(comp, NoiseSpec(p1=0.15), z(0, 1), rec, shots=300, seed=0)
    ref = dict(row, **A.nonlinear_fields(outs[0].result, ideal))
    for B in (60, 300, 1000):
        assert A.rmse_row(row, [B])[0] == pytest.approx(A.rmse_row(ref, [B])[0], rel=1e-12)


# ------------------------------------------------------------------ N51: live costs use the ledger's constraints
def _study(key, rc):
    from qemx import rec_context as RX
    return RX.StudyContext("x", RX.SCHEMA, MappingProxyType(dict(
        candidates=[key], protocols={key: dict(recipe=rc.to_json())},
        cost_policy=dict(name="fixed", integer=True, s_min=32, s_tr_min=32))))


def test_live_protocol_costs_agree_with_the_study_ledger_for_pec_and_post_selection():
    from qemx import rec_context as RX
    from qemx.circuit import parse
    from qemx.compile import build
    from qemx.methods import Recipe
    from qemx.noise import NoiseSpec
    from qemx.observables import parity, z
    from qemx.phase.ledger import Cost, Policy, min_budget
    rc = Recipe.make("pec", readout="none", shots_per_circuit=200)
    live = RX.live_context(build(parse("ry(1.2) 0", 1)), "numpy", NoiseSpec(p1=0.01), z(0, 1))
    c = RX.protocol_costs(_study("PEC", rc), live, 150)["PEC"]
    want = min_budget(Cost("PEC", b=0.0, v=1.0, m=1, per_circuit=200), Policy("fixed"))
    assert c["shots_per_circuit"] == 200 and c["min_budget"] == want == 200 and c["feasible"] is False
    sv = Recipe.make("symmetry", readout="none")
    live2 = RX.live_context(build(parse("h 0; cx 0 1", 2)), "numpy", NoiseSpec(p1=0.05, p2=0.2), parity([0, 1], 2))
    c2 = RX.protocol_costs(_study("SV", sv), live2, 50, symmetry=((0, 1), +1))["SV"]
    assert 0 < c2["accept"] < 1 and c2["min_budget"] == math.ceil(5 / c2["accept"])


# ------------------------------------------------------------------ N52: the Holm family is the declared one
def test_holm_adjusts_over_every_declared_primary_contrast():
    from qemx import transfer as TR
    pred = {c: dict(ideal=1.0, role="primary", methods={
        "RAW": dict(bias=-0.2, sd=0.01, rmse=0.2), "RO": dict(bias=-0.1, sd=0.01, rmse=0.1005),
        "ZNE-L": dict(bias=-0.01, sd=0.02, rmse=0.0224), "PEC": dict(bias=0.0, sd=0.05, rmse=0.05)})
        for c in ("qft4", "bv4")}
    rng = np.random.default_rng(1)
    rows = [dict(block=b, circuit="qft4", method=m, error=mu + 0.03 * rng.normal(), status="ok")
            for b in range(6) for m, mu in (("RO", -0.1), ("ZNE-L", -0.01), ("RAW", -0.2), ("PEC", 0.0))]
    rows += [dict(block=0, circuit="bv4", method=m, error=-0.05, status="ok") for m in ("RO", "ZNE-L", "RAW", "PEC")]
    an = TR.analyze(rows, pred)
    prim = {x["circuit"]: x for x in an["contrasts"] if x["kind"] == "primary"}
    assert math.isnan(prim["bv4"]["p_one_sided"]) and prim["bv4"]["confirmed"] is False
    assert prim["qft4"]["p_holm"] == pytest.approx(min(1.0, 2 * prim["qft4"]["p_one_sided"]))


# ------------------------------------------------------------------ N53: the once-only holdout is protected
def test_holdout_evaluation_refuses_another_context_before_evaluating(monkeypatch):
    from qemx import rec_context as RX
    from qemx import study as SD

    def ds(groups, seed):
        rng = np.random.default_rng(seed)
        n = len(groups)
        return dict(X=rng.normal(size=(n, 3)), Y=rng.normal(size=(n, 2)), F=np.ones((n, 2), bool),
                    methods=["RAW", "ZNE-L"], groups=np.array(groups), lam=np.full(n, 0.1), budget=np.full(n, 1e4),
                    names=["a", "b", "c"])
    monkeypatch.setattr(SD, "rec_dataset", lambda S, fs, budgets=None: ds(
        ["d%d" % (i // 4) for i in range(40)], 0) if S == "dev" else ds(["h%d" % (i // 4) for i in range(12)], 1))
    monkeypatch.setattr(RX, "study_context", lambda S, schema: dict(
        protocols={"ZNE-L": dict(recipe=dict(estimator="zne", readout="tensored" if S == "dev" else "none"))},
        noise=dict(devices=["fake_fez"] if S == "dev" else ["fake_torino"], readout=["on"], homogenized=[False],
                   backends=[], scales=[1.0], coherent_fractions=[0.0])))
    evaluated = []
    real = SD.RC.evaluate
    monkeypatch.setattr(SD.RC, "evaluate", lambda *a, **k: evaluated.append(1) or real(*a, **k))
    with pytest.raises(ValueError, match="does not match the development run"):
        SD.evaluate_holdout("dev", "hold")
    assert not evaluated                                                    # nothing was evaluated
    ev = SD.evaluate_holdout("dev", "hold", allow_context_mismatch=True)
    assert ev["meta"]["context_mismatch_allowed"] and len(ev["meta"]["problems"]) == 2


# ------------------------------------------------------------------ N54: no partial measurement-twirl average
def test_collect_drops_a_request_whose_masks_lost_a_job():
    from qemx import hardware as H

    class Job:
        def __init__(self, counts=None):
            self.counts = counts

        def status(self):
            return "DONE" if self.counts is not None else "ERROR"

        def usage(self):
            return 1.0

        def result(self):
            return [types.SimpleNamespace(data=types.SimpleNamespace(c=types.SimpleNamespace(
                get_counts=lambda c=c: c))) for c in self.counts]
    jobs = {"a": Job([{"00": 3}, {"01": 3}]), "b": Job(None)}
    recs = [dict(job_id="a", pubs=[("trex", 0), ("other", 0)]), dict(job_id="b", pubs=[("trex", 3)])]
    freqs, status = H.collect(types.SimpleNamespace(job=lambda jid: jobs[jid]), recs, 2, wait=True)
    assert set(freqs) == {"other"}                                   # "trex" lost one of its masks: dropped whole
    assert status[1]["incomplete_requests"] == ["trex"]


# ------------------------------------------------------------------ N55: a resumed transfer run never re-sends a job
def test_submit_reports_each_call_before_it_is_made_and_skips_jobs_already_sent(tmp_path, monkeypatch):
    from qemx import hardware as H
    warnings.simplefilter("ignore")
    plan, fake = _hw_plan()
    remote, _, service = _mock_runtime(monkeypatch, lose_on=2)
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    attempts = []
    with pytest.raises(H.SubmitError) as ei:
        H.submit(plan, _RealLike(fake), ledger=led, on_attempt=attempts.append)
    assert len(attempts) == 2 and all(a["job_id"] is None and a["outcome"] == "pending" for a in attempts)
    assert attempts[1]["reservation"] == ei.value.unknown[0]["reservation"]
    CG.resolve_unknown(led, CG.ibm_tag_lookup(service))                     # the lost job is found by its tag
    resolved = dict(attempts[1], job_id=CG.Ledger.load(led.path).entry(attempts[1]["reservation"])["job_id"])
    recs = H.submit(plan, _RealLike(fake), ledger=led, skip=ei.value.records + [resolved])
    assert len(remote) == 4 and len(recs) == 2                              # only the two jobs never sent
    sent = {H.job_identity(r) for r in ei.value.records + [resolved] + recs}
    assert len(sent) == 4


def test_run_ibm_resumes_an_interrupted_block_without_sending_a_job_twice(tmp_path, monkeypatch):
    import hardware_transfer as HT  # noqa: F401 - scripts/ is on the path below
    from qemx import hardware as H
    from qemx import transfer as TR
    warnings.simplefilter("ignore")
    plan, fake = _hw_plan()
    remote, _, service = _mock_runtime(monkeypatch, lose_on=2)
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    monkeypatch.setattr(TR, "resolve_on", lambda design, backend: [(dict(name="bell", role="primary"),
                                                                    types.SimpleNamespace())])
    monkeypatch.setattr(TR, "block_plan", lambda design, r, backend, b, ci: plan)
    monkeypatch.setattr(TR, "predictions_on", lambda design, backend, rs: {"bell": dict(frozen=True)})
    service.backend = lambda name: _RealLike(fake)
    monkeypatch.setattr(HT, "_service", lambda a: service)
    a = types.SimpleNamespace(out=str(tmp_path / "run"), ledger=led.path, backend="ibm_mockperth", blocks=1,
                              mode="job", ask_token=False)
    with pytest.raises(H.SubmitError):
        HT.cmd_run_ibm(a)                                                   # job 2: response lost
    import json
    j = json.load(open(os.path.join(a.out, "jobs.json"), encoding="utf-8"))["blocks"]["0/bell"]
    assert j["partial"] and len(j["records"]) == 1 and j["pending"][0]["client_tag"]
    with pytest.raises(CG.UnresolvedSubmission):
        HT.cmd_run_ibm(a)                                                   # refused until resolved
    CG.resolve_unknown(led, CG.ibm_tag_lookup(service))
    HT.cmd_run_ibm(a)
    j = json.load(open(os.path.join(a.out, "jobs.json"), encoding="utf-8"))["blocks"]["0/bell"]
    assert len(remote) == 4 and not j["partial"] and sorted(r["job_id"] for r in j["records"]) == \
        ["job-1", "job-2", "job-3", "job-4"]
    assert json.load(open(os.path.join(a.out, HT.FROZEN), encoding="utf-8"))["circuits"] == {"bell": dict(frozen=True)}
    HT.cmd_run_ibm(a)                                                       # complete: nothing more is sent
    assert len(remote) == 4


def test_frozen_predictions_use_the_designs_model_and_path():
    """run-ibm freezes predictions with the same model and simulation path as the design and power analysis
    (on the fake device of the design they are the same numbers), not the planner's recording model."""
    from qemx import device as D
    from qemx import transfer as TR
    design = dict(TR.DESIGN, circuits=[c for c in TR.DESIGN["circuits"] if c["name"] == "bv4"])
    b = D.fake_backend(design["device"])
    rs = TR.resolve_on(design, b)
    day = TR.predictions_on(design, b, rs)["bv4"]
    ref = TR._predict(design, [(c, r, TR.block_plan(design, r, b, 0, ci), design["device"])
                               for ci, (c, r) in enumerate(rs)])["bv4"]
    assert day["simulator"].startswith("cal:") and day["calibration_hash"] == ref["calibration_hash"]
    for m in ref["methods"]:
        for f in ("bias", "sd"):
            x, y = day["methods"][m][f], ref["methods"][m][f]
            assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, rel=1e-9, abs=1e-12), (m, f)


sys.path.insert(0, os.path.join(ROOT, "scripts"))
