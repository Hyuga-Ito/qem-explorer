"""1.5: the 600-second campaign ledger with a mock provider — admission under the conservative model, failures,
cancellations, crashes between submission and bookkeeping, resume without duplicate submissions, overruns."""
import json
import math
import types

import pytest

from qemx import campaign as CG


def _ledger(tmp_path, **kw):
    return CG.Ledger.create(str(tmp_path / "ledger.json"), **kw)


def test_usage_estimate_brackets_and_ibm_quick_formula():
    e = CG.estimate_usage(shots=100_000, circuits=50, jobs=1)
    assert e.low_s == pytest.approx(2 + 0.00035 * 100_000)
    assert e.low_s <= e.central_s <= e.high_s
    assert e.central_s == pytest.approx(2 + 0.00057 * 100_000) and e.high_s == pytest.approx(1.3 * e.central_s)
    assert CG.estimate_usage(1000, 250, 1).central_s == pytest.approx(3 * 2 + 0.57)     # 3 assumed sub-jobs
    with pytest.raises(ValueError):
        CG.estimate_usage(-1, 1)


def test_ledger_never_admits_beyond_the_ceiling_when_the_model_is_conservative(tmp_path):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    prov = CG.MockProvider(true_per_shot_s=0.00055)           # reality slightly cheaper than the central model
    jobs = [("job %d" % i, 1000, 40) for i in range(200)]     # far more than the allowance
    with pytest.raises(CG.CampaignLimit) as ei:
        CG.submit_with_ledger(led, jobs, lambda l, s, c: prov.run(s * c))
    recs = ei.value.records
    assert 0 < len(recs) < 200
    CG.reconcile_jobs(led, lambda jid: prov.jobs[jid])
    assert prov.total_usage() <= 600 - 60
    st = led.status()
    assert st["outstanding_s"] == 0 and st["reported_usage_s"] == pytest.approx(prov.total_usage())
    # reconciliation frees what the conservative estimate held back: a little more fits now
    more = CG.submit_with_ledger(led, [("late", 1000, 10)], lambda l, s, c: prov.run(s * c))
    assert len(more) == 1


def test_failed_and_cancelled_jobs_still_count(tmp_path):
    led = _ledger(tmp_path, ceiling_s=100, reserve_s=10)
    prov = CG.MockProvider(fail_every=2)
    CG.submit_with_ledger(led, [("a", 1000, 10), ("b", 1000, 10)], lambda l, s, c: prov.run(s * c))
    CG.reconcile_jobs(led, lambda jid: prov.jobs[jid])
    states = sorted(e["state"] for e in led.entries)
    assert states == ["completed", "failed"]
    assert led.committed_s() == pytest.approx(prov.total_usage())   # the failed job's usage is charged
    # a job that ended without reported usage keeps its high estimate
    rid = led.admit("c", CG.estimate_usage(1000, 10))
    led.submitted(rid, "x-1")
    led.reconcile("x-1", None, "cancelled")
    assert led.committed_s() == pytest.approx(prov.total_usage() + CG.estimate_usage(1000, 10).high_s)


def test_submission_error_releases_and_crash_keeps_the_reservation(tmp_path):
    led = _ledger(tmp_path, ceiling_s=300, reserve_s=30)
    prov = CG.MockProvider(submit_error_on=2)
    with pytest.raises(RuntimeError):
        CG.submit_with_ledger(led, [("a", 1000, 10), ("b", 1000, 10)], lambda l, s, c: prov.run(s * c))
    assert [e["state"] for e in led.entries] == ["submitted", "released"]
    prov2 = CG.MockProvider(crash_after_create_on=1, prefix="crash")
    with pytest.raises(KeyboardInterrupt):
        CG.submit_with_ledger(led, [("c", 1000, 10)], lambda l, s, c: prov2.run(s * c))
    orphan = led.unsubmitted_reservations()
    assert len(orphan) == 1                                   # unknown: maybe a job, so still committed
    before = led.committed_s()
    led.resolve(orphan[0]["id"], "crash-0001")                # found in the provider's job list
    assert led.committed_s() == pytest.approx(before) and not led.unsubmitted_reservations()
    with pytest.raises(ValueError):
        rid = led.admit("d", CG.estimate_usage(10, 1))
        led.submitted(rid, "mock-0001")                       # the same job id twice is refused
    reloaded = CG.Ledger.load(led.path)                       # everything was persisted
    assert reloaded.status() == led.status()


def test_overrun_stops_admission_and_outstanding_bound(tmp_path):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    prov = CG.MockProvider(true_per_shot_s=0.002)             # 3.5× the model: the model is wrong
    CG.submit_with_ledger(led, [("a", 1000, 10)], lambda l, s, c: prov.run(s * c))
    CG.reconcile_jobs(led, lambda jid: prov.jobs[jid])
    assert led.status()["overruns"]
    with pytest.raises(CG.CampaignLimit, match="recalibrate"):
        led.admit("b", CG.estimate_usage(1000, 10))
    led2 = CG.Ledger.create(str(tmp_path / "l2.json"), 600, 60, max_outstanding_s=20)
    led2.admit("x", CG.estimate_usage(10_000, 10))            # ≈ 10 s high
    with pytest.raises(CG.CampaignLimit, match="unreconciled"):
        led2.admit("y", CG.estimate_usage(10_000, 10))


def test_account_allowance_and_ledger_files(tmp_path):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60, account_remaining_s=200)
    assert led.remaining_s() == pytest.approx(140)
    with pytest.raises(FileExistsError):
        CG.Ledger.create(led.path)
    with pytest.raises(ValueError):
        CG.Ledger.create(str(tmp_path / "bad.json"), ceiling_s=50, reserve_s=60)


def test_preflight_of_a_plan():
    plan = dict(runs=[dict(shots=1500)] * 120 + [dict(shots=200)] * 600)
    pf = CG.preflight(plan, None, ceiling_s=600, reserve_s=60)
    assert pf["total_jobs"] == 3 and sum(r["executions"] for r in pf["jobs"]) == 120 * 1500 + 600 * 200
    md = CG.preflight_markdown(pf)
    assert "high, used for admission" in md and "max_execution_time" in md


def test_divergence_submit_persists_every_job_and_resume_never_duplicates(tmp_path, monkeypatch):
    """qemx.divergence.submit with a fake sampler: a crash after the second job leaves both job ids on disk and in
    the ledger; resuming submits only the runs not yet covered."""
    import qiskit_ibm_runtime
    from qemx import divergence as DV
    created = []

    class FakeJob:
        def __init__(self, i):
            self.i = i

        def job_id(self):
            return "fake-%d" % self.i

    class FakeSampler:
        crash_on = None

        def __init__(self, mode=None):
            self.options = types.SimpleNamespace(max_execution_time=None)

        def run(self, circs, shots):
            created.append((len(circs), shots, self.options.max_execution_time))
            if len(created) == FakeSampler.crash_on:
                raise KeyboardInterrupt("crash")
            return FakeJob(len(created))
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", FakeSampler)
    monkeypatch.setattr(DV, "_plain_sampler", lambda s: None)
    runs = [types.SimpleNamespace(key="r%d" % i) for i in range(700)]
    plan = types.SimpleNamespace(runs=runs, suite=types.SimpleNamespace(study="test"))
    monkeypatch.setattr(DV, "physical_circuits", lambda p, b, rs: [(r.key, "circ", 100) for r in rs])
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    disk = {}
    FakeSampler.crash_on = 3
    with pytest.raises(KeyboardInterrupt):
        DV.submit(plan, types.SimpleNamespace(name="fake"), ledger=led, on_record=lambda r: disk.update(r=r))
    assert [r["job_id"] for r in disk["r"]] == ["fake-1", "fake-2"]
    assert all(c[2] and c[2] > 0 for c in created)            # every job got a max_execution_time
    assert len(led.unsubmitted_reservations()) == 1           # the crashed one: unknown until resolved
    led.resolve(led.unsubmitted_reservations()[0]["id"], None)   # the provider shows no third job
    done = {k for r in disk["r"] for k in r["runs"]}
    FakeSampler.crash_on = None
    recs = DV.submit(plan, types.SimpleNamespace(name="fake"), skip=done, ledger=led)
    sent = [k for r in disk["r"] for k in r["runs"]] + [k for r in recs for k in r["runs"]]
    assert sorted(sent) == sorted(r.key for r in runs) and len(sent) == len(set(sent))
    assert len({e["job_id"] for e in led.entries if e["job_id"]}) == len(disk["r"]) + len(recs)
