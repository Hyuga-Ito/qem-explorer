"""1.5.1 spending integrity (independent review F03–F05): the campaign ledger is transactional across stale objects,
threads and processes; a submission whose outcome is unknown stays committed and is never sent again before it is
resolved; real-hardware submission requires a valid ledger; collection reconciles reported usage. Everything is
mocked: no account, no network, no QPU."""
import json
import multiprocessing as mp
import os
import sys
import threading
import types
import warnings

import pytest

from qemx import campaign as CG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIGH = 15.0                                   # s per admitted job: exactly 36 fit into 600 − 60 s


def _est(high):
    return CG.UsageEstimate(1, 1, 1, high / 2, high / 1.3, float(high), ())


def _ledger(tmp_path, **kw):
    return CG.Ledger.create(str(tmp_path / "ledger.json"), **kw)


def _disk(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ====================================================================== F03: transactions
def test_stale_instances_never_lose_a_reservation_or_over_admit(tmp_path):
    """Review §9.1: A and B load the same ledger; A admits 400 s; B, still holding the old snapshot, must decide on
    the current file (refused), and every admitted reservation must be in the file."""
    path = str(tmp_path / "ledger.json")
    CG.Ledger.create(path, ceiling_s=600, reserve_s=60)
    a, b = CG.Ledger.load(path), CG.Ledger.load(path)
    ra = a.admit("A", _est(400))
    with pytest.raises(CG.CampaignLimit, match="only 140.0 s remain"):
        b.admit("B", _est(400))
    assert [e["id"] for e in b.entries] == [ra]          # B's view was refreshed by its (refused) transaction
    rb = b.admit("B", _est(100))                         # what fits is admitted on top of A's entry …
    rc = a.admit("C", _est(40))                          # … and A, now stale, adds on top of B's
    final = CG.Ledger.load(path)
    assert [e["id"] for e in final.entries] == [ra, rb, rc]
    assert final.committed_s() == pytest.approx(540) and final.remaining_s() == pytest.approx(0)
    with pytest.raises(CG.CampaignLimit):
        a.admit("D", _est(0.5))
    # every transition goes through a transaction on the current file, whichever object performs it
    b.begin_submit(ra)
    a.submitted(ra, "job-a")                             # a's view did not know about begin_submit
    b.reconcile("job-a", 120.0, "completed")
    a.release(rc, "not needed")
    final = CG.Ledger.load(path)
    assert [(e["id"], e["state"]) for e in final.entries] == [(ra, "completed"), (rb, "reserved"),
                                                              (rc, "released")]
    assert final.committed_s() == pytest.approx(220) and a.committed_s() == pytest.approx(220)


def _admit_worker(path, wid, tries, barrier, q):
    """One process: admit, then walk each admitted job through submitting → submitted → completed."""
    try:
        sys.path.insert(0, ROOT)
        from qemx import campaign as CG_
        est = CG_.UsageEstimate(1, 1, 1, HIGH / 2, HIGH / 1.3, HIGH, ())
        barrier.wait(60)
        got, refused = [], 0
        for i in range(tries):
            led = CG_.Ledger.load(path)                   # a fresh object each time, as a CLI call would
            try:
                rid = led.admit("w%d-%d" % (wid, i), est)
            except CG_.CampaignLimit:
                refused += 1
                continue
            got.append(rid)
            if hasattr(led, "begin_submit"):              # 1.5.1 state; the invariants below do not depend on it
                led.begin_submit(rid, True, int(HIGH))       # the per-job limit fits the reservation (N12)
            led.submitted(rid, "job-%d-%d" % (wid, i))
            if i % 3 == 0:                                # some jobs report usage, some do not yet
                led.reconcile("job-%d-%d" % (wid, i), HIGH, "completed")
        q.put(dict(wid=wid, admitted=got, refused=refused, error=None))
    except BaseException as e:                            # noqa: BLE001 - reported to the parent
        q.put(dict(wid=wid, admitted=[], refused=0, error="%s: %s" % (type(e).__name__, e)))


def test_concurrent_processes_admit_transactionally(tmp_path, monkeypatch):
    """Several real processes admit and update one ledger at the same time: no entry is lost, the admitted high
    estimates never exceed ceiling − reserve, and the committed total is the sum over the surviving entries."""
    path = str(tmp_path / "ledger.json")
    CG.Ledger.create(path, ceiling_s=600, reserve_s=60)
    # spawn re-imports the parent's __main__ in every child; inside the full suite an earlier Streamlit AppTest
    # leaves its generated page script there, which cannot run on its own. A blank __main__ isolates the children.
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    ctx = mp.get_context("spawn")                         # as on Windows
    procs, tries = 4, 14
    barrier, q = ctx.Barrier(procs), ctx.Queue()
    ps = [ctx.Process(target=_admit_worker, args=(path, w, tries, barrier, q)) for w in range(procs)]
    for p in ps:
        p.start()
    res = [q.get(timeout=240) for _ in ps]
    for p in ps:
        p.join(60)
    assert [r["error"] for r in res] == [None] * procs
    admitted = [rid for r in res for rid in r["admitted"]]
    assert sum(r["refused"] for r in res) + len(admitted) == procs * tries
    final = CG.Ledger.load(path)
    by_id = {e["id"]: e for e in final.entries}
    assert set(admitted) == set(by_id) and len(admitted) == len(final.entries)        # nothing lost, no extras
    assert HIGH * len(admitted) <= 600 - 60 + 1e-9                                      # never over-admitted
    assert len(admitted) == 36                            # and nothing that fit was refused for a stale reason
    assert all(e["state"] in ("submitted", "completed") and e["job_id"] for e in final.entries)
    expect = sum(float(e["usage_s"]) if e["state"] == "completed" else float(e.get("reserved_s", e["est_high_s"]))
                 for e in final.entries)
    assert final.committed_s() == pytest.approx(expect) == pytest.approx(HIGH * len(admitted))
    assert len({e["job_id"] for e in final.entries}) == len(final.entries)
    assert sorted(os.listdir(tmp_path)) == ["ledger.json", "ledger.json.lock"]         # no temp file left behind


def test_threads_of_one_process_admit_transactionally(tmp_path):
    """Streamlit sessions are threads of one server process: separate Ledger objects in separate threads."""
    path = str(tmp_path / "ledger.json")
    CG.Ledger.create(path, ceiling_s=600, reserve_s=60)
    out, errors = [], []

    def run(w):
        try:
            for i in range(10):
                try:
                    out.append(CG.Ledger.load(path).admit("t%d-%d" % (w, i), _est(HIGH)))
                except CG.CampaignLimit:
                    pass
        except BaseException as e:                        # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=run, args=(w,)) for w in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    final = CG.Ledger.load(path)
    assert not errors and len(out) == 36 and {e["id"] for e in final.entries} == set(out)
    assert final.committed_s() == pytest.approx(540)


def test_failed_write_changes_nothing_and_leaves_no_temp_file(tmp_path, monkeypatch):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    rid = led.admit("a", _est(10))
    before = _disk(led.path)

    def broken(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(CG.os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        led.admit("b", _est(10))
    monkeypatch.undo()
    assert _disk(led.path) == before                                  # the file is unchanged …
    assert [e["id"] for e in led.entries] == [rid]                    # … and so is the object's view
    assert sorted(os.listdir(tmp_path)) == ["ledger.json", "ledger.json.lock"]
    with led.transaction() as d:                                      # nested transactions share the outer one
        d["note"] = "x"
        led.release(rid, "inside")
        assert led.entry(rid)["state"] == "released"
    assert _disk(led.path)["note"] == "x" and _disk(led.path)["entries"][0]["state"] == "released"


def test_v1_ledger_files_load_and_migrate_without_losing_records(tmp_path):
    """A 1.5.0 ledger (schema /1) is read, its committed total is unchanged, a reservation without a job id becomes
    submission_unknown (1.5.0 could not tell whether it was a job), and the next change writes schema /2 with every
    record kept."""
    path = str(tmp_path / "old.json")
    e = lambda i, st, **kw: dict(dict(id="r%d" % i, label="L%d" % i, campaign="", state=st, est_high_s=10.0 * i,
                                      est_central_s=7.0 * i, executions=100, circuits=1, job_id=None, usage_s=None,
                                      created="2026-10-17T00:00:00Z", updated="2026-10-17T00:00:00Z", note=""), **kw)
    v1 = dict(schema="qemx.campaign.ledger/1", created="2026-10-17T00:00:00Z", ceiling_s=600.0, reserve_s=60.0,
              account_remaining_s=None, max_outstanding_s=None, note="",
              entries=[e(1, "reserved"), e(2, "submitted", job_id="j2"), e(3, "completed", job_id="j3", usage_s=4.5),
                       e(4, "failed", job_id="j4"), e(5, "released")])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(v1, f)
    committed_v1 = 10 + 20 + 4.5 + 40                     # 1.5.0's rule: released excluded, usage when reported
    led = CG.Ledger.load(path)
    assert led.committed_s() == pytest.approx(committed_v1)
    assert [x["state"] for x in led.entries] == ["submission_unknown", "submitted", "completed", "failed", "released"]
    assert _disk(path)["schema"] == "qemx.campaign.ledger/1"          # reading does not rewrite the file
    with pytest.raises(CG.UnresolvedSubmission):
        CG.check_no_unresolved(led)
    led.resolve("r1", "j1")                                            # found on the provider
    d = _disk(path)
    assert d["schema"] == "qemx.campaign.ledger/2" and d["migrated_from"] == "qemx.campaign.ledger/1"
    assert [x["id"] for x in d["entries"]] == ["r1", "r2", "r3", "r4", "r5"]
    assert [x["est_high_s"] for x in d["entries"]] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert CG.Ledger.load(path).committed_s() == pytest.approx(committed_v1)
    # anything that is not a ledger is an explicit error
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CG.LedgerError):
        CG.Ledger.load(str(bad))
    bad.write_text(json.dumps(dict(schema="something/else", entries=[])), encoding="utf-8")
    with pytest.raises(CG.LedgerError, match="not a campaign ledger"):
        CG.Ledger.load(str(bad))
    with pytest.raises(FileNotFoundError):                             # LedgerNotFound is also a FileNotFoundError
        CG.Ledger.load(str(tmp_path / "missing.json"))
    assert not os.path.exists(str(tmp_path / "missing.json.lock"))


# ====================================================================== F04: ambiguous submission outcomes
def test_timeout_after_acceptance_keeps_the_reservation_and_blocks_resubmission(tmp_path):
    """Review §9.2: the provider creates the job, then the response is lost (TimeoutError)."""
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    remote = []

    def submit_then_lose_response(label, shots, circuits, client_tag=None):
        remote.append(client_tag)
        raise TimeoutError("response lost after acceptance")
    with pytest.raises(Exception) as ei:
        CG.submit_with_ledger(led, [("job", 1000, 10)], submit_then_lose_response)
    e = CG.Ledger.load(led.path).entries[0]
    est = CG.estimate_usage(10_000, 10)
    assert e["state"] == "submission_unknown"                          # 1.5.0: 'released'
    assert CG.Ledger.load(led.path).committed_s() == pytest.approx(CG.reservation_s(est))   # committed (1.5.0: 0)
    assert type(ei.value).__name__ == "SubmissionUnknown" and isinstance(ei.value.__cause__, TimeoutError)
    assert e["client_tag"] == remote[0] == ei.value.client_tag and e["tag_applied"] is True
    with pytest.raises(Exception) as ei:                               # retry/resume never re-sends it
        CG.submit_with_ledger(led, [("job", 1000, 10)], submit_then_lose_response)
    assert len(remote) == 1 and type(ei.value).__name__ == "UnresolvedSubmission"


def test_only_a_definite_rejection_releases(tmp_path):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    calls = []

    def run(tag):
        calls.append(tag)
        raise CG.SubmissionRejected("validation failed before any request")
    with pytest.raises(CG.SubmissionRejected):
        CG.submit_one(led, "a", _est(10), run)
    assert led.entries[-1]["state"] == "released"

    def bad_prepare(tag):
        raise ValueError("an option the primitive does not accept")
    with pytest.raises(ValueError):
        CG.submit_one(led, "b", _est(10), run, prepare=bad_prepare)
    assert led.entries[-1]["state"] == "released" and len(calls) == 1  # never called the provider
    for exc in (ConnectionResetError("reset"), RuntimeError("Failed to run program: 503"), OSError("broken pipe")):
        def ambiguous(tag, exc=exc):
            raise exc
        with pytest.raises(CG.SubmissionUnknown):
            CG.submit_one(led, "c", _est(10), ambiguous)
        assert led.entries[-1]["state"] == "submission_unknown"
    assert led.committed_s() == pytest.approx(30)

    def interrupted(tag):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):                             # re-raised after the bookkeeping
        CG.submit_one(led, "d", _est(10), interrupted)
    assert led.entries[-1]["state"] == "submission_unknown" and led.committed_s() == pytest.approx(40)


def test_job_id_that_cannot_be_recorded_is_reported(tmp_path, monkeypatch):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    monkeypatch.setattr(CG.Ledger, "submitted", lambda self, rid, jid: (_ for _ in ()).throw(CG.LedgerBusy("busy")))
    with pytest.raises(CG.SubmissionUnknown) as ei:
        CG.submit_one(led, "a", _est(10), lambda tag: CG.MockJob("job-77", 1.0))
    assert ei.value.job_id == "job-77" and "--job-id job-77" in str(ei.value)
    monkeypatch.undo()
    assert led.refresh().entries[0]["state"] == "submitting"           # committed and unresolved
    led.resolve(ei.value.rid, ei.value.job_id)
    assert led.entries[0]["state"] == "submitted"


def test_resolve_unknown_through_a_provider_lookup_by_tag(tmp_path):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    prov = CG.MockProvider(timeout_after_create_on=2)
    run = lambda label, shots, circ, client_tag=None: prov.run(shots * circ, client_tag)
    with pytest.raises(CG.SubmissionUnknown) as ei:
        CG.submit_with_ledger(led, [("a", 1000, 10), ("b", 2000, 10), ("c", 1000, 10)], run)
    assert len(ei.value.records) == 1 and len(prov.jobs) == 2          # the second job exists on the provider
    out = CG.resolve_unknown(led, prov.jobs_with_tag)
    assert [o["outcome"] for o in out] == ["attached"] and out[0]["job_ids"] == ["mock-0002"]
    assert [e["state"] for e in led.entries] == ["submitted", "submitted"]
    CG.reconcile_jobs(led, lambda jid: prov.jobs[jid])
    assert led.committed_s() == pytest.approx(prov.total_usage())      # exactly what the provider used
    # a call that failed before the provider created anything: not found → stays committed until it is old enough
    # and explicitly allowed to be released; an untagged call is never released automatically
    def lost(label, shots, circ, client_tag=None):
        raise ConnectionResetError("connection reset before the request was sent")
    with pytest.raises(CG.SubmissionUnknown):
        CG.submit_with_ledger(led, [("d", 1000, 10)], lost)
    now = CG._parse_time(led.unresolved()[0]["submit_started"])
    assert [o["outcome"] for o in CG.resolve_unknown(led, prov.jobs_with_tag, release_missing=True, now=now)] \
        == ["not found yet"]
    assert [o["outcome"] for o in CG.resolve_unknown(led, prov.jobs_with_tag, release_missing=True,
                                                      now=now + 1000)] == ["released"]
    with pytest.raises(CG.SubmissionUnknown):
        CG.submit_with_ledger(led, [("e", 1000, 10)], lambda label, shots, circ: (_ for _ in ()).throw(OSError()))
    out = CG.resolve_unknown(led, prov.jobs_with_tag, release_missing=True, now=now + 10_000)
    assert "not applied" in out[0]["outcome"] and len(led.unresolved()) == 1
    rid = led.unresolved()[0]["id"]
    out = CG.resolve_unknown(led, lambda tag: (_ for _ in ()).throw(RuntimeError("service down")))
    assert out[0]["outcome"].startswith("lookup failed") and led.unresolved()[0]["id"] == rid
    out = CG.resolve_unknown(led, lambda tag: ["x-1", "x-2"])
    assert "several" in out[0]["outcome"] and len(led.unresolved()) == 1


def test_cli_status_and_resolve(tmp_path, capsys):
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    with pytest.raises(CG.SubmissionUnknown):
        CG.submit_one(led, "a", _est(10), lambda tag: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(CG.SubmissionUnknown):
        CG.submit_one(led, "b", _est(10), lambda tag: (_ for _ in ()).throw(TimeoutError()))
    e1, e2 = led.unresolved()
    assert CG.main(["status", "--ledger", led.path]) == 0
    assert e1["client_tag"] in capsys.readouterr().err
    with pytest.raises(SystemExit):                                    # the outcome must be stated explicitly
        CG.main(["resolve", "--ledger", led.path, "--id", e1["id"]])
    assert CG.main(["resolve", "--ledger", led.path, "--id", e1["id"], "--job-id", "found-1"]) == 0
    assert CG.main(["resolve", "--ledger", led.path, "--tag", e2["client_tag"], "--not-submitted"]) == 0
    assert CG.main(["resolve", "--ledger", led.path, "--tag", "qemx-ledger-nope", "--not-submitted"]) == 2
    assert CG.main(["status", "--ledger", str(tmp_path / "missing.json")]) == 2
    final = CG.Ledger.load(led.path)
    assert [(e["state"], e["job_id"]) for e in final.entries] == [("submitted", "found-1"), ("released", None)]
    assert final.committed_s() == pytest.approx(10)


# ------------------------------------------------------------------ the app's submitter (qemx.hardware)
def _hw_plan():
    from qemx import device as D, hardware as H, templates as T
    from qemx.compile import Compiled, with_io
    from qemx.recipes import preset
    tpl = T.bell()
    b = D.fake_backend("fake_perth")
    spec, mm, phys, info = D.compile_for_target(with_io(tpl.spec), b, D.good_target_for(b), "FakePerth")
    plan = H.make_plan(Compiled(with_io(tpl.spec), spec, mm, info), tpl.observable,
                       [preset("Unmitigated"), preset("ZNE · Richardson")], 4000, tpl.symmetry, 0, b)
    return plan, b


class _RealLike:
    """A real provider's backend for the code under test: not a fake or simulator class."""
    name = "ibm_mockperth"

    def __init__(self, fake):
        self._fake = fake

    def __getattr__(self, k):
        return getattr(self._fake, k)


def _mock_runtime(monkeypatch, lose_on=None, batch_ok=False):
    import qiskit_ibm_runtime as R
    remote, made = {}, []

    class Job:
        def __init__(self, jid, circs, shots, tags):
            self.jid, self.circs, self.shots, self.tags = jid, circs, shots, tags

        def job_id(self):
            return self.jid

        def status(self):
            return "DONE"

        def usage(self):
            return 0.5

        def result(self):
            pub = lambda c: types.SimpleNamespace(data=types.SimpleNamespace(c=types.SimpleNamespace(
                get_counts=lambda: {"0" * c.num_clbits: self.shots})))
            return [pub(c) for c in self.circs]

    class Sampler:
        def __init__(self, mode=None):
            made.append(mode)
            self.options = types.SimpleNamespace(twirling=types.SimpleNamespace(),
                                                 dynamical_decoupling=types.SimpleNamespace(),
                                                 environment=types.SimpleNamespace(job_tags=None),
                                                 max_execution_time=None)

        def run(self, pubs, shots=None):
            jid = "job-%d" % (len(remote) + 1)
            remote[jid] = Job(jid, list(pubs), shots, list(self.options.environment.job_tags or []))
            if lose_on is not None and len(remote) == lose_on:
                raise TimeoutError("read timed out")
            return remote[jid]

    class Batch:
        def __init__(self, backend=None):
            if not batch_ok:
                raise RuntimeError("no batch mode on this plan")

        def close(self):
            pass

    class Service:
        def __init__(self, *a, **kw):
            pass

        def job(self, jid):
            return remote[jid]

        def jobs(self, job_tags=None, limit=None, **kw):
            return [j for j in remote.values() if set(job_tags or []) <= set(j.tags)]
    monkeypatch.setattr(R, "SamplerV2", Sampler)
    monkeypatch.setattr(R, "Batch", Batch)
    return remote, made, Service()


def test_hardware_submit_requires_a_ledger_for_a_real_provider(tmp_path, monkeypatch):
    from qemx import hardware as H
    warnings.simplefilter("ignore")
    plan, fake = _hw_plan()
    remote, made, _ = _mock_runtime(monkeypatch)
    with pytest.raises(Exception) as ei:                               # 1.5.0 submitted without a ledger
        H.submit(plan, _RealLike(fake))
    assert type(ei.value).__name__ == "LedgerRequired" and not remote
    with pytest.raises(CG.LedgerNotFound):                             # a configured path must exist
        H.submit(plan, _RealLike(fake), ledger=str(tmp_path / "typo.json"))
    with pytest.raises(CG.LedgerError):
        H.submit(plan, _RealLike(fake), ledger=object())
    assert not remote and not made                                     # nothing reached the provider
    recs = H.submit(plan, fake)                                        # a fake backend runs without a ledger
    assert len(recs) == 4 and len(remote) == 4
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    recs = H.submit(plan, _RealLike(fake), ledger=led.path)            # a path is opened as a ledger
    assert [r["job_id"] for r in recs] == [e["job_id"] for e in CG.Ledger.load(led.path).entries]
    assert all(r["mode"] == "job" and r["client_tag"] in remote[r["job_id"]].tags for r in recs)
    assert all(r["max_execution_time"] > 0 for r in recs)


def test_hardware_submit_unknown_outcome_is_kept_and_found_again(tmp_path, monkeypatch):
    from qemx import hardware as H
    warnings.simplefilter("ignore")
    plan, fake = _hw_plan()
    remote, made, service = _mock_runtime(monkeypatch, lose_on=2)
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    seen = []
    with pytest.raises(H.SubmitError) as ei:
        H.submit(plan, _RealLike(fake), ledger=led, on_record=lambda r: seen.append(len(r)))
    e = ei.value
    assert len(e.records) == 1 and seen == [1] and len(e.unknown) == 1 and len(remote) == 2
    assert e.unknown[0]["pubs"] and e.unknown[0]["client_tag"] in remote["job-2"].tags
    st = CG.Ledger.load(led.path)
    assert [x["state"] for x in st.entries] == ["submitted", "submission_unknown"]
    with pytest.raises(CG.UnresolvedSubmission):                       # retry is refused before any provider call
        H.submit(plan, _RealLike(fake), ledger=led)
    assert len(remote) == 2
    # collection: the lost job is found by its tag, attached, collected and reconciled like the others
    assert CG.resolve_unknown(led, CG.ibm_tag_lookup(service))[0]["outcome"] == "attached"
    got, gone, still = CG.sync_pending(led, e.unknown)
    assert [g["job_id"] for g in got] == ["job-2"] and not gone and not still
    freqs, status = H.collect(service, e.records + got, plan.compiled.implemented.n, ledger=led)
    assert freqs is not None and [s["usage_s"] for s in status] == [0.5, 0.5]
    assert CG.Ledger.load(led.path).committed_s() == pytest.approx(1.0)


def test_hardware_mode_is_explicit(tmp_path, monkeypatch):
    from qemx import hardware as H
    warnings.simplefilter("ignore")
    plan, fake = _hw_plan()
    with pytest.raises(ValueError):
        H.submit(plan, fake, mode="session")
    remote, made, _ = _mock_runtime(monkeypatch, batch_ok=True)
    recs = H.submit(plan, fake, mode="batch")
    assert all(r["mode"] == "batch" for r in recs) and all(m is not fake for m in made)
    remote, made, _ = _mock_runtime(monkeypatch, batch_ok=False)
    recs = H.submit(plan, fake, mode="batch")                          # no batch mode: job mode, and it says so
    assert all(r["mode"] == "job" for r in recs) and all(m is fake for m in made)


def test_hardware_collect_reconciles_reported_usage(tmp_path):
    from qemx import hardware as H

    class Job:
        def __init__(self, st, usage, counts=None):
            self.st, self.u, self.counts = st, usage, counts

        def status(self):
            return self.st

        def usage(self):
            return self.u

        def result(self):
            return [types.SimpleNamespace(data=types.SimpleNamespace(c=types.SimpleNamespace(
                get_counts=lambda: self.counts)))]
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    for jid in ("a", "b", "c"):
        rid = led.admit(jid, _est(20))
        led.submitted(rid, jid)
    jobs = {"a": Job("DONE", None, {"00": 3}), "b": Job("JobStatus.ERROR", 4.0), "c": Job("RUNNING", None),
            "z": Job("DONE", 1.0, {"00": 1})}                          # z: submitted before the ledger existed
    service = types.SimpleNamespace(job=lambda jid: jobs[jid])
    recs = [dict(job_id=j, pubs=[("k" + j, 0)]) for j in ("a", "b", "c", "z")]
    cache = {}
    freqs, status = H.collect(service, recs, 2, cache=cache, ledger=led)
    assert freqs is None                                               # c still runs
    st = {e["job_id"]: (e["state"], e["usage_s"]) for e in CG.Ledger.load(led.path).entries}
    assert st == {"a": ("completed", None), "b": ("failed", 4.0), "c": ("submitted", None)}
    jobs["a"].u, jobs["c"].st, jobs["c"].u, jobs["c"].counts = 2.0, "DONE", 3.0, {"11": 2}
    freqs, status = H.collect(service, recs, 2, cache=cache, ledger=led)
    st = {e["job_id"]: (e["state"], e["usage_s"]) for e in CG.Ledger.load(led.path).entries}
    assert st == {"a": ("completed", 2.0), "b": ("failed", 4.0), "c": ("completed", 3.0)}   # a: reported later
    assert set(freqs) == {"ka", "kc", "kz"} and CG.Ledger.load(led.path).committed_s() == pytest.approx(9.0)


# ------------------------------------------------------------------ the divergence submitter and CLI
def test_divergence_submit_requires_a_ledger_and_keeps_unknown_runs(tmp_path, monkeypatch):
    import qiskit_ibm_runtime
    from qemx import divergence as DVM
    from qemx import hardware as H
    remote = []

    class Job:
        def __init__(self, i):
            self.i = i

        def job_id(self):
            return "fake-%d" % self.i

    class Sampler:
        def __init__(self, mode=None):
            self.options = types.SimpleNamespace(max_execution_time=None,
                                                 environment=types.SimpleNamespace(job_tags=None))

        def run(self, circs, shots):
            remote.append(list(self.options.environment.job_tags))
            if len(remote) == 2:
                raise TimeoutError("lost")
            return Job(len(remote))
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", Sampler)
    monkeypatch.setattr(DVM, "_plain_sampler", lambda s: None)
    runs = [types.SimpleNamespace(key="r%d" % i) for i in range(700)]
    plan = types.SimpleNamespace(runs=runs, suite=types.SimpleNamespace(study="test"))
    monkeypatch.setattr(DVM, "physical_circuits", lambda p, b, rs: [(r.key, "circ", 100) for r in rs])
    with pytest.raises(Exception) as ei:                               # 1.5.0 submitted without a ledger
        DVM.submit(plan, types.SimpleNamespace(name="ibm_x"))
    assert type(ei.value).__name__ == "LedgerRequired" and not remote
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    attempts, saved = [], {}
    with pytest.raises(H.SubmitError) as ei:
        DVM.submit(plan, types.SimpleNamespace(name="ibm_x"), ledger=led, on_attempt=attempts.append,
                   on_record=lambda r: saved.update(r=r))
    assert [len(a["runs"]) for a in attempts] == [300, 300] and attempts[1]["job_id"] is None
    assert attempts[1]["client_tag"] == remote[1][-1] == ei.value.unknown[0]["client_tag"]
    assert ei.value.unknown[0]["runs"] == attempts[1]["runs"] and len(saved["r"]) == 1


def test_divergence_cli_lost_response_is_attached_and_never_resubmitted(tmp_path, monkeypatch):
    """run-ibm: the provider accepts the second job and the response is lost. jobs.json keeps that call's runs
    under "pending"; the next run-ibm finds the job by its tag, attaches it in the ledger and in jobs.json, and
    --resume submits only the runs no job covers: every run is measured exactly once."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import dirA_divergence as DV
    import qiskit_ibm_runtime as rt
    from qemx import device as D
    import test_divergence as TD
    backend = D.fake_backend("fake_fez")
    jobs, fail = {}, {"lose": 2}

    class Service:
        def __init__(self, token=None, **kw):
            pass

        def backend(self, name):
            return backend

        def job(self, jid):
            return jobs[jid]

        def jobs(self, job_tags=None, limit=None, **kw):
            return [j for j in jobs.values() if set(job_tags or []) <= set(j.tags)]

    class Sampler:
        def __init__(self, mode=None):
            self.options = types.SimpleNamespace(max_execution_time=None,
                                                 environment=types.SimpleNamespace(job_tags=None))

        def run(self, circs, shots=None):
            j = TD._Job("job-%d" % len(jobs), len(circs))
            j.tags = list(self.options.environment.job_tags or [])
            j.usage = lambda: 0.25
            jobs[j.job_id()] = j
            if len(jobs) == fail["lose"]:
                raise TimeoutError("read timed out")
            return j
    monkeypatch.setattr(rt, "QiskitRuntimeService", Service)
    monkeypatch.setattr(rt, "SamplerV2", Sampler)
    monkeypatch.setattr(rt, "Batch", lambda **kw: (_ for _ in ()).throw(RuntimeError("no batch")))
    run = str(tmp_path / "run")
    assert DV.main(["plan", TD._write_manifest(tmp_path), "--out", run]) == 0
    led = str(tmp_path / "ledger.json")
    with pytest.raises(SystemExit, match="campaign ledger"):
        DV.main(["collect-ibm", run, "--tag", "hw", "--ledger", led])     # a missing ledger is an error …
    CG.Ledger.create(led, 600, 60)
    L = ["--tag", "hw", "--backend", "fake_fez", "--ledger", led]
    with pytest.raises(SystemExit, match="unknown"):
        DV.main(["run-ibm", run] + L)
    jp = os.path.join(run, "obs", "hw", "jobs.json")
    first = _disk(jp)
    assert len(first["records"]) == 1 and len(first["pending"]) == 1 and first["pending"][0]["job_id"] is None
    assert DV.main(["collect-ibm", run, "--tag", "hw", "--ledger", led]) == 0    # finds the lost job by its tag
    after = _disk(jp)
    assert [r["job_id"] for r in after["records"]] == ["job-0", "job-1"] and after["pending"] == []
    assert DV.main(["run-ibm", run, "--resume"] + L) == 0
    done = _disk(jp)
    keys = [k for r in done["records"] for k in r["runs"]]
    plan = DV.load_plan(run)
    assert sorted(keys) == sorted(r.key for r in plan.runs) and len(keys) == len(set(keys))   # each run once
    assert len(jobs) == len(done["records"])                                                  # no duplicate job
    assert DV.main(["collect-ibm", run, "--tag", "hw", "--ledger", led]) == 0
    final = CG.Ledger.load(led)
    assert {e["state"] for e in final.entries} == {"completed"}
    assert final.committed_s() == pytest.approx(0.25 * len(jobs))
    assert DV.main(["collect-ibm", run, "--tag", "hw", "--no-ledger", "--ledger", "nope.json"]) == 0


def test_hardware_collect_keeps_other_jobs_when_one_result_fails(tmp_path):
    """N14: one job whose result() raises (it ended in error while being waited for) no longer aborts the whole
    collection; the ledger is reconciled only from a status the provider reports as ended."""
    from qemx import hardware as H

    class Job:
        def __init__(self, sts, counts=None):
            self.sts, self.counts = list(sts), counts

        def status(self):
            return self.sts.pop(0) if len(self.sts) > 1 else self.sts[0]

        def usage(self):
            return 1.5

        def result(self):
            if self.counts is None:
                raise RuntimeError("the job ended in error")
            return [types.SimpleNamespace(data=types.SimpleNamespace(c=types.SimpleNamespace(
                get_counts=lambda: self.counts)))]
    led = _ledger(tmp_path, ceiling_s=600, reserve_s=60)
    for jid in ("ok", "bad"):
        led.submitted(led.admit(jid, _est(20)), jid)
    jobs = {"ok": Job(["DONE"], {"01": 4}), "bad": Job(["RUNNING", "ERROR"])}
    freqs, status = H.collect(types.SimpleNamespace(job=lambda jid: jobs[jid]),
                              [dict(job_id="ok", pubs=[("k1", 0)]), dict(job_id="bad", pubs=[("k2", 0)])], 2,
                              wait=True, ledger=led)
    assert set(freqs) == {"k1"} and [s["failed"] for s in status] == [False, True]
    st = {e["job_id"]: (e["state"], e["usage_s"]) for e in CG.Ledger.load(led.path).entries}
    assert st == {"ok": ("completed", 1.5), "bad": ("failed", 1.5)}
