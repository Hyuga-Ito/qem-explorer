"""1.5.1 (review F04/F05): the IBM Quantum page never submits without a valid campaign ledger, says why, records job
ids in the ledger at submission, reconciles reported usage when collecting, keeps a submission whose outcome is
unknown committed (and finds it again by its tag), and exposes job/batch mode with job mode as the default.
Everything is mocked: the service, the sampler and the backend; the token is a fake string."""
import os
import types
import warnings

import pytest

from qemx import campaign as CG

ENV = "QEMX_CAMPAIGN_LEDGER"     # the variable the app and the exported script read

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = "FAKE-TOKEN-not-a-secret-123"

try:                                       # see tests/test_ui.py: import the runtime outside AppTest's thread
    import qiskit_ibm_runtime  # noqa: F401
    import qiskit_ibm_runtime.fake_provider  # noqa: F401
except Exception:                          # pragma: no cover
    pass


def _script(root):
    import os
    import sys
    if root not in sys.path:
        sys.path.insert(0, root)
    import streamlit as st
    from ui import state as S
    S.init()
    ss = st.session_state
    if not ss.get("_t_loaded"):
        S.from_json({"circuit": {"n": 2, "name": "bell", "ops": [
            {"name": "h", "qubits": [0], "params": []}, {"name": "cx", "qubits": [0, 1], "params": []}]},
            "observable": {"pauli": "Z0*Z1"}, "recipes": [{"estimator": "raw"}]})
        ss._t_loaded = True
        ss["w_hw_mode"] = "ibm"
    path = os.path.join(root, "pages", "7_IBM_Quantum.py")
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), {"__name__": "__main__", "__file__": path})


@pytest.fixture
def provider(monkeypatch):
    """A mocked IBM service: a real-like backend (not a fake class), a sampler that creates remote jobs (and can
    lose the response of chosen calls), job search by tag, job usage."""
    import qiskit_ibm_runtime as R
    from qemx import device as D
    warnings.simplefilter("ignore")
    fake = D.fake_backend("fake_perth")
    st = types.SimpleNamespace(remote={}, lose=set(), samplers=[], batch_ok=False, connects=0)

    class RealLike:
        name = "ibm_mockperth"

        def __getattr__(self, k):
            return getattr(fake, k)

    class Job:
        def __init__(self, jid, circs, shots, tags):
            self.jid, self.circs, self.shots, self.tags = jid, circs, shots, tags

        def job_id(self):
            return self.jid

        def status(self):
            return "DONE"

        def usage(self):
            return 1.25

        def result(self):
            pub = lambda c: types.SimpleNamespace(data=types.SimpleNamespace(c=types.SimpleNamespace(
                get_counts=lambda: {"0" * c.num_clbits: self.shots})))
            return [pub(c) for c in self.circs]

    class Sampler:
        def __init__(self, mode=None):
            st.samplers.append(mode)
            self.options = types.SimpleNamespace(twirling=types.SimpleNamespace(),
                                                 dynamical_decoupling=types.SimpleNamespace(),
                                                 environment=types.SimpleNamespace(job_tags=None),
                                                 max_execution_time=None)

        def run(self, pubs, shots=None):
            jid = "mockjob-%d" % (len(st.remote) + 1)
            st.remote[jid] = Job(jid, list(pubs), shots, list(self.options.environment.job_tags or []))
            if len(st.remote) in st.lose:
                raise TimeoutError("read timed out")
            return st.remote[jid]

    class Batch:
        def __init__(self, backend=None):
            if not st.batch_ok:
                raise RuntimeError("no batch mode")

        def close(self):
            pass

    class Service:
        def __init__(self, *a, **kw):
            st.connects += 1

        def backend(self, name=None):
            return RealLike()

        def least_busy(self, **kw):
            return RealLike()

        def job(self, jid):
            return st.remote[jid]

        def jobs(self, job_tags=None, limit=None, **kw):
            return [j for j in st.remote.values() if set(job_tags or []) <= set(j.tags)]
    monkeypatch.setattr(R, "SamplerV2", Sampler)
    monkeypatch.setattr(R, "Batch", Batch)
    monkeypatch.setattr(R, "QiskitRuntimeService", Service)
    return st


def _page(monkeypatch, ledger_env):
    from streamlit.testing.v1 import AppTest
    if ledger_env is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, ledger_env)
    at = AppTest.from_function(_script, default_timeout=600, kwargs=dict(root=ROOT)).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _submit(at):
    next(x for x in at.text_input if x.key == "hw_token").input(TOKEN)
    next(b for b in at.button if "Submit to IBM Quantum" in (b.label or "")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["hw_token"] == ""
    return at


def _fetch(at):
    next(x for x in at.text_input if x.key == "hw_token2").input(TOKEN)
    next(b for b in at.button if "fetch results" in (b.label or "")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _plain(s):
    return s.replace("\\", "")


@pytest.mark.parametrize("env", [None, "missing"])
def test_no_valid_ledger_means_no_submission(tmp_path, monkeypatch, provider, env):
    """Unset, or set to a path that does not exist (a typo): a clear message, no login, no job."""
    target = None if env is None else str(tmp_path / "typo" / "campaign_ledger.json")
    at = _page(monkeypatch, target)
    banner = " ".join(_plain(e.value) for e in at.error)
    assert "Submission is disabled until the campaign QPU ledger is set up" in banner
    assert ("does not exist" in banner) if env else ("set QEMX_CAMPAIGN_LEDGER" in banner)
    _submit(at)
    assert not provider.remote and not provider.samplers and provider.connects == 0
    assert "Not submitted — campaign QPU ledger" in _plain(at.session_state["hw_err"])
    assert not at.session_state["hw_jobs"] if "hw_jobs" in at.session_state else True
    assert sorted(os.listdir(tmp_path)) == []                    # a typo creates nothing (no lock file either)
    blob = repr({k: at.session_state[k] for k in at.session_state})
    assert TOKEN not in blob


def test_invalid_ledger_file_means_no_submission(tmp_path, monkeypatch, provider):
    bad = tmp_path / "campaign_ledger.json"
    bad.write_text("{\"schema\": \"something else\"}", encoding="utf-8")
    at = _submit(_page(monkeypatch, str(bad)))
    assert not provider.remote and "not a campaign ledger" in _plain(at.session_state["hw_err"])


def test_submission_records_job_ids_and_collection_reconciles_usage(tmp_path, monkeypatch, provider):
    lp = str(tmp_path / "campaign_ledger.json")
    CG.Ledger.create(lp, 600, 60)
    at = _page(monkeypatch, lp)
    assert any("Campaign QPU ledger" in _plain(i.value) for i in at.info)
    assert at.session_state["hw_exec_mode"] == "job"             # job mode is the default
    _submit(at)
    recs = at.session_state["hw_jobs"]["records"]
    led = CG.Ledger.load(lp)
    assert recs and [r["job_id"] for r in recs] == [e["job_id"] for e in led.entries]
    assert all(e["state"] == "submitted" for e in led.entries) and all(r["mode"] == "job" for r in recs)
    assert all(r["client_tag"] in provider.remote[r["job_id"]].tags for r in recs)
    assert led.committed_s() == pytest.approx(sum(e["reserved_s"] for e in led.entries))     # = Σ provider limits
    assert all(e["max_execution_time_s"] <= e["reserved_s"] for e in led.entries)
    _fetch(at)
    led = CG.Ledger.load(lp)
    assert [(e["state"], e["usage_s"]) for e in led.entries] == [("completed", 1.25)] * len(recs)
    assert led.committed_s() == pytest.approx(1.25 * len(recs))
    assert "reconciled into the campaign ledger" in _plain(at.session_state["hw_msg"])
    assert at.session_state["hw_res"] is not None


def test_unknown_outcome_stays_committed_blocks_resubmission_and_is_found_by_tag(tmp_path, monkeypatch, provider):
    lp = str(tmp_path / "campaign_ledger.json")
    CG.Ledger.create(lp, 600, 60)
    provider.lose = {1}                                          # the job is created, the response is lost
    at = _submit(_page(monkeypatch, lp))
    jobs = at.session_state["hw_jobs"]
    assert jobs["records"] == [] and len(jobs["unknown"]) == 1 and len(provider.remote) == 1
    assert "outcome of the next submission is unknown" in _plain(at.session_state["hw_err"])
    led = CG.Ledger.load(lp)
    assert [e["state"] for e in led.entries] == ["submission_unknown"] and led.committed_s() > 0
    _submit(at)                                                  # a second click sends nothing
    assert len(provider.remote) == 1 and "outcome is unknown" in _plain(at.session_state["hw_err"])
    _fetch(at)                                                   # found by its client tag, attached, collected
    led = CG.Ledger.load(lp)
    assert [(e["state"], e["job_id"], e["usage_s"]) for e in led.entries] == [("completed", "mockjob-1", 1.25)]
    assert at.session_state["hw_jobs"]["unknown"] == [] and at.session_state["hw_res"] is not None


def test_batch_mode_is_a_deliberate_choice(tmp_path, monkeypatch, provider):
    lp = str(tmp_path / "campaign_ledger.json")
    CG.Ledger.create(lp, 600, 60)
    provider.batch_ok = True
    at = _page(monkeypatch, lp)
    next(x for x in at.segmented_control if x.key == "hw_exec_mode").set_value("batch").run()
    _submit(at)
    recs = at.session_state["hw_jobs"]["records"]
    assert recs and all(r["mode"] == "batch" for r in recs)
    assert "batch mode" in _plain(at.session_state["hw_msg"])
