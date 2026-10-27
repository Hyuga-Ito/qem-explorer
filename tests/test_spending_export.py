"""1.5.1 (new finding N10): the exported standalone script's ``--ibm`` path is a real-hardware submitter too. It
submits only through the campaign ledger (admission, client tag, job id, reconciliation) and refuses without one.
Mocked service and sampler; no network."""
import dataclasses
import types
import warnings

from qemx import campaign as CG
from qemx import export as EX
from qemx import templates as T
from qemx.noise import PRESETS
from qemx.recipes import preset

ENV = "QEMX_CAMPAIGN_LEDGER"     # the variable the app and the exported script read


def _script_namespace(tmp_path, monkeypatch):
    tpl = T.bell()
    state = dict(app="QEM Explorer", version="1.5.1", circuit=tpl.spec.to_json(), inputs=["0"] * tpl.spec.n,
                 basis="Z" * tpl.spec.n, relabel=tpl.relabel,
                 compile=dict(mode="abstract", basis_2q="cx", coupling="all", layout="sabre", opt_level=1),
                 noise=dataclasses.asdict(PRESETS["Depolarizing"]),
                 observable=dict(key=tpl.observable.key, label=tpl.observable.label, tex=tpl.observable.tex,
                                 values=list(tpl.observable.values)), symmetry=None,
                 recipes=[preset("Unmitigated").to_json()], backend="numpy", shots=1000, t_shot_us=250.0)
    import importlib.util
    import sys
    f = tmp_path / "qem_explorer_export_t.py"
    f.write_text(EX.script(state), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("qem_explorer_export_t", str(f))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "qem_explorer_export_t", mod)
    spec.loader.exec_module(mod)                     # not __main__: the entry point does not run
    return vars(mod)


def _mocks(monkeypatch):
    import qiskit_ibm_runtime as R
    from qemx import device as D
    fake = D.fake_backend("fake_perth")
    st = types.SimpleNamespace(connects=0, remote={})

    class Job:
        def __init__(self, jid, circs, shots, tags):
            self.jid, self.circs, self.shots, self.tags = jid, circs, shots, tags

        def job_id(self):
            return self.jid

        def status(self):
            return "DONE"

        def usage(self):
            return 2.5

        def result(self):
            return [types.SimpleNamespace(data=types.SimpleNamespace(c=types.SimpleNamespace(
                get_counts=lambda c=c: {"0" * c.num_clbits: self.shots}))) for c in self.circs]

    class Sampler:
        def __init__(self, mode=None):
            self.options = types.SimpleNamespace(environment=types.SimpleNamespace(job_tags=None),
                                                 max_execution_time=None)

        def run(self, circs, shots=None):
            jid = "job-%d" % (len(st.remote) + 1)
            st.remote[jid] = Job(jid, list(circs), shots, list(self.options.environment.job_tags or []))
            return st.remote[jid]

    class Service:
        def __init__(self, *a, **kw):
            st.connects += 1

        def backend(self, name=None):
            return fake

        def least_busy(self, **kw):
            return fake
    monkeypatch.setattr(R, "SamplerV2", Sampler)
    monkeypatch.setattr(R, "QiskitRuntimeService", Service)
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "FAKE-TOKEN-xyz")
    return st


def test_exported_script_submits_only_through_the_ledger(tmp_path, monkeypatch, capsys):
    warnings.simplefilter("ignore")
    ns = _script_namespace(tmp_path, monkeypatch)
    st = _mocks(monkeypatch)
    monkeypatch.delenv(ENV, raising=False)
    assert ns["run_on_ibm"]("ibm_x") == 2                              # no ledger: no login, no job
    monkeypatch.setenv(ENV, str(tmp_path / "typo.json"))
    assert ns["run_on_ibm"]("ibm_x") == 2
    assert st.connects == 0 and not st.remote
    assert "not submitted" in capsys.readouterr().out
    lp = str(tmp_path / "ledger.json")
    CG.Ledger.create(lp, 600, 60)
    monkeypatch.setenv(ENV, lp)
    assert ns["run_on_ibm"]("ibm_x") == 0
    (e,) = CG.Ledger.load(lp).entries
    assert len(st.remote) == 1 and e["job_id"] == "job-1" and e["client_tag"] in st.remote["job-1"].tags
    assert (e["state"], e["usage_s"]) == ("completed", 2.5)            # reconciled after the result
    assert "FAKE-TOKEN" not in capsys.readouterr().out
