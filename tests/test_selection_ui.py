"""1.6.0: the QEM Performance page's protocol-selection tab reads the bundled study (no simulation) and drills down
to a task's protocol risks, every policy's choice, the task and circuit, the protocols' settings and allocation, the
shot ledger, the trial distribution and the provenance of the labels."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = next((os.path.join(ROOT, "reports", n) for n in ("selection_v170", "selection_pilot")
              if os.path.exists(os.path.join(ROOT, "reports", n, "analysis", "results.json"))),
             os.path.join(ROOT, "reports", "selection_pilot"))
try:
    import qiskit_ibm_runtime  # noqa: F401  (see tests/test_ui.py)
except Exception:                                        # pragma: no cover
    pass


def _has_trial_arrays(study: str) -> bool:
    t = os.path.join(study, "dev", "tasks")
    return os.path.isdir(t) and any(f.endswith(".npz") for f in os.listdir(t))


def _w(at, kind, key):
    return next(x for x in getattr(at, kind) if x.key == key)


@pytest.mark.slow
def test_selection_tab_reads_the_bundled_study_and_drills_down():
    if not os.path.isdir(STUDY):
        pytest.skip("no bundled selection study")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()
    at.switch_page("pages/10_QEM_Performance.py").run()
    at.session_state["w_qp_tabs"] = "Protocol selection study"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert _w(at, "selectbox", "w_sel_study").value == os.path.basename(STUDY)     # the newest study first
    assert any("Simulated observations" in c.value for c in at.caption)
    assert len(at.dataframe) >= 1                                        # the policy table
    at.session_state["w_sel_view"] = "Task drill-down"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    fam = _w(at, "selectbox", "w_sel_fam")
    assert fam.options and _w(at, "selectbox", "w_sel_task").options
    if not _has_trial_arrays(STUDY):
        # the source archive ships the corrected pilot without its per-trial arrays (trial-data archive): the
        # drill-down explains what is missing instead of failing, and the figures view still works
        assert any("trial files of this task are not in the study folder" in i.value for i in at.info)
        at.session_state["w_sel_view"] = "Figures"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return
    # drill-down: task and circuit, settings and allocation, ledger, trial distribution, provenance
    assert any(m.value.startswith("#### Task ") for m in at.markdown)
    assert any("Shot ledger at B = " in m.value for m in at.markdown)
    assert any("Trial distribution at B = " in m.value for m in at.markdown)
    assert any(e.label == "Provenance of these labels" for e in at.expander)
    n_tables = len(at.dataframe)
    at.session_state["w_sel_circ"] = True
    at.session_state["w_sel_half"] = "selection"
    at.session_state["w_sel_B"] = 1000
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(c.value.strip() for c in at.code)                         # the logical circuit's text drawing
    assert any("Trial distribution at B = 1000**" in m.value and "(selection half" in m.value for m in at.markdown)
    assert len(at.dataframe) == n_tables
    at.session_state["w_sel_view"] = "Figures"
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.slow
def test_recommend_tab_uses_the_bundle_and_marks_stale_results(tmp_path, monkeypatch):
    """1.7.0: "Recommend for my circuit" turns the Playground state into the shared DeploymentTask, recommends with a
    model bundle (no pickle) exactly as the CLI does, and marks the result out of date when the budget changes."""
    if not os.path.isdir(os.path.join(STUDY, "dev")):
        pytest.skip("no bundled selection study")
    if not _has_trial_arrays(STUDY):
        pytest.skip("the study's per-trial arrays are not extracted (trial-data archive, reports/selection_v170)")
    import numpy as np
    from qemx.selection import dataset as DT
    from qemx.selection import deploy as DP
    ds = DT.build(os.path.join(STUDY, "dev"))
    bdir = tmp_path / "bundles"
    DP.build_bundle(ds, np.arange(len(ds.rows)), "ridge", "deployable", str(bdir / "qemx-rec-test"),
                    version="qemx-rec-test", tau=0.0)
    monkeypatch.setattr(DP, "BUNDLE_DIR", str(bdir))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()
    at.switch_page("pages/10_QEM_Performance.py").run()
    at.session_state["w_qp_tabs"] = "Recommend for my circuit"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert _w(at, "selectbox", "w_sr_bundle").value == "qemx-rec-test"
    at.session_state["w_sr_dev"] = "fake_fez"
    at.session_state["w_sr_B"] = 100000
    at.run()
    _w(at, "button", "sr_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    rec = at.session_state["sr_rec"]
    assert rec["status"] in ("ok", "out-of-distribution", "no-model") and rec["choice"]
    assert not at.error, [e.value for e in at.error]
    assert any("feature" in str(df.value.columns.tolist()) for df in at.dataframe)       # the feature report
    assert rec["task"]["device"] == "fake_fez" and rec["task"]["budget"] == 100000
    at.session_state["w_sr_B"] = 200000                                    # a changed budget: stale
    at.run()
    assert any("changed since this recommendation" in w.value for w in at.warning)
