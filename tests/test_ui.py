"""Pages and session state through Streamlit's AppTest: every page renders, Run buttons work, and settings
survive reruns, tab switches and page switches (the state-reset regressions)."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["pages/1_Playground.py", "pages/2_Cost_vs_Precision.py", "pages/3_Bias_Variance.py",
         "pages/4_Model_Mismatch.py", "pages/5_Head_to_Head.py", "pages/6_Theory.py", "pages/7_IBM_Quantum.py"]

# Importing qiskit-ibm-runtime for the first time inside AppTest's script thread can crash the interpreter
# (a native-extension issue specific to the test harness; the live server is unaffected). Import it here.
try:
    import qiskit_ibm_runtime  # noqa: F401
    import qiskit_ibm_runtime.fake_provider  # noqa: F401
except Exception:                                            # pragma: no cover
    pass


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _w(at, kind, key):
    return next(x for x in getattr(at, kind) if x.key == key)


@pytest.mark.slow
def test_every_page_renders():
    at = _app()
    _ok(at)
    for p in PAGES:
        at.switch_page(p).run()
        _ok(at)


@pytest.mark.slow
def test_playground_run_and_tabs():
    at = _app()
    at.switch_page(PAGES[0]).run()
    at.session_state["run_requested"] = True
    at.run()
    _ok(at)
    assert at.session_state["results"] is not None and at.session_state["timeline"] is not None
    for tab in ["Circuits", "Time travel", "Mitigate & compare", "Export"]:
        at.session_state["w_pg_tabs"] = tab
        at.run()
        _ok(at)


@pytest.mark.slow
def test_settings_survive_page_switches():
    """The original bug: values typed on one page were reset after visiting another page."""
    at = _app()
    at.switch_page(PAGES[0]).run()
    _w(at, "selectbox", "w_tpl_cat").set_value("Basics").run()
    _w(at, "selectbox", "w_tpl").set_value("ghz").run()
    _w(at, "number_input", "w_tp_ghz_n").set_value(5).run()
    _w(at, "number_input", "w_shots_num").set_value(12345).run()
    _w(at, "number_input", "w_nz_p2_num").set_value(0.0234).run()
    _w(at, "select_slider", "w_cm_opt").set_value(3).run() if any(x.key == "w_cm_opt" for x in at.select_slider) else None
    _ok(at)
    assert at.session_state["shots"] == 12345 and abs(at.session_state["noise"].p2 - 0.0234) < 1e-12
    at.switch_page("app.py").run()
    at.switch_page(PAGES[2]).run()
    at.switch_page(PAGES[0]).run()
    _ok(at)
    assert _w(at, "selectbox", "w_tpl").value == "ghz"
    assert _w(at, "number_input", "w_tp_ghz_n").value == 5
    assert _w(at, "number_input", "w_shots_num").value == 12345
    assert abs(_w(at, "number_input", "w_nz_p2_num").value - 0.0234) < 1e-12
    assert at.session_state["template"] == "qft"          # selecting a template does not load it


@pytest.mark.slow
def test_template_load_resize_observable_and_text():
    at = _app()
    at.switch_page(PAGES[0]).run()
    _w(at, "selectbox", "w_tpl").set_value("aqft").run()
    _w(at, "button", "tpl_load").click().run()
    _ok(at)
    assert at.session_state["template"] == "aqft" and at.session_state["observable"].label.startswith("P(")
    # inputs follow the register and keep what the user set
    _w(at, "selectbox", "w_in_1").set_value("+").run()
    _ok(at)
    assert at.session_state["inputs"][1] == "+"
    _w(at, "selectbox", "w_obs").set_value("custom").run()
    _w(at, "text_input", "w_obs_bits").input("01x").run()
    _ok(at)
    assert any("0 or 1" in e.value for e in at.error)
    _w(at, "text_input", "w_obs_bits").input("0110").run()
    assert at.session_state["observable"].label == "P(0110)"
    from ui import state as S  # noqa: F401  (module import for the callback below)
    at.session_state["w_txt_%d" % at.session_state["ops_version"]] = " "
    at.run()
    _w(at, "button", "txt_apply").click().run()
    _ok(at)
    assert any("empty" in e.value for e in at.error)


@pytest.mark.slow
def test_fake_device_backend_runs():
    at = _app()
    at.switch_page(PAGES[0]).run()
    _w(at, "selectbox", "w_backend").set_value("fake_perth").run()
    _ok(at)
    at.session_state["run_requested"] = True
    at.run()
    _ok(at)
    res = at.session_state["results"]
    assert res is not None and res["backend"] == "fake_perth"
    assert at.session_state["device_info"]["phys"]


@pytest.mark.slow
def test_analysis_pages_run():
    at = _app()
    at.switch_page(PAGES[1]).run()
    at.session_state["cvp_run"] = True
    at.run()
    _ok(at)
    at.session_state["w_cvp_tabs"] = "Depth sweep"
    at.session_state["w_ds_depths"] = "0, 1"
    at.session_state["ds_run"] = True
    at.run()
    _ok(at)
    assert at.session_state["ds_result"]["rows"]
    at.switch_page(PAGES[2]).run()
    at.session_state["bb_run"] = True
    at.run()
    _ok(at)
    assert at.session_state["bb_result"] is not None
    at.switch_page(PAGES[3]).run()
    at.session_state["mm_run"] = True
    at.run()
    _ok(at)
    assert at.session_state["mm_result"]["rows"]
    at.session_state["w_mm_tabs"] = "Mismatch map"
    at.session_state["w_mm_gpts"] = 3
    at.session_state["mm_map_run"] = True
    at.run()
    _ok(at)
    assert at.session_state["mm_map"]["B"].shape == (3, 3)
    at.switch_page(PAGES[4]).run()
    at.session_state["w_hh_pts"] = 3
    at.session_state["hh_run"] = True
    at.run()
    _ok(at)
    assert at.session_state["hh_result"]["rows"]


@pytest.mark.slow
def test_token_never_kept(tmp_path, monkeypatch):
    """Submitting with a (fake) token fails to connect; the token is cleared and appears nowhere in the state or the
    error. The service is mocked (1.5.0 sent the fake token to the provider's sign-in endpoint over the network)
    and a valid campaign ledger is configured, so the connection path itself is exercised offline."""
    import qiskit_ibm_runtime as R
    from qemx import campaign as CG

    class Refuses:
        def __init__(self, *a, token=None, **kw):
            raise ValueError("invalid token %s for this channel" % token)
    monkeypatch.setattr(R, "QiskitRuntimeService", Refuses)
    led = CG.Ledger.create(str(tmp_path / "ledger.json"), ceiling_s=600, reserve_s=60)
    monkeypatch.setenv(CG.ENV_LEDGER, led.path)
    at = _app()
    at.switch_page(PAGES[6]).run()
    at.session_state["w_hw_mode"] = "ibm"
    at.run()
    _ok(at)
    secret = "FAKE-TOKEN-1234567890-abcdef"
    _w(at, "text_input", "hw_token").input(secret)
    btn = next(b for b in at.button if "Submit to IBM Quantum" in (b.label or ""))
    btn.click().run()
    _ok(at)
    assert at.session_state["hw_token"] == ""
    blob = repr({k: at.session_state[k] for k in at.session_state})
    assert secret not in blob
    assert at.error and all(secret not in e.value for e in at.error)
    assert not CG.Ledger.load(led.path).entries                        # nothing was admitted or submitted
    from ui import state as S  # noqa: F401


@pytest.mark.slow
def test_template_families_and_new_templates_load():
    at = _app()
    at.switch_page(PAGES[0]).run()
    for cat, key in (("FTQC · error correction", "steane"), ("NISQ · variational", "qaoa"),
                     ("FTQC · deep algorithms", "shor")):
        _w(at, "selectbox", "w_tpl_cat").set_value(cat).run()
        _w(at, "selectbox", "w_tpl").set_value(key).run()
        _w(at, "button", "tpl_load").click().run()
        _ok(at)
        assert at.session_state["template"] == key
    # a template that cannot be built reports an error instead of crashing the page
    _w(at, "selectbox", "w_tpl_cat").set_value("NISQ · variational").run()
    _w(at, "selectbox", "w_tpl").set_value("qaoa").run()
    _w(at, "number_input", "w_tp_qaoa_n").set_value(5).run()
    _w(at, "selectbox", "w_tp_qaoa_graph").set_value("3-regular").run()
    _w(at, "button", "tpl_load").click().run()
    _ok(at)
    assert at.session_state["template"] == "shor"                 # the previous circuit is kept
    assert any("Could not build this template" in e.value for e in at.error)


@pytest.mark.slow
def test_degenerate_learning_shows_a_warning_not_a_constant():
    at = _app()
    at.switch_page(PAGES[0]).run()
    _w(at, "selectbox", "w_tpl_cat").set_value("NISQ · variational").run()
    _w(at, "selectbox", "w_tpl").set_value("qaoa").run()
    _w(at, "button", "tpl_load").click().run()
    _w(at, "selectbox", "w_obs").set_value("Z0").run()      # ⟨Z0⟩ = 0 for every QAOA circuit (X⊗n symmetry)
    assert at.session_state["observable"].key == "Z0"
    at.session_state["w_rc_pick"] = ["Unmitigated", "CDR · OLS"]
    at.run()
    at.session_state["run_requested"] = True
    at.session_state["w_pg_tabs"] = "Mitigate & compare"
    at.run()
    _ok(at)
    rows = at.session_state["results"]["rows"]
    cdr = next(r for r in rows if r["family"] == "learning")
    assert cdr["status"].startswith("not applied")
    assert any("cannot train" in w.value for w in at.warning)
