"""1.5.1 (F13): interactions on the stateful, lazy tabs after edits and page switches — not only the first render.

The selected tab is app state (``page_tab``, ui.state.page_tabs): it must survive a rerun caused by another widget, a
validation error, and a switch to another page and back. In 1.5.0, with Streamlit 1.64's AppTest, a rerun triggered by
another widget dropped the keyed ``st.tabs`` state and the page fell back to its first tab; the app now restores the
selection from ``page_tab``, so these tests pass on 1.64.0 (oldest supported, ``ui.nav.MIN_STREAMLIT``) and on the
lock's 1.66.0."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()


def _w(at, kind, key):
    return next(x for x in getattr(at, kind) if x.key == key)


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_minimum_streamlit_version_is_enforced_consistently():
    """requirements.txt, pyproject.toml, the lock and the app's own check name the same minimum."""
    import re
    from ui import nav
    req = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read()
    pyp = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    lock = open(os.path.join(ROOT, "requirements-lock.txt"), encoding="utf-8").read()
    want = "%d.%d" % nav.MIN_STREAMLIT
    assert re.search(r"^streamlit>=%s\b" % re.escape(want), req, re.M), req
    assert '"streamlit>=%s' % want in pyp
    pin = re.search(r"^streamlit==(\d+)\.(\d+)", lock, re.M)
    assert pin and (int(pin.group(1)), int(pin.group(2))) >= nav.MIN_STREAMLIT
    for pkg in ("pyyaml", "jsonschema", "threadpoolctl", "qiskit", "numpy"):   # every runtime dependency is pinned
        assert re.search(r"^%s==" % pkg, lock, re.M | re.I), pkg
    assert nav.streamlit_version_ok("1.66.0") and nav.streamlit_version_ok("1.64.0") and \
        nav.streamlit_version_ok("2.0") and not nav.streamlit_version_ok("1.63.2")


@pytest.mark.slow
def test_distance_test_tab_survives_edits_errors_and_page_switches():
    at = _app()
    at.switch_page("pages/11_Noise_Discrepancy.py").run()
    _ok(at)
    at.session_state["w_nd_tabs"] = "Distance test"
    at.run()
    _ok(at)
    assert any("p-value" in m.label for m in at.metric)
    # an invalid edit: a validation error, the tab and the field stay
    _w(at, "text_input", "w_nd_c").input("1, 2, x").run()
    _ok(at)
    assert at.session_state["w_nd_tabs"] == "Distance test"
    assert any("could not convert" in e.value or "invalid literal" in e.value for e in at.error)
    # a valid edit after the error: the test runs again on the new counts
    _w(at, "text_input", "w_nd_c").input("400, 100, 100, 400").run()
    _ok(at)
    assert at.session_state["w_nd_tabs"] == "Distance test" and not at.error
    assert any("p-value" in m.label for m in at.metric)
    # another page and back: the tab is remembered (page_tab), the page renders without error
    at.switch_page("pages/10_QEM_Performance.py").run()
    _ok(at)
    at.switch_page("pages/11_Noise_Discrepancy.py").run()
    _ok(at)
    assert at.session_state["page_tab"]["nd"] == "Distance test"
    assert at.session_state["w_nd_tabs"] == "Distance test"
    assert any(x.key == "w_nd_c" for x in at.text_input)


@pytest.mark.slow
def test_selected_tabs_survive_page_switches():
    at = _app()
    at.switch_page("pages/10_QEM_Performance.py").run()
    _ok(at)
    at.session_state["w_qp_tabs"] = "Recommend for my circuit"
    at.run()
    _ok(at)
    at.switch_page("pages/1_Playground.py").run()
    _ok(at)
    at.session_state["w_pg_tabs"] = "Export"
    at.run()
    _ok(at)
    at.switch_page("pages/11_Noise_Discrepancy.py").run()
    _ok(at)
    at.switch_page("pages/10_QEM_Performance.py").run()
    _ok(at)
    assert at.session_state["w_qp_tabs"] == "Recommend for my circuit"
    at.switch_page("pages/1_Playground.py").run()
    _ok(at)
    assert at.session_state["w_pg_tabs"] == "Export"
    assert at.session_state["page_tab"]["playground"] == "Export"
