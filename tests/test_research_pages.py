"""1.5: the two research areas render, read stored study tables only, recommend (or abstain) for the current
circuit, and keep the four analysis pages reachable as drill-downs."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import qiskit_ibm_runtime  # noqa: F401  (see tests/test_ui.py)
except Exception:                                        # pragma: no cover
    pass


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()


def _w(at, kind, key):
    return next(x for x in getattr(at, kind) if x.key == key)


def test_navigation_lists_the_research_areas_and_their_drilldowns():
    from ui import nav
    paths = [p for p, *_ in nav.PAGES]
    assert "pages/10_QEM_Performance.py" in paths and "pages/11_Noise_Discrepancy.py" in paths
    subs = [p for v in nav.DRILLDOWNS.values() for p, *_ in v]
    for old in ("pages/2_Cost_vs_Precision.py", "pages/3_Bias_Variance.py", "pages/4_Model_Mismatch.py",
                "pages/5_Head_to_Head.py"):
        assert old in subs and os.path.exists(os.path.join(ROOT, old))     # still reachable, same address


@pytest.mark.slow
def test_research_pages_render_and_recommend():
    at = _app()
    at.switch_page("pages/10_QEM_Performance.py").run()
    assert not at.exception, [e.value for e in at.exception]
    if not os.path.isdir(os.path.join(ROOT, "reports", "pilot")):
        pytest.skip("no bundled pilot report")
    assert any("cells" in c.value and "base circuits" in c.value for c in at.caption)
    at.session_state["w_qp_tabs"] = "Recommend for my circuit"
    at.run()
    _w(at, "button", "qp_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    rec = at.session_state["qp_rec"]
    assert rec["kind"] in ("method", "raw", "abstain-coverage", "abstain-uncertain", "no-feasible")
    assert not any(k in rec["features"] for k in ("ideal", "noisy_fidelity", "noisy_purity"))   # no oracle inputs
    at.switch_page("pages/11_Noise_Discrepancy.py").run()
    assert not at.exception, [e.value for e in at.exception]
    at.session_state["w_nd_tabs"] = "Distance test"
    at.run()
    assert not at.exception
    assert any("p-value" in m.label for m in at.metric)
    _w(at, "text_input", "w_nd_c").input("1, 2, x").run()
    assert any("could not convert" in e.value or "invalid literal" in e.value for e in at.error)
