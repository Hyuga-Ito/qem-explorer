"""Sidebar navigation with the roadmap. Streamlit's automatic page list is hidden
(.streamlit/config.toml) so pages can have readable names and planned pages can be shown disabled."""
import streamlit as st

from ui.tooltips import install as _plain_tooltips

_plain_tooltips()          # Unicode math in every help tooltip (ui/tooltips.py); idempotent

PAGES = [
    ("app.py", "Home", ":material/home:", True),
    ("pages/9_Learn.py", "Learn: guides, algorithms & research", ":material/school:", True),
    ("pages/1_Playground.py", "Playground", ":material/science:", True),
    ("pages/8_Batch_Run.py", "Batch run", ":material/playlist_play:", True),
    ("pages/10_QEM_Performance.py", "QEM Performance & Selection", ":material/insights:", True),
    ("pages/11_Noise_Discrepancy.py", "Noise Models & Discrepancy", ":material/difference:", True),
    ("pages/6_Theory.py", "Theory & sources", ":material/menu_book:", True),
    ("pages/7_IBM_Quantum.py", "Run on IBM Quantum", ":material/cloud_upload:", True),
]
# 1.5: the four analysis pages are drill-down views of the two research areas. Their addresses are unchanged
# (bookmarks, Learn setups and tests keep working); the sidebar lists them under their research area.
DRILLDOWNS = {
    "pages/10_QEM_Performance.py": [("pages/2_Cost_vs_Precision.py", "Cost vs precision", ":material/trending_up:"),
                                    ("pages/5_Head_to_Head.py", "Head-to-head sweeps", ":material/compare_arrows:"),
                                    ("pages/3_Bias_Variance.py", "Bias–variance bullseye", ":material/adjust:")],
    "pages/11_Noise_Discrepancy.py": [("pages/4_Model_Mismatch.py", "Model mismatch", ":material/tune:")],
}


MIN_STREAMLIT = (1, 64)      # the oldest Streamlit CI runs the interaction tests with (the lock pins 1.66.0)


def streamlit_version_ok(version: str | None = None) -> bool:
    """Whether the running Streamlit is at least MIN_STREAMLIT (pre-release suffixes are ignored)."""
    import re
    v = version or st.__version__
    parts = [int(x) for x in re.findall(r"\d+", v)[:2]] + [0, 0]
    return tuple(parts[:2]) >= MIN_STREAMLIT


def sidebar_nav():
    with st.sidebar:
        st.markdown("### QEM Explorer")
        if not streamlit_version_ok():
            st.warning("Streamlit %s is older than the oldest tested %d.%d. Install the pinned versions: "
                       "`python -m pip install -r requirements-lock.txt`." % ((st.__version__,) + MIN_STREAMLIT),
                       icon=":material/warning:")
        for path, label, icon, live in PAGES:
            try:
                if live:
                    st.page_link(path, label=label, icon=icon)
                else:
                    st.page_link("app.py", label=label + " · soon", icon=icon, disabled=True)
            except Exception:                      # e.g. a page run as its own entry point
                st.markdown(("%s" if live else "%s · soon") % label)
            subs = DRILLDOWNS.get(path)
            if subs:
                with st.container(key="nav_sub_%s" % path.split("/")[-1].split(".")[0]):
                    for p2, l2, i2 in subs:
                        try:
                            st.page_link(p2, label="↳ " + l2, icon=i2)
                        except Exception:          # noqa: BLE001
                            st.markdown("↳ %s" % l2)
        st.divider()
