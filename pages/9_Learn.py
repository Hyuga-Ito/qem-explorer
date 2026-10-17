"""Learn: quantum-computing fundamentals for error mitigation, algorithm walkthroughs (VQE, QAOA, Shor, HHL, QEC,
Grover, HSP), the research topics behind QEM Explorer's studies (equal budgets, folding, context noise, device
divergence, hardware runs) and step-by-step hands-on guides that set up the Playground or an analysis page in one
click."""
import re

import streamlit as st

st.set_page_config(page_title="Learn · QEM Explorer", page_icon=":material/school:", layout="wide")

from qemx import learn as L  # noqa: E402
from ui import state as S  # noqa: E402
from ui import figures as FG  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.nav import sidebar_nav  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
target = ss.pop("goto", None)
if target:                                       # a setup button was pressed: its callback prepared the state
    st.switch_page(target)
sidebar_nav()


@st.cache_data(show_spinner=False)
def chapter_sections(key: str) -> tuple:
    """(title, intro, [(section title, body)]) of a chapter."""
    text = L.chapter(key)
    head, *parts = re.split(r"(?m)^## ", text)
    title = re.search(r"(?m)^# (.+)$", head).group(1).strip()
    intro = re.sub(r"(?m)^# .+$", "", head, count=1).strip()
    secs = []
    for p in parts:
        t, _, body = p.partition("\n")
        secs.append((t.strip(), body.strip()))
    return title, intro, secs


def setup_button(label: str, setup: dict, key: str, primary: bool = True):
    where = L.setup_label(setup)[len("Set up "):]                  # "in the Playground", "in Cost vs precision", …
    page = where.split(" ", 1)[1] if where.startswith(("in ", "on ")) else where
    mode = "merge" if ss.get("w_learn_merge") else "replace"
    st.button(label or L.setup_label(setup), key=key, on_click=S.apply_setup, args=(setup, mode),
              type="primary" if primary else "secondary", icon=":material/rocket_launch:",
              help="Loads the circuit, noise and contenders%s and opens %s. %s Your current circuit stays in the "
                   "Playground's undo history, and your saved custom recipes stay saved." % (
                       " (the %s tab)" % setup["tab"] if "tab" in setup else "", page,
                       "The guide's contenders are added to the ones you have." if mode == "merge" else
                       "The guide's contenders replace the active ones, so the guide starts from its own experiment."))


def chapter_view(k: str, setups: list, key_prefix: str, caption: str):
    """A chapter with its setup buttons and table of contents on the right."""
    title, intro, secs = chapter_sections(k)
    left, right = st.columns([2.4, 1], gap="large")
    with right:
        with st.container(border=True):
            st.markdown("**Try it**")
            for j, (label, setup) in enumerate(setups):
                setup_button(label, setup, "%s_%s_%d" % (key_prefix, k, j), primary=j == 0)
            st.caption(caption)
        with st.container(border=True):
            st.markdown("**On this page**")
            st.markdown("\n".join("- %s" % t for t, _ in secs))
    with left:
        st.subheader(title)
        FG.markdown(intro)
        for t, body in secs:
            st.markdown("#### %s" % t)
            FG.markdown(body)


st.title("Learn")
st.markdown("Everything you need to read the rest of the app: the quantum-computing basics behind error mitigation, "
            "the algorithms it is tested on, the research methods of QEM Explorer's studies, and short hands-on "
            "guides. Every guide, walkthrough and topic has a button that sets up the right page for you.")
if ss.get("learn_error"):
    st.error(W.md_escape(ss.learn_error), icon=":material/error:")
S.wdefault("w_learn_merge", False)
st.toggle("Keep my current contenders when a guide sets up a page", key="w_learn_merge", persist_state=P,
          help="Off (recommended): a guide's contenders replace the active ones, so every guide starts from the same "
               "experiment. On: they are added to yours. Saved custom recipes are never deleted either way.")

TABS = ["Hands-on guides", "Fundamentals", "Algorithm walkthroughs", "Research topics"]
S.wdefault("w_learn_tabs", ss.page_tab.get("learn", TABS[0]))
tabs = st.tabs(TABS, key="w_learn_tabs", on_change=S.cb_tab, args=("learn", "w_learn_tabs"))

with tabs[0]:
    if tabs[0].open is not False:
        st.caption("Each guide takes 8–20 minutes. Press its setup button, then follow the steps on the page it opens; "
                   "come back here for the next step (the Learn page remembers where you were).")
        levels = {"beginner": ":green-badge[beginner]", "intermediate": ":blue-badge[intermediate]",
                  "advanced": ":violet-badge[advanced]"}
        number = 0
        for track, track_title in L.GUIDE_TRACKS.items():
            st.markdown("##### %s" % track_title)
            for g in [g_ for g_ in L.GUIDES if L.guide_track(g_) == track]:
                number += 1
                with st.container(border=True):
                    c = st.columns([4, 1.3], vertical_alignment="center")
                    c[0].markdown("**%d. %s** %s :gray-badge[%d min]  \n%s" % (
                        number, g["title"], levels[g["level"]], g["minutes"], g["goal"]))
                    with c[1]:
                        setup_button("", g["setup"], "guide_%s" % g["key"])
                    with st.expander("Steps", icon=":material/format_list_numbered:", key="exp_guide_%s" % g["key"]):
                        st.markdown("\n".join("%d. %s" % (k + 1, s) for k, s in enumerate(g["steps"])))
                        for q, a in g["check"]:
                            with st.popover("Check: " + q, icon=":material/quiz:"):
                                st.markdown(a)

with tabs[1]:
    if tabs[1].open is not False:
        title, intro, secs = chapter_sections("fundamentals")
        st.subheader(title)
        FG.markdown(intro)
        for t, body in secs:
            with st.expander(t, expanded=False, key="exp_fund_%s" % re.sub(r"\W+", "_", t.lower())[:40]):
                FG.markdown(body)

with tabs[2]:
    if tabs[2].open is not False:
        S.wdefault("w_learn_algo", "vqe")
        keys = [k for k in L.CHAPTERS if k != "fundamentals"]
        if ss.w_learn_algo not in keys:
            ss.w_learn_algo = "vqe"
        st.segmented_control("Algorithm", keys, key="w_learn_algo", format_func=lambda k: L.CHAPTERS[k][0],
                             persist_state=P)
        k = ss.get("w_learn_algo") or "vqe"
        chapter_view(k, L.ALGORITHM_SETUPS.get(k, []), "algo",
                     "Loads the template with the noise model and contenders that suit it.")

with tabs[3]:
    if tabs[3].open is not False:
        st.caption("The methods behind QEM Explorer's research studies, in the order a thesis uses them: compare methods "
                   "at one budget, understand what folding amplifies, find context-dependent noise, measure where a "
                   "device departs from its own noise model, and run it on IBM hardware. Part VI of *Theory & sources* "
                   "has the formal treatment.")
        S.wdefault("w_learn_topic", "budget")
        keys = list(L.TOPICS)
        if ss.w_learn_topic not in keys:
            ss.w_learn_topic = "budget"
        st.segmented_control("Topic", keys, key="w_learn_topic", format_func=lambda k: L.TOPICS[k][0],
                             persist_state=P)
        k = ss.get("w_learn_topic") or "budget"
        chapter_view(k, L.TOPIC_SETUPS.get(k, []), "topic",
                     "Opens the page each step needs, with the circuit, noise and contenders already chosen.")
