"""The Learn overhaul of 1.4.2: the Research topics chapters, their setups (Playground and analysis pages), the research
guides, and the numbers the chapters quote (each one reproduced from the app's own engine, so the text cannot drift
away from the code)."""
import ast
import os
import re

import numpy as np
import pytest

from qemx import analysis as A
from qemx import compile as CP
from qemx import learn as L
from qemx import observables as OBS
from qemx import report as RP
from qemx import templates as T
from qemx.circuit import parse
from qemx.methods import Recipe
from qemx.noise import PRESETS
from qemx.recipes import PRESETS as RECIPE_PRESETS, compare, preset, table_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRITISH = re.compile(r"\b(colou?r(?<=colour)\w*|modell(ed|ing)\b|behaviour\w*|analys(e|ed|ing)\b|optimis\w*|"
                     r"normalis\w*|centre\w*|favour\w*|neighbour\w*|recognis\w*|summaris\w*|characteris\w*|"
                     r"randomis\w*|organis(e|ed|ing|ation)\b)", re.I)


def _page_tabs(page: str) -> list:
    src = open(os.path.join(ROOT, page), encoding="utf-8").read()
    m = re.search(r"^TABS = (\[.*?\])", src, re.M)
    return ast.literal_eval(m.group(1)) if m else []


def _all_setups():
    for g in L.GUIDES:
        yield "guide " + g["key"], g["setup"]
    for k, lst in list(L.ALGORITHM_SETUPS.items()) + list(L.TOPIC_SETUPS.items()):
        for label, s in lst:
            yield "%s: %s" % (k, label), s


# ------------------------------------------------------------------ chapters
def test_every_topic_chapter_is_clean_markdown():
    assert list(L.TOPICS) == ["budget", "folding", "context", "divergence", "hardware"]
    for k in list(L.CHAPTERS) + list(L.TOPICS):
        text = L.chapter(k)
        assert text.startswith("# "), k
        assert text.count("$") % 2 == 0, k                              # balanced math
        assert text.count("```") % 2 == 0, k
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text), k  # no "\t"heta accidents
        assert not BRITISH.search(text), (k, BRITISH.search(text).group(0))
        assert "forte" not in text.lower() and "ionq" not in text.lower(), k
    for k in L.TOPICS:
        assert "<!-- fig:" not in L.chapter(k), k                  # no figure marker without its drawing
        assert L.TOPIC_SETUPS.get(k), k
        assert "## Try it" in L.chapter(k), k


def test_theory_handbook_has_the_research_part_and_no_control_characters():
    text = open(os.path.join(ROOT, "docs", "theory.md"), encoding="utf-8").read()
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text)
    for h in ("### 16.7 Chained mitigation", "## 18. Equal-budget phase diagrams", "## 19. A device and its component",
              "## 20. Context-dependent noise", "## 21. Explaining the gap"):
        assert h in text, h
    assert text.count("$$") % 2 == 0


def test_chapters_name_only_buttons_that_exist():
    """The 'Try it' text of a topic names its setup buttons by their labels."""
    for k, lst in L.TOPIC_SETUPS.items():
        text = L.chapter(k)
        first = lst[0][0]
        assert "**%s**" % first in text, (k, first)


# ------------------------------------------------------------------ setups
def test_every_setup_is_valid_for_its_page():
    from ui import state as S
    sources = {p: open(os.path.join(ROOT, p), encoding="utf-8").read() for p in L.PAGES}
    for name, s in _all_setups():
        page = L.page_of(s)
        assert page in L.PAGES and os.path.exists(os.path.join(ROOT, page)), (name, page)
        for r in s.get("recipes", []) + s.get("batch", {}).get("methods", []):
            assert r in RECIPE_PRESETS, (name, r)
        for d in s.get("custom", []):
            S.validate_params(d["estimator"], d.get("params", {}))
            Recipe.from_json(d)
        for nz in [s.get("noise")] + s.get("batch", {}).get("noises", []):
            assert nz is None or nz in PRESETS, (name, nz)
        if "template" in s:
            T.build(s["template"][0], T.clamp_params(s["template"][0], dict(T.defaults(s["template"][0]),
                                                                            **s["template"][1])))
        if "circuit" in s:
            label, n, text = s["circuit"]
            parse(text, n, label)
        if "tab" in s:
            assert s["tab"] in _page_tabs("pages/1_Playground.py"), (name, s["tab"])
        for pk, tab in s.get("tabs", {}).items():
            assert pk in S.SETUP_PAGE_TABS, (name, pk)
            assert tab in _page_tabs(page), (name, tab)
        for wk in s.get("widgets", {}):                          # a real control of one of the analysis pages
            assert wk.startswith(S.SETUP_WIDGET_PREFIXES), (name, wk)
            assert any('"%s"' % wk in src for src in sources.values()), (name, wk)
        if "w_hh_rc" in s.get("widgets", {}):
            hh = open(os.path.join(ROOT, "pages/5_Head_to_Head.py"), encoding="utf-8").read()
            assert all(r in RECIPE_PRESETS for r in s["widgets"]["w_hh_rc"])
            assert s["widgets"]["w_hh_param"] in A.SWEEPS and '"w_hh_param"' in hh


def test_setup_labels_and_guide_markdown():
    assert L.setup_label(L.BUDGET_SETUP) == "Set up in Cost vs precision"
    assert L.setup_label(L.FOLDING_SETUP) == "Set up in the Playground"
    assert L.setup_label(next(g for g in L.GUIDES if g["key"] == "batch")["setup"]) == "Set up the Batch run"
    for g in L.GUIDES:
        if "**Set up" in g["steps"][0]:                     # the first step names the button the guide shows
            assert "**%s**" % L.setup_label(g["setup"]) in g["steps"][0], g["key"]
    md = L.guides_markdown()
    assert md.count("## ") == len(L.GUIDES)
    assert "Set up in" not in md and "Set up on" not in md and "Set up the" not in md
    assert [L.guide_track(g) for g in L.GUIDES].count("research") == 4


def test_apply_setup_adds_custom_contenders_tabs_and_whitelisted_widgets(monkeypatch):
    import streamlit as st
    from ui import state as S

    class SS(dict):
        __getattr__ = dict.get

        def __setattr__(self, k, v):
            self[k] = v
    fake = SS(custom_recipes=[], page_tab={}, time_limit=0)
    monkeypatch.setattr(st, "session_state", fake, raising=False)
    monkeypatch.setattr(S, "push_widgets", lambda: None)
    monkeypatch.setattr(S, "set_circuit", lambda spec: fake.update(n=spec.n, spec=spec))
    monkeypatch.setattr(S, "_load_template_model", lambda key, params: fake.update(n=T.build(key, params).spec.n))
    S.apply_setup(dict(L.FOLDING_SETUP))
    assert not fake.get("learn_error"), fake.get("learn_error")
    assert len(fake["custom_recipes"]) == 1 and fake["custom_recipes"][0]["params"]["lam_est"] == "fault-rate"
    S.apply_setup(dict(L.FOLDING_SETUP))                                  # the same contender is not added twice
    assert len(fake["custom_recipes"]) == 1
    assert fake["w_pg_tabs"] == "Mitigate & compare" and fake["goto"] == "pages/1_Playground.py"
    S.apply_setup(dict(L.BUDGET_SETUP, widgets=dict(L.BUDGET_SETUP["widgets"], secret_key="x", w_tshot=1.0)))
    assert fake["goto"] == "pages/2_Cost_vs_Precision.py" and fake["w_cvp_tabs"] == "Your circuit"
    assert fake["w_hh_param"] == "scale" and fake["w_hh_rc"] == L.BUDGET_RECIPES
    assert "secret_key" not in fake and "w_tshot" not in fake            # only whitelisted analysis controls
    bad = dict(L.FOLDING_SETUP, custom=[dict(estimator="zne", params=dict(folding="sideways"))])
    S.apply_setup(bad)
    assert fake["learn_error"].startswith("Could not set this up")


# ------------------------------------------------------------------ the numbers the chapters and guides quote
def _compiled(key, params, opts=None):
    tpl = T.build(key, T.clamp_params(key, dict(T.defaults(key), **params)))
    return tpl, CP.build(tpl.spec, relabel=tpl.relabel, opts=opts or CP.CompileOptions())


def _run(setup, noise=None):
    if "template" in setup:
        tpl, comp = _compiled(*setup["template"])
        obs = tpl.observable
    else:
        label, n, text = setup["circuit"]
        comp = CP.build(parse(text, n, label), relabel=False, opts=CP.CompileOptions())
        obs = OBS.z(setup["observable"][1], n)
    recs = [preset(r) for r in setup.get("recipes", [])] + [Recipe.from_json(d) for d in setup.get("custom", [])]
    ideal, outs, _ = compare(comp, PRESETS[noise or setup["noise"]], obs, recs, "numpy", 4096, None)
    return ideal, {o.recipe.title(): o for o in outs}


def _lam_eff(out):
    note = next(n for n in out.result.notes if "λ_eff" in n)
    return [float(x) for x in re.search(r"λ_eff = ([0-9., ]+);", note).group(1).split(",")]


def test_folding_numbers_in_chapter_and_guide():
    ideal, o = _run(L.FOLDING_SETUP)
    g, l, lf = (o["ZNE Richardson (global 1·3·5)"], o["ZNE Richardson (local 1·3·5)"],
                o["ZNE Richardson (local 1·3·5, λ from fault rate)"])
    assert g.result.mean - ideal == pytest.approx(-0.005, abs=6e-4)
    assert l.result.mean - ideal == pytest.approx(-0.012, abs=6e-4)
    assert lf.result.mean - ideal == pytest.approx(-0.003, abs=6e-4)
    assert _lam_eff(l) == pytest.approx([1.0, 2.85, 4.70], abs=6e-3)
    eta = np.array([15 / 8, -5 / 4, 3 / 8])
    assert float(eta @ _lam_eff(l)) == pytest.approx(0.07, abs=0.01)                     # ρ
    ideal, o = _run(L.FOLDING_SETUP, "Markovian dephasing")
    assert _lam_eff(o["ZNE Richardson (local 1·3·5)"]) == pytest.approx([1.0, 2.33, 3.67], abs=6e-3)
    assert o["ZNE Richardson (local 1·3·5)"].result.mean - ideal == pytest.approx(-0.033, abs=1e-3)
    assert o["ZNE Richardson (global 1·3·5)"].result.mean - ideal == pytest.approx(-0.001, abs=1e-3)
    ideal, o = _run(L.FOLDING_SETUP, "Coherent over-rotation")
    assert o["ZNE Richardson (global 1·3·5)"].result.mean == pytest.approx(1.15, abs=0.005)
    assert o["ZNE Richardson (local 1·3·5)"].result.mean == pytest.approx(o["Unmitigated"].result.mean)
    assert o["ZNE Richardson (local 1·3·5, λ from fault rate)"].result is None             # refuses, with a reason
    _, o = _run(L.TOPIC_SETUPS["folding"][2][1])                                         # twirl the right gates
    assert o["ZNE Richardson (global 1·3·5) + RC"].result.mean == pytest.approx(1.0, abs=0.002)
    assert o["ZNE Richardson (global 1·3·5) + twirl + TREX"].result.mean == pytest.approx(1.15, abs=0.005)


def test_fractional_zne_numbers():
    ideal, o = _run(L.TOPIC_SETUPS["folding"][1][1])
    fr = o["ZNE quadratic (local stratified 1·1.333·1.667·2)"].result
    gl = o["ZNE Richardson (global 1·3·5)"].result
    assert fr.mean - ideal == pytest.approx(-0.052, abs=1e-3) and gl.mean - ideal == pytest.approx(-0.087, abs=1e-3)
    assert np.sqrt(fr.var1) == pytest.approx(9.2, abs=0.1) and np.sqrt(gl.var1) == pytest.approx(1.5, abs=0.02)
    assert fr.var1 / gl.var1 == pytest.approx(37, rel=0.06)


def test_spectator_probe_numbers():
    zero = dict(L.TOPIC_SETUPS["context"][0][1])
    plus = dict(L.CONTEXT_SETUP)
    expect = {"Depolarizing": (0.996, 0.988, 1.000), "Crosstalk (spectator)": (0.842, 0.997, 1.473),
              "Markovian dephasing": (0.881, 0.863, 0.996), "Random field (quasi-static)": (0.491, 1.000, 0.881)}
    for nz, (raw, dd, zne) in expect.items():
        _, o = _run(zero, nz)
        assert all(x.result.mean == pytest.approx(1.0, abs=1e-6) for x in o.values()), nz       # |0⟩ is blind
        _, o = _run(plus, nz)
        got = [o[t].result.mean for t in ("Unmitigated", "DD-XY4", "ZNE Richardson (global 1·3·5)")]
        assert got == pytest.approx([raw, dd, zne], abs=1.5e-3), (nz, got)


def test_budget_numbers():
    tpl, comp = _compiled("qft", {"n": 4})
    recs = [preset(r) for r in L.BUDGET_RECIPES]
    ideal, outs, _ = compare(comp, PRESETS["Hardware-realistic (illustrative)"], tpl.observable, recs, "numpy", 4096,
                             None)
    _, rows = table_rows(outs, ideal, 4096, 250.0)
    res = dict(rows=rows, shots=4096)
    t1 = RP.cost_precision(res, 0.01, 600.0, 100, 10 ** 8, 250.0)[1]["cost_for_target"].set_index("method")
    t5 = RP.cost_precision(res, 0.05, 600.0, 100, 10 ** 8, 250.0)[1]["cost_for_target"].set_index("method")
    assert np.isinf(t1["budget for ε"]).all()                                           # nothing reaches 0.01
    assert t1.loc["Unmitigated", "bias"] == pytest.approx(-0.18, abs=0.001)
    cross = t1["beats unmitigated above B"]
    assert cross["PEC inverse"] == pytest.approx(9, abs=1)
    assert cross["ZNE exponential (global 1·3·5)"] == pytest.approx(37, abs=1.5)
    assert cross["ZNE Richardson (global 1·3·5)"] == pytest.approx(85, abs=2)
    assert cross["CDR (OLS)"] == pytest.approx(16400, rel=0.01)
    reach = t5["budget for ε"]
    assert np.isfinite(reach["CDR (OLS)"]) and reach.drop("CDR (OLS)").map(np.isinf).all()
    assert reach["CDR (OLS)"] == pytest.approx(1.7e4, rel=0.03)
    assert t5.loc["CDR (OLS)", "QPU time for ε (s)"] == pytest.approx(4.1, abs=0.1)
    rows = A.sweep(comp, PRESETS["Hardware-realistic (illustrative)"], tpl.observable, recs, "scale",
                   np.linspace(0.5, 3, 6), "numpy", 4096, None, lambda f: None)
    shots = np.array([1e3, 3e3, 1e5, 1e7])
    _, methods, win, _, _ = A.winner_map(rows, shots)
    names = [[methods[w] for w in row] for row in win]
    assert all(r[0] == r[1] == "ZNE exponential (global 1·3·5)" for r in names)          # the middle band
    assert all(r[2] == r[3] == "CDR (OLS)" for r in names)                               # the floor-limited band
    rows = A.sweep(comp, PRESETS["Depolarizing"], tpl.observable, recs, "scale", np.linspace(0.5, 3, 6), "numpy",
                   4096, None, lambda f: None)
    _, methods, win, _, _ = A.winner_map(rows, np.logspace(2, 8, 13))
    assert np.mean([methods[w] == "PEC inverse" for w in win.ravel()]) > 0.95            # PEC almost everywhere


def test_unmodeled_noise_numbers():
    import dataclasses
    tpl, comp = _compiled("qft", {"n": 4})
    nz = PRESETS["Depolarizing"]
    base = dataclasses.replace(type(nz)(p1=nz.p1, p2=nz.p2 or 0.01))
    assert (base.p1, base.p2) == (0.002, 0.02)
    rec = [Recipe.make("raw"), Recipe.make("pec", construction="inverse"), preset("PEC · learned model + RC")]
    rows = A.sweep(comp, base, tpl.observable, rec, "over_rot", [0.02, 0.04, 0.06, 0.08], "numpy", 4096, None,
                   lambda f: None)
    pec = [r["bias"] for r in rows if r["method"] == "PEC inverse"]
    lrn = [r for r in rows if r["method"] == "PEC inverse (learned) + RC"]
    assert pec == pytest.approx([-0.008, -0.031, -0.069, -0.119], abs=6e-4)
    assert all(abs(r["bias"]) < 1e-9 for r in lrn)
    assert [round(r["gamma"], 2) for r in lrn][::3] == [1.41, 1.78]


# ------------------------------------------------------------------ the page
@pytest.mark.slow
def test_learn_research_topics_tab_and_setups():
    from tests.test_round4 import _app, _ok, _tab
    at = _app()
    at.switch_page("pages/9_Learn.py").run()
    _ok(at)
    _tab(at, "Research topics", "learn", "w_learn_tabs")
    for k in L.TOPICS:
        at.session_state["w_learn_topic"] = k
        at.run()
        _ok(at)
        assert any(b.key == "topic_%s_0" % k for b in at.button), k
    for k, lst in L.TOPIC_SETUPS.items():
        for j, (label, s) in enumerate(lst):
            at.switch_page("pages/9_Learn.py").run()
            at.session_state["w_learn_topic"] = k
            _tab(at, "Research topics", "learn", "w_learn_tabs")
            next(b for b in at.button if b.key == "topic_%s_%d" % (k, j)).click().run()
            _ok(at)
            assert not at.session_state["learn_error"], (label, at.session_state["learn_error"])
            at.switch_page(L.page_of(s)).run()
            _ok(at)
