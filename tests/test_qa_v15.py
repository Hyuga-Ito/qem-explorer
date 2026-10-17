"""1.5.0: regression tests for the defects of the 1.4.2 QA report (QEM_Explorer_1.4.2_QA_Report.md).

Each test names the report section it covers. The numbers asserted here are closed-form (readout inverse norms,
quasi-probability sums) or properties of the statistical result contract (qemx.stats), not values copied from a
previous run."""
import io
import json
import math
import os
import zipfile

import numpy as np
import pytest

from qemx import compile as CP
from qemx import templates as T
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.methods import base as MB
from qemx.methods import nil as N
from qemx.methods import pec as PEC
from qemx.noise import NoiseSpec
from qemx.recipes import (INSUFFICIENT_BUDGET, PRESETS as RECIPE_PRESETS, budget_needed_text, compare, preflight,
                          preset, summary, table_rows)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _qft(n=3):
    tpl = T.build("qft", {"n": n})
    return tpl, CP.build(tpl.spec, relabel=tpl.relabel)


def _ctx(noise, readout="tensored", n=2):
    tpl, comp = _qft(n)
    rc = Recipe.make("raw", readout=readout)
    return Context(comp, noise, tpl.observable, rc, Runner("numpy"), shots=4096)


# ------------------------------------------------------------------ §4.3 readout inverse norm
@pytest.mark.parametrize("e,h,expect", [(0.0, 0.0, 1.0), (0.1, 0.1, 1.25), (0.02, 0.05, 1.03 / 0.93),
                                        (0.6, 0.6, 5.0), (0.9, 0.0, 1.9 / 0.1), (1.0, 1.0, 1.0)])
def test_assignment_inverse_norm_closed_form(e, h, expect):
    g = MB.assignment_inverse_gamma(e, h)
    assert g == pytest.approx(expect, rel=1e-12)
    A = np.array([[1 - e, h], [e, 1 - h]])
    assert g == pytest.approx(np.abs(np.linalg.inv(A)).sum(0).max(), rel=1e-10)     # ‖A⁻¹‖₁ (max column sum)


def test_assignment_inverse_norm_singular_and_invalid():
    assert MB.assignment_inverse_gamma(0.5, 0.5) == math.inf
    assert MB.assignment_inverse_gamma(0.3, 0.7) == math.inf
    for e, h in [(-0.1, 0.0), (0.0, 1.2), (math.nan, 0.0), (0.1, math.inf)]:
        with pytest.raises(ValueError):
            MB.assignment_inverse_gamma(e, h)


def test_two_qubit_readout_gamma_is_25_for_mostly_flipping_readout():
    """QA §4.3: e = h = 0.6 on two qubits. Each assignment matrix has determinant 1 − e − h = −0.2 (invertible: it
    mostly flips), ‖A⁻¹‖₁ = 1/0.2 = 5, and the product over the two bits is 25 (1.4.2 displayed 1e24, QA report §4.3)."""
    ctx = _ctx(NoiseSpec(ro01=0.6, ro10=0.6), "tensored", n=2)
    A = ctx.calibration()["A"]
    assert all(np.linalg.det(a) == pytest.approx(-0.2, abs=1e-12) for a in A)
    assert ctx.readout_gamma() == pytest.approx(25.0, rel=1e-12)
    assert ctx.theta_invertible(ctx.cal_theta())
    assert ctx.theta_condition() > 1.0


@pytest.mark.parametrize("mode", ["tensored", "m3", "trex"])
def test_singular_readout_is_a_clear_error(mode):
    ctx = _ctx(NoiseSpec(ro01=0.5, ro10=0.5), mode, n=2)
    with pytest.raises(ValueError, match="singular"):
        ctx.calibration()
    tpl, comp = _qft(2)
    ideal, outs, _ = compare(comp, NoiseSpec(ro01=0.5, ro10=0.5), tpl.observable, [Recipe.make("raw", readout=mode)])
    assert outs[0].result is None and "singular" in outs[0].error


def test_theta_invertible_accepts_negative_determinant_and_rejects_singular():
    ctx = _ctx(NoiseSpec(ro01=0.6, ro10=0.6), "tensored", n=2)
    assert ctx.theta_invertible([0.6, 0.6, 0.6, 0.6])
    assert not ctx.theta_invertible([0.5, 0.5, 0.1, 0.1])
    assert not ctx.theta_invertible([math.nan, 0.1, 0.1, 0.1])


# ------------------------------------------------------------------ §4.4 PEC inputs
def test_depol_quasi_domain():
    for bad in (1.0, 1.5, -0.01, math.nan, math.inf, "x"):
        with pytest.raises(ValueError):
            PEC.depol_quasi(bad, 1, "inverse")
    with pytest.raises(ValueError):
        PEC.depol_quasi(0.1, 0, "inverse")
    with pytest.raises(ValueError):
        PEC.depol_quasi(0.1, 1, "sideways")
    assert PEC.depol_quasi(0.0, 2, "inverse") == [(1.0, "II")]
    for k in (1, 2):
        for c in ("inverse", "compensation"):
            q = PEC.depol_quasi(0.05, k, c)
            assert sum(w for w, _ in q) == pytest.approx(1.0, abs=1e-12), (k, c)     # trace preserving
    d2, f = 16, 0.95
    gamma = sum(abs(w) for w, _ in PEC.depol_quasi(0.05, 2, "inverse"))
    assert gamma == pytest.approx((2 * (d2 - 1) / f - d2 + 2) / d2, rel=1e-12)
    assert sum(abs(w) for w, _ in PEC.depol_quasi(0.05, 2, "compensation")) == pytest.approx(1.05 / 0.95, rel=1e-12)


def test_learned_pec_refuses_a_non_invertible_channel():
    from qemx.circuit import parse
    spec = parse("cx 0 1", 2, "cx")
    op = next(o for o in spec.ops if o.k == 2)
    with pytest.raises(ValueError, match="not invertible"):
        PEC.learned_quasi(NoiseSpec(p2=1.0), op)
    q = PEC.learned_quasi(NoiseSpec(p2=0.02), op)
    assert sum(w for w, _ in q) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------ §4.4 NIL solver inputs
def test_nil_fit_validates_its_inputs():
    X, y = np.eye(3), np.ones(3)
    bad = [dict(X=np.ones(3), y=y), dict(X=X, y=np.ones(2)), dict(X=X * np.nan, y=y), dict(X=X, y=y, alpha=-1.0),
           dict(X=X, y=y, alpha=math.inf), dict(X=X, y=y, l1_ratio=1.5), dict(X=X, y=y, model="l1", l1_bound=-1.0),
           dict(X=X, y=y, model="l1", l1_bound=math.nan), dict(X=np.zeros((0, 3)), y=np.zeros(0))]
    for kw in bad:
        kw.setdefault("model", "ridge")
        with pytest.raises(ValueError):
            N.nil_fit(kw.pop("X"), kw.pop("y"), kw.pop("model"), **kw)
    c, b, info = N.nil_fit(X, y, "l1", l1_bound=0.0)                   # γ = 0: legitimate, c = 0
    assert np.all(c == 0) and b == 0.0


# ------------------------------------------------------------------ §4.1 FR id migration
def test_feature_regression_legacy_id_is_migrated():
    p, notes = MB.migrate_params("learning", dict(features="nil", model="ridge"))
    assert p["features"] == "fr" and len(notes) == 1 and "legacy" in notes[0]
    assert MB.migrate_params("learning", dict(features="fr"))[1] == []
    assert MB.migrate_params("nil", dict(features="nil"))[1] == []        # only the learning estimator's value
    assert Recipe.make("learning", features="nil").p["features"] == "fr"
    assert Recipe.make("learning", features="nil").key() == Recipe.make("learning", features="fr").key()
    for name, d in RECIPE_PRESETS.items():
        assert d.get("features") != "nil", name


def test_manifest_with_legacy_feature_id_validates_and_reports_migration():
    import yaml
    from qemx.phase import manifest as M
    with open(os.path.join(ROOT, "manifests", "smoke.yaml"), encoding="utf-8") as f:
        m = yaml.safe_load(f)
    legacy = sorted(k for k, d in m["methods"].items() if d.get("estimator") == "learning" and d.get("features") == "nil")
    assert legacy                                                       # the 1.3 manifests name FR "nil"
    M.check(m)                                                          # still valid …
    notes = M.migrations(m)
    assert sorted(n.split(" — ")[0].replace("method ", "") for n in notes) == legacy
    assert all("'fr'" in n for n in notes)
    assert all(m["methods"][k]["features"] == "nil" for k in legacy)    # … and never rewritten (cell ids unchanged)
    for bad in (dict(m, stages={"pilot": {}}), dict(m, methods=[1, 2]), dict(m, method_sets=["core"])):
        with pytest.raises(ValueError):                                 # a clear message, not an AttributeError
            M.check(bad)


# ------------------------------------------------------------------ §4.5 budget preflight
def test_insufficient_budget_is_detected_before_any_fit():
    tpl, comp = _qft(3)
    nil = preset("NIL · ℓ1 ≤ 2 (Chen et al.)")
    small = preset("NIL · small budget")
    ideal, outs, _ = compare(comp, NoiseSpec(p1=0.002, p2=0.02), tpl.observable, [nil, small, preset("Unmitigated")],
                             shots=4096, check_budget=True)
    o_nil, o_small, o_raw = outs
    assert o_nil.status == INSUFFICIENT_BUDGET and o_nil.result is None
    assert o_nil.preflight["min_budget"] > 4096 and not o_nil.preflight["feasible"]
    assert "needs at least" in o_nil.error and budget_needed_text(o_nil.preflight["min_budget"]) in o_nil.error
    assert o_small.result is not None, o_small.error                  # the budget-scaled preset fits 4 096 shots
    assert o_small.preflight is None or o_small.preflight["feasible"]
    assert o_raw.result is not None
    samples, rows = table_rows(outs, ideal, 4096, 250.0, repeats=20)
    assert len(rows) == 3                                               # the unaffordable row is shown, not dropped
    bad = [r for r in rows if r.get("status") == INSUFFICIENT_BUDGET or "needs at least" in str(r)]
    assert len(bad) == 1 and o_nil.recipe.title() not in samples
    # without check_budget (studies) every recipe runs, so cost curves stay comparable across budgets
    _, outs2, _ = compare(comp, NoiseSpec(p1=0.002, p2=0.02), tpl.observable, [small], shots=4096)
    assert outs2[0].result is not None


def test_preflight_minimum_is_inclusive():
    tpl, comp = _qft(3)
    rc = preset("NIL · small budget")
    noise = NoiseSpec(p2=0.02)

    def make_ctx(c, o, B):
        return Context(c, noise, o, rc, Runner("numpy"), shots=B)
    pf = preflight(rc, lambda c, o: make_ctx(c, o, 4096), comp, tpl.observable, 4096)
    mb = int(pf["min_budget"])
    assert pf["training"] == 6 * 4 * 128
    assert preflight(rc, lambda c, o: make_ctx(c, o, mb), comp, tpl.observable, mb)["feasible"]
    assert not preflight(rc, lambda c, o: make_ctx(c, o, mb - 1), comp, tpl.observable, mb - 1)["feasible"]


# ------------------------------------------------------------------ §4.2 statistical result contract
def test_learning_results_report_protocol_moments_and_reference():
    tpl, comp = _qft(3)
    rc = Recipe.make("learning", features="cdr", model="ols", n_train=8, train_shots=64)
    ideal, outs, _ = compare(comp, NoiseSpec(p1=0.003, p2=0.03), tpl.observable, [rc], shots=100_000, seed=13)
    r = outs[0].result
    assert r.estimate_kind == "protocol"
    assert r.reference_mean is not None and r.reference_var1 is not None
    mom = r.moments
    assert mom["repeats"] >= 500 and mom["finite"] >= 1 and mom["train_shots"] == 64
    assert mom["mean"] == pytest.approx(r.mean)
    # finite training data move the mean away from the exact-training fit by more than its MC error here
    assert abs(r.mean - r.reference_mean) > 0 or "protocol" in r.estimate_kind
    row = summary(outs[0], ideal, 100_000, 250.0)
    assert row["estimate"] == pytest.approx(r.mean)


# ------------------------------------------------------------------ §4.6 bounded export
@pytest.fixture
def _figs():
    import plotly.graph_objects as go
    return {"a": go.Figure(go.Scatter(x=[0, 1], y=[0, 1])), "b/c": go.Figure(go.Bar(x=[1], y=[2]))}


@pytest.mark.parametrize("fault", ["chrome", "crash", "hang"])
def test_export_never_waits_on_a_failing_renderer(fault, _figs, monkeypatch):
    from qemx import bundle as BD
    import time
    import pandas as pd
    monkeypatch.setenv("QEMX_RENDER_FAULT", fault)
    monkeypatch.setenv("QEMX_RENDER_TIMEOUT", "4")
    t0 = time.monotonic()
    data = BD.make_zip("t", tables={"rows": pd.DataFrame(dict(x=[1, 2]))}, figures=_figs, state={"k": 1})
    assert time.monotonic() - t0 < 15
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {"tables/rows.csv", "state/qem_explorer_state.json", "figures/a.json", "figures/b/c.json",
            "README.txt"} <= names
    readme = z.read("README.txt").decode()
    assert "not rendered" in readme and "complete regardless" in readme
    json.loads(z.read("figures/a.json"))


def test_export_data_only_skips_rendering(_figs, monkeypatch):
    from qemx import bundle as BD
    monkeypatch.setenv("QEMX_RENDER_FAULT", "hang")                    # would hang if rendering were attempted
    data = BD.make_zip("t", figures=_figs, images=False)
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "figures/a.json" in names and not any(n.endswith(".png") for n in names)


def test_render_worker_writes_atomically(tmp_path):
    from qemx import render_worker as RW
    RW._write(str(tmp_path / "x.png"), b"abc")
    assert (tmp_path / "x.png").read_bytes() == b"abc" and not (tmp_path / "x.png.part").exists()


# ------------------------------------------------------------------ §4.7 guide setups: atomic, replace and merge
class _SS(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


@pytest.fixture
def fake_state(monkeypatch):
    import streamlit as st
    from ui import state as S
    mine = Recipe.make("zne", rule="linear").to_json()
    fake = _SS(custom_recipes=[mine], custom_active=[Recipe.from_json(mine).key()], recipe_names=["Unmitigated",
               "PEC · inverse"], page_tab={}, time_limit=0, n=3)
    monkeypatch.setattr(st, "session_state", fake, raising=False)
    monkeypatch.setattr(S, "push_widgets", lambda: None)
    monkeypatch.setattr(S, "set_circuit", lambda spec: fake.update(n=spec.n, spec=spec))
    monkeypatch.setattr(S, "_load_template_model", lambda key, params: fake.update(n=T.build(key, params).spec.n))
    return fake, mine


def test_setup_replace_makes_the_guide_contenders_the_only_active_ones(fake_state):
    from qemx import learn as L
    from ui import state as S
    fake, mine = fake_state
    S.apply_setup(dict(L.FOLDING_SETUP), "replace")
    assert not fake.learn_error, fake.learn_error
    assert fake.recipe_names == list(L.FOLDING_SETUP.get("recipes", []))
    assert Recipe.from_json(mine).key() not in fake.custom_active      # deactivated …
    assert mine in fake.custom_recipes                                  # … but never deleted
    assert len(S.recipes()) == len(fake.recipe_names) + len(L.FOLDING_SETUP.get("custom", []))


def test_setup_merge_adds_the_guide_contenders(fake_state):
    from qemx import learn as L
    from ui import state as S
    fake, mine = fake_state
    S.apply_setup(dict(L.FOLDING_SETUP), "merge")
    assert not fake.learn_error, fake.learn_error
    assert {"Unmitigated", "PEC · inverse"} <= set(fake.recipe_names)
    assert Recipe.from_json(mine).key() in fake.custom_active
    assert len(fake.custom_active) == 1 + len(L.FOLDING_SETUP.get("custom", []))


def test_setup_is_atomic(fake_state, monkeypatch):
    from qemx import learn as L
    from ui import state as S
    fake, mine = fake_state
    def snap():
        return json.dumps({k: v for k, v in fake.items() if k != "learn_error"}, default=str, sort_keys=True)
    before = snap()
    S.apply_setup(dict(L.FOLDING_SETUP, recipes=["Unmitigated", "No such preset"]))   # invalid: rejected up front
    assert fake.learn_error.startswith("Could not set this up")
    fake.learn_error = ""
    assert snap() == before
    monkeypatch.setattr(S, "push_widgets", lambda: (_ for _ in ()).throw(RuntimeError("boom")))   # fails mid-way
    S.apply_setup(dict(L.FOLDING_SETUP))
    assert "nothing was changed" in fake.learn_error and "boom" in fake.learn_error
    fake.learn_error = ""
    assert snap() == before
    S.apply_setup(dict(L.FOLDING_SETUP), "sideways")
    assert "unknown mode" in fake.learn_error


def test_older_state_without_custom_active_keeps_every_saved_recipe_active(fake_state):
    from ui import state as S
    fake, mine = fake_state
    del fake["custom_active"]
    assert S.custom_active_keys() == [Recipe.from_json(mine).key()]


# ------------------------------------------------------------------ record schema
def test_record_schema_version_and_mixed_schema_guard():
    from qemx.phase import store as ST
    assert ST.SCHEMA_VERSION == "qemx.phase.cell/3"
    assert ST.READABLE_SCHEMAS[-1] == ST.SCHEMA_VERSION
    src = open(os.path.join(ROOT, "qemx", "phase", "run.py"), encoding="utf-8").read()
    assert "older record schema" in src
