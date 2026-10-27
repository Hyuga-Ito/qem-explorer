"""The chained estimator (qemx/methods/compose.py): an outer method over an inner mitigation of every circuit."""
from __future__ import annotations

import math

import numpy as np
import pytest

from qemx import compile as CP
from qemx import templates as T
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.noise import PRESETS


def _ctx(recipe, noise=None, tpl="qft", seed=3):
    t = T.build(tpl)
    comp = CP.build(t.spec, relabel=t.relabel)
    return Context(comp, noise or PRESETS["Depolarizing"], t.observable, recipe, Runner("numpy"), shots=4096,
                   seed=seed)


def _run(rc, **kw):
    ctx = _ctx(rc, **kw)
    return ctx, REGISTRY[rc.estimator].run(ctx, **rc.p)


@pytest.mark.parametrize("readout", ["none", "tensored"])
@pytest.mark.parametrize("rule", ["richardson", "linear", "exponential"])
def test_chain_zne_raw_equals_zne(readout, rule):
    z = Recipe.make("zne", "none", "none", readout, scales=[1.0, 2.0, 3.0], rule=rule)
    c = Recipe.make("chain", "none", "none", readout, outer="zne", zne_scales=[1.0, 2.0, 3.0],
                    zne_rule=rule)
    _, rz = _run(z, noise=PRESETS["Hardware-realistic (illustrative)"])
    _, rc = _run(c, noise=PRESETS["Hardware-realistic (illustrative)"])
    assert rc.mean == pytest.approx(rz.mean, abs=1e-12) and rc.var1 == pytest.approx(rz.var1, rel=1e-12)
    assert rc.circuits == rz.circuits and rc.cal_circuits == rz.cal_circuits
    if readout != "none":
        assert np.allclose(rc.cal_grad, rz.cal_grad, rtol=1e-6, atol=1e-12)
    a = rz.sampler(3000, 50, np.random.default_rng(5))
    b = rc.sampler(3000, 50, np.random.default_rng(5))
    assert np.allclose(a, b, equal_nan=True)                   # the same draws: one inner sampler per folded circuit


@pytest.mark.parametrize("params", [dict(model="l1", n_neighbors=4, n_train=6),
                                    dict(model="ridge", alpha=1e-3, n_neighbors=3, n_train=6, target_split="equal"),
                                    dict(model="lasso", alpha=1e-4, neighbors="fold", n_train=6)])
def test_chain_nil_raw_equals_nil(params):
    p = dict(params)
    n_train = p.pop("n_train")
    nil = Recipe.make("nil", "none", "none", "tensored", n_train=n_train, train_shots=512, **p)
    ch = Recipe.make("chain", "none", "none", "tensored", outer="nil", inner="raw", n_train=n_train, train_shots=512,
                     **{"nil_" + k: v for k, v in p.items()})
    _, rn = _run(nil)
    _, rc = _run(ch)
    assert rc.mean == pytest.approx(rn.mean, abs=1e-12) and rc.var1 == pytest.approx(rn.var1, rel=1e-10)
    assert rc.fixed_shots == rn.fixed_shots and rc.train_circuits == rn.train_circuits
    assert rc.circuits == rn.circuits
    assert np.allclose(rc.details["coef"], rn.details["coef"], atol=1e-12)
    assert rc.var_fixed == pytest.approx(rn.var_fixed, rel=1e-9)     # the same Monte-Carlo draws and refits
    e_n, v_n = rn.train_resampler(256, 40, np.random.default_rng(9))
    e_c, v_c = rc.train_resampler(256, 40, np.random.default_rng(9))
    assert np.allclose(e_n, e_c, atol=1e-12) and np.allclose(v_n, v_c, rtol=1e-10)


def test_chain_cdr_is_a_line_through_inner_values():
    rc = Recipe.make("chain", outer="cdr", inner="raw", n_train=8, train_shots=1024)       # near-Clifford (default)
    ctx, r = _run(rc)
    X, y = r.details["X"][:, 0], r.details["y_train"]
    a, b = np.polyfit(X, y, 1)
    assert r.reference_mean == pytest.approx(a * r.details["xt"][0] + b, abs=1e-8)     # the exact-training fit
    assert r.estimate_kind == "protocol" and r.mean == pytest.approx(r.moments["mean"])
    assert r.fixed_shots == 8 * 1024 and r.train_circuits == 8 and r.circuits == 1


def test_chain_zne_symmetry_is_richardson_over_postselected_values():
    from qemx.methods.zne import richardson_weights
    from qemx.transforms import achieved_lambda, fold_global
    from qemx.circuit import parse
    from qemx.observables import z
    rc = Recipe.make("chain", outer="zne", inner="symmetry", zne_scales=[1.0, 3.0], symmetry_qubits=[0, 1],
                     symmetry_eigenvalue=1)
    spec = parse("h 0; cx 0 1; ry(0.7) 1; cx 0 1; rx(0.4) 0", 2)    # the parity Z0Z1 is not conserved: P_accept < 1
    ctx = Context(CP.build(spec), PRESETS["Depolarizing"], z(0, 2), rc, Runner("numpy"), shots=4096, seed=3)
    r = REGISTRY["chain"].run(ctx, **rc.p)
    vals, lams = [], []
    for lam in (1.0, 3.0):
        fs = fold_global(ctx.spec, lam)
        sub = Recipe.make("symmetry", qubits=[0, 1], eigenvalue=1)
        import dataclasses
        c2 = Context(dataclasses.replace(ctx.compiled, implemented=fs), ctx.noise, ctx.obs, sub, ctx.runner,
                     seed=ctx.seed)
        vals.append(REGISTRY["symmetry"].run(c2, **sub.p).mean)
        lams.append(achieved_lambda(ctx.spec, fs))
    assert r.mean == pytest.approx(float(richardson_weights(lams) @ vals), abs=1e-12)
    assert 0 < r.accept < 1 and r.circuits == 2 and r.var1 > 0
    # the finite-shot sampler agrees with the first-order variance where post-selection keeps many shots
    x = r.sampler(20000, 400, np.random.default_rng(1))
    assert np.isfinite(x).all()
    assert np.var(x, ddof=1) == pytest.approx(r.var1 / 20000, rel=0.25)
    assert abs(np.mean(x) - r.mean) < 5 * math.sqrt(r.var1 / 20000 / 400)


def test_chain_inner_vd_and_pec_costs():
    _, r = _run(Recipe.make("chain", outer="zne", inner="vd", zne_scales=[1.0, 3.0], vd_M=2), tpl="bell")
    assert r.extra_qubits == 2 and r.cal_circuits == 0
    ctx, rp = _run(Recipe.make("chain", "none", "none", "none", outer="zne", inner="pec", zne_scales=[1.0, 3.0]),
                   tpl="bell")
    _, rz = _run(Recipe.make("zne", scales=[1.0, 3.0]), tpl="bell")
    assert rp.var1 > rz.var1                                   # every folded circuit pays PEC's γ²
    # depolarizing gate noise with PEC told the true rates: every folded circuit is mitigated exactly, so E(λ) is
    # flat at the ideal value and so is the extrapolation
    assert np.allclose(rp.details["E"], ctx.ideal_value(), atol=1e-12) and rp.mean == pytest.approx(
        ctx.ideal_value(), abs=1e-12)
    assert rp.gamma > rz.gamma


def test_chain_is_simulation_only_and_titles():
    m = REGISTRY["chain"]
    assert m.hardware_supported({})[0] is False
    from qemx.hardware import supported
    assert supported(Recipe.make("chain"))[0] is False
    t = Recipe.make("chain", outer="nil", inner="pec", nil_model="ridge").title()
    assert "∘" in t and "NIL" in t and "PEC" in t
    from qemx.recipes import family
    assert family(Recipe.make("chain", outer="cdr")) == "learning" and family(Recipe.make("chain")) == "zne"
    assert Recipe.make("chain").p["inner"] == "raw" and Recipe.make("chain").p["cdr_ensemble"] == "near"


def test_chain_params_validate_and_hide():
    import sys
    sys.path.insert(0, ".")
    from ui.state import validate_params
    ok = validate_params("chain", dict(outer="zne", inner="symmetry", symmetry_more="2 3; 0 1"))
    assert ok["symmetry_more"] == "2 3; 0 1"
    with pytest.raises(ValueError):
        validate_params("chain", dict(outer="zne", inner="symmetry", symmetry_more="__import__('os')"))
    with pytest.raises(ValueError):
        validate_params("chain", dict(outer="zne", inner="raw", zne_rule="cubic"))


def test_chain_in_a_pipeline_cell():
    from qemx.phase.cell import CellSpec, MethodSpec, cell_runner, cost_of
    from qemx.phase.ledger import PIPELINE_FIXED, allocate
    methods = (MethodSpec.make("RAW", "raw"),
               MethodSpec.make("ZNE∘SYM", "chain", outer="zne", inner="symmetry", zne_scales=[1.0, 3.0],
                               symmetry_qubits=[0, 1]),
               MethodSpec.make("NIL∘RAW", "chain", outer="nil", inner="raw", n_train=6, nil_n_neighbors=3,
                               train_shots=512))
    spec = CellSpec(stage="t", family="qft", n=4, fparams=(("m", None), ("s", 1)),
                    layout=(136, 143, 142, 141), methods=methods, seed=5, train_levels=(256,), train_reps=60,
                    train_reps_other=40, validate_budgets=(10 ** 6,), validate_reps=200, validate_policy=("fixed",))
    rec = cell_runner(spec)
    assert rec["status"] == "ok", [m.get("error") for m in rec["methods"]]
    rows = {r["key"]: r for r in rec["methods"]}
    z = rows["ZNE∘SYM"]
    assert z["m"] == 2 and len(z["a"]) == 2 and sum(z["a"]) ** 2 == pytest.approx(z["v"], rel=1e-9)
    n = rows["NIL∘RAW"]
    assert n["T"] > 0 and {l["s_tr"] for l in n["levels"]} == {256, 512} and all(l["ok"] for l in n["levels"])
    for r in rows.values():
        c = cost_of(r)
        al = allocate(c, 10 ** 6, PIPELINE_FIXED)
        assert al.feasible and al.N + c.T * al.s_tr + c.c * al.s == 10 ** 6
    vals = [v for v in rec["validation"] if "mc" in v]
    assert vals and all(v["identity"] for v in vals)
    for v in vals:
        assert abs(v["z_rmse"]) < 5 and abs(v["z_bias"]) < 5, v


@pytest.mark.slow
def test_playground_runs_and_shows_chains():
    import sys
    sys.path.insert(0, ".")
    from test_audit_app import _ok, _page_with_state
    ops = [{"name": "h", "qubits": [0], "params": []}, {"name": "cx", "qubits": [0, 1], "params": []},
           {"name": "ry", "qubits": [1], "params": [0.7]}, {"name": "cx", "qubits": [0, 1], "params": []}]
    state = {"circuit": {"n": 2, "name": "c2", "ops": ops}, "observable": {"pauli": "Z0"},
             "recipes": [{"estimator": "chain", "params": {"outer": "zne", "inner": "symmetry", "zne_scales": [1.0, 3.0],
                                                           "symmetry_qubits": [0, 1]}},
                         {"estimator": "chain", "params": {"outer": "nil", "inner": "raw", "n_train": 4,
                                                           "nil_n_neighbors": 2, "train_shots": 256}}]}
    at = _page_with_state("pages/1_Playground.py", state, preset={"w_pg_tabs": "Mitigate & compare",
                                                                  "run_requested": True})
    at.run()
    _ok(at)
    res = at.session_state["results"]
    assert len(res["outs"]) == 2 and all(o.result is not None for o in res["outs"]), [o.error for o in res["outs"]]
    assert {o.result.details["outer"] for o in res["outs"]} == {"zne", "nil"}


@pytest.mark.slow
def test_recipe_builder_shows_only_the_chosen_components():
    import sys
    sys.path.insert(0, ".")
    from test_audit_app import _ok, _page_with_state
    at = _page_with_state("pages/1_Playground.py", preset={"w_pg_tabs": "Mitigate & compare", "w_rb_est": "chain",
                                                           "w_rb_chain_outer": "nil", "w_rb_chain_inner": "pec"})
    at.run()
    _ok(at)
    keys = {w.key for w in list(at.selectbox) + list(at.number_input) + list(at.text_input) + list(at.toggle)}
    assert {"w_rb_chain_outer", "w_rb_chain_inner", "w_rb_chain_nil_model", "w_rb_chain_pec_construction",
            "w_rb_chain_n_train"} <= keys
    assert not keys & {"w_rb_chain_zne_rule", "w_rb_chain_symmetry_more", "w_rb_chain_vd_M", "w_rb_chain_cdr_ensemble"}
    next(b for b in at.button if b.key == "rb_add").click()
    at.run()
    _ok(at)
    added = at.session_state["custom_recipes"][-1]
    assert added["estimator"] == "chain" and added["params"]["outer"] == "nil" and added["params"]["inner"] == "pec"
