"""Device models, drag-and-drop edits, hardware record/replay, analysis, bundle and security helpers."""
import io
import json
import math
import zipfile

import numpy as np
import pytest

from qemx import analysis as A
from qemx import compile as CP
from qemx import device as D
from qemx import edits as E
from qemx import hardware as H
from qemx import resources as RS
from qemx import templates as T
from qemx.bundle import make_zip
from qemx.circuit import ParseError, eval_expr, parse
from qemx.compile import Compiled, with_io
from qemx.engine import simulate
from qemx.noise import NoiseSpec, PRESETS
from qemx.recipes import compare, preset


# ------------------------------------------------------------------ templates
@pytest.mark.parametrize("key,params,value", [
    ("shor", dict(a=7, t=3), 0.75), ("shor", dict(a=11, t=3), 0.5), ("hhl", {}, 0.5), ("simon", dict(n=3, s=5), 1.0),
    ("qpe_ising", dict(t=3, phase=0.375, state="01"), 1.0), ("qpe_ising", dict(t=3, phase=0.375, state="00"), 1.0)])
def test_new_templates_ideal_values(key, params, value):
    tpl = T.build(key, params)
    r = simulate(tpl.spec, NoiseSpec(), "numpy", ideal=True)
    assert abs(tpl.observable.o @ r.probs - value) < 1e-9


def test_hhl_conditional_solution():
    tpl = T.build("hhl")
    r = simulate(tpl.spec, NoiseSpec(), "numpy", ideal=True)
    b = (np.arange(16)[:, None] >> np.arange(4)[None, :]) & 1
    pa = r.probs @ b[:, 3]
    assert abs((r.probs * b[:, 3]) @ (1 - 2 * b[:, 0]) / pa - 0.8) < 1e-9


def test_template_params_are_clamped_not_capped():
    p = T.clamp_params("qft", dict(n=12, s=10 ** 9))
    assert p["n"] == 12 and p["s"] == 2 ** 12 - 1          # no upper limit on n; s follows n


# ------------------------------------------------------------------ parser hardening
@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "9**9**9", "(" * 500 + "1" + ")" * 500, "a.b",
                                 "[1][0]", "lambda: 1", "open('x')", "1e400", "sqrt(-1)", ""])
def test_expression_reader_rejects(bad):
    with pytest.raises(ValueError):
        eval_expr(bad)


def test_parse_checks_qubit_range():
    with pytest.raises(ParseError):
        parse("h 3", 3)
    with pytest.raises(ParseError):
        parse("cx 0 -1", 3)


# ------------------------------------------------------------------ drag-and-drop edits
def test_edit_actions_and_validation():
    s = parse("h 0; cx 0 1", 3)
    s = E.apply(s, dict(type="add", name="x", q=2, col=0))
    assert any(o.name == "x" and o.qubits == (2,) for o in s.ops)
    s = E.apply(s, dict(type="add", name="rz", q=1, col=5))
    i = next(k for k, o in enumerate(s.ops) if o.name == "rz")
    s = E.apply(s, dict(type="param", idx=i, text="pi/8"))
    assert abs(s.ops[i].params[0] - math.pi / 8) < 1e-15
    j = next(k for k, o in enumerate(s.ops) if o.name == "cx")
    s = E.apply(s, dict(type="retarget", idx=j, which=1, q=2))
    assert s.ops[[k for k, o in enumerate(s.ops) if o.name == "cx"][0]].qubits == (0, 2)
    s = E.apply(s, dict(type="qubits", n=5))
    assert s.n == 5
    for bad in [dict(type="add", name="os.system", q=0), dict(type="add", name="h", q=9), "x", None,
                dict(type="param", idx=0, text="__import__('os')"), dict(type="move", idx=99, q=0, col=0),
                dict(type="retarget", idx=0, which=1, q=0), dict(type="qubits", n=0), dict(type="add", name="h", q=1.5)]:
        with pytest.raises(E.EditError):
            E.apply(s, bad)


def test_edits_keep_a_valid_order():
    s = parse("h 0; cx 0 1; x 1", 2)
    s2 = E.apply(s, dict(type="move", idx=2, q=0, col=0))      # x moved before everything on qubit 0
    cols = E.display_columns(list(s2.ops), 2)
    assert cols == sorted(cols)


# ------------------------------------------------------------------ fake devices
def test_device_ideal_matches_and_noise_is_reasonable():
    tpl = T.build("bell")
    comp = CP.build(tpl.spec, opts=CP.CompileOptions(device="fake_perth"))
    phys = comp.info["phys"]
    r = D.run_device(comp.implemented, "fake_perth", phys, comp.measure_map)
    ri = D.run_device(comp.implemented, "fake_perth", phys, comp.measure_map, ideal=True)
    assert abs(tpl.observable.o @ ri.probs - 1) < 1e-9
    assert 0.8 < tpl.observable.o @ r.probs < 1.0
    assert tpl.observable.o @ r.probs_meas <= tpl.observable.o @ r.probs + 1e-12      # readout only hurts here


def test_device_snapshots_do_not_change_the_result():
    tpl = T.build("qft")
    comp = CP.build(tpl.spec, opts=CP.CompileOptions(device="fake_perth"))
    a = D.run_device(comp.implemented, "fake_perth", comp.info["phys"], comp.measure_map)
    b = D.run_device(comp.implemented, "fake_perth", comp.info["phys"], comp.measure_map, snapshots=True)
    assert np.abs(a.rho - b.rho).max() < 1e-12 and len(b.snapshots) > 0


def test_device_avoids_broken_couplers():
    comp = CP.build(T.build("ghz").spec, opts=CP.CompileOptions(device="fake_fez"))
    dm = D.get_model("fake_fez", tuple(comp.info["phys"]))
    assert all(e is None or e < 0.5 for (nm, qa), e in dm.errors.items() if len(qa) == 2)


def test_twirled_device_errors_are_pauli():
    from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error
    err = depolarizing_error(0.01, 1).compose(thermal_relaxation_error(50e-6, 30e-6, 60e-9))
    tw = D._twirl_error(err)
    from qiskit.quantum_info import Kraus
    K = Kraus(tw.to_quantumchannel()).data
    assert len(K) <= 4


def test_compare_on_a_device():
    tpl = T.build("bell")
    comp = CP.build(tpl.spec, opts=CP.CompileOptions(device="fake_perth"))
    ideal, outs, _ = compare(comp, NoiseSpec(), tpl.observable, [preset(k) for k in ("Unmitigated", "PEC · inverse",
                                                                                     "Readout · TREX")], "fake_perth")
    assert all(o.result is not None for o in outs)
    assert abs(outs[2].result.mean - ideal) < abs(outs[0].result.mean - ideal)     # TREX removes readout bias


# ------------------------------------------------------------------ hardware record / replay
def _plan():
    tpl = T.build("bell")
    b = D.fake_backend("fake_perth")
    spec, mm, phys, info = D.compile_for_target(with_io(tpl.spec), b, D.good_target_for(b), "FakePerth")
    comp = Compiled(with_io(tpl.spec), spec, mm, info)
    rec = [preset(k) for k in ("Unmitigated", "ZNE · linear", "Readout · TREX", "Readout · M3",
                               "Virtual distillation (M=2)")]
    return H.make_plan(comp, tpl.observable, rec, 1000, tpl.symmetry, 0, b), tpl


def test_hardware_plan_and_replay_on_exact_frequencies():
    plan, tpl = _plan()
    assert "VD M=2" in plan.skipped
    ideal_runner = H.HardwareRunner().ideal
    freqs = {}
    for k, req in plan.requests.items():
        freqs[k] = ideal_runner.run(req.spec, NoiseSpec(), req.mm, ideal=True, backend="numpy").probs
    res = H.analyze(plan, freqs, repeats=20)
    assert len(res["rows"]) == 4 and res["hardware"]
    for r in res["rows"]:
        assert abs(r["estimate"] - 1.0) < 1e-9, r
        assert r["target shots"] + r["training shots"] + r["calibration shots"] == 1000      # exact (review F08)


@pytest.mark.slow
def test_hardware_local_run_with_sampler():
    plan, tpl = _plan()
    freqs = H.run_local(plan, "fake_perth", seed=3)
    res = H.analyze(plan, freqs, repeats=20)
    ok = [r for r in res["rows"] if not r["status"]]
    assert len(ok) >= 4 and all(0.5 < r["estimate"] < 1.2 for r in ok)
    assert set(res["dist"]["series"]) >= {"ideal", "noisy (unmitigated)", "readout-corrected"}


# ------------------------------------------------------------------ analysis
def test_rmse_algebra():
    assert A.shots_for(0.02, 1.0, 0.01) == math.inf
    n = A.shots_for(0.0, 1.0, 0.01)
    assert abs(n - 1e4) < 1e-6
    x = A.crossover(0.0, 4.0, 0.1, 1.0)
    assert abs(A.rmse_curve(0.0, 4.0, [x])[0] - A.rmse_curve(0.1, 1.0, [x])[0]) < 1e-12


def test_pec_wall_is_exponential():
    w = A.pec_wall(np.array([0, 10, 20]), [0.01], 4, 0.0, 0.01, 1.0)[0.01]
    r1, r2 = w["shots"][1] / w["shots"][0], w["shots"][2] / w["shots"][1]
    assert abs(r1 - r2) / r1 < 1e-9 and r1 > 1


def test_pec_mismatch_is_unbiased_at_ratio_one():
    tpl = T.build("qft")
    comp = CP.build(tpl.spec)
    rows = A.pec_mismatch(comp, NoiseSpec(p1=0.002, p2=0.02), tpl.observable, [0.5, 1.0, 1.5])
    b = {r["ratio"]: r["bias"] for r in rows if r["ratio"] is not None}
    assert abs(b[1.0]) < 1e-9 and b[0.5] < 0 < b[1.5]


def test_winner_map_shapes():
    rows = [dict(value=v, method=m, family="raw", bias=bb, v=vv) for v in (0, 1)
            for m, bb, vv in (("a", 0.1, 1.0), ("b", 0.0, 10.0))]
    vals, methods, win, best, second = A.winner_map(rows, [10, 1e6])
    assert win.shape == (2, 2) and win[0, 0] == 0 and win[0, 1] == 1


# ------------------------------------------------------------------ bundle, resources, security
def test_zip_contents_without_secrets():
    import pandas as pd
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=[0, 1], y=[1, 0]))
    b = make_zip("t", tables={"tab": pd.DataFrame({"a": [1]})}, figures={"f": fig}, state={"x": 1}, images=False)
    z = zipfile.ZipFile(io.BytesIO(b))
    names = set(z.namelist())
    assert {"README.txt", "tables/tab.csv", "tables/tab.tex", "figures/f.json",
            "state/qem_explorer_state.json"} <= names
    assert not any(n.endswith(".html") or n.endswith(".js") for n in names)


def test_scrub_and_explain():
    assert "•••" in RS.scrub("bad token abcdef123", "abcdef123")
    t, m = RS.explain(MemoryError(), "the run", 20)
    assert "memory" in (t + m).lower()
    t, m = RS.explain(RS.RunTimeout(), "the run")
    assert "time limit" in (t + m).lower()


def test_deadline_stops_executor():
    tpl = T.build("shor")
    with RS.deadline(1e-6):
        import time
        time.sleep(0.01)
        with pytest.raises(RS.RunTimeout):
            simulate(tpl.spec, PRESETS["Depolarizing"], "numpy")


def test_state_validation_rejects_bad_files():
    import streamlit as st  # noqa: F401
    from ui.state import validate_state
    good = dict(circuit=dict(n=2, ops=[dict(name="h", qubits=[0], params=[], kind="gate")]),
                observable=dict(key="Z0", label="⟨Z0⟩", tex=r"\langle Z_0\rangle", values=[1, -1, 1, -1]))
    v = validate_state(good)
    assert v["n"] == 2
    for mutate in [lambda d: d["circuit"]["ops"].append(dict(name="__import__", qubits=[0], kind="gate")),
                   lambda d: d["circuit"]["ops"].append(dict(name="h", qubits=[5], kind="gate")),
                   lambda d: d["observable"].update(values=[1, 2]),
                   lambda d: d.update(recipes=[dict(estimator="zne", params=dict(evil=1))]),
                   lambda d: d.update(inputs=["<script>"] * 2)]:
        d = json.loads(json.dumps(good))
        mutate(d)
        with pytest.raises(ValueError):
            validate_state(d)
    d = json.loads(json.dumps(good))
    d["observable"]["tex"] = r"\href{javascript:alert(1)}{x}"
    d["observable"]["label"] = "<img src=x onerror=alert(1)>"
    v = validate_state(d)
    assert "href" not in v["observable"][2] and "<" not in v["observable"][1]
