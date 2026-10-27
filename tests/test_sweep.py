"""Sweeps of inputs, bases, template parameters and noise parameters (qemx/sweep.py) and register-wide inputs."""
import math

import pytest

from qemx import batch as B
from qemx import compile as CP
from qemx import observables as OBS
from qemx import sweep as SW
from qemx import templates as T
from qemx.circuit import CircuitSpec


def _qft_body(n, m=None):
    ops = tuple(T.qft_body(list(range(n)), m) + T.swap_network(list(range(n))))
    return CircuitSpec(n, ops, "QFT body")


def _current(spec, k):
    n = spec.n
    return dict(spec=spec, inputs=["0"] * n, basis="Z" * n, observable=OBS.projector(k, n), symmetry=None,
                relabel=False, name=spec.name)


@pytest.mark.parametrize("k", [0, 1, 5])
def test_exact_qft_maps_the_phase_ramp_to_k(k):
    ax = SW.parse("ramp: %d" % k)
    rows = B.run(B.plan([B.CURRENT], ["Depolarizing"], [1.0], ["numpy"], ax), ["Unmitigated"],
                 current=_current(_qft_body(4), k), opts=CP.CompileOptions(), shots=1000, t_shot_us=1, axes=ax)
    assert rows[0]["ideal"] == pytest.approx(1.0, abs=1e-9)
    assert rows[0]["algorithm"].endswith("ramp:%d" % k) and rows[0]["input"] == "ramp:%d" % k


def test_aqft_on_ramp1_hits_its_truncation_ceiling():
    """|<0001| AQFT_{m=2} |ramp:1>|^2 = 0.961940 at n = 4, while |+...+> (ramp:0) is blind to the truncation."""
    ax = SW.parse("input: ramp:1, plus")
    cur = _current(_qft_body(4, m=2), 1)
    rows = B.run(B.plan([B.CURRENT], ["Depolarizing"], [1.0], ["numpy"], ax), ["Unmitigated"], current=cur,
                 opts=CP.CompileOptions(), shots=1000, t_shot_us=1, axes=ax)
    ideal = {r["input"]: r["ideal"] for r in rows}
    assert ideal["ramp:1"] == pytest.approx(0.961940, abs=5e-7)
    assert ideal["plus"] == pytest.approx(0.0, abs=1e-12)
    cur0 = _current(_qft_body(4, m=2), 0)
    r0 = B.run(B.plan([B.CURRENT], ["Depolarizing"], [1.0], ["numpy"], SW.parse("ramp: 0")), ["Unmitigated"],
               current=cur0, opts=CP.CompileOptions(), shots=1000, t_shot_us=1, axes=SW.parse("ramp: 0"))
    assert r0[0]["ideal"] == pytest.approx(1.0, abs=1e-9)


def test_register_inputs_and_bases():
    assert SW.input_tokens("plus", 3) == ["+"] * 3 and SW.input_tokens("-i", 2) == ["-i", "-i"]
    assert SW.input_tokens("bits:011", 3) == ["1", "1", "0"]            # written q2 q1 q0
    toks = SW.input_tokens("ramp:1", 3)
    assert [float(t[3:]) for t in toks] == pytest.approx([0.875, 0.75, 0.5])
    assert all(CP.valid_input(t) for t in toks)
    assert SW.basis_string("x", 4) == "XXXX" and SW.basis_string("ZZXY", 4) == "ZZXY"
    for bad in (("bits:01", 3), ("ramp:x", 3), ("foo", 2)):
        with pytest.raises(SW.SweepError):
            SW.input_tokens(*bad)
    with pytest.raises(SW.SweepError):
        SW.basis_string("ZX", 3)


def test_parse_is_strict_and_knows_every_kind():
    ax = SW.parse("ramp: 0, 1\nbasis: Z, X; m: 1, 2\np2: 0.01, 2*0.01  # comment", ["aqft"])
    assert [(a.name, a.kind) for a in ax] == [("input", "input"), ("basis", "basis"), ("m", "param"), ("p2", "noise")]
    assert ax[3].values == (0.01, 0.02) and ax[2].values == (1, 2)
    assert SW.parse("angles: VQE-optimized", ["hea_vqe"])[0].values == ("VQE-optimized",)    # 1.0 spelling accepted
    for bad in ("m 1, 2", "m: ", "zz: 1", "p2: 2", "m: __import__('os')", "m: 1\nm: 2", "ramp: 0.5",
                "x" * (SW.MAX_TEXT + 1), "basis: ZQ"):
        with pytest.raises(SW.SweepError):
            SW.parse(bad, ["aqft"])


def test_plan_applies_each_axis_only_where_it_means_something():
    ax = SW.parse("m: 1, 2\np2: 0.01, 0.02", ["aqft", "ghz"])
    sc = B.plan(["aqft", "ghz"], ["Depolarizing"], [1.0], ["numpy", "fake_fez"], ax)
    by = {}
    for s in sc:
        by.setdefault((s.source, s.backend), []).append(s.variant)
    assert len(by[("aqft", "numpy")]) == 4 and len(by[("ghz", "numpy")]) == 2      # GHZ has no parameter m
    assert len(by[("aqft", "fake_fez")]) == 2 and by[("ghz", "fake_fez")] == [()]    # a device keeps its own noise
    with pytest.raises(SW.SweepError):
        SW.variants([SW.Axis("a", "param", tuple(range(64)))] * 3)


def test_noise_override_and_rows_carry_swept_values():
    ax = SW.parse("p2: 0, 0.05")
    rows = B.run(B.plan(["ghz"], ["Depolarizing"], [1.0], ["numpy"], ax), ["Unmitigated"], current=None,
                 opts=CP.CompileOptions(), shots=1000, t_shot_us=1, axes=ax)
    est = {r["p2"]: r["estimate"] for r in rows}
    assert est[0.0] > est[0.05] and all(r["sweep"].startswith("p2=") for r in rows)
    from qemx.viz import batch_charts as BC
    fig, shown, total = BC.versus_sweep(rows, 10000, "p2", metric="|bias|")
    assert shown == total == 1 and len(fig.data) == 1


def test_register_io_callback_sets_every_qubit():
    import streamlit as st
    from ui import state as S
    S.init()
    ss = st.session_state
    ss.w_io_all_in, ss.w_io_all_bs = "ramp:1", "X"
    S.cb_register_io()
    n = int(ss.n)
    assert not ss.io_error and ss.basis == "X" * n and ss.inputs == SW.input_tokens("ramp:1", n)
    assert CP.with_io(CircuitSpec(n, (), "x"), ss.inputs, ss.basis).ops[0].name == "h"
    ss.w_io_all_in, ss.w_io_all_bs = "bits:1", ""
    S.cb_register_io()
    assert "bits" in ss.io_error and ss.inputs == SW.input_tokens("ramp:1", n)      # unchanged on an error
    state = S.to_json()
    assert S.validate_state(state)["inputs"] == ss.inputs                           # phase inputs survive save/load


@pytest.mark.slow
def test_batch_page_sweep_and_playground_register_inputs_end_to_end():
    import os
    from streamlit.testing.v1 import AppTest
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=900).run()
    at.switch_page("pages/8_Batch_Run.py").run()
    at.session_state["w_b_algos"] = ["aqft"]
    at.session_state["w_b_methods"] = ["Unmitigated", "ZNE · Richardson"]
    at.session_state["w_b_noises"] = ["Depolarizing"]
    at.session_state["w_b_scales"] = [1.0]
    at.session_state["w_b_sims"] = ["numpy"]
    at.session_state["w_b_sweep"] = "m: 1, 2, 3\nbasis: Z, X"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    next(b for b in at.button if b.key == "b_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    rows = at.session_state["batch_result"]["rows"]
    assert len(rows) == 3 * 2 * 2 and not any(r["status"].startswith("failed") for r in rows)
    assert any("swept values" in s.value for s in at.subheader)
    at.session_state["w_b_sweep"] = "zz: 1"                     # an invalid sweep disables Run all, never crashes
    at.run()
    assert not at.exception and any("Sweep:" in e.value for e in at.error)
    assert next(b for b in at.button if b.key == "b_run").disabled
    at.switch_page("pages/1_Playground.py").run()
    at.session_state["w_io_all_in"] = "ramp:2"
    at.session_state["w_io_all_bs"] = "Y"
    at.run()
    next(b for b in at.button if b.key == "io_all_apply").click().run()
    assert not at.exception, [e.value for e in at.exception]
    n = int(at.session_state["n"])
    assert at.session_state["inputs"] == SW.input_tokens("ramp:2", n) and at.session_state["basis"] == "Y" * n
