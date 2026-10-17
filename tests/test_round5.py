"""Round 5: the analysis engine (automatic views with a manual override), the five-qubit and d = 2 surface-code
templates, and the regressions found by the second QA pass."""
import math
import os

import numpy as np
import pytest

from qemx import compile as CP
from qemx import insight as IN
from qemx import observables as OBS
from qemx import templates as T
from qemx.noise import PRESETS
from qemx.recipes import compare, preset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sv(ops, n):
    return T.statevector(n, ops)


def _ev(a, label):
    return float(np.vdot(a, T._pauli_matrix(label) @ a).real)


def _ideal(tpl):
    a = _sv(tpl.spec.ops, tpl.spec.n)
    return float(np.vdot(a, np.asarray(tpl.observable.values) * a).real)


# ------------------------------------------------------------------ 1. the five-qubit code
def test_five_qubit_encoder_prepares_the_code_space():
    for th in (0.0, 0.7, 2.1):
        a = _sv([T._g("ry", 0, params=(th,))] + T._five_ops(), 5)
        assert all(_ev(a, s) == pytest.approx(1.0, abs=1e-9) for s in T.FIVE_STABILISERS)
        assert _ev(a, "ZZZZZ") == pytest.approx(math.cos(th), abs=1e-9)
        assert _ev(a, "XXXXX") == pytest.approx(math.sin(th), abs=1e-9)


def test_five_qubit_decoder_table_is_perfect():
    tab = T.five_qubit_decoder_table()
    assert len(tab) == 16 and tab[(0, 0, 0, 0)] is False           # every syndrome used once; no error, no flip
    assert sum(tab.values()) == 10                                  # the five X and five Y errors flip Z_L


@pytest.mark.parametrize("error", ["X on qubit k", "Y on qubit k", "Z on qubit k"])
def test_five_qubit_code_corrects_every_single_qubit_error(error):
    for k in range(5):
        tpl = T.build("five_qubit", dict(theta=1.0, error=error, k=k))
        assert _ideal(tpl) == pytest.approx(math.cos(1.0), abs=1e-9), (error, k)
    raw = T.build("five_qubit", dict(theta=1.0, error="X on qubit k", k=2, readout="raw (no correction)"))
    assert _ideal(raw) == pytest.approx(-math.cos(1.0), abs=1e-9)


def test_five_qubit_decoding_beats_raw_readout_under_weak_noise():
    prm = dict(theta=0.0, error="depolarizing channel on every qubit", p=0.03)
    ideal = PRESETS["Ideal (no noise)"]
    vals = {}
    for ro in ("decoded (lookup table)", "raw (no correction)"):
        tpl = T.build("five_qubit", dict(prm, readout=ro))
        _, outs, _ = compare(CP.build(tpl.spec), ideal, tpl.observable, [preset("Unmitigated")])
        vals[ro] = outs[0].result.mean
    assert vals["decoded (lookup table)"] > vals["raw (no correction)"] + 0.02


# ------------------------------------------------------------------ 2. the d = 2 surface code
def test_surface_d2_code_space_and_logicals():
    for th in (0.0, 0.9):
        tpl = T.build("surface_d2", dict(theta=th))
        a = _sv(tpl.spec.ops, 4)
        for s in ("XXXX", "ZZII", "IIZZ"):
            assert _ev(a, s) == pytest.approx(1.0, abs=1e-9)
        assert _ev(a, "ZIZI") == pytest.approx(math.cos(th), abs=1e-9)
        assert _ev(a, "XXII") == pytest.approx(math.sin(th), abs=1e-9)


@pytest.mark.parametrize("syndrome,qubits", [("transversal readout", 4), ("ancilla extraction (3 ancillas)", 7)])
def test_surface_d2_detects_single_errors(syndrome, qubits):
    for err, k, detected, flips in (("X on qubit k", 0, True, True), ("X on qubit k", 1, True, False),
                                    ("Y on qubit k", 2, True, True), ("Z on qubit k", 3, False, False),
                                    ("none", 0, False, False)):          # Z_L = Z0Z2: only X/Y on q0, q2 flip it
        tpl = T.build("surface_d2", dict(theta=1.0, error=err, k=k, syndrome=syndrome))
        assert tpl.spec.n == qubits
        ideal, outs, _ = compare(CP.build(tpl.spec), PRESETS["Ideal (no noise)"], tpl.observable,
                                 [preset("Unmitigated"), preset("Symmetry verification", tpl.symmetry)],
                                 symmetry=tpl.symmetry)
        assert outs[0].result.mean == pytest.approx(-math.cos(1.0) if flips else math.cos(1.0), abs=1e-9)
        assert outs[1].result.accept == pytest.approx(0.0 if detected else 1.0, abs=1e-9)


def test_surface_d2_plaquette_ancilla_sees_z_errors():
    tpl = T.build("surface_d2", dict(theta=0.0, error="Z on qubit k", k=0, syndrome="ancilla extraction (3 ancillas)"))
    a = _sv(tpl.spec.ops, 7)
    p = np.abs(a) ** 2
    q4 = float(sum(p[x] for x in range(128) if (x >> 4) & 1))
    assert q4 == pytest.approx(1.0)                                  # the X plaquette fired


def test_qec_library_stays_small():
    for key in T.CATEGORIES["FTQC · error correction"]:
        assert T.build(key).spec.n <= 9, key


# ------------------------------------------------------------------ 3. the analysis engine
@pytest.mark.parametrize("key,params,primary", [
    ("shor", {}, "distribution"), ("bv", {}, "distribution"), ("grover", {}, "distribution"),
    ("ghz", {}, "expectation"), ("hea_vqe", {"n": 2, "layers": 0}, "expectation"),
    ("five_qubit", {}, "qec"), ("surface_d2", {}, "qec"), ("steane", {}, "qec"), ("qec_3q", {}, "qec"),
    ("repetition", {}, "qec")])
def test_profile_picks_the_view_from_the_mathematics(key, params, primary):
    tpl = T.build(key, params)
    prof = IN.profile(key, params, False, tpl.observable, ["raw", "zne"], tpl.spec.n)
    assert prof.primary == primary and prof.order[0] == primary
    assert prof.order[1] == "diagnostics" and prof.reasons
    assert set(prof.order) <= set(IN.VIEWS)


def test_custom_circuits_and_hamiltonians():
    from qemx import hamiltonian as HM
    tpl = T.build("five_qubit")
    prof = IN.profile("five_qubit", {}, True, tpl.observable, ["raw"], 5)       # edited: no known syndrome layout
    assert prof.primary == "expectation" and "qec" not in prof.order and prof.custom
    assert any("custom circuit" in r for r in prof.reasons)
    ps = HM.parse_pauli_sum("Z0*Z1 + 0.5*X0", 2)
    assert IN.profile(None, None, True, ps, [], 2).observable_kind == "hamiltonian"
    assert IN.value_kind([0, 1, 1, 0]) == "projector" and IN.value_kind([1, -1]) == "parity"
    assert IN.value_kind([0.5, 2]) == "general"


@pytest.mark.parametrize("key,params,backend,readout", [
    ("shor", {}, "numpy", "Readout · tensored"), ("bv", {"n": 4, "s": 11}, "numpy", "Readout · M3"),
    ("ghz", {"n": 3}, "fake_perth", "Readout · TREX")])
def test_mitigated_distributions_reproduce_the_estimators(key, params, backend, readout):
    tpl = T.build(key, params)
    opts = CP.CompileOptions(mode="isa", device=backend) if backend.startswith("fake") else CP.CompileOptions()
    comp = CP.build(tpl.spec, opts=opts)
    nz = PRESETS["Hardware-realistic (illustrative)"]
    ideal, outs, runner = compare(comp, nz, tpl.observable, [preset("Unmitigated"), preset("ZNE · Richardson"),
                                                             preset(readout)], backend)
    noise = runner.device_model.equivalent_noise() if runner.device_model is not None else nz
    d = IN.distributions(comp, noise, outs, runner, backend)
    v = np.asarray(tpl.observable.values)
    est = [o.result.mean for o in outs]
    assert float(v @ d["series"]["ideal"]) == pytest.approx(ideal, abs=1e-9)
    assert float(v @ d["series"]["noisy (unmitigated)"]) == pytest.approx(est[0], abs=1e-9)
    assert float(v @ d["series"]["ZNE-extrapolated"]) == pytest.approx(est[1], abs=1e-9)
    assert float(v @ d["series"]["readout-corrected"]) == pytest.approx(est[2], abs=1e-9)
    assert all(abs(float(np.sum(p)) - 1) < 1e-9 for p in d["series"].values())      # quasi-probabilities sum to 1
    assert len(d["readout"]) == comp.logical.n


def test_marginals_syndromes_and_metrics():
    tpl = T.build("shor")
    n = tpl.spec.n
    bits = IN.relevant_bits(tpl.observable.values, n)
    assert bits == [0, 1, 2]
    p = np.abs(_sv(tpl.spec.ops, n)) ** 2
    m = IN.marginal(p, bits, n)
    assert np.allclose(m[[0, 2, 4, 6]], 0.25) and m.sum() == pytest.approx(1.0)
    ch = IN.qec_checks("surface_d2", {}, 4)
    s = IN.syndrome_distribution(np.eye(16)[0b0001], ch, 4)          # bit q0 set: Z0Z1 fires, Z2Z3 does not
    assert s[0b10] == 1.0
    assert IN.tvd([1, 0], [0, 1]) == 1.0 and IN.hellinger_fidelity([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)
    assert IN.logical_error(0.3, 0.5) == pytest.approx(0.2) and math.isnan(IN.logical_error(0.3, 0.0))


def test_analysis_charts_build():
    from qemx.viz import charts as CH
    tpl = T.build("five_qubit", dict(error="depolarizing channel on every qubit"))
    comp = CP.build(tpl.spec)
    ideal, outs, runner = compare(comp, PRESETS["Depolarizing"], tpl.observable,
                                  [preset("Unmitigated"), preset("ZNE · Richardson"), preset("Readout · M3")])
    d = IN.distributions(comp, PRESETS["Depolarizing"], outs, runner)
    ch = IN.qec_checks("five_qubit", {}, 5)
    figs = [CH.output_histogram(d["series"], 5, [0, 1]), CH.readout_bars(d["readout"]),
            CH.syndrome_histogram({k: IN.syndrome_distribution(v, ch, 5) for k, v in d["series"].items()},
                                  [c[0] for c in ch]),
            CH.logical_error_bars([dict(method=o.title, family="raw", pL=IN.logical_error(o.result.mean, ideal))
                                   for o in outs], dark=True)]
    assert all(f.to_dict()["data"] for f in figs)


# ------------------------------------------------------------------ 4. the analysis panel in the app
@pytest.mark.slow
def test_analysis_panel_views_and_override():
    import qiskit_ibm_runtime.fake_provider  # noqa: F401
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()
    at.switch_page("pages/1_Playground.py").run()

    def tab(name):
        at.session_state["w_pg_tabs"] = name
        at.session_state["page_tab"] = {"playground": name}
        at.run()
        assert not at.exception, [e.value for e in at.exception]

    for key, primary in (("five_qubit", "qec"), ("shor", "distribution")):
        tab("Build")
        at.selectbox(key="w_tpl_cat").set_value(T.category_of(key)).run()
        at.selectbox(key="w_tpl").set_value(key).run()
        next(b for b in at.button if b.key == "tpl_load").click().run()
        at.session_state["recipe_names"] = ["Unmitigated", "ZNE · Richardson", "Readout · M3"]
        at.session_state["run_requested"] = True
        at.run()
        tab("Mitigate & compare")
        assert at.session_state["w_an_view"] == "auto"
        assert at.session_state["results"]["dist"]["series"]
        prof_label = at.selectbox(key="w_an_view").options[0]
        assert IN.VIEWS[primary] in prof_label
        for v in ("expectation", "distribution", "diagnostics"):
            at.selectbox(key="w_an_view").set_value(v).run()
            assert not at.exception, [e.value for e in at.exception]
        assert any("Manual view" in c.value for c in at.caption)


# ------------------------------------------------------------------ 5. the handbook
def test_handbook_covers_advanced_qem_and_the_qec_roadmap():
    import re
    text = open(os.path.join(ROOT, "docs", "theory.md"), encoding="utf-8").read()
    for heading in ("## 16. Advanced mitigation methods", "### 16.1 Symmetry verification and symmetry expansion",
                    "### 16.2 Zero-noise extrapolation at the pulse level", "### 16.3 Quantum subspace expansion",
                    "### 16.4 Classical shadows", "### 16.5 M3 in depth", "## 17. Toward fault tolerance",
                    "### 17.1 The stabilizer formalism", "### 17.2 A taxonomy of codes", "### 17.3 Decoding",
                    "### 17.5 Logical error mitigation"):
        assert heading in text, heading
    blocks = re.findall(r"\$\$(.+?)\$\$", text, flags=re.S)
    assert blocks and not any("\n" in b for b in blocks)          # KaTeX needs one-line display math here
    refs = text.split("## References", 1)[1]
    for key in ("Huang 2020", "McClean 2017", "Nation 2021", "Temme 2017", "Gottesman 1997", "Google 2025",
                "Piveteau 2021", "Wahl 2023"):
        assert "[%s]" % key in refs, key


# ------------------------------------------------------------------ 6. fixes from the second QA pass
def test_multi_check_post_selection_halves_the_residual_error():
    from qemx.methods.symmetry import parse_checks
    tpl = T.build("surface_d2", dict(theta=0.0, error="bit-flip channel on every data qubit", p=0.05))
    assert tpl.symmetry == ((0, 1), 1, "2 3")
    rc = preset("Symmetry verification", tpl.symmetry)
    assert rc.p["more"] == "2 3" and "Z2Z3=+1" in rc.title()
    single = preset("Symmetry verification", ((0, 1, 2, 3), 1))
    _, outs, _ = compare(CP.build(tpl.spec), PRESETS["Ideal (no noise)"], tpl.observable, [rc, single])
    pl_both, pl_one = ((1 - o.result.mean) / 2 for o in outs)
    assert pl_both < 0.6 * pl_one                                 # both checks catch X0X3-type pairs
    assert parse_checks("2 3; 5,6") == [(2, 3), (5, 6)] and parse_checks("") == []
    for bad in ("2 3; x", "__import__('os')", "1" * 300):
        with pytest.raises(ValueError):
            parse_checks(bad)


def test_state_files_keep_further_checks():
    import streamlit  # noqa: F401
    from ui.state import validate_state
    good = dict(circuit=dict(n=4, ops=[]), observable=dict(values=[1.0] * 16), symmetry=[0, 1, 1],
                symmetry_more="2 3")
    assert validate_state(good)["symmetry"] == ((0, 1), 1, "2 3")
    with pytest.raises(ValueError):
        validate_state(dict(good, symmetry_more="7 8"))                  # qubits the circuit does not have
    with pytest.raises(ValueError):
        validate_state(dict(good, symmetry_more="2; import os"))


def test_qec_view_needs_the_codes_own_logical_observable():
    tpl = T.build("five_qubit")
    assert IN.profile("five_qubit", {}, False, tpl.observable, ["raw"], 5, logical=False).primary == "expectation"
    assert IN.intended_logical("surface_d2", {"theta": 0.0}) == pytest.approx(1.0)
    assert IN.intended_logical("five_qubit", {}) == pytest.approx(math.cos(1.0472))
    assert IN.intended_logical("ghz", {}) is None


@pytest.mark.slow
def test_empty_and_failed_runs_and_stale_results_do_not_break_the_panel():
    import qiskit_ibm_runtime.fake_provider  # noqa: F401
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()
    at.switch_page("pages/1_Playground.py").run()

    def tab(name):
        at.session_state["w_pg_tabs"] = name
        at.session_state["page_tab"] = {"playground": name}
        at.run()
        assert not at.exception, [e.value for e in at.exception]

    at.session_state["recipe_names"] = []                          # no contenders: a warning, no stored result
    at.session_state["custom_recipes"] = []
    at.session_state["run_requested"] = True
    at.run()
    assert not at.exception and at.session_state["results"] is None
    at.session_state["ops"] = [dict(name="depol", qubits=[0], params=[0.1], kind="noise", tag="")]  # no gates:
    at.session_state["recipe_names"] = ["ZNE · Richardson", "ZNE · linear"]
    at.session_state["run_requested"] = True
    at.run()
    tab("Mitigate & compare")
    assert any("Every method failed" in e.value for e in at.error)
    tab("Build")                                                    # run Shor, then load a code without running
    at.selectbox(key="w_tpl_cat").set_value(T.category_of("shor")).run()
    at.selectbox(key="w_tpl").set_value("shor").run()
    next(b for b in at.button if b.key == "tpl_load").click().run()
    at.session_state["recipe_names"] = ["Unmitigated"]
    at.session_state["run_requested"] = True
    at.run()
    tab("Build")
    at.selectbox(key="w_tpl_cat").set_value(T.category_of("five_qubit")).run()
    at.selectbox(key="w_tpl").set_value("five_qubit").run()
    next(b for b in at.button if b.key == "tpl_load").click().run()
    tab("Mitigate & compare")
    assert IN.VIEWS["distribution"] in at.selectbox(key="w_an_view").options[0]    # the stored (Shor) result


def test_batch_log_axis_has_full_tick_labels():
    """Log axes of the batch chart label 1-2-5 ticks in full (0.005, 0.01 …), never as bare '2' / '5' digits."""
    from qemx.viz import batch_charts as BC
    assert BC.log_ticks([0.003, 0.1]) == [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    assert BC.log_ticks([1e-5, 0.3]) == [1e-05, 0.0001, 0.001, 0.01, 0.1]
    assert BC.log_ticks([]) == [] and BC.log_ticks([0.0123]) == []
    rows = [dict(algorithm="GHZ state", simulator="numpy", noise="Depolarizing", λ=lam, method=m, family=f,
                 estimate=1 - b, ideal=1.0, bias=-b, sd=0.5, var1=0.25, gamma=1.0, status="ok")
            for lam, b in ((0.5, 0.004), (1, 0.02), (2, 0.08)) for m, f in (("Unmitigated", "raw"),)]
    fig = BC.versus_scale(rows, 10000, "Depolarizing", metric="|bias|")
    ax = fig.layout.yaxis
    assert ax.type == "log" and ax.dtick is None
    assert list(ax.ticktext) == ["%g" % t for t in ax.tickvals] and "0.01" in ax.ticktext


def test_qiskit_entry_points_share_one_process_wide_lock():
    """Every entry into Qiskit's native code holds the same re-entrant lock (Streamlit runs sessions in threads)."""
    import threading
    from qemx import backends, circuit, compile as C, device, layout_search, resources as R, stages, templates
    from qemx.backends import aer
    from qemx.viz import qiskit_draw, topology
    assert C.QISKIT_LOCK is R.QISKIT_LOCK
    for fn in (C.build, C.coupling_map, C.lattice_target, stages.compile_stages, device.compile_for_target,
               device.device_circuit, device.run_device, layout_search.rank_layouts, circuit.to_qiskit,
               templates.shor15, templates.hhl2, aer.to_aer_circuit, aer.AerBackend.execute, qiskit_draw.png):
        assert getattr(fn, "__wrapped__", None) is not None, fn.__name__
    with R.QISKIT_LOCK:                         # re-entrant: a locked function may call another one
        C.coupling_map("line", 3)
    # several threads compiling and building circuits at once finish with identical results
    spec = templates.REGISTRY["ghz"][1](n=4).spec
    opts = C.CompileOptions(mode="isa", coupling="line")
    out, errors = [], []

    def work():
        try:
            out.append(C.build(spec, opts=opts).implemented.count("2q"))
        except Exception as e:                  # pragma: no cover - reported below
            errors.append(e)
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(set(out)) == 1


def test_streamlit_runs_do_not_overlap():
    """fastReruns is off: an interrupted run finishes before the next one starts (no two runs of one session)."""
    import pathlib
    import tomllib
    cfg = tomllib.loads((pathlib.Path(__file__).parents[1] / ".streamlit" / "config.toml").read_text())
    assert cfg["runner"]["fastReruns"] is False
