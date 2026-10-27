"""Round 4: collision-free drawings, the responsive Bloch grid, LaTeX tooltips, crosstalk and DD, Hamiltonians,
compile stages and the logical/physical diff, topology and noise-aware layout, live variational parameters, batch
runs and the Learn page."""
import dataclasses
import itertools
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from qemx import batch as B
from qemx import compile as CP
from qemx import export as EX
from qemx import hamiltonian as HM
from qemx import layout_search as LS
from qemx import learn as L
from qemx import observables as OBS
from qemx import stages as STG
from qemx import templates as T
from qemx import variational as V
from qemx.circuit import parse
from qemx.noise import PRESETS, NoiseSpec
from qemx.recipes import PRESETS as RECIPE_PRESETS, compare, preset
from qemx.timeline import preview
from qemx.viz import composer as CV
from qemx.viz import states as SV
from qemx.viz import topology as TOPO
from qemx.textio import child_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTRACT = CP.CompileOptions()


# ------------------------------------------------------------------ 1. drawings without collisions
def _boxes(fig):
    """Gate and measurement boxes: filled rectangles narrower than one column."""
    out = []
    for s in fig.layout.shapes:
        if s.type != "rect" or s.layer != "below" or (s.x1 - s.x0) > 0.8:
            continue
        if str(s.fillcolor or "").replace(" ", "") in ("", "rgba(0,0,0,0)"):
            continue
        out.append((s.x0, s.x1, s.y0, s.y1))
    return out


def _overlaps(boxes, eps=1e-6):
    bad = []
    for a, b in itertools.combinations(boxes, 2):
        if a[0] < b[1] - eps and b[0] < a[1] - eps and a[2] < b[3] - eps and b[2] < a[3] - eps:
            bad.append((a, b))
    return bad


@pytest.mark.parametrize("kind,folding,lam", [("zne", "global", 3.0), ("zne", "local", 3.0), ("zne", "global", 5.0),
                                              ("pec", "global", 3.0), ("dd", "global", 3.0), ("twirl", "global", 3.0)])
def test_method_circuits_draw_without_overlapping_boxes(kind, folding, lam):
    tpl = T.build("qft", {"n": 3})
    comp = CP.build(tpl.spec)
    title, spec = preview(kind, comp, PRESETS["Hardware-realistic (illustrative)"], lam=lam, folding=folding,
                          dd="XY4", seed=3, pec_min1=True)
    for width in (560, 1150):
        fig = CV.draw(spec, width_px=width, measure_map=comp.measure_map)
        boxes = _boxes(fig)
        assert boxes and not _overlaps(boxes), (kind, folding, width, _overlaps(boxes)[:2])


def test_fold_bands_never_share_a_column():
    tpl = T.build("ghz", {"n": 4})
    comp = CP.build(tpl.spec)
    _, spec = preview("zne", comp, PRESETS["Depolarizing"], lam=3.0, folding="local")
    cols, _ = CV.columns(spec)
    seg = {}
    for c, i in cols:
        seg.setdefault(c, set()).add(spec.ops[i].tag in ("fold", "fold-inv") and spec.ops[i].tag)
    assert all(len(v - {False, ""}) <= 1 for v in seg.values())


def test_diff_drawing_is_collision_free_and_complete():
    tpl = T.build("qft", {"n": 4})
    st = STG.compile_stages(tpl.spec, opts=CP.CompileOptions(mode="isa", coupling="heavy-hex", layout="trivial"))
    kw = STG.combined(st)
    spec = kw.pop("spec")
    assert spec.n == st.logical.n + st.native.n
    placed = [i for _, i in kw["positions"]]
    assert len(placed) == len(set(placed))
    assert set(range(len(st.logical.ops), len(spec.ops))) <= set(placed)          # every native gate is drawn
    fig = CV.draw(spec, width_px=820, height_per_qubit=38, **kw)
    assert not _overlaps(_boxes(fig))


# ------------------------------------------------------------------ 2. responsive Bloch grid
@pytest.mark.parametrize("n", [1, 4, 5, 8, 9, 16, 17])
def test_bloch_spheres_wrap_into_rows(n):
    r = np.tile([0.0, 0.0, 1.0], (n, 1))
    fig = SV.bloch_row(r, r)
    scenes = [k for k in fig.layout if str(k).startswith("scene")]
    assert len(scenes) == n
    doms = [fig.layout[k].domain for k in scenes]
    per_row = n if n <= 4 else (4 if n <= 16 else 5)
    assert len({round(d.y[0], 4) for d in doms}) == math.ceil(n / per_row)
    for a, b in itertools.combinations(doms, 2):                        # no two spheres share space
        assert a.x[1] <= b.x[0] + 1e-9 or b.x[1] <= a.x[0] + 1e-9 or a.y[1] <= b.y[0] + 1e-9 or b.y[1] <= a.y[0] + 1e-9


def test_flat_projections_pair_up_per_qubit():
    n = 6
    r = np.tile([0.5, 0.0, 0.5], (n, 1))
    fig = SV.bloch_projections(r, r)
    used = {t.xaxis for t in fig.data}                            # one panel per view (side, top) per qubit
    assert len(used) == 2 * n


# ------------------------------------------------------------------ 3. LaTeX tooltips
def test_cost_table_headers_have_formula_tooltips():
    import streamlit  # noqa: F401
    from ui import metrics as MX
    cfg = MX.column_config(["method", "estimate", "bias", "sd", "RMSE", "γ", "QPU s", "status"])
    for c in ("bias", "sd", "RMSE", "γ", "QPU s"):
        h = cfg[c]["help"]
        # v1.2: tooltips show the formula as Unicode math (KaTeX is unreliable in Streamlit's help popups)
        assert "$" not in h and "\\" not in h and "=" in h and len(h) > 40, c
    assert "√" in cfg["sd"]["help"] and "γ" in cfg["γ"]["help"]
    assert "method" not in cfg                                  # plain text columns keep their defaults
    assert all("$" in MX.LAYOUT_HELP[k] for k in ("F", "1 − F", "vs now"))     # source text keeps its TeX


# ------------------------------------------------------------------ 4. crosstalk and dynamical decoupling
def test_crosstalk_is_refocused_by_dd():
    spec = parse(L.SPECTATOR_RAMSEY, 3)
    comp = CP.build(spec)
    ideal, outs, _ = compare(comp, PRESETS["Crosstalk (spectator)"], OBS.z(0, 3),
                             [preset("Unmitigated"), preset("Dynamical decoupling (XY4)")])
    raw, dd = (o.result.mean for o in outs)
    assert ideal == pytest.approx(1.0)
    assert raw < 0.9 and dd > 0.95


def test_crosstalk_off_changes_nothing_and_follows_coupling():
    spec = parse(L.SPECTATOR_RAMSEY, 3)
    comp = CP.build(spec)
    nz = PRESETS["Crosstalk (spectator)"]
    _, o0, _ = compare(comp, dataclasses.replace(nz, xtalk_z=0.0, xtalk_zz=0.0), OBS.z(0, 3), [preset("Unmitigated")])
    assert o0[0].result.mean > 0.99
    far = comp.implemented.replace(coupling=((0, 2),))                  # q1 is no longer q0's neighbor
    _, o1, _ = compare(dataclasses.replace(comp, implemented=far), nz, OBS.z(0, 3), [preset("Unmitigated")])
    assert o1[0].result.mean > 0.99


# ------------------------------------------------------------------ 5. Hamiltonians
def test_parser_accepts_the_documented_forms():
    ps = HM.parse_pauli_sum("0.5*Z0*Z1 + 0.2*X0 - 0.1*Y1", 2)
    assert len(ps.terms) == 3 and ps.constant == 0
    ps = HM.parse_pauli_sum("pi/4 X0 X1 - sqrt(2)/2*Z1 + 1", 2)
    assert ps.constant == pytest.approx(1.0)
    assert HM.parse_pauli_sum("X0*X0", 1).constant == pytest.approx(1.0)       # X² = I


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "exec('1')", "Z0*Z9", "1 +", "X0*Y0",
                                 "Z0 ; import os", "lambda: 0", "Z0" + " + Z0" * 400, "a" * 5000, "", "2**1000*Z0",
                                 "Z0.__class__", "open('x')"])
def test_parser_rejects_everything_else(bad):
    with pytest.raises(HM.HamiltonianError):
        HM.parse_pauli_sum(bad, 2)


def test_h2_ground_energy_and_grouping():
    txt = HM.examples(2)["H₂ molecule (STO-3G, 0.735 Å, 2 qubits)"]
    ps = HM.parse_pauli_sum(txt, 2)
    assert np.linalg.eigvalsh(ps.matrix())[0] == pytest.approx(-1.85727503, abs=1e-6)
    groups = ps.groups()
    assert len(groups) == 2                                                   # {Z0, Z1, Z0Z1} and {X0X1}


@pytest.mark.parametrize("opts", [ABSTRACT, CP.CompileOptions(mode="isa", coupling="line", basis_2q="cz"),
                                  CP.CompileOptions(mode="isa", coupling="heavy-hex")])
def test_hamiltonian_estimates_are_exact_without_noise(opts):
    tpl = T.build("hea_vqe", {"n": 3, "layers": 1, "angles": "random"})
    ps = HM.parse_pauli_sum("0.7*Z0*Z1 - 0.4*X1 + 0.25*Y0*Y2 + 0.1", 3)
    a = T.statevector(3, tpl.spec.ops)
    exact = float(np.real(np.vdot(a, ps.matrix() @ a)))
    comp = CP.build(tpl.spec, opts=opts)
    ideal, outs, _ = compare(comp, PRESETS["Ideal (no noise)"], ps, [preset("Unmitigated"), preset("ZNE · linear")])
    assert ideal == pytest.approx(exact, abs=1e-9)
    assert all(o.result.mean == pytest.approx(exact, abs=1e-9) for o in outs)


def test_state_file_hamiltonian_is_validated():
    import streamlit  # noqa: F401
    from ui.state import validate_state
    good = dict(circuit=dict(n=2, ops=[]), observable=dict(pauli="Z0*Z1 + 0.5*X0"))
    validate_state(good)
    with pytest.raises(ValueError):
        validate_state(dict(circuit=dict(n=2, ops=[]), observable=dict(pauli="__import__('os')")))


# ------------------------------------------------------------------ 6. lattices, compile stages, diff
@pytest.mark.parametrize("n", [2, 5, 12, 13, 30, 64])
def test_heavy_hex_lattice_is_heavy_hex(n):
    size, edges = CP.heavy_hex_lattice(*CP.heavy_hex_shape(n))
    assert size >= n
    deg = np.zeros(size, int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    assert deg.max() <= 3 and deg.min() >= 1
    import rustworkx as rx
    g = rx.PyGraph()
    g.add_nodes_from(range(size))
    g.add_edges_from_no_data(edges)
    assert rx.is_connected(g)
    pos = TOPO.coordinates(size, edges, "heavy-hex")
    assert len({(round(x, 6), round(y, 6)) for x, y in pos.values()}) == size


@pytest.mark.parametrize("coupling", ["line", "ring", "grid", "heavy-hex"])
def test_lattice_compilation_is_exact_and_stages_match(coupling):
    tpl = T.build("qft", {"n": 4})
    o = CP.CompileOptions(mode="isa", coupling=coupling, basis_2q="cz")
    comp = CP.build(tpl.spec, opts=o)
    ideal, outs, _ = compare(comp, PRESETS["Ideal (no noise)"], tpl.observable, [preset("Unmitigated")])
    assert outs[0].result.mean == pytest.approx(ideal, abs=1e-9) == pytest.approx(1.0)
    st = STG.compile_stages(tpl.spec, opts=o)
    phys = comp.info.get("phys") or list(range(comp.implemented.n))
    assert [st.final_layout[q] for q in range(4)] == [phys[comp.measure_map[q]] for q in range(4)]
    for a, b in st.swap_edges:                                  # SWAPs only on couplers
        assert (a, b) in {tuple(sorted(e)) for e in CP.coupling_map(coupling, 4).get_edges()}


def test_decomposition_table_names_native_gates():
    st = STG.compile_stages(T.bell().spec, opts=CP.CompileOptions(mode="isa", coupling="line", basis_2q="cz"))
    assert st.decomposition["H"] == "RZ(π/2) · √X · RZ(π/2)"
    assert "CZ" in st.decomposition["CX"]
    st0 = STG.compile_stages(T.bell().spec, opts=ABSTRACT)
    assert st0.n_swaps == 0 and "Abstract" in st0.note


# ------------------------------------------------------------------ 7. topology and noise-aware layout
def test_layout_search_finds_a_better_equivalent_placement():
    tpl = T.build("ghz", {"n": 4})
    o = CP.CompileOptions(mode="isa", device="fake_fez")
    comp = CP.build(tpl.spec, opts=o)
    res = LS.rank_layouts(comp, "fake_fez")
    Fs = [r["F"] for r in res["rows"]]
    assert Fs == sorted(Fs, reverse=True) and Fs[0] >= res["current"] - 1e-12
    best = res["rows"][0]
    comp2 = CP.build(tpl.spec, opts=dataclasses.replace(o, initial_layout=tuple(best["initial_layout"])))
    assert sorted(comp2.info["phys"]) == best["qubits"]
    assert LS.score(comp2, "fake_fez") == pytest.approx(best["F"], rel=1e-9)
    assert comp2.implemented.count("2q") == comp.implemented.count("2q")


def test_device_graph_has_distinct_positions():
    di = LS.device_info("fake_fez")
    pos = TOPO.coordinates(di.n, di.edges)
    assert len({(round(x, 6), round(y, 6)) for x, y in pos.values()}) == di.n
    fig = TOPO.figure(di.n, di.edges, used={3: "q0"}, node_err=di.readout, edge_err=di.e2,
                      broken_nodes=di.broken_nodes, broken_edges=di.broken_edges)
    assert fig.layout.height >= 260


# ------------------------------------------------------------------ 8. live variational parameters
def test_qaoa_parameters_rewrite_the_circuit_and_are_periodic():
    tpl = T.build("qaoa", {"n": 4, "graph": "ring", "p": 1})
    ps = tpl.variational
    assert [p["name"] for p in ps] == ["γ1", "β1"]
    mk = lambda s: CP.build(s)                                   # noqa: E731
    g, b = ps[0]["value"], ps[1]["value"]
    e0 = V.energy(mk, tpl.spec, ps, [g, b], PRESETS["Ideal (no noise)"], tpl.observable)[0]
    assert e0 == pytest.approx(0.75, abs=1e-6)
    assert V.energy(mk, tpl.spec, ps, [g + 2 * math.pi, b], PRESETS["Ideal (no noise)"], tpl.observable)[0] == \
        pytest.approx(e0, abs=1e-9)
    assert V.energy(mk, tpl.spec, ps, [g, b + math.pi / 2], PRESETS["Ideal (no noise)"], tpl.observable)[0] == \
        pytest.approx(e0, abs=1e-9)
    moved = V.apply(tpl.spec, ps, [0.3, 0.2])
    assert {o.params for o in moved.ops if o.name == "rzz"} == {(-0.3,)}
    assert {o.params for o in moved.ops if o.name == "rx"} == {(0.4,)}


def test_vqe_parameters_and_scan():
    tpl = T.build("hea_vqe", {"n": 3, "layers": 1})
    ps = tpl.variational
    assert len(ps) == 2 * 3 * 2 and V.valid(tpl.spec, ps)
    vals = [p["value"] for p in ps]
    mk = lambda s: CP.build(s)                                   # noqa: E731
    res = V.scan(mk, tpl.spec, ps, vals, PRESETS["Depolarizing"], tpl.observable, [(0, np.linspace(0, 2 * np.pi, 7))])
    assert res["ideal"].shape == (7,) and np.all(res["noisy"] >= res["ideal"] - 1e-9)   # noise lifts a minimum
    assert not V.valid(tpl.spec.replace(ops=tpl.spec.ops[:3]), ps)


# ------------------------------------------------------------------ 9. batch runs
def test_batch_plan_and_run_isolate_failures():
    sc = B.plan([B.CURRENT, "ghz"], ["Depolarizing", "Coherent over-rotation"], [1.0, 2.0], ["numpy", "fake_perth"])
    assert len(sc) == 2 * (2 * 2 + 1)                            # a device runs once per algorithm
    recipes = [preset("Unmitigated"), preset("ZNE · Richardson")]
    rows = B.run(sc, recipes, current=None, opts=ABSTRACT, shots=4096, t_shot_us=250.0)
    assert len(rows) == len(sc) * len(recipes)
    cur = [r for r in rows if r["algorithm"].startswith("Playground")]
    assert cur and all(r["status"].startswith("failed") for r in cur)           # no current circuit: reported
    ghz = [r for r in rows if r["algorithm"] == "GHZ state"]
    assert all(not r["status"] for r in ghz) and all(np.isfinite(r["RMSE"]) for r in ghz)
    raw1 = next(r for r in ghz if r["method"] == "Unmitigated" and r["noise"] == "Depolarizing" and r["λ"] == 1.0)
    raw2 = next(r for r in ghz if r["method"] == "Unmitigated" and r["noise"] == "Depolarizing" and r["λ"] == 2.0)
    assert abs(raw2["bias"]) > abs(raw1["bias"])
    w = B.winners(rows, 10 ** 6)
    assert len(w) == len({B.scenario_key(r) for r in ghz})
    assert np.allclose(B.rmse_at(ghz, 4096), [r["RMSE"] for r in ghz], rtol=1e-9)


def test_batch_charts_build():
    from qemx.viz import batch_charts as BC
    sc = B.plan(["ghz", "bell"], ["Depolarizing"], [0.5, 1.0], ["numpy"])
    rows = B.run(sc, [preset("Unmitigated"), preset("PEC · inverse")], current=None, opts=ABSTRACT, shots=1000,
                 t_shot_us=250.0)
    for f in (BC.heatmap(rows, 1000), BC.wins(rows, 1000), BC.versus_scale(rows, 1000, "Depolarizing"),
              BC.heatmap(rows, 1000, "estimate", dark=True)):
        assert f.to_dict()["data"]


# ------------------------------------------------------------------ 10. Learn
def test_every_guide_and_walkthrough_setup_is_valid():
    for g in L.GUIDES:
        s = g["setup"]
        for r in s.get("recipes", []) + s.get("batch", {}).get("methods", []):
            assert r in RECIPE_PRESETS, (g["key"], r)
        for nz in [s.get("noise")] + s.get("batch", {}).get("noises", []):
            assert nz is None or nz in PRESETS, (g["key"], nz)
        if "template" in s:
            T.build(s["template"][0], s["template"][1])
        assert g["steps"] and g["check"]
    for key, lst in L.ALGORITHM_SETUPS.items():
        assert key in L.CHAPTERS
        for _, s in lst:
            T.build(s["template"][0], s["template"][1])
    for k in L.CHAPTERS:
        text = L.chapter(k)
        assert text.startswith("# ") and text.count("$") % 2 == 0, k      # balanced inline math
    md = L.guides_markdown()
    assert md.count("## ") == len(L.GUIDES)


# ------------------------------------------------------------------ 11. export with lattices and Hamiltonians
def test_export_script_reproduces_a_heavy_hex_hamiltonian_run(tmp_path):
    import streamlit  # noqa: F401
    tpl = T.build("ghz", {"n": 3})
    ham = "0.5*Z0*Z1 + 0.3*X0*X1*X2 - 0.2*Y0*Y1"
    ps = HM.parse_pauli_sum(ham, 3)
    recipes = [preset("Unmitigated"), preset("ZNE · Richardson")]
    o = CP.CompileOptions(mode="isa", coupling="heavy-hex", basis_2q="cz")
    comp = CP.build(tpl.spec, opts=o)
    _, outs, _ = compare(comp, PRESETS["Depolarizing"], ps, recipes)
    state = dict(app="QEM Explorer", version="1.1.0", circuit=tpl.spec.to_json(), inputs=["0"] * 3, basis="ZZZ",
                 relabel=False, compile=dict(mode="isa", basis_2q="cz", coupling="heavy-hex", layout="sabre",
                                             opt_level=1, seed=11, initial_layout="0, 1, 99"),
                 noise=dataclasses.asdict(PRESETS["Depolarizing"]), observable=dict(pauli=ham, n=3), symmetry=None,
                 recipes=[r.to_json() for r in recipes], backend="numpy", shots=4096, t_shot_us=250.0)
    src = EX.script(state, {x.title: x.result.mean for x in outs})
    p = tmp_path / "export.py"
    p.write_text(src, encoding="utf-8")
    out = subprocess.run([sys.executable, str(p), "--numpy"], capture_output=True, text=True, timeout=600,
                         cwd=tmp_path, encoding="utf-8", env=child_env())
    assert out.returncode == 0, out.stderr[-2000:]
    assert sum(ln.rstrip().endswith("✓") for ln in out.stdout.splitlines()) == len(recipes), out.stdout[-2000:]
    assert "0, 1, 99" in src                        # the out-of-range layout travels in the state and is ignored


# ------------------------------------------------------------------ 12. the new UI, end to end (AppTest)
def _app():
    import qiskit_ibm_runtime.fake_provider  # noqa: F401  (see test_ui.py)
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _tab(at, name, page="playground", key="w_pg_tabs"):
    at.session_state[key] = name
    at.session_state["page_tab"] = {page: name}
    at.run()
    _ok(at)


@pytest.mark.slow
def test_topology_tab_layout_search_and_diff_toggle():
    at = _app()
    at.switch_page("pages/1_Playground.py").run()
    _tab(at, "Topology & layout")
    next(b for b in at.button if b.key == "topo_try_grid").click().run()
    _ok(at)
    assert at.session_state["compile"]["coupling"] == "grid" and at.session_state["compile"]["mode"] == "isa"
    assert any(m.label == "routing SWAPs" for m in at.metric)
    at.session_state["backend"] = "fake_fez"
    at.run()
    next(b for b in at.button if b.key == "topo_rank").click().run()
    _ok(at)
    F0 = float(next(m.value for m in at.metric if m.label == "success F"))
    next(b for b in at.button if b.key == "topo_use").click().run()
    _ok(at)
    F1 = float(next(m.value for m in at.metric if m.label == "success F"))
    assert at.session_state["compile"]["initial_layout"] and F1 >= F0
    _tab(at, "Build")
    at.toggle(key="w_diff").set_value(True).run()
    for v in ("aligned", "routed", "native", "executed"):
        at.session_state["w_diff_view"] = v
        at.run()
        _ok(at)
    # a layout that no longer fits (a larger circuit) is ignored with a warning, never a crash
    at.session_state["n"] = 4
    at.session_state["ops"] = [dict(name="h", qubits=[0], params=[], kind="gate", tag="")]
    at.session_state["inputs"], at.session_state["basis"] = ["0"] * 4, "ZZZZ"
    at.run()
    _ok(at)
    assert any("forced layout is ignored" in w.value for w in at.warning)


@pytest.mark.slow
def test_live_variational_sliders_and_landscape():
    at = _app()
    at.switch_page("pages/1_Playground.py").run()
    at.selectbox(key="w_tpl_cat").set_value(T.category_of("qaoa")).run()
    at.selectbox(key="w_tpl").set_value("qaoa").run()
    next(b for b in at.button if b.key == "tpl_load").click().run()
    _tab(at, "Time travel")
    vals = {m.label: m.value for m in at.metric}
    assert vals["⟨O⟩ ideal"] == "0.7500"
    at.slider(key="w_var_0").set_value(0.3).run()
    _ok(at)
    assert {m.label: m.value for m in at.metric}["⟨O⟩ ideal"] != "0.7500"
    at.session_state["w_var_scan_pts"] = 7
    at.run()
    next(b for b in at.button if b.key == "var_scan_btn").click().run()
    _ok(at)
    assert at.session_state["var_scan"]["res"]["ideal"].shape == (7, 7)
    next(b for b in at.button if b.key == "var_apply").click().run()
    _ok(at)
    assert at.session_state["var_spec"]["params"][0]["value"] == pytest.approx(0.3)


@pytest.mark.slow
def test_batch_page_and_learn_setups():
    at = _app()
    at.switch_page("pages/9_Learn.py").run()
    _ok(at)
    for t in ("Fundamentals", "Algorithm walkthroughs", "Research topics", "Hands-on guides"):
        _tab(at, t, "learn", "w_learn_tabs")
    next(b for b in at.button if b.key == "guide_batch").click().run()
    _ok(at)
    at.switch_page("pages/8_Batch_Run.py").run()
    _ok(at)
    at.session_state["w_b_scales"] = [1.0]
    at.session_state["w_b_algos"] = [B.CURRENT, "ghz"]
    at.run()
    next(b for b in at.button if b.key == "b_run").click().run()
    _ok(at)
    res = at.session_state["batch_result"]
    assert len(res["rows"]) == 2 * 3 * 4 and not any(r["status"].startswith("failed") for r in res["rows"])
    at.session_state["w_b_metric"] = "|bias|"
    at.run()
    _ok(at)
    at.switch_page("pages/9_Learn.py").run()
    _tab(at, "Hands-on guides", "learn", "w_learn_tabs")
    next(b for b in at.button if b.key == "guide_dd_crosstalk").click().run()
    _ok(at)
    assert at.session_state["template_name"] == "Spectator Ramsey"
    at.switch_page("pages/1_Playground.py")
    at.session_state["run_requested"] = True
    at.run()
    _ok(at)
    est = {r["method"]: r["estimate"] for r in at.session_state["results"]["rows"]}
    assert est["DD-XY4"] > est["Unmitigated"] + 0.05


# ------------------------------------------------------------------ 13. fixes from the Sonnet QA pass
def test_batch_symmetry_follows_each_algorithm():
    sc = B.plan(["steane"], ["Depolarizing"], [1.0], ["numpy"])
    rows = B.run(sc, ["Unmitigated", "Symmetry verification"], current=None, opts=ABSTRACT, shots=4096,
                 t_shot_us=250.0)
    sym = next(r for r in rows if r["family"] == "symmetry")
    raw = next(r for r in rows if r["method"] == "Unmitigated")
    assert "3" in sym["method"] and "6" in sym["method"]              # the Steane check Z3Z4Z5Z6, not Z0Z1
    assert abs(sym["bias"]) < abs(raw["bias"])


def test_batch_runs_the_current_circuit_on_every_simulator():
    tpl = T.build("ghz", {"n": 3})
    cur = dict(spec=tpl.spec, inputs=["0"] * 3, basis="ZZZ", observable=tpl.observable, symmetry=None, relabel=False,
               name="GHZ")
    sc = B.plan([B.CURRENT], ["Depolarizing"], [1.0], ["numpy", "fake_perth"])
    rows = B.run(sc, ["Unmitigated"], current=cur, opts=ABSTRACT, shots=1000, t_shot_us=250.0)
    assert len(rows) == 2 and not any(r["status"] for r in rows)
    assert B.winners([dict(rows[0], bias=float("nan"), v=float("nan"))], 100) == []


def test_comments_may_contain_semicolons():
    sp = parse("h 0 # prepare |+>; cx 0 1\ncx 0 1; x 2 # note; then", 3)
    assert [o.name for o in sp.ops] == ["h", "cx", "x"]


@pytest.mark.parametrize("text,value", [("Z0\nZ1", None), ("1E-3*Z0", 0.001), ("2**-3*Z0", 0.125), ("2*-Z0", -2.0),
                                        ("Z0*-1", -1.0), ("Z0/2", 0.5), ("0.5 Z0 Z1", 0.5), ("2Z0", 2.0)])
def test_hamiltonian_edge_cases_read_correctly(text, value):
    ps = HM.parse_pauli_sum(text, 2)
    if value is None:
        assert len(ps.terms) == 2                                    # one term per line
    else:
        assert ps.terms[0][0] == pytest.approx(value)


@pytest.mark.parametrize("text", ["Z0 2", "Z1.5", "2*Z0**2", "2**Z0*3", "1/Z0*2"])
def test_hamiltonian_ambiguous_forms_are_rejected(text):
    with pytest.raises(HM.HamiltonianError):
        HM.parse_pauli_sum(text, 2)


def test_state_files_check_noise_arity_and_layouts():
    import streamlit  # noqa: F401
    from ui.state import parse_layout, validate_state
    bad = dict(circuit=dict(n=2, ops=[dict(name="ampdamp", qubits=[0, 1], params=[0.1], kind="noise")]),
               observable=dict(values=[1, -1, -1, 1]))
    with pytest.raises(ValueError):
        validate_state(bad)
    ok = dict(circuit=dict(n=2, ops=[dict(name="depol", qubits=[0, 1], params=[0.1], kind="noise")]),
              observable=dict(values=[1, -1, -1, 1]))
    validate_state(ok)
    for t in ("²", "①", "9" * 5000):
        with pytest.raises(ValueError, match="non-negative integers|too long"):
            parse_layout(t)
    assert parse_layout("3, 4 5") == [3, 4, 5]


def test_bloch_titles_use_two_lines_from_three_per_row():
    r = np.tile([0.0, 0.0, 1.0], (4, 1))
    fig = SV.bloch_row(r, r)
    assert all("<br>" in a.text for a in fig.layout.annotations if a.text and a.text.startswith("<b>q"))
    fig2 = SV.bloch_row(r[:2], r[:2])
    assert not any("<br>" in a.text for a in fig2.layout.annotations if a.text and a.text.startswith("<b>q"))
