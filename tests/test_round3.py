"""Round 3: gate-only time-travel cursor, the expanded template library, the adaptive learning training set,
the Jupyter-safe export, and the hardening found by the QA pass."""
import dataclasses
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from qemx import compile as CP
from qemx import export as EX
from qemx import observables as OBS
from qemx import templates as T
from qemx.circuit import CircuitSpec, eval_expr, parse
from qemx.methods import Recipe
from qemx.noise import PRESETS
from qemx.recipes import compare, preset, summary
from qemx.viz import composer as CV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTRACT = CP.CompileOptions(mode="abstract", basis_2q="cx", coupling="all", layout="sabre", opt_level=1)


def ideal_value(tpl):
    a = T.statevector(tpl.spec.n, tpl.spec.ops)
    return float(np.vdot(a, tpl.observable.o * a).real)


# ------------------------------------------------------------------ 1. time-travel cursor
def _cursor_rects(fig):
    return [s for s in fig.layout.shapes if s.type == "rect" and s.layer == "above"]


def test_cursor_boxes_only_the_gate_and_its_wires():
    sp = parse("h 0; cx 0 2; x 1; cz 0 1", 3)
    fig = CV.draw(sp, cursor_op=1)                  # cx 0 2 skips q1
    rects = _cursor_rects(fig)
    assert len(rects) == 2                          # one box on q0, one on q2, none on q1
    spans = sorted((r.y0, r.y1) for r in rects)
    assert spans[0][0] < 0 < spans[0][1] and spans[1][0] < 2 < spans[1][1]
    assert all(not (r.y0 < 1 < r.y1) for r in rects)
    fig = CV.draw(sp, cursor_op=3)                  # cz 0 1: neighboring qubits share one box
    rects = _cursor_rects(fig)
    assert len(rects) == 1 and rects[0].y0 < 0 and rects[0].y1 > 1 and rects[0].y1 < 2
    fig = CV.draw(sp, cursor_op=2)                  # x 1: a single box on q1
    (r,) = _cursor_rects(fig)
    assert r.y0 < 1 < r.y1 and r.y0 > 0 and r.y1 < 2


def test_long_circuits_are_drawn_in_a_window():
    big = parse("\n".join("h %d; cx %d %d" % (i % 5, i % 5, (i + 1) % 5) for i in range(3000)), 5)
    fig = CV.draw(big, cursor_op=4000, window=28)
    assert len(fig.layout.shapes) < 3000
    lo, hi = fig.layout.xaxis.range
    col = CV.column_of(big)[4000]
    assert lo <= col <= hi
    assert any("around the cursor" in (a.text or "") for a in fig.layout.annotations)


# ------------------------------------------------------------------ 2. templates
def test_every_template_is_in_one_category_and_builds():
    keys = [k for ks in T.CATEGORIES.values() for k in ks]
    assert sorted(keys) == sorted(T.REGISTRY) and len(keys) == len(set(keys))
    for k in T.REGISTRY:
        tpl = T.build(k)
        assert tpl.spec.n == tpl.observable.n
        assert all(o.kind in ("gate", "noise", "barrier") for o in tpl.spec.ops)


def test_statevector_matches_qiskit():
    from qiskit.quantum_info import Statevector
    from qemx.circuit import to_qiskit
    for k, p in (("qml_zz", {}), ("qaoa", dict(graph="3-regular", n=6, p=2)), ("steane", {}), ("qec_3q", {}),
                 ("hea_vqe", dict(angles="random", entanglement="full"))):
        tpl = T.build(k, p)
        sp = CircuitSpec(tpl.spec.n, tuple(o for o in tpl.spec.ops if o.kind == "gate"), "x")
        assert abs(abs(np.vdot(Statevector(to_qiskit(sp)).data, T.statevector(sp.n, sp.ops))) - 1) < 1e-9


def test_vqe_and_qaoa_reach_their_optima():
    assert ideal_value(T.build("hea_vqe", dict(n=4, layers=2))) < -0.999
    assert abs(ideal_value(T.build("qaoa", dict(n=4, graph="ring", p=1))) - 0.75) < 1e-3     # known p = 1 optimum
    assert ideal_value(T.build("qaoa", dict(n=6, graph="3-regular", p=2))) > 0.85
    with pytest.raises(ValueError):
        T.build("qaoa", dict(n=5, graph="3-regular"))


@pytest.mark.parametrize("code", ["bit-flip", "phase-flip"])
def test_three_qubit_codes_correct_one_error(code):
    for k in range(3):
        tpl = T.build("qec_3q", dict(code=code, mode="syndrome + correction", error="one error on qubit k", k=k))
        assert abs(ideal_value(tpl) - math.cos(1.0472)) < 1e-6
        tpl = T.build("qec_3q", dict(code=code, mode="majority vote readout", error="one error on qubit k", k=k))
        assert abs(ideal_value(tpl) - math.cos(1.0472)) < 1e-6
    tpl = T.build("qec_3q", dict(code=code, mode="syndrome only (no correction)", error="one error on qubit k", k=0))
    assert abs(ideal_value(tpl) + math.cos(1.0472)) < 1e-6


def test_steane_code_space_and_decoder():
    from qiskit.quantum_info import Pauli, Statevector
    tpl = T.build("steane", dict(theta=0.7))
    sv = Statevector(T.statevector(7, tpl.spec.ops))
    for supp in ([3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]):
        for P in "XZ":
            lab = "".join(P if (6 - i) in supp else "I" for i in range(7))
            assert abs(sv.expectation_value(Pauli(lab)).real - 1) < 1e-9
    for k in range(7):
        dec = T.build("steane", dict(error="X on qubit k", k=k, readout="decoded"))
        raw = T.build("steane", dict(error="X on qubit k", k=k, readout="raw parity"))
        c = math.cos(1.0472)
        assert abs(ideal_value(dec) - c) < 1e-9 and abs(ideal_value(raw) + c) < 1e-9
    assert T.build("steane").symmetry == ((3, 4, 5, 6), +1)


def test_repetition_code_and_deep_shor():
    tpl = T.build("repetition", dict(d=5, rounds=1, theta=0.5, p=0.0))
    assert tpl.spec.n == 9 and abs(ideal_value(tpl) - math.cos(0.5)) < 1e-9
    base = T.build("shor")
    deep = T.build("shor", dict(powers="repeated", multiplier="permutation"))
    assert deep.spec.count("2q") > 3 * base.spec.count("2q")
    assert abs(ideal_value(deep) - 0.75) < 1e-9 and abs(ideal_value(base) - 0.75) < 1e-9


@pytest.mark.slow
def test_synthesised_shor_is_deep_and_correct():
    tpl = T.build("shor", dict(multiplier="synthesized"))
    assert tpl.spec.count("2q") > 300 and abs(ideal_value(tpl) - 0.75) < 1e-9


def test_new_templates_run_through_the_engine():
    for k in ("bv", "qml_reupload", "qec_3q", "steane"):
        tpl = T.build(k)
        comp = CP.build(tpl.spec, opts=ABSTRACT)
        ideal, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, [preset("Unmitigated")])
        assert abs(ideal - ideal_value(tpl)) < 1e-9 and outs[0].result is not None


# ------------------------------------------------------------------ 3. learning: adaptive training set
def test_adaptive_training_set_avoids_collapse():
    tpl = T.build("qpe")
    comp = CP.build(tpl.spec, opts=ABSTRACT)
    ob = OBS.z(0, tpl.spec.n)                     # blind to near-Clifford replacements of this circuit
    ideal, outs, _ = compare(comp, PRESETS["Depolarizing"], ob,
                             [Recipe.make("learning", strategy="fixed"), Recipe.make("learning")])
    fixed, adaptive = (o.result.details for o in outs)
    assert fixed["degenerate"] and fixed["spread"] < 1e-9
    assert not adaptive["degenerate"] and adaptive["spread"] >= adaptive["spread_needed"]
    assert len(adaptive["stages"]) > 1
    assert abs(outs[1].result.mean - ideal) < abs(outs[0].result.mean - ideal) + 1e-9


def test_keep_is_honoured_and_titles_differ():
    r0, r2 = Recipe.make("learning"), Recipe.make("learning", keep=2, strategy="fixed")
    assert r0.title() != r2.title() and "keep 2" in r2.title()
    tpl = T.build("ising")
    comp = CP.build(tpl.spec, opts=ABSTRACT)
    _, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, [r2])
    assert outs[0].result.details["keep_used"] == 2


def test_degenerate_training_data_are_flagged_not_plotted_as_a_constant():
    sp = parse("h 0; t 0; h 1", 2)                 # H then T (or I, S) on |0⟩ or |1⟩: ⟨Z0⟩ = 0 for every training circuit
    comp = CP.build(sp, opts=CP.CompileOptions(mode="abstract", basis_2q="cx", coupling="all", layout="sabre",
                                               opt_level=0))
    ideal, outs, _ = compare(comp, PRESETS["Depolarizing"], OBS.z(0, 2), [Recipe.make("learning"), preset("Unmitigated")])
    d = outs[0].result.details
    assert d["degenerate"] and "cannot train" in d["warning"]
    assert outs[0].result.mean == pytest.approx(outs[1].result.mean)        # falls back to the unmitigated value
    row = summary(outs[0], ideal, 4096, 250.0)
    assert row["status"].startswith("not applied")
    rng = np.random.default_rng(0)
    s = outs[0].result.sampler(4096, 20, rng)
    assert np.std(s) > 0                                                    # shot noise, not a flat line


def test_identical_contenders_get_distinct_titles():
    tpl = T.bell()
    comp = CP.build(tpl.spec, opts=ABSTRACT)
    _, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, [preset("Unmitigated"), preset("Unmitigated")])
    assert outs[0].title != outs[1].title


# ------------------------------------------------------------------ 4. export: terminal and Jupyter
def _state(tpl, recipes):
    return dict(app="QEM Explorer", version="1.0.0", circuit=tpl.spec.to_json(), inputs=["0"] * tpl.spec.n,
                basis="Z" * tpl.spec.n, relabel=tpl.relabel,
                compile=dict(mode="abstract", basis_2q="cx", coupling="all", layout="sabre", opt_level=1),
                noise=dataclasses.asdict(PRESETS["Depolarizing"]),
                observable=dict(key=tpl.observable.key, label=tpl.observable.label, tex=tpl.observable.tex,
                                values=list(tpl.observable.values)), symmetry=None,
                recipes=[r.to_json() for r in recipes], backend="numpy", shots=4096, t_shot_us=250.0)


def test_export_has_dual_entry_point():
    from qemx import __version__
    src = EX.script(_state(T.bell(), [preset("Unmitigated")]))
    assert "def main(argv=None):" in src and "_in_ipython()" in src and "main([])" in src
    assert "sys.exit(main())" in src and "QEM Explorer %s" % __version__ in src
    compile(src, "export", "exec")


@pytest.mark.slow
def test_export_runs_in_terminal_and_in_a_jupyter_kernel(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    tpl = T.build("steane")
    recipes = [preset("Unmitigated"), Recipe.make("learning")]
    comp = CP.build(tpl.spec, opts=ABSTRACT)
    _, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, recipes)
    src = EX.script(_state(tpl, recipes), {o.title: o.result.mean for o in outs})
    p = tmp_path / "export.py"
    p.write_text(src)
    out = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert sum(ln.rstrip().endswith("✓") for ln in out.stdout.splitlines()) == len(recipes)
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(src),
                                         nbformat.v4.new_code_cell('rc = main(["--numpy"]); print("RC", rc)')])
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    text = "".join(o.get("text", "") for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "stream")
    errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors
    assert text.count("✓") == 2 * len(recipes) and "RC 0" in text


# ------------------------------------------------------------------ 5. hardening from the QA pass
def test_expression_errors_are_readable():
    for bad, frag in (("pi/", "not a valid number"), ("1/0", "division by zero"), ("sqrt(-1)", "math domain"),
                      ("__import__('os')", "not a number")):
        with pytest.raises(ValueError) as e:
            eval_expr(bad)
        assert frag in str(e.value)


def test_md_escape_neutralises_markdown():
    import streamlit  # noqa: F401
    from ui.widgets import md_escape, plural
    s = md_escape("[x](http://e.vil) **b** $x$ <img>")
    assert "](" not in s.replace("\\]\\(", "") and "\\$" in s and "\\<" in s
    assert plural(1, "qubit") == "1 qubit" and plural(3, "qubit") == "3 qubits"


def test_validate_state_rejects_malformed_files_without_crashing():
    import streamlit  # noqa: F401
    from ui.state import validate_state
    good = dict(circuit=dict(n=1, ops=[]), observable=dict(values=[1, -1]))
    validate_state(good)
    for bad in ({"circuit": {"n": 1e999, "ops": []}, "observable": {"values": [1, -1]}},
                {"circuit": {"n": 10 ** 19, "ops": []}},
                dict(good, shots=float("inf")), dict(good, compile=[1]), dict(good, noise=3),
                dict(good, compile={"opt_level": 9}), dict(good, recipes={"a": 1}),
                {"circuit": {"n": 2, "ops": []}, "observable": {"values": [1, -1]}}, "junk", [], None):
        with pytest.raises(ValueError):
            validate_state(bad)


def test_state_values_are_bounded():
    import streamlit  # noqa: F401
    from ui.state import validate_state
    good = dict(circuit=dict(n=1, ops=[]), observable=dict(values=[1, -1]))
    for bad in (dict(good, noise={"p1": 5}), dict(good, noise={"ro01": -0.1}), dict(good, shots=2 ** 62),
                dict(good, recipes=[{"estimator": "learning", "params": {"n_train": 10 ** 9}}]),
                dict(good, recipes=[{"estimator": "learning", "params": {"min_spread": 7.0}}])):
        with pytest.raises(ValueError):
            validate_state(bad)
    ok = validate_state(dict(good, noise={"p1": 0.01, "over_rot": 0.05},
                             recipes=[{"estimator": "learning", "params": {"n_train": 200, "strategy": "fixed"}}]))
    assert ok["noise"]["p1"] == 0.01


def test_pec_wall_handles_full_depolarization():
    from qemx import analysis as A
    assert A.depol_gamma(1.0, 1) == float("inf") and np.isfinite(A.depol_gamma(0.75, 1))
    w = A.pec_wall(np.array([0, 1, 5]), [0.01], 4, 1.0, 0.01, 250.0)[0.01]
    assert w["gamma"][0] == 1.0 and np.isinf(w["gamma"][1])


def test_repetition_logical_error_matches_simulation():
    tpl = T.build("repetition", dict(d=3, rounds=1, p=0.05))
    comp = CP.build(tpl.spec, opts=CP.CompileOptions(mode="abstract", basis_2q="cx", coupling="all", layout="sabre",
                                                     opt_level=0))
    from qemx.noise import NoiseSpec
    _, outs, _ = compare(comp, NoiseSpec(), tpl.observable, [preset("Unmitigated")])
    peff = (1 - (1 - 2 * 0.05) ** 2) / 2
    pl = 3 * peff ** 2 - 2 * peff ** 3
    assert abs((1 - outs[0].result.mean) / 2 - pl) < 1e-9


def test_degenerate_learning_costs_nothing_extra():
    tpl = T.build("qaoa")
    comp = CP.build(tpl.spec, opts=ABSTRACT)
    _, outs, _ = compare(comp, PRESETS["Depolarizing"], OBS.z(0, tpl.spec.n), [Recipe.make("learning"),
                                                                              preset("Unmitigated")])
    cdr, raw = outs[0].result, outs[1].result
    assert cdr.details["degenerate"] and cdr.fixed_shots == raw.fixed_shots and cdr.circuits == raw.circuits
    assert cdr.var1 == pytest.approx(raw.var1) and cdr.gamma == pytest.approx(raw.gamma)
