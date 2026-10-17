"""Regression tests for the app audit (October 2026): shared caches keyed by what they hold, large registers and
huge template bounds never crash a page, callbacks read current widget values, recipes are identified by key or
position (never by title), method parameters are validated against their schema (including ``when``), the
hardware path keeps partial submissions and reports failed jobs, and the exported script is safe and collision-free."""
import ast
import json
import os
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:                                       # see tests/test_ui.py: import the runtime outside AppTest's thread
    import qiskit_ibm_runtime  # noqa: F401
    import qiskit_ibm_runtime.fake_provider  # noqa: F401
except Exception:                          # pragma: no cover
    pass


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _page_with_state(page, state=None, preset=None, timeout=600):
    """AppTest of one page after loading ``state`` through the app's own validator (the Export tab's path)."""
    from streamlit.testing.v1 import AppTest

    def script(root, page, state, preset):
        import os
        import sys
        if root not in sys.path:
            sys.path.insert(0, root)
        import streamlit as st
        from ui import state as S
        S.init()
        ss = st.session_state
        if state is not None and not ss.get("_t_loaded"):
            S.from_json(state)
            ss._t_loaded = True
        if preset and not ss.get("_t_preset"):
            for k, v in preset.items():
                ss[k] = v
            ss._t_preset = True
        path = os.path.join(root, page)
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), {"__name__": "__main__", "__file__": path})

    return AppTest.from_function(script, default_timeout=timeout,
                                 kwargs=dict(root=ROOT, page=page, state=state, preset=preset))


def _ghz_state(n, pauli="Z0*Z1"):
    ops = [{"name": "h", "qubits": [0], "params": []}] + [{"name": "cx", "qubits": [i, i + 1], "params": []}
                                                          for i in range(n - 1)]
    return {"circuit": {"n": n, "name": "ghz%d" % n, "ops": ops}, "observable": {"pauli": pauli},
            "recipes": [{"estimator": "raw"}]}


# ------------------------------------------------------------------ 1. shared time-travel cache
def test_timeline_cache_key_includes_the_circuit():
    from qemx import compile as CP
    from qemx.circuit import parse
    from qemx.noise import NoiseSpec
    from ui import cache as K
    nj = K.noise_json(NoiseSpec(p2=0.02))
    a = CP.build(parse("h 0; cx 0 1; rx(0.123) 1; cz 0 1", 2), None, None, CP.CompileOptions())
    b = CP.build(parse("h 0; cx 0 1", 2), None, None, CP.CompileOptions())
    ta = K.timeline("same-signature", a, nj, "implemented", 3.0, "global", "X2", "numpy")
    tb = K.timeline("same-signature", b, nj, "implemented", 3.0, "global", "X2", "numpy")
    assert ta is not tb and [o.name for o in tb.ops] != [o.name for o in ta.ops]


@pytest.mark.slow
def test_time_travel_never_pairs_an_old_signature_with_a_new_circuit():
    """Run, edit the circuit, open Time travel on another kind: every signature seen by the shared cache maps to
    one circuit (before the fix the last run's signature was reused with the edited circuit)."""
    from ui import cache as K
    seen, orig = {}, K._timeline

    def recorder(sig, circuit, *a, **kw):
        seen.setdefault(sig, set()).add(circuit)
        return orig(sig, circuit, *a, **kw)
    K._timeline = recorder
    try:
        at = _page_with_state("pages/1_Playground.py", preset={"recipe_names": ["Unmitigated"],
                                                               "w_rc_pick": ["Unmitigated"]})
        at.run()
        at.session_state["run_requested"] = True
        at.run()
        _ok(at)
        ops = list(at.session_state["ops"]) + [dict(name="rx", qubits=[0], params=[0.321], kind="gate", tag="")]
        at.session_state["ops"] = ops
        at.session_state["ops_version"] = at.session_state["ops_version"] + 1
        at.session_state["w_pg_tabs"] = "Time travel"
        at.session_state["w_tt_kind"] = "zne"
        at.run()
        _ok(at)
    finally:
        K._timeline = orig
    assert len(seen) >= 2 and all(len(v) == 1 for v in seen.values()), seen


# ------------------------------------------------------------------ 2. large registers
def test_catalogue_is_lazy_and_matches_the_dense_values():
    from qemx import observables as OBS
    for n in (1, 2, 3, 6):
        b = OBS._bits(n)
        for qs in ([0], [n - 1], list(range(n))):
            old = np.prod(1 - 2 * b[:, qs], axis=1).astype(float)
            assert np.array_equal(np.asarray(OBS.z_string(qs, n).values), old)
        assert np.array_equal(np.asarray(OBS.magnetization(n).values), np.mean(1 - 2 * b, axis=1))
        assert list(OBS.catalogue(n)) == list(OBS.catalogue_labels(n))
    t = time.time()
    labels = OBS.catalogue_labels(64)                       # names only: instant for any n
    assert len(labels) == 2 * 64 + 1 and time.time() - t < 0.5
    big = OBS.catalogue_item(64, "Z0")                       # too large to store densely: the same Pauli sum
    assert big.n == 64 and hasattr(big, "terms")
    with pytest.raises(ValueError):
        OBS.catalogue_item(3, "nope")


@pytest.mark.slow
@pytest.mark.parametrize("page", ["pages/1_Playground.py", "pages/3_Bias_Variance.py"])
def test_a_validated_64_qubit_state_renders(page):
    at = _page_with_state(page, _ghz_state(64))
    at.run()
    _ok(at)


def test_set_circuit_is_atomic_when_the_resize_fails():
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st
        from qemx import observables as OBS
        from qemx.circuit import CircuitSpec
        from ui import state as S
        S.init()
        ss = st.session_state
        before = (ss.n, list(ss.inputs), ss.basis, len(ss.get("history", [])), ss.ops_version)
        orig = OBS.default_observable

        def boom(n):
            raise MemoryError("too large")
        OBS.default_observable = boom
        try:
            S.set_circuit(CircuitSpec(ss.n + 5, ()))
            ss.t_raised = False
        except MemoryError:
            ss.t_raised = True
        finally:
            OBS.default_observable = orig
        ss.t_same = before == (ss.n, list(ss.inputs), ss.basis, len(ss.get("history", [])), ss.ops_version)

    at = AppTest.from_function(script, default_timeout=120).run()
    _ok(at)
    assert at.session_state["t_raised"] and at.session_state["t_same"]


@pytest.mark.slow
def test_template_card_survives_bounds_beyond_javascript_numbers():
    at = _page_with_state("pages/1_Playground.py", preset={"w_tp_qft_n": 60, "w_tp_qft_s": 2 ** 59})
    at.run()
    _ok(at)
    s_box = next(x for x in at.number_input if x.key == "w_tp_qft_s")
    assert s_box.value <= 2 ** 53 - 1


# ------------------------------------------------------------------ 3. callbacks read current values
@pytest.mark.slow
def test_recipe_builder_adds_the_values_typed_just_before_the_click():
    at = _page_with_state("pages/1_Playground.py", preset={"w_pg_tabs": "Mitigate & compare"})
    at.run()
    _ok(at)
    box = next(t for t in at.text_input if t.key == "w_rb_zne_scales")
    box.input("1, 2, 3")
    next(b for b in at.button if b.key == "rb_add").click()
    at.run()
    _ok(at)
    added = at.session_state["custom_recipes"]
    assert added and added[-1]["estimator"] == "zne" and added[-1]["params"]["scales"] == [1.0, 2.0, 3.0]


# ------------------------------------------------------------------ 4. recipe identity (never the title)
def _two_pec():
    from qemx.methods import Recipe
    a, b = Recipe.make("pec", p2=0.01, p1=0.001), Recipe.make("pec", p2=0.03, p1=0.001)
    assert a.title() == b.title() and a.key() != b.key()
    return a, b


def test_export_matches_app_values_by_position():
    from qemx import export as EX
    a, b = _two_pec()
    state = dict(recipes=[a.to_json(), b.to_json()])
    assert EX.recipe_labels([a, b]) == ["PEC inverse", "PEC inverse (#2)"]
    assert EX.app_values(state, {"PEC inverse": 0.98, "PEC inverse (#2)": 1.01}) == [0.98, 1.01]
    assert EX.app_values(state, [0.98, None]) == [0.98, None]
    assert EX.app_values(state, None) == []


@pytest.mark.slow
def test_exported_script_with_two_same_title_contenders_reproduces_both(tmp_path):
    import dataclasses
    import subprocess
    import sys
    from qemx import compile as CP, export as EX, templates as T
    from qemx.noise import PRESETS
    from qemx.recipes import compare
    a, b = _two_pec()
    tpl = T.bell()
    comp = CP.build(tpl.spec, None, None, CP.CompileOptions())
    _, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, [a, b])
    state = dict(app="QEM Explorer", version="t", circuit=tpl.spec.to_json(), inputs=["0", "0"], basis="ZZ",
                 relabel=False, compile=dict(mode="abstract", basis_2q="cx", coupling="all", layout="sabre",
                                             opt_level=1), noise=dataclasses.asdict(PRESETS["Depolarizing"]),
                 observable=dict(key=tpl.observable.key, label=tpl.observable.label, tex=tpl.observable.tex,
                                 values=list(tpl.observable.values)), symmetry=None,
                 recipes=[a.to_json(), b.to_json()], backend="numpy", shots=4096, t_shot_us=250.0)
    p = tmp_path / "export.py"
    p.write_text(EX.script(state, {o.title: o.result.mean for o in outs}))
    out = subprocess.run([sys.executable, str(p), "--numpy"], capture_output=True, text=True, timeout=600,
                         cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert sum(ln.rstrip().endswith("✓") for ln in out.stdout.splitlines()) == 2, out.stdout


def test_custom_contenders_with_one_title_stay_apart():
    from ui import widgets as W
    a, b = _two_pec()
    opts = W.custom_options([a.to_json(), b.to_json(), a.to_json()])
    assert len(opts) == 2 and all(k.startswith("custom:") for k in opts)
    labels = list(W.custom_labels(opts).values())
    assert labels == ["Custom: PEC inverse", "Custom: PEC inverse (#2)"]


def test_hardware_plan_keeps_same_title_recipes_apart():
    from qemx import compile as CP, hardware as H, templates as T
    from qemx.methods import Recipe
    tpl = T.bell()
    comp = CP.build(tpl.spec, None, None, CP.CompileOptions())
    vd1, vd2 = Recipe.make("vd", M=2), Recipe.make("vd", M=2, )
    a, b = _two_pec()
    plan = H.make_plan(comp, tpl.observable, [vd1, a, b, vd2], 20000, None, 0, None, predict=False)
    assert set(plan.skipped_idx) == {0, 3} and {"VD M=2", "VD M=2 (#2)"} <= set(plan.skipped)
    assert [r["contender"] for r in plan.table()] == ["VD M=2", "PEC inverse", "PEC inverse (#2)", "VD M=2 (#2)"]
    assert {1, 2} <= set(plan.alloc)


# ------------------------------------------------------------------ 5. routing through unmeasured qubits
def test_forced_layout_through_extra_qubits_is_rejected_readably():
    from qemx import compile as CP, engine, templates as T
    tpl = T.bell()
    with pytest.raises(ValueError, match="not part"):
        CP.build(tpl.spec, ["0", "0"], "ZZ", CP.CompileOptions(mode="isa", device="fake_perth", initial_layout=(0, 3)))
    with pytest.raises(ValueError, match="measured exactly once"):
        engine.permute_probs(np.full(8, 1 / 8), {0: 1, 1: 0}, 3)


# ------------------------------------------------------------------ 6. parameters, `when`, strict text
def test_state_params_follow_the_schema():
    from ui import state as S
    S.validate_params("nil", {"train_shots": 200000})          # legal (hi 1e8), rejected before the fix
    for est, params in (("pec", {"instances": 5000}), ("learning", {"n_train": 5000}),
                        ("symmetry", {"qubits": [0.5]}), ("zne", {"scales": [1.0, float("nan")]})):
        with pytest.raises(ValueError):
            S.validate_params(est, params)
    base = {"circuit": {"n": 2, "ops": []}, "observable": {"values": [1, -1, -1, 1]}}
    S.validate_state(dict(base, recipes=[{"estimator": "nil", "params": {"train_shots": 200000}}]))
    with pytest.raises(ValueError):
        S.validate_state(dict(base, recipes=[{"estimator": "pec", "params": {"instances": 5000}}]))


class _Dummy:
    """A method whose controls depend on each other (Param.when)."""
    key, label, family, tex, why = "t_dummy", "Dummy", "raw", "", ""

    def __init__(self):
        from qemx.methods.base import Param
        self.params = (Param("mode", "mode", "choice", "a", ("a", "b")),
                       Param("k", "k", "int", 3, lo=1, hi=9, when=(("mode", ("b",)),)),
                       Param("x", "x", "float", 0.5, lo=0.0, hi=1.0, when=(("mode", ("b",)), ("k", (3, 4)))))

    def defaults(self):
        return {p.name: p.default for p in self.params}

    def hardware_supported(self, params):
        return (params.get("mode") == "a"), "mode b is simulation only"


@pytest.fixture
def dummy_method():
    from qemx.methods import REGISTRY
    REGISTRY["t_dummy"] = _Dummy()
    try:
        yield REGISTRY["t_dummy"]
    finally:
        REGISTRY.pop("t_dummy", None)


def test_when_conditions_in_validation(dummy_method):
    from ui import state as S
    p = {q.name: q for q in dummy_method.params}
    assert not S.param_visible(p["k"], {"mode": "a"}) and S.param_visible(p["k"], {"mode": "b"})
    assert not S.param_visible(p["x"], {"mode": "b", "k": 7}) and S.param_visible(p["x"], {"mode": "b", "k": 4})
    assert S.validate_params("t_dummy", {"mode": "a", "k": 3, "x": 0.5}) == {"mode": "a", "k": 3, "x": 0.5}
    S.validate_params("t_dummy", {"mode": "b", "k": 9})       # x hidden (k = 9), at its default
    with pytest.raises(ValueError):
        S.validate_params("t_dummy", {"mode": "b", "k": 10})  # visible and out of range
    with pytest.raises(ValueError):
        S.validate_params("t_dummy", {"mode": "a", "k": 99})  # hidden but not at its default: validated


def test_when_conditions_in_the_widgets(dummy_method):
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st
        from qemx.noise import NoiseSpec
        from ui import state as S
        from ui import widgets as W
        S.init()
        st.session_state.t_out = W.param_inputs("t_dummy", "tt", NoiseSpec(), 3)

    at = AppTest.from_function(script, default_timeout=120).run()
    _ok(at)
    assert [w.key for w in at.number_input if w.key.startswith("w_tt_")] == []          # mode a: k, x hidden
    assert at.session_state["t_out"] == {"mode": "a", "k": 3, "x": 0.5}
    next(w for w in at.selectbox if w.key == "w_tt_t_dummy_mode").set_value("b").run()
    _ok(at)
    keys = [w.key for w in at.number_input]
    assert "w_tt_t_dummy_k" in keys and "w_tt_t_dummy_x" in keys
    k_box = next(w for w in at.number_input if w.key == "w_tt_t_dummy_k")
    assert k_box.max == 9 and k_box.min == 1                  # the schema's bounds reach the widget
    k_box.set_value(7).run()
    _ok(at)
    assert "w_tt_t_dummy_x" not in [w.key for w in at.number_input]
    assert at.session_state["t_out"] == {"mode": "b", "k": 7, "x": 0.5}


def test_hardware_support_hook_and_nil_split(dummy_method):
    from qemx import hardware as H
    from qemx.methods import Recipe
    ok, why = H.supported(Recipe.make("nil"))
    assert not ok and "equal" in why
    assert H.supported(Recipe.make("nil", target_split="equal"))[0]
    assert H.supported(Recipe.make("t_dummy", mode="a"))[0]
    assert H.supported(Recipe.make("t_dummy", mode="b")) == (False, "mode b is simulation only")


def test_parsers_accept_ascii_decimal_numbers_only():
    from qemx import sweep as SW
    from qemx.circuit import ParseError, parse
    from qemx.compile import valid_input
    for bad in ("h ٣", "h 1_0", "rx(0x10) 0", "rx(True) 0", "rx(1_000) 0"):
        with pytest.raises(ParseError):
            parse(bad, 12)
    assert parse("rx(2.5e-3) 0; cx 0,1", 2).text()
    assert not valid_input("ph:٣") and valid_input("ph:0.25")
    with pytest.raises(SW.SweepError):
        SW.input_tokens("ramp:٣", 3)
    from ui import widgets as W
    assert W.parse_floats("1, 2.5e-3; -4") == [1.0, 0.0025, -4.0]
    assert W.parse_ints("0, 3") == [0, 3]
    for bad in ("nan", "inf", "1_0", "0x1"):
        with pytest.raises(ValueError):
            W.parse_floats(bad)
    with pytest.raises(ValueError):
        W.parse_ints("٣")


# ------------------------------------------------------------------ 7. hardware: partial submission, failed jobs, time
def _bell_plan(recipes):
    from qemx import device as D, hardware as H, templates as T
    from qemx.compile import Compiled, with_io
    tpl = T.bell()
    b = D.fake_backend("fake_perth")
    spec, mm, phys, info = D.compile_for_target(with_io(tpl.spec), b, D.good_target_for(b), "FakePerth")
    comp = Compiled(with_io(tpl.spec), spec, mm, info)
    return H.make_plan(comp, tpl.observable, recipes, 4000, tpl.symmetry, 0, b), b


@pytest.mark.slow
def test_submit_keeps_the_jobs_sent_before_a_failure(monkeypatch):
    import qiskit_ibm_runtime as R
    from qemx import hardware as H
    from qemx.recipes import preset
    plan, b = _bell_plan([preset("ZNE · Richardson")])
    calls = []

    class FakeJob:
        def __init__(self, i):
            self.i = i

        def job_id(self):
            return "job-%d" % self.i

    class FakeSampler:
        def __init__(self, mode=None):
            from types import SimpleNamespace
            self.options = SimpleNamespace(twirling=SimpleNamespace(), dynamical_decoupling=SimpleNamespace())

        def run(self, pubs, shots=None):
            calls.append(shots)
            if len(calls) == 2:
                raise RuntimeError("too many pending jobs")
            return FakeJob(len(calls))

    def no_batch(**kw):
        raise RuntimeError("open plan: no batch mode")
    monkeypatch.setattr(R, "SamplerV2", FakeSampler)
    monkeypatch.setattr(R, "Batch", no_batch)
    with pytest.raises(H.SubmitError) as ei:
        H.submit(plan, b)
    assert [r["job_id"] for r in ei.value.records] == ["job-1"] and len(calls) == 2


def test_collect_reports_failed_jobs_and_caches_results():
    from qemx import hardware as H

    class Counts:
        def __init__(self, c):
            self.c = c

        def get_counts(self):
            return self.c

    class Pub:
        def __init__(self, c):
            from types import SimpleNamespace
            self.data = SimpleNamespace(c=Counts(c))

    class Job:
        def __init__(self, status, counts=None):
            self.s, self.counts, self.fetched = status, counts, 0

        def status(self):
            return self.s

        def result(self):
            self.fetched += 1
            return [Pub(self.counts)]

    jobs = {"a": Job("DONE", {"00": 3, "11": 1}), "b": Job("ERROR")}

    class Service:
        def job(self, jid):
            return jobs[jid]
    recs = [dict(job_id="a", pubs=[("k1", 0)]), dict(job_id="b", pubs=[("k2", 0)])]
    cache = {}
    freqs, status = H.collect(Service(), recs, 2, cache=cache)
    assert set(freqs) == {"k1"} and np.allclose(freqs["k1"], [0.75, 0, 0, 0.25])
    assert [s["failed"] for s in status] == [False, True]
    H.collect(Service(), recs, 2, cache=cache)
    assert jobs["a"].fetched == 1                            # finished results are not downloaded twice


@pytest.mark.slow
def test_make_plan_stops_at_the_time_limit():
    from qemx import compile as CP, hardware as H, templates as T
    from qemx.recipes import preset
    from qemx.resources import RunTimeout, deadline
    tpl = T.build("qft")
    comp = CP.build(tpl.spec, None, None, CP.CompileOptions())
    with deadline(1e-6):
        time.sleep(0.01)
        with pytest.raises(RunTimeout):
            H.make_plan(comp, tpl.observable, [preset("ZNE · Richardson"), preset("PEC · inverse")], 20000)


# ------------------------------------------------------------------ 8. export: collisions and docstring
def test_flattened_engine_has_no_conflicting_module_level_names():
    from qemx.export import ENGINE_MODULES
    defs = {}
    for m in ENGINE_MODULES:
        src = open(os.path.join(ROOT, "qemx", m), encoding="utf-8").read()
        for node in ast.parse(src).body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            for nm in names:
                defs.setdefault(nm, set()).add((m, ast.get_source_segment(src, node)))
    clashes = {nm: sorted(x[0] for x in v) for nm, v in defs.items()
               if len({x[0] for x in v}) > 1 and len({x[1] for x in v}) > 1}
    assert not clashes, clashes


def test_export_docstring_cannot_be_broken_by_a_name():
    from qemx import export as EX
    evil = 'x"""\nimport os; os.system("echo PWNED")\n"""\\'
    state = dict(circuit=dict(n=1, name=evil, ops=[]), inputs=["0"], basis="Z", relabel=False,
                 compile=dict(mode="abstract"), noise={}, observable=dict(key="k", label="l", tex="t", values=[1, -1]),
                 symmetry=None, recipes=[], backend="numpy", shots=10, t_shot_us=1.0)
    src = EX.script(state)
    compile(src, "export", "exec")
    tree = ast.parse(src)
    assert isinstance(tree.body[0], ast.Expr) and "QEM Explorer export" in tree.body[0].value.value
    calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert not any(ast.unparse(n).startswith("os.system") for n in calls)


# ------------------------------------------------------------------ 9. small things
def test_cost_rows_carry_the_cost_model_fields():
    from qemx import report as RP
    res = dict(shots=1000, rows=[dict(method="Sym", family="symmetry", estimate=0.9, bias=-0.1, sd=0.01, v=1.0,
                                      fixed_shots=0, cal_circuits=0, w_cal=0.0, u_train=0.0, n_circuits=3,
                                      accepted=0.25)])
    r = RP.cost_rows(res)[0]
    assert r["n_circuits"] == 3 and r["accepted"] == 0.25


def test_state_version_is_the_package_version():
    import qemx
    from ui import state as S
    assert S.VERSION == qemx.__version__


def test_spec_and_signature_are_cached_per_circuit_version():
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st
        from ui import state as S
        S.init()
        ss = st.session_state
        a, b = S.spec(), S.spec()
        s1 = S.signature()
        ss.ops = list(ss.ops) + [dict(name="h", qubits=[0], params=[], kind="gate", tag="")]
        ss.ops_version += 1
        c = S.spec()
        ss.t_res = (a is b, c is not a, len(c.ops) == len(a.ops) + 1, S.signature() != s1)

    at = AppTest.from_function(script, default_timeout=120).run()
    _ok(at)
    assert at.session_state["t_res"] == (True, True, True, True)


def test_dense_layout_never_reaches_a_lattice_compile():
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st
        from ui import state as S
        S.init()
        ss = st.session_state
        ss.compile = dict(S.COMPILE_DEFAULT, mode="isa", coupling="grid", layout="dense")
        ss.backend = "numpy"
        ss.t_layout = S.compile_options().layout

    at = AppTest.from_function(script, default_timeout=120).run()
    _ok(at)
    assert at.session_state["t_layout"] == "sabre"


def test_json_state_download_never_holds_a_token():
    """The state written for downloads is built from the model only."""
    from streamlit.testing.v1 import AppTest

    def script():
        import json
        import streamlit as st
        from ui import state as S
        S.init()
        st.session_state["hw_token"] = "FAKE-TOKEN-0987654321"
        st.session_state.t_blob = json.dumps(S.to_json())

    at = AppTest.from_function(script, default_timeout=120).run()
    _ok(at)
    assert "FAKE-TOKEN" not in at.session_state["t_blob"]
    assert json.loads(at.session_state["t_blob"])["circuit"]["ops"]
