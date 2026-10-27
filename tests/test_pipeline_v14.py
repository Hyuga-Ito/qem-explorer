"""v1.4 audit fixes of the Direction A pipeline (qemx.phase): exact integer allocations, validated costs, per-level
calibration coefficients, failure handling, collect restricted to the manifest, fresh tables, manifest checks."""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os

import numpy as np
import pytest

from qemx.budget import optimal_integer_split
from qemx.phase import ledger as L


# ------------------------------------------------------------------ ledger
def _brute(cost, B, pol, levels_v=False):
    """min over every integer (s_tr, s) of the variance (first-order u = u1/s_tr, or the levels)."""
    best = math.inf
    ts = ([cost.s_tr] if pol.name == "fixed" else range(pol.s_tr_min, B // max(cost.T, 1) + 1)) if cost.T else [0]
    for t in ts:
        R = B - cost.T * t
        ss = [0] if cost.c == 0 else range(pol.s_min, (R - cost.m) // cost.c + 1)
        for s_ in ss:
            N = R - cost.c * s_
            if N < cost.m:
                continue
            vt = optimal_integer_split(cost.a, N)[0] if len(cost.a) > 1 else cost.v / N
            best = min(best, vt + (cost.w / s_ if cost.c else 0.0) + (cost.u1 / t if cost.T else 0.0))
    return best


def test_integer_allocation_is_exact_against_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(400):
        m = int(rng.choice([1, 3]))
        a = tuple(rng.uniform(0.01, 1, m)) if m > 1 else ()
        c = L.Cost("x", b=0.0, v=float(rng.uniform(0.01, 3)), w=float(rng.uniform(0, 2)), c=int(rng.integers(0, 4)),
                   T=int(rng.integers(0, 5)), s_tr=int(rng.integers(1, 6)), u1=float(rng.uniform(0, 2)), m=m, a=a)
        for name in ("fixed", "opt"):
            pol = L.Policy(name, True, int(rng.integers(1, 3)), int(rng.integers(1, 3)))
            B = int(rng.integers(L.min_budget(c, pol), L.min_budget(c, pol) + 60))
            al = L.allocate(c, B, pol)
            best = _brute(c, B, pol)
            assert al.feasible == math.isfinite(best)
            if al.feasible:
                assert al.var == pytest.approx(best, rel=1e-12), (c, B, name)


def test_audit_examples_rounding_and_first_order_search():
    # rounding s* was not optimal for several target circuits; RMSE(8) < RMSE(9) contradicted monotonicity
    c = L.Cost("z", b=0.0, v=(0.058 + 0.571 + 0.022) ** 2, w=0.889, c=2, m=3, a=(0.058, 0.571, 0.022))
    pol = L.Policy("fixed", True, 1, 1)
    r = [L.rmse(c, B, pol) for B in range(5, 400)]
    assert all(r[i + 1] <= r[i] for i in range(len(r) - 1))
    # the unit-move descent of the first-order opt path stopped at (132, 46); the optimum is (69, 47)
    rng = np.random.default_rng(11)
    for _ in range(40):
        T = int(rng.choice([8, 16, 208]))
        cost = L.Cost("L", b=0.001, v=float(rng.uniform(0.2, 5)), w=float(rng.uniform(1e-3, 2)), c=2, T=T, s_tr=1024,
                      u1=float(rng.uniform(1e-3, 50)), m=1)
        for B in (3000, 10000):
            al = L.allocate(cost, B, L.PIPELINE_OPT)
            if not al.feasible:
                continue
            best = min(L._given_tr(dataclasses.replace(cost, s_tr=t), B, t, L.PIPELINE_FIXED).var
                       for t in range(32, (B - 1 - 64) // T + 1))
            assert al.var == pytest.approx(best, rel=1e-12)


def test_cost_rejects_nonfinite_and_malformed_values():
    for kw in (dict(b=math.nan), dict(v=-1.0), dict(w=math.inf), dict(u1=math.nan), dict(m=0), dict(c=1.5),
               dict(a=(0.1, math.nan)), dict(T=2, s_tr=4, levels=((4, 0.0, math.nan),)),
               dict(T=2, s_tr=4, levels=((4, 0.0, 1.0), (4, 0.0, 1.0))), dict(T=2, s_tr=4, levels=((0, 0.0, 1.0),))):
        base = dict(b=0.0, v=1.0)
        base.update(kw)
        with pytest.raises(ValueError):
            L.Cost("x", **base)


def test_levels_carry_their_own_v_and_w():
    c = L.Cost("nil", b=0.0, v=1.0, w=0.5, c=2, T=16, s_tr=1024,
               levels=((256, 0.01, 1e-3, 2.0, 3.0), (1024, 0.002, 2e-4, 1.5)))
    assert c.at_level(256) == (0.01, 1e-3, 2.0, 3.0)
    assert c.at_level(1024) == (0.002, 2e-4, 1.5, 0.5)          # no w at this level: the method's
    a = L.allocate(c, 10 ** 6, L.Policy("opt", True, 1, 1))
    for t, (b, u, v, w) in ((256, c.at_level(256)), (1024, c.at_level(1024))):
        R = 10 ** 6 - 16 * t
        best = min(v / (R - 2 * s) + w / s for s in range(1, R // 2))
        if a.s_tr == t:
            assert a.var == pytest.approx(best + u, rel=1e-12) and a.var_cal == pytest.approx(w / a.s)
    assert L.floor_mse(c, "opt") == pytest.approx(min(0.01 ** 2 + 1e-3, 0.002 ** 2 + 2e-4))


def test_crossover_through_ties_is_found(monkeypatch):
    c1, c2 = L.Cost("one", b=0.0, v=1.0), L.Cost("two", b=0.0, v=1.0)

    def fake(cost, B, policy="fixed"):
        if cost.method == "one":
            return 1.0
        return 2.0 if B < 5000 else (1.0 if B < 6000 else 0.5)    # a tie on [5000, 6000)
    monkeypatch.setattr(L, "rmse", fake)
    for pts in (5, 11, 401):
        assert L.crossovers(c1, c2, lo=1e3, hi=1e4, points=pts) == [(6000.0, "two")]


# ------------------------------------------------------------------ training levels, failures, validation
class _FakeRes:
    """train_resampler that fails or produces non-finite estimates at chosen levels."""
    details = {"crn_key": "nil:test"}
    joint_calibration = False

    def __init__(self, bad):
        self.bad = bad

    def train_resampler(self, s_tr, R, rng, theta=None):
        if self.bad.get(s_tr) == "raise":
            raise np.linalg.LinAlgError("singular")
        est = 0.3 + rng.normal(0, 1 / math.sqrt(s_tr), R)
        if self.bad.get(s_tr) == "nan":
            est[3] = np.nan
        return est, np.full(R, 0.8)


def _spec(**kw):
    from qemx.phase.cell import CellSpec
    base = dict(stage="t", family="ghz", n=4, fparams=(("theta", 1.0),), train_levels=(64, 256, 1024),
                train_reps=50, train_reps_other=40)
    base.update(kw)
    return CellSpec(**base)


def test_failed_levels_are_flagged_and_left_out_of_the_ledger():
    from qemx.phase.cell import cost_of, training_levels
    row = dict(key="NIL-X", s_tr=1024, status="ok", b=0.0, v=0.8, w=0.0, c=0, T=16, m=1, a=[])
    lv = training_levels(_spec(), row, _FakeRes({64: "raise", 256: "nan"}), ideal=0.3)
    st = {l["s_tr"]: l for l in lv}
    assert not st[64]["ok"] and "LinAlgError" in st[64]["error"]
    assert not st[256]["ok"] and st[256]["finite"] == 39 and "non-finite" in st[256]["error"]
    assert st[1024]["ok"] and st[1024]["reps"] == 50 and {"b", "u", "v", "se_b", "se_u"} <= set(st[1024])
    row["levels"] = lv
    c = cost_of(row)
    assert [l[0] for l in c.levels] == [1024]                          # the opt policy cannot pick a failed level


def test_streams_are_shared_by_rules_on_the_same_circuits():
    from qemx.phase.cell import training_levels
    row = dict(key="A", s_tr=1024, status="ok", b=0.0, v=0.8, w=0.0, c=0, T=16, m=1, a=[])
    l1 = training_levels(_spec(), row, _FakeRes({}), ideal=0.3)
    l2 = training_levels(_spec(), dict(row, key="B"), _FakeRes({}), ideal=0.3)
    assert [l["u"] for l in l1] == [l["u"] for l in l2]                 # same crn_key: the same sampled data


def test_time_limit_and_memory_errors_abort_the_cell(monkeypatch, tmp_path):
    from qemx.phase import cell as C
    from qemx.phase.cell import MethodSpec
    from qemx.phase.run import _run_group
    from qemx.phase.store import read_cell
    from qemx.resources import RunTimeout
    spec = _spec(methods=(MethodSpec.make("CDR", "learning", features="cdr", model="ols", n_train=4,
                                          train_shots=256),), layout=(136, 143, 142, 141))
    for exc, status in ((RunTimeout("the run passed its time limit"), "timeout"), (MemoryError("big"), "oom")):
        def boom(*a, exc=exc, **k):
            raise exc
        monkeypatch.setattr(C, "training_levels", boom)
        with pytest.raises(type(exc)):
            C.cell_runner(spec)
        res = _run_group((str(tmp_path), [spec.to_json()], None))
        assert res[0][1] == status and read_cell(str(tmp_path), spec.cell_id)["status"] == status


def test_zero_standard_error_never_hides_a_difference():
    from qemx.phase.validate import _z
    assert _z(0.2, 0.0) == math.inf and _z(-0.1, 0.0) == -math.inf and _z(0.0, 0.0) == 0.0
    assert _z(0.3, 0.1) == pytest.approx(3.0)


# ------------------------------------------------------------------ manifests, output directories, collect
SMALL = """
manifest_version: 1
study: test
defaults: {device: fake_fez, layout: [136, 143, 142, 141], train_levels: [256], train_reps: 20, train_reps_other: 20}
budgets: {grid: {log10_min: 3, log10_max: 5, points: 5}}
methods:
  RAW: {estimator: raw}
  ZNE-R: {estimator: zne, scales: [1.0, 3.0], rule: richardson}
method_sets: {core: [RAW, ZNE-R]}
circuits: {g: [{family: ghz, n: 4, theta: 1.0}]}
stages:
  - {name: S, engine: E2, circuits: g, methods: core, scales: [1.0]}
"""


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


@pytest.mark.parametrize("mutate,msg", [
    (lambda m: m["method_sets"].update(core=["RAW", "RAW"]), "twice"),
    (lambda m: m["stages"][0].update(methods=["RAW", "RAW"]), "twice"),
    (lambda m: m["stages"][0].update(train_reps=1), "train_reps"),
    (lambda m: m["stages"][0].update(train_levels=[0, -5]), "train_levels"),
    (lambda m: m["stages"][0].update(validate={"budgets": [1.0e+4]}), "fraction"),
    (lambda m: m["budgets"].update(grid={"log10_min": 5, "log10_max": 3}), "log10"),
    (lambda m: m["budgets"].update(grid={"points": 1}), "points"),
    (lambda m: m["budgets"].update(grids={}), "allowed"),
])
def test_manifest_checks_close_the_audit_gaps(mutate, msg):
    import yaml
    from qemx.phase.manifest import check
    m = yaml.safe_load(SMALL)
    check(m)
    mutate(m)
    with pytest.raises(ValueError, match=msg):
        check(m)


def test_output_directory_is_bound_to_one_manifest(tmp_path):
    from qemx.phase.run import prepare_out
    m1 = _write(tmp_path / "m1.yaml", SMALL)
    m2 = _write(tmp_path / "m2.yaml", SMALL.replace("study: test", "study: other"))
    out = str(tmp_path / "out")
    prepare_out(out, m1)
    prepare_out(out, m1)                                                 # resume: same manifest
    with pytest.raises(SystemExit):
        prepare_out(out, m2)
    prepare_out(out, m2, force=True)
    with open(os.path.join(out, "run_history.jsonl"), encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 3
    with open(os.path.join(out, "manifest.yaml"), encoding="utf-8") as f:
        assert "study: other" in f.read()


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    from qemx.phase.cell import cell_runner
    from qemx.phase.manifest import expand, load
    from qemx.phase.run import prepare_out
    from qemx.phase.store import write_cell
    d = tmp_path_factory.mktemp("run")
    mp = _write(d / "m.yaml", SMALL)
    out = str(d / "out")
    prepare_out(out, mp)
    cells = expand(load(mp))
    rec = cell_runner(cells[0])
    write_cell(out, rec)
    return out, rec


def test_collect_uses_the_manifest_cells_and_rewrites_every_table(small_run, tmp_path):
    import shutil
    import pandas as pd
    from qemx.phase.store import collect, write_cell
    src, rec = small_run
    out = str(tmp_path / "o")
    shutil.copytree(src, out)
    stale = json.loads(json.dumps(rec))
    stale["cell_id"] = "f" * 20                                         # a cell of an earlier version of the manifest
    write_cell(out, stale)
    t = os.path.join(out, "tables")
    os.makedirs(t, exist_ok=True)
    pd.DataFrame([dict(cell_id="old", method="RAW", z_rmse=9.9)]).to_csv(os.path.join(t, "validation.csv"),
                                                                         index=False, encoding="utf-8")
    n = collect(out)
    assert n["used"] == 1 and n["stale"] == 1 and n["missing"] == 0 and n["budgets"] == 5
    assert len(pd.read_csv(os.path.join(t, "validation.csv"), encoding="utf-8")) == 0      # the old table did not survive
    led = pd.read_csv(os.path.join(t, "ledger.csv"), encoding="utf-8")
    assert set(led.cell_id) == {rec["cell_id"]} and len(led) == 2 * 5 * 2   # methods × budgets × policies
    assert sorted(led.B.unique()) == sorted({int(round(x)) for x in np.logspace(3, 5, 5)})
    with open(os.path.join(t, "collect.json"), encoding="utf-8") as f:
        assert json.load(f)["stale_ids"] == ["f" * 20]
    n_all = collect(out, only_manifest=False)
    assert n_all["used"] == 2


def test_one_bad_record_never_aborts_collect(small_run, tmp_path, monkeypatch):
    import shutil
    import pandas as pd
    from qemx.phase import store as S
    src, rec = small_run
    out = str(tmp_path / "o")
    shutil.copytree(src, out)
    orig = S.ledger_rows

    def flaky(r, *a, **k):
        if r["cell_id"] == rec["cell_id"]:
            raise KeyError("broken")
        return orig(r, *a, **k)
    monkeypatch.setattr(S, "ledger_rows", flaky)
    n = S.collect(out)
    assert n["tables"]["methods"]["rows"] == 0 and n["tables"]["errors"]["rows"] == 1
    er = pd.read_csv(os.path.join(out, "tables", "errors.csv"), encoding="utf-8")
    assert er.status.tolist() == ["collect-error"] and "KeyError" in er.error[0]


def test_parquet_failure_removes_the_old_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from qemx.phase.store import _write_table
    base = str(tmp_path / "x")
    pd.DataFrame([dict(a=1)]).to_parquet(base + ".parquet")
    info = _write_table([dict(a=1, b=1), dict(a=2, b="text")], base)     # a mixed column: no parquet
    assert not info["parquet"] and "warning" in info and not os.path.exists(base + ".parquet")
    assert len(pd.read_csv(base + ".csv", encoding="utf-8")) == 2


def test_ledger_rows_accept_policy_names_and_use_the_cells_bounds(small_run):
    from qemx.phase.store import cell_policies, ledger_rows
    _, rec = small_run
    r2 = json.loads(json.dumps(rec))
    r2["cell"]["s_min"] = 200
    pol = cell_policies(r2, ["fixed"])[0]
    assert pol.s_min == 200 and pol.s_tr_min == rec["cell"]["s_tr_min"]
    L_, W_ = ledger_rows(r2, [10 ** 4], ["fixed"])
    raw = next(l for l in L_ if l["method"] == "RAW")
    assert raw["s"] >= 200


def test_winner_map_draws_one_candidate_set():
    pd = pytest.importorskip("pandas")
    import matplotlib
    matplotlib.use("Agg")
    from qemx.phase.plots import STYLE, _style, winner_map
    rows, wins = [], []
    for i, lam in enumerate((0.1, 0.3)):
        cid = "%020d" % i
        rows.append(dict(cell_id=cid, stage="S", family="ghz", n=4, lam_lc=lam, fparams="{}"))
        for B in (10 ** 3, 10 ** 4):
            wins.append(dict(cell_id=cid, policy="fixed", B=B, candidates="all", winner="FR-L"))
            wins.append(dict(cell_id=cid, policy="fixed", B=B, candidates="no_optimistic", winner="NIL-G2"))
    ax = winner_map(pd.DataFrame(wins), pd.DataFrame(rows), stage="S", family="ghz", n=4)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [STYLE["FR-L"][2]]
    ax = winner_map(pd.DataFrame(wins), pd.DataFrame(rows), stage="S", family="ghz", n=4,
                    candidates="no_optimistic")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [STYLE["NIL-G2"][2]]
    assert _style("NEW-KEY")[2] == "NEW-KEY"


def test_records_are_checked_against_the_schema_on_write(tmp_path, small_run):
    pytest.importorskip("jsonschema")
    from qemx.phase.store import failure_record, write_cell
    _, rec = small_run
    bad = json.loads(json.dumps(rec))
    bad["status"] = "fine"
    with pytest.raises(ValueError, match="schema"):
        write_cell(str(tmp_path), bad)
    write_cell(str(tmp_path), failure_record(rec["cell"], rec["cell_id"], "timeout", "limit"))
    with pytest.raises(ValueError):
        failure_record(rec["cell"], rec["cell_id"], "ok", "")


def test_lightcone_gate_count_excludes_dd_pulses():
    from qemx.phase import families as FAM
    from qemx.phase.lightcone import calculate_lambda_lc
    from qemx import device as D
    fc = FAM.build("ghz", 4, theta=1.0)
    comp = FAM.compile_family(fc, "fake_fez", initial_layout=(136, 143, 142, 141))
    dm = D.get_model("fake_fez", tuple(comp.info["phys"]))
    rep = calculate_lambda_lc(comp, dm, fc.observable, layout=comp.info["phys"])
    gates = sum(1 for o in comp.implemented.ops if o.kind == "gate")
    assert 0 < rep.lc_gates <= gates


def test_bootstrap_standard_errors_are_recorded_beside_the_plug_in_ones():
    from qemx.phase.validate import _stats
    rng = np.random.default_rng(4)
    st = _stats(rng.normal(0.1, 1.0, 2000), 0.0)
    assert st["se_rmse_boot"] == pytest.approx(st["se_rmse"], rel=0.3)
    assert st["se_var_boot"] == pytest.approx(st["se_var"], rel=0.3)
    heavy = _stats(rng.standard_t(3, 2000), 0.0)
    assert heavy["se_var_boot"] > 0 and heavy["se_rmse_boot"] > 0
    assert "se_rmse_boot" not in _stats(rng.normal(size=20), 0.0)         # too few replicates for a bootstrap


def test_collect_warns_instead_of_failing_on_a_bad_manifest_copy(small_run, tmp_path):
    import shutil
    from qemx.phase.store import collect
    src, rec = small_run
    out = str(tmp_path / "o")
    shutil.copytree(src, out)
    with open(os.path.join(out, "manifest.yaml"), "a", encoding="utf-8") as f:
        f.write("\nunknown_key: 1\n")                        # no longer a valid manifest
    n = collect(out)
    assert n["used"] == 1 and n["warnings"] and "could not be expanded" in n["warnings"][0]
    os.unlink(os.path.join(out, "manifest.yaml"))
    assert collect(out)["warnings"][0].startswith("no manifest.yaml")
