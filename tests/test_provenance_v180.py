"""1.8.0: run binding at collection (F3), seed-scheme namespaces (N1), deployment variance (F5), strict JSON (F9)."""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [1000, 30000]
PROTOCOLS = ["RAW", "CDR"]
V2 = {"mode": "fresh", "per_half": 2, "seed_scheme": 2, "crn": "task"}


def _specs():
    from qemx.selection import task as TK
    return [TK.make_spec("ising", 3, steps=1), TK.make_spec("hea", 3, layers=1, angle_seed=4)]


def _cfg(designs=V2):
    return dict(budgets=BUDGETS, protocols=PROTOCOLS, R=8, master_seed=77, learning_designs=designs)


def _label(target, specs, designs=V2, run_id=None):
    from qemx.selection import binding as BD
    from qemx.selection import labels as LB
    cfg = _cfg(designs)
    run = BD.bind(str(target), cfg, specs, specs, [], "dev")
    rid = run["identity_digest"] if run_id is None else run_id
    for s in specs:
        LB.label_task(s, str(target), BUDGETS, PROTOCOLS, 8, 77, designs=designs, run_id=rid)
    return run


# ------------------------------------------------------------------------------------------------------ N1: seeds
def test_stream_seeds_are_namespaced_and_process_independent():
    from qemx.selection import emulate as EM
    a = EM.stream_seed(5, "task-a", "CDR", 0, 3, "design")
    assert a == EM.stream_seed(5, "task-a", "CDR", 0, 3, "design")
    others = {EM.stream_seed(5, "task-b", "CDR", 0, 3, "design"), EM.stream_seed(5, "task-a", "NIL", 0, 3, "design"),
              EM.stream_seed(5, "task-a", "CDR", 1, 3, "design"), EM.stream_seed(5, "task-a", "CDR", 0, 4, "design"),
              EM.stream_seed(5, "task-a", "CDR", -1, 0, "allocation"), EM.stream_seed(6, "task-a", "CDR", 0, 3, "design")}
    assert a not in others and len(others) == 6
    code = "import sys; sys.path.insert(0, %r); from qemx.selection import emulate as EM; " \
           "print(EM.stream_seed(5, 'task-a', 'CDR', 0, 3, 'design'))" % ROOT
    for hs in ("0", "12345"):                                      # Python's hash randomization does not enter
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8",
                             env=dict(os.environ, PYTHONHASHSEED=hs), timeout=120)
        assert out.stdout.strip() == str(a), out.stderr


def test_scheme_2_gives_each_task_its_own_designs_and_crn_base_shares_them():
    from qemx.selection import emulate as EM
    from qemx.selection import task as TK
    s1 = TK.make_spec("hea", 3, layers=1, angle_seed=4)
    s2 = TK.make_spec("hea", 3, layers=1, angle_seed=4, observable="z_all")      # same base circuit, other task
    assert s1.base_id == s2.base_id and s1.task_id != s2.task_id

    def seeds(spec, plan):
        p = EM.prepare(TK.build(spec), ["CDR"], seed=77, designs=plan)
        return [r["seed"] for r in p.design_records["CDR"]]
    v1 = {"mode": "fresh", "per_half": 2}
    assert seeds(s1, v1) == seeds(s2, v1)                           # 1.7.0: one pool for every task (N1)
    a, b = seeds(s1, V2), seeds(s2, V2)
    assert not set(a) & set(b)                                      # scheme 2: independent per task
    base = dict(V2, crn="base")
    assert seeds(s1, base) == seeds(s2, base)                       # declared common random numbers
    assert len(set(a)) == len(a)


def test_configuration_digests_keep_1_7_and_separate_scheme_2():
    from qemx.selection import emulate as EM
    from qemx.selection import labels as LB
    v1 = LB.config_digest(BUDGETS, PROTOCOLS, 8, 77, {"mode": "fresh", "per_half": 2})
    assert v1 == LB.config_digest(BUDGETS, PROTOCOLS, 8, 77, EM.DesignPlan("fresh", 1, 2))
    assert v1 != LB.config_digest(BUDGETS, PROTOCOLS, 8, 77, V2) != LB.config_digest(BUDGETS, PROTOCOLS, 8, 77,
                                                                                         dict(V2, crn="base"))
    with pytest.raises(ValueError):
        EM.DesignPlan.from_config({"mode": "fresh", "per_half": 2, "crn": "base"})          # crn needs scheme 2
    with pytest.raises(ValueError):
        EM.DesignPlan.from_config({"mode": "fixed-shared", "count": 2, "seed_scheme": 2})


def test_labels_reproduce_after_reorder_and_resume_and_refuse_mixed_schemes(tmp_path):
    from qemx.selection import labels as LB
    specs = _specs()
    _label(tmp_path / "a", specs)
    _label(tmp_path / "b", specs[::-1])
    for s in specs:
        x = LB.load_arrays(str(tmp_path / "a"), s.task_id)
        y = LB.load_arrays(str(tmp_path / "b"), s.task_id)
        for k in ("est", "flags", "design"):
            assert np.array_equal(x[k], y[k], equal_nan=True), k
        r = LB.load_record(str(tmp_path / "a"), s.task_id)
        assert r["seed_scheme"] == 2 and r["crn_group"] == s.task_id
        assert all("instance_id" in d for d in r["design_records"]["CDR"])
    again = LB.label_task(specs[0], str(tmp_path / "a"), BUDGETS, PROTOCOLS, 8, 77, designs=V2)
    assert again.get("skipped")                                      # resume: the checkpoint is reused
    with pytest.raises(RuntimeError, match="another configuration"):  # resume with scheme 1 refuses to mix
        LB.label_task(specs[0], str(tmp_path / "a"), BUDGETS, PROTOCOLS, 8, 77,
                      designs={"mode": "fresh", "per_half": 2})


def test_process_count_does_not_change_labels(tmp_path):
    from qemx.selection import labels as LB
    from qemx.selection import run as RN
    specs = _specs()
    for w, d in ((1, "w1"), (2, "w2")):
        RN._supervised_label(specs, str(tmp_path / d), BUDGETS, PROTOCOLS, 8, 77, V2, {}, w, lambda *a: None)
    for s in specs:
        x = LB.load_arrays(str(tmp_path / "w1"), s.task_id)
        y = LB.load_arrays(str(tmp_path / "w2"), s.task_id)
        assert np.array_equal(x["est"], y["est"], equal_nan=True)


# --------------------------------------------------------------------------------------------- F3: collection
def test_collection_rejects_records_of_another_configuration_or_run(tmp_path):
    from qemx.selection import binding as BD
    from qemx.selection import labels as LB
    specs = _specs()
    t = tmp_path / "run"
    _label(t, specs)
    ds, rep = BD.collect(str(t), require_complete=True)
    assert rep["rejected"] == [] and rep["legacy_unbound"] == [] and len(ds.rows) > 0
    jp = t / "tasks" / (specs[0].task_id + ".json")
    good = jp.read_text(encoding="utf-8")
    # 1) a wrong JSON configuration digest with a valid NPZ checksum
    r = json.loads(good)
    r["config_digest"] = "0" * 16
    jp.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(BD.IncompatibleRun, match="label configuration"):
        BD.collect(str(t))
    # 2) another run's identity
    r = json.loads(good)
    r["run_id"] = "f" * 16
    jp.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(BD.IncompatibleRun, match="not this run"):
        BD.collect(str(t))
    # 3) a record naming another task
    r = json.loads(good)
    r["task_id"] = specs[1].task_id
    jp.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(BD.IncompatibleRun, match="names task"):
        BD.collect(str(t))
    # 4) a 1.8.0 record that lost its run id: rejected; a legacy (≤ 1.7.0, no seed_scheme) one: accepted, reported
    r = json.loads(good)
    r["run_id"] = ""
    jp.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(BD.IncompatibleRun, match="not this run"):
        BD.collect(str(t))
    r.pop("seed_scheme")
    r.pop("crn_group")
    jp.write_text(json.dumps(r), encoding="utf-8")
    _, rep = BD.collect(str(t))
    assert rep["legacy_unbound"] == [specs[0].task_id]
    # 5) arrays that do not match the record
    jp.write_text(good, encoding="utf-8")
    npz = t / "tasks" / (specs[0].task_id + ".npz")
    raw = npz.read_bytes()
    npz.write_bytes(raw + b"\0")
    with pytest.raises(BD.IncompatibleRun, match="arrays"):
        BD.collect(str(t))
    npz.write_bytes(raw)
    assert BD.collect(str(t))[1]["rejected"] == []
    del LB


def test_explicit_imports_are_accepted_and_audited(tmp_path):
    from qemx.selection import binding as BD
    specs = _specs()
    src, dst = tmp_path / "src", tmp_path / "dst"
    _label(src, specs)
    BD.bind(str(dst), _cfg(), specs, specs, [], "dev")
    BD.import_tasks(str(src), str(dst), [s.task_id for s in specs], reason="test reuse")
    _, rep = BD.collect(str(dst), require_complete=True)
    assert sorted(rep["imported"]) == sorted(s.task_id for s in specs) and rep["rejected"] == []
    jp = dst / "tasks" / (specs[0].task_id + ".json")
    jp.write_text(jp.read_text(encoding="utf-8").replace('"R": 8', '"R": 8 '), encoding="utf-8")
    with pytest.raises(BD.IncompatibleRun, match="changed after its import"):
        BD.collect(str(dst))


def test_mismatch_labeling_binds_its_run_id():
    import inspect
    from qemx.selection import run as RN
    src = inspect.getsource(RN.mismatch_label)
    assert 'run_id=run["identity_digest"]' in src


# --------------------------------------------------------------------------------------------- F5: variance
def test_deployment_variance_matches_the_random_effects_model():
    """Review's synthetic check: K = m = 10, σ_D² = 1, σ_W² = 0.01. The pooled sample variance's expectation is
    σ_W² + m(K−1)/(Km−1)·σ_D² = 0.9191, not the deployment variance 1.01; var_deploy is unbiased for 1.01."""
    from qemx.selection import labels as LB
    rng = np.random.default_rng(20261024)
    K, m, reps = 10, 10, 4000
    design = np.repeat(np.arange(K), m)
    pooled, dep = [], []
    for _ in range(reps):
        x = rng.normal(0, 1.0, K)[design] + rng.normal(0, 0.1, K * m)
        d = LB.deployment_variance(x, design)
        pooled.append(d["var_pooled"])
        dep.append(d["var_deploy"])
        assert d["between_coef"] == pytest.approx(m * (K - 1) / (K * m - 1))
    se = 1.01 * math.sqrt(2 / (K - 1)) / math.sqrt(reps) * 1.5
    assert np.mean(pooled) == pytest.approx(0.01 + 90 / 99, abs=4 * se)
    assert np.mean(dep) == pytest.approx(1.01, abs=4 * se)


def test_deployment_variance_unbalanced_groups_and_independent_trials():
    from qemx.selection import emulate as EM
    from qemx.selection import labels as LB
    rng = np.random.default_rng(3)
    sizes = [3, 7, 12, 5, 9, 4]
    design = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])
    est = []
    for _ in range(4000):
        x = rng.normal(0, 0.5, len(sizes))[design] + rng.normal(0, 1.0, len(design))
        est.append(LB.deployment_variance(x, design)["var_deploy"])
    assert np.mean(est) == pytest.approx(1.25, rel=0.05)              # σ_D² + σ_W² = 0.25 + 1
    x = rng.normal(size=50)
    d = LB.deployment_variance(x, np.full(50, EM.NO_DESIGN))
    assert d["var_deploy"] == d["var_pooled"] == pytest.approx(float(np.var(x, ddof=1)))


# --------------------------------------------------------------------------------------------- F9: strict JSON
def test_current_exports_are_strict_json_and_legacy_tokens_stay_readable(tmp_path):
    from qemx import textio as TX
    from qemx.selection import deploy as DP
    obj = dict(margin=math.inf, missing=math.nan, rows=[1.0, -math.inf], ok=0.5)
    p = tmp_path / "x.json"
    TX.write_json(p, obj)
    text = p.read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))   # strict parser accepts it
    back = TX.read_json(p, strict_tokens=True)
    assert back["margin"] == math.inf and math.isnan(back["missing"]) and back["rows"][1] == -math.inf
    assert json.loads(text)["$nonfinite"] == {"/margin": "inf", "/missing": "nan", "/rows/1": "-inf"}
    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"a": NaN, "b": Infinity}', encoding="utf-8")
    assert math.isnan(TX.read_json(legacy)["a"])                     # versioned compatibility, never coerced
    with pytest.raises(TX.NonStrictJSON):
        TX.read_json(legacy, strict_tokens=True)
    b = DP.load_bundle(os.path.join(ROOT, "qemx", "selection", "bundles", "qemx-rec-1.7.0-deployable"))
    rec = DP.recommend(DP.from_spec(_specs()[0], 30000), b)
    TX.loads(TX.dumps(rec, default=str), strict_tokens=True)
