"""1.7.0: the review's findings as scientific and operational invariants of the selection study.

R1 fresh-design sampling and design-level uncertainty; R2 nested selective evaluation; R3/R4 canonical identity and the
ancestry registry; R5 statement kinds; S1 supervised execution; S2 failed and corrupted records; S3 run binding and
freeze enforcement; S4 union of flags and NIL/CDR fallback propagation; S5 declared weights and the SBS population;
S6 schemas; the deployment path (features without simulation, bundles, UI = CLI)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qemx.selection import emulate as EM  # noqa: E402
from qemx.selection import labels as LB  # noqa: E402

PROTOS = ["RAW", "RO", "CDR"]


def _fake(dir_, tid, *, mode="fresh", R=8, K=2, budgets=(1000, 10000), status="ok", npz=True, mu=0.5, seed=0,
          base="b" * 16, flags=None):
    """A label record with consistent arrays (protocols RAW, RO, CDR)."""
    os.makedirs(os.path.join(dir_, "tasks"), exist_ok=True)
    rng = np.random.default_rng(seed)
    nB = len(budgets)
    spec = dict(family="ghz", n=3, params=[], inputs="default", observable="default", device="fake_fez",
                layout="best", opt_level=1, seed=11, truth="")
    if status != "ok":
        rec = dict(schema=LB.SCHEMA, task_id=tid, base_id=base, spec=spec, config_digest="x", status=status,
                   error_kind="build", error="failed on purpose")
        LB.write_json(os.path.join(dir_, "tasks", tid + ".json"), rec)
        return rec
    est = mu + 0.1 * rng.normal(size=(2, len(PROTOS), nB, R))
    fl = np.zeros((2, len(PROTOS), nB, 4, R), bool) if flags is None else flags
    design = np.full((2, len(PROTOS), nB, R), -1, np.int16)
    design[:, 2] = np.arange(R) % K
    allocs = [dict(protocol=k, B=int(B), feasible=True, reason="", status="ok", ledger=dict(execution=int(B)))
              for k in PROTOS for B in budgets]
    rec = dict(schema=LB.SCHEMA, task_id=tid, base_id=base, spec=spec, config_digest="x", status="ok",
               budgets=list(budgets), protocols=PROTOS, R=R, ideal=mu, allocations=allocs,
               costs=[dict(protocol=k, B=int(B), t_exec=1.0, circuits_sent=1) for k in PROTOS for B in budgets],
               features=dict(n=3.0), flags=list(EM.FLAGS), learning_designs=dict(mode="fresh", per_half=K)
               if mode == "fresh" else K, design_plan=dict(mode=mode, count=K if mode != "fresh" else 1,
                                                           per_half=K if mode == "fresh" else 0))
    if npz:
        np.savez_compressed(os.path.join(dir_, "tasks", tid + ".npz"), est=est, flags=fl, design=design,
                            trial_seconds=np.zeros((len(PROTOS), nB)))
    LB.write_json(os.path.join(dir_, "tasks", tid + ".json"), rec)
    return rec



def _pilot_dev() -> str:
    """The bundled pilot's development labels with trial arrays: the corrected 1.7.0 pilot when shipped, else the
    1.6.0 pilot."""
    for name in ("selection_v170", "selection_pilot"):
        d = os.path.join(ROOT, "reports", name, "dev")
        t = os.path.join(d, "tasks")
        if os.path.isdir(t) and any(f.endswith(".npz") for f in os.listdir(t)):
            return d
    if "selection_v170" in os.environ.get("QEMX_REQUIRE_TRIALS", ""):        # 1.9.0: the trial-data workflow
        pytest.fail("QEMX_REQUIRE_TRIALS names selection_v170 but its per-trial arrays are missing")
    pytest.skip("no bundled pilot labels with trial arrays (extract the trial-data archive into the repository root)")

# ------------------------------------------------------------------ R1: designs
def test_design_plan_parses_both_estimands_and_keeps_1_6_digests():
    assert EM.DesignPlan.from_config(6) == EM.DesignPlan("fixed-shared", 6, 0)
    p = EM.DesignPlan.from_config({"mode": "fresh", "per_half": 10})
    assert p.mode == "fresh" and p.per_half == 10 and "fresh" in p.estimand
    for bad in (0, {"mode": "fresh"}, {"mode": "x", "per_half": 2}, {"mode": "fresh", "per_half": 2, "k": 1}, True):
        with pytest.raises(ValueError):
            EM.DesignPlan.from_config(bad)
    # a 1.6.0 configuration keeps its digest (resumed 1.6.0 runs and their records stay valid)
    old = LB.config_digest([1000], ["RAW", "CDR"], 100, 5, 6)
    assert old == LB.config_digest([1000], ["RAW", "CDR"], 100, 5, EM.DesignPlan("fixed-shared", 6, 0))
    assert old != LB.config_digest([1000], ["RAW", "CDR"], 100, 5, {"mode": "fresh", "per_half": 6})


def test_fresh_design_pools_are_disjoint_between_halves_and_apart_from_the_allocation_design():
    seeds = {(h, k): EM.pool_seed(20261019, h, k) for h in (0, 1) for k in range(50)}
    assert len(set(seeds.values())) == len(seeds)
    assert 20261019 % 2 ** 31 not in set(seeds.values())
    assert EM.pool_seed(1, 0, 3) == EM.pool_seed(1, 0, 3)                      # reproducible


@pytest.mark.slow
def test_fresh_designs_in_a_real_task_use_their_own_half_and_record_every_design():
    from qemx.selection import task as TK
    spec = TK.make_spec("ghz", 3, theta=1.0472)
    prep = EM.prepare(TK.build(spec), ["CDR"], seed=7, designs={"mode": "fresh", "per_half": 3})
    recs = prep.design_records["CDR"]
    assert sorted((r["half"], r["index"]) for r in recs) == [(h, k) for h in (0, 1) for k in range(3)]
    s0 = {r["seed"] for r in recs if r["half"] == 0}
    s1 = {r["seed"] for r in recs if r["half"] == 1}
    assert not s0 & s1 and 7 not in s0 | s1
    for h in (0, 1):
        tr = EM.trials(prep, "CDR", 100000, 7, np.random.default_rng(1), half=h)
        assert tr.design.tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert all(r["diagnostics"]["n_train"] == 16 for r in recs)


def test_design_level_standard_error_contains_the_between_design_variance():
    rng = np.random.default_rng(0)
    K, m = 10, 10
    design = np.repeat(np.arange(K), m)
    sd_b, sd_w = 1.0, 0.2
    est = []
    for _ in range(2000):
        L = rng.normal(0, sd_b, K)[design] + rng.normal(0, sd_w, K * m)
        est.append(L.mean())
    true_sd = float(np.std(est))                                                 # √(σ_b²/K + σ_w²/(Km))
    assert true_sd == pytest.approx(math.sqrt(sd_b ** 2 / K + sd_w ** 2 / (K * m)), rel=0.05)
    se_t, se_d, vb, vw = [], [], [], []
    for _ in range(400):
        L = rng.normal(0, sd_b, K)[design] + rng.normal(0, sd_w, K * m)
        st = LB.design_stats(L, design)
        se_t.append(st["se_trial"]), se_d.append(st["se_design"]), vb.append(st["var_between"])
        vw.append(st["var_within"])
    assert np.mean(se_d) == pytest.approx(true_sd, rel=0.1)                     # the design-level SE is right
    assert np.mean(se_t) < 0.5 * true_sd                                         # the trial-level SE is far too small
    assert np.mean(vb) == pytest.approx(sd_b ** 2, rel=0.15) and np.mean(vw) == pytest.approx(sd_w ** 2, rel=0.05)
    # balanced: the cluster-robust SE equals std(design means)/√K
    L = rng.normal(size=K * m)
    means = np.array([L[design == k].mean() for k in range(K)])
    assert LB.design_stats(L, design)["se_design"] == pytest.approx(means.std(ddof=1) / math.sqrt(K), rel=1e-12)
    # unequal repetitions are handled (no balanced-only formula)
    d2 = np.array([0] * 3 + [1] * 7 + [2] * 5)
    st = LB.design_stats(rng.normal(size=15), d2)
    assert st["designs"] == 3 and np.isfinite(st["se_design"])
    # no design: trial level
    st = LB.design_stats(np.arange(10.0), np.full(10, -1))
    assert st["se_design"] == st["se_trial"] and st["designs"] == 0


def test_increasing_repetitions_within_fixed_designs_does_not_shrink_the_design_term():
    rng = np.random.default_rng(1)
    K = 6
    out = {}
    for m in (10, 100):
        design = np.repeat(np.arange(K), m)
        ses = []
        for _ in range(200):
            L = rng.normal(0, 1.0, K)[design] + rng.normal(0, 0.1, K * m)
            ses.append(LB.design_stats(L, design)["se_design"])
        out[m] = np.mean(ses)
    assert out[100] > 0.8 * out[10]                    # not the √10 reduction of independent repetitions


def test_structural_fallback_of_learning_results_is_flagged_for_every_trial():
    class R:
        def __init__(self, d):
            self.details = d
    assert EM.structural_fallback(R(dict(degenerate=True))) == "degenerate-training"
    assert EM.structural_fallback(R({})) == "clifford"
    assert EM.structural_fallback(R(dict(n_features=3, names=["a"]))) == ""
    assert EM.structural_fallback(RuntimeError("x")) == ""


# ------------------------------------------------------------------ S2, S4: records, flags
def test_error_records_without_arrays_are_collected_and_corrupted_records_refused(tmp_path):
    from qemx.selection import dataset as DT
    d = str(tmp_path / "dev")
    _fake(d, "a" * 20, base="1" * 16)
    _fake(d, "b" * 20, status="error", base="2" * 16)
    fails = []
    lab = LB.summarize(d, failures=fails)
    assert set(lab.task_id) == {"a" * 20} and [f["task_id"] for f in fails] == ["b" * 20]
    ds = DT.build(d)
    assert len(ds.failures) == 1 and ds.failures[0]["status"] == "error"
    assert LB.load_task(d, "b" * 20)[1] is None
    _fake(d, "c" * 20, npz=False, base="3" * 16)
    with pytest.raises(LB.CorruptedRecord):
        DT.build(d)


def test_failure_share_is_the_union_of_overlapping_flags(tmp_path):
    from qemx.selection import dataset as DT
    R = 10
    fl = np.zeros((2, 3, 2, 4, R), bool)
    fl[1, 2, 0, 0, :5] = True                        # cal_fallback on trials 0–4
    fl[1, 2, 0, 2, 2:7] = True                       # learn_fallback on trials 2–6 (overlap 2–4)
    d = str(tmp_path / "dev")
    _fake(d, "a" * 20, R=R, flags=fl)
    ds = DT.build(d)
    i = [k for k, r in enumerate(ds.rows) if r["B"] == 1000][0]
    assert ds.fail[i, 2] == pytest.approx(0.7)                 # 7 of 10 trials have a flag (1.6.0 summed: 1.0)
    assert ds.flag_rates[i, 2].sum() == pytest.approx(1.0)
    lab = LB.summarize(d)
    row = lab[(lab.protocol == "CDR") & (lab.B == 1000)].iloc[0]
    assert row.fail_any_eval == pytest.approx(0.7) and row.flags_per_trial_eval == pytest.approx(1.0)


def test_fresh_design_labels_report_the_design_level_standard_error(tmp_path):
    from qemx.selection import dataset as DT
    d = str(tmp_path / "dev")
    _fake(d, "a" * 20, R=8, K=2)
    ds = DT.build(d)
    assert ds.design_mode == "fresh"
    i, j = 0, 2
    assert ds.SE_eval[i, j] != ds.SE_eval_trial[i, j]           # CDR: design level
    assert ds.SE_eval[i, 0] == ds.SE_eval_trial[i, 0]           # RAW: trials
    _fake(str(tmp_path / "mixed"), "a" * 20, mode="fixed-shared", K=2)
    _fake(str(tmp_path / "mixed"), "b" * 20, mode="fresh", K=2, base="c" * 16)
    with pytest.raises(ValueError, match="different design plans"):
        DT.build(str(tmp_path / "mixed"))


# ------------------------------------------------------------------ S1: supervision
def test_supervisor_enforces_time_and_memory_in_the_parent_and_reports_crashes():
    from qemx.selection import supervise as SV
    jobs = [SV.Job("ok", "_supervise_targets:sleep", (0.1,), seconds=60),
            SV.Job("slow", "_supervise_targets:sleep", (60,), seconds=2),
            SV.Job("crash", "_supervise_targets:crash", ()),
            SV.Job("boom", "_supervise_targets:boom", ()),
            SV.Job("hog", "_supervise_targets:hog", (2.0,), seconds=120, memory_bytes=int(0.6 * 2 ** 30))]
    out = {o.key: o for o in SV.run_jobs(jobs, workers=2)}
    assert out["ok"].status == "ok" and out["ok"].value == "slept"
    assert out["slow"].status == "timeout" and out["slow"].seconds < 30
    assert out["crash"].status == "crashed" and out["crash"].exitcode == 7
    assert out["boom"].status == "error" and "exploded on purpose" in out["boom"].message
    assert out["hog"].status == "memory"
    sup = SV.support()
    assert sup["timeout"].startswith("parent-enforced")


def test_one_worker_labeling_never_limits_the_parent_process():
    from qemx.selection import supervise as SV
    import _supervise_targets as T
    before = T.rlimit_now()
    out = SV.run_jobs([SV.Job("a", "_supervise_targets:rlimit_now", (), memory_bytes=2 ** 31)], workers=1)
    assert T.rlimit_now() == before                                      # the parent is untouched
    if before is not None:
        assert tuple(out[0].value) == (2 ** 31, 2 ** 31)                 # the child limited itself (POSIX)


def test_supervised_children_never_rerun_the_parents_main_script(tmp_path, monkeypatch):
    """1.7.0: "spawn" re-imports the parent's main script in the child; a Streamlit page or a test harness's generated
    script must not run again there (found after a Streamlit AppTest replaced __main__ in the test process)."""
    import types
    from qemx.selection import supervise as SV
    script = tmp_path / "parent_main.py"
    script.write_text("raise SystemExit('the parent main script must not run in a child')\n", encoding="utf-8")
    fake = types.ModuleType("__main__")
    fake.__file__ = str(script)
    fake.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", fake)
    out = SV.run_jobs([SV.Job("ok", "_supervise_targets:sleep", (0.05,), seconds=60)], workers=1)
    assert out[0].status == "ok" and out[0].value == "slept", out[0]
    assert sys.modules["__main__"] is fake                               # the parent's module is restored


# ------------------------------------------------------------------ S3: run binding, freeze
def _tiny_cfg(**over):
    cfg = dict(schema="qemx.selection.config/1", name="tiny", master_seed=3, R=4, budgets=[1000],
               learning_designs={"mode": "fresh", "per_half": 2}, protocols=["RAW", "RO"], final_test_fraction=0.0,
               tasks=dict(widths=[3], families={"ghz": [{"theta": 1.0}]}, devices=["fake_fez"], variants_per_base=0))
    cfg.update(over)
    return cfg


def test_run_binding_refuses_incompatible_resumes_and_mixing(tmp_path, monkeypatch):
    from qemx.selection import binding as BD
    from qemx.selection import design as DS
    cfg = _tiny_cfg()
    specs = DS.expand(cfg)
    t = str(tmp_path / "dev")
    BD.bind(t, cfg, specs, specs, [], "dev")
    BD.bind(t, cfg, specs, specs, [], "dev")                              # same run: a new session
    assert len(BD.load(t)["sessions"]) == 2
    with pytest.raises(BD.IncompatibleRun, match="labels_config"):
        BD.bind(t, _tiny_cfg(R=8), specs, specs, [], "dev")
    other = DS.expand(_tiny_cfg(tasks=dict(widths=[4], families={"ghz": [{"theta": 1.0}]}, devices=["fake_fez"])))
    with pytest.raises(BD.IncompatibleRun, match="tasks"):
        BD.bind(t, cfg, other, other, [], "dev")
    monkeypatch.setattr(BD, "code_digest", lambda: "changed")
    with pytest.raises(BD.IncompatibleRun, match="other code"):
        BD.bind(t, cfg, specs, specs, [], "dev")
    rec = BD.bind(t, cfg, specs, specs, [], "dev", accept_changes=True)
    assert rec["sessions"][-1]["accepted_changes"] == ["code"]
    legacy = str(tmp_path / "legacy")
    _fake(legacy, "a" * 20)
    with pytest.raises(BD.IncompatibleRun, match="without a run binding"):
        BD.bind(legacy, cfg, specs, specs, [], "dev")


def test_collection_uses_the_manifest_only_and_imports_are_recorded(tmp_path):
    from qemx.selection import binding as BD
    from qemx.selection import design as DS
    cfg = _tiny_cfg()
    specs = DS.expand(cfg)
    t = str(tmp_path / "dev")
    BD.bind(t, cfg, specs, specs, [], "dev")
    _fake(t, "f" * 20)                                                     # a stale file of another grid
    with pytest.raises(ValueError):                                        # nothing of the manifest is labeled
        BD.collect(t)
    with pytest.raises(BD.IncompatibleRun, match="not in the manifest"):
        BD.import_tasks(t, t, ["f" * 20], "test")
    ds_rep = None
    try:
        BD.collect(t, require_complete=True)
    except (BD.IncompatibleRun, ValueError) as e:
        ds_rep = str(e)
    assert ds_rep


def test_freeze_must_match_the_configuration(tmp_path):
    from qemx import textio as TX
    from qemx.selection import binding as BD
    from qemx.selection import design as DS
    cfg = _tiny_cfg()
    TX.write_json(str(tmp_path / "freeze.json"), dict(policy="MODEL[hgb|deployable]", config_digest="0" * 64))
    with pytest.raises(BD.IncompatibleRun, match="another configuration"):
        BD.check_freeze(str(tmp_path), cfg)
    TX.write_json(str(tmp_path / "freeze.json"), dict(policy="MODEL[hgb|deployable]",
                                                      config_digest=DS.freeze_digest(cfg)))
    assert BD.check_freeze(str(tmp_path), cfg)["policy"].startswith("MODEL[")


def test_final_command_refuses_a_wrong_freeze_before_labeling(tmp_path, monkeypatch):
    import yaml
    from qemx import textio as TX
    from qemx.selection import __main__ as CLI
    from qemx.selection import run as RN
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump(_tiny_cfg()), encoding="utf-8")
    TX.write_json(str(tmp_path / "freeze.json"), dict(policy="MODEL[hgb|deployable]", config_digest="0" * 64))
    called = []
    monkeypatch.setattr(RN, "label", lambda *a, **k: called.append(1))
    assert CLI.main(["final", str(cfgp), "--out", str(tmp_path)]) == 2 and not called


# ------------------------------------------------------------------ S6: schemas
def test_every_shipped_selection_configuration_satisfies_the_schema():
    from qemx.selection import design as DS
    d = os.path.join(ROOT, "configs", "selection")
    n = 0
    for f in sorted(os.listdir(d)):
        if f.endswith(".yaml") and f not in ("figures.yaml", "figures18.yaml", "hardware17.yaml") \
                and not f.startswith("figure_spec"):          # plotting, hardware, figure/table specification
            cfg = DS.load(os.path.join(d, f))
            specs = DS.expand(cfg)
            assert specs, f
            # every device of a shipped configuration can be bound to a run (1.7.0: a coherent-fraction device of
            # the noise-composition study made the run binding fail before any task ran)
            from qemx.selection import binding as BD
            for dev in sorted({s.device for s in specs}):
                assert len(BD.device_digest(dev)) == 16, (f, dev)
            n += 1
    assert n >= 10


@pytest.mark.parametrize("mut,msg", [
    (dict(analysis_typo=1), "analysis_typo"),
    (dict(tasks=dict(widths=[3], families={"ghz": [{"thetaa": 1.0}]})), "thetaa"),
    (dict(tasks=dict(widths=[3], families={"ghz": [{}]}, devices=["fake_fez@zz=2"])), "zz"),
    (dict(tasks=dict(widths=[3], families={"ghz": [{}]}, variants_per_base="some")), "variants_per_base"),
    (dict(learning_designs={"mode": "fresh", "per_half": 0}), "learning_designs")])
def test_unknown_or_invalid_parameters_are_refused_with_their_location(mut, msg):
    from qemx.selection import design as DS
    with pytest.raises(DS.ConfigError) as e:
        DS.validate(_tiny_cfg(**mut))
    assert msg in str(e.value) or "does not satisfy" in str(e.value)


def test_invalid_cases_are_errors_not_silently_dropped():
    from qemx.selection import task as TK
    with pytest.raises(ValueError, match="does not take"):
        TK.make_spec("hea", 3, layers=1, seed_angle=2)


# ------------------------------------------------------------------ R3, R4: canonical identity
def test_logical_identity_ignores_spelling_and_the_transpiler_seed():
    from qemx.selection import ancestry as AY
    from qemx.selection import task as TK
    a = TK.make_spec("hea", 4, layers=2)
    b = TK.make_spec("hea", 4, layers=2, angle_seed=0)
    c = TK.make_spec("hea", 4, layers=2, seed=5)                          # transpiler seed (D24)
    d = TK.make_spec("hea", 4, layers=2, angle_seed=1)
    assert a.base_id != b.base_id                                          # written differently …
    assert AY.logical_id(a) == AY.logical_id(b) == AY.logical_id(c)        # … the same logical circuit
    assert AY.logical_id(d) != AY.logical_id(a)
    v = TK.make_spec("hea", 4, layers=2, observable="z_all", device="fake_kyiv")
    assert AY.logical_id(v) == AY.logical_id(a)                            # variants share it


def test_full_study_counts_and_pilot_overlap_are_reported_honestly():
    from qemx.selection import ancestry as AY
    from qemx.selection import design as DS
    full = DS.load(os.path.join(ROOT, "configs", "selection", "full.yaml"))
    pilot = DS.load(os.path.join(ROOT, "configs", "selection", "pilot.yaml"))
    fd, ff = DS.final_test(DS.expand(full), 0.3)
    pd_, pf = DS.final_test(DS.expand(pilot), 0.3)
    L = lambda S: {AY.logical_id(s) for s in S}                            # noqa: E731
    assert len(L(fd)) == 45 and len(L(ff)) == 18                           # not 192 development circuits
    assert len(L(ff) & L(pd_)) == 3 and len(L(ff) & L(pf)) == 9            # 12 of 18 already seen


def test_confirmation_set_shares_no_logical_circuit_with_any_other_study():
    from qemx.selection import ancestry as AY
    from qemx.selection import design as DS
    reg = AY.load()
    cfg = DS.load(os.path.join(ROOT, "configs", "selection", "confirm17.yaml"))
    AY.check_confirmation(reg, DS.expand(cfg), cfg["name"])
    pilot = DS.expand(DS.load(os.path.join(ROOT, "configs", "selection", "pilot.yaml")))
    with pytest.raises(ValueError, match="shares logical circuits"):
        AY.check_confirmation(reg, pilot[:3], "a-new-confirmation")


# ------------------------------------------------------------------ S5: weights, SBS
def test_policy_mse_weights_base_circuits_equally():
    from qemx.selection import evaluate as EV
    per = {"P": dict(log_risk=np.zeros(3), risk=np.array([1.0, 1.0, 100.0]), log_ratio=np.zeros(3),
                     accepted=np.ones(3, bool))}
    s = [d for d in EV.summarize_scores(per, np.array(["A", "A", "B"]), np.array([1, 1, 1]), reps=50)
         if d["B"] == "all"][0]
    assert s["policy_mse"] == pytest.approx(50.5) and s["policy_mse_rows"] == pytest.approx(34.0)
    assert EV.wquantile([1, 2, 100], ["A", "A", "B"], 0.5) == 2


def test_single_best_protocol_is_ranked_on_one_population_with_a_declared_fallback():
    from qemx.selection.policies import _rank_from_train
    Y = np.array([[1e-2, 1e-2], [1e-2, 1e-2], [1e-1, np.nan], [1e-1, np.nan]])     # protocol 1 infeasible on hard rows
    Y[:2, 1] = 5e-3
    F = np.isfinite(Y)
    assert _rank_from_train(Y, F, groups=np.array([0, 1, 2, 3]), fallback=0)[0] == 1
    Y2 = Y.copy()
    Y2[:2, 1] = 2e-2                                       # worse where feasible: 1.6.0's feasible-only mean could
    assert _rank_from_train(Y2, np.isfinite(Y2), groups=np.arange(4), fallback=0)[0] == 0   # still prefer it


# ------------------------------------------------------------------ R5: statement kinds, certificates
def test_statement_vocabulary_and_certificate_units():
    from qemx.selection import certify as CE
    from qemx.selection.claims import KINDS, describe
    assert {"empirical", "cv-residual", "split-conformal", "finite-sample-bound", "model-conditional",
            "post-hoc-check", "ood-warning"} <= set(KINDS)
    with pytest.raises(KeyError):
        describe("guaranteed")
    est = np.vstack([np.full(40, 0.5), 0.5 + 0.01 * np.random.default_rng(0).normal(size=40)])
    design = np.vstack([np.full(40, -1), np.arange(40) % 4])
    c = CE.certify(est, 0.5, 0, 0.05, "hoeffding", design=design, fresh=True)
    assert c["statement"] == "finite-sample-bound" and "clipped" in c["scope"]
    c2 = CE.certify(est, 0.5, 0, 0.05, "hoeffding", design=design, fresh=False)
    assert c2["units_chosen"] in (40,) and c["units_chosen"] in (4, 40)
    assert CE.regret_two_eps([0.1, 0.2], [0.1, 0.2])["statement"] == "post-hoc-check"


# ------------------------------------------------------------------ the structured risk model
def test_structured_risk_model_recovers_bias_floor_and_sampling_term():
    from qemx.selection.models import StructuredRisk, struct_fit_task
    B = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    b2, v = 3e-6, 0.8
    R = b2 + v / (0.75 * B)
    assert struct_fit_task(R, 0.75 * B) == pytest.approx((b2, v), rel=1e-6)
    rng = np.random.default_rng(0)
    X, y, t = [], [], []
    for k in range(12):
        x = rng.normal(size=2)
        bb, vv = 10 ** (-6 + 0.5 * x[0]), 10 ** (-0.2 + 0.3 * x[1])
        for Bj in B:
            X.append([x[0], x[1], math.log10(Bj)])
            y.append(math.log10(bb + vv / (0.75 * Bj) + 1e-6))
            t.append(k)
    m = StructuredRisk(0.25, 2, alpha=1e-6).fit(np.array(X), np.array(y), np.array(t))
    assert np.max(np.abs(m.predict(np.array(X)) - np.array(y))) < 0.05


# ------------------------------------------------------------------ R2: nested selective evaluation
@pytest.mark.slow
def test_nested_threshold_of_a_fold_never_depends_on_that_folds_labels():
    from qemx.selection import analysis as AN
    from qemx.selection import dataset as DT
    ds = DT.build(_pilot_dev())
    keep = np.flatnonzero(np.isin(ds.col("B"), [10000, 1000000]))
    ds = ds.subset(keep)
    r1 = AN.nested_selective(ds, "ridge", "deployable", k=4, inner_k=3, reps=50, random_draws=10,
                             fixed=dict(alpha=1.0))
    folds = AN._folds(ds, 4, 0)
    _, _, te = folds[1]
    ds2 = ds.subset(np.arange(len(ds.rows)))
    rng = np.random.default_rng(3)
    ds2.Y_sel[te] = ds2.Y_sel[te] * 10 ** rng.uniform(-2, 2, size=ds2.Y_sel[te].shape)
    ds2.Y_eval[te] = ds2.Y_eval[te] * 10 ** rng.uniform(-2, 2, size=ds2.Y_eval[te].shape)
    r2 = AN.nested_selective(ds2, "ridge", "deployable", k=4, inner_k=3, reps=50, random_draws=10,
                             fixed=dict(alpha=1.0))
    assert r1["folds"][1]["tau"] == r2["folds"][1]["tau"]                 # fold 2's τ: outer-training data only
    s = {d["policy"]: d for d in r1["summary"]}
    assert {"policy", "model", "fallback (SBS-B)", "random rejection", "SIZE-B", "SBS"} <= set(s)
    assert 0 <= s["policy"]["coverage"] <= 1


# ------------------------------------------------------------------ the deployment path
@pytest.mark.slow
def test_deployment_features_equal_the_studys_and_never_simulate_the_target(monkeypatch):
    from qemx.selection import deploy as DP
    from qemx.selection import features as FE
    from qemx.selection import task as TK
    from qemx import templates as Tm
    from qemx.methods import base as MB
    names = [f.name for f in FE.DICTIONARY if f.info == "available"]
    d = os.path.join(_pilot_dev(), "tasks")
    recs = [json.load(open(os.path.join(d, f), encoding="utf-8")) for f in sorted(os.listdir(d))
            if f.endswith(".json")][:6]

    def no_sim(*a, **k):
        raise AssertionError("the cheap path simulated a state")
    monkeypatch.setattr(Tm, "statevector", no_sim)
    seen = []
    orig = MB.Runner.run

    def spy(self, spec, *a, **k):
        seen.append(spec.name)
        return orig(self, spec, *a, **k)
    monkeypatch.setattr(MB.Runner, "run", spy)
    # 1.8.0: the 1.8.0 study's development records (with the four 1.8.0 noise descriptors) when they are present;
    # the 1.7.0 records hold the features defined then (the 1.8.0 descriptors did not exist)
    d18 = os.path.join(ROOT, "reports", "selection_v180", "dev", "tasks")
    if os.path.isdir(d18):
        recs += [json.load(open(os.path.join(d18, f), encoding="utf-8")) for f in sorted(os.listdir(d18))
                 if f.endswith(".json")][:4]
    checked = set()
    for r in recs:
        have = [n for n in names if n in r["features"]]
        v, rep = DP.features(DP.from_spec(TK.TaskSpec.from_json(r["spec"]), 100000), have, use_cache=False)
        for n in have:
            if n != "log10_budget":
                assert v[n] == pytest.approx(r["features"][n], rel=1e-12, abs=1e-15), (n, r["task_id"])
                checked.add(n)
        assert all(row["info"] == "available" for row in rep["rows"])
    assert set(seen) <= {"cal-0", "cal-1"}                                  # only the calibration circuits
    assert len(checked) >= 20


@pytest.mark.slow
def test_simulated_features_only_on_request_and_equal_to_the_studys():
    """1.7.0: the simulator-only augmentation runs only when allowed (a deployment has no ideal value; 1.7.0 fixed a
    crash there) and then reproduces the study's simulated features."""
    from qemx.selection import deploy as DP
    from qemx.selection import features as FE
    from qemx.selection import task as TK
    sim = [f.name for f in FE.DICTIONARY if f.info == "simulated"]
    d = os.path.join(_pilot_dev(), "tasks")
    recs = [json.load(open(os.path.join(d, f), encoding="utf-8")) for f in sorted(os.listdir(d)) if f.endswith(".json")]
    recs = [r for r in recs if r.get("status") == "ok" and r["spec"]["n"] == 3][:2]
    for r in recs:
        t = DP.from_spec(TK.TaskSpec.from_json(r["spec"]), 100000)
        v0, rep0 = DP.features(t, sim)
        assert all(math.isnan(v0[n]) for n in sim) and not any(x["computed"] for x in rep0["rows"])
        v1, rep1 = DP.features(t, sim, allow_simulation=True)
        for n in sim:
            assert v1[n] == pytest.approx(r["features"][n], rel=1e-9, abs=1e-12), (n, r["task_id"])
        assert all(x["info"] == "simulated" for x in rep1["rows"])


@pytest.mark.slow
def test_structural_feasibility_matches_the_study_except_degenerate_training(tmp_path):
    """1.8.0 (finding F1): structural feasibility never claims more than the study's allocation — every structurally
    feasible candidate is feasible in the study; a learning candidate is not structurally feasible (it is
    *unverified* until its design is built, qemx.selection.preflight), and the design level agrees with the study."""
    from qemx.selection import deploy as DP
    from qemx.selection import preflight as PF
    from qemx.selection import task as TK
    d = os.path.join(_pilot_dev(), "tasks")
    bad, unverified = [], 0
    for f in sorted(os.listdir(d))[::7]:
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, f), encoding="utf-8"))
        t = DP.from_spec(TK.TaskSpec.from_json(r["spec"]), 1000)
        comp = t.compile()
        for i, k in enumerate(r["protocols"]):
            for j, B in enumerate(r["budgets"]):
                a = r["allocations"][i * len(r["budgets"]) + j]
                ok, _ = DP.structural_feasibility(t, comp, k, B)
                if ok and not a["feasible"]:
                    bad.append((k, B, a.get("reason", "")))
                if k in ("CDR", "NIL"):
                    p = PF.structural(comp, k, B)
                    assert not ok and p.status in (PF.UNVERIFIED, PF.INFEASIBLE)
                    if p.status == PF.UNVERIFIED:
                        unverified += 1
                        dz = PF.design(comp, t.obs(), k, B, supplied=t.supplied_backend,
                                       seed=int(r["master_seed"]) % (2 ** 31))
                        assert (dz.status == PF.VERIFIED) == bool(a["feasible"]), (k, B, dz.to_json(), a)
                elif not ok:
                    assert not a["feasible"], (k, B)
    assert not bad and unverified > 0


@pytest.mark.slow
def test_bundle_round_trip_without_pickle_and_ui_equals_cli(tmp_path):
    from qemx.selection import dataset as DT
    from qemx.selection import deploy as DP
    from qemx.selection import task as TK
    from qemx.selection import __main__ as CLI
    ds = DT.build(_pilot_dev())
    b = DP.build_bundle(ds, np.arange(len(ds.rows)), "ridge", "deployable", str(tmp_path / "bundle"),
                        version="test", tau=0.5)
    assert b.meta["verification"]["max_abs_difference_vs_sklearn"] < 1e-9
    lb = DP.load_bundle(str(tmp_path / "bundle"))
    X = ds.X(b.features)[:50]
    assert np.allclose(lb.predict(X), b.predict(X), atol=0, rtol=0, equal_nan=True)
    # a state document (the Playground's export) → the same recommendation from the library and the CLI
    from qemx import experiment as EX
    st = EX.to_state(EX.load(os.path.join(ROOT, "examples", "experiments", "qft_known_answer.yaml")))
    task = DP.from_state(st, 100000, "fake_fez")
    rec = DP.recommend(task, lb)
    p = tmp_path / "state.json"
    p.write_text(json.dumps(st), encoding="utf-8")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        CLI.main(["recommend", "--state", str(p), "--budget", "100000", "--device", "fake_fez", "--bundle",
                  str(tmp_path / "bundle")])
    from qemx import textio as TX
    cli = TX.loads(buf.getvalue(), strict_tokens=True)       # 1.8.0: strict JSON
    for k in ("task", "choice", "model_choice", "features", "predicted_log10_risk", "feasibility", "status"):
        assert TX.dumps(cli[k], sort_keys=True, default=str) == TX.dumps(rec[k], sort_keys=True, default=str), k
    # a study task and its deployment twin resolve to one compiled experiment
    spec = TK.make_spec("ising", 4, steps=2)
    t2 = DP.from_spec(spec, 1000)
    c1, c2 = t2.compile(), TK.build(spec).compiled
    assert c1.implemented.ops == c2.implemented.ops and c1.measure_map == c2.measure_map
    # tampering is detected
    with open(tmp_path / "bundle" / "arrays.npz", "ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="digest"):
        DP.load_bundle(str(tmp_path / "bundle"))


def test_unsupported_experiments_are_reported_not_guessed():
    from qemx import experiment as EX
    from qemx.selection import deploy as DP
    st = EX.to_state(EX.load(os.path.join(ROOT, "examples", "experiments", "qft_known_answer.yaml")))
    with pytest.raises(DP.Unsupported, match="fake device"):
        DP.from_state(st, 1000)                                              # a noise injector, no device given
    st2 = EX.to_state(EX.load(os.path.join(ROOT, "examples", "experiments", "ghz_parity_hamiltonian.yaml")))
    with pytest.raises(DP.Unsupported):
        DP.from_state(st2, 1000, "fake_fez")


# ------------------------------------------------------------------ hardware: plan gate (no submission)
def _hw_plan():
    from qemx.selection import hardware as HWS
    content = dict(schema="qemx.selection.hardware-plan/1", budget=20000, device_model="fake_fez", backend="ibm_fez",
                   blocks=2, ceiling_s=600.0, reserve_s=60.0, runs=[],
                   jobs=[dict(job="t · RAW · block %d" % b, executions=20000, circuits=1, low_s=9.0, central_s=13.4,
                              high_s=17.4, reserved_s=18.0, admitted=True) for b in (1, 2)],
                   bundle=None, bundle_digest=None)
    return dict(content, digest=HWS._digest(content))


def test_hardware_execution_needs_the_authorization_of_this_exact_plan(tmp_path):
    from qemx import campaign as CG
    from qemx.selection import hardware as HWS
    p = _hw_plan()
    with pytest.raises(HWS.NotAuthorized, match="needs the owner's authorization"):
        HWS.require_authorization(p, env={})
    with pytest.raises(HWS.NotAuthorized, match="another plan"):
        HWS.require_authorization(p, env={HWS.ENV_AUTH: "0" * 64})
    bad = dict(p, budget=30000)
    with pytest.raises(HWS.NotAuthorized, match="changed after"):
        HWS.require_authorization(bad, env={HWS.ENV_AUTH: p["digest"]})
    HWS.require_authorization(p, env={HWS.ENV_AUTH: p["digest"]})
    with pytest.raises(CG.LedgerRequired):
        HWS.execute(p, None, lambda *a, **k: None, env={HWS.ENV_AUTH: p["digest"]})
    led = CG.Ledger.create(str(tmp_path / "ledger.json"), ceiling_s=600, reserve_s=60)
    prov = CG.MockProvider()
    recs = HWS.execute(p, led, lambda label, shots, circ, client_tag=None: prov.run(shots * circ, client_tag),
                       env={HWS.ENV_AUTH: p["digest"]})
    assert len(recs) == 2 and len(prov.jobs) == 2
    assert led.refresh().remaining_s() < 540.0                         # the reservations are on the ledger


def test_hardware_plan_charges_each_protocol_run_exactly_its_budget():
    """1.7.0: the plan counts the executions of a run as the sum of its circuits' shots. Merged PEC circuits carry the
    shares of every occurrence in their shots already; multiplying by the occurrence count reported PEC runs of
    hundreds of millions of shots at B = 20 000 and made a feasible plan look infeasible."""
    from qemx.selection import hardware as HWS
    cfg = {"hardware": dict(ceiling_s=600, reserve_s=60, budget=20000, blocks=1, device="fake_fez",
                            backend_name="ibm_fez", protocols=["RAW", "RO", "PEC"],
                            tasks=[dict(family="ghz", n=4, theta=1.0472)])}
    p = HWS.plan(cfg)
    runs = {r["protocol"]: r for r in p["runs"]}
    assert set(runs) == {"RAW", "RO", "PEC"}
    for r in runs.values():
        assert r["supported"] and r["shots"] == 20000, r
    assert all(j["executions"] == 20000 for j in p["jobs"])
    assert p["fits"] and not p["executed"]


def test_every_noise_composition_device_builds_a_task():
    """1.7.0: each device of the noise-composition study (one component changed at a time, a coherent fraction on the
    exact engine) builds a task on the base device's chain; the coherent-fraction devices failed to find their base
    device before this fix."""
    from qemx import device as DV
    from qemx.selection import design as DS
    from qemx.selection import task as TK
    cfg = DS.load(os.path.join(ROOT, "configs", "selection", "noisecomp17.yaml"))
    devs = sorted({s.device for s in DS.expand(cfg)})
    assert len(devs) == 10
    for d in devs:
        assert DV.base_name(d) == "fake_fez", d
        b = TK.build(TK.make_spec("ghz", 3, device=d))
        assert np.isfinite(float(b.ideal))
        if "chi=" in d:
            assert "eng=exact" in b.spec.supplied_backend


def test_a_run_bound_before_new_device_modifiers_existed_still_resumes(tmp_path):
    """1.7.0: device digests of runs bound before the modifiers g1, g2, t existed digested the older modifier set;
    such a run is the same device and resumes, while a real change of the device is still refused."""
    from qemx.selection import binding as BD
    from qemx.selection import task as TK
    specs = [TK.make_spec("ghz", 3, device="fake_fez")]
    cfg = dict(budgets=[1000], protocols=["RAW"], R=4, master_seed=1)
    rec = BD.bind(str(tmp_path), cfg, specs, specs, [], "dev")
    legacy = {d: BD.device_digest(d, keys=BD.MODIFIER_KEY_SETS[0]) for d in rec["identity"]["devices"]}
    assert all(legacy[d] != v for d, v in rec["identity"]["devices"].items())   # the conventions differ
    import json as _json
    p = tmp_path / "run.json"
    r = _json.loads(p.read_text(encoding="utf-8"))
    r["identity"]["devices"] = legacy
    p.write_text(_json.dumps(r), encoding="utf-8")
    BD.bind(str(tmp_path), cfg, specs, specs, [], "dev")                          # resumes
    assert BD.devices_equivalent("fake_fez@g2=2", BD.device_digest("fake_fez@g2=2", keys=BD.MODIFIER_KEY_SETS[0])) \
        is False                                                                  # g2 was not in the old convention
    r["identity"]["devices"] = {d: "0" * 16 for d in legacy}
    p.write_text(_json.dumps(r), encoding="utf-8")
    with pytest.raises(BD.IncompatibleRun, match="devices"):
        BD.bind(str(tmp_path), cfg, specs, specs, [], "dev")
