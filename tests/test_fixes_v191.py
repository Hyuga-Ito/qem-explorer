"""1.9.1 corrective release: regressions for the independent review of 1.9.0 (F1 stage order and nested reuse, F2
stale freezes, F3 persistent resource accounting, F4 runtime table). Process supervision (F5) is tested in
tests/test_readiness_v190.py and tests/test_pipeline_v190.py. Tests that need the pilot's per-trial arrays skip
without them, unless QEMX_REQUIRE_TRIALS names pilot_v190."""
import copy
import os
import shutil

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = os.path.join(ROOT, "reports", "pilot_v190")


def _pilot_arrays():
    d = os.path.join(PILOT, "dev", "tasks")
    ok = os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))
    if not ok:
        if "pilot_v190" in os.environ.get("QEMX_REQUIRE_TRIALS", ""):
            pytest.fail("QEMX_REQUIRE_TRIALS names pilot_v190 but its per-trial arrays are missing")
        pytest.skip("the pilot's per-trial arrays are not extracted (qem-explorer-1.9.0-trials-pilot.zip)")


def _copy_pilot(dst, analysis=True):
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(os.path.join(PILOT, "dev"), os.path.join(dst, "dev"))
    shutil.copy2(os.path.join(PILOT, "manifest.json"), os.path.join(dst, "manifest.json"))
    if analysis:
        shutil.copytree(os.path.join(PILOT, "analysis"), os.path.join(dst, "analysis"))
    for p in (os.path.join(dst, "dev", ".qemx-run.lock"),):
        if os.path.exists(p):
            os.remove(p)


def _write_cfg(path, cfg):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)       # key order matters for the expansion
    return str(path)


# ------------------------------------------------------------------ F2: what a first freeze accepts (no arrays)
def _fake(cfg, tmp_path, *, data="d" * 64, grid=None, splits=("same-distribution",), code_changes=None,
          identity=True, code=None):
    """A results.json and pooled table as an analysis of ``cfg`` would write them, without labels."""
    import pandas as pd
    from qemx.selection import run as RN
    from qemx.selection.__main__ import declared_grid
    adir = tmp_path / "analysis"
    adir.mkdir(exist_ok=True)
    comp = dict(data_digest=data)
    ident = RN.analysis_identity(cfg, comp)
    if code:
        ident = dict(ident, code=code)
    res = dict(evaluations={k: {} for k in splits}, data=dict(status="complete"))
    if identity:
        res["identity"] = ident
    if code_changes:
        res["code_changes"] = code_changes
    rows = [dict(policy=g, B="all", mean_log10_risk=-4.0 - i * 0.01) for i, g in
            enumerate(grid if grid is not None else declared_grid(cfg))]
    pd.DataFrame(rows + [dict(policy="SBS-B", B="all", mean_log10_risk=-3.9)]).to_csv(
        adir / "pooled_same-distribution.csv", index=False, encoding="utf-8")
    return comp, res, str(adir)


def _small_cfg():
    from qemx.selection import design as DS
    cfg = DS.load(os.path.join(ROOT, "configs", "selection", "smoke19.yaml"))
    cfg["analysis"] = dict(splits=["same-distribution"], feature_sets=["size_budget", "lightcone"],
                           model_families=["ridge"], bootstrap=20)
    return cfg


def test_first_freeze_accepts_the_matching_analysis_only(tmp_path):
    from qemx.selection.__main__ import _check_analysis_for_freeze
    cfg = _small_cfg()
    comp, res, adir = _fake(cfg, tmp_path)
    assert _check_analysis_for_freeze(cfg, comp, res, adir, accept=False) is None
    # changed analysis settings (the review's probe: another declared grid, no reanalysis)
    other = copy.deepcopy(cfg)
    other["analysis"]["model_families"] = ["hgb"]
    with pytest.raises(ValueError, match="analysis settings"):
        _check_analysis_for_freeze(other, comp, res, adir, accept=False)
    # other development data than the analysis was made on
    with pytest.raises(ValueError, match="development data"):
        _check_analysis_for_freeze(cfg, dict(data_digest="e" * 64), res, adir, accept=False)


def test_first_freeze_needs_every_declared_split_and_grid_configuration(tmp_path):
    from qemx.selection.__main__ import _check_analysis_for_freeze
    cfg = _small_cfg()
    comp, res, adir = _fake(cfg, tmp_path, grid=["MODEL[ridge|size_budget]"])      # lightcone missing
    with pytest.raises(ValueError, match="lacks configurations of the declared grid: MODEL\\[ridge\\|lightcone\\]"):
        _check_analysis_for_freeze(cfg, comp, res, adir, accept=False)
    cfg2 = copy.deepcopy(cfg)
    cfg2["analysis"]["splits"] = ["same-distribution", "noise"]
    comp, res, adir = _fake(cfg2, tmp_path)
    with pytest.raises(ValueError, match="declared splits not in the analysis: noise"):
        _check_analysis_for_freeze(cfg2, comp, res, adir, accept=False)


def test_code_changes_and_legacy_analyses_need_recorded_acceptance(tmp_path):
    from qemx.selection import binding as BD
    from qemx.selection.__main__ import _check_analysis_for_freeze
    cfg = _small_cfg()
    comp, res, adir = _fake(cfg, tmp_path, code="0123456789abcdef")
    with pytest.raises(ValueError, match="other code"):
        _check_analysis_for_freeze(cfg, comp, res, adir, accept=False)
    acc = _check_analysis_for_freeze(cfg, comp, res, adir, accept=True)
    assert acc["code_change"] == dict(analysis="0123456789abcdef", now=BD.code_digest())
    res["code_changes"] = [dict(recorded="0123456789abcdef", now=BD.code_digest())]   # analyze --accept-changes
    assert _check_analysis_for_freeze(cfg, comp, res, adir, accept=False) is None
    comp, res, adir = _fake(cfg, tmp_path, identity=False)                          # made before 1.9.0
    with pytest.raises(ValueError, match="records no identity"):
        _check_analysis_for_freeze(cfg, comp, res, adir, accept=False)
    assert "legacy_analysis" in _check_analysis_for_freeze(cfg, comp, res, adir, accept=True)


# ------------------------------------------------------------------ F2 on copied pilot labels (arrays needed)
def test_stale_freeze_of_a_copied_pilot_is_refused_and_a_valid_one_is_reproduced(tmp_path):
    """The review's probe: the declared grid changed to ridge|size_budget without reanalysis → refused (1.9.0 froze
    hgb|deployable18+simulated, outside the declared grid). The unchanged configuration freezes the shipped choice
    (the code change since 1.9.0 accepted explicitly and recorded); --reuse then verifies; a policy outside the
    declared grid is never chosen, even when the table contains a better one."""
    import pandas as pd
    from qemx import textio as TX
    from qemx.selection import design as DS
    from qemx.selection.__main__ import main
    _pilot_arrays()
    cfg = DS.load(os.path.join(ROOT, "configs", "selection", "pilot19.yaml"))
    out = str(tmp_path / "copy")
    _copy_pilot(out)
    stale = copy.deepcopy(cfg)
    stale["analysis"]["feature_sets"], stale["analysis"]["model_families"] = ["size_budget"], ["ridge"]
    sp = _write_cfg(tmp_path / "stale.yaml", stale)
    assert main(["freeze", sp, "--out", out, "--accept-changes"]) == 2
    assert not os.path.exists(os.path.join(out, "freeze.json"))
    # a better row of a configuration outside the declared grid is ignored
    src = os.path.join(out, "analysis", "pooled_same-distribution.csv")
    q = pd.read_csv(src, encoding="utf-8")
    extra = q[(q.B.astype(str) == "all") & (q.policy == "MODEL[hgb|deployable18]")].copy()
    extra["policy"], extra["mean_log10_risk"] = "MODEL[mlp|size_budget]", -99.0
    pd.concat([q, extra]).to_csv(src, index=False, encoding="utf-8")
    cp = _write_cfg(tmp_path / "pilot19.yaml", cfg)
    assert main(["freeze", cp, "--out", out]) == 2                       # code changed since 1.9.0: explicit
    assert main(["freeze", cp, "--out", out, "--accept-changes"]) == 0
    frz = TX.read_json(os.path.join(out, "freeze.json"))
    shipped = TX.read_json(os.path.join(PILOT, "freeze.json"))
    assert frz["policy"] == shipped["policy"] == "MODEL[hgb|deployable18+simulated]"
    assert frz["accepted"]["code_change"]["analysis"] == shipped["analysis_identity"]["code"]
    assert "MODEL[mlp|size_budget]" not in frz["declared_grid"]
    before = TX.read_text(os.path.join(out, "freeze.json"))
    assert main(["freeze", cp, "--out", out, "--reuse"]) == 0            # verified, unchanged
    assert main(["freeze", cp, "--out", out]) == 1                       # already done (unchanged exit code)
    assert TX.read_text(os.path.join(out, "freeze.json")) == before


# ------------------------------------------------------------------ F1: the shipped full-development stage order
DEV = {"schema": "qemx.selection.config/1", "name": "fullorder-probe", "master_seed": 515151, "R": 6,
       "budgets": [1000, 100000], "protocols": ["RAW", "RO", "ZNE-L", "PEC"], "final_test_fraction": 0.0,
       "tasks": {"widths": [3], "families": {"ising": [{"steps": 1, "h": 0.613}, {"steps": 1, "h": 0.427}],
                                             "hea": [{"layers": 1, "angle_seed": 7741}],
                                             "qaoa": [{"p": 1, "angle_seed": 7742}]},
                 "include_default": True, "variants_per_base": 0},
       "limits": {"task_seconds": 600, "memory_gb": 3},
       # a grid that cannot select 1.9.0's pre-freeze default (hgb|deployable+simulated); the extras that matter
       # for the order defect, as in configs/selection/fulldev19.yaml
       "analysis": {"splits": ["same-distribution"], "split_options": {"same-distribution": {"k": 2}},
                    "feature_sets": ["size_budget", "lightcone"], "model_families": ["ridge"],
                    "abstain_family": "ridge", "abstain_set": "size_budget", "bootstrap": 30, "nested_inner_k": 2,
                    "extras": ["deployed", "nested_selective", "design_diagnostics"]}}


def _shipped_order():
    with open(os.path.join(ROOT, "configs", "pipelines", "fulldev19.yaml"), encoding="utf-8") as f:
        stages = yaml.safe_load(f)["stages"]
    return stages[: stages.index("freeze-complete") + 1]


@pytest.mark.slow
def test_shipped_full_development_order_completes_with_the_frozen_configuration(tmp_path):
    """Independent review F1: with the shipped fulldev19 stage order (plan, label, analyze, freeze, nested,
    freeze-complete) and these extras, 1.9.0 evaluated hgb|deployable+simulated before the freeze, reused it after
    freezing another configuration and stopped with exit 2 at freeze-complete. Now the frozen-configuration extras are
    deferred to the 'nested' stage; a rerun verifies and keeps everything."""
    import pandas as pd
    from qemx import textio as TX
    from qemx.selection import pipeline as PL
    order = _shipped_order()
    assert order == ["plan", "label", "analyze", "freeze", "nested", "freeze-complete"]
    dev = _write_cfg(tmp_path / "dev.yaml", DEV)
    pipe = PL.load(_write_cfg(tmp_path / "pipe.yaml", {"schema": PL.SCHEMA, "name": "fullorder",
                                                        "purpose": "development", "development": dev,
                                                        "stages": order}))
    out = str(tmp_path / "run テスト")
    r = PL.Runner(pipe, out, workers=2, echo=lambda *_: None)
    assert r.run() == 0
    frz = TX.read_json(os.path.join(out, "freeze.json"))
    assert frz["policy"] in ("MODEL[ridge|size_budget]", "MODEL[ridge|lightcone]")
    assert frz["deploy_tau"] is not None
    conf = pd.read_csv(os.path.join(out, "analysis", "nested_operating.csv"), encoding="utf-8").configuration.iloc[0]
    assert conf == frz["policy"]
    res = TX.read_json(os.path.join(out, "analysis", "results.json"))
    assert res["extras"]["nested_selective"]["binding"]["configuration"] == frz["policy"]
    assert res["extras"]["nested_selective"]["binding"]["data"] == frz["dev_data_digest"]
    lines = []
    assert PL.Runner(pipe, out, workers=2, echo=lines.append).run() == 0          # unchanged rerun
    assert any("[nested] kept" in x for x in lines)
    assert TX.read_json(os.path.join(out, "freeze.json")) == frz


@pytest.mark.slow
def test_premature_nested_tables_are_set_aside_and_recomputed_for_the_frozen_configuration(tmp_path):
    """Recovery for a run that already holds nested tables of another configuration (a 1.9.0 run of fulldev19, here
    imitated by declaring deployed_family/deployed_set before the freeze): the pipeline moves them to
    analysis/stale_nested_*/, recomputes the nested evaluation for the frozen configuration, completes the freeze, and
    never changes the frozen model choice."""
    import pandas as pd
    from qemx import textio as TX
    from qemx.selection import pipeline as PL
    cfg = copy.deepcopy(DEV)
    cfg["name"] = "fullorder-recovery"
    cfg["analysis"].update(deployed_family="ridge", deployed_set="lightcone", feature_sets=["size_budget"])
    dev = _write_cfg(tmp_path / "dev.yaml", cfg)
    out = str(tmp_path / "run")
    pre = PL.load(_write_cfg(tmp_path / "pre.yaml", {"schema": PL.SCHEMA, "name": "pre", "purpose": "development",
                                                      "development": dev, "stages": ["plan", "label"]}))
    assert PL.Runner(pre, out, workers=2, echo=lambda *_: None).run() == 0
    from qemx.selection.__main__ import main
    assert main(["analyze", dev, "--out", out]) == 0          # declared ridge|lightcone evaluated before the freeze
    first = pd.read_csv(os.path.join(out, "analysis", "nested_operating.csv"), encoding="utf-8").configuration.iloc[0]
    assert first == "MODEL[ridge|lightcone]"
    pipe = PL.load(_write_cfg(tmp_path / "pipe.yaml", {"schema": PL.SCHEMA, "name": "post", "purpose": "development",
                                                        "development": dev,
                                                        "stages": ["analyze", "freeze", "nested", "freeze-complete"]}))
    assert PL.Runner(pipe, out, workers=2, echo=lambda *_: None).run() == 0
    frz = TX.read_json(os.path.join(out, "freeze.json"))
    assert frz["policy"] == "MODEL[ridge|size_budget]"
    now = pd.read_csv(os.path.join(out, "analysis", "nested_operating.csv"), encoding="utf-8").configuration.iloc[0]
    assert now == frz["policy"]
    stale = [d for d in os.listdir(os.path.join(out, "analysis")) if d.startswith("stale_nested_")]
    assert len(stale) == 1
    old = pd.read_csv(os.path.join(out, "analysis", stale[0], "nested_operating.csv"), encoding="utf-8")
    assert old.configuration.iloc[0] == "MODEL[ridge|lightcone]"
    res = TX.read_json(os.path.join(out, "analysis", "results.json"))
    assert res["stale_nested"][0]["moved_to"] == stale[0]          # the audit record survives the re-analysis
    assert res["extras"]["nested_selective"]["binding"]["configuration"] == frz["policy"]


# ------------------------------------------------------------------ F3: persistent consumed-resource ledger
TINY = {"schema": "qemx.selection.config/1", "name": "ledger-probe", "master_seed": 123456, "R": 2,
        "budgets": [1000], "protocols": ["RAW", "RO"], "final_test_fraction": 0.0,
        "tasks": {"widths": [2], "families": {"ising": [{"steps": 1, "h": 0.712345}, {"steps": 1, "h": 0.512345}]},
                  "include_default": True, "variants_per_base": 0},
        "limits": {"task_seconds": 120, "memory_gb": 3}}


def _tiny(tmp_path, name, **limits):
    c = copy.deepcopy(TINY)
    c["limits"] = dict(c["limits"], **limits)
    return _write_cfg(tmp_path / name, c)


def _record(d, tid, seconds, mtime, sub="tasks", name=None):
    import json
    p = os.path.join(d, sub, (name or tid) + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"task_id": tid, "status": "ok", "seconds": seconds}, f)
    os.utime(p, (mtime, mtime))
    return p


def _current_record_seconds(d):
    import json
    t = os.path.join(d, "tasks")
    return sum(json.load(open(os.path.join(t, f), encoding="utf-8")).get("seconds") or 0.0
               for f in os.listdir(t) if f.endswith(".json"))


def test_consumed_ledger_counts_every_attempt_once_including_legacy_and_imported_records(tmp_path):
    """The migration and no-double-counting rules of ``integrity.consumed_seconds`` on a synthetic labels directory."""
    from qemx.selection import integrity as IN
    d = str(tmp_path / "dev")
    _record(d, "a", 10.0, 1000)                      # labeled before 1.9.0 (or imported): no attempt entry
    _record(d, "b", 5.0, 1000)
    c = IN.consumed_seconds(d)
    assert c["total_seconds"] == pytest.approx(15.0) and c["records_without_attempts"] == 2 and c["attempts"] == 0
    # a 1.9.0 retry of b: the old record was copied to failed_attempts/ without an entry; the new one has an entry
    _record(d, "b", 5.0, 1000, sub="failed_attempts", name="b.attempt1")
    _record(d, "b", 7.0, 2000)
    IN.append_attempt(d, dict(task_id="b", status="ok", seconds=7.5))
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(10.0 + 5.0 + 7.5)
    # quarantine keeps a's seconds in the ledger itself (once), so deleting the quarantine folder loses nothing
    IN.quarantine(d, ["a"], "test")
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(22.5)
    shutil.rmtree(os.path.join(d, "quarantine"))
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(22.5)
    assert [x["status"] for x in IN.attempts(d) if x["task_id"] == "a"] == ["ledger-record"]
    IN.keep_in_ledger(d, "a", "again")
    IN.keep_in_ledger(d, "b", "again")               # b's current record has an entry: nothing is added
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(22.5)
    # an interrupted task counts its time; when its child had written the record, that record is not added again
    IN.append_attempt(d, dict(task_id="c", status="interrupted", seconds=3.0, record_written=False))
    _record(d, "e", 4.0, 3000)
    IN.append_attempt(d, dict(task_id="e", status="interrupted", seconds=4.2, record_written=True))
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(22.5 + 3.0 + 4.2)
    # a skipped (checkpoint) attempt writes no record, so it does not hide a legacy record
    _record(d, "f", 6.0, 1000)
    IN.append_attempt(d, dict(task_id="f", status="skipped", seconds=0.5))
    c = IN.consumed_seconds(d)
    assert c["total_seconds"] == pytest.approx(29.7 + 6.5) and c["records_without_attempts"] == 2   # b.attempt1, f
    assert IN.consumed_seconds(d, task_ids=["b"])["total_seconds"] == pytest.approx(12.5)
    assert IN.completion(d, check_arrays="none")["consumed"]["total_seconds"] == pytest.approx(c["total_seconds"])


def test_jobs_running_at_an_interruption_are_reported_with_their_time(tmp_path):
    """``supervise.run_jobs(on_interrupt=...)``: a job still running when the parent is interrupted is stopped and
    reported with its elapsed seconds (the labeling command records it as an ``interrupted`` attempt)."""
    from qemx.selection import supervise as SV
    hb = str(tmp_path / "hb.txt")
    seen = []

    def done(o):
        raise KeyboardInterrupt                      # Ctrl+C arrives while job "b" runs

    with pytest.raises(KeyboardInterrupt):
        SV.run_jobs([SV.Job("a", "tests._supervise_targets:sleep", (1.0,)),
                     SV.Job("b", "tests._supervise_targets:heartbeat", (hb, 120))], workers=2, on_done=done,
                    on_interrupt=lambda key, sec: seen.append((key, sec)))
    assert [k for k, _ in seen] == ["b"] and seen[0][1] > 0.5


@pytest.mark.slow
def test_cpu_ceiling_starts_from_the_consumed_ledger_across_invocations(tmp_path, monkeypatch):
    """Real labeling: failed attempts replaced by a retry, a quarantined record and an interrupted invocation stay
    consumed; the CPU ceiling of a later invocation starts from that ledger (1.9.0 started from the current records,
    so it labeled again under a cap that was already used up)."""
    import json
    from qemx.selection.__main__ import main
    from qemx.selection import integrity as IN, run as RN
    out = str(tmp_path / "run")
    d = os.path.join(out, "dev")
    assert main(["label", _tiny(tmp_path, "t.yaml", task_seconds=0.001), "--out", out]) == 1       # both time out
    assert main(["label", _tiny(tmp_path, "r.yaml"), "--out", out, "--retry-failed"]) == 0          # the retry
    att = IN.attempts(d)
    assert [a["status"] for a in att] == ["timeout", "timeout", "ok", "ok"]
    c = IN.consumed_seconds(d)
    assert c["total_seconds"] == pytest.approx(sum(a["seconds"] for a in att))       # copies not counted again
    assert c["records_without_attempts"] == 0 and c["by_status"]["timeout"] > 0
    # quarantine one finished task (as --quarantine-corrupted does): its time stays consumed
    tid = sorted(f[:-5] for f in os.listdir(os.path.join(d, "tasks")) if f.endswith(".json"))[0]
    before = IN.consumed_seconds(d)["total_seconds"]
    current = _current_record_seconds(d)
    IN.quarantine(d, [tid], "test: corrupted on purpose")
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(before)
    gap = before - _current_record_seconds(d)
    assert gap > 0.5 and current > _current_record_seconds(d)
    # a cap between the current records' seconds and the consumed seconds: admission refuses (consumed + remaining
    # exceeds it), and with a recorded override the monitored ceiling starts from the ledger, so no task starts
    cap_h = (_current_record_seconds(d) + before) / 2 / 3600
    capped = _tiny(tmp_path, "cap.yaml", cpu_hours=cap_h)
    assert main(["plan", capped, "--out", out]) == 3
    assert main(["label", capped, "--out", out]) == 2
    assert main(["label", capped, "--out", out, "--override-limits", "test: the ceiling itself"]) == 1
    s = json.load(open(os.path.join(d, "label_summary.json"), encoding="utf-8"))
    assert s["stopped"] and "cpu_hours" in s["stopped"] and tid in s["run"]["missing"]
    assert not os.path.exists(os.path.join(d, "tasks", tid + ".json"))
    # an interrupted invocation (Ctrl+C one second after the task started) keeps the running task's time
    import time
    from qemx.selection import design as DS, supervise as SV
    orig = SV.run_jobs
    calls = []

    def run_then_interrupt(jobs, workers=1, on_done=None, admit=None, on_interrupt=None):
        def ctrl_c_after_start(running_seconds):
            calls.append(running_seconds)
            if len(calls) > 1:                       # asked again while the first task runs
                time.sleep(1.0)
                raise KeyboardInterrupt
            return True
        pad = SV.Job("pad", "tests._supervise_targets:sleep", (60,))        # never starts
        return orig(list(jobs) + [pad], workers=2, on_done=on_done, admit=ctrl_c_after_start,
                    on_interrupt=on_interrupt)
    monkeypatch.setattr(SV, "run_jobs", run_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        RN.label(DS.load(_tiny(tmp_path, "i.yaml")), out, progress=lambda *_: None)
    monkeypatch.setattr(SV, "run_jobs", orig)
    inter = [a for a in IN.attempts(d) if a["status"] == "interrupted"]
    assert len(inter) == 1 and inter[0]["task_id"] == tid and inter[0]["seconds"] > 0
    after_interrupt = IN.consumed_seconds(d)["total_seconds"]
    assert after_interrupt == pytest.approx(before + inter[0]["seconds"])
    assert not os.path.exists(os.path.join(d, ".qemx-run.lock"))
    # resume with a sufficient cap: completes; the ledger only grows
    assert main(["label", _tiny(tmp_path, "ok.yaml", cpu_hours=10), "--out", out]) == 0
    final = IN.consumed_seconds(d)
    if inter[0]["record_written"]:                   # the child had finished just before the stop: nothing to redo
        assert final["total_seconds"] == pytest.approx(after_interrupt)
    else:
        assert final["total_seconds"] > after_interrupt
    assert final["total_seconds"] == pytest.approx(sum(a["seconds"] for a in IN.attempts(d)))


# ------------------------------------------------------------------ F4: cost table
def test_cost_table_wall_time_agrees_with_its_caption_on_any_computer(monkeypatch):
    """1.9.0 printed CPU-hours / min(7, this computer's CPUs) under the caption "7 workers" (4.66 h = 9.32 / 2 on the
    generating computer). The wall time is now a declared scenario, computed for it, and the same on every computer."""
    import re
    if not os.path.isdir(os.path.join(PILOT, "dev", "tasks")):
        pytest.skip("no pilot records")
    from qemx.selection import resources as RS, tables19 as T
    real = RS.system
    values = []
    for cpus in (2, 64):
        monkeypatch.setattr(RS, "system", lambda d=".", _c=cpus: dict(real(d), cpus=_c))
        tab = T.s6_costs(dict(pilot=PILOT, root=ROOT, v18=""))
        rows = {n.get("item"): (r, n) for r, n in zip(tab.rows, tab.numeric)}
        cells, num = rows["fulldev19_wall_hours_scenario"]
        cpu_cells, cpu = rows["fulldev19_cpu_hours"]
        k = int(re.search(r"(\d+) workers", cells[0]).group(1))
        assert k == num["workers"] == T.SCENARIO_WORKERS and "CPU-hours / %d" % k in cells[2]
        assert num["value"] == pytest.approx(cpu["value"] / k) and cells[1] == "$%.1f$ h" % (cpu["value"] / k)
        assert "hypothetical %d-core" % T.SCENARIO_CORES in cells[2] and "%d-core" % T.SCENARIO_CORES in tab.note
        assert "conditional estimate" in cpu_cells[2] and "Not a proven upper bound" in cpu_cells[2]
        fits = rows["fulldev19_analysis_fits"][1]
        assert fits["value"] > fits["pilot_fits"] and fits["row_fits"] / fits["pilot_row_fits"] > fits["rows"] / \
            fits["pilot_rows"]                       # fit-count growth beyond row scaling (independent review §4)
        values.append(num["value"])
    assert values[0] == values[1]


def test_analysis_workload_counts_the_declared_split_parts():
    """Split parts as the independent review enumerated them from the expanded tasks (design_audit.json)."""
    from qemx.selection import design as DS, resources as RS
    w = RS.analysis_workload(DS.load(os.path.join(ROOT, "configs", "selection", "fulldev19.yaml")))
    assert {k: v["parts"] for k, v in w["splits"].items()} == {
        "same-distribution": 5, "noise": 35, "device-base": 15, "joint": 43, "width": 1, "budget-extrap": 5}
    assert w["rows"] == 2460 and w["splits"]["same-distribution"]["model_policies_per_part"] == 26
    p = RS.analysis_workload(DS.load(os.path.join(ROOT, "configs", "selection", "pilot19.yaml")))
    assert {k: v["parts"] for k, v in p["splits"].items()} == {"same-distribution": 5, "noise": 17,
                                                               "device-base": 9, "joint": 1}


# ------------------------------------------------------------------ §4: one-circuit test strata
def test_strata_with_one_or_few_test_circuits_are_marked_descriptive():
    """A held-out-noise or joint part with one independent test circuit has a degenerate bootstrap over circuits."""
    import numpy as np
    from qemx.selection import evaluate as EV
    per = {"P": dict(log_risk=np.array([0.1, 0.2, 0.3]), risk=np.ones(3), log_ratio=np.zeros(3),
                     accepted=np.ones(3, bool))}
    one = [d for d in EV.summarize_scores(per, np.array(["A", "A", "A"]), np.array([1, 2, 2]), reps=50)
           if d["B"] == "all"][0]
    assert one["base_circuits"] == 1 and one["inference"].startswith("descriptive")
    few = EV.summarize_scores(per, np.array(["A", "B", "C"]), np.array([1, 1, 1]), reps=50)[0]
    assert "descriptive" in few["inference"] and "3" in few["inference"]
    many = EV.summarize_scores(dict(P=dict(log_risk=np.zeros(6), risk=np.ones(6), log_ratio=np.zeros(6),
                                           accepted=np.ones(6, bool))), np.array(list("ABCDEF")), np.ones(6), reps=50)
    assert many[0]["inference"] == "bootstrap over 6 base circuits"


def test_pooling_only_the_evaluated_splits_leaves_the_frozen_table_untouched(tmp_path):
    """Found while testing the recovery of a 1.9.0 fulldev run: the nested stage's extras-only analysis re-pooled
    every split, so a pooled table produced by another version (the one the freeze was chosen on) was rewritten and
    the freeze no longer verified. Only the splits evaluated now are re-pooled."""
    import hashlib
    import pandas as pd
    src = os.path.join(PILOT, "analysis")
    if not os.path.isdir(src):
        pytest.skip("no pilot analysis")
    from qemx.selection import run as RN
    adir = str(tmp_path / "analysis")
    shutil.copytree(src, adir)

    def digests():
        return {f: hashlib.sha256(open(os.path.join(adir, f), "rb").read()).hexdigest() for f in os.listdir(adir)
                if os.path.isfile(os.path.join(adir, f))}
    before = digests()
    assert RN.pool(adir, reps=20, only=[])["pooled"] == []
    assert digests() == before                                   # an extras-only analysis pools nothing
    RN.pool(adir, reps=20, only=["joint"])
    after = digests()
    changed = {f for f in before if before[f] != after.get(f)}
    assert changed <= {"pooled_joint.csv", "optimism.csv"}
    opt = pd.read_csv(os.path.join(adir, "optimism.csv"), encoding="utf-8")
    old = pd.read_csv(os.path.join(src, "optimism.csv"), encoding="utf-8")
    assert sorted(opt.split.unique()) == sorted(old.split.unique())       # the other splits' rows are kept
    assert len(opt) == len(old)



# ------------------------------------------------------------------ independent debugging of 1.9.1 (Sonnet, Fable)
@pytest.mark.slow
def test_nested_stage_moves_nothing_it_cannot_recompute_and_keeps_legacy_tables_of_the_frozen_configuration(tmp_path):
    """The 'nested' stage refuses before a freeze and before moving anything when the re-analysis would be refused
    (other code without --accept-changes); tables made before 1.9.1 (no binding) that evaluate the frozen
    configuration are kept, as `freeze --complete` accepts them."""
    import json
    from qemx import textio as TX
    from qemx.selection import pipeline as PL
    from qemx.selection.__main__ import main
    dev = _write_cfg(tmp_path / "dev.yaml", DEV)
    out = str(tmp_path / "run")

    def pipe(name, stages):
        return PL.load(_write_cfg(tmp_path / (name + ".yaml"), {"schema": PL.SCHEMA, "name": name,
                                                                 "purpose": "development", "development": dev,
                                                                 "stages": stages}))
    assert PL.Runner(pipe("a", ["plan", "label", "analyze"]), out, workers=2, echo=lambda *_: None).run() == 0
    with pytest.raises(PL.PipelineError, match="freeze"):          # --only nested on an unfrozen run
        PL.Runner(pipe("n", ["nested"]), out, workers=2, echo=lambda *_: None).st_nested()
    assert not [d for d in os.listdir(os.path.join(out, "analysis")) if d.startswith("stale_nested_")]
    assert main(["analyze", dev, "--out", out, "--splits=", "--extras", "deployed"]) == 2      # explicit: refused
    assert PL.Runner(pipe("f", ["freeze", "nested"]), out, workers=2, echo=lambda *_: None).run() == 0
    adir = os.path.join(out, "analysis")
    rj = os.path.join(adir, "results.json")
    # imitate 1.9.0 tables of the frozen configuration: no binding
    res = TX.read_json(rj)
    for k in ("deployed", "nested_selective"):
        res["extras"][k].pop("binding")
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(res, f)
    before = sorted(os.listdir(adir))
    lines = []
    assert PL.Runner(pipe("g", ["nested"]), out, workers=2, echo=lines.append).run() == 0
    assert any("made before 1.9.1" in x for x in lines) and sorted(os.listdir(adir)) == before
    # other code without --accept-changes: refused with nothing moved
    res = TX.read_json(rj)
    res["identity"]["code"] = "0000000000000000"
    res["extras"]["nested_selective"]["binding"] = dict(configuration="MODEL[other|x]")
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(res, f)
    r = PL.Runner(pipe("h", ["nested"]), out, workers=2, echo=lambda *_: None)
    with pytest.raises(PL.PipelineError, match="accept-changes"):
        r.st_nested()
    assert sorted(os.listdir(adir)) == before


def test_moving_stale_tables_stops_cleanly_when_a_file_cannot_be_moved(tmp_path, monkeypatch):
    """Windows: a table open in another program cannot be moved; the stage stops with a PipelineError, the note is
    already written and results.json is unchanged."""
    import json
    import shutil as SH
    from qemx.selection import pipeline as PL
    adir = tmp_path / "run" / "analysis"
    adir.mkdir(parents=True)
    for f in ("nested_operating.csv", "nested_summary.csv", "selective_rows.csv", "deployed_rows.csv"):
        (adir / f).write_text("x\n1\n", encoding="utf-8")
    (adir / "results.json").write_text(json.dumps({"extras": {"deployed": {}, "nested_selective": {}}}),
                                       encoding="utf-8")
    real, calls = SH.move, []

    def flaky(a, b):
        calls.append(a)
        if len(calls) == 2:
            raise PermissionError("in use")
        return real(a, b)
    monkeypatch.setattr(SH, "move", flaky)
    with pytest.raises(PL.PipelineError, match="close any program"):
        PL.archive_stale_nested(str(tmp_path / "run"), ["test"])
    d = [x for x in os.listdir(adir) if x.startswith("stale_nested_")][0]
    assert os.path.exists(adir / d / "README.txt")
    assert json.loads((adir / "results.json").read_text(encoding="utf-8")) == {
        "extras": {"deployed": {}, "nested_selective": {}}}


def test_a_retry_copy_is_not_a_second_attempt(tmp_path):
    """Fable's D1: --retry-failed copies the failed record to failed_attempts/ before relabeling; the copy and the
    record are one attempt until a new record replaces it, and a retry that relabels nothing adds no second copy."""
    from qemx.selection import integrity as IN, run as RN
    d = str(tmp_path / "dev")
    _record(d, "a", 1.0, 1000)
    IN.append_attempt(d, dict(task_id="a", status="timeout", seconds=1.0))
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(1.0)
    RN._keep_failed_copy(d, "a")
    RN._keep_failed_copy(d, "a")                     # a second retry that wrote nothing
    assert os.listdir(os.path.join(d, "failed_attempts")) == ["a.attempt1.json"]
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(1.0)
    _record(d, "a", 2.0, 2000)                       # the retry's record replaces it
    IN.append_attempt(d, dict(task_id="a", status="ok", seconds=2.1))
    c = IN.consumed_seconds(d)
    assert c["total_seconds"] == pytest.approx(3.1) and c["records_without_attempts"] == 0


@pytest.mark.slow
def test_retries_under_an_exhausted_cap_do_not_inflate_the_ledger(tmp_path):
    """Fable's D1 reproduction: timeouts, then --retry-failed under a used-up cpu cap (twice; nothing runs), then a
    normal retry: the ledger equals the attempts' seconds at every step."""
    from qemx.selection.__main__ import main
    from qemx.selection import integrity as IN
    out = str(tmp_path / "run")
    d = os.path.join(out, "dev")
    assert main(["label", _tiny(tmp_path, "t.yaml", task_seconds=0.001), "--out", out]) == 1
    spent = IN.consumed_seconds(d)["total_seconds"]
    capped = _tiny(tmp_path, "c.yaml", cpu_hours=spent / 2 / 3600)
    for _ in range(2):
        assert main(["label", capped, "--out", out, "--retry-failed", "--override-limits", "test"]) == 1
        assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(spent)
    assert len(os.listdir(os.path.join(d, "failed_attempts"))) == 2      # one copy per task
    assert main(["label", _tiny(tmp_path, "r.yaml"), "--out", out, "--retry-failed"]) == 0
    assert IN.consumed_seconds(d)["total_seconds"] == pytest.approx(sum(a["seconds"] for a in IN.attempts(d)))


def test_regime_study_and_profile_keep_a_model_before_any_freeze(tmp_path):
    """Sonnet's D1: the regime analysis and `profile` unpacked deployed_configuration, which is None before a freeze
    since 1.9.1; they keep 1.9.0's model (hgb|deployable+simulated unless declared)."""
    from qemx.selection import run as RN
    assert RN.deployed_configuration(str(tmp_path), {}) is None
    assert RN.deployed_configuration_or_default(str(tmp_path), {}) == ("hgb", "deployable+simulated")
    assert RN.deployed_configuration_or_default(str(tmp_path), {"deployed_family": "ridge"}) == (
        "ridge", "deployable+simulated")
    src = open(RN.__file__, encoding="utf-8").read()
    assert "fam, sname = deployed_configuration(run_dir, a)" not in src


@pytest.mark.slow
def test_a_pipeline_override_naming_frozen_extras_defers_them_too(tmp_path):
    """Sonnet's re-check N1: a pipeline file whose analyze override lists deployed/nested_selective (list or string)
    stopped at 'analyze' once explicit premature extras were refused; the pipeline defers them like the
    configuration's own."""
    import pandas as pd
    from qemx import textio as TX
    from qemx.selection import pipeline as PL
    dev = _write_cfg(tmp_path / "dev.yaml", DEV)

    def run(name, override, out, stages):
        pipe = PL.load(_write_cfg(tmp_path / (name + ".yaml"), {
            "schema": PL.SCHEMA, "name": name, "purpose": "development", "development": dev,
            "analyze": {"extras": override}, "stages": stages}))
        lines = []
        assert PL.Runner(pipe, out, workers=2, echo=lines.append).run() == 0, name
        assert any("deferred to the 'nested' stage" in x for x in lines), name
        frz = TX.read_json(os.path.join(out, "freeze.json"))
        conf = pd.read_csv(os.path.join(out, "analysis", "nested_operating.csv"), encoding="utf-8").configuration
        assert conf.iloc[0] == frz["policy"], name
    a = str(tmp_path / "list")
    run("list", ["design_diagnostics", "deployed", "nested_selective"], a,
        ["plan", "label", "analyze", "freeze", "nested", "freeze-complete"])
    b = str(tmp_path / "string")                     # the string form, on a copy of the same labels
    shutil.copytree(os.path.join(a, "dev"), os.path.join(b, "dev"))
    run("string", "deployed,nested_selective", b, ["label", "analyze", "freeze", "nested"])
