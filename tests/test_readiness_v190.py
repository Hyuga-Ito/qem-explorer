"""1.9.0 local-readiness regressions (second review N1–N8, independent review R1/R8).

Each test reproduces the failure mode the review observed on 1.8.0 and checks the corrected behaviour through the
same entry point a user calls (the command line ``main`` or the module function). Labeling here uses a tiny
configuration (one or two width-2/3 tasks, RAW and RO, two trials per half) so that real supervised child processes,
real timeouts and real arrays are exercised in seconds."""
import copy
import json
import os
import shutil

import numpy as np
import pytest
import yaml

from qemx.selection.__main__ import main

TINY = {"schema": "qemx.selection.config/1", "name": "readiness-probe", "master_seed": 123456, "R": 2,
        "budgets": [1000], "protocols": ["RAW", "RO"], "final_test_fraction": 0.0,
        "tasks": {"widths": [2], "families": {"ising": [{"steps": 1, "h": 0.712345}, {"steps": 1, "h": 0.512345}]},
                  "include_default": True, "variants_per_base": 0},
        "limits": {"task_seconds": 120, "memory_gb": 3}}


def _cfg(tmp_path, name="c.yaml", **over):
    c = copy.deepcopy(TINY)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k] = dict(c[k], **v)
        else:
            c[k] = v
    p = tmp_path / name
    p.write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    return str(p), c


def _summary(out, stage="dev"):
    return json.load(open(os.path.join(out, stage, "label_summary.json"), encoding="utf-8"))


def _tasks(out, stage="dev"):
    d = os.path.join(out, stage, "tasks")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


# ------------------------------------------------------------------ N1: truthful completion and exit codes
def test_timeout_is_retained_on_resume_and_retry_recovers(tmp_path):
    cfg, _ = _cfg(tmp_path, limits={"task_seconds": 0.001})
    out = str(tmp_path / "run")
    assert main(["label", cfg, "--out", out]) == 1
    s = _summary(out)
    assert s["status"] == "incomplete" and s["errors"] == 2 and s["run"]["failed"] == {"timeout": _tasks(out)}
    # an unchanged second invocation: 1.8.0 reported tasks=0, errors=0 and exited 0
    assert main(["label", cfg, "--out", out]) == 1
    s = _summary(out)
    assert s["tasks"] == 0 and s["errors"] == 0 and sorted(s["retained_failures"]) == _tasks(out)
    assert s["run"]["n_failed"] == 2 and s["exit_code"] == 1
    recs = [json.load(open(os.path.join(out, "dev", "tasks", t + ".json"), encoding="utf-8")) for t in _tasks(out)]
    assert {r["status"] for r in recs} == {"timeout"}                         # the failed records are kept
    att = [json.loads(x) for x in open(os.path.join(out, "dev", "attempts.jsonl"), encoding="utf-8")]
    assert [a["status"] for a in att] == ["timeout", "timeout"]
    # raise the limit (operational, not scientific) and retry: the run completes; the failed attempt stays recorded
    cfg2, _ = _cfg(tmp_path, name="c2.yaml", limits={"task_seconds": 120})
    assert main(["label", cfg2, "--out", out, "--retry-failed"]) == 0
    s = _summary(out)
    assert s["status"] == "complete" and s["run"]["completed"] == 2 and s["run"]["n_failed"] == 0
    assert len(os.listdir(os.path.join(out, "dev", "failed_attempts"))) == 2
    att = [json.loads(x) for x in open(os.path.join(out, "dev", "attempts.jsonl"), encoding="utf-8")]
    assert [a["status"] for a in att] == ["timeout", "timeout", "ok", "ok"]
    # full unchanged resume: nothing relabeled, no child started, exit 0
    mt = {t: os.path.getmtime(os.path.join(out, "dev", "tasks", t + ".npz")) for t in _tasks(out)}
    assert main(["label", cfg2, "--out", out]) == 0
    s = _summary(out)
    assert s["skipped"] == s["ok"] == 2 and s["per_task_seconds"] == [] and s["status"] == "complete"
    assert {t: os.path.getmtime(os.path.join(out, "dev", "tasks", t + ".npz")) for t in _tasks(out)} == mt
    assert main(["status", "--out", out]) == 0


def test_missing_and_corrupted_arrays_stop_labeling_until_quarantined(tmp_path):
    cfg, _ = _cfg(tmp_path)
    out = str(tmp_path / "run")
    assert main(["label", cfg, "--out", out]) == 0
    t0, t1 = _tasks(out)
    os.remove(os.path.join(out, "dev", "tasks", t0 + ".npz"))                    # missing array
    with open(os.path.join(out, "dev", "tasks", t1 + ".npz"), "ab") as f:        # changed array
        f.write(b"x")
    assert main(["status", "--out", out]) == 1
    from qemx.selection import integrity as IN
    c = IN.completion(os.path.join(out, "dev"))
    assert {x["task_id"] for x in c["corrupted"]} == {t0, t1} and not c["complete"]
    assert any("missing" in x["why"] for x in c["corrupted"]) and any("digest" in x["why"] for x in c["corrupted"])
    assert main(["label", cfg, "--out", out]) == 2                              # never overwritten silently
    assert os.path.exists(os.path.join(out, "dev", "tasks", t1 + ".npz"))
    assert main(["analyze", cfg, "--out", out]) == 2                            # nor analyzed
    assert main(["label", cfg, "--out", out, "--quarantine-corrupted"]) == 0
    run = json.load(open(os.path.join(out, "dev", "run.json"), encoding="utf-8"))
    q = run["quarantined"][0]
    assert sorted(q["tasks"]) == sorted([t0, t1])
    assert os.path.exists(os.path.join(out, "dev", q["directory"], t1 + ".npz"))
    assert IN.completion(os.path.join(out, "dev"))["complete"]


def test_partial_by_limit_needs_explicit_partial_analysis_and_cannot_be_frozen(tmp_path):
    cfg, _ = _cfg(tmp_path, analysis={"splits": ["same-distribution"], "split_options": {"same-distribution": {"k": 2}},
                                      "feature_sets": ["size_budget"], "model_families": ["ridge"],
                                      "abstain_family": "ridge", "abstain_set": "size_budget", "bootstrap": 20,
                                      "extras": []})
    out = str(tmp_path / "run")
    assert main(["label", cfg, "--out", out, "--limit", "1"]) == 0
    s = _summary(out)
    assert s["status"] == "partial-by-limit" and s["run"]["completed"] == 1 and len(s["run"]["missing"]) == 1
    assert main(["analyze", cfg, "--out", out]) == 2                            # not silently a completed study
    assert main(["status", "--out", out]) == 1
    assert main(["label", cfg, "--out", out]) == 0                              # the rest; then complete
    assert main(["status", "--out", out]) == 0


def test_partial_analysis_is_marked_and_refused_by_freeze(tmp_path):
    cfg, _ = _cfg(tmp_path, analysis={"splits": ["same-distribution"], "split_options": {"same-distribution": {"k": 2}},
                                      "feature_sets": ["size_budget"], "model_families": ["ridge"],
                                      "abstain_family": "ridge", "abstain_set": "size_budget", "bootstrap": 20,
                                      "extras": []},
                  tasks={"widths": [2, 3], "families": {"ising": [{"steps": 1, "h": 0.712345}]},
                         "include_default": True, "variants_per_base": 0})
    out = str(tmp_path / "run")
    assert main(["label", cfg, "--out", out, "--limit", "1"]) == 0
    assert main(["analyze", cfg, "--out", out, "--allow-partial", "--splits", "", "--extras", ""]) == 0
    res = json.load(open(os.path.join(out, "analysis", "results.json"), encoding="utf-8"))
    assert res["data"]["status"] == "partial" and res["data"]["completed"] == 1 and res["data"]["expected"] == 2
    assert main(["freeze", cfg, "--out", out]) == 2


def test_run_returns_the_labeling_failure_and_skips_analysis(monkeypatch):
    from types import SimpleNamespace
    from qemx.selection import __main__ as CLI
    calls = []
    monkeypatch.setattr(CLI, "cmd_label", lambda a: 1)
    monkeypatch.setattr(CLI, "cmd_analyze", lambda a: calls.append("analyze") or 0)
    monkeypatch.setattr(CLI, "cmd_figures", lambda a: calls.append("figures") or 0)
    assert CLI.cmd_run(SimpleNamespace(allow_partial=False)) == 1 and calls == []
    assert CLI.cmd_run(SimpleNamespace(allow_partial=True)) == 1 and calls == ["analyze", "figures"]
    monkeypatch.setattr(CLI, "cmd_label", lambda a: 0)
    monkeypatch.setattr(CLI, "cmd_figures", lambda a: 1)
    assert CLI.cmd_run(SimpleNamespace(allow_partial=False)) == 1               # a figures failure is returned too


# ------------------------------------------------------------------ N2: admission and monitored ceilings
def test_known_cap_violation_is_refused_unless_overridden_and_recorded(tmp_path):
    cfg, _ = _cfg(tmp_path, limits={"task_seconds": 120, "memory_gb": 3, "disk_gb": 1e-12})
    out = str(tmp_path / "run")
    assert main(["plan", cfg]) == 3
    assert main(["label", cfg, "--out", out]) == 2
    assert not os.path.exists(os.path.join(out, "dev", "tasks"))              # nothing labeled
    assert main(["label", cfg, "--out", out, "--override-limits", "test: tiny cap on purpose"]) in (0, 1)
    run = json.load(open(os.path.join(out, "dev", "run.json"), encoding="utf-8"))
    adm = run["sessions"][-1]["admission"]
    assert adm["override"] == "test: tiny cap on purpose" and adm["exceeded"] == ["disk_gb"]
    # the monitored ceiling still applies: the labels directory already exceeds the cap, so no task starts
    s = _summary(out)
    assert s["stopped"] and "disk_gb" in s["stopped"] and s["status"] == "incomplete" and s["not_started"] >= 1


def test_unknown_cpu_estimate_is_admitted_as_unknown_and_the_ceiling_stops_labeling(tmp_path):
    cfg, _ = _cfg(tmp_path, limits={"task_seconds": 120, "memory_gb": 3, "cpu_hours": 1e-9})
    out = str(tmp_path / "run")
    assert main(["label", cfg, "--out", out, "--workers", "1"]) == 1
    s = _summary(out)
    assert s["admission"]["unknown"] == ["cpu_hours"] and "cpu_hours" in s["stopped"]
    assert s["run"]["completed"] == 1 and len(s["run"]["missing"]) == 1    # checkpoint-safe: one finished task
    cfg2, _ = _cfg(tmp_path, name="c2.yaml", limits={"task_seconds": 120, "memory_gb": 3, "cpu_hours": 10})
    assert main(["label", cfg2, "--out", out]) == 0                         # raised cap (operational): completes


def test_preflight_uses_measured_rates_of_the_run_and_reports_unknowns(tmp_path):
    from qemx.selection import resources as RS
    cfg, c = _cfg(tmp_path)
    pre = RS.preflight(c, None)
    assert pre["cpu"]["status"] == "unknown" and pre["wall"]["status"] == "unknown"
    assert "analysis and model search" in " ".join(pre["not_included"])
    out = str(tmp_path / "run")
    assert main(["label", cfg, "--out", out, "--limit", "1"]) == 0
    pre = RS.preflight(c, os.path.join(out, "dev"), workers=2)
    assert pre["cpu"]["status"] == "estimated" and pre["tasks_done"] == 1 and pre["tasks_remaining"] == 1
    assert pre["cpu"]["remaining_hours"] > 0 and "measured_peak_per_task_gb" in pre["memory"]
    assert RS.recommended_workers(c, dict(cpus=8, ram_gb=16.0)) == 3        # memory-bound: 3 GB limit per task


def test_single_writer_lock(tmp_path):
    from qemx.selection import integrity as IN
    d = str(tmp_path / "x")
    with IN.RunLock(d, "first"):
        with pytest.raises(IN.RunLocked):
            with IN.RunLock(d, "second"):
                pass
    with open(os.path.join(d, ".qemx-run.lock"), "w", encoding="utf-8") as f:   # a lock of a process that is gone
        json.dump(dict(pid=2 ** 22 + 12345, host=__import__("socket").gethostname()), f)
    with IN.RunLock(d, "third") as lk:
        assert lk.stale is not None


# ------------------------------------------------------------------ N4: replay bound to the manifest
@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_run")
    c = copy.deepcopy(TINY)
    c["protocols"] = ["RAW", "RO", "ZNE-L", "PEC"]
    c["tasks"] = {"widths": [3], "families": {"ising": [{"steps": 1, "h": 0.712345}, {"steps": 1, "h": 0.412345}],
                                             "hea": [{"layers": 1, "angle_seed": 9901}]},
                  "include_default": True, "variants_per_base": 0}
    p = tmp / "c.yaml"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    out = str(tmp / "run")
    assert main(["label", str(p), "--out", out, "--workers", "2"]) == 0
    return str(p), out


def test_replay_refuses_an_incomplete_population_and_records_a_declared_subset(tiny_run, tmp_path):
    from qemx.selection import deploy as DP, integrity as IN, replay as RP
    _, out = tiny_run
    t = str(tmp_path / "copy")
    shutil.copytree(os.path.join(out, "dev"), t)
    ids = _tasks(os.path.dirname(t), os.path.basename(t))
    for ext in ("json", "npz"):
        os.remove(os.path.join(t, "tasks", "%s.%s" % (ids[0], ext)))
    b = os.path.join(DP.BUNDLE_DIR, "qemx-rec-1.8.0-deployable")
    with pytest.raises(IN.IncompleteRun):
        RP.replay(t, b, budgets=[1000])
    with pytest.raises(ValueError):                                       # a subset needs its reason
        RP.replay(t, b, budgets=[1000], subset=ids[1:])
    ex, prov = [], {}
    rows = RP.replay(t, b, budgets=[1000], subset=ids[1:], subset_reason="test", excluded=ex, provenance=prov)
    assert len(rows) == len(ids) - 1 and [e["task_id"] for e in ex] == [ids[0]]
    assert prov["labels"]["subset"] and prov["labels"]["intended"] == len(ids) - 1
    assert prov["bundle"]["bundle_json_sha256"] and prov["labels"]["data_digest"]


def test_cached_labels_table_is_checked_against_the_records(tiny_run, tmp_path):
    from qemx.selection import replay as RP
    _, out = tiny_run
    t = str(tmp_path / "copy")
    shutil.copytree(os.path.join(out, "dev"), t)
    good = RP._labels(t)
    good.to_csv(os.path.join(t, "labels.csv"), index=False, encoding="utf-8")
    assert len(RP._labels(t)) == len(good)                                  # an exact derivation is accepted
    bad = good.copy()
    i = bad.mse_eval.astype(float).notna().idxmax()
    bad.loc[i, "mse_eval"] = float(bad.loc[i, "mse_eval"]) * 1.5
    bad.to_csv(os.path.join(t, "labels.csv"), index=False, encoding="utf-8")
    with pytest.raises(RP.StaleTable):
        RP._labels(t)
    fake = tmp_path / "only_csv"
    fake.mkdir()
    (fake / "labels.csv").write_text("task_id,protocol,B,feasible,mse_eval\ninvented,RAW,1000,True,123.45\n",
                                     encoding="utf-8")
    with pytest.raises(ValueError):
        RP._labels(str(fake))


def test_analysis_of_other_data_is_not_merged(tiny_run, tmp_path):
    cfgp, out = tiny_run
    run = str(tmp_path / "run")
    shutil.copytree(out, run)
    c = yaml.safe_load(open(cfgp, encoding="utf-8"))
    c["analysis"] = {"splits": ["same-distribution"], "split_options": {"same-distribution": {"k": 2}},
                     "feature_sets": ["size_budget"], "model_families": ["ridge"], "abstain_family": "ridge",
                     "abstain_set": "size_budget", "bootstrap": 20, "extras": []}
    p1 = tmp_path / "a1.yaml"
    p1.write_text(yaml.safe_dump(c), encoding="utf-8")
    assert main(["analyze", str(p1), "--out", run]) == 0
    c["analysis"]["bootstrap"] = 21
    p2 = tmp_path / "a2.yaml"
    p2.write_text(yaml.safe_dump(c), encoding="utf-8")
    assert main(["analyze", str(p2), "--out", run]) == 2                   # another analysis section: refused
    assert main(["analyze", str(p2), "--out", run, "--reset-analysis"]) == 0
    assert any(d.startswith("analysis.superseded-") for d in os.listdir(run))
    assert main(["freeze", str(p2), "--out", run]) == 0
    frz = json.load(open(os.path.join(run, "freeze.json"), encoding="utf-8"))
    assert frz["dev_data_digest"] and frz["scientific_config_digest"]
    assert main(["freeze", str(p2), "--out", run]) == 1                    # once only, verified, unchanged
    assert main(["freeze", str(p2), "--out", run, "--reuse"]) == 0
    c["limits"] = dict(c["limits"], task_seconds=999)                       # operational change: still the freeze
    p3 = tmp_path / "a3.yaml"
    p3.write_text(yaml.safe_dump(c), encoding="utf-8")
    assert main(["freeze", str(p3), "--out", run, "--reuse"]) == 0
    with open(os.path.join(run, "analysis", "pooled_same-distribution.csv"), "a", encoding="utf-8") as f:
        f.write("\n")                                                       # the table the freeze was chosen on
    assert main(["freeze", str(p3), "--out", run, "--reuse"]) == 2


# ------------------------------------------------------------------ N6: configuration semantics
@pytest.mark.parametrize("patch,needle", [
    ({"analysis": {"splits": ["same-distribution"], "feature_set": ["x"]}}, "feature_set"),
    ({"budgets": [1000, 1000]}, "non-unique"),
    ({"protocols": ["RAW", "RAW"]}, "non-unique"),
    ({"master_seed": -1}, "minimum"),
    ({"tasks": {"widths": [2], "families": {"ising": [{"steps": 1}]}, "inputs": []}}, "tasks"),
    ({"mismatch": {"scenarios": ["none"], "diag_fraktion": 0.1}}, "diag_fraktion"),
    ({"analysis": {"split_options": {"same-distribution": {"kk": 3}}}}, "kk"),
])
def test_schema_rejects_typos_duplicates_and_empty_grids(patch, needle):
    from qemx.selection import design as DS
    c = copy.deepcopy(TINY)
    c.update(patch)
    with pytest.raises(DS.ConfigError) as e:
        DS.validate(c)
    assert needle in str(e.value)


def test_semantic_checks_before_labeling(tmp_path):
    from qemx.selection import design as DS
    c = copy.deepcopy(TINY)
    c.update(R=2, learning_designs={"mode": "fresh", "per_half": 5, "seed_scheme": 2}, protocols=["RAW", "CDR"])
    assert any("per_half" in e for e in DS.check(c)["errors"])
    c = copy.deepcopy(TINY)
    c.update(final_test_fraction=1.0)
    assert any("no development tasks" in e for e in DS.check(c)["errors"])
    c = copy.deepcopy(TINY)
    c["analysis"] = {"splits": ["device"], "feature_sets": ["no_such_set"]}
    errs = DS.check(c)["errors"]
    assert any("no_such_set" in e for e in errs) and any("2 devices" in e for e in errs)
    p, _ = _cfg(tmp_path, final_test_fraction=1.0)
    assert main(["label", p, "--out", str(tmp_path / "r")]) == 2           # refused before anything runs
    assert main(["plan", p]) == 2


def test_expansion_report_shows_inserted_defaults_and_forced_inputs(tmp_path):
    from qemx.selection import design as DS
    c = copy.deepcopy(TINY)
    c["tasks"] = [{"widths": [3], "families": {"ising": [{"steps": 1}]}, "inputs": ["plus"], "observables": ["z_mid"],
                   "layouts": ["alt"], "variants_per_base": "all"},
                  {"widths": [3], "families": {"ghz": [{"theta": 0.5}]}, "inputs": ["plus"], "variants_per_base": 0}]
    rep = DS.expansion_report(c)
    assert rep["blocks"][0]["inserted_defaults"] == 1 and rep["blocks"][0]["tasks"] == 2
    assert rep["blocks"][1]["forced_input_families"] == ["ghz"]
    assert any("include_default: false" in n for n in rep["notes"])
    p = tmp_path / "tasks.csv"
    assert DS.write_task_table(c, str(p)) == 3
    import csv
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert {r["inserted_default"] for r in rows} == {"True", "False"} and all(r["logical_id"] for r in rows)
    c["expansion_policy"] = "explicit"
    errs = DS.check(c)["errors"]
    assert any("include_default" in e for e in errs) and any("family-defined input" in e for e in errs)


def test_every_shipped_yaml_is_valid_with_its_own_parser():
    from qemx.selection import formats as FM
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = FM.inventory(root)
    bad = [(r["path"], r["errors"]) for r in rows if not r["ok"]]
    assert not bad, bad
    kinds = {r["format"] for r in rows}
    assert {"selection-config", "pipeline", "plot-config", "experiment", "divergence-manifest", "phase-manifest",
            "hardware-section"} <= kinds


def test_historical_task_identities_are_unchanged():
    """The expansion of every shipped selection configuration is the 1.8.0 one (ids pinned in tests/data)."""
    from qemx import textio as TX
    from qemx.selection import design as DS
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pinned = TX.read_json(os.path.join(root, "tests", "data", "expansion_ids_v180.json"))
    for name, ids in pinned.items():
        cfg = DS.load(os.path.join(root, "configs", "selection", name))
        assert [s.task_id for s in DS.expand(cfg)] == ids, name


# ------------------------------------------------------------------ R1, N8, R8: one experiment, every path
GHZ = dict(family="ghz", n=6, theta=0.8, basis="Z", device="fake_kyiv", observable="z_mid")


def _ghz_spec(**kw):
    from qemx.selection import task as TK
    d = dict(GHZ, **kw)
    fam, n = d.pop("family"), d.pop("n")
    return TK.make_spec(fam, n, **d)


def test_state_and_study_paths_give_the_same_task_and_recommendation():
    from qemx.selection import deploy as DP
    spec = _ghz_spec()
    t_spec = DP.from_spec(spec, 1000)
    t_state = DP.from_state(DP.to_state(spec), 1000)
    assert t_spec.key() == t_state.key() and t_spec.symmetry == t_state.symmetry is not None
    assert t_state.symmetry_source == "state, validated" and t_spec.symmetry_source == "construction"
    b = DP.load_bundle(os.path.join(DP.BUNDLE_DIR, "qemx-rec-1.8.0"))
    r1, r2 = DP.recommend(t_spec, b, allow_simulation=True), DP.recommend(t_state, b, allow_simulation=True)
    for k in ("choice", "status", "model_choice", "physical_layout", "effective_device", "signature"):
        assert r1[k] == r2[k], k
    def nosec(d):
        return {k: v for k, v in d.items() if k != "seconds"}
    assert nosec(r1["preflight"]["SV"]) == nosec(r2["preflight"]["SV"])
    assert r2["preflight"]["SV"]["status"] == "verified-feasible"
    # the budget ledger of the chosen allocation is the same object on both paths
    assert r1["preflight"][r1["choice"]].get("evidence") == r2["preflight"][r2["choice"]].get("evidence")


def test_experiment_document_and_cli_resolve_the_same_experiment(tmp_path, capsys):
    from qemx import experiment as EX
    from qemx.selection import deploy as DP
    spec = _ghz_spec()
    st = DP.to_state(spec)
    doc = {"schema": "qemx.experiment/1", "name": "ghz parity", "state": st}
    p = tmp_path / "ghz実験 test.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    t_doc = DP.from_state(EX.to_state(EX.load(str(p))), 1000)
    assert t_doc.key() == DP.from_spec(spec, 1000).key()
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(st), encoding="utf-8")
    b = os.path.join(DP.BUNDLE_DIR, "qemx-rec-1.8.0")
    assert main(["recommend", "--state", str(sp), "--budget", "1000", "--bundle", b, "--allow-simulation"]) == 0
    cli = json.loads(capsys.readouterr().out)
    api = DP.recommend(DP.from_spec(spec, 1000), DP.load_bundle(b), allow_simulation=True)
    assert cli["choice"] == api["choice"] and cli["task"]["key"] == api["task"]["key"]


@pytest.mark.parametrize("change", ["basis", "input", "relabel", "edit"])
def test_a_declaration_that_does_not_hold_is_not_used(change):
    from qemx.selection import deploy as DP
    st = DP.to_state(_ghz_spec())
    if change == "basis":
        st["basis"] = "X" + st["basis"][1:]
    elif change == "input":                         # |+> on qubit 1 breaks Z1Z2 (on qubit 0 it would not)
        st["inputs"] = [st["inputs"][0], "+"] + list(st["inputs"][2:])
    elif change == "relabel":                       # bit order reversal keeps a symmetric chain's parities valid
        st["relabel"] = True
    else:
        st["circuit"]["ops"] = list(st["circuit"]["ops"]) + [dict(name="x", qubits=[0], params=[], kind="gate", tag="")]
    t = DP.from_state(st, 1000)
    if change == "relabel":
        assert t.symmetry is not None                                       # Z_iZ_{i+1} = +1 is order-symmetric
    else:
        assert t.symmetry is None and any("was not used" in x for x in t.notes)
        assert t.key() != DP.from_spec(_ghz_spec(), 1000).key()            # the declaration is part of the identity


@pytest.mark.parametrize("token", ["plus", "minus", "product"])
def test_input_kinds_round_trip_between_study_and_state(token):
    from qemx.selection import deploy as DP, task as TK
    kind = {"plus": "plus", "minus": "plus", "product": "product"}[token]
    spec = TK.make_spec("hea", 3, layers=1, angle_seed=7, inputs=kind, device="fake_fez")
    st = DP.to_state(spec)
    if token == "minus":
        st["inputs"] = ["-"] * len(st["inputs"])
    t = DP.from_state(st, 30000)
    assert list(t.inputs) == list(st["inputs"]) and t.symmetry is None
    if token != "minus":
        assert t.key() == DP.from_spec(spec, 30000).key()


def test_explicit_device_overrides_the_state_backend_and_checks_the_layout():
    from qemx import experiment as EX
    from qemx.selection import deploy as DP
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = EX.to_state(EX.load(os.path.join(root, "examples/experiments/qft_known_answer.yaml")))
    s["backend"] = "fake_fez"
    t = DP.from_state(s, 1000, "fake_kyiv")
    assert t.device == "fake_kyiv" and any("replaces" in x for x in t.notes)
    assert DP.from_state(s, 1000).device == "fake_fez"
    s2 = dict(s, compile=dict(s.get("compile") or {}, initial_layout="140,141,142,143"))
    with pytest.raises(DP.Unsupported):                       # fake_kyiv has 127 qubits: not a valid layout there
        DP.from_state(s2, 1000, "fake_kyiv")
    t3 = DP.from_state(s2, 1000)                              # fake_fez has 156 qubits
    assert t3.initial_layout == (140, 141, 142, 143)


def test_recommendation_signature_includes_simulation_permission_and_symmetry():
    from qemx.selection import deploy as DP
    b = DP.load_bundle(os.path.join(DP.BUNDLE_DIR, "qemx-rec-1.8.0"))
    st = DP.to_state(_ghz_spec())
    t = DP.from_state(st, 1000)
    a = DP.recommendation_signature(t, b, allow_simulation=False)
    assert a != DP.recommendation_signature(t, b, allow_simulation=True)
    st2 = dict(st, symmetry=None, symmetry_more="")
    assert a != DP.recommendation_signature(DP.from_state(st2, 1000), b, allow_simulation=False)


# ------------------------------------------------------------------ 1.9.0 independent verification (Sonnet) findings
def test_experiment_documents_keep_a_templates_further_parity_checks():
    from qemx import experiment as EX
    for key in ("hsp", "surface_d2"):
        st = EX.to_state({"schema": "qemx.experiment/1", "name": key, "circuit": {"template": key},
                          "backend": "fake_fez"})
        assert st["symmetry_more"], key                      # 1.8.0 wrote "" and dropped them


def test_invalid_device_layout_or_symmetry_is_unsupported_or_noted():
    from qemx import experiment as EX
    from qemx.selection import deploy as DP
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = EX.to_state(EX.load(os.path.join(root, "examples/experiments/qft_known_answer.yaml")))
    with pytest.raises(DP.Unsupported):
        DP.from_state(s, 1000, "fake_fez@zz=2")
    with pytest.raises(DP.Unsupported):
        DP.from_state(dict(s, compile=dict(s.get("compile") or {}, initial_layout="-1,2,3,4")), 1000, "fake_fez")
    st = DP.to_state(_ghz_spec())
    st["symmetry"] = [0, 0, 1]                                 # a vacuous "check" of qubit 0 with itself
    t = DP.from_state(st, 1000)
    assert t.symmetry is None and any("twice" in x for x in t.notes)


def test_recommend_cli_reports_unreadable_input_as_unsupported(tmp_path, capsys):
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    assert main(["recommend", "--state", str(p), "--budget", "1000"]) == 3
    assert "cannot read" in capsys.readouterr().out


def test_quarantined_imports_converge(tmp_path):
    from qemx.selection import binding as BD, integrity as IN
    cfg, _ = _cfg(tmp_path)
    src = str(tmp_path / "src")
    assert main(["label", cfg, "--out", src]) == 0
    out = str(tmp_path / "dst")
    assert main(["label", cfg, "--out", out, "--limit", "0"]) == 0             # bind the target run, label nothing
    ids = _tasks(src)
    BD.import_tasks(os.path.join(src, "dev"), os.path.join(out, "dev"), ids, reason="test reuse")
    with open(os.path.join(out, "dev", "tasks", ids[0] + ".npz"), "ab") as f:
        f.write(b"x")
    assert main(["label", cfg, "--out", out, "--quarantine-corrupted"]) == 0     # converges in one run
    c = IN.completion(os.path.join(out, "dev"))
    assert c["complete"] and not c["corrupted"]
    run = json.load(open(os.path.join(out, "dev", "run.json"), encoding="utf-8"))
    assert run["quarantined"][0]["removed_imports"][0]["task_id"] == ids[0]


def test_code_change_is_accepted_explicitly_and_a_frozen_analysis_is_never_archived(tiny_run, tmp_path, monkeypatch):
    from qemx.selection import binding as BD
    cfgp, out = tiny_run
    run = str(tmp_path / "run")
    shutil.copytree(out, run)
    c = yaml.safe_load(open(cfgp, encoding="utf-8"))
    c["analysis"] = {"splits": ["same-distribution"], "split_options": {"same-distribution": {"k": 2}},
                     "feature_sets": ["size_budget"], "model_families": ["ridge"], "abstain_family": "ridge",
                     "abstain_set": "size_budget", "bootstrap": 20, "extras": []}
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    assert main(["analyze", str(p), "--out", run]) == 0
    assert main(["freeze", str(p), "--out", run]) == 0
    real = BD.code_digest()
    monkeypatch.setattr(BD, "code_digest", lambda: "f" * 16 if real != "f" * 16 else "e" * 16)
    assert main(["analyze", str(p), "--out", run, "--splits", "", "--extras", "deployed"]) == 2
    assert main(["analyze", str(p), "--out", run, "--reset-analysis"]) == 2         # frozen: never archived
    assert not any(d.startswith("analysis.superseded-") for d in os.listdir(run))
    assert main(["analyze", str(p), "--out", run, "--splits", "", "--extras", "deployed", "--accept-changes"]) == 0
    from qemx import textio as TX
    res = TX.read_json(os.path.join(run, "analysis", "results.json"))
    assert res["code_changes"] and res["identity"]["code"] == real
    assert main(["freeze", str(p), "--out", run, "--reuse"]) == 0                   # the freeze still verifies


def test_mismatch_labeling_has_admission_retry_and_completion(tmp_path):
    cfg, _ = _cfg(tmp_path, limits={"task_seconds": 0.001, "memory_gb": 3},
                  tasks={"widths": [2], "families": {"ising": [{"steps": 1, "h": 0.712345}]},
                         "include_default": True, "variants_per_base": 0},
                  mismatch={"scenarios": ["none", "drift-2"], "diag_fraction": 0.02, "diag_min": 200})
    out = str(tmp_path / "run")
    assert main(["mismatch", cfg, "--out", out]) == 1                             # timeouts: incomplete
    assert main(["mismatch", cfg, "--out", out]) == 1                             # retained, still incomplete
    cfg2, _ = _cfg(tmp_path, name="c2.yaml", limits={"task_seconds": 300, "memory_gb": 3},
                   tasks={"widths": [2], "families": {"ising": [{"steps": 1, "h": 0.712345}]},
                          "include_default": True, "variants_per_base": 0},
                   mismatch={"scenarios": ["none", "drift-2"], "diag_fraction": 0.02, "diag_min": 200})
    assert main(["mismatch", cfg2, "--out", out, "--retry-failed"]) == 0
    cfg3, _ = _cfg(tmp_path, name="c3.yaml", limits={"task_seconds": 300, "memory_gb": 3, "disk_gb": 1e-12},
                   tasks={"widths": [2], "families": {"ising": [{"steps": 1, "h": 0.512345}]},
                          "include_default": True, "variants_per_base": 0},
                   mismatch={"scenarios": ["none"], "diag_fraction": 0.02, "diag_min": 200})
    assert main(["mismatch", cfg3, "--out", str(tmp_path / "run3")]) == 2         # admission applies too


def _lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _wait_for(cond, seconds):
    import time
    t_end = time.time() + seconds
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


def test_children_end_when_the_supervising_parent_is_killed(tmp_path):
    """A supervised child stops writing once its parent is killed (SIGKILL on POSIX, TerminateProcess on Windows).
    1.9.1 (independent review F5): observed through the child's own output, not through process IDs, which belong to
    another PID view in some sandboxes; the child's watchdog is the multiprocessing parent sentinel."""
    import subprocess
    import sys
    import time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hb = str(tmp_path / "heartbeat.txt")
    code = ("import sys; sys.path.insert(0, %r); from qemx.selection import supervise as SV; "
            "SV.run_jobs([SV.Job('a', 'tests._supervise_targets:heartbeat', (%r, 120))], workers=1)" % (root, hb))
    p = subprocess.Popen([sys.executable, "-c", code], cwd=root)
    try:
        assert _wait_for(lambda: _lines(hb) > 5, 60), "the child did not start"
        p.kill()                                     # the supervising parent dies without any cleanup
        p.wait(10)
        time.sleep(2.0)                              # the sentinel fires at once; allow for scheduling
        n = _lines(hb)
        time.sleep(1.5)
        assert _lines(hb) == n, "the child kept writing after its parent was killed"
    finally:
        if p.poll() is None:
            p.kill()


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
def test_ctrl_c_and_sigterm_stop_the_children_and_release_the_lock(tmp_path):
    """SIGINT (Ctrl+C) and SIGTERM to a labeling command end its supervised children and release the run lock."""
    import signal
    import subprocess
    import sys
    import time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for sig in (signal.SIGINT, signal.SIGTERM):
        hb = str(tmp_path / ("hb_%d.txt" % sig))
        d = tmp_path / ("run_%d" % sig)
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from qemx.selection import supervise as SV, integrity as IN\n"
                "from qemx.selection.__main__ import _terminate_cleanly\n"
                "_terminate_cleanly()\n"
                "with IN.RunLock(%r, 'test'):\n"
                "    SV.run_jobs([SV.Job('a', 'tests._supervise_targets:heartbeat', (%r, 120))], workers=1)\n"
                % (root, str(d), hb))
        p = subprocess.Popen([sys.executable, "-c", code], cwd=root)
        try:
            assert _wait_for(lambda: _lines(hb) > 5, 60)
            assert os.path.exists(os.path.join(str(d), ".qemx-run.lock"))
            p.send_signal(sig)
            p.wait(30)
            time.sleep(1.0)
            n = _lines(hb)
            time.sleep(1.5)
            assert _lines(hb) == n, sig
            assert not os.path.exists(os.path.join(str(d), ".qemx-run.lock")), sig
        finally:
            if p.poll() is None:
                p.kill()


def test_a_live_lock_is_never_taken_over_and_a_dead_one_is(tmp_path):
    """1.9.1 (independent review F5): lock liveness is checked in this process's own PID view; a live writer's lock is
    never declared stale (1.9.0 asked psutil only, which can look at another PID namespace)."""
    import json
    import subprocess
    import sys
    from qemx.selection import integrity as IN
    d = str(tmp_path / "run")
    os.makedirs(d)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with open(os.path.join(d, ".qemx-run.lock"), "w", encoding="utf-8") as f:
            json.dump(dict(pid=p.pid, host=__import__("socket").gethostname(), purpose="other"), f)
        with pytest.raises(IN.RunLocked):
            with IN.RunLock(d, "test"):
                pass
    finally:
        p.kill()
        p.wait(10)
    with IN.RunLock(d, "test") as lk:                # the writer has ended: its lock is stale and replaced
        assert lk.stale and lk.stale["pid"] == p.pid


def test_estimated_disk_is_not_optimistic_for_the_shipped_studies():
    from qemx.selection import design as DS, resources as RS
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, stage in (("dev18", "dev"), ("conf18", "confirm")):
        d = os.path.join(root, "reports", "selection_v180", stage, "tasks")
        if not any(f.endswith(".npz") for f in os.listdir(d)):
            pytest.skip("trial arrays not extracted")
        actual = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 2 ** 30
        pred = RS.preflight(DS.load(os.path.join(root, "configs", "selection", name + ".yaml")))["disk"]
        assert actual <= pred["labels_total_gb"] <= 1.6 * actual, (name, actual, pred["labels_total_gb"])


def test_trial_data_job_unpacks_only_the_requested_studys_attachments(tmp_path):
    """N5: a 1.9.0 release carries attachments of two studies; the optional CI job must select the requested study's
    archives (by reports/TRIAL_ATTACHMENTS.json), ignore the others and fail on unknown names or none at all. The
    selection script is taken verbatim from the workflow and executed."""
    import re
    import subprocess
    import sys
    import textwrap
    import yaml
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wf = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "ci.yml"), encoding="utf-8"))
    step = [s for s in wf["jobs"]["trial-data"]["steps"] if "TRIAL_ATTACHMENTS" in str(s.get("run", ""))][0]
    body = re.search(r"<<'PY'\n(.*?)\n\s*PY\n", step["run"], re.S).group(1)
    script = tmp_path / "select.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")

    def run(study, names):
        d = tmp_path / ("att_" + study + "_" + str(len(names)))
        d.mkdir()
        for n in names:
            (d / n).write_bytes(b"")
        return subprocess.run([sys.executable, str(script), study, str(d)], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", env=dict(os.environ, TAG="v1.9.0"))

    rel = ["qem-explorer-1.9.0-trials-1.zip", "qem-explorer-1.9.0-trials-2.zip", "qem-explorer-1.9.0-trials-pilot.zip"]
    r = run("selection_v180", rel)
    assert r.returncode == 0 and r.stdout.split() == rel[:2], (r.stdout, r.stderr)
    r = run("pilot_v190", rel)
    assert r.returncode == 0 and r.stdout.split() == rel[2:]
    assert run("selection_v170", rel).returncode != 0                       # none of this release's archives
    assert run("selection_v180", rel + ["other-trials-9.zip"]).returncode != 0   # unknown attachment


def test_every_command_refuses_an_invalid_configuration_with_exit_2(tmp_path, capsys):
    """Independent verification (F5): a schema error is a refusal (exit 2 with its reason) for every command, not a
    traceback with exit 1 (which the table of exit codes reserves for an incomplete run)."""
    import yaml
    from qemx.selection.__main__ import main
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"schema": "qemx.selection.config/1", "name": "x", "unknown_top_key": 1}),
                 encoding="utf-8")
    out = str(tmp_path / "run")
    for cmd in ("plan", "analyze", "freeze", "figures", "profile", "report", "final", "mismatch", "regimes",
                "confirm", "bundle", "label", "run"):
        argv = [cmd, str(p)] + ([] if cmd == "plan" else ["--out", out])
        if cmd == "confirm":
            argv += ["--dev-config", str(p)]
        assert main(argv) == 2, cmd
        assert "refused" in capsys.readouterr().out, cmd


def test_advice_after_a_ceiling_stop_says_to_raise_the_cap():
    """Independent verification (F10): after a monitored cap stopped labeling, 'run the same command again' would be
    refused by the same cap; the advice says to raise it."""
    from qemx.selection import integrity as IN
    c = dict(n_failed=0, failed={}, missing=["a"], corrupted=[])
    assert any("run the same label command again" in x for x in IN.advice(c))
    adv = " ".join(IN.advice(c, stopped="cpu_hours ceiling"))
    assert "raise limits.cpu_hours" in adv and "run the same label command again" not in adv


def test_omitted_splits_do_not_warn_but_an_empty_list_does():
    from qemx.selection import design as DS
    base = {"schema": "qemx.selection.config/1", "name": "w", "master_seed": 1, "R": 4, "budgets": [1000],
            "protocols": ["RAW", "RO"], "tasks": {"widths": [3], "families": {"ising": [{"steps": 1, "h": 0.5}]}}}
    w = DS.check(dict(base, analysis={"bootstrap": 20}))["warnings"]
    assert not any("analysis.splits is empty" in x for x in w)
    w = DS.check(dict(base, analysis={"splits": []}))["warnings"]
    assert any("analysis.splits is empty" in x for x in w)


def test_the_1_9_studies_are_registered_and_disjoint_from_every_other_study():
    """The prospective 1.9.0 designs use new logical circuits only (ancestry registry), and the reserved confirmation
    passes the confirmation check against everything registered before it."""
    from qemx.selection import ancestry as AY, design as DS
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg = AY.load()
    for name in ("smoke19", "pilot19", "fulldev19", "conf19"):
        specs = DS.expand(DS.load(os.path.join(root, "configs", "selection", name + ".yaml")))
        assert name in reg["studies"] and reg["studies"][name]["logical_circuits"] == AY.counts(specs)["logical_circuits"]
        assert AY.overlap(reg, specs, exclude_study=name) == {}, name
    AY.check_confirmation(reg, DS.expand(DS.load(os.path.join(root, "configs", "selection", "conf19.yaml"))), "conf19")


def test_measured_rates_without_arrays_do_not_shrink_the_disk_estimate(tmp_path):
    """Fact-check finding: a measured run whose arrays are not extracted (a shipped report) gave 0 bytes per trial and a
    disk estimate of the JSON records only; it now falls back to the array shapes (the CPU rates are still used)."""
    import shutil
    from qemx.selection import design as DS, resources as RS
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "reports", "pilot_v190", "dev", "tasks")
    d = tmp_path / "dev" / "tasks"
    d.mkdir(parents=True)
    for f in sorted(os.listdir(src))[:12]:
        if f.endswith(".json"):
            shutil.copy(os.path.join(src, f), str(d / f))
    cfg = DS.load(os.path.join(root, "configs", "selection", "fulldev19.yaml"))
    pre = RS.preflight(cfg, None, workers=1, measured_dir=str(tmp_path / "dev"))
    assert pre["cpu"]["status"] == "estimated"
    assert pre["disk"]["source"].startswith("array shapes") and pre["disk"]["labels_total_gb"] > 0.03
