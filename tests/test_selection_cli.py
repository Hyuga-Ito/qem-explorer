"""1.6.0 end to end on a tiny configuration: label (with resume), analyze, schema validation, freeze, the
once-only final evaluation and the figures/tables drawn from the saved tables. Slow-marked (a few minutes)."""
import json
import os

import pytest
import yaml

from qemx.selection import schemas as SC
from qemx.selection.__main__ import main

TINY = dict(schema="qemx.selection.config/1", name="tiny", master_seed=5, R=8, budgets=[1000, 100000],
            protocols=["RAW", "RO", "ZNE-L", "PEC"], final_test_fraction=0.34,
            tasks=dict(widths=[3], families={"ghz": [{"theta": 1.0472}], "ising": [{"steps": 1}, {"steps": 2}],
                                             "hea": [{"layers": 1, "seed": 0}], "qaoa": [{"p": 1, "seed": 0}]},
                       inputs=["default"], observables=["default"], layouts=["best"], devices=["fake_fez"],
                       variants_per_base=0),
            limits=dict(task_seconds=600, memory_gb=4),
            analysis=dict(splits=["same-distribution"], split_options={"same-distribution": {"k": 2}},
                          feature_sets=["size_budget"], model_families=["ridge"], abstain_family="ridge",
                          abstain_set="size_budget", bootstrap=50, extras=["certificates"]))


@pytest.mark.slow
def test_label_resume_analyze_freeze_and_final_once(tmp_path):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump(TINY), encoding="utf-8")
    out = str(tmp_path / "run")
    SC.validate(TINY, "config")
    assert main(["label", str(cfg), "--out", out]) == 0
    tasks = sorted(os.listdir(os.path.join(out, "dev", "tasks")))
    assert tasks and all(t.endswith((".json", ".npz")) for t in tasks)
    for t in tasks:
        if t.endswith(".json"):
            SC.validate(json.load(open(os.path.join(out, "dev", "tasks", t), encoding="utf-8")), "label")
    mtimes = {t: os.path.getmtime(os.path.join(out, "dev", "tasks", t)) for t in tasks}
    assert main(["label", str(cfg), "--out", out]) == 0                     # resume: nothing is recomputed
    assert {t: os.path.getmtime(os.path.join(out, "dev", "tasks", t)) for t in tasks} == mtimes
    summ = json.load(open(os.path.join(out, "dev", "label_summary.json"), encoding="utf-8"))
    assert summ["skipped"] == summ["ok"] == len(tasks) // 2 and summ["per_task_seconds"] == []
    prov = json.load(open(os.path.join(out, "dev", "provenance.json"), encoding="utf-8"))
    assert len(prov["sessions"]) == 2 and prov["sessions"][1]["changed_since_first"] == []
    assert main(["analyze", str(cfg), "--out", out]) == 0
    for f in ("summary_same-distribution.csv", "choices_same-distribution.csv", "labels.csv", "certificates.csv",
              "task_features.csv", "results.json"):
        assert os.path.exists(os.path.join(out, "analysis", f)), f
    assert main(["final", str(cfg), "--out", out]) == 2                     # refused before the freeze
    assert main(["freeze", str(cfg), "--out", out]) == 0
    assert main(["final", str(cfg), "--out", out]) == 0
    assert os.path.exists(os.path.join(out, "final", "summary_final.csv"))
    assert main(["final", str(cfg), "--out", out]) == 1                     # evaluated once only
    assert main(["figures", str(cfg), "--out", out]) == 0                   # from saved tables only
    assert os.path.getsize(os.path.join(out, "figures", "fig01_budget_efficiency.pdf")) > 1000
    pending = open(os.path.join(out, "figures", "PENDING.md"), encoding="utf-8").read()
    assert "fig02_performance_regimes: missing regimes/analysis/regime_cells.csv" in pending
    assert "Figures that failed" not in pending
    for t in ("table1_design", "table2_protocols", "table3_policy_results"):
        assert os.path.exists(os.path.join(out, "tables", t + ".csv")), t
    # 1.7.0: one figure at a time; an unknown name is refused
    assert main(["figures", str(cfg), "--out", out, "--only", "supp_failures_costs"]) == 0
    assert main(["figures", str(cfg), "--out", out, "--only", "no_such_figure"]) == 2
    # 1.7.0: a bundle's validation belongs to its own configuration; only the frozen one carries the threshold
    frz = json.load(open(os.path.join(out, "freeze.json"), encoding="utf-8"))
    fam, frozen_set = frz["policy"][len("MODEL["):-1].split("|")
    other = next(s_ for s_ in ("size_budget", "deployable", "structure") if s_ != frozen_set)
    for fs, name in ((frozen_set, "b_frozen"), (other, "b_other")):
        bo = os.path.join(str(tmp_path), name)
        assert main(["bundle", str(cfg), "--out", out, "--version", name, "--feature-set", fs, "--train", "dev",
                     "--bundle-out", bo]) == 0
        meta = json.load(open(os.path.join(bo, "bundle.json"), encoding="utf-8"))
        conf = "MODEL[%s|%s]" % (fam, fs)
        assert meta["configuration"] == conf and meta["validation"]["configuration"] == conf
        assert {r["policy"] for r in meta["validation"].get("out_of_fold", [])} <= {conf, "SBS-B", "ALL-CANDIDATE"}
        if fs != frozen_set:
            assert meta["selective"]["tau"] is None and "nested_cv" not in meta["validation"]
