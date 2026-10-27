"""Tests that use the per-trial arrays of the 1.8.0 study (reports/selection_v180; 1.9.0, second review N5).

The arrays are release attachments (``reports/TRIAL_ATTACHMENTS.json``), not repository files. Without them these tests
are skipped — unless ``QEMX_REQUIRE_TRIALS`` names the study (the trial-data workflow sets it), in which case missing
arrays are a failure, never a skip. With them, the saved study is checked end to end: arrays against the study's own
checksum manifest, every labels directory complete and bound, cached label tables equal to the tables derived from the
arrays, the saved replay rows scored with the derived labels, the published primary contrasts recomputed, and the
development and confirmation circuits disjoint."""
import json
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "reports", "selection_v180")
STAGES = ["dev", "confirm", "regimes", "noisecomp"] + ["audit18/root_%d" % r for r in (18001, 18002, 18003, 18004)]


def _arrays_present() -> bool:
    d = os.path.join(STUDY, "confirm", "tasks")
    return os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))


@pytest.fixture(scope="module", autouse=True)
def arrays():
    if not _arrays_present():
        if "selection_v180" in os.environ.get("QEMX_REQUIRE_TRIALS", ""):
            pytest.fail("QEMX_REQUIRE_TRIALS names selection_v180 but its per-trial arrays are missing")
        pytest.skip("the per-trial arrays of selection_v180 are not extracted (release attachments; "
                    "reports/TRIAL_ATTACHMENTS.json)")


def test_arrays_match_the_studys_own_manifest():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_trials", os.path.join(ROOT, "scripts", "verify_trials.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = m.verify(STUDY)
    assert r["ok"] and r["listed"] == 217 and r["missing"] == r["changed"] == 0, r


@pytest.mark.parametrize("stage", STAGES)
def test_every_labels_directory_is_complete_and_bound(stage):
    from qemx.selection import integrity as IN
    c = IN.completion(os.path.join(STUDY, stage))
    assert c["bound"] and c["complete"] and not c["corrupted"] and not c["unexpected"], IN.describe(c)


def test_cached_confirmation_labels_equal_the_tables_derived_from_the_arrays():
    from qemx.selection import replay as RP
    pop = RP.bound_population(os.path.join(STUDY, "confirm"))       # raises StaleTable on any difference
    assert pop["intended"] == 58 and not pop["excluded"] and len(pop["table"]) == 58 * 17 * 10


@pytest.mark.parametrize("bundle,diff", [("qemx-rec-1.8.0", -0.1440553398389816),
                                         ("qemx-rec-1.8.0-deployable", -0.08397027297857552)])
def test_saved_replay_rows_are_scored_with_the_derived_labels(bundle, diff):
    import pandas as pd
    from qemx.selection import replay as RP
    pop = RP.bound_population(os.path.join(STUDY, "confirm"))
    tab = pop["table"]
    risk = {(r.task_id, r.protocol, int(r.B)): (RP._feasible(r.feasible), float(r.mse_eval)) for r in tab.itertuples()}
    rows = pd.read_csv(os.path.join(STUDY, "replay", "replay_%s.csv" % bundle), encoding="utf-8").to_dict("records")
    assert len(rows) == 580 and len({r["base_id"] for r in rows}) == 29
    for r in rows:
        feas, mse = risk[(r["task_id"], r["choice"], int(r["B"]))]
        assert bool(r["scored"]) == (feas and math.isfinite(mse))
        if r["scored"]:
            assert float(r["risk"]) == pytest.approx(mse, rel=1e-12, abs=0)
    ch = pd.read_csv(os.path.join(STUDY, "confirm", "choices_confirm.csv"), encoding="utf-8")
    sbsb = {(t, int(b)): risk[(t, c, int(b))][1] for t, b, c in zip(ch.task_id, ch.B, ch["SBS-B|choice"])}
    for r in rows:
        r["scored"] = bool(r["scored"])
    s = RP.summarize(rows, sbsb)
    saved = json.load(open(os.path.join(STUDY, "replay", "replay_%s.json" % bundle), encoding="utf-8"))
    assert s["diff"] == pytest.approx(diff, abs=1e-12) and s["diff"] == pytest.approx(saved["diff"], abs=1e-12)
    assert s["diff_lo"] == pytest.approx(saved["diff_lo"], abs=1e-12)


def test_development_and_confirmation_circuits_are_disjoint():
    from qemx.selection import ancestry as AY, labels as LB, task as TK

    def ids(stage):
        d = os.path.join(STUDY, stage)
        return {AY.logical_id(TK.TaskSpec.from_json(LB.load_record(d, t)["spec"])) for t in LB.task_ids(d)}
    dev, conf = ids("dev"), ids("confirm")
    assert len(dev) == 44 and len(conf) == 29 and not (dev & conf)


def test_dev_arrays_reproduce_the_saved_development_labels():
    from qemx.selection import replay as RP
    pop = RP.bound_population(os.path.join(STUDY, "dev"), require_complete=True)
    import pandas as pd
    saved = pd.read_csv(os.path.join(STUDY, "analysis", "labels.csv"), encoding="utf-8")
    t = pop["table"].merge(saved[["task_id", "protocol", "B", "mse_eval"]], on=["task_id", "protocol", "B"],
                           suffixes=("", "_saved"))
    assert len(t) == len(saved) == len(pop["table"])
    x, y = t.mse_eval.astype(float).to_numpy(), t.mse_eval_saved.astype(float).to_numpy()
    assert np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=1e-12)


def test_replay_script_refuses_an_edited_baseline_column(tmp_path):
    """1.9.0 (independent verification): choices_confirm.csv's SBS-B column must follow the SBS-B rule of the
    bundles' training labels; an edited baseline is refused before any replay (exit 2)."""
    import shutil
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import replay_confirmation as RC
    d = str(tmp_path / "confirm")
    shutil.copytree(os.path.join(STUDY, "confirm"), d)
    p = os.path.join(d, "choices_confirm.csv")
    lines = open(p, encoding="utf-8").read().splitlines()
    head = lines[0].split(",")
    j = head.index("SBS-B|choice")
    row = lines[1].split(",")
    row[j] = "RAW" if row[j] != "RAW" else "RO"
    lines[1] = ",".join(row)
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    assert RC.main(["--labels", d, "--out", str(tmp_path / "replay")]) == 2
    assert not os.path.exists(str(tmp_path / "replay" / "replay_summary.json"))


def _flip_sbsb(d):
    p = os.path.join(d, "choices_confirm.csv")
    lines = open(p, encoding="utf-8").read().splitlines()
    j = lines[0].split(",").index("SBS-B|choice")
    for i in (1, 2, 3):
        row = lines[i].split(",")
        row[j] = "RAW" if row[j] != "RAW" else "RO"
        lines[i] = ",".join(row)
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def test_the_baseline_rule_is_refitted_on_the_training_labels_and_unverifiable_baselines_are_refused(tmp_path):
    """Second verification round (F2, F3): (a) the rule refitted on the run's own development labels reproduces the
    shipped SBS-B column on all 580 rows; (b) the thesis tables refuse an edited column; (c) without training labels and
    without a bundle trained on the evaluation's parts the replay refuses instead of skipping the check."""
    import json
    import shutil
    import sys
    from qemx.selection import replay as RP, tables19 as T
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import replay_confirmation as RC
    pop = RP.bound_population(os.path.join(STUDY, "confirm"))
    rule, source = RP.sbsb_rule_choices(os.path.join(STUDY, "confirm"), pop["table"])
    assert source.startswith("refitted on the training labels")
    import pandas as pd
    ch = pd.read_csv(os.path.join(STUDY, "confirm", "choices_confirm.csv"), encoding="utf-8")
    keyed = {(t, int(b)): {"SBS-B|choice": c} for t, b, c in zip(ch.task_id, ch.B, ch["SBS-B|choice"])}
    assert RP.check_sbsb_column(keyed, rule) == len(keyed) == 580
    # (b) the tables: a copy without dev/ falls back to the shipped bundles' ranks, and an edit is refused
    v18 = tmp_path / "v18"
    shutil.copytree(os.path.join(STUDY, "confirm"), str(v18 / "confirm"))
    shutil.copytree(os.path.join(STUDY, "replay"), str(v18 / "replay"))
    src = T.sources(ROOT, str(v18), None, None)
    assert src["baseline_check"]["rows_checked"] == 580 and "bundle" in src["baseline_check"]["source"]
    _flip_sbsb(str(v18 / "confirm"))
    with pytest.raises(ValueError):
        T.sources(ROOT, str(v18), None, None)
    # (c) a bundle trained on other parts and no training labels: refused, not skipped
    bdir = tmp_path / "bundles"
    for name in ("qemx-rec-1.8.0", "qemx-rec-1.8.0-deployable"):
        shutil.copytree(os.path.join(ROOT, "qemx", "selection", "bundles", name), str(bdir / name))
        bj = bdir / name / "bundle.json"
        meta = json.loads(bj.read_text(encoding="utf-8"))
        meta.setdefault("training", {})["parts"] = "dev,final"
        bj.write_text(json.dumps(meta), encoding="utf-8")
    shutil.rmtree(str(v18 / "confirm"))
    shutil.copytree(os.path.join(STUDY, "confirm"), str(v18 / "confirm"))
    _flip_sbsb(str(v18 / "confirm"))
    assert RC.main(["--labels", str(v18 / "confirm"), "--out", str(tmp_path / "replay"), "--bundle-dir",
                    str(bdir)]) == 2
    assert not os.path.exists(str(tmp_path / "replay" / "replay_summary.json"))
