"""1.9.0 stage-aware pipeline (second review N3): interrupted after labeling, analysis, freeze, freeze completion and
during the confirmation's labeling, then resumed — completed scientific outputs are unchanged, missing later outputs
are completed, once-only decisions are verified and never redone, and an inconsistency stops the pipeline. Runs in a
directory whose path has spaces and Japanese characters. Slow (child processes for every command)."""
import hashlib
import json
import os

import pytest
import yaml

from qemx import textio as TX
from qemx.selection import pipeline as PL

DEV = {"schema": "qemx.selection.config/1", "name": "pipeline-dev-probe", "master_seed": 424242, "R": 6,
       "budgets": [1000, 100000], "protocols": ["RAW", "RO", "ZNE-L", "PEC"], "final_test_fraction": 0.0,
       "tasks": {"widths": [3], "families": {"ising": [{"steps": 1, "h": 0.731}, {"steps": 1, "h": 0.379}],
                                             "hea": [{"layers": 1, "angle_seed": 9921}],
                                             "qaoa": [{"p": 1, "angle_seed": 9922}]},
                 "include_default": True, "variants_per_base": 0},
       "limits": {"task_seconds": 600, "memory_gb": 3},
       "analysis": {"splits": ["same-distribution"], "split_options": {"same-distribution": {"k": 2}},
                    "feature_sets": ["size_budget", "lightcone"], "model_families": ["ridge"],
                    "abstain_family": "ridge", "abstain_set": "size_budget", "bootstrap": 30, "extras": []}}
CONF = {"schema": "qemx.selection.config/1", "name": "pipeline-conf-probe", "master_seed": 434343, "R": 6,
        "budgets": [1000, 100000], "protocols": ["RAW", "RO", "ZNE-L", "PEC"], "final_test_fraction": 0.0,
        "tasks": {"widths": [3], "families": {"ising": [{"steps": 1, "h": 0.257}],
                                              "hea": [{"layers": 1, "angle_seed": 9931}]},
                  "include_default": True, "variants_per_base": 0},
        "limits": {"task_seconds": 600, "memory_gb": 3}}


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _write(tmp, name, obj):
    p = tmp / name
    p.write_text(yaml.safe_dump(obj, allow_unicode=True), encoding="utf-8")
    return str(p)


def _run(pipe, out, **kw):
    lines = []
    r = PL.Runner(PL.load(pipe), out, workers=2, echo=lines.append, **{k: v for k, v in kw.items() if k != "until"})
    try:
        rc = r.run(until=kw.get("until"))
    except PL.PipelineError as e:
        return e.code, lines
    return rc, lines


@pytest.mark.slow
def test_pipeline_resumes_after_every_interruption_without_redoing_once_only_steps(tmp_path):
    dev = _write(tmp_path, "dev.yaml", DEV)
    conf = _write(tmp_path, "conf.yaml", CONF)
    pipe = _write(tmp_path, "pipe.yaml", {
        "schema": "qemx.selection.pipeline/1", "name": "probe", "purpose": "smoke", "development": dev,
        "confirmation": {"config": conf, "train": "dev"},
        "bundles": [{"version": "probe-b", "feature_set": "frozen", "simulation": False}],
        "stages": ["plan", "label", "analyze", "freeze", "nested", "freeze-complete", "confirm", "bundles", "replay",
                   "report", "status"]})
    out = str(tmp_path / "run dir テスト")
    rc, _ = _run(pipe, out, until="label")
    assert rc == 0 and len([f for f in os.listdir(os.path.join(out, "dev", "tasks")) if f.endswith(".npz")]) == 4
    rc, _ = _run(pipe, out, until="analyze")
    assert rc == 0 and os.path.exists(os.path.join(out, "analysis", "results.json"))
    rc, _ = _run(pipe, out, until="freeze")
    frz = os.path.join(out, "freeze.json")
    assert rc == 0 and TX.read_json(frz)["deploy_tau"] is None
    h_frz = _sha(frz)
    rc, lines = _run(pipe, out, until="freeze")               # resumed at the same point: nothing changes
    assert rc == 0 and _sha(frz) == h_frz and any("kept" in x for x in lines)
    rc, _ = _run(pipe, out, until="freeze-complete")
    assert rc == 0 and TX.read_json(frz)["deploy_tau"] is not None
    h_frz = _sha(frz)
    h_pool = _sha(os.path.join(out, "analysis", "pooled_same-distribution.csv"))
    # an interruption during the confirmation's labeling: one of its two tasks labeled
    from qemx.selection.__main__ import main
    assert main(["label", conf, "--out", out, "--subdir", "confirm", "--limit", "1"]) == 0
    rc, lines = _run(pipe, out)
    assert rc == 0, "\n".join(lines[-30:])
    ev = os.path.join(out, "confirm", "EVALUATED")
    assert os.path.exists(ev) and _sha(frz) == h_frz
    assert _sha(os.path.join(out, "analysis", "pooled_same-distribution.csv")) == h_pool
    rep = TX.read_json(os.path.join(out, "replay", "replay_probe-b.json"))
    assert rep["confirmation"] is True and rep["provenance"]["labels"]["evaluated_tasks"] == 2
    h_ev = _sha(ev)
    # the once-only confirmation is bound to its configuration and training parts (independent verification)
    assert main(["confirm", conf, "--out", out, "--dev-config", dev, "--train", "dev", "--reuse"]) == 0
    assert main(["confirm", conf, "--out", out, "--dev-config", dev, "--train", "dev,final", "--reuse"]) == 2
    other = _write(tmp_path, "conf_other.yaml", dict(CONF, R=8))
    assert main(["confirm", other, "--out", out, "--dev-config", dev, "--train", "dev", "--reuse"]) == 2
    # a complete rerun verifies everything and changes no decision
    rc, lines = _run(pipe, out)
    assert rc == 0 and _sha(frz) == h_frz and _sha(ev) == h_ev
    assert sum("kept" in x or "verified" in x for x in lines) >= 3
    st = TX.read_json(os.path.join(out, "pipeline", "completion.json"))
    assert TX.read_json(os.path.join(out, "pipeline", "status.json"))["completion"] == "completion.json"
    assert st["dev"]["complete"] and st["confirm"]["complete"]
    # an inconsistency stops the pipeline (exit 2) and is never repaired by redoing a once-only step
    with open(os.path.join(out, "analysis", "pooled_same-distribution.csv"), "a", encoding="utf-8") as f:
        f.write("\n")
    rc, lines = _run(pipe, out)
    assert rc == 2 and _sha(frz) == h_frz


@pytest.mark.slow
def test_label_failure_stops_dependent_stages(tmp_path):
    bad = dict(DEV, limits={"task_seconds": 0.001, "memory_gb": 3})
    dev = _write(tmp_path, "dev.yaml", bad)
    pipe = _write(tmp_path, "pipe.yaml", {"schema": "qemx.selection.pipeline/1", "name": "probe2", "purpose": "smoke",
                                          "development": dev, "stages": ["label", "analyze", "freeze", "status"]})
    out = str(tmp_path / "r")
    rc, lines = _run(pipe, out)
    assert rc == 1 and not os.path.exists(os.path.join(out, "analysis"))
    assert not os.path.exists(os.path.join(out, "freeze.json"))


def test_pipeline_file_validation(tmp_path):
    p = _write(tmp_path, "p.yaml", {"schema": "qemx.selection.pipeline/1", "name": "x", "purpose": "smoke",
                                    "development": "configs/selection/smoke19.yaml", "stages": ["label", "nope"]})
    with pytest.raises(PL.PipelineError):
        PL.load(p)
    p = _write(tmp_path, "q.yaml", {"schema": "qemx.selection.pipeline/1", "name": "x", "purpose": "smoke",
                                    "development": "configs/selection/smoke19.yaml", "typo_key": 1})
    with pytest.raises(PL.PipelineError):
        PL.load(p)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in os.listdir(os.path.join(root, "configs", "pipelines")):
        PL.load(os.path.join(root, "configs", "pipelines", f))


def test_dry_run_uses_argument_arrays_with_unicode_and_spaces(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines = []
    r = PL.Runner(PL.load(os.path.join(root, "configs", "pipelines", "reproduce_v18.yaml")),
                  str(tmp_path / "結果 dir"), workers=2, echo=lines.append, dry_run=True,
                  python=str(tmp_path / "Python 3.12 環境" / "python.exe"))
    assert r.run() == 0
    argv = [c["argv"] for c in r.log]
    assert all(isinstance(a, list) for a in argv)
    assert any(a[0].endswith("python.exe") and "環境" in a[0] for a in argv)
    stages = [c["stage"] for c in r.log]
    assert stages.index("freeze") < stages.index("confirm") < stages.index("replay")
    conf = [a for a in argv if "confirm" in a and "--dev-config" in a][0]
    assert "--reuse" in conf                                   # never "|| true": verified reuse instead
    assert all("|| true" not in " ".join(a) for a in argv)


@pytest.mark.slow
def test_terminating_the_pipeline_stops_its_running_stage_command(tmp_path):
    """Independent verification (second round, F6): SIGTERM to the pipeline runner during labeling must stop the
    running ``label`` command and its workers and release the run lock (1.9.0 first version: the label command kept
    running, holding the lock, until its output pipe broke). 1.9.1 (independent review F5): observed through the lock
    file, the label command's own PID (recorded in the lock, in this process's PID view) and the absence of further
    writes — not through psutil, whose process view can belong to another PID namespace."""
    import signal
    import subprocess
    import sys
    import time
    if os.name == "nt":
        pytest.skip("POSIX signals")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev = _write(tmp_path, "dev.yaml", dict(DEV, name="pipeline-term-probe", R=3000))
    pipe = _write(tmp_path, "pipe.yaml", {"schema": "qemx.selection.pipeline/1", "name": "term", "purpose": "smoke",
                                          "development": dev, "stages": ["plan", "label", "status"]})
    out = str(tmp_path / "run テスト")
    lock = os.path.join(out, "dev", ".qemx-run.lock")
    p = subprocess.Popen([sys.executable, "-m", "qemx.selection", "pipeline", pipe, "--out", out, "--workers", "1"],
                         cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t_end = time.time() + 120
        while time.time() < t_end and not os.path.exists(lock):
            time.sleep(0.1)
        assert os.path.exists(lock), "the label stage did not start"
        time.sleep(3.0)                               # a worker is labeling now
        label_pid = int(TX.read_json(lock)["pid"])
        p.send_signal(signal.SIGTERM)
        p.wait(90)
        t_end = time.time() + 20                     # stopped by the runner within seconds
        while time.time() < t_end and os.path.exists(lock):
            time.sleep(0.2)
        assert not os.path.exists(lock), "the label command still holds the run lock"
        gone = False
        for _ in range(50):
            try:
                os.kill(label_pid, 0)
            except ProcessLookupError:
                gone = True
                break
            time.sleep(0.2)
        assert gone, "the label command is still running"
        tasks = os.path.join(out, "dev", "tasks")
        before = sorted(os.listdir(tasks)) if os.path.isdir(tasks) else []
        time.sleep(3.0)
        after = sorted(os.listdir(tasks)) if os.path.isdir(tasks) else []
        assert before == after, "a worker kept writing after the pipeline was terminated"
        log = open(os.path.join(out, "pipeline", "label.log"), "rb").read().decode("utf-8")
        assert "stage command stopped by the runner" in log
    finally:
        if p.poll() is None:
            p.kill()
