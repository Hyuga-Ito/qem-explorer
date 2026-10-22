"""The exported script reproduces the app; the pages run."""
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from qemx import compile as CP
from qemx import export as EX
from qemx import templates as T
from qemx.noise import PRESETS
from qemx.recipes import compare, preset
from qemx.textio import child_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _state(tpl, opts, noise, recipes, backend="numpy"):
    return dict(app="QEM Explorer", version="0.1.0", circuit=tpl.spec.to_json(), inputs=["0"] * tpl.spec.n,
                basis="Z" * tpl.spec.n, relabel=tpl.relabel, compile=opts, noise=dataclasses.asdict(noise),
                observable=dict(key=tpl.observable.key, label=tpl.observable.label, tex=tpl.observable.tex,
                                values=list(tpl.observable.values)), symmetry=None,
                recipes=[r.to_json() for r in recipes], backend=backend, shots=4096, t_shot_us=250.0)


def test_script_contains_security_sentence_and_no_token():
    tpl = T.bell()
    src = EX.script(_state(tpl, dict(mode="abstract", basis_2q="cx", coupling="all", layout="sabre", opt_level=1),
                           PRESETS["Depolarizing"], [preset("Unmitigated")]))
    assert "A token is a password. Before this notebook goes into a repository or an appendix, clear it and use one of" in src
    assert "token = os.environ.get" in src and "print(token" not in src
    compile(src, "export", "exec")


@pytest.mark.slow
def test_exported_script_reproduces_the_app(tmp_path):
    tpl = T.qft(3, 5, "relabel")
    opts = dict(mode="isa", basis_2q="cz", coupling="line", layout="sabre", opt_level=1)
    comp = CP.build(tpl.spec, relabel=tpl.relabel, opts=CP.CompileOptions(**opts))
    noise = PRESETS["Hardware-realistic (illustrative)"].replace(field_samples=2)
    recipes = [preset(k) for k in ("Unmitigated", "ZNE · Richardson", "PEC · inverse", "CDR · OLS", "Readout · TREX",
                                   "Dynamical decoupling (XY4)")]
    ideal, outs, _ = compare(comp, noise, tpl.observable, recipes)
    expected = {o.title: o.result.mean for o in outs}
    p = tmp_path / "export.py"
    p.write_text(EX.script(_state(tpl, opts, noise, recipes), expected), encoding="utf-8")
    out = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=900, cwd=tmp_path, encoding="utf-8", env=child_env())
    assert out.returncode == 0, out.stderr[-2000:]
    marks = [ln for ln in out.stdout.splitlines() if ln.rstrip().endswith("✓")]
    assert len(marks) == len(recipes), out.stdout


@pytest.mark.slow
def test_exported_script_runs_a_fake_device(tmp_path):
    """A state with a fake-device backend exports a script that compiles for that device and reproduces the app."""
    tpl = T.bell()
    opts = dict(mode="abstract", basis_2q="cx", coupling="line", layout="sabre", opt_level=1, seed=11,
                initial_layout="")
    comp = CP.build(tpl.spec, opts=CP.CompileOptions(**{k: v for k, v in opts.items() if k != "initial_layout"},
                                                     device="fake_perth"))
    recipes = [preset(k) for k in ("Unmitigated", "ZNE · linear", "Readout · TREX")]
    ideal, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, recipes, backend="fake_perth")
    expected = {o.title: o.result.mean for o in outs}
    p = tmp_path / "export_dev.py"
    p.write_text(EX.script(_state(tpl, opts, PRESETS["Depolarizing"], recipes, backend="fake_perth"), expected), encoding="utf-8")
    out = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=900, cwd=tmp_path, encoding="utf-8", env=child_env())
    assert out.returncode == 0, out.stderr[-3000:]
    assert "device model" in out.stdout
    marks = [ln for ln in out.stdout.splitlines() if ln.rstrip().endswith("✓")]
    assert len(marks) == len(recipes), out.stdout
