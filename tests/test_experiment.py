"""1.5: one experiment model for UI, YAML and CLI — validation with field paths, stable ids and hashes, and a
UI state → YAML → CLI round trip that reproduces the app's numbers."""
import json
import os
import subprocess
import sys

import pytest

from qemx import experiment as E
from qemx.textio import child_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = dict(schema=E.SCHEMA, name="qft-pilot",
           circuit=dict(template="qft", params=dict(n=3, s=5, mode="swap", prep="fourier")),
           noise=dict(p1=0.001, p2=0.01, ro01=0.02, ro10=0.03),
           methods=[dict(preset="Unmitigated"), dict(preset="ZNE · Richardson"),
                    dict(estimator="raw", readout="tensored"), dict(preset="NIL · small budget")],
           budget=dict(shots=4096), run=dict(seed=3, repeats=40))


def _err(doc):
    with pytest.raises(E.ExperimentError) as ei:
        E.resolve(doc)
    return ei.value


def test_short_form_resolves_validates_and_plans():
    r = E.resolve(DOC)
    assert r.compiled.logical.n == 3 and len(r.recipes) == 4 and r.shots == 4096
    assert len(set(r.method_ids)) == 4 and all("@" in m for m in r.method_ids)
    p = E.plan(r)
    assert p["config_hash"] == r.config_hash and [m["preflight"] for m in p["methods"]][-1] == "ok"
    assert r.state["circuit"]["template"] == dict(key="qft", params=dict(n=3, s=5, mode="swap", prep="fourier"))


def test_errors_name_the_field():
    assert _err(dict(DOC, schema="qemx.experiment/9")).path == "schema"
    assert _err(dict(DOC, circuit=dict(template="qft", params=dict(n=3, s=99)))).path == "circuit.params"
    assert _err(dict(DOC, circuit=dict(template="qft", params=dict(n=3, colour=1)))).path == "circuit.params"
    e = _err(dict(DOC, methods=[dict(preset="Unmitigated"), dict(estimator="zne", params=dict(rule="sideways"))]))
    assert e.path == "methods[1].params"
    assert _err(dict(DOC, methods=[dict(estimator="raw", readout="magic")])).path == "methods[0].readout"
    assert _err(dict(DOC, noise=dict(p2=1.5))).path == "state"
    assert _err(dict(DOC, extra=1)).path == ""
    assert _err(dict(DOC, run=dict(seed=-1))).path == "run"
    assert _err(dict(DOC, observable=dict(bitstring="10"))).path == "observable.bitstring"


def test_credentials_are_rejected_anywhere():
    e = _err(dict(DOC, budget=dict(shots=10), backend="numpy", compile=dict(ibm_token="x")))
    assert e.path == "compile.ibm_token" and "credentials" in str(e)
    st = E.to_state(DOC)
    e = _err(dict(schema=E.SCHEMA, state=dict(st, api_key="x")))
    assert e.path == "state.api_key"


def test_yaml_is_loaded_safely(tmp_path):
    p = tmp_path / "evil.yaml"
    p.write_text("schema: !!python/object/apply:os.system ['echo hacked']\n", encoding="utf-8")
    import yaml
    with pytest.raises(yaml.YAMLError):
        E.load(str(p))


def test_state_and_short_forms_have_the_same_hash_and_numbers(tmp_path):
    r1 = E.resolve(DOC)
    doc2 = E.from_state(r1.state, "same", run=DOC["run"])
    text = E.dump_yaml(doc2)
    doc3 = E.loads(text)
    r3 = E.resolve(doc3)
    assert r3.config_hash == r1.config_hash and r3.method_ids == r1.method_ids
    a, b = E.execute(r1), E.execute(r3)
    ka = {row["method_id"]: row["estimate"] for row in a["rows"]}
    kb = {row["method_id"]: row["estimate"] for row in b["rows"]}
    assert ka == kb and a["ideal"] == b["ideal"] == pytest.approx(1.0)
    changed = E.resolve(dict(DOC, run=dict(seed=4, repeats=40)))
    assert changed.config_hash != r1.config_hash                          # seeds are part of the configuration


def test_ui_state_round_trips_through_yaml_and_the_cli(tmp_path, monkeypatch):
    """The Playground's own state export → YAML → `python -m qemx.experiment run` gives the app's numbers."""
    import streamlit as st
    from qemx import compile as CP
    from qemx.recipes import compare, table_rows
    from ui import state as S

    class SS(dict):
        __getattr__ = dict.get

        def __setattr__(self, k, v):
            self[k] = v
    fake = SS()
    monkeypatch.setattr(st, "session_state", fake, raising=False)
    monkeypatch.setattr(S, "push_widgets", lambda: None)
    S.init()
    S.load_template("qft", dict(n=3, s=6, mode="relabel", prep="fourier"))
    fake.noise = fake.noise.replace(p1=0.002, p2=0.02, ro01=0.01, ro10=0.02)
    fake.recipe_names = ["Unmitigated", "ZNE · linear", "Readout · TREX", "NIL · small budget"]
    fake.shots = 8192
    state = S.to_json()
    # the app's path (ui/cache.comparison + run_all's table)
    comp = CP.build(S.spec(), fake.inputs, fake.basis, S.compile_options(), fake.relabel)
    ideal, outs, _ = compare(comp, fake.noise, fake.observable, S.recipes(), fake.backend, int(fake.shots),
                             fake.symmetry, check_budget=True)
    _, rows = table_rows(outs, ideal, int(fake.shots), float(fake.t_shot_us))
    app = {r["method"]: r["estimate"] for r in rows}
    # YAML → CLI
    y = tmp_path / "exp.yaml"
    y.write_text(E.dump_yaml(E.from_state(state, "ui-roundtrip")), encoding="utf-8")
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=ROOT)
    for cmd in ("validate", "plan"):
        p = subprocess.run([sys.executable, "-m", "qemx.experiment", cmd, str(y)], cwd=ROOT, env=child_env(env),
                           capture_output=True, text=True, timeout=600, encoding="utf-8")
        assert p.returncode == 0, p.stderr
    p = subprocess.run([sys.executable, "-m", "qemx.experiment", "run", str(y), "--out", str(out)], cwd=ROOT, env=child_env(env),
                       capture_output=True, text=True, timeout=900, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    cli = {r["method"]: r["estimate"] for r in res["rows"]}
    assert set(cli) == set(app)
    for k in app:
        assert cli[k] == pytest.approx(app[k], rel=1e-12, abs=1e-12), k
    assert res["ideal"] == pytest.approx(ideal) and res["provenance"]["app_version"]
    assert "token" not in json.dumps(res).lower()
    # a second run into the same directory with another configuration is refused
    y2 = tmp_path / "exp2.yaml"
    d2 = E.from_state(dict(state, shots=4096), "other")
    y2.write_text(E.dump_yaml(d2), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "qemx.experiment", "run", str(y2), "--out", str(out)], cwd=ROOT, env=child_env(env),
                       capture_output=True, text=True, timeout=900, encoding="utf-8")
    assert p.returncode == 2 and "another configuration" in p.stderr


@pytest.mark.parametrize("path", sorted(
    os.path.join(ROOT, "examples", "experiments", f) for f in os.listdir(os.path.join(ROOT, "examples", "experiments"))
    if f.endswith((".yaml", ".yml"))) if os.path.isdir(os.path.join(ROOT, "examples", "experiments")) else [])
def test_example_experiments_validate(path):
    r = E.resolve(E.load(path))
    assert E.plan(r)["methods"]
