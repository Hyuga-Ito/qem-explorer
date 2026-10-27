"""Text I/O contract (1.7.0): explicit UTF-8 everywhere, no hidden decoding errors, Unicode paths, a legacy default
encoding (the situation of a Windows machine) must not change what is read or written."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = "量子誤り緩和 · ⟨O⟩ = Σ_k c_k ⟨P_k⟩ · λ, γ, ε₀ ≤ 10⁻⁶ · ∀x∈ℝ: x² ≥ 0 · αβγδ · 日本語のコメント"


def _checker():
    spec = importlib.util.spec_from_file_location("check_text_io", os.path.join(ROOT, "scripts", "check_text_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_project_text_io_call_names_its_encoding():
    found = _checker().check_paths()
    assert not found, "\n".join(str(f) for f in found[:40])


def test_the_checker_catches_what_it_should_and_accepts_binary_io():
    c = _checker()
    bad = c.check_source("open('a.txt')\nopen('a', 'w')\nimport pathlib\npathlib.Path('x').read_text()\n"
                         "import pandas as pd\npd.read_csv('x.csv')\ndf.to_csv('y.csv')\n"
                         "import subprocess\nsubprocess.run(['x'], text=True)\n"
                         "open('a', encoding='utf-8', errors='ignore')\nopen(p, mode)\n")
    msgs = [f[2] for f in bad]
    assert len([m for m in msgs if "without encoding" in m]) == 6
    assert any("hides decoding errors" in m for m in msgs) and any("non-literal mode" in m for m in msgs)
    ok = c.check_source("open('a.npz', 'rb')\nopen('a', 'wb')\nimport tomllib\ntomllib.load(open('c.toml', 'rb'))\n"
                        "open('a', 'w', encoding='utf-8')\nimport json\njson.load(open('x', encoding='utf-8'))\n"
                        "df.to_csv(index=False)\n")
    assert not ok, ok


@pytest.fixture
def unicode_dir(tmp_path):
    d = tmp_path / "データ αβγ ∑ résumé"
    d.mkdir()
    return d


def test_textio_round_trips_non_ascii_text_in_unicode_paths(unicode_dir):
    from qemx import textio as TX
    p = unicode_dir / "結果_λ.json"
    TX.write_json(p, {"note": TEXT, "値": [1, 2.5]})
    assert TX.read_json(p) == {"note": TEXT, "値": [1, 2.5]}
    raw = p.read_bytes()
    assert TEXT.encode("utf-8") in raw.replace(b"\\n", b"\n") or "量子".encode("utf-8") in raw   # stored as UTF-8
    q = unicode_dir / "メモ.md"
    TX.write_text(q, TEXT + "\n")
    assert TX.read_text(q) == TEXT + "\n"


def test_byte_order_mark_is_accepted_only_where_declared_and_legacy_bytes_are_refused(unicode_dir):
    from qemx import textio as TX
    p = unicode_dir / "bom.yaml"
    p.write_bytes(b"\xef\xbb\xbf" + "name: テスト\n".encode("utf-8"))
    assert TX.read_yaml(p) == {"name": "テスト"}                  # explicit compatibility path
    assert TX.read_text(p).startswith("﻿")                 # ordinary reads keep the mark visible
    q = unicode_dir / "cp1252.yaml"
    q.write_bytes("name: café\n".encode("cp1252"))
    with pytest.raises(TX.TextEncodingError) as e:
        TX.read_yaml(q)
    assert "cp1252.yaml" in str(e.value) and "byte 9" in str(e.value)


def test_selection_configuration_and_labels_records_in_unicode_paths(unicode_dir):
    from qemx.selection import design as DS
    from qemx.selection import labels as LB
    src = open(os.path.join(ROOT, "configs", "selection", "smoke.yaml"), encoding="utf-8").read()
    p = unicode_dir / "設定_⟨O⟩.yaml"
    p.write_text("# " + TEXT + "\n" + src, encoding="utf-8")
    cfg = DS.load(str(p))
    assert cfg["name"] == "smoke" and len(DS.expand(cfg)) > 0
    d = unicode_dir / "ラベル"
    (d / "tasks").mkdir(parents=True)
    rec = dict(schema=LB.SCHEMA, task_id="0" * 20, base_id="1" * 16, status="error", error=TEXT, error_kind="build")
    LB.write_json(str(d / "tasks" / ("0" * 20 + ".json")), rec)
    assert LB.load_record(str(d), "0" * 20)["error"] == TEXT


def test_experiment_document_with_non_ascii_text_round_trips(unicode_dir):
    from qemx import experiment as E
    doc = E.load(os.path.join(ROOT, "examples", "experiments", "qft_known_answer.yaml"))
    doc["description"] = TEXT
    p = unicode_dir / "実験 ⟨Z⟩.yaml"
    p.write_text(E.dump_yaml(doc), encoding="utf-8")
    assert E.load(str(p))["description"] == TEXT


def test_csv_tables_round_trip_non_ascii_text(unicode_dir):
    import pandas as pd
    df = pd.DataFrame({"protocol": ["ZNE-λ", "NIL", "誤り緩和"], "note": [TEXT, "γ", "∑"]})
    p = unicode_dir / "表.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    assert pd.read_csv(p, encoding="utf-8").equals(df)


LEGACY_SCRIPT = r'''
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
from qemx import textio as TX
from qemx.selection import design as DS, labels as LB
from qemx import experiment as E
with open(sys.argv[2], encoding="utf-8") as f:     # not argv: arguments are decoded with the locale's encoding
    text = f.read()
d = tempfile.mkdtemp(prefix="qemx_text_")
p = os.path.join(d, "x.json")
TX.write_json(p, {"t": text})
assert TX.read_json(p)["t"] == text
cfg = DS.load(os.path.join(sys.argv[1], "configs", "selection", "smoke.yaml"))
doc = E.load(os.path.join(sys.argv[1], "examples", "experiments", "qft_known_answer.yaml"))
doc["description"] = text
q = os.path.join(d, "e.yaml")
with open(q, "w", encoding="utf-8") as f:
    f.write(E.dump_yaml(doc))
assert E.load(q)["description"] == text
print("ok", text)
'''


def test_project_text_io_does_not_depend_on_the_default_encoding(tmp_path):
    """A child Python with PEP 597's EncodingWarning turned into an error: any project text I/O that falls back to
    the locale's encoding fails. Where a legacy (ISO-8859-1) locale is installed (see the CI job), the child also
    runs with that default encoding and UTF-8 mode off."""
    env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="utf-8")
    loc = os.environ.get("QEMX_LEGACY_LOCALE")                 # e.g. en_US.ISO-8859-1 (CI); optional locally
    if loc:
        env["LC_ALL"] = loc
    s = tmp_path / "legacy.py"
    s.write_text(LEGACY_SCRIPT, encoding="utf-8")
    t = tmp_path / "text.txt"
    t.write_text(TEXT, encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", str(s), ROOT,
                        str(t)], capture_output=True, encoding="utf-8", env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip() == "ok " + TEXT
    if loc:
        chk = subprocess.run([sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
                             capture_output=True, encoding="utf-8", env=env, timeout=60)
        assert "utf" not in chk.stdout.lower(), chk.stdout


def test_command_line_prints_utf8_to_a_pipe_under_a_legacy_default(tmp_path):
    env = dict(os.environ, PYTHONUTF8="0")
    env.pop("PYTHONIOENCODING", None)
    loc = os.environ.get("QEMX_LEGACY_LOCALE")
    if loc:
        env["LC_ALL"] = loc
    r = subprocess.run([sys.executable, "-m", "qemx.selection", "plan", os.path.join(ROOT, "configs", "selection",
                                                                                    "smoke.yaml")],
                       capture_output=True, encoding="utf-8", env=env, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert '"tasks"' in r.stdout
