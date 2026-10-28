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


# File names vs file contents (1.8.0, docs/CI_FAILURE_ANALYSIS.md). Contents are always UTF-8 (the contract under
# test). Names are encoded by the operating system: always representable on Windows (UTF-16 names; Python's file-system
# encoding is UTF-8 whatever the code page), but on Linux/macOS the file-system encoding follows the locale, and under
# a legacy ISO-8859-1 locale (CI's text-encoding job) no program can create "データ". The fixtures therefore use the
# richest non-ASCII name the interpreter's file-system encoding can represent: the full Japanese/Greek/math names
# wherever possible (every UTF-8 system, Windows), Latin-1 accented names under a Latin-1 locale. A UTF-8 file-name
# encoding with a legacy *text* encoding (the actual Windows situation) is covered separately by
# test_unicode_file_names_with_a_legacy_text_encoding, and test_path_names_are_not_degraded_where_representable
# guarantees the full names are used wherever they can be.
NAMES = {   # full name: Latin-1 counterpart (both non-ASCII)
    "データ αβγ ∑ résumé": "données àéî résumé",
    "結果_λ.json": "résultat_µ.json",
    "メモ.md": "mémo.md",
    "設定_⟨O⟩.yaml": "réglage_±.yaml",
    "ラベル": "étiquettes",
    "実験 ⟨Z⟩.yaml": "expérience µ.yaml",
    "表.csv": "tableau_é.csv",
}


def _representable(name: str) -> bool:
    try:
        name.encode(sys.getfilesystemencoding())
        return True
    except UnicodeEncodeError:
        return False


def fs_name(full: str) -> str:
    """``full`` if this interpreter can name a file so, else its Latin-1 counterpart; never an ASCII-only name."""
    if _representable(full):
        return full
    alt = NAMES[full]
    if not _representable(alt):
        pytest.fail("this interpreter's file-system encoding (%s) cannot represent any non-ASCII file name; the locale "
                    "named by LC_ALL/LANG is probably not installed" % sys.getfilesystemencoding())
    return alt


@pytest.fixture
def unicode_dir(tmp_path):
    d = tmp_path / fs_name("データ αβγ ∑ résumé")
    d.mkdir()
    return d


def test_path_names_are_not_degraded_where_representable():
    enc = sys.getfilesystemencoding().lower().replace("_", "-")
    for full, alt in NAMES.items():
        assert any(ord(c) > 127 for c in alt) and _representable(alt) == (enc != "ascii")
        if enc in ("utf-8", "utf8") or sys.platform == "win32":
            assert fs_name(full) == full                        # every UTF-8 system and Windows: the full names
    if sys.platform == "win32":
        assert enc == "utf-8"                                  # PEP 529: independent of the ANSI code page


def test_a_file_name_the_locale_cannot_represent_is_a_clear_path_error(tmp_path, monkeypatch):
    """What a user sees on Linux under a Latin-1 locale: a PathEncodingError naming the encoding and the remedy,
    distinct from TextEncodingError (bad contents). Simulated on every platform by a file-system encoder that refuses
    the characters, as ISO-8859-1 does."""
    from qemx import textio as TX
    real_open = open

    def latin1_open(path, *a, **kw):
        s = os.fspath(path)
        if isinstance(s, str):
            s.encode("latin-1")                                # raises UnicodeEncodeError for "データ"
        return real_open(path, *a, **kw)
    monkeypatch.setattr("builtins.open", latin1_open)
    p = tmp_path / "データ.json"
    with pytest.raises(TX.PathEncodingError) as e:
        TX.write_json(p, {"x": 1}, atomic=False)
    msg = str(e.value)
    assert "file name" in msg and "UTF-8 locale" in msg and "データ" in msg
    with pytest.raises(TX.PathEncodingError):
        TX.read_text(p)
    assert not isinstance(e.value, TX.TextEncodingError)
    ok = tmp_path / "résumé.json"                              # representable in Latin-1: works
    TX.write_json(ok, {"t": TEXT}, atomic=False)
    assert TX.read_json(ok) == {"t": TEXT}
    assert TX.path_representable(tmp_path / "plain.json")


def test_textio_round_trips_non_ascii_text_in_unicode_paths(unicode_dir):
    from qemx import textio as TX
    p = unicode_dir / fs_name("結果_λ.json")
    TX.write_json(p, {"note": TEXT, "値": [1, 2.5]})
    assert TX.read_json(p) == {"note": TEXT, "値": [1, 2.5]}
    raw = p.read_bytes()
    assert TEXT.encode("utf-8") in raw.replace(b"\\n", b"\n") or "量子".encode("utf-8") in raw   # stored as UTF-8
    q = unicode_dir / fs_name("メモ.md")
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
    p = unicode_dir / fs_name("設定_⟨O⟩.yaml")
    p.write_text("# " + TEXT + "\n" + src, encoding="utf-8")
    cfg = DS.load(str(p))
    assert cfg["name"] == "smoke" and len(DS.expand(cfg)) > 0
    d = unicode_dir / fs_name("ラベル")
    (d / "tasks").mkdir(parents=True)
    rec = dict(schema=LB.SCHEMA, task_id="0" * 20, base_id="1" * 16, status="error", error=TEXT, error_kind="build")
    LB.write_json(str(d / "tasks" / ("0" * 20 + ".json")), rec)
    assert LB.load_record(str(d), "0" * 20)["error"] == TEXT


def test_experiment_document_with_non_ascii_text_round_trips(unicode_dir):
    from qemx import experiment as E
    doc = E.load(os.path.join(ROOT, "examples", "experiments", "qft_known_answer.yaml"))
    doc["description"] = TEXT
    p = unicode_dir / fs_name("実験 ⟨Z⟩.yaml")
    p.write_text(E.dump_yaml(doc), encoding="utf-8")
    assert E.load(str(p))["description"] == TEXT


def test_csv_tables_round_trip_non_ascii_text(unicode_dir):
    import pandas as pd
    df = pd.DataFrame({"protocol": ["ZNE-λ", "NIL", "誤り緩和"], "note": [TEXT, "γ", "∑"]})
    p = unicode_dir / fs_name("表.csv")
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


SPLIT_SCRIPT = r"""
import locale, os, sys, tempfile
legacy = sys.argv[3]
if legacy and sys.platform != "win32":
    # start-up fixed the file-system encoding (UTF-8 locale); switching LC_CTYPE now changes only the default *text*
    # encoding of open(): exactly Windows without UTF-8 mode (UTF-8 file names, cp1252 text default)
    locale.setlocale(locale.LC_CTYPE, legacy)
getenc = getattr(locale, "getencoding", None) or (lambda: locale.getpreferredencoding(False))   # 3.10: no getencoding
fs, text_default = sys.getfilesystemencoding(), getenc()
print("fs=%s default=%s" % (fs, text_default))
sys.path.insert(0, sys.argv[1])
from qemx import textio as TX
from qemx.selection import design as DS, labels as LB
from qemx import experiment as E
with open(sys.argv[2], encoding="utf-8") as f:
    text = f.read()
d = os.path.join(tempfile.mkdtemp(prefix="qemx_split_"), "データ αβγ ∑")
os.makedirs(os.path.join(d, "ラベル", "tasks"))
TX.write_json(os.path.join(d, "結果_λ.json"), {"t": text})
assert TX.read_json(os.path.join(d, "結果_λ.json"))["t"] == text
src = TX.read_text(os.path.join(sys.argv[1], "configs", "selection", "smoke.yaml"))
TX.write_text(os.path.join(d, "設定_⟨O⟩.yaml"), "# " + text + "\n" + src)
assert DS.load(os.path.join(d, "設定_⟨O⟩.yaml"))["name"] == "smoke"
rec = dict(schema=LB.SCHEMA, task_id="0" * 20, base_id="1" * 16, status="error", error=text, error_kind="build")
LB.write_json(os.path.join(d, "ラベル", "tasks", "0" * 20 + ".json"), rec)
assert LB.load_record(os.path.join(d, "ラベル"), "0" * 20)["error"] == text
doc = E.load(os.path.join(sys.argv[1], "examples", "experiments", "qft_known_answer.yaml"))
doc["description"] = text
TX.write_text(os.path.join(d, "実験 ⟨Z⟩.yaml"), E.dump_yaml(doc))
assert E.load(os.path.join(d, "実験 ⟨Z⟩.yaml"))["description"] == text
print("ok")
"""


def test_unicode_file_names_with_a_legacy_text_encoding(tmp_path):
    """The Windows situation, also emulated on Linux: Japanese/Greek/math *file names* (UTF-8 file-system encoding)
    while the *default text encoding* is a legacy one (UTF-8 mode off, EncodingWarning an error). On Windows the child
    runs as it is (cp1252 default, UTF-16 names). On Linux, where the CI's text-encoding job installs a legacy locale
    (QEMX_LEGACY_LOCALE), the child starts under C.UTF-8 and then switches only LC_CTYPE to the legacy locale."""
    env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="utf-8")
    legacy = os.environ.get("QEMX_LEGACY_LOCALE", "") if sys.platform != "win32" else ""
    if sys.platform != "win32":
        env["LC_ALL"] = "C.UTF-8"
    sc = tmp_path / "split.py"
    sc.write_text(SPLIT_SCRIPT, encoding="utf-8")
    t = tmp_path / "text.txt"
    t.write_text(TEXT, encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", str(sc), ROOT,
                        str(t), legacy], capture_output=True, encoding="utf-8", env=env, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    first, last = r.stdout.splitlines()[0], r.stdout.splitlines()[-1]
    assert first.startswith("fs=utf-8 ") and last == "ok", r.stdout
    if legacy or sys.platform == "win32" and not sys.flags.utf8_mode:
        assert "utf" not in first.split("default=")[1].lower(), first        # the split was really in effect


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


STRICT_PROBE = r"""
import sys, warnings
sys.path.insert(0, sys.argv[1])
import strict_encoding as SE
target = sys.argv[2]
results = {}
for name in ("qemx.fake_module", "ui.fake", "dill._dill"):
    with warnings.catch_warnings(record=True) as w:
        SE.install()
        g = {"__name__": name, "target": target}
        try:
            exec(compile("open(target, 'w').close()", name, "exec"), g)
            results[name] = "warning" if any(issubclass(x.category, EncodingWarning) for x in w) else "nothing"
        except EncodingWarning:
            results[name] = "error"
print(results)
"""


def test_strict_encoding_wrapper_errors_for_project_code_and_only_shows_third_party(tmp_path):
    """scripts/strict_encoding.py (CI's text-encoding job): a default-encoding open() attributed to a project module
    is an error; the same call attributed to a third-party module stays a visible warning (never silenced)."""
    probe = tmp_path / "probe.py"
    probe.write_text(STRICT_PROBE, encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "warn_default_encoding", str(probe), os.path.join(ROOT, "scripts"),
                        str(tmp_path / "x.txt")], capture_output=True, encoding="utf-8",
                       env=dict(os.environ, PYTHONUTF8="0"), timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "{'qemx.fake_module': 'error', 'ui.fake': 'error', 'dill._dill': 'warning'}", r.stdout
    bare = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "strict_encoding.py"), "-m", "json.tool"],
                          capture_output=True, encoding="utf-8", timeout=120)
    assert bare.returncode != 0 and "warn_default_encoding" in (bare.stdout + bare.stderr)
