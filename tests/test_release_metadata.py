"""1.8.0: one authoritative version and consistent release metadata (release-consistency audit)."""
import json
import os
import re

import qemx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def test_the_version_has_one_source_and_every_statement_of_it_agrees():
    v = qemx.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", v)
    py = _read("pyproject.toml")
    assert 'dynamic = ["version"]' in py and 'version = {attr = "qemx.__version__"}' in py
    assert not re.search(r'^version\s*=\s*"', py, re.M)                 # no second literal
    cff = _read("CITATION.cff")
    assert re.search(r"^version: %s$" % re.escape(v), cff, re.M)
    assert re.search(r"^date-released: \d{4}-\d{2}-\d{2}$", cff, re.M)
    assert _read("README.md").splitlines()[0] == "# QEM Explorer %s" % v
    assert _read("docs", "RELEASE_NOTES.md").splitlines()[0].startswith("# QEM Explorer %s" % v)
    assert "QEM Explorer %s was tested with" % v in _read("requirements-lock.txt").splitlines()[0]
    assert "https://github.com/Hyuga-Ito/qem-explorer" in py and "https://github.com/Hyuga-Ito/qem-explorer" in cff


def test_dev_extra_matches_requirements_dev():
    py = _read("pyproject.toml")
    dev = _read("requirements-dev.txt")
    for pkg in ("pytest", "nbclient", "ipykernel", "tomli"):
        assert pkg in py and pkg in dev
    assert "playwright" not in py                                         # imported by nothing that ships


def test_shipped_json_schemas_are_the_code_schemas_and_accept_every_shipped_configuration():
    import glob
    import jsonschema
    import yaml
    from qemx.selection import schemas as S
    for k, sch in S.KINDS.items():
        on_disk = json.loads(_read("docs", "selection", "schemas", "%s.schema.json" % k))
        assert on_disk == json.loads(json.dumps(sch)), k
    cfg_schema = json.loads(_read("docs", "selection", "schemas", "config.schema.json"))
    for p in sorted(glob.glob(os.path.join(ROOT, "configs", "selection", "*.yaml"))):
        d = yaml.safe_load(open(p, encoding="utf-8"))
        if d.get("schema") != "qemx.selection.config/1":
            continue
        jsonschema.validate(d, cfg_schema)


def test_protocol_keys_in_the_schema_are_the_portfolio():
    from qemx.selection import protocols as PR
    from qemx.selection import schemas as S
    assert tuple(S._PORTFOLIO_KEYS) == PR.PORTFOLIO_18
