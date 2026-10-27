"""1.9.0 figure/table specification and the source-column mapping of the 1.9.0 tables (prompt §5): a table that
looks right but took the wrong column must fail here."""
import importlib
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V18 = os.path.join(ROOT, "reports", "selection_v180")
AUDIT = os.path.join(ROOT, "reports", "execution_audit_v190")
PILOT = os.path.join(ROOT, "reports", "pilot_v190")


def _arrays():
    d = os.path.join(V18, "confirm", "tasks")
    return os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))


def test_specification_names_existing_generators_and_honest_statuses():
    from qemx import textio as TX
    spec = TX.read_yaml(os.path.join(ROOT, "configs", "selection", "figure_spec19.yaml"))
    assert spec["schema"] == "qemx.selection.figure-spec/1"
    allowed = ("confirmatory", "exploratory", "descriptive", "pending", "PENDING")
    entries = [p for f in spec["figures"] for p in f["panels"]] + spec["tables"]
    for e in entries:
        mod, fn = e["generator"].rsplit(".", 1)
        assert hasattr(importlib.import_module(mod), fn), e["generator"]
        assert str(e["status"]).split()[0].rstrip(",;") in allowed, (e["id"], e["status"])
    f6 = [p for f in spec["figures"] if f["id"] == "F6" for p in f["panels"]]
    assert f6 and all(str(p["status"]).upper().startswith("PENDING") for p in f6)


def test_contrast_is_policy_minus_baseline_per_circuit():
    import pandas as pd
    from qemx.selection import tables19 as T
    t = pd.DataFrame(dict(base_id=["a", "a", "b"], risk=[1e-4, 1e-4, 1e-3], sbsb=[1e-3, 1e-3, 1e-3]))
    c = T.contrast(t)
    want = np.mean([np.log10(1e-4 + 1e-6) - np.log10(1e-3 + 1e-6), 0.0])     # circuit a, then b; circuits equal
    assert c["diff"] == pytest.approx(want) and c["circuits"] == 2
    assert T.contrast(t, col="sbsb", base="risk")["diff"] == pytest.approx(-want)   # a swapped column flips the sign


@pytest.mark.skipif(not _arrays(), reason="the 1.8.0 per-trial arrays are not extracted")
def test_primary_table_overall_rows_equal_the_saved_replay_summaries():
    from qemx import textio as TX
    from qemx.selection import tables19 as T
    src = T.sources(ROOT, V18, None, None)
    tab = T.s3_primary(src)
    rows = [n for n in tab.numeric if n["B"] == "all"]
    assert len(rows) == 2
    for n in rows:
        saved = TX.read_json(os.path.join(V18, "replay", "replay_%s.json" % n["bundle"]))
        assert n["diff"] == pytest.approx(saved["diff"], abs=1e-12)
        assert n["lo"] == pytest.approx(saved["diff_lo"], abs=1e-12) and n["hi"] == pytest.approx(saved["diff_hi"],
                                                                                                    abs=1e-12)
        assert n["mse_policy"] == pytest.approx(saved["policy_mse"], rel=1e-12)
        assert n["mse_sbsb"] == pytest.approx(saved["baseline_mse"], rel=1e-12)
        assert n["circuits"] == saved["base_circuits"] == 29
    prim = [n for n in rows if n["bundle"] == "qemx-rec-1.8.0"][0]
    assert prim["diff"] == pytest.approx(-0.1440553398389816, abs=1e-12)


@pytest.mark.skipif(not os.path.exists(os.path.join(AUDIT, "execution_audit_summary.json")),
                    reason="no execution audit")
def test_execution_table_maps_executed_and_fresh_columns():
    from qemx import textio as TX
    from qemx.selection import tables19 as T
    s = TX.read_json(os.path.join(AUDIT, "execution_audit_summary.json"))
    tab = T.s4_execution(dict(audit=s))
    for n in tab.numeric:
        v = s["policies"][n["policy"]]
        assert n["executed"] == v["executed"]["diff_vs_sbsb"] and n["fresh"] == v["fresh"]["diff_vs_sbsb"]
        assert n["difference"] == v["executed_minus_fresh_contrast"]["diff"]


def _require_pilot():
    if not os.path.isdir(os.path.join(PILOT, "analysis")):
        if "pilot_v190" in os.environ.get("QEMX_REQUIRE_TRIALS", ""):
            pytest.fail("QEMX_REQUIRE_TRIALS names pilot_v190 but reports/pilot_v190/analysis is missing")
        pytest.skip("no pilot report")


def test_pilot_table_takes_the_pooled_columns():
    _require_pilot()
    import pandas as pd
    from qemx.selection import tables19 as T
    tab = T.s5_pilot(dict(pilot=PILOT, root=ROOT))
    for n in tab.numeric:
        if n["split"] == "portfolio":
            continue
        q = pd.read_csv(os.path.join(PILOT, "analysis", "pooled_%s.csv" % n["split"]), encoding="utf-8")
        r = q[(q.B.astype(str) == "all") & (q.policy == n["policy"])].iloc[0]
        assert (n["diff"], n["lo"], n["hi"]) == (float(r.diff_vs_sbsb), float(r.diff_vs_sbsb_lo),
                                                 float(r.diff_vs_sbsb_hi))


def test_tables_build_without_optional_sources(tmp_path):
    from qemx.selection import tables19 as T
    p = T.build(dict(root=ROOT, v18=str(tmp_path / "none"), confirmation={}, pilot=None), str(tmp_path / "out"),
                compile_pdf=False)
    assert "s1_datasets" in p["made"] and "s2_candidates" in p["made"]
    assert {"s3_primary", "s4_execution", "s5_pilot"} <= set(p["skipped"])
    tex = open(os.path.join(str(tmp_path / "out"), "s1_datasets.tex"), encoding="utf-8").read()
    assert "\\toprule" in tex and "\\bottomrule" in tex


def test_check_config_refuses_a_malformed_figure_specification(tmp_path):
    """Independent verification (F11): unknown keys, missing version, duplicate ids, unknown statuses and missing
    generators are errors of check-config, not only of this test file."""
    from qemx.selection import formats as FM
    p = tmp_path / "spec.yaml"
    p.write_text("schema: qemx.selection.figure-spec/1\nfigurez: []\nfigures: []\ntables:\n"
                 "  - {id: a, question: q, source: s, status: whatever, generator: qemx.selection.tables19.nope}\n"
                 "  - {id: a, question: q, source: s, status: descriptive, generator: qemx.selection.tables19.s1_datasets}\n",
                 encoding="utf-8")
    r = FM.check(str(p))
    text = " ".join(r["errors"])
    for needle in ("figurez", "version", "duplicate id a", "whatever", "nope"):
        assert needle in text, (needle, r["errors"])
    assert FM.check(os.path.join(ROOT, "configs", "selection", "figure_spec19.yaml"))["errors"] == []
