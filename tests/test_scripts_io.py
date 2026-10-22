"""The Direction A analysis scripts with input and output separated (scripts/dirA_io.py): sources, reports, sinks;
the same results from a directory and from memory; the vectorized winner rules equal qemx's; tie order."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dirA_checks as C  # noqa: E402
import dirA_figures as F  # noqa: E402
import dirA_io as IO  # noqa: E402
import dirA_nil as N  # noqa: E402

MANIFEST = """
manifest_version: 1
study: scripts-test
defaults: {device: fake_fez, layout: {4: [136, 143, 142, 141]}, train_levels: [256], train_reps: 30,
           train_reps_other: 20}
budgets: {grid: {log10_min: 3, log10_max: 7, points: 9}}
methods:
  RAW:   {estimator: raw}
  ZNE-R: {estimator: zne, scales: [1.0, 3.0, 5.0], rule: richardson}
  CDR:   {estimator: learning, features: cdr, model: ols, n_train: 6, train_shots: 512}
  FR-L:  {estimator: learning, features: nil, model: lasso, alpha: 1.0e-4, n_train: 6, train_shots: 512}
  VD:    {estimator: vd, M: 2}
  NIL-G2:    {estimator: nil, n_neighbors: 3, model: l1, l1_bound: 2.0, n_train: 6, train_shots: 512}
  NIL-RIDGE: {estimator: nil, n_neighbors: 3, model: ridge, alpha: 1.0e-3, n_train: 6, train_shots: 512}
method_sets: {core: [RAW, ZNE-R, CDR, FR-L, VD, NIL-G2, NIL-RIDGE]}
circuits: {q: [{family: qft, n: 4, m: null, s: 1}]}
stages:
  - name: A1-primary
    engine: E2
    circuits: q
    methods: core
    scales: [1.0, 2.0]
    validate: {fraction: 0.5, budgets: [1.0e+5], reps: 100, policies: [fixed]}
"""


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from qemx.phase.run import main as run_main
    d = tmp_path_factory.mktemp("dirA")
    mp = d / "m.yaml"
    mp.write_text(MANIFEST, encoding="utf-8")
    out = str(d / "out")
    run_main(["run", str(mp), "--out", out])
    run_main(["collect", out])
    return out


def _memory(run_dir):
    src = IO.DirectorySource(run_dir)
    tables = {k: src.table(k, required=False) for k in IO.TABLES}
    return IO.MemorySource(tables, list(src.records()), src.run_info(), src.expected_cells())


def test_sources_agree(run_dir):
    d, m = IO.DirectorySource(run_dir), _memory(run_dir)
    for k in IO.TABLES:
        pd.testing.assert_frame_equal(d.table(k, required=False), m.table(k, required=False))
    assert [r["cell_id"] for r in d.records()] == [r["cell_id"] for r in m.records()]
    assert d.expected_cells() == 2 and d.run_info()["manifest_sha256"]
    with pytest.raises(FileNotFoundError):
        IO.MemorySource({}).table("methods")
    assert IO.MemorySource({}).table("levels", required=False, columns=["a"]).columns.tolist() == ["a"]


def test_checks_are_the_same_from_disk_and_memory_and_write_nothing(run_dir, tmp_path):
    before = set(os.listdir(run_dir))
    r1 = C.run_checks(C.Run(run_dir), eps=(0.01,))
    r2 = C.run_checks(C.Run(_memory(run_dir)), eps=(0.01,))
    assert set(os.listdir(run_dir)) == before                          # computations open no file
    pd.testing.assert_frame_equal(r1.summary, r2.summary)
    assert r1.render_text() == r2.render_text()
    s = r1.summary.set_index("check")["result"]
    assert s["validation-identity"] == "PASS" and s["errors"] == "PASS"
    md = r1.render_markdown()
    assert "## Summary" in md and "| validation-identity | PASS |" in md


def test_checks_main_writes_the_report(run_dir, tmp_path):
    out = str(tmp_path / "checks")
    assert C.main([run_dir, "--out", out, "--quiet", "--eps", "0.01"]) == 0
    files = set(os.listdir(out))
    assert {"summary.csv", "report.md", "report.txt", "validation.csv", "achievability_cells.csv"} <= files
    S = pd.read_csv(os.path.join(out, "summary.csv"), encoding="utf-8")
    assert list(S.columns) == ["check", "result", "detail"] and len(S)


def test_vectorized_winners_equal_qemx_rules(run_dir):
    T = F.load(run_dir)
    keys = list(T.methods["method"].unique())
    W = F._restricted_winners(T.methods, T.ledger, keys).sort_values(["cell_id", "policy", "B", "candidates"])
    ref = T.winners.sort_values(["cell_id", "policy", "B", "candidates"])
    assert len(W) == len(ref)
    for col in ("winner", "runner_up", "n_methods", "n_feasible"):
        assert W[col].fillna("").astype(str).tolist() == ref[col].fillna("").astype(str).tolist(), col
    assert np.allclose(W["rmse"].astype(float), ref["rmse"].astype(float), rtol=1e-12)
    m1, m2 = W["margin"].astype(float).values, ref["margin"].astype(float).values
    assert np.allclose(m1, m2, rtol=1e-9, equal_nan=True)


def test_figures_through_a_memory_sink(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    saved = list(F.METHODS)
    try:
        T = F.load(_memory(run_dir))
        rep = IO.Report()
        F.health(T, rep=rep)
        assert "HEALTH" in rep.render_text() and "status" in rep.tables
        sink = IO.MemoryFigureSink()
        made = F.make_figures(T, sink, rep, only=["phase", "family", "curves", "budget", "validation"],
                              methods="nil")
        assert any(n.startswith("phase_A1-primary_n4_fixed_all") for n in made)
        assert any(n.startswith("family_A1-primary") for n in made)
        assert any(n.startswith("bstar_A1-primary") for n in made)
        assert "validation_z" in made
        assert F.METHODS == F.NIL_MAP_METHODS
    finally:
        F.use_methods(saved)
        plt.close("all")


def test_budget_to_target_uses_qemx_ledger(run_dir):
    from qemx.phase.cell import cost_of
    from qemx.phase.ledger import Policy, budget_for
    T = F.load(run_dir)
    D = F.budget_to_target(T, list(T.cells.index), 0.05, "fixed")
    recs = {r["cell_id"]: r for r in IO.DirectorySource(run_dir).records()}
    for row in D.itertuples():
        rec = recs[row.cell_id]
        mr = next(m for m in rec["methods"] if m["key"] == row.method)
        c = cost_of(mr)
        pol = Policy("fixed", True, rec["cell"]["s_min"], rec["cell"]["s_tr_min"])
        exp = budget_for(c, 0.05, pol)
        assert (row.Bstar == exp) or (math.isinf(row.Bstar) and math.isinf(exp)), (row.method, row.Bstar, exp)


def test_family_map_breaks_ties_by_method_order():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = []
    for i, (lam, win) in enumerate([(0.1, "PEC"), (0.1001, "ZNE-R")]):        # one bin, one vote each: a tie
        cells.append(dict(cell_id="c%d" % i, stage="S", family="ghz", n=4, fparams=json.dumps({"theta": i}),
                          method="RAW", estimator="raw", status="ok", device="fake_fez", engine="E2", scale=1.0,
                          readout="on", coherent_fraction=0.0, opt_level=1, lam_lc=lam, ideal=0.5, flags=""))
    methods = pd.DataFrame(cells)
    wins = []
    for i, win in enumerate(["PEC", "ZNE-R"]):
        for B in (1000, 10000):
            wins.append(dict(cell_id="c%d" % i, policy="fixed", B=B, candidates="all", winner=win, rmse=0.1,
                             runner_up=None, rmse_runner_up=None, margin=None, n_methods=1, n_feasible=1,
                             missing=""))
    for order in (["PEC", "ZNE-R"], ["ZNE-R", "PEC"]):                         # data order must not matter
        w = pd.DataFrame(wins)
        w = pd.concat([w[w.winner == order[0]], w[w.winner == order[1]]])
        T = F.load(IO.MemorySource(dict(methods=methods, winners=w)), load_ledger=False)
        _, grid = F.family_map(T, "S", "ghz", bins=1)
        assert {F.METHODS[k] for k in grid.ravel()} == {"ZNE-R"}                # ZNE-R is listed before PEC
    plt.close("all")


def test_nil_analysis(run_dir, tmp_path):
    T = F.load(run_dir)
    rep = N.analyse(T, eps=0.05)
    R = rep.tables["rules"]
    assert set(R["method"]) == {"NIL-G2", "NIL-RIDGE"}
    g2 = R[R["method"] == "NIL-G2"]
    assert np.allclose(g2["bias_ratio"].dropna(), 1.0) and np.allclose(g2["v_ratio"].dropna(), 1.0)
    H = rep.tables["head_to_head"]
    assert set(H["reference"]) <= set(N.REFERENCES) and H["share_a_wins"].between(0, 1).all()
    out = str(tmp_path / "nil")
    assert N.main([run_dir, "--out", out, "--eps", "0.05", "--formats", "png"]) == 0
    assert {"rules.csv", "summary.csv", "report.md"} <= set(os.listdir(os.path.join(out, "tables")))
    assert "nil_rules.png" in os.listdir(out)


def test_report_sinks(tmp_path):
    rep = IO.Report()
    rep.section("A")
    rep.verdict("x", True, "fine")
    rep.verdict("y", False, "bad")
    rep.table("t", pd.DataFrame(dict(a=[1, 2])), show=True)
    txt = rep.render_text()
    assert "[PASS] x: fine" in txt and "[FAIL] y: bad" in txt and len(rep.failed) == 1
    paths = IO.DirectoryReportSink(str(tmp_path)).write(rep)
    assert os.path.exists(tmp_path / "t.csv") and any(p.endswith("summary.csv") for p in paths)
    mem = IO.MemoryReportSink()
    mem.write(rep)
    assert mem.reports == [rep]


def test_tables_of_earlier_runs_still_load(run_dir):
    """Tables collected by v1.3 (runs 1-2) lack the v1.4 columns and key feature regression as NIL-L / NIL-R."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src = IO.DirectorySource(run_dir)
    t = {k: src.table(k, required=False) for k in IO.TABLES}
    t["methods"] = t["methods"].drop(columns=["s_min", "s_tr_min"]).replace({"method": {"FR-L": "NIL-L"}})
    t["levels"] = t["levels"].drop(columns=["ok", "w", "se_w", "w_pairs", "w_ref_shots"], errors="ignore") \
        .replace({"method": {"FR-L": "NIL-L"}})
    t["validation"] = t["validation"].drop(columns=["calibration_mc", "z_rmse_boot"], errors="ignore") \
        .replace({"method": {"FR-L": "NIL-L"}})
    t["ledger"] = t["ledger"].replace({"method": {"FR-L": "NIL-L"}})
    t["winners"] = t["winners"].replace({"winner": {"FR-L": "NIL-L"}, "runner_up": {"FR-L": "NIL-L"}})
    old = IO.MemorySource(t, list(src.records()))
    R = C.Run(old)
    assert R.renamed == {"NIL-L": "FR-L"} and "FR-L" in set(R.methods["method"])
    rep = C.run_checks(R, eps=(0.05,))
    assert "validation-identity" in set(rep.summary["check"])
    T = F.load(old)
    D = F.budget_to_target(T, list(T.cells.index), 0.05, "fixed")
    new = F.budget_to_target(F.load(run_dir), list(T.cells.index), 0.05, "fixed")
    a = D.set_index(["cell_id", "method"])["Bstar"].sort_index()
    b = new.set_index(["cell_id", "method"])["Bstar"].sort_index()
    nonjoint = [i for i in a.index if i[1] in ("RAW", "ZNE-R", "VD")]          # no level w, same s_min: identical
    assert np.allclose(a.loc[nonjoint].values, b.loc[nonjoint].values, equal_nan=True)
    plt.close("all")
