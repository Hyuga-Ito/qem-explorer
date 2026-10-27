"""1.8.0: the candidate registry and the expanded portfolio (prompt section 3)."""
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_candidate_has_a_capability_record_and_the_inventory_covers_every_implementation():
    from qemx.selection import protocols as PR
    from qemx.selection import registry as RG
    assert set(RG.CAPABILITIES) == set(PR.PORTFOLIO_18) == set(PR.PORTFOLIO)
    assert PR.PORTFOLIO_18[:8] == PR.PORTFOLIO_16
    inv = RG.inventory()
    missing = [r for r in inv["rows"] if not r["covered"]]
    assert not missing, missing                      # every estimator, readout mode, twirl and DD option accounted for
    est = {r["name"] for r in inv["rows"] if r["kind"] == "estimator"}
    assert {"raw", "zne", "pec", "learning", "nil", "symmetry", "vd", "chain"} <= est
    assert any("vd" in u["item"] and u["status"] == "blocked" for u in inv["unsupported"])
    for k in PR.PORTFOLIO_18:
        d = RG.describe(k)
        assert d["definition"]["key"] == k and d["validation"] in RG.VALIDATION
        json.loads(json.dumps(d))


def test_the_eight_historical_candidates_are_unchanged():
    """portfolio-1.6 is immutable: the definitions equal those stored in the frozen 1.7.0 bundles, byte for byte."""
    from qemx.selection import protocols as PR
    b = json.load(open(os.path.join(ROOT, "qemx", "selection", "bundles", "qemx-rec-1.7.0", "bundle.json"),
                       encoding="utf-8"))
    for k in PR.PORTFOLIO_16:
        assert json.loads(json.dumps(PR.PORTFOLIO[k].to_json())) == b["protocol_definitions"][k], k


def test_declared_symmetry_comes_from_the_construction_only():
    from qemx.selection import protocols as PR
    from qemx.selection import task as TK
    g = TK.make_spec("ghz", 5, theta=0.7, basis="Z", observable="z_mid")
    sym = PR.declared_symmetry(g)
    assert sym["qubits"] == (0, 1) and sym["eigenvalue"] == 1 and sym["more"] == "1 2; 2 3; 3 4"
    assert PR.declared_symmetry(TK.make_spec("ghz", 5, theta=0.7)) is None             # X basis: not checkable
    assert PR.declared_symmetry(TK.make_spec("hea", 4, layers=1)) is None
    rc = PR.PORTFOLIO["SV"].recipe(g)
    assert rc.p["more"] == "1 2; 2 3; 3 4"
    with pytest.raises(PR.NotApplicable):
        PR.PORTFOLIO["SV"].recipe(TK.make_spec("hea", 4, layers=1))
    # the declared checks hold for the ideal state (a consistency check of the declaration, not how it is found)
    built = TK.build(g)
    from qemx.engine import simulate
    from qemx.noise import NoiseSpec
    comp = built.compiled
    r = simulate(comp.implemented, NoiseSpec(), "numpy", measure_map=comp.measure_map)
    idx = np.arange(2 ** 5)
    bit = lambda b: (idx >> b) & 1                                                   # noqa: E731
    ok = np.ones(2 ** 5, bool)
    for i in range(4):
        ok &= bit(i) == bit(i + 1)
    assert float(r.probs[ok].sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.fixture(scope="module")
def ghz_z():
    from qemx.selection import emulate as EM
    from qemx.selection import protocols as PR
    from qemx.selection import task as TK
    built = TK.build(TK.make_spec("ghz", 4, theta=1.1, basis="Z", observable="z_mid"))
    return built, EM.prepare(built, list(PR.PORTFOLIO_18), seed=5,
                             designs={"mode": "fresh", "per_half": 2, "seed_scheme": 2})


@pytest.mark.parametrize("B", [1000, 30000, 3000000])
def test_every_expanded_candidate_spends_exactly_the_budget(ghz_z, B):
    from qemx.selection import protocols as PR
    _, prep = ghz_z
    for k in PR.PORTFOLIO_18:
        res = prep.supplied[k]
        if isinstance(res, Exception):
            assert isinstance(res, PR.NotApplicable) or k in ("CDR", "FR", "NIL"), (k, res)
            continue
        a = PR.allocate(PR.PORTFOLIO[k], res, B)
        if not a.feasible:
            assert a.reason
            continue
        assert sum(a.ledger.values()) == B, k
        if PR.PORTFOLIO[k].needs_split:
            assert sum(a.split) == a.N and min(a.split) >= 1


def test_expanded_candidates_label_and_zne_e_b_is_the_projection_of_zne_e(tmp_path):
    from qemx.selection import labels as LB
    from qemx.selection import protocols as PR
    from qemx.selection import task as TK
    s = TK.make_spec("ghz", 4, theta=1.1, basis="Z", observable="z_mid")
    r = LB.label_task(s, str(tmp_path), [3000, 300000], list(PR.PORTFOLIO_18), 12, 9,
                      designs={"mode": "fresh", "per_half": 2, "seed_scheme": 2})
    a = LB.load_arrays(str(tmp_path), s.task_id, r)
    i, j = r["protocols"].index("ZNE-E"), r["protocols"].index("ZNE-E-B")
    assert np.array_equal(np.clip(a["est"][:, i], -1, 1), a["est"][:, j], equal_nan=True)
    feas = {al["protocol"]: al["feasible"] for al in r["allocations"] if al["B"] == 300000}
    assert feas["SV"] and feas["ZNE-L∘SV"] and feas["RO-TW"] and feas["PEC-TW"]
    sv = r["protocols"].index("SV")
    e = a["est"][1, sv, 1] - r["ideal"]
    assert np.all(np.isfinite(a["est"][1, sv, 1])) and float((e ** 2).mean()) < 0.05


def test_registry_cli_and_markdown_cover_the_portfolio():
    import contextlib
    import io
    from qemx.selection import __main__ as CLI
    from qemx.selection import protocols as PR
    from qemx.selection import registry as RG
    md = RG.markdown()
    assert all("| %s |" % k in md for k in PR.PORTFOLIO_18)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        CLI.main(["registry", "--candidate", "ZNE-E-B"])
    assert json.loads(buf.getvalue())["definition"]["post"] == "clip-observable-range"
    doc = os.path.join(ROOT, "docs", "selection", "CANDIDATES.md")
    if os.path.exists(doc):                                     # generated file: no drift from the code
        assert open(doc, encoding="utf-8").read() == md
