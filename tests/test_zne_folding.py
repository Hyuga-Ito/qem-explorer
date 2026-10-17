"""Global against local folding (v1.4.1): the effective noise scale of ZNE (lam_est), local folding as an outer method
(chain) and as a feature generator (NIL fold neighbors), the new benchmark families, and scripts/dirA_zne_folding.py
on a small pipeline run (exact crossovers checked against the ledger)."""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dirA_figures as F  # noqa: E402
import dirA_io as IO  # noqa: E402
import dirA_zne_folding as Z  # noqa: E402

LAYOUT4 = (136, 143, 142, 141)


@pytest.fixture(scope="module")
def cell():
    """One E2 cell (Brick-4, L=4) with every folding variant, plain and chained."""
    from qemx.phase.cell import CellSpec, MethodSpec, cell_runner
    M = (MethodSpec.make("RAW", "raw"),
         MethodSpec.make("ZNE-G", "zne", scales=[1.0, 3.0, 5.0], folding="global"),
         MethodSpec.make("ZNE-L", "zne", scales=[1.0, 3.0, 5.0], folding="local"),
         MethodSpec.make("ZNE-Gf", "zne", scales=[1.0, 3.0, 5.0], folding="global", lam_est="fault-rate"),
         MethodSpec.make("ZNE-Lf", "zne", scales=[1.0, 3.0, 5.0], folding="local", lam_est="fault-rate"),
         MethodSpec.make("CH-ZNE-Lf", "chain", outer="zne", inner="raw", zne_scales=[1.0, 3.0, 5.0],
                         zne_folding="local", zne_lam_est="fault-rate"),
         MethodSpec.make("NIL-ZNE-L", "nil", neighbors="fold", scales=[1.0, 3.0, 5.0], folding="local", model="l1",
                         l1_bound=5.0, n_train=6, train_shots=512),
         MethodSpec.make("CH-NIL-L", "chain", outer="nil", inner="raw", nil_neighbors="fold",
                         nil_scales=[1.0, 3.0, 5.0], nil_folding="local", nil_model="l1", nil_l1_bound=5.0,
                         n_train=6, train_shots=512))
    spec = CellSpec("T", "brickwork", 4, (("layers", 4), ("seed", 0)), layout=LAYOUT4, scale=2.0, methods=M, seed=5,
                    train_levels=(512,), train_reps=30, train_reps_other=20)
    rec = cell_runner(spec)
    assert rec["status"] == "ok", [r.get("error") for r in rec["methods"]]
    return {r["key"]: r for r in rec["methods"]}, rec


def test_effective_lambda_local_below_nominal_global_above(cell):
    rows, _ = cell
    gl, lo = rows["ZNE-G"]["details"], rows["ZNE-L"]["details"]
    assert gl["lams"] == pytest.approx([1, 3, 5]) and lo["lams"] == pytest.approx([1, 3, 5])     # nominal fit
    assert lo["lams_gates"] == pytest.approx([1, 3, 5]) and gl["lams_gates"] == pytest.approx([1, 3, 5])
    lf, gf = np.array(lo["lams_fault"]), np.array(gl["lams_fault"])
    assert lf[0] == pytest.approx(1.0) and gf[0] == pytest.approx(1.0)
    assert 1 < lf[1] < 3 and 3 < lf[2] < 5                     # one-qubit gates and idle time are not folded
    assert gf[1] > lf[1] and gf[2] > lf[2]
    assert lo["twoq"] == gl["twoq"]                            # equal two-qubit counts: equal cost per shot
    assert all(a <= b for a, b in zip(lo["depths"], gl["depths"]))   # and local folding is never deeper
    assert lo["folding"] == "local" and lo["lam_est"] == "gates"


def test_fault_rate_fit_uses_the_effective_lambda(cell):
    rows, _ = cell
    d = rows["ZNE-Lf"]["details"]
    assert d["lam_est"] == "fault-rate"
    assert d["lams"] == pytest.approx(d["lams_fault"])
    assert d["lams_fault"] == pytest.approx(rows["ZNE-L"]["details"]["lams_fault"])     # the same folded circuits
    # Richardson in λ_eff reproduces any linear E(λ_eff): its residual fraction is 0
    assert Z.residual_fraction("richardson", d["lams"], d["lams_fault"]) == pytest.approx(0, abs=1e-12)
    rho_l = Z.residual_fraction("richardson", rows["ZNE-L"]["details"]["lams"], rows["ZNE-L"]["details"]["lams_fault"])
    assert rho_l > 0.05                                        # nominal-λ local folding under-corrects


def test_local_folding_inside_the_chain_equals_the_method(cell):
    rows, _ = cell
    a, b = rows["ZNE-Lf"], rows["CH-ZNE-Lf"]
    assert b["mean"] == pytest.approx(a["mean"], abs=1e-12)
    assert b["v"] == pytest.approx(a["v"], rel=1e-10)
    assert b["details"]["lams"] == pytest.approx(a["details"]["lams"])
    assert b["details"]["folding"] == "local" and b["details"]["lam_est"] == "fault-rate"
    c, d = rows["NIL-ZNE-L"], rows["CH-NIL-L"]
    assert d["mean"] == pytest.approx(c["mean"], abs=1e-12) and d["v"] == pytest.approx(c["v"], rel=1e-9)


def test_cell_guards_cover_local_fold_neighbors(cell):
    _, rec = cell
    assert {g["folding"] for g in rec["guards"]} == {"global", "local"}


def test_fault_rate_lambdas_without_a_model():
    from qemx.methods.zne import fault_rate_lambdas
    ctx = SimpleNamespace(runner=SimpleNamespace(device_model=None), noise=None)
    lam, why = fault_rate_lambdas(ctx, [])
    assert lam is None and "no noise model" in why


def test_nil_local_fold_neighbors_fold_the_same_gates_in_training_circuits():
    from qemx.methods.base import REGISTRY
    from qemx.phase.families import build
    from qemx.transforms import clifford_replace
    spec = build("brickwork", 4, layers=4, seed=0).spec
    ctx = SimpleNamespace(spec=spec, seed=11)
    feats = REGISTRY["nil"].neighbors(ctx, "fold", scales=(1.0, 2.0, 3.0), folding="local")
    assert [f[0] for f in feats] == ["λ=1·local", "λ=2·local", "λ=3·local"]
    train = clifford_replace(spec, np.random.default_rng(0), sigma=None, resample_all=True)
    a, b = feats[1][1](spec), feats[1][1](train)                # fractional λ = 2: a random half folded once
    tags = lambda s: [i for i, o in enumerate(s.ops) if getattr(o, "tag", "") == "fold"]   # noqa: E731
    assert tags(a) == tags(b) and len(tags(a)) > 0
    with pytest.raises(ValueError):
        REGISTRY["nil"].neighbors(ctx, "fold", folding="sideways")


def test_families_grover_bv_ghz_z():
    from qemx.engine import simulate
    from qemx.noise import NoiseSpec
    from qemx.phase.families import build
    for fam, n, p, want in (("grover", 3, dict(k=1), 0.78125), ("grover", 4, dict(k=2), 0.908447265625),
                            ("bv", 5, {}, 1.0), ("ghz", 4, dict(theta=math.pi / 2, basis="Z"), 1.0)):
        fc = build(fam, n, **p)
        from qemx.compile import with_io
        rr = simulate(with_io(fc.spec, None, fc.basis), NoiseSpec(), "numpy", ideal=True)
        assert float(fc.observable.o @ rr.probs) == pytest.approx(want, abs=1e-9), (fam, n)
    assert build("ghz", 4, theta=1.0).params == (("theta", 1.0),)                  # X basis: identity unchanged
    assert dict(build("ghz", 4, theta=1.0, basis="Z").params)["basis"] == "Z"
    with pytest.raises(ValueError):
        build("ghz", 4, basis="Y")


def test_weights_and_first_order_crossover():
    from qemx.methods.zne import linear_weights, richardson_weights
    for lams in ([1, 3, 5], [1, 2.62, 4.23], [1, 1.5, 2, 3]):
        assert np.allclose(Z.richardson_weights(lams), richardson_weights(lams))
    assert np.allclose(Z.linear_weights([1, 3]), linear_weights([1, 3]))
    assert Z.residual_fraction("richardson", [1, 3, 5], [1, 3, 5]) == pytest.approx(0, abs=1e-12)
    assert Z.residual_fraction("richardson", [1, 3, 5], [1, 2.664, 4.326]) == pytest.approx(0.1675, abs=1e-3)
    assert math.isnan(Z.residual_fraction("exponential", [1, 3, 5], [1, 3, 5]))
    # RMSE² = b² + v/N; local: larger bias, smaller variance → crossover at (v_g − v_l)/(b_l² − b_g²)
    assert Z.first_order_crossover(0.02, 1.0, 0.0, 2.0) == pytest.approx(1.0 / 4e-4)
    assert Z.first_order_crossover(0.02, 2.0, 0.0, 1.0) == -math.inf        # global better in both
    assert Z.first_order_crossover(0.0, 1.0, 0.02, 2.0) == math.inf         # local better in both
    # local: smaller bias, larger variance — global wins only at small budgets, local never breaks down
    assert Z.first_order_crossover(0.01, 2.0, 0.02, 1.0) == math.inf
    assert Z._regime((2, 1), (1, 2), [(3333.0, "L")], "L", "G")[1] == math.inf      # … as B_break says
    assert math.isnan(Z.first_order_crossover(0.0, 1.0, 0.0, 1.0))


def test_regime_classification():
    a, g = "L", "G"
    assert Z._regime((1, 2), (3, 2), [(500.0, "G")], a, g)[:2] == ("local, then global", 500.0)
    assert Z._regime((2, 1), (2, 1), [], a, g)[:2] == ("global throughout", -math.inf)
    assert Z._regime((1, 2), (1, 2), [], a, g)[:2] == ("local throughout", math.inf)
    r, bb, first = Z._regime((2, 1), (1, 2), [(800.0, "L")], a, g)
    assert r == "global, then local" and bb == math.inf and first == 800.0
    r, bb, _ = Z._regime((1, 2), (3, 2), [(10.0, "G"), (20.0, "L"), (40.0, "G")], a, g)
    assert r == "multiple" and bb == 40.0
    assert Z._regime((math.inf, math.inf), (math.inf, math.inf), [], a, g)[0] == "neither feasible"


def test_folding_manifest_plans():
    from qemx.phase.manifest import check, expand, load
    m = load(os.path.join(ROOT, "manifests", "direction_A_zne_folding_v1.yaml"))
    check(m)
    cells = expand(m)
    by = {}
    for c in cells:
        by[c.stage] = by.get(c.stage, 0) + 1
    assert by == {"F1-folding": 196, "F2-gate-only": 196, "F3-coherent": 336, "F4-pipeline": 16,
                  "F5-fractional": 154, "F6-subsets": 16}


MINI = """
manifest_version: 1
study: folding-test
defaults: {device: fake_fez, layout: {3: [143, 142, 141], 4: [136, 143, 142, 141]}}
budgets: {grid: {log10_min: 3, log10_max: 7, points: 9}}
methods:
  RAW:    {estimator: raw}
  ZNE-G:  {estimator: zne, scales: [1.0, 3.0, 5.0], folding: global}
  ZNE-L:  {estimator: zne, scales: [1.0, 3.0, 5.0], folding: local}
  ZNE-Lf: {estimator: zne, scales: [1.0, 3.0, 5.0], folding: local, lam_est: fault-rate}
method_sets: {f: [ZNE-G, ZNE-L, ZNE-Lf, RAW]}
circuits: {c: [{family: grover, n: 3, k: 1}, {family: brickwork, n: 4, layers: 4, seed: 0}]}
stages:
  - {name: F1-folding, engine: E2, circuits: c, methods: f, scales: [0.5, 2.0], readout: "on"}
"""


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from qemx.phase.run import main as run_main
    d = tmp_path_factory.mktemp("fold")
    (d / "m.yaml").write_text(MINI)
    out = str(d / "out")
    run_main(["run", str(d / "m.yaml"), "--out", out])
    run_main(["collect", out])
    return out


def test_folding_script_tables(run_dir):
    T = F.load(run_dir)
    rep = Z.analyse(T)
    Ls = rep.tables["lambda_summary"]
    loc = Ls[Ls["folding"] == "local"]["lam_eff_median"].iloc[0].split(", ")
    glo = Ls[Ls["folding"] == "global"]["lam_eff_median"].iloc[0].split(", ")
    assert float(loc[2]) < 5 < float(glo[2])
    assert {v["check"]: v["result"] for v in rep.verdicts}["local-not-deeper"] == "PASS"
    assert {v["check"]: v["result"] for v in rep.verdicts}["equal-two-qubit-count"] == "PASS"
    H = rep.tables["head_to_head"]
    assert set(H["pair"]) == {"ZNE-L vs ZNE-G", "ZNE-Lf vs ZNE-G"}
    assert len(H) == 2 * 4 and H["exact"].all()


def test_breaks_are_exact_ledger_crossovers(run_dir):
    from qemx.phase.ledger import Policy, rmse
    T = F.load(run_dir)
    H = Z.head_to_head(T)
    costs = F.table_costs(T)
    checked = 0
    for r in H.itertuples():
        b = float(r.B_break)
        if not (math.isfinite(b) and b > r.B_lo):
            continue
        (ca, s_min, s_tr), (cg, _, _) = costs[(r.cell_id, r.local)], costs[(r.cell_id, r.glob)]
        pol = Policy("fixed", True, s_min, s_tr)
        assert rmse(ca, int(b), pol) > rmse(cg, int(b), pol)              # global strictly better from B_break …
        assert rmse(ca, int(b) - 1, pol) <= rmse(cg, int(b) - 1, pol)      # … and not one shot earlier
        for B in np.geomspace(b, r.B_hi, 7).astype(int):
            assert rmse(ca, int(B), pol) >= rmse(cg, int(B), pol)          # … at every larger budget (sampled)
        checked += 1
    assert checked >= 1


def test_grid_fallback_agrees_with_exact(run_dir, monkeypatch):
    T = F.load(run_dir)
    exact = Z.head_to_head(T).set_index(["cell_id", "pair"])
    monkeypatch.setattr(F, "table_costs", lambda *a, **k: {})            # as without qemx
    grid = Z.head_to_head(T).set_index(["cell_id", "pair"])
    assert not grid["exact"].any()
    same = (exact["regime"] == grid["regime"]).mean()
    assert same >= 0.75                                                    # 9-point ledger grid: coarse but consistent


def test_folding_figures(run_dir):
    T = F.load(run_dir)
    rep = Z.analyse(T)
    sink = IO.MemoryFigureSink()
    F.setup_matplotlib()
    names = Z.make_figures(T, rep, sink)
    assert "folding_lambda_eff" in names and "folding_depth" in names
    assert any(n.startswith("folding_map_F1_folding_ZNE-L_vs_ZNE-G") for n in names)
    assert "folding_bias_F1_folding" in names
    import matplotlib.pyplot as plt
    plt.close("all")


def test_folding_main_writes(run_dir, tmp_path):
    out = str(tmp_path / "figs")
    assert Z.main([run_dir, "--out", out, "--formats", "png"]) == 0
    assert os.path.exists(os.path.join(out, "tables", "head_to_head.csv"))
    assert os.path.exists(os.path.join(out, "folding_lambda_eff.png"))


def test_lam_est_in_a_noise_spec_context_and_manifest_choices():
    """The Playground's NoiseSpec runner supports the fault-rate λ; the manifest rejects unknown choices."""
    from qemx.compile import build as compile_build
    from qemx.methods.base import Recipe
    from qemx.noise import NoiseSpec
    from qemx.phase.families import build
    from qemx.phase.manifest import _check_method
    from qemx.recipes import compare
    fc = build("qft", 3)
    comp = compile_build(fc.spec, None, fc.basis, relabel=fc.relabel)
    rcs = [Recipe.make("zne", folding="local", lam_est="fault-rate"), Recipe.make("zne", folding="local")]
    ideal, outs, _ = compare(comp, NoiseSpec(p1=1e-3, p2=1e-2), fc.observable, rcs)
    a, b = outs[0].result, outs[1].result
    assert a is not None and b is not None, (outs[0].error, outs[1].error)
    assert a.details["lams"] == pytest.approx(a.details["lams_fault"])
    assert 1 < a.details["lams"][1] < 3 and b.details["lams"] == pytest.approx([1, 3, 5])
    assert abs(a.mean - ideal) < abs(b.mean - ideal)                 # the calibrated λ removes the under-correction
    with pytest.raises(ValueError):
        _check_method("Z", {"estimator": "zne", "lam_est": "bogus"})
    with pytest.raises(ValueError):
        _check_method("N", {"estimator": "nil", "neighbors": "fold", "folding": "diagonal"})
