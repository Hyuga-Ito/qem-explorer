"""1.5.1 (finding F02): the live recommendation reproduces the study's protocol and noise context.

Key invariant: a stored study cell (real pilot records, ``tests/data/pilot_cells``, written by 1.5.0's cell runner),
reconstructed as a live experiment — the same logical circuit compiled by ``qemx.compile.build`` with the options the
app derives for the cell's device backend and forced layout — has identical recommendation-time features and identical
resolved protocol costs (recipe, calibration circuits, training shots, and feasibility on the study's budget grid
under the study's cost policy).
"""
import glob
import json
import math
import os

import pytest

from qemx import features as FT

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pilot_cells")
RECS = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(DATA, "*.json")))]
GRID = [int(round(10 ** (3 + 0.1 * k))) for k in range(41)]           # the pilot's budget grid (manifest)


def _injector():
    from qemx.noise import PRESETS
    return PRESETS["Hardware-realistic (illustrative)"]              # the app's default injector: must be ignored


def _live(rec, backend=None, observable=None):
    """The cell as a live experiment, built only from what the record stores (no cell-runner helpers)."""
    from qemx import compile as CP
    from qemx import rec_context as RX
    from qemx.phase import families as FAM
    c = rec["cell"]
    fc = FAM.build(c["family"], int(c["n"]), **dict(c["fparams"]))
    backend = backend or rec["engine"]["backend"]
    opts = CP.CompileOptions(device=backend, opt_level=int(c["opt_level"]), seed=int(c["seed_transpiler"]),
                             initial_layout=tuple(rec["circuit"]["chain"]))
    comp = CP.build(fc.spec, None, fc.basis, opts, relabel=fc.relabel)
    return fc, comp, RX.live_context(comp, backend, _injector(), observable or fc.observable)


def _study(tmp_path, recs=RECS):
    from qemx import rec_context as RX
    ctx = RX.context_from_records(recs, study="fixture", candidates=None, upper_bounds=("VD",), budgets=GRID,
                                  feature_schema=FT.VERSION)
    RX.write_methods(str(tmp_path / "tables" / "methods.json"), ctx)
    return RX.load_study(str(tmp_path))


@pytest.mark.parametrize("rec", RECS, ids=[r["cell_id"] for r in RECS])
def test_stored_cell_as_live_experiment_has_identical_features(rec):
    from qemx import rec_context as RX
    fc, comp, ctx = _live(rec)
    ci = rec["circuit"]
    assert list(comp.info["phys"]) == ci["phys"] and comp.implemented.count("2q") == ci["two_qubit"]
    assert {str(k): v for k, v in comp.measure_map.items()} == ci["measure_map"]
    B = 31623.0
    names = [f.name for f in FT.DICTIONARY if f.recommendable]
    live, lmiss = RX.live_features(ctx, B, names)
    stored, smiss = FT.from_record(rec, B, FT.profile_of_cell(rec["cell"]),
                                   observable=FT.observable_of_cell(rec["cell"]))
    for k in names:
        a, b = live[k], stored[k]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-12, abs=1e-15), (k, a, b)
    assert not lmiss and not smiss
    # lam_lc is a sum of ~10² fault probabilities: recomputed on another NumPy its last bits can differ (pairwise
    # summation order; 1.3e-14 relative observed on Python 3.10). The same 1e-12 relative tolerance as the features
    # above is three orders of magnitude tighter than any physical difference and far looser than rounding.
    assert live["scale"] == rec["cell"]["scale"]
    assert live["lam_lc"] == pytest.approx(rec["lambda"]["lam_lc"], rel=1e-12, abs=0.0)


@pytest.mark.parametrize("rec", RECS, ids=[r["cell_id"] for r in RECS])
def test_stored_cell_as_live_experiment_has_identical_resolved_protocol_costs(rec, tmp_path):
    from qemx import rec_context as RX
    from qemx.phase.cell import CellSpec, cost_of, recipe_for
    from qemx.phase.ledger import Policy, allocate
    study = _study(tmp_path)
    assert study.usable, study.problems
    fc, comp, ctx = _live(rec)
    spec = CellSpec.from_json(rec["cell"])
    pol = Policy("fixed", True, int(rec["cell"]["s_min"]), int(rec["cell"]["s_tr_min"]))
    assert study.policy() == pol
    rows = {r["key"]: r for r in rec["methods"] if r.get("status") == "ok"}
    costs = RX.protocol_costs(study, ctx, 10_000, keys=list(rows))
    for ms in spec.methods:
        row = rows[ms.key]
        rc = study.recipe(ms.key)
        assert rc == recipe_for(ms, spec, {})                          # the protocol the cell ran
        assert rc.readout == row["readout_mit"] and rc.title() == row["title"]
        c = costs[ms.key]
        assert not c["error"], c
        assert (c["calibration_circuits"], c["training_shots"], c["T"], c["s_tr"]) == \
               (row["c"], row["F"], row["T"], row["s_tr"]), ms.key
        assert 1 <= c["target_circuits"] <= row["m"]                  # a lower bound before running (ZNE, NIL)
        if row["m"] == 1:
            assert c["target_circuits"] == 1
        study_cost = cost_of(row)
        for B in GRID:                                                 # the feasibility labels the model learned
            assert (B >= c["min_budget"]) == allocate(study_cost, B, pol).feasible, (ms.key, B)


def test_live_noise_features_come_from_the_selected_device_not_the_injector():
    from qemx import rec_context as RX
    from qemx.noise import NoiseSpec
    from qemx.phase.lightcone import calculate_lambda_lc
    rec = next(r for r in RECS if r["cell"]["family"] == "qft")
    fc, comp, ctx = _live(rec)
    for inj in (NoiseSpec(), NoiseSpec(p1=0.01, p2=0.05), _injector()):
        c2 = RX.live_context(comp, ctx.backend, inj, fc.observable)
        v, _ = RX.live_features(c2, 1e4, ["lam_lc", "lam_ro", "scale", "coherent_fraction"])
        assert v == dict(lam_lc=rec["lambda"]["lam_lc"], lam_ro=rec["lambda"]["lam_ro"], scale=0.5,
                         coherent_fraction=0.0)
    inj_lam = calculate_lambda_lc(comp, _injector(), fc.observable).lam_lc
    assert abs(inj_lam - rec["lambda"]["lam_lc"]) > 0.1                 # what 1.5.0's page computed instead
    # the injector backend: no device scale (NaN with a reason), never a silent 1.0
    c3 = RX.live_context(comp, "numpy", _injector(), fc.observable)
    v, m = RX.live_features(c3, 1e4, ["scale", "lam_lc"])
    assert math.isnan(v["scale"]) and "injector" in m["scale"] and v["lam_lc"] == pytest.approx(inj_lam)


def test_methods_json_v2_names_the_protocol_and_v1_is_refused(tmp_path):
    from qemx import rec_context as RX
    study = _study(tmp_path / "v2")
    assert study.schema == RX.SCHEMA and study.baseline == "RAW"
    assert "VD" not in study.candidates and "VD" in study.data["upper_bounds"]
    assert study.recipe("RAW").readout == "tensored"
    assert study.data["noise"]["devices"] == ["fake_fez"] and study.data["feature_schema"] == FT.VERSION
    text = RX.describe_protocol(study, "RAW")
    assert "RO-inv" in text and "tensored readout-error inversion" in text and "no QEM estimator" in text
    # 1.5.0's bare declarations
    v1 = tmp_path / "v1" / "tables"
    v1.mkdir(parents=True)
    (v1 / "methods.json").write_text(json.dumps({"RAW": {"estimator": "raw"}, "ZNE-R": {"estimator": "zne"}}), encoding="utf-8")
    old = RX.load_study(str(tmp_path / "v1"))
    assert not old.usable and old.schema == "qemx.methods/1" and old.candidates == ["RAW", "ZNE-R"]
    assert "qemx.methods/1" in old.problems[0] and "python -m qemx.study context" in old.problems[0]
    fc, comp, ctx = _live(RECS[0])
    problems, _ = RX.compatibility(old, ctx, 1e4)
    assert problems == list(old.problems)


def test_compatibility_is_checked_before_prediction(tmp_path):
    from qemx import rec_context as RX
    from qemx.hamiltonian import parse_pauli_sum
    study = _study(tmp_path)
    rec = next(r for r in RECS if r["cell"]["family"] == "qft")
    fc, comp, ctx = _live(rec)
    names = list(FT.SETS["structural"])
    cols = ["x_" + n for n in names if n != "log10_budget"] + ["x_log10_budget"]
    assert RX.compatibility(study, ctx, 1e4, names, cols)[0] == []

    def problems(backend=None, B=1e4, observable=None, cols_=cols, names_=names):
        c = RX.live_context(comp, backend or ctx.backend, _injector(), observable or fc.observable)
        return " | ".join(RX.compatibility(study, c, B, names_, cols_)[0])
    assert "noise injector" in problems("numpy") and "noise injector" in problems("aer")
    assert "fake_torino" in problems("fake_torino")
    assert "outside the studied scales" in problems("fake_fez@s=4.0,eng=exact")
    assert "readout" in problems("fake_fez@ro=0.0,eng=exact")
    assert "homogenized" in problems("fake_fez@hom=1,eng=exact")
    assert "budget" in problems(B=500) and "budget" in problems(B=1e9)
    assert "pauli-sum" in problems(observable=parse_pauli_sum("Z0*Z1 + X1*X2", 3))
    assert problems("fake_fez") == ""                                   # Aer engine, same device model: a note only
    assert any("engine" in nt for nt in RX.compatibility(study, RX.live_context(
        comp, "fake_fez", _injector(), fc.observable), 1e4)[1])
    # schema /1 tables: renamed columns with identical values are read through the aliases
    norm = list(FT.SETS["normalized"])
    old_cols = [("x_" + FT.NEW_TO_OLD[n][0]) if n in FT.NEW_TO_OLD else "x_" + n for n in norm]
    assert problems(names_=norm, cols_=old_cols) == ""
    assert "not in the study tables" in problems(names_=norm, cols_=cols)


def test_context_cli_migrates_a_v1_report(tmp_path):
    """``python -m qemx.study context`` writes /2 next to an existing report, keeps the 1.5.0 file and records the
    tables' feature schema (detected from their columns)."""
    import shutil
    from qemx import rec_context as RX
    from qemx import study as SY
    run = tmp_path / "run"
    (run / "cells").mkdir(parents=True)
    for p in glob.glob(os.path.join(DATA, "*.json")):
        shutil.copy(p, run / "cells")
    man = os.path.join(os.path.dirname(DATA), "..", "..", "manifests", "direction_A_v2_pilot.yaml")
    rep = tmp_path / "report" / "tables"
    rep.mkdir(parents=True)
    (rep / "methods.json").write_text(json.dumps({"RAW": {"estimator": "raw"}}), encoding="utf-8")
    (rep / "cells_features.csv").write_text("cell_id,x_n,x_routing_ratio,x_two_qubit_density\n", encoding="utf-8")
    assert SY.main(["context", str(run), "--manifest", man, "--out", str(tmp_path / "report")]) == 0
    st = RX.load_study(str(tmp_path / "report"))
    assert st.usable and st.feature_schema == "qemx.features/1" and st.data["cells"] == len(RECS)
    assert json.loads((rep / "methods.v1.json").read_text(encoding="utf-8")) == {"RAW": {"estimator": "raw"}}
    assert st.recipe("CDR").readout == "tensored"
