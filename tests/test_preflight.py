"""1.8.0: the shared preflight contract, the deployment guard, factorized calibration and bundle compatibility
(review findings F1, F2, F4; docs/CI_FAILURE_ANALYSIS.md is separate)."""
import json
import math
import os
import shutil

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = json.load(open(os.path.join(ROOT, "tests", "data", "deployment_regression_cases_1.7.0.json"),
                       encoding="utf-8"))["cases"]
DEPLOYABLE = os.path.join(ROOT, "qemx", "selection", "bundles", "qemx-rec-1.7.0-deployable")
STUDY_SEED = 20261101 % (2 ** 31)          # the confirmation study's allocation design (labels.label_task)


def _task(spec_json, B):
    from qemx.selection import deploy as DP
    from qemx.selection import task as TK
    return DP.from_spec(TK.TaskSpec.from_json(spec_json), int(B))


# ------------------------------------------------------------------------------------------- the ten historical cases
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_historical_deployment_regressions_are_resolved_not_hidden(case):
    """1.7.0 accepted CDR/NIL here although the benchmark's allocation had no training circuits (degenerate training
    labels). 1.8.0: structural preflight says *unverified*, the design preflight (study seed and deployment seed)
    *infeasible*; the policy skips it, records why, and the choice is a verified candidate."""
    from qemx.selection import deploy as DP
    from qemx.selection import preflight as PF
    t = _task(case["task_spec"], case["budget"])
    comp = t.compile()
    k = case["observed_deployment_choice"]
    st = PF.structural(comp, k, case["budget"])
    assert st.status == PF.UNVERIFIED and not st.feasible
    for seed in (STUDY_SEED, DP.design_seed(t, k)):
        d = PF.design(comp, t.obs(), k, case["budget"], supplied=t.supplied_backend, seed=seed)
        assert d.status == PF.INFEASIBLE and d.evidence["degenerate"] and d.level == "design", d.to_json()
    b = DP.load_bundle(DEPLOYABLE)
    r = DP.recommend(t, b, design_seeds={"CDR": STUDY_SEED, "NIL": STUDY_SEED})
    assert r["choice"] != k and r["preflight"][r["choice"]]["status"] == PF.VERIFIED
    assert k in [s["protocol"] for s in r["skipped_candidates"]]
    assert r["choice"] == case["benchmark_choice"]              # here the benchmark's choice, verified
    assert r["policy"] == DP.POLICY


def test_unverified_candidates_are_never_accepted():
    """With preflight resolution switched off (level 'structural'), learning candidates stay unverified and are never
    the choice or the fallback."""
    from qemx.selection import deploy as DP
    from qemx.selection import preflight as PF
    b = DP.load_bundle(DEPLOYABLE)
    for c in CASES[:3]:
        r = DP.recommend(_task(c["task_spec"], c["budget"]), b, preflight_level="structural")
        assert r["preflight"][r["choice"]]["status"] == PF.VERIFIED
        assert r["preflight"][r["fallback"]]["status"] == PF.VERIFIED
        assert all(x["status"] == PF.UNVERIFIED for x in r["skipped_candidates"] if x["protocol"] in ("CDR", "NIL"))


def test_structural_preflight_matches_the_allocation_rule_on_stored_study_records():
    """RAW/RO/ZNE/PEC: verified/infeasible exactly as protocols.allocate decided in the 1.6.0 pilot fixtures (the full
    1.7.0 agreement, 27,760 rows of all partitions, is in the release evidence)."""
    from qemx.selection import preflight as PF
    from qemx.selection import task as TK
    d = os.path.join(ROOT, "reports", "selection_pilot", "dev", "tasks")
    if not os.path.isdir(d):
        d = os.path.join(ROOT, "reports", "selection_v170", "dev", "tasks")
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))[::15]
    n = 0
    for f in files:
        r = json.load(open(os.path.join(d, f), encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        comp = TK.build(TK.TaskSpec.from_json(r["spec"])).compiled
        for a in r["allocations"]:
            if a["protocol"] in ("CDR", "NIL"):
                continue
            p = PF.structural(comp, a["protocol"], a["B"])
            assert p.status in (PF.VERIFIED, PF.INFEASIBLE)
            assert (p.status == PF.VERIFIED) == bool(a["feasible"]), (a, p.to_json())
            n += 1
    assert n >= 100


def test_preflight_results_are_plain_json_and_use_no_shots():
    from qemx.selection import preflight as PF
    t = _task(CASES[0]["task_spec"], 30000)
    comp = t.compile()
    for k in ("RAW", "RO", "ZNE-E", "PEC", "CDR", "NIL"):
        r = PF.resolve(comp, t.obs(), k, 30000, supplied=t.supplied_backend, seed=1)
        j = json.loads(json.dumps(r.to_json(), allow_nan=False))
        assert j["shots_used"] == 0 and j["status"] in PF.STATUSES and j["level"] in PF.LEVELS
    with pytest.raises(ValueError):
        PF.Preflight("RAW", 1, "maybe", "design")


# --------------------------------------------------------------------------------------- factorized calibration (F2)
@pytest.mark.parametrize("device,layout", [("fake_fez", "best"), ("fake_torino@s=2,ro=2", "alt"),
                                           ("fake_kyiv@chi=0.5", "best"), ("fake_fez@s=0.5,ro=0", "alt")])
def test_factorized_calibration_equals_the_dense_engine(device, layout):
    from qemx import device as DV
    from qemx import device_exact as DE
    from qemx.methods import Context, Runner
    from qemx.selection import protocols as PR
    from qemx.selection import task as TK
    spec = TK.TaskSpec.from_json(dict(family="hea", n=4, params=[["layers", 2], ["angle_seed", 3]], inputs="product", observable="default",
                                      device=device, layout=layout, opt_level=1, seed=11, truth=""))
    comp = TK.build(spec).compiled
    phys = tuple(comp.info["phys"])
    dm = DV.get_model(spec.supplied_backend, phys)
    ctx = Context(comp, dm.equivalent_noise(), None, PR.PORTFOLIO["PEC"].recipe(),
                  Runner(backend=spec.supplied_backend, layout=phys))
    dense = ctx.calibration()["A"]
    fact = DE.product_calibration(comp.implemented.n, dm, comp.measure_map)
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(dense, fact)) < 1e-12


def test_a_circuit_with_two_qubit_instructions_is_not_factorized():
    from qemx import device as DV
    from qemx import device_exact as DE
    from qemx.selection import task as TK
    comp = TK.build(TK.make_spec("ghz", 3)).compiled
    dm = DV.get_model("fake_fez@eng=exact", tuple(comp.info["phys"]))
    with pytest.raises(DE.NotFactorizable):
        DE.product_marginals(comp.implemented, dm)


# ------------------------------------------------------------------------------------- bounded-cost guard (F2)
def _chain_task(n, budget=100000, device="fake_fez"):
    from qemx.phase.families import best_chain
    from qemx.selection import deploy as DP
    ops = (("ry", (0,), (1.0,), "gate", ""),) + tuple(("cx", (i, i + 1), (), "gate", "") for i in range(n - 1))
    return DP.DeploymentTask(n, ops, tuple("0" * n), "Z" * n, False, ("z_mid", "Z on the middle bit", ()), device,
                             budget, initial_layout=tuple(best_chain(device, n)), obs_terms=(((n // 2,), 1.0),))


def test_wide_request_never_reaches_the_dense_engine(monkeypatch):
    """Sentinel: a 24-qubit experiment (a dense 24-qubit density matrix would need 4 PiB... 2^48·16 bytes) is
    answered — out-of-distribution, SBS-B fallback — with every dense path replaced by a tripwire."""
    from qemx import device_exact as DE
    from qemx.backends import numpy_dm
    from qemx.selection import deploy as DP

    def tripwire(*a, **k):
        raise AssertionError("a dense engine was called")
    monkeypatch.setattr(DE, "final_state", tripwire)
    monkeypatch.setattr(numpy_dm.NumpyBackend, "execute", tripwire)
    b = DP.load_bundle(DEPLOYABLE)
    r = DP.recommend(_chain_task(24), b)
    assert r["status"] == "out-of-distribution" and not r["accepted"]
    assert r["feature_report"]["calibration_method"] == "factorized"
    assert np.isfinite(r["features"]["pec_log_gamma"]) and np.isfinite(r["features"]["lam_lc"])
    assert r["preflight"][r["choice"]]["status"] == "verified-feasible"


def test_guard_rejects_before_compilation(monkeypatch):
    from qemx.selection import deploy as DP
    b = DP.load_bundle(DEPLOYABLE)

    def no_compile(self):
        raise AssertionError("compiled before the guard")
    monkeypatch.setattr(DP.DeploymentTask, "compile", no_compile)
    t = _chain_task(6)
    for bad, why in ((dict(budget=0), "positive integer"), (dict(n=400, ops=(), inputs=tuple("0" * 400),
                                                                basis="Z" * 400, initial_layout=None,
                                                                obs_terms=(((1,), 1.0),)), "do not fit"),
                     (dict(device="numpy"), "fake-device"),
                     (dict(n=22, ops=(), inputs=tuple("0" * 22), basis="Z" * 22, initial_layout=None,
                           observable=("o", "o", (0.0,) * 8), obs_terms=None), "dense observable"),
                     (dict(obs_terms=(((0,), 0.8), ((1,), 0.8))), "|O| ≤ 1")):
        import dataclasses
        with pytest.raises(DP.Unsupported, match=why):
            DP.recommend(dataclasses.replace(t, **bad), b)


def test_compact_and_dense_observables_give_identical_features():
    from qemx.selection import deploy as DP
    b = DP.load_bundle(DEPLOYABLE)
    t = _chain_task(5)
    dense = DP.DeploymentTask(**{**t.__dict__, "observable": ("z_mid", "Z on the middle bit",
                                                               tuple(t.obs().o)), "obs_terms": None})
    v1, _ = DP.features(t, b.features)
    v2, _ = DP.features(dense, b.features)
    for k in b.features:
        assert v1[k] == pytest.approx(v2[k], rel=1e-12, abs=1e-15), k
    assert t.key() != dense.key()                          # different representations, both stable identities


# --------------------------------------------------------------------------------------- bundle compatibility (F4)
def _copy(tmp_path):
    d = tmp_path / "b"
    shutil.copytree(DEPLOYABLE, d)
    return d


def _edit(d, fn):
    p = d / "bundle.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    fn(m)
    p.write_text(json.dumps(m, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("change,match", [
    (lambda m: m["protocol_definitions"]["ZNE-L"]["params"].__setitem__(3, ["scales", [1.0, 2.0]]), "ZNE-L"),
    (lambda m: m["protocol_definitions"]["RO"].__setitem__("readout", "m3"), "candidate RO"),
    (lambda m: m["feature_classes"].__setitem__("lam_lc", "simulated"), "information class"),
])
def test_scientifically_incompatible_bundles_are_refused(tmp_path, change, match):
    from qemx.selection import deploy as DP
    d = _copy(tmp_path)
    _edit(d, change)
    with pytest.raises(DP.BundleIncompatible, match=match):
        DP.load_bundle(str(d))


def test_changed_feature_meaning_is_refused(monkeypatch):
    from qemx.selection import deploy as DP
    from qemx.selection import features as FE
    f = FE.BY_NAME["lam_lc"]
    monkeypatch.setitem(FE.BY_NAME, "lam_lc", type(f)(f.name, f.info, f.group, f.definition + " (redefined)"))
    with pytest.raises(DP.BundleIncompatible, match="feature definitions differ"):
        DP.load_bundle(DEPLOYABLE)


def test_new_calibration_snapshot_is_a_transfer_not_an_error(monkeypatch):
    from qemx import device as DV
    from qemx.selection import deploy as DP
    monkeypatch.setattr(DV, "snapshot_fingerprint", lambda name: "0" * 16)
    b = DP.load_bundle(DEPLOYABLE)
    assert len(b.meta["compatibility"]["transfer"]) == 3 and not b.meta["compatibility"]["errors"]
    r = DP.recommend(_task(CASES[0]["task_spec"], 30000), b)
    assert r["status"] == "out-of-distribution" and any("snapshot" in x for x in r["ood"])


def test_shipped_bundles_are_compatible_and_their_legacy_fingerprints_are_current():
    from qemx import device as DV
    from qemx.selection import deploy as DP
    for name in ("qemx-rec-1.7.0", "qemx-rec-1.7.0-deployable"):
        b = DP.load_bundle(os.path.join(ROOT, "qemx", "selection", "bundles", name))
        assert b.meta["compatibility"] == dict(errors=[], transfer=[], checks=b.meta["compatibility"]["checks"])
        assert DP.LEGACY_FINGERPRINTS["features"][name] == DP.feature_fingerprint(b.features)
    for d, fp in DP.LEGACY_FINGERPRINTS["snapshots"].items():
        assert DV.snapshot_fingerprint(d) == fp


def test_design_seed_is_stable_and_budget_free():
    from qemx.selection import deploy as DP
    t = _task(CASES[0]["task_spec"], 30000)
    t2 = _task(CASES[0]["task_spec"], 3000000)
    assert DP.design_seed(t, "CDR") == DP.design_seed(t2, "CDR") != DP.design_seed(t, "NIL")
    assert 0 <= DP.design_seed(t, "CDR") < 2 ** 31
    assert not math.isnan(DP.design_seed(t, "CDR"))
