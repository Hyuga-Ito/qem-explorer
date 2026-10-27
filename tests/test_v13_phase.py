"""v1.3 · Direction A: device specifications, the exact device engine, λ_LC, noise scaling, the integer ledger,
fold survival, the cell runner, validation, manifests and storage."""
from __future__ import annotations

import itertools
import json
import math
import os
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")

from qemx import device as D
from qemx.engine import simulate
from qemx.noise import NoiseSpec, Instruction, depol_mixture, kraus_amp, kraus_phase, rz, rzz
from qemx.phase import families as FAM
from qemx.phase import ledger as L
from qemx.phase import lightcone as LC
from qemx.phase import scaling as SC
from qemx.textio import child_env


@pytest.fixture(scope="module")
def qft4():
    fc = FAM.build("qft", 4)
    return fc, FAM.compile_family(fc, "fake_fez", initial_layout=(136, 143, 142, 141))


@pytest.fixture(scope="module")
def ising4():
    fc = FAM.build("ising", 4, steps=2)
    return fc, FAM.compile_family(fc, "fake_fez", initial_layout=(136, 143, 142, 141))


# ------------------------------------------------------------------ device specifications
def test_device_spec_parsing():
    comp = {"g1": 1.0, "g2": 1.0, "t": 1.0}    # 1.7.0: component scales (noise composition), defaults 1
    assert D.parse_device("fake_fez") == ("fake_fez", {"s": 1.0, "ro": 1.0, "eng": "aer", "hom": 0.0, "chi": 0.0,
                                                      **comp})
    assert D.parse_device("fake_fez@s=2.5,ro=0,eng=exact") == ("fake_fez", {"s": 2.5, "ro": 0.0, "eng": "exact",
                                                                            "hom": 0.0, "chi": 0.0, **comp})
    assert D.parse_device("fake_fez@g2=3,t=2")[1] == {"s": 1.0, "ro": 1.0, "eng": "aer", "hom": 0.0, "chi": 0.0,
                                                      "g1": 1.0, "g2": 3.0, "t": 2.0}
    assert D.device_spec("fake_fez", 2.0, 1.0, "exact", hom=True, chi=0.25) == "fake_fez@hom=1,chi=0.25,s=2.0,eng=exact"
    assert D.is_device("fake_fez@s=2") and not D.is_device("numpy") and not D.is_device("aer")
    assert D.base_name("fake_fez@s=2,eng=exact") == "fake_fez"
    assert D.device_spec("fake_fez", 2.0, 1.0, "exact") == "fake_fez@s=2.0,eng=exact"
    assert D.supports_negative("numpy") and D.supports_negative("fake_fez@eng=exact")
    assert not D.supports_negative("fake_fez") and not D.supports_negative("aer")
    for bad in ("fake_fez@s=-1", "fake_fez@s=0", "fake_fez@x=1", "fake_fez@s=1,s=2", "fake_fez@eng=gpu",
                "fake_fez@s=nan", "fake_fez@s=inf", "fake_fez;rm -rf", "Fake_Fez"):
        with pytest.raises(ValueError):
            D.parse_device(bad)


def test_scaled_model_is_exact(qft4):
    _, comp = qft4
    phys = tuple(comp.info["phys"])
    a, b = D.get_model("fake_fez", phys), D.get_model("fake_fez@s=2.5,ro=0.5", phys)
    for k, e in a.errors.items():
        assert (e is None and b.errors[k] is None) or b.errors[k] == pytest.approx(2.5 * e, rel=1e-15)
    for qa, qb in zip(a.qubits, b.qubits):
        assert qb.t1 == pytest.approx(qa.t1 / 2.5) and qb.t2 == pytest.approx(qa.t2 / 2.5)
        assert qb.p01 == pytest.approx(0.5 * qa.p01) and qb.p10 == pytest.approx(0.5 * qa.p10)
    assert a.durations == b.durations
    with pytest.raises(ValueError):
        D.get_model("fake_fez@s=1000", phys)
    with pytest.raises(ValueError):
        D.get_model("fake_fez@ro=500", phys)


# ------------------------------------------------------------------ exact device engine
@pytest.mark.parametrize("twirl,dd,spec", [("none", "none", "fake_fez"), ("2q", "XY4", "fake_fez@s=3,ro=0.5"),
                                           ("all", "none", "fake_fez@s=0.5")])
def test_exact_engine_matches_aer(ising4, twirl, dd, spec):
    _, comp = ising4
    kw = dict(measure_map=comp.measure_map, twirl=twirl, dd=dd, layout=comp.info["phys"])
    a = simulate(comp.implemented, NoiseSpec(), spec, **kw)
    ex = spec + (",eng=exact" if "@" in spec else "@eng=exact")
    b = simulate(comp.implemented, NoiseSpec(), ex, **kw)
    # Aer's internal channel representation differs from the declared QuantumError by up to ~1e-9 per gate
    assert np.abs(a.rho - b.rho).max() < 1e-7
    assert np.abs(a.probs_meas - b.probs_meas).max() < 1e-7


def test_exact_kraus_reproduces_channel():
    from qiskit.quantum_info import Kraus, SuperOp
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
    from qemx.device_exact import exact_kraus
    e = depolarizing_error(3e-3, 2).compose(thermal_relaxation_error(80e-6, 60e-6, 70e-9).tensor(
        thermal_relaxation_error(120e-6, 90e-6, 70e-9)))
    K = exact_kraus(e)
    assert np.abs(SuperOp(Kraus(K)).data - SuperOp(e.to_quantumchannel()).data).max() < 1e-13


def test_pec_exact_on_device_and_aer_refuses_signed_maps(qft4):
    from qemx.methods import Recipe
    from qemx.recipes import compare
    fc, comp = qft4
    rc = Recipe.make("pec", "none", "none", "none", instances=128)
    ideal, outs, _ = compare(comp, NoiseSpec(), fc.observable, [rc], backend="fake_fez@eng=exact", seed=3)
    ex = outs[0].result
    assert ex.details["p_net"] is not None and ex.details["mc_err"] == 0.0
    assert ex.details["p_net"].min() < 0                       # a signed quasi-distribution, kept as is
    assert ex.details["p_abs"].sum() == pytest.approx(ex.details["gamma_path"][-1], rel=1e-9)
    x = ex.sampler(10 ** 6, 400, np.random.default_rng(0))
    assert abs(x.mean() - ex.mean) < 5 * math.sqrt(ex.var1 / 1e6) / math.sqrt(400) + 1e-12
    ideal2, outs2, _ = compare(comp, NoiseSpec(), fc.observable, [rc], backend="fake_fez", seed=3)
    aer = outs2[0].result
    assert abs(aer.mean - ex.mean) < 4 * aer.details["mc_err"]
    # a signed correction on the Aer engine is an error, never silently wrong
    corr = lambda o, step: [Instruction("mixture", o.qubits, depol_mixture(0.01, o.k), step, "pec", "x")] \
        if o.kind == "gate" and o.k == 2 else []
    with pytest.raises(ValueError):
        simulate(comp.implemented, NoiseSpec(), "fake_fez", measure_map=comp.measure_map, corrections=corr,
                 layout=comp.info["phys"])


# ------------------------------------------------------------------ fault probabilities and λ
def _ins(kind, qs, data, src="noise", label=""):
    return Instruction(kind, qs, data, 0, src, label)


@pytest.mark.parametrize("x", [1e-4, 0.01, 0.2, 0.7])
def test_fault_probability_closed_forms(x):
    assert LC.fault_probability(_ins("mixture", (0,), depol_mixture(x, 1))) == pytest.approx(SC.pf_depol(x, 1))
    assert LC.fault_probability(_ins("mixture", (0, 1), depol_mixture(x, 2))) == pytest.approx(SC.pf_depol(x, 2))
    assert LC.fault_probability(_ins("kraus", (0,), kraus_amp(x))) == pytest.approx(SC.pf_amp(x))
    assert LC.fault_probability(_ins("kraus", (0,), kraus_phase(x))) == pytest.approx(SC.pf_phase(x))
    assert LC.fault_probability(_ins("unitary", (0,), rz(x))) == pytest.approx(SC.pf_angle(x))
    assert LC.fault_probability(_ins("unitary", (0, 1), rzz(x))) == pytest.approx(SC.pf_angle(x))
    # restricted to qubit 0, a two-qubit depolarizing channel counts the Paulis that act on qubit 0: 12 of 16
    assert LC.fault_probability(_ins("mixture", (0, 1), depol_mixture(x, 2)), on={0}) == pytest.approx(x * 12 / 16)


def test_scale_noise_is_exactly_linear_in_lambda(ising4):
    _, comp = ising4
    ns = NoiseSpec(p1=4e-4, p2=6e-3, amp_damp=1e-3, phase_damp=8e-4, zz_phase=0.05, z_phase=0.02, kick=0.03,
                   idle_dephase=2e-3, idle_damp=1e-3, ro01=0.01, ro10=0.02)
    lam0 = LC.calculate_lambda_lc(comp, ns, None)
    for s in (0.25, 2.0, 5.0):
        lam = LC.calculate_lambda_lc(comp, SC.scale_noise(ns, s), None)
        assert lam.lam_total == pytest.approx(s * lam0.lam_total, rel=1e-10)
        assert lam.lam_lc == pytest.approx(s * lam0.lam_lc, rel=1e-10)
        assert lam.lam_ro == pytest.approx(lam0.lam_ro)          # readout is not scaled by default
    # v1.2's NoiseSpec.scaled multiplies coherent angles by s, so their fault probability grows like s²
    coh = NoiseSpec(zz_phase=0.1)
    l1 = LC.calculate_lambda_lc(comp, coh, None).lam_total
    assert LC.calculate_lambda_lc(comp, coh.scaled(3.0), None).lam_total == pytest.approx(9 * l1, rel=0.01)
    assert LC.calculate_lambda_lc(comp, SC.scale_noise(coh, 3.0), None).lam_total == pytest.approx(3 * l1, rel=1e-10)
    with pytest.raises(ValueError):
        SC.scale_noise(NoiseSpec(p2=0.3), 4.0)


def test_scale_over_rotation_exact_on_sqrt_x():
    from qemx.circuit import CircuitSpec, Op
    spec = CircuitSpec(1, tuple(Op("sx", (0,), (), "gate") for _ in range(5)))
    ns = NoiseSpec(over_rot=0.02)
    l1 = LC.calculate_lambda_lc(spec, ns, None).lam_total
    assert LC.calculate_lambda_lc(spec, SC.scale_noise(ns, 3.0), None).lam_total == pytest.approx(3 * l1, rel=1e-10)


@pytest.mark.parametrize("chi", [0.0, 0.25, 0.5])
def test_calibrated_noise_reproduces_infidelities(qft4, chi):
    _, comp = qft4
    cal = SC.calibration_from_model(D.get_model("fake_fez", tuple(comp.info["phys"])))
    ns = SC.calibrated_noise(cal, chi)
    inf = SC.gate_infidelities(ns)
    assert inf["r1"] == pytest.approx(cal.r1, rel=1e-9) and inf["r2"] == pytest.approx(cal.r2, rel=1e-9)
    assert (2 / 3) * math.sin(ns.over_rot * math.pi / 4) ** 2 == pytest.approx(chi * cal.r1, abs=1e-15)
    assert 0.8 * math.sin(ns.zz_phase / 2) ** 2 == pytest.approx(chi * cal.r2, abs=1e-15)
    assert ns.idle_damp == pytest.approx(1 - math.exp(-cal.t_2q / cal.t1))
    assert ns.idle_dephase == pytest.approx(1 - math.exp(-cal.t_2q * (2 / cal.t2 - 1 / cal.t1)))


def test_device_lambda_scales_and_light_cone(qft4, ising4):
    _, comp = qft4
    l1 = LC.calculate_lambda_lc(comp, "fake_fez", FAM.build("qft", 4).observable)
    l2 = LC.calculate_lambda_lc(comp, "fake_fez@s=2", FAM.build("qft", 4).observable)
    assert l2.lam_lc / l1.lam_lc == pytest.approx(2.0, rel=0.02)    # first-order linear (relaxation saturates)
    assert l1.lam_lc_exact >= l1.lam_lc
    assert l1.lam_lc_gate + l1.lam_lc_idle == pytest.approx(l1.lam_lc)
    # a local observable at r = 1 Trotter step: the light cone leaves the far qubit out
    fc = FAM.build("ising", 6, steps=1)
    c6 = FAM.compile_family(fc, "fake_fez", initial_layout=(124, 123, 136, 143, 142, 141))
    rep = LC.calculate_lambda_lc(c6, "fake_fez", fc.observable)
    assert rep.lam_lc < rep.lam_total and len(rep.support) == 1


def test_light_cone_hand_example():
    from qemx.circuit import CircuitSpec, Op
    # q1 alone carries noise; observable Z0: without an interaction q1 is outside the light cone
    spec = CircuitSpec(2, (Op("sx", (1,), (), "gate"), Op("sx", (0,), (), "gate")))
    from qemx.observables import z
    ns = NoiseSpec(p1=0.01)
    r = LC.calculate_lambda_lc(spec, ns, z(0, 2))
    assert r.lam_lc == pytest.approx(0.01 * 3 / 4) and r.lam_total == pytest.approx(2 * 0.01 * 3 / 4)
    # a CZ after both couples q1 into the light cone of Z0: every fault location counts
    spec2 = spec.replace(ops=spec.ops + (Op("cz", (0, 1), (), "gate"),))
    r2 = LC.calculate_lambda_lc(spec2, ns, z(0, 2))
    assert r2.lam_lc == pytest.approx(r2.lam_total)


# ------------------------------------------------------------------ the ledger
def test_ledger_fixed_policy_brute_force():
    c = L.Cost("x", b=0.01, v=0.8, w=3.0, c=2)
    for B in (100, 1000, 12345):
        a = L.allocate(c, B, L.Policy("fixed"))
        best = min((c.v / (B - 2 * s) + c.w / s, s) for s in range(1, (B - 1) // 2 + 1) if B - 2 * s >= 1)
        assert a.N + 2 * a.s == B and a.var == pytest.approx(best[0], rel=1e-14)
    a = L.allocate(c, 1000, L.PIPELINE_FIXED)
    assert a.s >= 32


def test_ledger_opt_policy_first_order_brute_force():
    c = L.Cost("nil", b=0.002, v=1.3, w=2.0, c=2, T=16, s_tr=1024, u1=0.9)
    for B in (2000, 5001):
        a = L.allocate(c, B, L.Policy("opt"))
        best = min(c.v / (B - 2 * s - 16 * t) + c.w / s + c.u1 / t for s in range(1, B // 2) for t in range(1, B // 16)
                   if B - 2 * s - 16 * t >= 1)
        assert a.N + 2 * a.s + 16 * a.s_tr == B and a.var == pytest.approx(best, rel=1e-13)
    a = L.allocate(c, 10 ** 8, L.Policy("opt", integer=False))
    assert a.var == pytest.approx((math.sqrt(c.v) + math.sqrt(16 * c.u1) + math.sqrt(2 * c.w)) ** 2 / 1e8, rel=1e-9)


def test_ledger_levels_policy():
    c = L.Cost("cdr", b=0.0, v=1.0, w=0.0, c=0, T=16, s_tr=1024,
               levels=((64, 0.02, 1e-2), (1024, 0.004, 1e-3), (16384, 0.001, 1e-4)))
    a = L.allocate(c, 10 ** 6, "fixed")
    assert a.s_tr == 1024 and a.b == 0.004 and a.var_train == 1e-3 and a.N == 10 ** 6 - 16 * 1024
    b = L.allocate(c, 10 ** 7, "opt")
    cands = [((bb ** 2 + u + 1.0 / (10 ** 7 - 16 * s)), s) for s, bb, u in c.levels]
    assert b.s_tr == min(cands)[1]
    assert L.floor_mse(c, "fixed") == pytest.approx(0.004 ** 2 + 1e-3)
    with pytest.raises(KeyError):
        L.Cost("x", 0, 1, T=2, s_tr=100, levels=((64, 0, 0),)).at_level(100)


def test_integer_split_and_budget_identity_random():
    from qemx.budget import optimal_integer_split
    rng = np.random.default_rng(5)
    for _ in range(40):
        a = rng.uniform(0.01, 3, 3)
        N = int(rng.integers(3, 60))
        var, nj = optimal_integer_split(a, N)
        best = min(sum(x ** 2 / n for x, n in zip(a, ns)) for ns in itertools.product(range(1, N + 1), repeat=3)
                   if sum(ns) == N)
        assert sum(nj) == N and var == pytest.approx(best, rel=1e-14)
    for _ in range(200):
        c = L.Cost("r", b=float(rng.normal(0, .01)), v=float(rng.uniform(0, 5)), w=float(rng.uniform(0, 1)),
                   c=int(rng.integers(0, 3)), T=int(rng.integers(0, 3)) * 8, s_tr=int(rng.integers(1, 2000)),
                   u1=float(rng.uniform(0, 2)), m=3, a=tuple(rng.uniform(0, 1, 3)))
        for pol in (L.Policy("fixed", True, 32, 32), L.Policy("opt", True, 1, 16)):
            B = int(rng.integers(L.min_budget(c, pol), 10 ** 7))
            a = L.allocate(c, B, pol)
            assert a.feasible
            assert a.N + c.c * a.s + c.T * a.s_tr == B and sum(a.N_j) == a.N


def test_budget_for_is_minimal_and_crossover_exact():
    c = L.Cost("x", b=0.01, v=1.0)
    B = L.budget_for(c, 0.02)
    assert L.rmse(c, B) <= 0.02 < L.rmse(c, B - 1)
    assert math.isinf(L.budget_for(c, 0.01))
    x = L.crossovers(L.Cost("a", b=0.05, v=1.0), L.Cost("b", b=0.0, v=50.0))
    assert x == [(19600.0, "b")]                                  # 49/B = 0.05²  ->  B = 19600


# ------------------------------------------------------------------ folding must survive compilation
def test_fold_guard_on_device(qft4):
    from qemx.phase.cell import fold_guard
    _, comp = qft4
    g = fold_guard(comp, "fake_fez@eng=exact")
    assert g["spec_2q"] == [g["base_2q"] * k for k in (1, 3, 5)] == g["native_2q"]


def test_qiskit_transpiler_behaviour_on_folds():
    """Documents Qiskit 2.x: optimization level 0 keeps folded CZ triples; levels 1-3 cancel them unless barriers
    separate the fold segments. The hardware path transpiles at level 0 and checks the counts."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime.fake_provider import FakeFez
    b = FakeFez()
    qc = QuantumCircuit(2)
    qc.cz(0, 1), qc.cz(0, 1), qc.cz(0, 1)
    qc.measure_all()
    counts = [transpile(qc, b, optimization_level=k, initial_layout=[0, 1], seed_transpiler=1).count_ops().get("cz", 0)
              for k in range(4)]
    assert counts[0] == 3 and all(c == 1 for c in counts[1:])
    qb = QuantumCircuit(2)
    qb.cz(0, 1), qb.barrier(), qb.cz(0, 1), qb.barrier(), qb.cz(0, 1)
    qb.measure_all()
    assert all(transpile(qb, b, optimization_level=k, initial_layout=[0, 1], seed_transpiler=1).count_ops()["cz"] == 3
               for k in range(4))


# ------------------------------------------------------------------ cell runner, validation, storage
CORE = None


def _core():
    from qemx.phase.cell import MethodSpec
    return (MethodSpec.make("RAW", "raw"),
            MethodSpec.make("ZNE-R", "zne", scales=[1.0, 3.0, 5.0], folding="global", rule="richardson"),
            MethodSpec.make("ZNE-E", "zne", scales=[1.0, 3.0, 5.0], folding="global", rule="exponential"),
            MethodSpec.make("PEC", "pec", construction="inverse", model="depolarizing", scope="all",
                            shots_per_circuit=1),
            MethodSpec.make("CDR", "learning", features="cdr", model="ols", n_train=8, train_shots=1024),
            MethodSpec.make("VD", "vd", M=2),
            MethodSpec.make("NIL-R", "learning", features="nil", model="ridge", alpha=1e-3, n_train=8,
                            train_shots=1024))


@pytest.fixture(scope="module")
def cell_record():
    from qemx.phase.cell import CellSpec, cell_runner
    spec = CellSpec(stage="test", family="qft", n=4, fparams=(("m", None), ("s", 1)),
                    layout=(136, 143, 142, 141), engine="E2", scale=1.5, methods=_core(), seed=11,
                    train_levels=(256, 1024), train_reps=300, train_reps_other=200,
                    validate_budgets=(10 ** 5,), validate_reps=600, validate_policy=("fixed", "opt"))
    return spec, cell_runner(spec)


def test_cell_record_contents(cell_record):
    from qemx.phase.store import clean, validate_record
    spec, rec = cell_record
    validate_record(json.loads(json.dumps(clean(rec), allow_nan=False)))
    assert rec["cell_id"] == spec.cell_id and len(spec.cell_id) == 20
    rows = {r["key"]: r for r in rec["methods"]}
    assert all(r["status"] == "ok" for r in rows.values())
    assert rows["PEC"]["details"]["exact"] and "mc-mean" not in rows["PEC"]["flags"]
    assert rows["ZNE-R"]["m"] == 3 and len(rows["ZNE-R"]["a"]) == 3
    assert sum(rows["ZNE-R"]["a"]) ** 2 == pytest.approx(rows["ZNE-R"]["v"], rel=1e-9)
    for k in ("CDR", "NIL-R"):
        assert rows[k]["T"] == 8 and rows[k]["s_tr"] == 1024 and rows[k]["F"] == 8 * 1024
        assert {l["s_tr"] for l in rows[k]["levels"]} == {256, 1024}
    assert rows["VD"]["c"] == 0 and "optimistic" in rows["VD"]["flags"]
    assert rec["guards"][0]["native_2q"] == [rec["circuit"]["two_qubit"] * k for k in (1, 3, 5)]
    lam = rec["lambda"]
    assert 0 < lam["lam_lc"] <= lam["lam_total"] + 1e-12 and lam["lam_lc_exact"] >= lam["lam_lc"]
    assert rec["provenance"]["calibration_hash"] and rec["provenance"]["qiskit"].startswith("2.")


def test_cell_validation_passes(cell_record):
    _, rec = cell_record
    vals = [v for v in rec["validation"] if v.get("feasible") and "mc" in v]
    assert vals and all(v["identity"] for v in vals)
    for v in vals:
        if v["independent"]:
            assert abs(v["z_rmse"]) < 5 and abs(v["z_bias"]) < 5, v
            assert v["mc"]["frac_nonfinite"] == 0


def test_cell_seed_is_shared_along_lambda_axis():
    from qemx.phase.manifest import load, expand
    m = load(os.path.join(os.path.dirname(__file__), "..", "manifests", "direction_A_v1.yaml"))
    cells = expand(m, ["A1-primary"])
    by = {}
    for c in cells:
        by.setdefault((c.family, c.n, c.fparams), set()).add(c.seed)
    assert all(len(s) == 1 for s in by.values())


def test_manifest_expansion():
    from qemx.phase.manifest import check, expand, load
    m = load(os.path.join(os.path.dirname(__file__), "..", "manifests", "direction_A_v1.yaml"))
    cells = expand(m)
    assert len(cells) == 1705 and len({c.cell_id for c in cells}) == 1705
    a1 = [c for c in cells if c.stage == "A1-primary"]
    strata = {}
    for c in a1:
        strata.setdefault((c.family, c.n), []).append(bool(c.validate_budgets))
    assert all(sum(v) == math.ceil(0.15 * len(v)) for v in strata.values())
    bad = dict(m)
    bad["stages"] = [dict(m["stages"][0], readout=True)]
    with pytest.raises(ValueError):
        check(bad)


def test_store_roundtrip_and_collect(tmp_path, cell_record):
    from qemx.phase.store import collect, iter_cells, read_cell, write_cell
    _, rec = cell_record
    write_cell(str(tmp_path), rec)
    back = read_cell(str(tmp_path), rec["cell_id"])
    assert back["cell_id"] == rec["cell_id"] and back["methods"][0]["key"] == "RAW"
    assert len(list(iter_cells(str(tmp_path)))) == 1
    n = collect(str(tmp_path), budgets=[10 ** 3, 10 ** 5, 10 ** 8])
    assert n["tables"]["methods"]["rows"] == len(rec["methods"])
    assert n["tables"]["winners"]["rows"] == 12                         # 3 budgets × 2 policies × 2 candidate sets
    import pandas as pd
    led = pd.read_csv(tmp_path / "tables" / "ledger.csv", encoding="utf-8")
    ok = led[led.feasible]
    assert (ok.rmse > 0).all()


# ------------------------------------------------------------------ fixes from the independent review
def test_readout_calibration_follows_measurement_map():
    """Tensored mitigation must use, for bit i, the confusion matrix of the qubit read into bit i (v1.2 used the
    matrix of qubit i). With gate noise switched off, mitigation must recover the ideal value exactly."""
    from qemx.methods import Recipe
    from qemx.recipes import compare
    fc = FAM.build("ising", 6, steps=2)
    comp = FAM.compile_family(fc, "fake_fez", initial_layout=(124, 123, 136, 143, 142, 141))
    assert any(k != v for k, v in comp.measure_map.items())             # the bits are permuted
    from qemx.observables import z
    for q in range(6):
        obs = z(q, 6)
        ideal, outs, _ = compare(comp, NoiseSpec(), obs, [Recipe.make("raw", "none", "none", "tensored")],
                                 backend="fake_fez@s=1e-12,ro=20.0,eng=exact", seed=1)
        assert abs(outs[0].result.mean - ideal) < 1e-9


def test_lambda_ignores_idle_before_first_gate():
    fc = FAM.build("ghz", 6)
    comp = FAM.compile_family(fc, "fake_fez", initial_layout=(124, 123, 136, 143, 142, 141))
    from qemx import device_exact
    dm = D.get_model("fake_fez", tuple(comp.info["phys"]))
    instr, _ = device_exact.instructions(comp.implemented, dm)
    started, lead = set(), 0.0
    for ins in instr:
        if ins.source == "gate":
            started.update(ins.qubits)
        elif ins.source == "idle" and ins.qubits[0] not in started:
            lead += LC.fault_probability(ins)
    assert lead > 0                                                     # the schedule pads the start of the wires
    rep = LC.calculate_lambda_lc(comp, dm, None)
    tot = sum(LC.fault_probability(i) for i in instr if i.source not in ("gate", "dd", "pec"))
    assert rep.lam_total == pytest.approx(tot - lead, rel=1e-12)


def test_no_dd_before_first_gate():
    from qemx.circuit import CircuitSpec, Op
    spec = CircuitSpec(2, (Op("sx", (0,), (), "gate"),) * 30 + (Op("cz", (0, 1), (), "gate"),))
    dm = D.get_model("fake_fez", (136, 143))
    qc, _ = D.device_circuit(spec, dm, dd="XY4")
    first = next(i for i, ins in enumerate(qc.data) if 1 in [qc.find_bit(q).index for q in ins.qubits]
                 and ins.operation.name not in ("quantum_channel", "save_density_matrix"))
    assert qc.data[first].operation.name == "cz"                       # qubit 1 idles undisturbed until its CZ


def test_validation_error_is_recorded_not_raised(monkeypatch, qft4):
    import qemx.phase.validate as V
    from qemx.phase.cell import CellSpec, MethodSpec, cell_runner
    monkeypatch.setattr(V, "validate_method", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    spec = CellSpec(stage="t", family="ghz", n=4, fparams=(("theta", 1.0471975511965976),),
                    layout=(136, 143, 142, 141), methods=(MethodSpec.make("RAW", "raw"),), seed=1,
                    validate_budgets=(10 ** 4,), validate_reps=10)
    rec = cell_runner(spec)
    assert rec["status"] == "partial" and rec["validation"][0]["error"].startswith("RuntimeError")


def test_manifest_rejects_typos():
    from qemx.phase.manifest import check, load
    m = load(os.path.join(os.path.dirname(__file__), "..", "manifests", "direction_A_v1.yaml"))
    for mutate in (lambda x: x["methods"]["ZNE-R"].update(rule="richardsn"),
                   lambda x: x["methods"]["NIL-L"].update(alpha="1.0e3"),
                   lambda x: x["methods"]["PEC"].update(instance=10),
                   lambda x: x["methods"].update(BAD={"estimator": "zne2"}),
                   lambda x: x["stages"][0]["validate"].update(policies=["optimal"]),
                   lambda x: x["stages"][0]["validate"].update(budgets=["1.0e3"]),
                   lambda x: x["stages"][0].update(coherent_fraction=[0.25]),
                   lambda x: x["defaults"].update(readout_mit="tensord"),
                   lambda x: x["stages"][0].update(devices=["fake_fezz"]),
                   lambda x: x["methods"]["ZNE-R"].update(scales=[0.5, 1.0, 3.0]),
                   lambda x: x["defaults"]["layout"].pop(6),
                   lambda x: x["stages"][0].update(devices=["fake_torino"]),
                   lambda x: x["stages"][0]["validate"].update(reps=1),
                   lambda x: x["stages"][0]["validate"].pop("budgets"),
                   lambda x: x["stages"][0].update(opt_levels=[7]),
                   lambda x: x["defaults"].update(s_min=0)):
        import copy
        bad = copy.deepcopy(m)
        mutate(bad)
        with pytest.raises(ValueError):
            check(bad)


def test_seed_streams_are_distinct():
    from qemx.phase.cell import CellSpec
    from qemx.phase.validate import stream_seed
    spec = CellSpec(stage="t", family="ghz", n=4, fparams=(), seed=3)
    a = stream_seed(spec, "train", "NIL-RL", 1024).generate_state(2)
    b = stream_seed(spec, "train", "NIL-LR", 1024).generate_state(2)                 # anagrams differ
    assert not np.array_equal(a, b)


# ------------------------------------------------------------------ E1: homogenized device with a coherent fraction
def test_homogenized_model_and_coherent_channels():
    from qiskit.quantum_info import Kraus, SuperOp, average_gate_fidelity
    from qemx import device_exact as X
    phys = (123, 124, 136, 141, 142, 143)
    base = D.get_model("fake_fez", phys)
    hom = D.get_model("fake_fez@hom=1", phys)
    assert len({(q.t1, q.t2, q.p01, q.p10) for q in hom.qubits}) == 1
    assert hom.qubits[0].t1 == pytest.approx(float(np.median([q.t1 for q in base.qubits])))
    cz = [e for (nm, _), e in base.errors.items() if nm == "cz"]
    assert {e for (nm, _), e in hom.errors.items() if nm == "cz"} == {float(np.median(cz))}
    with pytest.raises(ValueError):
        D.parse_device("fake_fez@hom=1,chi=0.5")                   # coherent errors need the exact engine
    with pytest.raises(ValueError):
        D.parse_device("fake_fez@hom=2")
    for chi in (0.25, 0.5, 1.0):
        dm = D.get_model("fake_fez@hom=1,chi=%r,eng=exact" % chi, phys)
        tab = X.gate_error_table(dm, "none")
        for key in (("sx", (0,)), ("x", (2,)), ("cz", (0, 1)), ("cz", (1, 0))):
            assert 1 - average_gate_fidelity(Kraus(tab[key])) == pytest.approx(dm.errors[key], rel=1e-9)
        # the coherent part is not a Pauli channel: twirling changes the channel but not its infidelity
        tw = X.gate_error_table(dm, "2q")[("cz", (0, 1))]
        assert np.abs(SuperOp(Kraus(tw)).data - SuperOp(Kraus(tab[("cz", (0, 1))])).data).max() > 1e-6
        assert 1 - average_gate_fidelity(Kraus(tw)) == pytest.approx(dm.errors[("cz", (0, 1))], rel=1e-9)
    # χ = 0 is Aer's model exactly
    t0 = X.gate_error_table(D.get_model("fake_fez@hom=1,eng=exact", phys), "none")
    hom_aer = D.get_model("fake_fez@hom=1", phys)
    from qiskit_aer.noise.device.models import basic_device_gate_errors
    ref = {(n, tuple(q)): e for n, q, e in basic_device_gate_errors(target=D._sub_target(hom_aer))}
    assert np.abs(SuperOp(Kraus(t0[("cz", (0, 1))])).data - SuperOp(ref[("cz", (0, 1))].to_quantumchannel()).data).max() < 1e-12


def test_e1_lambda_is_independent_of_chi_and_matches_aer_at_chi_zero():
    spec_of = lambda c: "fake_fez@hom=1,chi=%r,eng=exact" % c if c else "fake_fez@hom=1,eng=exact"
    # a local observable: λ (all locations) is exactly χ-independent; λ_LC only to a few % (light-cone marginals)
    fi = FAM.build("ising", 6, steps=2)
    ci = FAM.compile_family(fi, "fake_fez", initial_layout=(124, 123, 136, 143, 142, 141))
    reps = [LC.calculate_lambda_lc(ci, spec_of(c), fi.observable) for c in (0.0, 0.25, 0.5)]
    assert all(r.lam_total == pytest.approx(reps[0].lam_total, rel=1e-9) for r in reps)
    assert all(r.lam_lc == pytest.approx(reps[0].lam_lc, rel=0.05) for r in reps)
    fc = FAM.build("ghz", 6)
    comp = FAM.compile_family(fc, "fake_fez", initial_layout=(124, 123, 136, 143, 142, 141))
    lam = [LC.calculate_lambda_lc(comp, spec_of(c), fc.observable).lam_lc for c in (0.0, 0.25, 0.5)]
    assert lam[1] == pytest.approx(lam[0], rel=1e-9) and lam[2] == pytest.approx(lam[0], rel=1e-9)   # full support
    kw = dict(measure_map=comp.measure_map, layout=comp.info["phys"])
    a = simulate(comp.implemented, NoiseSpec(), "fake_fez@hom=1", **kw)
    b = simulate(comp.implemented, NoiseSpec(), "fake_fez@hom=1,eng=exact", **kw)
    assert np.abs(a.rho - b.rho).max() < 1e-7
    # the E1 engine schedules like the device (λ within a few % of E2's), unlike a per-layer NoiseSpec idle model
    e2 = LC.calculate_lambda_lc(comp, "fake_fez@eng=exact", fc.observable).lam_lc
    assert lam[0] == pytest.approx(e2, rel=0.1)


@pytest.mark.slow
def test_runner_survives_a_dead_worker(tmp_path):
    """A worker killed mid-run (fault injection) must not abort the study: the other cells finish in parallel, and the
    cell that kills its process every time is recorded as 'oom'."""
    import subprocess
    import sys
    man = tmp_path / "m.yaml"
    man.write_text('''manifest_version: 1
study: runner-crash-test
defaults: {device: fake_fez, layout: {4: [136, 143, 142, 141]}, train_reps: 20, train_reps_other: 20}
methods: {RAW: {estimator: raw}}
method_sets: {one: [RAW]}
circuits: {c: [{family: ghz, n: 4, theta: 1.0471975511965976}, {family: ising, n: 4, steps: 1}]}
stages:
  - {name: S, engine: E2, circuits: c, methods: one, scales: [1.0, 2.0], readout: "on"}
''', encoding="utf-8")
    from qemx.phase.manifest import expand, load
    cells = expand(load(str(man)))
    victim = cells[0].cell_id
    env = dict(os.environ, QEMX_PHASE_CRASH_CELL=victim, PYTHONPATH=os.path.dirname(os.path.dirname(__file__)))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "qemx.phase.run", "run", str(man), "--out", str(out), "--workers", "2"],
                       env=child_env(env), capture_output=True, text=True, timeout=900, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    from qemx.phase.store import read_cell
    st = {c.cell_id: read_cell(str(out), c.cell_id)["status"] for c in cells}
    assert st[victim] == "oom" and all(v == "ok" for k, v in st.items() if k != victim)


def test_exact_engine_refuses_snapshots(qft4):
    _, comp = qft4
    with pytest.raises(ValueError):
        simulate(comp.implemented, NoiseSpec(), "fake_fez@eng=exact", measure_map=comp.measure_map,
                 layout=comp.info["phys"], snapshots=True)


def test_fit_table_never_pools_stages():
    import pandas as pd
    from qemx.phase.fit import GROUP, fit_table
    assert {"stage", "device", "opt_level"} <= set(GROUP)
    rows = []
    for stage in ("A1", "E"):
        for i, lam in enumerate(np.linspace(0.05, 1.0, 8)):
            rows.append(dict(stage=stage, device="fake_fez", opt_level=1, engine="E2", readout="on",
                             coherent_fraction=0.0, family="qft", n=4, method="ZNE-R", status="ok", lam_lc=lam,
                             v=2.0 * np.exp(1.5 * lam), c=0, w=0.0, b=0.01 * lam))
    t = fit_table(pd.DataFrame(rows), boot=50)
    assert len(t) == 2 and np.allclose(t.var_beta, 1.5) and np.allclose(t.var_A, 2.0)
