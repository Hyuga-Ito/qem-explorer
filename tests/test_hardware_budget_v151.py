"""Execution budgets 1.5.1 — hardware plans (review F08, F09, F10).

F08  the hardware planner allocates integer shots that add up to the declared budget B (or records a smaller
     effective budget with its reason), also after merging repeated requests and after the measurement-twirl /
     gate-twirl expansion into the circuits actually built;
F09  hardware PEC keeps the configured shots per sampled circuit s (and derives the number of sampled circuits);
     an explicit hardware limit can change s, and then the plan records requested and effective protocol and its
     simulation uses the effective one — checked against an exact enumeration of PEC's instance distribution;
F10  one feasibility contract: the interactive cost model, the phase ledger and the hardware planner give the same
     minimum budget and the same verdict for every recipe of a grid (post-selection with low acceptance, learning
     methods with training shots, readout calibration, PEC with s > 1), under the default and the pipeline minima.
"""
import itertools
import math

import numpy as np
import pytest

from qemx import budget as BG
from qemx import hardware as H
from qemx import observables as OBS
from qemx.circuit import parse
from qemx.compile import build
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.methods.base import MethodResult
from qemx.noise import NoiseSpec


def test_review_9_4_three_equal_target_circuits_spend_exactly_B():
    r = MethodResult(0, 1, circuits=3, details={"alloc": [1 / 3] * 3})
    parts = [([("target", "a"), ("target", "b"), ("target", "c")], r)]
    for B in (100, 101, 3, 4, 5, 1000003):
        shots, info = H._allocate(r, parts, B)
        assert sum(shots.values()) == B, (B, shots)
        assert min(shots.values()) >= 1 and max(shots.values()) - min(shots.values()) <= 1
        assert info["effective_budget"] == B and info["target"] == B
    shots, info = H._allocate(r, parts, 2)                 # one shot per target circuit is the minimum
    assert shots is None and info["min_budget"] == 3


def _random_recipe(rng):
    """A synthetic recorded recipe: G measurement settings, each with its target circuits (shares, repeated
    requests), training circuits (some requested twice, some shared across roles) and the shared calibration
    circuits; returns (combined result, parts)."""
    G = int(rng.integers(1, 4))
    c = int(rng.choice([0, 0, 2, 3]))
    T = int(rng.choice([0, 0, 3, 5]))
    s_tr = int(rng.integers(1, 40))
    cal = [("calibration", "cal%d" % j) for j in range(c)]
    parts, k_all = [], 0
    for g in range(G):
        k = int(rng.integers(1, 6))
        k_all += k
        keys = ["t%d_%d" % (g, j) for j in range(k)]
        if k > 1 and rng.random() < 0.4:
            keys[-1] = keys[0]                             # the estimator asks for the same circuit twice
        log = [("target", x) for x in keys]
        tr = ["tr%d_%d" % (g, j) for j in range(T)]
        if T > 1 and rng.random() < 0.4:
            tr[-1] = tr[0]
        if T and rng.random() < 0.2:
            tr[0] = keys[0]                                # a training circuit identical to a target circuit
        log += [("training", x) for x in tr] + cal
        rg = MethodResult(0.0, float(rng.uniform(0.1, 5.0)), circuits=k, fixed_shots=T * s_tr, train_circuits=T,
                          details={"alloc": list(rng.dirichlet(np.ones(k)))})
        parts.append((log, rg))
    v = float(sum(np.sqrt(p[1].var1) for p in parts) ** 2)
    R = MethodResult(0.0, v, circuits=k_all, fixed_shots=G * T * s_tr, train_circuits=G * T, cal_circuits=c,
                     cal_grad=np.full(c, 0.3) if c else None, cal_theta=np.full(c, 0.05) if c else None)
    return R, parts


@pytest.mark.parametrize("cal_min", [1, 32])
def test_hardware_allocation_conserves_every_random_budget(cal_min):
    """Property test over 600 random role mixes and budgets: the shots add up to B exactly, every distinct circuit
    gets at least one shot (calibration: cal_min), the role totals add up, and the verdict is the shared contract's
    (unless the hardware needs more distinct circuits than the cost model counts)."""
    rng = np.random.default_rng(151)
    rule = BG.Feasibility(cal_min=cal_min) if cal_min != 1 else None     # cal_min = 1: the 1.5.0 call signature
    kw = dict(rule=rule) if rule is not None else {}
    checked = 0
    for _ in range(600):
        R, parts = _random_recipe(rng)
        m = BG.CostModel.of(R, **kw)
        mb = int(math.ceil(BG.min_budget(m)))
        for B in (mb - 1, mb, mb + 1, mb + int(rng.integers(2, 5000))):
            shots, info = H._allocate(R, parts, B, *([rule] if rule is not None else []))
            if B < mb:
                assert shots is None
                continue
            assert shots is not None, (B, mb, info)
            assert sum(shots.values()) == B == info.get("effective_budget", B)
            assert info["target"] + info["training"] + info["calibration"] == B
            assert min(shots.values()) >= 1
            cal_keys = {k for log, _ in parts for role, k in log if role == "calibration"}
            assert all(shots[k] >= cal_min for k in cal_keys if k in shots)
            assert info["training"] == int(R.fixed_shots)
            checked += 1
    assert checked > 1500


def _bell_on_device():
    from qemx import device as D, templates as T
    from qemx.compile import Compiled, with_io
    tpl = T.build("bell")
    b = D.fake_backend("fake_perth")
    spec, mm, phys, info = D.compile_for_target(with_io(tpl.spec), b, D.good_target_for(b), "FakePerth")
    return Compiled(with_io(tpl.spec), spec, mm, info), tpl, b


@pytest.mark.parametrize("B", [1001, 4097])
def test_planned_and_executed_totals_equal_the_budget(B):
    """The real planner on a fake backend, budgets that do not divide evenly; then the local expansion (TREX masks,
    gate-twirl instances) and the native one: every contender's executed shots equal its budget."""
    from qemx.recipes import preset
    comp, tpl, b = _bell_on_device()
    rc = [preset(k) for k in ("Unmitigated", "ZNE · Richardson", "Readout · M3", "Readout · TREX")]
    rc.append(Recipe.make("zne", "2q", "none", "trex", rule="linear"))
    rc.append(Recipe.make("pec", "none", "none", "tensored", shots_per_circuit=3))
    plan = H.make_plan(comp, tpl.observable, rc, B, tpl.symmetry, 0, b)
    assert set(plan.alloc) == set(range(len(rc))), plan.skipped
    for i, a in plan.alloc.items():
        tot = sum(r.shots for r in plan.requests.values() if r.tag == i)
        assert tot == B == a.get("effective_budget", B), (plan.recipes[i].title(), tot)
    for native in (False, True):
        jobs = H.build_jobs(plan, b, native_twirl=native, native_dd=native, seed=0)
        sent = {}
        for _, pubs, _ in jobs:
            for k, _, _, s in pubs:
                sent[k] = sent.get(k, 0) + s
        assert sent == {k: r.shots for k, r in plan.requests.items()}
        assert plan.check_budgets(sent) == {i: B for i in plan.alloc}
    res = plan.resources(native_twirl=False)["total"]
    assert res["shots"] == B * len(rc) and res["distinct_circuits"] == len(plan.requests)
    assert res["circuit_instances"] >= res["distinct_circuits"] and res["qpu_s"]["high"] >= res["qpu_s"]["low"]


def test_pauli_sum_settings_share_calibration_and_conserve_the_budget():
    from qemx import hamiltonian as HM
    comp = build(parse("ry(1.1) 0; cx 0 1; rz(0.7) 1; ry(0.4) 1", 2))
    ps = HM.parse_pauli_sum("0.5*Z0*Z1 + 0.3*X0 - 0.2*Y1", 2)
    rc = [Recipe.make("raw", readout="tensored"), Recipe.make("zne", readout="tensored", rule="linear")]
    for B in (997, 4096):
        plan = H.make_plan(comp, ps, rc, B, None, 0, None)
        for i in (0, 1):
            reqs = [r for r in plan.requests.values() if r.tag == i]
            assert sum(r.shots for r in reqs) == B
            cal = [r for r in reqs if r.role == "calibration"]
            assert len(cal) == 2 and all(r.multiplicity >= 2 for r in cal)    # one calibration, several settings


def test_check_budgets_catches_a_lost_shot():
    comp, tpl, b = _bell_on_device()
    plan = H.make_plan(comp, tpl.observable, [Recipe.make("zne")], 1000, tpl.symmetry, 0, b)
    k = next(iter(plan.requests))
    plan.requests[k].shots -= 1
    with pytest.raises(AssertionError):
        plan.check_budgets()
    with pytest.raises(AssertionError):
        H.build_jobs(plan, b, native_twirl=False, native_dd=False)


def test_pec_hardware_plan_keeps_the_configured_shots_per_circuit():
    """Review reproduction: B = 256, instances = 8, shots_per_circuit = 1 became 32 shots per circuit. The plan must
    run 256 sampled circuits of one shot each (identical draws merged) and say so."""
    comp, tpl, b = _bell_on_device()
    rc = Recipe.make("pec", construction="inverse", instances=8, shots_per_circuit=1)
    plan = H.make_plan(comp, tpl.observable, [rc], 256, tpl.symmetry, 0, b)
    a = plan.alloc[0]
    assert a["spc"] == 1 and a["group_params"][0]["instances"] == 256
    tgt = [r for r in plan.requests.values() if r.tag == 0 and r.role == "target"]
    assert sum(r.multiplicity for r in tgt) == 256                       # sampled circuits, before merging
    assert all(r.shots == r.multiplicity for r in tgt)                   # one shot per sampled circuit
    pr = plan.protocols[0]
    assert pr["requested"]["shots_per_circuit"] == pr["effective"]["shots_per_circuit"] == 1
    assert not pr["differences"] and plan.effective_recipe(0) is rc


def test_pec_hardware_limit_is_an_explicit_recorded_protocol_change():
    comp, tpl, b = _bell_on_device()
    rc = Recipe.make("pec", construction="inverse", instances=8, shots_per_circuit=1)
    plan = H.make_plan(comp, tpl.observable, [rc], 256, tpl.symmetry, 0, b, pec_max_circuits=8)
    a = plan.alloc[0]
    assert a["spc"] == 32 and a["group_params"][0]["instances"] == 8 and a["effective_budget"] == 256
    rows = {r["parameter"]: r for r in plan.protocol_table()}
    assert rows["shots_per_circuit"]["requested"] == "1" and rows["shots_per_circuit"]["effective"] == "32"
    assert not rows["shots_per_circuit"]["same"] and rows["twirl"]["same"]
    assert any("hardware limit" in n for n in plan.notes)
    assert plan.table()[0]["protocol"].startswith("differs")
    assert plan.effective_recipe(0).p["shots_per_circuit"] == 32
    # 300 shots: s = 38 sends 7 circuits (266 shots) ... the remainder must be recorded, never hidden
    plan2 = H.make_plan(comp, tpl.observable, [rc], 301, tpl.symmetry, 0, b, pec_max_circuits=8)
    a2 = plan2.alloc[0]
    tot = sum(r.shots for r in plan2.requests.values())
    assert tot == a2["effective_budget"] <= 301
    if tot < 301:
        assert a2["budget_note"] and plan2.protocols[0]["differences"]


def _pec_exact(p1: float):
    """Exact instance distribution of PEC (inverse, assumed depolarizing p1) on 'ry(1.1) 0; h 0' with a noiseless
    simulator: per-shot variance v1 = Γ² − μ² (o = ±1) and instance variance v_inst = E[(Γ·sign·⟨Z⟩_i)²] − μ²,
    enumerated over every Pauli draw (independent of qemx.methods.pec)."""
    from qemx.methods.pec import depol_quasi
    q = depol_quasi(p1, 1, "inverse")
    G = sum(abs(v) for v, _ in q) ** 2
    vals, probs = [], []
    for (va, la), (vb, lb) in itertools.product(q, q):
        circ = "ry(1.1) 0;" + ("" if la == "I" else " %s 0;" % la.lower()) + " h 0" + \
               ("" if lb == "I" else "; %s 0" % lb.lower())
        comp = build(parse(circ, 1))
        ctx = Context(comp, NoiseSpec(), OBS.z(0, 1), Recipe.make("raw"), Runner())
        z = ctx.ideal_value()
        vals.append(G * np.sign(va) * np.sign(vb) * z)
        probs.append(abs(va) * abs(vb) / G)
    vals, probs = np.array(vals), np.array(probs)
    mu = float(probs @ vals)
    return mu, G ** 2 - mu ** 2, float(probs @ vals ** 2) - mu ** 2


def test_pec_variance_of_the_executed_protocol_matches_exact_enumeration():
    """The protocol identity, checked against an independent exact enumeration: with s shots per sampled circuit
    Var = (v1 + (s − 1)·v_inst)/N_t. The simulator's PEC (configured s), the hardware plan (configured s, sampled
    circuits derived) and — under a hardware limit — the plan and the simulation of its effective recipe all report
    that per-shot variance (Monte-Carlo estimates: 20 % tolerance)."""
    mu, v1, vi = _pec_exact(0.1)
    assert vi > 0.05 * v1
    comp = build(parse("ry(1.1) 0; h 0", 1))
    obs = OBS.z(0, 1)

    def sim_var(s):
        rc = Recipe.make("pec", p1=0.1, p2=0.1, shots_per_circuit=s, instances=4096)
        ctx = Context(comp, NoiseSpec(), obs, rc, Runner(), seed=3)
        r = REGISTRY["pec"].run(ctx, **rc.p)
        assert abs(r.mean - mu) < 1e-9
        return r.var1
    for s in (1, 8):
        assert sim_var(s) == pytest.approx(v1 + (s - 1) * vi, rel=0.2), s
    rc8 = Recipe.make("pec", p1=0.1, p2=0.1, shots_per_circuit=8)                   # instances: default 64
    plan = H.make_plan(comp, obs, [rc8], 4096, None, 1, None)
    assert plan.alloc[0]["spc"] == 8 and plan.alloc[0]["group_params"][0]["instances"] == 512
    assert plan.alloc[0]["predicted"]["var1"] == pytest.approx(v1 + 7 * vi, rel=0.2)
    # a hardware limit of 512 sampled circuits turns the configured s = 1 into s = 8: the plan's prediction and the
    # simulation of its effective recipe describe that protocol, not the configured one
    rc1 = Recipe.make("pec", p1=0.1, p2=0.1, shots_per_circuit=1)
    capped = H.make_plan(comp, obs, [rc1], 4096, None, 1, None, pec_max_circuits=512)
    s_eff = capped.alloc[0]["spc"]
    assert s_eff == 8 and capped.effective_recipe(0).p["shots_per_circuit"] == s_eff
    want = v1 + (s_eff - 1) * vi
    assert capped.alloc[0]["predicted"]["var1"] == pytest.approx(want, rel=0.25)    # 512 sampled circuits
    assert sim_var(s_eff) == pytest.approx(want, rel=0.2)
    assert not capped.alloc[0]["predicted"]["var1"] == pytest.approx(v1, rel=0.25)  # not the configured s = 1


C2 = "ry(1.1) 0; cx 0 1; rz(0.7) 1; ry(0.4) 1; cx 0 1"


GHZ6 = "h 0; cx 0 1; cx 1 2; cx 2 3; cx 3 4; cx 4 5"


MILD = NoiseSpec(p1=0.01, p2=0.05, ro01=0.05, ro10=0.08)


HEAVY = NoiseSpec(p1=0.02, p2=0.12, ro01=0.15, ro10=0.2)


GRID = [  # (id, circuit, n, observable, noise, recipe)
    ("raw", C2, 2, OBS.z(0, 2), MILD, Recipe.make("raw")),
    ("raw-tensored", C2, 2, OBS.z(0, 2), MILD, Recipe.make("raw", readout="tensored")),
    ("raw-trex", C2, 2, OBS.z(0, 2), MILD, Recipe.make("raw", readout="trex")),
    ("zne", C2, 2, OBS.z(0, 2), MILD, Recipe.make("zne")),
    ("zne-tensored", C2, 2, OBS.z(0, 2), MILD, Recipe.make("zne", readout="tensored", rule="linear")),
    ("pec", C2, 2, OBS.z(0, 2), MILD, Recipe.make("pec", p1=0.01, p2=0.05)),
    ("pec-s4-tensored", C2, 2, OBS.z(0, 2), MILD, Recipe.make("pec", readout="tensored", p1=0.01, p2=0.05,
                                                              shots_per_circuit=4)),
    ("cdr", C2, 2, OBS.z(0, 2), MILD, Recipe.make("learning", features="cdr", model="ols", n_train=6,
                                                  train_shots=50)),
    ("nil", C2, 2, OBS.z(0, 2), MILD, Recipe.make("nil", n_train=4, n_neighbors=3, train_shots=40,
                                                  target_split="equal")),
    ("symmetry-low-acceptance", GHZ6, 6, OBS.z(0, 6), HEAVY,
     Recipe.make("symmetry", qubits=(0, 1), more="1 2; 2 3; 3 4; 4 5")),
    ("symmetry-tensored", GHZ6, 6, OBS.z(0, 6), HEAVY,
     Recipe.make("symmetry", readout="tensored", qubits=(0, 1), more="1 2; 2 3; 3 4; 4 5")),
]


@pytest.mark.parametrize("case", GRID, ids=[g[0] for g in GRID])
@pytest.mark.parametrize("cal_min", [1, 32])
def test_feasibility_contract_is_the_same_in_every_adapter(case, cal_min, monkeypatch):
    """Invariant: for the same recorded protocol, the interactive cost model (budget.CostModel / split / account),
    the phase ledger (cell.cost_of → ledger.min_budget / allocate) and the hardware planner (make_plan) report the
    same minimum budget and the same feasible / infeasible verdict at B = min − 1, min, min + 1 and min + 37.
    cal_min = 32 is the phase pipeline's calibration minimum (ledger.Policy s_min)."""
    from qemx.phase import ledger as L
    from qemx.phase.cell import MethodSpec, _method_row, cost_of
    name, circ, n, obs, noise, rc = case
    comp = build(parse(circ, n))
    sim = Runner("numpy")

    def predict(spec, mm, twirl, dd):                      # stands in for a backend's calibrated noise model
        return sim.run(spec, noise, mm, twirl=twirl, dd=dd).probs_meas
    monkeypatch.setattr(H, "_predictor", lambda compiled, nz: predict)
    # cal_min = 1 uses only the 1.5.0 interface (so the same test shows the defect on 1.5.0); 32 needs the rule
    rule = BG.Feasibility(cal_min=cal_min) if cal_min != 1 else None
    kw = dict(rule=rule) if rule is not None else {}
    pol = L.Policy("fixed", True, cal_min, 1)
    stub = H.Plan(comp, obs, [rc], 4096, None, 0, NoiseSpec(), None)
    R, _ = H._record(H.HardwareRunner(predict=predict), 0, rc, comp, obs, stub, {})
    if name.startswith("symmetry"):
        assert R.accept < 0.25                             # low acceptance: ⌈5/P⌉ dominates
    m = BG.CostModel.of(R, **kw)
    mb = BG.min_budget(m)
    row = _method_row(MethodSpec.make(name, rc.estimator, **rc.p), rc, R, 0.0)
    cost = cost_of(row)
    assert cost is not None
    assert int(mb) == mb == L.min_budget(cost, pol), (name, mb, L.min_budget(cost, pol))
    mb = int(mb)
    for B in (mb - 1, mb, mb + 1, mb + 37):
        ok = B >= mb
        assert (BG.split(m, B)[0] >= 1) == ok, (name, B)
        assert L.allocate(cost, B, pol).feasible == ok, (name, B)
        if hasattr(BG, "feasible"):
            assert BG.feasible(B, m.circuits, m.accept, m.per_circuit, m.fixed, m.c, rule or BG.DEFAULT_FEASIBILITY) == ok
        plan = H.make_plan(comp, obs, [rc], B, None, 0, None, **kw)
        assert (0 in plan.alloc) == ok, (name, B, plan.skipped)
        if ok:
            assert sum(r.shots for r in plan.requests.values()) == plan.alloc[0].get("effective_budget", B)
        else:
            assert "needs at least %s shots" % f"{mb:,}" in plan.skipped_idx[0], plan.skipped_idx[0]
    mb1 = int(BG.min_budget(BG.CostModel.of(R)))              # account() is the interactive table: default minima
    assert not BG.account(R, mb1 - 1)["feasible"] and BG.account(R, mb1)["feasible"]
    assert BG.account(R, mb1)["min_budget"] == mb1
