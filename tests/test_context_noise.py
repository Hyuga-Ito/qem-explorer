"""Context-dependent noise (qemx/context_noise.py) and the divergence-study analyses built on it: the plurality vote
of Chen et al. (against brute-force enumeration), the as-executed circuit features (ALAP schedule, spectators,
parallel adjacent gates), the divergence budget (NNLS recovers a planted mechanism), heavy-hex drawing
coordinates, the spectator-context probe family, layout/twirl variants, heterogeneity, the folding fingerprint
(calibrated under the null, A > 1 under amplified gate noise) and the t2 injection."""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

from qemx import context_noise as C
from qemx import divergence as D
from qemx.circuit import CircuitSpec, Op

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


# ------------------------------------------------------------------ plurality vote
def test_poisson_binomial_tail_against_enumeration():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 1, (5, 6))
    for t in range(0, 8):
        want = np.zeros(5)
        for bits in itertools.product((0, 1), repeat=6):
            b = np.array(bits)
            pr = np.prod(np.where(b, P, 1 - P), axis=1)
            want += pr * (b.sum() >= t)
        assert np.allclose(C.poisson_binomial_tail(P, t), want), t


def test_default_threshold():
    assert C.default_threshold(25) == 7                  # Chen et al.: t starts at 7 of 25 variants
    assert C.default_threshold(3) == 3 and C.default_threshold(2) == 2 and C.default_threshold(10) == 3


def test_plurality_vote_is_the_probability_of_t_votes():
    """p_b ∝ P(b is drawn by at least t of the variants when each draws once from its own measured frequencies — not
    renormalized over the survivors, as in Chen et al.'s reference implementation), over the bit strings present in
    at least t variants, by enumeration of every joint draw."""
    rng = np.random.default_rng(4)
    V, Dm = 5, 4
    F = rng.dirichlet(np.ones(Dm) * 0.7, size=V)
    F[1, 3] = F[2, 3] = F[3, 3] = 0.0                    # outcome 3 in only two variants: it cannot survive t = 3
    F /= F.sum(axis=1, keepdims=True)
    keep = (F > 0).sum(axis=0) >= 3
    for renorm in (False, True):
        dist, t = C.plurality_vote(F, 3, renormalize=renorm)
        assert t == 3
        G = F / F[:, keep].sum(axis=1, keepdims=True) if renorm else F
        want = np.zeros(Dm)
        for draw in itertools.product(range(Dm), repeat=V):
            pr = np.prod([G[v, d] if keep[d] or not renorm else 0.0 for v, d in enumerate(draw)])
            cnt = np.bincount(draw, minlength=Dm)
            want += pr * (cnt >= t) * keep
        assert np.allclose(dist, want / want.sum()) and dist[3] == 0, renorm


def test_plurality_vote_raw_and_renormalized_differ():
    """The two readings of Eq. 5 differ when the variants put different mass outside the survivors."""
    F = np.array([[0.5, 0.2, 0.3, 0.0], [0.5, 0.3, 0.0, 0.2], [0.2, 0.2, 0.0, 0.6], [0.6, 0.1, 0.3, 0.0]])
    raw, _ = C.plurality_vote(F, 3)
    ren, _ = C.plurality_vote(F, 3, renormalize=True)
    assert raw[2] == raw[3] == ren[2] == ren[3] == 0 and not np.allclose(raw, ren)
    assert raw.sum() == pytest.approx(1) and ren.sum() == pytest.approx(1)


def test_plurality_vote_single_survivor_and_threshold_cap():
    F = np.array([[0.9, 0.1, 0.0], [0.8, 0.0, 0.2], [0.7, 0.0, 0.3]])
    dist, t = C.plurality_vote(F, 7)                       # t = min(7, largest occurrence 3)
    assert t == 3 and np.allclose(dist, [1, 0, 0])


def test_plurality_vote_rejects_layout_specific_errors():
    """A wrong outcome that appears only on one layout is voted out; the consensus peak is amplified."""
    good = np.array([0.7, 0.1, 0.1, 0.1, 0.0])
    F = np.array([good] * 4 + [[0.4, 0.05, 0.05, 0.05, 0.45]])
    dist, t = C.plurality_vote(F, 3)
    assert dist[4] == 0 and dist[0] > 0.9 and t == 3
    agg = F.mean(axis=0)
    assert dist[0] > agg[0]


def test_plurality_vote_falls_back_to_the_average_without_consensus():
    F = np.eye(4)                                        # every variant a different single outcome
    dist, t = C.plurality_vote(F, 3)
    assert t == 2 and np.allclose(dist, 0.25)
    rows = C.plurality_vote_rows([np.tile(f, (3, 1)) for f in F], 3)
    assert rows.shape == (3, 4) and np.allclose(rows, 0.25)


# ------------------------------------------------------------------ circuit features
LINE = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}           # a line 0-1-2-3 of device qubits
DUR = {("cz", (0, 1)): 100e-9, ("cz", (1, 2)): 100e-9, ("cz", (2, 3)): 100e-9, ("sx", (2,)): 30e-9,
       ("ecr", (0, 1)): 100e-9, ("ecr", (1, 2)): 100e-9}


def test_alap_schedule_and_features():
    spec = CircuitSpec(3, (Op("cz", (0, 1)), Op("cz", (1, 2))), "two cz")
    iv, lead, busy, T = C.alap_schedule(spec, DUR)
    assert T == pytest.approx(200e-9)
    assert iv[0] == pytest.approx((0, 100e-9)) and iv[1] == pytest.approx((100e-9, 200e-9))
    assert list(lead) == pytest.approx([0, 0, 100e-9])                # ALAP: qubit 2 waits in |0⟩, not idle
    f = C.circuit_features(spec, [0, 1, 2], LINE, DUR)
    assert f["n2q"] == 2 and f["n1q"] == 0 and f["measured"] == 3
    assert f["active_spectators"] == 2                   # cz(0,1) sees 2, cz(1,2) sees 0 — both in the circuit
    assert f["passive_spectators"] == 1                  # cz(1,2) sees device qubit 3, outside the circuit
    assert f["control_spectators"] == 0                  # CZ is symmetric: no driven control
    assert f["idle_us"] == pytest.approx(0.1)            # qubit 0 waits 100 ns for the measurement
    assert f["duration_us"] == pytest.approx(0.2)
    g = C.circuit_features(CircuitSpec(3, (Op("ecr", (0, 1)), Op("ecr", (1, 2))), "ecr"), [0, 1, 2], LINE, DUR,
                           two_qubit="ecr")
    assert g["control_spectators"] == 1                  # ecr(1,2): control 1 has the other neighbour 0


def test_parallel_adjacent_gates():
    spec = CircuitSpec(4, (Op("cz", (0, 1)), Op("cz", (2, 3))), "parallel")
    f = C.circuit_features(spec, [0, 1, 2, 3], LINE, DUR)
    assert f["parallel_adjacent"] == 1                   # simultaneous, on couplers joined by 1-2
    spec = CircuitSpec(4, (Op("cz", (0, 1)), Op("barrier", (0, 1, 2, 3), (), "barrier", "keep"), Op("cz", (2, 3))),
                       "sequential")
    assert C.circuit_features(spec, [0, 1, 2, 3], LINE, DUR)["parallel_adjacent"] == 0


def test_device_neighbors_and_heavy_hex_coordinates():
    from qemx.device import fake_backend
    for name in ("fake_fez", "fake_torino"):
        be = fake_backend(name)
        nb = C.device_neighbors(be)
        assert max(len(v) for v in nb.values()) == 3     # heavy hex: degree ≤ 3
        edges = sorted({(min(a, b), max(a, b)) for a, v in nb.items() for b in v})
        pos = C.heavy_hex_coordinates(edges, be.num_qubits)
        assert len(pos) == be.num_qubits
        assert len({(round(x, 6), round(y, 6)) for x, y in pos.values()}) == be.num_qubits
        lengths = [math.dist(pos[a], pos[b]) for a, b in edges]
        assert max(lengths) <= 1 + 1e-9, name             # every coupler drawn as a unit segment
    ring = [(i, (i + 1) % 6) for i in range(6)]          # not heavy hex: a force-directed fallback
    assert len(C.heavy_hex_coordinates(ring, 6)) == 6


# ------------------------------------------------------------------ attribution
def test_nnls_attribution_recovers_a_planted_mechanism():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.integers(5, 80, 60), rng.integers(0, 40, 60), rng.uniform(0, 5, 60), np.zeros(60)])
    beta = np.array([0.004, 0.0, 0.01, 0.0])
    y = X @ beta + rng.normal(0, 0.003, 60)
    r = C.nnls_attribution(X, y, ["a", "b", "c", "zero"], boot=300, seed=2)
    assert r["coef"][0] == pytest.approx(0.004, rel=0.1) and r["coef"][2] == pytest.approx(0.01, rel=0.15)
    assert r["lo"][1] == 0 and r["hi"][1] < 0.001       # the absent mechanism: an interval at 0
    assert r["lo"][0] > 0 and r["lo"][2] > 0
    assert math.isnan(r["coef"][3])                      # a column of zeros prices nothing
    assert r["r2"] > 0.9 and r["circuits"] == 60
    assert np.nansum(r["share"]) == pytest.approx(1.0, abs=1e-9) and r["share"][1] < 0.01
    few = C.nnls_attribution(X[:2], y[:2], ["a", "b", "c", "zero"], boot=0)
    assert np.isnan(few["coef"]).all()


def test_qubit_attribution_finds_the_bad_qubit():
    rng = np.random.default_rng(3)
    loads, y = [], []
    for _ in range(80):
        L = {q: int(rng.integers(0, 6)) for q in rng.choice(10, 4, replace=False)}
        loads.append(L)
        y.append(0.02 * L.get(7, 0) + 0.001 * sum(L.values()) + rng.normal(0, 0.002))
    Q = C.qubit_attribution(loads, y, boot=200)
    best = max(Q, key=lambda q: Q[q][0])
    assert best == 7 and Q[7][1] > 0.01


# ------------------------------------------------------------------ the context probe family
@pytest.mark.parametrize("mode", ["zero", "plus", "driven"])
def test_context_probe_ideal_value_does_not_depend_on_the_spectators(mode):
    from qemx.phase.families import build
    ref = D.ideal_distribution(build("context", 6, layers=4, spectators="zero").spec, {i: i for i in range(6)})
    fc = build("context", 6, layers=4, spectators=mode)
    p = D.ideal_distribution(fc.spec, {i: i for i in range(6)})
    assert np.allclose(p, ref, atol=1e-12)               # the spectators return to |0⟩: the same ideal output
    bars = [o for o in fc.spec.ops if o.kind == "barrier"]
    assert len(bars) == 5 and all(o.tag == "keep" for o in bars)
    assert dict(fc.params)["spectators"] == mode


def test_context_probe_validation():
    from qemx.phase.families import build
    for kw in (dict(n=5), dict(n=4, spectators="driven"), dict(n=6, spectators="loud"), dict(n=6, layers=0)):
        n = kw.pop("n")
        with pytest.raises(ValueError):
            build("context", n, **kw)


def test_keep_barriers_survive_compilation_and_folding(backend):
    from qemx.transforms import fold_global, fold_local
    suite = D.suite_from_manifest(dict(
        study="t", kind="divergence", device="ibm_fez", fake="fake_fez", budget=3000, cal_shots=100, layout="chain",
        methods={"RAW": {"kind": "raw"}, "ZNE-G": {"kind": "zne", "folding": "global", "scales": [1, 3]}},
        circuits=[{"family": "context", "n": 6, "layers": 3, "spectators": "driven"}]))
    plan = D.make_plan(suite, backend)
    assert not plan.errors
    cs = plan.instances[0].spec                          # compiled to fake_fez's CZ and √X, on 6 physical qubits
    keep = lambda s: sum(1 for o in s.ops if o.kind == "barrier" and o.tag == "keep")   # noqa: E731
    assert keep(cs) == 4
    assert cs.count("2q") == 3 * 3                       # the driven spectators' CZ pairs are not cancelled
    assert keep(fold_global(cs, 3.0)) == 3 * 4           # U (U† U): every copy keeps its alignment barriers
    assert keep(fold_local(cs, 5 / 3, 0, "stratified")) == 4
    f = plan.instances[0].features
    assert f["active_spectators"] > 0 and f["parallel_adjacent"] > 0


# ------------------------------------------------------------------ variants and heterogeneity
def _suite(mode, count=3, **extra):
    m = dict(study="t", kind="divergence", device="ibm_fez", fake="fake_fez", budget=3000, cal_shots=100,
             replicates=120, bootstrap=60, exact_max_qubits=3, aer_dm_max_qubits=6, layout="chain",
             variants={"count": count, "mode": mode},
             methods={"RAW": {"kind": "raw"}, "ZNE-G": {"kind": "zne", "folding": "global", "scales": [1, 3, 5]},
                      "ZNE-L": {"kind": "zne", "folding": "local", "scales": [1, 3, 5]}},
             circuits=[{"family": "ghz", "n": 3, "theta": math.pi / 2, "basis": "Z"},
                       {"family": "qft", "n": 3, "m": None, "s": 1}])
    m.update(extra)
    return D.suite_from_manifest(m)


@pytest.fixture(scope="module")
def backend():
    from qemx.device import fake_backend
    return fake_backend("fake_fez")


@pytest.fixture(scope="module")
def vplan(backend):
    p = D.make_plan(_suite("layouts"), backend)
    assert not p.errors
    return p


def test_layout_variants_use_disjoint_chains(vplan):
    groups = {}
    for i in vplan.instances:
        groups.setdefault(i.group, []).append(i)
    assert len(groups) == 2 and all(len(g) == 3 for g in groups.values())
    for g in groups.values():
        sets = [set(i.phys) for i in g]
        assert all(not (a & b) for a, b in itertools.combinations(sets, 2))
        assert sorted(i.variant for i in g) == [0, 1, 2]
        assert len({i.key for i in g}) == 3 and all(i.key.endswith("#v%d" % i.variant) for i in g)
    for i in vplan.instances:                            # the budget is shared: 1000 shots per variant
        assert sum(r.shots for r in vplan.runs_of(i.key, "RAW")) == 1000


def test_twirl_variants_share_the_ideal_output(backend):
    p = D.make_plan(_suite("twirl"), backend)
    assert not p.errors
    for g in {i.group for i in p.instances}:
        insts = [i for i in p.instances if i.group == g]
        assert len({tuple(i.phys) for i in insts}) == 1                   # one layout …
        assert len({D.spec_digest(i.spec) for i in insts}) == 3           # … three physical circuits
        for i in insts[1:]:
            assert np.allclose(i.ideal, insts[0].ideal, atol=1e-10)


def test_disjoint_chains_reverse_then_overlap_but_never_repeat(backend):
    """More variants than the chip has disjoint chains: disjoint chains first, then the same chains reversed (a
    different qubit-to-position assignment), then overlapping chains — never one assignment twice."""
    ch = D.disjoint_chains(backend, 6, 40)
    assert len(ch) == 40 and len({tuple(c) for c in ch}) == 40
    first = [set(c) for c in ch[:3]]
    assert all(not (a & b) for a, b in itertools.combinations(first, 2))
    tuples = {tuple(c) for c in ch}
    assert any(tuple(reversed(c)) in tuples for c in ch)                 # orientations are used
    edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    for c in ch:                                                          # every chain is a path on the chip
        assert all(tuple(sorted((a, b))) in edges for a, b in zip(c, c[1:]))


def test_layouts_plus_twirl_variants(backend):
    """Variants that change both the qubit assignment and the gate frame (Chen et al.'s design) on heavy hex: a
    different chain *and* a different twirl per variant, the same ideal output."""
    p = D.make_plan(_suite("layouts+twirl"), backend)
    assert not p.errors
    for g in {i.group for i in p.instances}:
        insts = [i for i in p.instances if i.group == g]
        assert len({tuple(i.phys) for i in insts}) == 3
        assert len({D.spec_digest(i.spec) for i in insts}) == 3
        for i in insts[1:]:
            assert np.allclose(i.ideal, insts[0].ideal, atol=1e-10)


def test_layout_variants_need_a_chain_layout():
    with pytest.raises(ValueError):
        _suite("layouts", layout={3: [0, 1, 2]})


def test_heterogeneity():
    h = D.heterogeneity([0.01, 0.01, 0.01], [0.005, 0.005, 0.005])
    assert h["het_Q"] == pytest.approx(0) and h["het_p"] == pytest.approx(1) and h["het_I2"] == 0
    h = D.heterogeneity([0.0, 0.05, 0.10], [0.005, 0.005, 0.005])
    assert h["het_p"] < 1e-6 and h["het_I2"] > 0.9
    assert math.isnan(D.heterogeneity([0.1], [0.01])["het_Q"])


def test_zscore_and_excess():
    assert D.zscore(0.03, 0.01, 0.01) == pytest.approx(3.0)
    assert D.zscore(0.03, 0.0, math.sqrt(2) * 0.01) == pytest.approx(3.0)      # pooled: finite with sd_pred = 0
    assert math.isnan(D.zscore(0.03, 0.0, 0.0)) and math.isnan(D.zscore(math.nan, 0.1, 0.1))
    e = D.excess(0.9, 0.8, 0.8, 0.6, 20)
    assert e["ex_ln_fc"] == pytest.approx(math.log(0.9 / 0.8)) and e["ex_nf_per_2q"] == pytest.approx(
        math.log(0.8 / 0.6) / 20)
    assert math.isnan(D.excess(0.9, 0.01, 0.8, 0.6, 20)["ex_ln_fc"])          # below the floor: no logarithm


# ------------------------------------------------------------------ emulated ground truth
@pytest.fixture(scope="module")
def emu(vplan, backend):
    models = D.snapshot(vplan, backend)
    pred, errs = D.predict(vplan, models, ("depolarizing",))
    assert not errs
    return models, pred


def test_group_rows_aggregate_and_vote(vplan, emu):
    models, pred = emu
    counts, errs = D.emulate(vplan, models, truth="depolarizing", seed=5)
    assert not errs
    rows = pd.DataFrame(D.compare(vplan, counts, pred, seed=1))
    g = rows[rows["variant"] == -1]
    assert set(g["method"]) == {"RAW·agg", "RAW·PV"} and len(g) == 4
    assert (g["status"] == "ok").all() and (g["budget"] == 3000).all()
    assert g["z"].abs().max() < 4.5                       # the null: no divergence in the aggregates either
    assert ((g["het_p"] >= 0) & (g["het_p"] <= 1)).all()
    assert (g[g["method"] == "RAW·PV"]["pv_threshold"] == 3).all()


@pytest.fixture(scope="module")
def deep(backend):
    """Deeper circuits at a large budget: κ is measured to a few per cent."""
    s = D.suite_from_manifest(dict(
        study="t", kind="divergence", device="ibm_fez", fake="fake_fez", budget=60000, cal_shots=500,
        replicates=150, bootstrap=100, exact_max_qubits=3, aer_dm_max_qubits=6,
        methods={"ZNE-G": {"kind": "zne", "folding": "global", "scales": [1, 3, 5]},
                 "ZNE-L": {"kind": "zne", "folding": "local", "scales": [1, 3, 5]}},
        circuits=[{"family": "qft", "n": [3, 4], "m": None, "s": 1},
                  {"family": "brickwork", "n": 4, "layers": 4, "seed": 0}]))
    plan = D.make_plan(s, backend)
    models = D.snapshot(plan, backend)
    pred, errs = D.predict(plan, models, ("depolarizing",))
    assert not errs and not plan.errors
    return plan, models, pred


def test_folding_fingerprint_null_and_amplified(deep):
    """Under the null the device answers folding as its model does (pooled A ≈ 1, no κ beyond 3σ, reading "as
    modeled"); with every gate error doubled κ_obs ≈ 2 κ_pred under both foldings ("gate-attached")."""
    import dirA_divergence as DV
    plan, models, pred = deep
    out = {}
    for tag, inj in (("null", {}), ("scale", {"scale": 2.0})):
        counts, errs = D.emulate(plan, models, truth="depolarizing", inject=inj, seed=7)
        assert not errs
        FP = pd.DataFrame(D.fold_response(plan, counts, pred, seed=3))
        assert len(FP) == len(plan.instances) * 2
        assert np.isfinite(FP["kappa_pred_exact"]).all()
        if tag == "null":
            assert (FP["z_kappa"].abs() < 4).all()
        out[tag] = DV.fingerprint_summary(DV.fingerprint_pairs(FP)).set_index("family").loc["all"]
    a, b = out["null"], out["scale"]
    assert abs(a["A_global"] - 1) < 3 * a["A_global_se"] + 0.05 and abs(a["A_local"] - 1) < 3 * a["A_local_se"] + 0.05
    assert a["reading"] == "as modeled"
    assert b["A_global"] == pytest.approx(2.0, abs=0.35) and b["A_local"] == pytest.approx(2.0, abs=0.35)
    assert b["reading"] == "gate-attached"


def test_fingerprint_reading():
    import dirA_divergence as DV
    assert DV.fingerprint_reading(1.02, 0.97, 0.05, 0.05) == "as modeled"
    assert DV.fingerprint_reading(2.0, 1.9, 0.1, 0.1) == "gate-attached"
    assert DV.fingerprint_reading(2.0, 1.1, 0.1, 0.1) == "time-attached"
    assert DV.fingerprint_reading(1.1, 2.0, 0.1, 0.1) == "repetition-coherent"
    assert DV.fingerprint_reading(0.5, 0.6, 0.05, 0.05) == "over-modeled"
    assert DV.fingerprint_reading(1.25, 1.0, 0.1, 0.1) == "inconclusive"     # one 2.5σ excess, no clear pattern
    assert DV.fingerprint_reading(math.nan, 1.0) == ""
    A, se, chi = D.pooled_anomaly([0.2, 0.0], [0.1, 0.1], [0.02, 0.02])
    assert (A, se) == pytest.approx((1.0, math.sqrt(8e-4) / 0.2)) and chi == pytest.approx(50.0)
    assert D.pooled_anomaly([0.2, 0.0], [0.1, 0.1])[0] == pytest.approx(1.0)            # no SEs: Σκ_obs / Σκ_pred
    A, se, _ = D.pooled_anomaly([0.30, 0.02], [0.10, 0.01], [0.01, 0.05])               # the precise circuit rules
    assert A == pytest.approx(3.0, abs=0.01)


def test_t2_injection_lowers_t2_only(vplan, emu):
    models, _ = emu
    base = D._truth_model(next(iter(models.values())), "depolarizing", {})
    dm = D._truth_model(next(iter(models.values())), "depolarizing", {"t2": 30})
    assert all(q.t2 <= 30e-6 + 1e-15 for q in dm.qubits)
    assert all(abs(a.t1 - b.t1) < 1e-12 for a, b in zip(base.qubits, dm.qubits))
    assert dm.errors == base.errors
    assert not D.relaxation_free(dm) and D.relaxation_free(base)
    with pytest.raises(ValueError):
        D.parse_injection(["t2=0"])
    assert D.parse_injection(["t2=40", "crosstalk=0.05"]) == {"t2": 40.0, "crosstalk": 0.05}


# ------------------------------------------------------------------ the χ² ladder and the budget on planted data
def _planted(kind: str, n: int = 90, seed: int = 0) -> pd.DataFrame:
    """A divergence table whose excess log-infidelity is planted: none, a rescaled calibration (every error 1.5×)
    or idle dephasing (0.01 per µs), with shot noise of known SE."""
    rng = np.random.default_rng(seed)
    priced = rng.uniform(0.02, 0.6, n)
    idle = rng.uniform(0, 3, n)
    nf_pred = np.exp(-priced)
    se = 0.004 + 0.01 * rng.uniform(size=n)
    planted = {"null": 0 * priced, "scale": 0.5 * priced, "idle": 0.01 * idle}[kind]
    nf_obs = np.exp(np.log(nf_pred) - planted + rng.normal(0, se / nf_pred))
    return pd.DataFrame(dict(predictor="depolarizing", method="RO", status="ok", variant=0,
                             family=rng.choice(["qft", "ghz", "ising"], n), nf_pred=nf_pred, nf_obs=nf_obs,
                             nf_obs_se=se, nf_pred_sd=se, ex_ln_nf=np.log(nf_pred) - np.log(nf_obs),
                             f_active_spectators=rng.integers(0, 30, n).astype(float),
                             f_passive_spectators=rng.integers(0, 30, n).astype(float),
                             f_control_spectators=0.0, f_parallel_adjacent=rng.integers(0, 6, n).astype(float),
                             f_idle_us=idle))


def test_model_ladder_names_the_planted_explanation():
    import dirA_divergence as DV
    L, v = DV.model_ladder(_planted("null"))
    assert v.startswith("as modeled") and 0.6 < L["chi2_per_dof"].iloc[0] < 1.5
    L, v = DV.model_ladder(_planted("scale"))
    assert v.startswith("rescaled calibration") and L["priced"].iloc[1] == pytest.approx(0.5, abs=0.05)
    assert DV._short_verdict(v).startswith("rescaled calibration ×1.5")
    L, v = DV.model_ladder(_planted("idle"))
    assert v.startswith("context") and "non-linear" not in DV._short_verdict(v)
    assert L["chi2_per_dof"].iloc[3] < 1.5 < L["chi2_per_dof"].iloc[1]
    coef, budget, meta = DV.attribution(_planted("idle"), boot=100)
    c = coef.set_index("mechanism")
    assert c.loc["idle_us", "coef"] == pytest.approx(0.01, rel=0.15) and c.loc["idle_us", "share"] > 0.9
    assert meta["weighted"] and set(budget["mechanism"]) >= {"idle_us", "unexplained"}
    assert DV.ladder_verdict(L) == v                                          # the verdict from a saved table


def test_probe_reading_and_diagnostic_matrix(tmp_path):
    import matplotlib.pyplot as plt

    import dirA_figures as F
    assert F.probe_reading([]) == "no spectator effect" and F.probe_reading(["plus"]) == "idle dephasing"
    assert F.probe_reading(["driven"]) == "gate error" and F.probe_reading(["plus", "driven"]) == "spectator crosstalk"
    R = pd.DataFrame([dict(data="null", ladder="as modeled", budget="—", fingerprint="as modeled (A_g 1.0±0.1, …)",
                           fingerprint_reading="as modeled", probe_plus=0.001, probe_plus_z=0.5, probe_driven=0.0,
                           probe_driven_z=0.1, eps_ratio=1.0),
                      dict(data="T2", ladder="context + non-linear", budget="idle time 100%",
                           fingerprint="time-attached (…)", fingerprint_reading="time-attached", probe_plus=0.03,
                           probe_plus_z=9.0, probe_driven=0.0, probe_driven_z=0.2, eps_ratio=1.3)])
    fig = F.diagnostic_matrix(R, title="test")
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "idle dephasing" in texts and "time-attached" in texts and "no spectator effect" in texts
    plt.close("all")


def test_ladder_verdict_material_context_rule():
    """With many precise circuits a first-order rescaling is rejected somewhere; context that removes < 5 % of the
    χ² all mechanisms remove is immaterial (the v2 ×1.6 emulation), context that removes more is not (its T2 one)."""
    import dirA_divergence as DV
    from scipy.stats import chi2 as CHI2

    def table(c2, dof, priced, p_add):
        return pd.DataFrame(dict(model=["none", "calibration (priced)", "context", "calibration + context"],
                                 chi2=c2, dof=dof, chi2_per_dof=[a / b for a, b in zip(c2, dof)],
                                 p=[CHI2.sf(a, b) for a, b in zip(c2, dof)], circuits=156,
                                 priced=[np.nan, priced, np.nan, priced], p_context_added=[np.nan] * 3 + [p_add]))
    L = table([1567.8, 288.8, 537.9, 280.5], [156, 155, 153, 154], 0.575, 0.004)
    assert DV.calibration_share(L) == pytest.approx(0.9935, abs=1e-3)
    v = DV.ladder_verdict(L)
    assert v.startswith("rescaled calibration") and "adds little" in v and "non-linear" not in DV._short_verdict(v)
    assert DV._short_verdict(v) == "rescaled calibration ×1.57"
    L = table([1763.7, 1508.3, 949.6, 949.6], [156, 155, 155, 155], 0.236, 1e-120)
    assert DV._short_verdict(DV.ladder_verdict(L)) == "context + non-linear"
