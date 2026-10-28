"""1.6.0 protocol-selection study: task identity, complete-protocol ledgers, trial-level emulation (validated against
direct execution), failure handling and mismatch runners."""
import math

import numpy as np
import pytest

from qemx.selection import emulate as EM
from qemx.selection import mismatch as MM
from qemx.selection import protocols as PR
from qemx.selection import task as TK


@pytest.fixture(scope="module")
def ghz3():
    built = TK.build(TK.make_spec("ghz", 3, theta=1.0472))
    return built, EM.prepare(built, seed=1)


# ------------------------------------------------------------------ tasks
def test_task_identity_is_deterministic_and_variants_share_their_ancestry():
    a = TK.make_spec("ising", 4, steps=2)
    b = TK.make_spec("ising", 4, steps=2)
    c = TK.make_spec("ising", 4, steps=2, observable="z_all", layout="alt", device="fake_torino")
    assert a.task_id == b.task_id and a.task_id != c.task_id
    assert a.base_id == c.base_id                                    # same logical circuit: same split group
    assert TK.make_spec("ising", 4, steps=3).base_id != a.base_id
    assert TK.TaskSpec.from_json(c.to_json()) == c


def test_fixed_input_families_refuse_other_inputs_and_alt_layouts_use_other_qubits():
    with pytest.raises(ValueError):
        TK.build(TK.make_spec("qft", 3, inputs="plus"))
    best = TK.build(TK.make_spec("hea", 3, layers=1))
    alt = TK.build(TK.make_spec("hea", 3, layers=1, layout="alt"))
    assert not set(best.layout) & set(alt.layout)
    assert best.ideal == pytest.approx(alt.ideal, abs=1e-12)        # same logical task, other physical qubits


def test_ideal_value_agrees_with_the_method_context(ghz3):
    built, prep = ghz3
    assert built.ideal == pytest.approx(prep.ctx_truth["RAW"].ideal_value(), abs=1e-12)
    assert built.ideal == pytest.approx(math.sin(1.0472), abs=1e-9)


# ------------------------------------------------------------------ ledgers
@pytest.mark.parametrize("B", PR.BUDGETS)
def test_every_feasible_allocation_spends_exactly_the_budget_in_integers(ghz3, B):
    _, prep = ghz3
    for key in PR.PORTFOLIO_16:                       # the expanded portfolio: tests/test_registry.py
        pr = PR.PORTFOLIO[key]
        a = PR.allocate(pr, prep.supplied[key], B)
        if not a.feasible:
            assert a.reason
            continue
        assert sum(a.ledger.values()) == B and all(isinstance(v, int) for v in a.ledger.values())
        assert a.ledger["execution"] == a.N and a.ledger["calibration"] == a.c * a.s
        if pr.estimator == "zne":
            assert sum(a.split) == a.N and min(a.split) >= 1
        if pr.learning:
            assert a.s_tr >= PR.TRAIN_MIN and a.ledger["training"] == a.s_tr * a.train_circuits


def test_charged_diagnostics_come_off_the_execution_budget(ghz3):
    _, prep = ghz3
    a0 = PR.allocate(PR.PORTFOLIO["RO"], prep.supplied["RO"], 10000)
    a1 = PR.allocate(PR.PORTFOLIO["RO"], prep.supplied["RO"], 10000, diagnostics=500)
    assert a1.ledger["diagnostics"] == 500 and sum(a1.ledger.values()) == 10000
    assert a1.N + a1.c * a1.s == a0.N + a0.c * a0.s - 500


def test_learning_protocols_are_infeasible_below_their_training_minimum(ghz3):
    _, prep = ghz3
    a = PR.allocate(PR.PORTFOLIO["NIL"], prep.supplied["NIL"], 1000)
    assert not a.feasible and "training" in a.reason


# ------------------------------------------------------------------ emulation
def test_trials_are_reproducible_and_the_two_halves_are_independent(ghz3):
    from qemx.selection.labels import rng_for
    _, prep = ghz3
    x = EM.trials(prep, "ZNE-L", 30000, 50, rng_for(1, "t", 30000, "ZNE-L", 0)).estimates
    y = EM.trials(prep, "ZNE-L", 30000, 50, rng_for(1, "t", 30000, "ZNE-L", 0)).estimates
    z = EM.trials(prep, "ZNE-L", 30000, 50, rng_for(1, "t", 30000, "ZNE-L", 1)).estimates
    assert np.array_equal(x, y) and not np.array_equal(x, z)


def test_pec_trials_are_unbiased_for_the_supplied_model_mean(ghz3):
    """The sign-resolved law reproduces the exact PEC mean (with the exact readout calibration at large s)."""
    _, prep = ghz3
    tr = EM.trials(prep, "PEC", 30_000_000, 200, np.random.default_rng(3))
    se = tr.estimates.std(ddof=1) / math.sqrt(len(tr.estimates))
    assert abs(tr.estimates.mean() - prep.truth["PEC"].mean) < 4 * se + 2e-4


def test_finite_training_data_raise_the_learning_variance(ghz3):
    _, prep = ghz3
    res = prep.truth["CDR"]
    rng = np.random.default_rng(5)
    few = res.sampler(200000, 300, rng, deployable=True, s_tr=64)
    many = res.sampler(200000, 300, rng, deployable=True, s_tr=8192)
    assert few.var() > 2 * many.var()


@pytest.mark.parametrize("key", ["RAW", "RO", "ZNE-L", "ZNE-R", "PEC"])
def test_emulated_trials_match_direct_shot_by_shot_execution(ghz3, key):
    """The cached-distribution emulator against an independent re-implementation that draws every shot (PEC: an
    explicit instance circuit per shot): means within 4 SE, variances not rejected at p = 1e-3."""
    _, prep = ghz3
    rng = np.random.default_rng([11, len(key)])
    em = EM.trials(prep, key, 1500, 300, rng).estimates
    di = EM.direct_trials(prep, key, 1500, 300, rng)
    c = EM.compare_laws(em, di)
    assert abs(c["z_mean"]) < 4 and c["p_var"] > 1e-3, c


def test_nonresponse_is_scored_with_the_declared_loss_not_dropped():
    tr = EM.Trials("X", 1000, {}, np.array([0.5, np.nan]), {f: np.zeros(2, bool) for f in EM.FLAGS})
    assert list(tr.errors(0.5)) == [0.0, -0.5]


# ------------------------------------------------------------------ mismatch
def test_truth_modifiers_parse_and_unknown_ones_are_refused():
    assert MM.parse_truth("chi=0.25,xt=0.05") == ("chi=0.25", {"xt": 0.05})
    with pytest.raises(ValueError):
        MM.parse_truth("zz=1")


def test_crosstalk_changes_the_noisy_law_but_not_ideal_values():
    spec = TK.make_spec("ising", 4, steps=2, truth="xt=0.05")
    built = TK.build(spec)
    prep = EM.prepare(built, ["RAW"], seed=1)
    ctx = prep.ctx_truth["RAW"]
    assert ctx.ideal_value() == pytest.approx(built.ideal, abs=1e-12)
    clean = EM.prepare(TK.build(TK.make_spec("ising", 4, steps=2)), ["RAW"], seed=1).truth["RAW"].mean
    assert abs(prep.truth["RAW"].mean - clean) > 1e-3


def test_temporal_mixture_is_the_average_of_its_two_laws():
    from qemx.methods import Runner
    spec = TK.make_spec("ghz", 3, theta=1.0472, truth="mix=2")
    built = TK.build(spec)
    phys = tuple(built.compiled.info["phys"])
    r = MM.truth_runner(spec, phys)
    a = Runner(backend="fake_fez@eng=exact", layout=phys)
    b = Runner(backend="fake_fez@eng=exact,s=2.0", layout=phys)
    from qemx.noise import NoiseSpec
    args = (built.compiled.implemented, NoiseSpec(), built.compiled.measure_map)
    assert np.allclose(r.run(*args).probs_meas, 0.5 * a.run(*args).probs_meas + 0.5 * b.run(*args).probs_meas)


def test_echo_circuit_is_the_identity_and_its_null_false_positive_rate_is_alpha():
    from qemx import templates as T
    built = TK.build(TK.make_spec("hea", 3, layers=2))
    echo = MM.echo_circuit(built.compiled)
    sv = T.statevector(echo.n, echo.ops)
    assert abs(sv[0]) ** 2 == pytest.approx(1.0, abs=1e-9)
    assert abs(MM.null_rate(0.9, 2000, 0.05, R=20000) - 0.05) < 0.012
    assert MM.diag_shots(1000) == 200 and MM.diag_shots(10 ** 6) == 20000


def test_learning_protocols_redraw_their_training_design_across_trials():
    """A deployment draws a fresh training design (CDR's Clifford substitutions, NIL's training circuits and
    neighbors); with K designs the trials are spread round-robin over K independently drawn designs."""
    built = TK.build(TK.make_spec("hea", 3, layers=1, seed=0))
    prep = EM.prepare(built, ["CDR", "NIL"], seed=11, designs=3)
    for key in ("CDR", "NIL"):
        ds = prep.designs[key]
        assert len(ds) == 3 and len({c.seed for _, c in ds}) == 3
    refs = [r.reference_mean for r, _ in prep.designs["NIL"]]
    assert len(set(np.round(refs, 12))) > 1                          # different designs, different fits
    a = EM.trials(prep, "NIL", 3_000_000, 9, np.random.default_rng(2)).estimates
    b = EM.trials(prep, "NIL", 3_000_000, 9, np.random.default_rng(2)).estimates
    assert np.array_equal(a, b)
    one = EM.prepare(built, ["NIL"], seed=11, designs=1)
    assert "NIL" not in one.designs


def test_learning_protocols_on_a_clifford_circuit_fall_back_without_crashing():
    """A Clifford circuit has no slots to train on: CDR and NIL return the readout-corrected value; their early-return
    samplers accept the emulator's keywords (θ̂, training shots) like the full ones."""
    built = TK.build(TK.make_spec("ghz", 3, theta=math.pi / 2))
    prep = EM.prepare(built, ["RO", "CDR", "NIL"], seed=2, designs=2)
    ctx = prep.ctx_truth["RO"]
    th = ctx.sample_cal_theta(1000, 6, np.random.default_rng(0))
    for key in ("CDR", "NIL"):
        res = prep.truth[key]
        x = res.sampler(100000, 6, np.random.default_rng(1), theta=th, s_tr=64)     # a design that fell back
        assert np.all(np.isfinite(x))
        tr = EM.trials(prep, key, 3_000_000, 6, np.random.default_rng(1))
        assert not tr.allocation["feasible"] and "training" in tr.allocation["reason"]   # nothing to train on


def test_each_labeling_session_records_its_code_and_a_resume_lists_what_changed(tmp_path, monkeypatch):
    import json
    from qemx.selection import run as RN
    RN.record_provenance(str(tmp_path), "development")
    rec = json.load(open(tmp_path / "provenance.json", encoding="utf-8"))
    s = rec["sessions"][0]
    assert rec["stage"] == "development" and "qemx/selection/emulate.py" in s["engine_sha256"]
    assert s["provenance"]["app_version"] and s["provenance"]["source_sha256"]
    assert "inventory" not in s["provenance"]                 # digest only; no local paths, no package list
    real = RN.engine_digests()
    monkeypatch.setattr(RN, "engine_digests", lambda: dict(real, **{"qemx/methods/nil.py": "0" * 64}))
    rec = RN.record_provenance(str(tmp_path), "development")
    assert len(rec["sessions"]) == 2 and rec["sessions"][1]["changed_since_first"] == ["qemx/methods/nil.py"]


def test_resource_estimate_scales_trials_and_extrapolates_widths_upward():
    from qemx.selection import run as RN
    cfg = dict(R=10, budgets=[1000, 10000], protocols=["RAW", "RO"], final_test_fraction=0.0,
               tasks=dict(widths=[3, 5], families={"ghz": [{"theta": 1.0}]}, inputs=["default"],
                          observables=["default"], layouts=["best"], devices=["fake_fez"], variants_per_base=0))
    m = [dict(n=3, prepare=1.0, per_trial=1e-3, designs=1, json_bytes=1000, npz_bytes_per_trial=8.0),
         dict(n=4, prepare=2.0, per_trial=2e-3, designs=1, json_bytes=1000, npz_bytes_per_trial=8.0)]
    est = RN.estimate(cfg, measured=m)
    # n = 3: 1 + 1e-3·(2·10·2·2) = 1.08 s; n = 5 extrapolated (doubling per qubit): 4 + 4e-3·80 = 4.32 s
    assert est["tasks"] == 2 and est["extrapolated_widths"] == [5]
    assert est["cpu_hours"] * 3600 == pytest.approx(1.08 + 4.32, rel=1e-6)
    assert est["disk_mb"] == pytest.approx(2 * (1000 + 8.0 * 80) / 1e6)
