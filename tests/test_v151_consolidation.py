"""1.5.1 consolidation: invariants that join the repaired clusters (independent review F01/F05/F08 follow-ups N12,
N32, N33, the simulated ZNE distribution, provenance digests). Everything is local: mocks for the provider."""
import json
import math
import os
import shutil
import warnings

import numpy as np
import pytest

from qemx import campaign as CG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _est(high):
    return CG.UsageEstimate(1, 1, 1, high / 2, high / 1.3, float(high), ())


# ====================================================================== N12: provider limits inside the allowance
def test_ledger_reserves_the_provider_limit_so_limits_never_exceed_the_allowance(tmp_path):
    """1.5.0 reserved the high estimate but sent max_execution_time = ⌈1.1·high⌉ + 1 s: with every job in flight
    overrunning, provider-enforced usage could exceed ceiling − reserve. Now Σ limits ≤ allowance."""
    led = CG.Ledger.create(str(tmp_path / "l.json"), ceiling_s=600, reserve_s=60)
    rng = np.random.default_rng(3)
    limits = []
    while True:
        est = _est(float(rng.uniform(5, 60)))
        met = CG.default_max_execution_time(est)
        assert met >= est.high_s and met < est.high_s + 1           # the high estimate, rounded up
        try:
            rid = led.admit("j", est, max_execution_time=met)
        except CG.CampaignLimit:
            break
        led.begin_submit(rid, max_execution_time=met)
        limits.append(met)
    assert sum(limits) <= 540 + 1e-9
    assert led.committed_s() == pytest.approx(sum(limits))
    # a larger explicit limit reserves that much; a limit above the reservation is refused before any provider call
    led2 = CG.Ledger.create(str(tmp_path / "m.json"), ceiling_s=600, reserve_s=60)
    rid = led2.admit("big", _est(10.2), max_execution_time=30)
    assert led2.entry(rid)["reserved_s"] == 30 and led2.committed_s() == pytest.approx(30)
    rid2 = led2.admit("small", _est(10.2))
    with pytest.raises(ValueError, match="run past it"):
        led2.begin_submit(rid2, max_execution_time=12)
    # submit_one always sends a per-job limit (the default when none is given)
    seen = {}

    class Job:
        def job_id(self):
            return "J1"

    def prepare(tag):
        return True
    CG.submit_one(led2, "s", _est(7.3), lambda tag: Job(), prepare=prepare)
    e = CG.Ledger.load(led2.path).entry_for_job("J1")
    assert e["max_execution_time_s"] == 8 and e["reserved_s"] == 8


def test_preflight_admits_on_the_reservation():
    plan = dict(runs=[dict(shots=1000) for _ in range(120)])
    pf = CG.preflight(plan)
    assert all(r["reserved_s"] >= r["high_s"] and r["reserved_s"] == r["max_execution_time_s"] for r in pf["jobs"])
    assert pf["reserved_s"] == pytest.approx(sum(r["reserved_s"] for r in pf["jobs"]))


def test_ledgers_written_before_reserved_s_keep_their_committed_total(tmp_path):
    led = CG.Ledger.create(str(tmp_path / "l.json"), ceiling_s=600, reserve_s=60)
    led.admit("a", _est(10.4))
    d = json.load(open(led.path, encoding="utf-8"))
    for e in d["entries"]:
        e.pop("reserved_s")                                         # as written by an earlier 1.5.1 candidate
    json.dump(d, open(led.path, "w", encoding="utf-8"))
    assert CG.Ledger.load(led.path).committed_s() == pytest.approx(10.4)


# ====================================================================== N32: jobs split by the circuit limit
def test_submit_splits_pubs_by_the_backends_circuit_limit(tmp_path, monkeypatch):
    """1.5.0 sent every pub of one shot count in a single Sampler call, however many circuits that was (review N32
    in the budgets cluster). Each part is now its own admitted, recorded job."""
    from qiskit import QuantumCircuit
    from qemx import hardware as H
    import test_spending as TS
    warnings.simplefilter("ignore")
    plan, fake = TS._hw_plan()
    remote, made, _ = TS._mock_runtime(monkeypatch)
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    pubs = [("target", i, qc, 100) for i in range(7)] + [("cal", 0, qc, 50)]
    monkeypatch.setattr(H, "build_jobs", lambda *a, **k: [({}, pubs, ("none", "none"))])
    led = CG.Ledger.create(str(tmp_path / "l.json"), ceiling_s=600, reserve_s=60)
    recs = H.submit(plan, TS._RealLike(fake), ledger=led.path, max_circuits_per_job=3)
    assert [len(r["pubs"]) for r in recs] == [3, 3, 1, 1] and [r["shots"] for r in recs] == [100, 100, 100, 50]
    assert len(CG.Ledger.load(led.path).entries) == len(recs) == len(remote)
    assert [len(j.circs) for j in remote.values()] == [3, 3, 1, 1]
    assert H.job_circuit_limit(fake) == 300                         # fake backends report max_experiments = 300
    assert H.job_circuit_limit(object()) == H.DEFAULT_MAX_CIRCUITS_PER_JOB
    with pytest.raises(ValueError):
        H.job_circuit_limit(fake, 0)


# ====================================================================== N33: PEC per-circuit minimum in rows
def test_pec_rows_carry_their_per_circuit_minimum():
    from qemx import budget as BG
    from qemx.circuit import parse
    from qemx.compile import build
    from qemx.methods import Recipe
    from qemx.noise import NoiseSpec
    from qemx.observables import z
    from qemx.recipes import compare, summary
    ideal, outs, _ = compare(build(parse("h 0; cx 0 1", 2)), NoiseSpec(p1=0.01, p2=0.03), z(1, 2),
                             [Recipe.make("pec", shots_per_circuit=8)], shots=4000, seed=2)
    row = summary(outs[0], ideal, 4000, 1.0)
    assert row["per_circuit"] == 8
    m = BG.CostModel.of_row(row)
    assert m.per_circuit == 8 and BG.CostModel.of(outs[0].result).n_min == m.n_min >= 8


# ====================================================================== F01 in the phase ledger
def test_phase_ledger_uses_budget_dependent_moments_for_exponential_zne():
    """The interactive table and the phase ledger describe the same nonlinear protocol: below the validated
    first-order range the ledger's bias and variance are the Monte-Carlo moments, above it the first-order ones."""
    from qemx.circuit import parse
    from qemx.compile import build
    from qemx.methods import Recipe
    from qemx.noise import NoiseSpec
    from qemx.observables import z
    from qemx.phase.ledger import Cost, allocate
    from qemx.recipes import compare
    from qemx.stats import grid_at
    ideal, outs, _ = compare(build(parse("ry(1.2) 0", 1)), NoiseSpec(p1=0.15), z(0, 1),
                             [Recipe.make("zne", rule="exponential")], shots=300, seed=13)
    r = outs[0].result
    g = r.details["target_moments"]
    assert r.estimate_kind == "reference" and g["first_order_from"] is not None
    cost = Cost("ZNE-exp", b=float(r.mean) - ideal, v=float(r.var1), m=int(g["n_min"]), target_grid=g)
    plain = Cost("ZNE-exp", b=float(r.mean) - ideal, v=float(r.var1), m=int(g["n_min"]))
    lo = allocate(cost, 300)
    m = grid_at(g, lo.N)
    assert lo.source != "first-order (validated)"
    assert lo.b == pytest.approx(m["mean"] - ideal) and lo.var_target == pytest.approx(m["var"])
    assert lo.b != pytest.approx(allocate(plain, 300).b)            # 1.5.0: the reference fit's bias
    hi = allocate(cost, 4 * int(g["first_order_from"]))
    assert hi.source == "first-order (validated)"
    assert hi.mse == pytest.approx(allocate(plain, 4 * int(g["first_order_from"])).mse)
    assert Cost.from_json(cost.to_json()).target_grid == g


def test_phase_record_schema_and_rows_carry_the_moment_grid():
    from qemx.phase import store as ST
    from qemx.phase.cell import cost_of
    assert ST.SCHEMA_VERSION == "qemx.phase.cell/4" and "qemx.phase.cell/3" in ST.READABLE_SCHEMAS
    row = dict(key="Z", status="ok", b=0.1, v=1.0, m=3, target_moments={"kind": "target-moment-grid", "N": [3]})
    assert cost_of(row).target_grid == row["target_moments"]
    assert cost_of(dict(row, target_moments=None)).target_grid is None


# ====================================================================== simulated ZNE distribution
def test_zne_distribution_uses_the_estimators_folded_circuits():
    """The simulated 'ZNE-extrapolated' distribution must be the same linear combination of the same folded
    circuits as the estimator: with local folding, fractional λ and several fold draws, 1.5.0 used one draw."""
    from qemx import insight as IN
    from qemx import templates as T
    from qemx import compile as CP
    from qemx.methods import Recipe
    from qemx.noise import PRESETS
    from qemx.recipes import compare
    tpl = T.build("ghz", {"n": 3})
    comp = CP.build(tpl.spec, opts=CP.CompileOptions())
    nz = PRESETS["Hardware-realistic (illustrative)"]
    rc = Recipe.make("zne", folding="local", scales=(1, 2, 3), fold_draws=3, rule="richardson")
    ideal, outs, runner = compare(comp, nz, tpl.observable, [rc], "numpy")
    d = IN.distributions(comp, nz, outs, runner, "numpy")
    v = np.asarray(tpl.observable.values)
    assert float(v @ d["series"]["ZNE-extrapolated"]) == pytest.approx(outs[0].result.mean, abs=1e-9)


# ====================================================================== provenance
def test_source_digest_identifies_the_tree_without_git(tmp_path):
    from qemx import provenance as P
    src = tmp_path / "tree"
    for d in ("qemx", "ui", "pages", "scripts", "manifests"):
        shutil.copytree(os.path.join(ROOT, d), src / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("app.py", "pyproject.toml", "requirements.txt", "requirements-lock.txt"):
        shutil.copy(os.path.join(ROOT, f), src / f)
    a = P.source_digest(str(src))
    assert a == P.source_digest(str(src)) and a["files"] > 50
    with open(src / "qemx" / "stats.py", "a", encoding="utf-8") as f:
        f.write("\n# changed\n")
    P.source_digest.cache_clear()
    assert P.source_digest(str(src))["sha256"] != a["sha256"]
    p = P.provenance(root=str(src))
    assert p["git_commit"] is None and p["source_sha256"] and p["lock_sha256"]
    assert P.redact_path([os.path.expanduser("~/data/x.json"), "plain"]) == ["~/data/x.json", "plain"]


def test_results_and_records_carry_the_source_digest():
    from qemx import experiment as EX
    from qemx.phase.cell import provenance as cell_prov
    p = EX.provenance()
    assert p["source_sha256"] and p["inventory_sha256"] and "numpy" in p
    assert cell_prov()["source_sha256"] == p["source_sha256"]


# ====================================================================== discrepancy: mixture null, projection mass
def test_distance_test_with_uncertain_model_uses_a_mixture_null():
    """Review 5.9: counts drawn under one realization of an uncertain parameter are a mixture of multinomials; a
    single multinomial from the average distribution is anti-conservative. The mixture null is calibrated."""
    from qemx import discrepancy as DS
    rng = np.random.default_rng(11)
    p_model = np.array([0.5, 0.5])
    a = rng.uniform(0.44, 0.56, 200)
    draws = np.c_[a, 1 - a]                                     # the model's parameter uncertainty
    shots, flagged_plain, flagged_mix = 2000, 0, 0
    for k in range(150):
        truth = draws[rng.integers(len(draws))]                 # the world is one realization of the parameter
        f = rng.multinomial(shots, truth) / shots
        flagged_plain += DS.distance_test(p_model, f, shots, reps=400, rng=k)["p_value"] < 0.05
        flagged_mix += DS.distance_test(p_model, f, shots, reps=400, rng=k, p_draws=draws)["p_value"] < 0.05
    assert flagged_plain / 150 > 0.25                           # far above the nominal 5 %
    assert flagged_mix / 150 < 0.12
    r = DS.distance_test(p_model, [0.5, 0.5], shots, p_draws=draws)
    assert r["null"].startswith("mixture")
    with pytest.raises(ValueError):
        DS.null_distance(p_model, 10, p_draws=np.ones((2, 3)) / 3)


def test_projected_distributions_report_the_moved_mass():
    from qemx import divergence as DV
    n = 1
    meth = DV.Method(key="RO", kind="ro")
    F = {"target": np.array([[0.97, 0.03]]), "cal0": np.array([[0.9, 0.1]]), "cal1": np.array([[0.1, 0.9]])}
    info = {}
    q, flags = DV.estimate(meth, n, F, None, info)
    assert info["moved_mass"][0] > 0
    p01, p10 = DV.readout_flips(F["cal0"], F["cal1"], n)
    Q, _ = DV.mitigate_tensored(F["target"], p01, p10, n)
    assert info["moved_mass"][0] == pytest.approx(0.5 * np.abs(np.atleast_2d(q) - Q).sum())


# ====================================================================== export renderer: queue wait, process tree
def _alive(pid: int) -> bool:
    try:
        with open("/proc/%d/status" % pid, encoding="utf-8") as f:
            return not any(line.startswith("State:") and "Z" in line.split()[1] for line in f)
    except OSError:
        return False


@pytest.mark.skipif(os.name == "nt" or not os.path.isdir("/proc"), reason="POSIX process groups and /proc")
def test_renderer_timeout_stops_the_whole_process_tree(tmp_path, monkeypatch):
    """Review 6.5: killing only the Python worker can leave the browser it started running. A stand-in worker starts
    a long-lived grandchild and hangs; after the time limit the grandchild must be gone too."""
    import plotly.graph_objects as go
    from qemx import bundle as BU
    pidfile = tmp_path / "grandchild.pid"
    worker = tmp_path / "hang_worker.py"
    worker.write_text("import subprocess, sys, time\n"
                      "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                      "open(%r, 'w').write(str(p.pid))\n"
                      "time.sleep(120)\n" % str(pidfile), encoding="utf-8")
    monkeypatch.setattr(BU, "_WORKER_CMD", (str(worker),))
    out, err = BU.render_images({"f": go.Figure()}, timeout=3)
    assert out == {} and "did not finish" in err
    pid = int(pidfile.read_text(encoding="utf-8"))
    import time
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.1)
    assert not _alive(pid)


def test_renderer_queue_wait_counts_toward_the_limit():
    import threading
    import time
    import plotly.graph_objects as go
    from qemx import bundle as BU
    BU._RENDER_LOCK.acquire()                            # another export is rendering
    try:
        t0 = time.monotonic()
        out, err = BU.render_images({"f": go.Figure()}, timeout=1.0)
        assert out == {} and "busy" in err and time.monotonic() - t0 < 5
    finally:
        BU._RENDER_LOCK.release()
