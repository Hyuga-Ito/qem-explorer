"""Hardware against fake-device divergence (qemx/divergence.py, scripts/dirA_divergence.py, the difference volumetric
plot): metrics, the plan (layouts, ideal distributions, equal budgets, as-executed circuits), the predictors' noise
models, a null test (no unmodeled noise: no divergence), an injection test (known unmodeled crosstalk: found where it
acts and nowhere else), token hygiene of the hardware path (mocked service), and the command line end to end."""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dirA_divergence as DV  # noqa: E402
import dirA_figures as F  # noqa: E402
import dirA_io as IO  # noqa: E402
from qemx import divergence as D  # noqa: E402

SECRET = "SECRET-TOKEN-7f3a9c"

SUITE = dict(study="div-test", kind="divergence", device="ibm_fez", fake="fake_fez", budget=2000, cal_shots=100,
             replicates=120, bootstrap=60, exact_max_qubits=3, aer_dm_max_qubits=6,
             methods={"RAW": {"kind": "raw"}, "RO": {"kind": "ro"},
                      "ZNE-G": {"kind": "zne", "folding": "global", "scales": [1, 3, 5]},
                      "ZNE-L": {"kind": "zne", "folding": "local", "scales": [1, 3, 5]}},
             circuits=[{"family": "ghz", "n": [2, 4], "theta": math.pi / 2, "basis": "Z"},
                       {"family": "qft", "n": 3, "m": None, "s": 1},
                       {"family": "brickwork", "n": 4, "layers": 4, "seed": 0}])


# ------------------------------------------------------------------ metrics
def test_hellinger_and_normalized_fidelity():
    p = np.array([0.5, 0.5, 0, 0])
    assert D.hellinger_fidelity(p, p) == pytest.approx(1.0)
    assert D.hellinger_fidelity(p, [0, 0, 0.5, 0.5]) == pytest.approx(0.0)
    assert D.hellinger_fidelity(p, [1, 0, 0, 0]) == pytest.approx(0.5)
    assert D.normalized_fidelity(p, p) == pytest.approx(1.0)
    assert D.normalized_fidelity(p, np.full(4, 0.25)) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(D.normalized_fidelity(np.full(4, 0.25), p))
    Q = np.array([p, [1, 0, 0, 0]])
    assert np.allclose(D.hellinger_fidelity(p, Q), [1.0, 0.5])


def test_nearest_probability_is_the_simplex_projection():
    from scipy.optimize import minimize
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.normal(0, 0.3, 8)
        q += (1 - q.sum()) / 8                                       # a quasi-distribution (sums to 1)
        x = D.nearest_probability(q)
        assert x.min() >= 0 and x.sum() == pytest.approx(1.0)
        res = minimize(lambda y: np.sum((y - q) ** 2), np.full(8, 1 / 8), method="SLSQP", bounds=[(0, 1)] * 8,
                       constraints=[{"type": "eq", "fun": lambda y: y.sum() - 1}], options=dict(ftol=1e-14))
        assert np.allclose(x, res.x, atol=1e-6)
    p = rng.dirichlet(np.ones(8))
    assert np.allclose(D.nearest_probability(p), p)                 # a distribution is its own projection
    Q = rng.normal(0, 0.2, (5, 8)) + 1 / 8
    assert np.allclose(D.nearest_probability(Q), np.array([D.nearest_probability(r) for r in Q]))


def test_tensored_readout_mitigation_inverts_the_confusion():
    rng = np.random.default_rng(1)
    n = 3
    t = rng.dirichlet(np.ones(2 ** n))
    p01, p10 = np.array([0.02, 0.05, 0.01]), np.array([0.04, 0.03, 0.08])

    def confuse(v):
        T = v.reshape([2] * n)
        for i in range(n):
            A = np.array([[1 - p01[i], p10[i]], [p01[i], 1 - p10[i]]])
            ax = n - 1 - i
            T = np.moveaxis(np.tensordot(A, T, axes=([1], [ax])), 0, ax)
        return T.reshape(-1)
    m = confuse(t)
    f0, f1 = confuse(np.eye(2 ** n)[0]), confuse(np.eye(2 ** n)[-1])
    a, b = D.readout_flips(f0, f1, n)
    assert np.allclose(a, p01) and np.allclose(b, p10)
    q, bad = D.mitigate_tensored(m, a, b, n)
    assert np.allclose(q[0], t, atol=1e-12) and bad == []
    q, bad = D.mitigate_tensored(m, [0.6, 0.0, 0.0], [0.5, 0.0, 0.0], n)       # bit 0 near-singular: left alone
    assert bad == [0]
    assert np.allclose(D.bit_marginals(np.eye(8)[1], 3), [[1, 0, 0]])           # index bit i = classical bit i


def test_suite_validation():
    s = D.suite_from_manifest(SUITE)
    assert [m.key for m in s.methods] == ["RAW", "RO", "ZNE-G", "ZNE-L"] and len(s.benchmarks) == 4
    for bad in (dict(SUITE, extra=1), dict(SUITE, budget=150),
                dict(SUITE, methods={"Z": {"kind": "zne", "folding": "diagonal"}}),
                dict(SUITE, methods={"R": {"kind": "raw", "scales": [1, 3]}}),
                dict(SUITE, circuits=[{"family": "teleport", "n": 3}]), dict(SUITE, predictors=["exact"]),
                dict(SUITE, replicates=1), dict(SUITE, methods={"Z": {"kind": "zne", "scales": [1, 3, 3]}})):
        with pytest.raises(ValueError):
            D.suite_from_manifest(bad)
    with open(os.path.join(ROOT, "manifests", "direction_A_divergence_v1.yaml"), encoding="utf-8") as f:
        import yaml
        full = D.suite_from_manifest(yaml.safe_load(f))
    assert len(full.benchmarks) == 59 and full.predictors == ("depolarizing", "chen", "device")


# ------------------------------------------------------------------ plan
@pytest.fixture(scope="module")
def backend():
    from qemx.device import fake_backend
    return fake_backend("fake_fez")


@pytest.fixture(scope="module")
def plan(backend):
    p = D.make_plan(D.suite_from_manifest(SUITE), backend)
    assert not p.errors
    return p


@pytest.fixture(scope="module")
def models(plan, backend):
    return D.snapshot(plan, backend)


def test_chains_are_connected_paths(backend):
    tg = backend.target
    for n in (2, 3, 5, 9):
        c = D.chain_for_backend(backend, n)
        assert len(set(c)) == n
        for a, b in zip(c[:-1], c[1:]):
            assert (a, b) in tg["cz"] or (b, a) in tg["cz"]


def test_plan_instances_and_budgets(plan):
    from qemx.compile import with_io
    from qemx.engine import simulate
    from qemx.noise import NoiseSpec
    for inst in plan.instances:
        assert len(inst.phys) == inst.n and inst.ideal.sum() == pytest.approx(1.0)
        rr = simulate(inst.spec, NoiseSpec(), "numpy", measure_map=inst.mm, ideal=True)
        assert np.allclose(rr.probs_meas, inst.ideal, atol=1e-10)               # statevector = density matrix
        for m in plan.suite.methods:
            runs = plan.runs_of(inst.key, m.key)
            assert sum(r.shots for r in runs) == plan.suite.budget                # equal total budget
            if m.kind == "zne":
                folds = [r for r in runs if r.role.startswith("fold:")]
                assert [r.lam for r in folds] == pytest.approx([1, 3, 5])
                assert folds[0].shots > folds[1].shots > folds[2].shots          # N_j ∝ |η_j| = 15/8, 5/4, 3/8
    ghz = next(i for i in plan.instances if i.label == "GHZ-4 (Z)")
    assert ghz.ideal[0] == pytest.approx(0.5) and ghz.ideal[15] == pytest.approx(0.5)


def test_plan_round_trip_and_digests(plan):
    p2 = D.Plan.from_json(json.loads(json.dumps(plan.to_json())))
    assert [r.digest for r in p2.runs] == [r.digest for r in plan.runs]
    for r in p2.runs:
        assert D.spec_digest(p2.spec_of(r)) == r.digest
    bad = D.Plan.from_json(json.loads(json.dumps(plan.to_json())))
    r = next(x for x in bad.runs if x.role.startswith("fold:local"))
    r.digest = "0" * 20
    with pytest.raises(RuntimeError, match="digest"):
        bad.spec_of(r)


def test_physical_circuits_are_isa_and_keep_their_folds(plan, backend):
    circs = D.physical_circuits(plan, backend)
    assert len(circs) == len(plan.runs)
    names = set(backend.target.operation_names)
    by = {r.key: r for r in plan.runs}
    for k, qc, shots in circs:
        assert {ins.operation.name for ins in qc.data} <= names | {"barrier"}
        r = by[k]
        inst = plan.instance(r.instance)
        n2 = sum(1 for ins in qc.data if ins.operation.num_qubits == 2)
        if r.lam is not None:
            assert n2 == round(r.lam * inst.two_qubit)
        assert shots == r.shots


# ------------------------------------------------------------------ predictors
def test_predictor_models(plan, models):
    from qemx.device_exact import gate_error_table
    inst = plan.instances[1]
    dm = D.dm_from_json(models[inst.key])
    dep, chen, dev = (D.predictor_model(dm, k) for k in ("depolarizing", "chen", "device"))
    assert dev is dm and D.relaxation_free(dep) and D.relaxation_free(chen) and not D.relaxation_free(dm)
    assert dep.errors == dm.errors and [q.p01 for q in dep.qubits] == [q.p01 for q in dm.qubits]
    e2 = {v for (nm, qa), v in chen.errors.items() if nm == "cz"}
    assert len(e2) == 1 and all(q.p01 == 0 and q.p10 == 0 for q in chen.qubits)
    # the depolarizing channel installed after a CZ has the reported error as its average gate infidelity
    (nm, qa), r = next(((k, v) for k, v in dep.errors.items() if k[0] == "cz"))
    K = gate_error_table(dep)[(nm, qa)]
    d = 4
    Fpro = sum(abs(np.trace(k)) ** 2 for k in K) / d ** 2
    assert 1 - (d * Fpro + 1) / (d + 1) == pytest.approx(r, rel=1e-6)


def test_engines_agree(plan, models):
    inst = next(i for i in plan.instances if i.label == "QFT-3")
    dm = D.predictor_model(D.dm_from_json(models[inst.key]), "device")
    p_ex = D.measured_probs(inst.spec, inst.mm, dm, engine="exact")
    p_dm = D.measured_probs(inst.spec, inst.mm, dm, engine="aer-dm")
    p_mc = D.measured_probs(inst.spec, inst.mm, dm, engine="aer-mc", shots=40000, seed=3)
    assert np.abs(p_ex - p_dm).max() < 1e-6
    assert np.abs(p_ex - p_mc).max() < 5 * np.sqrt(0.25 / 40000) + 1e-3


# ------------------------------------------------------------------ null and injection tests
@pytest.fixture(scope="module")
def predicted(plan, models):
    pred, errs = D.predict(plan, models, ("depolarizing",))
    assert not errs
    return pred


def test_null_no_unmodeled_noise_no_divergence(plan, models, predicted):
    counts, errs = D.emulate(plan, models, truth="depolarizing", seed=11)
    assert not errs and len(counts) == len(plan.runs)
    rows = pd.DataFrame(D.compare(plan, counts, predicted, seed=2))
    assert (rows["status"] == "ok").all() and len(rows) == len(plan.instances) * 4
    assert rows["z"].abs().max() < 4.5                          # shot noise only
    assert abs(rows["z"].mean()) < 1.0


def test_injected_crosstalk_is_found_where_it_acts(plan, models, predicted):
    counts, errs = D.emulate(plan, models, truth="depolarizing", inject={"crosstalk": 0.35}, seed=11)
    assert not errs
    rows = DV.divergence_table(D.compare(plan, counts, predicted, seed=2))
    raw = rows[rows["method"] == "RAW"].set_index("label")
    # GHZ-2 has no spectator qubit, so the injected crosstalk never acts on it: no divergence
    assert (rows[rows["label"] == "GHZ-2 (Z)"]["z"].abs() < 4.5).all()
    # every other circuit has spectators next to its CZ gates (in the compiled circuit the ZZ phases are turned into
    # population errors by the following single-qubit gates): the device is significantly worse than its model
    for lab in ("GHZ-4 (Z)", "QFT-3", "Brick-4 (L=4, seed 0)"):
        assert raw.loc[lab, "z"] > 3 and raw.loc[lab, "delta"] > 0, lab
    top = DV.top_divergent(rows)
    assert "GHZ-2 (Z)" not in set(top["label"])
    assert top["label"].iloc[0] == raw["delta"].drop("GHZ-2 (Z)").idxmax() or top["method"].iloc[0] != "RAW"
    S = DV.structure_summary(rows)
    assert set(S["family"]) == {"ghz", "qft", "brickwork"}
    fig = F.difference_volumetric(rows, "RAW", "depolarizing", annotate=4)
    texts = _texts(fig)
    assert any(top["label"].iloc[0] in t for t in texts)                 # the plot names the worst structure
    import matplotlib.pyplot as plt
    plt.close("all")


def test_emulation_does_not_depend_on_workers(plan, models, monkeypatch):
    import types
    # spawned workers re-import __main__; under pytest (or after an AppTest run) it is not a script they can import
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    a, _ = D.emulate(plan, models, truth="chen", seed=5)
    b, _ = D.emulate(plan, models, truth="chen", seed=5, workers=2)
    assert a == b


def test_pool_falls_back_to_serial_when_workers_cannot_start(monkeypatch):
    import types
    main = types.ModuleType("__main__")
    main.__file__ = "/nonexistent/not_a_script.py"                    # every spawned worker fails to start
    monkeypatch.setitem(sys.modules, "__main__", main)
    out = dict(D._pool_map(abs, [-1, -2, -3], 2, None, ["a", "b", "c"]))
    assert out == {0: 1, 1: 2, 2: 3}


def test_pool_reports_a_dead_worker_and_never_reruns_its_task(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    # os._exit kills the worker that runs it, as the out-of-memory killer would; rerunning it here would end pytest
    out = dict(D._pool_map(os._exit, [3, 3], 2, None, ["a", "b"]))
    assert set(out) == {0, 1} and all(isinstance(v, RuntimeError) and "died" in str(v) for v in out.values())


def test_run_local_sampler_path(plan, backend, models):
    """The hardware path end to end in Qiskit's local mode: ISA circuits through SamplerV2, counts per run."""
    counts = D.run_local(plan, backend, {k: D.dm_from_json(v) for k, v in models.items()}, seed=3)
    assert set(counts) == {r.key for r in plan.runs}
    for r in plan.runs:
        n = plan.instance(r.instance).n
        assert sum(counts[r.key].values()) == r.shots
        assert all(0 <= int(k) < 2 ** n for k in counts[r.key])
    cal1 = next(r for r in plan.runs if r.role == "cal1")
    n = plan.instance(cal1.instance).n
    assert counts[cal1.key].get(str(2 ** n - 1), 0) > 0.8 * cal1.shots          # |1…1⟩ reads mostly 1…1


# ------------------------------------------------------------------ figures
def _synthetic():
    rows = []
    rng = np.random.default_rng(0)
    for fam, mk in (("qft", "QFT"), ("ghz", "GHZ"), ("ising", "Ising")):
        for n in (2, 4, 6):
            for m in ("RAW", "ZNE-G"):
                d = (0.3 if fam == "ising" and n == 6 else 0.01) + rng.normal(0, 0.005)
                rows.append(dict(instance="%s|%d" % (fam, n), family=fam, label="%s-%d" % (mk, n), width=n,
                                 two_qubit=3 * n * (3 if fam == "ising" else 1), depth=10 * n, norm_depth=4 * n,
                                 method=m, predictor="depolarizing", status="ok", fc_obs=0.9 - d, fc_pred=0.9,
                                 fc_pred_sd=0.01, fc_obs_se=0.01, delta=d, z=d / 0.01, delta_nf=d, z_nf=d / 0.01))
    return pd.DataFrame(rows)


def _texts(fig) -> list:
    """Every text of a figure: axes texts, legends (axes and figure), figure texts."""
    out = [t.get_text() for t in fig.texts]
    for leg in list(fig.legends) + [a.get_legend() for a in fig.axes if a.get_legend() is not None]:
        out += [t.get_text() for t in leg.get_texts()]
    for a in fig.axes:
        out += [t.get_text() for t in a.texts]
    return out


def test_difference_volumetric_plot():
    import matplotlib.pyplot as plt
    D_ = _synthetic()
    fig = F.difference_volumetric(D_, "RAW", annotate=3)
    assert any("Ising-6" in t for t in _texts(fig))                     # the worst circuit is numbered and named
    assert "1" in [t.get_text() for t in fig.axes[0].texts]
    assert fig.axes[0].get_ylabel() == "circuit width (qubits)"
    plain = F.difference_volumetric(D_, "RAW")                          # default: tiles and significance dots only
    assert not plain.axes[0].texts
    sig_dots = [ln for ln in plain.axes[0].lines if ln.get_marker() == "o"]
    assert sig_dots                                                      # Ising-6 tiles: z > 3
    F.difference_volumetric(D_, "RAW", bins="sqrt2", circuits=True)
    for x in ("depth", "norm_depth"):
        F.difference_volumetric(D_, "ZNE-G", x=x, value="delta_nf")
    g = F.difference_volumetric_grid(D_, ["RAW", "ZNE-G"])
    panels = [a for a in g.axes if a.get_title(loc="left")]
    assert len(panels) >= 2
    assert all(a.get_ylim() == panels[0].get_ylim() and a.get_xlim() == panels[0].get_xlim()
               for a in panels)                                          # the same rows and bins in every panel
    F.predicted_vs_observed(D_, ["RAW", "ZNE-G"])
    empty = F.difference_volumetric(D_, "NONE")
    assert "no comparable" in empty.axes[0].texts[0].get_text()
    plt.close("all")


# ------------------------------------------------------------------ the command line and the token
def _write_manifest(tmp_path):
    import yaml
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(dict(SUITE, circuits=SUITE["circuits"][:2])), encoding="utf-8")
    return str(p)


def test_cli_plan_emulate_analyze(tmp_path):
    run = str(tmp_path / "run")
    assert DV.main(["plan", _write_manifest(tmp_path), "--out", run]) == 0
    assert DV.main(["emulate", run, "--tag", "x", "--truth", "depolarizing", "--inject", "crosstalk=0.3"]) == 0
    out = str(tmp_path / "figs")
    assert DV.main(["analyze", run, "--tag", "x", "--out", out, "--formats", "png", "--replicates", "60",
                    "--bootstrap", "30", "--predictors", "depolarizing", "chen"]) == 0
    for t in ("divergence", "structures", "rates", "overview"):
        assert os.path.exists(os.path.join(out, "tables", t + ".csv")), t
    assert os.path.exists(os.path.join(out, "divergence_volumetric_depolarizing.png"))
    rep = open(os.path.join(out, "tables", "report.md"), encoding="utf-8").read()
    assert "EMULATED" in rep
    # the cached predictions are reused (same key) and the analysis is reproducible
    assert DV.main(["analyze", run, "--tag", "x", "--out", out, "--no-figures", "--replicates", "60",
                    "--bootstrap", "30", "--predictors", "depolarizing", "chen"]) == 0
    with pytest.raises(SystemExit):
        DV.obs_dir(run, "../escape")
    # redraw from the saved table on Lubinski's normalized depth, for the normalized fidelity
    assert DV.main(["figures", os.path.join(out, "tables", "divergence.csv"), "--out", out, "--x", "norm_depth",
                    "--value", "delta_nf", "--formats", "png"]) == 0
    assert os.path.exists(os.path.join(out, "divergence_volumetric_depolarizing_norm_depth_nf.png"))


class _Job:
    def __init__(self, jid, n_pubs):
        self._id, self._n = jid, n_pubs

    def job_id(self):
        return self._id

    def status(self):
        return "DONE"

    def result(self):
        pub = SimpleNamespace(data=SimpleNamespace(c=SimpleNamespace(get_counts=lambda: {"00": 7, "11": 3})))
        return [pub] * self._n

    def metrics(self):
        return {"timestamps": {"running": "2026-10-01T10:00:00Z"}}


def test_token_never_written_or_printed(tmp_path, monkeypatch, capsys, backend):
    import qiskit_ibm_runtime as rt
    seen = {}
    jobs = {}

    class Service:
        def __init__(self, token=None, **kw):
            seen["token"] = token

        def backend(self, name):
            return backend

        def job(self, jid):
            return jobs[jid]

    class Sampler:
        def __init__(self, mode=None):
            self.options = SimpleNamespace(max_execution_time=None)

        def run(self, circs, shots=None):
            j = _Job("job-%d" % len(jobs), len(circs))
            jobs[j.job_id()] = j
            return j

    def no_batch(**kw):
        raise RuntimeError("batch mode not available")
    monkeypatch.setattr(rt, "QiskitRuntimeService", Service)
    monkeypatch.setattr(rt, "SamplerV2", Sampler)
    monkeypatch.setattr(rt, "Batch", no_batch)
    monkeypatch.setenv("QISKIT_IBM_TOKEN", SECRET)
    run = str(tmp_path / "run")
    assert DV.main(["plan", _write_manifest(tmp_path), "--out", run]) == 0
    led = str(tmp_path / "ledger.json")
    with pytest.raises(SystemExit, match="campaign ledger"):          # 1.5: no ledger, no submission (and no login)
        DV.main(["run-ibm", run, "--tag", "hw", "--backend", "fake_fez", "--ledger", led])
    assert "token" not in seen
    from qemx import campaign as CG
    CG.Ledger.create(led, 600, 60)
    assert DV.main(["run-ibm", run, "--tag", "hw", "--backend", "fake_fez", "--ledger", led]) == 0
    assert seen["token"] == SECRET                                  # the token reached the service …
    assert DV.main(["collect-ibm", run, "--tag", "hw", "--ledger", led]) == 0
    st = CG.Ledger.load(led).status()
    assert st["jobs_by_state"].get("completed") == len(jobs) and st["outstanding_s"] == 0
    out = capsys.readouterr()
    assert SECRET not in out.out and SECRET not in out.err          # … and nothing else
    for dirpath, _, files in os.walk(run):
        for f in files:
            with open(os.path.join(dirpath, f), "rb") as fh:
                assert SECRET.encode() not in fh.read(), f
    jobs_json = json.load(open(os.path.join(run, "obs", "hw", "jobs.json"), encoding="utf-8"))
    assert all(set(r) == {"job_id", "backend", "shots", "runs", "submitted", "max_execution_time", "mode",
                          "reservation", "client_tag"} for r in jobs_json["records"])     # 1.5.1: + mode, ledger ids
    assert all(r["mode"] == "job" and r["client_tag"] == CG.job_tag(r["reservation"]) for r in jobs_json["records"])
    assert all(r["max_execution_time"] > 0 for r in jobs_json["records"])          # from each job's high estimate
    assert os.path.exists(os.path.join(run, "obs", "hw", "counts.json"))


def test_partial_submission_resumes_without_resubmitting(tmp_path, monkeypatch, backend):
    """A provider error after one job ("the queue is full") does not prove that no job was created (1.5.1, review
    F04): the reservation stays committed as submission_unknown and blocks every further submission. Only after a
    provider check (no job carries the call's client tag) is it resolved as not submitted; --resume then submits
    exactly the runs not yet covered."""
    import qiskit_ibm_runtime as rt
    jobs, fail = {}, {"after": 1}

    class Service:
        def __init__(self, token=None, **kw):
            pass

        def backend(self, name):
            return backend

        def jobs(self, job_tags=None, limit=None, **kw):              # the provider's job search by tag
            return [j for j in jobs.values() if set(job_tags or []) <= set(j.tags)]

    class Sampler:
        def __init__(self, mode=None):
            self.options = SimpleNamespace(max_execution_time=None, environment=SimpleNamespace(job_tags=None))

        def run(self, circs, shots=None):
            if fail["after"] is not None and len(jobs) >= fail["after"]:
                raise RuntimeError("the queue is full")
            j = _Job("job-%d" % len(jobs), len(circs))
            j.tags = list(self.options.environment.job_tags or [])
            jobs[j.job_id()] = j
            return j

    def no_batch(**kw):
        raise RuntimeError("batch mode not available")
    monkeypatch.setattr(rt, "QiskitRuntimeService", Service)
    monkeypatch.setattr(rt, "SamplerV2", Sampler)
    monkeypatch.setattr(rt, "Batch", no_batch)
    run = str(tmp_path / "run")
    assert DV.main(["plan", _write_manifest(tmp_path), "--out", run]) == 0
    from qemx import campaign as CG
    led = str(tmp_path / "ledger.json")
    CG.Ledger.create(led, 600, 60)
    L = ["--ledger", led]
    jp = os.path.join(run, "obs", "hw", "jobs.json")
    with pytest.raises(SystemExit, match="outcome of the next call is unknown"):
        DV.main(["run-ibm", run, "--tag", "hw", "--backend", "fake_fez"] + L)
    first = json.load(open(jp, encoding="utf-8"))
    assert first["partial"] and len(first["records"]) == 1 and len(first["pending"]) == 1
    assert first["records"][0]["client_tag"] in jobs["job-0"].tags                 # every job carries its tag
    led_entries = CG.Ledger.load(led).entries
    assert [e["state"] for e in led_entries] == ["submitted", "submission_unknown"]  # may be a job: still committed
    tag = first["pending"][0]["client_tag"]
    for extra in ([], ["--resume"], ["--force"]):                                  # nothing is sent while unknown
        with pytest.raises(SystemExit, match="unknown"):
            DV.main(["run-ibm", run, "--tag", "hw", "--backend", "fake_fez"] + L + extra)
    assert len(jobs) == 1
    assert CG.main(["resolve", "--ledger", led, "--tag", tag, "--not-submitted"]) == 0   # provider checked: no job
    with pytest.raises(SystemExit, match="partial"):
        DV.main(["run-ibm", run, "--tag", "hw", "--backend", "fake_fez"] + L)      # never resubmits silently
    fail["after"] = None
    assert DV.main(["run-ibm", run, "--tag", "hw", "--backend", "fake_fez", "--resume"] + L) == 0
    done = json.load(open(jp, encoding="utf-8"))
    keys = [k for r in done["records"] for k in r["runs"]]
    plan = DV.load_plan(run)
    assert not done["partial"] and not done["pending"] and done["records"][0] == first["records"][0]
    assert sorted(keys) == sorted(r.key for r in plan.runs) and len(keys) == len(set(keys))   # each run once
    assert len(jobs) == len(done["records"])


def test_collect_keeps_the_counts_of_finished_jobs_when_one_fails():
    class Failed(_Job):
        def status(self):
            return "RUNNING"

        def result(self):
            raise RuntimeError("the job ended in error")
    jobs = {"a": _Job("a", 2), "b": Failed("b", 1)}
    service = SimpleNamespace(job=lambda jid: jobs[jid])
    counts, status = D.collect(service, [dict(job_id="a", runs=["r1", "r2"]), dict(job_id="b", runs=["r3"])],
                               wait=True)
    assert set(counts) == {"r1", "r2"} and counts["r1"] == {"0": 7, "3": 3}
    assert [s["failed"] for s in status] == [False, True]


def test_volumetric_significance_follows_the_plotted_value():
    import matplotlib.pyplot as plt
    D_ = _synthetic()
    D_["z_nf"] = 0.0                                                     # nothing significant in normalized fidelity
    fig = F.difference_volumetric(D_, "RAW", value="delta_nf")
    assert not fig.axes[0].texts
    assert all(max(c.get_linewidths(), default=0) < 1 for c in fig.axes[0].collections)
    plt.close("all")


def test_connection_errors_mask_the_token(monkeypatch):
    import qiskit_ibm_runtime as rt

    class Broken:
        def __init__(self, token=None, **kw):
            raise ValueError("invalid token %s for channel" % token)
    monkeypatch.setattr(rt, "QiskitRuntimeService", Broken)
    monkeypatch.setenv("QISKIT_IBM_TOKEN", SECRET)
    with pytest.raises(SystemExit) as e:
        DV.connect()
    assert SECRET not in str(e.value) and "***" in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


def test_submitted_circuits_run_as_written(backend):
    """No dynamical decoupling (it would fill the idle windows the analysis measures) and no service-side twirling,
    whatever the runtime's defaults; an options object without those fields is left alone."""
    import warnings

    from qiskit_ibm_runtime import SamplerV2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.twirling.enable_gates = True
    D._plain_sampler(s)
    assert s.options.dynamical_decoupling.enable is False and s.options.twirling.enable_gates is False
    D._plain_sampler(SimpleNamespace(options=SimpleNamespace()))                 # no such options: no error


def test_refresh_features_of_an_old_plan(tmp_path):
    """A plan made before 1.4.1 has no context features: refresh-features recomputes them without touching the
    circuits (digests), keeps the old plan, and the χ² ladder says so when they are missing."""
    run = str(tmp_path / "run")
    assert DV.main(["plan", _write_manifest(tmp_path), "--out", run]) == 0
    with open(os.path.join(run, "plan.json"), encoding="utf-8") as f:
        d = json.load(f)
    digests = [r["digest"] for r in d["runs"]]
    for i in d["instances"]:
        i["features"] = None
    with open(os.path.join(run, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(d, f)
    assert DV.main(["refresh-features", run]) == 0
    assert os.path.exists(os.path.join(run, "plan.before-features.json"))
    plan = DV.load_plan(run)
    assert [r.digest for r in plan.runs] == digests
    assert all(i.features and "gate_error" in i.features and "idle_us" in i.features for i in plan.instances)
    L = pd.DataFrame(dict(model=["none", "calibration (priced)", "context", "calibration + context"],
                          chi2=[100.0, 60.0, 100.0, 60.0], dof=[50, 49, 50, 49], chi2_per_dof=[2.0, 1.22, 2.0, 1.22],
                          p=[1e-5, 0.1, 1e-5, 0.1], circuits=50, priced=[np.nan, 0.4, np.nan, 0.4],
                          p_context_added=[np.nan] * 3 + [1.0], context_features=0))
    assert DV.ladder_verdict(L).startswith("no context features in this plan")
