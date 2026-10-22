"""v1.2: theory conformance of algorithms and estimators, the equal-budget accounting, hardware parity, exports
and the UI helpers (angles in π, composer I/O boxes, tooltip math, figures)."""
import io
import math
import os
import re
import zipfile

import numpy as np
import pytest

from qemx import budget as BG
from qemx import compile as CP
from qemx import templates as T
from qemx.circuit import CircuitSpec, eval_expr, fmt_angle, normalize_expr
from qemx.engine import simulate
from qemx.methods import Recipe
from qemx.noise import NoiseSpec
from qemx.recipes import compare, preset, table_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def probs(spec):
    return np.asarray(simulate(spec, NoiseSpec(), "numpy", ideal=True).probs, float)


# ------------------------------------------------------------------ algorithms
def test_qft_is_the_dft_matrix():
    from qiskit.quantum_info import Operator
    for n in (2, 3, 4):
        q = list(range(n))
        qc = CP.to_qiskit(CircuitSpec(n, tuple(T.qft_body(q) + T.swap_network(q)), "qft"))
        U = Operator(qc).data
        N = 2 ** n
        F = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
        ph = U[0, 0] / F[0, 0]
        assert abs(abs(ph) - 1) < 1e-12 and np.allclose(U, ph * F, atol=1e-12)


@pytest.mark.parametrize("n,marked", [(2, 3), (3, 5), (4, 6)])
def test_grover_success_probability(n, marked):
    th = math.asin(1 / math.sqrt(2 ** n))
    for k in (1, 2, T.grover_optimal(n)):
        tpl = T.build("grover_search", dict(n=n, marked=marked, iterations="custom", k=k))
        p = probs(tpl.spec)
        assert abs(p[marked] - math.sin((2 * k + 1) * th) ** 2) < 1e-10
        rest = np.delete(p[:2 ** n], marked)
        assert np.ptp(rest) < 1e-12                       # the unmarked items stay uniform


def test_hsp_outcomes_lie_in_the_annihilator():
    tpl = T.build("hsp", dict(group=T.HSP_GROUPS[0], n=3, h1=5, h2=3))
    p = probs(tpl.spec)
    for x in np.flatnonzero(p > 1e-12):
        y = x & 0b111
        assert bin(y & 5).count("1") % 2 == 0 and bin(y & 3).count("1") % 2 == 0
    ys = {x & 0b111 for x in np.flatnonzero(p > 1e-12)}
    assert ys == {0, 7} and abs(sum(p[x] for x in range(len(p)) if (x & 7) == 7) - 0.5) < 1e-12
    cyc = T.build("hsp", dict(group=T.HSP_GROUPS[1], n=4, j=2))
    pc = probs(cyc.spec)
    peaks = {x & 15 for x in np.flatnonzero(pc > 1e-12)}
    assert peaks == {0, 4, 8, 12}


def test_symmetry_is_a_qubit_property_under_relabeled_readout():
    tpl = T.build("hsp")
    for rl in (False, True):
        comp = CP.build(tpl.spec, None, None, CP.CompileOptions(), rl)
        _, outs, _ = compare(comp, NoiseSpec(), tpl.observable, [preset("Symmetry verification", tpl.symmetry)],
                             "numpy", 1000, tpl.symmetry)
        assert abs(outs[0].result.accept - 1.0) < 1e-12


# ------------------------------------------------------------------ estimators
def test_zne_weights_follow_their_definitions():
    from qemx.methods.zne import linear_weights, richardson_weights
    lams = np.array([1.0, 3.0, 5.0, 7.0])
    eta = richardson_weights(lams)
    for k in range(len(lams)):
        assert abs(eta @ lams ** k - (1.0 if k == 0 else 0.0)) < 1e-9
    w = linear_weights(lams)
    A = np.stack([np.ones_like(lams), lams], 1)
    assert np.allclose(w, np.linalg.pinv(A)[0], atol=1e-12)          # least-squares intercept


def test_folding_counts_only_physical_gates():
    from qemx.circuit import Op
    from qemx.transforms import achieved_lambda, fold_global
    spec = CircuitSpec(1, (Op("h", (0,)), Op("rz", (0,), (0.3,)), Op("rz", (0,), (0.2,)), Op("sx", (0,))), "1q")
    f = fold_global(spec, 3.0)
    assert abs(achieved_lambda(spec, f) - 3.0) < 1e-12


def test_pec_quasi_probabilities_invert_the_channel():
    from qemx.methods.pec import depol_quasi, gamma_of
    from qemx.noise import pauli_matrix
    for k, p in ((1, 0.05), (2, 0.02)):
        q = depol_quasi(p, k, "inverse")
        d = 2 ** k
        f = 1 - p
        if k == 1:
            assert abs(gamma_of(q) - (1 + p / 2) / (1 - p)) < 1e-12
        # (D∘N)(P) = P for every Pauli P, N the depolarizing channel ρ ↦ fρ + (1−f) I/d
        for lab in ("X" * k, "Z" + "I" * (k - 1)):
            P = pauli_matrix(lab)
            NP = f * P                                   # traceless input
            out = sum(v * pauli_matrix(l) @ NP @ pauli_matrix(l).conj().T for v, l in q)
            assert np.allclose(out, P, atol=1e-12)
        out = sum(v * pauli_matrix(l) @ np.eye(d) @ pauli_matrix(l).conj().T for v, l in q)
        assert np.allclose(out, np.eye(d), atol=1e-12)
    with pytest.raises(ValueError):
        depol_quasi(1.0, 1, "inverse")


def test_m3_keeps_signed_quasi_probabilities():
    from qemx.methods.base import m3_solve
    rng = np.random.default_rng(3)
    A = [np.array([[0.97, 0.05], [0.03, 0.95]]), np.array([[0.96, 0.08], [0.04, 0.92]])]
    p = rng.normal(0.25, 0.2, 4)
    p[1] = -0.05                                         # a signed (PEC-like) distribution
    full = np.linalg.solve(np.kron(A[1], A[0]), p)
    assert np.allclose(m3_solve(p, A, 2), full, atol=1e-12)


def test_pec_sampled_variance_matches_the_exact_one_with_readout():
    tpl = T.build("bell")
    comp = CP.build(tpl.spec)
    noise = NoiseSpec(p1=0.0, p2=0.02, ro01=0.03, ro10=0.07)
    rc = Recipe.make("pec", "none", "none", "tensored", construction="inverse", instances=256)
    _, exact, _ = compare(comp, noise, tpl.observable, [rc], "numpy", 4096)
    _, aer, _ = compare(comp, noise, tpl.observable, [rc], "aer", 4096)
    v0, v1 = exact[0].result.var1, aer[0].result.var1
    assert abs(v1 / v0 - 1) < 0.15, (v0, v1)


def test_lasso_minimizes_the_documented_objective():
    from qemx.methods.learning import fit
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 3))
    y = X @ np.array([0.8, 0.0, -0.3]) + 0.1 + 0.01 * rng.normal(size=16)
    alpha = 0.5
    c, b = fit(X, y, "lasso", alpha)
    r = y - b - X @ c                                   # KKT of Σr² + α‖c‖₁: 2Xᵀr = α·sign(c) on the support
    g = 2 * (X - X.mean(0)).T @ r
    for j in range(3):
        if abs(c[j]) > 1e-8:
            assert abs(g[j] - alpha * np.sign(c[j])) < 1e-4
        else:
            assert abs(g[j]) <= alpha + 1e-4


def test_learning_variance_includes_training_noise():
    tpl = T.build("qft")
    comp = CP.build(tpl.spec)
    from qemx.noise import PRESETS
    _, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, [preset("CDR · OLS")], "numpy", 32768)
    r = outs[0].result
    m = BG.CostModel.of(r)
    assert m.u > 0 and r.var_fixed == m.u
    x = BG.draw(r, 32768, 300, np.random.default_rng(1))
    assert abs(np.std(x, ddof=1) / math.sqrt(BG.variance(m, 32768)) - 1) < 0.2
    assert BG.budget_for(0.0, m, math.sqrt(m.u) * 0.99) == math.inf       # the floor cannot be bought down


# ------------------------------------------------------------------ equal-budget accounting
def test_calibration_split_is_optimal_and_never_free():
    m = BG.CostModel(v=1.3, fixed=1000, c=2, w=0.4)
    B = 50000
    N, s = BG.split(m, B)
    assert abs(N + m.fixed + m.c * s - B) < 1e-6
    grid = np.linspace(1, (B - m.fixed - 1) / m.c, 20001)
    best = grid[np.argmin(m.v / (B - m.fixed - m.c * grid) + m.w / grid)]
    assert abs(s - best) / best < 1e-3
    N0, s0 = BG.split(BG.CostModel(v=1.0, c=2, w=0.0), 1000)
    assert s0 == 1.0 and N0 == 998.0
    assert BG.min_budget(BG.CostModel(v=1.0, c=2, w=0.0)) == 3


def test_every_method_spends_the_same_budget():
    tpl = T.build("qft")
    comp = CP.build(tpl.spec)
    from qemx.noise import PRESETS
    names = ["Unmitigated", "ZNE · Richardson", "PEC · inverse", "CDR · OLS", "Readout · M3", "Readout · TREX"]
    ideal, outs, _ = compare(comp, PRESETS["Depolarizing"], tpl.observable, [preset(k) for k in names], "numpy", 32768)
    _, rows = table_rows(outs, ideal, 32768, 250.0, repeats=30)
    for r in rows:
        assert abs(r["target shots"] + r["training shots"] + r["calibration shots"] - 32768) <= 1, r["method"]
    pec = next(r for r in rows if r["method"].startswith("PEC"))
    assert pec["circuits"] == 32768                     # one sampled circuit per shot (s = 1)


# ------------------------------------------------------------------ hardware parity
def _hw_plan(B=4096):
    from qemx import device as D
    from qemx import hardware as H
    from qemx.compile import Compiled, with_io
    tpl = T.build("bell")
    b = D.fake_backend("fake_perth")
    spec, mm, phys, info = D.compile_for_target(with_io(tpl.spec), b, D.good_target_for(b), "FakePerth")
    comp = Compiled(with_io(tpl.spec), spec, mm, info)
    rc = [preset(k) for k in ("Unmitigated", "ZNE · Richardson", "Readout · M3", "CDR · OLS")]
    pec = preset("PEC · inverse")
    rc.append(Recipe.make("pec", pec.twirl, pec.dd, pec.readout, **dict(pec.p, instances=8)))
    return H.make_plan(comp, tpl.observable, rc, B, tpl.symmetry, 0, b), tpl


def test_hardware_plan_allocates_the_budget_like_the_simulator():
    plan, _ = _hw_plan()
    for i, a in plan.alloc.items():
        reqs = [r for r in plan.requests.values() if r.tag == i]
        tot = sum(r.shots for r in reqs)
        assert tot == plan.shots == a["effective_budget"], (plan.recipes[i].title(), tot)   # exact (review F08)
    tags = {}
    for k, r in plan.requests.items():                  # contenders never share measured counts
        tags.setdefault(r.spec.key(), set()).add(r.tag)
    zne = next(i for i, rc in enumerate(plan.recipes) if rc.estimator == "zne")
    shots = [r.shots for r in plan.requests.values() if r.tag == zne]
    assert len(set(shots)) > 1                            # N_j ∝ |η_j|σ_j, not an even split
    pec = next(i for i, rc in enumerate(plan.recipes) if rc.estimator == "pec")
    # the configured protocol is kept (review F09): one shot per sampled circuit, N_t sampled circuits — not
    # instances = 8 circuits with N_t/8 shots each
    assert plan.alloc[pec]["spc"] == 1
    tgt = [r for r in plan.requests.values() if r.tag == pec and r.role == "target"]
    assert sum(r.multiplicity for r in tgt) == plan.alloc[pec]["target"] == plan.shots


def test_hardware_results_have_the_simulator_structure():
    from qemx import hardware as H
    plan, tpl = _hw_plan()
    ideal = H.HardwareRunner().ideal
    freqs = {k: ideal.run(r.spec, NoiseSpec(), r.mm, ideal=True, backend="numpy").probs
             for k, r in plan.requests.items()}
    res = H.analyze(plan, freqs, 250.0, repeats=20)
    comp = CP.build(tpl.spec)
    ideal_v, outs, _ = compare(comp, NoiseSpec(p2=0.01), tpl.observable, [preset("Unmitigated")], "numpy", 4096)
    _, sim_rows = table_rows(outs, ideal_v, 4096, 250.0, repeats=10)
    assert set(sim_rows[0]) <= set(res["rows"][0])
    assert {"ideal", "outs", "rows", "samples", "shots", "dist", "observable"} <= set(res)
    from qemx import report as RP
    figs, tabs = RP.cross_tab(res)
    assert "cost_precision/rmse_vs_shots" in figs and "cost_precision/cost_for_target" in tabs


# ------------------------------------------------------------------ exports
def test_zip_has_no_html_and_safe_folders():
    import plotly.graph_objects as go
    from qemx.bundle import make_zip, safe_path
    assert safe_path("../../etc/passwd") == "etc/passwd"
    b = make_zip("t", tables={"cost_precision/x": {"a": [1]}}, figures={"comparison/f": go.Figure()}, images=False)
    names = zipfile.ZipFile(io.BytesIO(b)).namelist()
    assert "tables/cost_precision/x.csv" in names and "figures/comparison/f.json" in names
    assert not any(n.endswith((".html", ".js")) for n in names)


# ------------------------------------------------------------------ UI helpers
@pytest.mark.parametrize("text,value", [("pi/4", math.pi / 4), ("\\pi/128", math.pi / 128), ("2pi", 2 * math.pi),
                                        ("3π/4", 3 * math.pi / 4), ("\\frac{\\pi}{8}", math.pi / 8),
                                        ("−π/2", -math.pi / 2), ("Pi/3", math.pi / 3), ("2·π", 2 * math.pi)])
def test_pi_in_angle_inputs(text, value):
    assert abs(eval_expr(text) - value) < 1e-12


def test_angles_are_formatted_as_fractions_of_pi():
    assert fmt_angle(math.pi / 128) == "pi/128"
    assert fmt_angle(-3 * math.pi / 4) == "-3*pi/4"
    assert fmt_angle(0.1234) == "0.1234"
    assert normalize_expr("2pi") == "2*pi"
    with pytest.raises(ValueError):
        eval_expr("__import__('os')")


def test_composer_draws_inputs_and_bases():
    from ui.composer_dnd import io_gates
    prep, post = io_gates(["0", "1", "+"], ["Z", "X", "Y"])
    assert prep[0] == [] and prep[1] == ["X"] and prep[2] == ["H"]
    assert post[0] == [] and post[1] == ["H"] and post[2] == ["S†", "H"]


def test_tooltip_math_is_unicode():
    from ui.tooltips import plain
    assert plain(r"$\mathrm{RMSE}=\sqrt{b^2+v/N}$") == "RMSE = √(b²+v/N)"
    assert plain(r"$\gamma=\sum_i|q_i|$") == "γ = ∑ᵢ|qᵢ|"
    assert "$" not in plain(r"**sd** $\sigma=\sqrt{v/N_t+w/s}$ at $B$")


def test_every_figure_marker_has_its_svg():
    from ui.figures import FIG_DIR, MARK
    docs = [os.path.join(ROOT, "docs", "theory.md")] + [os.path.join(ROOT, "docs", "learn", f)
                                                         for f in os.listdir(os.path.join(ROOT, "docs", "learn"))]
    names = set()
    for p in docs:
        names |= set(MARK.findall(open(p, encoding="utf-8").read()))
    assert {"grover", "hsp", "qft", "shor"} <= names
    for n in names:
        assert os.path.exists(os.path.join(FIG_DIR, n + ".svg")), n


def test_ftqc_category_lists_grover_and_hsp():
    cat = T.CATEGORIES["FTQC · deep algorithms"]
    assert "grover_search" in cat and "hsp" in cat
    assert not re.search(r"colour|behaviour|optimis|analys(e|ing)\b", open(os.path.join(ROOT, "docs", "theory.md"),
                                                                          encoding="utf-8").read())


def test_submission_sends_every_request_with_its_shots():
    from qemx import hardware as H
    from qemx import device as D
    plan, _ = _hw_plan(2048)
    recs = H.submit(plan, D.fake_backend("fake_perth"))
    assert all(len({r["shots"]}) == 1 for r in recs)                  # one shot count per job
    sent = {}
    for r in recs:
        for k, _ in r["pubs"]:
            sent[k] = sent.get(k, 0) + r["shots"]
    assert sent == {k: q.shots for k, q in plan.requests.items()}


@pytest.mark.slow
def test_local_rehearsal_matches_the_full_device_model():
    from qemx import hardware as H
    plan, _ = _hw_plan(4096)
    freqs = H.run_local(plan, "fake_perth", seed=2)
    assert set(freqs) == set(plan.requests)
    raw = next(k for k, r in plan.requests.items() if r.tag == 0)
    assert 0.80 < freqs[raw][0] + freqs[raw][3] < 1.0                 # Bell: mostly 00 and 11, some noise


def test_diagrams_are_sized_by_their_text_and_captions_live_outside():
    import json
    from ui import figures as FG
    caps = json.load(open(os.path.join(FG.FIG_DIR, "captions.json"), encoding="utf-8"))
    assert set(caps) == set(FG.names())
    assert caps["grover"] and "fires only" in caps["grover"] and "\\ket" not in caps["grover"]
    for n in FG.names():                                   # no caption line inside a picture any more
        tex = open(os.path.join(FG.FIG_DIR, n + ".tex"), encoding="utf-8").read()
        assert "\\small $P(m)" not in tex and not re.search(r"\\\\\[\dpt\]\s*\n\\small", tex), n
    assert "object-fit:contain" in FG.CSS and FG.SCALE >= 1.8    # Streamlit's scale-down would never enlarge
