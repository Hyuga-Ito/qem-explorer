"""1.6.0 policies, evaluation conventions, certificates and figure/data consistency, on synthetic decision datasets
(no simulation: every number is constructed, so each convention can be checked exactly)."""
import math
import os

import numpy as np
import pandas as pd
import pytest

from qemx.selection import certify as CE
from qemx.selection import evaluate as EV
from qemx.selection import policies as PO
from qemx.selection.dataset import Dataset
from qemx.selection.models import EPS0

PROTOCOLS = ["RAW", "RO", "ZNE-L", "ZNE-R", "ZNE-E", "PEC", "CDR", "NIL"]


def synthetic(G=24, budgets=(1000, 100000, 10000000), seed=0, noise=0.05, equal=False):
    """G base circuits × 2 variants × budgets. True log10 risk: RAW flat (bias floor), mitigations fall with B at
    different rates and floors, so the best protocol depends on (n, B). Labels: two independent noisy halves."""
    rng = np.random.default_rng(seed)
    rows, ys, ye, se, F = [], [], [], [], []
    for g in range(G):
        n = 3 + g % 4
        lam = 0.01 * (1 + g % 5)
        for v in range(2):
            for B in budgets:
                lb = math.log10(B)
                if equal:
                    true = np.full(len(PROTOCOLS), -3.0)
                else:
                    true = np.array([-2.5 + 0.0 * lb, -3.0, -1.0 - 0.6 * lb + 10 * lam, -0.5 - 0.6 * lb + 5 * lam,
                                     -0.8 - 0.6 * lb + 8 * lam, 0.2 - 0.9 * lb + 30 * lam, -4.5 + 20 * lam - 0.05 * lb,
                                     -4.0 + 0.1 * n])
                    true = np.maximum(true, -6.5)
                feas = np.ones(len(PROTOCOLS), bool)
                if B < 10000:
                    feas[-2:] = False
                ys.append(np.where(feas, 10 ** (true + rng.normal(0, noise, len(true))), np.nan))
                ye.append(np.where(feas, 10 ** (true + rng.normal(0, noise, len(true))), np.nan))
                se.append(np.where(feas, 0.05 * 10 ** true, np.nan))
                F.append(feas)
                rows.append(dict(task_id="t%d_%d" % (g, v), base_id="b%d" % g, family="f%d" % (g % 3), n=n,
                                 device="d%d" % (v % 2), B=int(B), ideal=0.5,
                                 feats=dict(n=n, two_qubit=2 * n, depth=10 * n, lam_lc=lam, obs_weight=1 + v,
                                            lc_qubits=n, lc_gates=5 * n, lc_2q=n, lc_locations=6 * n)))
    ys, ye, se, F = map(np.array, (ys, ye, se, F))
    return Dataset(PROTOCOLS, list(budgets), rows, ys, ye, se, F, ye.copy(), np.zeros_like(ys))


def test_sbs_is_chosen_on_training_rows_only():
    ds = synthetic()
    tr = np.arange(0, len(ds.rows) // 2)
    te = np.arange(len(ds.rows) // 2, len(ds.rows))
    pol = PO.SBS().fit(ds, tr)
    ds2 = synthetic()
    ds2.Y_sel[te] = 1.0                                              # test labels changed: the SBS must not move
    assert PO.SBS().fit(ds2, tr).rank == pol.rank


def test_reference_and_gap_closed_conventions():
    ds = synthetic(noise=0.0)
    tr, te = EV.splits(ds, "same-distribution", k=3)[0][1:]
    ev = EV.evaluate(ds, tr, te, [PO.SBS(), PO.AllCandidate(), PO.Raw()], reps=200)
    s = {r["policy"]: r for r in ev["summary"] if r["B"] == "all"}
    assert s["SBS"]["gap_closed"] == pytest.approx(0.0) and s["ALL-CANDIDATE"]["gap_closed"] == pytest.approx(1.0)
    assert np.allclose(ev["per_policy"]["ALL-CANDIDATE"]["log_ratio"], 0.0)       # noise-free: oracle has no regret
    assert s["RAW"]["gap_closed"] < 0


def test_choosing_on_noisy_risks_is_optimistic_and_independent_scoring_removes_it():
    """Equal true risks: the minimum of noisy selection-half labels is biased low, but the chosen protocol scored on
    the independent half has no regret on average (optimizer's curse removed)."""
    ds = synthetic(G=60, noise=0.2, equal=True, seed=3)
    idx = np.arange(len(ds.rows))
    ch, _ = PO.AllCandidate().choose(ds, idx)
    sel_min = np.log10(np.array([ds.Y_sel[i, j] for i, j in zip(idx, ch)]))
    ev_val = np.log10(np.array([ds.Y_eval[i, j] for i, j in zip(idx, ch)]))
    assert sel_min.mean() < -3.0 - 0.15                                # optimistic
    assert abs(ev_val.mean() - (-3.0)) < 0.03                        # unbiased


def test_group_bootstrap_ignores_duplicated_rows():
    v = {"a": np.array([1.0, 2.0, 3.0, 4.0])}
    g = np.array(["x", "y", "z", "w"])
    ci1 = EV.bootstrap(v, g, 500, 1)[0]["a"]
    v2 = {"a": np.repeat(v["a"], 5)}
    ci2 = EV.bootstrap(v2, np.repeat(g, 5), 500, 1)[0]["a"]
    assert ci1 == pytest.approx(ci2)


def test_splits_keep_base_circuits_together_and_hold_out_what_they_name():
    ds = synthetic()
    for _, tr, te in EV.splits(ds, "same-distribution", k=4):
        assert not set(ds.groups[tr]) & set(ds.groups[te])
    for name, tr, te in EV.splits(ds, "family"):
        assert set(ds.col("family")[te]) == {name.split("=")[1]} and name.split("=")[1] not in set(ds.col("family")[tr])
    for kind, kw in (("budget-extrap", dict(max_train_B=100000)), ("budget-interp", {}),
                     ("device", dict(test_device="d1")), ("width", dict(min_test_n=6))):
        parts = EV.splits(ds, kind, k=4, **kw)
        assert parts
        for _, tr, te in parts:                       # no base circuit on both sides of any split
            assert not set(ds.groups[tr]) & set(ds.groups[te])
        tested = np.concatenate([te for _, _, te in parts])
        assert len(tested) == len(set(tested.tolist()))              # every test row once
    for _, tr, te in EV.splits(ds, "budget-extrap", k=4, max_train_B=100000):
        assert ds.col("B")[tr].max() <= 100000 < ds.col("B")[te].min()
    for _, tr, te in EV.splits(ds, "device", k=4, test_device="d1"):
        assert set(ds.col("device")[te]) == {"d1"} and "d1" not in set(ds.col("device")[tr])


def test_models_never_choose_an_infeasible_protocol_and_abstention_falls_back():
    ds = synthetic(noise=0.1)
    tr, te = EV.splits(ds, "same-distribution", k=3)[0][1:]
    names = ("n", "two_qubit", "lam_lc", "log10_budget")
    for pol in (PO.Model("ridge", names, "t"), PO.ModelAbstain("ridge", names, "t", alpha=0.2), PO.SizeBudget()):
        pol.fit(ds, tr)
        ch, acc = pol.choose(ds, te)
        assert all(ds.F[i, j] for i, j in zip(te, ch))
        if pol.abstains:
            fb = pol.fallback
            for i, j, a in zip(te, ch, acc):
                if not a:
                    assert j == PO._first_feasible(fb.rank_for(ds.rows[i]["B"]), ds.F[i])


def test_oracle_features_are_refused_as_model_inputs():
    from qemx.selection import features as FE
    with pytest.raises(ValueError):
        FE.matrix([dict(ideal=0.3)], ["ideal"])
    assert all(FE.BY_NAME[n].info != "oracle" for s in FE.SETS.values() for n in s)


# ------------------------------------------------------------------ certificates
def test_hoeffding_intervals_hold_simultaneously_at_the_stated_level():
    rng = np.random.default_rng(0)
    K, r, H = 4, 200, 4.0
    true = np.array([0.5, 1.0, 1.5, 2.0])
    miss = 0
    for _ in range(400):
        losses = np.clip(true[:, None] + rng.uniform(-0.5, 0.5, (K, r)), 0, H)
        L, U = CE.simultaneous(losses, 0.05)
        miss += not np.all((L <= true) & (true <= U))
    assert miss / 400 <= 0.05
    assert CE.hoeffding_t(200, 4, 0.05) == pytest.approx(4 * math.sqrt(math.log(160) / 400))
    assert CE.r_needed_hoeffding(CE.hoeffding_t(200, 4, 0.05), 4, 0.05) == pytest.approx(200)


def test_certificate_uses_the_declared_clipped_variant_and_keeps_the_unclipped_loss():
    est = np.array([[1.8, 1.8, 1.8], [0.5, 0.5, 0.5]])             # an extrapolation beyond [-1, 1], and RAW
    c = CE.certify(est, 0.9, raw_index=1)
    assert c["chosen"] == 0                                            # the clipped variant looks best ...
    assert c["mse_clipped"] == pytest.approx(0.01)
    assert c["mse_unclipped"] == pytest.approx(0.81)                   # ... the unclipped estimator is not
    assert not c["informative"]                                        # 3 trials: nothing is certified


def test_two_epsilon_regret_and_reference_transform():
    r = CE.regret_two_eps([1.0, 1.1, 3.0], [1.2, 0.95, 3.0])
    assert r["eps"] == pytest.approx(0.2) and r["regret"] == pytest.approx(0.25 - 0.0) and r["holds"]
    assert r["bound"] == pytest.approx(0.4)
    lo, hi = CE.reference_transform(0.04, 0.09, 0.05)
    assert lo == pytest.approx(0.15 ** 2) and hi == pytest.approx(0.35 ** 2)
    assert CE.reference_transform(0.0001, 0.01, 0.05)[0] == 0.0


# ------------------------------------------------------------------ figures from saved tables
def _saved_run(tmp_path, G=6, budgets=(1000, 100000), seed=3):
    """A run directory holding only the saved tables a figure reads: per-row choices of four policies (two variants
    per base circuit) and the per-protocol evaluation-half MSE labels."""
    rng = np.random.default_rng(seed)
    adir = tmp_path / "run" / "analysis"
    adir.mkdir(parents=True)
    ch, lab = [], []
    for g in range(G):
        for v in range(2):
            t = "t%d_%d" % (g, v)
            for B in budgets:
                m = {p: float(10 ** rng.normal(-4, 0.5)) for p in ("RAW", "RO", "PEC")}
                ref = min(m, key=m.get)
                ch.append({"task_id": t, "base_id": "b%d" % g, "family": "ising" if g % 2 else "ghz", "n": 3,
                           "device": "fake_fez", "B": B, "part": "fold %d" % (g % 2),
                           "RAW|choice": "RAW", "SBS|choice": "RO", "SBS-B|choice": "PEC" if B > 1000 else "RO",
                           "ALL-CANDIDATE|choice": ref})
                lab += [dict(task_id=t, B=B, protocol=p, mse_eval=x, feasible=True) for p, x in m.items()]
    pd.DataFrame(ch).to_csv(adir / "choices_same-distribution.csv", index=False, encoding="utf-8")
    pd.DataFrame(lab).to_csv(adir / "labels.csv", index=False, encoding="utf-8")
    return str(tmp_path / "run"), pd.DataFrame(ch), pd.DataFrame(lab)


def test_figure_one_source_data_equals_base_circuit_weighted_risks_recomputed_from_saved_tables(tmp_path):
    from qemx.selection import plots as PL
    run, ch, lab = _saved_run(tmp_path)
    cfg = PL.load_config()
    cfg["regret_profile"]["budgets"] = {"low": 1000, "medium": 100000, "high": 10000000}
    out = PL.fig01_budget_efficiency(run, str(tmp_path / "fig"), cfg)
    for ext in ("pdf", "svg", "png"):
        assert os.path.getsize(out[ext]) > 1000
    curves = pd.read_csv(out["data:curves"], encoding="utf-8").set_index(["policy", "B"])
    look = lab.set_index(["task_id", "B", "protocol"]).mse_eval
    for pol, col in (("RAW", "RAW|choice"), ("SBS", "SBS|choice"), ("SBS-B", "SBS-B|choice")):
        for B in (1000, 100000):
            e = ch[ch.B == B]
            risk = np.array([look[(t, B, c)] for t, c in zip(e.task_id, e[col])])
            want = pd.Series(risk).groupby(e.base_id.to_numpy()).mean().mean()   # variants first, then circuits
            assert curves.loc[(pol, B), "mean_mse"] == pytest.approx(want, rel=1e-12)
            assert curves.loc[(pol, B), "base_circuits"] == 6
    prof = pd.read_csv(out["data:profiles"], encoding="utf-8")
    assert set(prof.condition) == {"low", "medium"}                     # 10^7 is not in the grid: nothing merged
    top = prof[prof.tolerance == prof.tolerance.max()]
    assert (top.fraction_within <= 1 + 1e-12).all() and (top.fraction_within >= 0.99).all()
    md = open(str(tmp_path / "fig" / "fig01_budget_efficiency.md"), encoding="utf-8").read()
    assert "Simulated observations" in md and "**Configuration.**" in md


def test_figure_profiles_count_negative_regret_within_every_tolerance(tmp_path):
    """The reference is estimated on the other label half, so a policy can beat it: such rows stay negative (never
    clipped) and count as within every tolerance of the performance profile."""
    from qemx.selection import plots as PL
    run, ch, lab = _saved_run(tmp_path, seed=11)
    rows = PL.policy_rows(run)
    sbs = rows[rows.policy == "SBS"]
    assert (sbs.regret < 0).any() or (sbs.regret == 0).any()
    out = PL.fig01_budget_efficiency(run, str(tmp_path / "fig"), PL.load_config())
    prof = pd.read_csv(out["data:profiles"], encoding="utf-8")
    low = prof[(prof.policy == "SBS") & (prof.condition == "low")]
    e = sbs[sbs.B == 1000]
    w = 1.0 / e.groupby("base_id").task_id.transform("count") / e.base_id.nunique()
    assert low.fraction_negative_regret.iloc[0] == pytest.approx(float(w[e.regret < 0].sum()))
    assert low.fraction_within.min() >= low.fraction_negative_regret.iloc[0] - 1e-12


def test_palettes_separate_their_colors_under_color_vision_deficiency():
    from qemx.selection import figures as FG
    from qemx.selection import protocols as PR
    for colors in ([FG.PROTOCOL_COLORS[k] for k in PR.PORTFOLIO_16], [c for c, _, _ in FG.POLICY_STYLE.values()]):
        rep = FG.palette_report(colors)
        assert rep["normal"] >= 15                                     # full-color readers
        assert min(rep["protan"], rep["deutan"], rep["tritan"]) >= 6   # with markers as secondary encoding


def test_the_17_candidate_palette_is_a_composite_encoding():
    """1.8.0: all 17 candidate colours stay apart (≥ 10 normal, ≥ 6 under every simulated deficiency) and every
    candidate has its own marker, so colour is never the only identity; the eight 1.6/1.7 colours are unchanged."""
    from qemx.selection import figures as FG
    from qemx.selection import protocols as PR
    keys = list(PR.PORTFOLIO_18)
    assert set(keys) <= set(FG.PROTOCOL_COLORS) and set(keys) <= set(FG.PROTOCOL_MARKERS)
    rep = FG.palette_report([FG.PROTOCOL_COLORS[k] for k in keys])
    assert rep["normal"] >= 10 and min(rep["protan"], rep["deutan"], rep["tritan"]) >= 6, rep
    assert len({FG.PROTOCOL_MARKERS[k] for k in keys}) == len(keys)
    assert [FG.PROTOCOL_COLORS[k] for k in PR.PORTFOLIO_16] == ["#C8C8C8", "#E69F00", "#56B4E9", "#0072B2",
                                                                 "#003F63", "#D55E00", "#009E73", "#CC79A7"]


def _choices_and_labels(ds, parts):
    """Saved-table form (choices per test row, labels per task × budget × protocol) of K-fold evaluations."""
    rows = []
    pols = [PO.Raw(), PO.SBS(), PO.SBSBudget(), PO.AllCandidate()]
    summaries = []
    for name, tr, te in parts:
        ev = EV.evaluate(ds, tr, te, pols, reps=100)
        rows += [dict(part=name, **r) for r in ev["rows"]]
        summaries += [dict(part=name, **s) for s in ev["summary"]]
    lab = []
    for i, r in enumerate(ds.rows):
        for j, p in enumerate(ds.protocols):
            lab.append(dict(task_id=r["task_id"], B=r["B"], protocol=p, mse_eval=ds.Y_eval[i, j]))
    return pd.DataFrame(rows), pd.DataFrame(lab), pd.DataFrame(summaries)


def test_pooled_evaluation_from_saved_tables_reproduces_each_part_and_pools_all_test_circuits():
    ds = synthetic(G=20, noise=0.1, seed=4)
    parts = EV.splits(ds, "same-distribution", k=4)
    ch, lab, summ = _choices_and_labels(ds, parts)
    for name, d in ch.groupby("part"):                     # the saved tables reproduce every part's summary exactly
        got = {x["policy"]: x["mean_log10_risk"] for x in EV.pooled(d.reset_index(drop=True), lab, 50)
               if x["B"] == "all"}
        want = summ[(summ.part == name) & (summ.B.astype(str) == "all")].set_index("policy").mean_log10_risk
        assert all(abs(got[p] - want[p]) < 1e-12 for p in got)
    pooled = {x["policy"]: x for x in EV.pooled(ch, lab, 200) if x["B"] == "all"}
    assert pooled["SBS"]["base_circuits"] == 20 and pooled["SBS"]["diff_vs_sbs"] == 0.0
    assert pooled["ALL-CANDIDATE"]["gap_closed"] == pytest.approx(1.0)


def test_bias_corrected_cv_removes_the_optimism_of_picking_the_best_of_equal_configurations():
    """Ten configurations whose per-circuit scores are SBS plus independent noise of mean 0: the naive winner looks
    better than SBS by construction; BBC-CV's out-of-bag estimate of the selection procedure does not."""
    rng = np.random.default_rng(7)
    G = 40
    base = rng.normal(-3.0, 0.3, G)
    lab, rows = [], []
    for g in range(G):
        rows.append({"task_id": "t%d" % g, "base_id": "b%d" % g, "B": 1000, "SBS|choice": "SBS",
                     "ALL-CANDIDATE|choice": "SBS"})
        lab.append(dict(task_id="t%d" % g, B=1000, protocol="SBS", mse_eval=10 ** base[g] - EPS0 + 1e-12))
        for c in range(10):
            p = "C%d" % c
            rows[-1]["MODEL[%s]|choice" % p] = p
            lab.append(dict(task_id="t%d" % g, B=1000, protocol=p,
                            mse_eval=10 ** (base[g] + rng.normal(0, 0.2)) - EPS0 + 1e-12))
    out = EV.bbc_cv(pd.DataFrame(rows), pd.DataFrame(lab), reps=1000, seed=1)
    assert out["naive_diff_vs_ref"] < -0.03                         # optimistic winner
    assert out["optimism"] > 0.02 and abs(out["bbc_diff_vs_ref"]) < 0.03


def test_mismatch_figure_differences_equal_the_policy_table(tmp_path):
    """Figure 6 (1.7.0 tables): panel C's equal-budget differences equal the paired differences recomputed from the
    policy table, panel A's regret equals the mean log ratio of TARGET@B to BEST, and the notes state the counts and
    the false-positive rate."""
    from qemx.selection import plots as PL
    rng = np.random.default_rng(2)
    diag, pol, draws = [], [], []
    offs = {"TARGET@B": 0.0, "NO-DIAG@B-d": 0.01, "FIXED→ZNE-L": 0.12, "FIXED→SBS-B": 0.05, "DIAG-RESELECT": -0.1,
            "RANDOM→reaction": -0.02, "BEST": -0.3}
    for scn, typ in (("none", "none"), ("drift-2", "drift")):
        for b in range(5):
            for B in (1000, 100000):
                diag.append(dict(scenario=scn, type=typ, task_id="t%d%s" % (b, scn), base_id="b%d" % b, B=B, d=200,
                                 p_true=0.9, p_sup=0.9, threshold=1.96, alpha=0.05,
                                 flag_rate=0.05 if scn == "none" else 0.8, z_q10=-1.0, z_median=0.0, z_q90=1.0,
                                 model_log_regret=0.3, model_choice="PEC"))
                base = rng.normal(-4, 0.3)
                for name, off in offs.items():
                    pol.append(dict(scenario=scn, type=typ, task_id="t%d%s" % (b, scn), base_id="b%d" % b,
                                    family="ghz", B=B, policy=name, risk=10 ** (base + off),
                                    mean_log10_risk=base + off, model_choice="PEC", p_flag=0.5, diag_shots=200,
                                    reaction="ZNE-L"))
                for k in range(3):
                    draws.append(dict(scenario=scn, type=typ, task_id="t%d%s" % (b, scn), B=B, draw=k,
                                      abs_z=float(rng.uniform(0, 4)), flagged=False, model_log_regret=0.1 * k))
    m = tmp_path / "run" / "mismatch"
    (m / "analysis").mkdir(parents=True)
    pd.DataFrame(diag).to_csv(m / "diagnostics.csv", index=False, encoding="utf-8")
    pd.DataFrame(pol).to_csv(m / "policies.csv", index=False, encoding="utf-8")
    pd.DataFrame(draws).to_csv(m / "analysis" / "deployment_draws.csv", index=False, encoding="utf-8")
    pd.DataFrame([dict(B=B, deployments=30, failures=10, auc_abs_z=0.6, flag_rate_failures=0.5, flag_rate_ok=0.2)
                  for B in (1000, 100000)]).to_csv(m / "analysis" / "discrimination.csv", index=False,
                                                   encoding="utf-8")
    pd.DataFrame([dict(held_out_type=t, B=B, reaction="ZNE-L") for t in ("none", "drift") for B in (1000, 100000)]
                 ).to_csv(m / "analysis" / "reactions.csv", index=False, encoding="utf-8")
    cfg = PL.load_config()
    cfg["mismatch"]["budgets"] = [1000, 100000]
    out = PL.fig06_mismatch_diagnostics(str(tmp_path / "run"), str(tmp_path / "fig"), cfg)
    eq = pd.read_csv(out["data:equal_budget"], encoding="utf-8").set_index(["type", "policy"]).diff_log10
    for (typ, name), off in [((t, n), o) for t in ("none", "drift") for n, o in offs.items()
                             if n not in ("TARGET@B", "BEST")]:
        assert eq[(typ, name)] == pytest.approx(off, abs=1e-9)          # paired, every budget pooled
    resp = pd.read_csv(out["data:response"], encoding="utf-8")
    r = resp[(resp.facet == "drift") & (resp.scenario == "drift-2")]
    P = pd.DataFrame(pol)
    for B in (1000, 100000):
        q = P[(P.scenario == "drift-2") & (P.B == B)].pivot_table(index="base_id", columns="policy", values="risk")
        want = float(np.mean(np.log10((q["TARGET@B"] + EPS0) / (q["BEST"] + EPS0))))
        assert r[r.B == B].mean_log_regret.iloc[0] == pytest.approx(want, rel=1e-12)
    md = open(str(tmp_path / "fig" / "fig06_mismatch_diagnostics.md"), encoding="utf-8").read()
    assert "5 base circuits × 2 execution laws" in md and "5.0 %" in md
