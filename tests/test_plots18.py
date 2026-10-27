"""1.8.0 supplementary figures (qemx.selection.plots18) and thesis tables (qemx.selection.thesis_tables): drawn from
a small synthetic run directory; the plotted / tabulated numbers are recomputed from the source tables."""
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

KEYS = ["RAW", "RO", "ZNE-L", "ZNE-R", "ZNE-E", "PEC", "CDR", "NIL", "RO-TREX", "RO-TW", "RO-DD", "ZNE-L-TW",
        "ZNE-E-B", "PEC-TW", "FR", "SV", "ZNE-L∘SV"]
BUDGETS = [1000, 100000, 10000000]


def _labels(rng, tasks):
    rows = []
    for tid, base in tasks:
        raw = {B: 10 ** rng.uniform(-4, -2) for B in BUDGETS}
        for k in KEYS:
            for B in BUDGETS:
                feas = not (k in ("CDR", "NIL", "FR") and B == 1000) and not (k in ("SV", "ZNE-L∘SV")
                                                                              and base != "b0")
                m = raw[B] if k == "RAW" else raw[B] * 10 ** rng.normal(-0.2, 0.6)
                if k == "NIL" and B == 10000000:
                    m = raw[B] * 30                          # a severe degradation by construction
                rows.append(dict(task_id=tid, base_id=base, family="hea", n=4, device="fake_fez", protocol=k, B=B,
                                 feasible=feas, mse_eval=m if feas else np.nan, fail_any_eval=0.05 if feas else np.nan,
                                 nonresponse_eval=0.0, shots_calibration=0.1 * B, shots_training=0.25 * B
                                 if k in ("CDR", "NIL", "FR") else 0.0, shots_execution=0.65 * B))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from qemx import textio as TX
    rng = np.random.default_rng(3)
    d = tmp_path_factory.mktemp("run18")
    tasks = [("t%d" % i, "b%d" % (i // 2)) for i in range(8)]
    for sub in ("analysis", "portfolio", "audit18/analysis", "confirm", "replay"):
        os.makedirs(d / sub, exist_ok=True)
    lab = _labels(rng, tasks)
    lab.to_csv(d / "analysis" / "labels.csv", index=False, encoding="utf-8")
    dd = [dict(task_id=t, protocol=k, B=B, half=h, design=j, mean_loss=float(rng.exponential(1e-3)))
          for t, _ in tasks for k in ("CDR", "NIL") for B in BUDGETS[1:] for h in (0, 1) for j in range(6)]
    pd.DataFrame(dd).to_csv(d / "analysis" / "design_diagnostics.csv", index=False, encoding="utf-8")
    # portfolio comparison
    con = [dict(contrast=c, a="x", b="y", diff=v, lo=v - 0.05, hi=v + 0.05, ratio_geomean_mse_plus_floor=10 ** v)
           for c, v in (("expansion_fixed_policy", -0.10), ("expansion_attainable", -0.12),
                        ("learning_within_1.6_hgb", -0.02), ("learning_within_1.8_hgb", -0.04),
                        ("total_hgb", -0.14))]
    pd.DataFrame(con).to_csv(d / "portfolio" / "portfolio_contrasts.csv", index=False, encoding="utf-8")
    pol = [dict(policy="%s|%s" % (p, pf), mean_log10=-4 - 0.1 * i, mse=1e-4, p_severe_abs=0.0, q95_risk=1e-3,
                worst5_loss_share=0.2, p_severe_vs_raw=0.01 * i)
           for i, (p, pf) in enumerate((p, pf) for p in ("SBS-B", "MODEL[hgb]", "ALL-CANDIDATE") for pf in ("1.6",
                                                                                                         "1.8"))]
    pd.DataFrame(pol).to_csv(d / "portfolio" / "portfolio_policies.csv", index=False, encoding="utf-8")
    TX.write_json(str(d / "portfolio" / "portfolio_comparison.json"), dict(
        rows=24, base_circuits=4, feature_set="deployable18", metric="m", portfolios={"1.6": KEYS[:8], "1.8": KEYS},
        policies={p["policy"]: dict(chosen={"RO": 10, "PEC-TW": 14}) for p in pol}))
    # root audit
    st = [dict(task_id=t, protocol=k, B=B, roots_feasible=4, var_ratio=float(rng.lognormal(0, 1)))
          for t, _ in tasks[:4] for k in KEYS[:6] for B in (10000, 300000, 10000000)]
    pd.DataFrame(st).to_csv(d / "audit18" / "analysis" / "label_stability.csv", index=False, encoding="utf-8")
    rk = [dict(portfolio=pf, task_id=t, B=B, distinct_winners=int(rng.integers(1, 4)), mean_cross_root_log_regret=0.1)
          for pf in ("1.8", "1.6") for t, _ in tasks[:4] for B in (10000, 300000, 10000000)]
    pd.DataFrame(rk).to_csv(d / "audit18" / "analysis" / "ranking_stability.csv", index=False, encoding="utf-8")
    ct = [dict(task_id=t, root=r, protocol=k, B=10000, preflight="verified-feasible", labels_feasible=True,
               agree=True) for t, _ in tasks[:4] for r in (1, 2) for k in ("CDR", "NIL", "FR")]
    pd.DataFrame(ct).to_csv(d / "audit18" / "analysis" / "contract.csv", index=False, encoding="utf-8")
    TX.write_json(str(d / "audit18" / "analysis" / "audit_summary.json"), dict(
        roots=[1, 2, 3, 4], budgets=[10000, 300000, 10000000], tasks=4, ranking_cells=12, cells_with_one_winner=5,
        mean_cross_root_log_regret=0.1, contract_agreement=1.0, var_ratio_median_by_protocol={"RO": 1.1},
        policy_by_root={"1": dict(mean_log10=-4.0)}))
    # confirmation and its replay
    clab = lab.copy()
    clab.to_csv(d / "confirm" / "labels.csv", index=False, encoding="utf-8")
    ch = pd.DataFrame([{"task_id": t, "B": B, "SBS-B|choice": "RO", "MODEL[hgb|deployable18]|choice": "ZNE-L"}
                       for t, _ in tasks for B in BUDGETS])
    ch.to_csv(d / "confirm" / "choices_confirm.csv", index=False, encoding="utf-8")
    met = [dict(policy=p, B="all", rows=24, base_circuits=4, mean_log10_risk=v, ci_lo=v - 0.1, ci_hi=v + 0.1,
                policy_mse=1e-4, diff_vs_sbsb=v + 4, diff_vs_sbsb_lo=v + 3.9, diff_vs_sbsb_hi=v + 4.1,
                p_severe_vs_raw=0.0, q95_risk=1e-3)
           for p, v in (("RAW", -3.0), ("SBS", -3.9), ("SBS-B", -4.0), ("SIZE-B", -4.0),
                        ("MODEL[hgb|deployable18]", -4.1), ("ALL-CANDIDATE", -4.3))]
    pd.DataFrame(met).to_csv(d / "confirm" / "metrics_confirm.csv", index=False, encoding="utf-8")
    risk = {(r.task_id, r.protocol, r.B): r.mse_eval for r in clab.itertuples()}
    summ = {}
    for name, acc in (("qemx-rec-1.8.0-deployable", True), ("qemx-rec-1.8.0", False)):
        rows = []
        for i, (t, base) in enumerate(tasks):
            for B in BUDGETS:
                st_ = "out-of-distribution" if (i == 0 and B == 1000) else "ok"
                a_ = acc and st_ == "ok"
                c = "ZNE-L" if a_ else "RO"
                rows.append(dict(task_id=t, base_id=base, family="hea", n=4, B=B, status=st_, accepted=a_, choice=c,
                                 model_choice="ZNE-L", fallback="RO", ood=int(st_ != "ok"), margin=0.5,
                                 skipped="CDR:infeasible" if (a_ and B == 1000) else "",
                                 choice_preflight="verified-feasible", choice_level="structural", scored=True,
                                 risk=risk[(t, c, B)], seconds=0.1))
        pd.DataFrame(rows).to_csv(d / "replay" / ("replay_%s.csv" % name), index=False, encoding="utf-8")
        s = dict(rows=len(rows), base_circuits=4, unscored=0, diff=-0.05, diff_lo=-0.08, diff_hi=-0.01,
                 policy_mse=1e-4, model_component_configuration="MODEL[hgb|deployable18]",
                 model_component_benchmark=dict(diff=-0.07, diff_lo=-0.1, diff_hi=-0.03), policy="p/2")
        TX.write_json(str(d / "replay" / ("replay_%s.json" % name)), s)
        summ[name] = {k: s[k] for k in ("rows", "unscored", "diff", "diff_lo", "diff_hi")}
    TX.write_json(str(d / "replay" / "replay_summary.json"), summ)
    return str(d)


def test_outcome_categories_and_their_palette():
    from qemx.selection import figures as FG
    from qemx.selection import plots18 as P
    assert P.outcome(dict(status="unsupported")) == "unsupported"
    assert P.outcome(dict(status="out-of-distribution", accepted=False)) == "ood-fallback"
    assert P.outcome(dict(status="no-model", accepted=False)) == "no-model"
    assert P.outcome(dict(status="ok", accepted="False")) == "margin-fallback"
    assert P.outcome(dict(status="ok", accepted=True, skipped="CDR:infeasible")) == "accepted-after-skip"
    assert P.outcome(dict(status="ok", accepted=True, skipped=float("nan"))) == "accepted"
    rep = FG.palette_report([c for _, _, c in P.OUTCOMES])
    assert rep["normal"] >= 15 and min(rep["protan"], rep["deutan"], rep["tritan"]) >= 8, rep


def test_new_figures_draw_and_their_numbers_match_the_sources(run):
    from qemx.selection import plots as PL
    from qemx.selection import plots18 as P
    fdir = os.path.join(run, "figures")
    cfg = PL.load_config(None)
    for name in P.NEED_18:
        getattr(P, name)(run, fdir, cfg)
        for ext in ("pdf", "svg", "png", "md"):
            assert os.path.exists(os.path.join(fdir, "%s.%s" % (name, ext))), (name, ext)
        assert name in PL.FIG_META and name in PL.FIG_METHODS
    # deployment: outcome counts and the plotted full-policy difference
    oc = pd.read_csv(os.path.join(fdir, "supp_deployment_policy__outcomes.csv"), encoding="utf-8")
    dep = oc[oc.bundle == "qemx-rec-1.8.0-deployable"].set_index("outcome").rows
    assert dep["ood-fallback"] == 1 and dep["accepted-after-skip"] == 7 and dep.sum() == 24
    sim = oc[oc.bundle == "qemx-rec-1.8.0"].set_index("outcome").rows
    assert sim["margin-fallback"] == 23
    cmp_ = pd.read_csv(os.path.join(fdir, "supp_deployment_policy__comparison.csv"), encoding="utf-8").set_index("kind")
    assert math.isclose(cmp_.loc["POLICY|qemx-rec-1.8.0-deployable", "diff"], -0.05)
    assert math.isclose(cmp_.loc["MODEL-COMPONENT|qemx-rec-1.8.0-deployable", "diff"], -0.07)
    rb = pd.read_csv(os.path.join(fdir, "supp_root_audit__ranking.csv"), dtype={"portfolio": str}, encoding="utf-8")
    assert set(rb.portfolio) == {"1.8", "1.6"} and rb.groupby("portfolio").cells.sum().eq(12).all()
    # tails: severe rate of NIL at 10^7 is 1 (constructed), RAW's is 0
    sv = pd.read_csv(os.path.join(fdir, "supp_tails__severe.csv"), encoding="utf-8")
    assert sv[(sv.protocol == "NIL") & (sv.B == 10000000)].p_severe_vs_raw.iloc[0] == 1.0
    assert sv[(sv.protocol == "RAW")].p_severe_vs_raw.max() == 0.0
    assert sv[(sv.protocol == "CDR") & (sv.B == 1000)].p_severe_vs_raw.isna().all()      # not feasible: no value
    # portfolio contrasts are the source table's
    pc = pd.read_csv(os.path.join(fdir, "supp_portfolio_expansion__contrasts.csv"), encoding="utf-8")
    src = pd.read_csv(os.path.join(run, "portfolio", "portfolio_contrasts.csv"), encoding="utf-8")
    assert np.allclose(pc.set_index("contrast").loc[src.contrast, "diff"], src["diff"])


def test_thesis_tables_agree_with_their_sources(run):
    from qemx.selection import thesis_tables as TT
    out = os.path.join(run, "thesis_tables")
    r = TT.build(run, out, compile_pdf=False)
    assert {"t2_candidates", "t3_confirmation", "t4_per_budget", "t5_tails", "t7_portfolio"} <= set(r["made"])
    t3 = pd.read_csv(os.path.join(out, "t3_confirmation.csv"), encoding="utf-8")
    met = pd.read_csv(os.path.join(run, "confirm", "metrics_confirm.csv"), encoding="utf-8").set_index("policy")
    for row in t3[~t3.policy.str.startswith("POLICY|")].itertuples():
        assert math.isclose(row.mean_log10, met.loc[row.policy, "mean_log10_risk"])
    full = t3[t3.policy == "POLICY|qemx-rec-1.8.0-deployable"].iloc[0]
    assert math.isclose(full["diff"], -0.05)
    rp = pd.read_csv(os.path.join(run, "replay", "replay_qemx-rec-1.8.0-deployable.csv"), encoding="utf-8")
    per = rp.groupby("base_id").risk.apply(lambda v: np.mean(np.log10(v + 1e-6)))
    assert math.isclose(full["mean_log10"], per.mean())
    t4 = pd.read_csv(os.path.join(run, "thesis_tables", "t4_per_budget.csv"), encoding="utf-8").set_index("protocol")
    lab = pd.read_csv(os.path.join(run, "analysis", "labels.csv"), encoding="utf-8")
    q = lab[(lab.protocol == "PEC") & (lab.B == 100000) & lab.feasible]
    assert math.isclose(t4.loc["PEC", "median_mse_100000"], float(np.median(q.mse_eval)))
    assert t4.loc["CDR", "feasible_1000"] == 0
    tex = open(os.path.join(out, "t3_confirmation.tex"), encoding="utf-8").read()
    assert "\\toprule" in tex and "tab:qem180-confirmation" in tex and "$-0.050" in tex
    md = open(os.path.join(out, "thesis_tables.md"), encoding="utf-8").read()
    assert "-0.050 [-0.080, -0.010]" in md                     # the same digits as the LaTeX cell
    prov = json.load(open(os.path.join(out, "provenance.json"), encoding="utf-8"))
    assert prov["tables"]["t3_confirmation"]["sources"]


def test_thesis_table_1_reports_each_records_own_seed_scheme(tmp_path):
    """1.8.0 records store their seed scheme at the top level (the design plan has no such key): Table 1 must report
    scheme 2 for the shipped 1.8.0 study (an earlier generator read only the plan and reported 1)."""
    from qemx.selection import thesis_tables as TT
    shipped = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "selection_v180")
    if not os.path.isdir(os.path.join(shipped, "dev", "tasks")):
        pytest.skip("the shipped 1.8.0 study is not in this tree")
    TT.build(shipped, str(tmp_path), compile_pdf=False)
    t1 = pd.read_csv(os.path.join(str(tmp_path), "t1_studies.csv"), encoding="utf-8", dtype={"scheme": str})
    assert set(t1.scheme) == {"2"}, t1[["study", "scheme"]]
