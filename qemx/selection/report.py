"""Results tables of a protocol-selection run in Markdown (1.6.0), generated from the saved analysis tables only
(``python -m qemx.selection report CONFIG --out RUN``). The interpretation in ``docs/selection/RESULTS.md`` quotes
these numbers; regenerate this file after any re-analysis."""
from __future__ import annotations

import math
import os

import numpy as np

from .. import textio as TX


def _present(path: str) -> bool:
    """A table or file exists and is not empty (1.9.0: an analysis that found nothing writes an empty CSV, which
    pandas cannot read; its section is then left out instead of failing the report)."""
    if not os.path.exists(path):
        return False
    if os.path.isfile(path) and path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            return bool(f.readline().strip())
    return True


def _md(df, floatfmt="%.3g") -> str:
    from .figures import markdown_table
    return markdown_table(df)


def _regret_columns(tab, rows):
    """Append the regret metrics of the specification (1.6.0 re-analysis) to a policy table when present."""
    if "mean_log_regret" not in rows.columns:
        return tab
    g = rows.groupby("policy")[["mean_log_regret", "log_regret_lo", "log_regret_hi", "median_regret", "p90_regret",
                                "p_negative_regret", "tol_success_10pct", "raw_degradation", "fail_share"]].mean()
    g = g.reindex(tab.policy)
    tab = tab.copy()
    tab["mean_log10_regret"] = [_ci(*r) for r in g[["mean_log_regret", "log_regret_lo", "log_regret_hi"]].values]
    tab["median_regret"] = g.median_regret.map(lambda v: "%.2g" % v).values
    tab["p90_regret"] = g.p90_regret.map(lambda v: "%.2g" % v).values
    tab["share_negative_regret"] = g.p_negative_regret.round(3).values
    tab["within_10pct"] = g.tol_success_10pct.round(3).values
    tab["worse_than_RAW"] = g.raw_degradation.round(3).values
    tab["fallback_trials"] = g.fail_share.round(4).values
    return tab


def _ci(m, lo, hi, f="%.3f") -> str:
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in (m, lo, hi)):
        return "—"
    return (f + " [" + f + ", " + f + "]") % (m, lo, hi)


def write(run_dir: str, out_path: str) -> str:
    import pandas as pd
    a = os.path.join(run_dir, "analysis")
    lines = ["# Results tables — %s" % os.path.basename(os.path.normpath(run_dir)), "",
             "Generated from `%s/analysis/` by `python -m qemx.selection report`. Simulated observations (emulated "
             "fake-device trials); no hardware data. Risks are evaluation-half MSEs; log values are log10(MSE + "
             "1e-6); intervals are 95 %% bootstrap intervals over test base circuits." % os.path.basename(
                 os.path.normpath(run_dir)), ""]
    lab = pd.read_csv(os.path.join(a, "labels.csv"), encoding="utf-8")
    lines += ["## Data", "",
              "%d tasks from %d base circuits; %d budgets; %d protocols; %d (task, protocol, budget) labels, of which "
              "%d feasible." % (lab.task_id.nunique(), lab.base_id.nunique(), lab.B.nunique(), lab.protocol.nunique(),
                                len(lab), int(lab.feasible.sum())), ""]
    f = lab[lab.feasible].copy()
    f["lg"] = np.log10(f.mse_eval + 1e-6)
    med = f.pivot_table(index="protocol", columns="B", values="lg", aggfunc="median")
    lines += ["### Median log10(MSE + 10⁻⁶) by protocol and budget (evaluation half, feasible tasks)", "",
              _md(med.reset_index().rename(columns={c: "%.0e" % c for c in med.columns if not isinstance(c, str)})),
              ""]
    best = f.loc[f.groupby(["task_id", "B"]).mse_eval.idxmin()]
    wins = best.pivot_table(index="protocol", columns="B", values="task_id", aggfunc="count").fillna(0).astype(int)
    lines += ["### How often each protocol is the measured best (tasks per budget)", "",
              _md(wins.reset_index().rename(columns={c: "%.0e" % c for c in wins.columns if not isinstance(c, str)})),
              ""]
    feas = lab.pivot_table(index="protocol", columns="B", values="feasible", aggfunc="mean")
    lines += ["### Share of tasks on which each protocol is feasible", "",
              _md(feas.reset_index().rename(columns={c: "%.0e" % c for c in feas.columns if not isinstance(c, str)})),
              ""]
    pooled = _present(os.path.join(a, "pooled_same-distribution.csv"))
    s = pd.read_csv(os.path.join(a, ("pooled_%s.csv" if pooled else "summary_%s.csv") % "same-distribution"), encoding="utf-8")
    cols = ["mean_log10_risk", "ci_lo", "ci_hi", "diff_vs_sbs", "diff_lo", "diff_hi", "gap_closed", "gap_lo",
            "gap_hi", "policy_mse", "p_regret_gt_2x", "coverage"]
    for c in ("diff_vs_sbsb", "diff_vs_sbsb_lo", "diff_vs_sbsb_hi"):
        if c not in s.columns:
            s[c] = np.nan
    cols += ["diff_vs_sbsb", "diff_vs_sbsb_lo", "diff_vs_sbsb_hi"]
    sa = s[s.B.astype(str) == "all"].groupby("policy")[cols].mean()
    sa = sa.sort_values("mean_log10_risk").reset_index()
    tab = pd.DataFrame(dict(policy=sa.policy,
                            mean_log10_mse=[_ci(*r) for r in sa[["mean_log10_risk", "ci_lo", "ci_hi"]].values],
                            diff_vs_SBS=[_ci(*r) for r in sa[["diff_vs_sbs", "diff_lo", "diff_hi"]].values],
                            diff_vs_SBS_B=[_ci(*r) for r in sa[["diff_vs_sbsb", "diff_vs_sbsb_lo",
                                                                 "diff_vs_sbsb_hi"]].values],
                            gap_closed=[_ci(*r, f="%.2f") for r in sa[["gap_closed", "gap_lo", "gap_hi"]].values],
                            policy_MSE=sa.policy_mse, P_regret_gt_2x=sa.p_regret_gt_2x, coverage=sa.coverage))
    tab = _regret_columns(tab, s[s.B.astype(str) == "all"])
    how = "out-of-fold pooled" if pooled else "mean over folds"
    lines += ["## Policies on held-out base circuits (group 5-fold over base circuits, all budgets; %s)" % how, "",
              _md(tab), ""]
    sb = s[s.B.astype(str) != "all"].copy()
    sb["B"] = sb.B.astype(float)
    models = [p for p in sa.policy if p.startswith("MODEL[")]
    top = models[0] if models else None
    if top:
        rows = []
        for B, d in sb.groupby("B"):
            g = d.groupby("policy")[["mean_log10_risk", "diff_vs_sbs", "diff_lo", "diff_hi", "diff_vs_sbsb",
                                     "diff_vs_sbsb_lo", "diff_vs_sbsb_hi"]].mean()
            row = dict(B="%.0e" % B)
            for label, pol in (("best_model", top), ("SBS_B", "SBS-B"), ("SIZE_B", "SIZE-B"),
                               ("oracle", "ALL-CANDIDATE")):
                row[label] = _ci(*g.loc[pol, ["diff_vs_sbs", "diff_lo", "diff_hi"]]) if pol in g.index else "—"
            row["best_model_vs_SBS_B"] = _ci(*g.loc[top, ["diff_vs_sbsb", "diff_vs_sbsb_lo", "diff_vs_sbsb_hi"]]) \
                if top in g.index else "—"
            rows.append(row)
        lines += ["### Paired difference to SBS by budget (Δ mean log10(MSE + 10⁻⁶); < 0 is better). Best model: `%s` "
                  "(selected on these scores)" % top, "", _md(pd.DataFrame(rows)), ""]
    op = os.path.join(a, "optimism.csv")
    if _present(op):
        o = pd.read_csv(op, encoding="utf-8")
        ot = pd.DataFrame(dict(split=o.split, reference=o.ref, candidates=o.candidates, naive_best=o.naive_best,
                               naive_diff=o.naive_diff_vs_ref.round(3),
                               bias_corrected_diff=[_ci(*r) for r in o[["bbc_diff_vs_ref", "bbc_diff_lo",
                                                                          "bbc_diff_hi"]].values],
                               P_diff_below_0=o.p_diff_below_0.round(3), optimism=o.optimism.round(3)))
        lines += ["## Optimism of the selected configuration (BBC-CV, Tsamardinos et al. 2018)", "",
                  "The best model configuration is chosen on the same pooled scores it is reported with. BBC-CV "
                  "selects on bootstrap samples of base circuits and scores the selection on the out-of-bag "
                  "circuits; `optimism` is the difference to the naive winner's score (log10 units, > 0: the naive "
                  "score is too good). Interval: 2.5–97.5 % of the out-of-bag values.", "", _md(ot), ""]
    gen = []
    for kind in ("same-distribution", "family", "device", "width", "budget-extrap", "budget-interp"):
        pp = os.path.join(a, "pooled_%s.csv" % kind)
        p = pp if _present(pp) else os.path.join(a, "summary_%s.csv" % kind)
        if not _present(p):
            continue
        g = pd.read_csv(p, encoding="utf-8")
        g = g[g.B.astype(str) == "all"]
        parts = len(set(pd.read_csv(os.path.join(a, "summary_%s.csv" % kind), encoding="utf-8").part)) \
            if _present(os.path.join(a, "summary_%s.csv" % kind)) else 1
        for pol in sorted(set(g.policy)):
            if pol.startswith("MODEL[") or pol in ("SBS-B", "SIZE-B", "ALL-CANDIDATE"):
                d = g[g.policy == pol]
                sbsb = [d[c].mean() if c in d.columns else np.nan
                        for c in ("diff_vs_sbsb", "diff_vs_sbsb_lo", "diff_vs_sbsb_hi")]
                gen.append(dict(split=kind, policy=pol, parts=parts, base_circuits=int(d.base_circuits.max()),
                                mean_log10_mse=_ci(d.mean_log10_risk.mean(), d.ci_lo.mean(), d.ci_hi.mean()),
                                diff_vs_SBS=_ci(d.diff_vs_sbs.mean(), d.diff_lo.mean(), d.diff_hi.mean()),
                                diff_vs_SBS_B=_ci(*sbsb),
                                gap_closed=_ci(d.gap_closed.mean(), d.gap_lo.mean(), d.gap_hi.mean(), f="%.2f"),
                                mean_log10_regret=_ci(d.mean_log_regret.mean(), d.log_regret_lo.mean(),
                                                      d.log_regret_hi.mean()) if "mean_log_regret" in d.columns
                                else "—"))
    if gen:
        lines += ["## Generalization (all budgets; test rows of all parts of a split pooled)", "",
                  _md(pd.DataFrame(gen)), ""]
    ab_p = os.path.join(a, "ablation.csv")
    tuned = _present(ab_p) and "tuned" in pd.read_csv(ab_p, encoding="utf-8").columns
    ab_title = ("Feature-group ablation (frozen configuration, every variant retrained and retuned; Δ mean log10 "
                "regret vs all features, > 0: the group helps)" if tuned else
                "Feature-group ablation (HGB, fixed setting; Δ vs all features, > 0: the group helps)")
    for name, title in (("ablation", ab_title),
                        ("floor_sensitivity", "Floor ε0 of the log target (HGB, fixed setting)"),
                        ("abstention", "Abstention (HGB, deployable features, fixed setting)"),
                        ("interval_coverage", "Interval coverage on held-out base circuits (nominal 0.90)")):
        p = os.path.join(a, name + ".csv")
        if not _present(p):
            continue
        d = pd.read_csv(p, encoding="utf-8")
        if name == "interval_coverage":
            d = d.groupby("kind")[["coverage", "nominal"]].agg(["mean", "min", "max"]).round(3)
            d.columns = ["_".join(c) for c in d.columns]
            d = d.reset_index()
        if name == "abstention":
            d = d[[c for c in d.columns if not c.startswith("coverage|")]]
        lines += ["## " + title, "", _md(d), ""]
    lines += _deployed_sections(run_dir, a)
    p = os.path.join(a, "certificates.csv")
    if _present(p):
        c = pd.read_csv(p, encoding="utf-8")
        col = "radius" if "radius" in c.columns else "halfwidth"
        hb = c[c.method.isin(["hoeffding", "bernstein"])].groupby(["method", "B"]).agg(
            informative=("informative", "mean"), median_radius=(col, "median"),
            median_mse_chosen=("mse_clipped", "median")).reset_index()
        te = c[c.method == "two-eps"]
        lines += ["## Certificates (clipped variant, H = 4, K feasible protocols, α = 0.05; r = trials of the "
                  "evaluation half)", "", _md(hb), "",
                  "2ε bound (Prop. 1.5) with the selection-half risks as predictions: holds on %d of %d rows; median "
                  "regret / bound = %.3g." % (int(te.holds.sum()), len(te), float(np.nanmedian(
                      te.regret / te.bound.replace(0, np.nan)))), ""]
    mdir = os.path.join(run_dir, "mismatch")
    if _present(os.path.join(mdir, "policies.csv")):
        dg = pd.read_csv(os.path.join(mdir, "diagnostics.csv"), encoding="utf-8")
        po = pd.read_csv(os.path.join(mdir, "policies.csv"), encoding="utf-8")
        t1 = dg.groupby("scenario").agg(flag_rate=("flag_rate", "mean"), z_median=("z_median", "median"),
                                        echo_p_true=("p_true", "mean"), echo_p_sup=("p_sup", "mean")).reset_index()
        t2 = po.pivot_table(index="scenario", columns="policy", values="mean_log10_risk", aggfunc="mean").reset_index()
        lines += ["## Mismatch", "", "### Echo diagnostic per scenario (mean over tasks and budgets)", "", _md(t1), "",
                  "### Mean log10(MSE + 10⁻⁶) per scenario and policy (diagnostic shots charged)", "", _md(t2), ""]
        nc = os.path.join(mdir, "null_check.csv")
        if _present(nc):
            n = pd.read_csv(nc, encoding="utf-8")
            lines += ["Simulated false-positive rate of the echo test under the finite-shot null: mean %.4f over %d "
                      "settings (α = 0.05)." % (float(n.simulated_false_positive.mean()), len(n)), ""]
    fdir = os.path.join(run_dir, "final")
    if _present(os.path.join(fdir, "summary_final.csv")):
        fs = pd.read_csv(os.path.join(fdir, "summary_final.csv"), encoding="utf-8")
        fa = fs[fs.B.astype(str) == "all"]
        tab = pd.DataFrame(dict(policy=fa.policy,
                                mean_log10_mse=[_ci(*r) for r in fa[["mean_log10_risk", "ci_lo", "ci_hi"]].values],
                                diff_vs_SBS=[_ci(*r) for r in fa[["diff_vs_sbs", "diff_lo", "diff_hi"]].values],
                                diff_vs_SBS_B=[_ci(*r) for r in fa[["diff_vs_sbsb", "diff_vs_sbsb_lo",
                                                                     "diff_vs_sbsb_hi"]].values]
                                if "diff_vs_sbsb" in fa.columns else "—",
                                gap_closed=[_ci(*r, f="%.2f") for r in fa[["gap_closed", "gap_lo", "gap_hi"]].values],
                                policy_MSE=fa.policy_mse, coverage=fa.coverage))
        mf = os.path.join(fdir, "metrics_final.csv")
        if _present(mf):
            m_ = pd.read_csv(mf, encoding="utf-8")
            tab = _regret_columns(tab, m_[m_.B.astype(str) == "all"])
        frz = TX.read_json(os.path.join(run_dir, "freeze.json"))
        lines += ["## Historical final test (the 1.6.0 final test relabeled with fresh designs, evaluated once with the "
                  "frozen policy `%s`; not confirmatory)" % frz["policy"], "", _md(tab), ""]
    lines += _v170_sections(run_dir)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


def _deployed_sections(run_dir: str, a: str) -> list:
    """Family benefit, selective recommendation, calibration, learning curve, budget generalization, regime study
    and classical cost — the analyses of the 1.6.0 re-analysis (decision D25), when their tables exist."""
    import pandas as pd
    out = []
    p = os.path.join(a, "family_benefit.csv")
    if _present(p):
        f = pd.read_csv(p, encoding="utf-8")
        t = pd.DataFrame(dict(family=f.family, base_circuits=f.base_circuits,
                              diff_log10_mse=[_ci(*r) for r in f[["diff_log", "diff_log_lo", "diff_log_hi"]].values],
                              diff_mse=["%.2g [%.2g, %.2g]" % tuple(r) for r in f[["diff_mse", "diff_mse_lo",
                                                                                    "diff_mse_hi"]].values]))
        out += ["## Paired benefit of the frozen configuration over SBS-B by family (all budgets; < 0 favors the "
                "frozen configuration)", "", _md(t), ""]
    p = os.path.join(a, "selective_operating.csv")
    if _present(p):
        o = pd.read_csv(p, encoding="utf-8").iloc[0]
        fo = pd.read_csv(os.path.join(a, "selective_folds.csv"), encoding="utf-8")
        t = pd.DataFrame([dict(policy="model only (coverage 1)", mean_log10_mse=o.model_log_risk,
                               mean_log10_regret=o.model_log_regret, mean_mse=o.get("model_mse", np.nan)),
                          dict(policy="selective, cross-fitted threshold (coverage %.3f)" % o.coverage,
                               mean_log10_mse=o.policy_log_risk, mean_log10_regret=o.policy_log_regret,
                               mean_mse=o.get("policy_mse", np.nan)),
                          dict(policy="SBS-B only (coverage 0)", mean_log10_mse=o.fallback_log_risk,
                               mean_log10_regret=o.fallback_log_regret, mean_mse=o.get("fallback_mse", np.nan))])
        out += ["## Selective recommendation of the frozen configuration (margin threshold, fallback SBS-B)", "",
                _md(t.round(5)), "",
                "Thresholds per test fold (chosen on the other folds): %s; accepted-row mean log10 regret at the "
                "operating point %.3f; deployment threshold on all development rows τ = %.4g."
                % (", ".join("%s %.3g" % (r.fold, r.tau) for r in fo.itertuples()), o.accepted_log_regret,
                   o.deploy_tau), ""]
    p = os.path.join(a, "calibration.csv")
    if _present(p):
        c = pd.read_csv(p, encoding="utf-8")
        t = c.groupby(["kind", "nominal"]).agg(coverage=("coverage", "mean"), fold_min=("coverage", "min"),
                                               fold_max=("coverage", "max"), mean_width=("mean_width", "mean"))
        out += ["## Calibration of the frozen configuration's log-risk intervals (held out; inner-CV residuals)", "",
                _md(t.round(3).reset_index()), ""]
    p = os.path.join(a, "learning_curve.csv")
    if _present(p):
        lc = pd.read_csv(p, encoding="utf-8")
        if "mean_log_regret" in lc.columns:
            t = lc.groupby("train_base_circuits").agg(model=("mean_log_regret", "mean"),
                                                      q10=("mean_log_regret", lambda x: x.quantile(0.1)),
                                                      q90=("mean_log_regret", lambda x: x.quantile(0.9)),
                                                      SBS=("sbs_mean_log_regret", "mean"),
                                                      draws=("mean_log_regret", "size"))
            out += ["## Learning curve (mean log10 regret on held-out folds)", "", _md(t.round(3).reset_index()), ""]
    p = os.path.join(a, "budget_generalization.csv")
    if _present(p):
        b = pd.read_csv(p, encoding="utf-8")
        t = b.pivot_table(index=["max_train_B", "policy"], columns="B", values="mean_log_regret").round(3)
        t.columns = ["%.0e" % c for c in t.columns]
        out += ["## Budget generalization (mean log10 regret by test budget; training restricted to B ≤ max_train_B)",
                "", _md(t.reset_index()), ""]
    rc = os.path.join(run_dir, "regimes", "analysis", "regime_cells.csv")
    if _present(rc):
        r = pd.read_csv(rc, encoding="utf-8")
        m = r[(r.observable == "default") & (r.layout == "best")]
        t = m.groupby(["family", "noise_scale"]).agg(
            cells=("B", "size"), unresolved=("unresolved", "sum"), abstained=("abstained", "sum"),
            model_equals_reference=("model_choice", lambda x: int((x == m.loc[x.index, "reference"]).sum())),
            policy_log_regret=("log_regret_policy", "mean"), model_log_regret=("log_regret_model", "mean"),
            fallback_log_regret=("log_regret_fallback", "mean")).round(3).reset_index()
        out += ["## Controlled regime study (decision D23; per slice and error scale, all ten budgets)", "", _md(t),
                ""]
    nc = os.path.join(run_dir, "noisecomp", "analysis", "regime_cells.csv")
    if _present(nc):
        r = pd.read_csv(nc, encoding="utf-8")
        r["noise_change"] = r.noise_change.fillna("none")
        t = r[r.B == 100000][["family", "noise_change", "lam_total", "reference", "runner_up", "unresolved",
                               "model_choice", "log_regret_policy"]].sort_values(["family", "noise_change"])
        out += ["## Noise composition (decision D35; B = 10⁵; one component changed at a time)", "",
                _md(t.round(4)), ""]
        w = r.groupby("noise_change").agg(cells=("B", "size"), unresolved=("unresolved", "mean"),
                                          model_equals_reference=("model_choice",
                                                                  lambda x: float((x == r.loc[x.index, "reference"])
                                                                                  .mean())),
                                          policy_log_regret=("log_regret_policy", "mean")).round(3).reset_index()
        out += ["### All budgets, per noise change", "", _md(w), ""]
    rt = os.path.join(a, "runtime.csv")
    if _present(rt):
        r = pd.read_csv(rt, encoding="utf-8")
        cols = [c for c in ("compile_s", "deployable_s", "simulated_s", "inference_s", "label_s", "prepare_s",
                            "features_s", "peak_rss_mb") if c in r.columns]
        t = r.groupby("n")[cols].median().round(4).reset_index()
        out += ["## Classical cost of one recommendation (median of repeats, seconds; label_s: one development task's "
                "label generation, for comparison)", "", _md(t), ""]
    return out


def _v170_sections(run_dir: str) -> list:
    """1.7.0 tables: the old/new estimand comparison, design variance, failure profile, nested selective
    evaluation, the design study and the confirmation."""
    import pandas as pd
    a = os.path.join(run_dir, "analysis")
    out = []
    p = os.path.join(run_dir, "analysis_estimands", "summary.csv")
    if _present(p):
        e = pd.read_csv(p, encoding="utf-8")
        out += ["## 1.6.0 against corrected CDR/NIL labels (same tasks and seeds; evaluation half)", "",
                _md(e[["protocol", "B", "tasks", "median_mse_old", "median_mse_new", "mean_mse_old", "mean_mse_new",
                       "median_se_ratio", "share_old_outside_new_interval"]].round(5)), ""]
    p = os.path.join(a, "design_variance.csv")
    if _present(p):
        out += ["## Learning-design variance of the labels (evaluation half)", "",
                _md(pd.read_csv(p, encoding="utf-8").round(3)), ""]
    p = os.path.join(a, "failure_profile.csv")
    if _present(p):
        f = pd.read_csv(p, encoding="utf-8")
        f = f[f.B.astype(str) == "all"]
        out += ["## Failure profile (all budgets; same-distribution held-out rows)", "", _md(f.round(4)), ""]
    p = os.path.join(a, "nested_summary.csv")
    if _present(p):
        n = pd.read_csv(p, encoding="utf-8")
        t = pd.DataFrame(dict(policy=n.policy, coverage=n.coverage.round(3),
                              mse=["%.3g [%.3g, %.3g]" % tuple(r) for r in n[["mse", "mse_lo", "mse_hi"]].values],
                              mean_log10_mse=[_ci(*r) for r in n[["mean_log10_risk", "log_lo", "log_hi"]].values],
                              diff_log_vs_SBS_B=[_ci(*r) for r in n[["diff_log_vs_sbsb", "diff_log_vs_sbsb_lo",
                                                                     "diff_log_vs_sbsb_hi"]].values],
                              diff_log_vs_random=[_ci(*r) for r in n[["diff_log_vs_random", "diff_log_vs_random_lo",
                                                                      "diff_log_vs_random_hi"]].values]))
        fo = pd.read_csv(os.path.join(a, "nested_folds.csv"), encoding="utf-8")
        op = pd.read_csv(os.path.join(a, "nested_operating.csv"), encoding="utf-8").iloc[0]
        out += ["## Selective recommendation, nested evaluation (`%s`)" % n.configuration.iloc[0], "", _md(t), "",
                "Outer folds' τ: %s; deployment τ = %.4g." % (", ".join("%.3g" % x for x in fo.tau),
                                                              float(op.deploy_tau)), ""]
    ds = os.path.join(run_dir, "design_study", "summary.csv")
    if _present(ds):
        d = pd.read_csv(ds, encoding="utf-8")
        cols = [c for c in ("task", "variant", "B", "designs", "mse", "se_design", "median_design_loss",
                            "max_design_loss", "severe_share", "severe_contribution", "fallback_share") if c in d]
        out += ["## Design study (experiment B; design-level losses)", "", _md(d[cols].round(6)), ""]
    cdir = os.path.join(run_dir, "confirm")
    mc = os.path.join(cdir, "metrics_confirm.csv")
    if _present(mc):
        m = pd.read_csv(mc, encoding="utf-8")
        m = m[m.B.astype(str) == "all"]
        t = pd.DataFrame(dict(policy=m.policy, base_circuits=m.base_circuits,
                              mean_log10_mse=[_ci(*r) for r in m[["mean_log10_risk", "ci_lo", "ci_hi"]].values],
                              mse=m.policy_mse.map(lambda v: "%.3g" % v),
                              diff_vs_SBS_B=[_ci(*r) if np.isfinite(r).all() else "" for r in
                                             m.reindex(columns=["diff_vs_sbsb", "diff_vs_sbsb_lo",
                                                                "diff_vs_sbsb_hi"]).values],
                              mean_log10_regret=[_ci(*r) for r in m[["mean_log_regret", "log_regret_lo",
                                                                     "log_regret_hi"]].values],
                              coverage=m.coverage.round(3)))
        out += ["## Confirmation (new logical circuits; evaluated once; confirmatory)", "", _md(t), ""]
    return out
