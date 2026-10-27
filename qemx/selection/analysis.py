"""Analyses of a labeled study (1.6.0). Every function reads the decision dataset (saved trials) and writes one
source-data table; figures and tables are drawn from those tables only (``qemx.selection.figures``).

==========================  ==========================================================================================
table                       content
==========================  ==========================================================================================
summary_<split>.csv         per policy, budget and split part: geometric-mean risk with base-circuit bootstrap CI,
                            policy MSE, gap closed, regret tails, coverage (``evaluate.evaluate``)
choices_<split>.csv         per test row: every policy's choice, log regret ratio, acceptance
ablation.csv                held-out mean log risk of a model without each feature group, and its paired difference
learning_curve.csv          gap closed against the number of training base circuits (repeated subsamples)
abstention.csv              accepted-task and all-task risk against coverage over the conformal level
interval_coverage.csv       empirical coverage of conformal (marginal, simultaneous) and residual-Gaussian intervals
certificates.csv            Hoeffding and empirical-Bernstein certificates per test row and budget
prediction.csv              predicted vs evaluation-half log risk per row and protocol (one model)
labels.csv                  the labels themselves (task, protocol, budget: MSE, bias, variance, SE, failures, costs)
==========================  ==========================================================================================
"""
from __future__ import annotations

import math

import numpy as np

from . import evaluate as EV
from . import features as FE
from . import policies as PO
from .models import EPS0, FIXED


def _folds(ds, k, seed):
    return EV.splits(ds, "same-distribution", k=k, seed=seed)


def ablation(ds, family: str = "hgb", base_set: str = "deployable+simulated", k: int = 5, seed: int = 0,
             reps: int = 1000, tuned: bool = True) -> list:
    """Drop one feature group at a time from ``base_set``; each ablated model is retrained and (``tuned``) tuned by
    the same inner group-K-fold protocol as the full model. Scored on held-out base circuits (group-K-fold): mean
    log10 regret ratio against the per-task reference and its paired increase over the full model (> 0: the group
    helps; negative values mean removing the group helped). The reference is the same for every variant, so the
    increase in mean log10 regret equals the increase in mean log10 MSE."""
    base = list(FE.SETS[base_set])
    groups = sorted({FE.BY_NAME[n].group for n in base})
    variants = [("all", tuple(base))] + [("-" + g, tuple(n for n in base if FE.BY_NAME[n].group != g))
                                         for g in groups]
    for name, drop in FE.SEMANTIC_ABLATIONS.items():          # 1.7.0: remove everything derived from a source
        kept = tuple(n for n in base if n not in drop)
        if len(kept) < len(base) and all(kept != v for _, v in variants):
            variants.append((name, kept))
    per = {name: np.full(len(ds.rows), np.nan) for name, _ in variants}
    per_abs = {name: np.full(len(ds.rows), np.nan) for name, _ in variants}
    for _, tr, te in _folds(ds, k, seed):
        for name, names in variants:
            if not names:
                continue
            pol = PO.Model(family, names, name, fixed=None if tuned else FIXED[family]).fit(ds, tr)
            ch, _ = pol.choose(ds, te)
            sc = EV.scores(ds, te, ch)
            per[name][te] = sc["log_ratio"]
            per_abs[name][te] = sc["regret"]
    ci, M, picks = EV.bootstrap(per, ds.groups, reps, seed)
    out = []
    for name, _ in variants:
        diff = M[name] - M["all"]
        bs = np.nanmean(diff[picks], axis=1)
        out.append(dict(variant=name, family=family, base_set=base_set, tuned=bool(tuned),
                        kind=("semantic" if name in FE.SEMANTIC_ABLATIONS else ("full" if name == "all" else "group")),
                        removed=",".join(sorted(set(base) - set(dict(variants)[name]))),
                        base_circuits=int(len(set(ds.groups.tolist()))), uncertainty="bootstrap over base circuits",
                        mean_log_regret=ci[name][0], ci_lo=ci[name][1], ci_hi=ci[name][2],
                        mean_regret=float(np.nanmean(EV._by_group(per_abs[name], ds.groups)[0])),
                        diff_vs_all=float(np.nanmean(diff)), diff_lo=float(np.nanquantile(bs, 0.025)),
                        diff_hi=float(np.nanquantile(bs, 0.975))))
    return out


def learning_curve(ds, family: str = "hgb", set_name: str = "deployable", fractions=(0.25, 0.5, 0.75, 1.0),
                   repeats: int = 3, k: int = 5, seed: int = 0) -> list:
    """Held-out mean log10 regret (and gap closed) against the number of independent training base circuits
    (subsampled within each training fold, ``repeats`` draws; the test folds stay fixed; the model's setting is fixed
    so that tuning does not change with the training size)."""
    rng = np.random.default_rng(seed)
    out = []
    for fold, tr, te in _folds(ds, k, seed):
        g_tr = sorted(set(ds.groups[tr].tolist()))
        for f in fractions:
            for rep in range(repeats if f < 1 else 1):
                keep = set(rng.choice(g_tr, size=max(3, int(round(f * len(g_tr)))), replace=False).tolist())
                sub = np.array([i for i in tr if ds.groups[i] in keep])
                pols = [PO.SBS(), PO.Model(family, FE.SETS[set_name], set_name, fixed=FIXED[family]),
                        PO.AllCandidate()]
                ev = EV.evaluate(ds, sub, te, pols, reps=200, seed=seed + rep)
                s = [x for x in ev["summary"] if x["B"] == "all" and x["policy"].startswith("MODEL")][0]
                sb = [x for x in ev["summary"] if x["B"] == "all" and x["policy"] == "SBS"][0]
                out.append(dict(fold=fold, fraction=f, repeat=rep, train_base_circuits=len(keep),
                                test_base_circuits=s["base_circuits"], gap_closed=s.get("gap_closed"),
                                mean_log10_risk=s["mean_log10_risk"], diff_vs_sbs=s.get("diff_vs_sbs"),
                                mean_log_regret=s.get("mean_log_regret"),
                                sbs_mean_log_regret=sb.get("mean_log_regret")))
    return out


def abstention(ds, family: str = "hgb", set_name: str = "deployable", alphas=(0.5, 0.3, 0.2, 0.1, 0.05, 0.02),
               k: int = 5, seed: int = 0) -> list:
    """Coverage κ, accepted-task and fallback-inclusive (all-task) geometric-mean risk over the conformal level of
    the abstaining policy, per family and device as well (group-K-fold held-out rows)."""
    out = []
    for a in alphas:
        acc = np.zeros(len(ds.rows), bool)
        lr = np.full(len(ds.rows), np.nan)
        for _, tr, te in _folds(ds, k, seed):
            pol = PO.ModelAbstain(family, FE.SETS[set_name], set_name, alpha=a, fixed=FIXED[family]).fit(ds, tr)
            ch, ok = pol.choose(ds, te)
            acc[te] = ok
            lr[te] = EV.scores(ds, te, ch)["log_risk"]
        fam, dev = ds.col("family"), ds.col("device")
        row = dict(alpha=a, coverage=float(acc.mean()),
                   accepted_log10_risk=float(np.nanmean(lr[acc])) if acc.any() else math.nan,
                   all_log10_risk=float(np.nanmean(lr)))
        for f in sorted(set(fam.tolist())):
            row["coverage|family=%s" % f] = float(acc[fam == f].mean())
        for d in sorted(set(dev.tolist())):
            row["coverage|device=%s" % d] = float(acc[dev == d].mean())
        out.append(row)
    return out


def interval_coverage(ds, family: str = "hgb", set_name: str = "deployable", alpha: float = 0.1, k: int = 5,
                      seed: int = 0) -> list:
    from .certify import interval_coverage as IC, split_conformal as SC
    out = []
    for fold, tr, te in _folds(ds, k, seed):
        pol = PO.Model(family, FE.SETS[set_name], set_name).fit(ds, tr)
        for kind, d in IC(pol, ds, te, alpha).items():
            out.append(dict(fold=fold, kind=kind, **d))
        for kind, d in SC(family, FE.SETS[set_name], set_name, ds, tr, te, alpha, seed=seed).items():
            out.append(dict(fold=fold, kind=kind, **d))
    return out


def certificates(ds, labels_dir: str, alpha: float = 0.05) -> list:
    """Per (task, budget): Hoeffding and empirical-Bernstein simultaneous certificates on the clipped evaluation
    half; the 2ε bound with the measured selection-half risks as the 'prediction'."""
    from . import labels as LB
    from .certify import certify, regret_two_eps
    out = []
    cache = {}
    raw = ds.protocols.index("RAW")
    for i, r in enumerate(ds.rows):
        tid = r["task_id"]
        if tid not in cache:
            cache = {tid: LB.load_task(labels_dir, tid)}
        rec, arr = cache[tid]
        j = rec["budgets"].index(r["B"])
        est = arr["est"][1, :, j, :]
        R = est.shape[1]
        fresh = LB.plan_of(rec).get("mode") == "fresh"
        des = arr["design"][1, :, j, :]
        for method in ("hoeffding", "bernstein"):
            c = certify(est, r["ideal"], raw, alpha, method, design=des, fresh=fresh)
            out.append(dict(task_id=tid, family=r["family"], B=r["B"], method=method, r=R, **c))
        e2 = regret_two_eps(np.where(ds.F[i], ds.Y_sel[i], np.nan), np.where(ds.F[i], ds.Y_eval[i], np.nan))
        out.append(dict(task_id=tid, family=r["family"], B=r["B"], method="two-eps", r=R, **e2))
    return out


def floor_sensitivity(ds, family: str = "hgb", set_name: str = "deployable", floors=(1e-7, 1e-6, 1e-5), k: int = 5,
                      seed: int = 0, reps: int = 1000) -> list:
    """The model trained with each floor ε0 of its log target, scored with the common primary aggregate
    (ε0 = 1e-6): does the floor change the decisions?"""
    per = {}
    for f in floors:
        v = np.full(len(ds.rows), np.nan)
        for _, tr, te in _folds(ds, k, seed):
            pol = PO.Model(family, FE.SETS[set_name], set_name, eps0=f, fixed=FIXED[family]).fit(ds, tr)
            ch, _ = pol.choose(ds, te)
            v[te] = EV.scores(ds, te, ch)["log_risk"]
        per["eps0=%g" % f] = v
    ci, M, picks = EV.bootstrap(per, ds.groups, reps, seed)
    ref = "eps0=%g" % EPS0
    out = []
    for name in per:
        diff = M[name] - M[ref]
        bs = np.nanmean(diff[picks], axis=1)
        out.append(dict(variant=name, mean_log10_risk=ci[name][0], ci_lo=ci[name][1], ci_hi=ci[name][2],
                        diff_vs_default=float(np.nanmean(diff)), diff_lo=float(np.nanquantile(bs, 0.025)),
                        diff_hi=float(np.nanquantile(bs, 0.975))))
    return out


# ------------------------------------------------------------------ the deployed recommender (Figures 5, S3)
def deployed(ds, family: str = "hgb", set_name: str = "deployable+simulated", k: int = 5, seed: int = 0,
             levels=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """Out-of-fold analyses of one recommender configuration (by default the configuration frozen for the final
    test), each fold's model fitted once and reused:

    * ``rows``: per held-out row its choice, the fallback's choice (SBS-B of the same training fold), the confidence
      margin z = (P₂ − P₁)/√(s₁² + s₂²) between the two lowest predicted log risks (s = that protocol's inner-CV RMSE
      of the log target), and the realized log10 regret ratio, log10 risk and absolute regret of both;
    * ``prediction``: predicted against evaluation-half log10 risk per row and feasible protocol, with the label's
      standard error on the log scale (delta method);
    * ``calibration``: empirical coverage (base-circuit weighted) and mean width of the log-risk intervals at several
      nominal levels — per candidate (marginal group-conformal), for the selected candidate only (post-selection, the
      same marginal intervals) and Bonferroni-simultaneous over the feasible candidates (all covered jointly)."""
    rows, pred, cal = [], [], []
    for fold, tr, te in _folds(ds, k, seed):
        pol = PO.Model(family, FE.SETS[set_name], set_name).fit(ds, tr)
        fb = PO.SBSBudget().fit(ds, tr)
        P = pol.predict(ds, te)
        ch, _ = pol.choose(ds, te)
        fch, _ = fb.choose(ds, te)
        sm = EV.scores(ds, te, ch)
        sf = EV.scores(ds, te, fch)
        rm = pol.rm
        for r, i in enumerate(te):
            feas = [j for j, p in enumerate(ds.protocols) if ds.F[i, j] and np.isfinite(P[r, j])]
            order = sorted(feas, key=lambda j: P[r, j])
            if len(order) >= 2:
                j1, j2 = order[0], order[1]
                s1 = rm.cv_rmse.get(ds.protocols[j1], np.nan)
                s2 = rm.cv_rmse.get(ds.protocols[j2], np.nan)
                margin = float((P[r, j2] - P[r, j1]) / math.sqrt(s1 ** 2 + s2 ** 2))
            else:
                margin = float("inf")
            row = ds.rows[i]
            rows.append(dict(fold=fold, task_id=row["task_id"], base_id=row["base_id"], family=row["family"],
                             device=row["device"], n=row["n"], B=row["B"], choice=ds.protocols[ch[r]],
                             fallback=ds.protocols[fch[r]], margin=margin,
                             predicted_log_risk=float(P[r, ch[r]]),
                             log_regret=float(sm["log_ratio"][r]), log_regret_fallback=float(sf["log_ratio"][r]),
                             log_risk=float(sm["log_risk"][r]), log_risk_fallback=float(sf["log_risk"][r]),
                             regret=float(sm["regret"][r]), regret_fallback=float(sf["regret"][r]),
                             risk=float(sm["risk"][r]), risk_fallback=float(sf["risk"][r])))
            for j, p in enumerate(ds.protocols):
                if ds.F[i, j] and np.isfinite(P[r, j]):
                    y = ds.Y_eval[i, j]
                    pred.append(dict(fold=fold, task_id=row["task_id"], family=row["family"], B=row["B"],
                                     protocol=p, predicted=float(P[r, j]), observed=float(math.log10(y + EPS0)),
                                     observed_se=float(ds.SE_eval[i, j] / ((y + EPS0) * math.log(10))),
                                     observed_sel=float(math.log10(ds.Y_sel[i, j] + EPS0)),
                                     chosen=bool(j == ch[r])))
        g = ds.groups[te]
        for lev in levels:
            a = 1 - lev
            hm = rm.conformal_halfwidth(a, simultaneous=False)
            hs = rm.conformal_halfwidth(a, simultaneous=True)
            hit_m, w_m, gm, hit_sel, w_sel, gs, hit_j, gj = [], [], [], [], [], [], [], []
            for r, i in enumerate(te):
                ok_all = []
                for j, p in enumerate(ds.protocols):
                    if not (ds.F[i, j] and np.isfinite(P[r, j]) and p in hm):
                        continue
                    y = math.log10(ds.Y_eval[i, j] + EPS0)
                    hit = abs(y - P[r, j]) <= hm[p]
                    hit_m.append(hit), w_m.append(2 * hm[p]), gm.append(g[r])
                    if j == ch[r]:
                        hit_sel.append(hit), w_sel.append(2 * hm[p]), gs.append(g[r])
                    ok_all.append(abs(y - P[r, j]) <= hs.get(p, -1))
                if ok_all:
                    hit_j.append(all(ok_all)), gj.append(g[r])
            for kind, hits, widths, gg in (("per candidate", hit_m, w_m, gm),
                                            ("selected candidate", hit_sel, w_sel, gs),
                                            ("simultaneous", hit_j, [np.nan] * len(hit_j), gj)):
                if not hits:
                    continue
                hh, gg = np.array(hits, float), np.array(gg)
                ww = np.array(widths, float)
                cal.append(dict(fold=fold, nominal=lev, kind=kind, coverage=float(np.mean(EV._by_group(hh, gg)[0])),
                                mean_width=float(np.mean(ww)) if np.isfinite(ww).all() and len(ww) else
                                (math.inf if len(ww) and not np.isnan(ww).all() else math.nan),
                                rows=int(len(hh))))
    return dict(rows=rows, prediction=pred, calibration=cal)


def selective(rows, grid: int = 51, reps: int = 400, seed: int = 0) -> dict:
    """Selective recommendation from the deployed rows (``deployed``): accept the recommender's choice when its
    margin z ≥ τ, otherwise use the fallback (SBS-B trained on the same rows; it needs no measurement, so abstaining
    costs no shots). For each τ: coverage (base-circuit-weighted share of rows accepted), mean log10 regret over the
    accepted rows, and the whole-policy mean log10 risk and log10 regret over all rows, with base-circuit bootstrap
    bands, and the whole-policy arithmetic mean MSE (base-circuit weighted; rows' ``risk`` / ``risk_fallback``).
    The accepted-row mean pools the accepted rows (row weighted), so a base circuit with more accepted rows weighs
    more there. Random rejection at the same coverage is the baseline (its expected values are exact mixtures). The
    deployment threshold is chosen on validation rows by cross-fitting: for each fold, the τ that minimizes the
    whole-policy mean log10 risk on the other folds' rows, applied to that fold (the test operating point)."""
    import pandas as pd
    d = pd.DataFrame(rows)
    d = d[np.isfinite(d.log_regret) & np.isfinite(d.log_regret_fallback)].reset_index(drop=True)
    m = d.margin.to_numpy(float)
    fin = m[np.isfinite(m)]
    taus = np.unique(np.concatenate([[-np.inf], np.quantile(fin, np.linspace(0, 1, grid)) if len(fin) else [],
                                     [np.inf]]))
    g = d.base_id.to_numpy()
    u = sorted(set(g.tolist()))
    gi = {k_: i for i, k_ in enumerate(u)}
    gidx = np.array([gi[k_] for k_ in g])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(u), size=(int(reps), len(u)))

    def sums(v, mask=None):
        mask = np.ones(len(v), bool) if mask is None else mask
        return (np.bincount(gidx[mask], weights=np.asarray(v, float)[mask], minlength=len(u)),
                np.bincount(gidx[mask], minlength=len(u)).astype(float))

    lrm, lrf = d.log_regret.to_numpy(), d.log_regret_fallback.to_numpy()
    lkm, lkf = d.log_risk.to_numpy(), d.log_risk_fallback.to_numpy()
    has_mse = "risk" in d.columns and "risk_fallback" in d.columns
    rkm = d.risk.to_numpy(float) if has_mse else np.full(len(d), np.nan)
    rkf = d.risk_fallback.to_numpy(float) if has_mse else np.full(len(d), np.nan)
    _, c_all = sums(np.ones(len(d)))

    def wmean(v, mask=None):
        s_, c_ = sums(v, mask)
        ok = c_ > 0
        return float(np.mean(s_[ok] / c_[ok])) if ok.any() else math.nan

    curve = []
    for t in taus:
        acc = m >= t
        s_cnt, _ = sums(acc.astype(float))
        s_a, c_a = sums(lrm, acc)
        s_r, _ = sums(np.where(acc, lkm, lkf))
        s_q, _ = sums(np.where(acc, rkm, rkf))
        mse_b = (s_q / c_all)[picks].mean(axis=1)
        cov_b = (s_cnt / c_all)[picks].mean(axis=1)
        acc_b = np.array([s_a[pk].sum() / c_a[pk].sum() if c_a[pk].sum() else np.nan for pk in picks])
        risk_b = (s_r / c_all)[picks].mean(axis=1)
        cov = float(np.mean(s_cnt / c_all))
        curve.append(dict(tau=float(t), coverage=cov, coverage_lo=float(np.quantile(cov_b, 0.025)),
                          coverage_hi=float(np.quantile(cov_b, 0.975)), accepted_rows=int(acc.sum()),
                          accepted_log_regret=float(s_a.sum() / c_a.sum()) if c_a.sum() else math.nan,
                          accepted_lo=float(np.nanquantile(acc_b, 0.025)) if np.isfinite(acc_b).any() else math.nan,
                          accepted_hi=float(np.nanquantile(acc_b, 0.975)) if np.isfinite(acc_b).any() else math.nan,
                          policy_log_risk=float(np.mean(s_r / c_all)), policy_lo=float(np.quantile(risk_b, 0.025)),
                          policy_hi=float(np.quantile(risk_b, 0.975)),
                          policy_log_regret=wmean(np.where(acc, lrm, lrf)),
                          random_accepted_log_regret=wmean(lrm),
                          random_policy_log_risk=float(cov * wmean(lkm) + (1 - cov) * wmean(lkf)),
                          policy_mse=float(np.mean(s_q / c_all)), policy_mse_lo=float(np.quantile(mse_b, 0.025)),
                          policy_mse_hi=float(np.quantile(mse_b, 0.975)),
                          random_policy_mse=float(cov * wmean(rkm) + (1 - cov) * wmean(rkf))))
    folds = sorted(set(d.fold.tolist()))
    acc_cf = np.zeros(len(d), bool)
    ops = []
    for f in folds:
        val = (d.fold != f).to_numpy()
        best_t, best_v = -np.inf, np.inf
        for t in taus:
            v = wmean(np.where(m >= t, lkm, lkf), val)
            if v < best_v - 1e-12:
                best_t, best_v = t, v
        te = ~val
        acc_cf[te] = m[te] >= best_t
        ops.append(dict(fold=f, tau=float(best_t), validation_policy_log_risk=float(best_v),
                        test_rows=int(te.sum()), test_accepted=int(acc_cf[te].sum())))
    s_a, c_a = sums(lrm, acc_cf)
    deploy_tau, best_v = None, np.inf
    for t in taus:
        v = wmean(np.where(m >= t, lkm, lkf))
        if v < best_v - 1e-12:
            deploy_tau, best_v = float(t), v
    op = dict(coverage=wmean(acc_cf.astype(float)),
              accepted_log_regret=float(s_a.sum() / c_a.sum()) if c_a.sum() else math.nan,
              policy_log_risk=wmean(np.where(acc_cf, lkm, lkf)), policy_log_regret=wmean(np.where(acc_cf, lrm, lrf)),
              model_log_risk=wmean(lkm), model_log_regret=wmean(lrm), fallback_log_risk=wmean(lkf),
              fallback_log_regret=wmean(lrf), deploy_tau=deploy_tau,
              policy_mse=wmean(np.where(acc_cf, rkm, rkf)), model_mse=wmean(rkm), fallback_mse=wmean(rkf))
    d["accepted_crossfit"] = acc_cf
    grp = []
    for col in ("family", "device"):
        for k_, e in d.groupby(col):
            grp.append(dict(group=col, value=k_, rows=int(len(e)), base_circuits=int(e.base_id.nunique()),
                            rejection_rate=float(1 - e.accepted_crossfit.mean())))
    return dict(curve=curve, folds=ops, operating=[op], groups=grp,
                rows=d[["fold", "task_id", "base_id", "B", "margin", "accepted_crossfit"]].to_dict("records"))


def budget_generalization(ds, families=("ridge", "hgb"), set_name: str = "deployable",
                          max_train=(None, 300000, 10000), k: int = 5, seed: int = 0, reps: int = 1000) -> list:
    """Models trained on the training circuits' rows with B ≤ a maximum (None: the full range) and evaluated at
    every budget of the held-out circuits (group-K-fold, so no circuit is on both sides): mean log10 regret per test
    budget, marked in range or above the training range (extrapolation). The budget-dependent fixed policy trained
    on the same rows is the reference (at an unseen budget it uses the nearest trained budget)."""
    out = []
    B = ds.col("B").astype(float)
    for mx in max_train:
        per = {}
        for _, tr, te in _folds(ds, k, seed):
            trr = tr if mx is None else tr[B[tr] <= mx]
            pols = [PO.SBSBudget()] + [PO.Model(f, FE.SETS[set_name], set_name, fixed=FIXED[f]) for f in families]
            for pol in pols:
                pol.fit(ds, trr)
                ch, _ = pol.choose(ds, te)
                v = per.setdefault(pol.name, np.full(len(ds.rows), np.nan))
                v[te] = EV.scores(ds, te, ch)["log_ratio"]
        for b in sorted(set(B.tolist())):
            sel = B == b
            ci = EV.bootstrap({p: v[sel] for p, v in per.items()}, ds.groups[sel], reps, seed)[0]
            for p in per:
                out.append(dict(max_train_B=("full" if mx is None else int(mx)), policy=p, B=int(b),
                                regime=("in range" if mx is None or b <= mx else "extrapolation"),
                                mean_log_regret=ci[p][0], ci_lo=ci[p][1], ci_hi=ci[p][2],
                                base_circuits=int(len(set(ds.groups[sel].tolist())))))
    return out


# ------------------------------------------------------------------ nested selective recommendation (1.7.0, R2)
def _margins(pol, ds, idx) -> tuple:
    """(choices, fallback-free margins z = (P₂ − P₁)/√(s₁² + s₂²)) of a fitted model policy on rows ``idx``."""
    P = pol.predict(ds, idx)
    ch, _ = pol.choose(ds, idx)
    rm = pol.rm
    z = np.full(len(idx), np.inf)
    for r, i in enumerate(idx):
        feas = [j for j in range(len(ds.protocols)) if ds.F[i, j] and np.isfinite(P[r, j])]
        o = sorted(feas, key=lambda j: P[r, j])
        if len(o) >= 2:
            s1 = rm.cv_rmse.get(ds.protocols[o[0]], np.nan)
            s2 = rm.cv_rmse.get(ds.protocols[o[1]], np.nan)
            z[r] = float((P[r, o[1]] - P[r, o[0]]) / math.sqrt(s1 ** 2 + s2 ** 2))
    return ch, z


def _choose_tau(z, lk_model, lk_fb, groups, grid: int = 41) -> float:
    """The threshold τ minimizing the whole-policy mean log10 risk (base circuits weighted equally) on the given
    rows; ties go to the larger τ (fewer acceptances). τ = +inf means "always the fallback"."""
    fin = z[np.isfinite(z)]
    taus = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(fin, np.linspace(0, 1, grid)) if len(fin)
                                     else []]))
    u = sorted(set(groups.tolist()))
    gi = np.array([u.index(g) for g in groups])
    cnt = np.bincount(gi, minlength=len(u)).astype(float)
    best_t, best_v = np.inf, np.inf
    for t in taus[::-1]:
        v = np.where(z >= t, lk_model, lk_fb)
        m = float(np.mean(np.bincount(gi, weights=v, minlength=len(u)) / cnt))
        if m < best_v - 1e-12:
            best_t, best_v = float(t), m
    return best_t


def nested_selective(ds, family: str = "hgb", set_name: str = "deployable+simulated", k: int = 5, inner_k: int = 4,
                     seed: int = 0, grid: int = 41, reps: int = 1000, random_draws: int = 200,
                     fixed: dict | None = None) -> dict:
    """Selective recommendation evaluated by genuinely nested cross-validation over base circuits (1.7.0; review R2).

    1.6.0 chose each outer fold's threshold on the other folds' out-of-fold rows — but the models that produced
    those rows had been trained on the target fold's labels, so the threshold depended indirectly on the outer test
    data. Here, for every outer fold (group-K-fold over base circuits):

    1. inner group-K-fold on the outer *training* rows only: the model and the fallback (SBS-B) are fitted on each
       inner training part and give margins z and log risks on the inner held-out rows;
    2. the threshold τ is chosen on those inner rows (whole-policy mean log10 risk, base circuits weighted equally);
    3. the model and the fallback are refitted on all outer training rows and the frozen rule "accept if z ≥ τ,
       else the fallback" is applied once to the outer test rows.

    Every base circuit is tested once. Reported (base-circuit weighted, bootstrap over base circuits): coverage, the
    whole-policy arithmetic MSE and mean log10 risk *including the fallback's loss*, the model alone, the fallback
    alone, SIZE-B and SBS, random rejection at the same coverage per fold (exact expectation, and the 2.5–97.5 %
    range over ``random_draws`` random acceptance sets), and the accepted rows' regret. ``fixed``: one model setting
    instead of the tuning grid (the tuning, when used, runs inside every inner and outer fit)."""
    from . import policies as PO
    rows, folds = [], []
    rng = np.random.default_rng(seed)
    for fold, tr, te in _folds(ds, k, seed):
        inner = EV.splits(ds.subset(tr), "same-distribution", k=inner_k, seed=seed + 1)
        zi, lkm_i, lkf_i, gi = [], [], [], []
        for _, itr, ite in inner:
            a, b = tr[itr], tr[ite]
            pol = PO.Model(family, FE.SETS[set_name], set_name, fixed=fixed).fit(ds, a)
            fb = PO.SBSBudget().fit(ds, a)
            ch, z = _margins(pol, ds, b)
            fch, _ = fb.choose(ds, b)
            zi.append(z), lkm_i.append(EV.scores(ds, b, ch)["log_risk"]), lkf_i.append(EV.scores(ds, b, fch)["log_risk"])
            gi.append(ds.groups[b])
        zi, lkm_i, lkf_i, gi = map(np.concatenate, (zi, lkm_i, lkf_i, gi))
        ok = np.isfinite(lkm_i) & np.isfinite(lkf_i)
        tau = _choose_tau(zi[ok], lkm_i[ok], lkf_i[ok], gi[ok], grid)
        pol = PO.Model(family, FE.SETS[set_name], set_name, fixed=fixed).fit(ds, tr)
        fb = PO.SBSBudget().fit(ds, tr)
        sz = PO.SizeBudget().fit(ds, tr)
        sbs = PO.SBS().fit(ds, tr)
        ch, z = _margins(pol, ds, te)
        fch, _ = fb.choose(ds, te)
        szc, _ = sz.choose(ds, te)
        sbc, _ = sbs.choose(ds, te)
        acc = z >= tau
        sm, sf = EV.scores(ds, te, ch), EV.scores(ds, te, fch)
        ss, sb = EV.scores(ds, te, szc), EV.scores(ds, te, sbc)
        folds.append(dict(fold=fold, tau=float(tau), inner_rows=int(ok.sum()), test_rows=int(len(te)),
                          test_base_circuits=int(len(set(ds.groups[te].tolist()))),
                          inner_base_circuits=int(len(set(gi.tolist()))), accepted=int(acc.sum())))
        for r, i in enumerate(te):
            row = ds.rows[i]
            rows.append(dict(fold=fold, task_id=row["task_id"], base_id=row["base_id"], family=row["family"],
                             device=row["device"], n=row["n"], B=row["B"], margin=float(z[r]), tau=float(tau),
                             accepted=bool(acc[r]), choice=ds.protocols[ch[r]], fallback=ds.protocols[fch[r]],
                             risk_model=float(sm["risk"][r]), risk_fallback=float(sf["risk"][r]),
                             risk_sizeb=float(ss["risk"][r]), risk_sbs=float(sb["risk"][r]),
                             risk_ref=float(sm["ref"][r]),
                             log_risk_model=float(sm["log_risk"][r]), log_risk_fallback=float(sf["log_risk"][r]),
                             log_risk_sizeb=float(ss["log_risk"][r]), log_risk_sbs=float(sb["log_risk"][r]),
                             log_regret_model=float(sm["log_ratio"][r]),
                             log_regret_fallback=float(sf["log_ratio"][r])))
    import pandas as pd
    d = pd.DataFrame(rows)
    d = d[np.isfinite(d.risk_model) & np.isfinite(d.risk_fallback)].reset_index(drop=True)
    d["risk_policy"] = np.where(d.accepted, d.risk_model, d.risk_fallback)
    d["log_risk_policy"] = np.where(d.accepted, d.log_risk_model, d.log_risk_fallback)
    g = d.base_id.to_numpy()
    u = sorted(set(g.tolist()))
    gidx = np.array([u.index(x) for x in g])
    cnt = np.bincount(gidx, minlength=len(u)).astype(float)

    def per(v):
        return np.bincount(gidx, weights=np.asarray(v, float), minlength=len(u)) / cnt
    # random rejection at the same coverage, fold by fold: exact expectation and random acceptance sets
    cov_f = d.groupby("fold").accepted.mean().to_dict()
    pacc = d.fold.map(cov_f).to_numpy(float)
    rnd_mse = pacc * d.risk_model + (1 - pacc) * d.risk_fallback
    rnd_log = pacc * d.log_risk_model + (1 - pacc) * d.log_risk_fallback
    draws_mse, draws_log = [], []
    for _ in range(int(random_draws)):
        a = np.zeros(len(d), bool)
        for f, c in cov_f.items():
            ix = np.flatnonzero(d.fold.to_numpy() == f)
            a[rng.choice(ix, size=int(round(c * len(ix))), replace=False)] = True
        draws_mse.append(np.mean(per(np.where(a, d.risk_model, d.risk_fallback))))
        draws_log.append(np.mean(per(np.where(a, d.log_risk_model, d.log_risk_fallback))))
    M = {"policy": per(d.risk_policy), "model": per(d.risk_model), "fallback (SBS-B)": per(d.risk_fallback),
         "SIZE-B": per(d.risk_sizeb), "SBS": per(d.risk_sbs), "random rejection": per(rnd_mse)}
    L = {"policy": per(d.log_risk_policy), "model": per(d.log_risk_model), "fallback (SBS-B)": per(d.log_risk_fallback),
         "SIZE-B": per(d.log_risk_sizeb), "SBS": per(d.log_risk_sbs), "random rejection": per(rnd_log)}
    picks = rng.integers(0, len(u), size=(int(reps), len(u)))
    summ = []
    for name in M:
        bm, bl = M[name][picks].mean(axis=1), L[name][picks].mean(axis=1)
        dm = (M[name] - M["fallback (SBS-B)"])[picks].mean(axis=1)
        dl = (L[name] - L["fallback (SBS-B)"])[picks].mean(axis=1)
        dr = (L[name] - L["random rejection"])[picks].mean(axis=1)
        summ.append(dict(policy=name, base_circuits=len(u), rows=int(len(d)),
                         coverage=float(np.mean(per(d.accepted))) if name == "policy" else
                         (float(np.mean(per(pacc))) if name == "random rejection" else
                          (1.0 if name == "model" else 0.0) if name in ("model", "fallback (SBS-B)") else math.nan),
                         mse=float(np.mean(M[name])), mse_lo=float(np.quantile(bm, 0.025)),
                         mse_hi=float(np.quantile(bm, 0.975)),
                         mean_log10_risk=float(np.mean(L[name])), log_lo=float(np.quantile(bl, 0.025)),
                         log_hi=float(np.quantile(bl, 0.975)),
                         diff_mse_vs_sbsb=float(np.mean(M[name] - M["fallback (SBS-B)"])),
                         diff_mse_vs_sbsb_lo=float(np.quantile(dm, 0.025)),
                         diff_mse_vs_sbsb_hi=float(np.quantile(dm, 0.975)),
                         diff_log_vs_sbsb=float(np.mean(L[name] - L["fallback (SBS-B)"])),
                         diff_log_vs_sbsb_lo=float(np.quantile(dl, 0.025)),
                         diff_log_vs_sbsb_hi=float(np.quantile(dl, 0.975)),
                         diff_log_vs_random=float(np.mean(L[name] - L["random rejection"])),
                         diff_log_vs_random_lo=float(np.quantile(dr, 0.025)),
                         diff_log_vs_random_hi=float(np.quantile(dr, 0.975)),
                         random_draws_mse_lo=float(np.quantile(draws_mse, 0.025)) if name == "random rejection"
                         else math.nan,
                         random_draws_mse_hi=float(np.quantile(draws_mse, 0.975)) if name == "random rejection"
                         else math.nan,
                         random_draws_log_lo=float(np.quantile(draws_log, 0.025)) if name == "random rejection"
                         else math.nan,
                         random_draws_log_hi=float(np.quantile(draws_log, 0.975)) if name == "random rejection"
                         else math.nan,
                         accepted_log_regret=float(d.log_regret_model[d.accepted].mean()) if name == "policy"
                         and d.accepted.any() else math.nan,
                         configuration="MODEL[%s|%s]" % (family, set_name), evaluation="nested group-CV "
                         "(outer %d, inner %d folds)" % (k, inner_k)))
    # coverage–risk curve on the outer rows (descriptive: every τ, no selection)
    curve = []
    z = d.margin.to_numpy(float)
    fin = z[np.isfinite(z)]
    for t in np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(fin, np.linspace(0, 1, grid)) if len(fin)
                                       else []])):
        a = z >= t
        cov = float(np.mean(per(a)))
        curve.append(dict(tau=float(t), coverage=cov,
                          policy_mse=float(np.mean(per(np.where(a, d.risk_model, d.risk_fallback)))),
                          policy_log10_risk=float(np.mean(per(np.where(a, d.log_risk_model, d.log_risk_fallback)))),
                          random_policy_mse=float(cov * np.mean(M["model"]) + (1 - cov) * np.mean(M["fallback (SBS-B)"])),
                          random_policy_log10_risk=float(cov * np.mean(L["model"]) + (1 - cov) *
                                                         np.mean(L["fallback (SBS-B)"])),
                          accepted_log_regret=float(d.log_regret_model[a].mean()) if a.any() else math.nan))
    ok = np.isfinite(d.log_risk_model) & np.isfinite(d.log_risk_fallback)
    deploy_tau = _choose_tau(d.margin.to_numpy(float)[ok], d.log_risk_model.to_numpy(float)[ok],
                             d.log_risk_fallback.to_numpy(float)[ok], g[ok], grid)
    op = dict(deploy_tau=float(deploy_tau), rule="accept the model's choice when its margin z ≥ τ, else SBS-B",
              chosen_on="out-of-fold rows of the whole development set (the outer folds' models)",
              performance="estimated by the nested procedure (summary), not on the rows τ was chosen on",
              configuration="MODEL[%s|%s]" % (family, set_name))
    return dict(summary=summ, folds=folds, rows=d.to_dict("records"), curve=curve, operating=[op])


# ------------------------------------------------------------------ design-level tables (1.7.0, R1)
def design_diagnostics(labels_dir: str, budgets=(10000, 1000000, 30000000), task_ids=None) -> list:
    """One row per (task, learning protocol, label half, design, budget): the design's identity and diagnostics
    (training-label spread and range, rank resolved above shot noise, singular values and condition of the exact
    training features, coefficient ℓ1 norm and nonzeros, fallback, and the research-only oracle diagnostics: the
    exact-feature fit's error and whether μ lies inside the training labels' range) with the design's mean loss and
    mean estimate at the budget (evaluation and selection halves). The input of the design-level figures and of the
    investigation of severe learning-protocol failures."""
    from . import labels as LB
    from . import protocols as PR
    out = []
    for tid in (task_ids if task_ids is not None else LB.task_ids(labels_dir)):
        rec = LB.load_record(labels_dir, tid)
        if rec.get("status") != "ok":
            continue
        arr = LB.load_arrays(labels_dir, tid, rec)
        mu = float(rec["ideal"])
        recs = rec.get("design_records") or {}
        for i, k in enumerate(rec["protocols"]):
            if not PR.PORTFOLIO[k].learning:
                continue
            by_idx = {}
            for d in recs.get(k, []):
                by_idx[(int(d["half"]), int(d["index"]))] = d
            for j, B in enumerate(rec["budgets"]):
                if int(B) not in budgets:
                    continue
                a = rec["allocations"][i * len(rec["budgets"]) + j]
                if not a.get("feasible"):
                    continue
                for h in (0, 1):
                    x = arr["est"][h, i, j]
                    dd = arr["design"][h, i, j]
                    fl = arr["flags"][h, i, j]
                    for kidx in np.unique(dd[dd >= 0]):
                        m = dd == kidx
                        e = np.where(np.isfinite(x[m]), x[m], 0.0)
                        dr = by_idx.get((h, int(kidx))) or by_idx.get((-1, int(kidx))) or {}
                        dg = dr.get("diagnostics") or {}
                        out.append(dict(task_id=tid, base_id=rec["base_id"], family=rec["spec"]["family"],
                                        n=rec["spec"]["n"], device=rec["spec"]["device"], protocol=k, B=int(B),
                                        half="sel" if h == 0 else "eval", design=int(kidx),
                                        design_id=dr.get("design_id"), seed=dr.get("seed"),
                                        status=dr.get("status", "ok"), fallback=dr.get("fallback", ""),
                                        trials=int(m.sum()), mean_loss=float(np.mean((e - mu) ** 2)),
                                        mean_estimate=float(np.mean(e)), ideal=mu,
                                        fallback_rate=float(fl.any(axis=0)[m].mean()),
                                        **{"diag_" + kk: vv for kk, vv in dg.items() if kk != "warning"}))
    return out


def design_variance(labels_table) -> list:
    """Per learning protocol and budget: how much of the evaluation-half label uncertainty is between designs —
    median ratio of the design-level to the trial-level standard error, median share of the between-design variance
    component, and the number of designs (from ``labels.csv``)."""
    lab = labels_table
    out = []
    for (p, B), e in lab[lab.protocol.isin(["CDR", "NIL"])].groupby(["protocol", "B"]):
        e = e[np.isfinite(e.se_mse_trial_eval) & (e.se_mse_trial_eval > 0)]
        if e.empty:
            continue
        vb, vw = e.var_between_design_eval, e.var_within_design_eval
        m = e.r.astype(float) / np.maximum(e.designs_eval.astype(float), 1.0)      # trials per design
        out.append(dict(protocol=p, B=int(B), tasks=int(len(e)), designs=int(np.median(e.designs_eval)),
                        median_se_ratio=float(np.median(e.se_mse_design_eval / e.se_mse_trial_eval)),
                        p90_se_ratio=float(np.quantile(e.se_mse_design_eval / e.se_mse_trial_eval, 0.9)),
                        median_between_share=float(np.nanmedian(vb / (vb + vw))),
                        median_design_share_of_label_variance=float(np.nanmedian(vb / (vb + vw / m))),
                        share_tasks_between_dominant=float(np.nanmean(vb > vw))))
    return out


def failure_profile(choices, labels, policies=None) -> list:
    """Failure-aware summary of every policy in a choices table (evaluation half): the arithmetic mean MSE (base
    circuits weighted equally), the share of the total squared error contributed by the worst 1 % and 5 % of rows,
    the rate of severe misselection (risk > 10× the reference's) and of degradation beyond RAW, per budget and over
    all budgets."""
    from .evaluate import _per_policy_from_choices, wmean
    per = _per_policy_from_choices(choices, labels)
    pols = policies or list(per)
    g = choices.base_id.to_numpy()
    B = choices.B.to_numpy(int)
    raw = per.get("RAW")
    out = []
    for b in sorted(set(B.tolist())) + ["all"]:
        sel = np.ones(len(B), bool) if b == "all" else B == b
        for p in pols:
            r = per[p]["risk"][sel]
            ok = np.isfinite(r)
            if not ok.any():
                continue
            srt = np.sort(r[ok])[::-1]
            tot = srt.sum()
            k1, k5 = max(1, int(round(0.01 * len(srt)))), max(1, int(round(0.05 * len(srt))))
            lr = per[p]["log_ratio"][sel]
            d = dict(policy=p, B=b, rows=int(ok.sum()), base_circuits=int(len(set(g[sel].tolist()))),
                     mse=wmean(r, g[sel]), worst1_share=float(srt[:k1].sum() / tot) if tot > 0 else math.nan,
                     worst5_share=float(srt[:k5].sum() / tot) if tot > 0 else math.nan,
                     severe_rate=wmean((lr > 1.0).astype(float), g[sel]),
                     max_log_regret=float(np.nanmax(lr)))
            if raw is not None:
                d["worse_than_raw_2x"] = wmean((np.log10((r + 1e-6) / (raw["risk"][sel] + 1e-6)) > math.log10(2))
                                               .astype(float), g[sel])
            out.append(d)
    return out
