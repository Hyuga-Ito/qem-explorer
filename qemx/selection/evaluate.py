r"""Policy evaluation (1.6.0).

Conventions, fixed before any result was computed:

* **Scoring half.** A policy's choice on a test row is scored by the *evaluation* half R̂_eval of the chosen
  protocol; anything that chooses by measured risks (the all-candidate reference) chooses on the selection half.
  The minimum of noisy estimates is biased low (optimizer's curse); scoring on independent trials removes it.
* **Reference.** a*(row) = argmin over feasible protocols of R̂_sel (the VBS identified on the selection half);
  regret Δ = R̂_eval(π) − R̂_eval(a*) and the ratio ρ = (R̂_eval(π) + ε0)/(R̂_eval(a*) + ε0) (theory companion
  Eq. 1.18; the comparator is the declared candidate set, not every conceivable protocol).
* **Aggregate.** The primary score of a policy at a budget is the mean over test base circuits (each base circuit
  weighs the same; its rows averaged first) of log10(R̂_eval(π) + ε0), ε0 = 1e-6 — the log of the geometric mean of
  *risk plus the floor*, robust to the few tasks whose risk dominates an arithmetic mean. A difference Δ in it is a
  ratio 10^Δ of geometric means of (MSE + 1e-6), not of MSE itself (1.8.0 wording; finding F6). The arithmetic
  policy MSE and, from 1.8.0, declared tail metrics (``tail_metrics``, ``p_severe_vs_raw``) are reported beside it.
* **Gap closed** = (m_SBS − m_π)/(m_SBS − m_VBS) with m the primary aggregate: 1 is the per-task oracle, 0 the
  training-selected single best protocol, below 0 worse than it (ASlib orientation; Lindauer et al. use the opposite).
* **Uncertainty.** 95 % percentile intervals from a bootstrap over test base circuits (2 000 resamples; paired
  between policies, so differences use the same resamples). Finite-trial uncertainty of the labels is separate:
  ``label_noise`` redraws each R̂_eval from its standard error.
* **Abstention.** For abstaining policies: coverage κ (accepted share), accepted-task risk (rows accepted) and the
  fallback-inclusive risk over all rows (Eqs. 1.25–1.26), with κ per family and device.
* **Pooling.** Results of a split are reported out-of-fold pooled (``pooled``): every base circuit is tested once,
  all test rows are scored together and resampled in one bootstrap; per-part summaries are kept beside them.
* **Optimism of a selected configuration.** The best of several model configurations, picked on the same pooled
  scores, is reported with its bootstrap bias-corrected estimate (``bbc_cv``); the frozen final test removes the
  optimism entirely.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .models import EPS0


SEVERE_VS_RAW = 10.0        # 1.8.0: severe degradation = risk > 10 × RAW's risk on the same row (declared)
SEVERE_ABS = 0.1            # 1.8.0: severe absolute error = risk > 0.1 (RMSE > 0.32 on an observable with |O| ≤ 1)
TAIL_SHARE = 0.05           # 1.8.0: concentration = share of the total loss from the worst 5 % of base circuits


def tail_metrics(risk, groups) -> dict:
    """Arithmetic-scale tail of a policy's chosen risks (1.8.0; finding F7), beside the shifted-log primary metric:
    base-circuit-weighted probability of a severe absolute risk, the 95 % quantile of the risk, and the share of the
    total (circuit-averaged) loss carried by the worst TAIL_SHARE of base circuits — how concentrated the loss is."""
    r = np.asarray(risk, float)
    g = np.asarray(groups)
    m, _ = _by_group(r, g)
    m = m[np.isfinite(m)]
    if not m.size:
        return dict(p_severe_abs=math.nan, q95_risk=math.nan, worst5_loss_share=math.nan)
    k = max(1, int(math.ceil(TAIL_SHARE * m.size)))
    tot = float(m.sum())
    sev = np.where(np.isfinite(r), (r > SEVERE_ABS).astype(float), np.nan)    # a missing risk is not "not severe"
    return dict(p_severe_abs=wmean(sev, g), q95_risk=wquantile(r, g, 0.95),
                worst5_loss_share=float(np.sort(m)[::-1][:k].sum() / tot) if tot > 0 else 0.0)


def scores(ds, idx, chosen) -> dict:
    """Per-row scores of a choice vector on rows ``idx``."""
    idx = np.asarray(idx)
    chosen = np.asarray(chosen)
    ref = np.array([int(np.argmin(np.where(ds.F[i], ds.Y_sel[i], np.inf))) for i in idx])
    r_pi = np.array([ds.Y_eval[i, j] if j >= 0 else np.nan for i, j in zip(idx, chosen)])
    r_ref = np.array([ds.Y_eval[i, j] for i, j in zip(idx, ref)])
    return dict(risk=r_pi, ref=r_ref, regret=r_pi - r_ref,
                log_ratio=np.log10((r_pi + EPS0) / (r_ref + EPS0)), log_risk=np.log10(r_pi + EPS0),
                ref_index=ref)


def group_weights(groups) -> np.ndarray:
    """Row weights that give every base circuit the same total weight (1/rows of its circuit, normalized)."""
    g = np.asarray(groups)
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.sum()


def wmean(vals, groups) -> float:
    """Base-circuit-weighted mean (each circuit's rows averaged first; rows with NaN dropped within a circuit)."""
    m, _ = _by_group(vals, groups)
    return float(np.nanmean(m)) if np.isfinite(m).any() else math.nan


def wquantile(vals, groups, q: float) -> float:
    """Base-circuit-weighted quantile: every circuit has total weight 1 spread over its rows (for balanced designs
    the ordinary row quantile)."""
    v = np.asarray(vals, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    if not ok.any():
        return math.nan
    w = group_weights(g[ok])
    o = np.argsort(v[ok], kind="stable")
    cw = np.cumsum(w[o])
    return float(v[ok][o][min(int(np.searchsorted(cw, q * cw[-1] - 1e-12)), len(o) - 1)])


def _by_group(vals, groups) -> np.ndarray:
    """Mean per group (base circuit), in sorted group order."""
    g = np.asarray(groups)
    u = sorted(set(g.tolist()))
    return np.array([np.nanmean(np.asarray(vals, float)[g == k]) for k in u]), u


def bootstrap(values_by_policy: dict, groups, reps: int = 2000, seed: int = 0) -> dict:
    """Paired percentile intervals of the group-weighted mean of each policy's per-row values."""
    g = np.asarray(groups)
    u = sorted(set(g.tolist()))
    M = {p: np.array([np.nanmean(np.asarray(v, float)[g == k]) for k in u]) for p, v in values_by_policy.items()}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(u), size=(int(reps), len(u)))
    out = {}
    for p, m in M.items():
        bs = np.nanmean(m[picks], axis=1)
        out[p] = (float(np.nanmean(m)), float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975)))
    return out, M, picks


def _inference(circuits: int) -> str:
    """What an interval over ``circuits`` independent base circuits can mean (1.9.1; independent review §4): with
    one circuit the bootstrap over circuits resamples one group, so its interval is degenerate and says nothing about
    between-circuit variability — such a stratum (e.g. one part of a held-out-noise or joint split) is descriptive."""
    if circuits < 2:
        return "descriptive: one base circuit (no between-circuit interval)"
    if circuits < 5:
        return "few base circuits (%d): interval unreliable, descriptive" % circuits
    return "bootstrap over %d base circuits" % circuits


def summarize_scores(per_policy: dict, groups, B, reps: int = 2000, seed: int = 0) -> list:
    """Per policy and per budget (and over all budgets): base-circuit-weighted mean log10 risk with a bootstrap
    interval over base circuits, the arithmetic policy MSE, regret tails, coverage, the gap closed and the paired
    difference to SBS. ``per_policy[name]`` holds per-row arrays ``log_risk``, ``risk``, ``log_ratio``,
    ``accepted``; ``groups`` and ``B`` are per-row base-circuit ids and budgets. ``inference`` (1.9.1) marks rows
    over fewer than five base circuits as descriptive (``_inference``)."""
    groups, B = np.asarray(groups), np.asarray(B)
    budgets = sorted(set(B.tolist()))
    summary = []
    raw = per_policy.get("RAW")
    for b in budgets + ["all"]:
        sel = np.ones(len(B), bool) if b == "all" else (B == b)
        if not sel.any():
            continue
        vals = {p: v["log_risk"][sel] for p, v in per_policy.items()}
        ci, M, picks = bootstrap(vals, groups[sel], reps, seed)
        ci_lr = bootstrap({p: v["log_ratio"][sel] for p, v in per_policy.items()}, groups[sel], reps, seed)[0]
        m_sbs, m_vbs = M.get("SBS"), M.get("ALL-CANDIDATE")
        for p, v in per_policy.items():
            gs = groups[sel]
            lr_s = v["log_ratio"][sel]
            d = dict(policy=p, B=b, rows=int(sel.sum()), base_circuits=len(set(gs.tolist())),
                     mean_log10_risk=ci[p][0], ci_lo=ci[p][1], ci_hi=ci[p][2],
                     policy_mse=wmean(v["risk"][sel], gs),              # 1.7.0: base-circuit weighted (S5)
                     policy_mse_rows=float(np.nanmean(v["risk"][sel])),  # the 1.6.0 row mean, for comparison
                     median_log_ratio=wquantile(lr_s, gs, 0.5),
                     p_regret_gt_10pct=wmean((lr_s > math.log10(1.1)).astype(float), gs),
                     p_regret_gt_2x=wmean((lr_s > math.log10(2.0)).astype(float), gs),
                     coverage=wmean(np.asarray(v["accepted"][sel], float), gs), weighting="base circuits",
                     inference=_inference(len(set(gs.tolist()))))
            if m_sbs is not None and m_vbs is not None:
                m = M[p]
                gap = (np.nanmean(m_sbs) - np.nanmean(m)) / (np.nanmean(m_sbs) - np.nanmean(m_vbs)) \
                    if abs(np.nanmean(m_sbs) - np.nanmean(m_vbs)) > 1e-12 else math.nan
                bs = []
                for pk in picks:
                    den = np.nanmean(m_sbs[pk]) - np.nanmean(m_vbs[pk])
                    bs.append((np.nanmean(m_sbs[pk]) - np.nanmean(m[pk])) / den if abs(den) > 1e-12 else np.nan)
                d.update(gap_closed=float(gap), gap_lo=float(np.nanquantile(bs, 0.025)),
                         gap_hi=float(np.nanquantile(bs, 0.975)))
                diff = m - m_sbs
                bsd = np.nanmean(diff[picks], axis=1)
                d.update(diff_vs_sbs=float(np.nanmean(diff)), diff_lo=float(np.nanquantile(bsd, 0.025)),
                         diff_hi=float(np.nanquantile(bsd, 0.975)))
            for ref, tag in (("SBS-B", "sbsb"), ("SIZE-B", "sizeb")):      # the strongest fixed baselines
                if ref in M:
                    diff = M[p] - M[ref]
                    bsd = np.nanmean(diff[picks], axis=1)
                    d.update({"diff_vs_" + tag: float(np.nanmean(diff)),
                              "diff_vs_%s_lo" % tag: float(np.nanquantile(bsd, 0.025)),
                              "diff_vs_%s_hi" % tag: float(np.nanquantile(bsd, 0.975))})
            g_ = groups[sel]
            lr_ = v["log_ratio"][sel]
            d.update(mean_log_regret=ci_lr[p][0], log_regret_lo=ci_lr[p][1], log_regret_hi=ci_lr[p][2],
                     p90_log_regret=wquantile(lr_, g_, 0.9),
                     tol_success_10pct=float(np.nanmean(_by_group(lr_ <= math.log10(1.1), g_)[0])),
                     tol_success_2x=float(np.nanmean(_by_group(lr_ <= math.log10(2.0), g_)[0])))
            if "regret" in v:
                rg = np.asarray(v["regret"], float)[sel]
                d.update(mean_regret=float(np.nanmean(_by_group(rg, g_)[0])), median_regret=wquantile(rg, g_, 0.5),
                         p90_regret=wquantile(rg, g_, 0.9), p_negative_regret=wmean((rg < 0).astype(float), g_))
            if raw is not None:
                worse = np.log10((v["risk"][sel] + EPS0) / (raw["risk"][sel] + EPS0)) > math.log10(1.1)
                d.update(raw_degradation=float(np.nanmean(_by_group(worse.astype(float), g_)[0])))
                # 1.8.0 (finding F7), declared before any 1.8.0 result: a severe degradation is a chosen risk more
                # than ten times RAW's on the same row
                rp, rr = np.asarray(v["risk"][sel], float), np.asarray(raw["risk"][sel], float)
                sev = np.where(np.isfinite(rp) & np.isfinite(rr), (rp > SEVERE_VS_RAW * rr).astype(float), np.nan)
                d.update(p_severe_vs_raw=float(np.nanmean(_by_group(sev, g_)[0])))
            d.update(tail_metrics(v["risk"][sel], g_))
            if "fail" in v:
                d.update(fail_share=float(np.nanmean(_by_group(np.asarray(v["fail"], float)[sel], g_)[0])))
            if v["accepted"][sel].size and not v["accepted"][sel].all():
                a = np.asarray(v["accepted"][sel], bool)
                d.update(accepted_log10_risk=wmean(v["log_risk"][sel][a], gs[a]) if a.any() else math.nan)
            summary.append(d)
    return summary


def evaluate(ds, train_idx, test_idx, policies, reps: int = 2000, seed: int = 0) -> dict:
    """Fit every policy on ``train_idx`` and score it on ``test_idx``. Returns per-policy, per-budget summaries and
    the per-row choices (for figures and drill-down)."""
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    rows, per_policy = [], {}
    for pol in policies:
        pol.fit(ds, train_idx)
        ch, acc = pol.choose(ds, test_idx)
        sc = scores(ds, test_idx, ch)
        fail = np.array([ds.fail[i, j] if j >= 0 else np.nan for i, j in zip(test_idx, ch)], float)
        per_policy[pol.name] = dict(chosen=ch, accepted=acc, fail=fail, **sc)
    groups = ds.groups[test_idx]
    B = ds.col("B")[test_idx]
    summary = summarize_scores(per_policy, groups, B, reps, seed)
    for k, i in enumerate(test_idx):
        r = ds.rows[i]
        rows.append(dict(task_id=r["task_id"], base_id=r["base_id"], family=r["family"], n=r["n"],
                         device=r["device"], B=r["B"],
                         **{"%s|choice" % p: ds.protocols[v["chosen"][k]] if v["chosen"][k] >= 0 else None
                            for p, v in per_policy.items()},
                         **{"%s|log_ratio" % p: float(v["log_ratio"][k]) for p, v in per_policy.items()},
                         **{"%s|accepted" % p: bool(v["accepted"][k]) for p, v in per_policy.items()}))
    return dict(summary=summary, rows=rows, per_policy=per_policy)


def _per_policy_from_choices(choices, labels) -> dict:
    """Per-row arrays of every policy in a saved choices table, with risks looked up in the saved labels table
    (evaluation half; the per-task reference is the ALL-CANDIDATE choice, identified on the selection half)."""
    risk = {(t, int(b), p): float(m) for t, b, p, m in
            labels[["task_id", "B", "protocol", "mse_eval"]].itertuples(index=False)}
    pols = [c[: -len("|choice")] for c in choices.columns if c.endswith("|choice")]
    tid, Bs = choices.task_id.tolist(), [int(b) for b in choices.B]

    def look(col):
        return np.array([risk.get((t, b, c), np.nan) if isinstance(c, str) else np.nan
                         for t, b, c in zip(tid, Bs, choices[col].tolist())])
    ref = look("ALL-CANDIDATE|choice") if "ALL-CANDIDATE|choice" in choices.columns else None
    # the share of trials with any failure or fallback flag (1.7.0: the union, ``fail_any_eval``); a 1.6.0 labels
    # table has only the individual rates, whose sum double-counts overlapping flags: the share is then unknown
    fail_of = {(t, int(b), p): float(f) for t, b, p, f in zip(labels.task_id, labels.B, labels.protocol,
                                                               labels["fail_any_eval"])} \
        if "fail_any_eval" in labels.columns else {}
    out = {}
    for p in pols:
        r = look(p + "|choice")
        acc = choices[p + "|accepted"].astype(bool).to_numpy() if p + "|accepted" in choices.columns \
            else np.ones(len(r), bool)
        lr = np.log10((r + EPS0) / (ref + EPS0)) if ref is not None else choices[p + "|log_ratio"].to_numpy(float)
        out[p] = dict(risk=r, log_risk=np.log10(r + EPS0), log_ratio=lr, accepted=acc)
        if ref is not None:
            out[p]["regret"] = r - ref
        if fail_of:
            out[p]["fail"] = np.array([fail_of.get((t, b, c), np.nan) if isinstance(c, str) else np.nan
                                       for t, b, c in zip(tid, Bs, choices[p + "|choice"].tolist())])
    return out


def pooled(choices, labels, reps: int = 2000, seed: int = 0) -> list:
    """The out-of-fold pooled evaluation of a split: every test row of every part (each base circuit is tested once
    in a K-fold or leave-one-group-out split) scored together, with one bootstrap over all test base circuits. Its
    intervals reflect the sampling of test circuits, not the variability of the training sets."""
    per = _per_policy_from_choices(choices, labels)
    return summarize_scores(per, choices.base_id.to_numpy(), choices.B.to_numpy(int), reps, seed)


def bbc_cv(choices, labels, candidates=None, reps: int = 2000, seed: int = 0, ref: str = "SBS") -> dict:
    """Bootstrap bias-corrected cross-validation (BBC-CV; Tsamardinos, Greasidou & Borboudakis 2018, Machine
    Learning 107, 1895–1922) of the step "keep the model configuration with the lowest pooled out-of-fold mean log10
    risk". Reporting the winner's own score is optimistic: it was selected on that score. Per bootstrap sample of
    base circuits the configuration with the lowest in-bag mean is selected and scored on the out-of-bag base
    circuits (its mean log10 risk and its paired difference to ``ref``); the mean over samples estimates the
    performance of the selection procedure, and its difference to the naive winner's score estimates the optimism.
    ``candidates``: the policies to choose from (default: every ``MODEL[`` policy)."""
    per = _per_policy_from_choices(choices, labels)
    cands = candidates or [p for p in per if p.startswith("MODEL[")]
    g = choices.base_id.to_numpy()
    u = sorted(set(g.tolist()))
    M = {p: np.array([np.nanmean(per[p]["log_risk"][g == k]) for k in u]) for p in set(cands) | {ref}}
    naive = min(cands, key=lambda p: np.nanmean(M[p]))
    rng = np.random.default_rng(seed)
    oob_mean, oob_diff, picked = [], [], []
    for _ in range(int(reps)):
        inbag = rng.integers(0, len(u), size=len(u))
        oob = np.setdiff1d(np.arange(len(u)), inbag)
        if oob.size == 0:
            continue
        best = min(cands, key=lambda p: np.nanmean(M[p][inbag]))
        picked.append(best)
        oob_mean.append(np.nanmean(M[best][oob]))
        oob_diff.append(np.nanmean(M[best][oob] - M[ref][oob]))
    oob_mean, oob_diff = np.array(oob_mean), np.array(oob_diff)
    freq = {p: picked.count(p) / max(len(picked), 1) for p in cands}
    return dict(candidates=len(cands), base_circuits=len(u), naive_best=naive,
                naive_mean_log10_risk=float(np.nanmean(M[naive])),
                naive_diff_vs_ref=float(np.nanmean(M[naive] - M[ref])),
                bbc_mean_log10_risk=float(np.nanmean(oob_mean)), bbc_lo=float(np.nanquantile(oob_mean, 0.025)),
                bbc_hi=float(np.nanquantile(oob_mean, 0.975)), bbc_diff_vs_ref=float(np.nanmean(oob_diff)),
                bbc_diff_lo=float(np.nanquantile(oob_diff, 0.025)), bbc_diff_hi=float(np.nanquantile(oob_diff, 0.975)),
                p_diff_below_0=float(np.mean(oob_diff < 0)),
                optimism=float(np.nanmean(oob_mean) - np.nanmean(M[naive])), ref=ref, samples=int(len(picked)),
                selection_frequency=json.dumps(dict(sorted(freq.items(), key=lambda kv: -kv[1])[:5])))


def splits(ds, kind: str, *, k: int = 5, seed: int = 0, **kw) -> list:
    """(name, train rows, test rows) for a generalization test.

    same-distribution  group-K-fold over base circuits
    family             leave one family out (a base circuit has one family)
    device             train on every device but ``test_device``, crossed with group-K-fold over base circuits
    width              train on n < ``min_test_n``, test on n ≥ it (a base circuit has one width)
    budget-extrap      train on B ≤ ``max_train_B``, test above (extrapolation in budget), crossed with the folds
    budget-interp      train on every other budget of the grid, test on the rest, crossed with the folds
    device-base        (1.9.0) train on every *base* device but ``test_device`` (all its noise settings held out
                       together), crossed with the folds — a new device, not a new noise setting of a known one
    noise              (1.9.0) train on every noise setting (the device modifiers after '@', or ``native``) but
                       ``test_noise``, crossed with the folds — a held-out noise strength or composition
    joint              (1.9.0) hold out a family *and* a noise setting together: test rows of family f under noise
                       c; train rows of other families under other noise settings (``test_family``, ``test_noise``;
                       default every pair with both sides non-empty) — a joint circuit/noise shift

    No base circuit is on both sides of any split. Noise settings are calibration-supplied snapshot modifiers (what
    a user's calibration shows), never hidden execution-law parameters.
    """
    from .models import group_kfold
    allr = np.arange(len(ds.rows))
    g = ds.groups
    if kind == "same-distribution":
        return [("fold %d" % (f + 1), tr, te) for f, (tr, te) in enumerate(group_kfold(g, k, seed))]
    if kind == "family":
        fam = ds.col("family")
        return [("family=%s" % f, allr[fam != f], allr[fam == f]) for f in sorted(set(fam.tolist()))]
    def crossed(name, train_ok, test_ok):
        """The held-out condition crossed with group-K-fold over base circuits: a base circuit whose variants or
        budgets are tested never contributes training rows (variants on other devices and the same circuit at other
        budgets share its ancestry). Every base circuit is tested in exactly one fold."""
        out = []
        for f, (tr, te) in enumerate(group_kfold(g, k, seed)):
            tr, te = tr[train_ok[tr]], te[test_ok[te]]
            if len(tr) and len(te):
                out.append(("%s|fold %d" % (name, f + 1), tr, te))
        return out
    if kind == "device":
        dev = ds.col("device")
        td = kw.get("test_device")
        devs = [td] if td else sorted(set(dev.tolist()))
        return [p for d in devs if (dev != d).any() for p in crossed("device=%s" % d, dev != d, dev == d)]
    if kind == "device-base":
        base = np.array([str(d).split("@")[0] for d in ds.col("device")])
        td = kw.get("test_device")
        devs = [td] if td else sorted(set(base.tolist()))
        return [p for d in devs if (base != d).any() for p in crossed("device-base=%s" % d, base != d, base == d)]
    if kind == "noise":
        nz = noise_setting(ds.col("device"))
        tn = kw.get("test_noise")
        sets = [tn] if tn else sorted(set(nz.tolist()))
        return [p for c in sets if (nz != c).any() for p in crossed("noise=%s" % c, nz != c, nz == c)]
    if kind == "joint":
        nz = noise_setting(ds.col("device"))
        fam = ds.col("family")
        fs = [kw["test_family"]] if kw.get("test_family") else sorted(set(fam.tolist()))
        cs = [kw["test_noise"]] if kw.get("test_noise") else sorted(set(nz.tolist()))
        out = []
        for f in fs:
            for c in cs:
                te = allr[(fam == f) & (nz == c)]
                tr = allr[(fam != f) & (nz != c)]
                if len(te) and len(tr):
                    out.append(("family=%s&noise=%s" % (f, c), tr, te))
        return out
    if kind == "width":                      # a base circuit has one width: no variant crosses this split
        n = ds.col("n").astype(int)
        m = int(kw.get("min_test_n", n.max()))
        return [("n>=%d" % m, allr[n < m], allr[n >= m])]
    if kind == "budget-extrap":
        B = ds.col("B").astype(float)
        mx = float(kw.get("max_train_B", 1e5))
        return crossed("B>%g" % mx, B <= mx, B > mx)
    if kind == "budget-interp":
        B = ds.col("B").astype(int)
        grid = sorted(set(B.tolist()))
        tr_b = np.isin(B, grid[0::2])
        return crossed("interp", tr_b, ~tr_b)
    raise ValueError(kind)


def noise_setting(devices) -> np.ndarray:
    """The noise setting of each device specification: its modifiers after '@' in a canonical order, or 'native'."""
    out = []
    for d in devices:
        d = str(d)
        if "@" not in d:
            out.append("native")
        else:
            out.append(",".join(sorted(x for x in d.split("@", 1)[1].split(",") if not x.startswith("eng="))) or
                       "native")
    return np.array(out)


def label_noise(ds, idx, chosen, draws: int = 200, seed: int = 0) -> tuple:
    """Spread (2.5–97.5 %) of the group-weighted mean log10 risk when each R̂_eval is redrawn from N(R̂, SE²)
    (clipped at 0): the part of the uncertainty a bootstrap over base circuits does not contain."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    r = np.array([ds.Y_eval[i, j] for i, j in zip(idx, chosen)])
    s = np.array([ds.SE_eval[i, j] for i, j in zip(idx, chosen)])
    g = ds.groups[idx]
    out = []
    for _ in range(int(draws)):
        x = np.clip(r + rng.normal(size=len(r)) * s, 0, None)
        out.append(np.nanmean(_by_group(np.log10(x + EPS0), g)[0]))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))
