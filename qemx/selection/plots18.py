r"""Supplementary figures added in 1.8.0 (``docs/selection/FIGURES.md``, section "1.8.0 additions"), drawn — like every
figure of the study — only from saved tables of a run directory, never from a live computation:

* ``supp_deployment_policy`` — what the deployed application policy did on the once-only confirmation set (model
  choice, preflight skips, margin and out-of-distribution fallbacks, execution fallbacks of the final candidate,
  classical cost), and its full-policy result beside the model-component benchmark (finding F1);
* ``supp_portfolio_expansion`` — the original eight candidates against the 17 of portfolio-1.8 on identical tasks and
  labels: the benefit of more candidates separated from the benefit of learning to select them;
* ``supp_root_audit`` — the crossed circuit × preparation-root audit (stage 2): label stability across independent
  preparation roots against the labels' own standard errors, ranking stability, and the execution contract;
* ``supp_tails`` — complete loss distributions relative to RAW, severe-degradation rates per candidate and budget,
  loss concentration, and the contribution of single training designs to learning-protocol losses (finding F7).

Conventions as in ``qemx.selection.plots``: base circuits weigh equally (variants averaged first); log10 values use
log10(MSE + 10⁻⁶); every figure states its status (confirmatory, exploratory, descriptive) and that its observations
are simulated.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import textio as TX
from .figures import LABEL_SIM, PROTOCOL_COLORS, PROTOCOL_MARKERS, POLICY_STYLE, _plt, _save, _seq_cmap
from .models import EPS0

# outcome categories of a deployed recommendation, in a fixed order with fixed colours (Okabe–Ito and a dark neutral;
# figures.palette_report: smallest pairwise ΔE×100 15.6 normal, 9.6 protan/deutan, 8.5 tritan; kept by
# tests/test_plots18.py)
OUTCOMES = (
    ("accepted", "model's top-ranked candidate accepted", "#0072B2"),
    ("accepted-after-skip", "accepted after skipping unverified/infeasible candidates", "#56B4E9"),
    ("margin-fallback", "fallback SBS-B: margin below τ", "#E69F00"),
    ("ood-fallback", "fallback SBS-B: out of distribution", "#D55E00"),
    ("no-model", "fallback SBS-B: no prediction", "#CC79A7"),
    ("unsupported", "unsupported input (guard)", "#4D4D4D"),
)
OUTCOME_LABEL = {k: lab for k, lab, _ in OUTCOMES}
OUTCOME_COLOR = {k: c for k, _, c in OUTCOMES}
NEW_18 = ("RO-TREX", "RO-TW", "RO-DD", "ZNE-L-TW", "ZNE-E-B", "PEC-TW", "FR", "SV", "ZNE-L∘SV")


def outcome(row) -> str:
    """Outcome category of one replayed recommendation (a row of ``replay_<bundle>.csv``)."""
    st = str(row.get("status", ""))
    if st == "unsupported":
        return "unsupported"
    if st == "out-of-distribution":
        return "ood-fallback"
    if st == "no-model":
        return "no-model"
    acc = str(row.get("accepted", "")).lower() in ("true", "1", "1.0")
    if not acc:
        return "margin-fallback"
    sk = row.get("skipped")
    return "accepted-after-skip" if isinstance(sk, str) and sk.strip() else "accepted"


def _bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def _protocol_order(keys):
    from . import protocols as PR
    order = list(PR.PORTFOLIO_18)
    return [k for k in order if k in set(keys)] + sorted(set(keys) - set(order))


def _wmean(v, g):
    from .evaluate import wmean
    return wmean(np.asarray(v, float), np.asarray(g))


def _boot_diff(d_by_circuit: np.ndarray, reps: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(d_by_circuit), size=(int(reps), len(d_by_circuit)))
    bs = d_by_circuit[picks].mean(1)
    return float(d_by_circuit.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


# ----------------------------------------------------------------------------------- deployment policy (F1)
def _sci(B) -> str:
    """A budget as mathtext: 10⁴, 3·10⁵ (or the plain number when it is not of that form)."""
    B = int(B)
    e = int(np.floor(np.log10(B))) if B > 0 else 0
    m = B / 10 ** e if B > 0 else 0
    if B > 0 and abs(m - round(m)) < 1e-9:
        return r"$10^{%d}$" % e if round(m) == 1 else r"$%d\cdot10^{%d}$" % (round(m), e)
    return f"{B:,}"


def _replay_tables(run_dir: str):
    """The replayed rows of every bundle with their outcome, the SBS-B and model-component risks of the same rows, and
    the chosen candidate's execution-fallback share (``fail_any_eval``) — all from saved tables."""
    import pandas as pd
    rdir = os.path.join(run_dir, "replay")
    cdir = os.path.join(run_dir, "confirm")
    lab = pd.read_csv(os.path.join(cdir, "labels.csv"), encoding="utf-8")
    risk = {(r.task_id, r.protocol, int(r.B)): float(r.mse_eval) for r in lab.itertuples()}
    fail = {(r.task_id, r.protocol, int(r.B)): float(getattr(r, "fail_any_eval", np.nan)) for r in lab.itertuples()}
    ch = pd.read_csv(os.path.join(cdir, "choices_confirm.csv"), encoding="utf-8")
    key = {(r["task_id"], int(r["B"])): r for r in ch.to_dict("records")}
    summ = TX.read_json(os.path.join(rdir, "replay_summary.json"))
    tabs = {}
    for name in summ:
        p = os.path.join(rdir, "replay_%s.csv" % name)
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p, encoding="utf-8")
        meta = TX.read_json(os.path.join(rdir, "replay_%s.json" % name))
        conf = meta.get("model_component_configuration")
        t["outcome"] = [outcome(r) for r in t.to_dict("records")]
        t["sbsb_choice"] = [key.get((r.task_id, int(r.B)), {}).get("SBS-B|choice") for r in t.itertuples()]
        t["sbsb_risk"] = [risk.get((r.task_id, r.sbsb_choice, int(r.B)), np.nan) for r in t.itertuples()]
        if conf and ("%s|choice" % conf) in ch.columns:
            t["component_choice"] = [key.get((r.task_id, int(r.B)), {}).get("%s|choice" % conf)
                                     for r in t.itertuples()]
        elif conf and "predicted" in t.columns:
            # the confirmation's evaluation did not include this configuration: the component is the bundle's own
            # predictions minimised over the label-feasible candidates (as scripts/replay_confirmation.py scores it)
            from . import replay as RP
            feas = {(r.task_id, r.protocol, int(r.B)): RP._feasible(r.feasible) for r in lab.itertuples()}
            t["component_choice"] = [c["choice"] for c in
                                     RP.component_rows_from_predictions(t.to_dict("records"), risk, feas)]
        if "component_choice" in t.columns:
            t["component_risk"] = [risk.get((r.task_id, r.component_choice, int(r.B)), np.nan)
                                   for r in t.itertuples()]
        t["exec_fallback_share"] = [fail.get((r.task_id, r.choice, int(r.B)), np.nan) for r in t.itertuples()]
        t["log_ratio_vs_sbsb"] = np.log10((t.risk + EPS0) / (t.sbsb_risk + EPS0))
        t["bundle"] = name
        tabs[name] = (t, meta)
    return tabs, summ, ch


def order_bundles(run_dir: str, tabs: dict) -> tuple:
    """(bundle names with the primary first, the primary) — 1.9.0 (review R2): the primary is the bundle whose
    configuration is the frozen one (``freeze.json``) and that was replayed with simulation allowed (preflight at the
    allocation level, D41); without a freeze, the simulation-allowed bundle. 1.8.0 put the default deployable bundle
    first and its caption generator called the frozen configuration secondary."""
    frz = os.path.join(run_dir, "freeze.json")
    policy = TX.read_json(frz).get("policy") if os.path.exists(frz) else None

    def is_primary(n):
        meta = tabs[n][1]
        return bool(meta.get("allow_simulation")) and (policy is None or
                                                        meta.get("model_component_configuration") == policy)
    prim = [n for n in sorted(tabs) if is_primary(n)]
    primary = prim[0] if prim else sorted(tabs, key=lambda n: (0 if "deployable" not in n else 1, n))[0]
    return [primary] + sorted(n for n in tabs if n != primary), primary


def supp_deployment_policy(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """(A) Outcome of every deployed recommendation on the confirmation set, per bundle; (B) the full deployed policy
    against SBS-B beside the model-component benchmark and the fixed baselines (paired, base circuits); (C) per-row
    log10 ratio of (risk + 10⁻⁶) to SBS-B by outcome; (D) the candidates the full policy and the model component chose; (E) the
    execution-fallback share of the final candidate's evaluation trials by outcome."""
    import pandas as pd
    plt = _plt()
    tabs, summ, ch = _replay_tables(run_dir)
    if not tabs:
        raise ValueError("no replay tables in %s/replay" % run_dir)
    names, primary = order_bundles(run_dir, tabs)            # 1.9.0: the primary contrast (D41) first
    short = {n: ("default (deployable features)" if "deployable" in n else "simulator features") for n in names}
    role = {n: ("primary" if n == primary else "secondary") for n in names}
    met = pd.read_csv(os.path.join(run_dir, "confirm", "metrics_confirm.csv"), encoding="utf-8")
    met = met[met.B.astype(str) == "all"]
    fig = plt.figure(figsize=(7.4, 8.6))
    top = fig.add_gridspec(2, 1, height_ratios=[0.42, 1.0], left=0.36, right=0.97, top=0.95, bottom=0.61,
                           hspace=1.1)
    low = fig.add_gridspec(1, 3, left=0.12, right=0.98, top=0.52, bottom=0.08, wspace=0.85)
    axA = fig.add_subplot(top[0, 0])
    axB = fig.add_subplot(top[1, 0])
    axC = fig.add_subplot(low[0, 0])
    axD = fig.add_subplot(low[0, 1])
    axE = fig.add_subplot(low[0, 2])
    def _fresh(t):          # classical seconds of a call that recomputed its features (the first budget of a task)
        q = t[t.cache == "miss"] if "cache" in t.columns else t
        return float(np.nanmedian(q.seconds)) if len(q) else float("nan")
    secs = {n: _fresh(tabs[n][0]) for n in names}
    # A: outcome composition
    comp = []
    for i, n in enumerate(names):
        t = tabs[n][0]
        left = 0
        for k, lab, c in OUTCOMES:
            m = int((t.outcome == k).sum())
            comp.append(dict(bundle=n, outcome=k, rows=m, share=m / len(t), base_circuits=t.base_id.nunique()))
            if m:
                axA.barh(i, m, left=left, color=c, edgecolor="#FFFFFF", linewidth=1.0, height=0.6)
                if m / len(t) > 0.04:
                    axA.text(left + m / 2, i, str(m), ha="center", va="center", fontsize=6.5,
                             color="#FFFFFF" if k in ("accepted", "ood-fallback", "unsupported") else "#222222")
            left += m
        axA.text(left * 1.01, i, "%d rows · %d base circuits\nmedian %.2f s per fresh call (classical, no shots)"
                 % (len(t), t.base_id.nunique(), secs[n]), va="center", fontsize=6, color="#444444")
    nmax = max(len(tabs[n][0]) for n in names)
    axA.set_xlim(0, nmax * 1.75)
    step = 100 if nmax <= 1000 else 10 ** int(np.floor(np.log10(nmax)))
    axA.set_xticks(np.arange(0, nmax + 1, step))          # no ticks over the annotation
    axA.spines["bottom"].set_bounds(0, nmax)
    axA.set_yticks(range(len(names)))
    axA.set_yticklabels([short[n].replace(" (", "\n(") for n in names])
    axA.set_xlabel("rows (task × budget)")
    axA.invert_yaxis()
    axA.grid(axis="y", visible=False)
    axA.set_title("A  what the application did", loc="left", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for k, _, c in OUTCOMES]
    axA.legend(handles, [lab for _, lab, _ in OUTCOMES], frameon=False, fontsize=6, loc="upper center",
               bbox_to_anchor=(0.32, -0.55), ncol=3)
    # B: paired differences to SBS-B
    rowsB = []
    for p_ in ("RAW", "SBS", "SIZE-B", "ALL-CANDIDATE"):
        q = met[met.policy == p_]
        if len(q):
            q = q.iloc[0]
            rowsB.append(dict(label={"ALL-CANDIDATE": "empirical reference (selection half)"}.get(p_, p_),
                              kind=p_, diff=q.diff_vs_sbsb, lo=q.diff_vs_sbsb_lo, hi=q.diff_vs_sbsb_hi,
                              source="metrics_confirm.csv"))
    for n in names:
        s = dict(summ.get(n, {}), **tabs[n][1])      # the bundle's full summary (replay_<bundle>.json)
        meta = tabs[n][1]
        mc = meta.get("model_component_benchmark")
        if isinstance(mc, dict) and "diff" in mc:
            rowsB.append(dict(label="model component, %s" % meta.get("model_component_configuration", ""),
                              kind="MODEL-COMPONENT|" + n, diff=mc["diff"], lo=mc["diff_lo"], hi=mc["diff_hi"],
                              source="replay_%s.json (model_component_benchmark)" % n))
        if s.get("diff") is not None and np.isfinite(s.get("diff", np.nan)):
            rowsB.append(dict(label="full policy, %s" % short[n], kind="POLICY|" + n, diff=s["diff"],
                              lo=s["diff_lo"], hi=s["diff_hi"], source="replay_%s.json" % n))
        else:
            rowsB.append(dict(label="full policy, %s (undefined: %s unscored)" % (short[n], s.get("unscored")),
                              kind="POLICY|" + n, diff=np.nan, lo=np.nan, hi=np.nan, source="replay_%s.json" % n))
    B_ = pd.DataFrame(rowsB)
    for i, r in B_.iterrows():
        if r.kind.startswith("POLICY"):
            c, mk = "#0072B2", "o"
        elif r.kind.startswith("MODEL-COMPONENT"):
            c, mk = "#56B4E9", "D"
        else:
            c, mk = POLICY_STYLE.get(r.kind, ("#555555", "s", "-"))[:2]
            mk = mk or "s"
        if np.isfinite(r["diff"]):
            axB.plot([r.lo, r.hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
            axB.plot(r["diff"], i, mk, color=c, ms=6, mec="#FFFFFF", mew=1.0)
            axB.text(max(r.hi, r["diff"]) + 0.02, i, "%+.3f [%+.3f, %+.3f]" % (r["diff"], r.lo, r.hi),
                     va="center", fontsize=6.5, color="#333333",
                     bbox=dict(boxstyle="square,pad=0.08", fc="#FFFFFF", ec="none", alpha=0.85))
    axB.axvline(0, color="#7F7F7F", lw=0.8)
    axB.set_yticks(range(len(B_)))
    axB.set_yticklabels(B_.label, fontsize=6.5)
    axB.invert_yaxis()
    lo_ = np.nanmin(B_.lo.to_numpy(float)) if len(B_) else -0.1
    hi_ = np.nanmax(B_.hi.to_numpy(float)) if len(B_) else 0.1
    axB.set_xlim(min(lo_, -0.05) - 0.05, max(hi_, 0.05) + 0.45)
    axB.set_xlabel("Δ mean log10(MSE + 10⁻⁶) vs SBS-B  (negative = better)")
    axB.set_title("B  confirmation set: full policy beside the model component (paired, base circuits)", loc="left",
                  fontweight="bold")
    # C: per-row log ratio by outcome (1.9.0: the primary bundle; 1.8.0 drew the default deployable bundle here)
    n0 = primary
    t0 = tabs[n0][0]
    cats = [k for k, _, _ in OUTCOMES if (t0.outcome == k).any()]
    rowsC = []
    rng = np.random.default_rng(0)
    allv = t0.log_ratio_vs_sbsb.to_numpy(float)
    allv = allv[np.isfinite(allv)]
    lim = float(min(2.5, max(0.25, 1.1 * np.quantile(np.abs(allv), 0.995)))) if len(allv) else 1.0
    for j, k in enumerate(cats):
        v = t0.loc[t0.outcome == k, "log_ratio_vs_sbsb"].to_numpy(float)
        v = v[np.isfinite(v)]
        rowsC.append(dict(outcome=k, rows=len(v), median=float(np.median(v)) if len(v) else np.nan,
                          q10=float(np.quantile(v, 0.1)) if len(v) else np.nan,
                          q90=float(np.quantile(v, 0.9)) if len(v) else np.nan,
                          share_better=float(np.mean(v < 0)) if len(v) else np.nan,
                          share_worse=float(np.mean(v > 0)) if len(v) else np.nan))
        y = j + rng.uniform(-0.22, 0.22, len(v))
        axC.scatter(np.clip(v, -lim, lim), y, s=5, color=OUTCOME_COLOR[k], alpha=0.55, linewidths=0)
        if len(v):
            axC.plot([np.median(v)] * 2, [j - 0.32, j + 0.32], color="#222222", lw=1.4)
    axC.axvline(0, color="#7F7F7F", lw=0.8)
    axC.set_yticks(range(len(cats)))
    axC.set_yticklabels([k.replace("-", " ") for k in cats], fontsize=6.5)
    axC.invert_yaxis()
    axC.set_xlim(-1.05 * lim, 1.05 * lim)
    axC.set_xlabel("log10 ratio of (risk + 10⁻⁶) to SBS-B, per row\n(clipped to ±%.2g for display)" % lim)
    axC.set_title("C  where gains and losses arise", loc="left", fontweight="bold")
    # D: chosen candidates, full policy vs model component
    cc = t0.choice.value_counts()
    mc = t0.component_choice.value_counts() if "component_choice" in t0 else pd.Series(dtype=int)
    keys = _protocol_order(set(cc.index) | set(mc.index))
    rowsD = []
    for j, k in enumerate(keys):
        a, b = int(cc.get(k, 0)), int(mc.get(k, 0))
        rowsD.append(dict(protocol=k, full_policy=a, model_component=b, new_in_18=k in NEW_18))
        axD.plot([b, a], [j, j], color="#C8C8C8", lw=1.0)
        axD.plot(b, j, "D", color="#56B4E9", ms=4.5, mec="#FFFFFF", mew=0.6)
        axD.plot(a, j, "o", color="#0072B2", ms=4.5, mec="#FFFFFF", mew=0.6)
    axD.set_yticks(range(len(keys)))
    axD.set_yticklabels([k + (" *" if k in NEW_18 else "") for k in keys], fontsize=6.5)
    axD.invert_yaxis()
    axD.set_xlabel("rows choosing the candidate")
    axD.legend([plt.Line2D([], [], marker="o", color="#0072B2", ls=""),
                plt.Line2D([], [], marker="D", color="#56B4E9", ls="")],
               ["full policy", "model component"], frameon=True, framealpha=0.9, edgecolor="#FFFFFF",
               fontsize=6, loc="lower right", markerscale=0.7)
    axD.set_title("D  candidates chosen", loc="left", fontweight="bold")
    # E: execution fallback share of the final candidate by outcome, per bundle
    rowsE = []
    w = 0.8 / max(len(names), 1)
    for b_i, n in enumerate(names):
        t = tabs[n][0]
        for j, k in enumerate(cats):
            v = t.loc[t.outcome == k, "exec_fallback_share"].to_numpy(float)
            v = v[np.isfinite(v)]
            m = float(v.mean()) if len(v) else np.nan
            rowsE.append(dict(bundle=n, outcome=k, rows=len(v), mean_exec_fallback_share=m,
                              rows_with_any_fallback=int((v > 0).sum())))
            if len(v):
                yb = j + (b_i - (len(names) - 1) / 2) * w
                axE.barh(yb, m, height=w * 0.92,
                         color=OUTCOME_COLOR[k], alpha=1.0 if b_i == 0 else 0.45, edgecolor="#FFFFFF", lw=0.8)
                axE.text(m, yb, " %d/%d rows" % (int((v > 0).sum()), len(v)), va="center", fontsize=5.5,
                         color="#444444")
    axE.set_yticks(range(len(cats)))
    axE.set_yticklabels([k.replace("-", " ") for k in cats], fontsize=6.5)
    axE.invert_yaxis()
    if len(names) > 1:
        axE.legend([plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=a_) for a_ in (1.0, 0.45)],
                   [short[n].split(" (")[0] for n in names], frameon=True, framealpha=0.9, edgecolor="#FFFFFF",
                   fontsize=6, loc="lower right")
    mE = [r_["mean_exec_fallback_share"] for r_ in rowsE if np.isfinite(r_["mean_exec_fallback_share"])]
    axE.set_xlim(0, max(max(mE) if mE else 0.0, 1e-4) * 1.75)        # room for the row counts
    axE.set_xlabel("trials with a fallback or failure\nflag, final candidate (mean share)")
    axE.set_title("E  execution fallbacks", loc="left", fontweight="bold")
    fig.suptitle(LABEL_SIM + "  Confirmation set, evaluated once after the freeze (full deployed policy).",
                 fontsize=6.5, color="#555555")
    s0 = dict(summ.get(n0, {}), **tabs[n0][1])
    sec = [n for n in names if n != primary]
    cap = ("Once-only confirmation of the frozen 1.8.0 policy (%d rows, %d base circuits). Primary contrast (D41, "
           "fixed after the development results and before any confirmation result): the full deployed policy of the "
           "frozen configuration (%s, preflight at the allocation level) against SBS-B; secondary: %s. Every row is a "
           "call of the "
           "deployment API (deploy.recommend, policy %s) on a confirmation task at one total budget, scored with the "
           "labels' evaluation half; nothing is approximated by benchmark-only feasibility. (A) Outcome of each call: "
           "the model's top-ranked candidate accepted; accepted after the preflight skipped higher-ranked candidates "
           "that were unverified or infeasible; the SBS-B fallback because the margin was below the frozen τ, because "
           "an out-of-distribution check fired, or because no prediction existed; unsupported inputs. (B) Paired "
           "difference to SBS-B of the mean over base circuits of log10(MSE + 10⁻⁶) with 95 %% bootstrap intervals "
           "over base circuits: the full policy of each bundle, the model component of the same configuration "
           "(the model's best label-feasible candidate, without OOD, preflight or margin rules), and the fixed "
           "baselines and the empirical reference of the confirmation evaluation. A difference of Δ means the "
           "geometric mean of (MSE + 10⁻⁶) changes by 10^Δ — not the MSE itself. (C) Per-row log10 ratio of the "
           "policy's risk to SBS-B's for the primary bundle, by outcome (bar = median). (D) Candidates chosen by the "
           "full policy and the model component (primary bundle; * = new in 1.8.0). (E) Share of the final "
           "candidate's evaluation "
           "trials carrying a method fallback or failure flag (part of the risk; never removed), per outcome; "
           "solid = primary bundle, light = secondary. Risk: the fresh-design average risk of the returned candidate "
           "(the labels' evaluation half averages independently drawn training designs), not the risk conditional "
           "on the one design the API returns for execution. Seconds: median classical wall time of a call "
           "that recomputed its features (the first budget of each task; later budgets reuse the cached "
           "budget-independent features). "
           "Unscored rows (a returned candidate without feasible labels): %s." % (
               int(s0.get("rows", 0)), int(s0.get("base_circuits", 0)), short[primary],
               ", ".join(short[n] for n in sec) or "none", tabs[n0][1].get("policy", ""),
               ", ".join("%s %d" % (short[n].split(" (")[0], int(tabs[n][1].get("unscored", 0))) for n in names)))
    interp = ("Observation (primary contrast D41, specified after the development results and before the confirmation "
              "results; evaluated once): the full deployed policy of the frozen configuration (%s) differs from SBS-B by "
              "%+.3f [%+.3f, %+.3f] in the base-circuit mean of log10(MSE + 10⁻⁶) (a factor %.2f on the geometric mean "
              "of MSE + 10⁻⁶, not on the MSE itself); %d of %d rows used the fallback and %d accepted a candidate after "
              "the preflight skipped a higher-ranked one. The model-component benchmark is not the application's "
              "performance; panel B shows both."
              % (short[primary], s0.get("diff", np.nan), s0.get("diff_lo", np.nan), s0.get("diff_hi", np.nan),
                 10 ** s0.get("diff", np.nan),
                 int((t0.outcome.str.contains("fallback") | (t0.outcome == "no-model")).sum()), len(t0),
                 int((t0.outcome == "accepted-after-skip").sum())))
    for n in sec:
        s2 = dict(summ.get(n, {}), **tabs[n][1])
        interp += (" Secondary (declared before the confirmation): the %s bundle's full policy %+.3f [%+.3f, %+.3f]."
                   % (short[n], s2.get("diff", np.nan), s2.get("diff_lo", np.nan), s2.get("diff_hi", np.nan)))
    for n in names:                          # 1.9.0: per bundle, named (1.8.0 reported the default bundle only)
        tn = tabs[n][0]
        if "component_choice" not in tn.columns:
            continue
        # rows where the component's (label-feasible) choice was left unverified by the deployed preflight; the
        # out-of-distribution subgroup is counted separately (a row can be both: not mutually exclusive mechanisms)
        unv = [r for r in tn.to_dict("records") if r.get("component_choice") and r["choice"] != r["component_choice"]
               and ("%s:unverified" % r["component_choice"]) in str(r.get("skipped") or "")]
        if unv:
            top = pd.Series([r["component_choice"] for r in unv]).value_counts()
            n_ood = sum(1 for r in unv if r.get("outcome") == "ood-fallback")
            interp += (" %s bundle: on %d rows (%d base circuits) the model component's candidate (%s) was left "
                       "unverified by the deployed preflight, which never accepts an unverified candidate: on %d of "
                       "them the policy accepted another candidate, on %d the request was also out of distribution and "
                       "fell back to SBS-B." % (short[n].split(" (")[0].capitalize(), len(unv),
                                                len({r["base_id"] for r in unv}),
                                                ", ".join("%s %d" % kv for kv in top.items()), len(unv) - n_ood, n_ood))
    return _save(fig, out_dir, "supp_deployment_policy",
                 dict(outcomes=pd.DataFrame(comp), comparison=B_, log_ratio=pd.DataFrame(rowsC),
                      chosen=pd.DataFrame(rowsD), execution=pd.DataFrame(rowsE),
                      rows=pd.concat([tabs[n][0] for n in names], ignore_index=True)), cap, interp)


# ----------------------------------------------------------------------------------- portfolio expansion
CONTRAST_LABEL = {
    "expansion_fixed_policy": "more candidates, fixed policy\n(SBS-B 1.8 − SBS-B 1.6)",
    "expansion_attainable": "more candidates, attainable\n(reference 1.8 − reference 1.6)",
    "learning_within_1.6_ridge": "learning within 1.6 (Ridge − SBS-B)",
    "learning_within_1.6_hgb": "learning within 1.6 (HGB − SBS-B)",
    "learning_within_1.8_ridge": "learning within 1.8 (Ridge − SBS-B)",
    "learning_within_1.8_hgb": "learning within 1.8 (HGB − SBS-B)",
    "total_ridge": "total (Ridge 1.8 − SBS-B 1.6)",
    "total_hgb": "total (HGB 1.8 − SBS-B 1.6)",
}


def supp_portfolio_expansion(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """(A) Paired contrasts separating the benefit of more candidates from the benefit of learned selection;
    (B) mean log10(risk + 10⁻⁶) of each policy with the 1.6 and the 1.8 portfolio; (C) the candidates each 1.8 policy chose;
    (D) the declared tails (severe degradation against RAW, 95 % quantile of the risk) per policy and portfolio."""
    import pandas as pd
    plt = _plt()
    pdir = os.path.join(run_dir, "portfolio")
    con = pd.read_csv(os.path.join(pdir, "portfolio_contrasts.csv"), encoding="utf-8")
    pol = pd.read_csv(os.path.join(pdir, "portfolio_policies.csv"), encoding="utf-8")
    res = TX.read_json(os.path.join(pdir, "portfolio_comparison.json"))
    fig = plt.figure(figsize=(7.4, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    order = [c for c in CONTRAST_LABEL if c in set(con.contrast)]
    con = con.set_index("contrast").loc[order].reset_index()
    for i, r in con.iterrows():
        c = "#7F7F7F" if r.contrast.startswith("expansion") else ("#0072B2" if "1.8" in r.contrast or
                                                                    r.contrast.startswith("total") else "#56B4E9")
        axA.plot([r.lo, r.hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        axA.plot(r["diff"], i, "o", color=c, ms=5.5, mec="#FFFFFF", mew=0.8)
        axA.text(r.hi + 0.015, i, "%+.3f" % r["diff"], va="center", fontsize=6.5, color="#333333")
    axA.axvline(0, color="#7F7F7F", lw=0.8)
    axA.set_yticks(range(len(con)))
    axA.set_yticklabels([CONTRAST_LABEL[c] for c in con.contrast], fontsize=6.5)
    axA.invert_yaxis()
    axA.set_xlim(min(con.lo.min(), -0.05) - 0.03, max(con.hi.max(), 0.05) + 0.12)
    axA.set_xlabel("Δ mean log10(MSE + 10⁻⁶)  (negative = better)")
    axA.set_title("A  more candidates vs learning to select", loc="left", fontweight="bold")
    # B: slope chart per policy
    pol["portfolio"] = pol.policy.str.split("|").str[1]
    pol["name"] = pol.policy.str.split("|").str[0]
    names = [n for n in ("SBS-B", "MODEL[ridge]", "MODEL[hgb]", "ALL-CANDIDATE") if n in set(pol.name)]
    sty = {"SBS-B": POLICY_STYLE["SBS-B"], "MODEL[ridge]": ("#56B4E9", "o", "-"),
           "MODEL[hgb]": ("#0072B2", "o", "-"), "ALL-CANDIDATE": POLICY_STYLE["ALL-CANDIDATE"]}
    labs = []
    for n in names:
        q = pol[pol.name == n].set_index("portfolio")
        ys = [q.loc[p, "mean_log10"] if p in q.index else np.nan for p in ("1.6", "1.8")]
        c, mk, ls = sty[n]
        axB.plot([0, 1], ys, ls=ls, color=c, marker=mk or "s", ms=5, lw=1.4, mec="#FFFFFF", mew=0.6)
        labs.append([float(ys[1]), n.replace("MODEL[", "").replace("]", "").replace("ALL-CANDIDATE", "reference")
                     .replace("ridge", "Ridge").replace("hgb", "HGB")])
    # end labels: pushed apart to a minimum gap so that close end points stay readable
    fin = [l_ for l_ in labs if np.isfinite(l_[0])]
    if fin:
        span = max(l_[0] for l_ in fin) - min(l_[0] for l_ in fin)
        gap = 0.065 * max(span, 1e-9)
        fin.sort(key=lambda l_: l_[0])
        pos = [l_[0] for l_ in fin]
        for _ in range(50):
            for i in range(1, len(pos)):
                if pos[i] - pos[i - 1] < gap:
                    mid = (pos[i] + pos[i - 1]) / 2
                    pos[i - 1], pos[i] = mid - gap / 2, mid + gap / 2
        for (y0, lab_), y1 in zip(fin, pos):
            axB.annotate(lab_, (1.0, y0), xytext=(1.08, y1), textcoords="data", va="center", fontsize=6.5,
                         color="#333333", arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.5,
                                                          shrinkA=0, shrinkB=3) if abs(y1 - y0) > gap / 4 else None)
    axB.set_xticks([0, 1])
    npf = res.get("portfolios", {})
    axB.set_xticklabels(["portfolio-%s\n(%s candidates)" % (p, len(npf[p]) if p in npf else "?")
                         for p in ("1.6", "1.8")])
    axB.set_xlim(-0.25, 1.5)
    axB.set_ylabel("mean log10(MSE + 10⁻⁶) over base circuits")
    axB.set_title("B  same tasks, same labels", loc="left", fontweight="bold")
    # C: chosen shares (1.8 policies)
    pols18 = [p for p in ("SBS-B|1.8", "MODEL[ridge]|1.8", "MODEL[hgb]|1.8", "ALL-CANDIDATE|1.8")
              if p in res["policies"]]
    keys = _protocol_order(set().union(*[set(res["policies"][p]["chosen"]) for p in pols18]))
    M = np.array([[res["policies"][p]["chosen"].get(k, 0) for p in pols18] for k in keys], float)
    M = M / np.maximum(M.sum(0, keepdims=True), 1)
    im = axC.imshow(M, aspect="auto", cmap=_seq_cmap(), vmin=0, vmax=max(0.5, float(M.max())))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.005:
                axC.text(j, i, "%.0f" % (100 * M[i, j]), ha="center", va="center", fontsize=6,
                         color="#FFFFFF" if M[i, j] > 0.3 else "#222222")
    axC.set_yticks(range(len(keys)))
    axC.set_yticklabels([k + (" *" if k in NEW_18 else "") for k in keys], fontsize=6.5)
    axC.set_xticks(range(len(pols18)))
    axC.set_xticklabels([p.split("|")[0].replace("MODEL[", "").replace("]", "").replace("ALL-CANDIDATE", "ref.")
                         .replace("ridge", "Ridge").replace("hgb", "HGB") for p in pols18], fontsize=6.5)
    axC.grid(False)
    fig.colorbar(im, ax=axC, shrink=0.7, label="share of rows (%)", format=lambda x, _: "%.0f" % (100 * x))
    axC.set_title("C  candidates chosen, portfolio-1.8 (* new)", loc="left", fontweight="bold")
    # D: tails
    rowsD = []
    for j, n in enumerate(names):
        for p, mk in (("1.6", "o"), ("1.8", "s")):
            q = pol[(pol.name == n) & (pol.portfolio == p)]
            if not len(q):
                continue
            q = q.iloc[0]
            v = float(q.get("p_severe_vs_raw", np.nan))
            rowsD.append(dict(policy=n, portfolio=p, p_severe_vs_raw=v, q95_risk=q.q95_risk,
                              worst5_loss_share=q.worst5_loss_share, p_severe_abs=q.p_severe_abs))
            c = sty[n][0]
            axD.plot(100 * v, j + (0.12 if p == "1.8" else -0.12), mk, color=c,
                     mfc=c if p == "1.8" else "#FFFFFF", ms=5.5, mew=1.2)
    axD.set_yticks(range(len(names)))
    axD.set_yticklabels([n.replace("MODEL[", "").replace("]", "").replace("ALL-CANDIDATE", "reference")
                         .replace("ridge", "Ridge").replace("hgb", "HGB") for n in names], fontsize=6.5)
    axD.invert_yaxis()
    axD.set_xlabel("rows with R > 10 R_RAW (%, circuits weighted)")
    axD.set_xlim(left=-0.5, right=max(5.0, 1.15 * 100 * max([r["p_severe_vs_raw"] for r in rowsD
                                                             if np.isfinite(r["p_severe_vs_raw"])] or [0])))
    axD.legend([plt.Line2D([], [], marker="o", mfc="#FFFFFF", color="#555555", ls=""),
                plt.Line2D([], [], marker="s", color="#555555", ls="")], ["portfolio-1.6", "portfolio-1.8"],
               frameon=False, fontsize=6, loc="lower right")
    axD.set_title("D  severe degradation against RAW", loc="left", fontweight="bold")
    fig.suptitle(LABEL_SIM + "  Development data, out-of-fold (5 folds grouped by base circuit); exploratory.",
                 fontsize=6.5, color="#555555")
    cap = ("Original (portfolio-1.6, the eight candidates of 1.6/1.7) against expanded (portfolio-1.8, 17 candidates) "
           "portfolio on identical development tasks and labels (%d rows, %d base circuits; out-of-fold, 5 folds "
           "grouped by base circuit; feature set %s). Every policy is refitted for each portfolio. (A) Paired "
           "contrasts of the mean over base circuits of log10(MSE + 10⁻⁶) with 95 %% bootstrap intervals over base "
           "circuits (%s): grey — what more candidates give a fixed policy and an oracle-like reference (the "
           "candidate best on the independent selection half, scored on the evaluation half); blue — what learned "
           "selection adds within each portfolio, and the total. (B) The policies' means with each portfolio. "
           "(C) Share of rows on which each 1.8 policy chose a candidate (%%). (D) Declared tail: rows whose chosen "
           "risk exceeds ten times RAW's on the same row." % (res.get("rows", 0), res.get("base_circuits", 0),
                                                              res.get("feature_set", ""), res.get("metric", "")))
    c_ = {r.contrast: r for r in con.itertuples()}

    def _f(k):
        r = c_.get(k)
        return "%+.3f [%+.3f, %+.3f]" % (r.diff, r.lo, r.hi) if r is not None else "n/a"
    interp = ("Observation (exploratory, development data): more candidates change SBS-B by %s and the attainable "
              "reference by %s; learned selection (HGB) changes the 1.8 portfolio's SBS-B by %s. A larger portfolio "
              "helps a policy only if the policy can identify where the new candidates win."
              % (_f("expansion_fixed_policy"), _f("expansion_attainable"), _f("learning_within_1.8_hgb")))
    return _save(fig, out_dir, "supp_portfolio_expansion",
                 dict(contrasts=con, policies=pol, chosen=pd.DataFrame(M, index=keys, columns=pols18).reset_index()
                      .rename(columns={"index": "protocol"}), tails=pd.DataFrame(rowsD)), cap, interp)


# ----------------------------------------------------------------------------------- root audit
def supp_root_audit(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """(A) Between-root variance of each label against its own design-level SE² (ratio 1 = honest SE); (B) whether
    the best candidate of a (task, budget) is the same under every preparation root; (C) the cost of using another
    root's winner; (D) design-level preflight against each root's allocation, and the deployed policy per root."""
    import pandas as pd
    plt = _plt()
    adir = os.path.join(run_dir, "audit18", "analysis")
    st = pd.read_csv(os.path.join(adir, "label_stability.csv"), encoding="utf-8")
    rk = pd.read_csv(os.path.join(adir, "ranking_stability.csv"), encoding="utf-8")
    ct = pd.read_csv(os.path.join(adir, "contract.csv"), encoding="utf-8")
    sm = TX.read_json(os.path.join(adir, "audit_summary.json"))
    fig = plt.figure(figsize=(7.4, 5.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    axA, axB = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    ok = st.dropna(subset=["var_ratio"]) if "var_ratio" in st else st.iloc[0:0]
    keys = _protocol_order(set(ok.protocol))
    rng = np.random.default_rng(1)
    rowsA = []
    budgets = sorted(set(int(b) for b in ok.B)) if len(ok) else []
    bcol = dict(zip(budgets, ("#9ECAE1", "#4292C6", "#08306B")))
    for j, k in enumerate(keys):
        q = ok[ok.protocol == k]
        for B in budgets:
            v = q[q.B == B].var_ratio.to_numpy(float)
            v = v[v > 0]
            if len(v):
                axA.scatter(j + (budgets.index(B) - 1) * 0.22 + rng.uniform(-0.06, 0.06, len(v)), v, s=7,
                            color=bcol[B], linewidths=0, alpha=0.85)
        v = q.var_ratio.to_numpy(float)
        rowsA.append(dict(protocol=k, rows=len(v), median=float(np.median(v)) if len(v) else np.nan,
                          q90=float(np.quantile(v, 0.9)) if len(v) else np.nan, share_gt_4=float(np.mean(v > 4))
                          if len(v) else np.nan))
        if len(v):
            axA.plot([j - 0.36, j + 0.36], [np.median(v)] * 2, color="#222222", lw=1.3)
    axA.set_yscale("log")
    axA.axhline(1, color="#7F7F7F", lw=0.8)
    axA.axhline(4, color="#7F7F7F", lw=0.8, ls="--")
    axA.set_xticks(range(len(keys)))
    axA.set_xticklabels([k + (" *" if k in NEW_18 else "") for k in keys], rotation=45, ha="right", fontsize=6.5)
    axA.set_ylabel("between-root variance of the label\n/ mean label SE² (log)")
    axA.legend([plt.Line2D([], [], marker="o", color=bcol[B], ls="") for B in budgets] +
               [plt.Line2D([], [], color="#222222")],
               ["B = %s" % f"{B:,}" for B in budgets] + ["median"], frameon=False, fontsize=6, ncol=4,
               loc="upper left")
    axA.set_title("A  are the labels' standard errors honest across preparation roots? (1 = yes; dashed: 4)",
                  loc="left", fontweight="bold")
    # B: ranking stability (17 candidates and, where computed, the original eight held fixed)
    rowsB = []
    rk = rk.assign(portfolio=rk.portfolio.astype(str)) if "portfolio" in rk.columns else rk.assign(portfolio="1.8")
    pfs = [p for p in ("1.8", "1.6") if p in set(rk.portfolio)]
    xs = sorted(set(int(b) for b in rk.B))
    seq = ("#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
    wbar = 0.38 if len(pfs) > 1 else 0.6
    top = 0.0
    for pi, pf in enumerate(pfs):
        q = rk[rk.portfolio == pf]
        pos = np.arange(len(xs)) + (pi - (len(pfs) - 1) / 2) * (wbar + 0.04)
        bottoms = np.zeros(len(xs))
        for w in (1, 2, 3, 4):
            h = np.array([int(((q.B == B) & (q.distinct_winners == w)).sum()) for B in xs], float)
            rowsB += [dict(portfolio=pf, B=B, distinct_winners=w, cells=int(x)) for B, x in zip(xs, h)]
            axB.bar(pos, h, bottom=bottoms, color=seq[w - 1], edgecolor="#FFFFFF", lw=1.0, width=wbar,
                    label=("%d distinct" % w) if pi == 0 else None)
            bottoms += h
        for i, B in enumerate(xs):
            m = q[q.B == B].mean_cross_root_log_regret.to_numpy(float)
            axB.text(pos[i], bottoms[i] + 0.15, "%s\n%.2f" % ("17" if pf == "1.8" else "8",
                                                              np.nanmean(m) if len(m) else np.nan),
                     ha="center", va="bottom", fontsize=5.8, color="#333333")
        top = max(top, float(bottoms.max()) if len(bottoms) else 0.0)
    axB.set_xticks(range(len(xs)))
    axB.set_xticklabels(["B = %s" % _sci(B) for B in xs])
    axB.set_ylim(0, top * 1.45 if top else 1)
    axB.set_xlabel("bar labels: candidates (17 or the original 8)\nand mean cross-root log10 regret", fontsize=6,
                   color="#555555")
    axB.set_ylabel("(task, budget) cells")
    axB.legend(frameon=False, fontsize=6, title="winners over 4 roots", title_fontsize=6, loc="upper left",
               ncol=2)
    axB.set_title("B  is the winner the same under every root?", loc="left", fontweight="bold")
    # D: contract and policy by root
    rowsD = []
    if len(ct):
        g = ct.groupby("protocol").agg(rows=("agree", "size"), agree=("agree", "mean")).reset_index()
        for j, r in enumerate(g.itertuples()):
            rowsD.append(dict(protocol=r.protocol, rows=int(r.rows), agreement=float(r.agree)))
            axD.barh(j, 100 * r.agree, color=PROTOCOL_COLORS.get(r.protocol, "#555555"), height=0.6)
            axD.text(100 * r.agree - 1, j, "%.1f %% of %d" % (100 * r.agree, r.rows), ha="right", va="center",
                     fontsize=6.5, color="#FFFFFF")
        axD.set_yticks(range(len(g)))
        axD.set_yticklabels(g.protocol)
        axD.set_xlim(0, 100)
    pbr = sm.get("policy_by_root") or {}
    axD.set_xlabel("design preflight agrees with the root's allocation (%)" + (
        "\ndeployed policy per root (mean log10(risk + 10⁻⁶)):\n" + ", ".join(
            "%s %.2f" % (k, v["mean_log10"]) for k, v in pbr.items()) if pbr else ""))
    axD.set_title("C  execution contract", loc="left", fontweight="bold")
    fig.suptitle(LABEL_SIM + "  Crossed circuit × preparation-root audit; exploratory (stage 2).", fontsize=6.5,
                 color="#555555")
    cap = ("Crossed audit: %d base circuits (the circuits of the ten 1.7.0 deployment regression cases, earlier "
           "failure types and controls) labeled under %d independently namespaced preparation roots (seed scheme 2) "
           "at budgets %s with the 17 candidates, r = 40 trials and 6 fresh designs per half. (A) For every (task, "
           "candidate, budget) feasible under ≥ 2 roots: the between-root variance of the evaluation-half MSE divided "
           "by the mean squared label standard error (design-level for learning candidates). Under honest standard "
           "errors the ratio scatters around 1; ratios ≫ 1 mean the label depends on rare designs or preparation "
           "draws the SE does not see. (B) Number of distinct winning candidates over the four roots per (task, "
           "budget) — with the 17 candidates and with the original eight held fixed (a subset of the same labels) — "
           "and the mean log10 regret of one root's winner judged by another root's labels. (C) Agreement "
           "of the design-level preflight (qemx.selection.preflight, each root's allocation seed) with the "
           "feasibility the labels recorded." % (sm.get("tasks", 0), len(sm.get("roots", [])),
                                                 ", ".join(f"{b:,}" for b in sm.get("budgets", []))))
    interp = ("Observation (exploratory): median variance ratios by candidate %s; %d of %d (task, budget) cells "
              "have one winner under all roots; mean cross-root log regret %.3f; contract agreement %.4f. Ratios "
              "well above 1 identify candidates whose labels — and hence any model trained on them — are less "
              "certain than their standard errors state." % (
                  ", ".join("%s %.2g" % (k, v) for k, v in sorted(sm.get("var_ratio_median_by_protocol", {}).items())),
                  sm.get("cells_with_one_winner", 0), sm.get("ranking_cells", 0),
                  sm.get("mean_cross_root_log_regret", np.nan), sm.get("contract_agreement", np.nan)))
    return _save(fig, out_dir, "supp_root_audit",
                 dict(var_ratio=pd.DataFrame(rowsA), stability=st, ranking=pd.DataFrame(rowsB), cells=rk,
                      contract=pd.DataFrame(rowsD)), cap, interp)


# ----------------------------------------------------------------------------------- tails (F7)
def supp_tails(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """(A) Each candidate's distribution of log10(R_k / R_RAW) over feasible development rows at three budgets
    (median, 90 % and maximum); (B) the rate of severe degradation (R_k > 10 R_RAW) per candidate and budget;
    (C) loss concentration — the share of a candidate's total loss carried by its worst 5 % of base circuits;
    (D) the worst single training design's share of a learning label's total design loss."""
    import pandas as pd
    from .evaluate import SEVERE_VS_RAW, tail_metrics
    plt = _plt()
    cfg = cfg or {}
    lab = pd.read_csv(os.path.join(run_dir, "analysis", "labels.csv"), encoding="utf-8")
    lab["feasible"] = _bool(lab.feasible)
    raw = lab[lab.protocol == "RAW"].set_index(["task_id", "B"]).mse_eval
    lab = lab.join(raw.rename("raw_mse"), on=["task_id", "B"])
    lab["log_ratio_raw"] = np.log10((lab.mse_eval + EPS0) / (lab.raw_mse + EPS0))
    f = lab[lab.feasible & np.isfinite(lab.mse_eval)]
    keys = _protocol_order(set(lab.protocol))
    pb = cfg.get("regret_profile", {}).get("budgets", {"low": 1000, "medium": 100000, "high": 10000000})
    bud = [int(pb[k]) for k in ("low", "medium", "high") if k in pb]
    bud = [b for b in bud if b in set(lab.B.astype(int))]
    allB = sorted(set(lab.B.astype(int)))
    fig = plt.figure(figsize=(7.4, 7.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    rowsA = []
    bcol = dict(zip(bud, ("#9ECAE1", "#4292C6", "#08306B")))
    for j, k in enumerate(keys):
        for bi, B in enumerate(bud):
            v = f[(f.protocol == k) & (f.B == B)].log_ratio_raw.to_numpy(float)
            tot = int(((lab.protocol == k) & (lab.B == B)).sum())
            y = j + (bi - (len(bud) - 1) / 2) * 0.26
            if len(v):
                med, q90, mx = np.median(v), np.quantile(v, 0.9), v.max()
                axA.plot([med, mx], [y, y], color=bcol[B], lw=1.0)
                axA.plot(med, y, "o", color=bcol[B], ms=3.6)
                axA.plot(q90, y, "|", color=bcol[B], ms=6, mew=1.4)
                axA.plot(mx, y, "x", color=bcol[B], ms=3.6, mew=1.0)
                rowsA.append(dict(protocol=k, B=B, feasible_rows=len(v), rows=tot, median=med, q90=q90, max=mx))
            else:
                rowsA.append(dict(protocol=k, B=B, feasible_rows=0, rows=tot, median=np.nan, q90=np.nan,
                                  max=np.nan))
    for j, k in enumerate(keys):
        if not any(r["feasible_rows"] for r in rowsA if r["protocol"] == k):
            axA.text(0.02, j, "not feasible at these budgets", transform=axA.get_yaxis_transform(), va="center",
                     fontsize=6, color="#7F7F7F")
    axA.axvline(0, color="#7F7F7F", lw=0.8)
    axA.axvline(math.log10(SEVERE_VS_RAW), color="#D55E00", lw=0.8, ls="--")
    axA.set_yticks(range(len(keys)))
    axA.set_yticklabels([k + (" *" if k in NEW_18 else "") for k in keys], fontsize=6.5)
    axA.invert_yaxis()
    axA.set_xlabel("log10[(R_k + 10⁻⁶)/(R_RAW + 10⁻⁶)] per feasible row")
    axA.legend([plt.Line2D([], [], marker="o", color=bcol[B], ls="") for B in bud] +
               [plt.Line2D([], [], marker="|", color="#555555", ls="", ms=6),
                plt.Line2D([], [], marker="x", color="#555555", ls="")],
               ["B = %s" % f"{B:,}" for B in bud] + ["90 % quantile", "maximum"], frameon=False, fontsize=6,
               loc="lower right")
    axA.set_title("A  loss relative to RAW (dashed: 10×)", loc="left", fontweight="bold")
    # B: severe rate heatmap (base-circuit weighted)
    S = np.full((len(keys), len(allB)), np.nan)
    rowsB = []
    for i, k in enumerate(keys):
        for j, B in enumerate(allB):
            q = f[(f.protocol == k) & (f.B == B)]
            if len(q):
                S[i, j] = _wmean((q.mse_eval > SEVERE_VS_RAW * q.raw_mse).astype(float), q.base_id)
            rowsB.append(dict(protocol=k, B=B, feasible_rows=len(q), base_circuits=q.base_id.nunique(),
                              p_severe_vs_raw=S[i, j]))
    cm = _seq_cmap().with_extremes(bad="#FFFFFF")
    im = axB.imshow(np.ma.masked_invalid(S), aspect="auto", cmap=cm, vmin=0, vmax=max(0.2, np.nanmax(S) if
                                                                                         np.isfinite(S).any() else 0))
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if not np.isfinite(S[i, j]):
                axB.text(j, i, "–", ha="center", va="center", fontsize=6, color="#999999")
            elif S[i, j] >= 0.005:
                axB.text(j, i, "%.0f" % (100 * S[i, j]), ha="center", va="center", fontsize=5.5,
                         color="#FFFFFF" if S[i, j] > 0.12 else "#222222")
    axB.set_yticks(range(len(keys)))
    axB.set_yticklabels(keys, fontsize=6.5)
    axB.set_xticks(range(len(allB)))
    axB.set_xticklabels(["%g" % B for B in allB], rotation=60, fontsize=6)
    axB.set_xlabel("total budget B (shots)")
    axB.grid(False)
    fig.colorbar(im, ax=axB, shrink=0.6, label="rows with R > 10 R_RAW (%)", format=lambda x, _: "%.0f" % (100 * x))
    axB.set_title("B  severe degradation rate (– = not feasible)", loc="left", fontweight="bold")
    # C: loss concentration per candidate (pooled over budgets, per budget dots)
    rowsC = []
    for j, k in enumerate(keys):
        for B in bud:
            q = f[(f.protocol == k) & (f.B == B)]
            if len(q):
                tm = tail_metrics(q.mse_eval.to_numpy(float), q.base_id.to_numpy())
                rowsC.append(dict(protocol=k, B=B, **tm, base_circuits=q.base_id.nunique()))
                axC.plot(100 * tm["worst5_loss_share"], j, "o", color=bcol[B], ms=3.8)
    for j, k in enumerate(keys):
        nk = int(f[f.protocol == k].base_id.nunique())
        axC.text(1.01, j, "%d" % nk, transform=axC.get_yaxis_transform(), va="center", fontsize=6,
                 color="#555555", clip_on=False)
    axC.text(1.01, -0.9, "circuits", transform=axC.get_yaxis_transform(), va="center", fontsize=6,
             color="#555555", clip_on=False)
    nb = int(f.base_id.nunique())
    axC.axvline(100 * max(1, math.ceil(0.05 * nb)) / max(nb, 1), color="#7F7F7F", lw=0.8, ls=":")
    axC.set_yticks(range(len(keys)))
    axC.set_yticklabels(keys, fontsize=6.5)
    axC.invert_yaxis()
    axC.set_xlim(0, 100)
    axC.set_xlabel("share of total loss from the worst 5 % of base circuits (%)\n(dotted: equal losses)")
    axC.set_title("C  loss concentration", loc="left", fontweight="bold")
    # D: worst design share (learning candidates)
    rowsD = []
    dd_path = os.path.join(run_dir, "analysis", "design_diagnostics.csv")
    if os.path.exists(dd_path) and os.path.getsize(dd_path) > 2:     # 1.9.0: an empty table means no diagnostics
        dd = pd.read_csv(dd_path, encoding="utf-8")
        dd = dd[np.isfinite(dd.mean_loss.astype(float))]
        grp = dd.groupby(["task_id", "protocol", "B", "half"]).mean_loss
        share = (grp.max() / grp.sum().replace(0, np.nan)).rename("worst_share").reset_index()
        nd = grp.size().rename("designs").reset_index()
        share = share.merge(nd, on=["task_id", "protocol", "B", "half"])
        lk = _protocol_order(set(share.protocol))
        for j, k in enumerate(lk):
            v = share[share.protocol == k].worst_share.dropna().to_numpy(float)
            K = int(share[share.protocol == k].designs.median()) if len(v) else 0
            if len(v):
                parts = axD.violinplot([v], positions=[j], widths=0.7, showextrema=False, showmedians=True)
                for b_ in parts["bodies"]:
                    b_.set_facecolor(PROTOCOL_COLORS.get(k, "#555555"))
                    b_.set_edgecolor("#FFFFFF")
                    b_.set_alpha(0.75)
                parts["cmedians"].set_color("#222222")
                axD.plot([j - 0.35, j + 0.35], [1 / K] * 2, color="#7F7F7F", lw=0.8, ls=":")
            rowsD.append(dict(protocol=k, labels=len(v), designs=K, median_worst_share=float(np.median(v))
                              if len(v) else np.nan, q90_worst_share=float(np.quantile(v, 0.9)) if len(v) else np.nan,
                              share_labels_worst_gt_half=float(np.mean(v > 0.5)) if len(v) else np.nan))
        axD.set_xticks(range(len(lk)))
        axD.set_xticklabels(lk)
        axD.set_ylim(0, 1)
    else:
        axD.text(0.5, 0.5, "no design diagnostics in this run\n(analysis extra design_diagnostics)",
                 transform=axD.transAxes, ha="center", va="center", fontsize=7, color="#7F7F7F")
        axD.set_xticks([])
    axD.set_ylabel("worst design's share of the label's\ntotal design loss (dotted: 1/K)")
    axD.set_title("D  single training designs", loc="left", fontweight="bold")
    fig.suptitle(LABEL_SIM + "  Development data; descriptive.", fontsize=6.5, color="#555555")
    cap = ("Complete loss distributions on the development set (%d tasks, %d base circuits; feasible rows only, each "
           "candidate's own feasible subset — the panels describe each candidate's tail, they do not rank candidates "
           "on a common subset). (A) Per candidate and budget: median, 90 %% quantile and maximum over feasible rows "
           "of the log10 ratio of its evaluation-half risk to RAW's on the same row. (B) Base-circuit-weighted share "
           "of feasible rows with risk above %g × RAW's (the declared severe degradation, finding F7), every budget; "
           "– = no feasible row. (C) Share of the candidate's total circuit-averaged loss carried by the worst 5 %% "
           "of base circuits (dotted line: the share if all circuits lost equally; right: feasible base circuits — with "
           "few circuits the share is necessarily high). (D) For the learning candidates: "
           "the largest single training design's share of the summed design losses of a label half (K designs per "
           "half; dotted 1/K = equal designs); values near 1 mean one design carried the label's loss. No estimate "
           "is clipped or removed; failures and fallbacks are part of every risk." % (
               lab.task_id.nunique(), lab.base_id.nunique(), SEVERE_VS_RAW))
    worst = sorted(((r["p_severe_vs_raw"], r["protocol"], r["B"]) for r in rowsB if np.isfinite(r["p_severe_vs_raw"])),
                   reverse=True)[:3]
    interp = ("Observation (descriptive): the highest severe-degradation rates are %s. Heavy tails are a property of "
              "a candidate at a budget, not of its median: compare panel A's maxima with its medians before "
              "relying on a candidate whose typical loss is low." % (
                  "; ".join("%s at B = %s: %.0f %%" % (k, f"{B:,}", 100 * v) for v, k, B in worst) or "none"))
    return _save(fig, out_dir, "supp_tails",
                 dict(relative_to_raw=pd.DataFrame(rowsA), severe=pd.DataFrame(rowsB),
                      concentration=pd.DataFrame(rowsC), designs=pd.DataFrame(rowsD)), cap, interp)


FIG_META_18 = {
    "supp_deployment_policy": (
        "What does the deployed application policy do on new circuits, and how does its full-policy result compare "
        "with the model-component benchmark?",
        {"replay/replay_<bundle>.csv": "task_id, base_id, B, status, accepted, choice, model_choice, skipped, risk, "
                                       "scored, seconds",
         "replay/replay_<bundle>.json, replay/replay_summary.json": "diff, diff_lo, diff_hi, unscored, "
                                                                    "model_component_benchmark",
         "confirm/choices_confirm.csv": "task_id, B, SBS-B|choice, <configuration>|choice",
         "confirm/labels.csv": "task_id, protocol, B, mse_eval, fail_any_eval",
         "confirm/metrics_confirm.csv": "policy, B (= all), diff_vs_sbsb(_lo/_hi)"}),
    "supp_portfolio_expansion": (
        "Do more candidates help, and is the benefit from the candidates or from learning to select them?",
        {"portfolio/portfolio_contrasts.csv": "contrast, diff, lo, hi",
         "portfolio/portfolio_policies.csv": "policy, mean_log10, mse, p_severe_vs_raw, q95_risk, worst5_loss_share",
         "portfolio/portfolio_comparison.json": "policies.<policy>.chosen, rows, base_circuits, feature_set"}),
    "supp_root_audit": (
        "Are the labels, the rankings and the execution contract stable across independent preparation roots?",
        {"audit18/analysis/label_stability.csv": "task_id, protocol, B, var_ratio",
         "audit18/analysis/ranking_stability.csv": "task_id, B, distinct_winners, mean_cross_root_log_regret",
         "audit18/analysis/contract.csv": "task_id, root, protocol, B, preflight, labels_feasible, agree",
         "audit18/analysis/audit_summary.json": "roots, budgets, tasks, policy_by_root"}),
    "supp_tails": (
        "How heavy are each candidate's loss tails, and do single training designs carry them?",
        {"analysis/labels.csv": "task_id, base_id, protocol, B, feasible, mse_eval",
         "analysis/design_diagnostics.csv": "task_id, protocol, B, half, design, mean_loss"}),
}
FIG_METHODS_18 = {
    "supp_deployment_policy": dict(
        varied="the bundle (default deployable-feature bundle, simulator-feature bundle); held fixed: the frozen "
               "configuration and the confirmation labels",
        counts="{conf_tasks} confirmation tasks on {conf_base} base circuits (none in development); r = {conf_r} "
               "trials per half",
        uncertainty="95 % bootstrap over base circuits of the paired difference to SBS-B (each circuit's rows "
                    "averaged first)",
        status="confirmatory, once only: the primary contrast (D41) is the frozen simulator-feature configuration "
               "at allocation-level preflight against SBS-B, specified after the development results and before the "
               "confirmation results (not a pre-development registration); the default deployable-feature bundle is "
               "a secondary analysis",
        missing="every row kept: fallbacks, skipped candidates and execution fallbacks are part of the policy; a "
                "returned candidate without feasible labels is counted as unscored (no imputed risk)"),
    "supp_portfolio_expansion": dict(
        varied="the portfolio (8 or 17 candidates) and the policy; held fixed: tasks, labels, folds, feature set",
        counts="{dev_tasks} development tasks on {dev_base} independent base circuits; r = {r} trials per half; "
               "{designs} fresh designs per half",
        uncertainty="95 % paired bootstrap over base circuits", status="exploratory (development, out-of-fold)",
        missing="infeasible candidates are not candidates of a row; fallbacks stay in the risk"),
    "supp_root_audit": dict(
        varied="the preparation root (4 independently namespaced master seeds) at 3 budgets; held fixed: circuits, "
               "candidates, R, designs per half",
        counts="8 base circuits × 4 roots; r = 40 trials and 6 fresh designs per half",
        uncertainty="variance ratios per (task, candidate, budget) over 4 roots (few roots: a ratio is itself "
                    "uncertain by a factor of about 2–3)", status="exploratory (stage-2 audit)",
        missing="cells feasible under fewer than 2 roots have no ratio (listed in the source data)"),
    "supp_tails": dict(
        varied="candidate and budget", counts="{dev_tasks} development tasks on {dev_base} independent base circuits; "
        "{designs} fresh designs per half", uncertainty="none (distributions and rates of the labels)",
        status="descriptive (development data)",
        missing="each candidate on its own feasible subset (– where none); nothing clipped or removed"),
}
NEED_18 = dict(
    supp_deployment_policy=["replay/replay_summary.json", "confirm/choices_confirm.csv", "confirm/labels.csv",
                            "confirm/metrics_confirm.csv"],
    supp_portfolio_expansion=["portfolio/portfolio_contrasts.csv", "portfolio/portfolio_policies.csv",
                              "portfolio/portfolio_comparison.json"],
    supp_root_audit=["audit18/analysis/label_stability.csv", "audit18/analysis/ranking_stability.csv",
                     "audit18/analysis/contract.csv", "audit18/analysis/audit_summary.json"],
    supp_tails=["analysis/labels.csv"],
)
