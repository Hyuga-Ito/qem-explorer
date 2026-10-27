r"""Thesis-ready numerical tables of a protocol-selection run (1.8.0): one versioned generator, one source table per
reported value, written as CSV (the numbers), includeable LaTeX fragments (``booktabs``; ``\input`` them), Markdown
(the same strings as the LaTeX, for the documentation) and a standalone wrapper that compiles to a PDF when
``pdflatex`` is available.

    python -m qemx.selection.thesis_tables --run reports/selection_v180 --out reports/selection_v180/thesis_tables

Tables (labels ``tab:qem180-*``):

1. ``t1_studies`` — the executed studies: role, status, tasks and independent logical circuits counted separately,
   budgets, candidates, trials and fresh designs per half, seed scheme;
2. ``t2_candidates`` — the 17 candidates: family, validation status, feasible development tasks at three budgets
   (with the denominator) and the median shot ledger at B = 10⁵;
3. ``t3_confirmation`` — the once-only confirmation: policies (fixed baselines, model components, the full deployed
   policy of both bundles, the empirical reference) with paired circuit-level intervals against SBS-B, arithmetic
   MSE and declared tails; ``t3b_development`` the same columns out-of-fold on the development set;
4. ``t4_per_budget`` — each candidate's median evaluation-half MSE at three budgets with its feasible denominator
   (descriptive: candidates have different feasible subsets);
5. ``t5_tails`` — infeasibility, method fallbacks and failures, severe degradation against RAW, the 95 % quantile and
   the loss concentration per candidate;
6. ``t6_costs`` — classical cost of a recommendation (compilation, features, preflight, inference, the whole call),
   benchmark-label cost, and early rejection outside the training widths;
7. ``t7_portfolio`` — original against expanded portfolio: the benefit of more candidates and of learned selection.

A table whose source is missing is skipped and listed in ``README.md`` of the output folder; nothing is computed
from trials here except what the named source tables contain. Every value carries the source file it came from in
``provenance.json`` (with the source's sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import subprocess
import sys

import numpy as np

from .. import textio as TX

GENERATOR = "qemx.selection.thesis_tables/1"
EPS0 = 1e-6
BUDGETS3 = (1000, 100000, 10000000)


# ------------------------------------------------------------------------------------------------ formatting
def tex_escape(s) -> str:
    s = str(s)
    rep = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
           ("}", r"\}"), ("∘", r"$\circ$"), ("τ", r"$\tau$"), ("≤", r"$\le$"), ("≥", r"$\ge$"), ("×", r"$\times$"),
           ("—", "---"), ("–", "--"), ("−", "-"), ("→", r"$\to$"), ("⟨", r"$\langle$"), ("⟩", r"$\rangle$"),
           ("λ", r"$\lambda$"), ("ℓ", r"$\ell$"), ("Δ", r"$\Delta$"), ("·", r"$\cdot$"), ("⁻⁶", "$\x00{-6}$"),
           ("⁵", "$\x00{5}$"), ("³", "$\x00{3}$"), ("⁷", "$\x00{7}$"), (">", "$>$"), ("<", "$<$")]
    for a, b in rep:
        s = s.replace(a, b)
    import re
    s = s.replace("10^(Delta l)", "$10\x00{\\Delta\\ell}$").replace("Delta l", r"$\Delta\ell$")
    s = re.sub(r"10\^(-?\d+)", "$10\x00{\\1}$", s)
    return s.replace("^", r"\textasciicircum{}").replace("\x00", "^")


def sci(x, digits: int = 2) -> str:
    """LaTeX scientific notation, e.g. $1.23\\times10^{-4}$; an em dash for missing values."""
    if x is None or not np.isfinite(x):
        return r"\textemdash"
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return r"$%.*f\times10^{%d}$" % (digits, m, e)


def sci_md(x, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return "%.*e" % (digits, x)


def signed(x, d: int = 3) -> str:
    return r"\textemdash" if x is None or not np.isfinite(x) else "$%+.*f$" % (d, x)


def ci(m, lo, hi, d: int = 3, sign: bool = True) -> str:
    if m is None or not np.isfinite(m):
        return r"\textemdash"
    f = "%+.*f" if sign else "%.*f"
    return ("$" + f + r"\;[" + f + "," + f + "]$") % (d, m, d, lo, d, hi)


def ci_md(m, lo, hi, d: int = 3, sign: bool = True) -> str:
    if m is None or not np.isfinite(m):
        return "—"
    f = "%+.*f" if sign else "%.*f"
    return (f + " [" + f + ", " + f + "]") % (d, m, d, lo, d, hi)


def pct(x, d: int = 1) -> str:
    return r"\textemdash" if x is None or not np.isfinite(x) else r"$%.*f$\,\%%" % (d, 100 * x)


def pct_md(x, d: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else "%.*f %%" % (d, 100 * x)


def _tex_to_md(s: str) -> str:
    """The Markdown twin of a LaTeX cell (same digits)."""
    out = str(s)
    for a, b in ((r"\textemdash", "—"), (r"\;", " "), (r"\,", " "), (r"\%", "%"), (r"\_", "_"), ("$", ""),
                 (r"\times10^{", "e"), (r"\circ", "∘"), (r"\tau", "τ"), (r"\le", "≤"), (r"\ge", "≥"),
                 (r"\ell", "ℓ"), (r"\Delta", "Δ"), (r"\textbackslash{}", "\\"), (r"\&", "&"), (r"\#", "#")):
        out = out.replace(a, b)
    out = out.replace("}", "").replace("{", "")
    import re
    return re.sub(r",(?=[^\s])", ", ", out)


class Table:
    """One thesis table: rows of LaTeX cell strings + the numeric CSV + caption, label, note."""

    def __init__(self, key, caption, label, columns, align, note, source):
        self.key, self.caption, self.label = key, caption, label
        self.columns, self.align, self.note, self.source = columns, align, note, source
        self.rows, self.numeric, self.rules = [], [], []

    def add(self, cells, numeric: dict | None = None, rule_before: bool = False):
        if rule_before and self.rows:
            self.rules.append(len(self.rows))
        self.rows.append([str(c) for c in cells])
        self.numeric.append(numeric or {})

    def tex(self) -> str:
        head = " & ".join(tex_escape(c) if "$" not in c else c for c in self.columns)
        body = []
        for i, r in enumerate(self.rows):
            if i in self.rules:
                body.append(r"\midrule")
            body.append(" & ".join(r) + r" \\")
        return "\n".join([
            "%% Generated by %s from %s; do not hand-edit numerical values." % (GENERATOR, ", ".join(self.source)),
            r"\begin{table}[p]", r"\centering", r"\small" if len(self.columns) <= 5 else r"\footnotesize",
            r"\setlength{\tabcolsep}{%.1fpt}" % (3.8 if len(self.columns) <= 5 else 3.0),       # local to the float
            r"\caption{%s}" % tex_escape(self.caption), r"\label{%s}" % self.label,
            r"\begin{tabular}{%s}" % self.align, r"\toprule", head + r" \\", r"\midrule", *body, r"\bottomrule",
            r"\end{tabular}", r"\par\vspace{0.5em}", r"\begin{minipage}{0.97\linewidth}",
            r"\footnotesize " + tex_escape(self.note), r"\end{minipage}", r"\end{table}", ""])

    def markdown(self) -> str:
        cols = [_tex_to_md(c) for c in self.columns]
        out = ["### %s (`%s`)" % (self.caption, self.label), "", "| " + " | ".join(cols) + " |",
               "|" + "|".join("---" for _ in cols) + "|"]
        out += ["| " + " | ".join(_tex_to_md(c) for c in r) + " |" for r in self.rows]
        out += ["", "*%s* Source: %s." % (self.note, ", ".join("`%s`" % s for s in self.source)), ""]
        return "\n".join(out)

    def frame(self):
        import pandas as pd
        return pd.DataFrame([dict(row=i, **n) for i, n in enumerate(self.numeric)])


# ------------------------------------------------------------------------------------------------ helpers
def _boot(d: np.ndarray, reps: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(d), size=(int(reps), len(d)))
    bs = d[picks].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _by_circuit(v, g):
    u = sorted(set(g))
    return np.array([np.nanmean(v[g == k]) for k in u])


def _labels(path):
    import pandas as pd
    t = pd.read_csv(path, encoding="utf-8")
    t["feasible"] = t.feasible.astype(str).str.lower().isin(["true", "1", "1.0"])
    return t


def _part_counts(d: str) -> dict | None:
    from . import labels as LB
    if not os.path.isdir(os.path.join(d, "tasks")):
        return None
    ok, base, failed, R, K, scheme, prot, budgets, name = 0, set(), 0, set(), set(), set(), set(), set(), None
    for tid in LB.task_ids(d):
        try:
            rec = LB.load_record(d, tid)
        except Exception:  # noqa: BLE001 - a corrupted record counts as failed
            failed += 1
            continue
        if rec.get("status") != "ok":
            failed += 1
            continue
        ok += 1
        base.add(rec.get("base_id"))
        R.add(int(rec.get("R", 0)))
        plan = LB.plan_of(rec)
        K.add(int(plan.get("per_half") or 0))
        # the record's own seed scheme (top level since 1.8.0; the plan's key, else scheme 1, for older records)
        scheme.add(int(rec.get("seed_scheme") or plan.get("seed_scheme") or 1))
        prot.add(len(rec.get("protocols", [])))
        budgets.add(len(rec.get("budgets", [])))
    run = os.path.join(d, "run.json")
    if os.path.exists(run):
        name = (TX.read_json(run).get("config") or {}).get("name")
    return dict(tasks=ok, base=len(base), failed=failed, R="/".join(map(str, sorted(R))),
                K="/".join(map(str, sorted(K))), scheme="/".join(map(str, sorted(scheme))),
                protocols="/".join(map(str, sorted(prot))), budgets="/".join(map(str, sorted(budgets))), name=name)


# ------------------------------------------------------------------------------------------------ tables
def t1_studies(run: str) -> Table | None:
    parts = [("dev", "Development", "development"),
             ("confirm", "Confirmation", "confirmatory"),
             ("regimes", "Regime study", "exploratory"),
             ("noisecomp", "Noise composition", "exploratory")]
    roots = sorted(x for x in os.listdir(os.path.join(run, "audit18"))) if os.path.isdir(
        os.path.join(run, "audit18")) else []
    t = Table("t1_studies", "Executed simulation studies of QEM Explorer 1.8.0", "tab:qem180-studies",
              ["Study", "Status", "Tasks", "Circuits", "Budgets", "Candidates", "$r$/half", "$K$/half", "Seeds"],
              "llrrrrrrr",
              "All results are fake-device simulations, not hardware measurements. Circuits are independent logical "
              "(base) circuits; a task is a circuit with an input, observable, layout and device. The controlled "
              "studies reuse circuits of earlier releases' exploration sets and must not be added to obtain a total "
              "independent sample size. Development: out-of-fold, the configuration frozen once on it; confirmation: "
              "new circuits, evaluated once after the freeze. Seeds: seed scheme (2 = namespaced preparation "
              "streams; finding N1). "
              "K = fresh training designs per label half for the learning candidates.",
              [p for p, *_ in parts if os.path.isdir(os.path.join(run, p))] + (["audit18/root_*"] if roots else []))
    any_ = False
    for p, lab, status in parts:
        c = _part_counts(os.path.join(run, p))
        if not c:
            continue
        any_ = True
        t.add([lab + (" (%s)" % c["name"] if c["name"] else ""), status, c["tasks"], c["base"], c["budgets"],
               c["protocols"], c["R"], c["K"], c["scheme"]], dict(study=lab, **c))
    rc = [(r, _part_counts(os.path.join(run, "audit18", r))) for r in roots if r.startswith("root_")]
    rc = [(r, c) for r, c in rc if c]
    if rc:
        any_ = True
        c0 = rc[0][1]
        t.add(["Root audit (%d roots)" % len(rc), "exploratory", "%d$\\times$%d" % (c0["tasks"], len(rc)),
               c0["base"], c0["budgets"], c0["protocols"], c0["R"], c0["K"], c0["scheme"]],
              dict(study="root audit", roots=len(rc), **c0))
    return t if any_ else None


def t2_candidates(run: str) -> Table | None:
    from . import protocols as PR
    from . import registry as RG
    p = os.path.join(run, "analysis", "labels.csv")
    if not os.path.exists(p):
        return None
    lab = _labels(p)
    keys = [k for k in PR.PORTFOLIO_18 if k in set(lab.protocol)]
    t = Table("t2_candidates", "Candidates of portfolio-1.8 and their shot ledgers", "tab:qem180-candidates",
              ["Candidate", "Family", "Validation", "Feasible $10^3$", "Feasible $10^5$", "Feasible $10^7$",
               "Cal.", "Train.", "Target"], "lllrrrrrr",
              "Feasible: development tasks on which the candidate's allocation is feasible at the total budget B "
              "(feasible/all). Ledger: median share of B over feasible development tasks at B = 10^5 spent on "
              "calibration, training and the target circuits (rejected post-selection shots count as target). "
              "Validation: direct = agrees with direct shot-by-shot execution; reference = agrees with an independently "
              "written shot-level reference; both = direct+reference. New in 1.8.0: marked *.",
              ["analysis/labels.csv", "qemx/selection/registry.py"])
    for k in keys:
        q = lab[lab.protocol == k]
        feas = []
        for B in BUDGETS3:
            qb = q[q.B == B]
            feas.append("%d/%d" % (int(qb.feasible.sum()), len(qb)) if len(qb) else r"\textemdash")
        f5 = q[(q.B == 100000) & q.feasible]
        sh = {}
        for col in ("calibration", "training", "execution"):
            c = "shots_%s" % col
            sh[col] = float(np.median(f5[c] / f5.B)) if len(f5) and c in f5 else np.nan
        cap = RG.CAPABILITIES.get(k)
        val = {"direct+reference": "both"}.get(cap.validation, cap.validation) if cap else ""
        t.add([tex_escape(k + (" *" if k not in PR.PORTFOLIO_16 else "")), tex_escape(cap.family if cap else ""),
               val, *feas, pct(sh["calibration"], 0), pct(sh["training"], 0), pct(sh["execution"], 0)],
              dict(protocol=k, validation=cap.validation if cap else "", **{"feasible_%d" % B: f for B, f in
                                                                             zip(BUDGETS3, feas)},
                   **{"share_" + c: v for c, v in sh.items()}))
    return t


POLICY_LABEL = {"RAW": "RAW", "SBS": "SBS", "SBS-B": "SBS-B", "SIZE-B": "SIZE-B",
                "ALL-CANDIDATE": "Empirical reference"}


def _plabel(p: str) -> str:
    if p in POLICY_LABEL:
        return POLICY_LABEL[p]
    for pre, name in (("SELECTIVE[", "Selective "), ("MODEL[", "")):
        if p.startswith(pre):
            fam, fs = p[len(pre):-1].split("|")
            fam = {"hgb": "HGB", "ridge": "Ridge", "struct": "Structured", "enet": "ENet", "mlp": "MLP"}.get(fam, fam)
            fs = {"deployable18": "D18", "deployable18+simulated": "D18+S", "deployable": "D",
                  "deployable+simulated": "D+S"}.get(fs, fs)
            return "%s%s (%s)" % (name, fam, fs)
    return p


def _policy_rows(t: Table, f, frozen: str | None, order):
    for p in order:
        q = f[f.policy == p]
        if not len(q):
            continue
        q = q.iloc[0]
        lab = _plabel(p) + (", frozen" if p == frozen else "")
        t.add([tex_escape(lab), ci(q.mean_log10_risk, q.ci_lo, q.ci_hi, sign=False),
               ci(q.diff_vs_sbsb, q.diff_vs_sbsb_lo, q.diff_vs_sbsb_hi), sci(q.policy_mse),
               pct(q.get("p_severe_vs_raw", np.nan)), sci(q.get("q95_risk", np.nan))],
              dict(policy=p, mean_log10=q.mean_log10_risk, ci_lo=q.ci_lo, ci_hi=q.ci_hi, diff_vs_sbsb=q.diff_vs_sbsb,
                   diff_lo=q.diff_vs_sbsb_lo, diff_hi=q.diff_vs_sbsb_hi, mse=q.policy_mse,
                   p_severe_vs_raw=q.get("p_severe_vs_raw", np.nan), q95_risk=q.get("q95_risk", np.nan)))


def _policy_order(f, frozen):
    pols = list(dict.fromkeys(f.policy))
    fixed = [p for p in ("RAW", "SBS", "SBS-B", "SIZE-B") if p in pols]
    models = sorted(p for p in pols if p.startswith("MODEL["))
    sel = sorted(p for p in pols if p.startswith("SELECTIVE["))
    ref = [p for p in ("ALL-CANDIDATE",) if p in pols]
    return fixed, models + sel, ref


COLS3 = ["Policy", r"Mean $\ell$ [95\%]", r"$\Delta\ell$ vs SBS-B [95\%]", "Mean MSE",
         r"$R>10R_{\mathrm{RAW}}$", r"$q_{95}$ risk"]
NOTE3 = ("All budgets are averaged within a base circuit, then circuits receive equal weight. The primary objective is "
         "l = log10(MSE + 10^-6); Delta l = l(policy) - l(SBS-B), paired percentile bootstrap over base circuits (1000 "
         "resamples). A difference Delta l changes the geometric mean of (MSE + 10^-6) by the factor 10^(Delta l); it "
         "is not a percentage change of the MSE. Mean MSE: arithmetic, circuit-weighted. Severe (risk > 10 x RAW's "
         "risk): declared severe degradation (share of rows); q95: 95 % quantile of the chosen risk. D18 = deployable "
         "features of 1.8.0; D = the deployable features of 1.7.0 (without the 1.8.0 noise-composition features), kept "
         "as frozen comparators; +S = with simulator-only features.")


def _replay_rows(run: str):
    """Full-policy rows of the confirmation (per bundle): mean l with its own bootstrap, the paired difference from
    the replay summary, arithmetic MSE, severe degradation against RAW and q95, from the saved replay tables."""
    import pandas as pd
    from .evaluate import wmean, wquantile
    rdir = os.path.join(run, "replay")
    summ = os.path.join(rdir, "replay_summary.json")
    if not os.path.exists(summ):
        return []
    lab = _labels(os.path.join(run, "confirm", "labels.csv"))
    raw = {(r.task_id, int(r.B)): float(r.mse_eval) for r in lab[lab.protocol == "RAW"].itertuples()}
    out = []
    for name in sorted(TX.read_json(summ), key=lambda n: (0 if "deployable" in n else 1, n)):
        js = os.path.join(rdir, "replay_%s.json" % name)
        cs = os.path.join(rdir, "replay_%s.csv" % name)
        if not (os.path.exists(js) and os.path.exists(cs)):
            continue
        s = TX.read_json(js)
        t = pd.read_csv(cs, encoding="utf-8")
        g = t.base_id.to_numpy()
        r = t.risk.to_numpy(float)
        rr = np.array([raw.get((a, int(b)), np.nan) for a, b in zip(t.task_id, t.B)])
        m, lo, hi = _boot(_by_circuit(np.log10(r + EPS0), g)) if np.isfinite(r).all() else (np.nan,) * 3
        out.append(dict(bundle=name, label="Full policy, %s bundle" % ("default" if "deployable" in name else
                                                                         "simulator"),
                        mean_log10=m, ci_lo=lo, ci_hi=hi, diff=s.get("diff", np.nan), diff_lo=s.get("diff_lo", np.nan),
                        diff_hi=s.get("diff_hi", np.nan), mse=s.get("policy_mse", np.nan),
                        p_severe_vs_raw=wmean(np.where(np.isfinite(r) & np.isfinite(rr), (r > 10 * rr).astype(float),
                                                       np.nan), g), q95_risk=wquantile(r, g, 0.95),
                        unscored=int(s.get("unscored", 0)), rows=len(t),
                        component=s.get("model_component_configuration")))
    return out


def t3_confirmation(run: str) -> Table | None:
    import pandas as pd
    p = os.path.join(run, "confirm", "metrics_confirm.csv")
    if not os.path.exists(p):
        return None
    f = pd.read_csv(p, encoding="utf-8")
    f = f[f.B.astype(str) == "all"]
    frozen = TX.read_json(os.path.join(run, "freeze.json"))["policy"] if os.path.exists(
        os.path.join(run, "freeze.json")) else None
    nb = int(f.base_circuits.max()) if "base_circuits" in f else 0
    nr = int(f.rows.max()) if "rows" in f else 0
    t = Table("t3_confirmation", "Once-only confirmation on %d new logical circuits (%d rows)" % (nb, nr),
              "tab:qem180-confirmation", COLS3, "lrrrrr",
              NOTE3 + " Model rows are model components (the model's best label-feasible candidate); the full-policy "
              "rows call the deployment API itself (guard, out-of-distribution checks, preflight, margin rule, "
              "fallback) and are the application's result.",
              ["confirm/metrics_confirm.csv", "replay/replay_<bundle>.json/.csv", "confirm/labels.csv"])
    fixed, models, ref = _policy_order(f, frozen)
    _policy_rows(t, f, frozen, fixed)
    if models:
        t.rules.append(len(t.rows))
        _policy_rows(t, f, frozen, models)
    rows = _replay_rows(run)
    if rows:
        t.rules.append(len(t.rows))
    for r in rows:
        t.add([tex_escape(r["label"]), ci(r["mean_log10"], r["ci_lo"], r["ci_hi"], sign=False),
               ci(r["diff"], r["diff_lo"], r["diff_hi"]) if np.isfinite(r["diff"]) else
               "undefined (%d unscored)" % r["unscored"], sci(r["mse"]), pct(r["p_severe_vs_raw"]),
               sci(r["q95_risk"])], dict(policy="POLICY|" + r["bundle"], **{k: v for k, v in r.items()
                                                                             if k not in ("label",)}))
    if ref:
        t.rules.append(len(t.rows))
        _policy_rows(t, f, frozen, ref)
    return t


def t3b_development(run: str) -> Table | None:
    import pandas as pd
    p = os.path.join(run, "analysis", "pooled_same-distribution.csv")
    if not os.path.exists(p):
        return None
    f = pd.read_csv(p, encoding="utf-8")
    f = f[f.B.astype(str) == "all"]
    frozen = TX.read_json(os.path.join(run, "freeze.json"))["policy"] if os.path.exists(
        os.path.join(run, "freeze.json")) else None
    nb = int(f.base_circuits.max()) if "base_circuits" in f else 0
    nr = int(f.rows.max()) if "rows" in f else 0
    t = Table("t3b_development", "Development performance, grouped five-fold out-of-fold (%d circuits, %d rows)"
              % (nb, nr), "tab:qem180-development", COLS3, "lrrrrr",
              NOTE3 + " Out-of-fold predictions (folds grouped by base circuit); the frozen configuration was chosen "
              "on these numbers, so they are not confirmatory.", ["analysis/pooled_same-distribution.csv"])
    fixed, models, ref = _policy_order(f, frozen)
    _policy_rows(t, f, frozen, fixed)
    t.rules.append(len(t.rows))
    keep = [m for m in models if any(s in m for s in ("deployable18", "size_budget")) or m == frozen]
    _policy_rows(t, f, frozen, keep)
    t.rules.append(len(t.rows))
    _policy_rows(t, f, frozen, ref)
    return t


def t4_per_budget(run: str) -> Table | None:
    from . import protocols as PR
    p = os.path.join(run, "analysis", "labels.csv")
    if not os.path.exists(p):
        return None
    lab = _labels(p)
    n_tasks = lab.task_id.nunique()
    t = Table("t4_per_budget", "Representative per-candidate MSEs at three total shot budgets", "tab:qem180-per-budget",
              ["Candidate", "$B=10^3$", "$B=10^5$", "$B=10^7$"], "lrrr",
              "Median evaluation-half MSE over the feasible development tasks, with the number of feasible tasks in "
              "brackets (out of %d). Candidates have different feasible subsets: this table is descriptive and must "
              "not be read as a ranking; paired comparisons are in the confirmation and development tables." % n_tasks,
              ["analysis/labels.csv"])
    for k in [k for k in PR.PORTFOLIO_18 if k in set(lab.protocol)]:
        cells, num = [], dict(protocol=k)
        for B in BUDGETS3:
            q = lab[(lab.protocol == k) & (lab.B == B) & lab.feasible]
            m = float(np.median(q.mse_eval)) if len(q) else np.nan
            cells.append("%s [%d]" % (sci(m), len(q)))
            num.update({"median_mse_%d" % B: m, "feasible_%d" % B: len(q)})
        t.add([tex_escape(k)] + cells, num)
    return t


def t5_tails(run: str) -> Table | None:
    from . import protocols as PR
    from .evaluate import SEVERE_VS_RAW, tail_metrics, wmean, wquantile
    p = os.path.join(run, "analysis", "labels.csv")
    if not os.path.exists(p):
        return None
    lab = _labels(p)
    raw = lab[lab.protocol == "RAW"].set_index(["task_id", "B"]).mse_eval
    lab = lab.join(raw.rename("raw_mse"), on=["task_id", "B"])
    t = Table("t5_tails", "Infeasibility, fallbacks, severe degradation and loss concentration per candidate",
              "tab:qem180-tails",
              ["Candidate", "Infeasible/error", "Flagged trials", "No estimate", r"$R>10R_{\mathrm{RAW}}$",
               r"$q_{95}$ MSE", "Worst 5 %"], "lrrrrrr",
              "Development set, all budgets. Infeasible/error: share of (task, budget) rows without a feasible "
              "allocation or with a recorded method error. Flagged trials: share of the "
              "evaluation trials of feasible rows with any method fallback or failure flag (part of the risk, never "
              "removed); No estimate: nonresponse share. Severe (risk > 10 x RAW's risk): circuit-weighted share of feasible rows whose "
              "risk exceeds ten times RAW's on the same row (declared before the 1.8.0 results). q95: 95 % quantile of "
              "the risk over feasible rows. Worst 5 %: share of the candidate's total circuit-averaged loss carried by "
              "its worst 5 % of base circuits. Each candidate on its own feasible subset (descriptive).",
              ["analysis/labels.csv"])
    for k in [k for k in PR.PORTFOLIO_18 if k in set(lab.protocol)]:
        q = lab[lab.protocol == k]
        fz = q[q.feasible & np.isfinite(q.mse_eval)]
        inf_ = float(1 - q.feasible.mean())
        flag = float(fz.fail_any_eval.mean()) if len(fz) and "fail_any_eval" in fz else np.nan
        nores = float(fz.nonresponse_eval.mean()) if len(fz) and "nonresponse_eval" in fz else np.nan
        g = fz.base_id.to_numpy()
        sev = wmean((fz.mse_eval > SEVERE_VS_RAW * fz.raw_mse).astype(float).to_numpy(), g) if len(fz) else np.nan
        q95 = wquantile(fz.mse_eval.to_numpy(float), g, 0.95) if len(fz) else np.nan
        w5 = tail_metrics(fz.mse_eval.to_numpy(float), g)["worst5_loss_share"] if len(fz) else np.nan
        t.add([tex_escape(k), pct(inf_), pct(flag), pct(nores), pct(sev), sci(q95), pct(w5, 0)],
              dict(protocol=k, infeasible=inf_, flagged_trials=flag, no_estimate=nores, p_severe_vs_raw=sev,
                   q95_mse=q95, worst5_loss_share=w5, feasible_rows=len(fz), rows=len(q)))
    return t


def t6_costs(run: str) -> Table | None:
    import pandas as pd
    p = os.path.join(run, "analysis", "runtime.csv")
    if not os.path.exists(p):
        return None
    rt = pd.read_csv(p, encoding="utf-8")
    rej = rt[rt.kind == "rejection"] if "kind" in rt else rt.iloc[0:0]
    w = rt[rt.kind == "width"] if "kind" in rt else rt
    cols = [("compile_s", "Compile"), ("deployable_s", "Features"), ("preflight_structural_s", "Struct. pref."),
            ("preflight_design_s", "Design pref."), ("inference_s", "Inference"), ("recommend_s", "Whole call"),
            ("simulated_s", "Sim. features"), ("label_s", "Labels (study)")]
    cols = [(c, n) for c, n in cols if c in w.columns]
    g = w.groupby("n")[[c for c, _ in cols]].median()
    t = Table("t6_costs", "Classical cost of one recommendation (seconds, median of repeats)", "tab:qem180-costs",
              ["$n$"] + [n for _, n in cols], "r" + "r" * len(cols),
              "Measured on the build machine (2 shared cores): magnitudes, not a benchmark. %s circuit, fake_fez, "
              "B = 10^5. Features: the deployable features without cache; Struct./Design pref.: the structural "
              "preflight of every candidate and the design-level preflight of the learning candidates; Whole call: "
              "deploy.recommend with the default bundle; Sim. features: the simulator-only features (exponential in "
              "n; only for a model that uses them); Labels: generating one development task's benchmark labels (a "
              "study cost, not a recommendation cost). No component uses quantum shots.%s" % (
                  str(w.family.iloc[0]) if "family" in w and len(w) else "",
                  (" Early rejection (whole call, GHZ-type chain outside the training widths): " + "; ".join(
                      "n = %d: %.2f s, %s" % (r.n, r.recommend_s, r.recommend_status) for r in rej.itertuples()))
                  if len(rej) else ""), ["analysis/runtime.csv"])
    for n, r in g.iterrows():
        t.add(["%d" % n] + [("%.3g" % r[c]) if np.isfinite(r[c]) else r"\textemdash" for c, _ in cols],
              dict(n=int(n), **{c: float(r[c]) for c, _ in cols}))
    for r in rej.itertuples():                    # in the CSV and the note, not as table rows
        t.numeric.append(dict(kind="rejection", n=int(r.n), recommend_s=float(r.recommend_s),
                              status=r.recommend_status))
    return t


CONTRAST_TEX = {
    "expansion_fixed_policy": "More candidates, fixed policy (SBS-B 1.8 - SBS-B 1.6)",
    "expansion_attainable": "More candidates, attainable (reference 1.8 - reference 1.6)",
    "learning_within_1.6_ridge": "Learning within portfolio-1.6 (Ridge - SBS-B)",
    "learning_within_1.6_hgb": "Learning within portfolio-1.6 (HGB - SBS-B)",
    "learning_within_1.8_ridge": "Learning within portfolio-1.8 (Ridge - SBS-B)",
    "learning_within_1.8_hgb": "Learning within portfolio-1.8 (HGB - SBS-B)",
    "total_ridge": "Total (Ridge 1.8 - SBS-B 1.6)",
    "total_hgb": "Total (HGB 1.8 - SBS-B 1.6)",
}


def t7_portfolio(run: str) -> Table | None:
    import pandas as pd
    p = os.path.join(run, "portfolio", "portfolio_contrasts.csv")
    if not os.path.exists(p):
        return None
    con = pd.read_csv(p, encoding="utf-8").set_index("contrast")
    res = TX.read_json(os.path.join(run, "portfolio", "portfolio_comparison.json"))
    t = Table("t7_portfolio", "Original against expanded portfolio on identical development tasks",
              "tab:qem180-portfolio", ["Contrast", r"$\Delta\ell$ [95\%]", r"Factor $10^{\Delta\ell}$"], "lrr",
              "Out-of-fold development results (%d rows, %d base circuits; feature set %s; every policy refitted per "
              "portfolio). Delta l: paired difference of the mean over base circuits of log10(MSE + 10^-6), percentile "
              "bootstrap over base circuits; the factor is the change of the geometric mean of (MSE + 10^-6). "
              "Exploratory." % (res.get("rows", 0), res.get("base_circuits", 0), res.get("feature_set", "")),
              ["portfolio/portfolio_contrasts.csv", "portfolio/portfolio_comparison.json"])
    for k, lab in CONTRAST_TEX.items():
        if k in con.index:
            r = con.loc[k]
            t.add([tex_escape(lab), ci(r["diff"], r["lo"], r["hi"]), "$%.3f$" % (10 ** r["diff"])],
                  dict(contrast=k, diff=r["diff"], lo=r["lo"], hi=r["hi"], factor=10 ** r["diff"]))
    return t


TABLES = (t1_studies, t2_candidates, t3_confirmation, t3b_development, t4_per_budget, t5_tails, t6_costs,
          t7_portfolio)

WRAPPER = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{booktabs,amsmath,textcomp}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,skip=8pt}
\setlength{\tabcolsep}{3.8pt}
\renewcommand{\arraystretch}{1.2}
\renewcommand{\floatpagefraction}{0.25}
\hypersetup{pdftitle={QEM Explorer %(release)s: numerical tables}}
\pagestyle{myheadings}
\markright{QEM Explorer %(release)s -- simulation tables (fake devices; no hardware data)}
\begin{document}
%(inputs)s
\end{document}
"""


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(run: str, out: str, *, release: str | None = None, compile_pdf: bool = True) -> dict:
    from .. import __version__
    release = release or __version__
    os.makedirs(out, exist_ok=True)
    made, skipped, prov = [], [], dict(generator=GENERATOR, release=release, run=run, tables={})   # the path as given
    md = ["# Numerical tables — QEM Explorer %s" % release, "",
          "Generated by `%s` from the saved tables of the run; the LaTeX fragments (`*.tex`), the CSV files (the "
          "numbers) and this Markdown carry the same values. Simulated observations (emulated fake-device trials); "
          "no hardware data." % GENERATOR, ""]
    for fn in TABLES:
        t = fn(run)
        if t is None:
            skipped.append(fn.__name__)
            continue
        with open(os.path.join(out, t.key + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t.tex())
        t.frame().to_csv(os.path.join(out, t.key + ".csv"), index=False, encoding="utf-8")
        md.append(t.markdown())
        srcs = {}
        for s in t.source:
            p = os.path.join(run, s)
            if os.path.isfile(p):
                srcs[s] = _sha(p)
        prov["tables"][t.key] = dict(label=t.label, caption=t.caption, sources=srcs)
        made.append(t.key)
    if skipped:
        md += ["## Not generated", "", "Source tables missing for: %s." % ", ".join(skipped), ""]
    with open(os.path.join(out, "thesis_tables.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))
    name = "qem%s_thesis_tables" % release.replace(".", "")
    with open(os.path.join(out, name + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(WRAPPER % dict(release=release, inputs="\n\\clearpage\n".join(r"\input{%s.tex}" % k for k in made)))
    pdf = None
    if compile_pdf and shutil.which("pdflatex") and made:
        try:
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=out,
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            pdf = name + ".pdf"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            pdf = "failed: %s (see %s.log)" % (type(e).__name__, name)
        for ext in (".aux", ".out"):
            p = os.path.join(out, name + ext)
            if os.path.exists(p):
                os.remove(p)
    prov.update(made=made, skipped=skipped, pdf=pdf or "not compiled (pdflatex unavailable or disabled)")
    TX.write_json(os.path.join(out, "provenance.json"), prov)
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="thesis-ready numerical tables (CSV, LaTeX, Markdown, PDF)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    r = build(a.run, a.out, compile_pdf=not a.no_pdf)
    print(TX.dumps(dict(made=r["made"], skipped=r["skipped"], pdf=r["pdf"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
