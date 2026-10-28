r"""Figure 1 and the numerical tables of the 1.9.0 research-readiness release, from saved tables only (1.9.0;
prompt §5; ``docs/selection/FIGURE_TABLE_SPEC_v19.md``).

    python -m qemx.selection.tables19 --v18 reports/selection_v180 --audit reports/execution_audit_v190 \
        --pilot reports/pilot_v190 --out reports/tables_v190

Tables (``tab:qem190-*``; CSV with the numbers, includeable UTF-8 LaTeX with ``booktabs``, Markdown, and a minimal
compilable wrapper, compiled to PDF when ``pdflatex`` exists):

* ``s1_datasets`` — studies, their status (historical, exploratory pilot, planned), tasks, base and logical circuits,
  families, widths, base devices, noise settings, budgets, trials and designs per half;
* ``s2_candidates`` — the 17 candidates: family, learning, prerequisites, resources and ledger rule, fallback;
* ``s3_primary`` — the 1.8.0 once-only confirmation by budget and overall: full deployed policy minus SBS-B in the
  base-circuit mean of log10(R + 10⁻⁶) with 95 % bootstrap intervals over base circuits, arithmetic MSE, circuits;
  primary (D41) and secondary bundle; recomputed from the replay rows and the labels derived from the arrays;
* ``s4_execution`` — the joint recommendation → verified design → execution audit: the contrast to SBS-B under the
  executed (returned-design) risk and under the fresh-design labels, and their difference;
* ``s5_pilot`` — the 1.9.0 pilot (exploratory): out-of-fold contrasts to SBS-B per split (same distribution, held-out
  noise setting, base device, joint family × noise) and the portfolio contrasts;
* ``s6_costs`` — measured labeling cost per width (pilot), the conditional estimate of the planned development study
  (its wall time for a declared hypothetical computer, 1.9.1), the structural size of its analysis (model fits, not a
  time; 1.9.1), and the classical seconds of a recommendation.

Every value is taken from a named source file (``provenance.json`` lists them with their sha256); the mapping from
source columns to table columns is tested (``tests/test_tables19.py``).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from .. import textio as TX
from .thesis_tables import Table, ci, ci_md, sci, _sha
from .thesis_tables import tex_escape as _tex_escape
from .models import EPS0

# 1.9.1 (independent review F4): the planning table's wall time is a declared hypothetical computer, not the computer
# that generates the table (1.9.0 capped the workers at the generating computer's CPUs under a fixed caption)
SCENARIO_CORES = 8
SCENARIO_WORKERS = 7


def tex_escape(s) -> str:
    """``thesis_tables.tex_escape`` plus '|' as \\textbar (policy names such as MODEL[hgb|deployable18])."""
    return _tex_escape(s).replace("|", r"\textbar{}")


GENERATOR = "qemx.selection.tables19/1"
LABELS = {"qemx-rec-1.8.0": "primary: frozen configuration, simulator features (allocation-level preflight)",
          "qemx-rec-1.8.0-deployable": "secondary: default bundle, deployable features"}
METRIC = ("Δ = mean over base circuits g of the mean over that circuit's rows (task variants × budgets) of "
          "log10(R_policy + 10⁻⁶) − log10(R_SBS-B + 10⁻⁶); R = fresh-design average risk (evaluation half); 10^Δ is a "
          "factor on the geometric mean of R + 10⁻⁶, not on the MSE; 95 % percentile bootstrap over base circuits "
          "(1000 resamples, seed 0).")


def _boot(d, reps=1000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(int(reps), len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


# ------------------------------------------------------------------------------------------------ sources
BASELINE_CHECK: dict = {}          # how the confirmation's SBS-B column was checked (written to provenance.json)


def confirmation_rows(v18: str) -> dict:
    """{bundle: DataFrame of replay rows with the SBS-B risk} from the 1.8.0 confirmation (validated labels)."""
    import pandas as pd
    from . import replay as RP
    pop = RP.bound_population(os.path.join(v18, "confirm"))
    tab = pop["table"]
    risk = {(r.task_id, r.protocol, int(r.B)): float(r.mse_eval) for r in tab.itertuples()}
    ch = pd.read_csv(os.path.join(v18, "confirm", "choices_confirm.csv"), encoding="utf-8")
    # the baseline column must follow the SBS-B rule (independent verification): refitted on the confirmation's own
    # training labels when they are extracted, else the fallback ranks of the shipped bundles trained on them
    from . import deploy as DP
    metas = [(n, DP.load_bundle(os.path.join(DP.BUNDLE_DIR, n)).meta) for n in LABELS
             if os.path.isdir(os.path.join(DP.BUNDLE_DIR, n))]
    rule, rule_source = RP.sbsb_rule_choices(os.path.join(v18, "confirm"), tab, metas)
    keyed = {(t, int(b)): {"SBS-B|choice": c} for t, b, c in zip(ch.task_id, ch.B, ch["SBS-B|choice"])}
    checked = RP.check_sbsb_column(keyed, rule, what="choices_confirm.csv")
    BASELINE_CHECK.clear()
    BASELINE_CHECK.update(source=rule_source, rows_checked=checked, rows=len(keyed))
    sb = {(t, int(b)): risk[(t, c, int(b))] for t, b, c in zip(ch.task_id, ch.B, ch["SBS-B|choice"])}
    out = {}
    for name in LABELS:
        p = os.path.join(v18, "replay", "replay_%s.csv" % name)
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p, encoding="utf-8")
        t["sbsb"] = [sb[(a, int(b))] for a, b in zip(t.task_id, t.B)]
        out[name] = t
    return out


def contrast(t, col="risk", base="sbsb", reps=1000, seed=0) -> dict:
    d = np.log10(t[col].astype(float) + EPS0) - np.log10(t[base].astype(float) + EPS0)
    g = t.base_id.to_numpy()
    per = np.array([d[g == k].mean() for k in sorted(set(g))])
    m, lo, hi = _boot(per, reps, seed)
    mse_p = np.array([t[col].astype(float)[g == k].mean() for k in sorted(set(g))]).mean()
    mse_b = np.array([t[base].astype(float)[g == k].mean() for k in sorted(set(g))]).mean()
    return dict(diff=m, lo=lo, hi=hi, factor=10 ** m, mse_policy=float(mse_p), mse_sbsb=float(mse_b),
                circuits=int(len(per)), rows=int(len(t)))


# ------------------------------------------------------------------------------------------------ tables
def s1_datasets(src: dict):
    from . import design as DS, ancestry as AY
    root = src["root"]
    t = Table("s1_datasets", "Datasets of the 1.8.0 study, the 1.9.0 pilot and the planned 1.9 studies",
              "tab:qem190-datasets",
              ["study", "status", "tasks", "base", "logical", "families", "widths", "devices", "noise settings",
               "budgets", "$r$/half", "designs/half"],
              "lp{2.3cm}rrrrllp{3.9cm}lrr",
              "Counts from each configuration's expansion (historical task identities unchanged); a task is a "
              "(circuit, input, observable, layout, device) variant; base and logical circuits are the independent "
              "units. Noise settings are calibration-supplied snapshot modifiers (s: overall scale, g2: two-qubit "
              "errors, t: relaxation, ro: readout, chi: coherent fraction). Planned studies are not executed.",
              ["configs/selection/*.yaml"])
    for name, status in (("dev18", "historical development (1.8.0)"), ("conf18", "historical confirmation (1.8.0)"),
                         ("pilot19", "exploratory pilot (1.9.0)"), ("fulldev19", "planned development"),
                         ("conf19", "planned confirmation")):
        p = os.path.join(root, "configs", "selection", name + ".yaml")
        if not os.path.exists(p):
            continue
        cfg = DS.load(p)
        rep = DS.expansion_report(cfg)
        specs = DS.expand(cfg)
        ld = cfg.get("learning_designs") or {}
        nz = ", ".join("%s %d" % kv for kv in rep["by_noise"].items())
        bs = sorted(int(b) for b in cfg["budgets"])
        t.add([name, tex_escape(status), str(rep["tasks"]), str(rep["base_circuits"]), str(rep["logical_circuits"]),
               str(len(rep["by_family"])), "%d--%d" % (min(rep["by_width"]), max(rep["by_width"])),
               str(len(rep["by_device"])), tex_escape(nz), "%d: %s--%s" % (len(bs), sci(bs[0], 0), sci(bs[-1], 0)),
               str(cfg["R"]), str(ld.get("per_half", r"\textemdash"))],
              dict(study=name, status=status, tasks=rep["tasks"], base_circuits=rep["base_circuits"],
                   logical_circuits=rep["logical_circuits"], families=len(rep["by_family"]), devices=len(rep["by_device"]),
                   noise_settings=len(rep["by_noise"]), budgets=len(bs), R=cfg["R"], designs_per_half=ld.get("per_half"),
                   independent=len({AY.logical_id(s) for s in specs})))
    return t


def s2_candidates(src: dict):
    from . import registry as RG, protocols as PR
    t = Table("s2_candidates", "The 17 candidate protocols: applicability and resources", "tab:qem190-candidates",
              ["candidate", "family", "learning", "prerequisites", "resources and shot ledger", "on failure"],
              "lllp{3.6cm}p{4.6cm}p{3.4cm}",
              "From the capability registry (qemx/selection/registry.py). Every shot of calibration, training and "
              "target circuits is charged to the total budget B; learning candidates also build a training design "
              "classically before execution (preflight level 'design'). VD and M3 are not candidates: no faithful "
              "finite-shot implementation with a resource ledger exists in the study.",
              ["qemx/selection/registry.py"])
    for k in PR.PORTFOLIO_18:
        d = RG.describe(k)
        t.add([tex_escape(k), tex_escape(d["family"]), "yes" if PR.PORTFOLIO[k].learning else "no",
               tex_escape(d.get("prerequisites", "")), tex_escape("%s; %s" % (d.get("resources", ""), d.get("ledger", ""))),
               tex_escape(d.get("failure", ""))],
              dict(candidate=k, family=d["family"], learning=bool(PR.PORTFOLIO[k].learning)))
    return t


def s3_primary(src: dict):
    rows = src.get("confirmation")
    if not rows:
        return None
    t = Table("s3_primary", "Once-only confirmation of the 1.8.0 policy by budget (full deployed policy − SBS-B)",
              "tab:qem190-primary", ["bundle", "budget", "$\\Delta$ [95\\% CI]", "$10^{\\Delta}$", "MSE policy",
                                     "MSE SBS-B", "circuits"], "llrrrrr",
              METRIC + " Primary contrast (D41) specified after the development results and before the confirmation "
                       "results; 29 base circuits, 58 tasks, 10 budgets. Per-budget rows are descriptive (10 "
                       "comparisons, no multiplicity correction).",
              ["replay/replay_<bundle>.csv", "confirm/choices_confirm.csv", "confirm/tasks/*.npz"])
    for name, tb in rows.items():
        first = True
        for B in sorted(set(int(b) for b in tb.B)):
            c = contrast(tb[tb.B == B])
            t.add([tex_escape(LABELS[name]) if first else "", sci(B, 0), ci(c["diff"], c["lo"], c["hi"]),
                   "$%.2f$" % c["factor"], sci(c["mse_policy"]), sci(c["mse_sbsb"]), str(c["circuits"])],
                  dict(bundle=name, B=B, **c), rule_before=first)
            first = False
        c = contrast(tb)
        t.add(["", "all", ci(c["diff"], c["lo"], c["hi"]), "$%.2f$" % c["factor"], sci(c["mse_policy"]),
               sci(c["mse_sbsb"]), str(c["circuits"])], dict(bundle=name, B="all", **c))
    return t


def s4_execution(src: dict):
    s = src.get("audit")
    if not s:
        return None
    t = Table("s4_execution", "Executed (returned-design) risk against the fresh-design label: joint audit",
              "tab:qem190-execution", ["policy", "$\\Delta$ executed", "$\\Delta$ fresh-design",
                                       "executed − fresh", "rows", "circuits"], "p{4.6cm}rrrrr",
              "Eight inspected base circuits (the 1.8.0 root audit's tasks), budgets 10⁴, 3·10⁵, 10⁷; deployment roots: "
              "the API's own design seeds and four independent redraws; each returned candidate executed with its "
              "verified design and the API's allocation (%d trials per row); fresh-design labels: mean of four "
              "preparation roots. Δ: base-circuit mean of log10(R + 10⁻⁶) minus SBS-B's (the candidate the API verifies "
              "for SBS-B, executed the same way), 95 %% bootstrap over base circuits. Exploratory: not a confirmation."
              % s.get("trials_per_row", 0),
              ["execution_audit_summary.json"])
    for pol, v in s["policies"].items():
        e, f, d = v.get("executed", {}), v.get("fresh", {}), v.get("executed_minus_fresh_contrast", {})
        name = pol.replace("POLICY|", "")
        t.add([tex_escape(LABELS.get(name, name)), ci(e.get("diff_vs_sbsb"), e.get("lo"), e.get("hi")),
               ci(f.get("diff_vs_sbsb"), f.get("lo"), f.get("hi")), ci(d.get("diff"), d.get("lo"), d.get("hi")),
               str(v.get("rows")), str(s["base_circuits"])],
              dict(policy=pol, executed=e.get("diff_vs_sbsb"), executed_lo=e.get("lo"), executed_hi=e.get("hi"),
                   fresh=f.get("diff_vs_sbsb"), fresh_lo=f.get("lo"), fresh_hi=f.get("hi"), difference=d.get("diff"),
                   difference_lo=d.get("lo"), difference_hi=d.get("hi"), rows=v.get("rows")))
    return t


def s5_pilot(src: dict):
    import pandas as pd
    pilot = src.get("pilot")
    if not pilot or not os.path.isdir(os.path.join(pilot, "analysis")):
        return None
    frz = os.path.join(pilot, "freeze.json")
    policy = TX.read_json(frz)["policy"] if os.path.exists(frz) else None
    t = Table("s5_pilot", "The 1.9.0 pilot (exploratory): learned selection against SBS-B by split",
              "tab:qem190-pilot", ["split", "policy", "$\\Delta$ vs SBS-B [95\\% CI]", "held-out rows", "circuits"],
              "llrrr",
              "Out-of-fold pooled results of the pilot (27 base circuits, 54 tasks, 4 budgets, noise-diverse "
              "training). On the same-distribution split the policy is the configuration the pilot's freeze chose; "
              "the other splits evaluate a declared subset of model families and feature sets, all listed (no "
              "post-hoc best). ALL-CANDIDATE is the empirical reference (best on the selection half). Exploratory: a "
              "pilot for "
              "planning, too small for conclusions; held-out-noise and joint splits test transfer to an unseen noise "
              "setting. " + METRIC.split(";")[0] + ".",
              ["analysis/pooled_<split>.csv", "freeze.json"])
    for f in sorted(os.listdir(os.path.join(pilot, "analysis"))):
        if not (f.startswith("pooled_") and f.endswith(".csv")):
            continue
        split = f[len("pooled_"):-4]
        q = pd.read_csv(os.path.join(pilot, "analysis", f), encoding="utf-8")
        q = q[q.B.astype(str) == "all"]
        cols = [c for c in ("diff_vs_sbsb", "diff_vs_sbsb_lo", "diff_vs_sbsb_hi") if c in q.columns]
        if len(cols) < 3:
            continue
        # only the frozen configuration is reported as "the policy"; a split that did not evaluate it (other splits
        # compare a declared subset of families and sets) shows each declared comparator, never a post-hoc best model
        if policy and policy in set(q.policy):
            pols = [policy]
        else:
            pols = sorted(p for p in set(q.policy) if str(p).startswith("MODEL["))
        pols += [p for p in ("ALL-CANDIDATE", "SBS") if p in set(q.policy)]
        first = True
        for p in pols:
            r = q[q.policy == p].iloc[0]
            t.add([tex_escape(split) if first else "", tex_escape(p), ci(r.diff_vs_sbsb, r.diff_vs_sbsb_lo,
                                                                         r.diff_vs_sbsb_hi),
                   str(int(r.get("rows", 0) or 0)), str(int(r.get("base_circuits", 0) or 0))],
                  dict(split=split, policy=p, diff=float(r.diff_vs_sbsb), lo=float(r.diff_vs_sbsb_lo),
                       hi=float(r.diff_vs_sbsb_hi)), rule_before=first)
            first = False
    pc = os.path.join(pilot, "portfolio", "portfolio_contrasts.csv")
    if os.path.exists(pc):
        con = pd.read_csv(pc, encoding="utf-8")
        first = True
        for r in con.itertuples():
            t.add(["portfolio" if first else "", tex_escape(r.contrast), ci(r.diff, r.lo, r.hi), "", ""],
                  dict(split="portfolio", policy=r.contrast, diff=float(r.diff), lo=float(r.lo), hi=float(r.hi)),
                  rule_before=first)
            first = False
    return t


def s6_costs(src: dict):
    pilot = src.get("pilot")
    t = Table("s6_costs", "Measured costs: labeling per task (pilot) and the planned development study",
              "tab:qem190-costs", ["item", "value", "basis"], "lrp{8.5cm}",
              "Labeling: wall seconds of a task's supervised child process (one numerical thread; 17 candidates, every "
              "budget, both halves) — the cost of building the study, not of a recommendation. The wall time is a "
              "declared scenario (a hypothetical %d-core computer running %d workers with perfect parallel "
              "efficiency), not a measurement and independent of the computer that generates this table. Analysis "
              "is sized by its model fits, not in hours (not measured for fulldev19)." % (SCENARIO_CORES,
                                                                                      SCENARIO_WORKERS),
              ["dev/tasks/*.json", "plan --measured", "configs/selection/*.yaml"])
    if pilot and os.path.isdir(os.path.join(pilot, "dev", "tasks")):
        from . import labels as LB, run as RN
        m = RN.measured_rates(os.path.join(pilot, "dev"))
        recs = [LB.load_record(os.path.join(pilot, "dev"), x) for x in LB.task_ids(os.path.join(pilot, "dev"))][:1]
        shape = ""
        if recs:
            r0 = recs[0]
            ld0 = r0.get("learning_designs")
            ph0 = ld0.get("per_half", "?") if isinstance(ld0, dict) else ld0
            shape = " (R = %s per half, %d budgets, %s designs/half)" % (r0.get("R"), len(r0.get("budgets") or []), ph0)
        for n in sorted({x["n"] for x in m}):
            v = [x["seconds"] for x in m if x["n"] == n]
            t.add(["pilot labeling, width %d" % n, "$%.0f$ s" % float(np.median(v)),
                   "median of %d tasks%s" % (len(v), shape)],
                  dict(item="label_seconds_median", n=n, seconds=float(np.median(v)), tasks=len(v)))
        fd = os.path.join(src["root"], "configs", "selection", "fulldev19.yaml")
        if os.path.exists(fd):
            from . import design as DS, resources as RS
            fdc = DS.load(fd)
            pre = RS.preflight(fdc, None, workers=1, measured_dir=os.path.join(pilot, "dev"))
            c = pre["cpu"]
            t.add(["fulldev19 labeling (estimate)", "$%.1f$ CPU-h" % c["total_hours"],
                   tex_escape("conditional estimate from the pilot's measured rates (measured on two shared cores, so "
                              "contention-inflated) for fulldev19's R = %s, %d budgets and %s designs/half: per-trial "
                              "time per trial, preparation scaled by the number of designs (x%s); widths extrapolated: "
                              "%s. Not a proven upper bound: the coherent (chi) tasks are unmeasured" % (
                                  fdc.get("R"), len(fdc.get("budgets") or []),
                                  (fdc.get("learning_designs") or {}).get("per_half", "?"),
                                  "%.2f" % float(c.get("preparation_scale") or 1.0), c["extrapolated_widths"] or "none"))],
                  dict(item="fulldev19_cpu_hours", value=c["total_hours"]))
            wall = c["total_hours"] / SCENARIO_WORKERS                    # 1.9.1 (F4): computed for the scenario
            t.add(["fulldev19 labeling wall time, scenario: %d workers" % SCENARIO_WORKERS, "$%.1f$ h" % wall,
                   "CPU-hours / %d: a hypothetical %d-core computer with one core left free, perfect parallel "
                   "efficiency; on another computer, plan --workers auto reports its own value" % (
                       SCENARIO_WORKERS, SCENARIO_CORES)],
                  dict(item="fulldev19_wall_hours_scenario", value=wall, workers=SCENARIO_WORKERS,
                       cores=SCENARIO_CORES, cpu_hours=c["total_hours"]))
            t.add(["fulldev19 labels on disk (estimate)", "$%.2f$ GB" % pre["disk"]["labels_total_gb"],
                   "measured file sizes of the pilot per trial"],
                  dict(item="fulldev19_disk_gb", value=pre["disk"]["labels_total_gb"]))
            pc = os.path.join(src["root"], "configs", "selection", "pilot19.yaml")
            if os.path.exists(pc):
                wf, wp = RS.analysis_workload(fdc), RS.analysis_workload(DS.load(pc))
                fits_tex = "$%s$" % format(wf["fits"], ",").replace(",", r"\,")
                t.add(["fulldev19 analysis, main splits (model fits)", fits_tex,
                       tex_escape("not a time: %d split parts, fits including inner tuning; %.1fx the pilot's %s fits "
                                  "and %.0fx its fits weighted by training rows (rows: %.1fx). Per-fit cost differs by "
                                  "family; %d extras not counted. Time analyze --splits same-distribution first" % (
                                      sum(v["parts"] for v in wf["splits"].values()), wf["fits"] / wp["fits"],
                                      format(wp["fits"], ",").replace(",", " "), wf["row_fits"] / wp["row_fits"],
                                      wf["rows"] / wp["rows"], len(wf["extras"])))],
                      dict(item="fulldev19_analysis_fits", value=wf["fits"], row_fits=wf["row_fits"],
                           pilot_fits=wp["fits"], pilot_row_fits=wp["row_fits"], rows=wf["rows"],
                           pilot_rows=wp["rows"]))
    rs = os.path.join(src.get("v18", ""), "replay", "replay_summary.json")
    if os.path.exists(rs):
        s = TX.read_json(rs)
        for name, v in s.items():
            t.add([tex_escape("recommendation call, %s" % name), "$%.2f$ s" % v.get("seconds_median_fresh", math.nan),
                   "median classical wall time of a call that recomputed its features (1.8.0 confirmation)"],
                  dict(item="recommend_seconds", bundle=name, value=v.get("seconds_median_fresh")))
    return t if t.rows else None


TABLES = (s1_datasets, s2_candidates, s3_primary, s4_execution, s5_pilot, s6_costs)


# ------------------------------------------------------------------------------------------------ Figure 1
def figure1(src: dict, out: str) -> str | None:
    """Figure 1 of the 1.9.0 specification: actual policy benefit (see the module text and the spec)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _plt
    plt = _plt()
    rows = src.get("confirmation") or {}
    if not rows and not src.get("audit") and not src.get("pilot"):
        return None
    col = {"qemx-rec-1.8.0": "#0072B2", "qemx-rec-1.8.0-deployable": "#E69F00"}
    mk = {"qemx-rec-1.8.0": "o", "qemx-rec-1.8.0-deployable": "s"}
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.4), layout="constrained")
    A, Bp, C, D = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    data = []
    for k, (name, tb) in enumerate(rows.items()):
        bs = sorted(set(int(b) for b in tb.B))
        cs = [contrast(tb[tb.B == b]) for b in bs]
        off = 1.0 + (k - 0.5) * 0.12
        x = np.array(bs, float) * off
        A.errorbar(x, [c["diff"] for c in cs], yerr=[[c["diff"] - c["lo"] for c in cs], [c["hi"] - c["diff"] for c in cs]],
                   fmt=mk[name], color=col[name], ms=4.5, lw=1.1, capsize=0, label=LABELS[name].split(":")[0])
        ov = contrast(tb)
        A.errorbar([3e8 * off], [ov["diff"]], yerr=[[ov["diff"] - ov["lo"]], [ov["hi"] - ov["diff"]]], fmt=mk[name],
                   color=col[name], ms=6, lw=1.6, capsize=0)
        Bp.plot(bs, [c["mse_policy"] for c in cs], marker=mk[name], color=col[name], lw=1.2, ms=4,
                label="full policy, " + LABELS[name].split(":")[0])
        if k == 0:
            Bp.plot(bs, [c["mse_sbsb"] for c in cs], marker="^", color="#7F7F7F", lw=1.2, ms=4, label="SBS-B")
        for b, c in zip(bs, cs):
            data.append(dict(panel="A/B", bundle=name, B=b, **c))
        data.append(dict(panel="A", bundle=name, B="all", **ov))
    if not rows:                                             # A and B need the confirmation's per-trial arrays
        for P in (A, Bp):
            P.axis("off")
            P.text(0.5, 0.5, "1.8.0 confirmation arrays not extracted", ha="center", va="center", fontsize=8)
    A.axhline(0, color="#7F7F7F", lw=0.8)
    A.set_xscale("log")
    A.axvline(1.0e8, color="#BBBBBB", lw=0.8, ls="--")
    A.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 3e8])
    A.set_xticklabels(["10³", "10⁴", "10⁵", "10⁶", "10⁷", "all"])
    A.set_xlabel("total shot budget B")
    A.set_ylabel("Δ log10(R + 10⁻⁶) vs SBS-B\n(base-circuit mean; < 0 better)")
    A.set_title("A  confirmation: full policy − SBS-B", loc="left", fontweight="bold")
    A.legend(frameon=False, fontsize=6.5, loc="lower left")
    Bp.set_xscale("log")
    Bp.set_yscale("log")
    Bp.set_xlabel("total shot budget B")
    Bp.set_ylabel("arithmetic MSE (base-circuit mean)")
    Bp.set_title("B  arithmetic MSE (not the primary metric)", loc="left", fontweight="bold")
    Bp.legend(frameon=False, fontsize=6.5)
    s = src.get("audit")
    if s:
        names = [p for p in s["policies"] if p.startswith("POLICY|")]
        for i, p in enumerate(names):
            v = s["policies"][p]
            nm = p.replace("POLICY|", "")
            for j, (tgt, lab) in enumerate((("fresh", "fresh-design label"), ("executed", "executed design"))):
                e = v.get(tgt, {})
                if "diff_vs_sbsb" not in e:
                    continue
                y = i + (j - 0.5) * 0.25
                C.plot([e["lo"], e["hi"]], [y, y], color=col.get(nm, "#555"), lw=1.4,
                       alpha=1.0 if tgt == "executed" else 0.5)
                C.plot(e["diff_vs_sbsb"], y, "o" if tgt == "executed" else "D", color=col.get(nm, "#555"), ms=5,
                       mfc=col.get(nm, "#555") if tgt == "executed" else "white")
                data.append(dict(panel="C", bundle=nm, target=tgt, diff=e["diff_vs_sbsb"], lo=e["lo"], hi=e["hi"]))
        C.set_yticks(range(len(names)))
        C.set_yticklabels([LABELS.get(p.replace("POLICY|", ""), p).split(":")[0] for p in names])
        C.axvline(0, color="#7F7F7F", lw=0.8)
        C.set_xlabel("Δ log10(R + 10⁻⁶) vs SBS-B (8 audit circuits)")
        C.set_title("C  executed vs fresh-design risk", loc="left", fontweight="bold")
        C.plot([], [], "o", color="#555", label="executed (returned design)")
        C.plot([], [], "D", color="#555", mfc="white", label="fresh-design label")
        C.legend(frameon=False, fontsize=6.5, loc="lower right")
        C.set_ylim(-0.7, len(names) - 0.3)
    pilot = src.get("pilot")
    pc = os.path.join(pilot, "portfolio", "portfolio_contrasts.csv") if pilot else ""
    if pc and os.path.exists(pc):
        import pandas as pd
        con = pd.read_csv(pc, encoding="utf-8")
        for i, r in enumerate(con.itertuples()):
            c = "#7F7F7F" if r.contrast.startswith("expansion") else "#009E73"
            D.plot([r.lo, r.hi], [i, i], color=c, lw=1.4)
            D.plot(r.diff, i, "o", color=c, ms=4.5)
            data.append(dict(panel="D", contrast=r.contrast, diff=r.diff, lo=r.lo, hi=r.hi))
        D.set_yticks(range(len(con)))
        D.set_yticklabels([x.replace("_", " ") for x in con.contrast], fontsize=6)
        D.invert_yaxis()
        D.axvline(0, color="#7F7F7F", lw=0.8)
        D.set_xlabel("Δ log10(R + 10⁻⁶) (pilot, out-of-fold)")
        D.set_title("D  pilot: candidates vs learning", loc="left", fontweight="bold")
    else:
        D.axis("off")
        D.text(0.5, 0.5, "pilot portfolio comparison not available", ha="center", va="center", fontsize=8)
    fig.suptitle("Simulated (emulated fake-device trials); no hardware data. A–B: 1.8.0 once-only confirmation (29 "
                 "circuits); C: exploratory execution audit; D: exploratory pilot.", fontsize=6.5, color="#555555")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(out, "fig_v19_policy_benefit.%s" % ext), dpi=200)
    plt.close(fig)
    import pandas as pd
    pd.DataFrame(data).to_csv(os.path.join(out, "fig_v19_policy_benefit.csv"), index=False, encoding="utf-8")
    return "fig_v19_policy_benefit"


UNICODE_TEX = {"∝": r"\ensuremath{\propto}", "≈": r"\ensuremath{\approx}", "±": r"\ensuremath{\pm}",
               "√": r"\ensuremath{\surd}", "μ": r"\ensuremath{\mu}", "σ": r"\ensuremath{\sigma}",
               "γ": r"\ensuremath{\gamma}", "η": r"\ensuremath{\eta}", "ρ": r"\ensuremath{\rho}",
               "χ": r"\ensuremath{\chi}", "ε": r"\ensuremath{\varepsilon}", "⊗": r"\ensuremath{\otimes}",
               "∑": r"\ensuremath{\sum}", "∞": r"\ensuremath{\infty}", "→": r"\ensuremath{\to}",
               "²": r"\ensuremath{^{2}}", "⁴": r"\ensuremath{^{4}}", "⁶": r"\ensuremath{^{6}}",
               "⁸": r"\ensuremath{^{8}}", "⁰": r"\ensuremath{^{0}}", "¹": r"\ensuremath{^{1}}",
               "⁻": r"\ensuremath{^{-}}", "≠": r"\ensuremath{\neq}", "…": r"\ldots{}"}


def unicode_preamble(texts) -> str:
    r"""\DeclareUnicodeCharacter lines for every non-ASCII character left in the generated tables (pdflatex knows
    few of them): a mapping where one is known, else the code point shown visibly — never a failed PDF."""
    chars = sorted({c for t in texts for c in t if ord(c) > 127})
    out = []
    for c in chars:
        if c in "äöüÄÖÜéèàçñ":                           # covered by inputenc/T1
            continue
        out.append(r"\DeclareUnicodeCharacter{%04X}{%s}" % (ord(c), UNICODE_TEX.get(c, "[U+%04X]" % ord(c))))
    return "\n".join(out)


WRAPPER = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=16mm]{geometry}
\usepackage{booktabs,amsmath,amssymb,textcomp,array,adjustbox}
\usepackage{caption}
%s
\captionsetup{font=small,labelfont=bf,skip=8pt}
\renewcommand{\arraystretch}{1.15}
\renewcommand{\floatpagefraction}{0.25}
\begin{document}
%s
\end{document}
"""


def build(src: dict, out: str, compile_pdf: bool = True) -> dict:
    import shutil
    import subprocess
    os.makedirs(out, exist_ok=True)
    root = src.get("root") or ""

    def _portable(v):                               # repository-relative where possible (shipped provenance)
        if isinstance(v, str) and root and os.path.isabs(v):
            try:
                r = os.path.relpath(v, root)
                return v if r.startswith("..") else r.replace(os.sep, "/")
            except ValueError:                      # another drive on Windows
                return v
        return v
    made, skipped, prov = [], [], dict(generator=GENERATOR, sources={k: _portable(v) for k, v in src.items()
                                                                     if isinstance(v, str) and k != "root"},
                                       tables={})
    md = ["# Numerical tables — QEM Explorer 1.9.0", "", "Generated by `%s`. Simulated observations (emulated "
          "fake-device trials); no hardware data." % GENERATOR, ""]
    for fn in TABLES:
        try:
            t = fn(src)
        except FileNotFoundError:
            t = None
        if t is None:
            skipped.append(fn.__name__)
            continue
        with open(os.path.join(out, t.key + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t.tex().replace(r"\begin{table}[p]", r"\begin{table}[htbp]")      # never wider than the text
                     .replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\textwidth}\begin{tabular}")
                     .replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}"))
        t.frame().to_csv(os.path.join(out, t.key + ".csv"), index=False, encoding="utf-8")
        md.append(t.markdown())
        prov["tables"][t.key] = dict(label=t.label, caption=t.caption, sources=t.source)
        made.append(t.key)
    fig = figure1(src, out)
    prov["figure"] = fig
    with open(os.path.join(out, "tables_v19.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md + (["## Not generated", "", ", ".join(skipped), ""] if skipped else [])))
    texts = [open(os.path.join(out, k + ".tex"), encoding="utf-8").read() for k in made]
    with open(os.path.join(out, "tables_v19.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(WRAPPER % (unicode_preamble(texts), "\n\\clearpage\n".join(r"\input{%s.tex}" % k for k in made)))
    pdf = "not compiled (pdflatex unavailable or disabled)"
    if compile_pdf and shutil.which("pdflatex") and made:
        try:
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tables_v19.tex"], cwd=out,
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            pdf = "tables_v19.pdf"
            if os.path.exists(os.path.join(out, "tables_v19.log")):
                os.remove(os.path.join(out, "tables_v19.log"))     # machine paths; kept only on failure
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            pdf = "failed: %s (see tables_v19.log)" % type(e).__name__
        for ext in (".aux", ".out"):
            p = os.path.join(out, "tables_v19" + ext)
            if os.path.exists(p):
                os.remove(p)
    hashes = {}
    for k, p in src.items():
        if isinstance(p, str) and os.path.isfile(p):
            hashes[k] = _sha(p)
    prov.update(made=made, skipped=skipped, pdf=pdf, source_sha256=hashes,
                baseline_check=src.get("baseline_check"), notes=[v for k, v in src.items() if k.endswith("_skipped")])
    TX.write_json(os.path.join(out, "provenance.json"), prov, indent=1, default=str)
    return prov


def sources(root: str, v18: str, audit: str | None, pilot: str | None) -> dict:
    src = dict(root=root, v18=v18, pilot=pilot)
    tdir = os.path.join(v18, "confirm", "tasks")
    if os.path.isdir(tdir) and any(f.endswith(".npz") for f in os.listdir(tdir)):
        src["confirmation"] = confirmation_rows(v18)        # stale, corrupted or inconsistent data: an error
        src["baseline_check"] = dict(BASELINE_CHECK)
    else:                                                   # the trial-data archives are not extracted
        src["confirmation"] = {}
        src["confirmation_skipped"] = ("the 1.8.0 confirmation's per-trial arrays are not in %s (extract the "
                                       "trial-data archives): s3_primary and Figure 1 A-B are not drawn" % tdir)
    if audit and os.path.exists(os.path.join(audit, "execution_audit_summary.json")):
        src["audit"] = TX.read_json(os.path.join(audit, "execution_audit_summary.json"))
        src["audit_path"] = os.path.join(audit, "execution_audit_summary.json")
    return src


def main(argv=None) -> int:
    from ..textio import utf8_stdio
    utf8_stdio()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v18", default=os.path.join(root, "reports", "selection_v180"))
    ap.add_argument("--audit", default=os.path.join(root, "reports", "execution_audit_v190"))
    ap.add_argument("--pilot", default=os.path.join(root, "reports", "pilot_v190"))
    ap.add_argument("--out", default=os.path.join(root, "reports", "tables_v190"))
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    from . import integrity as IN, replay as RP
    try:
        src = sources(root, a.v18, a.audit or None, a.pilot or None)
    except (IN.IncompleteRun, IN.CorruptedRun, RP.StaleTable, ValueError, OSError) as e:   # incl. UncheckedBaseline
        print("refused: %s" % e)
        return 2
    for k, v in src.items():
        if k.endswith("_skipped"):
            print("note: %s" % v)
    p = build(src, a.out, compile_pdf=not a.no_pdf)
    print(TX.dumps(dict(made=p["made"], skipped=p["skipped"], figure=p["figure"], pdf=p["pdf"],
                        baseline_check=p.get("baseline_check")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
