r"""The six main figures of the protocol-selection study (1.6.0), implemented to their panel specifications
(``docs/selection/FIGURES.md``) and drawn only from saved tables of a run directory — never from a live simulation.

Every figure is written as PDF and SVG plus a 300-dpi PNG (``figures._save``) with its caption, interpretation and
plotted data (``<name>__<key>.csv``). Plotting choices (policies per role, budget facets, slices, targets) come from
the versioned plotting configuration ``configs/selection/figures.yaml`` (``load_config``).

Conventions shared by all figures:

* unit of analysis: a test row (task × budget) of an independent base circuit; aggregation weights: each base circuit
  weighs the same, its variants averaged first;
* risk: evaluation-half MSE of the chosen protocol (normalized observables, ‖O‖∞ ≤ 1); regret: the same against the
  evaluation-half MSE of the protocol identified on the independent selection half (so it can be negative: that is
  estimation noise of the reference, kept, never clipped into theoretical regret);
* log10 values use log10(MSE + 10⁻⁶); intervals are 95 % percentile intervals of a bootstrap over base circuits;
* failures and fallbacks are part of every risk (nonresponse is scored as estimate 0); infeasible candidates are
  excluded from a row's candidate set and counted in the support maps.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import textio as TX

from .figures import (LABEL_SIM, PROTOCOL_COLORS, PROTOCOL_MARKERS, _plt, _save, _seq_cmap, _div_cmap)
from .models import EPS0

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULTS = dict(
    schema="qemx.selection.figures/1", version=2, weights="base_circuit",
    policies=dict(raw="RAW", fixed="SBS", budget_fixed="SBS-B", linear="MODEL[ridge|deployable+simulated]",
                  nonlinear="frozen", nonlinear_abstain="frozen+abstain"),
    regret_profile=dict(budgets=dict(low=1000, medium=100000, high=10000000), tolerance_min=1e-8,
                        tolerance_max=1e-1),
    families_order=["qft", "ghz", "ising", "brickwork", "grover", "bv", "hea", "qaoa"],
    regime=dict(slices=["ising", "hea", "qaoa"], min_trials=30),
    observable_panel=dict(family="ising", budget=100000, device="fake_fez", layout="best"),
    noise_panel=dict(family="ising", budget=100000, layout="best", observable="default"),
    noise_composition_panel=dict(budget=100000, families=["ising", "hea", "qaoa"]),
    generalization=dict(rows=["same-distribution", "family", "device", "width"],
                        columns=dict(reference="SBS-B", simple="SIZE-B", linear="MODEL[ridge|deployable]",
                                     structured="MODEL[struct|deployable]", nonlinear="MODEL[hgb|deployable]")),
    budget_generalization=dict(policies=["SBS-B", "MODEL[ridge|deployable]", "MODEL[struct|deployable]",
                                         "MODEL[hgb|deployable]"]),
    mismatch=dict(budgets=[1000, 100000, 10000000], facets=dict(
        coherent=dict(label="coherent fraction χ", points={"none": 0.0, "coherent-0.25": 0.25, "coherent-0.5": 0.5}),
        crosstalk=dict(label="ZZ crosstalk angle θ (rad)",
                       points={"none": 0.0, "crosstalk-0.02": 0.02, "crosstalk-0.05": 0.05}),
        drift=dict(label="error-rate drift factor s", points={"none": 1.0, "drift-1.5": 1.5, "drift-2": 2.0}))),
    mse_targets=[1e-4, 1e-5, 1e-6],
    bias_variance=dict(protocols=["RAW", "RO", "ZNE-L", "PEC", "CDR"], noise_scale=1.0),
    representative=dict(budget=100000),
)
GROUP_LABEL = {"budget": "budget", "protocol": "protocol overheads", "structure": "basic structure",
               "observable": "observable", "lightcone": "light cone", "noise": "device (snapshot)",
               "state": "simulator-only state"}


# question answered and saved input tables (with the columns read) of every figure; written into each figure's notes
FIG_META = {
    "fig01_budget_efficiency": (
        "Does recommendation improve budget efficiency over raw execution and fixed protocols?",
        {"analysis/choices_same-distribution.csv": "task_id, base_id, family, n, device, B, <policy>|choice",
         "analysis/labels.csv": "task_id, B, protocol, mse_eval, cal/fit/learn_fallback_eval, nonresponse_eval",
         "analysis/nested_rows.csv": "task_id, B, base_id, accepted, choice, fallback, risk_policy",
         "analysis/family_benefit.csv": "family, diff_log(_lo/_hi), diff_mse(_lo/_hi), base_circuits",
         "analysis/failure_profile.csv": "policy, B, mse, worst1_share, worst5_share, severe_rate",
         "freeze.json": "policy"}),
    "fig02_performance_regimes": (
        "Where should each protocol be used along budget and noise strength, and does the recommender follow it?",
        {"regimes/analysis/regime_cells.csv": "family, noise_scale, B, observable, layout, reference, unresolved, "
                                              "model_choice, abstained, policy_choice, log_regret_policy, "
                                              "n_feasible, trials_per_candidate"}),
    "fig03_feature_mechanisms": (
        "Which information changes the decision?",
        {"analysis/ablation.csv": "variant, diff_vs_all, diff_lo, diff_hi, tuned",
         "regimes/analysis/regime_protocols.csv": "family, device, observable, layout, B, protocol, mse_pooled, "
                                                  "se_pooled, chosen",
         "regimes/analysis/regime_features.csv": "family, device, observable, layout, obs_weight, lc_qubits, lam_lc"}),
    "fig04_generalization_learning": (
        "Does the model generalize beyond its training distribution, and how much data does it need?",
        {"analysis/pooled_<split>.csv": "policy, B (= all), mean_log_regret, log_regret_lo, log_regret_hi, "
                                        "base_circuits, rows",
         "analysis/learning_curve.csv": "train_base_circuits, mean_log_regret, sbs_mean_log_regret",
         "analysis/budget_generalization.csv": "max_train_B, policy, B, regime, mean_log_regret"}),
    "fig05_reliability_abstention": (
        "Can the system recognize unreliable recommendations (nested evaluation)?",
        {"analysis/nested_curve.csv": "coverage, accepted_log_regret, policy_mse, random_policy_mse",
         "analysis/nested_summary.csv": "policy, coverage, mse, mse_lo, mse_hi, diff_log_vs_sbsb(_lo/_hi), "
                                        "diff_log_vs_random(_lo/_hi), random_draws_mse_lo/hi, accepted_log_regret",
         "analysis/nested_rows.csv": "fold, accepted, log_regret_model", "analysis/nested_folds.csv": "fold, tau",
         "analysis/calibration.csv": "kind, nominal, coverage",
         "analysis/interval_coverage.csv": "kind, nominal, coverage, finite, statement"}),
    "fig06_mismatch_diagnostics": (
        "Does model mismatch predict recommendation failure, and does reacting to a diagnostic pay for its shots?",
        {"mismatch/diagnostics.csv": "scenario, type, task_id, B, d, threshold, alpha, flag_rate",
         "mismatch/policies.csv": "scenario, type, task_id, base_id, B, policy, risk, mean_log10_risk",
         "mismatch/analysis/deployment_draws.csv": "B, abs_z, model_log_regret",
         "mismatch/analysis/discrimination.csv": "B, auc_abs_z",
         "mismatch/analysis/reactions.csv": "held_out_type, B, reaction"}),
    "supp_target_precision": (
        "What budget does each protocol need to reach an MSE target, by the point estimate and with uncertainty?",
        {"analysis/labels.csv": "task_id, protocol, B, feasible, mse_eval, se_mse_eval"}),
    "supp_bias_variance": (
        "Which part of each protocol's error is a floor that more shots do not remove?",
        {"regimes/tasks/<task_id>.json, .npz": "spec, budgets, protocols, ideal, allocations, est, design",
         "regimes/analysis/regime_cells.csv": "task_id, noise_scale, observable, layout"}),
    "supp_prediction_quality": (
        "How accurate are the held-out risk predictions per candidate, given the labels' own uncertainty?",
        {"analysis/prediction.csv": "protocol, observed, predicted, observed_se"}),
    "supp_dataset_coverage": (
        "Which part of the feature space do the development and final-test sets cover?",
        {"dev/tasks/*.json, final/tasks/*.json": "spec (family, n), base_id, ideal, features"}),
    "supp_coverage_certificates": (
        "Do the label intervals and finite-sample certificates hold empirically?",
        {"analysis/interval_coverage.csv": "see DATA_DICTIONARY", "analysis/certificates.csv": "radius, holds"}),
    "supp_failures_costs": (
        "How often do protocols fail or fall back, and where do the shots go?",
        {"analysis/labels.csv": "protocol, B, feasible, *_fallback_eval, nonresponse_eval, shots_*"}),
    "supp_representative": (
        "What do successful, typical, failed and abstained recommendations look like trial by trial?",
        {"analysis/deployed_rows.csv": "task_id, B, log_regret, margin, choice, fallback",
         "analysis/selective_rows.csv": "task_id, B, accepted_crossfit",
         "dev/tasks/<task_id>.json, .npz": "spec, budgets, protocols, ideal, est (trial estimates)"}),
    "supp_runtime": (
        "What does one recommendation cost classically as the circuit grows?",
        {"analysis/runtime.csv": "kind, n, compile_s, deployable_s, simulated_s, inference_s, "
                                 "preflight_structural_s, preflight_design_s, recommend_s, recommend_status, label_s, "
                                 "label_tasks, peak_rss_mb, rss_kind (1.8.0 columns optional)"}),
    "supp_design_variability": (
        "How much do CDR's and NIL's errors depend on the training design drawn, and does another training "
        "allocation change that?",
        {"analysis/design_diagnostics.csv": "task_id, protocol, B, half, design, mean_loss",
         "analysis/design_variance.csv": "protocol, B, designs, median_se_ratio, "
                                         "median_design_share_of_label_variance",
         "design_study/designs.csv": "task, variant, B, status, design, mean_loss"}),
}


# The methods box of every figure (prompt §9): what was varied and held fixed; units, budget definition and
# estimator; independent counts; uncertainty construction and scope; simulated or hardware observations;
# confirmatory or exploratory status; infeasible, failed, unresolved and missing cases; log-floor and clipping
# conventions. {placeholders} are filled from the run's records (``run_counts``).
UNITS = ("MSE of the estimate of ⟨O⟩ (‖O‖∞ ≤ 1) over r trials of a label half; B = total shots of a deployment "
         "(calibration, training, diagnostics and target shots all charged)")
LOGFLOOR = ("log summaries use log10(MSE + 10⁻⁶) and regret log10((R_chosen + 10⁻⁶)/(R_ref + 10⁻⁶)); a difference in "
            "them is not an exact factor in raw MSE at small risks; estimates are never clipped in the risk (clipping "
            "only inside the bounded-loss certificates, so marked)")
SIM = "simulated (emulated fake-device trials); no hardware observations"
DEV = "{dev_tasks} development tasks on {dev_base} independent base circuits; {designs} fresh designs per half for " \
      "CDR and NIL; r = {r} trials per half"
FIG_METHODS = {
    "fig01_budget_efficiency": dict(
        varied="total budget B and the selection policy; held fixed: the development tasks and their labels",
        counts=DEV, uncertainty="95 % bootstrap over base circuits (each base circuit weighted equally; its variants "
        "averaged first); pointwise per budget", status="exploratory (development data, group 5-fold held-out "
        "predictions); the frozen policy is evaluated once on the confirmation set elsewhere",
        missing="infeasible protocols are not candidates; fallback trials stay in the risk; failed label tasks are "
        "listed in label_failures.csv ({dev_failed} here) and excluded"),
    "fig02_performance_regimes": dict(
        varied="budget and overall noise scale s along controlled slices (one circuit each); held fixed: circuit, "
               "input, layout and observable within a slice",
        counts="{reg_tasks} regime tasks on {reg_base} base circuits; r = {reg_r} trials per half",
        uncertainty="unresolved cells: runner-up within 1.96 approximate SE of the difference (pointwise, not "
        "simultaneous over cells or candidates; design-level SE for the learning candidates (CDR, NIL; FR in 1.8.0))",
        status="exploratory, descriptive empirical map — not a phase diagram and not a simultaneous guarantee",
        missing="hatched/white: unresolved or fewer trials than the plotting minimum; infeasible candidates shown "
        "in the support panel; signed regret keeps negative empirical values"),
    "fig03_feature_mechanisms": dict(
        varied="(A) the information available to the model; (B) the observable; (C) the overall noise scale; "
               "(D) one noise component at a time — everything else held fixed within a panel",
        counts=DEV + "; panels B–D: single tasks per point ({nc_tasks} noise-composition tasks on {nc_base} base "
        "circuits)", uncertainty="(A) 95 % bootstrap over base circuits of the paired difference; (B, C) ±1.96 SE "
        "over trials (over designs for the learning candidates (CDR, NIL; FR in 1.8.0)); (D) pointwise resolution of the top two",
        status="exploratory; (B–D) controlled single-task comparisons of hypotheses declared before labeling",
        missing="infeasible candidates omitted per cell; unresolved top-two hatched in (D)"),
    "fig04_generalization_learning": dict(
        varied="the split (family, device, width, budget range held out) and the training-set size; held fixed: "
               "features and model settings per column", counts=DEV,
        uncertainty="95 % bootstrap over held-out base circuits; learning-curve band 10–90 % of fold × repeat "
        "draws (not a confidence interval)", status="exploratory (development data; held-out splits)",
        missing="cells without a held-out circuit are blank; extrapolated budgets marked"),
    "fig05_reliability_abstention": dict(
        varied="the abstention threshold (coverage); held fixed: the frozen model and its fallback SBS-B",
        counts=DEV, uncertainty="nested group cross-validation (outer 5 folds, threshold chosen on inner folds); "
        "95 % bootstrap over base circuits; random rejection at matched coverage over repeated draws",
        status="exploratory; measured coverage is not a guarantee; interval panels labelled by statement kind",
        missing="abstained rows use the fallback and stay in the whole-policy MSE"),
    "fig06_mismatch_diagnostics": dict(
        varied="the injected discrepancy (type and strength) between the snapshot and the execution law",
        counts="{mm_tasks} mismatch tasks on {mm_base} base circuits; r = {mm_r} trials per half",
        uncertainty="per-deployment diagnostic draws (one observation per deployment) vs repetition summaries, "
        "kept apart; equal-budget comparisons charge the diagnostic shots", status="exploratory; reactions trained "
        "and tested on different discrepancy types (leave one type out)",
        missing="infeasible diagnostics at small B are reported as such"),
    "supp_target_precision": dict(varied="the MSE target and budget", counts=DEV,
        uncertainty="pointwise approximate upper limit MSE + 1.96 SE (not a minimum-budget guarantee after a grid "
        "search)", status="descriptive (development data)", missing="unattained targets censored at 3·10⁷"),
    "supp_bias_variance": dict(varied="budget per protocol at s = 1 on the three regime slices",
        counts="{reg_tasks} regime tasks; r = {reg_r} trials per half; {designs} designs per half",
        uncertainty="95 % interval of the mean error with the SE matching the estimand (design-level for CDR, NIL)",
        status="descriptive", missing="bias² estimates ≤ 0 drawn as open markers at 10⁻¹⁰"),
    "supp_prediction_quality": dict(varied="held-out rows", counts=DEV,
        uncertainty="±1.96 label SE (delta method); best–runner-up differences with independent-label SE",
        status="descriptive (held-out predictions)", missing="labels beyond the axis counted in titles"),
    "supp_dataset_coverage": dict(varied="—", counts=DEV + "; final test {final_tasks} tasks on {final_base} base "
        "circuits; confirmation {conf_tasks} tasks on {conf_base} base circuits", uncertainty="none (distributions)", status="descriptive",
        missing="failed label tasks counted separately"),
    "supp_coverage_certificates": dict(varied="held-out folds and budgets", counts=DEV,
        uncertainty="per-fold coverage; bounds at α = 0.05", status="split-conformal / cv-residual / "
        "finite-sample-bound (clipped) / post-hoc-check, each labelled", missing="vacuous bounds reported as such"),
    "supp_failures_costs": dict(varied="protocol and budget", counts=DEV, uncertainty="none (rates)",
        status="descriptive", missing="infeasible, method error, fallback (union of flags), no estimate and "
        "software failures ({dev_failed} label tasks) reported separately"),
    "supp_representative": dict(varied="four rows chosen by documented rules", counts=DEV,
        uncertainty="none (individual trials)", status="illustrative", missing="—"),
    "supp_runtime": dict(varied="circuit width", counts="one circuit family; repeats in one process",
        uncertainty="median of repeats", status="measurement on one machine", missing="label time missing where "
        "the development set has no task of that width"),
    "supp_design_variability": dict(varied="the training design (fresh draws) and, in the design study, the "
        "training allocation", counts=DEV + "; design study: {ds_designs} designs per variant",
        uncertainty="designs as the independent unit within a task", status="exploratory",
        missing="infeasible variants listed in design_study/designs.csv"),
}


# 1.8.0 supplementary figures (qemx.selection.plots18): their questions, inputs and methods boxes
from .plots18 import FIG_META_18 as _FM18, FIG_METHODS_18 as _FX18  # noqa: E402
FIG_META.update(_FM18)
FIG_METHODS.update(_FX18)


def run_counts(run_dir: str) -> dict:
    """Counts for the methods boxes from the run's label records (tasks, base circuits, failed tasks, designs per
    half, trials per half) of every labeled part."""
    from . import labels as LB
    out = dict(designs="n/a", r="n/a", ds_designs="n/a")
    rs, ks = {}, set()
    for part, key in (("dev", "dev"), ("final", "final"), ("regimes", "reg"), ("noisecomp", "nc"),
                      ("mismatch", "mm"), ("confirm", "conf")):
        d = os.path.join(run_dir, part)
        tasks, base, failed = 0, set(), 0
        if os.path.isdir(os.path.join(d, "tasks")):
            for tid in LB.task_ids(d):
                try:
                    rec = LB.load_record(d, tid)
                except Exception:  # noqa: BLE001 — a corrupted record is counted as failed
                    failed += 1
                    continue
                if rec.get("status") != "ok":
                    failed += 1
                    continue
                tasks += 1
                base.add(rec.get("base_id"))
                rs.setdefault(key, set()).add(int(rec.get("R", 0)))
                plan = LB.plan_of(rec)
                if plan.get("mode") == "fresh":
                    ks.add(int(plan.get("per_half") or 0))
        out.update({key + "_tasks": tasks, key + "_base": len(base), key + "_failed": failed})
    for key, v in rs.items():                       # trials per half of each part (the mismatch study has its own)
        out[key + "_r"] = "/".join(map(str, sorted(v)))
    out["r"] = out.get("dev_r", "n/a")
    out["designs"] = "/".join(map(str, sorted(ks))) or "fixed"
    p = os.path.join(run_dir, "design_study", "designs.csv")
    if os.path.exists(p):
        import pandas as pd
        d = pd.read_csv(p, encoding="utf-8")
        if "design" in d.columns and len(d):
            out["ds_designs"] = int(d.dropna(subset=["design"]).groupby(["task", "variant", "B"]).size().median())
    return out


def methods_box(name: str, counts: dict) -> str:
    m = FIG_METHODS.get(name)
    if not m:
        return ""

    class _D(dict):
        def __missing__(self, k):
            return "n/a"
    f = lambda s: s.format_map(_D(counts))  # noqa: E731
    rows = [("Varied / held fixed", m["varied"]), ("Units, budget, estimator", UNITS),
            ("Independent counts", f(m["counts"])), ("Uncertainty and its scope", m["uncertainty"]),
            ("Observations", SIM), ("Status", m["status"]), ("Infeasible, failed, unresolved, missing",
                                                             f(m["missing"])),
            ("Log floor and clipping", LOGFLOOR)]
    return "\n**Methods box.**\n\n| | |\n|---|---|\n" + "".join("| %s | %s |\n" % (k, v.replace("|", "\\|"))
                                                              for k, v in rows)


def annotate(out_dir: str, name: str, command: str, cfg_path: str | None = None, counts: dict | None = None) -> None:
    """Prepend the question and append the methods box (with ``counts`` from ``run_counts``), the inputs, the command
    and the plotting configuration's version and hash to a figure's notes (``<name>.md``)."""
    import hashlib
    path = os.path.join(out_dir, name + ".md")
    if name not in FIG_META or not os.path.exists(path):
        return
    q, inputs = FIG_META[name]
    body = open(path, encoding="utf-8").read()
    cfg_path = cfg_path or os.path.join(ROOT, "configs", "selection", "figures.yaml")
    h = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:16] if os.path.exists(cfg_path) else "defaults"
    ver = load_config(cfg_path if os.path.exists(cfg_path) else None).get("version")
    head, rest = body.split("\n", 1)
    out = "%s\n\n**Question.** %s\n%s" % (head, q, rest)
    out += methods_box(name, counts or {})
    out += "\n**Inputs.**\n\n" + "".join("- `%s`: %s\n" % (k, v) for k, v in inputs.items())
    out += ("\n**Command.** `%s`\n\n**Plotting configuration.** `%s` (version %s, sha256 %s…)\n"
            % (command, os.path.relpath(cfg_path, ROOT) if cfg_path.startswith(ROOT) else cfg_path, ver, h))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)


PLOT_DESCRIPTIVE_KEYS = ("release", "description")      # recorded with a plotting configuration, not read


def load_config(path: str | None = None) -> dict:
    """The plotting configuration (``configs/selection/figures.yaml`` by default), merged over the defaults.

    1.9.0 (second review N6): an explicitly given file must exist (1.8.0 silently fell back to the defaults), is read
    as UTF-8 with an optional byte-order mark, and unknown keys are refused (``release`` and ``description`` are
    descriptive)."""
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    explicit = path is not None
    path = path or os.path.join(ROOT, "configs", "selection", "figures.yaml")
    if explicit and not os.path.exists(path):
        raise FileNotFoundError("plotting configuration %s does not exist" % path)
    if os.path.exists(path):
        user = TX.read_yaml(path) or {}
        if user.get("schema", cfg["schema"]) != cfg["schema"]:
            raise ValueError("%s: schema must be %s" % (path, cfg["schema"]))
        unknown = sorted(set(user) - set(DEFAULTS) - set(PLOT_DESCRIPTIVE_KEYS))
        if unknown:
            raise ValueError("%s: unknown plotting keys %s (known: %s)" % (path, unknown, sorted(DEFAULTS)))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def frozen_policy(run_dir: str) -> str | None:
    p = os.path.join(run_dir, "freeze.json")
    return TX.read_json(p)["policy"] if os.path.exists(p) else None


def _resolve(name: str, run_dir: str) -> str:
    fz = frozen_policy(run_dir)
    if name in ("frozen", "frozen+abstain"):
        if fz is None:
            raise ValueError("the run has no freeze.json; name the policy explicitly in the plotting configuration")
        return fz if name == "frozen" else fz + "+abstain"
    return name


def _blabel(b) -> str:
    e = int(round(math.log10(b)))
    m = b / 10 ** e
    return ("10$^{%d}$" % e) if abs(m - 1) < 1e-9 else ("%g·10$^{%d}$" % (m, e))


def _btext(b) -> str:
    """Plain-text budget label (3·10⁷), for captions and notes."""
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    e = int(math.floor(math.log10(b) + 1e-9))
    m = b / 10 ** e
    return ("10" if abs(m - 1) < 1e-9 else "%g·10" % m) + str(e).translate(sup)


def _wmean_ci(values, groups, reps=1000, seed=0):
    """Base-circuit-weighted mean and 95 % bootstrap interval."""
    from .evaluate import bootstrap
    ci = bootstrap({"v": np.asarray(values, float)}, np.asarray(groups), reps, seed)[0]["v"]
    return ci


# ------------------------------------------------------------------ row tables (choices × labels)
def policy_rows(run_dir: str, split: str = "same-distribution") -> "pd.DataFrame":
    """Long table of held-out rows × policies: risk, log10 risk, regret, log10 regret ratio, failure share and the
    chosen protocol, from ``choices_<split>.csv`` and ``labels.csv`` (plus the abstaining variant of the frozen
    model from ``selective_rows.csv`` when present)."""
    import pandas as pd
    from . import evaluate as EV
    adir = os.path.join(run_dir, "analysis")
    ch = pd.read_csv(os.path.join(adir, "choices_%s.csv" % split), encoding="utf-8")
    lab = pd.read_csv(os.path.join(adir, "labels.csv"), encoding="utf-8")
    per = EV._per_policy_from_choices(ch, lab)
    out = []
    for p, v in per.items():
        out.append(pd.DataFrame(dict(task_id=ch.task_id, base_id=ch.base_id, family=ch.family, n=ch.n,
                                     device=ch.device, B=ch.B, policy=p, choice=ch[p + "|choice"], risk=v["risk"],
                                     log_risk=v["log_risk"], regret=v.get("regret", np.nan),
                                     log_regret=v["log_ratio"], fail=v.get("fail", np.nan),
                                     accepted=v["accepted"])))
    rows = pd.concat(out, ignore_index=True)
    fz = frozen_policy(run_dir)
    sp = os.path.join(adir, "selective_rows.csv")
    dp = os.path.join(adir, "deployed_rows.csv")
    npth = os.path.join(adir, "nested_rows.csv")
    if fz and split == "same-distribution" and os.path.exists(npth):        # 1.7.0: the nested evaluation
        nr = pd.read_csv(npth, encoding="utf-8")
        ref = rows[rows.policy == "ALL-CANDIDATE"].set_index(["task_id", "B"]).risk
        rr = ref.reindex(list(zip(nr.task_id, nr.B))).to_numpy()
        r = nr.risk_policy.to_numpy(float)
        ab = pd.DataFrame(dict(task_id=nr.task_id, base_id=nr.base_id, family=nr.family, n=nr.n, device=nr.device,
                               B=nr.B, policy=fz + "+abstain", choice=np.where(nr.accepted, nr.choice, nr.fallback),
                               risk=r, log_risk=np.log10(r + EPS0), regret=r - rr,
                               log_regret=np.log10((r + EPS0) / (rr + EPS0)), fail=np.nan,
                               accepted=nr.accepted.astype(bool)))
        rows = pd.concat([rows, ab], ignore_index=True)
    elif fz and split == "same-distribution" and os.path.exists(sp) and os.path.exists(dp):
        sel = pd.read_csv(sp, encoding="utf-8")
        dep = pd.read_csv(dp, encoding="utf-8").merge(sel[["task_id", "B", "accepted_crossfit"]], on=["task_id", "B"])
        acc = dep.accepted_crossfit.astype(bool)
        ref = rows[rows.policy == "ALL-CANDIDATE"].set_index(["task_id", "B"]).risk
        r = np.where(acc, dep.risk, dep.risk_fallback)
        rr = ref.reindex(list(zip(dep.task_id, dep.B))).to_numpy()
        ab = pd.DataFrame(dict(task_id=dep.task_id, base_id=dep.base_id, family=dep.family, n=dep.n,
                               device=dep.device, B=dep.B, policy=fz + "+abstain",
                               choice=np.where(acc, dep.choice, dep.fallback), risk=r, log_risk=np.log10(r + EPS0),
                               regret=r - rr, log_regret=np.log10((r + EPS0) / (rr + EPS0)), fail=np.nan,
                               accepted=acc))
        rows = pd.concat([rows, ab], ignore_index=True)
    return rows


# ------------------------------------------------------------------ Figure 1
def fig01_budget_efficiency(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Does recommendation improve budget efficiency? (A) budget–risk curves, (B) regret performance profiles at
    three preselected budgets, (C) paired benefit of the proposed policy over SBS-B by circuit family."""
    import pandas as pd
    cfg = cfg or load_config()
    plt = _plt()
    rows = policy_rows(run_dir)
    roles = cfg["policies"]
    names = {}
    for role in ("raw", "fixed", "budget_fixed", "linear", "nonlinear", "nonlinear_abstain"):
        try:
            nm = _resolve(roles[role], run_dir)
        except ValueError:
            continue
        if nm in set(rows.policy):
            names[role] = nm
    def _cfg_short(nm):                               # MODEL[hgb|deployable+simulated] → hgb | deployable+simulated
        return nm[nm.find("[") + 1:-1].replace("|", " | ") if "[" in nm else nm
    # short legend entries (1.7.0: the full configuration names made the legend wider than panel A and collapsed the
    # layout); the configurations are named in full in the notes
    label = dict(raw="RAW", fixed="SBS (best fixed protocol)", budget_fixed="SBS-B (budget-dependent fixed)",
                 linear="linear: %s" % _cfg_short(names.get("linear", "")),
                 nonlinear="frozen: %s" % _cfg_short(names.get("nonlinear", "")),
                 nonlinear_abstain="frozen + selective rule (nested CV)")
    style = dict(raw=("#C8C8C8", "o", ":"), fixed=("#E69F00", "s", "--"), budget_fixed=("#D55E00", "D", "--"),
                 linear=("#56B4E9", "^", "-"), nonlinear=("#0072B2", "o", "-"),
                 nonlinear_abstain=("#009E73", "v", "-."))
    budgets = sorted(rows.B.unique())
    curve = []
    for role, nm in names.items():
        d = rows[rows.policy == nm]
        for b in budgets:
            e = d[d.B == b]
            m, lo, hi = _wmean_ci(e.risk, e.base_id)
            gm, glo, ghi = _wmean_ci(e.log_risk, e.base_id)
            curve.append(dict(role=role, policy=nm, B=int(b), mean_mse=m, ci_lo=lo, ci_hi=hi, mean_log10_mse=gm,
                              log_ci_lo=glo, log_ci_hi=ghi, base_circuits=int(e.base_id.nunique()), rows=len(e)))
    curve = pd.DataFrame(curve)
    fig = plt.figure(figsize=(7.4, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    axA = fig.add_subplot(gs[0, :2])
    axC = fig.add_subplot(gs[0, 2])
    axC2 = fig.add_subplot(gs[0, 3], sharey=axC)
    axB = [fig.add_subplot(gs[1, k]) for k in range(3)]
    axD = fig.add_subplot(gs[1, 3])
    for role in names:
        d = curve[curve.role == role].sort_values("B")
        c, mk, ls = style[role]
        axA.plot(d.B, d.mean_mse, ls=ls, marker=mk, ms=3.5, lw=1.4, color=c, label=label[role])
        if role in ("budget_fixed", "nonlinear"):           # bands of the key comparison only (all in the data)
            axA.fill_between(d.B, d.ci_lo, d.ci_hi, color=c, alpha=0.12, lw=0)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("total protocol shot budget B (all costs charged)")
    axA.set_ylabel("mean MSE (normalized observable)")
    axA.set_title("A  budget–risk curves (matched held-out tasks)", loc="left")
    axA.legend(frameon=False, fontsize=6.5, loc="lower left")
    # B: regret performance profiles
    prof = []
    bud = cfg["regret_profile"]["budgets"]
    tols = np.logspace(math.log10(cfg["regret_profile"]["tolerance_min"]),
                       math.log10(cfg["regret_profile"]["tolerance_max"]), 61)
    for k, (cond, b) in enumerate(bud.items()):
        a_ = axB[k]
        if not (rows.B == b).any():
            a_.text(0.5, 0.5, "budget not in this run's grid\n(no rows; nothing merged)", ha="center",
                    va="center", fontsize=6.5, color="#555555", transform=a_.transAxes)
        for role, nm in names.items():
            d = rows[(rows.policy == nm) & (rows.B == b)]
            if d.empty:
                continue
            w = 1.0 / d.groupby("base_id").task_id.transform("count") / d.base_id.nunique()
            reg = d.regret.to_numpy()
            frac = np.array([float(np.sum(w[reg <= t])) for t in tols])
            c, mk, ls = style[role]
            a_.plot(tols, frac, ls=ls, color=c, lw=1.3)
            neg = float(np.sum(w[reg < 0]))
            for t, f in zip(tols, frac):
                prof.append(dict(condition=cond, B=int(b), role=role, policy=nm, tolerance=float(t),
                                 fraction_within=float(f), fraction_negative_regret=neg,
                                 base_circuits=int(d.base_id.nunique())))
        a_.set_xscale("log")
        a_.set_ylim(0, 1.02)
        a_.set_xlabel("absolute regret tolerance (MSE units)" if k == 1 else "")    # one label for the row
        a_.set_title("B  %s budget, B = %s" % (cond, _blabel(b)), loc="left", fontsize=7)
        if k == 0:
            a_.set_ylabel("weighted fraction of tasks\nwithin tolerance")
    # C: paired benefit by family
    fb_path = os.path.join(run_dir, "analysis", "family_benefit.csv")
    fb = pd.read_csv(fb_path, encoding="utf-8") if os.path.exists(fb_path) else pd.DataFrame()
    if not fb.empty:
        order = [f for f in cfg["families_order"] if f in set(fb.family)] + \
                [f for f in fb.family if f not in cfg["families_order"]]
        fb = fb.set_index("family").loc[order].reset_index()
        y = np.arange(len(fb))[::-1]
        many = (fb.base_circuits >= 3).to_numpy()
        axC.errorbar(fb.diff_log[many], y[many], xerr=[(fb.diff_log - fb.diff_log_lo)[many],
                                                       (fb.diff_log_hi - fb.diff_log)[many]],
                     fmt="o", color="#0072B2", ms=3.5, capsize=2, lw=1)
        axC.plot(fb.diff_log[~many], y[~many], "o", mfc="white", mec="#0072B2", ms=3.5, ls="")
        for yy, xx in zip(y[~many], fb.diff_log[~many]):
            axC.annotate("< 3 circuits: no interval", (xx, yy), xytext=(0, -9), textcoords="offset points",
                         fontsize=6, color="#555555", ha="left", va="top",
                         bbox=dict(fc="white", ec="none", pad=0.3))
        axC.axvline(0, color="#333333", lw=0.8)
        axC.set_yticks(y, ["%s (%d)" % (f, n) for f, n in zip(fb.family, fb.base_circuits)])
        axC.set_xlabel("Δ mean log10(MSE + 10⁻⁶)\nproposed − SBS-B (< 0 favors it)")
        axC.set_title("C  paired effect by family", loc="left")
        axC2.errorbar(fb.diff_mse[many], y[many], xerr=[(fb.diff_mse - fb.diff_mse_lo)[many],
                                                        (fb.diff_mse_hi - fb.diff_mse)[many]],
                      fmt="s", color="#D55E00", ms=3.5, capsize=2, lw=1)
        axC2.plot(fb.diff_mse[~many], y[~many], "s", mfc="white", mec="#D55E00", ms=3.5, ls="")
        axC2.axvline(0, color="#333333", lw=0.8)
        lim = float(np.nanmax(np.abs(np.r_[fb.diff_mse_lo, fb.diff_mse_hi]))) if len(fb) else 1.0
        axC2.set_xscale("symlog", linthresh=max(lim * 1e-3, 1e-8))
        e_ = int(math.ceil(math.log10(max(lim, 1e-12))))       # three labelled ticks: the symlog default overlaps
        axC2.set_xticks([-10.0 ** e_, 0.0, 10.0 ** e_], ["−10$^{%d}$" % e_, "0", "10$^{%d}$" % e_])
        axC2.set_xlim(-10.0 ** e_, 10.0 ** e_)
        axC2.xaxis.set_minor_locator(plt.NullLocator())
        axC2.set_xlabel("Δ mean MSE (absolute)\nproposed − SBS-B")
        axC2.tick_params(labelleft=False)
        axC2.set_title("absolute scale", loc="left", fontsize=7)
    # D: catastrophic-error contributions (failure_profile.csv)
    fp_path = os.path.join(run_dir, "analysis", "failure_profile.csv")
    fpro = pd.read_csv(fp_path, encoding="utf-8") if os.path.exists(fp_path) else pd.DataFrame()
    Dd = []
    if not fpro.empty:
        e = fpro[fpro.B.astype(str) == "all"].set_index("policy")
        roles_d = [r_ for r_ in ("raw", "fixed", "budget_fixed", "linear", "nonlinear") if r_ in names
                   and names[r_] in e.index]
        yy = np.arange(len(roles_d))[::-1]
        for yv, role in zip(yy, roles_d):
            rr_ = e.loc[names[role]]
            axD.barh(yv, rr_.worst5_share, color=style[role][0], height=0.6)
            axD.text(min(rr_.worst5_share + 0.02, 0.98), yv, "%.0f %%; severe %.1f %%" % (
                100 * rr_.worst5_share, 100 * rr_.severe_rate), va="center", fontsize=6.5)
            Dd.append(dict(role=role, policy=names[role], worst5_share=float(rr_.worst5_share),
                           worst1_share=float(rr_.worst1_share), severe_rate=float(rr_.severe_rate),
                           mse=float(rr_.mse)))
        axD.set_yticks(yy, [dict(raw="RAW", fixed="SBS", budget_fixed="SBS-B", linear="linear",
                                 nonlinear="frozen")[r_] for r_ in roles_d])
        axD.set_xlim(0, 1.25)
        axD.set_xlabel("share of total squared error\nfrom the worst 5 % of rows")
        axD.set_title("D  catastrophic errors", loc="left")
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    fz = frozen_policy(run_dir)
    n_bc = int(rows.base_id.nunique())
    neg_txt = ", ".join("%s %.0f %%" % (c, 100 * float(pd.DataFrame(prof).query("condition == @c and role == "
                                                                              "'nonlinear'").fraction_negative_regret
                                                         .max() if prof else 0)) for c in bud)
    cap = ("Budget efficiency on held-out base circuits (development set, group 5-fold over %d base circuits, "
           "out-of-fold; every base circuit weighs the same). (A) Mean evaluation-half MSE of each policy's chosen "
           "protocol against the total shot budget (all calibration, training and execution shots charged), with 95 "
           "%% bootstrap bands over base circuits for SBS-B and the nonlinear recommender (the other policies' "
           "intervals are in the source data); the nonlinear recommender is the configuration frozen for the "
           "final test (%s); 'frozen + selective rule' accepts its choice when the confidence margin reaches τ and otherwise "
           "uses SBS-B, evaluated by nested group cross-validation (τ chosen on the inner folds only). "
           "(B) Regret performance profiles at three budgets fixed in the plotting configuration: the weighted "
           "fraction of tasks whose regret (policy MSE minus the MSE of the protocol identified on the independent "
           "selection half) is at most the tolerance; budgets are not merged. The reference is estimated, so some "
           "regrets are negative (share for the selected model: %s); they count as within every tolerance and are "
           "not clipped. (C) Paired difference of log10(MSE + 10⁻⁶) between the proposed policy and SBS-B by circuit family, "
           "all budgets, 95 %% bootstrap intervals over the family's base circuits (count in parentheses; a family "
           "with fewer than three base circuits is drawn open, without an interval); negative values favor the "
           "proposed policy; beside it the same paired difference in absolute mean MSE (symmetric-log axis), "
           "which the few large-MSE tasks dominate. (D) Share of the total squared error contributed by the worst 5 %% "
           "of held-out rows, all budgets, and the rate of severe misselection (risk more than 10× the reference's). "
           "%s" % (n_bc, fz, neg_txt, LABEL_SIM))
    parts = []
    if "nonlinear" in names and "budget_fixed" in names:
        a_ = rows[rows.policy == names["nonlinear"]].set_index(["task_id", "B"])
        b_ = rows[rows.policy == names["budget_fixed"]].set_index(["task_id", "B"]).reindex(a_.index)
        dl = (a_.log_risk - b_.log_risk).to_numpy()
        dm = (a_.risk - b_.risk).to_numpy()
        m, lo, hi = _wmean_ci(dl, a_.base_id.to_numpy())
        mm, mlo, mhi = _wmean_ci(dm, a_.base_id.to_numpy())
        parts.append("over all budgets the frozen recommender's paired difference to SBS-B is %+.3f [%+.3f, %+.3f] "
                     "in mean log10(MSE + 10⁻⁶) and %+.2g [%+.2g, %+.2g] in mean MSE (negative favors the recommender)"
                     % (m, lo, hi, mm, mlo, mhi))
        if m < 0 < mm or mm < 0 < m:
            both = lo < 0 < hi and mlo < 0 < mhi
            parts.append("the point estimates on the two scales differ in sign%s: the typical (log) task and the "
                         "large-MSE tasks that dominate the arithmetic mean need not favor the same policy"
                         % (" (neither interval excludes 0)" if both else ""))
    if not fb.empty:
        ok3 = fb[fb.base_circuits >= 3]
        neg = [f for f, h in zip(ok3.family, ok3.diff_log_hi) if h < 0]
        pos = [f for f, l_ in zip(ok3.family, ok3.diff_log_lo) if l_ > 0]
        parts.append("families (≥ 3 base circuits) whose interval excludes 0 in favor of the recommender: %s; "
                     "against it: %s"
                     % (", ".join(neg) or "none", ", ".join(pos) or "none"))
    interp = ("Observation: " + "; ".join(parts) + ". " if parts else "") + (
        "Differences inside the bootstrap intervals are not evidence of a difference. Hypothesis, not tested here: "
        "per-family differences reflect which families have ranking reversals inside the budget grid.")
    return _save(fig, out_dir, "fig01_budget_efficiency",
                 dict(curves=curve, profiles=pd.DataFrame(prof), families=fb, catastrophic=pd.DataFrame(Dd)), cap,
                 interp, dict(policies=names, regret_profile=bud, weights=cfg["weights"]))


# ------------------------------------------------------------------ Figure 2
def fig02_performance_regimes(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Where should each protocol be used? Empirical performance regime maps of the controlled regime study: per
    preregistered slice (circuit, input, observable, layout fixed) a triptych over total budget × device error
    scale s — reference preference, recommendation (abstention marked), regret of the deployed policy — plus a
    companion support map."""
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    cfg = cfg or load_config()
    plt = _plt()
    path = os.path.join(run_dir, "regimes", "analysis", "regime_cells.csv")
    cells = pd.read_csv(path, encoding="utf-8")
    cells = cells[(cells.observable == "default") & (cells.layout == "best")]
    slices = [s for s in cfg["regime"]["slices"] if s in set(cells.family)]
    budgets = sorted(cells.B.unique())
    scales = sorted(cells.noise_scale.unique())
    protos = list(PROTOCOL_COLORS)
    cmap = ListedColormap([PROTOCOL_COLORS[p] for p in protos])
    # 1.8.0: the number of candidates of the study (8 in 1.6/1.7, 17 in 1.8), not of the palette — a cell is marked
    # "a candidate infeasible" against the candidates the study labeled
    lab_p = os.path.join(run_dir, "regimes", "analysis", "labels.csv")
    n_cand = (int(pd.read_csv(lab_p, encoding="utf-8", usecols=["protocol"]).protocol.nunique())
              if os.path.exists(lab_p) else int(cells.n_feasible.max()))
    used = [p for p in protos if p in set(cells.reference) | set(cells.policy_choice)]
    reg = cells.log_regret_policy.to_numpy(float)
    # signed empirical regret on a zero-centred diverging scale (1.7.0; review of Figure 2): negative values (the
    # reference estimated on the other half was worse) are drawn as they are, never clipped to 0
    vmax = float(np.nanquantile(np.abs(reg), 0.98)) if np.isfinite(reg).any() else 1.0
    vmax = max(vmax, 0.05)
    fig, axes = plt.subplots(len(slices), 4, figsize=(7.4, 1.95 * len(slices) + 0.9), layout="constrained",
                             squeeze=False, gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]))
    out = []
    xt = np.arange(len(budgets))
    for r, fam in enumerate(slices):
        d = cells[cells.family == fam]
        G_ref = np.full((len(scales), len(budgets)), np.nan)
        G_rec = np.full_like(G_ref, np.nan)
        G_reg = np.full_like(G_ref, np.nan)
        G_sup = np.full_like(G_ref, np.nan)
        unres = np.zeros_like(G_ref, bool)
        abst = np.zeros_like(G_ref, bool)
        insuff = np.ones_like(G_ref, bool)
        # 1.8.0: "a candidate infeasible" against the candidates applicable to this slice (feasible in some cell of
        # it), so a candidate the circuit never admits (e.g. SV without a declared symmetry) does not mark every cell
        n_slice = int(d.n_feasible.max()) if len(d) else n_cand
        for _, c in d.iterrows():
            i, j = scales.index(c.noise_scale), budgets.index(c.B)
            ok = c.trials_per_candidate >= 2 * cfg["regime"]["min_trials"]
            insuff[i, j] = not ok
            G_sup[i, j] = c.n_feasible
            if not ok:
                continue
            G_ref[i, j] = protos.index(c.reference)
            unres[i, j] = bool(c.unresolved)
            G_rec[i, j] = protos.index(c.policy_choice)          # the deployed recommendation (SBS-B's if abstained)
            abst[i, j] = bool(c.abstained)
            G_reg[i, j] = float(c.log_regret_policy)
            out.append(dict(slice=fam, noise_scale=c.noise_scale, B=int(c.B), reference=c.reference,
                            unresolved=bool(c.unresolved), model_choice=c.model_choice, abstained=bool(c.abstained),
                            policy_choice=c.policy_choice, log_regret_policy=float(c.log_regret_policy),
                            plotted_regret=G_reg[i, j], n_feasible=int(c.n_feasible),
                            trials_per_candidate=int(c.trials_per_candidate)))
        ext = [-0.5, len(budgets) - 0.5, -0.5, len(scales) - 0.5]
        for k, (G, title) in enumerate(((G_ref, "reference preference"), (G_rec, "recommendation"))):
            a_ = axes[r, k]
            a_.imshow(np.ma.masked_invalid(G), cmap=cmap, vmin=-0.5, vmax=len(protos) - 0.5, origin="lower",
                      aspect="auto", extent=ext, interpolation="nearest")
            for (i, j), u in np.ndenumerate(unres if k == 0 else abst):
                if u and k == 0:
                    light = protos[int(G[i, j])] in ("RAW", "RO", "ZNE-L") if np.isfinite(G[i, j]) else False
                    a_.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           ec="#555555" if light else "#FFFFFF", lw=0))
                if u and k == 1:
                    a_.plot(j, i, marker="o", ms=4, mfc="none", mec="#000000", mew=0.9)
            for (i, j), u in np.ndenumerate(insuff):
                if u:
                    a_.plot(j, i, marker="x", ms=3, color="#999999")
            for (i, j), nf in np.ndenumerate(G_sup):
                if np.isfinite(nf) and nf < n_slice:
                    a_.plot(j + 0.38, i + 0.36, marker="v", ms=2.2, color="#000000", ls="")
            a_.set_title("%s — %s" % (fam, title), fontsize=7, loc="left")
        a_ = axes[r, 2]
        im = a_.imshow(np.ma.masked_invalid(G_reg), cmap=_div_cmap(), vmin=-vmax, vmax=vmax, origin="lower",
                       aspect="auto", extent=ext, interpolation="nearest")
        for (i, j), u in np.ndenumerate(insuff):
            if u:
                a_.plot(j, i, marker="x", ms=3, color="#999999")
        a_.set_title("%s — signed empirical regret" % fam, fontsize=7, loc="left")
        a_ = axes[r, 3]
        a_.imshow(np.ma.masked_invalid(G_sup), cmap=_seq_cmap(), vmin=0, vmax=n_cand, origin="lower",
                  aspect="auto", extent=ext, interpolation="nearest")
        for (i, j), nf in np.ndenumerate(G_sup):
            if np.isfinite(nf) and (j % 2 == 0 or (nf < n_slice and len(budgets) <= 6)):
                a_.text(j, i, "%d" % nf, ha="center", va="center", fontsize=6.0 if n_cand < 10 else 5.5,
                        color="#FFFFFF" if nf > 0.6 * n_cand else "#000000")
        a_.set_title("support: feasible candidates", fontsize=7, loc="left")
        for k in range(4):
            a_ = axes[r, k]
            a_.grid(False)
            a_.set_xticks(xt[::3], [_blabel(b) for b in budgets[::3]], fontsize=6.5)
            a_.set_yticks(range(len(scales)), ["%g" % s for s in scales], fontsize=6.5)
            if k == 0:
                a_.set_ylabel("error scale s")
            if r == len(slices) - 1:
                a_.set_xlabel("total budget B", fontsize=6.5)
    fig.colorbar(im, ax=axes[:, 2], fraction=0.05, pad=0.01, label="log10 regret ratio\n(signed; < 0: reference noise)")
    handles = [plt.Line2D([], [], marker="s", ls="", color=PROTOCOL_COLORS[p], label=p, ms=5) for p in used]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#000000",
                           label="abstained (SBS-B's choice shown)"),
                plt.Line2D([], [], marker="v", ls="", color="#000000", ms=3, label="a candidate of the slice infeasible here"),
                plt.Rectangle((0, 0), 1, 1, fc="#888888", hatch="////", ec="#FFFFFF", label="unresolved")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=6.5)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    n_neg = int(np.sum(cells.log_regret_policy < 0))
    tr = sorted(set(int(x) for x in cells.trials_per_candidate))
    n_short = int(np.sum(cells.trials_per_candidate < 2 * cfg["regime"]["min_trials"]))
    short = ("no cell is short of data (a cross would mark one)" if n_short == 0 else
             "%d cells have fewer than %d trials per half and are drawn blank with a cross"
             % (n_short, cfg["regime"]["min_trials"]))
    cap = ("Empirical performance regime maps of the controlled regime study (decision D23): three preregistered "
           "slices (4-qubit Ising, HEA and QAOA circuits; default input and observable; calibration-best layout on "
           "fake_fez). Within a slice only the device error scale s changes (every reported gate error and "
           "relaxation rate of the snapshot × s, applied to the snapshot the protocols use and to the execution law "
           "alike). Cells are the evaluated (budget, s) points; nothing is interpolated. Reference preference: the "
           "candidate with the lowest MSE over both label halves (%s trials); hatched where the runner-up is within "
           "1.96 standard errors of the difference (unresolved — overlap is not equivalence, and no equivalence "
           "procedure is implemented). Recommendation: the frozen recommender's choice (trained on the development "
           "set without these circuits) as deployed; a ring marks abstention to SBS-B under the threshold chosen on development "
           "rows, and the cell then shows SBS-B's choice (the model's own choice is in the source data). Regret: log10 "
           "regret ratio of the deployed policy (with abstention) on a shared, zero-centred scale; the %d negative "
           "estimates (the reference, identified on the selection half, scored worse on the evaluation half: "
           "estimation noise, not negative true regret) are drawn as they are. A triangle marks cells where a "
           "candidate is infeasible (counts in the support map); %s. The maps are descriptive and pointwise: the "
           "'unresolved' rule compares the top two candidates of one cell with an approximate standard error "
           "(design-level for learning protocols under fresh designs); it is not a simultaneous statement over "
           "candidates or cells. %s" % ("/".join(map(str, tr)), n_neg, short, LABEL_SIM))
    od = pd.DataFrame(out)
    if len(od):
        res = od[~od.unresolved]
        agree = float((res.model_choice == res.reference).mean()) if len(res) else float("nan")
        nref = {f: int(od[od.slice == f].reference.nunique()) for f in slices}
        agree_p = float((res.policy_choice == res.reference).mean()) if len(res) else float("nan")
        obs = ("%d of %d cells are resolved; among them the model's own choice equals the reference in %.0f %% and "
               "the deployed choice (after abstention) in %.0f %%; it abstained in %d of %d cells; distinct reference "
               "protocols per slice: %s; largest deployed log10 regret %.2f. "
               % (len(res), len(od), 100 * agree, 100 * agree_p, int(od.abstained.sum()), len(od),
                  ", ".join("%s %d" % kv for kv in nref.items()), float(od.log_regret_policy.max())))
    else:
        obs = ""
    interp = ("Observation: " + obs + "These are empirical regime maps, not phase diagrams: boundaries are known "
              "only to the grid's resolution and were not refined; an unresolved cell is not a tie.")
    return _save(fig, out_dir, "fig02_performance_regimes", dict(cells=pd.DataFrame(out)), cap, interp,
                 dict(slices=slices, min_trials=cfg["regime"]["min_trials"], regret_vmax=vmax))


# ------------------------------------------------------------------ Figure 3
def _regime_features(run_dir: str):
    import pandas as pd
    p = os.path.join(run_dir, "regimes", "analysis", "regime_features.csv")
    return pd.read_csv(p, encoding="utf-8") if os.path.exists(p) else pd.DataFrame()


SEMANTIC_LABEL = {"-lightcone (all derived)": "light cone + all its derived sums",
                  "-noise (calibration)": "every calibration-derived quantity",
                  "-simulated": "every simulated quantity"}
NOISE_SHORT = {"none": "baseline", "g1=3": "1q gates\n×3", "g2=2": "2q gates\n×2", "g2=3": "2q gates\n×3",
               "t=2": "T1, T2\n÷2", "t=3": "T1, T2\n÷3", "ro=2": "readout\n×2", "ro=0.5": "readout\n×0.5",
               "chi=0.25": "coherent\n25 %", "chi=0.5": "coherent\n50 %"}


def _noisecomp(run_dir: str):
    import pandas as pd
    d = os.path.join(run_dir, "noisecomp", "analysis")
    if not os.path.exists(os.path.join(d, "regime_cells.csv")):
        return pd.DataFrame(), pd.DataFrame()
    return (pd.read_csv(os.path.join(d, "regime_cells.csv"), encoding="utf-8"),
            pd.read_csv(os.path.join(d, "regime_protocols.csv"), encoding="utf-8"))


def fig03_feature_mechanisms(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Which information changes the decision? (A) retrained, retuned ablations: feature groups and semantic
    removals (everything derived from one source); (B) the same circuit, input, device and budget under four
    observables; (C) the same task under a controlled change of the device's overall error scale s; (D) three circuits
    under a controlled change of one noise component at a time (noise composition), the measured best protocol per
    cell with its resolution."""
    import pandas as pd
    from matplotlib.patches import Rectangle
    cfg = cfg or load_config()
    plt = _plt()
    adir = os.path.join(run_dir, "analysis")
    ab = pd.read_csv(os.path.join(adir, "ablation.csv"), encoding="utf-8")
    rp = pd.read_csv(os.path.join(run_dir, "regimes", "analysis", "regime_protocols.csv"), encoding="utf-8")
    feats = _regime_features(run_dir)
    ncells, nprot = _noisecomp(run_dir)
    fig = plt.figure(figsize=(7.4, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.3, 0.95], height_ratios=[1.0, 0.85])
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))
    axD = fig.add_subplot(gs[1, :])
    # A: group and semantic ablations (each variant retrained and retuned)
    a = ab[ab.variant != "all"].copy()
    kind = a["kind"] if "kind" in a.columns else pd.Series("group", index=a.index)
    a["kind"] = kind
    a["label"] = [SEMANTIC_LABEL.get(v, v) if k == "semantic" else GROUP_LABEL.get(v[1:], v[1:])
                  for v, k in zip(a.variant, a.kind)]
    a = pd.concat([a[a.kind == "group"].sort_values("diff_vs_all"), a[a.kind == "semantic"].sort_values("diff_vs_all")])
    y = np.arange(len(a))[::-1]
    for (yy, (_, r_)) in zip(y, a.iterrows()):
        sem = r_.kind == "semantic"
        axA.errorbar([r_.diff_vs_all], [yy], xerr=[[r_.diff_vs_all - r_.diff_lo], [r_.diff_hi - r_.diff_vs_all]],
                     fmt="D" if sem else "o", color="#D55E00" if sem else "#0072B2", ms=3.5, capsize=2, lw=1)
    ng = int((a.kind == "group").sum())
    if ng and ng < len(a):
        ysep = y[ng - 1] - 0.5
        axA.axhline(ysep, color="#999999", lw=0.6, ls=":")
        # the two kinds are named at the separator (a legend box collided with the tick labels)
        import matplotlib.transforms as mtrans
        tr = mtrans.blended_transform_factory(axA.transAxes, axA.transData)
        axA.text(0.98, ysep + 0.12, "● feature groups", transform=tr, ha="right", va="bottom", fontsize=6,
                 color="#0072B2")
        axA.text(0.98, ysep - 0.12, "◆ semantic removals", transform=tr, ha="right", va="top", fontsize=6,
                 color="#D55E00")
    axA.axvline(0, color="#333333", lw=0.8)
    axA.set_yticks(y, a.label, fontsize=6.5)
    axA.set_xlabel("Δ held-out mean log10 regret\n(removed − full; > 0: the information helps)")
    nA = int(ab.base_circuits.iloc[0]) if "base_circuits" in ab.columns else None
    axA.set_title("A  ablations, retrained%s" % (" (n = %d)" % nA if nA else ""), loc="left")
    # B: observable dependence
    op = cfg["observable_panel"]
    sb = rp[(rp.family == op["family"]) & (rp.B == op["budget"]) & (rp.device == op["device"]) &
            (rp.layout == op["layout"])]
    obs_order = [o for o in ("z_mid", "zz_mid", "default", "z_all") if o in set(sb.observable)]
    protos = [p for p in PROTOCOL_COLORS if p in set(rp.protocol) | set(nprot.get("protocol", []))]
    width = 0.8 / max(len(protos), 1)
    rowsB = []
    for k, o in enumerate(obs_order):
        e = sb[sb.observable == o].set_index("protocol")
        best = e.mse_pooled.idxmin()
        for q, p in enumerate(protos):
            if p not in e.index:
                continue
            x = k - 0.4 + (q + 0.5) * width
            m, se = float(e.loc[p, "mse_pooled"]), float(e.loc[p, "se_pooled"])
            lo = max(m - 1.96 * se, m * 1e-3)
            axB.errorbar([x], [m], yerr=[[m - lo], [1.96 * se]], fmt=PROTOCOL_MARKERS[p], color=PROTOCOL_COLORS[p],
                         ms=3, lw=0.8, capsize=0)
            if bool(e.loc[p, "chosen"]):
                axB.plot([x], [m], marker="o", ms=7, mfc="none", mec="#000000", mew=0.9)
            if p == best:
                axB.plot([x], [m * 0.45], marker="^", ms=3.5, color="#000000")
            rowsB.append(dict(observable=o, protocol=p, mse=m, se=se, chosen=bool(e.loc[p, "chosen"]),
                              measured_best=bool(p == best)))
    axB.set_yscale("log")
    yl = axB.get_ylim()
    axB.set_ylim(yl[0], yl[1] * 20)                  # headroom: the legend sits above the data
    lab_ = []
    for o in obs_order:
        txt = o
        if not feats.empty:
            f = feats[(feats.family == op["family"]) & (feats.observable == o) & (feats.device == op["device"]) &
                      (feats.layout == op["layout"])]
            if len(f):
                txt = "%s\nw%d·LC%d\nλ=%s" % (o, int(f.obs_weight.iloc[0]), int(f.lc_qubits.iloc[0]),
                                             ("%.3f" % float(f.lam_lc.iloc[0])).lstrip("0"))
        lab_.append(txt)
    axB.set_xticks(range(len(obs_order)), lab_, fontsize=6 if len(obs_order) <= 3 else 5.5)
    axB.plot([], [], ls="", marker="o", ms=7, mfc="none", mec="#000000", mew=0.9, label="model's choice")
    axB.plot([], [], ls="", marker="^", ms=3.5, color="#000000", label="measured best (below its point)")
    axB.legend(frameon=False, fontsize=6.5, loc="upper left")
    axB.set_ylabel("MSE (both halves, ±1.96 SE)")
    axB.set_title("B  observable (%s, B = %s)" % (op["family"], _blabel(op["budget"])), loc="left")
    # C: overall noise strength
    npn = cfg["noise_panel"]
    sc = rp[(rp.family == npn["family"]) & (rp.B == npn["budget"]) & (rp.layout == npn["layout"]) &
            (rp.observable == npn["observable"])].copy()
    sc["s"] = sc.device.map(lambda d: float(d.split("s=")[1].split(",")[0]) if "s=" in d else 1.0)
    rowsC = []
    for p in protos:
        e = sc[sc.protocol == p].sort_values("s")
        if e.empty:
            continue
        axC.errorbar(e.s, e.mse_pooled, yerr=1.96 * e.se_pooled, fmt=PROTOCOL_MARKERS[p] + "-",
                     color=PROTOCOL_COLORS[p], ms=3, lw=1, capsize=0, label=p)
        ch = e[e.chosen]
        axC.plot(ch.s, ch.mse_pooled, ls="", marker="o", ms=7, mfc="none", mec="#000000", mew=0.9)
        for _, r_ in e.iterrows():
            rowsC.append(dict(s=r_.s, protocol=p, mse=r_.mse_pooled, se=r_.se_pooled, chosen=bool(r_.chosen)))
    best_s = sc.loc[sc.groupby("s").mse_pooled.idxmin()] if len(sc) else sc
    for _, r_ in best_s.iterrows():
        axC.plot([r_.s], [r_.mse_pooled * 0.45], marker="^", ms=3.5, color="#000000")
    axC.set_yscale("log")
    yl = axC.get_ylim()
    many = len(protos) > 8                           # 1.8.0: 17 candidates — one legend for B and C below the figure
    axC.set_ylim(yl[0], yl[1] * (3 if many else 60))  # headroom: the legend sits above the data (≤ 8 candidates)
    axC.set_xlabel("device error scale s (all rates × s)")
    axC.set_title("C  noise strength (%s, B = %s)" % (npn["family"], _blabel(npn["budget"])), loc="left")
    if many:
        fig.legend(handles=[plt.Line2D([], [], marker=PROTOCOL_MARKERS[p], color=PROTOCOL_COLORS[p], ms=4, lw=1,
                                       label=p) for p in protos],
                   loc="outside lower center", ncol=9, frameon=False, fontsize=6, title="candidates (B, C)",
                   title_fontsize=6)
    else:
        axC.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    # D: noise composition (one component changed at a time; three circuits)
    ncp = cfg.get("noise_composition_panel", {}) or {}
    BD_ = int(ncp.get("budget", 100000))
    rowsD = []
    if len(ncells):
        e = ncells[ncells.B == BD_].copy()
        e["change"] = e.noise_change.fillna("none").astype(str)
        order = [k for k in NOISE_SHORT if k in set(e.change)] + sorted(set(e.change) - set(NOISE_SHORT))
        fams = [f for f in ncp.get("families", ["ising", "hea", "qaoa"]) if f in set(e.family)]
        for i, fam in enumerate(fams):
            for j, ch_ in enumerate(order):
                c = e[(e.family == fam) & (e.change == ch_)]
                if c.empty:
                    axD.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#FFFFFF", ec="#BBBBBB", lw=0.5))
                    continue
                r_ = c.iloc[0]
                col = PROTOCOL_COLORS.get(r_.reference, "#BBBBBB")
                axD.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=col, alpha=0.38, ec="#FFFFFF", lw=1.5))
                if bool(r_.unresolved):
                    axD.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none", ec="#555555", lw=0, hatch="////"))
                agree = r_.model_choice == r_.reference
                # a light box keeps the text legible over the hatching of unresolved cells
                bx = dict(boxstyle="round,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.85) if bool(r_.unresolved) \
                    else None
                axD.text(j, i - 0.2, r_.reference, ha="center", va="center", fontsize=6.5, fontweight="bold",
                         bbox=bx)
                axD.text(j, i + 0.05, "λ=%.2f" % r_.lam_total, ha="center", va="center", fontsize=6, bbox=bx)
                axD.text(j, i + 0.28, "model %s" % r_.model_choice, ha="center", va="center", fontsize=6,
                         color="#000000", bbox=bx)
                if not agree:                         # the disagreement mark sits in the cell's corner
                    axD.text(j + 0.44, i - 0.4, "✗", ha="right", va="top", fontsize=7, color="#000000",
                             fontweight="bold")
                rowsD.append(dict(family=fam, change=ch_, B=BD_, reference=r_.reference, runner_up=r_.runner_up,
                                  unresolved=bool(r_.unresolved), model_choice=r_.model_choice, lam_total=r_.lam_total,
                                  log_regret_model=r_.log_regret_model))
        axD.set_xlim(-0.5, len(order) - 0.5)
        axD.set_ylim(len(fams) - 0.5, -0.5)
        axD.set_xticks(range(len(order)), [NOISE_SHORT.get(k, k) for k in order], fontsize=6)
        axD.set_yticks(range(len(fams)), ["%s (n=4)" % f for f in fams], fontsize=6.5)
        axD.tick_params(length=0)
        axD.grid(False)
        for s_ in axD.spines.values():
            s_.set_visible(False)
        axD.add_patch(Rectangle((0, 0), 0, 0, fc="none", ec="#555555", hatch="////", label="top two unresolved "
                                "(within 1.96 SE of their difference)"))
        axD.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2)
    else:
        axD.text(0.5, 0.5, "noise-composition study not analyzed in this run", ha="center", va="center",
                 transform=axD.transAxes, fontsize=7)
        axD.set_axis_off()
    axD.set_title("D  noise composition: measured best protocol, one component changed at a time (B = %s;\n"
                  "fill: best protocol; λ: total fault weight; model: the frozen model's choice; ✗ (corner): it differs)"
                  % _blabel(BD_), loc="left", fontsize=7)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    n_comp = len({r["family"] for r in rowsD})
    cap = ("Which information changes the decision? (A) Ablations of the deployed configuration's feature set, each "
           "variant retrained and retuned with the same inner group-4-fold protocol and scored on held-out base "
           "circuits (group 5-fold; %s independent base circuits): feature groups (circles) and semantic removals "
           "(diamonds: every quantity derived from one source — the light cone with all its noise-weighted sums; "
           "every simulated quantity; removing every calibration-derived noise sum is the noise group itself and is "
           "shown once). Paired difference in mean log10 regret "
           "to the full model with 95 %% bootstrap intervals over base circuits; predictive, not causal, importance. "
           "(B) One circuit, input, device and budget with only the observable changing (support weight w, light-cone "
           "qubits LC and light-cone fault weight λ under each label): candidate MSEs over both label halves with "
           "±1.96 SE (SE over trials; over learning designs for the learning candidates under fresh designs); ring: the frozen "
           "model's own choice (before abstention); triangle: the measured best. (C) The same task with every "
           "calibrated error rate of the device (snapshot and truth alike) multiplied by s at a fixed budget; marks "
           "as in (B). (D) The noise-composition study: %d circuits (one task per cell; n = 4 on the calibration-best "
           "chain of fake_fez) under one noise component changed at a time — one- or two-qubit gate errors, "
           "relaxation (T1, T2 divided), readout, or a coherent fraction of every gate error (exact engine) — the "
           "snapshot and the execution law changing together. Each cell names the measured best protocol (both "
           "halves), the total fault weight λ of the setting and the frozen model's choice (✗: differs); hatching: "
           "the runner-up within 1.96 SE of the difference (pointwise, not an equivalence test). Panels B–D are "
           "single-task controlled comparisons: they show whether the ranking moves with the controlled variable for "
           "these tasks, not how often it does in general. %s"
           % (nA if nA else "n", n_comp, LABEL_SIM))
    bests = ", ".join("s=%g: %s" % (r_.s, r_.protocol) for _, r_ in best_s.sort_values("s").iterrows())
    bestB = ", ".join("%s: %s" % (r_["observable"], r_["protocol"]) for r_ in rowsB if r_["measured_best"])
    sem = a[a.kind == "semantic"]
    semtxt = "; ".join("%s %+.3f [%+.3f, %+.3f]" % (SEMANTIC_LABEL.get(r_.variant, r_.variant), r_.diff_vs_all,
                                                     r_.diff_lo, r_.diff_hi) for _, r_ in sem.iterrows())
    if rowsD:
        dD = pd.DataFrame(rowsD)
        base = dD[dD.change == "none"].set_index("family").reference.to_dict()
        moved = dD[(dD.change != "none") & dD.apply(lambda r_: base.get(r_.family) not in (None, r_.reference),
                                                     axis=1)]
        moved_res = moved[~moved.unresolved]
        dtxt = ("the measured best differs from the circuit's baseline in %d of %d changed cells (%d of them "
                "resolved); the model's choice differs from the measured best in %d of %d cells"
                % (len(moved), int((dD.change != "none").sum()), len(moved_res),
                   int((dD.model_choice != dD.reference).sum()), len(dD)))
    else:
        dtxt = "noise-composition study not available"
    interp = ("Semantic ablations (Δ mean log10 regret): %s. Measured best per observable (B): %s. Measured best per "
              "error scale (C): %s. Noise composition (D): %s. A change of the measured best between settings whose "
              "top two are unresolved is not evidence of a regime change."
              % (semtxt or "n/a", bestB or "n/a", bests or "n/a", dtxt))
    return _save(fig, out_dir, "fig03_feature_mechanisms",
                 dict(ablation=ab, observable=pd.DataFrame(rowsB), noise=pd.DataFrame(rowsC),
                      composition=pd.DataFrame(rowsD)), cap, interp,
                 dict(observable_panel=op, noise_panel=npn, noise_composition_panel=ncp))


# ------------------------------------------------------------------ Figure 4
BG_COLORS = {"SBS-B": "#D55E00", "MODEL[ridge|deployable]": "#E69F00", "MODEL[struct|deployable]": "#009E73",
             "MODEL[hgb|deployable]": "#0072B2"}
BG_MARKERS = {"SBS-B": "D", "MODEL[ridge|deployable]": "s", "MODEL[struct|deployable]": "^",
              "MODEL[hgb|deployable]": "o"}
_BG_FAMILY = {"ridge": ("#E69F00", "s"), "struct": ("#009E73", "^"), "hgb": ("#0072B2", "o")}


def _bg_style(p: str) -> tuple:
    """(colour, marker) of a policy in the budget-extrapolation panel: by name, else by model family (1.8.0: the
    configurations are named with the deployable18 set; a family keeps its colour whatever the feature set)."""
    if p in BG_COLORS:
        return BG_COLORS[p], BG_MARKERS.get(p, "o")
    if p.startswith("MODEL["):
        return _BG_FAMILY.get(p[len("MODEL["):].split("|")[0], ("#555555", "o"))
    return "#555555", "o"


def _short_policy(p: str) -> str:
    return {"SBS-B": "SBS-B (budget-only)", "SIZE-B": "SIZE-B"}.get(
        p, p.replace("MODEL[", "").replace("]", "").replace("ridge", "Ridge").replace("hgb", "boosted trees")
        .replace("struct", "structured b²+v/B").replace("|", " | "))


def fig04_generalization_learning(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Does the model generalize, and how much data does it need? (A) generalization matrix of mean log10 regret,
    (B) learning curve in independent training base circuits, (C) budget generalization with restricted training
    ranges, one facet per training range: SBS-B, Ridge, the structured risk model and boosted trees."""
    import pandas as pd
    cfg = cfg or load_config()
    plt = _plt()
    adir = os.path.join(run_dir, "analysis")
    gcfg = cfg["generalization"]
    cols = list(gcfg["columns"].items())
    rows_ = [r for r in gcfg["rows"] if os.path.exists(os.path.join(adir, "pooled_%s.csv" % r))]
    M = np.full((len(rows_), len(cols)), np.nan)
    cells = []
    for i, split in enumerate(rows_):
        s = pd.read_csv(os.path.join(adir, "pooled_%s.csv" % split), encoding="utf-8")
        s = s[s.B.astype(str) == "all"].set_index("policy")
        for j, (role, pol) in enumerate(cols):
            if pol not in s.index or "mean_log_regret" not in s.columns:
                continue
            r_ = s.loc[pol]
            M[i, j] = r_.mean_log_regret
            cells.append(dict(split=split, role=role, policy=pol, mean_log_regret=r_.mean_log_regret,
                              ci_lo=r_.log_regret_lo, ci_hi=r_.log_regret_hi, base_circuits=int(r_.base_circuits),
                              rows=int(r_.rows)))
    bg = pd.read_csv(os.path.join(adir, "budget_generalization.csv"), encoding="utf-8")
    pols = [p for p in cfg["budget_generalization"]["policies"] if p in set(bg.policy)]
    ranges = list(dict.fromkeys(bg.max_train_B.astype(str)))
    fig = plt.figure(figsize=(7.4, 5.6), layout="constrained")
    gs0 = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    top = gs0[0].subgridspec(1, 2, width_ratios=[2.1, 1.0])
    bot = gs0[1].subgridspec(1, max(len(ranges), 1))
    axA = fig.add_subplot(top[0, 0])
    axB = fig.add_subplot(top[0, 1])
    vmax = float(np.nanmax(M)) if np.isfinite(M).any() else 1.0
    im = axA.imshow(np.ma.masked_invalid(M), cmap=_seq_cmap(), vmin=0, vmax=vmax, aspect="auto")
    cdf = pd.DataFrame(cells)
    for c in cells:
        i, j = rows_.index(c["split"]), [k for k, _ in cols].index(c["role"])
        axA.text(j, i, "%.2f\n[%.2f, %.2f]\nn=%d" % (c["mean_log_regret"], c["ci_lo"], c["ci_hi"],
                                                     c["base_circuits"]),
                 ha="center", va="center", fontsize=6.5, color="#FFFFFF" if c["mean_log_regret"] > 0.6 * vmax
                 else "#000000")
    axA.set_xticks(range(len(cols)), ["%s\n%s" % (k, p.replace("MODEL[", "").replace("]", "").replace("|", "\n"))
                                      for k, p in cols], fontsize=6.5)
    axA.set_yticks(range(len(rows_)), rows_, fontsize=6.5)
    axA.grid(False)
    axA.set_title("A  mean log10 regret (held out)", loc="left")
    cb = fig.colorbar(im, ax=axA, fraction=0.04, pad=0.02)
    cb.set_label("mean log10 regret", fontsize=6.5)
    # B learning curve
    lc = pd.read_csv(os.path.join(adir, "learning_curve.csv"), encoding="utf-8")
    key = "mean_log_regret" if "mean_log_regret" in lc.columns else "diff_vs_sbs"
    # one point per training fraction (every fold and repeat of it), at its mean number of training circuits: the
    # fold sizes differ by one circuit, and grouping by the exact count would draw single draws as separate points
    grp = "fraction" if "fraction" in lc.columns else "train_base_circuits"
    g = lc.groupby(grp).agg(n=("train_base_circuits", "mean"), m=(key, "mean"),
                            lo=(key, lambda x: np.nanquantile(x, 0.1)), hi=(key, lambda x: np.nanquantile(x, 0.9)),
                            draws=(key, "size"))
    g = g.reset_index().sort_values("n")
    axB.plot(g.n, g.m, "o-", color="#0072B2", ms=3.5, lw=1.4, label="model (fixed setting)")
    axB.fill_between(g.n, g.lo, g.hi, color="#0072B2", alpha=0.15, lw=0)
    gs_ = None
    if "sbs_mean_log_regret" in lc.columns:
        gs_ = lc.groupby(grp).agg(n=("train_base_circuits", "mean"), m=("sbs_mean_log_regret", "mean")).sort_values("n")
        axB.plot(gs_.n, gs_.m, "s--", color="#E69F00", ms=3, lw=1.1, label="SBS (same subset)")
    axB.set_xlabel("independent training base circuits")
    axB.set_ylabel("mean log10 regret (held-out folds)")
    axB.set_title("B  learning curve", loc="left")
    axB.legend(frameon=False, fontsize=6.5)
    # C budget generalization: one facet per training range
    axC = []
    for k, rg in enumerate(ranges):
        ax = fig.add_subplot(bot[0, k], sharey=axC[0] if axC else None)
        axC.append(ax)
        for p in pols:
            e = bg[(bg.max_train_B.astype(str) == rg) & (bg.policy == p)].sort_values("B")
            if e.empty:
                continue
            c, mk = _bg_style(p)
            ax.plot(e.B, e.mean_log_regret, "-", color=c, lw=1.1)
            inr = e[e.regime == "in range"]
            ext = e[e.regime != "in range"]
            ax.plot(inr.B, inr.mean_log_regret, mk, color=c, ms=3.2, ls="")
            ax.plot(ext.B, ext.mean_log_regret, mk, mfc="white", mec=c, ms=3.6, ls="")
        if rg != "full":
            ax.axvspan(float(rg) * 1.3, float(bg.B.max()) * 1.5, color="#EEEEEE", lw=0, zorder=0)
            ax.axvline(float(rg), color="#555555", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("test budget B")
        ax.set_title("%s  train B ≤ %s" % ("C" if k == 0 else "", (_blabel(float(bg.B.max())) + " (full range)") if rg == "full"
                                            else _blabel(float(rg))), loc="left")
        if k == 0:
            ax.set_ylabel("mean log10 regret")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    if axC:
        hd = [plt.Line2D([], [], color=_bg_style(p)[0], marker=_bg_style(p)[1], ms=3.2, lw=1.1,
                         label=_short_policy(p)) for p in pols]
        hd += [plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#333333", label="extrapolation (shaded)")]
        fig.legend(handles=hd, frameon=False, fontsize=6.5, loc="outside lower center", ncol=5)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    nC = int(bg.base_circuits.max()) if "base_circuits" in bg.columns and len(bg) else None
    cap = ("Generalization and data needs. (A) Mean log10 regret ratio against the per-task reference (protocol "
           "identified on the selection half, scored on the evaluation half — the same reference definition in every "
           "cell), all budgets, on held-out base circuits: same distribution (group 5-fold), leave one family out, "
           "held-out device (fake_kyiv, crossed with the folds), larger width (train n ≤ 5, test n = 6); columns: "
           "the budget-dependent fixed policy as reference, the size-and-budget tree, Ridge, the structured risk "
           "model (b² + v/((1−f)B) per protocol with ridge-regularized task effects) and boosted trees, all on "
           "deployable features; 95 %% base-circuit bootstrap intervals and the number n of independent test base "
           "circuits in each cell. (B) Learning curve: the model with a fixed setting (deployable features) and SBS "
           "trained on random subsets of each training fold's base circuits, scored on the fixed held-out fold; mean "
           "over folds and repeats, band 10–90 %% of fold × repeat draws (not a confidence interval). (C) SBS-B, "
           "Ridge, the structured model and boosted trees trained only on budgets up to a maximum (dotted line; one "
           "facet per maximum) and scored at every budget of held-out circuits (group 5-fold%s); filled: in range, "
           "open on grey: extrapolation; at an unseen budget SBS-B uses the nearest trained budget. %s"
           % (", %d base circuits" % nC if nC else "", LABEL_SIM))
    diffs = []
    nl = dict(cols).get("nonlinear")
    for split in rows_:
        t = pd.read_csv(os.path.join(adir, "pooled_%s.csv" % split), encoding="utf-8")
        t = t[(t.B.astype(str) == "all") & (t.policy == nl)]
        if len(t) and "diff_vs_sbsb" in t.columns:
            r_ = t.iloc[0]
            diffs.append("%s %+.3f [%+.3f, %+.3f]" % (split, r_.diff_vs_sbsb, r_.diff_vs_sbsb_lo, r_.diff_vs_sbsb_hi))
    lc_txt = ("learning curve: mean log10 regret %.3f at %.0f and %.3f at %.0f training base circuits (mean over "
              "folds and repeats; SBS %.3f and %.3f)" % (g.m.iloc[0], g.n.iloc[0], g.m.iloc[-1], g.n.iloc[-1],
                                                        gs_.m.iloc[0], gs_.m.iloc[-1])) \
        if len(g) and gs_ is not None else ""
    ext_txt = []
    for rg in ranges:
        if rg == "full":
            continue
        e = bg[(bg.max_train_B.astype(str) == rg) & (bg.regime != "in range")]
        if len(e):
            m_ = e.groupby("policy").mean_log_regret.mean()
            ext_txt.append("train B ≤ %s: %s" % (rg, ", ".join("%s %.3f" % (_short_policy(p), m_[p]) for p in pols
                                                                if p in m_.index)))
    interp = ("Observation: paired difference of the nonlinear column to SBS-B in mean log10(MSE + 10⁻⁶) (< 0 favors the "
              "model): %s; %s. Mean log10 regret over the extrapolated budgets: %s. Matrix rows are different claims "
              "(interpolation in the same distribution, extrapolation in family, device and width); panel C "
              "separates budget interpolation from extrapolation." % ("; ".join(diffs) or "n/a", lc_txt or "n/a",
                                                                      "; ".join(ext_txt) or "n/a"))
    return _save(fig, out_dir, "fig04_generalization_learning",
                 dict(matrix=cdf, learning_curve=g, budget=bg), cap, interp,
                 dict(rows=rows_, columns=dict(cols), budget_policies=pols))


# ------------------------------------------------------------------ Figure 5
def fig05_reliability_abstention(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Can the system recognize unreliable recommendations? Evaluated by nested cross-validation (1.7.0, review R2):
    (A) coverage against the accepted rows' regret, with random rejection and each outer fold's operating point;
    (B) whole-policy arithmetic MSE after the fallback against coverage, with random rejection at matched coverage;
    (C) the nested operating point against the model alone, SBS-B, random rejection, SIZE-B and SBS; (D) measured
    coverage of the log-risk intervals, labelled by what they guarantee."""
    import pandas as pd
    cfg = cfg or load_config()
    plt = _plt()
    adir = os.path.join(run_dir, "analysis")
    cv = pd.read_csv(os.path.join(adir, "nested_curve.csv"), encoding="utf-8").sort_values("coverage")
    sm = pd.read_csv(os.path.join(adir, "nested_summary.csv"), encoding="utf-8").set_index("policy")
    rows = pd.read_csv(os.path.join(adir, "nested_rows.csv"), encoding="utf-8")
    folds = pd.read_csv(os.path.join(adir, "nested_folds.csv"), encoding="utf-8")
    cal = pd.read_csv(os.path.join(adir, "calibration.csv"), encoding="utf-8")
    ic = pd.read_csv(os.path.join(adir, "interval_coverage.csv"), encoding="utf-8")
    conf = str(sm.configuration.iloc[0])
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.0), layout="constrained")
    a = ax[0, 0]
    a.plot(cv.coverage, cv.accepted_log_regret, "-", color="#0072B2", lw=1.4, label="margin threshold (outer rows)")
    a.axhline(float(rows.log_regret_model.mean()), color="#999999", ls="--", lw=1.1,
              label="random rejection (any coverage)")
    fr = rows.groupby("fold").apply(lambda e: pd.Series(dict(coverage=e.accepted.mean(),
                                                             acc=e.log_regret_model[e.accepted].mean()
                                                             if e.accepted.any() else np.nan)),
                                    include_groups=False).reset_index()
    a.plot(fr.coverage, fr.acc, "o", mfc="white", mec="#D55E00", ms=4, ls="", label="each outer fold (nested τ)")
    a.plot([sm.loc["policy", "coverage"]], [sm.loc["policy", "accepted_log_regret"]], "*", ms=9, color="#D55E00",
           ls="", label="nested operating point")
    a.set_xlabel("coverage (share receiving the recommendation)")
    a.set_ylabel("mean log10 regret, accepted rows")
    a.set_title("A  coverage–risk (accepted rows)", loc="left")
    a.legend(frameon=False, fontsize=6.5, loc="best")
    a = ax[0, 1]
    a.plot(cv.coverage, cv.policy_mse, "-", color="#009E73", lw=1.4, label="selective policy")
    a.plot(cv.coverage, cv.random_policy_mse, "--", color="#999999", lw=1.1, label="random rejection (expected)")
    a.axhline(sm.loc["fallback (SBS-B)", "mse"], color="#D55E00", lw=0.9, ls=":", label="SBS-B only")
    a.axhline(sm.loc["model", "mse"], color="#0072B2", lw=0.9, ls=":", label="model only")
    p_ = sm.loc["policy"]
    a.errorbar([p_.coverage], [p_.mse], yerr=[[p_.mse - p_.mse_lo], [p_.mse_hi - p_.mse]], fmt="*", ms=9,
               color="#D55E00", capsize=2, label="nested operating point (95 % CI)")
    rr = sm.loc["random rejection"]
    a.errorbar([rr.coverage], [rr.mse], yerr=[[rr.mse - rr.random_draws_mse_lo], [rr.random_draws_mse_hi - rr.mse]],
               fmt="s", ms=4, mfc="white", color="#555555", capsize=2, label="random rejection, matched coverage")
    a.set_yscale("log")
    yl = a.get_ylim()
    a.set_ylim(yl[0] / 2.2, yl[1])                   # room below the curves for the legend
    a.set_xlabel("coverage")
    a.set_ylabel("mean MSE, all rows (fallback included)")
    a.set_title("B  whole-policy arithmetic MSE", loc="left")
    a.legend(frameon=False, fontsize=6, loc="lower left")
    a = ax[1, 0]
    order = ["policy", "model", "fallback (SBS-B)", "random rejection", "SIZE-B", "SBS"]
    order = [o for o in order if o in sm.index]
    y = np.arange(len(order))[::-1]
    a.errorbar([sm.loc[o, "diff_log_vs_sbsb"] for o in order], y,
               xerr=[[sm.loc[o, "diff_log_vs_sbsb"] - sm.loc[o, "diff_log_vs_sbsb_lo"] for o in order],
                     [sm.loc[o, "diff_log_vs_sbsb_hi"] - sm.loc[o, "diff_log_vs_sbsb"] for o in order]],
               fmt="o", color="#0072B2", ms=4, capsize=2)
    for yy, o in zip(y, order):
        a.annotate("MSE %.2g" % sm.loc[o, "mse"], (sm.loc[o, "diff_log_vs_sbsb_hi"], yy), xytext=(5, 0), va="center",
                   textcoords="offset points", fontsize=6.5, color="#333333")
    a.axvline(0, color="#333333", lw=0.8)
    a.set_yticks(y, [o if o != "policy" else "selective (nested τ)" for o in order])
    a.set_xlabel("Δ mean log10(MSE + 10⁻⁶) vs SBS-B (< 0 better), 95 % CI")
    a.set_title("C  nested operating point vs baselines", loc="left")
    a = ax[1, 1]
    g = cal.groupby(["kind", "nominal"]).agg(cov=("coverage", "mean"), lo=("coverage", "min"),
                                             hi=("coverage", "max")).reset_index()
    for kind, c, mk in (("per candidate", "#0072B2", "o"), ("selected candidate", "#D55E00", "s"),
                        ("simultaneous", "#009E73", "^")):
        e = g[g.kind == kind].sort_values("nominal")
        if e.empty:
            continue
        a.errorbar(e.nominal, e["cov"], yerr=[e["cov"] - e.lo, e.hi - e["cov"]], fmt=mk + "-", color=c, ms=3.5,
                   lw=1, capsize=2, label="cv-residual, %s" % kind)
    sc = ic[ic.kind.isin(["split-conformal", "split-conformal-simultaneous"])]
    for kind, c, mk in (("split-conformal", "#CC79A7", "D"), ("split-conformal-simultaneous", "#56B4E9", "v")):
        e = sc[sc.kind == kind]
        if e.empty:
            continue
        x = float(e.nominal.iloc[0]) + (0.01 if kind.endswith("simultaneous") else -0.01)
        a.errorbar([x], [e.coverage.mean()], yerr=[[e.coverage.mean() - e.coverage.min()],
                                                   [e.coverage.max() - e.coverage.mean()]],
                   fmt=mk, color=c, ms=4, capsize=2,
                   label="split-conformal, %s%s" % (
                       "simultaneous" if kind.endswith("simultaneous") else "per protocol",
                       " (∞ in %d/%d folds)" % (int((~e.finite.astype(bool)).sum()), len(e))
                       if "finite" in e and (~e.finite.astype(bool)).any() else ""))
    a.plot([0.45, 1], [0.45, 1], color="#333333", lw=0.8)
    lo_ = min([0.45] + [float(v) for v in g.lo.dropna()])
    a.set_ylim(max(0.0, lo_ - 0.45), 1.03)          # room below the diagonal for the legend
    a.set_xlabel("nominal coverage")
    a.set_ylabel("measured coverage (held-out circuits)")
    a.set_title("D  interval coverage and what it guarantees", loc="left")
    a.legend(frameon=False, fontsize=6, loc="lower right",
             title="cv-residual: no guarantee\nsplit-conformal: ≥ 1−α if exchangeable", title_fontsize=6)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    tau_txt = ", ".join("%.2f" % t for t in folds.tau)
    cap = ("Reliability of the recommender %s, evaluated by nested cross-validation over base circuits (outer group "
           "5-fold; in each outer training set an inner group 4-fold chooses the threshold τ; the frozen rule is "
           "applied once to the outer test fold; every base circuit is tested once; %d base circuits, %d rows). The "
           "rule: use the model's choice when the margin z = (P₂ − P₁)/√(s₁² + s₂²) between its two lowest predicted "
           "log10 risks is at least τ, otherwise SBS-B (no shots). Outer folds' τ: %s. (A) Mean log10 regret of the "
           "accepted rows against coverage (descriptive sweep over τ on the outer rows), random rejection, each fold's "
           "nested operating point and the pooled one. (B) Whole-policy arithmetic MSE over all rows with the "
           "fallback's loss included (base circuits weighted equally); random rejection at the operating point's "
           "coverage per fold (expected value; bar: 2.5–97.5 %% of random acceptance sets). (C) Paired difference to "
           "SBS-B in mean log10(MSE + 10⁻⁶) with 95 %% base-circuit bootstrap intervals; arithmetic MSE annotated. (D) Measured "
           "coverage of log10-risk intervals on held-out circuits: inner-CV residual intervals (statement kind "
           "cv-residual: no finite-sample guarantee; bars: range over folds) per candidate, for the selected candidate "
           "and Bonferroni-simultaneous; group split-conformal intervals (calibration circuits held out of the fit; "
           "finite-sample coverage ≥ 1 − α for all rows of a new circuit drawn from the same distribution — an "
           "assumption that fails for new families, devices or budget ranges). Measured coverage is a property of these "
           "circuits, not a guarantee for others. %s" % (conf, int(sm.base_circuits.iloc[0]), int(sm.rows.iloc[0]),
                                                         tau_txt, LABEL_SIM))
    verdict = ("At the nested operating point (coverage %.2f) the whole-policy mean MSE is %.3g [%.3g, %.3g], against "
               "%.3g for the model alone, %.3g for SBS-B alone and %.3g for random rejection at the same coverage; "
               "Δ mean log10(MSE + 10⁻⁶) vs SBS-B %+.3f [%+.3f, %+.3f], vs random rejection %+.3f [%+.3f, %+.3f]." % (
                   p_.coverage, p_.mse, p_.mse_lo, p_.mse_hi, sm.loc["model", "mse"], sm.loc["fallback (SBS-B)", "mse"],
                   rr.mse, p_.diff_log_vs_sbsb, p_.diff_log_vs_sbsb_lo, p_.diff_log_vs_sbsb_hi, p_.diff_log_vs_random,
                   p_.diff_log_vs_random_lo, p_.diff_log_vs_random_hi))
    interp = (verdict + " Abstention is useful only if the selective policy beats both the model alone and random "
              "rejection at matched coverage on the whole-policy scale; a lower accepted-subset regret (A) alone does "
              "not establish reliability.")
    return _save(fig, out_dir, "fig05_reliability_abstention",
                 dict(curve=cv, summary=sm.reset_index(), folds=folds, calibration=g, intervals=ic), cap, interp)


# ------------------------------------------------------------------ Figure 6
def fig06_mismatch_diagnostics(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Does model mismatch predict recommendation failure, and does reacting to a diagnostic pay for its shots? (A)
    controlled mismatch response of the recommender's regret; (B) a single deployment's diagnostic statistic against
    the subsequent recommendation failure; (C) equal-budget gains and losses of reacting to the diagnostic, against
    spending the same shots on the target and against random reactions, with the learned reaction evaluated on
    discrepancy types it was not trained on (1.7.0)."""
    import pandas as pd
    from .evaluate import bootstrap
    cfg = cfg or load_config()
    plt = _plt()
    mdir = os.path.join(run_dir, "mismatch")
    adir = os.path.join(mdir, "analysis")
    d = pd.read_csv(os.path.join(mdir, "diagnostics.csv"), encoding="utf-8")
    p = pd.read_csv(os.path.join(mdir, "policies.csv"), encoding="utf-8")
    draws = pd.read_csv(os.path.join(adir, "deployment_draws.csv"), encoding="utf-8")
    disc = pd.read_csv(os.path.join(adir, "discrimination.csv"), encoding="utf-8")
    react = pd.read_csv(os.path.join(adir, "reactions.csv"), encoding="utf-8")
    wide = p.pivot_table(index=["scenario", "type", "task_id", "base_id", "B"], columns="policy",
                         values="risk").reset_index()
    wide["log_regret"] = np.log10((wide["TARGET@B"] + EPS0) / (wide["BEST"] + EPS0))
    mc = cfg["mismatch"]
    fig = plt.figure(figsize=(7.4, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 6)
    axA = [fig.add_subplot(gs[0, 2 * k:2 * k + 2]) for k in range(3)]
    axB = fig.add_subplot(gs[1, :3])
    axC = fig.add_subplot(gs[1, 3:])
    bcol = {b: c for b, c in zip(mc["budgets"], ("#9ECAE1", "#3182BD", "#08306B"))}
    A = []
    for k, (fname, fc) in enumerate(mc["facets"].items()):
        a_ = axA[k]
        for b in mc["budgets"]:
            xs, ms, los, his = [], [], [], []
            for scen, x in fc["points"].items():
                e = wide[(wide.scenario == scen) & (wide.B == b)]
                if e.empty:
                    continue
                m, lo, hi = bootstrap({"v": e.log_regret.to_numpy()}, e.base_id.to_numpy(), 1000, 0)[0]["v"]
                xs.append(x), ms.append(m), los.append(lo), his.append(hi)
                A.append(dict(facet=fname, scenario=scen, strength=x, B=int(b), mean_log_regret=m, ci_lo=lo,
                              ci_hi=hi, base_circuits=int(e.base_id.nunique())))
            if xs:
                a_.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms], fmt="o-", color=bcol[b], ms=3,
                            lw=1, capsize=1.5, label="B = %s" % _blabel(b))
        a_.set_xlabel(fc["label"], fontsize=7)
        a_.set_title("A  %s" % fname, loc="left", fontsize=7)
        if k == 0:
            a_.set_ylabel("recommender mean log10 regret")
            a_.legend(frameon=False, fontsize=6.5)
    # B: single-deployment statistic vs the failure that follows
    thr = float(d.threshold.iloc[0])
    axB.axvspan(0, thr, color="#E6E6E6", lw=0, label="no flag (|z| ≤ %.2f, α = %.2g)" % (thr, d.alpha.iloc[0]))
    rng = np.random.default_rng(0)
    for b in mc["budgets"]:
        e = draws[draws.B == b]
        if e.empty:
            continue
        auc = disc.set_index("B").auc_abs_z.get(b, np.nan)
        axB.scatter(e.abs_z, e.model_log_regret + rng.uniform(-0.01, 0.01, len(e)), s=4, color=bcol[b], alpha=0.35,
                    lw=0)
        axB.plot([], [], "o", color=bcol[b], ms=3, label="B = %s (AUC %.2f)" % (_blabel(b), auc))
    axB.axhline(0.3, color="#D55E00", lw=0.8, ls="--", label="failure: log10 regret > 0.3")
    axB.set_xscale("symlog", linthresh=thr)
    axB.set_xlim(left=0)
    axB.set_xlabel("one deployment's echo statistic |z| (symlog)")
    axB.set_ylabel("recommender log10 regret on that task")
    axB.set_title("B  what one deployment sees vs the failure that follows", loc="left", fontsize=7)
    yl = axB.get_ylim()
    axB.set_ylim(yl[0], yl[1] + 0.75 * (yl[1] - yl[0]))     # headroom: the legend sits above the points
    axB.legend(frameon=False, fontsize=6, loc="upper right")
    # C: equal-budget gains and losses, every grid budget pooled, by discrepancy type
    grid = sorted(set(mc["budgets"]) | set(int(x) for x in d.B.unique()))
    q = p[p.B.isin(grid)].pivot_table(index=["type", "base_id", "task_id", "B"], columns="policy",
                                      values="mean_log10_risk").reset_index()
    pols = [x for x in ("DIAG-RESELECT", "RANDOM→reaction", "FIXED→ZNE-L", "FIXED→SBS-B", "NO-DIAG@B-d")
            if x in q.columns]
    colp = {"DIAG-RESELECT": "#009E73", "RANDOM→reaction": "#999999", "FIXED→ZNE-L": "#0072B2",
            "FIXED→SBS-B": "#CC79A7", "NO-DIAG@B-d": "#E69F00"}
    mkp = {"DIAG-RESELECT": "o", "RANDOM→reaction": "x", "FIXED→ZNE-L": "s", "FIXED→SBS-B": "D", "NO-DIAG@B-d": "v"}
    types = ["none"] + sorted(t for t in q.type.unique() if t != "none")
    C = []
    off = np.linspace(-0.3, 0.3, len(pols))
    for k, T in enumerate(types):
        e = q[q.type == T]
        for w, pol in zip(off, pols):
            dl = (e[pol] - e["TARGET@B"]).to_numpy()
            m, lo, hi = bootstrap({"v": dl}, e.base_id.to_numpy(), 1000, 0)[0]["v"]
            axC.errorbar([m], [k + w], xerr=[[m - lo], [hi - m]], fmt=mkp[pol], color=colp[pol], ms=3, capsize=1.5,
                         lw=1)
            C.append(dict(type=T, policy=pol, reference="TARGET@B", diff_log10=m, ci_lo=lo, ci_hi=hi,
                          base_circuits=int(e.base_id.nunique()), rows=int(len(e)),
                          flag_rate=float(d[d.type == T].flag_rate.mean())))
    axC.axvline(0, color="#333333", lw=0.8)
    fr = d.groupby("type").flag_rate.mean()
    axC.set_yticks(range(len(types)), ["%s (flag %.0f %%)" % (t, 100 * fr.get(t, np.nan)) for t in types])
    axC.invert_yaxis()
    axC.set_xlabel("Δ mean log10(MSE + 10⁻⁶) vs no diagnostic (all B shots on\nthe target; < 0: the reaction helps; equal "
                   "total budget)")
    axC.set_title("C  equal-budget gains and losses", loc="left", fontsize=7)
    lab = {"DIAG-RESELECT": "learned reselection\n(type held out)", "RANDOM→reaction": "same reaction,\nrandom trigger",
           "FIXED→ZNE-L": "flag → ZNE-L", "FIXED→SBS-B": "flag → SBS-B", "NO-DIAG@B-d": "diagnostic shots\nwithheld"}
    for pol in pols:
        axC.plot([], [], mkp[pol], color=colp[pol], ms=3, label=lab[pol])
    axC.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    G, S = int(p.base_id.nunique()), int(d.scenario.nunique())
    fp = float(fr.get("none", np.nan))
    cap = ("Model mismatch, diagnostics and reactions at equal total budget (%d base circuits × %d execution laws; "
           "the recommender, PEC's model and every allocation see only the calibration snapshot; injected parameters "
           "are hidden). (A) Mean log10 regret of the recommender's choice with all B shots on the target, against the "
           "injected strength in physical units, at three budgets; 95 %% bootstrap intervals over base circuits. (B) "
           "What one deployment observes: single draws of the echo statistic |z| (d(B) = %d … %d shots charged to B) "
           "against the recommender's log10 regret on that task and budget; AUC: how well one deployment's |z| ranks "
           "failures (log10 regret > 0.3) above non-failures. (C) Paired difference of mean log10(MSE + 10⁻⁶) to spending all "
           "B shots on the target without a diagnostic, every grid budget pooled, by discrepancy type: withholding "
           "the diagnostic's shots without reacting; reacting to a flag with ZNE-L or SBS-B (declared fallbacks); a "
           "learned reselection (per budget, the reaction that minimized the risk on the other discrepancy types — "
           "evaluated only on the held-out type); the learned reaction triggered at random with the same mean flag "
           "rate. Under 'none' the flag rate is the false-positive rate (%.1f %% at α = 0.05). Detecting a "
           "discrepancy does not identify its physical mechanism. %s" % (G, S, int(d.d.min()), int(d.d.max()),
                                                                          100 * fp, LABEL_SIM))
    Cd = pd.DataFrame(C)
    parts = []
    for pol in pols:
        e = Cd[Cd.policy == pol]
        tname = lambda t: "no mismatch" if t == "none" else t                      # noqa: E731
        helped = sorted(tname(r_.type) for r_ in e.itertuples() if r_.ci_hi < 0)
        hurt = sorted(tname(r_.type) for r_ in e.itertuples() if r_.ci_lo > 0)
        parts.append("%s helps (interval < 0) under: %s; hurts under: %s" % (
            lab[pol].replace("\n", " "), ", ".join(helped) or "no scenario", ", ".join(hurt) or "no scenario"))
    interp = ("Observation: false-positive rate %.1f %%; single-deployment AUC %s; %s. A diagnostic earns its cost "
              "only where reacting to it lowers the risk after its shots are charged and beats the same reaction "
              "triggered at random. Simulated robustness is not hardware transfer."
              % (100 * fp, ", ".join("%s: %.2f" % (_btext(r_.B), r_.auc_abs_z) for r_ in disc.itertuples()),
                 "; ".join(parts)))
    return _save(fig, out_dir, "fig06_mismatch_diagnostics",
                 dict(response=pd.DataFrame(A), draws=draws, discrimination=disc, reactions=react, equal_budget=Cd),
                 cap, interp, dict(mismatch=mc))


# ------------------------------------------------------------------ supplementary figures
def supp_target_precision(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Minimum evaluated budget at which each protocol meets each MSE target, per task: by the point estimate and
    uncertainty-qualified (upper 95 % bound MSE + 1.96 SE ≤ target); unattained targets are censored at the grid's
    end, never extrapolated."""
    import pandas as pd
    cfg = cfg or load_config()
    plt = _plt()
    lab = pd.read_csv(os.path.join(run_dir, "analysis", "labels.csv"), encoding="utf-8")
    lab = lab[lab.feasible]
    targets = cfg["mse_targets"]
    rows = []
    for (t, p), d in lab.groupby(["task_id", "protocol"]):
        d = d.sort_values("B")
        for eps in targets:
            for kind, v in (("point", d.mse_eval), ("upper 95 %", d.mse_eval + 1.96 * d.se_mse_eval)):
                ok = np.flatnonzero(v.to_numpy() <= eps)
                rows.append(dict(task_id=t, protocol=p, target=eps, kind=kind,
                                 B_min=float(d.B.iloc[ok[0]]) if len(ok) else math.inf, attained=bool(len(ok))))
    df = pd.DataFrame(rows)
    protos = [p for p in PROTOCOL_COLORS if p in set(df.protocol)]
    # the height grows with the portfolio (1.8.0: 17 candidates) so that the two rows of a protocol stay legible
    fig, ax = plt.subplots(1, len(targets), figsize=(7.4, max(2.8, 0.29 * len(protos) + 0.8)), sharey=True,
                           layout="constrained")
    summ = []
    fs = 6.5 if len(protos) <= 8 else 5.6
    for k, eps in enumerate(targets):
        for q, p in enumerate(protos):
            res = {}
            for kind in ("point", "upper 95 %"):
                e = df[(df.target == eps) & (df.protocol == p) & (df.kind == kind)]
                med = float(np.median(e.B_min)) if len(e) else math.inf
                frac = float(e.attained.mean()) if len(e) else 0.0
                res[kind] = (med, frac)
                summ.append(dict(target=eps, protocol=p, kind=kind, median_B_min=med, share_attained=frac,
                                 tasks=int(len(e))))
            if not any(np.isfinite(res[k_][0]) for k_ in res):      # neither attained: one line for both
                ax[k].text(1.3e3, q, "not reached (%.0f %% | %.0f %%)" % (100 * res["point"][1],
                                                                       100 * res["upper 95 %"][1]),
                           fontsize=fs, va="center", color="#777777")
                continue
            for kind, off, mk in (("point", -0.2, "o"), ("upper 95 %", 0.2, "s")):
                med, frac = res[kind]
                if np.isfinite(med):
                    ax[k].plot([med], [q + off], marker=mk, color=PROTOCOL_COLORS[p], ms=3.6,
                               mfc=PROTOCOL_COLORS[p] if kind == "point" else "white", ls="")
                    ax[k].text(med * 1.6, q + off, "%.0f %%" % (100 * frac), fontsize=fs, va="center",
                               color="#333333")
                else:
                    ax[k].text(1.3e3, q + off, "%s not reached (%.0f %%)" % ("upper" if off > 0 else "point",
                                                                         100 * frac),
                               fontsize=fs, va="center", color="#777777")
        ax[k].set_xscale("log")
        ax[k].set_xlim(8e2, 3e8)
        ax[k].set_title("MSE target %s" % _blabel(eps).replace("10", "10", 1), loc="left", fontsize=7)
        ax[k].set_xlabel("median minimal budget B")
    ax[0].set_yticks(range(len(protos)), protos)
    ax[0].set_ylim(len(protos) - 0.4, -0.6)
    hd = [plt.Line2D([], [], marker="o", color="#555555", ls=""),
          plt.Line2D([], [], marker="s", mfc="white", color="#555555", ls="")]
    hd.append(plt.Line2D([], [], ls=""))
    fig.legend(hd, ["point estimate", "upper 95 % bound (pointwise)",
                    "%: share of tasks attaining; not reached (point | upper)"], frameon=False, fontsize=6.5,
               loc="outside lower center", ncol=3)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    ntask = int(df.task_id.nunique())
    cap = ("Budget needed to meet an MSE target (%d development tasks, evaluation half): per task and protocol the "
           "smallest evaluated budget whose MSE point estimate (filled) or approximate pointwise upper limit "
           "MSE + 1.96 SE (open; SE matching the estimand, over training designs for the learning candidates (CDR, NIL; FR in 1.8.0) under fresh designs) "
           "is at most the target; the median over tasks is drawn with the share of tasks that attain it. The upper "
           "limit is pointwise at each budget: the first grid budget that passes it is not a guaranteed minimum budget "
           "(the grid is searched, and the intervals are not simultaneous). Infeasible protocol–budget pairs count as "
           "not attained at that budget. Unattained targets are censored at the grid's largest budget (3·10⁷): 'not "
           "reached' means the median task did not attain it; nothing is extrapolated. Descriptive (development "
           "data). %s" % (ntask, LABEL_SIM))
    return _save(fig, out_dir, "supp_target_precision", dict(budgets=df, summary=pd.DataFrame(summ)), cap,
                 "Uncertainty-qualified attainment needs more budget than point attainment; the gap is the price of "
                 "the (pointwise, approximate) confidence about the target.")


def _bias_variance_table(labels_dir: str, task_ids, protocols, half: int = 1):
    """Per (task, protocol, budget) of one label half: mean error, its standard error matching the estimand
    (design-level for learning protocols under fresh designs), the finite-sample corrected squared bias
    ē² − SE(ē)², the variance of one estimate and its between-design / within-design components (one-way ANOVA on the
    errors: the part of an estimate's variance that comes from drawing a training design)."""
    import pandas as pd
    from . import labels as LB
    from . import protocols as PR
    rows = []
    for tid in task_ids:
        rec, arr = LB.load_task(labels_dir, tid)
        if arr is None:
            continue
        plan = LB.plan_of(rec)
        for i, p in enumerate(rec["protocols"]):
            if p not in protocols:
                continue
            learning = PR.PORTFOLIO[p].learning
            for j, B in enumerate(rec["budgets"]):
                a = rec["allocations"][i * len(rec["budgets"]) + j]
                if not a["feasible"]:
                    continue
                x = arr["est"][half, i, j]
                e = np.where(np.isfinite(x), x, 0.0) - rec["ideal"]
                st = LB.design_stats(e, arr["design"][half, i, j])
                se = float(LB.risk_se(st, plan, learning))
                eb = float(e.mean())
                rows.append(dict(task_id=tid, family=rec["spec"]["family"], protocol=p, B=int(B), r=len(e),
                                 designs=st["designs"], mean_error=eb, se_mean_error=se,
                                 se_kind="design" if learning and plan.get("mode") == "fresh" else "trial",
                                 bias2_corrected=eb ** 2 - se ** 2,
                                 bias2_lo=max(abs(eb) - 1.96 * se, 0.0) ** 2, bias2_hi=(abs(eb) + 1.96 * se) ** 2,
                                 # 1.8.0 (finding F5): one fresh deployment's variance σ̂_W² + σ̂_D², not the pooled
                                 # sample variance (kept as var_pooled)
                                 var_one=LB.deployment_variance(e, arr["design"][half, i, j])["var_deploy"],
                                 var_pooled=float(e.var(ddof=1)), var_between_design=st["var_between"],
                                 var_within_design=st["var_within"], mse=float((e ** 2).mean())))
    return pd.DataFrame(rows)


def supp_bias_variance(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Bias² and variance against budget for the regime study's slices at s = 1 and selected protocols. The squared
    bias is the finite-sample corrected ē² − SE(ē)², with SE(ē) matching the estimand (over training designs for
    CDR and NIL under fresh designs); the variance of one estimate is split, for learning protocols, into its
    between-design part (drawing a training design) and the rest."""
    import pandas as pd
    cfg = cfg or load_config()
    plt = _plt()
    bv = cfg["bias_variance"]
    cells = pd.read_csv(os.path.join(run_dir, "regimes", "analysis", "regime_cells.csv"), encoding="utf-8")
    keep = sorted(cells[(cells.noise_scale == bv["noise_scale"]) & (cells.observable == "default") &
                        (cells.layout == "best")].task_id.unique())
    lab = _bias_variance_table(os.path.join(run_dir, "regimes"), keep, set(bv["protocols"]))
    lab["bias2_resolved"] = lab.bias2_corrected > 0
    fams = [f for f in cfg["regime"]["slices"] if f in set(lab.family)]
    fig, ax = plt.subplots(2, len(fams), figsize=(7.4, 4.4), sharex=True, layout="constrained", squeeze=False)
    floor = 1e-10
    handles = []
    for k, f in enumerate(fams):
        for p in bv["protocols"]:
            e = lab[(lab.family == f) & (lab.protocol == p)].sort_values("B")
            if e.empty:
                continue
            c, mk = PROTOCOL_COLORS[p], PROTOCOL_MARKERS[p]
            yb = np.where(e.bias2_resolved, e.bias2_corrected, np.nan)      # the line breaks at unresolved points
            h, = ax[0, k].plot(e.B, yb, mk + "-", color=c, ms=3, lw=1, label=p)
            ax[0, k].fill_between(e.B, np.maximum(e.bias2_lo, floor), e.bias2_hi, color=c, alpha=0.10, lw=0)
            neg = e[~e.bias2_resolved]
            ax[0, k].plot(neg.B, np.full(len(neg), floor), mk, color=c, mfc="white", ms=3.5)
            ax[1, k].plot(e.B, e.var_one, mk + "-", color=c, ms=3, lw=1)
            if e.var_between_design.notna().any():
                vb = np.where(e.var_between_design > 0, e.var_between_design, np.nan)
                ax[1, k].plot(e.B, vb, mk + "--", color=c, ms=2.5, lw=0.9, mfc="white")
            if k == 0:
                handles.append(h)
        ax[0, k].set_title("%s-4, s = %g" % (f, bv["noise_scale"]), loc="left", fontsize=7)
        for a_ in ax[:, k]:
            a_.set_xscale("log")
            a_.set_yscale("log")
        ax[1, k].set_xlabel("total budget B")
    ax[0, 0].set_ylabel("bias² (finite-sample corrected)")
    ax[1, 0].set_ylabel("variance of one estimate")
    handles.append(ax[0, 0].plot([], [], "o", color="#555555", mfc="white", ms=3.5,
                                 label="bias² estimate ≤ 0 (unresolved), drawn at 10⁻¹⁰")[0])
    handles.append(ax[1, 0].plot([], [], "--", color="#555555", lw=0.9, label="between-design part (CDR, NIL)")[0])
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    rr = sorted(set(int(x) for x in lab.r.dropna()))
    kk = sorted(set(int(x) for x in lab[lab.designs > 0].designs)) if (lab.designs > 0).any() else []
    cap = ("Bias–variance decomposition and error floors for the three regime slices at s = %g (evaluation half, "
           "r = %s trials per cell) and %d protocols. Top: squared bias corrected for finite samples, ē² − SE(ē)², "
           "where SE(ē) matches the estimand — over trials for protocols without a training design, over the "
           "%s fresh training designs of a half for the learning candidates (CDR, NIL; FR in 1.8.0) (cluster-robust); bands: squares of the 95 %% "
           "interval of the mean error, lower end clipped at 0; estimates ≤ 0 (a bias below the resolution of the "
           "data) are drawn as open markers at 10⁻¹⁰ and the line is broken there. Bottom: variance of one estimate "
           "(solid) and, for the learning protocols, its between-design component Var_D(E[X̂|D]) from a one-way "
           "analysis of variance (dashed; omitted where estimated as 0). MSE = bias² + variance of one estimate; the "
           "between-design part does not fall with more target shots. %s"
           % (bv["noise_scale"], "/".join(map(str, rr)) or "n/a", len(bv["protocols"]),
              "/".join(map(str, kk)) or "K", LABEL_SIM))
    bmax = lab.B.max()
    top = lab[lab.B == bmax]
    floors = ["%s: %s" % (f, ", ".join(sorted(top[(top.family == f) & (top.bias2_corrected > top.var_one)]
                                              .protocol)) or "none") for f in fams]
    lr = top[top.var_between_design.notna()]
    shares = ", ".join("%s %s %.0f %%" % (r_.family, r_.protocol, 100 * r_.var_between_design / r_.var_one)
                       for _, r_ in lr.iterrows() if r_.var_one > 0)
    return _save(fig, out_dir, "supp_bias_variance", dict(curves=lab), cap,
                 "Observation: at B = %s the corrected squared bias exceeds the variance of one estimate for — %s. "
                 "Between-design share of the variance at that budget: %s. A bias² that stays flat while the "
                 "variance falls is an error floor: more shots do not remove it; a between-design share that grows "
                 "with B is variance that only more (or better) training designs remove."
                 % (_btext(bmax), "; ".join(floors), shares or "n/a"))


def _runner_up_table(pr):
    """Per held-out row (task, budget): the predicted best and runner-up protocols, their predicted log-risk
    difference and the evaluated difference of the same two protocols (evaluation half) with its approximate SE
    (independent labels; delta method)."""
    import pandas as pd
    out = []
    for (t, B), d in pr.groupby(["task_id", "B"]):
        d = d.dropna(subset=["predicted", "observed"]).sort_values("predicted")
        if len(d) < 2:
            continue
        a, b = d.iloc[0], d.iloc[1]
        se = float(math.sqrt(np.nan_to_num(a.observed_se) ** 2 + np.nan_to_num(b.observed_se) ** 2))
        dobs = float(b.observed - a.observed)
        out.append(dict(task_id=t, B=int(B), family=a.family, best=a.protocol, runner_up=b.protocol,
                        pred_diff=float(b.predicted - a.predicted), obs_diff=dobs, obs_diff_se=se,
                        ordered=bool(dobs > 0), resolved=bool(abs(dobs) > 1.96 * se)))
    return pd.DataFrame(out)


def supp_prediction_quality(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Predicted against evaluated log10(MSE + 10⁻⁶) per candidate (facets), with the labels' standard errors, and the
    quantity closest to the selection problem: the predicted against the evaluated risk difference between the
    predicted best and runner-up protocols."""
    import pandas as pd
    plt = _plt()
    pr = pd.read_csv(os.path.join(run_dir, "analysis", "prediction.csv"), encoding="utf-8")
    protos = [p for p in PROTOCOL_COLORS if p in set(pr.protocol)]
    # every candidate gets a facet (1.8.0: 17 candidates on a 6-column grid; 1.7.0's 8 on the original 4 × 2); the
    # facets share their axes, labelled once for the block
    ncol = 4 if len(protos) <= 8 else 6
    nrow = max(2, int(math.ceil(len(protos) / ncol)))
    fig = plt.figure(figsize=(7.4, 3.9 if ncol == 4 else 1.45 * nrow + 1.0), layout="constrained")
    sfL, sfR = fig.subfigures(1, 2, width_ratios=[2.7, 1.25] if ncol == 4 else [3.3, 1.3], wspace=0.02)
    ax = np.atleast_2d(sfL.subplots(nrow, ncol, sharex=True, sharey=True))
    axR = sfR.subplots(1, 1)
    lo = float(np.nanmin(pr[["observed", "predicted"]].values))
    hi = 1.0
    stats = []
    for k, p in enumerate(protos):
        a_ = ax.flat[k]
        d = pr[pr.protocol == p]
        inr = d[d.observed <= hi]
        a_.errorbar(inr.observed, inr.predicted, xerr=np.minimum(1.96 * inr.observed_se.fillna(0), 3), fmt="o",
                    ms=1.4, color=PROTOCOL_COLORS[p], alpha=0.35, elinewidth=0.3, capsize=0)
        a_.plot([lo, hi], [lo, hi], color="#333333", lw=0.7)
        rm = float(np.sqrt(np.mean((d.predicted - d.observed) ** 2)))
        out = int((d.observed > hi).sum())
        stats.append(dict(protocol=p, rows=int(len(d)), rmse_log=rm, beyond_range=out,
                          median_label_se=float(np.nanmedian(d.observed_se))))
        a_.set_title("%s\nRMSE %.2f%s" % (p, rm, ("\n+%d off axis" % out) if out else ""), fontsize=6.2,
                     loc="left")
        a_.set_xlim(lo - 0.3, hi + 0.3)
        a_.set_ylim(lo - 0.3, hi + 0.3)
    for k in range(len(protos), ax.size):
        ax.flat[k].set_axis_off()
        up = k - ncol                          # the facet above an empty slot is the lowest of its column
        if 0 <= up < len(protos):
            ax.flat[up].xaxis.set_tick_params(labelbottom=True)
    sfL.supxlabel("evaluated log10(MSE + 10⁻⁶) (evaluation half)", fontsize=6.8)
    sfL.supylabel("predicted log10(MSE + 10⁻⁶) (held-out fold)", fontsize=6.8)
    ru = _runner_up_table(pr)
    if len(ru):
        res = ru[ru.resolved]
        unr = ru[~ru.resolved]
        axR.errorbar(unr.pred_diff, unr.obs_diff, yerr=np.minimum(1.96 * unr.obs_diff_se, 3), fmt="o", ms=1.6,
                     color="#999999", alpha=0.5, elinewidth=0.3, capsize=0, label="evaluated difference unresolved")
        axR.errorbar(res.pred_diff, res.obs_diff, yerr=np.minimum(1.96 * res.obs_diff_se, 3), fmt="o", ms=1.8,
                     color="#0072B2", alpha=0.7, elinewidth=0.3, capsize=0, label="resolved (|Δ| > 1.96 SE)")
        axR.axhline(0, color="#333333", lw=0.7)
        axR.set_xlabel("predicted Δ log10(risk + 10⁻⁶)\n(runner-up − best)")
        axR.set_ylabel("evaluated Δ log10(MSE + 10⁻⁶) of the same two")
        axR.legend(frameon=True, framealpha=0.9, edgecolor="#FFFFFF", fontsize=6.5, loc="upper right")
        axR.set_title("best vs runner-up: order right %.0f %%\n(resolved rows: %.0f %% of %d)"
                      % (100 * ru.ordered.mean(), 100 * res.ordered.mean() if len(res) else float("nan"), len(res)),
                      fontsize=6.5, loc="left")
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    nb = pr.task_id.map(lambda t: t).nunique()
    cap = ("Prediction quality of the deployed recommender on held-out base circuits (group 5-fold; %d tasks). Left: "
           "predicted against evaluated log10(MSE + 10⁻⁶) per candidate, with ±1.96 standard errors of the label "
           "(delta method on the SE matching the estimand — over training designs for the learning candidates (CDR, NIL; FR in 1.8.0) under fresh "
           "designs; capped at 3 decades for display). Points beyond the axis (evaluated log10(MSE + 10⁻⁶) > 1: divergent "
           "exponential-ZNE labels) are counted in the titles and kept in the source data; the RMSE includes them. "
           "Right: per row, the predicted log-risk difference between the predicted best and runner-up protocols "
           "against the evaluated difference of the same two protocols (evaluation half; ±1.96 SE, the two labels "
           "treated as independent): a selection is right when the evaluated difference is positive. Grey: the "
           "evaluated difference is within 1.96 SE of zero (the labels cannot order the pair). Pointwise "
           "intervals; descriptive. The 10⁻⁶ floor compresses differences between very small risks. %s"
           % (nb, LABEL_SIM))
    interp = ("Good choices need correct orderings within a task, not accurate values; errors shared across "
              "candidates cancel. The predicted best is evaluated better than the predicted runner-up in %.0f %% of "
              "rows (%.0f %% of the %d rows whose evaluated difference is resolved); correlation of predicted and "
              "evaluated differences %.2f." % (100 * ru.ordered.mean() if len(ru) else float("nan"),
                                                100 * ru[ru.resolved].ordered.mean() if ru.resolved.any() else
                                                float("nan"), int(ru.resolved.sum()) if len(ru) else 0,
                                                float(np.corrcoef(ru.pred_diff, ru.obs_diff)[0, 1]) if len(ru) > 2
                                                else float("nan")))
    return _save(fig, out_dir, "supp_prediction_quality", dict(points=pr, stats=pd.DataFrame(stats), runner_up=ru),
                 cap, interp)


def supp_dataset_coverage(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Feature distributions of the development and final-test tasks (deployable features), the oracle ideal value
    separately, the coverage of noise components (gate, idle and readout parts of the snapshot's fault weight, with
    the controlled noise-composition settings for comparison), and counts of tasks, independent base circuits,
    learning designs and trials."""
    import pandas as pd
    plt = _plt()
    from . import labels as LB
    recs = []
    parts = [("dev", "development"), ("final", "final test (historical)"), ("confirm", "confirmation"),
             ("noisecomp", "noise-composition study")]
    counts = []
    for part, label in parts:
        tdir = os.path.join(run_dir, part)
        if not os.path.isdir(tdir):
            continue
        st = dict(part=label, tasks=0, failed_tasks=0, base_circuits=set(), designs_per_half=set(), r=set())
        for tid in LB.task_ids(tdir):
            rec = TX.read_json(os.path.join(tdir, "tasks", tid + ".json"))
            if rec.get("status") != "ok":
                st["failed_tasks"] += 1
                continue
            st["tasks"] += 1
            st["base_circuits"].add(rec["base_id"])
            st["r"].add(int(rec.get("R", 0)))
            plan = LB.plan_of(rec)
            if plan.get("mode") == "fresh":
                st["designs_per_half"].add(int(plan.get("per_half") or plan.get("count") or 0))
            f = rec.get("features") or {}
            lam = float(f.get("lam_lc") or np.nan)
            recs.append(dict(part=label, task_id=tid, base_id=rec["base_id"], family=rec["spec"]["family"],
                             n=rec["spec"]["n"], device=rec["spec"]["device"], two_qubit=f.get("two_qubit"),
                             depth=f.get("depth"), lam_lc=f.get("lam_lc"), obs_weight=f.get("obs_weight"),
                             lc_qubits=f.get("lc_qubits"), lam_lc_gate=f.get("lam_lc_gate"),
                             lam_lc_idle=f.get("lam_lc_idle"), lam_ro=f.get("lam_ro"),
                             gate_share=float(f.get("lam_lc_gate") or np.nan) / lam if lam and lam > 0 else np.nan,
                             abs_ideal=abs(float(rec["ideal"]))))
        counts.append(dict(part=label, tasks=st["tasks"], failed_tasks=st["failed_tasks"],
                           base_circuits=len(st["base_circuits"]),
                           designs_per_half="/".join(map(str, sorted(st["designs_per_half"]))) or "fixed",
                           trials_per_half="/".join(map(str, sorted(st["r"])))))
    df = pd.DataFrame(recs)
    cnt = pd.DataFrame(counts)
    feats = [("n", "width n"), ("two_qubit", "two-qubit gates"), ("depth", "compiled depth"),
             ("lam_lc", "light-cone fault weight λ_LC"), ("obs_weight", "observable weight"),
             ("abs_ideal", "|μ| — oracle (labels only)")]
    fig, ax = plt.subplots(2, 4, figsize=(7.4, 4.4), layout="constrained")
    col = {"development": "#0072B2", "final test (historical)": "#D55E00", "confirmation": "#CC79A7",
           "noise-composition study": "#009E73"}
    discrete = ("n", "obs_weight")
    main = df[df.part != "noise-composition study"]
    for a_, (k, title) in zip(ax.flat[:6], feats):
        allv = main[k].dropna().astype(float)
        for part, e in main.groupby("part"):
            v = e[k].dropna().astype(float)
            if k in discrete:                       # one bin per integer value; share of tasks
                bins = np.arange(allv.min() - 0.5, allv.max() + 1.5)
                a_.hist(v, bins=bins, weights=np.full(len(v), 1.0 / max(len(v), 1)), histtype="step", lw=1.3,
                        color=col[part], label="%s (%d tasks)" % (part, len(v)))
            elif k == "abs_ideal":                  # fixed bins on [0, 1]; share of tasks
                a_.hist(v, bins=np.linspace(0, 1, 21), weights=np.full(len(v), 1.0 / max(len(v), 1)),
                        histtype="step", lw=1.3, color=col[part])
            else:                                   # quantile bins of the pooled data; density (unequal widths)
                bins = np.unique(np.quantile(allv, np.linspace(0, 1, 13)))
                a_.hist(v, bins=bins if len(bins) >= 3 else 10, density=True, histtype="step", lw=1.3,
                        color=col[part])
        a_.set_ylabel("share of tasks" if (k in discrete or k == "abs_ideal") else "density", fontsize=6.5)
        a_.set_title(title, fontsize=7, loc="left", color="#B2182B" if k == "abs_ideal" else "#000000")
        if k == "abs_ideal":
            a_.axvline(0.1, color="#999999", lw=0.8, ls=":")
    # noise components
    a7 = ax.flat[6]
    for part, e in df.groupby("part"):
        e = e[np.isfinite(e.gate_share.astype(float)) & (e.lam_ro.astype(float) > 0)]
        a7.scatter(e.gate_share, e.lam_ro, s=6 if part != "noise-composition study" else 12,
                   marker="o" if part != "noise-composition study" else "D", color=col[part], alpha=0.6, lw=0,
                   label=part)
    a7.set_yscale("log")
    a7.set_xlabel("gate share of λ_LC (rest: idle)", fontsize=6.5)
    a7.set_ylabel("readout weight λ_RO", fontsize=6.5)
    a7.set_title("noise components", fontsize=7, loc="left")
    a8 = ax.flat[7]
    a8.set_axis_off()
    lines = ["%s:\n  %d tasks (%d failed)\n  %d base circuits\n  %s designs/half, r = %s/half"
             % (r_.part, r_.tasks, r_.failed_tasks, r_.base_circuits, r_.designs_per_half, r_.trials_per_half)
             for _, r_ in cnt.iterrows()]
    t8 = a8.text(0.0, 1.0, "counts\n" + "\n".join(lines), va="top", ha="left", fontsize=6,
                 transform=a8.transAxes)
    t8.set_in_layout(False)                          # the text must not widen its column (it fits the panel)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], color=col[p_], lw=1.3, label=p_) for p_ in col if p_ in set(main.part)]
    if "noise-composition study" in set(df.part):
        hd.append(Line2D([], [], ls="", marker="D", ms=4, color=col["noise-composition study"], alpha=0.6,
                         label="noise-composition study (noise panel only)"))
    fig.legend(handles=hd, loc="outside lower center", ncol=len(hd), frameon=False, fontsize=6.5)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    sup = df.groupby(["part", "family", "n"]).agg(tasks=("task_id", "nunique"),
                                                  base_circuits=("base_id", "nunique")).reset_index()
    near0 = float((main.abs_ideal < 0.1).mean()) if len(main) else float("nan")
    cap = ("Dataset coverage: distributions of deployable features for the development, historical final-test and "
           "confirmation tasks (step "
           "histograms: share of tasks for the integer-valued width and observable weight, density over quantile "
           "bins otherwise), and — separately, in red, because it is oracle information used only for labels — the "
           "magnitude of the ideal value |μ|; %.0f %% of these tasks have |μ| < 0.1 (dotted line). Noise components: "
           "the gate share of the light-cone fault weight against the readout weight for every task, with the "
           "controlled settings of the noise-composition study (diamonds) for comparison — the development devices "
           "vary the components only jointly, so the controlled study reaches compositions the development set does "
           "not. Counts: tasks (failed label tasks listed separately), independent base circuits (the unit of every "
           "bootstrap), fresh learning designs per label half for the learning candidates (CDR, NIL; FR in 1.8.0), and trials per half. Support by "
           "family and width is in the source data. %s" % (100 * near0, LABEL_SIM))
    med = main.groupby("part")[["n", "two_qubit", "depth", "lam_lc", "obs_weight"]].median()
    txt = "; ".join("%s: median n %g, two-qubit gates %g, depth %g, λ_LC %.3f, weight %g" % (
        part, r_.n, r_.two_qubit, r_.depth, r_.lam_lc, r_.obs_weight) for part, r_ in med.iterrows())
    nb = df.groupby("part").base_id.nunique().to_dict()
    return _save(fig, out_dir, "supp_dataset_coverage", dict(tasks=df, support=sup, counts=cnt), cap,
                 "Observation: %s; base circuits: %s. Similar medians do not establish that the final test covers "
                 "the same regions of feature space; the histograms and the support table show where it is thin."
                 % (txt, ", ".join("%s %d" % kv for kv in nb.items())))


def supp_failures_costs(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Infeasibility and method errors per protocol and budget, the union failure rate of trials (share of trials
    with at least one fallback or no estimate — flags overlap and are never summed) with its no-estimate part,
    software failures of label tasks, and the shot-budget decomposition of every protocol."""
    import pandas as pd
    plt = _plt()
    adir = os.path.join(run_dir, "analysis")
    lab = pd.read_csv(os.path.join(adir, "labels.csv"), encoding="utf-8")
    if "status" not in lab.columns:
        lab["status"] = np.where(lab.feasible, "ok", "infeasible")
    st = (lab.groupby(["protocol", "B"]).status.value_counts(normalize=True).unstack(fill_value=0.0)
          .reset_index())
    for c in ("infeasible", "method-error"):
        if c not in st.columns:
            st[c] = 0.0
    fl = lab[lab.feasible].copy()
    if "fail_any_eval" not in fl.columns:          # 1.6.0 tables: the union is not recoverable from the rates
        raise ValueError("labels.csv has no fail_any_eval (union failure rate); re-run the analysis with 1.7.0")
    fr = fl.groupby(["protocol", "B"]).agg(fail_any=("fail_any_eval", "mean"),
                                          nonresponse=("nonresponse_eval", "mean")).reset_index()
    lf_p = os.path.join(adir, "label_failures.csv")
    lf = pd.read_csv(lf_p, encoding="utf-8") if os.path.exists(lf_p) else pd.DataFrame(columns=["task_id", "status"])
    roles = [c for c in ("shots_calibration", "shots_training", "shots_diagnostics", "shots_selection",
                         "shots_execution") if c in fl.columns]
    B0 = 100000
    dec = fl[fl.B == B0].groupby("protocol")[roles].median().reindex([p for p in PROTOCOL_COLORS
                                                                      if p in set(fl.protocol)])
    dec = dec.div(B0)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0), layout="constrained")
    for p in PROTOCOL_COLORS:
        e = st[st.protocol == p].sort_values("B")
        if not e.empty:
            ax[0].plot(e.B, e.infeasible, marker=PROTOCOL_MARKERS[p], ms=3, lw=1.1, color=PROTOCOL_COLORS[p],
                       label=p)
            me = e[e["method-error"] > 0]
            ax[0].plot(me.B, me["method-error"], marker="x", ms=4, ls="", color=PROTOCOL_COLORS[p])
        e = fr[fr.protocol == p].sort_values("B")
        if not e.empty:
            ax[1].plot(e.B, e.fail_any, marker=PROTOCOL_MARKERS[p], ms=3, lw=1.1, color=PROTOCOL_COLORS[p])
            nr = e[e.nonresponse > 0]
            ax[1].plot(nr.B, nr.nonresponse, marker=PROTOCOL_MARKERS[p], ms=3, lw=0.8, ls=":", mfc="white",
                       color=PROTOCOL_COLORS[p])
    ax[0].plot([], [], "x", color="#555555", label="method error")
    for a_, t in zip(ax[:2], ("(a) share of tasks infeasible", "(b) share of trials with ≥ 1 fallback")):
        a_.set_xscale("log")
        a_.set_xlabel("total budget B")
        a_.set_title(t, loc="left", fontsize=7)
    ax[1].plot([], [], ":", marker="o", mfc="white", color="#555555", lw=0.8, label="of which: no estimate")
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    left = np.zeros(len(dec))
    rc = {"shots_calibration": "#E69F00", "shots_training": "#009E73", "shots_diagnostics": "#CC79A7",
          "shots_selection": "#999999", "shots_execution": "#0072B2"}
    for r_ in roles:
        v = dec[r_].fillna(0).to_numpy()
        ax[2].barh(range(len(dec)), v, left=left, color=rc[r_], height=0.6, label=r_.replace("shots_", ""))
        left += v
    ax[2].set_yticks(range(len(dec)), dec.index)
    ax[2].set_xlim(0, 1.0)
    ax[2].set_xlabel("share of B = 10⁵ (median over tasks)")
    ax[2].set_title("(c) shot-budget decomposition", loc="left", fontsize=7)
    ax[2].set_axisbelow(True)
    ax[2].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="outside lower center", ncol=9, frameon=False, fontsize=6.5)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    nsoft = int(len(lf))
    soft = ", ".join("%s %d" % kv for kv in lf.status.value_counts().items()) if nsoft else "none"
    cap = ("Failures and costs (development labels, %d tasks). (a) Share of tasks on which a protocol is infeasible "
           "at B (its minimum calibration or training shots do not fit, or it has nothing to train on; a declared "
           "property of the protocol at that budget) and, crosses, the share whose preparation raised an error "
           "(method error, recorded with its message). (b) Share of trials with at least one declared fallback "
           "(singular calibration, failed exponential fit, CDR or NIL training fallback) or no estimate — the union "
           "over the overlapping flags, never their sum; dotted, open: the no-estimate part. Every such trial is in "
           "the protocol's risk. Software failures of whole label tasks (error, timeout, memory) are not protocol "
           "outcomes; they are listed in label_failures.csv (%s) and excluded from the dataset. (c) Ledger of each "
           "protocol at B = 10⁵ by role (integers summing to B in every allocation; shares are medians over tasks). "
           "%s" % (int(lab.task_id.nunique()), soft, LABEL_SIM))
    return _save(fig, out_dir, "supp_failures_costs",
                 dict(status=st, fallbacks=fr, software_failures=lf, ledger=dec.reset_index()), cap,
                 "Infeasibility, fallback and software failure are different events: the first is a property of the "
                 "protocol at a budget, the second part of its risk, the third an execution problem of the study "
                 "(%d label tasks)." % nsoft)


def supp_representative(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Per-trial outcomes of four held-out rows chosen by documented rules at a preselected budget: the deployed
    recommender's most successful, typical and worst row (lowest, median, highest log10 regret) and the abstained
    row with the median margin (nested evaluation's acceptance when available). Learning protocols' trials are drawn
    by training design (one column of points per design, design mean marked) to show multimodality."""
    import pandas as pd
    from . import labels as LB
    from . import protocols as PR
    cfg = cfg or load_config()
    plt = _plt()
    b0 = cfg["representative"]["budget"]
    adir = os.path.join(run_dir, "analysis")
    dep = pd.read_csv(os.path.join(adir, "deployed_rows.csv"), encoding="utf-8")
    nested_p = os.path.join(adir, "nested_rows.csv")
    if os.path.exists(nested_p):
        nr = pd.read_csv(nested_p, encoding="utf-8")
        acc = nr[["task_id", "B", "accepted"]].drop_duplicates(["task_id", "B"])
        acc_src = "nested evaluation (outer-fold threshold chosen on inner folds)"
    else:
        sel = pd.read_csv(os.path.join(adir, "selective_rows.csv"), encoding="utf-8")
        acc = sel[["task_id", "B", "accepted_crossfit"]].rename(columns={"accepted_crossfit": "accepted"})
        acc_src = "cross-fitted threshold"
    dep = dep.merge(acc, on=["task_id", "B"], how="left")
    dep["accepted"] = dep.accepted.fillna(True).astype(bool)
    b0 = min(dep.B.unique(), key=lambda b: abs(math.log10(b) - math.log10(b0)))      # nearest evaluated budget
    d = dep[dep.B == b0].sort_values("log_regret").reset_index(drop=True)
    pick = [("successful", d.iloc[0]), ("typical", d.iloc[len(d) // 2]), ("failed", d.iloc[-1])]
    ab = d[~d.accepted].sort_values("margin").reset_index(drop=True)
    if (ab.choice != ab.fallback).any():                 # an abstention that changes the deployed protocol
        ab = ab[ab.choice != ab.fallback].reset_index(drop=True)
    if len(ab):
        pick.append(("abstained", ab.iloc[len(ab) // 2]))
    fig, ax = plt.subplots(1, len(pick), figsize=(7.4, 3.3), layout="constrained")
    rows = []
    shade = ("#333333", "#9A9A9A")
    for a_, (kind, r_) in zip(np.atleast_1d(ax), pick):
        rec, arr = LB.load_task(os.path.join(run_dir, "dev"), r_.task_id)
        j = rec["budgets"].index(int(r_.B))
        marks, rmse, errs = [], {}, {}
        for i, p in enumerate(rec["protocols"]):
            x = arr["est"][1, i, j]
            if not np.isfinite(x).any():
                continue
            e = np.where(np.isfinite(x), x, 0.0) - rec["ideal"]
            dz = arr["design"][1, i, j]
            rmse[p] = float(np.sqrt(np.mean(e ** 2)))
            errs[i] = (p, e)
            if PR.PORTFOLIO[p].learning and (dz >= 0).any():
                ks = np.unique(dz[dz >= 0])
                xs = i - 0.35 + 0.7 * (np.searchsorted(ks, dz) + 0.5) / len(ks)
                for q, k in enumerate(ks):
                    m = dz == k
                    a_.plot(xs[m], e[m], ".", ms=1.6, color=shade[q % 2], alpha=0.8)
                    a_.plot([xs[m][0] - 0.03, xs[m][0] + 0.03], [e[m].mean()] * 2, color=PROTOCOL_COLORS[p], lw=1.4)
            else:
                a_.boxplot(e, positions=[i], widths=0.6, showfliers=True, patch_artist=True,
                           boxprops=dict(facecolor=PROTOCOL_COLORS.get(p, "#999999"), lw=0.5),
                           medianprops=dict(color="#000000", lw=0.8),
                           flierprops=dict(marker=".", ms=1.5, mfc="#333333", mec="none", alpha=0.6))
            if p == r_.choice:
                marks.append((i, "#000000"))
            if p == r_.fallback and kind == "abstained" and p != r_.choice:
                marks.append((i, "white"))
            rows += [dict(kind=kind, task_id=r_.task_id, protocol=p, design=int(dd), error=float(v))
                     for v, dd in zip(e, dz)]
        a_.axhline(0, color="#999999", lw=0.8)
        # 1.8.0: one candidate with errors orders of magnitude larger than the rest (FR's divergent fits) would
        # flatten every other box; the axis covers the candidates within 10× the median spread (and the choice),
        # and the points beyond it are counted at the axis edge — nothing is dropped from the source data
        if errs:
            spread = {i: float(np.quantile(np.abs(e), 0.99)) for i, (p, e) in errs.items()}
            ref = float(np.median(list(spread.values())))
            keep = [i for i in errs if spread[i] <= 10 * max(ref, 1e-12) or errs[i][0] == r_.choice]
            lo_ = min(float(np.quantile(errs[i][1], 0.002)) for i in keep)
            hi_ = max(float(np.quantile(errs[i][1], 0.998)) for i in keep)
            pad = 0.08 * (hi_ - lo_ if hi_ > lo_ else 1.0)
            lo_, hi_ = min(lo_, 0.0) - pad, max(hi_, 0.0) + pad
            a_.set_ylim(lo_, hi_)
            for i, (p, e) in errs.items():
                nup, ndn = int((e > hi_).sum()), int((e < lo_).sum())
                if nup:
                    a_.text(i, hi_, "↑%d" % nup, ha="center", va="top", fontsize=5.2, color="#555555")
                if ndn:
                    a_.text(i, lo_, "↓%d" % ndn, ha="center", va="bottom", fontsize=5.2, color="#555555")
        top = a_.get_ylim()[1]
        for i, fc in marks:                                  # above the data, at the top of the axis
            a_.plot(i, top, marker="v", mfc=fc, mec="#000000", ms=4, clip_on=False, ls="")
        a_.set_xticks(range(len(rec["protocols"])), rec["protocols"], rotation=90,
                      fontsize=6.5 if len(rec["protocols"]) <= 8 else 5.8)
        sp = rec["spec"]
        fb_txt = (" → deployed %s" % r_.fallback) if kind == "abstained" else ""
        a_.set_title("%s: %s-%d %s\nmodel chose %s%s\nlog10 regret %.2f, z = %.2f\nRMSE of the choice %.2g"
                     % (kind, sp["family"], sp["n"], sp["observable"], r_.choice, fb_txt, r_.log_regret, r_.margin,
                        rmse.get(r_.choice, float("nan"))), fontsize=6, loc="left")
    np.atleast_1d(ax)[0].set_ylabel("trial error μ̂ − μ")
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    cap = ("Representative outcomes of the deployed recommender at B = %s on held-out base circuits, chosen by "
           "documented rules: the rows with the lowest, the median and the highest log10 regret ratio, and among the "
           "rows where the %s abstains (preferring rows where abstaining changes the deployed protocol) the one with "
           "the median margin z. Protocols without a training design: boxes "
           "of per-trial errors (evaluation half; quartiles, whiskers 1.5 IQR, dots beyond them — the rare large "
           "errors that dominate an MSE). CDR and NIL: every trial as a point, one column per fresh training design "
           "(alternating grey), the design's mean error as a coloured tick — separated clusters are designs, not "
           "shot noise. Filled triangle: the recommender's choice (its RMSE in the title); open triangle: the "
           "fallback used on abstention. The axis covers the candidates whose 99th-percentile error is within 10× "
           "the panel's median (and the choice); arrows count trials beyond it. Illustrative cases selected by "
           "rule, not a random sample. %s"
           % (_btext(b0), acc_src, LABEL_SIM))
    return _save(fig, out_dir, "supp_representative", dict(trials=pd.DataFrame(rows),
                                                           rows=pd.DataFrame([r for _, r in pick])), cap,
                 "Spread is variance, offset from zero is bias, and for the learning protocols the spread between "
                 "design means is the design component of the variance; all are part of each protocol's risk.")


def supp_runtime(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Classical cost of a recommendation against width (``python -m qemx.selection profile``), separated into
    compilation, deployable features, simulator-only augmentation and inference, with benchmark-label generation for
    comparison."""
    import pandas as pd
    plt = _plt()
    rt = pd.read_csv(os.path.join(run_dir, "analysis", "runtime.csv"), encoding="utf-8")
    rej = rt[rt.kind == "rejection"].copy() if "kind" in rt.columns else rt.iloc[0:0]   # 1.8.0 early-rejection probes
    if "kind" in rt.columns:
        rt = rt[rt.kind == "width"].copy()
    comps = [("compile_s", "#999999", "s", "compilation"),
             ("deployable_s", "#0072B2", "^", "deployable features (no simulation)"),
             ("simulated_s", "#D55E00", "o", "simulator-only augmentation (exact simulation)"),
             ("inference_s", "#009E73", "D", "model inference"),
             ("preflight_structural_s", "#E69F00", "v", "structural preflight, every candidate (1.8.0)"),
             ("preflight_design_s", "#CC79A7", "P", "design preflight, learning candidates (1.8.0)"),
             ("recommend_s", "#56B4E9", "*", "whole recommend() call, deployable bundle (1.8.0)"),
             ("label_s", "#000000", "x", "benchmark-label generation, one task (study cost)")]
    if "deployable_s" not in rt.columns:              # a 1.6.0 table: the old components, labelled as such
        raise ValueError("runtime.csv is from 1.6.0 (no separated components); re-run python -m qemx.selection "
                         "profile")
    cols = [c for c, *_ in comps if c in rt.columns]
    g = rt.groupby("n")[cols + ["peak_rss_mb"]].median().reset_index()
    if len(rej):
        fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.4, 3.8), layout="constrained", width_ratios=[1.35, 1])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(3.9, 3.8), layout="constrained")
    for col, c, mk, lab in comps:
        if col not in g.columns or not g[col].notna().any():
            continue
        ax.plot(g.n, g[col], mk + ("--" if col == "label_s" else "-"), color=c, ms=3.5, lw=1.2 if col != "label_s"
                else 0.9, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("circuit width n")
    ax.set_ylabel("seconds (median of repeats)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax.set_xticks(sorted(int(x) for x in g.n))
    if len(rej):
        ax.set_title("A  cost of one recommendation against width", loc="left", fontweight="bold")
        rj = rej.reset_index(drop=True)
        xs = np.arange(len(rj))
        axr.bar(xs, rj.recommend_s, color="#56B4E9", width=0.6, edgecolor="#FFFFFF", linewidth=2)
        for i, r_ in rj.iterrows():
            st_ = str(r_.recommend_status).replace("out-of-distribution", "out of\ndistribution")
            axr.text(i, float(r_.recommend_s) * 1.04, "%s\n%.0f MB peak" % (st_, r_.peak_rss_mb),
                     ha="center", va="bottom", fontsize=6.5, color="#333333")
        axr.set_xticks(xs)
        axr.set_xticklabels(["%s-%d" % (r_.family, r_.n) for r_ in rj.itertuples()])
        axr.set_ylabel("seconds, whole recommend() call")
        axr.set_ylim(0, float(rj.recommend_s.max()) * 1.35)
        axr.set_xlim(-0.75, len(xs) - 0.25)            # room for the first bar's label
        axr.set_title("B  early rejection outside the training widths", loc="left", fontweight="bold")
    fig.suptitle("Measured on the build machine (2 cores shared with other jobs): magnitudes, not a benchmark.",
                 fontsize=6.5, color="#555555")
    fam = rt.family.iloc[0] if "family" in rt.columns and len(rt) else "ising"
    reps = int(rt.groupby("n").size().max()) if len(rt) else 0
    nl = rt.groupby("n").label_tasks.max().to_dict() if "label_tasks" in rt.columns else {}
    cap = ("Classical cost of one recommendation against width for a 2-step %s circuit on fake_fez at B = 10⁵, "
           "through the deployment path of the Playground and the recommend command: compilation, the deployable "
           "features (structure, observable, light cone, calibration-derived noise sums and the closed-form protocol "
           "overheads — no simulation of the circuit), the simulator-only augmentation (per-shot variances and signal "
           "of RO, ZNE-R and PEC on the snapshot by exact density-matrix simulation and the ideal state's entropies — "
           "exponential in width; only for a model that uses them) and model inference; medians of %d repeats in one "
           "process. Dashed: for comparison, the median wall time of generating the benchmark labels of one "
           "development task of that width (all protocols, all budgets, both label halves; %s tasks per width; "
           "missing where the development set has no task of that width) — a cost of building the study, not of a "
           "recommendation. Memory: %s, in the source data. Measured on one machine; not a benchmark of other "
           "hardware." % (fam, reps, "/".join(str(nl[k]) for k in sorted(nl)) or "n/a",
                          rt.rss_kind.iloc[0] if "rss_kind" in rt.columns and len(rt) else "peak resident memory"))
    if "recommend_s" in rt.columns:
        cap += (" 1.8.0 components: the structural preflight of every candidate of the portfolio, the design-level "
                "preflight of the learning candidates (training design with its deployment seed, ideal training values "
                "by state vector), and the whole recommend() call with the deployable bundle (guard, compilation, "
                "features without cache, out-of-distribution checks, preflight in the model's order, margin rule, "
                "fallback). No component uses quantum shots.")
    if len(rej):
        cap += (" (B) Wide GHZ chains (%s qubits; observable z_mid) outside every training width: the whole call "
                "and the process's peak memory; status as returned (an out-of-distribution answer uses the fallback; "
                "no dense state of the circuit is built)." % ", ".join(str(int(x)) for x in rej.n))
    shares = g.set_index("n")
    big = int(shares.index.max()) if len(shares) else 0
    dom = (shares.loc[big, [c for c in ("compile_s", "deployable_s", "simulated_s", "inference_s")
                            if c in shares.columns]].idxmax().replace("_s", "") if len(shares) else "n/a")
    grow = (float(shares.simulated_s.iloc[-1] / shares.simulated_s.iloc[0]) if len(shares) > 1 else float("nan"))
    data = dict(runtime=rt, medians=g)
    if len(rej):
        data["rejection"] = rej
    return _save(fig, out_dir, "supp_runtime", data, cap,
                 "Observation: at n = %d the largest component of a recommendation is '%s'; the simulator-only "
                 "augmentation grows by a factor %.1f from n = %d to n = %d, while a recommendation from deployable "
                 "features needs no simulation. Label generation, the study's own cost, is orders of magnitude above "
                 "a recommendation." % (big, dom, grow, int(shares.index.min()) if len(shares) else 0, big))


def supp_design_variability(run_dir: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Learning-design variability of CDR and NIL: (a) the distribution of design-level mean losses relative to the
    task's fresh-design risk at three budgets (development labels); (b) variance decomposition of the label — the
    median share of the between-design component in the variance of the evaluation-half MSE, and the ratio of the
    design-level to the trial-level standard error; (c) the focused design study (experiment B): design-level MSE
    distributions of CDR, NIL and alternative training allocations on the review's severe-NIL tasks."""
    import pandas as pd
    plt = _plt()
    adir = os.path.join(run_dir, "analysis")
    dd = pd.read_csv(os.path.join(adir, "design_diagnostics.csv"), encoding="utf-8")
    dv = pd.read_csv(os.path.join(adir, "design_variance.csv"), encoding="utf-8")
    ds_dir = os.path.join(run_dir, "design_study")
    study = (pd.read_csv(os.path.join(ds_dir, "designs.csv"), encoding="utf-8")
             if os.path.exists(os.path.join(ds_dir, "designs.csv")) else pd.DataFrame())
    fig, ax4 = plt.subplots(2, 2, figsize=(7.4, 5.4), layout="constrained")
    axA, axB, axS, axC = ax4[0, 0], ax4[0, 1], ax4[1, 0], ax4[1, 1]
    e = dd[dd.half == "eval"].copy()
    med = e.groupby(["task_id", "protocol", "B"]).mean_loss.transform("median")
    e["rel"] = np.log10((e.mean_loss + EPS0) / (med + EPS0))
    budgets = sorted(e.B.unique())
    protos = [p for p in ("CDR", "NIL", "FR") if p in set(e.protocol)]       # the learning candidates (FR: 1.8.0)
    rowsA = []
    for q, p in enumerate(protos):
        for k, B in enumerate(budgets):
            v = e[(e.protocol == p) & (e.B == B)].rel.dropna().to_numpy()
            if not len(v):
                continue
            dx = 0.32 if len(protos) <= 2 else 0.25
            x = k + (q - (len(protos) - 1) / 2) * dx
            parts = axA.violinplot([v], positions=[x], widths=0.85 * dx, showextrema=False)
            for b_ in parts["bodies"]:
                b_.set_facecolor(PROTOCOL_COLORS[p])
                b_.set_alpha(0.45)
                b_.set_edgecolor("none")
            axA.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color="#000000", lw=0.8)
            top = np.sort(v)[-3:]
            axA.plot(np.full(len(top), x), top, marker=PROTOCOL_MARKERS[p], ls="", ms=3, color=PROTOCOL_COLORS[p])
            rowsA.append(dict(protocol=p, B=int(B), designs=int(len(v)), median_rel=float(np.median(v)),
                              q90_rel=float(np.quantile(v, 0.9)), max_rel=float(v.max()),
                              share_above_10x=float(np.mean(v > 1.0))))
    axA.axhline(0, color="#333333", lw=0.7)
    axA.axhline(1, color="#999999", lw=0.6, ls=":")
    axA.set_xticks(range(len(budgets)), [_blabel(b) for b in budgets])
    for k in range(1, len(budgets)):                       # budget groups separated
        axA.axvline(k - 0.5, color="#DDDDDD", lw=0.6)
    axA.grid(axis="x", visible=False)
    axA.set_xlabel("total budget B")
    axA.set_ylabel("log10 design MSE / task's median design")
    axA.set_title("(a) design-level losses (dev)", loc="left", fontsize=7)
    for p in protos:
        axA.plot([], [], marker=PROTOCOL_MARKERS[p], ls="", color=PROTOCOL_COLORS[p], ms=3, label=p)
    axA.legend(frameon=False, fontsize=6.5, loc="upper left")
    for p in protos:
        f = dv[dv.protocol == p].sort_values("B")
        if f.empty:
            continue
        axB.plot(f.B, f.median_design_share_of_label_variance, PROTOCOL_MARKERS[p] + "-", color=PROTOCOL_COLORS[p],
                 ms=3, lw=1.1, label=p)
        axS.plot(f.B, f.median_se_ratio, PROTOCOL_MARKERS[p] + "-", color=PROTOCOL_COLORS[p], ms=3, lw=1.1,
                 label="%s median" % p)
        axS.plot(f.B, f.p90_se_ratio, PROTOCOL_MARKERS[p] + ":", mfc="white", color=PROTOCOL_COLORS[p], ms=3,
                 lw=0.9, label="%s 90 %% quantile" % p)
    axB.set_xscale("log")
    axB.set_ylim(0, 1)
    axB.set_xlabel("total budget B")
    axB.set_ylabel("median between-design share")
    axB.set_title("(b) share of the label variance from designs", loc="left", fontsize=7)
    axB.legend(frameon=False, fontsize=6.5, loc="best")
    axS.axhline(1, color="#333333", lw=0.8)
    axS.set_xscale("log")
    axS.set_xlabel("total budget B")
    axS.set_ylabel("design-level SE / trial-level SE")
    axS.set_title("(c) understatement of a trial-level SE", loc="left", fontsize=7)
    axS.legend(frameon=False, fontsize=6.5, loc="best")
    rowsC = []
    if len(study):
        s = study[(study.status == "ok") & (study.B == study.B.max())].copy()
        tasks = list(dict.fromkeys(s.task))
        variants = sorted(dict.fromkeys(s.variant), key=lambda v: (not v.startswith("CDR"), v != v.split("-")[0], v))
        for i, t in enumerate(tasks):
            for j, vname in enumerate(variants):
                v = s[(s.task == t) & (s.variant == vname)].mean_loss.to_numpy()
                if not len(v):
                    continue
                y = i + ((len(variants) - 1) / 2 - j) * (0.8 / len(variants))     # first variant on top
                c = PROTOCOL_COLORS["NIL"] if vname.startswith("NIL") else PROTOCOL_COLORS["CDR"]
                axC.plot(np.log10(v + EPS0), np.full(len(v), y), "|", color=c, ms=3, alpha=0.6)
                axC.plot([np.log10(v.mean() + EPS0)], [y], "o", mfc="white", mec=c, ms=3)
                rowsC.append(dict(task=t, variant=vname, B=int(s.B.max()), designs=int(len(v)), mse=float(v.mean()),
                                  se_design=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else math.nan,
                                  max_design_mse=float(v.max())))
        def _tlab(t):                                 # family-n(params) and the observable, without the device
            parts = t.split("|")
            obs = [p_ for p_ in parts if p_.startswith("O=")]
            return parts[0] + ("\n" + obs[0] if obs else "")
        axC.set_yticks([i for i in range(len(tasks))], [_tlab(t) for t in tasks], fontsize=6)
        axC.text(0.99, 0.99, "rows in each circuit, top to bottom:\n" + ", ".join(variants[:3]) + ",\n"
                 + ", ".join(variants[3:]), transform=axC.transAxes, ha="right", va="top", fontsize=6,
                 color="#333333", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
        axC.set_ylim(-0.6, len(tasks) - 0.4 + 0.5)
        axC.axvline(-2, color="#999999", lw=0.6, ls=":")
        axC.set_xlabel("log10 design MSE (B = %s)" % _blabel(float(s.B.max())))
        axC.set_title("(d) design study: %d variants, ticks = designs" % len(variants), loc="left", fontsize=7)
    else:
        axC.text(0.5, 0.5, "design study not run", ha="center", va="center", transform=axC.transAxes, fontsize=7)
        axC.set_axis_off()
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    kk = sorted(set(int(x) for x in dv.designs)) if len(dv) else []
    cap = ("Learning-design variability of the learning candidates (%s). (a) Development labels, evaluation half: "
           "for every task, budget and fresh training design (%s designs per half), the design's mean loss relative to the median "
           "design of the same task and budget (log10 of the ratio, both + 10⁻⁶); violins over all designs, black: "
           "median, markers: the three largest; dotted line: a design ten times worse than the task's median design. "
           "(b) Median over tasks of the between-design share of the variance of the evaluation-half MSE, "
           "Var_D(E[L|D])/K ÷ (that + E Var(L|D)/(Km)), from a one-way analysis of variance. (c) Ratio of the "
           "design-level (cluster-robust) to the trial-level standard error of the label, median and 90 %% quantile "
           "over tasks: a trial-level SE understates the fresh-design uncertainty by this factor (1 = no "
           "understatement). (d) The focused design study "
           "(experiment B, scripts/design_study.py): on the review's severe-NIL tasks and controls, design-level MSEs "
           "(ticks; open circle: mean over designs) of CDR, NIL and alternative training allocations (more training "
           "circuits, another share of B, the bare circuit added) at the largest budget; dotted: MSE 10⁻² (RMSE "
           "0.1, the study's severe threshold). Exploratory; designs are the independent unit within a task, tasks "
           "are few. %s" % (", ".join(protos), "/".join(map(str, kk)) or "K", LABEL_SIM))
    a_ = pd.DataFrame(rowsA)
    interp = ("Share of designs at least ten times worse than their task's median design: %s. Median between-design share of "
              "the label variance at the largest budget: %s."
              % (", ".join("%s B=%s %.0f %%" % (r_.protocol, _btext(r_.B), 100 * r_.share_above_10x)
                           for _, r_ in a_.iterrows()) or "n/a",
                 ", ".join("%s %.2f" % (p, float(dv[(dv.protocol == p)].sort_values("B")
                                                 .median_design_share_of_label_variance.iloc[-1]))
                           for p in protos if (dv.protocol == p).any()) or "n/a"))
    return _save(fig, out_dir, "supp_design_variability",
                 dict(design_losses=a_, variance=dv, design_study=pd.DataFrame(rowsC)), cap, interp)


# ------------------------------------------------------------------ tables
PROTOCOL_INFO = {
    # noise information used, training/calibration cost, allocation rule, failure policy, approximation status
    "RAW": ("none", "none", "all shots on the target circuit", "none needed (nonresponse → 0)",
            "exact multinomial sampling of the target's distribution"),
    "RO": ("readout calibration (measured, charged)", "2 calibration circuits", "variance-optimal calibration/target "
           "split from the snapshot's cost model, rounded down", "singular calibration → uncorrected estimate",
           "exact multinomial sampling, calibration drawn per trial"),
    "ZNE-L": ("snapshot error rates (fold allocation only)", "2 calibration circuits", "integer minimizer of "
              "Σ a_j²/N_j over folds λ = 1, 3 (a_j from the snapshot)", "singular calibration → uncorrected folds",
              "exact sampling per fold; global folding"),
    "ZNE-R": ("snapshot error rates (fold allocation only)", "2 calibration circuits", "as ZNE-L over λ = 1, 3, 5",
              "as ZNE-L", "exact sampling per fold"),
    "ZNE-E": ("snapshot error rates (fold allocation only)", "2 calibration circuits", "as ZNE-R",
              "failed exponential fit → linear fit of the same data", "exact sampling per fold; fit per trial"),
    "PEC": ("snapshot's median-rate depolarizing model (supplied, not the truth)", "2 calibration circuits",
            "one shot per sampled circuit; calibration split as RO", "singular calibration → uncorrected; a model "
            "that cannot be inverted → infeasible", "exact sign-resolved two-stage law"),
    "CDR": ("none (classical simulation of 16 near-Clifford training circuits)", "25 % of B on training, equal per "
            "circuit; 2 calibration circuits", "≥ 32 shots per training circuit, else infeasible",
            "no usable training direction (this trial's data) → readout-corrected value",
            "exact sampling; training design redrawn per deployment (6 designs per task)"),
    "NIL": ("none (classical simulation of 16 2-design training circuits)", "50 % of B on training, equal split; 2 "
            "calibration circuits", "≥ 32 shots per training-neighbor circuit, else infeasible",
            "training labels without spread (or no slot to train) → the readout-corrected target value",
            "exact sampling; design redrawn per deployment; 16 training circuits (paper: 256–5 000)"),
}


def _policy_table(f):
    """Table 3's columns from a pooled metrics table (B = all): log and arithmetic risk, regret and its tails, the
    paired difference to SBS-B, coverage and the union failure share; base-circuit weighted throughout."""
    import pandas as pd
    f = f[f.B.astype(str) == "all"].copy()

    def ci(m, lo, hi, fmt="%.3f"):
        return (fmt + " [" + fmt + ", " + fmt + "]") % (m, lo, hi)

    def col(name, default=math.nan):
        return f[name] if name in f.columns else pd.Series(default, index=f.index)
    return pd.DataFrame(dict(
        policy=f.policy, base_circuits=col("base_circuits").astype("Int64"), rows=col("rows").astype("Int64"),
        mean_log10_MSE=[ci(*r) for r in f[["mean_log10_risk", "ci_lo", "ci_hi"]].values],
        arithmetic_MSE=col("policy_mse").map(lambda v: "%.3g" % v),
        mean_log10_regret=[ci(*r) for r in f[["mean_log_regret", "log_regret_lo", "log_regret_hi"]].values],
        median_regret=col("median_regret").map(lambda v: "%.2g" % v),
        p90_regret=col("p90_regret").map(lambda v: "%.2g" % v),
        p90_log10_regret=col("p90_log_regret").round(3),
        share_regret_gt_2x=col("p_regret_gt_2x").round(3),
        within_10pct_of_reference=col("tol_success_10pct").round(2), worse_than_RAW=col("raw_degradation").round(2),
        diff_vs_SBS_B=[ci(*r) if np.isfinite(r).all() else "" for r in
                       f.reindex(columns=["diff_vs_sbsb", "diff_vs_sbsb_lo", "diff_vs_sbsb_hi"]).values],
        coverage=col("coverage").round(2),
        trials_with_any_fallback=col("fail_share").round(3), weighting=col("weighting", "base circuits")))


def tables(run_dir: str, out_dir: str, cfg: dict | None = None, pilot_cfg: dict | None = None) -> dict:
    """Table 1 (experimental design: tasks and independent base circuits counted separately, per family, plus every
    study of the ancestry registry with its role and status), Table 2 (the protocol configurations compared), Table 3
    (policy results: the confirmation set when it was evaluated, and separately the historical re-evaluation of the
    1.6.0 final test), as CSV and Markdown, plus Table S1 (nested-evaluation rejection by family and device, with
    independent group counts)."""
    import pandas as pd
    from . import protocols as PR
    from .figures import markdown_table
    os.makedirs(out_dir, exist_ok=True)
    man = TX.read_json(os.path.join(run_dir, "manifest.json"))
    cfgd = man["config"]
    feats = {}
    for part in ("dev", "final", "confirm"):
        tdir = os.path.join(run_dir, part, "tasks")
        if os.path.isdir(tdir):
            for f in os.listdir(tdir):
                if f.endswith(".json"):
                    r = TX.read_json(os.path.join(tdir, f))
                    if r.get("status", "ok") != "ok":
                        continue
                    feats[r["task_id"]] = (r.get("features") or {}, r["spec"], part, r.get("base_id"))
    t1 = []
    for fam in sorted({v[1]["family"] for v in feats.values()}):
        vs = [v for v in feats.values() if v[1]["family"] == fam]
        depth = [v[0].get("depth") for v in vs if v[0].get("depth") is not None]
        row = dict(family=fam, widths=",".join(str(n) for n in sorted({v[1]["n"] for v in vs})),
                   compiled_depth="%d–%d" % (min(depth), max(depth)) if depth else "",
                   observables=",".join(sorted({v[1]["observable"] for v in vs})),
                   devices=",".join(sorted({v[1]["device"] for v in vs})))
        for part, tag in (("dev", "dev"), ("final", "final"), ("confirm", "confirm")):
            row["tasks_" + tag] = sum(1 for v in vs if v[2] == part)
            row["base_circuits_" + tag] = len({v[3] for v in vs if v[2] == part})
        t1.append(row)
    t1 = pd.DataFrame(t1)
    bud = sorted(cfgd["budgets"])
    t1["budgets"] = "%d values, %s–%s" % (len(bud), _btext(bud[0]), _btext(bud[-1]))
    t1["trials_per_label"] = "%d + %d (selection + evaluation)" % (cfgd["R"], cfgd["R"])
    ld = cfgd.get("learning_designs")
    t1["learning_designs"] = ("fresh, %d per half" % ld.get("per_half") if isinstance(ld, dict) and
                              ld.get("mode", "fresh") == "fresh" else "fixed shared (%s)" % ld)
    k = int(cfgd.get("analysis", {}).get("split_options", {}).get("same-distribution", {}).get("k", 5))
    t1["split"] = ("group %d-fold (development); historical final test by keyed hash of the base circuit (%g %%); "
                   "confirmation: a separate set of new logical circuits" % (k, 100 * float(cfgd.get(
                       "final_test_fraction", 0))))
    from . import ancestry as AC
    reg = AC.load()
    counts = run_counts(run_dir)
    keymap = {"pilot/dev": "dev", "pilot/final": "final", "regimes": "reg", "noisecomp17": "nc",
              "mismatch": "mm", "confirm17": "conf"}
    t1b = []
    for name, st in sorted((reg.get("studies") or {}).items()):
        key = keymap.get(name)
        t1b.append(dict(study=name, role=st.get("role", ""), logical_circuits=st.get("logical_circuits", ""),
                        executed="no (planned)" if st.get("role") == "planned" else
                        ("yes" if key and counts.get(key + "_tasks") else "not in this run"), tasks_labeled=counts.get(key + "_tasks", "") if key else "",
                        base_circuits_labeled=counts.get(key + "_base", "") if key else "",
                        failed_tasks=counts.get(key + "_failed", "") if key else "",
                        note=st.get("note", "")))
    t1b = pd.DataFrame(t1b)
    t2 = []
    for p in (PR.PORTFOLIO[k] for k in cfgd.get("protocols", PR.PORTFOLIO_16)):     # this run's candidates
        info = PROTOCOL_INFO.get(p.key, ("", "", "", "", ""))
        t2.append(dict(protocol=p.key, estimator=p.estimator, readout=p.readout,
                       settings=", ".join("%s=%s" % kv for kv in p.params), noise_information=info[0],
                       training_calibration_cost=info[1], allocation_rule=info[2], failure_policy=info[3],
                       approximation_status=info[4],
                       scope="the configuration compared in this study, not every implementation of the method"))
    t2 = pd.DataFrame(t2)
    t3, t3h = pd.DataFrame(), pd.DataFrame()
    fc = os.path.join(run_dir, "confirm", "metrics_confirm.csv")
    if os.path.exists(fc):
        t3 = _policy_table(pd.read_csv(fc, encoding="utf-8"))
        t3.insert(0, "statement", "confirmatory (evaluated once, frozen configuration)")
    fm = os.path.join(run_dir, "final", "metrics_final.csv")
    if os.path.exists(fm):
        t3h = _policy_table(pd.read_csv(fm, encoding="utf-8"))
        t3h.insert(0, "statement", "historical final test (re-evaluated; overlaps earlier studies, not confirmatory)")
    paths = {}
    for name, t in (("table1_design", t1), ("table1b_studies", t1b), ("table2_protocols", t2),
                    ("table3_policy_results", t3), ("table3h_historical_final", t3h)):
        p = os.path.join(out_dir, name + ".csv")
        t.to_csv(p, index=False, encoding="utf-8")
        with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as fh:
            fh.write(markdown_table(t) if len(t) else "(no data: not evaluated in this run)\n")
        paths[name] = p
    for src, name in (("nested_groups.csv", "table_s1_rejection"), ("selective_groups.csv", "table_s1_rejection")):
        sg = os.path.join(run_dir, "analysis", src)
        if os.path.exists(sg):
            s = pd.read_csv(sg, encoding="utf-8")
            s.insert(0, "evaluation", "nested" if src.startswith("nested") else "cross-fitted (1.6.0 procedure)")
            s.to_csv(os.path.join(out_dir, name + ".csv"), index=False, encoding="utf-8")
            with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as fh:
                fh.write(markdown_table(s))
            paths[name] = os.path.join(out_dir, name + ".csv")
            break
    return paths
