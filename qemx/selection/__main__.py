"""Command line of the protocol-selection study: ``python -m qemx.selection <command> …`` (1.6.0; 1.7.0).

  plan      CONFIG                         tasks, base circuits, design table and a resource estimate
  label     CONFIG --out DIR [--workers N] [--limit K]
                                           label the development tasks (resumable)
  analyze   CONFIG --out DIR               every evaluation of the configuration, from saved trials
  figures   CONFIG --out DIR               the figures and tables (from the analysis tables only)
  mismatch  CONFIG --out DIR --dev DIR [--workers N] [--skip-label]
                                           label the mismatch scenarios, then diagnostics and policies
  validate  --out DIR [--tasks K] [--B N] [--R N]
                                           emulator against direct shot-by-shot execution
  freeze    CONFIG --out DIR               freeze the development choices (policy, features, digests)
  final     CONFIG --out DIR [--workers N] label and evaluate the untouched final test, once
  report    CONFIG --out DIR               the results tables in Markdown (RESULTS_TABLES.md), from saved tables
  run       CONFIG --out DIR [--workers N] label + analyze + figures
  regimes   CONFIG --out DIR [--workers N] [--skip-label]
                                           the controlled regime study: label into DIR/regimes, analyze with DIR's
                                           frozen recommender
  profile   CONFIG --out DIR [--widths 3,4,5,6,7]
                                           classical cost of feature extraction and inference against width
  confirm   CONFIG --out DIR --dev-config DEVCONFIG [--train dev,final]
                                           the once-only confirmation set (1.7.0): new logical circuits only
  bundle    CONFIG --out DIR [--train dev,final] [--feature-set NAME]
                                           the versioned model bundle of the frozen recommender
  recommend (--state FILE | --experiment FILE) --budget B [--device D] [--bundle DIR] [--allow-simulation]
                                           the recommendation for one experiment (as in the Playground)
  status    --out DIR [--stage S]          completion of every labels directory (1.9.0)
  check-config FILE...                     validate any YAML file with its own parser (1.9.0)
  pipeline  PIPELINE.yaml --out DIR        stage-aware, resumable study pipeline (1.9.0; configs/pipelines)

Exit codes (1.9.0): 0 success; 1 incomplete labels or a once-only step already done (unchanged); 2 refused
(configuration, binding, admission, corrupted or stale inputs, inconsistent once-only step); 3 plan: a configured
cap is exceeded by the estimate.

Nothing here contacts a quantum provider.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from .. import textio as TX


def cmd_registry(a):
    from . import registry as RG
    if a.markdown:
        sys.stdout.write(RG.markdown())
    elif a.candidate:
        print(TX.dumps(RG.describe(a.candidate)))
    else:
        print(TX.dumps(RG.inventory()))
    return 0


def _cfg(path):
    from . import design as DS
    return DS.load(path)


def _workers(a, cfg=None) -> int:
    """--workers N, or 'auto' (``resources.recommended_workers``: CPUs − 1, and the per-task memory × workers within
    70 % of RAM)."""
    w = getattr(a, "workers", 1)
    if str(w).lower() == "auto":
        from . import resources as RS
        return RS.recommended_workers(cfg or {}, RS.system(getattr(a, "out", None) or "."))
    return max(1, int(w))


def _stage_dir(a):
    out = getattr(a, "out", None)
    if not out:
        return None
    return os.path.join(out, getattr(a, "subdir", None) or ("final" if getattr(a, "final_set", False) else "dev"))


def cmd_plan(a):
    """Tasks, the expansion report, the design table, the resource preflight and the configuration checks — before
    anything is labeled. Exit 0: admissible; 2: the configuration has errors; 3: a configured cap would refuse it."""
    from . import design as DS, resources as RS, run as RN
    cfg = _cfg(a.config)
    chk = DS.check(cfg)
    specs = DS.expand(cfg)
    dev, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    est = RN.estimate(cfg, measured=RN.measured_rates(a.measured) if a.measured else None)
    print(TX.dumps(dict(est, development=len(dev), final=len(fin)), indent=1))
    for r in DS.design_table(specs):
        print(r)
    rep = DS.expansion_report(cfg)
    print("\nexpansion (%s): %s" % (rep["policy"], rep["summary"]))
    for w in rep["notes"]:
        print("  note: " + w)
    sd = _stage_dir(a)
    pre = RS.preflight(cfg, sd if sd and os.path.isdir(sd) else None, workers=_workers(a, cfg),
                       measured_dir=a.measured, final=bool(getattr(a, "final_set", False)),
                       subdir=getattr(a, "subdir", None))
    print("\nresource preflight:\n" + RS.text(pre))
    try:                                             # 1.9.1 (independent review F4): analysis sized separately
        wl = RS.analysis_workload(cfg)
        print("\nanalysis workload (structural counts; labeling caps do not limit analysis time):\n"
              + RS.workload_text(wl))
    except Exception as e:                           # noqa: BLE001 - a planning aid, never a reason to fail
        wl = dict(error=str(e))
        print("\nanalysis workload: not available (%s)" % e)
    for w in chk["warnings"]:
        print("warning: " + w)
    for e in chk["errors"]:
        print("ERROR: " + e)
    if getattr(a, "tasks_csv", None):
        DS.write_task_table(cfg, a.tasks_csv)
        print("expanded task table: %s" % a.tasks_csv)
    if getattr(a, "json", None):
        TX.write_json(a.json, dict(estimate=est, expansion=rep, preflight=pre, analysis_workload=wl, checks=chk),
                      indent=1, default=str)
    if chk["errors"]:
        return 2
    if any(v["status"] == "exceeds" for v in pre["caps"].values()):
        print("ADMISSION: a configured cap is exceeded by the estimate; label would refuse without --override-limits")
        return 3
    return 0


def _refusal(e) -> int:
    print("refused: %s" % e)
    return 2


def cmd_label(a):
    """Label (resumable). Exit 0: the run is complete, or deliberately partial with --limit and nothing failed;
    1: the run is incomplete (failed, corrupted or remaining tasks; see the advice printed); 2: refused before
    labeling (configuration, binding, admission, corrupted records, another writer)."""
    from . import binding as BD, design as DS, integrity as IN, resources as RS, run as RN
    try:
        cfg = _cfg(a.config)
        chk = DS.check(cfg)
        if chk["errors"]:
            return _refusal("; ".join(chk["errors"]))
        s = RN.label(cfg, a.out, workers=_workers(a, cfg), limit=a.limit, accept_changes=a.accept_changes,
                     retry_failed=a.retry_failed, final=bool(getattr(a, "final_set", False)),
                     subdir=getattr(a, "subdir", None), quarantine_corrupted=bool(getattr(a, "quarantine_corrupted",
                                                                                          False)),
                     override_limits=getattr(a, "override_limits", None))
    except (RS.AdmissionRefused, IN.CorruptedRun, BD.IncompatibleRun, IN.RunLocked, DS.ConfigError) as e:
        return _refusal(e)
    print(TX.dumps({k: v for k, v in s.items() if k not in ("per_task_seconds", "preflight", "run", "supervision")}))
    _print_completion(s["run"], s.get("stopped"))
    return int(s["exit_code"])


def _print_completion(comp, stopped=None):
    from . import integrity as IN
    print("run: " + IN.describe(comp))
    if stopped:
        print("stopped by a monitored ceiling: %s (running tasks finished and were saved)" % stopped)
    for x in IN.advice(comp, stopped):
        print("  - " + x)


def cmd_status(a):
    """Completion of every labels directory of a run (dev, final, confirm, regimes, noisecomp, mismatch, …), or of
    --stage only; exit 0 when all are complete, 1 otherwise. Reads only."""
    from . import integrity as IN
    stages = [a.stage] if a.stage else sorted(d for d in os.listdir(a.out)
                                               if os.path.isdir(os.path.join(a.out, d, "tasks")))
    allok, out = True, {}
    for st in stages:
        c = IN.completion(os.path.join(a.out, st), check_arrays=a.check)
        out[st] = {k: v for k, v in c.items() if k != "completed_ids"}
        print("%-10s %s" % (st, IN.describe(c)))
        if c.get("consumed"):                        # 1.9.1 (F3): the persistent ledger
            k = c["consumed"]
            print("%-10s labeling time consumed: %.3f h in %d attempt(s)%s (summed worker wall time, every invocation)"
                  % ("", k["total_seconds"] / 3600.0, k["attempts"], (" + %d record(s) without attempts"
                                                                       % k["records_without_attempts"])
                     if k["records_without_attempts"] else ""))
        allok &= c["complete"]
    if a.json:
        TX.write_json(a.json, out, indent=1, default=str)
    return 0 if allok and stages else 1


def cmd_analyze(a):
    from . import integrity as IN, run as RN
    cfg = _cfg(a.config)
    named = [x for x in (a.extras.split(",") if a.extras is not None else []) if x in RN.FROZEN_EXTRAS]
    if named and RN.deployed_configuration(a.out, cfg.get("analysis", {})) is None:
        an = cfg.get("analysis", {})                 # 1.9.1: an explicit request is refused, not silently deferred
        half = (" (analysis.deployed_family and analysis.deployed_set are used only together; one is declared)"
                if bool(an.get("deployed_family")) != bool(an.get("deployed_set")) else "")
        return _refusal("%s evaluate%s the frozen model configuration, which is not known before `freeze`%s: run "
                        "freeze first, then analyze CONFIG --out OUT --splits= --extras deployed,nested_selective (the "
                        "pipeline's 'nested' stage does this), or declare analysis.deployed_family and deployed_set"
                        % (", ".join(named), "" if len(named) > 1 else "s", half))
    try:
        r = RN.analyze(a.out, cfg, only_splits=a.splits.split(",") if a.splits is not None else None,
                       only_extras=a.extras.split(",") if a.extras is not None else None,
                       allow_partial=bool(getattr(a, "allow_partial", False)),
                       reset=bool(getattr(a, "reset_analysis", False)),
                       accept_changes=bool(getattr(a, "accept_changes", False)))
    except (IN.IncompleteRun, IN.CorruptedRun, RN.StaleAnalysis, IN.RunLocked) as e:
        return _refusal(e)
    print(TX.dumps({k: v for k, v in r.items() if k != "evaluations"}, indent=1, default=str))
    if r.get("data", {}).get("status") == "partial":
        print("PARTIAL: exploratory analysis of %d of %d development tasks (not a completed study; a freeze refuses it)"
              % (r["data"]["completed"], r["data"]["expected"]))
    return 0


def cmd_figures(a):
    """The six main figures, the supplementary figures and the tables, from the saved tables of OUT only. A figure
    whose input tables are missing is reported as pending (not drawn); nothing is simulated."""
    from . import figures as FG, plots as PL, plots18 as P18, run as RN
    cfg = _cfg(a.config)
    adir = os.path.join(a.out, "analysis")
    fdir = getattr(a, "figures_out", None) or os.path.join(a.out, "figures")
    unpooled = [f[len("choices_"):-4] for f in os.listdir(adir) if f.startswith("choices_") and f.endswith(".csv")
                and not os.path.exists(os.path.join(adir, "pooled_" + f[len("choices_"):]))]
    if unpooled:                                     # tables of an older analysis; 1.9.1: only the missing ones
        RN.pool(adir, reps=int(cfg.get("analysis", {}).get("bootstrap", 1000)), only=unpooled)
    pcfg = PL.load_config(a.plot_config)
    need = dict(
        fig01_budget_efficiency=["analysis/choices_same-distribution.csv", "analysis/labels.csv"],
        fig02_performance_regimes=["regimes/analysis/regime_cells.csv"],
        fig03_feature_mechanisms=["analysis/ablation.csv", "regimes/analysis/regime_protocols.csv"],
        fig04_generalization_learning=["analysis/pooled_same-distribution.csv", "analysis/learning_curve.csv",
                                       "analysis/budget_generalization.csv"],
        fig05_reliability_abstention=["analysis/nested_curve.csv", "analysis/nested_summary.csv",
                                      "analysis/calibration.csv", "analysis/interval_coverage.csv"],
        fig06_mismatch_diagnostics=["mismatch/policies.csv", "mismatch/diagnostics.csv",
                                    "mismatch/analysis/deployment_draws.csv", "mismatch/analysis/reactions.csv"],
        supp_target_precision=["analysis/labels.csv"],
        supp_bias_variance=["regimes/analysis/regime_cells.csv", "regimes/tasks/*.npz"],     # per-trial arrays
        supp_prediction_quality=["analysis/prediction.csv"],
        supp_dataset_coverage=["dev/tasks"],
        supp_coverage_certificates=["analysis/interval_coverage.csv", "analysis/certificates.csv"],
        supp_failures_costs=["analysis/labels.csv"],
        supp_representative=["analysis/deployed_rows.csv", "analysis/selective_rows.csv", "dev/tasks/*.npz"],
        supp_runtime=["analysis/runtime.csv"],
        supp_design_variability=["analysis/design_diagnostics.csv", "analysis/design_variance.csv"],
        **P18.NEED_18,                                # 1.8.0 supplementary figures
        tables=["manifest.json"])
    counts = PL.run_counts(a.out)
    fn = dict(supp_coverage_certificates=lambda: FG.supp_coverage_certificates(adir, fdir),
              **{k: (lambda k=k: getattr(P18, k)(a.out, fdir, pcfg)) for k in P18.NEED_18},
              tables=lambda: PL.tables(a.out, os.path.join(a.out, "tables"), pcfg))
    done, failed, pending = [], [], []
    only = [x.strip() for x in (getattr(a, "only", None) or "").split(",") if x.strip()]
    unknown = sorted(set(only) - set(need))
    if unknown:
        print("unknown figure(s): %s (known: %s)" % (", ".join(unknown), ", ".join(need)))
        return 2
    for name, req in need.items():
        if only and name not in only:
            continue
        missing = [r for r in req if not (glob.glob(os.path.join(a.out, r)) if "*" in r
                                          else _present(os.path.join(a.out, r)))]
        if missing:
            pending.append("%s: missing %s" % (name, ", ".join(missing)))
            continue
        try:
            (fn.get(name) or (lambda n=name: getattr(PL, n)(a.out, fdir, pcfg)))()
            rel = os.path.relpath(a.out)
            PL.annotate(fdir, name, "python -m qemx.selection figures %s --out %s%s"
                        % (a.config, a.out if rel.startswith("..") else rel,
                           (" --plot-config " + a.plot_config) if a.plot_config else ""), a.plot_config, counts)
            done.append(name)
        except Exception as e:                       # noqa: BLE001 - report every figure that could not be drawn
            failed.append("%s: %s: %s" % (name, type(e).__name__, e))
    pend = os.path.join(fdir, "PENDING.md")
    if pending or failed:
        os.makedirs(fdir, exist_ok=True)
        with open(pend, "w", encoding="utf-8") as fh:
            if pending:
                fh.write("# Figures not drawn (input tables missing)\n\n" + "\n".join("- " + p for p in pending)
                         + "\n")
            if failed:
                fh.write("\n# Figures that failed\n\n" + "\n".join("- " + p for p in failed) + "\n")
    elif os.path.exists(pend) and not only:
        os.remove(pend)                              # every figure drawn: no stale pending list
    print(TX.dumps(dict(done=done, failed=failed, pending=pending), indent=1))
    return 0 if not failed else 1


def _present(path: str) -> bool:
    """A figure's input exists and, for a table, has a header row (1.9.0: an analysis that found nothing writes an
    empty CSV; its figure is then pending, not failed)."""
    if not os.path.exists(path):
        return False
    if path.endswith(".csv") and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return bool(f.readline().strip())
    return True


def cmd_mismatch(a):
    from . import run as RN
    cfg = _cfg(a.config)
    if not a.skip_label:
        from . import binding as BD, integrity as IN, resources as RS
        try:
            s = RN.mismatch_label(cfg, a.out, workers=_workers(a, cfg), accept_changes=a.accept_changes,
                                  retry_failed=a.retry_failed, override_limits=getattr(a, "override_limits", None))
        except (RS.AdmissionRefused, IN.CorruptedRun, BD.IncompatibleRun, IN.RunLocked) as e:
            return _refusal(e)
        print(TX.dumps({k: v for k, v in s.items() if k not in ("per_task_seconds", "run")}))
        if s["exit_code"]:
            _print_completion(s["run"])
            print("the mismatch labels are incomplete: the analysis was not run")
            return int(s["exit_code"])
    if a.dev is None:
        print("labels only (no --dev: the analysis needs the development labels and the frozen recommender)")
        return 0
    print(TX.dumps(RN.mismatch_analyze(cfg, a.out, a.dev)))
    return 0


def cmd_validate(a):
    import numpy as np
    from . import emulate as EM, task as TK
    out = []
    dev = getattr(a, "device", None) or "fake_fez"
    specs = [TK.make_spec("ghz", 3, theta=1.0472, device=dev), TK.make_spec("ising", 4, steps=2, device=dev),
             TK.make_spec("brickwork", 3, layers=2, angle_seed=1, device=dev)][: a.tasks]
    for spec in specs:
        bt = TK.build(spec)
        prep = EM.prepare(bt, ["RAW", "RO", "ZNE-L", "ZNE-R", "PEC"], seed=3)
        for key in ("RAW", "RO", "ZNE-L", "ZNE-R", "PEC"):
            rng = np.random.default_rng([7, len(out)])
            em = EM.trials(prep, key, a.B, a.R, rng).estimates
            di = EM.direct_trials(prep, key, a.B, a.R, rng)
            c = EM.compare_laws(em, di)
            out.append(dict(task=spec.label, protocol=key, B=a.B, **c))
            print(TX.dumps(out[-1]))
    os.makedirs(a.out, exist_ok=True)
    TX.write_json(os.path.join(a.out, "emulator_validation.json"), out)
    bad = [r for r in out if abs(r["z_mean"]) > 4 or r["p_var"] < 1e-3]
    print("validation: %d comparisons, %d outside |z| ≤ 4 or p_var ≥ 1e-3" % (len(out), len(bad)))
    return 0 if not bad else 1


def cmd_freeze(a):
    """Freeze the development choices. Step 1 (no --complete): the model configuration with the lowest out-of-fold
    pooled mean log10 risk, bound to the configuration digest and the development labels' run identity. Step 2
    (--complete), after ``analyze --extras nested_selective`` has evaluated that configuration by nested
    cross-validation: the deployment threshold τ of the selective recommender and the digest of its nested
    evaluation. Each step is done once.

    1.9.0 (second review N3, freeze provenance): a freeze needs complete development labels and a complete, current
    analysis; it records the development data digest and the scientific configuration digest. When the step was
    already done, the existing freeze is *verified* (``binding.verify_freeze``) and never redone: exit 1 (unchanged,
    as before) or 0 with --reuse when it is consistent; 2 when it is inconsistent."""
    from . import binding as BD, design as DS, integrity as IN
    from .. import textio as TX
    import pandas as pd
    cfg = _cfg(a.config)
    adir = os.path.join(a.out, "analysis")
    path = os.path.join(a.out, "freeze.json")
    reuse = bool(getattr(a, "reuse", False))
    if os.path.exists(path):
        frz = TX.read_json(path)
        done = (frz.get("deploy_tau") is not None) if a.complete else True
        if done:
            probs = BD.verify_freeze(a.out, cfg)
            if probs:
                print("the existing freeze is INCONSISTENT with the current files (it is never redone): " +
                      "; ".join(probs))
                return 2
            print("already %s (verified: configuration, development labels and analysis unchanged): %s" % (
                "completed: τ = %s" % frz["deploy_tau"] if a.complete else "frozen", frz["policy"]))
            return 0 if reuse else 1
    try:
        comp = IN.require_complete(os.path.join(a.out, "dev"), "a freeze")
    except (IN.IncompleteRun, IN.CorruptedRun) as e:
        return _refusal(e)
    rj = os.path.join(adir, "results.json")
    res = TX.read_json(rj) if os.path.exists(rj) else {}
    if res.get("data", {}).get("status") == "partial":
        return _refusal("the analysis was made on partial development data (--allow-partial); analyze the complete "
                        "labels (--reset-analysis) before freezing")
    if res.get("identity") and res["identity"].get("data") != (comp["data_digest"] or "")[:16]:
        return _refusal("the analysis is stale: it was made on other development data than the current labels")
    dev_run = BD.load(os.path.join(a.out, "dev"))
    accept = bool(getattr(a, "accept_changes", False))
    if not a.complete and not os.path.exists(path):
        try:                                         # 1.9.1 (independent review F2): before the first freeze
            accepted = _check_analysis_for_freeze(cfg, comp, res, adir, accept)
        except ValueError as e:
            return _refusal(e)
    if a.complete:
        if not os.path.exists(path):
            print("freeze the configuration first (python -m qemx.selection freeze CONFIG --out OUT)")
            return 2
        frz = BD.check_freeze(a.out, cfg)
        op = os.path.join(adir, "nested_operating.csv")
        sm = os.path.join(adir, "nested_summary.csv")
        if not (os.path.exists(op) and os.path.exists(sm)):
            print("run the nested evaluation first (python -m qemx.selection analyze CONFIG --out OUT --splits '' "
                  "--extras nested_selective)")
            return 2
        o = pd.read_csv(op, encoding="utf-8").iloc[0]
        if o.configuration != frz["policy"]:
            print("the nested evaluation is of %s, the frozen configuration is %s" % (o.configuration, frz["policy"]))
            return 2
        from . import run as RN
        probs = RN.nested_reuse_problems(a.out, cfg, required=("nested_selective",))   # 1.9.1 (review F1)
        legacy = [p for p in probs if p.endswith(RN.LEGACY_NO_BINDING)]
        if [p for p in probs if p not in legacy]:
            return _refusal("the nested evaluation cannot complete this freeze: %s. Recompute it for the frozen "
                            "configuration (python -m qemx.selection analyze CONFIG --out OUT --splits '' --extras "
                            "deployed,nested_selective; the pipeline does this itself)" % "; ".join(probs))
        rjn = TX.read_json(rj) if os.path.exists(rj) else {}
        frz.update(deploy_tau=float(o.deploy_tau), selective_rule=str(o.rule),
                   nested_digest=DS.freeze_digest(TX.read_text(sm)), schema="qemx.selection.freeze/2",
                   nested_binding=((rjn.get("extras") or {}).get("nested_selective") or {}).get("binding") or
                   "legacy tables (made before 1.9.1): only their configuration was checked")
        TX.write_json(path, frz, default=str, atomic=True)
        print(TX.dumps(frz, indent=1, default=str))
        return 0
    src = os.path.join(adir, "pooled_same-distribution.csv")             # the reported, out-of-fold pooled scores
    if not os.path.exists(src):
        src = os.path.join(adir, "summary_same-distribution.csv")
    if not os.path.exists(src):
        return _refusal("no same-distribution analysis in %s: run analyze first" % adir)
    s = pd.read_csv(src, encoding="utf-8")
    s = s[(s.B.astype(str) == "all") & s.policy.str.startswith("MODEL[")]
    grid = set(declared_grid(cfg))
    s = s[s.policy.isin(grid)]                       # 1.9.1 (F2): only the declared grid can be chosen
    if s.empty:
        return _refusal("the analysis has no configuration of the declared grid (%s)" % ", ".join(sorted(grid)))
    best = s.groupby("policy").mean_log10_risk.mean().idxmin()
    from ..provenance import source_digest
    frz = dict(schema="qemx.selection.freeze/2", policy=best, config_digest=DS.freeze_digest(cfg),
               scientific_config_digest=DS.scientific_digest(cfg),
               source=source_digest(), chosen_on=os.path.basename(src),
               analysis_digest=DS.freeze_digest(TX.read_text(src)),
               dev_identity=dev_run["identity_digest"] if dev_run else None, dev_data_digest=comp["data_digest"],
               dev_tasks=comp["expected"], analysis_identity=res.get("identity"), deploy_tau=None,
               declared_grid=sorted(grid), **({"accepted": accepted} if accepted else {}))
    TX.write_json(path, frz, default=str, atomic=True)
    print(TX.dumps(frz, indent=1, default=str))
    return 0


def declared_grid(cfg: dict) -> list:
    """The model configurations a selection configuration compares on the same-distribution split (the defaults of
    ``run._analyze`` when the analysis section names none)."""
    a = cfg.get("analysis", {}) or {}
    sets = a.get("feature_sets", ["size_budget", "lightcone", "deployable"])
    fams = a.get("model_families", ["ridge", "enet", "hgb", "mlp"])
    return ["MODEL[%s|%s]" % (f, s) for f in fams for s in sets]


def _check_analysis_for_freeze(cfg: dict, comp: dict, res: dict, adir: str, accept: bool):
    """Before the first freeze (1.9.1; independent review F2): the analysis must be the one this configuration and
    these labels produce — the same analysis section, label-determining configuration and evaluated data
    (``run.analysis_identity``) — it must cover every declared split and every configuration of the declared grid on the
    same-distribution split, and the code that made it must be this code, an accepted code change recorded by
    ``analyze --accept-changes``, or accepted now with ``--accept-changes`` (recorded in the freeze). An analysis made
    before 1.9.0 records no identity: it is frozen only with ``--accept-changes`` (recorded), after the coverage checks.
    Returns what was accepted (None if nothing); raises ValueError with every problem otherwise."""
    import pandas as pd
    from . import run as RN
    from . import binding as BD
    if not res:
        raise ValueError("no analysis in %s: run analyze first" % adir)
    probs, accepted = [], None
    ident = RN.analysis_identity(cfg, comp)
    old = res.get("identity")
    if old is None:
        if not accept:
            probs.append("the analysis records no identity (made before 1.9.0): its settings cannot be checked; "
                         "re-analyze (--reset-analysis), or freeze with --accept-changes to record the acceptance")
        else:
            accepted = dict(legacy_analysis="no recorded identity; coverage of the declared splits and grid checked")
    else:
        for k, what in (("analysis", "analysis settings (the configuration's analysis section)"),
                        ("labels", "label-determining configuration"), ("data", "development data")):
            if old.get(k) != ident[k]:
                probs.append("the analysis was made with other %s than the configuration and labels given now"
                             % what)
        if old.get("code") != ident["code"]:
            recorded = any(c.get("now") == ident["code"] for c in res.get("code_changes", []) or [])
            if not recorded and not accept:
                probs.append("the analysis was made by other code (%s, now %s): re-analyze, or accept the code "
                             "change explicitly with --accept-changes (recorded)" % (old.get("code"), ident["code"]))
            elif not recorded:
                accepted = dict(code_change=dict(analysis=old.get("code"), now=ident["code"]))
    a = cfg.get("analysis", {}) or {}
    missing_splits = [k for k in a.get("splits", ["same-distribution"]) if k not in (res.get("evaluations") or {})]
    if "same-distribution" not in (res.get("evaluations") or {}) and "same-distribution" not in missing_splits:
        missing_splits.append("same-distribution")
    if missing_splits:
        probs.append("declared splits not in the analysis: %s" % ", ".join(missing_splits))
    src = os.path.join(adir, "pooled_same-distribution.csv")
    if os.path.exists(src):
        q = pd.read_csv(src, encoding="utf-8")
        have = set(q[q.B.astype(str) == "all"].policy)
        lack = [g for g in declared_grid(cfg) if g not in have]
        if lack:
            probs.append("the same-distribution comparison lacks configurations of the declared grid: %s"
                         % ", ".join(lack))
    else:
        probs.append("no pooled same-distribution table (pooled_same-distribution.csv)")
    if probs:
        raise ValueError("the analysis cannot be frozen with this configuration: " + "; ".join(probs))
    return accepted


def _train_on(out_dir: str, parts) -> object:
    """The decision dataset of a run's complete labels directories ``parts`` (e.g. dev and the historical final)."""
    from . import binding as BD, dataset as DT
    ds = None
    for p in parts:
        d, _ = BD.collect(os.path.join(out_dir, p), require_complete=True)
        ds = d if ds is None else DT.concat(ds, d)
    return ds


def _frozen_policies(frz, PO, FE):
    fam, sname = frz["policy"][len("MODEL["):-1].split("|")
    pols = [PO.Raw(), PO.SBS(), PO.SBSBudget(), PO.SizeBudget(), PO.Model(fam, FE.SETS[sname], sname)]
    for f2 in ("ridge", "struct", "hgb"):              # secondary comparators, declared before confirmation (D39)
        if (f2, "deployable") != (fam, sname):
            pols.append(PO.Model(f2, FE.SETS["deployable"], "deployable"))
    if frz.get("deploy_tau") is not None:
        pols.append(PO.SelectiveMargin(fam, FE.SETS[sname], sname, tau=float(frz["deploy_tau"])))
    pols.append(PO.AllCandidate())
    return pols


def _logical_ids(out_dir: str, parts) -> set:
    """Logical circuit identities of every task record in the labels directories ``parts`` of a run."""
    from . import ancestry as AY, labels as LB, task as TK
    ids = set()
    for p in parts:
        d = os.path.join(out_dir, p)
        for tid in LB.task_ids(d):
            ids.add(AY.logical_id(TK.TaskSpec.from_json(LB.load_record(d, tid)["spec"])))
    return ids


def _output_digests(d: str, names) -> dict:
    from . import labels as LB
    return {n: LB.file_digest(os.path.join(d, n)) for n in names if os.path.exists(os.path.join(d, n))}


def _evaluated_once(done: str, frz: dict, labels_dir: str, reuse: bool, what: str, cfg: dict | None = None,
                    train: str | None = None) -> int:
    """An already evaluated once-only set: verified against the current freeze and labels, never evaluated again.
    Exit 1 (unchanged), 0 with --reuse when consistent, 2 when inconsistent. 1.9.0 (independent verification): also
    the evaluated configuration, the training parts and the saved evaluation outputs, when the record names them."""
    from . import design as DS
    from . import integrity as IN
    ev = TX.read_json(done)
    probs = []
    if cfg is not None and ev.get("evaluation_config_digest") and \
            ev["evaluation_config_digest"] != DS.scientific_digest(cfg):
        probs.append("another configuration was evaluated (scientific digest %s…, given %s…)"
                     % (ev["evaluation_config_digest"][:12], DS.scientific_digest(cfg)[:12]))
    if train is not None and ev.get("train_parts") is not None and str(ev["train_parts"]) != str(train):
        probs.append("it was trained on %s, not %s" % (ev["train_parts"], train))
    for n, h in (ev.get("outputs") or {}).items():
        now = _output_digests(labels_dir, [n]).get(n)
        if now != h:
            probs.append("the saved evaluation output %s %s" % (n, "is missing" if now is None else "changed"))
    for k in ("policy", "deploy_tau"):
        if ev.get(k) != frz.get(k):
            probs.append("%s differs (evaluated %s, frozen %s)" % (k, ev.get(k), frz.get(k)))
    if ev.get("freeze") and ev["freeze"].get("dev_identity") != frz.get("dev_identity"):
        probs.append("the freeze's development identity differs")
    if ev.get("data_digest"):
        now = IN.data_digest(labels_dir)
        if now != ev["data_digest"]:
            probs.append("the labels changed after the evaluation")
    if probs:
        print("the %s was evaluated once with ANOTHER freeze or other labels (never re-evaluated): %s"
              % (what, "; ".join(probs)))
        return 2
    print("the %s was already evaluated once (verified against the current freeze%s); its results are in %s"
          % (what, " and labels" if ev.get("data_digest") else "", labels_dir))
    return 0 if reuse else 1


def cmd_final(a):
    """Label and evaluate the final-test tasks once, with the frozen choices. Refused when the freeze does not match
    the configuration or the development labels, when the development labels are incomplete, or when a final-test
    logical circuit also occurs in the development set."""
    import numpy as np
    import pandas as pd
    from . import ancestry as AY, binding as BD, design as DS, evaluate as EV, features as FE, policies as PO
    from . import run as RN
    from .. import textio as TX
    cfg = _cfg(a.config)
    try:
        frz = BD.check_freeze(a.out, cfg)
        dev = _train_on(a.out, ["dev"])
    except BD.IncompatibleRun as e:
        print(str(e))
        return 2
    done = os.path.join(a.out, "final", "EVALUATED")
    if os.path.exists(done):
        return _evaluated_once(done, frz, os.path.join(a.out, "final"), bool(getattr(a, "reuse", False)), "final test",
                               cfg=cfg)
    specs = DS.expand(cfg)
    dv, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    shared = {AY.logical_id(s) for s in fin} & {AY.logical_id(s) for s in dv}
    if shared:
        print("refused: %d final-test logical circuits also occur in the development set" % len(shared))
        return 2
    from . import integrity as IN, resources as RS
    try:
        s = RN.label(cfg, a.out, workers=_workers(a, cfg), final=True, accept_changes=a.accept_changes,
                     retry_failed=a.retry_failed, override_limits=getattr(a, "override_limits", None))
    except (RS.AdmissionRefused, IN.CorruptedRun, BD.IncompatibleRun, IN.RunLocked) as e:
        return _refusal(e)
    print(TX.dumps({k: v for k, v in s.items() if k not in ("per_task_seconds", "preflight", "run", "supervision")}))
    if s["exit_code"]:
        _print_completion(s["run"], s.get("stopped"))
        print("the final-test labels are incomplete: nothing was evaluated (rerun the same command to resume)")
        return int(s["exit_code"])
    fin_ds, _ = BD.collect(os.path.join(a.out, "final"), require_complete=True)
    from . import dataset as DT
    allds = DT.concat(dev, fin_ds)
    tr = np.arange(len(dev.rows))
    te = np.arange(len(dev.rows), len(allds.rows))
    ev = EV.evaluate(allds, tr, te, _frozen_policies(frz, PO, FE), reps=int(cfg.get("analysis", {}).get("bootstrap",
                                                                                                         1000)))
    pd.DataFrame(ev["summary"]).to_csv(os.path.join(a.out, "final", "summary_final.csv"), index=False, encoding="utf-8")
    pd.DataFrame(ev["rows"]).to_csv(os.path.join(a.out, "final", "choices_final.csv"), index=False, encoding="utf-8")
    TX.write_json(done, dict(policy=frz["policy"], deploy_tau=frz.get("deploy_tau"), rows=len(te), freeze=frz,
                             data_digest=IN.data_digest(os.path.join(a.out, "final")),
                             evaluation_config_digest=DS.scientific_digest(cfg), train_parts="dev",
                             outputs=_output_digests(os.path.join(a.out, "final"),
                                                     ["summary_final.csv", "choices_final.csv"])))
    print("final test evaluated once: %d rows" % len(te))
    return 0


def cmd_confirm(a):
    """The once-only confirmation (1.7.0): label the confirmation configuration's tasks into OUT/confirm and evaluate
    the frozen choices, trained on the development labels and the historical final test (--train dev,final), on them.
    Refused unless the freeze matches the development configuration (--dev-config) and is complete (τ), the training
    labels are complete, and no confirmation circuit occurs in any other study of the ancestry registry."""
    import numpy as np
    import pandas as pd
    from . import ancestry as AY, binding as BD, dataset as DT, design as DS, evaluate as EV, features as FE
    from . import labels as LB, policies as PO, run as RN
    from .. import textio as TX
    cfg = _cfg(a.config)
    dcfg = _cfg(a.dev_config)
    try:
        frz = BD.check_freeze(a.out, dcfg)
    except BD.IncompatibleRun as e:
        print(str(e))
        return 2
    if frz.get("deploy_tau") is None:
        print("complete the freeze first (freeze --complete): the selective threshold is part of the frozen policy")
        return 2
    cdir = os.path.join(a.out, "confirm")
    done = os.path.join(cdir, "EVALUATED")
    if os.path.exists(done):
        return _evaluated_once(done, frz, cdir, bool(getattr(a, "reuse", False)), "confirmation set", cfg=cfg,
                               train=a.train)
    specs = DS.expand(cfg)
    try:
        AY.check_confirmation(AY.load(), specs, cfg["name"])
    except ValueError as e:
        print("refused: %s" % e)
        return 2
    parts = [p for p in a.train.split(",") if p]
    try:
        train = _train_on(a.out, parts)
    except BD.IncompatibleRun as e:
        print(str(e))
        return 2
    shared = _logical_ids(a.out, parts) & {AY.logical_id(s_) for s_ in specs}
    if shared:                              # 1.9.0: against the actual training data, not only the registry
        print("refused: %d confirmation logical circuits also occur in the training labels (%s)"
              % (len(shared), ", ".join(sorted(shared)[:5])))
        return 2
    from . import integrity as IN, resources as RS
    try:
        s = RN.label(cfg, a.out, workers=_workers(a, cfg), subdir="confirm", accept_changes=a.accept_changes,
                     retry_failed=a.retry_failed, override_limits=getattr(a, "override_limits", None))
    except (RS.AdmissionRefused, IN.CorruptedRun, BD.IncompatibleRun, IN.RunLocked) as e:
        return _refusal(e)
    print(TX.dumps({k: v for k, v in s.items() if k not in ("per_task_seconds", "preflight", "run", "supervision")}))
    if s["exit_code"]:
        _print_completion(s["run"], s.get("stopped"))
        print("the confirmation labels are incomplete: nothing was evaluated (rerun the same command to resume)")
        return int(s["exit_code"])
    conf, rep = BD.collect(cdir, require_complete=True)
    allds = DT.concat(train, conf)
    tr = np.arange(len(train.rows))
    te = np.arange(len(train.rows), len(allds.rows))
    ev = EV.evaluate(allds, tr, te, _frozen_policies(frz, PO, FE), reps=int(dcfg.get("analysis", {}).get(
        "bootstrap", 1000)))
    pd.DataFrame(ev["summary"]).to_csv(os.path.join(cdir, "summary_confirm.csv"), index=False, encoding="utf-8")
    pd.DataFrame(ev["rows"]).to_csv(os.path.join(cdir, "choices_confirm.csv"), index=False, encoding="utf-8")
    lab = LB.summarize(cdir)
    lab.to_csv(os.path.join(cdir, "labels.csv"), index=False, encoding="utf-8")
    pd.DataFrame(EV.pooled(pd.DataFrame(ev["rows"]), lab, int(dcfg.get("analysis", {}).get("bootstrap", 1000)))) \
        .to_csv(os.path.join(cdir, "metrics_confirm.csv"), index=False, encoding="utf-8")
    TX.write_json(done, dict(policy=frz["policy"], deploy_tau=frz["deploy_tau"], rows=len(te),
                             train_parts=a.train, train_rows=len(tr), train_base_circuits=len(set(train.groups)),
                             confirm_base_circuits=len(set(conf.groups)), freeze=frz,
                             data_digest=IN.data_digest(cdir, sorted(set(conf.col("task_id").tolist()))),
                             evaluation_config_digest=DS.scientific_digest(cfg),
                             outputs=_output_digests(cdir, ["summary_confirm.csv", "choices_confirm.csv",
                                                            "metrics_confirm.csv", "labels.csv"])))
    print("confirmation evaluated once: %d rows, %d base circuits" % (len(te), len(set(conf.groups))))
    return 0


def cmd_regimes(a):
    """Label a controlled study (the regime study, or with --subdir another one such as noisecomp) into OUT/SUBDIR
    (resumable) and analyze it with OUT's frozen recommender."""
    from . import run as RN
    cfg = _cfg(a.config)
    sub_ = a.subdir or "regimes"
    if not a.skip_label:
        from . import binding as BD, integrity as IN, resources as RS
        try:
            s = RN.label(cfg, a.out, workers=_workers(a, cfg), subdir=sub_, accept_changes=a.accept_changes,
                         retry_failed=a.retry_failed, override_limits=getattr(a, "override_limits", None))
        except (RS.AdmissionRefused, IN.CorruptedRun, BD.IncompatibleRun, IN.RunLocked) as e:
            return _refusal(e)
        print(TX.dumps({k: v for k, v in s.items() if k not in ("per_task_seconds", "preflight", "run",
                                                                 "supervision")}))
        if s["exit_code"]:
            _print_completion(s["run"], s.get("stopped"))
            print("the %s labels are incomplete: the analysis was not run" % sub_)
            return int(s["exit_code"])
    print(TX.dumps(RN.regime_analyze(cfg, a.out, sub=sub_)))
    return 0


def cmd_profile(a):
    """Classical cost of a recommendation against width — compilation, deployable features, simulator-only
    augmentation, inference — with label generation for comparison (writes OUT/analysis/runtime.csv)."""
    import pandas as pd
    from . import run as RN
    _cfg(a.config)                                   # 1.9.0: the configuration named is valid (else refused)
    widths = [int(w) for w in (a.widths or "3,4,5,6,7").split(",")]
    try:
        rows = RN.profile(a.out, widths=widths, bundle=getattr(a, "bundle", None))
    except ValueError as e:                          # e.g. no labeled development tasks to build a bundle from
        return _refusal(e)
    os.makedirs(os.path.join(a.out, "analysis"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "analysis", "runtime.csv"), index=False, encoding="utf-8")
    print(pd.DataFrame(rows).groupby("n")[["compile_s", "deployable_s", "simulated_s", "inference_s",
                                          "label_s"]].median())
    return 0


def cmd_report(a):
    from . import report as RP
    _cfg(a.config)                                   # 1.9.0: the configuration named is valid (else refused)
    if not os.path.isdir(os.path.join(a.out, "analysis")):
        return _refusal("%s has no analysis yet: run analyze first" % a.out)
    print(RP.write(a.out, os.path.join(a.out, "RESULTS_TABLES.md")))
    return 0


def cmd_run(a):
    """label + analyze + figures. 1.9.0 (second review N1): an incomplete labeling stops here with its exit code
    unless --allow-partial (then the analysis is marked partial); every stage's failure is returned."""
    rc = cmd_label(a)
    if rc == 2 or (rc and not getattr(a, "allow_partial", False)):
        print("labeling is incomplete (exit %d): analysis and figures were not run. Finish labeling (see above), or "
              "analyze the labeled part explicitly with --allow-partial." % rc)
        return rc
    rc2 = cmd_analyze(a)
    if rc2:
        return rc2
    return max(rc, cmd_figures(a))


def cmd_bundle(a):
    """Build the versioned model bundle of the frozen recommender: trained on the run's development labels (--train),
    with the frozen selective threshold and the nested-CV validation summary; written to --bundle-out (default
    RUN/bundles/<version> since 1.9.0; never over an existing bundle)."""
    import pandas as pd
    from . import binding as BD, deploy as DP
    from .. import textio as TX
    cfg = _cfg(a.config)
    parts = [p for p in a.train.split(",") if p]
    missing = [p for p in parts if not os.path.isdir(os.path.join(a.out, p, "tasks"))]
    if missing:                                      # independent verification F15: a refusal, not a traceback
        return _refusal("--train %s: no labels directory %s in %s (a run without a final test set: --train dev)"
                        % (a.train, ", ".join(missing), a.out))
    try:
        frz = BD.check_freeze(a.out, cfg)
        ds = _train_on(a.out, parts)
    except BD.IncompatibleRun as e:
        print(str(e))
        return 2
    except ValueError as e:                          # e.g. a labels directory without labeled tasks
        return _refusal(e)
    fam, sname = frz["policy"][len("MODEL["):-1].split("|")
    if a.feature_set:
        sname = a.feature_set
    adir = os.path.join(a.out, "analysis")
    conf_name = "MODEL[%s|%s]" % (fam, sname)
    frozen = conf_name == frz["policy"]
    val = {"configuration": conf_name}
    # 1.7.0: every number in a bundle's validation belongs to the bundle's own configuration (or names the one it
    # belongs to): out-of-fold development results, the nested selective evaluation (frozen configuration only), and
    # the once-only confirmation when it exists
    po = os.path.join(adir, "pooled_same-distribution.csv")
    if os.path.exists(po):
        q = pd.read_csv(po, encoding="utf-8")
        val["out_of_fold"] = [r for r in q[q.B.astype(str) == "all"].to_dict("records")
                              if r["policy"] in (conf_name, "SBS-B", "ALL-CANDIDATE")]
    sm = os.path.join(adir, "nested_summary.csv")
    if os.path.exists(sm) and frozen:
        val["nested_cv"] = pd.read_csv(sm, encoding="utf-8").to_dict("records")
    elif os.path.exists(sm):
        val["nested_cv_note"] = ("the nested selective evaluation was run for the frozen configuration %s, not for this "
                                 "one; this bundle has no validated selective threshold, so its recommendation is the "
                                 "model's choice" % frz["policy"])
    cf = os.path.join(a.out, "confirm", "metrics_confirm.csv")
    if os.path.exists(cf):
        q = pd.read_csv(cf, encoding="utf-8")
        keep = {conf_name, "SBS-B", "ALL-CANDIDATE"} | ({"SELECTIVE[%s|%s]" % (fam, sname)} if frozen else set())
        val["confirmation"] = [r for r in q[q.B.astype(str) == "all"].to_dict("records") if r["policy"] in keep]
    version = a.version or "qemx-rec-%s" % __import__("qemx").__version__
    out = a.bundle_out or os.path.join(a.out, "bundles", version)      # 1.9.0: in the run, never the shipped folder
    if os.path.exists(os.path.join(out, "bundle.json")):
        old = TX.read_json(os.path.join(out, "bundle.json"))
        same = (old.get("configuration") == "MODEL[%s|%s]" % (fam, sname)
                and ((old.get("training") or {}).get("freeze") or {}) == frz
                and (old.get("training") or {}).get("parts") == a.train)
        if same:
            print("bundle %s exists and was built from this freeze, configuration and training labels: kept" % out)
            return 0
        print("refused: %s holds another bundle (configuration %s); bundles are never overwritten — choose another "
              "--bundle-out" % (out, old.get("configuration")))
        return 2
    tau = frz.get("deploy_tau") if sname == frz["policy"][len("MODEL["):-1].split("|")[1] else None
    import numpy as np
    b = DP.build_bundle(ds, np.arange(len(ds.rows)), fam, sname, out, version=version, tau=tau, validation=val,
                        training=dict(run=os.path.basename(os.path.normpath(a.out)), parts=a.train, freeze=frz,
                                      run_identities={p: (BD.load(os.path.join(a.out, p)) or {}).get("identity_digest")
                                                      for p in a.train.split(",") if p}))
    print(TX.dumps(dict(bundle=out, version=version, configuration=b.meta["configuration"], tau=tau,
                          verification=b.meta["verification"]), indent=1))
    del TX
    return 0


def cmd_recommend(a):
    """Recommend a protocol for one experiment: an app state (--state, the Playground's export) or an experiment
    document (--experiment), at budget --budget. ``--device`` (1.9.0) is the calibration snapshot and replaces the
    state's backend, the same rule as the Playground's "Device snapshot" (``deploy.from_state``); without it the state's
    backend must be a fake device. Prints the same recommendation the Playground shows. Exit 0 a recommendation
    (accepted or fallback), 3 unsupported input (with the reason), 2 usage errors."""
    from . import deploy as DP
    from .. import experiment as EX
    from .. import textio as TX
    if bool(a.state) == bool(a.experiment):
        print("give exactly one of --state or --experiment")
        return 2
    try:
        state = TX.read_json(a.state) if a.state else EX.to_state(EX.load(a.experiment))
    except (OSError, ValueError) as e:              # 1.9.0: unreadable or invalid input is reported, not a traceback
        print(TX.dumps(dict(status="unsupported", reason="cannot read the experiment: %s" % e), indent=1))
        return 3
    names = DP.bundles()
    bpath = a.bundle or (os.path.join(DP.BUNDLE_DIR, names[0]) if names else None)
    if not bpath:
        print("no model bundle installed; give --bundle")
        return 2
    b = DP.load_bundle(bpath)
    try:
        task = DP.from_state(state, int(a.budget), a.device)
        rec = DP.recommend(task, b, allow_simulation=bool(a.allow_simulation))
    except DP.Unsupported as e:
        rec = dict(status="unsupported", reason=str(e))
    print(TX.dumps(rec, indent=1, default=str))
    return 0 if rec.get("status") != "unsupported" else 3


def cmd_hardware(a):
    """Hardware validation plan of the recommender (plan only; execution needs the owner's authorization of the plan's
    digest, a campaign ledger and credentials — never submitted automatically)."""
    from . import hardware as HWS
    from .. import textio as TX
    if a.action == "plan":
        cfg = _cfg(a.config) if a.config else {}
        if a.hardware_config:
            cfg = dict(cfg, hardware=TX.read_yaml(a.hardware_config))
        ledger = None
        if a.ledger:
            from .. import campaign as CG
            ledger = CG.open_ledger(a.ledger)
        p = HWS.plan(cfg, a.bundle, ledger=ledger)
        path = HWS.write(p, a.out)
        print(TX.dumps({k: p[k] for k in ("digest", "budget", "blocks", "remaining_s", "reserved_s", "admitted_jobs",
                                             "total_jobs", "fits", "note")}, indent=1))
        print("written: %s" % path)
        return 0
    p = TX.read_json(a.plan)
    try:
        HWS.require_authorization(p)
    except HWS.NotAuthorized as e:
        print("refused: %s" % e)
        return 3
    print("authorized plan %s…: submission goes through the approved IBM workflow with the campaign ledger "
          "(pages/7_IBM_Quantum.py or scripts/hardware_transfer.py); this command never submits jobs."
          % p["digest"][:16])
    return 0


def cmd_check_config(a):
    """Validate any YAML file of the project with its own parser (1.9.0; second review N6): selection
    configurations (schema + semantic checks + expansion report), phase manifests, divergence manifests, experiment
    documents, plotting configurations, pipelines and hardware sections. Exit 0 valid, 2 invalid."""
    from . import formats as FM
    rc = 0
    for path in a.files:
        r = FM.check(path)
        print("%s: %s (%s; parser %s)" % (path, "OK" if r["ok"] else "INVALID", r["format"], r["parser"]))
        for x in r.get("errors", []):
            print("  ERROR: " + x)
        for x in r.get("warnings", []):
            print("  warning: " + x)
        if r.get("summary"):
            print("  " + r["summary"])
        if r.get("command"):
            print("  run with: " + r["command"])
        rc = rc or (0 if r["ok"] else 2)
    return rc


def _terminate_cleanly() -> None:
    """1.9.0: SIGTERM (and Ctrl+Break on Windows) end the command like Ctrl+C — running labeling children are stopped
    and the run's lock is released (``integrity.RunLock``) — instead of leaving them behind."""
    import signal
    import threading
    old = {}
    if threading.current_thread() is not threading.main_thread():
        return old

    def _term(signum, frame):
        raise SystemExit(128 + int(signum))
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                old[getattr(signal, name)] = signal.signal(getattr(signal, name), _term)
            except (ValueError, OSError):
                pass
    return old


def _restore_signals(old: dict) -> None:
    import signal
    for sig, h in old.items():
        try:
            signal.signal(sig, h)
        except (ValueError, OSError, TypeError):
            pass


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    argv = list(sys.argv[1:] if argv is None else argv)
    saved = _terminate_cleanly()
    try:
        return _main(argv)
    finally:
        _restore_signals(saved)


def _main(argv) -> int:
    if argv[:1] == ["pipeline"]:       # 1.9.0: the stage-aware runner has its own options (qemx.selection.pipeline)
        from . import pipeline as PL
        return PL.main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m qemx.selection", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("label", cmd_label), ("analyze", cmd_analyze), ("figures", cmd_figures),
                     ("mismatch", cmd_mismatch), ("freeze", cmd_freeze), ("final", cmd_final), ("report", cmd_report),
                     ("run", cmd_run), ("regimes", cmd_regimes), ("profile", cmd_profile),
                     ("confirm", cmd_confirm)):
        p = sub.add_parser(name)
        p.add_argument("config")
        if name != "plan":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--out", default=None, help="plan: the run directory (its finished tasks give measured "
                           "rates and the remaining work)")
            p.add_argument("--tasks-csv", default=None, help="plan: write the expanded task table (every task with "
                           "its family, width, input, observable, layout, device, noise modifiers, split and ancestry)")
            p.add_argument("--json", default=None, help="plan: write the estimate, expansion and preflight as JSON")
        p.add_argument("--workers", default="1", help="parallel labeling processes, or 'auto' (CPUs − 1, limited by "
                       "memory; see plan)")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--measured", default=None, help="labels directory of an executed run (e.g. runs/pilot/dev): "
                       "per-width preparation and trial times for the CPU estimate")
        p.add_argument("--dev", default=None)
        p.add_argument("--skip-label", action="store_true")
        p.add_argument("--accept-changes", action="store_true", help="label: resume a run whose code or environment "
                       "changed (the change is recorded in run.json); without it such a resume is refused")
        p.add_argument("--retry-failed", action="store_true", help="label: run tasks recorded as error, timeout or "
                       "memory again")
        p.add_argument("--subdir", default=None, help="label: label this configuration's tasks into OUT/SUBDIR (a "
                       "separate study bound to its own run, e.g. regimes), without analysis")
        p.add_argument("--final-set", action="store_true", help="label: label the final-test tasks into OUT/final "
                       "without evaluating them (the evaluation stays once-only: python -m qemx.selection final)")
        p.add_argument("--splits", default=None, help="analyze: only these splits (comma-separated; '' for none)")
        p.add_argument("--extras", default=None, help="analyze: only these extra analyses ('' for none)")
        p.add_argument("--widths", default=None, help="profile: widths to time (comma-separated)")
        p.add_argument("--plot-config", default=None, help="figures: plotting configuration (default "
                       "configs/selection/figures.yaml)")
        p.add_argument("--only", default=None, help="figures: draw only these figures (comma-separated names)")
        p.add_argument("--complete", action="store_true", help="freeze: add the nested-CV selective threshold")
        p.add_argument("--dev-config", default=None, help="confirm: the development configuration that was frozen")
        p.add_argument("--train", default="dev,final", help="confirm: labels directories of OUT to train on")
        p.add_argument("--allow-partial", action="store_true", help="analyze/run: analyze incomplete labels "
                       "(exploratory; results.json says partial and a freeze refuses it)")
        p.add_argument("--reset-analysis", action="store_true", help="analyze: archive an analysis directory made "
                       "from another configuration, other data or other code (analysis.superseded-<time>)")
        p.add_argument("--override-limits", default=None, metavar="REASON", help="label: start although a "
                       "configured cap is exceeded by the estimate; the reason is recorded in run.json")
        p.add_argument("--quarantine-corrupted", action="store_true", help="label: move corrupted records to "
                       "quarantine/ (listed in run.json) and relabel them")
        p.add_argument("--reuse", action="store_true", help="freeze/final/confirm: when the once-only step was "
                       "already done and verifies against the current files, exit 0 instead of 1")
        p.add_argument("--figures-out", default=None, help="figures: write into this folder instead of OUT/figures")
        p.add_argument("--bundle", default=None, help="profile: the bundle to time (default: one built from this run)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("check-config", help="validate YAML files with the parser of their format")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_check_config)
    sub.add_parser("pipeline", help="stage-aware resumable study pipeline: python -m qemx.selection pipeline FILE "
                   "--out DIR (see configs/pipelines)")
    p = sub.add_parser("status", help="completion of a run's labels directories (exit 0 when all are complete)")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default=None, help="one labels directory (dev, final, confirm, regimes, …)")
    p.add_argument("--check", default="digest", choices=["exists", "digest", "load"],
                   help="how arrays are checked (default: their digest)")
    p.add_argument("--json", default=None)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("bundle", help="build the versioned model bundle of the frozen recommender")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--train", default="dev,final")
    p.add_argument("--version", default=None)
    p.add_argument("--bundle-out", default=None)
    p.add_argument("--feature-set", default=None, help="another feature set of the frozen family (e.g. deployable, "
                   "for a recommender that needs no simulation); its bundle carries no selective threshold")
    p.set_defaults(fn=cmd_bundle)
    p = sub.add_parser("recommend", help="recommend a protocol for one experiment (the Playground's recommender)")
    p.add_argument("--state", default=None, help="an app state (JSON, the Playground's export)")
    p.add_argument("--experiment", default=None, help="an experiment document (YAML/JSON, qemx.experiment)")
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--device", default=None, help="fake device snapshot; overrides the state's backend (1.9.0; the "
                   "physical layout is checked on it)")
    p.add_argument("--bundle", default=None)
    p.add_argument("--allow-simulation", action="store_true")
    p.set_defaults(fn=cmd_recommend)
    p = sub.add_parser("hardware", help="hardware validation plan (never submits automatically)")
    p.add_argument("action", choices=["plan", "check"])
    p.add_argument("config", nargs="?", default=None)
    p.add_argument("--hardware-config", default=None, help="YAML with the hardware section (tasks, budget, blocks, "
                   "ceiling_s, reserve_s, protocols, device)")
    p.add_argument("--bundle", default=None)
    p.add_argument("--ledger", default=None)
    p.add_argument("--plan", default=None)
    p.add_argument("--out", default="runs/hardware_plan",
                   help="output folder (1.8.0: no longer the frozen reports/selection_v170/hardware by default)")
    p.set_defaults(fn=cmd_hardware)
    p = sub.add_parser("registry", help="the candidate registry: capability records and the method inventory")
    p.add_argument("--markdown", action="store_true", help="print docs/selection/CANDIDATES.md")
    p.add_argument("--candidate", default=None, help="one candidate's capability record")
    p.set_defaults(fn=cmd_registry)
    p = sub.add_parser("validate")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", type=int, default=3)
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--R", type=int, default=300)
    p.add_argument("--device", default="fake_fez", help="validate: device, e.g. fake_fez or fake_fez@s=2 (a scaled "
                                                          "snapshot as in the regime study)")
    p.set_defaults(fn=cmd_validate)
    a = ap.parse_args(argv)
    from . import binding as BD, design as DS, integrity as IN, replay as RP, resources as RS, run as RN
    try:
        return a.fn(a)
    except (DS.ConfigError, IN.RunLocked, IN.CorruptedRun, IN.IncompleteRun, BD.IncompatibleRun, RN.StaleAnalysis,
            RP.StaleTable, RS.AdmissionRefused) as e:        # 1.9.0: every refusal is exit 2 with its reason,
        return _refusal(e)                                    # never a traceback (independent verification F5)


if __name__ == "__main__":
    sys.exit(main())
