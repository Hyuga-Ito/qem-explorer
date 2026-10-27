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

Nothing here contacts a quantum provider.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _cfg(path):
    from . import design as DS
    return DS.load(path)


def cmd_plan(a):
    from . import design as DS, run as RN
    cfg = _cfg(a.config)
    specs = DS.expand(cfg)
    dev, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    est = RN.estimate(cfg, measured=RN.measured_rates(a.measured) if a.measured else None)
    print(json.dumps(dict(est, development=len(dev), final=len(fin)), indent=1))
    for r in DS.design_table(specs):
        print(r)
    return 0


def cmd_label(a):
    from . import run as RN
    s = RN.label(_cfg(a.config), a.out, workers=a.workers, limit=a.limit, accept_changes=a.accept_changes,
                 retry_failed=a.retry_failed, final=bool(getattr(a, "final_set", False)),
                 subdir=getattr(a, "subdir", None))
    print(json.dumps({k: v for k, v in s.items() if k != "per_task_seconds"}))
    return 0 if s["errors"] == 0 else 1


def cmd_analyze(a):
    from . import run as RN
    r = RN.analyze(a.out, _cfg(a.config), only_splits=a.splits.split(",") if a.splits is not None else None,
                   only_extras=a.extras.split(",") if a.extras is not None else None)
    print(json.dumps({k: v for k, v in r.items() if k != "evaluations"}, indent=1, default=str))
    return 0


def cmd_figures(a):
    """The six main figures, the supplementary figures and the tables, from the saved tables of OUT only. A figure
    whose input tables are missing is reported as pending (not drawn); nothing is simulated."""
    from . import figures as FG, plots as PL, run as RN
    cfg = _cfg(a.config)
    adir = os.path.join(a.out, "analysis")
    fdir = os.path.join(a.out, "figures")
    if any(f.startswith("choices_") and not os.path.exists(os.path.join(adir, "pooled_" + f[len("choices_"):]))
           for f in os.listdir(adir)):
        RN.pool(adir, reps=int(cfg.get("analysis", {}).get("bootstrap", 1000)))   # tables of an older analysis
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
        tables=["manifest.json"])
    counts = PL.run_counts(a.out)
    fn = dict(supp_coverage_certificates=lambda: FG.supp_coverage_certificates(adir, fdir),
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
                                          else os.path.exists(os.path.join(a.out, r)))]
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
    print(json.dumps(dict(done=done, failed=failed, pending=pending), indent=1))
    return 0 if not failed else 1


def cmd_mismatch(a):
    from . import run as RN
    cfg = _cfg(a.config)
    if not a.skip_label:
        print(json.dumps(RN.mismatch_label(cfg, a.out, workers=a.workers, accept_changes=a.accept_changes)))
    if a.dev is None:
        print("labels only (no --dev: the analysis needs the development labels and the frozen recommender)")
        return 0
    print(json.dumps(RN.mismatch_analyze(cfg, a.out, a.dev)))
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
            print(json.dumps(out[-1]))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "emulator_validation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    bad = [r for r in out if abs(r["z_mean"]) > 4 or r["p_var"] < 1e-3]
    print("validation: %d comparisons, %d outside |z| ≤ 4 or p_var ≥ 1e-3" % (len(out), len(bad)))
    return 0 if not bad else 1


def cmd_freeze(a):
    """Freeze the development choices. Step 1 (no --complete): the model configuration with the lowest out-of-fold
    pooled mean log10 risk, bound to the configuration digest and the development labels' run identity. Step 2
    (--complete), after ``analyze --extras nested_selective`` has evaluated that configuration by nested
    cross-validation: the deployment threshold τ of the selective recommender and the digest of its nested
    evaluation. Each step is done once."""
    from . import binding as BD, design as DS
    from .. import textio as TX
    import pandas as pd
    cfg = _cfg(a.config)
    adir = os.path.join(a.out, "analysis")
    path = os.path.join(a.out, "freeze.json")
    dev_run = BD.load(os.path.join(a.out, "dev"))
    if a.complete:
        if not os.path.exists(path):
            print("freeze the configuration first (python -m qemx.selection freeze CONFIG --out OUT)")
            return 2
        frz = BD.check_freeze(a.out, cfg)
        if frz.get("deploy_tau") is not None:
            print("already completed: τ = %s" % frz["deploy_tau"])
            return 1
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
        frz.update(deploy_tau=float(o.deploy_tau), selective_rule=str(o.rule),
                   nested_digest=DS.freeze_digest(TX.read_text(sm)), schema="qemx.selection.freeze/2")
        TX.write_json(path, frz, default=str)
        print(json.dumps(frz, indent=1, default=str))
        return 0
    src = os.path.join(adir, "pooled_same-distribution.csv")             # the reported, out-of-fold pooled scores
    if not os.path.exists(src):
        src = os.path.join(adir, "summary_same-distribution.csv")
    s = pd.read_csv(src, encoding="utf-8")
    s = s[(s.B.astype(str) == "all") & s.policy.str.startswith("MODEL[")]
    best = s.groupby("policy").mean_log10_risk.mean().idxmin()
    from ..provenance import source_digest
    frz = dict(schema="qemx.selection.freeze/2", policy=best, config_digest=DS.freeze_digest(cfg),
               source=source_digest(), chosen_on=os.path.basename(src),
               analysis_digest=DS.freeze_digest(TX.read_text(src)),
               dev_identity=dev_run["identity_digest"] if dev_run else None, deploy_tau=None)
    if os.path.exists(path):
        print("already frozen: %s" % TX.read_json(path)["policy"])
        return 1
    TX.write_json(path, frz, default=str)
    print(json.dumps(frz, indent=1, default=str))
    return 0


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
        print("the final test was already evaluated once; its results are in %s" % os.path.dirname(done))
        return 1
    specs = DS.expand(cfg)
    dv, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    shared = {AY.logical_id(s) for s in fin} & {AY.logical_id(s) for s in dv}
    if shared:
        print("refused: %d final-test logical circuits also occur in the development set" % len(shared))
        return 2
    s = RN.label(cfg, a.out, workers=a.workers, final=True, accept_changes=a.accept_changes)
    print(json.dumps({k: v for k, v in s.items() if k != "per_task_seconds"}))
    fin_ds, _ = BD.collect(os.path.join(a.out, "final"), require_complete=True)
    from . import dataset as DT
    allds = DT.concat(dev, fin_ds)
    tr = np.arange(len(dev.rows))
    te = np.arange(len(dev.rows), len(allds.rows))
    ev = EV.evaluate(allds, tr, te, _frozen_policies(frz, PO, FE), reps=int(cfg.get("analysis", {}).get("bootstrap",
                                                                                                         1000)))
    pd.DataFrame(ev["summary"]).to_csv(os.path.join(a.out, "final", "summary_final.csv"), index=False, encoding="utf-8")
    pd.DataFrame(ev["rows"]).to_csv(os.path.join(a.out, "final", "choices_final.csv"), index=False, encoding="utf-8")
    TX.write_json(done, dict(policy=frz["policy"], deploy_tau=frz.get("deploy_tau"), rows=len(te)))
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
        print("the confirmation set was already evaluated once; its results are in %s" % cdir)
        return 1
    specs = DS.expand(cfg)
    try:
        AY.check_confirmation(AY.load(), specs, cfg["name"])
    except ValueError as e:
        print("refused: %s" % e)
        return 2
    try:
        train = _train_on(a.out, [p for p in a.train.split(",") if p])
    except BD.IncompatibleRun as e:
        print(str(e))
        return 2
    s = RN.label(cfg, a.out, workers=a.workers, subdir="confirm", accept_changes=a.accept_changes)
    print(json.dumps({k: v for k, v in s.items() if k != "per_task_seconds"}))
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
                             confirm_base_circuits=len(set(conf.groups)), freeze=frz))
    print("confirmation evaluated once: %d rows, %d base circuits" % (len(te), len(set(conf.groups))))
    return 0


def cmd_regimes(a):
    """Label a controlled study (the regime study, or with --subdir another one such as noisecomp) into OUT/SUBDIR
    (resumable) and analyze it with OUT's frozen recommender."""
    from . import run as RN
    cfg = _cfg(a.config)
    sub_ = a.subdir or "regimes"
    if not a.skip_label:
        s = RN.label(cfg, a.out, workers=a.workers, subdir=sub_, accept_changes=a.accept_changes,
                     retry_failed=a.retry_failed)
        print(json.dumps({k: v for k, v in s.items() if k != "per_task_seconds"}))
    print(json.dumps(RN.regime_analyze(cfg, a.out, sub=sub_)))
    return 0


def cmd_profile(a):
    """Classical cost of a recommendation against width — compilation, deployable features, simulator-only
    augmentation, inference — with label generation for comparison (writes OUT/analysis/runtime.csv)."""
    import pandas as pd
    from . import run as RN
    widths = [int(w) for w in (a.widths or "3,4,5,6,7").split(",")]
    rows = RN.profile(a.out, widths=widths)
    os.makedirs(os.path.join(a.out, "analysis"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "analysis", "runtime.csv"), index=False, encoding="utf-8")
    print(pd.DataFrame(rows).groupby("n")[["compile_s", "deployable_s", "simulated_s", "inference_s",
                                          "label_s"]].median())
    return 0


def cmd_report(a):
    from . import report as RP
    print(RP.write(a.out, os.path.join(a.out, "RESULTS_TABLES.md")))
    return 0


def cmd_run(a):
    rc = cmd_label(a)
    if rc:
        print("labeling reported errors; analysis continues on the labeled tasks")
    cmd_analyze(a)
    return cmd_figures(a)


def cmd_bundle(a):
    """Build the versioned model bundle of the frozen recommender: trained on the run's development labels (--train),
    with the frozen selective threshold and the nested-CV validation summary; written to --bundle-out (default
    qemx/selection/bundles/<version>)."""
    import pandas as pd
    from . import binding as BD, deploy as DP
    from .. import textio as TX
    cfg = _cfg(a.config)
    try:
        frz = BD.check_freeze(a.out, cfg)
        ds = _train_on(a.out, [p for p in a.train.split(",") if p])
    except BD.IncompatibleRun as e:
        print(str(e))
        return 2
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
    out = a.bundle_out or os.path.join(DP.BUNDLE_DIR, version)
    tau = frz.get("deploy_tau") if sname == frz["policy"][len("MODEL["):-1].split("|")[1] else None
    import numpy as np
    b = DP.build_bundle(ds, np.arange(len(ds.rows)), fam, sname, out, version=version, tau=tau, validation=val,
                        training=dict(run=os.path.basename(os.path.normpath(a.out)), parts=a.train, freeze=frz,
                                      run_identities={p: (BD.load(os.path.join(a.out, p)) or {}).get("identity_digest")
                                                      for p in a.train.split(",") if p}))
    print(json.dumps(dict(bundle=out, version=version, configuration=b.meta["configuration"], tau=tau,
                          verification=b.meta["verification"]), indent=1))
    del TX
    return 0


def cmd_recommend(a):
    """Recommend a protocol for one experiment: an app state (--state, the Playground's export) or an experiment
    document (--experiment), at budget --budget on --device (when the state's backend is not a fake device). Prints
    the same recommendation the Playground shows."""
    from . import deploy as DP
    from .. import experiment as EX
    from .. import textio as TX
    if bool(a.state) == bool(a.experiment):
        print("give exactly one of --state or --experiment")
        return 2
    state = TX.read_json(a.state) if a.state else EX.to_state(EX.load(a.experiment))
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
    print(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
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
        print(json.dumps({k: p[k] for k in ("digest", "budget", "blocks", "remaining_s", "reserved_s", "admitted_jobs",
                                             "total_jobs", "fits", "note")}, indent=1, ensure_ascii=False))
        print("written: %s" % path)
        return 0
    p = TX.read_json(a.plan)
    try:
        HWS.require_authorization(p)
    except HWS.NotAuthorized as e:
        print("refused: %s" % e)
        return 3
    print("authorized plan %s…: submission goes through the approved IBM workflow with the campaign ledger "
          "(pages/7_IBM_Quantum.py or scripts/hardware_transfer.py); this command does not submit jobs in 1.7.0."
          % p["digest"][:16])
    return 0


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
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
        p.add_argument("--workers", type=int, default=1)
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
        p.set_defaults(fn=fn)
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
    p.add_argument("--device", default=None, help="fake device when the state's backend is a noise injector")
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
    p.add_argument("--out", default="reports/selection_v170/hardware")
    p.set_defaults(fn=cmd_hardware)
    p = sub.add_parser("validate")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", type=int, default=3)
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--R", type=int, default=300)
    p.add_argument("--device", default="fake_fez", help="validate: device, e.g. fake_fez or fake_fez@s=2 (a scaled "
                                                          "snapshot as in the regime study)")
    p.set_defaults(fn=cmd_validate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
