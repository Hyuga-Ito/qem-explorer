"""Workflow runner of the protocol-selection study (1.6.0): label tasks in parallel with checkpoints and resource
limits, then analyze, certify and plot from the saved trials only.

Workflows (``configs/selection/<name>.yaml``): ``smoke`` (minutes; CI-sized), ``pilot`` (the executed study of this
release), ``full`` (the full budget grid with more trials; resource estimate only), ``ablation`` (feature groups),
``generalization`` (held-out families, devices, widths, budgets), ``boundary`` (refines budgets near observed
ranking reversals), ``mismatch`` (hidden perturbations and diagnostics).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from .. import textio as TX
from . import design as DS
from . import labels as LB
from . import protocols as PR
from . import task as TK


def _label_one(spec_json, out_dir, budgets, protocols, R, seed, designs, run_id=""):
    """Label one task (runs in a supervised child process, ``supervise.run_jobs``); returns (task id, status,
    message). Nothing here limits the process: the supervisor does."""
    spec = TK.TaskSpec.from_json(spec_json)
    rec = LB.label_task(spec, out_dir, budgets, protocols, R, seed, designs=designs, run_id=run_id)
    status = "skipped" if rec.get("skipped") else rec.get("status")
    return spec.task_id, status, rec.get("error", "")


def _keep_failed_copy(target: str, task_id: str) -> None:
    """Copy a failed record to ``failed_attempts/<id>.attemptN.json`` before ``--retry-failed`` relabels it — unless an
    identical copy is already there (a retry that wrote no new record: a ceiling or an interruption; 1.9.1)."""
    import shutil
    src = os.path.join(target, "tasks", task_id + ".json")
    fdir = os.path.join(target, "failed_attempts")
    os.makedirs(fdir, exist_ok=True)
    have = [f for f in os.listdir(fdir) if f.startswith(task_id + ".attempt")]
    d = LB.file_digest(src)
    if any(LB.file_digest(os.path.join(fdir, f)) == d for f in have):
        return
    shutil.copy2(src, os.path.join(fdir, "%s.attempt%d.json" % (task_id, len(have) + 1)))


def _supervised_label(specs, target, budgets, protocols, R, seed, designs, lim, workers, progress, run_id="",
                      ceilings=None, session=""):
    """Label ``specs`` into ``target`` with every task in a supervised child process (time and memory limits
    enforced by the parent, on every platform; ``supervise``). A task stopped by the supervisor gets a record with
    status timeout, memory or error (crashed), so a resumed run retries it only when asked to. Every attempt is
    appended to ``target``/attempts.jsonl (1.9.0). ``ceilings`` (``resources.Ceilings``): no new task starts once a
    whole-run ceiling is reached; tasks not started are absent from the returned list."""
    from . import integrity as IN
    from . import supervise as SV
    cfg_d = LB.config_digest(budgets, protocols, R, seed, designs)
    mem = int(float(lim.get("memory_gb", 0) or 0) * 2 ** 30)
    jobs = [SV.Job(s.task_id, "qemx.selection.run:_label_one",
                   (s.to_json(), target, budgets, protocols, R, seed, designs, run_id),
                   float(lim.get("task_seconds", 0) or 0), mem) for s in specs]
    by_id = {s.task_id: s for s in specs}
    done = []

    def finish(o):
        if o.status == "ok":
            tid, status, msg = o.value
        else:
            tid, status, msg = o.key, ("error" if o.status == "crashed" else o.status), o.message
            path = os.path.join(target, "tasks", tid + ".json")
            spec = by_id[tid]
            rec = LB.error_record(spec, cfg_d, status, o.status if o.status != "error" else "exception", msg,
                                  o.seconds, run_id=run_id, peak_rss_bytes=int(o.peak_rss_bytes),
                                  exitcode=o.exitcode)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            LB.write_json(path, rec)
        done.append((tid, status, o.seconds, msg or ""))
        IN.append_attempt(target, dict(task_id=tid, status=status, seconds=round(float(o.seconds), 3),
                                       peak_rss_bytes=int(o.peak_rss_bytes), message=(msg or "")[:500],
                                       session=session, run_id=run_id))
        if ceilings is not None:
            ceilings.finished(tid, o.seconds)
        progress("%s %s %.1fs %s" % (tid, status, o.seconds, (msg or "")[:120]))
    def interrupted(key, seconds):                   # 1.9.1 (F3): a task stopped by Ctrl+C/SIGTERM keeps its time
        p = os.path.join(target, "tasks", key + ".json")         # did the child write its record before the stop?
        wrote = os.path.exists(p) and os.path.getmtime(p) >= time.time() - float(seconds) - 1.0
        IN.append_attempt(target, dict(task_id=key, status="interrupted", seconds=round(float(seconds), 3),
                                       peak_rss_bytes=0, message="stopped when the labeling command was interrupted",
                                       record_written=bool(wrote), session=session, run_id=run_id))
    SV.run_jobs(jobs, workers=workers, on_done=finish, admit=ceilings.admit if ceilings is not None else None,
                on_interrupt=interrupted)
    return done


def engine_digests() -> dict:
    """sha256 of every Python file of the engine (``qemx/``), keyed by its repository-relative path."""
    from .. import provenance as PV
    out = {}
    base = os.path.join(PV.ROOT, "qemx")
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not d.startswith("."))
        for f in sorted(filenames):
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                out[os.path.relpath(path, PV.ROOT).replace(os.sep, "/")] = PV.file_digest(path)
    return out


def record_provenance(target_dir: str, stage: str) -> dict:
    """Append this labeling session to ``target_dir``/provenance.json: the provenance block (app version, source,
    lock and inventory digests, git commit, key package versions) and the digest of every engine file. A session
    that resumes a run with other engine code lists the changed files (``changed_since_first``), so labels made by
    different code are never mixed silently."""
    from datetime import datetime, timezone
    from .. import provenance as PV
    path = os.path.join(target_dir, "provenance.json")
    files = engine_digests()
    entry = dict(started=datetime.now(timezone.utc).isoformat(timespec="seconds"), provenance=PV.provenance(),
                 engine_sha256=files)
    if os.path.exists(path):
        rec = TX.read_json(path)
        first = rec["sessions"][0].get("engine_sha256", {})
        entry["changed_since_first"] = sorted(f for f in set(first) | set(files) if first.get(f) != files.get(f))
        rec["sessions"].append(entry)
    else:
        rec = dict(schema="qemx.selection.provenance/2", stage=stage, sessions=[entry])
    LB._write_json(path, rec)
    return rec


def _session_summary(done: list, seconds: float) -> dict:
    """Counts of a labeling session: ``ok`` = tasks with a valid label (made now or by an earlier session),
    ``skipped`` = those of an earlier session, ``errors``; ``per_task_seconds`` of the tasks labeled now only."""
    ok = sum(1 for d in done if d[1] in ("ok", "skipped"))
    return dict(tasks=len(done), ok=ok, skipped=sum(1 for d in done if d[1] == "skipped"), errors=len(done) - ok,
                seconds=seconds, per_task_seconds=[d[2] for d in done if d[1] == "ok"])


def run_status(comp: dict, *, limit=None, stopped=None) -> tuple:
    """(status, exit code) of a labeling command from the run-wide completion (1.9.0; second review N1):

    * ``complete`` (0) — every manifest task has a valid label;
    * ``partial-by-limit`` (0) — ``--limit`` was given deliberately, nothing failed or is corrupted, tasks remain;
    * ``incomplete`` (1) — failed or corrupted records exist (this session's or retained from an earlier one), or
      tasks remain without ``--limit`` (e.g. a ceiling stopped the run).
    A partial run is never reported as complete; analysis of it needs ``--allow-partial``."""
    if comp["complete"]:
        return "complete", 0
    if comp["n_failed"] or comp["corrupted"]:
        return "incomplete", 1
    if limit is not None and not stopped:
        return "partial-by-limit", 0
    return "incomplete", 1


def label(cfg: dict, out_dir: str, *, workers: int = 1, final: bool = False, limit: int | None = None,
          progress=print, subdir: str | None = None, accept_changes: bool = False, retry_failed: bool = False,
          quarantine_corrupted: bool = False, override_limits: str | None = None) -> dict:
    """Label the development tasks (or, with ``final``, the final-test tasks into ``out_dir``/final; with
    ``subdir``, a separate study such as the controlled regimes into ``out_dir``/``subdir``).

    The labels directory is bound to this run (``binding.bind``: configuration, task manifest, split, portfolio,
    device snapshots, code and environment); an incompatible resume is refused before anything runs. Tasks already
    labeled are skipped (checked in this process, without starting a child); tasks recorded as error, timeout or
    memory are retried only with ``retry_failed`` (the failed record is kept in attempts.jsonl and under
    failed_attempts/). Corrupted records stop the command unless ``quarantine_corrupted`` moves them aside (listed in
    run.json) to be relabeled.

    1.9.0 (second review N1, N2): a resource preflight runs first (``resources``); a configured cap that the known
    estimate exceeds refuses the run unless ``override_limits`` gives a reason (recorded); whole-run ceilings are
    monitored while labeling. The summary reports this session *and* the whole run (``run``: ``integrity.completion``)
    with ``status`` complete | partial-by-limit | incomplete and the command's ``exit_code``."""
    from datetime import datetime, timezone
    from . import binding as BD
    from . import integrity as IN
    from . import resources as RS
    from . import supervise as SV
    specs = DS.expand(cfg)
    dev, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    stage = subdir or ("final" if final else "dev")
    todo = fin if final else dev
    target = os.path.join(out_dir, stage)
    sup = SV.support()
    pre = RS.preflight(cfg, target if os.path.isdir(target) else None, workers=workers, final=final, subdir=subdir)
    adm = RS.admission(pre, override_limits)            # raises AdmissionRefused before anything is written
    with IN.RunLock(target, "label"):
        run = BD.bind(target, cfg, todo, dev, fin, stage, accept_changes=accept_changes, support=sup)
        session = run["sessions"][-1]["started"]
        if override_limits or adm["unknown"]:
            rec_ = BD.load(target)
            rec_["sessions"][-1]["admission"] = dict(adm, preflight_caps=pre["caps"])
            TX.write_json(os.path.join(target, "run.json"), rec_, indent=1, default=str, atomic=True)
        os.makedirs(os.path.join(target, "tasks"), exist_ok=True)
        budgets = [int(b) for b in cfg.get("budgets", PR.BUDGETS)]
        protocols = list(cfg.get("protocols", list(PR.PORTFOLIO_16)))
        R, seed = int(cfg.get("R", 200)), int(cfg.get("master_seed", 20261019))
        lim = cfg.get("limits", {})
        manifest = dict(schema="qemx.selection.manifest/2", config=cfg, development=[s.to_json() for s in dev],
                        final=[s.task_id for s in fin], run_identity=run["identity_digest"])
        mpath = os.path.join(target if subdir else out_dir, "manifest.json")
        if os.path.exists(mpath):
            old = TX.read_json(mpath)
            if DS.label_part(old.get("config") or {}) != DS.label_part(cfg) \
                    or [d for d in old.get("development", [])] != manifest["development"] \
                    or old.get("final") != manifest["final"]:
                raise BD.IncompatibleRun("%s describes another configuration or task split; use a new output "
                                         "directory" % mpath)
        else:
            LB.write_json(mpath, manifest)
        designs = cfg.get("learning_designs", 1)
        record_provenance(target, stage)
        before = IN.completion(target)
        if before["corrupted"]:
            if not quarantine_corrupted:
                raise IN.CorruptedRun("%s has %d corrupted record(s): %s. Nothing was labeled. Inspect them, or move "
                                      "them aside and relabel with --quarantine-corrupted." % (
                                          target, len(before["corrupted"]), "; ".join(
                                              "%s (%s)" % (c["task_id"], c["why"]) for c in before["corrupted"][:5])))
            IN.quarantine(target, [c["task_id"] for c in before["corrupted"]],
                          "corrupted: " + "; ".join("%s (%s)" % (c["task_id"], c["why"]) for c in before["corrupted"]))
            before = IN.completion(target)
        done_ok = set(before["completed_ids"])
        failed_ids = {t for v in before["failed"].values() for t in v}
        if limit is not None:
            todo = todo[: int(limit)]
        skipped = [s for s in todo if s.task_id in done_ok]
        retained = [s for s in todo if s.task_id in failed_ids and not retry_failed]
        keep = {s.task_id for s in retained} | done_ok
        work = [s for s in todo if s.task_id not in keep]
        if retry_failed:
            for s in work:
                if s.task_id in failed_ids:                  # keep the failed attempt's record (audit trail)
                    IN.keep_in_ledger(target, s.task_id, "replaced by a retry")      # 1.9.1 (F3)
                    _keep_failed_copy(target, s.task_id)
        ceil = RS.Ceilings(cfg, target, IN.consumed_seconds(target)["total_seconds"])     # 1.9.1 (F3): persistent
        t0 = time.perf_counter()
        done = _supervised_label(work, target, budgets, protocols, R, seed, designs, lim, workers, progress,
                                 run_id=run["identity_digest"], ceilings=ceil, session=session)
        sess = _session_summary(done + [(s.task_id, "skipped", 0.0, "") for s in skipped], time.perf_counter() - t0)
        after = IN.completion(target)
        status, code = run_status(after, limit=limit, stopped=ceil.stopped)
        summary = dict(sess, retained_failures=[s.task_id for s in retained], not_started=len(work) - len(done),
                       stopped=ceil.stopped, limit=limit, status=status, exit_code=code,
                       run={k: v for k, v in after.items() if k != "completed_ids"}, admission=adm,
                       preflight=dict(cpu=pre["cpu"], wall=pre["wall"], disk=pre["disk"], memory=pre["memory"],
                                      caps=pre["caps"]),
                       supervision=sup, stage=stage,
                       finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        LB.write_json(os.path.join(target, "label_summary.json"), summary)
    return summary


def measured_rates(labels_dir: str) -> list:
    """Per task labeled in ``labels_dir`` (a ``dev``, ``final`` or ``mismatch`` directory): width, preparation
    seconds (exact values, allocations, training designs — independent of r and the budgets) and seconds per trial
    (one protocol, one budget, one half)."""
    rows = []
    tdir = os.path.join(labels_dir, "tasks")
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".json"):
            continue
        if ".tmp" in f:
            continue
        rec = TX.read_json(os.path.join(tdir, f))
        if rec.get("status") != "ok":
            continue
        ps = rec.get("prepare_seconds") or 0.0
        prep = float(sum(ps.values())) if isinstance(ps, dict) else float(ps)
        trials = 2 * int(rec["R"]) * len(rec["budgets"]) * len(rec["protocols"])
        npz = os.path.join(tdir, f[:-5] + ".npz")
        rows.append(dict(n=int(rec["spec"]["n"]), prepare=prep, per_trial=max(float(rec["seconds"]) - prep, 0.0)
                         / trials, designs=rec.get("learning_designs", 1), seconds=float(rec["seconds"]),
                         json_bytes=os.path.getsize(os.path.join(tdir, f)),
                         npz_bytes_per_trial=os.path.getsize(npz) / trials if os.path.exists(npz) else None))
    return rows


def estimate(cfg: dict, measured: list | None = None) -> dict:
    """Resource estimate of a configuration: tasks × budgets × protocols × 2r trials and disk from the stored
    arrays (from measured file sizes when ``measured`` is given); with ``measured`` (``measured_rates`` of an executed run with the same protocols and number of training
    designs) the CPU time, as per-task preparation plus per-trial time, medians per width; widths beyond the
    measured ones are extrapolated log-linearly in n from the per-width medians (never below the widest measured)."""
    from . import ancestry as AY
    from .emulate import DesignPlan
    specs = DS.expand(cfg)
    dev, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    nB = len(cfg.get("budgets", PR.BUDGETS))
    protos = list(cfg.get("protocols", PR.PORTFOLIO))
    P = len(protos)
    R = int(cfg.get("R", 200))
    plan = DesignPlan.from_config(cfg.get("learning_designs", 1))
    n_learn = sum(1 for k in protos if PR.PORTFOLIO[k].learning)
    trials = len(specs) * nB * P * 2 * R
    budgets = [int(b) for b in cfg.get("budgets", PR.BUDGETS)]
    out = dict(tasks=len(specs), development=len(dev), final=len(fin), base_circuits=len({s.base_id for s in specs}),
               independent_logical_circuits=len({AY.logical_id(s) for s in specs}),
               development_logical_circuits=len({AY.logical_id(s) for s in dev}),
               final_logical_circuits=len({AY.logical_id(s) for s in fin}),
               variants_per_logical_circuit=round(len(specs) / max(1, len({AY.logical_id(s) for s in specs})), 2),
               budgets=nB, total_shots_per_deployment=budgets, protocols=P,
               repetitions_per_half=R, R=R, design_plan=plan.to_json(), design_estimand=plan.estimand,
               independent_designs_per_half=(plan.per_half if plan.mode == "fresh" else plan.count),
               repetitions_per_design=(round(R / plan.per_half, 2) if plan.mode == "fresh" else round(R / plan.count, 2)),
               designs_prepared_per_learning_protocol_and_task=(2 * plan.per_half + 1 if plan.mode == "fresh"
                                                                else plan.count),
               learning_protocols=n_learn, trials=int(trials),
               disk_mb=float(len(specs) * (2 * P * nB * R * 8 + 2 * P * nB * len(("a", "b", "c", "d")) * R) / 1e6
                             / 3.0))
    if measured:
        from .resources import rate_function
        ns = sorted({m["n"] for m in measured})
        rate = rate_function(measured)
        cpu = sum(rate(s.n)[0] + rate(s.n)[1] * 2 * R * nB * P for s in specs)
        jb = float(np.median([m["json_bytes"] for m in measured]))
        sizes = [m["npz_bytes_per_trial"] for m in measured if m["npz_bytes_per_trial"] is not None]
        if sizes:                                    # 1.9.0: records without their arrays tell nothing about size
            tb = float(np.median(sizes))
            out.update(disk_mb=float(len(specs) * (jb + tb * 2 * R * nB * P) / 1e6))
        out.update(cpu_hours=float(cpu / 3600.0), measured_tasks=len(measured), measured_widths=ns,
                   extrapolated_widths=sorted({s.n for s in specs} - set(ns)),
                   designs_measured=sorted({str(m["designs"]) for m in measured}),
                   designs_target=str(plan.to_json()))
    else:
        out.update(cpu_hours=None, cpu_status="unknown (no measured rates; plan --measured DIR, or label a few tasks "
                                            "first and plan --out the run)")
    lim = cfg.get("limits", {})
    caps = {}
    if lim.get("cpu_hours") is not None:            # 1.9.0: a cap is never reported as met when its estimate is unknown
        caps["cpu_hours"] = dict(cap=float(lim["cpu_hours"]),
                                 within=None if out.get("cpu_hours") is None
                                 else bool(out["cpu_hours"] <= float(lim["cpu_hours"])))
    if lim.get("disk_gb") is not None:
        caps["disk_gb"] = dict(cap=float(lim["disk_gb"]), within=bool(out["disk_mb"] / 1e3 <= float(lim["disk_gb"])))
    out["caps"] = caps
    out["registry_overlap"] = {k: len(v) for k, v in AY.overlap(AY.load(), specs, exclude_study=cfg.get("name"))
                               .items()}
    return out


def family_benefit(ch, lab, run_dir: str, reps: int = 1000, seed: int = 0) -> list:
    """Per circuit family: the paired difference of the proposed policy (the frozen configuration, else the best
    model) to SBS-B over held-out rows of that family — mean log10 MSE difference (primary) and mean absolute MSE
    difference — with a bootstrap over the family's base circuits, and the number of independent base circuits."""
    from . import evaluate as EV
    per = EV._per_policy_from_choices(ch, lab)
    frz = os.path.join(run_dir, "freeze.json")
    prop = TX.read_json(frz)["policy"] if os.path.exists(frz) else None
    if prop not in per:
        cands = [p for p in per if p.startswith("MODEL[")]
        if not cands:
            return []
        prop = min(cands, key=lambda p: np.nanmean(per[p]["log_risk"]))
    out = []
    fam = ch.family.to_numpy()
    for f in sorted(set(fam.tolist())):
        m = fam == f
        g = ch.base_id.to_numpy()[m]
        dl = per[prop]["log_risk"][m] - per["SBS-B"]["log_risk"][m]
        da = per[prop]["risk"][m] - per["SBS-B"]["risk"][m]
        ci = EV.bootstrap({"log": dl, "abs": da}, g, reps, seed)[0]
        out.append(dict(family=f, policy=prop, reference="SBS-B", base_circuits=int(len(set(g.tolist()))),
                        rows=int(m.sum()), diff_log=ci["log"][0], diff_log_lo=ci["log"][1], diff_log_hi=ci["log"][2],
                        diff_mse=ci["abs"][0], diff_mse_lo=ci["abs"][1], diff_mse_hi=ci["abs"][2]))
    return out


FROZEN_EXTRAS = ("deployed", "selective", "nested_selective")      # evaluate one model configuration: the frozen one


def deployed_configuration(out_dir: str, a: dict):
    """(family, feature set) of the recommender analyzed as deployed: the frozen configuration when the run has a
    ``freeze.json`` (decision D25); before a freeze, the configuration the analysis section declares explicitly
    (``analysis.deployed_family`` *and* ``deployed_set``); otherwise None — the configuration is not known yet.

    1.9.1 (independent review F1): 1.9.0 fell back to ``hgb|deployable+simulated`` before a freeze, a configuration
    outside the 1.9 studies' grids, so an initial analysis with these extras evaluated a model that could never be
    frozen and the pipeline later reused those tables."""
    frz = os.path.join(out_dir, "freeze.json")
    if os.path.exists(frz):
        pol = TX.read_json(frz)["policy"]
        fam, sname = pol[len("MODEL["):-1].split("|")
        return fam, sname
    if a.get("deployed_family") and a.get("deployed_set"):
        return a["deployed_family"], a["deployed_set"]
    return None


def deployed_configuration_or_default(out_dir: str, a: dict) -> tuple:
    """The frozen or declared configuration (``deployed_configuration``), else 1.9.0's default
    (``analysis.deployed_family`` or hgb, ``analysis.deployed_set`` or deployable+simulated). For analyses that only need
    *a* fitted recommender and are not the frozen-configuration evaluation — the controlled regime study and the
    runtime profile (their 1.9.0 behaviour, kept; independent debugging of 1.9.1)."""
    conf = deployed_configuration(out_dir, a)
    return conf if conf is not None else (a.get("deployed_family", "hgb"), a.get("deployed_set", "deployable+simulated"))


def frozen_extras_settings(a: dict) -> dict:
    """The evaluation settings the deployed/selective/nested tables depend on (beyond the configuration and data)."""
    return dict(k=int(a.get("split_options", {}).get("same-distribution", {}).get("k", 5)),
                nested_inner_k=int(a.get("nested_inner_k", 4)), bootstrap=int(a.get("bootstrap", 1000)),
                selective_bootstrap=int(a.get("selective_bootstrap", 400)))


def frozen_extras_binding(out_dir: str, a: dict, data_digest: str | None) -> dict:
    """What the tables of ``FROZEN_EXTRAS`` are bound to (1.9.1; F1): the evaluated configuration and where it came
    from, the development data, the evaluation settings and the engine code."""
    from . import binding as BD
    conf = deployed_configuration(out_dir, a)
    frz = os.path.join(out_dir, "freeze.json")
    return dict(configuration="MODEL[%s|%s]" % conf if conf else None,
                source="freeze" if os.path.exists(frz) else "declared (analysis.deployed_family/deployed_set)",
                data=data_digest, settings=DS.freeze_digest(frozen_extras_settings(a))[:16], code=BD.code_digest())


LEGACY_NO_BINDING = "record no binding"                          # suffix of a pre-1.9.1 table problem


def nested_reuse_problems(out_dir: str, cfg: dict, required=("deployed", "nested_selective")) -> list:
    """Why the existing deployed/nested tables of ``out_dir`` cannot be reused for the frozen configuration (1.9.1;
    independent review F1): an empty list means they evaluate exactly the frozen configuration, on the frozen
    development data, with the configured evaluation settings and the current code (or a recorded, accepted code
    change). File existence alone is never enough. ``required``: the tables that must exist (``freeze --complete``
    needs only ``nested_selective``, as before 1.9.1; any other frozen-configuration table present is checked too).
    Problems ending in ``LEGACY_NO_BINDING`` concern tables made before 1.9.1, which record no binding."""
    from . import binding as BD
    adir = os.path.join(out_dir, "analysis")
    fz = os.path.join(out_dir, "freeze.json")
    if not os.path.exists(fz):
        return ["the run is not frozen"]
    frz = TX.read_json(fz)
    rj = os.path.join(adir, "results.json")
    res = TX.read_json(rj) if os.path.exists(rj) else {}
    probs = []
    for name in ("deployed", "nested_selective"):
        e = (res.get("extras") or {}).get(name)
        if not e:
            if name in required:
                probs.append("no %s analysis yet" % name)
            continue
        b = e.get("binding")
        if not b:
            probs.append("the %s tables were made before 1.9.1 and %s" % (name, LEGACY_NO_BINDING))
            continue
        if b.get("configuration") != frz["policy"]:
            probs.append("the %s tables evaluate %s, the frozen configuration is %s" % (name, b.get("configuration"),
                                                                                       frz["policy"]))
        if frz.get("dev_data_digest") and b.get("data") != frz["dev_data_digest"]:
            probs.append("the %s tables were made on other development data" % name)
        if b.get("settings") != DS.freeze_digest(frozen_extras_settings(cfg.get("analysis", {})))[:16]:
            probs.append("the %s tables were made with other evaluation settings (k, nested_inner_k, bootstrap)"
                         % name)
        code_ok = b.get("code") == BD.code_digest() or any(
            c.get("now") == BD.code_digest() for c in res.get("code_changes", []) or [])
        if not code_ok:
            probs.append("the %s tables were made by other code" % name)
    op = os.path.join(adir, "nested_operating.csv")
    if os.path.exists(op):
        import pandas as pd
        conf = str(pd.read_csv(op, encoding="utf-8").configuration.iloc[0])
        if conf != frz["policy"]:
            probs.append("nested_operating.csv evaluates %s, the frozen configuration is %s" % (conf, frz["policy"]))
    else:
        probs.append("nested_operating.csv is missing")
    return sorted(set(probs))


def _deployed_tables(ds, out_dir, adir, a, k):
    import pandas as pd
    from . import analysis as AN
    fam, sname = deployed_configuration(out_dir, a)
    dep = AN.deployed(ds, fam, sname, k=k)
    pd.DataFrame(dep["rows"]).assign(configuration="MODEL[%s|%s]" % (fam, sname)).to_csv(
        os.path.join(adir, "deployed_rows.csv"), index=False, encoding="utf-8")
    pd.DataFrame(dep["prediction"]).to_csv(os.path.join(adir, "prediction.csv"), index=False, encoding="utf-8")
    pd.DataFrame(dep["calibration"]).to_csv(os.path.join(adir, "calibration.csv"), index=False, encoding="utf-8")
    _selective_tables(adir, a, dep["rows"])
    return None


def _nested_tables(ds, out_dir, adir, a, k):
    """Selective recommendation by nested cross-validation (``analysis.nested_selective``; review R2): tables
    nested_summary, nested_folds, nested_rows, nested_curve and nested_operating (the deployment threshold)."""
    import pandas as pd
    from . import analysis as AN
    fam, sname = deployed_configuration(out_dir, a)
    r = AN.nested_selective(ds, fam, sname, k=k, inner_k=int(a.get("nested_inner_k", 4)),
                            reps=int(a.get("bootstrap", 1000)))
    for key in ("summary", "folds", "rows", "curve", "operating"):
        pd.DataFrame(r[key]).to_csv(os.path.join(adir, "nested_%s.csv" % key), index=False, encoding="utf-8")
    rows = pd.DataFrame(r["rows"])                     # Table S1: rejection by family and device (nested threshold)
    grp = []
    for col in ("family", "device"):
        for v, e in rows.groupby(col):
            per = e.groupby("base_id").accepted.mean()           # each base circuit weighs the same
            grp.append(dict(group=col, value=v, rows=int(len(e)), base_circuits=int(e.base_id.nunique()),
                            rejection_rate=float(1.0 - per.mean()), statement="empirical"))
    pd.DataFrame(grp).to_csv(os.path.join(adir, "nested_groups.csv"), index=False, encoding="utf-8")
    return None


def _selective_tables(adir, a, rows=None):
    """Selective recommendation tables (``selective_*.csv``) from the deployed rows; without ``rows`` they are read
    from ``deployed_rows.csv`` (the ``selective`` analysis: no model is refitted)."""
    import pandas as pd
    from . import analysis as AN
    if rows is None:
        p = os.path.join(adir, "deployed_rows.csv")
        if not os.path.exists(p):
            raise FileNotFoundError("%s: run the 'deployed' analysis first" % p)
        rows = pd.read_csv(p, encoding="utf-8").to_dict("records")
    sel = AN.selective(rows, reps=int(a.get("selective_bootstrap", 400)))
    for key in ("curve", "folds", "operating", "groups", "rows"):
        pd.DataFrame(sel[key]).to_csv(os.path.join(adir, "selective_%s.csv" % key), index=False, encoding="utf-8")
    return None


def pool(adir: str, reps: int = 1000, seed: int = 0, only=None) -> dict:
    """From the saved tables only: the out-of-fold pooled summary of every split (``pooled_<split>.csv``) and the
    bias-corrected estimate of the best model configuration per split (``optimism.csv``).

    ``only`` (1.9.1): re-pool just these splits and leave every other pooled table — in particular the one a freeze
    was chosen on — byte for byte as it is (an extras-only analysis, such as the nested stage after a freeze, pools
    nothing; ``optimism.csv`` keeps the other splits' rows; the final evaluation is re-summarized only when ``only`` is
    None)."""
    import pandas as pd
    from . import evaluate as EV
    lab = pd.read_csv(os.path.join(adir, "labels.csv"), encoding="utf-8")
    opt, done = [], []
    for f in sorted(os.listdir(adir)):
        if not (f.startswith("choices_") and f.endswith(".csv")):
            continue
        kind = f[len("choices_"):-4]
        if only is not None and kind not in only:
            continue
        ch = pd.read_csv(os.path.join(adir, f), encoding="utf-8")
        if ch.empty:
            continue
        pd.DataFrame(EV.pooled(ch, lab, reps, seed)).assign(split=kind).to_csv(
            os.path.join(adir, "pooled_%s.csv" % kind), index=False, encoding="utf-8")
        if kind == "same-distribution":
            fb = family_benefit(ch, lab, os.path.dirname(os.path.normpath(adir)), reps, seed)
            if fb:
                pd.DataFrame(fb).to_csv(os.path.join(adir, "family_benefit.csv"), index=False, encoding="utf-8")
            from . import analysis as AN
            pd.DataFrame(AN.failure_profile(ch, lab)).to_csv(os.path.join(adir, "failure_profile.csv"), index=False,
                                                             encoding="utf-8")
        if any(c.startswith("MODEL[") and c.endswith("|choice") for c in ch.columns):
            for ref in ("SBS", "SBS-B"):
                if ref + "|choice" in ch.columns:
                    opt.append(dict(split=kind, **EV.bbc_cv(ch, lab, reps=reps, seed=seed, ref=ref)))
        done.append(kind)
    if opt:
        op = os.path.join(adir, "optimism.csv")
        new = pd.DataFrame(opt)
        if only is not None and os.path.exists(op):              # keep the rows of the splits not re-pooled
            old = pd.read_csv(op, encoding="utf-8")
            if "split" in old.columns:
                new = pd.concat([old[~old.split.isin(done)], new], ignore_index=True)
        new.to_csv(op, index=False, encoding="utf-8")
    if only is not None:
        return dict(pooled=done, optimism=len(opt))
    fdir = os.path.join(os.path.dirname(os.path.normpath(adir)), "final")
    fch = os.path.join(fdir, "choices_final.csv")
    if os.path.exists(fch):                      # the once-only final evaluation's saved rows, re-summarized
        flab = os.path.join(fdir, "labels.csv")
        if not os.path.exists(flab):
            LB.summarize(fdir).to_csv(flab, index=False, encoding="utf-8")
        pd.DataFrame(EV.pooled(pd.read_csv(fch, encoding="utf-8"), pd.read_csv(flab, encoding="utf-8"), reps, seed)).to_csv(
            os.path.join(fdir, "metrics_final.csv"), index=False, encoding="utf-8")
        done.append("final")
    return dict(pooled=done, optimism=len(opt))


class StaleAnalysis(RuntimeError):
    """An analysis directory produced from another analysis configuration, other labels, other data or other code."""


def analysis_identity(cfg: dict, comp: dict) -> dict:
    """What an analysis directory's tables depend on (1.9.0; second review, analysis outputs): the analysis section,
    the label-determining configuration, the evaluated data content (``integrity.completion`` data digest) and the
    engine code. Tables of different identities are never merged into one directory."""
    from . import binding as BD
    return dict(analysis=DS.freeze_digest(cfg.get("analysis", {}))[:16],
                labels=DS.freeze_digest(DS.label_part(cfg))[:16], data=(comp.get("data_digest") or "")[:16],
                code=BD.code_digest())


def _check_analysis_identity(adir: str, ident: dict, reset: bool, accept_code: bool = False) -> dict | None:
    """None when the directory is new or has the same identity; archives it when ``reset``; else StaleAnalysis. A 1.8.0
    directory without a recorded identity is reported (``legacy``) and adopted.

    Only a *code* difference (same analysis section, labels and data) can be accepted explicitly (``accept_code``,
    ``--accept-changes``): the new tables are then added beside the old ones and the change is recorded in results.json
    (``code_changes``), as labeling records accepted code changes. A frozen run's analysis is never archived by
    ``reset``: the freeze is bound to its tables (1.9.0, independent verification)."""
    p = os.path.join(adir, "results.json")
    if not os.path.exists(p):
        return None
    old = TX.read_json(p).get("identity")
    if old is None:
        return dict(legacy=True)
    diff = sorted(k for k in ident if old.get(k) != ident[k])
    if not diff:
        return None
    if diff == ["code"] and accept_code:
        return dict(accepted_code_change=dict(recorded=old.get("code"), now=ident["code"]))
    frozen = os.path.exists(os.path.join(os.path.dirname(os.path.normpath(adir)), "freeze.json"))
    if not reset or frozen:
        hint = ("this run is frozen on these tables: a frozen decision is never re-analyzed in place — %s"
                % ("accept the code change explicitly (--accept-changes; recorded) to add analyses, or analyze "
                   "a copy of the labels in a new run directory" if diff == ["code"] else
                   "analyze a copy of the labels in a new run directory")) if frozen else (
            "Analyze into a new run directory, or archive these tables with --reset-analysis%s." %
            (" (or, for a code change only, add to them with --accept-changes, recorded)" if diff == ["code"] else ""))
        raise StaleAnalysis("%s holds tables of another %s (recorded %s, now %s); merging them would mix results. %s"
                            % (adir, "/".join(diff), {k: old.get(k) for k in diff}, {k: ident[k] for k in diff}, hint))
    from datetime import datetime, timezone
    dst = adir.rstrip("/\\") + ".superseded-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.replace(adir, dst)
    os.makedirs(adir, exist_ok=True)
    return dict(archived=os.path.basename(dst), differed=diff)


def analyze(out_dir: str, cfg: dict, only_splits=None, only_extras=None, allow_partial: bool = False,
            reset: bool = False, accept_changes: bool = False) -> dict:
    """Every evaluation the configuration asks for, from the saved trials of ``out_dir``/dev. Writes
    ``analysis/results.json`` (merged with what other analysis processes wrote) and CSV tables. The same-distribution
    split compares every model family × feature set; the other splits compare the families and feature sets listed
    in ``analysis.other_split_families`` / ``other_split_sets`` (compute).

    1.9.0 (second review N1, analysis outputs): the development labels must be complete unless ``allow_partial``
    (then results.json says ``data.status = partial`` with the missing and failed tasks, and a freeze refuses it); the
    directory's identity (``analysis_identity``) must match an existing analysis there, or ``reset`` archives it."""
    import pandas as pd
    from . import evaluate as EV
    from . import features as FE
    from . import policies as PO
    from . import binding as BD
    from . import integrity as IN
    ddir = os.path.join(out_dir, "dev")
    comp = IN.completion(ddir)
    if comp["corrupted"]:
        raise IN.CorruptedRun("%s: %s" % (ddir, IN.describe(comp)))
    if not comp["complete"] and not allow_partial:
        raise IN.IncompleteRun("the development labels are not complete (%s). Finish labeling, or analyze the "
                               "labeled part explicitly with --allow-partial (exploratory; a freeze refuses it)."
                               % IN.describe(comp))
    if comp["completed"] == 0:
        raise IN.IncompleteRun("no development task is labeled yet (%s): nothing to analyze" % IN.describe(comp))
    a = cfg.get("analysis", {})
    adir = os.path.join(out_dir, "analysis")
    ident = analysis_identity(cfg, comp)
    with IN.RunLock(out_dir, "analyze"):
        note = _check_analysis_identity(adir, ident, reset, accept_code=accept_changes)
        ds, collection = BD.collect(ddir)
        return _analyze(out_dir, cfg, a, ds, collection, comp, ident, note, adir, only_splits, only_extras)


def _analyze(out_dir, cfg, a, ds, collection, comp, ident, note, adir, only_splits, only_extras) -> dict:
    import pandas as pd
    from . import evaluate as EV
    from . import features as FE
    from . import policies as PO
    sets = {k: FE.SETS[k] for k in a.get("feature_sets", ["size_budget", "lightcone", "deployable"])}
    fams = tuple(a.get("model_families", ["ridge", "enet", "hgb", "mlp"]))
    res = dict(schema="qemx.selection.results/1", protocols=ds.protocols, budgets=ds.budgets, rows=len(ds.rows),
               tasks=len(set(ds.col("task_id").tolist())), base_circuits=len(set(ds.groups.tolist())),
               design_mode=ds.design_mode, evaluations={}, identity=ident,
               data=dict(status="complete" if comp["complete"] else "partial", expected=comp["expected"],
                         completed=comp["completed"], failed=comp["failed"], missing=comp["missing"],
                         data_digest=comp["data_digest"]))
    if note:
        res["identity_note"] = note
        if note.get("accepted_code_change"):
            prev = TX.read_json(os.path.join(adir, "results.json"))
            res["identity"] = prev.get("identity", ident)            # the tables' original identity is kept
            res["code_changes"] = list(prev.get("code_changes", [])) + [note["accepted_code_change"]]
    os.makedirs(adir, exist_ok=True)
    LB.write_json(os.path.join(adir, "collection.json"), dict(collection, rows=len(ds.rows),
                                                              design_mode=ds.design_mode))
    old_res = os.path.join(adir, "results.json")

    def checkpoint():                          # 1.7.0: written after every split and extra (survives interruptions)
        if os.path.exists(old_res):
            prev_ = TX.read_json(old_res)
            res["evaluations"] = dict(prev_.get("evaluations", {}), **res["evaluations"])
            res["extras"] = dict(prev_.get("extras", {}), **res.get("extras", {}))
        LB._write_json(old_res, res)
    if os.path.exists(old_res):
        prev = TX.read_json(old_res)
        res["evaluations"].update(prev.get("evaluations", {}))
        res["extras"] = prev.get("extras", {})
        for key in ("stale_nested", "deferred_extras"):     # 1.9.1: the audit trail survives a re-analysis
            if prev.get(key):
                res[key] = prev[key]
    evaluated = []
    for kind in a.get("splits", ["same-distribution"]):
        if only_splits is not None and kind not in only_splits:
            continue
        evaluated.append(kind)
        kw = dict(a.get("split_options", {}).get(kind, {}))
        parts = EV.splits(ds, kind, **kw)
        summ, rows = [], []
        if kind == "same-distribution":
            k_sets, k_fams = sets, fams
        else:
            k_sets = {k: FE.SETS[k] for k in a.get("other_split_sets", ["size_budget", "deployable"])}
            k_fams = tuple(a.get("other_split_families", ["ridge", "hgb"]))
        for name, tr, te in parts:
            if len(tr) == 0 or len(te) == 0:
                continue
            pols = PO.standard(k_sets, k_fams, a.get("abstain_family", "hgb"), a.get("abstain_set", "deployable"))
            ev = EV.evaluate(ds, tr, te, pols, reps=int(a.get("bootstrap", 1000)))
            for s in ev["summary"]:
                summ.append(dict(split=kind, part=name, **s))
            for r in ev["rows"]:
                rows.append(dict(split=kind, part=name, **r))
        pd.DataFrame(summ).to_csv(os.path.join(adir, "summary_%s.csv" % kind), index=False, encoding="utf-8")
        pd.DataFrame(rows).to_csv(os.path.join(adir, "choices_%s.csv" % kind), index=False, encoding="utf-8")
        res["evaluations"][kind] = dict(parts=[p[0] for p in parts], summary_csv="summary_%s.csv" % kind,
                                        choices_csv="choices_%s.csv" % kind)
        checkpoint()
    from . import analysis as AN
    k = int(a.get("split_options", {}).get("same-distribution", {}).get("k", 5))
    fam, sname = a.get("abstain_family", "hgb"), a.get("abstain_set", "deployable")
    extra = dict(
        ablation=lambda: AN.ablation(ds, fam, a.get("ablation_set", "deployable+simulated"), k=k,
                                     reps=int(a.get("bootstrap", 1000))),
        learning_curve=lambda: AN.learning_curve(ds, fam, sname, k=k, repeats=int(a.get("lc_repeats", 3))),
        abstention=lambda: AN.abstention(ds, fam, sname, k=k),
        interval_coverage=lambda: AN.interval_coverage(ds, fam, sname, k=k),
        certificates=lambda: AN.certificates(ds, os.path.join(out_dir, "dev")),
        floor_sensitivity=lambda: AN.floor_sensitivity(ds, fam, sname, k=k, reps=int(a.get("bootstrap", 1000))),
        budget_generalization=lambda: AN.budget_generalization(ds, tuple(a.get("other_split_families",
                                                                               ["ridge", "hgb"])), sname, k=k,
                                                               reps=int(a.get("bootstrap", 1000))),
        deployed=lambda: _deployed_tables(ds, out_dir, adir, a, k),
        selective=lambda: _selective_tables(adir, a),
        nested_selective=lambda: _nested_tables(ds, out_dir, adir, a, k),
        design_diagnostics=lambda: AN.design_diagnostics(os.path.join(out_dir, "dev"),
                                                         task_ids=sorted(set(ds.col("task_id").tolist()))),
    )
    names = [n for n in only_extras if n] if only_extras is not None else a.get("extras", list(extra))
    for name in names:
        if name not in extra:
            raise ValueError("unknown analysis %r (choose from %s)" % (name, ", ".join(extra)))
        if name in FROZEN_EXTRAS and deployed_configuration(out_dir, a) is None:
            # 1.9.1 (independent review F1): deferred until the model configuration is frozen
            half = (" (analysis.deployed_family and deployed_set are used only together; only one is declared)"
                    if bool(a.get("deployed_family")) != bool(a.get("deployed_set")) else "")
            res.setdefault("deferred_extras", {})[name] = (
                "needs the frozen configuration: run after `freeze` (pipeline stage 'nested', or analyze --splits= "
                "--extras deployed,nested_selective)" + half)
            print("deferred: %s evaluates the frozen configuration, which is not known before `freeze`%s" % (name, half))
            checkpoint()
            continue
        t0 = time.perf_counter()
        tab = extra[name]()
        if tab is not None:                                   # None: the analysis wrote its own tables
            pd.DataFrame(tab).to_csv(os.path.join(adir, "%s.csv" % name), index=False, encoding="utf-8")
        res.setdefault("extras", {})[name] = dict(csv="%s.csv" % name if tab is not None else None,
                                                  seconds=time.perf_counter() - t0)
        if name in FROZEN_EXTRAS:
            res["extras"][name]["binding"] = frozen_extras_binding(out_dir, a, comp["data_digest"])
            (res.get("deferred_extras") or {}).pop(name, None)
        checkpoint()
    fails = []
    lab = LB.summarize(os.path.join(out_dir, "dev"), failures=fails, task_ids=sorted(set(ds.col("task_id").tolist()))
                       + sorted({f["task_id"] for f in ds.failures}))       # the run's manifest only
    lab.to_csv(os.path.join(adir, "labels.csv"), index=False, encoding="utf-8")
    pd.DataFrame(fails, columns=["task_id", "status", "error_kind", "error"]).to_csv(
        os.path.join(adir, "label_failures.csv"), index=False, encoding="utf-8")
    if ds.design_mode == "fresh" or "CDR" in ds.protocols:
        pd.DataFrame(AN.design_variance(lab)).to_csv(os.path.join(adir, "design_variance.csv"), index=False,
                                                     encoding="utf-8")
    # 1.9.1: re-pool only the splits evaluated now (all declared ones when no --splits was given); the pooled table a
    # freeze was chosen on is never rewritten by an extras-only analysis such as the nested stage
    pool(adir, reps=int(a.get("bootstrap", 1000)), only=None if only_splits is None else evaluated)
    tf = {}
    for r in ds.rows:
        tf[r["task_id"]] = dict(task_id=r["task_id"], base_id=r["base_id"], family=r["family"], device=r["device"],
                                observable=r["observable"], inputs=r["inputs"], layout=r["layout"], **r["feats"])
    pd.DataFrame(list(tf.values())).to_csv(os.path.join(adir, "task_features.csv"), index=False, encoding="utf-8")
    if os.path.exists(old_res):                         # merge with what another analysis process wrote meanwhile
        prev = TX.read_json(old_res)
        res["evaluations"] = dict(prev.get("evaluations", {}), **res["evaluations"])
        res["extras"] = dict(prev.get("extras", {}), **res.get("extras", {}))
    LB._write_json(old_res, res)
    return res


def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


# ------------------------------------------------------------------ mismatch study
def mismatch_budgets(budgets, fraction: float = 0.02, minimum: int = 200) -> list:
    """The budget grid plus, for every B, the budget left after the declared diagnostic: B − d(B)."""
    from .mismatch import diag_shots
    out = set(int(b) for b in budgets)
    for b in budgets:
        d = diag_shots(int(b), fraction, minimum)
        if int(b) - d > 0:
            out.add(int(b) - d)
    return sorted(out)


def mismatch_specs(cfg: dict) -> list:
    from .mismatch import SCENARIOS
    base = DS.expand(dict(cfg, truth=""))
    out = []
    for name in cfg["mismatch"]["scenarios"]:
        for s in base:
            d = s.to_json()
            d["truth"] = SCENARIOS[name]
            out.append((name, TK.TaskSpec.from_json(d)))
    return out


def mismatch_label(cfg: dict, out_dir: str, *, workers: int = 1, progress=print, accept_changes: bool = False,
                   retry_failed: bool = False, override_limits: str | None = None) -> dict:
    """Label every (scenario, task) with the budget grid extended by the diagnostic-reduced budgets. 1.9.0: the same
    resource admission, monitored ceilings, retained failures and ``retry_failed`` as ``label``."""
    m = cfg["mismatch"]
    budgets = mismatch_budgets(cfg["budgets"], m.get("diag_fraction", 0.02), m.get("diag_min", 200))
    target = os.path.join(out_dir, "mismatch")
    os.makedirs(os.path.join(target, "tasks"), exist_ok=True)
    protocols = list(cfg.get("protocols", list(PR.PORTFOLIO_16)))
    R, seed = int(cfg.get("R", 200)), int(cfg.get("master_seed", 20261019))
    lim = cfg.get("limits", {})
    pairs = mismatch_specs(cfg)
    from . import binding as BD
    from . import resources as RS
    from . import supervise as SV
    specs = [s for _, s in pairs]
    pre = RS.preflight(dict(cfg, budgets=budgets), target if os.path.isdir(os.path.join(target, "tasks")) else None,
                       workers=workers, specs=specs)
    adm = RS.admission(pre, override_limits)
    from . import integrity as IN
    with IN.RunLock(target, "mismatch label"):
        run = BD.bind(target, dict(cfg, budgets=budgets), specs, specs, [], "mismatch", accept_changes=accept_changes,
                      support=SV.support())
        LB._write_json(os.path.join(target, "scenarios.json"), [dict(scenario=n, task_id=s.task_id, truth=s.truth,
                                                                     base_task=s.base_id) for n, s in pairs])
        record_provenance(target, "mismatch")
        from . import integrity as IN
        before = IN.completion(target)
        if before["corrupted"]:
            raise IN.CorruptedRun("%s has %d corrupted record(s); nothing was labeled" % (target, len(before["corrupted"])))
        done_ok = set(before["completed_ids"])
        failed = {t for v in before["failed"].values() for t in v}
        work = [s for _, s in pairs if s.task_id not in done_ok and (retry_failed or s.task_id not in failed)]
        if retry_failed:
            for s in work:
                if s.task_id in failed:                  # keep the failed attempt's record (audit trail)
                    IN.keep_in_ledger(target, s.task_id, "replaced by a retry")      # 1.9.1 (F3)
                    _keep_failed_copy(target, s.task_id)
        ceil = RS.Ceilings(dict(cfg, budgets=budgets), target, IN.consumed_seconds(target)["total_seconds"])
        t0 = time.perf_counter()
        done = _supervised_label(work, target, budgets, protocols, R, seed,
                                 cfg.get("learning_designs", 1), lim, workers, progress,
                                 run_id=run["identity_digest"],      # 1.8.0 (finding F3): 1.7.0 left run_id empty here
                                 session=run["sessions"][-1]["started"], ceilings=ceil)
        skipped = [(t, "skipped", 0.0, "") for t in sorted(done_ok)]
        after = IN.completion(target)
        status, code = run_status(after, stopped=ceil.stopped)
        summary = dict(_session_summary(done + skipped, time.perf_counter() - t0), budgets=budgets,
                       retained_failures=sorted(failed - {s.task_id for s in work}), status=status, exit_code=code,
                       stopped=ceil.stopped, admission=adm, not_started=len(work) - len(done),
                       run={k: v for k, v in after.items() if k != "completed_ids"})
        LB._write_json(os.path.join(target, "label_summary.json"), summary)
    return summary


SCENARIO_TYPE = {"none": "none", "coherent": "coherent", "drift": "drift", "readout": "readout",
                 "crosstalk": "crosstalk", "temporal": "temporal"}


def scenario_type(name: str) -> str:
    return name.split("-")[0]


def mismatch_analyze(cfg: dict, out_dir: str, dev_dir: str, train_parts=("dev", "final")) -> dict:
    """Diagnostics and reactions under mismatch at *equal total budget* (1.7.0; review of Figure 6).

    The recommender is the frozen configuration of ``out_dir`` trained on its development labels (``train_parts``)
    without the base circuits the mismatch study scores. A mismatched task has the same features as the matched one
    (features use the snapshot only), so the model's choice does not change; what changes is the risk.

    The echo diagnostic spends d(B) shots (2 % of B, at least 200); a single deployment observes one statistic z and
    flags when |z| exceeds the critical value at level α (no correction: one test). Its flag probability is exact
    (binomial counts under the execution law). Every policy below spends exactly B shots in total:

    * ``TARGET@B`` — no diagnostic: the model's choice with all B shots ("the same shots spent on the target");
    * ``NO-DIAG@B−d`` — the model's choice at B − d, the diagnostic shots withheld (isolates the information value);
    * ``FIXED→ZNE-L``, ``FIXED→SBS-B`` — the 1.6.0 declared reactions: when flagged, that protocol at B − d;
    * ``DIAG-RESELECT`` — a learned reaction: when flagged, the protocol (or SBS-B) that minimized the mean log10 risk
      of the flagged-reaction policy on the training scenarios at that budget, trained *without the held-out
      discrepancy type* (leave-one-type-out; the matched scenario "none" is evaluated with a reaction trained on every
      mismatch type);
    * ``RANDOM→reaction`` — the same reaction triggered at random with the diagnostic's mean flag rate over the
      evaluated rows of that budget (matched rate, no information);
    * ``BEST`` — the per-task reference (identified on the selection half, scored on the evaluation half).

    Expected risks of flag-triggered policies are exact mixtures: P(flag)·R_react(B − d) + (1 − P(flag))·R_model(B − d)
    (the diagnostic's shots are independent of the protocol's). Detecting a discrepancy does not identify its physical
    mechanism; the question here is whether the diagnostic predicts or prevents recommendation failures."""
    import pandas as pd
    from scipy.stats import binom
    from . import binding as BD
    from . import dataset as DT
    from . import evaluate as EV
    from . import features as FE
    from . import mismatch as MM
    from . import policies as PO
    from .models import EPS0
    from ..methods import Runner
    m = cfg["mismatch"]
    alpha = float(m.get("alpha", 0.05))
    thr = MM.flag_threshold(alpha)
    mdir = os.path.join(out_dir, "mismatch")
    adir = os.path.join(mdir, "analysis")
    os.makedirs(adir, exist_ok=True)
    scen = {d["task_id"]: d["scenario"] for d in TX.read_json(os.path.join(mdir, "scenarios.json"))}
    ds, _ = BD.collect(mdir)
    dev = None
    for part in train_parts:
        d_, _ = BD.collect(os.path.join(os.path.dirname(os.path.normpath(dev_dir)), part))
        dev = d_ if dev is None else DT.concat(dev, d_)
    fam, sname = deployed_configuration(out_dir, {"deployed_family": m.get("model_family", "hgb"),
                                                  "deployed_set": m.get("model_set", "deployable")})
    held = {r["base_id"] for r in ds.rows}
    train = np.array([i for i, r in enumerate(dev.rows) if r["base_id"] not in held])
    pol = PO.Model(fam, FE.SETS[sname], sname).fit(dev, train)
    sbsb = PO.SBSBudget().fit(dev, train)
    LB.write_json(os.path.join(adir, "model_training.json"),
                  dict(policy=pol.name, train_rows=int(len(train)), train_parts=list(train_parts),
                       excluded_base_circuits=sorted(held & {r["base_id"] for r in dev.rows}),
                       train_base_circuits=len({dev.rows[i]["base_id"] for i in train})))
    grid = [int(b) for b in cfg["budgets"]]
    idx_of = {(r["task_id"], int(r["B"])): i for i, r in enumerate(ds.rows)}
    rng = np.random.default_rng(int(cfg.get("master_seed", 0)) + 77)
    draws_n = int(m.get("diagnostic_draws", 200))
    echo_cache, base = {}, []
    for i, r in enumerate(ds.rows):
        B = int(r["B"])
        if B not in grid:
            continue
        d = MM.diag_shots(B, m.get("diag_fraction", 0.02), m.get("diag_min", 200))
        iB = idx_of.get((r["task_id"], B - d))
        if iB is None:
            continue
        tid = r["task_id"]
        if tid not in echo_cache:
            spec = TK.TaskSpec.from_json(LB.load_record(mdir, tid)["spec"])
            built = TK.build(spec)
            phys = tuple(built.compiled.info["phys"])
            from .. import device
            noise = device.get_model(spec.supplied_backend, phys).equivalent_noise()
            from .mismatch import truth_runner
            rt = Runner(backend=spec.supplied_backend, layout=phys) if not spec.truth else truth_runner(spec, phys)
            rs = Runner(backend=spec.supplied_backend, layout=phys)
            echo_cache[tid] = (MM.echo_p0(rt, built.compiled, noise), MM.echo_p0(rs, built.compiled, noise))
        p_true, p_sup = echo_cache[tid]
        se = math.sqrt(max(p_sup * (1 - p_sup), 1e-12) / d)
        lo_k, hi_k = math.ceil(d * (p_sup - thr * se)) - 1, math.floor(d * (p_sup + thr * se)) + 1
        p_flag = float(binom.cdf(lo_k, d, p_true) + binom.sf(hi_k - 1, d, p_true))
        z = MM.echo_trials(p_true, p_sup, d, draws_n, rng)
        ch, _ = pol.choose(ds, np.array([i]))
        chd, _ = pol.choose(ds, np.array([iB]))
        fbd, _ = sbsb.choose(ds, np.array([iB]))
        j_ref = int(np.argmin(np.where(ds.F[i], ds.Y_sel[i], np.inf)))
        base.append(dict(i=i, iB=iB, scenario=scen.get(tid, "?"), type=scenario_type(scen.get(tid, "?")),
                         task_id=tid, base_id=r["base_id"], family=r["family"], B=B, d=d, p_true=p_true,
                         p_sup=p_sup, p_flag=p_flag, z=z, model=int(ch[0]), model_d=int(chd[0]), sbsb_d=int(fbd[0]),
                         ref=j_ref))
    P = len(ds.protocols)
    react_opts = list(range(P)) + ["SBS-B"]

    def r_react(b, a):
        if a == "SBS-B":
            return ds.Y_eval[b["iB"], b["sbsb_d"]]
        return ds.Y_eval[b["iB"], a] if ds.F[b["iB"], a] else np.nan

    def r_model_d(b):
        return ds.Y_eval[b["iB"], b["model_d"]]

    def mix(b, a, q=None):
        q = b["p_flag"] if q is None else q
        ra = r_react(b, a)
        return np.nan if not np.isfinite(ra) else q * ra + (1 - q) * r_model_d(b)

    def learn(rows):
        """Per budget: the reaction minimizing the circuit-weighted mean log10 risk on ``rows``."""
        out = {}
        for B in sorted({b["B"] for b in rows}):
            rb = [b for b in rows if b["B"] == B]
            g = np.array([b["base_id"] for b in rb])
            best, best_v = "SBS-B", np.inf
            for a in react_opts:
                v = np.array([math.log10(mix(b, a) + EPS0) if np.isfinite(mix(b, a)) else np.nan for b in rb])
                if np.isnan(v).any():
                    continue
                mv = EV.wmean(v, g)
                if mv < best_v - 1e-12:
                    best, best_v = a, mv
            out[B] = best
        return out
    types = sorted({b["type"] for b in base} - {"none"})
    reaction = {}
    for T in types + ["none"]:
        tr_rows = [b for b in base if b["type"] != T and (T != "none" or b["type"] != "none")]
        if T == "none":
            tr_rows = [b for b in base if b["type"] != "none"]
        reaction[T] = learn(tr_rows)
    react_rows = [dict(held_out_type=T, B=B, reaction=(ds.protocols[a] if a != "SBS-B" else "SBS-B"))
                  for T, mp in reaction.items() for B, a in sorted(mp.items())]
    qrate = {}
    for b in base:
        qrate.setdefault((b["type"], b["B"]), []).append(b["p_flag"])
    qrate = {k: float(np.mean(v)) for k, v in qrate.items()}
    pol_rows, diag_rows, draw_rows = [], [], []
    for b in base:
        a_learn = reaction[b["type"]][b["B"]]
        i = b["i"]
        vals = {"TARGET@B": ds.Y_eval[i, b["model"]], "NO-DIAG@B-d": r_model_d(b),
                "FIXED→ZNE-L": mix(b, ds.protocols.index("ZNE-L")), "FIXED→SBS-B": mix(b, "SBS-B"),
                "DIAG-RESELECT": mix(b, a_learn), "RANDOM→reaction": mix(b, a_learn, qrate[(b["type"], b["B"])]),
                "BEST": ds.Y_eval[i, b["ref"]]}
        for name, val in vals.items():
            pol_rows.append(dict(scenario=b["scenario"], type=b["type"], task_id=b["task_id"], base_id=b["base_id"],
                                 family=b["family"], B=b["B"], policy=name, risk=float(val),
                                 mean_log10_risk=float(math.log10(val + EPS0)) if np.isfinite(val) else np.nan,
                                 model_choice=ds.protocols[b["model"]], p_flag=b["p_flag"], diag_shots=b["d"],
                                 reaction=(ds.protocols[a_learn] if a_learn != "SBS-B" else "SBS-B")
                                 if name in ("DIAG-RESELECT", "RANDOM→reaction") else ""))
        lr_model = math.log10((ds.Y_eval[i, b["model"]] + EPS0) / (ds.Y_eval[i, b["ref"]] + EPS0))
        z = b["z"]
        diag_rows.append(dict(scenario=b["scenario"], type=b["type"], task_id=b["task_id"], base_id=b["base_id"],
                              B=b["B"], d=b["d"], p_true=b["p_true"], p_sup=b["p_sup"], threshold=thr, alpha=alpha,
                              flag_rate=b["p_flag"], z_q10=float(np.quantile(np.abs(z), 0.1)),
                              z_median=float(np.median(np.abs(z))), z_q90=float(np.quantile(np.abs(z), 0.9)),
                              model_log_regret=lr_model, model_choice=ds.protocols[b["model"]]))
        for k_, zz in enumerate(z[:20]):                       # single-deployment statistics (Figure 6B)
            draw_rows.append(dict(scenario=b["scenario"], type=b["type"], task_id=b["task_id"], B=b["B"], draw=k_,
                                  abs_z=float(abs(zz)), flagged=bool(abs(zz) > thr), model_log_regret=lr_model))
    pdf = pd.DataFrame(pol_rows)
    pdf.to_csv(os.path.join(mdir, "policies.csv"), index=False, encoding="utf-8")
    pd.DataFrame(diag_rows).to_csv(os.path.join(mdir, "diagnostics.csv"), index=False, encoding="utf-8")
    pd.DataFrame(draw_rows).to_csv(os.path.join(adir, "deployment_draws.csv"), index=False, encoding="utf-8")
    pd.DataFrame(react_rows).to_csv(os.path.join(adir, "reactions.csv"), index=False, encoding="utf-8")
    # equal-budget paired differences against the two no-reaction comparators, per discrepancy type
    eq = []
    for (T, B), e in pdf[pdf.B.isin(grid)].groupby(["type", "B"]):
        wide = e.pivot_table(index=["task_id", "base_id"], columns="policy", values="mean_log10_risk").reset_index()
        g = wide.base_id.to_numpy()
        for ref in ("TARGET@B", "NO-DIAG@B-d"):
            for p_ in ("FIXED→ZNE-L", "FIXED→SBS-B", "DIAG-RESELECT", "RANDOM→reaction", "NO-DIAG@B-d"):
                if p_ == ref or p_ not in wide or ref not in wide:
                    continue
                diff = (wide[p_] - wide[ref]).to_numpy(float)
                ci = EV.bootstrap({"d": diff}, g, 1000, 0)[0]["d"]
                eq.append(dict(type=T, B=int(B), policy=p_, reference=ref, diff_log10=ci[0], lo=ci[1], hi=ci[2],
                               base_circuits=int(len(set(g.tolist()))), rows=int(len(diff))))
    pd.DataFrame(eq).to_csv(os.path.join(adir, "equal_budget.csv"), index=False, encoding="utf-8")
    # discrimination of a single deployment's |z| for a recommendation failure (log regret > 0.3)
    disc = []
    dr = pd.DataFrame(draw_rows)
    for B, e in dr.groupby("B"):
        fail = e.model_log_regret > 0.3
        if fail.any() and (~fail).any():
            from scipy.stats import mannwhitneyu
            u = mannwhitneyu(e.abs_z[fail], e.abs_z[~fail]).statistic
            auc = float(u / (fail.sum() * (~fail).sum()))
        else:
            auc = math.nan
        disc.append(dict(B=int(B), deployments=int(len(e)), failures=int(fail.sum()), auc_abs_z=auc,
                         flag_rate_failures=float(e.flagged[fail].mean()) if fail.any() else math.nan,
                         flag_rate_ok=float(e.flagged[~fail].mean()) if (~fail).any() else math.nan))
    pd.DataFrame(disc).to_csv(os.path.join(adir, "discrimination.csv"), index=False, encoding="utf-8")
    null = [dict(p_sup=row["p_sup"], d=row["d"], simulated_false_positive=MM.null_rate(row["p_sup"], row["d"], alpha,
                                                                                        R=4000))
            for row in diag_rows if row["scenario"] == "none"][:40]
    pd.DataFrame(null).to_csv(os.path.join(mdir, "null_check.csv"), index=False, encoding="utf-8")
    LB.summarize(mdir).to_csv(os.path.join(adir, "labels.csv"), index=False, encoding="utf-8")
    return dict(rows=len(pol_rows), diagnostics=len(diag_rows), reactions=len(react_rows))


# ------------------------------------------------------------------ controlled regime study (decision D23)
def _device_modifiers(device: str) -> dict:
    """The modifiers of a device label ``fake_fez@g2=2,ro=0.5`` as a dict (empty for a plain device; ``s`` the overall
    error scale). Parsed with the device module's own grammar."""
    from .. import device as DV
    try:
        mods = DV.parse_device(device)[1]
        return {k: v for k, v in mods.items() if k != "eng" and v != DV.DEVICE_MODS[k]}
    except Exception:  # noqa: BLE001 — a label this grammar does not parse is reported as unmodified
        if "@" not in device:
            return {}
        out = {}
        for kv in device.split("@", 1)[1].split(","):
            k, _, v = kv.partition("=")
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
        return out


def regime_analyze(cfg: dict, run_dir: str, sub: str = "regimes", train_parts=("dev", "final")) -> dict:
    """Reference preference, recommendation and regret per (controlled slice, noise scale, budget) of the regime
    study labeled into ``run_dir``/``sub``. The recommender is the configuration frozen in ``run_dir`` trained on the
    development set without the regime study's base circuits; it abstains to the budget-dependent fixed policy when
    its margin is below the deployment threshold chosen on the development rows (``analysis/selective_operating.csv``).

    Reference map: the protocol with the lowest MSE over both label halves (2r trials); "unresolved" when the
    runner-up's MSE lies within 1.96 standard errors of the difference (independent trials; overlap of intervals is
    not evidence of equivalence and no equivalence procedure is implemented). Regret: the main evaluation's
    convention — the evaluation-half MSE of the policy's protocol against that of the protocol identified on the
    selection half. Writes ``sub``/analysis/regime_cells.csv and regime_protocols.csv."""
    import pandas as pd
    from . import dataset as DT
    from . import features as FE
    from . import policies as PO
    from . import binding as BD
    a = cfg.get("analysis", {})
    rdir = os.path.join(run_dir, sub)
    ds, _ = BD.collect(rdir)
    dev = None
    for part in train_parts:
        p_ = os.path.join(run_dir, part)
        if os.path.isdir(p_):
            d_, _ = BD.collect(p_)
            dev = d_ if dev is None else DT.concat(dev, d_)
    held = {r["base_id"] for r in ds.rows}
    train = np.array([i for i, r in enumerate(dev.rows) if r["base_id"] not in held])
    fam, sname = deployed_configuration_or_default(run_dir, a)
    pol = PO.Model(fam, FE.SETS[sname], sname).fit(dev, train)
    fb = PO.SBSBudget().fit(dev, train)
    tau, tau_src = -math.inf, "none (every recommendation accepted)"
    frz_p = os.path.join(run_dir, "freeze.json")
    frz = TX.read_json(frz_p) if os.path.exists(frz_p) else {}
    for src, getter in ((frz_p, lambda: frz.get("deploy_tau")),
                        (os.path.join(run_dir, "analysis", "nested_operating.csv"),
                         lambda: pd.read_csv(os.path.join(run_dir, "analysis", "nested_operating.csv"),
                                             encoding="utf-8").deploy_tau.iloc[0]),
                        (os.path.join(run_dir, "analysis", "selective_operating.csv"),
                         lambda: pd.read_csv(os.path.join(run_dir, "analysis", "selective_operating.csv"),
                                             encoding="utf-8").deploy_tau.iloc[0])):
        if os.path.exists(src):
            v = getter()
            if v is not None and np.isfinite(float(v)) or (v is not None and float(v) in (math.inf, -math.inf)):
                tau, tau_src = float(v), os.path.basename(src)
                break
    idx = np.arange(len(ds.rows))
    P = pol.predict(ds, idx)
    ch, _ = pol.choose(ds, idx)
    fch, _ = fb.choose(ds, idx)
    from . import evaluate as EV
    sm, sf = EV.scores(ds, idx, ch), EV.scores(ds, idx, fch)
    cells, prot = [], []
    cache = {}
    for r, i in enumerate(idx):
        row = ds.rows[i]
        tid = row["task_id"]
        if tid not in cache:
            cache[tid] = LB.load_task(rdir, tid)
        rec, arr = cache[tid]
        j_b = rec["budgets"].index(row["B"])
        mu = rec["ideal"]
        pooled, se = {}, {}
        fresh = LB.plan_of(rec).get("mode") == "fresh"
        for j, p in enumerate(ds.protocols):
            if not ds.F[i, j]:
                continue
            x = np.concatenate([arr["est"][0, j, j_b], arr["est"][1, j, j_b]])
            dz = np.concatenate([arr["design"][0, j, j_b], np.where(arr["design"][1, j, j_b] >= 0,
                                                                     arr["design"][1, j, j_b] + 1000, -1)])
            loss = (np.where(np.isfinite(x), x, 0.0) - mu) ** 2
            st_ = LB.design_stats(loss, dz)
            pooled[p] = float(loss.mean())
            se[p] = float(LB.risk_se(st_, dict(mode="fresh" if fresh else "fixed-shared"),
                                     PR.PORTFOLIO[p].learning))
            prot.append(dict(task_id=tid, base_id=row["base_id"], family=row["family"], device=row["device"],
                             observable=row["observable"], layout=row["layout"], B=row["B"], protocol=p,
                             mse_pooled=pooled[p], se_pooled=se[p], mse_eval=float(ds.Y_eval[i, j]),
                             mse_sel=float(ds.Y_sel[i, j]), predicted_log_risk=float(P[r, j]),
                             chosen=bool(j == ch[r])))
        order = sorted(pooled, key=pooled.get)
        best = order[0] if order else None
        unresolved = bool(len(order) > 1 and abs(pooled[order[1]] - pooled[best]) <= 1.96 * math.sqrt(
            se[order[1]] ** 2 + se[best] ** 2))
        feas = [j for j in range(len(ds.protocols)) if ds.F[i, j] and np.isfinite(P[r, j])]
        o = sorted(feas, key=lambda j: P[r, j])
        if len(o) >= 2:
            s1 = pol.rm.cv_rmse.get(ds.protocols[o[0]], np.nan)
            s2 = pol.rm.cv_rmse.get(ds.protocols[o[1]], np.nan)
            margin = float((P[r, o[1]] - P[r, o[0]]) / math.sqrt(s1 ** 2 + s2 ** 2))
        else:
            margin = math.inf
        abst = bool(margin < tau)
        dev_s = row["device"]
        mods = _device_modifiers(dev_s)
        s_scale = float(mods.get("s", 1.0))
        comp = ",".join("%s=%g" % kv for kv in sorted(mods.items())) or "none"
        cells.append(dict(task_id=tid, base_id=row["base_id"], family=row["family"], device=dev_s,
                          noise_scale=s_scale, noise_change=comp, lam_total=float(row["feats"].get("lam_total", np.nan)),
                          lam_lc=float(row["feats"].get("lam_lc", np.nan)), observable=row["observable"], layout=row["layout"], B=row["B"],
                          n_feasible=len(pooled), trials_per_candidate=int(2 * arr["est"].shape[-1]),
                          reference=best, runner_up=order[1] if len(order) > 1 else None, unresolved=unresolved,
                          model_choice=ds.protocols[ch[r]], margin=margin, abstained=abst,
                          fallback_choice=ds.protocols[fch[r]],
                          policy_choice=ds.protocols[fch[r]] if abst else ds.protocols[ch[r]],
                          log_regret_model=float(sm["log_ratio"][r]), regret_model=float(sm["regret"][r]),
                          log_regret_policy=float((sf if abst else sm)["log_ratio"][r]),
                          log_regret_fallback=float(sf["log_ratio"][r]), deploy_tau=tau, tau_source=tau_src,
                          configuration="MODEL[%s|%s]" % (fam, sname), statement="descriptive",
                          resolution="unresolved: top two within 1.96 approximate SE of their difference (pointwise; "
                                     "design-level SE for learning protocols under fresh designs)"))
    adir = os.path.join(rdir, "analysis")
    os.makedirs(adir, exist_ok=True)
    pd.DataFrame(cells).to_csv(os.path.join(adir, "regime_cells.csv"), index=False, encoding="utf-8")
    tf = {}
    for row in ds.rows:
        tf[row["task_id"]] = dict(task_id=row["task_id"], base_id=row["base_id"], family=row["family"],
                                  device=row["device"], observable=row["observable"], inputs=row["inputs"],
                                  layout=row["layout"], **row["feats"])
    pd.DataFrame(list(tf.values())).to_csv(os.path.join(adir, "regime_features.csv"), index=False, encoding="utf-8")
    LB.summarize(rdir).to_csv(os.path.join(adir, "labels.csv"), index=False, encoding="utf-8")
    pd.DataFrame(prot).to_csv(os.path.join(adir, "regime_protocols.csv"), index=False, encoding="utf-8")
    LB._write_json(os.path.join(adir, "model_training.json"),
                   dict(policy="MODEL[%s|%s]" % (fam, sname), train_rows=int(len(train)), dev_rows=len(dev.rows),
                        train_parts=list(train_parts), tau_source=tau_src,
                        excluded_base_circuits=sorted(held & {r["base_id"] for r in dev.rows}),
                        deploy_tau=tau if math.isfinite(tau) else ("+inf" if tau > 0 else "-inf")))   # strict JSON
    return dict(cells=len(cells), protocols=len(prot))


FEATURE_PROTOCOLS = ["RAW", "RO", "ZNE-L", "ZNE-R", "PEC"]     # the supplied-model quantities the features read


def _peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (``resource`` on POSIX; psutil's current RSS elsewhere, labelled
    by the caller)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0
    except ImportError:                               # Windows: no resource module
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def deployment_bundle_of(run_dir: str, prefer_deployable: bool = True) -> str | None:
    """Path of a bundle trained from this run (its freeze's development identity), preferring one whose features need
    no simulation; None when the run has no freeze or no bundle was built from it. 1.9.0: the run's own bundles
    (``run_dir``/bundles, where ``bundle`` writes by default) are searched before the shipped ones."""
    from . import deploy as DP
    frz = os.path.join(run_dir, "freeze.json")
    if not os.path.exists(frz):
        return None
    ident = TX.read_json(frz).get("dev_identity")
    for base in (os.path.join(run_dir, "bundles"), DP.BUNDLE_DIR):
        if not os.path.isdir(base):
            continue
        found = []
        for name in sorted(os.listdir(base)):
            bj = os.path.join(base, name, "bundle.json")
            if not os.path.exists(bj):
                continue
            m = TX.read_json(bj)
            if ident and (m.get("training") or {}).get("freeze", {}).get("dev_identity") == ident:
                sim = "simulated" in str(m.get("feature_set", ""))
                found.append((sim if prefer_deployable else not sim, name))
        if found:
            return os.path.join(base, sorted(found)[0][1])
    return None


REJECTION_WIDTHS = (12, 24, 40)


def wide_chain_task(n: int, budget: int = 100000, device: str = "fake_fez"):
    """A deployment task as the Playground or an experiment document would send it: a GHZ-type chain (one RY, a CX
    ladder) on the device's calibration-best chain, observable Z on the middle qubit given as a compact Pauli term (no
    dense 2^n vector) — the probe of the early-rejection timing."""
    from ..phase.families import best_chain
    from . import deploy as DP
    ops = (("ry", (0,), (1.0,), "gate", ""),) + tuple(("cx", (i, i + 1), (), "gate", "") for i in range(n - 1))
    return DP.DeploymentTask(n, ops, tuple("0" * n), "Z" * n, False, ("z_mid", "Z on the middle bit", ()), device,
                             budget, initial_layout=tuple(best_chain(device, n)), obs_terms=(((n // 2,), 1.0),))


def profile(run_dir: str, widths=(3, 4, 5, 6, 7), family: str = "ising", params: dict | None = None,
            repeats: int = 3, bundle: str | None = None, rejection: bool = True) -> list:
    """Classical cost of one recommendation at deployment against circuit width, through the deployment path the
    Playground and ``recommend`` use (1.7.0, review of supp_runtime: components separated; 1.8.0: the full upstream
    path and early rejection, finding F2):

    * ``compile_s`` — compilation for the device;
    * ``deployable_s`` — the deployable features of the run's deployable set (structure, observable, light cone,
      calibration-derived noise sums with factorized calibration, the cheap protocol overheads): no simulation of the
      circuit; timed without the feature cache (1.8.0), so every repeat recomputes;
    * ``simulated_s`` — the simulator-only augmentation (supplied-model per-shot variances and signal by exact
      density-matrix simulation of the snapshot, ideal-state entropies), exponential in width;
    * ``inference_s`` — the frozen recommender's prediction from the feature vector;
    * ``preflight_structural_s`` — the structural preflight of every candidate of the portfolio (1.8.0);
    * ``preflight_design_s`` — the design-level preflight of the learning candidates (training design built with its
      deployment seed, ideal training values by state vector) (1.8.0);
    * ``recommend_s`` — the whole ``deploy.recommend`` call with the run's deployable bundle (guard, compile, features,
      OOD, preflight in the model's order, margin rule, fallback), when that bundle exists (1.8.0);
    * ``label_s`` — for comparison, benchmark-label generation: the median wall time of labeling one development task
      of that width (every protocol, every budget, both halves; from the labels' records), a cost of building the
      study, not of a recommendation.

    With ``rejection``, rows with ``kind = "rejection"`` time the recommendation of wide GHZ-type chains
    (``wide_chain_task``; 12, 24, 40 qubits, outside every training width): the bounded path must answer
    (out-of-distribution fallback or an unsupported-input error) without a dense state of the circuit.

    Timed in this process after a warm-up of the libraries; ``peak_rss_mb`` is the process's peak resident memory
    (``rss_kind`` says how it was measured)."""
    from . import dataset as DT
    from . import deploy as DP
    from . import features as FE
    from . import policies as PO
    from . import preflight as PF
    from . import protocols as PR
    a = {}
    fam, sname = deployed_configuration_or_default(run_dir, a)
    dev = DT.build(os.path.join(run_dir, "dev"))
    pol = PO.Model(fam, FE.SETS[sname], sname).fit(dev, np.arange(len(dev.rows)))
    lab_s = {}
    for tid in LB.task_ids(os.path.join(run_dir, "dev")):
        rec = LB.load_record(os.path.join(run_dir, "dev"), tid)
        if rec.get("status") == "ok" and rec.get("seconds") is not None:
            lab_s.setdefault(int(rec["spec"]["n"]), []).append(float(rec["seconds"]))
    params = dict(params or {"steps": 2})
    dep_set = "deployable18" if "deployable18" in FE.SETS and "18" in sname else "deployable"
    sim_set = dep_set + "+simulated"
    dep_names = [n for n in FE.SETS[dep_set]]
    sim_names = [n for n in FE.SETS[sim_set] if FE.BY_NAME[n].info == "simulated"]
    keys = [k for k in dev.protocols if k in PR.PORTFOLIO]
    learning = [k for k in keys if PR.PORTFOLIO[k].learning]
    bpath = bundle or deployment_bundle_of(run_dir)
    bnd = DP.load_bundle(bpath) if bpath else None
    try:
        import resource  # noqa: F401
        rss_kind = "peak (resource)"
    except ImportError:
        rss_kind = "current (psutil; no peak on this platform)"
    out = []
    DP.from_spec(TK.make_spec(family, 3, **params), 100000).compile()        # warm-up (imports, device data)
    for n in widths:
        for rep in range(int(repeats)):
            task = DP.from_spec(TK.make_spec(family, int(n), **params), 100000)
            t0 = time.perf_counter()
            comp = task.compile()
            t1 = time.perf_counter()
            v1, r1 = DP.features(task, dep_names, compiled=comp, use_cache=False)
            t2 = time.perf_counter()
            v2, r2 = DP.features(task, sim_names, allow_simulation=True, compiled=comp, use_cache=False)
            t3 = time.perf_counter()
            row = dict(v1, **v2)
            x = np.array([[row.get(k, np.nan) for k in pol.features]], float)
            t4 = time.perf_counter()
            pol.rm.predict(x)
            t5 = time.perf_counter()
            for k in keys:
                PF.structural(comp, k, task.budget, symmetry=task.symmetry)
            t6 = time.perf_counter()
            obs = task.obs()
            for k in learning:
                PF.design(comp, obs, k, task.budget, supplied=task.supplied_backend, seed=DP.design_seed(task, k),
                          symmetry=task.symmetry)
            t7 = time.perf_counter()
            rec_s, rec_status = math.nan, ""
            if bnd is not None:
                DP._FEATURE_CACHE.clear()
                t8 = time.perf_counter()
                rr = DP.recommend(task, bnd)
                rec_s, rec_status = time.perf_counter() - t8, rr["status"]
            ls = lab_s.get(int(n))
            out.append(dict(kind="width", family=family, n=int(n), repeat=rep, compile_s=t1 - t0,
                            deployable_s=t2 - t1, simulated_s=t3 - t2, inference_s=t5 - t4,
                            preflight_structural_s=t6 - t5, preflight_design_s=t7 - t6,
                            preflight_candidates=len(keys), preflight_learning=len(learning),
                            recommend_s=rec_s, recommend_status=rec_status,
                            bundle=os.path.basename(bpath) if bpath else "",
                            label_s=float(np.median(ls)) if ls else math.nan, label_tasks=len(ls or []),
                            peak_rss_mb=_peak_rss_mb(), rss_kind=rss_kind,
                            configuration="MODEL[%s|%s]" % (fam, sname), feature_set_timed=dep_set))
    if rejection and bnd is not None:
        for n in REJECTION_WIDTHS:
            DP._FEATURE_CACHE.clear()
            t0 = time.perf_counter()
            try:
                rr = DP.recommend(wide_chain_task(int(n)), bnd)
                status, note = rr["status"], "; ".join(rr.get("ood", []))[:200]
            except DP.Unsupported as e:
                status, note = "unsupported", str(e)[:200]
            out.append(dict(kind="rejection", family="ghz-chain", n=int(n), repeat=0,
                            recommend_s=time.perf_counter() - t0,
                            recommend_status=status, note=note, bundle=os.path.basename(bpath),
                            peak_rss_mb=_peak_rss_mb(), rss_kind=rss_kind,
                            configuration=bnd.meta["configuration"]))
    return out

