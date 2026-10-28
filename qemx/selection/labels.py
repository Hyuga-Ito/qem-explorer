r"""Supervised labels: per-trial outcomes of every protocol on every task and budget (1.6.0; 1.7.0 record format).

For each task 𝒯 and budget B, every protocol a is run for 2r complete trials (``emulate.trials``) in two halves with
separate random streams:

* the **selection half** (r trials) gives the label a model may learn from, R̂_sel(𝒯, a, B) = mean of squared
  errors (theory companion Eq. 1.16);
* the **evaluation half** (r trials) gives an independent estimate R̂_eval used only to score decisions. Choosing on
  R̂_sel and scoring on R̂_eval removes the optimizer's curse from reported regret (Smith & Winkler 2006).

Learning protocols (CDR, NIL) also draw a *training design* (``emulate.DesignPlan``). With fresh designs (1.7.0) the
two halves use disjoint, independent design pools, so they are independent unconditionally; with the 1.6.0 fixed
designs shared by both halves they are independent only given those designs, and the labels estimate the risk
conditional on them. The record states which (``design_plan``, ``estimand``); the trial→design index is stored with
the trials (``design``), every design's seed, identity and diagnostics in the record (``design_records``).

Per-trial estimates, failure flags and design indices are stored (``tasks/<task_id>.npz``) — never only summaries —
together with a JSON record (allocations with their status, ledgers, timings, seeds; the code and environment of a
labeling session are in ``provenance.json`` and the run binding in ``run.json``, see ``run``). A task whose record
exists with the same configuration digest is skipped (checkpoint/resume); a record from another configuration is
refused unless ``force``. Writes are atomic (temporary file, then rename).

Record status: ``ok`` (labels and arrays written), ``error`` (the task could not be built or prepared: a software or
modeling failure, with ``error_kind`` and the message), ``timeout`` and ``memory`` (written by the supervisor,
``run.label``). An ``ok`` record whose arrays are missing or inconsistent is *corrupted* data and is never skipped
silently (``CorruptedRecord``).

Costs per protocol and budget (kept apart, theory companion §1.3.3): shots by role (the ledger), circuit instances
sent, distinct circuits, a duration-weighted execution time T_exec ≈ Σ_j N_j (t_j + t_reset) from the device's ALAP
schedule of each executed circuit (t_reset = 250 µs, IBM's default repetition delay), and the classical seconds
the emulation took (not a deployment cost).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

from .. import textio as TX
from . import emulate as EM
from . import protocols as PR
from . import task as TK

SCHEMA = "qemx.selection.labels/2"
SCHEMAS_READ = ("qemx.selection.labels/1", SCHEMA)
T_RESET = 250e-6
STATUSES = ("ok", "error", "timeout", "memory")


class CorruptedRecord(RuntimeError):
    """A label record that claims success but whose arrays are missing or do not match it."""


def config_digest(budgets, protocols, R: int, master_seed: int, designs=1) -> str:
    """Digest of everything that changes a task's labels. 1.6.0 configurations (an integer number of fixed designs)
    keep their digests; a fresh-design plan adds its own entry."""
    d = dict(schema="qemx.selection.labels/1", budgets=[int(b) for b in budgets], R=int(R), seed=int(master_seed),
             protocols=[PR.PORTFOLIO[k].to_json() for k in protocols])
    plan = EM.DesignPlan.from_config(designs)
    if plan.mode == "fixed-shared":
        if plan.count != 1:
            d["learning_designs"] = int(plan.count)
    else:
        d["learning_designs"] = plan.to_json()
        d["record_schema"] = SCHEMA
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def rng_for(master_seed: int, task_id: str, B: int, key: str, half: int) -> np.random.Generator:
    """An independent random stream per (task, budget, protocol, half): SeedSequence of integers derived from the
    identifiers, so a resumed or reordered run reproduces every trial."""
    h = int(hashlib.sha256(("%s|%d|%s|%d" % (task_id, int(B), key, int(half))).encode()).hexdigest()[:15], 16)
    return np.random.default_rng(np.random.SeedSequence([int(master_seed), h]))


def _durations(prep, key) -> list:
    """Scheduled durations (s) of the target circuits a protocol executes (ZNE: one per fold scale), from the
    device's ALAP schedule with its calibrated gate durations."""
    from .. import device
    from ..context_noise import alap_schedule
    built = prep.built
    dm = device.get_model(built.spec.supplied_backend, tuple(built.compiled.info["phys"]))
    specs = [built.compiled.implemented]
    if PR.PORTFOLIO[key].estimator == "zne":
        from ..transforms import fold_global
        lams = dict(PR.PORTFOLIO[key].params).get("scales", (1.0,))
        specs = [fold_global(built.compiled.implemented, float(l)) for l in lams]
    return [float(alap_schedule(sp, dm.durations)[3]) for sp in specs]


def costs(prep, key: str, alloc: dict) -> dict:
    """Cost fields of one allocation (shots by role are the ledger itself)."""
    if not alloc.get("feasible"):
        return dict(t_exec=math.nan, circuits_sent=0)
    ts = _durations(prep, key)
    t_tgt = ts[0]
    N = alloc["N"]
    split = alloc.get("split") or [N]
    t_exec = sum(nj * (tj + T_RESET) for nj, tj in zip(split, ts if len(ts) == len(split) else [t_tgt] * len(split)))
    t_exec += alloc["ledger"]["training"] * (t_tgt + T_RESET)        # training circuits: like the target in size
    t_exec += alloc["ledger"]["calibration"] * T_RESET                # calibration circuits: a layer of X at most
    est = PR.PORTFOLIO[key].estimator
    sent = (N if est == "pec" else len(split)) + alloc.get("train_circuits", 0) + alloc.get("c", 0)
    return dict(t_exec=float(t_exec), circuits_sent=int(sent))


def error_record(spec: TK.TaskSpec, cfg_digest: str, status: str, kind: str, message: str, seconds: float,
                 **extra) -> dict:
    """The record of a task without labels (``status`` error, timeout or memory)."""
    if status not in STATUSES or status == "ok":
        raise ValueError(status)
    return dict(schema=SCHEMA, task_id=spec.task_id, base_id=spec.base_id, spec=spec.to_json(),
                config_digest=cfg_digest, status=status, error_kind=kind, error=str(message)[:2000],
                seconds=float(seconds), **extra)


def label_task(spec: TK.TaskSpec, out_dir: str, budgets=PR.BUDGETS, protocols=None, R: int = 200,
               master_seed: int = 20261019, force: bool = False, designs=1, run_id: str = "") -> dict:
    """Label one task (both halves, every budget and protocol); returns its record. Resumable. ``designs``: the
    learning protocols' design plan (``emulate.DesignPlan`` or its configuration value). ``run_id``: the run this
    record belongs to (``run.json``)."""
    protocols = list(protocols or PR.PORTFOLIO_16)
    plan = EM.DesignPlan.from_config(designs)
    cfg = config_digest(budgets, protocols, R, master_seed, designs)
    tid = spec.task_id
    os.makedirs(os.path.join(out_dir, "tasks"), exist_ok=True)
    jpath = os.path.join(out_dir, "tasks", tid + ".json")
    if os.path.exists(jpath):
        old = load_record(out_dir, tid)
        if old.get("config_digest") == cfg and old.get("status") == "ok":
            load_arrays(out_dir, tid, old)              # an ok record must have consistent arrays (else: corrupted)
            return dict(old, skipped=True)               # checkpoint: labeled by an earlier session (not saved)
        if old.get("config_digest") != cfg and not force:
            raise RuntimeError("task %s was labeled with another configuration (%s ≠ %s); use a new output "
                               "directory or force" % (tid, old.get("config_digest"), cfg))
    t0 = time.perf_counter()
    rec = dict(schema=SCHEMA, task_id=tid, base_id=spec.base_id, spec=spec.to_json(), config_digest=cfg,
               budgets=[int(b) for b in budgets], protocols=protocols, R=int(R), master_seed=int(master_seed),
               learning_designs=plan.to_json(), design_plan=dict(mode=plan.mode, count=plan.count,
                                                                 per_half=plan.per_half),
               estimand=plan.estimand, run_id=str(run_id), seed_scheme=int(plan.seed_scheme),
               crn_group=("" if plan.seed_scheme == 1 else (spec.task_id if plan.crn == "task" else spec.base_id)))
    try:
        built = TK.build(spec)
    except Exception as e:                           # noqa: BLE001 - a task that cannot be built is recorded
        r = error_record(spec, cfg, "error", "build", "%s: %s" % (type(e).__name__, str(e)[:500]),
                         time.perf_counter() - t0, run_id=str(run_id))
        write_json(jpath, r)
        return r
    try:
        prep = EM.prepare(built, protocols, seed=master_seed % (2 ** 31), designs=plan)
    except Exception as e:                           # noqa: BLE001
        r = error_record(spec, cfg, "error", "prepare", "%s: %s" % (type(e).__name__, str(e)[:500]),
                         time.perf_counter() - t0, run_id=str(run_id))
        write_json(jpath, r)
        return r
    from . import features as FE
    try:
        rec["features"] = FE.task_features(prep)
    except Exception as e:                           # noqa: BLE001 - features missing are recorded, labels still made
        rec["features"] = {}
        rec["features_error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    P, nB = len(protocols), len(budgets)
    est = np.full((2, P, nB, int(R)), np.nan)
    flags = np.zeros((2, P, nB, len(EM.FLAGS), int(R)), bool)
    design = np.full((2, P, nB, int(R)), EM.NO_DESIGN, np.int16)
    allocs, cost_rows, secs = [], [], np.zeros((P, nB))
    for i, k in enumerate(protocols):
        for j, B in enumerate(budgets):
            a_json = None
            for half in (0, 1):
                stream = PR.PORTFOLIO[k].crn_with or k          # 1.8.0: declared common random numbers
                tr = EM.trials(prep, k, int(B), int(R), rng_for(master_seed, tid, B, stream, half), half=half)
                est[half, i, j] = tr.estimates
                for f, name in enumerate(EM.FLAGS):
                    flags[half, i, j, f] = tr.flags[name]
                if tr.design is not None:
                    design[half, i, j] = tr.design
                secs[i, j] += tr.seconds
                a_json = tr.allocation
            allocs.append(a_json)
            cost_rows.append(dict(protocol=k, B=int(B), **costs(prep, k, a_json)))
    rec.update(status="ok", ideal=float(built.ideal), task=built.record, allocations=allocs, costs=cost_rows,
               method_status={k: ("ok" if not isinstance(v, Exception) else "%s: %s" % (type(v).__name__, str(v)[:300]))
                              for k, v in prep.truth.items()},
               design_records=prep.design_records,
               design_status={k: [r_.get("status", "ok") if r_.get("status") != "error" else r_.get("error", "error")
                                  for r_ in v] for k, v in prep.design_records.items()},
               prepare_seconds={k: float(v) for k, v in prep.seconds.items()},
               seconds=time.perf_counter() - t0, flags=list(EM.FLAGS))
    npz = os.path.join(out_dir, "tasks", tid + ".npz")
    _write_npz(npz, est=est, flags=flags, trial_seconds=secs, design=design)
    rec["arrays"] = dict(shape=list(est.shape), sha256=file_digest(npz))
    write_json(jpath, rec)
    return rec


def write_json(path: str, obj) -> None:
    """Atomic UTF-8 JSON (temporary file, then rename)."""
    TX.write_json(path, obj, indent=1, default=_np, atomic=True)


_write_json = write_json                                 # 1.6.0 name, kept for callers


def _write_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp%d.npz" % os.getpid()
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_record(out_dir: str, task_id: str) -> dict:
    """The JSON record of one task (any status)."""
    rec = TX.read_json(os.path.join(out_dir, "tasks", task_id + ".json"))
    if rec.get("schema") not in SCHEMAS_READ:
        raise CorruptedRecord("task %s: unknown record schema %r" % (task_id, rec.get("schema")))
    return rec


def load_arrays(out_dir: str, task_id: str, rec: dict | None = None) -> dict:
    """The trial arrays of an ``ok`` record; ``CorruptedRecord`` when they are missing or do not match it. Arrays are
    loaded without pickle. Records of 1.6.0 have no ``design`` array: it is reconstructed from their declared
    round-robin assignment (fixed designs shared by both halves)."""
    rec = rec if rec is not None else load_record(out_dir, task_id)
    if rec.get("status") != "ok":
        raise ValueError("task %s has status %s: no arrays" % (task_id, rec.get("status")))
    path = os.path.join(out_dir, "tasks", task_id + ".npz")
    if not os.path.exists(path):
        raise CorruptedRecord("task %s: the record says ok but %s is missing" % (task_id, path))
    want = (rec.get("arrays") or {}).get("sha256")
    if want and file_digest(path) != want:
        raise CorruptedRecord("task %s: %s does not match the digest in its record" % (task_id, path))
    try:
        with np.load(path, allow_pickle=False) as z:
            arr = {k: z[k] for k in z.files}
    except Exception as e:                           # noqa: BLE001 - an unreadable archive is corrupted data
        raise CorruptedRecord("task %s: %s cannot be read (%s)" % (task_id, path, e)) from None
    shape = (2, len(rec["protocols"]), len(rec["budgets"]), int(rec["R"]))
    if arr.get("est") is None or tuple(arr["est"].shape) != shape:
        raise CorruptedRecord("task %s: estimates have shape %s, the record says %s"
                              % (task_id, None if arr.get("est") is None else arr["est"].shape, shape))
    if "design" not in arr:
        arr["design"] = legacy_design_index(rec)
    return arr


def legacy_design_index(rec: dict) -> np.ndarray:
    """Design index of each trial of a 1.6.0 record: learning protocols spread trials over their K fixed designs
    round-robin (trial r → design r mod K, the same designs in both halves); other protocols have none."""
    P, nB, R = len(rec["protocols"]), len(rec["budgets"]), int(rec["R"])
    K = int(rec.get("learning_designs", 1) or 1)
    out = np.full((2, P, nB, R), EM.NO_DESIGN, np.int16)
    for i, k in enumerate(rec["protocols"]):
        if PR.PORTFOLIO[k].learning:
            out[:, i, :, :] = (np.arange(R) % max(K, 1)).astype(np.int16)
    return out


def load_task(out_dir: str, task_id: str) -> tuple:
    """(record, arrays) of one task; arrays is None for a record without labels (error, timeout, memory)."""
    rec = load_record(out_dir, task_id)
    if rec.get("status") != "ok":
        return rec, None
    return rec, load_arrays(out_dir, task_id, rec)


def task_ids(out_dir: str) -> list:
    d = os.path.join(out_dir, "tasks")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and ".tmp" not in f) if os.path.isdir(d) \
        else []


_list_task_ids = task_ids                                # for functions with a ``task_ids`` parameter


# ------------------------------------------------------------------ risk summaries of one (task, protocol, budget)
def design_stats(loss: np.ndarray, design: np.ndarray) -> dict:
    """Design-aware summary of one half's per-trial losses. ``design`` gives each trial's design index (NO_DESIGN:
    the protocol has no training design; then the trials are independent and the design-level quantities equal the
    trial-level ones).

    * ``se_trial`` — std(L)/√n: the standard error if the trials were independent deployments;
    * ``se_design`` — cluster-robust standard error over designs, √(K/(K−1) Σ_k (S_k − n_k R̂)²)/n with S_k the
      sum of design k's losses (= std of the design means / √K for balanced designs): the standard error of the
      fresh-design risk, containing Var_D(E[L|D])/K;
    * ``var_between`` / ``var_within`` — method-of-moments (one-way ANOVA) estimates of Var_D(E[L|D]) and
      E_D[Var(L|D)];
    * ``designs`` — K, the number of designs with trials."""
    loss = np.asarray(loss, float)
    design = np.asarray(design)
    n = len(loss)
    R = float(loss.mean()) if n else math.nan
    se_t = float(loss.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    if n == 0 or np.all(design == EM.NO_DESIGN):
        return dict(se_trial=se_t, se_design=se_t, var_between=math.nan, var_within=float(loss.var(ddof=1))
                    if n > 1 else math.nan, designs=0)
    ks = np.unique(design)
    K = len(ks)
    nk = np.array([np.sum(design == k) for k in ks], float)
    Sk = np.array([loss[design == k].sum() for k in ks])
    mk = Sk / nk
    se_d = float(math.sqrt(K / (K - 1) * np.sum((Sk - nk * R) ** 2)) / n) if K > 1 else math.nan
    ssw = float(sum(((loss[design == k] - m) ** 2).sum() for k, m in zip(ks, mk)))
    vw = ssw / (n - K) if n > K else math.nan
    msb = float(np.sum(nk * (mk - R) ** 2) / (K - 1)) if K > 1 else math.nan
    m0 = (n - float(np.sum(nk ** 2)) / n) / (K - 1) if K > 1 else math.nan
    vb = max(0.0, (msb - vw) / m0) if K > 1 and np.isfinite(vw) and m0 > 0 else math.nan
    return dict(se_trial=se_t, se_design=se_d, var_between=vb, var_within=vw, designs=int(K))


def deployment_variance(err: np.ndarray, design: np.ndarray) -> dict:
    """Variance of *one* fresh deployment's estimate from trials grouped by design (1.8.0; review finding F5).

    The pooled sample variance s² = var(ddof=1) over all K·m trials is not that quantity: under the random-effects
    model X_ki = μ + A_k + E_ki (Var A = σ_D², Var E = σ_W²) its expectation is σ_W² + m(K−1)/(Km−1)·σ_D² (balanced
    groups), not σ_D² + σ_W². This returns

    * ``var_deploy`` = σ̂_W² + σ̂_D²: the one-way ANOVA (method-of-moments) estimates, unbiased for unbalanced groups
      too (m0 = (n − Σn_k²/n)/(K−1)), with σ̂_D² truncated at 0 (the usual nonnegative convention; the truncation
      adds a small positive bias when σ_D² is near 0, reported by ``truncated``);
    * ``var_pooled``: the descriptive pooled variance (kept, labeled as such);
    * ``between_coef``: m0(K−1)/(n−1), the coefficient of σ_D² in E[s²] (m(K−1)/(Km−1) when balanced).

    Without designs (NO_DESIGN: independent trials) var_deploy = var_pooled. More designs in a benchmark estimate
    these better; they do not reduce one deployment's variance unless the deployed protocol itself averages over
    designs and pays for them."""
    e = np.asarray(err, float)
    d = np.asarray(design)
    n = len(e)
    if n < 2:
        return dict(var_deploy=math.nan, var_pooled=math.nan, var_within=math.nan, var_between=math.nan,
                    between_coef=math.nan, designs=0, truncated=False)
    pooled = float(e.var(ddof=1))
    if np.all(d == EM.NO_DESIGN):
        return dict(var_deploy=pooled, var_pooled=pooled, var_within=pooled, var_between=0.0, between_coef=0.0,
                    designs=0, truncated=False)
    ks = np.unique(d)
    K = len(ks)
    nk = np.array([np.sum(d == k) for k in ks], float)
    mk = np.array([e[d == k].mean() for k in ks])
    grand = float(e.mean())
    if K < 2 or n <= K:
        return dict(var_deploy=math.nan, var_pooled=pooled, var_within=math.nan, var_between=math.nan,
                    between_coef=math.nan, designs=int(K), truncated=False)
    msw = float(sum(((e[d == k] - m) ** 2).sum() for k, m in zip(ks, mk))) / (n - K)
    msb = float(np.sum(nk * (mk - grand) ** 2)) / (K - 1)
    m0 = (n - float(np.sum(nk ** 2)) / n) / (K - 1)
    raw = (msb - msw) / m0
    vb = max(0.0, raw)
    return dict(var_deploy=msw + vb, var_pooled=pooled, var_within=msw, var_between=vb,
                between_coef=float(m0 * (K - 1) / (n - 1)), designs=int(K), truncated=bool(raw < 0))


def risk_se(se: dict, plan: dict, learning: bool) -> float:
    """The standard error that matches the record's estimand: design-level for learning protocols under fresh
    designs (the fresh-design risk), trial-level otherwise (independent trials; for 1.6.0's fixed designs the risk
    conditional on them)."""
    if learning and (plan or {}).get("mode") == "fresh":
        return se["se_design"]
    return se["se_trial"]


def plan_of(rec: dict) -> dict:
    """The design plan of a record (1.6.0 records: fixed designs shared by both halves)."""
    return rec.get("design_plan") or dict(mode="fixed-shared", count=int(rec.get("learning_designs", 1) or 1),
                                          per_half=0)


def summarize(out_dir: str, failures: list | None = None, task_ids=None):
    """One row per (task, protocol, budget): status, feasibility, ledger, costs, and the risk labels of both halves —
    MSE, bias, variance, the standard error of the MSE matching the estimand (``se_mse``) with its trial-level and
    design-level versions, the between/within-design variance components, the clipped-estimate MSE (estimate
    projected onto [−1, 1], for the bounded-loss certificates), every flag's rate, and the union failure rate
    ``fail_any`` (share of trials with at least one flag; flags overlap, so their rates must not be summed).
    Records without labels are listed in ``failures`` (if given), never silently dropped. ``task_ids`` (1.9.0): only
    these tasks (a run's manifest), never other task files of the directory."""
    import pandas as pd
    rows = []
    for tid in (sorted(task_ids) if task_ids is not None else _list_task_ids(out_dir)):
        rec = load_record(out_dir, tid)
        if rec.get("status") != "ok":
            if failures is not None:
                failures.append(dict(task_id=tid, status=rec.get("status"), error_kind=rec.get("error_kind", ""),
                                     error=str(rec.get("error", ""))[:300]))
            continue
        arr = load_arrays(out_dir, tid, rec)
        mu = rec["ideal"]
        est, flags, design = arr["est"], arr["flags"], arr["design"]
        plan = plan_of(rec)
        for i, k in enumerate(rec["protocols"]):
            learning = PR.PORTFOLIO[k].learning
            for j, B in enumerate(rec["budgets"]):
                a = rec["allocations"][i * len(rec["budgets"]) + j]
                c = rec["costs"][i * len(rec["budgets"]) + j]
                status = allocation_status(a)
                row = dict(task_id=tid, base_id=rec["base_id"], family=rec["spec"]["family"], n=rec["spec"]["n"],
                           device=rec["spec"]["device"], truth=rec["spec"]["truth"], protocol=k, B=int(B),
                           feasible=bool(a["feasible"]), status=status, reason=a.get("reason", ""), ideal=mu,
                           design_mode=plan.get("mode", "fixed-shared"), r=int(rec["R"]),
                           **{"shots_" + r: v for r, v in (a.get("ledger") or {}).items()},
                           t_exec=c.get("t_exec"), circuits_sent=c.get("circuits_sent"))
                for h, tag in ((0, "sel"), (1, "eval")):
                    x = est[h, i, j]
                    if not a["feasible"]:
                        row.update({"%s_%s" % (m, tag): math.nan for m in ("mse", "bias", "var", "var_deploy", "se_mse",
                                                                          "se_mse_trial", "se_mse_design",
                                                                          "var_between_design", "var_within_design",
                                                                          "mse_clip", "fail_any", "flags_per_trial")})
                        row["designs_" + tag] = 0
                        continue
                    e = np.where(np.isfinite(x), x, 0.0) - mu
                    ec = np.clip(np.where(np.isfinite(x), x, 0.0), -1, 1) - mu
                    l = e ** 2
                    ds_ = design_stats(l, design[h, i, j])
                    dv = deployment_variance(e, design[h, i, j])
                    row.update({"mse_" + tag: float(l.mean()), "bias_" + tag: float(e.mean()),
                                "var_" + tag: float(e.var(ddof=1)),       # pooled, descriptive (see var_deploy)
                                "var_deploy_" + tag: dv["var_deploy"],     # 1.8.0: one fresh deployment (F5)
                                "se_mse_" + tag: float(risk_se(ds_, plan, learning)),
                                "se_mse_trial_" + tag: ds_["se_trial"], "se_mse_design_" + tag: ds_["se_design"],
                                "var_between_design_" + tag: ds_["var_between"],
                                "var_within_design_" + tag: ds_["var_within"], "designs_" + tag: ds_["designs"],
                                "mse_clip_" + tag: float((ec ** 2).mean())})
                    fl = flags[h, i, j]                              # (flags, R)
                    for f, name in enumerate(rec["flags"]):
                        row["%s_%s" % (name, tag)] = float(fl[f].mean())
                    row["fail_any_" + tag] = float(fl.any(axis=0).mean())
                    row["flags_per_trial_" + tag] = float(fl.sum(axis=0).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def allocation_status(a: dict) -> str:
    """ok | infeasible | method-error of one allocation (1.6.0 records: from the reason text)."""
    if a.get("status"):
        return a["status"]
    if str(a.get("reason", "")).startswith("method failed"):
        return "method-error"
    return "ok" if a.get("feasible") else "infeasible"
