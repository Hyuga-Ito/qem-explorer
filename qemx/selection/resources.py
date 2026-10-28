"""Resource preflight, admission and monitored ceilings of a labeling run (1.9.0; second review N2).

Before 1.9.0 the whole-run limits ``limits.cpu_hours`` and ``limits.disk_gb`` were *reported* by ``plan`` only:
labeling ignored them (a configuration with a disk cap of 1e-12 GB labeled successfully). Per-task limits
(``task_seconds``, ``memory_gb``) were and are enforced by the supervisor (``supervise``). This module separates four
different things and never presents one as another:

1. **Estimate** (``preflight``): labeling CPU-hours and wall time from *measured* per-task rates (a finished part of
   this run, or ``--measured DIR`` of another run with the same candidates and design plan; widths outside the
   measured ones extrapolated log-linearly and listed). Without measurements the CPU estimate is ``unknown`` — never
   assumed small. Disk: the labels' predicted size from measured file sizes (else from the array shapes). Memory: the
   per-task limit × workers (a limit, not a usage), the measured peak resident memory per task when attempts were
   recorded, and the computer's RAM. Analysis, model search, figures and temporary files are *not* in these numbers
   (stated in ``not_included``; measure them on a pilot).
2. **Admission** (``admission``): a configured cap that the *known* estimate exceeds refuses the run before anything
   starts (1.9.1: for ``cpu_hours`` the estimate is the seconds already consumed in the labels directory plus the
   estimated remaining work, so a retry after long failed attempts is judged by what it will really have used), unless a deliberate override (``--override-limits "reason"``) is given; the override is recorded in the run's
   session. An unknown estimate is admitted *as unknown* and the monitored ceiling below applies.
3. **Monitored ceilings** (``Ceilings``): while labeling, no new task starts once (a) the labels directory's
   *consumed* task seconds (1.9.1, independent review F3: the persistent ledger ``integrity.consumed_seconds`` — every
   attempt of every invocation, including failed attempts replaced by a retry, quarantined records and interrupted
   tasks — plus this session's running tasks; with one numerical thread per worker this approximates CPU seconds;
   before 1.9.1 only the current records' seconds counted, so a retry reset part of the consumption) reach
   ``limits.cpu_hours``, (b) the labels directory reaches ``limits.disk_gb``, or (c) the
   free disk space falls below ``limits.min_free_disk_gb`` (default 1 GB). Running tasks finish and are saved, so the
   stop is checkpoint-safe; the run is then *incomplete* and the command exits non-zero.
4. **Per-task limits**: ``task_seconds`` and ``memory_gb`` per child process (``supervise``); they do not bound the
   total memory of several workers plus the parent process.

Nothing here claims an exact CPU ceiling: a task already running when a ceiling is reached completes.
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np

from .. import textio as TX

SCHEMA = "qemx.selection.resources/1"
DEFAULT_MIN_FREE_GB = 1.0
PARENT_RSS_GB = 0.6               # the supervising parent process (Python, NumPy, Qiskit imported): measured ~0.4–0.6


class AdmissionRefused(RuntimeError):
    """A run whose known resource estimate exceeds a configured cap (and no override was given)."""


def system(path: str | None = None) -> dict:
    """CPUs, RAM and free disk of this computer (``path``: the output location; its nearest existing parent)."""
    cpus = os.cpu_count() or 1
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2 ** 30
        avail = psutil.virtual_memory().available / 2 ** 30
    except Exception:                                        # noqa: BLE001
        ram, avail = math.nan, math.nan
    p = os.path.abspath(path or ".")
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    du = shutil.disk_usage(p)
    return dict(cpus=int(cpus), ram_gb=float(ram), ram_available_gb=float(avail), free_disk_gb=du.free / 2 ** 30,
                disk_checked_at=p)


def rate_function(measured: list):
    """Seconds model from measured task rates (``run.measured_rates``): per width, median preparation seconds and
    median seconds per trial; widths outside the measured ones extrapolated log-linearly in n from the per-width
    medians (never below the widest measured). Returns rate(n) -> (prepare_s, per_trial_s)."""
    ns = sorted({m["n"] for m in measured})
    med = {n: (float(np.median([m["prepare"] for m in measured if m["n"] == n])),
               float(np.median([m["per_trial"] for m in measured if m["n"] == n]))) for n in ns}

    def rate(n):
        if n in med:
            return med[n]
        if len(ns) < 2:
            return med[ns[-1]]
        vals = []
        for k in (0, 1):
            y = np.log([max(med[m][k], 1e-9) for m in ns])
            b, a = np.polyfit(ns, y, 1)
            vals.append(max(float(np.exp(a + b * n)), med[ns[-1]][k]) if n > ns[-1] else med[ns[0]][k])
        return tuple(vals)
    return rate


def _dir_bytes(path: str) -> int:
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def measured_peak_rss_gb(target: str) -> float | None:
    """Largest resident memory of one labeling task recorded in ``target``/attempts.jsonl (None if unrecorded)."""
    from . import integrity as IN
    vals = [float(a.get("peak_rss_bytes") or 0) for a in IN.attempts(target)]
    vals = [v for v in vals if v > 0]
    return max(vals) / 2 ** 30 if vals else None


def preflight(cfg: dict, stage_dir: str | None = None, *, workers: int = 1, measured_dir: str | None = None,
              final: bool = False, subdir: str | None = None, specs=None) -> dict:
    """Resource preflight of labeling ``cfg`` into ``stage_dir`` (see the module text). ``measured_dir``: a labels
    directory of another run whose rates to use; otherwise the finished tasks of ``stage_dir`` itself. ``specs``: the
    tasks to label when they are not the configuration's expansion (the mismatch study's scenario tasks)."""
    from . import design as DS
    from . import labels as LB
    from . import protocols as PR
    from . import run as RN
    from .emulate import DesignPlan
    if specs is not None:
        stage = list(specs)
    else:
        specs = DS.expand(cfg)
        dev, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
        stage = fin if final else dev                # as run.label: a separate study (subdir) labels its dev split
    done_ids = set()
    if stage_dir and os.path.isdir(os.path.join(stage_dir, "tasks")):
        for tid in LB.task_ids(stage_dir):
            try:
                if LB.load_record(stage_dir, tid).get("status") == "ok":
                    done_ids.add(tid)
            except Exception:                                # noqa: BLE001 - counted as not done
                pass
    remaining = [s for s in stage if s.task_id not in done_ids]
    src, measured = None, []
    if measured_dir:
        measured, src = RN.measured_rates(measured_dir), os.path.abspath(measured_dir)
    elif stage_dir and done_ids:
        measured, src = RN.measured_rates(stage_dir), "this run's finished tasks"
    budgets = [int(b) for b in cfg.get("budgets", PR.BUDGETS)]
    protos = list(cfg.get("protocols", list(PR.PORTFOLIO_16)))
    R = int(cfg.get("R", 200))
    trials_per_task = 2 * R * len(budgets) * len(protos)
    lim = cfg.get("limits", {}) or {}
    plan = DesignPlan.from_config(cfg.get("learning_designs", 1))
    out = dict(schema=SCHEMA, tasks=len(stage), tasks_done=len(stage) - len(remaining), tasks_remaining=len(remaining),
               trials_per_task=trials_per_task, measured_source=src, measured_tasks=len(measured),
               not_included=["analysis and model search (cross-validation, nested selection, bootstrap)",
                             "figures and tables", "replay and profiling", "temporary files during writes"],
               design_plan=plan.to_json())
    if measured:
        rate = rate_function(measured)
        # preparation builds 2·per_half + 1 designs per learning candidate: when the measured run drew fewer designs,
        # its preparation time is scaled up by that ratio (an upper bound: preparation also has design-free parts)
        def _ph(d):
            try:
                return int((d or {}).get("per_half", 0)) if isinstance(d, dict) else 0
            except (TypeError, ValueError):
                return 0
        ph_m = max([_ph(m["designs"]) for m in measured] + [0])
        ph_t = plan.per_half if plan.mode == "fresh" else 0
        prep_scale = (2 * ph_t + 1) / (2 * ph_m + 1) if ph_m and ph_t > ph_m else 1.0
        per = {s.task_id: rate(s.n)[0] * prep_scale + rate(s.n)[1] * trials_per_task for s in stage}
        cpu_total = sum(per.values()) / 3600.0
        cpu_rem = sum(per[s.task_id] for s in remaining) / 3600.0
        mw = sorted({m["n"] for m in measured})
        out["cpu"] = dict(status="estimated", total_hours=cpu_total, remaining_hours=cpu_rem,
                          measured_widths=mw, extrapolated_widths=sorted({s.n for s in stage} - set(mw)),
                          designs_measured=sorted({str(m["designs"]) for m in measured}),
                          preparation_scale=prep_scale,
                          note="per-task preparation plus per-trial time, medians per width; a rough estimate: "
                               "design-plan, candidate or budget differences from the measured run change it")
        sizes = [m["npz_bytes_per_trial"] for m in measured if m.get("npz_bytes_per_trial") is not None]
        if sizes:
            jb = float(np.median([m["json_bytes"] for m in measured]))
            per_task_bytes = jb + float(np.median(sizes)) * trials_per_task
            disk_src = "measured file sizes"
        else:                                        # records without their arrays (e.g. a shipped report whose
            per_task_bytes = None                    # trial data are not extracted): fall back to the array shapes
    else:
        out["cpu"] = dict(status="unknown", total_hours=None, remaining_hours=None,
                          note="no measured rates: label a few tasks first (label --limit N, then plan --out the same "
                               "directory) or pass --measured DIR of a comparable run")
        per_task_bytes = None
    if per_task_bytes is None:
        P, nB = len(protos), len(budgets)
        # 1.9.0 (independent verification): one third of the raw arrays under-predicted the shipped 1.8.0 studies by
        # ~2.4×; the factor below over-predicts dev18 and conf18 slightly (checked in tests)
        per_task_bytes = 0.75 * (2 * P * nB * R * (8 + 5 + 2)) + 6000 + 350 * P * nB
        disk_src = "array shapes (calibrated on the 1.8.0 studies; measured sizes replace it once arrays exist)"
    if stage_dir and os.path.isdir(stage_dir):           # 1.9.1 (F3): what this directory has already consumed
        from . import integrity as IN
        out["cpu"]["consumed_hours"] = IN.consumed_seconds(stage_dir)["total_seconds"] / 3600.0
        if out["cpu"]["status"] == "estimated":
            out["cpu"]["projected_hours"] = out["cpu"]["consumed_hours"] + out["cpu"]["remaining_hours"]
    sysinfo = system(stage_dir or ".")
    cur = _dir_bytes(stage_dir) if stage_dir and os.path.isdir(stage_dir) else 0
    out["disk"] = dict(source=disk_src, labels_total_gb=per_task_bytes * len(stage) / 2 ** 30,
                       labels_remaining_gb=per_task_bytes * len(remaining) / 2 ** 30, current_gb=cur / 2 ** 30,
                       free_gb=sysinfo["free_disk_gb"], min_free_gb=float(lim.get("min_free_disk_gb",
                                                                                   DEFAULT_MIN_FREE_GB)))
    w = max(1, int(workers))
    eff = min(w, sysinfo["cpus"])
    if out["cpu"]["status"] == "estimated":
        out["wall"] = dict(status="estimated", remaining_hours=out["cpu"]["remaining_hours"] / eff,
                           parallel=eff, note="remaining CPU-hours / min(workers, CPUs): assumes perfect parallel "
                                              "efficiency and one numerical thread per worker")
    else:
        out["wall"] = dict(status="unknown", remaining_hours=None, parallel=eff)
    peak = measured_peak_rss_gb(stage_dir) if stage_dir else None
    mem_lim = float(lim.get("memory_gb", 0) or 0)
    out["memory"] = dict(workers=w, per_task_limit_gb=mem_lim or None,
                         workers_times_limit_gb=(w * mem_lim + PARENT_RSS_GB) if mem_lim else None,
                         measured_peak_per_task_gb=peak,
                         expected_peak_gb=(w * peak + PARENT_RSS_GB) if peak else None,
                         ram_gb=sysinfo["ram_gb"], ram_available_gb=sysinfo["ram_available_gb"],
                         note="memory_gb is a per-task limit enforced by the supervisor, not a usage; the total of "
                              "several workers plus the parent is not limited by it")
    out["system"] = sysinfo
    out["recommended_workers"] = recommended_workers(cfg, sysinfo, peak)
    out["caps"] = caps(out, lim)
    return out


class _SplitRows:
    """The analysis rows of a configuration (one per development task and budget) without labels: enough for
    ``evaluate.splits`` to enumerate the split parts (``analysis_workload``)."""

    def __init__(self, rows):
        self.rows = rows
        self.groups = np.array([r["base_id"] for r in rows])

    def col(self, key):
        return np.array([r[key] for r in self.rows])


def analysis_workload(cfg: dict) -> dict:
    """Structural size of a configuration's declared analysis (1.9.1; independent review F4): counts, **not a time**.

    For every declared split: the outer parts (from the expanded tasks, no labels needed), the model policies of each
    part, and the model fits including the inner tuning — per candidate protocol, a tuned family fits its grid on the
    inner group folds (4) and once more on the whole training part. ``row_fits`` weights each fit by its training
    rows. Fit costs differ by family (an MLP or the structured model costs far more per fit than a ridge
    regression), so neither count converts into hours; extras (ablation, nested selection, learning curve, ...) are
    listed, not counted. Measure the time on the labeled study (``analyze --splits same-distribution --extras ""``)
    before committing a computer to the whole analysis; labeling caps do not cap analysis time."""
    from . import design as DS
    from . import evaluate as EV
    from . import features as FE
    from . import models as MD
    from . import protocols as PR
    a = cfg.get("analysis", {}) or {}
    dev, _ = DS.final_test(DS.expand(cfg), float(cfg.get("final_test_fraction", 0.0)))
    budgets = [int(b) for b in cfg.get("budgets", PR.BUDGETS)]
    P = len(cfg.get("protocols", list(PR.PORTFOLIO_16)))
    meta = _SplitRows([dict(task_id=s.task_id, base_id=s.base_id, family=s.family, n=s.n, device=s.device, B=b)
                       for s in dev for b in budgets])
    sets = list(a.get("feature_sets", ["size_budget", "lightcone", "deployable"]))
    fams = list(a.get("model_families", ["ridge", "enet", "hgb", "mlp"]))
    inner = 4                                                # RiskModel.fit's inner group folds
    out, by_family, total_fits, total_rows = {}, {}, 0, 0
    for kind in a.get("splits", ["same-distribution"]):
        parts = [(n, tr, te) for n, tr, te in EV.splits(meta, kind, **dict(a.get("split_options", {}).get(kind, {})))
                 if len(tr) and len(te)]
        if kind == "same-distribution":
            k_sets, k_fams = sets, fams
        else:
            k_sets = list(a.get("other_split_sets", ["size_budget", "deployable"]))
            k_fams = list(a.get("other_split_families", ["ridge", "hgb"]))
        models = [f for f in k_fams for _ in k_sets]
        if a.get("abstain_set", "deployable") in k_sets:
            models.append(a.get("abstain_family", "hgb"))
        fits = row_fits = 0
        for _, tr, _ in parts:
            k_in = max(2, min(inner, len(set(meta.groups[tr].tolist()))))
            for f in models:
                g = len(MD._grid(f))
                n_f = P * (g * k_in + 1)
                fits += n_f
                row_fits += P * (g * (k_in - 1) + 1) * len(tr)
                by_family[f] = by_family.get(f, 0) + n_f
        out[kind] = dict(parts=len(parts), model_policies_per_part=len(models), fits=fits, row_fits=row_fits)
        total_fits += fits
        total_rows += row_fits
    del FE
    return dict(rows=len(meta.rows), tasks=len(dev), protocols=P, splits=out, fits=total_fits, row_fits=total_rows,
                fits_by_family=by_family, extras=list(a.get("extras", [])),
                note="structural counts of the declared main splits, not a time; per-fit cost differs by family; "
                     "extras are listed, not counted; measure on the labeled study before running everything")


def workload_text(w: dict) -> str:
    """``analysis_workload`` as a few lines for people."""
    lines = ["%d analysis rows (%d tasks × budgets), %d candidates; main splits: %s model fits, %.3g row-fits "
             "(fits × training rows)" % (w["rows"], w["tasks"], w["protocols"], format(w["fits"], ","), w["row_fits"])]
    for k, v in w["splits"].items():
        share = v["row_fits"] / w["row_fits"] if w["row_fits"] else 0.0
        lines.append("  %s: %d part(s) × %d model policies → %s fits, %.1f %% of the row-fits" % (
            k, v["parts"], v["model_policies_per_part"], format(v["fits"], ","), 100 * share))
    lines.append("  fits by family: " + ", ".join("%s %s" % (f, format(n, ",")) for f, n in
                                                  sorted(w["fits_by_family"].items())))
    if w["extras"]:
        lines.append("  extras (not counted): " + ", ".join(w["extras"]))
    sd = w["splits"].get("same-distribution")
    if sd and w["row_fits"]:
        lines.append("  NOT a time. To estimate one: time `analyze --splits same-distribution --extras \"\"` on the "
                     "labeled run; the main splits are roughly that time ÷ %.3f (row-fit share; per-fit costs differ "
                     "by family), plus the extras" % (sd["row_fits"] / w["row_fits"]))
    return "\n".join(lines)


def recommended_workers(cfg: dict, sysinfo: dict, peak_gb: float | None = None) -> int:
    """Workers for this computer: one CPU left for the system, and the per-task memory (measured peak when known,
    else the per-task limit, else 2 GB) times workers within 70 % of RAM."""
    lim = cfg.get("limits", {}) or {}
    per = peak_gb or float(lim.get("memory_gb", 0) or 0) or 2.0
    by_cpu = max(1, int(sysinfo["cpus"]) - 1)
    ram = sysinfo.get("ram_gb")
    by_mem = max(1, int((0.7 * ram - PARENT_RSS_GB) // per)) if ram and math.isfinite(ram) else 1
    return int(max(1, min(by_cpu, by_mem)))


def caps(pre: dict, lim: dict) -> dict:
    """Each configured cap against its estimate: within | exceeds | unknown (and whether it is monitored)."""
    out = {}
    if lim.get("cpu_hours") is not None:
        cap = float(lim["cpu_hours"])
        t = pre["cpu"].get("projected_hours", pre["cpu"].get("total_hours"))     # 1.9.1: consumed + remaining
        out["cpu_hours"] = dict(cap=cap, estimate=t, status="unknown" if t is None else
                                ("within" if t <= cap else "exceeds"), monitored=True,
                                enforcement="no new task starts once the consumed task seconds (every attempt, "
                                            "all invocations; 1.9.1) reach the cap")
    if lim.get("disk_gb") is not None:
        cap = float(lim["disk_gb"])
        t = pre["disk"]["labels_total_gb"]
        out["disk_gb"] = dict(cap=cap, estimate=t, status="within" if t <= cap else "exceeds", monitored=True,
                              enforcement="no new task starts once the labels directory reaches the cap")
    need = pre["disk"]["labels_remaining_gb"] + pre["disk"]["min_free_gb"]
    out["free_disk"] = dict(cap=pre["disk"]["free_gb"], estimate=need,
                            status="within" if need <= pre["disk"]["free_gb"] else "exceeds", monitored=True,
                            enforcement="no new task starts while free space is below limits.min_free_disk_gb")
    m = pre["memory"]
    if m.get("expected_peak_gb") and m.get("ram_gb") and math.isfinite(m["ram_gb"]):
        out["memory"] = dict(cap=0.9 * m["ram_gb"], estimate=m["expected_peak_gb"],
                             status="within" if m["expected_peak_gb"] <= 0.9 * m["ram_gb"] else "exceeds",
                             monitored=False, enforcement="per task only (memory_gb); measured peak × workers + parent")
    return out


def admission(pre: dict, override: str | None = None) -> dict:
    """Admit or refuse: refused when a cap's *known* estimate is exceeded, unless ``override`` (a reason) is given.
    Unknown estimates are admitted as unknown (the monitored ceilings apply). Raises ``AdmissionRefused``."""
    bad = {k: v for k, v in pre["caps"].items() if v["status"] == "exceeds"}
    unknown = sorted(k for k, v in pre["caps"].items() if v["status"] == "unknown")
    dec = dict(admitted=not bad or bool(override), exceeded=sorted(bad), unknown=unknown,
               override=override or None)
    if bad and not override:
        raise AdmissionRefused("refused before labeling: the estimate exceeds %s (%s). Change the configuration, "
                               "raise the limit (limits.* are operational: a freeze stays valid), choose another "
                               "output location, or start anyway with --override-limits \"<reason>\" (recorded in "
                               "run.json) when you expect the estimate to be pessimistic — the monitored ceiling still "
                               "stops labeling at the cap itself." % (", ".join(sorted(bad)), "; ".join(
                                   "%s: estimate %.3g, cap %.3g" % (k, v["estimate"], v["cap"]) for k, v in bad.items())))
    return dec


class Ceilings:
    """Monitored whole-run ceilings during labeling (see the module text). ``admit(running_seconds)`` is asked before
    each task starts; ``finished(task_id, seconds)`` after each finishes."""

    def __init__(self, cfg: dict, stage_dir: str, recorded_task_seconds: float = 0.0):
        """``recorded_task_seconds``: the seconds already consumed in ``stage_dir`` (1.9.1: callers pass
        ``integrity.consumed_seconds(stage_dir)["total_seconds"]``)."""
        lim = cfg.get("limits", {}) or {}
        self.cpu_cap_s = float(lim["cpu_hours"]) * 3600.0 if lim.get("cpu_hours") is not None else None
        self.disk_cap = float(lim["disk_gb"]) * 2 ** 30 if lim.get("disk_gb") is not None else None
        self.min_free = float(lim.get("min_free_disk_gb", DEFAULT_MIN_FREE_GB)) * 2 ** 30
        self.dir = stage_dir
        self.task_seconds = float(recorded_task_seconds)
        self.bytes = _dir_bytes(stage_dir) if os.path.isdir(stage_dir) else 0
        self.stopped = None

    def finished(self, task_id: str, seconds: float) -> None:
        self.task_seconds += float(seconds or 0.0)
        for ext in (".json", ".npz"):
            p = os.path.join(self.dir, "tasks", task_id + ext)
            if os.path.exists(p):
                self.bytes += os.path.getsize(p)

    def admit(self, running_seconds: float = 0.0) -> bool:
        if self.stopped:
            return False
        if self.cpu_cap_s is not None and self.task_seconds + running_seconds >= self.cpu_cap_s:
            self.stopped = ("cpu_hours: the consumed task seconds %.0f s (every attempt, all invocations) reached "
                            "the cap %.0f s"
                            % (self.task_seconds + running_seconds, self.cpu_cap_s))
        elif self.disk_cap is not None and self.bytes >= self.disk_cap:
            self.stopped = "disk_gb: the labels directory holds %.3g GB, the cap is %.3g GB" % (
                self.bytes / 2 ** 30, self.disk_cap / 2 ** 30)
        else:
            free = shutil.disk_usage(self.dir).free
            if free < self.min_free:
                self.stopped = "min_free_disk_gb: %.2f GB free, below %.2f GB" % (free / 2 ** 30,
                                                                                 self.min_free / 2 ** 30)
        return self.stopped is None


def text(pre: dict) -> str:
    """The preflight as a few lines for people."""
    c, d, m, w = pre["cpu"], pre["disk"], pre["memory"], pre["wall"]
    lines = ["tasks: %d (%d done, %d remaining); %d trials per task" % (pre["tasks"], pre["tasks_done"],
                                                                         pre["tasks_remaining"], pre["trials_per_task"])]
    if c["status"] == "estimated":
        lines.append("labeling CPU: %.2f h total, %.2f h remaining (measured: %s; extrapolated widths: %s)" % (
            c["total_hours"], c["remaining_hours"], pre["measured_source"], c["extrapolated_widths"] or "none"))
        if c.get("projected_hours") is not None:
            lines.append("labeling CPU already consumed here: %.2f h (every attempt, including failed, retried and "
                         "interrupted ones) → projected %.2f h" % (c["consumed_hours"], c["projected_hours"]))
        lines.append("labeling wall time: ~%.2f h with %d parallel workers (rough)" % (w["remaining_hours"],
                                                                                     w["parallel"]))
    else:
        lines.append("labeling CPU and wall time: UNKNOWN — " + c["note"])
    lines.append("disk: labels %.3g GB in total, %.3g GB remaining (%s); %.1f GB free, keep %.1f GB free" % (
        d["labels_total_gb"], d["labels_remaining_gb"], d["source"], d["free_gb"], d["min_free_gb"]))
    lines.append("memory: %d worker(s) × limit %s GB per task%s; RAM %.1f GB; recommended workers here: %d" % (
        m["workers"], m["per_task_limit_gb"], ("; measured peak %.2f GB per task → ~%.1f GB expected" % (
            m["measured_peak_per_task_gb"], m["expected_peak_gb"])) if m["measured_peak_per_task_gb"] else "",
        m["ram_gb"], pre["recommended_workers"]))
    for k, v in pre["caps"].items():
        lines.append("cap %s: %s (estimate %s, cap %s; %s)" % (
            k, v["status"].upper(), "unknown" if v["estimate"] is None else "%.3g" % v["estimate"], "%.3g" % v["cap"],
            v["enforcement"]))
    lines.append("not included: " + "; ".join(pre["not_included"]))
    return "\n".join(lines)


def write(pre: dict, path: str) -> None:
    TX.write_json(path, pre, indent=1, default=str, atomic=True)
