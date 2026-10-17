r"""QPU campaign planning and a conservative usage ledger (1.5).

Policy. One shared QPU-usage ceiling for the whole thesis campaign — **600 s by default**, covering the main study
and the discrepancy study together — minus a reserve, and never more than the account's remaining allowance when
that is known. The ceiling is an execution constraint, not a target.

Why an app-side ledger. IBM's ``max_execution_time`` limits one job's QPU usage only. A batch's or session's
``max_time`` is an elapsed-time timer that fails queued jobs; it is not a usage budget. No provider option bounds
the total usage of a campaign, apart from the plan's allowance (Open Plan: 10 min per 28-day rolling window) and,
on paid plans, instance cost limits. A job cancelled by the user or by its own limit still reports the usage
consumed up to that point. (IBM Quantum documentation, "Workload usage" and "Maximum execution time", read
2026-10-17; see docs/literature_matrix.md.)

Usage model (``UsageModel``):

* low     — IBM's quick formula, 2 s + 0.00035 s × executions per sub-job set (default rep_delay, ~100 µs circuits);
* central — 2 s per sub-job + per-shot cost × executions, with the per-shot cost measured on the target device
            (0.57 ms per shot on ibm_fez in this project's earlier runs: replace it with a pilot's ``job.usage()``);
* high    — central × margin (1.3 by default).

"Executions" are circuits × shots. The number of sub-jobs IBM creates is not documented. The model assumes
one sub-job per ``circuits_per_subjob`` circuits, an assumption to calibrate. Admission always uses **high**.

Ledger (``Ledger``, a JSON file written atomically after every change). Each job passes through these states:

1. **reserved** — admitted, with its high estimate.
2. **submitted** — has a job id.
3. **completed / failed / cancelled** — reconciled with the usage the provider reports, or the high estimate while
   that is unknown.
4. **released** — never submitted.

    committed = Σ reconciled usage + Σ high estimates of everything not yet reconciled
    remaining = min(ceiling, account allowance) − reserve − committed

A job is admitted only if its high estimate fits within ``remaining``, and within ``max_outstanding_s`` of
unreconciled work when that bound is set. Rules for retries and resume:

* A reservation that crashed between submission and recording its job id stays committed until it is reconciled
  by hand.
* Nothing is retried automatically.
* A job id is never recorded twice.

The guarantee is conditional. The ceiling holds if each job's real usage stays within its high estimate. If a job
exceeds it, reconciliation shows the overrun and further admission stops. A per-job ``max_execution_time`` (set
from the high estimate) bounds such an overrun, but the provider may still bill a little over it.

    python -m qemx.campaign init      --ledger campaign_ledger.json --ceiling 600 --reserve 60
    python -m qemx.campaign estimate  --shots 720000 --circuits 1680 --jobs 12
    python -m qemx.campaign preflight data/dirA_div_qpu10 --ledger campaign_ledger.json --out preflight.md
    python -m qemx.campaign status    --ledger campaign_ledger.json
    python -m qemx.campaign reconcile --ledger campaign_ledger.json --job-id JOB --usage 41.2 --state completed
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import os
import sys
import uuid

DEFAULT_CEILING_S = 600.0
DEFAULT_RESERVE_S = 60.0
OPEN_PLAN_BATCH_TTL_S = 600.0          # Open plan: default batch max TTL (elapsed time, starts with the first job)
IBM_QUICK_PER_EXEC_S = 0.00035          # IBM "Workload usage": quick formula 2 + 0.00035 × executions
IBM_SUBJOB_OVERHEAD_S = 2.0             # "approximately 2s per sub-job"
MEASURED_PER_SHOT_S = 0.00057           # this project's ibm_fez measurement (replace with a pilot's job.usage())
FINAL_STATES = ("completed", "failed", "cancelled")
STATES = ("reserved", "submitted") + FINAL_STATES + ("released",)


class CampaignLimit(RuntimeError):
    """Admitting this job could exceed the campaign's conservative remaining allowance."""


@dataclasses.dataclass(frozen=True)
class UsageModel:
    per_shot_s: float = MEASURED_PER_SHOT_S
    margin: float = 1.3
    subjob_overhead_s: float = IBM_SUBJOB_OVERHEAD_S
    circuits_per_subjob: int = 100          # assumption: IBM's sub-job split is not documented; calibrate

    def subjobs(self, circuits: int) -> int:
        return max(1, math.ceil(int(circuits) / max(1, int(self.circuits_per_subjob))))


@dataclasses.dataclass(frozen=True)
class UsageEstimate:
    executions: int
    circuits: int
    jobs: int
    low_s: float
    central_s: float
    high_s: float
    assumptions: tuple

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def estimate_usage(shots: int, circuits: int, jobs: int = 1, model: UsageModel | None = None) -> UsageEstimate:
    """QPU usage of ``jobs`` jobs that together execute ``shots`` circuit executions (Σ shots over circuits)."""
    m = model or UsageModel()
    shots, circuits, jobs = int(shots), int(circuits), max(1, int(jobs))
    if shots < 0 or circuits < 0:
        raise ValueError("shots and circuits must be ≥ 0")
    per_job_circ = math.ceil(circuits / jobs) if circuits else 0
    sub = jobs * m.subjobs(per_job_circ)
    low = sub * IBM_SUBJOB_OVERHEAD_S + IBM_QUICK_PER_EXEC_S * shots
    central = sub * m.subjob_overhead_s + m.per_shot_s * shots
    high = central * m.margin
    return UsageEstimate(shots, circuits, jobs, low, central, max(high, low),
                         ("per-shot %.2f ms (%s)" % (1e3 * m.per_shot_s, "measured on ibm_fez in earlier runs"
                                                     if m.per_shot_s == MEASURED_PER_SHOT_S else "user-set"),
                          "%.0f s per sub-job, %d sub-job(s) (assumed: one per %d circuits)" % (
                              m.subjob_overhead_s, sub, m.circuits_per_subjob),
                          "high = central × %.2f; low = IBM quick formula" % m.margin))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ledger:
    """The campaign's usage ledger (see the module text). Every mutation is persisted atomically."""

    def __init__(self, path: str, data: dict):
        self.path = path
        self.d = data

    # ------------------------------------------------------------------ files
    @classmethod
    def create(cls, path: str, ceiling_s: float = DEFAULT_CEILING_S, reserve_s: float = DEFAULT_RESERVE_S,
               account_remaining_s: float | None = None, max_outstanding_s: float | None = None,
               note: str = "") -> "Ledger":
        if os.path.exists(path):
            raise FileExistsError("%s exists; a campaign ledger is never overwritten" % path)
        for v, nm in ((ceiling_s, "ceiling"), (reserve_s, "reserve")):
            if not (math.isfinite(float(v)) and float(v) >= 0):
                raise ValueError("%s must be finite and ≥ 0" % nm)
        if float(reserve_s) >= float(ceiling_s):
            raise ValueError("the reserve must be smaller than the ceiling")
        led = cls(path, dict(schema="qemx.campaign.ledger/1", created=_now(), ceiling_s=float(ceiling_s),
                             reserve_s=float(reserve_s), account_remaining_s=account_remaining_s,
                             max_outstanding_s=max_outstanding_s, note=str(note)[:500], entries=[]))
        led.save()
        return led

    @classmethod
    def load(cls, path: str) -> "Ledger":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("schema") != "qemx.campaign.ledger/1":
            raise ValueError("%s is not a campaign ledger" % path)
        return cls(path, d)

    def save(self) -> None:
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.d, f, indent=1)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ accounting
    @property
    def entries(self) -> list:
        return self.d["entries"]

    def committed_s(self) -> float:
        tot = 0.0
        for e in self.entries:
            if e["state"] == "released":
                continue
            if e["state"] in FINAL_STATES and e.get("usage_s") is not None:
                tot += float(e["usage_s"])
            else:                                     # outstanding, or ended without a reported usage
                tot += float(e["est_high_s"])
        return tot

    def outstanding_s(self) -> float:
        return sum(float(e["est_high_s"]) for e in self.entries if e["state"] in ("reserved", "submitted"))

    def limit_s(self) -> float:
        a = self.d.get("account_remaining_s")
        return min(float(self.d["ceiling_s"]), float(a)) if a is not None else float(self.d["ceiling_s"])

    def remaining_s(self) -> float:
        return self.limit_s() - float(self.d["reserve_s"]) - self.committed_s()

    def status(self) -> dict:
        by = {}
        for e in self.entries:
            by[e["state"]] = by.get(e["state"], 0) + 1
        over = [e["job_id"] for e in self.entries if e.get("usage_s") is not None
                and float(e["usage_s"]) > float(e["est_high_s"])]
        return dict(unsubmitted_reservations=[e["id"] for e in self.unsubmitted_reservations()],
                    ceiling_s=self.d["ceiling_s"], account_remaining_s=self.d.get("account_remaining_s"),
                    reserve_s=self.d["reserve_s"], committed_s=self.committed_s(), outstanding_s=self.outstanding_s(),
                    remaining_s=self.remaining_s(), jobs_by_state=by, overruns=over,
                    reported_usage_s=sum(float(e["usage_s"]) for e in self.entries if e.get("usage_s") is not None))

    # ------------------------------------------------------------------ transitions
    def admit(self, label: str, est: UsageEstimate, campaign: str = "") -> str:
        """Reserve a job's high estimate; CampaignLimit if it does not fit. Returns the reservation id."""
        over = [e for e in self.entries if e.get("usage_s") is not None and float(e["usage_s"]) > float(e["est_high_s"])]
        if over:
            raise CampaignLimit("job %s used %.1f s, more than its high estimate %.1f s: the usage model is too "
                                "optimistic; recalibrate it before admitting more work" % (
                                    over[0]["job_id"], float(over[0]["usage_s"]), float(over[0]["est_high_s"])))
        rem = self.remaining_s()
        if est.high_s > rem:
            raise CampaignLimit("%s needs up to %.1f s but only %.1f s remain of the %.0f s ceiling (reserve %.0f s, "
                                "committed %.1f s)" % (label, est.high_s, rem, self.limit_s(), self.d["reserve_s"],
                                                      self.committed_s()))
        mo = self.d.get("max_outstanding_s")
        if mo is not None and self.outstanding_s() + est.high_s > float(mo):
            raise CampaignLimit("%s would raise unreconciled work to %.1f s (bound %.0f s): collect and reconcile the "
                                "jobs in flight first" % (label, self.outstanding_s() + est.high_s, float(mo)))
        rid = uuid.uuid4().hex[:12]
        self.entries.append(dict(id=rid, label=str(label)[:200], campaign=str(campaign)[:80], state="reserved",
                                 est_high_s=float(est.high_s), est_central_s=float(est.central_s),
                                 executions=int(est.executions), circuits=int(est.circuits), job_id=None,
                                 usage_s=None, created=_now(), updated=_now(), note=""))
        self.save()
        return rid

    def _entry(self, rid: str) -> dict:
        for e in self.entries:
            if e["id"] == rid:
                return e
        raise KeyError("no reservation %s" % rid)

    def submitted(self, rid: str, job_id: str) -> None:
        if any(e.get("job_id") == job_id for e in self.entries):
            raise ValueError("job %s is already recorded" % job_id)
        e = self._entry(rid)
        if e["state"] != "reserved":
            raise ValueError("reservation %s is %s, not reserved" % (rid, e["state"]))
        e.update(state="submitted", job_id=str(job_id), updated=_now())
        self.save()

    def release(self, rid: str, reason: str = "") -> None:
        """Give back a reservation that never became a job (the submission call failed before returning an id).
        If a job might exist anyway, do not release: reconcile it after checking the provider's job list."""
        e = self._entry(rid)
        if e["state"] != "reserved":
            raise ValueError("only an unsubmitted reservation can be released (%s is %s)" % (rid, e["state"]))
        e.update(state="released", note=str(reason)[:300], updated=_now())
        self.save()

    def reconcile(self, job_id: str, usage_s: float | None, state: str = "completed", note: str = "") -> None:
        """Record what the provider reports for a job (``usage_s`` None: not reported yet — the high estimate stays
        committed)."""
        if state not in FINAL_STATES + ("submitted",):
            raise ValueError("state must be one of %s" % ", ".join(FINAL_STATES))
        hit = [e for e in self.entries if e.get("job_id") == job_id]
        if not hit:
            raise KeyError("job %s is not in the ledger" % job_id)
        e = hit[0]
        if usage_s is not None and not (math.isfinite(float(usage_s)) and float(usage_s) >= 0):
            raise ValueError("usage must be finite and ≥ 0")
        e.update(state=state, usage_s=None if usage_s is None else float(usage_s), note=str(note)[:300],
                 updated=_now())
        self.save()

    def resolve(self, rid: str, job_id: str | None) -> None:
        """After an interruption: a reservation without a job id either became a job (attach its id, found in the
        provider's job list) or did not (``job_id`` None: released)."""
        if job_id:
            self.submitted(rid, job_id)
        else:
            self.release(rid, "resolved by hand: no job was created")

    def unsubmitted_reservations(self) -> list:
        """Reservations without a job id: after a crash these may or may not be jobs. They stay committed."""
        return [e for e in self.entries if e["state"] == "reserved"]


# ------------------------------------------------------------------ planning a run directory
def jobs_of_plan(plan: dict, max_circuits_per_job: int = 300) -> list:
    """The jobs ``qemx.divergence.submit`` would create for a divergence plan (grouped by shots, ≤ 300 circuits):
    [(label, shots per circuit, circuits)]."""
    by = {}
    for r in plan.get("runs", []):
        by.setdefault(int(r["shots"]), []).append(r)
    out = []
    for shots, rs in sorted(by.items()):
        for lo in range(0, len(rs), max_circuits_per_job):
            part = rs[lo:lo + max_circuits_per_job]
            out.append(("%d circuits × %d shots" % (len(part), shots), shots, len(part)))
    return out


def preflight(plan: dict, ledger: Ledger | None = None, model: UsageModel | None = None,
              ceiling_s: float = DEFAULT_CEILING_S, reserve_s: float = DEFAULT_RESERVE_S) -> dict:
    """Job-by-job admission under the conservative model, without submitting anything."""
    m = model or UsageModel()
    jobs = jobs_of_plan(plan)
    if ledger is not None:
        remaining = ledger.remaining_s()
    else:
        remaining = ceiling_s - reserve_s
    rows, cum_high, admitted = [], 0.0, 0
    for label, shots, circ in jobs:
        est = estimate_usage(shots * circ, circ, 1, m)
        fits = cum_high + est.high_s <= remaining
        if fits:
            cum_high += est.high_s
            admitted += 1
        rows.append(dict(job=label, executions=shots * circ, circuits=circ, low_s=est.low_s, central_s=est.central_s,
                         high_s=est.high_s, max_execution_time_s=int(math.ceil(est.high_s * 1.1)) + 1,
                         admitted=bool(fits)))
    tot = estimate_usage(sum(s * c for _, s, c in jobs), sum(c for _, _, c in jobs), len(jobs), m)
    return dict(jobs=rows, total=tot.as_dict(), remaining_s=remaining, admitted_jobs=admitted, total_jobs=len(jobs),
                fits=admitted == len(jobs), open_plan_batch_ttl_s=OPEN_PLAN_BATCH_TTL_S,
                batch_ttl_risk=bool(tot.high_s > 0.5 * OPEN_PLAN_BATCH_TTL_S))


def preflight_markdown(pf: dict, title: str = "Hardware preflight") -> str:
    t = pf["total"]
    L = ["# %s" % title, "",
         "**Planned QPU usage:** %.0f s (low, IBM quick formula) · %.0f s (central) · **%.0f s (high, used for "
         "admission)** for %d jobs, %s executions." % (t["low_s"], t["central_s"], t["high_s"], pf["total_jobs"],
                                                       f"{t['executions']:,}"),
         "**Conservative allowance available:** %.0f s. %s" % (
             pf["remaining_s"], "All jobs fit." if pf["fits"] else "**Only %d of %d jobs fit: the plan must be cut "
             "or split across allowance windows.**" % (pf["admitted_jobs"], pf["total_jobs"])), "",
         "Assumptions: " + "; ".join(t["assumptions"]) + ".", ""]
    if pf["batch_ttl_risk"]:
        L += ["**Batch TTL risk.** The Open plan's default batch max TTL is 10 min of elapsed time from the first "
              "job's start; queue waits count. Submit in job mode (`--mode job`, the default) or in small batches.", ""]
    L += ["| job | executions | circuits | low s | central s | high s | max_execution_time s | admitted |",
          "|---|---|---|---|---|---|---|---|"]
    for r in pf["jobs"]:
        L.append("| %s | %s | %d | %.1f | %.1f | %.1f | %d | %s |" % (
            r["job"], f"{r['executions']:,}", r["circuits"], r["low_s"], r["central_s"], r["high_s"],
            r["max_execution_time_s"], "yes" if r["admitted"] else "**no**"))
    L += ["", "Stopping rules: a job is submitted only after the ledger admits its high estimate; nothing is retried "
          "automatically; a failed or cancelled job keeps its reported usage (or its high estimate until reported); "
          "if any job's reported usage exceeds its high estimate, admission stops until the usage model is "
          "recalibrated. A per-job `max_execution_time` bounds a single job only."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ a mock provider for tests and rehearsals
class MockJob:
    def __init__(self, jid, usage, fail=False):
        self._id, self._usage, self._fail = jid, usage, fail

    def job_id(self):
        return self._id

    def usage(self):
        return self._usage

    def status(self):
        return "ERROR" if self._fail else "DONE"


class MockProvider:
    """Deterministic stand-in for a QPU service: true usage = overhead + true per-shot cost × executions. It can
    fail jobs (usage still consumed), fail at submission (no job), or 'crash' after creating a job."""

    def __init__(self, true_per_shot_s: float = MEASURED_PER_SHOT_S, overhead_s: float = 2.0, fail_every: int = 0,
                 submit_error_on: int = -1, crash_after_create_on: int = -1, prefix: str = "mock"):
        self.prefix = prefix
        self.true_per_shot_s, self.overhead_s = float(true_per_shot_s), float(overhead_s)
        self.fail_every, self.submit_error_on, self.crash_after_create_on = fail_every, submit_error_on, crash_after_create_on
        self.jobs, self.n = {}, 0

    def run(self, executions: int):
        self.n += 1
        if self.n == self.submit_error_on:
            raise RuntimeError("submission rejected")
        jid = "%s-%04d" % (self.prefix, self.n)
        job = MockJob(jid, self.overhead_s + self.true_per_shot_s * int(executions),
                      fail=bool(self.fail_every and self.n % self.fail_every == 0))
        self.jobs[jid] = job
        if self.n == self.crash_after_create_on:
            raise KeyboardInterrupt("crash after the job was created")
        return job

    def total_usage(self) -> float:
        return sum(j.usage() for j in self.jobs.values())


def submit_with_ledger(ledger: Ledger, jobs: list, run_job, model: UsageModel | None = None, campaign: str = "") -> list:
    """Admit and submit jobs one by one: [(label, shots, circuits)] → records. ``run_job(label, shots, circuits)``
    returns a job object with ``job_id()``. Stops at the first job the ledger refuses (CampaignLimit is re-raised
    with the records so far attached as ``.records``)."""
    m = model or UsageModel()
    records = []
    for label, shots, circ in jobs:
        est = estimate_usage(shots * circ, circ, 1, m)
        try:
            rid = ledger.admit(label, est, campaign)
        except CampaignLimit as e:
            e.records = records
            raise
        try:
            job = run_job(label, shots, circ)
        except Exception:
            ledger.release(rid, "the submission call failed before returning a job")
            raise
        ledger.submitted(rid, job.job_id())
        records.append(dict(job_id=job.job_id(), label=label, reservation=rid, est_high_s=est.high_s))
    return records


def reconcile_jobs(ledger: Ledger, job_lookup) -> list:
    """Reconcile every submitted job: ``job_lookup(job_id)`` returns a job with ``status()`` and ``usage()``
    (``qiskit_ibm_runtime`` ≥ 0.30 jobs have both). Returns [(job_id, state, usage)]."""
    out = []
    for e in list(ledger.entries):
        if e["state"] != "submitted":
            continue
        job = job_lookup(e["job_id"])
        st = str(job.status()).split(".")[-1].upper()
        if st not in ("DONE", "ERROR", "CANCELLED"):
            continue
        try:
            u = job.usage()
            u = None if u is None else float(u)
        except Exception:                                # noqa: BLE001 - not reported (yet)
            u = None
        state = {"DONE": "completed", "ERROR": "failed", "CANCELLED": "cancelled"}[st]
        ledger.reconcile(e["job_id"], u, state)
        out.append((e["job_id"], state, u))
    return out


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qemx.campaign", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--ledger", required=True)
    p.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_S)
    p.add_argument("--reserve", type=float, default=DEFAULT_RESERVE_S)
    p.add_argument("--account-remaining", type=float, default=None, help="the account's remaining allowance (s)")
    p.add_argument("--max-outstanding", type=float, default=None, help="bound on unreconciled work (s)")
    p.add_argument("--note", default="")
    p = sub.add_parser("status")
    p.add_argument("--ledger", required=True)
    p = sub.add_parser("estimate")
    p.add_argument("--shots", type=int, required=True, help="executions: Σ shots over circuits")
    p.add_argument("--circuits", type=int, required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--per-shot-ms", type=float, default=MEASURED_PER_SHOT_S * 1e3)
    p.add_argument("--margin", type=float, default=1.3)
    p = sub.add_parser("preflight")
    p.add_argument("run_dir", help="a divergence run directory (plan.json)")
    p.add_argument("--ledger", default=None)
    p.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_S)
    p.add_argument("--reserve", type=float, default=DEFAULT_RESERVE_S)
    p.add_argument("--per-shot-ms", type=float, default=MEASURED_PER_SHOT_S * 1e3)
    p.add_argument("--margin", type=float, default=1.3)
    p.add_argument("--out", default=None)
    p = sub.add_parser("reconcile")
    p.add_argument("--ledger", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--usage", type=float, default=None)
    p.add_argument("--state", default="completed", choices=FINAL_STATES)
    p = sub.add_parser("resolve", help="settle a reservation left without a job id by an interruption")
    p.add_argument("--ledger", required=True)
    p.add_argument("--id", required=True, help="the reservation id (status lists them)")
    p.add_argument("--job-id", default=None, help="the job it became (omit if no job was created)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "resolve":
            led = Ledger.load(a.ledger)
            led.resolve(a.id, a.job_id)
            print(json.dumps(led.status(), indent=1))
            return 0
        if a.cmd == "init":
            Ledger.create(a.ledger, a.ceiling, a.reserve, a.account_remaining, a.max_outstanding, a.note)
            print("created %s (ceiling %.0f s, reserve %.0f s)" % (a.ledger, a.ceiling, a.reserve))
        elif a.cmd == "status":
            print(json.dumps(Ledger.load(a.ledger).status(), indent=1))
        elif a.cmd == "estimate":
            m = UsageModel(per_shot_s=a.per_shot_ms / 1e3, margin=a.margin)
            print(json.dumps(estimate_usage(a.shots, a.circuits, a.jobs, m).as_dict(), indent=1))
        elif a.cmd == "preflight":
            with open(os.path.join(a.run_dir, "plan.json"), encoding="utf-8") as f:
                plan = json.load(f)
            led = Ledger.load(a.ledger) if a.ledger else None
            m = UsageModel(per_shot_s=a.per_shot_ms / 1e3, margin=a.margin)
            md = preflight_markdown(preflight(plan, led, m, a.ceiling, a.reserve),
                                    "Hardware preflight: %s" % os.path.basename(os.path.abspath(a.run_dir)))
            if a.out:
                with open(a.out, "w", encoding="utf-8") as f:
                    f.write(md)
            print(md)
        elif a.cmd == "reconcile":
            led = Ledger.load(a.ledger)
            led.reconcile(a.job_id, a.usage, a.state)
            print(json.dumps(led.status(), indent=1))
        return 0
    except (ValueError, KeyError, FileExistsError, CampaignLimit) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
