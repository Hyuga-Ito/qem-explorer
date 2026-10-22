r"""QPU campaign planning and a conservative, transactional usage ledger (1.5.1).

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

Ledger (``Ledger``, schema ``qemx.campaign.ledger/2``). A JSON file shared by every process that submits for the
campaign (the app, the CLIs). **Every read-modify-write is one transaction**: an exclusive inter-process lock on the
sidecar file ``<ledger>.lock`` (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows, plus a per-path thread lock
inside one process) is held while the ledger is re-read from disk, checked, changed and written through a uniquely
named temporary file (fsync, then ``os.replace``). Two stale ``Ledger`` objects therefore cannot overwrite each other
and admission always decides on the current file; each object's in-memory view is refreshed by every transaction.
Each job passes through these states:

1. **reserved** — admitted, with its high estimate; the provider has not been called.
2. **submitting** — persisted (with the client tag ``qemx-ledger-<id>``, sent as the job's tag) right before the
   provider call. An entry left here by a crash may be a job.
3. **submission_unknown** — the provider call failed in a way that does not prove that no job exists (a timeout, a
   lost connection, any unexpected error, an interruption). It stays committed at its high estimate and blocks new
   submissions until it is resolved: the job found by its tag is attached (→ submitted), or a provider check shows
   that none exists (→ released).
4. **submitted** — has a job id.
5. **completed / failed / cancelled** — reconciled with the usage the provider reports, or the high estimate while
   that is unknown.
6. **released** — never became a job: rejected before any provider call (local preparation failed, or the submit
   callable raised :class:`SubmissionRejected`), or resolved as "not submitted" after checking the provider.

    committed = Σ reconciled usage + Σ high estimates of everything not yet reconciled (states 1–4, and 5 without
                reported usage)
    remaining = min(ceiling, account allowance) − reserve − committed

A job is admitted only if its high estimate fits within ``remaining``, and within ``max_outstanding_s`` of
unreconciled work when that bound is set. Rules for retries and resume:

* Only an explicit "definitely not submitted" signal releases a reservation; every other failure of the provider
  call leaves it committed as ``submission_unknown`` (:func:`submit_one`, used by every submitter).
* Nothing is retried automatically, and no submitter starts while the ledger holds an unresolved submission
  (:func:`check_no_unresolved`).
* A job id is never recorded twice.

Real-hardware submission requires a ledger (:func:`require_ledger`; only fake backends and local simulators run
without one), and a configured ledger path that does not exist is an error, never "no ledger".

The guarantee is conditional. Each job is submitted with a provider-enforced per-job ``max_execution_time`` (by
default its high estimate rounded up) and the ledger commits the larger of that limit and the high estimate, so
the per-job limits of all unreconciled jobs fit inside the admitted allowance (ceiling − reserve). The ceiling
therefore holds if the provider stops each job at its limit; billing granularity or cancellation latency beyond the
limit must fit in the reserve. A job that would need more than its limit is cancelled by the provider (its usage up
to then is still billed) — the price of a hard ceiling. If a job's reported usage exceeds its high estimate,
reconciliation shows the overrun and further admission stops until the usage model is recalibrated. The lock assumes a
local file system (network shares and cloud-synced folders may not honor it).

    python -m qemx.campaign init      --ledger campaign_ledger.json --ceiling 600 --reserve 60
    python -m qemx.campaign estimate  --shots 720000 --circuits 1680 --jobs 12
    python -m qemx.campaign preflight data/dirA_div_qpu10 --ledger campaign_ledger.json --out preflight.md
    python -m qemx.campaign status    --ledger campaign_ledger.json
    python -m qemx.campaign reconcile --ledger campaign_ledger.json --job-id JOB --usage 41.2 --state completed
    python -m qemx.campaign resolve   --ledger campaign_ledger.json --id RES --job-id JOB      # found by its tag
    python -m qemx.campaign resolve   --ledger campaign_ledger.json --id RES --not-submitted  # provider shows none
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as _dt
import errno
import inspect
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
import uuid

DEFAULT_CEILING_S = 600.0
DEFAULT_RESERVE_S = 60.0
OPEN_PLAN_BATCH_TTL_S = 600.0          # Open plan: default batch max TTL (elapsed time, starts with the first job)
IBM_QUICK_PER_EXEC_S = 0.00035          # IBM "Workload usage": quick formula 2 + 0.00035 × executions
IBM_SUBJOB_OVERHEAD_S = 2.0             # "approximately 2s per sub-job"
MEASURED_PER_SHOT_S = 0.00057           # this project's ibm_fez measurement (replace with a pilot's job.usage())

SCHEMA_V1 = "qemx.campaign.ledger/1"
SCHEMA = "qemx.campaign.ledger/2"
FINAL_STATES = ("completed", "failed", "cancelled")
UNRESOLVED_STATES = ("submitting", "submission_unknown")          # the provider may have created a job
OPEN_STATES = ("reserved",) + UNRESOLVED_STATES + ("submitted",)  # unreconciled: committed at the high estimate
STATES = ("reserved", "submitting", "submission_unknown", "submitted") + FINAL_STATES + ("released",)
TAG_PREFIX = "qemx-ledger-"
LOCK_TIMEOUT_S = 60.0
ENV_LEDGER = "QEMX_CAMPAIGN_LEDGER"
SUBMIT_MODES = ("job", "batch")


class CampaignLimit(RuntimeError):
    """Admitting this job could exceed the campaign's conservative remaining allowance."""


class UnresolvedSubmission(CampaignLimit):
    """The ledger holds a submission whose outcome is unknown; nothing new is submitted until it is resolved."""


class LedgerError(ValueError):
    """The campaign ledger is missing, unreadable, invalid or misconfigured."""


class LedgerNotFound(LedgerError, FileNotFoundError):
    """A configured ledger path does not exist (never treated as "no ledger")."""


class LedgerRequired(LedgerError):
    """Real-hardware submission was attempted without a campaign ledger."""


class LedgerBusy(TimeoutError):
    """Another process held the ledger's lock for longer than the timeout."""


class SubmissionRejected(RuntimeError):
    """Raised by a submit callable to state that the provider *definitely* did not create a job (for example, a
    local validation failed before any request was sent). It is the only provider-call outcome that releases a
    reservation. (Classification is by this type only: a provider's own RuntimeError proves nothing.)"""


class SubmissionUnknown(RuntimeError):
    """The provider call failed in a way that does not prove that no job exists, or the job id could not be
    recorded. The reservation stays committed (``submission_unknown``, or ``submitting`` when even that could not
    be written) until it is resolved. ``job_id`` is set when the provider returned one that could not be recorded."""

    def __init__(self, message: str, rid: str, client_tag: str, job_id: str | None = None, ledger_path: str = ""):
        super().__init__(message)
        self.rid, self.client_tag, self.job_id, self.ledger_path = rid, client_tag, job_id, ledger_path


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


def default_max_execution_time(est: UsageEstimate) -> int:
    """The per-job ``max_execution_time`` used when none is given: the job's high estimate, rounded up to whole
    seconds (at least 1). The ledger reserves at least this much (:func:`reservation_s`), so the provider-enforced
    per-job limits of all unreconciled jobs never add up to more than the admitted allowance (1.5.1, review N12:
    1.5.0 set 1.1 × high + 1 s while reserving only the high estimate)."""
    return max(1, int(math.ceil(float(est.high_s) - 1e-9)))


def reservation_s(est: UsageEstimate, max_execution_time: int | None = None) -> float:
    """What the ledger commits for a job until it is reconciled: the larger of its high estimate and the per-job
    ``max_execution_time`` it will be submitted with (the default limit when None)."""
    lim = default_max_execution_time(est) if max_execution_time is None else int(max_execution_time)
    return float(max(float(est.high_s), float(lim)))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(s) -> float | None:
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


# ====================================================================== inter-process lock and atomic files
if os.name == "nt":                                       # pragma: no cover - exercised on Windows only
    import msvcrt

    def _try_os_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)       # one byte at offset 0 (beyond EOF is allowed)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EDEADLK, getattr(errno, "EDEADLOCK", errno.EDEADLK), errno.EAGAIN):
                return False
            raise

    def _os_unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_os_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # per open file: also excludes other threads' fds
            return True
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                return False
            raise

    def _os_unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class _PathLock:
    """Process-wide state of one ledger path: a thread lock (Streamlit sessions are threads of one process) and the
    OS lock on ``<ledger>.lock``, taken once per outermost transaction."""

    def __init__(self):
        self.rlock = threading.RLock()
        self.depth = 0
        self.thread = None
        self.owner = None
        self.fd = None


_PATH_LOCKS: dict = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _key(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _path_lock(path: str) -> _PathLock:
    k = _key(path)
    with _PATH_LOCKS_GUARD:
        pl = _PATH_LOCKS.get(k)
        if pl is None:
            pl = _PATH_LOCKS[k] = _PathLock()
        return pl


@contextlib.contextmanager
def _exclusive(path: str, timeout: float, owner=None):
    """Hold the ledger's exclusive lock (thread + inter-process). Re-entrant for the same thread and owner."""
    pl = _path_lock(path)
    me = threading.get_ident()
    if pl.depth and pl.thread == me:
        if owner is not None and pl.owner is not None and pl.owner is not owner:
            raise RuntimeError("a second Ledger object on %s started a transaction inside another one's; use one "
                               "object per transaction" % path)
        pl.depth += 1
        try:
            yield True
        finally:
            pl.depth -= 1
        return
    end = time.monotonic() + max(0.0, float(timeout))
    if not pl.rlock.acquire(timeout=max(0.0, float(timeout))):
        raise LedgerBusy("the campaign ledger %s is busy (another session holds it); try again" % path)
    try:
        lock_path = path + ".lock"
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        except FileNotFoundError:
            raise LedgerNotFound("the folder of the campaign ledger %s does not exist" % path) from None
        try:
            wait = 0.002
            while not _try_os_lock(fd):
                if time.monotonic() >= end:
                    raise LedgerBusy("the campaign ledger %s is locked by another process (%s); try again"
                                     % (path, lock_path))
                time.sleep(wait * (0.5 + random.random()))
                wait = min(wait * 2, 0.05)
            pl.depth, pl.thread, pl.owner, pl.fd = 1, me, owner, fd
            try:
                yield False
            finally:
                pl.depth, pl.thread, pl.owner, pl.fd = 0, None, None, None
                _os_unlock(fd)
        finally:
            os.close(fd)
    finally:
        pl.rlock.release()


def _replace(src: str, dst: str) -> None:
    for i in range(50):                                   # Windows: an antivirus/indexer may hold the file briefly
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or i == 49:
                raise
            time.sleep(0.05)


def _atomic_write_json(path: str, data: dict) -> None:
    """Write ``data`` to ``path`` atomically: a uniquely named temporary file in the same directory, flushed and
    fsynced, then ``os.replace``. Readers see the old or the new file, never a mixture."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if os.name != "nt":                                   # make the rename itself durable (POSIX)
        try:
            dfd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass


def _migrate(d: dict, path: str) -> dict:
    """A ledger read from disk in the current schema. ``qemx.campaign.ledger/1`` files are upgraded in memory (and
    written as /2 by their next change); no committed record is dropped and the committed total is unchanged:
    a v1 ``reserved`` entry may have become a job before 1.5.0 recorded its id, so it becomes
    ``submission_unknown`` (still committed at its high estimate) until it is resolved."""
    if not isinstance(d, dict):
        raise LedgerError("%s is not a campaign ledger" % path)
    schema = d.get("schema")
    if schema == SCHEMA:
        return d
    if schema != SCHEMA_V1:
        raise LedgerError("%s is not a campaign ledger (schema %r)" % (path, schema))
    out = dict(d, schema=SCHEMA, migrated_from=SCHEMA_V1)
    entries = []
    for e in d.get("entries", []):
        e = dict(e)
        e.setdefault("client_tag", None)
        e.setdefault("tag_applied", None)
        e.setdefault("max_execution_time_s", None)
        e.setdefault("history", [])
        if e.get("state") == "reserved":
            e["state"] = "submission_unknown"
            e["note"] = ("migrated from %s: a reservation without a job id (1.5.0 could not tell whether it became "
                         "a job); check the provider's job list and resolve it" % SCHEMA_V1)
            e["history"] = list(e["history"]) + [[_now(), "submission_unknown", "migrated from ledger/1 'reserved'"]]
        entries.append(e)
    out["entries"] = entries
    return out


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        raise LedgerNotFound("the campaign ledger %s does not exist (create it once with `python -m qemx.campaign "
                             "init --ledger %s`)" % (path, path)) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LedgerError("cannot read the campaign ledger %s (%s: %s)" % (path, type(e).__name__, e)) from None
    d = _migrate(d, path)
    if not isinstance(d.get("entries"), list):
        raise LedgerError("%s is not a campaign ledger (no entries)" % path)
    for e in d["entries"]:
        if e.get("state") not in STATES:
            raise LedgerError("%s: entry %s has an unknown state %r" % (path, e.get("id"), e.get("state")))
    return d


def _reserved(e: dict) -> float:
    """What an unreconciled entry commits: ``reserved_s`` (1.5.1: max of the high estimate and the per-job limit),
    or the high estimate for entries written before that field existed."""
    r = e.get("reserved_s")
    return float(r) if r is not None else float(e["est_high_s"])


def job_tag(rid: str) -> str:
    """The client tag of a reservation, sent as the job's tag so a lost submission response can be found again."""
    return TAG_PREFIX + str(rid)


# ====================================================================== the ledger
class Ledger:
    """The campaign's usage ledger (see the module text). Every method that changes it runs one transaction:
    lock → re-read from disk → check → change → atomic write → unlock; the object's view (``d``, ``entries``) is the
    file as of its last transaction (``refresh()`` re-reads it)."""

    def __init__(self, path: str, data: dict, lock_timeout: float = LOCK_TIMEOUT_S):
        self.path = os.path.abspath(path)
        self.d = data
        self.lock_timeout = float(lock_timeout)

    # ------------------------------------------------------------------ files and transactions
    @classmethod
    def create(cls, path: str, ceiling_s: float = DEFAULT_CEILING_S, reserve_s: float = DEFAULT_RESERVE_S,
               account_remaining_s: float | None = None, max_outstanding_s: float | None = None,
               note: str = "", lock_timeout: float = LOCK_TIMEOUT_S) -> "Ledger":
        for v, nm in ((ceiling_s, "ceiling"), (reserve_s, "reserve")):
            if not (math.isfinite(float(v)) and float(v) >= 0):
                raise ValueError("%s must be finite and ≥ 0" % nm)
        if float(reserve_s) >= float(ceiling_s):
            raise ValueError("the reserve must be smaller than the ceiling")
        path = os.path.abspath(path)
        if not os.path.isdir(os.path.dirname(path)):
            raise LedgerNotFound("the folder of %s does not exist" % path)
        data = dict(schema=SCHEMA, created=_now(), ceiling_s=float(ceiling_s), reserve_s=float(reserve_s),
                    account_remaining_s=None if account_remaining_s is None else float(account_remaining_s),
                    max_outstanding_s=None if max_outstanding_s is None else float(max_outstanding_s),
                    note=str(note)[:500], entries=[])
        led = cls(path, data, lock_timeout)
        with _exclusive(path, lock_timeout, led):
            if os.path.exists(path):
                raise FileExistsError("%s exists; a campaign ledger is never overwritten" % path)
            _atomic_write_json(path, data)
        return led

    @classmethod
    def load(cls, path: str, lock_timeout: float = LOCK_TIMEOUT_S) -> "Ledger":
        """The ledger at ``path`` (LedgerNotFound if there is none, LedgerError if it is not a valid ledger)."""
        led = cls(path, {}, lock_timeout)
        led.refresh()
        return led

    def _require_file(self) -> None:
        if os.path.isdir(self.path):
            raise LedgerError("%s is a folder, not a campaign ledger" % self.path)
        if not os.path.exists(self.path):                 # checked first: a misspelled path leaves no lock file
            raise LedgerNotFound("the campaign ledger %s does not exist (create it once with `python -m "
                                 "qemx.campaign init --ledger %s`)" % (self.path, self.path))

    def refresh(self) -> "Ledger":
        """Re-read the file (under the lock, so never a half-written state)."""
        self._require_file()
        with _exclusive(self.path, self.lock_timeout, self) as nested:
            if not nested:
                self.d = _read(self.path)
        return self

    @contextlib.contextmanager
    def transaction(self):
        """``with ledger.transaction() as d:`` — the current file as a dict, under the exclusive lock; changes to
        ``d`` are written atomically when the block ends without an exception and discarded otherwise. Nested
        transactions of the same object share the outer one."""
        self._require_file()
        with _exclusive(self.path, self.lock_timeout, self) as nested:
            if nested:
                yield self.d
                return
            disk = _read(self.path)
            work = copy.deepcopy(disk)
            self.d = work
            try:
                yield work
                if work != disk:
                    work["updated"] = _now()
                    _atomic_write_json(self.path, work)
            except BaseException:
                self.d = disk                             # the view is the file as it is: nothing was committed
                raise
            self.d = work

    # ------------------------------------------------------------------ accounting (on the current view)
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
            else:                                     # outstanding, unresolved, or ended without a reported usage
                tot += _reserved(e)
        return tot

    def outstanding_s(self) -> float:
        return sum(_reserved(e) for e in self.entries if e["state"] in OPEN_STATES)

    def limit_s(self) -> float:
        a = self.d.get("account_remaining_s")
        return min(float(self.d["ceiling_s"]), float(a)) if a is not None else float(self.d["ceiling_s"])

    def remaining_s(self) -> float:
        return self.limit_s() - float(self.d["reserve_s"]) - self.committed_s()

    def overruns(self) -> list:
        return [e for e in self.entries if e.get("usage_s") is not None
                and float(e["usage_s"]) > float(e["est_high_s"])]

    def unsubmitted_reservations(self) -> list:
        """Committed entries without a job id: reserved (never sent), submitting or submission_unknown (may be a
        job). They stay committed until submitted, resolved or released."""
        return [e for e in self.entries if e["state"] in ("reserved",) + UNRESOLVED_STATES]

    def unresolved(self) -> list:
        """Entries whose submission outcome is unknown (``submitting`` or ``submission_unknown``)."""
        return [e for e in self.entries if e["state"] in UNRESOLVED_STATES]

    def status(self, refresh: bool = True) -> dict:
        if refresh:
            self.refresh()
        by = {}
        for e in self.entries:
            by[e["state"]] = by.get(e["state"], 0) + 1
        return dict(unsubmitted_reservations=[e["id"] for e in self.unsubmitted_reservations()],
                    unresolved=[dict(id=e["id"], state=e["state"], client_tag=e.get("client_tag"),
                                     label=e.get("label"), since=e.get("updated")) for e in self.unresolved()],
                    ceiling_s=self.d["ceiling_s"], account_remaining_s=self.d.get("account_remaining_s"),
                    reserve_s=self.d["reserve_s"], committed_s=self.committed_s(), outstanding_s=self.outstanding_s(),
                    remaining_s=self.remaining_s(), jobs_by_state=by,
                    overruns=[e["job_id"] for e in self.overruns()],
                    reported_usage_s=sum(float(e["usage_s"]) for e in self.entries if e.get("usage_s") is not None))

    # ------------------------------------------------------------------ transitions (one transaction each)
    def _entry(self, rid: str) -> dict:
        for e in self.entries:
            if e["id"] == rid:
                return e
        raise KeyError("no reservation %s" % rid)

    def entry(self, rid: str) -> dict:
        """A copy of reservation ``rid`` as of this object's view."""
        return dict(self._entry(rid))

    def entry_for_job(self, job_id: str) -> dict | None:
        hit = [e for e in self.entries if e.get("job_id") == job_id]
        return dict(hit[0]) if hit else None

    def entry_for_tag(self, tag: str) -> dict | None:
        hit = [e for e in self.entries if e.get("client_tag") == tag]
        return dict(hit[0]) if hit else None

    @staticmethod
    def _move(e: dict, state: str, note: str = "", **kw) -> None:
        e.update(state=state, updated=_now(), **kw)
        if note:
            e["note"] = str(note)[:300]
        e.setdefault("history", []).append([e["updated"], state, str(note)[:200]])

    def admit(self, label: str, est: UsageEstimate, campaign: str = "", max_execution_time: int | None = None) -> str:
        """Reserve a job — the larger of its high estimate and the per-job ``max_execution_time`` it will be sent
        with (:func:`reservation_s`); CampaignLimit if that does not fit. Returns the reservation id. The check and
        the reservation are one transaction on the current file."""
        with self.transaction():
            over = self.overruns()
            if over:
                raise CampaignLimit("job %s used %.1f s, more than its high estimate %.1f s: the usage model is too "
                                    "optimistic; recalibrate it before admitting more work" % (
                                        over[0]["job_id"], float(over[0]["usage_s"]), float(over[0]["est_high_s"])))
            if not (math.isfinite(float(est.high_s)) and float(est.high_s) >= 0):
                raise ValueError("the high estimate must be finite and ≥ 0")
            if max_execution_time is not None and int(max_execution_time) < 1:
                raise ValueError("max_execution_time must be ≥ 1 s")
            met = default_max_execution_time(est) if max_execution_time is None else int(max_execution_time)
            need = reservation_s(est, met)
            rem = self.remaining_s()
            if need > rem:
                raise CampaignLimit("%s needs up to %.1f s (high estimate %.1f s, per-job limit %d s) but only %.1f s "
                                    "remain of the %.0f s ceiling (reserve %.0f s, committed %.1f s)" % (
                                        label, need, est.high_s, met, rem, self.limit_s(), self.d["reserve_s"],
                                        self.committed_s()))
            mo = self.d.get("max_outstanding_s")
            if mo is not None and self.outstanding_s() + need > float(mo):
                raise CampaignLimit("%s would raise unreconciled work to %.1f s (bound %.0f s): collect and reconcile "
                                    "the jobs in flight first" % (label, self.outstanding_s() + need, float(mo)))
            ids = {e["id"] for e in self.entries}
            rid = uuid.uuid4().hex[:12]
            while rid in ids:
                rid = uuid.uuid4().hex[:12]
            now = _now()
            self.entries.append(dict(id=rid, label=str(label)[:200], campaign=str(campaign)[:80], state="reserved",
                                     est_high_s=float(est.high_s), est_central_s=float(est.central_s),
                                     reserved_s=float(need), planned_max_execution_time_s=int(met),
                                     executions=int(est.executions), circuits=int(est.circuits), job_id=None,
                                     usage_s=None, created=now, updated=now, note="", client_tag=job_tag(rid),
                                     tag_applied=None, max_execution_time_s=None,
                                     history=[[now, "reserved", ""]]))
        return rid

    def begin_submit(self, rid: str, tag_applied: bool = False, max_execution_time: int | None = None) -> str:
        """reserved → submitting, persisted right before the provider call. Returns the client tag."""
        with self.transaction():
            e = self._entry(rid)
            if e["state"] != "reserved":
                raise ValueError("reservation %s is %s, not reserved" % (rid, e["state"]))
            if max_execution_time is not None and float(max_execution_time) > _reserved(e) + 1e-9:
                raise ValueError("reservation %s holds %.1f s; a per-job max_execution_time of %d s would let the "
                                 "provider run past it" % (rid, _reserved(e), int(max_execution_time)))
            self._move(e, "submitting", "provider call started", tag_applied=bool(tag_applied),
                       max_execution_time_s=None if max_execution_time is None else int(max_execution_time),
                       submit_started=_now(), owner_pid=os.getpid())
            return e["client_tag"]

    def submitted(self, rid: str, job_id: str) -> None:
        """Attach the job id (from reserved, submitting or submission_unknown)."""
        job_id = str(job_id or "")
        if not job_id:
            raise ValueError("a job id is required")
        with self.transaction():
            if any(e.get("job_id") == job_id for e in self.entries):
                raise ValueError("job %s is already recorded" % job_id)
            e = self._entry(rid)
            if e["state"] not in ("reserved",) + UNRESOLVED_STATES:
                raise ValueError("reservation %s is %s, not awaiting a job id" % (rid, e["state"]))
            self._move(e, "submitted", "", job_id=job_id)

    def mark_unknown(self, rid: str, reason: str = "") -> None:
        """reserved/submitting → submission_unknown: a job may exist; the high estimate stays committed."""
        with self.transaction():
            e = self._entry(rid)
            if e["state"] == "submission_unknown":
                return
            if e["state"] not in ("reserved", "submitting"):
                raise ValueError("reservation %s is %s; only an unsettled submission can become unknown"
                                 % (rid, e["state"]))
            self._move(e, "submission_unknown", reason or "the provider call failed; a job may exist")

    def release(self, rid: str, reason: str = "") -> None:
        """Give back a reservation that was never sent to the provider (state ``reserved``). An entry whose
        provider call started may be a job: it is settled only by :meth:`resolve` after checking the provider."""
        with self.transaction():
            e = self._entry(rid)
            if e["state"] != "reserved":
                raise ValueError("only a reservation that was never sent can be released (%s is %s; use resolve "
                                 "after checking the provider)" % (rid, e["state"]))
            self._move(e, "released", reason or "never sent")

    def resolve(self, rid: str, job_id: str | None, note: str = "") -> None:
        """Settle a reservation without a job id (reserved, submitting or submission_unknown) after checking the
        provider: ``job_id`` — the job it became (found by its client tag) → submitted; ``None`` — no job exists →
        released."""
        if job_id:
            with self.transaction():
                self.submitted(rid, job_id)
                e = self._entry(rid)
                e["note"] = str(note or "resolved: job found on the provider")[:300]
            return
        with self.transaction():
            e = self._entry(rid)
            if e["state"] not in ("reserved",) + UNRESOLVED_STATES:
                raise ValueError("reservation %s is %s; nothing to resolve" % (rid, e["state"]))
            self._move(e, "released", note or "resolved: the provider shows no job for this reservation")

    def reconcile(self, job_id: str, usage_s: float | None, state: str = "completed", note: str = "") -> None:
        """Record what the provider reports for a job (``usage_s`` None: not reported yet — the high estimate stays
        committed)."""
        if state not in FINAL_STATES + ("submitted",):
            raise ValueError("state must be one of %s" % ", ".join(FINAL_STATES))
        if usage_s is not None and not (math.isfinite(float(usage_s)) and float(usage_s) >= 0):
            raise ValueError("usage must be finite and ≥ 0")
        with self.transaction():
            hit = [e for e in self.entries if e.get("job_id") == job_id]
            if not hit:
                raise KeyError("job %s is not in the ledger" % job_id)
            e = hit[0]
            u = None if usage_s is None else float(usage_s)
            if e["state"] == state and e.get("usage_s") == u and (not note or e.get("note") == str(note)[:300]):
                return                                    # nothing new: no write
            self._move(e, state, note, usage_s=u)


# ====================================================================== configuration and guards
def open_ledger(path, lock_timeout: float = LOCK_TIMEOUT_S) -> Ledger:
    """The ledger configured at ``path``: LedgerRequired when no path is configured, LedgerNotFound when the path
    does not exist, LedgerError when the file is not a valid ledger. Never None."""
    if path is None or not str(path).strip():
        raise LedgerRequired("no campaign ledger is configured")
    return Ledger.load(os.path.expanduser(str(path).strip()), lock_timeout)


def ledger_from_env(var: str = ENV_LEDGER) -> Ledger:
    """The ledger named by the environment variable ``var`` (default QEMX_CAMPAIGN_LEDGER), with the errors of
    :func:`open_ledger`; an unset variable is LedgerRequired, a misspelled path LedgerNotFound."""
    path = os.environ.get(var, "")
    if not path.strip():
        raise LedgerRequired("real-hardware submission needs the campaign ledger: set %s to a ledger created with "
                             "`python -m qemx.campaign init --ledger campaign_ledger.json --ceiling 600 --reserve 60`"
                             % var)
    try:
        return open_ledger(path)
    except LedgerNotFound as e:
        raise LedgerNotFound("%s=%s does not exist: %s" % (var, path, e)) from None


def is_local_backend(backend) -> bool:
    """True only for backends that run on this machine and cost no QPU time: qiskit-ibm-runtime fake backends,
    Qiskit's GenericBackendV2 and Aer simulators. Anything else is treated as a real provider."""
    checks = (("qiskit_ibm_runtime.fake_provider.fake_backend", "FakeBackendV2"),
              ("qiskit.providers.fake_provider", "GenericBackendV2"),
              ("qiskit_aer", "AerSimulator"))
    for mod, cls in checks:
        try:
            import importlib
            C = getattr(importlib.import_module(mod), cls)
        except Exception:                                 # noqa: BLE001 - optional packages
            continue
        if isinstance(backend, C):
            return True
    return False


def require_ledger(ledger, backend=None):
    """The ledger to use for a submission to ``backend``: a :class:`Ledger` (a path is opened with
    :func:`open_ledger`), or None for a local backend only. LedgerRequired for a real provider without one."""
    if ledger is None:
        if backend is not None and is_local_backend(backend):
            return None
        raise LedgerRequired("real-hardware submission needs the campaign ledger (one shared QPU ceiling for the "
                             "whole thesis); create it with `python -m qemx.campaign init --ledger "
                             "campaign_ledger.json --ceiling 600 --reserve 60` and pass it (the app reads %s)"
                             % ENV_LEDGER)
    if isinstance(ledger, (str, os.PathLike)):
        return open_ledger(ledger)
    if not isinstance(ledger, Ledger):
        raise LedgerError("ledger must be a qemx.campaign.Ledger, not %s" % type(ledger).__name__)
    return ledger


def resolve_hint(ledger_path: str, e: dict) -> str:
    return ("reservation %s (client tag %s, %s): if the provider lists a job with that tag, `python -m qemx.campaign "
            "resolve --ledger %s --id %s --job-id <JOB>`; if it lists none, `… --id %s --not-submitted`"
            % (e["id"], e.get("client_tag") or "none", e["state"], ledger_path, e["id"], e["id"]))


def check_no_unresolved(ledger: Ledger) -> None:
    """UnresolvedSubmission if the (re-read) ledger holds a submission whose outcome is unknown: work that may
    already be a job is never sent again before it is resolved. (An entry another live process is submitting right
    now looks the same for a moment; try again then.)"""
    ledger.refresh()
    bad = ledger.unresolved()
    if bad:
        raise UnresolvedSubmission("the campaign ledger holds %d submission(s) whose outcome is unknown — they may "
                                   "be jobs and stay committed. Resolve before submitting more: %s" % (
                                       len(bad), "; ".join(resolve_hint(ledger.path, e) for e in bad[:3])))


def tag_sampler(sampler, tag: str) -> bool:
    """Set a qiskit-ibm-runtime primitive's job tag (``options.environment.job_tags``); False if the object has no
    such option (then a lost response can only be found in the provider's job list by time and size)."""
    try:
        env = sampler.options.environment
        tags = [t for t in (getattr(env, "job_tags", None) or []) if not str(t).startswith(TAG_PREFIX)]
        env.job_tags = tags + [tag]
        return True
    except (AttributeError, TypeError, ValueError):
        return False


# ====================================================================== one guarded submission (every submitter)
def submit_one(ledger: Ledger, label: str, est: UsageEstimate, run, campaign: str = "", prepare=None,
               on_attempt=None, max_execution_time: int | None = None) -> tuple:
    """Admit ``est`` and make one provider call, with the spending policy every submitter shares:

    1. admit (CampaignLimit if it does not fit) — reserved;
    2. ``prepare(tag)`` and ``on_attempt(dict(reservation, client_tag))``: local work before any provider call
       (set options and the job tag — ``prepare`` returns whether the tag was applied —, write a pending record).
       If either raises, nothing was sent: the reservation is released;
    3. persist ``submitting`` with the client tag;
    4. ``run(tag)`` → a job; its ``job_id()``. :class:`SubmissionRejected` (definitely no job) releases the
       reservation and is re-raised. Any other exception — a timeout or lost connection after the provider may
       have accepted the job, anything unexpected — moves it to ``submission_unknown`` (committed, blocking) and
       raises :class:`SubmissionUnknown`; KeyboardInterrupt/SystemExit are re-raised after that bookkeeping;
    5. record the job id (``submitted``). If that write fails, SubmissionUnknown carries the job id.

    Returns (job, job_id, reservation id, client tag)."""
    if max_execution_time is None:
        max_execution_time = default_max_execution_time(est)
    rid = ledger.admit(label, est, campaign, max_execution_time)
    tag = job_tag(rid)
    try:
        applied = bool(prepare(tag)) if prepare is not None else False
        if on_attempt is not None:
            on_attempt(dict(reservation=rid, client_tag=tag))
    except BaseException as e:
        try:
            ledger.release(rid, "not sent: local preparation failed (%s)" % type(e).__name__)
        except BaseException:                             # noqa: BLE001 - stays reserved (committed) if this fails
            pass
        raise
    try:
        ledger.begin_submit(rid, applied, max_execution_time)
    except BaseException:
        try:
            ledger.release(rid, "not sent: could not record the submission start")
        except BaseException:                             # noqa: BLE001
            pass
        raise
    try:
        job = run(tag)
        job_id = job.job_id()
        if not job_id:
            raise RuntimeError("the provider returned a job without an id")
    except SubmissionRejected as e:
        try:
            ledger.resolve(rid, None, "rejected before submission: %s" % str(e)[:200])
        except BaseException:                             # noqa: BLE001 - then it stays committed (submitting)
            pass
        raise
    except BaseException as e:
        try:
            ledger.mark_unknown(rid, "%s during the provider call: %s" % (type(e).__name__, str(e)[:200]))
        except BaseException:                             # noqa: BLE001 - it stays 'submitting' (also unresolved)
            pass
        if not isinstance(e, Exception):
            raise
        raise SubmissionUnknown("the provider call for reservation %s failed with %s (%s); a job may exist. Its "
                                "high estimate stays committed and nothing more is submitted until it is resolved: "
                                "look for a job tagged %s, then `python -m qemx.campaign resolve --ledger %s --id %s "
                                "--job-id <JOB>` (found) or `--not-submitted` (none)"
                                % (rid, type(e).__name__, str(e)[:200], tag, ledger.path, rid),
                                rid, tag, None, ledger.path) from e
    try:
        ledger.submitted(rid, job_id)
    except BaseException as e:
        if not isinstance(e, Exception):
            raise
        raise SubmissionUnknown("job %s was created but could not be recorded in the ledger (%s: %s); attach it: "
                                "`python -m qemx.campaign resolve --ledger %s --id %s --job-id %s`"
                                % (job_id, type(e).__name__, str(e)[:200], ledger.path, rid, job_id),
                                rid, tag, str(job_id), ledger.path) from e
    return job, str(job_id), rid, tag


# ====================================================================== provider results → ledger (every collector)
PROVIDER_FINAL = {"DONE": "completed", "ERROR": "failed", "CANCELLED": "cancelled"}


def provider_state(status) -> str:
    """'DONE', 'ERROR', 'CANCELLED', 'RUNNING', … from a provider status (string or JobStatus enum)."""
    return str(status).split(".")[-1].upper()


def job_usage(job) -> float | None:
    """The QPU seconds a provider job reports (qiskit-ibm-runtime ≥ 0.30: ``job.usage()``), None if not reported."""
    try:
        u = job.usage()
        u = None if u is None else float(u)
        return u if u is not None and math.isfinite(u) and u >= 0 else None
    except Exception:                                     # noqa: BLE001 - not reported (yet)
        return None


def reconcile_job(ledger: Ledger, job_id: str, job, status=None):
    """Record one ended provider job's state and reported usage in the ledger — the single reconciliation step
    used by the app, the CLIs and :func:`reconcile_jobs`. Returns (ledger state, usage) or None when the job has not
    ended or is not in this ledger."""
    st = provider_state(job.status() if status is None else status)
    if st not in PROVIDER_FINAL:
        return None
    u = job_usage(job)
    try:
        ledger.reconcile(job_id, u, PROVIDER_FINAL[st])
    except KeyError:                                      # submitted before this ledger existed
        return None
    return PROVIDER_FINAL[st], u


def reconcile_jobs(ledger: Ledger, job_lookup) -> list:
    """Reconcile every submitted job, and every ended job whose usage was not reported yet: ``job_lookup(job_id)``
    returns a job with ``status()`` and ``usage()`` (``qiskit_ibm_runtime`` ≥ 0.30 jobs have both). Returns
    [(job_id, state, usage)]."""
    out = []
    ledger.refresh()
    for e in list(ledger.entries):
        if not (e["state"] == "submitted" or (e["state"] in FINAL_STATES and e.get("usage_s") is None)):
            continue
        r = reconcile_job(ledger, e["job_id"], job_lookup(e["job_id"]))
        if r is not None:
            out.append((e["job_id"],) + tuple(r))
    return out


def ibm_tag_lookup(service, limit: int = 5):
    """``find_jobs(tag) → [job id]`` for :func:`resolve_unknown`, through ``service.jobs(job_tags=[tag])``."""
    def find(tag):
        return [j.job_id() for j in service.jobs(job_tags=[tag], limit=limit)]
    return find


def resolve_unknown(ledger: Ledger, find_jobs, release_missing: bool = False, min_age_s: float = 900.0,
                    now: float | None = None) -> list:
    """Settle unresolved submissions through the provider: ``find_jobs(client_tag)`` returns the ids of the jobs
    carrying that tag. Exactly one → attached (submitted). None → released only when ``release_missing`` is set, the
    tag was applied to the call and the attempt is at least ``min_age_s`` old (a job list can lag behind a
    submission); otherwise left unresolved. Several → left unresolved (a duplicate needs a human). A failing lookup
    leaves the entry unchanged. Returns [dict(id, client_tag, outcome, job_ids)]."""
    out = []
    ledger.refresh()
    now = time.time() if now is None else float(now)
    for e in list(ledger.unresolved()):
        tag = e.get("client_tag")
        row = dict(id=e["id"], client_tag=tag, outcome="unresolved", job_ids=[])
        if not tag:
            row["outcome"] = "no client tag: check the provider's job list by time and resolve by hand"
            out.append(row)
            continue
        try:
            ids = [str(j) for j in (find_jobs(tag) or [])]
        except Exception as ex:                           # noqa: BLE001 - lookup failed: nothing is concluded
            row["outcome"] = "lookup failed (%s)" % type(ex).__name__
            out.append(row)
            continue
        row["job_ids"] = ids
        if len(ids) == 1:
            ledger.resolve(e["id"], ids[0], "resolved: job found on the provider by its client tag %s" % tag)
            row["outcome"] = "attached"
        elif len(ids) > 1:
            row["outcome"] = "several jobs carry this tag: resolve by hand"
        else:
            started = _parse_time(e.get("submit_started") or e.get("updated"))
            old = started is not None and now - started >= float(min_age_s)
            if release_missing and e.get("tag_applied") and old:
                ledger.resolve(e["id"], None, "resolved: no job with client tag %s on the provider" % tag)
                row["outcome"] = "released"
            else:
                row["outcome"] = "not found yet" if e.get("tag_applied") else \
                    "not found (the tag was not applied to this call: check the job list by hand)"
        out.append(row)
    return out


def sync_pending(ledger: Ledger, pending: list) -> tuple:
    """Pending submission records (dicts with ``reservation``; written by a submitter before its provider call)
    against the ledger: (records that became jobs — with their ``job_id`` filled in —, records that were not
    submitted, records still unresolved)."""
    ledger.refresh()
    jobs, gone, open_ = [], [], []
    for p in pending or []:
        try:
            e = ledger.entry(p["reservation"])
        except KeyError:
            open_.append(p)
            continue
        if e.get("job_id"):
            q = dict(p, job_id=e["job_id"])
            q.pop("outcome", None)
            jobs.append(q)
        elif e["state"] == "released":
            gone.append(p)
        else:
            open_.append(p)
    return jobs, gone, open_


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
        remaining = ledger.refresh().remaining_s()
    else:
        remaining = ceiling_s - reserve_s
    rows, cum_high, admitted = [], 0.0, 0
    for label, shots, circ in jobs:
        est = estimate_usage(shots * circ, circ, 1, m)
        need = reservation_s(est)
        fits = cum_high + need <= remaining
        if fits:
            cum_high += need
            admitted += 1
        rows.append(dict(job=label, executions=shots * circ, circuits=circ, low_s=est.low_s, central_s=est.central_s,
                         high_s=est.high_s, max_execution_time_s=default_max_execution_time(est), reserved_s=need,
                         admitted=bool(fits)))
    tot = estimate_usage(sum(s * c for _, s, c in jobs), sum(c for _, _, c in jobs), len(jobs), m)
    return dict(jobs=rows, total=tot.as_dict(), remaining_s=remaining, admitted_jobs=admitted, total_jobs=len(jobs),
                reserved_s=sum(r["reserved_s"] for r in rows),
                fits=admitted == len(jobs), open_plan_batch_ttl_s=OPEN_PLAN_BATCH_TTL_S,
                batch_ttl_risk=bool(tot.high_s > 0.5 * OPEN_PLAN_BATCH_TTL_S))


def preflight_markdown(pf: dict, title: str = "Hardware preflight") -> str:
    t = pf["total"]
    L = ["# %s" % title, "",
         "**Planned QPU usage:** %.0f s (low, IBM quick formula) · %.0f s (central) · **%.0f s (high, used for "
         "admission)** for %d jobs, %s executions." % (t["low_s"], t["central_s"], t["high_s"], pf["total_jobs"],
                                                       f"{t['executions']:,}"),
         "**Reserved by the ledger** (per job: the larger of the high estimate and its provider-enforced "
         "`max_execution_time`): %.0f s. **Conservative allowance available:** %.0f s. %s" % (
             pf.get("reserved_s", t["high_s"]), pf["remaining_s"], "All jobs fit." if pf["fits"] else "**Only %d of %d jobs fit: the plan must be cut "
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
          "automatically; a submission whose outcome is unknown stays committed and blocks further submissions until "
          "it is resolved; a failed or cancelled job keeps its reported usage (or its high estimate until reported); "
          "if any job's reported usage exceeds its high estimate, admission stops until the usage model is "
          "recalibrated. A per-job `max_execution_time` bounds a single job only."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ a mock provider for tests and rehearsals
class MockJob:
    def __init__(self, jid, usage, fail=False, tags=()):
        self._id, self._usage, self._fail, self.tags = jid, usage, fail, list(tags)

    def job_id(self):
        return self._id

    def usage(self):
        return self._usage

    def status(self):
        return "ERROR" if self._fail else "DONE"


class MockProvider:
    """Deterministic stand-in for a QPU service: true usage = overhead + true per-shot cost × executions. It can
    fail jobs (usage still consumed), reject a submission definitely (no job: :class:`SubmissionRejected`), accept a
    job and then lose the response (``timeout_after_create_on``: TimeoutError), or 'crash' after creating a job.
    Jobs carry the client tag they were sent with, and ``jobs_with_tag`` is the provider-side job search."""

    def __init__(self, true_per_shot_s: float = MEASURED_PER_SHOT_S, overhead_s: float = 2.0, fail_every: int = 0,
                 submit_error_on: int = -1, crash_after_create_on: int = -1, prefix: str = "mock",
                 timeout_after_create_on: int = -1):
        self.prefix = prefix
        self.true_per_shot_s, self.overhead_s = float(true_per_shot_s), float(overhead_s)
        self.fail_every, self.submit_error_on, self.crash_after_create_on = fail_every, submit_error_on, crash_after_create_on
        self.timeout_after_create_on = timeout_after_create_on
        self.jobs, self.n = {}, 0

    def run(self, executions: int, client_tag: str | None = None):
        self.n += 1
        if self.n == self.submit_error_on:
            raise SubmissionRejected("submission rejected")
        jid = "%s-%04d" % (self.prefix, self.n)
        job = MockJob(jid, self.overhead_s + self.true_per_shot_s * int(executions),
                      fail=bool(self.fail_every and self.n % self.fail_every == 0),
                      tags=[client_tag] if client_tag else [])
        self.jobs[jid] = job
        if self.n == self.crash_after_create_on:
            raise KeyboardInterrupt("crash after the job was created")
        if self.n == self.timeout_after_create_on:
            raise TimeoutError("the response was lost after the job was accepted")
        return job

    def jobs_with_tag(self, tag: str) -> list:
        return [jid for jid, j in self.jobs.items() if tag in j.tags]

    def total_usage(self) -> float:
        return sum(j.usage() for j in self.jobs.values())


def _accepts_tag(fn) -> bool:
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "client_tag" in ps or any(p.kind == p.VAR_KEYWORD for p in ps.values())


def submit_with_ledger(ledger: Ledger, jobs: list, run_job, model: UsageModel | None = None, campaign: str = "") -> list:
    """Admit and submit jobs one by one through :func:`submit_one`: [(label, shots, circuits)] → records.
    ``run_job(label, shots, circuits)`` returns a job object with ``job_id()``; it also receives
    ``client_tag=...`` when it accepts that keyword (send it as the job's tag). Refuses to start while the ledger
    holds an unresolved submission. Stops at the first job the ledger refuses (CampaignLimit is re-raised with the
    records so far attached as ``.records``) and at the first submission whose outcome is unknown
    (SubmissionUnknown, also with ``.records``)."""
    m = model or UsageModel()
    check_no_unresolved(ledger)
    records = []
    tagged = _accepts_tag(run_job)
    for label, shots, circ in jobs:
        est = estimate_usage(shots * circ, circ, 1, m)

        def run(tag, label=label, shots=shots, circ=circ):
            return run_job(label, shots, circ, client_tag=tag) if tagged else run_job(label, shots, circ)
        try:
            job, jid, rid, tag = submit_one(ledger, label, est, run, campaign, prepare=lambda tag: tagged)
        except (CampaignLimit, SubmissionUnknown, SubmissionRejected) as e:
            e.records = records
            raise
        records.append(dict(job_id=jid, label=label, reservation=rid, client_tag=tag, est_high_s=est.high_s))
    return records


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
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
    p = sub.add_parser("resolve", help="settle a submission whose outcome is unknown, after checking the provider's "
                                       "job list for its client tag (status lists them)")
    p.add_argument("--ledger", required=True)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--id", help="the reservation id")
    who.add_argument("--tag", help="the client tag (%s<id>)" % TAG_PREFIX)
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument("--job-id", default=None, help="the job the provider lists with that tag")
    how.add_argument("--not-submitted", action="store_true",
                     help="the provider lists no job with that tag (checked): release the reservation")
    p.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "resolve":
            led = open_ledger(a.ledger)
            rid = a.id
            if a.tag:
                e = led.entry_for_tag(a.tag)
                if e is None:
                    raise KeyError("no reservation has client tag %s" % a.tag)
                rid = e["id"]
            led.resolve(rid, None if a.not_submitted else a.job_id,
                        a.note or ("resolved by hand: provider checked, no job" if a.not_submitted
                                   else "resolved by hand: job found on the provider"))
            print(json.dumps(led.status(), indent=1))
            return 0
        if a.cmd == "init":
            Ledger.create(a.ledger, a.ceiling, a.reserve, a.account_remaining, a.max_outstanding, a.note)
            print("created %s (ceiling %.0f s, reserve %.0f s)" % (a.ledger, a.ceiling, a.reserve))
        elif a.cmd == "status":
            led = open_ledger(a.ledger)
            st = led.status()
            print(json.dumps(st, indent=1))
            for e in led.unresolved():
                print("unresolved: " + resolve_hint(led.path, e), file=sys.stderr)
        elif a.cmd == "estimate":
            m = UsageModel(per_shot_s=a.per_shot_ms / 1e3, margin=a.margin)
            print(json.dumps(estimate_usage(a.shots, a.circuits, a.jobs, m).as_dict(), indent=1))
        elif a.cmd == "preflight":
            with open(os.path.join(a.run_dir, "plan.json"), encoding="utf-8") as f:
                plan = json.load(f)
            led = open_ledger(a.ledger) if a.ledger else None
            m = UsageModel(per_shot_s=a.per_shot_ms / 1e3, margin=a.margin)
            md = preflight_markdown(preflight(plan, led, m, a.ceiling, a.reserve),
                                    "Hardware preflight: %s" % os.path.basename(os.path.abspath(a.run_dir)))
            if a.out:
                with open(a.out, "w", encoding="utf-8") as f:
                    f.write(md)
            print(md)
        elif a.cmd == "reconcile":
            led = open_ledger(a.ledger)
            led.reconcile(a.job_id, a.usage, a.state)
            print(json.dumps(led.status(), indent=1))
        return 0
    except (ValueError, KeyError, FileExistsError, CampaignLimit, LedgerBusy) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
