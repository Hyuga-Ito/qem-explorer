"""Completion and data identity of a labels directory (1.9.0; second review N1, N4).

A labels directory (``dev``, ``final``, ``confirm``, ``regimes`` …) is bound to a task manifest (``run.json``,
``binding.bind``). Before 1.9.0 a labeling session reported only what *it* did: a task that had failed in an earlier
session (``timeout``, ``memory``, ``error``) was skipped on resume and the session reported zero errors, so an
unchanged second invocation exited 0 while the run stayed incomplete. ``completion`` describes the *whole run*:

* ``expected`` — the manifest's tasks;
* ``completed`` — records with status ``ok`` whose arrays exist and match the digest in the record;
* ``failed`` — records without labels, by status (software or resource failures: never mathematical infeasibility,
  which lives inside an ``ok`` record as an infeasible allocation);
* ``missing`` — manifest tasks without a record (never attempted, interrupted, or stopped by a ceiling);
* ``corrupted`` — records that cannot be used: unreadable, another task id, another label configuration or run, an
  ``ok`` record whose arrays are missing or differ from its digest;
* ``unexpected`` — task files that are not in the manifest (never used, reported).

``data_digest`` identifies the evaluated *content*: the sorted (task id, JSON digest, array digest) of the completed
manifest tasks. A freeze, an analysis and a replay record it, so a result can be traced to exactly the records it
was computed from, and a later change of those records is detected.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from .. import textio as TX
from . import labels as LB

SCHEMA = "qemx.selection.completion/1"
FAILED = ("error", "timeout", "memory")


class IncompleteRun(RuntimeError):
    """A labels directory whose manifest is not completely labeled, used where completeness is required."""


class CorruptedRun(RuntimeError):
    """A labels directory with records that cannot be used (unreadable, foreign, or arrays missing/changed)."""


def _sha(path: str) -> str:
    return LB.file_digest(path)


def _check_one(target: str, tid: str, run: dict | None, imports: dict, check_arrays: str) -> tuple:
    """(kind, detail, record) of one task record: kind ok | failed | corrupted."""
    jp = os.path.join(target, "tasks", tid + ".json")
    try:
        r = LB.load_record(target, tid)
    except Exception as e:                                   # noqa: BLE001 - an unreadable record is corrupted data
        return "corrupted", "unreadable record: %s" % str(e)[:160], None
    if r.get("task_id") != tid:
        return "corrupted", "the record names task %s" % r.get("task_id"), r
    if run is not None:
        want = run["identity"]["labels_config"]
        if r.get("config_digest") != want:
            return "corrupted", "label configuration %s, the run's is %s" % (r.get("config_digest"), want), r
        rid = r.get("run_id", "")
        if tid in imports:
            if _sha(jp) != imports[tid].get("json_sha256"):
                return "corrupted", "imported record changed after its import", r
        elif rid != run["identity_digest"] and not (rid == "" and "seed_scheme" not in r):
            return "corrupted", "run %s, not this run %s" % (rid or "(none)", run["identity_digest"]), r
    st = r.get("status")
    if st in FAILED:
        return "failed", st, r
    if st != "ok":
        return "corrupted", "unknown status %r" % st, r
    npz = os.path.join(target, "tasks", tid + ".npz")
    if not os.path.exists(npz):
        return "corrupted", "the record says ok but its arrays (%s.npz) are missing" % tid, r
    if check_arrays in ("digest", "load"):
        want = (r.get("arrays") or {}).get("sha256")
        if want and _sha(npz) != want:
            return "corrupted", "arrays do not match the digest in the record", r
        if check_arrays == "load":
            try:
                LB.load_arrays(target, tid, r)
            except Exception as e:                           # noqa: BLE001
                return "corrupted", str(e)[:200], r
    return "ok", "", r


def completion(target: str, *, check_arrays: str = "digest", task_ids=None) -> dict:
    """Run-wide completion of the labels directory ``target`` (see the module text). ``check_arrays``: 'exists'
    (fast: arrays present), 'digest' (default: arrays match the record's digest), 'load' (also shape and readability).
    ``task_ids``: the expected tasks when the directory has no run binding (default: the manifest, else every
    record). Never modifies anything."""
    from . import binding as BD
    run = BD.load(target)
    files = set(LB.task_ids(target))
    if run is not None:
        expected = list(run["manifest"])
    else:
        expected = sorted(task_ids) if task_ids is not None else sorted(files)
    imports = {}
    for imp in (run or {}).get("imports", []) or []:
        for t in imp.get("tasks", []):
            imports[t["task_id"]] = t
    completed, failed, corrupted, seconds = [], {}, [], 0.0
    digests = []
    for tid in sorted(set(expected) & files):
        kind, detail, rec = _check_one(target, tid, run, imports, check_arrays)
        if rec is not None:
            try:
                seconds += float(rec.get("seconds") or 0.0)
            except (TypeError, ValueError):
                pass
        if kind == "ok":
            completed.append(tid)
            digests.append([tid, _sha(os.path.join(target, "tasks", tid + ".json")),
                            (rec.get("arrays") or {}).get("sha256", "")])
        elif kind == "failed":
            failed.setdefault(detail, []).append(tid)
        else:
            corrupted.append(dict(task_id=tid, why=detail))
    missing = sorted(set(expected) - files)
    n_failed = sum(len(v) for v in failed.values())
    complete = bool(expected) and len(completed) == len(set(expected)) and not corrupted
    return dict(schema=SCHEMA, target=os.path.basename(os.path.normpath(target)), bound=run is not None,
                run_identity=(run or {}).get("identity_digest"), expected=len(set(expected)),
                completed=len(completed), failed={k: sorted(v) for k, v in sorted(failed.items())},
                n_failed=n_failed, missing=missing, corrupted=corrupted, unexpected=sorted(files - set(expected)),
                complete=complete, status="complete" if complete else ("empty" if not expected else "incomplete"),
                check_arrays=check_arrays, recorded_task_seconds=seconds,
                consumed=consumed_seconds(target) if os.path.isdir(target) else None,
                data_digest=_digest(digests) if digests else None, completed_ids=sorted(completed))


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def data_digest(target: str, task_ids=None) -> str | None:
    """Digest of the evaluated content of a labels directory: (task id, JSON sha256, array sha256 from the record) of
    every ok record among ``task_ids`` (default: every record). Array files are not re-read here; ``completion``
    checks them."""
    ids = sorted(task_ids) if task_ids is not None else LB.task_ids(target)
    rows = []
    for tid in ids:
        jp = os.path.join(target, "tasks", tid + ".json")
        if not os.path.exists(jp):
            continue
        r = LB.load_record(target, tid)
        if r.get("status") == "ok":
            rows.append([tid, _sha(jp), (r.get("arrays") or {}).get("sha256", "")])
    return _digest(rows) if rows else None


def describe(c: dict) -> str:
    """One line for people: what is complete and what is not, and how to proceed."""
    parts = ["%d/%d tasks labeled" % (c["completed"], c["expected"])]
    if c["n_failed"]:
        parts.append("%d failed (%s)" % (c["n_failed"], ", ".join("%s %d" % (k, len(v)) for k, v in c["failed"].items())))
    if c["missing"]:
        parts.append("%d not labeled yet" % len(c["missing"]))
    if c["corrupted"]:
        parts.append("%d corrupted" % len(c["corrupted"]))
    if c["unexpected"]:
        parts.append("%d unexpected files ignored" % len(c["unexpected"]))
    return "%s: %s" % (c["status"], "; ".join(parts))


def advice(c: dict, stopped=None) -> list:
    """What to do about an incomplete run (printed by the command line); ``stopped``: the monitored ceiling that ended
    the session, if any (a plain rerun would be refused by the same cap)."""
    out = []
    if stopped:
        out.append("the run stopped at a configured cap: raise limits.cpu_hours / disk_gb / min_free_disk_gb in the "
                   "configuration (operational keys: labels and a freeze stay valid), then run the same command "
                   "again (it resumes); --override-limits only admits a start, it does not lift the cap")
    if c["n_failed"]:
        out.append("failed tasks are kept as records and are not retried automatically: after raising the limit "
                   "(limits.task_seconds / memory_gb) or fixing the environment, run the same command with "
                   "--retry-failed (the scientific inputs are unchanged; the failed attempt stays in attempts.jsonl)")
    if c["missing"] and not stopped:
        out.append("tasks not labeled yet: run the same label command again (it resumes)")
    if c["corrupted"]:
        out.append("corrupted records are never overwritten silently: inspect them, or move them aside and relabel "
                   "with --quarantine-corrupted (they are kept under quarantine/ and listed in run.json)")
    return out


def require_complete(target: str, what: str = "this step") -> dict:
    """``completion`` of ``target``; raises ``CorruptedRun`` or ``IncompleteRun`` (with the reason) unless complete."""
    c = completion(target)
    if c["corrupted"]:
        raise CorruptedRun("%s: %s has %d corrupted record(s) (%s)" % (
            what, target, len(c["corrupted"]), "; ".join("%s: %s" % (x["task_id"], x["why"]) for x in c["corrupted"][:3])))
    if not c["complete"]:
        raise IncompleteRun("%s needs a complete labels directory; %s — %s" % (what, target, describe(c)))
    return c


def quarantine(target: str, task_ids, reason: str) -> str:
    """Move the records (and arrays) ``task_ids`` of ``target`` to ``target``/quarantine/<time>/ and list them in
    run.json (``quarantined``): an auditable removal, never a deletion. Returns the quarantine directory."""
    import shutil
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qdir = os.path.join(target, "quarantine", stamp)
    os.makedirs(qdir, exist_ok=True)
    moved = []
    for tid in task_ids:
        keep_in_ledger(target, tid, "quarantined")           # 1.9.1 (F3): its time stays consumed
        for ext in (".json", ".npz"):
            p = os.path.join(target, "tasks", tid + ext)
            if os.path.exists(p):
                shutil.move(p, os.path.join(qdir, tid + ext))
        moved.append(tid)
    rp = os.path.join(target, "run.json")
    if os.path.exists(rp):
        rec = TX.read_json(rp)
        dropped = []                                 # 1.9.0: a quarantined imported record is no longer an import;
        for imp in rec.get("imports", []) or []:     # its relabel belongs to this run (else it never converges)
            keep = [t for t in imp.get("tasks", []) if t["task_id"] not in set(moved)]
            dropped += [dict(t, source=imp.get("source")) for t in imp.get("tasks", []) if t["task_id"] in set(moved)]
            imp["tasks"] = keep
        rec.setdefault("quarantined", []).append(dict(at=stamp, reason=reason, tasks=moved,
                                                      directory=os.path.relpath(qdir, target).replace(os.sep, "/"),
                                                      removed_imports=dropped))
        TX.write_json(rp, rec, indent=1, default=str, atomic=True)
    return qdir


def append_attempt(target: str, entry: dict) -> None:
    """Append one task attempt (status, seconds, message, peak memory) to ``target``/attempts.jsonl: the audit trail
    of every attempt, kept when a failed task is retried."""
    p = os.path.join(target, "attempts.jsonl")
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str, ensure_ascii=False) + "\n")


def keep_in_ledger(target: str, task_id: str, why: str) -> None:
    """Before a current record is replaced (``--retry-failed``) or moved aside (quarantine), make sure its seconds stay
    in the consumed-resource ledger (1.9.1; independent review F3): a record labeled before attempts were logged
    (pre-1.9.0) or imported from another run has no attempt entry that wrote a record, so one is appended for it; a
    record that has one needs nothing (no double counting; ``consumed_seconds``)."""
    if any(x.get("task_id") == task_id and (str(x.get("status")) not in _RECORDLESS or x.get("record_written"))
           for x in attempts(target)):
        return
    p = os.path.join(target, "tasks", task_id + ".json")
    if not os.path.exists(p):
        return
    try:
        rec = TX.read_json(p)
        sec = float(rec.get("seconds") or 0.0)
    except Exception:                                        # noqa: BLE001 - an unreadable record has no time
        sec = 0.0
        rec = {}
    append_attempt(target, dict(task_id=task_id, status="ledger-record", record_status=rec.get("status"),
                                seconds=round(sec, 3), message="seconds of a record without an attempt entry, kept "
                                "before it was %s" % why, session="ledger"))


_RECORDLESS = ("interrupted", "skipped")                   # attempt statuses that write no new record


def _records_on_disk(target: str) -> dict:
    """task id -> [paths] of every record of the labels directory still on disk: the current record (tasks/), copies
    of records replaced by a retry (failed_attempts/) and quarantined records (quarantine/*/). Listing only."""
    out = {}
    for tid in LB.task_ids(target):
        out.setdefault(tid, []).append(os.path.join(target, "tasks", tid + ".json"))
    fdir = os.path.join(target, "failed_attempts")
    if os.path.isdir(fdir):
        for f in sorted(os.listdir(fdir)):
            m = re.match(r"(.+)\.attempt\d+\.json$", f)
            if m:
                out.setdefault(m.group(1), []).append(os.path.join(fdir, f))
    qroot = os.path.join(target, "quarantine")
    if os.path.isdir(qroot):
        for d in sorted(os.listdir(qroot)):
            qd = os.path.join(qroot, d)
            if os.path.isdir(qd):
                for f in sorted(os.listdir(qd)):
                    if f.endswith(".json") and ".tmp" not in f:
                        out.setdefault(f[:-5], []).append(os.path.join(qd, f))
    return out


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:                                          # moved meanwhile by another process: counted as oldest
        return 0.0


def _distinct(paths: list) -> list:
    """``paths`` without byte-identical duplicates: ``--retry-failed`` copies a failed record to failed_attempts/
    before relabeling, so until the retry writes a new record (or when it never does: a ceiling, an interruption)
    the copy and the record are one attempt, not two (independent debugging of 1.9.1)."""
    seen, out = set(), []
    for q in paths:
        try:
            d = _sha(q)
        except OSError:
            continue
        if d not in seen:
            seen.add(d)
            out.append(q)
    return out


def _record_seconds(path: str) -> float:
    try:
        return float(TX.read_json(path).get("seconds") or 0.0)
    except Exception:                                        # noqa: BLE001 - an unreadable record has no time
        return 0.0


def consumed_seconds(target: str, task_ids=None) -> dict:
    """The consumed-resource ledger of a labels directory (1.9.1; independent review F3). Monotone across
    invocations: nothing that was consumed leaves it when a record is replaced by a retry, quarantined or interrupted.

    Rules (no double counting):

    1. every attempt in ``attempts.jsonl`` counts with its seconds — successful, failed, timed out, replaced by a
       retry, ``interrupted`` (Ctrl+C/SIGTERM; 1.9.1) and ``ledger-record`` entries (``keep_in_ledger``);
    2. a record on disk (current, ``failed_attempts/`` copy or quarantined) that has no attempt entry — labeled before
       attempts were logged (before 1.9.0) or imported from another run — counts with its own recorded seconds; per
       task, the records that write a record (every status except interrupted/skipped, or an interrupted attempt whose
       record was written) are matched to records oldest-first, and only the records left over count (migration);
    3. imported records count like labeled ones: their labeling was spent for this study, on whichever computer.

    The unit is summed worker wall-clock seconds as measured by the supervisor (one numerical thread per worker): an
    approximation of CPU time, not a CPU meter. A task running when its supervisor was killed outright (SIGKILL,
    power loss) is not in it."""
    want = None if task_ids is None else set(task_ids)
    atts = [x for x in attempts(target) if "task_id" in x and (want is None or x["task_id"] in want)]
    by_status, writes = {}, {}
    for x in atts:
        st = str(x.get("status"))
        by_status[st] = by_status.get(st, 0.0) + float(x.get("seconds") or 0.0)
        if st not in _RECORDLESS or x.get("record_written"):
            writes[x["task_id"]] = writes.get(x["task_id"], 0) + 1
    legacy, legacy_records = 0.0, 0
    for tid, paths in _records_on_disk(target).items():
        if len(paths) - writes.get(tid, 0) <= 0 or (want is not None and tid not in want):
            continue                                         # only records without attempts are read
        paths = _distinct(paths)                             # a retry's copy is the same record until replaced
        n_extra = len(paths) - writes.get(tid, 0)
        if n_extra <= 0:
            continue
        extra = sorted(paths, key=_mtime)[:n_extra]          # the oldest records are the unlogged ones
        legacy += sum(_record_seconds(q) for q in extra)
        legacy_records += len(extra)
    att = sum(by_status.values())
    return dict(total_seconds=att + legacy, attempt_seconds=att, attempts=len(atts), by_status=by_status,
                records_without_attempts=legacy_records, record_seconds_without_attempts=legacy,
                unit="summed worker wall-clock seconds (supervisor-measured; approximates CPU time with one thread "
                     "per worker)")


def attempts(target: str) -> list:
    p = os.path.join(target, "attempts.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append(dict(unreadable=line[:200]))
    return out


# ------------------------------------------------------------------ single writer per run directory
class RunLocked(RuntimeError):
    """Another process is writing to the same run directory."""


class RunLock:
    """An exclusive lock file (``<dir>/.qemx-run.lock``) held while a command writes into a run directory: two
    processes labeling or analyzing the same directory at once would interleave writes. A lock whose process no longer
    runs on this computer is stale and is replaced (reported); a lock of a live process refuses the second writer."""

    def __init__(self, directory: str, purpose: str):
        self.dir, self.purpose = directory, purpose
        self.path = os.path.join(directory, ".qemx-run.lock")
        self.stale = None

    def __enter__(self):
        import socket
        import time
        os.makedirs(self.dir, exist_ok=True)
        me = dict(pid=os.getpid(), host=socket.gethostname(), purpose=self.purpose,
                  since=time.strftime("%Y-%m-%dT%H:%M:%S"))
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    other = TX.read_json(self.path)
                except Exception:                            # noqa: BLE001 - a half-written lock
                    other = {}
                if other.get("host") == me["host"] and other.get("pid") and not _alive(int(other["pid"])):
                    self.stale = other
                    try:
                        os.remove(self.path)
                    except FileNotFoundError:
                        pass
                    continue
                raise RunLocked("%s is being written by another process (%s, pid %s on %s since %s); wait for it "
                                "to finish. If no such process exists, delete %s." % (
                                    self.dir, other.get("purpose", "?"), other.get("pid", "?"), other.get("host", "?"),
                                    other.get("since", "?"), self.path)) from None
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(me, f)
            return self
        raise RunLocked("could not take the lock %s" % self.path)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        return False


def _alive(pid: int) -> bool:
    """Whether the process that wrote a lock still runs (a zombie — ended, not yet reaped — counts as ended).

    1.9.1 (independent review F5): on POSIX the PID is checked in this process's own PID view (``os.kill(pid, 0)``,
    the view in which the lock's ``os.getpid()`` was recorded); ``psutil`` is consulted only for the zombie state and
    only when its view is consistent (``supervise.process_view_consistent``). 1.9.0 asked ``psutil`` alone, which in a
    PID namespace could declare a live writer's lock stale. On Windows (no PID namespaces; ``os.kill`` would terminate
    the process) ``psutil`` is used."""
    if os.name == "posix":
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:                              # exists, owned by another user
            return True
        except OSError:
            return True
        from . import supervise as SV
        if SV.process_view_consistent():
            try:
                import psutil
                return psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
            except Exception:                                # noqa: BLE001 - when unsure, the lock is held
                return True
        return True
    try:
        import psutil
    except ImportError:                                      # pragma: no cover - psutil is a declared dependency
        return True
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False