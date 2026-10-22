"""Supervised execution of labeling tasks: every task in its own child process (1.7.0; review finding S1).

1.6.0 enforced the per-task time limit with ``signal.SIGALRM`` (missing on Windows, so a configuration with
``task_seconds`` failed there with ``AttributeError``) and set the address-space limit with ``resource.setrlimit``
in whichever process ran the task — with one worker, the parent itself, which kept the limit for the analysis that
followed. A signal also cannot interrupt a long native computation.

Here the *parent* supervises; nothing is ever imposed on it:

* every task runs in a fresh child process (``multiprocessing`` "spawn" context, the same on every platform), with
  one worker as with many;
* time: the parent stops a child that exceeds its wall-clock limit (terminate, then kill) and records ``timeout``;
* memory: a resident-memory watchdog in the parent (``psutil``: the child and its descendants, every
  ``POLL_SECONDS``) stops a child above the limit and records ``memory``; on POSIX the child additionally limits its
  own address space (``RLIMIT_AS``), so most overruns surface as a ``MemoryError`` inside the child, recorded the same
  way. ``support()`` states which mechanisms are active on this platform;
* a child that dies without reporting (killed by the operating system, a crash in native code) is recorded as
  ``crashed`` with its exit code.

The child runs a function named by ``"module:function"`` with picklable arguments and sends back its (small,
picklable) return value. Results arrive in completion order; ``on_done`` sees each as it arrives.

The child never re-runs the parent's ``__main__``: "spawn" normally re-imports the parent's main script in every
child, which is wrong when that script is a Streamlit page, a test harness's generated script or any script with side
effects (1.7.0: found when the supervisor ran after a Streamlit ``AppTest`` in the same process — the child re-ran the
generated script and failed). The child needs only this module and the job's importable target.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
import threading
import time
import traceback
import types
from dataclasses import dataclass, field

POLL_SECONDS = 0.5
KILL_GRACE_SECONDS = 5.0


def support() -> dict:
    """The supervision mechanisms available on this platform (recorded with every labeling session)."""
    try:
        import psutil  # noqa: F401
        watchdog = True
    except ImportError:                              # pragma: no cover - psutil is a declared dependency
        watchdog = False
    rlimit = False
    if os.name == "posix":
        try:
            import resource  # noqa: F401
            rlimit = hasattr(resource, "RLIMIT_AS")
        except ImportError:                          # pragma: no cover
            rlimit = False
    return dict(platform=os.name, timeout="parent-enforced wall clock (all platforms)",
                memory_watchdog=watchdog, child_address_space_limit=rlimit,
                memory="%s%s" % ("resident-memory watchdog (psutil)" if watchdog else "no memory limit",
                                 " + child address-space limit (RLIMIT_AS)" if rlimit else ""))


@dataclass
class Job:
    key: str                                   # identifies the job in results (e.g. the task id)
    target: str                                # "package.module:function" run in the child
    args: tuple = ()
    seconds: float = 0.0                       # wall-clock limit (0: none)
    memory_bytes: int = 0                      # resident-memory limit (0: none)


@dataclass
class Outcome:
    key: str
    status: str                                # ok | error | timeout | memory | crashed
    value: object = None                       # the function's return value (status ok)
    message: str = ""
    seconds: float = 0.0
    peak_rss_bytes: int = 0
    exitcode: int | None = None
    extra: dict = field(default_factory=dict)


def _child_main(target: str, args: tuple, memory_bytes: int, conn) -> None:   # pragma: no cover - runs in the child
    if memory_bytes and os.name == "posix":
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (int(memory_bytes), int(memory_bytes)))
        except (ImportError, ValueError, OSError):
            pass
    try:
        mod, name = target.split(":", 1)
        fn = getattr(importlib.import_module(mod), name)
        conn.send(("ok", fn(*args), ""))
    except MemoryError as e:
        conn.send(("memory", None, "MemoryError: %s" % e))
    except BaseException as e:                         # noqa: BLE001 - every failure is reported to the parent
        conn.send(("error", None, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:])))
    finally:
        conn.close()


def _tree_rss(pid: int) -> int:
    try:
        import psutil
        p = psutil.Process(pid)
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return int(total)
    except Exception:                                # noqa: BLE001 - the process may have just ended
        return 0


def _stop(proc) -> None:
    try:
        import psutil
        for c in psutil.Process(proc.pid).children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
    except Exception:                                # noqa: BLE001
        pass
    proc.terminate()
    proc.join(KILL_GRACE_SECONDS)
    if proc.is_alive():
        proc.kill()
        proc.join(KILL_GRACE_SECONDS)


_MAIN_LOCK = threading.Lock()


@contextlib.contextmanager
def _isolated_main():
    """While a child is being started, ``__main__`` is a bare module (no file, no spec), so the "spawn" child does
    not import the parent's main script; the parent's module is restored immediately afterwards."""
    with _MAIN_LOCK:
        saved = sys.modules.get("__main__")
        stub = types.ModuleType("__main__")
        stub.__spec__ = None
        sys.modules["__main__"] = stub
        try:
            yield
        finally:
            if saved is not None:
                sys.modules["__main__"] = saved
            else:
                sys.modules.pop("__main__", None)


def run_jobs(jobs, workers: int = 1, on_done=None) -> list:
    """Run ``jobs`` (``Job``) in supervised child processes, at most ``workers`` at a time; returns the
    ``Outcome`` of every job (completion order). ``on_done(outcome)`` is called as each finishes."""
    import multiprocessing as mp
    from multiprocessing.connection import wait
    ctx = mp.get_context("spawn")
    pending = list(jobs)
    running = {}
    done = []
    workers = max(1, int(workers))
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child_main, args=(job.target, tuple(job.args), int(job.memory_bytes or 0), w),
                            daemon=True)
            with _isolated_main():
                p.start()
            w.close()
            running[p.sentinel] = dict(job=job, proc=p, conn=r, t0=time.monotonic(), peak=0, msg=None)
        ready = set(wait(list(running) + [st["conn"] for st in running.values()], timeout=POLL_SECONDS))
        now = time.monotonic()
        for sentinel in list(running):
            st = running[sentinel]
            job, p, conn = st["job"], st["proc"], st["conn"]
            if st["msg"] is None and (conn in ready or conn.poll()):
                try:
                    st["msg"] = conn.recv()
                except (EOFError, OSError):
                    st["msg"] = ("crashed", None, "the child closed its pipe without a result")
            finished = st["msg"] is not None or sentinel in ready or not p.is_alive()
            out = None
            if finished:
                p.join(KILL_GRACE_SECONDS)
                if p.is_alive():
                    _stop(p)
                msg = st["msg"]
                if msg is None:
                    code = p.exitcode
                    msg = ("crashed", None, "the child process ended without a result (exit code %s%s)"
                           % (code, "; killed by the operating system, possibly out of memory" if code in (-9, 137)
                              else ""))
                out = Outcome(job.key, msg[0], msg[1], msg[2], now - st["t0"], st["peak"], p.exitcode)
            else:
                rss = _tree_rss(p.pid)
                st["peak"] = max(st["peak"], rss)
                if job.seconds and now - st["t0"] > float(job.seconds):
                    _stop(p)
                    out = Outcome(job.key, "timeout", None, "the task exceeded its limit of %g s (stopped by the "
                                  "supervisor)" % float(job.seconds), now - st["t0"], st["peak"], p.exitcode)
                elif job.memory_bytes and rss > int(job.memory_bytes):
                    _stop(p)
                    out = Outcome(job.key, "memory", None, "the task's resident memory %.2f GB exceeded its limit of "
                                  "%.2f GB (stopped by the supervisor)" % (rss / 2 ** 30, job.memory_bytes / 2 ** 30),
                                  now - st["t0"], st["peak"], p.exitcode)
            if out is not None:
                conn.close()
                del running[sentinel]
                done.append(out)
                if on_done is not None:
                    on_done(out)
    return done
