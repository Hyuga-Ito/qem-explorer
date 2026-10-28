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
    view = process_view_consistent()
    watchdog = watchdog and view
    return dict(platform=os.name, timeout="parent-enforced wall clock (all platforms)",
                threads_per_worker={k: os.environ.get(k, "1 (set at start)") for k in THREAD_VARIABLES},
                parent_watch="multiprocessing parent sentinel (no PID lookup)",
                process_view=("consistent" if view else
                              "INCONSISTENT: psutil does not see this process as the operating system reports it "
                              "(e.g. a PID namespace); the resident-memory watchdog and process-tree stopping are "
                              "off, the time limit and the parent sentinel are unaffected"),
                memory_watchdog=watchdog, child_address_space_limit=rlimit,
                memory="%s%s" % ("resident-memory watchdog (psutil)" if watchdog else "no resident-memory watchdog",
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


def process_view_consistent() -> bool:
    """Whether ``psutil`` sees this process as the operating system reports it to us (1.9.1; independent review F5).

    In some sandboxes ``os.getpid()``/``os.getppid()`` are PIDs of a private PID namespace while ``psutil`` reads a
    ``/proc`` of another namespace: the same number can then name no process or an unrelated one. Process inspection
    (memory watchdog, process-tree stopping, zombie checks) is used only when this check passes; otherwise the
    supervisor relies on mechanisms that do not depend on PID views (the parent sentinel, ``Process.terminate/kill``,
    ``RLIMIT_AS``) and says so in ``support()``."""
    global _VIEW
    if _VIEW is None:
        try:
            import psutil
            me = psutil.Process(os.getpid())
            ok = True
            if sys.platform.startswith("linux"):     # PID namespaces: /proc/self names this process in the PID view
                ok = int(os.readlink("/proc/self")) == os.getpid()      # of /proc, which psutil reads
            if ok and os.name == "posix":            # (macOS: no namespaces; its framework builds run the
                ok = me.ppid() == os.getppid()       # interpreter under another path than sys.executable, so no
            _VIEW = bool(ok)                         # path comparison; Windows: one PID view)
        except Exception:                            # noqa: BLE001 - not visible, or access denied
            _VIEW = False
    return _VIEW


_VIEW = None


def _watch_parent(ppid: int = 0) -> None:   # pragma: no cover - runs in the child
    """A child whose supervising parent ended (SIGKILL, a closed console) ends itself immediately instead of running on
    and writing a record nobody supervises (1.9.0). 1.9.1 (independent review F5): liveness comes from the parent
    *sentinel* that ``multiprocessing`` gives every child — a pipe whose write end only the parent holds (POSIX) or a
    handle on the parent process (Windows) — which becomes ready exactly when the parent ends. It does not compare PID
    numbers, so it is correct in PID namespaces, where 1.9.0's ``getppid``/``psutil`` check killed live workers."""
    import multiprocessing as mp
    parent = mp.parent_process()
    if parent is not None:
        def wait_for_parent():
            parent.join()                            # blocks until the sentinel is ready: the parent has ended
            os._exit(70)
        threading.Thread(target=wait_for_parent, daemon=True, name="qemx-parent-watch").start()
        return
    if os.name == "posix" and ppid:                  # not started by multiprocessing: re-parenting, no PID lookup
        def poll():
            while True:
                time.sleep(1.0)
                if os.getppid() != ppid:
                    os._exit(70)
        threading.Thread(target=poll, daemon=True, name="qemx-parent-watch").start()


def _child_main(target: str, args: tuple, memory_bytes: int, conn, ppid: int = 0) -> None:   # pragma: no cover
    _watch_parent(int(ppid or 0))
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
    if not process_view_consistent():                # another PID view: the number may name an unrelated process
        return 0
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
    if process_view_consistent():                    # descendants first (only when the PID view is our own)
        try:
            import psutil
            for c in psutil.Process(proc.pid).children(recursive=True):
                try:
                    c.kill()
                except psutil.Error:
                    pass
        except Exception:                            # noqa: BLE001
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


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def child_environment() -> dict:
    """1.9.0: one numerical thread per child unless the user set these variables: several workers with all-core BLAS
    each oversubscribe the computer, and reductions can depend on the thread count. Applied to ``os.environ`` (inherited
    by every child at start); returns what was set."""
    out = {}
    for k in THREAD_VARIABLES:
        if k not in os.environ:
            os.environ[k] = "1"
            out[k] = "1"
    return out


def run_jobs(jobs, workers: int = 1, on_done=None, admit=None, on_interrupt=None) -> list:
    """Run ``jobs`` (``Job``) in supervised child processes, at most ``workers`` at a time; returns the
    ``Outcome`` of every job (completion order). ``on_done(outcome)`` is called as each finishes.

    ``admit(running_seconds)`` (1.9.0): asked before each job starts with the summed elapsed seconds of the jobs
    running now; when it returns False no further job starts (the running ones finish and are reported; the jobs never
    started are not in the result) — a checkpoint-safe stop for whole-run ceilings (``resources.Ceilings``).
    ``on_interrupt(key, seconds)`` (1.9.1): called for every job still running when the parent is interrupted (Ctrl+C,
    SIGTERM, an error), after the job was stopped, with its elapsed seconds (resource accounting)."""
    import multiprocessing as mp
    from multiprocessing.connection import wait
    ctx = mp.get_context("spawn")
    child_environment()
    pending = list(jobs)
    running = {}
    done = []
    workers = max(1, int(workers))
    try:
        return _loop(ctx, pending, running, done, workers, on_done, admit)
    finally:                                         # 1.9.0: an interrupted parent stops its children
        for st in list(running.values()):
            try:
                if st["proc"].is_alive():
                    _stop(st["proc"])
            except Exception:                        # noqa: BLE001
                pass
            if on_interrupt is not None:
                try:
                    on_interrupt(st["job"].key, time.monotonic() - st["t0"])
                except Exception:                    # noqa: BLE001 - accounting must not mask the interruption
                    pass


def _loop(ctx, pending, running, done, workers, on_done, admit) -> list:
    from multiprocessing.connection import wait
    while pending or running:
        while pending and len(running) < workers:
            if admit is not None and not admit(sum(time.monotonic() - st["t0"] for st in running.values())):
                pending = []
                break
            job = pending.pop(0)
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child_main, args=(job.target, tuple(job.args), int(job.memory_bytes or 0), w,
                                                      os.getpid()), daemon=True)
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
