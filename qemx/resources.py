"""Resources: memory estimates, cooperative time limits and readable error messages.

There are deliberately **no hard limits** on qubits or depth. Instead the app

* estimates what a run will need (a density matrix of n qubits is 16·4ⁿ bytes) and warns before it starts,
* lets the executors stop cleanly when a user-chosen time limit passes (``deadline``), and
* turns MemoryError, Aer memory failures, timeouts and other exceptions into messages a person can act on
  (``explain``) instead of a crash.

An operator deploying the app publicly can still set soft defaults through environment variables
(``QEMX_TIME_LIMIT`` seconds, ``QEMX_WARN_QUBITS``); users can change them in the UI.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import threading
import time
from dataclasses import dataclass

BYTES_PER_AMPLITUDE = 16                     # complex128


class RunTimeout(RuntimeError):
    """The run passed its time limit and was stopped between two operations."""


_deadline: contextvars.ContextVar = contextvars.ContextVar("qemx_deadline", default=None)

# ------------------------------------------------------------------ one Qiskit call at a time
# Streamlit runs every browser session (and, with fast reruns, an interrupted run and its successor) in its own
# thread. Qiskit's compiled extensions (transpiler passes, synthesis, Aer) are not designed to be entered from several
# Python threads at once; a crash in native code cannot be caught by try/except and would take the whole server down.
# Every entry point into Qiskit therefore holds this process-wide re-entrant lock. NumPy simulation stays parallel.
QISKIT_LOCK = threading.RLock()


def qiskit_serial(fn):
    """Decorator: run `fn` while holding QISKIT_LOCK (re-entrant, so decorated functions may call each other)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with QISKIT_LOCK:
            return fn(*args, **kwargs)
    return wrapper


@contextlib.contextmanager
def deadline(seconds: float | None):
    """Within this block, ``check()`` raises RunTimeout once `seconds` have passed (None or 0 = no limit)."""
    token = _deadline.set(time.monotonic() + float(seconds) if seconds and seconds > 0 else None)
    try:
        yield
    finally:
        _deadline.reset(token)


def check():
    """Called by the executors between operations."""
    d = _deadline.get()
    if d is not None and time.monotonic() > d:
        raise RunTimeout("the run passed its time limit")


def default_time_limit() -> float:
    try:
        return float(os.environ.get("QEMX_TIME_LIMIT", "900"))
    except ValueError:
        return 900.0


# ------------------------------------------------------------------ memory
def dm_bytes(n: int) -> int:
    return BYTES_PER_AMPLITUDE * 4 ** int(n)


def available_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:                                   # pragma: no cover - psutil missing
        return 8 * 2 ** 30


def human(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(b) < 1024 or unit == "PB":
            return ("%.0f %s" if unit == "B" else "%.1f %s") % (b, unit)
        b /= 1024
    return "%.1f PB" % b


@dataclass
class Estimate:
    n: int
    peak: int                 # bytes needed at the same time
    stored: int               # bytes kept after the run (caches, snapshots)
    available: int
    note: str = ""

    @property
    def ratio(self) -> float:
        return (self.peak + self.stored) / max(self.available, 1)

    @property
    def level(self) -> str:
        """ok | heavy | risky: risky = more memory than the machine has free right now."""
        r = self.ratio
        return "ok" if r < 0.25 else ("heavy" if r < 0.9 else "risky")

    def text(self) -> str:
        return "needs about %s at peak and keeps %s (free memory now: %s)" % (
            human(self.peak), human(self.stored), human(self.available))


def timeline_should_be_compact(n: int, steps: int) -> bool:
    """Keep every density matrix only when that takes a small share of free memory."""
    full = 2 * dm_bytes(n) * (steps + 1)
    return full > min(0.2 * available_bytes(), 1.5 * 2 ** 30)


def keep_states(n: int) -> bool:
    """Whether per-circuit final density matrices are cached (virtual distillation needs them)."""
    return dm_bytes(n) <= min(0.02 * available_bytes(), 64 * 2 ** 20)


# ------------------------------------------------------------------ errors
def _is_memory_error(e: BaseException) -> bool:
    if isinstance(e, MemoryError):
        return True
    t = (type(e).__name__ + " " + str(e)).lower()
    return any(k in t for k in ("insufficient memory", "out of memory", "bad_alloc", "cannot allocate",
                                "unable to allocate", "memoryerror", "arraymemoryerror", "required memory"))


def explain(e: BaseException, what: str = "the run", n: int | None = None) -> tuple:
    """(title, message) for an exception, written for the person using the app."""
    size = "" if n is None else " A density matrix of %d qubits takes %s." % (n, human(dm_bytes(n)))
    if _is_memory_error(e):
        return ("Not enough memory",
                "%s needed more memory than this machine could give.%s Try fewer qubits, fewer contenders or "
                "folds, the NumPy backend with fewer field samples, or run the exported script on a bigger "
                "machine. Nothing else was changed." % (what.capitalize(), size))
    if isinstance(e, RunTimeout):
        return ("Stopped at the time limit",
                "%s passed its time limit and was stopped cleanly. Raise the limit in the sidebar "
                "(0 = no limit) or make the task smaller.%s" % (what.capitalize(), size))
    if isinstance(e, RecursionError):
        return ("Input too deeply nested", "An expression or file was nested too deeply to read safely.")
    try:
        from .backends.base import NotSupported
        if isinstance(e, NotSupported):
            return ("Not supported by this backend", str(e))
    except Exception:                                    # pragma: no cover
        pass
    if isinstance(e, (ValueError, KeyError, TypeError)):
        return ("Invalid input", "%s could not use the input: %s" % (what.capitalize(), _short(e)))
    return ("Something went wrong", "%s failed with %s: %s" % (what.capitalize(), type(e).__name__, _short(e)))


def _short(e: BaseException, limit: int = 400) -> str:
    s = str(e).strip() or type(e).__name__
    return s if len(s) <= limit else s[:limit] + "…"


def scrub(text: str, *secrets: str) -> str:
    """Remove secrets (e.g. an API token) from a message before it is shown or logged."""
    for s in secrets:
        if s and len(s) >= 4:
            text = text.replace(s, "•••")
    return text
