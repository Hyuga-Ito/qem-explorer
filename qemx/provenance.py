"""Provenance of a run: which source, dependencies and environment produced it (1.5.1, review 6.6).

A configuration hash identifies *what* was asked for; it does not identify the code that answered. The same YAML
can give other numbers after a change of the source, a dependency, a fake-backend snapshot or the compiler. Every
result document, phase-study record and run file therefore carries:

* ``source_sha256`` — a digest of the application source as it is on disk (``qemx/``, ``ui/``, ``pages/``,
  ``app.py``, ``scripts/``, ``manifests/``, the requirement files and ``pyproject.toml``): path names and bytes,
  in sorted order. It works on an unpacked ZIP without git, and it changes when any of those files changes;
* ``lock_sha256`` — the digest of ``requirements-lock.txt`` (the tested pins), when the file is present;
* ``git_commit`` — when the source is a git checkout (None otherwise; the digest is the identity that always
  exists);
* the versions of the scientifically relevant packages, and ``inventory_sha256``, a digest of every installed
  distribution ``name==version`` (the full list with ``full=True``).

A digest identifies a snapshot; it cannot reconstruct one that was not kept. Calibration snapshots, compiled
circuits and raw counts are stored by the code paths that produce them (phase records: ``calibration_hash``,
``circuit``; hardware: counts and job ids).
"""
from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIRS = ("qemx", "ui", "pages", "scripts", "manifests")
SOURCE_FILES = ("app.py", "pyproject.toml", "requirements.txt", "requirements-lock.txt", "requirements-dev.txt")
SOURCE_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".txt", ".toml", ".md")
KEY_PACKAGES = ("numpy", "scipy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "pandas", "scikit-learn",
                "streamlit", "plotly")


def _source_files(root: str) -> list:
    out = []
    for d in SOURCE_DIRS:
        base = os.path.join(root, d)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(x for x in dirnames if x != "__pycache__" and not x.startswith("."))
            for f in sorted(filenames):
                if f.endswith(SOURCE_SUFFIXES):
                    out.append(os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/"))
    out += [f for f in SOURCE_FILES if os.path.isfile(os.path.join(root, f))]
    return sorted(set(out))


@functools.lru_cache(maxsize=4)
def source_digest(root: str = ROOT) -> dict:
    """{"sha256", "files"}: the digest of the source tree at ``root`` (cached per process: the source of a running
    process does not change under it in any supported workflow)."""
    h = hashlib.sha256()
    files = _source_files(root)
    for rel in files:
        h.update(rel.encode("utf-8") + b"\0")
        with open(os.path.join(root, rel), "rb") as f:
            data = f.read()
        if rel.endswith(SOURCE_SUFFIXES):                 # line endings must not change the identity (Windows)
            data = data.replace(b"\r\n", b"\n")
        h.update(hashlib.sha256(data).digest())
    return dict(sha256=h.hexdigest(), files=len(files))


def file_digest(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None


def _git_commit(root: str) -> str | None:
    if not os.path.isdir(os.path.join(root, ".git")) and not os.path.isfile(os.path.join(root, ".git")):
        return None
    try:
        import subprocess
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, encoding="utf-8")
        return out.stdout.strip() or None
    except Exception:                                    # noqa: BLE001 - git is optional
        return None


def _git_dirty(root: str) -> bool | None:
    try:
        import subprocess
        out = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, timeout=5, encoding="utf-8")
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:                                    # noqa: BLE001
        return None


@functools.lru_cache(maxsize=1)
def _inventory() -> tuple:
    from importlib import metadata
    seen = {}
    for d in metadata.distributions():
        try:
            name = (d.metadata["Name"] or "").strip()
        except Exception:                                # noqa: BLE001 - broken metadata
            continue
        if name:
            seen[name.lower()] = d.version
    return tuple(sorted("%s==%s" % kv for kv in seen.items()))


def package_versions(names=KEY_PACKAGES) -> dict:
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:                                # noqa: BLE001 - not installed
            out[n] = None
    return out


def redact_path(x):
    """A path (or a list of them) with the user's home folder replaced by "~", so run files can be shared without
    revealing the local user name (non-path strings pass through unchanged)."""
    if isinstance(x, (list, tuple)):
        return [redact_path(v) for v in x]
    if not isinstance(x, str):
        return x
    home = os.path.expanduser("~")
    out = x
    if home and home not in ("/", "~"):
        for h in {home, home.replace("\\", "/")}:
            out = out.replace(h, "~")
    return out


def provenance(full: bool = False, root: str = ROOT) -> dict:
    """The provenance block of a result (see the module text). ``full`` adds the complete package inventory."""
    from . import __version__
    inv = _inventory()
    src = source_digest(root)
    commit = _git_commit(root)
    out = dict(app_version=__version__, source_sha256=src["sha256"], source_files=src["files"],
               lock_sha256=file_digest(os.path.join(root, "requirements-lock.txt")),
               git_commit=commit, git_dirty=_git_dirty(root) if commit else None,
               python=platform.python_version(), implementation=sys.implementation.name,
               platform=platform.platform(), packages=package_versions(),
               inventory_sha256=hashlib.sha256("\n".join(inv).encode()).hexdigest(),
               created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    if full:
        out["inventory"] = list(inv)
    return out
