"""Run a module with PEP 597 EncodingWarnings turned into errors for *project* code only (1.8.0).

``python -X warn_default_encoding -W error::EncodingWarning`` makes every text ``open()`` without an encoding an
error, including those inside third-party packages (for example ``dill``, imported by Qiskit, opens ``os.devnull``
without one at import time) that QEM Explorer neither owns nor relies on for its files. The ``-W`` command-line form
only matches exact module names, so it cannot say "the qemx package". This wrapper installs the filter with a module
pattern instead: an EncodingWarning attributed to project code (``qemx``, ``ui``, ``pages``, ``app``, ``scripts``,
``tests``) raises; one attributed to a third-party module is still *shown* (never silenced), as Python's default.

Usage (the CI's text-encoding job; any platform)::

    python -X warn_default_encoding scripts/strict_encoding.py -m qemx.selection plan configs/selection/smoke.yaml
    python -X warn_default_encoding scripts/strict_encoding.py -m pytest -q tests/test_selection_core.py
"""
from __future__ import annotations

import os
import runpy
import sys
import warnings

PROJECT = r"(qemx|ui|pages|app|scripts|tests|test_[A-Za-z0-9_]+|conftest)(\..*)?\Z"


def install() -> None:
    if not sys.flags.warn_default_encoding:
        raise SystemExit("run with python -X warn_default_encoding (otherwise no EncodingWarning is ever emitted)")
    warnings.filterwarnings("default", category=EncodingWarning)
    warnings.filterwarnings("error", category=EncodingWarning, module=PROJECT)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "-m":
        print(__doc__)
        return 2
    install()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.argv = [argv[1]] + argv[2:]
    try:
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
