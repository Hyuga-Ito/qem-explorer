#!/usr/bin/env python3
r"""Input and output of the Direction A analysis scripts, kept apart from their computations.

The analysis scripts (``dirA_checks.py``, ``dirA_figures.py``, ``dirA_nil.py``) compute on in-memory tables. Where the
tables come from and where results go are separate objects:

* **sources** (input) — :class:`DirectorySource` reads a run directory written by
  ``python -m qemx.phase.run run/collect`` (``tables/*.csv|parquet``, ``cells/*.json``, ``run.json``,
  ``manifest.yaml``); :class:`MemorySource` serves the same from dictionaries (tests, notebooks, other pipelines);
* **reports** (text output) — computations append sections, verdicts, text and tables to a :class:`Report` instead of
  printing; :meth:`Report.render_text` and :meth:`Report.render_markdown` turn it into text, and a
  :class:`DirectoryReportSink` writes every table as CSV plus ``summary.csv`` and ``report.md``;
* **figure sinks** (graphical output) — :class:`DirectoryFigureSink` saves figures (PDF + PNG by default);
  :class:`MemoryFigureSink` keeps them (tests).

Nothing here computes a result; nothing in the computations opens a file.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import pandas as pd

TABLES = ("methods", "levels", "validation", "ledger", "winners", "errors")


def _find_qemx() -> bool:
    """Make qemx importable from a source checkout: when ``import qemx`` fails and the parent of this scripts folder
    holds the package, that folder is *appended* to sys.path (an installed qemx still wins). qemx stays optional:
    the scripts compute what they can without it (e.g. B*(ε) by interpolation instead of the exact ledger)."""
    import importlib.util
    import sys
    if importlib.util.find_spec("qemx") is not None:
        return True
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(root, "qemx", "__init__.py")) and root not in sys.path:
        sys.path.append(root)
    return importlib.util.find_spec("qemx") is not None


HAVE_QEMX = _find_qemx()


# ------------------------------------------------------------------ sources (input)
class RunSource:
    """What an analysis may read from a run. Subclasses implement the four accessors."""

    def table(self, name: str, required: bool = True, columns=None) -> pd.DataFrame:   # pragma: no cover
        raise NotImplementedError

    def records(self, ids=None):                                                       # pragma: no cover
        """The per-cell records (dicts), optionally only those of ``ids``."""
        raise NotImplementedError

    def run_info(self) -> dict:                                                        # pragma: no cover
        raise NotImplementedError

    def expected_cells(self) -> int | None:                                            # pragma: no cover
        """Cells in the run's manifest (None when unknown)."""
        raise NotImplementedError


class DirectorySource(RunSource):
    """A run directory: ``tables/`` (from collect), ``cells/``, ``run.json``, ``manifest.yaml``."""

    def __init__(self, run_dir: str):
        self.dir = run_dir
        self.tdir = os.path.join(run_dir, "tables")

    def table(self, name, required=True, columns=None):
        pq, csv = os.path.join(self.tdir, name + ".parquet"), os.path.join(self.tdir, name + ".csv")
        if os.path.exists(pq):
            try:
                return pd.read_parquet(pq, columns=columns)
            except Exception:                              # noqa: BLE001 - pyarrow missing or a partial file: CSV
                pass
        if os.path.exists(csv) and os.path.getsize(csv) > 0:
            try:
                df = pd.read_csv(csv)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            if columns is not None:
                df = df.reindex(columns=list(columns))
            return df
        if required:
            raise FileNotFoundError("missing table %s in %s (run: python -m qemx.phase.run collect %s)"
                                    % (name, self.tdir, self.dir))
        return pd.DataFrame(columns=list(columns) if columns is not None else None)

    def records(self, ids=None):
        want = None if ids is None else set(ids)
        for f in sorted(glob.glob(os.path.join(self.dir, "cells", "*.json"))):
            cid = os.path.basename(f)[:-5]
            if cid.startswith(".tmp-") or (want is not None and cid not in want):
                continue
            try:
                with open(f) as fh:
                    yield json.load(fh)
            except (OSError, ValueError):
                continue

    def run_info(self):
        p = os.path.join(self.dir, "run.json")
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    def expected_cells(self):
        try:
            from qemx.phase import manifest as M
            return len(M.expand(M.load(os.path.join(self.dir, "manifest.yaml"))))
        except Exception:                                  # noqa: BLE001 - qemx or the manifest copy unavailable
            return None


class MemorySource(RunSource):
    """Tables, records and run info held in memory (a dict of DataFrames, a list of record dicts)."""

    def __init__(self, tables: dict, records: list | None = None, run_info: dict | None = None,
                 expected: int | None = None):
        self.tables = {k: v.copy() for k, v in tables.items()}
        self._records = list(records or [])
        self._info = dict(run_info or {})
        self._expected = expected
        self.dir = "<memory>"

    def table(self, name, required=True, columns=None):
        if name not in self.tables:
            if required:
                raise FileNotFoundError("missing table %s" % name)
            return pd.DataFrame(columns=list(columns) if columns is not None else None)
        df = self.tables[name].copy()
        return df.reindex(columns=list(columns)) if columns is not None else df

    def records(self, ids=None):
        want = None if ids is None else set(ids)
        for r in self._records:
            if want is None or r.get("cell_id") in want:
                yield r

    def run_info(self):
        return dict(self._info)

    def expected_cells(self):
        return self._expected


def source(x) -> RunSource:
    """A RunSource from a source or a run directory path."""
    return x if isinstance(x, RunSource) else DirectorySource(str(x))


def read_text(path: str | None) -> str | None:
    """A text input (lock file); None when absent."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def read_csv(path: str | None) -> pd.DataFrame | None:
    """A CSV input (master list); None when absent."""
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ------------------------------------------------------------------ reports (text output)
@dataclass
class Report:
    """Sections of text, verdicts and named tables, in the order they were produced."""
    items: list = field(default_factory=list)          # ("section"|"text"|"verdict"|"table", payload)
    tables: dict = field(default_factory=dict)          # name -> DataFrame (written as CSV by a sink)
    verdicts: list = field(default_factory=list)

    def section(self, title: str):
        self.items.append(("section", str(title)))

    def text(self, s: str):
        self.items.append(("text", str(s)))

    def verdict(self, check: str, ok: bool | None, detail: str):
        tag = "PASS" if ok else ("FAIL" if ok is not None else "INFO")
        v = dict(check=check, result=tag, detail=detail)
        self.verdicts.append(v)
        self.items.append(("verdict", v))

    def table(self, name: str, df: pd.DataFrame, show: bool = False, max_rows: int | None = None):
        """Keep a table for the sink; with ``show`` it is also part of the text."""
        if df is not None and isinstance(df, pd.DataFrame):
            self.tables[name] = df
            if show:
                self.items.append(("frame", (df if max_rows is None else df.head(max_rows))))

    def frame(self, df: pd.DataFrame, max_rows: int | None = None):
        """A table shown in the text only."""
        self.items.append(("frame", df if max_rows is None else df.head(max_rows)))

    @property
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.verdicts, columns=["check", "result", "detail"])

    @property
    def failed(self) -> list:
        return [v for v in self.verdicts if v["result"] == "FAIL"]

    def render_text(self) -> str:
        out = []
        for kind, x in self.items:
            if kind == "section":
                out.append("\n" + "=" * 100 + "\n" + x + "\n" + "=" * 100)
            elif kind == "text":
                out.append(x)
            elif kind == "verdict":
                out.append("[%s] %s: %s" % (x["result"], x["check"], x["detail"]))
            elif kind == "frame":
                out.append(_frame_text(x))
        return "\n".join(out) + "\n"

    def render_markdown(self) -> str:
        out = []
        for kind, x in self.items:
            if kind == "section":
                out.append("\n## " + x + "\n")
            elif kind == "text":
                out.append("```\n" + x + "\n```" if "\n" in x else x + "\n")
            elif kind == "verdict":
                out.append("- **%s** `%s`: %s" % (x["result"], x["check"], x["detail"]))
            elif kind == "frame":
                out.append("```\n" + _frame_text(x) + "\n```")
        if self.verdicts:
            out.append("\n## Summary\n")
            out.append("| check | result | detail |\n|---|---|---|")
            for v in self.verdicts:
                out.append("| %s | %s | %s |" % (v["check"], v["result"], str(v["detail"]).replace("|", "/")))
        return "\n".join(out) + "\n"


def _frame_text(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "(empty)"
    with pd.option_context("display.width", 200, "display.max_columns", 40, "display.max_colwidth", 60):
        idx = not isinstance(df.index, pd.RangeIndex)
        return df.to_string(index=idx)


class DirectoryReportSink:
    """Writes a report: every non-empty table as <name>.csv, summary.csv, report.md and report.txt."""

    def __init__(self, out_dir: str):
        self.dir = out_dir

    def write(self, rep: Report) -> list:
        os.makedirs(self.dir, exist_ok=True)
        paths = []
        for name, df in rep.tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                p = os.path.join(self.dir, name + ".csv")
                df.to_csv(p, index=not isinstance(df.index, pd.RangeIndex))
                paths.append(p)
        for fname, text in (("summary.csv", None), ("report.md", rep.render_markdown()),
                            ("report.txt", rep.render_text())):
            p = os.path.join(self.dir, fname)
            if text is None:
                rep.summary.to_csv(p, index=False)
            else:
                with open(p, "w") as f:
                    f.write(text)
            paths.append(p)
        return paths


class MemoryReportSink:
    def __init__(self):
        self.reports = []

    def write(self, rep: Report) -> list:
        self.reports.append(rep)
        return []


# ------------------------------------------------------------------ figure sinks (graphical output)
class DirectoryFigureSink:
    """Saves each figure as <out>/<name>.<ext> for every format, then closes it."""

    def __init__(self, out_dir: str, formats=("pdf", "png")):
        self.dir, self.formats = out_dir, tuple(formats)
        self.saved = []

    def save(self, fig, name: str) -> str:
        import matplotlib.pyplot as plt
        os.makedirs(self.dir, exist_ok=True)
        for ext in self.formats:
            fig.savefig(os.path.join(self.dir, "%s.%s" % (name, ext)))
        plt.close(fig)
        path = os.path.join(self.dir, "%s.%s" % (name, self.formats[0]))
        self.saved.append(path)
        return path


class MemoryFigureSink:
    """Keeps the figures by name (open: the caller may inspect and close them)."""

    def __init__(self):
        self.figures = {}

    def save(self, fig, name: str) -> str:
        self.figures[name] = fig
        return name

    @property
    def saved(self) -> list:
        return list(self.figures)
