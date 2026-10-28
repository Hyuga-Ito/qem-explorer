"""The YAML formats of the project, each routed to its own parser (1.9.0; second review N6).

A YAML file is identified by its content, never by its folder:

=====================================  =========================================  ===============================================
format (identifying key)               parser                                     command
=====================================  =========================================  ===============================================
selection study configuration          ``qemx.selection.design.load`` + ``check``  ``python -m qemx.selection plan|label|… FILE``
(``schema: qemx.selection.config/1``)
study pipeline                         ``qemx.selection.pipeline.load``           ``python -m qemx.selection pipeline FILE``
(``schema: qemx.selection.pipeline/1``)
selection plotting configuration       ``qemx.selection.plots.load_config``       ``… figures CONFIG --plot-config FILE``
(``schema: qemx.selection.figures/1``)
figure/table specification             keys checked here; generators tested      ``python -m qemx.selection.tables19``
(``schema: qemx.selection.figure-spec/1``)
experiment document                    ``qemx.experiment.load`` + ``resolve``     ``python -m qemx.experiment validate|plan|run``;
(``schema: qemx.experiment/1``)                                                  ``… selection recommend --experiment FILE``
divergence manifest                    ``qemx.divergence.suite_from_manifest``    ``python scripts/dirA_divergence.py plan FILE``
(``manifest_version`` + ``kind: divergence``)
phase-study manifest                   ``qemx.phase.manifest.load`` + ``expand``  ``python -m qemx.phase.run plan FILE``
(``manifest_version``, no ``kind``)
hardware section                       ``qemx.selection.hardware.plan``           ``… selection hardware plan --hardware-config``
(``backend_name``, ``blocks``, ``budget``)                                        (plans only; never submits)
GitHub Actions workflow                (GitHub)                                   —
=====================================  =========================================  ===============================================

``check(path)`` validates a file with the parser of its format and returns a uniform report. Parsing is not
execution: a valid large configuration has not been run.
"""
from __future__ import annotations

import os

from .. import textio as TX

FORMATS = ("selection-config", "pipeline", "plot-config", "figure-spec", "experiment", "divergence-manifest",
           "phase-manifest", "hardware-section", "github-workflow", "unknown")
SPEC_PANEL_KEYS = ("id", "question", "source", "status", "generator")


def identify(doc) -> str:
    if not isinstance(doc, dict):
        return "unknown"
    sch = doc.get("schema")
    if sch == "qemx.selection.config/1":
        return "selection-config"
    if sch == "qemx.selection.pipeline/1":
        return "pipeline"
    if sch == "qemx.selection.figures/1":
        return "plot-config"
    if sch == "qemx.selection.figure-spec/1":
        return "figure-spec"
    if sch == "qemx.experiment/1":
        return "experiment"
    if "manifest_version" in doc:
        return "divergence-manifest" if doc.get("kind") == "divergence" else "phase-manifest"
    if {"backend_name", "blocks", "budget"} <= set(doc):
        return "hardware-section"
    if "jobs" in doc and ("on" in doc or True in doc):
        return "github-workflow"
    return "unknown"


SPEC_TOP_KEYS = {"schema", "version", "metric", "figures", "tables"}
SPEC_STATUSES = ("confirmatory", "exploratory", "descriptive", "pending")


def check_figure_spec(doc: dict) -> list:
    """Errors of a figure/table specification (1.9.0; independent verification F11): closed top-level keys, a version,
    unique figure, panel and table ids, every required key, a status from the vocabulary (its first word) and a
    generator that exists (``module.function``)."""
    import importlib
    errs = []
    if not isinstance(doc, dict):
        return ["the specification is not a mapping"]
    extra = sorted(set(doc) - SPEC_TOP_KEYS)
    if extra:
        errs.append("unknown top-level key(s) %s (allowed: %s)" % (extra, sorted(SPEC_TOP_KEYS)))
    if not doc.get("version"):
        errs.append("version is missing")
    figs, tabs = doc.get("figures"), doc.get("tables")
    if not isinstance(figs, list) or not isinstance(tabs, list):
        return errs + ["figures and tables must be lists"]
    seen = set()
    entries = []
    for f in figs:
        if not isinstance(f, dict) or not isinstance(f.get("panels"), list) or not f.get("id"):
            errs.append("figure %r: needs an id and a list of panels" % (f.get("id") if isinstance(f, dict) else f))
            continue
        for pnl in f["panels"]:
            if not isinstance(pnl, dict):
                errs.append("figure %s: a panel is not a mapping" % f["id"])
                continue
            miss = [k for k in SPEC_PANEL_KEYS if k not in pnl]
            if miss:
                errs.append("panel %s lacks %s" % (pnl.get("id", "?"), miss))
            entries.append(pnl)
    for t in tabs:
        if not isinstance(t, dict):
            errs.append("a table entry is not a mapping")
            continue
        miss = [k for k in ("id", "question", "source", "status", "generator") if k not in t]
        if miss:
            errs.append("table %s lacks %s" % (t.get("id", "?"), miss))
        entries.append(t)
    for e in [f for f in figs if isinstance(f, dict)] + entries:
        i = e.get("id")
        if i in seen:
            errs.append("duplicate id %s" % i)
        seen.add(i)
    for e in entries:
        st = str(e.get("status", "")).split()[0].rstrip(",;").lower() if e.get("status") else ""
        if st and st not in SPEC_STATUSES:
            errs.append("%s: status %r is not one of %s" % (e.get("id", "?"), e.get("status"), SPEC_STATUSES))
        g = str(e.get("generator", ""))
        if g:
            mod, _, fn = g.rpartition(".")
            try:
                if not mod or not hasattr(importlib.import_module(mod), fn):
                    errs.append("%s: generator %s does not exist" % (e.get("id", "?"), g))
            except ImportError:
                errs.append("%s: generator module %s cannot be imported" % (e.get("id", "?"), mod))
    return errs


def check(path: str) -> dict:
    """{ok, format, parser, errors, warnings, summary, command} of one YAML file."""
    out = dict(path=path, ok=False, format="unknown", parser="", errors=[], warnings=[], summary="", command="")
    try:
        doc = TX.read_yaml(path)
    except Exception as e:                                   # noqa: BLE001
        out["errors"].append("not readable as UTF-8 YAML: %s" % str(e)[:300])
        return out
    fmt = identify(doc)
    out["format"] = fmt
    try:
        if fmt == "selection-config":
            from . import design as DS
            out["parser"] = "qemx.selection.design.load + check"
            cfg = DS.load(path)
            chk = DS.check(cfg)
            out["errors"] += chk["errors"]
            out["warnings"] += chk["warnings"]
            if not chk["errors"]:
                rep = DS.expansion_report(cfg)
                out["summary"] = rep["summary"]
                out["warnings"] += rep["notes"]
            out["command"] = "python -m qemx.selection plan %s" % path
        elif fmt == "pipeline":
            from . import pipeline as PL
            out["parser"] = "qemx.selection.pipeline.load"
            p = PL.load(path)
            out["summary"] = "%s pipeline %r: development %s%s" % (
                p["purpose"], p.get("name"), os.path.basename(p["development"]),
                ", confirmation %s" % os.path.basename(p["confirmation"]["config"]) if p.get("confirmation") else "")
            out["command"] = "python -m qemx.selection pipeline %s --out RUN_DIR" % path
        elif fmt == "plot-config":
            from . import plots as PLT
            out["parser"] = "qemx.selection.plots.load_config"
            cfg = PLT.load_config(path)
            out["summary"] = "plotting configuration (%d settings)" % len(cfg)
            out["command"] = "python -m qemx.selection figures CONFIG --out RUN --plot-config %s" % path
        elif fmt == "figure-spec":
            out["parser"] = "qemx.selection.formats.check_figure_spec (keys, ids, statuses, generators)"
            out["errors"].extend(check_figure_spec(doc))
            out["summary"] = "figure/table specification %s (%d figures, %d tables)" % (
                doc.get("version"), len(doc.get("figures") or []) if isinstance(doc.get("figures"), list) else 0,
                len(doc.get("tables") or []) if isinstance(doc.get("tables"), list) else 0)
            out["command"] = "python -m qemx.selection.tables19"
        elif fmt == "experiment":
            from .. import experiment as EX
            out["parser"] = "qemx.experiment.load + resolve"
            EX.resolve(EX.load(path))
            out["summary"] = "experiment document"
            out["command"] = "python -m qemx.experiment validate %s" % path
        elif fmt == "divergence-manifest":
            from ..divergence import suite_from_manifest
            out["parser"] = "qemx.divergence.suite_from_manifest"
            suite_from_manifest(doc)
            out["summary"] = "divergence manifest (study %s)" % doc.get("study")
            out["command"] = "python scripts/dirA_divergence.py plan %s --out DIR" % path
        elif fmt == "phase-manifest":
            from ..phase import manifest as PM
            out["parser"] = "qemx.phase.manifest.load"
            m = PM.load(path)
            out["summary"] = "phase-study manifest (study %s, %d stage(s))" % (m.get("study"), len(m.get("stages", [])))
            out["command"] = "python -m qemx.phase.run plan %s" % path
        elif fmt == "hardware-section":
            out["parser"] = "keys of the hardware section (qemx.selection.hardware.plan reads it)"
            need = {"backend_name", "blocks", "budget", "ceiling_s", "device", "protocols"}
            miss = sorted(need - set(doc))
            if miss:
                out["errors"].append("missing keys %s" % miss)
            out["summary"] = "hardware validation section (plan only; never submits)"
            out["command"] = "python -m qemx.selection hardware plan --hardware-config %s --bundle BUNDLE" % path
        elif fmt == "github-workflow":
            out["parser"] = "(GitHub Actions)"
            out["summary"] = "GitHub Actions workflow (not executed locally)"
        else:
            out["errors"].append("unknown YAML format (no schema, manifest_version or known keys)")
    except Exception as e:                                   # noqa: BLE001 - every parser's refusal is reported
        out["errors"].append("%s: %s" % (type(e).__name__, str(e)[:600]))
    out["ok"] = not out["errors"]
    return out


def inventory(root: str) -> list:
    """``check`` of every YAML file under ``root`` (excluding run outputs, data and caches)."""
    rows = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__", ".git", "runs", "reports", "data", "node_modules"))
        for f in sorted(fn):
            if f.endswith((".yaml", ".yml")):
                rows.append(check(os.path.join(dp, f)))
    return rows
