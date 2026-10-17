r"""Storage of Direction A data: one JSON record per cell, flat tables for analysis, a versioned schema.

Layout of an output directory::

    <out>/manifest.yaml            the manifest that was run (copied verbatim) and its SHA-256 in run.json
    <out>/run.json                 provenance of the run (versions, machine, start time, manifest hash)
    <out>/cells/<cell_id>.json     one record per cell (schema below); written atomically (tmp file + rename)
    <out>/tables/methods.{parquet,csv}     one row per (cell, method): the ledger primitives and cell facts
    <out>/tables/ledger.{parquet,csv}      one row per (cell, method, policy, B): integer allocation and RMSE
    <out>/tables/winners.{parquet,csv}     one row per (cell, policy, B): winner, runner-up, margin
    <out>/tables/levels.{parquet,csv}      learning methods: b and u per training level
    <out>/tables/validation.{parquet,csv}  Monte-Carlo checks
    <out>/tables/errors.{parquet,csv}      cells, methods, levels or checks that failed, with the error
    <out>/tables/collect.json              what collect used: records, stale ids (not in the manifest), missing ids

Records never contain NaN or infinity (JSON has neither): non-finite numbers are stored as null and the method's
``status`` says why ("nonfinite", "error"). The tables are derived data: ``collect`` rebuilds them from the records.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np

SCHEMA_VERSION = "qemx.phase.cell/3"
# /2 (v1.4): status "partial", per-level ok/v/w/se_w.
# /3 (v1.5): learning rows (CDR / FR / NIL / chains) report the *protocol* moments at their training shots in
#     mean / b / v (qemx/stats.py), the exact-training reference fit in reference_mean / b_inf, and their Monte-Carlo
#     record in ``moments``; every row carries estimate_kind and flags. /1–/2 records keep their own meaning: there
#     mean / b / v of a learning row are the reference fit's (the ledger used the measured training levels anyway).
READABLE_SCHEMAS = ("qemx.phase.cell/1", "qemx.phase.cell/2", SCHEMA_VERSION)
CELL_STATUS = ("ok", "partial", "error", "timeout", "oom", "crashed")
USABLE_STATUS = ("ok", "partial")                          # cells whose methods enter the tables

_NUM = {"type": ["number", "null"]}
_INT = {"type": "integer"}
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qem-explorer/schemas/phase-cell-1.json",
    "title": "QEM Explorer v1.3 · Direction A · one phase-diagram cell",
    "type": "object",
    "required": ["schema", "cell_id", "cell", "status", "provenance"],
    "properties": {
        "schema": {"enum": list(READABLE_SCHEMAS)},
        "cell_id": {"type": "string", "pattern": "^[0-9a-f]{20}$"},
        "status": {"enum": list(CELL_STATUS)},
        "error": {"type": "string"},
        "cell": {"type": "object", "required": ["stage", "family", "n", "fparams", "device", "engine", "scale",
                                                "readout", "readout_mit", "coherent_fraction", "methods", "seed"],
                 "properties": {"n": _INT, "scale": {"type": "number", "exclusiveMinimum": 0},
                                "engine": {"enum": ["E1", "E2", "E3"]}, "readout": {"enum": ["on", "off"]},
                                "coherent_fraction": {"type": "number", "minimum": 0, "maximum": 1}}},
        "circuit": {"type": "object", "required": ["phys", "two_qubit", "depth", "measure_map"]},
        "lambda": {"type": "object", "required": ["lam_lc", "lam_lc_exact", "lam_total", "lam_ro", "support"],
                   "properties": {"lam_lc": {"type": "number", "minimum": 0},
                                  "lam_total": {"type": "number", "minimum": 0}}},
        "descriptors": {"type": "object"},
        "ideal": {"type": "number"},
        "noisy": {"type": "object"},
        "methods": {"type": "array", "items": {
            "type": "object", "required": ["key", "estimator", "status"],
            "properties": {"status": {"enum": ["ok", "nonfinite", "error"]}, "b": _NUM, "v": _NUM, "w": _NUM,
                           "c": _INT, "T": _INT, "s_tr": _INT, "F": _INT, "u": _NUM, "m": _INT,
                           "a": {"type": "array", "items": {"type": "number"}},
                           "levels": {"type": "array", "items": {
                               "type": "object", "required": ["s_tr", "reps"],
                               "if": {"properties": {"ok": {"const": False}}, "required": ["ok"]},
                               "then": {"required": ["error"]},
                               "else": {"required": ["b", "u", "se_b", "se_u"]}}}}}},
        "guards": {"type": "array"},
        "validation": {"type": "array"},
        "provenance": {"type": "object", "required": ["qemx", "qiskit", "qiskit_aer", "calibration_hash", "seconds"]},
    },
}


def clean(obj):
    """JSON-safe copy: numpy scalars/arrays to Python, non-finite floats to None, tuples to lists."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    return obj


def validate_record(rec: dict) -> None:
    """Raise jsonschema.ValidationError if the record does not match SCHEMA (jsonschema is required for this)."""
    import jsonschema
    jsonschema.validate(rec, SCHEMA)


def failure_record(spec_json: dict, cell_id: str, status: str, error: str, seconds: float = 0.0) -> dict:
    """The record of a cell that did not complete (time limit, out of memory, crash, pipeline error)."""
    from .cell import provenance
    if status not in CELL_STATUS or status in USABLE_STATUS:
        raise ValueError("not a failure status: %r" % status)
    return dict(schema=SCHEMA_VERSION, cell_id=cell_id, cell=spec_json, status=status, error=str(error),
                provenance=dict(provenance(), calibration_hash="", seconds=float(seconds)))


def cell_path(out_dir: str, cell_id: str) -> str:
    return os.path.join(out_dir, "cells", "%s.json" % cell_id)


def write_cell(out_dir: str, rec: dict, validate: bool = True) -> str:
    """Atomic write (a crash never leaves a half-written record that resume would trust). With ``validate`` the record
    is checked against SCHEMA first (if jsonschema is installed): an invalid record raises ValueError, it is never
    written."""
    d = os.path.join(out_dir, "cells")
    os.makedirs(d, exist_ok=True)
    path = cell_path(out_dir, rec["cell_id"])
    rec = clean(rec)
    if validate:
        try:
            import jsonschema
        except ImportError:                              # optional dependency: the check is skipped, not faked
            jsonschema = None
        if jsonschema is not None:
            try:
                jsonschema.validate(rec, SCHEMA)
            except jsonschema.ValidationError as e:
                raise ValueError("cell record %s does not match the schema: %s at %s" % (
                    rec.get("cell_id"), e.message, "/".join(map(str, e.absolute_path)))) from None
    data = json.dumps(rec, allow_nan=False, sort_keys=True, separators=(",", ":"))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)                           # mkstemp creates 0600; records are shared data
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_cell(out_dir: str, cell_id: str) -> dict | None:
    p = cell_path(out_dir, cell_id)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def cell_ids(out_dir: str) -> list:
    """Ids of the records in the directory (sorted)."""
    d = os.path.join(out_dir, "cells")
    if not os.path.isdir(d):
        return []
    return sorted(n[:-5] for n in os.listdir(d) if n.endswith(".json") and not n.startswith(".tmp-"))


def iter_cells(out_dir: str, ids=None):
    """The records (sorted by id), or those of ``ids`` that exist."""
    for cid in (cell_ids(out_dir) if ids is None else sorted(ids)):
        r = read_cell(out_dir, cid)
        if r is not None:
            yield r


def clean_tmp(out_dir: str) -> int:
    """Remove temporary files of interrupted writes (call only when no runner writes to the directory)."""
    d = os.path.join(out_dir, "cells")
    n = 0
    if os.path.isdir(d):
        for name in os.listdir(d):
            if name.startswith(".tmp-"):
                os.unlink(os.path.join(d, name))
                n += 1
    return n


# ------------------------------------------------------------------ flat tables
def _cell_cols(rec: dict) -> dict:
    c = rec["cell"]
    lam = rec.get("lambda") or {}
    de = rec.get("descriptors") or {}
    no = rec.get("noisy") or {}
    ci = rec.get("circuit") or {}
    out = dict(cell_id=rec["cell_id"], stage=c["stage"], family=c["family"], n=c["n"],
               fparams=json.dumps(c["fparams"], sort_keys=True), device=c["device"], engine=c["engine"],
               scale=c["scale"], readout=c["readout"], readout_mit=c["readout_mit"],
               coherent_fraction=c["coherent_fraction"], opt_level=c["opt_level"], seed=c["seed"],
               s_min=c.get("s_min", 32), s_tr_min=c.get("s_tr_min", 32),
               cell_status=rec["status"], ideal=rec.get("ideal"), two_qubit=ci.get("two_qubit"),
               depth=ci.get("depth"), phys=json.dumps(ci.get("phys")))
    for k in ("lam_lc", "lam_lc_exact", "lam_total", "lam_ro", "lam_lc_gate", "lam_lc_idle", "n_loc_lc", "lc_gates",
              "lc_2q_gates"):
        out[k] = lam.get(k)
    for k, v in de.items():
        out["d_" + k] = v
    for k in ("raw_value", "fidelity", "purity"):
        out["noisy_" + k] = no.get(k)
    for k, v in (c.get("fparams") or {}).items():
        out["f_" + k] = v
    return out


def method_rows(rec: dict) -> list:
    base = _cell_cols(rec)
    out = []
    for r in rec.get("methods") or []:
        row = dict(base)
        row.update(method=r["key"], estimator=r["estimator"], status=r["status"], error=r.get("error"))
        for k in ("mean", "b", "b_inf", "v", "w", "c", "T", "s_tr", "F", "u", "u_delta", "u1", "m", "gamma",
                  "circuits", "extra_qubits"):
            row[k] = r.get(k)
        row["a"] = json.dumps(r.get("a") or [])
        row["flags"] = ",".join(r.get("flags") or [])
        det = r.get("details") or {}
        for k in ("degenerate", "rank", "nonzero", "l1", "G", "trace", "dominant_value", "n_features", "lam",
                  "solver_ok", "model", "target_split"):
            if k in det:
                row["x_" + k] = det[k]
        for k in ("lams", "lams_gates", "lams_fault", "depths", "twoq", "draws", "E_draws"):
            if det.get(k) is not None:
                row["x_" + k] = json.dumps(det[k])
        for k in ("folding", "lam_est", "rule", "subset"):
            if k in det:
                row["x_" + k] = det[k]
        out.append(row)
    return out


def level_rows(rec: dict) -> list:
    out = []
    for r in rec.get("methods") or []:
        for l in r.get("levels") or []:
            out.append(dict(cell_id=rec["cell_id"], method=r["key"], **l))
    return out


def validation_rows(rec: dict) -> list:
    out = []
    for v in rec.get("validation") or []:
        row = dict(cell_id=rec["cell_id"], method=v.get("method"), policy=v.get("policy"), B=v.get("B"),
                   feasible=v.get("feasible"), N=v.get("N"), s=v.get("s"), s_tr=v.get("s_tr"),
                   identity=v.get("identity"), passed=v.get("passed"), independent=v.get("independent"),
                   z_rmse=v.get("z_rmse"), z_bias=v.get("z_bias"), z_rmse_boot=v.get("z_rmse_boot"),
                   var_ratio=v.get("var_ratio"), calibration_mc=v.get("calibration_mc"),
                   error=v.get("error"), skipped=v.get("skipped"))
        for k, val in (v.get("mc") or {}).items():
            row["mc_" + k] = val
        for k, val in (v.get("analytic") or {}).items():
            row["an_" + k] = val
        out.append(row)
    return out


def cell_policies(rec: dict, names=("fixed", "opt")) -> list:
    """The ledger policies of a cell: its own lower bounds s_min, s_tr_min (as used by its validation)."""
    from .ledger import Policy
    c = rec["cell"]
    return [Policy(n, True, int(c.get("s_min", 32)), int(c.get("s_tr_min", 32))) for n in names]


def ledger_rows(rec: dict, budgets, policies=("fixed", "opt")) -> tuple:
    """(ledger rows, winner rows) of one cell over a budget grid. ``policies``: ledger.Policy objects, or names
    (then the cell's own lower bounds are used)."""
    from .cell import cost_of
    from .ledger import Policy, allocate
    names = [p for p in policies if isinstance(p, str)]
    pols = [p for p in policies if isinstance(p, Policy)] + (cell_policies(rec, names) if names else [])
    policies = pols
    allc = [(r["key"], cost_of(r)) for r in rec.get("methods") or []]
    costs = [(k, c) for k, c in allc if c is not None]
    missing = ",".join(k for k, c in allc if c is None)        # methods without a finite estimate in this cell
    optimistic = {r["key"] for r in rec.get("methods") or [] if "optimistic" in (r.get("flags") or [])}
    L, W = [], []
    for pol in policies:
        for B in budgets:
            res = []
            for k, c in costs:
                a = allocate(c, int(B), pol)
                rm = math.sqrt(a.mse) if a.feasible else math.inf
                L.append(dict(cell_id=rec["cell_id"], method=k, policy=pol.name, B=int(B), feasible=a.feasible,
                              N=a.N if a.feasible else None, s=a.s if a.feasible else None,
                              s_tr=a.s_tr if a.feasible else None, b=a.b if a.feasible else None,
                              var=a.var if a.feasible else None, rmse=rm if math.isfinite(rm) else None))
                res.append((rm, k))
            for field, pool in (("all", res), ("no_optimistic", [t for t in res if t[1] not in optimistic])):
                pool = sorted(pool, key=lambda t: t[0])
                if pool and math.isfinite(pool[0][0]):
                    ru = pool[1] if len(pool) > 1 else (math.inf, None)
                    W.append(dict(cell_id=rec["cell_id"], policy=pol.name, B=int(B), candidates=field,
                                  n_methods=len(pool), n_feasible=sum(1 for r_, _ in pool if math.isfinite(r_)),
                                  missing=missing, winner=pool[0][1], rmse=pool[0][0], runner_up=ru[1],
                                  rmse_runner_up=ru[0] if math.isfinite(ru[0]) else None,
                                  margin=(ru[0] / pool[0][0] - 1) if math.isfinite(ru[0]) and pool[0][0] > 0
                                  else None))
    return L, W


ERROR_COLUMNS = ["cell_id", "stage", "status", "method", "error"]
TABLE_KEYS = {  # columns an empty table still carries (readers can filter it without special cases)
    "methods": ["cell_id", "stage", "method", "estimator", "status"],
    "levels": ["cell_id", "method", "s_tr", "ok", "b", "u", "v", "w"],
    "validation": ["cell_id", "method", "policy", "B", "passed"],
    "ledger": ["cell_id", "method", "policy", "B", "feasible", "rmse"],
    "winners": ["cell_id", "policy", "B", "candidates", "winner", "rmse"],
    "errors": ERROR_COLUMNS,
}


def _write_table(rows: list, path_base: str, columns=None) -> dict:
    """Write <base>.csv and <base>.parquet, replacing old files first (a stale table must never survive a collect).
    Returns {"rows": n, "parquet": bool, "warning": str?}."""
    import pandas as pd
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns or [])
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    for ext in (".csv", ".parquet"):
        if os.path.exists(path_base + ext):
            os.unlink(path_base + ext)
    df.to_csv(path_base + ".csv", index=False)
    out = dict(rows=int(len(df)), parquet=False)
    try:
        df.to_parquet(path_base + ".parquet", index=False)
        out["parquet"] = True
    except Exception as e:                               # noqa: BLE001 - parquet is optional (pyarrow); CSV is canonical
        if os.path.exists(path_base + ".parquet"):
            os.unlink(path_base + ".parquet")
        out["warning"] = "no parquet: %s: %s" % (type(e).__name__, str(e)[:200])
    return out


def budget_grid(manifest: dict | None) -> list:
    """The manifest's budget grid (budgets.grid: log10_min, log10_max, points), else 10³ … 10⁸ with 51 points."""
    g = ((manifest or {}).get("budgets") or {}).get("grid") or {}
    lo, hi, n = float(g.get("log10_min", 3)), float(g.get("log10_max", 8)), int(g.get("points", 51))
    return sorted({int(round(x)) for x in np.logspace(lo, hi, n)})


def collect(out_dir: str, budgets=None, policies=("fixed", "opt"), manifest_path: str | None = None,
            only_manifest: bool = True) -> dict:
    """Rebuild every table from the cell records.

    The records used are those of the run's manifest (``<out>/manifest.yaml``, or ``manifest_path``): a cell whose
    specification changed has a new id, and its old record must not be counted next to the new one. Records of
    other ids are reported as ``stale`` and left out (``only_manifest=False`` uses every record). Cells of the
    manifest without a record are reported as ``missing``. The budget grid is the manifest's ``budgets.grid``
    unless ``budgets`` is given; the policies use each cell's own lower bounds. A record that cannot be processed is
    an entry of the errors table, never an abort."""
    from .cell import CellSpec  # noqa: F401 - imported for the manifest expansion's types
    mpath = manifest_path or os.path.join(out_dir, "manifest.yaml")
    manifest, wanted, warnings_ = None, None, []
    if os.path.exists(mpath):
        from .manifest import expand, load
        try:
            manifest = load(mpath)
            if only_manifest:
                wanted = {c.cell_id for c in expand(manifest)}
        except Exception as e:                           # noqa: BLE001 - reported; every record is used instead
            manifest, wanted = None, None
            warnings_.append("the run's manifest could not be expanded (%s: %s); every record is used"
                             % (type(e).__name__, str(e)[:300]))
    else:
        warnings_.append("no manifest.yaml in %s: every record is used" % out_dir)
    budgets = list(budgets) if budgets is not None else budget_grid(manifest)
    present = set(cell_ids(out_dir))
    if wanted is not None and present and not (present & wanted):
        warnings_.append("none of the %d records is a cell of the manifest (records of another manifest or of "
                         "another qemx version?); nothing is tabulated — check the directory, or collect with "
                         "--all-records" % len(present))
    stale = sorted(present - wanted) if wanted is not None else []
    missing = sorted(wanted - present) if wanted is not None else []
    M, LV, VA, LE, WI, ER = [], [], [], [], [], []
    for rec in iter_cells(out_dir, sorted(present & wanted) if wanted is not None else None):
        try:
            if rec.get("schema") not in READABLE_SCHEMAS:
                raise ValueError("unknown record schema %r" % rec.get("schema"))
            if rec["status"] not in USABLE_STATUS:
                ER.append(dict(cell_id=rec["cell_id"], stage=rec["cell"]["stage"], status=rec["status"],
                               method=None, error=(rec.get("error") or "")[:2000]))
                continue
            m_, lv_, va_ = method_rows(rec), level_rows(rec), validation_rows(rec)
            l_, w_ = ledger_rows(rec, budgets, policies)
            er_ = []
            for r in rec.get("methods") or []:
                if r["status"] == "error":
                    er_.append(dict(cell_id=rec["cell_id"], stage=rec["cell"]["stage"], status="method-error",
                                    method=r["key"], error=(r.get("error") or "")[:2000]))
                for l in r.get("levels") or []:
                    if l.get("ok") is False:
                        er_.append(dict(cell_id=rec["cell_id"], stage=rec["cell"]["stage"], status="level-error",
                                        method=r["key"], error="s_tr=%s: %s" % (l.get("s_tr"), l.get("error"))))
            for v in rec.get("validation") or []:
                if v.get("error"):
                    er_.append(dict(cell_id=rec["cell_id"], stage=rec["cell"]["stage"], status="validation-error",
                                    method=v.get("method"), error="%s B=%s: %s" % (v.get("policy"), v.get("B"),
                                                                                   v["error"])[:2000]))
        except Exception as e:                           # noqa: BLE001 - one bad record never aborts a collect
            ER.append(dict(cell_id=rec.get("cell_id"), stage=(rec.get("cell") or {}).get("stage"),
                           status="collect-error", method=None, error="%s: %s" % (type(e).__name__, e)))
            continue
        M += m_
        LV += lv_
        VA += va_
        LE += l_
        WI += w_
        ER += er_
    t = os.path.join(out_dir, "tables")
    out = dict(records=len(present), used=len(present & wanted) if wanted is not None else len(present),
               stale=len(stale), missing=len(missing), budgets=len(budgets), warnings=warnings_, tables={})
    for name, rows in (("methods", M), ("levels", LV), ("validation", VA), ("ledger", LE), ("winners", WI),
                       ("errors", ER)):
        out["tables"][name] = _write_table(rows, os.path.join(t, name), ERROR_COLUMNS if name == "errors"
                                           else (None if rows else TABLE_KEYS[name]))
    with open(os.path.join(t, "schema.json"), "w") as f:
        json.dump(SCHEMA, f, indent=1)
    with open(os.path.join(t, "collect.json"), "w") as f:
        json.dump(dict(out, stale_ids=stale, missing_ids=missing), f, indent=1)
    return out
