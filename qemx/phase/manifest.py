r"""Manifests: the complete, versioned list of cells of a Direction A study, expanded deterministically.

A manifest (YAML) has five parts::

    manifest_version: 1
    study: <name>
    defaults:  {device, opt_level, seed_transpiler, readout_mit, s_min, s_tr_min, train_levels, train_reps, …}
    methods:   {KEY: {estimator, <parameters>, twirl?, dd?}, …}          the pre-registered method definitions
    method_sets: {name: [KEY, …]}
    circuits:  {name: [{family, n, <family parameters>}, …]}              named circuit lists (grids allowed, below)
    stages:    [ {name, engine, circuits, methods, scales, readout, coherent_fraction, devices?, layout?,
                  opt_levels?, validate?: {fraction, budgets, reps, policies}}, … ]

Grid shorthand inside a circuit entry: any parameter given as a list expands to all combinations, e.g.
``{family: ising, n: [4, 6], steps: [1, 2, 4, 6, 8]}`` is ten circuits. A stage expands to
circuits × devices × opt_levels × scales × readout arms × coherent fractions. ``validate.fraction`` selects that
fraction of every (family, n) stratum of the stage for Monte-Carlo validation: the cells with the smallest hash of
their id (deterministic, balanced, independent of order).

The method seed of a cell is a hash of its circuit (family, n, parameters): the same training circuits at every noise
scale, engine and readout arm (common random numbers along the λ axis).
"""
from __future__ import annotations

import hashlib
import os
import sys
import itertools
import json
import math

from .cell import CellSpec, MethodSpec

ALLOWED_TOP = {"manifest_version", "study", "description", "defaults", "methods", "method_sets", "circuits", "stages",
               "qpu", "budgets"}
ALLOWED_STAGE = {"name", "engine", "circuits", "methods", "scales", "readout", "readout_mit", "coherent_fraction",
                 "devices", "layout", "opt_levels", "validate", "description", "train_levels", "train_reps",
                 "train_reps_other", "design", "confirmatory"}
DESIGN_KINDS = ("grid", "maximin")


def load(path: str) -> dict:
    import yaml
    with open(path) as f:
        m = yaml.safe_load(f)
    check(m)
    for note in migrations(m):
        print("manifest %s: %s" % (os.path.basename(path), note), file=sys.stderr)
    return m


def migrations(m: dict) -> list:
    """Notes on legacy parameter values in the manifest's methods (methods.base.PARAM_ALIASES). The manifest is not
    changed — its methods, hence its cell ids, stay exactly as pre-registered; the legacy value is resolved when the
    recipe is built (Recipe.make), and the record stores the parameters as written."""
    from ..methods.base import migrate_params
    out = []
    for key, d in (m.get("methods") or {}).items():
        d = dict(d)
        est = d.pop("estimator", None)
        for note in migrate_params(est, d)[1]:
            out.append("method %s — %s" % (key, note))
    return out


def manifest_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check(m: dict) -> None:
    """Structural validation with precise messages (a typo must not silently drop part of a study)."""
    if not isinstance(m, dict):
        raise ValueError("a manifest is a mapping")
    bad = set(m) - ALLOWED_TOP
    if bad:
        raise ValueError("unknown top-level keys: %s" % sorted(bad))
    if int(m.get("manifest_version", 0)) != 1:
        raise ValueError("manifest_version must be 1")
    for k in ("methods", "circuits", "stages"):
        if k not in m:
            raise ValueError("missing %r" % k)
    if not isinstance(m["methods"], dict) or not all(isinstance(d, dict) for d in m["methods"].values()):
        raise ValueError("'methods' must map method keys to parameter mappings")
    if not isinstance(m["stages"], list) or not all(isinstance(st, dict) and "name" in st for st in m["stages"]):
        raise ValueError("'stages' must be a list of mappings, each with a 'name'")
    if not isinstance(m.get("method_sets", {}) or {}, dict):
        raise ValueError("'method_sets' must map set names to lists of method keys")
    for key, d in m["methods"].items():
        _check_method(key, d)
    sets = m.get("method_sets", {})
    for name, keys in sets.items():
        for k in keys:
            if k not in m["methods"]:
                raise ValueError("method set %s: unknown method %s" % (name, k))
        if len(set(keys)) != len(keys):
            raise ValueError("method set %s lists a method twice: %s" % (
                name, sorted({k for k in keys if list(keys).count(k) > 1})))
    _check_budgets(m.get("budgets"))
    from ..device import FAKE_DEVICES
    dflt = m.get("defaults", {}) or {}
    bad = set(dflt) - {"device", "layout", "opt_level", "seed_transpiler", "readout_mit", "s_min", "s_tr_min",
                       "train_levels", "train_reps", "train_reps_other"}
    if bad:
        raise ValueError("unknown defaults: %s" % sorted(bad))
    if dflt.get("readout_mit", "tensored") not in ("none", "tensored", "m3", "trex"):
        raise ValueError("defaults.readout_mit must be none, tensored, m3 or trex")
    if dflt.get("device", "fake_fez") not in FAKE_DEVICES:
        raise ValueError("defaults.device %r is not a fake device" % dflt.get("device"))
    for k in ("s_min", "s_tr_min", "train_reps", "train_reps_other"):
        if k in dflt and not (_num(dflt[k]) and int(dflt[k]) == dflt[k] and dflt[k] >= (2 if "reps" in k else 1)):
            raise ValueError("defaults.%s must be an integer >= %d" % (k, 2 if "reps" in k else 1))
    if not all(_num(x) and x >= 1 and int(x) == x for x in dflt.get("train_levels", [1])):
        raise ValueError("defaults.train_levels must be integers >= 1")
    if int(dflt.get("opt_level", 1)) not in (0, 1, 2, 3):
        raise ValueError("defaults.opt_level must be 0..3")
    for key, d in m["methods"].items():
        if d.get("estimator") == "zne":
            sc = d.get("scales", [1.0, 3.0, 5.0])
            if any(not _num(x) or x < 1 for x in sc) or len(set(sc)) < 2:
                raise ValueError("method %s: ZNE scales must be >= 1 and at least two distinct values" % key)
    names = set()
    for st in m["stages"]:
        bad = set(st) - ALLOWED_STAGE
        if bad:
            raise ValueError("stage %s: unknown keys %s" % (st.get("name"), sorted(bad)))
        if st["name"] in names:
            raise ValueError("duplicate stage %s" % st["name"])
        dz = st.get("design")
        if dz is not None:
            if not isinstance(dz, dict) or dz.get("kind", "grid") not in DESIGN_KINDS:
                raise ValueError("stage %s: design.kind must be one of %s" % (st["name"], ", ".join(DESIGN_KINDS)))
            if dz.get("kind") == "maximin" and not (isinstance(dz.get("per_stratum"), int) and dz["per_stratum"] >= 1):
                raise ValueError("stage %s: design.per_stratum must be an integer >= 1" % st["name"])
        cf = st.get("confirmatory")
        if cf is not None and not (isinstance(cf, dict) and _num(cf.get("fraction")) and 0 < cf["fraction"] < 1):
            raise ValueError("stage %s: confirmatory.fraction must be in (0, 1)" % st["name"])
        names.add(st["name"])
        if st["engine"] not in ("E1", "E2", "E3"):
            raise ValueError("stage %s: engine must be E1, E2 or E3" % st["name"])
        for c in _as_list(st["circuits"]):
            if c not in m["circuits"]:
                raise ValueError("stage %s: unknown circuit list %s" % (st["name"], c))
        ms = st["methods"]
        if isinstance(ms, str):
            if ms not in sets:
                raise ValueError("stage %s: unknown method set %s" % (st["name"], ms))
        else:
            for k in ms:
                if k not in m["methods"]:
                    raise ValueError("stage %s: unknown method %s" % (st["name"], k))
            if len(set(ms)) != len(ms):
                raise ValueError("stage %s lists a method twice" % st["name"])
        for k in ("train_reps", "train_reps_other"):
            if k in st and not (_num(st[k]) and int(st[k]) == st[k] and st[k] >= 2):
                raise ValueError("stage %s: %s must be an integer >= 2" % (st["name"], k))
        if "train_levels" in st and not (isinstance(st["train_levels"], (list, tuple)) and st["train_levels"] and all(
                _num(x) and x >= 1 and int(x) == x for x in st["train_levels"])):
            raise ValueError("stage %s: train_levels must be a non-empty list of integers >= 1" % st["name"])
        for ro in _as_list(st.get("readout", "on")):
            if ro not in ("on", "off"):
                raise ValueError("stage %s: readout must be the string 'on' or 'off' (quote it in YAML; unquoted "
                                 "on/off are booleans), got %r" % (st["name"], ro))
        for s in st["scales"]:
            if not (_num(s) and s > 0 and math.isfinite(s)):
                raise ValueError("stage %s: bad scale %r" % (st["name"], s))
        for chi in _as_list(st.get("coherent_fraction", 0.0)):
            if not (_num(chi) and 0 <= chi <= 1):
                raise ValueError("stage %s: coherent_fraction must be numbers in [0, 1], got %r" % (st["name"], chi))
            if chi and st["engine"] != "E1":
                raise ValueError("stage %s: a coherent fraction needs engine E1 (a fake device has none)" % st["name"])
        if st.get("readout_mit", "tensored") not in ("none", "tensored", "m3", "trex"):
            raise ValueError("stage %s: readout_mit must be none, tensored, m3 or trex" % st["name"])
        devs = _as_list(st.get("devices", dflt.get("device", "fake_fez")))
        for dv in devs:
            if dv not in FAKE_DEVICES:
                raise ValueError("stage %s: %r is not a fake device" % (st["name"], dv))
        for ol in _as_list(st.get("opt_levels", dflt.get("opt_level", 1))):
            if ol not in (0, 1, 2, 3):
                raise ValueError("stage %s: opt level %r (allowed 0..3)" % (st["name"], ol))
        lay = st["layout"] if "layout" in st else dflt.get("layout")
        if lay is not None:
            if "layout" not in st and any(dv != dflt.get("device", "fake_fez") for dv in devs):
                raise ValueError("stage %s: the default layout belongs to %s; set layout (or layout: null for each "
                                 "device's best chain) for other devices" % (st["name"], dflt.get("device", "fake_fez")))
            ns = {n for cname in _as_list(st["circuits"]) for (_, n, _) in expand_circuits(m["circuits"][cname])}
            if isinstance(lay, dict):
                missing = sorted(n for n in ns if str(n) not in {str(k) for k in lay})
                if missing:
                    raise ValueError("stage %s: the layout has no entry for n = %s (it would silently fall back to the "
                                     "best chain)" % (st["name"], missing))
                for k, v in lay.items():
                    if len(v) != int(k) or len(set(v)) != len(v):
                        raise ValueError("stage %s: layout for n = %s must list %s distinct qubits" % (st["name"], k, k))
        val = st.get("validate")
        if val:
            if "budgets" not in val:
                raise ValueError("stage %s: validate needs budgets" % st["name"])
            if "fraction" not in val:
                raise ValueError("stage %s: validate needs fraction (without it no cell would be validated)"
                                 % st["name"])
            if not (_num(val.get("reps", 2000)) and val.get("reps", 2000) >= 2):
                raise ValueError("stage %s: validate.reps must be >= 2" % st["name"])
            bad = set(val) - {"fraction", "budgets", "reps", "policies"}
            if bad:
                raise ValueError("stage %s: unknown validate keys %s" % (st["name"], sorted(bad)))
            for pol in val.get("policies", ["fixed"]):
                if pol not in ("fixed", "opt"):
                    raise ValueError("stage %s: validation policy %r (allowed: fixed, opt)" % (st["name"], pol))
            for b in val.get("budgets", []):
                if not (_num(b) and b >= 1):
                    raise ValueError("stage %s: budget %r is not a number >= 1 (in YAML write 1.0e+3, not 1.0e3, "
                                     "which PyYAML reads as a string)" % (st["name"], b))
            if not (_num(val.get("fraction", 0)) and 0 <= val.get("fraction", 0) <= 1):
                raise ValueError("stage %s: validate.fraction must be in [0, 1]" % st["name"])


def _check_budgets(b) -> None:
    """budgets: {grid: {log10_min, log10_max, points}, epsilons: [...]} — the grid of collect's ledger tables."""
    if b is None:
        return
    if not isinstance(b, dict) or set(b) - {"grid", "epsilons"}:
        raise ValueError("budgets: allowed keys are grid and epsilons")
    g = b.get("grid")
    if g is not None:
        if not isinstance(g, dict) or set(g) - {"log10_min", "log10_max", "points"}:
            raise ValueError("budgets.grid: allowed keys are log10_min, log10_max, points")
        lo, hi, n = g.get("log10_min", 3), g.get("log10_max", 8), g.get("points", 51)
        if not (_num(lo) and _num(hi) and 0 <= lo < hi <= 15):
            raise ValueError("budgets.grid: need 0 <= log10_min < log10_max <= 15")
        if not (_num(n) and int(n) == n and n >= 2):
            raise ValueError("budgets.grid.points must be an integer >= 2")
    eps = b.get("epsilons")
    if eps is not None and not (isinstance(eps, (list, tuple)) and all(_num(e) and e > 0 for e in eps)):
        raise ValueError("budgets.epsilons must be positive numbers")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


STRING_PARAMS = {"choice", "text"}


def _check_method(key: str, d: dict) -> None:
    """A method definition must name a registered estimator and only its parameters, with values of the right type:
    YAML typos (unknown keys, '1.0e3' read as a string, unquoted on/off read as booleans) are errors, not defaults."""
    from ..methods import REGISTRY
    from ..methods.base import DD, TWIRL
    if not isinstance(d, dict) or "estimator" not in d:
        raise ValueError("method %s: no estimator" % key)
    est = d["estimator"]
    if est not in REGISTRY:
        raise ValueError("method %s: unknown estimator %r (registered: %s)" % (key, est, ", ".join(sorted(REGISTRY))))
    params = {p.name: p for p in REGISTRY[est].params}
    from ..methods.base import migrate_params
    d = dict(d, **migrate_params(est, {k: v for k, v in d.items() if k != "estimator"})[0])   # legacy names
    for k, v in d.items():
        if k == "estimator":
            continue
        if k == "twirl":
            if v not in TWIRL:
                raise ValueError("method %s: twirl must be one of %s" % (key, list(TWIRL)))
            continue
        if k == "dd":
            if v not in DD:
                raise ValueError("method %s: dd must be one of %s" % (key, list(DD)))
            continue
        if k not in params:
            raise ValueError("method %s: %s has no parameter %r (parameters: %s)" % (key, est, k, ", ".join(params)))
        p = params[k]
        if p.kind == "choice":
            if v not in p.options:
                raise ValueError("method %s: %s=%r is not one of %s" % (key, k, v, list(p.options)))
        elif p.kind in ("float", "int"):
            if v is None and p.default is None:
                continue
            if not _num(v):
                raise ValueError("method %s: %s=%r must be a number (in YAML write 1.0e-4 / 1.0e+3; 1.0e3 is a "
                                 "string)" % (key, k, v))
            if p.kind == "int" and int(v) != v:
                raise ValueError("method %s: %s=%r must be an integer" % (key, k, v))
            if (p.lo is not None and v < p.lo) or (p.hi is not None and v > p.hi):
                raise ValueError("method %s: %s=%r is outside [%s, %s]" % (key, k, v, p.lo, p.hi))
        elif p.kind in ("floats", "ints"):
            if not isinstance(v, (list, tuple)) or not all(_num(x) for x in v):
                raise ValueError("method %s: %s must be a list of numbers" % (key, k))
        elif p.kind == "bool" and not isinstance(v, bool):
            raise ValueError("method %s: %s must be true/false" % (key, k))


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def expand_circuits(entries: list) -> list:
    """[(family, n, params dict)] with list-valued parameters expanded (n included)."""
    out = []
    for e in entries:
        e = dict(e)
        fam = e.pop("family")
        keys = sorted(e)
        vals = [_as_list(e[k]) for k in keys]
        for combo in itertools.product(*vals):
            d = dict(zip(keys, combo))
            n = int(d.pop("n"))
            out.append((fam, n, d))
    return out


def _method_specs(m: dict, which) -> tuple:
    keys = m["method_sets"][which] if isinstance(which, str) else list(which)
    out = []
    for k in keys:
        d = dict(m["methods"][k])
        est = d.pop("estimator")
        tw, dd = d.pop("twirl", "none"), d.pop("dd", "none")
        out.append(MethodSpec.make(k, est, tw, dd, **d))
    return tuple(out)


def circuit_seed(family: str, n: int, params: dict) -> int:
    h = hashlib.sha256(json.dumps([family, int(n), params], sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(h[:4], "little")


def _select(kws: list, fraction: float) -> set:
    """Indices of the cells to validate: in every stratum (family, n, engine-independent), the ⌈fraction·size⌉ cells
    with the smallest hash of their id. Deterministic, balanced across families and sizes, independent of order."""
    if fraction <= 0:
        return set()
    strata = {}
    for i, kw in enumerate(kws):
        strata.setdefault((kw["family"], kw["n"]), []).append(i)
    out = set()
    for idx in strata.values():
        k = min(len(idx), int(math.ceil(fraction * len(idx) - 1e-12)))
        ranked = sorted(idx, key=lambda i: hashlib.sha256(("validate|" + CellSpec(**kws[i]).cell_id).encode())
                        .hexdigest())
        out.update(ranked[:k])
    return out


def _design_vector(kw: dict) -> list:
    """Numeric coordinates of a cell inside its (family, n) stratum: its numeric family parameters, log noise scale
    and coherent fraction (each later rescaled to [0, 1] within the stratum)."""
    v = [float(x) for _, x in kw["fparams"] if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return v + [math.log(max(float(kw["scale"]), 1e-12)), float(kw["coherent_fraction"])]


def _maximin(kws: list, per_stratum: int, seed: int) -> list:
    """Indices kept by a space-filling design: in every (family, n) stratum, ``per_stratum`` cells chosen greedily
    to maximize the smallest distance to the cells already chosen (coordinates rescaled to [0, 1]; the first cell is
    the one with the smallest seeded hash of its id). Deterministic and independent of the cells' order."""
    strata = {}
    for i, kw in enumerate(kws):
        strata.setdefault((kw["family"], kw["n"]), []).append(i)
    keep = []
    for idx in strata.values():
        idx = sorted(idx, key=lambda i: hashlib.sha256(("design|%d|" % seed + CellSpec(**kws[i]).cell_id).encode())
                     .hexdigest())
        if len(idx) <= per_stratum:
            keep += idx
            continue
        X = [_design_vector(kws[i]) for i in idx]
        dim = max(len(x) for x in X)
        X = [x + [0.0] * (dim - len(x)) for x in X]
        lo = [min(x[d] for x in X) for d in range(dim)]
        hi = [max(x[d] for x in X) for d in range(dim)]
        Z = [[(x[d] - lo[d]) / (hi[d] - lo[d]) if hi[d] > lo[d] else 0.0 for d in range(dim)] for x in X]
        chosen = [0]
        dist = [math.dist(Z[0], z) for z in Z]
        while len(chosen) < per_stratum:
            j = max((k for k in range(len(idx)) if k not in chosen), key=lambda k: (dist[k], -k))
            chosen.append(j)
            dist = [min(dist[k], math.dist(Z[j], Z[k])) for k in range(len(idx))]
        keep += [idx[k] for k in chosen]
    return sorted(keep)


def confirmatory_split(m: dict) -> dict:
    """{cell id: "confirm" | "explore"} for the stages with ``confirmatory: {fraction, seed}``. The split is by base
    circuit (family, n, parameters) — every noise scale, device and readout arm of a circuit lands on the same side,
    so no circuit is seen in exploration and then 'confirmed' — and is a hash of the circuit's identity and the
    seed, fixed before any outcome exists."""
    out = {}
    for st in m["stages"]:
        conf = st.get("confirmatory")
        for c in expand(m, [st["name"]]):
            side = "explore"
            if conf:
                key = json.dumps([c.family, c.n, list(c.fparams)], sort_keys=True, default=str)
                h = int(hashlib.sha256(("confirm|%s|%s" % (conf.get("seed", 0), key)).encode()).hexdigest()[:8], 16)
                side = "confirm" if h / 0xFFFFFFFF < float(conf["fraction"]) else "explore"
            out[c.cell_id] = side
    return out


def expand(m: dict, stages=None) -> list:
    """All CellSpecs of the manifest (or of the named stages), in a deterministic order."""
    check(m)
    dflt = dict(m.get("defaults", {}))
    out = []
    for st in m["stages"]:
        if stages and st["name"] not in stages:
            continue
        methods = _method_specs(m, st["methods"])
        circs = []
        for cname in _as_list(st["circuits"]):
            circs += expand_circuits(m["circuits"][cname])
        devices = _as_list(st.get("devices", dflt.get("device", "fake_fez")))
        opts = _as_list(st.get("opt_levels", dflt.get("opt_level", 1)))
        readouts = _as_list(st.get("readout", "on"))
        chis = _as_list(st.get("coherent_fraction", 0.0))
        val = st.get("validate") or {}
        stage_kw = []
        for (fam, n, fp), dev, ol, s, ro, chi in itertools.product(circs, devices, opts, st["scales"], readouts, chis):
            lay = st.get("layout", dflt.get("layout"))
            if isinstance(lay, dict):
                lay = lay.get(str(n), lay.get(n))
            kw = dict(stage=st["name"], family=fam, n=n, fparams=tuple(sorted(fp.items())), device=dev,
                      layout=tuple(lay) if lay else None, opt_level=int(ol),
                      seed_transpiler=int(dflt.get("seed_transpiler", 11)), engine=st["engine"], scale=float(s),
                      readout=ro, readout_mit=st.get("readout_mit", dflt.get("readout_mit", "tensored")) if ro == "on"
                      else "none", coherent_fraction=float(chi), methods=methods,
                      seed=circuit_seed(fam, n, fp), s_min=int(dflt.get("s_min", 32)),
                      s_tr_min=int(dflt.get("s_tr_min", 32)),
                      train_levels=tuple(int(x) for x in st.get("train_levels", dflt.get(
                          "train_levels", (64, 256, 1024, 4096, 16384)))),
                      train_reps=int(st.get("train_reps", dflt.get("train_reps", 2000))),
                      train_reps_other=int(st.get("train_reps_other", dflt.get("train_reps_other", 500))))
            stage_kw.append(kw)
        design = st.get("design") or {"kind": "grid"}
        if design.get("kind", "grid") == "maximin":
            stage_kw = [stage_kw[i] for i in _maximin(stage_kw, int(design["per_stratum"]), int(design.get("seed", 0)))]
        chosen = _select(stage_kw, float(val.get("fraction", 0.0))) if val else set()
        for i, kw in enumerate(stage_kw):
            if i in chosen:
                out.append(CellSpec(**kw, validate_budgets=tuple(int(round(float(b))) for b in val["budgets"]),
                                    validate_reps=int(val.get("reps", 2000)),
                                    validate_policy=tuple(val.get("policies", ["fixed"]))))
            else:
                out.append(CellSpec(**kw))
    ids = [c.cell_id for c in out]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError("the manifest defines the same cell twice (ids %s …)" % dup)
    return out
