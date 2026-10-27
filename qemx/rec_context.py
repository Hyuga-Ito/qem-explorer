r"""Resolved recommendation context (``qemx.methods/2``, 1.5.1).

A study recommends a *protocol*: an estimator with its parameters, transforms, readout mitigation and calibration,
training and neighbor settings, run under one noise context and judged feasible under one cost policy. In 1.5.0 the
report's ``tables/methods.json`` held the bare manifest declarations (``{"estimator": "raw"}``); the live page rebuilt
them with the defaults (readout "none") and computed the noise features from the noise injector even when a fake
device was selected, with scale 1 (finding F02). This module exports what the study actually ran and resolves the same
context for a live experiment:

* study side — :func:`context_from_records` / :func:`study_context` write ``tables/methods.json``;
* live side — :func:`live_context` (the noise model the experiment executes with: the selected device specification
  with its scale, readout and coherent fraction, or the injector), :func:`live_features` (``qemx.features.compute``
  under that model), :func:`protocol_costs` (each study protocol's fixed cost on the live circuit and its feasibility
  under the study's cost policy);
* :func:`compatibility` — runs **before** any prediction: noise-model family and device, homogenization, readout arm,
  feature schema, budget range, noise-scale and coherent-fraction range, observable kind and measurement settings.
  Any problem → the page abstains and says why.

``methods.json`` of schema ``qemx.methods/2``::

    schema, study, created_by, feature_schema (of the tables in this report directory), candidates (ordered),
    upper_bounds, baseline, protocols {key: protocol}, noise, compile, cost_policy, training, budgets, observables,
    split, problems

Each protocol: ``recipe`` (``Recipe.to_json`` of ``qemx.phase.cell.recipe_for``: the recipe every cell ran, inherited
readout mitigation and every default filled in, checked against each record's ``readout_mit`` and ``title``),
``title``, ``manifest`` (the declaration), ``calibration``, ``training``, ``observed_cost`` (the (m, c, T, s_tr) of
the records with their counts), ``candidate``.

A ``qemx.methods/1`` file (1.5.0: bare declarations) is refused for live recommendation, with the command that
writes ``/2`` from the study's run directory (``python -m qemx.study context RUN_DIR --manifest M --out REPORT``).
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType

from . import features as FT

SCHEMA = "qemx.methods/2"
OLD_SCHEMAS = ("qemx.methods/1",)
COST_POLICY_VERSION = "qemx.phase.ledger.min_budget/1"
COST_RULE = "B_min = m + T·s_tr + c·s_min (fixed policy: the protocol's own training shots s_tr)"
READOUT_TEXT = {"none": "no readout correction", "tensored": "tensored readout-error inversion",
                "m3": "M3 readout mitigation", "trex": "TREX readout mitigation"}
MIGRATE_HINT = ("regenerate it from the study's run directory: python -m qemx.study context RUN_DIR --manifest "
                "MANIFEST --out REPORT_DIR (or rerun python -m qemx.study report)")


def _sorted(xs):
    return sorted(xs, key=lambda x: (str(type(x)), x))


def _jsonable(x):
    if isinstance(x, (MappingProxyType, dict)):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item"):
        return x.item()
    return x


# ------------------------------------------------------------------ study side
def context_from_records(records, *, study: str = "", manifest_methods: dict | None = None, candidates=None,
                         upper_bounds=(), budgets=None, feature_schema: str = FT.VERSION, policy: str = "fixed",
                         split=None, baseline: str = "RAW") -> dict:
    """The content of ``methods.json`` (schema /2) from a study's cell records (dicts as written by the cell
    runner). Every method row is resolved with ``qemx.phase.cell.recipe_for`` — the function the runner used — and
    checked against the row's recorded ``readout_mit`` and ``title``."""
    from . import __version__
    from .phase.cell import CellSpec, recipe_for
    from .phase.descriptors import observable_descriptors
    problems = []
    prot: dict = {}
    noise = dict(engines=Counter(), backends=set(), devices=set(), scales=set(), coherent_fractions=set(),
                 readout=set(), homogenized=set(), calibration_hashes=set(), qiskit_ibm_runtime=set())
    comp = dict(opt_level=set(), seed_transpiler=set(), layout=set())
    train = dict(train_levels=set(), train_reps=set(), train_reps_other=set())
    pol = dict(s_min=set(), s_tr_min=set())
    obs = dict(kinds=set(), settings=set(), groups=set())
    versions = set()
    n_cells = 0
    for rec in records:
        if rec.get("status") not in ("ok", "partial"):
            continue
        n_cells += 1
        spec = CellSpec.from_json(rec["cell"])
        by_key = {m.key: m for m in spec.methods}
        eng = rec.get("engine") or {}
        noise["engines"][spec.engine] += 1
        noise["backends"].add(eng.get("backend"))
        noise["devices"].add(spec.device)
        noise["scales"].add(float(spec.scale))
        noise["coherent_fractions"].add(float(spec.coherent_fraction))
        noise["readout"].add(spec.readout)
        noise["homogenized"].add(spec.engine == "E1")
        pv = rec.get("provenance") or {}
        noise["calibration_hashes"].add(pv.get("calibration_hash"))
        noise["qiskit_ibm_runtime"].add(pv.get("qiskit_ibm_runtime"))
        versions.add(pv.get("qemx"))
        comp["opt_level"].add(spec.opt_level)
        comp["seed_transpiler"].add(spec.seed_transpiler)
        comp["layout"].add("best chain of the device" if spec.layout is None else "fixed %s" % (list(spec.layout),))
        train["train_levels"].add(tuple(spec.train_levels))
        train["train_reps"].add(spec.train_reps)
        train["train_reps_other"].add(spec.train_reps_other)
        pol["s_min"].add(int(spec.s_min))
        pol["s_tr_min"].add(int(spec.s_tr_min))
        gk = json.dumps([spec.family, spec.n, dict(spec.fparams)], sort_keys=True, default=str)
        if gk not in obs["groups"]:
            obs["groups"].add(gk)
            try:
                od = observable_descriptors(FT.observable_of_cell(rec["cell"]))
                obs["kinds"].add(od["obs_kind"])
                obs["settings"].add(int(od["obs_settings"]))
            except Exception as e:                       # noqa: BLE001 - recorded
                problems.append("observable of %s could not be rebuilt (%s)" % (gk, type(e).__name__))
        for row in rec.get("methods") or []:
            key = row["key"]
            ms = by_key.get(key)
            if ms is None:
                problems.append("cell %s: method row %s is not in the cell specification" % (rec["cell_id"], key))
                continue
            rc = recipe_for(ms, spec, {})
            if row.get("readout_mit") is not None and row["readout_mit"] != rc.readout:
                problems.append("cell %s, %s: the record ran readout %r but recipe_for resolves %r (code changed "
                                "since the study ran)" % (rec["cell_id"], key, row["readout_mit"], rc.readout))
            if row.get("title") is not None and row["title"] != rc.title():
                problems.append("cell %s, %s: recorded title %r differs from the resolved recipe's %r"
                                % (rec["cell_id"], key, row["title"], rc.title()))
            p = prot.setdefault(key, dict(variants={}, cost=Counter(), cells=0, ms=ms, rc={}))
            p["variants"][rc.key()] = p["variants"].get(rc.key(), 0) + 1
            p["rc"][rc.key()] = rc
            p["cells"] += 1
            if row.get("status") == "ok":
                p["cost"][(int(row.get("m") or 1), int(row.get("c") or 0), int(row.get("T") or 0),
                           int(row.get("s_tr") or 0))] += 1
    if not n_cells:
        problems.append("no completed cell records")
    keys = list(prot)
    upper = [k for k in keys if k in set(upper_bounds)]
    cand = [k for k in (candidates if candidates is not None else keys) if k in prot and k not in upper]
    for k in candidates or ():
        if k not in prot:
            problems.append("candidate %s has no record" % k)
    if len(pol["s_min"]) > 1 or len(pol["s_tr_min"]) > 1:
        problems.append("the cells use different cost-policy lower bounds (s_min %s, s_tr_min %s): one feasibility "
                        "rule cannot reproduce every training label" % (sorted(pol["s_min"]), sorted(pol["s_tr_min"])))
    protocols = {}
    for k, p in prot.items():
        if len(p["variants"]) > 1:
            problems.append("protocol %s ran as %d different recipes across cells (e.g. stages with different readout "
                            "mitigation); a recommendation of %s would be ambiguous" % (k, len(p["variants"]), k))
        rc = p["rc"][max(p["variants"], key=p["variants"].get)]
        ms = p["ms"]
        cost = sorted(p["cost"].items())
        T = sorted({t for (m, c, t, s), _ in cost})
        protocols[k] = dict(
            key=k, candidate=k in cand, upper_bound=k in upper, title=rc.title(), estimator=rc.estimator,
            recipe=rc.to_json(), manifest=(manifest_methods or {}).get(k), variants=len(p["variants"]),
            cells=p["cells"],
            transforms=dict(twirl=rc.twirl, dd=rc.dd),
            calibration=dict(readout_mit=rc.readout, readout_text=READOUT_TEXT.get(rc.readout, rc.readout),
                             circuits=sorted({c for (m, c, t, s), _ in cost}),
                             s_min=(sorted(pol["s_min"]) or [None])[0],
                             shots="optimal split of the remaining budget, at least s_min per circuit (ledger)"),
            training=(dict(n_train=rc.p.get("n_train"), train_shots=rc.p.get("train_shots"), T=T,
                           s_tr=sorted({s for (m, c, t, s), _ in cost}),
                           target_split=rc.p.get("target_split"), neighbors=rc.p.get("neighbors"),
                           n_neighbors=rc.p.get("n_neighbors"))
                      if any(t > 0 for t in T) or "train_shots" in rc.p else None),
            observed_cost=[dict(m=m, c=c, T=t, s_tr=s, cells=n) for (m, c, t, s), n in cost],
            declared=dict(estimator=ms.estimator, params=ms.p, twirl=ms.twirl, dd=ms.dd))
    grid = sorted(int(b) for b in (budgets or ()))
    return _jsonable(dict(
        schema=SCHEMA, study=study, created_by="qemx %s" % __version__, feature_schema=feature_schema,
        candidates=cand, upper_bounds=upper, baseline=baseline if baseline in prot else None,
        protocols=protocols,
        noise=dict(kind="device", devices=_sorted(noise["devices"]), engines=dict(noise["engines"]),
                   backends=_sorted(b for b in noise["backends"] if b), scales=_sorted(noise["scales"]),
                   coherent_fractions=_sorted(noise["coherent_fractions"]), readout=_sorted(noise["readout"]),
                   homogenized=_sorted(noise["homogenized"]),
                   calibration_hashes=_sorted(h for h in noise["calibration_hashes"] if h),
                   qiskit_ibm_runtime=_sorted(v for v in noise["qiskit_ibm_runtime"] if v),
                   noise_features="λ features under each cell's device model (backend specification with scale, "
                                  "readout and coherent fraction)"),
        compile=dict(opt_level=_sorted(comp["opt_level"]), seed_transpiler=_sorted(comp["seed_transpiler"]),
                     layout=_sorted(comp["layout"])),
        cost_policy=dict(version=COST_POLICY_VERSION, name=policy, integer=True,
                         s_min=(sorted(pol["s_min"]) or [32])[0], s_tr_min=(sorted(pol["s_tr_min"]) or [32])[0],
                         rule=COST_RULE),
        training=dict(train_levels=[list(t) for t in _sorted(train["train_levels"])],
                      train_reps=_sorted(train["train_reps"]), train_reps_other=_sorted(train["train_reps_other"])),
        budgets=dict(grid=grid, min=grid[0] if grid else None, max=grid[-1] if grid else None),
        observables=dict(kinds=_sorted(obs["kinds"]), settings_max=max(obs["settings"]) if obs["settings"] else None),
        split=split, cells=n_cells, record_qemx_versions=_sorted(v for v in versions if v), problems=problems))


def study_context(S: dict, feature_schema: str = FT.VERSION) -> dict:
    """methods.json (/2) for a loaded study (``qemx.study.load``)."""
    from .phase import store as ST
    keep = set(S["cells"]["cell_id"])
    recs = [r for r in ST.iter_cells(S["out_dir"]) if r.get("cell_id") in keep]
    man = S.get("manifest") or {}
    split = [dict(stage=s.get("name"), confirmatory=s.get("confirmatory")) for s in (man.get("stages") or [])]
    return context_from_records(recs, study=str(man.get("study", "")), manifest_methods=man.get("methods") or {},
                                candidates=S["candidates"], upper_bounds=S.get("optimistic") or (),
                                budgets=sorted(set(int(b) for b in S["ledger"]["B"])), feature_schema=feature_schema,
                                split=split)


def write_methods(path: str, ctx: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=1, sort_keys=False)
    os.replace(tmp, path)
    return path


def detect_table_schema(tables_dir: str) -> str:
    """Feature schema of an existing report's tables, from its column names (``x_routing_ratio`` exists only in
    /1 tables, ``x_two_qubit_compile_ratio`` only in /2)."""
    import csv
    for name in ("cells_features.csv", "rmse_wide.csv"):
        p = os.path.join(tables_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                cols = set(next(csv.reader(f)))
            if "x_two_qubit_compile_ratio" in cols or "x_two_qubit_per_qubit_layer" in cols:
                return FT.VERSION
            if "x_routing_ratio" in cols or "x_two_qubit_density" in cols:
                return "qemx.features/1"
    return FT.VERSION


# ------------------------------------------------------------------ loading
@dataclass(frozen=True)
class StudyContext:
    path: str
    schema: str
    data: MappingProxyType
    problems: tuple = ()

    @property
    def usable(self) -> bool:
        return not self.problems

    @property
    def candidates(self) -> list:
        return list(self.data.get("candidates") or [])

    @property
    def protocols(self) -> dict:
        return dict(self.data.get("protocols") or {})

    @property
    def baseline(self):
        return self.data.get("baseline")

    @property
    def feature_schema(self) -> str:
        return self.data.get("feature_schema") or FT.VERSION

    def policy(self):
        from .phase.ledger import Policy
        cp = self.data.get("cost_policy") or {}
        return Policy(cp.get("name", "fixed"), bool(cp.get("integer", True)), int(cp.get("s_min", 32)),
                      int(cp.get("s_tr_min", 32)))

    def recipe(self, key: str):
        from .methods import Recipe
        return Recipe.from_json(self.protocols[key]["recipe"])


def load_study(report_dir: str) -> StudyContext:
    """The study context of a report directory (``tables/methods.json``). Schema /1 or an unreadable file gives a
    context with ``problems`` (the page then refuses a live recommendation and shows them)."""
    path = os.path.join(report_dir, "tables", "methods.json")
    if not os.path.exists(path):
        return StudyContext(path, "", MappingProxyType({}), ("the report has no tables/methods.json; " + MIGRATE_HINT,))
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:                               # noqa: BLE001 - reported
        return StudyContext(path, "", MappingProxyType({}), ("tables/methods.json is unreadable (%s)" % e,))
    if not isinstance(d, dict):
        return StudyContext(path, "", MappingProxyType({}), ("tables/methods.json is not an object",))
    schema = d.get("schema")
    if schema is None:                                   # 1.5.0: {key: bare manifest declaration}
        data = dict(candidates=list(d), protocols={}, schema="qemx.methods/1")
        return StudyContext(path, "qemx.methods/1", MappingProxyType(data), (
            "this study's methods.json is schema qemx.methods/1 (QEM Explorer 1.5.0): bare method declarations "
            "without the readout mitigation, calibration, noise model, feature schema and cost policy the study "
            "actually used, so a live recommendation could name a different protocol than the one measured; "
            + MIGRATE_HINT,))
    if schema != SCHEMA:
        return StudyContext(path, str(schema), MappingProxyType(dict(d)),
                            ("unknown methods.json schema %r (this code reads %s)" % (schema, SCHEMA),))
    probs = list(d.get("problems") or [])
    for k in d.get("candidates") or []:
        if k not in (d.get("protocols") or {}):
            probs.append("candidate %s has no resolved protocol" % k)
    return StudyContext(path, schema, MappingProxyType(dict(d)), tuple(probs))


# ------------------------------------------------------------------ live side
@dataclass(frozen=True)
class LiveContext:
    compiled: object
    observable: object
    backend: str
    kind: str                         # "device" | "injector"
    noise_model: object               # device specification (str) or NoiseSpec: what the experiment executes with
    device: str | None = None         # base fake device
    modifiers: dict = field(default_factory=dict)
    scale: float | None = None
    coherent_fraction: float | None = None
    readout_arm: str | None = None    # "on" | "off" | "×r" (device readout errors scaled by r)
    homogenized: bool | None = None
    engine: str | None = None
    layout: tuple = ()
    context_reason: str = ""

    def method_noise(self):
        """The NoiseSpec the methods may assume (PEC rates, TREX): the device's equivalent model, as compare() and
        the cell runner use; the injector itself otherwise."""
        if self.kind == "device":
            from . import device
            return device.get_model(self.backend, tuple(self.layout)).equivalent_noise()
        return self.noise_model

    def runner(self):
        from .methods import Runner
        return Runner(self.backend, layout=self.layout if self.kind == "device" else None)

    def summary(self) -> dict:
        return _jsonable(dict(backend=self.backend, kind=self.kind, device=self.device, scale=self.scale,
                              coherent_fraction=self.coherent_fraction, readout=self.readout_arm,
                              homogenized=self.homogenized, engine=self.engine, layout=list(self.layout)))


def live_context(compiled, backend: str, injector_noise, observable) -> LiveContext:
    """The resolved noise context of a live experiment: in device mode the selected device specification (its
    scale s, readout factor ro, homogenization and coherent fraction χ) — never the inactive injector; otherwise the
    injector's NoiseSpec (no device scale or coherent fraction)."""
    from . import device
    layout = tuple(int(p) for p in (compiled.info.get("phys") or ()))
    if device.is_device(backend):
        base, mods = device.parse_device(backend)
        ro = float(mods["ro"])
        arm = "on" if ro == 1.0 else ("off" if ro == 0.0 else "×%g" % ro)
        return LiveContext(compiled, observable, backend, "device", backend, base, dict(mods), float(mods["s"]),
                           float(mods["chi"]), arm, bool(mods["hom"]), str(mods["eng"]), layout, "")
    _, _, why = FT.device_context(injector_noise)
    return LiveContext(compiled, observable, str(backend), "injector", injector_noise, None, {}, None, None, None,
                       None, str(backend), layout, why)


def live_features(ctx: LiveContext, budget: float, names=None, state_max_qubits: int = FT.PROFILE_MAX_QUBITS):
    """(values, missing) of the requested features under the resolved noise context."""
    return FT.compute(ctx.compiled, ctx.observable, ctx.noise_model, budget, scale=ctx.scale,
                      coherent_fraction=ctx.coherent_fraction, names=names, state_max_qubits=state_max_qubits,
                      context_reason=ctx.context_reason)


def _shot_constraints(rc) -> tuple:
    """(shots per target circuit, post-selected?) of a recipe, as the study's ledger rows record them
    (``shots_per_circuit`` and ``accept``, phase/cell.py): PEC with s shots per sampled circuit, a chain with an inner
    PEC, and symmetry verification (alone or as a chain's inner method)."""
    from .methods import REGISTRY
    m = REGISTRY[rc.estimator]
    p = dict(m.defaults(), **rc.p) if hasattr(m, "defaults") else dict(rc.p)
    if rc.estimator == "pec":
        return max(int(p.get("shots_per_circuit", 1) or 1), 1), False
    if rc.estimator == "chain":
        if p.get("inner") == "pec":
            return max(int(p.get("pec_shots_per_circuit", 1) or 1), 1), False
        return 1, p.get("inner") == "symmetry"
    return 1, rc.estimator == "symmetry"


def protocol_costs(study: StudyContext, ctx: LiveContext, budget: int, keys=None, symmetry=None) -> dict:
    """{key: cost facts} of each study protocol on the live circuit: the resolved recipe, its calibration circuits
    (``Context.cal_circuits``), training shots and target circuits (the method's preflight ``min_cost``), and the
    smallest budget under the study's cost policy (``qemx.phase.ledger.min_budget``, the rule of the training labels).
    Target circuits are a lower bound when the method has no preflight count (ZNE: its folds; NIL with an optimal
    target split: the circuits that receive shots depend on the fitted coefficients).

    The same feasibility constraints as the study's ledger rows (F10, ``phase.cell.cost_of``): the shots one target
    circuit needs (PEC's shots per sampled circuit) and, for post-selected protocols, the acceptance — measured by
    running the protocol once on the live circuit with ``symmetry`` (the experiment's), as the study's cells did."""
    from .methods import REGISTRY, Context
    from .phase.ledger import Cost, min_budget
    pol = study.policy()
    out = {}
    runner = ctx.runner()
    noise = ctx.method_noise()
    for key in (keys or study.candidates):
        try:
            rc = study.recipe(key)
            mctx = Context(ctx.compiled, noise, ctx.observable, rc, runner, shots=int(budget), symmetry=symmetry)
            c = int(mctx.cal_circuits())
            mc = REGISTRY[rc.estimator].min_cost(mctx, **rc.p)
            F = int(mc["training"]) if mc else 0
            m = max(int(mc.get("target_circuits", 1)), 1) if mc else 1
            s_tr = int(rc.p.get("train_shots", 0) or 0) if F else 0
            T = F // s_tr if s_tr else 0
            if s_tr and T * s_tr != F:
                raise ValueError("training shots %d are not T × %d" % (F, s_tr))
            per_circuit, post = _shot_constraints(rc)
            accept = 1.0
            if post:                                       # the acceptance on this circuit, as the cells measured it
                res = REGISTRY[rc.estimator].run(mctx, **rc.p)
                a_ = getattr(res, "accept", None)
                accept = 1.0 if a_ is None or not math.isfinite(float(a_)) else min(max(float(a_), 0.0), 1.0)
            mb = int(min_budget(Cost(key, b=0.0, v=0.0, c=c, T=T, s_tr=s_tr, m=m, accept=accept,
                                     per_circuit=per_circuit), pol))
            out[key] = dict(key=key, title=rc.title(), readout=rc.readout, recipe=rc.to_json(),
                            calibration_circuits=c, training_shots=F, T=T, s_tr=s_tr, target_circuits=m,
                            shots_per_circuit=per_circuit, accept=accept if post else None,
                            min_budget=mb, feasible=bool(int(budget) >= mb),
                            policy="%s (s_min %d)" % (pol.name, pol.s_min), error="")
        except Exception as e:                           # noqa: BLE001 - this protocol cannot be costed
            out[key] = dict(key=key, feasible=False, min_budget=None, error="%s: %s" % (type(e).__name__, e))
    return out


def compatibility(study: StudyContext, ctx: LiveContext, budget: float, names=None, columns=None,
                  live_values: dict | None = None) -> tuple:
    """(problems, notes) — run before predicting. Any problem means the study's model does not describe this
    experiment's protocol or noise context: abstain and show the problems."""
    problems, notes = list(study.problems), []
    d = study.data
    N = d.get("noise") or {}
    if N.get("kind") == "device" and N.get("devices"):
        devs = list(N.get("devices") or [])
        if ctx.kind != "device":
            problems.append("the study simulated on the fake device model %s (backends %s); this experiment runs the "
                            "noise injector, so its noise features and predicted errors would refer to another noise "
                            "model — select the %s backend in the sidebar" % (", ".join(devs),
                                                                              ", ".join(N.get("backends") or []),
                                                                              devs[0]))
        else:
            if ctx.device not in devs:
                problems.append("the study used the device model %s; this experiment uses %s (transfer across "
                                "devices was not studied)" % (", ".join(devs), ctx.device))
            if N.get("homogenized") and ctx.homogenized not in N["homogenized"]:
                problems.append("the study's device models are %s; this one is %s" % (
                    " / ".join("homogenized (E1)" if h else "the device's own calibration" for h in N["homogenized"]),
                    "homogenized" if ctx.homogenized else "the device's own calibration"))
            if N.get("readout") and ctx.readout_arm not in N["readout"]:
                problems.append("the study's readout arm is %s; this experiment's device readout is %s"
                                % (" / ".join(N["readout"]), ctx.readout_arm))
            sc = [float(x) for x in N.get("scales") or []]
            if sc and not (min(sc) - 1e-12 <= float(ctx.scale) <= max(sc) + 1e-12):
                problems.append("noise scale s = %g lies outside the studied scales %g–%g" % (ctx.scale, min(sc),
                                                                                           max(sc)))
            cf = [float(x) for x in N.get("coherent_fractions") or []]
            if cf and not (min(cf) - 1e-12 <= float(ctx.coherent_fraction) <= max(cf) + 1e-12):
                problems.append("coherent fraction χ = %g lies outside the studied %g–%g" % (ctx.coherent_fraction,
                                                                                          min(cf), max(cf)))
            if ctx.engine != "exact":
                notes.append("the study computed exact moments (engine exact); this backend executes the same noise "
                             "model with %s" % ctx.engine)
            try:
                import qiskit_ibm_runtime as _r
                rv = getattr(_r, "__version__", None)
            except Exception:                            # noqa: BLE001 - informational
                rv = None
            if N.get("qiskit_ibm_runtime") and rv and rv not in N["qiskit_ibm_runtime"]:
                notes.append("the fake-device calibration snapshot comes from qiskit-ibm-runtime %s here and %s in the "
                             "study" % (rv, ", ".join(N["qiskit_ibm_runtime"])))
    B = d.get("budgets") or {}
    if B.get("min") is not None and not (float(B["min"]) <= float(budget) <= float(B["max"])):
        problems.append("budget B = %s lies outside the study's budgets %s–%s" % (
            f"{int(budget):,}", f"{int(B['min']):,}", f"{int(B['max']):,}"))
    if names is not None and columns is not None:
        _, nts, prb = FT.resolve_columns(columns, names, study.feature_schema)
        notes += nts
        problems += prb
    elif study.feature_schema not in (FT.VERSION,) + FT.PREVIOUS:
        problems.append("the study's feature schema %s is unknown to this code (%s)" % (study.feature_schema,
                                                                                    FT.VERSION))
    O = d.get("observables") or {}
    od = None
    try:
        from .phase.descriptors import observable_descriptors
        od = observable_descriptors(ctx.observable)
    except Exception:                                    # noqa: BLE001 - the feature step reports it
        od = None
    if od is not None and O.get("kinds") and od["obs_kind"] not in O["kinds"]:
        problems.append("the study measured %s observables only; this one is a %s with %d measurement setting(s)"
                        % (" / ".join(O["kinds"]), od["obs_kind"], od["obs_settings"]))
    elif od is not None and O.get("settings_max") is not None and od["obs_settings"] > int(O["settings_max"]):
        problems.append("this observable needs %d measurement settings; the study's needed at most %d"
                        % (od["obs_settings"], int(O["settings_max"])))
    return problems, notes


def describe_protocol(study: StudyContext, key: str, cost: dict | None = None) -> str:
    """One sentence naming the exact protocol: estimator, readout mitigation, calibration, transforms, training."""
    p = study.protocols.get(key)
    if not p:
        return key
    r = p["recipe"]
    parts = ["%s = %s" % (key, p["title"])]
    est = r["estimator"]
    if est == "raw":
        parts.append("no QEM estimator" + (", but readout-corrected by %s" % READOUT_TEXT.get(r["readout"],
                                                                                            r["readout"])
                                           if r["readout"] != "none" else ", fully unmitigated (no readout "
                                                                         "correction)"))
    else:
        parts.append("readout: %s" % READOUT_TEXT.get(r["readout"], r["readout"]))
    c = (cost or {}).get("calibration_circuits")
    if r["readout"] != "none":
        cp = study.data.get("cost_policy") or {}
        parts.append("%s calibration circuits, at least %s shots each" % (c if c is not None else
                                                                          "/".join(map(str, p["calibration"]["circuits"])),
                                                                          cp.get("s_min", "?")))
    if r.get("twirl", "none") != "none" or r.get("dd", "none") != "none":
        parts.append("transforms: twirl %s, DD %s" % (r.get("twirl"), r.get("dd")))
    t = p.get("training")
    if t and (cost or {}).get("training_shots", 1):
        parts.append("training: %s circuits × %s shots" % (t.get("n_train"), t.get("train_shots"))
                     + (" (%s target split)" % t["target_split"] if t.get("target_split") else ""))
    return "; ".join(parts)
