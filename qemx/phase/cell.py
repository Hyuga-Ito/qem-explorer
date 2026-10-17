r"""One grid coordinate of a phase diagram: compile, measure λ, run every method, extract (b, v, w, u, …).

A **cell** is (circuit family and parameters, n, device and layout, transpiler settings, noise engine, noise scale
s, readout arm, coherent fraction, recipes). :func:`cell_runner` returns one JSON-serializable record:

* ``cell``        the specification (and its id: a hash of the canonical JSON)
* ``circuit``     compiled-circuit facts (physical qubits, two-qubit gates, depth, measurement map)
* ``lambda``      λ_LC, λ_LC^exact, λ, λ_ro and their breakdown (``qemx.phase.lightcone``)
* ``descriptors`` entanglement and structure of the ideal circuit (``qemx.phase.descriptors``)
* ``ideal``, ``noisy``  the ideal value; the unmitigated noisy value, state fidelity and purity
* ``methods``     per method: the ledger primitives b, v, w, c, T, s_tr, u, u1, m, a (``qemx.phase.ledger.Cost``)
                  plus γ, circuits, flags and method details
* ``guards``      the fold-survival check of the ZNE circuits
* ``validation``  Monte-Carlo checks of the ledger at selected budgets (``qemx.phase.validate``), if requested
* ``provenance``  package versions, calibration hash, seeds, timing

``status`` is "ok" when every method and every validation check completed, "partial" when the cell completed but
some method or check failed (the failures are in the record; ``run --retry-errors`` re-runs the cell). A time limit,
an out-of-memory error or an interrupt is never recorded inside a cell: it aborts the cell, and the runner records
the cell as "timeout" or "oom".

Exactness. Engine ``E2`` (fake device, ``eng=exact``) and ``E1`` (NoiseSpec) compute every mean from exact
probabilities, PEC included (signed maps applied exactly). Engine ``E3`` (fake device on Aer) estimates PEC's mean
from sampled instances and records that Monte Carlo error in ``details.mc_err``; it is a cross-check only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

ENGINES = {"E1": "fake device homogenized to the chain's medians, coherent fraction χ, scaled, exact engine",
           "E2": "fake device, scaled, exact superoperator engine",
           "E3": "fake device, scaled, Qiskit Aer (cross-check)"}


@dataclass(frozen=True)
class MethodSpec:
    key: str                          # name in the manifest, e.g. "ZNE-R"
    estimator: str                    # registry key: raw | zne | pec | learning | nil | vd
    params: tuple = ()                # sorted (name, value) pairs
    twirl: str = "none"
    dd: str = "none"

    @property
    def p(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params}

    @staticmethod
    def make(key, estimator, twirl="none", dd="none", **params) -> "MethodSpec":
        return MethodSpec(key, estimator, tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                                                       for k, v in params.items())), twirl, dd)

    def to_json(self) -> dict:
        return dict(key=self.key, estimator=self.estimator, params=self.p, twirl=self.twirl, dd=self.dd)


@dataclass(frozen=True)
class CellSpec:
    stage: str
    family: str
    n: int
    fparams: tuple                    # sorted (name, value) pairs of the family
    device: str = "fake_fez"          # base device: compilation, layout, calibration
    layout: tuple | None = None       # initial layout (None: the device's best chain of n qubits)
    opt_level: int = 1
    seed_transpiler: int = 11
    engine: str = "E2"                # E1 | E2 | E3
    scale: float = 1.0                # s: gate errors and relaxation rates × s
    readout: str = "on"               # on: the device's readout errors; off: none
    readout_mit: str = "tensored"     # readout mitigation of every method (none | tensored | m3 | trex)
    coherent_fraction: float = 0.0    # E1 only: χ
    methods: tuple = ()               # MethodSpec, …
    seed: int = 0                     # method seed (training sets): the same for every scale of a circuit
    validate_budgets: tuple = ()      # budgets for the Monte-Carlo check (empty: none)
    validate_reps: int = 2000
    validate_policy: tuple = ("fixed",)
    s_min: int = 32                   # ledger: minimum shots per calibration circuit
    s_tr_min: int = 32                # ledger (first-order opt path): minimum shots per training circuit
    train_levels: tuple = (64, 256, 1024, 4096, 16384)   # learning: training shots at which b and u are measured
    train_reps: int = 2000            # learning: Monte-Carlo replicates at the protocol's training level
    train_reps_other: int = 500       # … and at every other level

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["fparams"] = dict(self.fparams)
        d["layout"] = list(self.layout) if self.layout else None
        d["methods"] = [m.to_json() for m in self.methods]
        d["validate_budgets"] = [int(b) for b in self.validate_budgets]
        d["validate_policy"] = list(self.validate_policy)
        d["train_levels"] = [int(x) for x in self.train_levels]
        return d

    @property
    def cell_id(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True, default=str).encode()).hexdigest()[:20]

    @staticmethod
    def from_json(d: dict) -> "CellSpec":
        d = dict(d)
        d["fparams"] = tuple(sorted(d["fparams"].items()))
        d["layout"] = tuple(d["layout"]) if d.get("layout") else None
        d["methods"] = tuple(MethodSpec.make(m["key"], m["estimator"], m.get("twirl", "none"), m.get("dd", "none"),
                                             **m.get("params", {})) for m in d["methods"])
        d["validate_budgets"] = tuple(int(b) for b in d.get("validate_budgets", ()))
        d["validate_policy"] = tuple(d.get("validate_policy", ("fixed",)))
        if "train_levels" in d:
            d["train_levels"] = tuple(int(x) for x in d["train_levels"])
        return CellSpec(**{k: v for k, v in d.items() if k in CellSpec.__dataclass_fields__})


# ------------------------------------------------------------------ caches (per process)
_COMPILED: dict = {}


def compiled_for(spec: CellSpec):
    """(FamilyCircuit, Compiled) — cached: every scale, engine and readout arm of a circuit shares one compilation."""
    from . import families as FAM
    key = (spec.family, spec.n, spec.fparams, spec.device, spec.layout, spec.opt_level, spec.seed_transpiler)
    hit = _COMPILED.get(key)
    if hit is None:
        fc = FAM.build(spec.family, spec.n, **dict(spec.fparams))
        comp = FAM.compile_family(fc, spec.device, opt_level=spec.opt_level, seed=spec.seed_transpiler,
                                  initial_layout=spec.layout)
        hit = (fc, comp)
        if len(_COMPILED) > 512:
            _COMPILED.clear()
        _COMPILED[key] = hit
    return hit


def noise_for(spec: CellSpec, comp) -> tuple:
    """(backend name, NoiseSpec handed to compare(), noise model for λ, extra facts).

    E2/E3: the fake device with gate errors and relaxation rates × s (exact engine / Aer).
    E1: the same device homogenized to the medians of the chain (every qubit and every gate of a kind alike) with a
    coherent fraction χ of each gate's non-relaxation error, on the exact engine. It runs the *same* schedule as E2,
    so its idle noise is the device's, and its gate infidelities equal the medians exactly for every χ. The twirled
    fault probability of a gate is fixed by its infidelity, so λ (all locations) does not depend on χ: χ changes the
    coherence of the noise at fixed λ. λ_LC may differ slightly (a few %) for local observables, because the light-cone
    marginal of a gate error depends on how its Pauli weight is distributed (a ZZ phase always acts on both qubits)."""
    from .. import device
    from ..noise import NoiseSpec
    from .scaling import calibration_from_model, device_at
    if spec.readout not in ("on", "off"):
        raise ValueError("readout must be 'on' or 'off'")
    ro = 1.0 if spec.readout == "on" else 0.0
    phys = tuple(comp.info["phys"])
    if spec.engine in ("E2", "E3"):
        if spec.coherent_fraction:
            raise ValueError("coherent_fraction needs engine E1")
        be = device_at(spec.device, spec.scale, ro, "exact" if spec.engine == "E2" else "aer")
        return be, NoiseSpec(), be, {}
    if spec.engine == "E1":
        be = device.device_spec(spec.device, s=spec.scale, ro=ro, eng="exact", hom=True,
                                chi=float(spec.coherent_fraction))
        cal = calibration_from_model(device.get_model(device.device_spec(spec.device, hom=True), phys),
                                     "%s on %s, homogenized" % (spec.device, list(phys)))
        return be, NoiseSpec(), be, dict(calibration=cal.to_json())
    raise ValueError("unknown engine %r" % spec.engine)


def recipe_for(ms: MethodSpec, spec: CellSpec, extra: dict):
    """The v1.2 Recipe of a method in this cell. On every engine PEC's depolarizing model comes from the (scaled)
    device model's median rates, p = r·d/(d−1) (``DeviceModel.equivalent_noise``); VD has no readout mitigation."""
    from ..methods import Recipe
    ro = "none" if ms.estimator == "vd" else spec.readout_mit
    return Recipe.make(ms.estimator, ms.twirl, ms.dd, ro, **ms.p)


def fold_guard(comp, backend: str, scales=(1.0, 3.0, 5.0), folding: str = "global", seed: int = 0) -> dict:
    """ZNE folds must reach the device unaltered. Checks (1) the folded circuits' two-qubit count is exactly λ_j times
    the base count; (2) on a device engine, the executed native circuit has exactly the folded circuit's two-qubit
    gates (the per-gate translation must not merge or cancel anything). Raises on failure; returns the counts."""
    from .. import device
    from ..transforms import achieved_lambda, fold_global, fold_local
    base = comp.implemented
    n2 = base.count("2q")
    out = dict(base_2q=n2, scales=[], spec_2q=[], native_2q=[])
    for lam in scales:
        fs = fold_global(base, lam) if folding == "global" else fold_local(base, lam, seed)
        got = fs.count("2q")
        out["scales"].append(float(achieved_lambda(base, fs)))
        out["spec_2q"].append(int(got))
        if folding == "global" and abs(lam - round(lam)) < 1e-12 and int(round(lam)) % 2 == 1 and got != n2 * lam:
            raise AssertionError("fold λ=%g has %d two-qubit gates, expected %d" % (lam, got, n2 * lam))
        if device.is_device(backend):
            from .. import device_exact
            dm = device.get_model(backend, tuple(comp.info["phys"]))
            instr, _ = device_exact.instructions(fs, dm)
            nat = sum(1 for i in instr if i.source == "gate" and len(i.qubits) == 2)
            out["native_2q"].append(int(nat))
            if nat != got:
                raise AssertionError("fold λ=%g: %d two-qubit gates in the circuit, %d executed" % (lam, got, nat))
    return out


def _calibration_hash(spec: CellSpec, phys) -> str:
    from .. import device
    dm = device.get_model(spec.device, tuple(phys))
    blob = dict(errors=sorted((k[0], list(k[1]), v) for k, v in dm.errors.items()),
                durations=sorted((k[0], list(k[1]), v) for k, v in dm.durations.items()),
                qubits=[dataclasses.asdict(q) for q in dm.qubits], phys=list(dm.phys))
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=float).encode()).hexdigest()[:16]


def provenance() -> dict:
    import importlib
    out = dict(python=platform.python_version(), platform=platform.platform())
    for mod in ("qemx", "qiskit", "qiskit_aer", "qiskit_ibm_runtime", "numpy", "scipy", "sklearn"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:                                    # noqa: BLE001 - recorded as missing
            out[mod] = None
    return out


def _f(x):
    """JSON-safe float (nan/inf as strings would break numeric columns; they become None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _method_row(ms: MethodSpec, rc, res, ideal: float) -> dict:
    from ..budget import w_cal
    row = dict(key=ms.key, estimator=ms.estimator, params=ms.p, twirl=ms.twirl, dd=ms.dd, readout_mit=rc.readout,
               title=rc.title(), status="ok")
    mean = float(res.mean)
    finite = math.isfinite(mean)
    T = int(res.train_circuits or 0)
    F = int(res.fixed_shots or 0)
    s_tr = (F // T) if T else 0
    if T and s_tr * T != F:
        raise AssertionError("training shots %d are not T × s_tr" % F)
    u = float(res.var_fixed or 0.0)
    u = max(u, 0.0) if math.isfinite(u) else math.nan          # max(nan, 0) would be nan anyway; keep it visible
    a = ()
    det = res.details or {}
    if ms.estimator == "zne" and finite:
        eta, sig = np.asarray(det.get("eta"), float), np.asarray(det.get("sigma"), float)
        a = tuple(float(x) for x in np.abs(eta) * sig)
        if abs(sum(a) ** 2 - float(res.var1)) > 1e-9 * max(1.0, float(res.var1)):
            raise AssertionError("ZNE var1 differs from (Σ|η|σ)²")
        if det.get("a_circuits") is not None and len(det["a_circuits"]) != len(a):    # averaged fold draws: one
            a = tuple(float(x) for x in det["a_circuits"])                            # weight per circuit
    row.update(mean=_f(mean), b=_f(mean - ideal) if finite else None, v=_f(res.var1), w=_f(w_cal(res)),
               c=int(res.cal_circuits or 0), T=T, s_tr=s_tr, F=F, u=_f(u), u1=_f(u * s_tr) if T else 0.0,
               m=int(res.circuits or 1), a=list(a), gamma=_f(res.gamma), circuits=int(res.circuits or 0),
               shots_per_circuit=res.shots_per_circuit, extra_qubits=int(res.extra_qubits or 0),
               accept=_f(res.accept), notes=list(res.notes or []))
    if not finite or not math.isfinite(u) or not math.isfinite(float(res.var1)):
        row["status"] = "nonfinite"
    # the statistical contract (qemx/stats.py, record schema /3): what mean/b refer to
    row["estimate_kind"] = getattr(res, "estimate_kind", "exact")
    ref = getattr(res, "reference_mean", None)
    row["reference_mean"] = _f(ref) if ref is not None else None
    mom = getattr(res, "moments", None)
    if mom:
        row["moments"] = {k: (_f(v) if isinstance(v, float) else v) for k, v in mom.items()}
    flags = list(getattr(res, "flags", ()) or ())
    if ms.estimator == "zne":
        lf = det.get("lams_fault")
        row["details"] = dict(lams=[float(x) for x in det.get("lams", [])], E=[_f(x) for x in det.get("E", [])],
                              eta=[_f(x) for x in np.atleast_1d(det.get("eta", []))],
                              sigma=[_f(x) for x in det.get("sigma", [])], rule=det.get("rule"),
                              folding=det.get("folding"), lam_est=det.get("lam_est", "gates"),
                              subset=det.get("subset"), draws=[int(x) for x in det.get("draws", [])],
                              E_draws=[[_f(x) for x in g] for g in det.get("E_draws", [])],
                              lams_gates=[float(x) for x in det.get("lams_gates", det.get("lams", []))],
                              lams_fault=None if lf is None else [_f(x) for x in lf],
                              depths=[int(x) for x in det.get("depths", [])],
                              twoq=[int(x) for x in det.get("twoq", [])])
        if det.get("rule") == "exponential":
            flags.append("nonlinear")
    elif ms.estimator == "learning":
        coef = np.asarray(det.get("coef", []), float)
        row["details"] = dict(degenerate=bool(det.get("degenerate")), rank=int(det.get("rank", 0)),
                              n_features=int(det.get("n_features", 0)), spread=_f(det.get("spread")),
                              spread_needed=_f(det.get("spread_needed")), keep_used=det.get("keep_used"),
                              sampling=det.get("sampling"), nonzero=int(np.sum(np.abs(coef) > 1e-12)),
                              l1=_f(np.abs(coef).sum()), warning=det.get("warning", ""),
                              y_train=[_f(x) for x in np.atleast_1d(det.get("y_train", []))])
        flags.append("nonlinear")
        if det.get("degenerate"):
            flags.append("degenerate")
    elif ms.estimator == "nil":
        coef = np.asarray(det.get("coef", []), float)
        row["details"] = dict(degenerate=bool(det.get("degenerate")), rank=int(det.get("rank", 0)),
                              n_features=int(det.get("n_features", 0)), n_train=int(det.get("n_train", 0)),
                              model=det.get("model"), ensemble=det.get("ensemble"),
                              target_split=det.get("target_split"), nonzero=int(np.sum(np.abs(coef) > 0)),
                              l1=_f(np.abs(coef).sum()), lam=_f(det.get("lam")),
                              solver_ok=bool(det.get("solver_ok", True)), spread=_f(det.get("spread")),
                              warning=det.get("warning", ""), names=list(det.get("names", [])),
                              coef=[_f(x) for x in coef], intercept=_f(det.get("intercept", 0.0)),
                              sigma=[_f(x) for x in np.atleast_1d(det.get("sigma", []))],
                              y_train=[_f(x) for x in np.atleast_1d(det.get("y_train", []))])
        flags.append("nonlinear")
        if det.get("degenerate"):
            flags.append("degenerate")
        if not det.get("solver_ok", True):
            flags.append("solver")
    elif ms.estimator == "chain":
        outer, inner = det.get("outer"), det.get("inner")
        row["details"] = dict(outer=outer, inner=inner, inner_recipe=det.get("inner_recipe"),
                              degenerate=bool(det.get("degenerate")),
                              accept=[_f(x) for x in np.atleast_1d(det.get("accept", []))])
        if outer == "zne" and finite and det.get("eta") is not None:
            eta, sig = np.asarray(det.get("eta"), float), np.asarray(det.get("sigma"), float)
            a = tuple(float(x) for x in np.abs(eta) * sig)
            if abs(sum(a) ** 2 - float(res.var1)) > 1e-9 * max(1.0, float(res.var1)):
                raise AssertionError("chain ZNE var1 differs from (Σ|η|σ)²")
            row["a"] = list(a) if len(a) == int(row["m"]) else []     # per-circuit split only when one circuit each
            lf = det.get("lams_fault")
            row["details"].update(lams=[float(x) for x in det.get("lams", [])], rule=det.get("rule"),
                                  folding=det.get("folding"), lam_est=det.get("lam_est", "gates"),
                                  subset=det.get("subset"),
                                  lams_gates=[float(x) for x in det.get("lams_gates", det.get("lams", []))],
                                  lams_fault=None if lf is None else [_f(x) for x in lf],
                                  depths=[int(x) for x in det.get("depths", [])],
                                  twoq=[int(x) for x in det.get("twoq", [])])
            if det.get("rule") == "exponential":
                flags.append("nonlinear")
        elif outer in ("cdr", "nil"):
            coef = np.asarray(det.get("coef", []), float)
            row["details"].update(model=det.get("model"), n_features=int(det.get("n_features", 0)),
                                  n_train=int(det.get("n_train", 0)), nonzero=int(np.sum(np.abs(coef) > 0)),
                                  l1=_f(np.abs(coef).sum()) if coef.size else None,
                                  solver_ok=bool(det.get("solver_ok", True)))
            flags.append("nonlinear")
            if det.get("degenerate"):
                flags.append("degenerate")
        if inner == "vd":
            flags += ["optimistic", "gaussian-sampler"]
        if inner == "pec" and int((det.get("inner_recipe") or {}).get("params", {}).get("shots_per_circuit", 1)) > 1:
            flags.append("gaussian-sampler")
        if inner == "symmetry":
            flags.append("nonlinear")
    elif ms.estimator == "pec":
        gp = det.get("gamma_path")
        row["details"] = dict(G=_f(gp[-1]) if gp is not None and len(gp) else None, mc_err=_f(det.get("mc_err", 0.0)),
                              exact=bool(det.get("p_net") is not None))
        if det.get("p_net") is None:
            flags.append("mc-mean")
    elif ms.estimator == "vd":
        row["details"] = dict(trace=_f(det.get("trace")), dominant_value=_f(det.get("dominant_value")),
                              M=int(det.get("M", 2)))
        flags += ["nonlinear", "optimistic", "gaussian-sampler"]
    row["flags"] = flags
    return row


def cost_of(row: dict):
    """ledger.Cost from a method row (None if the method produced no finite estimate). Learning methods carry their
    Monte-Carlo training levels (b, u, v and w at each number of training shots; levels flagged ``ok: false`` are
    left out, so no policy can choose them); the first-order u1 is used only when no levels were measured."""
    from .ledger import Cost
    if row.get("status") != "ok" or row.get("b") is None or row.get("v") is None:
        return None
    levels = tuple(level_tuple(l, row) for l in (row.get("levels") or ()) if l.get("ok", True))
    return Cost(row["key"], b=row["b"], v=row["v"], w=row.get("w") or 0.0, c=int(row.get("c") or 0),
                T=int(row.get("T") or 0), s_tr=int(row.get("s_tr") or 0), u1=row.get("u1") or 0.0,
                u_extra=0.0, m=int(row.get("m") or 1), a=tuple(row.get("a") or ()), levels=levels)


def level_tuple(l: dict, row: dict) -> tuple:
    """A measured training level as ledger.Cost.levels expects it: (s_tr, b, u, v, w); v and w fall back to the
    method's when the level did not measure them (records written before v1.4)."""
    v = l.get("v")
    w = l.get("w")
    return (int(l["s_tr"]), float(l["b"]), float(l["u"]), float(row["v"] if v is None else v),
            float((row.get("w") or 0.0) if w is None else w))


def training_levels(spec: CellSpec, row: dict, res, ideal: float, ctx=None) -> list:
    """Monte Carlo of the training-shot error of a learning method: at every level s_tr, R refits on training features
    sampled with s_tr shots per training circuit (exact target). b(s_tr) = mean − ideal, u(s_tr) = variance of the
    estimate, and v(s_tr) = the mean per-shot target variance of the refitted estimators (E_c[var1(c)]), with
    standard errors. Then Var(N_t, s_tr) = v(s_tr)/N_t + u(s_tr) holds exactly (law of total variance) for linear readout
    mitigation (none, tensored, TREX; M3: to first order in 1/N_t). The protocol's
    own level is always included.

    Calibration: for a method whose calibration acts jointly on its training and target data (``joint_calibration``)
    and that has calibration circuits, every level also measures w(s_tr) = s_ref·Var[θ̂-mitigated − θ-mitigated
    estimate] over the same refits (the same sampled training data, mitigated with R calibration draws θ̂ at s_ref
    shots per calibration circuit and with the exact θ), with its standard error se_w.

    Common random numbers: the streams are keyed by the method's ``crn_key`` (NIL: rules that share neighbor and
    training circuits) or its key, so rules on the same circuits are measured on the same sampled data.

    A level whose resampling fails or produces a non-finite estimate is recorded with ``ok: false`` and its reason;
    the ledger does not use it (cost_of). A time limit, out-of-memory error or interrupt propagates."""
    from ..methods.learning import CAL_REF_SHOTS
    from ..resources import RunTimeout, check
    from .validate import stream_seed
    levels = sorted({int(x) for x in spec.train_levels} | {int(row["s_tr"])})
    crn = (res.details or {}).get("crn_key") or row["key"]
    joint = bool(getattr(res, "joint_calibration", False)) and int(row.get("c") or 0) > 0 and ctx is not None
    out = []
    for s_tr in levels:
        check()
        R = int(spec.train_reps) if s_tr == int(row["s_tr"]) else int(spec.train_reps_other)
        lv = dict(s_tr=int(s_tr), reps=R, ok=True)
        try:
            est, v1 = res.train_resampler(s_tr, R, np.random.default_rng(stream_seed(spec, "train", crn, s_tr)))
            est, v1 = np.asarray(est, float), np.asarray(v1, float)
            fin = np.isfinite(est) & np.isfinite(v1)
            lv["finite"] = int(fin.sum())
            if not fin.all():
                raise FloatingPointError("%d of %d refits gave a non-finite estimate" % (int((~fin).sum()), R))
            u = float(np.var(est, ddof=1))
            m4 = float(np.mean((est - est.mean()) ** 4))
            # SE of the sample variance from the fourth central moment (no Gaussian assumption: NIL estimates at small
            # s_tr are heavy-tailed): Var(s²) ≈ (μ4 − σ⁴ (n−3)/(n−1)) / n
            lv.update(b=float(np.mean(est) - ideal), u=u, v=float(np.mean(v1)),
                      se_b=float(np.sqrt(u / R)), se_u=float(np.sqrt(max(m4 - u * u * (R - 3) / (R - 1), 0.0) / R)),
                      se_v=float(np.std(v1, ddof=1) / np.sqrt(R)))
            if joint:
                check()
                th = ctx.sample_cal_theta(int(CAL_REF_SHOTS), R,
                                          np.random.default_rng(stream_seed(spec, "train-cal", crn, s_tr)))
                e1, _ = res.train_resampler(s_tr, R, np.random.default_rng(stream_seed(spec, "train", crn, s_tr)),
                                            theta=th)
                d = np.asarray(e1, float) - est
                d = d[np.isfinite(d)]
                if len(d) < 2:
                    raise FloatingPointError("no finite calibration pairs")
                vd = float(np.var(d, ddof=1))
                m4d = float(np.mean((d - d.mean()) ** 4))
                n_ = len(d)
                lv.update(w=float(CAL_REF_SHOTS * vd),
                          se_w=float(CAL_REF_SHOTS * np.sqrt(max(m4d - vd * vd * (n_ - 3) / (n_ - 1), 0.0) / n_)),
                          w_pairs=int(n_), w_ref_shots=int(CAL_REF_SHOTS))
        except (RunTimeout, MemoryError, KeyboardInterrupt):
            raise
        except Exception as e:                           # noqa: BLE001 - this level is unusable; recorded
            lv.update(ok=False, error="%s: %s" % (type(e).__name__, e))
        out.append(lv)
    return out


def cell_runner(spec: CellSpec, *, time_limit: float | None = None) -> dict:
    """Execute one grid coordinate; never raises for a scientific failure of one method (it is recorded), but does
    raise for errors of the pipeline itself (compilation, λ, guards) — the batch runner records those per cell."""
    from ..recipes import compare
    from ..resources import deadline
    from ..engine import fidelity, purity, simulate
    from ..noise import NoiseSpec
    from .descriptors import describe
    from .lightcone import calculate_lambda_lc
    t0 = time.time()
    from .store import SCHEMA_VERSION
    rec = dict(schema=SCHEMA_VERSION, cell_id=spec.cell_id, cell=spec.to_json(), status="ok")
    with deadline(time_limit):
        fc, comp = compiled_for(spec)
        backend, noise, lam_model, extra = noise_for(spec, comp)
        phys = list(comp.info["phys"])
        rec["circuit"] = dict(name=fc.name, phys=phys, chain=comp.info.get("chain"),
                              initial_layout=comp.info.get("initial_layout"), two_qubit=comp.info["two_qubit"],
                              depth=comp.info["depth"], logical_two_qubit=comp.info["logical_two_qubit"],
                              routing_overhead=comp.info["routing_overhead"],
                              measure_map={str(k): int(v) for k, v in comp.measure_map.items()},
                              observable=fc.observable.label, basis=fc.basis, relabel=fc.relabel)
        rec["engine"] = dict(engine=spec.engine, backend=backend, description=ENGINES[spec.engine], **extra)
        lam = calculate_lambda_lc(comp, lam_model, fc.observable, layout=phys)
        rec["lambda"] = lam.to_json()
        rec["descriptors"] = describe(comp.logical, comp.implemented, fc.observable)
        zne = sorted({(tuple(float(x) for x in m.p.get("scales", (1.0, 3.0, 5.0))), m.p.get("folding", "global"))
                      for m in spec.methods if m.estimator == "zne"}
                     | {(tuple(float(x) for x in m.p.get("scales", (1.0, 3.0, 5.0))), m.p.get("folding", "global"))
                        for m in spec.methods if m.estimator == "nil" and m.p.get("neighbors") == "fold"}
                     | {(tuple(float(x) for x in m.p.get("nil_scales", (1.0, 3.0, 5.0))),
                         m.p.get("nil_folding", "global"))
                        for m in spec.methods if m.estimator == "chain" and m.p.get("outer") == "nil"
                        and m.p.get("nil_neighbors") == "fold"}
                     | {(tuple(float(x) for x in m.p.get("zne_scales", (1.0, 3.0, 5.0))), m.p.get("zne_folding", "global"))
                        for m in spec.methods if m.estimator == "chain" and m.p.get("outer", "zne") == "zne"})
        rec["guards"] = [dict(fold_guard(comp, backend, sc, fo, int(spec.seed)), folding=fo) for sc, fo in zne]
        recipes = [recipe_for(m, spec, extra) for m in spec.methods]
        ideal, outs, runner = compare(comp, noise, fc.observable, recipes, backend=backend, shots=4096,
                                      symmetry=None, seed=int(spec.seed), check_budget=False)   # costed on a grid
        rec["ideal"] = float(ideal)
        # the noisy state itself (target circuit, no mitigation)
        rr = runner.run(comp.implemented, noise if backend == "numpy" else runner_noise(runner), comp.measure_map,
                        need_state=True)
        ideal_rr = simulate(comp.implemented, NoiseSpec(), "numpy", measure_map=comp.measure_map, ideal=True)
        rec["noisy"] = dict(raw_value=float(fc.observable.o @ rr.probs_meas), raw_value_no_ro=float(
            fc.observable.o @ rr.probs), fidelity=float(fidelity(rr.rho, ideal_rr.rho)), purity=float(purity(rr.rho)),
            simulations=int(runner.calls))
        rows = []
        ctxs = {}
        from ..methods import Context
        from ..resources import RunTimeout
        noise_used = noise if backend == "numpy" else runner_noise(runner)
        keys = [ms.key for ms in spec.methods]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate method keys in the cell: %s" % sorted({k for k in keys if keys.count(k) > 1}))
        for ms, rc, o in zip(spec.methods, recipes, outs):
            if o.result is None:
                rows.append(dict(key=ms.key, estimator=ms.estimator, params=ms.p, status="error", error=o.error))
                continue
            try:
                row = _method_row(ms, rc, o.result, ideal)
                if row["status"] == "ok" and int(row.get("T") or 0) > 0 and o.result.train_resampler is not None:
                    ref = getattr(o.result, "reference_mean", None)
                    row["b_inf"] = (_f(float(ref) - ideal) if ref is not None   # exact training features (the reference fit)
                                    else row["b"])
                    row["u_delta"] = row["u"]                   # v1.2's first-order (delta-method) training variance
                    row["w_run"] = row["w"]                     # the method's own Monte-Carlo or first-order w
                    ctx = Context(comp, noise_used, fc.observable, rc, runner, shots=4096, seed=int(spec.seed))
                    row["levels"] = training_levels(spec, row, o.result, ideal, ctx)
                    lv = next(l for l in row["levels"] if l["s_tr"] == row["s_tr"])
                    if not lv["ok"]:
                        raise RuntimeError("the protocol's own training level s_tr=%d failed: %s"
                                           % (row["s_tr"], lv.get("error")))
                    row["u"] = lv["u"]                          # the protocol's level, measured
                    if "w" in lv:
                        row["w"] = lv["w"]                      # measured on the same refits as u
                    if not all(l["ok"] for l in row["levels"]):
                        row["flags"].append("level-failed")
                rows.append(row)
            except (RunTimeout, MemoryError, KeyboardInterrupt):
                raise                                         # the runner records the whole cell
            except Exception as e:                           # noqa: BLE001 - one method's bookkeeping failure
                rows.append(dict(key=ms.key, estimator=ms.estimator, params=ms.p, status="error",
                                 error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc()))
            ctxs[ms.key] = (rc, o.result)
        rec["methods"] = rows
        if spec.validate_budgets:
            from .validate import validate_cell
            rec["validation"] = validate_cell(spec, comp, fc, noise, backend, runner, recipes, rows, ctxs, ideal)
        if any(r.get("status") == "error" for r in rows) or any("error" in v for v in rec.get("validation") or ()):
            rec["status"] = "partial"
    rec["provenance"] = dict(provenance(), calibration_hash=_calibration_hash(spec, phys), seconds=time.time() - t0,
                             finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return rec


def runner_noise(runner):
    """The NoiseSpec compare() hands to methods on a device backend (the device's equivalent model)."""
    dm = runner.device_model
    return dm.equivalent_noise() if dm is not None else None
