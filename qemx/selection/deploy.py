r"""Deployment path of the protocol-selection recommender (1.7.0; review §8.2–8.3, prompt section 7).

The research study labels *benchmark tasks*: exact ideal values, every candidate executed many times, oracle
comparisons. A *deployment* has none of that: a circuit, its input preparation, a diagonal observable, a device
snapshot, a compilation and a total budget B. This module keeps the two apart.

* ``DeploymentTask`` — the shared task representation of the Playground, the command line and batch runs. It is built
  from the app's state document (``from_state``: the Playground's ``Export`` and ``qemx.experiment`` documents resolve
  to the same state) or from a study task (``from_spec``), and identified by ``key()``: the same experiment gives the
  same key, features and recommendation from the UI and from the CLI.
* ``features`` — only the features a model needs, each with its information class, its cost (seconds) and how it
  was computed. Structural, observable, light-cone and calibration features (class ``available``) use the compiled
  circuit and the snapshot's calibration only — no state is simulated. Features of class ``simulated`` need an
  exact simulation of the snapshot model (exponential in width); they are computed only when explicitly allowed and
  reported as such.
* ``ModelBundle`` — a versioned, self-contained recommender: fitted preprocessing (imputation medians,
  standardization) and per-protocol models serialized as plain arrays (JSON and NPZ, loaded without pickle, evaluated
  by NumPy code here and verified against scikit-learn when the bundle is built), the feature names and order, the
  candidate protocols with their definitions and digest, the training support (feature ranges, families, widths,
  devices, budgets), a validation summary (nested cross-validation, confirmation if run), the selective rule and its
  frozen threshold, the fallback (SBS-B ranks per budget) and the guarantee status of every number it reports.
* ``recommend`` — features → OOD checks → structural feasibility at B (no simulation) → predicted log risks
  (model-conditional) → the choice, its margin and the frozen selective rule (accept or fall back to SBS-B), with
  every assumption and cost in the result. Outside the declared capabilities the result is ``unsupported``; outside
  the training support it is flagged ``out-of-distribution`` and falls back.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .. import textio as TX
from . import features as FE
from . import protocols as PR

BUNDLE_SCHEMA = "qemx.selection.bundle/1"
BUNDLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundles")


class Unsupported(ValueError):
    """An experiment outside the recommender's declared capabilities."""


# ====================================================================== the shared task representation
@dataclass(frozen=True)
class DeploymentTask:
    n: int
    ops: tuple                         # ((name, qubits, params, kind, tag), …) of the circuit body (no preparation)
    inputs: tuple                      # per-qubit input tokens (prepared once, before the circuit)
    basis: str                         # measurement basis per qubit (q0 first)
    relabel: bool                      # classical bits read in reverse order
    observable: tuple                  # (key, label, values) of a diagonal observable over the measured bits
    device: str                        # calibration snapshot (fake device specification)
    budget: int                        # total shot budget B
    opt_level: int = 1
    seed: int = 11
    initial_layout: tuple | None = None
    name: str = "circuit"

    def key(self) -> str:
        d = dict(n=self.n, ops=[list(map(_plain, o)) for o in self.ops], inputs=list(self.inputs), basis=self.basis,
                 relabel=self.relabel, observable=[self.observable[0], hashlib.sha256(np.asarray(
                     self.observable[2], float).tobytes()).hexdigest()[:16]], device=self.device, budget=self.budget,
                 opt_level=self.opt_level, seed=self.seed, initial_layout=list(self.initial_layout or ()))
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:20]

    def body(self):
        from ..circuit import CircuitSpec, Op
        return CircuitSpec(int(self.n), tuple(Op(o[0], tuple(o[1]), tuple(o[2]), o[3], o[4]) for o in self.ops),
                           self.name)

    def obs(self):
        from ..observables import Observable
        k, label, vals = self.observable
        return Observable(k, label, "", tuple(float(x) for x in vals))

    @property
    def supplied_backend(self) -> str:
        from .task import with_modifiers
        return self.device if "eng=" in self.device else with_modifiers(self.device, "eng=exact")

    def compile(self):
        from .. import compile as CP
        return CP.build(self.body(), list(self.inputs), self.basis,
                        CP.CompileOptions(device=self.device, opt_level=int(self.opt_level), seed=int(self.seed),
                                          initial_layout=tuple(self.initial_layout) if self.initial_layout else None),
                        relabel=bool(self.relabel))

    def to_json(self) -> dict:
        return dict(n=self.n, ops=[list(map(_plain, o)) for o in self.ops], inputs=list(self.inputs),
                    basis=self.basis, relabel=self.relabel, observable=dict(key=self.observable[0],
                                                                           label=self.observable[1]),
                    device=self.device, budget=self.budget, opt_level=self.opt_level, seed=self.seed,
                    initial_layout=list(self.initial_layout) if self.initial_layout else None, key=self.key())


def _plain(x):
    if isinstance(x, (list, tuple)):
        return [_plain(y) for y in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    return x


def _ops_of(spec) -> tuple:
    return tuple((o.name, tuple(int(q) for q in o.qubits), tuple(float(p) for p in o.params), o.kind, o.tag or "")
                 for o in spec.ops)


def from_state(state: dict, budget: int, device: str | None = None) -> DeploymentTask:
    """The deployment task of an app state document (the Playground's state, an exported state or a
    ``qemx.experiment`` document's state). ``device``: the calibration snapshot when the state's backend is not a
    fake device. The input preparation is the per-qubit inputs, prepared once (a preparation saved in the circuit is
    folded into them, as the Playground does)."""
    from .. import device as DV
    from .. import io_semantics as IO
    from ..circuit import CircuitSpec, Op
    from ..state_schema import parse_layout, validate_state
    v = validate_state(state)
    sp = CircuitSpec(int(v["n"]), tuple(Op(o["name"], tuple(o["qubits"]), tuple(o["params"]), o["kind"],
                                          o.get("tag", "")) for o in v["ops"]), v["name"])
    inputs, body = IO.split_preparation(sp, v["inputs"])
    ob = v["observable"]
    if hasattr(ob, "terms") or isinstance(ob, dict) and "pauli" in ob:
        raise Unsupported("the recommender supports diagonal observables (Z strings, projectors); a Hamiltonian with "
                          "X or Y terms is measured in several settings and is outside the study's task class")
    if isinstance(ob, tuple):
        key, label, _tex, vals = ob
    else:
        key, label, vals = ob.key, ob.label, ob.values
    vals = np.asarray(vals, float)
    if vals.size != 2 ** int(v["n"]) or np.max(np.abs(vals)) > 1 + 1e-12:
        raise Unsupported("the observable must be diagonal over the measured bits with |O| ≤ 1 (the study's task "
                          "class)")
    backend = v["backend"]
    dev = backend if DV.is_device(backend) else device
    if not dev or not DV.is_device(dev):
        raise Unsupported("the recommender is trained on fake-device calibration snapshots; choose a fake device "
                          "(the state's backend %r is a noise injector)" % backend)
    cm = dict(v.get("compile") or {})
    init = parse_layout(cm.get("initial_layout", "")) or None
    return DeploymentTask(int(v["n"]), _ops_of(body), tuple(inputs), str(v["basis"]), bool(v["relabel"]),
                          (str(key), str(label), tuple(float(x) for x in vals)), str(dev), int(budget),
                          int(cm.get("opt_level", 1)), int(cm.get("seed", 11)),
                          tuple(int(q) for q in init) if init else None, str(v["name"]))


def from_spec(spec, budget: int) -> DeploymentTask:
    """The deployment task of a study task (``task.TaskSpec``): the same physical experiment, without its ideal
    value. Compiling it reproduces the study's compiled circuit (layout, seed and optimization level as the study)."""
    from .task import family_circuit, input_tokens, layout_for, observable_of
    body, basis, default_obs, relabel, fam_tokens = family_circuit(spec)
    tokens = input_tokens(spec, fam_tokens)
    obs = observable_of(spec, default_obs, int(body.n))
    lay = layout_for(spec, int(body.n))
    return DeploymentTask(int(body.n), _ops_of(body), tuple(tokens), str(basis), bool(relabel),
                          (obs.key, obs.label, tuple(float(x) for x in obs.o)), spec.device, int(budget),
                          int(spec.opt_level), int(spec.seed), tuple(int(q) for q in lay), spec.label)


# ====================================================================== features at deployment
STRUCTURAL = ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "pulses_1q", "non_clifford_slots",
              "idle_fraction", "obs_weight", "obs_terms", "obs_support")
LIGHTCONE = ("lc_qubits", "lc_gates", "lc_2q", "lc_locations", "lam_lc", "lam_lc_exact", "lam_lc_gate", "lam_lc_idle",
             "lam_total", "lc_share", "lam_ro")
SIMULATED = tuple(f.name for f in FE.DICTIONARY if f.info == "simulated")
METHOD = {**{k: "compiled circuit and observable (no simulation)" for k in STRUCTURAL},
          **{k: "backward light cone with the snapshot's calibrated error rates (no simulation)" for k in LIGHTCONE},
          "duration_us": "ALAP schedule with the snapshot's gate durations (no simulation)",
          "pec_log_gamma": "product of the depolarizing quasi-probability norms of the snapshot's median rates and "
                           "the readout inverse's norm from the two one-layer calibration circuits (no simulation of "
                           "the target circuit)",
          "zne_eta_l1_linear": "fold scales achieved by global folding (gate counts) (no simulation)",
          "zne_eta_l1_richardson": "fold scales achieved by global folding (gate counts) (no simulation)",
          "log10_budget": "the budget",
          **{k: "exact simulation of the snapshot model (exponential in width)" for k in SIMULATED}}


def _achieved(spec, lam):
    from ..methods.zne import achieved_lambda
    from ..transforms import fold_global
    return achieved_lambda(spec, fold_global(spec, float(lam)))


def cheap_protocol_features(comp, supplied: str) -> dict:
    """pec_log_gamma and the ZNE noise-amplification norms without simulating anything (the same quantities the
    study reads from the methods' results; equality is tested on the study's tasks)."""
    from .. import device
    from ..methods.base import assignment_inverse_gamma
    from ..methods.pec import depol_quasi, gamma_of
    phys = tuple(comp.info["phys"])
    dm = device.get_model(supplied, phys)
    noise = dm.equivalent_noise()
    lg = 0.0
    for o in comp.implemented.ops:
        if o.kind != "gate" or o.is_virtual:
            continue
        p = noise.p1 if o.k == 1 else noise.p2
        lg += math.log(gamma_of(depol_quasi(float(p), o.k, "inverse")))
    # the readout inverse's norm from the two one-layer calibration circuits (|0…0⟩ and X on every qubit) emulated on
    # the snapshot, exactly as the protocol's calibration does (the X gates carry their gate error): product
    # circuits, no state of the target circuit is simulated
    from ..methods import Context, Runner
    ctx = Context(comp, noise, None, PR.PORTFOLIO["PEC"].recipe(), Runner(backend=supplied, layout=phys))
    for A in ctx.calibration()["A"]:
        lg += math.log(assignment_inverse_gamma(A[1, 0], A[0, 1]))
    out = dict(pec_log_gamma=lg)
    for key, deg in (("ZNE-L", 1), ("ZNE-R", 2)):
        lams = [_achieved(comp.implemented, l) for l in dict(PR.PORTFOLIO[key].params)["scales"]]
        out["zne_eta_l1_%s" % ("linear" if deg == 1 else "richardson")] = FE._eta_l1(lams, deg)
    return out


def features(task: DeploymentTask, names, *, allow_simulation: bool = False, compiled=None) -> tuple:
    """(values, report) of the requested features only. ``report``: one row per feature with its information class,
    the seconds it took (shared work is charged to the first feature that needs it) and the method. Simulated
    features are computed only with ``allow_simulation`` (else NaN, reported as not computed)."""
    from .. import features as FT
    from ..context_noise import alap_schedule
    from ..phase.lightcone import calculate_lambda_lc
    names = list(names)
    bad = [n for n in names if n not in FE.BY_NAME or FE.BY_NAME[n].info == "oracle"]
    if bad:
        raise ValueError("not deployable features: %s" % ", ".join(bad))
    vals, rep = {}, []
    t0 = time.perf_counter()
    comp = compiled if compiled is not None else task.compile()
    t_compile = time.perf_counter() - t0
    obs = task.obs()
    sup = task.supplied_backend

    def record(keys, seconds, computed=True):
        for i, k in enumerate(keys):
            rep.append(dict(feature=k, info=FE.BY_NAME[k].info, seconds=float(seconds if i == 0 else 0.0),
                            method=METHOD.get(k, ""), computed=bool(computed and np.isfinite(vals.get(k, math.nan)))))
    want = set(names)
    if want & set(STRUCTURAL):
        t = time.perf_counter()
        ks = [k for k in STRUCTURAL if k in want]
        v, _ = FT.compute(comp, obs, sup, None, names=ks)
        vals.update({k: float(v.get(k, math.nan)) for k in ks})
        record(ks, time.perf_counter() - t)
    if "duration_us" in want:
        from .. import device
        t = time.perf_counter()
        dm = device.get_model(sup, tuple(comp.info["phys"]))
        vals["duration_us"] = float(alap_schedule(comp.implemented, dm.durations)[3]) * 1e6
        record(["duration_us"], time.perf_counter() - t)
    if want & set(LIGHTCONE):
        t = time.perf_counter()
        r = calculate_lambda_lc(comp, sup, obs)
        lc = dict(lc_qubits=float(len(r.lc_qubits)), lc_gates=float(r.lc_gates), lc_2q=float(r.lc_2q_gates),
                  lc_locations=float(r.n_loc_lc), lam_lc=float(r.lam_lc), lam_lc_exact=float(r.lam_lc_exact),
                  lam_lc_gate=float(r.lam_lc_gate), lam_lc_idle=float(r.lam_lc_idle), lam_total=float(r.lam_total),
                  lc_share=float(r.lam_lc / r.lam_total) if r.lam_total > 0 else 0.0, lam_ro=float(r.lam_ro))
        ks = [k for k in LIGHTCONE if k in want]
        vals.update({k: lc[k] for k in ks})
        record(ks, time.perf_counter() - t)
    cheap = [k for k in ("pec_log_gamma", "zne_eta_l1_linear", "zne_eta_l1_richardson") if k in want]
    if cheap:
        t = time.perf_counter()
        c = cheap_protocol_features(comp, sup)
        vals.update({k: c[k] for k in cheap})
        record(cheap, time.perf_counter() - t)
    if "log10_budget" in want:
        vals["log10_budget"] = math.log10(int(task.budget))
        record(["log10_budget"], 0.0)
    sim = [k for k in names if FE.BY_NAME[k].info == "simulated"]
    if sim:
        if allow_simulation:
            t = time.perf_counter()
            vals.update({k: v for k, v in simulated_features(task, comp).items() if k in sim})
            record(sim, time.perf_counter() - t)
        else:
            for k in sim:
                vals[k] = math.nan
            record(sim, 0.0, computed=False)
    return {k: vals.get(k, math.nan) for k in names}, dict(rows=rep, compile_seconds=float(t_compile))


def simulated_features(task: DeploymentTask, comp) -> dict:
    """The ``simulated`` features: the supplied-model methods run exactly on the snapshot (RO, ZNE-R, PEC per-shot
    variances, the noisy signal) and the ideal state's entropies. Exponential in width."""
    from .. import features as FT
    from . import emulate as EM
    from .run import FEATURE_PROTOCOLS

    class _Spec:
        supplied_backend = task.supplied_backend
        truth = ""
        device = task.device
    built = type("Deployment", (), dict(spec=_Spec(), compiled=comp, observable=task.obs(), ideal=None))()
    prep = EM.prepare(built, FEATURE_PROTOCOLS, seed=0, designs=1)
    s = prep.supplied
    out = {}

    def ok(r):
        return r is not None and not isinstance(r, Exception)
    ro, zr, pec, raw = s.get("RO"), s.get("ZNE-R"), s.get("PEC"), s.get("RAW")
    out["log10_v_raw_sup"] = FE._log10(ro.var1) if ok(ro) else math.nan
    out["log10_v_zne_sup"] = FE._log10(zr.var1) if ok(zr) else math.nan
    out["log10_v_pec_sup"] = FE._log10(pec.var1) if ok(pec) else math.nan
    if ok(raw):
        ctx = prep.ctx_truth["RAW"]
        run = ctx.runner.run(comp.implemented, ctx.noise, comp.measure_map)
        out["abs_signal_sup"] = float(abs(task.obs().o @ run.probs))
    else:
        out["abs_signal_sup"] = math.nan
    v, _ = FT.compute(comp, task.obs(), task.supplied_backend, None, names=["entropy_half", "output_entropy_bits"])
    out.update({k: float(v.get(k, math.nan)) for k in ("entropy_half", "output_entropy_bits")})
    return out


# ====================================================================== structural feasibility (no simulation)
def structural_feasibility(task: DeploymentTask, comp, key: str, B: int) -> tuple:
    """(feasible, reason) of a protocol at budget B from structural counts only: the minimum shots of its
    calibration, training and target circuits (the study's allocation rule, ``protocols.allocate``, needs the
    supplied model's variances to split the shots, not to decide feasibility except at the boundary). Learning
    protocols on a Clifford circuit have no training circuits and are infeasible, as in the study; a training design
    whose labels turn out degenerate (known only after its training circuits' ideal values are computed) is not
    detected here."""
    pr = PR.PORTFOLIO[key]
    B = int(B)
    c = 2 if pr.readout != "none" else 0                  # two calibration circuits, ≥ 1 shot each
    if pr.estimator == "raw":
        return (B >= 1 + c), ("" if B >= 1 + c else "needs %d shots" % (1 + c))
    if pr.estimator == "zne":
        k = len(dict(pr.params)["scales"])
        return (B >= k + c), ("" if B >= k + c else "needs one shot per fold scale and the calibration")
    if pr.estimator == "pec":
        return (B >= 1 + c), ""
    from ..circuit import is_clifford
    from ..methods.learning import non_clifford_slots
    if not non_clifford_slots(comp.implemented):
        return False, "Clifford circuit: no training circuits (the study treats the protocol as infeasible)"
    if pr.estimator == "learning":
        T = int(dict(pr.params).get("n_train", 16))
        k_target = 1
    else:
        from ..methods.nil import nil_pool
        prm = dict(pr.params)
        pool = nil_pool(comp.implemented, prm.get("positions", "all"))
        k_target = min(12, len(pool)) + (1 if prm.get("include_bare") else 0)
        T = int(prm.get("n_train", 16)) * max(k_target, 1)
    rem = B
    s_tr = int(math.floor(pr.train_fraction * rem / T))
    if s_tr < PR.TRAIN_MIN:
        return False, "training needs ≥ %d shots on each of %d circuits" % (PR.TRAIN_MIN, T)
    left = rem - s_tr * T
    if left < c + k_target:
        return False, "no shots left for calibration and the target circuits"
    del is_clifford
    return True, ""


# ====================================================================== model serialization (no pickle)
def _ser_pipeline(est) -> tuple:
    """(meta, arrays) of a fitted estimator: Pipeline(StandardScaler, Ridge|ElasticNet|MLP), HGB, or the structured
    model (two ridge pipelines)."""
    from sklearn.pipeline import Pipeline
    from .models import StructuredRisk
    if isinstance(est, StructuredRisk):
        m1, a1 = _ser_pipeline(est.m1)
        m2, a2 = _ser_pipeline(est.m2)
        return (dict(kind="struct", f=est.f, budget_col=est.bc, eps0=est.eps0, b2=m1, v=m2),
                {**{"b2_" + k: v for k, v in a1.items()}, **{"v_" + k: v for k, v in a2.items()}})
    if isinstance(est, Pipeline):
        sc, reg = est.steps[0][1], est.steps[-1][1]
        arrays = dict(mean=np.asarray(sc.mean_, float), scale=np.asarray(sc.scale_, float))
        name = type(reg).__name__
        if name in ("Ridge", "ElasticNet"):
            arrays.update(coef=np.asarray(reg.coef_, float).ravel(), intercept=np.atleast_1d(float(reg.intercept_)))
            return dict(kind="linear", estimator=name), arrays
        if name == "MLPRegressor":
            if reg.activation != "relu" or reg.out_activation_ != "identity":
                raise ValueError("unsupported MLP activation")
            for i, (W, b) in enumerate(zip(reg.coefs_, reg.intercepts_)):
                arrays["W%d" % i], arrays["b%d" % i] = np.asarray(W, float), np.asarray(b, float)
            return dict(kind="mlp", layers=len(reg.coefs_)), arrays
        raise ValueError("unsupported estimator %s" % name)
    if type(est).__name__ == "HistGradientBoostingRegressor":
        trees = [p[0].nodes for p in est._predictors]
        offs = np.cumsum([0] + [len(t) for t in trees])
        nodes = np.concatenate(trees) if trees else np.zeros(0)
        cols = {k: np.asarray(nodes[k]) for k in ("value", "feature_idx", "num_threshold", "missing_go_to_left",
                                                  "left", "right", "is_leaf")}
        return (dict(kind="hgb", trees=len(trees), baseline=float(np.ravel(est._baseline_prediction)[0])),
                {**{"node_" + k: v for k, v in cols.items()}, "offsets": offs.astype(np.int64)})
    raise ValueError("unsupported estimator %s" % type(est).__name__)


def _std(X, a, prefix=""):
    sc = a[prefix + "scale"]
    return (np.asarray(X, float) - a[prefix + "mean"]) / np.where(sc == 0, 1.0, sc)


def _predict_ser(meta, a, X, prefix="") -> np.ndarray:
    X = np.asarray(X, float)
    k = meta["kind"]
    if k == "linear":
        return _std(X, a, prefix) @ a[prefix + "coef"] + a[prefix + "intercept"][0]
    if k == "mlp":
        h = _std(X, a, prefix)
        for i in range(meta["layers"]):
            h = h @ a[prefix + "W%d" % i] + a[prefix + "b%d" % i]
            if i < meta["layers"] - 1:
                h = np.maximum(h, 0.0)
        return h.ravel()
    if k == "hgb":
        out = np.full(len(X), meta["baseline"])
        offs = a[prefix + "offsets"]
        val, fi, th = a[prefix + "node_value"], a[prefix + "node_feature_idx"], a[prefix + "node_num_threshold"]
        mgl, lt, rt, leaf = (a[prefix + "node_missing_go_to_left"], a[prefix + "node_left"], a[prefix + "node_right"],
                             a[prefix + "node_is_leaf"])
        for t in range(meta["trees"]):
            o = int(offs[t])
            for r in range(len(X)):
                node = 0
                while not leaf[o + node]:
                    x = X[r, fi[o + node]]
                    if np.isnan(x):
                        node = int(lt[o + node] if mgl[o + node] else rt[o + node])
                    else:
                        node = int(lt[o + node] if x <= th[o + node] else rt[o + node])
                out[r] += val[o + node]
        return out
    if k == "struct":
        Xn = np.delete(X, meta["budget_col"], axis=1)
        Be = (1.0 - meta["f"]) * 10.0 ** X[:, meta["budget_col"]]
        b2 = 10.0 ** _predict_ser(meta["b2"], a, Xn, prefix + "b2_")
        v = 10.0 ** _predict_ser(meta["v"], a, Xn, prefix + "v_")
        from .models import STRUCT_FLOOR
        return np.log10(np.maximum(b2 - STRUCT_FLOOR, 0.0) + np.maximum(v - STRUCT_FLOOR, 0.0) / Be + meta["eps0"])
    raise ValueError(k)


# ====================================================================== the bundle
@dataclass
class ModelBundle:
    meta: dict
    arrays: dict = field(default_factory=dict)
    path: str = ""

    @property
    def features(self) -> list:
        return list(self.meta["features"])

    @property
    def protocols(self) -> list:
        return list(self.meta["protocols"])

    def predict(self, X) -> np.ndarray:
        """(rows, protocols) predicted log10(R + ε0) (NaN where a protocol has no model)."""
        X = np.asarray(X, float).copy()
        med = np.asarray(self.meta["impute"], float)
        bad = ~np.isfinite(X)
        X[bad] = np.take(med, np.nonzero(bad)[1])
        out = np.full((len(X), len(self.protocols)), np.nan)
        for j, p in enumerate(self.protocols):
            m = self.meta["models"].get(p)
            if m is not None:
                out[:, j] = _predict_ser(m, self.arrays, X, prefix="%s__" % p)
        return out


def build_bundle(ds, train_idx, family: str, set_name: str, out_dir: str, *, version: str, tau: float | None,
                 validation: dict | None = None, training: dict | None = None) -> ModelBundle:
    """Fit the configuration on ``train_idx`` and write a bundle to ``out_dir`` (bundle.json + arrays.npz). The
    serialized models are evaluated by this module's NumPy code and compared with scikit-learn's predictions on the
    training rows (the largest difference is stored; the build fails above 1e-9)."""
    from . import policies as PO
    from .claims import KINDS
    train_idx = np.asarray(train_idx)
    names = FE.SETS[set_name]
    pol = PO.Model(family, names, set_name).fit(ds, train_idx)
    fb = PO.SBSBudget().fit(ds, train_idx)
    rm = pol.rm
    models, arrays = {}, {}
    for p, est in rm.models.items():
        meta, arr = _ser_pipeline(est)
        models[p] = meta
        arrays.update({"%s__%s" % (p, k): v for k, v in arr.items()})
    X = ds.X(names)[train_idx]
    support = {}
    for j, n in enumerate(names):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            support[n] = dict(min=float(col.min()), max=float(col.max()), q01=float(np.quantile(col, 0.01)),
                              q99=float(np.quantile(col, 0.99)))
    rows = [ds.rows[i] for i in train_idx]
    meta = dict(schema=BUNDLE_SCHEMA, version=version, created=time.strftime("%Y-%m-%d"),
                configuration="MODEL[%s|%s]" % (family, set_name), family=family, feature_set=set_name,
                features=list(names), feature_classes={n: FE.BY_NAME[n].info for n in names},
                protocols=list(ds.protocols),
                protocol_definitions={k: PR.PORTFOLIO[k].to_json() for k in ds.protocols},
                portfolio_digest=hashlib.sha256(json.dumps([PR.PORTFOLIO[k].to_json() for k in ds.protocols],
                                                           sort_keys=True).encode()).hexdigest()[:16],
                eps0=float(rm.eps0), impute=[float(x) for x in rm.impute], models=models,
                chosen=rm.chosen, cv_rmse={k: float(v) for k, v in rm.cv_rmse.items()},
                selective=dict(rule="accept the model's choice when its margin z = (P2 − P1)/sqrt(s1² + s2²) ≥ τ, "
                                    "else SBS-B", tau=None if tau is None else float(tau)),
                fallback=dict(policy="SBS-B", ranks={str(b): [ds.protocols[j] for j in r] for b, r in fb.rank.items()},
                              global_rank=[ds.protocols[j] for j in fb.global_rank],
                              rule="first feasible protocol in the training ranking of the nearest budget (log scale)"),
                support=dict(features=support, budgets=[int(min(r["B"] for r in rows)), int(max(r["B"] for r in rows))],
                             families=sorted({r["family"] for r in rows}), widths=sorted({int(r["n"]) for r in rows}),
                             devices=sorted({r["device"] for r in rows}), base_circuits=len({r["base_id"] for r in rows}),
                             rows=len(rows)),
                validation=validation or {}, training=training or {},
                guarantees={"predicted log risks": "model-conditional", "margin and selective rule": "model-conditional",
                            "nested-CV performance": "empirical", "training-support check": "ood-warning"},
                statement_kinds={k: KINDS[k] for k in ("model-conditional", "empirical", "ood-warning")},
                capabilities=dict(task_class="fake-device calibration snapshots (exact engine); diagonal observables "
                                             "with |O| ≤ 1; product-state inputs; circuits the study's compiler maps",
                                  cheap_path=all(FE.BY_NAME[n].info == "available" for n in names)))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "arrays.npz"), **arrays)
    b = ModelBundle(meta, {k: np.asarray(v) for k, v in arrays.items()}, out_dir)
    sk = rm.predict(X)
    mine = b.predict(X)
    ok = np.isfinite(sk) & np.isfinite(mine)
    diff = float(np.max(np.abs(sk[ok] - mine[ok]))) if ok.any() else 0.0
    if diff > 1e-9:
        raise AssertionError("the serialized models differ from scikit-learn by %.3g" % diff)
    meta["verification"] = dict(max_abs_difference_vs_sklearn=diff, rows=int(len(X)))
    meta["arrays_sha256"] = _sha(os.path.join(out_dir, "arrays.npz"))
    TX.write_json(os.path.join(out_dir, "bundle.json"), meta, indent=1, sort_keys=True)
    return b


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_bundle(path: str) -> ModelBundle:
    """A bundle directory (bundle.json + arrays.npz): its digest is checked, the arrays are loaded without pickle."""
    meta = TX.read_json(os.path.join(path, "bundle.json"))
    if meta.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("%s is not a model bundle (%s)" % (path, meta.get("schema")))
    ap = os.path.join(path, "arrays.npz")
    if _sha(ap) != meta.get("arrays_sha256"):
        raise ValueError("%s: arrays.npz does not match its digest in bundle.json" % path)
    with np.load(ap, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return ModelBundle(meta, arrays, path)


def bundles() -> list:
    """The bundles shipped with the application (newest first)."""
    if not os.path.isdir(BUNDLE_DIR):
        return []
    return sorted([d for d in os.listdir(BUNDLE_DIR) if os.path.exists(os.path.join(BUNDLE_DIR, d, "bundle.json"))],
                  reverse=True)


# ====================================================================== recommendation
def recommend(task: DeploymentTask, bundle: ModelBundle, *, allow_simulation: bool = False) -> dict:
    """The recommendation for one deployment task (plain data; the UI and the CLI show the same dictionary)."""
    from .. import device as DV
    t0 = time.perf_counter()
    out = dict(task=task.to_json(), bundle=bundle.meta["version"], configuration=bundle.meta["configuration"],
               status="ok", warnings=[], statement_kinds=bundle.meta.get("guarantees", {}))
    names = bundle.features
    needs_sim = [n for n in names if FE.BY_NAME[n].info == "simulated"]
    if needs_sim and not allow_simulation:
        out["warnings"].append("the model uses simulated features (%s); they were not computed (enable simulation to "
                               "compute them; cost grows exponentially with width)" % ", ".join(needs_sim))
    try:
        comp = task.compile()
    except Exception as e:                           # noqa: BLE001
        raise Unsupported("the circuit does not compile for %s: %s" % (task.device, str(e)[:200])) from None
    vals, rep = features(task, names, allow_simulation=allow_simulation, compiled=comp)
    out["features"] = vals
    out["feature_report"] = rep
    sup = bundle.meta["support"]
    ood = []
    base = DV.base_name(task.device)
    if base not in {DV.base_name(d) for d in sup["devices"]}:
        ood.append("device %s was not in the training data (%s)" % (base, ", ".join(sup["devices"])))
    if not (sup["budgets"][0] <= int(task.budget) <= sup["budgets"][1]):
        ood.append("budget %s outside the training range %s–%s" % (f"{task.budget:,}", f"{sup['budgets'][0]:,}",
                                                                    f"{sup['budgets'][1]:,}"))
    if int(task.n) not in sup["widths"]:
        ood.append("width %d outside the training widths %s" % (task.n, sup["widths"]))
    for n_, v in vals.items():
        s = sup["features"].get(n_)
        if s is not None and np.isfinite(v):
            span = max(s["max"] - s["min"], 1e-12)
            if v < s["min"] - 0.05 * span or v > s["max"] + 0.05 * span:
                ood.append("feature %s = %.4g outside the training range [%.4g, %.4g]" % (n_, v, s["min"], s["max"]))
    missing = [n for n, v in vals.items() if not np.isfinite(v)]
    if missing:
        ood.append("features not computed (imputed with training medians): %s" % ", ".join(missing))
    out["ood"] = ood
    feas = {}
    for k in bundle.protocols:
        f, why = structural_feasibility(task, comp, k, task.budget)
        feas[k] = dict(feasible=bool(f), note=why)
    out["feasibility"] = feas
    out["feasibility_method"] = ("structural minimum shots (no simulation); the exact allocation is computed when the "
                                 "protocol runs")
    x = np.array([[vals[n] for n in names]], float)
    P = bundle.predict(x)[0]
    pred = {p: float(P[j]) for j, p in enumerate(bundle.protocols) if np.isfinite(P[j])}
    out["predicted_log10_risk"] = pred
    cands = [p for p in pred if feas[p]["feasible"]]
    fbr = _fallback_rank(bundle, task.budget)
    fallback = next((p for p in fbr if feas.get(p, {}).get("feasible")), "RAW")
    out["fallback"] = fallback
    if not cands:
        out.update(status="no-model", choice=fallback, accepted=False,
                   reason="no feasible protocol has a model prediction; the fallback is used")
        return _finish(out, t0)
    order = sorted(cands, key=lambda p: pred[p])
    best = order[0]
    if len(order) >= 2:
        s1 = bundle.meta["cv_rmse"].get(best, math.nan)
        s2 = bundle.meta["cv_rmse"].get(order[1], math.nan)
        margin = float((pred[order[1]] - pred[best]) / math.sqrt(s1 ** 2 + s2 ** 2))
    else:
        margin = math.inf
    tau = bundle.meta["selective"]["tau"]
    out.update(model_choice=best, runner_up=order[1] if len(order) > 1 else None, margin=margin, tau=tau)
    if ood:
        out.update(status="out-of-distribution", choice=fallback, accepted=False,
                   reason="outside the training support: the model's prediction is not used (fallback SBS-B)")
    elif tau is not None and not (margin >= tau):
        out.update(choice=fallback, accepted=False,
                   reason="margin %.2f below the frozen threshold τ = %.2f: fallback SBS-B" % (margin, tau))
    else:
        out.update(choice=best, accepted=True, reason="the model's choice (margin %.2f ≥ τ)" % margin
                   if tau is not None else "the model's choice (no selective threshold in this bundle)")
    out["protocol_definition"] = bundle.meta["protocol_definitions"].get(out["choice"], {})
    return _finish(out, t0)


def _fallback_rank(bundle, B) -> list:
    ranks = bundle.meta["fallback"]["ranks"]
    if not ranks:
        return list(bundle.meta["fallback"]["global_rank"])
    bs = np.array(sorted(int(b) for b in ranks))
    b = int(bs[np.argmin(np.abs(np.log10(bs) - math.log10(max(int(B), 1))))])
    return list(ranks[str(b)])


def _finish(out, t0):
    out["seconds"] = float(time.perf_counter() - t0)
    return out
