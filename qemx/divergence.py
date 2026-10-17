r"""Hardware against component-model divergence: where a device underperforms the noise model built from its own
component-level metrics (Chen et al., *Benchmarking a trapped-ion quantum computer with 30 qubits*, Quantum 8, 1516
(2024), §4.3 and Fig. 11; Lubinski et al., *Application-oriented performance benchmarks for quantum computing*,
IEEE TQE 4, 3100332 (2023)).

The same application circuits run on a device (an IBM backend through the Sampler primitive, a fake backend in local
mode, or an emulator with known unmodeled noise) and are *predicted* by simulating those same as-executed circuits
under noise models built from the device's calibration snapshot of the day:

=================  ================================================================================================
predictor          noise model of every as-executed circuit (compact physical qubits)
=================  ================================================================================================
``depolarizing``   every gate followed by the depolarizing channel that has the gate's reported error as its
                   average gate infidelity (per gate and qubit pair, as calibrated), each qubit's own asymmetric
                   readout error; no relaxation and no idle error. Chen et al.'s model with heterogeneous rates and
                   readout — the "fake device from component-level metrics".
``chen``           Chen et al.'s model as published: one rate for every one-qubit gate and one for every two-qubit
                   gate (the medians), no readout error. A depolarizing channel Λ_ε ρ = (1−ε)ρ + ε/(4ⁿ−1) Σ_{P≠I} PρP
                   with entanglement infidelity ε has average gate infidelity r = ε·d/(d+1), d = 2ⁿ; IBM reports r
                   (RB error per gate), Chen et al. ε (DRB): both describe the same channel.
``device``         the fake-device model of Qiskit: thermal relaxation over every gate and every scheduled idle
                   window (T1, T2), depolarizing for the rest of each gate's error, readout (qemx's E2 model)
=================  ================================================================================================

Every **method** of the suite turns the counts of its own circuits into an output distribution, at its own total
budget of B shots (equal for every method; circuits are never shared between methods):

* ``raw``  the measured distribution;
* ``ro``   tensored readout mitigation: per-bit flip probabilities from |0…0⟩ and |1…1⟩ (s_cal shots each),
  inverted bit by bit, then the nearest probability distribution (Euclidean projection on the simplex: Smolin,
  Gambetta & Smith, PRL 108, 070502 (2012));
* ``zne``  readout-mitigated distributions of the circuit folded to λ_j (global or local folding, as in
  :mod:`qemx.methods.zne`), extrapolated entry by entry, Σ_j η_j q_j, then the nearest probability distribution;
  target shots split N_j ∝ |η_j|.

The **fidelity** of an output distribution q to the ideal p is Fc(p, q) = (Σ_i √(p_i q_i))² (Chen et al., Eq. 1)
and Lubinski et al.'s normalized fidelity max((Fc − Fc(p, u))/(1 − Fc(p, u)), 0), u uniform. A prediction is evaluated
exactly as the observation is: R Monte-Carlo replicates of the method at the same shots per circuit, from the
predicted probabilities, through the same estimator — the predicted Fc has the observation's finite-shot bias, and its
spread is the model's sampling distribution. Δ = mean predicted Fc − observed Fc and z = Δ / (replicate standard
deviation): |z| > 3 is a divergence the shot noise of the budget does not explain. Observed Fc carries a bootstrap
standard error (multinomial resampling of every circuit's counts).

Everything here is computation on in-memory objects: the plan, counts and calibration snapshots are JSON-ready dicts;
reading and writing them, the IBM connection and its credentials belong to ``scripts/dirA_divergence.py``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .circuit import CircuitSpec, Op
from .transforms import FOLD_SUBSETS

SCHEMA = 1
PREDICTORS = ("depolarizing", "chen", "device")
METHOD_KINDS = ("raw", "ro", "zne")
ZNE_RULES = ("richardson", "linear", "poly2")
INFINITE_T = 1.0e3            # seconds: T1 = T2 of the depolarizing models (relaxation over a 100 ns gate: ~1e-10)
DET_MIN = 0.05                # readout mitigation of a bit is skipped when 1 − p01 − p10 < DET_MIN (flagged)


# ====================================================================== metrics (vectorized over rows)
def hellinger_fidelity(p, q) -> np.ndarray:
    """Fc(p, q) = (Σ_i √(p_i q_i))² (Chen et al. Eq. 1; the square of the Bhattacharyya coefficient). ``q`` may hold
    one distribution per row; negative entries count as 0."""
    p = np.clip(np.asarray(p, float), 0, None)
    q = np.clip(np.asarray(q, float), 0, None)
    s = np.sum(np.sqrt(p * q), axis=-1)
    return s * s


def normalized_fidelity(p, q) -> np.ndarray:
    """Lubinski et al.: max((Fc(p, q) − Fc(p, u)) / (1 − Fc(p, u)), 0), u the uniform distribution. NaN when the
    ideal distribution is itself uniform (the normalization is undefined)."""
    p = np.asarray(p, float)
    fu = float(np.sum(np.sqrt(np.clip(p, 0, None))) ** 2 / p.size)
    f = hellinger_fidelity(p, q)
    if fu >= 1 - 1e-12:
        return np.full(np.shape(f), np.nan)
    return np.maximum((f - fu) / (1 - fu), 0.0)


def nearest_probability(q) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex (for a row that sums to 1 this is Smolin,
    Gambetta & Smith's maximum-likelihood distribution under Gaussian noise). Sort-based (Duchi et al. 2008)."""
    Q = np.atleast_2d(np.asarray(q, float))
    U = -np.sort(-Q, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    k = np.arange(1, Q.shape[1] + 1)
    rho = np.sum(U - css / k > 0, axis=1) - 1                 # the condition holds on a prefix
    theta = css[np.arange(Q.shape[0]), rho] / (rho + 1)
    out = np.maximum(Q - theta[:, None], 0.0)
    return out[0] if np.ndim(q) == 1 else out


def bit_marginals(F, n: int) -> np.ndarray:
    """P(classical bit i = 1) of each row (index bit i = classical bit i, little-endian)."""
    F = np.atleast_2d(np.asarray(F, float))
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)    # (2^n, n)
    return F @ bits


def readout_flips(f0, f1, n: int) -> tuple:
    """Per-bit (p01, p10): P(read 1 | prepared 0) from the |0…0⟩ calibration, P(read 0 | prepared 1) from |1…1⟩."""
    return bit_marginals(f0, n), 1.0 - bit_marginals(f1, n)


def mitigate_tensored(F, p01, p10, n: int) -> tuple:
    """(quasi-distributions, bits not mitigated): each bit's 2×2 confusion matrix inverted, row by row. A bit whose
    matrix is near-singular (1 − p01 − p10 < DET_MIN) is left as measured."""
    F = np.atleast_2d(np.asarray(F, float))
    R = F.shape[0]
    p01 = np.broadcast_to(np.atleast_2d(p01), (R, n)).copy()
    p10 = np.broadcast_to(np.atleast_2d(p10), (R, n)).copy()
    det = 1.0 - p01 - p10
    bad = det < DET_MIN
    p01[bad], p10[bad], det[bad] = 0.0, 0.0, 1.0
    T = F.reshape((R,) + (2,) * n)
    for i in range(n):
        a, b, d = p01[:, i], p10[:, i], det[:, i]
        inv = np.stack([np.stack([1 - b, -b], -1), np.stack([-a, 1 - a], -1)], -2) / d[:, None, None]   # (R,2,2)
        ax = 1 + (n - 1 - i)
        T = np.moveaxis(T, ax, -1)
        T = np.einsum("r...j,rkj->r...k", T, inv)
        T = np.moveaxis(T, -1, ax)
    return T.reshape(R, -1), sorted(set(np.where(bad.any(axis=0))[0].tolist()))


def extrapolation_weights(rule: str, lams) -> np.ndarray:
    from .methods.zne import linear_weights, poly_weights, richardson_weights
    if rule == "richardson":
        return richardson_weights(lams)
    if rule == "linear":
        return linear_weights(lams)
    if rule == "poly2":
        return poly_weights(lams, 2)
    raise ValueError("ZNE rule %r (allowed: %s)" % (rule, ", ".join(ZNE_RULES)))


# ====================================================================== the suite (manifest → objects)
@dataclass(frozen=True)
class Method:
    key: str
    kind: str                                  # raw | ro | zne
    folding: str = "global"
    scales: tuple = (1.0, 3.0, 5.0)
    rule: str = "richardson"
    subset: str = "random"                     # local folding, fractional λ: which gates are folded once more

    def to_json(self) -> dict:
        d = dict(key=self.key, kind=self.kind)
        if self.kind == "zne":
            d.update(folding=self.folding, scales=list(self.scales), rule=self.rule, subset=self.subset)
        return d

    @property
    def fold_kind(self) -> str:
        """The folding part of a run role: global, local (random subset) or local.<subset>."""
        return self.folding if (self.folding == "global" or self.subset == "random") else "local." + self.subset


@dataclass(frozen=True)
class Benchmark:
    family: str
    n: int
    params: tuple                              # sorted (name, value) pairs

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def key(self) -> str:
        return "%s|%d|%s" % (self.family, self.n, json.dumps(self.p, sort_keys=True))

    @property
    def seed(self) -> int:
        return int.from_bytes(hashlib.sha256(self.key.encode()).digest()[:4], "little")


@dataclass
class Suite:
    study: str
    device: str
    fake: str
    budget: int
    cal_shots: int
    methods: tuple
    benchmarks: tuple
    opt_level: int = 1
    seed_transpiler: int = 11
    layout: object = "chain"                   # "chain" (best chain per width on the day's calibration) or {n: [...]}
    replicates: int = 400
    bootstrap: int = 200
    exact_max_qubits: int = 8
    aer_dm_max_qubits: int = 12
    mc_shots: int = 100000
    predictors: tuple = PREDICTORS
    variants: int = 1                          # equivalent circuits per benchmark (Chen et al.: 25 variants)
    variant_mode: str = "layouts"              # layouts | twirl | layouts+twirl | seeds (VARIANT_MODES)
    pv_threshold: int | None = None            # plurality-vote starting threshold t (default ≈ 0.28 · variants)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["methods"] = [m.to_json() for m in self.methods]
        d["benchmarks"] = [dict(family=b.family, n=b.n, **b.p) for b in self.benchmarks]
        d["predictors"] = list(self.predictors)
        return d


SUITE_KEYS = {"manifest_version", "study", "kind", "description", "device", "fake", "budget", "cal_shots", "methods",
              "circuits", "opt_level", "seed_transpiler", "layout", "replicates", "bootstrap", "exact_max_qubits",
              "aer_dm_max_qubits", "mc_shots", "predictors", "variants"}
# How the variants of a benchmark differ. Two independent ways to vary a circuit without changing its ideal output
# (Chen et al.'s variant design, Quantum 8, 1516 (2024), changes both: the gate frames and the qubit assignment):
#   layouts        another qubit assignment: a disjoint chain of the chip per variant (reversed chains when the chip
#                  runs out of disjoint ones; never the same assignment twice)
#   twirl          another gate frame: an independent Pauli twirl of every two-qubit gate, same chain
#   layouts+twirl  both at once
#   seeds          another transpiler seed (routing / synthesis choices), same chain
VARIANT_MODES = ("layouts", "twirl", "layouts+twirl", "seeds")


def suite_from_manifest(m: dict) -> Suite:
    """Validate a divergence manifest (YAML loaded as a dict). Unknown keys, wrong types and impossible budgets are
    errors, never defaults."""
    from .phase.families import FAMILIES
    from .phase.manifest import expand_circuits
    if not isinstance(m, dict):
        raise ValueError("a divergence manifest is a mapping")
    bad = set(m) - SUITE_KEYS
    if bad:
        raise ValueError("unknown manifest keys: %s" % sorted(bad))
    if m.get("kind", "divergence") != "divergence":
        raise ValueError("kind must be 'divergence'")
    for k in ("study", "device", "budget", "cal_shots", "methods", "circuits"):
        if k not in m:
            raise ValueError("the manifest needs %r" % k)
    budget, cal = m["budget"], m["cal_shots"]
    if not (isinstance(budget, int) and budget >= 1 and isinstance(cal, int) and cal >= 1):
        raise ValueError("budget and cal_shots must be positive integers")
    methods = []
    for key, d in m["methods"].items():
        d = dict(d)
        kind = d.pop("kind", None)
        if kind not in METHOD_KINDS:
            raise ValueError("method %s: kind must be one of %s" % (key, ", ".join(METHOD_KINDS)))
        if kind != "zne" and d:
            raise ValueError("method %s: %s takes no parameters (%s)" % (key, kind, sorted(d)))
        if kind == "zne":
            extra = set(d) - {"folding", "scales", "rule", "subset"}
            if extra:
                raise ValueError("method %s: unknown ZNE parameters %s" % (key, sorted(extra)))
            if d.get("subset", "random") not in FOLD_SUBSETS:
                raise ValueError("method %s: subset must be one of %s" % (key, ", ".join(FOLD_SUBSETS)))
            if d.get("rule") == "poly2" and len(d.get("scales", (1, 3, 5))) < 3:
                raise ValueError("method %s: the quadratic fit needs at least three scales" % key)
            sc = tuple(float(x) for x in d.get("scales", (1.0, 3.0, 5.0)))
            if len(set(sc)) < 2 or len(set(sc)) != len(sc) or min(sc) < 1:
                raise ValueError("method %s: ZNE needs at least two scales, distinct and all >= 1" % key)
            if d.get("folding", "global") not in ("global", "local"):
                raise ValueError("method %s: folding must be global or local" % key)
            if d.get("rule", "richardson") not in ZNE_RULES:
                raise ValueError("method %s: rule must be one of %s" % (key, ", ".join(ZNE_RULES)))
            methods.append(Method(key, kind, d.get("folding", "global"), tuple(sorted(sc)),
                                  d.get("rule", "richardson"), d.get("subset", "random")))
            need = 2 * cal + len(sc)
        else:
            methods.append(Method(key, kind))
            need = 2 * cal + 1 if kind == "ro" else 1
        nvar = int((m.get("variants") or {}).get("count", 1)) if isinstance(m.get("variants"), dict) else 1
        if budget // max(nvar, 1) < need:
            raise ValueError("method %s needs a budget of at least %d shots per variant (budget %d over %d variants)"
                             % (key, need, budget, nvar))
    if len({m_.key for m_ in methods}) != len(methods):
        raise ValueError("duplicate method keys")
    benches = []
    for fam, n, p in expand_circuits(m["circuits"]):
        if fam not in FAMILIES:
            raise ValueError("unknown family %r (choose from %s)" % (fam, ", ".join(FAMILIES)))
        benches.append(Benchmark(fam, int(n), tuple(sorted(p.items()))))
    if len({b.key for b in benches}) != len(benches):
        raise ValueError("duplicate circuits in the manifest")
    preds = tuple(m.get("predictors", PREDICTORS))
    if not preds or any(p not in PREDICTORS for p in preds):
        raise ValueError("predictors must be a non-empty subset of %s" % (PREDICTORS,))
    lay = m.get("layout", "chain")
    if lay != "chain" and not isinstance(lay, dict):
        raise ValueError("layout must be 'chain' or a mapping {n: [physical qubits]}")
    ints = {}
    for k, lo in (("opt_level", 0), ("seed_transpiler", 0), ("replicates", 2), ("bootstrap", 2),
                  ("exact_max_qubits", 0), ("aer_dm_max_qubits", 0), ("mc_shots", 100)):
        if k in m:
            if not (isinstance(m[k], int) and m[k] >= lo):
                raise ValueError("%s must be an integer >= %d" % (k, lo))
            ints[k] = int(m[k])
    if ints.get("opt_level", 1) > 3:
        raise ValueError("opt_level must be 0..3")
    var = m.get("variants") or {}
    if not isinstance(var, dict) or set(var) - {"count", "mode", "threshold"}:
        raise ValueError("variants must be a mapping {count, mode, threshold}")
    vcount = var.get("count", 1)
    if not (isinstance(vcount, int) and vcount >= 1):
        raise ValueError("variants.count must be an integer >= 1")
    vmode = var.get("mode", "layouts")
    if vmode not in VARIANT_MODES:
        raise ValueError("variants.mode must be one of %s" % ", ".join(VARIANT_MODES))
    thr = var.get("threshold")
    if thr is not None and not (isinstance(thr, int) and 2 <= thr <= vcount):
        raise ValueError("variants.threshold must be an integer in [2, count]")
    if vmode.startswith("layouts") and vcount > 1 and lay != "chain":
        raise ValueError("layout variants need layout: chain (one chain per variant)")
    return Suite(study=str(m["study"]), device=str(m["device"]), fake=str(m.get("fake", m["device"])),
                 budget=int(budget), cal_shots=int(cal), methods=tuple(methods), benchmarks=tuple(benches),
                 layout=lay, predictors=preds, variants=int(vcount), variant_mode=vmode, pv_threshold=thr, **ints)


def benchmark_label(family: str, n: int, p: dict) -> str:
    if family == "qft":
        return "QFT-%d" % n if p.get("m") is None else "AQFT-%d (m=%d)" % (n, p["m"])
    if family == "ghz":
        return "GHZ-%d%s" % (n, " (Z)" if p.get("basis", "X") == "Z" else "")
    if family == "ising":
        return "Ising-%d (r=%d)" % (n, p["steps"])
    if family == "brickwork":
        return "Brick-%d (L=%d, seed %d)" % (n, p["layers"], p.get("seed", 0))
    if family == "grover":
        return "Grover-%d (k=%d)" % (n, p.get("k", 1))
    if family == "bv":
        return "BV-%d" % n
    if family == "context":
        return "Context-%d (L=%d, %s)" % (n, p.get("layers", 4), p.get("spectators", "zero"))
    return "%s-%d" % (family, n)


# ====================================================================== layout on a backend
def _native_two(target) -> str:
    return next((g for g in ("cz", "ecr", "cx") if g in target.operation_names), "cx")


def chain_for_backend(backend, n: int, beam: int = 4096, target=None, exclude=frozenset(), avoid=frozenset()) -> tuple:
    """The n-qubit path of couplers with the smallest summed calibrated error (two-qubit gate errors of its edges,
    readout and √X errors of its qubits), avoiding parts the calibration marks as broken. Beam search over paths grown
    at both ends (exhaustive while fewer than ``beam`` paths exist); deterministic (ties: smallest path)."""
    from .device import good_target_for
    tg = target if target is not None else good_target_for(backend)
    two = _native_two(tg)
    qerr = {}
    for q in range(backend.num_qubits):
        try:
            e_m = tg["measure"][(q,)].error
            e_s = tg["sx"][(q,)].error
        except (KeyError, TypeError):
            continue
        if q in exclude:
            continue
        qerr[q] = float(e_m or 0.0) + float(e_s or 0.0)
    edge = {}
    for qa, ip in tg[two].items():
        if qa is None or len(qa) != 2 or qa[0] not in qerr or qa[1] not in qerr:
            continue
        e = float(ip.error) if ip is not None and ip.error is not None else 0.0
        k = (min(qa), max(qa))
        edge[k] = min(edge.get(k, math.inf), e)
    nb = {}
    for a, b in edge:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    if n < 1:
        raise ValueError("n >= 1")
    paths = {(q,): qerr[q] for q in sorted(qerr)}
    for _ in range(n - 1):
        nxt = {}
        for path, c in paths.items():
            for end, other in ((path[-1], None), (path[0], None)):
                for q in sorted(nb.get(end, ())):
                    if q in path:
                        continue
                    new = path + (q,) if end == path[-1] else (q,) + path
                    if new[0] > new[-1]:
                        new = new[::-1]
                    cost = c + qerr[q] + edge[(min(end, q), max(end, q))]
                    if new not in nxt or cost < nxt[new]:
                        nxt[new] = cost
        if not nxt:
            raise ValueError("no %d-qubit chain of working couplers on %s" % (n, getattr(backend, "name", "backend")))
        paths = dict(sorted(nxt.items(), key=lambda kv: (round(kv[1], 15), kv[0]))[:beam])
    cand = [kv for kv in paths.items() if kv[0] not in avoid and kv[0][::-1] not in avoid]
    if not cand:
        raise ValueError("no further %d-qubit chain on %s" % (n, getattr(backend, "name", "backend")))
    best = min(cand, key=lambda kv: (round(kv[1], 15), kv[0]))
    return tuple(int(q) for q in best[0])


def chain_target(backend, chain):
    """The backend's working target restricted to the chain's qubits and couplers: routing cannot leave the chain."""
    from qiskit.transpiler import Target
    from .device import good_target_for
    base = good_target_for(backend)
    keep = set(int(q) for q in chain)
    qprops = [backend.qubit_properties(q) for q in range(backend.num_qubits)] \
        if hasattr(backend, "qubit_properties") else None
    try:
        new = Target(num_qubits=backend.num_qubits, dt=backend.dt, qubit_properties=qprops)
    except Exception:                                            # noqa: BLE001 - qubit properties are optional
        new = Target(num_qubits=backend.num_qubits, dt=backend.dt)
    for nm in base.operation_names:
        props = {qa: ip for qa, ip in base[nm].items() if qa is not None and all(q in keep for q in qa)}
        if props:
            new.add_instruction(base.operation_from_name(nm), props)
    return new


# ====================================================================== plan
@dataclass
class Instance:
    key: str
    family: str
    n: int
    params: dict
    label: str
    seed: int
    phys: tuple
    mm: dict
    spec: CircuitSpec                            # as executed, on the compact qubits 0..n-1 (phys[i] = qubit i)
    ideal: np.ndarray                            # ideal distribution over the classical bits
    two_qubit: int
    depth: int
    logical_two_qubit: int
    routing_overhead: int
    norm_depth: float                            # Lubinski et al.: depth after transpiling to {rx, ry, rz, cx}
    group: str = ""                              # the benchmark this circuit is a variant of ("" = itself)
    variant: int = 0
    fold_weights: list | None = None             # calibrated error of every two-qubit gate (balanced fractional folds)
    features: dict = field(default_factory=dict)  # context features (qemx.context_noise.circuit_features)

    @property
    def bench(self) -> str:
        return self.group or self.key

    def to_json(self) -> dict:
        nz = np.nonzero(self.ideal > 1e-15)[0]
        return dict(key=self.key, family=self.family, n=self.n, params=self.params, label=self.label, seed=self.seed,
                    phys=list(self.phys), mm={str(k): int(v) for k, v in self.mm.items()}, spec=self.spec.to_json(),
                    ideal={str(int(i)): float(self.ideal[i]) for i in nz}, two_qubit=self.two_qubit,
                    depth=self.depth, logical_two_qubit=self.logical_two_qubit,
                    routing_overhead=self.routing_overhead, norm_depth=self.norm_depth, group=self.group,
                    variant=self.variant, fold_weights=self.fold_weights, features=self.features)

    @staticmethod
    def from_json(d: dict) -> "Instance":
        n = int(d["n"])
        ideal = np.zeros(2 ** n)
        for k, v in d["ideal"].items():
            ideal[int(k)] = float(v)
        return Instance(d["key"], d["family"], n, dict(d["params"]), d["label"], int(d["seed"]),
                        tuple(int(q) for q in d["phys"]), {int(k): int(v) for k, v in d["mm"].items()},
                        CircuitSpec.from_json(d["spec"]), ideal, int(d["two_qubit"]), int(d["depth"]),
                        int(d["logical_two_qubit"]), int(d["routing_overhead"]), float(d["norm_depth"]),
                        d.get("group", ""), int(d.get("variant", 0)), d.get("fold_weights"),
                        dict(d.get("features") or {}))


@dataclass
class Run:
    key: str                                     # "<instance key>|<method>|<role>"
    instance: str
    method: str
    role: str                                    # target | fold:<scale> | cal0 | cal1
    shots: int
    lam: float | None = None                     # achieved λ (two-qubit gates ÷ original) of a folded circuit
    digest: str = ""                             # SHA-256 of the as-executed circuit (compact qubits)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def spec_digest(spec: CircuitSpec) -> str:
    return hashlib.sha256(json.dumps(spec.to_json(), sort_keys=True).encode()).hexdigest()[:20]


def calibration_spec(n: int, ones: bool) -> CircuitSpec:
    ops = tuple(Op("x", (q,), (), "gate", "cal") for q in range(n)) if ones else ()
    return CircuitSpec(n, ops, "readout calibration |%s⟩" % ("1" * n if ones else "0" * n))


def run_spec(inst: Instance, role: str) -> CircuitSpec:
    """The as-executed circuit of a run: a deterministic function of the instance (folds use the instance seed)."""
    from .transforms import fold_global, fold_local
    if role == "target":
        return inst.spec
    if role == "cal0":
        return calibration_spec(inst.n, False)
    if role == "cal1":
        return calibration_spec(inst.n, True)
    if role.startswith("fold:"):
        kind, lam = role.split(":")[1:] if role.count(":") == 2 else ("global", role.split(":")[1])
        lam = float(lam)
        if kind == "global":
            return fold_global(inst.spec, lam)
        folding, _, subset = kind.partition(".")              # local | local.nested | local.stratified | …
        if folding != "local":
            raise ValueError("unknown folding in run role %r" % role)
        return fold_local(inst.spec, lam, inst.seed, subset or "random",
                          inst.fold_weights if subset == "balanced" else None)
    raise ValueError("unknown run role %r" % role)


def method_runs(inst: Instance, m: Method, budget: int, cal: int) -> list:
    """The runs of one method on one instance at total budget ``budget`` (calibration shots included)."""
    from .methods.zne import largest_remainder_split
    from .transforms import achieved_lambda
    base = "%s|%s|" % (inst.key, m.key)
    if m.kind == "raw":
        return [Run(base + "target", inst.key, m.key, "target", int(budget))]
    N = int(budget) - 2 * int(cal)
    cal_runs = [Run(base + "cal0", inst.key, m.key, "cal0", int(cal)),
                Run(base + "cal1", inst.key, m.key, "cal1", int(cal))]
    if m.kind == "ro":
        return [Run(base + "target", inst.key, m.key, "target", N)] + cal_runs
    roles = ["fold:%s:%g" % (m.fold_kind, s) for s in m.scales]
    specs = [run_spec(inst, r) for r in roles]
    lams = [achieved_lambda(inst.spec, sp) for sp in specs]
    if len(set(np.round(lams, 9))) < len(lams):
        raise ValueError("%s on %s: the scales %s fold to the same λ" % (m.key, inst.label, list(m.scales)))
    eta = extrapolation_weights(m.rule, lams)
    Nj = largest_remainder_split(np.abs(eta), N)
    return [Run(base + r, inst.key, m.key, r, int(nj), float(l)) for r, nj, l in zip(roles, Nj, lams)] + cal_runs


def ideal_distribution(spec: CircuitSpec, mm: dict) -> np.ndarray:
    """Exact output distribution over the classical bits (statevector; no width limit other than memory)."""
    from qiskit.quantum_info import Statevector
    from .circuit import to_qiskit
    from .engine import permute_probs
    p = Statevector(to_qiskit(spec)).probabilities()
    p = np.clip(np.real(p), 0, None)
    return permute_probs(p / p.sum(), mm, spec.n)


def lubinski_depth(logical: CircuitSpec, seed: int = 11) -> float:
    """Normalized depth (Lubinski et al. §III.C): depth after transpiling to {rx, ry, rz, cx}, all-to-all coupling."""
    from qiskit import transpile
    from .circuit import to_qiskit
    from .compile import QISKIT_LOCK
    with QISKIT_LOCK:
        tq = transpile(to_qiskit(logical), basis_gates=["rx", "ry", "rz", "cx"], optimization_level=1,
                       seed_transpiler=int(seed))
    return float(tq.depth(lambda ins: ins.operation.name not in ("barrier",)))


def backend_twoq_errors(backend, spec: CircuitSpec, phys) -> list | None:
    """Calibrated error of every two-qubit gate of ``spec`` (compact qubits; ``phys`` maps them to the device), in
    circuit order — the weights of the error-balanced fractional fold. None when the target reports none."""
    tg = backend.target
    out = []
    for o in spec.ops:
        if o.kind != "gate" or len(o.qubits) != 2:
            continue
        a, b_ = (int(phys[q]) for q in o.qubits)
        e = None
        for qa in ((a, b_), (b_, a)):
            try:
                ip = tg[o.name][qa]
                e = None if ip is None else ip.error
            except (KeyError, TypeError):
                continue
            if e is not None:
                break
        out.append(math.nan if e is None else float(e))
    w = np.asarray(out, float)
    if not len(w) or not np.isfinite(w).any():
        return None
    w[~np.isfinite(w)] = float(np.nanmean(w))
    return [float(x) for x in w]


def instance_features(spec: CircuitSpec, phys, backend) -> dict:
    """Context features of the as-executed circuit (qemx.context_noise) on the backend's coupling graph, with the
    gate durations of its calibration."""
    from .context_noise import circuit_features, device_neighbors
    from .device import model_for
    dm = model_for(backend, tuple(int(p) for p in phys))
    return circuit_features(spec, phys, device_neighbors(backend), dm.durations, dm.two_qubit, errors=dm.errors,
                            readout=[(q.p01, q.p10) for q in dm.qubits])


def build_instance(b: Benchmark, backend, suite: Suite, *, chain=None, variant: int = 0, twirl_seed=None,
                   seed_transpiler=None) -> Instance:
    """Compile a benchmark for the backend on a chain of its best qubits (routing confined to the chain). A variant
    (variant > 0, or several variants) is the same benchmark in another physical form: another chain (``chain``),
    a Pauli twirl of every two-qubit gate (``twirl_seed``) or another transpiler seed."""
    from .compile import with_io
    from .device import compile_for_target
    from .phase.families import build
    from .transforms import twirl_instance
    fc = build(b.family, b.n, **b.p)
    logical = with_io(fc.spec, None, fc.basis)
    if chain is None:
        if suite.layout == "chain":
            chain = chain_for_backend(backend, fc.n)
        else:
            lay = suite.layout.get(fc.n, suite.layout.get(str(fc.n)))
            if lay is None:
                raise ValueError("layout has no entry for n = %d" % fc.n)
            chain = tuple(int(q) for q in lay)
    tg = chain_target(backend, chain)
    st = suite.seed_transpiler if seed_transpiler is None else int(seed_transpiler)
    spec, mm, phys, info = compile_for_target(logical, backend, tg, getattr(backend, "name", "backend"),
                                              suite.opt_level, st, initial_layout=list(chain))
    if fc.relabel:
        mm = {i: mm[fc.n - 1 - i] for i in range(fc.n)}
    if set(phys) != set(chain):
        raise RuntimeError("%s left its chain %s (used %s)" % (fc.name, chain, phys))
    if twirl_seed is not None:
        spec = twirl_instance(spec, int(twirl_seed)).replace(name=spec.name + " (twirl %d)" % int(twirl_seed))
    try:
        nd = lubinski_depth(logical, suite.seed_transpiler)
    except Exception:                                   # noqa: BLE001 - informational
        nd = math.nan
    many = suite.variants > 1
    key = b.key + ("#v%d" % variant if many else "")
    label = benchmark_label(b.family, fc.n, b.p) + (" · v%d" % variant if many else "")
    try:
        feats = instance_features(spec, phys, backend)
    except Exception:                                   # noqa: BLE001 - features are an analysis aid
        feats = {}
    return Instance(key, b.family, fc.n, b.p, label, b.seed + 7919 * variant, tuple(phys), mm, spec,
                    ideal_distribution(spec, mm), int(spec.count("2q")), int(spec.depth()),
                    int(info["logical_two_qubit"]), int(info["routing_overhead"]), nd,
                    group=b.key if many else "", variant=int(variant),
                    fold_weights=backend_twoq_errors(backend, spec, phys), features=feats)


def disjoint_chains(backend, n: int, count: int) -> list:
    """``count`` distinct qubit assignments for the layout variants of an n-qubit benchmark, in this order:

    1. disjoint chains in order of calibrated quality, each avoiding the qubits of the earlier ones (the variants
       spread over the chip: the heavy-hex counterpart of a per-variant qubit assignment);
    2. when the chip has no disjoint chain left, the same chains reversed (logical qubit i on chain position
       n − 1 − i: the same couplers, other roles for the qubits);
    3. then overlapping chains, never one already used (in either orientation).

    Fewer than ``count`` only when the device has no further n-qubit chain at all."""
    out, used = [], set()
    count = int(count)
    while len(out) < count:
        try:
            c = chain_for_backend(backend, n, exclude=frozenset(used))
        except ValueError:
            break
        out.append(tuple(c))
        used |= set(c)
    if len(out) < count and n > 1:                      # the chip ran out of disjoint chains: reversed orientations
        for c in list(out):
            if len(out) >= count:
                break
            out.append(tuple(reversed(c)))
    while len(out) < count:                             # overlapping chains, no assignment twice
        seen = frozenset(out) | frozenset(tuple(reversed(x)) for x in out)
        try:
            c = chain_for_backend(backend, n, exclude=frozenset(), avoid=seen)
        except ValueError:
            break
        if tuple(c) in seen:
            break
        out.append(tuple(c))
    return out


@dataclass
class Plan:
    suite: Suite
    backend: dict                                # name, num_qubits, two_qubit, kind (fake | ibm)
    instances: list
    runs: list
    errors: list = field(default_factory=list)   # [{instance, error}] benchmarks that could not be planned
    created: str = ""

    def instance(self, key: str) -> Instance:
        return self._by_key()[key]

    def _by_key(self) -> dict:
        if getattr(self, "_idx", None) is None or len(self._idx) != len(self.instances):
            self._idx = {i.key: i for i in self.instances}
        return self._idx

    def runs_of(self, inst_key: str, method: str | None = None) -> list:
        return [r for r in self.runs if r.instance == inst_key and (method is None or r.method == method)]

    def spec_of(self, run: Run) -> CircuitSpec:
        spec = run_spec(self.instance(run.instance), run.role)
        if run.digest and spec_digest(spec) != run.digest:
            raise RuntimeError("run %s: the rebuilt circuit differs from the planned one (digest mismatch); the "
                               "plan was made by another version of qemx" % run.key)
        return spec

    def summary(self) -> dict:
        shots = sum(r.shots for r in self.runs)
        return dict(instances=len(self.instances), runs=len(self.runs), circuits_distinct=len({r.digest for r in
                    self.runs}), shots=int(shots), methods=[m.key for m in self.suite.methods],
                    widths=sorted({i.n for i in self.instances}), errors=len(self.errors))

    def to_json(self) -> dict:
        return dict(schema=SCHEMA, created=self.created, suite=self.suite.to_json(), backend=self.backend,
                    instances=[i.to_json() for i in self.instances], runs=[r.to_json() for r in self.runs],
                    errors=self.errors)

    @staticmethod
    def from_json(d: dict) -> "Plan":
        if int(d.get("schema", 0)) != SCHEMA:
            raise ValueError("plan schema %s (this version reads %d)" % (d.get("schema"), SCHEMA))
        s = dict(d["suite"])
        s["methods"] = tuple(Method(m["key"], m["kind"], m.get("folding", "global"),
                                    tuple(float(x) for x in m.get("scales", (1.0, 3.0, 5.0))),
                                    m.get("rule", "richardson"), m.get("subset", "random")) for m in s["methods"])
        s["benchmarks"] = tuple(Benchmark(b.pop("family"), int(b.pop("n")), tuple(sorted(b.items())))
                                for b in (dict(x) for x in s["benchmarks"]))
        s["predictors"] = tuple(s.get("predictors", PREDICTORS))
        suite = Suite(**{k: v for k, v in s.items() if k in Suite.__dataclass_fields__})
        return Plan(suite, dict(d["backend"]), [Instance.from_json(i) for i in d["instances"]],
                    [Run(**r) for r in d["runs"]], list(d.get("errors", [])), d.get("created", ""))


def make_plan(suite: Suite, backend, kind: str = "fake", progress=None) -> Plan:
    """Compile every benchmark for ``backend`` and lay out the runs of every method. A benchmark that cannot be
    compiled or simulated ideally (e.g. memory) is recorded in ``plan.errors`` and skipped; there is no size limit."""
    from .phase.families import build
    from .resources import RunTimeout
    instances, runs, errors = [], [], []
    V = int(suite.variants)
    per_variant = int(suite.budget) // V                 # the budget of a benchmark is shared by its variants
    for i, b in enumerate(suite.benchmarks):
        try:
            if V == 1:
                forms = [dict()]
            elif suite.variant_mode in ("layouts", "layouts+twirl"):
                width = build(b.family, b.n, **b.p).n
                both = suite.variant_mode == "layouts+twirl"
                forms = [dict(chain=c, **(dict(twirl_seed=b.seed + 101 * j) if both else {}))
                         for j, c in enumerate(disjoint_chains(backend, width, V))]
            elif suite.variant_mode == "twirl":
                forms = [dict(twirl_seed=b.seed + 101 * j) for j in range(V)]
            else:
                forms = [dict(seed_transpiler=suite.seed_transpiler + j) for j in range(V)]
        except Exception as e:                           # noqa: BLE001 - one benchmark; the plan continues
            errors.append(dict(instance=b.key, error="%s: %s" % (type(e).__name__, str(e)[:300])))
            continue
        for j, form in enumerate(forms):
            try:
                inst = build_instance(b, backend, suite, variant=j, **form)
                rr = []
                for m in suite.methods:
                    try:                                 # a method that cannot run here (e.g. fractional scales
                        mr = method_runs(inst, m, per_variant, suite.cal_shots)   # on 3 gates) skips this circuit
                    except ValueError as e:
                        errors.append(dict(instance=inst.key, method=m.key, error="ValueError: %s" % str(e)[:300]))
                        continue
                    rr += mr
                for r in rr:
                    r.digest = spec_digest(run_spec(inst, r.role))
            except (MemoryError, RunTimeout) as e:
                errors.append(dict(instance=b.key + ("#v%d" % j if V > 1 else ""),
                                   error="%s: %s" % (type(e).__name__, str(e)[:300])))
                continue
            except Exception as e:                       # noqa: BLE001 - one variant; the plan continues
                errors.append(dict(instance=b.key + ("#v%d" % j if V > 1 else ""),
                                   error="%s: %s" % (type(e).__name__, str(e)[:300])))
                continue
            instances.append(inst)
            runs += rr
        if progress:
            progress((i + 1) / len(suite.benchmarks), b.key)
    two = _native_two(backend.target)
    from .context_noise import device_neighbors
    edges = sorted({(min(a, c), max(a, c)) for a, nb in device_neighbors(backend).items() for c in nb})
    return Plan(suite, dict(name=getattr(backend, "name", "backend"), num_qubits=int(backend.num_qubits),
                            two_qubit=two, kind=kind, edges=[list(e) for e in edges]), instances, runs, errors,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# ====================================================================== execution (hardware, local, emulated)
def physical_circuits(plan: Plan, backend, runs=None) -> list:
    """[(run key, ISA QuantumCircuit on the backend's qubits, shots)]: translated gate by gate (optimization level 0,
    identity layout) — every fold reaches the device unaltered (two-qubit counts checked)."""
    from types import SimpleNamespace
    from qiskit import transpile
    from .compile import QISKIT_LOCK
    from .hardware import fold_guard_transpiled, to_physical
    runs = plan.runs if runs is None else runs
    N = int(backend.num_qubits)
    pre = []
    for r in runs:
        inst = plan.instance(r.instance)
        pre.append((r.key, to_physical(SimpleNamespace(spec=plan.spec_of(r), mm=inst.mm), list(inst.phys), N),
                    int(r.shots)))
    with QISKIT_LOCK:
        circs = transpile([p[1] for p in pre], target=backend.target, optimization_level=0,
                          initial_layout=list(range(N)), seed_transpiler=int(plan.suite.seed_transpiler))
    if not isinstance(circs, list):
        circs = [circs]
    fold_guard_transpiled([p[1] for p in pre], circs, [p[0] for p in pre])
    return [(k, c, s) for (k, _, s), c in zip(pre, circs)]


def counts_from_pub(pub_result) -> dict:
    from .hardware import _pub_counts
    return {str(int(b.replace(" ", ""), 2)): int(c) for b, c in _pub_counts(pub_result).items()}


def _by_shots(items) -> list:
    groups = {}
    for it in items:
        groups.setdefault(int(it[2]), []).append(it)
    return list(groups.items())


def aer_simulator(dm, method: str = "automatic"):
    """AerSimulator with the device model's gate errors (depolarizing + relaxation over each gate) and readout errors
    on the compact qubits — what ``AerSimulator.from_backend`` builds, restricted to the qubits used."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from qiskit_aer.noise.device.models import basic_device_gate_errors
    from .device import _LOCK, _sub_target
    with _LOCK:
        nm = NoiseModel(basis_gates=sorted({"rz", "sx", "x", dm.two_qubit, "id"}))
        for name, qargs, err in basic_device_gate_errors(target=_sub_target(dm), thermal_relaxation=True):
            if name not in ("measure", "delay", "reset"):
                nm.add_quantum_error(err, name, list(qargs))
        for i, q in enumerate(dm.qubits):
            if q.p01 > 0 or q.p10 > 0:
                nm.add_readout_error(ReadoutError([[1 - q.p01, q.p01], [q.p10, 1 - q.p10]]), [i])
    return AerSimulator(noise_model=nm, method=method)


def run_local(plan: Plan, backend, models: dict, seed: int = 0, progress=None) -> dict:
    """Execute the plan with the Sampler primitive in local mode: the ISA circuits sent to the device, compacted to
    each instance's qubits and run on Aer with that instance's calibration model (``models``: {instance key:
    DeviceModel}). Returns counts {run key: {outcome index: count}}."""
    from qiskit_ibm_runtime import SamplerV2
    from .hardware import _compact
    out = {}
    circs = physical_circuits(plan, backend)
    by_inst = {}
    for k, c, s in circs:
        by_inst.setdefault(k.rsplit("|", 2)[0], []).append((k, c, s))
    done = 0
    for j, (ik, items) in enumerate(by_inst.items()):
        inst = plan.instance(ik)
        dm = models[ik]
        index = {p: i for i, p in enumerate(inst.phys)}
        sim = aer_simulator(dm)
        for shots, grp in _by_shots(items):
            sampler = SamplerV2(mode=sim)
            sampler.options.simulator.seed_simulator = int(seed + 7919 * j + shots) % (2 ** 31)
            res = sampler.run([_compact(c, index) for _, c, _ in grp], shots=shots).result()
            for (k, _, _), pr in zip(grp, res):
                out[k] = counts_from_pub(pr)
        done += len(items)
        if progress:
            progress(done / len(circs), inst.label)
    return out


def _plain_sampler(sampler) -> None:
    """The circuits must run exactly as written: no dynamical decoupling (it would fill the idle windows the idle
    feature and the folding fingerprint measure) and no service-side twirling (variants are planned explicitly).
    Both are off by default for SamplerV2; they are set explicitly so a changed default cannot alter the experiment.
    Options a runtime version does not have are skipped (API drift: see the research plan's debugging guide)."""
    opts = getattr(sampler, "options", None)
    for path, value in (("dynamical_decoupling.enable", False), ("twirling.enable_gates", False),
                        ("twirling.enable_measure", False)):
        obj = opts
        try:
            *head, last = path.split(".")
            for h in head:
                obj = getattr(obj, h)
            setattr(obj, last, value)
        except (AttributeError, TypeError, ValueError):
            pass


def submit(plan: Plan, backend, max_execution_time: int | None = None, skip=(), ledger=None, model=None,
           mode: str = "job", campaign: str = "", on_record=None) -> list:
    """Submit the runs to a real backend, except the run keys in ``skip`` (runs of jobs already submitted: resuming a
    partial submission). Returns job records — job id, backend, shots and run keys; never a credential. A failure
    after some jobs went through raises :class:`qemx.hardware.SubmitError` carrying the records of the jobs submitted
    by this call.

    ``ledger`` (:class:`qemx.campaign.Ledger`): every job is admitted against the campaign's conservative remaining
    QPU allowance *before* it is submitted (its high usage estimate under ``model``), its id is recorded at once, and
    the first job that does not fit stops the submission (partial, resumable). Without an explicit
    ``max_execution_time`` each job gets one from its own high estimate. ``mode``: "job" (default; each job stands
    alone) or "batch" (one Batch; on the Open plan its 10-minute elapsed-time TTL can fail queued jobs).
    ``on_record(records)`` is called after every job with the records so far (to write them to disk at once)."""
    from qiskit_ibm_runtime import SamplerV2
    from .hardware import SubmitError
    if mode not in ("job", "batch"):
        raise ValueError("mode must be 'job' or 'batch'")
    skip = set(skip)
    circs = physical_circuits(plan, backend, [r for r in plan.runs if r.key not in skip])   # all checked first
    batch = None
    if mode == "batch":
        try:
            from qiskit_ibm_runtime import Batch
            batch = Batch(backend=backend)
        except Exception:                              # noqa: BLE001 - accounts without batch mode use job mode
            batch = None
    records = []
    try:
        for shots, grp in _by_shots(circs):
            for lo in range(0, len(grp), 300):           # at most 300 circuits per job
                part = grp[lo:lo + 300]
                rid, met = None, max_execution_time
                if ledger is not None:
                    from .campaign import CampaignLimit, estimate_usage
                    est = estimate_usage(int(shots) * len(part), len(part), 1, model)
                    try:
                        rid = ledger.admit("%s: %d circuits × %d shots" % (plan.suite.study, len(part), shots), est,
                                           campaign)
                    except CampaignLimit as e:
                        if not records:
                            raise
                        raise SubmitError("campaign ledger: %s" % e, records) from e
                    if not met:
                        met = int(math.ceil(est.high_s * 1.1)) + 1
                sampler = SamplerV2(mode=batch if batch is not None else backend)
                _plain_sampler(sampler)
                if met:
                    sampler.options.max_execution_time = int(met)
                try:
                    job = sampler.run([c for _, c, _ in part], shots=shots)
                    jid = job.job_id()
                except Exception as e:
                    if rid is not None:
                        ledger.release(rid, "submission failed: %s" % type(e).__name__)
                    if not records:
                        raise
                    raise SubmitError("%s: %s" % (type(e).__name__, e), records) from e
                if rid is not None:
                    ledger.submitted(rid, jid)
                records.append(dict(job_id=jid, backend=getattr(backend, "name", "backend"), shots=int(shots),
                                    runs=[k for k, _, _ in part], max_execution_time=met,
                                    submitted=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                if on_record is not None:                # persist every job id at once: a crash cannot lose one
                    on_record(list(records))
    finally:
        if batch is not None:
            try:
                batch.close()
            except Exception:                         # noqa: BLE001
                pass
    return records


def _job_usage(job):
    try:
        u = job.usage()                                # qiskit-ibm-runtime ≥ 0.30: QPU seconds
        return None if u is None else float(u)
    except Exception:                                  # noqa: BLE001 - not reported (yet)
        return None


def collect(service, records: list, wait: bool = False, ledger=None) -> tuple:
    """(counts or None, statuses): counts {run key: {outcome: count}} once every job has ended (a failed job's runs
    have no counts). Statuses carry the execution timestamps and the reported QPU usage where the service reports
    them; with a ``ledger`` every ended job is reconciled with its reported usage."""
    FIN, FAIL = ("DONE", "JobStatus.DONE"), ("ERROR", "CANCELLED", "JobStatus.ERROR", "JobStatus.CANCELLED")
    counts, status, ended = {}, [], True

    def recon(job, jid, st):
        u = _job_usage(job)
        if ledger is not None:
            state = "completed" if st in FIN else ("cancelled" if "CANCEL" in st else "failed")
            try:
                ledger.reconcile(jid, u, state)
            except KeyError:                           # submitted before the ledger existed
                pass
        return u
    for rec in records:
        job = service.job(rec["job_id"])
        st = str(job.status())
        entry = dict(job_id=rec["job_id"], status=st, failed=st in FAIL)
        if st in FAIL:
            entry["usage_s"] = recon(job, rec["job_id"], st)
            status.append(entry)
            continue
        if not wait and st not in FIN:
            ended = False
            status.append(entry)
            continue
        try:
            res = job.result()                         # waits; raises if the job ends in error or is cancelled
        except Exception as e:                         # noqa: BLE001 - this job's runs get no counts; others keep theirs
            entry.update(failed=True, status="%s (%s)" % (st, type(e).__name__))
            entry["usage_s"] = recon(job, rec["job_id"], "ERROR")
            status.append(entry)
            continue
        try:
            m = job.metrics()
            entry["executed"] = (m.get("timestamps") or {}).get("running")
        except Exception:                             # noqa: BLE001 - informational
            pass
        for k, pr in zip(rec["runs"], res):
            counts[k] = counts_from_pub(pr)
        entry["usage_s"] = recon(job, rec["job_id"], "DONE")
        status.append(entry)
    return (counts if ended else None), status


# ====================================================================== calibration snapshots and predictors
def dm_to_json(dm) -> dict:
    return dict(name=dm.name, phys=list(dm.phys), dt=float(dm.dt), two_qubit=dm.two_qubit,
                coherent=float(getattr(dm, "coherent", 0.0) or 0.0),
                qubits=[[float(q.t1), float(q.t2), float(q.p01), float(q.p10)] for q in dm.qubits],
                durations=[[nm, list(qa), float(v)] for (nm, qa), v in sorted(dm.durations.items())],
                errors=[[nm, list(qa), None if v is None else float(v)] for (nm, qa), v in sorted(dm.errors.items())])


def dm_from_json(d: dict):
    from .device import DeviceModel, QubitInfo
    return DeviceModel(d["name"], tuple(int(q) for q in d["phys"]), float(d["dt"]),
                       [QubitInfo(*map(float, q)) for q in d["qubits"]], d["two_qubit"],
                       {(nm, tuple(int(x) for x in qa)): float(v) for nm, qa, v in d["durations"]},
                       {(nm, tuple(int(x) for x in qa)): (None if v is None else float(v)) for nm, qa, v in d["errors"]},
                       coherent=float(d.get("coherent", 0.0)))


def snapshot(plan: Plan, backend) -> dict:
    """{instance key: DeviceModel JSON}: the backend's calibration on each instance's qubits, as reported now (or as
    reported at the time ``backend`` stands for; see ``backend_at`` in the script)."""
    from .device import model_for
    return {i.key: dm_to_json(model_for(backend, tuple(i.phys), "%s|%s" % (getattr(backend, "name", "b"),
                                                                          ",".join(map(str, i.phys)))))
            for i in plan.instances}


def predictor_model(dm, kind: str):
    """The noise model a predictor simulates (module docstring)."""
    from .device import DeviceModel, QubitInfo, homogenize
    if kind == "device":
        return dm
    if kind == "depolarizing":
        return DeviceModel(dm.name + "#depolarizing", dm.phys, dm.dt,
                           [QubitInfo(INFINITE_T, INFINITE_T, q.p01, q.p10) for q in dm.qubits], dm.two_qubit,
                           dict(dm.durations), dict(dm.errors))
    if kind == "chen":
        h = homogenize(dm)
        return DeviceModel(dm.name + "#chen", dm.phys, dm.dt, [QubitInfo(INFINITE_T, INFINITE_T, 0.0, 0.0)
                                                               for _ in dm.qubits], dm.two_qubit,
                           dict(h.durations), dict(h.errors))
    raise ValueError("unknown predictor %r (choose from %s)" % (kind, ", ".join(PREDICTORS)))


def crosstalk_corrections(spec: CircuitSpec, zeta: float):
    """Unmodeled coherent ZZ crosstalk (for emulation): after every two-qubit gate on (a, b), exp(−i ζ/2 Z⊗Z) on every
    coupled pair (a, s) and (b, s) with a spectator s — a context-dependent error no per-gate channel describes."""
    from .noise import Instruction
    edges = [tuple(e) for e in (spec.coupling or tuple((q, q + 1) for q in range(spec.n - 1)))]
    nb = {}
    for a, b in edges:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    U = np.diag(np.exp(-0.5j * zeta * np.array([1, -1, -1, 1])))

    def f(o, step):
        if o.kind != "gate" or len(o.qubits) != 2:
            return []
        a, b = o.qubits
        pairs = sorted({(min(x, s), max(x, s)) for x in (a, b) for s in nb.get(x, ()) if s not in (a, b)})
        return [Instruction("unitary", p, U, step, "explicit", "ZZ crosstalk") for p in pairs]
    return f


def relaxation_free(dm) -> bool:
    """True for the depolarizing predictors (T1 = T2 = INFINITE_T): their idle windows are identities to < 1e-9 (below
    the engines' agreement), so the simulation skips them."""
    return all(q.t1 >= INFINITE_T and q.t2 >= INFINITE_T for q in dm.qubits)


def _without_idle(qc):
    """The device circuit without its idle-window relaxation channels (``quantum_channel``); gate errors come from
    the noise model and stay."""
    out = qc.copy_empty_like()
    for ins in qc.data:
        if ins.operation.name != "quantum_channel":
            out.append(ins)
    return out


def measured_probs(spec: CircuitSpec, mm: dict, dm, *, engine: str = "aer-dm", shots: int = 100000, seed: int = 0,
                   corrections=None) -> np.ndarray:
    """Output distribution over the classical bits under the device model ``dm`` (readout included).

    engine ``exact`` (NumPy superoperators, qemx's E2: exact, coherent fractions and signed maps supported),
    ``aer-dm`` (Aer density matrix: the same channels, agrees with ``exact`` to ~1e-8, faster), ``aer-mc`` (Aer
    trajectories with ``shots`` samples: any width the memory allows; a Monte-Carlo estimate)."""
    from .device import device_circuit, device_noise_model, finish
    n = spec.n
    no_idle = relaxation_free(dm)
    if engine == "exact":
        from . import device_exact
        from .backends.numpy_dm import NumpyBackend
        instr, ex = device_exact.instructions(spec, dm, corrections=corrections)
        if no_idle:
            instr = [i for i in instr if i.source != "idle"]
        rho, _ = NumpyBackend().execute(instr, n)
        p = finish(rho, dm, mm, n, ex, None, exact=True).probs_meas
        p = np.clip(np.real(p), 0, None)
        return p / p.sum()
    if engine == "aer-dm":
        from qiskit_aer import AerSimulator
        qc, ex = device_circuit(spec, dm, corrections=corrections)
        if no_idle:
            qc = _without_idle(qc)
        res = AerSimulator(method="density_matrix", noise_model=device_noise_model(dm)).run(qc, shots=1).result()
        if not res.success:
            raise RuntimeError("Aer: %s" % getattr(res, "status", "failed"))
        rho = np.asarray(res.data(0)["final"])
        return finish(rho, dm, mm, n, ex, None).probs_meas
    if engine == "aer-mc":
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        qc, ex = device_circuit(spec, dm, corrections=corrections)
        out = QuantumCircuit(n, n)
        for ins in qc.data:
            if ins.operation.name == "save_density_matrix" or (no_idle and ins.operation.name == "quantum_channel"):
                continue
            out.append(ins.operation, [qc.find_bit(q).index for q in ins.qubits])
        for c in range(n):
            out.measure(mm[c], c)
        sim = AerSimulator(method="statevector", noise_model=device_noise_model(dm), seed_simulator=int(seed))
        res = sim.run(out, shots=int(shots)).result()
        if not res.success:
            raise RuntimeError("Aer: %s" % getattr(res, "status", "failed"))
        f = np.zeros(2 ** n)
        for b, c in res.get_counts(0).items():
            f[int(b.replace(" ", ""), 2)] += c
        f /= f.sum()
        # readout: bit i read from qubit mm[i] with that qubit's confusion matrix (as finish() does)
        A = dm.readout_matrices()
        t = f.reshape([2] * n)
        for i in range(n):
            ax = n - 1 - i
            t = np.moveaxis(np.tensordot(A[mm.get(i, i)], t, axes=([1], [ax])), 0, ax)
        return t.reshape(-1)
    raise ValueError("engine must be exact, aer-dm or aer-mc")


def engine_for(n: int, suite: Suite) -> str:
    if n <= suite.exact_max_qubits:
        return "exact"
    if n <= suite.aer_dm_max_qubits:
        return "aer-dm"
    return "aer-mc"


def _sub_plan(plan: Plan, inst: Instance) -> Plan:
    return Plan(plan.suite, plan.backend, [inst], plan.runs_of(inst.key), [], plan.created)


def _model(m):
    return dm_from_json(m) if isinstance(m, dict) else m


def _predict_one(args) -> tuple:
    """Worker: (instance key, {predictor: {run key: distribution}}, errors) of one instance."""
    from .resources import RunTimeout
    sub, model, kinds = args
    inst = sub.instances[0]
    eng = engine_for(inst.n, sub.suite)
    out, errors = {k: {} for k in kinds}, []
    for k in kinds:
        try:
            dm = predictor_model(_model(model), k)
            cache = {}
            for r in sub.runs:
                if r.digest not in cache:
                    cache[r.digest] = measured_probs(sub.spec_of(r), inst.mm, dm, engine=eng,
                                                     shots=sub.suite.mc_shots, seed=inst.seed)
                out[k][r.key] = cache[r.digest]
        except (MemoryError, RunTimeout) as e:
            errors.append(dict(instance=inst.key, predictor=k, error="%s: %s (no hard limit is imposed; lower "
                               "aer_dm_max_qubits to use trajectories)" % (type(e).__name__, str(e)[:200])))
        except Exception as e:                         # noqa: BLE001 - one instance; the rest continues
            errors.append(dict(instance=inst.key, predictor=k, error="%s: %s" % (type(e).__name__, str(e)[:300])))
    return inst.key, out, errors


def _ping(x):
    return x


def _pool_map(fn, tasks, workers: int, progress, labels):
    """(index, result or exception) of fn over tasks, in order of completion; with workers > 1 in a pool of spawned
    processes (one BLAS thread each). The pool is probed with a no-op task first: if workers cannot start at all
    (e.g. an interactive __main__ they cannot import) everything runs in this process. A worker that dies during the
    run (e.g. killed out of memory) breaks the pool: every unfinished task is reported as failed — never re-run here,
    where the same task could take this process down too."""
    if workers <= 1 or len(tasks) <= 1:
        for i, t in enumerate(tasks):
            yield i, fn(t)
            if progress:
                progress((i + 1) / len(tasks), labels[i])
        return
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    # spawn, not fork: a forked child of a process whose Aer/OpenMP threads already ran can deadlock
    ex = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"))
    try:
        ex.submit(_ping, 0).result()
    except BrokenProcessPool:
        ex.shutdown(wait=False, cancel_futures=True)
        ex = None
    if ex is None:                                     # workers cannot start: this process does the work
        for i, t in enumerate(tasks):
            yield i, fn(t)
            if progress:
                progress((i + 1) / len(tasks), labels[i])
        return
    with ex:
        futs = {ex.submit(fn, t): i for i, t in enumerate(tasks)}
        for done, f in enumerate(as_completed(futs)):
            i = futs[f]
            try:
                res = f.result()
            except BrokenProcessPool:
                res = RuntimeError("a worker process died (killed by the system, e.g. out of memory); rerun with "
                                   "fewer --workers or without the widest circuits")
            except Exception as e:                     # noqa: BLE001 - reported per task
                res = e
            yield i, res
            if progress:
                progress((done + 1) / len(tasks), labels[i])


def predict(plan: Plan, models: dict, kinds=None, progress=None, workers: int = 1) -> tuple:
    """({predictor: {run key: distribution}}, errors): every run of every instance simulated under each predictor's
    model of that instance's calibration (identical circuits are simulated once). Each instance runs inside
    try/except: a simulation too large for memory, or any other failure, is recorded and the rest continues.
    ``workers`` > 1 simulates instances in parallel processes."""
    kinds = tuple(kinds or plan.suite.predictors)
    out = {k: {} for k in kinds}
    errors = [dict(instance=i.key, predictor="*", error="no calibration snapshot for this instance")
              for i in plan.instances if i.key not in models]
    insts = [i for i in plan.instances if i.key in models]
    tasks = [(_sub_plan(plan, i), models[i.key], kinds) for i in insts]
    for i, res in _pool_map(_predict_one, tasks, workers, progress, [x.label for x in insts]):
        if isinstance(res, Exception):
            errors.append(dict(instance=insts[i].key, predictor="*", error="worker failed: %s: %s"
                               % (type(res).__name__, str(res)[:200])))
            continue
        _, o, e = res
        for k in kinds:
            out[k].update(o[k])
        errors += e
    return out, errors


INJECTIONS = {"crosstalk": "ZZ rotation ζ (rad) on every spectator pair after every two-qubit gate (gate-attached)",
              "scale": "every gate error and relaxation rate × s (an under-reporting calibration; gate-attached)",
              "coherent": "a coherent fraction χ of every gate's error (over-rotations at the same infidelity)",
              "t2": "every qubit's T2 lowered to this many µs (dephasing in idle windows; time-attached)",
              "hotspot": "a defect on this physical qubit: every gate on it HOTSPOT_FACTOR× its reported error "
                         "(a local miscalibration, e.g. a two-level-system defect; position-attached)"}
HOTSPOT_FACTOR = 5.0


def parse_injection(items) -> dict:
    """["crosstalk=0.05", "scale=1.5", "coherent=0.3", "t2=40", "hotspot=107"] → {name: value} (:data:`INJECTIONS`;
    hotspot: a physical qubit index)."""
    out = {}
    for it in items or ():
        if "=" not in it:
            raise ValueError("injection %r: use name=value (%s)" % (it, ", ".join(INJECTIONS)))
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in INJECTIONS:
            raise ValueError("unknown injection %r (%s)" % (k, "; ".join("%s: %s" % kv for kv in INJECTIONS.items())))
        out[k] = float(v)
        if k == "t2" and not out[k] > 0:
            raise ValueError("t2 must be positive (µs)")
        if k == "hotspot" and (out[k] < 0 or out[k] != int(out[k])):
            raise ValueError("hotspot is a physical qubit index (an integer ≥ 0)")
    return out


def _truth_model(base, truth: str, inj: dict):
    from .device import DeviceModel, scale_model
    dm = predictor_model(_model(base), truth)
    if inj.get("scale", 1.0) != 1.0:
        dm = scale_model(dm, inj["scale"], 1.0, dm.name + "@scale=%g" % inj["scale"])
        if truth != "device":                          # keep the relaxation-free models free of relaxation
            from .device import QubitInfo
            dm = DeviceModel(dm.name, dm.phys, dm.dt, [QubitInfo(INFINITE_T, INFINITE_T, q.p01, q.p10)
                                                       for q in dm.qubits], dm.two_qubit, dict(dm.durations),
                             dict(dm.errors))
    if inj.get("t2"):
        from .device import QubitInfo
        t2 = float(inj["t2"]) * 1e-6
        dm = DeviceModel(dm.name + "@t2=%gus" % inj["t2"], dm.phys, dm.dt,
                         [QubitInfo(q.t1, min(q.t2, t2, 2 * q.t1), q.p01, q.p10) for q in dm.qubits], dm.two_qubit,
                         dict(dm.durations), dict(dm.errors), coherent=float(getattr(dm, "coherent", 0.0) or 0.0))
    if inj.get("coherent", 0.0):
        dm = DeviceModel(dm.name + "@chi=%g" % inj["coherent"], dm.phys, dm.dt, list(dm.qubits), dm.two_qubit,
                         dict(dm.durations), dict(dm.errors), coherent=float(inj["coherent"]))
    if "hotspot" in inj and int(inj["hotspot"]) in tuple(int(p) for p in dm.phys):
        c = tuple(int(p) for p in dm.phys).index(int(inj["hotspot"]))
        err = {k: (min(float(v) * HOTSPOT_FACTOR, 0.75) if (v is not None and c in tuple(k[1])) else v)
               for k, v in dm.errors.items()}
        dm = DeviceModel(dm.name + "@hot=%d" % int(inj["hotspot"]), dm.phys, dm.dt, list(dm.qubits), dm.two_qubit,
                         dict(dm.durations), err, coherent=float(getattr(dm, "coherent", 0.0) or 0.0))
    return dm


def _emulate_one(args) -> tuple:
    """Worker: (instance key, {run key: probabilities}, error or None) under the truth model."""
    sub, model, truth, inj = args
    inst = sub.instances[0]
    try:
        dm = _truth_model(model, truth, inj)
        cache, out = {}, {}
        for r in sub.runs:
            if r.digest not in cache:
                spec = sub.spec_of(r)
                c = crosstalk_corrections(spec, inj["crosstalk"]) if (inj.get("crosstalk") and r.role not in
                                                                       ("cal0", "cal1")) else None
                eng = "exact" if inj.get("coherent") else engine_for(inst.n, sub.suite)
                cache[r.digest] = measured_probs(spec, inst.mm, dm, engine=eng, shots=sub.suite.mc_shots,
                                                 seed=inst.seed, corrections=c)
            out[r.key] = cache[r.digest]
        return inst.key, out, None
    except Exception as e:                             # noqa: BLE001 - recorded per instance
        return inst.key, {}, "%s: %s" % (type(e).__name__, str(e)[:300])


def emulate(plan: Plan, models: dict, truth: str = "device", inject=None, seed: int = 0, progress=None,
            workers: int = 1) -> tuple:
    """(counts, errors): an emulated device whose noise is a predictor's model of the snapshot (``truth``) plus known
    unmodeled noise: ``scale`` (every gate error and relaxation rate × s: a calibration that under-reports),
    ``coherent`` (a coherent fraction χ of every gate's error: over-rotations at the same infidelity), ``crosstalk``
    (ZZ rotation ζ on spectator pairs after every two-qubit gate, crosstalk_corrections). Exact probabilities,
    multinomial counts with the planned shots (drawn in plan order from one seeded generator, so the counts do not
    depend on ``workers``). For tests and dry runs; never a substitute for hardware data."""
    inj = dict(inject or {})
    insts = [i for i in plan.instances if i.key in models]
    errors = [dict(instance=i.key, error="no calibration snapshot") for i in plan.instances if i.key not in models]
    tasks = [(_sub_plan(plan, i), models[i.key], truth, inj) for i in insts]
    probs = {}
    for i, res in _pool_map(_emulate_one, tasks, workers, progress, [x.label for x in insts]):
        if isinstance(res, Exception):
            errors.append(dict(instance=insts[i].key, error="worker failed: %s: %s" % (type(res).__name__, res)))
            continue
        k, out, err = res
        if err:
            errors.append(dict(instance=k, error=err))
        probs.update(out)
    rng = np.random.default_rng(seed)
    counts = {}
    for r in plan.runs:
        if r.key in probs:
            p = np.clip(probs[r.key], 0, None)
            x = rng.multinomial(int(r.shots), p / p.sum())
            counts[r.key] = {str(int(j)): int(x[j]) for j in np.nonzero(x)[0]}
    return counts, errors


# ====================================================================== estimation and comparison
def freqs_from_counts(c: dict, n: int) -> tuple:
    """(frequency vector, shots) from {outcome index (int, or its decimal string): count} — the format of
    ``counts_from_pub`` and of counts.json (index bit i = classical bit i)."""
    f = np.zeros(2 ** n)
    for k, v in c.items():
        f[int(k)] += v
    s = f.sum()
    return (f / s if s > 0 else f), int(s)


def estimate(method: Method, n: int, F: dict, lams: dict | None = None) -> tuple:
    """(distributions, flags) of a method from the frequency rows of its runs (``F``: {role: (R, 2ⁿ) array}).
    ``lams``: {role: achieved λ} of the folded runs."""
    flags = []
    if method.kind == "raw":
        return np.atleast_2d(F["target"]), flags
    p01, p10 = readout_flips(F["cal0"], F["cal1"], n)

    def mit(X):
        q, bad = mitigate_tensored(X, p01, p10, n)
        if bad:
            flags.append("readout-unmitigated:%s" % ",".join(map(str, bad)))
        return q
    if method.kind == "ro":
        return nearest_probability(mit(F["target"])), flags
    roles = sorted((r for r in F if r.startswith("fold:")), key=lambda r: lams[r])
    eta = extrapolation_weights(method.rule, [lams[r] for r in roles])
    Q = sum(e * mit(F[r]) for e, r in zip(eta, roles))
    return nearest_probability(Q), flags


def _sample(p: np.ndarray, shots: int, R: int, rng) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0, None)
    return rng.multinomial(int(shots), p / p.sum(), size=int(R)) / float(shots)


def compare(plan: Plan, counts: dict, predicted: dict, *, replicates: int | None = None, bootstrap: int | None = None,
            seed: int = 0, source: str = "observed", progress=None) -> list:
    """One row per (instance, method, predictor): observed Fc and normalized fidelity (bootstrap SE), predicted Fc
    at the same shots (replicate mean and SD, and the infinite-shot value), Δ = predicted − observed, z = Δ/SD.
    Rows with missing counts or predictions carry ``status``."""
    R = int(replicates or plan.suite.replicates)
    Rb = int(bootstrap or plan.suite.bootstrap)
    rows = []
    for ii, inst in enumerate(plan.instances):
        n, p_id = inst.n, inst.ideal
        for m in plan.suite.methods:
            runs = plan.runs_of(inst.key, m.key)
            if not runs:                         # the method was skipped on this circuit (see plan.errors)
                continue
            lams = {r.role: r.lam for r in runs if r.lam is not None}
            base = dict(instance=inst.key, group=inst.bench, variant=inst.variant, family=inst.family,
                        params=json.dumps(inst.params, sort_keys=True), label=inst.label, width=n,
                        two_qubit=inst.two_qubit,
                        depth=inst.depth, norm_depth=inst.norm_depth, logical_two_qubit=inst.logical_two_qubit,
                        method=m.key, kind=m.kind, folding=m.folding if m.kind == "zne" else "",
                        budget=int(sum(r.shots for r in runs)), source=source,
                        two_qubit_executed=int(max((r.lam or 1.0) for r in runs) * inst.two_qubit),
                        phys=",".join(map(str, inst.phys)),
                        **{"f_" + k: v for k, v in (inst.features or {}).items()})
            mkey = int.from_bytes(hashlib.sha256(m.key.encode()).digest()[:4], "little")
            rng = np.random.default_rng([int(seed), inst.seed, mkey])        # one stream per (instance, method)
            if not runs:                                 # the method was not planned on this circuit (plan.errors)
                continue
            if not all(r.key in counts for r in runs):
                for k in plan.suite.predictors:
                    rows.append(dict(base, predictor=k, status="no counts"))
                continue
            Fo = {r.role: freqs_from_counts(counts[r.key], n)[0] for r in runs}
            shots = {r.role: int(r.shots) for r in runs}
            q_obs, flags = estimate(m, n, Fo, lams)
            fc_obs = float(hellinger_fidelity(p_id, q_obs)[0])
            nf_obs = float(normalized_fidelity(p_id, q_obs)[0])
            Fb = {role: _sample(Fo[role], shots[role], Rb, rng) for role in Fo}
            qb, _ = estimate(m, n, Fb, lams)
            fc_b, nf_b = hellinger_fidelity(p_id, qb), normalized_fidelity(p_id, qb)
            obs = dict(fc_obs=fc_obs, fc_obs_se=float(np.std(fc_b, ddof=1)), nf_obs=nf_obs,
                       nf_obs_se=float(np.nanstd(nf_b, ddof=1)) if np.isfinite(nf_b).any() else math.nan,
                       flags=";".join(sorted(set(flags))))
            for k, pk in predicted.items():
                if not all(r.key in pk for r in runs):
                    rows.append(dict(base, **obs, predictor=k, status="no prediction"))
                    continue
                Pk = {r.role: pk[r.key] for r in runs}
                q_inf, _ = estimate(m, n, Pk, lams)
                Fr = {role: _sample(Pk[role], shots[role], R, rng) for role in Pk}
                qr, _ = estimate(m, n, Fr, lams)
                fc_r, nf_r = hellinger_fidelity(p_id, qr), normalized_fidelity(p_id, qr)
                mu, sd = float(np.mean(fc_r)), float(np.std(fc_r, ddof=1))
                nmu = float(np.nanmean(nf_r)) if np.isfinite(nf_r).any() else math.nan
                nsd = float(np.nanstd(nf_r, ddof=1)) if np.isfinite(nf_r).any() else math.nan
                rows.append(dict(base, **obs, predictor=k, status="ok", fc_pred=mu, fc_pred_sd=sd,
                                 fc_pred_inf=float(hellinger_fidelity(p_id, q_inf)[0]), nf_pred=nmu, nf_pred_sd=nsd,
                                 delta=mu - fc_obs, z=zscore(mu - fc_obs, sd, obs["fc_obs_se"]),
                                 delta_nf=nmu - nf_obs if math.isfinite(nmu) and math.isfinite(nf_obs) else math.nan,
                                 z_nf=zscore(nmu - nf_obs, nsd, obs["nf_obs_se"]),
                                 **excess(mu, fc_obs, nmu, nf_obs, inst.two_qubit)))
        if progress:
            progress((ii + 1) / len(plan.instances), inst.label)
    rows += group_rows(plan, counts, predicted, R=R, Rb=Rb, seed=seed, source=source, per_variant=rows)
    return rows


EXCESS_FLOOR = 0.02            # fidelities below this are noise-dominated: their logarithm is not used


def zscore(delta: float, sd_pred: float, se_obs: float) -> float:
    """Δ over the pooled shot-noise SD √((sd_pred² + se_obs²)/2): the model's replicate SD and the observation's
    bootstrap SE estimate the same sampling spread when the model is right (z ~ N(0, 1)); pooling keeps z finite for
    estimators the model predicts to be almost deterministic (a plurality vote on a clean peak)."""
    v = [x * x for x in (sd_pred, se_obs) if x is not None and math.isfinite(x)]
    if not v or not math.isfinite(delta):
        return math.nan
    s_ = math.sqrt(sum(v) / len(v))
    return delta / s_ if s_ > 0 else math.nan


def heterogeneity(deltas, sds) -> dict:
    """Cochran's Q of the variants' Δ (weights 1/sd²): are the variants' divergences one and the same (Q ≈ V − 1,
    p large) or does the unmodeled noise depend on where or how the circuit runs (p small; I² = the share of the
    spread that is not shot noise)? Variants that differ only in what the calibration knows have equal Δ."""
    from scipy.stats import chi2
    d, s_ = np.asarray(deltas, float), np.asarray(sds, float)
    ok = np.isfinite(d) & np.isfinite(s_) & (s_ > 0)
    if ok.sum() < 2:
        return dict(het_Q=math.nan, het_p=math.nan, het_I2=math.nan)
    w = 1 / s_[ok] ** 2
    m = float(np.sum(w * d[ok]) / np.sum(w))
    Q = float(np.sum(w * (d[ok] - m) ** 2))
    k = int(ok.sum()) - 1
    return dict(het_Q=Q, het_p=float(chi2.sf(Q, k)), het_I2=max(0.0, (Q - k) / Q) if Q > 0 else 0.0)


def excess(fc_pred: float, fc_obs: float, nf_pred: float, nf_obs: float, n2q: int) -> dict:
    """Unmodeled error as a log-infidelity, total and per two-qubit gate:

    * ex_ln_fc = ln Fc_pred − ln Fc_obs, ex_ln_nf = ln F_pred − ln F_obs (normalized fidelity);
    * ex_nf_per_2q = ex_ln_nf / n₂ — the excess error each two-qubit gate carries beyond its calibration. Unlike
      Δ = Fc_pred − Fc_obs (which grows with any circuit's size, and vanishes again once both decay to the uniform
      floor) it compares structures of different size on one scale.

    NaN where a fidelity is below EXCESS_FLOOR (its logarithm would be noise)."""
    def ln(x):
        return math.log(x) if (x is not None and math.isfinite(x) and x >= EXCESS_FLOOR) else math.nan
    a, b = ln(fc_pred) - ln(fc_obs), ln(nf_pred) - ln(nf_obs)
    return dict(ex_ln_fc=a, ex_ln_nf=b, ex_per_2q=a / n2q if n2q else math.nan,
                ex_nf_per_2q=b / n2q if n2q else math.nan)


def group_rows(plan: Plan, counts: dict, predicted: dict, *, R: int, Rb: int, seed: int = 0, source: str = "observed",
               per_variant=()) -> list:
    """Rows of the variant aggregates of every benchmark with ≥ 2 variants (RAW runs; Chen et al.'s two
    aggregations): ``RAW·agg`` — the variants' histograms added (simple aggregation), ``RAW·PV`` — the plurality
    vote (qemx.context_noise.plurality_vote). Predicted values come from replicates of every variant at its shots,
    aggregated the same way. Each row also carries Cochran's heterogeneity of the variants' RAW divergences
    (:func:`heterogeneity`: het_Q, het_p, het_I2): p small when the unmodeled noise depends on where (or how) the
    circuit runs — context the per-gate model cannot see."""
    from .context_noise import default_threshold, plurality_vote, plurality_vote_rows
    raw = next((m for m in plan.suite.methods if m.kind == "raw"), None)
    if raw is None or plan.suite.variants < 2:
        return []
    groups = {}
    for inst in plan.instances:
        if inst.group:
            groups.setdefault(inst.group, []).append(inst)
    t0 = plan.suite.pv_threshold or default_threshold(plan.suite.variants)
    pv = [r for r in per_variant if r.get("method") == raw.key and r.get("status") == "ok"]
    out = []
    for g, insts in groups.items():
        if len(insts) < 2:
            continue
        insts = sorted(insts, key=lambda i: i.variant)
        n, p_id = insts[0].n, insts[0].ideal
        runs = [plan.runs_of(i.key, raw.key)[0] for i in insts]
        base = dict(instance=g, group=g, variant=-1, family=insts[0].family,
                    params=json.dumps(insts[0].params, sort_keys=True), label=insts[0].label.split(" · v")[0], width=n,
                    two_qubit=float(np.mean([i.two_qubit for i in insts])),
                    depth=float(np.mean([i.depth for i in insts])), norm_depth=insts[0].norm_depth,
                    logical_two_qubit=insts[0].logical_two_qubit, folding="", source=source,
                    budget=int(sum(r.shots for r in runs)), variants=len(insts),
                    phys=";".join(",".join(map(str, i.phys)) for i in insts))
        mkey = int.from_bytes(hashlib.sha256(g.encode()).digest()[:4], "little")
        rng = np.random.default_rng([int(seed), mkey, 17])
        have = all(r.key in counts for r in runs)
        if have:
            Fo = [freqs_from_counts(counts[r.key], n)[0] for r in runs]
            w = np.array([r.shots for r in runs], float)
            agg_o = (np.array(Fo) * w[:, None]).sum(0) / w.sum()
            pv_o, t_used = plurality_vote(np.array(Fo), t0)
            Fb = [_sample(f, r.shots, Rb, rng) for f, r in zip(Fo, runs)]
            agg_b = sum(f * wi for f, wi in zip(Fb, w)) / w.sum()
            pv_b = plurality_vote_rows(Fb, t0)
        dv = {}
        for r in pv:
            if r["group"] == g:
                sd_ = math.sqrt(((r.get("fc_pred_sd") or 0) ** 2 + (r.get("fc_obs_se") or 0) ** 2) / 2)
                dv.setdefault(r["predictor"], []).append((r["delta"], sd_))
        for k, pk in predicted.items():
            d_ = dv.get(k, [])
            disp = heterogeneity([x for x, _ in d_], [y for _, y in d_])
            for meth, kind in (("%s·agg" % raw.key, "agg"), ("%s·PV" % raw.key, "pv")):
                row = dict(base, method=meth, kind=kind, predictor=k, **disp)
                if not have:
                    out.append(dict(row, status="no counts"))
                    continue
                if not all(r.key in pk for r in runs):
                    out.append(dict(row, status="no prediction"))
                    continue
                q_o = agg_o if kind == "agg" else pv_o
                qb = agg_b if kind == "agg" else pv_b
                Fr = [_sample(pk[r.key], r.shots, R, rng) for r in runs]
                qr = sum(f * wi for f, wi in zip(Fr, w)) / w.sum() if kind == "agg" else plurality_vote_rows(Fr, t0)
                fc_o, nf_o = float(hellinger_fidelity(p_id, q_o)), float(normalized_fidelity(p_id, q_o))
                fc_r, nf_r = hellinger_fidelity(p_id, qr), normalized_fidelity(p_id, qr)
                fc_b, nf_b = hellinger_fidelity(p_id, qb), normalized_fidelity(p_id, qb)
                mu, sd = float(np.mean(fc_r)), float(np.std(fc_r, ddof=1))
                nmu = float(np.nanmean(nf_r)) if np.isfinite(nf_r).any() else math.nan
                nsd = float(np.nanstd(nf_r, ddof=1)) if np.isfinite(nf_r).any() else math.nan
                se_o, nse_o = float(np.std(fc_b, ddof=1)), float(np.nanstd(nf_b, ddof=1))
                out.append(dict(row, status="ok", fc_obs=fc_o, fc_obs_se=se_o, nf_obs=nf_o,
                                nf_obs_se=nse_o, fc_pred=mu, fc_pred_sd=sd, nf_pred=nmu,
                                nf_pred_sd=nsd, delta=mu - fc_o, z=zscore(mu - fc_o, sd, se_o),
                                delta_nf=nmu - nf_o if math.isfinite(nmu) else math.nan,
                                z_nf=zscore(nmu - nf_o, nsd, nse_o),
                                pv_threshold=t_used if kind == "pv" else math.nan,
                                **excess(mu, fc_o, nmu, nf_o, int(round(base["two_qubit"])))))
    return out


def fold_response(plan: Plan, counts: dict, predicted: dict, *, floor: float = 0.05, bootstrap: int | None = None,
                  replicates: int | None = None, seed: int = 0) -> list:
    """The folding fingerprint of unmodeled noise, from the ZNE runs (no extra shots).

    For every instance and ZNE method, the readout-mitigated normalized fidelity F(λ) of each folded circuit decays
    as ln F ≈ a − κλ. κ is fitted to the observation and to every predictor (same estimator), and A = κ_obs / κ_pred
    is the amplification anomaly: 1 when the device responds to folding as its model does. Global folding repeats
    *everything* (gates, idle windows); local folding repeats only the two-qubit gates. So:

    * A_local ≈ A_global > 1  → the unmodeled noise is attached to the two-qubit gates (crosstalk during the gate,
      under-reported gate error);
    * A_global > A_local ≈ 1  → it is attached to time (idle dephasing, slow drifts) — local folding does not stretch
      idle windows;
    * A_local > A_global      → it grows faster than linearly under repetition of the same gate (coherent errors).

    Finite shots bias κ: the fidelity of a histogram is lower than that of its distribution, the more so the fewer
    shots a fold gets (ZNE gives the high-λ folds the fewest) and the flatter the ideal distribution. So κ_pred is
    the mean κ of the predictor's *replicates* at the observation's shots (the bias is the same on both sides), never
    the κ of its exact probabilities (reported as kappa_pred_exact). The interval of A is the basic bootstrap of
    κ_obs (2κ_obs − the bootstrap's 84/16 % quantiles: the bootstrap's own extra shot noise biases its centre), and
    z_kappa = (κ_obs − κ_pred) / pooled SD, as for Δ.

    Rows: instance, method, folding, predictor, kappa_obs, kappa_pred, kappa_pred_sd, kappa_pred_exact, A, A_lo,
    A_hi, z_kappa, points."""
    Rb = int(bootstrap or plan.suite.bootstrap)
    R = int(replicates or min(int(plan.suite.replicates), 200))
    out = []
    for inst in plan.instances:
        n, p_id = inst.n, inst.ideal
        for m in plan.suite.methods:
            if m.kind != "zne":
                continue
            runs = plan.runs_of(inst.key, m.key)
            if not runs or not all(r.key in counts for r in runs):
                continue
            folds = sorted((r for r in runs if r.role.startswith("fold:")), key=lambda r: r.lam)
            lam = np.array([r.lam for r in folds], float)
            cal = {r.role: r for r in runs if r.role in ("cal0", "cal1")}
            used = folds + list(cal.values())
            mkey = int.from_bytes(hashlib.sha256((inst.key + m.key).encode()).digest()[:4], "little")
            rng = np.random.default_rng([int(seed), mkey, 29])

            def nf_curve(F):
                """F: {role: (R, D)} → (R, len(folds)) normalized fidelities of the mitigated folded circuits."""
                p01, p10 = readout_flips(F["cal0"], F["cal1"], n)
                cols = []
                for r in folds:
                    q, _ = mitigate_tensored(F[r.role], p01, p10, n)
                    cols.append(normalized_fidelity(p_id, nearest_probability(q)))
                return np.stack(cols, axis=1)

            def slope(Y):
                """κ of every row (−d ln F / dλ over the points ≥ floor; NaN with fewer than two)."""
                k = np.full(Y.shape[0], np.nan)
                for i, y in enumerate(Y):
                    ok = np.isfinite(y) & (y >= floor)
                    if ok.sum() >= 2:
                        k[i] = -np.polyfit(lam[ok], np.log(y[ok]), 1)[0]
                return k
            Fo = {r.role: freqs_from_counts(counts[r.key], n)[0][None, :] for r in used}
            curve_o = nf_curve(Fo)[0]
            k_obs = float(slope(curve_o[None, :])[0])
            kb = slope(nf_curve({r.role: _sample(Fo[r.role][0], r.shots, Rb, rng) for r in used}))
            kb = kb[np.isfinite(kb)]
            se_o = float(np.std(kb, ddof=1)) if len(kb) > 10 else math.nan
            q16, q84 = (np.percentile(kb, [16, 84]) if len(kb) > 10 else (math.nan, math.nan))
            for k, pk in predicted.items():
                if not all(r.key in pk for r in runs):
                    continue
                k_exact = float(slope(nf_curve({r.role: np.atleast_2d(pk[r.key]) for r in used}))[0])
                kr = slope(nf_curve({r.role: _sample(pk[r.key], r.shots, R, rng) for r in used}))
                kr = kr[np.isfinite(kr)]
                k_pred = float(np.mean(kr)) if len(kr) > 10 else math.nan
                sd = float(np.std(kr, ddof=1)) if len(kr) > 10 else math.nan
                ok = math.isfinite(k_pred) and k_pred > 1e-9 and math.isfinite(k_obs)
                out.append(dict(instance=inst.key, group=inst.bench, family=inst.family, label=inst.label,
                                width=n, two_qubit=inst.two_qubit, method=m.key, folding=m.folding,
                                subset=m.subset if m.folding == "local" else "", scales=",".join("%g" % x for x in lam),
                                predictor=k, kappa_obs=k_obs, kappa_obs_se=se_o, kappa_pred=k_pred,
                                kappa_pred_sd=sd, kappa_pred_exact=k_exact,
                                A=k_obs / k_pred if ok else math.nan,
                                A_lo=(2 * k_obs - q84) / k_pred if ok else math.nan,
                                A_hi=(2 * k_obs - q16) / k_pred if ok else math.nan,
                                z_kappa=zscore(k_obs - k_pred, sd, se_o) if ok else math.nan,
                                points=int(np.sum(np.isfinite(curve_o) & (curve_o >= floor)))))
    return out


def pooled_anomaly(kappa_obs, kappa_pred, kappa_obs_se=None) -> tuple:
    """(A, SE, χ²/dof) of a group of circuits that share one anomaly, κ_obs,c = A·κ_pred,c + e_c:

    * with the observations' bootstrap SEs, the weighted least-squares A = Σ κ_p κ_o / se² ÷ Σ κ_p² / se² (SE =
      1 / √(Σ κ_p² / se²)) — precise circuits count most, a ratio of small noisy κ never dominates; χ²/dof of the fit
      is ≈ 1 when the circuits do share one A (≫ 1: the anomaly differs between circuits);
    * without SEs, Σ κ_obs / Σ κ_pred (SE and χ²/dof NaN).

    The replicate-mean κ_pred is exact by comparison with the single observation."""
    o, p = np.asarray(kappa_obs, float), np.asarray(kappa_pred, float)
    se = np.asarray(kappa_obs_se if kappa_obs_se is not None else np.full(len(o), np.nan), float)
    ok = np.isfinite(o) & np.isfinite(p)
    if not ok.any() or p[ok].sum() <= 0:
        return math.nan, math.nan, math.nan
    w_ok = ok & np.isfinite(se) & (se > 0)
    if w_ok.sum() >= 2:
        w = p[w_ok] ** 2 / se[w_ok] ** 2
        A = float(np.sum(p[w_ok] * o[w_ok] / se[w_ok] ** 2) / np.sum(w))
        chi = float(np.sum(((o[w_ok] - A * p[w_ok]) / se[w_ok]) ** 2) / (w_ok.sum() - 1))
        return A, float(1 / math.sqrt(np.sum(w))), chi
    return float(o[ok].sum() / p[ok].sum()), math.nan, math.nan


def effective_rate(two_qubit, fc, floor: float = 0.05) -> tuple:
    """(ε_eff, intercept, points): least-squares fit of log Fc = a + N₂·log(1 − ε) over the circuits with Fc ≥ floor
    (Chen et al. Fig. 9's lines). NaN with fewer than 3 points or no spread in N₂."""
    x, y = np.asarray(two_qubit, float), np.asarray(fc, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y >= floor)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return math.nan, math.nan, int(ok.sum())
    s, a = np.polyfit(x[ok], np.log(y[ok]), 1)
    return float(1 - math.exp(s)), float(math.exp(a)), int(ok.sum())
