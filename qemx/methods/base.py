r"""The plug-in contract for QEM methods, recipes, and the run context.

A **method** (estimator) is a class with a key, a family, a TeX formula, a parameter schema,
and ``run(ctx, **params) -> MethodResult``. ``@register`` adds it to :data:`REGISTRY`; the UI
builds its controls from the schema, so a new method is one new file.

A **recipe** combines circuit transforms (Pauli twirling, dynamical decoupling), one estimator,
and a readout mitigation. Any list of recipes can be compared.

Every estimator reports an exact mean (from exact probabilities) and a per-shot variance
``var1`` so that, with N shots spent on the target circuits, Var = var1 / N. Training shots (``fixed_shots``, whose
shot noise adds the constant ``var_fixed``) and the readout calibration (``cal_circuits``; its finite-shot error
through ``cal_grad`` and ``cal_theta``) are accounted for in ``qemx/budget.py``, which compares every method at the
same total shot budget: Var(B) = var1/N_t + w/s + var_fixed.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..circuit import CircuitSpec
from ..compile import Compiled
from ..engine import RunResult, simulate
from ..noise import NoiseSpec, readout_matrix
from ..observables import Observable, shot_var

REGISTRY: dict = {}
FAMILY_LABEL = {"raw": "Unmitigated", "zne": "ZNE", "pec": "PEC", "learning": "Learning (CDR / FR / NIL)",
                "vd": "Virtual distillation", "symmetry": "Symmetry verification", "readout": "Readout mitigation"}


@dataclass(frozen=True)
class Param:
    name: str
    label: str                     # may contain TeX between $...$
    kind: str                      # float | int | choice | bool | floats | ints | text
    default: object
    options: tuple = ()            # for choice
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    help: str = ""
    when: tuple = ()               # ((other param, (values…)), …): shown only if every listed param has one of the
                                   # values (controls of composite methods); a hidden param keeps its default


@dataclass
class MethodResult:
    mean: float
    var1: float                           # per-shot variance of the estimator
    circuits: int = 1                     # distinct target circuits (before randomization)
    fixed_shots: int = 0                  # training shots (CDR/FR/NIL), part of the budget
    train_circuits: int = 0               # training circuits (CDR/FR/NIL)
    cal_circuits: int = 0                 # readout-calibration circuits
    cal_grad: object = None               # ∂mean/∂θ for the estimated readout flip probabilities θ
    cal_theta: object = None              # θ (the calibration's estimated flip probabilities)
    cal_fun: Callable | None = None       # fun(mitigate) -> the estimate recomputed with another readout mitigation
    shots_per_circuit: int | None = None  # randomized protocol (PEC): shots per sampled circuit
    var_fixed: float = 0.0                # variance from the fixed training shots (CDR/FR/NIL): does not shrink with B
    gamma: float = 1.0                    # sampling-overhead factor (std amplification)
    extra_qubits: int = 0
    accept: float = 1.0                   # post-selection acceptance
    implemented: list = field(default_factory=list)   # [(title, CircuitSpec)]
    details: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    sampler: Callable | None = None       # (N, B, rng) -> B estimates at N target shots
    train_resampler: Callable | None = None   # (s_tr, R, rng) -> R estimates with resampled training data and
                                              # the exact target (learning methods; training-shot noise)
    cal_w: float | None = None            # calibration coefficient measured by Monte Carlo (learning methods): it
                                          # replaces the first-order w = Σ g²θ(1−θ) where the refit is nonlinear
    joint_calibration: bool = False       # sampler(N, B, rng, theta=…) and train_resampler(…, theta=…) accept
                                          # per-replicate calibration draws θ̂ and apply them to every circuit
    # ---- the statistical contract (qemx/stats.py): what ``mean`` / ``var1`` / ``var_fixed`` refer to
    estimate_kind: str = "exact"          # exact | protocol | measured (see qemx.stats.KINDS)
    reference_mean: float | None = None   # learning: the exact-probability reference fit (an oracle quantity)
    reference_var1: float | None = None   # its per-shot target variance
    moments: dict | None = None           # protocol Monte-Carlo record (qemx.stats.Moments.as_dict)
    flags: tuple = ()                     # approximation / oracle / validity flags (strings), shown with the result


class QEMMethod:
    key = "base"
    label = "base"
    family = "raw"
    tex = ""
    why = ""
    params: tuple = ()

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def run(self, ctx: "Context", **params) -> MethodResult:     # pragma: no cover
        raise NotImplementedError

    def min_cost(self, ctx: "Context", **params) -> dict | None:
        """Budget preflight, *before* any simulation or fit: {"training": shots the protocol must spend on training
        circuits, "target_circuits": distinct target circuits that each need ≥ 1 shot, "note": text}, or None when
        the cost is not known in advance (then the method runs and its result's cost model decides). Must be cheap:
        no simulation, no fitting."""
        return None


def register(cls):
    REGISTRY[cls.key] = cls()
    return cls


# ------------------------------------------------------------------ readout assignment matrices
READOUT_SINGULAR_TOL = 1e-12     # |1 − e − h| at or below this: the assignment matrix is singular (not invertible)
READOUT_COND_WARN = 100.0        # condition number above which first-order calibration propagation is unreliable


def assignment_inverse_gamma(e, h) -> float:
    """‖A⁻¹‖₁ for the one-bit assignment matrix A = [[1−e, h], [e, 1−h]] (e = P(1|0), h = P(0|1); columns are the
    prepared states): (1 + |e − h|) / |1 − e − h|, ∞ when singular. Valid for every e, h in [0, 1], including
    1 − e − h < 0 (an invertible matrix that mostly flips). ValueError for rates outside [0, 1] or non-finite."""
    e, h = float(e), float(h)
    if not (math.isfinite(e) and math.isfinite(h) and 0.0 <= e <= 1.0 and 0.0 <= h <= 1.0):
        raise ValueError("readout flip probabilities must be finite and in [0, 1], got e = %r, h = %r" % (e, h))
    det = abs(1.0 - e - h)
    if det <= READOUT_SINGULAR_TOL:
        return math.inf
    return (1.0 + abs(e - h)) / det


# ------------------------------------------------------------------ legacy names (explicit migration)
# (estimator, parameter, legacy value) -> current value. Recipe.make applies these, so every entry point (presets,
# saved states, manifests, exported scripts) resolves a legacy configuration to the same recipe; loaders call
# migrate_params themselves to *report* what was migrated.
PARAM_ALIASES = {
    # 1.5.0: the learning estimator's feature-regression variant was selected by features="nil", which read as
    # neighbor-informed learning (a different method: estimator "nil"). Its id is now "fr".
    ("learning", "features", "nil"): "fr",
}


def migrate_params(estimator: str, params: dict) -> tuple:
    """(params with legacy values replaced, [migration notes]). Never changes a current value."""
    out, notes = dict(params or {}), []
    for (est, name, old), new in PARAM_ALIASES.items():
        if est == estimator and out.get(name) == old:
            out[name] = new
            notes.append("%s: %s=%r is the legacy name of %s=%r (migrated)" % (estimator, name, old, name, new))
    return out, notes


# ------------------------------------------------------------------ recipes
TWIRL = {"none": "no twirling", "2q": "Pauli twirling (2q gates)", "all": "randomized compiling (all gates)"}
DD = {"none": "no DD", "X2": "DD: X2 (CPMG)", "XY4": "DD: XY4"}
READOUT = {"none": "no readout mitigation", "tensored": "tensored inverse", "m3": "M3 (subspace)",
           "trex": "TREX (twirled readout)"}


@dataclass(frozen=True)
class Recipe:
    estimator: str
    params: tuple = ()             # sorted (name, value) pairs
    twirl: str = "none"
    dd: str = "none"
    readout: str = "none"

    @staticmethod
    def make(estimator: str, twirl="none", dd="none", readout="none", **params) -> "Recipe":
        m = REGISTRY[estimator]
        full = m.defaults()
        full.update(migrate_params(estimator, params)[0])
        return Recipe(estimator, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in full.items())),
                      twirl, dd, readout)

    @property
    def p(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params}

    def title(self) -> str:
        m = REGISTRY[self.estimator]
        t = m.short(self.p) if hasattr(m, "short") else m.label
        extras = []
        if self.twirl != "none":
            extras.append("twirl" if self.twirl == "2q" else "RC")
        if self.dd != "none":
            extras.append("DD-" + self.dd)
        if self.readout != "none":
            extras.append({"tensored": "RO-inv", "m3": "M3", "trex": "TREX"}[self.readout])
        if self.estimator == "raw" and extras:       # "Unmitigated + TREX" reads as a contradiction
            return " + ".join(extras)
        return t + (" + " + " + ".join(extras) if extras else "")

    def key(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)

    def to_json(self) -> dict:
        return dict(estimator=self.estimator, params=self.p, twirl=self.twirl, dd=self.dd, readout=self.readout)

    @staticmethod
    def from_json(d: dict) -> "Recipe":
        return Recipe.make(d["estimator"], d.get("twirl", "none"), d.get("dd", "none"), d.get("readout", "none"),
                           **d.get("params", {}))


# ------------------------------------------------------------------ context
class Runner:
    """Memoized simulate() shared by every recipe of one comparison."""

    def __init__(self, backend: str = "numpy", layout=None):
        self.backend = backend
        self.layout = tuple(layout) if layout is not None else None     # physical qubits on a device backend
        self.cache: dict = {}
        self.calls = 0

    @property
    def device_model(self):
        from .. import device
        if device.is_device(self.backend) and self.layout is not None:
            return device.get_model(self.backend, self.layout)
        return None

    def run(self, spec: CircuitSpec, noise: NoiseSpec, measure_map: dict, *, ideal=False, twirl="none", dd="none",
            corrections=None, corr_key=None, snapshots=False, backend=None, need_state=False,
            role: str = "target") -> RunResult:
        """Memoized. For large registers the cached result drops its final density matrix (only the
        distributions are needed by most estimators); ``need_state`` keeps it (virtual distillation). ``role``
        (target / training / calibration) says which part of the shot budget the circuit draws on; simulation
        ignores it, the hardware planner allocates shots by it."""
        from ..resources import keep_states
        be = backend or self.backend
        key = (spec.key(), noise.key(), tuple(sorted(measure_map.items())), ideal, twirl, dd, corr_key, snapshots, be)
        hit = self.cache.get(key)
        if hit is None or (need_state and hit.rho is None):
            self.calls += 1
            hit = simulate(spec, noise, be, measure_map=measure_map, ideal=ideal, twirl=twirl, dd=dd,
                           corrections=corrections, snapshots=snapshots, layout=self.layout)
            if need_state:
                return hit
            if not keep_states(spec.n):
                hit.rho = None
            self.cache[key] = hit
        return hit


class Context:
    def __init__(self, compiled: Compiled, noise: NoiseSpec, observable: Observable, recipe: Recipe,
                 runner: Runner, shots: int = 4096, symmetry=None, seed: int = 0):
        self.compiled = compiled
        self.noise = noise
        self.obs = observable
        self.recipe = recipe
        self.runner = runner
        self.shots = shots
        self.symmetry = symmetry
        self.seed = seed
        self.n = compiled.implemented.n
        self._cal = None
        self._oeff_cache = {}               # o_eff per observable vector, valid for one calibration (see o_eff)

    # --------------------------------------------------------- execution
    @property
    def spec(self) -> CircuitSpec:
        return self.compiled.implemented

    @property
    def mm(self) -> dict:
        return self.compiled.measure_map

    def run(self, spec: CircuitSpec | None = None, **kw) -> RunResult:
        from ..resources import check
        check()
        kw.setdefault("twirl", self.recipe.twirl)
        kw.setdefault("dd", self.recipe.dd)
        return self.runner.run(spec or self.spec, kw.pop("noise", self.noise), self.mm, **kw)

    def ideal_value(self, spec: CircuitSpec | None = None, o=None) -> float:
        r = self.runner.run(spec or self.spec, self.noise, self.mm, ideal=True, backend="numpy")
        return float((self.obs.o if o is None else o) @ r.probs)

    # --------------------------------------------------------- readout
    def calibration(self) -> dict:
        """Per-bit assignment matrices from two calibration circuits (all 0, all 1), through the
        same noise as the target (the X gates of the |1...1> circuit carry their gate error).

        The calibration circuits are measured with the *target's* measurement map, so classical bit i is read from
        the same qubit mm[i] in calibration and target: A[i] is the confusion matrix of the qubit behind bit i.
        (Before v1.3 they were measured with the identity map, which applied qubit i's matrix to bit i — wrong
        whenever routing or relabeling permutes the bits, on devices with per-qubit readout errors.)"""
        if self._cal is None:
            n = self.n
            zero = CircuitSpec(n, (), "cal-0")
            from ..circuit import Op
            one = CircuitSpec(n, tuple(Op("x", (q,), (), "gate", "prep") for q in range(n)), "cal-1")
            p0 = self.runner.run(zero, self.noise, self.mm, role="calibration").probs_meas
            p1 = self.runner.run(one, self.noise, self.mm, role="calibration").probs_meas
            bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
            A = []
            for i in range(n):
                q1_given0 = float(p0 @ bits[:, i])
                q0_given1 = float(p1 @ (1 - bits[:, i]))
                A.append(np.array([[1 - q1_given0, q0_given1], [q1_given0, 1 - q0_given1]]))
            dm = self.runner.device_model
            if dm is not None:                       # TREX on a device: each bit's own symmetrized flip probability
                pbar = [(dm.qubits[self.mm.get(i, i)].p01 + dm.qubits[self.mm.get(i, i)].p10) / 2 for i in range(n)]
            else:
                pbar = [(self.noise.ro01 + self.noise.ro10) / 2] * n      # TREX: symmetrized flip probability
            if self.recipe.readout in ("tensored", "m3", "trex"):
                # a singular assignment matrix (P(1|0) + P(0|1) = 1, TREX: p̄ = ½) makes the outcome independent of
                # the prepared state: no inverse exists. Say so instead of failing later in a linear solve.
                if self.recipe.readout == "trex":
                    bad = [i for i, pb in enumerate(pbar) if abs(1 - 2 * pb) <= READOUT_SINGULAR_TOL]
                else:
                    bad = [i for i, a in enumerate(A) if abs(1 - a[1, 0] - a[0, 1]) <= READOUT_SINGULAR_TOL]
                if bad:
                    raise ValueError(
                        "readout mitigation is impossible: the assignment matrix of bit%s %s is singular "
                        "(P(1|0) + P(0|1) = 1), so the measured outcome carries no information about the prepared "
                        "state" % ("" if len(bad) == 1 else "s", ", ".join(map(str, bad))))
            self._cal = dict(A=A, pbar=pbar, circuits=1 if self.recipe.readout == "trex" else 2)
        return self._cal

    # --------------------------------------------------------- calibration error (delta method)
    def cal_circuits(self) -> int:
        """Readout-calibration circuits: |0…0⟩ and |1…1⟩ (tensored, M3), or one twirled |0…0⟩ (TREX)."""
        return {"tensored": 2, "m3": 2, "trex": 1}.get(self.recipe.readout, 0)

    def cal_theta(self) -> np.ndarray:
        """The flip probabilities a finite calibration estimates: P(1|0), P(0|1) per bit (tensored, M3) or the
        symmetrized p̄ per bit (TREX)."""
        mode = self.recipe.readout
        if mode in ("tensored", "m3"):
            return np.array([v for A in self.calibration()["A"] for v in (A[1, 0], A[0, 1])], float)
        if mode == "trex":
            return np.array(self.calibration()["pbar"], float)
        return np.zeros(0)

    def sample_cal_theta(self, s: int, R: int, rng) -> np.ndarray | None:
        """R draws of θ̂ (the order of cal_theta()) from the calibration circuits of calibration() run with s shots
        each: multinomial counts over their full output distributions, flip frequencies per bit. None without
        readout mitigation."""
        mode = self.recipe.readout
        if mode == "none":
            return None
        n = self.n
        from ..circuit import Op
        zero = CircuitSpec(n, (), "cal-0")
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        r0 = self.runner.run(zero, self.noise, self.mm, role="calibration")      # the circuits calibration() uses
        if mode in ("tensored", "m3"):
            one = CircuitSpec(n, tuple(Op("x", (q,), (), "gate", "prep") for q in range(n)), "cal-1")
            r1 = self.runner.run(one, self.noise, self.mm, role="calibration")
            p0 = np.clip(r0.probs_meas, 0, None)
            p1 = np.clip(r1.probs_meas, 0, None)
            c0 = rng.multinomial(int(s), p0 / p0.sum(), size=int(R))
            c1 = rng.multinomial(int(s), p1 / p1.sum(), size=int(R))
            th = np.empty((int(R), 2 * n))
            th[:, 0::2] = c0 @ bits / float(s)               # P(read 1 | prepared 0) per bit
            th[:, 1::2] = c1 @ (1 - bits) / float(s)         # P(read 0 | prepared 1) per bit
            return th
        if mode == "trex":                                   # one twirled |0…0⟩ circuit: symmetrized flips per bit
            p = np.clip(self._tensor_apply(self._sym(), r0.probs), 0, None)
            c = rng.multinomial(int(s), p / p.sum(), size=int(R))
            return c @ bits / float(s)
        raise ValueError(mode)

    def theta_rows(self, theta, R: int):
        """``theta`` as an (R, d) array of calibration draws (one per replicate), or None; ValueError if it does not
        have exactly R rows of this recipe's calibration length (a silent truncation would mispair the draws)."""
        if theta is None:
            return None
        th = np.asarray(theta, float)
        d = len(self.cal_theta()) if self.recipe.readout != "none" else None
        if th.ndim != 2 or th.shape[0] != int(R) or (d is not None and th.shape[1] != d):
            raise ValueError("calibration draws: expected %d rows%s, got shape %s"
                             % (int(R), "" if d is None else " of length %d" % d, th.shape))
        return th

    def theta_invertible(self, theta) -> bool:
        """Whether a calibration θ̂ is mathematically invertible: every per-bit assignment matrix has a nonzero
        determinant 1 − e − h (TREX: 1 − 2p̄). A *negative* determinant (more flips than not) is invertible and is
        accepted — the inverse then also undoes the systematic flip. Near-singular draws are invertible but badly
        conditioned; see ``theta_condition``. (A finite calibration can be this bad; such draws are not clamped.)"""
        t = np.asarray(theta, float)
        if not np.all(np.isfinite(t)):
            return False
        if self.recipe.readout in ("tensored", "m3"):
            return bool(np.all(np.abs(1 - t[0::2] - t[1::2]) > READOUT_SINGULAR_TOL))
        if self.recipe.readout == "trex":
            return bool(np.all(np.abs(1 - 2 * t) > READOUT_SINGULAR_TOL))
        return True

    def theta_condition(self, theta=None) -> float:
        """Largest per-bit 1-norm condition number of the assignment matrices of calibration θ (default: this
        context's): 1 for perfect readout, ∞ when singular. Above READOUT_COND_WARN first-order (delta-method)
        propagation of the calibration error is unreliable."""
        t = np.asarray(self.cal_theta() if theta is None else theta, float)
        mode = self.recipe.readout
        if mode in ("tensored", "m3"):
            pairs = zip(t[0::2], t[1::2])
        elif mode == "trex":
            pairs = ((x, x) for x in t)
        else:
            return 1.0
        worst = 1.0
        for e, h in pairs:
            g = assignment_inverse_gamma(e, h)
            worst = max(worst, g * max(1 - e + h, 1 - h + e))       # ‖A⁻¹‖₁ · ‖A‖₁
        return float(worst)

    def o_eff_theta(self, theta, o=None) -> np.ndarray:
        """Per-shot values Mᵀ(θ)o with the mitigation built from a calibration θ (the linear map _mitigator(θ)
        applies; M3 uses the tensored inverse, as in _mitigator), in O(n·2ⁿ). ``o`` may be a (2ⁿ, k) matrix of k
        observables (all mapped at once)."""
        o = np.asarray(self.obs.o if o is None else o, float)
        mode = self.recipe.readout
        if mode == "none":
            return o.copy()
        t = np.asarray(theta, float)
        if mode in ("tensored", "m3"):
            mats = [np.linalg.inv(np.array([[1 - t[2 * i], t[2 * i + 1]], [t[2 * i], 1 - t[2 * i + 1]]])).T
                    for i in range(self.n)]
        else:
            mats = [np.linalg.inv(np.array([[1 - x, x], [x, 1 - x]])).T for x in t]
        return self._tensor_apply(mats, o)

    def _mitigator(self, theta):
        """f ↦ mitigated distribution, with the calibration replaced by θ. For M3 this is the tensored inverse (M3's
        subspace solve depends on the sampled bitstrings and is not one linear map), so calibration resampling
        through it (qemx.phase.validate) approximates M3 by the tensored mitigation."""
        mode = self.recipe.readout
        n = self.n
        if mode in ("tensored", "m3"):
            mats = [np.linalg.inv(np.array([[1 - theta[2 * i], theta[2 * i + 1]],
                                            [theta[2 * i], 1 - theta[2 * i + 1]]])) for i in range(n)]
        else:
            mats = [np.linalg.inv(np.array([[1 - t, t], [t, 1 - t]])) for t in theta]
        return lambda f: self._tensor_apply(mats, f)

    def cal_gradient(self, fun) -> tuple:
        """(∂mean/∂θ, θ) by central differences; ``fun(mitigate)`` recomputes the estimate from cached data with a
        given mitigation map. Empty when the recipe has no readout mitigation."""
        th = self.cal_theta()
        if th.size == 0:
            return None, None
        g = np.zeros(th.size)
        for k in range(th.size):
            h = 1e-6 if 1e-6 < th[k] < 0.5 - 1e-6 else 1e-7
            up, dn = th.copy(), th.copy()
            up[k] += h
            dn[k] = max(dn[k] - h, 0.0)
            g[k] = (fun(self._mitigator(up)) - fun(self._mitigator(dn))) / (up[k] - dn[k])
        g[~np.isfinite(g)] = 0.0
        return g, th

    def readout_gamma(self) -> float:
        """Sampling overhead of the readout inversion: the product over bits of ‖A_i⁻¹‖₁ (the inverse's 1-norm, which
        bounds how much it amplifies a ±1 per-shot value). ∞ for a singular assignment matrix."""
        mode = self.recipe.readout
        if mode in ("tensored", "m3"):
            g = 1.0
            for A in self.calibration()["A"]:
                g *= assignment_inverse_gamma(A[1, 0], A[0, 1])
            return g
        if mode == "trex":
            g = 1.0
            for pb in self.calibration()["pbar"]:
                g *= assignment_inverse_gamma(pb, pb)
            return g
        return 1.0

    def _tensor_apply(self, mats, p):
        """(⊗_i mats[i]) p with mats[i] acting on classical bit i; p of shape (2ⁿ,) or (2ⁿ, k) (k vectors at once)."""
        n = self.n
        p = np.asarray(p)
        t = p.reshape([2] * n + list(p.shape[1:]))
        for i, M in enumerate(mats):
            ax = n - 1 - i
            t = np.moveaxis(np.tensordot(M, t, axes=([1], [ax])), 0, ax)
        return t.reshape(p.shape)

    def _sym(self):
        return [np.array([[1 - pb, pb], [pb, 1 - pb]]) for pb in self.calibration()["pbar"]]

    def measured(self, rr: RunResult) -> np.ndarray:
        """The distribution the device reports under this recipe. TREX twirls the readout (random X
        before measurement, classical bit flip after), which makes the confusion symmetric."""
        if self.recipe.readout == "trex":
            return self._tensor_apply(self._sym(), rr.probs)
        return rr.probs_meas

    def mitigate(self, f: np.ndarray) -> np.ndarray:
        """Readout-mitigated quasi-distribution from a measured (exact or sampled) distribution f."""
        mode = self.recipe.readout
        if mode == "none":
            return f
        cal = self.calibration()
        if mode == "tensored":
            return self._tensor_apply([np.linalg.inv(A) for A in cal["A"]], f)
        if mode == "m3":
            return m3_solve(f, cal["A"], self.n)
        if mode == "trex":
            # dividing every <Z_S> by its calibration value prod(1 - 2 p̄) = applying the inverse symmetric matrix
            return self._tensor_apply([np.linalg.inv(A) for A in self._sym()], f)
        raise ValueError(mode)

    def expval(self, rr: RunResult, o=None) -> float:
        o = self.obs.o if o is None else o
        return float(o @ self.mitigate(self.measured(rr)))

    def _lin_mats(self):
        """Per-bit matrices of the linear mitigation map, M = ⊗_i mats[i] with mats[i] acting on classical bit i
        (the order _tensor_apply uses); None without readout mitigation. M3 maps to the tensored inverse, which it
        equals when every bitstring is observed (exact data)."""
        mode = self.recipe.readout
        if mode == "none":
            return None
        if mode in ("tensored", "m3"):
            return [np.linalg.inv(A) for A in self.calibration()["A"]]
        if mode == "trex":
            return [np.linalg.inv(A) for A in self._sym()]
        raise ValueError(mode)

    def _per_bit_ok(self) -> bool:
        """Whether mitigate() is the per-bit tensor map of _lin_mats(). Always for this class; a subclass that
        overrides mitigate() is checked once per calibration on a random full-support distribution (otherwise the
        dense matrix is built from mitigate() itself, as before)."""
        if type(self).mitigate is Context.mitigate or self.recipe.readout == "m3":
            return True
        cal = self.calibration()
        memo = getattr(self, "_per_bit_memo", None)
        if memo is not None and memo[0] is cal:
            return memo[1]
        f = np.random.default_rng(20261010).dirichlet(np.ones(2 ** self.n))
        ref = np.asarray(self.mitigate(f), float)
        ok = bool(np.allclose(self._tensor_apply(self._lin_mats(), f), ref, rtol=1e-9, atol=1e-12))
        self._per_bit_memo = (cal, ok)
        return ok

    def mitigation_matrix(self) -> np.ndarray:
        """M with mitigate(f) = M f for the linear readout mitigations (M3 equals the tensored inverse when every
        bitstring is observed, which exact data guarantee). Built as a Kronecker product of the per-bit matrices
        (O(4ⁿ), dense: prefer o_eff, which never forms it)."""
        d = 2 ** self.n
        if self.recipe.readout == "none":
            return np.eye(d)
        if self._per_bit_ok():
            M = np.ones((1, 1))
            for Mi in reversed(self._lin_mats()):        # bit n−1 is the most significant factor
                M = np.kron(M, Mi)
            return M
        return np.stack([self.mitigate(e) for e in np.eye(d)], 1)

    def o_eff(self, o=None) -> np.ndarray:
        """Per-shot values with the readout mitigation folded in: o_eff = M^T o, so o_eff · f = o · (M f).
        Computed as Mᵀo = (⊗_i M_iᵀ) o in O(n·2ⁿ) without forming M, and cached per observable vector for the
        context's calibration (a fresh copy is returned)."""
        o = np.asarray(self.obs.o if o is None else o, float)
        if self.recipe.readout == "none":
            return o.copy()
        cal = self.calibration()
        cache = getattr(self, "_oeff_cache", None)
        if cache is None:
            cache = self._oeff_cache = {}
        if cache.get("cal") is not cal:
            cache.clear()
            cache["cal"] = cal
        key = (self.recipe.readout, o.shape, o.tobytes())
        hit = cache.get(key)
        if hit is None:
            if self._per_bit_ok():
                hit = self._tensor_apply([Mi.T for Mi in self._lin_mats()], o)
            else:
                hit = self.mitigation_matrix().T @ o
            cache[key] = hit
        return hit.copy()

    def var_shot(self, rr: RunResult, o=None) -> float:
        """Exact per-shot variance of the (readout-mitigated) estimator on the measured distribution."""
        return shot_var(self.o_eff(o), np.clip(self.measured(rr), 0, None))

    def sample_freqs(self, rr: RunResult, N: int, B: int, rng) -> np.ndarray:
        p = np.clip(self.measured(rr), 0, None)
        p = p / p.sum()
        N = max(int(N), 1)
        return rng.multinomial(N, p, size=B) / N

    def sample_expvals(self, rr: RunResult, N: int, B: int, rng, o=None) -> np.ndarray:
        """B estimates, each from N shots of this circuit, with this recipe's readout mitigation. The linear
        mitigations (none, tensored, TREX) fold into per-shot values, one product for all repeats: o·(M f) = o_eff·f;
        M3 is nonlinear in sampled data (it solves on the observed bitstrings) and runs repeat by repeat."""
        o = self.obs.o if o is None else o
        freqs = self.sample_freqs(rr, N, B, rng)
        if self.recipe.readout in ("none", "tensored", "trex"):
            return freqs @ self.o_eff(o)
        return np.array([o @ self.mitigate(f) for f in freqs])


def m3_solve(p_meas: np.ndarray, A_list: list, n: int, hamming: int | None = None, tol: float = 1e-12) -> np.ndarray:
    """M3 (Nation et al., PRX Quantum 2, 040326 (2021)): solve the assignment equations only on the
    bitstrings that were observed, with the reduced matrix's columns renormalized, optionally
    truncated at Hamming distance `hamming`. Exact infinite-shot data observe every bitstring, and
    M3 then equals the full tensored inverse."""
    obs = np.flatnonzero(np.abs(p_meas) > tol)          # signed (quasi-probability) data keep their negatives
    if len(obs) == 0:
        return p_meas
    m = len(obs)
    # Ared[a, b] = Π_i A_i[x_a,i, x_b,i], built bit by bit with whole-matrix products (the same multiplication order
    # as the element-wise definition, so identical values) — O(n·m²) array work instead of a Python triple loop
    Ared = np.ones((m, m))
    ham = np.zeros((m, m), dtype=np.int64) if hamming is not None else None
    for i in range(n):
        xb = (obs >> i) & 1
        Ared *= np.asarray(A_list[i], float)[xb[:, None], xb[None, :]]
        if ham is not None:
            ham += xb[:, None] != xb[None, :]
    if ham is not None:
        Ared[ham > hamming] = 0.0
    Ared /= Ared.sum(axis=0, keepdims=True)
    xsol = np.linalg.solve(Ared, p_meas[obs])
    out = np.zeros_like(p_meas)
    out[obs] = xsol
    return out
