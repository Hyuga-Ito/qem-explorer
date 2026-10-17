r"""Run mitigation recipes on IBM Quantum hardware (or locally on a fake backend) with the *same* estimator
code, the *same* equal-budget accounting and the *same* results suite as the simulator.

How it works (record → allocate → execute → replay)
---------------------------------------------------
1. **Record.** Every recipe runs against a :class:`HardwareRunner` in recording mode. Whenever an estimator asks
   for a circuit (a folded circuit, a PEC instance, a training circuit, a calibration circuit, …) the runner writes
   the request down with its *role* (target, training or calibration) and answers with a prediction: the circuit
   simulated with the device's calibrated median error rates (ideal answers for registers too large to predict).
   Circuits are deterministic (all randomness is seeded), so recording twice gives the same requests.
2. **Allocate.** The recorded result gives each recipe's cost model (qemx/budget.py), so the shot budget B per
   recipe is split exactly as in simulation: training shots as the method prescribes, calibration shots s* per
   calibration circuit, and the target shots N_t among the target circuits as the estimator's variance assumes —
   ZNE's optimal split N_j ∝ |η_j|σ_j, a Pauli-sum Hamiltonian's N_g ∝ √v_g, and N_t/K shots for each of PEC's K
   sampled circuits (shots_per_circuit). Requests of different recipes are never shared, so every contender is
   measured on its own budget: an exact equal-cost comparison.
3. **Execute.** The requests become Qiskit circuits on the backend's physical qubits and are sent with the Sampler
   primitive (one job per option group — gate twirling and dynamical decoupling are primitive options — and shot
   count). Measurement twirling for TREX is done explicitly: random X gates before measurement, bits flipped back
   afterwards.
4. **Replay.** The recipes run again; now every request is answered with the measured frequencies, and the
   estimator's usual arithmetic turns them into a mitigated value, its per-shot variance and its calibration
   sensitivity. :func:`analyze` returns the same result structure as the Playground's simulation (rows, samples,
   distributions, method internals), so the app draws the same figures and tables and exports the same ZIP.

Credentials never enter this module's state: callers pass a connected ``QiskitRuntimeService`` or a
backend object, and only job IDs are returned.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .backends.base import NotSupported
from .circuit import CircuitSpec, Op, layer_order
from .compile import Compiled
from .engine import RunResult
from .methods.base import REGISTRY, Context, Recipe, Runner
from .noise import NoiseSpec, pauli_matrix
from .resources import qiskit_serial

MEAS_TWIRLS = 8          # randomizations for explicit measurement twirling (TREX)
LOCAL_GATE_TWIRLS = 8    # twirl instances per circuit when the backend cannot twirl natively (local mode)
UNSUPPORTED = {"vd": "virtual distillation needs M copies and an entangling derangement circuit; "
                     "this version runs it only in simulation"}


NIL_OPTIMAL = ("NIL's optimal target split N_j ∝ |c_j|σ_j needs the fitted coefficients before the target circuits "
               "run, but a hardware plan sends every circuit in one batch: use target shot split = equal on hardware")


def supported(recipe: Recipe) -> tuple:
    """(runs on hardware?, why not). A method class may decide itself by defining ``hardware_supported(params)``
    returning (bool, reason); otherwise the known restrictions below apply."""
    if recipe.estimator in UNSUPPORTED:
        return False, UNSUPPORTED[recipe.estimator]
    m = REGISTRY.get(recipe.estimator)
    hook = getattr(m, "hardware_supported", None)
    if callable(hook):
        try:
            ok, why = hook(recipe.p)
        except Exception as e:                           # noqa: BLE001 - a faulty hook means "not supported"
            return False, "cannot tell whether this method runs on hardware (%s: %s)" % (type(e).__name__, e)
        return bool(ok), str(why or "")
    if recipe.estimator == "nil" and recipe.p.get("target_split", "optimal") == "optimal":
        return False, NIL_OPTIMAL
    return True, ""


# ------------------------------------------------------------------ requests
@dataclass
class Request:
    spec: CircuitSpec
    mm: dict
    twirl: str = "none"
    dd: str = "none"
    meas_twirl: bool = False
    tag: int = -1               # the recipe that asked for it: recipes never share measured counts
    role: str = "target"        # target / training / calibration: which part of the budget pays for it
    shots: int = 0              # assigned by the planner

    @property
    def key(self) -> str:
        d = dict(spec=self.spec.key(), mm=sorted(self.mm.items()), twirl=self.twirl, dd=self.dd, mt=self.meas_twirl,
                 tag=self.tag)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def group(self) -> tuple:
        return (self.twirl, self.dd)


def _pauli_label_of(M: np.ndarray, k: int) -> str | None:
    from .noise import pauli_labels
    for l in pauli_labels(k):
        P = pauli_matrix(l)
        ov = np.trace(P.conj().T @ M) / 2 ** k
        if abs(abs(ov) - 1) < 1e-9:
            return l
    return None


def explicit_corrections(spec: CircuitSpec, corrections) -> CircuitSpec:
    """Turn a PEC correction callback (Pauli matrices after gates) into explicit Pauli gates."""
    if corrections is None:
        return spec
    ops_all, _, _ = layer_order(spec)
    exec_ops = [o for o in ops_all if o.kind != "barrier"]
    out = []
    for step, o in enumerate(exec_ops):
        out.append(o)
        for ins in corrections(o, step):
            if ins.kind != "unitary":
                raise NotSupported("only Pauli corrections can be sent to hardware")
            lab = _pauli_label_of(np.asarray(ins.data), len(ins.qubits))
            if lab is None:
                raise NotSupported("a correction is not a Pauli")
            # label character j acts on ins.qubits[j] (little-endian labels as in noise.pauli_matrix)
            for ch, q in zip(lab, ins.qubits):
                if ch != "I":
                    out.append(Op(ch.lower(), (q,), (), "gate", "pec"))
    return spec.replace(ops=tuple(out))


class HardwareRunner(Runner):
    """Records circuit requests (answering with a prediction), or answers them with measured frequencies."""

    def __init__(self, freqs: dict | None = None, dm=None, predict=None):
        super().__init__("hardware")
        self.ideal = Runner("numpy")
        self.freqs = freqs
        self.predict = predict           # (spec, mm, twirl, dd) -> predicted noisy distribution, or None
        self.requests: dict = {}
        self.log: list = []              # (role, key) of every call, in order (duplicates kept)
        self.tag = -1                    # the recipe being recorded or replayed
        self.missing: list = []
        self._dm = dm

    @property
    def device_model(self):
        return self._dm

    @property
    def recording(self) -> bool:
        return self.freqs is None

    def run(self, spec, noise, measure_map, *, ideal=False, twirl="none", dd="none", corrections=None, corr_key=None,
            snapshots=False, backend=None, need_state=False, meas_twirl=False, role="target") -> RunResult:
        if ideal or backend == "numpy":
            return self.ideal.run(spec, noise, measure_map, ideal=ideal, twirl=twirl, dd=dd, corrections=corrections,
                                  corr_key=corr_key, backend="numpy")
        if need_state:
            raise NotSupported("this estimator needs the full quantum state, which hardware does not return")
        spec2 = explicit_corrections(spec, corrections)
        req = Request(spec2, dict(measure_map), twirl, dd, bool(meas_twirl), int(self.tag), role)
        k = req.key
        self.log.append((role, k))
        if self.recording:
            self.requests.setdefault(k, req)
            self.calls += 1
            p = None
            if self.predict is not None:
                try:
                    p = self.predict(spec2, measure_map, twirl, dd)
                except Exception:                       # noqa: BLE001 - fall back to the ideal answer
                    p = None
            if p is None:
                p = self.ideal.run(spec2, NoiseSpec(), measure_map, ideal=True, backend="numpy").probs
            return RunResult(spec.n, p, p, None, [], [], None, 1)
        f = self.freqs.get(k)
        if f is None:
            self.missing.append(k)
            raise KeyError("no measured counts for a requested circuit (was the job for this recipe completed?)")
        return RunResult(spec.n, f, f, None, [], [], None, 1)


class HardwareContext(Context):
    """Context whose readout calibration and TREX come from calibration circuits run on the device."""

    def run(self, spec: CircuitSpec | None = None, **kw) -> RunResult:
        kw.setdefault("twirl", self.recipe.twirl)
        kw.setdefault("dd", self.recipe.dd)
        kw.setdefault("meas_twirl", self.recipe.readout == "trex")
        return self.runner.run(spec or self.spec, kw.pop("noise", self.noise), self.mm, **kw)

    def measured(self, rr: RunResult) -> np.ndarray:
        return rr.probs_meas

    def calibration(self) -> dict:
        if self._cal is None:
            n = self.n
            trex = self.recipe.readout == "trex"
            zero = CircuitSpec(n, (), "cal-0")
            one = CircuitSpec(n, tuple(Op("x", (q,), (), "gate", "prep") for q in range(n)), "cal-1")
            # measured with the target's map: bit i is read from qubit mm[i] in calibration and target alike
            p0 = self.runner.run(zero, self.noise, self.mm, meas_twirl=trex, role="calibration").probs_meas
            bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
            if trex:
                # twirled readout is symmetric: P(flip) = p̄_i, measured on |0…0⟩
                pbar = [float(p0 @ bits[:, i]) for i in range(n)]
                self._cal = dict(A=[np.array([[1 - p, p], [p, 1 - p]]) for p in pbar], pbar=pbar, circuits=1)
            else:
                p1 = self.runner.run(one, self.noise, self.mm, role="calibration").probs_meas
                A = []
                for i in range(n):
                    e = float(p0 @ bits[:, i])
                    h = float(p1 @ (1 - bits[:, i]))
                    A.append(np.array([[1 - e, h], [e, 1 - h]]))
                self._cal = dict(A=A, pbar=[(a[1, 0] + a[0, 1]) / 2 for a in A], circuits=2)
        return self._cal

    def mitigate(self, f: np.ndarray) -> np.ndarray:
        if self.recipe.readout == "trex":
            return self._tensor_apply([np.linalg.inv(np.array([[1 - p, p], [p, 1 - p]]))
                                       for p in self.calibration()["pbar"]], f)
        return super().mitigate(f)


# ------------------------------------------------------------------ plan
PREDICT_MAX_QUBITS = 10   # predicted (noisy) answers for the allocation up to this size; ideal answers beyond


@dataclass
class Plan:
    compiled: Compiled
    observable: object
    recipes: list
    shots: int                                                 # the shot budget B per recipe
    symmetry: object = None
    seed: int = 0
    noise: NoiseSpec = field(default_factory=NoiseSpec)       # the model PEC/TREX assume (device medians)
    dm: object = None                                          # DeviceModel of the used qubits (per-gate errors)
    requests: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)                # recipe label -> reason (display; "(#k)" for repeats)
    skipped_idx: dict = field(default_factory=dict)            # recipe index -> reason (identity)
    alloc: dict = field(default_factory=dict)                  # recipe index -> allocation (see _allocate)
    pubs: dict = field(default_factory=dict)                   # request key -> circuits sent (twirl/mask variants)
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        groups = {}
        for r in self.requests.values():
            groups.setdefault(r.group, 0)
            groups[r.group] += 1
        return dict(circuits=len(self.requests), groups={"%s / %s" % g: c for g, c in groups.items()},
                    shots=int(sum(r.shots for r in self.requests.values())))

    def table(self) -> list:
        """One row per recipe: how its budget is spent on hardware."""
        out = []
        for i, (rc, t) in enumerate(zip(self.recipes, self.labels())):
            a = self.alloc.get(i)
            if a is None:
                out.append({"contender": t, "status": getattr(self, "skipped_idx", {}).get(i, "not planned")})
                continue
            reqs = [r for r in self.requests.values() if r.tag == i]
            out.append({"contender": t, "status": "planned", "circuits": len(reqs),
                        "target shots": a["target"], "training shots": a["training"],
                        "calibration shots": a["calibration"], "total shots": a["target"] + a["training"] + a["calibration"],
                        "shots per PEC circuit": a.get("spc") or None})
        return out

    def labels(self) -> list:
        """Contender labels as the results show them (a repeated title gets "(#k)")."""
        from .export import recipe_labels
        return recipe_labels(self.recipes)

    def skip(self, idx: int, reason: str):
        self.skipped_idx[idx] = reason
        self.skipped[self.labels()[idx]] = reason

    def request_table(self) -> list:
        labels = self.labels()
        return [dict(key=k, recipe=labels[r.tag] if 0 <= r.tag < len(self.recipes) else "", role=r.role,
                     shots=r.shots, circuit=r.spec.name or "", twirl=r.twirl, dd=r.dd, measurement_twirl=r.meas_twirl)
                for k, r in self.requests.items()]


def _predictor(compiled: Compiled, noise: NoiseSpec):
    """Predicted measured distribution of a request under the device's median error rates (for the allocation)."""
    if compiled.implemented.n > PREDICT_MAX_QUBITS or noise.is_ideal:
        return None
    sim = Runner("numpy")

    def predict(spec, mm, twirl, dd):
        return sim.run(spec, noise, mm, twirl=twirl, dd=dd).probs_meas
    return predict


def _group_weights(parts) -> np.ndarray:
    """N_g ∝ √v_g, exactly as hamiltonian.combine splits a Pauli sum's shots (one group: [1])."""
    sv = np.sqrt([max(float(r.var1), 0.0) for _, r in parts])
    return sv / sv.sum() if sv.sum() > 0 else np.full(len(parts), 1.0 / max(len(parts), 1))


def _shares(r, n: int) -> np.ndarray:
    """The estimator's own split of its target shots among its n target circuits (ZNE: N_j ∝ |η_j|σ_j)."""
    a = (r.details or {}).get("alloc")
    if a is not None and len(a) == n:
        a = np.asarray(a, float)
        if np.all(np.isfinite(a)) and a.sum() > 0:
            return a / a.sum()
    return np.full(n, 1.0 / n) if n else np.zeros(0)


def _record(runner, idx, rc, compiled, observable, plan, group_params):
    """Run recipe idx once on the runner; returns (combined result, [(log of this group, group result)])."""
    from . import hamiltonian as HM
    parts, counter = [], [0]
    runner.tag = idx

    def make_ctx(c, o):
        return HardwareContext(c, plan.noise, o, rc, runner, shots=plan.shots, symmetry=plan.symmetry, seed=plan.seed)

    def run_method(ctx):
        g = counter[0]
        counter[0] += 1
        p = dict(rc.p)
        p.update(group_params.get(g, {}))
        i0 = len(runner.log)
        r = REGISTRY[rc.estimator].run(ctx, **p)
        parts.append((list(runner.log[i0:]), r))
        return r
    R = HM.evaluate(make_ctx, run_method, compiled, observable, plan.shots)
    return R, parts


def _allocate(R, parts, B: int) -> tuple:
    """({request key: shots}, allocation summary) for one recipe at budget B, from its recorded cost model."""
    from . import budget as BG
    m = BG.CostModel.of(R)
    N, s = BG.split(m, B)
    if N < 1:
        return None, dict(min_budget=BG.min_budget(m))
    w = _group_weights(parts)
    shots, roles = {}, {}
    for g, (log, r) in enumerate(parts):
        tgt = [k for role, k in log if role == "target"]
        for k, a in zip(tgt, _shares(r, len(tgt))):
            shots[k] = shots.get(k, 0.0) + N * w[g] * a
            roles[k] = "target"
        per = (r.fixed_shots / r.train_circuits) if r.train_circuits else 0.0
        for role, k in log:
            if role == "training":
                shots[k] = shots.get(k, 0.0) + per
                roles[k] = "training"
        for role, k in log:
            if role == "calibration":                 # one calibration serves every group: set, never summed
                shots[k] = max(1.0, round(s))
                roles[k] = "calibration"
    shots = {k: max(int(round(v)), 1) for k, v in shots.items()}
    tot = {x: int(sum(v for k, v in shots.items() if roles[k] == x)) for x in ("target", "training", "calibration")}
    ncal = sum(1 for k in shots if roles[k] == "calibration")
    return shots, dict(N_t=float(N), s=float(s), target=tot["target"], training=tot["training"],
                       calibration=tot["calibration"], cal_circuits=ncal, roles=roles)


def make_plan(compiled: Compiled, observable, recipes: list, shots: int, symmetry=None, seed: int = 0,
              backend=None, predict: bool = True) -> Plan:
    """Record every recipe and allocate its budget of ``shots`` (B) exactly as the simulator's accounting does.

    backend: the target BackendV2; its calibration gives PEC's assumed rates, the learned per-gate model and the
    predicted distributions the allocation is computed from."""
    noise, dm = NoiseSpec(), None
    if backend is not None:
        from .device import model_for
        try:
            dm = model_for(backend, tuple(compiled.info["phys"]))
            noise = dm.equivalent_noise()
        except Exception:
            dm = None
    from .resources import RunTimeout
    plan = Plan(compiled, observable, list(recipes), int(shots), symmetry, seed, noise, dm)
    if any(o.kind == "noise" for o in compiled.implemented.ops):
        for i in range(len(plan.recipes)):
            plan.skip(i, "the circuit contains noise channels you placed; hardware has its own noise")
        return plan
    pred = _predictor(compiled, noise) if predict else None
    if pred is None and predict and compiled.implemented.n > PREDICT_MAX_QUBITS:
        plan.notes.append("%d qubits: the shot split is planned from the ideal circuit (no noisy prediction)."
                          % compiled.implemented.n)
    runner = HardwareRunner(dm=dm, predict=pred)
    B = int(shots)
    for idx, rc in enumerate(plan.recipes):
        ok, why = supported(rc)
        if not ok:
            plan.skip(idx, why)
            continue
        try:
            gp = {}
            R, parts = _record(runner, idx, rc, compiled, observable, plan, gp)
            if rc.estimator == "pec":
                # K sampled circuits per group, N_t·w_g/K shots each: the estimator's variance must know s
                K = max(int(rc.p.get("instances", 64)), 1)
                for _ in range(3):
                    from . import budget as BG
                    N, _s = BG.split(BG.CostModel.of(R), B)
                    w = _group_weights(parts)
                    new = {g: dict(shots_per_circuit=max(1, int(round(N * w[g] / K)))) for g in range(len(parts))}
                    if new == gp:
                        break
                    gp = new
                    R, parts = _record(runner, idx, rc, compiled, observable, plan, gp)
            shots_by_key, info = _allocate(R, parts, B)
            if shots_by_key is None:
                plan.skip(idx, "insufficient_budget: needs at least %s shots" % f"{int(math.ceil(info['min_budget'])):,}")
                continue
            for k, n_ in shots_by_key.items():
                req = runner.requests[k]
                req.shots, req.role = n_, info["roles"][k]
                plan.requests[k] = req
            info.pop("roles")
            info.update(group_params=gp, spc=(gp.get(0) or {}).get("shots_per_circuit"))
            plan.alloc[idx] = info
        except NotSupported as e:
            plan.skip(idx, str(e))
        except (MemoryError, KeyboardInterrupt, RunTimeout):
            raise                                        # the time limit / memory stops the whole plan
        except Exception as e:                           # noqa: BLE001 - one recipe must not stop the plan
            plan.skip(idx, "%s: %s" % (type(e).__name__, e))
    return plan


def _replay(plan: Plan, runner, idx: int, rc):
    info = plan.alloc[idx]
    R, _ = _record(runner, idx, rc, plan.compiled, plan.observable, plan, info.get("group_params") or {})
    return R


def distributions(plan: Plan, runner, outs) -> dict:
    """Measured distributions for the analysis views: ideal, measured (unmitigated), readout-corrected and
    ZNE-extrapolated — each from the counts of the contender that measured it, mitigated by that contender."""
    from .insight import MAX_DIST_QUBITS
    from .methods.zne import linear_weights, richardson_weights
    from .transforms import achieved_lambda, fold_global, fold_local
    comp = plan.compiled
    n = comp.logical.n
    if n > MAX_DIST_QUBITS:
        return dict(series={}, notes=["%d classical bits: the distribution is not drawn." % n], readout=[], n=n)
    if hasattr(plan.observable, "terms"):
        return dict(series={}, notes=["The observable is a Hamiltonian measured in several settings; the "
                                      "computational-basis distribution was not measured on its own."], readout=[], n=n)
    idx_of = {id(o): i for i, o in enumerate(outs)}
    done = [o for o in outs if o.result is not None]
    ideal = runner.ideal.run(comp.implemented, NoiseSpec(), comp.measure_map, ideal=True, backend="numpy")
    series = {"ideal": np.asarray(ideal.probs, float)}
    notes, mats = [], []

    def ctx_for(o):
        runner.tag = idx_of[id(o)]
        return HardwareContext(comp, plan.noise, None, o.recipe, runner, shots=plan.shots, symmetry=plan.symmetry,
                               seed=plan.seed)
    raw = next((o for o in done if o.recipe.estimator == "raw" and o.recipe.readout == "none"
                and o.recipe.twirl == "none" and o.recipe.dd == "none"), None)
    try:
        if raw is not None:
            series["noisy (unmitigated)"] = np.asarray(ctx_for(raw).run().probs_meas, float)
            notes.append("Measured on the device (%s): the unmitigated contender's counts." % raw.title)
    except Exception:                                    # noqa: BLE001
        pass
    ro = next((o for o in done if o.recipe.estimator == "raw" and o.recipe.readout != "none"), None)
    try:
        if ro is not None:
            ctx = ctx_for(ro)
            mats = [np.asarray(A, float) for A in ctx.calibration()["A"]]
            series["readout-corrected"] = np.asarray(ctx.mitigate(ctx.measured(ctx.run())), float)
            notes.append("Readout-corrected (%s): the %s contender's counts and its own calibration circuits." % (
                {"tensored": "tensored inverse", "m3": "M3", "trex": "TREX"}[ro.recipe.readout], ro.title))
    except Exception:                                    # noqa: BLE001
        pass
    zne = next((o for o in done if o.recipe.estimator == "zne" and o.recipe.p.get("rule") in ("richardson", "linear")),
               None)
    try:
        if zne is not None:
            p = zne.recipe.p
            ctx = ctx_for(zne)
            ps, lams = [], []
            spec = comp.implemented
            from .methods.zne import draw_seed, is_fractional, local_fold_weights
            sub = p.get("subset", "random")
            fw = local_fold_weights(ctx, spec) if sub == "balanced" else None
            for lam in sorted(float(x) for x in p.get("scales", (1, 3, 5))):
                if p.get("folding", "global") == "global":
                    folds = [fold_global(spec, lam)]
                else:                                    # the same folded circuits the method ran (every draw)
                    K = int(p.get("fold_draws", 1) or 1) if is_fractional(lam) else 1
                    folds = [fold_local(spec, lam, draw_seed(ctx.seed, k), sub, fw) for k in range(K)]
                ps.append(np.mean([np.asarray(ctx.mitigate(ctx.measured(ctx.run(fs))), float) for fs in folds], 0))
                lams.append(achieved_lambda(spec, folds[0]))
            eta = richardson_weights(lams) if p["rule"] == "richardson" else linear_weights(lams)
            series["ZNE-extrapolated"] = np.tensordot(eta, np.stack(ps), axes=1)
            notes.append("ZNE-extrapolated (%s, λ = %s): the measured distributions combined with η = %s." % (
                p["rule"], ", ".join("%.3g" % x for x in lams), ", ".join("%+.3f" % e for e in eta)))
    except Exception:                                    # noqa: BLE001
        pass
    if "noisy (unmitigated)" not in series:
        notes.append("Add the Unmitigated contender to see the measured distribution.")
        return dict(series={}, notes=notes, readout=[A.tolist() for A in mats], n=n)
    return dict(series=series, notes=notes, readout=[A.tolist() for A in mats], n=n)


def analyze(plan: Plan, freqs: dict, t_shot_us: float = 0.0, repeats: int = 120, seed: int = 5,
            label: str = "hardware") -> dict:
    """Replay every recipe on measured frequencies. Returns the same result structure as the Playground's
    simulation: {ideal, outs, rows, samples, shots, dist, observable, hardware, …}."""
    from . import hamiltonian as HM
    from .recipes import Outcome, table_rows
    from .resources import RunTimeout
    runner = HardwareRunner(freqs, dm=plan.dm)
    ideal = HM.ideal_value(lambda c, o: Context(c, NoiseSpec(), o, Recipe.make("raw"), Runner("numpy")),
                           plan.compiled, plan.observable)
    outs, allocs = [], {}
    for idx, rc in enumerate(plan.recipes):
        if idx not in plan.alloc:
            outs.append(Outcome(rc, None, getattr(plan, "skipped_idx", {}).get(idx, "not planned")))
            continue
        try:
            R_ = _replay(plan, runner, idx, rc)
            # the statistical contract: a replay of measured counts is one experiment's estimate, not an expectation
            R_.estimate_kind = "measured"
            R_.flags = tuple(dict.fromkeys(tuple(R_.flags or ()) + ("measured",)))
            outs.append(Outcome(rc, R_))
        except (MemoryError, KeyboardInterrupt, RunTimeout):
            raise
        except Exception as e:                           # noqa: BLE001 - the other recipes still report
            outs.append(Outcome(rc, None, "%s: %s" % (type(e).__name__, e)))
    seen = {}
    for idx, o in enumerate(outs):
        t = o.recipe.title()
        seen[t] = seen.get(t, 0) + 1
        o.label = t if seen[t] == 1 else "%s (#%d)" % (t, seen[t])
        a = plan.alloc.get(idx)
        if a is not None:
            sent = [r for r in plan.requests.values() if r.tag == idx]
            n_cal = sum(1 for r in sent if r.role == "calibration")
            s_cal = (a["calibration"] / n_cal) if n_cal else 0.0
            by_role = {x: int(sum(plan.pubs.get(r.key, 1) for r in sent if r.role == x))
                       for x in ("target", "training", "calibration")}
            allocs[o.title] = dict(N=float(a["target"]), s=float(s_cal), training=float(a["training"]),
                                   calibration=float(a["calibration"]), circuits_by_role=by_role,
                                   shots=int(a["target"] + a["training"] + a["calibration"]))
    samples, rows = table_rows(outs, ideal, plan.shots, t_shot_us, repeats=repeats, seed=seed, allocs=allocs)
    try:
        dist = distributions(plan, runner, outs)
    except Exception as e:                               # noqa: BLE001
        dist = dict(series={}, notes=["Distributions unavailable: %s" % e], readout=[], n=plan.compiled.logical.n)
    return dict(ideal=ideal, outs=outs, rows=rows, samples=samples, shots=int(plan.shots), calls=len(plan.requests),
                backend=label, dist=dist, observable=plan.observable, hardware=True, n=plan.compiled.logical.n,
                allocs=allocs)


# ------------------------------------------------------------------ circuits
@qiskit_serial
def to_physical(req: Request, phys: list, num_qubits: int, rng=None, mask: int = 0):
    """A QuantumCircuit on the device's physical qubits (measurement mask = explicit readout twirl)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    spec = req.spec
    n = spec.n
    qc = QuantumCircuit(QuantumRegister(num_qubits, "q"), ClassicalRegister(n, "c"))
    for o in spec.ops:
        if o.kind == "barrier":
            qs = [phys[q] for q in (o.qubits or range(n))]
            qc.barrier(*qs)
        elif o.kind == "gate":
            getattr(qc, o.name)(*o.params, *[phys[q] for q in o.qubits])
        elif o.kind == "noise":
            raise NotSupported("noise channels placed in the circuit cannot run on hardware; remove them first")
    inv = {c: q for c, q in req.mm.items()}
    if sorted(inv) != list(range(n)):
        raise NotSupported("the circuit has %d qubits but its measurement map reads %d classical bits; every qubit "
                           "must be measured exactly once" % (n, len(inv)))
    for c in range(n):
        if (mask >> c) & 1:
            qc.x(phys[inv[c]])
    for c in range(n):
        qc.measure(phys[inv[c]], c)
    return qc


TWO_QUBIT_NATIVE = ("cz", "ecr", "cx")


def two_qubit_count(qc) -> int:
    """Two-qubit gates of a QuantumCircuit (barriers and measurements excluded)."""
    return sum(1 for ins in qc.data if ins.operation.num_qubits == 2 and ins.operation.name not in ("barrier",))


def fold_guard_transpiled(before: list, after: list, keys: list) -> None:
    """ZNE's folded circuits must reach the device unaltered: every circuit sent must have exactly the two-qubit gates
    it had before transpilation (optimization level 0 translates gate by gate; any level >= 1 cancels G·G†·G back to
    G, see tests/test_v13_phase.py). The circuits here are already compiled for the device (native two-qubit gates
    only), so the counts must be equal. Raises RuntimeError naming the first circuit that changed."""
    for k, b, a in zip(keys, before, after):
        nb, na = two_qubit_count(b), two_qubit_count(a)
        nonnative = [ins.operation.name for ins in b.data if ins.operation.num_qubits == 2
                     and ins.operation.name not in TWO_QUBIT_NATIVE + ("barrier",)]
        if nonnative:
            continue                       # a non-native two-qubit gate is decomposed (e.g. CP -> 2 CZ): not 1:1
        if nb != na:
            raise RuntimeError("transpilation changed the two-qubit gate count of %s from %d to %d: folds or gates were "
                               "merged; refusing to submit" % (k, nb, na))


@qiskit_serial
def build_jobs(plan: Plan, backend, native_twirl: bool, native_dd: bool, seed: int = 0) -> list:
    """[(options dict, [(request key, mask, QuantumCircuit, shots)])], one entry per job."""
    from qiskit import transpile
    from .compile import QISKIT_LOCK
    from .transforms import twirl_instance
    phys = plan.compiled.info["phys"]
    N = backend.num_qubits
    rng = np.random.default_rng(seed + 101)
    groups = {}
    for key, req in plan.requests.items():
        groups.setdefault(req.group, []).append((key, req))
    jobs = []
    for (tw, dd), items in groups.items():
        pubs = []
        for key, req in items:
            variants = [req]
            if tw != "none" and not native_twirl:
                variants = [dataclasses.replace(req, spec=twirl_instance(req.spec, seed=1000 + j))
                            for j in range(LOCAL_GATE_TWIRLS)]
            masks = [int(rng.integers(0, 2 ** req.spec.n)) for _ in range(MEAS_TWIRLS)] if req.meas_twirl else [0]
            combos = [(v, m) for v in variants for m in masks]
            total = int(req.shots or plan.shots)
            L = min(len(combos), total)
            plan.pubs[key] = L
            for j, (v, m) in enumerate(combos[:L]):            # the request's shots, split exactly
                pubs.append((key, m, to_physical(v, phys, N, mask=m), total // L + (1 if j < total % L else 0)))
        with QISKIT_LOCK:
            circs = transpile([p[2] for p in pubs], target=backend.target, optimization_level=0,
                              initial_layout=list(range(N)), seed_transpiler=seed)
        fold_guard_transpiled([p[2] for p in pubs], circs, [p[0] for p in pubs])
        pubs = [(k, m, c, s) for (k, m, _, s), c in zip(pubs, circs)]
        opts = {}
        if tw != "none" and native_twirl:
            opts["twirling"] = dict(enable_gates=True, enable_measure=False)
        if dd != "none" and native_dd:
            opts["dynamical_decoupling"] = dict(enable=True, sequence_type="XY4" if dd == "XY4" else "XX")
        jobs.append((opts, pubs, (tw, dd)))
    return jobs


def counts_to_freqs(results: list, n: int) -> dict:
    """results: [(request key, mask, counts dict)] -> {key: frequency vector over n classical bits}."""
    acc = {}
    for key, mask, counts in results:
        v = acc.setdefault(key, np.zeros(2 ** n))
        for bitstr, c in counts.items():
            x = int(bitstr.replace(" ", ""), 2) ^ mask
            v[x] += c
    return {k: v / v.sum() for k, v in acc.items() if v.sum() > 0}


def _pub_counts(pub_result):
    data = pub_result.data
    for name in ("c", "meas"):
        reg = getattr(data, name, None)
        if reg is not None:
            return reg.get_counts()
    return pub_result.join_data().get_counts()


def _by_shots(pubs) -> list:
    """Pubs grouped by their shot count, in first-seen order: one Sampler call per shot count (a single call with
    mixed per-pub shots is deprecated in qiskit-ibm-runtime ≥ 0.48)."""
    groups = {}
    for p in pubs:
        groups.setdefault(int(p[3]), []).append(p)
    return list(groups.items())


@qiskit_serial
def _local_simulator(dm):
    """AerSimulator with the device's noise snapshot restricted to the qubits the plan uses: the same gate errors
    (depolarizing + thermal relaxation over each gate's duration) and readout errors that
    ``AerSimulator.from_backend`` builds for the whole device, without simulating the noise model of every idle
    qubit (a 156-qubit model costs seconds per circuit). Cached on the DeviceModel."""
    sim = dm.noise.get("__local_sim__")
    if sim is not None:
        return sim
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device.models import basic_device_gate_errors, basic_device_readout_errors
    from .device import _sub_target
    tg = _sub_target(dm)
    nm = NoiseModel(basis_gates=sorted({"rz", "sx", "x", dm.two_qubit, "id"}))
    for qubits, err in basic_device_readout_errors(target=tg):
        nm.add_readout_error(err, qubits)
    for name, qargs, err in basic_device_gate_errors(target=tg, thermal_relaxation=True):
        nm.add_quantum_error(err, name, qargs)
    sim = AerSimulator(noise_model=nm, method="density_matrix")
    dm.noise["__local_sim__"] = sim
    return sim


def _compact(qc, index: dict):
    """The circuit on the compact qubits 0..k−1 (physical qubit index[p] → compact qubit)."""
    from qiskit import QuantumCircuit
    out = QuantumCircuit(len(index), qc.num_clbits)
    for ins in qc.data:
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "barrier":
            qs = [index[q] for q in qs if q in index]
            if qs:
                out.barrier(*qs)
            continue
        out.append(ins.operation, [index[q] for q in qs], [qc.find_bit(c).index for c in ins.clbits])
    return out


@qiskit_serial
def run_local(plan: Plan, fake_name: str, seed: int = 0, progress=None) -> dict:
    """Execute the plan with the Sampler in local testing mode on a fake backend (no account needed). The circuits
    are transpiled for the device exactly as for submission; the Sampler then runs them on Qiskit Aer with the
    device's noise snapshot on the qubits they use."""
    from qiskit_ibm_runtime import SamplerV2
    from .device import fake_backend, model_for
    b = fake_backend(fake_name)
    results = []
    jobs = build_jobs(plan, b, native_twirl=False, native_dd=False, seed=seed)
    used = sorted({qc_q for _, pubs, _ in jobs for _, _, c, _ in pubs
                   for ins in c.data for qc_q in (c.find_bit(q).index for q in ins.qubits)
                   if ins.operation.name != "barrier"})
    dm = plan.dm if (plan.dm is not None and set(used) <= set(plan.dm.phys)) else model_for(b, tuple(used))
    index = {p: i for i, p in enumerate(dm.phys)}
    sim = _local_simulator(dm)
    calls = [(j, s_, grp) for j, (_, pubs, _) in enumerate(jobs) for s_, grp in _by_shots(pubs)]
    for i, (j, shots, grp) in enumerate(calls):
        sampler = SamplerV2(mode=sim)
        sampler.options.simulator.seed_simulator = seed + 7919 * j + i
        res = sampler.run([_compact(c, index) for _, _, c, _ in grp], shots=shots).result()
        for (k, m, _, _), pr in zip(grp, res):
            results.append((k, m, _pub_counts(pr)))
        if progress:
            progress((i + 1) / len(calls))
    return counts_to_freqs(results, plan.compiled.implemented.n)


class SubmitError(RuntimeError):
    """Submission stopped part-way. ``records`` are the jobs that WERE submitted (they run and are billed), so the
    caller can keep them and collect their results."""

    def __init__(self, message: str, records: list):
        super().__init__(message)
        self.records = list(records)


def submit(plan: Plan, backend, max_execution_time: int | None = None, seed: int = 0, ledger=None, model=None,
           mode: str = "batch") -> list:
    """Submit to a real backend (a BackendV2 from a connected service). Returns job records (no secrets).

    One job per option group (twirling / DD) and shot count, inside a Batch when the account allows it and ``mode``
    is "batch" (job mode otherwise). If a submission fails after others went through, SubmitError carries the
    records of the jobs already submitted, so none of them is lost. With a campaign ``ledger``
    (:mod:`qemx.campaign`) every job is admitted against the campaign's remaining QPU allowance before it is
    submitted, and its id is recorded at once; a job that does not fit stops the submission."""
    from qiskit_ibm_runtime import SamplerV2
    batch = None
    if mode == "batch":
        try:
            from qiskit_ibm_runtime import Batch
            batch = Batch(backend=backend)
        except Exception:                                # noqa: BLE001 - plans without batch mode use job mode
            batch = None
    records = []
    try:
        jobs = build_jobs(plan, backend, native_twirl=True, native_dd=True, seed=seed)   # everything checked first
    except BaseException:
        if batch is not None:
            try:
                batch.close()
            except Exception:                           # noqa: BLE001
                pass
        raise
    try:
        for opts, pubs, group in jobs:
            for shots, grp in _by_shots(pubs):
                sampler = SamplerV2(mode=batch if batch is not None else backend)
                if "twirling" in opts:
                    sampler.options.twirling.enable_gates = True
                    sampler.options.twirling.enable_measure = False
                if "dynamical_decoupling" in opts:
                    sampler.options.dynamical_decoupling.enable = True
                    sampler.options.dynamical_decoupling.sequence_type = opts["dynamical_decoupling"]["sequence_type"]
                rid, met = None, max_execution_time
                if ledger is not None:
                    from .campaign import CampaignLimit, estimate_usage
                    est = estimate_usage(int(shots) * len(grp), len(grp), 1, model)
                    try:
                        rid = ledger.admit("app: %d circuits × %d shots" % (len(grp), shots), est, "playground")
                    except CampaignLimit as e:
                        if not records:
                            raise
                        raise SubmitError("campaign ledger: %s" % e, records) from e
                    if not met:
                        met = int(math.ceil(est.high_s * 1.1)) + 1
                if met:
                    sampler.options.max_execution_time = int(met)
                try:
                    job = sampler.run([c for _, _, c, _ in grp], shots=shots)
                    job_id = job.job_id()
                except Exception as e:
                    if rid is not None:
                        ledger.release(rid, "submission failed: %s" % type(e).__name__)
                    if not records:
                        raise
                    raise SubmitError("%s: %s" % (type(e).__name__, e), records) from e
                if rid is not None:
                    ledger.submitted(rid, job_id)
                records.append(dict(job_id=job_id, backend=backend.name, group=list(group), shots=int(shots),
                                    pubs=[(k, m) for k, m, _, _ in grp]))
    finally:
        if batch is not None:
            try:
                batch.close()                           # no new jobs; the submitted ones still run
            except Exception:                           # noqa: BLE001
                pass
    return records


FINISHED = ("DONE", "JobStatus.DONE")
FAILED = ("ERROR", "CANCELLED", "JobStatus.ERROR", "JobStatus.CANCELLED")


def collect(service, records: list, n: int, wait: bool = False, cache: dict | None = None) -> tuple:
    """(frequencies or None, [status per job]) for submitted jobs.

    Frequencies are returned once every job has ended — finished, failed or cancelled; the requests of a failed job
    simply have no counts (their recipes then report it). Each status carries ``failed``. ``cache``
    ({job id: [(key, mask, counts)]}) keeps the counts of finished jobs, so checking again downloads only new ones."""
    results, status = [], []
    ended = True
    cache = {} if cache is None else cache
    for rec in records:
        jid = rec["job_id"]
        if jid in cache:
            status.append(dict(job_id=jid, status="DONE", failed=False))
            results.extend(cache[jid])
            continue
        job = service.job(jid)
        st = str(job.status())
        failed = st in FAILED
        status.append(dict(job_id=jid, status=st, failed=failed))
        if failed:
            continue
        if not wait and st not in FINISHED:
            ended = False
            continue
        res = job.result()
        got = [(k, m, _pub_counts(pr)) for (k, m), pr in zip(rec["pubs"], res)]
        cache[jid] = got
        results.extend(got)
    return (counts_to_freqs(results, n) if ended and results else None), status
