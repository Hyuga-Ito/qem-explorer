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
   recipe is split as in simulation, in integers that add up to B exactly: integer role budgets first (training
   shots as the method prescribes, an integer number s of shots per calibration circuit — the integer optimum near
   s* — and the target shots N_t = the rest), then each role is split over its *distinct* circuits with a
   budget-conserving rule (``budget.integer_split``: ZNE's optimal split N_j ∝ |η_j|σ_j, a Pauli-sum Hamiltonian's
   N_g ∝ √v_g; a circuit requested twice is measured once with the sum of its shares). PEC keeps its configured
   protocol: s = ``shots_per_circuit`` shots per sampled circuit and K_g = ⌊N_g/s⌋ sampled circuits (one shot per
   circuit by default, the textbook estimator); only an explicit hardware limit on the number of sampled circuits
   (``pec_max_circuits``) raises s, and the plan records requested and effective protocol side by side
   (:meth:`Plan.protocol_table`, :meth:`Plan.effective_recipe`). The feasibility verdict is the shared contract of
   ``budget.min_total_budget``. Every plan asserts that each contender's requests add up to its budget, or records a
   smaller ``effective_budget`` with its reason (PEC with s > 1 and no calibration circuit to take the remainder);
   :func:`build_jobs` checks the same totals after the measurement-twirl and gate-twirl expansion. Requests of
   different recipes are never shared, so every contender is measured on its own budget: an equal-*shot* comparison
   (shots, distinct circuits, circuit instances and QPU seconds are different resources: :meth:`Plan.resources`).
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
    multiplicity: int = 1       # times the recipe asked for this very circuit (merged: measured once, shares added)

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

    @property
    def returns_measured_counts(self) -> bool:
        """Replay: answers are measured frequencies (finite shots), not predicted probabilities."""
        return self.freqs is not None

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
    protocols: dict = field(default_factory=dict)              # recipe index -> requested vs effective protocol
    rule: object = None                                        # budget.Feasibility used (None: the default)
    pec_max_circuits: int | None = None                        # hardware limit on PEC's sampled circuits (None: none)
    executed: dict = field(default_factory=dict)               # request key -> shots in the circuits built (build_jobs)

    def summary(self) -> dict:
        groups = {}
        for r in self.requests.values():
            groups.setdefault(r.group, 0)
            groups[r.group] += 1
        return dict(circuits=len(self.requests), groups={"%s / %s" % g: c for g, c in groups.items()},
                    shots=int(sum(r.shots for r in self.requests.values())), distinct_circuits=len(self.requests),
                    budget=int(self.shots),
                    effective_budgets={i: int(a.get("effective_budget", self.shots)) for i, a in self.alloc.items()})

    def table(self) -> list:
        """One row per recipe: how its budget is spent on hardware (shots, distinct circuits and protocol)."""
        out = []
        for i, (rc, t) in enumerate(zip(self.recipes, self.labels())):
            a = self.alloc.get(i)
            if a is None:
                out.append({"contender": t, "status": getattr(self, "skipped_idx", {}).get(i, "not planned")})
                continue
            reqs = [r for r in self.requests.values() if r.tag == i]
            pr = self.protocols.get(i) or {}
            pec = (pr.get("effective") or {}) if rc.estimator == "pec" else {}
            out.append({"contender": t, "status": "planned", "circuits": len(reqs),
                        "target shots": a["target"], "training shots": a["training"],
                        "calibration shots": a["calibration"], "total shots": a["target"] + a["training"] + a["calibration"],
                        "declared budget": int(self.shots), "effective budget": int(a.get("effective_budget", self.shots)),
                        "budget note": a.get("budget_note", ""),
                        "shots per PEC circuit": a.get("spc") or None,
                        "PEC shots per circuit (requested)": (pr.get("requested") or {}).get("shots_per_circuit")
                        if rc.estimator == "pec" else None,
                        "PEC sampled circuits": pec.get("sampled_circuits"),
                        "protocol": "differs from the recipe: see the protocol table" if pr.get("differences")
                        else "as configured",
                        "allocation information": (a.get("information") or {}).get("allocation", "")})
        return out

    def protocol_table(self) -> list:
        """Requested (recipe) and effective (executed) protocol parameters side by side, one row per parameter that
        a hardware plan can change, plus every recorded difference. Export this with the results."""
        def txt(x):                                    # one column type (tables are exported as CSV / Arrow)
            return "" if x is None else str(x)
        out = []
        labels = self.labels()
        for i, pr in sorted(self.protocols.items()):
            req, eff = pr.get("requested") or {}, pr.get("effective") or {}
            for k in sorted(set(req) | set(eff)):
                out.append(dict(contender=labels[i], parameter=k, requested=txt(req.get(k)),
                                effective=txt(eff.get(k)), same=req.get(k) == eff.get(k), note=""))
            for d in pr.get("differences") or ():
                out.append(dict(contender=labels[i], parameter="note", requested="", effective="", same=False, note=d))
        return out

    def effective_recipe(self, idx: int):
        """The recipe whose simulation matches what the plan executes for contender ``idx`` (PEC: the effective shots
        per sampled circuit). Use it for transfer predictions; the configured recipe is ``recipes[idx]``."""
        rc = self.recipes[idx]
        eff = (self.protocols.get(idx) or {}).get("effective") or {}
        if rc.estimator == "pec" and eff.get("shots_per_circuit") not in (None, rc.p.get("shots_per_circuit")):
            p = dict(rc.p)
            p["shots_per_circuit"] = int(eff["shots_per_circuit"])
            return Recipe.make(rc.estimator, rc.twirl, rc.dd, rc.readout, **p)
        return rc

    def effective_recipes(self) -> list:
        return [self.effective_recipe(i) for i in range(len(self.recipes))]

    def check_budgets(self, executed: dict | None = None) -> dict:
        """{recipe index: shots} after asserting that every planned contender's requests add up to its effective
        budget (= the declared budget unless the allocation recorded why not) and, with ``executed`` ({request key:
        shots in the circuits built}), that the expansion into twirl / measurement-mask variants kept every shot."""
        out = {}
        for i, a in self.alloc.items():
            keys = [k for k, r in self.requests.items() if r.tag == i]
            planned = int(sum(self.requests[k].shots for k in keys))
            want = int(a.get("effective_budget", self.shots))
            if planned != want:
                raise AssertionError("contender %d: %d shots planned, budget %d" % (i, planned, want))
            if want != int(self.shots) and not a.get("budget_note"):
                raise AssertionError("contender %d: effective budget %d != %d without a reason" % (i, want, self.shots))
            if executed is not None:
                got = int(sum(int(executed.get(k, 0)) for k in keys))
                if got != want:
                    raise AssertionError("contender %d: %d shots in the circuits built, budget %d" % (i, got, want))
            out[i] = planned
        return out

    def resources(self, model=None, native_twirl: bool = True) -> dict:
        """The resources of the plan kept apart (review §5.2): shots (= circuit executions), distinct circuits,
        circuit instances sent (measurement-mask and local twirl variants; with native twirling the primitive's own
        randomizations are listed separately under its documented "auto" rule), Sampler calls and a QPU-time estimate
        (qemx.campaign.estimate_usage; a model, not the provider's billed usage)."""
        from .campaign import estimate_usage
        per, calls = {}, set()
        tot = dict(shots=0, distinct_circuits=0, circuit_instances=0, native_randomizations=0)
        for k, r in self.requests.items():
            sh = int(r.shots)
            if k in self.pubs:
                inst = int(self.pubs[k])
            else:
                v = LOCAL_GATE_TWIRLS if (r.twirl != "none" and not native_twirl) else 1
                inst = min(v * (MEAS_TWIRLS if r.meas_twirl else 1), max(sh, 1))
            nat = 0
            if r.twirl != "none" and native_twirl:      # qiskit-ibm-runtime TwirlingOptions "auto" (both auto)
                spr = max(64, int(math.ceil(sh / 32)))
                nat = int(math.ceil(sh / spr)) * inst
            d = per.setdefault(r.tag, dict(shots=0, distinct_circuits=0, circuit_instances=0, native_randomizations=0))
            for dd in (d, tot):
                dd["shots"] += sh
                dd["distinct_circuits"] += 1
                dd["circuit_instances"] += inst
                dd["native_randomizations"] += nat
            calls.add((r.group, sh))
        est = estimate_usage(tot["shots"], tot["circuit_instances"], max(len(calls), 1), model)
        tot.update(sampler_calls=len(calls), qpu_s=dict(low=est.low_s, central=est.central_s, high=est.high_s),
                   qpu_assumptions=list(est.assumptions))
        return dict(total=tot, per_recipe=per)

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
                     shots=r.shots, circuit=r.spec.name or "", twirl=r.twirl, dd=r.dd, measurement_twirl=r.meas_twirl,
                     requested_times=int(r.multiplicity))
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


ROLE_ORDER = ("target", "training", "calibration")      # a circuit asked for in two roles is listed under the first


def _cal_integer(m, R: int, n_cal: int, nmin: int, smin: int) -> int | None:
    """Integer shots per calibration circuit when R shots remain after training: the integer minimizer of
    v/(R − n_cal·s) + w/s (convex in s) with s ≥ smin and R − n_cal·s ≥ nmin; None when no such s exists."""
    smax = (R - nmin) // n_cal
    if smax < smin:
        return None
    if not (math.isfinite(m.v) and math.isfinite(m.w)) or m.w <= 0:
        return int(smin)
    if m.v <= 0:
        return int(smax)

    def var(s_):
        return m.v / (R - n_cal * s_) + m.w / s_
    k = math.sqrt(m.w / (n_cal * m.v))
    s0 = min(max(R * k / (1 + n_cal * k), smin), smax)
    best = min({min(max(int(math.floor(s0)), smin), smax), min(max(int(math.ceil(s0)), smin), smax)}, key=var)
    for step in (-1, 1):                                   # discrete convexity: unit steps reach the integer optimum
        while smin <= best + step <= smax and var(best + step) < var(best):
            best += step
    return int(best)


def _occurrences(parts) -> tuple:
    """Per role, {request key: (summed share, groups, times asked)} from the recorded log of every measurement
    setting, in first-seen order. Target shares are w_g·a_j (Pauli-sum weight × the estimator's own split);
    training shares are the method's shots per training circuit; calibration keys are counted once."""
    w = _group_weights(parts)
    occ = {x: {} for x in ROLE_ORDER}
    for g, (log, r) in enumerate(parts):
        tgt = [k for role, k in log if role == "target"]
        for k, a in zip(tgt, _shares(r, len(tgt))):
            sh, gs, n = occ["target"].get(k, (0.0, (), 0))
            occ["target"][k] = (sh + float(w[g] * a), tuple(dict.fromkeys(gs + (g,))), n + 1)
        per = (float(r.fixed_shots) / r.train_circuits) if r.train_circuits else 0.0
        for role, k in log:
            if role == "training":
                sh, gs, n = occ["training"].get(k, (0.0, (), 0))
                occ["training"][k] = (sh + per, tuple(dict.fromkeys(gs + (g,))), n + 1)
            elif role == "calibration":                   # one calibration serves every setting: counted once
                sh, gs, n = occ["calibration"].get(k, (1.0, (), 0))
                occ["calibration"][k] = (1.0, tuple(dict.fromkeys(gs + (g,))), n + 1)
    return occ


def _allocate(R, parts, B: int, rule=None, pec: dict | None = None) -> tuple:
    """({request key: integer shots}, allocation summary) for one recipe at budget B, from its recorded cost model.

    Budget-conserving (review F08): integer role budgets first — training F (the method's fixed shots), calibration
    n_cal·s with the integer optimum s ≥ ``rule.cal_min`` near the continuous s*, target N_t = B − F − n_cal·s — then
    each role is split over its *distinct* request keys (a circuit requested twice is measured once; its shares add
    up) with ``budget.integer_split`` (at least one shot each). The shots add up to B exactly. PEC (``pec`` = {setting
    g: s_g}) gets s_g shots for each sampled circuit it recorded, the calibration circuits the rest; without
    calibration circuits the remainder (< s_g per setting) is not spent and the summary records the smaller
    ``effective_budget`` and why. Infeasible budgets (``budget.min_total_budget``, plus one shot per distinct target
    circuit) return (None, {min_budget, reason})."""
    from . import budget as BG
    rule = rule or BG.DEFAULT_FEASIBILITY
    m = BG.CostModel.of(R, rule)
    N, s = BG.split(m, B)
    occ = _occurrences(parts)
    tgt, trn, cal = occ["target"], occ["training"], occ["calibration"]
    n_cal, n_tgt = len(cal), len(tgt)
    F = int(round(m.fixed)) if math.isfinite(m.fixed) else None
    smin = int(rule.cal_min)
    nmin = int(m.n_min)
    if pec is not None:                                    # each setting needs one complete sampled circuit
        need_t = sum(int(pec.get(g, 1)) for g in range(len(parts)))
    else:
        need_t = max(nmin, n_tgt)
    mb = max(float(BG.min_budget(m)), float((F or 0) + n_cal * smin + need_t))

    def infeasible(reason):
        return None, dict(min_budget=mb, reason=reason)
    if F is None:
        return infeasible("the training cost is not finite")
    if N < 1 or B < mb:
        return infeasible("insufficient_budget")
    if trn and F < len(trn):
        return infeasible("the training shots (%d) do not reach every training circuit (%d)" % (F, len(trn)))
    shots_role = {x: {} for x in ROLE_ORDER}
    unspent, notes = 0, []
    # ---- training: F shots over the distinct training circuits (their per-circuit shares, merged repeats added)
    if trn:
        keys = list(trn)
        for k, n_ in zip(keys, BG.integer_split([trn[k][0] for k in keys], F, 1, exact_max=0)):
            shots_role["training"][k] = int(n_)
    elif F > 0:
        unspent += F
        notes.append("%d training shots of the cost model have no training circuit to go to" % F)
    if n_cal and m.c and n_cal != m.c:
        notes.append("%d distinct calibration circuits sent, %d in the cost model" % (n_cal, m.c))
    R_ = B - F
    if pec is None:
        # ---- calibration: an integer s per circuit; target: the rest, split over the distinct target circuits
        if n_cal:
            s_int = _cal_integer(m, R_, n_cal, need_t, smin)
            if s_int is None:
                return infeasible("insufficient_budget")
            for k in cal:
                shots_role["calibration"][k] = s_int
        else:
            s_int = 0
        Nt = R_ - n_cal * s_int
        if n_tgt:
            keys = list(tgt)
            for k, n_ in zip(keys, BG.integer_split([tgt[k][0] for k in keys], Nt, 1)):
                shots_role["target"][k] = int(n_)
        else:
            unspent += Nt
            notes.append("no target circuit: %d target shots not spent" % Nt)
    else:
        # ---- PEC: s_g shots per recorded sampled circuit (identical draws merged: their shots add up)
        for g, (log, r) in enumerate(parts):
            for role, k in log:
                if role == "target":
                    shots_role["target"][k] = shots_role["target"].get(k, 0) + int(pec[g])
        Nt = int(sum(shots_role["target"].values()))
        rest = R_ - Nt
        if rest < n_cal * smin:
            return infeasible("insufficient_budget")
        if n_cal:
            for k, n_ in zip(list(cal), BG.integer_split(np.ones(n_cal), rest, smin, exact_max=0)):
                shots_role["calibration"][k] = int(n_)
        elif rest > 0:
            unspent += rest
            notes.append("PEC sends whole sampled circuits of s shots each: %d target shot%s (fewer than s per "
                         "measurement setting) cannot form another circuit and are not spent" % (rest, "" if rest == 1
                                                                                                 else "s"))
    # ---- merge the roles (a circuit asked for in two roles is measured once, with both shares)
    shots, roles, times = {}, {}, {}
    for role in ROLE_ORDER:
        for k, n_ in shots_role[role].items():
            shots[k] = shots.get(k, 0) + int(n_)
            roles.setdefault(k, role)
            times[k] = times.get(k, 0) + occ[role].get(k, (0, (), 1))[2]
    shared = sum(1 for k in shots if sum(k in shots_role[x] for x in ROLE_ORDER) > 1)
    if shared:
        notes.append("%d circuit%s requested in two roles: measured once with the shots of both" % (
            shared, "" if shared == 1 else "s"))
    tot = {x: int(sum(shots_role[x].values())) for x in ROLE_ORDER}
    eff = int(B) - unspent
    if int(sum(shots.values())) != eff or sum(tot.values()) != eff or min(shots.values(), default=1) < 1:
        raise AssertionError("hardware allocation does not conserve the budget: %d planned, %d expected" % (
            sum(shots.values()), eff))
    s_cal = sorted(set(shots_role["calibration"].values()))
    return shots, dict(N_t=float(N), s=float(s), target=tot["target"], training=tot["training"],
                       calibration=tot["calibration"], cal_circuits=n_cal, cal_circuits_model=int(m.c),
                       s_cal=(s_cal[0] if len(s_cal) == 1 else s_cal) if s_cal else 0, roles=roles, times=times,
                       budget=int(B), effective_budget=eff, budget_note="; ".join(notes) if unspent else "",
                       notes=notes, distinct_circuits=len(shots), shared_keys=shared, min_budget=mb)


def _pec_groups(R, parts, B: int, rule, s_req: int, cap: int | None) -> tuple:
    """(per-setting shots per sampled circuit s_g, sampled circuits K_g, reason) for PEC on hardware.

    The configured protocol is kept: s = ``shots_per_circuit`` and K_g = ⌊N_g/s⌋ sampled circuits, N_g the integer
    target budget of setting g (the shared integer split of :func:`_allocate`). Only a hardware limit ``cap`` on the
    total number of sampled circuits raises s — to the smallest value with Σ_g ⌊N_g/s⌋ ≤ cap (without calibration
    circuits to take the remainder: the value up to twice that which leaves the fewest shots unspent) — which changes
    the protocol (the instance-to-instance variance enters as (s − 1)·Var_inst) and is reported as the reason. Returns
    (None, None, reason) when some setting cannot afford one complete sampled circuit."""
    from . import budget as BG
    rule = rule or BG.DEFAULT_FEASIBILITY
    m = BG.CostModel.of(R, rule)
    occ = _occurrences(parts)
    n_cal = len(occ["calibration"])
    F = int(round(m.fixed)) if math.isfinite(m.fixed) else 0
    G = len(parts)
    if n_cal:
        s_int = _cal_integer(m, B - F, n_cal, max(int(m.n_min), G * s_req), int(rule.cal_min))
        if s_int is None:
            return None, None, "insufficient_budget"
    else:
        s_int = 0
    Nt = B - F - n_cal * s_int
    if Nt < G:
        return None, None, "insufficient_budget"
    Ng = BG.integer_split(_group_weights(parts), Nt, 1)
    s_eff, reason = int(s_req), ""
    if cap is not None and sum(n // s_eff for n in Ng) > int(cap):
        lo, hi = s_eff, max(Ng)
        while lo < hi:                                     # Σ ⌊N_g/s⌋ decreases in s: smallest s that fits
            mid = (lo + hi) // 2
            if sum(n // mid for n in Ng) <= int(cap):
                hi = mid
            else:
                lo = mid + 1
        if not n_cal:
            # nothing takes the remainder: among s in [lo, 2·lo] that fit, the one leaving the fewest shots unspent
            # (ties: the smallest s, the least change of protocol)
            fits = [x for x in range(lo, min(2 * lo, max(Ng)) + 1)
                    if sum(n // x for n in Ng) <= int(cap) and min(n // x for n in Ng) >= 1]
            if fits:
                lo = min(fits, key=lambda x: (sum(n % x for n in Ng), x))
        reason = ("hardware limit of %d PEC sampled circuits: %d shots per sampled circuit instead of the configured "
                  "%d (a different protocol: the instance-to-instance variance enters as (s − 1)·Var_inst)"
                  % (int(cap), lo, s_eff))
        s_eff = lo
    K = [n // s_eff for n in Ng]
    if min(K) < 1:
        return None, None, ("insufficient_budget: PEC with %d shots per sampled circuit needs at least that many "
                            "target shots in every measurement setting (got %s)" % (s_eff, ", ".join(map(str, Ng))))
    return {g: s_eff for g in range(G)}, {g: K[g] for g in range(G)}, reason


def _protocol(rc, gp: dict, info: dict, plan) -> dict:
    """Requested (recipe) and effective (plan) protocol of one contender, with every recorded difference."""
    req = dict(twirl=rc.twirl, dd=rc.dd, readout=rc.readout)
    eff = dict(req)
    diffs = []
    if rc.estimator == "pec":
        s_req = max(int(rc.p.get("shots_per_circuit", 1) or 1), 1)
        req.update(shots_per_circuit=s_req, sampled_circuits="⌊N_t/s⌋ (simulation: N_t/s)",
                   simulated_instances=int(rc.p.get("instances", 64)))
        sg = sorted({int(v["shots_per_circuit"]) for v in gp.values()})
        eff.update(shots_per_circuit=sg[0] if len(sg) == 1 else sg,
                   sampled_circuits=int(sum(int(v["instances"]) for v in gp.values())),
                   simulated_instances="(not used on hardware)")
        if sg != [s_req]:
            diffs.append(info.get("pec_reason") or "shots per sampled circuit %s instead of %d" % (sg, s_req))
    if rc.twirl != "none":
        eff["twirl_randomization"] = ("local rehearsal: %d random instances per circuit, shots split evenly; IBM "
                                      "hardware: the Sampler's own randomizations (TwirlingOptions 'auto')"
                                      % LOCAL_GATE_TWIRLS)
        req["twirl_randomization"] = "simulation: the exact twirled channel (a fresh instance for every shot)"
        diffs.append("gate twirling uses a finite number of random instances per circuit on hardware; the simulation "
                     "averages the twirl exactly, so the hardware estimate has an extra instance-to-instance spread "
                     "that the cost model does not include")
    if rc.readout == "trex":
        req["measurement_masks"] = "simulation: exact symmetrized readout"
        eff["measurement_masks"] = "%d random X masks per circuit (fewer when it has fewer shots)" % MEAS_TWIRLS
    if int(info.get("effective_budget", plan.shots)) != int(plan.shots):
        req["budget"], eff["budget"] = int(plan.shots), int(info["effective_budget"])
        diffs.append("effective budget %d of %d: %s" % (info["effective_budget"], plan.shots, info.get("budget_note")))
    return dict(requested=req, effective=eff, differences=diffs)


def _information(rc, R, plan, pred) -> dict:
    """What the allocation and the method's noise model were allowed to know (review §5.3, §5.5)."""
    from . import budget as BG
    out = dict(allocation=(BG.SUPPLIED + " (predicted from the device calibration snapshot)") if pred is not None
               else "ideal circuit (no noise prediction)")
    reg = (R.details or {}).get("information_regime")
    if reg:
        out["noise_model"] = reg
    if int(getattr(R, "cal_circuits", 0) or 0) > 0:
        out["readout_calibration"] = BG.LEARNED
    if int(getattr(R, "train_circuits", 0) or 0) > 0:
        out["training"] = BG.LEARNED
    return out


def make_plan(compiled: Compiled, observable, recipes: list, shots: int, symmetry=None, seed: int = 0,
              backend=None, predict: bool = True, pec_max_circuits: int | None = None, rule=None) -> Plan:
    """Record every recipe and allocate its budget of ``shots`` (B) exactly as the simulator's accounting does.

    backend: the target BackendV2; its calibration gives PEC's assumed rates, the learned per-gate model and the
    predicted distributions the allocation is computed from.
    pec_max_circuits: an optional hardware limit on PEC's sampled circuits per contender. Without it PEC keeps its
    configured shots per sampled circuit; with it s may be raised (recorded as the effective protocol).
    rule: the budget.Feasibility rule (default: one shot per calibration circuit, ⌈5/P_accept⌉ target shots)."""
    from . import budget as BG
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
    plan.rule = rule or BG.DEFAULT_FEASIBILITY
    plan.pec_max_circuits = None if pec_max_circuits is None else max(int(pec_max_circuits), 1)
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
            gp, pec, pec_reason = {}, None, ""
            R, parts = _record(runner, idx, rc, compiled, observable, plan, gp)
            if rc.estimator == "pec":
                # keep the configured shots per sampled circuit; the number of sampled circuits follows from N_g
                s_req = max(int(rc.p.get("shots_per_circuit", 1) or 1), 1)
                pec, K, pec_reason = _pec_groups(R, parts, B, plan.rule, s_req, plan.pec_max_circuits)
                if pec is not None:
                    gp = {g: dict(shots_per_circuit=pec[g], instances=K[g]) for g in pec}
                    R, parts = _record(runner, idx, rc, compiled, observable, plan, gp)
            if rc.estimator == "pec" and pec is None:
                m0 = BG.CostModel.of(R, plan.rule)
                n_cal = len(_occurrences(parts)["calibration"])
                shots_by_key, info = None, dict(reason=pec_reason, min_budget=max(
                    float(BG.min_budget(m0)), m0.fixed + n_cal * plan.rule.cal_min + len(parts) * m0.per_circuit))
            else:
                shots_by_key, info = _allocate(R, parts, B, plan.rule, pec)
            if shots_by_key is None:
                if str(info.get("reason", "")).startswith("insufficient_budget"):
                    plan.skip(idx, "insufficient_budget: needs at least %s shots" % f"{int(math.ceil(info['min_budget'])):,}")
                else:
                    plan.skip(idx, "%s (needs at least %s shots)" % (info.get("reason"),
                                                                    f"{int(math.ceil(info['min_budget'])):,}"))
                continue
            for k, n_ in shots_by_key.items():
                req = runner.requests[k]
                req.shots, req.role = n_, info["roles"][k]
                req.multiplicity = int(info["times"].get(k, 1))
                plan.requests[k] = req
            info.pop("roles")
            info.pop("times")
            info.update(group_params=gp, spc=(gp.get(0) or {}).get("shots_per_circuit"), pec_reason=pec_reason)
            info["information"] = _information(rc, R, plan, pred)
            info["information_regime"] = BG.SUPPLIED if pred is not None else "ideal-model"
            # the transfer prediction of this plan: the recorded result *is* the simulation of the effective protocol
            # (device-median model, PEC with its effective s and sampled circuits, instance-variance term included)
            m = BG.CostModel.of(R, plan.rule)
            s_c = (info["calibration"] / info["cal_circuits"]) if info["cal_circuits"] else 0.0
            var = m.v / max(info["target"], 1) + (m.w / s_c if (m.w and s_c) else 0.0) + m.u
            info["predicted"] = dict(mean=float(R.mean), var1=float(R.var1), shots_per_circuit=R.shots_per_circuit,
                                     sd=float(math.sqrt(var)) if math.isfinite(var) else math.inf)
            plan.alloc[idx] = info
            plan.protocols[idx] = _protocol(rc, gp, info, plan)
            if plan.protocols[idx]["differences"] and rc.estimator == "pec" and pec_reason:
                plan.notes.append("%s: %s." % (plan.labels()[idx], pec_reason))
            if info["effective_budget"] != B:
                plan.notes.append("%s spends %d of its %d shots: %s." % (plan.labels()[idx], info["effective_budget"],
                                                                         B, info["budget_note"]))
        except NotSupported as e:
            plan.skip(idx, str(e))
        except (MemoryError, KeyboardInterrupt, RunTimeout):
            raise                                        # the time limit / memory stops the whole plan
        except Exception as e:                           # noqa: BLE001 - one recipe must not stop the plan
            plan.skip(idx, "%s: %s" % (type(e).__name__, e))
    plan.check_budgets()
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
    """[(options dict, [(request key, mask, QuantumCircuit, shots)])], one entry per job.

    Every request's shots are split exactly over its variants (measurement masks for TREX, local gate-twirl
    instances); afterwards the shots in the circuits built are checked against each contender's (effective) budget
    (:meth:`Plan.check_budgets`), so the expansion can never lose or add a shot unnoticed."""
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
            total = int(req.shots)
            if total < 1:
                raise ValueError("request %s has no shots assigned: plan it with make_plan" % key)
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
    executed = {}
    for _, pubs, _ in jobs:
        for k, _, _, sh in pubs:
            executed[k] = executed.get(k, 0) + int(sh)
    plan.executed = executed
    plan.check_budgets(executed)
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
    (depolarizing + thermal relaxation over each gate's duration) that ``AerSimulator.from_backend`` builds for the
    whole device and each qubit's asymmetric readout assignment matrix — the device model the predictions use —
    without simulating the noise model of every idle qubit (a 156-qubit model costs seconds per circuit). Idle
    relaxation is added per circuit (:func:`_with_idle_relaxation`). Cached on the DeviceModel."""
    sim = dm.noise.get("__local_sim__")
    if sim is not None:
        return sim
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device.models import basic_device_gate_errors
    from .device import _sub_target
    from qiskit_aer.noise import ReadoutError
    tg = _sub_target(dm)
    nm = NoiseModel(basis_gates=sorted({"rz", "sx", "x", dm.two_qubit, "id"}))
    # each qubit's own assignment matrix, P(1|0) ≠ P(0|1), as in the device model of the predictions (1.5.1, N44:
    # basic_device_readout_errors(target=…) builds a symmetric error from the target's single measure error)
    for q, qi in enumerate(dm.qubits):
        nm.add_readout_error(ReadoutError([[1 - qi.p01, qi.p01], [qi.p10, 1 - qi.p10]]), [q])
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
@qiskit_serial
def _with_idle_relaxation(qc, dm):
    """The compacted circuit scheduled as late as possible with the device durations, every idle window replaced by
    thermal relaxation exp(−t/T1), exp(−t/T2) — the same idle physics as the device model that makes the predictions
    (qemx.device.device_circuit). Without it the local emulation of the hardware path omits idle decoherence and is
    systematically quieter than the predictions (1.5.1, review N44)."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import InstructionDurations, PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
    from .device import relax_instruction
    durs = [(nm, list(qa), int(round(d / dm.dt))) for (nm, qa), d in dm.durations.items()]
    # readout is applied after the circuit (as in the predictions): measurements take no schedule time here
    durs += [("barrier", None, 0), ("measure", None, 0)]
    idur = InstructionDurations(durs, dt=dm.dt)
    sq = PassManager([ALAPScheduleAnalysis(durations=idur), PadDelay(durations=idur)]).run(qc)
    out = QuantumCircuit(*sq.qregs, *sq.cregs)
    for ins in sq.data:
        if ins.operation.name == "delay":
            T = float(ins.operation.duration) * (dm.dt if ins.operation.unit == "dt" else 1.0)
            q = sq.find_bit(ins.qubits[0]).index
            if T > 0:
                out.append(relax_instruction(dm, q, T), [out.qubits[q]])
        else:
            out.append(ins.operation, [out.qubits[sq.find_bit(x).index] for x in ins.qubits],
                       [out.clbits[sq.find_bit(x).index] for x in ins.clbits])
    return out


def run_local(plan: Plan, fake_name: str, seed: int = 0, progress=None, idle_relaxation: bool = True) -> dict:
    """Execute the plan with the Sampler in local testing mode on a fake backend (no account needed). The circuits
    are transpiled for the device exactly as for submission; the Sampler then runs them on Qiskit Aer with the
    device's noise snapshot on the qubits they use: gate errors and readout errors, and (1.5.1, ``idle_relaxation``)
    thermal relaxation in every idle window of the ALAP schedule, as in the device model of the predictions."""
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
        circs = [_compact(c, index) for _, _, c, _ in grp]
        if idle_relaxation:
            circs = [_with_idle_relaxation(c, dm) for c in circs]
        res = sampler.run(circs, shots=shots).result()
        for (k, m, _, _), pr in zip(grp, res):
            results.append((k, m, _pub_counts(pr)))
        if progress:
            progress((i + 1) / len(calls))
    return counts_to_freqs(results, plan.compiled.implemented.n)


class SubmitError(RuntimeError):
    """Submission stopped part-way. ``records`` are the jobs that WERE submitted (they run and are billed), so the
    caller can keep them and collect their results. ``unknown`` lists submissions whose outcome is unknown (the
    provider call failed after it may have created a job): each has ``reservation`` and ``client_tag`` and its
    requests (``pubs``/``runs``); the campaign ledger keeps it committed until it is resolved."""

    def __init__(self, message: str, records: list, unknown=()):
        super().__init__(message)
        self.records = list(records)
        self.unknown = list(unknown)


def open_mode(backend, mode: str) -> tuple:
    """(Batch or None, the mode actually used) for ``mode`` "job" (the default everywhere: every job stands alone)
    or "batch" (one Batch; on the Open plan its 10-minute elapsed-time TTL can fail queued jobs). A Batch that
    cannot be opened (an account or backend without batch mode) falls back to job mode; every record says which
    mode its job used. Shared by the app's and the divergence CLI's submitters."""
    if mode not in ("job", "batch"):
        raise ValueError("mode must be 'job' or 'batch'")
    if mode != "batch":
        return None, "job"
    try:
        from qiskit_ibm_runtime import Batch
        return Batch(backend=backend), "batch"
    except Exception:                                    # noqa: BLE001 - plans without batch mode use job mode
        return None, "job"


def close_mode(batch) -> None:
    if batch is not None:
        try:
            batch.close()                               # no new jobs; the submitted ones still run
        except Exception:                               # noqa: BLE001
            pass


DEFAULT_MAX_CIRCUITS_PER_JOB = 300                 # fake IBM backends report max_experiments = 300


def job_circuit_limit(backend, override: int | None = None) -> int:
    """How many circuits one Sampler job may hold: ``override`` if given, else the backend's ``max_experiments``
    (target or configuration) when it reports one, else 300. Each larger group of pubs is split into several jobs,
    each admitted and recorded on its own (1.5.1, review N32)."""
    if override is not None:
        if int(override) < 1:
            raise ValueError("max_circuits_per_job must be ≥ 1")
        return int(override)
    for get in (lambda b: b.max_experiments, lambda b: b.configuration().max_experiments):
        try:
            v = get(backend)
            if v is not None and int(v) >= 1:
                return int(v)
        except Exception:                                # noqa: BLE001 - backends differ
            continue
    return DEFAULT_MAX_CIRCUITS_PER_JOB


def job_identity(rec: dict) -> tuple:
    """What a job record sends — option group, shot count, (request, mask) pubs — independent of its job id: the key
    that lets a resumed submission of the same plan skip jobs already sent."""
    return (tuple(rec.get("group") or ()), int(rec["shots"]), tuple((str(k), int(m)) for k, m in rec["pubs"]))


def submit(plan: Plan, backend, max_execution_time: int | None = None, seed: int = 0, ledger=None, model=None,
           mode: str = "job", campaign: str = "playground", on_record=None,
           max_circuits_per_job: int | None = None, on_attempt=None, skip=()) -> list:
    """Submit to a real backend (a BackendV2 from a connected service). Returns job records (no secrets).

    One job per option group (twirling / DD) and shot count, in job mode (default) or inside a Batch (``mode``
    "batch", see :func:`open_mode`). **A real provider requires the campaign** ``ledger`` (:mod:`qemx.campaign`;
    a :class:`~qemx.campaign.Ledger` or its path): without one, LedgerRequired is raised before any provider
    contact; only fake backends and local simulators run without it. Nothing is submitted while the ledger holds a
    submission whose outcome is unknown. Every job goes through :func:`qemx.campaign.submit_one`: admitted against
    the campaign's remaining QPU allowance, tagged with its client tag, its id recorded at once; a job that does not
    fit stops the submission, and a provider call that fails without proving that no job exists leaves the
    reservation committed (``submission_unknown``). If a submission stops after others went through, SubmitError
    carries the records of the jobs already submitted (and ``unknown``), so none of them is lost.
    ``on_record(records)`` is called after every job with the records so far; ``on_attempt(record)`` right before
    each provider call with the job's record (no job id yet, ``outcome="pending"``, its reservation and client tag),
    so a caller can persist what is about to be sent. ``skip``: job records (or :func:`job_identity` keys) of this
    same plan that were already sent — a resumed submission sends only the others."""
    from .campaign import (CampaignLimit, SubmissionRejected, SubmissionUnknown, check_no_unresolved,
                           default_max_execution_time, estimate_usage, require_ledger, submit_one, tag_sampler)
    if mode not in ("job", "batch"):
        raise ValueError("mode must be 'job' or 'batch'")
    ledger = require_ledger(ledger, backend)             # a real provider never runs without the campaign ledger
    if ledger is not None:
        check_no_unresolved(ledger)                      # never re-send work whose earlier outcome is unknown
    from qiskit_ibm_runtime import SamplerV2
    jobs = build_jobs(plan, backend, native_twirl=True, native_dd=True, seed=seed)   # everything checked first
    batch, used = open_mode(backend, mode)
    records = []
    limit = job_circuit_limit(backend, max_circuits_per_job)
    done = {x if isinstance(x, tuple) else job_identity(x) for x in (skip or ())}
    try:
        for opts, pubs, group in jobs:
            for shots, grp in ((sh, g[lo:lo + limit]) for sh, g in _by_shots(pubs) for lo in range(0, len(g), limit)):
                if job_identity(dict(group=list(group), shots=int(shots),
                                     pubs=[(k, m) for k, m, _, _ in grp])) in done:
                    continue                             # sent before (a resumed submission of the same plan)
                sampler = SamplerV2(mode=batch if batch is not None else backend)
                if "twirling" in opts:
                    sampler.options.twirling.enable_gates = True
                    sampler.options.twirling.enable_measure = False
                if "dynamical_decoupling" in opts:
                    sampler.options.dynamical_decoupling.enable = True
                    sampler.options.dynamical_decoupling.sequence_type = opts["dynamical_decoupling"]["sequence_type"]
                circs = [c for _, _, c, _ in grp]
                rec = dict(backend=backend.name, group=list(group), shots=int(shots),
                           pubs=[(k, m) for k, m, _, _ in grp], mode=used)
                if ledger is None:                       # a local backend: no QPU time is spent
                    if max_execution_time:
                        sampler.options.max_execution_time = int(max_execution_time)
                    try:
                        job = sampler.run(circs, shots=shots)
                        job_id = job.job_id()
                    except Exception as e:
                        if not records:
                            raise
                        raise SubmitError("%s: %s" % (type(e).__name__, e), records) from e
                    records.append(dict(job_id=job_id, **rec))
                else:
                    est = estimate_usage(int(shots) * len(grp), len(grp), 1, model)
                    met = int(max_execution_time) if max_execution_time else default_max_execution_time(est)

                    def prepare(tag, sampler=sampler, met=met):
                        sampler.options.max_execution_time = int(met)
                        return tag_sampler(sampler, tag)

                    def run(tag, sampler=sampler, circs=circs, shots=shots):
                        return sampler.run(circs, shots=shots)
                    def attempt(att, rec=rec, met=met):
                        if on_attempt is not None:       # on disk before the provider call
                            on_attempt(dict(rec, job_id=None, reservation=att["reservation"],
                                            client_tag=att["client_tag"], max_execution_time=met, outcome="pending"))
                    try:
                        _, job_id, rid, tag = submit_one(
                            ledger, "app: %d circuits × %d shots on %s" % (len(grp), shots, backend.name), est, run,
                            campaign, prepare=prepare, on_attempt=attempt, max_execution_time=met)
                    except CampaignLimit as e:
                        if not records:
                            raise
                        raise SubmitError("campaign ledger: %s" % e, records) from e
                    except SubmissionUnknown as e:
                        raise SubmitError(str(e), records, [dict(rec, job_id=e.job_id, reservation=e.rid,
                                                                 client_tag=e.client_tag,
                                                                 outcome="unknown")]) from e
                    except SubmissionRejected as e:
                        if not records:
                            raise
                        raise SubmitError("%s: %s" % (type(e).__name__, e), records) from e
                    records.append(dict(job_id=job_id, reservation=rid, client_tag=tag, max_execution_time=met,
                                        **rec))
                if on_record is not None:                # keep every job id at once
                    on_record(list(records))
    finally:
        close_mode(batch)
    return records


FINISHED = ("DONE", "JobStatus.DONE")
FAILED = ("ERROR", "CANCELLED", "JobStatus.ERROR", "JobStatus.CANCELLED")


def _reconcile_ended(ledger, job, jid, status=None):
    """Reconcile one ended job into the campaign ledger (qemx.campaign.reconcile_job); the reported usage or None."""
    if ledger is None:
        return None
    from .campaign import PROVIDER_FINAL, provider_state, reconcile_job
    if status is None:
        try:
            status = job.status()
        except Exception:                                # noqa: BLE001 - unknown: leave the ledger as it is
            return None
    if provider_state(status) not in PROVIDER_FINAL:
        return None
    r = reconcile_job(ledger, jid, job, status)
    return None if r is None else r[1]


def collect(service, records: list, n: int, wait: bool = False, cache: dict | None = None, ledger=None) -> tuple:
    """(frequencies or None, [status per job]) for submitted jobs.

    Frequencies are returned once every job has ended — finished, failed or cancelled; the requests of a failed job
    have no counts (their recipes then report it). A request whose pubs were split over several jobs (its
    measurement-twirl masks go out in one Sampler call per shot count) is dropped *entirely* when any of those jobs
    failed — never averaged over the surviving masks — and the failed job's status lists it under
    ``incomplete_requests``. Each status carries ``failed``. ``cache``
    ({job id: [(key, mask, counts)]}) keeps the counts of finished jobs, so checking again downloads only new ones.
    With the campaign ``ledger`` every ended job's reported usage (``job.usage()``) is reconciled into it through
    :func:`qemx.campaign.reconcile_job`, the step the CLIs use (a cached job is checked again only while its usage
    has not been reported)."""
    results, status = [], []
    ended = True
    lost = set()                                        # requests with pubs in a failed job
    cache = {} if cache is None else cache
    if ledger is not None:
        ledger.refresh()
    for rec in records:
        jid = rec["job_id"]
        if jid in cache:
            st = dict(job_id=jid, status="DONE", failed=False)
            results.extend(cache[jid])
            e = ledger.entry_for_job(jid) if ledger is not None else None
            if e is not None and (e["state"] == "submitted" or e.get("usage_s") is None):
                st["usage_s"] = _reconcile_ended(ledger, service.job(jid), jid)
            status.append(st)
            continue
        job = service.job(jid)
        st = str(job.status())
        failed = st in FAILED
        entry = dict(job_id=jid, status=st, failed=failed)
        if failed:
            entry["usage_s"] = _reconcile_ended(ledger, job, jid, st)
            entry["incomplete_requests"] = sorted({k for k, _ in rec["pubs"]})
            lost.update(entry["incomplete_requests"])
            status.append(entry)
            continue
        if not wait and st not in FINISHED:
            ended = False
            status.append(entry)
            continue
        try:
            res = job.result()                          # waits; raises if the job ends in error or is cancelled
        except Exception as e:                          # noqa: BLE001 - this job's requests get no counts
            entry.update(failed=True, status="%s (%s)" % (st, type(e).__name__))
            entry["usage_s"] = _reconcile_ended(ledger, job, jid)     # only if the provider now reports an end
            entry["incomplete_requests"] = sorted({k for k, _ in rec["pubs"]})
            lost.update(entry["incomplete_requests"])
            status.append(entry)
            continue
        got = [(k, m, _pub_counts(pr)) for (k, m), pr in zip(rec["pubs"], res)]
        cache[jid] = got
        results.extend(got)
        entry["usage_s"] = _reconcile_ended(ledger, job, jid, "DONE")
        status.append(entry)
    if not (ended and results):
        return None, status
    freqs = counts_to_freqs([r for r in results if r[0] not in lost], n)   # no partial mask average
    return freqs, status
