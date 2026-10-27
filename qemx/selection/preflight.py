r"""The shared execution/preflight contract of a candidate protocol at a budget (1.8.0; review finding F1).

1.7.0 decided a deployment candidate's feasibility with a structural shot count (``deploy.structural_feasibility``)
while the study decided it with the protocol's actual allocation (``protocols.allocate`` on the method result of the
supplied model). The two disagreed where a learning protocol's training design is degenerate — every training
circuit has the same ideal value, so the method returns its declared fallback and the study's allocation has no
training circuits (ten accepted deployment choices on two confirmation circuits). This module is the one contract used
by study labeling, the recommender (UI, CLI, batch) and hardware planning. Every answer is one of three statuses:

``verified-feasible``  the protocol's own allocation rule is met at B, established at the stated level;
``infeasible``         it is not (the reason is recorded);
``unverified``         the information needed to decide was not acquired at this level (e.g. a learning protocol's
                       training design has not been constructed). Never reported as feasible.

and carries the *level* of evidence, from cheapest to most complete:

``structural``  compiled circuit, recipe and B only (no simulation of any kind). Exact for the estimators whose
                allocation minima depend only on counts — RAW/readout-only, ZNE, PEC (shown by exhaustive agreement
                with the study's allocations, ``tests/test_preflight.py``). Learning protocols: only *infeasible*
                (Clifford circuit, or minima that fail even for a non-degenerate design) or *unverified*.
                Post-selection (symmetry verification): *unverified* (its minimum target shots depend on the
                acceptance probability).
``design``      additionally constructs the learning protocol's training design exactly as the method does (same
                code, same seed) and computes its training circuits' **ideal** values classically (state-vector
                simulation, 2^n amplitudes — not the 4^n density-matrix engine; no noisy simulation, no shots). This is information a deployment has
                before it runs anything — the protocol must build these circuits anyway — so it is ``available``.
                Its classical cost is reported. The design's seed is part of the result, so execution replays the
                verified design. Width above ``DESIGN_MAX_WIDTH`` → *unverified* (bounded cost).
``allocation``  the study's rule itself: the method run on the supplied model (exact density-matrix simulation,
                ``simulated`` information class) and ``protocols.allocate``. What study labels record.

A diagnostic or preflight that consumed shots would be charged to the ledger (``diagnostics``); none of these levels
uses shots. Classical seconds are reported separately from shots.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import protocols as PR

SCHEMA = "qemx.selection.preflight/1"
VERIFIED, INFEASIBLE, UNVERIFIED = "verified-feasible", "infeasible", "unverified"
STATUSES = (VERIFIED, INFEASIBLE, UNVERIFIED)
LEVELS = ("structural", "design", "allocation")
DESIGN_MAX_WIDTH = 20            # ideal state vector of 2^20 amplitudes (16 MiB) per training circuit
CAL_CIRCUITS = {"none": 0, "tensored": 2, "m3": 2, "trex": 1}


@dataclass
class Preflight:
    protocol: str
    B: int
    status: str
    level: str
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    seconds: float = 0.0             # classical seconds of this check (no shots are used by any level)
    design_seed: int | None = None   # learning protocols at level 'design': the verified design's method seed

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError("unknown preflight status %r" % (self.status,))
        if self.level not in LEVELS:
            raise ValueError("unknown preflight level %r" % (self.level,))

    @property
    def feasible(self) -> bool:
        return self.status == VERIFIED

    def to_json(self) -> dict:
        return dict(schema=SCHEMA, protocol=self.protocol, B=int(self.B), status=self.status, level=self.level,
                    reason=self.reason, evidence=dict(self.evidence), seconds=float(self.seconds),
                    design_seed=None if self.design_seed is None else int(self.design_seed), shots_used=0)


def _protocol(key, portfolio=None):
    if isinstance(key, PR.Protocol):
        return key
    return (portfolio or PR.PORTFOLIO)[key]


def _minima(pr, B: int, *, circuits: int, train_total: int = 0, rule=None) -> tuple:
    """(feasible, reason) of ``protocols.allocate``'s minima with known counts: ``circuits`` target circuits (one shot
    each at least), ``train_total`` training circuits (learning), the recipe's calibration circuits."""
    from ..budget import DEFAULT_FEASIBILITY, min_target_shots
    rule = rule or DEFAULT_FEASIBILITY
    B = int(B)
    rem = B
    if rem <= 0:
        return False, "diagnostic and selection shots exhaust the budget"
    if pr.learning:
        if train_total <= 0:
            return False, "no training circuits"
        s_tr = int(math.floor(pr.train_fraction * rem / train_total))
        if s_tr < PR.TRAIN_MIN:
            return False, ("training needs ≥ %d shots on each of %d circuits (%d) — %d available"
                           % (PR.TRAIN_MIN, train_total, PR.TRAIN_MIN * train_total, int(pr.train_fraction * rem)))
        rem -= s_tr * train_total
    c = CAL_CIRCUITS[pr.readout]
    n_min = min_target_shots(circuits, 1.0, 1, rule)
    if c > 0 and rem < c * rule.cal_min + n_min:
        return False, ("calibration (%d circuits × ≥ %d shots) and %d target circuits do not fit in %d shots"
                       % (c, rule.cal_min, n_min, rem))
    if c == 0 and rem < n_min:
        return False, "the target circuits need ≥ %d shots, %d left" % (n_min, rem)
    return True, ""


def target_circuits(pr, comp) -> int:
    """Target circuits of a non-learning recipe from its structure (ZNE: one per fold scale; a chain with ZNE outside:
    one per fold; RAW, PEC with one shot per sampled circuit, symmetry verification: one)."""
    if pr.estimator == "zne":
        return len(dict(pr.params)["scales"]) * int(dict(pr.params).get("fold_draws", 1) or 1)
    if pr.estimator == "chain" and pr.outer == "zne":
        return len(dict(pr.params).get("zne_scales", (1.0, 3.0, 5.0)))
    return 1


def structural(comp, key, B: int, *, portfolio=None, symmetry=None) -> Preflight:
    """Level 'structural': no simulation. See the module text for what it can and cannot establish. ``symmetry``:
    the task's declared symmetry (``protocols.declared_symmetry``); candidates that need one are infeasible ("not
    applicable") without it."""
    t0 = time.perf_counter()
    pr = _protocol(key, portfolio)
    B = int(B)

    def out(status, reason="", **ev):
        return Preflight(pr.key, B, status, "structural", reason, ev, time.perf_counter() - t0)
    if any(v == PR.DECLARED for v in dict(pr.params).values()) and not symmetry:
        return out(INFEASIBLE, "not applicable: %s needs a symmetry declared by the task's construction" % pr.key)
    if pr.estimator in ("raw", "zne", "pec") or (pr.estimator == "chain" and pr.inner in ("raw", "pec")
                                                  and pr.outer == "zne"):
        k = target_circuits(pr, comp)
        ok, why = _minima(pr, B, circuits=k)
        return out(VERIFIED if ok else INFEASIBLE, why, target_circuits=k, calibration_circuits=CAL_CIRCUITS[pr.readout])
    if pr.postselects:
        ok, why = _minima(pr, B, circuits=target_circuits(pr, comp))
        if not ok:
            return out(INFEASIBLE, why)
        return out(UNVERIFIED, "post-selection: the minimum target shots ⌈k/P_accept⌉ depend on the acceptance "
                               "probability, which needs the noisy model (level 'allocation') or a pilot run")
    if pr.learning:
        from ..transforms import non_clifford_slots
        if not non_clifford_slots(comp.implemented):
            return out(INFEASIBLE, "Clifford circuit: no training circuits (the protocol returns its fallback; the "
                                   "study treats it as infeasible)")
        lo = _learning_train_lower_bound(pr, comp)
        ok, why = _minima(pr, B, circuits=lo["target_circuits"], train_total=lo["train_total"])
        if not ok:
            return out(INFEASIBLE, why + " (even for a non-degenerate design)", **lo)
        return out(UNVERIFIED, "training design not constructed: whether its training labels are degenerate (then "
                               "the protocol has no training circuits) is known only after their ideal values are "
                               "computed (level 'design')", **lo)
    return out(UNVERIFIED, "no structural rule for estimator %r" % pr.estimator)


def _learning_train_lower_bound(pr, comp) -> dict:
    """Training/target circuit counts of a learning recipe when its design is not degenerate."""
    p = dict(pr.params)
    if pr.estimator == "learning":
        return dict(train_total=int(p.get("n_train", 16)), target_circuits=1)
    from ..methods.nil import nil_pool
    if p.get("neighbors", "pauli") == "fold":
        k = len(p.get("scales", (1.0, 3.0, 5.0)))
    else:
        pool = nil_pool(comp.implemented, p.get("positions", "2q"))
        k = min(int(p.get("n_neighbors", 12)), len(pool)) + (1 if p.get("include_bare", True) else 0)
    return dict(train_total=int(p.get("n_train", 16)) * max(k, 1), target_circuits=max(k, 1))


class _IdealContext:
    """What the methods' design code reads from a Context — the target circuit, the method seed, the observable — with
    ``ideal_value`` computed from the ideal *state vector* (2^n amplitudes; ``Context.ideal_value`` uses the
    density-matrix engine, 4^n). Same values to rounding; the design code's decisions use margins of 1e-12 and 1e-9."""

    def __init__(self, comp, o, seed: int):
        self.spec = comp.implemented
        self.n = int(comp.implemented.n)
        self.mm = dict(comp.measure_map or {i: i for i in range(self.n)})
        self.seed = int(seed)
        self._o = np.asarray(o, float)

    @property
    def obs(self):
        return type("Obs", (), dict(o=self._o))()

    def ideal_value(self, spec=None, o=None) -> float:
        from ..engine import permute_probs
        from ..phase.descriptors import _statevector
        psi = _statevector(spec or self.spec)
        p = permute_probs(np.abs(psi) ** 2, self.mm, self.n)
        return float((self._o if o is None else o) @ p)


def design(comp, obs, key, B: int, *, supplied: str, seed: int, portfolio=None, symmetry=None) -> Preflight:
    """Level 'design': construct the learning protocol's training design with the method's own code and seed and
    compute its training circuits' ideal values (no noisy simulation, no shots). Non-learning protocols: the
    structural answer (already exact, or unverified for post-selection)."""
    pr = _protocol(key, portfolio)
    st = structural(comp, pr, B, portfolio=portfolio, symmetry=symmetry)
    if not pr.learning or st.status == INFEASIBLE:
        return st
    t0 = time.perf_counter()
    B = int(B)
    n = int(comp.implemented.n)

    def out(status, reason="", **ev):
        return Preflight(pr.key, B, status, "design", reason, ev, st.seconds + time.perf_counter() - t0, int(seed))
    if n > DESIGN_MAX_WIDTH:
        return Preflight(pr.key, B, UNVERIFIED, "design", "width %d above the design-preflight limit %d (ideal "
                         "training values need 2^n amplitudes)" % (n, DESIGN_MAX_WIDTH), dict(st.evidence),
                         st.seconds, None)
    from ..transforms import non_clifford_slots
    ctx = _IdealContext(comp, obs.o if hasattr(obs, "o") else obs, seed)
    p = dict(pr.params)
    slots = non_clifford_slots(ctx.spec)
    span = float(np.ptp(ctx.obs.o)) or 1.0
    if pr.estimator == "learning":
        from ..methods.learning import choose_training_set
        keep_eff = min(int(p.get("keep", 0)), max(len(slots) - 1, 0))
        need = float(p.get("min_spread", 0.05)) * span
        specs, y, plan, tried = choose_training_set(ctx, slots, int(p.get("n_train", 16)), keep_eff,
                                                    p.get("strategy", "adaptive"), need)
        k = 1
        T = len(specs)
        ev = dict(stages_tried=len(tried), stage=plan.get("text"))
    else:
        from ..methods import REGISTRY
        from ..methods.nil import _nil_seed, nil_training_circuit
        meth = REGISTRY["nil"]
        feats = meth.neighbors(ctx, p.get("neighbors", "pauli"), p.get("positions", "2q"), int(p.get("n_neighbors", 12)),
                               bool(p.get("include_bare", True)), tuple(p.get("scales", (1.0, 3.0, 5.0))),
                               p.get("folding", "global"))
        k = len(feats)
        rng_tr = _nil_seed(ctx.seed, 2)
        T = int(p.get("n_train", 16))
        specs = [nil_training_circuit(ctx.spec, rng_tr, p.get("ensemble", "design4")) for _ in range(T)]
        y = np.array([ctx.ideal_value(sp) for sp in specs], float)
        ev = dict(neighbors=k)
    y = np.asarray(y, float)
    ev.update(training_circuits=int(T), target_circuits=int(k), label_min=float(y.min()), label_max=float(y.max()),
              label_spread=float(np.std(y)), seed=int(seed))
    if np.ptp(y) < 1e-9:
        return out(INFEASIBLE, "no training circuits: every training circuit of this design has the same ideal value "
                               "%.4g (degenerate training labels; the protocol would return its unmitigated fallback)"
                               % float(y[0]), degenerate=True, **ev)
    ok, why = _minima(pr, B, circuits=k, train_total=int(T) * int(k))
    return out(VERIFIED if ok else INFEASIBLE, why, degenerate=False, **ev)


def allocation(supplied_result, key, B: int, *, portfolio=None) -> Preflight:
    """Level 'allocation': the study's rule on the supplied model's method result (``protocols.allocate``)."""
    t0 = time.perf_counter()
    pr = _protocol(key, portfolio)
    if supplied_result is None or isinstance(supplied_result, Exception):
        return Preflight(pr.key, int(B), INFEASIBLE, "allocation", "method failed: %r" % (supplied_result,)[:200])
    a = PR.allocate(pr, supplied_result, int(B))
    return Preflight(pr.key, int(B), VERIFIED if a.feasible else INFEASIBLE, "allocation", a.reason,
                     dict(allocation=a.to_json()), time.perf_counter() - t0)


def from_allocation_json(a: dict) -> Preflight:
    """The preflight a study label record implies (its stored allocation)."""
    st = VERIFIED if a.get("feasible") else INFEASIBLE
    return Preflight(str(a["protocol"]), int(a["B"]), st, "allocation", str(a.get("reason", "")),
                     dict(allocation=dict(a)))


def resolve(comp, obs, key, B: int, *, level: str = "design", supplied: str, seed: int, portfolio=None,
            supplied_result=None, symmetry=None) -> Preflight:
    """The most complete answer allowed by ``level`` (structural ≤ design ≤ allocation)."""
    if level not in LEVELS:
        raise ValueError("level must be one of %s" % ", ".join(LEVELS))
    if level == "structural":
        return structural(comp, key, B, portfolio=portfolio, symmetry=symmetry)
    if level == "allocation":
        if supplied_result is None:
            raise ValueError("level 'allocation' needs the supplied model's method result")
        return allocation(supplied_result, key, B, portfolio=portfolio)
    return design(comp, obs, key, B, supplied=supplied, seed=seed, portfolio=portfolio, symmetry=symmetry)
