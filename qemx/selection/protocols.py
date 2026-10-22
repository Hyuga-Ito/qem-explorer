r"""Complete executable QEM protocols and their budget ledgers (1.6.0).

A *candidate* is a complete procedure (theory companion, Definition 1.3): circuit transformations, training,
calibration, integer shot allocation, post-processing and failure handling, all fixed before any data are seen.
"ZNE" is not a candidate; "linear ZNE at fold scales 1, 3 with global folding, tensored readout correction, shots
split by the supplied model's variances, linear fit" is. Changing any of these makes another candidate.

Information access (``task.INFORMATION``). Allocations use only the *supplied* model — the device's calibration
snapshot simulated classically (information class ``simulated``): never the execution law that generates the data.
PEC cancels the device's median-rate depolarizing model (``DeviceModel.equivalent_noise``), as deployed PEC would
from a calibration record; it is unbiased only when that model is the truth. Learning protocols (CDR, NIL) train on
Clifford-substituted circuits whose ideal values are computed classically (deployable for the near-Clifford training
sets used here).

Budget ledger (Eq. 1.8): B_total = B_calibration + B_training + B_diagnostics + B_selection + B_execution, integer
shots, conserved exactly (``Allocation.check``). A protocol that cannot meet its minima at B is *infeasible* at B,
with the reason recorded; it is never silently given more shots.

Failure handling is part of each protocol (Eq. 1.7) and is applied trial by trial in ``qemx.selection.emulate``:

* a finite calibration whose assignment matrix is singular → that trial uses no readout correction (flag
  ``cal_fallback``);
* exponential ZNE whose fit fails → the linear fit of the same data (``fit_fallback``);
* CDR whose sampled training data resolve no feature direction → the readout-corrected target value
  (``learn_fallback``);
* any trial without an estimate (none expected at feasible budgets) → reported as ``nonresponse``; its loss is the
  declared nonresponse loss μ² (the estimate 0), never dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..budget import CostModel, optimal_integer_split, split as cost_split

SCHEMA = "qemx.protocol/1"
BUDGETS = (1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000)
CAL_MIN = 1                     # shots per readout-calibration circuit at least
TRAIN_MIN = 32                  # shots per training circuit at least (learning protocols)


@dataclass(frozen=True)
class Protocol:
    key: str
    estimator: str
    readout: str = "none"
    params: tuple = ()
    train_fraction: float = 0.0          # learning: share of B spent on training circuits (declared)
    description: str = ""

    def recipe(self):
        from ..methods import Recipe
        return Recipe.make(self.estimator, readout=self.readout, **dict(self.params))

    @property
    def learning(self) -> bool:
        return self.estimator in ("learning", "nil")

    def to_json(self) -> dict:
        return dict(schema=SCHEMA, key=self.key, estimator=self.estimator, readout=self.readout,
                    params=[list(kv) for kv in self.params], train_fraction=self.train_fraction,
                    description=self.description)


def _p(**kw) -> tuple:
    return tuple(sorted(kw.items()))


PORTFOLIO = {
    "RAW": Protocol("RAW", "raw", "none", (), 0.0,
                    "Unmitigated: all B shots on the target circuit; sample mean of the observable."),
    "RO": Protocol("RO", "raw", "tensored", (), 0.0,
                   "Readout correction only: tensored inverse of per-qubit assignment matrices estimated from two "
                   "calibration circuits (|0…0⟩, |1…1⟩); calibration/target split by the supplied model."),
    "ZNE-L": Protocol("ZNE-L", "zne", "tensored", _p(scales=(1.0, 3.0), folding="global", rule="linear",
                                                     lam_est="gates"), 0.0,
                      "Zero-noise extrapolation, global folding at λ = 1, 3, linear fit, tensored readout correction; "
                      "shots per scale ∝ |η_j|σ_j of the supplied model (integer optimal split)."),
    "ZNE-R": Protocol("ZNE-R", "zne", "tensored", _p(scales=(1.0, 3.0, 5.0), folding="global", rule="richardson",
                                                     lam_est="gates"), 0.0,
                      "ZNE, global folding at λ = 1, 3, 5, Richardson (quadratic) extrapolation, tensored readout "
                      "correction; supplied-model allocation."),
    "ZNE-E": Protocol("ZNE-E", "zne", "tensored", _p(scales=(1.0, 3.0, 5.0), folding="global", rule="exponential",
                                                     on_fail="linear", lam_est="gates"), 0.0,
                      "ZNE, global folding at λ = 1, 3, 5, exponential fit with a declared linear fallback when the "
                      "fit fails (counted per trial); tensored readout correction; supplied-model allocation."),
    "PEC": Protocol("PEC", "pec", "tensored", _p(construction="inverse", model="depolarizing", scope="all",
                                                 shots_per_circuit=1), 0.0,
                    "Probabilistic error cancellation of the device's median-rate depolarizing model (from the "
                    "calibration record), one shot per sampled circuit (exact signed law), tensored readout "
                    "correction."),
    "CDR": Protocol("CDR", "learning", "tensored", _p(features="cdr", model="ols", n_train=16, keep=0,
                                                      strategy="adaptive"), 0.25,
                    "Clifford data regression: 16 near-Clifford training circuits with classically computed ideal "
                    "values, OLS fit, 25 % of B on training (equal per circuit); fallback decided from each trial's "
                    "own training data; tensored readout correction."),
    "NIL": Protocol("NIL", "nil", "tensored", _p(neighbors="pauli", positions="all", include_bare=False, model="l1",
                                                 l1_bound=2.0, intercept=False, ensemble="design4",
                                                 target_split="equal", n_train=16), 0.5,
                    "Neighbor-informed learning (Chen et al.): weight-1 Pauli neighbors at every position, "
                    "2-design training ensemble, ℓ1-bounded (≤ 2) regression without intercept, equal target split; "
                    "50 % of B on training; tensored readout correction."),
}


# ------------------------------------------------------------------ allocation
@dataclass
class Allocation:
    protocol: str
    B: int
    feasible: bool
    reason: str = ""
    N: int = 0                          # target (execution) shots in total
    s: int = 0                          # shots per readout-calibration circuit
    c: int = 0                          # readout-calibration circuits
    s_tr: int = 0                       # shots per training circuit
    train_circuits: int = 0
    split: tuple = ()                   # ZNE: shots per fold scale
    ledger: dict = field(default_factory=dict)
    information: str = "supplied"

    def check(self) -> None:
        if not self.feasible:
            return
        tot = sum(int(v) for v in self.ledger.values())
        if tot != int(self.B):
            raise AssertionError("ledger %s sums to %d, not B = %d" % (self.ledger, tot, self.B))
        if self.split and sum(self.split) != self.N:
            raise AssertionError("ZNE split %s does not sum to N = %d" % (self.split, self.N))

    def to_json(self) -> dict:
        return dict(protocol=self.protocol, B=int(self.B), feasible=bool(self.feasible), reason=self.reason,
                    N=int(self.N), s=int(self.s), c=int(self.c), s_tr=int(self.s_tr),
                    train_circuits=int(self.train_circuits), split=[int(x) for x in self.split],
                    ledger={k: int(v) for k, v in self.ledger.items()}, information=self.information)


def _ledger(cal=0, train=0, diag=0, sel=0, exe=0) -> dict:
    return dict(calibration=int(cal), training=int(train), diagnostics=int(diag), selection=int(sel),
                execution=int(exe))


def allocate(protocol: Protocol, supplied, B: int, *, diagnostics: int = 0, selection: int = 0) -> Allocation:
    """Integer allocation of a total budget B for ``protocol`` from the method result of the *supplied* model
    (``supplied``: a MethodResult). ``diagnostics`` and ``selection`` shots charged by a policy are taken off the
    top. Returns an infeasible Allocation (with the reason) when a minimum cannot be met."""
    B = int(B)
    rem = B - int(diagnostics) - int(selection)
    bad = lambda why: Allocation(protocol.key, B, False, why)                       # noqa: E731
    if rem <= 0:
        return bad("diagnostic and selection shots exhaust the budget")
    m = CostModel.of(supplied)
    c = int(m.c)
    s_tr, T = 0, int(getattr(supplied, "train_circuits", 0) or 0)
    train = 0
    if protocol.learning:
        if T <= 0:
            return bad("no training circuits")
        s_tr = int(math.floor(protocol.train_fraction * rem / T))
        if s_tr < TRAIN_MIN:
            return bad("training needs ≥ %d shots on each of %d circuits (%d) — %d available"
                       % (TRAIN_MIN, T, TRAIN_MIN * T, int(protocol.train_fraction * rem)))
        train = s_tr * T
    R = rem - train
    n_min = int(m.n_min)
    if c > 0:
        mm = CostModel(v=m.v, fixed=0.0, c=c, w=m.w, u=0.0, circuits=m.circuits, accept=m.accept,
                       per_circuit=m.per_circuit, rule=m.rule)
        N_f, s_f = cost_split(mm, float(R))
        if N_f <= 0:
            return bad("calibration (%d circuits × ≥ %d shots) and %d target circuits do not fit in %d shots"
                       % (c, CAL_MIN, n_min, R))
        s = max(int(math.floor(s_f)), CAL_MIN)
        N = R - c * s
    else:
        s, N = 0, R
    if N < n_min:
        return bad("the target circuits need ≥ %d shots, %d left" % (n_min, N))
    split = ()
    if protocol.estimator == "zne":
        w = np.asarray(supplied.details.get("split_weights") or [], float)
        k = int(supplied.details.get("n_circuits") or len(w))
        if len(w) != k or k == 0:
            return bad("no supplied split weights")
        if N < k:
            return bad("ZNE needs one shot per fold scale")
        if np.all(w <= 0) or not np.all(np.isfinite(w)):
            base = np.full(k, N // k, dtype=np.int64)
            base[: N - int(base.sum())] += 1
            split = tuple(int(x) for x in base)
        else:
            split = tuple(int(x) for x in optimal_integer_split(np.maximum(w, 1e-300), int(N))[1])
    if protocol.estimator == "nil":
        k = int(supplied.details.get("n_features") or 1)
        if N < k:
            return bad("NIL's equal split needs one shot per neighbor circuit (%d)" % k)
    a = Allocation(protocol.key, B, True, "", int(N), int(s), int(c), int(s_tr), int(T), split,
                   _ledger(c * s, train, diagnostics, selection, N))
    a.check()
    return a
