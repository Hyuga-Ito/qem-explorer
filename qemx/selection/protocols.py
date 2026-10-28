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


DECLARED = "declared"             # a parameter resolved from the task's declared structure (symmetry checks)
POST = {"": "none",
        "clip-observable-range": "the estimate is projected onto the observable's eigenvalue range [min o, max o] "
                                 "(known without the ideal value; never farther from μ, which lies in that range)"}


class NotApplicable(ValueError):
    """A candidate whose preconditions the task does not meet (e.g. symmetry verification without a declared
    symmetry): infeasible for that task, with the reason — not a method failure."""


@dataclass(frozen=True)
class Protocol:
    key: str
    estimator: str
    readout: str = "none"
    params: tuple = ()
    train_fraction: float = 0.0          # learning: share of B spent on training circuits (declared)
    description: str = ""
    twirl: str = "none"                  # 1.8.0: circuit transforms of the recipe (Pauli twirling, DD)
    dd: str = "none"
    post: str = ""                       # 1.8.0: declared post-processing of the estimate (POST)
    crn_with: str = ""                   # 1.8.0: label trials share this candidate's random streams (declared common
    #                                      random numbers, e.g. ZNE-E-B = projection of the very same ZNE-E trials)

    def recipe(self, spec=None):
        """The method recipe. Parameters given as DECLARED are resolved from the task ``spec`` (its declared
        symmetry); without a spec or a declaration the candidate is NotApplicable."""
        from ..methods import Recipe
        prm = dict(self.params)
        if any(v == DECLARED for v in prm.values()):
            sym = None if spec is None else declared_symmetry(spec)
            if sym is None:
                raise NotApplicable("%s needs a symmetry declared by the task's construction (none declared)"
                                    % self.key)
            prm = {k: (sym.get(k) if v == DECLARED else v) for k, v in prm.items()}
            prm = {k: v for k, v in prm.items() if v is not None}
        return Recipe.make(self.estimator, twirl=self.twirl, dd=self.dd, readout=self.readout, **prm)

    @property
    def learning(self) -> bool:
        return self.estimator in ("learning", "nil")

    @property
    def inner(self) -> str:
        """Chained candidates: the inner estimator ('' otherwise)."""
        return str(dict(self.params).get("inner", "")) if self.estimator == "chain" else ""

    @property
    def outer(self) -> str:
        return str(dict(self.params).get("outer", "")) if self.estimator == "chain" else ""

    @property
    def postselects(self) -> bool:
        return self.estimator == "symmetry" or self.inner == "symmetry"

    @property
    def needs_split(self) -> bool:
        return self.estimator == "zne" or self.outer == "zne"

    def to_json(self) -> dict:
        """Identity of the candidate. The 1.8.0 fields appear only when they are not the default, so the eight 1.6/1.7
        candidates keep their exact definitions (and every digest that contains them)."""
        d = dict(schema=SCHEMA, key=self.key, estimator=self.estimator, readout=self.readout,
                 params=[list(kv) for kv in self.params], train_fraction=self.train_fraction,
                 description=self.description)
        if self.twirl != "none":
            d["twirl"] = self.twirl
        if self.dd != "none":
            d["dd"] = self.dd
        if self.post:
            d["post"] = self.post
        if self.crn_with:
            d["crn_with"] = self.crn_with
        return d


def declared_symmetry(spec) -> dict | None:
    """The Z-parity checks a task's *construction* guarantees (1.8.0), as symmetry-verification parameters, or None.
    Declared from the circuit family and input alone — never found from the ideal output:

    * GHZ in the Z basis with its own input: RY(θ) on q0 and a CNOT chain from |0…0⟩ give cos(θ/2)|0…0⟩ +
      sin(θ/2)|1…1⟩, so every adjacent pair has Z_iZ_{i+1} = +1 on the measured bits, for any θ.

    Other families of the study have no declared Z-parity that their measurement basis can check."""
    sym = getattr(spec, "symmetry", None)              # a deployment task carries its declaration (DeploymentTask)
    if isinstance(sym, dict):
        return sym
    if isinstance(sym, tuple) and sym:
        q, ev, more = sym
        return dict(qubits=tuple(q), eigenvalue=int(ev), more=str(more), symmetry_qubits=tuple(q),
                    symmetry_eigenvalue=int(ev), symmetry_more=str(more))
    fam = getattr(spec, "family", None)
    p = dict(getattr(spec, "params", ()) or ())
    if fam == "ghz" and p.get("basis") == "Z" and getattr(spec, "inputs", "default") == "default":
        n = int(spec.n)
        if n < 2:
            return None
        more = "; ".join("%d %d" % (i, i + 1) for i in range(1, n - 1))
        return dict(qubits=(0, 1), eigenvalue=1, more=more, symmetry_qubits=(0, 1), symmetry_eigenvalue=1,
                    symmetry_more=more)
    return None


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
    # ---- 1.8.0: the expanded portfolio (qemx.selection.registry documents every implemented family and option)
    "RO-TREX": Protocol("RO-TREX", "raw", "trex", (), 0.0,
                        "Readout correction by TREX: the measurement is twirled (random X before it, classical flip "
                        "after; one randomization per shot), each Z string divided by its value on one twirled "
                        "|0…0⟩ calibration circuit; calibration/target split by the supplied model."),
    "RO-TW": Protocol("RO-TW", "raw", "tensored", (), 0.0,
                      "Pauli twirling of the two-qubit gates (one random compilation per shot; gate errors become "
                      "their Pauli twirl), no estimator, tensored readout correction.", twirl="2q"),
    "RO-DD": Protocol("RO-DD", "raw", "tensored", (), 0.0,
                      "Dynamical decoupling (XY4 in idle windows; the pulses carry their gate errors), no estimator, "
                      "tensored readout correction.", dd="XY4"),
    "ZNE-L-TW": Protocol("ZNE-L-TW", "zne", "tensored", _p(scales=(1.0, 3.0), folding="global", rule="linear",
                                                           lam_est="gates"), 0.0,
                         "ZNE-L on Pauli-twirled two-qubit gates (one random compilation per shot): global folding "
                         "at λ = 1, 3, linear fit, tensored readout correction, supplied-model split.", twirl="2q"),
    "ZNE-E-B": Protocol("ZNE-E-B", "zne", "tensored", _p(scales=(1.0, 3.0, 5.0), folding="global", rule="exponential",
                                                         on_fail="linear", lam_est="gates"), 0.0,
                        "ZNE-E (λ = 1, 3, 5, exponential fit, linear fallback) followed by a declared projection of "
                        "the estimate onto the observable's eigenvalue range — a separately named bounded estimator "
                        "(review finding F7); ZNE-E itself is unchanged. Labels use ZNE-E's random streams (common "
                        "random numbers), so each ZNE-E-B trial is the projection of the paired ZNE-E trial.",
                        post="clip-observable-range", crn_with="ZNE-E"),
    "PEC-TW": Protocol("PEC-TW", "pec", "tensored", _p(construction="inverse", model="depolarizing", scope="all",
                                                       shots_per_circuit=1), 0.0,
                       "PEC of the median-rate depolarizing model on Pauli-twirled two-qubit gates (twirling makes "
                       "the executed gate errors Pauli channels, closer to the cancelled model); one shot per sampled "
                       "circuit; tensored readout correction.", twirl="2q"),
    "FR": Protocol("FR", "learning", "tensored", _p(features="fr", model="ols", n_train=16, keep=0,
                                                    strategy="adaptive"), 0.25,
                   "Feature regression (this project's CDR extension, not NIL): OLS on ⟨O⟩, every ⟨Z_i⟩ and "
                   "⟨Z_iZ_j⟩ of 16 near-Clifford training circuits, 25 % of B on training, tensored readout "
                   "correction; CDR's adaptive training set and fallback."),
    "SV": Protocol("SV", "symmetry", "tensored", _p(qubits=DECLARED, eigenvalue=DECLARED, more=DECLARED), 0.0,
                   "Symmetry verification: post-selection on the Z-parity checks the task's construction declares "
                   "(never inferred from its ideal output), rejected shots counted in B, a repeat with no accepted "
                   "shot is a nonresponse; tensored readout correction. Applicable only with a declared symmetry."),
    "ZNE-L∘SV": Protocol("ZNE-L∘SV", "chain", "tensored",
                         _p(outer="zne", inner="symmetry", zne_scales=(1.0, 3.0), zne_folding="global",
                            zne_rule="linear", zne_lam_est="gates", symmetry_qubits=DECLARED,
                            symmetry_eigenvalue=DECLARED, symmetry_more=DECLARED), 0.0,
                         "Chained: linear ZNE (λ = 1, 3, global folding) whose every folded circuit is "
                         "symmetry-verified with the task's declared checks (the inner post-selection applies to "
                         "each fold; its rejected shots are paid per fold); one calibration shared by every fold; "
                         "tensored readout correction. Applicable only with a declared symmetry."),
}
PORTFOLIO_16 = ("RAW", "RO", "ZNE-L", "ZNE-R", "ZNE-E", "PEC", "CDR", "NIL")      # 1.6/1.7, immutable
PORTFOLIO_18 = PORTFOLIO_16 + ("RO-TREX", "RO-TW", "RO-DD", "ZNE-L-TW", "ZNE-E-B", "PEC-TW", "FR", "SV", "ZNE-L∘SV")


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
    if protocol.needs_split:
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
