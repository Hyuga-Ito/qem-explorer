r"""The shared, versioned candidate registry of the protocol-selection study (1.8.0; prompt section 3).

A *candidate* is a complete executable protocol (``protocols.Protocol``): estimator, parameters, circuit transforms,
readout mitigation, allocation rule, failure handling and post-processing. The registry adds, for every candidate,
its **capability record** — applicability, prerequisites, resources, ledger, failure behavior, information class and
validation status — and an **inventory** of everything QEM Explorer implements (``qemx.methods.REGISTRY``, readout
modes, transforms, compositions), each either mapped to a candidate or listed as unsupported in the selection study
with the reason. The UI, YAML configurations, the command line and the studies resolve candidates by the same key
and the same definition (``protocols.PORTFOLIO``; its JSON is the identity bundles are checked against).

Portfolios (immutable tuples of keys): ``portfolio-1.6`` — the eight candidates of the 1.6/1.7 studies, kept for
comparison; ``portfolio-1.8`` — those plus nine candidates that cover every implemented estimator family with a sound
executable path (virtual distillation excepted; see ``UNSUPPORTED``). A bounded grid: one or two representative
settings per family, not a Cartesian product of every option.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import protocols as PR

SCHEMA = "qemx.selection.registry/1"
PORTFOLIOS = {"portfolio-1.6": PR.PORTFOLIO_16, "portfolio-1.8": PR.PORTFOLIO_18}

# validation status vocabulary (what independent evidence exists for the emulated labels of a candidate)
VALIDATION = {
    "direct": "emulated trials agree with an independently implemented shot-by-shot execution "
              "(emulate.direct_trials / tests/test_emulator_independent.py)",
    "direct-components": "its components are validated by direct execution; the composition is checked for exact "
                         "pairing/identities",
    "reference": "emulated trials agree in distribution with an independently written shot-level reference "
                 "(qemx/selection/reference.py: counts drawn from the exact output laws, the estimator re-implemented), "
                 "including stress cases (tests/test_emulator_independent.py)",
    "direct+reference": "both: direct shot-by-shot execution and the independently written reference",
    "emulated": "trial-level emulation from exact output laws; no independent shot-level reference yet",
}


@dataclass(frozen=True)
class Capability:
    key: str
    family: str                         # estimator family (FAMILY_OF)
    components: tuple                   # (estimator, transforms…, readout, post-processing)
    applies_to: str                     # circuits / observables / devices
    prerequisites: str                  # what must exist before execution
    resources: str                      # circuits, calibration, training, randomizations, extra qubits
    ledger: str                         # how B is split
    failure: str                        # failure handling (declared fallback or nonresponse)
    allocation_information: str         # information class the allocation uses
    validation: str                     # VALIDATION key
    notes: str = ""

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["components"] = list(self.components)
        d["definition"] = PR.PORTFOLIO[self.key].to_json()
        d["validation_meaning"] = VALIDATION[self.validation]
        return d


DIAG = "diagonal observables over the measured bits (|O| ≤ 1); product-state inputs; fake-device calibration snapshots"
SUPPLIED = "simulated (the supplied snapshot model's per-circuit variances split the shots)"
ROLEDGER = "B = c·s (calibration) + N (target); s from the supplied model's calibration-error coefficient"

CAPABILITIES = {c.key: c for c in (
    Capability("RAW", "raw", ("raw",), DIAG, "none", "1 target circuit", "B = N", "none needed",
               "available (no allocation choice)", "direct"),
    Capability("RO", "raw", ("raw", "tensored readout"), DIAG, "2 calibration circuits (|0…0⟩, X on every qubit)",
               "1 target + 2 calibration circuits", ROLEDGER,
               "a singular calibration draw → no readout correction in that trial (cal_fallback)", SUPPLIED, "direct"),
    Capability("ZNE-L", "zne", ("zne linear 1,3 global folding", "tensored readout"), DIAG, "calibration",
               "2 folded target circuits + 2 calibration circuits", ROLEDGER + "; N split over folds ∝ |η_j|σ_j",
               "cal_fallback", SUPPLIED, "direct"),
    Capability("ZNE-R", "zne", ("zne Richardson 1,3,5 global folding", "tensored readout"), DIAG, "calibration",
               "3 folded target circuits + 2 calibration circuits", ROLEDGER + "; N split over folds",
               "cal_fallback", SUPPLIED, "direct"),
    Capability("ZNE-E", "zne", ("zne exponential 1,3,5, linear fallback", "tensored readout"), DIAG, "calibration",
               "3 folded target circuits + 2 calibration circuits", ROLEDGER + "; N split over folds",
               "exponential fit failure → linear fit of the same data (fit_fallback); cal_fallback", SUPPLIED,
               "direct+reference"),
    Capability("PEC", "pec", ("pec inverse of median-rate depolarizing model", "tensored readout"), DIAG,
               "the snapshot's median error rates (a calibration record)",
               "one sampled circuit per shot (instances drawn from the quasi-probability), 2 calibration circuits",
               ROLEDGER, "cal_fallback; noninvertible model → method error (recorded)", SUPPLIED, "direct",
               "unbiased only if the depolarizing model were the truth; the execution law is richer"),
    Capability("CDR", "learning", ("CDR OLS, 16 near-Clifford training circuits", "tensored readout"), DIAG,
               "non-Clifford gates; training circuits whose ideal values vary (classically computed)",
               "16 training circuits, 1 target, 2 calibration circuits", "B = 25 % training (equal per circuit) + "
               "calibration + target", "degenerate training data → readout-corrected target value (learn_fallback)",
               SUPPLIED + "; training design built classically before execution (preflight level 'design')",
               "direct+reference", "ideal training values: state vector (2^n) in this implementation"),
    Capability("NIL", "nil", ("NIL ℓ1 ≤ 2, weight-1 Pauli neighbors at every position", "tensored readout"), DIAG,
               "non-Clifford gates; varying training labels", "16 training circuits × k neighbors, k target "
               "neighbors, 2 calibration circuits", "B = 50 % training + calibration + target (equal split)",
               "degenerate training data → readout-corrected target value (learn_fallback)", SUPPLIED,
               "direct+reference"),
    Capability("RO-TREX", "raw", ("raw", "TREX readout"), DIAG + "; Z-string terms", "1 twirled calibration circuit",
               "1 target circuit with a random X layer before measurement per shot, 1 calibration circuit",
               "B = s (calibration) + N", "singular symmetric calibration → no correction (cal_fallback)", SUPPLIED,
               "reference", "the measurement twirl is emulated as one randomization per shot (the averaged law)"),
    Capability("RO-TW", "raw", ("raw", "Pauli twirl of 2q gates", "tensored readout"), DIAG,
               "a Clifford two-qubit gate set (Pauli frame compensation)",
               "1 target circuit re-randomized per shot (N random compilations), 2 calibration circuits", ROLEDGER,
               "cal_fallback", SUPPLIED, "direct",
               "one randomization per shot = the twirled channel's law; fewer randomizations add variance not "
               "emulated here; compilation of N instances is a classical cost"),
    Capability("RO-DD", "raw", ("raw", "XY4 dynamical decoupling", "tensored readout"), DIAG,
               "idle windows long enough for the sequence", "1 target circuit with DD pulses, 2 calibration "
               "circuits", ROLEDGER, "cal_fallback", SUPPLIED, "direct",
               "the device model has no slow detuning to refocus: DD mainly adds pulse errors here"),
    Capability("ZNE-L-TW", "zne", ("zne linear 1,3", "Pauli twirl of 2q gates", "tensored readout"), DIAG,
               "calibration", "2 folded circuits re-randomized per shot, 2 calibration circuits",
               ROLEDGER + "; N split over folds", "cal_fallback", SUPPLIED, "direct"),
    Capability("ZNE-E-B", "zne", ("zne exponential 1,3,5, linear fallback", "tensored readout",
                                  "projection onto the observable's range"), DIAG,
               "the observable's eigenvalue range (known)", "as ZNE-E", "as ZNE-E",
               "as ZNE-E; then projection (deterministic, never increases a trial's error)", SUPPLIED,
               "reference", "trials paired with ZNE-E's (common random numbers): each is clip(ZNE-E trial)"),
    Capability("PEC-TW", "pec", ("pec inverse depolarizing", "Pauli twirl of 2q gates", "tensored readout"), DIAG,
               "median error rates", "one sampled, twirled circuit per shot; 2 calibration circuits", ROLEDGER,
               "cal_fallback", SUPPLIED, "direct"),
    Capability("FR", "learning", ("feature regression OLS on ⟨O⟩, ⟨Z_i⟩, ⟨Z_iZ_j⟩", "tensored readout"), DIAG,
               "as CDR", "as CDR (16 training circuits)", "as CDR", "as CDR", SUPPLIED, "reference",
               "1 + n + n(n−1)/2 features from 16 training circuits: underdetermined for n ≥ 5 (a known weakness of "
               "this project's heuristic, kept as a candidate, not tuned)"),
    Capability("SV", "symmetry", ("symmetry verification of declared Z-parities", "tensored readout"),
               DIAG + "; tasks with a declared symmetry (GHZ in the Z basis from its own input)",
               "a symmetry known from the construction, checkable in the measurement basis",
               "1 target circuit; rejected shots are spent and counted; 2 calibration circuits",
               ROLEDGER + "; minimum target shots ⌈5/P_accept⌉ (needs the noisy model: preflight 'allocation')",
               "a repeat with no accepted shot → nonresponse (declared loss μ²)", SUPPLIED, "reference",
               "an observable that is a function of the checks is trivially exact after post-selection"),
    Capability("ZNE-L∘SV", "chain", ("zne linear 1,3", "symmetry verification of every fold", "tensored readout"),
               DIAG + "; tasks with a declared symmetry", "as SV", "2 folded circuits, each post-selected; "
               "2 calibration circuits shared by both folds", ROLEDGER + "; N split over folds; ⌈5/P_j⌉ per fold",
               "a fold with no accepted shot → NaN → nonresponse", SUPPLIED, "reference"),
)}

FAMILY_OF = {"raw": "unmitigated / readout only", "zne": "zero-noise extrapolation",
             "pec": "probabilistic error cancellation", "learning": "Clifford data regression / feature regression",
             "nil": "neighbor-informed learning", "symmetry": "symmetry verification", "vd": "virtual distillation",
             "chain": "chained mitigation (outer ∘ inner)"}

# Everything implemented in qemx.methods that the selection portfolios do not contain, with the reason.
UNSUPPORTED = (
    dict(item="vd (virtual distillation, M = 2, 3, 4)", status="blocked",
         reason="the implementation evaluates Tr(ρ^M O)/Tr(ρ^M) with ideal copies and an ideal derangement and samples "
                "a Gaussian with the delta-method variance; a finite-shot multi-copy protocol (M·n qubits, controlled "
                "derangement, numerator/denominator estimation and covariance, noisy copies) is not implemented, so an "
                "equal-budget comparison would be invented. Kept as a simulator-only upper bound in the app."),
    dict(item="readout m3 (M3 subspace solve)", status="not in portfolio",
         reason="supported in the app; in the trial emulator a calibration draw enters through the tensored inverse "
                "(Context._mitigator), so emulated M3 labels would equal RO's law rather than finite-shot M3 — a "
                "faithful label needs per-repeat subspace solves"),
    dict(item="twirl 'all' (randomized compiling of every gate)", status="not in portfolio",
         reason="bounded grid: two-qubit twirling represents the transform; 'all' adds one-qubit frames"),
    dict(item="dd 'X2' (CPMG)", status="not in portfolio", reason="bounded grid: XY4 represents DD"),
    dict(item="zne folding 'local', fractional scales, fold_draws, lam_est 'fault-rate', rule 'poly2', "
              "on_fail 'abstain'", status="not in portfolio",
         reason="bounded grid: three global-folding rules (linear, Richardson, exponential) represent ZNE"),
    dict(item="pec model 'learned', construction 'compensation', scope '2q', shots_per_circuit > 1, instances",
         status="not in portfolio",
         reason="'learned' assumes the true channel is known without a characterization budget (an idealized "
                "information access); the others are settings of the represented construction"),
    dict(item="learning model ridge/lasso, strategy 'fixed', keep > 0", status="not in portfolio",
         reason="bounded grid: CDR (OLS) and FR represent the learning estimator"),
    dict(item="nil neighbors 'fold', positions '2q', other regression models/ensembles/target_split 'optimal'",
         status="not in portfolio",
         reason="bounded grid; 'optimal' target split needs fitted coefficients before the target runs (not "
                "deployable on hardware)"),
    dict(item="chain outer cdr/nil, inner pec/vd; chain outer zne with inner raw/pec", status="not in portfolio",
         reason="ZNE∘SV represents chained mitigation; cdr/nil outer multiplies the T×k training circuits by the "
                "inner cost (compute), inner vd inherits VD's blocker, zne∘raw equals ZNE"),
)


def capability(key: str) -> Capability:
    return CAPABILITIES[key]


def inventory() -> dict:
    """Every implemented estimator, readout mode, transform and composition, mapped to the candidates that use it or
    to an UNSUPPORTED entry. ``tests/test_registry.py`` checks that nothing implemented is missing."""
    from ..methods import DD, READOUT, REGISTRY, TWIRL
    used = {"estimator": {}, "readout": {}, "twirl": {}, "dd": {}, "chain": {}}
    for k in PR.PORTFOLIO_18:
        p = PR.PORTFOLIO[k]
        used["estimator"].setdefault(p.estimator, []).append(k)
        used["readout"].setdefault(p.readout, []).append(k)
        used["twirl"].setdefault(p.twirl, []).append(k)
        used["dd"].setdefault(p.dd, []).append(k)
        if p.estimator == "chain":
            used["chain"].setdefault("%s∘%s" % (p.outer, p.inner), []).append(k)
    unsup = " ".join(u["item"] for u in UNSUPPORTED)
    rows = []
    for kind, names in (("estimator", list(REGISTRY)), ("readout", list(READOUT)), ("twirl", list(TWIRL)),
                        ("dd", list(DD))):
        for nm in names:
            cands = used[kind].get(nm, [])
            covered = bool(cands) or (nm in unsup)
            rows.append(dict(kind=kind, name=nm, candidates=cands, covered=covered,
                             status="in portfolio-1.8" if cands else "see UNSUPPORTED"))
    return dict(schema=SCHEMA, portfolios={k: list(v) for k, v in PORTFOLIOS.items()}, rows=rows,
                unsupported=list(UNSUPPORTED), chains=used["chain"])


def describe(key: str) -> dict:
    """Capability record of a candidate as plain data (UI, CLI, documentation)."""
    return capability(key).to_json()


def markdown() -> str:
    """docs/selection/CANDIDATES.md (generated: python -m qemx.selection registry --markdown)."""
    inv = inventory()
    L = ["# Candidate registry (%s)" % SCHEMA, "",
         "Generated from `qemx/selection/registry.py`; do not edit by hand.", "",
         "Portfolios: " + "; ".join("`%s` (%d candidates)" % (k, len(v)) for k, v in PORTFOLIOS.items()) + ".", "",
         "| Candidate | Family | Components | Applies to | Resources | Ledger | Failure handling | Validation |",
         "|---|---|---|---|---|---|---|---|"]
    for k in PR.PORTFOLIO_18:
        c = CAPABILITIES[k]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            k, FAMILY_OF[c.family], "; ".join(c.components), c.applies_to, c.resources, c.ledger, c.failure,
            c.validation))
    L += ["", "## Definitions (the identity bundles are checked against)", ""]
    for k in PR.PORTFOLIO_18:
        p = PR.PORTFOLIO[k]
        L.append("* **%s** — %s" % (k, p.description))
    L += ["", "## Implemented but not in the selection portfolios", "", "| Item | Status | Reason |", "|---|---|---|"]
    for u in UNSUPPORTED:
        L.append("| %s | %s | %s |" % (u["item"], u["status"], u["reason"]))
    L += ["", "## Inventory", "", "| Kind | Implemented | Candidates |", "|---|---|---|"]
    for r in inv["rows"]:
        L.append("| %s | %s | %s |" % (r["kind"], r["name"], ", ".join(r["candidates"]) or r["status"]))
    return "\n".join(L) + "\n"
