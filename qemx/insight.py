r"""The analysis engine: decide how a result should be looked at, and compute what that view needs.

Every experiment in the app ends in one of three mathematical shapes, and each has a natural picture:

* **a probability** of some outcomes (the observable is a projector, values in {0, 1}: Shor's success, Grover's
  target, Bernstein–Vazirani's secret) — the object is the whole *output distribution*; the picture is a histogram of
  bit strings, ideal against noisy and mitigated, with the counted outcomes marked;
* **an expectation value** of ±1 parities or a Hamiltonian (VQE, Ising, GHZ parities) — the object is a scalar with
  an error bar; the picture is estimates ± σ and the spread of repeated experiments;
* **a logical observable of an error-correcting code** — the objects are the logical error and the syndromes; the
  picture is the effective logical error per method and the syndrome histogram.

Independently, each mitigation method has its own *diagnostic curve* (ZNE: E(λ) and the extrapolation; PEC: the
quasi-probabilities and γ; learning: the training fit; VD: the spectrum of ρ; symmetry verification: the accepted
fraction; readout: the assignment matrix). ``profile`` picks the primary view and says why; the page offers the same
choices as a manual override.

Distributions are exact (density-matrix simulation): ideal, noisy (with readout error), readout-corrected (the
calibrated inverse applied to the noisy distribution, a quasi-probability) and ZNE-extrapolated (the linear
extrapolation weights η applied to every outcome's probability, :math:`p_0(x)=\sum_j\eta_j\,p_{\lambda_j}(x)`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VIEWS = {
    "expectation": "Expectation values ± error bars",
    "distribution": "Output distribution (histogram)",
    "qec": "Logical error & syndromes",
    "diagnostics": "Method diagnostics",
}
QEC_TEMPLATES = ("qec_3q", "repetition", "steane", "five_qubit", "surface_d2", "qec_algorithm")
DIAGNOSTIC = {"zne": "extrapolation E(λ) → λ = 0", "pec": "quasi-probabilities and γ", "learning": "training fit",
              "vd": "spectrum of ρ", "symmetry": "accepted fraction", "readout": "assignment matrix"}
MAX_DIST_QUBITS = 16                  # distributions over more bits are summarized, not drawn


@dataclass
class Profile:
    primary: str                       # a key of VIEWS
    order: list                        # every view, recommended first
    reasons: list = field(default_factory=list)
    observable_kind: str = "general"   # projector | parity | hamiltonian | general
    checks: list = field(default_factory=list)        # QEC syndrome checks: (label, clbits)
    custom: bool = False


def observable_kind(obs) -> str:
    if hasattr(obs, "terms"):                          # a PauliSum
        return "hamiltonian"
    return value_kind(getattr(obs, "values", ()))


def value_kind(values) -> str:
    """projector (values 0/1), parity (±1) or general, from an observable's values over outcomes."""
    v = np.asarray(values, float)
    if v.size and np.all((np.abs(v) < 1e-12) | (np.abs(v - 1) < 1e-12)):
        return "projector"
    if v.size and np.all(np.abs(np.abs(v) - 1) < 1e-12):
        return "parity"
    return "general"


def qec_checks(key: str | None, params: dict | None, n: int) -> list:
    """Syndrome checks of a QEC template as (label, classical bits whose parity is the check)."""
    p = params or {}
    if key == "qec_algorithm":                     # encoded algorithms carry their own checks (qemx/qec.py)
        from . import templates as T
        return T.qec_checks(key, p)
    if key == "qec_3q":
        if p.get("mode", "syndrome + correction") == "majority vote readout":
            return [("Z0Z1", (0, 1)), ("Z1Z2", (1, 2))]
        return [("Z0Z1 (q3)", (3,)), ("Z1Z2 (q4)", (4,))]
    if key == "repetition":
        d = int(p.get("d", 3))
        return [("Z%dZ%d" % (i, i + 1), (i, i + 1)) for i in range(d - 1)]
    if key == "steane":
        return [("Z3Z4Z5Z6", (3, 4, 5, 6)), ("Z1Z2Z5Z6", (1, 2, 5, 6)), ("Z0Z2Z4Z6", (0, 2, 4, 6))]
    if key == "five_qubit":
        return [("S%d (q%d)" % (j, j), (j,)) for j in range(1, 5)]
    if key == "surface_d2":
        if str(p.get("syndrome", "")).startswith("ancilla"):
            return [("X plaquette (q4)", (4,)), ("Z0Z1 (q5)", (5,)), ("Z2Z3 (q6)", (6,))]
        return [("Z0Z1", (0, 1)), ("Z2Z3", (2, 3))]
    return []


def profile(template: str | None, params: dict | None, custom: bool, observable, estimators, n: int,
            logical: bool = True) -> Profile:
    """The recommended views for this circuit, observable and set of methods, with the reasons. ``logical``: the
    observable is the code's own logical observable, read in the template's bit order (else no QEC view)."""
    kind = observable_kind(observable)
    reasons = []
    checks = qec_checks(template, params, n) if (template in QEC_TEMPLATES and not custom and logical) else []
    if checks:
        primary = "qec"
        reasons.append("an error-correcting code: the logical observable and its syndromes tell the story")
    elif kind == "projector":
        primary = "distribution"
        reasons.append("the observable is a probability (values 0 and 1), so the whole output distribution matters")
    elif kind == "hamiltonian":
        primary = "expectation"
        reasons.append("the observable is a Hamiltonian ⟨H⟩ = Σ c⟨P⟩: a scalar with an error bar")
    else:
        primary = "expectation"
        reasons.append("the observable is an expectation value of ±1 outcomes: a scalar with an error bar"
                       if kind == "parity" else "the observable is a weighted sum over outcomes: a scalar estimate")
    diag = [e for e in dict.fromkeys(estimators) if e in DIAGNOSTIC]
    order = [primary] + [v for v in ("distribution", "expectation", "diagnostics", "qec") if v != primary]
    if not checks:
        order.remove("qec")
    if diag:
        order.remove("diagnostics")
        order.insert(1, "diagnostics")
        reasons.append("method diagnostics: " + "; ".join("%s → %s" % (e, DIAGNOSTIC[e]) for e in diag))
    if custom:
        reasons.append("custom circuit: the choice follows the observable — pick another view if it fits better")
    return Profile(primary, order, reasons, kind, checks, custom)


# ------------------------------------------------------------------ exact distributions
def distributions(compiled, noise, outs, runner, backend: str = "numpy") -> dict:
    """{'series': {name: probabilities over classical bits}, 'notes': [...], 'readout': calibrated [A_c]} — exact,
    from the same (memoized) simulations and the same calibration as the comparison's estimators, so the observable
    summed over a series reproduces the matching estimate."""
    from .methods.base import Context, Runner
    from .methods.zne import linear_weights, richardson_weights
    from .transforms import achieved_lambda, fold_global, fold_local
    n = compiled.logical.n
    if n > MAX_DIST_QUBITS:
        return dict(series={}, notes=["%d classical bits: the distribution has %s outcomes and is not drawn." % (
            n, f"{2 ** n:,}")], readout=[])
    runner = runner or Runner(backend, layout=compiled.info.get("phys"))
    mm = compiled.measure_map
    spec = compiled.implemented
    ideal = runner.run(spec, noise, mm, ideal=True)
    noisy = runner.run(spec, noise, mm)
    series = {"ideal": np.asarray(ideal.probs, float), "noisy (unmitigated)": np.asarray(noisy.probs_meas, float)}
    notes, mats = [], []
    done = [o for o in (outs or []) if o.result is not None]
    ro = next((o.recipe for o in done if o.recipe.readout != "none"), None)
    if ro is not None:
        ctx = Context(compiled, noise, None, ro, runner)
        mats = [np.asarray(A, float) for A in ctx.calibration()["A"]]
        label = {"tensored": "tensored inverse", "m3": "M3", "trex": "TREX"}[ro.readout]
        series["readout-corrected"] = np.asarray(ctx.mitigate(ctx.measured(noisy)), float)
        notes.append("Readout-corrected (%s, calibrated like the estimator): a quasi-probability, so small "
                     "negative values are possible and meaningful." % label)
    zne = next((o for o in done if o.recipe.estimator == "zne" and o.recipe.p.get("rule") in ("richardson", "linear")),
               None)
    if zne is not None:
        p = zne.recipe.p
        ctx = Context(compiled, noise, None, zne.recipe, runner)
        ps, lams = [], []
        for lam in sorted(float(s) for s in p.get("scales", (1, 3, 5))):
            fs = fold_global(spec, lam) if p.get("folding", "global") == "global" else fold_local(spec, lam, 0)
            rr = ctx.run(fs)
            ps.append(np.asarray(ctx.mitigate(ctx.measured(rr)), float))
            lams.append(achieved_lambda(spec, fs))
        if len(set(np.round(lams, 9))) == len(lams):
            eta = richardson_weights(lams) if p["rule"] == "richardson" else linear_weights(lams)
            series["ZNE-extrapolated"] = np.tensordot(eta, np.stack(ps), axes=1)
            notes.append("ZNE-extrapolated (%s, λ = %s): the weights η = %s applied to every outcome's "
                         "probability — the same linear combination as the estimator." % (
                             p["rule"], ", ".join("%.3g" % x for x in lams), ", ".join("%+.3f" % e for e in eta)))
    elif any(o.recipe.estimator == "zne" for o in done):
        notes.append("The exponential ZNE rule is not linear in the probabilities, so it has no distribution-level "
                     "version; its estimate is in the other views.")
    if any(o.recipe.estimator == "pec" for o in done):
        notes.append("PEC is not drawn as a distribution: it corrects expectation values through a signed sum over "
                     "sampled circuits (see the expectation and diagnostics views).")
    return dict(series=series, notes=notes, readout=[A.tolist() for A in mats])


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p, float) - np.asarray(q, float))))


def hellinger_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 0, None)
    q = np.clip(np.asarray(q, float), 0, None)
    p, q = p / max(p.sum(), 1e-300), q / max(q.sum(), 1e-300)
    return float(np.sum(np.sqrt(p * q)) ** 2)


def syndrome_distribution(p: np.ndarray, checks: list, n: int) -> np.ndarray:
    """P(syndrome = s) with s = (parity of check 0, parity of check 1, …) read as a binary number, check 0 first
    (the most significant digit of the label)."""
    k = len(checks)
    x = np.arange(2 ** n)
    s = np.zeros(2 ** n, dtype=int)
    for i, (_, bits) in enumerate(checks):
        par = np.zeros(2 ** n, dtype=int)
        for b in bits:
            par ^= (x >> b) & 1
        s = s | (par << (k - 1 - i))
    out = np.zeros(2 ** k)
    np.add.at(out, s, np.asarray(p, float))
    return out


def intended_logical(template: str | None, params: dict | None) -> float | None:
    """The logical value the code is meant to protect, ⟨Z_L⟩ = cos θ of the encoded RY(θ)|0⟩ (injected errors are part
    of what the code has to handle, so they do not change the reference)."""
    if template not in QEC_TEMPLATES:
        return None
    from . import templates as T
    if template in T.QEC_ALGORITHM_KEYS:           # the error-free value of the decoded logical observable
        return T.intended_logical(template, params)
    theta = float({**T.defaults(template), **(params or {})}.get("theta", 0.0))
    return float(np.cos(theta))


def logical_error(estimate: float, ideal: float) -> float:
    """Effective logical flip probability from ⟨Z_L⟩ = (1 − 2 p_L) ⟨Z_L⟩_ideal (NaN when the ideal value is ~0)."""
    if not (np.isfinite(estimate) and np.isfinite(ideal)) or abs(ideal) < 0.05:
        return float("nan")
    return float((1 - estimate / ideal) / 2)


def relevant_bits(values, n: int) -> list:
    """Classical bits the observable actually reads: flipping any other bit never changes its value."""
    v = np.asarray(values, float)
    if v.size != 2 ** n:
        return list(range(n))
    x = np.arange(2 ** n)
    return [b for b in range(n) if not np.allclose(v, v[x ^ (1 << b)])]


def marginal(p, bits: list, n: int) -> np.ndarray:
    """Distribution over the listed bits; outcome index = Σ bit_k 2^k in the order given (bits[0] least significant)."""
    x = np.arange(2 ** n)
    idx = np.zeros(2 ** n, dtype=int)
    for k, b in enumerate(bits):
        idx |= ((x >> b) & 1) << k
    out = np.zeros(2 ** len(bits))
    np.add.at(out, idx, np.asarray(p, float))
    return out
