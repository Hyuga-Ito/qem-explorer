"""One vocabulary for what a number in the selection study does and does not guarantee (1.7.0; review finding R5).

Every table, figure note, certificate and model bundle that reports an interval, a bound or a coverage names its
*statement kind*, so a measured coverage is never read as a guarantee, a post-hoc check never as a certificate, and a
bound on a clipped estimator never as a bound on the unclipped one.
"""
from __future__ import annotations

KINDS = {
    "empirical": (
        "Measured on the held-out base circuits of this study (a finite sample of one task distribution). "
        "No guarantee for other circuits, devices, budgets or distributions."),
    "cv-residual": (
        "Interval from inner cross-validation residuals placed around the model refitted on all training rows (the CV "
        "construction). No finite-sample coverage guarantee (CV+ would guarantee 1 − 2α under exchangeability; Barber "
        "et al. 2021). Its coverage is measured, never assumed."),
    "split-conformal": (
        "Split-conformal interval with base circuits as the exchangeable units: a calibration set of base circuits "
        "never used for fitting, one score per circuit (the largest residual over its rows), the ⌈(1 − α)(G + 1)⌉-th "
        "smallest score. Finite-sample marginal coverage ≥ 1 − α for all rows of a new base circuit drawn from the same "
        "distribution as the calibration circuits. Assumption: exchangeability of base circuits (true by construction "
        "of a random split; not true for a new family, device or budget range). With G < 1/α − 1 calibration circuits "
        "the interval is unbounded."),
    "finite-sample-bound": (
        "Hoeffding or empirical-Bernstein bound with a union bound over the candidates, from independent units of the "
        "*clipped* estimator Π[−1,1](μ̂) (loss ≤ 4): independent trials for protocols without a training design, "
        "independent design means for learning protocols under fresh designs. Valid for the clipped estimator only; "
        "the unclipped estimator's risk is reported beside it without a bound."),
    "model-conditional": (
        "A statement about the fitted model's predictions (a predicted risk, a margin between predicted risks): it "
        "holds only to the extent that the model is right for the task."),
    "post-hoc-check": (
        "A relation checked after the fact with benchmark risks that are unknown at deployment (e.g. the 2ε regret "
        "bound with ε measured from the labels). It describes this study; it is not an operational certificate."),
    "ood-warning": (
        "The task lies outside the training support of the model (a feature outside the training range, an unseen "
        "family, device or width, a budget beyond the training range). No statement is made."),
    "descriptive": (
        "A point estimate or pointwise comparison (e.g. the protocol with the lowest measured risk in one map cell, "
        "'unresolved' when the top two lie within 1.96 approximate standard errors). Not a simultaneous statement over "
        "candidates or cells."),
}


def describe(kind: str) -> str:
    if kind not in KINDS:
        raise KeyError("unknown statement kind %r (choose from %s)" % (kind, ", ".join(KINDS)))
    return KINDS[kind]


def markdown_table() -> str:
    lines = ["| statement kind | meaning |", "|---|---|"]
    for k, v in KINDS.items():
        lines.append("| `%s` | %s |" % (k, v))
    return "\n".join(lines)
