r"""Symmetry verification by post-selection (Bonet-Monroig, Sagastizabal, Singh & O'Brien,
PRA 98, 062339 (2018); McArdle, Yuan & Benjamin, PRL 122, 180501 (2019)).

If the ideal state is an eigenstate of a symmetry :math:`S` (here a Z-parity of chosen qubits)
with eigenvalue :math:`s`, any shot that violates it must contain an error: discard it.

.. math:: \hat O=\frac{\sum_{x:\,S(x)=s}\hat p(x)\,o(x)}{\sum_{x:\,S(x)=s}\hat p(x)},\qquad
          \text{cost}\ \propto\ 1/P_{\rm accept}

Errors that commute with S (e.g. two bit flips on the checked pair, or phase errors) pass the check.
Several checks can be verified together (``more``: further Z-parities that must all be +1, e.g. every Z check of a
code), which is error detection with all of them.

Statistics. The per-shot variance is the delta method for the ratio of two means, Var = Var_acc(o)/(P_accept·N)
without readout mitigation. It is first order: it holds once N·P_accept ≫ 1. With few target shots a repeat may keep
no shot at all (probability (1 − P_accept)^N, a NaN estimate counted as failed) and the spread exceeds var1/N, so the
cost model (qemx/budget.py) needs at least ⌈5/P_accept⌉ target shots, and the comparison table reports the spread of
the repeats when it clearly exceeds the analytic value.
"""
from __future__ import annotations

import math

import numpy as np

from .base import MethodResult, Param, QEMMethod, register
from .raw import cal_fields, display_circuit
from ..circuit import Op


MAX_CHECK_TEXT = 200


def parse_checks(text) -> list:
    """'2 3; 5 6' -> [(2, 3), (5, 6)]: further parity checks (each must be +1). Strict: digits, spaces, commas and
    semicolons only."""
    import re
    t = str(text or "").strip()
    if not t:
        return []
    if len(t) > MAX_CHECK_TEXT or re.search(r"[^\d\s,;]", t):
        raise ValueError("further parities look like '2 3; 5 6' (qubit numbers, groups separated by ';')")
    out = []
    for grp in t.split(";"):
        qs = tuple(int(x) for x in re.split(r"[\s,]+", grp.strip()) if x)
        if qs:
            if any(len(str(q)) > 4 for q in qs):
                raise ValueError("qubit numbers are too large")
            out.append(qs)
    return out


def parity_mask(n: int, qubits, value: int) -> np.ndarray:
    x = np.arange(2 ** n)
    par = np.zeros(2 ** n, dtype=int)
    for q in qubits:
        par ^= (x >> q) & 1
    return (1 - 2 * par) == value


@register
class Symmetry(QEMMethod):
    key = "symmetry"
    label = "Symmetry verification"
    family = "symmetry"
    tex = (r"\hat O=\frac{\sum_{x:\,S(x)=s}\hat p(x)\,o(x)}{\sum_{x:\,S(x)=s}\hat p(x)},\qquad "
           r"\mathrm{Var}\propto\frac{1}{P_{\rm accept}}")
    why = (r"A symmetry of the ideal state flags errors for free: a shot with the wrong parity cannot come from "
           r"the ideal state, so it is discarded. Errors that commute with the symmetry pass unnoticed, and every "
           r"discarded shot is paid for: the cost grows as $1/P_{\rm accept}$.")
    params = (
        Param("qubits", "parity of qubits", "ints", (0, 1)),
        Param("eigenvalue", "ideal eigenvalue", "choice", 1, (1, -1)),
        Param("more", "further parities, each +1 (e.g. 2 3; 5 6)", "text", "",
              help="More Z-parity checks that must all hold, separated by ';' — e.g. every Z check of a code. "
                   "Leave empty for a single symmetry."),
    )

    def short(self, p):
        base = "Symmetry Z%s=%+d" % ("Z".join(str(q) for q in p["qubits"]), p["eigenvalue"])
        try:
            more = parse_checks(p.get("more", ""))
        except ValueError:
            more = []
        return base + "".join(" · Z%s=+1" % "Z".join(map(str, g)) for g in more)

    def run(self, ctx, qubits=(0, 1), eigenvalue=1, more="") -> MethodResult:
        qubits = [int(q) for q in qubits if 0 <= int(q) < ctx.n]
        if not qubits:
            raise ValueError("choose at least one qubit for the parity check")
        groups = parse_checks(more)
        if any(q >= ctx.n for g in groups for q in g):
            raise ValueError("a further parity uses a qubit the circuit does not have")
        rr = ctx.run()
        # a symmetry is a property of the qubits; with relabeled readout qubit q is read on classical bit n−1−q
        relabel = bool((ctx.compiled.info or {}).get("relabel"))
        bit = (lambda q: ctx.n - 1 - q) if relabel else (lambda q: q)
        mask = parity_mask(ctx.n, [bit(q) for q in qubits], int(eigenvalue))
        for g in groups:
            mask &= parity_mask(ctx.n, [bit(q) for q in g], 1)
        f = ctx.mitigate(ctx.measured(rr))
        o = ctx.obs.o
        p_meas = np.clip(ctx.measured(rr), 0, None)
        acc_meas = float(p_meas[mask].sum())
        acc = float(f[mask].sum())
        mean = float((f * mask) @ o / acc) if acc > 0 else float("nan")
        # ratio estimator R = n/d with per-shot values a = Mᵀ(mask·o) and b = Mᵀ mask (readout mitigation folded in):
        # delta method, Var(R) = [Var(a) − 2R Cov(a,b) + R² Var(b)] / d² per shot (exact to first order)
        a, b = ctx.o_eff(mask * o), ctx.o_eff(mask.astype(float))
        ea, eb = p_meas @ a, p_meas @ b
        va, vb, cab = p_meas @ a ** 2 - ea ** 2, p_meas @ b ** 2 - eb ** 2, p_meas @ (a * b) - ea * eb
        R = ea / eb if eb > 0 else float("nan")
        var1 = float(max(va - 2 * R * cab + R * R * vb, 0.0) / max(eb, 1e-12) ** 2) if eb > 0 else float("inf")
        notes = []
        ideal_ok = ctx.ideal_value(o=mask.astype(float))
        if ideal_ok < 1 - 1e-9:
            notes.append("The ideal state is not an eigenstate of this symmetry (P = %.3f): post-selection "
                         "biases the result." % max(ideal_ok, 0.0))
        if 0 < acc_meas < 1 - 1e-12:
            from ..budget import MIN_ACCEPTED
            n_min = int(math.ceil(min(MIN_ACCEPTED / acc_meas, 1e18) - 1e-9))
            notes.append(
                "Post-selection keeps P_accept = %.3g of the shots. The variance var1/N is first order (delta method "
                "for a ratio): it holds once N·P_accept ≫ 1. With fewer target shots some repeats keep no shot at all "
                "(probability (1 − P_accept)^N; they are counted as failed) and the spread of the others exceeds "
                "var1/N. The cost model therefore needs at least ⌈%d/P_accept⌉ = %d target shots (%d accepted shots "
                "expected)." % (acc_meas, MIN_ACCEPTED, n_min, MIN_ACCEPTED))

        linear_ro = ctx.recipe.readout in ("none", "tensored", "trex")

        def sampler(N, B, rng):
            """B ratio estimates from N target shots each (rejected shots included in N); NaN when no shot is kept.
            Linear readout mitigations fold into per-shot values (one product for all repeats); M3 is nonlinear in
            the sampled data and is applied repeat by repeat."""
            freqs = ctx.sample_freqs(rr, N, B, rng)
            if linear_ro:
                num, den = freqs @ a, freqs @ b
            else:
                g = np.array([ctx.mitigate(fr) * mask for fr in freqs])
                num, den = g @ o, g.sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

        disp = display_circuit(ctx)
        disp = disp.replace(ops=tuple(disp.ops) + (Op("barrier", tuple(qubits), (), "barrier", "check"),),
                            name="post-select Z-parity of q%s = %+d" % (",".join(map(str, qubits)), eigenvalue))
        f_meas = ctx.measured(rr)

        def ratio(mit):
            g = mit(f_meas) * mask
            return float(g @ o / g.sum()) if g.sum() > 0 else float("nan")
        return MethodResult(mean=mean, var1=var1, circuits=1, **cal_fields(ctx, ratio),
                            gamma=float(1 / np.sqrt(max(acc_meas, 1e-12))) * ctx.readout_gamma(), accept=acc_meas,
                            implemented=[("Implemented circuit + classical parity check", disp)],
                            details=dict(accept=acc_meas, qubits=qubits, eigenvalue=eigenvalue,
                                         more=[list(g) for g in groups]), notes=notes,
                            sampler=sampler)
