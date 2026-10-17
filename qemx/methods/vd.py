r"""Virtual distillation / error suppression by derangement (Huggins et al., PRX 11, 041036
(2021); Koczor, PRX 11, 031057 (2021)).

With M copies of the noisy state and a derangement circuit,

.. math:: \langle O\rangle_M=\frac{\mathrm{Tr}(\rho^MO)}{\mathrm{Tr}(\rho^M)},

which suppresses every eigenvector of ρ but the dominant one exponentially in M. The sampling
cost is about :math:`\mathrm{Tr}(\rho^M)^{-2}`, on M·n qubits. The simulation uses ideal copies,
an ideal derangement and ideal readout: an upper bound on hardware performance. What remains is
the *coherent mismatch*: the dominant eigenvector is not the ideal state when the noise rotates it.
"""
from __future__ import annotations

import numpy as np

from .base import MethodResult, Param, QEMMethod, register
from ..circuit import CircuitSpec, Op
from ..engine import permute_probs


def phys_values(o: np.ndarray, mm: dict, n: int) -> np.ndarray:
    """o over classical bits -> o over physical-qubit bitstrings (clbit i <- qubit mm[i])."""
    x = np.arange(2 ** n)
    c = np.zeros(2 ** n, dtype=int)
    for i in range(n):
        c |= ((x >> mm[i]) & 1) << i
    return o[c]


def vd_moments(rho: np.ndarray, o_diag: np.ndarray, M: int) -> dict:
    """Exact per-shot moments of the virtual-distillation estimators on M copies of ρ.

    The derangement circuit measures, in one basis, the commuting Hermitian operators
    y = H and x = O_sym·H with H = (D + D⁻¹)/2 (D the cyclic shift of the copies; H = D for M = 2) and
    O_sym = (1/M)Σ_m O⁽ᵐ⁾. Then E[x] = Tr(ρᴹO), E[y] = Tr(ρᴹ), and every second moment is a product of traces
    over the cycles of Dᵏ: E[⊗A_j · Dᵏ] = Π_cycles Tr(ρA_{c1} ρA_{c2} ⋯). Returns the means, variances, covariance
    and the per-shot variance of the ratio R = E[x]/E[y] (delta method)."""
    from math import gcd
    O = np.diag(np.asarray(o_diag, float)).astype(complex)
    O2 = O @ O
    pw = [np.eye(len(rho), dtype=complex)]
    for _ in range(M):
        pw.append(pw[-1] @ rho)
    tr = lambda X: float(np.real(np.trace(X)))
    T = [tr(pw[L]) for L in range(M + 1)]                     # Tr(ρ^L)

    def cyc(k):
        k %= M
        if k == 0:
            return 1, M, M                                     # identity: M cycles of length 1
        g = gcd(k, M)
        return g, M // g, g

    def P(k):                                                  # E[D^k]
        g, L, _ = cyc(k)
        return 1.0 if k % M == 0 else T[L] ** g

    def Q(k, A):                                               # E[A^(1) D^k]
        if k % M == 0:
            return tr(rho @ A)
        g, L, _ = cyc(k)
        return tr(pw[L] @ A) * T[L] ** (g - 1)

    def S(k, m1, m2):                                          # E[O^(m1) O^(m2) D^k], m1 != m2
        if k % M == 0:
            return tr(rho @ O) ** 2
        g, L, _ = cyc(k)
        if (m2 - m1) % g:                                      # different cycles
            return tr(pw[L] @ O) ** 2 * T[L] ** (g - 2)
        d = next(dd for dd in range(1, L) if (dd * k - (m2 - m1)) % M == 0)
        return tr(pw[d] @ O @ pw[L - d] @ O) * T[L] ** (g - 1)

    ks = (2, 0, -2) if M > 2 else (0, 0, 0)                    # H² = (D² + 2 + D⁻²)/4; H² = I for M = 2
    wts = (0.25, 0.5, 0.25)
    Ey = T[M]
    Ex = tr(pw[M] @ O)
    Ey2 = sum(w * P(k) for w, k in zip(wts, ks))
    Exy = sum(w * Q(k, O) for w, k in zip(wts, ks))
    Ex2 = 0.0
    for w, k in zip(wts, ks):
        diag = M * Q(k, O2)
        off = sum(S(k, m1, m2) for m1 in range(M) for m2 in range(M) if m1 != m2)
        Ex2 += w * (diag + off) / M ** 2
    vx, vy, cxy = Ex2 - Ex ** 2, Ey2 - Ey ** 2, Exy - Ex * Ey
    R = Ex / Ey
    var_ratio = max(vx - 2 * R * cxy + R * R * vy, 0.0) / Ey ** 2
    return dict(Ex=Ex, Ey=Ey, var_x=vx, var_y=vy, cov=cxy, R=R, var1=var_ratio)


@register
class VD(QEMMethod):
    key = "vd"
    label = "Virtual distillation"
    family = "vd"
    tex = r"\langle O\rangle_M=\frac{\mathrm{Tr}(\rho^M O)}{\mathrm{Tr}(\rho^M)},\qquad \text{cost}\approx\mathrm{Tr}(\rho^M)^{-2}"
    why = (r"Write $\rho=\sum_k\lambda_k|\psi_k\rangle\langle\psi_k|$. Raising $\rho$ to the power $M$ keeps "
           r"$\lambda_1^M$ and crushes the rest, so $\rho^M/\mathrm{Tr}\rho^M\to|\psi_1\rangle\langle\psi_1|$. "
           r"Incoherent errors disappear; a coherent error that rotates $|\psi_1\rangle$ away from the ideal state "
           r"does not.")
    params = (Param("M", "copies M", "choice", 2, (2, 3, 4)),)

    def short(self, p):
        return "VD M=%d" % int(p["M"])

    def run(self, ctx, M=2) -> MethodResult:
        M = int(M)
        rr = ctx.run(need_state=True)
        rho = rr.rho
        o_phys = phys_values(ctx.obs.o, ctx.mm, ctx.n)
        rM = np.linalg.matrix_power(rho, M)
        tr = float(np.real(np.trace(rM)))
        mean = float(np.real(np.diag(rM)) @ o_phys) / tr
        w, V = np.linalg.eigh((rho + rho.conj().T) / 2)
        dom = V[:, -1]
        dom_val = float(np.real(np.conj(dom) @ (o_phys * dom)))
        mom = vd_moments(rho, o_phys, M)
        var1 = float(mom["var1"])
        n = ctx.n
        # display: M copies of the circuit and the derangement (drawn as boxes, not simulated)
        ops = []
        for m in range(M):
            ops += [Op(o.name, tuple(q + m * n for q in o.qubits), o.params, o.kind, o.tag) for o in ctx.spec.ops]
        ops.append(Op("barrier", tuple(range(M * n)), (), "barrier"))
        for q in range(n):
            ops.append(Op("derange", tuple(q + m * n for m in range(M)), (), "box", "vd"))
        disp = CircuitSpec(M * n, tuple(ops), "%d copies + derangement" % M) if M * n <= 12 else None
        notes = ["Ideal copies, derangement and readout: an upper bound on what VD achieves on hardware. The "
                 "variance is exact for that measurement: numerator and denominator are estimated from the same "
                 "shots (delta method for their ratio)."]
        if ctx.recipe.readout != "none":
            notes.append("Readout mitigation does not apply to the derangement measurement and is ignored here.")
        return MethodResult(mean=mean, var1=var1, circuits=1, gamma=1 / tr, extra_qubits=(M - 1) * n,
                            implemented=[("%d copies + derangement (%d qubits)" % (M, M * n), disp)] if disp else [],
                            details=dict(eigs=np.sort(w)[::-1], trace=tr, dominant_value=dom_val, M=M, moments=mom),
                            notes=notes,
                            sampler=lambda N, B, rng: rng.normal(mean, np.sqrt(var1 / max(N, 1)), size=B))
