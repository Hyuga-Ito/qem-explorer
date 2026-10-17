r"""Circuit and state descriptors recorded with every cell (inputs for hypothesis A3: does structure — entanglement,
non-Clifford content, observable weight — predict mitigation performance once λ is fixed?).

All quantities are computed from the *ideal* circuits (exact NumPy statevector of the logical circuit, including its
input preparation and measurement-basis change), so they describe the problem, not the noise.

* ``meyer_wallach``  Q = 2(1 − (1/n) Σ_k Tr ρ_k²) ∈ [0, 1] (Meyer & Wallach, J. Math. Phys. 43, 4273 (2002))
* ``entropy_half``   von Neumann entropy (bits) of the first ⌊n/2⌋ logical qubits
* ``entropy_max_cut``  the largest entropy over all contiguous bipartitions of the logical line
* ``non_clifford``   non-Clifford gates of the logical circuit; ``non_clifford_isa``: non-Clifford RZ angles of the
                     compiled circuit (what CDR/FR/NIL replace)
* ``obs_weight``     largest Z-string weight in the observable's decomposition; ``obs_terms`` the number of strings
* ``idle_fraction``  fraction of (qubit, layer) slots of the compiled circuit in which an already-started qubit idles
"""
from __future__ import annotations

import numpy as np

from ..circuit import CircuitSpec, is_clifford, layer_order


def _statevector(spec: CircuitSpec) -> np.ndarray:
    from ..templates import statevector
    return statevector(spec.n, [o for o in spec.ops if o.kind == "gate"])


def _reduced(psi: np.ndarray, keep: list, n: int) -> np.ndarray:
    """Reduced density matrix of the qubits ``keep`` (little-endian: qubit q is bit q of the index)."""
    t = psi.reshape([2] * n)                      # axis j = qubit n-1-j
    axes_keep = [n - 1 - q for q in sorted(keep, reverse=True)]
    axes_tr = [a for a in range(n) if a not in axes_keep]
    t = np.transpose(t, axes_keep + axes_tr).reshape(2 ** len(keep), -1)
    return t @ t.conj().T


def _entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log2(w)))


def describe(logical: CircuitSpec, implemented: CircuitSpec, observable) -> dict:
    n = logical.n
    psi = _statevector(logical)
    pur = [float(np.real(np.trace(r @ r))) for r in (_reduced(psi, [q], n) for q in range(n))]
    mw = 2.0 * (1.0 - float(np.mean(pur)))
    half = _entropy(_reduced(psi, list(range(n // 2)), n)) if n >= 2 else 0.0
    cuts = [_entropy(_reduced(psi, list(range(k)), n)) for k in range(1, n)]
    dec = observable.z_decomposition()
    weights = [bin(int(m)).count("1") for m, c in dec.items() if abs(c) > 1e-12 and m]
    ops, lay, busy = layer_order(implemented)
    nL = len(busy)
    first = {}
    for o, l in zip(ops, lay):
        if o.kind == "gate":
            for q in o.qubits:
                first.setdefault(q, l)
    slots = sum(nL - first[q] for q in first)
    idle = sum(1 for q in first for l in range(first[q], nL) if q not in busy[l])
    p_ideal = np.abs(psi) ** 2
    return dict(
        meyer_wallach=mw, entropy_half=half, entropy_max_cut=max(cuts) if cuts else 0.0,
        non_clifford=sum(1 for o in logical.ops if o.kind == "gate" and not is_clifford(o)),
        non_clifford_isa=sum(1 for o in implemented.ops if o.kind == "gate" and not is_clifford(o)),
        two_qubit_logical=logical.count("2q"), two_qubit=implemented.count("2q"),
        pulses_1q=implemented.count("1q"), depth=implemented.depth(),
        obs_weight=max(weights) if weights else 0, obs_terms=len(weights),
        idle_fraction=float(idle / slots) if slots else 0.0,
        output_entropy_bits=float(-np.sum(p_ideal[p_ideal > 0] * np.log2(p_ideal[p_ideal > 0]))))
