r"""Circuit transforms that change what the device runs.

fold_global      U -> U (U^\dagger U)^k plus a partial fold of the last gates (Giurgica-Tiron et al. 2020)
fold_local       G -> G (G^\dagger G)^k on two-qubit gates; fractional λ folds a subset once more (random, or a
                 nested order: random, stratified over time, error-balanced) — fractional ZNE
clifford_replace non-Clifford rotations -> Clifford angles (training circuits of CDR / FR)
twirl_instance   one sampled Pauli twirl of every Clifford two-qubit gate (for display)
dd_display       the dynamical-decoupling pulses the simulator applies, placed in their idle windows
pec_instance     one sampled PEC circuit with its Pauli insertions and sign

Explicit noise operations fold with the gates around them (noise is not inverted). What scales by λ, and what
does not:

* gate noise of the counted gates (two-qubit gates; physical one-qubit gates when there are none) and the explicit
  noise operations scale exactly by λ for odd integer λ under global folding; a partial fold (fractional λ) repeats
  only the last gates, so λ — counted in two-qubit gates — does not describe the one-qubit gates' noise;
* **idle noise does not scale by λ.** Idle windows are set by the schedule of the folded circuit: qubits that wait
  before their first gate are noiseless in U but wait entangled through U† and the repeated U (GHZ-4 under global
  folding: λ = 3, 5 give 7×, 13× the idle qubit-layers), and ASAP layering can merge layers across fold boundaries;
* readout error is not folded at all.

ZNE therefore extrapolates a noise that is a mixture of sources scaled by different factors, as on hardware.
"""
from __future__ import annotations

import math

import numpy as np

from .circuit import CircuitSpec, GATES, Op, inverse, is_clifford, CLIFFORD_ANGLE, layer_order
from .noise import DD_SEQUENCES, idle_windows


def _inv(o: Op) -> Op:
    if o.kind == "gate":
        return inverse(o).with_tag("fold-inv")
    if o.kind == "barrier" and o.tag == "keep":          # alignment barriers stay alignment barriers in every copy
        return o
    return o.with_tag("fold-inv")


def _copy(o: Op) -> Op:
    return o if (o.kind == "barrier" and o.tag == "keep") else o.with_tag("fold")


def _units(ops) -> tuple:
    """Folding counts two-qubit gates; a circuit without any counts its physical one-qubit gates (virtual RZ-type
    gates are noiseless frame changes, so folding them would not scale the noise)."""
    has2q = any(o.kind == "gate" and o.k == 2 for o in ops)
    return (lambda o: o.kind == "gate" and (o.k == 2 if has2q else not o.is_virtual)), has2q


def fold_global(spec: CircuitSpec, lam: float) -> CircuitSpec:
    """Global folding U (U†U)^k plus a partial fold of the last gates, to the scale factor λ >= 1 counted in
    two-qubit gates (all gates if there are none). The partial fold takes the number of gates closest to the
    request. Barriers stay in place (they shape the schedule, hence the idle noise)."""
    body = list(spec.ops)
    if lam <= 1 + 1e-12:
        return spec
    unit, _ = _units(body)
    base = sum(1 for o in body if unit(o))
    if base == 0:
        raise ValueError("the circuit has no gates to fold")
    k = int((lam - 1 + 1e-9) // 2)
    ops = list(body)
    for _ in range(k):
        ops += [_inv(o) for o in reversed(body)] + [_copy(o) for o in body]
    g = int(round((lam - 1 - 2 * k) * base / 2))          # gates of the partial fold (each adds two)
    if g > 0:
        tail, acc = [], 0
        for o in reversed(body):
            tail.append(o)
            if unit(o):
                acc += 1
                if acc >= g:
                    break
        tail.reverse()
        ops += [_inv(o) for o in reversed(tail)] + [_copy(o) for o in tail]
    return spec.replace(ops=ops, name="%s ·fold λ=%g" % (spec.name, lam))


FOLD_SUBSETS = ("random", "nested", "stratified", "balanced")


def _bit_reversal_order(n: int) -> list:
    """0..n−1 in bit-reversal (van der Corput) order: every prefix is spread evenly over the range."""
    if n <= 1:
        return list(range(n))
    m = (n - 1).bit_length()
    rev = [int(format(i, "0%db" % m)[::-1], 2) for i in range(2 ** m)]
    return [r for r in rev if r < n]


def fold_order(n: int, subset: str = "nested", seed: int = 0, weights=None) -> list:
    """The order in which the n foldable gates (positions 0..n−1 in circuit order) are folded once more for a
    fractional λ: a fold of k extra gates takes the first k. Every order is *nested* — the gates folded at a smaller
    λ are folded at every larger one — so the noise grows monotonically along the scales.

    nested      a seeded random permutation
    stratified  bit-reversal order of the positions: deterministic, spread evenly over the circuit's duration
    balanced    error-balanced: greedily, each prefix has the mean ``weights`` (e.g. the calibrated two-qubit gate
                errors) closest to the circuit's mean, so the folded *noise* grows like the folded gate count (λ in
                gates ≈ λ in fault rate); ties follow the stratified order. Without weights: stratified."""
    if subset == "nested":
        return [int(x) for x in np.random.default_rng(seed).permutation(n)]
    strat = _bit_reversal_order(n)
    if subset == "stratified" or (subset == "balanced" and weights is None):
        return strat
    if subset != "balanced":
        raise ValueError("fold subset must be one of %s" % ", ".join(FOLD_SUBSETS))
    w = np.asarray(weights, float)
    if w.shape != (n,) or not np.all(np.isfinite(w)):
        raise ValueError("balanced folding needs one finite weight per foldable gate (%d)" % n)
    rank = {p: r for r, p in enumerate(strat)}
    target = float(w.mean())
    left, order, s = set(range(n)), [], 0.0
    for j in range(1, n + 1):
        best = min(left, key=lambda p: (abs(s + w[p] - j * target), rank[p]))
        order.append(best)
        left.discard(best)
        s += w[best]
    return order


def fold_local(spec: CircuitSpec, lam: float, seed: int = 0, subset: str = "random", weights=None) -> CircuitSpec:
    """Local folding: each chosen gate G becomes G (G†G)^k -- two-qubit gates, or all gates when there are
    none -- with a subset folded once more for a fractional λ: λ = 1 + 2(k + m/n) for m of the n gates.

    ``subset`` chooses those m gates: ``random`` (a seeded random subset; the behaviour of 1.4.0, not nested across
    scales), or one of :func:`fold_order`'s nested orders (``nested``, ``stratified``, ``balanced`` with ``weights``,
    one per foldable gate in circuit order). Fractional scales between 1 and 3 (e.g. λ = 1.33, 1.67, 2) amplify the
    noise in small steps: the extrapolation sees the circuit before its signal decays."""
    unit, _ = _units(spec.ops)
    idx = [i for i, o in enumerate(spec.ops) if unit(o)]
    if lam <= 1 + 1e-12:
        return spec
    if not idx:
        raise ValueError("the circuit has no gates to fold")
    k = int((lam - 1 + 1e-9) // 2)
    frac = (lam - 1 - 2 * k) / 2
    m = int(round(frac * len(idx))) if frac > 1e-9 else 0
    if subset == "random":
        rng = np.random.default_rng(seed)
        extra = set(rng.choice(idx, size=m, replace=False).tolist()) if frac > 1e-9 else set()
    else:
        extra = {idx[p] for p in fold_order(len(idx), subset, seed, weights)[:m]}
    ops = []
    for i, o in enumerate(spec.ops):
        ops.append(o)
        if i in idx:
            for _ in range(k + (1 if i in extra else 0)):
                ops += [_inv(o), o.with_tag("fold")]
    return spec.replace(ops=ops, name="%s ·local fold λ=%g" % (spec.name, lam))


def achieved_lambda(spec: CircuitSpec, folded: CircuitSpec) -> float:
    a, b = spec.count("2q"), folded.count("2q")
    if a:
        return b / a
    return folded.count("1q") / max(spec.count("1q"), 1)       # the same units _units folds


# ------------------------------------------------------------ training circuits
def non_clifford_slots(spec: CircuitSpec) -> list:
    return [i for i, o in enumerate(spec.ops) if o.kind == "gate" and not is_clifford(o)]


def clifford_replace(spec: CircuitSpec, rng, keep: int = 0, sigma: float | None = 1.0,
                     resample_all: bool = False, flip_inputs: bool = False) -> CircuitSpec:
    """Replace every non-Clifford angle but `keep` random ones by a Clifford angle, chosen with
    probability ∝ exp(-(Δθ/σ)²) (near-Clifford training circuits, Czarnik et al. 2021). σ = None draws the
    Clifford angle uniformly (more diverse training circuits). resample_all=True also redraws the rotations that
    already have Clifford angles (uniformly), which diversifies circuits whose observable is blind to the
    non-Clifford gates alone. flip_inputs=True additionally starts from a random computational basis state
    (X on a random subset of qubits), for observables that only see the input. Every training circuit stays
    Clifford apart from the kept gates."""
    slots = non_clifford_slots(spec)
    kept = set(rng.choice(slots, size=min(keep, len(slots)), replace=False).tolist()) if keep and slots else set()
    ops = list(spec.ops)
    extra = [i for i, o in enumerate(ops) if resample_all and o.kind == "gate" and o.name in CLIFFORD_ANGLE
             and i not in slots]
    for i in list(slots) + extra:
        if i in kept:
            continue
        o = ops[i]
        if o.name in ("t", "tdg"):
            choices = ["id", "s"] if o.name == "t" else ["id", "sdg"]
            ops[i] = Op(choices[int(rng.integers(2))], o.qubits, (), "gate", "train")
            continue
        step = CLIFFORD_ANGLE[o.name]
        th = o.params[0]
        cands = np.array([j * step for j in range(int(round(2 * math.pi / step)))])
        if sigma is None or i in extra:
            c = float(rng.choice(cands))
        else:
            d = np.angle(np.exp(1j * (cands - th)))
            w = np.exp(-(d / sigma) ** 2)
            c = float(rng.choice(cands, p=w / w.sum()))
        ops[i] = Op(o.name, o.qubits, (c,), "gate", "train")
    if flip_inputs:
        flips = [q for q in range(spec.n) if rng.random() < 0.5]
        ops = [Op("x", (q,), (), "gate", "train") for q in flips] + ops
    return spec.replace(ops=ops, name=spec.name + " (training)")


# ------------------------------------------------------------ display helpers
CLIFF2 = {"cx", "cz", "ecr", "swap"}


def _conj_pauli(gate: str, pa: str, pb: str) -> tuple:
    """Paulis after a CX/CZ that undo the ones before: U (Pa⊗Pb) U† (up to sign)."""
    from .noise import pauli_matrix
    from .circuit import gate_matrix
    U = gate_matrix(gate, ())
    P = pauli_matrix(pa + pb)
    Q = U @ P @ U.conj().T
    for a in "IXYZ":
        for b in "IXYZ":
            M = pauli_matrix(a + b)
            ov = np.trace(M.conj().T @ Q) / 4
            if abs(abs(ov) - 1) < 1e-9:
                return a, b
    raise RuntimeError


def twirl_instance(spec: CircuitSpec, seed: int = 0) -> CircuitSpec:
    rng = np.random.default_rng(seed)
    ops = []
    for o in spec.ops:
        if o.kind == "gate" and o.k == 2 and o.name in CLIFF2:
            pa, pb = "IXYZ"[rng.integers(4)], "IXYZ"[rng.integers(4)]
            qa, qb = _conj_pauli(o.name, pa, pb)
            ops += [Op(p.lower(), (q,), (), "gate", "twirl") for p, q in ((pa, o.qubits[0]), (pb, o.qubits[1])) if p != "I"]
            ops.append(o)
            ops += [Op(p.lower(), (q,), (), "gate", "twirl") for p, q in ((qa, o.qubits[0]), (qb, o.qubits[1])) if p != "I"]
        else:
            ops.append(o)
    return spec.replace(ops=ops, name=spec.name + " (twirl instance)")


def dd_display(spec: CircuitSpec, seq: str = "X2", dd_min: int = 2) -> CircuitSpec:
    """The circuit with DD pulses drawn in the idle windows where the simulator applies them."""
    if seq == "none":
        return spec
    ops_exec, lay, busy = layer_order(spec)
    win = idle_windows(spec)
    by_layer = {}
    for q, runs in win.items():
        for a, b in runs:
            if b - a + 1 < dd_min:
                continue
            L = b - a + 1
            for g, t in DD_SEQUENCES[seq]:
                l = a + min(int(t * L), L - 1)
                by_layer.setdefault(l, []).append(Op(g, (q,), (), "gate", "dd"))
    ops = []
    for i, (o, l) in enumerate(zip(ops_exec, lay)):
        ops.append(o)
        last = (i + 1 == len(ops_exec)) or lay[i + 1] != l
        if last:
            ops += by_layer.get(l, [])
    return spec.replace(ops=ops, name=spec.name + " (DD %s)" % seq)
