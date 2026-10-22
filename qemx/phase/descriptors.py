r"""Circuit, observable and state descriptors: the extractors behind the feature dictionary (``qemx.features``) and
the ``descriptors`` block of every phase-study cell record.

Three independent extractors (1.5.1). Statevector availability never controls a gate count:

* :func:`structural_descriptors` — counts of the logical and the compiled circuit. O(gates).
* :func:`observable_descriptors` — Pauli terms of the observable: for a diagonal observable the Z strings of its
  Walsh–Hadamard decomposition (O(n·2^n) on the 2^n stored values), for a Pauli sum its terms and its qubit-wise
  commuting measurement settings (O(terms·n)).
* :func:`state_descriptors` — exact properties of the *ideal pure* state of the logical circuit (including its input
  preparation and measurement-basis change); only the requested ones are computed.

The cell record keeps the keys it always had (:func:`describe`):

* ``meyer_wallach``  Q = 2(1 − (1/n) Σ_k Tr ρ_k²) ∈ [0, 1] (Meyer & Wallach, J. Math. Phys. 43, 4273 (2002))
* ``entropy_half``   von Neumann entropy (bits) of the first ⌊n/2⌋ logical qubits
* ``entropy_max_cut``  the largest entropy over all contiguous bipartitions (first k qubits | the rest)
* ``non_clifford``   non-Clifford gates of the logical circuit; ``non_clifford_isa``: non-Clifford gates of the
                     compiled circuit (``qemx.transforms.non_clifford_slots``: what CDR/FR/NIL training replaces)
* ``obs_weight``     largest Z-string weight in the observable's decomposition; ``obs_terms`` the number of strings
* ``idle_fraction``  fraction of (qubit, layer) slots of the compiled circuit in which an already-started qubit idles

Cost of the exact state features (n qubits, G gates, L layers; 1.5.1)
----------------------------------------------------------------------
* **Storage**: one statevector, 16·2^n bytes (complex128), plus a temporary of the same size per gate application.
* **Simulation**: O(G·2^n) time (each gate is a tensor contraction over the state).
* **A contiguous cut** (first k qubits | rest): the Schmidt spectrum from the *smaller* side. The statevector reshaped
  to M (2^(n−k) × 2^k) — little-endian, so the columns are the first k qubits — gives the reduced state of the smaller
  side as the Gram matrix of d = 2^min(k, n−k): O(2^n·d) time for the product, d² memory, O(d³) for the
  eigenvalues. The half cut costs O(2^(3n/2)) time and 2^n·16 bytes; no matrix larger than 2^⌊n/2⌋ square is formed.
  (Up to 1.5.0 every cut kept the first k qubits: the last cut formed a 2^(n−1) × 2^(n−1) matrix — 1 GiB at n = 14 —
  and its O(2^(3(n−1))) eigen-decomposition.)
* **entropy_max_cut**: all n − 1 cuts, Σ_k O(2^n·2^min(k,n−k)) = O(2^(3n/2)) time.
* **Meyer–Wallach**: n one-qubit reduced states, O(n·2^n).
* **Entanglement profile** (``entropy_profile_max``): one statevector advanced layer by layer — G gate applications
  in total (1.5.0 re-simulated every prefix: G·(L+1)/2 for one gate per layer) — and one half cut per layer:
  O(G·2^n + L·2^(3n/2)) time, O(2^n) memory.

Every loop calls :func:`qemx.resources.check`, so the session's time limit stops a long computation between steps
(a single NumPy call is not interruptible).

Caching. Descriptors are cached per process by the identity of what they are computed from (the circuit's content,
the observable's values or terms, the requested names) and of their definitions (:data:`DEFINITIONS`); the cached
values are immutable (``types.MappingProxyType`` / tuples), so callers cannot alter a shared entry.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from types import MappingProxyType

import numpy as np

from ..circuit import CircuitSpec, is_clifford, layer_order

# identity of the descriptor definitions; part of every cache key (change it when a definition changes)
DEFINITIONS = "qemx.phase.descriptors/2"
STATE_KEYS = ("meyer_wallach", "entropy_half", "entropy_max_cut", "output_entropy_bits")
EIG_TOL = 1e-14                                   # eigenvalues at or below this are dropped from an entropy
COEF_TOL = 1e-12                                  # Z-string / Pauli coefficients at or below this are not terms


# ------------------------------------------------------------------ cache (immutable values, bounded, thread-safe)
_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
CACHE_ENTRIES = 256


def _frozen(value):
    if isinstance(value, dict):
        return MappingProxyType(dict(value))
    if isinstance(value, list):
        return tuple(value)
    return value


def cached(key: tuple, make):
    """``make()`` once per key (LRU, at most CACHE_ENTRIES entries); the stored value is made immutable."""
    key = (DEFINITIONS,) + tuple(key)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    value = _frozen(make())
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_ENTRIES:
            _CACHE.popitem(last=False)
    return value


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def circuit_id(spec: CircuitSpec) -> str:
    """Content identity of a circuit for the descriptors: its width and operations (not its display name)."""
    blob = repr((int(spec.n), tuple((o.name, tuple(o.qubits), tuple(float(x) for x in o.params), o.kind)
                                    for o in spec.ops)))
    return hashlib.sha256(blob.encode()).hexdigest()


def observable_id(observable) -> str:
    if hasattr(observable, "terms"):
        blob = repr((int(observable.n), tuple(observable.terms))).encode()
    else:
        blob = np.ascontiguousarray(np.asarray(observable.values, float)).tobytes()
    return hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------------ pure-state simulation
def _apply(psi: np.ndarray, n: int, o) -> np.ndarray:
    """One gate on the state tensor (shape (2,)*n, axis j = qubit n−1−j), exactly as templates.statevector."""
    from .. import templates as _T
    U = _T._mat(o.name, o.params)                  # one call per gate application (counted by the F06 probe)
    k = len(o.qubits)
    ax = [n - 1 - q for q in reversed(o.qubits)]
    out = np.tensordot(U.reshape((2,) * (2 * k)), psi, axes=(list(range(k, 2 * k)), ax))
    return np.moveaxis(out, list(range(k)), ax)


def _zero_state(n: int) -> np.ndarray:
    psi = np.zeros((2,) * n, complex)
    psi[(0,) * n] = 1.0
    return psi


def _statevector(spec: CircuitSpec) -> np.ndarray:
    """Amplitudes of the ideal pure state of the gate operations (noise skipped), little-endian."""
    from ..resources import check
    n = spec.n
    psi = _zero_state(n)
    for o in spec.ops:
        if o.kind == "gate":
            psi = _apply(psi, n, o)
            check()
    return psi.reshape(-1)


# ------------------------------------------------------------------ entropies of contiguous cuts
def cut_spectrum(psi: np.ndarray, k: int, n: int) -> np.ndarray:
    """Eigenvalues of the reduced state of qubits 0 … k−1 (little-endian: qubit q is bit q of the index) of the pure
    state psi, from the smaller side of the cut. The Schmidt spectrum is shared by both sides, so the Gram matrix of
    the smaller one (2^min(k, n−k) square) suffices."""
    if not 0 < k < n:
        return np.ones(1)
    M = np.asarray(psi).reshape(2 ** (n - k), 2 ** k)        # rows: qubits k … n−1; columns: qubits 0 … k−1
    G = M.conj().T @ M if k <= n - k else M @ M.conj().T
    return np.linalg.eigvalsh((G + G.conj().T) / 2)


def entropy_of_spectrum(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    w = w[w > EIG_TOL]
    return float(-np.sum(w * np.log2(w)))


def cut_entropy(psi: np.ndarray, k: int, n: int) -> float:
    """von Neumann entropy (bits) of qubits 0 … k−1 of the pure state psi (0 for k = 0 or k = n)."""
    return entropy_of_spectrum(cut_spectrum(psi, k, n)) if 0 < k < n else 0.0


def single_qubit_purities(psi: np.ndarray, n: int) -> list:
    """Tr ρ_q² of every one-qubit reduced state (2 × 2 each)."""
    out = []
    for q in range(n):
        T = np.asarray(psi).reshape(2 ** (n - q - 1), 2, 2 ** q)
        r = np.einsum("aib,ajb->ij", T, T.conj())
        out.append(float(np.real(np.trace(r @ r))))
    return out


# general helpers kept for callers of 1.5.0 (arbitrary subsets; contiguous cuts use cut_spectrum)
def _reduced(psi: np.ndarray, keep: list, n: int) -> np.ndarray:
    """Reduced density matrix of the qubits ``keep`` (little-endian: qubit q is bit q of the index). Dense: 2^|keep|
    square — use :func:`cut_spectrum` for entropies."""
    t = psi.reshape([2] * n)                      # axis j = qubit n-1-j
    axes_keep = [n - 1 - q for q in sorted(keep, reverse=True)]
    axes_tr = [a for a in range(n) if a not in axes_keep]
    t = np.transpose(t, axes_keep + axes_tr).reshape(2 ** len(keep), -1)
    return t @ t.conj().T


def _entropy(rho: np.ndarray) -> float:
    return entropy_of_spectrum(np.linalg.eigvalsh((rho + rho.conj().T) / 2))


def state_bytes(n: int) -> int:
    """Peak working memory of the state features (bytes): the statevector, one temporary of the same size per gate
    application, the half-cut Gram matrix and LAPACK workspace (small next to 2^n)."""
    d = 2 ** (n // 2)
    return int(3 * 16 * 2 ** n + 3 * 16 * d * d)


# ------------------------------------------------------------------ the extractors
def structural_descriptors(logical: CircuitSpec, implemented: CircuitSpec) -> dict:
    """Counts of the logical circuit (as written, with its input preparation and basis change) and of the compiled
    circuit; no simulation."""
    def make():
        ops, lay, busy = layer_order(implemented)
        nL = len(busy)
        first = {}
        for o, l in zip(ops, lay):
            if o.kind == "gate":
                for q in o.qubits:
                    first.setdefault(q, l)
        slots = sum(nL - first[q] for q in first)
        idle = sum(1 for q in first for l in range(first[q], nL) if q not in busy[l])
        return dict(non_clifford=sum(1 for o in logical.ops if o.kind == "gate" and not is_clifford(o)),
                    non_clifford_isa=sum(1 for o in implemented.ops if o.kind == "gate" and not is_clifford(o)),
                    two_qubit_logical=logical.count("2q"), two_qubit=implemented.count("2q"),
                    pulses_1q=implemented.count("1q"), depth=nL,
                    idle_fraction=float(idle / slots) if slots else 0.0)
    return cached(("structure", circuit_id(logical), circuit_id(implemented)), make)


def _wht(values: np.ndarray) -> np.ndarray:
    """Walsh–Hadamard coefficients c_S of o = Σ_S c_S Z_S (vectorized; the order of Observable.z_decomposition)."""
    v = np.array(values, float)
    N = len(v)
    h = 1
    while h < N:
        v = v.reshape(-1, 2, h)
        v = np.stack((v[:, 0, :] + v[:, 1, :], v[:, 0, :] - v[:, 1, :]), axis=1).reshape(N)
        h *= 2
    return v / N


def observable_terms(observable) -> list:
    """[(coefficient, frozenset of qubits/bits it acts on, Pauli letters)] of the non-identity terms. Diagonal
    observable: Z strings over classical bits; Pauli sum: its terms over logical qubits."""
    if hasattr(observable, "terms"):
        out = []
        for c, fac in observable.terms:
            if fac and abs(float(c)) > COEF_TOL:
                out.append((float(c), frozenset(int(q) for q, _ in fac), tuple(p for _, p in fac)))
        return out
    c = _wht(observable.values)
    idx = np.flatnonzero(np.abs(c) > COEF_TOL)
    return [(float(c[m]), frozenset(b for b in range(int(m).bit_length()) if (int(m) >> b) & 1), ())
            for m in idx if m]


def observable_descriptors(observable) -> dict:
    """obs_weight (largest term weight), obs_terms (non-identity terms), obs_weight_mean, obs_support (qubits any term
    acts on), obs_settings (measurement settings: 1 for a diagonal observable, the qubit-wise-commuting groups of
    ``qemx.hamiltonian.qwc_groups`` for a Pauli sum), obs_kind."""
    def make():
        terms = observable_terms(observable)
        w = [len(s) for _, s, _ in terms]
        support = set().union(*[s for _, s, _ in terms]) if terms else set()
        if hasattr(observable, "terms"):
            from ..hamiltonian import qwc_groups
            settings = len(qwc_groups([(c, f) for c, f in observable.terms if f and abs(float(c)) > COEF_TOL]))
            kind = "pauli-sum"
        else:
            settings, kind = (1 if terms else 0), "diagonal"
        return dict(obs_weight=max(w) if w else 0, obs_terms=len(w),
                    obs_weight_mean=float(np.mean(w)) if w else 0.0, obs_support=len(support),
                    obs_settings=int(settings), obs_kind=kind)
    return cached(("observable", observable_id(observable)), make)


def state_descriptors(logical: CircuitSpec, want=STATE_KEYS) -> dict:
    """The requested exact state features of the ideal pure state of ``logical`` (keys of STATE_KEYS)."""
    want = tuple(k for k in STATE_KEYS if k in set(want))
    if not want:
        return MappingProxyType({})

    def make():
        from ..resources import check
        n = logical.n
        psi = _statevector(logical)
        out = {}
        if "meyer_wallach" in want:
            out["meyer_wallach"] = 2.0 * (1.0 - float(np.mean(single_qubit_purities(psi, n))))
        if "entropy_half" in want:
            out["entropy_half"] = cut_entropy(psi, n // 2, n) if n >= 2 else 0.0
        if "entropy_max_cut" in want:
            cuts = []
            for k in range(1, n):
                check()
                cuts.append(cut_entropy(psi, k, n))
            out["entropy_max_cut"] = max(cuts) if cuts else 0.0
        if "output_entropy_bits" in want:
            p = np.abs(psi) ** 2
            out["output_entropy_bits"] = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
        return out
    return cached(("state", circuit_id(logical), want), make)


def entanglement_profile(spec: CircuitSpec) -> tuple:
    """Half-cut von Neumann entropy (bits) of the ideal pure state after each layer of the circuit (gates only; layers
    as in ``circuit.layer_order``). Exact. One statevector is advanced layer by layer (each gate applied once)."""
    def make():
        from ..resources import check
        n = spec.n
        ops, lay, _ = layer_order(spec)
        by_layer = {}
        for o, l in zip(ops, lay):
            if o.kind == "gate":
                by_layer.setdefault(l, []).append(o)
        psi = _zero_state(n)
        out = []
        for L in sorted(by_layer):
            for o in by_layer[L]:
                psi = _apply(psi, n, o)
            check()
            out.append(cut_entropy(psi.reshape(-1), n // 2, n) if n >= 2 else 0.0)
        return [float(x) for x in out]
    return cached(("profile", circuit_id(spec)), make)


def describe(logical: CircuitSpec, implemented: CircuitSpec, observable) -> dict:
    """The ``descriptors`` block of a cell record (keys unchanged since 1.3): structure, observable and every exact
    state feature."""
    s = structural_descriptors(logical, implemented)
    o = observable_descriptors(observable)
    st = state_descriptors(logical, STATE_KEYS)
    return dict(meyer_wallach=st["meyer_wallach"], entropy_half=st["entropy_half"],
                entropy_max_cut=st["entropy_max_cut"], non_clifford=s["non_clifford"],
                non_clifford_isa=s["non_clifford_isa"], two_qubit_logical=s["two_qubit_logical"],
                two_qubit=s["two_qubit"], pulses_1q=s["pulses_1q"], depth=s["depth"], obs_weight=o["obs_weight"],
                obs_terms=o["obs_terms"], idle_fraction=s["idle_fraction"],
                output_entropy_bits=st["output_entropy_bits"])
