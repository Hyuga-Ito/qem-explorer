"""simulate(): resolve a circuit with its noise, execute it on a backend, average over the
quasi-static field samples, map qubits to classical bits and apply readout error."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends.base import get_backend, probs_from_rho
from .circuit import CircuitSpec
from .noise import NoiseSpec, apply_readout, field_deltas, resolve


@dataclass
class RunResult:
    n: int
    probs: np.ndarray            # over classical bits, before readout error
    probs_meas: np.ndarray       # with readout error: what the device reports
    rho: np.ndarray              # final state of the physical qubits
    ops: list                    # executed operations (layer order)
    layers: list                 # layer of each executed op
    snapshots: dict | None = None    # step -> rho after that step (averaged over field samples)
    samples: int = 1


def permute_probs(p_qubits: np.ndarray, mm: dict, n: int) -> np.ndarray:
    """Distribution over classical bits from one over qubits, clbit i <- qubit mm[i]."""
    if all(mm.get(i, i) == i for i in range(n)):
        return p_qubits
    if sorted(mm) != list(range(n)) or sorted(mm.values()) != list(range(n)):
        raise ValueError("the measurement map reads %d classical bit(s) from qubits %s, but the circuit has %d "
                         "qubits; every qubit must be measured exactly once (a forced layout that routes through "
                         "other physical qubits adds unmeasured ones)" % (len(mm), sorted(set(mm.values())), n))
    t = p_qubits.reshape([2] * n)             # axis j = qubit n-1-j
    order = [n - 1 - mm[n - 1 - j] for j in range(n)]
    return np.transpose(t, order).reshape(-1)


def simulate(spec: CircuitSpec, noise: NoiseSpec, backend: str = "numpy", *, measure_map: dict | None = None,
             snapshots: bool = False, ideal: bool = False, twirl: str = "none", dd: str = "none",
             corrections=None, seed: int = 7, layout=None) -> RunResult:
    from . import device
    if device.is_device(backend):
        if layout is None:
            raise ValueError("a device backend needs the physical layout of the compiled circuit")
        return device.run_device(spec, backend, tuple(layout), measure_map, ideal=ideal, twirl=twirl, dd=dd,
                               corrections=corrections, snapshots=bool(snapshots))
    be = get_backend(backend)
    n = spec.n
    deltas = [None] if ideal else field_deltas(noise, n, seed)
    rho_acc, snap_acc, ops, lay = None, None, None, None
    from .resources import check
    for d in deltas:
        check()
        instr, ops, lay = resolve(spec, noise, delta=d, ideal=ideal, twirl=twirl, dd=dd, corrections=corrections)
        rho, snaps = be.execute(instr, n, snapshots=snapshots)
        rho_acc = rho if rho_acc is None else rho_acc + rho
        if snapshots is True:
            if snap_acc is None:
                snap_acc = {k: v.copy() for k, v in snaps.items()}
            else:
                for k, v in snaps.items():
                    snap_acc[k] += v
    m = len(deltas)
    rho = rho_acc / m
    if snap_acc is not None:
        snap_acc = {k: v / m for k, v in snap_acc.items()}
    mm = measure_map or {i: i for i in range(n)}
    p = permute_probs(probs_from_rho(rho) if backend == "aer" else np.real(np.diag(rho)), mm, n)
    pm = p if ideal else apply_readout(p, noise, n)
    return RunResult(n, p, pm, rho, ops, lay, snap_acc, m)


# ------------------------------------------------------------ state metrics
def reduced(rho: np.ndarray, q: int, n: int) -> np.ndarray:
    """One-qubit reduced density matrix of qubit q."""
    A, B = 2 ** (n - 1 - q), 2 ** q
    return np.einsum("ajbakb->jk", rho.reshape(A, 2, B, A, 2, B))


def bloch_vectors(rho: np.ndarray, n: int) -> np.ndarray:
    """(n, 3) Bloch vectors: rho_q = (I + x X + y Y + z Z)/2."""
    out = np.zeros((n, 3))
    for q in range(n):
        r = reduced(rho, q, n)
        out[q] = [2 * np.real(r[0, 1]), -2 * np.imag(r[0, 1]), np.real(r[0, 0] - r[1, 1])]
    return out


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity; exact overlap when sigma is pure."""
    w, V = np.linalg.eigh((sigma + sigma.conj().T) / 2)
    if w[-1] > 1 - 1e-9:                       # sigma pure
        psi = V[:, -1]
        return float(np.real(np.conj(psi) @ rho @ psi))
    from scipy.linalg import sqrtm
    s = sqrtm(sigma)
    return float(np.real(np.trace(sqrtm(s @ rho @ s))) ** 2)


def purity(rho: np.ndarray) -> float:
    """Tr ρ² = Σ|ρ_ij|² for Hermitian ρ (O(4^n) instead of a matrix product)."""
    return float(np.real(np.vdot(rho, rho)))
