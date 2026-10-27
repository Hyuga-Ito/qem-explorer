"""Exact density-matrix executor in NumPy.

The state is a tensor with 2n axes; row axis j holds qubit n-1-j (so the flat index is
little-endian, as in Qiskit). A k-qubit matrix M given little-endian over the qubit list
(q_0, ..., q_{k-1}) acts on the row axes of those qubits; M† acts on the column axes.
Mixtures may carry negative weights, so quasi-probability maps (PEC) run exactly.
"""
from __future__ import annotations

import numpy as np

from ..resources import check


def _apply(t: np.ndarray, M: np.ndarray, qs: tuple, n: int, cols: bool = False) -> np.ndarray:
    k = len(qs)
    Mt = M.reshape((2,) * (2 * k))
    base = n if cols else 0
    axes = [base + n - 1 - q for q in reversed(qs)]          # in_{k-1} ... in_0
    if cols:
        Mt = Mt.conj()
    t = np.tensordot(Mt, t, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(t, list(range(k)), axes)


def conj_by(t: np.ndarray, U: np.ndarray, qs: tuple, n: int) -> np.ndarray:
    return _apply(_apply(t, U, qs, n), U, qs, n, cols=True)


def _embed(M: np.ndarray, qs: tuple, Q: tuple) -> np.ndarray:
    """Matrix M on qubits qs (little-endian) as an operator on the qubit list Q (qs ⊆ Q)."""
    if tuple(qs) == tuple(Q):
        return M
    k = len(Q)
    local = tuple(Q.index(q) for q in qs)
    t = np.eye(2 ** k, dtype=complex).reshape((2,) * (2 * k))
    return _apply(t, M, local, k).reshape(2 ** k, 2 ** k)


def _kron_dm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """np.kron for 2-D arrays without its generic-shape overhead (the fused path builds many small ones)."""
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])


def _superop(ins, Q: tuple) -> np.ndarray:
    """Row-major superoperator S (vec ρ = ρ.ravel()) of one instruction, on the qubit list Q."""
    if ins.kind == "unitary":
        U = _embed(ins.data, ins.qubits, Q)
        return _kron_dm(U, U.conj())
    if ins.kind == "kraus":
        S = 0
        for K in ins.data:
            Ke = _embed(K, ins.qubits, Q)
            S = S + _kron_dm(Ke, Ke.conj())
        return S
    if ins.kind == "mixture":
        S = 0
        for w, U in ins.data:
            if w == 0:
                continue
            Ue = _embed(U, ins.qubits, Q)
            S = S + w * _kron_dm(Ue, Ue.conj())
        return S
    raise ValueError(ins.kind)


def fuse(instructions: list, max_k: int = 2) -> list:
    """Group consecutive instructions of one circuit step whose qubits lie inside the first one's qubits
    (a gate and its noise, an idle segment) into single superoperators: [(Q, S, step, last_of_step)].
    Exact: channels compose; negative weights (PEC) are kept."""
    blocks = []
    cur_Q, cur_S, cur_step = None, None, None
    for idx, ins in enumerate(instructions):
        qs = tuple(ins.qubits)
        if cur_Q is not None and ins.step == cur_step and set(qs) <= set(cur_Q):
            cur_S = _superop(ins, cur_Q) @ cur_S
        else:
            if cur_Q is not None:
                blocks.append([cur_Q, cur_S, cur_step])
            cur_Q = qs if len(qs) <= max_k else qs
            cur_S = _superop(ins, cur_Q)
            cur_step = ins.step
    if cur_Q is not None:
        blocks.append([cur_Q, cur_S, cur_step])
    for i, b in enumerate(blocks):
        b.append(i + 1 == len(blocks) or blocks[i + 1][2] != b[2])
    return blocks


def apply_superop(t: np.ndarray, S: np.ndarray, Q: tuple, n: int) -> np.ndarray:
    k = len(Q)
    St = S.reshape((2,) * (4 * k))
    rows = [n - 1 - q for q in reversed(Q)]
    cols = [n + n - 1 - q for q in reversed(Q)]
    t = np.tensordot(St, t, axes=(list(range(2 * k, 4 * k)), rows + cols))
    return np.moveaxis(t, list(range(2 * k)), rows + cols)


class NumpyBackend:
    name = "numpy"
    label = "Exact density matrix (NumPy)"
    supports_negative = True

    def execute(self, instructions: list, n: int, snapshots=False, rho0=None) -> tuple:
        """snapshots: False, True (keep ρ after every step) or a callable f(step, ρ) that receives each
        step's state and keeps only what it needs (compact time travel for large registers)."""
        d = 2 ** n
        keep = snapshots is True
        hook = snapshots if callable(snapshots) else None
        if rho0 is None:
            rho0 = np.zeros((d, d), dtype=complex)
            rho0[0, 0] = 1.0
        t = np.asarray(rho0, dtype=complex).reshape((2,) * (2 * n))
        snaps = {} if keep else None
        if n >= 3 and all(len(i.qubits) <= 2 for i in instructions):
            # fused path: one superoperator per (step, qubit block) instead of one or two tensordots per term
            for Q, S, step, last in fuse(instructions):
                check()
                t = apply_superop(t, S, Q, n)
                if (keep or hook) and last:
                    if keep:
                        snaps[step] = t.reshape(d, d).copy()
                    else:
                        hook(step, t.reshape(d, d))
            return t.reshape(d, d), snaps
        for idx, ins in enumerate(instructions):
            check()
            if ins.kind == "unitary":
                t = conj_by(t, ins.data, ins.qubits, n)
            elif ins.kind == "kraus":
                t = sum(conj_by(t, K, ins.qubits, n) for K in ins.data)
            elif ins.kind == "mixture":
                acc = None
                for w, U in ins.data:
                    if w == 0:
                        continue
                    term = w * conj_by(t, U, ins.qubits, n)
                    acc = term if acc is None else acc + term
                t = acc
            else:
                raise ValueError(ins.kind)
            if (keep or hook) and (idx + 1 == len(instructions) or instructions[idx + 1].step != ins.step):
                if keep:
                    snaps[ins.step] = t.reshape(d, d).copy()
                else:
                    hook(ins.step, t.reshape(d, d))
        return t.reshape(d, d), snaps
