"""The backend contract. A backend executes a resolved instruction list on n qubits
(starting from |0...0>) and returns the final density matrix and, optionally, the
density matrix after every circuit step (for the time-travel inspector)."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class NotSupported(RuntimeError):
    """The backend cannot execute this instruction (e.g. a negative-weight PEC map on Aer)."""


class Backend(Protocol):
    name: str
    label: str
    supports_negative: bool      # can apply non-CP quasi-probability maps (exact PEC means)

    def execute(self, instructions: list, n: int, snapshots: bool = False) -> tuple:
        """-> (rho_final (2^n x 2^n), {step: rho_after_step} or None)."""
        ...


def get_backend(name: str) -> Backend:
    if name == "numpy":
        from .numpy_dm import NumpyBackend
        return NumpyBackend()
    if name == "aer":
        from .aer import AerBackend
        return AerBackend()
    raise ValueError("unknown backend '%s'" % name)


BACKENDS = {"numpy": "Exact density matrix (NumPy)", "aer": "Qiskit Aer (density_matrix)"}


def probs_from_rho(rho: np.ndarray) -> np.ndarray:
    p = np.clip(np.real(np.diag(rho)), 0.0, None)
    s = p.sum()
    return p / s if s > 0 else p
