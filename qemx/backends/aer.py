"""Qiskit Aer executor: the same instruction list as a QuantumCircuit run with
``AerSimulator(method="density_matrix")``. Unitaries become ``UnitaryGate``s, Kraus
channels and non-negative mixtures become ``Kraus`` channel instructions, and snapshots
are ``save_density_matrix`` instructions after every circuit step.

Works with Qiskit 1.x and 2.x and qiskit-aer >= 0.13."""
from __future__ import annotations

import numpy as np

from .base import NotSupported
from ..resources import qiskit_serial

_SIM = None


@qiskit_serial
def simulator():
    global _SIM
    if _SIM is None:
        from qiskit_aer import AerSimulator
        _SIM = AerSimulator(method="density_matrix")
    return _SIM


@qiskit_serial
def to_aer_circuit(instructions: list, n: int, snapshots: bool = False):
    import qiskit_aer  # noqa: F401  (registers save_density_matrix on QuantumCircuit)
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Kraus
    qc = QuantumCircuit(n)
    for idx, ins in enumerate(instructions):
        qs = list(ins.qubits)
        if ins.kind == "unitary":
            qc.append(UnitaryGate(ins.data, check_input=False), qs)
        elif ins.kind == "kraus":
            qc.append(Kraus(list(ins.data)).to_instruction(), qs)
        elif ins.kind == "mixture":
            if any(w < -1e-15 for w, _ in ins.data):
                raise NotSupported("Aer cannot apply a quasi-probability map with negative weights")
            ks = [np.sqrt(max(w, 0.0)) * U for w, U in ins.data if w > 0]
            qc.append(Kraus(ks).to_instruction(), qs)
        if snapshots and (idx + 1 == len(instructions) or instructions[idx + 1].step != ins.step):
            qc.save_density_matrix(label="s%d" % ins.step)
    qc.save_density_matrix(label="final")
    return qc


class AerBackend:
    name = "aer"
    label = "Qiskit Aer (density_matrix)"
    supports_negative = False

    @qiskit_serial
    def execute(self, instructions: list, n: int, snapshots=False) -> tuple:
        from ..resources import check
        check()
        if callable(snapshots):
            raise NotSupported("compact time travel runs on the NumPy executor")
        qc = to_aer_circuit(instructions, n, snapshots)
        res = simulator().run(qc, shots=1).result()
        if not res.success:
            raise RuntimeError("Aer: %s" % (getattr(res, "status", "") or "simulation failed"))
        data = res.data(0)
        rho = np.asarray(data["final"])
        snaps = None
        if snapshots:
            snaps = {int(k[1:]): np.asarray(v) for k, v in data.items() if k.startswith("s")}
        return rho, snaps
