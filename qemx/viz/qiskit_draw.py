"""Qiskit's own Matplotlib drawer in the IBM Quantum ('iqp') style, for the side-by-side view.
Noise operations appear as labeled barriers; drawn-only boxes (VD derangement) as opaque gates."""
from __future__ import annotations

import io

from ..circuit import CircuitSpec, NOISE_OPS
from ..compile import QISKIT_LOCK
from ..resources import qiskit_serial


@qiskit_serial
def to_display_qiskit(spec: CircuitSpec):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Barrier, Gate
    qc = QuantumCircuit(spec.n)
    for o in spec.ops:
        if o.kind == "barrier":
            qc.barrier(*(o.qubits or range(spec.n)))
        elif o.kind == "gate":
            getattr(qc, o.name)(*o.params, *o.qubits)
        elif o.kind == "noise":
            qc.append(Barrier(o.k, label="%s(%g)" % (NOISE_OPS[o.name][2], o.params[0])), list(o.qubits))
        elif o.kind == "box":
            qc.append(Gate(o.label(), o.k, []), list(o.qubits))
    return qc


@qiskit_serial
def png(spec: CircuitSpec, dark: bool = False, fold: int = 22, scale: float = 0.8) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with QISKIT_LOCK:
        qc = to_display_qiskit(spec)
        fig = qc.draw("mpl", style="iqp-dark" if dark else "iqp", fold=fold, scale=scale)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
