r"""Diagonal observables :math:`O=\sum_x o(x)\,|x\rangle\langle x|` in the measured basis.

A measurement basis other than Z is handled by the circuit (basis-change gates before
measurement), so every estimator only ever needs the vector ``o``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observable:
    key: str
    label: str            # plain text, e.g. "⟨Z0 Z1⟩"
    tex: str              # TeX, e.g. r"\langle Z_0Z_1\rangle"
    values: tuple         # o(x), x = 0 .. 2^n - 1 (little-endian)

    @property
    def o(self) -> np.ndarray:
        return np.asarray(self.values, float)

    @property
    def n(self) -> int:
        return int(round(np.log2(len(self.values))))

    def z_decomposition(self) -> dict:
        """o = sum_S c_S Z_S (Walsh–Hadamard): {mask: c_S}, used by TREX."""
        v = self.o.copy()
        n = self.n
        h = 1
        while h < len(v):
            for i in range(0, len(v), 2 * h):
                a, b = v[i:i + h].copy(), v[i + h:i + 2 * h].copy()
                v[i:i + h], v[i + h:i + 2 * h] = a + b, a - b
            h *= 2
        c = v / 2 ** n
        return {m: float(c[m]) for m in range(2 ** n) if abs(c[m]) > 1e-12}


def _bits(n):
    x = np.arange(2 ** n)
    return (x[:, None] >> np.arange(n)[None, :]) & 1


def _parity_values(qubits, n: int) -> np.ndarray:
    """±1 parity of `qubits` for every outcome x = 0 … 2^n − 1, in O(2^n) memory (no 2^n × n bit table)."""
    x = np.arange(2 ** n, dtype=np.int64)
    par = np.zeros(2 ** n, dtype=np.int8)
    for q in qubits:
        par ^= ((x >> int(q)) & 1).astype(np.int8)
    return 1.0 - 2.0 * par


def z_string(qubits, n: int) -> Observable:
    v = _parity_values(qubits, n)
    lab = " ".join("Z%d" % q for q in qubits)
    tex = "".join("Z_{%d}" % q for q in qubits)
    return Observable("Z" + "".join(map(str, qubits)), "⟨%s⟩" % lab, r"\langle %s\rangle" % tex, tuple(v.astype(float)))


parity = z_string


def z(q: int, n: int) -> Observable:
    return z_string([q], n)


def projector(target: int, n: int, label: str | None = None) -> Observable:
    v = np.zeros(2 ** n)
    v[target] = 1.0
    bs = format(target, "0%db" % n)
    return Observable("P" + bs, label or "P(%s)" % bs, r"P(%s)" % bs, tuple(v))


def subset_projector(qubits, value: int, n: int, label: str | None = None) -> Observable:
    """P(bits on `qubits` read `value`), qubits[0] least significant."""
    b = _bits(n)
    got = sum(b[:, q] << i for i, q in enumerate(qubits))
    v = (got == value).astype(float)
    bs = format(value, "0%db" % len(qubits))
    return Observable("P%s@%s" % (bs, "".join(map(str, qubits))), label or "P(%s)" % bs,
                      r"P_{q%s}(%s)" % (",".join(map(str, qubits)), bs), tuple(v))


def magnetization(n: int) -> Observable:
    x = np.arange(2 ** n, dtype=np.int64)
    s = np.zeros(2 ** n, dtype=np.int64)
    for q in range(n):
        s += 1 - 2 * ((x >> q) & 1)
    v = s / n
    return Observable("M", "⟨M⟩ = ⟨Σ Zi⟩/n", r"\langle M\rangle=\tfrac1n\sum_i\langle Z_i\rangle", tuple(v))


# ------------------------------------------------------------------ the UI's catalogue, built lazily
BYTES_PER_VALUE = 32                 # one entry of Observable.values: a tuple slot plus a Python float


def dense_ok(n: int) -> bool:
    """Whether a diagonal observable of n qubits (2^n Python floats) fits comfortably in free memory."""
    from .resources import available_bytes
    n = int(n)
    return n <= 40 and BYTES_PER_VALUE * 2 ** n <= min(0.05 * available_bytes(), 512 * 2 ** 20)


def catalogue_entries(n: int) -> list:
    """[(key, label, qubits or None for the magnetization)] of the observables offered for n qubits — names only,
    nothing of size 2^n is built (the UI lists them on every rerun)."""
    n = int(n)
    out = [("Z%d" % q, "⟨Z%d⟩" % q, (q,)) for q in range(n)]
    out += [(_cat_key((q, q + 1), n), "⟨Z%d Z%d⟩" % (q, q + 1), (q, q + 1)) for q in range(n - 1)]
    if n > 2:
        qs = tuple(range(n))
        out.append((_cat_key(qs, n), "⟨%s⟩" % " ".join("Z%d" % q for q in qs), qs))
    out.append(("M", "⟨M⟩ = ⟨Σ Zi⟩/n", None))
    return out


def _cat_key(qubits, n: int) -> str:
    """Catalogue key of a Z string: "Z01" as before while every index is one digit; with ten or more qubits the
    indices are separated ("Z1.2"), so the pair Z1·Z2 and the single qubit Z12 never share a key."""
    return "Z" + ("".join if n <= 10 else ".".join)(str(q) for q in qubits)


def catalogue_labels(n: int) -> dict:
    """{key: label} of the catalogue (cheap)."""
    return {k: lab for k, lab, _ in catalogue_entries(n)}


def catalogue_item(n: int, key: str):
    """One catalogue observable. A register too large for 2^n stored values gets the same observable as a Pauli
    sum (measured exactly the same way, without the dense vector). Raises ValueError for an unknown key or an
    observable that cannot be represented."""
    n = int(n)
    ent = next((e for e in catalogue_entries(n) if e[0] == key), None)
    if ent is None:
        raise ValueError("unknown observable %r for %d qubits" % (str(key)[:40], n))
    import dataclasses
    _, label, qs = ent
    if dense_ok(n):
        o = magnetization(n) if qs is None else z_string(list(qs), n)
    else:
        from .hamiltonian import HamiltonianError, parse_pauli_sum
        try:
            if qs is None:
                o = parse_pauli_sum(" + ".join("%r*Z%d" % (1.0 / n, q) for q in range(n)), n)
            else:
                o = parse_pauli_sum("*".join("Z%d" % q for q in qs), n)
        except HamiltonianError as e:
            raise ValueError("%s is too large to represent for %d qubits (%s)" % (label, n, e)) from None
    return o if o.key == key else dataclasses.replace(o, key=key)


def default_observable(n: int):
    """The observable a register of n qubits starts with (⟨Z0⟩), dense when it fits in memory."""
    return catalogue_item(n, "Z0")


def catalogue(n: int) -> dict:
    """Observables offered in the UI for n qubits (every one built; prefer catalogue_labels/catalogue_item)."""
    return {k: catalogue_item(n, k) for k, _, _ in catalogue_entries(n)}


def expval(o: np.ndarray, p: np.ndarray) -> float:
    return float(np.asarray(o) @ np.asarray(p))


def shot_var(o: np.ndarray, p: np.ndarray) -> float:
    m = o @ p
    return max(float((o * o) @ p - m * m), 0.0)
