r"""Ready-made circuits for the builder. Every template returns a :class:`Template`:
the circuit, a default observable and a one-line TeX description.

QFT conventions follow Qiskit: :math:`F|j\rangle=2^{-n/2}\sum_k e^{2\pi i jk/2^n}|k\rangle`
with qubit 0 the least significant bit. The textbook circuit is a body :math:`Q_0` of
Hadamards and controlled phases followed by a SWAP network :math:`R` (bit reversal):
:math:`F=R\,Q_0`. Three ways to handle :math:`R`:

``swap``        run the SWAP network (3 CX each after compilation)
``bitreverse``  no SWAPs; prepare :math:`F^\dagger|\mathrm{rev}(s)\rangle` instead of :math:`F^\dagger|s\rangle`.
                Since :math:`Q_0=R\,F`, the body gives :math:`Q_0F^\dagger|\mathrm{rev}(s)\rangle=R\,|\mathrm{rev}(s)\rangle=|s\rangle`
``relabel``     no SWAPs; read the output bits in reverse order (classical relabeling)
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from dataclasses import dataclass, field

from .circuit import CircuitSpec, Op
from . import observables as OBS
from . import qec as QEC
from .resources import qiskit_serial


@dataclass
class Template:
    spec: CircuitSpec
    observable: "OBS.Observable"
    tex: str
    notes: str = ""
    symmetry: tuple | None = None          # (qubits, eigenvalue) of a Z-parity the ideal state satisfies
    relabel: bool = False                  # read output bits in reverse order
    variational: tuple = ()                # live parameters: dicts name, tex, value, slots ((op, factor), …), lo, hi
    checks: tuple = ()                     # encoded algorithms: ((label, qubits), …), Z-parities of measured bits = +1
    logical: float | None = None           # encoded algorithms: ideal (error-free) value of the decoded observable
    qec: dict | None = None                # encoded algorithms: code, gate table with fault-tolerance flags, op ranges


def _g(name, *qs, params=(), tag=""):
    return Op(name, tuple(qs), tuple(params), "gate", tag)


def rev(s: int, n: int) -> int:
    return int(format(s, "0%db" % n)[::-1], 2)


def fourier_prep(n: int, s: int, qubits=None) -> list:
    """Product state F^dagger|s>: qubit q (weight 2^q) in (|0> + e^{-2 pi i s 2^q / 2^n}|1>)/sqrt 2."""
    qubits = list(range(n)) if qubits is None else list(qubits)
    ops = []
    for i, q in enumerate(qubits):
        ops.append(_g("h", q, tag="prep"))
        ang = -2 * math.pi * s * (2 ** i) / 2 ** n
        ang = math.remainder(ang, 2 * math.pi)
        if abs(ang) > 1e-12:
            ops.append(_g("p", q, params=(ang,), tag="prep"))
    return ops


def qft_body(qubits, m: int | None = None, inverse: bool = False) -> list:
    """Q_0 (no SWAPs) on `qubits` (qubits[0] least significant); AQFT keeps controlled phases
    between qubits at distance d <= m."""
    n = len(qubits)
    ops = []
    for j in reversed(range(n)):
        ops.append(_g("h", qubits[j]))
        for k in reversed(range(j)):
            d = j - k
            if m is not None and d > m:
                continue
            ops.append(_g("cp", qubits[k], qubits[j], params=(math.pi / 2 ** d,)))
    if inverse:
        ops = [Op(o.name, o.qubits, tuple(-p for p in o.params), "gate", o.tag) for o in reversed(ops)]
    return ops


def swap_network(qubits) -> list:
    n = len(qubits)
    return [_g("swap", qubits[i], qubits[n - 1 - i]) for i in range(n // 2)]


# ---------------------------------------------------------------- templates
def bell() -> Template:
    spec = CircuitSpec(2, (_g("h", 0), _g("cx", 0, 1)), "Bell")
    return Template(spec, OBS.parity([0, 1], 2), r"|\Phi^+\rangle=\tfrac{1}{\sqrt2}(|00\rangle+|11\rangle),"
                    r"\ \ \langle Z_0Z_1\rangle=1", symmetry=((0, 1), +1))


def ghz(n: int = 3) -> Template:
    ops = [_g("h", 0)] + [_g("cx", i, i + 1) for i in range(n - 1)]
    return Template(CircuitSpec(n, tuple(ops), "GHZ-%d" % n), OBS.parity([0, n - 1], n),
                    r"\tfrac{1}{\sqrt2}(|0\rangle^{\otimes %d}+|1\rangle^{\otimes %d}),\ \ \langle Z_0Z_{%d}\rangle=1"
                    % (n, n, n - 1), symmetry=((0, 1), +1))


def _qrange(a: int, b: int) -> str:
    """'q3' for one qubit, 'q3…q5' for a range."""
    return "q%d" % a if a == b else "q%d…q%d" % (a, b)


QFT_PREPS = ("fourier", "basis")
QFT_PREP_LABELS = {"fourier": "Fourier state F†|s⟩ (known answer |s⟩)", "basis": "basis state |s⟩"}


def basis_prep(n: int, x: int, qubits=None) -> list:
    """X gates (tagged ``prep``) that prepare the computational-basis state |x⟩, x read little-endian (q0 = 1s bit)."""
    qubits = list(range(n)) if qubits is None else list(qubits)
    return [_g("x", q, tag="prep") for i, q in enumerate(qubits) if (int(x) >> i) & 1]


def qft(n: int = 3, s: int = 1, mode: str = "swap", m: int | None = None, prep: str = "fourier") -> Template:
    """The QFT F|y⟩ = N^{-1/2} Σ_k e^{2πi yk/N}|k⟩ (N = 2^n, little-endian integers) applied to a prepared input.

    Input and output. ``prep`` and ``s`` define the *input*: its preparation gates are part of the circuit (tagged
    ``prep``, drawn as preparation). The output is derived from them and never edited:

    * ``prep="fourier"`` (known-answer benchmark, the default and the only form before 1.5): the product state
      F†|s⟩ (a Hadamard and a phase on every qubit; F†|0⟩ = |+⟩^⊗n) is prepared, so the ideal output is |s⟩ with
      probability 1. Adapted from the QFT(2) benchmark of Lubinski et al. (arXiv:2110.03137v3, App. A.2, Fig. 26),
      which prepares the Fourier state of x and applies the *inverse* QFT.
    * ``prep="basis"``: |s⟩ is prepared with X gates. The ideal output F|s⟩ is a uniform distribution over all N
      outcomes (probability 1/N each) whose input-dependent phases e^{2πi sk/N} are invisible in the Z basis.

    ``mode`` is the bit-order convention (SWAP network, bit-reversed input load, or classical relabeling)."""
    if prep not in QFT_PREPS:
        raise ValueError("prep must be one of %s, got %r" % (", ".join(QFT_PREPS), prep))
    if prep == "basis":
        return _qft_basis_input(n, s, mode, m)
    q = list(range(n))
    if mode == "swap":
        ops = fourier_prep(n, s) + qft_body(q, m) + swap_network(q)
        target, relabel = s, False
        note = "SWAP network included: %d SWAPs." % (n // 2)
    elif mode == "bitreverse":
        ops = fourier_prep(n, rev(s, n)) + qft_body(q, m)
        target, relabel = s, False
        note = "No SWAPs: the Fourier state of rev(s) = %d is loaded so the output is |s⟩ directly." % rev(s, n)
        if m:
            note += (" With the cutoff m the approximation error depends on the input, and this mode feeds the "
                     "truncated body the Fourier state of rev(s): its success probability differs from the SWAP and "
                     "relabel modes, which approximate F on the Fourier state of s.")
    elif mode == "relabel":
        ops = fourier_prep(n, s) + qft_body(q, m)
        target, relabel = s, True
        note = "No SWAPs: the device outputs |rev(s)⟩ = |%s⟩, relabeled classically." % format(rev(s, n), "0%db" % n)
    else:
        raise ValueError(mode)
    name = ("AQFT(m=%d)" % m if m else "QFT") + "-%d" % n
    src = rev(s, n) if mode == "bitreverse" else s
    tex = (r"F\,F^\dagger|%d\rangle=|%d\rangle" % (s, s)) if not m else \
          (r"\text{AQFT}_{m=%d}\,F^\dagger|%d\rangle\approx|%d\rangle" % (m, src, s))
    return Template(CircuitSpec(n, tuple(ops), name), OBS.projector(target, n, label="P(%s)" % format(s, "0%db" % n)),
                    tex, note, relabel=relabel)


def _qft_basis_input(n: int, x: int, mode: str, m: int | None) -> Template:
    q = list(range(n))
    if mode == "bitreverse":
        raise ValueError("the bit-reversed input load is defined for the Fourier-state preparation; with a basis-state "
                         "input use the SWAP network or classical relabeling")
    if mode not in ("swap", "relabel"):
        raise ValueError(mode)
    ops = basis_prep(n, x) + qft_body(q, m) + (swap_network(q) if mode == "swap" else [])
    name = ("AQFT(m=%d)" % m if m else "QFT") + "-%d" % n
    N = 2 ** n
    tex = (r"F|%d\rangle=\tfrac{1}{\sqrt{%d}}\textstyle\sum_k e^{2\pi i\,%d k/%d}|k\rangle" % (x, N, x, N))
    note = ("Input |%d⟩ = |%s⟩ (X gates). Ideal output: every outcome with probability 1/%d; the input sets only the "
            "phases e^{2πi·%d·k/%d}, which a Z-basis measurement cannot see (measure qubits in X or Y to probe them). "
            "Observable: P(0…0) = 1/%d." % (x, format(x, "0%db" % n), N, x, N, N))
    if m:
        note += " With the cutoff m the output is only approximately uniform."
    if mode == "relabel":
        note += " No SWAPs: the output bits are relabeled classically."
    return Template(CircuitSpec(n, tuple(ops), name), OBS.projector(0, n, label="P(%s)" % ("0" * n)), tex, note,
                    relabel=mode == "relabel")


def qpe(t: int = 3, phi: float = 0.25) -> Template:
    """Counting qubits 0..t-1, eigenstate qubit t (|1> of P(2 pi phi))."""
    n = t + 1
    ops = [_g("x", t, tag="prep")] + [_g("h", k) for k in range(t)]
    for k in range(t):
        ang = math.remainder(2 * math.pi * phi * 2 ** k, 2 * math.pi)
        if abs(ang) > 1e-12:
            ops.append(_g("cp", k, t, params=(ang,)))
    q = list(range(t))
    ops += swap_network(q) + qft_body(q, inverse=True)      # F^dagger = Q_0^dagger R
    j = int(round(phi * 2 ** t)) % 2 ** t
    obs = OBS.subset_projector(list(range(t)), j, n, label="P(j=%d)" % j)
    return Template(CircuitSpec(n, tuple(ops), "QPE-%d" % t), obs,
                    r"U=P(2\pi\varphi),\ \varphi=%g\ \Rightarrow\ j=\varphi\,2^{%d}=%g" % (phi, t, phi * 2 ** t),
                    "Counting register q0…q%d, eigenstate |1⟩ on q%d." % (t - 1, t))


def ising(n: int = 3, J: float = 1.0, h: float = 1.0, dt: float = 0.25, steps: int = 2) -> Template:
    """First-order Trotter steps of H = -J sum Z_i Z_{i+1} - h sum X_i from |0...0>."""
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops.append(_g("rzz", i, i + 1, params=(-2 * J * dt,)))
        for i in range(n):
            ops.append(_g("rx", i, params=(-2 * h * dt,)))
    return Template(CircuitSpec(n, tuple(ops), "Ising-%d" % n), OBS.magnetization(n),
                    r"e^{-iHt},\ H=-J\sum_i Z_iZ_{i+1}-h\sum_i X_i,\ t=%g\ (%d\ \text{Trotter steps})" % (dt * steps, steps))


def grover2() -> Template:
    ops = [_g("h", 0), _g("h", 1), _g("cz", 0, 1), _g("h", 0), _g("h", 1), _g("x", 0), _g("x", 1),
           _g("cz", 0, 1), _g("x", 0), _g("x", 1), _g("h", 0), _g("h", 1)]
    return Template(CircuitSpec(2, tuple(ops), "Grover-2"), OBS.projector(3, 2, label="P(11)"),
                    r"\text{one Grover iteration marks }|11\rangle:\ P(11)=1")


# ---------------------------------------------------------------- algorithms built with Qiskit
APP_BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "p", "cx", "cz", "cp", "swap"]


@qiskit_serial
def _from_qiskit_circuit(qc, name: str) -> CircuitSpec:
    """Decompose a Qiskit circuit into the app's gate set (no routing, no layout change)."""
    from qiskit import transpile
    from .circuit import from_qiskit
    from .compile import QISKIT_LOCK
    with QISKIT_LOCK:
        tq = transpile(qc, basis_gates=APP_BASIS, optimization_level=1, seed_transpiler=3)
        spec, _ = from_qiskit(tq, name)
    return spec


def _mult_mod15(qc, b: int, work: list, ctrl: int | None = None):
    """(Controlled) multiplication by b mod 15 on a 4-qubit work register holding y in 1..14.

    The permutations are the textbook ones (Qiskit textbook, "Shor's algorithm"): multiplying by
    2, 4, 8 cyclically shifts the bits; 7, 11, 13 = -(8, 4, 2) mod 15 add a NOT of all bits; 14 = -1."""
    b %= 15
    if b == 1:
        return
    w = work

    def sw(i, j):
        if ctrl is None:
            qc.swap(w[i], w[j])
        else:
            qc.cswap(ctrl, w[i], w[j])
    if b in (2, 13):
        sw(2, 3); sw(1, 2); sw(0, 1)
    elif b in (7, 8):
        sw(0, 1); sw(1, 2); sw(2, 3)
    elif b in (4, 11):
        sw(1, 3); sw(0, 2)
    if b in (7, 11, 13, 14):
        for q in w:
            if ctrl is None:
                qc.x(q)
            else:
                qc.cx(ctrl, q)


def _mult_mod15_unitary(b: int) -> np.ndarray:
    """The 16×16 permutation |y⟩ → |b·y mod 15⟩ (y < 15), |15⟩ → |15⟩, little-endian on the work register."""
    M = np.zeros((16, 16))
    for y in range(16):
        M[(b * y) % 15 if y < 15 else 15, y] = 1.0
    return M


@qiskit_serial
def shor15(a: int = 7, t: int = 3, powers: str = "precomputed", multiplier: str = "permutation") -> Template:
    """Order finding for N = 15 (the quantum part of Shor's algorithm) with t counting qubits
    q0..q(t-1) and a 4-qubit work register q(t)..q(t+3) prepared in |1>.

    powers = "precomputed": controlled-U^(2^k) is one multiplication by a^(2^k) mod 15 (a "compiled" circuit, standard
    for demonstrations, e.g. Vandersypen et al., Nature 414, 883 (2001)); "repeated": U_a is applied 2^k times, as a
    circuit that does not know the answer would (2^t − 1 controlled multiplications).
    multiplier = "permutation": the textbook SWAP/NOT network; "synthesized": the controlled 5-qubit permutation
    unitary decomposed by Qiskit's generic unitary synthesis (a deep circuit with hundreds of CX per call)."""
    from qiskit import QuantumCircuit
    from math import gcd
    if a not in (2, 4, 7, 8, 11, 13, 14) or gcd(a, 15) != 1:
        raise ValueError("a must be coprime to 15 and not 1: 2, 4, 7, 8, 11, 13 or 14")
    n = t + 4
    work = list(range(t, t + 4))
    qc = QuantumCircuit(n)
    qc.x(work[0])
    for k in range(t):
        qc.h(k)
    gates = {}

    def cmult(b, ctrl):
        if multiplier == "synthesized":
            if b % 15 == 1:
                return
            if b not in gates:
                from qiskit.circuit.library import UnitaryGate
                gates[b] = UnitaryGate(_mult_mod15_unitary(b), label="×%d mod 15" % b).control(1)
            qc.append(gates[b], [ctrl] + work)
        else:
            _mult_mod15(qc, b, work, ctrl=ctrl)
    for k in range(t):
        if powers == "repeated":
            for _ in range(2 ** k):
                cmult(a, k)
        else:
            cmult(pow(a, 2 ** k, 15), k)
    # inverse QFT on the counting register (with its SWAPs): outcome y = s 2^t / r
    for i in range(t // 2):
        qc.swap(i, t - 1 - i)
    for j in range(t):
        for k in range(j):
            qc.cp(-math.pi / 2 ** (j - k), k, j)
        qc.h(j)
    spec = _from_qiskit_circuit(qc, "Shor-15 (a=%d%s)" % (a, "" if powers == "precomputed" and multiplier == "permutation"
                                                          else ", " + ", ".join(x for x in (powers, multiplier)
                                                                                if x not in ("precomputed", "permutation"))))
    r = next(r for r in range(1, 16) if pow(a, r, 15) == 1)
    good = [y for y in range(2 ** t) if _cf_order(y, t, a) == r and y != 0]
    b = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
    y = sum(b[:, q] << q for q in range(t))
    v = np.isin(y, good).astype(float)
    obs = OBS.Observable("shor_ok", "P(y gives r=%d)" % r, r"P(y\in\{%s\})" % ",".join(map(str, good)), tuple(v))
    ideal_p = len(good) / r if (2 ** t) % r == 0 else float("nan")
    return Template(spec, obs,
                    r"a^r\equiv1\ (\mathrm{mod}\ 15):\ a=%d,\ r=%d;\ \ y=s\,2^{%d}/r" % (a, r, t),
                    "Counting register q0…q%d, work register q%d…q%d = |1⟩. Outcomes y whose continued fraction "
                    "gives r = %d: %s%s. %s powers, %s multiplier: %d CX-type gates before compilation." % (
                        t - 1, t, t + 3, r, good, "" if not np.isfinite(ideal_p) else " (ideal probability %.3g)" % ideal_p,
                        powers.capitalize(), multiplier, spec.count("2q")))


def _cf_order(y: int, t: int, a: int) -> int:
    """Smallest r found from y / 2^t by continued fractions such that a^r = 1 mod 15 (0 if none)."""
    from fractions import Fraction
    if y == 0:
        return 0
    f = Fraction(y, 2 ** t).limit_denominator(15)
    for m in range(1, 5):
        r = f.denominator * m
        if r <= 15 and pow(a, r, 15) == 1:
            return r
    return 0


@qiskit_serial
def hhl2() -> Template:
    """HHL for the 2x2 system A = [[1, -1/3], [-1/3, 1]], |b> = |0> (Cao et al.; Qiskit textbook example).

    q0: |b> / solution, q1 q2: clock (eigenvalues 2/3, 4/3 -> 1, 2 with t = 3pi/4), q3: ancilla.
    With ancilla = 1 the system qubit holds |x> ∝ A^{-1}|b> = (3|0> + |1>)/sqrt(10)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from scipy.linalg import expm
    A = np.array([[1, -1 / 3], [-1 / 3, 1]])
    t0 = 3 * math.pi / 4
    qc = QuantumCircuit(4)
    b, c1, c2, anc = 0, 1, 2, 3

    def phase_est(inv=False):
        ops = []
        ops.append(("h", c1)); ops.append(("h", c2))
        ops.append(("cu", c1, 1)); ops.append(("cu", c2, 2))
        ops.append(("iqft",))
        if inv:
            ops = ops[::-1]
        for op in ops:
            if op[0] == "h":
                qc.h(op[1])
            elif op[0] == "cu":
                U = expm(1j * A * t0 * op[2] * (-1 if inv else 1))
                qc.append(UnitaryGate(U).control(1), [op[1], b])
            else:
                # 2-qubit (inverse) QFT on (c1, c2), c1 least significant
                if not inv:
                    qc.swap(c1, c2); qc.h(c1); qc.cp(-math.pi / 2, c1, c2); qc.h(c2)
                else:
                    qc.h(c2); qc.cp(math.pi / 2, c1, c2); qc.h(c1); qc.swap(c1, c2)
    phase_est()
    # eigenvalue register holds 1 (c1) or 2 (c2): rotate the ancilla by 2 arcsin(C / lambda), C = 1
    qc.cry(math.pi, c1, anc)
    qc.cry(2 * math.asin(0.5), c2, anc)
    phase_est(inv=True)
    spec = _from_qiskit_circuit(qc, "HHL 2x2")
    bits = (np.arange(16)[:, None] >> np.arange(4)[None, :]) & 1
    v = bits[:, anc] * (1 - 2 * bits[:, b])                      # |1><1|_anc ⊗ Z_b
    obs = OBS.Observable("hhl_anc_zb", "⟨P₁(anc) Z_b⟩", r"\langle |1\rangle\!\langle1|_{\rm anc}\otimes Z_b\rangle",
                         tuple(v.astype(float)))
    return Template(spec, obs,
                    r"A=\begin{pmatrix}1&-\tfrac13\\-\tfrac13&1\end{pmatrix},\ |b\rangle=|0\rangle\ \Rightarrow\ "
                    r"|x\rangle\propto A^{-1}|b\rangle=\tfrac{3|0\rangle+|1\rangle}{\sqrt{10}}",
                    "q0 system, q1 q2 clock, q3 ancilla. The observable is P(anc=1)·⟨Z_b⟩ given anc=1; divide by "
                    "P(anc=1) (observable P(1) on q3) for ⟨x|Z|x⟩ = 0.8, or post-select with symmetry verification "
                    "on q3 (parity −1).", symmetry=((anc,), -1))


def simon(n: int = 2, s: int = 3) -> Template:
    """Simon's algorithm, the hidden-subgroup problem over Z_2^n: f(x) = f(x ⊕ s).

    Input q0..q(n-1), output q(n)..q(2n-1). Every measured y satisfies y·s = 0 (mod 2), so the
    parity of the input bits selected by s is +1: a symmetry that verification can post-select on."""
    if not 1 <= s < 2 ** n:
        raise ValueError("the secret s must be in 1..2^n-1")
    ops = [_g("h", i) for i in range(n)]
    ops += [_g("cx", i, n + i) for i in range(n)]                     # copy x
    j = next(i for i in range(n) if (s >> i) & 1)                     # a set bit of s
    ops += [_g("cx", j, n + i) for i in range(n) if (s >> i) & 1]      # f(x) = x ⊕ (x_j s)
    ops += [_g("h", i) for i in range(n)]
    S = [i for i in range(n) if (s >> i) & 1]
    obs = OBS.parity(S, 2 * n)
    sb = format(s, "0%db" % n)
    return Template(CircuitSpec(2 * n, tuple(ops), "Simon (s=%s)" % sb), obs,
                    r"f(x)=f(x\oplus s),\ s=%s:\ \ y\cdot s=0\ \Rightarrow\ \langle %s\rangle=1"
                    % (sb, "".join("Z_{%d}" % q for q in S)),
                    "Input q0…q%d, output q%d…q%d. Hidden subgroup {0, s}." % (n - 1, n, 2 * n - 1),
                    symmetry=(tuple(S), +1))


# ---------------------------------------------------------------- deep algorithms: Grover search and the HSP
def mcz_ops(qubits) -> list:
    """A multi-controlled Z on `qubits` (symmetric in its qubits) in the app's gate set: CZ for two, the 6-CNOT
    T-gate decomposition of CCZ for three, and Qiskit's synthesis (cached) beyond."""
    qs = list(qubits)
    if len(qs) == 1:
        return [_g("z", qs[0])]
    if len(qs) == 2:
        return [_g("cz", qs[0], qs[1])]
    if len(qs) == 3:
        a, b, c = qs
        return [_g("cx", b, c), _g("tdg", c), _g("cx", a, c), _g("t", c), _g("cx", b, c), _g("tdg", c),
                _g("cx", a, c), _g("t", b), _g("t", c), _g("cx", a, b), _g("t", a), _g("tdg", b), _g("cx", a, b)]
    spec = _mcz_spec(len(qs))
    return [Op(o.name, tuple(qs[q] for q in o.qubits), o.params, o.kind, o.tag) for o in spec.ops]


@lru_cache(maxsize=16)
def _mcz_spec(k: int) -> CircuitSpec:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZGate
    qc = QuantumCircuit(k)
    qc.append(ZGate().control(k - 1), list(range(k)))
    return _from_qiskit_circuit(qc, "MCZ-%d" % k)


def grover_theta(n: int, marked: int = 1) -> float:
    """sin θ = √(M/N): one Grover iteration rotates the state by 2θ toward the marked subspace."""
    return math.asin(math.sqrt(marked / 2 ** n))


def grover_optimal(n: int) -> int:
    """k* = ⌊π/(4θ)⌋, the iteration count that maximizes P(marked) = sin²((2k+1)θ) for one marked item."""
    return max(int(math.floor(math.pi / (4 * grover_theta(n)))), 0)


def grover(n: int = 3, marked: int = 5, iterations: str = "optimal", k: int = 2) -> Template:
    """Grover search for one marked item m among N = 2^n: phase oracle O_m = I − 2|m⟩⟨m| and diffuser
    D = 2|s⟩⟨s| − I, k times. P(m) = sin²((2k+1)θ) with sin θ = 1/√N (Grover 1996; Boyer et al. 1998)."""
    if not 0 <= marked < 2 ** n:
        raise ValueError("the marked item must be in 0..2^n−1")
    kk = grover_optimal(n) if iterations == "optimal" else int(k)
    if kk < 0:
        raise ValueError("the number of iterations must be at least 0")
    q = list(range(n))
    flips = [i for i in q if not (marked >> i) & 1]
    ops = [_g("h", i) for i in q]
    for _ in range(kk):
        ops += [_g("x", i) for i in flips] + mcz_ops(q) + [_g("x", i) for i in flips]            # oracle
        ops += [_g("h", i) for i in q] + [_g("x", i) for i in q] + mcz_ops(q) + [_g("x", i) for i in q] + \
            [_g("h", i) for i in q]                                                                 # diffuser
    th = grover_theta(n)
    p = math.sin((2 * kk + 1) * th) ** 2
    mb = format(marked, "0%db" % n)
    return Template(CircuitSpec(n, tuple(ops), "Grover-%d (k=%d)" % (n, kk)), OBS.projector(marked, n),
                    r"P(%s)=\sin^2\!\big((2k+1)\theta\big)=%.4f,\ \ \sin\theta=1/\sqrt{%d},\ k=%d" % (mb, p, 2 ** n, kk),
                    "%d iteration%s (k* = ⌊π/4θ⌋ = %d is optimal); each iteration is an oracle and a diffuser, two "
                    "%d-qubit controlled-Z gates." % (kk, "" if kk == 1 else "s", grover_optimal(n), n))


def _gf2_rref(rows: list, n: int) -> list:
    """Row-reduced basis (as ints) of the GF(2) span of `rows`."""
    basis = []
    for r in rows:
        for b in basis:
            r = min(r, r ^ b)
        if r:
            basis.append(r)
            basis.sort(reverse=True)
    return basis


def gf2_complement(gens: list, n: int) -> list:
    """A basis of H^⊥ = {y : y·h = 0 (mod 2) for every h in H} for H = span(gens) ⊂ Z_2^n."""
    out = []
    for y in range(1, 2 ** n):
        if all(bin(y & g).count("1") % 2 == 0 for g in gens):
            if len(_gf2_rref(out + [y], n)) > len(_gf2_rref(out, n)):
                out.append(y)
    return out


HSP_GROUPS = ("Z2^n (Simon-type)", "Z_N cyclic, N = 2^n (QFT)")


def hsp(group: str = "Z2^n (Simon-type)", n: int = 3, h1: int = 5, h2: int = 3, j: int = 2) -> Template:
    """The abelian hidden-subgroup problem: f is constant on the cosets of a hidden subgroup H and distinct across
    them; Fourier sampling returns a uniformly random character that is trivial on H.

    * G = Z_2^n: H = span{h1, h2}; f(x) = (a_1·x, …, a_{n−k}·x) with {a_i} a basis of H^⊥, so ker f = H. Hadamards
      are the Fourier transform of Z_2^n; every outcome y lies in H^⊥ (y·h = 0 for every h in H).
    * G = Z_N, N = 2^n: H = ⟨2^j⟩, f(x) = x mod 2^j (the low j bits), the QFT over Z_N; every outcome is a multiple
      of N/2^j — period finding, the core of Shor's algorithm."""
    q_in = list(range(n))
    if group == HSP_GROUPS[0]:
        gens = _gf2_rref([g for g in (int(h1), int(h2)) if g], n)
        if any(g >= 2 ** n for g in (int(h1), int(h2))):
            raise ValueError("the generators must be n-bit integers (below 2^n)")
        if not gens:
            raise ValueError("choose at least one non-zero generator h1")
        dual = gf2_complement(gens, n)
        m = len(dual)
        ops = [_g("h", i) for i in q_in]
        for t, a in enumerate(dual):
            ops += [_g("cx", i, n + t) for i in q_in if (a >> i) & 1]
        ops += [_g("h", i) for i in q_in]
        nt = n + m
        x = np.arange(2 ** nt)
        y = x & (2 ** n - 1)
        ok = np.ones(2 ** nt, bool)
        for g in gens:
            ok &= np.array([bin(int(v) & g).count("1") % 2 == 0 for v in y])
        H = sorted({_xor_comb(gens, cs) for cs in np.ndindex(*([2] * len(gens)))})
        obs = OBS.Observable("hsp_perp", "P(y ∈ H⊥)", r"P(y\in H^\perp)", tuple(ok.astype(float)))
        sym_q = tuple(i for i in q_in if (gens[0] >> i) & 1)
        more = "; ".join(" ".join(str(i) for i in q_in if (g >> i) & 1) for g in gens[1:])
        hs = ", ".join(format(h, "0%db" % n) for h in H)
        return Template(CircuitSpec(nt, tuple(ops), "HSP Z2^%d (dim H = %d)" % (n, len(gens))), obs,
                        r"H=\{%s\},\ \ y\cdot h=0\ \forall h\in H:\ P(y\in H^\perp)=1" % hs,
                        "Input %s, output %s (f(x) = A x, the rows of A a basis of H⊥). |H| = %d, so the "
                        "outcomes are uniform over the %d characters trivial on H." % (
                            _qrange(0, n - 1), _qrange(n, nt - 1), 2 ** len(gens), 2 ** (n - len(gens))),
                        symmetry=(sym_q, +1, more))
    if group == HSP_GROUPS[1]:
        j = int(j)
        if not 1 <= j <= n - 1:
            raise ValueError("choose 1 ≤ j ≤ n − 1 (H = ⟨2^j⟩ ⊂ Z_N)")
        N, d = 2 ** n, 2 ** j
        ops = [_g("h", i) for i in q_in]
        ops += [_g("cx", i, n + i) for i in range(j)]              # f(x) = x mod 2^j: copy the low j bits
        ops += qft_body(q_in) + swap_network(q_in)
        nt = n + j
        x = np.arange(2 ** nt)
        y = x & (N - 1)
        ok = (y % (N // d) == 0).astype(float)
        obs = OBS.Observable("hsp_period", "P(y ≡ 0 mod %d)" % (N // d), r"P(y\equiv0 \bmod %d)" % (N // d),
                             tuple(ok))
        return Template(CircuitSpec(nt, tuple(ops), "HSP Z_%d (H = ⟨%d⟩)" % (N, d)), obs,
                        r"H=\langle %d\rangle\subset\mathbb Z_{%d},\ f(x)=x\bmod %d:\ y\in\{0,%d,\dots\},"
                        r"\ P=1" % (d, N, d, N // d),
                        "Input %s, output %s. The QFT over Z_%d turns the coset state into peaks at the "
                        "multiples of N/d = %d: %d equally likely outcomes." % (_qrange(0, n - 1), _qrange(n, nt - 1),
                                                                               N, N // d, d))
    raise ValueError("unknown group %r" % group)


def _xor_comb(gens, cs) -> int:
    v = 0
    for g, c in zip(gens, cs):
        if c:
            v ^= g
    return v


@qiskit_serial
def qpe_ising(t: int = 3, phase: float = 0.375, state: str = "01") -> Template:
    """Phase estimation of U = exp(-i H τ) with H = J Z_a Z_b on the eigenstate |ab> (a Hamiltonian-simulation
    QPE). ZZ|01> = -|01>, so U|01> = e^{+iJτ}|01>; Jτ is chosen so the phase Jτ/2π equals `phase`.
    Controlled-U^(2^k) = CX · CRZ(2Jτ 2^k) · CX on the system pair (b, a)."""
    from qiskit import QuantumCircuit
    n = t + 2
    sa, sb = t, t + 1
    Jtau = 2 * math.pi * phase
    zz = -1 if state in ("01", "10") else 1
    qc = QuantumCircuit(n)
    for q, bit in ((sa, state[1]), (sb, state[0])):
        if bit == "1":
            qc.x(q)
    for k in range(t):
        qc.h(k)
    for k in range(t):
        ang = 2 * Jtau * 2 ** k
        qc.cx(sa, sb)
        qc.crz(ang, k, sb)
        qc.cx(sa, sb)
    for i in range(t // 2):
        qc.swap(i, t - 1 - i)
    for j in range(t):
        for k in range(j):
            qc.cp(-math.pi / 2 ** (j - k), k, j)
        qc.h(j)
    spec = _from_qiskit_circuit(qc, "QPE of exp(-iJZZτ)")
    ph = (-zz * phase) % 1.0             # eigenphase of U on the chosen eigenstate, in turns
    j = int(round(ph * 2 ** t)) % 2 ** t
    obs = OBS.subset_projector(list(range(t)), j, n, label="P(j=%d)" % j)
    return Template(spec, obs,
                    r"H=J\,Z_aZ_b,\ U=e^{-iH\tau},\ |%s\rangle:\ E=%+d\,J\ \Rightarrow\ j=%d" % (state, zz, j),
                    "Counting q0…q%d, system q%d q%d in |%s⟩ (written q%d q%d, an eigenstate of ZZ). Eigenphase "
                    "%.4g turns." % (t - 1, sa, sb, state, sb, sa, ph))


# ---------------------------------------------------------------- a small statevector for variational angles
def _mat(name: str, params: tuple) -> np.ndarray:
    """Gate matrices in closed form for the parametrized gates (keeps optimization loops out of the
    gate-matrix cache); fixed gates come from Qiskit via ``gate_matrix``."""
    if name in ("rx", "ry", "rz", "p", "rzz", "cp"):
        t = float(params[0])
        c, s = math.cos(t / 2), math.sin(t / 2)
        if name == "rx":
            return np.array([[c, -1j * s], [-1j * s, c]])
        if name == "ry":
            return np.array([[c, -s], [s, c]], complex)
        if name == "rz":
            return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
        if name == "p":
            return np.diag([1, np.exp(1j * t)])
        if name == "rzz":
            a, b = np.exp(-0.5j * t), np.exp(0.5j * t)
            return np.diag([a, b, b, a])
        return np.diag([1, 1, 1, np.exp(1j * t)])
    from .circuit import gate_matrix
    return gate_matrix(name, tuple(params))


def statevector(n: int, ops) -> np.ndarray:
    """Pure-state simulation of the gate operations (noise operations are skipped). Returns amplitudes in
    little-endian order (index x = sum_q b_q 2^q). Used only to choose variational angles for templates."""
    psi = np.zeros((2,) * n, complex)
    psi[(0,) * n] = 1.0
    for o in ops:
        if o.kind != "gate":
            continue
        U = _mat(o.name, o.params)
        k = len(o.qubits)
        ax = [n - 1 - q for q in reversed(o.qubits)]            # tensor axis of the most significant op qubit first
        Ut = U.reshape((2,) * (2 * k))
        psi = np.tensordot(Ut, psi, axes=(list(range(k, 2 * k)), ax))
        psi = np.moveaxis(psi, list(range(k)), ax)
    return psi.reshape(-1)


def _expect(n: int, ops, diag: np.ndarray) -> float:
    from .resources import check
    check()                                          # the session's time limit also bounds angle optimization
    a = statevector(n, ops)
    return float(np.real(np.vdot(a, diag * a)))


def _minimise(f, x0_list, maxiter: int = 400):
    """Best local minimum of f over several starts (SciPy L-BFGS-B, numerical gradients)."""
    from scipy.optimize import minimize
    best = None
    for x0 in x0_list:
        r = minimize(f, np.asarray(x0, float), method="L-BFGS-B", options=dict(maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r
    return best


# ---------------------------------------------------------------- NISQ: variational circuits
def _entangler(n: int, pattern: str) -> list:
    if n < 2:
        return []
    if pattern == "linear":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif pattern == "circular":
        pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    elif pattern == "full":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError("entanglement must be linear, circular or full")
    return [_g("cx", a, b) for a, b in pairs]


def _hea_ops(n: int, layers: int, ent: str, th) -> list:
    """Hardware-efficient ansatz (Kandala et al. 2017): RY·RZ on every qubit, then `layers` × (CX entangler,
    RY·RZ on every qubit). 2n(layers+1) angles."""
    th = list(th)
    ops, k = [], 0
    for layer in range(layers + 1):
        if layer:
            ops += _entangler(n, ent)
        for q in range(n):
            ops.append(_g("ry", q, params=(th[k],)))
            ops.append(_g("rz", q, params=(th[k + 1],)))
            k += 2
    return ops


def zz_chain(n: int) -> OBS.Observable:
    """H = sum_i Z_i Z_{i+1} on an open chain, divided by n-1 (so it lies in [-1, 1]; ground energy -1)."""
    b = OBS._bits(n)
    s = 1 - 2 * b
    v = np.sum(s[:, :-1] * s[:, 1:], axis=1) / max(n - 1, 1)
    return OBS.Observable("HZZ", "⟨H⟩/(n-1), H = Σ Z(i)Z(i+1)", r"\tfrac{1}{n-1}\langle \textstyle\sum_i Z_iZ_{i+1}\rangle",
                          tuple(v.astype(float)))


def hea_vqe(n: int = 4, layers: int = 2, entanglement: str = "linear", angles: str = "VQE-optimized",
            seed: int = 7) -> Template:
    """VQE with a hardware-efficient ansatz for the antiferromagnetic chain H = Σ Z_i Z_{i+1}."""
    npar = 2 * n * (layers + 1)
    rng = np.random.default_rng(seed)
    obs = zz_chain(n)
    diag = obs.o
    if angles == "random":
        th = rng.uniform(0, 2 * math.pi, npar)
        note = "Random angles (seed %d): a generic HEA state." % seed
    else:
        def f(x):
            return _expect(n, _hea_ops(n, layers, entanglement, x), diag)
        starts = [rng.uniform(-0.3, 0.3, npar) + (math.pi / 2)]
        if n <= 8:                                   # a second, random start is cheap for small registers
            starts.append(rng.uniform(0, 2 * math.pi, npar))
        r = _minimise(f, starts)
        th = np.mod(r.x, 2 * math.pi)
        note = ("Angles optimized classically on a noiseless statevector (L-BFGS-B, %d parameters): "
                "E/(n-1) = %.4f, ground state -1." % (npar, r.fun))
    ops = _hea_ops(n, layers, entanglement, th)
    rot = [i for i, o in enumerate(ops) if o.name in ("ry", "rz")]
    var = tuple(dict(name="θ%d" % k, tex=r"\theta_{%d}" % k, value=float(th[k]), slots=((i, 1.0),), lo=0.0,
                     hi=2 * math.pi, group="layer %d" % (k // (2 * n)),
                     what="%s on q%d" % (ops[i].name.upper(), ops[i].qubits[0]))
                for k, i in enumerate(rot))
    return Template(CircuitSpec(n, tuple(ops), "HEA-VQE-%d" % n), obs,
                    r"\begin{gathered}\min_\theta\langle\psi(\theta)|H|\psi(\theta)\rangle,\quad H=\textstyle\sum_i Z_iZ_{i+1}\\ "
                    r"|\psi(\theta)\rangle=\textstyle\prod_{\ell}[R_ZR_Y\,U_{\rm ent}]\,R_ZR_Y|0\rangle^{\otimes %d}\end{gathered}" % n,
                    note + " %d layers, %s entanglement, %d CX." % (layers, entanglement,
                                                                   sum(1 for o in ops if o.name == "cx")),
                    variational=var)


def maxcut_graph(n: int, graph: str, seed: int = 1) -> list:
    if graph == "ring":
        return [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    if graph == "path":
        return [(i, i + 1) for i in range(n - 1)]
    if graph == "3-regular":
        if n % 2 or n < 4:
            raise ValueError("the 3-regular (Möbius ladder) graph needs an even number of nodes ≥ 4")
        return [(i, (i + 1) % n) for i in range(n)] + [(i, i + n // 2) for i in range(n // 2)]
    if graph == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if graph == "random":
        rng = np.random.default_rng(seed)
        e = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
        return e or [(0, 1)]
    raise ValueError("unknown graph")


def _qaoa_ops(n, edges, gammas, betas) -> list:
    ops = [_g("h", q) for q in range(n)]
    for g, b in zip(gammas, betas):
        ops += [_g("rzz", i, j, params=(-g,)) for i, j in edges]      # e^{-iγC}, C = Σ (1 - Z_i Z_j)/2
        ops += [_g("rx", q, params=(2 * b,)) for q in range(n)]      # e^{-iβ Σ X}
    return ops


def qaoa_maxcut(n: int = 4, graph: str = "ring", p: int = 1, angles: str = "optimized", gamma: float = 0.6,
                beta: float = 0.4, seed: int = 1) -> Template:
    """QAOA (Farhi, Goldstone, Gutmann 2014) for MaxCut; the observable is the approximation ratio
    ⟨C⟩/C_max with C(x) the number of cut edges."""
    edges = maxcut_graph(n, graph, seed)
    b = OBS._bits(n)
    cut = np.zeros(2 ** n)
    for i, j in edges:
        cut += b[:, i] ^ b[:, j]
    cmax = float(cut.max())
    obs = OBS.Observable("cut_ratio", "⟨C⟩/C_max (MaxCut)", r"\langle C\rangle/C_{\max}", tuple(cut / cmax))
    if angles == "optimized":
        diag = -cut / cmax

        def f(x):
            return _expect(n, _qaoa_ops(n, edges, x[:p], x[p:]), diag)
        ramp = [(l + 0.5) / p for l in range(p)]
        starts = [np.r_[[0.8 * r for r in ramp], [0.6 * (1 - r) for r in ramp]]]
        rng = np.random.default_rng(seed)
        starts += [np.r_[rng.uniform(0, math.pi, p), rng.uniform(0, math.pi / 2, p)] for _ in range(3)]
        r = _minimise(f, starts)
        gs, bs = list(r.x[:p]), list(r.x[p:])
        note = "Angles optimized on a noiseless statevector: ratio %.4f (random guessing gives %.3f)." % (
            -r.fun, float(np.mean(cut)) / cmax)
    else:
        gs, bs = [gamma] * p, [beta] * p
        note = "Fixed angles γ = %.3f, β = %.3f in every layer." % (gamma, beta)
    ops = _qaoa_ops(n, edges, gs, bs)
    E = len(edges)
    var = []
    for l in range(p):
        base = n + l * (E + n)
        g = (gs[l] + math.pi) % (2 * math.pi) - math.pi              # e^{-iγC} has period 2π in γ
        b_ = (bs[l] + math.pi / 2) % math.pi - math.pi / 2            # e^{-iβB} has period π in β
        var.append(dict(name="γ%d" % (l + 1), tex=r"\gamma_{%d}" % (l + 1), value=float(g), lo=-math.pi, hi=math.pi,
                        slots=tuple((base + e, -1.0) for e in range(E)), group="layer %d" % (l + 1),
                        what="cost layer: RZZ(−γ) on every edge"))
        var.append(dict(name="β%d" % (l + 1), tex=r"\beta_{%d}" % (l + 1), value=float(b_), lo=-math.pi / 2,
                        hi=math.pi / 2, slots=tuple((base + E + q, 2.0) for q in range(n)), group="layer %d" % (l + 1),
                        what="mixer layer: RX(2β) on every qubit"))
    return Template(CircuitSpec(n, tuple(ops), "QAOA-MaxCut-%d" % n), obs,
                    r"\begin{gathered}|\gamma,\beta\rangle=\textstyle\prod_{\ell=1}^{%d}e^{-i\beta_\ell B}e^{-i\gamma_\ell C}"
                    r"|+\rangle^{\otimes %d}\\ C=\textstyle\sum_{(i,j)\in E}\tfrac{1-Z_iZ_j}{2},\quad B=\sum_i X_i"
                    r"\end{gathered}" % (p, n),
                    note + " Graph: %s, %d edges, C_max = %d. γ = (%s), β = (%s)." % (
                        graph, len(edges), int(cmax), ", ".join("%.3f" % g for g in gs), ", ".join("%.3f" % x for x in bs)),
                    variational=tuple(var))


def qml_zz(n: int = 3, fm_reps: int = 2, ansatz_reps: int = 2, data_seed: int = 0, weight_seed: int = 1) -> Template:
    """A quantum-kernel style classifier: the ZZ feature map (Havlíček et al., Nature 567, 209 (2019)) followed by a
    RealAmplitudes-type variational layer; the model output is ⟨Z⊗n⟩."""
    x = np.random.default_rng(data_seed).uniform(0, 2 * math.pi, n)
    w = np.random.default_rng(weight_seed).uniform(0, 2 * math.pi, n * (ansatz_reps + 1))
    ops = []
    for _ in range(fm_reps):
        ops += [_g("h", q) for q in range(n)]
        ops += [_g("p", q, params=(math.remainder(2 * x[q], 2 * math.pi),)) for q in range(n)]
        for i in range(n - 1):
            j = i + 1
            ang = math.remainder(2 * (math.pi - x[i]) * (math.pi - x[j]), 2 * math.pi)
            ops += [_g("cx", i, j), _g("p", j, params=(ang,)), _g("cx", i, j)]
    k = 0
    for r in range(ansatz_reps + 1):
        if r:
            ops += [_g("cx", i, i + 1) for i in range(n - 1)]
        for q in range(n):
            ops.append(_g("ry", q, params=(w[k],)))
            k += 1
    obs = OBS.z_string(list(range(n)), n)
    return Template(CircuitSpec(n, tuple(ops), "QML ZZ-map classifier"), obs,
                    r"\begin{gathered}f(x)=\langle 0|U_\Phi^\dagger W^\dagger Z^{\otimes %d}W\,U_\Phi|0\rangle\\ "
                    r"U_\Phi(x)=\big[e^{-i\sum_{i}\phi_{i,i+1}(x)Z_iZ_{i+1}}e^{-i\sum_i x_iZ_i}H^{\otimes n}\big]^{%d}"
                    r"\end{gathered}" % (n, fm_reps),
                    "Data point x = (%s) (seed %d); weights from seed %d (an untrained model: for QEM the circuit "
                    "structure is what matters). Feature map ×%d, ansatz ×%d."
                    % (", ".join("%.2f" % v for v in x), data_seed, weight_seed, fm_reps, ansatz_reps))


def qml_reupload(n: int = 2, layers: int = 3, data_seed: int = 0, weight_seed: int = 1) -> Template:
    """Data re-uploading classifier (Pérez-Salinas et al., Quantum 4, 226 (2020)): every layer encodes the data
    again, RY(w·x + b) RZ(w'·x + b') on each qubit, followed by a CZ chain; output ⟨Z0⟩."""
    rng_x, rng_w = np.random.default_rng(data_seed), np.random.default_rng(weight_seed)
    x = rng_x.uniform(-1, 1, n)
    W = rng_w.normal(0, 1.5, (layers, n, 4))
    ops = []
    for l in range(layers):
        for q in range(n):
            a = W[l, q, 0] * x[q] + W[l, q, 1]
            b = W[l, q, 2] * x[q] + W[l, q, 3]
            ops += [_g("ry", q, params=(math.remainder(a, 2 * math.pi),)),
                    _g("rz", q, params=(math.remainder(b, 2 * math.pi),))]
        if n > 1 and l < layers - 1:
            ops += [_g("cz", i, i + 1) for i in range(n - 1)]
    return Template(CircuitSpec(n, tuple(ops), "QML re-uploading"), OBS.z(0, n),
                    r"\begin{gathered}f(x)=\langle Z_0\rangle\ \text{on}\\ \textstyle\prod_{\ell=1}^{%d}\big[\mathrm{CZ}\ \bigotimes_q "
                    r"R_Z(w'_{\ell q}x_q+b'_{\ell q})R_Y(w_{\ell q}x_q+b_{\ell q})\big]|0\rangle\end{gathered}" % layers,
                    "Data x = (%s) (seed %d), weights from seed %d; %d layers re-upload the data."
                    % (", ".join("%.2f" % v for v in x), data_seed, weight_seed, layers))


def bernstein_vazirani(n: int = 4, s: int = 11) -> Template:
    """Bernstein–Vazirani: one query of f(x) = s·x (mod 2) reveals s. Input q0..q(n-1), ancilla q(n) in |−⟩."""
    if not 0 <= s < 2 ** n:
        raise ValueError("s must be an n-bit integer")
    a = n
    ops = [_g("x", a, tag="prep"), _g("h", a, tag="prep")] + [_g("h", q) for q in range(n)]
    ops += [_g("cx", q, a) for q in range(n) if (s >> q) & 1]
    ops += [_g("h", q) for q in range(n)]
    sb = format(s, "0%db" % n)
    return Template(CircuitSpec(n + 1, tuple(ops), "Bernstein-Vazirani (s=%s)" % sb),
                    OBS.subset_projector(list(range(n)), s, n + 1, label="P(s=%s)" % sb),
                    r"f(x)=s\cdot x\ (\mathrm{mod}\ 2),\ s=%s:\ \ H^{\otimes n}U_fH^{\otimes n}|0\rangle|-\rangle=|s\rangle|-\rangle" % sb,
                    "Input q0…q%d, ancilla q%d prepared in |−⟩ (phase kickback)." % (n - 1, n))


# ---------------------------------------------------------------- FTQC: error-correction primitives
def _majority(bits: np.ndarray) -> np.ndarray:
    return (2 * bits.sum(axis=1) > bits.shape[1]).astype(int)


@qiskit_serial
def qec_3q(code: str = "bit-flip", mode: str = "syndrome + correction", theta: float = 1.0472,
           error: str = "none", k: int = 0, p: float = 0.1) -> Template:
    """Three-qubit repetition codes (Nielsen & Chuang §10.1). Logical state RY(θ)|0⟩ on q0, encoded into q0 q1 q2.

    bit-flip:   |0_L⟩ = |000⟩, |1_L⟩ = |111⟩, corrects one X error.
    phase-flip: the same code in the Hadamard basis, corrects one Z error.
    Modes: ancillas q3 q4 measure Z0Z1 and Z1Z2 and Toffoli gates correct coherently, then the code is decoded
    so that ⟨Z0⟩ = cos θ; or the same without the correction; or no ancillas and a majority vote of the three
    measured data bits (the classical decoder)."""
    if code not in ("bit-flip", "phase-flip"):
        raise ValueError("code must be bit-flip or phase-flip")
    from qiskit import QuantumCircuit
    anc = mode != "majority vote readout"
    n = 5 if anc else 3
    qc = QuantumCircuit(n)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    if code == "phase-flip":
        for q in range(3):
            qc.h(q)
    spec_pre = _from_qiskit_circuit(qc, "enc")
    ops = list(spec_pre.ops)
    err_gate = "x" if code == "bit-flip" else "z"
    if error == "one error on qubit k":
        ops.append(_g(err_gate, int(k)))
    elif error == "channel on every data qubit":
        ops += [Op("bitflip" if code == "bit-flip" else "phaseflip", (q,), (float(p),), "noise") for q in range(3)]
    qc2 = QuantumCircuit(n)
    if code == "phase-flip":
        for q in range(3):
            qc2.h(q)
    if anc:
        qc2.cx(0, 3); qc2.cx(1, 3); qc2.cx(1, 4); qc2.cx(2, 4)
        if mode == "syndrome + correction":
            qc2.x(4); qc2.ccx(3, 4, 0); qc2.x(4)            # syndrome 10 -> q0
            qc2.ccx(3, 4, 1)                                  # syndrome 11 -> q1
            qc2.x(3); qc2.ccx(3, 4, 2); qc2.x(3)            # syndrome 01 -> q2
        qc2.cx(0, 2)
        qc2.cx(0, 1)
    ops += list(_from_qiskit_circuit(qc2, "dec").ops)
    name = "%s code (%s)" % (code, "majority vote" if not anc else "corrected" if mode == "syndrome + correction"
                              else "detect")
    if anc:
        obs = OBS.z(0, n)
        sym = ((3,), +1) if mode == "syndrome only (no correction)" else None
    else:
        b = OBS._bits(3)
        v = 1 - 2 * _majority(b)
        obs = OBS.Observable("maj3", "⟨Z_L⟩ (majority vote)", r"\langle Z_L\rangle_{\rm maj}", tuple(v.astype(float)))
        sym = None
    return Template(CircuitSpec(n, tuple(ops), name), obs,
                    (r"|\psi_L\rangle=\cos\tfrac\theta2|{+}{+}{+}\rangle+\sin\tfrac\theta2|{-}{-}{-}\rangle" if code == "phase-flip"
                     else r"|\psi_L\rangle=\cos\tfrac\theta2|000\rangle+\sin\tfrac\theta2|111\rangle")
                    + r",\ \ \langle Z_L\rangle=\cos\theta=%.4f" % math.cos(theta),
                    ("Data q0 q1 q2" + (", syndrome ancillas q3 (Z0Z1) and q4 (Z1Z2)" if anc else "") + ". "
                     + {"syndrome + correction": "Toffoli gates apply the correction the syndrome points to; any single "
                                                 "%s error is undone." % err_gate.upper(),
                        "syndrome only (no correction)": "The syndrome is extracted but not used; symmetry verification "
                                                         "post-selects on Z0Z1 = +1 (ancilla q3 = 0).",
                        "majority vote readout": "No ancillas: the three data bits are measured and decoded by majority "
                                                 "vote."}.get(mode, "")
                     + " Error: %s." % (error if error == "none" else "%s on q%d" % (err_gate.upper(), k)
                                        if error.startswith("one") else "%s channel, p = %g" % (
                                            "bit-flip" if code == "bit-flip" else "phase-flip", p))),
                    symmetry=sym)


def repetition(d: int = 3, rounds: int = 1, theta: float = 0.0, p: float = 0.05) -> Template:
    """Distance-d bit-flip repetition memory with `rounds` of syndrome extraction into fresh ancillas (no mid-circuit
    measurement or reset), a bit-flip channel p on every data qubit before each round and once more at the end
    (rounds + 1 exposures), and a majority-vote decoder on the final data readout. Qubits: data 0..d-1, then the
    ancillas of round r at d + r(d-1) .. . The r+1 channels compose to one flip with p_eff = (1 - (1 - 2p)^(r+1))/2."""
    if d < 3 or d % 2 == 0:
        raise ValueError("the distance d must be odd and at least 3")
    n = d + rounds * (d - 1)
    ops = []
    if theta:
        ops.append(_g("ry", 0, params=(theta,)))
    ops += [_g("cx", 0, i) for i in range(1, d)]
    for r in range(rounds):
        if p > 0:
            ops += [Op("bitflip", (q,), (float(p),), "noise") for q in range(d)]
        base = d + r * (d - 1)
        for i in range(d - 1):
            ops += [_g("cx", i, base + i), _g("cx", i + 1, base + i)]
    if p > 0:
        ops += [Op("bitflip", (q,), (float(p),), "noise") for q in range(d)]
    b = OBS._bits(n)[:, :d]
    v = 1 - 2 * _majority(b)
    obs = OBS.Observable("rep_maj", "⟨Z_L⟩ (majority of %d)" % d, r"\langle Z_L\rangle_{\rm maj}", tuple(v.astype(float)))
    peff = (1 - (1 - 2 * p) ** (rounds + 1)) / 2              # r + 1 bit-flip channels in sequence
    pl = sum(math.comb(d, j) * peff ** j * (1 - peff) ** (d - j) for j in range((d + 1) // 2, d + 1))
    return Template(CircuitSpec(n, tuple(ops), "Repetition code d=%d" % d), obs,
                    r"\begin{gathered}|0_L\rangle=|0\rangle^{\otimes %d},\ \ p_L=\textstyle\sum_{j>d/2}\binom{d}{j}"
                    r"p_{\rm eff}^j(1-p_{\rm eff})^{d-j}\\ p_{\rm eff}=\tfrac12\bigl[1-(1-2p)^{r+1}\bigr]\end{gathered}" % d,
                    "Data q0…q%d, %d round(s) of Z(i)Z(i+1) checks%s. The bit-flip channels (p = %g, "
                    "applied %d times: before each round and before readout) are part of the circuit: each data qubit "
                    "ends flipped with p_eff = %.3g, and a majority vote of the %d data bits fails with p_L = %.2e (one "
                    "bare qubit: %.2e), before the gate noise of the checks is added." % (
                        d - 1, rounds, (" into ancillas " + _qrange(d, n - 1)) if n > d else "", p, rounds + 1, peff,
                        d, pl, peff),
                    symmetry=((0, 1), +1))


STEANE_H = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]])


def steane_decode(bits: np.ndarray) -> np.ndarray:
    """Classical decoding of a transversal Z readout: the syndrome H·x names the flipped qubit (q + 1 in binary)."""
    s = (bits @ STEANE_H.T) % 2
    pos = 4 * s[:, 0] + 2 * s[:, 1] + s[:, 2] - 1
    out = bits.copy()
    idx = np.where(pos >= 0)[0]
    out[idx, pos[idx]] ^= 1
    return out


def steane(theta: float = 1.0472, error: str = "none", k: int = 0, p: float = 0.05,
           readout: str = "decoded") -> Template:
    """Steane [[7,1,3]] code: encoder for RY(θ)|0⟩ and a transversal logical-Z readout.

    Stabilizers are X and Z on the rows of the Hamming check matrix: {3,4,5,6}, {1,2,5,6}, {0,2,4,6}. The input sits
    on q2; CX q2→q4, q5 applies the logical X representative X2X4X5, then the pivots q0 q1 q3 (H) spread the
    codewords of the dual code. ⟨Z_L⟩ = ⟨Z⊗7⟩ = cos θ; the decoded readout first corrects any single bit flip."""
    ops = [_g("ry", 2, params=(theta,))] if theta else []
    ops += [_g("cx", 2, 4), _g("cx", 2, 5)]
    ops += [_g("h", q) for q in (0, 1, 3)]
    ops += [_g("cx", 3, t) for t in (4, 5, 6)] + [_g("cx", 1, t) for t in (2, 5, 6)] + [_g("cx", 0, t) for t in (2, 4, 6)]
    if error == "X on qubit k":
        ops.append(_g("x", int(k)))
    elif error == "Z on qubit k":
        ops.append(_g("z", int(k)))
    elif error == "bit-flip channel on every qubit":
        ops += [Op("bitflip", (q,), (float(p),), "noise") for q in range(7)]
    b = OBS._bits(7)
    if readout == "decoded":
        v = 1 - 2 * (steane_decode(b).sum(axis=1) % 2)
        obs = OBS.Observable("steane_dec", "⟨Z_L⟩ decoded", r"\langle Z_L\rangle_{\rm decoded}", tuple(v.astype(float)))
    else:
        obs = OBS.z_string(list(range(7)), 7)
    return Template(CircuitSpec(7, tuple(ops), "Steane [[7,1,3]]"), obs,
                    r"\begin{gathered}|\psi_L\rangle=\cos\tfrac\theta2|0_L\rangle+\sin\tfrac\theta2|1_L\rangle,\quad "
                    r"|0_L\rangle=\tfrac{1}{\sqrt8}\textstyle\sum_{c\in\mathcal C^{\perp}_{\rm Hamming}}|c\rangle\\ "
                    r"\langle Z_L\rangle=\cos\theta=%.4f\end{gathered}" % math.cos(theta),
                    "Transversal readout of all 7 qubits. %s Symmetry verification post-selects on the stabilizer "
                    "Z3Z4Z5Z6 = +1. Error: %s." % (
                        "The decoded observable corrects one flipped bit using the Hamming syndrome, so any single X "
                        "error leaves ⟨Z_L⟩ unchanged." if readout == "decoded" else
                        "Raw parity: one X error flips the sign of ⟨Z_L⟩.",
                        error if not error.endswith("qubit k") else error.replace("qubit k", "q%d" % k)),
                    symmetry=((3, 4, 5, 6), +1))


# ---------------------------------------------------------------- the five-qubit perfect code
FIVE_STABILISERS = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")        # character j acts on qubit j
# A Clifford encoder U with U Z_0 U† = Z_L = ZZZZZ, U X_0 U† = X_L = XXXXX and U Z_j U† = S_j (j = 1..4),
# synthesized once from that stabilizer tableau (qiskit Clifford.to_circuit, then optimized) and checked in the tests.
FIVE_ENCODER = (("h", 1), ("h", 3), ("s", 4), ("cx", 0, 4), ("cx", 1, 0), ("h", 1), ("cx", 2, 1), ("s", 1), ("h", 1),
                ("cx", 3, 0), ("x", 0), ("cx", 3, 4), ("cx", 4, 2), ("cx", 2, 3), ("y", 2), ("s", 3), ("h", 3),
                ("cx", 1, 3), ("h", 4), ("cx", 3, 4), ("s", 3), ("cx", 4, 1), ("h", 1), ("cx", 3, 1), ("s", 1),
                ("h", 1), ("x", 3))
_INVERSE = {"s": "sdg", "sdg": "s", "h": "h", "x": "x", "y": "y", "z": "z", "cx": "cx"}


def _five_ops(inverse: bool = False) -> list:
    seq = reversed(FIVE_ENCODER) if inverse else FIVE_ENCODER
    return [_g(_INVERSE[g[0]] if inverse else g[0], *g[1:]) for g in seq]


def _pauli_matrix(label: str) -> np.ndarray:
    """Little-endian 2^n matrix of a Pauli string whose character j acts on qubit j."""
    P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1.0, -1.0])}
    m = np.array([[1.0 + 0j]])
    for ch in label:                               # qubit 0 first: kron(later, earlier)
        m = np.kron(P[ch], m)
    return m


@lru_cache(maxsize=1)
def five_qubit_decoder_table() -> dict:
    """Syndrome (bits of q1..q4 after the inverse encoder) -> whether the logical Z readout on q0 must be flipped.
    Built from U† E U for the 16 errors {I} ∪ {X, Y, Z on one qubit}: the code is perfect, every syndrome is used
    exactly once."""
    from .circuit import gate_matrix
    U = np.eye(32, dtype=complex)
    for o in _five_ops():
        U = _embed_gate(gate_matrix(o.name, o.params), o.qubits, 5) @ U
    table = {}
    errors = ["IIIII"] + ["".join(p if j == k else "I" for j in range(5)) for k in range(5) for p in "XYZ"]
    zs = [_pauli_matrix("".join("Z" if j == q else "I" for j in range(5))) for q in range(5)]
    for e in errors:
        M = U.conj().T @ _pauli_matrix(e) @ U           # a Pauli (up to phase): read its X part qubit by qubit
        xbit = [not np.allclose(M @ Z, Z @ M) for Z in zs]
        table[tuple(int(x) for x in xbit[1:])] = bool(xbit[0])
    return table


def _embed_gate(G: np.ndarray, qubits, n: int) -> np.ndarray:
    k = len(qubits)
    T = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for col in range(2 ** n):
        sub = sum(((col >> q) & 1) << i for i, q in enumerate(qubits))
        rest = col & ~sum(1 << q for q in qubits)
        for out in range(2 ** k):
            amp = G[out, sub]
            if amp == 0:
                continue
            row = rest | sum(((out >> i) & 1) << q for i, q in enumerate(qubits))
            T[row, col] += amp
    return T


def five_qubit(theta: float = 1.0472, error: str = "none", k: int = 0, p: float = 0.05,
               readout: str = "decoded (lookup table)") -> Template:
    """Laflamme's [[5,1,3]] code (Laflamme, Miquel, Paz, Zurek, PRL 77, 198 (1996); stabilizers from Bennett et al.,
    PRA 54, 3824 (1996)): the smallest code that corrects any single-qubit error. RY(θ)|0⟩ on q0 is encoded, an error
    acts, the inverse encoder maps the error to (logical Pauli on q0) ⊗ (syndrome on q1…q4), and the readout decodes
    with the 16-entry lookup table."""
    ops = [_g("ry", 0, params=(theta,))] if theta else []
    ops += _five_ops()
    if error.endswith("on qubit k"):
        ops.append(_g(error[0].lower(), int(k)))
    elif error.startswith("depolarizing"):
        ops += [Op("depol", (q,), (float(p),), "noise") for q in range(5)]
    ops += _five_ops(inverse=True)
    b = OBS._bits(5)
    if readout.startswith("decoded"):
        table = five_qubit_decoder_table()
        flip = np.array([table[tuple(int(x) for x in row[1:5])] for row in b], dtype=int)
        v = 1 - 2 * ((b[:, 0] + flip) % 2)
        obs = OBS.Observable("five_dec", "⟨Z_L⟩ decoded (5-qubit code)", r"\langle Z_L\rangle_{\rm decoded}",
                             tuple(v.astype(float)))
    else:
        obs = OBS.z(0, 5)
    ncx = sum(1 for o in ops if o.name == "cx")
    return Template(CircuitSpec(5, tuple(ops), "Five-qubit code [[5,1,3]]"), obs,
                    r"\begin{gathered}S=\{XZZXI,\,IXZZX,\,XIXZZ,\,ZXIXZ\},\ \ Z_L=Z^{\otimes5},\ X_L=X^{\otimes5}\\ "
                    r"\langle Z_L\rangle=\cos\theta=%.4f\end{gathered}" % math.cos(theta),
                    "Encode (Clifford, %d CX), error, then the inverse encoder: q0 carries the logical qubit, q1…q4 the "
                    "syndrome (the four stabilizer outcomes). %s Error: %s." % (
                        ncx // 2, "The decoded observable flips q0 when the syndrome names an error that acts as X or Y "
                                  "on the logical qubit, so any single X, Y or Z error leaves ⟨Z_L⟩ = cos θ."
                        if readout.startswith("decoded") else "Raw readout of q0: single errors are not corrected.",
                        error.replace("qubit k", "q%d" % k) if error.endswith("qubit k") else
                        ("depolarizing p = %g on every qubit" % p if error.startswith("depolarizing") else error)))


# ---------------------------------------------------------------- the smallest surface code
def surface_d2(theta: float = 1.0472, error: str = "none", k: int = 0, p: float = 0.05,
               syndrome: str = "transversal readout") -> Template:
    """Distance-2 rotated surface code [[4,1,2]]: one weight-4 X plaquette X0X1X2X3 and two weight-2 Z boundary checks
    Z0Z1, Z2Z3 on a 2×2 patch (q0 q1 / q2 q3). Logical Z_L = Z0Z2, X_L = X0X1. Distance 2: every single-qubit error is
    *detected*, none can be corrected — the regime where post-selection (symmetry verification) is the decoder.

    |0_L⟩ = (|0000⟩ + |1111⟩)/√2, |1_L⟩ = (|1100⟩ + |0011⟩)/√2 (bits q0 q1 q2 q3). With ancillas, q4 measures the X
    plaquette and q5, q6 the two Z checks."""
    anc = syndrome.startswith("ancilla")
    n = 7 if anc else 4
    ops = [_g("ry", 0, params=(theta,))] if theta else []
    ops += [_g("cx", 0, 1), _g("h", 2), _g("cx", 2, 3), _g("cx", 2, 0), _g("cx", 2, 1)]
    if error.endswith("on qubit k"):
        ops.append(_g(error[0].lower(), int(k)))
    elif error.startswith("bit-flip"):
        ops += [Op("bitflip", (q,), (float(p),), "noise") for q in range(4)]
    if anc:
        ops += [_g("h", 4)] + [_g("cx", 4, q) for q in range(4)] + [_g("h", 4)]          # X0X1X2X3 -> q4
        ops += [_g("cx", 0, 5), _g("cx", 1, 5), _g("cx", 2, 6), _g("cx", 3, 6)]        # Z0Z1 -> q5, Z2Z3 -> q6
    obs = OBS.z_string([0, 2], n)
    sym = ((5,), +1, "6") if anc else ((0, 1), +1, "2 3")         # every Z check must read +1
    return Template(CircuitSpec(n, tuple(ops), "Surface code d=2 (one plaquette)"), obs,
                    r"\begin{gathered}S=\{X_0X_1X_2X_3,\ Z_0Z_1,\ Z_2Z_3\},\ \ Z_L=Z_0Z_2,\ X_L=X_0X_1\\ "
                    r"|0_L\rangle=\tfrac{|0000\rangle+|1111\rangle}{\sqrt2},\ \ \langle Z_L\rangle=\cos\theta=%.4f"
                    r"\end{gathered}" % math.cos(theta),
                    "Data q0 q1 (top row), q2 q3 (bottom row)%s. Z_L = Z0Z2 is read from the data bits. A single X or "
                    "Y error flips %s, which symmetry verification uses to discard the shot (error detection with "
                    "both Z checks); Z errors do not change Z_L%s. Error: %s." % (
                        "; ancillas q4 (X plaquette), q5 (Z0Z1), q6 (Z2Z3)" if anc else "",
                        "one of the Z-check ancillas q5, q6" if anc else "one of the checks Z0Z1, Z2Z3",
                        " but flip the X-plaquette ancilla q4" if anc else "",
                        error.replace("qubit k", "q%d" % k) if error.endswith("qubit k") else
                        ("bit-flip p = %g on every data qubit" % p if error.startswith("bit-flip") else error)),
                    symmetry=sym)


# ---------------------------------------------------------------- benchmark algorithms on small codes
def qec_algorithm(code: str = "4-2-2", algorithm: str = "Grover-2", theta: float = 1.0472, s: int = 3,
                  oracle: str = "balanced x", logical_basis: str = "Z", decoder: str = "majority / lookup",
                  error: str = "none", k: int = 0, where: str = "before readout") -> Template:
    """A benchmark written as a logical circuit (state preparation, Bell, GHZ, Deutsch–Jozsa, Bernstein–Vazirani, a
    non-Clifford logical rotation, Grover or QFT on two logical qubits) compiled onto blocks of a small code
    (qemx/qec.py): encoder with the logical inputs injected, transversal logical gates where the code has them and
    explicitly non-fault-tolerant Pauli-rotation gadgets elsewhere, transversal readout. The observable is the decoded
    logical value (majority vote / lookup table, or the raw logical parity); ``symmetry`` holds every Z check, so
    symmetry verification is error detection. ``notes`` state what is and is not fault-tolerant."""
    b = QEC.build(code, algorithm, theta, s, oracle, logical_basis, decoder, error, k, where)
    return Template(b.spec, b.observable, b.tex, b.notes, symmetry=b.symmetry, checks=b.checks, logical=b.logical,
                    qec=b.info)


# ---------------------------------------------------------------- registry with parameter schemas
@dataclass(frozen=True)
class TParam:
    name: str
    label: str
    kind: str                  # int | float | choice
    default: object
    lo: object = None          # number or callable(params) -> number
    hi: object = None
    step: float | None = None
    options: tuple = ()
    help: str = ""

    def bound(self, v, params):
        return v(params) if callable(v) else v


QFT_MODES = ("swap", "bitreverse", "relabel")
_QFT_S_HELP = ("Part of the INPUT, read little-endian (q0 = 1s bit): which state the preparation gates load. The output "
               "is derived from it (Output panel): |s⟩ for the Fourier-state preparation, a uniform distribution for "
               "a basis-state input.")
_QFT_PREP_HELP = ("fourier: prepare F†|s⟩ with a Hadamard and a phase per qubit (F†|0⟩ = |+⟩^n), a known-answer "
                  "benchmark whose ideal output is |s⟩. basis: prepare |s⟩ with X gates; the ideal output is uniform. "
                  "Configurations from before 1.5 have no prep and mean fourier.")
_BITS = {"swap": "SWAP network", "bitreverse": "reversed input", "relabel": "relabel outputs"}

REGISTRY = {
    "bell": ("Bell pair", bell, ()),
    "ghz": ("GHZ state", ghz, (TParam("n", "qubits", "int", 3, 2, None, 1),)),
    "qft": ("QFT", qft, (TParam("n", "qubits", "int", 3, 2, None, 1),
                         TParam("s", "input index s (integer)", "int", 5, 0, lambda p: 2 ** p["n"] - 1, 1,
                                help=_QFT_S_HELP),
                         TParam("mode", "bit order", "choice", "swap", options=QFT_MODES,
                                help="SWAP network; no SWAPs with a bit-reversed input load (Fourier-state input "
                                     "only); or no SWAPs and classically relabeled output bits."),
                         TParam("prep", "input preparation", "choice", "fourier", options=QFT_PREPS,
                                help=_QFT_PREP_HELP))),
    "aqft": ("Approximate QFT", lambda n=4, s=1, mode="swap", m=1, prep="fourier": qft(n, s, mode, m, prep),
             (TParam("n", "qubits", "int", 4, 3, None, 1),
              TParam("s", "input index s (integer)", "int", 1, 0, lambda p: 2 ** p["n"] - 1, 1, help=_QFT_S_HELP),
              TParam("mode", "bit order", "choice", "swap", options=QFT_MODES),
              TParam("m", "AQFT degree m", "int", 1, 1, lambda p: p["n"] - 1, 1,
                     help="Keep the controlled phases π/2^d between qubits at distance d ≤ m. Coppersmith (1994, "
                          "arXiv:quant-ph/0201067v1) deletes those with d ≥ m_C, so this m = m_C − 1; his bound on "
                          "each entry's phase error is 2π·n·2^(−m_C)."),
              TParam("prep", "input preparation", "choice", "fourier", options=QFT_PREPS, help=_QFT_PREP_HELP))),
    "qpe": ("Phase estimation", qpe, (TParam("t", "counting qubits", "int", 3, 1, None, 1),
                                      TParam("phi", "phase φ", "float", 0.25, 0.0, 1.0, 1 / 16))),
    "qpe_ising": ("QPE of an Ising Hamiltonian", qpe_ising,
                  (TParam("t", "counting qubits", "int", 3, 1, None, 1),
                   TParam("phase", "Jτ/2π", "float", 0.375, 0.0, 1.0, 1 / 16),
                   TParam("state", "eigenstate", "choice", "01", options=("00", "01", "10", "11")))),
    "ising": ("Ising (Trotter)", ising, (TParam("n", "spins", "int", 3, 2, None, 1),
                                         TParam("steps", "Trotter steps", "int", 2, 1, None, 1),
                                         TParam("h", "field h", "float", 1.0, 0.0, 5.0, 0.1),
                                         TParam("dt", "time step", "float", 0.25, 0.01, 2.0, 0.05))),
    "shor": ("Shor order finding, N = 15", shor15,
             (TParam("a", "base a", "choice", 7, options=(2, 4, 7, 8, 11, 13, 14)),
              TParam("t", "counting qubits", "int", 3, 2, None, 1),
              TParam("powers", "powers U^(2^k)", "choice", "precomputed", options=("precomputed", "repeated"),
                     help="precomputed: one multiplication by a^(2^k) mod 15; repeated: 2^k multiplications by a "
                          "(2^t − 1 in all, a much deeper circuit)."),
              TParam("multiplier", "multiplier", "choice", "permutation", options=("permutation", "synthesized"),
                     help="permutation: the textbook SWAP/NOT network; synthesized: generic unitary synthesis of the "
                          "controlled multiplication (hundreds of CX each)."))),
    "hhl": ("HHL, 2×2 system", hhl2, ()),
    "grover_search": ("Grover search", grover,
                      (TParam("n", "qubits (N = 2^n items)", "int", 3, 2, None, 1),
                       TParam("marked", "marked item m (integer)", "int", 5, 0, lambda p: 2 ** p["n"] - 1, 1,
                              help="Read little-endian (q0 = 1s bit), like every bitstring in the app."),
                       TParam("iterations", "iterations", "choice", "optimal", options=("optimal", "custom"),
                              help="optimal: k* = ⌊π/(4θ)⌋ with sin θ = 1/√N; custom: the k below."),
                       TParam("k", "k (custom)", "int", 2, 0, None, 1))),
    "hsp": ("Hidden subgroup problem (abelian)", hsp,
            (TParam("group", "group G", "choice", HSP_GROUPS[0], options=HSP_GROUPS,
                    help="Z2^n: Simon-type, Hadamard transform; Z_N: cyclic, QFT (period finding)."),
             TParam("n", "input qubits n", "int", 3, 2, None, 1),
             TParam("h1", "generator h1 (Z2^n)", "int", 5, 1, lambda p: 2 ** p["n"] - 1, 1),
             TParam("h2", "generator h2 (Z2^n; 0 = none)", "int", 3, 0, lambda p: 2 ** p["n"] - 1, 1),
             TParam("j", "H = ⟨2^j⟩ (Z_N)", "int", 2, 1, lambda p: max(p["n"] - 1, 1), 1))),
    "simon": ("Simon (hidden subgroup)", simon, (TParam("n", "input qubits", "int", 2, 1, None, 1),
                                                 TParam("s", "secret s", "int", 3, 1, lambda p: 2 ** p["n"] - 1, 1))),
    "grover": ("Grover, 2 qubits", grover2, ()),
    "bv": ("Bernstein–Vazirani", bernstein_vazirani,
           (TParam("n", "input qubits", "int", 4, 1, None, 1),
            TParam("s", "secret s (integer)", "int", 11, 0, lambda p: 2 ** p["n"] - 1, 1))),
    # NISQ: variational
    "hea_vqe": ("VQE · hardware-efficient ansatz", hea_vqe,
                (TParam("n", "qubits", "int", 4, 2, None, 1),
                 TParam("layers", "entangling layers", "int", 2, 0, None, 1),
                 TParam("entanglement", "entanglement", "choice", "linear", options=("linear", "circular", "full")),
                 TParam("angles", "angles", "choice", "VQE-optimized", options=("VQE-optimized", "random"),
                        help="VQE-optimized: minimize ⟨H⟩ on a noiseless statevector when the template loads "
                             "(cost grows as 2^n); random: seeded angles."),
                 TParam("seed", "seed", "int", 7, 0, None, 1))),
    "qaoa": ("QAOA · MaxCut", qaoa_maxcut,
             (TParam("n", "nodes (qubits)", "int", 4, 2, None, 1),
              TParam("graph", "graph", "choice", "ring", options=("ring", "path", "3-regular", "complete", "random"),
                     help="3-regular is the Möbius ladder (needs an even n ≥ 4); random is G(n, 1/2) from the seed."),
              TParam("p", "QAOA depth p", "int", 1, 1, None, 1),
              TParam("angles", "angles", "choice", "optimized", options=("optimized", "fixed")),
              TParam("gamma", "γ (fixed)", "float", 0.6, -math.pi, math.pi, 0.05),
              TParam("beta", "β (fixed)", "float", 0.4, -math.pi, math.pi, 0.05),
              TParam("seed", "seed", "int", 1, 0, None, 1))),
    "qml_zz": ("QML · ZZ feature map classifier", qml_zz,
               (TParam("n", "qubits = features", "int", 3, 2, None, 1),
                TParam("fm_reps", "feature-map repetitions", "int", 2, 1, None, 1),
                TParam("ansatz_reps", "ansatz repetitions", "int", 2, 0, None, 1),
                TParam("data_seed", "data seed", "int", 0, 0, None, 1),
                TParam("weight_seed", "weight seed", "int", 1, 0, None, 1))),
    "qml_reupload": ("QML · data re-uploading", qml_reupload,
                     (TParam("n", "qubits = features", "int", 2, 1, None, 1),
                      TParam("layers", "layers", "int", 3, 1, None, 1),
                      TParam("data_seed", "data seed", "int", 0, 0, None, 1),
                      TParam("weight_seed", "weight seed", "int", 1, 0, None, 1))),
    # FTQC: error correction
    "qec_3q": ("3-qubit bit-/phase-flip code", qec_3q,
               (TParam("code", "code", "choice", "bit-flip", options=("bit-flip", "phase-flip")),
                TParam("mode", "decoder", "choice", "syndrome + correction",
                       options=("syndrome + correction", "syndrome only (no correction)", "majority vote readout")),
                TParam("theta", "logical θ (RY)", "float", 1.0472, 0.0, 2 * math.pi, 0.05),
                TParam("error", "error", "choice", "none",
                       options=("none", "one error on qubit k", "channel on every data qubit")),
                TParam("k", "qubit k", "int", 0, 0, 2, 1),
                TParam("p", "channel p", "float", 0.1, 0.0, 1.0, 0.01))),
    "repetition": ("Repetition code memory (distance d)", repetition,
                   (TParam("d", "distance d (odd)", "int", 3, 3, None, 2),
                    TParam("rounds", "syndrome rounds", "int", 1, 0, None, 1),
                    TParam("theta", "logical θ (RY)", "float", 0.0, 0.0, 2 * math.pi, 0.05),
                    TParam("p", "bit-flip p per exposure (r + 1 exposures)", "float", 0.05, 0.0, 1.0, 0.01))),
    "steane": ("Steane [[7,1,3]] code", steane,
               (TParam("theta", "logical θ (RY)", "float", 1.0472, 0.0, 2 * math.pi, 0.05),
                TParam("error", "error", "choice", "none",
                       options=("none", "X on qubit k", "Z on qubit k", "bit-flip channel on every qubit")),
                TParam("k", "qubit k", "int", 0, 0, 6, 1),
                TParam("p", "channel p", "float", 0.05, 0.0, 1.0, 0.01),
                TParam("readout", "logical readout", "choice", "decoded", options=("decoded", "raw parity")))),
    "five_qubit": ("Five-qubit code [[5,1,3]] (Laflamme)", five_qubit,
                   (TParam("theta", "logical θ (RY)", "float", 1.0472, 0.0, 2 * math.pi, 0.05),
                    TParam("error", "error", "choice", "none",
                           options=("none", "X on qubit k", "Y on qubit k", "Z on qubit k",
                                    "depolarizing channel on every qubit")),
                    TParam("k", "qubit k", "int", 0, 0, 4, 1),
                    TParam("p", "channel p", "float", 0.05, 0.0, 1.0, 0.01),
                    TParam("readout", "logical readout", "choice", "decoded (lookup table)",
                           options=("decoded (lookup table)", "raw (no correction)")))),
    "surface_d2": ("Surface code d = 2 (one plaquette)", surface_d2,
                   (TParam("theta", "logical θ (RY)", "float", 1.0472, 0.0, 2 * math.pi, 0.05),
                    TParam("error", "error", "choice", "none",
                           options=("none", "X on qubit k", "Y on qubit k", "Z on qubit k",
                                    "bit-flip channel on every data qubit")),
                    TParam("k", "data qubit k", "int", 0, 0, 3, 1),
                    TParam("p", "channel p", "float", 0.05, 0.0, 1.0, 0.01),
                    TParam("syndrome", "syndrome", "choice", "transversal readout",
                           options=("transversal readout", "ancilla extraction (3 ancillas)"),
                           help="Transversal: 4 qubits, checks computed from the data bits. Ancillas: 7 qubits, "
                                "the X plaquette and both Z checks measured on q4, q5, q6."))),
    # FTQC: benchmark algorithms on small codes (qemx/qec.py)
    "qec_algorithm": ("Encoded algorithm (code × benchmark)", qec_algorithm,
                      (TParam("code", "code", "choice", "4-2-2", options=QEC.CODE_KEYS,
                              help="bit-flip / phase-flip: the 3-qubit repetition codes (one logical qubit per 3 "
                                   "qubits; correct one X resp. Z error). 4-2-2: the [[4,2,2]] code (two logical "
                                   "qubits on 4; detects one error, corrects none). 5-1-3: the [[5,1,3]] code (one "
                                   "logical qubit on 5; corrects any one error). At most %d qubits." % QEC.MAX_QUBITS),
                       TParam("algorithm", "benchmark", "choice", "Grover-2", options=QEC.ALGORITHMS,
                              help="The logical circuit: state prep RY(θ) (1 logical qubit), Bell (2), GHZ-3 (3), "
                                   "Deutsch-Jozsa (input + ancilla), Bernstein-Vazirani (2 inputs, phase oracle), "
                                   "RX rotation RX_L(θ)|0_L⟩ and RZ Ramsey H_L RZ_L(θ)|+_L⟩ (non-Clifford for "
                                   "learning-based mitigation), Grover-2 and QFT-2 (2 logical qubits)."),
                       TParam("theta", "logical θ (prep, RX, RZ)", "float", 1.0472, 0.0, 2 * math.pi, 0.05,
                              help="Angle of state prep, RX rotation and RZ Ramsey. Away from multiples of π/2 the "
                                   "rotation is non-Clifford, so Clifford data regression has something to replace."),
                       TParam("s", "s: secret / marked / output", "int", 3, 0, 3, 1,
                              help="Bernstein-Vazirani secret, Grover marked item, QFT output: two logical bits, "
                                   "L0 = the 1s bit (little-endian like every bitstring in the app)."),
                       TParam("oracle", "Deutsch-Jozsa oracle", "choice", "balanced x", options=QEC.ORACLES,
                              help="f(x) for the 1-bit Deutsch-Jozsa query U_f|x, y⟩ = |x, y ⊕ f(x)⟩."),
                       TParam("logical_basis", "logical readout basis", "choice", "Z", options=("Z", "X"),
                              help="Z_L or X_L, for state prep, Bell and GHZ-3 (the others read Z_L). The repetition "
                                   "codes have no checks in their X_L readout: it is unprotected."),
                       TParam("decoder", "decoder", "choice", QEC.DECODERS[0], options=QEC.DECODERS,
                              help="majority / lookup: the measured syndrome selects a correction (majority vote for "
                                   "the repetition codes, a 16-entry table for 5-1-3; 4-2-2 cannot correct). raw "
                                   "logical parity: the logical operator read from the data bits, no correction."),
                       TParam("error", "injected error", "choice", "none", options=QEC.ERRORS),
                       TParam("k", "data qubit k", "int", 0, 0,
                              lambda p: QEC.num_qubits(p.get("code"), p.get("algorithm")) - 1, 1),
                       TParam("where", "error position", "choice", QEC.WHERE[0], options=QEC.WHERE,
                              help="before readout: after the last logical gate; after encoding: before the first, so "
                                   "the error passes through every logical gate (fault-tolerant ones keep it "
                                   "correctable or detectable, the others need not)."))),
}

CATEGORIES = {
    "Basics": ("bell", "ghz", "grover"),
    "Algorithms": ("qft", "aqft", "qpe", "qpe_ising", "ising", "bv", "simon"),
    "NISQ · variational": ("hea_vqe", "qaoa", "qml_zz", "qml_reupload"),
    "FTQC · error correction": ("qec_3q", "repetition", "surface_d2", "five_qubit", "steane"),
    "FTQC · encoded algorithms": ("qec_algorithm",),
    "FTQC · deep algorithms": ("shor", "grover_search", "hsp", "hhl"),
}

# encoded-algorithm templates: their syndrome checks and intended logical value come from the template itself
QEC_ALGORITHM_KEYS = ("qec_algorithm",)
QEC_PRESETS = tuple((name, "qec_algorithm", dict(prm)) for name, prm in QEC.PRESETS.items())


def qec_checks(key: str | None, params: dict | None = None) -> list:
    """[(label, qubits)]: the checks of an encoded-algorithm template, Z-parities of measured bits that read +1 without
    error (the format of insight.qec_checks); [] for other templates or parameters that do not build."""
    if key not in QEC_ALGORITHM_KEYS:
        return []
    try:
        return list(build(key, params or {}).checks)
    except ValueError:
        return []


def intended_logical(key: str | None, params: dict | None = None) -> float | None:
    """The error-free value of an encoded-algorithm template's decoded observable (the reference for the logical error
    p_L = (1 − ⟨O⟩/⟨O⟩_ideal)/2); None for other templates or parameters that do not build."""
    if key not in QEC_ALGORITHM_KEYS:
        return None
    try:
        return build(key, params or {}).logical
    except ValueError:
        return None


def category_of(key: str) -> str:
    return next((c for c, ks in CATEGORIES.items() if key in ks), "Basics")


def defaults(key: str) -> dict:
    return {p.name: p.default for p in REGISTRY[key][2]}


# Option values renamed in 1.1 (US spelling): the 1.0 spelling -> the current value, so state files and sweeps written
# for 1.0 still load. A key must never be a current option value of any template (it would be rewritten).
LEGACY_CHOICES = {"VQE-optimised": "VQE-optimized", "optimised": "optimized", "synthesised": "synthesized"}


def clamp_params(key: str, params: dict) -> dict:
    """Parameters made valid for the template: missing ones take defaults, numbers are clamped to their bounds
    (bounds may depend on other parameters, e.g. s < 2^n). There is no upper limit on sizes by design."""
    out = defaults(key)
    out.update({k: v for k, v in params.items() if k in out})
    for p in REGISTRY[key][2]:
        v = out[p.name]
        if p.kind == "choice":
            v = LEGACY_CHOICES.get(v, v) if isinstance(v, str) else v
            if v not in p.options:
                v = p.default
        else:
            v = int(v) if p.kind == "int" else float(v)
            lo, hi = p.bound(p.lo, out), p.bound(p.hi, out)
            if lo is not None:
                v = max(v, lo)
            if hi is not None:
                v = min(v, hi)
        out[p.name] = v
    return out


def build(key: str, params: dict | None = None) -> Template:
    params = clamp_params(key, params or {})
    return REGISTRY[key][1](**params)
