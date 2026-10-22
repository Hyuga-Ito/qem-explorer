r"""Small stabilizer codes as data, and benchmark algorithms compiled onto them ("encoded algorithms").

Conventions
-----------
* Qubit 0 is the least significant bit, as everywhere in the app. Inside a code the data qubits are *roles*
  0 … n−1; character j of a Pauli label acts on role j. Block b of a circuit starts on qubit b·n, and the compiler
  keeps a role → qubit map per block, because some logical gates are permutations of the data qubits, done by
  relabeling (no physical gate).
* Every logical qubit is read in its logical Z basis (or X_L for the Bell, GHZ and state-preparation benchmarks).
  The readout gates turn the code's checks into Z-parities of measured bits, so that
  - a **decoder** is a function of the measured bits, i.e. a diagonal observable o(x) of the app, and
  - **error detection** is post-selection on those parities, i.e. the app's symmetry verification with every check
    (``Template.symmetry`` = (first check, +1, "the other checks")).

Codes (option value)
--------------------
``bit-flip``    3-qubit repetition code |0_L⟩ = |000⟩, |1_L⟩ = |111⟩ (Nielsen & Chuang §10.1.1): stabilizers Z0Z1, Z1Z2,
                Z_L = Z0, X_L = X0X1X2. Corrects one X error per block; one Z error is a logical error ([[3,1,1]]).
``phase-flip``  the same code in the Hadamard basis, |0_L⟩ = |+++⟩: stabilizers X0X1, X1X2, Z_L = X0, X_L = Z0Z1Z2.
                Corrects one Z error per block; one X error is a logical error.
``4-2-2``       the [[4,2,2]] error-detecting code: stabilizers XXXX, ZZZZ and two logical qubits, X_L0 = X0X1,
                Z_L0 = Z0Z2, X_L1 = X0X2, Z_L1 = Z0Z1. Detects any single-qubit error, corrects none. H⊗4 is
                (H_L⊗H_L)·SWAP_L, S·S†·S†·S is CZ_L, and permutations of the data qubits are CNOT_L and SWAP_L — the
                gate set of the small-experiment proposals and demonstrations (Gottesman, arXiv:1610.03507; Vuillot,
                QIC 18, 949 (2018); Harper & Flammia, PRL 122, 080504 (2019)); every identity is checked numerically
                in tests/test_qec_templates.py.
``5-1-3``       Laflamme's [[5,1,3]] code with the encoder of ``templates.five_qubit``: corrects any single-qubit error.
                It has no Z-type stabilizer, so it is read through the inverse encoder, which leaves the syndrome on
                roles 1…4 and the logical qubit on role 0.

Fault tolerance: what is claimed and what is not
------------------------------------------------
A logical gate is called fault-tolerant (FT) here when one fault in it, or one error the code handles on its input,
leaves at most one correctable (for [[4,2,2]]: detectable) error per block. That holds for logical Paulis, transversal
gates (one physical gate per qubit, or CX between matching qubits of two blocks) and permutations of the data qubits.
Every other logical gate is compiled to Pauli rotations exp(−iθP/2) about a minimum-weight representative P of a
logical Pauli — a CX parity ladder (Nielsen & Chuang §4.7) or, at weight 1, one physical rotation — and is
**not** fault-tolerant:

* ladder: one Z fault on the ladder's last qubit between its two CX layers spreads to ±P, which is a logical
  operator — an undetectable logical error;
* weight-1 rotation (repetition codes only, whose logical Z or X has weight 1): an error that the code corrects on
  that qubit anticommutes with P; if it occurs before the gate it reverses the rotation, R(θ)E = E·R(−θ), so the
  decoder removes E but the logical state keeps the wrong angle.

Also not fault-tolerant: the encoder (the logical inputs are prepared on bare qubits and spread by CX gates: state
injection), the inverse-encoder readout of [[5,1,3]], and the X_L readout of the repetition codes (no checks exist in
that basis). There are no syndrome-extraction rounds between gates — the code is checked once, at the readout — so the
errors of every gate accumulate and only their net effect is corrected (one error per block) or detected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import observables as OBS
from .circuit import CircuitSpec, Op

MAX_QUBITS = 12
CODE_KEYS = ("bit-flip", "phase-flip", "4-2-2", "5-1-3")
ALGORITHMS = ("state prep", "Bell", "GHZ-3", "Deutsch-Jozsa", "Bernstein-Vazirani", "RX rotation", "RZ Ramsey",
              "Grover-2", "QFT-2")
ALGORITHM_QUBITS = {"state prep": 1, "Bell": 2, "GHZ-3": 3, "Deutsch-Jozsa": 2, "Bernstein-Vazirani": 2,
                    "RX rotation": 1, "RZ Ramsey": 1, "Grover-2": 2, "QFT-2": 2}
X_BASIS_ALGORITHMS = ("state prep", "Bell", "GHZ-3")          # the ones that may read X_L instead of Z_L
DECODERS = ("majority / lookup", "raw logical parity")
ORACLES = ("constant 0", "constant 1", "balanced x", "balanced not x")
ERRORS = ("none", "X on qubit k", "Y on qubit k", "Z on qubit k")
WHERE = ("before readout", "after encoding")


# ================================================================== Pauli algebra
def _pop(v: int) -> int:
    return bin(v).count("1")


@dataclass(frozen=True)
class Pauli:
    """i^p · Π_q X_q^(x_q) Z_q^(z_q) (X written before Z on each qubit); x and z are bit masks over qubits."""
    x: int = 0
    z: int = 0
    p: int = 0

    @staticmethod
    def parse(label: str, qubits=None) -> "Pauli":
        """'XZZXI' (character j on qubit j, or on qubits[j]), with an optional leading '+' or '-'."""
        p = 0
        if label[:1] in ("+", "-"):
            p, label = (2 if label[0] == "-" else 0), label[1:]
        x = z = 0
        for j, ch in enumerate(label):
            q = j if qubits is None else qubits[j]
            if ch in "XY":
                x |= 1 << q
            if ch in "ZY":
                z |= 1 << q
            if ch == "Y":
                p += 1                                   # Y = i X Z
            if ch not in "IXYZ":
                raise ValueError("not a Pauli label: %r" % label)
        return Pauli(x, z, p % 4)

    def __mul__(self, o: "Pauli") -> "Pauli":
        # (X^a Z^b)(X^c Z^d) = (−1)^(b·c) X^(a⊕c) Z^(b⊕d)
        return Pauli(self.x ^ o.x, self.z ^ o.z, (self.p + o.p + 2 * _pop(self.z & o.x)) % 4)

    def commutes(self, o: "Pauli") -> bool:
        return (_pop(self.x & o.z) + _pop(self.z & o.x)) % 2 == 0

    @property
    def weight(self) -> int:
        return _pop(self.x | self.z)

    @property
    def support(self) -> list:
        m = self.x | self.z
        return [q for q in range(m.bit_length()) if (m >> q) & 1]

    def letters(self) -> dict:
        """{qubit: 'X' | 'Y' | 'Z'} on the support."""
        return {q: "Y" if (self.x >> q) & (self.z >> q) & 1 else ("X" if (self.x >> q) & 1 else "Z")
                for q in self.support}

    def sign(self) -> int:
        """The ±1 in front of the product of letters (Π X^x Z^z = (−i)^(#Y) Π letters); a non-Hermitian
        product raises."""
        c = (self.p - _pop(self.x & self.z)) % 4
        if c % 2:
            raise ValueError("not a Hermitian Pauli")
        return 1 if c == 0 else -1

    def on(self, qubits) -> "Pauli":
        """The same Pauli with role j moved to qubit qubits[j]."""
        x = z = 0
        for j, q in enumerate(qubits):
            x |= ((self.x >> j) & 1) << q
            z |= ((self.z >> j) & 1) << q
        return Pauli(x, z, self.p)

    def text(self, n: int | None = None) -> str:
        """'−X0X1' style (qubits named), or a full label of length n."""
        let = self.letters()
        s = "−" if self.sign() < 0 else ""
        if n is not None:
            return s + "".join(let.get(q, "I") for q in range(n))
        return s + ("".join("%s%d" % (let[q], q) for q in sorted(let)) or "I")

    def matrix(self, n: int) -> np.ndarray:
        """Little-endian 2^n × 2^n matrix (tests)."""
        P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
             "Z": np.diag([1.0, -1.0])}
        let = self.letters()
        m = np.array([[1.0 + 0j]])
        for q in range(n):
            m = np.kron(P[let.get(q, "I")], m)
        return self.sign() * m


def _group(gens) -> list:
    """Every element of the group generated by commuting Paulis, identity first (2^len(gens) elements)."""
    out = [Pauli()]
    for g in gens:
        out += [e * g for e in out]
    return out


def propagate(x: int, z: int, ops) -> tuple:
    """Phase-free Heisenberg propagation of the Pauli (x, z) through Clifford gates (name, qubit, …) in order:
    returns (x', z') of U P U† — what the error has become after the gates."""
    for o in ops:
        name, qs = o[0], o[1:]
        if name == "h":
            q = qs[0]
            bx, bz = (x >> q) & 1, (z >> q) & 1
            x = (x & ~(1 << q)) | (bz << q)
            z = (z & ~(1 << q)) | (bx << q)
        elif name in ("s", "sdg"):
            z ^= ((x >> qs[0]) & 1) << qs[0]
        elif name == "cx":
            c, t = qs
            x ^= ((x >> c) & 1) << t
            z ^= ((z >> t) & 1) << c
        elif name == "cz":
            a, b = qs
            z ^= ((x >> b) & 1) << a
            z ^= ((x >> a) & 1) << b
        elif name == "swap":
            a, b = qs
            for v in ("x", "z"):
                m = x if v == "x" else z
                ba, bb = (m >> a) & 1, (m >> b) & 1
                m = (m & ~((1 << a) | (1 << b))) | (bb << a) | (ba << b)
                if v == "x":
                    x = m
                else:
                    z = m
        elif name not in ("x", "y", "z", "id"):
            raise ValueError("propagate: %s is not a Clifford gate of the table" % name)
    return x, z


# ================================================================== codes as data
@dataclass(frozen=True)
class Readout:
    basis: str                  # the logical basis read: "Z" or "X"
    ops: tuple                  # gates (name, role, …) before the measurement: a basis change or a unitary decoder
    checks: tuple               # role sets whose measured parity is even on the code space (a +1 stabilizer each)
    check_names: tuple          # the stabilizer each check reads (role label "ZZI", or a name such as "S1")
    logical: tuple              # per logical qubit, the role set whose parity is the raw logical bit
    table: dict                 # syndrome (check parities) -> per-logical-qubit flips, for correctable syndromes only
    fault_tolerant: bool
    note: str


@dataclass(frozen=True)
class Code:
    key: str
    name: str
    n: int
    k: int
    d: int                      # distance against arbitrary single-qubit Paulis
    stabilizers: tuple          # role labels, each +1 on the code space
    logical_x: tuple            # per logical qubit, role label (the compiler lowers the weight with stabilizers)
    logical_z: tuple
    inputs: tuple               # roles holding the bare logical inputs before encoding
    encoder: tuple              # gates (name, role, …) from |0…0⟩ with the inputs on `inputs`
    readouts: dict              # "Z" / "X" -> Readout
    protects: str               # what the code corrects / detects
    gates: tuple                # the logical gate table: (gate, implementation, fault-tolerant?, why)
    # transversal / permutation implementations the compiler uses (on roles)
    hh: tuple = ()              # H on every logical qubit of a block …
    hh_relabel: tuple = ()      # … followed by these role swaps
    cz_intra: tuple = ()        # CZ_L between the two logical qubits of a block
    cx_intra: dict = field(default_factory=dict)    # (control, target) logical indices -> role swap
    swap_intra: tuple = ()      # role swap = SWAP_L inside a block
    cx_inter: str = ""          # transversal CNOT_L between blocks: "forward" CX a_r→b_r, "reverse" CX b_r→a_r

    def stabilizer_group(self) -> list:
        return _group([Pauli.parse(s) for s in self.stabilizers])

    def x_l(self, j: int) -> Pauli:
        return Pauli.parse(self.logical_x[j])

    def z_l(self, j: int) -> Pauli:
        return Pauli.parse(self.logical_z[j])


def bitflip_table(checks, logical, n: int) -> dict:
    """Minimum-weight decoding of bit flips on the n measured bits: syndrome -> logical flips, kept only when every
    minimum-weight flip pattern with that syndrome has the same logical effect (else the syndrome is detected but not
    corrected). For the 3-bit repetition code this is the majority vote."""
    best = {}
    for e in range(2 ** n):
        s = tuple(_pop(e & _mask(c)) % 2 for c in checks)
        f = tuple(_pop(e & _mask(lg)) % 2 for lg in logical)
        w = _pop(e)
        if s not in best or w < best[s][0]:
            best[s] = (w, {f})
        elif w == best[s][0]:
            best[s][1].add(f)
    return {s: next(iter(fs)) for s, (w, fs) in best.items() if len(fs) == 1}


def unitary_table(code_n: int, ops, checks, logical) -> dict:
    """Lookup decoder of a readout that ends in a unitary decoder: every single-qubit X, Y, Z error on the data
    before `ops` is propagated through them; its X part on the measured bits gives the syndrome and the logical flips.
    Raises if two of these errors share a syndrome but need different corrections."""
    table = {}
    errs = [(0, 0)] + [(((1 if P in "XY" else 0) << q), ((1 if P in "ZY" else 0) << q))
                       for q in range(code_n) for P in "XYZ"]
    for x, z in errs:
        x2, _ = propagate(x, z, ops)
        s = tuple(_pop(x2 & _mask(c)) % 2 for c in checks)
        f = tuple(_pop(x2 & _mask(lg)) % 2 for lg in logical)
        if table.setdefault(s, f) != f:
            raise ValueError("single-qubit errors with syndrome %s need different corrections" % (s,))
    return table


def _mask(roles) -> int:
    m = 0
    for r in roles:
        m |= 1 << r
    return m


_WHY_PAULI = "a logical Pauli is a product of single-qubit Paulis: one fault, one single-qubit error"
_WHY_TRANSVERSAL = ("transversal: each physical gate touches one qubit per block, so one fault leaves at most one "
                    "error per block")
_WHY_RELABEL = "no physical gate: the data qubits are relabeled (a permutation spreads no error)"
_WHY_LADDER = ("a CX parity ladder: one Z fault on its last qubit between the two CX layers spreads to the whole "
               "rotation axis, which is a logical operator — an undetectable logical error")
_WHY_W1 = ("one physical rotation on the weight-1 representative: an error the code corrects on that qubit anticommutes "
           "with the rotation axis, so if it occurs before the gate it reverses the rotation (R(θ)E = E·R(−θ)); the "
           "decoder removes E, the wrong logical angle stays")
_WHY_NATIVE2 = ("a two-qubit gate on the weight-1 representatives: an error the code corrects on one of them, occurring "
                "before the gate, is turned into a logical error on the other block (e.g. CZ: X ⊗ I → X ⊗ Z)")


def _rep_gates(kind: str) -> tuple:
    zl, xl, err = ("Z0", "X0X1X2", "X") if kind == "bit-flip" else ("X0", "Z0Z1Z2", "Z")
    tc = "CX a_i → b_i" if kind == "bit-flip" else "CX b_i → a_i (the direction flips in the Hadamard basis)"
    return (
        ("X_L = %s, Z_L = %s" % (xl, zl), "single-qubit Paulis", True, _WHY_PAULI),
        ("CNOT_L (block a → block b)", tc + ", i = 0, 1, 2: transversal", True,
         _WHY_TRANSVERSAL + "; an %s error spreads to the matching qubit of the other block only" % err),
        ("SWAP_L (two blocks)", "relabel the blocks", True, _WHY_RELABEL),
        ("RZ_L(θ), S_L, T_L", "%s(θ) on q0 (Z_L = %s has weight 1)" % ("RZ" if kind == "bit-flip" else "RX", zl), False,
         _WHY_W1),
        ("RX_L(θ)", "exp(−iθ %s/2): %sCX parity ladder" % (xl, "H on every qubit + " if kind == "bit-flip" else ""),
         False, _WHY_LADDER),
        ("H_L", "RZ_L(π/2)·RX_L(π/2)·RZ_L(π/2)", False, "built from the non-fault-tolerant rotations above"),
        ("CZ_L, CP_L(φ)", "CZ / CP on q0 of the two blocks%s" % ("" if kind == "bit-flip" else " in the X basis"),
         False, _WHY_NATIVE2),
        ("other logical gates", "Pauli rotations (CX ladders)", False, _WHY_LADDER),
    )


def _bit_flip() -> Code:
    rz = Readout("Z", (), ((0, 1), (1, 2)), ("ZZI", "IZZ"), ((0, 1, 2),),
                 bitflip_table(((0, 1), (1, 2)), ((0, 1, 2),), 3), True,
                 "transversal Z measurement; the two Z-parity checks of a block locate one flipped bit (majority vote)")
    rx = Readout("X", (("h", 0), ("h", 1), ("h", 2)), (), (), ((0, 1, 2),), {(): (0,)}, False,
                 "transversal X measurement, X_L = parity of the three bits: the bit-flip code has no X-type "
                 "stabilizer, so there is no check and one Z error (or one readout error) flips X_L unnoticed")
    return Code("bit-flip", "3-qubit bit-flip repetition code", 3, 1, 1, ("ZZI", "IZZ"), ("XXX",), ("ZII",), (0,),
                (("cx", 0, 1), ("cx", 0, 2)), {"Z": rz, "X": rx},
                "corrects one X (bit-flip) error per block; one Z error is a logical error (distance 1 against Z)",
                _rep_gates("bit-flip"), cx_inter="forward")


def _phase_flip() -> Code:
    rz = Readout("Z", (("h", 0), ("h", 1), ("h", 2)), ((0, 1), (1, 2)), ("XXI", "IXX"), ((0, 1, 2),),
                 bitflip_table(((0, 1), (1, 2)), ((0, 1, 2),), 3), True,
                 "transversal X measurement (H, then Z); the two X-parity checks of a block locate one flipped bit "
                 "(majority vote)")
    rx = Readout("X", (), (), (), ((0, 1, 2),), {(): (0,)}, False,
                 "transversal Z measurement, X_L = Z0Z1Z2: the phase-flip code has no Z-type stabilizer, so there is "
                 "no check and one X error (or one readout error) flips X_L unnoticed")
    return Code("phase-flip", "3-qubit phase-flip repetition code", 3, 1, 1, ("XXI", "IXX"), ("ZZZ",), ("XII",), (0,),
                (("cx", 0, 1), ("cx", 0, 2), ("h", 0), ("h", 1), ("h", 2)), {"Z": rz, "X": rx},
                "corrects one Z (phase-flip) error per block; one X error is a logical error (distance 1 against X)",
                _rep_gates("phase-flip"), cx_inter="reverse")


def _four_two_two() -> Code:
    allq = (0, 1, 2, 3)
    rz = Readout("Z", (), (allq,), ("ZZZZ",), ((0, 2), (0, 1)), bitflip_table((allq,), ((0, 2), (0, 1)), 4), True,
                 "transversal Z measurement; the check ZZZZ detects any odd number of bit flips")
    rx = Readout("X", tuple(("h", q) for q in allq), (allq,), ("XXXX",), ((0, 1), (0, 2)),
                 bitflip_table((allq,), ((0, 1), (0, 2)), 4), True,
                 "transversal X measurement; the check XXXX detects any odd number of flips")
    gates = (
        ("X_L0 = X0X1, X_L1 = X0X2, Z_L0 = Z0Z2, Z_L1 = Z0Z1", "single-qubit Paulis", True, _WHY_PAULI),
        ("H_L ⊗ H_L", "H on all four qubits, then relabel q1 ↔ q2", True,
         "transversal; H⊗4 alone is (H_L⊗H_L)·SWAP_L, the relabeling undoes the logical SWAP"),
        ("CZ_L", "S, S†, S†, S on q0, q1, q2, q3", True, _WHY_TRANSVERSAL),
        ("CNOT_L(0→1), CNOT_L(1→0), SWAP_L", "relabel q0 ↔ q2, q0 ↔ q1, q1 ↔ q2", True, _WHY_RELABEL),
        ("H_L on one logical qubit, S_L, T_L, RZ_L(θ), RX_L(θ), CP_L(φ)",
         "Pauli rotations on weight-2 representatives (CX ladders)", False, _WHY_LADDER),
        ("logical gates between two blocks", "Pauli rotations (CX ladders)", False, _WHY_LADDER),
    )
    return Code("4-2-2", "[[4,2,2]] error-detecting code", 4, 2, 2, ("XXXX", "ZZZZ"), ("XXII", "XIXI"),
                ("ZIZI", "ZZII"), (1, 2),
                (("cx", 1, 0), ("cx", 2, 0), ("h", 3), ("cx", 3, 0), ("cx", 3, 1), ("cx", 3, 2)), {"Z": rz, "X": rx},
                "detects any single-qubit error and corrects none (distance 2)", gates,
                hh=tuple(("h", q) for q in allq), hh_relabel=((1, 2),),
                cz_intra=(("s", 0), ("sdg", 1), ("sdg", 2), ("s", 3)), cx_intra={(0, 1): (0, 2), (1, 0): (0, 1)},
                swap_intra=(1, 2))


_INV1 = {"s": "sdg", "sdg": "s"}


def _five_one_three() -> Code:
    from .templates import FIVE_ENCODER, FIVE_STABILISERS
    enc = tuple(FIVE_ENCODER)
    dec = tuple((_INV1.get(g[0], g[0]),) + tuple(g[1:]) for g in reversed(enc))
    checks = ((1,), (2,), (3,), (4,))
    names = ("S1", "S2", "S3", "S4")
    rz = Readout("Z", dec, checks, names, ((0,),), unitary_table(5, dec, checks, ((0,),)), False,
                 "the inverse encoder leaves the syndrome on the block's 2nd…5th qubits (each Z = +1 on the code "
                 "space) and Z_L on its 1st; a 16-entry lookup table corrects any single-qubit error present before "
                 "it. The inverse encoder is a unitary decoder of %d CX: its own faults spread, so this readout is not "
                 "fault-tolerant"
                 % sum(1 for g in dec if g[0] == "cx"))
    decx = dec + (("h", 0),)
    rx = Readout("X", decx, checks, names, ((0,),), unitary_table(5, decx, checks, ((0,),)), False,
                 "the inverse encoder, then H on the block's 1st qubit (X_L); lookup correction as for Z_L; not "
                 "fault-tolerant")
    gates = (
        ("X_L = X⊗5, Z_L = Z⊗5 (weight-3 equivalents used)", "single-qubit Paulis", True, _WHY_PAULI),
        ("every other logical gate (rotations, H_L, CNOT_L between blocks)", "Pauli rotations on weight-3 (one block) "
         "or weight-6 (two blocks) representatives: CX ladders", False, _WHY_LADDER),
    )
    return Code("5-1-3", "[[5,1,3]] five-qubit code", 5, 1, 3, tuple(FIVE_STABILISERS), ("XXXXX",), ("ZZZZZ",), (0,),
                enc, {"Z": rz, "X": rx}, "corrects any single-qubit error (distance 3)", gates)


@lru_cache(maxsize=1)
def codes() -> dict:
    return {c.key: c for c in (_bit_flip(), _phase_flip(), _four_two_two(), _five_one_three())}


def code(key: str) -> Code:
    if key not in CODE_KEYS:
        raise ValueError("unknown code %r: choose one of %s" % (key, ", ".join(CODE_KEYS)))
    return codes()[key]


def num_qubits(code_key: str, algorithm: str) -> int:
    """Data qubits of the encoded circuit: one block per code.k logical qubits (no ancillas)."""
    c = code(code_key) if code_key in CODE_KEYS else None
    if c is None or algorithm not in ALGORITHM_QUBITS:
        return 1
    return c.n * -(-ALGORITHM_QUBITS[algorithm] // c.k)


# ================================================================== benchmark algorithms as logical circuits
@dataclass
class Algorithm:
    name: str
    k: int                       # logical qubits the algorithm uses
    prep: dict                   # logical qubit -> [(gate, params)] applied to its bare input before encoding
    body: list                   # logical gates (name, (logical qubits…), (params…)); "h" may list several qubits
    basis: str                   # logical readout basis
    value: object                # f(list of 0/1 arrays, one per logical qubit) -> ±1 array
    label: str
    tex: str
    target: str = ""             # the expected answer, for the notes
    answer: str = ""             # the expected answer as a short token (observable key)


def _sign(cond) -> np.ndarray:
    return np.where(cond, 1.0, -1.0)


def _par(bits) -> np.ndarray:
    out = np.zeros_like(bits[0])
    for b in bits:
        out = out ^ b
    return 1.0 - 2.0 * out


def algorithm(name: str, code_k: int, theta: float = 1.0472, s: int = 3, oracle: str = "balanced x",
              basis: str = "Z") -> Algorithm:
    """The benchmark as a logical circuit on its own logical qubits. ``code_k`` is the number of logical qubits per
    block: a [[4,2,2]] block running a 1-qubit benchmark has a spectator logical qubit, which the Ramsey benchmark
    also turns with H so that H_L⊗H_L stays transversal."""
    if name not in ALGORITHMS:
        raise ValueError("unknown algorithm %r: choose one of %s" % (name, ", ".join(ALGORITHMS)))
    basis = basis if name in X_BASIS_ALGORITHMS else "Z"
    if basis not in ("Z", "X"):
        raise ValueError("the logical readout basis is Z or X")
    s = int(s) % 4
    sb = format(s, "02b")
    th = float(theta)
    if name == "state prep":
        return Algorithm(name, 1, {0: [("ry", (th,))]}, [], basis, lambda b: _par([b[0]]),
                         "⟨%s_L⟩" % basis, r"|\psi_L\rangle=R_Y(\theta)|0\rangle\ \text{encoded},\ \ \langle %s_L\rangle"
                         r"=%s\theta=%.4f" % (basis, r"\cos" if basis == "Z" else r"\sin",
                                              math.cos(th) if basis == "Z" else math.sin(th)))
    if name == "Bell":
        return Algorithm(name, 2, {0: [("h", ())]}, [("cx", (0, 1), ())], basis, lambda b: _par([b[0], b[1]]),
                         "⟨%s_L %s_L⟩" % (basis, basis),
                         r"|\Phi^+_L\rangle=\mathrm{CNOT}_L\big(|{+}_L\rangle|0_L\rangle\big),\ \ \langle %s_L%s_L\rangle=1"
                         % (basis, basis))
    if name == "GHZ-3":
        if basis == "Z":
            val, lab = (lambda b: _sign((b[0] == b[1]) & (b[1] == b[2]))), "GHZ sign (+1 if the 3 logical bits agree)"
            tex = r"\langle(-1)^{[b_0b_1b_2\notin\{000,111\}]}\rangle=1"
        else:
            val, lab, tex = (lambda b: _par([b[0], b[1], b[2]])), "⟨X_L X_L X_L⟩", r"\langle X_LX_LX_L\rangle=1"
        return Algorithm(name, 3, {0: [("h", ())]}, [("cx", (0, 1), ()), ("cx", (1, 2), ())], basis, val, lab,
                         r"|\mathrm{GHZ}_L\rangle=\tfrac{1}{\sqrt2}(|0_L0_L0_L\rangle+|1_L1_L1_L\rangle),\ \ " + tex)
    if name == "Deutsch-Jozsa":
        if oracle not in ORACLES:
            raise ValueError("unknown oracle %r" % oracle)
        body = {"constant 0": [], "constant 1": [("x", (1,), ())], "balanced x": [("cx", (0, 1), ())],
                "balanced not x": [("cx", (0, 1), ()), ("x", (1,), ())]}[oracle] + [("h", (0, 1), ())]
        ans = 0 if oracle.startswith("constant") else 1
        return Algorithm(name, 2, {0: [("h", ())], 1: [("x", ()), ("h", ())]}, body, "Z",
                         lambda b: _sign(b[0] == ans), "answer sign (+1 if the input reads %d)" % ans,
                         r"H_L^{\otimes2}\,U_f\,|{+}_L\rangle|{-}_L\rangle:\ f\ \text{%s}\ \Rightarrow\ x=%d"
                         % ("constant" if ans == 0 else "balanced", ans),
                         "input L0 = %d (%s), ancilla L1 = 1" % (ans, "constant" if ans == 0 else "balanced"), str(ans))
    if name == "Bernstein-Vazirani":
        body = [("z", (q,), ()) for q in range(2) if (s >> q) & 1] + [("h", (0, 1), ())]
        return Algorithm(name, 2, {0: [("h", ())], 1: [("h", ())]}, body, "Z",
                         lambda b: _sign((b[0] == (s & 1)) & (b[1] == (s >> 1))),
                         "success sign (+1 if the decoded s = %s)" % sb,
                         r"H_L^{\otimes2}\,Z_L^{s}\,|{+}_L{+}_L\rangle=|s_L\rangle,\ s=%s\ (\text{phase oracle})" % sb,
                         "s = %s" % sb, sb)
    if name == "RX rotation":
        return Algorithm(name, 1, {}, [("rx", (0,), (th,))], "Z", lambda b: _par([b[0]]), "⟨Z_L⟩",
                         r"R_{X,L}(\theta)|0_L\rangle,\ \ \langle Z_L\rangle=\cos\theta=%.4f" % math.cos(th))
    if name == "RZ Ramsey":
        spect = tuple(range(1, code_k))                     # the spectator of a [[4,2,2]] block
        prep = {q: [("h", ())] for q in (0,) + spect}
        return Algorithm(name, 1, prep, [("rz", (0,), (th,)), ("h", (0,) + spect, ())], "Z", lambda b: _par([b[0]]),
                         "⟨Z_L⟩",
                         r"H_L\,R_{Z,L}(\theta)\,|{+}_L\rangle,\ \ \langle Z_L\rangle=\cos\theta=%.4f" % math.cos(th))
    if name == "Grover-2":
        flips = [q for q in range(2) if not (s >> q) & 1]
        body = [("x", (q,), ()) for q in flips] + [("cz", (0, 1), ())] + [("x", (q,), ()) for q in flips]
        body += [("h", (0, 1), ()), ("x", (0,), ()), ("x", (1,), ()), ("cz", (0, 1), ()), ("x", (0,), ()),
                 ("x", (1,), ()), ("h", (0, 1), ())]
        return Algorithm(name, 2, {0: [("h", ())], 1: [("h", ())]}, body, "Z",
                         lambda b: _sign((b[0] == (s & 1)) & (b[1] == (s >> 1))),
                         "success sign (+1 if the decoded item = %s)" % sb,
                         r"\text{one Grover iteration on 2 logical qubits marks }|%s_L\rangle:\ P=1" % sb, "m = %s" % sb,
                         sb)
    if name == "QFT-2":
        from .templates import fourier_prep, qft_body, swap_network
        prep = {}
        for o in fourier_prep(2, s):
            prep.setdefault(o.qubits[0], []).append((o.name, tuple(o.params)))
        body = [(o.name, tuple(o.qubits), tuple(o.params)) for o in qft_body([0, 1]) + swap_network([0, 1])]
        return Algorithm(name, 2, prep, body, "Z", lambda b: _sign((b[0] == (s & 1)) & (b[1] == (s >> 1))),
                         "success sign (+1 if the decoded output = %s)" % sb,
                         r"F_L\,F^\dagger|%d\rangle=|%d_L\rangle\ \ (\text{QFT with its SWAP})" % (s, s), "s = %s" % sb,
                         sb)
    raise AssertionError(name)                                         # pragma: no cover


def logical_ops(alg: Algorithm) -> list:
    """The benchmark as app operations on its k logical qubits (prep, body, readout basis change)."""
    ops = []
    for q in sorted(alg.prep):
        if q < alg.k:
            ops += [Op(g, (q,), tuple(p), "gate", "prep") for g, p in alg.prep[q]]
    for name, qs, prm in alg.body:
        if name == "h":
            ops += [Op("h", (q,), (), "gate") for q in qs if q < alg.k]
        else:
            ops.append(Op(name, tuple(qs), tuple(prm), "gate"))
    if alg.basis == "X":
        ops += [Op("h", (q,), (), "gate", "basis") for q in range(alg.k)]
    return ops


def logical_ideal(alg: Algorithm) -> float:
    """The exact value of the benchmark's observable on its logical circuit (no code)."""
    from .templates import statevector
    a = statevector(alg.k, logical_ops(alg))
    x = np.arange(2 ** alg.k)
    bits = [(x >> q) & 1 for q in range(alg.k)]
    return float(np.abs(a) ** 2 @ alg.value(bits))


# ================================================================== the compiler
@dataclass
class Built:
    spec: CircuitSpec
    observable: OBS.Observable
    tex: str
    notes: str
    symmetry: tuple | None
    checks: tuple                    # ((label, qubits), …): Z-parities of measured bits, each +1 without error
    logical: float                   # the ideal (error-free) value of the decoded observable
    info: dict


class _Compiler:
    def __init__(self, c: Code, nblocks: int):
        self.c = c
        self.roles = [list(range(b * c.n, (b + 1) * c.n)) for b in range(nblocks)]
        self.n = nblocks * c.n
        self.ops = []
        self.log = []                # (gate, implementation, fault-tolerant, why)
        self._stab = [Pauli.parse(s) for s in c.stabilizers]

    # ------------------------------------------------------------- helpers
    def loc(self, lq: int) -> tuple:
        return divmod(lq, self.c.k)

    def emit(self, name, qubits, params=(), tag=""):
        self.ops.append(Op(name, tuple(int(q) for q in qubits), tuple(float(p) for p in params), "gate", tag))

    def emit_roles(self, b: int, gates, tag=""):
        for g in gates:
            self.emit(g[0], [self.roles[b][r] for r in g[1:]], (), tag)

    def relabel(self, b: int, swaps):
        for r1, r2 in swaps:
            self.roles[b][r1], self.roles[b][r2] = self.roles[b][r2], self.roles[b][r1]

    def rep(self, paulis: dict) -> Pauli:
        """Minimum-weight physical representative of the logical Pauli {logical qubit: 'X'|'Y'|'Z'} (multiplied by
        stabilizers of the blocks it touches), on the current qubits."""
        P, blocks = Pauli(), set()
        for lq, ch in sorted(paulis.items()):
            b, j = self.loc(lq)
            blocks.add(b)
            xl, zl = self.c.x_l(j).on(self.roles[b]), self.c.z_l(j).on(self.roles[b])
            P = P * {"X": xl, "Z": zl, "Y": Pauli(0, 0, 1) * xl * zl}[ch]          # Y_L = i X_L Z_L
        best = P
        group = [Pauli()]
        for b in sorted(blocks):
            group = [g * h for g in group for h in _group([s.on(self.roles[b]) for s in self._stab])]
        for g in group:
            Q = P * g
            if Q.weight < best.weight:
                best = Q
        return best

    def pauli_gate(self, P: Pauli):
        for q, ch in sorted(P.letters().items()):
            self.emit(ch.lower(), (q,))

    def rotation(self, P: Pauli, theta: float) -> tuple:
        """exp(−iθP/2) for a Hermitian Pauli P; returns (fault-tolerant, implementation, why)."""
        sg, let = P.sign(), P.letters()
        th = sg * float(theta)
        qs = sorted(let)
        if len(qs) == 1:
            q = qs[0]
            self.emit({"X": "rx", "Y": "ry", "Z": "rz"}[let[q]], (q,), (th,))
            return False, "R%s on q%d" % (let[q], q), _WHY_W1
        pre = []                                  # basis change B with B·letter·B† = Z: H for X, S† then H for Y
        for q in qs:
            if let[q] == "X":
                pre.append(("h", q))
            elif let[q] == "Y":
                pre += [("sdg", q), ("h", q)]
        for g, q in pre:
            self.emit(g, (q,))
        for a, b in zip(qs, qs[1:]):              # parity of the support onto its last qubit
            self.emit("cx", (a, b))
        self.emit("rz", (qs[-1],), (th,))
        for a, b in reversed(list(zip(qs, qs[1:]))):
            self.emit("cx", (a, b))
        for g, q in reversed(pre):                # B† in reverse order
            self.emit({"h": "h", "sdg": "s"}[g], (q,))
        return False, "CX-ladder rotation about %s" % P.text(), _WHY_LADDER

    def lrot(self, axis: dict, theta: float) -> tuple:
        return self.rotation(self.rep(axis), theta)

    def record(self, gate: str, parts):
        """Log one logical gate made of several (ft, implementation, why) parts."""
        ft = all(p[0] for p in parts)
        impl = ", then ".join(p[1] for p in parts)
        whys = [p[2] for p in parts if not p[0]] or [p[2] for p in parts]
        self.log.append((gate, impl, ft, " / ".join(dict.fromkeys(whys))))

    # ------------------------------------------------------------- logical gates
    def paulis(self, gates):
        """A run of logical Paulis [(name, logical qubit)], applied as one physical Pauli (they commute up to a
        global phase, so X_L0 X_L1 on [[4,2,2]] is X1X2 rather than X0X2 then X0X1)."""
        bits = {}
        for name, q in gates:                                    # letters only: the phase is global
            bx, bz = bits.get(q, (0, 0))
            bits[q] = (bx ^ (name in "xy"), bz ^ (name in "yz"))
        let = {q: {(1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[b] for q, b in bits.items() if any(b)}
        P = self.rep(let) if let else Pauli()
        self.pauli_gate(P)
        self.log.append((" ".join("%s_L(L%d)" % (nm.upper(), q) for nm, q in gates),
                         "%s (single-qubit Paulis)" % P.text(), True, _WHY_PAULI))

    def apply(self, name: str, qs: tuple, prm: tuple):
        c = self.c
        L = lambda q: "L%d" % q                                            # noqa: E731
        if name in ("x", "y", "z"):
            self.paulis([(name, qs[0])])
            return
        if name == "h":
            rest = list(qs)
            for b in range(len(self.roles)):
                mine = [b * c.k + j for j in range(c.k)]
                if c.hh and all(q in rest for q in mine):
                    before = list(self.roles[b])
                    self.emit_roles(b, c.hh)
                    self.relabel(b, c.hh_relabel)
                    rest = [q for q in rest if q not in mine]
                    self.log.append(("H_L⊗H_L(%s)" % ",".join(map(L, mine)),
                                     "H on %s + relabel %s (transversal)" % (
                                         " ".join("q%d" % q for q in sorted(before)),
                                         ", ".join("q%d ↔ q%d" % (before[r1], before[r2]) for r1, r2 in c.hh_relabel)),
                                     True, "transversal; H⊗4 alone is (H_L⊗H_L)·SWAP_L, the relabeling undoes the "
                                           "logical SWAP"))
            for q in rest:
                parts = [self.lrot({q: "Z"}, math.pi / 2), self.lrot({q: "X"}, math.pi / 2),
                         self.lrot({q: "Z"}, math.pi / 2)]
                self.record("H_L(%s)" % L(q), parts)
            return
        ang = {"s": math.pi / 2, "sdg": -math.pi / 2, "t": math.pi / 4, "tdg": -math.pi / 4}
        if name in ang or name in ("rz", "p"):
            th = ang.get(name, prm[0] if prm else 0.0)
            ft, impl, why = self.lrot({qs[0]: "Z"}, th)
            self.log.append(("%s_L(%s)" % (name.upper() if name in ang else "RZ", L(qs[0])), impl, ft, why))
            return
        if name in ("rx", "ry"):
            ft, impl, why = self.lrot({qs[0]: name[1].upper()}, prm[0])
            self.log.append(("%s_L(%s)" % (name.upper(), L(qs[0])), impl, ft, why))
            return
        if name in ("cz", "cp"):
            a, b = qs
            phi = math.pi if name == "cz" else float(prm[0])
            (ba, ja), (bb, jb) = self.loc(a), self.loc(b)
            gname = "%s_L(%s,%s)" % (name.upper(), L(a), L(b))
            if name == "cz" and ba == bb and c.cz_intra:
                self.emit_roles(ba, c.cz_intra)
                self.log.append((gname, "S·S†·S†·S (transversal)", True, _WHY_TRANSVERSAL))
                return
            za, zb = self.rep({a: "Z"}), self.rep({b: "Z"})
            if za.weight == 1 and zb.weight == 1 and za.sign() > 0 and zb.sign() > 0 and ba != bb:
                (qa, la), = za.letters().items()
                (qb, lb), = zb.letters().items()
                basis = [q for q, l in ((qa, la), (qb, lb)) if l == "X"]
                if all(l in "XZ" for l in (la, lb)):
                    for q in basis:
                        self.emit("h", (q,))
                    self.emit("cz" if name == "cz" else "cp", (qa, qb), () if name == "cz" else (phi,))
                    for q in basis:
                        self.emit("h", (q,))
                    self.log.append((gname, "%s on q%d, q%d%s" % (name.upper(), qa, qb, " in the X basis" if basis else ""),
                                     False, _WHY_NATIVE2))
                    return
            # diag(1, 1, 1, e^{iφ}) = e^{iφ/4} RZ_a(φ/2) RZ_b(φ/2) R_{Z_aZ_b}(−φ/2)
            parts = [self.lrot({a: "Z"}, phi / 2), self.lrot({b: "Z"}, phi / 2), self.lrot({a: "Z", b: "Z"}, -phi / 2)]
            self.record(gname, parts)
            return
        if name == "cx":
            ct, tg = qs
            (bc, jc), (bt, jt) = self.loc(ct), self.loc(tg)
            gname = "CNOT_L(%s→%s)" % (L(ct), L(tg))
            if bc == bt and (jc, jt) in c.cx_intra:
                r1, r2 = c.cx_intra[(jc, jt)]
                q1, q2 = self.roles[bc][r1], self.roles[bc][r2]
                self.relabel(bc, [(r1, r2)])
                self.log.append((gname, "relabel q%d ↔ q%d" % (q1, q2), True, _WHY_RELABEL))
                return
            if bc != bt and c.cx_inter and c.k == 1:
                src, dst = (bc, bt) if c.cx_inter == "forward" else (bt, bc)
                for r in range(c.n):
                    self.emit("cx", (self.roles[src][r], self.roles[dst][r]))
                self.log.append((gname, "CX on matching qubits (transversal%s)" % (
                    "" if c.cx_inter == "forward" else ", target block → control block"), True, _WHY_TRANSVERSAL))
                return
            # CNOT = exp(iπ (1−Z_c)(1−X_t)/4) = e^{iπ/4} RZ_c(π/2) RX_t(π/2) R_{Z_cX_t}(−π/2)
            parts = [self.lrot({ct: "Z"}, math.pi / 2), self.lrot({tg: "X"}, math.pi / 2),
                     self.lrot({ct: "Z", tg: "X"}, -math.pi / 2)]
            self.record(gname, parts)
            return
        if name == "swap":
            a, b = qs
            (ba, ja), (bb, jb) = self.loc(a), self.loc(b)
            gname = "SWAP_L(%s,%s)" % (L(a), L(b))
            if ba == bb and c.swap_intra and {ja, jb} == {0, 1}:
                r1, r2 = c.swap_intra
                q1, q2 = self.roles[ba][r1], self.roles[ba][r2]
                self.relabel(ba, [(r1, r2)])
                self.log.append((gname, "relabel q%d ↔ q%d" % (q1, q2), True, _WHY_RELABEL))
                return
            if ba != bb and c.k == 1:
                self.roles[ba], self.roles[bb] = self.roles[bb], self.roles[ba]
                self.log.append((gname, "relabel the blocks", True, _WHY_RELABEL))
                return
            for x, y in ((a, b), (b, a), (a, b)):
                self.apply("cx", (x, y), ())
            return
        raise ValueError("logical gate %r is not supported" % name)


def _fmt_ops(gates) -> str:
    from .circuit import fmt_angle
    return ", ".join("%s(%s)" % (g, ", ".join(fmt_angle(p, 4) for p in prm)) if prm else g for g, prm in gates)


def build(code_key: str = "4-2-2", algorithm_name: str = "Grover-2", theta: float = 1.0472, s: int = 3,
          oracle: str = "balanced x", logical_basis: str = "Z", decoder: str = "majority / lookup",
          error: str = "none", k: int = 0, where: str = "before readout") -> Built:
    """The benchmark ``algorithm_name`` compiled onto blocks of ``code_key``: encoder (with the logical inputs
    injected on the bare input qubits), the logical gates, an optional injected single-qubit error, the transversal
    readout, and the decoded logical observable."""
    c = code(code_key)
    return compile_algorithm(c, algorithm(algorithm_name, c.k, theta, s, oracle, logical_basis), decoder, error, k,
                             where)


def compile_algorithm(c: Code, alg: Algorithm, decoder: str = "majority / lookup", error: str = "none", k: int = 0,
                      where: str = "before readout") -> Built:
    """Compile any logical circuit given as an :class:`Algorithm` onto blocks of the code ``c`` (``build`` does it for
    the named benchmarks; the tests also feed random logical circuits)."""
    if decoder not in DECODERS:
        raise ValueError("unknown decoder %r: choose one of %s" % (decoder, ", ".join(DECODERS)))
    if error not in ERRORS:
        raise ValueError("unknown error %r" % error)
    if where not in WHERE:
        raise ValueError("the error is injected %s" % " or ".join(WHERE))
    nb = -(-alg.k // c.k)
    n = nb * c.n
    if n > MAX_QUBITS:
        raise ValueError("%s on the %s needs %d qubits (%d blocks of %d); the encoded templates stop at %d"
                         % (alg.name, c.name, n, nb, c.n, MAX_QUBITS))
    cp = _Compiler(c, nb)
    nl = nb * c.k                                         # logical qubits incl. spectators of a [[4,2,2]] block
    # 1. injection and encoding
    inj = []
    for lq in range(nl):
        b, j = cp.loc(lq)
        for g, prm in alg.prep.get(lq, []):
            cp.emit(g, (cp.roles[b][c.inputs[j]],), prm, "encode")      # logical injection: part of the circuit
            inj.append("%s on q%d" % (_fmt_ops([(g, prm)]), cp.roles[b][c.inputs[j]]))
    for b in range(nb):
        cp.emit_roles(b, c.encoder, "encode")
    enc_end = len(cp.ops)
    err_op = None
    if error != "none":
        kq = int(k)
        if not 0 <= kq < n:
            raise ValueError("qubit k = %d is outside the %d data qubits" % (kq, n))
        err_op = Op(error[0].lower(), (kq,), (), "gate", "")
    if err_op is not None and where == "after encoding":
        cp.ops.append(err_op)
    body_start = len(cp.ops)
    # 2. the logical circuit (a run of logical Paulis becomes one physical Pauli)
    run = []
    for name, qs, prm in list(alg.body) + [("end", (), ())]:
        if name in ("x", "y", "z"):
            run.append((name, qs[0]))
            continue
        if run:
            cp.paulis(run)
            run = []
        if name != "end":
            cp.apply(name, tuple(qs), tuple(prm))
    body_end = len(cp.ops)
    if err_op is not None and where == "before readout":
        cp.ops.append(err_op)
    # 3. readout
    ro = c.readouts[alg.basis]
    for b in range(nb):
        cp.emit_roles(b, ro.ops, "basis")
    spec = CircuitSpec(n, tuple(cp.ops), "%s on %s" % (alg.name, c.name))
    # 4. decoding: a diagonal observable over the measured bits
    lookup = decoder == DECODERS[0]
    x = np.arange(2 ** n, dtype=np.int64)

    def parity(qubits):
        out = np.zeros(2 ** n, dtype=np.int64)
        for q in qubits:
            out ^= (x >> q) & 1
        return out
    lbits, checks, read = [], [], []
    for b in range(nb):
        rq = cp.roles[b]
        syn = [parity([rq[r] for r in chk]) for chk in ro.checks]
        raw = [parity([rq[r] for r in lg]) for lg in ro.logical]
        read += ["⊕".join("q%d" % q for q in sorted(rq[r] for r in lg)) for lg in ro.logical]
        if lookup and ro.table:
            sidx = np.zeros(2 ** n, dtype=np.int64)
            for i, sv in enumerate(syn):
                sidx |= sv << i
            for sv, flips in ro.table.items():
                hit = sidx == sum(int(v) << i for i, v in enumerate(sv))
                for j, f in enumerate(flips):
                    if f:
                        raw[j] = raw[j] ^ hit.astype(np.int64)
        lbits += raw
        for chk, nm in zip(ro.checks, ro.check_names):
            qs = tuple(sorted(rq[r] for r in chk))
            if len(nm) == c.n and set(nm) <= set("IXYZ"):
                lab = "".join("%s%d" % (nm[r], rq[r]) for r in sorted(range(c.n), key=lambda r: rq[r]) if nm[r] != "I")
            else:
                lab = "%s (%s)" % (nm, " ".join("q%d" % q for q in qs))
            checks.append((lab, qs))
    vals = alg.value(lbits[:alg.k])
    dec_lab = "majority / lookup" if lookup else "raw parity"
    obs = OBS.Observable("qecL_%s_%s_%s_%s%s" % (c.key, alg.name.replace(" ", ""), alg.basis, "lk" if lookup else "raw",
                                                 "_" + alg.answer if alg.answer else ""),
                         "%s decoded (%s)" % (alg.label, dec_lab),
                         r"\langle O_L\rangle_{\rm %s}" % ("dec" if lookup else "raw"), tuple(vals.astype(float)))
    sym = None
    if checks:
        sym = (checks[0][1], +1, "; ".join(" ".join(map(str, q)) for _, q in checks[1:]))
    ideal = logical_ideal(alg)
    # 5. what is and is not fault-tolerant
    groups = {}                                   # (gate name, fault-tolerant, why) -> [labels], first implementation
    for gate, impl, ft, why in cp.log:
        base = "logical Paulis" if impl.endswith("(single-qubit Paulis)") else gate.split("(")[0]
        g = groups.setdefault((base, ft, why), dict(labels=[], impl=impl))
        if gate not in g["labels"]:
            g["labels"].append(gate)

    def _listing(ft):
        out = []
        for (base, f, why), g in groups.items():
            if f != ft:
                continue
            lab = base if base == "logical Paulis" else ", ".join(g["labels"])
            out.append("%s → %s%s" % (lab, g["impl"], " (first shown)" if len(g["labels"]) > 1 else ""))
        return out
    ft_parts, nft_parts = _listing(True), _listing(False)
    all_ft = not nft_parts
    blocks_txt = ", ".join("block %d = q%d…q%d" % (b, b * c.n, b * c.n + c.n - 1) for b in range(nb))
    lq_txt = ", ".join("L%d" % q for q in range(alg.k))
    spect = [q for q in range(alg.k, nl)]
    parts = [
        "%s — %s. %s; logical qubits %s%s%s." % (
            c.name, c.protects, blocks_txt[:1].upper() + blocks_txt[1:], lq_txt,
            " (spectator %s)" % ", ".join("L%d" % q for q in spect) if spect else "",
            "; answer: %s" % alg.target if alg.target else ""),
        "Encoding — NOT fault-tolerant: %s, then the encoder (%d CX per block); a single fault on an input qubit before "
        "or during encoding can become a logical error (state injection)." % (
            "inputs prepared on the bare qubits (%s)" % "; ".join(inj) if inj else "inputs |0⟩",
            sum(1 for g in c.encoder if g[0] == "cx")),
    ]
    if ft_parts:
        parts.append("Fault-tolerant logical gates: %s." % "; ".join(ft_parts))
    if nft_parts:
        whys = list(dict.fromkeys(w for (_, f, why) in groups if not f for w in why.split(" / ")))
        parts.append("NOT fault-tolerant: %s. Why: %s." % ("; ".join(nft_parts), " / ".join(whys)))
    elif alg.body:
        parts.append("Every logical gate of this benchmark is fault-tolerant on this code.")
    parts.append("Readout — %s: %s%s. Raw logical bits: %s." % (
        "fault-tolerant" if ro.fault_tolerant else "NOT fault-tolerant", ro.note,
        "" if lookup or not ro.checks else "; decoder off: the raw logical parity is read, without correction",
        ", ".join("L%d = %s" % (q, r) for q, r in enumerate(read[:alg.k]))))
    if lookup and ro.checks and set(ro.table) == {tuple([0] * len(ro.checks))}:
        parts.append("Decoder: a nonzero syndrome does not identify the error, so nothing is corrected and the "
                     "observable is the raw logical parity; detection is post-selection on the checks.")
    if checks:
        parts.append("Checks for symmetry verification — parities of measured bits, +1 without error, named by the "
                     "stabilizer they read; post-selecting on all of them is error detection: %s."
                     % ", ".join(lab for lab, _ in checks))
    else:
        parts.append("No checks exist in this readout basis: symmetry verification has nothing to post-select on.")
    parts.append("The code is checked once, at the readout: there are no syndrome-extraction rounds between gates, so "
                 "the errors of every gate accumulate and only their net effect is corrected or detected.")
    if err_op is not None:
        parts.append("Injected error: %s on q%d %s." % (error[0], err_op.qubits[0], where))
    notes = " ".join(parts)
    tex = r"\text{%s}:\ %s" % (c.name, alg.tex)
    info = dict(code=c.key, code_name=c.name, algorithm=alg.name, logical_qubits=alg.k, blocks=nb,
                block_qubits=[list(range(b * c.n, (b + 1) * c.n)) for b in range(nb)],
                final_roles=[list(r) for r in cp.roles],
                readout_basis=alg.basis, decoder=decoder, encoder=(0, enc_end), body=(body_start, body_end),
                readout=(len(cp.ops) - nb * len(ro.ops), len(cp.ops)),
                gates=[dict(gate=g, implementation=i, fault_tolerant=f, why=w) for g, i, f, w in cp.log],
                all_gates_fault_tolerant=all_ft, readout_fault_tolerant=ro.fault_tolerant,
                code_gates=[dict(gate=g, implementation=i, fault_tolerant=f, why=w) for g, i, f, w in c.gates],
                target=alg.target)
    return Built(spec, obs, tex, notes, sym, tuple(checks), ideal, info)


# ================================================================== presets (code × algorithm pairs worth a look)
PRESETS = {
    "Grover-2 on [[4,2,2]]: every logical gate transversal": dict(code="4-2-2", algorithm="Grover-2", s=3),
    "Bernstein-Vazirani on [[4,2,2]]: transversal H⊗4": dict(code="4-2-2", algorithm="Bernstein-Vazirani", s=2),
    "Deutsch-Jozsa on [[4,2,2]]: CNOT_L by relabeling": dict(code="4-2-2", algorithm="Deutsch-Jozsa",
                                                             oracle="balanced x"),
    "GHZ-3 on bit-flip blocks: transversal CNOTs": dict(code="bit-flip", algorithm="GHZ-3"),
    "Bell pair on phase-flip blocks, X_L readout (unprotected)": dict(code="phase-flip", algorithm="Bell",
                                                                      logical_basis="X"),
    "RX_L(θ) on [[4,2,2]]: non-Clifford, for learning-based mitigation": dict(code="4-2-2", algorithm="RX rotation",
                                                                              theta=1.0472),
    "RZ_L(θ) Ramsey on the bit-flip code": dict(code="bit-flip", algorithm="RZ Ramsey", theta=1.0472),
    "QFT-2 on [[5,1,3]] (10 qubits, non-FT gadgets)": dict(code="5-1-3", algorithm="QFT-2", s=2),
}
