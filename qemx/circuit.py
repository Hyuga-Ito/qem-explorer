r"""Circuits as plain data: operations, a text-to-gate language, layering, Qiskit conversion.

Conventions
-----------
* Qubit 0 is the least significant bit (Qiskit order): outcome ``x = sum_i b_i 2^i``.
* A gate's matrix is Qiskit's ``Gate.to_matrix()`` for the qubit list as written, so
  ``cx 0 1`` has control 0 and target 1.
* Noise can be placed in the circuit as an operation (``kind="noise"``), for example
  ``depol(0.05) 0`` or ``ampdamp(0.1) 1``.

Text-to-gate language (one operation per line or separated by ``;``)::

    h 0
    cx 0 1
    rz(pi/4) 1
    cp(pi/2) 0 2
    depol(0.02) 0 1        # a noise channel, drawn in the circuit
    barrier
"""
from __future__ import annotations

import ast
import dataclasses
import json
import math
import operator
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from .resources import qiskit_serial

# ------------------------------------------------------------------ gate table
# name: (qubits, parameters, label for drawings, family for coloring, virtual Z?)
GATES = {
    "id": (1, 0, "I", "pauli", True),
    "x": (1, 0, "X", "pauli", False),
    "y": (1, 0, "Y", "pauli", False),
    "z": (1, 0, "Z", "pauli", True),
    "h": (1, 0, "H", "h", False),
    "s": (1, 0, "S", "phase", True),
    "sdg": (1, 0, "S†", "phase", True),
    "t": (1, 0, "T", "phase", True),
    "tdg": (1, 0, "T†", "phase", True),
    "sx": (1, 0, "√X", "pulse", False),
    "sxdg": (1, 0, "√X†", "pulse", False),
    "rx": (1, 1, "RX", "rot", False),
    "ry": (1, 1, "RY", "rot", False),
    "rz": (1, 1, "RZ", "phase", True),
    "p": (1, 1, "P", "phase", True),
    "cx": (2, 0, "CX", "two", False),
    "cz": (2, 0, "CZ", "two", False),
    "swap": (2, 0, "SWAP", "two", False),
    "cp": (2, 1, "CP", "two", False),
    "rzz": (2, 1, "RZZ", "two", False),
    "ecr": (2, 0, "ECR", "two", False),       # native two-qubit gate of IBM Eagle processors
}
# explicit noise operations: name -> (qubits (0 = any number), parameters, label)
NOISE_OPS = {
    "depol": (0, 1, "Dep"),        # depolarizing, probability p on the listed qubits jointly
    "ampdamp": (1, 1, "AD"),       # amplitude damping, gamma
    "phasedamp": (1, 1, "PD"),     # phase damping, lambda (coherence × sqrt(1-lambda))
    "bitflip": (1, 1, "BF"),       # X with probability p
    "phaseflip": (1, 1, "PF"),     # Z with probability p
    "overrot": (1, 1, "OR"),       # coherent RX(eps)
    "zphase": (1, 1, "ZP"),        # coherent RZ(phi)
}
CLIFFORD_ANGLE = {"rx": math.pi / 2, "ry": math.pi / 2, "rz": math.pi / 2, "p": math.pi / 2,
                  "cp": math.pi, "rzz": math.pi / 2}


@dataclass(frozen=True)
class Op:
    name: str
    qubits: tuple
    params: tuple = ()
    kind: str = "gate"            # gate | noise | barrier | box (drawn only, never simulated)
    tag: str = ""                 # "" | prep | basis | fold | fold-inv | pec | twirl | dd | check

    @property
    def k(self) -> int:
        return len(self.qubits)

    @property
    def is_virtual(self) -> bool:
        return self.kind == "gate" and GATES[self.name][4]

    def label(self) -> str:
        if self.kind == "barrier":
            return "barrier"
        if self.kind == "box":
            return {"derange": "D"}.get(self.name, self.name)
        return GATES[self.name][2] if self.kind == "gate" else NOISE_OPS[self.name][2]

    def text(self) -> str:
        """The operation in the text-to-gate language."""
        if self.kind == "barrier":
            return "barrier" + ("" if not self.qubits else " " + " ".join(map(str, self.qubits)))
        par = "(%s)" % ", ".join(fmt_angle(p) if self.kind == "gate" else fmt_param(p) for p in self.params) \
            if self.params else ""
        return "%s%s %s" % (self.name, par, " ".join(map(str, self.qubits)))

    def with_tag(self, tag: str) -> "Op":
        return dataclasses.replace(self, tag=tag)


@dataclass(frozen=True)
class CircuitSpec:
    n: int
    ops: tuple
    name: str = "circuit"
    coupling: tuple = ()          # (a, b) pairs of connected qubits (compiled circuits); () = a line 0-1-…-(n-1)

    def __post_init__(self):
        for o in self.ops:
            for q in o.qubits:
                if not 0 <= q < self.n:
                    raise ValueError("qubit %d outside 0..%d in '%s'" % (q, self.n - 1, o.text()))

    # ------------------------------------------------------------- basics
    @property
    def gates(self) -> list:
        return [o for o in self.ops if o.kind == "gate"]

    def count(self, what: str = "2q") -> int:
        if what == "2q":
            return sum(1 for o in self.gates if o.k == 2)
        if what == "1q":
            return sum(1 for o in self.gates if o.k == 1 and not o.is_virtual)
        return len(self.gates)

    def depth(self) -> int:
        return len(layers(self))

    def replace(self, ops=None, **kw) -> "CircuitSpec":
        return dataclasses.replace(self, ops=tuple(self.ops if ops is None else ops), **kw)

    def text(self) -> str:
        return "\n".join(o.text() for o in self.ops)

    def neighbours(self) -> dict:
        """{qubit: set of connected qubits} from ``coupling``, or nearest neighbors on a line."""
        nb = {q: set() for q in range(self.n)}
        pairs = self.coupling or tuple((q, q + 1) for q in range(self.n - 1))
        for a, b in pairs:
            if a != b and 0 <= a < self.n and 0 <= b < self.n:
                nb[a].add(b)
                nb[b].add(a)
        return nb

    def to_json(self) -> dict:
        d = dict(n=self.n, name=self.name,
                 ops=[dict(name=o.name, qubits=list(o.qubits), params=list(o.params), kind=o.kind, tag=o.tag)
                      for o in self.ops])
        if self.coupling:
            d["coupling"] = [list(e) for e in self.coupling]
        return d

    @staticmethod
    def from_json(d: dict) -> "CircuitSpec":
        return CircuitSpec(int(d["n"]), tuple(Op(o["name"], tuple(o["qubits"]), tuple(o.get("params", ())),
                                                 o.get("kind", "gate"), o.get("tag", "")) for o in d["ops"]),
                           d.get("name", "circuit"), tuple(tuple(int(x) for x in e) for e in d.get("coupling", ())))

    def key(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True)


# ------------------------------------------------------------------ angles
MAX_PI_DEN = 1024                    # largest denominator written as a fraction of π (π/1024 ≈ 0.0031)


def fmt_angle(x: float, digits: int = 12) -> str:
    """0.785398... -> 'pi/4', 0.0245437 -> 'pi/128', 2.35619 -> '3*pi/4' when x is a rational multiple of π with a
    denominator up to 1024 (to 1e-10), else the number with `digits` significant digits (12 by default, so text and
    table edits round-trip compiled angles)."""
    from fractions import Fraction
    if not math.isfinite(x):
        return "%r" % x
    f = x / math.pi
    fr = Fraction(f).limit_denominator(MAX_PI_DEN)
    if abs(float(fr) - f) < 1e-10 * max(1.0, abs(f)):
        num, den = fr.numerator, fr.denominator
        if num == 0:
            return "0"
        sign = "-" if num < 0 else ""
        a = abs(num)
        top = "pi" if a == 1 else "%d*pi" % a
        return sign + (top if den == 1 else "%s/%d" % (top, den))
    return "%.*g" % (digits, x)


def fmt_param(x: float, digits: int = 12) -> str:
    return "%.*g" % (digits, x)


def fmt_angle_tex(x: float, digits: int = 3) -> str:
    """Short form for circuit labels: 'π/4', '0.785'."""
    s = fmt_angle(x, digits)
    return s.replace("*pi", "π").replace("pi", "π")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_NAMES = {"pi": math.pi, "π": math.pi, "e": math.e, "tau": 2 * math.pi}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "asin": math.asin, "acos": math.acos,
          "atan": math.atan}


MAX_EXPR_CHARS = 200                 # one gate parameter; a whole circuit text may be far longer
_NONDECIMAL = re.compile(r"(?<![\w.])0[xXoObB]|\d_\d")
_QUBIT = re.compile(r"[0-9]+")         # ASCII digits only (int() would also accept '٣' or '1_0')
MAX_TEXT_CHARS = 5_000_000           # guards the server against pasted multi-gigabyte text, not circuit size


_TEX = ((r"\\cdot", "*"), (r"\\times", "*"), ("×", "*"), ("÷", "/"), ("−", "-"), ("·", "*"), (r"\\left", ""),
        (r"\\right", ""))


def normalize_expr(s: str) -> str:
    """Accept the ways people write angles: π, pi, Pi, \\pi, 2pi, 3π/4, \\frac{\\pi}{4}, \\sqrt{2}, 2·π, −π/8.
    Only rewrites text; the result still goes through the whitelist interpreter."""
    s = s.strip()[:4 * MAX_EXPR_CHARS]
    for a, b in _TEX:
        s = re.sub(a, b, s)
    for _ in range(4):                                     # nested \frac / \sqrt with braces
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("π", "pi").replace("\\pi", "pi").replace("\\tau", "tau")
    s = re.sub(r"(?i)\bpi\b", "pi", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d|\))\s*(pi|tau|sqrt|\()", r"\1*\2", s)       # implicit product: 2pi, 3π, 2(pi/3)
    s = re.sub(r"(pi|tau)\s*(\d|\()", r"\1*\2", s)                # pi2, pi(…) -> pi*2, pi*(…)
    return s


def eval_expr(s: str) -> float:
    """Evaluate a numeric expression such as 'pi/4', '2*pi/3', '0.1', 'sqrt(2)/2'.

    This is a whitelist interpreter over Python's AST, not ``eval``: only number literals, + - * / **,
    unary signs, the names pi, e, tau and the functions sqrt, sin, cos, asin, acos, atan are accepted.
    Anything else (attribute access, subscripts, other names, calls, strings, comprehensions, lambdas)
    is rejected, so a parameter field cannot execute code."""
    if not isinstance(s, str):
        raise ValueError("a parameter must be text")
    s = normalize_expr(s)
    if len(s) > MAX_EXPR_CHARS:
        raise ValueError("parameter expression longer than %d characters" % MAX_EXPR_CHARS)
    if not s:
        raise ValueError("empty parameter")
    if _NONDECIMAL.search(s):                       # 0x10, 0o7, 0b1, 1_000: Python literals, not angles
        raise ValueError("'%s' is not a decimal number" % s)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(b) > 1e4 or abs(a) > 1e150):
                raise ValueError("exponent too large in '%s'" % s)
            return _OPS[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _NAMES:
            return _NAMES[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
                and len(node.args) == 1:
            return _FUNCS[node.func.id](ev(node.args[0]))
        raise ValueError("not a number: '%s'" % s)
    try:
        v = float(ev(ast.parse(s, mode="eval")))
    except SyntaxError:
        raise ValueError("'%s' is not a valid number or expression (examples: 0.3, pi/4, 2*pi/3, sqrt(2)/2)" % s) from None
    except ZeroDivisionError:
        raise ValueError("division by zero in '%s'" % s) from None
    except (ArithmeticError, RecursionError, MemoryError, TypeError) as e:
        raise ValueError("cannot evaluate '%s' (%s)" % (s, "too deeply nested" if isinstance(e, RecursionError) else
                                                      "too large" if isinstance(e, (OverflowError, MemoryError)) else
                                                      "math domain error")) from None
    if not math.isfinite(v):
        raise ValueError("not a finite number: '%s'" % s)
    return v


# ------------------------------------------------------------------ parser
class ParseError(ValueError):
    pass


_NAME = re.compile(r"^\s*([A-Za-z_]\w*)\s*")


def _split_line(line: str):
    """(name, parameter text or None, rest): the parameter list may contain nested parentheses."""
    m = _NAME.match(line)
    if not m:
        raise ParseError("cannot read: %s" % line)
    name, i = m.group(1), m.end()
    if i < len(line) and line[i] == "(":
        depth, j = 0, i
        while j < len(line):
            if line[j] == "(":
                depth += 1
            elif line[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            raise ParseError("unbalanced parentheses: %s" % line)
        return name, line[i + 1:j], line[j + 1:].strip()
    return name, None, line[i:].strip()


def _split_params(text: str) -> list:
    """Split on top-level commas only."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [p for p in out if p.strip()]


def parse_line(line: str, n: int | None = None) -> Op | None:
    """One operation of the text-to-gate language. With ``n``, qubit indices must lie in 0..n-1."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    name, ptxt, rest = _split_line(line)
    name = name.lower()
    params = tuple(eval_expr(p) for p in _split_params(ptxt or ""))
    toks = rest.replace(",", " ").split()
    if not all(_QUBIT.fullmatch(t) and len(t) <= 9 for t in toks):
        raise ParseError("qubits must be non-negative integers written with the digits 0-9: %s" % line)
    qs = tuple(int(t) for t in toks)
    if any(q < 0 for q in qs):
        raise ParseError("qubit indices cannot be negative: %s" % line)
    if n is not None and any(q >= n for q in qs):
        raise ParseError("qubit %d does not exist (the register has qubits 0..%d): %s" % (max(qs), n - 1, line))
    if name == "barrier":
        return Op("barrier", qs, (), "barrier")
    if name in GATES:
        nq, npar, *_ = GATES[name]
        if len(qs) != nq:
            raise ParseError("'%s' acts on %d qubit%s, got %d: %s" % (name, nq, "" if nq == 1 else "s", len(qs), line))
        if len(params) != npar:
            raise ParseError("'%s' takes %d parameter%s, got %d: %s" % (name, npar, "" if npar == 1 else "s",
                                                                        len(params), line))
        if len(set(qs)) != len(qs):
            raise ParseError("repeated qubit in: %s" % line)
        return Op(name, qs, params, "gate")
    if name in NOISE_OPS:
        nq, npar, _ = NOISE_OPS[name]
        if nq and len(qs) != nq:
            raise ParseError("'%s' acts on %d qubit, got %d: %s" % (name, nq, len(qs), line))
        if not qs:
            raise ParseError("'%s' needs at least one qubit: %s" % (name, line))
        if len(set(qs)) != len(qs):
            raise ParseError("repeated qubit in: %s" % line)
        if len(params) != npar:
            raise ParseError("'%s' takes %d parameter: %s" % (name, npar, line))
        if name not in ("overrot", "zphase") and not (0 <= params[0] <= 1):
            raise ParseError("'%s' probability must be in [0, 1]: %s" % (name, line))
        return Op(name, qs, params, "noise")
    raise ParseError("unknown operation '%s'. Gates: %s. Noise: %s."
                     % (name, ", ".join(GATES), ", ".join(NOISE_OPS)))


def parse(text: str, n: int, name: str = "circuit") -> CircuitSpec:
    if not isinstance(text, str):
        raise ParseError("the circuit must be text")
    if len(text) > MAX_TEXT_CHARS:
        raise ParseError("the text is %d characters; the limit for one paste is %d" % (len(text), MAX_TEXT_CHARS))
    ops = []
    for k, physical in enumerate(text.splitlines(), 1):
        for raw in physical.split("#", 1)[0].split(";"):    # a comment runs to the end of its line, ';' included
            try:
                o = parse_line(raw, n)
            except ParseError as e:
                raise ParseError("line %d: %s" % (k, _clip(str(e)))) from None
            except (SyntaxError, ValueError) as e:
                raise ParseError("line %d: %s" % (k, _clip(str(e)))) from None
            if o is not None:
                ops.append(o)
    return CircuitSpec(n, tuple(ops), name)


def _clip(msg: str, limit: int = 240) -> str:
    return msg if len(msg) <= limit else msg[:limit] + "…"


# ------------------------------------------------------------------ matrices
@lru_cache(maxsize=4096)
@qiskit_serial
def gate_matrix(name: str, params: tuple = ()) -> np.ndarray:
    """Qiskit's matrix for the gate (little-endian over the op's qubit list)."""
    from qiskit.circuit import library as L
    cls = {"id": L.IGate, "x": L.XGate, "y": L.YGate, "z": L.ZGate, "h": L.HGate, "s": L.SGate,
           "sdg": L.SdgGate, "t": L.TGate, "tdg": L.TdgGate, "sx": L.SXGate, "sxdg": L.SXdgGate,
           "rx": L.RXGate, "ry": L.RYGate, "rz": L.RZGate, "p": L.PhaseGate, "cx": L.CXGate, "cz": L.CZGate,
           "swap": L.SwapGate, "cp": L.CPhaseGate, "rzz": L.RZZGate, "ecr": L.ECRGate}[name]
    m = np.asarray(cls(*params).to_matrix(), dtype=complex)
    m.setflags(write=False)
    return m


def op_matrix(o: Op) -> np.ndarray:
    return gate_matrix(o.name, tuple(float(p) for p in o.params))


INVERSE = {"id": ("id", 1), "x": ("x", 1), "y": ("y", 1), "z": ("z", 1), "h": ("h", 1), "s": ("sdg", 1),
           "sdg": ("s", 1), "t": ("tdg", 1), "tdg": ("t", 1), "sx": ("sxdg", 1), "sxdg": ("sx", 1),
           "rx": ("rx", -1), "ry": ("ry", -1), "rz": ("rz", -1), "p": ("p", -1), "cx": ("cx", 1),
           "cz": ("cz", 1), "swap": ("swap", 1), "cp": ("cp", -1), "rzz": ("rzz", -1), "ecr": ("ecr", 1)}


def inverse(o: Op) -> Op:
    name, sign = INVERSE[o.name]
    return Op(name, o.qubits, tuple(sign * p for p in o.params), "gate", o.tag)


def is_clifford(o: Op) -> bool:
    if o.kind != "gate":
        return True
    if o.name in CLIFFORD_ANGLE:
        a = CLIFFORD_ANGLE[o.name]
        return abs(o.params[0] / a - round(o.params[0] / a)) < 1e-9
    return o.name not in ("t", "tdg")


# ------------------------------------------------------------------ layering
def layers(spec: CircuitSpec) -> list:
    """ASAP layers of op indices. Noise ops take no time: they join the layer of the
    last gate on their qubits. Barriers align their qubits (all qubits if none listed)."""
    n = spec.n
    level = [0] * n
    out: list = []
    for i, o in enumerate(spec.ops):
        if o.kind == "barrier":
            qs = o.qubits or tuple(range(n))
            m = max(level[q] for q in qs)
            for q in qs:
                level[q] = m
            continue
        if o.kind == "noise":
            L = max(level[q] for q in o.qubits) - 1
            L = max(L, 0)
            while len(out) <= L:
                out.append([])
            out[L].append(i)
            continue
        L = max(level[q] for q in o.qubits)
        while len(out) <= L:
            out.append([])
        out[L].append(i)
        for q in o.qubits:
            level[q] = L + 1
    return out


def layer_order(spec: CircuitSpec) -> tuple:
    """(ops in execution order, layer index of each, busy qubits per layer)."""
    Ls = layers(spec)
    ops, lay, busy = [], [], []
    for li, idx in enumerate(Ls):
        b = set()
        for i in idx:
            ops.append(spec.ops[i])
            lay.append(li)
            if spec.ops[i].kind == "gate":
                b.update(spec.ops[i].qubits)
        busy.append(b)
    return ops, lay, busy


# ------------------------------------------------------------------ Qiskit
@qiskit_serial
def to_qiskit(spec: CircuitSpec, measure: bool = False, noise_as_opaque: bool = True):
    """A QuantumCircuit with the same gates. Noise operations become labeled barriers
    ('qx|kind|value'): the transpiler keeps them in place and routing remaps their qubits."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(spec.n, spec.n if measure else 0, name=spec.name)
    for o in spec.ops:
        if o.kind == "barrier":
            if o.tag == "keep":                  # a barrier the compiled circuit must keep (scheduling alignment)
                from qiskit.circuit import Barrier
                qs = list(o.qubits or range(spec.n))
                qc.append(Barrier(len(qs), label="keep"), qs)
            else:
                qc.barrier(*(o.qubits or range(spec.n)))
        elif o.kind == "gate":
            getattr(qc, {"id": "id"}.get(o.name, o.name))(*o.params, *o.qubits)
        elif noise_as_opaque:
            # a labeled barrier: drawn, kept in place by the transpiler, remapped by routing
            from qiskit.circuit import Barrier
            qc.append(Barrier(len(o.qubits), label="qx|%s|%r" % (o.name, float(o.params[0]))), list(o.qubits))
    if measure:
        qc.measure(range(spec.n), range(spec.n))
    return qc


def from_qiskit(qc, name: str = "circuit") -> tuple:
    """(CircuitSpec, measure_map) from a QuantumCircuit; measure_map[clbit] = qubit."""
    ops, mm = [], {}
    for ins in qc.data:
        nm = ins.operation.name
        qs = tuple(qc.find_bit(q).index for q in ins.qubits)
        if nm == "measure":
            mm[qc.find_bit(ins.clbits[0]).index] = qs[0]
            continue
        if nm in ("delay", "reset"):
            continue
        if nm == "barrier":
            lab = getattr(ins.operation, "label", None) or ""
            if lab.startswith("qx|"):
                _, kind, val = lab.split("|")
                ops.append(Op(kind, qs, (float(val),), "noise"))
            else:
                ops.append(Op("barrier", qs, (), "barrier"))
        elif nm in GATES:
            ops.append(Op(nm, qs, tuple(float(p) for p in ins.operation.params), "gate"))
        else:
            raise ValueError("gate '%s' is not in the app's gate set" % nm)
    return CircuitSpec(qc.num_qubits, tuple(ops), name), mm
