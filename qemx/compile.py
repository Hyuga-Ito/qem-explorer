"""From the circuit the user drew to the circuit a device would run.

    logical ops  →  + input-state preparation  →  + measurement-basis change
                 →  (ISA: Qiskit transpile to {RZ, √X, X, CX|CZ} on a coupling map)
                 →  implemented CircuitSpec + measurement map (classical bit ← qubit)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .circuit import CircuitSpec, Op, from_qiskit, to_qiskit
from .resources import qiskit_serial

INPUT_STATES = {"0": (), "1": ("x",), "+": ("h",), "-": ("x", "h"), "+i": ("h", "s"), "-i": ("h", "sdg")}
INPUT_TEX = {"0": r"|0\rangle", "1": r"|1\rangle", "+": r"|+\rangle", "-": r"|-\rangle", "+i": r"|{+i}\rangle",
             "-i": r"|{-i}\rangle"}
BASIS_OPS = {"Z": (), "X": ("h",), "Y": ("sdg", "h")}
from .resources import QISKIT_LOCK   # one process-wide lock for every entry into Qiskit (see resources.py)

# A phase input "ph:t" is (|0⟩ + e^{2πi t}|1⟩)/√2 = P(2πt)·H|0⟩, t in turns. Register inputs such as the phase ramp
# |ramp:k⟩ = F†|k⟩ = (1/√N) Σ_j e^{-2πijk/N}|j⟩ are products of these (see qemx/sweep.py).
_PHASE = re.compile(r"^ph:(-?\d{1,6}(?:\.\d{1,15})?(?:e-?\d{1,3})?)$", re.ASCII)


def phase_token(turns: float) -> str:
    """The input token of (|0⟩ + e^{2πi·turns}|1⟩)/√2, with turns reduced to [0, 1)."""
    t = float(turns) % 1.0
    return "ph:%.12g" % (0.0 if abs(t - 1.0) < 1e-12 else t)


def valid_input(token) -> bool:
    return isinstance(token, str) and (token in INPUT_STATES or bool(_PHASE.match(token)))


def input_ops(token: str) -> list:
    """(gate, params) pairs that prepare the input from |0⟩."""
    if token in INPUT_STATES:
        return [(g, ()) for g in INPUT_STATES[token]]
    m = _PHASE.match(token or "")
    if not m:
        raise ValueError("unknown input %r (use %s or ph:<turns>)" % (str(token)[:20], ", ".join(INPUT_STATES)))
    t = float(m.group(1)) % 1.0
    return [("h", ())] + ([("p", (2 * math.pi * t,))] if abs(t) > 1e-15 and abs(t - 1) > 1e-15 else [])


def input_short(token: str) -> str:
    """A label that fits in front of a wire: the token itself, or '+φ' for a phase input."""
    return token if token in INPUT_STATES else "+φ"


def input_label(token: str) -> str:
    """Short text for a qubit input: |0⟩, |+i⟩, or a phase input |φt⟩ with t in turns, e.g. |φ.875⟩."""
    if token in INPUT_STATES:
        return "|%s⟩" % token
    m = _PHASE.match(token or "")
    if not m:
        return str(token)[:20]
    t = ("%.4f" % (float(m.group(1)) % 1.0)).rstrip("0").rstrip(".")
    return "|φ%s⟩" % (t[1:] if t.startswith("0.") else t)


@dataclass(frozen=True)
class CompileOptions:
    mode: str = "abstract"               # abstract | isa
    basis_2q: str = "cx"                 # cx | cz
    coupling: str = "all"                # all | line | ring | grid | heavy-hex
    layout: str = "sabre"                # trivial | sabre | dense
    opt_level: int = 1
    seed: int = 11
    device: str | None = None            # a fake backend name: compile for that device (overrides the rest)
    initial_layout: tuple | None = None  # physical qubits for a device, optional


@dataclass
class Compiled:
    logical: CircuitSpec                 # with prep and basis-change ops (tagged)
    implemented: CircuitSpec
    measure_map: dict                    # clbit -> qubit
    info: dict = field(default_factory=dict)


def with_io(spec: CircuitSpec, inputs=None, basis=None) -> CircuitSpec:
    n = spec.n
    inputs = inputs or ("0",) * n
    basis = basis or "Z" * n
    pre = [Op(g, (q,), pr, "gate", "prep") for q, s in enumerate(inputs) for g, pr in input_ops(s)]
    post = [Op(g, (q,), (), "gate", "basis") for q, b in enumerate(basis) for g in BASIS_OPS[b]]
    return spec.replace(ops=tuple(pre) + tuple(spec.ops) + tuple(post))


def grid_shape(n: int) -> tuple:
    """(rows, columns) of the smallest near-square grid with at least n qubits."""
    import math
    r = max(1, int(math.floor(math.sqrt(n))))
    return r, int(math.ceil(n / r))


def heavy_hex_lattice(rows: int, length: int) -> tuple:
    """IBM-style heavy-hex patch: `rows` chains of `length` qubits, joined by bridge qubits every 4 columns
    (columns 0, 4, 8, … below even rows, 2, 6, 10, … below odd rows), numbered row by row with each row's bridges
    after it — the numbering of IBM's Eagle and Heron processors. Returns (number of qubits, undirected edges)."""
    edges, q, prev = [], 0, None
    for r in range(rows):
        row = list(range(q, q + length))
        q += length
        edges += [(a, a + 1) for a in row[:-1]]
        if prev is not None:
            for c, b in prev:
                edges += [(prev_row[c], b), (b, row[c])]
        prev_row = row
        if r < rows - 1:
            cols = [c for c in range(0 if r % 2 == 0 else 2, length, 4)]
            prev = [(c, q + i) for i, c in enumerate(cols)]
            q += len(cols)
    return q, sorted(edges)


def heavy_hex_shape(n: int) -> tuple:
    """(rows, row length) of the smallest IBM-style heavy-hex patch with at least n qubits (near square)."""
    import math
    best = None
    for length in range(5, 64, 4):
        for rows in range(2, 16):
            size = heavy_hex_lattice(rows, length)[0]
            if size >= n:
                key = (size * (1 + 0.25 * abs(math.log(length / (2.0 * rows) / 1.6))), size)
                if best is None or key < best[0]:
                    best = (key, (rows, length))
                break
    return best[1]


@qiskit_serial
def coupling_map(kind: str, n: int):
    """line / ring: exactly n qubits; grid: the smallest near-square grid; heavy-hex: the smallest heavy-hex patch
    (IBM's lattice). The last two may have more physical qubits than the circuit; the compiled circuit keeps only
    the ones it uses."""
    from qiskit.transpiler import CouplingMap
    if kind == "line" and n > 1:
        return CouplingMap.from_line(n)
    if kind == "ring" and n > 2:
        return CouplingMap.from_ring(n)
    if kind == "grid" and n > 1:
        return CouplingMap.from_grid(*grid_shape(n))
    if kind == "heavy-hex" and n > 1:
        size, edges = heavy_hex_lattice(*heavy_hex_shape(n))
        cm = CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
        return cm
    return None


def layout_problem(layout, n: int, coupling: str = "all", device_name: str | None = None) -> str:
    """Why a forced initial layout (physical qubit of q0, q1, …) does not fit ('' when it does)."""
    layout = list(layout or ())
    if not layout:
        return ""
    if len(layout) != n:
        return "it lists %d physical qubit%s but the circuit has %d qubits" % (len(layout), "" if len(layout) == 1
                                                                               else "s", n)
    if len(set(layout)) != len(layout):
        return "a physical qubit appears twice"
    if min(layout) < 0:
        return "physical qubits are numbered from 0"
    try:
        if device_name:
            from .device import fake_backend
            size = fake_backend(device_name).num_qubits
        else:
            cm = coupling_map(coupling, n)
            size = cm.size() if cm is not None else n
    except Exception:                                   # noqa: BLE001 - let the compiler report anything else
        return ""
    if max(layout) >= size:
        return "the target has %d qubits (0–%d)" % (size, size - 1)
    return ""


@qiskit_serial
def lattice_target(opts: "CompileOptions", n: int):
    """A Target for grid / heavy-hex lattices that are larger than the circuit (None otherwise)."""
    cm = coupling_map(opts.coupling, n)
    if cm is None or cm.size() <= n:
        return None
    from qiskit.transpiler import Target
    return Target.from_configuration(basis_gates=["rz", "sx", "x", opts.basis_2q, "measure", "delay"],
                                     coupling_map=cm)


def lattice_tag(opts: "CompileOptions", n: int) -> str:
    if opts.coupling == "grid":
        return "ISA, grid %d×%d" % grid_shape(n)
    return "ISA, heavy-hex %d rows × %d" % heavy_hex_shape(n)


@qiskit_serial
def build(spec: CircuitSpec, inputs=None, basis=None, opts: CompileOptions = CompileOptions(),
          relabel: bool = False) -> Compiled:
    logical = with_io(spec, inputs, basis)
    n = spec.n
    if opts.device:
        from . import device
        impl, mm, phys, info = device.compile_for_device(logical, opts.device, opts.opt_level, opts.seed, opts.layout,
                                                    opts.initial_layout)
        if relabel:
            mm = {i: mm[n - 1 - i] for i in range(n)}
        info.update(mode="device", relabel=bool(relabel))
        return Compiled(logical, impl, mm, info)
    if opts.mode == "abstract":
        mm = {i: (n - 1 - i if relabel else i) for i in range(n)}
        return Compiled(logical, logical.replace(name=spec.name + " (abstract)"), mm,
                        dict(two_qubit=logical.count("2q"), depth=logical.depth(), swaps=0, mode="abstract",
                             relabel=bool(relabel)))
    from qiskit import transpile
    tgt = lattice_target(opts, n)
    if tgt is not None:                               # a lattice larger than the circuit: compile and compact
        from types import SimpleNamespace
        from . import device
        tag = lattice_tag(opts, n)
        init = list(opts.initial_layout) if opts.initial_layout else (list(range(n)) if opts.layout == "trivial"
                                                                       else None)
        impl, mm, phys, info = device.compile_for_target(logical, SimpleNamespace(num_qubits=tgt.num_qubits, name=tag),
                                                         tgt, tag, opts.opt_level, opts.seed, opts.layout, init)
        if relabel:
            mm = {i: mm[n - 1 - i] for i in range(n)}
        info.update(mode="isa", relabel=bool(relabel), lattice=opts.coupling, lattice_size=tgt.num_qubits)
        return Compiled(logical, impl, mm, info)
    cm0 = coupling_map(opts.coupling, n)
    with QISKIT_LOCK:
        qc = to_qiskit(logical, measure=True)
        cm = cm0
        basis_gates = ["rz", "sx", "x", opts.basis_2q]
        tq = transpile(qc, basis_gates=basis_gates, coupling_map=cm, optimization_level=opts.opt_level,
                       layout_method=(opts.layout if cm is not None and not opts.initial_layout else None),
                       initial_layout=(list(opts.initial_layout) if (cm is not None and opts.initial_layout) else
                                       list(range(n)) if (cm is not None and opts.layout == "trivial") else None),
                       seed_transpiler=opts.seed)
        if tq.num_qubits != n:
            raise RuntimeError("routing needed ancilla qubits")
        impl, mm = from_qiskit(tq, spec.name + " (ISA)")
        if cm is not None:                    # the coupling map travels with the circuit (crosstalk neighbors)
            impl = impl.replace(coupling=tuple(sorted({tuple(sorted(e)) for e in cm.get_edges()})))
    if relabel:
        mm = {i: mm[n - 1 - i] for i in range(n)}
    # SWAPs inserted by routing: count via a routing-only pass for information
    info = dict(two_qubit=impl.count("2q"), depth=impl.depth(), logical_two_qubit=logical.count("2q"), mode="isa",
                relabel=bool(relabel))
    return Compiled(logical, impl, mm, info)
