r"""Circuit families of the Direction A grid, a fixed physical layout, and compilation.

Every family is a function of (n, a depth or structure parameter, an instance seed) returning a
:class:`FamilyCircuit`: the logical circuit (with its input-state preparation), the measurement basis, the
observable and whether the output bits are read in reverse order. All angles are deterministic functions of the
arguments, so a cell is reproducible from its specification alone.

=============  ==============================================================  ==========================================
family         circuit                                                         observable (ideal value)
=============  ==============================================================  ==========================================
``qft``        F applied to the Fourier state F†|1⟩ ("ramp:1"); AQFT keeps      P(1) = |⟨1|F_m F†|1⟩|² (1 for m = None)
               controlled phases at distance ≤ m (Coppersmith); no SWAP
               network, output read in reverse (``relabel``)
``ghz``        RY(θ) on q0, CX ladder: cos(θ/2)|0…0⟩ + sin(θ/2)|1…1⟩,           ⟨X⊗…⊗X⟩ = sin θ (measured in the X basis)
               θ = π/3 by default (non-Clifford, so CDR/FR/NIL can train);      with ``basis: Z``: P(0…0) + P(1…1) = 1
               ``basis: Z`` measures it in the computational basis
``ising``      r first-order Trotter steps of H = −J Σ ZZ − h Σ X over a fixed   ⟨Z_{⌊n/2⌋}⟩ (local; its light cone grows
               time T (dt = T/r), from |0…0⟩                                    with r)
``brickwork``  L layers of [RY(θ)RZ(φ) on every qubit, CZ on a brick pattern],   ⟨Z_{⌊n/2⌋}⟩
               then a final rotation layer; θ, φ uniform in [0, 2π)
``grover``     k Grover iterations (phase oracle for one marked item m, diffuser)  P(m) = sin²((2k+1)θ), sin θ = 2^{−n/2}
               from H^{⊗n}|0…0⟩; multi-controlled Z as in ``templates.mcz_ops``
               (CZ, the 6-CNOT CCZ, Qiskit's synthesis beyond); m = ⌊(2ⁿ−1)/3⌋
               (alternating bits) unless given
``bv``         Bernstein–Vazirani for an n-bit secret s on n + 1 qubits, the      P(output = |1⟩_anc|s⟩) = 1
               ancilla rotated back to |1⟩ by a final H, so the ideal output is
               one bitstring; s = alternating bits unless given
``context``    spectator-context probe on a chain a–s0–c0–c1–s1–b (n = 6; n = 4   ⟨Z_c0 Z_c1⟩ of the core; the
               drops a, b): L layers of random rotations + CZ on the core pair    spectators end in |0⟩
               (c0, c1); spectators idle in |0⟩ (``zero``), in |+⟩ (``plus``, H
               … H) or in |+⟩ while (a, s0), (s1, b) run CZ in the same layers
               (``driven``; a stays |0⟩, so those CZ are identities). A per-gate
               model predicts almost the same output for the three; ZZ and drive
               crosstalk do not
=============  ==============================================================  ==========================================

The GHZ default observable ⟨Z₀Z_{n−1}⟩ of the app's template is avoided on purpose: it equals 1 for every θ, so every
Clifford training circuit has the same ideal value and CDR/FR/NIL cannot be trained (they fall back to the
unmitigated value).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from ..circuit import CircuitSpec, Op
from .. import observables as OBS

FAMILIES = ("qft", "ghz", "ising", "brickwork", "grover", "bv", "context")
SPECTATOR_MODES = ("zero", "plus", "driven")


@dataclass(frozen=True)
class FamilyCircuit:
    family: str
    n: int
    params: tuple                 # sorted (name, value) pairs: the family parameters
    spec: CircuitSpec             # logical circuit including input preparation
    basis: str                    # measurement basis per qubit ("Z…Z", "X…X")
    observable: OBS.Observable
    relabel: bool = False
    notes: str = ""

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def name(self) -> str:
        return "%s-%d(%s)" % (self.family, self.n, ",".join("%s=%s" % kv for kv in self.params))


def _g(name, *qs, params=()):
    return Op(name, tuple(int(q) for q in qs), tuple(float(p) for p in params), "gate", "")


def _rng(*key) -> np.random.Generator:
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def qft(n: int, m: int | None = None, s: int = 1) -> FamilyCircuit:
    from .. import templates as T
    if m is not None and not 1 <= int(m) < n - 1:
        raise ValueError("AQFT cutoff m must satisfy 1 <= m < n-1 (m >= n-1 is the full QFT)")
    tpl = T.qft(int(n), int(s), "relabel", None if m is None else int(m))
    return FamilyCircuit("qft", int(n), (("m", m), ("s", int(s))), tpl.spec, "Z" * n, tpl.observable, True,
                         tpl.notes)


def ghz(n: int, theta: float = math.pi / 3, basis: str = "X") -> FamilyCircuit:
    """basis "X" (default; the Direction A cells): ⟨X^⊗n⟩ = sin θ. basis "Z": the computational-basis distribution
    cos²(θ/2) on 0…0 and sin²(θ/2) on 1…1, observable P(0…0) + P(1…1) (ideal 1) — the peaked output distribution of
    the application benchmarks (dirA_divergence). ``basis`` enters the parameters only when it is not "X", so the
    X-basis circuits keep their identity."""
    if basis not in ("X", "Z"):
        raise ValueError("GHZ basis must be 'X' or 'Z'")
    ops = [_g("ry", 0, params=(theta,))] + [_g("cx", i, i + 1) for i in range(n - 1)]
    if basis == "Z":
        v = np.zeros(2 ** n)
        v[0] = v[-1] = 1.0
        obs = OBS.Observable("P0P1", "P(0…0) + P(1…1)", r"P(0^{%d})+P(1^{%d})" % (n, n), tuple(v))
        return FamilyCircuit("ghz", int(n), (("basis", "Z"), ("theta", float(theta))),
                             CircuitSpec(n, tuple(ops), "GHZθ-%d (Z)" % n), "Z" * n, obs, False,
                             "cos(θ/2)|0…0⟩ + sin(θ/2)|1…1⟩ measured in the Z basis; ideal P(0…0) + P(1…1) = 1.")
    obs = OBS.z_string(list(range(n)), n)
    obs = OBS.Observable("X" * n, "⟨X^⊗%d⟩" % n, r"\langle X^{\otimes %d}\rangle" % n, obs.values)
    return FamilyCircuit("ghz", int(n), (("theta", float(theta)),),
                         CircuitSpec(n, tuple(ops), "GHZθ-%d" % n), "X" * n, obs, False,
                         "cos(θ/2)|0…0⟩ + sin(θ/2)|1…1⟩ measured in the X basis; ideal ⟨X^⊗n⟩ = sin θ.")


def ising(n: int, steps: int, T: float = 1.0, J: float = 1.0, h: float = 1.0) -> FamilyCircuit:
    if steps < 1:
        raise ValueError("steps >= 1")
    dt = float(T) / int(steps)
    ops = []
    for _ in range(int(steps)):
        for i in range(n - 1):
            ops.append(_g("rzz", i, i + 1, params=(-2.0 * J * dt,)))
        for i in range(n):
            ops.append(_g("rx", i, params=(-2.0 * h * dt,)))
    q = n // 2
    return FamilyCircuit("ising", int(n), (("J", float(J)), ("T", float(T)), ("h", float(h)), ("steps", int(steps))),
                         CircuitSpec(n, tuple(ops), "Ising-%d-r%d" % (n, steps)), "Z" * n, OBS.z(q, n), False,
                         "First-order Trotter, dt = T/r = %g, from |0…0⟩." % dt)


def brickwork(n: int, layers: int, seed: int = 0) -> FamilyCircuit:
    rng = _rng("brickwork", int(n), int(layers), int(seed))
    ops = []

    def rot():
        th = rng.uniform(0, 2 * math.pi, n)
        ph = rng.uniform(0, 2 * math.pi, n)
        for q in range(n):
            ops.append(_g("ry", q, params=(th[q],)))
            ops.append(_g("rz", q, params=(ph[q],)))
    for l in range(int(layers)):
        rot()
        for i in range(l % 2, n - 1, 2):
            ops.append(_g("cz", i, i + 1))
    rot()
    q = n // 2
    return FamilyCircuit("brickwork", int(n), (("layers", int(layers)), ("seed", int(seed))),
                         CircuitSpec(n, tuple(ops), "Brick-%d-L%d-s%d" % (n, layers, seed)), "Z" * n, OBS.z(q, n),
                         False, "Random RY·RZ rotations and brick-pattern CZ layers.")


def grover(n: int, k: int = 1, marked: int | None = None) -> FamilyCircuit:
    from .. import templates as T
    if n < 2:
        raise ValueError("Grover needs n >= 2")
    m = (2 ** n - 1) // 3 if marked is None else int(marked)
    tpl = T.grover(int(n), m, "custom", int(k))
    return FamilyCircuit("grover", int(n), (("k", int(k)), ("marked", marked)), tpl.spec, "Z" * n, tpl.observable,
                         False, tpl.notes)


def bv(n: int, secret: int | None = None) -> FamilyCircuit:
    """n data qubits and one ancilla (qubit n); the final H on the ancilla makes the ideal output |1⟩|s⟩."""
    from .. import templates as T
    if n < 1:
        raise ValueError("BV needs n >= 1")
    s_ = int("".join("1" if i % 2 == 0 else "0" for i in range(n)), 2) if secret is None else int(secret)
    tpl = T.bernstein_vazirani(int(n), s_)
    spec = tpl.spec.replace(ops=tuple(tpl.spec.ops) + (_g("h", n),), name="BV-%d (s=%s)" % (n, format(s_, "0%db" % n)))
    target = s_ | (1 << n)
    return FamilyCircuit("bv", int(n) + 1, (("n_data", int(n)), ("secret", secret)), spec, "Z" * (n + 1),
                         OBS.projector(target, n + 1, label="P(|1⟩|s⟩)"), False,
                         "BV with s = %s on %d data qubits; the ancilla ends in |1⟩." % (format(s_, "0%db" % n), n))


def context(n: int = 6, layers: int = 4, spectators: str = "zero", seed: int = 0) -> FamilyCircuit:
    """Spectator-context probe (see the module table). Barriers after every layer keep the layers aligned and stop
    the compiler from cancelling the spectators' CZ pairs or H pairs."""
    if n not in (4, 6):
        raise ValueError("the context probe has n = 4 (s0 c0 c1 s1) or n = 6 (a s0 c0 c1 s1 b)")
    if spectators not in SPECTATOR_MODES:
        raise ValueError("spectators must be one of %s" % ", ".join(SPECTATOR_MODES))
    if spectators == "driven" and n != 6:
        raise ValueError("driven spectators need n = 6 (their CZ partners a and b)")
    if layers < 1:
        raise ValueError("layers >= 1")
    off = 1 if n == 6 else 0
    s0, c0, c1, s1 = off, off + 1, off + 2, off + 3
    rng = _rng("context", int(n), int(layers), int(seed))           # the same core for every spectator mode
    bar = Op("barrier", tuple(range(n)), (), "barrier", "keep")
    ops = []
    if spectators in ("plus", "driven"):
        ops += [_g("h", s0), _g("h", s1)]
    ops.append(bar)
    for _ in range(int(layers)):
        th, ph = rng.uniform(0, 2 * math.pi, 2), rng.uniform(0, 2 * math.pi, 2)
        for j, q in enumerate((c0, c1)):
            ops += [_g("ry", q, params=(th[j],)), _g("rz", q, params=(ph[j],))]
        ops.append(_g("cz", c0, c1))
        if spectators == "driven":
            ops += [_g("cz", 0, s0), _g("cz", s1, n - 1)]
        ops.append(bar)
    th = rng.uniform(0, 2 * math.pi, 2)
    ops += [_g("ry", c0, params=(th[0],)), _g("ry", c1, params=(th[1],))]
    if spectators in ("plus", "driven"):
        ops += [_g("h", s0), _g("h", s1)]
    return FamilyCircuit("context", int(n), (("layers", int(layers)), ("seed", int(seed)),
                                            ("spectators", spectators)),
                         CircuitSpec(n, tuple(ops), "Context-%d-L%d-%s" % (n, layers, spectators)), "Z" * n,
                         OBS.z_string([c0, c1], n), False,
                         "Core pair (%d, %d), spectators %s." % (c0, c1, spectators))


def build(family: str, n: int, **params) -> FamilyCircuit:
    if family == "qft":
        return qft(n, params.get("m"), params.get("s", 1))
    if family == "ghz":
        return ghz(n, params.get("theta", math.pi / 3), params.get("basis", "X"))
    if family == "ising":
        return ising(n, params["steps"], params.get("T", 1.0), params.get("J", 1.0), params.get("h", 1.0))
    if family == "brickwork":
        return brickwork(n, params["layers"], params.get("seed", 0))
    if family == "grover":
        return grover(n, params.get("k", 1), params.get("marked"))
    if family == "bv":
        if n < 2:
            raise ValueError("BV: n counts every qubit (data + ancilla), so n >= 2")
        return bv(n - 1, params.get("secret"))
    if family == "context":
        return context(n, params.get("layers", 4), params.get("spectators", "zero"), params.get("seed", 0))
    raise ValueError("unknown family %r (choose from %s)" % (family, ", ".join(FAMILIES)))


# ------------------------------------------------------------------ layout
def chain_cost(name: str, path: tuple) -> float:
    """Calibration cost of a chain: Σ two-qubit gate errors on its couplers + Σ (√X error + mean readout error)
    on its qubits (lower is better; the expected-success proxy of mapomatic, linearized)."""
    from ..layout_search import device_info
    info = device_info(name)
    c = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c += info.e2[(min(a, b), max(a, b))]
    for q in path:
        c += info.e1.get(q, 0.0) + info.readout.get(q, 0.0)
    return float(c)


def best_chain(name: str, n: int) -> tuple:
    """The n-qubit path (a line of couplers the compiler may use) with the smallest chain_cost on the device's
    calibration snapshot; ties broken by the lexicographically smallest path. Deterministic."""
    from .. import device
    from ..layout_search import device_info
    base = device.base_name(name)
    info = device_info(base)
    nb = {}
    for a, b in info.good_edges:
        if (min(a, b), max(a, b)) in info.e2:
            nb.setdefault(a, set()).add(b)
            nb.setdefault(b, set()).add(a)
    bad = set(info.broken_nodes)
    best = None

    def dfs(path):
        nonlocal best
        if len(path) == n:
            if path[0] > path[-1]:                       # each line once (it has two orientations)
                return
            key = (round(chain_cost(base, tuple(path)), 15), tuple(path))
            if best is None or key < best:
                best = key
            return
        for q in sorted(nb.get(path[-1], ())):
            if q not in path and q not in bad:
                dfs(path + [q])
    for q0 in sorted(nb):
        if q0 not in bad:
            dfs([q0])
    if best is None:
        raise ValueError("no %d-qubit chain on %s" % (n, base))
    return best[1]


def compile_family(fc: FamilyCircuit, device_name: str, *, opt_level: int = 1, seed: int = 11,
                   initial_layout: tuple | None = None):
    """Compile for the base device with a fixed initial layout (logical qubit i -> initial_layout[i]); returns
    :class:`~qemx.compile.Compiled`. The layout must be used as given: the compiled circuit must occupy exactly
    those physical qubits (checked)."""
    from .. import compile as CP
    from .. import device
    base = device.base_name(device_name)
    lay = tuple(initial_layout) if initial_layout else best_chain(base, fc.n)
    comp = CP.build(fc.spec, None, fc.basis, opts=CP.CompileOptions(device=base, opt_level=int(opt_level),
                                                                   seed=int(seed), initial_layout=lay),
                    relabel=fc.relabel)
    phys = tuple(comp.info["phys"])
    if set(phys) != set(lay):
        raise RuntimeError("the compiler left the requested layout %s (used %s)" % (lay, phys))
    init = comp.info.get("initial_layout") or []
    if init and tuple(init) != tuple(lay):
        raise RuntimeError("initial layout %s was not honored (%s)" % (lay, init))
    comp.info["chain"] = list(lay)
    return comp
