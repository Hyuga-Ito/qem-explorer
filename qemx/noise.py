r"""The noise injector: a :class:`NoiseSpec`, channel constructors, and :func:`resolve`,
which turns an implemented circuit plus a noise specification into a flat list of
instructions that any backend can execute.

Noise model (per physical gate, applied after the ideal gate, in this order)
---------------------------------------------------------------------------
coherent      over-rotation   U -> U^{1+ε} for pulsed one-qubit gates (X, Y, √X, RX, RY, H)
              fixed phase     exp(-i θ Z⊗Z/2) after every two-qubit gate; RZ(φ) after every gate
stochastic    random kick     a rotation by κ about a uniformly random axis; on average a
                              depolarizing channel with Pauli eigenvalue f = (1 + 2 cos κ)/3
              depolarizing    ρ -> (1-p) ρ + p Tr(ρ) I/d   (p₁ for one-qubit, p₂ for two-qubit gates)
              damping         amplitude damping γ, phase damping λ on every qubit the gate touches
idle          quasi-static    a random detuning δ ~ N(0, σ²) per qubit, fixed within a shot:
              field           RZ(δ) per idle layer (refocused by dynamical decoupling)
              Markovian       phase damping and amplitude damping per idle layer (not refocused)
readout       confusion       P(1|0) = ε₀₁, P(0|1) = ε₁₀ on every measured bit
crosstalk     spectator Z     RZ(δx) on every idle neighbor of a driven qubit (drive-induced Stark shift)
              spectator ZZ    exp(-i ζ Z_a Z_j / 2) between a driven qubit a and each idle neighbor j
                              (neighbors from the compiled circuit's coupling map; a line by default);
                              both are coherent and are refocused by dynamical decoupling on the spectator

Virtual Z rotations (RZ, P, Z, S, T and their inverses) are noiseless, as on IBM hardware.
All values are illustrative: no preset describes a specific device.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import fractional_matrix_power

from .circuit import CircuitSpec, Op, GATES, is_clifford, layer_order, op_matrix

ROTATING = {"x", "y", "sx", "sxdg", "rx", "ry", "h"}


@dataclass(frozen=True)
class NoiseSpec:
    p1: float = 0.0             # depolarizing, one-qubit gates
    p2: float = 0.0             # depolarizing, two-qubit gates
    amp_damp: float = 0.0       # amplitude damping γ per gate and qubit
    phase_damp: float = 0.0     # phase damping λ per gate and qubit
    over_rot: float = 0.0       # fractional over-rotation ε of pulsed one-qubit gates
    zz_phase: float = 0.0       # rad, coherent ZZ phase after two-qubit gates
    z_phase: float = 0.0        # rad, coherent Z phase after every physical gate
    kick: float = 0.0           # rad, random-axis kick after every physical gate
    field_sigma: float = 0.0    # rad per idle layer, quasi-static detuning (standard deviation)
    idle_dephase: float = 0.0   # phase damping λ per idle layer
    idle_damp: float = 0.0      # amplitude damping γ per idle layer
    ro01: float = 0.0           # P(read 1 | 0)
    ro10: float = 0.0           # P(read 0 | 1)
    field_samples: int = 16     # stratified samples of the quasi-static field
    xtalk_z: float = 0.0        # rad, Z phase on each idle neighbor of a driven qubit (spectator error)
    xtalk_zz: float = 0.0       # rad, ZZ phase between a driven qubit and each idle neighbor

    def replace(self, **kw) -> "NoiseSpec":
        return dataclasses.replace(self, **kw)

    def scaled(self, lam: float) -> "NoiseSpec":
        """Every error rate × λ (probabilities capped); coherent angles × λ."""
        c = lambda x: min(max(x * lam, 0.0), 1.0)
        return self.replace(p1=c(self.p1), p2=min(self.p2 * lam, 1.0), amp_damp=c(self.amp_damp),
                            phase_damp=c(self.phase_damp), over_rot=self.over_rot * lam, zz_phase=self.zz_phase * lam,
                            z_phase=self.z_phase * lam, kick=self.kick * lam, field_sigma=self.field_sigma * lam,
                            idle_dephase=c(self.idle_dephase), idle_damp=c(self.idle_damp),
                            ro01=min(self.ro01 * lam, 0.5), ro10=min(self.ro10 * lam, 0.5),
                            xtalk_z=self.xtalk_z * lam, xtalk_zz=self.xtalk_zz * lam)

    @property
    def is_ideal(self) -> bool:
        d = dataclasses.asdict(self)
        d.pop("field_samples")
        return all(v == 0 for v in d.values())

    def key(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    def gate_infidelity(self, k: int) -> float:
        """Average gate infidelity r of one noisy k-qubit gate (for the 'datasheet' number)."""
        S = gate_noise_superop(self, k, "x" if k == 1 else "cx")
        d = 2 ** k
        Fe = np.real(np.trace(ptm(S))) / d ** 2
        return float(1 - (d * Fe + 1) / (d + 1))


PRESETS = {
    "Ideal (no noise)": NoiseSpec(),
    "Depolarizing": NoiseSpec(p1=0.002, p2=0.02),
    "Amplitude damping": NoiseSpec(amp_damp=0.02, idle_damp=0.01),
    "Coherent over-rotation": NoiseSpec(over_rot=0.06),
    "Fixed phase (ZZ)": NoiseSpec(zz_phase=0.12),
    "Random field (quasi-static)": NoiseSpec(field_sigma=0.15),
    "Markovian dephasing": NoiseSpec(idle_dephase=0.03, phase_damp=0.01),
    "Readout only": NoiseSpec(ro01=0.02, ro10=0.05),
    "Crosstalk (spectator)": NoiseSpec(p1=5e-4, p2=0.005, xtalk_z=0.06, xtalk_zz=0.04),
    "Hardware-realistic (illustrative)": NoiseSpec(p1=5e-4, p2=0.01, amp_damp=0.002, phase_damp=0.002, over_rot=0.01,
                                                   zz_phase=0.02, field_sigma=0.03, idle_dephase=0.004,
                                                   idle_damp=0.002, ro01=0.015, ro10=0.03),
}


# ------------------------------------------------------------------ channels
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI1 = {"I": I2, "X": PX, "Y": PY, "Z": PZ}


def pauli_labels(k: int) -> list:
    out = [""]
    for _ in range(k):
        out = [a + b for a in out for b in "IXYZ"]
    return out


def pauli_matrix(label: str) -> np.ndarray:
    """label[i] acts on the op's i-th qubit; little-endian: M = P_{k-1} ⊗ ... ⊗ P_0."""
    m = np.array([[1.0 + 0j]])
    for ch in label:
        m = np.kron(PAULI1[ch], m)
    return m


def kraus_amp(g: float) -> list:
    return [np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex), np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)]


def kraus_phase(l: float) -> list:
    return [np.array([[1, 0], [0, np.sqrt(1 - l)]], dtype=complex), np.array([[0, 0], [0, np.sqrt(l)]], dtype=complex)]


def depol_mixture(p: float, k: int) -> list:
    """(1-p) ρ + p I/d as a Pauli mixture: w_I = 1 - p + p/4^k, w_P = p/4^k."""
    d2 = 4 ** k
    return [((1 - p + p / d2) if lab == "I" * k else p / d2, pauli_matrix(lab)) for lab in pauli_labels(k)]


def kick_p(kappa: float) -> float:
    """Depolarizing probability equal to a random-axis rotation by κ: f = (1 + 2 cos κ)/3."""
    return (2 - 2 * np.cos(kappa)) / 3


def rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def rzz(theta: float) -> np.ndarray:
    return np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))


OVERROT_AXIS = {"x": ("x", np.pi), "y": ("y", np.pi), "sx": ("x", np.pi / 2), "sxdg": ("x", -np.pi / 2),
                "h": ("h", np.pi)}


@lru_cache(maxsize=2048)
def _frac_power(name: str, params: tuple, eps: float) -> np.ndarray:
    """The extra rotation U^ε of a pulse that over-rotates by the fraction ε: the same axis, angle ε·θ.
    Analytic (no matrix logarithm), so large angles and θ = 2πm keep their sign."""
    if name in ("rx", "ry"):
        axis, theta = name[1], float(params[0])
    else:
        axis, theta = OVERROT_AXIS[name]
    a = eps * theta
    if axis == "x":
        n = PX
    elif axis == "y":
        n = PY
    else:                                   # H: rotation by pi about (x + z)/sqrt 2
        n = (PX + PZ) / np.sqrt(2)
    return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * n


def _kron(A, B):
    """np.kron for 2-D arrays without the generic-shape overhead (superoperators are built very often)."""
    A, B = np.asarray(A), np.asarray(B)
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])


def superop_unitary(U):
    return _kron(U, np.conj(U))


def superop_kraus(Ks):
    return sum(_kron(K, np.conj(K)) for K in Ks)


def superop_mixture(items):
    return sum(w * _kron(U, np.conj(U)) for w, U in items)


def ptm(S: np.ndarray) -> np.ndarray:
    d = int(round(np.sqrt(S.shape[0])))
    k = int(round(np.log2(d)))
    P = [pauli_matrix(l) for l in pauli_labels(k)]
    out = np.empty((d * d, d * d))
    for j, Pj in enumerate(P):
        Ej = (S @ Pj.reshape(-1)).reshape(d, d)
        for i, Pi in enumerate(P):
            out[i, j] = np.real(np.trace(Pi @ Ej)) / d
    return out


def pauli_probs_from_eigs(eigs: np.ndarray, k: int) -> np.ndarray:
    labels = pauli_labels(k)

    def anti(a, b):
        return sum(1 for x, y in zip(a, b) if x != "I" and y != "I" and x != y) % 2
    return np.array([sum((-1) ** anti(p, q) * eigs[i] for i, p in enumerate(labels)) for q in labels]) / len(labels)


# ------------------------------------------------------------------ instructions
@dataclass
class Instruction:
    kind: str            # unitary | kraus | mixture
    qubits: tuple
    data: object         # matrix | [K...] | [(w, U)...]
    step: int            # index into the executed op list this instruction belongs to
    source: str          # gate | noise | explicit | idle | dd | pec | twirl
    label: str = ""

    @property
    def cptp(self) -> bool:
        if self.kind != "mixture":
            return True
        return all(w >= -1e-15 for w, _ in self.data)


def gate_noise_instructions(noise: NoiseSpec, o: Op, step: int) -> list:
    """Noise after one physical gate (empty for virtual Z gates)."""
    if o.kind != "gate" or o.is_virtual:
        return []
    out = []
    k = o.k
    if noise.over_rot and o.name in ROTATING:
        out.append(Instruction("unitary", o.qubits, _frac_power(o.name, tuple(float(p) for p in o.params),
                                                                 float(noise.over_rot)), step, "noise",
                               "over-rotation ε=%g" % noise.over_rot))
    if noise.zz_phase and k == 2:
        out.append(Instruction("unitary", o.qubits, rzz(noise.zz_phase), step, "noise", "ZZ phase %g" % noise.zz_phase))
    if noise.z_phase:
        for q in o.qubits:
            out.append(Instruction("unitary", (q,), rz(noise.z_phase), step, "noise", "Z phase %g" % noise.z_phase))
    if noise.kick:
        for q in o.qubits:
            out.append(Instruction("mixture", (q,), depol_mixture(kick_p(noise.kick), 1), step, "noise",
                                   "random kick %g" % noise.kick))
    p = noise.p1 if k == 1 else noise.p2
    if p:
        out.append(Instruction("mixture", o.qubits, depol_mixture(p, k), step, "noise", "depolarizing %g" % p))
    if noise.amp_damp:
        for q in o.qubits:
            out.append(Instruction("kraus", (q,), kraus_amp(noise.amp_damp), step, "noise", "damping %g" % noise.amp_damp))
    if noise.phase_damp:
        for q in o.qubits:
            out.append(Instruction("kraus", (q,), kraus_phase(noise.phase_damp), step, "noise",
                                   "dephasing %g" % noise.phase_damp))
    return out


def explicit_noise(o: Op, step: int) -> list:
    p = float(o.params[0])
    if o.name == "depol":
        return [Instruction("mixture", o.qubits, depol_mixture(p, o.k), step, "explicit", "depolarizing %g" % p)]
    q = o.qubits
    if o.name == "ampdamp":
        return [Instruction("kraus", q, kraus_amp(p), step, "explicit", "amplitude damping %g" % p)]
    if o.name == "phasedamp":
        return [Instruction("kraus", q, kraus_phase(p), step, "explicit", "phase damping %g" % p)]
    if o.name == "bitflip":
        return [Instruction("mixture", q, [(1 - p, I2), (p, PX)], step, "explicit", "bit flip %g" % p)]
    if o.name == "phaseflip":
        return [Instruction("mixture", q, [(1 - p, I2), (p, PZ)], step, "explicit", "phase flip %g" % p)]
    if o.name == "overrot":
        from .circuit import gate_matrix
        return [Instruction("unitary", q, gate_matrix("rx", (p,)), step, "explicit", "RX(%g)" % p)]
    if o.name == "zphase":
        return [Instruction("unitary", q, rz(p), step, "explicit", "RZ(%g)" % p)]
    raise ValueError(o.name)


# ------------------------------------------------------------ superoperators
def instr_superop(ins: Instruction, qubits: tuple) -> np.ndarray:
    """Superoperator of `ins` on the ordered qubit tuple `qubits` (⊇ ins.qubits, length ≤ 2)."""
    def full(M):
        if ins.qubits == qubits:
            return M
        if len(qubits) == 2 and len(ins.qubits) == 1:
            return _kron(I2, M) if ins.qubits[0] == qubits[0] else _kron(M, I2)
        if len(qubits) == 2 and ins.qubits == (qubits[1], qubits[0]):
            SW = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
            return SW @ M @ SW
        raise ValueError("cannot embed")
    if ins.kind == "unitary":
        return superop_unitary(full(ins.data))
    if ins.kind == "kraus":
        return superop_kraus([full(K) for K in ins.data])
    return superop_mixture([(w, full(U)) for w, U in ins.data])


def gate_noise_superop(noise: NoiseSpec, k: int, name: str = None) -> np.ndarray:
    name = name or ("x" if k == 1 else "cx")
    o = Op(name, tuple(range(k)), (), "gate")
    S = np.eye(4 ** k, dtype=complex)
    for ins in gate_noise_instructions(noise, o, 0):
        S = instr_superop(ins, o.qubits) @ S
    return S


def twirl_instructions(instrs: list, qubits: tuple, step: int) -> list:
    """Pauli twirl of the composed noise channel: keep its PTM diagonal, as a Pauli mixture."""
    if not instrs:
        return []
    k = len(qubits)
    S = np.eye(4 ** k, dtype=complex)
    for ins in instrs:
        S = instr_superop(ins, qubits) @ S
    eigs = np.diag(ptm(S))
    probs = pauli_probs_from_eigs(eigs, k)
    items = [(float(p), pauli_matrix(l)) for p, l in zip(probs, pauli_labels(k)) if abs(p) > 1e-15]
    return [Instruction("mixture", qubits, items, step, "twirl", "twirled noise")]


# ------------------------------------------------------------------ DD windows
DD_SEQUENCES = {"X2": (("x", 0.25), ("x", 0.75)),
                "XY4": (("x", 0.125), ("y", 0.375), ("x", 0.625), ("y", 0.875))}


def _idle_segment(noise: NoiseSpec, q: int, dt: float, delta: float, step: int) -> list:
    out = []
    if delta and dt > 0:
        out.append(Instruction("unitary", (q,), rz(delta * dt), step, "idle", "field"))
    if noise.idle_dephase and dt > 0:
        out.append(Instruction("kraus", (q,), kraus_phase(1 - (1 - noise.idle_dephase) ** dt), step, "idle", "dephasing"))
    if noise.idle_damp and dt > 0:
        out.append(Instruction("kraus", (q,), kraus_amp(1 - (1 - noise.idle_damp) ** dt), step, "idle", "damping"))
    return out


def crosstalk_instructions(noise: NoiseSpec, o: Op, step: int, busy_layer: set, nbrs: dict) -> list:
    """Spectator errors caused by driving the qubits of ``o`` (a physical gate): every neighbor that is idle in
    this layer picks up RZ(δx) and a ZZ phase ζ with the driven qubit."""
    if o.kind != "gate" or o.is_virtual or not (noise.xtalk_z or noise.xtalk_zz):
        return []
    out = []
    for a in o.qubits:
        for j in sorted(nbrs.get(a, ())):
            if j in busy_layer or j in o.qubits:
                continue
            if noise.xtalk_z:
                out.append(Instruction("unitary", (j,), rz(noise.xtalk_z), step, "crosstalk",
                                       "spectator Z %g on q%d" % (noise.xtalk_z, j)))
            if noise.xtalk_zz:
                out.append(Instruction("unitary", (a, j), rzz(noise.xtalk_zz), step, "crosstalk",
                                       "ZZ crosstalk %g q%d–q%d" % (noise.xtalk_zz, a, j)))
    return out


def idle_windows(spec: CircuitSpec):
    """For each qubit, maximal runs of idle layers after its first gate: {q: [(first, last), ...]}."""
    ops, lay, busy = layer_order(spec)
    nL = len(busy)
    first = {}
    for o, l in zip(ops, lay):
        if o.kind == "gate":
            for q in o.qubits:
                first.setdefault(q, l)
    out = {}
    for q in range(spec.n):
        if q not in first:
            continue
        runs, start = [], None
        for l in range(first[q], nL):
            idle = q not in busy[l]
            if idle and start is None:
                start = l
            if not idle and start is not None:
                runs.append((start, l - 1))
                start = None
        if start is not None:
            runs.append((start, nL - 1))
        out[q] = runs
    return out


# ------------------------------------------------------------------ resolve
def resolve(spec: CircuitSpec, noise: NoiseSpec, *, delta=None, ideal: bool = False, twirl: str = "none",
            dd: str = "none", dd_min: int = 2, corrections=None) -> tuple:
    """Implemented circuit + noise -> (instructions, executed ops, layer of each op).

    twirl        'none' | '2q' | 'all': Pauli-twirl the noise of Clifford two-qubit gates (and of
                 one-qubit gates for 'all') -- the exact average over twirl instances
    dd           'none' | 'X2' | 'XY4': pulses in idle windows of at least `dd_min` layers
    delta        per-qubit quasi-static detuning (rad per layer) for this Monte Carlo sample
    corrections  callable(op, step) -> list[Instruction] appended after the gate's noise (PEC)
    """
    ops_all, lay_all, busy = layer_order(spec)
    ops, lay = [], []
    for o, l in zip(ops_all, lay_all):
        if o.kind != "barrier":
            ops.append(o)
            lay.append(l)
    nL = len(busy)
    last_step_of_layer = {}
    for i, l in enumerate(lay):
        last_step_of_layer[l] = i
    windows = idle_windows(spec) if not ideal else {}
    xtalk = (noise.xtalk_z or noise.xtalk_zz) and not ideal
    nbrs = spec.neighbours() if xtalk else {}
    # idle work per layer, per qubit: list of (dt, pulse) segments
    idle_plan = {l: [] for l in range(nL)}
    for q, runs in windows.items():
        for a, b in runs:
            L = b - a + 1
            pulses = DD_SEQUENCES.get(dd, ()) if (dd != "none" and L >= dd_min) else ()
            times = [(t * L, g) for g, t in pulses]
            for l in range(a, b + 1):
                t0, t1 = l - a, l - a + 1
                segs, cur = [], t0
                for t, g in times:
                    if t0 <= t < t1:
                        segs.append((t - cur, None))
                        segs.append((0.0, g))
                        cur = t
                segs.append((t1 - cur, None))
                idle_plan[l].append((q, segs))
    out = []
    for i, o in enumerate(ops):
        if o.kind == "gate":
            out.append(Instruction("unitary", o.qubits, op_matrix(o), i, "gate", o.name))
            if not ideal:
                gn = gate_noise_instructions(noise, o, i)
                # only Clifford gates can be twirled exactly (the Paulis are undone by conjugation through the gate)
                tw = is_clifford(o) and ((twirl == "all" and o.k == 1) or (twirl in ("2q", "all") and o.k == 2))
                out += twirl_instructions(gn, o.qubits, i) if tw else gn
            if xtalk:
                out += crosstalk_instructions(noise, o, i, busy[lay[i]], nbrs)
            if corrections is not None:
                out += corrections(o, i)
        elif o.kind == "noise" and not ideal:
            out += explicit_noise(o, i)
        # idle noise after the last op of each layer
        l = lay[i]
        if not ideal and last_step_of_layer.get(l) == i:
            for q, segs in idle_plan[l]:
                d = float(delta[q]) if delta is not None else 0.0
                for dt, g in segs:
                    if g is None:
                        out += _idle_segment(noise, q, dt, d, i)
                    else:
                        po = Op(g, (q,), (), "gate", "dd")
                        out.append(Instruction("unitary", (q,), op_matrix(po), i, "dd", "DD " + g.upper()))
                        gn = gate_noise_instructions(noise, po, i)
                        tw = twirl == "all"
                        out += twirl_instructions(gn, (q,), i) if tw else gn
    return out, ops, lay


def field_deltas(noise: NoiseSpec, n: int, seed: int = 7) -> list:
    """Samples of the quasi-static detuning δ ~ N(0, σ²), one per qubit: Latin-hypercube sampling with the
    midpoint quantiles Φ⁻¹((k + ½)/M) of each qubit, independently permuted across qubits. Every qubit's
    marginal is then stratified (error O(1/M²) for smooth single-qubit effects), and the set is symmetric
    (zero mean). A single zero sample when σ = 0."""
    if not noise.field_sigma:
        return [None]
    from scipy.stats import norm
    m = max(2, int(noise.field_samples))
    z = norm.ppf((np.arange(m) + 0.5) / m)
    rng = np.random.default_rng(seed)
    cols = [z if q == 0 else rng.permutation(z) for q in range(n)]
    return [noise.field_sigma * np.array([c[k] for c in cols]) for k in range(m)]


def readout_matrix(noise: NoiseSpec) -> np.ndarray:
    """A[read, prepared] for one bit."""
    return np.array([[1 - noise.ro01, noise.ro10], [noise.ro01, 1 - noise.ro10]])


def apply_readout(p: np.ndarray, noise: NoiseSpec, n: int) -> np.ndarray:
    if not (noise.ro01 or noise.ro10):
        return p
    A = readout_matrix(noise)
    t = p.reshape([2] * n)
    for ax in range(n):
        t = np.moveaxis(np.tensordot(A, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def describe(noise: NoiseSpec) -> list:
    """Plain-language list of the active channels."""
    d = []
    if noise.p1 or noise.p2:
        d.append("depolarizing p₁ = %.3g, p₂ = %.3g" % (noise.p1, noise.p2))
    if noise.amp_damp:
        d.append("amplitude damping γ = %.3g per gate" % noise.amp_damp)
    if noise.phase_damp:
        d.append("phase damping λ = %.3g per gate" % noise.phase_damp)
    if noise.over_rot:
        d.append("over-rotation ε = %.3g" % noise.over_rot)
    if noise.zz_phase:
        d.append("ZZ phase θ = %.3g rad" % noise.zz_phase)
    if noise.z_phase:
        d.append("Z phase φ = %.3g rad" % noise.z_phase)
    if noise.kick:
        d.append("random kick κ = %.3g rad" % noise.kick)
    if noise.field_sigma:
        d.append("quasi-static field σ = %.3g rad/layer" % noise.field_sigma)
    if noise.idle_dephase or noise.idle_damp:
        d.append("idle dephasing %.3g, damping %.3g per layer" % (noise.idle_dephase, noise.idle_damp))
    if noise.ro01 or noise.ro10:
        d.append("readout ε₀₁ = %.3g, ε₁₀ = %.3g" % (noise.ro01, noise.ro10))
    if noise.xtalk_z or noise.xtalk_zz:
        d.append("crosstalk: spectator Z %.3g rad, ZZ %.3g rad per neighbor pulse" % (noise.xtalk_z, noise.xtalk_zz))
    return d or ["no noise"]
