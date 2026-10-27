r"""Noise scaling for the λ axis of the phase diagrams, and NoiseSpec models calibrated to a fake device.

Two noise engines produce the λ sweep (see docs/research/direction_A_v1_design.md):

E2  the fake device itself, every gate error and every relaxation rate multiplied by s (``device_at``; the
    device specification ``fake_fez@s=2,eng=exact`` is executed exactly by ``qemx.device_exact``);
E1  a homogeneous :class:`~qemx.noise.NoiseSpec` calibrated to the same device's medians (``calibrated_noise``),
    with a *coherent fraction* χ of the gate infidelity that a fake device cannot express, scaled by
    ``scale_noise``.

Exact fault-probability scaling
-------------------------------
``scale_noise(spec, s)`` multiplies the **fault probability** of every elementary channel by exactly s — the
probability that its Pauli twirl applies a non-identity Pauli, the quantity summed in the circuit fault rate
:math:`\lambda=\sum_f p_f` (Cai et al., RMP 95, 045005 (2023)). Each parameter is mapped through its channel's
closed form:

==================  ==========================================  =============================================
channel             fault probability p_f                        scaled parameter
==================  ==========================================  =============================================
depolarizing p, k   p (1 − 4^{−k})                               p·s
amplitude damping γ (2 + γ − 2√(1−γ)) / 4                        root of p_f(γ') = s·p_f(γ)
phase damping λ     (1 − √(1−λ)) / 2                             λ' = 1 − (1 − 2 s p_f)²
rotation by θ       sin²(θ/2)  (RZ(θ), exp(−iθZZ/2), kick κ)     θ' = 2 asin(√s · sin(θ/2))  (sign kept)
over-rotation ε     sin²(εφ/2), φ = π/2 (√X, the reference)      ε' = (4/π) asin(√s · sin(επ/4))
quasi-static σ      (1 − e^{−σ²/2}) / 2 per idle layer           σ' = √(−2 ln(1 − s(1 − e^{−σ²/2})))
==================  ==========================================  =============================================

so coherent angles scale like √s for small angles, *not* like s (``NoiseSpec.scaled`` scales angles linearly,
which multiplies their fault probability by s², and must not be used for a λ sweep). Over-rotation of gates other
than √X (angle φ ≠ π/2) scales only to first order; the fault rate of a scaled circuit is always *measured*
(``qemx.phase.lightcone``), never assumed. A target outside a channel's range raises ``ValueError``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from ..noise import NoiseSpec


# ------------------------------------------------------------------ fault probabilities of elementary channels
def pf_depol(p: float, k: int) -> float:
    return float(p) * (1.0 - 4.0 ** (-int(k)))


def pf_amp(g: float) -> float:
    g = float(g)
    return (2.0 + g - 2.0 * math.sqrt(max(1.0 - g, 0.0))) / 4.0


def pf_phase(l: float) -> float:
    return (1.0 - math.sqrt(max(1.0 - float(l), 0.0))) / 2.0


def pf_angle(theta: float) -> float:
    return math.sin(float(theta) / 2.0) ** 2


def pf_field(sigma: float) -> float:
    return (1.0 - math.exp(-float(sigma) ** 2 / 2.0)) / 2.0


def _inv_amp(target: float) -> float:
    """γ with pf_amp(γ) = target (pf_amp increases from 0 at γ=0 to 3/4 at γ=1)."""
    if target <= 0:
        return 0.0
    if target >= 0.75:
        raise ValueError("amplitude-damping fault probability %.4g is out of range (max 3/4)" % target)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if pf_amp(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _inv_phase(target: float) -> float:
    if target <= 0:
        return 0.0
    if target > 0.5:
        raise ValueError("phase-damping fault probability %.4g is out of range (max 1/2)" % target)
    return 1.0 - (1.0 - 2.0 * target) ** 2


def _scale_angle(theta: float, s: float) -> float:
    if theta == 0:
        return 0.0
    x = math.sqrt(s) * abs(math.sin(theta / 2.0))
    if x > 1.0:
        raise ValueError("coherent angle %.4g cannot be scaled by %g (fault probability would exceed 1)" % (theta, s))
    return math.copysign(2.0 * math.asin(x), theta)


def _scale_prob(p: float, s: float, name: str, cap: float = 1.0) -> float:
    v = float(p) * s
    if v > cap + 1e-15:
        raise ValueError("%s = %.4g × %g exceeds its maximum %.4g" % (name, p, s, cap))
    return min(v, cap)


def scale_noise(spec: NoiseSpec, s: float, readout_scale: float = 1.0) -> NoiseSpec:
    """The NoiseSpec whose every gate and idle channel has exactly s times the fault probability (table above).
    Readout flip probabilities are multiplied by ``readout_scale`` (default 1: unchanged, so λ is the gate and idle
    fault rate alone). ``field_samples`` is kept."""
    s = float(s)
    if not (math.isfinite(s) and s > 0):
        raise ValueError("scale factor must be finite and > 0")
    if s == 1.0 and readout_scale == 1.0:
        return spec
    f_sx = abs(math.sin(spec.over_rot * math.pi / 4.0))
    if f_sx * math.sqrt(s) > 1.0:
        raise ValueError("over-rotation %.4g cannot be scaled by %g" % (spec.over_rot, s))
    over = math.copysign(4.0 / math.pi * math.asin(math.sqrt(s) * f_sx), spec.over_rot) if spec.over_rot else 0.0
    sig = 0.0
    if spec.field_sigma:
        t = s * pf_field(spec.field_sigma)
        if t >= 0.5:
            raise ValueError("quasi-static field σ = %.4g cannot be scaled by %g" % (spec.field_sigma, s))
        sig = math.sqrt(-2.0 * math.log(1.0 - 2.0 * t))
    ro = float(readout_scale)
    if not (math.isfinite(ro) and ro >= 0):
        raise ValueError("readout_scale must be finite and >= 0")
    if spec.ro01 * ro >= 0.5 or spec.ro10 * ro >= 0.5:
        raise ValueError("readout flip probability × %g reaches 1/2" % ro)
    return spec.replace(
        p1=_scale_prob(spec.p1, s, "p1"), p2=_scale_prob(spec.p2, s, "p2"),
        amp_damp=_inv_amp(s * pf_amp(spec.amp_damp)), phase_damp=_inv_phase(s * pf_phase(spec.phase_damp)),
        over_rot=over, zz_phase=_scale_angle(spec.zz_phase, s), z_phase=_scale_angle(spec.z_phase, s),
        kick=_scale_angle(spec.kick, s), field_sigma=sig,
        idle_dephase=_inv_phase(s * pf_phase(spec.idle_dephase)), idle_damp=_inv_amp(s * pf_amp(spec.idle_damp)),
        ro01=spec.ro01 * ro, ro10=spec.ro10 * ro,
        xtalk_z=_scale_angle(spec.xtalk_z, s), xtalk_zz=_scale_angle(spec.xtalk_zz, s))


# ------------------------------------------------------------------ calibration from a device
@dataclass(frozen=True)
class Calibration:
    """Median calibration numbers of a device (on a set of qubits): what E1 is calibrated to."""
    source: str                 # device specification and scope
    r1: float                   # average gate infidelity of √X (median)
    r2: float                   # average gate infidelity of the native two-qubit gate (median over couplers used)
    t1: float                   # s (median)
    t2: float                   # s (median), ≤ 2 t1
    t_1q: float                 # s, √X duration (median)
    t_2q: float                 # s, two-qubit gate duration (median)
    ro01: float                 # mean P(1|0)
    ro10: float                 # mean P(0|1)
    two_qubit: str = "cz"

    def to_json(self) -> dict:
        return asdict(self)


def calibration_from_model(dm, source: str | None = None) -> Calibration:
    """Medians of a :class:`~qemx.device.DeviceModel` (the qubits and couplers it covers)."""
    e1 = [v for (nm, qa), v in dm.errors.items() if nm == "sx" and v is not None]
    e2 = [v for (nm, qa), v in dm.errors.items() if nm == dm.two_qubit and v is not None]
    d1 = [v for (nm, qa), v in dm.durations.items() if nm == "sx" and v]
    d2 = [v for (nm, qa), v in dm.durations.items() if nm == dm.two_qubit and v]
    if not e1 or not e2 or not d1 or not d2:
        raise ValueError("the device model lacks √X or two-qubit calibration data")
    t1 = float(np.median([q.t1 for q in dm.qubits]))
    t2 = min(float(np.median([q.t2 for q in dm.qubits])), 2 * t1)
    return Calibration(source or dm.name, float(np.median(e1)), float(np.median(e2)), t1, t2,
                       float(np.median(d1)), float(np.median(d2)),
                       float(np.mean([q.p01 for q in dm.qubits])), float(np.mean([q.p10 for q in dm.qubits])),
                       dm.two_qubit)


def _depol_for(r: float, d: int, F_U: float) -> float:
    """Depolarizing p such that (unitary error with process fidelity F_U) followed by depolarizing p has average
    gate infidelity r:  p = (F_U − F_target)/(F_U − 1/d²),  F_target = 1 − r(d+1)/d."""
    F_t = 1.0 - r * (d + 1) / d
    if F_U < F_t - 1e-15:
        raise ValueError("the coherent part alone exceeds the target infidelity")
    p = (F_U - F_t) / (F_U - 1.0 / d ** 2)
    return max(p, 0.0)


def calibrated_noise(cal: Calibration, coherent_fraction: float = 0.0, idle_layer: str = "2q",
                     readout: bool = True) -> NoiseSpec:
    """E1: a homogeneous NoiseSpec with the device's median rates.

    * one-qubit pulses (√X): average gate infidelity r1; two-qubit gates: r2 — split into a coherent part (a fraction
      χ = ``coherent_fraction`` of the infidelity: over-rotation ε of √X, (2/3) sin²(επ/4) = χ r1; a ZZ phase θ after
      every two-qubit gate, (4/5) sin²(θ/2) = χ r2) and depolarizing p making up the rest *exactly*
      (``_depol_for``; relaxation during the gate is part of r, as in the device's reported error);
    * idle qubits: amplitude damping γ = 1 − e^{−t/T1} and phase damping λ = 1 − e^{−t(2/T2 − 1/T1)} per idle
      layer of duration t = the median two-qubit gate duration (``idle_layer="2q"``) or √X duration ("1q");
    * readout: mean P(1|0), P(0|1) (``readout=False``: none).

    Note: over-rotation also applies to X pulses (twice the √X angle), so X carries a larger coherent error than √X.
    """
    chi = float(coherent_fraction)
    if not 0.0 <= chi <= 1.0:
        raise ValueError("coherent_fraction must be in [0, 1]")
    # coherent parts
    a1 = 2.0 * math.asin(math.sqrt(1.5 * chi * cal.r1)) if chi else 0.0          # √X over-rotation angle
    eps = a1 / (math.pi / 2.0)
    th = 2.0 * math.asin(math.sqrt(1.25 * chi * cal.r2)) if chi else 0.0          # ZZ phase
    p1 = _depol_for(cal.r1, 2, math.cos(a1 / 2.0) ** 2)
    p2 = _depol_for(cal.r2, 4, math.cos(th / 2.0) ** 2)
    t = cal.t_2q if idle_layer == "2q" else cal.t_1q
    rate_phi = 2.0 / cal.t2 - 1.0 / cal.t1
    if rate_phi < -1e-12:
        raise ValueError("T2 > 2 T1")
    return NoiseSpec(p1=p1, p2=p2, over_rot=eps, zz_phase=th,
                     idle_damp=1.0 - math.exp(-t / cal.t1), idle_dephase=1.0 - math.exp(-t * max(rate_phi, 0.0)),
                     ro01=cal.ro01 if readout else 0.0, ro10=cal.ro10 if readout else 0.0, field_samples=1)


def gate_infidelities(noise: NoiseSpec) -> dict:
    """Average gate infidelity of a √X and of a CZ under ``noise`` (exact, from the composed superoperator)."""
    from ..noise import gate_noise_superop, ptm
    out = {}
    for k, name in ((1, "sx"), (2, "cz")):
        S = gate_noise_superop(noise, k, name)
        d = 2 ** k
        Fe = float(np.real(np.trace(ptm(S)))) / d ** 2
        out["r%d" % k] = 1.0 - (d * Fe + 1.0) / (d + 1.0)
    return out


# ------------------------------------------------------------------ the fake-device wrapper (E2)
def device_at(base: str, s: float = 1.0, readout_scale: float = 1.0, engine: str = "exact") -> str:
    """Device specification of ``base`` with gate errors and relaxation rates × s and readout × readout_scale,
    executed by ``engine`` ("exact": NumPy superoperators, signed maps allowed; "aer": Qiskit Aer). Use the result as
    the ``backend`` of :func:`qemx.recipes.compare`; compile with the base device (the circuit must not change with
    s)."""
    from ..device import device_spec
    return device_spec(base, s=s, ro=readout_scale, eng=engine)
