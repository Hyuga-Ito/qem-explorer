"""QEM methods. Importing this package registers every estimator in REGISTRY."""
from .base import (REGISTRY, FAMILY_LABEL, Context, MethodResult, Param, QEMMethod, Recipe, Runner,  # noqa: F401
                   TWIRL, DD, READOUT, register)
from . import raw, zne, pec, learning, nil, symmetry, vd  # noqa: F401
from . import compose  # noqa: F401,E402 - after its components (it copies their parameter schemas)

# Transforms and readout mitigations are recipe options rather than estimators; their theory lives here.
TRANSFORM_INFO = {
    "twirl": dict(
        label="Pauli twirling / randomized compiling",
        tex=r"\mathcal T(\mathcal E)=\frac{1}{4^n}\sum_P P^\dagger\,\mathcal E(P\,\cdot\,P^\dagger)\,P",
        why=r"Sandwiching each Clifford gate between random Paulis (compensated after the gate) turns any noise "
            r"into a Pauli channel with the same average fidelity: coherent errors, which add in amplitude "
            r"($\propto N^2$), become stochastic ones, which add in probability ($\propto N$). ZNE and PEC then "
            r"see a noise they can model (Wallman & Emerson, PRA 94, 052325 (2016))."),
    "dd": dict(
        label="Dynamical decoupling",
        tex=r"e^{-i\delta Z\tau/2}\,X\,e^{-i\delta Z\tau}\,X\,e^{-i\delta Z\tau/2}=\mathbb 1",
        why=r"An idle qubit with a slow detuning $\delta$ accumulates a phase $\delta\tau$. Two X pulses in the "
            r"idle window reverse the sign of that phase halfway, so a static detuning cancels (a spin echo). "
            r"Fast (Markovian) dephasing and damping are not refocused, and each pulse brings its own gate error "
            r"(Viola & Lloyd, PRA 58, 2733 (1998); Ezzell et al., PR Applied 20, 064027 (2023))."),
    "readout": dict(
        label="Readout mitigation (tensored, M3, TREX)",
        tex=r"\hat{\vec p}=A^{-1}\vec p_{\rm meas},\quad A=\bigotimes_iA_i;\qquad "
            r"\langle Z_S\rangle_{\rm TREX}=\frac{\langle Z_S\rangle_{\rm meas}}{\langle Z_S\rangle_{\rm cal}}",
        why=r"Readout flips bits after the circuit, which calibration circuits measure. Inverting the assignment "
            r"matrix gives a quasi-distribution (negative entries allowed) at a cost "
            r"$\Gamma=\prod_i\frac{1+|\epsilon_i-\eta_i|}{1-\epsilon_i-\eta_i}$. M3 solves only on the observed "
            r"bitstrings (Nation et al., PRX Quantum 2, 040326 (2021)); TREX twirls the measurement so the "
            r"confusion becomes symmetric and each Z-string is divided by one calibration number "
            r"(van den Berg, Minev & Temme, PRA 105, 032620 (2022))."),
}
