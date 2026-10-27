r"""Model mismatch: hidden perturbations of the execution law, finite-shot diagnostics, and diagnostic-triggered
fallback (1.6.0, secondary theme).

The recommender and every model-based protocol (PEC's decomposition, the supplied-model allocations) use the
device's calibration snapshot. A mismatch scenario changes only the *execution law* (``TaskSpec.truth``); its
parameters are oracle information, never features.

=====================  ====================================================================================
scenario               execution law
=====================  ====================================================================================
``chi=c``              coherent fraction c of each gate's non-relaxation error turned into a unitary over-rotation
                       (one-qubit) or ZZ phase (two-qubit), total infidelity unchanged (exact engine)
``s=x``                static drift: every gate error and relaxation rate × x
``ro=x``               readout drift: every readout flip probability × x
``xt=θ``               ZZ crosstalk: after every two-qubit gate on (a, b), exp(−iθ/2 Z_q Z_c) between each of a, b
                       and its chain neighbours c in the register (always-on coupling during the gate)
``mix=x``              temporal drift: shots randomly interleaved between the snapshot law and the ``s=x`` law
                       (a static mixture — exact for random interleaving; not for a monotone drift within a run)
=====================  ====================================================================================

Diagnostics (acquired information; their shots are charged to B):

* **echo** — the compiled circuit followed by its inverse, from |0…0⟩, N_d shots; statistic z = (P̂(0…0) −
  P_sup)/√(P_sup(1 − P_sup)/N_d) against the supplied model's prediction of the same circuit *with the same
  measurement map and readout error* (matched conventions);
* **readout** — the protocol's own calibration counts (already paid) against the snapshot's flip probabilities:
  the largest |z| over the 2n binomial tests, Bonferroni-adjusted.

Both are compared with genuine probability distributions or binomial proportions only; no divergence is computed on
quasi-probabilities. Their null distributions are finite-shot (binomial) laws under the supplied model; the false
positive rate is checked by simulation under the null (``null_rate``). A non-rejection does not establish that the
model is correct (theory companion §1.7.3).
"""
from __future__ import annotations

import math
import re

import numpy as np

DEVICE_KEYS = ("s", "ro", "chi")
SCENARIOS = {
    "none": "",
    "coherent-0.25": "chi=0.25", "coherent-0.5": "chi=0.5",
    "drift-1.5": "s=1.5", "drift-2": "s=2",
    "readout-2": "ro=2",
    "crosstalk-0.02": "xt=0.02", "crosstalk-0.05": "xt=0.05",
    "temporal-2": "mix=2",
}


def parse_truth(truth: str) -> tuple:
    """(device modifiers string, {xt: θ, mix: x}) of a TaskSpec.truth string."""
    dev, extra = [], {}
    for part in [p for p in (truth or "").split(",") if p]:
        m = re.fullmatch(r"([a-z]{1,4})=([0-9.eE+\-]+)", part)
        if not m:
            raise ValueError("bad truth modifier %r" % part)
        k, v = m.group(1), m.group(2)
        if k in DEVICE_KEYS:
            dev.append(part)
        elif k in ("xt", "mix"):
            extra[k] = float(v)
        else:
            raise ValueError("unknown truth modifier %r" % k)
    return ",".join(dev), extra


def chain_neighbors(n: int) -> dict:
    """Neighbours of each compact qubit along the task's layout chain (the compiled register is a line)."""
    return {q: [c for c in (q - 1, q + 1) if 0 <= c < n] for q in range(n)}


def zz_unitary(theta: float) -> np.ndarray:
    return np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))


class CrosstalkRunner:
    """A runner whose execution law adds ZZ crosstalk after every two-qubit gate (scenario ``xt``)."""

    def __init__(self, base, theta: float, n: int):
        self.base, self.theta, self.nb = base, float(theta), chain_neighbors(n)
        self.backend, self.layout, self.cache = base.backend, base.layout, base.cache
        self.calls = 0

    @property
    def device_model(self):
        return self.base.device_model

    def _xt(self, inner):
        from ..noise import Instruction
        U = zz_unitary(self.theta)

        def f(o, step):
            out = list(inner(o, step)) if inner is not None else []
            if getattr(o, "kind", "gate") == "gate" and len(o.qubits) == 2:
                a, b = o.qubits
                for q in (a, b):
                    for c in self.nb.get(q, []):
                        if c not in (a, b):
                            out.append(Instruction("unitary", (q, c), U, step, "explicit", "ZZ crosstalk"))
            return out
        return f

    def run(self, spec, noise, measure_map, *, corrections=None, corr_key=None, **kw):
        if kw.get("ideal"):                              # ideal values (labels, training targets): no crosstalk
            return self.base.run(spec, noise, measure_map, corrections=corrections, corr_key=corr_key, **kw)
        return self.base.run(spec, noise, measure_map, corrections=self._xt(corrections),
                             corr_key=("xt", self.theta, corr_key), **kw)


class MixtureRunner:
    """A runner whose law is the w-weighted mixture of two runners' laws (scenario ``mix``): distributions and
    signed PEC laws mix linearly."""

    def __init__(self, a, b, w: float = 0.5):
        self.a, self.b, self.w = a, b, float(w)
        self.backend, self.layout = a.backend, a.layout
        self.cache = {}
        self.calls = 0

    @property
    def device_model(self):
        return self.a.device_model

    def run(self, spec, noise, measure_map, **kw):
        from copy import copy
        ra = self.a.run(spec, noise, measure_map, **kw)
        if kw.get("ideal"):
            return ra
        rb = self.b.run(spec, noise, measure_map, **kw)
        out = copy(ra)
        out.probs = (1 - self.w) * np.asarray(ra.probs) + self.w * np.asarray(rb.probs)
        out.probs_meas = (1 - self.w) * np.asarray(ra.probs_meas) + self.w * np.asarray(rb.probs_meas)
        out.rho = None
        return out


def truth_runner(spec, phys):
    """The runner of a task's execution law (device modifiers, crosstalk and temporal mixture applied)."""
    from ..methods import Runner
    dev, extra = parse_truth(spec.truth)
    from .task import with_modifiers
    base = Runner(backend=with_modifiers(spec.device, "eng=exact", dev), layout=phys)
    r = base
    if "mix" in extra:
        drift = Runner(backend=with_modifiers(spec.device, "eng=exact", "s=%r" % extra["mix"], dev),
                       layout=phys)
        r = MixtureRunner(base, drift, 0.5)
    if "xt" in extra:
        r = CrosstalkRunner(r, extra["xt"], len(phys))
    return r


# ------------------------------------------------------------------ diagnostics
def echo_circuit(compiled):
    """The compiled circuit followed by its inverse (gates reversed and inverted): ideal output |0…0⟩."""
    from ..circuit import inverse
    spec = compiled.implemented
    gates = [o for o in spec.ops if o.kind == "gate"]
    inv = tuple(inverse(o) for o in reversed(gates))
    return spec.replace(ops=tuple(spec.ops) + inv, name=spec.name + " · echo")


def echo_p0(runner, compiled, noise) -> float:
    rr = runner.run(echo_circuit(compiled), noise, compiled.measure_map, role="diagnostic")
    return float(np.clip(rr.probs_meas[0], 0.0, 1.0))


def echo_statistic(p_hat, p_sup: float, N: int) -> np.ndarray:
    se = math.sqrt(max(p_sup * (1 - p_sup), 1e-12) / max(int(N), 1))
    return (np.asarray(p_hat, float) - p_sup) / se


def echo_trials(p_true: float, p_sup: float, N: int, R: int, rng) -> np.ndarray:
    """R diagnostic statistics from N shots each under the execution law (binomial counts of 0…0)."""
    k = rng.binomial(int(N), p_true, size=int(R))
    return echo_statistic(k / float(N), p_sup, N)


def readout_statistic(theta_hat, theta_sup, s: int) -> np.ndarray:
    """max |z| over the 2n flip probabilities of R finite calibrations (rows) against the snapshot's."""
    th = np.atleast_2d(np.asarray(theta_hat, float))
    t0 = np.clip(np.asarray(theta_sup, float), 1e-6, 1 - 1e-6)
    z = (th - t0) / np.sqrt(t0 * (1 - t0) / max(int(s), 1))
    return np.max(np.abs(z), axis=1)


def flag_threshold(alpha: float, tests: int = 1, two_sided: bool = True) -> float:
    """Normal-approximation critical value with a Bonferroni split over ``tests`` statistics."""
    from scipy.stats import norm
    a = alpha / max(int(tests), 1)
    return float(norm.ppf(1 - a / 2) if two_sided else norm.ppf(1 - a))


def null_rate(p_sup: float, N: int, alpha: float, R: int = 20000, seed: int = 0) -> float:
    """Simulated false-positive rate of the echo diagnostic under the supplied law (finite-shot null)."""
    rng = np.random.default_rng(seed)
    z = echo_trials(p_sup, p_sup, N, R, rng)
    return float(np.mean(np.abs(z) > flag_threshold(alpha)))


def diag_shots(B: int, fraction: float = 0.02, minimum: int = 200) -> int:
    """Declared diagnostic budget: 2 % of B, at least 200 shots."""
    return int(max(minimum, round(fraction * int(B))))
