"""Time travel: the state after every circuit step, noisy and ideal, and what each step did.

Also builds the implemented circuit a method would run (a preview that needs no simulation), so
the Circuits tab can show logical and implemented circuits side by side before anything runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import CircuitSpec, Op
from .compile import Compiled
from .engine import bloch_vectors, fidelity, purity, simulate
from .noise import NoiseSpec
from .transforms import clifford_replace, dd_display, fold_global, fold_local, twirl_instance, achieved_lambda

INSPECT = {
    "implemented": "Implemented circuit",
    "zne": "ZNE: folded circuit",
    "dd": "Dynamical decoupling",
    "twirl": "Pauli-twirled noise",
}


@dataclass
class Timeline:
    spec: CircuitSpec             # the circuit executed (layer order is used for the steps)
    ops: list                     # executed ops, one per step
    labels: list                  # text for each step
    bloch_noisy: np.ndarray       # (steps + 1, n, 3), index 0 = initial state
    bloch_ideal: np.ndarray
    fid: np.ndarray               # (steps + 1,)
    pur: np.ndarray
    rho_noisy: list | None        # density matrices (steps + 1); None in compact mode
    rho_ideal: list | None
    probs_meas: np.ndarray
    note: str = ""
    compact: bool = False         # large registers: per-step summaries only (Bloch vectors, fidelity, purity)


def build(compiled: Compiled, noise: NoiseSpec, kind: str = "implemented", lam: float = 3.0, folding: str = "global",
          dd: str = "X2", backend: str = "numpy", compact: bool | None = None) -> Timeline:
    spec = compiled.implemented
    twirl, ddseq, note = "none", "none", ""
    if kind == "zne":
        spec = fold_global(spec, lam) if folding == "global" else fold_local(spec, lam, 0)
        base = compiled.implemented
        note = "λ achieved = %.3g for requested %.3g (two-qubit gates ×%.3g, all gates ×%.3g)" % (
            achieved_lambda(base, spec), lam, spec.count("2q") / max(base.count("2q"), 1),
            spec.count("all") / max(base.count("all"), 1))
    elif kind == "dd":
        ddseq = dd
        note = "DD pulses act inside idle windows; they appear in the step of the layer they end."
    elif kind == "twirl":
        twirl = "all"
        note = "Exact average over twirl instances: every gate's noise replaced by its Pauli twirl."
    from .resources import timeline_should_be_compact
    from . import device
    dev = device.is_device(backend)
    if compact is None:
        compact = timeline_should_be_compact(spec.n, len(spec.ops)) and not dev
    if dev:
        compact = False                  # the device model runs on Aer with per-step saves
        note = (note + " " if note else "") + "Device model of %s on physical qubits %s." % (
            device.device_label(backend), ", ".join(map(str, compiled.info.get("phys", []))))
    if compact:
        return _build_compact(compiled, spec, noise, twirl, ddseq, note, backend)
    noisy = simulate(spec, noise, backend, measure_map=compiled.measure_map, snapshots=True, twirl=twirl, dd=ddseq,
                     layout=compiled.info.get("phys") if dev else None)
    ideal = simulate(spec, noise, "numpy", measure_map=compiled.measure_map, snapshots=True, ideal=True)
    n = spec.n
    d = 2 ** n
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[0, 0] = 1
    steps = len(noisy.ops)
    rn, ri = [rho0], [rho0]
    for k in range(steps):
        rn.append(noisy.snapshots.get(k, rn[-1]))
        ri.append(ideal.snapshots.get(k, ri[-1]))
    bn = np.array([bloch_vectors(r, n) for r in rn])
    bi = np.array([bloch_vectors(r, n) for r in ri])
    fid = np.array([fidelity(a, b) for a, b in zip(rn, ri)])
    pur = np.array([purity(r) for r in rn])
    labels = ["start |0…0⟩"] + ["%s%s" % (o.text(), (" · " + o.tag) if o.tag else "") for o in noisy.ops]
    return Timeline(spec, noisy.ops, labels, bn, bi, fid, pur, rn, ri, noisy.probs_meas, note)


def _build_compact(compiled, spec, noise, twirl, ddseq, note, backend) -> Timeline:
    """Per-step summaries without keeping 2·(steps+1) density matrices: the ideal pass keeps the pure state
    vector of every step (2^n amplitudes), the noisy pass accumulates Bloch vectors, the fidelity with the ideal
    state and the purity through a hook in the NumPy executor."""
    n = spec.n
    psis = {}

    def ideal_hook(step, rho):
        k = int(np.argmax(np.real(np.diag(rho))))
        psis[step] = rho[:, k] / np.sqrt(max(float(np.real(rho[k, k])), 1e-300))

    ideal = simulate(spec, noise, "numpy", measure_map=compiled.measure_map, snapshots=ideal_hook, ideal=True)
    acc = {}

    def noisy_hook(step, rho):
        b = bloch_vectors(rho, n)
        psi = psis.get(step)
        f = float(np.real(np.conj(psi) @ (rho @ psi))) if psi is not None else float("nan")
        pu = purity(rho)
        if step in acc:
            a = acc[step]
            a[0] += b
            a[1] += f
            a[2] += pu
        else:
            acc[step] = [b, f, pu]

    noisy = simulate(spec, noise, "numpy", measure_map=compiled.measure_map, snapshots=noisy_hook, twirl=twirl,
                     dd=ddseq)
    m = noisy.samples
    steps = len(noisy.ops)
    b0 = np.zeros((n, 3))
    b0[:, 2] = 1.0
    bn, bi, fid, pur = [b0], [b0], [1.0], [1.0]
    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1
    last_psi = psi0
    for k in range(steps):
        if k in acc:
            bn.append(acc[k][0] / m)
            fid.append(acc[k][1] / m)
            pur.append(acc[k][2] / m)
        else:
            bn.append(bn[-1]); fid.append(fid[-1]); pur.append(pur[-1])
        psi = psis.get(k, last_psi)
        last_psi = psi
        bi.append(bloch_vectors(np.outer(psi, psi.conj()), n) if n <= 12 else _bloch_from_psi(psi, n))
    labels = ["start |0…0⟩"] + ["%s%s" % (o.text(), (" · " + o.tag) if o.tag else "") for o in noisy.ops]
    extra = ("Compact mode: %d qubits make a density matrix %s, so per-step states are summarized "
             "(Bloch vectors, fidelity, purity%s) instead of stored." % (
                 n, _human(16 * 4 ** n), "; with field sampling the purity is the mean over samples" if m > 1 else ""))
    if backend != "numpy":
        extra += " Time travel used the NumPy executor."
    return Timeline(spec, noisy.ops, labels, np.array(bn), np.array(bi), np.array(fid), np.array(pur), None, None,
                    noisy.probs_meas, (note + " " if note else "") + extra, compact=True)


def _bloch_from_psi(psi: np.ndarray, n: int) -> np.ndarray:
    t = psi.reshape([2] * n)
    out = np.zeros((n, 3))
    for q in range(n):
        ax = n - 1 - q
        a = np.moveaxis(t, ax, 0).reshape(2, -1)
        r = a @ a.conj().T
        out[q] = [2 * np.real(r[0, 1]), -2 * np.imag(r[0, 1]), np.real(r[0, 0] - r[1, 1])]
    return out


def _human(b):
    from .resources import human
    return human(b)


# ------------------------------------------------------------------ previews
def preview(kind: str, compiled: Compiled, noise: NoiseSpec, *, lam=3.0, folding="global", dd="X2", seed=0,
            p2=None, pec_min1=True) -> tuple:
    """(title, CircuitSpec) of what a method runs, without simulating."""
    spec = compiled.implemented
    if kind == "implemented":
        return "Implemented circuit", spec
    if kind == "zne":
        f = fold_global(spec, lam) if folding == "global" else fold_local(spec, lam, seed)
        return "ZNE, %s folding, λ = %.3g" % (folding, achieved_lambda(spec, f)), f
    if kind == "pec":
        from .methods.pec import depol_quasi
        rng = np.random.default_rng(seed)
        slots = []
        for o in spec.ops:
            if o.kind == "gate" and not o.is_virtual:
                p = (noise.p2 if p2 is None else p2) if o.k == 2 else noise.p1
                qs = depol_quasi(float(p), o.k, "inverse")
                w = np.array([abs(v) for v, _ in qs])
                slots.append((qs, w / w.sum()))
            else:
                slots.append(None)
        # P(an instance inserts nothing): the identity branch is drawn at every gate
        p_none = float(np.prod([s[1][0] for s in slots if s is not None])) if any(slots) else 1.0
        for _ in range(500 if pec_min1 else 1):             # optionally condition on at least one insertion
            ops, sign, nins = [], 1, 0
            for o, s in zip(spec.ops, slots):
                ops.append(o)
                if s is None:
                    continue
                j = rng.choice(len(s[0]), p=s[1])
                sign *= int(np.sign(s[0][j][0]))
                for ch, q in zip(s[0][j][1], o.qubits):
                    if ch != "I":
                        ops.append(Op(ch.lower(), (q,), (), "gate", "pec"))
                        nins += 1
            if nins:
                break
        title = "PEC instance: %d Pauli insertion%s, sign %+d" % (nins, "" if nins == 1 else "s", sign)
        title += " (%.0f%% of instances insert nothing%s)" % (100 * p_none, "; showing one that does" if pec_min1 and nins
                                                                else "")
        return title, spec.replace(ops=ops)
    if kind == "twirl":
        return "One Pauli-twirl instance", twirl_instance(spec, seed)
    if kind == "dd":
        return "Dynamical decoupling (%s)" % dd, dd_display(spec, dd)
    if kind == "cdr":
        t = clifford_replace(spec, np.random.default_rng(seed))
        return "CDR training circuit (Clifford angles)", t
    if kind == "vd":
        n = spec.n
        ops = [Op(o.name, tuple(q + m * n for q in o.qubits), o.params, o.kind, o.tag) for m in range(2) for o in spec.ops]
        ops.append(Op("barrier", tuple(range(2 * n)), (), "barrier"))
        ops += [Op("derange", (q, q + n), (), "box", "vd") for q in range(n)]
        return "Virtual distillation: 2 copies + derangement", CircuitSpec(2 * n, tuple(ops), "VD")
    raise ValueError(kind)


PREVIEWS = {"implemented": "Compiled only", "zne": "ZNE (folding)", "pec": "PEC (sampled instance)",
            "twirl": "Pauli twirling", "dd": "Dynamical decoupling", "cdr": "CDR / FR training circuit",
            "vd": "Virtual distillation"}
