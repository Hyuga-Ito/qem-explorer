r"""Live parameters of variational templates (VQE, QAOA): rewrite the circuit's angles, and scan the energy
landscape E(θ) with and without noise.

A template lists its parameters as ``dict(name, value, slots=((op index, factor), …), lo, hi)``; parameter θ sets
``params = (factor · θ,)`` on every op in its slots (QAOA's γ drives RZZ(−γ) on every edge, β drives RX(2β)).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .circuit import CircuitSpec

MAX_SCAN_POINTS = 2500


def apply(spec: CircuitSpec, params, values) -> CircuitSpec:
    """The circuit with the parameters set to ``values``."""
    ops = list(spec.ops)
    for p, v in zip(params, values):
        for i, f in p["slots"]:
            if 0 <= i < len(ops) and len(ops[i].params) == 1:
                ops[i] = dataclasses.replace(ops[i], params=(float(f) * float(v),))
    return spec.replace(ops=tuple(ops))


def valid(spec: CircuitSpec, params) -> bool:
    """The slots still point at one-angle gates (the circuit was not edited since the template loaded)."""
    try:
        return all(0 <= i < len(spec.ops) and spec.ops[i].kind == "gate" and len(spec.ops[i].params) == 1
                   for p in params for i, _ in p["slots"])
    except Exception:                                       # noqa: BLE001 - malformed parameter records
        return False


def energy(make_compiled, spec, params, values, noise, observable, backend="numpy"):
    """(ideal, noisy) ⟨O⟩ at these values, exact means (no shot noise; noisy includes readout error)."""
    from .recipes import compare, preset
    comp = make_compiled(apply(spec, params, values))
    ideal, outs, _ = compare(comp, noise, observable, [preset("Unmitigated")], backend)
    return float(ideal), float(outs[0].result.mean) if outs and outs[0].result is not None else float("nan")


def scan(make_compiled, spec, params, values, noise, observable, axes, backend="numpy", progress=None) -> dict:
    """E over a grid. ``axes`` is [(parameter index, grid values)] with one or two entries."""
    from .resources import check
    grids = [np.asarray(g, float) for _, g in axes]
    shape = tuple(len(g) for g in grids)
    if int(np.prod(shape)) > MAX_SCAN_POINTS:
        raise ValueError("a scan has at most %d points" % MAX_SCAN_POINTS)
    ideal = np.full(shape, np.nan)
    noisy = np.full(shape, np.nan)
    total = int(np.prod(shape))
    for flat, idx in enumerate(np.ndindex(*shape)):
        check()                                             # the session's time limit
        vals = list(values)
        for (k, _), g, j in zip(axes, grids, idx):
            vals[k] = g[j]
        ideal[idx], noisy[idx] = energy(make_compiled, spec, params, vals, noise, observable, backend)
        if progress is not None and (flat % 10 == 0 or flat == total - 1):
            progress((flat + 1) / total)
    return dict(axes=[k for k, _ in axes], grids=[g.tolist() for g in grids], ideal=ideal, noisy=noisy)
