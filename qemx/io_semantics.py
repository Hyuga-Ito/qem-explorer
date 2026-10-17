"""Input and output semantics of an experiment (1.5).

The editable source of truth of an experiment is its **input** (the per-qubit input states plus any preparation gates
a template puts at the start of the circuit, tagged ``prep``), the **circuit**, its **parameters** and the
**measurement** (per-qubit basis and the classical bit order). The **output** is derived from these and is never an
editable control: reading or refreshing it changes nothing.

Three kinds of output are kept apart everywhere in the app:

* the *ideal prediction* — the noiseless output distribution (and observable value) of the logical circuit,
  computed here from the statevector when it is affordable;
* the *noisy simulation prediction* — the same under the selected noise model (the Playground's comparison);
* *sampled* or *measured outcomes* — finitely many shots, from a simulator or hardware.

An output is in general a probability distribution (or an expectation value), not one bitstring: ``derived_output``
says whether it is deterministic. Above ``OUTPUT_PREVIEW_MAX_QUBITS`` the interactive preview is not computed (2^n
amplitudes); the reason is reported instead of an invented answer. This is a limit of the preview only, not of the
experiments, which keep their own resource checks.
"""
from __future__ import annotations

import math

import numpy as np

from .circuit import CircuitSpec

OUTPUT_PREVIEW_MAX_QUBITS = 20          # 2^20 amplitudes (16 MiB): the interactive preview only, not an experiment limit
DETERMINISTIC_TOL = 1e-9


def preparation_ops(spec: CircuitSpec) -> list:
    """The template preparation: the leading operations tagged ``prep`` (a template's input-loading gates)."""
    out = []
    for o in spec.ops:
        if o.tag != "prep":
            break
        out.append(o)
    return out


def describe_input(spec: CircuitSpec, inputs) -> dict:
    """What the input is, in words: the per-qubit input states and the template's preparation gates."""
    from .compile import input_label
    inputs = list(inputs or ["0"] * spec.n)
    prep = preparation_ops(spec)
    per_qubit = [input_label(t) for t in inputs]
    gates = {}
    for o in prep:
        gates[o.name] = gates.get(o.name, 0) + 1
    return dict(per_qubit=per_qubit, nonzero=[q for q, t in enumerate(inputs) if t != "0"],
                prep_gates=len(prep), prep_summary=", ".join("%d %s" % (v, k.upper()) for k, v in gates.items()))


def measure_map(n: int, relabel: bool) -> dict:
    """Classical bit i ← qubit mm[i] of the logical circuit (relabel: bits read in reverse order)."""
    return {i: (n - 1 - i if relabel else i) for i in range(int(n))}


def ideal_statevector(spec: CircuitSpec) -> np.ndarray:
    """Noiseless statevector (little-endian amplitudes) of the gate operations; noise operations are skipped."""
    from .templates import statevector
    return statevector(spec.n, spec.ops)


def derived_output(logical: CircuitSpec, relabel: bool = False, observable=None, top: int = 8,
                   max_qubits: int = OUTPUT_PREVIEW_MAX_QUBITS) -> dict:
    """The ideal (noiseless) prediction derived from input preparation, circuit and measurement.

    ``logical`` must already contain the input preparation and the basis-change gates (``compile.with_io`` or
    ``Compiled.logical``). Returns ``available`` (False with a ``reason`` above ``max_qubits``), the distribution's
    ``top`` outcomes over classical bits (bitstrings written c_{n-1}…c_0), ``support`` (number of outcomes with
    probability > 1e-12), ``deterministic`` (one outcome with probability 1), ``entropy_bits`` of the classical
    distribution (not an entanglement measure), and ``observable_value`` for a diagonal observable.
    Pure: nothing passed in is modified."""
    n = int(logical.n)
    out = dict(kind="ideal prediction", n=n, available=False, reason="", top=[], support=None, deterministic=None,
               entropy_bits=None, observable_value=None)
    if n > int(max_qubits):
        out["reason"] = ("the exact ideal output of %d qubits has 2^%d = %s amplitudes; the interactive preview stops at "
                         "%d qubits. The comparison and the CLI compute the observable's ideal value where memory "
                         "allows." % (n, n, f"{2 ** n:,}", int(max_qubits)))
        return out
    try:
        psi = ideal_statevector(logical)
    except MemoryError:
        out["reason"] = "not enough memory for the statevector of %d qubits" % n
        return out
    p = np.abs(psi) ** 2
    p = p / p.sum()
    from .engine import permute_probs
    p = permute_probs(p, measure_map(n, relabel), n)
    nz = np.flatnonzero(p > 1e-12)
    order = nz[np.argsort(-p[nz], kind="stable")][: int(top)]
    out.update(available=True, top=[(format(int(i), "0%db" % n), float(p[i])) for i in order], support=int(len(nz)),
               deterministic=bool(len(nz) >= 1 and p[order[0]] >= 1 - DETERMINISTIC_TOL),
               entropy_bits=float(-np.sum(p[nz] * np.log2(p[nz]))))
    if observable is not None and not hasattr(observable, "terms"):
        vals = np.asarray(getattr(observable, "values", ()), float)
        if vals.shape == p.shape:
            out["observable_value"] = float(vals @ p)
    return out


def output_summary(d: dict) -> str:
    """One line for the Output panel."""
    if not d.get("available"):
        return "Ideal output not computed: %s." % d.get("reason", "unavailable")
    n = d["n"]
    if d["deterministic"]:
        s = "Deterministic: |%s⟩ with probability 1." % d["top"][0][0]
    elif d["support"] == 2 ** n and max(pr for _, pr in d["top"]) - min(pr for _, pr in d["top"]) < 1e-9 \
            and math.isclose(d["top"][0][1], 2.0 ** -n, rel_tol=1e-6):
        s = "Uniform over all %s outcomes (probability 1/%s each)." % (f"{2 ** n:,}", f"{2 ** n:,}")
    else:
        s = "A distribution over %s outcome%s; most likely %s." % (
            f"{d['support']:,}", "" if d["support"] == 1 else "s",
            ", ".join("|%s⟩ %.3f" % (b, pr) for b, pr in d["top"][:3]))
    if d.get("observable_value") is not None:
        s += " Ideal ⟨O⟩ = %.6g." % d["observable_value"]
    return s
