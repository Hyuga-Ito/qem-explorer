r"""The versioned feature dictionary of the main study (``qemx.features/1``).

Every feature has a definition, a normalization, a computational cost, a level and an availability rule:

* **logical** — the circuit as written (before compilation): independent of the device;
* **compiled** — after routing to a device or lattice (device- and layout-specific);
* **observable** — the measured quantity;
* **state** — properties of the ideal pure state. These are exact and cost O(2^n), so they are **not scalable**:
  available only where the statevector is affordable. The scalable proxies are the logical features;
* **noise** — light-cone fault rates of the noise model or calibration (``qemx.phase.lightcone``). They need the
  noise model, not the ideal value.
* **oracle** — quantities that need the exact ideal or noisy simulation (ideal value, noisy-state fidelity,
  purity). They are recorded for analysis but **never allowed as recommendation inputs**: a recommender that
  needed them could not run on a new circuit.

Missing values are NaN with a reason (``missing``), never zero. Two-qubit gate count is not entanglement: the
entanglement features are von Neumann entropies of reduced states of the *pure* ideal state (a valid entanglement
measure for pure states) and the Meyer–Wallach measure. The classical output entropy is *not* an entanglement
measure and is labeled so.

``from_record`` reads a phase-study cell record (no simulation: figures and models can be regenerated from stored
data). ``vector`` returns the inputs allowed at recommendation time, with per-qubit normalized variants whose value
for cross-width comparison is tested in the study (leave-width-out evaluation), not assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

VERSION = "qemx.features/1"
LEVELS = ("logical", "compiled", "observable", "state", "noise", "context", "oracle")


@dataclass(frozen=True)
class FeatureDef:
    name: str
    level: str
    definition: str
    normalization: str = "none"
    cost: str = "O(gates)"
    scalable: bool = True
    unit: str = ""
    provenance: str = ""

    @property
    def recommendable(self) -> bool:
        return self.level != "oracle"


D = FeatureDef
DICTIONARY = (
    D("n", "logical", "number of qubits of the logical circuit (width)", unit="qubits",
      provenance="cell.n"),
    D("two_qubit_logical", "logical", "two-qubit gates of the logical circuit", provenance="descriptors"),
    D("two_qubit_logical_per_qubit", "logical", "two_qubit_logical / n", normalization="per qubit"),
    D("non_clifford", "logical", "non-Clifford gates of the logical circuit (what CDR/FR/NIL training replaces)",
      provenance="descriptors"),
    D("non_clifford_per_qubit", "logical", "non_clifford / n", normalization="per qubit"),
    D("two_qubit", "compiled", "two-qubit gates after compilation (routing included)", provenance="circuit"),
    D("depth", "compiled", "depth of the compiled circuit (layers of physical gates)", provenance="circuit"),
    D("pulses_1q", "compiled", "one-qubit physical gates after compilation", provenance="descriptors"),
    D("routing_overhead", "compiled", "two-qubit gates added by routing (compiled − logical)", provenance="circuit"),
    D("routing_ratio", "compiled", "two_qubit / max(two_qubit_logical, 1)", normalization="ratio"),
    D("two_qubit_density", "compiled", "two_qubit / (n · depth): share of qubit-layer slots in two-qubit gates",
      normalization="per qubit-layer"),
    D("idle_fraction", "compiled", "share of (qubit, layer) slots after a qubit's first gate in which it idles",
      normalization="fraction", provenance="descriptors"),
    D("obs_weight", "observable", "largest Z-string weight in the observable's decomposition",
      provenance="descriptors"),
    D("obs_terms", "observable", "number of Z strings in the observable's decomposition", provenance="descriptors"),
    D("obs_locality", "observable", "obs_weight / n", normalization="per qubit"),
    D("meyer_wallach", "state", "Meyer–Wallach Q = 2(1 − mean_k Tr ρ_k²) of the ideal pure state", cost="O(2^n)",
      scalable=False, provenance="descriptors"),
    D("entropy_half", "state", "von Neumann entropy (bits) of the first ⌊n/2⌋ qubits of the ideal pure state",
      cost="O(2^n)", scalable=False, unit="bits", provenance="descriptors"),
    D("entropy_half_per_qubit", "state", "entropy_half / ⌊n/2⌋ (fraction of the maximum)", normalization="per qubit",
      cost="O(2^n)", scalable=False),
    D("entropy_max_cut", "state", "largest entropy over contiguous bipartitions of the logical line", cost="O(2^n)",
      scalable=False, unit="bits", provenance="descriptors"),
    D("entropy_profile_max", "state", "largest half-cut entropy (bits) of the ideal pure state after any layer of "
      "the logical circuit: entanglement the final state may conceal (the known-answer QFT ends in a basis state)",
      cost="O(layers · 2^n)", scalable=False, unit="bits", provenance="qemx.features.entanglement_profile"),
    D("output_entropy_bits", "state", "Shannon entropy of the ideal computational-basis output distribution — a "
      "classical quantity, NOT an entanglement measure", cost="O(2^n)", scalable=False, unit="bits",
      provenance="descriptors"),
    D("lam_lc", "noise", "light-cone fault rate λ_LC: summed error probability of the operations in the "
      "observable's backward light cone (gates and idles)", provenance="lambda"),
    D("lam_lc_gate", "noise", "λ_LC from gate errors", provenance="lambda"),
    D("lam_lc_idle", "noise", "λ_LC from idle (relaxation/dephasing) windows", provenance="lambda"),
    D("lam_ro", "noise", "readout fault rate of the measured qubits", provenance="lambda"),
    D("idle_share", "noise", "lam_lc_idle / lam_lc", normalization="fraction"),
    D("lam_per_2q", "noise", "lam_lc / max(light-cone two-qubit gates, 1)", normalization="per two-qubit gate"),
    D("scale", "context", "noise scale s applied to the device model (1 = as calibrated)", provenance="cell"),
    D("coherent_fraction", "context", "share χ of each gate error that is coherent (engine E1)", provenance="cell"),
    D("log10_budget", "context", "log10 of the total shot budget B of the estimate", unit="log10 shots"),
    D("ideal", "oracle", "exact ideal value of the observable", cost="O(2^n)", scalable=False),
    D("noisy_fidelity", "oracle", "fidelity of the exact noisy state to the ideal state", cost="O(4^n)",
      scalable=False),
    D("noisy_purity", "oracle", "purity of the exact noisy state", cost="O(4^n)", scalable=False),
)
BY_NAME = {f.name: f for f in DICTIONARY}


PROFILE_MAX_QUBITS = 14


def entanglement_profile(spec) -> list:
    """Half-cut von Neumann entropy (bits) of the ideal pure state after each layer of the circuit (gates only;
    layers as in ``circuit.layer_order``). Exact; O(layers · 2^n)."""
    import numpy as np
    from .circuit import layer_order
    from .phase.descriptors import _entropy, _reduced
    from .templates import statevector
    n = spec.n
    ops, lay, _ = layer_order(spec)
    gates = [(o, l) for o, l in zip(ops, lay) if o.kind == "gate"]
    out, done = [], []
    for L in sorted(set(l for _, l in gates)):
        done += [o for o, l in gates if l == L]
        psi = statevector(n, done)
        out.append(_entropy(_reduced(psi, list(range(n // 2)), n)) if n >= 2 else 0.0)
    return [float(x) for x in np.asarray(out)]


def profile_of_cell(cell: dict, max_qubits: int = PROFILE_MAX_QUBITS) -> float:
    """entropy_profile_max of a phase cell's logical circuit (rebuilt from its family and parameters)."""
    if int(cell.get("n", 0)) > max_qubits:
        return math.nan
    from .phase.families import build
    fc = build(cell["family"], int(cell["n"]), **dict(cell.get("fparams") or {}))
    prof = entanglement_profile(fc.spec)
    return max(prof) if prof else 0.0


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def from_record(rec: dict, budget: float | None = None, profile: float | None = None) -> tuple:
    """(values, missing): every feature of the dictionary from a phase-study cell record. ``profile``: the cell's
    entropy_profile_max if already computed (``profile_of_cell``), else it is left missing."""
    c, ci = rec.get("cell") or {}, rec.get("circuit") or {}
    de, lam, no = rec.get("descriptors") or {}, rec.get("lambda") or {}, rec.get("noisy") or {}
    v, miss = {}, {}

    def put(name, val, why="not recorded"):
        val = _f(val)
        v[name] = val
        if math.isnan(val):
            miss[name] = why
    n = _f(c.get("n"))
    put("n", n)
    put("two_qubit_logical", de.get("two_qubit_logical", ci.get("logical_two_qubit")))
    put("non_clifford", de.get("non_clifford"))
    put("two_qubit", ci.get("two_qubit", de.get("two_qubit")))
    put("depth", ci.get("depth", de.get("depth")))
    put("pulses_1q", de.get("pulses_1q"))
    put("routing_overhead", ci.get("routing_overhead"))
    put("idle_fraction", de.get("idle_fraction"))
    put("obs_weight", de.get("obs_weight"))
    put("obs_terms", de.get("obs_terms"))
    for k in ("meyer_wallach", "entropy_half", "entropy_max_cut", "output_entropy_bits"):
        put(k, de.get(k), "state features were not computed (exact statevector skipped)")
    for k in ("lam_lc", "lam_lc_gate", "lam_lc_idle", "lam_ro"):
        put(k, lam.get(k), "no light-cone fault rate (compilation or noise model unavailable)")
    put("entropy_profile_max", profile, "not computed (pass profile_of_cell(rec['cell']))")
    put("scale", c.get("scale"))
    put("coherent_fraction", c.get("coherent_fraction"))
    put("log10_budget", math.log10(budget) if budget else math.nan, "no budget given")
    put("ideal", rec.get("ideal"))
    put("noisy_fidelity", no.get("fidelity"))
    put("noisy_purity", no.get("purity"))
    # derived
    div = (lambda a, b: a / b if (math.isfinite(a) and math.isfinite(b) and b > 0) else math.nan)
    put("two_qubit_logical_per_qubit", div(v["two_qubit_logical"], n))
    put("non_clifford_per_qubit", div(v["non_clifford"], n))
    put("routing_ratio", div(v["two_qubit"], max(v["two_qubit_logical"], 1.0) if math.isfinite(v["two_qubit_logical"])
                             else math.nan))
    put("two_qubit_density", div(v["two_qubit"], n * v["depth"] if math.isfinite(v["depth"]) else math.nan))
    put("obs_locality", div(v["obs_weight"], n))
    put("entropy_half_per_qubit", div(v["entropy_half"], math.floor(n / 2) if math.isfinite(n) else math.nan),
        "state features were not computed")
    put("idle_share", div(v["lam_lc_idle"], v["lam_lc"]) if v["lam_lc"] > 0 else (0.0 if v["lam_lc"] == 0 else
                                                                                     math.nan))
    put("lam_per_2q", div(v["lam_lc"], max(_f(lam.get("lc_2q_gates")), 1.0) if lam.get("lc_2q_gates") is not None
                          else math.nan))
    return v, miss


def from_live(compiled, noise, observable, budget: float, scale: float = 1.0, coherent_fraction: float = 0.0,
              state_max_qubits: int = PROFILE_MAX_QUBITS) -> tuple:
    """(values, missing) for a circuit being built in the app: the same features as ``from_record``, computed from
    the compiled circuit, the noise model and the observable (no ideal value, no noisy simulation)."""
    import dataclasses as _dc
    rec = dict(cell=dict(n=compiled.logical.n, scale=scale, coherent_fraction=coherent_fraction),
               circuit=dict(two_qubit=compiled.implemented.count("2q"), depth=compiled.implemented.depth(),
                            logical_two_qubit=compiled.logical.count("2q"),
                            routing_overhead=compiled.implemented.count("2q") - compiled.logical.count("2q")))
    miss_extra, prof = {}, None
    try:
        from .phase.lightcone import calculate_lambda_lc
        rec["lambda"] = _dc.asdict(calculate_lambda_lc(compiled, noise, observable))
    except Exception as e:                               # noqa: BLE001 - reported as missing
        miss_extra["lam_lc"] = "λ_LC could not be computed (%s)" % type(e).__name__
    if compiled.logical.n <= state_max_qubits and not hasattr(observable, "terms"):
        try:
            from .phase.descriptors import describe
            rec["descriptors"] = describe(compiled.logical, compiled.implemented, observable)
            prof = max(entanglement_profile(compiled.logical) or [0.0])
        except Exception as e:                           # noqa: BLE001
            miss_extra["entropy_half"] = "state features failed (%s)" % type(e).__name__
    v, miss = from_record(rec, budget, prof)
    miss.update({k: m for k, m in miss_extra.items() if k in miss})
    if compiled.logical.n > state_max_qubits:
        for k in ("meyer_wallach", "entropy_half", "entropy_max_cut", "output_entropy_bits", "entropy_profile_max",
                  "entropy_half_per_qubit"):
            miss[k] = "exact state features need 2^n amplitudes; skipped above %d qubits" % state_max_qubits
    return v, miss


# feature sets compared in the study (recommendation inputs only; never oracle features)
SETS = {
    "structural": ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "obs_weight", "lam_lc",
                   "lam_ro", "scale", "coherent_fraction", "log10_budget"),
    "normalized": ("two_qubit_logical_per_qubit", "non_clifford_per_qubit", "two_qubit_density", "routing_ratio",
                   "obs_locality", "lam_lc", "lam_per_2q", "idle_share", "lam_ro", "coherent_fraction",
                   "log10_budget"),
    "with_state": ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "obs_weight", "lam_lc", "lam_ro",
                   "scale", "coherent_fraction", "log10_budget", "entropy_half", "entropy_profile_max",
                   "meyer_wallach"),
}


def vector(values: dict, names) -> list:
    for nm in names:
        if not BY_NAME[nm].recommendable:
            raise ValueError("%s is an oracle feature: it cannot be a recommendation input" % nm)
    return [values.get(nm, math.nan) for nm in names]


def dictionary_markdown() -> str:
    L = ["# Feature dictionary (%s)" % VERSION, "",
         "Level: logical (device-independent), compiled (device/layout-specific), observable, state (exact ideal "
         "pure state: O(2^n), not scalable), noise (light-cone fault rates from the noise model or calibration), "
         "context (study settings), oracle (needs the exact simulation: never a recommendation input).", "",
         "| feature | level | definition | normalization | cost | scalable | unit | source |",
         "|---|---|---|---|---|---|---|---|"]
    for f in DICTIONARY:
        L.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (f.name, f.level, f.definition.replace("|", "\\|"),
                                                              f.normalization, f.cost, "yes" if f.scalable else "no",
                                                              f.unit, f.provenance))
    L += ["", "Missing values are NaN with a reason; a model that cannot handle a missing input abstains."]
    return "\n".join(L) + "\n"
