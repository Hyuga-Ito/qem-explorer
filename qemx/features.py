r"""The versioned feature dictionary of the main study (``qemx.features/2``, 1.5.1).

Every feature has a definition, a normalization, a computational cost, a level, an acquisition tier and an
availability rule. Levels:

* **logical** — the circuit as written (before compilation, with its input preparation and basis change);
* **compiled** — after compilation for a device or lattice (device-, layout- and compiler-specific);
* **observable** — the measured quantity (diagonal observable or Pauli sum, with its measurement settings);
* **state** — exact properties of the ideal pure state: storage 16·2^n bytes and O(2^(3n/2)) time for the
  entanglement entropies (see ``qemx.phase.descriptors``), so they are **not scalable** and computed only on request
  and only up to a qubit cutoff;
* **noise** — light-cone fault rates under the noise model of the experiment (``qemx.phase.lightcone``): the selected
  device model with its scale and coherent fraction, or the noise injector. Model-dependent descriptors, not
  sufficient statistics for mitigation performance;
* **context** — the study settings (noise scale, coherent fraction) and the shot budget;
* **oracle** — quantities that need the exact ideal or noisy simulation (ideal value, noisy-state fidelity, purity).
  Recorded for analysis but **never allowed as recommendation inputs**.

Acquisition tiers (review §7.4: feature acquisition is part of the decision): 1 = circuit, observable and budget
(no noise model, no simulation); 2 = the calibrated noise model (λ features, scale, coherent fraction); 3 = exact
state (exponential); 0 = oracle (never an input).

Extractors are independent (1.5.1): :func:`structural_features`, :func:`observable_features`,
:func:`noise_features`, :func:`state_features`; :func:`compute` runs only those needed for the requested names.
Statevector availability never controls a count. Missing values are NaN with a reason (``missing``), never zero.

Two-qubit gate count is not entanglement: the entanglement features are von Neumann entropies of reduced states of the
*pure* ideal state (a valid entanglement measure for pure states) and the Meyer–Wallach measure. The classical output
entropy is *not* an entanglement measure and is labeled so.

``from_record`` reads a phase-study cell record (no simulation: figures and models can be regenerated from stored
data); ``ALIASES`` maps the names of schema /1 tables and records. ``vector`` returns the inputs allowed at
recommendation time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

VERSION = "qemx.features/2"
PREVIOUS = ("qemx.features/1",)
LEVELS = ("logical", "compiled", "observable", "state", "noise", "context", "oracle")
TIERS = {1: "circuit, observable and budget (no noise model, no simulation)",
         2: "calibrated noise model (light-cone fault rates, device scale, coherent fraction)",
         3: "exact ideal state (exponential in n; small systems only)",
         0: "oracle (needs the exact ideal or noisy simulation; never a recommendation input)"}


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
    tier: int = 1

    @property
    def recommendable(self) -> bool:
        return self.level != "oracle"


D = FeatureDef
_LAM_NOTE = ("A model-dependent descriptor under an independent-location surrogate (Pauli-twirled fault "
             "probabilities: coherent orientation and accumulation are discarded; structural light cones can exceed "
             "the exact Heisenberg support; quasi-static noise enters through its per-segment marginal), not a "
             "sufficient statistic for mitigation performance")
_COST_STATE = "storage 16·2^n B; time O(gates·2^n)"
DICTIONARY = (
    D("n", "logical", "number of qubits of the logical circuit (width)", unit="qubits", provenance="cell.n"),
    D("two_qubit_logical", "logical", "two-qubit gates of the logical circuit as written (input preparation and "
      "basis change add none)", provenance="descriptors"),
    D("two_qubit_logical_per_qubit", "logical", "two_qubit_logical / n", normalization="per qubit"),
    D("non_clifford", "logical", "non-Clifford gates of the logical circuit as written (T, T† and rotations by "
      "non-Clifford angles): logical content before decomposition into native gates — not the number of slots a "
      "training generator replaces (see non_clifford_slots)", provenance="descriptors"),
    D("non_clifford_per_qubit", "logical", "non_clifford / n", normalization="per qubit"),
    D("two_qubit", "compiled", "two-qubit gates of the compiled circuit (native gates after layout, routing and "
      "optimization)", provenance="circuit"),
    D("depth", "compiled", "ASAP layers of the compiled circuit (qemx.circuit.layers): every gate, virtual RZ "
      "included, takes one layer on its qubits whatever its duration; noise operations and barriers take none",
      provenance="circuit"),
    D("pulses_1q", "compiled", "non-virtual one-qubit gates of the compiled circuit (√X, X, …; RZ is virtual)",
      provenance="descriptors"),
    D("non_clifford_slots", "compiled", "non-Clifford gates of the compiled circuit (native basis: RZ angles that are "
      "not multiples of π/2): exactly the slots the CDR/FR/NIL training generators replace by Clifford gates "
      "(qemx.transforms.non_clifford_slots)", provenance="descriptors.non_clifford_isa"),
    D("two_qubit_compile_overhead", "compiled", "two_qubit − two_qubit_logical (signed). It mixes the decomposition "
      "of logical gates into native two-qubit gates (a controlled phase becomes two CZ), SWAPs inserted by routing "
      "and gates removed by optimization; it is not a routing count (schema /1: routing_overhead)",
      provenance="circuit"),
    D("two_qubit_compile_ratio", "compiled", "two_qubit / max(two_qubit_logical, 1): the same mixture of "
      "decomposition, routing and optimization as a ratio (schema /1: routing_ratio)", normalization="ratio"),
    D("two_qubit_per_qubit_layer", "compiled", "two_qubit / (n · depth): compiled two-qubit gates per (qubit, layer) "
      "slot, a rate with depth as defined above. A two-qubit gate occupies two slots of one layer, so the share of "
      "slots occupied by two-qubit gates is twice this value (schema /1: two_qubit_density)",
      normalization="per qubit-layer"),
    D("idle_fraction", "compiled", "share of (qubit, layer) slots after a qubit's first gate in which it has no gate "
      "(layers as in depth: a scheduling proxy, not durations)", normalization="fraction",
      provenance="descriptors"),
    D("obs_weight", "observable", "largest Pauli weight among the observable's non-identity terms: the Z strings of a "
      "diagonal observable's Walsh–Hadamard decomposition, or the Pauli strings of a Pauli sum",
      cost="O(n·2^n) diagonal (2^n stored values); O(terms·n) Pauli sum", provenance="descriptors"),
    D("obs_terms", "observable", "number of non-identity terms (|coefficient| > 1e-12)",
      cost="O(n·2^n) diagonal; O(terms·n) Pauli sum", provenance="descriptors"),
    D("obs_weight_mean", "observable", "mean weight of the non-identity terms (unweighted)",
      cost="O(n·2^n) diagonal; O(terms·n) Pauli sum", provenance="qemx.phase.descriptors.observable_descriptors"),
    D("obs_support", "observable", "number of qubits (classical bits for a diagonal observable) that any term acts on",
      unit="qubits", cost="O(n·2^n) diagonal; O(terms·n) Pauli sum",
      provenance="qemx.phase.descriptors.observable_descriptors"),
    D("obs_settings", "observable", "measurement settings per estimate: 1 for a diagonal observable (one basis); for "
      "a Pauli sum the qubit-wise-commuting groups of the app's greedy grouping (qemx.hamiltonian.qwc_groups), each "
      "a separately executed circuit", cost="O(terms²·n)",
      provenance="qemx.phase.descriptors.observable_descriptors"),
    D("obs_locality", "observable", "obs_weight / n", normalization="per qubit"),
    D("meyer_wallach", "state", "Meyer–Wallach Q = 2(1 − mean_k Tr ρ_k²) of the ideal pure state",
      cost=_COST_STATE + " + O(n·2^n)", scalable=False, provenance="descriptors", tier=3),
    D("entropy_half", "state", "von Neumann entropy (bits) of the first ⌊n/2⌋ qubits of the ideal pure state "
      "(Schmidt spectrum of the smaller side)", cost=_COST_STATE + " + O(2^(3n/2))", scalable=False, unit="bits",
      provenance="descriptors", tier=3),
    D("entropy_half_per_qubit", "state", "entropy_half / ⌊n/2⌋ (fraction of the maximum)", normalization="per qubit",
      cost=_COST_STATE + " + O(2^(3n/2))", scalable=False, tier=3),
    D("entropy_max_cut", "state", "largest entropy over the n − 1 contiguous bipartitions (first k qubits | rest) of "
      "the logical qubit order, each from its smaller side", cost=_COST_STATE + " + O(2^(3n/2))", scalable=False,
      unit="bits", provenance="descriptors", tier=3),
    D("entropy_profile_max", "state", "largest half-cut entropy (bits) of the ideal pure state after any layer of "
      "the logical circuit: entanglement the final state may conceal (the known-answer QFT ends in a basis state)",
      cost="storage 16·2^n B; time O(gates·2^n + layers·2^(3n/2))", scalable=False, unit="bits",
      provenance="qemx.phase.descriptors.entanglement_profile", tier=3),
    D("output_entropy_bits", "state", "Shannon entropy of the ideal computational-basis output distribution — a "
      "classical quantity, NOT an entanglement measure", cost=_COST_STATE, scalable=False, unit="bits",
      provenance="descriptors", tier=3),
    D("lam_lc", "noise", "light-cone fault rate λ_LC: summed Pauli-twirled fault probability of the fault locations "
      "(gate errors, idle windows) in the observable's backward structural light cone, under the experiment's noise "
      "model (device mode: the selected device model with its scale and coherent fraction; injector mode: the "
      "injector). Pauli sums: the |coefficient|-weighted mean over terms of each term's λ_LC (its qubits mapped "
      "through the measurement map; the settings' basis rotations are not included). " + _LAM_NOTE,
      cost="O(locations)", provenance="lambda", tier=2),
    D("lam_lc_gate", "noise", "λ_LC from gate errors (same rule)", cost="O(locations)", provenance="lambda", tier=2),
    D("lam_lc_idle", "noise", "λ_LC from idle (relaxation/dephasing) windows (same rule)", cost="O(locations)",
      provenance="lambda", tier=2),
    D("lam_ro", "noise", "readout fault rate: Σ over the measured support of the mean flip probability "
      "(P(1|0) + P(0|1))/2; Pauli sums: |coefficient|-weighted mean over terms", provenance="lambda", tier=2),
    D("idle_share", "noise", "lam_lc_idle / lam_lc", normalization="fraction", tier=2),
    D("lam_per_2q", "noise", "lam_lc / max(light-cone two-qubit gates, 1)", normalization="per two-qubit gate",
      tier=2),
    D("scale", "context", "noise scale s of the device model (device-specification modifier s; 1 = as calibrated). "
      "Not defined for the noise injector (NaN)", provenance="cell / backend specification", tier=2),
    D("coherent_fraction", "context", "coherent fraction χ of the homogenized device model (modifier chi; engine "
      "E1); 0 for the device's own model. Not defined for the noise injector (NaN)",
      provenance="cell / backend specification", tier=2),
    D("log10_budget", "context", "log10 of the total shot budget B of the estimate", unit="log10 shots", tier=1),
    D("ideal", "oracle", "exact ideal value of the observable", cost="O(2^n)", scalable=False, tier=0),
    D("noisy_fidelity", "oracle", "fidelity of the exact noisy state to the ideal state", cost="O(4^n)",
      scalable=False, tier=0),
    D("noisy_purity", "oracle", "purity of the exact noisy state", cost="O(4^n)", scalable=False, tier=0),
)
BY_NAME = {f.name: f for f in DICTIONARY}

# schema /1 → /2: old name -> (new name, values identical?, note). Tables and records of /1 load through it.
ALIASES = {
    "routing_overhead": ("two_qubit_compile_overhead", False,
                         "renamed: it is not a routing count. /1 records stored max(compiled − logical, 0) and the /1 "
                         "live path the signed difference; /2 is the signed difference everywhere (equal when ≥ 0)"),
    "routing_ratio": ("two_qubit_compile_ratio", True, "renamed (same formula): not a routing ratio"),
    "two_qubit_density": ("two_qubit_per_qubit_layer", True,
                          "renamed (same formula): a rate per (qubit, layer) slot, not the share of occupied slots"),
}
NEW_TO_OLD = {new: (old, same, note) for old, (new, same, note) in ALIASES.items()}
CHANGES = (
    "/2 (1.5.1): routing_overhead → two_qubit_compile_overhead (signed; decomposition + routing − optimization), "
    "routing_ratio → two_qubit_compile_ratio, two_qubit_density → two_qubit_per_qubit_layer (a rate; the occupied-slot "
    "share is twice it). New: non_clifford_slots (compiled slots replaced by training), obs_weight_mean, obs_support, "
    "obs_settings (setting-aware Pauli sums). Pauli-sum observables get observable and λ features (|c|-weighted "
    "rule) instead of NaN. Noise features in device mode come from the selected device model with its scale and "
    "coherent fraction (1.5.0's live path used the inactive injector and scale 1). Structural and observable "
    "features no longer depend on the exact-state cutoff. State features use the smaller side of every cut and one "
    "incrementally advanced statevector (values unchanged to 1e-10).",
)

PROFILE_MAX_QUBITS = 14                 # default cutoff of the exact state features in the app (live path)
_STATE_NAMES = ("meyer_wallach", "entropy_half", "entropy_max_cut", "output_entropy_bits")
_STATE_ALL = _STATE_NAMES + ("entropy_profile_max", "entropy_half_per_qubit")
_STRUCT = ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "pulses_1q", "non_clifford_slots",
           "two_qubit_compile_overhead", "idle_fraction")
_OBS = ("obs_weight", "obs_terms", "obs_weight_mean", "obs_support", "obs_settings")
_NOISE = ("lam_lc", "lam_lc_gate", "lam_lc_idle", "lam_ro")
DERIVED = {"two_qubit_logical_per_qubit": ("two_qubit_logical", "n"), "non_clifford_per_qubit": ("non_clifford", "n"),
           "two_qubit_compile_ratio": ("two_qubit", "two_qubit_logical"),
           "two_qubit_per_qubit_layer": ("two_qubit", "n", "depth"), "obs_locality": ("obs_weight", "n"),
           "entropy_half_per_qubit": ("entropy_half", "n"), "idle_share": ("lam_lc_idle", "lam_lc"),
           "lam_per_2q": ("lam_lc",)}


def entanglement_profile(spec) -> tuple:
    """Half-cut von Neumann entropy (bits) of the ideal pure state after each layer of the circuit (gates only;
    layers as in ``circuit.layer_order``). Exact; one statevector advanced layer by layer (see
    ``qemx.phase.descriptors``)."""
    from .phase.descriptors import entanglement_profile as _ep
    return _ep(spec)


def profile_of_cell(cell: dict, max_qubits: int = PROFILE_MAX_QUBITS) -> float:
    """entropy_profile_max of a phase cell's logical circuit (rebuilt from its family and parameters)."""
    if int(cell.get("n", 0)) > max_qubits:
        return math.nan
    from .phase.families import build
    fc = build(cell["family"], int(cell["n"]), **dict(cell.get("fparams") or {}))
    prof = entanglement_profile(fc.spec)
    return max(prof) if prof else 0.0


def observable_of_cell(cell: dict):
    """The observable a phase cell measures (rebuilt from its family and parameters; no simulation)."""
    from .phase.families import build
    return build(cell["family"], int(cell["n"]), **dict(cell.get("fparams") or {})).observable


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _div(a, b):
    return a / b if (math.isfinite(a) and math.isfinite(b) and b > 0) else math.nan


def _derive(v: dict, miss: dict, put) -> None:
    """Normalized features from the base values in ``v`` (NaN propagates with the reason of its input)."""
    def why(*names):
        for nm in names:
            if nm in miss:
                return "needs %s: %s" % (nm, miss[nm])
        return "not computed"
    g = lambda k: v.get(k, math.nan)                     # noqa: E731
    n = g("n")
    if "two_qubit_logical" in v:
        put("two_qubit_logical_per_qubit", _div(g("two_qubit_logical"), n), why("two_qubit_logical", "n"))
    if "non_clifford" in v:
        put("non_clifford_per_qubit", _div(g("non_clifford"), n), why("non_clifford", "n"))
    if "two_qubit" in v and "two_qubit_logical" in v:
        t2l = g("two_qubit_logical")
        put("two_qubit_compile_ratio", _div(g("two_qubit"), max(t2l, 1.0) if math.isfinite(t2l) else math.nan),
            why("two_qubit", "two_qubit_logical"))
    if "two_qubit" in v and "depth" in v:
        put("two_qubit_per_qubit_layer", _div(g("two_qubit"), n * g("depth") if math.isfinite(g("depth"))
                                              else math.nan), why("two_qubit", "depth", "n"))
    if "obs_weight" in v:
        put("obs_locality", _div(g("obs_weight"), n), why("obs_weight", "n"))
    if "entropy_half" in v:
        put("entropy_half_per_qubit", _div(g("entropy_half"), math.floor(n / 2) if math.isfinite(n) else math.nan),
            why("entropy_half", "n"))
    if "lam_lc" in v:
        lam = g("lam_lc")
        put("idle_share", _div(g("lam_lc_idle"), lam) if lam > 0 else (0.0 if lam == 0 else math.nan),
            why("lam_lc", "lam_lc_idle"))
        put("lam_per_2q", _div(lam, max(g("_lc_2q_gates"), 1.0) if math.isfinite(g("_lc_2q_gates")) else math.nan),
            why("lam_lc", "_lc_2q_gates"))


# ------------------------------------------------------------------ records (no simulation)
def from_record(rec: dict, budget: float | None = None, profile: float | None = None, observable=None) -> tuple:
    """(values, missing): every feature of the dictionary from a phase-study cell record. ``profile``: the cell's
    entropy_profile_max if already computed (``profile_of_cell``), else it is left missing. ``observable``: the
    cell's observable (``observable_of_cell``) for the observable features records before /2 did not store."""
    c, ci = rec.get("cell") or {}, rec.get("circuit") or {}
    de, lam, no = rec.get("descriptors") or {}, rec.get("lambda") or {}, rec.get("noisy") or {}
    v, miss = {}, {}

    def put(name, val, why="not recorded"):
        val = _f(val)
        v[name] = val
        if math.isnan(val):
            miss[name] = why
        else:
            miss.pop(name, None)
    put("n", c.get("n"))
    put("two_qubit_logical", de.get("two_qubit_logical", ci.get("logical_two_qubit")))
    put("non_clifford", de.get("non_clifford"))
    put("non_clifford_slots", de.get("non_clifford_slots", de.get("non_clifford_isa")))
    put("two_qubit", ci.get("two_qubit", de.get("two_qubit")))
    put("depth", ci.get("depth", de.get("depth")))
    put("pulses_1q", de.get("pulses_1q"))
    t2, t2l = _f(ci.get("two_qubit", de.get("two_qubit"))), _f(ci.get("logical_two_qubit",
                                                                       de.get("two_qubit_logical")))
    put("two_qubit_compile_overhead", t2 - t2l)
    put("idle_fraction", de.get("idle_fraction"))
    put("obs_weight", de.get("obs_weight"))
    put("obs_terms", de.get("obs_terms"))
    od = None
    if observable is not None and any(k not in de for k in ("obs_weight_mean", "obs_support", "obs_settings")):
        from .phase.descriptors import observable_descriptors
        od = observable_descriptors(observable)
    for k in ("obs_weight_mean", "obs_support", "obs_settings"):
        put(k, de.get(k, od[k] if od is not None else None),
            "not recorded (records before qemx.features/2; pass observable=observable_of_cell(cell))")
    for k in _STATE_NAMES:
        put(k, de.get(k), "state features were not computed (exact statevector skipped)")
    for k in _NOISE:
        put(k, lam.get(k), "no light-cone fault rate (compilation or noise model unavailable)")
    put("_lc_2q_gates", lam.get("lc_2q_gates"), "not recorded")
    put("entropy_profile_max", profile, "not computed (pass profile_of_cell(rec['cell']))")
    put("scale", c.get("scale"))
    put("coherent_fraction", c.get("coherent_fraction"))
    put("log10_budget", math.log10(budget) if budget else math.nan, "no budget given")
    put("ideal", rec.get("ideal"))
    put("noisy_fidelity", no.get("fidelity"))
    put("noisy_purity", no.get("purity"))
    _derive(v, miss, put)
    v.pop("_lc_2q_gates", None)
    miss.pop("_lc_2q_gates", None)
    return v, miss


# ------------------------------------------------------------------ live extractors
def structural_features(compiled) -> dict:
    """Logical and compiled counts (O(gates)); never NaN."""
    from .phase.descriptors import structural_descriptors
    s = structural_descriptors(compiled.logical, compiled.implemented)
    return dict(n=float(compiled.logical.n), two_qubit_logical=float(s["two_qubit_logical"]),
                non_clifford=float(s["non_clifford"]), two_qubit=float(s["two_qubit"]), depth=float(s["depth"]),
                pulses_1q=float(s["pulses_1q"]), non_clifford_slots=float(s["non_clifford_isa"]),
                two_qubit_compile_overhead=float(s["two_qubit"] - s["two_qubit_logical"]),
                idle_fraction=float(s["idle_fraction"]))


def observable_features(observable) -> dict:
    """Weights, terms, support and measurement settings of a diagonal observable or a Pauli sum."""
    from .phase.descriptors import observable_descriptors
    o = observable_descriptors(observable)
    return {k: float(o[k]) for k in _OBS}


def term_supports(compiled, observable) -> list:
    """[(|coefficient|, physical (compact) qubits)] of each non-identity term: Z strings over classical bits mapped
    with the measurement map, or Pauli terms over logical qubits mapped bit → qubit (``hamiltonian.clbit_of``)."""
    from .phase.descriptors import observable_terms
    mm = compiled.measure_map
    out = []
    if hasattr(observable, "terms"):
        from .hamiltonian import clbit_of
        for c, qs, _ in observable_terms(observable):
            out.append((abs(c), tuple(sorted({int(mm[clbit_of(compiled, q)]) for q in qs}))))
    else:
        for c, bits, _ in observable_terms(observable):
            out.append((abs(c), tuple(sorted({int(mm.get(b, b)) for b in bits}))))
    return out


def noise_features(compiled, noise_model, observable) -> tuple:
    """(values, missing): λ features under ``noise_model`` (a NoiseSpec, a device specification such as
    ``fake_fez@s=0.5,eng=exact`` or a DeviceModel). Diagonal observable: the union of its Z strings' supports (as the
    cell runner); Pauli sum: the |coefficient|-weighted mean over terms of each term's light-cone rates."""
    from .phase.lightcone import calculate_lambda_lc
    v, miss = {}, {}
    try:
        layout = compiled.info.get("phys")
        if hasattr(observable, "terms"):
            terms = term_supports(compiled, observable)
            if not terms:
                raise ValueError("the Pauli sum has no non-identity term")
            wsum = sum(w for w, _ in terms)
            acc = dict(lam_lc=0.0, lam_lc_gate=0.0, lam_lc_idle=0.0, lam_ro=0.0, _lc_2q_gates=0.0)
            for w, sup in terms:
                r = calculate_lambda_lc(compiled, noise_model, None, layout=layout, support=sup)
                for k, x in (("lam_lc", r.lam_lc), ("lam_lc_gate", r.lam_lc_gate), ("lam_lc_idle", r.lam_lc_idle),
                             ("lam_ro", r.lam_ro), ("_lc_2q_gates", r.lc_2q_gates)):
                    acc[k] += w * float(x) / wsum
            v.update(acc)
        else:
            sup = sorted(set().union(*[set(s) for _, s in term_supports(compiled, observable)]) or set())
            r = calculate_lambda_lc(compiled, noise_model, None, layout=layout, support=tuple(sup))
            v.update(lam_lc=r.lam_lc, lam_lc_gate=r.lam_lc_gate, lam_lc_idle=r.lam_lc_idle, lam_ro=r.lam_ro,
                     _lc_2q_gates=float(r.lc_2q_gates))
    except Exception as e:                               # noqa: BLE001 - reported as missing
        for k in _NOISE + ("_lc_2q_gates",):
            v[k] = math.nan
            miss[k] = "λ_LC could not be computed (%s: %s)" % (type(e).__name__, str(e)[:120])
    return v, miss


def state_features(logical, names, max_qubits: int = PROFILE_MAX_QUBITS) -> tuple:
    """(values, missing) of the requested exact state features of the logical circuit (NaN with a reason above the
    qubit cutoff or when the estimated working memory exceeds the available memory)."""
    from .phase import descriptors as DS
    from .resources import RunTimeout, available_bytes, human
    want = [k for k in _STATE_ALL if k in set(names)]
    if "entropy_half_per_qubit" in want and "entropy_half" not in want:
        want.append("entropy_half")
    v, miss = {}, {}
    if not want:
        return v, miss
    n = logical.n
    reason = None
    if n > max_qubits:
        reason = "exact state features need 2^n amplitudes; skipped above %d qubits" % max_qubits
    else:
        need, avail = DS.state_bytes(n), available_bytes()
        if avail and need > avail:
            reason = "exact state features need about %s of memory; %s available" % (human(need), human(avail))
    if reason:
        for k in want:
            v[k], miss[k] = math.nan, reason
        return v, miss
    try:
        st = DS.state_descriptors(logical, [k for k in want if k in _STATE_NAMES])
        v.update({k: float(x) for k, x in st.items()})
        if "entropy_profile_max" in want:
            prof = DS.entanglement_profile(logical)
            v["entropy_profile_max"] = float(max(prof)) if prof else 0.0
    except (RunTimeout, KeyboardInterrupt):
        raise                                            # the session's time limit stops the whole computation
    except Exception as e:                               # noqa: BLE001 - MemoryError included: reported as missing
        for k in want:
            v[k], miss[k] = math.nan, "state features failed (%s)" % type(e).__name__
    return v, miss


def _needed(names) -> set:
    out = set()
    stack = list(names)
    while stack:
        k = stack.pop()
        if k in out:
            continue
        out.add(k)
        stack.extend(DERIVED.get(k, ()))
    return out


def compute(compiled, observable, noise_model, budget: float | None = None, *, scale=None,
            coherent_fraction=None, names=None, state_max_qubits: int = PROFILE_MAX_QUBITS,
            context_reason: str = "") -> tuple:
    """(values, missing) for a circuit being built in the app: the features ``names`` (default: every
    recommendable feature), each from its own extractor, computing only what they need. ``noise_model``: what the
    experiment executes with (device specification in device mode, NoiseSpec otherwise). ``scale`` /
    ``coherent_fraction``: the device context (None → NaN with ``context_reason``)."""
    names = [f.name for f in DICTIONARY if f.recommendable] if names is None else list(names)
    for nm in names:
        if nm not in BY_NAME:
            raise KeyError("unknown feature %r (schema %s)" % (nm, VERSION))
    need = _needed(names)
    v, miss = {}, {}

    def put(name, val, why="not computed"):
        val = _f(val)
        v[name] = val
        if math.isnan(val):
            miss[name] = why
        else:
            miss.pop(name, None)
    if need & (set(_STRUCT) | {"n"}):
        for k, x in structural_features(compiled).items():
            put(k, x)
    if need & set(_OBS):
        for k, x in observable_features(observable).items():
            put(k, x)
    if need & (set(_NOISE) | {"idle_share", "lam_per_2q"}):
        nv, nm_ = noise_features(compiled, noise_model, observable)
        for k, x in nv.items():
            put(k, x, nm_.get(k, "not computed"))
    if need & set(_STATE_ALL):
        sv, sm = state_features(compiled.logical, need & set(_STATE_ALL), state_max_qubits)
        for k, x in sv.items():
            put(k, x, sm.get(k, "not computed"))
    if "scale" in need:
        put("scale", scale if scale is not None else math.nan, context_reason or "no device scale")
    if "coherent_fraction" in need:
        put("coherent_fraction", coherent_fraction if coherent_fraction is not None else math.nan,
            context_reason or "no coherent fraction")
    if "log10_budget" in need:
        put("log10_budget", math.log10(budget) if budget else math.nan, "no budget given")
    _derive(v, miss, put)
    v.pop("_lc_2q_gates", None)
    miss.pop("_lc_2q_gates", None)
    keep = set(names)
    return {k: x for k, x in v.items() if k in keep}, {k: m for k, m in miss.items() if k in keep}


def device_context(noise_model) -> tuple:
    """(scale, coherent_fraction, reason) of a noise model: the modifiers of a device specification, or
    (None, None, reason) for a noise injector."""
    from . import device
    if isinstance(noise_model, str) and device.is_device(noise_model):
        _, mods = device.parse_device(noise_model)
        return float(mods["s"]), float(mods["chi"]), ""
    name = getattr(noise_model, "name", None)
    if isinstance(name, str) and device.is_device(name):
        _, mods = device.parse_device(name)
        return float(mods["s"]), float(mods["chi"]), ""
    return None, None, "not defined for the noise injector (the study's scale and coherent fraction modify a device " \
                       "model)"


def from_live(compiled, noise_model, observable, budget: float, scale=None, coherent_fraction=None,
              state_max_qubits: int = PROFILE_MAX_QUBITS, names=None) -> tuple:
    """(values, missing) of a live circuit under ``noise_model`` — pass what the experiment executes with: the device
    specification (e.g. ``ss.backend``) in device mode, the injector's NoiseSpec otherwise. ``scale`` and
    ``coherent_fraction`` default to the device specification's modifiers (NaN for an injector). 1.5.1: the defaults
    were 1.0 / 0.0 and the app passed the injector even in device mode (finding F02)."""
    s, chi, why = device_context(noise_model)
    return compute(compiled, observable, noise_model, budget, scale=s if scale is None else scale,
                   coherent_fraction=chi if coherent_fraction is None else coherent_fraction, names=names,
                   state_max_qubits=state_max_qubits, context_reason=why)


# feature sets compared in the study (recommendation inputs only; never oracle features)
SETS = {
    # tier 1 only (circuit, observable, budget): what a recommendation can use without any noise model
    "circuit_observable": ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "obs_weight",
                           "log10_budget"),
    "structural": ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "obs_weight", "lam_lc",
                   "lam_ro", "scale", "coherent_fraction", "log10_budget"),
    "normalized": ("two_qubit_logical_per_qubit", "non_clifford_per_qubit", "two_qubit_per_qubit_layer",
                   "two_qubit_compile_ratio", "obs_locality", "lam_lc", "lam_per_2q", "idle_share", "lam_ro",
                   "coherent_fraction", "log10_budget"),
    "with_state": ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "obs_weight", "lam_lc", "lam_ro",
                   "scale", "coherent_fraction", "log10_budget", "entropy_half", "entropy_profile_max",
                   "meyer_wallach"),
}


def set_tiers(names) -> dict:
    """{tier: [features]} of a feature set (what acquiring its inputs costs)."""
    out = {}
    for nm in names:
        out.setdefault(BY_NAME[nm].tier, []).append(nm)
    return out


def vector(values: dict, names) -> list:
    for nm in names:
        if not BY_NAME[nm].recommendable:
            raise ValueError("%s is an oracle feature: it cannot be a recommendation input" % nm)
    return [values.get(nm, math.nan) for nm in names]


def resolve_columns(columns, names, schema: str = VERSION, prefix: str = "x_") -> tuple:
    """({feature: column}, notes, problems) for reading ``names`` from a table written under feature ``schema``.
    A /1 table is read through ALIASES when the values are identical; a renamed feature whose values changed is a
    problem (the model would mix two definitions), as is a missing column."""
    cols = set(columns)
    out, notes, problems = {}, [], []
    if schema not in (VERSION,) + PREVIOUS:
        problems.append("the study tables use feature schema %r; this code reads %s and %s"
                        % (schema, VERSION, ", ".join(PREVIOUS)))
        return out, notes, problems
    for nm in names:
        if prefix + nm in cols:
            out[nm] = prefix + nm
            continue
        old = NEW_TO_OLD.get(nm)
        if old and prefix + old[0] in cols:
            if old[1]:
                out[nm] = prefix + old[0]
                notes.append("%s read from the %s column %s (%s)" % (nm, schema, prefix + old[0], old[2]))
            else:
                problems.append("%s: the %s column %s has a different definition (%s)" % (nm, schema,
                                                                                         prefix + old[0], old[2]))
            continue
        problems.append("feature %s is not in the study tables" % nm)
    return out, notes, problems


def dictionary_markdown() -> str:
    L = ["# Feature dictionary (%s)" % VERSION, "",
         "Generated from `qemx/features.py` (`python -m qemx.study features --out docs/feature_dictionary.md`).", "",
         "Level: logical (the circuit as written), compiled (device/layout/compiler-specific), observable, state "
         "(exact ideal pure state: exponential, not scalable), noise (light-cone fault rates under the experiment's "
         "noise model — model-dependent descriptors, not sufficient statistics), context (study settings and budget), "
         "oracle (needs the exact simulation: never a recommendation input).", "",
         "Acquisition tier: " + "; ".join("%d = %s" % (k, TIERS[k]) for k in (1, 2, 3, 0)) + ".", "",
         "| feature | level | tier | definition | normalization | cost | scalable | unit | source |",
         "|---|---|---|---|---|---|---|---|---|"]
    for f in DICTIONARY:
        L.append("| `%s` | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            f.name, f.level, f.tier, f.definition.replace("|", "\\|"), f.normalization, f.cost,
            "yes" if f.scalable else "no", f.unit, f.provenance))
    L += ["", "Missing values are NaN with a reason; a model that cannot handle a missing input abstains.", "",
          "## Names of earlier schemas", "", "| schema /1 name | /2 name | values identical | note |",
          "|---|---|---|---|"]
    for old, (new, same, note) in ALIASES.items():
        L.append("| `%s` | `%s` | %s | %s |" % (old, new, "yes" if same else "no", note))
    L += ["", "## Changes", ""] + ["* " + c for c in CHANGES]
    L += ["", "## Feature sets of the study", ""]
    for nm, cols in SETS.items():
        L.append("* **%s**: %s (tiers %s)" % (nm, ", ".join("`%s`" % c for c in cols),
                                              ", ".join(str(t) for t in sorted(set_tiers(cols)))))
    return "\n".join(L) + "\n"
