r"""Task representation of the budget-aware protocol-selection study (1.6.0, schema ``qemx.task/1``).

A *task* is the context of one estimation problem (theory companion, Definition 1.2):

    𝒯 = (ρ_in, C, O, D_t, 𝓘, B)

* ρ_in — the input product state, as per-qubit input tokens (``qemx.compile``; prepared once by the compiler);
* C — the logical circuit of a family instance and its compilation (device, layout, optimization level, seed);
* O — a diagonal observable over the measured classical bits, |O| ≤ 1 (so μ ∈ [−1, 1]);
* D_t — the device: a fake backend's calibration snapshot (the *supplied* model) and, separately, the execution law
  that generates the data (the *truth*; equal to the snapshot unless a mismatch scenario says otherwise);
* 𝓘 — the information classes (``INFORMATION``): what a deployed policy may use, what it must pay for, and what only
  the research evaluation may see;
* B — the total shot budget; it is not part of the task identity (labels are stored per task and budget).

The identity of a task (``TaskSpec.task_id``) covers everything that changes its numbers. Its *ancestry* groups
variants for honest splits: every task derived from one logical circuit (``base_id``: family, width and parameters)
— whatever its input, observable, layout or device — stays on one side of every train/test split. The
reproducibility record (``BuiltTask.record``: compiled facts, software versions, digests, seeds) is kept separately
from the features, so that no provenance field can leak into a model.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA = "qemx.task/1"

# Information classes (theory companion, Table 1.1). Every feature and every protocol input names its class.
INFORMATION = {
    "available": "known before any shot: circuit, observable, compiled facts, the device's calibration snapshot, the "
                 "budget; usable by a deployed policy for free (classical cost recorded)",
    "acquired": "measured at deployment (diagnostic or pilot shots, a fresh calibration): usable only with its shots "
                "charged to the same budget B",
    "simulated": "computed by classically simulating the supplied (snapshot) model of the circuit: deployable in "
                 "principle, but its cost grows exponentially with width; reported as a simulator-only augmentation",
    "oracle": "the exact ideal value, the true execution law and injected mismatch parameters: labels and research "
              "evaluation only, never a policy input",
}

FAMILIES = ("qft", "ghz", "ising", "brickwork", "grover", "bv", "hea", "qaoa")
OBSERVABLES = ("default", "z_mid", "zz_mid", "z_all")
INPUT_KINDS = ("default", "plus", "product")
LAYOUTS = ("best", "alt")
FAMILY_PARAMS = {"qft": ("m", "s"), "ghz": ("theta", "basis"), "ising": ("steps", "T", "J", "h"),
                 "brickwork": ("layers", "angle_seed"), "grover": ("k", "marked"), "bv": ("secret",),
                 "hea": ("layers", "angle_seed"), "qaoa": ("p", "angle_seed")}


@dataclass(frozen=True)
class TaskSpec:
    family: str
    n: int
    params: tuple = ()                 # sorted (name, value) pairs of the family
    inputs: str = "default"            # default | plus | product (seeded random product state)
    observable: str = "default"        # default | z_mid | zz_mid | z_all
    device: str = "fake_fez"           # calibration snapshot (base fake device)
    layout: str = "best"               # best (calibration-best chain) | alt (the second-best disjoint chain)
    opt_level: int = 1
    seed: int = 11                     # transpiler seed
    truth: str = ""                    # device-spec modifiers of the execution law ("" = the snapshot itself)

    # ---------------------------------------------------------------- identity
    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def base_id(self) -> str:
        """Ancestry: the logical circuit (family, width, parameters). All variants share it."""
        return _digest(["base", self.family, int(self.n), [list(kv) for kv in self.params]])[:16]

    @property
    def family_id(self) -> str:
        return self.family

    @property
    def task_id(self) -> str:
        return _digest(["task", SCHEMA, self.to_json()])[:20]

    def to_json(self) -> dict:
        d = asdict(self)
        d["params"] = [list(kv) for kv in self.params]
        return d

    @staticmethod
    def from_json(d: dict) -> "TaskSpec":
        d = dict(d)
        d["params"] = tuple(tuple(kv) for kv in d.get("params") or ())
        return TaskSpec(**d)

    @property
    def label(self) -> str:
        par = ",".join("%s=%s" % kv for kv in self.params)
        return "%s-%d(%s)|in=%s|O=%s|%s/%s" % (self.family, self.n, par, self.inputs, self.observable, self.device,
                                             self.layout)

    # ---------------------------------------------------------------- execution laws
    @property
    def supplied_backend(self) -> str:
        """The calibration snapshot executed exactly: the model a deployed protocol may use (PEC's model, ZNE's
        allocation, the readout prior). A device with modifiers (``fake_fez@s=2``: every reported gate error and
        relaxation rate × 2) is a scaled snapshot; the execution law follows it unless ``truth`` departs."""
        return with_modifiers(self.device, "eng=exact")

    @property
    def truth_backend(self) -> str:
        """The device part of the execution law (``mismatch.parse_truth``); crosstalk and temporal mixtures are
        applied by ``mismatch.truth_runner`` on top of it."""
        from .mismatch import parse_truth
        dev, _ = parse_truth(self.truth)
        return with_modifiers(self.device, "eng=exact", dev)


def with_modifiers(device: str, *mods: str) -> str:
    """A device specification with further modifiers (``with_modifiers("fake_fez@s=2", "eng=exact")`` →
    ``fake_fez@s=2,eng=exact``); for a plain device name the result is ``name@mods`` as before. A modifier given
    twice is refused by the device parser."""
    base, _, own = device.partition("@")
    parts = [p for p in own.split(",") if p] + [m for m in mods if m]
    return "%s@%s" % (base, ",".join(parts)) if parts else base


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def make_spec(family: str, n: int, *, inputs: str = "default", observable: str = "default", device: str = "fake_fez",
              layout: str = "best", truth: str = "", opt_level: int = 1, seed: int = 11, **params) -> TaskSpec:
    if family not in FAMILIES:
        raise ValueError("unknown family %r (choose from %s)" % (family, ", ".join(FAMILIES)))
    if observable not in OBSERVABLES:
        raise ValueError("observable must be one of %s" % ", ".join(OBSERVABLES))
    if inputs not in INPUT_KINDS:
        raise ValueError("inputs must be one of %s" % ", ".join(INPUT_KINDS))
    if layout not in LAYOUTS:
        raise ValueError("layout must be one of %s" % ", ".join(LAYOUTS))
    unknown = sorted(set(params) - set(FAMILY_PARAMS[family]))
    if unknown:
        raise ValueError("%s does not take the parameter%s %s (it takes %s; the transpiler seed is `seed`)"
                         % (family, "" if len(unknown) == 1 else "s", ", ".join(unknown),
                            ", ".join(FAMILY_PARAMS[family]) or "none"))
    return TaskSpec(family, int(n), tuple(sorted((str(k), v) for k, v in params.items())), inputs, observable, device,
                    layout, int(opt_level), int(seed), str(truth))


# ------------------------------------------------------------------ circuits
def _hea(n: int, layers: int, seed: int):
    """Hardware-efficient ansatz: L layers of RY·RZ on every qubit and a CX ladder, then a final rotation layer;
    angles uniform in [0, 2π) from (n, layers, seed)."""
    from ..circuit import CircuitSpec, Op
    from ..phase.families import _rng
    rng = _rng("hea", int(n), int(layers), int(seed))
    ops = []

    def rot():
        th, ph = rng.uniform(0, 2 * math.pi, n), rng.uniform(0, 2 * math.pi, n)
        for q in range(n):
            ops.append(Op("ry", (q,), (float(th[q]),), "gate", ""))
            ops.append(Op("rz", (q,), (float(ph[q]),), "gate", ""))
    for _ in range(int(layers)):
        rot()
        for q in range(n - 1):
            ops.append(Op("cx", (q, q + 1), (), "gate", ""))
    rot()
    return CircuitSpec(n, tuple(ops), "HEA-%d-L%d-s%d" % (n, layers, seed))


def _qaoa(n: int, p: int, seed: int):
    """QAOA on a ring (MaxCut cost Σ Z_i Z_{i+1}): p layers of RZZ(2γ_l) on every edge and RX(2β_l) mixers, from
    |+…+⟩ (prepared by the circuit's H layer); (γ_l, β_l) uniform in [0, π/2) from (n, p, seed)."""
    from ..circuit import CircuitSpec, Op
    from ..phase.families import _rng
    rng = _rng("qaoa", int(n), int(p), int(seed))
    ops = [Op("h", (q,), (), "gate", "") for q in range(n)]
    edges = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    for _ in range(int(p)):
        g, b = rng.uniform(0, math.pi / 2, 2)
        for a, c in edges:
            ops.append(Op("rzz", (a, c), (float(2 * g),), "gate", ""))
        for q in range(n):
            ops.append(Op("rx", (q,), (float(2 * b),), "gate", ""))
    return CircuitSpec(n, tuple(ops), "QAOA-%d-p%d-s%d" % (n, p, seed))


def family_circuit(spec: TaskSpec):
    """(logical circuit without input preparation, measurement basis, default observable, relabel, ideal input
    tokens of the family) for a task."""
    from .. import observables as OBS
    from ..io_semantics import split_preparation
    from ..phase import families as FAM
    p = dict(spec.p)
    n = spec.n
    # the instance seed of a random family is the parameter ``angle_seed`` (default 0). A configuration key ``seed``
    # is the transpiler seed of the task (``TaskSpec.seed``), not an instance seed (decision D24).
    aseed = int(p.pop("angle_seed", 0))
    if spec.family == "hea":
        return _hea(n, int(p.get("layers", 2)), aseed), "Z" * n, OBS.z(n // 2, n), False, ["0"] * n
    if spec.family == "qaoa":
        return _qaoa(n, int(p.get("p", 1)), aseed), "Z" * n, OBS.z_string([0, 1], n), False, ["0"] * n
    if aseed:
        p["seed"] = aseed
    fc = FAM.build(spec.family, n, **p)
    tokens, body = split_preparation(fc.spec)          # a family's own preparation becomes its input tokens
    return body, fc.basis, fc.observable, fc.relabel, list(tokens)


def input_tokens(spec: TaskSpec, family_tokens: list) -> list:
    """The task's input: the family's own (``default``), |+⟩ on every qubit (``plus``), or a seeded random product
    state of Bloch tokens (``product``). Families whose input is part of their known answer (QFT, BV, Grover, GHZ)
    accept only ``default``."""
    if spec.inputs == "default":
        return list(family_tokens)
    if spec.family in ("qft", "bv", "grover", "ghz"):
        raise ValueError("%s has a fixed input (its known answer depends on it)" % spec.family)
    if spec.inputs == "plus":
        return ["+"] * spec.n
    from ..phase.families import _rng
    rng = _rng("input", spec.base_id, spec.inputs)
    out = []
    for _ in range(spec.n):
        th, ph = float(rng.uniform(0.2, math.pi - 0.2)), float(rng.uniform(0, 2 * math.pi))
        out.append("bloch:%.6f,%.6f" % (th, ph))
    return out


def observable_of(spec: TaskSpec, default_obs, n: int):
    from .. import observables as OBS
    mid = n // 2
    if spec.observable == "default":
        return default_obs
    if spec.observable == "z_mid":
        return OBS.z(mid, n)
    if spec.observable == "zz_mid":
        return OBS.z_string([mid - 1, mid], n)
    if spec.observable == "z_all":
        return OBS.z_string(list(range(n)), n)
    raise ValueError(spec.observable)


def layout_for(spec: TaskSpec, n: int) -> tuple:
    """``best``: the calibration-best chain (``phase.families.best_chain``). ``alt``: the best chain disjoint from
    it — a different set of physical qubits of the same device (a local variant of the same logical task)."""
    from ..phase.families import best_chain, chain_cost
    from ..layout_search import device_info
    from .. import device
    base = device.base_name(spec.device)
    best = best_chain(base, n)
    if spec.layout == "best":
        return tuple(best)
    info = device_info(base)
    nb = {}
    for a, b in info.good_edges:
        if (min(a, b), max(a, b)) in info.e2:
            nb.setdefault(a, set()).add(b)
            nb.setdefault(b, set()).add(a)
    used, bad = set(best), set(info.broken_nodes)
    found = None

    def dfs(path):
        nonlocal found
        if len(path) == n:
            if path[0] > path[-1]:
                return
            key = (round(chain_cost(base, tuple(path)), 15), tuple(path))
            if found is None or key < found:
                found = key
            return
        for q in sorted(nb.get(path[-1], ())):
            if q not in path and q not in bad and q not in used:
                dfs(path + [q])
    for q0 in sorted(nb):
        if q0 not in bad and q0 not in used:
            dfs([q0])
    if found is None:
        raise ValueError("no second %d-qubit chain on %s" % (n, base))
    return found[1]


@dataclass
class BuiltTask:
    spec: TaskSpec
    compiled: object
    observable: object
    inputs: list
    ideal: float                        # oracle: exact ideal value μ(𝒯)
    layout: tuple
    record: dict = field(default_factory=dict)     # reproducibility record (never a feature)

    @property
    def n(self) -> int:
        return int(self.compiled.logical.n)


def build(spec: TaskSpec) -> BuiltTask:
    """Compile a task for its device and layout and compute its ideal value (an oracle quantity, for labels)."""
    from .. import compile as CP
    from .. import templates as T
    body, basis, default_obs, relabel, fam_tokens = family_circuit(spec)
    n = int(body.n)
    tokens = input_tokens(spec, fam_tokens)
    obs = observable_of(spec, default_obs, n)
    lay = layout_for(spec, n)
    comp = CP.build(body, tokens, basis, opts=CP.CompileOptions(device=spec.device, opt_level=spec.opt_level,
                                                               seed=spec.seed, initial_layout=lay), relabel=relabel)
    phys = tuple(comp.info["phys"])
    if set(phys) != set(lay):
        raise RuntimeError("the compiler left the requested layout %s (used %s)" % (lay, phys))
    sv = T.statevector(comp.logical.n, comp.logical.ops)
    p = np.abs(np.asarray(sv)) ** 2
    from ..engine import permute_probs
    # ideal distribution over classical bits of the logical circuit (measured with the logical measurement map)
    mm_log = {i: (n - 1 - i if relabel else i) for i in range(n)}
    p_bits = permute_probs(p, mm_log, n)
    ideal = float(obs.o @ p_bits)
    if not np.all(np.abs(obs.o) <= 1 + 1e-12):
        raise ValueError("observable values must lie in [-1, 1] (normalize O so that ||O|| <= 1)")
    rec = dict(schema=SCHEMA, task_id=spec.task_id, base_id=spec.base_id, spec=spec.to_json(), inputs=list(tokens),
               observable=dict(key=obs.key, label=obs.label), basis=basis, relabel=bool(relabel),
               layout=[int(q) for q in lay], phys=[int(q) for q in phys],
               measure_map={str(k): int(v) for k, v in comp.measure_map.items()},
               two_qubit=int(comp.implemented.count("2q")), depth=int(comp.implemented.depth()),
               logical_ops=len(comp.logical.ops), implemented_ops=len(comp.implemented.ops),
               circuit_digest=_digest([[o.name, list(o.qubits), [round(float(x), 12) for x in o.params]]
                                       for o in comp.implemented.ops])[:16])
    return BuiltTask(spec, comp, obs, tokens, ideal, lay, rec)
