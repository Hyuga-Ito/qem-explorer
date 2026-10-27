r"""What the compiler did to your circuit, step by step, with provenance.

    logical  →  routed (layout + SWAPs)  →  native (each routed gate translated on its own)  →  executed

The routed and native stages come from the same Qiskit preset pass manager the app compiles with (same target
or coupling map, optimization level, layout method and seed), run stage by stage. Every logical instruction is
labeled before compiling, so the routed circuit says which logical gate each routed gate came from; gates
without a label are the SWAPs routing inserted. For the native stage a labeled barrier follows every routed gate
before the translation stage, so each native gate can be traced to the routed gate it implements. The executed
circuit is the app's compiled circuit (after optimization, which may merge or cancel gates across these groups).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .circuit import GATES, CircuitSpec, Op, fmt_angle, to_qiskit
from .compile import QISKIT_LOCK, CompileOptions, coupling_map, lattice_target, with_io
from .resources import qiskit_serial

MAX_DIFF_OPS = 600            # the aligned diff is drawn for circuits up to this many routed operations


@dataclass
class Stages:
    logical: CircuitSpec                       # logical circuit with input preparation and basis change
    routed: CircuitSpec                        # on the used physical qubits (compact indices), SWAPs tagged "route"
    native: CircuitSpec                        # routed gates translated one by one (unoptimized)
    src_routed: list                           # per routed op: logical op index, or -1 for a routing SWAP
    src_native: list                           # per native op: routed op index
    phys: list                                 # compact index -> physical qubit
    initial_layout: dict                       # logical qubit -> physical qubit before routing
    final_layout: dict                         # logical qubit -> physical qubit where it is measured
    swap_edges: list                           # physical (a, b) pairs on which routing swapped
    decomposition: dict = field(default_factory=dict)   # "h" -> "RZ(π/2) · √X · RZ(π/2)"
    note: str = ""

    @property
    def n_swaps(self) -> int:
        return sum(1 for s in self.src_routed if s < 0)


def _labelled(logical: CircuitSpec):
    qc = to_qiskit(logical, measure=True)
    k = 0
    for i, ins in enumerate(qc.data):
        nm = ins.operation.name
        if nm in ("measure", "barrier"):
            continue
        op = ins.operation.to_mutable()
        op.label = "qx:%d" % k
        qc.data[i] = ins.replace(operation=op)
        k += 1
    return qc


def _pass_manager(opts: CompileOptions, n: int):
    from qiskit.transpiler import generate_preset_pass_manager
    kw = dict(optimization_level=int(opts.opt_level), seed_transpiler=int(opts.seed))
    if opts.device:
        from . import device
        kw["target"] = device.good_target(opts.device)
        if opts.initial_layout:
            kw["initial_layout"] = list(opts.initial_layout)
        elif opts.layout in ("sabre", "dense", "trivial"):
            kw["layout_method"] = opts.layout
    elif lattice_target(opts, n) is not None:
        kw["target"] = lattice_target(opts, n)
        if opts.initial_layout:
            kw["initial_layout"] = list(opts.initial_layout)
        elif opts.layout == "trivial":
            kw["initial_layout"] = list(range(n))
        else:
            kw["layout_method"] = opts.layout
    else:
        cm = coupling_map(opts.coupling, n)
        kw["basis_gates"] = ["rz", "sx", "x", opts.basis_2q]
        if cm is not None:
            kw["coupling_map"] = cm
            if opts.initial_layout:
                kw["initial_layout"] = list(opts.initial_layout)
            else:
                kw["layout_method"] = opts.layout
                if opts.layout == "trivial":
                    kw["initial_layout"] = list(range(n))
    return generate_preset_pass_manager(**kw)


def _label_index(op) -> int | None:
    lab = getattr(op, "label", None) or ""
    return int(lab[3:]) if lab.startswith("qx:") else None


def _fmt(o) -> str:
    lab = GATES[o.name][2] if o.name in GATES else o.name.upper()
    return lab + ("(%s)" % ", ".join(fmt_angle(p, 4).replace("*", "").replace("pi", "π") for p in o.params)
                  if o.params else "")


@qiskit_serial
def compile_stages(spec: CircuitSpec, inputs=None, basis=None, opts: CompileOptions = CompileOptions(),
                   relabel: bool = False) -> Stages:
    """The stages for the current compile options. Abstract mode has nothing to show: all stages are the logical
    circuit."""
    from qiskit.transpiler import StagedPassManager
    logical = with_io(spec, inputs, basis)
    n = logical.n
    if not opts.device and opts.mode == "abstract":
        idx = list(range(len(logical.ops)))
        lay = {q: q for q in range(n)}
        return Stages(logical, logical, logical, idx, idx, list(range(n)), lay, lay, [],
                      note="Abstract mode runs the gates as drawn: nothing is routed or translated.")
    with QISKIT_LOCK:
        pm = _pass_manager(opts, n)
        qc = _labelled(logical)
        front = StagedPassManager(stages=["init", "layout", "routing"], init=pm.init, layout=pm.layout,
                                  routing=pm.routing)
        routed = front.run(qc)
        # the logical order of labeled ops (to_qiskit may skip barriers)
        logical_ops = [o for o in logical.ops if o.kind != "barrier"]
        # initial / final layouts
        init_lay, final_lay = {}, {}
        if routed.layout is not None:
            il = routed.layout.initial_index_layout(filter_ancillas=True)
            init_lay = {q: int(il[q]) for q in range(n)}
        else:
            init_lay = {q: q for q in range(n)}
        for ins in routed.data:
            if ins.operation.name == "measure":
                c = routed.find_bit(ins.clbits[0]).index
                final_lay[c] = routed.find_bit(ins.qubits[0]).index
        if relabel:
            final_lay = {q: final_lay.get(n - 1 - q) for q in range(n)}
        # routed ops, their sources, and the physical qubits used
        rops, srcs, used = [], [], set()
        for ins in routed.data:
            nm = ins.operation.name
            if nm in ("measure", "barrier", "delay"):
                if nm == "measure":
                    used.add(routed.find_bit(ins.qubits[0]).index)
                continue
            qs = tuple(routed.find_bit(q).index for q in ins.qubits)
            used.update(qs)
            src = _label_index(ins.operation)
            rops.append((nm, qs, tuple(float(p) for p in ins.operation.params)))
            srcs.append(src if src is not None else -1)
        # compact rows follow the logical qubits (their initial physical qubit), then any extra qubits routing used
        phys = [init_lay[q] for q in range(n)] + sorted(used - set(init_lay.values()))
        cidx = {p: i for i, p in enumerate(phys)}
        # native: translate with a labeled barrier after every routed op
        from qiskit import QuantumCircuit
        from qiskit.circuit import Barrier
        body = [ins for ins in routed.data if ins.operation.name not in ("measure", "barrier", "delay")]
        big = QuantumCircuit(routed.num_qubits)
        for j, ins in enumerate(body):
            qs = [big.qubits[routed.find_bit(q).index] for q in ins.qubits]
            big.append(ins.operation, qs)
            big.append(Barrier(len(qs), label="qxb:%d" % j), qs)
        back = StagedPassManager(stages=["translation"], translation=pm.translation)
        nat = back.run(big)
    # every native gate belongs to the next labeled barrier on its qubits
    pending = {q: [] for q in range(nat.num_qubits)}
    native_ops, src_native, seq = [], [], 0
    for ins in nat.data:
        nm = ins.operation.name
        qs = tuple(nat.find_bit(q).index for q in ins.qubits)
        lab = getattr(ins.operation, "label", None) or ""
        if nm == "barrier":
            if lab.startswith("qxb:"):
                j = int(lab[4:])
                batch = []
                for q in qs:
                    batch += pending[q]
                    pending[q] = []
                for _, o in sorted(batch, key=lambda x: x[0]):
                    native_ops.append(o)
                    src_native.append(j)
            continue
        if nm in GATES and all(q in cidx for q in qs):
            pending[qs[0]].append((seq, Op(nm, tuple(cidx[q] for q in qs), tuple(float(x) for x in ins.operation.params),
                                          "gate")))
            seq += 1
    route_ops = []
    for (nm, qs, ps), s in zip(rops, srcs):
        tag = "route" if s < 0 else (logical_ops[s].tag if 0 <= s < len(logical_ops) else "")
        route_ops.append(Op(nm, tuple(cidx[q] for q in qs), ps, "gate", tag))
    m = len(phys)
    edges = tuple(sorted({tuple(sorted((cidx[a], cidx[b]))) for (nm, (a, *rest), _), s in zip(rops, srcs)
                          for b in rest}))
    routed_spec = CircuitSpec(m, tuple(route_ops), logical.name + " (routed)", edges)
    native_spec = CircuitSpec(m, tuple(native_ops), logical.name + " (native)", edges)
    swap_edges = sorted({tuple(sorted(qs)) for (nm, qs, _), s in zip(rops, srcs) if s < 0 and len(qs) == 2})
    # one decomposition per distinct logical gate
    dec = {}
    for j, ((nm, qs, ps), s) in enumerate(zip(rops, srcs)):
        key = "SWAP (routing)" if s < 0 else _fmt(Op(nm, qs, ps, "gate"))
        if key in dec:
            continue
        seq = [o for o, sj in zip(native_ops, src_native) if sj == j]
        base = {q: "ab"[i] for i, q in enumerate(sorted({cidx[q] for q in qs}))} if len(qs) > 1 else {}
        dec[key] = " · ".join(_fmt(o) + ("[%s]" % ",".join(base.get(q, "?") for q in o.qubits) if base else "")
                              for o in seq) or "(nothing: a virtual gate)"
    return Stages(logical, routed_spec, native_spec, srcs, src_native, phys, init_lay, final_lay,
                  [(a, b) for a, b in swap_edges], dec)


# ------------------------------------------------------------------ aligned drawing positions
def aligned_positions(st: Stages) -> dict:
    """Column positions that align the logical circuit above the native circuit: every routed gate becomes a
    block of columns (its native gates, laid out left to right), blocks are packed as early as their qubits
    allow, and each logical gate sits in the middle of its block. Routing SWAPs get blocks with nothing above."""
    from .viz.composer import columns
    m, n = st.native.n, st.logical.n
    groups = {}
    for i, j in enumerate(st.src_native):
        groups.setdefault(j, []).append(i)
    level = [0] * m
    llevel = [0] * n                        # logical rows too: two logical gates never share columns and rows
    native_pos, logical_pos, frames = [], [], []
    logical_ops = [i for i, o in enumerate(st.logical.ops) if o.kind != "barrier"]
    inv_init = {p: q for q, p in st.initial_layout.items()}
    for j, ro in enumerate(st.routed.ops):
        idx = groups.get(j, [])
        span_q = sorted(set(ro.qubits) | {q for i in idx for q in st.native.ops[i].qubits})
        lo, hi = span_q[0], span_q[-1]
        s = st.src_routed[j]
        lrows = ()
        if 0 <= s < len(logical_ops):
            lq = st.logical.ops[logical_ops[s]].qubits or tuple(range(n))
            lrows = range(min(lq), max(lq) + 1)
        start = max([level[q] for q in range(lo, hi + 1)] + [llevel[r] for r in lrows])
        if idx:
            sub = CircuitSpec(m, tuple(st.native.ops[i] for i in idx))
            cols, width = columns(sub)
            for (c, k) in cols:
                native_pos.append((start + c, idx[k]))
        else:
            width = 1
        width = max(width, 1)
        for q in range(lo, hi + 1):
            level[q] = start + width
        for r in lrows:
            llevel[r] = start + width
        lo_op = st.logical.ops[logical_ops[s]] if 0 <= s < len(logical_ops) else None
        label = "SWAP (routing)" if s < 0 else ("%s on %s" % (_fmt(lo_op), ", ".join("q%d" % q for q in lo_op.qubits))
                                                if lo_op is not None else ro.name)
        frames.append(dict(x0=start, x1=start + width - 1, q0=lo, q1=hi, label=label, kind="route" if s < 0 else
                           ("prep" if ro.tag in ("prep", "basis") else "")))
        if s >= 0 and s < len(logical_ops):
            logical_pos.append((start + (width - 1) / 2, logical_ops[s]))
    ncol = max(level) if level else 0
    cidx = {p: i for i, p in enumerate(st.phys)}
    mm = {c: cidx[p] for c, p in st.final_layout.items() if p in cidx}
    return dict(native=native_pos, logical=logical_pos, frames=frames, n_columns=ncol, measure_map=mm,
                qubit_labels=["p%d (q%s)" % (p, inv_init[p]) if p in inv_init else "p%d (extra)" % p
                              for p in st.phys])


def combined(st: Stages) -> dict:
    """One drawing with the logical circuit (top rows) over the native circuit (bottom rows), columns aligned:
    kwargs for ``composer.draw`` plus the stacked spec."""
    a = aligned_positions(st)
    n, m = st.logical.n, st.native.n
    k = len(st.logical.ops)
    part = {}
    for f_i, f in enumerate(a["frames"]):
        part[f_i] = "part of the routing SWAP" if f["kind"] == "route" else "part of " + f["label"]
    ops = list(st.logical.ops) + [dataclasses.replace(o, qubits=tuple(q + n for q in o.qubits),
                                                      tag=part.get(st.src_native[i], o.tag))
                                  for i, o in enumerate(st.native.ops)]
    spec = CircuitSpec(n + m, tuple(ops), st.logical.name + " (logical over physical)")
    positions = [(c, i) for c, i in a["logical"]] + [(c, k + i) for c, i in a["native"]]
    frames = [dict(f, q0=f["q0"] + n, q1=f["q1"] + n, label="⇄ SWAP" if f["kind"] == "route" else f["label"],
                   label_y=n - 0.5 if f["kind"] == "route" else None) for f in a["frames"]]
    mm = {"c%d (logical)" % q: q for q in range(n)}
    mm.update({"c%d" % c: n + r for c, r in a["measure_map"].items()})
    labels = ["q<sub>%d</sub>" % q for q in range(n)] + a["qubit_labels"]
    return dict(spec=spec, positions=positions, n_columns=a["n_columns"], groups=frames, qubit_labels=labels,
                measure_map=mm, sections=[(0, "logical"), (n, "physical: native gates framed by their logical gate "
                                                     "(amber: routing SWAP)")])
