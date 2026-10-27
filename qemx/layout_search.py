r"""Noise-aware qubit selection, in the spirit of *mapomatic* (Nation & Treinish, PRX Quantum 4, 010327 (2023)).

The compiled circuit already fits the device: its two-qubit gates sit on couplers. Its **interaction graph**
(active qubits, and an edge wherever a two-qubit gate acts) can be placed on the device in many other ways — every
subgraph embedding (VF2, from rustworkx) gives an equivalent circuit on different physical qubits, with the same
gates and SWAPs. Each placement is scored with the calibration:

.. math::  F \approx \prod_{\text{2q gates}} (1-e_{ab}) \prod_{\text{1q pulses}} (1-e_a) \prod_{\text{measured}} (1-r_a),

an estimate of the probability that no error occurs (the *expected success* score). ``1 − F`` is the ranking cost;
lower is better. The best placement is turned back into an initial layout for the transpiler, so recompiling with
it reproduces the same circuit on the chosen qubits. Idle decoherence is not in the score (mapomatic's default
neither); the simulation after recompiling includes it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import device
from .resources import qiskit_serial

MAX_MAPPINGS = 20000       # VF2 embeddings examined (the first ones found; the search stops there)


@dataclass
class DeviceInfo:
    name: str
    n: int
    edges: list                              # every coupler (a < b)
    good_edges: list                         # couplers the compiler may use
    broken_nodes: list
    broken_edges: list
    readout: dict                            # q -> mean assignment error (P(1|0) + P(0|1)) / 2
    e1: dict                                 # q -> √X error
    e2: dict                                 # (a, b) -> best two-qubit gate error (either direction)
    t1: dict
    t2: dict
    two_qubit: str
    hover: dict = field(default_factory=dict)


def _err(ip) -> float | None:
    return None if ip is None or ip.error is None else float(ip.error)


@lru_cache(maxsize=16)
@qiskit_serial
def device_info(name: str) -> DeviceInfo:
    b = device.fake_backend(name)
    tg, good = b.target, device.good_target(name)
    n = b.num_qubits
    two = next((g for g in ("cz", "ecr", "cx") if g in tg.operation_names), "cx")
    e2, edges = {}, set()
    for qa, ip in tg[two].items():
        if qa is None or len(qa) != 2:
            continue
        k = tuple(sorted(qa))
        edges.add(k)
        e = _err(ip)
        if e is not None:
            e2[k] = min(e, e2.get(k, 1.0))
    good_edges = {tuple(sorted(qa)) for qa in (good[two].keys() if two in good.operation_names else ()) if qa}
    good_nodes = {qa[0] for qa in good["measure"].keys() if qa} if "measure" in good.operation_names else set(range(n))
    props = device._props(b)
    ro, e1, t1, t2 = {}, {}, {}, {}
    for q in range(n):
        p01 = p10 = None
        if props is not None:
            try:
                d = props.qubit_property(q)
                p01, p10 = float(d["prob_meas1_prep0"][0]), float(d["prob_meas0_prep1"][0])
            except Exception:                                   # noqa: BLE001 - fall back to the target
                p01 = p10 = None
        if p01 is None:
            try:
                p01 = p10 = float(tg["measure"][(q,)].error or 0.0)
            except Exception:                                   # noqa: BLE001
                p01 = p10 = 0.0
        ro[q] = (p01 + p10) / 2
        try:
            e1[q] = float(tg["sx"][(q,)].error or 0.0)
        except Exception:                                       # noqa: BLE001
            e1[q] = 0.0
        qp = b.qubit_properties(q) if hasattr(b, "qubit_properties") else None
        t1[q] = float(getattr(qp, "t1", None) or float("nan"))
        t2[q] = float(getattr(qp, "t2", None) or float("nan"))
    broken_nodes = sorted(set(range(n)) - good_nodes)
    broken_edges = sorted(edges - good_edges)
    hover = {q: "p%d · readout %.2f%% · √X %.1e · T1 %.0f µs · T2 %.0f µs%s" % (
        q, 100 * ro[q], e1[q], 1e6 * t1[q], 1e6 * t2[q], " · <b>broken</b>" if q in broken_nodes else "")
        for q in range(n)}
    return DeviceInfo(name, n, sorted(edges), sorted(good_edges), broken_nodes, broken_edges, ro, e1, e2, t1, t2,
                      two, hover)


# ------------------------------------------------------------------ the circuit's footprint
@dataclass
class Footprint:
    k: int                                   # active (compact) qubits
    pair_counts: dict                        # (a, b) compact -> number of two-qubit gates
    pulses: np.ndarray                       # per compact qubit: √X / X pulses
    measured: np.ndarray                     # per compact qubit: 1 if measured


def footprint(compiled) -> Footprint:
    spec = compiled.implemented
    k = spec.n
    pairs, pulses = {}, np.zeros(k)
    for o in spec.ops:
        if o.kind != "gate":
            continue
        if len(o.qubits) == 2:
            key = tuple(sorted(o.qubits))
            pairs[key] = pairs.get(key, 0) + 1
        elif len(o.qubits) == 1 and o.name in ("sx", "x", "sxdg", "y", "h", "ry", "rx", "u"):
            pulses[o.qubits[0]] += 1 if o.name in ("sx", "x", "sxdg", "y") else 2
    measured = np.zeros(k)
    for c, q in compiled.measure_map.items():
        if 0 <= q < k:
            measured[q] = 1
    return Footprint(k, pairs, pulses, measured)


def _cost(fp: Footprint, di: DeviceInfo, m) -> float:
    """−log F for the placement m (compact qubit -> physical qubit)."""
    c = 0.0
    for (a, b), cnt in fp.pair_counts.items():
        e = di.e2.get(tuple(sorted((m[a], m[b]))))
        if e is None or e >= 1:
            return math.inf
        c -= cnt * math.log1p(-e)
    for a in range(fp.k):
        if fp.pulses[a]:
            c -= fp.pulses[a] * math.log1p(-min(di.e1.get(m[a], 0.0), 0.999999))
        if fp.measured[a]:
            c -= math.log1p(-min(di.readout.get(m[a], 0.0), 0.999999))
    return c


def score(compiled, name: str, phys=None) -> float:
    """Estimated success probability F of the compiled circuit on physical qubits phys (default: where it is)."""
    di = device_info(name)
    fp = footprint(compiled)
    phys = list(phys if phys is not None else compiled.info.get("phys", range(fp.k)))
    return float(math.exp(-_cost(fp, di, phys)))


@qiskit_serial
def rank_layouts(compiled, name: str, top: int = 12, max_maps: int = MAX_MAPPINGS) -> dict:
    """The best placements of the compiled circuit on the device.

    Returns dict(rows=[...], current=F of the current placement, examined=number of embeddings scored,
    complete=whether VF2 enumerated them all). Each row: rank, phys (compact i -> physical), F, error=1−F,
    initial_layout (logical q -> physical, for the transpiler), qubits (sorted set)."""
    import rustworkx as rx
    di = device_info(name)
    fp = footprint(compiled)
    cur_phys = list(compiled.info.get("phys", range(fp.k)))
    logical_init = list(compiled.info.get("initial_layout") or [])
    bad = set(di.broken_nodes)
    G = rx.PyGraph()
    G.add_nodes_from(range(di.n))
    G.add_edges_from_no_data([e for e in di.good_edges if e[0] not in bad and e[1] not in bad])
    # the connected part of the circuit goes through VF2; isolated qubits are placed greedily afterwards
    conn = sorted({q for e in fp.pair_counts for q in e})
    iso = [q for q in range(fp.k) if q not in conn]
    H = rx.PyGraph()
    H.add_nodes_from(range(len(conn)))
    pos = {q: i for i, q in enumerate(conn)}
    H.add_edges_from_no_data([(pos[a], pos[b]) for a, b in fp.pair_counts])
    single = sorted((q for q in range(di.n) if q not in bad),
                    key=lambda q: -math.log1p(-min(di.readout[q], 0.999999)) - math.log1p(-min(di.e1[q], 0.999999)))
    best, examined, complete = {}, 0, True
    if conn:
        it = rx.vf2_mapping(G, H, subgraph=True, induced=False, id_order=False)
        for mp in it:
            examined += 1
            m = [None] * fp.k
            for dev_q, h_q in mp.items():
                m[conn[h_q]] = dev_q
            taken = set(mp.keys())
            free = (q for q in single if q not in taken)
            for q in iso:
                m[q] = next(free)
            c = _cost(fp, di, m)
            key = frozenset(m)
            if math.isfinite(c) and (key not in best or c < best[key][0]):
                best[key] = (c, m)
            if examined >= max_maps:
                complete = False
                break
    else:                                      # no two-qubit gates: the k best single qubits
        m = single[:fp.k]
        best[frozenset(m)] = (_cost(fp, di, m), m)
        examined = 1
    rows = []
    cur_idx = {p: i for i, p in enumerate(cur_phys)}
    for r, (c, m) in enumerate(sorted(best.values(), key=lambda t: t[0])[:top]):
        init = [m[cur_idx[p]] for p in logical_init] if logical_init and all(p in cur_idx for p in logical_init) \
            else []
        rows.append(dict(rank=r + 1, phys=list(m), F=math.exp(-c), error=1 - math.exp(-c), initial_layout=init,
                         qubits=sorted(m)))
    cur = score(compiled, name, cur_phys) if cur_phys else float("nan")
    return dict(rows=rows, current=cur, examined=examined, complete=complete, distinct=len(best))
