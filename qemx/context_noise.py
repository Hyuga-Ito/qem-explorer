r"""Context-dependent noise on IBM heavy-hex devices: what an as-executed circuit exposes, and estimators that use it.

Average gate fidelities describe each gate *alone*. On a superconducting heavy-hex device a gate's error also depends
on its context (Rudinger et al., PRX 9, 021045 (2019); Sarovar et al., Quantum 4, 321 (2020)):

* **spectators** — always-on ZZ coupling and, for cross-resonance (ECR/CX) gates, the drive on the control qubit
  shift the neighbours of a gate's qubits; a spectator *inside* the circuit (in superposition) dephases, one outside
  (in |0⟩) mostly adds a calibrated frequency shift;
* **simultaneous neighbouring gates** — two-qubit gates driven at the same time on adjacent couplers interfere
  (Zhao et al., PRX Quantum 3, 020301 (2022); Murali et al., ASPLOS 2020);
* **idle time** — qubits waiting between gates dephase (T2, low-frequency noise, residual ZZ), and a calibration's
  per-gate error never sees the idle windows a particular circuit has.

This module turns those mechanisms into countable *features* of every as-executed circuit (:func:`circuit_features`),
and provides the estimators of the divergence study that need no simulation:

=============================  =====================================================================================
:func:`plurality_vote`          Chen et al.'s plurality-vote aggregation of variant histograms (Quantum 8, 1516
                                (2024), App. B, Eqs. 4–6, after Maksymov et al., arXiv:2301.07233)
:func:`nnls_attribution`        the divergence budget: excess log-infidelity of every circuit regressed (non-negative
                                least squares, bootstrap over circuits) on its features — how much unmodeled error
                                each mechanism carries, per event
:func:`qubit_attribution`       the same per physical qubit (for the heavy-hex map)
:func:`heavy_hex_coordinates`   drawing coordinates of an IBM heavy-hex coupling graph (rows and bridge qubits)
=============================  =====================================================================================

Everything here is NumPy (SciPy for NNLS); nothing reads or writes a file.
"""
from __future__ import annotations

import math

import numpy as np

# feature name → (unit, what it counts); the order is the column order of the attribution
FEATURES = {
    "n2q": ("gate", "two-qubit gates (the component model already prices each one: a positive coefficient means "
                    "the calibrated error is too low)"),
    "n1q": ("gate", "physical one-qubit gates (√X, X; RZ is virtual)"),
    "active_spectators": ("gate·neighbour", "Σ over two-qubit gates of the circuit's own qubits coupled to the gate's "
                                            "qubits (spectators in superposition)"),
    "passive_spectators": ("gate·neighbour", "Σ over two-qubit gates of device qubits outside the circuit coupled to "
                                             "the gate's qubits (spectators in |0⟩)"),
    "control_spectators": ("gate·neighbour", "cross-resonance gates only: neighbours of the control qubit (the "
                                             "driven one)"),
    "parallel_adjacent": ("pair", "pairs of two-qubit gates that run at the same time on adjacent couplers"),
    "idle_us": ("µs", "idle time of the circuit's qubits between their first gate and the measurement (ALAP)"),
    "measured": ("qubit", "measured qubits (readout beyond its calibrated confusion matrix)"),
    "gate_error": ("Σ r", "the calibrated error the model already prices: Σ of the reported average gate infidelities "
                          "of every physical gate (a coefficient c: the calibration under-reports the gate errors "
                          "by about (1 + c)×)"),
    "readout_error": ("Σ p", "Σ over measured qubits of the reported readout error (p01 + p10)/2 (a coefficient: "
                             "readout worse than calibrated)"),
}
# features that come from the model's prediction, not from the circuit (computed by the analysis per predictor)
MODEL_FEATURES = {
    "priced": ("nat", "the error the model already prices, −ln F_pred (its own predicted log-infidelity): a "
                      "coefficient c means every calibrated error is (1 + c)× too low — a rescaled calibration, "
                      "no context"),
}
# the context mechanisms of the divergence budget, beside "priced" (n2q, n1q, measured and gate_error count events
# the model already prices, so they are nearly collinear with it: reported as features, kept out of the budget)
CONTEXT = ("active_spectators", "passive_spectators", "control_spectators", "parallel_adjacent", "idle_us")
BUDGET = ("priced",) + CONTEXT
DIRECTIONAL = ("ecr", "cx")


# ------------------------------------------------------------------ device graph
def device_neighbors(backend) -> dict:
    """{physical qubit: set of coupled physical qubits} from a backend's coupling map (or target)."""
    cm = getattr(backend, "coupling_map", None)
    edges = list(cm.get_edges()) if cm is not None else []
    if not edges and getattr(backend, "target", None) is not None:
        edges = [qa for qa in (backend.target.qargs or ()) if qa is not None and len(qa) == 2]
    nb = {}
    for a, b in edges:
        nb.setdefault(int(a), set()).add(int(b))
        nb.setdefault(int(b), set()).add(int(a))
    return nb


def _duration(durations: dict, name: str, qa: tuple) -> float:
    if name in ("rz", "z", "id", "barrier") or name.startswith("noise"):
        return 0.0
    for key in ((name, qa), (name, tuple(reversed(qa)))):
        if key in durations:
            return float(durations[key] or 0.0)
    if name == "y":                                   # a twirl Pauli: as long as an X
        return _duration(durations, "x", qa)
    return 0.0


def alap_schedule(spec, durations: dict) -> tuple:
    """(intervals, lead, busy, T): the ALAP schedule IBM backends use. intervals[i] = (start, end) of op i (seconds;
    barriers synchronize their qubits and take no time), lead[q] = time before qubit q's first operation (spent in
    |0⟩), busy[q] = time it is operated on, T = duration up to the (simultaneous) measurement."""
    n = spec.n
    ops = list(spec.ops)
    t = np.zeros(n)
    rev = [None] * len(ops)
    for i in range(len(ops) - 1, -1, -1):           # ASAP on the reversed circuit
        o = ops[i]
        qs = list(o.qubits) if o.qubits else list(range(n))
        start = float(max(t[q] for q in qs))
        d = _duration(durations, o.name, tuple(o.qubits)) if o.kind == "gate" else 0.0
        for q in qs:
            t[q] = start + d
        rev[i] = (start, start + d)
    T = float(t.max()) if n else 0.0
    intervals = [(T - e, T - s) for (s, e) in rev]   # mirror: ALAP in forward time
    lead, busy = np.full(n, T), np.zeros(n)
    for o, (s, e) in zip(ops, intervals):
        if o.kind != "gate":
            continue
        for q in o.qubits:
            lead[q] = min(lead[q], s)
            busy[q] += e - s
    return intervals, lead, busy, T


def circuit_features(spec, phys, neighbors: dict, durations: dict, two_qubit: str = "cz", errors: dict | None = None,
                     readout=None) -> dict:
    """The context features (:data:`FEATURES`) of one as-executed circuit on the compact qubits 0..n−1, where compact
    qubit i is physical qubit ``phys[i]``; ``neighbors`` from :func:`device_neighbors`; ``durations`` and ``errors``
    the device model's {(gate, compact qubits): seconds | reported error}; ``readout`` [(p01, p10)] per compact
    qubit. Without errors / readout, gate_error / readout_error are NaN."""
    phys = [int(p) for p in phys]
    inside = set(phys)
    gates = [(i, o) for i, o in enumerate(spec.ops) if o.kind == "gate"]
    two = [(i, o) for i, o in gates if len(o.qubits) == 2]
    f = dict.fromkeys(FEATURES, 0.0)
    f["n2q"] = float(len(two))
    f["n1q"] = float(sum(1 for _, o in gates if len(o.qubits) == 1 and o.name in ("sx", "x", "y", "sxdg")))
    f["measured"] = float(spec.n)
    if errors is None:
        f["gate_error"] = math.nan
    else:
        def err(o):
            qa = tuple(o.qubits)
            for key in ((o.name, qa), (o.name, tuple(reversed(qa)))):
                if key in errors:
                    return float(errors[key] or 0.0)
            return 0.0                                  # virtual gates (RZ) and gates the calibration does not list
        f["gate_error"] = float(sum(err(o) for _, o in gates))
    f["readout_error"] = (float(sum((a + b) / 2 for a, b in readout)) if readout is not None else math.nan)
    for _, o in two:
        a, b = (phys[q] for q in o.qubits)
        around = (neighbors.get(a, set()) | neighbors.get(b, set())) - {a, b}
        f["active_spectators"] += len(around & inside)
        f["passive_spectators"] += len(around - inside)
        if o.name in DIRECTIONAL:
            f["control_spectators"] += len(neighbors.get(a, set()) - {b})
    intervals, lead, busy, T = alap_schedule(spec, durations)
    f["idle_us"] = float(sum(max(T - lead[q] - busy[q], 0.0) for q in range(spec.n) if lead[q] < T)) * 1e6
    f["duration_us"] = T * 1e6
    for x in range(len(two)):
        i, o = two[x]
        s1, e1 = intervals[i]
        qa = {phys[q] for q in o.qubits}
        for y in range(x + 1, len(two)):
            j, p = two[y]
            s2, e2 = intervals[j]
            if not (s1 < e2 - 1e-15 and s2 < e1 - 1e-15):
                continue
            qb = {phys[q] for q in p.qubits}
            if qa & qb:
                continue
            if any(nb in qb for q in qa for nb in neighbors.get(q, ())):
                f["parallel_adjacent"] += 1
    return f


# ------------------------------------------------------------------ plurality voting (Chen et al. App. B)
def poisson_binomial_tail(P: np.ndarray, t: int) -> np.ndarray:
    """P(at least t successes) for independent trials with success probabilities P[b, v] (row b, trial v)."""
    P = np.atleast_2d(np.asarray(P, float))
    B, V = P.shape
    dist = np.zeros((B, V + 1))
    dist[:, 0] = 1.0
    for v in range(V):
        p = P[:, v][:, None]
        shifted = np.zeros_like(dist)
        shifted[:, 1:] = dist[:, :-1]
        dist = dist * (1 - p) + shifted * p
    return dist[:, int(t):].sum(axis=1) if t <= V else np.zeros(B)


def default_threshold(variants: int) -> int:
    """Chen et al. start at t = 7 for their 25 variants (``approximate_plurality_vote(hists, t=7)`` in the data
    repository); for other variant counts the same ratio, at least 3 (t ≤ 2 is no vote at all)."""
    return int(min(variants, max(3, round(0.28 * variants))))


def plurality_vote(F, t_start: int | None = None, renormalize: bool = False) -> tuple:
    """(distribution, threshold used) of Chen et al.'s approximate plurality vote over variant histograms F[v, b]
    (frequencies; every variant ran the same circuit in a different physical form — here: a different heavy-hex
    chain and/or Pauli twirl). It follows the authors' published reference implementation (App. B, Eqs. 4–6, and
    their ``pluralityvote.py``) step by step, and is tested against brute-force enumeration of every joint draw:

    1. occurrences o_b = number of variants with b; the threshold is t = min(t_start, max_b o_b) — not decreased
       one by one: the largest occurrence is where the first survivor appears;
    2. t ≤ 2: no consensus — the bit-string-wise average of the variants;
    3. survivors B = {b : o_b ≥ t}; a single survivor is returned with probability 1;
    4. variants without a survivor are dropped; p_b = P(b is drawn in ≥ t of the remaining N_v variants when each is
       sampled once) = Σ_{m≥t} Σ_{|S|=m} Π_{v∈S} f_vb Π_{v∉S} (1 − f_vb) (Eqs. 4–5), with f_vb the variant's measured
       frequency of b — not renormalized over B. The reference enumerates the subsets (Eq. 6 below N_v/2); the
       Poisson-binomial recursion of :func:`poisson_binomial_tail` gives the same numbers in O(N_v²);
    5. the p_b are normalized.

    ``renormalize=True`` renormalizes every variant's frequencies over B before step 4 (the reading of "all
    subsequent probabilities are relative to this reduced collection" in the paper's App. B); it is not what the
    reference implementation does."""
    F = np.atleast_2d(np.asarray(F, float))
    V, D = F.shape
    if D == 0:
        return np.zeros(0), 2
    occ = (F > 0).sum(axis=0)
    t = default_threshold(V) if t_start is None else int(t_start)
    t = int(min(t, occ.max()))
    if t <= 2:
        avg = F.mean(axis=0)
        return (avg / avg.sum() if avg.sum() > 0 else avg), 2
    keep = occ >= t
    out = np.zeros(D)
    if keep.sum() == 1:
        out[keep] = 1.0
        return out, t
    G = F[:, keep]
    G = G[G.sum(axis=1) > 0]
    if renormalize:
        G = G / G.sum(axis=1, keepdims=True)
    tail = poisson_binomial_tail(G.T, t)
    out[keep] = tail / tail.sum()
    return out, t


def plurality_vote_rows(Fs: list, t_start: int | None = None, renormalize: bool = False) -> np.ndarray:
    """The vote replicate by replicate: Fs[v] is an (R, D) array of variant v's frequency rows."""
    A = np.stack([np.atleast_2d(x) for x in Fs], axis=1)            # (R, V, D)
    return np.array([plurality_vote(A[r], t_start, renormalize)[0] for r in range(A.shape[0])])


# ------------------------------------------------------------------ attribution (the divergence budget)
def _vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor of every column (collinearity: > 10 means the column's coefficient is poorly
    identified by these circuits)."""
    out = np.full(X.shape[1], np.nan)
    for k in range(X.shape[1]):
        others = np.delete(X, k, axis=1)
        y = X[:, k]
        A = np.column_stack([others, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - A @ coef) ** 2) / ss if ss > 0 else 1.0
        out[k] = 1 / max(1 - r2, 1e-12)
    return out


def nnls_attribution(X, y, names, boot: int = 500, seed: int = 0, ridge: float = 0.0, weights=None) -> dict:
    """y_c ≈ Σ_k β_k X_ck with β ≥ 0 (non-negative least squares, optional ridge α‖β‖²): the unmodeled log-infidelity
    of every circuit c split into mechanisms k (features), each priced per event. ``weights`` w_c (e.g. 1 / SE of
    y_c) make it weighted least squares — a circuit whose excess is mostly shot noise does not pull the fit.
    Bootstrap over circuits gives the percentile intervals. Columns of zeros are dropped (reported as NaN).
    Returns {names, coef, lo, hi (2.5 / 97.5 %), r2 (weighted), fitted, residual, vif, cond, share
    (Σ_c β_k X_ck / Σ_c ŷ_c), circuits}."""
    from scipy.optimize import nnls
    X, y = np.asarray(X, float), np.asarray(y, float)
    W = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(W) & (W > 0)
    X, y, W = X[ok], y[ok], W[ok]
    names = list(names)
    use = [k for k in range(X.shape[1]) if np.any(X[:, k] != 0)]      # a column of zeros prices nothing
    if len(y) < max(3, len(use) + 1) or not use:
        nan = np.full(len(names), np.nan)
        return dict(names=names, chi2=math.nan, dof=int(len(y)), coef=nan, lo=nan, hi=nan, r2=math.nan,
                    fitted=np.full(len(y), np.nan),
                    residual=y, vif=nan, cond=math.nan, share=nan, circuits=int(len(y)))
    Xu = X[:, use]
    scale = np.abs(Xu).max(axis=0)                     # conditioning only; β is reported in the features' units

    def fit(A, b, w):
        As = A / scale * w[:, None]
        b = b * w
        if ridge > 0:
            As = np.vstack([As, math.sqrt(ridge) * np.eye(As.shape[1])])
            b = np.concatenate([b, np.zeros(As.shape[1])])
        c, _ = nnls(As, b)
        return c / scale
    beta = fit(Xu, y, W)
    rng = np.random.default_rng(seed)
    draws = (rng.integers(0, len(y), len(y)) for _ in range(int(boot)))
    B = np.array([fit(Xu[i], y[i], W[i]) for i in draws]) if boot else np.empty((0, len(use)))
    fitted = Xu @ beta
    ybar = float(np.sum(W ** 2 * y) / np.sum(W ** 2))
    ss = float(np.sum(W ** 2 * (y - ybar) ** 2))
    r2 = 1 - float(np.sum(W ** 2 * (y - fitted) ** 2)) / ss if ss > 0 else math.nan
    full = lambda v: np.array([v[use.index(k)] if k in use else np.nan for k in range(len(names))])   # noqa: E731
    contrib = (Xu * beta).sum(axis=0)
    tot = fitted.sum()
    chi2 = float(np.sum((W * (y - fitted)) ** 2))
    return dict(names=names, chi2=chi2, dof=int(len(y) - np.sum(beta > 0)),
                coef=full(beta), lo=full(np.percentile(B, 2.5, axis=0)) if len(B) else full(beta * np.nan),
                hi=full(np.percentile(B, 97.5, axis=0)) if len(B) else full(beta * np.nan), r2=r2, fitted=fitted,
                residual=y - fitted, vif=full(_vif(Xu)) if Xu.shape[1] > 1 else full(np.ones(1)),
                cond=float(np.linalg.cond(Xu / scale)), share=full(contrib / tot if tot > 0 else contrib * np.nan),
                circuits=int(len(y)))


def qubit_attribution(loads: list, y, boot: int = 300, seed: int = 0, ridge: float = 1e-3, weights=None) -> dict:
    """Per physical qubit: excess log-infidelity per two-qubit-gate participation. ``loads`` = [{physical qubit:
    two-qubit gates on it}] per circuit, ``y`` = their excess log-infidelities. Non-negative ridge least squares
    (qubits seen by few circuits are poorly determined: their interval says so). Returns {qubit: (β, lo, hi,
    circuits using it)}."""
    qubits = sorted({q for L in loads for q in L})
    if not qubits:
        return {}
    X = np.array([[L.get(q, 0) for q in qubits] for L in loads], float)
    r = nnls_attribution(X, y, qubits, boot=boot, seed=seed, ridge=ridge, weights=weights)
    used = (X > 0).sum(axis=0)
    return {q: (float(r["coef"][k]), float(r["lo"][k]), float(r["hi"][k]), int(used[k])) for k, q in enumerate(qubits)}


# ------------------------------------------------------------------ heavy-hex drawing coordinates
def heavy_hex_coordinates(edges, num_qubits: int) -> dict:
    """{qubit: (x, y)} for an IBM heavy-hex coupling graph: rows are runs of consecutively numbered qubits joined by
    (i, i+1) couplers, the qubits left over are bridges between rows; rows are aligned through their bridges. Other
    graphs fall back to a force-directed layout."""
    E = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    nb = {q: set() for q in range(num_qubits)}
    for a, b in E:
        nb[a].add(b)
        nb[b].add(a)
    rows, cur = [], [0]
    for q in range(1, num_qubits):
        if (q - 1, q) in E:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    rows = [r for r in rows if len(r) >= 3]
    in_row = {q: i for i, r in enumerate(rows) for q in r}
    bridges = [q for q in range(num_qubits) if q not in in_row]
    if len(rows) < 2 or any(len(nb[b]) == 0 for b in bridges):
        return _spring(E, num_qubits)
    pos, offset = {}, {0: 0}
    for i in range(1, len(rows)):                       # align each row to an earlier one through a bridge
        for b in bridges:
            ends = [(in_row[q], q) for q in nb[b] if q in in_row]
            rs = {r for r, _ in ends}
            if i in rs and any(r < i and r in offset for r in rs):
                u = next(q for r, q in ends if r < i and r in offset)
                v = next(q for r, q in ends if r == i)
                offset[i] = offset[in_row[u]] + (u - rows[in_row[u]][0]) - (v - rows[i][0])
                break
        offset.setdefault(i, 0)
    for i, r in enumerate(rows):
        for q in r:
            pos[q] = (float(offset[i] + q - r[0]), -2.0 * i)
    for b in bridges:
        pts = [pos[q] for q in nb[b] if q in pos]
        if len(pts) >= 2:                               # a bridge between two rows: half-way
            pos[b] = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        elif pts:                                       # a pendant qubit (e.g. below the last row of ibm_torino):
            r = in_row[next(q for q in nb[b] if q in pos)]   # one unit outwards, away from the other rows
            pos[b] = (pts[0][0], pts[0][1] + (1.0 if r == 0 and len(rows) > 1 else -1.0))
    if len(pos) < num_qubits:
        return _spring(E, num_qubits)
    return pos


def _spring(E, n) -> dict:
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(E)
        return {int(k): (float(v[0]), float(v[1])) for k, v in nx.kamada_kawai_layout(G).items()}
    except Exception:                                   # noqa: BLE001 - drawing only
        return {q: (math.cos(2 * math.pi * q / max(n, 1)), math.sin(2 * math.pi * q / max(n, 1))) for q in range(n)}
