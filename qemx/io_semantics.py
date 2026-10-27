"""Input and output semantics of an experiment (1.5; one input model since 1.5.1).

The editable source of truth of an experiment is its **input**, the **circuit**, its **parameters** and the
**measurement** (per-qubit basis and the classical bit order). The **output** is derived from these and is never an
editable control: reading or refreshing it changes nothing.

**One input model (1.5.1).** The input is the product state the circuit starts from: one single-qubit state per qubit,
written as an input token (``qemx.compile``: ``0 1 + - +i -i``, ``ph:<turns>``, ``bloch:<θ>,<φ>``). The compiler
prepares it exactly once, before the circuit (``compile.with_io``). A template writes its input preparation as leading
single-qubit gates tagged ``prep``; ``split_preparation`` turns those into input tokens and returns the circuit
without them, so the Playground, the experiment documents and the batch runner all hold *tokens + circuit body* and
never prepare twice. Template parameters with ``role="input"`` (the QFT's input index and preparation, the eigenstate
of the Ising QPE, the logical θ of the error-correcting codes) only choose those tokens: ``template_inputs`` gives the
tokens for a parameter set and ``infer_template_input`` finds the parameters that reproduce given tokens (or None).
Everything a template builds after the input — encoders, oracles, an algorithm's own Hadamard layers — is circuit.
The phase-study families (``qemx.phase.families``) keep the preparation written into their circuits; that is the same
state and keeps the identity of the frozen pilot and holdout circuits.

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
    """The input preparation written into a circuit: its leading single-qubit gates tagged ``prep`` (a template's
    input-loading gates, or a circuit saved before 1.5.1). ``split_preparation`` turns them into input tokens."""
    out = []
    for o in spec.ops:
        if o.tag != "prep" or o.kind != "gate" or len(o.qubits) != 1:
            break
        out.append(o)
    return out


def split_preparation(spec: CircuitSpec, inputs=None) -> tuple:
    """(input tokens, circuit body): the leading ``prep`` gates of ``spec`` folded into the per-qubit inputs.

    Each qubit's token is the state its given input (default |0⟩) reaches through its preparation gates, written with
    the same gates whenever a token can (``faithful_token``), so ``compile.with_io(body, tokens)`` prepares the same
    state as ``with_io(spec, inputs)`` — once — and, for a template, with the same compiled gates as before 1.5.1. A circuit without
    preparation gates comes back unchanged. Pure: nothing passed in is modified."""
    from .circuit import gate_matrix
    from .compile import input_state, state_token
    n = int(spec.n)
    inputs = list(inputs) if inputs is not None else ["0"] * n
    if len(inputs) != n:
        raise ValueError("inputs must list one token per qubit (%d)" % n)
    from .compile import input_ops
    prep = preparation_ops(spec)
    if not prep:
        return inputs, spec
    vec, seq = {}, {}
    for o in prep:
        q = o.qubits[0]
        if q not in vec:
            vec[q], seq[q] = input_state(inputs[q]), list(input_ops(inputs[q]))
        vec[q] = np.asarray(gate_matrix(o.name, o.params), dtype=complex) @ vec[q]
        seq[q].append((o.name, tuple(o.params)))
    tokens = [faithful_token(seq[q], vec[q]) if q in vec else inputs[q] for q in range(n)]
    return tokens, spec.replace(ops=tuple(spec.ops[len(prep):]))


def _same_gates(a, b) -> bool:
    """Same gate names and angles (a phase gate's angle modulo 2π: P(φ + 2π) = P(φ))."""
    if [g for g, _ in a] != [g for g, _ in b]:
        return False
    for (g, pa), (_, pb) in zip(a, b):
        pa, pb = [float(x) for x in pa], [float(x) for x in pb]
        if len(pa) != len(pb):
            return False
        for x, y in zip(pa, pb):
            d = (x - y + math.pi) % (2 * math.pi) - math.pi if g == "p" else x - y
            if abs(d) > 1e-9:
                return False
    return True


def faithful_token(seq, v) -> str:
    """The input token of state ``v`` whose preparation (``compile.input_ops``) is exactly the gate sequence ``seq``
    — names and angles — when one exists, else the canonical token (``compile.state_token``).

    Moving a template's preparation gates into the inputs then leaves its compiled circuit, and so its noisy
    simulation, exactly as before 1.5.1: the QFT's Fourier input H·P(π)|0⟩ stays the phase input ``ph:0.5`` instead of
    the canonical |−⟩ = H·X|0⟩, which an abstract noise model charges for one more gate."""
    from .compile import INPUT_STATES, STATE_TOL, input_ops, input_state, phase_token, state_token, valid_input
    seq = [(g, tuple(p)) for g, p in seq]
    names = [g for g, _ in seq]
    cands = [state_token(v)] + list(INPUT_STATES)
    if names == ["h", "p"]:
        cands.append(phase_token(float(seq[1][1][0]) / (2 * math.pi)))
    if names in (["ry"], ["ry", "p"]):
        cands.append("bloch:%.12g,%.12g" % (float(seq[0][1][0]), float(seq[1][1][0]) % (2 * math.pi)
                                            if len(seq) == 2 else 0.0))
    for tok in cands:
        if valid_input(tok) and _same_gates(input_ops(tok), seq) \
                and abs(np.vdot(input_state(tok), v)) >= 1 - STATE_TOL:
            return tok
    return state_token(v)


def template_inputs(key: str, params: dict | None = None) -> tuple:
    """(input tokens, circuit body, Template) of a template: its preparation as inputs, the rest as circuit.

    Encoded templates whose analysis indexes the circuit's operations (``Template.qec``) or whose live parameters
    point at operations (``Template.variational``) carry no preparation, so their circuit is never shifted."""
    from . import templates as T
    tpl = T.build(key, params or {})
    tokens, body = split_preparation(tpl.spec)
    if (tpl.qec or tpl.variational) and body is not tpl.spec:   # pragma: no cover - guarded by tests
        raise AssertionError("template %s puts input preparation before indexed operations" % key)
    return tokens, body, tpl


def input_params(key: str) -> list:
    """The template's parameters that choose only its input (``TParam.role == "input"``)."""
    from . import templates as T
    return [p for p in T.REGISTRY[key][2] if getattr(p, "role", "circuit") == "input"] if key in T.REGISTRY else []


def same_inputs(a, b) -> bool:
    from .compile import same_input
    return len(list(a)) == len(list(b)) and all(same_input(x, y) for x, y in zip(a, b))


def infer_template_input(key: str, params: dict, tokens) -> dict | None:
    """The input parameters of template ``key`` (with its circuit parameters from ``params``) whose input equals
    ``tokens``; None when no setting of the input parameters gives these inputs. Candidates come from the tokens
    themselves (the index s of a Fourier or basis state, the system bits, the angle θ) and are then checked by
    building the template, so a match is exact (to the state tolerance of ``compile.same_input``)."""
    from . import templates as T
    from .compile import input_state
    ips = input_params(key)
    if not ips:
        return None
    base = T.clamp_params(key, params or {})
    tokens = list(tokens)
    cands = []
    try:
        if key in ("qft", "aqft"):
            n = int(base["n"])
            if all(t in ("0", "1") for t in tokens):
                cands.append(dict(prep="basis", s=sum(1 << q for q, t in enumerate(tokens) if t == "1")))
            v = input_state(tokens[0])
            if abs(abs(v[0]) - abs(v[1])) < 1e-9:                 # Fourier: qubit 0 carries e^{-2πi x/N}
                x = int(round(-float(np.angle(v[1] / v[0])) / (2 * math.pi) * 2 ** n)) % 2 ** n
                s = T.rev(x, n) if base.get("mode") == "bitreverse" else x
                cands.append(dict(prep="fourier", s=s))
        elif key == "qpe_ising":
            t = int(base["t"])
            if len(tokens) == t + 2 and all(x in ("0", "1") for x in tokens[t:]):
                cands.append(dict(state=tokens[t + 1] + tokens[t]))      # written q(t+1) q(t), as the template
        elif any(p.name == "theta" for p in ips):
            q = T.THETA_QUBIT[key]
            v = input_state(tokens[q])
            v = v * np.exp(-1j * np.angle(v[0] if abs(v[0]) > 1e-12 else v[1]))   # remove the global phase
            if abs(v.imag).max() < 1e-9:                             # RY(θ)|0⟩ has real amplitudes
                c, s_ = float(v[0].real), float(v[1].real)
                if s_ < -1e-12 or (abs(s_) <= 1e-12 and c < 0):
                    c, s_ = -c, -s_
                cands.append(dict(theta=float("%.12g" % (float(2 * math.atan2(s_, c)) % (2 * math.pi)))))
    except (ValueError, IndexError, KeyError):
        return None
    for c in cands:
        trial = T.clamp_params(key, dict(base, **c))
        if any(trial[k] != v for k, v in c.items() if not isinstance(v, float)):
            continue
        try:
            want = template_inputs(key, trial)[0]
        except (ValueError, ArithmeticError):
            continue
        if same_inputs(want, tokens):
            return {p.name: trial[p.name] for p in ips}
    return None


def describe_input(spec: CircuitSpec, inputs) -> dict:
    """What the input is, in words: the per-qubit input states (and any preparation gates still written into the
    circuit, e.g. in a circuit typed by hand with ``prep`` tags)."""
    from .compile import input_label
    inputs = list(inputs or ["0"] * spec.n)
    prep = preparation_ops(spec)
    per_qubit = [input_label(t) for t in inputs]
    gates = {}
    for o in prep:
        gates[o.name] = gates.get(o.name, 0) + 1
    return dict(per_qubit=per_qubit, nonzero=[q for q, t in enumerate(inputs) if t != "0"],
                prep_gates=len(prep), prep_summary=", ".join("%d %s" % (v, k.upper()) for k, v in gates.items()))


def bit_order_text(n: int, relabel: bool) -> str:
    """The classical bit order of the measurement, in words."""
    if n < 2:
        return "c0 ← q0"
    if relabel:
        return "read in reverse order: c%d ← q0, …, c0 ← q%d (classical relabeling)" % (n - 1, n - 1)
    return "c%d ← q%d, …, c0 ← q0" % (n - 1, n - 1)


def basis_text(basis: str) -> str:
    b = str(basis)
    if len(set(b)) <= 1:
        return "%s on every qubit" % (b[:1] or "Z")
    return "%s (q0 first)" % b


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


# ====================================================================== A · B · C · D (1.7.0)
# The Playground keeps four things apart (review of 1.6.0, the "Inputs and measurement" report):
#   A  the actual input state ρ_in (the per-qubit input tokens, prepared exactly once);
#   B  the ideal output state before measurement, ρ_out = C(ρ_in) — it does not depend on the measurement basis;
#   C  the distribution of the classical bits in the selected bases, p_b(z) = Tr[|z⟩⟨z| R_b ρ_out R_b†], with the
#      declared bit order (c_i ← qubit mm[i]);
#   D  the observable, a function of the measured classical bits, and its expectation value Σ_z o(z) p_b(z).
# A diagonal observable such as P(0000) is defined on the *measured bits*: in the Z basis it is the computational-basis
# projector |0000⟩⟨0000|; in other bases its physical operator is R_b† O R_b (P(0000) measured in X on every qubit is
# the projector onto |++++⟩). ``observable_meaning`` says which, so changing the basis never changes the physical
# observable silently.
NAMED_STATES = (("0", (0.0, 0.0, 1.0)), ("1", (0.0, 0.0, -1.0)), ("+", (1.0, 0.0, 0.0)), ("−", (-1.0, 0.0, 0.0)),
                ("+i", (0.0, 1.0, 0.0)), ("−i", (0.0, -1.0, 0.0)))
EIGENSTATE = {("Z", "0"): "0", ("Z", "1"): "1", ("X", "0"): "+", ("X", "1"): "−", ("Y", "0"): "+i", ("Y", "1"): "−i"}
PURE_TOL = 1e-9


def experiment_key(body: CircuitSpec, inputs, basis: str, relabel: bool, observable=None) -> str:
    """One canonical description of an experiment (input tokens, circuit body, bases, bit order, observable): every
    derived preview is keyed by it, so any change of one of them recomputes or invalidates the preview."""
    import hashlib
    obs = getattr(observable, "key", None) if observable is not None else None
    vals = getattr(observable, "values", None)
    vdig = hashlib.sha1(np.asarray(vals, float).tobytes()).hexdigest()[:12] if vals is not None and len(vals) else ""
    return hashlib.sha1(repr((int(body.n), body.key(), tuple(inputs), str(basis), bool(relabel), obs,
                              vdig)).encode()).hexdigest()


def _bloch(psi: np.ndarray, n: int, q: int) -> np.ndarray:
    """Bloch vector of qubit q of an n-qubit pure state (little-endian amplitudes: bit q of the index is qubit q)."""
    t = np.asarray(psi, complex).reshape([2] * n)           # axes: q(n−1), …, q0
    ax = n - 1 - q
    t = np.moveaxis(t, ax, 0).reshape(2, -1)
    rho = t @ t.conj().T
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])


def qubit_state_text(r) -> str:
    """A qubit's reduced state in words: a named pure state, a pure state on the Bloch sphere, or mixed."""
    r = np.asarray(r, float)
    norm = float(np.linalg.norm(r))
    if norm < 1 - 1e-6:
        return "mixed (entangled with the other qubits; |r| = %.3f)" % norm
    for name, v in NAMED_STATES:
        if np.allclose(r, v, atol=1e-6):
            return "|%s⟩" % name
    th = math.acos(max(-1.0, min(1.0, r[2])))
    ph = math.atan2(r[1], r[0]) % (2 * math.pi)
    return "|θ=%.3f, φ=%.3f⟩" % (th, ph)


def output_state(body: CircuitSpec, inputs, max_qubits: int = OUTPUT_PREVIEW_MAX_QUBITS, top: int = 8) -> dict:
    """B. The ideal output state before measurement: the circuit's gates (noise operations skipped) applied to the
    input product state, *without* any measurement-basis rotation — so it does not depend on the selected bases.
    Returns ``status`` (exact | not computed), whether it is a product state, every qubit's reduced state (Bloch
    vector and words), and the largest computational-basis amplitudes (|q(n−1)…q0⟩, qubit order, before any
    classical relabeling)."""
    n = int(body.n)
    out = dict(kind="ideal output state (before measurement)", n=n, status="not computed", reason="", product=None,
               qubits=[], amplitudes=[], text="")
    if n > int(max_qubits):
        out["reason"] = ("an exact state of %d qubits has 2^%d amplitudes; the preview stops at %d qubits"
                         % (n, n, int(max_qubits)))
        return out
    from .compile import with_io
    try:
        psi = ideal_statevector(with_io(body, list(inputs), "Z" * n))
    except MemoryError:
        out["reason"] = "not enough memory for the statevector of %d qubits" % n
        return out
    psi = np.asarray(psi, complex)
    psi = psi / np.linalg.norm(psi)
    rs = [_bloch(psi, n, q) for q in range(n)]
    product = all(np.linalg.norm(r) >= 1 - 1e-6 for r in rs)
    p = np.abs(psi) ** 2
    nz = np.flatnonzero(p > 1e-12)
    order = nz[np.argsort(-p[nz], kind="stable")][: int(top)]
    k0 = order[0] if len(order) else 0
    ph0 = np.angle(psi[k0]) if len(order) else 0.0
    amps = [(format(int(i), "0%db" % n), complex(psi[i] * np.exp(-1j * ph0))) for i in order]
    words = [qubit_state_text(r) for r in rs]
    if product:
        if len(set(words)) == 1:
            text = "Product state: %s on every qubit" % words[0]
        elif all(w in ("|0⟩", "|1⟩") for w in words):
            text = "Product state |%s⟩ (q%d … q0)" % ("".join(w[1] for w in reversed(words)), n - 1)
        else:
            text = "Product state " + " ⊗ ".join(reversed(words)) + " (q%d … q0)" % (n - 1)
    else:
        ent = [q for q, r in enumerate(rs) if np.linalg.norm(r) < 1 - 1e-6]
        text = ("Entangled state: %d of %d qubits are not in a pure state of their own (%s); the largest "
                "computational-basis amplitudes: %s" % (len(ent), n, ", ".join("q%d" % q for q in ent),
                                                         ", ".join("%s|%s⟩" % (_amp(a), b) for b, a in amps[:4])))
    out.update(status="exact", product=bool(product), qubits=[dict(qubit=q, bloch=[float(x) for x in r],
                                                                    state=w) for q, (r, w) in enumerate(zip(rs, words))],
               amplitudes=amps, support=int(len(nz)), text=text)
    return out


def _amp(a: complex) -> str:
    if abs(a.imag) < 1e-9:
        return "%+.3f" % a.real
    if abs(a.real) < 1e-9:
        return "%+.3fi" % a.imag
    return "(%.3f%+.3fi)" % (a.real, a.imag)


def measurement_distribution(body: CircuitSpec, inputs, basis: str, relabel: bool, observable=None,
                             max_qubits: int = OUTPUT_PREVIEW_MAX_QUBITS, top: int = 8) -> dict:
    """C (and D's value). The ideal distribution of the classical bits c(n−1)…c0 when qubit q is measured in
    ``basis[q]`` (q0 first) and bit i is read from qubit mm[i] (``relabel``: reversed). Same fields as
    ``derived_output`` plus the bases and ``status``."""
    from .compile import with_io
    n = int(body.n)
    d = derived_output(with_io(body, list(inputs), str(basis)), bool(relabel), observable, top=top,
                       max_qubits=max_qubits)
    d.update(kind="ideal measurement distribution", basis=str(basis), relabel=bool(relabel),
             status="exact" if d.get("available") else "not computed")
    if d.get("available") and not d.get("deterministic"):
        d["uniform"] = bool(d["support"] == 2 ** n and abs(d["top"][0][1] - 2.0 ** -n) < 1e-9)
    return d


def observable_meaning(observable, basis: str, relabel: bool, n: int) -> str:
    """D. What a diagonal observable measures: it is a function of the measured classical bits, so its physical
    operator depends on the bases. For a projector P(z) the eigenstate it projects onto is named."""
    if observable is None or hasattr(observable, "terms"):
        return ""
    label = getattr(observable, "label", "⟨O⟩")
    allz = set(str(basis)) <= {"Z"}
    vals = np.asarray(getattr(observable, "values", ()), float)
    target = None
    if vals.size == 2 ** n and np.count_nonzero(vals) == 1 and np.isclose(vals.max(), 1.0):
        target = format(int(np.argmax(vals)), "0%db" % n)          # classical bits c(n−1)…c0
    mm = measure_map(n, relabel)
    if target is not None:
        per_q = {}
        for i in range(n):
            bit = target[n - 1 - i]
            q = mm[i]
            per_q[q] = EIGENSTATE[(str(basis)[q], bit)]
        state = "|" + "".join(per_q[q] for q in reversed(range(n))) + "⟩" if all(len(v) == 1 for v in per_q.values()) \
            else " ⊗ ".join("|%s⟩" % per_q[q] for q in reversed(range(n)))
        if allz:
            return ("%s is the probability that the measured bits c%d…c0 read %s: in the Z basis this is the "
                    "computational-basis projector onto the qubit state %s (q%d … q0%s)." % (
                        label, n - 1, target, state, n - 1, ", bits read in reverse order" if relabel else ""))
        return ("%s is the probability that the measured bits c%d…c0 read %s *in the selected bases* (%s). The "
                "basis rotations make it the projector onto %s (q%d … q0) — not the computational-basis projector "
                "of the same name: changing the measurement basis changed the physical observable." % (
                    label, n - 1, target, basis_text(basis), state, n - 1))
    if allz:
        return "%s is a function of the measured bits c%d…c0 (Z basis on every qubit)." % (label, n - 1)
    return ("%s is a function of the measured bits c%d…c0 in the selected bases (%s): its physical operator is "
            "R_b† O R_b, not the Z-basis operator of the same name." % (label, n - 1, basis_text(basis)))


def template_claim(template_active: bool, inputs_match: bool, basis: str, relabel: bool, template_relabel: bool,
                   template_basis: str | None = None) -> tuple:
    """(applies, reasons): whether a template's identity (its formula and notes, e.g. F F†|s⟩ = |s⟩) describes the
    current experiment. It does only while the circuit is unedited, the inputs are the template's own, the bases are
    the template's (Z on every qubit unless stated) and the bit order is the template's."""
    reasons = []
    if not template_active:
        reasons.append("the circuit was edited after loading the template")
    if template_active and not inputs_match:
        reasons.append("the inputs differ from the template's input")
    tb = template_basis or "Z" * len(str(basis))
    if str(basis) != tb:
        reasons.append("the measurement bases (%s) differ from the template's (%s)" % (basis_text(basis),
                                                                                   basis_text(tb)))
    if bool(relabel) != bool(template_relabel):
        reasons.append("the bit order differs from the template's")
    return (not reasons), reasons


def apply_summary(old_inputs, new_inputs, old_basis: str, new_basis: str, inputs_given: bool = True,
                  basis_given: bool = True) -> str:
    """What "Apply to all qubits" changed, in words. It may change nothing: |+⟩ on every qubit is, for example,
    already the QFT's Fourier input F†|0…0⟩ = |+⟩^⊗n, so the output stays the same — correctly."""
    from .compile import input_label, same_input
    n = len(new_inputs)
    parts = []
    if inputs_given:
        ch_in = [q for q in range(n) if not same_input(old_inputs[q], new_inputs[q])]
        parts.append("inputs: %d of %d qubits changed" % (len(ch_in), n) if ch_in else
                     "inputs unchanged — every qubit already was in that state (%s), so the output does not change"
                     % ", ".join(sorted({input_label(t) for t in new_inputs})))
    if basis_given:
        ch_bs = [q for q in range(n) if str(old_basis)[q] != str(new_basis)[q]]
        parts.append("bases: %d of %d qubits changed (the output state does not depend on them; the measured "
                     "distribution does)" % (len(ch_bs), n) if ch_bs else "bases unchanged")
    return "; ".join(parts)
