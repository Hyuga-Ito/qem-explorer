r"""Hamiltonian observables: real linear combinations of Pauli strings,

.. math:: H=\sum_k c_k\,P_k,\qquad P_k\in\{I,X,Y,Z\}^{\otimes n},

typed as text, e.g. ``0.5*Z0*Z1 + 0.2*X0 - 0.1*Y1``.

A Pauli string with X or Y factors is not diagonal in the computational basis, so it is measured after a
basis change (H for X, S†·H for Y on that qubit). Terms that agree qubit by qubit ("qubit-wise commuting", QWC)
share one measurement setting; the terms are grouped greedily, and every group becomes an ordinary *diagonal*
observable on a copy of the implemented circuit with the basis change appended before measurement. Every QEM
method therefore runs unchanged on every group, and the results are combined:

.. math:: \hat H=c_I+\sum_g\hat O_g,\qquad
          \mathrm{Var}(\hat H)=\sum_g\frac{v_g}{N_g}=\frac{\bigl(\sum_g\sqrt{v_g}\bigr)^2}{N}
          \quad\text{for the optimal split }N_g\propto\sqrt{v_g}.

The parser accepts numbers, simple expressions (``pi/4``, ``sqrt(2)/2``) through the app's whitelist expression
reader, and Pauli factors ``X0``, ``Y_3``, ``Z12`` joined by ``*``, spaces or nothing. Nothing is executed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .circuit import Op, eval_expr
from .observables import Observable

MAX_TEXT = 4000
MAX_TERMS = 256
MAX_COEF = 1e9                  # |coefficient| (larger values are almost certainly typos)
_FACTOR = re.compile(r"([IXYZ])_?(\d+)")
_COEF_OK = re.compile(r"^[0-9eE.+\-*/()\s πpitausqrcon]*$")
# Pauli products on one qubit: (a, b) -> (phase, result)
_MUL = {("X", "X"): (1, "I"), ("Y", "Y"): (1, "I"), ("Z", "Z"): (1, "I"),
        ("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
        ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y")}


class HamiltonianError(ValueError):
    pass


@dataclass(frozen=True)
class PauliSum:
    """H = Σ c_k P_k. ``terms``: ((coefficient, ((qubit, 'X'|'Y'|'Z'), ...)), ...), identity has no factors."""
    key: str
    label: str
    tex: str
    n: int
    terms: tuple
    text: str

    @property
    def constant(self) -> float:
        return float(sum(c for c, f in self.terms if not f))

    def groups(self) -> list:
        return qwc_groups(self.terms)

    def matrix(self) -> np.ndarray:
        """Dense 2^n × 2^n matrix (little-endian), for small n (time travel, tests)."""
        from .noise import PAULI1
        d = 2 ** self.n
        M = np.zeros((d, d), dtype=complex)
        for c, fac in self.terms:
            f = dict(fac)
            m = np.array([[1.0 + 0j]])
            for q in range(self.n):
                m = np.kron(PAULI1[f.get(q, "I")], m)
            M += c * m
        return M

    def norm1(self) -> float:
        return float(sum(abs(c) for c, _ in self.terms))


def _term_tex(c: float, fac, first: bool) -> str:
    s = ("-" if c < 0 else "") if first else (" - " if c < 0 else " + ")
    a = abs(c)
    coef = "" if (abs(a - 1) < 1e-12 and fac) else ("%g" % a)
    ops = "".join("%s_{%d}" % (p, q) for q, p in fac)
    return s + coef + ops


def _term_text(c: float, fac, first: bool) -> str:
    s = ("-" if c < 0 else "") if first else (" - " if c < 0 else " + ")
    a = abs(c)
    coef = "" if (abs(a - 1) < 1e-12 and fac) else ("%g" % a)
    ops = "·".join("%s%d" % (p, q) for q, p in fac)
    return s + coef + ("·" if coef and ops else "") + ops


def _split_terms(text: str) -> list:
    """Split at top-level + and − (not inside parentheses, not the sign of an exponent like 1e-3)."""
    out, cur, depth = [], "", 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise HamiltonianError("unbalanced parentheses")
        prev = cur.rstrip()[-1:]
        if ch in "+-" and depth == 0 and cur.strip() and not re.search(r"\d[eE]$", cur.rstrip()) \
                and prev not in ("*", "/", "("):                 # a sign after an operator is unary: 2*-Z0, 2**-3
            if cur.strip() not in ("+", "-"):
                out.append(cur)
                cur = ch
                continue
        cur += ch
    if depth != 0:
        raise HamiltonianError("unbalanced parentheses")
    if cur.strip():
        out.append(cur)
    return out


def parse_pauli_sum(text: str, n: int) -> PauliSum:
    """Parse ``text`` into a :class:`PauliSum` on ``n`` qubits (strict; raises HamiltonianError)."""
    if not isinstance(text, str):
        raise HamiltonianError("the Hamiltonian must be text")
    t = text.replace("−", "-").replace("·", "*").replace("×", "*").strip()
    if not t:
        raise HamiltonianError("empty Hamiltonian")
    if len(t) > MAX_TEXT:
        raise HamiltonianError("the Hamiltonian is longer than %d characters" % MAX_TEXT)
    t = re.sub(r"(?<=[\d.])E(?=[+-]?\d)", "e", t)             # 1E-3 is 1e-3
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = " ".join(ln if (i == 0 or ln[0] in "+-") else "+ " + ln for i, ln in enumerate(lines))  # a line is a term
    if re.search(r"[^0-9A-Za-z_.+\-*/()\sπ]", t):
        bad = sorted(set(re.findall(r"[^0-9A-Za-z_.+\-*/()\sπ]", t)))
        raise HamiltonianError("unexpected character(s): %s" % " ".join(bad))
    acc = {}
    raw_terms = _split_terms(t)
    if len(raw_terms) > MAX_TERMS:
        raise HamiltonianError("more than %d terms" % MAX_TERMS)
    for raw in raw_terms:
        s = raw.strip()
        if s in ("+", "-") or re.search(r"[+\-*/]\s*$", s):
            raise HamiltonianError("incomplete term '%s' (an operator without an operand)" % s)
        factors = _FACTOR.findall(s)
        if re.search(r"[IXYZ]_?\d+(?!\d)\s*[\d.]", s):
            raise HamiltonianError("in '%s' a number follows a Pauli factor; write the coefficient first "
                                   "(0.5*Z1, not Z1 0.5)" % s)
        rest = _FACTOR.sub(" @ ", s)                          # a placeholder per Pauli factor, operators kept
        if re.search(r"\*\*\s*@|@\s*\*\*", rest):
            raise HamiltonianError("'%s' raises a Pauli factor to a power; write the product out (Z0*Z0 = I)" % s)
        if re.search(r"/\s*@", rest):
            raise HamiltonianError("'%s' divides by a Pauli factor" % s)
        # each factor becomes the multiplicative identity (1); juxtaposition next to it becomes an explicit '*'
        rest = re.sub(r"(?<=[\w.)])\s*@", "*@", rest)
        rest = re.sub(r"@\s*(?=[\w.(@])", "@*", rest)
        rest = rest.replace("@", "(1)").strip()
        if re.search(r"[A-Za-z]", re.sub(r"pi|tau|sqrt|sin|cos|asin|acos|atan|e(?![A-Za-z])", "", rest)):
            raise HamiltonianError("cannot read the term '%s' (Pauli factors look like X0, Y1, Z12)" % s)
        if rest in ("", "+"):
            coef = 1.0
        elif rest == "-":
            coef = -1.0
        else:
            try:
                coef = eval_expr(rest.lstrip("+"))
            except ValueError as e:
                raise HamiltonianError("coefficient of '%s': %s" % (s, e)) from None
        phase = 1.0 + 0j
        paulis = {}
        for p, q in factors:
            q = int(q)
            if q >= n:
                raise HamiltonianError("'%s%d' acts on qubit %d, but the circuit has %d qubit%s" % (
                    p, q, q, n, "" if n == 1 else "s"))
            if p == "I":
                continue
            if q in paulis:
                ph, r = _MUL[(paulis[q], p)]
                phase *= ph
                if r == "I":
                    del paulis[q]
                else:
                    paulis[q] = r
            else:
                paulis[q] = p
        if abs(phase.imag) > 1e-12:
            raise HamiltonianError("'%s' multiplies different Paulis on one qubit, which gives an imaginary "
                                   "coefficient; write the product as a single Pauli (e.g. X0·Y0 = i·Z0)" % s)
        coef *= phase.real
        if not math.isfinite(coef):
            raise HamiltonianError("coefficient of '%s' is not finite" % s)
        if abs(coef) > MAX_COEF:
            raise HamiltonianError("coefficient of '%s' is %.3g; the limit is ±%g (rescale H instead)" % (
                s, coef, MAX_COEF))
        key = tuple(sorted(paulis.items()))
        acc[key] = acc.get(key, 0.0) + coef
    terms = tuple((float(c), k) for k, c in sorted(acc.items(), key=lambda kv: (len(kv[0]), kv[0]))
                  if abs(c) > 1e-14)
    if not terms:
        raise HamiltonianError("all terms cancel")
    label = "".join(_term_text(c, f, i == 0) for i, (c, f) in enumerate(terms))
    shown = terms[:8]
    tex = "".join(_term_tex(c, f, i == 0) for i, (c, f) in enumerate(shown))
    if len(terms) > len(shown):
        tex += r"+\dots\ (%d\ \text{terms})" % len(terms)
    short = label if len(label) <= 48 else label[:45] + "…"
    return PauliSum("H:" + label, "⟨H⟩ = ⟨%s⟩" % short, r"\langle H\rangle,\ H=%s" % tex, int(n), terms, text.strip())


def qwc_groups(terms) -> list:
    """Greedy qubit-wise commuting grouping (largest |c| first): [(basis {q: P}, [term indices])]."""
    order = sorted((i for i, (c, f) in enumerate(terms) if f), key=lambda i: (-len(terms[i][1]), -abs(terms[i][0])))
    groups = []
    for i in order:
        fac = dict(terms[i][1])
        for basis, idx in groups:
            if all(basis.get(q, p) == p for q, p in fac.items()):
                basis.update(fac)
                idx.append(i)
                break
        else:
            groups.append((dict(fac), [i]))
    return groups


def examples(n: int) -> dict:
    """Ready-made Hamiltonians for n qubits (text for the parser)."""
    out = {}
    if n >= 2:
        out["Transverse-field Ising (J = h = 1)"] = " ".join(
            ["- Z%d*Z%d" % (i, i + 1) for i in range(n - 1)] + ["- X%d" % i for i in range(n)])
        out["Heisenberg XXX chain"] = " + ".join("X%d*X%d + Y%d*Y%d + Z%d*Z%d" % (i, i + 1, i, i + 1, i, i + 1)
                                                 for i in range(n - 1))
        out["MaxCut cost on a ring"] = " + ".join("0.5 - 0.5*Z%d*Z%d" % (i, (i + 1) % n)
                                                  for i in range(n if n > 2 else 1))
    if n == 2:
        out["H₂ molecule (STO-3G, 0.735 Å, 2 qubits)"] = (
            "-1.052373245772859 + 0.39793742484318045*Z0 - 0.39793742484318045*Z1 "
            "- 0.01128010425623538*Z0*Z1 + 0.18093119978423156*X0*X1")
    out["Your mix (edit me)"] = "0.5*Z0" + ("*Z1 + 0.2*X0 - 0.1*Y1" if n >= 2 else " + 0.2*X0")
    return out


# ------------------------------------------------------------------ circuits and diagonal observables


def clbit_of(compiled, q: int) -> int:
    """Classical bit that holds logical qubit q (the app reverses the bit order when 'relabel' is on)."""
    n = compiled.logical.n
    return n - 1 - q if compiled.info.get("relabel") else q


def with_basis(compiled, basis: dict):
    """A copy of ``compiled`` whose implemented circuit ends with the basis change for ``basis`` {q: 'X'|'Y'|'Z'}."""
    import dataclasses
    impl = compiled.implemented
    native = compiled.info.get("mode") in ("isa", "device")     # stay in the native gate set {RZ, √X, …}
    extra = []
    for q, p in sorted(basis.items()):
        if p == "Z":
            continue
        phys = compiled.measure_map[clbit_of(compiled, q)]
        if native:
            seq = [("rz", (math.pi / 2,)), ("sx", ()), ("rz", (math.pi / 2,))] if p == "X" else [("sx", ())]
        else:
            seq = [("h", ())] if p == "X" else [("sdg", ()), ("h", ())]
        extra += [Op(g, (phys,), par, "gate", "basis") for g, par in seq]
    if not extra:
        return compiled
    return dataclasses.replace(compiled, implemented=impl.replace(ops=tuple(impl.ops) + tuple(extra)))


def group_observable(ps: PauliSum, compiled, idx: list) -> Observable:
    """The group's terms as a diagonal observable over classical bits."""
    n = compiled.logical.n
    x = np.arange(2 ** n)
    v = np.zeros(2 ** n)
    for i in idx:
        c, fac = ps.terms[i]
        s = np.ones(2 ** n)
        for q, _p in fac:
            s = s * (1 - 2 * ((x >> clbit_of(compiled, q)) & 1))
        v += c * s
    lab = " + ".join("%g·%s" % (ps.terms[i][0], "".join("%s%d" % (p, q) for q, p in ps.terms[i][1])) for i in idx)
    return Observable("Hg:" + lab[:60], lab[:60], "H_g", tuple(v))


def expand(ps: PauliSum, compiled) -> list:
    """[(compiled copy with basis change, diagonal observable, basis dict, term indices)] per measurement setting."""
    out = []
    for basis, idx in ps.groups():
        out.append((with_basis(compiled, basis), group_observable(ps, compiled, idx), basis, idx))
    return out


def observable_matrix(compiled, observable, n_impl: int | None = None) -> np.ndarray:
    """The observable as an operator on the implemented circuit's qubits (little-endian), *before* the final
    measurement-basis change: Tr(ρ_step M) is then "⟨O⟩ if measured after this step" (without readout error).
    Diagonal observables map classical bits to the measured qubits; Pauli sums map logical qubits the same way."""
    from .noise import PAULI1
    n_impl = compiled.implemented.n if n_impl is None else n_impl
    mm = compiled.measure_map
    if isinstance(observable, PauliSum):
        d = 2 ** n_impl
        M = np.zeros((d, d), dtype=complex)
        for c, fac in observable.terms:
            f = {mm[clbit_of(compiled, q)]: p for q, p in fac}
            m = np.array([[1.0 + 0j]])
            for q in range(n_impl):
                m = np.kron(PAULI1[f.get(q, "I")], m)
            M += c * m
        return M
    y = np.arange(2 ** n_impl)
    x = np.zeros_like(y)
    for c, p in mm.items():
        x |= ((y >> p) & 1) << c
    return np.diag(np.asarray(observable.o, float)[x]).astype(complex)


def basis_string(basis: dict, n: int) -> str:
    return "".join(basis.get(q, "·") for q in reversed(range(n)))


# ------------------------------------------------------------------ evaluation
def run_recipe(compiled, noise, observable, rc, runner, shots: int = 4096, symmetry=None, seed: int = 0):
    """One recipe on one (diagonal or Pauli-sum) observable with the simulator runner."""
    from .methods.base import REGISTRY, Context

    def make_ctx(c, o):
        return Context(c, noise, o, rc, runner, shots=shots, symmetry=symmetry, seed=seed)
    return evaluate(make_ctx, lambda ctx: REGISTRY[rc.estimator].run(ctx, **rc.p), compiled, observable, shots)


def ideal_of(compiled, observable, runner, noise=None) -> float:
    from .methods.base import Context, Recipe
    from .noise import NoiseSpec
    return ideal_value(lambda c, o: Context(c, noise or NoiseSpec(), o, Recipe.make("raw"), runner), compiled,
                       observable)


def ideal_value(make_ctx, compiled, observable) -> float:
    if not isinstance(observable, PauliSum):
        return make_ctx(compiled, observable).ideal_value()
    return observable.constant + sum(make_ctx(c, o).ideal_value() for c, o, _, _ in expand(observable, compiled))


def evaluate(make_ctx, run_method, compiled, observable, shots: int):
    """Run one recipe. ``make_ctx(compiled, diagonal observable)`` builds a context; ``run_method(ctx)`` runs the
    estimator and returns its MethodResult. A PauliSum runs once per measurement setting and is combined."""
    if not isinstance(observable, PauliSum):
        return run_method(make_ctx(compiled, observable))
    parts = []
    for c, o, basis, idx in expand(observable, compiled):
        ctx = make_ctx(c, o)
        if ctx.recipe.estimator == "symmetry":
            from .methods.symmetry import parse_checks
            qs = list(ctx.recipe.p.get("qubits") or (ctx.symmetry[0] if ctx.symmetry else ()))
            qs += [q for g in parse_checks(ctx.recipe.p.get("more", "")) for q in g]
            off = [q for q in dict.fromkeys(qs) if basis.get(q, "Z") != "Z"]
            if off:
                raise ValueError("symmetry verification post-selects on a Z-parity of q%s, but this Hamiltonian "
                                 "measures %s in the X/Y basis; use it with terms that are Z on those qubits"
                                 % (",".join(map(str, qs)), ", ".join("q%d" % q for q in off)))
        parts.append((basis, idx, run_method(ctx)))
    return combine(observable, parts, shots)


def combine(ps: PauliSum, parts: list, shots: int):
    """Combine per-setting results: sum of means, optimal shot split N_g ∝ √v_g, summed costs."""
    from .methods.base import MethodResult
    n = ps.n
    if not parts:                                      # only an identity term
        return MethodResult(ps.constant, 0.0, circuits=0, notes=["H is a multiple of the identity."])
    means = [r.mean for _, _, r in parts]
    sv = np.sqrt([max(r.var1, 0.0) for _, _, r in parts])
    w = sv / sv.sum() if sv.sum() > 0 else np.full(len(parts), 1.0 / len(parts))
    var1 = float(sv.sum() ** 2) if sv.sum() > 0 else 0.0
    alloc = [max(1, int(round(shots * wi))) for wi in w]
    first = parts[0][2]
    details = dict(first.details or {})
    details["hamiltonian"] = [dict(basis=basis_string(b, n), terms=len(idx), mean=float(r.mean), var1=float(r.var1),
                                   shots=int(a)) for (b, idx, r), a in zip(parts, alloc)]
    samplers = [r.sampler for _, _, r in parts]

    def sampler(N, B, rng):
        if any(s is None for s in samplers):
            return None
        tot = np.full(B, ps.constant, dtype=float)
        for s, wi in zip(samplers, w):
            tot = tot + np.asarray(s(max(1, int(round(N * wi))), B, rng), float)
        return tot

    notes = ["H has %d term%s measured in %d setting%s (qubit-wise commuting groups: %s); shots are split "
             "N_g ∝ √v_g (%s)." % (len(ps.terms), "" if len(ps.terms) == 1 else "s", len(parts),
                                   "" if len(parts) == 1 else "s",
                                   ", ".join(basis_string(b, n) for b, _, _ in parts),
                                   ", ".join("%.0f %%" % (100 * wi) for wi in w))]
    if len(parts) > 1:
        notes.append("The method's internal figures below show measurement setting 1 (%s)." %
                     basis_string(parts[0][0], n))
    notes += list(first.notes or [])
    for _, _, r in parts[1:]:
        for nt in (r.notes or []):
            if nt not in notes:
                notes.append(nt)
        if (r.details or {}).get("degenerate"):
            details["degenerate"] = True
            details.setdefault("warning", r.details.get("warning", ""))
        elif (r.details or {}).get("warning") and not details.get("warning"):
            details["warning"] = r.details["warning"]
    grads = [r.cal_grad for _, _, r in parts if r.cal_grad is not None]
    cal_grad = np.sum(grads, axis=0) if grads else None   # one calibration serves every setting: gradients add
    cal_theta = next((r.cal_theta for _, _, r in parts if r.cal_theta is not None), None)
    spc = [r.shots_per_circuit for _, _, r in parts if r.shots_per_circuit]
    # the statistical contract: the combination is a protocol mean if any setting's is; reference fits add up
    kinds = [getattr(r, "estimate_kind", "exact") for _, _, r in parts]
    kind = "measured" if "measured" in kinds else ("protocol" if "protocol" in kinds else kinds[0])
    refs = [r.reference_mean if r.reference_mean is not None else r.mean for _, _, r in parts]
    has_ref = any(r.reference_mean is not None for _, _, r in parts)
    moms = [r.moments for _, _, r in parts if r.moments]
    mom = None
    if moms:                                           # independent training sets: MC variances add
        mom = dict(kind=moms[0].get("kind", "protocol"), settings=len(moms),
                   mean_mcse=float(np.sqrt(sum(float(m.get("mean_mcse", 0.0)) ** 2 for m in moms))),
                   repeats=int(min(int(m.get("repeats", 0)) for m in moms)),
                   failed=int(sum(int(m.get("failed", 0)) for m in moms)),
                   conditional=bool(any(m.get("conditional") for m in moms)))
    flags = tuple(dict.fromkeys(f for _, _, r in parts for f in (r.flags or ())))
    return MethodResult(
        estimate_kind=kind, reference_mean=float(ps.constant + sum(refs)) if has_ref else None, moments=mom,
        flags=flags,
        mean=float(ps.constant + sum(means)), var1=var1, circuits=int(sum(r.circuits for _, _, r in parts)),
        fixed_shots=int(sum(r.fixed_shots for _, _, r in parts)), gamma=float(max(r.gamma for _, _, r in parts)),
        var_fixed=float(sum(r.var_fixed for _, _, r in parts)),     # independent training sets: variances add
        train_circuits=int(sum(r.train_circuits for _, _, r in parts)),
        cal_circuits=int(max(r.cal_circuits for _, _, r in parts)), cal_grad=cal_grad, cal_theta=cal_theta,
        shots_per_circuit=min(spc) if spc else None,
        extra_qubits=int(max(r.extra_qubits for _, _, r in parts)),
        accept=float(min(r.accept for _, _, r in parts)),
        implemented=list(first.implemented), details=details, notes=notes,
        sampler=sampler if all(s is not None for s in samplers) else None)
