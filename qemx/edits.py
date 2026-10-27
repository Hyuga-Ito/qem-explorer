"""Editing actions for the drag-and-drop composer.

The browser component never sends a circuit; it sends *actions* ("add H on qubit 1 at column 3",
"move op 5 to qubit 0, column 2", "set the parameter of op 4 to pi/8", ...). Every action is
validated here against the current circuit, so a stale or malformed message can only be rejected,
never produce an invalid circuit or run code. Parameters go through the whitelist expression
reader of :mod:`qemx.circuit`.

Columns follow the Composer: operations are left-aligned (as-soon-as-possible), and dropping at a
column means "after everything in earlier columns, before everything from this column on".
"""
from __future__ import annotations

import math

from .circuit import GATES, NOISE_OPS, CircuitSpec, Op, eval_expr

DEFAULT_PARAM = {"rx": math.pi / 2, "ry": math.pi / 2, "rz": math.pi / 4, "p": math.pi / 4, "cp": math.pi / 2,
                 "rzz": math.pi / 4}
DEFAULT_NOISE = {"depol": 0.05, "ampdamp": 0.05, "phasedamp": 0.05, "bitflip": 0.05, "phaseflip": 0.05,
                 "overrot": 0.1, "zphase": 0.1}
ACTIONS = ("add", "move", "retarget", "delete", "param", "flip", "duplicate", "clear", "qubits", "undo", "redo")


class EditError(ValueError):
    pass


def _int(v, name, lo=0, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise EditError("%s must be an integer" % name)
    if isinstance(v, float):
        if not v.is_integer():
            raise EditError("%s must be an integer" % name)
        v = int(v)
    if v < lo or (hi is not None and v > hi):
        raise EditError("%s out of range" % name)
    return v


def display_columns(ops: list, n: int) -> list:
    """Left-aligned column of each op (same rule as the drawing): an op occupies its qubit span."""
    level = [0] * n
    cols = []
    for o in ops:
        if o.kind == "barrier":
            qs = o.qubits or tuple(range(n))
            m = max(level[q] for q in qs)
            for q in qs:
                level[q] = m
            cols.append(m)
            continue
        lo, hi = min(o.qubits), max(o.qubits)
        c = max(level[q] for q in range(lo, hi + 1))
        cols.append(c)
        for q in range(lo, hi + 1):
            level[q] = c + 1
    return cols


def normalize(ops: list, n: int) -> list:
    """Sort by display column (stable): a valid execution order in which column order = list order."""
    cols = display_columns(ops, n)
    order = sorted(range(len(ops)), key=lambda i: (cols[i], i))
    return [ops[i] for i in order]


def _insert_at_column(ops: list, n: int, op: Op, col: int) -> list:
    # list order = column order, so "k ops before" means "everything in earlier columns"
    base = normalize(ops, n)
    cols = display_columns(base, n)
    k = sum(1 for c in cols if c < col)
    return base[:k] + [op] + base[k:]


def new_op(name: str, q: int, n: int, params=None) -> Op:
    if name == "barrier":
        return Op("barrier", (), (), "barrier")
    if name in GATES:
        nq, npar, *_ = GATES[name]
        if nq == 2:
            if n < 2:
                raise EditError("a two-qubit gate needs at least two qubits")
            q2 = q + 1 if q + 1 < n else q - 1
            qs = (q, q2)
        else:
            qs = (q,)
        ps = tuple(params) if params is not None else ((DEFAULT_PARAM.get(name, math.pi / 4),) if npar else ())
        return Op(name, qs, ps, "gate")
    if name in NOISE_OPS:
        ps = tuple(params) if params is not None else (DEFAULT_NOISE[name],)
        return Op(name, (q,), ps, "noise")
    raise EditError("unknown operation '%s'" % name)


def _param_value(name: str, text) -> float:
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        v = float(text)
        if not math.isfinite(v):
            raise EditError("not a finite number")
    elif isinstance(text, str):
        try:
            v = eval_expr(text)
        except ValueError as e:
            raise EditError(str(e)) from None
    else:
        raise EditError("the parameter must be a number or an expression like pi/4")
    if name in NOISE_OPS and name not in ("overrot", "zphase") and not 0 <= v <= 1:
        raise EditError("a probability must be between 0 and 1")
    return v


def apply(spec: CircuitSpec, action: dict) -> CircuitSpec:
    """The circuit after one validated action (undo/redo are handled by the caller's history)."""
    if not isinstance(action, dict):
        raise EditError("malformed action")
    kind = action.get("type")
    if kind not in ACTIONS:
        raise EditError("unknown action")
    n = spec.n
    ops = list(spec.ops)

    def idx():
        return _int(action.get("idx"), "operation index", 0, len(ops) - 1)

    if kind == "add":
        name = action.get("name")
        if not isinstance(name, str) or not (name in GATES or name in NOISE_OPS or name == "barrier"):
            raise EditError("unknown operation")
        q = _int(action.get("q"), "qubit", 0, n - 1)
        col = _int(action.get("col", 10 ** 9), "column", 0)
        op = new_op(name, q, n)
        ops = _insert_at_column(ops, n, op, col)
    elif kind == "move":
        i = idx()
        o = ops[i]
        q = _int(action.get("q"), "qubit", 0, n - 1)
        col = _int(action.get("col", 10 ** 9), "column", 0)
        if o.kind == "barrier":
            moved = o
        elif o.k == 1:
            moved = Op(o.name, (q,), o.params, o.kind, o.tag)
        else:
            shift = q - o.qubits[0]
            qs = tuple(x + shift for x in o.qubits)
            if min(qs) < 0 or max(qs) >= n:                 # keep the gate inside the register
                off = -min(qs) if min(qs) < 0 else (n - 1 - max(qs))
                qs = tuple(x + off for x in qs)
            moved = Op(o.name, qs, o.params, o.kind, o.tag)
        rest = ops[:i] + ops[i + 1:]
        ops = _insert_at_column(rest, n, moved, col)
    elif kind == "retarget":
        i = idx()
        o = ops[i]
        which = _int(action.get("which", 1), "end", 0, 1)
        q = _int(action.get("q"), "qubit", 0, n - 1)
        if o.kind != "gate" or o.k != 2:
            raise EditError("only two-qubit gates have a second end")
        qs = list(o.qubits)
        other = qs[1 - which]
        if q == other:
            raise EditError("both ends of a two-qubit gate cannot sit on the same qubit")
        qs[which] = q
        ops[i] = Op(o.name, tuple(qs), o.params, o.kind, o.tag)
    elif kind == "delete":
        i = idx()
        ops.pop(i)
    elif kind == "param":
        i = idx()
        o = ops[i]
        npar = GATES[o.name][1] if o.kind == "gate" else (NOISE_OPS[o.name][1] if o.kind == "noise" else 0)
        if not npar:
            raise EditError("'%s' has no parameter" % o.name)
        ops[i] = Op(o.name, o.qubits, (_param_value(o.name, action.get("text")),), o.kind, o.tag)
    elif kind == "flip":
        i = idx()
        o = ops[i]
        if o.k != 2:
            raise EditError("only two-qubit operations can be flipped")
        ops[i] = Op(o.name, tuple(reversed(o.qubits)), o.params, o.kind, o.tag)
    elif kind == "duplicate":
        i = idx()
        ops.insert(i + 1, ops[i])
    elif kind == "clear":
        ops = []
    elif kind == "qubits":
        m = _int(action.get("n"), "number of qubits", 1)
        ops = [o for o in ops if not o.qubits or max(o.qubits) < m]
        return CircuitSpec(m, tuple(normalize(ops, m)), spec.name)
    else:
        raise EditError("'%s' is handled by the history" % kind)
    return CircuitSpec(n, tuple(normalize(ops, n)), spec.name)


def component_payload(spec: CircuitSpec) -> list:
    """What the browser needs to draw: one dict per op with its display column."""
    from .circuit import fmt_angle_tex
    cols = display_columns(list(spec.ops), spec.n)
    out = []
    for i, (o, c) in enumerate(zip(spec.ops, cols)):
        if o.kind == "gate":
            lab = GATES[o.name][2]
            fam = GATES[o.name][3] if o.name not in ("x", "y", "z", "id") else "pauli"
            par = ", ".join(fmt_angle_tex(p) for p in o.params)
            ptext = ", ".join("%.12g" % p for p in o.params)
        elif o.kind == "noise":
            lab, fam = NOISE_OPS[o.name][2], "noise"
            par = ", ".join("%.3g" % p for p in o.params)
            ptext = par
        else:
            lab, fam, par, ptext = "", "barrier", "", ""
        out.append(dict(i=i, name=o.name, kind=o.kind, q=list(o.qubits), col=int(c), label=lab, fam=fam, par=par,
                        ptext=ptext, tag=o.tag, npar=len(o.params)))
    return out
