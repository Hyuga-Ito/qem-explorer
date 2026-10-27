r"""Sweeps and register-wide settings: inputs, measurement bases, template parameters and noise parameters.

One line per swept quantity, ``name: value, value, …`` (lines may also be separated by ``;``)::

    input: plus, ramp:1, ramp:2         register inputs (see below)
    ramp: 0, 1, 3                       shorthand for input: ramp:0, ramp:1, ramp:3
    basis: Z, X                         one letter for every qubit, or one letter per qubit (q0 first)
    m: 1, 2, 3                          any parameter of the selected templates (AQFT degree m here)
    p2: 0.005, 0.01, 0.02               any noise parameter; replaces that rate in every noise model

Several lines give every combination. **Register inputs** (all product states, prepared before the circuit):

* ``0``, ``1``, ``+``, ``-``, ``+i``, ``-i`` (also ``plus``, ``minus``) on every qubit;
* ``ramp:k`` — the phase ramp :math:`|{\rm ramp}{:}k\rangle=F^\dagger|k\rangle=\frac1{\sqrt N}\sum_j
  e^{-2\pi ijk/N}|j\rangle`, i.e. qubit q in :math:`(|0\rangle+e^{-2\pi i k2^q/N}|1\rangle)/\sqrt2`; ``ramp:0`` is
  :math:`|{+}\cdots{+}\rangle`, and the exact QFT maps ``ramp:k`` to :math:`|k\rangle`;
* ``bits:0101`` — a basis state written like the app's bitstrings (q_{n-1} … q_0);
* ``ph:t`` (a phase in turns) on every qubit, and — in the Playground's register-wide box, where commas do not separate
  values — ``bloch:θ,φ`` on every qubit or one state per qubit separated by spaces, q0 first (``0 + 1``).

A register input *replaces* the input, including a template's own one (the QFT's Fourier state, BV's ancilla |−⟩):
the input is prepared once (qemx/io_semantics.py).

Parsing is strict: names and values are checked against a whitelist, numbers go through the app's safe expression
reader (never ``eval``), and the text is limited in size. Nothing is executed.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from . import templates as T
from .circuit import eval_expr
from .compile import INPUT_STATES, phase_token, valid_input
from .noise import NoiseSpec

MAX_TEXT = 2000
MAX_AXES = 6
MAX_VALUES = 64
MAX_VARIANTS = 5000                  # the product of all axes; a guard against a typo, not a research limit
NOISE_FIELDS = tuple(f for f in NoiseSpec.__dataclass_fields__ if f != "field_samples")
PROBABILITY = {"p1", "p2", "amp_damp", "phase_damp", "idle_dephase", "idle_damp", "ro01", "ro10"}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
_VALUE = re.compile(r"^[A-Za-z0-9_.:+\-*/() πτ]{1,60}$")
_ALIASES = {"plus": "+", "minus": "-", "+1": "+", "-1": "-"}


class SweepError(ValueError):
    pass


@dataclass(frozen=True)
class Axis:
    name: str            # input | basis | a template parameter | a noise parameter
    kind: str            # input | basis | param | noise
    values: tuple


# ------------------------------------------------------------------ register inputs and bases
def input_tokens(value: str, n: int) -> list:
    """Per-qubit input tokens for a register input (see the module docstring)."""
    v = str(value).strip()
    v = _ALIASES.get(v.lower(), v)
    if v in INPUT_STATES or valid_input(v):              # one state on every qubit (also ph:t, bloch:θ,φ)
        return [v] * n
    parts = v.split()
    if len(parts) > 1:                                   # one state per qubit, q0 first
        toks = [_ALIASES.get(x.lower(), x) for x in parts]
        bad = [x for x in toks if not valid_input(x)]
        if bad:
            raise SweepError("unknown input %r in the per-qubit list" % bad[0][:30])
        if len(toks) != n:
            raise SweepError("%d per-qubit inputs for %d qubits (list one state per qubit, q0 first)" % (len(toks), n))
        return toks
    m = re.fullmatch(r"ramp:(-?\d{1,9})", v.replace(" ", ""), flags=re.I | re.ASCII)
    if m:
        k = int(m.group(1))
        return [phase_token(-k * 2 ** q / 2 ** n) for q in range(n)]
    m = re.fullmatch(r"bits:([01]{1,64})", v.replace(" ", ""), flags=re.I)
    if m:
        b = m.group(1)
        if len(b) != n:
            raise SweepError("bits:%s has %d bits, the circuit has %d qubits" % (b, len(b), n))
        return ["1" if b[n - 1 - q] == "1" else "0" for q in range(n)]
    raise SweepError("unknown input %r: use 0, 1, +, -, +i, -i, plus, minus, ph:t, bloch:θ,φ, ramp:k, bits:0101 or "
                     "one state per qubit separated by spaces" % v[:30])


def basis_string(value: str, n: int) -> str:
    """'X' → X on every qubit; 'ZZXY' → one letter per qubit, q0 first."""
    v = str(value).strip().upper()
    if not re.fullmatch(r"[XYZ]{1,64}", v):
        raise SweepError("unknown basis %r: use X, Y, Z or one letter per qubit (q0 first)" % str(value)[:30])
    if len(v) == 1:
        return v * n
    if len(v) != n:
        raise SweepError("basis %s has %d letters, the circuit has %d qubits" % (v, len(v), n))
    return v


def register_label(value: str) -> str:
    v = str(value).strip()
    return {"+": "plus", "-": "minus"}.get(_ALIASES.get(v.lower(), v), v)


# ------------------------------------------------------------------ parsing
def template_params(keys) -> dict:
    """{parameter name: TParam} over the given template keys (first definition wins)."""
    out = {}
    for k in keys:
        if k in T.REGISTRY:
            for p in T.REGISTRY[k][2]:
                out.setdefault(p.name, p)
    return out


def _number(txt: str) -> float:
    try:
        return float(eval_expr(txt))
    except Exception as e:                                   # noqa: BLE001 - reported as a readable message
        raise SweepError("%r is not a number (%s)" % (txt[:30], str(e)[:80]))


def parse(text: str, template_keys=()) -> list:
    """Axes from the sweep text. Raises SweepError with a readable message."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) > MAX_TEXT:
        raise SweepError("the sweep text is longer than %d characters" % MAX_TEXT)
    tparams = template_params(template_keys)
    axes, seen = [], set()
    for raw in re.split(r"[\n;]+", text):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise SweepError("%r: write name: value, value, …" % line[:40])
        name, vals = line.split(":", 1)
        name = name.strip()
        if not _NAME.match(name):
            raise SweepError("%r is not a parameter name" % name[:32])
        items = [v.strip() for v in vals.split(",") if v.strip()]
        if not items:
            raise SweepError("%s: no values" % name)
        if len(items) > MAX_VALUES:
            raise SweepError("%s: more than %d values" % (name, MAX_VALUES))
        for v in items:
            if not _VALUE.match(v):
                raise SweepError("%s: %r contains characters that are not allowed" % (name, v[:30]))
        low = name.lower()
        if low in ("input", "inputs"):
            for v in items:
                input_tokens(v, 8) if not v.lower().startswith("bits:") else None
            ax = Axis("input", "input", tuple(register_label(v) for v in items))
        elif low == "ramp":
            ks = []
            for v in items:
                k = _number(v)
                if k != int(k):
                    raise SweepError("ramp: %r is not an integer" % v)
                ks.append("ramp:%d" % int(k))
            ax = Axis("input", "input", tuple(ks))
        elif low in ("basis", "bases"):
            for v in items:
                basis_string(v, len(v) if len(v) > 1 else 1)
            ax = Axis("basis", "basis", tuple(v.upper() for v in items))
        elif name in NOISE_FIELDS:
            nums = tuple(_number(v) for v in items)
            if name in PROBABILITY and any(not 0 <= x <= 1 for x in nums):
                raise SweepError("%s is a probability: every value must lie in [0, 1]" % name)
            ax = Axis(name, "noise", nums)
        elif name in tparams:
            p = tparams[name]
            if p.kind == "choice":
                opts = {str(o).lower(): o for o in p.options}
                vals_ = []
                for v in items:
                    o = opts.get(T.LEGACY_CHOICES.get(v, v).lower())
                    if o is None:
                        raise SweepError("%s: %r is not one of %s" % (name, v[:30], ", ".join(map(str, p.options))))
                    vals_.append(o)
                ax = Axis(name, "param", tuple(vals_))
            else:
                nums = [_number(v) for v in items]
                ax = Axis(name, "param", tuple(int(x) if p.kind == "int" else x for x in nums))
        else:
            known = sorted(set(tparams) | set(NOISE_FIELDS))
            raise SweepError("%r is neither input, ramp, basis, a parameter of the selected templates nor a noise "
                             "parameter (%s)" % (name, ", ".join(known)[:300]))
        if ax.name in seen:
            raise SweepError("%s appears twice" % ax.name)
        seen.add(ax.name)
        axes.append(ax)
        if len(axes) > MAX_AXES:
            raise SweepError("at most %d swept quantities" % MAX_AXES)
    return axes


# ------------------------------------------------------------------ variants
def applicable(axes, source: str, device: bool = False) -> list:
    """The axes that apply to one algorithm: template parameters only to templates that have them, noise
    parameters not to fake devices (their noise is the calibration)."""
    own = {p.name for p in T.REGISTRY[source][2]} if source in T.REGISTRY else set()
    return [a for a in axes if a.kind in ("input", "basis") or (a.kind == "param" and a.name in own)
            or (a.kind == "noise" and not device)]


def variants(axes) -> list:
    """Every combination as a tuple of (name, value) pairs; [()] when nothing is swept."""
    if not axes:
        return [()]
    total = 1
    for a in axes:
        total *= len(a.values)
    if total > MAX_VARIANTS:
        raise SweepError("%d combinations; split the sweep (at most %d per batch)" % (total, MAX_VARIANTS))
    return [tuple(zip([a.name for a in axes], combo)) for combo in itertools.product(*[a.values for a in axes])]


def label(variant) -> str:
    return " · ".join("%s" % v if k == "input" else "%s=%s" % (k, _fmt(v)) for k, v in variant)


def _fmt(v) -> str:
    return "%g" % v if isinstance(v, float) else str(v)


def split(variant, kinds: dict) -> tuple:
    """(template params, input, basis, noise overrides) of a variant; kinds maps axis name → kind."""
    params, noise, inp, basis = {}, {}, None, None
    for k, v in variant:
        kind = kinds.get(k)
        if kind == "param":
            params[k] = v
        elif kind == "noise":
            noise[k] = float(v)
        elif kind == "input":
            inp = v
        elif kind == "basis":
            basis = v
    return params, inp, basis, noise
