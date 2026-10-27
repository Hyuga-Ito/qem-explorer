"""The drag-and-drop composer: a Streamlit components-v2 element (``composer_assets/``) plus the Python side
that validates every action it sends (``qemx.edits``) and updates the circuit in a callback."""
from __future__ import annotations

import os

import streamlit as st

from qemx import edits as E
from qemx.circuit import GATES, NOISE_OPS
from qemx.viz import theme as TH

from . import state as S
from qemx.compile import BASIS_OPS, input_ops, input_short
from qemx.circuit import fmt_angle_tex

_DIR = os.path.join(os.path.dirname(__file__), "composer_assets")
KEY = "qx_composer"


def _read(name: str) -> str:
    with open(os.path.join(_DIR, name), encoding="utf-8") as f:
        return f.read()


_COMPONENT = {}


def component():
    """Register once per Streamlit runtime (a server has one; tests may create several)."""
    try:
        from streamlit.runtime import Runtime
        rid = id(Runtime.instance()) if Runtime.exists() else 0
    except Exception:
        rid = 0
    if rid not in _COMPONENT:
        _COMPONENT.clear()
        _COMPONENT[rid] = st.components.v2.component("qx_composer", html="", css=_read("composer.css"),
                                                     js=_read("composer.js"))
    return _COMPONENT[rid]


GATE_HELP = {"h": "Hadamard", "x": "Pauli X (NOT)", "y": "Pauli Y", "z": "Pauli Z", "s": "S = √Z", "sdg": "S†",
             "t": "T = ⁴√Z", "tdg": "T†", "sx": "√X", "sxdg": "√X†", "rx": "RX(θ)", "ry": "RY(θ)", "rz": "RZ(θ)",
             "p": "phase P(λ)", "cx": "CNOT: drop on the control wire", "cz": "controlled Z", "swap": "SWAP",
             "cp": "controlled phase", "rzz": "RZZ(θ) = exp(−iθ ZZ/2)", "ecr": "echoed cross-resonance (IBM Eagle)"}
NOISE_HELP = {"depol": "depolarizing channel, probability p", "ampdamp": "amplitude damping γ (T1-like)",
              "phasedamp": "phase damping λ (T2-like)", "bitflip": "X with probability p",
              "phaseflip": "Z with probability p", "overrot": "coherent over-rotation RX(ε)",
              "zphase": "coherent phase RZ(φ)"}


def palette() -> list:
    out = []
    for g in ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"):
        fam = "pauli" if g in ("x", "y", "z") else GATES[g][3]
        out.append(dict(name=g, label=GATES[g][2], fam=fam, group="One-qubit gates", kind="gate", help=GATE_HELP[g]))
    for g in ("rx", "ry", "rz", "p"):
        out.append(dict(name=g, label=GATES[g][2], fam=GATES[g][3], group="Rotations", kind="gate", help=GATE_HELP[g]))
    for g in ("cx", "cz", "swap", "cp", "rzz", "ecr"):
        out.append(dict(name=g, label=GATES[g][2], fam="two", group="Two-qubit gates", kind="gate", help=GATE_HELP[g]))
    for k, (nq, npar, lab) in NOISE_OPS.items():
        out.append(dict(name=k, label=lab, fam="noise", group="Noise channels", kind="noise", help=NOISE_HELP.get(k, k)))
    out.append(dict(name="barrier", label="┆", fam="box", group="Layout", kind="barrier",
                    help="barrier: aligns all qubits"))
    return out


def _on_edit():
    s = st.session_state
    try:
        val = s[KEY]
        payload = val.get("edit") if hasattr(val, "get") else getattr(val, "edit", None)
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return
    if payload.get("v") != s.get("ops_version"):
        s.composer_msg = "The circuit changed before that edit arrived; please repeat it."
        return
    kind = payload.get("type")
    s.composer_msg = ""
    try:
        if kind == "undo":
            S.undo()
        elif kind == "redo":
            S.redo()
        else:
            old = S.spec()
            new = E.apply(old, payload)
            if kind == "qubits" and new.n < old.n and len(new.ops) < len(old.ops):
                s.composer_msg = ("Removing qubit%s %s also removed %d operation%s that used %s (Undo brings them back)."
                                  % ("" if old.n - new.n == 1 else "s",
                                     ", ".join("q%d" % q for q in range(new.n, old.n)),
                                     len(old.ops) - len(new.ops), "" if len(old.ops) - len(new.ops) == 1 else "s",
                                     "it" if old.n - new.n == 1 else "them"))
            if new.n != int(s.n):
                S.set_circuit(new)
                S.push_widgets()
            else:
                S.set_circuit(new)
    except E.EditError as e:
        s.composer_msg = str(e)
    except (MemoryError, ValueError, OverflowError) as e:  # e.g. a register too large for this machine: the
        s.composer_msg = ("That edit could not be applied (%s); the circuit is unchanged." %   # model is untouched
                          (str(e)[:200] or type(e).__name__))
    except Exception as e:                                 # noqa: BLE001 - a callback must never crash the page
        s.composer_msg = "That edit could not be applied (%s: %s)." % (type(e).__name__, str(e)[:200])


_GLYPH = {"x": "X", "h": "H", "s": "S", "sdg": "S†", "y": "Y", "z": "Z"}


def io_gates(inputs, basis) -> tuple:
    """Per-qubit labels of the input preparation and of the measurement-basis change, as the composer draws them
    (locked boxes before the first column and before the measurement)."""
    prep = []
    for t in inputs:
        try:
            prep.append([_GLYPH.get(g, g.upper()) if not pr else "%s(%s)" % (g.upper(), fmt_angle_tex(pr[0]))
                         for g, pr in input_ops(t)])
        except ValueError:
            prep.append([])
    post = [[_GLYPH.get(g, g.upper()) for g in BASIS_OPS.get(b, ())] for b in basis]
    return prep, post


def render(dark: bool = False):
    s = st.session_state
    spec = S.spec()
    prep, post = io_gates(s.inputs, s.basis)
    data = dict(n=spec.n, version=int(s.ops_version), ops=S.memo(spec, "payload", lambda: E.component_payload(spec)),
                palette=palette(), colors={k: list(v) for k, v in TH.GATE_COLOR.items()},
                inputs=[input_short(x) for x in s.inputs], measure=True, prep=prep, post=post, basis=list(s.basis),
                dark=bool(dark), can_undo=bool(s.get("history")), can_redo=bool(s.get("future")),
                message=s.get("composer_msg", ""))
    component()(key=KEY, data=data, on_edit_change=_on_edit, height="content")
