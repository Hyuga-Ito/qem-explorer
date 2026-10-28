"""Canonical circuit identity and the study-wide ancestry registry (1.7.0; review findings R3 and R4).

A task's ``base_id`` (``task.TaskSpec.base_id``) is a digest of how its circuit is *written*: family, width and the
parameters as given. Two spellings of one circuit — ``{layers: 2}`` and ``{layers: 2, angle_seed: 0}`` (0 is the
default), or a 1.6.0 ``seed`` key that is the transpiler seed and not part of the circuit — get different base ids,
so a circuit seen in one study could be counted as "unseen" in the next.

The *logical identity* (``logical_id``) is a digest of what is executed logically, independent of spelling: the gate
sequence of the circuit body (names, qubits, angles rounded to 1e-10), the family's own input tokens (its default
input), the measurement bases and the classical bit order. Variants of a base circuit (other inputs, observables,
layouts, devices) share it. It is the unit of independence of the study: splits, independent-circuit counts and the
separation of development and confirmation data use it.

The registry (``configs/selection/ancestry_registry.json``) lists every logical circuit that any study has used, with
the study and its role:

* ``development`` — used to fit, tune or choose anything (models, features, metrics, portfolio, thresholds);
* ``exploration`` — controlled studies whose results were inspected (regimes, mismatch, focused experiments);
* ``historical-final`` — a final test already evaluated once (1.6.0); after it is seen it is no longer untouched;
* ``confirmation`` — reserved for a once-only confirmation; must overlap with nothing else;
* ``planned`` — a configuration not yet executed.

``overlap`` reports which logical circuits of a configuration already appear in the registry, by role;
``check_confirmation`` refuses a confirmation set that shares any logical circuit with another role.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date

from .. import textio as TX

SCHEMA = "qemx.selection.ancestry/1"
ROLES = ("development", "exploration", "historical-final", "confirmation", "planned")
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "configs", "selection", "ancestry_registry.json")
_CACHE: dict = {}


def logical_id(spec) -> str:
    """Canonical identity of a task's logical circuit (see module text); 16 hexadecimal digits."""
    key = (spec.family, int(spec.n), tuple(spec.params))
    if key in _CACHE:
        return _CACHE[key]
    from .task import family_circuit
    body, basis, _obs, relabel, tokens = family_circuit(spec)
    ops = [[o.name, [int(q) for q in o.qubits], [round(float(x), 10) + 0.0 for x in o.params]]
           for o in body.ops if o.kind == "gate"]
    d = dict(n=int(body.n), ops=ops, inputs=list(tokens), basis=str(basis), relabel=bool(relabel))
    out = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    _CACHE[key] = out
    return out


def describe(spec) -> dict:
    return dict(logical_id=logical_id(spec), base_id=spec.base_id, family=spec.family, n=int(spec.n),
                params=[list(kv) for kv in spec.params])


def load(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return dict(schema=SCHEMA, circuits={}, studies={})
    reg = TX.read_json(path)
    if reg.get("schema") != SCHEMA:
        raise ValueError("%s: not an ancestry registry (%s)" % (path, reg.get("schema")))
    return reg


def save(reg: dict, path: str = DEFAULT_PATH) -> None:
    TX.write_json(path, reg, indent=1, sort_keys=True)


def register(reg: dict, study: str, role: str, specs, note: str = "", post_hoc_of=()) -> dict:
    """Add a study's logical circuits with a role (idempotent: a circuit/study pair is listed once).

    ``post_hoc_of`` (1.8.0): confirmation studies whose circuits this study reuses *after* their once-only evaluation
    (e.g. an audit of their regression cases). Declared explicitly, it is recorded in the study entry; such a reuse is
    not an overlap of that confirmation (it cannot have influenced it), but it is an overlap for every later study."""
    if role not in ROLES:
        raise ValueError("role must be one of %s" % ", ".join(ROLES))
    circ = reg.setdefault("circuits", {})
    seen = {}
    for s in specs:
        d = describe(s)
        seen.setdefault(d["logical_id"], d)
    for lid, d in seen.items():
        entry = circ.setdefault(lid, dict(family=d["family"], n=d["n"], uses=[]))
        if not any(u["study"] == study for u in entry["uses"]):
            entry["uses"].append(dict(study=study, role=role, base_id=d["base_id"], params=d["params"]))
        else:
            for u in entry["uses"]:
                if u["study"] == study:
                    u["role"] = role
    reg.setdefault("studies", {})[study] = dict(role=role, logical_circuits=len(seen), note=note,
                                                registered=str(date.today()))
    if post_hoc_of:
        reg["studies"][study]["post_hoc_of"] = sorted(str(x) for x in post_hoc_of)
    return reg


def overlap(reg: dict, specs, exclude_study: str | None = None) -> dict:
    """{role: sorted logical ids} of a configuration's circuits already in the registry (other studies)."""
    out = {}
    me = str(exclude_study).split("/")[0] if exclude_study is not None else None
    studies = reg.get("studies", {})
    for s in specs:
        lid = logical_id(s)
        for u in reg.get("circuits", {}).get(lid, {}).get("uses", []):
            other = u["study"].split("/")[0]
            if me is not None and (other == me or me in studies.get(u["study"], {}).get("post_hoc_of", [])):
                continue                     # the study itself, or a declared post-hoc reuse of its circuits
            out.setdefault(u["role"], set()).add(lid)
    return {k: sorted(v) for k, v in out.items()}


def check_confirmation(reg: dict, specs, study: str) -> None:
    """Refuse a confirmation set that shares a logical circuit with any other study."""
    ov = overlap(reg, specs, exclude_study=study)
    if ov:
        raise ValueError("the confirmation set %s shares logical circuits with earlier studies: %s" % (
            study, "; ".join("%s: %d" % (k, len(v)) for k, v in sorted(ov.items()))))


def counts(specs) -> dict:
    """Tasks, written base circuits and independent logical circuits of a list of task specifications."""
    return dict(tasks=len(specs), base_circuits=len({s.base_id for s in specs}),
                logical_circuits=len({logical_id(s) for s in specs}))
