"""Experimental design: task grids from a configuration file, staged acquisition and the untouched final test
(1.6.0).

A configuration (``configs/selection/*.yaml``, schema ``qemx.selection.config/1``) declares base circuits (family ×
width × parameters), the variants derived from each (inputs, observables, layouts, devices), the budget grid, the
protocols, the trials per half, the seed, and the final-test share. ``expand`` turns it into task specifications:
every base circuit gets its default variant plus ``variants_per_base`` further variants drawn without replacement
by a seeded generator — local variations that add coverage without pretending to be new circuits (they share the
base circuit's ancestry id, so splits keep them together).

The final test is chosen by a keyed hash of the base-circuit id (``final_test``), before any label exists; its tasks
are labeled into a separate directory and evaluated once, after the development choices are frozen
(``freeze_digest``).
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from . import task as TK

SCHEMA = "qemx.selection.config/1"
FIXED_INPUT = ("qft", "bv", "grover", "ghz")


class ConfigError(ValueError):
    """A configuration that does not satisfy the published schema or names an unknown parameter."""


def validate(cfg: dict, where: str = "configuration") -> dict:
    """Check a configuration against the published schema (``schemas.CONFIG``: unknown keys and family parameters are
    refused) and every device specification and design plan against their parsers; returns it unchanged."""
    import jsonschema
    from . import schemas as SC
    from .emulate import DesignPlan
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
        raise ConfigError("%s: schema must be %s" % (where, SCHEMA))
    errs = sorted(jsonschema.Draft202012Validator(SC.CONFIG).iter_errors(cfg), key=lambda e: list(e.absolute_path))
    if errs:
        raise ConfigError("%s does not satisfy %s:\n%s" % (where, SCHEMA, "\n".join(
            "  at %s: %s" % ("/".join(str(x) for x in e.absolute_path) or "(top)", e.message[:300]) for e in errs[:12])))
    try:
        DesignPlan.from_config(cfg.get("learning_designs", 1))
    except ValueError as e:
        raise ConfigError("%s: %s" % (where, e)) from None
    from .. import device
    blocks = cfg["tasks"] if isinstance(cfg["tasks"], list) else [cfg["tasks"]]
    for b in blocks:
        for d in b.get("devices", ["fake_fez"]):
            try:
                from .task import with_modifiers
                base, _ = device.parse_device(d if "eng=" in d else with_modifiers(d, "eng=exact"))
            except ValueError as e:
                raise ConfigError("%s: device %r: %s" % (where, d, e)) from None
            if base not in device.FAKE_DEVICES:
                raise ConfigError("%s: unknown fake device %r" % (where, base))
    return cfg


def load(path: str) -> dict:
    """A configuration file (UTF-8, a byte-order mark accepted), validated (``validate``)."""
    from .. import textio as TX
    cfg = TX.read_yaml(path)
    return validate(cfg, path)


def _base_list(cfg: dict) -> list:
    out = []
    t = cfg["tasks"]
    if isinstance(t, list):                                  # several blocks (see expand)
        for b in t:
            out += _base_list(dict(cfg, tasks=b))
        return out
    for fam, plist in t["families"].items():
        widths = t.get("widths_by_family", {}).get(fam, t["widths"])
        for n in widths:
            for p in plist or [{}]:
                out.append((fam, int(n), dict(p or {})))
    return out


def expand(cfg: dict) -> list:
    """Task specifications of a configuration (development and final test together; see ``final_test``).

    ``tasks`` is one block or a list of blocks (each with its own families, widths and variant lists; a controlled
    study lists one block per controlled factor). ``variants_per_base: all`` takes every variant of the block — the
    full factorial of its inputs, observables, layouts and devices — instead of a seeded sample."""
    t = cfg["tasks"]
    if isinstance(t, list):
        seen, out = set(), []
        for b in t:
            for s in expand(dict(cfg, tasks=b)):
                if s.task_id not in seen:
                    seen.add(s.task_id)
                    out.append(s)
        return out
    rng = np.random.default_rng(int(cfg.get("master_seed", 0)) + 101)
    specs = []
    for fam, n, p in _base_list(cfg):
        inputs = ["default"] if fam in FIXED_INPUT else list(t.get("inputs", ["default"]))
        variants = list(itertools.product(inputs, t.get("observables", ["default"]), t.get("layouts", ["best"]),
                                          t.get("devices", ["fake_fez"])))
        default = ("default", "default", "best", t.get("devices", ["fake_fez"])[0])
        rest = [v for v in variants if v != default]
        if str(t.get("variants_per_base", 0)) == "all":
            pick = [default] + rest
        else:
            k = min(int(t.get("variants_per_base", 0)), len(rest))
            pick = [default] + ([rest[i] for i in rng.choice(len(rest), size=k, replace=False)] if k else [])
        for inp, obs, lay, dev in pick:          # an invalid case is an error of the configuration, never skipped
            specs.append(TK.make_spec(fam, n, inputs=inp, observable=obs, device=dev, layout=lay,
                                      truth=str(cfg.get("truth", "")), **p))
    seen, out = set(), []
    for s in specs:
        if s.task_id not in seen:
            seen.add(s.task_id)
            out.append(s)
    return out


def final_test(specs: list, fraction: float, salt: str = "qemx-1.6-final") -> tuple:
    """(development, final) split of task specs by a keyed hash of the base-circuit id: deterministic, fixed before
    labeling, and independent of any outcome."""
    dev, fin = [], []
    for s in specs:
        h = int(hashlib.sha256(("%s|%s" % (salt, s.base_id)).encode()).hexdigest()[:8], 16) / 16 ** 8
        (fin if h < float(fraction) else dev).append(s)
    return dev, fin


LABEL_KEYS = ("schema", "master_seed", "R", "budgets", "learning_designs", "protocols", "final_test_fraction", "truth",
              "tasks", "mismatch")


def label_part(cfg: dict) -> dict:
    """The part of a configuration that determines the tasks and their labels (not the analysis, limits or
    descriptive fields): a labels directory may be analyzed again with another analysis section."""
    return {k: cfg[k] for k in LABEL_KEYS if k in cfg}


def freeze_digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def design_table(specs: list) -> list:
    """Rows of the experimental-design table (Table 1): counts by family, width, device and variant kind."""
    rows = {}
    for s in specs:
        key = (s.family, s.n, s.device)
        r = rows.setdefault(key, dict(family=s.family, n=s.n, device=s.device, tasks=0, base_circuits=set(),
                                      inputs=set(), observables=set(), layouts=set()))
        r["tasks"] += 1
        r["base_circuits"].add(s.base_id)
        r["inputs"].add(s.inputs)
        r["observables"].add(s.observable)
        r["layouts"].add(s.layout)
    out = []
    for r in rows.values():
        out.append(dict(r, base_circuits=len(r["base_circuits"]), inputs=",".join(sorted(r["inputs"])),
                        observables=",".join(sorted(r["observables"])), layouts=",".join(sorted(r["layouts"]))))
    return sorted(out, key=lambda d: (d["family"], d["n"], d["device"]))
