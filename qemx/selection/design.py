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
    full factorial of its inputs, observables, layouts and devices — instead of a seeded sample.

    Semantics kept from 1.6–1.8 (historical task identities must not change; ``expansion_report`` makes them visible):
    a block adds its *default variant* (default input, default observable, best layout, first device) to every base
    circuit unless ``include_default: false``, even when the lists name only other values; ``variants_per_base: k``
    then samples k further variants without replacement (seeded by master_seed + 101 per block); families with a
    family-defined input (qft, bv, grover, ghz) always use it, whatever ``inputs`` lists; a task produced by two blocks
    is kept once (first block)."""
    return [s for s, _ in expand_with_provenance(cfg)]


def expand_with_provenance(cfg: dict) -> list:
    """``expand`` with, per task, where it came from: block index, whether it is the inserted default variant,
    whether the family forced its own input, and whether the variant was sampled or taken from the full grid
    (1.9.0; second review N6)."""
    t = cfg["tasks"]
    if isinstance(t, list):
        seen, out = set(), []
        for bi, b in enumerate(t):
            for s, info in expand_with_provenance(dict(cfg, tasks=b)):
                if s.task_id not in seen:
                    seen.add(s.task_id)
                    out.append((s, dict(info, block=bi)))
        return out
    rng = np.random.default_rng(int(cfg.get("master_seed", 0)) + 101)
    specs = []
    for fam, n, p in _base_list(cfg):
        forced = fam in FIXED_INPUT
        inputs = ["default"] if forced else list(t.get("inputs", ["default"]))
        variants = list(itertools.product(inputs, t.get("observables", ["default"]), t.get("layouts", ["best"]),
                                          t.get("devices", ["fake_fez"])))
        default = ("default", "default", "best", t.get("devices", ["fake_fez"])[0])
        if t.get("include_default", True):
            rest = [v for v in variants if v != default]
            first = [default]
        else:                       # 1.8.0: a block whose variants exclude the default (e.g. an informative observable)
            rest, first = list(variants), []
        if str(t.get("variants_per_base", 0)) == "all":
            pick = first + rest
            how = "full grid"
        else:
            k = min(int(t.get("variants_per_base", 0)), len(rest))
            pick = first + ([rest[i] for i in rng.choice(len(rest), size=k, replace=False)] if k else [])
            how = "sampled %d of %d" % (k, len(rest))
        listed = (tuple(t.get("inputs", ["default"])), tuple(t.get("observables", ["default"])),
                  tuple(t.get("layouts", ["best"])), tuple(t.get("devices", ["fake_fez"])))
        for v in pick:               # an invalid case is an error of the configuration, never skipped
            inp, obs, lay, dev = v
            ins = bool(first) and v == default
            specs.append((TK.make_spec(fam, n, inputs=inp, observable=obs, device=dev, layout=lay,
                                       truth=str(cfg.get("truth", "")), **p),
                          dict(block=0, inserted_default=ins and not (obs in listed[1] and lay in listed[2]
                                                                       and (inp in listed[0] or forced)),
                               default_variant=ins, forced_input=forced and listed[0] != ("default",),
                               selection="default variant" if ins else how)))
    seen, out = set(), []
    for s, info in specs:
        if s.task_id not in seen:
            seen.add(s.task_id)
            out.append((s, info))
    return out


def expansion_report(cfg: dict) -> dict:
    """What the configuration expands to and which implicit rules acted (1.9.0; second review N6): per block, the
    requested lists, the tasks, the inserted default variants (a value the lists do not name), the families whose
    own input replaced the listed inputs, sampled versus full-grid selection, and duplicates across blocks; totals
    per device, noise modifier set, family, input and observable."""
    blocks = cfg["tasks"] if isinstance(cfg["tasks"], list) else [cfg["tasks"]]
    rows = expand_with_provenance(cfg)
    total_raw = 0
    per_block = []
    for bi, b in enumerate(blocks):
        raw = expand_with_provenance(dict(cfg, tasks=b))
        total_raw += len(raw)
        mine = [(s, i) for s, i in rows if i.get("block", 0) == bi]
        per_block.append(dict(
            block=bi, families=sorted(b["families"]), widths=b.get("widths"),
            inputs=b.get("inputs", ["default"]), observables=b.get("observables", ["default"]),
            layouts=b.get("layouts", ["best"]), devices=b.get("devices", ["fake_fez"]),
            include_default=b.get("include_default", True), variants_per_base=b.get("variants_per_base", 0),
            tasks=len(mine), duplicates_of_earlier_blocks=len(raw) - len(mine),
            inserted_defaults=sum(1 for _, i in mine if i["inserted_default"]),
            forced_input_families=sorted({s.family for s, i in mine if i["forced_input"]})))
    specs = [s for s, _ in rows]

    def count(f):
        out = {}
        for s in specs:
            k = f(s)
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))
    notes = []
    for b in per_block:
        if b["inserted_defaults"]:
            notes.append("block %d: %d default variant(s) (default input, default observable, best layout, first "
                         "device) were added although the lists do not name them (set include_default: false to "
                         "omit them)" % (b["block"], b["inserted_defaults"]))
        if b["forced_input_families"]:
            notes.append("block %d: %s use their family-defined input; the listed inputs %s do not apply to them"
                         % (b["block"], ", ".join(b["forced_input_families"]), b["inputs"]))
        if b["duplicates_of_earlier_blocks"]:
            notes.append("block %d: %d task(s) already produced by an earlier block were kept once"
                         % (b["block"], b["duplicates_of_earlier_blocks"]))
        if str(b["variants_per_base"]) != "all":
            notes.append("block %d: variants sampled (variants_per_base: %s), not the full grid"
                         % (b["block"], b["variants_per_base"]))
    from . import ancestry as AY
    dev, fin = final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    return dict(policy=cfg.get("expansion_policy", "legacy"), tasks=len(specs), base_circuits=len({s.base_id for s in specs}),
                logical_circuits=len({AY.logical_id(s) for s in specs}), development=len(dev), final=len(fin),
                blocks=per_block, notes=notes,
                by_device=count(lambda s: s.device.split("@")[0]),
                by_noise=count(lambda s: s.device.split("@", 1)[1] if "@" in s.device else "native"),
                by_family=count(lambda s: s.family), by_input=count(lambda s: s.inputs),
                by_observable=count(lambda s: s.observable), by_width=count(lambda s: s.n),
                summary="%d tasks, %d base circuits, %d logical circuits (%d development, %d final); %d block(s)%s" % (
                    len(specs), len({s.base_id for s in specs}), len({AY.logical_id(s) for s in specs}), len(dev),
                    len(fin), len(blocks), "" if total_raw == len(specs) else
                    "; %d duplicate task(s) across blocks kept once" % (total_raw - len(specs))))


def write_task_table(cfg: dict, path: str) -> int:
    """The expanded task table (CSV, UTF-8): every task with its identities, split, family, width, parameters, input,
    observable, layout, device and noise modifiers, and how it was produced. Returns the number of rows."""
    import csv
    import os
    from . import ancestry as AY
    rows = expand_with_provenance(cfg)
    dev, fin = final_test([s for s, _ in rows], float(cfg.get("final_test_fraction", 0.0)))
    fin_ids = {s.task_id for s in fin}
    cols = ["task_id", "base_id", "logical_id", "split", "block", "family", "n", "params", "inputs", "observable",
            "layout", "device", "device_base", "noise_modifiers", "truth", "selection", "inserted_default",
            "forced_input"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s, i in rows:
            w.writerow(dict(task_id=s.task_id, base_id=s.base_id, logical_id=AY.logical_id(s),
                            split="final" if s.task_id in fin_ids else "development", block=i.get("block", 0),
                            family=s.family, n=s.n, params=json.dumps(dict(s.params), sort_keys=True),
                            inputs=s.inputs, observable=s.observable, layout=s.layout, device=s.device,
                            device_base=s.device.split("@")[0],
                            noise_modifiers=s.device.split("@", 1)[1] if "@" in s.device else "",
                            truth=s.truth, selection=i["selection"], inserted_default=i["inserted_default"],
                            forced_input=i["forced_input"]))
    return len(rows)


def check(cfg: dict) -> dict:
    """Semantic checks beyond the schema, before anything expensive (1.9.0; second review N6): {errors, warnings}.

    Errors: unknown feature sets, scenarios or split options that cannot be met; more fresh designs per half than
    trials per half (designs without trials); a development set without base circuits; a held-out device or width
    that leaves one side empty; in an ``expansion_policy: explicit`` configuration, a block that leaves
    ``include_default`` or ``variants_per_base`` implicit or lists inputs for a family that forces its own.
    Warnings: fewer development base circuits than folds (k is reduced), a design plan without learning candidates,
    descriptive keys the code does not read."""
    from . import features as FE
    from . import protocols as PR
    from . import schemas as SC
    from .emulate import DesignPlan
    errs, warns = [], []
    a = cfg.get("analysis", {}) or {}
    for key in ("feature_sets", "other_split_sets"):
        for n in a.get(key, []) or []:
            if n not in FE.SETS:
                errs.append("analysis.%s: unknown feature set %r (known: %s)" % (key, n, ", ".join(sorted(FE.SETS))))
    for key in ("abstain_set", "ablation_set", "deployed_set"):
        if key in a and a[key] not in FE.SETS:
            errs.append("analysis.%s: unknown feature set %r" % (key, a[key]))
    for fam in set(a.get("model_families", []) or []) | set(a.get("other_split_families", []) or []):
        if fam == "struct":
            for n in (a.get("feature_sets", []) or []):
                if n in FE.SETS and "log10_budget" not in FE.SETS[n]:
                    warns.append("analysis: the structured model needs log10_budget; feature set %r has none" % n)
    for k in (a.get("split_options", {}) or {}):
        if k not in (a.get("splits", ["same-distribution"]) or []):
            warns.append("analysis.split_options.%s: options for a split that is not in analysis.splits" % k)
    ld = cfg.get("learning_designs")
    if isinstance(ld, dict):                           # 1.9.0: keys that do not belong to the chosen mode are errors
        mode = ld.get("mode", "fresh")
        if mode == "fresh" and "count" in ld:
            errs.append("learning_designs.count belongs to mode fixed-shared; fresh designs use per_half")
        if mode == "fixed-shared" and ({"per_half", "crn"} & set(ld)):
            errs.append("learning_designs.%s belongs to mode fresh" % sorted({"per_half", "crn"} & set(ld))[0])
    if "analysis" in cfg and "splits" in a and not (a.get("splits") or []):    # omitted = same-distribution
        warns.append("analysis.splits is empty: no model comparison is made, and a freeze needs the same-distribution "
                     "split")
    plan = DesignPlan.from_config(cfg.get("learning_designs", 1))
    R = int(cfg.get("R", 200))
    protos = list(cfg.get("protocols", []))
    n_learn = sum(1 for p in protos if PR.PORTFOLIO[p].learning)
    if plan.mode == "fresh" and plan.per_half > R:
        errs.append("learning_designs.per_half = %d > R = %d: each half has R trials, so %d designs per half would "
                    "get no trial" % (plan.per_half, R, plan.per_half - R))
    if isinstance(cfg.get("learning_designs"), dict) and not n_learn:
        warns.append("learning_designs is set but no learning candidate (CDR, NIL, FR) is in protocols")
    if "mismatch" in cfg:
        from .mismatch import SCENARIOS
        for sc in cfg["mismatch"].get("scenarios", []):
            if sc not in SCENARIOS:
                errs.append("mismatch.scenarios: unknown scenario %r (known: %s)" % (sc, ", ".join(SCENARIOS)))
    for dk in SC.DESCRIPTIVE_KEYS:
        sec, key = dk.split(".")
        if key in (cfg.get(sec) or {}):
            warns.append("%s is descriptive: recorded with the study, not read by the code" % dk)
    blocks = cfg["tasks"] if isinstance(cfg["tasks"], list) else [cfg["tasks"]]
    if cfg.get("expansion_policy") == "explicit":
        for i, b in enumerate(blocks):
            for key in ("include_default", "variants_per_base"):
                if key not in b:
                    errs.append("tasks block %d: expansion_policy explicit needs %s" % (i, key))
            forced = sorted(f for f in b["families"] if f in FIXED_INPUT)
            if forced and b.get("inputs", ["default"]) != ["default"]:
                errs.append("tasks block %d: %s use their family-defined input; list inputs: [default] for them (or "
                            "put them in their own block)" % (i, ", ".join(forced)))
    try:
        specs = expand(cfg)
    except ValueError as e:
        errs.append("tasks: %s" % e)
        return dict(errors=errs, warnings=warns)
    dev, fin = final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    if not dev:
        errs.append("no development tasks (final_test_fraction = %s leaves none)" % cfg.get("final_test_fraction"))
        return dict(errors=errs, warnings=warns)
    G = len({s.base_id for s in dev})
    so = a.get("split_options", {}) or {}
    for kind in a.get("splits", []) or []:
        k = int((so.get(kind) or {}).get("k", 5))
        if kind in ("same-distribution", "device", "budget-extrap", "budget-interp") and G < k:
            warns.append("analysis split %s: %d development base circuits < k = %d folds (k is reduced to %d)"
                         % (kind, G, k, max(2, G)))
        if G < 2:
            errs.append("analysis split %s needs at least 2 development base circuits (found %d)" % (kind, G))
        if kind == "device":
            devs = sorted({s.device for s in dev})
            td = (so.get("device") or {}).get("test_device")
            if td and td not in devs:
                errs.append("analysis.split_options.device.test_device %r is not a development device (%s)"
                            % (td, ", ".join(devs)))
            if len(devs) < 2:
                errs.append("analysis split device needs at least 2 devices (found %s)" % devs)
        if kind == "width":
            m = (so.get("width") or {}).get("min_test_n")
            ns = sorted({s.n for s in dev})
            if m is not None and not (ns[0] < int(m) <= ns[-1]):
                errs.append("analysis.split_options.width.min_test_n = %s leaves a side empty (widths %s)" % (m, ns))
        if kind in ("noise", "joint", "device-base"):
            from .evaluate import noise_setting
            nz = sorted(set(noise_setting([s.device for s in dev]).tolist()))
            bases = sorted({s.device.split("@")[0] for s in dev})
            o = so.get(kind) or {}
            if kind == "device-base" and (len(bases) < 2 or (o.get("test_device") and o["test_device"] not in bases)):
                errs.append("analysis split device-base needs at least 2 base devices including the test device "
                            "(found %s)" % bases)
            if kind in ("noise", "joint") and (len(nz) < 2 or (o.get("test_noise") and o["test_noise"] not in nz)):
                errs.append("analysis split %s needs at least 2 noise settings including test_noise (found %s)"
                            % (kind, nz))
            if kind == "joint" and o.get("test_family") and o["test_family"] not in {s.family for s in dev}:
                errs.append("analysis split joint: test_family %r is not a development family" % o["test_family"])
        if kind == "family" and len({s.family for s in dev}) < 2:
            errs.append("analysis split family needs at least 2 families")
        if kind == "budget-extrap":
            mx = float((so.get("budget-extrap") or {}).get("max_train_B", 1e5))
            bs = sorted(int(b) for b in cfg.get("budgets", []))
            if not (bs and bs[0] <= mx < bs[-1]):
                errs.append("analysis.split_options.budget-extrap.max_train_B = %g leaves a side empty (budgets %s)"
                            % (mx, bs))
        if kind == "budget-interp" and len(set(cfg.get("budgets", []))) < 2:
            errs.append("analysis split budget-interp needs at least 2 budgets")
    return dict(errors=errs, warnings=warns)


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


OPERATIONAL_KEYS = ("limits", "status")


def scientific_digest(cfg: dict) -> str:
    """Digest of a configuration without its operational keys (``limits``: per-task and whole-run resource limits;
    ``status``: a descriptive label) — 1.9.0. A freeze records it beside the full digest, so raising a time limit to
    recover a timed-out task, or marking a study executed, does not invalidate the frozen decisions; any change of
    tasks, labels, analysis or study sections still does."""
    return freeze_digest({k: v for k, v in cfg.items() if k not in OPERATIONAL_KEYS})


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
