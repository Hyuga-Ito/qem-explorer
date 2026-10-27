"""JSON Schemas of the protocol-selection study's files (1.6.0; 1.7.0: every shipped configuration validates —
task blocks, ``variants_per_base: all``, device modifiers, fresh-design plans — and unknown keys are refused): configurations, task specifications and label
records. ``validate(obj, kind)`` checks an object (jsonschema is required); ``write(dir)`` writes the schemas as
files for readers outside Python (``docs/selection/schemas``)."""
from __future__ import annotations

_PORTFOLIO_KEYS = ("RAW", "RO", "ZNE-L", "ZNE-R", "ZNE-E", "PEC", "CDR", "NIL", "RO-TREX", "RO-TW", "RO-DD",
                   "ZNE-L-TW", "ZNE-E-B", "PEC-TW", "FR", "SV", "ZNE-L∘SV")   # protocols.PORTFOLIO_18 (tested)

import json
import os

_NUM = {"type": "number"}
_INT = {"type": "integer"}
_STR = {"type": "string"}

_FAMILY_PARAMS = {                       # parameters each family accepts (``seed``: the transpiler seed, D24)
    "qft": {"m": {"type": ["integer", "null"], "minimum": 1}, "s": _INT},
    "ghz": {"theta": _NUM, "basis": {"enum": ["X", "Z"]}},
    "ising": {"steps": {"type": "integer", "minimum": 1}, "T": _NUM, "J": _NUM, "h": _NUM},
    "brickwork": {"layers": {"type": "integer", "minimum": 1}, "angle_seed": _INT},
    "grover": {"k": {"type": "integer", "minimum": 1}, "marked": {"type": ["integer", "null"]}},
    "bv": {"secret": {"type": ["integer", "null"]}},
    "hea": {"layers": {"type": "integer", "minimum": 1}, "angle_seed": _INT},
    "qaoa": {"p": {"type": "integer", "minimum": 1}, "angle_seed": _INT},
}
DEVICE_PATTERN = (r"^fake_[a-z0-9_]+(@[a-z][a-z0-9]{0,3}=[A-Za-z0-9.+\-]{1,24}"
                  r"(,[a-z][a-z0-9]{0,3}=[A-Za-z0-9.+\-]{1,24}){0,9})?$")


TASK = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "qemx.task/1",
    "type": "object",
    "required": ["family", "n", "params", "inputs", "observable", "device", "layout", "opt_level", "seed", "truth"],
    "properties": {
        "family": {"enum": ["qft", "ghz", "ising", "brickwork", "grover", "bv", "hea", "qaoa"]},
        "n": {"type": "integer", "minimum": 2, "maximum": 12},
        "params": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}},
        "inputs": {"enum": ["default", "plus", "product"]},
        "observable": {"enum": ["default", "z_mid", "zz_mid", "z_all"]},
        "device": {"type": "string", "pattern": DEVICE_PATTERN},
        "layout": {"enum": ["best", "alt"]},
        "opt_level": {"type": "integer", "minimum": 0, "maximum": 3},
        "seed": _INT,
        "truth": {"type": "string", "pattern": "^([a-z]{1,4}=[0-9.eE+\\-]+)?(,[a-z]{1,4}=[0-9.eE+\\-]+)*$"},
    },
    "additionalProperties": False,
}

def _family_items(fam: str) -> dict:
    props = dict(_FAMILY_PARAMS[fam], seed=_INT)
    return {"type": ["array", "null"], "items": {"type": ["object", "null"], "properties": props,
                                                 "additionalProperties": False}}


TASK_BLOCK = {
    "type": "object",
    "required": ["widths", "families"],
    "properties": {"widths": {"type": "array", "items": {"type": "integer", "minimum": 2, "maximum": 12},
                              "minItems": 1, "uniqueItems": True},
                   "widths_by_family": {"type": "object", "propertyNames": {"enum": sorted(_FAMILY_PARAMS)},
                                        "additionalProperties": {"type": "array", "items": _INT}},
                   "families": {"type": "object", "minProperties": 1,
                                "properties": {f: _family_items(f) for f in _FAMILY_PARAMS},
                                "additionalProperties": False},
                   "inputs": {"type": "array", "items": {"enum": ["default", "plus", "product"]}, "minItems": 1,
                              "uniqueItems": True},
                   "observables": {"type": "array", "items": {"enum": ["default", "z_mid", "zz_mid", "z_all"]},
                                   "minItems": 1, "uniqueItems": True},
                   "layouts": {"type": "array", "items": {"enum": ["best", "alt"]}, "minItems": 1,
                               "uniqueItems": True},
                   "devices": {"type": "array", "items": {"type": "string", "pattern": DEVICE_PATTERN},
                               "minItems": 1, "uniqueItems": True},
                   "variants_per_base": {"oneOf": [{"type": "integer", "minimum": 0}, {"const": "all"}]},
                   "include_default": {"type": "boolean"}},
    "additionalProperties": False,
}

SPLIT_KINDS = ("same-distribution", "family", "device", "width", "budget-extrap", "budget-interp", "device-base",
               "noise", "joint")
EXTRAS = ("ablation", "learning_curve", "abstention", "interval_coverage", "certificates", "floor_sensitivity",
          "budget_generalization", "deployed", "selective", "nested_selective", "design_diagnostics")
MODEL_FAMILIES = ("ridge", "enet", "hgb", "mlp", "struct")               # models.FAMILIES (tested)
_FOLD = {"k": {"type": "integer", "minimum": 2}, "seed": {"type": "integer", "minimum": 0}}
_SPLIT_OPTIONS = {
    "same-distribution": dict(_FOLD),
    "family": dict(_FOLD),
    "device": dict(_FOLD, test_device={"type": "string", "pattern": DEVICE_PATTERN}),
    "width": dict(_FOLD, min_test_n={"type": "integer", "minimum": 2}),
    "budget-extrap": dict(_FOLD, max_train_B={"type": "number", "exclusiveMinimum": 0}),
    "budget-interp": dict(_FOLD),
    "device-base": dict(_FOLD, test_device={"type": "string", "pattern": "^fake_[a-z0-9_]+$"}),
    "noise": dict(_FOLD, test_noise=_STR),
    "joint": dict(_FOLD, test_family=_STR, test_noise=_STR),
}
_NAMES = {"type": "array", "items": _STR, "minItems": 1, "uniqueItems": True}
_FAMS = {"type": "array", "items": {"enum": list(MODEL_FAMILIES)}, "minItems": 1, "uniqueItems": True}
# 1.9.0 (second review N6): the analysis, mismatch and regimes sections are validated key by key (1.8.0 accepted any
# object, so a typo such as ``feature_set`` silently fell back to the default). Names of feature sets and mismatch
# scenarios are checked against the code by ``design.check``.
ANALYSIS = {
    "type": "object",
    "properties": {
        "splits": {"type": "array", "items": {"enum": list(SPLIT_KINDS)}, "uniqueItems": True},
        "split_options": {"type": "object", "additionalProperties": False,
                          "properties": {k: {"type": "object", "properties": v, "additionalProperties": False}
                                         for k, v in _SPLIT_OPTIONS.items()}},
        "feature_sets": _NAMES, "model_families": _FAMS,
        "abstain_family": {"enum": list(MODEL_FAMILIES)}, "deployed_family": {"enum": list(MODEL_FAMILIES)},
        "abstain_set": _STR, "ablation_set": _STR, "deployed_set": _STR,
        "other_split_sets": _NAMES, "other_split_families": _FAMS,
        "bootstrap": {"type": "integer", "minimum": 1}, "selective_bootstrap": {"type": "integer", "minimum": 1},
        "lc_repeats": {"type": "integer", "minimum": 1}, "nested_inner_k": {"type": "integer", "minimum": 2},
        "extras": {"type": "array", "items": {"enum": list(EXTRAS)}, "uniqueItems": True},
    },
    "additionalProperties": False,
}
MISMATCH = {
    "type": "object",
    "required": ["scenarios"],
    "properties": {
        "scenarios": {"type": "array", "items": _STR, "minItems": 1, "uniqueItems": True},
        "diag_fraction": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        "diag_min": {"type": "integer", "minimum": 1},
        "alpha": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        "model_family": {"enum": list(MODEL_FAMILIES)}, "model_set": _STR,
        "diagnostic_draws": {"type": "integer", "minimum": 1},
        "robust": _STR, "reaction_holdout": {"enum": ["type"]},        # descriptive (recorded, not read by the code)
    },
    "additionalProperties": False,
}
REGIMES = {                                                          # descriptive: recorded with the study
    "type": "object",
    "properties": {"noise_axis": _STR, "model_policy": _STR, "observable_budget": {"type": "integer", "minimum": 1},
                   "noise_budget": {"type": "integer", "minimum": 1}},
    "additionalProperties": False,
}
DESCRIPTIVE_KEYS = ("mismatch.robust", "mismatch.reaction_holdout", "regimes.noise_axis", "regimes.model_policy",
                    "regimes.observable_budget", "regimes.noise_budget")

CONFIG = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "qemx.selection.config/1",
    "type": "object",
    "required": ["schema", "name", "master_seed", "R", "budgets", "protocols", "tasks"],
    "properties": {
        "schema": {"const": "qemx.selection.config/1"},
        "name": _STR,
        "status": {"enum": ["executed", "planned", "unexecuted", "smoke"]},
        "master_seed": {"type": "integer", "minimum": 0},
        "R": {"type": "integer", "minimum": 2},
        "expansion_policy": {"enum": ["legacy", "explicit"]},
        "learning_designs": {"oneOf": [
            {"type": "integer", "minimum": 1},
            {"type": "object", "required": ["mode"],
             "properties": {"mode": {"enum": ["fresh", "fixed-shared"]}, "per_half": {"type": "integer", "minimum": 1},
                            "count": {"type": "integer", "minimum": 1},
                            "seed_scheme": {"enum": [1, 2]}, "crn": {"enum": ["task", "base"]}},
             "additionalProperties": False}]},
        "budgets": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1, "uniqueItems": True},
        "protocols": {"type": "array", "items": {"enum": list(_PORTFOLIO_KEYS)}, "minItems": 1, "uniqueItems": True},
        "final_test_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "truth": _STR,
        "tasks": {"oneOf": [TASK_BLOCK, {"type": "array", "items": TASK_BLOCK, "minItems": 1}]},
        "limits": {"type": "object", "properties": {"task_seconds": {"type": "number", "minimum": 0},
                                                    "memory_gb": {"type": "number", "minimum": 0},
                                                    "cpu_hours": {"type": "number", "minimum": 0},
                                                    "disk_gb": {"type": "number", "minimum": 0},
                                                    "min_free_disk_gb": {"type": "number", "minimum": 0}},
                   "additionalProperties": False},
        "analysis": ANALYSIS,
        "mismatch": MISMATCH,
        "regimes": REGIMES,
        "study": {"type": "object"},
    },
    "additionalProperties": False,
}

LABEL = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "qemx.selection.labels/2",
    "type": "object",
    "required": ["schema", "task_id", "base_id", "spec", "config_digest", "budgets", "protocols", "R", "status"],
    "properties": {
        "schema": {"enum": ["qemx.selection.labels/1", "qemx.selection.labels/2"]},
        "task_id": {"type": "string", "pattern": "^[0-9a-f]{20}$"},
        "base_id": {"type": "string", "pattern": "^[0-9a-f]{16}$"},
        "spec": TASK,
        "config_digest": _STR,
        "budgets": {"type": "array", "items": _INT},
        "protocols": {"type": "array", "items": _STR},
        "R": _INT,
        "status": {"enum": ["ok", "error", "timeout", "memory"]},
        "ideal": _NUM,
        "allocations": {"type": "array", "items": {
            "type": "object", "required": ["protocol", "B", "feasible"],
            "properties": {"protocol": _STR, "B": _INT, "feasible": {"type": "boolean"}, "reason": _STR,
                           "N": _INT, "s": _INT, "c": _INT, "s_tr": _INT, "split": {"type": "array"},
                           "ledger": {"type": "object", "additionalProperties": _INT}}}},
        "features": {"type": "object"},
        "task": {"type": "object"},
    },
}

KINDS = {"task": TASK, "config": CONFIG, "label": LABEL}


def validate(obj, kind: str) -> None:
    import jsonschema
    jsonschema.validate(obj, KINDS[kind])


def write(out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, sch in KINDS.items():
        p = os.path.join(out_dir, "%s.schema.json" % k)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(sch, f, indent=1, ensure_ascii=False)
            f.write("\n")
        paths.append(p)
    return paths
