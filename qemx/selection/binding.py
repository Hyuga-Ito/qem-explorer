"""Run binding: what a labels directory is, and refusing to mix it with anything else (1.7.0; review finding S3).

1.6.0 wrote a manifest only if none existed (a changed configuration kept the old one), checked a configuration
digest per task only, collected every task file of a directory (stale files of another grid were mixed in), and did
not compare the freeze record with the configuration before the final test.

Every labels directory (``dev``, ``final``, ``regimes``, a confirmation set …) now has a ``run.json``:

* **identity** — must match exactly to resume or to collect: the label-determining configuration
  (``labels.config_digest``: budgets, protocols and their definitions, r, seed, design plan), the task manifest
  (task ids and their full specifications), the split (final-test fraction, salt, the development and final task
  sets), the portfolio's protocol definitions, and the device snapshots (digests of the fake backends' calibration
  files, modifiers included);
* **execution** — code (digest of every engine file) and environment (Python, platform, key package versions):
  a resume with different code or environment is refused unless explicitly accepted (``accept_changes``), and the
  accepted change is recorded with the session;
* **imports** — task records deliberately reused from another run (``import_tasks``): source, digests, reason.

``collect`` builds the decision dataset from the manifest's tasks only and reports missing, failed and unexpected
task files; ``require_complete`` refuses to evaluate a final or confirmation set with any of them.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone

from .. import textio as TX
from . import design as DS
from . import labels as LB
from . import protocols as PR

SCHEMA = "qemx.selection.run/1"
SALT = "qemx-1.6-final"


class IncompatibleRun(RuntimeError):
    """A labels directory that belongs to another configuration, task set, split, portfolio, device snapshot, or
    (unless accepted) other code or environment."""


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# The device-modifier keys of earlier 1.7.0 builds: runs bound then digested every modifier of this set with its value
# (defaults included). Later builds added keys (g1, g2, t); a run bound before is recognized as the same device when
# the specification sets none of the added keys (``devices_equivalent``).
MODIFIER_KEY_SETS = (("s", "ro", "eng", "hom", "chi"),)


def device_digest(spec_str: str, keys=None) -> str:
    """Digest of a device specification's calibration snapshot: the fake backend's property and configuration files
    (bytes) and the modifiers (``keys``: only these modifier keys — the convention of an earlier build); a registered
    calibration snapshot is digested from its model."""
    from .. import device
    try:
        base, mods = device.parse_device(spec_str)
    except ValueError:
        # a coherent fraction runs on the exact engine, which the tasks add themselves (task.with_modifiers); the
        # digest of every other specification is unchanged, so existing runs keep their identity
        if "chi=" not in spec_str or "eng=" in spec_str:
            raise
        from .task import with_modifiers
        base, mods = device.parse_device(with_modifiers(spec_str, "eng=exact"))
    if keys is not None:
        mods = {k: v for k, v in mods.items() if k in keys}
    h = hashlib.sha256(json.dumps(dict(mods=mods), sort_keys=True).encode())
    if base in device.FAKE_DEVICES:
        b = device.fake_backend(base)
        for attr in ("props_filename", "conf_filename"):
            fn = getattr(b, attr, None)
            p = os.path.join(getattr(b, "dirname", "") or "", fn) if fn else None
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    h.update(hashlib.sha256(f.read()).digest())
            else:
                h.update(("missing:%s" % attr).encode())
    else:
        h.update(base.encode())
    return h.hexdigest()[:16]


def environment() -> dict:
    from importlib import metadata
    pk = {}
    for name in ("numpy", "scipy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "scikit-learn", "pandas"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    return dict(python=sys.version.split()[0], implementation=platform.python_implementation(),
                platform=sys.platform, machine=platform.machine(), packages=pk)


def code_digest() -> str:
    from .run import engine_digests
    return _digest(engine_digests())[:16]


def identity(cfg: dict, stage_specs: list, dev: list, fin: list, stage: str) -> dict:
    budgets = [int(b) for b in cfg.get("budgets", PR.BUDGETS)]
    protocols = list(cfg.get("protocols", list(PR.PORTFOLIO_16)))
    R, seed = int(cfg.get("R", 200)), int(cfg.get("master_seed", 20261019))
    designs = cfg.get("learning_designs", 1)
    tasks = sorted((s.task_id, s.to_json()) for s in stage_specs)
    devices = sorted({s.device for s in stage_specs} | {s.supplied_backend for s in stage_specs})
    return dict(stage=stage, labels_config=LB.config_digest(budgets, protocols, R, seed, designs),
                tasks=_digest(tasks)[:16], n_tasks=len(tasks),
                split=dict(final_test_fraction=float(cfg.get("final_test_fraction", 0.0)), salt=SALT,
                           development=_digest(sorted(s.task_id for s in dev))[:16],
                           final=_digest(sorted(s.task_id for s in fin))[:16]),
                portfolio=_digest([PR.PORTFOLIO[k].to_json() for k in protocols])[:16],
                devices={d: device_digest(d) for d in devices})


def devices_equivalent(spec_str: str, stored: str) -> bool:
    """True when a stored device digest is this specification's digest under the current convention or under the
    convention of an earlier build whose modifier keys cover every modifier the specification sets."""
    from .. import device
    if stored == device_digest(spec_str):
        return True
    try:
        mods = device.parse_device(spec_str)[1]
    except ValueError:
        from .task import with_modifiers
        mods = device.parse_device(with_modifiers(spec_str, "eng=exact"))[1]
    for keys in MODIFIER_KEY_SETS:
        if all(k in keys or v == device.DEVICE_MODS[k] for k, v in mods.items()) and \
                stored == device_digest(spec_str, keys=keys):
            return True
    return False


def _diff(a: dict, b: dict, prefix: str = "") -> list:
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += _diff(x, y, prefix + k + ".")
        elif x != y:
            out.append(prefix + k)
    return out


def bind(target: str, cfg: dict, stage_specs: list, dev: list, fin: list, stage: str, *,
         accept_changes: bool = False, support: dict | None = None) -> dict:
    """Create ``target``/run.json, or check that an existing one is the same run; append this session. Raises
    ``IncompatibleRun`` (nothing is written) on any identity difference, and on a code or environment difference
    unless ``accept_changes``. A directory with task files but no run.json (a 1.6.0 run) is refused: its labels can be
    imported explicitly (``import_tasks``)."""
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, "run.json")
    ident = identity(cfg, stage_specs, dev, fin, stage)
    execu = dict(code=code_digest(), environment=environment())
    session = dict(started=datetime.now(timezone.utc).isoformat(timespec="seconds"), execution=execu,
                   supervision=support or {})
    if os.path.exists(path):
        rec = TX.read_json(path)
        d = _diff(rec["identity"], ident)
        d = [x for x in d if not (x.startswith("devices.") and x[len("devices."):] in ident.get("devices", {})
                                  and x[len("devices."):] in rec["identity"].get("devices", {})
                                  and devices_equivalent(x[len("devices."):],
                                                         rec["identity"]["devices"][x[len("devices."):]]))]
        if d:
            raise IncompatibleRun("%s belongs to another run: %s differ%s. Use a new output directory (or import the "
                                  "reusable tasks explicitly)." % (target, ", ".join(d), "s" if len(d) == 1 else ""))
        last = rec["sessions"][-1]["execution"]
        dx = _diff(last, execu)
        if dx:
            if not accept_changes:
                raise IncompatibleRun("%s was labeled with other code or another environment (%s). Resuming would mix "
                                      "labels made by different software; accept the change explicitly "
                                      "(--accept-changes) to continue and record it." % (target, ", ".join(dx)))
            session["accepted_changes"] = dx
        rec["sessions"].append(session)
    else:
        existing = LB.task_ids(target)
        if existing:
            raise IncompatibleRun("%s already holds %d task records without a run binding (a 1.6.0 run or a copy). "
                                  "Label into a new directory; reuse records explicitly with import_tasks."
                                  % (target, len(existing)))
        rec = dict(schema=SCHEMA, created=session["started"], identity=ident,
                   identity_digest=_digest(ident)[:16], config=cfg,
                   manifest=sorted(s.task_id for s in stage_specs), sessions=[session], imports=[])
    TX.write_json(path, rec, indent=1, default=str)
    return rec


def identity_differences(target: str, cfg: dict, stage: str) -> list | None:
    """The identity fields in which ``target``/run.json differs from what ``cfg`` would bind for ``stage`` (dev,
    final, or a separate study's subdirectory); None when there is no run.json. Device digests are compared as
    ``bind`` compares them (an equivalent earlier convention is the same device). Reads only."""
    rec = load(target)
    if rec is None:
        return None
    specs = DS.expand(cfg)
    dev, fin = DS.final_test(specs, float(cfg.get("final_test_fraction", 0.0)))
    todo = fin if stage == "final" else dev
    ident = identity(cfg, todo, dev, fin, stage)
    d = _diff(rec["identity"], ident)
    return [x for x in d if not (x.startswith("devices.") and x[len("devices."):] in ident.get("devices", {})
                                 and x[len("devices."):] in rec["identity"].get("devices", {})
                                 and devices_equivalent(x[len("devices."):],
                                                        rec["identity"]["devices"][x[len("devices."):]]))]


def load(target: str) -> dict | None:
    p = os.path.join(target, "run.json")
    return TX.read_json(p) if os.path.exists(p) else None


def collect(target: str, *, require_complete: bool = False):
    """(Dataset, report) of a labels directory: the run's manifest tasks only (all task files for a legacy 1.6.0
    directory without run.json, reported as such). ``report``: missing (in the manifest, no record), failures
    (records without labels), unexpected (task files not in the manifest; never used). ``require_complete``: raise
    if anything is missing or failed."""
    from . import dataset as DT
    rec = load(target)
    files = set(LB.task_ids(target))
    if rec is None:
        ds = DT.build(target)
        report = dict(legacy=True, manifest=None, missing=[], unexpected=[], failures=list(ds.failures))
    else:
        man = set(rec["manifest"])
        report = dict(legacy=False, manifest=len(man), missing=sorted(man - files),
                      unexpected=sorted(files - man), imports=len(rec.get("imports", [])))
        chk = check_records(target, rec, sorted(man & files))
        report.update(rejected=chk["rejected"], legacy_unbound=chk["legacy_unbound"], imported=chk["imported"])
        if chk["rejected"]:
            raise IncompatibleRun("%s: %d record(s) do not belong to this run: %s" % (
                target, len(chk["rejected"]), "; ".join("%s (%s)" % (r["task_id"], r["why"])
                                                        for r in chk["rejected"][:5])))
        ds = DT.build(target, task_ids=sorted(man & files))
        report["failures"] = list(ds.failures)
    if require_complete and (report["missing"] or report["failures"]):
        raise IncompatibleRun("%s is incomplete: %d tasks missing, %d without labels" % (
            target, len(report["missing"]), len(report["failures"])))
    return ds, report


def check_records(target: str, rec: dict, task_ids) -> dict:
    """Per-record compatibility with the bound run (1.8.0; review finding F3). Each record must name its own task id,
    the run's label configuration digest, and the run's identity — or be an explicit import listed in run.json with
    the same JSON and NPZ digests (``import_tasks``: an auditable origin-to-run rule; the record keeps its original
    run id). Its NPZ must match the record's array digest. Exception, documented rather than rewritten: records
    written by ≤ 1.7.0 (no ``seed_scheme`` field) with an *empty* run id (1.7.0's mismatch labeling did not propagate
    it) are accepted as ``legacy_unbound`` when everything else matches, and reported. A 1.8.0 record without its
    run id is rejected. Returns lists of rejected
    (task_id, why), legacy_unbound and imported ids."""
    want_cfg = rec["identity"]["labels_config"]
    run_id = rec["identity_digest"]
    imports = {}
    for imp in rec.get("imports", []) or []:
        for t in imp.get("tasks", []):
            imports[t["task_id"]] = t
    rejected, legacy, imported = [], [], []
    for tid in task_ids:
        jp = os.path.join(target, "tasks", tid + ".json")
        try:
            r = LB.load_record(target, tid)
        except Exception as e:                       # noqa: BLE001 - unreadable record
            rejected.append(dict(task_id=tid, why="unreadable: %s" % str(e)[:120]))
            continue
        if r.get("task_id") != tid:
            rejected.append(dict(task_id=tid, why="the record names task %s" % r.get("task_id")))
            continue
        if r.get("config_digest") != want_cfg:
            rejected.append(dict(task_id=tid, why="label configuration %s, the run's is %s"
                                 % (r.get("config_digest"), want_cfg)))
            continue
        rid = r.get("run_id", "")
        if tid in imports:
            if LB.file_digest(jp) != imports[tid]["json_sha256"]:
                rejected.append(dict(task_id=tid, why="imported record changed after its import"))
                continue
            imported.append(tid)
        elif rid != run_id:
            if rid == "" and "seed_scheme" not in r:     # written by ≤ 1.7.0 (1.8.0 records always carry it)
                legacy.append(tid)
            else:
                rejected.append(dict(task_id=tid, why="run %s, not this run %s" % (rid or "(none)", run_id)))
                continue
        if r.get("status") == "ok":
            npz = os.path.join(target, "tasks", tid + ".npz")
            want = (r.get("arrays") or {}).get("sha256")
            if os.path.exists(npz) and want and LB.file_digest(npz) != want:
                rejected.append(dict(task_id=tid, why="arrays do not match the record's digest"))
    return dict(rejected=rejected, legacy_unbound=legacy, imported=imported)


def import_tasks(src: str, target: str, task_ids, reason: str) -> list:
    """Copy task records of ``src`` into the bound run ``target`` (explicit, auditable reuse). Only records whose
    label configuration digest equals the target run's and whose tasks are in its manifest are copied; each import
    is listed in run.json with the source path, record digests and the reason."""
    rec = load(target)
    if rec is None:
        raise IncompatibleRun("%s has no run binding; bind it first (label --limit 0)" % target)
    want = rec["identity"]["labels_config"]
    man = set(rec["manifest"])
    done = []
    os.makedirs(os.path.join(target, "tasks"), exist_ok=True)
    for tid in task_ids:
        if tid not in man:
            raise IncompatibleRun("task %s is not in the manifest of %s" % (tid, target))
        r = LB.load_record(src, tid)
        if r.get("config_digest") != want or r.get("status") != "ok":
            raise IncompatibleRun("task %s of %s has configuration %s / status %s, the run needs %s / ok"
                                  % (tid, src, r.get("config_digest"), r.get("status"), want))
        LB.load_arrays(src, tid, r)
        for ext in (".json", ".npz"):
            if os.path.exists(os.path.join(target, "tasks", tid + ext)):
                raise IncompatibleRun("task %s already exists in %s" % (tid, target))
        for ext in (".json", ".npz"):
            shutil.copy2(os.path.join(src, "tasks", tid + ext), os.path.join(target, "tasks", tid + ext))
        done.append(dict(task_id=tid, json_sha256=LB.file_digest(os.path.join(src, "tasks", tid + ".json")),
                         npz_sha256=LB.file_digest(os.path.join(src, "tasks", tid + ".npz"))))
    rec.setdefault("imports", []).append(dict(source=os.path.abspath(src), reason=reason, tasks=done,
                                              at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    TX.write_json(os.path.join(target, "run.json"), rec, indent=1, default=str)
    return done


def check_freeze(run_dir: str, cfg: dict) -> dict:
    """The freeze record of a run, checked against the configuration (digest) and the development run binding;
    raises ``IncompatibleRun`` on any mismatch."""
    p = os.path.join(run_dir, "freeze.json")
    if not os.path.exists(p):
        raise IncompatibleRun("freeze the development choices first (python -m qemx.selection freeze …)")
    frz = TX.read_json(p)
    if frz.get("scientific_config_digest"):              # 1.9.0: operational limits may change after a freeze
        if frz["scientific_config_digest"] != DS.scientific_digest(cfg):
            raise IncompatibleRun("freeze.json was made for another configuration (scientific digest %s…, this "
                                  "configuration %s…; only limits and status may differ)"
                                  % (frz["scientific_config_digest"][:12], DS.scientific_digest(cfg)[:12]))
    elif frz.get("config_digest") != DS.freeze_digest(cfg):
        raise IncompatibleRun("freeze.json was made for another configuration (digest %s…, this configuration %s…)"
                              % (str(frz.get("config_digest"))[:12], DS.freeze_digest(cfg)[:12]))
    dev = load(os.path.join(run_dir, "dev"))
    if dev is not None and frz.get("dev_identity") and frz["dev_identity"] != dev["identity_digest"]:
        raise IncompatibleRun("freeze.json was made on other development labels (%s ≠ %s)"
                              % (frz["dev_identity"], dev["identity_digest"]))
    return frz


def verify_freeze(run_dir: str, cfg: dict) -> list:
    """Problems that make an existing freeze of ``run_dir`` inconsistent with ``cfg`` and the current files (1.9.0;
    second review N3): an empty list means the freeze can be reused as it is. Checked: the configuration (scientific
    digest, else the full digest), the development run identity, the development data content (when recorded), the
    analysis table the configuration was chosen on, and — once completed — the nested evaluation it was completed
    with. Nothing is changed."""
    from . import integrity as IN
    probs = []
    try:
        frz = check_freeze(run_dir, cfg)
    except IncompatibleRun as e:
        return [str(e)]
    if frz.get("dev_data_digest"):
        c = IN.completion(os.path.join(run_dir, "dev"))
        if c["data_digest"] != frz["dev_data_digest"]:
            probs.append("the development labels changed since the freeze (data digest %s…, frozen %s…)"
                         % (str(c["data_digest"])[:12], frz["dev_data_digest"][:12]))
    src = os.path.join(run_dir, "analysis", frz.get("chosen_on") or "pooled_same-distribution.csv")
    if frz.get("analysis_digest"):
        if not os.path.exists(src):
            probs.append("the table the configuration was chosen on (%s) is missing" % os.path.basename(src))
        elif DS.freeze_digest(TX.read_text(src)) != frz["analysis_digest"]:
            probs.append("the table the configuration was chosen on (%s) changed after the freeze"
                         % os.path.basename(src))
    if frz.get("deploy_tau") is not None and frz.get("nested_digest"):
        sm = os.path.join(run_dir, "analysis", "nested_summary.csv")
        if not os.path.exists(sm):
            probs.append("the nested evaluation the freeze was completed with (nested_summary.csv) is missing")
        elif DS.freeze_digest(TX.read_text(sm)) != frz["nested_digest"]:
            probs.append("the nested evaluation (nested_summary.csv) changed after the freeze was completed")
    return probs
