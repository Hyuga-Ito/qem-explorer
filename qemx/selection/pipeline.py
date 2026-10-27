"""Stage-aware, resumable study pipeline (1.9.0; second review N3).

``scripts/reproduce_v18.sh`` (1.8.0) ran every command under ``set -e`` but called ``freeze`` unconditionally — an
existing freeze returns 1, so a restarted run stopped there — and ran ``confirm … || true``, which hid every
confirmation failure. This runner replaces that pattern on every platform (Windows PowerShell included):

* a **pipeline file** (``configs/pipelines/*.yaml``, schema ``qemx.selection.pipeline/1``) names the development
  configuration, optional confirmation and controlled studies, the bundles, the figures and the stages;
* every stage is a list of commands run as **argument arrays** (``python -m qemx.selection …`` in a child process,
  never a shell string), so paths with spaces or non-ASCII characters — the Python executable's included — need no
  quoting; output is shown and logged to ``OUT/pipeline/<stage>.log``;
* a stage that already ran is **verified, not repeated**, when its result is a once-only decision (freeze, freeze
  completion, confirmation: ``binding.verify_freeze``, ``confirm --reuse``) and **resumed** when it is resumable
  (labeling skips labeled tasks; an analysis whose identity matches is kept). An inconsistency stops the pipeline
  (exit 2) — a once-only decision is never redone or silently replaced;
* **any unexpected non-zero exit stops** the stages that depend on it; nothing is ignored. The exit codes of the
  commands are passed through (1: incomplete labels; 2: refused or inconsistent);
* bundles are written into ``OUT/bundles`` and that location is passed to every later stage (replay, profile,
  recommendation check); the shipped bundles are never written;
* ``OUT/pipeline/<stage>.json`` records each stage's commands, exit codes, the code digest and the identities it
  verified, so a resumed run shows what was reused.

The historical reproduction of a frozen study (``configs/pipelines/reproduce_v18.yaml``) and new prospective studies
(``smoke19``, ``pilot19``, ``fulldev19``) are separate pipeline files; re-running a historical study into a new
directory reproduces it, it does not create new untouched evidence.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from .. import textio as TX

SCHEMA = "qemx.selection.pipeline/1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGES = ("plan", "label", "studies-label", "scripts-label", "analyze", "freeze", "nested", "freeze-complete",
          "bundles", "confirm", "replay", "studies-analyze", "scripts-analyze", "recommend-check", "profile", "figures",
          "report", "tables", "status")
PURPOSES = ("smoke", "pilot", "development", "historical-reproduction")
NESTED_EXTRAS = "deployed,nested_selective"


class PipelineError(RuntimeError):
    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


def load(path: str) -> dict:
    """A pipeline file, validated: required keys, known stages, existing configuration files (relative to the
    repository root or to the file)."""
    p = TX.read_yaml(path)
    if not isinstance(p, dict) or p.get("schema") != SCHEMA:
        raise PipelineError("%s: schema must be %s" % (path, SCHEMA))
    allowed = {"schema", "name", "purpose", "description", "out", "workers", "development", "confirmation",
               "studies", "bundles", "plot_config", "analyze", "profile", "scripts", "stages", "recommend_check",
               "tables"}
    extra = sorted(set(p) - allowed)
    if extra:
        raise PipelineError("%s: unknown keys %s" % (path, ", ".join(extra)))
    if p.get("purpose") not in PURPOSES:
        raise PipelineError("%s: purpose must be one of %s" % (path, ", ".join(PURPOSES)))
    if "development" not in p:
        raise PipelineError("%s: 'development' (the development configuration) is required" % path)
    base = os.path.dirname(os.path.abspath(path))

    def resolve(f):
        # 1.9.0 (independent verification): next to the pipeline file first, then the repository root; never a file
        # of the same name that happens to lie in the current directory (an absolute path is taken as given)
        cands = [f] if os.path.isabs(f) else [os.path.join(base, f), os.path.join(ROOT, f)]
        for cand in cands:
            if os.path.exists(cand):
                return os.path.abspath(cand)
        raise PipelineError("%s: file not found: %s (looked in %s)" % (path, f, ", ".join(cands)))
    p["development"] = resolve(p["development"])
    if p.get("confirmation"):
        p["confirmation"] = dict(p["confirmation"], config=resolve(p["confirmation"]["config"]))
    p["studies"] = [dict(s, config=resolve(s["config"])) for s in p.get("studies") or []]
    if p.get("plot_config"):
        p["plot_config"] = resolve(p["plot_config"])
    names = {s["name"] for s in p.get("scripts") or []}
    for st in p.get("stages") or STAGES:
        if st not in STAGES and st not in names:
            raise PipelineError("%s: unknown stage %r" % (path, st))
    for b in p.get("bundles") or []:
        if not {"version"} <= set(b):
            raise PipelineError("%s: every bundle needs a version" % path)
    return p


def _stop(proc, lf=None, *, child_signalled: bool = False, grace: float = 60.0) -> None:
    """Stop a running stage command the way Ctrl+C stops it — SIGTERM on POSIX, Ctrl+Break on Windows, both of which
    the command handles by stopping its labeling workers and releasing its run lock (``__main__._terminate_cleanly``)
    — and kill it if it has not ended after ``grace`` seconds. Its remaining output is still logged. When the command
    already received the interruption itself (``child_signalled``: Ctrl+C in a POSIX terminal reaches the whole
    foreground process group), it is first given time to finish its own cleanup undisturbed."""
    import signal
    if proc.poll() is not None:
        return
    if child_signalled:
        try:
            rest, _ = proc.communicate(timeout=15)
            if lf is not None and rest:
                lf.write(rest)
            return
        except Exception:                            # noqa: BLE001 - still running: stop it explicitly below
            pass
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGTERM)
    except (OSError, ValueError):
        pass
    try:
        rest, _ = proc.communicate(timeout=grace)
        if lf is not None and rest:
            lf.write(rest)
    except Exception:                                # noqa: BLE001 - timeout or a second interruption: kill it
        proc.kill()
        proc.wait()
    if lf is not None:
        lf.write(("[stage command stopped by the runner: exit %s]\n" % proc.returncode).encode("utf-8"))


NESTED_FILES = ("deployed_rows.csv", "prediction.csv", "calibration.csv")


def archive_stale_nested(out: str, problems: list) -> str | None:
    """Move deployed/selective/nested tables that cannot be reused (``run.nested_reuse_problems``) out of the analysis
    directory into ``analysis/stale_nested_<UTC time>/`` with a note — kept for inspection, never deleted — so that
    the nested evaluation of the frozen configuration can be computed (1.9.1; independent review F1). The freeze
    (the selected configuration) and every other table are untouched. Returns the directory, or None if nothing was
    there to move. The note is written first and ``results.json`` is changed only after every table was moved; a
    file that cannot be moved (e.g. open in another program on Windows) stops the stage with a PipelineError."""
    import shutil
    adir = os.path.join(out, "analysis")
    if not os.path.isdir(adir):
        return None
    names = [f for f in sorted(os.listdir(adir)) if f in NESTED_FILES or
             (f.endswith(".csv") and (f.startswith("nested_") or f.startswith("selective_")))]
    if not names:
        return None
    dest = os.path.join(adir, "stale_nested_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "README.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("Deployed/selective/nested tables moved aside by the pipeline (QEM Explorer 1.9.1) because they could "
                 "not be reused for the frozen configuration:\n" + "\n".join("- " + p for p in problems) +
                 "\nThe frozen model choice (freeze.json) is not changed; the pipeline's 'nested' stage recomputes "
                 "these tables for it (see pipeline/nested.log for whether that succeeded).\n")
    moved_files = []
    for f in names:
        try:
            shutil.move(os.path.join(adir, f), os.path.join(dest, f))
        except OSError as e:
            raise PipelineError("could not move %s to %s (%s): close any program that has it open and run the same "
                                "command again (already moved: %s)" % (f, dest, e, ", ".join(moved_files) or "none"),
                                code=2)
        moved_files.append(f)
    rj = os.path.join(adir, "results.json")
    if os.path.exists(rj):
        res = TX.read_json(rj)
        moved = {k: res.get("extras", {}).pop(k) for k in ("deployed", "selective", "nested_selective")
                 if k in res.get("extras", {})}
        res.setdefault("stale_nested", []).append(dict(moved_to=os.path.basename(dest), problems=problems,
                                                       extras=moved))
        from . import labels as LB
        LB._write_json(rj, res)
    return dest


class Runner:
    """Runs a loaded pipeline into ``out``. ``python``: the interpreter for the child commands (default: this one)."""

    def __init__(self, pipe: dict, out: str, *, workers=None, python: str | None = None, override_limits=None,
                 reset_analysis: bool = False, retry_failed: bool = False, echo=print, dry_run: bool = False,
                 accept_changes: bool = False):
        self.p, self.out = pipe, os.path.abspath(out)
        self.workers = str(workers or pipe.get("workers") or 1)
        self.py = python or sys.executable
        self.override, self.reset, self.retry = override_limits, reset_analysis, retry_failed
        self.accept = accept_changes
        self.echo, self.dry = echo, dry_run
        self.pdir = os.path.join(self.out, "pipeline")
        self.dev = pipe["development"]
        self.log = []

    # ------------------------------------------------------------------ helpers
    def _env(self) -> dict:
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(k, "1")
        if sys.platform.startswith("linux"):
            env.setdefault("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=2048")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    def sel(self, *args) -> list:
        return [self.py, "-m", "qemx.selection", *[str(a) for a in args]]

    def script(self, rel, *args) -> list:
        return [self.py, os.path.join(ROOT, rel), *[str(a) for a in args]]

    def call(self, stage: str, argv: list, ok=(0,)) -> int:
        """Run one command (an argument array) from the repository root; stream and log its output."""
        os.makedirs(self.pdir, exist_ok=True)
        shown = " ".join(('"%s"' % a) if (" " in a or not a) else a for a in argv)
        self.echo("[%s] $ %s" % (stage, shown))
        if self.dry:
            self.log.append(dict(stage=stage, argv=argv, returncode=None))
            return 0
        t0 = time.perf_counter()
        with open(os.path.join(self.pdir, stage + ".log"), "ab") as lf:      # the child's bytes, unchanged
            lf.write(("$ %s\n" % shown).encode("utf-8"))
            kw = {}
            if os.name == "nt":                      # so Ctrl+Break can be sent to the stage command alone
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(argv, cwd=ROOT, env=self._env(), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, **kw)
            try:
                for raw in proc.stdout:
                    lf.write(raw)
                    try:                             # the children print UTF-8 (PYTHONIOENCODING); anything else is
                        line = raw.decode("utf-8")   # shown escaped and marked, never silently replaced
                    except UnicodeDecodeError:
                        line = "[not UTF-8] %r" % raw
                    self.echo("  " + line.rstrip("\r\n"))
                rc = proc.wait()
            except KeyboardInterrupt:                # Ctrl+C: on POSIX the terminal sent it to the command as well
                _stop(proc, lf, child_signalled=os.name != "nt")
                raise
            except BaseException:                    # SIGTERM, Ctrl+Break, or an error of the runner itself: the
                _stop(proc, lf)                      # stage command stops too (its workers and lock with it)
                raise
            lf.write(("[exit %d, %.1f s]\n" % (rc, time.perf_counter() - t0)).encode("utf-8"))
        self.log.append(dict(stage=stage, argv=argv, returncode=rc, seconds=round(time.perf_counter() - t0, 2)))
        if rc not in ok:
            raise PipelineError("stage %s: command exited %d (%s); later stages were not run. See %s"
                                % (stage, rc, shown[:200], os.path.join(self.pdir, stage + ".log")), code=rc)
        return rc

    def mark(self, stage: str, **info) -> None:
        from . import binding as BD
        os.makedirs(self.pdir, exist_ok=True)
        TX.write_json(os.path.join(self.pdir, stage + ".json"),
                      dict(stage=stage, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), code=BD.code_digest(),
                           commands=[c for c in self.log if c["stage"] == stage], **info), indent=1, default=str,
                      atomic=True)

    def _cfg(self, path):
        from . import design as DS
        return DS.load(path)

    # ------------------------------------------------------------------ stages
    def st_plan(self):
        os.makedirs(os.path.join(self.out, "plan"), exist_ok=True)
        ok = (0, 3) if self.override else (0,)
        args = ["plan", self.dev, "--out", self.out, "--workers", self.workers,
                "--tasks-csv", os.path.join(self.out, "plan", "expanded_tasks.csv"),
                "--json", os.path.join(self.out, "plan", "plan.json")]
        self.call("plan", self.sel(*args), ok=ok)
        for s in self.p["studies"]:
            self.call("plan", self.sel("plan", s["config"], "--workers", self.workers, "--tasks-csv",
                                       os.path.join(self.out, "plan", "expanded_tasks_%s.csv" % s["subdir"])), ok=ok)
        self.mark("plan")

    def _label_args(self):
        a = ["--workers", self.workers]
        if self.override:
            a += ["--override-limits", self.override]
        if self.retry:
            a += ["--retry-failed"]
        if self.accept:
            a += ["--accept-changes"]
        return a

    def _labels_complete(self, cfg_path: str, stage: str) -> dict | None:
        """The completion of a labels directory when it is complete *and* bound to this configuration (identity:
        configuration, tasks, split, portfolio, devices); None otherwise. Complete labels are reused without calling
        ``label`` (their code may be older: adding labels with other code would need --accept-changes, reusing them
        does not); anything else is labeled (resumed) by the label command."""
        from . import binding as BD
        from . import integrity as IN
        target = os.path.join(self.out, stage)
        if self.dry or not os.path.isdir(target):
            return None
        diff = BD.identity_differences(target, self._cfg(cfg_path), stage)
        if diff is None or diff:
            return None
        c = IN.completion(target)
        return c if c["complete"] else None

    def st_label(self):
        from . import integrity as IN
        c = self._labels_complete(self.dev, "dev")
        if c is not None:
            self.echo("[label] kept: %s (bound to this configuration; arrays verified)" % IN.describe(c))
            self.mark("label", reused="complete", completion={k: v for k, v in c.items() if k != "completed_ids"})
            return
        self.call("label", self.sel("label", self.dev, "--out", self.out, *self._label_args()))
        self.mark("label", completion={k: v for k, v in IN.completion(os.path.join(self.out, "dev")).items()
                                       if k != "completed_ids"} if not self.dry else None)

    def st_studies_label(self):
        for s in self.p["studies"]:
            if self._labels_complete(s["config"], s["subdir"]) is not None:
                self.echo("[studies-label] kept: %s is complete and bound to %s" % (s["subdir"],
                                                                                  os.path.basename(s["config"])))
                continue
            self.call("studies-label", self.sel("label", s["config"], "--out", self.out, "--subdir", s["subdir"],
                                                *self._label_args()))
        self.mark("studies-label")

    def _frozen(self) -> bool:
        return os.path.exists(os.path.join(self.out, "freeze.json"))

    def _analysis_current(self, extras=None) -> bool:
        from . import integrity as IN
        from . import run as RN
        rj = os.path.join(self.out, "analysis", "results.json")
        if not os.path.exists(rj):
            return False
        res = TX.read_json(rj)
        cfg = self._cfg(self.dev)
        ident = RN.analysis_identity(cfg, IN.completion(os.path.join(self.out, "dev")))
        if res.get("identity") != ident:
            return False
        a = cfg.get("analysis", {})
        want_ev = set(a.get("splits", ["same-distribution"]))
        want_ex = set(extras if extras is not None else a.get("extras", [])) - set(RN.FROZEN_EXTRAS)
        return want_ev <= set(res.get("evaluations", {})) and want_ex <= set(res.get("extras", {}))

    def st_analyze(self):
        if self._frozen():
            from . import binding as BD
            probs = BD.verify_freeze(self.out, self._cfg(self.dev))
            if probs:
                raise PipelineError("the run is frozen but the freeze no longer matches its files: " + "; ".join(probs))
            self.echo("[analyze] kept: the run is frozen on these analysis tables (verified)")
            self.mark("analyze", reused="frozen")
            return
        from . import run as RN
        extras = (self.p.get("analyze") or {}).get("extras")
        # 1.9.1 (independent review F1): the extras of the pipeline file's override, else of the configuration,
        # without those that evaluate the frozen configuration — the 'nested' stage computes them after the freeze
        wanted = extras if extras is not None else self._cfg(self.dev).get("analysis", {}).get("extras")
        if isinstance(wanted, str):
            wanted = [x for x in wanted.split(",") if x]
        if wanted is not None and set(wanted) & set(RN.FROZEN_EXTRAS):
            extras = [x for x in wanted if x not in RN.FROZEN_EXTRAS]
            self.echo("[analyze] %s deferred to the 'nested' stage (they evaluate the frozen configuration)"
                      % ", ".join(x for x in wanted if x in RN.FROZEN_EXTRAS))
        ex = [x for x in (extras.split(",") if isinstance(extras, str) else extras or [])] if extras is not None \
            else None
        if self._analysis_current([x for x in ex if x] if ex is not None else None):
            self.echo("[analyze] kept: results.json has the same identity (configuration, labels, data, code)")
            self.mark("analyze", reused="identity")
            return
        args = ["analyze", self.dev, "--out", self.out]
        if extras is not None:
            args += ["--extras", extras if isinstance(extras, str) else ",".join(extras)]
        if self.reset:
            args += ["--reset-analysis"]
        if self.accept:
            args += ["--accept-changes"]
        self.call("analyze", self.sel(*args))
        self.mark("analyze")

    def st_freeze(self):
        self.call("freeze", self.sel("freeze", self.dev, "--out", self.out, "--reuse",
                                     *(["--accept-changes"] if self.accept else [])))
        self.mark("freeze", freeze=TX.read_json(os.path.join(self.out, "freeze.json")) if not self.dry else None)

    def st_nested(self):
        fp = os.path.join(self.out, "freeze.json")
        frz = TX.read_json(fp) if os.path.exists(fp) else {}
        if frz.get("deploy_tau") is not None:
            self.echo("[nested] kept: the freeze is completed with this nested evaluation (verified in freeze-complete)")
            self.mark("nested", reused="completed freeze")
            return
        from . import run as RN
        if self.dry:
            probs = ["dry run"]
        else:
            if not frz:                              # 1.9.1: nothing is evaluated or moved before a freeze
                raise PipelineError("the nested stage evaluates the frozen configuration: run the 'freeze' stage first",
                                    code=2)
            probs = RN.nested_reuse_problems(self.out, self._cfg(self.dev))
            legacy = [p for p in probs if p.endswith(RN.LEGACY_NO_BINDING)]
            if probs and len(legacy) == len(probs):  # tables made before 1.9.1 for the frozen configuration: kept,
                self.echo("[nested] kept: the deployed and nested tables were made before 1.9.1 (no binding) and "
                          "evaluate the frozen configuration; freeze-complete records that only their "
                          "configuration was checked")     # as `freeze --complete` accepts them
                self.mark("nested", reused="legacy tables of the frozen configuration")
                return
        if not probs:
            self.echo("[nested] kept: the deployed and nested tables evaluate the frozen configuration on the frozen "
                      "data with these settings and this code (verified)")
            self.mark("nested", reused="verified")
            return
        if not self.dry:
            self._nested_can_run()
        stale = [] if self.dry else archive_stale_nested(self.out, probs)      # 1.9.1 (independent review F1)
        if stale:
            self.echo("[nested] the existing deployed/nested tables cannot be reused (%s); moved to %s, recomputing "
                      "them for the frozen configuration — the frozen model choice is unchanged" % ("; ".join(probs),
                                                                                                   stale))
        self.call("nested", self.sel("analyze", self.dev, "--out", self.out, "--splits", "", "--extras", NESTED_EXTRAS,
                                     *(["--accept-changes"] if self.accept else [])))
        if not self.dry:
            left = RN.nested_reuse_problems(self.out, self._cfg(self.dev))
            if left:
                raise PipelineError("the nested evaluation still does not match the frozen configuration: "
                                    + "; ".join(left), code=2)
        self.mark("nested", recomputed=probs, archived=stale or None)

    def _nested_can_run(self):
        """Refuse before anything is moved when the nested re-analysis would be refused (1.9.1): the analysis tables
        were made with another analysis section, other labels or data (a frozen run is never re-analyzed in place),
        or by other code without --accept-changes."""
        from . import integrity as IN
        from . import run as RN
        rj = os.path.join(self.out, "analysis", "results.json")
        old = TX.read_json(rj).get("identity") if os.path.exists(rj) else None
        if not old:
            return
        now = RN.analysis_identity(self._cfg(self.dev), IN.completion(os.path.join(self.out, "dev")))
        diff = sorted(k for k in now if old.get(k) != now[k])
        if not diff or (diff == ["code"] and self.accept):
            return
        if diff == ["code"]:
            raise PipelineError("the analysis tables were made by other code (recorded %s, now %s): run the same "
                                "command again with --accept-changes (recorded in results.json) to compute the nested "
                                "evaluation beside them; nothing was moved" % (old.get("code"), now["code"]), code=2)
        raise PipelineError("the analysis tables were made with another %s than the frozen run has now; a frozen "
                            "decision is never re-analyzed in place — analyze a copy of the labels in a new run "
                            "directory; nothing was moved" % "/".join(diff), code=2)

    def st_freeze_complete(self):
        self.call("freeze-complete", self.sel("freeze", self.dev, "--out", self.out, "--complete", "--reuse"))
        self.mark("freeze-complete")

    def bundle_dir(self):
        return os.path.join(self.out, "bundles")

    def st_bundles(self):
        train = (self.p.get("confirmation") or {}).get("train", "dev")
        for b in self.p.get("bundles") or []:
            args = ["bundle", self.dev, "--out", self.out, "--version", b["version"], "--train", train,
                    "--bundle-out", os.path.join(self.bundle_dir(), b["version"])]
            if b.get("feature_set") and b["feature_set"] != "frozen":
                args += ["--feature-set", b["feature_set"]]
            self.call("bundles", self.sel(*args))
        self.mark("bundles", bundles=[b["version"] for b in self.p.get("bundles") or []])

    def st_confirm(self):
        c = self.p.get("confirmation")
        if not c:
            self.echo("[confirm] no confirmation set in this pipeline")
            return
        args = ["confirm", c["config"], "--out", self.out, "--dev-config", self.dev, "--train", c.get("train", "dev"),
                "--reuse", *self._label_args()]
        self.call("confirm", self.sel(*args))
        self.mark("confirm")

    def st_replay(self):
        c = self.p.get("confirmation")
        if not c:
            self.echo("[replay] no confirmation set: nothing to replay")
            return
        pairs = ",".join("%s:%s" % (b["version"], "sim" if b.get("simulation") else "nosim")
                         for b in self.p.get("bundles") or [])
        self.call("replay", self.script("scripts/replay_confirmation.py", "--labels", os.path.join(self.out, "confirm"),
                                        "--out", os.path.join(self.out, "replay"), "--bundle-dir", self.bundle_dir(),
                                        "--bundles", pairs))
        self.mark("replay")

    def st_studies_analyze(self):
        for s in self.p["studies"]:
            self.call("studies-analyze", self.sel("regimes", s["config"], "--out", self.out, "--skip-label",
                                                  "--subdir", s["subdir"]))
        self.mark("studies-analyze")

    def _script_stage(self, when: str):
        for sc in self.p.get("scripts") or []:
            if sc.get("after") != when:
                continue
            argv = [self._fill(a) for a in sc["argv"]]
            cmd = self.script(argv[0], *argv[1:]) if argv[0].endswith(".py") else self.sel(*argv)
            self.call(sc["name"], cmd)
            self.mark(sc["name"])

    def _fill(self, a: str) -> str:
        return str(a).replace("{out}", self.out).replace("{workers}", self.workers).replace(
            "{bundles}", self.bundle_dir())

    def st_scripts_label(self):
        self._script_stage("label")

    def st_scripts_analyze(self):
        self._script_stage("analyze")

    def st_recommend_check(self):
        rc = self.p.get("recommend_check") or {}
        bundles = self.p.get("bundles") or []
        if not bundles:
            return
        b = bundles[-1]
        ex = rc.get("experiment", "examples/experiments/qft_known_answer.yaml")
        args = ["recommend", "--experiment", os.path.join(ROOT, ex), "--budget", str(rc.get("budget", 100000)),
                "--device", rc.get("device", "fake_fez"), "--bundle", os.path.join(self.bundle_dir(), b["version"])]
        if b.get("simulation"):
            args.append("--allow-simulation")
        self.call("recommend-check", self.sel(*args))
        self.mark("recommend-check")

    def st_profile(self):
        pr = self.p.get("profile")
        if not pr:
            return
        dep = [b for b in self.p.get("bundles") or [] if not b.get("simulation")]
        args = ["profile", self.dev, "--out", self.out, "--widths", str(pr.get("widths", "3,4,5,6"))]
        if dep:
            args += ["--bundle", os.path.join(self.bundle_dir(), dep[0]["version"])]
        self.call("profile", self.sel(*args))
        self.mark("profile")

    def st_figures(self):
        args = ["figures", self.dev, "--out", self.out]
        if self.p.get("plot_config"):
            args += ["--plot-config", self.p["plot_config"]]
        self.call("figures", self.sel(*args))
        self.mark("figures")

    def st_report(self):
        self.call("report", self.sel("report", self.dev, "--out", self.out))
        self.mark("report")

    def st_tables(self):
        if not self.p.get("tables"):
            return
        self.call("tables", [self.py, "-m", "qemx.selection.thesis_tables", "--run", self.out, "--out",
                             os.path.join(self.out, "thesis_tables")])
        self.mark("tables")

    def st_status(self):
        # the completion report has its own name: the stage marker (pipeline/status.json) must not overwrite it
        self.call("status", self.sel("status", "--out", self.out, "--json",
                                     os.path.join(self.pdir, "completion.json")), ok=(0, 1))
        self.mark("status", completion="completion.json")

    # ------------------------------------------------------------------ driver
    def run(self, only=None, until=None) -> int:
        stages = list(self.p.get("stages") or STAGES)
        if only:
            stages = [s for s in stages if s in only]
        if until:
            if until not in stages:
                raise PipelineError("--until %s is not a stage of this pipeline (%s)" % (until, ", ".join(stages)))
            stages = stages[: stages.index(until) + 1]
        os.makedirs(self.pdir, exist_ok=True)
        TX.write_json(os.path.join(self.pdir, "pipeline.json"),
                      dict(name=self.p.get("name"), purpose=self.p.get("purpose"), development=self.dev,
                           confirmation=self.p.get("confirmation"), studies=self.p["studies"], stages=stages,
                           python=self.py, started=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1, default=str)
        for st in stages:
            fn = getattr(self, "st_" + st.replace("-", "_"), None)
            if fn is None:                                       # a named script stage
                sc = [x for x in self.p.get("scripts") or [] if x["name"] == st][0]
                argv = [self._fill(a) for a in sc["argv"]]
                self.call(st, self.script(argv[0], *argv[1:]) if argv[0].endswith(".py") else self.sel(*argv))
                self.mark(st)
                continue
            fn()
        return 0


def main(argv=None) -> int:
    import argparse
    from ..textio import utf8_stdio
    utf8_stdio()
    ap = argparse.ArgumentParser(prog="python -m qemx.selection pipeline", description=__doc__.splitlines()[0])
    ap.add_argument("pipeline")
    ap.add_argument("--out", default=None, help="run directory (default: the pipeline file's 'out')")
    ap.add_argument("--workers", default=None, help="N or auto (default: the pipeline file's 'workers', else 1)")
    ap.add_argument("--until", default=None, help="stop after this stage")
    ap.add_argument("--only", default=None, help="run only these stages (comma-separated)")
    ap.add_argument("--python", default=None, help="interpreter for the child commands (default: this one)")
    ap.add_argument("--override-limits", default=None, metavar="REASON")
    ap.add_argument("--reset-analysis", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--accept-changes", action="store_true", help="resume labeling with other code or another "
                    "environment (recorded in run.json); complete labels are reused without it")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args(argv)
    try:
        p = load(a.pipeline)
        out = a.out or p.get("out")
        if not out:
            raise PipelineError("give --out (the pipeline file has no 'out')")
        r = Runner(p, out, workers=a.workers, python=a.python, override_limits=a.override_limits,
                   reset_analysis=a.reset_analysis, retry_failed=a.retry_failed, dry_run=a.dry_run,
                   accept_changes=a.accept_changes)
        rc = r.run(only=[x for x in (a.only or "").split(",") if x] or None, until=a.until)
    except PipelineError as e:
        print("pipeline stopped: %s" % e)
        return int(getattr(e, "code", 2) or 2)
    print("pipeline finished: %s" % (out if not a.dry_run else "(dry run)"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
