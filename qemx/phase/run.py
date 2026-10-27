r"""Command-line runner for Direction A manifests.

    python -m qemx.phase.run plan    MANIFEST [--stage S …]
    python -m qemx.phase.run masterlist MANIFEST --out cells.csv [--with-lambda]
    python -m qemx.phase.run run     MANIFEST --out DIR [--stage S …] [--workers K] [--limit N]
                                     [--time-limit SEC] [--retry-errors]
    python -m qemx.phase.run collect DIR
    python -m qemx.phase.run status  DIR

``run`` is resumable: a cell whose record exists with status "ok" is skipped (``--retry-errors`` also re-runs cells
that failed or completed only partly). An output directory belongs to one manifest (its SHA-256 is checked; ``--force``
rebinds it). Every cell runs inside try/except: a failure (exception, time limit, out of memory) is written as a
record with its status and traceback, and the run continues. The time limit is cooperative (checked between
circuit simulations, training-level refits and validation replicates), not a hard kill. Cells of the same circuit go to the same worker, so the
compilation is shared. Each worker uses one BLAS thread (set before NumPy is imported).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _single_thread():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def _limit_blas():
    """One BLAS/OpenMP thread in this process, also if NumPy was imported before the environment was set."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:                                    # noqa: BLE001 - threadpoolctl is optional
        pass


def _run_group(args):
    """Worker: run a list of cells (same circuit) and write their records. Returns [(cell_id, status, seconds)]."""
    _single_thread()
    _limit_blas()
    import warnings
    warnings.simplefilter("ignore")
    out_dir, specs, time_limit = args
    from .cell import CellSpec, cell_runner
    from .store import failure_record, write_cell
    from ..resources import RunTimeout
    res = []
    crash = os.environ.get("QEMX_PHASE_CRASH_CELL", "")          # fault injection for the runner's own tests
    for d in specs:
        spec = CellSpec.from_json(d)
        if crash and spec.cell_id.startswith(crash):
            os._exit(9)                                            # simulate a worker killed by the OOM killer
        t0 = time.time()
        try:
            rec = cell_runner(spec, time_limit=time_limit)
        except RunTimeout as e:
            rec = failure_record(spec.to_json(), spec.cell_id, "timeout", str(e), time.time() - t0)
        except MemoryError as e:
            rec = failure_record(spec.to_json(), spec.cell_id, "oom", "MemoryError: %s" % e, time.time() - t0)
        except Exception as e:                                   # noqa: BLE001 - recorded, the study goes on
            rec = failure_record(spec.to_json(), spec.cell_id, "error", "%s: %s\n%s" % (
                type(e).__name__, e, traceback.format_exc()), time.time() - t0)
        if rec.get("cell_id") != spec.cell_id:
            raise AssertionError("record id mismatch")
        try:
            write_cell(out_dir, rec)
        except ValueError as e:                                  # a record that violates the schema: a pipeline bug
            rec = failure_record(spec.to_json(), spec.cell_id, "error", "invalid record: %s" % e, time.time() - t0)
            write_cell(out_dir, rec)
        res.append((spec.cell_id, rec["status"], time.time() - t0))
    return res


def _groups(cells: list) -> list:
    g = {}
    for c in cells:
        key = (c.family, c.n, c.fparams, c.device, c.layout, c.opt_level, c.seed_transpiler)
        g.setdefault(key, []).append(c)
    return list(g.values())


def cmd_plan(a):
    from .manifest import expand, load
    m = load(a.manifest)
    cells = expand(m, a.stage)
    by = {}
    for c in cells:
        k = c.stage
        e = by.setdefault(k, dict(cells=0, validated=0, circuits=set(), methods=len(c.methods), engine=c.engine))
        e["cells"] += 1
        e["validated"] += bool(c.validate_budgets)
        e["circuits"].add((c.family, c.n, c.fparams))
    for k, e in by.items():
        print("%-28s engine %s  cells %5d  circuits %3d  methods %d  validated %4d" % (
            k, e["engine"], e["cells"], len(e["circuits"]), e["methods"], e["validated"]))
    print("total cells: %d" % len(cells))


def prepare_out(out: str, manifest_path: str, force: bool = False) -> dict:
    """Bind an output directory to one manifest. A new directory gets a verbatim copy of the manifest and run.json;
    an existing one must hold the same manifest (SHA-256) — mixing records of two manifests in one directory would
    make collect's table a union of two studies — unless ``force`` (the copy and run.json are then replaced).
    Every invocation is appended to run_history.jsonl. Returns the run info."""
    from .manifest import manifest_hash
    from .cell import provenance
    os.makedirs(out, exist_ok=True)
    h = manifest_hash(manifest_path)
    copy = os.path.join(out, "manifest.yaml")
    if os.path.exists(copy) and manifest_hash(copy) != h and not force:
        raise SystemExit("%s holds a different manifest (sha256 %s… ≠ %s…); use another --out or --force"
                         % (out, manifest_hash(copy)[:12], h[:12]))
    from ..provenance import provenance as full_provenance, redact_path
    info = dict(manifest=redact_path(os.path.abspath(manifest_path)), manifest_sha256=h,
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), argv=redact_path(list(sys.argv)),
                provenance=dict(provenance(), environment=full_provenance(full=True)))
    if not os.path.exists(copy) or force:
        with open(manifest_path, encoding="utf-8") as f, open(copy, "w", encoding="utf-8") as g:
            g.write(f.read())
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
    with open(os.path.join(out, "run_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(info) + "\n")
    return info


def cmd_run(a):
    _single_thread()
    from .manifest import expand, load
    from .store import clean_tmp, read_cell
    m = load(a.manifest)
    cells = expand(m, a.stage)
    prepare_out(a.out, a.manifest, force=a.force)
    n_tmp = clean_tmp(a.out)
    if n_tmp:
        print("removed %d temporary files of interrupted writes" % n_tmp, flush=True)
    from .store import SCHEMA_VERSION
    todo, older = [], 0
    for c in cells:
        r = read_cell(a.out, c.cell_id)
        if r is not None and r.get("schema") != SCHEMA_VERSION:
            older += 1
        if r is None or (a.retry_errors and r.get("status") != "ok"):
            todo.append(c)
    if older and todo and not a.force:
        # never mix records of two statistical semantics in one table (stats contract, record schema /3)
        raise SystemExit("%s holds %d records written with an older record schema (statistical semantics differ: see "
                         "qemx/phase/store.py); finishing it with this version would mix them. Run into a new --out "
                         "directory (the old records stay as they are), or pass --force to accept the mixture."
                         % (a.out, older))
    if a.limit:
        todo = todo[: a.limit]
    print("%d cells, %d to run" % (len(cells), len(todo)), flush=True)
    groups = [(a.out, [c.to_json() for c in g], a.time_limit) for g in _groups(todo)]
    done, t0 = 0, time.time()
    _limit_blas()

    def report(res):
        nonlocal done
        for cid, st, sec in res:
            done += 1
            print("[%d/%d] %s %s %.1fs" % (done, len(todo), cid, st, sec), flush=True)
    if a.workers <= 1:
        for g in groups:
            report(_run_group(g))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from .cell import CellSpec
        from .store import failure_record, write_cell

        def unfinished(gs):
            """The groups' cells that have no record yet (a dead worker may have finished part of its group)."""
            out = []
            for g in gs:
                left = [d for d in g[1] if read_cell(a.out, CellSpec.from_json(d).cell_id) is None]
                if left:
                    out.append((g[0], left, g[2]))
            return out
        pending, attempt = groups, 0
        while pending and attempt < 3:
            # a worker that dies (OOM killer, signal) breaks the whole pool: every unfinished group is resubmitted to
            # a fresh pool, in parallel; finished cells are never re-run
            failed = []
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(_run_group, g): g for g in pending}
                for fu in as_completed(futs):
                    try:
                        report(fu.result())
                    except Exception as e:               # noqa: BLE001 - BrokenProcessPool and friends
                        failed.append(futs[fu])
                        err = "%s: %s" % (type(e).__name__, e)
            pending = unfinished(failed)
            attempt += 1
            if pending:
                print("%d groups unfinished after a worker failure (%s); resubmitting (attempt %d)" % (
                    len(pending), err[:200], attempt + 1), flush=True)
        for g in pending:                                # still failing: isolate every remaining cell
            for d in g[1]:
                spec = CellSpec.from_json(d)
                if read_cell(a.out, spec.cell_id) is not None:
                    continue
                try:
                    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex1:
                        report(ex1.submit(_run_group, (g[0], [d], g[2])).result())
                except Exception as e:                   # noqa: BLE001 - the cell itself kills its process
                    from concurrent.futures.process import BrokenProcessPool
                    status = "oom" if isinstance(e, (BrokenProcessPool, MemoryError)) else "crashed"
                    write_cell(a.out, failure_record(
                        spec.to_json(), spec.cell_id, status, "the worker process died while running this cell "
                        "alone (%s: %s); %s" % (type(e).__name__, e, "most likely out of memory"
                                               if status == "oom" else "see the error")))
                    report([(spec.cell_id, status, 0.0)])
    print("finished %d cells in %.1f min" % (done, (time.time() - t0) / 60), flush=True)


def cmd_masterlist(a):
    """Every cell of the manifest as one CSV row (the 'just execute this' list), optionally with its circuit facts and
    λ_LC computed in advance (compilation and λ only — no method is run)."""
    _limit_blas()
    import csv
    import warnings
    warnings.simplefilter("ignore")
    from .manifest import expand, load
    from .cell import compiled_for, noise_for
    m = load(a.manifest)
    cells = expand(m, a.stage)
    cols = ["cell_id", "stage", "family", "n", "fparams", "device", "layout", "opt_level", "seed_transpiler", "engine",
            "scale", "readout", "readout_mit", "coherent_fraction", "methods", "seed", "validate_budgets",
            "validate_policy", "validate_reps", "train_levels", "train_reps", "train_reps_other", "s_min"]
    extra = ["observable", "basis", "phys", "two_qubit", "depth", "ideal_note", "lam_lc", "lam_lc_exact", "lam_total",
             "lam_ro"] if a.with_lambda else []
    from .lightcone import calculate_lambda_lc
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + extra)
        for i, c in enumerate(cells):
            row = [c.cell_id, c.stage, c.family, c.n, json.dumps(dict(c.fparams), sort_keys=True), c.device,
                   json.dumps(list(c.layout)) if c.layout else "best chain", c.opt_level, c.seed_transpiler, c.engine,
                   c.scale, c.readout, c.readout_mit, c.coherent_fraction, " ".join(ms.key for ms in c.methods), c.seed,
                   " ".join(str(b) for b in c.validate_budgets), " ".join(c.validate_policy) if c.validate_budgets
                   else "", c.validate_reps if c.validate_budgets else "", " ".join(map(str, c.train_levels)),
                   c.train_reps, c.train_reps_other, c.s_min]
            if a.with_lambda:
                fc, comp = compiled_for(c)
                _, _, lam_model, _ = noise_for(c, comp)
                lam = calculate_lambda_lc(comp, lam_model, fc.observable, layout=comp.info["phys"])
                row += [fc.observable.label, fc.basis, json.dumps(comp.info["phys"]), comp.info["two_qubit"],
                        comp.info["depth"], fc.notes[:80], "%.6g" % lam.lam_lc, "%.6g" % lam.lam_lc_exact,
                        "%.6g" % lam.lam_total, "%.6g" % lam.lam_ro]
            w.writerow(row)
            if a.with_lambda and (i + 1) % 100 == 0:
                print("%d/%d" % (i + 1, len(cells)), flush=True)
    print("wrote %d cells to %s" % (len(cells), a.out))


def cmd_collect(a):
    from .store import collect
    print(json.dumps(collect(a.out, only_manifest=not a.all_records), indent=1))


def cmd_status(a):
    from .store import iter_cells
    st = {}
    for r in iter_cells(a.out):
        k = (r["cell"]["stage"], r["status"])
        st[k] = st.get(k, 0) + 1
    for (stage, s), n in sorted(st.items()):
        print("%-28s %-8s %d" % (stage, s, n))


def cmd_holdout_check(a):
    """The final holdout shares no base circuit with the development manifests."""
    import json
    from .manifest import check_disjoint, load
    out = check_disjoint(load(a.manifest), *[load(m) for m in a.against])
    print(json.dumps(out, indent=1))


def main(argv=None):
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    p = argparse.ArgumentParser(prog="python -m qemx.phase.run", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("plan")
    q.add_argument("manifest")
    q.add_argument("--stage", action="append")
    q.set_defaults(fn=cmd_plan)
    q = sub.add_parser("run")
    q.add_argument("manifest")
    q.add_argument("--out", required=True)
    q.add_argument("--stage", action="append")
    q.add_argument("--workers", type=int, default=1)
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--time-limit", type=float, default=None, help="seconds per cell (default: none)")
    q.add_argument("--retry-errors", action="store_true", help="also re-run cells whose status is not 'ok'")
    q.add_argument("--force", action="store_true", help="bind --out to this manifest even if it holds another one, or finish a directory that holds records of an "
                        "older record schema (mixed statistical semantics; not recommended)")
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("masterlist")
    q.add_argument("manifest")
    q.add_argument("--out", required=True)
    q.add_argument("--stage", action="append")
    q.add_argument("--with-lambda", action="store_true", help="compile every circuit and compute λ_LC (minutes)")
    q.set_defaults(fn=cmd_masterlist)
    q = sub.add_parser("collect")
    q.add_argument("out")
    q.add_argument("--all-records", action="store_true", help="also use records that are not cells of the manifest")
    q.set_defaults(fn=cmd_collect)
    q = sub.add_parser("status")
    q.add_argument("out")
    q.set_defaults(fn=cmd_status)
    q = sub.add_parser("holdout-check", help="check that a holdout manifest shares no base circuit with development "
                                             "manifests")
    q.add_argument("manifest")
    q.add_argument("--against", nargs="+", required=True)
    q.set_defaults(fn=cmd_holdout_check)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
