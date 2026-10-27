"""The main study's hardware transfer test (qemx.transfer): plan, emulate, and analyze; real submission only through
the campaign-ledger guard after explicit approval.

    python scripts/hardware_transfer.py plan     --out reports/hardware_transfer_plan          # offline, no account
    python scripts/hardware_transfer.py rehearse --out reports/hardware_transfer_rehearsal    # EMULATION on fake_fez

Hardware (after approval only; the token comes from QISKIT_IBM_TOKEN, a saved account or a hidden prompt — never a
command-line argument):

    python scripts/hardware_transfer.py run-ibm     --out data/transfer --ledger campaign_ledger.json --blocks K
    python scripts/hardware_transfer.py collect-ibm --out data/transfer --ledger campaign_ledger.json
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qemx import transfer as TR   # noqa: E402


def cmd_plan(a):
    from qemx import campaign as CG
    os.makedirs(a.out, exist_ok=True)
    pred = TR.predictions()
    model = CG.UsageModel(per_shot_s=a.per_shot_ms / 1e3) if a.per_shot_ms else CG.UsageModel()
    pf = TR.preflight(TR.DESIGN, model, a.blocks, pred)
    TR.save(os.path.join(a.out, "predictions.json"), pred)
    TR.save(os.path.join(a.out, "preflight.json"), pf)
    print(json.dumps({k: pf[k] for k in ("blocks", "block", "stage1", "total_reserved_s", "admissible_s", "fits",
                                         "max_blocks_that_fit")}, indent=1, default=float))
    for x in pf["contrasts"]:
        print("%-7s %-11s %-6s vs %-6s predicted RMSE %.4f vs %.4f  power(K=%d) %.2f" % (
            x["circuit"], x["kind"], x["better"], x["worse"], x["rmse_better"], x["rmse_worse"], pf["blocks"],
            x["power"]))
    return 0


def cmd_rehearse(a):
    os.makedirs(a.out, exist_ok=True)
    pred = TR.predictions()
    rows = TR.rehearse(TR.DESIGN, a.blocks, progress=lambda b, n: print("block %d/%d" % (b, n), flush=True))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "block_errors.csv"), index=False, encoding="utf-8")
    an = TR.analyze(rows, pred)
    TR.save(os.path.join(a.out, "predictions.json"), pred)
    TR.save(os.path.join(a.out, "analysis.json"), an)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(TR.report_markdown(an, None, "EMULATION on %s (local testing mode, %d blocks)" % (
            TR.DESIGN["device"], a.blocks)))
    print(json.dumps(an["contrasts"], indent=1, default=float))
    return 0


def _service(a):
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("QISKIT_IBM_TOKEN")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    try:
        return QiskitRuntimeService()                     # a saved account
    except Exception:                                     # noqa: BLE001
        if not a.ask_token:
            raise SystemExit("no QISKIT_IBM_TOKEN and no saved account; pass --ask-token for a hidden prompt")
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=getpass.getpass("IBM Quantum token: "))


def _ideals(rs):
    from qemx import hamiltonian as HM
    from qemx.methods.base import Context, Recipe, Runner
    from qemx.noise import NoiseSpec
    return {c["name"]: float(HM.ideal_value(lambda cc, o: Context(cc, NoiseSpec(), o, Recipe.make("raw"),
                                                                   Runner("numpy")), r.compiled, r.observable))
            for c, r in rs}


def _save_plan(out: str, key: str, plan) -> dict:
    """Pickle the submitted plan (local working file; its SHA-256 goes into jobs.json and is checked on loading).
    Never load plan files from someone else: unpickling runs code."""
    import hashlib
    import pickle
    os.makedirs(os.path.join(out, "plans"), exist_ok=True)
    name = "plans/%s.pkl" % key.replace("/", "_")
    blob = pickle.dumps(plan)
    with open(os.path.join(out, name), "wb") as f:
        f.write(blob)
    return dict(file=name, sha256=hashlib.sha256(blob).hexdigest())


def _load_plan(out: str, ref: dict):
    import hashlib
    import pickle
    with open(os.path.join(out, ref["file"]), "rb") as f:
        blob = f.read()
    if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
        raise SystemExit("%s does not match the digest recorded at submission" % ref["file"])
    return pickle.loads(blob)                             # noqa: S301 - the tool's own file, digest-checked


FROZEN = "frozen_predictions.json"     # run-ibm's predictions (never the plan's or a rehearsal's predictions.json)


def _settle_attempts(ent: dict, ledger) -> list:
    """The records of an interrupted block, with every pending provider call settled from the ledger: a call whose
    reservation was resolved to a job becomes that job's record (never sent again); one resolved as not submitted is
    dropped (it is sent again). A call the ledger still holds as reserved stops the run until it is resolved."""
    recs = list(ent.get("records") or [])
    for att in ent.get("pending") or []:
        e = None
        try:
            e = ledger.refresh().entry(att["reservation"]) if att.get("reservation") else None
        except (KeyError, ValueError):
            e = ledger.entry_for_tag(att.get("client_tag", "")) if att.get("client_tag") else None
        if e is not None and e.get("job_id"):
            recs.append(dict(att, job_id=e["job_id"], outcome="resolved in the ledger"))
        elif e is None or e.get("state") == "released":
            continue
        else:
            raise SystemExit("reservation %s (client tag %s) of an interrupted block is still %s: resolve it first "
                             "(python -m qemx.campaign resolve --ledger %s --id %s --job-id <JOB> or --not-submitted)"
                             % (e.get("id"), att.get("client_tag"), e.get("state"), ledger.path, e.get("id")))
    return recs


def cmd_run_ibm(a):
    """Compile for the day's backend, freeze the predictions made with its current calibration (the design's model
    and simulation path, ``transfer.predictions_on``), then submit K blocks; every job is admitted by the campaign
    ledger first, each provider call is on disk before it is made, and every job id is recorded at once. A resumed
    run never re-sends a recorded job: an interrupted block is finished with its own saved plan, sending only the
    jobs that are not recorded (calls whose outcome was unknown are settled from the ledger first)."""
    from qemx import campaign as CG
    from qemx import hardware as H
    ledger = CG.open_ledger(a.ledger)                     # required: a missing path is an error, never "no ledger"
    CG.check_no_unresolved(ledger)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "jobs.json")
    done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else dict(blocks={})
    service = _service(a)
    backend = service.backend(a.backend)
    rs = TR.resolve_on(TR.DESIGN, backend)
    pp = os.path.join(a.out, FROZEN)
    if not os.path.exists(pp):                            # frozen before the first job, never rewritten
        TR.save(pp, dict(backend=backend.name, design=TR.DESIGN["version"],
                         circuits=TR.predictions_on(TR.DESIGN, backend, rs)))
    for b in range(a.blocks):
        for ci, (c, r) in enumerate(rs):
            key = "%d/%s" % (b, c["name"])
            ent = done["blocks"].get(key)
            if ent is not None and not ent.get("partial"):
                continue                                  # complete: never re-send
            if ent is not None:                           # interrupted: its own plan, only the jobs not yet sent
                plan_file = ent["plan"]
                plan = _load_plan(a.out, plan_file)
                prev = _settle_attempts(ent, ledger)
            else:
                plan = TR.block_plan(TR.DESIGN, r, backend, b, ci)
                plan_file = _save_plan(a.out, key, plan)   # the exact plan the counts belong to (calibration of today)
                prev = []
            done["blocks"][key] = dict(records=prev, partial=True, plan=plan_file, pending=[])
            TR.save(path, done)

            def keep(recs, key=key, prev=prev):
                done["blocks"][key].update(records=prev + list(recs), pending=[])
                TR.save(path, done)

            def attempt(rec, key=key):
                done["blocks"][key]["pending"] = [rec]     # the call about to be made, until its job id is known
                TR.save(path, done)
            recs = H.submit(plan, backend, ledger=ledger, mode=a.mode, campaign="transfer", on_record=keep,
                            on_attempt=attempt, skip=prev, seed=TR.block_seed(b, ci))
            done["blocks"][key] = dict(records=prev + list(recs), partial=False, plan=plan_file, pending=[])
            TR.save(path, done)
            print("block %s: %d job(s)%s" % (key, len(prev) + len(recs),
                                             " (%d sent before)" % len(prev) if prev else ""), flush=True)
    return 0


def cmd_collect_ibm(a):
    from qemx import campaign as CG
    from qemx import hardware as H
    ledger = CG.open_ledger(a.ledger)
    done = json.load(open(os.path.join(a.out, "jobs.json"), encoding="utf-8"))
    fp = os.path.join(a.out, FROZEN)
    if not os.path.exists(fp):
        raise SystemExit("%s is missing: collect-ibm compares with the predictions run-ibm froze before the first job"
                         % fp)
    pred = json.load(open(fp, encoding="utf-8"))
    service = _service(a)
    rows = []
    for key, rec in sorted(done["blocks"].items()):
        b, name = key.split("/", 1)
        if rec.get("partial"):                           # some of its jobs were never sent: not a complete protocol
            print("block %s: incomplete (finish it with run-ibm); left out" % key)
            continue
        plan = _load_plan(a.out, rec["plan"])            # the plan the jobs were built from, not today's
        freqs, status = H.collect(service, rec["records"], plan.compiled.implemented.n, wait=a.wait, ledger=ledger)
        if freqs is None:
            print("block %s: not finished (%s)" % (key, status))
            continue
        lost = sorted({k for st in status for k in st.get("incomplete_requests", [])})
        if lost:
            print("block %s: %d request(s) lost with a failed job; their protocols report no estimate" % (key, len(lost)))
        rows += TR.block_errors(TR.DESIGN, plan, freqs, int(b), name)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "block_errors.csv"), index=False, encoding="utf-8")
    an = TR.analyze(rows, pred["circuits"])
    TR.save(os.path.join(a.out, "analysis.json"), an)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(TR.report_markdown(an, None, "hardware (%s)" % pred.get("backend")))
    return 0


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--blocks", type=int, default=None, help="fix K (default: the power target)")
    p.add_argument("--per-shot-ms", type=float, default=None, help="calibrated per-shot usage after Stage 1")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("rehearse")
    p.add_argument("--out", required=True)
    p.add_argument("--blocks", type=int, default=8)
    p.set_defaults(fn=cmd_rehearse)
    for name, fn in (("run-ibm", cmd_run_ibm), ("collect-ibm", cmd_collect_ibm)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--ledger", required=True)
        p.add_argument("--backend", default=TR.DESIGN["target_backend"])
        p.add_argument("--ask-token", action="store_true")
        if name == "run-ibm":
            p.add_argument("--blocks", type=int, required=True)
            p.add_argument("--mode", choices=("job", "batch"), default="job")
        else:
            p.add_argument("--wait", action="store_true")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
