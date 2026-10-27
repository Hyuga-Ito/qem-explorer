r"""Hardware validation of the selection recommender: planning, preflight and an authorization gate (1.7.0;
1.8.0: one deployed-policy call per task, the preflight status of every planned protocol recorded).

Hardware transfer is a secondary, optional stage of the study (roadmap stage F). This module prepares it; it never
submits anything by itself.

* ``plan`` — for a few declared tasks, the protocols to compare at one total budget B each (fixed names such as RAW
  and RO, the bundle's recommendation ``recommended`` and its budget-only fallback ``SBS-B``), recorded and allocated
  by the approved hardware path (``qemx.hardware.make_plan``: the same integer ledgers as the simulator, the target
  backend's calibration as the supplied model) and repeated ``blocks`` times (independent deployments, the unit of a
  hardware risk estimate). QPU usage is estimated per job with the campaign's conservative usage model and admitted
  job by job against **one shared ceiling (600 s by default) minus a reserve**, or against the campaign ledger's
  remaining allowance when a ledger is given (``qemx.campaign``). The plan is written with a digest of its content.
* ``require_authorization`` — execution needs the owner's explicit authorization of *this* plan: the environment
  variable ``QEMX_HW_AUTHORIZE`` must equal the plan's digest (printed by ``plan``). A changed plan has another digest.
* ``execute`` — after authorization, a campaign ledger (``QEMX_CAMPAIGN_LEDGER``) and credentials from the
  environment, a saved account or a hidden prompt (never an argument), jobs are submitted one by one through
  ``campaign.submit_with_ledger`` (admission on the high estimate, no automatic retries, unknown submissions block
  further ones). With ``provider`` given (``campaign.MockProvider``) nothing leaves the machine — the rehearsal used by
  the tests. No job was submitted for 1.7.0.
"""
from __future__ import annotations

import hashlib
import json
import os

from .. import textio as TX

ENV_AUTH = "QEMX_HW_AUTHORIZE"
DEFAULTS = dict(ceiling_s=600.0, reserve_s=60.0, blocks=8, budget=20000,
                protocols=["RAW", "RO", "recommended", "SBS-B"], device="fake_fez", backend_name="ibm_fez")


class NotAuthorized(PermissionError):
    """Hardware execution without the owner's authorization of this exact plan."""


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def plan(cfg: dict, bundle_path: str | None = None, ledger=None, model=None) -> dict:
    """The hardware validation plan of a configuration's ``hardware`` section (see module text)."""
    from .. import campaign as CG
    from .. import device as DV
    from .. import hardware as HW
    from . import deploy as DP
    from . import protocols as PR
    from . import task as TK
    hw = dict(DEFAULTS, **(cfg.get("hardware") or {}))
    B = int(hw["budget"])
    bundle = DP.load_bundle(bundle_path) if bundle_path else None
    rows, jobs = [], []
    for t in hw.get("tasks") or []:
        t = dict(t)
        spec = TK.make_spec(t.pop("family"), int(t.pop("n")), device=hw["device"], **t)
        dep = DP.from_spec(spec, B)
        comp = dep.compile()
        names = []
        rec = DP.recommend(dep, bundle) if bundle is not None else None     # one call: the deployed policy (1.8.0)
        for p in hw["protocols"]:
            if p == "recommended":
                if rec is None:
                    continue
                names.append(("recommended", rec["choice"]))
            elif p == "SBS-B":
                if rec is None:
                    continue
                names.append(("SBS-B", rec["fallback"]))
            else:
                names.append((p, p))
        seen = {}
        for role, k in names:
            seen.setdefault(k, []).append(role)
        recipes = [PR.PORTFOLIO[k].recipe(spec) for k in seen]    # declared parameters (e.g. SV's checks) resolved
        hp = HW.make_plan(comp, dep.obs(), recipes, B, backend=DV.fake_backend(hw["device"]))
        labels = hp.labels()
        for idx, (k, roles) in enumerate(seen.items()):
            req = [r for r in hp.requests.values() if r.tag == idx]
            skipped = hp.skipped_idx.get(idx)
            # a merged request is measured once with the shares of every occurrence already added to its shots
            # (qemx.hardware.Request.multiplicity counts the occurrences; multiplying by it overcounted PEC runs)
            shots = sum(int(r.shots) for r in req)
            pf = ((rec or {}).get("preflight") or {}).get(k, {})
            rows.append(dict(task=spec.label, task_id=spec.task_id, protocol=k, roles=roles, B=B,
                             label=labels[idx] if idx < len(labels) else k, supported=skipped is None,
                             reason=skipped or "", circuits=len(req), shots=shots, blocks=int(hw["blocks"]),
                             preflight=pf.get("status"), preflight_level=pf.get("level"),
                             recommendation_status=(rec or {}).get("status"),
                             recommendation_accepted=(rec or {}).get("accepted")))
            if skipped is None and req:
                for b in range(int(hw["blocks"])):
                    jobs.append(("%s · %s · block %d" % (spec.label[:40], k, b + 1), shots, len(req)))
    m = model or CG.UsageModel()
    remaining = ledger.refresh().remaining_s() if ledger is not None else float(hw["ceiling_s"]) - float(hw["reserve_s"])
    cum, adm, jrows = 0.0, 0, []
    for label, shots, circ in jobs:
        est = CG.estimate_usage(shots, circ, 1, m)
        need = CG.reservation_s(est)
        fits = cum + need <= remaining
        if fits:
            cum, adm = cum + need, adm + 1
        jrows.append(dict(job=label, executions=shots, circuits=circ, low_s=est.low_s, central_s=est.central_s,
                          high_s=est.high_s, reserved_s=need, admitted=bool(fits)))
    content = dict(schema="qemx.selection.hardware-plan/1", budget=B, device_model=hw["device"],
                   backend=hw["backend_name"], blocks=int(hw["blocks"]), ceiling_s=float(hw["ceiling_s"]),
                   reserve_s=float(hw["reserve_s"]), runs=rows, jobs=jrows,
                   bundle=bundle.meta["version"] if bundle else None,
                   bundle_digest=bundle.meta.get("arrays_sha256") if bundle else None,
                   policy=DP.POLICY if bundle else None)
    out = dict(content, digest=_digest(content), remaining_s=remaining, reserved_s=cum, admitted_jobs=adm,
               total_jobs=len(jobs), fits=adm == len(jobs), executed=False,
               note="Plan only: nothing was submitted. Execution needs QEMX_HW_AUTHORIZE=<digest>, a campaign ledger "
                    "and credentials (environment, saved account or hidden prompt).")
    return out


def write(p: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    TX.write_json(os.path.join(out_dir, "hardware_plan.json"), p, indent=1)
    L = ["# Hardware validation plan (not executed)", "",
         "Digest `%s`. Budget B = %s shots per protocol run, %d blocks (independent deployments) per run; device "
         "model %s (stand-in for %s)." % (p["digest"][:16], f"{p['budget']:,}", p["blocks"], p["device_model"],
                                          p["backend"]),
         "", "Shared QPU ceiling %.0f s, reserve %.0f s: %.0f s available; the plan reserves %.0f s (high estimates); "
         "%d of %d jobs fit." % (p["ceiling_s"], p["reserve_s"], p["remaining_s"], p["reserved_s"], p["admitted_jobs"],
                                 p["total_jobs"]), "",
         "| task | protocol | roles | supported | circuits | shots per run | blocks |", "|---|---|---|---|---|---|---|"]
    for r in p["runs"]:
        L.append("| %s | %s | %s | %s | %d | %s | %d |" % (r["task"], r["protocol"], ", ".join(r["roles"]),
                                                             "yes" if r["supported"] else "no: " + r["reason"],
                                                             r["circuits"], f"{r['shots']:,}", r["blocks"]))
    L += ["", "To execute (owner only): `python -m qemx.selection hardware check --plan %s` verifies the plan; set "
          "`%s=%s`, a campaign ledger (`QEMX_CAMPAIGN_LEDGER`) and credentials (environment, saved account or hidden "
          "prompt), then submit through the approved IBM workflow with the ledger (`docs/selection/HARDWARE.md`). "
          "Neither `plan` nor `check` submits anything." % ("hardware_plan.json", ENV_AUTH, p["digest"])]
    TX.write_text(os.path.join(out_dir, "hardware_plan.md"), "\n".join(L) + "\n")
    return os.path.join(out_dir, "hardware_plan.json")


def require_authorization(p: dict, env=None) -> None:
    env = os.environ if env is None else env
    got = (env.get(ENV_AUTH) or "").strip()
    if not got:
        raise NotAuthorized("hardware execution needs the owner's authorization of this plan: set %s to its digest "
                            "(%s…)" % (ENV_AUTH, p["digest"][:16]))
    content = {k: v for k, v in p.items() if k in ("schema", "budget", "device_model", "backend", "blocks",
                                                   "ceiling_s", "reserve_s", "runs", "jobs", "bundle",
                                                   "bundle_digest")}
    if _digest(content) != p.get("digest"):
        raise NotAuthorized("the plan file was changed after it was made (its content does not match its digest)")
    if got != p["digest"]:
        raise NotAuthorized("%s authorizes another plan (%s… ≠ %s…)" % (ENV_AUTH, got[:16], p["digest"][:16]))


def execute(p: dict, ledger, run_job, env=None, campaign: str = "selection-1.7") -> list:
    """Submit the plan's admitted jobs through the campaign ledger (``run_job(label, shots, circuits)`` builds and
    submits one job: the IBM path or ``campaign.MockProvider`` in rehearsals). Refused without authorization."""
    from .. import campaign as CG
    require_authorization(p, env)
    CG.require_ledger(ledger, backend=p["backend"])
    CG.check_no_unresolved(ledger)
    import math
    jobs = [(j["job"], int(math.ceil(j["executions"] / max(j["circuits"], 1))), j["circuits"]) for j in p["jobs"]
            if j["admitted"]]                       # shots per circuit (rounded up: a conservative estimate)
    return CG.submit_with_ledger(ledger, jobs, run_job, campaign=campaign)
