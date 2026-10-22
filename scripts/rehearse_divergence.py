#!/usr/bin/env python3
r"""End-to-end rehearsal of the divergence study without hardware (EMULATION, not hardware evidence).

The rehearsal runs the *same* path as a hardware campaign — the manifest, ``plan`` on the device's fake twin, the
counts files, ``analyze`` — but the observations come from emulated ground truths:

* **null**: the truth *is* the primary predictor's model (``--truth depolarizing``), so every flagged divergence is a
  false positive. Several seeds give the false-positive rate of the unadjusted |z| > 3 flag, of the per-test p < α
  and of the Holm-adjusted family-wise flag, plus the calibration of z (mean ≈ 0, sd ≈ 1).
* **alternatives**: the same truth plus a controlled, known unmodeled effect (``--alt crosstalk=0.05``,
  ``--alt scale=1.5``, ``--alt coherent=0.3``, ``--alt t2=40``, ``--alt hotspot=<qubit>``). The share of circuits
  flagged is the detection power at this budget, per method.

It also reports the planned QPU usage of the manifest under the planner's conservative per-shot cost
(``qemx.campaign``), so the rehearsal and the hardware preflight use one plan.

    python scripts/rehearse_divergence.py manifests/direction_A_divergence_v2_qpu10.yaml --out data/rehearsal_qpu10
    python scripts/rehearse_divergence.py manifests/direction_A_divergence_v2_qpu10.yaml --out data/reh \
        --seeds 3 --alt crosstalk=0.05 --alt scale=1.5 --workers 2

Outputs (``--out``): ``plan.json`` and ``obs/<tag>/`` (as for hardware), ``analysis/<tag>/tables``, and
``rehearsal.json`` + ``rehearsal.md`` (rates with binomial standard errors, every row labeled EMULATION).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _run(args, log):
    cmd = [sys.executable, os.path.join(HERE, "dirA_divergence.py")] + [str(a) for a in args]
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ %s\n" % " ".join(cmd[1:]))
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit("step failed (exit %d): %s — see %s" % (p.returncode, " ".join(map(str, args)), log))


def _rate(k: int, n: int) -> dict:
    r = k / n if n else math.nan
    return dict(count=int(k), n=int(n), rate=r, se=math.sqrt(r * (1 - r) / n) if n else math.nan)


def summarize(table: str, predictor: str, alpha: float = 0.05) -> pd.DataFrame:
    D = pd.read_csv(table, encoding="utf-8")
    d = D[(D["status"] == "ok") & (D["predictor"] == predictor)]
    rows = []
    for m, g in d.groupby("method"):
        z = g["z"].to_numpy(float)
        z = z[np.isfinite(z)]
        rows.append(dict(method=m, circuits=len(g), z_mean=float(np.mean(z)) if len(z) else math.nan,
                         z_sd=float(np.std(z, ddof=1)) if len(z) > 1 else math.nan,
                         flag_abs_z_gt_3=int((np.abs(z) > 3).sum()),
                         flag_p_lt_alpha=int((g["p_value"] < alpha).sum()),
                         flag_holm=int(g["significant"].sum()),
                         flag_holm_worse=int((g["significant"] & (g["z"] > 0)).sum())))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2, help="independent null emulations")
    ap.add_argument("--alt", action="append", default=None, help="alternative injections (repeatable)")
    ap.add_argument("--predictor", default="depolarizing", help="the primary predictor; the null truth is its model")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    alts = a.alt if a.alt is not None else ["crosstalk=0.05"]
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "rehearsal.log")
    if not os.path.exists(os.path.join(a.out, "plan.json")):
        _run(["plan", a.manifest, "--out", a.out], log)
    ana = ["--predictors", a.predictor, "--workers", a.workers, "--no-figures"] + (
        ["--replicates", a.replicates] if a.replicates else [])
    results = []
    for s in range(a.seeds):
        tag = "null%d" % s
        if not os.path.exists(os.path.join(a.out, "obs", tag, "counts.json")):
            _run(["emulate", a.out, "--truth", a.predictor, "--tag", tag, "--seed", 1000 + s, "--workers", a.workers],
                 log)
        _run(["analyze", a.out, "--tag", tag, "--out", os.path.join(a.out, "analysis", tag), "--seed", s] + ana, log)
        S = summarize(os.path.join(a.out, "analysis", tag, "tables", "divergence.csv"), a.predictor, a.alpha)
        S.insert(0, "scenario", "null")
        S.insert(1, "tag", tag)
        results.append(S)
    for k, inj in enumerate(alts):
        tag = "alt%d_%s" % (k, inj.replace("=", "_").replace(".", "p"))
        if not os.path.exists(os.path.join(a.out, "obs", tag, "counts.json")):
            _run(["emulate", a.out, "--truth", a.predictor, "--inject", inj, "--tag", tag, "--seed", 2000 + k,
                  "--workers", a.workers], log)
        _run(["analyze", a.out, "--tag", tag, "--out", os.path.join(a.out, "analysis", tag), "--seed", 100 + k] + ana,
             log)
        S = summarize(os.path.join(a.out, "analysis", tag, "tables", "divergence.csv"), a.predictor, a.alpha)
        S.insert(0, "scenario", "alternative: " + inj)
        S.insert(1, "tag", tag)
        results.append(S)
    R = pd.concat(results, ignore_index=True)
    R.to_csv(os.path.join(a.out, "rehearsal_rows.csv"), index=False, encoding="utf-8")
    null = R[R["scenario"] == "null"]
    n = int(null["circuits"].sum())
    fp = dict(abs_z_gt_3=_rate(int(null["flag_abs_z_gt_3"].sum()), n),
              p_lt_alpha=_rate(int(null["flag_p_lt_alpha"].sum()), n),
              holm=_rate(int(null["flag_holm"].sum()), n),
              families_with_any_holm_flag=_rate(int((null["flag_holm"] > 0).sum()), len(null)))
    power = {}
    for sc, g in R[R["scenario"] != "null"].groupby("scenario"):
        power[sc] = {m: dict(holm_worse=_rate(int(r["flag_holm_worse"]), int(r["circuits"])),
                             abs_z_gt_3=_rate(int(r["flag_abs_z_gt_3"]), int(r["circuits"])))
                     for m, r in g.set_index("method").iterrows()}
    usage = None
    try:
        from qemx import campaign as CG
        with open(os.path.join(a.out, "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
        pf = CG.preflight(plan)                           # the same job split and model as the hardware preflight
        usage = dict(pf["total"], fits_600s_policy=pf["fits"], jobs=pf["total_jobs"])
    except Exception as e:                                # noqa: BLE001 - the estimate is a convenience here
        usage = dict(error=str(e)[:200])
    out = dict(label="EMULATION — not hardware evidence", manifest=os.path.abspath(a.manifest),
               predictor=a.predictor, seeds=a.seeds, alternatives=alts, alpha=a.alpha, false_positive=fp,
               z_calibration=dict(mean=float(np.average(null["z_mean"], weights=null["circuits"])),
                                  sd=float(np.average(null["z_sd"], weights=null["circuits"]))),
               power=power, planned_usage=usage)
    with open(os.path.join(a.out, "rehearsal.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    md = ["# Divergence rehearsal — EMULATION (not hardware evidence)", "",
          "Manifest `%s`; primary predictor `%s`; %d null seed(s); alternatives: %s." % (
              os.path.basename(a.manifest), a.predictor, a.seeds, ", ".join(alts)), "",
          "## False positives (truth = the predictor's own model)", "",
          "| flag | flagged / tests | rate ± SE |", "|---|---|---|"]
    for k, v in fp.items():
        md.append("| %s | %d / %d | %.4f ± %.4f |" % (k, v["count"], v["n"], v["rate"], v["se"]))
    md += ["", "z under the null: mean %.3f, sd %.3f (calibrated: 0 and 1; sd < 1 is conservative)." % (
        out["z_calibration"]["mean"], out["z_calibration"]["sd"]), "", "## Detection power (Holm, predicted > observed)",
        "", "| alternative | method | flagged / circuits | rate ± SE |", "|---|---|---|---|"]
    for sc, ms in power.items():
        for m, v in ms.items():
            h = v["holm_worse"]
            md.append("| %s | %s | %d / %d | %.3f ± %.3f |" % (sc, m, h["count"], h["n"], h["rate"], h["se"]))
    if usage and "error" not in usage:
        md += ["", "Planned QPU usage of this manifest (conservative planner, qemx.campaign): %.0f–%.0f s "
               "(central %.0f s)." % (usage["low_s"], usage["high_s"], usage["central_s"])]
    with open(os.path.join(a.out, "rehearsal.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
