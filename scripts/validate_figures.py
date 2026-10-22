"""Validate the figures of a run directory (1.7.0, prompt §9): a successful plotting command is not enough.

    python scripts/validate_figures.py RUN_DIR [--json OUT.json]

Checks, for every figure in RUN_DIR/figures:

* files — the PDF, SVG and PNG exist and are complete: the SVG parses as XML and contains text elements; the PDF
  starts with %PDF, ends with %%EOF, and (when poppler's ``pdfinfo``/``pdftotext`` are installed) has ≥ 1 page and
  extractable text containing the figure's data label; the PNG opens and verifies (Pillow);
* notes — the Markdown notes contain the caption, the interpretation, the methods box and the inputs;
* data — every source CSV named in the notes exists and has rows;
* numbers — plotted values equal the analysis tables they come from (Figure 4's matrix against
  ``analysis/pooled_<split>.csv``; Figure 5's operating point against ``analysis/nested_summary.csv``; Figure 3's
  ablation and composition against ``analysis/ablation.csv`` and ``noisecomp/analysis/regime_cells.csv``; Figure 1's
  catastrophic panel against ``analysis/failure_profile.csv``; the design-variability figure against
  ``analysis/design_variance.csv``; Table 3 against ``confirm/metrics_confirm.csv``).

Exit status 0 when every check passes. PDFs are not compared byte for byte across environments.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qemx import textio as TX  # noqa: E402


def _check_svg(path: str) -> list:
    errs = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return ["%s: SVG does not parse: %s" % (os.path.basename(path), e)]
    texts = [el for el in root.iter() if el.tag.endswith("text")]
    if not texts:
        errs.append("%s: SVG has no text elements (text converted to paths?)" % os.path.basename(path))
    return errs


def _check_pdf(path: str, label: str) -> list:
    errs = []
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"%PDF"):
        errs.append("%s: not a PDF header" % os.path.basename(path))
    if b"%%EOF" not in data[-1024:]:
        errs.append("%s: no %%%%EOF trailer (truncated?)" % os.path.basename(path))
    if shutil.which("pdfinfo") and shutil.which("pdftotext"):
        info = subprocess.run(["pdfinfo", path], capture_output=True, encoding="utf-8", errors="strict")
        m = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
        if info.returncode != 0 or not m or int(m.group(1)) < 1:
            errs.append("%s: pdfinfo failed or no pages" % os.path.basename(path))
        txt = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, encoding="utf-8",
                             errors="strict")
        norm = re.sub(r"\s+", " ", txt.stdout)
        if txt.returncode != 0 or not norm.strip():
            errs.append("%s: no extractable text" % os.path.basename(path))
        elif label and label.split(" (")[0] not in norm:
            errs.append("%s: the data label %r is not in the PDF text" % (os.path.basename(path), label[:40]))
    return errs


def _check_png(path: str) -> list:
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return []
    except Exception as e:  # noqa: BLE001 - reported
        return ["%s: PNG does not verify: %s" % (os.path.basename(path), e)]


def _close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _numbers(run: str) -> list:
    """Cross-checks of plotted numbers against the analysis tables."""
    import pandas as pd
    fdir, adir = os.path.join(run, "figures"), os.path.join(run, "analysis")
    out = []

    def rd(p):
        return pd.read_csv(p, encoding="utf-8") if os.path.exists(p) else None
    m = rd(os.path.join(fdir, "fig04_generalization_learning__matrix.csv"))
    if m is not None:
        for _, r in m.iterrows():
            t = rd(os.path.join(adir, "pooled_%s.csv" % r.split))
            t = t[(t.B.astype(str) == "all") & (t.policy == r.policy)]
            ok = len(t) == 1 and _close(t.mean_log_regret.iloc[0], r.mean_log_regret) and \
                int(t.base_circuits.iloc[0]) == int(r.base_circuits)
            out.append(("fig04 matrix %s %s" % (r.split, r.role), ok))
    s = rd(os.path.join(fdir, "fig05_reliability_abstention__summary.csv"))
    ns = rd(os.path.join(adir, "nested_summary.csv"))
    if s is not None and ns is not None and "policy" in s.columns and "mse" in s.columns:
        for _, r in s.iterrows():
            t = ns[ns.policy == r.policy]
            out.append(("fig05 %s mse" % r.policy, len(t) == 1 and _close(t.mse.iloc[0], r.mse)))
    a = rd(os.path.join(fdir, "fig03_feature_mechanisms__ablation.csv"))
    ab = rd(os.path.join(adir, "ablation.csv"))
    if a is not None and ab is not None:
        mm = a.merge(ab, on="variant", suffixes=("", "_t"))
        out.append(("fig03 ablation", len(mm) == len(ab) and all(_close(x, y) for x, y in
                                                                 zip(mm.diff_vs_all, mm.diff_vs_all_t))))
    c = rd(os.path.join(fdir, "fig03_feature_mechanisms__composition.csv"))
    nc = rd(os.path.join(run, "noisecomp", "analysis", "regime_cells.csv"))
    if c is not None and nc is not None and len(c):
        ok = True
        for _, r in c.iterrows():
            t = nc[(nc.family == r.family) & (nc.B == r.B) & (nc.noise_change.fillna("none") == r.change)]
            ok &= len(t) == 1 and t.reference.iloc[0] == r.reference and bool(t.unresolved.iloc[0]) == \
                bool(r.unresolved) and t.model_choice.iloc[0] == r.model_choice
        out.append(("fig03 composition cells", bool(ok)))
    f1 = rd(os.path.join(fdir, "fig01_budget_efficiency__catastrophic.csv"))
    fp = rd(os.path.join(adir, "failure_profile.csv"))
    if f1 is not None and fp is not None and "worst1_share" in f1.columns:
        ok = True
        for _, r in f1.iterrows():
            t = fp[(fp.policy == r.policy) & (fp.B.astype(str) == str(r.B))] if "B" in f1.columns else \
                fp[(fp.policy == r.policy) & (fp.B.astype(str) == "all")]
            ok &= len(t) >= 1 and _close(t.worst1_share.iloc[0], r.worst1_share)
        out.append(("fig01 catastrophic", bool(ok)))
    dv = rd(os.path.join(fdir, "supp_design_variability__variance.csv"))
    dvt = rd(os.path.join(adir, "design_variance.csv"))
    if dv is not None and dvt is not None:
        out.append(("supp_design_variability variance", dv.round(12).equals(dvt.round(12))))
    t3 = rd(os.path.join(run, "tables", "table3_policy_results.csv"))
    mc = rd(os.path.join(run, "confirm", "metrics_confirm.csv"))
    if t3 is not None and mc is not None and len(t3):
        mc = mc[mc.B.astype(str) == "all"]
        ok = True
        for _, r in t3.iterrows():
            t = mc[mc.policy == r.policy]
            ok &= len(t) == 1 and r.mean_log10_MSE.startswith("%.3f" % t.mean_log10_risk.iloc[0])
        out.append(("table3 against metrics_confirm", bool(ok)))
    return out


def validate(run: str) -> dict:
    fdir = os.path.join(run, "figures")
    names = sorted({f[:-3] for f in os.listdir(fdir) if f.endswith(".md") and f != "PENDING.md"})
    res = dict(run=run, figures={}, numbers=[], pending=None)
    if os.path.exists(os.path.join(fdir, "PENDING.md")):
        res["pending"] = TX.read_text(os.path.join(fdir, "PENDING.md"))
    for n in names:
        errs = []
        notes = TX.read_text(os.path.join(fdir, n + ".md"))
        for ext in ("pdf", "svg", "png"):
            if not os.path.exists(os.path.join(fdir, "%s.%s" % (n, ext))):
                errs.append("%s.%s missing" % (n, ext))
        label = "Simulated observations" if "Simulated observations" in notes else ""
        if os.path.exists(os.path.join(fdir, n + ".svg")):
            errs += _check_svg(os.path.join(fdir, n + ".svg"))
        if os.path.exists(os.path.join(fdir, n + ".pdf")):
            errs += _check_pdf(os.path.join(fdir, n + ".pdf"), label if not n.startswith("supp_runtime") else "")
        if os.path.exists(os.path.join(fdir, n + ".png")):
            errs += _check_png(os.path.join(fdir, n + ".png"))
        for key in ("**Caption.**", "**Interpretation.**", "**Methods box.**", "**Inputs.**"):
            if key not in notes:
                errs.append("%s.md lacks %s" % (n, key))
        for csv in re.findall(r"`(%s__[^`]+\.csv)`" % re.escape(n), notes):
            p = os.path.join(fdir, csv)
            if not os.path.exists(p):
                errs.append("%s missing" % csv)
            else:
                with open(p, encoding="utf-8") as fh:
                    lines = sum(1 for _ in fh)
                if lines < 2 and not csv.endswith(("software_failures.csv", "design_study.csv",
                                                   "composition.csv")):
                    errs.append("%s has no rows" % csv)
        res["figures"][n] = errs
    res["numbers"] = [dict(check=c, ok=bool(ok)) for c, ok in _numbers(run)]
    res["ok"] = all(not e for e in res["figures"].values()) and all(x["ok"] for x in res["numbers"])
    return res


def main(argv=None) -> int:
    TX.utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = validate(a.run)
    for n, errs in res["figures"].items():
        print("%-34s %s" % (n, "ok" if not errs else "; ".join(errs)))
    for x in res["numbers"]:
        print("%-34s %s" % (x["check"], "ok" if x["ok"] else "MISMATCH"))
    if res["pending"]:
        print("PENDING.md present:\n" + res["pending"])
    if a.json:
        TX.write_json(a.json, res, indent=1)
    print("all checks passed" if res["ok"] else "CHECKS FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
