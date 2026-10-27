"""Reproduce the bundled pilot report from its primary records, without running any simulation.

``data/pilot_v2/`` holds the 75 primary cell records of the main-study pilot (record schema qemx.phase.cell/3,
written on 2026-10-17 by the 1.5.0 development tree) with their manifest and SHA-256 sums. This script

1. checks every record against ``data/pilot_v2/SHA256SUMS`` (a changed record stops it);
2. copies the records to a fresh working directory (the bundled folder is never written);
3. derives the tables from the records (``qemx.phase.run collect``: ledger, winners, levels);
4. writes the figure report (``qemx.study report``: Figures 1–4 and 6, tables, methods.json) into ``--out``;
5. with ``--compare reports/pilot``, checks that the regenerated Figure 3 table (paired change against raw) and the
   regime table equal the bundled ones.

Figure 5 needs the discrepancy rehearsal's raw analysis directory, which is not bundled (it is 15 MB and an
emulation); pass ``--rehearsal DIR`` from ``python scripts/rehearse_divergence.py`` to include it.

    python scripts/reproduce_pilot_report.py --out pilot_report --compare reports/pilot
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "pilot_v2")


def verify(data: str = DATA) -> int:
    n = 0
    with open(os.path.join(data, "SHA256SUMS"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            digest, rel = line.split(None, 1)
            rel = rel.strip()
            with open(os.path.join(data, rel), "rb") as g:             # line endings may be converted on checkout
                got = hashlib.sha256(g.read().replace(b"\r\n", b"\n")).hexdigest()
            if got != digest:
                raise SystemExit("%s differs from its recorded SHA-256 — the primary records were changed" % rel)
            n += 1
    return n


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="report directory to write")
    ap.add_argument("--rehearsal", default=None, help="a scripts/rehearse_divergence.py output directory (Figure 5)")
    ap.add_argument("--compare", default=None, help="a bundled report directory to compare tables with")
    a = ap.parse_args(argv)
    n = verify()
    print("%d primary files match their SHA-256 sums" % n)
    from qemx.phase import store as ST
    from qemx import study as SD
    with tempfile.TemporaryDirectory() as td:
        run = os.path.join(td, "pilot_v2")
        shutil.copytree(DATA, run)
        ST.collect(run, manifest_path=os.path.join(run, "manifest.yaml"))
        res = SD.report(run, os.path.join(run, "manifest.yaml"), a.out, a.rehearsal)
    print("report: %s" % res["report"])
    if a.compare:
        import pandas as pd
        for name, keys in (("paired_improvement.csv", ["method"]), ("regimes.csv", ["cell_id", "B"])):
            new = pd.read_csv(os.path.join(a.out, "tables", name), encoding="utf-8").sort_values(keys).reset_index(drop=True)
            old = pd.read_csv(os.path.join(a.compare, "tables", name), encoding="utf-8").sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(new, old, check_exact=False, rtol=1e-9, atol=1e-12)
            print("%s: identical to %s" % (name, a.compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
