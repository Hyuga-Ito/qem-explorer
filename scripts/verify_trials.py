"""Verify the per-trial arrays of ONE study against its own checksum manifest (1.9.0; second review N5).

Each trial-data attachment belongs to a study (``reports/TRIAL_ATTACHMENTS.json``). The 1.8.0 workflow verified every
manifest it found after unpacking one release's attachments, so a v1.8 attachment set failed on the v1.7 manifest
(347 files that were never in it). This checks exactly the requested study: every file its manifest lists must exist
and match. Missing or changed files fail (exit 1); nothing is skipped.

    python scripts/verify_trials.py reports/selection_v180
    python scripts/verify_trials.py --attachments qem-explorer-1.9.0-trials-1.zip qem-explorer-1.9.0-trials-2.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, "reports", "TRIAL_ATTACHMENTS.json")


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(study_dir: str) -> dict:
    man = os.path.join(study_dir, "TRIALS_SHA256SUMS")
    if not os.path.exists(man):
        return dict(study=study_dir, ok=False, error="no TRIALS_SHA256SUMS in %s" % study_dir)
    listed, missing, changed = 0, [], []
    with open(man, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            rel = rel.lstrip("*")
            listed += 1
            p = os.path.join(study_dir, rel)
            if not os.path.exists(p):
                missing.append(rel)
            elif _sha(p) != digest:
                changed.append(rel)
    return dict(study=study_dir, ok=not missing and not changed and listed > 0, listed=listed, missing=len(missing),
                changed=len(changed), examples=(missing + changed)[:5])


def studies_of(attachments) -> list:
    with open(MAP, encoding="utf-8") as f:
        m = json.load(f)
    out = []
    for a in attachments:
        name = os.path.basename(a)
        if name not in m["attachments"]:
            raise SystemExit("unknown attachment %s (known: %s)" % (name, ", ".join(sorted(m["attachments"]))))
        s = m["attachments"][name]["study"]
        if s not in out:
            out.append(s)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study", nargs="*", help="study directories (e.g. reports/selection_v180)")
    ap.add_argument("--attachments", nargs="*", default=None, help="attachment file names: verify their studies")
    a = ap.parse_args(argv)
    studies = list(a.study) + (studies_of(a.attachments) if a.attachments else [])
    if not studies:
        ap.error("name a study directory or the attachments")
    rc = 0
    for s in studies:
        d = s if os.path.isabs(s) else os.path.join(ROOT, s) if not os.path.isdir(s) else s
        r = verify(d)
        print(json.dumps(r))
        rc = rc or (0 if r["ok"] else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
