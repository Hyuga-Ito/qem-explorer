"""The bundled pilot report is reproducible from the bundled primary records alone (no simulation): checksums,
derived tables and the Figure 3 / regime tables (review 6.7)."""
import os
import subprocess
import sys

import pytest
from qemx.textio import child_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_primary_records_match_their_checksums():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import reproduce_pilot_report as RP
    assert RP.verify() == 76


@pytest.mark.slow
def test_pilot_report_tables_are_reproduced_from_the_records(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "reproduce_pilot_report.py"), "--out",
                          str(tmp_path / "rep"), "--compare", os.path.join(ROOT, "reports", "pilot")],
                         capture_output=True, text=True, timeout=1800, cwd=str(tmp_path), encoding="utf-8", env=child_env())
    assert out.returncode == 0, out.stderr[-2000:]
    assert "paired_improvement.csv: identical" in out.stdout and "regimes.csv: identical" in out.stdout
