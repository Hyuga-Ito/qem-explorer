"""The final holdout is frozen, disjoint from every development manifest by base circuit, and evaluated by a model
trained only on development data."""
import glob
import hashlib
import os

import numpy as np
import pytest

from qemx.phase import manifest as MF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT = os.path.join(ROOT, "manifests", "direction_A_v2_final_holdout.yaml")
FROZEN_SHA256 = "60fc000c95025825f139d7b21b68d8676b4a0989fec96a2bbdb36da6b6be08c3"
DEVELOPMENT = ("direction_A_v2_pilot.yaml", "direction_A_v2_study.yaml", "direction_A_v1.yaml",
               "direction_A_nil_v1.yaml", "direction_A_zne_folding_v1.yaml", "smoke.yaml")


def test_final_holdout_is_frozen():
    with open(HOLDOUT, "rb") as f:
        assert hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest() == FROZEN_SHA256, \
            "the frozen final-holdout manifest changed: a changed holdout is a new development set (FINAL_HOLDOUT.md)"


def test_final_holdout_shares_no_base_circuit_with_development_manifests():
    out = MF.check_disjoint(MF.load(HOLDOUT), *[MF.load(os.path.join(ROOT, "manifests", m)) for m in DEVELOPMENT])
    assert out["base_circuits"] == 23
    pilot = MF.load(os.path.join(ROOT, "manifests", "direction_A_v2_pilot.yaml"))
    with pytest.raises(ValueError, match="also in development"):
        MF.check_disjoint(pilot, MF.load(os.path.join(ROOT, "manifests", "direction_A_v2_pilot.yaml")))


@pytest.mark.slow
def test_holdout_evaluation_trains_on_development_cells_only(tmp_path):
    """Exercised on the bundled pilot records: explore cells as development, confirm cells as a stand-in holdout."""
    import shutil
    from qemx import study as SD
    from qemx.phase import store as ST
    run = tmp_path / "pilot"
    shutil.copytree(os.path.join(ROOT, "data", "pilot_v2"), run)
    ST.collect(str(run), manifest_path=str(run / "manifest.yaml"))
    S = SD.load(str(run), str(run / "manifest.yaml"))
    dev = SD.subset(S, S["cells"].loc[S["cells"]["split"] == "explore", "cell_id"])
    hold = SD.subset(S, S["cells"].loc[S["cells"]["split"] == "confirm", "cell_id"])
    ev = SD.evaluate_holdout(dev, hold)
    assert ev["meta"]["holdout_base_circuits"] == 7 and ev["meta"]["dev_base_circuits"] == 18
    assert not ev["meta"]["problems"] and ev["summary"]["rows"] == ev["meta"]["holdout_rows"]
    with pytest.raises(ValueError, match="both the development and the holdout"):
        SD.evaluate_holdout(S, hold)
