"""The 1.9.0 pilot's per-trial arrays (release attachment ``qem-explorer-1.9.0-trials-pilot.zip``; independent
verification V2-F4). Without the arrays these tests are skipped — unless ``QEMX_REQUIRE_TRIALS`` names ``pilot_v190``
(the trial-data workflow sets it), in which case missing arrays fail. With them: the arrays match the study's own
manifest, the labels directory is complete and bound to its run, its data digest is the one the freeze and the
completion report recorded, and the shipped labels table equals the table derived from the arrays."""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "reports", "pilot_v190")


def _arrays_present() -> bool:
    d = os.path.join(STUDY, "dev", "tasks")
    return os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))


@pytest.fixture(scope="module", autouse=True)
def arrays():
    if not _arrays_present():
        if "pilot_v190" in os.environ.get("QEMX_REQUIRE_TRIALS", ""):
            pytest.fail("QEMX_REQUIRE_TRIALS names pilot_v190 but its per-trial arrays are missing")
        pytest.skip("the per-trial arrays of pilot_v190 are not extracted (release attachment; "
                    "reports/TRIAL_ATTACHMENTS.json)")


def test_pilot_arrays_match_the_studys_own_manifest():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_trials", os.path.join(ROOT, "scripts", "verify_trials.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = m.verify(STUDY)
    assert r["ok"] and r["listed"] == 54 and r["missing"] == r["changed"] == 0, r


def test_pilot_labels_are_complete_bound_and_the_data_the_freeze_recorded():
    from qemx import textio as TX
    from qemx.selection import integrity as IN
    c = IN.completion(os.path.join(STUDY, "dev"))
    assert c["bound"] and c["complete"] and c["completed"] == 54 and not c["corrupted"], IN.describe(c)
    digest = IN.data_digest(os.path.join(STUDY, "dev"), c["completed_ids"])
    assert digest == TX.read_json(os.path.join(STUDY, "freeze.json"))["dev_data_digest"]
    assert digest == TX.read_json(os.path.join(STUDY, "completion.json"))["dev"]["data_digest"]


def test_pilot_labels_table_equals_the_table_derived_from_the_arrays():
    import pandas as pd
    from qemx.selection import labels as LB
    derived = LB.summarize(os.path.join(STUDY, "dev"))
    shipped = pd.read_csv(os.path.join(STUDY, "analysis", "labels.csv"), encoding="utf-8")
    key = ["task_id", "protocol", "B"]
    m = derived[key + ["mse_eval", "mse_sel"]].merge(shipped[key + ["mse_eval", "mse_sel"]], on=key,
                                                     suffixes=("", "_shipped"), how="outer", indicator=True)
    assert (m["_merge"] == "both").all() and len(m) == 54 * 17 * 4
    for col in ("mse_eval", "mse_sel"):
        a, b = m[col].astype(float).to_numpy(), m[col + "_shipped"].astype(float).to_numpy()
        same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-12, atol=0)
        assert same.all(), (col, int((~same).sum()))
