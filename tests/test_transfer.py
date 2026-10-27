"""qemx.transfer: the hardware transfer test's power, contrasts and analysis, and the local emulation of the hardware
path agreeing with the device model that makes the predictions (review N44)."""
import math

import numpy as np
import pytest

from qemx import transfer as TR


def _pred():
    m = lambda b, sd: dict(bias=b, sd=sd, rmse=math.hypot(b, sd))                   # noqa: E731
    return {"c1": dict(role="primary", ideal=1.0, methods=dict(RAW=m(-0.2, 0.01), RO=m(-0.15, 0.01),
                                                               **{"ZNE-L": m(-0.02, 0.02), "PEC": m(-0.03, 0.04)}))}


def test_power_and_contrasts_follow_the_prediction():
    C = TR.contrasts(_pred())
    prim = [x for x in C if x["kind"] == "primary"][0]
    assert (prim["better"], prim["worse"]) == ("ZNE-L", "RO") and prim["d_mean"] > 0
    assert TR.power(prim["d_mean"], prim["d_sd"], 4) > 0.99
    assert TR.power(0.0, 1.0, 100) == pytest.approx(0.05, abs=1e-3)       # no effect: the test's size
    assert TR.power(-1.0, 1.0, 10) < 0.05                                  # wrong direction


def test_analysis_confirms_a_real_difference_and_flags_a_biased_protocol():
    rng = np.random.default_rng(1)
    pred = _pred()
    rows = []
    for b in range(8):
        for m, (bias, sd) in dict(RAW=(-0.2, 0.01), RO=(-0.15, 0.01), **{"ZNE-L": (-0.02, 0.02),
                                                                         "PEC": (0.10, 0.04)}).items():
            rows.append(dict(block=b, circuit="c1", method=m, estimate=0.0, error=bias + sd * rng.normal(),
                             ideal=1.0, status="ok"))
    an = TR.analyze(rows, pred)
    prim = [x for x in an["contrasts"] if x["kind"] == "primary"][0]
    assert prim["confirmed"] and prim["p_holm"] < 0.01
    by = {r["method"]: r for r in an["per_method"]}
    assert abs(by["ZNE-L"]["bias_z"]) < 4 and by["PEC"]["bias_z"] > 4    # PEC's mean error is not the predicted one
    assert "Per protocol" in TR.report_markdown(an, None, "test")


@pytest.mark.slow
def test_local_emulation_matches_the_predicting_device_model():
    """The local emulation of the hardware path must have the predictor's physics: idle relaxation in the ALAP
    schedule and each qubit's asymmetric readout. 1.5.0 omitted both (QFT-4 raw error 0.143 vs predicted 0.179)."""
    from qemx import device as D
    from qemx import hardware as H
    from qemx.recipes import compare, summary
    design = dict(TR.DESIGN, budget=20000, methods=[dict(id="RAW", estimator="raw")],
                  circuits=[TR.DESIGN["circuits"][0]])
    (c, r), = TR.resolve(design)
    plan = TR.block_plan(design, r, D.fake_backend("fake_fez"), 0, 0)
    ideal, outs, _ = compare(r.compiled, r.noise, r.observable, plan.effective_recipes(), "fake_fez", 20000,
                             r.symmetry, seed=0)
    pred = summary(outs[0], ideal, 20000, 1.0)
    errs = [TR.block_errors(design, plan, H.run_local(plan, "fake_fez", seed=s), s, c["name"])[0]["error"]
            for s in range(3)]
    se = pred["sd"] / math.sqrt(len(errs))
    assert abs(np.mean(errs) - pred["bias"]) < 4 * se + 1e-3, (np.mean(errs), pred["bias"], se)


def test_run_and_collect_ibm_with_a_mocked_provider(tmp_path, monkeypatch):
    """The guarded CLI path end to end with a mock provider: predictions frozen before the first job, every job
    admitted by the ledger and recorded, a resumed run sends nothing twice, collection replays the saved plans."""
    import json
    import os
    import sys
    import warnings
    import qiskit_ibm_runtime as R
    from qemx import campaign as CG
    from qemx import device as D
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import hardware_transfer as HT
    import test_spending as TS
    warnings.simplefilter("ignore")
    remote, made, svc = TS._mock_runtime(monkeypatch)
    fake = D.fake_backend("fake_fez")
    real = TS._RealLike(fake)
    real.name = "ibm_mockfez"
    svc.backend = lambda name: real
    monkeypatch.setattr(R, "QiskitRuntimeService", lambda *a, **k: svc)
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "not-a-real-token")
    monkeypatch.setattr(TR, "DESIGN", dict(TR.DESIGN, circuits=TR.DESIGN["circuits"][:1],
                                           methods=TR.DESIGN["methods"][:2]))
    out = tmp_path / "run"
    with pytest.raises(Exception):                                   # no ledger file: nothing is submitted
        HT.main(["run-ibm", "--out", str(out), "--ledger", str(tmp_path / "missing.json"), "--blocks", "1"])
    assert not remote
    led = CG.Ledger.create(str(tmp_path / "ledger.json"), ceiling_s=600, reserve_s=60)
    assert HT.main(["run-ibm", "--out", str(out), "--ledger", led.path, "--blocks", "2"]) == 0
    jobs = json.load(open(out / "jobs.json", encoding="utf-8"))
    pred = json.load(open(out / HT.FROZEN, encoding="utf-8"))                           # frozen_predictions.json (1.5.1 re-review)
    assert sorted(jobs["blocks"]) == ["0/qft4", "1/qft4"] and pred["backend"] == "ibm_mockfez"
    n = len(remote)
    assert n == len(CG.Ledger.load(led.path).entries) and n > 0
    assert HT.main(["run-ibm", "--out", str(out), "--ledger", led.path, "--blocks", "2"]) == 0
    assert len(remote) == n                                          # resume: nothing sent twice
    assert HT.main(["collect-ibm", "--out", str(out), "--ledger", led.path]) == 0
    an = json.load(open(out / "analysis.json", encoding="utf-8"))
    import pandas as pd
    be = pd.read_csv(out / "block_errors.csv", encoding="utf-8")
    assert sorted(set(be["method"])) == ["RAW", "RO"] and len(be) == 4         # every protocol of every block
    # the mock returns |0…0⟩ for every circuit, so the readout calibration is singular: RO must report that failure
    # (status), not drop the row; RAW is analyzed
    assert set(be.loc[be["method"] == "RO", "status"]) != {"ok"}
    assert [r["method"] for r in an["per_method"]] == ["RAW"] and an["per_method"][0]["blocks"] == 2
    assert "not-a-real-token" not in "".join(open(p, encoding="utf-8").read()
                                             for p in [out / "jobs.json", out / HT.FROZEN, out / "report.md"])
