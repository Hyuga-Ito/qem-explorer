"""PEC 1.5.1 — measured replay variance (N31) and information regimes (review §5.3, §5.5).

N31  a PEC replay on measured counts with s > 1 shots per sampled circuit must not count the within-circuit shot
     noise twice (checked over repeated complete protocols);
§5.5 PEC's noise model and the hardware plan's allocation carry their information regime (oracle /
     supplied-calibration / learned-with-cost) without changing any number.
"""
import math

import numpy as np
import pytest

from qemx import budget as BG
from qemx import hardware as H
from qemx import observables as OBS
from qemx.circuit import parse
from qemx.compile import build
from qemx.methods import REGISTRY, Context, Recipe, Runner
from qemx.noise import NoiseSpec


def test_pec_replay_of_measured_counts_does_not_double_count_shot_noise():
    """N31. Over repeated complete protocols (fresh sampled circuits, fresh shot noise) N·Var(estimate) is the
    protocol's per-shot variance; the replay of measured counts must report it (the baseline added the measured
    circuit means' own shot noise again: about +30 % here)."""
    comp = build(parse("ry(1.1) 0; sx 0", 1))
    obs = OBS.z(0, 1)
    B, s = 256, 32
    rc = Recipe.make("pec", construction="inverse", p1=0.3, p2=0.1, instances=B // s, shots_per_circuit=s)
    rng = np.random.default_rng(7)
    ideal = H.HardwareRunner().ideal
    means, rep = [], []
    for seed in range(500):
        plan = H.make_plan(comp, obs, [rc], B, None, seed, None)
        assert plan.alloc[0]["spc"] == s
        freqs = {}
        for k, r in plan.requests.items():
            p = np.clip(ideal.run(r.spec, NoiseSpec(), r.mm, ideal=True, backend="numpy").probs, 0, None)
            freqs[k] = rng.multinomial(r.shots, p / p.sum()) / r.shots
        R = H._replay(plan, H.HardwareRunner(freqs), 0, rc)
        means.append(R.mean)
        rep.append(R.var1)
    emp = B * np.var(means, ddof=1)
    ratio = float(np.mean(rep)) / emp
    assert abs(ratio - 1) < 4 * math.sqrt(2 / (len(means) - 1)) + 0.03, ratio


def _bell_on_device():
    from qemx import device as D, templates as T
    from qemx.compile import Compiled, with_io
    tpl = T.build("bell")
    b = D.fake_backend("fake_perth")
    spec, mm, phys, info = D.compile_for_target(with_io(tpl.spec), b, D.good_target_for(b), "FakePerth")
    return Compiled(with_io(tpl.spec), spec, mm, info), tpl, b


C2 = "ry(1.1) 0; cx 0 1; rz(0.7) 1; ry(0.4) 1; cx 0 1"


MILD = NoiseSpec(p1=0.01, p2=0.05, ro01=0.05, ro10=0.08)


def test_information_regimes_are_labelled_without_changing_numbers():
    comp = build(parse(C2, 2))
    obs = OBS.z(0, 2)
    out = {}
    for label, p in (("blank", {}), ("typed", dict(p1=0.01, p2=0.05)), ("learned", dict(model="learned"))):
        rc = Recipe.make("pec", **p)
        r = REGISTRY["pec"].run(Context(comp, MILD, obs, rc, Runner(), seed=1), **rc.p)
        out[label] = r
    assert out["blank"].details["information_regime"] == BG.ORACLE and "oracle-noise-model" in out["blank"].flags
    assert out["learned"].details["information_regime"] == BG.ORACLE
    assert out["typed"].details["information_regime"] == BG.SUPPLIED and not out["typed"].flags
    # the label does not change the numbers: blank rates equal typing the true rates in
    assert out["blank"].mean == pytest.approx(out["typed"].mean, abs=1e-12)
    assert out["blank"].var1 == pytest.approx(out["typed"].var1, abs=1e-12)
    assert BG.account(out["typed"], 4096)["allocation_information"] == BG.ORACLE
    comp_d, tpl, b = _bell_on_device()
    plan = H.make_plan(comp_d, tpl.observable, [Recipe.make("pec"), Recipe.make("raw", readout="tensored")], 2000,
                       tpl.symmetry, 0, b)
    assert plan.alloc[0]["information"]["noise_model"] == BG.SUPPLIED
    assert plan.alloc[1]["information"]["readout_calibration"] == BG.LEARNED
    assert plan.alloc[0]["information_regime"] == BG.SUPPLIED
    acc = BG.account(out["typed"], 2000, alloc=dict(N=2000, s=0, information_regime=plan.alloc[0]["information_regime"]))
    assert acc["allocation_information"] == BG.SUPPLIED
    assert acc["executions"] == 2000 and acc["distinct_circuits"] == 1 and acc["circuit_instances"] >= 1
