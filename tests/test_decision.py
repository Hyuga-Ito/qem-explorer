"""qemx.decision: the diagnostic-conditioned policy is evaluated with the diagnostic's cost inside, as an expectation
over the flag; the flag probability is calibrated under the null."""
import math

import numpy as np
import pytest

from qemx import decision as DC

CANDS = ["RAW", "RO", "PEC"]


def _case(p_flag, true_B, true_BD, model_B, model_BD, B=1000, D=100, mismatch=True):
    return DC.Case("c", "g", "f", "s", mismatch, "held-out", {B: true_B, B - D: true_BD}, {B: model_B, B - D: model_BD},
                   {D: p_flag})


def test_expected_loss_is_a_mixture_over_the_flag_with_the_diagnostics_cost():
    # the model wrongly prefers PEC; the truth makes PEC bad and RO good
    c = _case(0.7, dict(RAW=2.0, RO=1.0, PEC=3.0), dict(RAW=2.1, RO=1.05, PEC=3.1),
              dict(RAW=2.0, RO=1.5, PEC=0.5), dict(RAW=2.1, RO=1.6, PEC=0.55))
    o = DC.case_outcomes(c, 1000, 100, CANDS, "RO")
    assert o["model_choice"] == "PEC" and o["oracle"] == "RO"
    assert o["regret_model"] == pytest.approx(2.0)
    assert o["regret_diag"] == pytest.approx(0.7 * 0.05 + 0.3 * 2.1)          # fallback and choice at B − D
    assert o["harm_model"] == 1.0 and o["harm_diag"] == pytest.approx(0.3)
    # under the null a perfect model loses only the diagnostic's cost (and its false flags)
    n = _case(0.01, dict(RAW=2.0, RO=1.5, PEC=0.5), dict(RAW=2.1, RO=1.6, PEC=0.55),
              dict(RAW=2.0, RO=1.5, PEC=0.5), dict(RAW=2.1, RO=1.6, PEC=0.55), mismatch=False)
    on = DC.case_outcomes(n, 1000, 100, CANDS, "RO")
    assert on["regret_model"] == 0 and on["regret_diag"] == pytest.approx(0.01 * 2.2 + 0.99 * 0.1)


def test_flag_probability_has_size_alpha_under_the_null_and_power_under_a_shift():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    null = DC.flag_probability(p, p, 500, 0.05, rng=1)
    assert abs(null["p_flag"] - 0.05) < 0.02 and null["realized_size"] <= 0.05 + 1e-9
    alt = DC.flag_probability(p, [0.25, 0.3, 0.25, 0.2], 500, 0.05, rng=2)
    assert alt["p_flag"] > 0.9 and alt["true_tvd"] == pytest.approx(0.15)


def test_choose_D_uses_only_the_exploration_rows_and_summaries_carry_counts():
    rows = []
    for D, (rn, rm) in {100: (0.1, 0.5), 400: (0.3, 0.2)}.items():
        for mm, r in ((False, rn), (True, rm)):
            rows.append(dict(D=D, mismatch=mm, regret_diag=r, regret_model=0.0, split="explore", group="g%d" % D,
                             scenario="s", p_flag=0.1, coverage_diag=0.9, regret_fallback=1.0, harm_model=0.0,
                             harm_diag=0.0, harm_fallback=0.0))
    assert DC.choose_D(rows, (100, 400)) == 400
    s = DC.summarize([r for r in rows if r["D"] == 100])
    assert sum(x["cases"] for x in s) == 2 and all(math.isnan(x["diff_lo"]) for x in s)   # one circuit: no CI
