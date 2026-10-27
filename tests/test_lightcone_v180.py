"""1.8.0 (review finding F10): what the light-cone exposure does and does not describe.

lam_lc sums Pauli-*twirled* fault probabilities. Two coherent Z rotations (θ, θ) and (θ, −θ) on |+⟩ have the same
twirled fault probabilities sin²(θ/2) at each location, so the same exposure; but the first reinforces (net rotation
2θ, ⟨X⟩ = cos 2θ) and the second cancels (⟨X⟩ = 1). The exposure is a model descriptor under the independent
twirled-location model, not a sufficient statistic for coherent noise (twirling the two locations would make both
pairs the same Pauli channel, which is exactly the information the scalar keeps). These tests run on both NumPy SIMD paths in CI (floating-point portability step)."""
import math

import numpy as np
import pytest


def _circ(theta1, theta2, observable_basis="X"):
    from qemx.circuit import CircuitSpec, Op
    ops = [Op("h", (0,), (), "gate", ""),
           Op("zphase", (0,), (float(theta1),), "noise", ""),
           Op("id", (0,), (), "gate", ""),
           Op("zphase", (0,), (float(theta2),), "noise", "")]
    if observable_basis == "X":
        ops.append(Op("h", (0,), (), "gate", ""))
    return CircuitSpec(1, tuple(ops), "coherent pair")


def _expect_z(spec, twirl="none"):
    from qemx.engine import simulate
    from qemx.noise import NoiseSpec
    r = simulate(spec, NoiseSpec(), "numpy", measure_map={0: 0}, twirl=twirl)
    p = r.probs
    return float(p[0] - p[1])


@pytest.mark.parametrize("theta", [0.05, 0.3, 0.7])
def test_matched_exposure_reinforcing_vs_canceling_coherent_errors(theta):
    from qemx.noise import NoiseSpec
    from qemx.observables import z
    from qemx.phase.lightcone import calculate_lambda_lc
    plus, minus = _circ(theta, theta), _circ(theta, -theta)
    obs = z(0, 1)
    la = calculate_lambda_lc(plus, NoiseSpec(), obs, {0: 0})
    lb = calculate_lambda_lc(minus, NoiseSpec(), obs, {0: 0})
    want = 2 * math.sin(theta / 2) ** 2
    assert la.lam_lc == pytest.approx(want, rel=1e-12) and lb.lam_lc == pytest.approx(la.lam_lc, rel=1e-12)
    assert la.n_loc_lc == lb.n_loc_lc == 2
    # ⟨X⟩ of the actual (untwirled) circuits: reinforcing vs canceling
    assert _expect_z(plus) == pytest.approx(math.cos(2 * theta), abs=1e-12)
    assert _expect_z(minus) == pytest.approx(1.0, abs=1e-12)
    assert abs(_expect_z(plus) - _expect_z(minus)) > 1e-3        # same exposure, different error


def test_the_observable_orientation_matters_at_equal_exposure():
    """A Z-basis readout of the same states is insensitive to both rotations (the error commutes with the
    observable): equal exposure, zero error — exposure is not error."""
    from qemx.noise import NoiseSpec
    from qemx.observables import z
    from qemx.phase.lightcone import calculate_lambda_lc
    a, b = _circ(0.3, 0.3, "Z"), _circ(0.3, 0.3, "X")
    assert calculate_lambda_lc(a, NoiseSpec(), z(0, 1), {0: 0}).lam_lc == pytest.approx(
        calculate_lambda_lc(b, NoiseSpec(), z(0, 1), {0: 0}).lam_lc, rel=1e-12)
    assert _expect_z(a) == pytest.approx(0.0, abs=1e-12)           # |+⟩ measured in Z: ⟨Z⟩ = 0 with or without error
    assert abs(_expect_z(b) - 1.0) > 0.1


def test_lam_lc_matches_its_definition_for_independent_pauli_channels():
    """Under the model the descriptor assumes (independent Pauli channels), the twirled probabilities are the
    channels' own and −Σ ln(1 − p) is exact for the no-fault probability of the light cone."""
    from qemx.circuit import CircuitSpec, Op
    from qemx.noise import NoiseSpec
    from qemx.observables import z
    from qemx.phase.lightcone import calculate_lambda_lc
    ps = (0.01, 0.02, 0.005)
    ops = [Op("h", (0,), (), "gate", "")] + [Op("phaseflip", (0,), (p,), "noise", "") for p in ps] + \
          [Op("h", (0,), (), "gate", "")]
    r = calculate_lambda_lc(CircuitSpec(1, tuple(ops), "pauli"), NoiseSpec(), z(0, 1), {0: 0})
    assert r.lam_lc == pytest.approx(sum(ps), rel=1e-12)
    assert r.lam_lc_exact == pytest.approx(-sum(math.log1p(-p) for p in ps), rel=1e-12)
    assert np.prod([1 - p for p in ps]) == pytest.approx(math.exp(-r.lam_lc_exact), rel=1e-12)
