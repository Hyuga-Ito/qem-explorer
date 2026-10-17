"""1.5: discrepancy measures — domains (no metrics on quasi-probabilities), agreement with the divergence pipeline,
calibrated finite-shot tests (matched-model false-positive rate and power), multiple testing, and noise-rate
conventions checked against the channel itself."""
import math

import numpy as np
import pytest

from qemx import discrepancy as D
from qemx import divergence as DV


def test_distances_closed_form_and_agreement_with_the_divergence_pipeline():
    p, q = np.array([0.5, 0.5, 0, 0]), np.array([0.25, 0.25, 0.25, 0.25])
    assert D.tvd(p, q) == pytest.approx(0.5)
    assert D.bhattacharyya(p, q) == pytest.approx(2 * math.sqrt(0.125))
    assert D.classical_fidelity(p, q) == pytest.approx(0.5)
    assert D.hellinger(p, q) == pytest.approx(math.sqrt(1 - math.sqrt(0.5)))
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
        assert D.classical_fidelity(a, b) == pytest.approx(float(DV.hellinger_fidelity(a, b)), rel=1e-12)
        assert D.normalized_fidelity(a, b) == pytest.approx(float(DV.normalized_fidelity(a, b)), rel=1e-12)
        assert 0 <= D.tvd(a, b) <= 1 and 0 <= D.hellinger(a, b) <= 1
        assert D.hellinger(a, b) ** 2 <= D.tvd(a, b) + 1e-12                # H² ≤ TVD ≤ √2 H
        assert D.tvd(a, b) <= math.sqrt(2) * D.hellinger(a, b) + 1e-12
    assert math.isnan(D.normalized_fidelity(np.full(4, 0.25), q))


def test_quasi_probabilities_are_never_silently_clipped():
    q = np.array([0.6, 0.5, -0.1, 0.0])
    p = np.array([0.5, 0.5, 0, 0])
    for f in (D.tvd, D.hellinger, D.classical_fidelity, D.normalized_fidelity):
        with pytest.raises(D.QuasiProbabilityError):
            f(p, q)
    with pytest.raises(D.QuasiProbabilityError):
        D.tvd(p, np.array([0.5, 0.4, 0, 0]))                              # does not sum to 1
    assert D.l1_signed(q, p) == pytest.approx(0.1)
    proj, moved = D.project_to_simplex(q)
    assert D.check_distribution(proj) is not None and moved > 0
    assert D.tvd(p, proj) >= 0                                             # metrics only after an explicit projection


def test_marginal():
    p = np.zeros(8)
    p[0b101] = 0.7
    p[0b010] = 0.3
    m = D.marginal(p, [0, 2], 3)
    assert m.tolist() == pytest.approx([0.3, 0, 0, 0.7])


@pytest.mark.parametrize("metric", ["tvd", "hellinger", "classical_fidelity"])
def test_matched_model_false_positive_rate_is_nominal(metric):
    """Data drawn from the model itself: p-values are uniform, so P(p < 0.05) ≈ 0.05 (binomial 4σ band)."""
    rng = np.random.default_rng(7)
    model = rng.dirichlet(np.ones(16) * 2)
    shots, trials = 2000, 300
    pv = []
    for t in range(trials):
        f = rng.multinomial(shots, model) / shots
        pv.append(D.distance_test(model, f, shots, metric, reps=400, rng=1000 + t)["p_value"])
    rate = np.mean(np.asarray(pv) < 0.05)
    assert abs(rate - 0.05) <= 4 * math.sqrt(0.05 * 0.95 / trials)


def test_power_against_a_controlled_alternative():
    rng = np.random.default_rng(8)
    model = np.full(8, 1 / 8)
    truth = model.copy()
    truth[0] += 0.04
    truth[1:] -= 0.04 / 7
    hits = sum(D.distance_test(model, rng.multinomial(4000, truth) / 4000, 4000, "tvd", 400, rng=t)["p_value"] < 0.05
               for t in range(60))
    assert hits / 60 > 0.8


def test_adjusted_pvalues_known_values_and_familywise_error():
    p = [0.01, 0.04, 0.03, 0.005, math.nan]
    assert D.adjust_pvalues(p, "holm")[:4].tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert D.adjust_pvalues(p, "hochberg")[:4].tolist() == pytest.approx([0.03, 0.04, 0.04, 0.02])
    assert D.adjust_pvalues(p, "bh")[:4].tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert D.adjust_pvalues(p, "bonferroni")[:4].tolist() == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert math.isnan(D.adjust_pvalues(p, "holm")[4])
    rng = np.random.default_rng(3)
    fw = np.mean([(D.adjust_pvalues(rng.uniform(size=20), "holm") < 0.05).any() for _ in range(2000)])
    raw = np.mean([(rng.uniform(size=20) < 0.05).any() for _ in range(2000)])
    assert fw <= 0.05 + 4 * math.sqrt(0.05 * 0.95 / 2000) and raw > 0.5     # unadjusted: a wall of false flags
    assert D.simultaneous_z(1) == pytest.approx(1.959964, abs=1e-5)
    assert D.simultaneous_z(100) > 3.4


def _depol_fidelities(p, k):
    """(ε, r) of Λ_p from its action on a maximally entangled state (Choi state) and F_avg = (d F_e + 1)/(d + 1)."""
    d = 2 ** k
    phi = np.eye(d).reshape(-1) / math.sqrt(d)                  # |Φ⟩ = Σ|ii⟩/√d
    rho = np.outer(phi, phi)
    red = np.trace(rho.reshape(d, d, d, d), axis1=0, axis2=2)  # the second system's reduced state (I/d)
    out = (1 - p) * rho + p * np.kron(np.eye(d) / d, red)
    Fe = float(np.real(phi @ out @ phi))
    return 1 - Fe, 1 - (d * Fe + 1) / (d + 1)


@pytest.mark.parametrize("k", [1, 2])
def test_noise_rate_conventions_match_the_channel(k):
    for p in (0.0, 0.01, 0.2):
        eps, r = _depol_fidelities(p, k)
        assert D.convert_error_rate(p, "depolarizing_p", "entanglement_infidelity", k) == pytest.approx(eps, abs=1e-12)
        assert D.convert_error_rate(p, "depolarizing_p", "average_gate_infidelity", k) == pytest.approx(r, abs=1e-12)
        for a in D.RATE_KINDS:
            for b in D.RATE_KINDS:
                v = D.convert_error_rate(D.convert_error_rate(p, "depolarizing_p", a, k), a, b, k)
                assert v == pytest.approx(D.convert_error_rate(p, "depolarizing_p", b, k), abs=1e-12)
    # an RB-style average infidelity r is not the simulator's p: for two qubits p = 4r/3
    assert D.convert_error_rate(0.0075, "average_gate_infidelity", "depolarizing_p", 2) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        D.convert_error_rate(-0.1, "depolarizing_p", "entanglement_infidelity")
