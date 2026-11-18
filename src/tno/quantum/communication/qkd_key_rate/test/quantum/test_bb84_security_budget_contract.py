"""Finite BB84 security-budget regression checks against Attema et al. Eq. (39).

These tests intentionally verify the *actual three LP calls*, rather than merely
matching one constant. The input laser intensities and sifted basis weights are
synthetic. No security claim beyond the exercised failure-budget allocation.
"""

from unittest.mock import patch

import numpy as np
import pytest

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum import bb84


EPSILON_LP = 2**-60
EPSILON_SMOOTHING = 2**-55
EPSILON_SECURITY = 2**-50
DEFAULT_PARAMS = {
    "mu": [.7, .2, .02],
    "attenuation": .1,
    "probability_basis_X": [.55, .25, .1],
    "probability_basis_Z": [.07, .02, .01],
}


def _estimator() -> bb84.BB84FiniteKeyRateEstimate:
    return bb84.BB84FiniteKeyRateEstimate(
        detector=standard_detector,
        number_of_pulses=int(1e9),
        number_of_decoy=2,
    )


@pytest.mark.parametrize(
    ("max_photons", "intensities"),
    [(5, 1), (17, 3), (20, 4), (100, 8)],
)
def test_all_lp_statistical_failures_are_in_the_budget(
    max_photons: int, intensities: int
) -> None:
    # Count two Z-basis LPs and one X-basis LP, each with individual
    # photon-number, decoy-intensity and photon-tail bounds.
    expected = (
        4 * EPSILON_SMOOTHING
        + 3 * (max_photons + 1 + 2 * intensities) * EPSILON_LP
    )
    actual = bb84._finite_bb84_failure_budget(
        max_photons,
        intensities,
        EPSILON_SMOOTHING,
        EPSILON_SMOOTHING,
        EPSILON_SMOOTHING,
        EPSILON_LP,
    )
    assert actual == pytest.approx(expected, rel=1e-14, abs=0)


def test_real_finite_rate_uses_paper_parameters_in_all_three_lps() -> None:
    estimator = _estimator()
    with (
        patch.object(bb84, "solve_finite_lp", wraps=bb84.solve_finite_lp) as lp,
        patch.object(bb84, "delta", wraps=bb84.delta) as phase_bound,
        patch.object(bb84, "delta_ec", wraps=bb84.delta_ec) as reconciliation,
    ):
        rate = estimator.compute_rate(**DEFAULT_PARAMS)

    assert float(np.asarray(rate).item()) > 0
    assert lp.call_count == 3
    assert phase_bound.call_count == 2
    assert reconciliation.call_count == 1
    assert reconciliation.call_args.args[0] == EPSILON_SECURITY
    for call in phase_bound.call_args_list:
        assert call.args[2] == EPSILON_SMOOTHING

    max_photons = lp.call_args_list[0].kwargs["max_num_photons"]
    for call in lp.call_args_list:
        eps_mu = call.kwargs["epsilon_mu_j"]
        eps_photons = call.kwargs["epsilon_num_photons_M"]
        eps_tail = call.kwargs["epsilon_num_photons_M_in_basis_B"]
        assert eps_mu.shape == (3,)
        assert eps_tail.shape == (3,)
        assert eps_photons.shape == (max_photons + 1,)
        np.testing.assert_array_equal(eps_mu, EPSILON_LP)
        np.testing.assert_array_equal(eps_photons, EPSILON_LP)
        np.testing.assert_array_equal(eps_tail, EPSILON_LP)

    epsilon_total = bb84._finite_bb84_failure_budget(
        max_photons, 3,
        EPSILON_SMOOTHING, EPSILON_SMOOTHING, EPSILON_SMOOTHING,
        EPSILON_LP,
    )
    assert 0 < epsilon_total < EPSILON_SECURITY


def test_zero_or_negative_secrecy_margin_is_rejected() -> None:
    # This is a failure of the theorem's input conditions, not a valid
    # numerical key rate whose logarithmic penalty happens to be finite.
    estimator = _estimator()
    with patch.object(
        bb84, "_finite_bb84_failure_budget",
        return_value=EPSILON_SECURITY,
    ):
        with pytest.raises(ValueError, match="secrecy budget"):
            estimator.compute_rate(**DEFAULT_PARAMS)


def test_truncation_tail_starts_at_first_omitted_photon_count() -> None:
    estimator = _estimator()
    original_sf = bb84.scipy.stats.poisson.sf
    with (
        patch.object(bb84.scipy.stats.poisson, "sf", wraps=original_sf) as sf,
        patch.object(bb84, "solve_finite_lp", wraps=bb84.solve_finite_lp) as lp,
    ):
        estimator.compute_rate(**DEFAULT_PARAMS)
    max_photons = lp.call_args_list[0].kwargs["max_num_photons"]
    assert sf.call_count == 3
    assert all(call.args[0] == max_photons for call in sf.call_args_list)
    # The LP explicitly models 0..M; the omitted tail is P(K >= M+1).
    mu = np.asarray(DEFAULT_PARAMS["mu"])
    assert (original_sf(max_photons, mu) > original_sf(max_photons + 1, mu)).all()
