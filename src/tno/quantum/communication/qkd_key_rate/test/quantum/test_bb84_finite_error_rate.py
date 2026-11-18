"""Regression tests for the finite BB84 privacy-amplification error rate.

The single-photon error rate is not exposed by ``compute_rate``, so these tests record
every input passed to the binary entropy function ``h`` during a single computation.
"""

from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from tno.quantum.communication.qkd_key_rate._utils import binary_entropy
from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum.bb84 import (  # type: ignore[attr-defined]
    BB84FiniteKeyRateEstimate,
)

ATTENUATION = 0.2


@pytest.fixture(scope="module")
def entropy_inputs() -> list[float]:
    """Compute a finite BB84 key rate and return the inputs passed to ``h`` in order."""
    entropy_inputs: list[float] = []

    def record_entropy_input(value: ArrayLike) -> NDArray[np.float64]:
        entropy_inputs.append(float(np.asarray(value).item()))
        return binary_entropy(value)

    with patch(
        "tno.quantum.communication.qkd_key_rate.quantum.bb84.h", record_entropy_input
    ):
        estimator = BB84FiniteKeyRateEstimate(
            detector=standard_detector,
            number_of_pulses=int(1e12),
            number_of_decoy=2,
        )
        estimator.compute_rate(
            mu=[0.980057214, 0.0576948581, 0.0000029565],
            attenuation=ATTENUATION,
            probability_basis_X=[0.7, 0.1, 0.05],
            probability_basis_Z=[0.1, 0.03, 0.02],
        )
    return entropy_inputs


def test_finite_entropy_inputs_are_probabilities(
    entropy_inputs: list[float],
) -> None:
    """Every input to ``h`` must be a probability, never an event count.

    ``binary_entropy`` returns 0 for inputs >= 1, so passing a count silently removes
    the privacy-amplification penalty from the key rate.
    """
    assert entropy_inputs
    assert all(0 <= value <= 1 for value in entropy_inputs)


def test_finite_single_photon_error_rate_is_upper_bound(
    entropy_inputs: list[float],
) -> None:
    """The estimated single-photon phase error rate must upper bound the true rate.

    The first call to ``h`` in ``compute_rate`` is the privacy-amplification term
    ``h(e_1,Z + delta)`` in the target of the X-basis linear program. Here ``e_1,Z`` is
    the single-photon error rate in the Z basis, which must be an upper bound (Attema
    et al. 2021, Eq. 33-34). Since the simulated data follows the detector model
    exactly, it can never be below the model's true single-photon error rate. If the
    order of the ``h`` calls in ``compute_rate`` changes, update the index below.
    """
    detector = standard_detector
    efficiency = detector.efficiency_party * 10 ** (-ATTENUATION / 10)
    yield_single = 1 - (1 - detector.dark_count_rate) ** 2 * (1 - efficiency)
    true_error_rate_single = (
        yield_single
        - (1 - detector.dark_count_rate)
        * efficiency
        * np.cos(2 * detector.polarization_drift)
    ) / (2 * yield_single)

    assert entropy_inputs[0] >= true_error_rate_single
