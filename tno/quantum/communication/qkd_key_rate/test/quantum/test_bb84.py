"""
Below we define parameters and detectors to compute key-rate plots.
We vary the loss of the channel between the two and compute the
corresponding key-rates. These are then plotted. We do this for the finite
and the asymptotic case.
"""
from copy import deepcopy

import numpy as np

from tno.quantum.communication.qkd_key_rate.protocols.quantum.bb84 import (
    BB84AsymptoticKeyRateEstimate,
    BB84FiniteKeyRateEstimate,
    BB84FullyAsymptoticKeyRateEstimate,
)
from tno.quantum.communication.qkd_key_rate.test.conftest import standard_detector


def test_fully_asymptotic() -> None:
    """Test BB84 Fully Asymptotic protocol"""
    detector = deepcopy(standard_detector)

    # All considered losses
    distance = np.arange(1, 100)
    attenuation_factor = 0.1
    attenuation = attenuation_factor * distance

    key_rate = [0 for _ in range(len(attenuation))]

    x_0 = None
    bounds = None

    fully_asymptotic_key_rate = BB84FullyAsymptoticKeyRateEstimate(detector=detector)
    for i, att in enumerate(attenuation):
        mu, rate = fully_asymptotic_key_rate.optimize_rate(
            x_0=x_0, bounds=bounds, attenuation=att
        )
        key_rate[i] = rate
        x_0 = mu

    assert all([rate > 0 for rate in key_rate])
    assert all([key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1)])


def test_asymptotic() -> None:
    """Test BB84 Asymptotic protocol"""
    detector = deepcopy(standard_detector)

    # All considered losses
    distance = np.arange(1, 8)
    attenuation_factor = 0.1
    attenuation = attenuation_factor * distance

    key_rate = [0 for _ in range(len(attenuation))]

    number_of_decoy = 3
    asymptotic_key_rate = BB84AsymptoticKeyRateEstimate(
        detector=detector, number_of_decoy=number_of_decoy
    )
    for i, att in enumerate(attenuation):
        _, rate = asymptotic_key_rate.optimize_rate(x_0=None, attenuation=att)
        key_rate[i] = rate

    assert all([rate > 0 for rate in key_rate])
    assert all([key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1)])


def test_finite() -> None:
    detector = deepcopy(standard_detector)

    # All considered losses
    distance = np.arange(1, 5)
    attenuation_factor = 1
    attenuation = attenuation_factor * distance

    # The detector used at Bob's side
    key_rate = [0 for _ in range(len(attenuation))]

    x_0 = None
    number_of_pulses = 10e8
    number_of_decoy = 1
    finite_key_rate = BB84FiniteKeyRateEstimate(
        detector=detector,
        number_of_pulses=number_of_pulses,
        number_of_decoy=number_of_decoy,
    )
    for i, att in enumerate(attenuation):
        mu, rate = finite_key_rate.optimize_rate(x_0=x_0, attenuation=att)
        key_rate[i] = rate
        x_0 = mu

    assert all([rate > 0 for rate in key_rate])
    assert all([key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1)])
