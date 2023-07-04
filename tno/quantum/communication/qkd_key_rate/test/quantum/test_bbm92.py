"""
Below we define parameters and detectors to compute key-rate plots.
We vary the loss of the channel between the two and compute the
corresponding key-rates. These are then plotted. We do this for the finite
and the asymptotic case.
"""

from copy import deepcopy

import numpy as np

from tno.quantum.communication.qkd_key_rate.protocols.quantum.bbm92 import (
    BBM92AsymptoticKeyRateEstimate,
    BBM92FiniteKeyRateEstimate,
)
from tno.quantum.communication.qkd_key_rate.test.conftest import standard_detector


def test_asymptotic() -> None:
    """Test BBM92 Fully Asymptotic protocol"""
    detector = deepcopy(standard_detector)

    distance = np.arange(1, 20)  # Distance between ALice and Bob in km
    attenuation_factor = 0.2  # Channel loss in dB per km
    attenuation = attenuation_factor * distance

    key_rate = [0 for _ in range(len(attenuation))]

    detector_Alice = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )
    detector_Bob = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )

    asymptotic_key_rate = BBM92AsymptoticKeyRateEstimate(
        detector=detector_Bob, detector_Alice=detector_Alice
    )

    for i, att in enumerate(attenuation):
        _, rate = asymptotic_key_rate.optimize_rate(attenuation=att)
        key_rate[i] = rate

    assert all([rate > 0 for rate in key_rate])
    assert all([key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1)])


def test_finite() -> None:
    """Test BBM92 Finite protocol"""
    detector = deepcopy(standard_detector)

    distance = np.arange(1, 10)  # Distance between ALice and Bob in km
    attenuation_factor = 0.2  # Channel loss in dB per km
    attenuation = attenuation_factor * distance

    key_rate = [0 for _ in range(len(attenuation))]

    detector_Alice = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )
    detector_Bob = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )

    number_of_pulses = 5e10
    finite_key_rate = BBM92FiniteKeyRateEstimate(
        detector=detector_Bob,
        detector_Alice=detector_Alice,
        number_of_pulses=number_of_pulses,
    )

    for i, att in enumerate(attenuation):
        _, rate = finite_key_rate.optimize_rate(attenuation=att)
        key_rate[i] = rate

    assert all([rate > 0 for rate in key_rate])
