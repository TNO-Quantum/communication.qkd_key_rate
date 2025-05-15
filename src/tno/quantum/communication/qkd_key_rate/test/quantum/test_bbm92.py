"""Test for BBM92 protocol."""

from copy import deepcopy

import numpy as np

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum.bbm92 import (
    BBM92AsymptoticKeyRateEstimate,
    BBM92FiniteKeyRateEstimate,
)


def test_asymptotic() -> None:
    """Test BBM92 Fully Asymptotic protocol"""
    detector = deepcopy(standard_detector)

    distance = np.arange(1, 20)  # Distance between ALice and Bob in km
    attenuation_factor = 0.2  # Channel loss in dB per km
    attenuation = attenuation_factor * distance

    key_rate: list[float] = [0 for _ in range(len(attenuation))]

    detector_alice = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )
    detector_bob = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )

    asymptotic_key_rate = BBM92AsymptoticKeyRateEstimate(
        detector=detector_bob, detector_alice=detector_alice
    )

    for i, att in enumerate(attenuation):
        _, rate = asymptotic_key_rate.optimize_rate(attenuation=float(att))
        key_rate[i] = rate

    assert all(rate > 0 for rate in key_rate)
    assert all(key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1))


def test_finite_bbm92() -> None:
    """Test BBM92 Finite protocol"""
    detector = deepcopy(standard_detector)

    distance = np.arange(1, 10)  # Distance between ALice and Bob in km
    attenuation_factor = 0.2  # Channel loss in dB per km
    attenuation = attenuation_factor * distance

    key_rate: list[float] = [0 for _ in range(len(attenuation))]

    detector_alice = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )
    detector_bob = detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0.1,
        efficiency_party=1,
    )

    number_of_pulses = int(5e10)
    finite_key_rate = BBM92FiniteKeyRateEstimate(
        detector=detector_bob,
        detector_alice=detector_alice,
        number_of_pulses=number_of_pulses,
    )

    for i, att in enumerate(attenuation):
        _, rate = finite_key_rate.optimize_rate(attenuation=float(att))
        key_rate[i] = rate

    assert all(rate > 0 for rate in key_rate)
