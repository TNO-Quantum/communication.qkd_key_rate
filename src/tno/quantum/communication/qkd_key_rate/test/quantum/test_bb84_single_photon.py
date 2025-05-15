"""Tests for BB84 single photon."""

import numpy as np

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum.bb84_single_photon import (
    BB84SingleAsymptoticKeyRateEstimate,
)


def test_zero_dark_count() -> None:
    detector = standard_detector.customise(dark_count_rate=0)

    distance = np.arange(1, 300)  # Distance between ALice and Bob in km
    attenuation_factor = 0.2  # Channel loss in dB per km
    attenuation = attenuation_factor * distance

    key_rate: list[float] = [0 for _ in range(len(attenuation))]

    x0 = None

    asymptotic_key_rate = BB84SingleAsymptoticKeyRateEstimate(detector=detector)

    for i, att in enumerate(attenuation):
        x, rate = asymptotic_key_rate.optimize_rate(
            x0=x0, bounds=None, attenuation=float(att)
        )
        key_rate[i] = rate
        x0 = x["mu"]

    assert all(rate > 0 for rate in key_rate)


def test_single_photon() -> None:
    distance = np.arange(1, 50)  # Distance between ALice and Bob in km
    attenuation_factor = 0.2  # Channel loss in dB per km
    attenuation = attenuation_factor * distance

    key_rate: list[float] = [0 for _ in range(len(attenuation))]

    x0 = None

    asymptotic_key_rate = BB84SingleAsymptoticKeyRateEstimate(
        detector=standard_detector
    )

    for i, att in enumerate(attenuation):
        x, rate = asymptotic_key_rate.optimize_rate(
            x0=x0, bounds=None, attenuation=float(att)
        )
        key_rate[i] = rate
        x0 = x["mu"]

    assert all(rate > 0 for rate in key_rate)
