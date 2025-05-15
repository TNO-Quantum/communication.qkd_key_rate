"""Tests for BB84 protocols."""

from copy import deepcopy

import numpy as np

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum.bb84 import (  # type: ignore[attr-defined]
    BB84AsymptoticKeyRateEstimate,
    BB84FiniteKeyRateEstimate,
    BB84FullyAsymptoticKeyRateEstimate,
)


def test_fully_asymptotic() -> None:
    """Test BB84 Fully Asymptotic protocol"""
    detector = deepcopy(standard_detector)

    # All considered losses
    distance = np.arange(1, 100)
    attenuation_factor = 0.1
    attenuation = attenuation_factor * distance

    key_rate: list[float] = [0 for _ in range(len(attenuation))]

    x0 = None
    bounds = None

    fully_asymptotic_key_rate = BB84FullyAsymptoticKeyRateEstimate(detector=detector)
    for i, att in enumerate(attenuation):
        x, rate = fully_asymptotic_key_rate.optimize_rate(
            x0=x0, bounds=bounds, attenuation=float(att)
        )
        key_rate[i] = rate
        x0 = x["mu"]

    assert all(rate > 0 for rate in key_rate)
    assert all(key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1))


def test_asymptotic() -> None:
    """Test BB84 Asymptotic protocol"""
    detector = deepcopy(standard_detector)

    # All considered losses
    distance = np.arange(1, 8)
    attenuation_factor = 0.1
    attenuation = attenuation_factor * distance

    key_rate: list[float] = [0 for _ in range(len(attenuation))]

    number_of_decoy = 3
    asymptotic_key_rate = BB84AsymptoticKeyRateEstimate(
        detector=detector, number_of_decoy=number_of_decoy
    )
    for i, att in enumerate(attenuation):
        _, rate = asymptotic_key_rate.optimize_rate(x0=None, attenuation=float(att))
        key_rate[i] = rate

    assert all(rate > 0 for rate in key_rate)
    assert all(key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1))


def test_finite_bb84() -> None:
    detector = deepcopy(standard_detector)

    # All considered losses
    distance = np.arange(1, 5)
    attenuation_factor = 1
    attenuation = attenuation_factor * distance

    # The detector used at Bob's side
    key_rate: list[float] = [0 for _ in range(len(attenuation))]

    x0 = None
    number_of_pulses = int(10e8)
    number_of_decoy = 1
    finite_key_rate = BB84FiniteKeyRateEstimate(
        detector=detector,
        number_of_pulses=number_of_pulses,
        number_of_decoy=number_of_decoy,
    )
    for i, att in enumerate(attenuation):
        x, rate = finite_key_rate.optimize_rate(x0=x0, attenuation=float(att))
        key_rate[i] = rate
        x0 = np.concatenate(
            (x["mu"], x["probability_basis_X"], x["probability_basis_Z"])
        )

    assert all(rate > 0 for rate in key_rate)
    assert all(key_rate[i] > key_rate[i + 1] for i in range(len(attenuation) - 1))
