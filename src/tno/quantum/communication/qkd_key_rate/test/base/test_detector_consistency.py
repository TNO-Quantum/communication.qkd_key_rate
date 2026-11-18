"""Regression tests for consistency of dependent detector parameters."""

import pytest

from tno.quantum.communication.qkd_key_rate.quantum import Detector
from tno.quantum.communication.qkd_key_rate.quantum.bb84 import (
    BB84FullyAsymptoticKeyRateEstimate,
)


def _detector() -> Detector:
    return Detector(
        name="test",
        jitter_source=0,
        jitter_detector=0,
        dead_time=1e-7,
        detection_window=1,
        interval=1e-7,
        dark_count_rate=1e-8,
        efficiency_system=0.5,
        efficiency_detector=0.2,
    )


def test_customise_system_efficiency_recomputes_party_efficiency() -> None:
    detector = _detector()
    changed = detector.customise(efficiency_system=0.8)
    expected = Detector(
        name="expected",
        jitter_source=0,
        jitter_detector=0,
        dead_time=1e-7,
        detection_window=1,
        interval=1e-7,
        dark_count_rate=1e-8,
        efficiency_system=0.8,
        efficiency_detector=0.2,
    )
    assert detector.efficiency_party == pytest.approx(0.1)
    assert changed.efficiency_party == pytest.approx(0.16)
    changed_rate = BB84FullyAsymptoticKeyRateEstimate(detector=changed).compute_rate(
        mu=0.5, attenuation=0.2
    )
    expected_rate = BB84FullyAsymptoticKeyRateEstimate(detector=expected).compute_rate(
        mu=0.5, attenuation=0.2
    )
    assert changed_rate == pytest.approx(expected_rate)


def test_both_signs_of_inconsistent_efficiency_are_rejected() -> None:
    detector = _detector()
    for party in (0.05, 0.3):
        with pytest.raises(ValueError, match="Incompatible efficiency"):
            detector.customise(efficiency_party=party, efficiency_detector=0.2)


def test_both_signs_of_inconsistent_dark_count_are_rejected() -> None:
    detector = _detector()
    for rate in (1e-8, 2e-3):
        with pytest.raises(ValueError, match="Incompatible dark count"):
            detector.customise(
                dark_count_rate=rate, dark_count_frequency=1000
            )


def test_interval_change_keeps_physical_dark_count_frequency() -> None:
    detector = _detector()
    changed = detector.customise(interval=2e-7)
    assert changed.dark_count_frequency == pytest.approx(detector.dark_count_frequency)
    assert changed.dark_count_rate == pytest.approx(2 * detector.dark_count_rate)
