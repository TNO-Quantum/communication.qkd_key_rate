"""Regression tests for sent-pulse units in finite BBM92 n_X overrides."""

import pytest

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum.bbm92 import (
    BBM92FiniteKeyRateEstimate,
)


NUMBER_OF_PULSES = int(5e10)


def _rate(probability_basis_x: float, n_x: int | None = None) -> float:
    detector = standard_detector.customise(
        dark_count_rate=1e-8,
        polarization_drift=0,
        error_detector=0,
        efficiency_party=1,
    )
    estimator = BBM92FiniteKeyRateEstimate(
        detector=detector,
        number_of_pulses=NUMBER_OF_PULSES,
    )
    return estimator.compute_rate(
        mu=0.5,
        attenuation=0.2,
        probability_basis_X=probability_basis_x,
        probability_basis_Z=1 - probability_basis_x,
        n_X=n_x,
    )


def test_sent_pulse_override_matches_internal_estimate() -> None:
    implicit = _rate(0.5)
    explicit = _rate(0.5, NUMBER_OF_PULSES // 2)
    assert implicit > 0
    assert explicit == pytest.approx(implicit, rel=1e-12)


def test_no_x_basis_pulses_yield_zero_key_material() -> None:
    assert _rate(0.0) == 0.0
    assert _rate(0.0, 0) == 0.0


@pytest.mark.parametrize("n_x", [-1, NUMBER_OF_PULSES + 1])
def test_reject_pulse_count_outside_total(n_x: int) -> None:
    with pytest.raises(ValueError, match="total pulse count"):
        _rate(0.5, n_x)


def test_reject_positive_x_count_when_x_probability_is_zero() -> None:
    with pytest.raises(ValueError, match="probability_basis_X is zero"):
        _rate(0.0, NUMBER_OF_PULSES // 2)


def test_reject_inconsistent_all_x_setting() -> None:
    with pytest.raises(ValueError, match="probability_basis_X is one"):
        _rate(1.0, NUMBER_OF_PULSES // 2)
