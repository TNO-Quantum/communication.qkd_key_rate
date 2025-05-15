"""Test the Detector class."""

import pytest

from tno.quantum.communication.qkd_key_rate.quantum import Detector


def test_detector() -> None:
    """Test creation of detector"""
    detector = Detector(
        name="standard",
        efficiency_detector=0.2,
        jitter_source=0,
        jitter_detector=5.00e-11,
        dead_time=4.50e-08,
        dark_count_frequency=100,
        detection_frequency=100,
        detection_window=1.00e07,
        efficiency_system=1,
    )
    assert detector


def test_missing_required_field() -> None:
    """Test invalid creation of detector due to missing required field"""
    expected_message = "Field 'name' is required field."
    with pytest.raises(ValueError, match=expected_message):
        Detector(
            name=None,  # type: ignore[arg-type]
            efficiency_detector=0.2,
            jitter_source=0,
            jitter_detector=5.00e-11,
            dead_time=4.50e-08,
            dark_count_frequency=100,
            detection_frequency=100,
            detection_window=1.00e07,
            efficiency_system=1,
        )


def test_invalid_interval_field() -> None:
    """Test invalid creation of detector due to either
    invalid interval or invalid detection_frequency"""
    expected_message = (
        "Either the field 'interval' or 'detection_frequency' is required."
    )
    with pytest.raises(ValueError, match=expected_message):
        Detector(
            name="standard",
            efficiency_detector=0.2,
            jitter_source=0,
            jitter_detector=5.00e-11,
            dead_time=4.50e-08,
            dark_count_frequency=100,
            detection_window=1.00e07,
            efficiency_system=1,
        )

    expected_message = "Incompatible interval and detection frequency."
    with pytest.raises(ValueError, match=expected_message):
        # Case where interval != 1 / detection_frequency
        Detector(
            name="standard",
            efficiency_detector=0.2,
            jitter_source=0,
            jitter_detector=5.00e-11,
            dead_time=4.50e-08,
            dark_count_frequency=100,
            detection_window=1.00e07,
            efficiency_system=1,
            interval=0.01,
            detection_frequency=100 + 20,
        )


def test_invalid_dark_count_rate_field() -> None:
    """Test invalid creation of detector due to either
    invalid dark_count_rate or dark_count_frequency"""
    expected_message = (
        "Either the field 'dark_count_rate' or 'dark_count_frequency' is required."
    )
    with pytest.raises(ValueError, match=expected_message):
        Detector(
            name="standard",
            efficiency_detector=0.2,
            jitter_source=0,
            jitter_detector=5.00e-11,
            dead_time=4.50e-08,
            detection_frequency=100,
            detection_window=1.00e07,
            efficiency_system=1,
        )

    expected_message = "Incompatible dark count rate and dark count frequency."
    with pytest.raises(ValueError, match=expected_message):
        # Case where dark_count_frequency != dark_count_rate * detection_frequency
        Detector(
            name="standard",
            efficiency_detector=0.2,
            jitter_source=0,
            jitter_detector=5.00e-11,
            dead_time=4.50e-08,
            detection_frequency=100,
            detection_window=1.00e07,
            efficiency_system=1,
            dark_count_frequency=100,
            dark_count_rate=1 + 0.05,
        )


def test_invalid_efficiency_field() -> None:
    """Test invalid creation of detector due to either
    invalid efficiency_party or efficiency_detector"""
    expected_message = (
        "Either the field 'efficiency_party' or 'efficiency_detector' is required."
    )
    with pytest.raises(ValueError, match=expected_message):
        Detector(
            name="standard",
            jitter_source=0,
            jitter_detector=5.00e-11,
            dead_time=4.50e-08,
            dark_count_frequency=100,
            detection_frequency=100,
            detection_window=1.00e07,
            efficiency_system=1,
        )

    expected_message = "Incompatible efficiency and efficiency detector."
    with pytest.raises(ValueError, match=expected_message):
        # Case where efficiency_party !=  efficiency_system * efficiency_detector
        Detector(
            name="standard",
            jitter_source=0,
            jitter_detector=5.00e-11,
            dead_time=4.50e-08,
            dark_count_frequency=100,
            detection_frequency=100,
            detection_window=1.00e07,
            efficiency_system=1,
            efficiency_detector=0.2,
            efficiency_party=0.2 + 0.1,
        )


def test_from_file() -> None:
    """Test detector from .csv file."""
    path = "src/tno/quantum/communication/qkd_key_rate/test/base/detectors.csv"
    detectors = Detector.from_file(path)
    for detector in detectors:
        assert isinstance(detector, Detector)


def test_customise() -> None:
    """Test customise detector"""
    detector = Detector(
        name="standard",
        efficiency_detector=0.2,
        jitter_source=0,
        jitter_detector=5.00e-11,
        dead_time=4.50e-08,
        dark_count_frequency=100,
        detection_frequency=1.00e07,
        detection_window=5,
        efficiency_system=1,
    )

    new_dark_count_rate = 1e-8
    new_polarization_drift = 0.1
    new_error_detector = 0.5
    new_efficiency_party = 0.5

    assert detector.customise(
        dark_count_rate=new_dark_count_rate,
        polarization_drift=new_polarization_drift,
        error_detector=new_error_detector,
        efficiency_party=1,
    )

    detector = detector.customise(
        dark_count_rate=new_dark_count_rate,
        polarization_drift=new_polarization_drift,
        error_detector=new_error_detector,
        efficiency_party=new_efficiency_party,
    )
    assert detector.dark_count_rate == new_dark_count_rate
    assert detector.polarization_drift == new_polarization_drift
    assert detector.error_detector == new_error_detector
    assert detector.efficiency_party == new_efficiency_party


def test_new_zero_parameter() -> None:
    """Test setting new parameter to zero."""
    detector = Detector(
        name="standard",
        efficiency_detector=0.2,
        jitter_source=0,
        jitter_detector=5.00e-11,
        dead_time=4.50e-08,
        dark_count_frequency=100,
        detection_frequency=1.00e07,
        detection_window=5,
        efficiency_system=1,
    )

    new_dark_count_rate = 0.0
    detector = detector.customise(dark_count_rate=new_dark_count_rate)
    assert detector.dark_count_rate == new_dark_count_rate
