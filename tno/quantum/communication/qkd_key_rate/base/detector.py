"""Base class for detector objects."""
from __future__ import annotations

from abc import ABC
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# pylint: disable=invalid-name

DEFAULT_POLARIZATION_DRIFT = np.arcsin(np.sqrt(0.015))
DEFAULT_ERROR_DETECTOR = 0


class Detector(ABC):
    """Class for the detectors to be used in the key-rate estimates."""

    required_fields = [
        "name",
        "jitter_source",
        "jitter_detector",
        "dead_time",
        "detection_window",
        "efficiency_system",
    ]

    def __init__(
        self,
        name: str,
        jitter_source: float,
        jitter_detector: float,
        dead_time: float,
        detection_window: float,
        efficiency_system: float,
        polarization_drift: float = DEFAULT_POLARIZATION_DRIFT,
        error_detector: float = DEFAULT_ERROR_DETECTOR,
        dark_count_frequency: Optional[float] = None,
        dark_count_rate: Optional[float] = None,
        detection_frequency: Optional[float] = None,
        interval: Optional[float] = None,
        efficiency_detector: Optional[float] = None,
        efficiency_party: Optional[float] = None,
    ) -> None:
        r"""
        Initialise a Detector instance.

        Args:
            name: Label of the detector party
            jitter_source: Time-delay introduced at the source
            jitter_detector: Time-delay introduced at the detector
            dead_time: The recovery time before a new event can be recorded
            detection_window: Time window used for event detection
            efficiency_system: Efficiency of the detecting side without the detector
            polarization_drift: Shift/drift in the encoding of the photons
            error_detector: Error-rate of detector
            dark_count_frequency: Number of dark-counts per second
            dark_count_rate: Number of dark-counts per detection window
            detection_frequency: Number of detection windows per second
            interval: Length of a single detection window in seconds
            efficiency_detector: Efficiency of the detector on the detecting side
            efficiency_party: Total efficiency of the detecting side

        If related optional arguments are both given they must satisfy

            - $\text{interval} = \frac{1}{\text{detection_frequency}}$
            - $\text{dark_count_frequency} = \text{dark_count_rate} \cdot \text{detection_frequency}$
            - $\text{efficiency_party} = \text{efficiency_system} \cdot \text{efficiency_detector}$

        If they are not given, they are calculated using the same formulas.

        Returns:
            Detector instance with specified parameters.

        Raises:
            ValueError: If a required field is missing.
            ValueError: If a needed optional field is missing.
            AssertionError: If inconsistent related optional fields are provided.
        """
        self.name = name
        self.jitter_source = jitter_source
        self.jitter_detector = jitter_detector
        self.dead_time = dead_time
        self.detection_window = detection_window
        self.efficiency_system = efficiency_system
        self.polarization_drift = polarization_drift
        self.error_detector = error_detector
        self.dark_count_frequency = dark_count_frequency
        self.dark_count_rate = dark_count_rate
        self.detection_frequency = detection_frequency
        self.interval = interval
        self.efficiency_detector = efficiency_detector
        self.efficiency_party = efficiency_party

        for k in self.required_fields:
            if self.__dict__[k] is None:
                raise ValueError(f"Field '{k}' is required field.")

        # Handle optional fields
        if interval is None and detection_frequency is None:
            raise ValueError(
                "Either the field 'interval' or 'detection_frequency' is required."
            )
        if interval is not None and detection_frequency is not None:
            assert abs(interval - 1 / detection_frequency) < 10e-6
        if interval is None:
            self.interval = 1 / self.detection_frequency
        if detection_frequency is None:
            self.detection_frequency = 1 / self.interval

        if dark_count_rate is None and dark_count_frequency is None:
            raise ValueError(
                "Either the field 'dark_count_rate' or 'dark_count_frequency' is required."
            )
        if dark_count_rate is not None and dark_count_frequency is not None:
            assert (dark_count_rate - dark_count_frequency * self.interval) < 10e-6
        if dark_count_rate is None:
            self.dark_count_rate = self.dark_count_frequency * self.interval
        if dark_count_frequency is None:
            self.dark_count_frequency = self.dark_count_rate * self.detection_frequency

        if efficiency_party is None and efficiency_detector is None:
            raise ValueError(
                "Either the field 'efficiency_party' or 'efficiency_detector' is required."
            )
        if efficiency_party is not None and efficiency_detector is not None:
            assert (
                efficiency_party - self.efficiency_system * efficiency_detector
            ) < 10e-6
        if efficiency_party is None:
            self.efficiency_party = self.efficiency_system * self.efficiency_detector
        if efficiency_detector is None:
            self.efficiency_detector = self.efficiency_party / self.efficiency_system

    @classmethod
    def from_file(cls, path: str) -> List[Detector]:
        """
        Construct Detectors from csv file.

        Args:
            path: Path to csv file.
        """
        df = pd.read_csv(path, delimiter=";")
        df = df.replace({np.nan: None})
        return [cls(**detector_kwargs) for detector_kwargs in df.to_dict("records")]

    def customise(
        self,
        name: Optional[str] = None,
        jitter_source: Optional[float] = None,
        jitter_detector: Optional[float] = None,
        dead_time: Optional[float] = None,
        detection_window: Optional[float] = None,
        efficiency_system: Optional[float] = None,
        polarization_drift: Optional[float] = None,
        error_detector: Optional[float] = None,
        dark_count_frequency: Optional[float] = None,
        dark_count_rate: Optional[float] = None,
        detection_frequency: Optional[float] = None,
        interval: Optional[float] = None,
        efficiency_detector: Optional[float] = None,
        efficiency_party: Optional[float] = None,
    ) -> Detector:
        """Create a detector with customised parameter from current detector

        See :py:meth:`~__init__` for parameter description.
        """
        new_parameters = dict(**self.__dict__)
        if jitter_source is not None:
            new_parameters.update(jitter_source=jitter_source)
        if jitter_detector is not None:
            new_parameters.update(jitter_detector=jitter_detector)
        if dead_time is not None:
            new_parameters.update(dead_time=dead_time)
        if detection_window is not None:
            new_parameters.update(detection_window=detection_window)
        if efficiency_system is not None:
            new_parameters.update(efficiency_system=efficiency_system)
        if polarization_drift is not None:
            new_parameters.update(polarization_drift=polarization_drift)
        if error_detector is not None:
            new_parameters.update(error_detector=error_detector)

        if dark_count_frequency is not None or dark_count_rate is not None:
            new_parameters.update(dark_count_frequency=dark_count_frequency)
            new_parameters.update(dark_count_rate=dark_count_rate)

        if detection_frequency is not None or interval is not None:
            new_parameters.update(detection_frequency=detection_frequency)
            new_parameters.update(interval=interval)

        if efficiency_detector is not None or efficiency_party is not None:
            new_parameters.update(efficiency_detector=efficiency_detector)
            new_parameters.update(efficiency_party=efficiency_party)

        new_name = name if name is not None else self.name + "(adjusted)"
        new_parameters.update(name=new_name)

        return Detector(**new_parameters)

    def get_parameters(self) -> dict[str, Any]:
        """Get all parameters of the Detector"""
        return dict(**self.__dict__)

    def __repr__(self) -> str:
        return f"Detector:{self.name}"
