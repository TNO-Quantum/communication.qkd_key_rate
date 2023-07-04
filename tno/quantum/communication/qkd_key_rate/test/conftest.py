"""
Pytest configuration file for QKD Key rate test suite.
"""

from tno.quantum.communication.qkd_key_rate.base import Detector

# defaults
TOL = 1e-3

standard_detector = Detector(
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
