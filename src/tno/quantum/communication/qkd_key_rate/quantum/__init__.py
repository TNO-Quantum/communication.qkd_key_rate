"""Quantum Key Distribution (QKD) Protocols.

The following quantum protocols are supported:

- BB84 protocol (:py:mod:`~tno.quantum.communication.qkd_key_rate.quantum.bb84`),
- BB84 protocol using a single photon source (:py:mod:`~tno.quantum.communication.qkd_key_rate.quantum.bb84_single_photon`),
- BBM92 protocol (:py:mod:`~tno.quantum.communication.qkd_key_rate.quantum.bbm92`).

Each protocol requires the user to define the properties of the :py:class:`~tno.quantum.communication.qkd_key_rate.quantum.Detector`
that is being used by Bob. The BBM92 additionally allows information on the detector
used by Alice. By default we consider the following standard detector

.. code-block:: python

    from tno.quantum.communication.qkd_key_rate.quantum import Detector
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

Throughout the codebase the following units are used:

.. list-table::
   :header-rows: 1

   * - Variable Name
     - Unit
   * - name
     - Unitless (string)
   * - jitter_source
     - Seconds (s)
   * - jitter_detector
     - Seconds (s)
   * - dead_time
     - Seconds (s)
   * - detection_window
     - Seconds (s)
   * - efficiency_system
     - Unitless (percentage or fraction)
   * - polarization_drift
     - Radians (rad)
   * - error_detector
     - Unitless (percentage or fraction)
   * - dark_count_frequency
     - Counts per second (Hz)
   * - dark_count_rate
     - Counts per detection window (unitless)
   * - detection_frequency
     - Detection windows per second (Hz)
   * - interval
     - Seconds (s)
   * - efficiency_detector
     - Unitless (percentage or fraction)
   * - efficiency_party
     - Unitless (percentage or fraction)
   * - Key-rate
     - Bits per second (bps)
   * - mu (Intensities of the laser)
     - Unitless (mean photon number per pulse)
   * - attenuation (Loss of the channel)
     - Decibels (dB)


Usage examples of the various protocols can be found in the relevant submodules.
"""  # noqa: E501

from tno.quantum.communication.qkd_key_rate.quantum._detector import (
    Detector,
    standard_detector,
)
from tno.quantum.communication.qkd_key_rate.quantum._keyrate import KeyRate

__all__ = [
    "Detector",
    "KeyRate",
    "standard_detector",
]
