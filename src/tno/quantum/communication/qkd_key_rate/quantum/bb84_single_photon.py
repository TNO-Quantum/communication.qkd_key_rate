"""Classes to perform key error rate estimate for the single photon BB84 QKD protocol.

The analysis is adjusted to using a single photon source.

This code is based on TNO's BB84 key-rate paper (doi: `10.1007/s11128-021-03078-0`_):

.. _10.1007/s11128-021-03078-0: https://doi.org/10.1007/s11128-021-03078-0


This setting is most similar to the originally proposed BB84 protocol, where single
photon quantum states are send. Generating single photon states is hard in practice,
hence the general approach is to use an attunable laser source and use multiple
intensity settings. If we instead use single photon states, much of the analysis
simplifies as we know each pulse is safe against for instance photon-number
splitting (PNS) attacks.

- Asymptotic Key Rate
    The number of pulses is asymptotic and we use only a single intensity setting,
    which we optimize. The other functions are similar to the standard BB84 protocol.

    >>> from tno.quantum.communication.qkd_key_rate.quantum.bb84_single_photon import (
    ...     BB84SingleAsymptoticKeyRateEstimate,
    ... )
    >>> from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
    >>>
    >>> detector = standard_detector.customise(
    ...     dark_count_rate=1e-8,
    ...     polarization_drift=0,
    ...     error_detector=0.1,
    ...     efficiency_party=1,
    ... )
    >>>
    >>> finite_key_rate = BB84SingleAsymptoticKeyRateEstimate(detector=detector)
    >>> x, rate = finite_key_rate.optimize_rate(attenuation=0.2)
    >>> print(f"{x['mu']=}, {rate=}")
    x['mu']=array([0.89916328]), rate=0.858693993504011
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

from tno.quantum.communication.qkd_key_rate._utils import binary_entropy as h
from tno.quantum.communication.qkd_key_rate.quantum._config import NLP_CONFIG
from tno.quantum.communication.qkd_key_rate.quantum._keyrate import (
    AsymptoticKeyRateEstimate,
)

if TYPE_CHECKING:
    from tno.quantum.communication.qkd_key_rate.quantum import Detector


def compute_gain_and_error_rate(
    detector: Detector, attenuation: float
) -> tuple[float, float]:
    """Computes the total gain and error rate of the channel.

    These computations are straight-forward and have a closed
    expression for single photon sources.

    Args:
        detector: Bob's detector
        attenuation: Attenuation of the channel

    Returns:
        - The gain per intensity. The probability of an event, given a pulse.
        - The error rate per intensity.
    """
    dark_count_rate = detector.dark_count_rate
    efficiency_bob = detector.efficiency_party
    polarization_drift = detector.polarization_drift

    efficiency_channel = np.power(10, -attenuation / 10)
    efficiency = efficiency_bob * efficiency_channel

    gain = 1 - np.power(1 - dark_count_rate, 2) * (1 - efficiency)
    yield_times_error_single = (
        1
        - (1 - dark_count_rate) * efficiency * np.cos(2 * polarization_drift)
        - np.power(1 - dark_count_rate, 2) * (1 - efficiency)
    ) / 2
    error_rate = yield_times_error_single / gain
    return gain, error_rate


class BB84SingleAsymptoticKeyRateEstimate(AsymptoticKeyRateEstimate):
    """Key-rate modules for BB84 when a single photon source is used.

    We assume that all states remain safe and photon number splitting attacks are
    not possible.
    """

    def __init__(self, detector: Detector, **kwargs: Any) -> None:
        """Init of BB84SingleAsymptoticKeyRateEstimate.

        Args:
            detector: The detector used at Bob's side
            kwargs: protocol specific input
        """
        super().__init__(detector=detector, args=kwargs)

    def compute_rate(self, mu: float | ArrayLike, attenuation: float) -> float:  # type: ignore[override]
        """Computes the key-rate given an intensity and an attenuation.

        Args:
            mu: Intensity
            attenuation: Attenuation

        Returns:
            Key-rate
        """
        mu = np.atleast_1d(mu)
        if mu.shape != (1,):
            error_msg = "Multiple intensities were given, only one is expected."
            raise ValueError(error_msg)

        Q, E = compute_gain_and_error_rate(self.detector, attenuation)
        return float(mu[0] * Q * (1 - 2 * h(E)))

    def _extract_parameters(
        self, x: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        return {"mu": x}

    def optimize_rate(
        self,
        *,
        attenuation: float,
        x0: ArrayLike | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], float]:
        """Function to optimize the key-rate.

        Args:
            attenuation: Loss in dB for the channel
            x0: Initial search value, default value [0.5] is chosen.
            bounds: Bounds on search range, default [(0.1, 0.9)]

        Returns:
            Optimized intensity and key-rate

        Raises:
            ValueError: When x0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """
        if bounds is None:
            # The lower and upper bound on the laser intensity
            lower_bound = np.array([0.1])
            upper_bound = np.array([0.9])
        else:
            if len(bounds) != 1:
                error_msg = (
                    "Invalid dimensions input bounds. Expected 1 upper- and lower"
                    f" bound but received {len(bounds)} bounds."
                )
                raise ValueError(error_msg)
            lower_bound = np.array([bound[0] for bound in bounds])
            upper_bound = np.array([bound[1] for bound in bounds])
        bounds = scipy.optimize.Bounds(lower_bound, upper_bound)

        if x0 is None:
            x0 = [0.5]
        elif len(np.asarray(x0)) != 1:
            error_msg = (
                "Invalid number of inputs. Expected 1 intensity."
                f" Received {len(np.asarray(x0))} intensities."
            )
            raise ValueError(error_msg)

        args = {"attenuation": attenuation}
        res = scipy.optimize.minimize(
            self._f, x0, args=args, bounds=bounds, **NLP_CONFIG
        )

        mu = res.x
        rate = np.atleast_1d(-res.fun)[0]
        if rate < 0:
            error_msg = "Optimization resulted in a negative key rate."
            raise ValueError(error_msg)
        return self._extract_parameters(mu), float(rate)
