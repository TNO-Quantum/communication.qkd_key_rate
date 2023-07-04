"""
Classes to perform key error rate estimate for the single photon BB84 QKD protocol.

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

Example usage:

    .. code-block:: python

        from tno.quantum.communication.qkd_key_rate.protocols.quantum.bb84_single_photon import (
            BB84SingleAsymptoticKeyRateEstimate,
        )
        from tno.quantum.communication.qkd_key_rate.test.conftest import standard_detector

        detector_Alice = standard_detector.customise(
            dark_count_rate=1e-8,
            polarization_drift=0,
            error_detector=0.1,
            efficiency_party=1,
        )
        detector_Bob = standard_detector.customise(
            dark_count_rate=1e-8,
            polarization_drift=0,
            error_detector=0.1,
            efficiency_party=1,
        )

        asymptotic_key_rate = BB84SingleAsymptoticKeyRateEstimate(
            detector=detector_Bob, detector_Alice=detector_Alice
        )

        mu, rate = asymptotic_key_rate.optimize_rate(attenuation=0.2)
"""
# pylint: disable=invalid-name

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

from tno.quantum.communication.qkd_key_rate.base import (
    AsymptoticKeyRateEstimate,
    Detector,
)
from tno.quantum.communication.qkd_key_rate.base.config import NLP_CONFIG
from tno.quantum.communication.qkd_key_rate.utils import binary_entropy as h


def compute_gain_and_error_rate(
    detector: Detector, attenuation: float
) -> Tuple[float, float]:
    """Computes the total gain and error rate of the channel

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
    """
    Key-rate modules when a single photon source is used. We may then assume
    all states remain save and photon number splitting attacks are impossible.
    """

    def __init__(self, detector: Detector, **kwargs):
        """
        Args:
            detector: The detector used at Bob's side
            kwargs: protocol specific input
        """
        super().__init__(detector=detector, args=kwargs)

    def compute_rate(self, mu: Union[float, ArrayLike], attenuation: float) -> float:
        """Computes the key-rate given an intensity and an attenuation.

        Args:
            mu: Intensity
            attenuation: Attenuation

        Returns:
            Key-rate
        """
        mu = np.atleast_1d(mu)
        if mu.shape != (1,):
            raise ValueError("Multiple intensities were given, only one is expected.")

        Q, E = compute_gain_and_error_rate(self.detector, attenuation)
        return mu[0] * Q * (1 - 2 * h(E))

    def _extract_parameters(self, x: ArrayLike) -> dict:
        return dict(mu=x)

    def optimize_rate(
        self,
        *,
        attenuation: float,
        x_0: Optional[ArrayLike] = None,
        bounds: Optional[List[ArrayLike]] = None,
    ) -> Tuple[NDArray[np.float_], float]:
        """Function to optimize the key-rate

        Args:
            attenuation: Loss in dB for the channel
            x_0: Initial search value, default value [0.5] is chosen.
            bounds: Bounds on search range, default [(0.1, 0.9)]

        Returns:
            Optimized intensity and key-rate

        Raises:
            ValueError: When x_0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """
        if bounds is None:
            # The lower and upper bound on the laser intensity
            lower_bound = 0.1
            upper_bound = 0.9
        else:
            lower_bound = np.array([bound[0] for bound in bounds])
            upper_bound = np.array([bound[1] for bound in bounds])
            if len(lower_bound) != 1 or len(upper_bound) != 1:
                raise ValueError(
                    f"Invalid dimensions input bounds. Expected 1 upper- and lower"
                    f" bound. Received {len(lower_bound)} lower- and {len(upper_bound)}"
                    f" upper bounds."
                )
        bounds = scipy.optimize.Bounds(lower_bound, upper_bound)

        if x_0 is None:
            x_0 = [0.5]
        elif len(x_0) != 1:
            raise ValueError(
                f"Invalid number of inputs. Expected 1 intensity."
                f" Received {len(x_0)} intensities."
            )

        args = {"attenuation": attenuation}
        res = scipy.optimize.minimize(
            self._f, x_0, args=args, bounds=bounds, **NLP_CONFIG
        )

        mu = res.x
        rate = np.atleast_1d(-res.fun)[0]
        if rate < 0:
            raise ValueError("Optimization resulted in a negative key rate.")
        return mu, rate
