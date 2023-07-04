"""Classes to perform key error rate estimate for the BBM92 QKD protocol.

Quantum Key Distribution (QKD) protocols rely on an entangled photon source that
produces entangled photon pairs, which are distributed over two parties. Both parties
randomly choose one of two predefined measurement bases. As the photons are entangled,
non-random results will only be obtained for specific combinations of basis choices.
Detection events are sifted, where only the detection events corresponding to events
where both parties measured in an appropriate basis should be kept. Part of the
resulting sifted key is made public which can be used to estimate the key error rate.

The BBM92 QKD protocol can be seen as a generalization of the famous BB84 protocol.
The protocols differ from each other by using entangled photon pairs that are sent
to two parties. Due to the entanglement, the measurements give both parties information
of each others outcome and allows them to create the key. The exact relation between
the measurement results depends on how the entanglement is created. We have an
tunable source that lets you modify the expected number of entangled photon pairs.
If one party was located at the source and directly measures its quantum state, we
effectively have the BB84 protocol again.


We consider two cases

- Asymptotic Key rate:
    The number of pulses is asymptotic and we only have to optimize the intensity of
    the source. As we have no decoy-states, we do not have to optimize probabilities
    for a measurement basis. For the gain and error-rate computation we include the
    link effects for both parties.

- Finite Key rate:
    The number of pulses is finite and chosen by the user. We optimize both the
    intensity of the source and the probability of measuring in the X and Z basis.
    We take finite-key effects into account by computing bounds on the expected number
    of states in a basis and we use security parameters (set as default, but changeable
    by the user) to tune the confidence of these bounds.

Typical usage example:

    .. code-block:: python

        from tno.quantum.communication.qkd_key_rate.protocols.quantum.bbm92 import (
            BBM92AsymptoticKeyRateEstimate,
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

        asymptotic_key_rate = BBM92AsymptoticKeyRateEstimate(
            detector=detector_Bob, detector_Alice=detector_Alice
        )

        mu, rate = asymptotic_key_rate.optimize_rate(attenuation=0.2)
"""
# pylint: disable=invalid-name

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

from tno.quantum.communication.qkd_key_rate.base import (
    AsymptoticKeyRateEstimate,
    Detector,
    FiniteKeyRateEstimate,
)
from tno.quantum.communication.qkd_key_rate.base.config import NLP_CONFIG
from tno.quantum.communication.qkd_key_rate.utils import binary_entropy as h
from tno.quantum.communication.qkd_key_rate.utils import (
    one_minus_binary_entropy as one_minus_h,
)

warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)


def efficiency_channel(attenuation: float) -> float:
    """Calculates the efficiency of the channel

    Args:
        attenuation: Loss in dB for the channel

    Returns:
        Efficiency channel
    """
    return np.power(10, -attenuation / 10)


def efficiency_system(efficiency: float, attenuation: float) -> float:
    """Calculates the efficiency of the system.
    This includes the efficiency of the communication link.

    Args:
        efficiency: Efficiency of source and detection apparatus
        attenuation: Loss in dB for the channel

    Returns:
        Efficiency system
    """
    return efficiency * efficiency_channel(attenuation)


def compute_gain_and_error_rate(
    detector_alice: Detector,
    detector_bob: Detector,
    attenuation_alice: float,
    attenuation_bob: float,
    half_mu: float,
) -> Tuple[float, float]:
    r"""Computes the total gain and error rate of the channel

    Args:
        detector_alice: Alice's detector
        detector_bob: Bob's detector
        attenuation_alice: Attenuation of Alice's channel
        attenuation_bob: Attenuation of Bob's channel
        half_mu: Half of the expected number of photon-pairs

    Returns:

        - The overall gain. The probability of a coincident event, given a pump pulse.
        - The overall error-rate.

    Errors can result from incorrect states generated, losses over the channel and
    losses at the detector side.

    - error_background:

        Error rate related to background errors, with error-rate 1/2

    - error_detector:

        Intrinsic detector errors. The probability that a photon hits the erroneous
        detector, this error indicates the relative error between the detectors at both sides.

    The initial states are generated using parametric down conversion (PDC), which gives
    the probability

    .. math::
        P(n) = (n+1) \cdot \lambda^n / [(1+\lambda)^{n+2}],

    to generate an $n$-photon-pair. The variable $\lambda$ corresponds to half the
    expected photon pair number.

    To compute the losses over the channel, we compute the overall transmittance of the
    channel, which is given by

    .. math::
        \eta_n = [1 - (1-\eta_A)^n] \cdot [1 - (1-\eta_B)^n],

    where:

        - $\eta_n$: The overall transmittance,
        - $\eta_x$: The channel loss of $x$.

    The probability of a detection event, conditional on an $n$-photon-pair being
    emitted is given by the yield:

    .. math::
        {Yield}_n = [1 - (1-Y_{0A}) \cdot (1-\eta_{A})^n] \cdot
        [1 - (1-Y_{0B}) \cdot (1-\eta_{B})^n],

    where:

        - $Y_{0x}$: The darkcount rate of $x$.

    The overall contribution of $n$-photon-pairs to the number of detected events is
    given by the product of the yield and probability:

    .. math::
        Q_n = Y_n \cdot P(n),

    where:

        - $Q_n$: The gain of the n-photon-pair.

    and the overall gain is then given by the sum of these individual gains. The overall
    gain corresponds to the probability of a detection event, conditional on a pulse
    being sent

    .. math::
        Q_\lambda = \sum_{n=0}^{\infty} Q_n.

    From this we compute the quantum bit-error rate (QBER):

    .. math::
        E_\lambda \cdot Q_\lambda = \sum_{n=0}^{\infty} e_n \cdot Y_n \cdot P(n),

    where

        - $e_n$: The error-rate for n-photon-pair states.

    The functions below compute $Q_{\lambda}$ and $E_{\lambda}$ and return them. We
    optimized the implementation of the functions and by that deviated from standard
    implementations in literature. Our implementation is equivalent, but gives better
    precision in high attenuation cases.

    Formulas derived from "Quantum key distribution with entangled photon sources"
    (doi: `10.1103/PhysRevA.76.012307`_)

    .. _10.1103/PhysRevA.76.012307: http://doi.org/10.1103/PhysRevA.76.012307
    """
    dark_count_rate_alice = detector_alice.dark_count_rate
    dark_count_rate_bob = detector_bob.dark_count_rate
    channel_efficiency_alice = efficiency_system(
        detector_alice.efficiency_party, attenuation_alice
    )
    channel_efficiency_bob = efficiency_system(
        detector_bob.efficiency_party, attenuation_bob
    )

    # Errors from background noise are uniform, thus equal to 0.5
    # Errors from the detector depend on the detector
    error_background = 0.5
    if hasattr(detector_alice, "error_detector"):
        error_detector = detector_alice.error_detector
    elif hasattr(detector_bob, "error_detector"):
        error_detector = detector_bob.error_detector
    else:
        error_detector = 0

    a = dark_count_rate_alice
    b = dark_count_rate_bob
    A = channel_efficiency_alice * half_mu
    B = channel_efficiency_bob * half_mu
    C = channel_efficiency_alice * channel_efficiency_bob * half_mu
    D = A + B - C
    # Note that this implementation, though equivalent, differs from literature
    # This however gives better results for high attenuations and hence for
    # small A, B and C
    error_rate_denominator = (
        (2 * A + A**2 + a) * (1 + B) * (1 + D) / (1 + A)
        + (2 * B + B**2 + b) * (1 + A) * (1 + D) / (1 + B)
        + (a * b - a - b - 2 * D - D**2) * (1 + A) * (1 + B) / (1 + D)
    )

    error_rate_numerator = (
        2
        * (error_background - error_detector)
        * channel_efficiency_alice
        * channel_efficiency_bob
        * half_mu
        * (1 + half_mu)
    )

    # Computing the overall gain and error rate
    gain_lambda = (
        (2 * A + A**2 + a) / np.power(1 + A, 2)
        + (2 * B + B**2 + b) / np.power(1 + B, 2)
        + (a * b - a - b - 2 * D - D**2) / np.power(1 + D, 2)
    )

    error_rate_lambda = error_background - error_rate_numerator / error_rate_denominator

    return gain_lambda, error_rate_lambda


def delta(N: int, delta_bit: float, e1: float) -> float:
    """Gives the upper bound delta in the finite case.

    Args:
        N: Number of detection events
        delta_bit: Error in the X-basis
        e1: Epsilon-security parameter

    Returns:
        Delta
    """
    return 2 * np.sqrt(np.max([-delta_bit * (1 - delta_bit) * np.log(e1) / N, 0]))


class BBM92AsymptoticKeyRateEstimate(AsymptoticKeyRateEstimate):
    """BBM92 Asymptotic Key Rate"""

    def __init__(
        self,
        detector: Detector,
        detector_alice: Optional[Detector] = None,
        **kwargs,
    ) -> None:
        """Class for asymptotic key-rate estimation

        Args:
            detector: The detector used at Bob's side
            detector_alice: The detector used at Alice's side.
                default same detector as Bob.
        """
        super().__init__(detector=detector, args=kwargs)
        self.detector_alice = detector if detector_alice is None else detector_alice

        self.last_positive = -1
        self.last_x = None

    def compute_rate(self, mu: Union[float, ArrayLike], attenuation: float) -> float:
        """Compute the key-rate for a specific intensity and attenuation.

        Note, the source can be placed anywhere between the two parties. If the
        source is placed at one party, we effectively have the BB84 protocol.
        The optimal gain is achieved with the source placed in middle between
        the two parties.

        Args:
            mu: Intensity
            attenuation: Attenuation

        Returns:
            Key-rate

        Raises:
            ValueError: When mu is given with invalid dimensions
        """
        scale_factor = np.power(10, attenuation)

        mu = np.atleast_1d(mu)
        if mu.shape != (1,):
            raise ValueError("Multiple intensities were given, only one is expected.")

        gain_lambda, error_rate_lambda = compute_gain_and_error_rate(
            self.detector, self.detector_alice, attenuation / 2, attenuation / 2, mu / 2
        )

        # These are the error rates corresponding to the X and Z basis
        # For the asymptotic case, the two are the same
        delta_bit = error_rate_lambda
        delta_phase = error_rate_lambda

        key_rate = (
            scale_factor * gain_lambda * (one_minus_h(delta_phase) - h(delta_bit))
        )

        self.last_positive = key_rate
        self.last_x = np.hstack([mu])
        return key_rate[0] / scale_factor

    def _extract_parameters(self, x: ArrayLike) -> dict:
        return dict(mu=x[0])

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
            bounds: Bounds on search range, default [(0.001, 0.5)]

        Returns:
            Optimized intensity and key-rate

        Raises:
            ValueError: When x_0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """
        if bounds is None:
            lower_bound = 0.001
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

        self.last_positive = -1
        self.last_x = x_0

        args = {"attenuation": attenuation}
        res = scipy.optimize.minimize(
            self._f, x_0, args=args, bounds=bounds, **NLP_CONFIG
        )

        mu = res.x
        rate = -res.fun
        if rate < 0:
            raise ValueError("Optimization resulted in a negative key rate.")
        return mu, rate


class BBM92FiniteKeyRateEstimate(FiniteKeyRateEstimate):
    """BBM92 Finite key-rate."""

    def __init__(
        self,
        detector: Detector,
        number_of_pulses: float = 1e12,
        detector_alice: Optional[Detector] = None,
        **kwargs,
    ) -> None:
        """Class for finite key-rate estimation

        Args:
            detector: The detector used at Bob's side
            number_of_pulses: Number of pulses sent in total
            detector_alice: The detector used at Alice's side.
                default same detector as Bob.
        """
        super().__init__(detector=detector, args=kwargs)
        self.number_of_pulses = number_of_pulses

        self.last_positive = -1
        self.last_x = None

        self.detector_alice = detector if detector_alice is None else detector_alice

    def compute_last_positive_distance(self, x: ArrayLike) -> float:
        """Computes the last positive distance.

        The optimization routine sometimes considers a parameter setting
        outside of the valid region. This function is used to push the
        parameters back to the valid regime.
        """
        if self.last_positive > -1:
            return self.last_positive - np.linalg.norm(x - self.last_x)
        return self.last_positive

    def compute_rate(
        self,
        mu: Union[float, ArrayLike],
        attenuation: float,
        probability_basis_X: ArrayLike,
        probability_basis_Z: ArrayLike,
        n_X: Optional[int] = None,
    ) -> float:
        """Compute the key-rate for a specific set of variables

        Note, the source can be placed anywhere between the two parties. If the
        source is placed at one party, we effectively have the BB84 protocol.
        The optimal gain is achieved with the source placed in middle between
        the two parties.

        Args:
            mu: Used intensities
            attenuation: Attenuation of the channel
            probability_basis_X: Probabilities for each intensity in X-basis
            probability_basis_Z: Probabilities for each intensity in Z-basis
            n_X: Number of pulses in the X-basis

        Returns:
            key-rate

        Raises:
            ValueError: When mu is given with invalid dimensions
        """
        scale_factor = np.power(10, attenuation / 10)

        mu = np.atleast_1d(mu)
        if mu.shape != (1,):
            raise ValueError("Multiple intensities were given, only one is expected.")

        # Two security parameters to be used
        p_abort = 2**-50
        epsilon_1 = 1e-12

        # Make sure the variables are within a valid range
        x = np.hstack((mu, probability_basis_X, probability_basis_Z))
        if (
            x.min() < 0
            or x.max() > 1
            or (probability_basis_X + probability_basis_Z) != 1
        ):
            return self.compute_last_positive_distance(x)

        gain_lambda, error_rate_lambda = compute_gain_and_error_rate(
            self.detector,
            self.detector_alice,
            attenuation / 2,
            attenuation / 2,
            float(mu / 2),
        )

        # Compute the number of pulses in the X basis. The Z-basis follows
        n_X_observed = (
            gain_lambda * probability_basis_X * self.number_of_pulses
            if n_X is None
            else n_X
        )

        # Check if the number of pulses in the X basis is positive
        if n_X_observed <= 0:
            return self.compute_last_positive_distance(x)

        # These are the error rates corresponding to the X and Z basis
        # A bound is used for delta_phase (Z) errors
        delta_bit = error_rate_lambda
        delta_phase = error_rate_lambda + delta(
            gain_lambda * self.number_of_pulses, delta_bit, epsilon_1
        )

        # Only the pulses in the X-basis are used to get key-material
        usable_pulses_lp = n_X_observed * (one_minus_h(delta_phase) - h(delta_bit))
        key_rate = (
            scale_factor * (1 - p_abort) * usable_pulses_lp / self.number_of_pulses
        )

        self.last_positive = key_rate
        self.last_x = x
        return self.last_positive

    def _extract_parameters(self, x: ArrayLike) -> dict:
        """Extract the parameters and assigns them correspondingly."""
        return dict(mu=x[0], probability_basis_X=x[1], probability_basis_Z=x[2])

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
            x_0: Initial search value
            bounds: Bounds on search range
        Returns:
            Optimized x=[intensity, probability_basis_X, probability_basis_Z]
            and found optimal key-rate

        Raises:
            ValueError: when x_0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """
        if bounds is None:
            lower_bound = np.array([0.01, 0, 0])
            upper_bound = np.array([0.8, 1, 1])
        else:
            lower_bound = np.array([bound[0] for bound in bounds])
            upper_bound = np.array([bound[1] for bound in bounds])
            if len(lower_bound) != 3 or len(upper_bound) != 3:
                raise ValueError(
                    f"Invalid dimensions input bounds. Expected 3 upper- and lower"
                    f" bound. Received {len(lower_bound)} lower- and {len(upper_bound)}"
                    f" upper bounds."
                )

        bounds = scipy.optimize.Bounds(lower_bound, upper_bound)

        if x_0 is None:
            x_0 = (lower_bound + upper_bound) / 2
        x_0 = np.asarray(x_0, dtype=float)
        if len(x_0) != 3:
            raise ValueError(
                f"Invalid number of inputs. Expected 3 inputs."
                f"Received {len(x_0)} inputs."
            )
        # The probabilities are normalized to 1
        x_0[1:] /= x_0[1:].sum()

        self.last_positive = -1
        self.last_x = x_0

        bounds = scipy.optimize.Bounds(lower_bound, upper_bound)
        # This forces the probabilities to sum to one
        b = np.ones(1)
        A = np.array([0, 1, 1])
        B = np.ones(1)

        constraint = scipy.optimize.LinearConstraint(A, b, B)

        args = {"attenuation": attenuation}
        res = scipy.optimize.minimize(
            self._f, x_0, args=args, constraints=constraint, bounds=bounds, **NLP_CONFIG
        )

        mu = res.x
        rate = -res.fun
        if rate < 0:
            raise ValueError("Optimization resulted in a negative key rate.")
        return mu, rate
