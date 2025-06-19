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

    >>> from tno.quantum.communication.qkd_key_rate.quantum.bbm92 import (
    ...     BBM92AsymptoticKeyRateEstimate,
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
    >>> finite_key_rate = BBM92AsymptoticKeyRateEstimate(detector=detector)
    >>> x, rate = finite_key_rate.optimize_rate(attenuation=0.2)
    >>> print(f"{x['mu']=}, {rate=}")  # doctest: +SKIP
    x['mu']=array([0.04789677]), rate=0.0013733145512722993

- Finite Key rate:
    The number of pulses is finite and chosen by the user. We optimize both the
    intensity of the source and the probability of measuring in the X and Z basis.
    We take finite-key effects into account by computing bounds on the expected number
    of states in a basis and we use security parameters (set as default, but changeable
    by the user) to tune the confidence of these bounds.

    >>> from tno.quantum.communication.qkd_key_rate.quantum.bbm92 import (
    ...     BBM92FiniteKeyRateEstimate,
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
    >>> finite_key_rate = BBM92FiniteKeyRateEstimate(detector=detector)
    >>> x, rate = finite_key_rate.optimize_rate(attenuation=0.2)
    >>> print(f"{x['mu']=}, {rate=}")  # doctest: +SKIP
    x['mu']=array([0.04781671]), rate=0.0007179260057907571
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

from tno.quantum.communication.qkd_key_rate._utils import binary_entropy as h
from tno.quantum.communication.qkd_key_rate._utils import (
    one_minus_binary_entropy as one_minus_h,
)
from tno.quantum.communication.qkd_key_rate.quantum._config import (
    NLP_CONFIG,
    OptimizationError,
)
from tno.quantum.communication.qkd_key_rate.quantum._keyrate import (
    AsymptoticKeyRateEstimate,
    FiniteKeyRateEstimate,
    _fallback_key_rate_estimate,
)

if TYPE_CHECKING:
    from tno.quantum.communication.qkd_key_rate.quantum import Detector

warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)


def efficiency_channel(attenuation: float) -> float:
    """Calculates the efficiency of the channel.

    Args:
        attenuation: Loss in dB for the channel

    Returns:
        Efficiency channel
    """
    return float(np.power(10, -attenuation / 10))


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
) -> tuple[float, float]:
    r"""Computes the total gain and error rate of the channel.

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
        detector, this error indicates the relative error between the detectors at both
        sides.

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


def delta(N: float, delta_bit: float, e1: float) -> float:
    """Gives the upper bound delta in the finite case.

    Args:
        N: Number of detection events
        delta_bit: Error in the X-basis
        e1: Epsilon-security parameter

    Returns:
        Delta
    """
    return float(
        2 * np.sqrt(np.max([-delta_bit * (1 - delta_bit) * np.log(e1) / N, 0]))
    )


class BBM92AsymptoticKeyRateEstimate(AsymptoticKeyRateEstimate):
    """BBM92 Asymptotic Key Rate."""

    def __init__(
        self,
        detector: Detector,
        detector_alice: Detector | None = None,
        **kwargs: Any,
    ) -> None:
        """Class for asymptotic key-rate estimation.

        Args:
            detector: The detector used at Bob's side
            detector_alice: The detector used at Alice's side.
                default same detector as Bob.
            kwargs: protocol specific arguments.
        """
        super().__init__(detector=detector, args=kwargs)
        self.detector_alice = detector if detector_alice is None else detector_alice

    def compute_rate(self, mu: float | ArrayLike, attenuation: float) -> float:  # type: ignore[override]
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
        mu = np.atleast_1d(mu)
        if mu.shape != (1,):
            error_msg = "Multiple intensities were given, only one is expected."
            raise ValueError(error_msg)

        gain_lambda, error_rate_lambda = compute_gain_and_error_rate(
            self.detector,
            self.detector_alice,
            attenuation / 2,
            attenuation / 2,
            float(mu) / 2,
        )

        # These are the error rates corresponding to the X and Z basis
        # For the asymptotic case, the two are the same
        delta_bit = error_rate_lambda
        delta_phase = error_rate_lambda

        return float(gain_lambda * (one_minus_h(delta_phase) - h(delta_bit)))

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
            bounds: Bounds on search range, default [(0.001, 0.5)]

        Returns:
            Optimized intensity and key-rate

        Raises:
            ValueError: When x0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """
        if bounds is None:
            # The lower and upper bound on the laser intensity
            lower_bound = np.array([0.001])
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
        rate = -res.fun
        if rate < 0:
            error_msg = "Optimization resulted in a negative key rate."
            raise ValueError(error_msg)
        return self._extract_parameters(mu), float(rate)


class BBM92FiniteKeyRateEstimate(FiniteKeyRateEstimate):
    """BBM92 Finite key-rate."""

    def __init__(
        self,
        detector: Detector,
        number_of_pulses: int = int(1e12),
        detector_alice: Detector | None = None,
        **kwargs: Any,
    ) -> None:
        """Class for finite key-rate estimation.

        Args:
            detector: The detector used at Bob's side
            number_of_pulses: Number of pulses sent in total
            detector_alice: The detector used at Alice's side.
                default same detector as Bob.
            kwargs: Protocol specific input.
        """
        super().__init__(detector=detector, args=kwargs)
        self.number_of_pulses = number_of_pulses

        self.last_positive_key_rate: float = -1
        self.last_x: NDArray[np.float64]

        self.detector_alice = detector if detector_alice is None else detector_alice

    def compute_rate(  # type: ignore[override]
        self,
        mu: float | ArrayLike,
        attenuation: float,
        probability_basis_X: ArrayLike,
        probability_basis_Z: ArrayLike,
        n_X: int | None = None,
    ) -> float:
        """Compute the key-rate for a specific set of variables.

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
            ValueError: When mu is given with invalid dimensions.
        """
        probability_basis_X = np.asarray(probability_basis_X)
        probability_basis_Z = np.asarray(probability_basis_Z)

        mu = np.atleast_1d(mu)
        if mu.shape != (1,):
            error_msg = "Multiple intensities were given, only one is expected."
            raise ValueError(error_msg)

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
            return _fallback_key_rate_estimate(
                x, self.last_x, self.last_positive_key_rate
            )

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
            return _fallback_key_rate_estimate(
                x, self.last_x, self.last_positive_key_rate
            )

        # These are the error rates corresponding to the X and Z basis
        # A bound is used for delta_phase (Z) errors
        delta_bit = error_rate_lambda
        delta_phase = error_rate_lambda + delta(
            gain_lambda * self.number_of_pulses, delta_bit, epsilon_1
        )

        # Only the pulses in the X-basis are used to get key-material
        usable_pulses_lp = n_X_observed * (one_minus_h(delta_phase) - h(delta_bit))
        key_rate = float((1 - p_abort) * usable_pulses_lp / self.number_of_pulses)

        self.last_positive_key_rate = key_rate
        self.last_x = x
        return key_rate

    def _extract_parameters(
        self, x: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Extract the parameters and assigns them correspondingly."""
        return {
            "mu": np.array([x[0]]),
            "probability_basis_X": np.array([x[1]]),
            "probability_basis_Z": np.array([x[2]]),
        }

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
            x0: Initial search value
            bounds: Bounds on search range
        Returns:
            Optimized x=[intensity, probability_basis_X, probability_basis_Z]
            and found optimal key-rate

        Raises:
            ValueError: when x0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """
        if bounds is None:
            lower_bound = np.array([0.01, 0, 0])
            upper_bound = np.array([0.8, 1, 1])
        else:
            lower_bound = np.array([bound[0] for bound in bounds])
            upper_bound = np.array([bound[1] for bound in bounds])
            if len(lower_bound) != 3 or len(upper_bound) != 3:
                error_msg = (
                    "Invalid dimensions input bounds. Expected 3 upper- and lower"
                    f" bound. Received {len(lower_bound)} lower- and {len(upper_bound)}"
                    f" upper bounds."
                )
                raise ValueError(error_msg)

        bounds = scipy.optimize.Bounds(lower_bound, upper_bound)

        if x0 is None:
            x0 = (lower_bound + upper_bound) / 2
        x0 = np.asarray(x0, dtype=float)
        if len(x0) != 3:
            error_msg = (
                "Invalid number of inputs. Expected 3 inputs."
                f"Received {len(x0)} inputs."
            )
            raise ValueError(error_msg)
        # The probabilities are normalized to 1
        x0[1:] /= x0[1:].sum()

        self.last_positive_key_rate = -1.0
        self.last_x = x0

        bounds = scipy.optimize.Bounds(lower_bound, upper_bound)
        # This forces the probabilities to sum to one
        b = np.ones(1)
        A = np.array([0, 1, 1])
        B = np.ones(1)

        constraint = scipy.optimize.LinearConstraint(A, b, B)

        args = {"attenuation": attenuation}
        num_attempts = 3
        for _ in range(num_attempts):  # Maximum 3 retries
            try:
                res = scipy.optimize.minimize(
                    self._f,
                    x0,
                    args=args,
                    constraints=constraint,
                    bounds=bounds,
                    **NLP_CONFIG,
                )
            except OptimizationError:  # noqa: PERF203 Redo calculation
                args["attenuation"] += 1e-8
            else:
                rate = np.atleast_1d(-res.fun)[0]
                if rate < 0:  # Retry when negative key-rate is found.
                    args["attenuation"] += 1e-8
                    continue
                return self._extract_parameters(res.x), float(rate)
        error_msg = "Unable to find solution for optimal key rate."
        raise OptimizationError(error_msg)
