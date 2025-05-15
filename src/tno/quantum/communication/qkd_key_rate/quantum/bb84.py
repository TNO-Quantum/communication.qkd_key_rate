# type: ignore

"""Classes to perform key error rate estimate for the BB84 QKD protocol.

This code is based on TNO's BB84 key-rate paper (doi: `10.1007/s11128-021-03078-0`_):

.. _10.1007/s11128-021-03078-0: https://doi.org/10.1007/s11128-021-03078-0

Quantum Key Distribution (QKD) protocols rely on an entangled photon source that
produces entangled photon pairs, which are distributed over two parties. Both parties
randomly choose one of two predefined measurement bases. As the photons are entangled,
non-random results will only be obtained for specific combinations of basis choices.
Detection events are sifted, where only the detection events corresponding to events
where both parties measured in an appropriate basis should be kept. Part of the
resulting sifted key is made public which can be used to estimate the key error rate.

The famous BB84 protocol by Charles Bennett and Gilles Brassard for establishing a
secure key between two parties, usually called Alice and Bob. Alice prepares a quantum
state in one of four ways, and Bob measures the quantum state in one of two ways. Based
on the way of measuring alone, both Alice and Bob can establish a key (assuming
noiseless operations). Classical post-processing routines can correct potential errors
still occurring and can detect eavesdroppers.

We consider three cases:

- Fully Asymptotic Key Rate
    Both the number of pulses and the number of used decoy states is infinite. Because
    of the asymptotic number of pulses and decoy states, we can simplify the
    computations and instead work with a single intensity setting which we vary.

    >>> from tno.quantum.communication.qkd_key_rate.quantum.bb84 import (
    ...     BB84FullyAsymptoticKeyRateEstimate,
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
    >>> fully_asymptotic_key_rate = BB84FullyAsymptoticKeyRateEstimate(detector=detector)
    >>> x, rate = fully_asymptotic_key_rate.optimize_rate(attenuation=0.2)
    >>> print(f"{x['mu']=}, {rate=}")  # doctest: +SKIP
    x['mu']=array([0.89982647]), rate=0.3494369214460756

- Asymptotic Key Rate
    Only the number of pulses is asymptotic, the number of decoy states is finite and
    chosen by the user. We have to optimize the probabilities for the X- and Z-basis
    for each intensity setting. So with two additional decoy states, we have three
    intensity settings to optimize and six probabilities in total. We use linear
    programs (LPs) for this.

    >>> from tno.quantum.communication.qkd_key_rate.quantum.bb84 import (
    ...     BB84AsymptoticKeyRateEstimate,
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
    >>> asymptotic_key_rate = BB84AsymptoticKeyRateEstimate(
    ...     detector=detector, number_of_decoy=3
    ... )
    >>> x, rate = asymptotic_key_rate.optimize_rate(
    ...     attenuation=0.2,
    ...     x0=[0.5, 0.3, 0.2, 0.4],
    ...     bounds=[(0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9)],
    ... )
    >>> print(f"{x['mu']=}, {rate=}")  # doctest: +SKIP
    x['mu']=array([0.8998254 , 0.45375467, 0.21464554, 0.13087223]), rate=0.3494368339817865

- Finite Key Rate:
    Both the number of pulses and the number of decoy states is finite and chosen by
    the user. The approach is similar to the asymptotic key rate module, however we
    have to take finite-key effects into account. We compute bounds on the effect of
    the finite key size and we use security parameters to impose a degree of certainty
    of these bounds.

    >>> from tno.quantum.communication.qkd_key_rate.quantum.bb84 import (
    ...     BB84FiniteKeyRateEstimate,
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
    >>> finite_key_rate = BB84FiniteKeyRateEstimate(detector=detector)
    >>> x, rate = finite_key_rate.optimize_rate(attenuation=0.2)
    >>> print(f"{x['mu']=}, {rate=}")  # doctest: +SKIP
    x['mu']=array([9.80057214e-01, 5.76948581e-02, 2.95649571e-06]), rate=0.3359095058745817
"""  # noqa: E501

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from tno.quantum.communication.qkd_key_rate._utils import binary_entropy as h
from tno.quantum.communication.qkd_key_rate.quantum._config import (
    LP_CONFIG,
    NLP_CONFIG,
    OptimizationError,
)
from tno.quantum.communication.qkd_key_rate.quantum._keyrate import (
    AsymptoticKeyRateEstimate,
    FiniteKeyRateEstimate,
)

if TYPE_CHECKING:
    from tno.quantum.communication.qkd_key_rate.quantum import Detector

warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)


def ensure_probability(p: ArrayLike) -> NDArray[np.float64]:
    """Ensure that we have a probability between zero and one.

    Other functions will otherwise throw an error.

    Args:
        p: Probability to be mapped to range $[0, 1]$.

    Returns:
        Probability
    """
    return np.clip(p, 0, 1)


def compute_gain_and_error_rate(
    detector: Detector, mu: ArrayLike, attenuation: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes the total gain and error rate of the channel.

    Args:
        detector: The used detector on Bob's side
        mu: Intensities of the laser
        attenuation: Loss of the channel

    Returns:
        - The gain per intensity. The probability of an event, given a pulse.
        - The error rate per intensity.
    """
    dark_count_rate = detector.dark_count_rate
    efficiency_bob = detector.efficiency_party
    polarization_drift = detector.polarization_drift

    mu = np.atleast_1d(mu)

    efficiency_channel = np.power(10, -attenuation / 10)
    coefficient = -1 * mu * efficiency_bob * efficiency_channel

    # Compute the overall gain and error rate.
    # For stability and accuracy reasons, the error_rate is computed in two steps
    gain = 1 - np.power(1 - dark_count_rate, 2) * np.exp(coefficient)
    error_rate_denominator = 2 * gain
    error_rate_numerator = (
        1
        + (1 - dark_count_rate)
        * (
            np.exp(coefficient * np.power(np.cos(polarization_drift), 2))
            - np.exp(coefficient * np.power(np.sin(polarization_drift), 2))
        )
        - np.exp(coefficient) * np.power(1 - dark_count_rate, 2)
    )
    error_rate = error_rate_numerator / error_rate_denominator
    error_rate[np.isnan(error_rate)] = 0
    return gain, error_rate


def lower_bound_matrix_gain(
    max_num_photons: int, mus: ArrayLike
) -> NDArray[np.float64]:
    """Computes a lower bound on the likeliness of the number of photons per intensity.

    Args:
        max_num_photons: Maximum on the number of photons per pulse to consider
        mus: All used intensities
    """
    mus = np.atleast_1d(mus)
    i, mu = np.meshgrid(range(max_num_photons + 1), mus)
    return scipy.stats.poisson.pmf(i, mu)


def bound_f(
    number_of_pulses: int, probability: float, epsilon: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Computes a bound used in the finite key-rate computations.

    Args:
        number_of_pulses: Number of pulses considered
        probability: Probability of the event for which the bound-term is computed
        epsilon: Security-parameter

    Returns:
        Bounds to be used in finite-key rate computations.
    """
    return -np.log(epsilon) * (
        1 + np.sqrt(1 - 2 * probability * number_of_pulses / np.log(epsilon))
    )


def delta(n_x: int, n_z: int, e1: float) -> float:
    """Computes bound based on the number of pulses sent and epsilon-security value.

    Args:
        n_x: Number of pulses in the X-basis
        n_z: Number of pulses in the Z-basis
        e1: Epsilon-security parameter

    Returns:
        Bound.
    """
    return float(
        np.sqrt(np.max([-(n_x + n_z) * (n_x + 1) * np.log(e1) / (2 * n_z * n_x**2), 0]))
    )


def delta_ec(p_abort: float, n_x: int) -> float:
    """Computes a bound on the losses due to error correction.

    Args:
        p_abort: Abort probability, used if there are too many errors
        n_x: Number of pulses in the X-basis

    Returns:
        Bound.
    """
    return float(np.sqrt(np.log(2 / p_abort) * 3 * np.log2(5) ** 2 / n_x))


def solve_lp(
    target_vacuum: float,
    target_single: float,
    mu: NDArray[np.float64],
    program_coefficients: NDArray[np.float64],
    max_num_photons: int,
) -> scipy.optimize.OptimizeResult:
    """Solves the linear program (LP) for the asymptotic case.

    Args:
        target_vacuum: Coefficient for the vacuum state term
        target_single: Coefficient for the single photon state terms
        mu: The used intensity
        program_coefficients: The coefficients in the LP, e.g., gain,
            error-rate or their product
        max_num_photons: The number of photons at which the sums are cut

    Raises:
        ValueError: If intensity not same size as program coefficients.
        OptimizationError: in case no solution was found for the LP.
    """
    if len(mu) != len(program_coefficients):
        error_msg = "Intensity not compatible with program coefficients."
        raise ValueError(error_msg)

    # Determine the LP-constraints
    Lb = lower_bound_matrix_gain(max_num_photons, mu)
    Uv = 1 - np.sum(Lb, axis=1)

    lower_bound = np.zeros(max_num_photons + 1)
    upper_bound = np.ones(max_num_photons + 1)

    args = dict(  # noqa: C408
        c=np.hstack((target_vacuum, target_single, np.zeros(max_num_photons - 1))),
        A_ub=np.vstack((Lb, -Lb)),
        b_ub=np.concatenate((program_coefficients, -program_coefficients + Uv)),
        bounds=np.vstack((lower_bound, upper_bound)).T,
    )

    # Optimize the LP
    result = scipy.optimize.linprog(**args, **LP_CONFIG)
    if result.status != 0:
        error_msg = f"No solution was found for the linear problem. {result.message}"
        raise OptimizationError(error_msg)
    return result


def solve_finite_lp(  # noqa: PLR0913
    target_vacuum: float,
    target_single: float,
    probabilities_intensity_j: NDArray[np.float64],
    mu: NDArray[np.float64],
    max_num_photons: int,
    number_of_pulses: int,
    observed_count: NDArray[np.float64],
    epsilon_mu_j: NDArray[np.float64],
    epsilon_num_photons_M: float,
    epsilon_num_photons_M_in_basis_B: NDArray[np.float64],
) -> scipy.optimize.OptimizeResult:
    r"""Solves the linear program (LP) for the finite case.

    Args:
        target_vacuum: Coefficient for the vacuum state term
        target_single: Coefficient for the single photon state terms
        probabilities_intensity_j: Probability for each decoy state
        mu: The used intensity
        max_num_photons: The number of photons at which the sums are cut
        number_of_pulses: Number of pulses sent in specific basis
        observed_count: Number of pulses observed in specific basis per intensity
        epsilon_mu_j: Epsilon terms for the intensities
        epsilon_num_photons_M: Epsilon terms for the number of photons
        epsilon_num_photons_M_in_basis_B: Epsilon terms for the number of photons in
            basis B

    Returns:
        Number of usable pulses

    The variables in the LP are
        - $n_0$: Number of vacuum pulses
        - $n_1$: Number of single photon pulses
        - $\ldots$
        - $n_M$: Number of pulses with M pulses
        - $\delta_{j,1}$:  Deviation for intensity 1
        - $\delta_{j,2}$:  Deviation for intensity 2
        - $\ldots$
        - $\delta_{j,m}$:  Deviation for intensity m
    """
    # Total number of events observed
    observed_count = np.asarray(observed_count)
    observed_total = observed_count.sum()

    # Compute the probability to have an $m$ photon state, given you use intensity $j$
    probability_m_photon_state_given_intensity_j = scipy.stats.poisson.pmf(
        *np.meshgrid(range(max_num_photons + 1), mu)
    )
    # And compute the probability to have an $m$ photon state at all
    # (independent on the used intensity)
    probabilities_intensity_j = np.asarray(probabilities_intensity_j)
    probability_m_photon_state = probabilities_intensity_j.dot(
        probability_m_photon_state_given_intensity_j
    )
    probability_m_photon_state = np.clip(probability_m_photon_state, 1e-150, 1)

    # Similarly, compute the other conditional probabilities:
    # The probability to have intensity $j$, given that you send an $m$ photon state
    probability_intensity_j_given_m_photon_state = (
        probability_m_photon_state_given_intensity_j
        * np.outer(probabilities_intensity_j, 1 / probability_m_photon_state)
    )

    number_of_intensities = len(mu)
    # Compute the probability that a photon state is send out with more than
    # the defined maximum number of photons.
    tail_probability_maximum_photons = float(
        scipy.stats.poisson.sf(max_num_photons + 1, mu).dot(probabilities_intensity_j)
    )

    # Bound on the contribution for the number of photons above the cut
    lambda_num_photons_cut = (
        tail_probability_maximum_photons * number_of_pulses
        + bound_f(
            number_of_pulses,
            tail_probability_maximum_photons,
            epsilon_num_photons_M_in_basis_B,
        )
    )

    # Below, constraints are defined. The first two letters of the variables
    # define the type of constraint: lb = lower bound, ub = upper bound, eq = equality
    # Constraints on the number of $m$ photon states per intensity $j$
    lb_num_photons_m_given_intensity_j_lhs = -np.hstack(
        (probability_intensity_j_given_m_photon_state, np.eye(number_of_intensities))
    )
    lb_num_photons_m_given_intensity_j_rhs = -observed_count
    ub_num_photons_m_given_intensity_j_lhs = np.hstack(
        (probability_intensity_j_given_m_photon_state, np.eye(number_of_intensities))
    )
    ub_num_photons_m_given_intensity_j_rhs = observed_count + lambda_num_photons_cut

    # Constraints on the number of $m$ photon states
    lb_num_photons_m_lhs = -np.eye(
        max_num_photons + 1, max_num_photons + number_of_intensities + 1
    )
    lb_num_photons_m_rhs = np.zeros(max_num_photons + 1)
    ub_num_photons_m_lhs = np.eye(
        max_num_photons + 1, max_num_photons + number_of_intensities + 1
    )
    ub_num_photons_m_rhs = np.minimum(
        (
            probability_m_photon_state * number_of_pulses
            + bound_f(
                number_of_pulses,
                probability_m_photon_state,
                epsilon_num_photons_M,
            )
            * number_of_pulses
        ),
        np.repeat(observed_total, max_num_photons + 1),
    )

    # Constraints on the states where intensity $j$ is used
    intensity_j_rhs = np.sqrt(-np.log(epsilon_mu_j / 2) * observed_total / 2)
    intensity_j_lhs = np.eye(
        number_of_intensities,
        max_num_photons + number_of_intensities + 1,
        max_num_photons + 1,
    )
    lb_intensity_j_rhs = intensity_j_rhs
    ub_intensity_j_rhs = intensity_j_rhs
    lb_intensity_j_lhs = intensity_j_lhs
    ub_intensity_j_lhs = -intensity_j_lhs

    # Constraint that the states per intensity should together sum to the observed count
    eq_sum_of_intensities_j_lhs = np.hstack(
        (np.zeros(max_num_photons + 1), np.ones(number_of_intensities))
    )
    eq_sum_of_intensities_j_rhs = 0

    args = dict(  # noqa: C408
        c=np.hstack(
            (
                target_vacuum,
                target_single,
                np.zeros(max_num_photons + number_of_intensities - 1),
            )
        ),
        A_ub=np.vstack(
            (
                lb_num_photons_m_given_intensity_j_lhs,
                ub_num_photons_m_given_intensity_j_lhs,
                lb_num_photons_m_lhs,
                ub_num_photons_m_lhs,
                lb_intensity_j_lhs,
                ub_intensity_j_lhs,
            )
        ),
        b_ub=np.hstack(
            (
                lb_num_photons_m_given_intensity_j_rhs,
                ub_num_photons_m_given_intensity_j_rhs,
                lb_num_photons_m_rhs,
                ub_num_photons_m_rhs,
                lb_intensity_j_rhs,
                ub_intensity_j_rhs,
            )
        )
        / number_of_pulses,
        A_eq=np.atleast_2d(eq_sum_of_intensities_j_lhs),
        b_eq=eq_sum_of_intensities_j_rhs / number_of_pulses,
        bounds=(None, None),
    )

    result = scipy.optimize.linprog(**args, **LP_CONFIG)
    if result.status != 0:
        error_msg = f"No solution was found for the linear problem. {result.message}"
        raise OptimizationError(error_msg)
    return result


class BB84FullyAsymptoticKeyRateEstimate(AsymptoticKeyRateEstimate):
    """The situation for an asymptotic number of intensity settings and pulses.

    In the fully asymptotic case we only have to consider a single intensity
    """

    def __init__(self, detector: Detector, **kwargs: Any) -> None:
        """Init of the BB84FullyAsymptoticKeyRateEstimate.

        Args:
            detector: The detector used at Bob's side
            kwargs: Protocol specific input
        """
        super().__init__(detector=detector, args=kwargs)

    def compute_rate(self, mu: float | ArrayLike, attenuation: float) -> float:  # type: ignore[override]
        """Computes the key-rate given an intensity and an attenuation.

        Only the vacuum states and single photon states can be safely used.
        The error rate for vacuum states is 0.5. For single photon states we
        must upper bound it.

        Args:
            mu: Intensity
            attenuation: Attenuation

        Returns:
            Key-rate
        """
        mu = np.atleast_1d(mu)
        detector = self.detector

        efficiency_bob = detector.efficiency_party
        efficiency_channel = np.power(10, -attenuation / 10)
        efficiency_system = efficiency_bob * efficiency_channel

        gain, error_rate = compute_gain_and_error_rate(detector, mu, attenuation)

        # Compute the relevant terms for vacuum and single photon states
        yield_vacuum = 1 - np.power(1 - detector.dark_count_rate, 2)
        yield_single = 1 - (
            np.power(1 - detector.dark_count_rate, 2) * (1 - efficiency_system)
        )

        yield_times_error_single = (
            yield_single
            - (1 - detector.dark_count_rate)
            * efficiency_system
            * np.cos(2 * detector.polarization_drift)
        ) / 2
        error_single = yield_times_error_single / yield_single
        # The probabilities of having precisely zero or one photon in a state
        (
            probability_vacuum_basis_X,
            probability_single_basis_X,
        ) = scipy.stats.poisson.pmf([0, 1], mu[0])

        # Pulses in Z-basis are used for error estimation, the pulses in the
        # X-basis are used to obtain key-material
        h_entropy_single_basis_Z = 1 - h(error_single)

        gain_times_error_basis_X = gain * h(error_rate)

        return float(
            probability_vacuum_basis_X * yield_vacuum
            + (probability_single_basis_X * yield_single * h_entropy_single_basis_Z)
            - gain_times_error_basis_X
        )

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

        Raises:
            ValueError: When x0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.

        Returns:
            Optimized intensity and key-rate
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


class BB84AsymptoticKeyRateEstimate(AsymptoticKeyRateEstimate):
    """The situation for an asymptotic number of pulses.

    We consider a fixed number of intensities (number_of_decoy + 1)
    """

    def __init__(
        self, detector: Detector, number_of_decoy: int = 2, **kwargs: Any
    ) -> None:
        """Init of BB84AsymptoticKeyRateEstimate.

        Args:
            detector:  The detector used at Bob's side
            number_of_decoy: Number of decoy intensities used
            kwargs: protocol specific arguments.
        """
        super().__init__(detector=detector, args=kwargs)
        self.number_of_decoy = number_of_decoy
        self.last_positive: float = -1
        self.last_x: NDArray[np.float64]

    def _compute_last_positive_distance(self, x: NDArray[np.float64]) -> float:
        """Computes the last positive distance.

        The optimization routine sometimes considers a parameter setting
        outside of the valid region. This function is used to push the
        parameters back to the valid regime.
        """
        if self.last_positive > -1:
            return self.last_positive - float(np.linalg.norm(x - self.last_x))
        return self.last_positive

    def compute_rate(self, mu: float | ArrayLike, attenuation: float) -> float:  # type: ignore[override]
        """Computes the key-rate given intensity-settings and an attenuation.

        Args:
            mu: Intensity
            attenuation: Attenuation

        Returns:
            Key-rate
        """
        mu = np.atleast_1d(mu)
        x = np.asarray(mu)

        if x.min() < 0 or x.max() > 1:
            # If the variable x is outside the possible range, push it back
            return self._compute_last_positive_distance(x)

        # Maximum on the number of photons per pulse to consider
        max_num_photons = np.max((int(scipy.stats.poisson.isf(1e-12, mu.max())), 5))

        gain, error_rate = compute_gain_and_error_rate(self.detector, mu, attenuation)

        # Compute the yield and error rate for single photon pulses in the Z basis
        yield_single_basis_Z = ensure_probability(
            solve_lp(
                target_vacuum=0,
                target_single=1,
                mu=mu,
                program_coefficients=gain,
                max_num_photons=max_num_photons,
            )["fun"]
        )
        yield_times_error_basis_Z = ensure_probability(
            -solve_lp(
                target_vacuum=0,
                target_single=-1,
                mu=mu,
                program_coefficients=gain * error_rate,
                max_num_photons=max_num_photons,
            )["fun"]
        )

        if yield_single_basis_Z == 0:
            error_single_basis_Z = 1e-19
        else:
            error_single_basis_Z = np.clip(
                yield_times_error_basis_Z / yield_single_basis_Z, 1e-19, 0.5
            )
        h_entropy_single_basis_Z = 1 - h(error_single_basis_Z)

        # Get the probability for vacuum and single photon pulses in the X basis
        (
            probability_vacuum_basis_X,
            probability_single_basis_X,
        ) = scipy.stats.poisson.pmf([0, 1], mu[0])

        # Compute the yield for vacuum and single photon pulses in the X basis
        yield_vacuum_single_basis_X = float(
            solve_lp(
                target_vacuum=probability_vacuum_basis_X,
                target_single=probability_single_basis_X * h_entropy_single_basis_Z,
                mu=mu,
                program_coefficients=gain,
                max_num_photons=max_num_photons,
            )["fun"]
        )
        if np.isnan(yield_vacuum_single_basis_X) or np.isinf(
            yield_vacuum_single_basis_X
        ):
            return self._compute_last_positive_distance(x)

        # Determine the overall key-rate
        gain_times_error_basis_X = float(gain[0] * h(error_rate[0]))
        key_rate = yield_vacuum_single_basis_X - gain_times_error_basis_X

        self.last_positive = key_rate
        self.last_x = np.asarray(mu)

        return float(key_rate)

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

        For certain input parameters it can happen that the resulting lp problem is
        unfeasible. In that case the attenuation is slightly modified (``+1e-8``) in an
        attempt to obtain a feasible lp problem that can be solved.

        Args:
            attenuation: Loss in dB for the channel
            x0: Initial search value, default midpoint search bounds
            bounds: Bounds on search range

        Returns:
            Optimized intensity and key-rate

        Raises:
            ValueError: When x0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
            OptimizationError: When lp solver is unsuccessful due to infeasible problem.
                Multiple attempts are made with slightly modified  attenuation before
                error is raised.
        """
        if bounds is None:
            # Lower and upper bounds on the considered laser intensities
            lower_bound = 0.00000001 * np.ones(self.number_of_decoy + 1)
            lower_bound[0] = 0.2  # Help the optimizer
            upper_bound = 0.95 * np.ones(self.number_of_decoy + 1)
            upper_bound[-1] = 0.3  # Help the optimizer
        else:
            lower_bound = np.array([bound[0] for bound in bounds])
            upper_bound = np.array([bound[1] for bound in bounds])
            if len(bounds) != self.number_of_decoy + 1:
                error_msg = (
                    f"Invalid dimensions input bounds. Expected "
                    f"{self.number_of_decoy + 1} upper- and lower bounds. Received "
                    f"{len(bounds)} bounds."
                )
                raise ValueError(error_msg)
        optimize_bounds = scipy.optimize.Bounds(lower_bound, upper_bound)

        less_one = 1 - 1e-16

        self.last_positive = -1
        # If no ansatz is given, choose the mean of the bounds.
        if x0 is None:
            x0 = (lower_bound + upper_bound) / 2
        x0 = np.asarray(x0)
        if len(x0) != self.number_of_decoy + 1:
            error_msg = (
                "Invalid number of inputs. Expected "
                f"{self.number_of_decoy + 1} intensities. Received "
                f"{len(x0)} intensities."
            )
            raise ValueError(error_msg)
        self.last_x = x0

        # The constraints enforce the intensities to be decreasing
        b = np.repeat(-np.inf, self.number_of_decoy)
        A = np.triu(
            np.ones((self.number_of_decoy, self.number_of_decoy + 1)), 1
        ) - less_one * np.eye(self.number_of_decoy, self.number_of_decoy + 1)
        B = np.zeros(self.number_of_decoy)
        constraint = scipy.optimize.LinearConstraint(A, b, B)

        args = {"attenuation": attenuation}
        num_attempts = 3
        for _ in range(num_attempts):  # Maximum 3 retries
            try:
                res = scipy.optimize.minimize(
                    self._f,
                    x0,
                    args=args,
                    constraints=(constraint),
                    bounds=optimize_bounds,
                    **NLP_CONFIG,
                )
            except OptimizationError:  # noqa: PERF203
                args["attenuation"] += 1e-8
            else:
                rate = np.atleast_1d(-res.fun)[0]
                if rate < 0:  # Retry when negative key-rate is found.
                    args["attenuation"] += 1e-8
                    continue
                return self._extract_parameters(res.x), float(rate)
        error_msg = "Unable to find solution for optimal key rate."
        raise OptimizationError(error_msg)


class BB84FiniteKeyRateEstimate(FiniteKeyRateEstimate):
    """BB84 protocol with a finite number of pulses.

    A fixed number of intensities is considered.
    The probabilities for both bases might vary.
    """

    def __init__(
        self,
        detector: Detector,
        number_of_pulses: int = int(1e12),
        number_of_decoy: int = 2,
        **kwargs: Any,
    ) -> None:
        """Init of the BB84FiniteKeyRateEstimate.

        Args:
            detector: The detector used at Bob's side
            number_of_pulses: Number of pulses sent
            number_of_decoy: Number of decoy intensities used
            kwargs: Any protocol dependent parameters.
        """
        super().__init__(detector=detector, args=kwargs)
        self.number_of_pulses = number_of_pulses
        self.number_of_decoy = number_of_decoy
        self.last_positive: float = -1
        self.last_x: NDArray[np.float64]

    def _compute_last_positive_distance(self, x: NDArray[np.float64]) -> float:
        """Computes the last positive distance.

        The optimization routine sometimes considers a parameter setting
        outside of the valid region. This function is used to push the
        parameters back to the valid regime.
        """
        if self.last_positive > -1:
            return self.last_positive - float(np.linalg.norm(x - self.last_x))
        return self.last_positive

    def compute_rate(  # type: ignore[override]
        self,
        mu: float | ArrayLike,
        attenuation: float,
        probability_basis_X: ArrayLike,
        probability_basis_Z: ArrayLike,
        n_X: int | None = None,
    ) -> float:
        """Compute the key-rate for a specific set of parameters.

        Args:
            mu: Used intensities
            attenuation: Attenuation of the channel
            probability_basis_X: Probabilities for each intensity in the X-basis
            probability_basis_Z: Probabilities for each intensity in the Z-basis
            n_X: Number of pulses in the X-basis.

        Returns:
            key-rate
        """
        # Security parameters
        probability_abort = 2e-50
        epsilon_security = 2e-50
        epsilon_correct = 2e-50
        epsilon_1 = 2e-55
        epsilon_2 = 2e-55
        epsilon_3 = 2e-55
        epsilon = 1e-7

        # The probabilities sum to 1, these are the conditional probabilities per basis
        probability_basis_X = np.asarray(probability_basis_X)
        probability_basis_Z = np.asarray(probability_basis_Z)
        probability_basis_X_normalized: NDArray[np.float64] = (
            probability_basis_X / probability_basis_X.sum()
        )
        probability_basis_Z_normalized: NDArray[np.float64] = (
            probability_basis_Z / probability_basis_Z.sum()
        )

        mu = np.asarray(mu)
        x = np.hstack((mu, probability_basis_X, probability_basis_Z))
        if x.min() < 0 or x.max() > 1:
            return self._compute_last_positive_distance(x)

        gain, error_rate = compute_gain_and_error_rate(self.detector, mu, attenuation)

        # Compute number of detection events per intensity per basis
        if n_X is None:
            n_X_observed_per_intensity: NDArray[np.float64] = (
                gain * ensure_probability(probability_basis_X) * self.number_of_pulses
            )
            n_Z_observed_per_intensity: NDArray[np.float64] = (
                gain * ensure_probability(probability_basis_Z) * self.number_of_pulses
            )
        else:
            n_X_observed_per_intensity = gain * probability_basis_X_normalized * n_X
            n_Z_observed_per_intensity = (
                gain * probability_basis_Z_normalized * (self.number_of_pulses - n_X)
            )

        n_X_observed = float(n_X_observed_per_intensity.sum())
        n_Z_observed = float(n_Z_observed_per_intensity.sum())

        if n_X_observed == 0 or n_Z_observed == 0:
            return self._compute_last_positive_distance(x)

        # And their corresponding number of errors
        number_of_errors_per_intensity_basis_X = error_rate * n_X_observed_per_intensity
        number_of_errors_per_intensity_basis_Z = error_rate * n_Z_observed_per_intensity
        number_of_errors_basis_X = float(number_of_errors_per_intensity_basis_X.sum())

        # Maximum on the number of photons per pulse to consider
        max_num_photons = int(
            np.max((int(scipy.stats.poisson.isf(1e-16, mu).max()), 5))
        )

        # Epsilon values to be considered in the optimization.
        # Values for the intensities and the number of photons per pulse are used,
        # as well as the number of photon per pulse given a basis
        epsilon_single_photon = 1e-7
        epsilon_intensity_j: NDArray[np.float64] = np.repeat(1e-8, mu.shape[0])
        epsilon_num_photons_M: NDArray[np.float64] = np.repeat(
            1e-8, int(max_num_photons + 1)
        )
        epsilon_num_photons_M_in_basis_B: NDArray[np.float64] = np.repeat(
            1e-8, int(mu.shape[0])
        )

        # The probability of a single photon pulse for each of the intensities
        probability_single_photon_per_intensity: NDArray[np.float64] = (
            scipy.stats.poisson.pmf(1, mu)
        )
        probability_single_photon_per_intensity: NDArray[np.float64] = (
            probability_single_photon_per_intensity
            / probability_single_photon_per_intensity.sum()
        )

        # Solve linear programs to get the number of single photon events
        # and the number of errors in these single photon events in the Z basis
        # For vacuum pulses, the error is 0.5
        (_, number_single_photon_events_in_basis_Z) = (
            solve_finite_lp(
                target_vacuum=0,
                target_single=1,
                probabilities_intensity_j=probability_basis_Z_normalized,
                mu=mu,
                max_num_photons=max_num_photons,
                number_of_pulses=n_Z_observed,
                observed_count=n_Z_observed_per_intensity,
                epsilon_mu_j=epsilon_intensity_j,
                epsilon_num_photons_M=epsilon_num_photons_M,
                epsilon_num_photons_M_in_basis_B=epsilon_num_photons_M_in_basis_B,
            )["x"][0:2]
            * n_Z_observed
        )
        (_, number_single_photon_errors_in_basis_Z) = (
            solve_finite_lp(
                target_vacuum=0,
                target_single=1,
                probabilities_intensity_j=probability_basis_Z_normalized,
                mu=mu,
                max_num_photons=max_num_photons,
                number_of_pulses=n_Z_observed,
                observed_count=number_of_errors_per_intensity_basis_Z,
                epsilon_mu_j=epsilon_intensity_j,
                epsilon_num_photons_M=epsilon_num_photons_M,
                epsilon_num_photons_M_in_basis_B=epsilon_num_photons_M_in_basis_B,
            )["x"][0:2]
            * n_Z_observed
        )
        if number_single_photon_events_in_basis_Z == 0:
            error_rate_single_photon_basis_Z = 1e-19
        else:
            error_rate_single_photon_basis_Z = np.clip(
                number_single_photon_errors_in_basis_Z
                / number_single_photon_events_in_basis_Z,
                1e-19,
                0.5,
            )

        # Solve the linear program for vacuum and single photon pulses in X-basis
        # Note the expression for target_single, this takes into account the
        # errors in the Z-basis and the number of pulses in both.

        res = solve_finite_lp(
            target_vacuum=1,
            target_single=1
            - h(
                error_rate_single_photon_basis_Z
                + delta(
                    n_X_observed,
                    n_Z_observed,
                    epsilon_single_photon,
                )
            ),
            probabilities_intensity_j=probability_basis_X_normalized,
            mu=mu,
            max_num_photons=max_num_photons,
            number_of_pulses=n_X_observed,
            observed_count=n_X_observed_per_intensity,
            epsilon_mu_j=epsilon_intensity_j,
            epsilon_num_photons_M=epsilon_num_photons_M,
            epsilon_num_photons_M_in_basis_B=epsilon_num_photons_M_in_basis_B,
        )

        (
            number_vacuum_events_basis_X,
            number_single_events_basis_X,
        ) = res["x"][0:2] * n_X_observed

        if np.isnan(res["fun"]) or np.isinf(res["fun"]):
            return self._compute_last_positive_distance(x)

        # With the results from the LP, we can determine the number of usable pulses
        number_vacuum_single_pulses = (
            number_vacuum_events_basis_X
            + number_single_events_basis_X
            - number_single_events_basis_X
            * (
                h(
                    number_single_photon_errors_in_basis_Z
                    + delta(
                        n_X_observed,
                        n_Z_observed,
                        epsilon_1,
                    )
                )
            )
        )

        error_rate_basis_X = number_of_errors_basis_X / n_X_observed
        usable_pulses_lp = (
            number_vacuum_single_pulses
            - n_X_observed
            * (h(error_rate_basis_X) + delta_ec(probability_abort, n_X_observed))
            + np.log2(
                epsilon_correct
                * np.power(epsilon_2 * epsilon_3 * (epsilon_security - epsilon), 2)
            )
            - 1
        )

        # With which we can determine the key-rate
        key_rate = (1 - probability_abort) * usable_pulses_lp / self.number_of_pulses

        self.last_positive = key_rate
        self.last_x = x
        return key_rate

    def _extract_parameters(
        self, x: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Extract the parameters and assigns them correspondingly."""
        return {
            "mu": np.array(x[0 : self.number_of_decoy + 1]),
            "probability_basis_X": np.array(
                x[self.number_of_decoy + 1 : 2 * self.number_of_decoy + 2]
            ),
            "probability_basis_Z": np.array(x[2 * self.number_of_decoy + 2 :]),
        }

    def optimize_rate(
        self,
        *,
        attenuation: float,
        x0: ArrayLike | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], float]:
        """Function to optimize the key-rate.

        The laser intensities should be ordered and the probabilities should
        sum to one. Probabilities for both X and Z-basis are considered
        simultaneously. We consider the Z-basis for error estimation and the
        X-basis for key-rate estimation, so no sifting rate is considered.

        For certain input parameters it can happen that the resulting lp problem is
        unfeasible. In that case the attenuation is slightly modified (``+1e-8``) in an
        attempt to obtain a feasible lp problem that can be solved.

        Args:
            attenuation: Loss in dB for the channel
            x0: Initial search value
            bounds: Bounds on search range
            args: Other variables to be optimized, for instance the attenuation

        Returns:
            Optimized x=[intensity, probability_basis_X, probability_basis_Z]
            and found optimal key-rate

        Raises:
            ValueError: When x0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
            OptimizationError: When lp solver is unsuccessful due to infeasible problem.
                Multiple attempts are made with slightly modified  attenuation before
                error is raised.
        """
        less_one = 1 - 1e-16

        if bounds is None:
            lower_bound = np.hstack(
                (
                    1e-16 * np.ones((self.number_of_decoy + 1)),
                    np.zeros(2 * self.number_of_decoy + 2),
                )
            )
            upper_bound = np.hstack(
                (
                    np.ones((self.number_of_decoy + 1)),
                    np.ones(2 * self.number_of_decoy + 2),
                )
            )
        else:
            lower_bound = np.array([bound[0] for bound in bounds])
            upper_bound = np.array([bound[1] for bound in bounds])
            if len(bounds) != 3 * (self.number_of_decoy + 1):
                error_msg = (
                    "Invalid dimensions input bounds. Expected "
                    f"{3 * (self.number_of_decoy + 1)} upper- and lower bounds."
                    f"Received {len(bounds)} bounds."
                )
                raise ValueError(error_msg)
        optimize_bounds = scipy.optimize.Bounds(lower_bound, upper_bound)

        if x0 is None:
            x0 = (lower_bound + upper_bound) / 2
        x0 = np.asarray(x0)
        if len(x0) != 3 * (self.number_of_decoy + 1):
            error_msg = (
                "Invalid number of inputs. Expected "
                f"{3 * (self.number_of_decoy + 1)} inputs. Received "
                f"{len(x0)} inputs."
            )
            raise ValueError(error_msg)

        # The probabilities are normalized to 1
        x0[self.number_of_decoy + 1 :] /= x0[self.number_of_decoy + 1 :].sum()

        self.last_positive = -1
        self.last_x = x0

        # The constraints enforce the intensities to be decreasing
        # Furthermore, it enforces that the probabilities are between zero and
        # one and that they together sum to one
        b = np.hstack(
            (
                np.repeat(-np.inf, self.number_of_decoy),
                np.zeros(self.number_of_decoy + 1),
                1,
            )
        )
        A = np.vstack(
            (
                np.flip(
                    np.hstack(
                        (
                            (
                                np.triu(
                                    np.ones(
                                        (
                                            self.number_of_decoy,
                                            self.number_of_decoy + 1,
                                        ),
                                        dtype=int,
                                    ),
                                    1,
                                )
                                - np.hstack(
                                    (
                                        np.diag(
                                            less_one
                                            * np.ones(self.number_of_decoy, dtype=int)
                                        ),
                                        np.zeros((self.number_of_decoy, 1), dtype=int),
                                    )
                                )
                            ),
                            np.zeros(
                                (self.number_of_decoy, 2 * self.number_of_decoy + 2),
                                dtype=int,
                            ),
                        )
                    ),
                    0,
                ),
                np.eye(self.number_of_decoy + 1, 3 * self.number_of_decoy + 3),
                np.hstack(
                    (
                        np.zeros(self.number_of_decoy + 1, dtype=int),
                        np.ones(2 * self.number_of_decoy + 2, dtype=int),
                    )
                ),
            )
        )
        B = np.hstack(
            (
                np.zeros(self.number_of_decoy, dtype=int),
                np.ones(self.number_of_decoy + 2),
            )
        )

        constraint = scipy.optimize.LinearConstraint(A, b, B)
        # In some cases the gradient appears to be zero, indicating a linear function.
        # It is not a linear function, hence we suppress the error
        args = {"attenuation": attenuation}
        num_attempts = 3
        for _ in range(num_attempts):  # Maximum 3 retries
            try:
                res = scipy.optimize.minimize(
                    self._f,
                    x0,
                    args=args,
                    constraints=(constraint),
                    bounds=optimize_bounds,
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
