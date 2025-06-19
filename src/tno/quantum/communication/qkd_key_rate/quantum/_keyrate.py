"""Base class for KeyRate objects."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from tno.quantum.communication.qkd_key_rate.quantum import Detector


def _fallback_key_rate_estimate(
    x: NDArray[np.float64],
    last_x: NDArray[np.float64],
    last_positive_key_rate: float,
) -> float:
    """Compute a key rate close to last found positive key rate.

    During the optimization routine, it is possible that a solution is found that has
    intensities that fall outside valid parameter ranges. In those cases we use this
    function to push the parameters back to the valid regime by assigning a key-rate
    close to last positive key rate.

    Args:
        x: Current input parameters.
        last_x: Last valid input parameters for which positive key rate was found.
        last_positive_key_rate: Key rate corresponding to last valid input parameters.

    Returns:
        Estimate of key-rate based based on difference between input parameters and the
            last seen positive key rate.
    """
    return last_positive_key_rate - float(np.linalg.norm(x - last_x))


class KeyRate(metaclass=abc.ABCMeta):
    """Key rate base class."""

    def __init__(self, detector: Detector, **kwargs: Any) -> None:
        """Init of KeyRate.

        Args:
            detector: The detector used at Bob's side
            kwargs: Protocol specific input
        """
        self.detector = detector
        self.__dict__.update(**kwargs)

    @abc.abstractmethod
    def optimize_rate(
        self,
        *,
        attenuation: float,
        x0: ArrayLike | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], float]:
        """Find the parameter setting that achieves the highest possible key-rate.

        Args:
            attenuation: Loss in dB for the channel
            x0: Initial search value
            bounds: Bounds on search range, provided by list of tuples containing lower-
                and upper bounds.

        Returns:
            Optimized arguments and key-rate

        Raises:
            ValueError: when x0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """

    @abc.abstractmethod
    def _extract_parameters(
        self, x: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Extract compute rate parameters from array."""

    @abc.abstractmethod
    def compute_rate(
        self, mu: float | ArrayLike, attenuation: float, **kwargs: Any
    ) -> float:
        """Computes the key-rate given intensity-settings and an attenuation.

        Args:
            mu: Used intensities
            attenuation: Attenuation of the channel
            kwargs: Depending on protocol more arguments can be supplied

        Returns:
            key-rate

        Raises:
            ValueError: When mu is given with invalid dimensions
        """

    def _f(self, x: ArrayLike, args: dict[str, NDArray[np.float64]]) -> float:
        """This function is minimized."""
        args.update(**self._extract_parameters(x))  # type: ignore[arg-type]
        return -self.compute_rate(**args)  # type: ignore[arg-type]


class AsymptoticKeyRateEstimate(KeyRate, metaclass=abc.ABCMeta):
    """Asymptotic key rate base class."""

    @abc.abstractmethod
    def compute_rate(self, *, mu: float | ArrayLike, attenuation: float) -> float:  # type: ignore[override]
        """Computes the key-rate given intensity-settings and an attenuation.

        Args:
            mu: Used intensities
            attenuation: Attenuation of the channel

        Returns:
            key-rate

        Raises:
            ValueError: When mu is given with invalid dimensions
        """


class FiniteKeyRateEstimate(KeyRate, metaclass=abc.ABCMeta):
    """Finite key rate base class."""

    @abc.abstractmethod
    def compute_rate(  # type: ignore[override]
        self,
        *,
        mu: float | ArrayLike,
        attenuation: float,
        probability_basis_X: ArrayLike,
        probability_basis_Z: ArrayLike,
        n_X: int | None = None,
    ) -> float:
        """Computes the key-rate given intensity-settings and an attenuation.

        Args:
            mu: Used intensities
            attenuation: Attenuation of the channel
            probability_basis_X: Probabilities for each intensity in the X-basis
            probability_basis_Z: Probabilities for each intensity in the Z-basis
            n_X: Number of pulses in the X-basis. If not provided, will be estimated
                from gain, probability_basis_X and total number of pules.

        Returns:
            key-rate

        Raises:
            ValueError: When mu is given with invalid dimensions
        """
