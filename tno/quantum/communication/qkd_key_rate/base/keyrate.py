"""Base class for KeyRate objects."""
import abc
from typing import Any, Dict, List, Optional, Tuple, Union

from numpy.typing import ArrayLike

from tno.quantum.communication.qkd_key_rate.base import Detector

# pylint: disable=invalid-name


class KeyRate(metaclass=abc.ABCMeta):
    """
    Key rate base class.
    """

    def __init__(self, detector: Detector, **kwargs: int) -> None:
        """
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
        x_0: Optional[ArrayLike] = None,
        bounds: Optional[List[ArrayLike]] = None,
    ) -> Tuple[float, float]:
        """Find the parameter setting that achieves the highest possible key-rate

        Args:
            attenuation: Loss in dB for the channel
            x_0: Initial search value
            bounds: Bounds on search range

        Returns:
            Optimized intensity and key-rate

        Raises:
            ValueError: when x_0 or bounds are given with invalid dimensions.
            ValueError: when the found key-rate is negative.
        """

    @abc.abstractmethod
    def _extract_parameters(self, x: ArrayLike) -> Dict[str, Any]:
        """Extract compute rate parameters from array."""

    @abc.abstractmethod
    def compute_rate(
        self, mu: Union[float, ArrayLike], attenuation: float, **kwargs: Any
    ):
        """Computes the key-rate given intensity-settings and an attenuation

        Args:
            mu: Used intensities
            attenuation: Attenuation of the channel
            kwargs: Depending on protocol more arguments can be supplied

        Returns:
            key-rate

        Raises:
            ValueError: When mu is given with invalid dimensions
        """

    def _f(self, x: ArrayLike, args: Dict[str, float]) -> float:
        """This function is minimized."""
        args.update(**self._extract_parameters(x))
        return -self.compute_rate(**args)


class AsymptoticKeyRateEstimate(KeyRate, metaclass=abc.ABCMeta):
    """
    Asymptotic key rate base class
    """

    @abc.abstractmethod
    def compute_rate(self, *, mu: Union[float, ArrayLike], attenuation: float) -> float:
        """Computes the key-rate given intensity-settings and an attenuation

        Args:
            mu: Used intensities
            attenuation: Attenuation of the channel

        Returns:
            key-rate

        Raises:
            ValueError: When mu is given with invalid dimensions
        """


class FiniteKeyRateEstimate(KeyRate, metaclass=abc.ABCMeta):
    """
    Finite key rate base class
    """

    @abc.abstractmethod
    def compute_rate(
        self,
        *,
        mu: Union[float, ArrayLike],
        attenuation: float,
        probability_basis_X: ArrayLike,
        probability_basis_Z: ArrayLike,
        n_X: Optional[int] = None,
    ) -> float:
        """Computes the key-rate given intensity-settings and an attenuation

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
