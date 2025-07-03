r"""Privacy amplification module.

The :py:class:`~tno.quantum.communication.qkd_key_rate.classical.privacy_amplification.PrivacyAmplification`
class can be used to compute the hash of an error corrected string. The length of the
hashed string equals the remaining entropy.

Typical usage example:

    >>> from tno.quantum.communication.qkd_key_rate.classical import Message
    >>> from tno.quantum.communication.qkd_key_rate.classical.privacy_amplification import (
    ...     PrivacyAmplification,
    ... )
    >>>
    >>> message = Message.random_message(message_length=100)
    >>> privacy = PrivacyAmplification(message.length, error_rate_basis_x=0)
    >>> entropy = privacy.get_entropy_estimate(error_correction_loss=10)
    >>> privacy.do_hash(message, entropy)  # doctest: +SKIP
    '110101100010101010011011111111101010000111101010111001011111010000000110101001000001111110'
"""  # noqa: E501

import hashlib

from tno.quantum.communication.qkd_key_rate._utils import (
    one_minus_binary_entropy as one_minus_h,
)
from tno.quantum.communication.qkd_key_rate.classical._message import Message


class PrivacyAmplification:
    """Privacy amplification.

    A hash of an error corrected string is computed and returned. The length of the
    hashed string equals the remaining entropy.
    """

    def __init__(self, observed_pulses_basis_x: int, error_rate_basis_x: float) -> None:
        """Init of PrivacyAmplification.

        Args:
            observed_pulses_basis_x: Number of pulses received in the X-basis
            error_rate_basis_x: Error rate in the pulses received in the X-basis
        """
        self.observed_pulses_basis_x = observed_pulses_basis_x
        self.error_rate_basis_x = error_rate_basis_x

    def get_entropy_estimate(self, error_correction_loss: int = 0) -> float:
        """Estimate the amount of entropy.

        Uses the key-rate estimation functions to determine the key-rate and
        obtains the number of secure bits by multiplying this by the number of
        pulses sent. Adjusts the remaining entropy for losses due to the
        error-correction.

        Args:
            error_correction_loss: Number of corrected errors

        Returns:
            The amount of entropy in bits.
        """
        entropy = self.observed_pulses_basis_x * float(
            one_minus_h(self.error_rate_basis_x)[0]
        )
        entropy -= error_correction_loss

        return entropy

    def do_hash(self, message: Message, entropy: float) -> str:
        """Computes the hash of a given bit message, returned as a bit string.

        The length of the hash equals the number of bits of entropy.

        Args:
            message: The message to hash.
            entropy: The amount of entropy in bits, used to determines size of the hash.

        Returns:
            The hashed digest of the given message (as a string of bits).
        """
        if entropy < 0:
            error_msg = "Entropy is smaller than 0. Secure hash cannot be computed"
            raise ValueError(error_msg)

        hash_function = hashlib.shake_256()
        hash_function.update(bytes(message))

        entropy_bits = int(entropy)
        entropy_bytes = (entropy_bits // 8) + 1

        hash_bytes = hash_function.digest(entropy_bytes)

        return "".join(f"{byte:08b}" for byte in hash_bytes)[:entropy_bits]
