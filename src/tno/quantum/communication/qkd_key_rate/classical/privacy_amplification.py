r"""Privacy amplification module.

The :py:class:`~tno.quantum.communication.qkd_key_rate.classical.privacy_amplification.PrivacyAmplification`
class can be used to compute the hash of an error corrected string. The length of the
hashed string equals the remaining entropy.

Typical usage example:

    >>> from tno.quantum.communication.qkd_key_rate.classical.privacy_amplification import (
    ...    PrivacyAmplification,
    ... )
    >>>
    >>> message = Message.random_message(message_length=100)
    >>> privacy = PrivacyAmplification(message.length, error_rate_basis_x=0)
    >>> entropy = privacy.get_entropy_estimate(error_correction_loss=10)
    >>> privacy.do_hash(message, entropy)  # doctest: +SKIP
    b'\x11\x8f\xba\xc8\xc2\x97\xce\xd7\xf0\xf4F\xd1\xfew\xf7\xd0\xd2Y>\x08J\x1e\x1a\xb9\x1d.\x8d8\xc5\x01\xa7(u\xc0\xcd\x1bc\xfck\xbc_!.\xea\xf5v\xd9\x90\xd4a\x89,\xdaZ\xf6tq\xdf\xe77\x16\x07\x15\xdc\x8d\x86`\x80Wy\x7fHU\xc8\xe3\xe7\xcf \xe6V\x8f\x19\x1c!sv\xad\x9b\xc4\x1c'

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
            The amount of entropy
        """
        entropy = self.observed_pulses_basis_x * float(
            one_minus_h(self.error_rate_basis_x)
        )
        entropy -= error_correction_loss

        return entropy

    def do_hash(self, message: Message, entropy: float) -> bytes:
        """Computes the hash given a message of bits.

        The length of the hash equals the secure entropy we have.
        """
        if entropy < 0:
            error_msg = "Entropy is smaller than 0. Secure hash cannot be computed"
            raise ValueError(error_msg)

        hash_function = hashlib.shake_256()
        hash_function.update(bytes(message))

        return hash_function.digest(int(entropy))
