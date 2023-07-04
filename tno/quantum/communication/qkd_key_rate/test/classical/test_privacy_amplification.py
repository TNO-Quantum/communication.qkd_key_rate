"""
Test the privacy amplification protocol.
"""
import hashlib

import pytest

from tno.quantum.communication.qkd_key_rate.base import Message
from tno.quantum.communication.qkd_key_rate.protocols.classical.privacy_amplification import (
    PrivacyAmplification,
)
from tno.quantum.communication.qkd_key_rate.utils import (
    one_minus_binary_entropy as one_minus_h,
)


@pytest.mark.parametrize(
    "error_rate_basis_x,error_correction_loss",
    [
        (0, 0),
        (0, 10),
        (0.5, 0),
        (0.5, 10),
    ],
)
def test_entropy_estimate(
    error_rate_basis_x: float, error_correction_loss: int
) -> None:
    """Test entropy estimate function"""
    observed_pulses_basis_x = 100
    privacy = PrivacyAmplification(observed_pulses_basis_x, error_rate_basis_x)
    assert (
        privacy.get_entropy_estimate(error_correction_loss)
        == observed_pulses_basis_x * one_minus_h(error_rate_basis_x)
        - error_correction_loss
    )


def test_correct_hash() -> None:
    """Test do_hash function"""
    message_length, entropy = 100, 80

    message = Message.random_message(message_length=message_length)
    hash_function = hashlib.shake_256()
    hash_function.update(bytes(message))
    expected_hash = hash_function.digest(entropy)

    privacy = PrivacyAmplification(message_length, 0)
    assert privacy.do_hash(message, entropy) == expected_hash


def test_zero_entropy() -> None:
    """Test zero entropy"""
    message_length = 100
    message = Message.random_message(message_length=message_length)
    privacy = PrivacyAmplification(message_length, 0)

    assert privacy.do_hash(message, entropy=0) == bytes(0)


def test_negative_entropy() -> None:
    """Test negative entropy"""
    message_length = 100
    message = Message.random_message(message_length=message_length)
    privacy = PrivacyAmplification(message_length, 0)

    expected_message = "Entropy is smaller than 0. Secure hash cannot be computed"
    with pytest.raises(ValueError, match=expected_message):
        privacy.do_hash(message, entropy=-1)
