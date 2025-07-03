"""
Test the privacy amplification protocol.
"""

import hashlib

import pytest

from tno.quantum.communication.qkd_key_rate._utils import (
    one_minus_binary_entropy as one_minus_h,
)
from tno.quantum.communication.qkd_key_rate.classical import Message
from tno.quantum.communication.qkd_key_rate.classical.privacy_amplification import (
    PrivacyAmplification,
)


@pytest.mark.parametrize(
    ("error_rate_basis_x", "error_correction_loss"),
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
    assert privacy.get_entropy_estimate(error_correction_loss) == float(
        observed_pulses_basis_x * one_minus_h(error_rate_basis_x)[0]
        - error_correction_loss
    )


@pytest.mark.parametrize(
    ("message_length", "entropy_bits"),
    [(10, 3), (10, 8), (10, 10), (17, 11), (17, 16), (32, 12), (63, 8), (63, 51)],
)
def test_correct_hash(message_length: int, entropy_bits: int) -> None:
    """Test do_hash function"""

    message = Message.random_message(message_length=message_length)
    hash_function = hashlib.shake_256()
    hash_function.update(bytes(message))
    expected_hash_raw = hash_function.digest(entropy_bits // 8 + 1)
    # Convert hash bytes to bits
    expected_hash = "".join(f"{byte:08b}" for byte in expected_hash_raw)[:entropy_bits]
    assert len(expected_hash) == entropy_bits
    assert all(bit in ("0", "1") for bit in expected_hash)

    privacy = PrivacyAmplification(message_length, 0)
    assert privacy.do_hash(message, entropy_bits) == expected_hash


def test_zero_entropy() -> None:
    """Test zero entropy"""
    message_length = 100
    message = Message.random_message(message_length=message_length)
    privacy = PrivacyAmplification(message_length, 0)

    assert privacy.do_hash(message, entropy=0) == ""


def test_negative_entropy() -> None:
    """Test negative entropy"""
    message_length = 100
    message = Message.random_message(message_length=message_length)
    privacy = PrivacyAmplification(message_length, 0)

    expected_message = "Entropy is smaller than 0. Secure hash cannot be computed"
    with pytest.raises(ValueError, match=expected_message):
        privacy.do_hash(message, entropy=-1)
