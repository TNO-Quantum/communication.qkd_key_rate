"""
Test entropy utils
"""
from tno.quantum.communication.qkd_key_rate.utils import binary_entropy
from tno.quantum.communication.qkd_key_rate.utils import (
    one_minus_binary_entropy as one_minus_h,
)


def test_one_minus_binary_entropy_zero_input() -> None:
    """Test one minus h for zero input"""
    assert one_minus_h(0) == 1


def test_one_minus_h() -> None:
    """Test one minus h"""
    assert all(one_minus_h([0.5, 0.5]) == [0.0, 0.0])


def test_binary_entropy_zero_input() -> None:
    """Test binary entropy function for zero input"""
    assert binary_entropy(0) == 0


def test_binary_entropy() -> None:
    """Test binary entropy function on array"""
    assert all(binary_entropy([0.5, 0.5]) == [1.0, 1.0])
