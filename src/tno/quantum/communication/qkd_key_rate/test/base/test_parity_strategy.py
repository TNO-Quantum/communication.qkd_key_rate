"""Test the ParityStrategy base object."""

import numpy as np
import pytest

from tno.quantum.communication.qkd_key_rate.classical import ParityStrategy


@pytest.mark.parametrize(
    ("error_rate", "expected_block_size"),
    [
        (0.009, 24),
        (0.019, 18),
        (0.029, 12),
        (0.039, 11),
        (0.049, 10),
        (0.059, 9),
        (0.069, 7),
        (0.079, 7),
        (0.089, 7),
        (0.099, 5),
        (0.109, 5),
        (0.119, 5),
        (0.129, 4),
        (0.139, 4),
        (0.149, 4),
    ],
)
def test_get_start_block_size(error_rate: float, expected_block_size: int) -> None:
    """
    Test get_start_block_size function
    """
    parity_strategy = ParityStrategy(error_rate=error_rate)
    start_block_size = parity_strategy.get_start_block_size()
    assert start_block_size == expected_block_size


def test_stop_high_error() -> None:
    """
    Test to high error rate
    """
    error_rate = 0.16
    parity_strategy = ParityStrategy(error_rate=error_rate)

    expected_msg = "Error rate too high for secure protocol"
    with pytest.raises(ValueError, match=expected_msg):
        parity_strategy.get_start_block_size()


def test_calculate_message_parity_strategy() -> None:
    """Test calculate message parity strategy"""
    error_rate = 0.149
    number_of_passes = 10
    switch_after_pass = 5
    sampling_fraction = 0.34

    parity_strategy = ParityStrategy(
        error_rate=error_rate,
        number_of_passes=number_of_passes,
        sampling_fraction=sampling_fraction,
        switch_after_pass=switch_after_pass,
    )

    message_size = 1024
    size_blocks_parities = parity_strategy.calculate_message_parity_strategy(
        message_size
    )

    # Normal passes
    expected_block_size, expected_number_of_blocks = 4, message_size / 4
    for block_size, number_of_blocks in size_blocks_parities[0:switch_after_pass]:
        assert block_size == expected_block_size
        assert number_of_blocks == expected_number_of_blocks

        expected_block_size *= 2
        expected_number_of_blocks /= 2

    # Random permutations
    expected_block_size = np.ceil(message_size * sampling_fraction)
    expected_number_of_blocks = np.ceil(message_size / expected_block_size)
    for block_size, number_of_blocks in size_blocks_parities[switch_after_pass:]:
        assert block_size == expected_block_size
        assert number_of_blocks == expected_number_of_blocks
