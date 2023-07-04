"""
Test the Cascade error correction protocol.
"""
import numpy as np
import pytest

from tno.quantum.communication.qkd_key_rate.base import (
    Message,
    ParityStrategy,
    Permutations,
)
from tno.quantum.communication.qkd_key_rate.protocols.classical.cascade import (
    CascadeCorrector,
    CascadeReceiver,
    CascadeSender,
)


@pytest.mark.parametrize("switch_after_pass", [None, 3])
def test_correctness(switch_after_pass) -> None:
    """
    Test correctness for two different ParityStrategies
        - Apply only normal passes
        - Switching after 3 passes
    """

    message_length = 100000
    error_rate = 0.05
    input_message = Message(
        [int(np.random.rand() > 0.5) for _ in range(message_length)]
    )
    error_message = Message(
        [x if np.random.rand() > error_rate else 1 - x for x in input_message]
    )

    number_of_passes = 8
    sampling_fraction = 0.34
    permutations = Permutations.random_permutation(
        number_of_passes=number_of_passes, message_size=message_length
    )
    parity_strategy = ParityStrategy(
        error_rate=error_rate,
        sampling_fraction=sampling_fraction,
        number_of_passes=number_of_passes,
        switch_after_pass=switch_after_pass,
    )

    alice = CascadeSender(message=input_message, permutations=permutations)
    bob = CascadeReceiver(
        message=error_message,
        permutations=permutations,
        parity_strategy=parity_strategy,
    )

    corrector = CascadeCorrector(alice=alice, bob=bob)
    results = corrector.correct_errors()

    assert results.output_alice == results.output_bob
    assert results.output_alice.length == message_length
    assert alice.number_of_exposed_bits == bob.number_of_exposed_bits
