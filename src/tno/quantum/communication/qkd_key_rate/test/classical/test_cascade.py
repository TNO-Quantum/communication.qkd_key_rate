"""Test the Cascade error correction protocol."""

from __future__ import annotations

import pytest

from tno.quantum.communication.qkd_key_rate.classical import (
    Message,
    ParityStrategy,
    Permutations,
)
from tno.quantum.communication.qkd_key_rate.classical.cascade import (
    CascadeCorrector,
    CascadeReceiver,
    CascadeSender,
)
from tno.quantum.utils.validation import check_random_state

random_state = check_random_state(None, "random_state")


@pytest.mark.parametrize("switch_after_pass", [None, 3])
def test_correctness(switch_after_pass: int | None) -> None:
    """
    Test correctness for two different ParityStrategies
        - Apply only normal passes
        - Switching after 3 passes
    """

    message_length = 100000
    error_rate = 0.05
    input_message = Message(
        [int(random_state.rand() > 1 / 2) for _ in range(message_length)]
    )
    error_message = input_message.apply_errors(error_rate=error_rate)

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
