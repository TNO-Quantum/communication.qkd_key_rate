"""
Test the Winnow error correction protocol.
"""
import numpy as np

from tno.quantum.communication.qkd_key_rate.base import Message, Permutations, Schedule
from tno.quantum.communication.qkd_key_rate.protocols.classical.winnow import (
    WinnowCorrector,
    WinnowReceiver,
    WinnowSender,
)


def test_correctness() -> None:
    """Test correctness for Winnow protocol"""
    message_length = 60000
    error_rate = 0.01
    input_message = Message.random_message(message_length=message_length)
    error_message = Message(
        [x if np.random.rand() > error_rate else 1 - x for x in input_message]
    )

    schedule = Schedule.schedule_from_error_rate(error_rate=error_rate)
    number_of_passes = np.sum(schedule.schedule)
    permutations = Permutations.random_permutation(
        number_of_passes=number_of_passes, message_size=message_length
    )

    alice = WinnowSender(
        message=input_message, permutations=permutations, schedule=schedule
    )
    bob = WinnowReceiver(
        message=error_message, permutations=permutations, schedule=schedule
    )

    corrector = WinnowCorrector(alice=alice, bob=bob)
    results = corrector.correct_errors()

    assert results.output_alice == results.output_bob
    assert abs(1 - results.input_error / error_rate) < 0.2
    assert results.output_error == 0
    assert (
        abs(
            (results.output_length - results.number_of_exposed_bits) / message_length
            - results.key_reconciliation_rate
        )
        < 0.001
    )
