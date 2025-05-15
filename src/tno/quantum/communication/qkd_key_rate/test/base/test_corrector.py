"""Test Corrector base functions"""

import hashlib
import hmac
from copy import deepcopy

import numpy as np

from tno.quantum.communication.qkd_key_rate.classical import (
    Corrector,
    Message,
    ParityStrategy,
    Permutations,
)
from tno.quantum.communication.qkd_key_rate.classical._schedule import (
    Schedule,
)
from tno.quantum.communication.qkd_key_rate.classical.cascade import (
    CascadeCorrector,
    CascadeReceiver,
    CascadeSender,
)
from tno.quantum.communication.qkd_key_rate.classical.winnow import (
    WinnowCorrector,
    WinnowReceiver,
    WinnowSender,
)
from tno.quantum.utils.validation import check_random_state

random_state = check_random_state(None, "random_state")


def test_calculate_number_of_errors() -> None:
    """Test calculation of number of errors"""
    message1 = Message.random_message(100)
    message2 = deepcopy(message1)
    assert Corrector.calculate_number_of_errors(message1, message2) == 0
    for i in range(message1.length):
        message2[i] = 1 - message1[i]
        assert Corrector.calculate_number_of_errors(message1, message2) == i + 1


def test_calculate_error_rate() -> None:
    """Test calculation of ratio number of errors"""
    message_length = 100
    message1 = Message.random_message(message_length)
    message2 = deepcopy(message1)
    assert Corrector.calculate_error_rate(message1, message2) == 0
    for i in range(message1.length):
        message2[i] = 1 - message1[i]
        assert (
            Corrector.calculate_error_rate(message1, message2)
            == (i + 1) / message_length
        )


def test_calculate_key_reconciliation_rate_cascade() -> None:
    """
    Test key reconciliation rate calculation with Cascade.
    """
    message_length = 1000
    error_rate = 0.01
    input_message = Message(
        [int(random_state.rand() > 1 / 2) for _ in range(message_length)]
    )
    error_message = input_message.apply_errors(error_rate=error_rate)

    number_of_passes = 5
    sampling_fraction = 0.5
    permutations = Permutations.random_permutation(
        number_of_passes=number_of_passes, message_size=message_length
    )
    parity_strategy = ParityStrategy(
        error_rate=error_rate,
        sampling_fraction=sampling_fraction,
        number_of_passes=number_of_passes,
        switch_after_pass=number_of_passes,
    )

    alice = CascadeSender(message=input_message, permutations=permutations)
    bob = CascadeReceiver(
        message=error_message,
        permutations=permutations,
        parity_strategy=parity_strategy,
    )

    corrector = CascadeCorrector(alice=alice, bob=bob)
    results = corrector.correct_errors()

    assert (
        results.key_reconciliation_rate
        == corrector.calculate_key_reconciliation_rate(exposed_bits=True)
    )


def test_calculate_key_reconciliation_rate_winnow() -> None:
    """
    Test key reconciliation rate calculation with Winnow.
    """
    message_length = 1000
    error_rate = 0.01
    input_message = Message.random_message(message_length=message_length)
    error_message = input_message.apply_errors(error_rate=error_rate)

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

    assert (
        results.key_reconciliation_rate
        == corrector.calculate_key_reconciliation_rate(exposed_bits=True)
    )


def test_calculate_key_reconciliation_rate_artificial() -> None:
    """
    Test key reconciliation rate calculation without running a protocol.
    """
    message_length = 1000
    exposed_bits = 150
    error_rate = 0.001
    input_message = Message.random_message(message_length=message_length)
    error_message = deepcopy(input_message)

    number_of_passes = 5
    sampling_fraction = 0.5
    permutations = Permutations.random_permutation(
        number_of_passes=number_of_passes, message_size=message_length
    )
    parity_strategy = ParityStrategy(
        error_rate=error_rate,
        sampling_fraction=sampling_fraction,
        number_of_passes=number_of_passes,
        switch_after_pass=number_of_passes,
    )

    alice = CascadeSender(message=input_message, permutations=permutations)
    bob = CascadeReceiver(
        message=error_message,
        permutations=permutations,
        parity_strategy=parity_strategy,
    )

    corrector = CascadeCorrector(alice=alice, bob=bob)
    corrector.alice._number_of_exposed_bits = exposed_bits
    corrector.alice._net_exposed_bits = exposed_bits

    assert corrector.calculate_key_reconciliation_rate(exposed_bits=True) == (
        1 - exposed_bits / message_length
    )


def test_create_message_tag_pair() -> None:
    """Test message-tag hashed pair"""
    message_length = 1000
    input_message = Message(
        [int(random_state.rand() > 1 / 2) for _ in range(message_length)]
    )

    shared_key = "key"
    message, tag = CascadeCorrector.create_message_tag_pair(
        input_message, shared_key=shared_key
    )

    message_string = "".join(str(x) for x in input_message.message)
    assert message == bytes(message_string.encode("utf-8"))
    assert (
        tag
        == hmac.new(
            key=bytes(shared_key.encode("utf-8")), msg=message, digestmod=hashlib.sha384
        ).digest()
    )
