"""Test the Message object."""

from copy import deepcopy

import numpy as np

from tno.quantum.communication.qkd_key_rate.classical import Message


def test_random_message_init() -> None:
    """Test creating random permutation"""
    message_size = 1000
    message = Message.random_message(message_length=message_size)

    assert isinstance(message, Message)
    assert message.length == message_size


def test_getter() -> None:
    """Test getter"""
    original_message = [1, 1, 0, 0, 1, 0, 1, 0, 1]
    message = Message(message=original_message)

    for i in range(message.length):
        assert original_message[i] == message[i]


def test_bytes() -> None:
    """Test bytes"""
    message = Message(message=[1, 1, 0, 1])
    assert bytes(message) == b"1101"


def test_permutation() -> None:
    """Test apply permutation"""
    raw_message = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    permutation = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]

    message = Message(message=raw_message)
    message.apply_permutation(permutation)

    assert message.message == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    message1 = Message.random_message(message_length=10)
    message2 = deepcopy(message1)
    inv_permutation: list[int] = np.argsort(permutation).tolist()
    message2.apply_permutation(permutation)
    message2.apply_permutation(inv_permutation)
    assert message1 == message2


def test_apply_error() -> None:
    """Test apply error."""
    message_length = 100_000
    message = Message.random_message(message_length=message_length)
    assert message.apply_errors(error_rate=1.0) == Message([not x for x in message])
    assert message.apply_errors(error_rate=0.0) == message

    error_rate = 0.5
    error_message = message.apply_errors(error_rate)
    number_of_errors = sum(x != y for (x, y) in zip(message, error_message))
    expected_number_of_errors = message_length * error_rate
    assert number_of_errors <= expected_number_of_errors + 0.1 * message_length
    assert expected_number_of_errors - 0.1 * message_length <= number_of_errors
