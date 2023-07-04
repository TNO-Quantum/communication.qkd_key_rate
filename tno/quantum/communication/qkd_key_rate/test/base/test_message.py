"""Test the Message object."""

from copy import deepcopy

import numpy as np

from tno.quantum.communication.qkd_key_rate.base import Message


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
    assert b"1101" == bytes(message)


def test_permutation() -> None:
    """Test apply permutation"""
    message = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    permutation = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]

    message = Message(message=message)
    message.apply_permutation(permutation)

    assert message.message == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    message1 = Message.random_message(message_length=10)
    message2 = deepcopy(message1)
    permutation = list(np.random.permutation(10))
    inv_permutation = list(np.argsort(permutation))
    message2.apply_permutation(permutation)
    message2.apply_permutation(inv_permutation)
    assert message1 == message2
