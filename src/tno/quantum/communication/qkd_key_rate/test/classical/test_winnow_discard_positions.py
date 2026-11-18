"""Winnow must discard parity/syndrome pivots in the intended blocks."""

import pytest

from tno.quantum.communication.qkd_key_rate.classical import (
    Message,
    Permutations,
    Schedule,
)
from tno.quantum.communication.qkd_key_rate.classical.winnow import (
    WinnowCorrector,
    WinnowReceiver,
    WinnowSender,
)


def _sender(length: int) -> WinnowSender:
    perms = Permutations([list(range(length)), list(range(length))])
    return WinnowSender(Message([0] * length), perms, Schedule([1]))


@pytest.mark.parametrize("length", [8, 16, 32, 64, 128, 512, 1024])
def test_drop_one_parity_bit_from_every_complete_block(length: int) -> None:
    sender = _sender(length)
    assert sender.message.length == length - length // 8
    assert sender.net_exposed_bits == 0


def test_discard_syndrome_pivots_from_flagged_block() -> None:
    sender = _sender(32)
    assert sender.block_size == 7
    sender.bad_blocks_array[0] = 2
    sender.number_of_bad_blocks = 1
    tracked = list(range(sender.message.length))
    removed = []

    def pop(index: int) -> int:
        removed.append(tracked.pop(index))
        return 0

    sender.message.pop = pop  # type: ignore[method-assign]
    sender.discard_syndrome_bits()
    assert sorted(removed) == [14, 15, 17]
    assert sender.net_exposed_bits == -3


def test_discard_pivots_from_two_distinct_bad_blocks() -> None:
    sender = _sender(32)
    sender.bad_blocks_array[:2] = [1, 3]
    sender.number_of_bad_blocks = 2
    tracked = list(range(sender.message.length))
    removed = []

    def pop(index: int) -> int:
        removed.append(tracked.pop(index))
        return 0

    sender.message.pop = pop  # type: ignore[method-assign]
    sender.discard_syndrome_bits()
    assert sorted(removed) == [7, 8, 10, 21, 22, 24]
    assert sender.net_exposed_bits == -6


def test_full_reconciliation_remains_consistent_with_fixed_deletions() -> None:
    length = 512
    original = Message.random_message(length, random_state=43)
    noisy = Message(original.message.copy())
    noisy.message[300] ^= 1
    perms = Permutations([list(range(length)), list(range(length))])
    alice = WinnowSender(original, perms, Schedule([1]))
    bob = WinnowReceiver(noisy, perms, Schedule([1]))
    output = WinnowCorrector(alice, bob).correct_errors()
    assert output.output_error == 0
    assert output.output_alice == output.output_bob
    assert output.output_length < length
    assert output.number_of_exposed_bits >= 0
