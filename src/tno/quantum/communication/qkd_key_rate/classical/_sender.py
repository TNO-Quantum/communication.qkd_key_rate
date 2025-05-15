"""Base class for Sending party."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tno.quantum.communication.qkd_key_rate.classical import Message, Permutations


class SenderBase:
    """This class encodes all functions available to both sender and receiver."""

    transcript: str
    """This keeps track of the exchanged messages."""

    def __init__(
        self,
        message: Message,
        permutations: Permutations,
        name: str | None = None,
    ) -> None:
        """Init of base sender class.

        Args:
            message: Input message of the sender party
            permutations: Permutations for each pass
            name: Name of the sender party
        """
        self._message = message
        self._original_message = deepcopy(message)

        self.permutations = permutations
        self._number_of_exposed_bits = 0
        self._net_exposed_bits = 0

        self.name = name
        self.transcript = ""

    @property
    def original_message(self) -> Message:
        """Returns original uncorrected message."""
        return self._original_message

    @property
    def message(self) -> Message:
        """Returns the (partially corrected) message."""
        return self._message

    @property
    def number_of_exposed_bits(self) -> int:
        """Counter to track number of exposed bits during protocol."""
        return self._number_of_exposed_bits

    @property
    def net_exposed_bits(self) -> int:
        """Counter to track the `net` number of exposed bits.

        During the Winnow protocol some bits are discarded. The net number of exposed
        bits is the number of exposed bits, minus number of discarded bits.

        For the Cascade protocol the net number of exposed bits is the same as the
        number of exposed bits.
        """
        return self._number_of_exposed_bits
