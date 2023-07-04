"""Base class for Sending party."""
from __future__ import annotations

import abc
from copy import deepcopy
from typing import Optional

from tno.quantum.communication.qkd_key_rate.base import Message, Permutations


class SenderBase(metaclass=abc.ABCMeta):
    """This class encodes all functions available to both sender and receiver."""

    def __init__(
        self,
        message: Message,
        permutations: Permutations,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            message: Input message of the sender party
            permutations: List containing permutations for each pass
            name: Name of the sender party
        """
        self._message = message
        self._original_message = deepcopy(message)

        self.permutations = permutations
        self.number_of_exposed_bits = 0
        self.net_exposed_bits = 0

        self.name = name
        self.transcript = ""  # This keeps track of the exchanged messages

    @property
    def original_message(self) -> Message:
        """Returns original uncorrected message."""
        return self._original_message

    @property
    def message(self) -> Message:
        """Returns the (partially corrected) message."""
        return self._message
