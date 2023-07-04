"""Base class for Receiving party."""
from __future__ import annotations

import abc
from typing import Optional

from tno.quantum.communication.qkd_key_rate.base import (
    Message,
    Permutations,
    SenderBase,
)


class ReceiverBase(SenderBase):
    """This class encodes all functions only available to the receiver.

    The receiver is assumed to have a string with errors and is thus assumed to
    correct the errors.
    """

    def __init__(
        self, message: Message, permutations: Permutations, name: Optional[str] = None
    ):
        """
        Args:
            message: Input message of the sender party
            permutations: List containing permutations for each pass
            name: Name of the sender party
        """
        super().__init__(message=message, permutations=permutations, name=name)

    @abc.abstractmethod
    def correct_errors(self, alice: SenderBase) -> None:
        """The main routine, find and correct errors."""

    def correct_individual_error(self, error_index: int) -> None:
        """Corrects a single error by flipping bit at location 'error_index'."""
        self._message[error_index] = 1 - self._message[error_index]
