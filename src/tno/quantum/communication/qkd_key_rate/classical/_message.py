"""Base class for Message object."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from numpy.random import RandomState

from tno.quantum.utils.validation import check_binary, check_random_state, check_real


@dataclass(init=False)
class Message:
    """Message object containing binary bits."""

    message: list[int]
    """Message object."""

    def __init__(self, message: Iterable[int | str]) -> None:
        """Init :py:class:`Message`.

        Args:
            message: The message, iterable object with binary items.

        Raises:
            TypeError: If `message` contains items that can't be converted to a binary.
            ValueError: If `message` contains items that can't be converted to a binary.
        """
        self.message = [
            check_binary(value, f"message[{i}]") for i, value in enumerate(message)
        ]

    @property
    def length(self) -> int:
        """Length of message."""
        return len(self.message)

    def __getitem__(self, key: int) -> int:
        """Return value of message for specific index.

        Args:
            key: The index at which the value should be returned.
        """
        return self.message[key]

    def __setitem__(self, key: int, value: int) -> None:
        """Set key of message to specific value.

        Args:
            key: The index at which the value should be set.
            value: The value to be inserted at the specified index.
        """
        self.message[key] = value

    def __bytes__(self) -> bytes:
        """Bytes representation of message."""
        return bytes("".join(str(x) for x in self.message).encode("utf-8"))

    def __str__(self) -> str:
        """String representation of message."""
        res = "".join(str(i) for i in self.message)[:50]
        if self.length > 50:
            res += "..."
        return res

    def pop(self, index: int = -1) -> int:
        """Remove bit at a specific index from message."""
        return self.message.pop(index)

    def apply_permutation(self, permutation: list[int]) -> None:
        """Apply a permutation to the message.

        Args:
            permutation: The permutation that is applied

        Raises:
            ValueError: If message is incompatible with permutation.
        """
        if self.length != len(permutation):
            error_msg = "Message is incompatible with permutation."
            raise ValueError(error_msg)
        self.message = [self.message[i] for i in permutation]

    @classmethod
    def random_message(
        cls, message_length: int, random_state: int | RandomState | None = None
    ) -> Message:
        """Generate a random message.

        Args:
            message_length: Length of random message
            random_state: Random state for reproducibility. Defaults to ``None``.

        Returns:
            random message
        """
        random_state = check_random_state(random_state, "random_state")
        return cls(list(random_state.randint(2, size=message_length)))

    def __iter__(self) -> Iterator[int]:
        """Create iterator for message bits."""
        return iter(self.message)

    def apply_errors(
        self, error_rate: float, random_state: int | RandomState | None = None
    ) -> Message:
        """Apply errors to message.

        Args:
            error_rate: probability that an error occurs.
            random_state: Random state for reproducibility. Defaults to ``None``

        Returns:
            Message to which errors are applied.

        Raises: ValueError if error rate not provided as percentage.
        """
        error_rate = check_real(error_rate, "error_rate", l_bound=0, u_bound=1)
        random_state = check_random_state(random_state, "random_state")

        error_message = [x if random_state.rand() > error_rate else 1 - x for x in self]
        return Message(error_message)
