"""Base class for Message object."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass
class Message:
    "Message object"
    message: Union[List[int], List[str], str]

    @property
    def length(self) -> int:
        """Length of message"""
        return len(self.message)

    def __getitem__(self, key: int) -> int:
        return self.message[key]

    def __eq__(self, other: Message) -> bool:
        return self.message == other.message

    def __setitem__(self, key: int, value: int) -> None:
        self.message[key] = value

    def __bytes__(self) -> bytes:
        return bytes("".join(str(x) for x in self.message).encode("utf-8"))

    def __str__(self) -> str:
        res = "".join(str(i) for i in self.message)[:50]
        if self.length > 50:
            res += "..."
        return res

    def pop(self, index: int = -1) -> int:
        """Remove bit at a specific index from message"""
        return self.message.pop(index)

    def apply_permutation(self, permutation: List[int]):
        """Apply a permutation to the message

        Args:
            permutation: The permutation that is applied
        """
        assert self.length == len(permutation)
        self.message = [self.message[i] for i in permutation]

    @classmethod
    def random_message(cls, message_length: int) -> Message:
        """Generate a random message

        Args:
            message_length: Length of random message

        Returns:
            random message
        """
        return cls(list(np.random.randint(2, size=message_length)))
