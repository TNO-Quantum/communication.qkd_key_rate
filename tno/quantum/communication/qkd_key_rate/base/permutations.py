"""Base class for Permutations object."""
from __future__ import annotations

from typing import List

import numpy as np


class Permutations:
    """Permutations object"""

    def __init__(self, permutations: List[List[int]]):
        self.permutations = permutations
        self.inverted_permutations = self.calculate_inverted_permutations(permutations)
        self.number_of_passes = len(permutations)

    def __add__(self, other: Permutations) -> Permutations:
        """Combine two permutations"""
        return Permutations(self.permutations + other.permutations)

    def __len__(self) -> int:
        "Return number of passes"
        return self.number_of_passes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutations):
            return NotImplemented
        return self.permutations == other.permutations

    def __getitem__(self, pass_number: int) -> List[int]:
        return self.permutations[pass_number]

    def shorten_pass(self, pass_idx: int, max_length: int) -> None:
        """In some protocols, the message length decreases, this function
        adjusts the permutations accordingly, by discarding large indices.

        Args:
            pass_idx: Index of the permutation to shorten
            max_length: New message length, and maximum of the entries in
            the permutation (value itself not included)
        """
        self.permutations[pass_idx] = [
            i for i in self.permutations[pass_idx] if i < max_length
        ]
        self.inverted_permutations[pass_idx] = list(
            np.argsort(self.permutations[pass_idx])
        )

    @staticmethod
    def calculate_inverted_permutations(
        permutations: List[List[int]],
    ) -> List[List[int]]:
        """Invert every permutation in a list of permutations"""
        return [list(np.argsort(permutation)) for permutation in permutations]

    @classmethod
    def random_permutation(
        cls, number_of_passes: int, message_size: int
    ) -> Permutations:
        """Generate a random Permutations object

        Args:
            number_of_passes: Total number of iterations
            message_size: Size of the message for which to calculate the parity strategy
        """
        permutations = [
            list(np.random.permutation(message_size)) for _ in range(number_of_passes)
        ]
        return cls(permutations)
