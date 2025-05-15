"""Classes to perform a Cascade error correction protocol.

The Cascade error correction protocol can be used to correct errors in sifted bit strings. Errors
are detected by calculating the parity between bitstrings of the two parties. The protocol first
divides the messages in different blocks sizes and repeats this a number of passes times where the
block size is doubled each pass. An error is detected and corrected when the parity for a block is
odd. In case the parity for a block is even, it is still possible that (an even number of) errors
exist. These errors can be detected in the next pass, when the block size is doubled and the message
is shuffled. Given enough passes, all errors are expected to be corrected. However, the required
communication of this protocol is high.


The choice of block size, depending on error-rate is based on the following thesis:
Calver, Timothy I., "An Empirical Analysis of the Cascade Secret Key Reconciliation Protocol for Quantum
Key Distribution" (2011). Theses and Dissertations. 1521. (https://scholar.afit.edu/etd/1521/)


Typical usage example:

    >>> from tno.quantum.communication.qkd_key_rate.classical import (
    ...     Message,
    ...     ParityStrategy,
    ...     Permutations,
    ... )
    >>> from tno.quantum.communication.qkd_key_rate.classical.cascade import (
    ...     CascadeCorrector,
    ...     CascadeReceiver,
    ...     CascadeSender,
    ... )
    >>>
    >>> message_length = 100_000
    >>> error_rate = 0.05
    >>> input_message = Message.random_message(message_length=message_length)
    >>> error_message = input_message.apply_errors(error_rate=0.05)
    >>>
    >>> number_of_passes = 8
    >>> sampling_fraction = 0.34
    >>> parity_strategy = ParityStrategy(
    ...     error_rate=error_rate,
    ...     sampling_fraction=sampling_fraction,
    ...     number_of_passes=number_of_passes,
    ... )
    >>>
    >>> permutations = Permutations.random_permutation(
    ...     number_of_passes=number_of_passes, message_size=message_length, random_state=42
    ... )
    >>> alice = CascadeSender(message=input_message, permutations=permutations)
    >>> bob = CascadeReceiver(
    ...     message=error_message,
    ...     permutations=permutations,
    ...     parity_strategy=parity_strategy,
    ... )
    >>> corrector = CascadeCorrector(alice=alice, bob=bob)
    >>>
    >>> print(corrector.correct_errors())  # doctest: +SKIP
    Corrector summary:
    input_alice (Message):   00010010000111101110111110001000001110111101000110...
    output_alice (Message):  00010010000111101110111110001000001110111101000110...
    input_bob (Message):     10010010001111111110111110001000000110111101000100...
    output_bob (Message):    00010010000111101110111110001000001110111101000110...
    input_error (float):     0.05065
    output_error (float):    0.0
    output_length (int):     100000
    number_of_exposed_bits (int):    62301
    key_reconciliation_rate (float):         0.37699
    number_of_communication_rounds (int):    147.0
    number_of_passes (int):  8
    switch_after_pass (int):         8
    sampling_fraction (float):       0.34
"""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tno.quantum.communication.qkd_key_rate.classical import (
    Corrector,
    CorrectorOutputBase,
    ParityStrategy,
    Permutations,
    ReceiverBase,
    SenderBase,
)

if TYPE_CHECKING:
    from tno.quantum.communication.qkd_key_rate.classical import Message


class CascadeSender(SenderBase):
    """This class encodes all functions available to Cascade sender."""

    def __init__(
        self,
        message: Message,
        permutations: Permutations,
        name: str | None = None,
    ) -> None:
        """Init of CascadeSender.

        Args:
            message: Input message of the sender party
            permutations: List containing permutations for each Cascade pass
            name: Name of the sender party
        """
        super().__init__(message=message, permutations=permutations, name=name)

        self._number_of_exposed_bits = 0
        self._net_exposed_bits = 0
        self.max_exposed_bits = 0
        self.min_exposed_bits = 0

        # Offset for permutation exchange
        self.maximum_number_of_communication_rounds = 1

        self.number_of_passes = permutations.number_of_passes

        self.transcript += "Permutations for bit string shared between alice and bob\n"

    def get_parity(self, index_start: int, index_end: int, pass_number: int) -> int:
        """Get the parity of a specific message block.

        Computation takes into account the permutation applied in that specific pass.

        Args:
            index_start: Start index of the message
            index_end: End index of the message
            pass_number: Cascade pass number

        Returns:
            Parity of substring
        """
        number_of_ones = 0
        for i in range(index_start, index_end):
            if self.message[self.permutations[pass_number][i]] == 1:
                number_of_ones += 1
        self._number_of_exposed_bits += 1
        self._net_exposed_bits += 1

        return number_of_ones % 2


class CascadeReceiver(CascadeSender, ReceiverBase):
    """This class encodes all functions only available to the receiver.

    The receiver is assumed to have a string with errors and is thus assumed to
    correct the errors.
    """

    def __init__(
        self,
        message: Message,
        permutations: Permutations,
        parity_strategy: ParityStrategy,
        name: str | None = None,
    ) -> None:
        """Init of CascadeReceiver.

        Args:
            message: Input message of the sender party
            permutations: List containing permutations for each Cascade pass
            parity_strategy: The cascade parity strategy.
            name: Name of the receiver party

        Raises:
            ValueError: If parity strategy incompatible with number of passes.
        """
        super().__init__(
            message,
            name=name,
            permutations=permutations,
        )
        self.parity_strategy = parity_strategy
        if self.parity_strategy.number_of_passes != self.number_of_passes:
            error_msg = "Parity strategy not compatible with number of passes."
            raise ValueError(error_msg)

        self.block_sizes: list[int] = [0 for _ in range(self.number_of_passes)]
        self.errors_found: list[list[int]] = [[] for _ in range(self.number_of_passes)]

    def correct_errors(self, alice: SenderBase) -> None:
        """This is the main routine.

        Errors in the strings alice and bob are found and corrected

        An upper bound on the number of communications is estimated. This is
        however a loose upper bound. The cascade-routine makes it difficult to
        give tighter bounds

        Args:
            alice: The sending party to correct errors with

        Raises:
            TypeError: If alice is not a cascade sender.
        """
        if not isinstance(alice, CascadeSender):
            error_msg = "Alice is not a cascade sender."
            raise TypeError(error_msg)

        size_blocks_parities = self.parity_strategy.calculate_message_parity_strategy(
            self.message.length
        )

        for index_pass in range(self.number_of_passes):
            # Get the block size and the number of blocks per pass
            (
                block_size,
                number_of_blocks,
            ) = size_blocks_parities[index_pass]
            self.transcript += (
                f"alice sends her parity string for pass {index_pass} to bob.\n"
            )

            self.block_sizes[index_pass] = block_size
            # One round is for sharing the bits from sender to receiver. The other round
            # is to indicate which blocks should be checked from receiver to sender.
            # This number is an upper bound, and in practice might be much lower.
            self.maximum_number_of_communication_rounds += 2 * (
                np.sum(np.ceil(np.log2(self.block_sizes[: index_pass + 1])))
            )

            self.transcript += (
                "bob determines which block parities disagree and corrects them.\n"
            )
            self.transcript += "Error found in (pass, block):\n"

            for index_block in range(number_of_blocks):
                # Check if the parity strings match and otherwise Cascade back
                # to locate the error is this and previous passes.
                self.do_cascade(
                    index_pass, index_block, block_size, alice, size_blocks_parities
                )
            if len(self.errors_found[index_pass]) == 0:
                # In this case we do not have to do all subsequent communication rounds
                self.maximum_number_of_communication_rounds -= 2 * (
                    np.sum(np.ceil(np.log2(self.block_sizes[:index_pass])))
                )

        self.transcript += (
            "Due to the implementation, it is hard to return"
            " the combined messages shared between the parties."
        )

    def check_match_of_parities(
        self,
        alice: CascadeSender,
        current_pass: int,
        current_block: int,
        block_size: int,
    ) -> bool:
        """Checks if the parity strings of alice and bob match.

        This requires communication between the two parties. The parities of
        multiple blocks can be combined in a single message.

        Args:
            alice: The sending party
            current_pass: Index of current pass
            current_block: Index of current block
            block_size: size of the block.

        Returns:
            Boolean if the parity strings of alice and bob match
        """
        index_start = current_block * block_size
        index_finish = min(index_start + block_size, self.message.length)
        parity_alice = alice.get_parity(
            index_start, index_finish, pass_number=current_pass
        )
        parity_bob = self.get_parity(
            index_start, index_finish, pass_number=current_pass
        )
        return bool(parity_alice == parity_bob)

    def do_cascade(
        self,
        current_pass: int,
        current_block: int,
        block_size: int,
        alice: CascadeSender,
        size_blocks_parities: tuple[tuple[int, int], ...],
    ) -> None:
        """Apply the Cascade error correction technique.

        This routine corrects errors in previous passes that become apparent later.
        It is recursively used as long as new errors are found.

        Args:
            current_pass: Index of current pass
            current_block: Index of current block
            block_size: Size of current block
            alice: The sending party
            size_blocks_parities: For each pass, size of the block and number of blocks
        """
        if self.check_match_of_parities(alice, current_pass, current_block, block_size):
            return

        # An error is found, because the parities do not match
        self.transcript += f"({current_pass}, {current_block}), "

        currently_found_error = self.get_error_index(
            current_block, current_pass, block_size, alice
        )
        self.correct_individual_error(currently_found_error)
        self.errors_found[current_pass].append(currently_found_error)

        # Check all preceding iterations if there now is an error to be corrected
        for index_pass in range(current_pass):
            (block_size, _) = size_blocks_parities[index_pass]
            index_block = self.get_block_index(
                currently_found_error, block_size, index_pass
            )
            self.max_exposed_bits += 1
            self.min_exposed_bits += 1
            self.do_cascade(
                index_pass, index_block, block_size, alice, size_blocks_parities
            )

    def get_block_index(self, index: int, block_size: int, index_pass: int) -> int:
        """Returns the block index corresponding to a certain index in a certain pass.

        Args:
            index: Index of a certain bit
            block_size: Size of current block
            index_pass: Index of the current pass

        Returns:
            The block index corresponding to a certain bit index in a certain pass
        """
        original_index = self.permutations.inverted_permutations[index_pass][index]
        return original_index // block_size

    def get_error_index(
        self, index_block: int, index_pass: int, block_size: int, alice: CascadeSender
    ) -> int:
        """Recursively checks the parity of half of the block of both parties.

        Args:
            index_block: Index of the block in which we expect an error
            index_pass: Index of the current pass
            block_size: Size of current block
            alice: The sending party

        Returns:
            The position index of an error
        """
        index_start = index_block * block_size

        # Determine index_finish, either index_start + block_size or message_size
        index_finish = min(self.message.length, index_start + block_size)

        # If the length of the considered part is not a power of 2, the number
        # of exposed bits can vary by one.
        self.max_exposed_bits += np.ceil(np.log2(index_finish - index_start))
        self.min_exposed_bits += np.floor(np.log2(index_finish - index_start))

        while True:
            if (index_finish - index_start) == 1:
                # If we have only a single bit, return that index
                return self.permutations[index_pass][index_start]
            if (index_finish - index_start) == 2:
                # If we have two bits, we check if the first bits agree
                # and return the index of the error accordingly

                parity_alice = alice.get_parity(
                    index_start, index_finish - 1, pass_number=index_pass
                )
                parity_bob = self.get_parity(
                    index_start, index_finish - 1, pass_number=index_pass
                )
                if parity_alice == parity_bob:
                    return self.permutations[index_pass][index_finish - 1]
                return self.permutations[index_pass][index_start]

            # Otherwise, the string contains at least 3 bits. Compute the parity
            # over half of the string
            parity_alice = alice.get_parity(
                index_start,
                index_start + (index_finish - index_start) // 2,
                pass_number=index_pass,
            )
            parity_bob = self.get_parity(
                index_start,
                index_start + (index_finish - index_start) // 2,
                pass_number=index_pass,
            )
            # If the parities match, the error was in the other half of the message
            if parity_alice == parity_bob:
                index_start = index_start + int(
                    np.floor((index_finish - index_start) / 2)
                )
            else:
                index_finish = index_start + int(
                    np.floor((index_finish - index_start) / 2)
                )

    def get_error_rate(self) -> float:
        """Gives the error rate, based on the found errors."""
        index_errors_found = [
            index_error
            for errors_in_pass in self.errors_found
            for index_error in errors_in_pass
        ]

        number_of_errors = len(index_errors_found)
        return number_of_errors / self.message.length

    def get_prior_error_rate(
        self,
        alice: CascadeSender,
        index_start: int = 0,
        index_finish: int | None = None,
    ) -> float:
        """Determine the initial error rate.

        This function is mainly for debugging purposes. Usually,
        the considered bits are private and this value cannot be
        computed.

        Args:
            alice: The sending party
            index_start: Start index of the message
            index_finish: End index of the message

        Returns:
            Initial error rate
        """
        if index_finish is None:
            index_finish = self.message.length
        number_of_differences = 0
        for i in range(index_start, index_finish):
            if alice.message[i] != self.message[i]:
                number_of_differences += 1
        return number_of_differences / self.message.length


@dataclass
class CascadeCorrectorOutput(CorrectorOutputBase):
    """Data class for Cascade Corrector output."""

    number_of_passes: int
    switch_after_pass: int
    sampling_fraction: float


class CascadeCorrector(Corrector):
    """Cascade corrector."""

    def __init__(self, alice: CascadeSender, bob: CascadeReceiver) -> None:
        """Init Cascade corrector.

        Args:
            alice: cascade sender.
            bob: cascade receiver.
        """
        self.alice: CascadeSender = alice
        self.bob: CascadeReceiver = bob

        if self.alice.permutations != self.bob.permutations:
            error_msg = "Permutations Alice not same as permutations Bob."
            raise ValueError(error_msg)

    def summary(self) -> CascadeCorrectorOutput:
        """Calculate a summary object for the error correction.

        Summary contains:

            - original message
            - corrected message
            - error rate (before and after correction)
            - number_of_exposed_bits
            - key_reconciliation_rate
            - protocol specific parameters
        """
        return CascadeCorrectorOutput(
            input_alice=self.alice.original_message,
            output_alice=self.alice.message,
            input_bob=self.bob.original_message,
            output_bob=self.bob.message,
            input_error=self.calculate_error_rate(
                self.alice.original_message, self.bob.original_message
            ),
            output_error=self.calculate_error_rate(
                self.alice.message, self.bob.message
            ),
            output_length=self.alice.message.length,
            number_of_exposed_bits=self.bob.net_exposed_bits,
            key_reconciliation_rate=self.calculate_key_reconciliation_rate(
                exposed_bits=True
            ),
            number_of_communication_rounds=self.bob.maximum_number_of_communication_rounds,
            number_of_passes=self.alice.permutations.number_of_passes,
            switch_after_pass=self.bob.parity_strategy.number_of_passes,
            sampling_fraction=self.bob.parity_strategy.sampling_fraction,
        )
