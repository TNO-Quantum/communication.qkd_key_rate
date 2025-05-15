"""Classes to perform a Winnow error correction protocol.

The Winnow error correction protocol is based on Hamming codes. An advantage of the protocol
is that it requires less communication than other error correction protocols. The protocol
however might introduce errors in specific cases. With every communication, the Winnow protocol
leaks information to potential eavesdroppers. This can be overcome by discarding message bits
equal to the amount of information leaks, thereby achieving privacy maintenance.

Typical usage example:

    >>> from tno.quantum.communication.qkd_key_rate.classical import (
    ...     Message,
    ...     Schedule,
    ...     Permutations,
    ... )
    >>> from tno.quantum.communication.qkd_key_rate.classical.winnow import (
    ...     WinnowCorrector,
    ...     WinnowReceiver,
    ...     WinnowSender,
    ... )
    >>>
    >>> message_length = 100_000
    >>> error_rate = 0.05
    >>> input_message = Message.random_message(message_length=message_length, random_state=43)
    >>> error_message = input_message.apply_errors(error_rate=0.05, random_state=42)
    >>>
    >>> schedule = Schedule.schedule_from_error_rate(error_rate=error_rate)
    >>> number_of_passes = sum(schedule.schedule)
    >>> permutations = Permutations.random_permutation(
    ...     number_of_passes=number_of_passes, message_size=message_length, random_state=42
    ... )
    >>>
    >>> alice = WinnowSender(
    ...     message=input_message, permutations=permutations, schedule=schedule
    ... )
    >>> bob = WinnowReceiver(
    ...     message=error_message, permutations=permutations, schedule=schedule
    ... )
    >>> corrector = WinnowCorrector(alice=alice, bob=bob)
    >>>
    >>> print(corrector.correct_errors())  # doctest: +SKIP
    Corrector summary:
    input_alice (Message):   00111001111000101101111101011000100010000100000011...
    output_alice (Message):  01111111100100100011010100110110101100010110101111...
    input_bob (Message):     00111001110000101101111101011100100010000110000011...
    output_bob (Message):    01111111100100100011010100110110101100010110101111...
    input_error (float):     0.04959
    output_error (float):    0.0
    output_length (int):     52063
    number_of_exposed_bits (int):    3647
    key_reconciliation_rate (float):         0.48416
    number_of_communication_rounds (int):    15
    schedule (list[int]):    [2, 1, 1, 0, 0, 1, 1, 1]
"""  # noqa: E501

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tno.quantum.communication.qkd_key_rate.classical import (
    Corrector,
    CorrectorOutputBase,
    Permutations,
    ReceiverBase,
    Schedule,
    SenderBase,
)

if TYPE_CHECKING:
    from tno.quantum.communication.qkd_key_rate.classical import Message


class WinnowSender(SenderBase):
    """This class encodes all functions available to both sender and receiver.

    It keeps track of the number of exposed bits and can compute syndromes and
    parities. Furthermore, it keeps track of the blocks with errors
    """

    def __init__(
        self,
        message: Message,
        permutations: Permutations,
        schedule: Schedule,
        name: str | None = None,
    ) -> None:
        """Init of a WinnowSender.

        Args:
            message: Input message of the sender party
            permutations: List containing permutations for each pass
            schedule: The winnow schedule
            name: Name of the sender party
        """
        super().__init__(message=message, permutations=permutations, name=name)
        self.schedule = schedule

        self.number_of_bad_blocks = 0
        self.block_size = 8
        self.number_of_blocks = 0
        self.maximum_number_of_communication_rounds = 0
        self._number_of_exposed_bits = 0
        self.number_of_passes = permutations.number_of_passes
        self._net_exposed_bits = 0
        self.syndrome_length = 3
        self.removed_bits: list[int] = []

        self.syndrome_array: list[int] = [0 for _ in range(self.message.length // 8)]

        # Stores the indices of blocks which contain errors
        self.bad_blocks_array: list[int] = [0 for _ in range(self.message.length // 8)]
        self.parity_string: list[int] = []

        self.parity_check_matrix = np.zeros([10, 1023], dtype=int)

        self.transcript = ""

        self.first_pass()

    def create_parity_check_matrix(self) -> None:
        """Creates a parity check matrix.

        This matrix is used to encode the bit strings.
        """
        size = 1 << self.syndrome_length
        for i in range(self.syndrome_length):
            for j in range(1, size):
                self.parity_check_matrix[i, j - 1] = int(j / (1 << i)) & 0x1

    def get_parity(self, index_start: int, index_end: int) -> int:
        """Get the parity of a specific message part between two indices.

        Args:
            index_start: Start index of the message
            index_end: End index of the message

        Returns:
            Parity of substring
        """
        number_of_ones = 0
        for i in range(index_start, index_end):
            if self.message[i] == 1:
                number_of_ones += 1
        return number_of_ones % 2

    def build_parity_string(self) -> None:
        """Builds a parity string for all blocks."""
        for i in range(self.number_of_blocks):
            index_start = i * self.block_size
            index_end = index_start + self.block_size

            parity = self.get_parity(index_start, index_end)
            self.parity_string[i] = parity

            self._number_of_exposed_bits += 1
            self._net_exposed_bits += 1

    def discard_parity_bits(self) -> None:
        """The first bit of every parity block is discarded."""
        old_index = 0
        counter = -1
        while old_index < self.message.length:
            if ((old_index % self.block_size) == 0) and (
                old_index != (self.number_of_blocks * self.block_size)
            ):
                # Is it the first bit of a block and is it not the last block
                counter += 1
                self._net_exposed_bits -= 1
                self.message.pop(old_index - counter)
            old_index += 1

        self.block_size -= 1
        self.transcript += (
            f"\tBoth discard the parity bits for pass {self.schedule.pass_number}.\n"
        )

    def get_syndrome(self, index_block: int) -> int:
        """Computes the syndrome of a block.

        Both parties compute their syndrome individually, hence, no
        communication is needed here.
        """
        if index_block > self.number_of_blocks:
            self.transcript += (
                "Illegal block number. Returning block_size + 1 for new syndrome.\n"
            )
            return self.block_size + 1

        placeholder = 0
        new_syndrome = 0
        # Computer he highest order bit of the syndrome first and then work down
        for i in range(self.syndrome_length - 1, -1, -1):
            new_syndrome <<= 1
            # Multiply the block by the (i-1)-th row of the parity check matrix
            # and add it to the syndrome
            for j in range(self.block_size):
                placeholder += (
                    self.parity_check_matrix[i, j]
                    * self.message[index_block * self.block_size + j]
                )
                placeholder &= 0x1
            new_syndrome += placeholder
            placeholder = 0

        self._number_of_exposed_bits += self.syndrome_length
        self._net_exposed_bits += self.syndrome_length
        return new_syndrome

    def disagreeing_block_parities(self, alice: WinnowSender) -> None:
        """Finds the disagreeing block parities.

        The found parities of both parties are compared. This can be done with
        two communication rounds (one both ways).
        Afterwards, both separately process the results.

        Args:
            alice: The sending party
        """
        self.maximum_number_of_communication_rounds += 1
        self.transcript += (
            f"Bob sends his parity string to alice for pass {self.schedule.pass_number}"
            ". Alice compares the string with hers and keeps track of the disagreeing"
            " blocks.\n"
        )

        counter_for_bad_blocks = 0
        for i in range(self.number_of_blocks):
            if alice.parity_string[i] != self.parity_string[i]:
                # The parities disagree, save the block-index as bad block.
                self.bad_blocks_array[counter_for_bad_blocks] = i
                alice.bad_blocks_array[counter_for_bad_blocks] = i
                counter_for_bad_blocks += 1
        self.number_of_bad_blocks = counter_for_bad_blocks
        alice.number_of_bad_blocks = counter_for_bad_blocks

    def discard_syndrome_bits(self) -> None:
        """Discards syndrome bits.

        Bits at indices $2^j-1$ are removed. These correspond to the linearly
        independent columns of the parity check matrix.

        In this function the number of bad blocks is known.

        No communication is needed to discard syndrome bits.
        """
        counter_for_error_blocks = 0
        removed_bits = []
        for index_block in range(self.number_of_blocks):
            # If we have not hit all bad blocks, and the counter is at a bad block
            if (
                counter_for_error_blocks < self.number_of_bad_blocks
                and self.bad_blocks_array[counter_for_error_blocks] == index_block
            ):
                power = 0
                counter_for_error_blocks += 1
                offset_counter = -1
                for index_bit in range(self.block_size):
                    if (index_bit + 1) == (1 << power):
                        # Discard bits if they are at a location with index a
                        # power of 2 - 1
                        power += 1
                        offset_counter += 1
                        self._net_exposed_bits -= 1
                        removed_bits.append(index_bit - offset_counter)
                        self.message.pop(index_bit - offset_counter)

        self.transcript += (
            f"\tBoth discard the syndrome bits for pass {self.schedule.pass_number}.\n"
        )

    def build_syndrome_string(self, alice: WinnowSender) -> None:
        """Create a syndrome string for all disagreeing blocks.

        Computes the syndrome for blocks with disagreeing parity.

        Args:
            alice: The sending party
        """
        self.disagreeing_block_parities(alice)

        for i in range(self.number_of_bad_blocks):
            alice.syndrome_array[i] = self.get_syndrome(alice.bad_blocks_array[i])

    def first_pass(self) -> None:
        """First pass with initializations and parity determination."""
        # Note that the incomplete last block is not included in each pass
        self.number_of_blocks = self.message.length // self.block_size
        self.parity_string = [0] * self.number_of_blocks

        self.create_parity_check_matrix()
        self.permute_buffer()
        self.build_parity_string()
        self.discard_parity_bits()

    def next_pass(self) -> None:
        """Performs computations to prepare for the next pass.

        Computations include permuting the message and creating a new parity string.
        """
        i = self.schedule.next_pass()
        self.syndrome_length = i + 3
        self.block_size = 1 << self.syndrome_length

        # Note that the incomplete last block is not included in each pass
        self.number_of_blocks = self.message.length // self.block_size
        self.create_parity_check_matrix()
        self.permute_buffer()
        self.build_parity_string()
        self.discard_parity_bits()

    def permute_buffer(self) -> None:
        """Permutes the message string."""
        self.permutations.shorten_pass(self.schedule.pass_number, self.message.length)
        self.message.apply_permutation(self.permutations[self.schedule.pass_number])


class WinnowReceiver(WinnowSender, ReceiverBase):
    """This class encodes all functions only available to the receiver.

    The receiver is assumed to have a string with errors and is thus assumed to
    correct the errors.
    """

    def __init__(
        self,
        message: Message,
        permutations: Permutations,
        schedule: Schedule,
        name: str | None = None,
    ) -> None:
        """Init of WinnowReceiver.

        Args:
            message: Input message of the sender party
            permutations: List containing permutations for each pass
            schedule: The winnow schedule.
            name: Name of the receiver party
        """
        super().__init__(
            message, name=name, permutations=permutations, schedule=schedule
        )

    def fix_errors_with_syndrome(self, alice: WinnowSender) -> None:
        """Corrects errors using the syndrome strings of alice and bob.

        Args:
            alice: The sending party
        """
        syndrome_alice = alice.syndrome_array

        for i in range(alice.number_of_bad_blocks):
            syndrome_bob = alice.get_syndrome(alice.bad_blocks_array[i])
            disagreeing_syndrome_bit = (
                syndrome_alice[i] ^ syndrome_bob
            )  # XOR the two syndromes

            if disagreeing_syndrome_bit == 0:
                # Erroneous bit was already discarded
                continue
            self.correct_individual_error(
                alice.bad_blocks_array[i] * self.block_size
                + disagreeing_syndrome_bit
                - 1
            )
        self.transcript += "bob computes the syndromes for his bit string and \
accordingly corrects bits in his own bit string based on the difference with \
the syndromes of alice.\n"

        self.discard_syndrome_bits()
        alice.discard_syndrome_bits()

    def correct_errors(self, alice: SenderBase) -> None:
        """The main routine, finds all errors and corrects them.

        It is assumed that Alice and Bob use one communication round to agree on
        the used permutations. Afterwards, they use two communication rounds per
        iteration to communicate the syndromes.

        Args:
            alice: The sending party

        Raises:
            TypeError: If alice is not a WinnowSender.
        """
        if not isinstance(alice, WinnowSender):
            error_msg = "alice is not of type `WinnowSender`."
            raise TypeError(error_msg)
        alice.schedule = deepcopy(self.schedule)

        self.maximum_number_of_communication_rounds += 1
        if (len(self.schedule) - len(self.permutations)) >= 0:
            # Add permutations, if there are not enough for the whole schedule
            self.permutations += Permutations.random_permutation(
                number_of_passes=len(self.schedule) - len(self.permutations) + 1,
                message_size=self.message.length,
            )
            self.transcript += (
                "Permutations for bit string shared between alice and bob.\n"
            )
            alice.permutations = deepcopy(self.permutations)

        number_of_remaining_passes = 1

        while number_of_remaining_passes > 0:
            self.maximum_number_of_communication_rounds += 1
            self.transcript += (
                "alice sends the syndromes for the disagreeing blocks of pass"
                f"{self.schedule.pass_number} to bob, as well as the block indices."
                "If there are no disagreeing blocks, both enter the next pass.\n"
            )
            self.build_syndrome_string(alice)

            # If we have blocks with disagreeing syndromes, correct bits accordingly
            if self.number_of_bad_blocks != 0:
                self.fix_errors_with_syndrome(alice)
            else:
                self.transcript += (
                    f"\tNo disagreeing parities in pass {self.schedule.pass_number}"
                    ", continue.\n"
                )

            self.next_pass()
            alice.next_pass()

            number_of_remaining_passes = self.schedule.remaining_passes


@dataclass
class WinnowCorrectorOutput(CorrectorOutputBase):
    """Data class for Winnow Corrector output."""

    schedule: Schedule


class WinnowCorrector(Corrector):
    """Winnow corrector."""

    def __init__(self, alice: WinnowSender, bob: WinnowReceiver) -> None:
        """Init of WinnowCorrector.

        Args:
            alice: Sender party.
            bob: Receiver party.

        Raises:
            ValueError: If permutations alice are not same as permutations bob.
        """
        self.alice: WinnowSender = alice
        self.bob: WinnowReceiver = bob

        if self.alice.permutations != self.bob.permutations:
            error_msg = "Permutations Alice not same as permutations Bob."
            raise ValueError(error_msg)

    def summary(self) -> WinnowCorrectorOutput:
        """Calculate a summary object for the error correction.

        Summary contains
            - original message
            - corrected message
            - error rate (before and after correction)
            - number_of_exposed_bits
            - key_reconciliation_rate
            - protocol specific parameters
        """
        return WinnowCorrectorOutput(
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
            schedule=self.bob.schedule,
        )
