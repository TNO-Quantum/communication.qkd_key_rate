"""Base class for ParityStrategy object."""

from typing import Optional, Tuple

import numpy as np

ERROR_RATES_TO_BLOCK_SIZES = {
    0.010: 24,
    0.020: 18,
    0.030: 12,
    0.040: 11,
    0.050: 10,
    0.060: 9,
    0.070: 7,
    0.080: 7,
    0.090: 7,
    0.100: 5,
    0.110: 5,
    0.120: 5,
    0.130: 4,
    0.140: 4,
    0.150: 4,
}


class ParityStrategy:
    """Parity strategy to be used in the Cascade protocol.

    In the Cascade protocol, for efficiency reasons, we may change the strategy
    of our blocks after some passes. This class deals with the parity strategy
    """

    def __init__(
        self,
        error_rate: float,
        sampling_fraction: float = 0.5,
        number_of_passes: int = 10,
        switch_after_pass: Optional[int] = None,
    ) -> None:
        """
        Args:
            error_rate: An estimate of the error rate in the message
            sampling_fraction: Fraction of the string that is sampled after
               switching strategies
            number_of_passes: Total number of Cascade iterations
            switch_after_pass: After which pass we switch the strategy
        """
        self.error_rate = error_rate
        self.number_of_passes = number_of_passes
        self.switch_after_pass = switch_after_pass
        self.sampling_fraction = sampling_fraction

    def get_start_block_size(self) -> int:
        """Determine starting block size.

        Returns:
            The largest block size that is compatible with given error-rate
        """
        possible_block_sizes = [
            block_size
            for (error_rate, block_size) in ERROR_RATES_TO_BLOCK_SIZES.items()
            if error_rate > self.error_rate
        ]

        if possible_block_sizes == []:
            raise ValueError("Error rate too high for secure protocol")
        return max(possible_block_sizes)

    def calculate_message_parity_strategy(
        self, message_size: int
    ) -> Tuple[Tuple[int, int], ...]:
        """Sets the parity strategy

        Initial strategy is to double the block size in each subsequent pass.
        After some number of passes, we may change the strategy by randomly
        sampling bits from the message in new blocks.

        Args:
            message_size: Size of the message for which to calculate the parity strategy

        Returns:
            size_blocks_parities: The block size and number of blocks for each pass
        """
        if self.switch_after_pass is None:
            self.switch_after_pass = self.number_of_passes
        start_block_size = self.get_start_block_size()

        # First passes using classical cascade
        block_sizes = (
            start_block_size << index_pass
            for index_pass in range(self.switch_after_pass)
        )
        size_blocks_parities = [
            (
                block_size,
                int(np.ceil(message_size / block_size)),
            )
            for block_size in block_sizes
        ]

        # Remaining passes with random permutations
        block_sizes = min(
            message_size,
            int(np.ceil(message_size * self.sampling_fraction)),
        )
        size_blocks_parities.extend(
            [
                (
                    block_sizes,
                    int(np.ceil(message_size / block_sizes)),
                )
            ]
            * (self.number_of_passes - self.switch_after_pass)
        )
        return tuple(size_blocks_parities)
