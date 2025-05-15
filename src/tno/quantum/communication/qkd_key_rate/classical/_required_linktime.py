"""Util functions for calculating the efficiency and communication rounds."""

from __future__ import annotations

import numpy as np

from tno.quantum.communication.qkd_key_rate.classical._parity_strategy import (
    ERROR_RATES_TO_BLOCK_SIZES,
)
from tno.quantum.communication.qkd_key_rate.classical._schedule import (
    ERROR_RATES_TO_SCHEDULE,
)

EFFICIENCY = {
    "winnow": {
        0.010: 0.90,
        0.020: 0.83,
        0.030: 0.80,
        0.040: 0.71,
        0.050: 0.65,
        0.060: 0.59,
        0.070: 0.54,
        0.080: 0.49,
        0.090: 0.44,
        0.100: 0.40,
        0.110: 0.34,
        0.120: 0.29,
        0.130: 0.27,
        0.140: 0.22,
        0.150: 0.19,
    },
    "cascade": {
        0.010: 0.83,
        0.020: 0.78,
        0.030: 0.73,
        0.040: 0.67,
        0.050: 0.63,
        0.060: 0.58,
        0.070: 0.55,
        0.080: 0.51,
        0.090: 0.47,
        0.100: 0.44,
        0.110: 0.39,
        0.120: 0.35,
        0.130: 0.32,
        0.140: 0.30,
        0.150: 0.27,
    },
}


def compute_efficiency(method: str, error_rate: float) -> float:
    """Calculate the efficiency of the error correction protocol.

    If the error-rate is above maximum defined error rate, set the efficiency to 0.20.

    Args:
        method: the error correction protocol (``'Winnow'`` or ``'Cascade'``)
        error_rate: estimated error rate

    Returns:
        Efficiency of the error correction protocol

    Raises:
        ValueError: If unknown method is provided.
    """
    try:
        efficiency_dict = EFFICIENCY[method.lower()]
    except KeyError as exc:
        error_msg = "Unknown passed corrector-method."
        raise ValueError(error_msg) from exc

    error_keys = efficiency_dict.keys()
    try:
        er = min(x for x in error_keys if x >= error_rate)
    except ValueError:
        return 0.20
    return efficiency_dict.get(er, 0.20)


def compute_estimate_on_communication(
    method: str, error_rate: float, message_length: int | None = None
) -> int:
    """Estimate on the number of communication rounds.

    Args:
        method: the error correction protocol (``'Winnow'`` or ``'Cascade'``)
        error_rate: the error rate
        message_length: Length of the message, only required for cascade protocol

    Returns:
        Estimate for the number of communication rounds. For the Cascade protocol only a
        rough upper bound can be provided.

    Raises:
        ValueError: If error rate is to high for a secure protocol.
        ValueError: If message length is not provided with Cascade protocol.
        ValueError: If provided method is not among valid options.
    """
    if method.lower() == "winnow":
        error_schedule_keys = ERROR_RATES_TO_SCHEDULE.keys()
        try:
            er = min(x for x in error_schedule_keys if x >= error_rate)
        except ValueError as exc:
            error_msg = "Error rate too high for secure protocol"
            raise ValueError(error_msg) from exc

        return 1 + 2 * sum(ERROR_RATES_TO_SCHEDULE[er])

    if method.lower() == "cascade":
        # Only a loose estimate on the number of communications can be given.
        if message_length is None:
            error_msg = "Message length needs to be provided for Cascade protocol"
            raise ValueError(error_msg)

        error_block_keys = ERROR_RATES_TO_BLOCK_SIZES.keys()
        try:
            er = min(x for x in error_block_keys if x >= error_rate)
        except ValueError as exc:
            error_msg = "Error rate too high for secure protocol"
            raise ValueError(error_msg) from exc

        start_block_size = ERROR_RATES_TO_BLOCK_SIZES[er]
        number_of_passes = 8
        block_sizes = [start_block_size * 2**i for i in range(number_of_passes)]
        max_sizes = np.repeat([message_length // 2], number_of_passes)
        block_sizes = np.vstack([block_sizes, max_sizes]).min(axis=0)

        return int(
            1
            + 2
            * np.sum(
                [
                    np.sum(np.ceil(np.log2(block_sizes[: i + 1])))
                    for i in range(number_of_passes)
                ],
                dtype=int,
            )
        )

    error_msg = "Unknown passed corrector-method."
    raise ValueError(error_msg)
