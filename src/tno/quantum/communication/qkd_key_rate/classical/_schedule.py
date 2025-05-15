"""Base class for Schedule object."""

from __future__ import annotations

ERROR_RATES_TO_SCHEDULE = {
    0.010: [1, 0, 1, 1, 1, 0, 0, 1],
    0.020: [1, 1, 2, 0, 0, 0, 0, 1],
    0.030: [1, 3, 0, 0, 0, 1, 0, 1],
    0.040: [2, 1, 1, 1, 0, 0, 0, 1],
    0.050: [2, 1, 1, 0, 0, 1, 1, 1],
    0.060: [2, 1, 2, 1, 0, 0, 0, 1],
    0.070: [2, 1, 2, 2, 1, 0, 0, 0],
    0.080: [2, 2, 1, 2, 0, 0, 0, 1],
    0.090: [2, 2, 1, 2, 1, 0, 0, 1],
    0.100: [3, 1, 2, 1, 0, 0, 0, 1],
    0.110: [3, 2, 1, 1, 0, 0, 0, 1],
    0.120: [3, 2, 3, 0, 0, 0, 0, 1],
    0.130: [4, 2, 2, 1, 0, 0, 0, 1],
    0.140: [4, 2, 2, 0, 0, 0, 0, 2],
    0.150: [4, 3, 2, 0, 0, 0, 1, 0],
}


class Schedule:
    """Schedule object for Winnow protocol."""

    def __init__(self, schedule: list[int]) -> None:
        """Init of Schedule.

        Args:
            schedule: a Winnow schedule.
        """
        self.schedule = schedule

        self.pass_number = 0
        self.processed_schedule = [0] * len(schedule)

    def __str__(self) -> str:
        """Schedule string representation."""
        return str(self.schedule)

    def __len__(self) -> int:
        """Length schedule."""
        return sum(self.schedule)

    def __eq__(self, other: object) -> bool:
        """Schedule equality."""
        if not isinstance(other, Schedule):
            return NotImplemented
        return self.schedule == other.schedule

    def next_pass(self) -> int:
        """Find iteration index for the next pass and update processed schedule.

        This iteration index describes the Hamming-distance used in that specific
        interaction in the protocol.

        Returns:
            iteration index
        """
        if self.pass_number >= len(self):
            error_msg = "Pass number is larger than length of schedule."
            raise IndexError(error_msg)
        self.pass_number += 1

        for i, schedule_i in enumerate(self.schedule):
            if self.processed_schedule[i] < schedule_i:
                self.processed_schedule[i] += 1
                return i
        return -1

    @property
    def remaining_passes(self) -> int:
        """Number of remaining passes in schedule."""
        return len(self) - sum(self.processed_schedule)

    @classmethod
    def schedule_from_error_rate(cls, error_rate: float = 0.15) -> Schedule:
        """Get schedule based on error rate."""
        error_keys = ERROR_RATES_TO_SCHEDULE.keys()
        try:
            min_error = min(x for x in error_keys if x >= error_rate)
        except ValueError as err:
            error_msg = "Error rate too high for secure protocol"
            raise ValueError(error_msg) from err

        schedule = ERROR_RATES_TO_SCHEDULE.get(min_error, [2, 1, 2, 2, 1, 0, 0, 0])
        return cls(schedule)
