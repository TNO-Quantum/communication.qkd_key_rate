"""Test the Schedule object."""
import pytest

from tno.quantum.communication.qkd_key_rate.base.schedule import Schedule


@pytest.mark.parametrize(
    "error_rate,expected_schedule",
    [
        (0.009, Schedule([1, 0, 1, 1, 1, 0, 0, 1])),
        (0.019, Schedule([1, 1, 2, 0, 0, 0, 0, 1])),
        (0.029, Schedule([1, 3, 0, 0, 0, 1, 0, 1])),
        (0.039, Schedule([2, 1, 1, 1, 0, 0, 0, 1])),
        (0.049, Schedule([2, 1, 1, 0, 0, 1, 1, 1])),
        (0.059, Schedule([2, 1, 2, 1, 0, 0, 0, 1])),
        (0.069, Schedule([2, 1, 2, 2, 1, 0, 0, 0])),
        (0.079, Schedule([2, 2, 1, 2, 0, 0, 0, 1])),
        (0.089, Schedule([2, 2, 1, 2, 1, 0, 0, 1])),
        (0.099, Schedule([3, 1, 2, 1, 0, 0, 0, 1])),
        (0.109, Schedule([3, 2, 1, 1, 0, 0, 0, 1])),
        (0.119, Schedule([3, 2, 3, 0, 0, 0, 0, 1])),
        (0.129, Schedule([4, 2, 2, 1, 0, 0, 0, 1])),
        (0.139, Schedule([4, 2, 2, 0, 0, 0, 0, 2])),
        (0.149, Schedule([4, 3, 2, 0, 0, 0, 1, 0])),
    ],
)
def test_init_schedule_from_error_rate(
    error_rate: float, expected_schedule: Schedule
) -> None:
    """Test schedule from error rate"""
    schedule = Schedule.schedule_from_error_rate(error_rate=error_rate)
    assert schedule == expected_schedule


def test_stop_high_error() -> None:
    """Test set block schedule from to high error rate"""
    estimated_error_rate = 0.16
    with pytest.raises(ValueError) as _:
        Schedule.schedule_from_error_rate(error_rate=estimated_error_rate)


@pytest.mark.parametrize("error_rate", [0.009, 0.019, 0.069, 0.139, 0.149])
def test_next_pass(error_rate: float) -> None:
    """Test next pass functionality"""
    schedule = Schedule.schedule_from_error_rate(error_rate=error_rate)

    assert schedule.remaining_passes == len(schedule)
    res = []
    for pass_number in range(len(schedule)):
        res.append(schedule.next_pass())
        assert schedule.remaining_passes == len(schedule) - pass_number - 1

    for i in range(8):
        assert res.count(i) == schedule.schedule[i]

    with pytest.raises(IndexError) as _:
        schedule.next_pass()
