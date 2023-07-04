"""Test the Permutations object."""

from tno.quantum.communication.qkd_key_rate.base import Permutations


def test_random_permutations_init() -> None:
    """Test creating random permutation"""
    number_of_passes = 2
    message_size = 6

    permutations = Permutations.random_permutation(
        number_of_passes=number_of_passes, message_size=message_size
    )

    assert isinstance(permutations, Permutations)
    assert permutations.number_of_passes == number_of_passes
    for permutation in permutations.permutations:
        assert len(permutation) == message_size


def test_inverted_permutations() -> None:
    """Test inverted permutation"""
    number_of_passes = 2
    message_size = 6

    permutations = Permutations.random_permutation(
        number_of_passes=number_of_passes, message_size=message_size
    )

    for permutation, inverted_permutation in zip(
        permutations.permutations, permutations.inverted_permutations
    ):
        assert [permutation[i] for i in inverted_permutation] == list(
            range(message_size)
        )


def test_addition() -> None:
    """Test combining permutations"""
    p1 = Permutations([[1, 2, 3], [4, 5, 6]])
    p2 = Permutations([[7, 8, 9]])
    p3 = Permutations([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert p1 + p2 == p3

    p1 += p2
    assert p1 == p3


def test_length() -> None:
    """Test length function of permutations"""
    number_of_passes = 10
    p = Permutations.random_permutation(
        number_of_passes=number_of_passes, message_size=100
    )
    assert len(p) == 10


def test_shorten_pass() -> None:
    """Test shortening permutations"""
    permutations = Permutations([list(range(10)), list(range(10))])

    permutations.shorten_pass(0, 7)
    permutations.shorten_pass(1, 4)
    assert permutations[0] == list(range(7))
    assert permutations[1] == list(range(4))
    assert permutations.inverted_permutations[0] == list(range(7))
    assert permutations.inverted_permutations[1] == list(range(4))

    permutations = Permutations.random_permutation(number_of_passes=3, message_size=100)
    permutations.shorten_pass(0, 5)
    permutations.shorten_pass(1, 50)
    permutations.shorten_pass(2, 110)

    for permutation, inverted_permutation in zip(
        permutations.permutations, permutations.inverted_permutations
    ):
        assert [permutation[i] for i in inverted_permutation] == list(
            range(len(permutation))
        )
