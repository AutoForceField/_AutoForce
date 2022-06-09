# +
from __future__ import annotations

from autoforce.functions.descriptors import Overlaps_des


def test_Overlaps_setup() -> bool:
    Overlaps_des(3, 3)
    return True


def test_Overlaps_perm() -> bool:
    """
    TODO:
    Test if the descriptor is invariant wrt permutations.

    """
    return True


def test_Overlaps_backward() -> bool:
    """
    TODO:
    Test if backward runs without error.

    """
    return True


def test_Overlaps_rotational_invariance() -> bool:
    """
    TODO:
    Test rotational invariance.
    """
    return True


def test_Overlaps_compressed_norm() -> bool:
    """
    TODO:
    Test if the norm is the the same when
        1) compress = True
        2) compress = False

    """
    return True


if __name__ == "__main__":
    test_Overlaps_perm()
    test_Overlaps_backward()
    test_Overlaps_rotational_invariance()
    test_Overlaps_compressed_norm()
