"""Utility functions used for entropy calculations.

For computational stability, we define two functions and use one of them based on the
applicable setting.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def binary_entropy(e: ArrayLike) -> NDArray[np.float64]:
    r"""Binary entropy function.

    $h(e):\mathbb{R} \rightarrow [0,1]$ is the binary entropy function.

    The domain of the function has been extended such that the optimizer can
    do a proper constrained non-linear optimization.

    Args:
        e: probabilities

    Returns:
        Shannon entropy
    """
    e_arr = np.atleast_1d(e)

    valid_ind = (e_arr > 0) & (e_arr < 1)
    boundary_ind = (e_arr == 0) | (e_arr == 1)
    valid_e = e_arr[valid_ind]

    # Domain extension to help optimizer
    h_e = np.zeros_like(e_arr)
    h_e[valid_ind] = np.abs(
        -valid_e * np.log2(np.abs(valid_e))
        - (1.0 - valid_e) * np.log2(np.abs(1.0 - valid_e))
    )
    h_e[boundary_ind] = 0
    return h_e


def one_minus_binary_entropy(e: ArrayLike) -> NDArray[np.float64]:
    r"""One minus the binary entropy function.

    $h(e):\mathbb{R} \rightarrow [0,1]$ is the binary entropy function.

    The domain of the function has been extended such that the optimizer can
    do a proper constrained non-linear optimization.

    This computation is more precise if $h(e)$ is close to zero or one.
    """
    e_arr = np.atleast_1d(e)

    valid_ind = (e_arr > 0) & (e_arr < 1)
    boundary_ind = (e_arr == 0) | (e_arr == 1)
    valid_e = e_arr[valid_ind]

    # Domain extension to help optimizer
    h_e = np.zeros_like(e_arr)
    h_e[valid_ind] = np.abs(
        valid_e * np.log2(2 * valid_e) + (1 - valid_e) * np.log2(2 - 2 * valid_e)
    )
    h_e[boundary_ind] = 1
    return h_e
