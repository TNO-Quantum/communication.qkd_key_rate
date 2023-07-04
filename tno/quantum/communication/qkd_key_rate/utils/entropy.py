"""
Functions used for entropy calculations. For computational stability, we define
two functions and use one based on the applicable setting.
"""
import numpy as np

# pylint: disable=invalid-name


def binary_entropy(e: float) -> float:
    r"""Binary entropy function.

    $h(e):\mathbb{R} \rightarrow [0,1]$ is the binary entropy function.

    The domain of the function has been extended such that the optimizer can
    do a proper constrained non-linear optimization.
    """
    e = np.atleast_1d(e)

    valid_ind = (e > 0) & (e < 1)
    boundary_ind = (e == 0) | (e == 1)
    valid_e = e[valid_ind]

    h_e = np.zeros_like(e)
    #   The log(1-p) can be rewritten as log1p(-p). The log1p is more accurate for
    #    computing logarithms of numbers that are close to 1

    # Domain extension to help optimizer
    h_e[valid_ind] = np.abs(
        -valid_e * np.log2(np.abs(valid_e))
        - (1.0 - valid_e) * np.log2(np.abs(1.0 - valid_e))
    )
    h_e[boundary_ind] = 0

    if h_e.shape == (1,):
        h_e = h_e[0]

    return h_e


def one_minus_binary_entropy(e: float) -> float:
    r"""One minus the binary entropy function.

    $h(e):\mathbb{R} \rightarrow [0,1]$ is the binary entropy function.

    The domain of the function has been extended such that the optimizer can
    do a proper constrained non-linear optimization.

    This computation is more precise if $h(e)$ is close to zero or one.
    """
    e = np.atleast_1d(e)

    valid_ind = (e > 0) & (e < 1)
    boundary_ind = (e == 0) | (e == 1)
    valid_e = e[valid_ind]

    h_e = np.zeros_like(e)
    #   The log(1-p) can be rewritten as log1p(-p). The log1p is more accurate for
    #    computing logarithms of numbers that are close to 1

    # Domain extension to help optimizer
    h_e[valid_ind] = np.abs(
        valid_e * np.log2(2 * valid_e) + (1 - valid_e) * np.log2(2 - 2 * valid_e)
    )
    h_e[boundary_ind] = 1

    if h_e.shape == (1,):
        h_e = h_e[0]

    return h_e
