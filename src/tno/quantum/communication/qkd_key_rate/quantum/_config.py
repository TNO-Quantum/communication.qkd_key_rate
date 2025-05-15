"""Base config settings."""

from scipy import optimize

# Options to be used for solving the linear programs
LP_CONFIG = {
    "method": "highs",
    "options": {"maxiter": 1000, "disp": False, "presolve": True},
}

NLP_CONFIG = {
    "method": "trust-constr",
    "jac": "2-point",
    "hess": optimize.BFGS(),
    "options": {"maxiter": 1000, "disp": False, "finite_diff_rel_step": 1e-6},
}


class OptimizationError(ValueError):
    """Raised when optimization is unsuccessful.

    This error typically thrown when the lp problem is infeasible.
    """
