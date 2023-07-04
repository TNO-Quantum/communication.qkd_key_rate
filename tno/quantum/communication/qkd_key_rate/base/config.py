"""Base config settings"""

from scipy import optimize

# Options to be used for solving the linear programs
_LP_SOLVER_OPTIONS = dict(maxiter=1000, disp=False, presolve=True)
LP_CONFIG = dict(method="highs", options=_LP_SOLVER_OPTIONS)
_NLP_OPTIONS = dict(maxiter=1000, disp=False, finite_diff_rel_step=1e-6)
NLP_CONFIG = dict(
    method="trust-constr", jac="2-point", hess=optimize.BFGS(), options=_NLP_OPTIONS
)
