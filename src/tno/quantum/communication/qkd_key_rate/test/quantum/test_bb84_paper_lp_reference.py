"""Independent executable LP implementation of Attema et al. Eqs. (54)-(56).

Unlike production solve_finite_lp, every inequality is created directly
from the paper with a separate variable-loop representation. The same
physical model is used to create feasible synthetic observation data.
"""
import math

import numpy as np
import pytest
from scipy.optimize import linprog

from tno.quantum.communication.qkd_key_rate.quantum.bb84 import solve_finite_lp


def _poisson(mu, l):
    return math.exp(-mu) * mu**l / math.factorial(l)


def _chernoff(N, p, epsilon):
    A = -math.log(epsilon)
    return A * (1 + math.sqrt(1 + 2*N*p/A))


def _expected_observed(N, mus, weights, yield_per_photon):
    # Physical decoy-state mixture, accounting for 0..20 photons.
    # With mus <= 0.65, the omitted >20 tail is negligible at this N.
    return np.array([
        N * w * sum(
            _poisson(mu, l) * yield_per_photon(l)
            for l in range(21)
        )
        for mu, w in zip(mus, weights)
    ])


def _reference_lp(
    mus, weights, M, sent, observed, eps,
    vacuum_coefficient, single_coefficient
):
    J = len(mus)
    L = M + 1
    dim = L + J
    count = float(sum(observed))
    pmf = [[_poisson(mu, l) for l in range(L)] for mu in mus]
    p_l = [sum(weights[j] * pmf[j][l] for j in range(J)) for l in range(L)]
    p_j_l = [
        [weights[j] * pmf[j][l] / p_l[l] for l in range(L)]
        for j in range(J)
    ]

    q_tail = sum(
        weights[j] * max(0., 1. - sum(pmf[j]))
        for j in range(J)
    )
    lam = q_tail * sent + _chernoff(sent, q_tail, eps)

    A, b = [], []
    for j in range(J):
        # Eq.54: observed_j + delta_j >= truncated model_j.
        upper = [0.] * dim
        for l in range(L):
            upper[l] = p_j_l[j][l]
        upper[L+j] = -1.
        A.append(upper)
        b.append(observed[j] / sent)

        # Eq.54: observed_j + delta_j <= model_j + Lambda.
        lower = [-t for t in upper]
        A.append(lower)
        b.append((-observed[j] + lam) / sent)

    bounds = []
    for l in range(L):
        photon_cap = min(
            p_l[l] * sent + _chernoff(sent, p_l[l], eps),
            count,
        )
        bounds.append((0., photon_cap / sent))
    for j in range(J):
        hoch = math.sqrt(-math.log(eps / 2) * count / 2)
        bounds.append((-hoch / sent, hoch / sent))

    c = [vacuum_coefficient, single_coefficient] + [0.] * (dim - 2)
    eq = [0.] * dim
    for j in range(J):
        eq[L+j] = 1.
    result = linprog(
        c, A_ub=A, b_ub=b, A_eq=[eq], b_eq=[0.],
        bounds=bounds, method="highs"
    )
    return result


@pytest.mark.parametrize("loss_factor", [.015, .08])
@pytest.mark.parametrize(
    ("objective_vacuum", "objective_single", "error_lp"),
    [(0., 1., False), (0., -1., True), (1., .25, False)]
)
def test_three_production_lp_objectives_equal_paper_reference(
    loss_factor, objective_vacuum, objective_single, error_lp
):
    N = 1e8
    M = 6
    eps = 1e-5
    mus = [0.10, 0.38, 0.65]
    weights = [.45, .35, .20]

    def detection_yield(l):
        return min(.02 + loss_factor * l, .95)

    def error_yield(l):
        return detection_yield(l) * (.02 + .003 * min(l, 5))

    observed = _expected_observed(
        N, mus, weights,
        error_yield if error_lp else detection_yield,
    )
    ref = _reference_lp(
        mus, weights, M, N, observed, eps,
        objective_vacuum, objective_single,
    )
    actual = solve_finite_lp(
        target_vacuum=objective_vacuum,
        target_single=objective_single,
        probabilities_intensity_j=np.asarray(weights),
        mu=np.asarray(mus),
        max_num_photons=M,
        number_of_pulses=N,
        observed_count=observed,
        epsilon_mu_j=np.full(len(mus), eps),
        epsilon_num_photons_M=np.full(M+1, eps),
        epsilon_num_photons_M_in_basis_B=np.full(len(mus), eps),
    )
    assert ref.success, str(ref.message)
    assert actual.success, str(actual.message)
    assert actual.fun == pytest.approx(ref.fun, abs=2e-8, rel=2e-5)
    assert actual.x[1] == pytest.approx(ref.x[1], abs=2e-8, rel=2e-5)
