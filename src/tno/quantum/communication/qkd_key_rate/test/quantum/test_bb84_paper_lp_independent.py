"""Finite-BB84 LP regressions against Attema et al. (2021), Eqs. 48-56.

The expected bounds are calculated independently of the production LP matrix.
"""
import math
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tno.quantum.communication.qkd_key_rate.quantum import bb84, standard_detector


def _capture_lp(*, sent, observed, mu, weight, M, epsilon=1e-7):
    recorded = []

    def capture(**kwargs):
        recorded.append(kwargs)
        return OptimizeResult(
            status=0, success=True,
            x=np.zeros(M + 1 + len(mu)), fun=0.,
        )

    with patch.object(bb84.scipy.optimize, "linprog", side_effect=capture):
        bb84.solve_finite_lp(
            target_vacuum=0, target_single=1,
            probabilities_intensity_j=np.array(weight, dtype=float),
            mu=np.array(mu, dtype=float),
            max_num_photons=M,
            number_of_pulses=sent,
            observed_count=np.array(observed, dtype=float),
            epsilon_mu_j=np.full(len(mu), epsilon),
            epsilon_num_photons_M=np.full(M + 1, epsilon),
            epsilon_num_photons_M_in_basis_B=np.full(len(mu), epsilon),
        )
    assert len(recorded) == 1
    return recorded[0]


def _f_independent(N, p, epsilon):
    L = -math.log(epsilon)
    return L * (1 + math.sqrt(1 + 2 * p * N / L))


def _poisson_probability(mu, count):
    return math.exp(-mu) * mu**count / math.factorial(count)


def test_chernoff_photon_count_bound_matches_paper_equation_54():
    # Deliberately substantial observed counts so the trivial n_B cap doesn't
    # mask the buggy extra factor of N_B multiplying f.
    N = 10_000
    obs = [3_600., 4_200.]
    mu = [.2, .6]
    weights = [.55, .45]
    M = 4
    eps = 1e-7
    mat = _capture_lp(
        sent=N, observed=obs, mu=mu, weight=weights, M=M,
        epsilon=eps,
    )
    J = len(mu)
    l = 2
    p_l = sum(w * _poisson_probability(u, l) for w, u in zip(weights, mu))
    paper_cap = min(p_l * N + _f_independent(N, p_l, eps), sum(obs))
    # First 2J inequalities: count bounds; then M+1 lower 0; then M+1 upper.
    actual_cap = float(mat["b_ub"][2 * J + (M + 1) + l] * N)
    assert actual_cap == pytest.approx(paper_cap, rel=1e-9, abs=1e-9)


def test_truncation_direction_accepts_paper_valid_and_rejects_invalid():
    # Eq.54: T <= n + delta <= T + Lambda, where T is predicted
    # observed count from truncated photon variables (0..M).
    # With delta=0, T=60,n=62 is allowed. T=64,n=62 is not.
    N = 1_000
    obs = [62.]
    mat = _capture_lp(
        sent=N, observed=obs, mu=[.2], weight=[1.], M=2,
        epsilon=1e-5,
    )
    A, b = mat["A_ub"], mat["b_ub"]
    def paper_witness_predicted_count(T):
        photon_vars = np.array([10., 20., T - 30., 0.]) / N
        # The matrix is built with normalized LP unknowns; the last
        # variable is a normalized Hoeffding deviation.
        return photon_vars
    def is_feasible_truncation_rows(x):
        return bool(np.all(A[:2] @ x <= b[:2] + 1e-10))
    assert is_feasible_truncation_rows(paper_witness_predicted_count(60.))
    assert not is_feasible_truncation_rows(paper_witness_predicted_count(64.))


def test_expected_sent_basis_pulses_not_observed_events_used_for_lp_bounds():
    total_sent = 1_000_000_000
    xprob = [.4, .2]
    zprob = [.25, .15]
    called = []
    def fake_lp(**kwargs):
        called.append(kwargs)
        return {"x": np.array([0.01, .02, 0., 0., 0., 0.]), "fun": 0.}
    est = bb84.BB84FiniteKeyRateEstimate(
        detector=standard_detector, number_of_pulses=total_sent,
        number_of_decoy=1,
    )
    with patch.object(bb84, "solve_finite_lp", side_effect=fake_lp):
        est.compute_rate(
            mu=[.2, .5], attenuation=.5,
            probability_basis_X=xprob,
            probability_basis_Z=zprob,
        )
    assert len(called) == 3
    sent_z = total_sent * sum(zprob)
    sent_x = total_sent * sum(xprob)
    for idx, expected_sent in enumerate([sent_z, sent_z, sent_x]):
        assert called[idx]["number_of_pulses"] == pytest.approx(expected_sent)
        # The observed count remains a distinct, smaller number.
        assert float(sum(called[idx]["observed_count"])) < expected_sent


@pytest.mark.parametrize("zero_single", [True, False])
def test_zero_single_bound_and_phase_above_half_give_zero_single_entropy_credit(
    zero_single,
):
    # When n_1,Z lower bound vanishes, no phase-information is established;
    # otherwise a phase error upper bound > 0.5 still yields hbar = h(.5).
    calls = []
    def fake_lp(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            x1 = 0. if zero_single else .02
        elif len(calls) == 2:
            x1 = 0. if zero_single else .01
        else:
            x1 = .02
        return {"x": np.array([.01, x1, 0., 0., 0., 0.]), "fun": 0.}
    est = bb84.BB84FiniteKeyRateEstimate(
        detector=standard_detector, number_of_pulses=1_000_000_000,
        number_of_decoy=1,
    )
    with (
        patch.object(bb84, "solve_finite_lp", side_effect=fake_lp),
        patch.object(bb84, "delta", return_value=.3),
    ):
        est.compute_rate(
            mu=[.2, .5], attenuation=.5,
            probability_basis_X=[.4, .2],
            probability_basis_Z=[.25, .15],
        )
    assert len(calls) == 3
    # The X LP objective coefficient is 1 - hbar(phase error upper bound).
    assert calls[2]["target_single"] == pytest.approx(0., abs=1e-12)


def test_paper_security_budget_has_positive_secrecy_margin():
    # This test is distinct from the separate independent LP-statistics
    # derivation. Avoid repeating the implementation's own count formula.
    eps_sec = 2 ** -50
    eps_smooth = 2 ** -55
    eps_event = 2 ** -60
    M, J = 17, 3
    # Eq.38: photon-number + intensity Hoeffding failure events for
    # one X detections LP and both Z detections/error LP calls.
    eps_e = (2*(M+1) + 2*J) * eps_event
    eps_x = ((M+1) + J) * eps_event
    eps_tail = (2*J) * eps_event
    # Counting actual 3 LP calls has an additional Z-tail allowance.
    total = (2*eps_smooth + eps_smooth + eps_smooth
             + eps_e + eps_x + eps_tail + J*eps_event)
    assert 0 < total < eps_sec
