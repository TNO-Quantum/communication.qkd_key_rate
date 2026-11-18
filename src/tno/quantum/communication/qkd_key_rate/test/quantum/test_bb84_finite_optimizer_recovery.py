"""GC regression for finite BB84 optimizer recovery and validation."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector, bb84
from tno.quantum.communication.qkd_key_rate.quantum._config import OptimizationError


def _estimator():
    return bb84.BB84FiniteKeyRateEstimate(
        detector=standard_detector, number_of_pulses=int(1e9),
        number_of_decoy=1,
    )


@pytest.mark.parametrize("attenuation", [1., 2., 3., 4.])
def test_positive_rate_at_original_native_suite_losses(attenuation):
    params, rate = _estimator().optimize_rate(attenuation=attenuation)
    independently = float(np.asarray(_estimator().compute_rate(
        attenuation=attenuation, **params,
    )).item())
    assert rate > 0
    assert independently == pytest.approx(rate, rel=1e-7, abs=1e-12)
    assert np.isclose(sum(params["probability_basis_X"])
                      + sum(params["probability_basis_Z"]), 1., atol=1e-7)


def test_original_solver_success_does_not_start_fallback():
    methods = []

    def original_success(*args, **kwargs):
        methods.append(kwargs.get("method"))
        return OptimizeResult(
            success=True, x=np.array([.2, .01, .6, .2, .1, .1]),
            fun=-0.001,
        )

    with patch.object(bb84.scipy.optimize, "minimize", side_effect=original_success):
        _, rate = _estimator().optimize_rate(attenuation=1.)
    assert rate == pytest.approx(.001)
    assert methods == ["trust-constr"]


def test_reject_solver_success_when_constraints_are_invalid():
    methods = []

    def invalid(*args, **kwargs):
        method = kwargs.get("method")
        methods.append(method)
        if method == "trust-constr":
            return OptimizeResult(
                success=True, x=np.array([.5, .5, .25, .25, .25, .25]),
                fun=0.0001,
            )
        return OptimizeResult(
            success=True, x=np.array([.3, .01, .9, .9, .9, .9]),
            fun=-0.01,
        )

    with patch.object(bb84.scipy.optimize, "minimize", side_effect=invalid):
        with pytest.raises(OptimizationError, match="Unable to find solution"):
            _estimator().optimize_rate(attenuation=1.)
    assert methods == ["trust-constr"] * 3 + ["SLSQP"] * 2
