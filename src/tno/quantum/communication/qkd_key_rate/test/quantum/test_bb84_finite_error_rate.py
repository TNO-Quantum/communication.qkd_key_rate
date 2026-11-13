"""Regression test for the finite BB84 privacy-amplification error rate."""

import numpy as np

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum import bb84


def test_finite_privacy_amplification_uses_error_rate(monkeypatch) -> None:
    """The final entropy term must consume a probability, not an event count."""
    entropy_inputs: list[float] = []
    original_h = bb84.h

    def record_entropy_input(value):
        entropy_inputs.append(float(np.asarray(value).item()))
        return original_h(value)

    monkeypatch.setattr(bb84, "h", record_entropy_input)
    estimator = bb84.BB84FiniteKeyRateEstimate(
        detector=standard_detector,
        number_of_pulses=int(1e12),
        number_of_decoy=2,
    )
    estimator.compute_rate(
        mu=[0.980057214, 0.0576948581, 0.0000029565],
        attenuation=0.2,
        probability_basis_X=[0.7, 0.1, 0.05],
        probability_basis_Z=[0.1, 0.03, 0.02],
    )

    # LP privacy term, final privacy term, error-correction term.
    assert len(entropy_inputs) == 3
    assert 0 < entropy_inputs[1] <= 1
