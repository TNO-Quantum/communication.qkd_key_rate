"""
Computes the maximum effective number of useful secure bits.

For this, an estimate on the number of communication rounds is given,
as well as an efficiency of the error correction routines. The secure
number of bits is determined using the finite key-rate estimates.

Here, BBM92 is used, however, it also works for BB84.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from tno.quantum.communication.qkd_key_rate.classical._required_linktime import (
    compute_efficiency,
    compute_estimate_on_communication,
)
from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum.bbm92 import (
    BBM92FiniteKeyRateEstimate,
)

detector = standard_detector.customise(
    dark_count_rate=6e-7,
    polarization_drift=0.0707,
    error_detector=5e-3,
    efficiency_detector=0.1,
)

detector = standard_detector


@pytest.mark.parametrize("error_correction_method", ["Winnow", "Cascade"])
@pytest.mark.parametrize("double_channel_loss", [5, 10, 15, 20])
def test_required_linktime(
    error_correction_method: str, double_channel_loss: float
) -> tuple[NDArray[np.int_], list[float]]:
    # We compute the entropy for these number of pulses
    number_of_pulses = np.array(
        [int(factor * 10**exp) for exp in [9, 10, 11] for factor in range(1, 10)]
    )
    detector_bob = detector

    error_estimate = 0.05  # Estimate for the efficiency of error correction
    epsilon_security = 1e-12
    message_length = 10000
    cost_per_communication = np.ceil(np.log2(1 / epsilon_security))

    number_of_communication_rounds = compute_estimate_on_communication(
        error_correction_method, error_estimate, message_length
    )

    ec_efficiency = compute_efficiency(error_correction_method, error_estimate)
    if ec_efficiency > 0.0 and ec_efficiency < 1.0:
        error_correction_factor = 1 / ec_efficiency

    entropy = []
    for num_pulses in number_of_pulses:
        finite_key_rate = BBM92FiniteKeyRateEstimate(
            detector=detector_bob,
            number_of_pulses=num_pulses,
        )

        # Compute optimal key-rate for the given attenuation, detectors and pulses
        _, optimal_rate = finite_key_rate.optimize_rate(attenuation=double_channel_loss)

        # We use only the pulses in the X-basis for the key
        optimal_number_of_pulses = int(
            np.floor(num_pulses * optimal_rate * error_correction_factor)
        )

        # Correct for the classical authentication required
        net_number_of_pulses = (
            optimal_number_of_pulses
            - number_of_communication_rounds * cost_per_communication
        )

        entropy.append(net_number_of_pulses)

    assert entropy[0] < entropy[-1]
    return number_of_pulses, entropy


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    number_of_pulses, entropy = test_required_linktime("Winnow", 5)

    fig, ax = plt.subplots()
    ax.loglog(number_of_pulses[: len(entropy)], entropy)
    ax.set_xlabel("Number of pulses")
    ax.set_ylabel("Entropy")
    plt.show()
