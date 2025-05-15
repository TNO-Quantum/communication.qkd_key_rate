"""Classical Error-Correction Modules.

The following classical error-correction protocols are supported:

- Cascade (:py:mod:`~tno.quantum.communication.qkd_key_rate.classical.cascade`),
- Winnow (:py:mod:`~tno.quantum.communication.qkd_key_rate.classical.winnow`).

The presented code can be used to:

- Correct errors in exchanged sifted keys for different QKD protocols,
- Enhance the reliability and security of the generated keys.

Usage examples can be found in the relevant submodules.
"""

# isort: skip_file

from tno.quantum.communication.qkd_key_rate.classical._message import (
    Message,
)
from tno.quantum.communication.qkd_key_rate.classical._permutations import (
    Permutations,
)
from tno.quantum.communication.qkd_key_rate.classical._schedule import (
    Schedule,
)
from tno.quantum.communication.qkd_key_rate.classical._parity_strategy import (
    ParityStrategy,
)

from tno.quantum.communication.qkd_key_rate.classical._sender import (
    SenderBase,
)
from tno.quantum.communication.qkd_key_rate.classical._receiver import (
    ReceiverBase,
)
from tno.quantum.communication.qkd_key_rate.classical._corrector import (
    Corrector,
    CorrectorOutputBase,
)
from tno.quantum.communication.qkd_key_rate.classical._required_linktime import (
    compute_estimate_on_communication,
    compute_efficiency,
)


__all__ = [
    "Corrector",
    "CorrectorOutputBase",
    "Message",
    "ParityStrategy",
    "Permutations",
    "ReceiverBase",
    "Schedule",
    "SenderBase",
    "compute_efficiency",
    "compute_estimate_on_communication",
]
