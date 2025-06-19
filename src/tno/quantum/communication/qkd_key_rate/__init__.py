"""TNO Quantum communication key-rate module.

The :py:mod:`tno.quantum.communication.qkd_key_rate` package provides python code to
compute optimal protocol parameters for different quantum key distribution (QKD)
protocols.

The codebase is based on the following papers:

- Attema et al. - Optimizing the decoy-state BB84 QKD protocol parameters (2021)(https://doi.org/10.1007/s11128-021-03078-0)
- Ma et al. - Quantum key distribution with entangled photon sources (2007)(http://doi.org/10.1103/PhysRevA.76.012307)


The following quantum protocols are supported:

- BB84 protocol (:py:mod:`~tno.quantum.communication.qkd_key_rate.quantum.bb84`),
- BB84 protocol using a single photon source (:py:mod:`~tno.quantum.communication.qkd_key_rate.quantum.bb84_single_photon`),
- BBM92 protocol (:py:mod:`~tno.quantum.communication.qkd_key_rate.quantum.bbm92`).

The following classical error-correction protocols are supported:

- Cascade (:py:mod:`~tno.quantum.communication.qkd_key_rate.classical.cascade`),
- Winnow (:py:mod:`~tno.quantum.communication.qkd_key_rate.classical.winnow`).

The presented code can be used to

- determine optimal parameter settings needed to obtain the maximum key rate,
- correct errors in exchanged sifted keys for the different QKD protocols,
- apply privacy amplification by calculating secure key using hash function.

Usage examples can be found in relevant submodules.
"""  # noqa: E501

__all__: list[str] = []


__version__ = "2.0.1"
