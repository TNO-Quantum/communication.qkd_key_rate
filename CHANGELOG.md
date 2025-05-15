# 2.0.0 (2025 - 05 - 15)

### Features

* **Consistent `optimize_rate`:** Now consistent dict output for all optimized arguments.
* **Documentation:** Improved documentation/examples.
* **Compatibility:** Compatible with python 3.11+ and `numpy>=2.0`

## Bugfixes

* **Exposed bits**: The method `CascadeSender.build_parity_string` checks all of the parities for the needed blocks in each iteration. However, these parities are then never used and instead checked again by `CascadeReceiver.check_match_of_parities` during the iteration. 

The exposed bits of the `Cascade` protocols remains an upper bound as it always completes `number_of_passes` passes, even if there are no more errors found. This means that it often does unnecessary communication rounds and leaks further bits. 

## BREAKING CHANGES

- Replace `x_0` for `x0`
- Remove the `protocols` namespace `tno.quantum.communication.qkd_key_rate.protocols.` which means all imports are relocated.

# 1.0.0 (2023 - 06 - 04)

Initial public release v1.0.0

* **Quantum protocols:** BB84, BB84 single photon, BBM92
* **Classical protocols:** Cascade, Winnow, Privacy amplification
