# Noisy GPG pulse sequences

Generated from the finite-cooperativity noisy GPG sweeps for the `(3,3,1)` BGM
recovery circuit.  These pulse tables use `gpg_mode="erroneous"` and include
cooperativities

`C = 1e6, 1e7, 1e8, 1e9, 1e10`

for the AD probabilities

`p = 5e-4, 1e-3, 5e-3`.

Files:
- `metrics.csv`: one row per `(p, C)` recovery point, including exact and GPG recovered infidelities.
- `all_pulses.csv`: one row per GPG pulse, including `p`, `cooperativity`, recovery component, eigen-index, and angles `[alpha, beta, gamma, kappa]` in radians.
- `manifest.csv`: one row per synthesized state-preparation sequence, including the controlled/rank-one phase metadata where available.
- `summary_by_point_and_factor.csv`: counts of sequences and total GPG pulses by `(p, C)` and recovery component.
- `sequences/*.csv`: one file per synthesized sequence, with columns `pulse, alpha, beta, gamma, kappa`.
- `cache/*.pkl`: copies of the active noisy sweep caches used to generate these tables.

Each pulse is applied as
`Rz(alpha) Ry(beta) Rz(gamma) G(kappa)`, with `G(kappa)=exp(-i kappa J_z^2)`.

The erroneous GPG optimizer scores a trace-decreasing density-matrix evolution
using the finite-cooperativity model from `value_err_H.m`; the tabulated
`state_infidelity` is therefore `1 - <target|rho|target>` without
renormalizing the noisy state.
