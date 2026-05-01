# Noiseless GPG pulse sequences

Generated from `datas/noiseless_gpgs_pulses/cache/gpg_exact_ad_sweep_cache.pkl` for the final noiseless GPG implementation of the `(3,3,1)` BGM recovery circuit.

Files:
- `all_pulses.csv`: one row per GPG pulse, including `p`, recovery component, eigen-index, and angles `[alpha, beta, gamma, kappa]` in radians.
- `manifest.csv`: one row per synthesized state-preparation sequence, including the controlled/rank-one phase metadata where available.
- `summary_by_point_and_factor.csv`: counts of sequences and total GPG pulses by damping strength and recovery component.
- `sequences/*.csv`: one file per synthesized sequence, with columns `pulse, alpha, beta, gamma, kappa`.
- `cache/*.pkl`: active sweep cache plus timestamped backups from refinement/evaluation passes.

Each pulse is applied as
`Rz(alpha) Ry(beta) Rz(gamma) G(kappa)`, with `G(kappa)=exp(-i kappa J_z^2)`.
