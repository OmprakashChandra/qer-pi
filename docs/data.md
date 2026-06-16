# Data

The data payload is `datas/final_gpg_pulses/`.

It contains:

- `manifest.csv`: one row per per-target sequence file.
- `all_pulses.csv`: flattened table of all selected pulses.
- `sequences/*.csv`: one CSV per target, with columns `pulse_index`, `alpha`,
  `beta`, `gamma`, `kappa`, and `detuning`.
- `best_metrics.csv` and `best_metrics.json`: selected-point metadata.

The bundle is limited to curated pulse sequences and summary metrics.
