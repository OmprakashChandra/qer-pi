# Data

The public data payload is `datas/final_gpg_pulses/`.

It contains:

- `manifest.csv`: one row per per-target sequence file.
- `all_pulses.csv`: flattened table of all selected pulses.
- `sequences/*.csv`: one CSV per target, with columns `pulse_index`, `alpha`,
  `beta`, `gamma`, `kappa`, and `detuning`.
- `best_metrics.csv` and `best_metrics.json`: selected-point metadata.

Older sweep caches, scratch files, intermediate amplitude-damping tables,
figures, and exploratory notebooks are intentionally not part of this lean
release branch.
