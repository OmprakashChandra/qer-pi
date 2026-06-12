# Data And Figures

The public data payload is `datas/final_gpg_pulses/`.

It contains:

- `manifest.csv`: one row per per-target sequence file.
- `all_pulses.csv`: flattened table of all selected pulses.
- `sequences/*.csv`: one CSV per target, with columns `pulse_index`, `alpha`,
  `beta`, `gamma`, `kappa`, and `detuning`.
- `best_metrics.csv` and `best_metrics.json`: selected-point metadata.

The public figure exports are in `plots/final_paper/`.

Exploratory plots are retained in `plots/other_plots/`. Older sweep caches,
scratch files, and intermediate amplitude-damping tables are not part of the
release branch.
