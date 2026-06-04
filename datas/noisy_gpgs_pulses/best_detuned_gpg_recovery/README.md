# Best detuned noisy-GPG recovery data

Curated package-ready data selected by the minimum final recovered `GPG infidelity` for each `(p, C)` point present in the cache.

Files:
- `best_metrics.csv` / `best_metrics.json`: selected best point per `(p,C)`.
- `best_cache.pkl`: compact pickle containing exactly the selected best points and pulse sequences.
- `manifest.csv`: one row per per-target pulse-sequence CSV.
- `all_pulses.csv`: flattened pulse table for all selected sequences.
- `sequences/*.csv`: per-target pulse sequences.
- `all_cache_metrics_before_selection.csv`: audit table of all candidate cache points considered before selecting the best per `(p,C)`.
- `source_plot_*`: copy of the current notebook plot/plot metrics.
- `best_detuned_gpg_cooperativity_sweep.pdf/png`: plot generated directly from `best_metrics.csv`.

Selected `p=5e-4` values:
- `C=1e+06`: `GPG infidelity=1.817435166363e-03` from `p=5.000000000000e-04|C=1.000000000000e+06|mode=detuned|restarts=3|maxiter=500|complete-prior-v1`
- `C=1e+07`: `GPG infidelity=1.966174131601e-04` from `p=5.000000000000e-04|C=1.000000000000e+07|mode=detuned|restarts=3|maxiter=500|complete-prior-v1`
- `C=1e+08`: `GPG infidelity=1.367446130174e-05` from `p=5.000000000000e-04|C=1.000000000000e+08|mode=detuned|restarts=3|maxiter=500|complete-prior-v1`
- `C=1e+09`: `GPG infidelity=8.023991106332e-06` from `p=5.000000000000e-04|C=1.000000000000e+09|mode=detuned|restarts=3|maxiter=500|complete-prior-v1`
- `C=1e+10`: `GPG infidelity=1.894021391169e-06` from `p=5.000000000000e-04|C=1.000000000000e+10|mode=detuned|restarts=3|maxiter=500|complete-prior-v1`

Selected `p=1e-3` values:
- `C=1e+06`: `GPG infidelity=1.804763401117e-03` from `p=1.000000000000e-03|C=1.000000000000e+06|mode=detuned|restarts=1|maxiter=180|complete-prior-v1`
- `C=1e+07`: `GPG infidelity=3.512596169049e-04` from `p=1.000000000000e-03|C=1.000000000000e+07|mode=detuned|restarts=1|maxiter=180|complete-prior-v1`
- `C=1e+08`: `GPG infidelity=4.532017858005e-05` from `p=1.000000000000e-03|C=1.000000000000e+08|mode=detuned|restarts=3|maxiter=500|complete-prior-v1`
- `C=1e+09`: `GPG infidelity=2.044358211595e-05` from `p=1.000000000000e-03|C=1.000000000000e+09|mode=detuned|restarts=3|maxiter=500|complete-prior-v1`
- `C=1e+10`: `GPG infidelity=5.070257629636e-06` from `p=1.000000000000e-03|C=1.000000000000e+10|mode=detuned|restarts=1|maxiter=180|complete-prior-v1`
