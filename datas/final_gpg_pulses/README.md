# Final Detuned GPG Pulse Data

Curated package-ready data selected by the minimum final recovered
`GPG infidelity` for each `(p, C)` point present in the cache. These are the
detuned noisy-GPG pulse sequences used for the paper figure.

Each pulse row has five pulse parameters:

`alpha, beta, gamma, kappa, detuning`

Files:
- `manifest.csv`: one row per per-target pulse-sequence CSV.
- `all_pulses.csv`: flattened pulse table for all selected sequences.
- `sequences/*.csv`: per-target pulse sequences with columns `pulse_index, alpha, beta, gamma, kappa, detuning`.
- `best_metrics.csv` / `best_metrics.json`: minimal selected-point metadata for the `(p,C)` values represented by the pulses.

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
