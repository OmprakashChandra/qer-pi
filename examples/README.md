# Examples

Start here after installing the package from the repository root:

```bash
python -m pip install -e ".[examples]"
```

- `basic_recovery.ipynb`: construct a small PI code, build an amplitude-damping
  channel, and run SDP recovery optimization with the free SCS solver.
- `finding_optimal_pulses.ipynb`: build the BGM code, apply exact collective
  amplitude damping, decompose the recovery into GPG synthesis targets, and
  optimize detuned pulse sequences for Dicke-space state preparation.
- `load_final_gpg_pulses.ipynb`: inspect the final detuned GPG pulse data used
  for the paper figure.
