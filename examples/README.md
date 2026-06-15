# Examples

Start here after installing the package from the repository root:

```bash
python -m pip install -e ".[examples]"
```

If a notebook reports that a dependency such as `pandas` is missing, check that
the notebook kernel is using the same Python environment where `qer` was
installed.

- `basic_recovery.ipynb`: construct a small PI code, build an amplitude-damping
  channel, and run recovery optimization with the free SCS solver.
- `load_final_gpg_pulses.ipynb`: inspect the final detuned GPG pulse data used
  for the paper figure.
