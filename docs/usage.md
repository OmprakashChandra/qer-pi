# Usage Notes

Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

To run the notebooks in `examples/`, install the notebook extra:

```bash
python -m pip install -e ".[examples]"
```

The main public modules are:

- `qer.codewords`: permutation-invariant code constructors.
- `qer.noisemodel`: PIQS-based noise-channel constructors.
- `qer.optimisation`: SDP recovery optimization helpers.
- `qer.bk_recovery`: Petz/Barnum-Knill recovery helpers.
- `qer.gpgs`: geometric-phase-gate pulse and recovery utilities.

The default optimization solver is SCS through CVXPY. MOSEK remains optional:
use `solver="mosek"` only when it is installed and licensed locally.

For examples, start with:

- `examples/basic_recovery.ipynb`
- `examples/finding_optimal_pulses.ipynb`

Paper-plot reproduction notebooks live in `paper_plots/`.
