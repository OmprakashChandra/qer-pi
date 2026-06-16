# QER

Quantum error recovery utilities and reproducibility material for the
permutation-invariant code numerical calculations in the accompanying paper.

This repository contains the installable Python package, curated GPG
pulse-sequence data, clean examples, helper scripts, and lightweight tests.

## Repository Map

- `src/qer/`: Python package. New code should import from `qer.*`.
- `examples/`: clean notebooks for first-time users.
- `scripts/`: advanced cache-driven helper for detuned noisy-GPG pulse search;
  it works with five-parameter pulses and requires explicit `--cache-path`
  inputs.
- `tests/`: lightweight checks for package behavior and bundled data.
- `docs/`: usage, testing, and data notes.
- `datas/final_gpg_pulses/`: final detuned GPG pulse data used for the paper
  figure; pulse rows contain `alpha`, `beta`, `gamma`, `kappa`, and
  `detuning`.

## Python Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To run the example notebooks, install the optional notebook extra:

```bash
python -m pip install -e ".[examples]"
```

If a notebook says a dependency such as `pandas` is missing, the notebook kernel
is using a different Python environment from the one where `qer` was installed.
Switch the notebook kernel to the environment used for the install command.

The package targets Python 3.10+ and was developed with Python 3.12.

## Examples

Clean starting-point notebooks are in `examples/`:

- `examples/basic_recovery.ipynb`
- `examples/finding_optimal_pulses.ipynb`
- `examples/load_final_gpg_pulses.ipynb`

## Quick Start

```python
from qer.codewords import bgmcode_piqs
from qer.noisemodel import noisemodel
from qer.optimisation import optimise

rho, logical0, logical1 = bgmcode_piqs(b=3, g=3, m=1)
num_qubits = 2*b*m + g
noise = noisemodel(
    "global symmetric amplitude damping",
    num_qubits=num_qubits,
    gamma=1e-3,
    dt=1.0,
    return_rep="choi",
)
fidelity = optimise(logical0, logical1, noise, solver="scs")
print(fidelity)
```

`optimise(..., solver="mosek")` can be faster for larger SDPs, but it requires a
local MOSEK installation and license. If you use MOSEK, set
`MOSEKLM_LICENSE_FILE` in your shell to your local license path.

## Solver Notes

Solver installations and license files are intentionally not tracked in this
repository. The default quick-start example uses SCS through CVXPY. MOSEK is
optional and requires a local MOSEK installation and license.

The `qer.optimisation.optimise` helper defaults to `solver="scs"` for a free
out-of-the-box solver. Use `solver="mosek"` only when MOSEK is installed and
licensed locally.

## Tests

```bash
python -m unittest discover -s tests
```

Some tests skip automatically if optional scientific dependencies are not
installed in the active environment.

## Data

The data payload is `datas/final_gpg_pulses/`. It contains the selected detuned
noisy-GPG pulse sequences used for the paper figure. Each pulse has the five
parameters `alpha`, `beta`, `gamma`, `kappa`, and `detuning`.

The bundle is limited to curated pulse sequences and summary metrics.
