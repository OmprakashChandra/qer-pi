# QER

Quantum error recovery utilities and reproducibility material for the
permutation-invariant code calculations in the accompanying paper.

This repository contains the installable Python package, generated
pulse-sequence data, notebooks, helper scripts, and final plotting outputs used
during the project.

## Repository Map

- `src/qer/`: public Python package. New code should import from `qer.*`.
- `scripts/`: long-running helper scripts used to generate or refine
  GPG pulse-search data.
- `notebooks/`: exploratory and figure-generation notebooks.
- `datas/`: tracked CSV summaries and pulse-sequence data.
- `plots/`: generated figures, including paper-ready outputs.
- `other/`: supplementary source files that are not part of the Python package.

## Python Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package targets Python 3.10+ and was developed with Python 3.12.

## Quick Start

```python
from qer.codewords import bgmcode_piqs
from qer.noisemodel import noisemodel
from qer.optimisation import optimise

rho, logical0, logical1 = bgmcode_piqs(b=3, g=3, m=1)
noise = noisemodel(
    "global symmetric amplitude damping",
    num_qubits=9,
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

## Data And Outputs

The tracked CSV files and pulse sequences under `datas/` are intended to be
reusable without rerunning the expensive searches. Large intermediate caches and
parallel-search scratch files are ignored by Git.

The `plots/final_paper/` directory contains the most stable paper-facing figure
exports. Other plot folders contain exploratory or intermediate outputs retained
for reproducibility.

## Release Notes

This branch is being prepared as the public release branch. Before final public
release, add a project license and a paper citation once the final citation text
is available.
