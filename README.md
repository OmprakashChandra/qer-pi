# QER

Quantum error recovery utilities and reproducibility material for the
permutation-invariant code calculations in the accompanying paper.

This repository contains the installable Python package, generated
pulse-sequence data, notebooks, helper scripts, and final plotting outputs used
during the project.

## Repository Map

- `src/qer/`: public Python package. New code should import from `qer.*`.
- `examples/`: clean notebooks for first-time users.
- `scripts/`: long-running helper scripts used to generate or refine
  GPG pulse-search data. These scripts are cache-driven and require explicit
  `--cache-path` inputs.
- `notebooks/stale/`: archived exploratory notebooks kept for provenance.
- `tests/`: lightweight release sanity checks.
- `docs/`: usage notes, data notes, and release checklist.
- `datas/final_gpg_pulses/`: final detuned GPG pulse data used for the paper
  figure; pulse rows contain `alpha`, `beta`, `gamma`, `kappa`, and
  `detuning`.
- `plots/final_paper/`: paper-ready figure exports.
- `plots/other_plots/`: archived exploratory and intermediate plot outputs.
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

## Examples

Clean starting-point notebooks are in `examples/`:

- `examples/basic_recovery.ipynb`
- `examples/load_final_gpg_pulses.ipynb`

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

The `qer.optimisation.optimise` helper defaults to `solver="scs"` for a free
out-of-the-box solver. Use `solver="mosek"` only when MOSEK is installed and
licensed locally.

## Tests

```bash
python -m unittest discover -s tests
```

Some tests skip automatically if optional scientific dependencies are not
installed in the active environment.

## Data And Outputs

The public data payload is `datas/final_gpg_pulses/`. It contains the selected
detuned noisy-GPG pulse sequences used for the paper figure. Each pulse has the
five parameters `alpha`, `beta`, `gamma`, `kappa`, and `detuning`.

Older sweep caches, intermediate amplitude-damping result tables, and exploratory
pulse-search outputs are intentionally not part of this release branch.

The `plots/final_paper/` directory contains the stable paper-facing figure
exports. Exploratory and intermediate plots are retained under
`plots/other_plots/` so they do not crowd the public-facing figure set.

## Release Notes

This branch is being prepared as the public release branch. Before final public
release, add a project license and a paper citation once the final citation text
is available.

See `docs/release_checklist.md` for the remaining release checks.
