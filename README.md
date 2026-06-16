# QER

Quantum error recovery utilities and reproducibility material for the
permutation-invariant code numerical calculations in the accompanying paper.

This repository lets you play with the variety of Permutation-Invariant quantum codes (PI codes) including gnu, bg, bgm, 7-qubit Pollatsek-Ruskai, Gross code, ... under global and local symmetric noise. In the paper, we have focused on amplitude damping noise. However, we also have support for depolarizing noise. For more look at `src/qer/noisemodel.py`. For building the Lindbladians, we have used Permutation-Invariant Quantum Solver (PIQS) (link). Once your state goes through the noise model, you can run optimal error recovery implemented in ``src/qer/optimisation.py`, and construct the recovery circuit (with compilation done under the hood) and find the pulse sequences using geometric phase gate sequences to build the primitives involved in the recovery circuit implemented in `src/qer/gpgs.py`. 

## Repository Map

- `src/qer/`: Python package. New code should import from `qer.*`.
- `examples/`: clean notebooks for first-time users.
- `scripts/`: advanced cache-driven helper for detuned noisy-GPG pulse search; requires explicit `--cache-path` inputs.
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
python -m pip install -e .
```

To run the example notebooks, install the optional notebook extra:

```bash
python -m pip install -e ".[examples]"
```

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


