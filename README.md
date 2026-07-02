# Quantum Error Recovery for Permutation-Invariant Codes (qer-pi)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21126476.svg)](https://doi.org/10.5281/zenodo.21126476)

This repository implements quantum error recovery protocols for permutation-invariant quantum codes (PI codes). The main workflow is to choose a PI code, apply a physical noise model, use semidefinite programming (SDP) to compute an optimized recovery map, and then compile the recovery map into a coherent recovery circuit.

The repository supports a range of short-length PI codes, including GNU codes, binomial-like PI codes, bg/bgm codes, the 7-qubit Pollatsek--Ruskai code, Gross code, and the CAD PI codes introduced in our paper. We study their performance under both global symmetric and local symmetric noise. In the paper, we focus mainly on amplitude-damping (AD) noise, but the code also supports depolarizing noise and can be extended to custom error models. See `src/qer/noisemodel.py` for the available noise channels and how to define new ones.

For constructing collective open-system dynamics, we use the [Permutation-Invariant Quantum Solver (PIQS)](https://qutip.org/docs/latest/apidoc/piqs.html). After the encoded state evolves under the chosen noise model, the optimal recovery map can be computed using the SDP routines in `src/qer/optimisation.py`. The resulting recovery map can then be converted into a coherent recovery circuit, with the compilation handled under the hood. The geometric phase gate (GPG) primitives and pulse-sequence tools used for the compiled recovery circuits are implemented in `src/qer/gpgs.py`.

In short, this repository provides tools to benchmark PI codes under structured noise, compute optimized quantum error recovery maps, and study how those abstract recovery maps can be compiled into physically motivated collective-control primitives.

## Associated Paper

This repository accompanies **Recovery Algorithm for Correlated Errors in
Permutation-Invariant Quantum Codes** by Omprakash Chandra, Yingkai Ouyang,
Gopikrishnan Muraleedharan, and Gavin K. Brennen. See `docs/paper.md` for the
paper-specific repository map and `CITATION.cff` for citation metadata. The
v0.1.0 repository release is archived on Zenodo:
<https://doi.org/10.5281/zenodo.21126476>.

The paper is coming soon. Until the public paper link is available, please cite
the archived repository release directly; the citation metadata will be updated
with the paper identifier after release.

## Repository Map

- `src/qer/`: Python package. New code should import from `qer.*`.
- `examples/`: clean notebooks for first-time users.
- `paper_plots/`: notebooks for reproducing plots and data summaries used in
  the paper.
- `scripts/`: advanced cache-driven helpers for detuned noisy-GPG pulse search; requires explicit `--cache-path` inputs.
- `tests/`: lightweight checks for package behavior and bundled data.
- `docs/`: usage, testing, and data notes.
- `datas/final_gpg_pulses/`: final detuned GPG pulse data used for the paper figures. Pulse rows contain `alpha`, `beta`, `gamma`, `kappa`, and `detuning`.

## Python Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

To run the example notebooks, install the optional notebook extra:

```bash
python -m pip install -e ".[examples]"
```

## Examples

Clean starting-point notebooks are in `examples/`:

- `examples/basic_recovery.ipynb`
- `examples/finding_optimal_pulses.ipynb`

Paper-plot notebooks are in `paper_plots/`:

- `paper_plots/load_final_gpg_pulses.ipynb`

## Quick Start

```python
from qer.codewords import bgmcode_piqs
from qer.noisemodel import noisemodel
from qer.optimisation import optimise

b, g, m = 3, 3, 1
rho, logical0, logical1 = bgmcode_piqs(b=b, g=g, m=m)
num_qubits = 2 * b * m + g

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

`optimise(..., solver="mosek")` can be faster for larger SDPs, but it requires a local MOSEK installation and license. If you use MOSEK, set `MOSEKLM_LICENSE_FILE` in your shell to your local license path.

## License

This project is released under the MIT License. See `LICENSE`.
