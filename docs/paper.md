# Associated Paper

This repository accompanies:

**Recovery Algorithm for Correlated Errors in Permutation-Invariant Quantum Codes**

Authors: Omprakash Chandra, Yingkai Ouyang, Gopikrishnan Muraleedharan, and Gavin K. Brennen.


The paper studies quantum error recovery for permutation-invariant quantum
codes under correlated amplitude-damping noise, introduces new family of PI codes called CAD codes giving explicit construction of 4 qubit CAD4 and 9 qubit CAD9 PI
codes, and compiles recovery maps using geometric phase gate primitives.

Repository map for the paper:

- `src/qer/`: code constructors, noise models, recovery optimization, Petz/BK
  recovery, and GPG compilation utilities.
- `examples/`: executed tutorial notebooks showing basic recovery and GPG
  synthesis workflows.
- `paper_plots/`: executed notebooks for paper plot/data summaries.
- `datas/final_gpg_pulses/`: curated detuned GPG pulse data used for the paper
  figure.
- `scripts/parallel_noisy_gpg_search.py`: advanced helper for cache-driven
  detuned noisy-GPG pulse searches.

Citation metadata is provided in `CITATION.cff`.
