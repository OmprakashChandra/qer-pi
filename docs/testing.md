# Testing

Run the release sanity checks with:

```bash
python -m unittest discover -s tests
```

The tests cover:

- final GPG pulse-data schema and sequence-file headers,
- example notebook JSON validity and unexecuted outputs,
- basic codeword normalization when scientific dependencies are installed,
- the SCS optimization path on a tiny identity-channel problem when scientific
  dependencies are installed.

Tests that require optional scientific dependencies skip automatically when
those packages are absent from the active Python environment.
