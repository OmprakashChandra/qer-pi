# Release Checklist

Before making the repository public:

- Choose and add a project license.
- Add the final paper citation once the citation text is available.
- Recreate the environment with Python 3.10+ and run `pip install -e ".[examples]"`.
- Run `python -m unittest discover -s tests`.
- Open both notebooks in `examples/` and run them from a clean checkout.
- Confirm `datas/final_gpg_pulses/` contains only the intended final pulse data.
- Confirm exploratory notebooks, plots, and supplementary paper-only files remain
  on the archival branch rather than this lean release branch.
