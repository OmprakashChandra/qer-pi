# Release Checklist

Before making the repository public:

- Choose and add a project license.
- Add the final paper citation once the citation text is available.
- Recreate the environment with Python 3.10+ and run `pip install -e .`.
- Run `python -m unittest discover -s tests`.
- Open both notebooks in `examples/` and run them from a clean checkout.
- Confirm `plots/final_paper/` contains the exact figures linked from the paper.
- Confirm `datas/final_gpg_pulses/` contains only the intended final pulse data.
- Review `notebooks/stale/` and remove it before release if you do not want to
  publish exploratory provenance notebooks.
