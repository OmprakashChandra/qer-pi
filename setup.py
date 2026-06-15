from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="qer",
    version="0.1.0",
    description="Quantum error recovery utilities for permutation-invariant code calculations.",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "cvxpy",
        "matplotlib",
        "numpy",
        "pandas",
        "qutip",
        "scipy",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
        "examples": [
            "ipykernel",
            "notebook",
        ],
    },
)
