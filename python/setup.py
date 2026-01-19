from setuptools import setup, find_packages

setup(
    name="qer-codes",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "qutip",
        "cvxpy",
        "matplotlib",
    ],
)