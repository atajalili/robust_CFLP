from setuptools import setup, find_packages

setup(
    name="rcflp",
    version="0.1.0",
    description="Robust Congested Facility Location Problem — solvers and utilities",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "openpyxl",
        # gurobipy must be installed separately with a valid licence
    ],
)
