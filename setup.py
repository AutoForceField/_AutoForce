# +
from __future__ import annotations

from setuptools import find_packages, setup

with open("autoforce/version.py") as f:
    version: dict[str, str] = {}
    exec(f.read(), version)
__version__ = version["__version__"]


setup(
    name="autoforce",
    version=__version__,
    author="Amir Hajibabaei",
    author_email="autoforcefield@gmail.com",
    description="machine learning force-fields",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy", "scipy", "torch>=1.10", "ase"],
    url="https://github.com/AutoForceField/AutoForce",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
