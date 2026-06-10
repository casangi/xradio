# xradio
Xarray Radio Astronomy Data IO is still in development.

[![Python 3.11 3.12 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3130/)
[![Linux Tests](https://github.com/casangi/xradio/actions/workflows/python-testing-linux.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/python-testing-linux.yml?query=branch%3Amain)
[![macOS Tests](https://github.com/casangi/xradio/actions/workflows/python-testing-macos.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/python-testing-macos.yml?query=branch%3Amain)
[![ipynb Tests](https://github.com/casangi/xradio/actions/workflows/run-ipynb.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/run-ipynb.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/casangi/xradio/branch/main/graph/badge.svg)](https://codecov.io/gh/casangi/xradio/branch/main/xradio)
[![Documentation Status](https://readthedocs.org/projects/xradio/badge/?version=latest)](https://xradio.readthedocs.io)
[![Version Status](https://img.shields.io/pypi/v/xradio.svg)](https://pypi.python.org/pypi/xradio/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-Tests-orange)](https://casangi.github.io/benchviper/xradio)

# Installing
XRADIO can be installed in virtual environments via pip. It is recommended to use the conda environment manager from [miniforge](https://github.com/conda-forge/miniforge) to create a clean, self-contained runtime where XRADIO and all its dependencies can be installed, for example:
```sh
conda create --name xradio python=3.13 --no-default-packages
conda activate xradio
```
> 📝 MSv2=>MSv4 conversion and CASA image I/O use [`casacoretables`](https://github.com/casangi/casacoretables), a standalone, dependency-light build of casacore's table system that works the same on **Linux and macOS** (no `conda install` of `python-casacore` and no `casatools` are needed). PyPI wheels for `casacoretables` are not published yet, so it currently has to be installed from source.

XRADIO can now be installed using:
```sh
pip install xradio
```
This installs only the minimal dependencies for XRADIO, which allow you to use the schema checker and export schemas to JSON. **Note that if only the minimal dependencies are installed, the functionality to open data stored using zarr and to convert MSv2 to MSv4 will not be available.**

To install the zarr backend use:
```sh
pip install "xradio[zarr]"
```
This allows for opening data stored using zarr. 

To install the zarr backend and the interactive components (JupyterLab) use:
```sh
pip install "xradio[interactive]"
```

To install the casacore backend along with the zarr backend which enables conversion from MSv2 to MSv4 use (works on Linux and macOS):
```sh
pip install "xradio[casacore]"
```
This pulls in `casacoretables`. Until its PyPI wheels are published, install it from source first (e.g. `pip install /path/to/casacoretables`).

To installs all the needed packages to run the unit tests:
```sh
pip install "xradio[test]"
```
This also installs the zarr backend and the `casacoretables` backend, which works on both Linux and macOS.

Multiple-dependencies can be installed using:
```sh
pip install "xradio[interactive,casacore,test]"
```

To install a more complete set of dependencies:
```sh
pip install "xradio[all]"
```
This will include the dependencies required to run the interactive Jupyter notebooks, run tests, build documentation,
and python-casacore to enable MSv2=>MSv4 functionality on Linux.

Instruction of how to setup a developer environment can be found at [Development](https://xradio.readthedocs.io/en/latest/development.html).

Instruction of how to setup a developer environment using casatools instead of python-casacore can be found at [casatools I/O backend guide](docs/source/measurement_set/guides/backends.md).
