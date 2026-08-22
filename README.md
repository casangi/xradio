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
[![Read the Docs Status](https://readthedocs.org)](https://xradio.readthedocs.io/en/latest/?badge=latest)


# Installing
XRADIO can be installed in virtual environments via pip. It is recommended to use the conda environment manager from [miniforge](https://github.com/conda-forge/miniforge) to create a clean, self-contained runtime where XRADIO and all its dependencies can be installed, for example:
```sh
conda create --name xradio python=3.13 --no-default-packages
conda activate xradio
```
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

To install the casacore-table backend (based on [arcae](https://github.com/ska-sa/arcae)) along with the zarr backend, which enables conversion from MSv2 to MSv4 and CASA image IO on both Linux and macOS, use:
```sh
pip install "xradio[casacore]"
```

To installs all the needed packages to run the unit tests:
```sh
pip install "xradio[test]"
```
This also installs the zarr backend and the casacore-table backend (arcae).

Multiple-dependencies can be installed using:
```sh
pip install "xradio[interactive,casacore,test]"
```

To install a more complete set of dependencies:
```sh
pip install "xradio[all]"
```
This will include the dependencies required to run the interactive Jupyter notebooks, run tests, build documentation,
and arcae to enable MSv2=>MSv4 functionality.

Instruction of how to setup a developer environment can be found at [Development](https://xradio.readthedocs.io/en/latest/development.html).
