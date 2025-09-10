# xradio
Xarray Radio Astronomy Data IO is still in development.

[![Python 3.11 3.12 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3130/)
[![Linux Tests](https://github.com/casangi/xradio/actions/workflows/python-testing-linux.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/python-testing-linux.yml?query=branch%3Amain)
[![macOS Tests](https://github.com/casangi/xradio/actions/workflows/python-testing-macos.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/python-testing-macos.yml?query=branch%3Amain)
[![ipynb Tests](https://github.com/casangi/xradio/actions/workflows/run-ipynb.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/run-ipynb.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/casangi/xradio/branch/main/graph/badge.svg)](https://codecov.io/gh/casangi/xradio/branch/main/xradio)
[![Documentation Status](https://readthedocs.org/projects/xradio/badge/?version=latest)](https://xradio.readthedocs.io)
[![Version Status](https://img.shields.io/pypi/v/xradio.svg)](https://pypi.python.org/pypi/xradio/)

# Installing
XRADIO can be installed in virtual environments via pip. It is recommended to use the conda environment manager from [miniforge](https://github.com/conda-forge/miniforge) to create a clean, self-contained runtime where XRADIO and all its dependencies can be installed, for example:
```sh
conda create --name xradio python=3.12 --no-default-packages
conda activate xradio
```
> 📝 On macOS it is required to pre-install `python-casacore` using `conda install -c conda-forge python-casacore`.

XRADIO can now be installed using:
```sh
pip install xradio
```
This will also install the minimal dependencies for XRADIO.

Note that if only the minimal dependencies are installed, the functionality to convert MSv2 to MSv4 will not be available.
This requires installing `python-casacore` (also included in the `all` group, see below), or alternatively the
`casatools` backend, as explained in the [casatools I/O backend guide](docs/source/measurement_set/guides/backends.md).

To install the minimal dependencies and the interactive components (JupyterLab) use:
```sh
pip install "xradio[interactive]"
```

To enable conversion from MSv2 to MSv4 use (this only works for Linux):
```sh
pip install "xradio[python-casacore]"
```
To be able to run tests:
```sh
pip install "xradio[test]"
```
Multiple-dependencies can be installed using:
```sh
pip install "xradio[interactive,python-casacore,test]"
```

To install a more complete set of dependencies:
```sh
pip install "xradio[all]"
```
This will include the dependencies required to run the interactive Jupyter notebooks, run tests, build documentation,
and python-casacore to enable MSv2=>MSv4 functionality.
