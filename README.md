# xradio
Xarray Radio Astronomy Data IO is still in development.

[![Python 3.9 3.10 3.11 3.12](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%203.12-blue)](https://www.python.org/downloads/release/python-380/)

# Installing
It is recommended to use the conda environment manager from [miniforge](https://github.com/conda-forge/miniforge) to create a clean, self-contained runtime where XRADIO and all its dependencies can be installed:
```sh
conda create --name xradio python=3.12 --no-default-packages
conda activate xradio
```
> 📝 On macOS it is required to pre-install `python-casacore` using `conda install -c conda-forge python-casacore`.

XRADIO can now be installed using:
```sh
pip install xradio
```
This will also install the minimal dependencies for XRADIO. To install the minimal dependencies and the interactive components (JupyterLab) use:
```sh
pip install "xradio[interactive]"
```