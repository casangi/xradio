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

Making XRADIO available for download from conda-forge directly is pending, so until then the current recommendation is to sully that pristine environment by calling pip [from within conda](https://www.anaconda.com/blog/using-pip-in-a-conda-environment), like this:
```sh
pip install xradio
```
