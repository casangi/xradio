# xradio
Xarray Radio Astronomy Data IO is still in development.

[![Python 3.9 3.10 3.11](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/downloads/release/python-380/)

# Installing
It is recommended to use the [conda](https://docs.conda.io/projects/conda/en/latest/) environment manager to create a clean, self-contained runtime where xradio and all its dependencies can be installed:
```sh
conda create --name xradio python=3.11 --no-default-packages
conda activate xradio

```
> 📝 On macOS it is required to pre-install `python-casacore` using `conda install -c conda-forge python-casacore`.

Making xradio available for download from conda-forge directly is pending, so until then the current recommendation is to sully that pristine environment by calling pip [from within conda](https://www.anaconda.com/blog/using-pip-in-a-conda-environment), like this:
```sh
pip install xradio
```
