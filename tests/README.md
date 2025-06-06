# XRADIO Tests
Xarray Radio Astronomy Data IO Tests

[![Python 3.11 3.12 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3130/)
[![Linux Tests](https://github.com/casangi/xradio/actions/workflows/python-testing-linux.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/python-testing-linux.yml?query=branch%3Amain)
[![macOS Tests](https://github.com/casangi/xradio/actions/workflows/python-testing-macos.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/python-testing-macos.yml?query=branch%3Amain)
[![ipynb Tests](https://github.com/casangi/xradio/actions/workflows/run-ipynb.yml/badge.svg?branch=main)](https://github.com/casangi/xradio/actions/workflows/run-ipynb.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/casangi/xradio/branch/main/graph/badge.svg)](https://codecov.io/gh/casangi/xradio/branch/main/xradio)
[![Documentation Status](https://readthedocs.org/projects/xradio/badge/?version=latest)](https://xradio.readthedocs.io)
[![Version Status](https://img.shields.io/pypi/v/xradio.svg)](https://pypi.python.org/pypi/xradio/)

# Test Framework
XRADIO Tests use pytest and are located in the `tests` directory. There are two types of tests:
- Unit Tests: located in `tests/unit`
- Component Tests: located in `tests/component`

Helper functions used in the tests:
- Helper and utitility functions for tests: located in `tests/_utils`
- Pytest configuration is located in: `tests/conftest.py`

# Running Tests
After building XRADIO using the ```pip install "xradio[test]``` command as described in the XRADIO README, tests can be run using:
```sh
pytest tests/unit
```

To run the component tests, use:
```sh
pytest tests/component
```

To run all tests, use:
```sh
pytest tests
```
To check the coverage of the tests in a local branch, use:
```sh
pytest --cov=xradio tests
```

# Test Development
There is a template for unit tests in `tests/__template__.py` that should be used as an example to write a new test script.

