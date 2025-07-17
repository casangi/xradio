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
- stakeholder Tests: located in `tests/stakeholder`

Helper functions used in the tests:
- Helper and utitility functions for tests: located in `tests/_utils`
- Pytest configuration is located in: `tests/*/conftest.py`

# Test Development
There is a template for unit tests in `tests/_utils/__template__.py` that should be used as an example to write a new test script.

Here's documentation on how to use `pytest` fixtures with `conftest.py`, including examples and best practices:

## What is conftest.py?

* `conftest.py` is a special configuration file recognized by `pytest`.
* It allows you to define fixtures that are **automatically discovered** and made available to all test modules in the same directory and subdirectories.
* No need to import fixtures explicitly from `conftest.py`.

## Creating Tests and Using Fixtures

### **1. Define a Fixture in `conftest.py`**

```python
# conftest.py
import pytest
from pathlib import Path
from xradio.measurement_set import load_processing_set
from xradio.schema.check import check_datatree

def download_measurement_set(input_ms, directory="/tmp"):
    """Returns path to test MeasurementSet v2"""
    # Download MS
    download(file=input_ms, folder=directory)
    return Path(os.path.join(directory, input_ms))


@pytest.fixture
def convert_measurement_set_to_processing_set(request, tmp_path):
    """Create a processing set from test MS for testing"""
    ps_path = tmp_path / "test_processing_set.ps.zarr"
    # Convert MS to processing set
    convert_msv2_to_processing_set(
        in_file=str(download_measurement_set(request.param, tmp_path)),
        out_file=str(ps_path),
        partition_scheme=[],
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        use_table_iter=False,
        overwrite=True,
        parallel_mode="none",
    )
    yield ps_path
    shutil.rmtree(ps_path)
    shutil.rmtree(download_measurement_set(request.param, tmp_path))
```
> Because `download_measurement_set` is a function that is used within `convert_measurement_set_to_processing_set` it does not need to be decorated as a fixture in `conftest.py` In pytest, yield is used in fixtures to define setup and teardown. When the fixture is first called, it runs until it hits the first yield. It pauses and returns the value to the test. After the test executes, the fixture resumes from where it left off.

Fixtures can have different **scopes**, controlling how often they are created:

* `function` (default): new instance for each test function
* `class`: one instance per test class
* `module`: one instance per module
* `session`: one instance for the entire test session

### **2. Use the Fixture in a Test**

```python
# test_load_processing_set.py

import pytest
from pathlib import Path
from xradio.measurement_set import load_processing_set
from xradio.schema.check import check_datatree

class TestLoadProcessingSet:
    """Tests for load_processing_set using real data"""

    @pytest.mark.parametrize(
        "convert_measurement_set_to_processing_set", ["Antennae_North.cal.lsrk.split.ms"], indirect=True
    )
    def test_check_datatree(self, convert_measurement_set_to_processing_set):
        """Test that the converted MS to PS complies with the datatree schema checker"""
        ps_xdt = load_processing_set(str(convert_measurement_set_to_processing_set))
        issues = check_datatree(ps_xdt)
        # The check_datatree function returns a SchemaIssues object, not a string
        assert (
            str(issues) == "No schema issues found"
        ), f"Schema validation failed: {issues}"
```

> Because `convert_measurement_set_to_processing_set` is defined in `conftest.py`, you can use it in your test function without any import. Additionally, `@pytest.mark.parametrize` is a pytest decorator used to run a test multiple times with different values for a given input. `indirect=True` tells pytest not to pass the value directly to the test function. Instead, pytest will look for a fixture named convert_measurement_set_to_processing_set. The value "Antennae_North.cal.lsrk.split.ms" will be passed to that fixture (not to the test itself). The return value of the convert_measurement_set_to_processing_set fixture will be passed to the test function.

### **3. Use an existing Fixture in another Test**

As long as the test files are in the same directory or subdirectories of the one containing `conftest.py`, fixtures defined in `conftest.py` are available without import.

```python
# test_processing_set_xdt.py

import pytest
from pathlib import Path
from xradio.measurement_set import load_processing_set
from xradio.schema.check import check_datatree

class TestProcessingSetXdtWithEphemerisData:
    """Tests for ProcessingSetXdt using real ephemeris data loaded from disk"""

    @pytest.mark.parametrize(
        "convert_measurement_set_to_processing_set", ["ALMA_uid___A002_X1003af4_X75a3.split.avg.ms"], indirect=True
    )
    def test_check_ephemeris_datatree(self, convert_measurement_set_to_processing_set):
        """Test that the converted MS to PS complies with the datatree schema checker"""
        ps_xdt = load_processing_set(str(convert_measurement_set_to_processing_set))

        issues = check_datatree(ps_xdt)
        # The check_datatree function returns a SchemaIssues object, not a string
        assert (
            str(issues) == "No schema issues found"
        ), f"Schema validation failed: {issues}"
```

### Best Practices

* Use `conftest.py` for fixtures that are reused across multiple files.
* Use descriptive fixture names.
* Avoid using `autouse=True` unless necessary—it can make tests harder to understand.

# Running Tests
After building XRADIO using the ```pip install "xradio[test]``` command as described in the XRADIO README, tests can be run using:
```sh
pytest tests/unit
```

To run the stakeholder tests, use:
```sh
pytest tests/stakeholder
```

To run all tests, use:
```sh
pytest tests
```
To check the coverage of the tests in a local branch, use:
```sh
pytest --cov=xradio tests
```



---

