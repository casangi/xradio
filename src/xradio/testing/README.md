# XRADIO Testing Module

XRADIO provides a testing module (`xradio.testing`) as part of its public API. This module contains utilities for testing that are designed to be  used also in external projects, including ASV benchmark tests stored in [benchviper](https://github.com/casangi/benchviper).

## Submodules

The testing module includes the following submodules:

<!-- **`assertions (TBD)`**: Functions for validating data structures and schemas
- **`fixtures (TBD)`**: Reusable test fixtures for setting up test data -->
- **`measurement_set`**: Utilities for generating, checking, and manipulating MeasurementSet test data
- **`_utils (TBD)`**: Private testing utilities

The `measurement_set` submodule contains several Python modules that provide specific functionality:

- **`io.py`**: I/O helpers to download test MeasurementSets (non-casacore dependent)
- **`checker.py`**: Validation functions to check MSv4 data structures against expected specifications
- **`msv2_io.py`**: MSv2-specific I/O operations and test data generators (casacore-dependent)

These are Python modules and their functions are exported through the package's `__init__.py` for convenient access.

# Test Framework

XRADIO Tests use pytest and are located in the `tests` directory. There are two types of tests:
- Unit Tests: located in `tests/unit`
- stakeholder Tests: located in `tests/stakeholder`

Helper functions used in the tests are defined in this module `xradio.testing` where new helper functions should be added.
- conftest.py files should import helper functions from `xradio.testing`.
- Pytest configuration is located in: `tests/*/conftest.py`



# Test Development
There is a template for unit tests in `_utils/__template__.py` that should be used as an example to write a new test script.

Here's documentation on how to use `pytest` fixtures with `conftest.py`, including examples and best practices:

## What is conftest.py?

* `conftest.py` is a special configuration file recognized by `pytest`.
* It allows you to define fixtures that are **automatically discovered** and made available to all test modules in the same directory and subdirectories.
* No need to import fixtures explicitly from `conftest.py`.

## Creating Tests, Using Fixtures and Helper Functions

### **1. Define a Fixture in `conftest.py`**

```python
# conftest.py
from collections import namedtuple
import os
from pathlib import Path
import pytest
import shutil

import xarray as xr
from xradio.measurement_set import open_processing_set

from xradio.testing.measurement_set.msv2_io import (
    gen_test_ms,
    gen_minimal_ms,
    make_ms_empty,
    build_minimal_msv4_xdt,
    build_processing_set_from_msv2
)
from xradio.testing.measurement_set.io import (
    download_measurement_set
)

@pytest.fixture
def convert_measurement_set_to_processing_set(request, tmp_path):
    """Create a processing set from test MS for testing"""
    ps_path = tmp_path / "test_processing_set.ps.zarr"
    ms_path = download_measurement_set(request.param, tmp_path)
    # Convert MS to processing set
    build_processing_set_from_msv2(
        ms_path,
        ps_path,
        partition_scheme=[],
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        use_table_iter=False,
        overwrite=True,
        parallel_mode="none",
    )
    open_processing_set(str(ps_path))  # check it opens
    yield ps_path
    shutil.rmtree(ps_path)
    shutil.rmtree(ms_path)
```

> In pytest, yield is used in fixtures to define setup and teardown. When the fixture is first called, it runs until it hits the first yield. It pauses and returns the value to the test. After the test executes, the fixture resumes from where it left off.

Fixtures can have different **scopes**, controlling how often they are created:

* `function` (default): new instance for each test function
* `class`: one instance per test class
* `module`: one instance per module
* `session`: one instance for the entire test session

### **2. Use the Fixture in a Unit Test**

```python
# Example from tests/unit/measurement_set/test_load_processing_set.py

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

Fixtures can be used in other tests. As long as the test files are in the same directory or subdirectories of the one containing `conftest.py`, fixtures defined in `conftest.py` are available without import.

### Best Practices

* Use `conftest.py` for fixtures that are reused across multiple files.
* Use descriptive fixture names.
* Avoid using `autouse=True` unless necessary—it can make tests harder to understand.


### **3. Use the Testing Module in ASV Benchmark Tests**

For ASV (airspeed velocity) benchmarks, you can use the testing utilities to set up benchmark tests
even though the tests are stored in an external project.

```python
# Example from https://github.com/casangi/benchviper/xradio/benchmarks/measurement_set.py

from xradio.testing.measurement_set.io import download_measurement_set
from xradio.testing.measurement_set.msv2_io import (
    build_processing_set_from_msv2, 
    build_minimal_msv4_xdt
)

class TestLoadProcessingSet:
    MeasurementSet = "Antennae_North.cal.lsrk.split.ms"
    processing_set = "/tmp/benchmark_processing_set.zarr"

    def setup_cache(self):

        # Download test measurement set
        ms_path = download_measurement_set(self.MeasurementSet)

        # Convert MS to processing set
        ps_path = self.processing_set

        build_processing_set_from_msv2(
            in_file=ms_path,
            out_file=self.processing_set,
            partition_scheme=[],
            overwrite=True,
            parallel_mode="none",
            main_chunksize=0.01,
            pointing_chunksize=0.00001,
            pointing_interpolate=True,
            ephemeris_interpolate=True,
            use_table_iter=False,
        )

    def time_basic_load(self):
        """Test basic loading of processing set without parameters"""
        ps_xdt = load_processing_set(self.processing_set)
    
```

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

# Contributing

When adding new testing utilities, place them in the appropriate submodule within `src/xradio/testing/`. Ensure they are well-documented and include type hints where possible.
