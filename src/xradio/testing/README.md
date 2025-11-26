# XRADIO Testing Module

The `xradio.testing` module provides utilities for testing and benchmarking XRADIO functionality. This module is part of XRADIO's public API and can be used in external projects, including ASV benchmark tests stored in separate repositories.

## Installation

See the main [README.md](https://github.com/casangi/xradio/blob/main/README.md) for installation instructions.


## Submodules

The testing module includes the following submodules:

- **`assertions`**: Functions for validating data structures and schemas
- **`fixtures`**: Reusable test fixtures for setting up test data
- **`measurement_set`**: Utilities for generating, checking, and manipulating measurement set test data
- **`utils`**: General testing utilities

## Usage

Import the testing utilities in your code:

```python
from xradio.testing.measurement_set.io import download_measurement_set, convert_msv2_to_processing_set
from xradio.testing.measurement_set.generator import gen_test_ms
```

### Example: Using in ASV Benchmarks

For ASV (airspeed velocity) benchmarks, you can use the testing utilities to set up benchmark tests:

```python
from xradio.testing.measurement_set.io import (
    download_measurement_set, 
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

### Example: Checker Functions

Use checker functions to check conversions of MSv2 to MSv4, etc.

```python
from xradio.testing.measurement_set.checker import check_msv4_matches_descr
from xradio.testing.measurement_set.generator import gen_minimal_ms

msname, description = gen_minimal_ms.msname()


# Assuming you have msv4_xdt and msv2_descr
check_msv4_matches_descr(msv4_xdt, ms_minimal_required.descr)
if issues:
    print(f"Validation issues: {issues}")
```

## Contributing

When adding new testing utilities, place them in the appropriate submodule within `src/xradio/testing/`. Ensure they are well-documented and include type hints where possible.
