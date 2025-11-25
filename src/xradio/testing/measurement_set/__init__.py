"""
Test utilities for xradio - usable in both pytest and ASV benchmarks.

This module provides reusable test utilities for generating test data,
custom assertions, and helper functions for testing xradio functionality.
"""

__all__ = [
    # Generators
    "gen_test_ms",
    "make_ms_empty",
    "build_minimal_ms",
    # Validators
    "check_msv4_matches_descr",
    "check_processing_set_matches_msv2_descr",
    # IO helpers
    "download_measurement_set",
    "build_processing_set_from_msv2",
    "build_msv4_partition",
    "build_minimal_msv4_xdt",
]

# Export generators
from xradio.testing.measurement_set.generate_ms import (
    gen_test_ms,
    make_ms_empty,
    build_minimal_ms,
)

# Export validators
from xradio.testing.measurement_set.check_ms import (
    check_msv4_matches_descr,
    check_processing_set_matches_msv2_descr,
)

# Export IO helpers
from xradio.testing.measurement_set.io import (
    download_measurement_set,
    convert_msv2_to_processing_set,
    build_msv4_partition,
    build_minimal_msv4_xdt,
)
