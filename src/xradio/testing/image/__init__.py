"""
Test utilities for xradio image functionality.

Usable in pytest, ASV benchmarks, and any other framework – no pytest
dependency is introduced by importing this package.

Examples
--------
>>> from xradio.testing.image import (
...     download_image,
...     download_and_open_image,
...     remove_path,
...     make_beam_fit_params,
...     create_empty_test_image,
...     create_bzero_bscale_fits,
...     scale_data_for_int16,
...     normalize_image_coords_for_compare,
...     assert_image_block_equal,
... )
"""

__all__ = [
    # IO helpers
    "download_image",
    "download_and_open_image",
    "remove_path",
    # Generators
    "make_beam_fit_params",
    "create_empty_test_image",
    "create_bzero_bscale_fits",
    "scale_data_for_int16",
    # Assertions / comparators
    "normalize_image_coords_for_compare",
    "assert_image_block_equal",
]

from xradio.testing.image.assertions import (
    assert_image_block_equal,
    normalize_image_coords_for_compare,
)
from xradio.testing.image.generators import (
    create_bzero_bscale_fits,
    create_empty_test_image,
    make_beam_fit_params,
    scale_data_for_int16,
)
from xradio.testing.image.io import (
    download_and_open_image,
    download_image,
    remove_path,
)
