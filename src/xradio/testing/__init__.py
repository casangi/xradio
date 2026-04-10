from xradio.testing.assertions import (
    assert_attrs_dicts_equal,
    assert_xarray_datasets_equal,
)

__all__ = [
    "assert_attrs_dicts_equal",
    "assert_xarray_datasets_equal",
    # image sub-package (imported by name so external projects can use it)
    "image",
]

import xradio.testing.image  # noqa: F401, E402  – register the sub-package
