"""
Processing Set and Measurement Set v4 API. Includes functions and classes to open, load,
convert, and retrieve information from Processing Set and Measurement Sets nodes of the
Processing Set DataTree
"""

import toolviper.utils.logger as _logger

from .processing_set_xdt import ProcessingSetXdt
from .open_processing_set import open_processing_set
from .load_processing_set import load_processing_set
from .measurement_set_xdt import MeasurementSetXdt
from .schema import SpectrumXds, VisibilityXds

__all__ = [
    "ProcessingSetXdt",
    "MeasurementSetXdt",
    "open_processing_set",
    "load_processing_set",
    "SpectrumXds",
    "VisibilityXds",
]

try:
    from .convert_msv2_to_processing_set import (
        convert_msv2_to_processing_set,
        estimate_conversion_memory_and_cores,
    )
except ModuleNotFoundError as exc:
    _logger.warning(
        "Could not import the function to convert from MSv2 to MSv4. "
        f"That functionality will not be available. Details: {exc}"
    )
else:
    __all__.extend(
        ["convert_msv2_to_processing_set", "estimate_conversion_memory_and_cores"]
    )
