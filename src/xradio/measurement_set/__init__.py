"""
Processing Set and Measurement Set v4 API. Includes functions and classes to open, load,
convert, and retrieve information from Processing Set and Measurement Sets nodes of the
Processing Set DataTree
"""

import importlib.util as _importlib_util

from xradio.measurement_set.load_processing_set import load_processing_set
from xradio.measurement_set.measurement_set_xdt import MeasurementSetXdt
from xradio.measurement_set.open_processing_set import open_processing_set
from xradio.measurement_set.processing_set_xdt import ProcessingSetXdt
from xradio.measurement_set.schema import SpectrumXds, VisibilityXds

__all__ = [
    "ProcessingSetXdt",
    "MeasurementSetXdt",
    "open_processing_set",
    "load_processing_set",
    "SpectrumXds",
    "VisibilityXds",
]

try:
    from xradio.measurement_set.convert_msv2_to_processing_set import (
        convert_msv2_to_processing_set as convert_msv2_to_processing_set,
    )
    from xradio.measurement_set.convert_msv2_to_processing_set import (
        estimate_conversion_memory_and_cores as estimate_conversion_memory_and_cores,
    )
except ModuleNotFoundError:
    # Optional MSv2-conversion backend not installed; the conversion functions
    # are simply not exported.
    pass
else:
    __all__.extend(
        ["convert_msv2_to_processing_set", "estimate_conversion_memory_and_cores"]
    )


def _asdm_backend_available() -> bool:
    """Check (without importing it) whether pyasdm, needed by open_asdm, is installed."""
    try:
        return _importlib_util.find_spec("pyasdm") is not None
    except (ImportError, ValueError):
        return False


# open_asdm is exported lazily (PEP 562 module __getattr__) so that importing
# xradio.measurement_set does not import pyasdm (the optional ASDM backend).
_LAZY_ASDM_EXPORTS = ("open_asdm",)

if _asdm_backend_available():
    __all__.extend(_LAZY_ASDM_EXPORTS)


def __getattr__(name: str):
    """
    Import the optional ASDM backend function(s) on first access.

    Parameters
    ----------
    name : str
        Name of the attribute requested from the module.

    Returns
    -------
    Callable
        The requested function (``open_asdm``).

    Raises
    ------
    ModuleNotFoundError
        If ``open_asdm`` is requested but the optional ASDM dependencies
        (``pip install "xradio[asdm]"``) are not installed.
    AttributeError
        If the module has no attribute ``name``.
    """
    if name in _LAZY_ASDM_EXPORTS:
        try:
            from xradio.measurement_set._utils._asdm.open_asdm import open_asdm
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.split(".")[0] == "xradio":
                raise
            raise ModuleNotFoundError(
                f"xradio.measurement_set.{name} requires the optional ASDM backend "
                f"dependencies, but the module {exc.name!r} could not be imported. "
                'Install them with: pip install "xradio[asdm]"',
                name=exc.name,
            ) from exc

        # Cache it in the module namespace so __getattr__ is not called again.
        globals()[name] = open_asdm
        return open_asdm

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the module attributes, including the lazily imported ones."""
    return sorted(set(globals()) | set(__all__))
