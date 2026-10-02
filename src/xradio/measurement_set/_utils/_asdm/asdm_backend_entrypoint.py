"""
Backward compatible location of the ``xradio_asdm`` xarray backend entry point.

The entry point class lives in the lightweight module
:mod:`xradio._asdm_backend_entrypoint` (registered in pyproject.toml), so that
xarray can load it without importing xradio.measurement_set or pyasdm.
"""

from xradio._asdm_backend_entrypoint import ASDMBackendEntryPoint

__all__ = ["ASDMBackendEntryPoint"]
