"""casacore-table backend selection.

XRADIO accesses CASA tables (MSv2 conversion, CASA image IO) through the
python-casacore ``tables``/``images``/``coordinates`` API surface, which
:mod:`xradio._utils._casacore.casacore_from_arcae` and
:mod:`xradio._utils._casacore.images_from_arcae` emulate on top of
`arcae <https://github.com/ska-sa/arcae>`_. This module is the single place
where the implementation of that surface is chosen.

Consumers must import ``tables`` / ``images`` / ``coordinates`` (and the
MS-creation helpers) from this module instead of importing a backend directly.
"""

import importlib
import os

# Unsupported testing hook: XRADIO_CASACORE_BACKEND=casacore selects python-casacore,
# XRADIO_CASACORE_BACKEND=casatools the casatools shim.
_backend = os.environ.get("XRADIO_CASACORE_BACKEND", "arcae")
# Alias python-casacore -> casacore for consistent submodule imports.
_backend = "casacore" if _backend == "python-casacore" else _backend

if _backend == "arcae":
    try:
        from xradio._utils._casacore import casacore_from_arcae as tables
        from xradio._utils._casacore import images_from_arcae as coordinates
        from xradio._utils._casacore import images_from_arcae as images
        from xradio._utils._casacore.casacore_from_arcae import (
            complete_ms_desc,
            default_ms,
            default_ms_subtable,
            makearrcoldesc,
            makedminfo,
            makescacoldesc,
            maketabdesc,
            required_ms_desc,
        )

        BACKEND = "arcae"
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] not in ("arcae", "pyarrow"):
            raise
        raise ModuleNotFoundError(f"arcae cannot be imported: {e}") from e
elif _backend == "casatools":
    from xradio._utils._casacore import casacore_from_casatools as tables

    images = coordinates = tables
    BACKEND = "casatools"
elif _backend == "casacore":
    try:
        _tables = importlib.import_module(f"{_backend}.tables")
        _images = importlib.import_module(f"{_backend}.images")
        _msutil = importlib.import_module(f"{_backend}.tables.msutil")
        _tableutil = importlib.import_module(f"{_backend}.tables.tableutil")
    except ImportError as e:
        raise ModuleNotFoundError(
            f"Backend '{_backend}' is not properly installed or does not "
            f"expose the required casacore-compatible submodules."
        ) from e

    tables = _tables
    images = _images
    coordinates = _images.coordinates
    default_ms = _tables.default_ms
    default_ms_subtable = _tables.default_ms_subtable
    complete_ms_desc = _msutil.complete_ms_desc
    required_ms_desc = _msutil.required_ms_desc
    makearrcoldesc = _tableutil.makearrcoldesc
    makedminfo = _tableutil.makedminfo
    makescacoldesc = _tableutil.makescacoldesc
    maketabdesc = _tableutil.maketabdesc
    BACKEND = _backend
else:
    raise ValueError(
        f"XRADIO_CASACORE_BACKEND={_backend!r}: expected arcae, casacore or casatools"
    )

__all__ = [
    "BACKEND",
    "tables",
    "images",
    "coordinates",
    "default_ms",
    "default_ms_subtable",
    "complete_ms_desc",
    "required_ms_desc",
    "makescacoldesc",
    "makearrcoldesc",
    "maketabdesc",
    "makedminfo",
]
