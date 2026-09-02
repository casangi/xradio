"""casacore-table backend selection.

XRADIO accesses CASA tables (MSv2 conversion, CASA image IO) through the
python-casacore ``tables``/``images``/``coordinates`` API surface. This
module is the single place where the implementation of that surface is
chosen:

* **arcae** (primary): :mod:`xradio._utils._casacore.casacore_from_arcae`
  and :mod:`xradio._utils._casacore.images_from_arcae` emulate the
  python-casacore API on top of `arcae <https://github.com/ska-sa/arcae>`_.
* **alternative backend**: set ``XRADIO_CASACORE_BACKEND=<dotted.module.path>``
  (e.g. ``casacore`` or its alias ``python-casacore``, ``xradio_casatools_backend``)
  to load any installed python-casacore-like backend.

Consumers must import ``tables`` / ``images`` / ``coordinates`` (and the
MS-creation helpers) from this module instead of importing a backend directly.
"""

import importlib
import os

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
    except ImportError:
        _backend = "python-casacore"  # arcae unavailable; fall through

if _backend != "arcae":
    try:
        _tables = importlib.import_module(f"{_backend}.tables")
        _images = importlib.import_module(f"{_backend}.images")
        _msutil = importlib.import_module(f"{_backend}.tables.msutil")
        _tableutil = importlib.import_module(f"{_backend}.tables.tableutil")
    except ImportError as e:
        raise ImportError(
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
