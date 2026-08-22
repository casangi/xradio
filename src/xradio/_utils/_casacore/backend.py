"""
casacore-table backend selection.

XRADIO accesses CASA tables (MSv2 conversion, CASA image IO) through the
python-casacore ``tables``/``images``/``coordinates`` API surface. This
module is the single place where the implementation of that surface is
chosen:

* **arcae** (primary): :mod:`xradio._utils._casacore.casacore_from_arcae`
  and :mod:`xradio._utils._casacore.images_from_arcae` emulate the
  python-casacore API on top of `arcae <https://github.com/ska-sa/arcae>`_.
* **python-casacore** (fallback): used only when arcae is not installed.
  This fallback is intentionally undocumented and is not installed by any
  XRADIO extra; it exists as a backup implementation only.

To swap in another backend, provide modules implementing the same API
subset and add them to the selection below. Consumers must import
``tables`` / ``images`` / ``coordinates`` (and the MS-creation helpers)
from this module instead of importing a backend directly.
"""

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
    # Undocumented backup backend: python-casacore, if installed.
    from casacore import (
        images,  # noqa: F401
        tables,  # noqa: F401
    )
    from casacore.images import coordinates  # noqa: F401
    from casacore.tables import (  # noqa: F401
        default_ms,
        default_ms_subtable,
    )
    from casacore.tables.msutil import (  # noqa: F401
        complete_ms_desc,
        required_ms_desc,
    )
    from casacore.tables.tableutil import (  # noqa: F401
        makearrcoldesc,
        makedminfo,
        makescacoldesc,
        maketabdesc,
    )

    BACKEND = "python-casacore"

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
