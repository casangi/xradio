"""casacoretables-backed replacements for casacore's MeasurementSet helpers.

``casacoretables`` deliberately ships only casacore's *table* layer, not the
MeasurementSet (``ms``) module, so the helpers python-casacore exposes for
building MSs -- ``default_ms``, ``default_ms_subtable``, ``required_ms_desc``
and ``complete_ms_desc`` -- are not available there.

xradio only needs these helpers to *generate* synthetic MSv2 datasets for the
test-suite (see ``msv2_io.py``). Those helpers are entirely descriptor-driven:
``required_ms_desc`` / ``complete_ms_desc`` return the standard MS table
descriptors, and ``default_ms`` just creates the MAIN table plus the standard
empty subtables from those descriptors and links them with ``Table:`` keywords.

So we vendor the descriptors (captured once from python-casacore into
``_ms_descriptors.json``) and reimplement the four helpers on top of
``casacoretables``. The descriptors carry every column's type, shape, and
keywords (QuantumUnits / MEASINFO), so the generated tables are structurally
identical to what python-casacore's ``default_ms`` produced.
"""

import copy
import json
import os
import shutil
from importlib.resources import files

from casacoretables import tables
from casacoretables.tables import makedminfo

# Subtables that casacore's default_ms creates and links into the MAIN table.
_STANDARD_SUBTABLES = [
    "ANTENNA",
    "DATA_DESCRIPTION",
    "FEED",
    "FLAG_CMD",
    "FIELD",
    "HISTORY",
    "OBSERVATION",
    "POINTING",
    "POLARIZATION",
    "PROCESSOR",
    "SPECTRAL_WINDOW",
    "STATE",
]

_MAIN_INFO = {
    "type": "Measurement Set",
    "subType": "",
    "readme": "This is a MeasurementSet Table holding measurements from a Telescope\n",
}


def _load_descriptors() -> dict:
    text = (files(__package__) / "_ms_descriptors.json").read_text()
    return json.loads(text)


_DESCRIPTORS = _load_descriptors()


def required_ms_desc(table: str = "MAIN") -> dict:
    """Return the *required*-columns table descriptor for an MS (sub)table."""
    return copy.deepcopy(_DESCRIPTORS["required"][table or "MAIN"])


def complete_ms_desc(table: str = "MAIN") -> dict:
    """Return the *complete*-columns table descriptor for an MS (sub)table."""
    return copy.deepcopy(_DESCRIPTORS["complete"][table or "MAIN"])


def _make_table(name: str, tabdesc: dict, dminfo: dict = None):
    if not dminfo:
        dminfo = makedminfo(tabdesc)
    return tables.table(str(name), tabdesc, nrow=0, dminfo=dminfo, ack=False)


def default_ms(name, tabdesc: dict = None, dminfo: dict = None):
    """Create an empty MeasurementSet (MAIN + standard subtables) on disk.

    Mirrors ``casacore.tables.default_ms``: returns the open MAIN table (usable
    as a context manager). ``tabdesc`` defaults to the required MAIN descriptor.
    """
    name = str(name)
    if tabdesc is None:
        tabdesc = required_ms_desc("MAIN")
    shutil.rmtree(name, ignore_errors=True)

    main = _make_table(name, tabdesc, dminfo)
    main.putinfo(_MAIN_INFO)
    main.putkeyword("MS_VERSION", 2.0)

    for sub in _STANDARD_SUBTABLES:
        sub_path = os.path.join(name, sub)
        # Standard subtables are created with their *required* columns only,
        # matching casacore's default_ms (optional columns such as FIELD's
        # EPHEMERIS_ID are added later by the caller).
        sub_desc = required_ms_desc(sub)
        sub_tb = _make_table(sub_path, sub_desc)
        sub_tb.close()
        main.putkeyword(sub, f"Table: {sub_path}")

    main.flush()
    return main


def default_ms_subtable(
    subtable: str, name=None, tabdesc: dict = None, dminfo: dict = None
):
    """Create a single empty MS subtable on disk and return the open table.

    Mirrors ``casacore.tables.default_ms_subtable``. ``tabdesc`` defaults to the
    *complete* descriptor for ``subtable``. The caller is responsible for linking
    it into a parent MS (as ``msv2_io`` does).
    """
    if name is None or name == "":
        name = subtable
    if tabdesc is None:
        tabdesc = complete_ms_desc(subtable)
    name = str(name)
    shutil.rmtree(name, ignore_errors=True)
    return _make_table(name, tabdesc, dminfo)
