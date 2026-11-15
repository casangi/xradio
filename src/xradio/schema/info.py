"""
Information bases shared by the xradio schemas.

The classes here hold the dictionaries, dataset attributes and data variable
attributes that the dataset schemas have in common: who created a dataset,
the data groups, the schema version, and the attributes of the data
variables that can appear in several versions, flags among them. They are
bases, not schemas: a schema derives its own decorated class from a base,
keeping its class name, and adds or redeclares what is specific to it. The
field docstrings of the bases that the measurement set v4 schema derives
from are its text.

The measurement set v4 schema derives its creator dictionary and its
versioned and flag data variables from these bases. Its data group
dictionary and its main datasets (``VisibilityXds`` and ``SpectrumXds``)
declare ``description``, ``date`` and ``schema_version`` themselves: base
fields come first, so deriving them would change the order of their
constructor parameters, keys and attributes.

The attribute-only array bases come before the ``data`` field of the arrays
that derive from them, so their attributes follow a ``KW_ONLY`` marker:
``data`` stays the first positional constructor parameter.
"""

from __future__ import annotations

import dataclasses
from dataclasses import KW_ONLY

from xradio.schema.typing import Attr


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class CreatorDictBase:
    """Base of the creator dictionaries: the software that created a dataset."""

    software_name: str
    """ Software that created the Measurement Set (XRadio, etc.). """
    version: str
    """ Version of the software. """


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class GenericDataGroupDict:
    """Base for data group dictionaries: the keys of a data group besides the
    names of its data variables."""

    description: str | None
    """ More details about the data group. """
    date: str | None
    """ Creation date-time, in ISO 8601 format: 'YYYY-MM-DDTHH:mm:ss.SSS'. """


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class SchemaVersionAttrs:
    """Base for dataset schemas: the version of the schema that a dataset
    conforms to."""

    schema_version: Attr[str]
    """Semantic version (MAJOR.MINOR.PATCH) of the schema that the dataset
    conforms to."""


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class VersionedArrayBase:
    """Base of the data variables that can appear in several versions, named
    by their canonical name followed by an underscore separated suffix (for
    example ``VISIBILITY`` and ``VISIBILITY_MODEL``)."""

    _: KW_ONLY

    allow_multiple_versions: Attr[bool] | None = True


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class FlagArrayBase(VersionedArrayBase):
    """Base of the flag data variables of the measurement set v4 schema, whose
    flags may be bit masks."""

    _: KW_ONLY

    flag_bits: Attr[list[str]] | None = (
        "UNSPECIFIED_BIT",
        "STATIC_BIT",
        "CAM_BIT",
        "DATA_LOST_BIT",
        "INGEST_RFI_BIT",
        "PREDICTED_RFI_BIT",
        "CAL_RFI_BIT",
        "POSTPROC_BIT",
    )
    """
    Labels associated with bits used in ``FlagArray`` data. See
    :ref:`flag bits` for suggested semantics.
    """
