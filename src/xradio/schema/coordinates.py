"""
Coordinate bases shared by the xradio schemas.

The classes here hold what the coordinates of the dataset schemas have in
common. They are bases, not schemas: a schema derives its own decorated
coordinate class from a base, keeping its class name, and adds or redeclares
what is specific to it (for example
:py:class:`xradio.measurement_set.schema.FrequencyArray`). Their field
docstrings are those of the measurement set v4 schema.

A derived class lists the fields of its base first. The bases are frozen
dataclasses, as the schema decorators make, with ``data`` first. Where a
derived class adds attributes without a default, the defaulted attributes of
the base follow a ``KW_ONLY`` marker, and the derived class puts the base
attributes it redeclares after its own ``KW_ONLY`` marker.
"""

from __future__ import annotations

import dataclasses
from dataclasses import KW_ONLY

import numpy

from xradio.schema.dimensions import (
    AntennaName,
    BaselineId,
    Frequency,
    ReceptorLabel,
    Time,
)
from xradio.schema.measures import (
    AllowedSpectralCoordFrames,
    AllowedTimeFormats,
    AllowedTimeScales,
    QuantityInHertzArray,
    QuantityInSecondsArray,
    SpectralCoord,
    UnitsHertz,
    UnitsSeconds,
)
from xradio.schema.typing import Attr, Data


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class TimeCoordArrayBase:
    """Base of the time coordinates: times since an epoch, see also
    :py:class:`~xradio.schema.measures.TimeArray`."""

    data: Data[Time, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``), see also see :py:class:`~xradio.schema.measures.TimeArray`.
    """

    type: Attr[Time] = "time"
    """ Coordinate type. Should be ``"time"``. """

    units: Attr[UnitsSeconds] = "s"
    """ Units to associate with axis"""

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`~xradio.schema.measures.TimeArray` """

    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`~xradio.schema.measures.TimeArray`"""

    integration_time: Attr[QuantityInSecondsArray] | None = None
    """ The nominal sampling interval (ms v2). Units of seconds. """


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class FrequencyCoordArrayBase:
    """Base of the frequency coordinates: the center frequency of each
    channel, with its spectral reference frame."""

    data: Data[Frequency, float]
    """ Center frequencies for each channel. """

    _: KW_ONLY

    type: Attr[SpectralCoord] = "spectral_coord"
    """ Coordinate type. Should be ``"spectral_coord"``. """
    long_name: Attr[str] | None = "Frequency"
    """ Long-form name to use for axis"""
    units: Attr[UnitsHertz] = "Hz"
    """ Units to associate with axis"""
    observer: Attr[AllowedSpectralCoordFrames] | None = None
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """
    # Not SpectralCoord, as it is a difference
    channel_width: Attr[QuantityInHertzArray] | None = None
    """ The nominal channel bandwidth. Same units as data array (see units key). """


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class AntennaNameArrayBase:
    """Base of the antenna_name coordinates."""

    data: Data[AntennaName, str]
    """Unique name for each antenna(_station)."""
    long_name: Attr[str] | None = "Antenna name"


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class BaselineArrayBase:
    """Base of the baseline_id coordinates."""

    data: Data[BaselineId, numpy.int64 | numpy.int32]
    """Unique id for each baseline."""
    long_name: Attr[str] | None = "Baseline ID"


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class BaselineAntennaNameArrayBase:
    """Base of the coordinates that give an antenna name for each baseline_id."""

    data: Data[BaselineId, str]
    """Unique id for each baseline."""
    long_name: Attr[str] | None = "Antenna name by baseline_id"


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class ScanArrayBase:
    """Base of the scan name coordinates."""

    data: Data[Time, str]
    """Scan name for each time sample."""
    scan_intents: Attr[list[str]]
    """ An intent string identifies one intention of the scan, such as to calibrate or observe a
    target. See :ref:`scan intents` for possible intent/subintent values. When converting from MSv2,
    the list of intents is derived from the OBS_MODE column of MSv2 state table (every comma
    separated value is taken as an intent).
    A common convention used in the MSv2 OBS_MODE column is to specify multiple intents separated
    by commas, each of them giving a main intent and a subintent separated by a '#' character. This
    is represented in this attribute as a list of "intent#subintent" strings. These are a few
    example lists:
    ["CALIBRATE_DELAY#ON_SOURCE" , "CALIBRATE_PHASE#ON_SOURCE", "CALIBRATE_WVR#ON_SOURCE"],
    ["CALIBRATE_FLUX#ON_SOURCE" , "CALIBRATE_WVR#ON_SOURCE"],
    ["CALIBRATE_POINTING#ON_SOURCE", "CALIBRATE_WVR#ON_SOURCE", "CALIBRATE_DELAY#ON_SOURCE"],
    ["CALIBRATE_ATMOSPHERE#AMBIENT", "CALIBRATE_WVR#AMBIENT"],
    ["CALIBRATE_FOCUS#ON_SOURCE" , "CALIBRATE_WVR#ON_SOURCE"],
    ["OBSERVE_TARGET#ON_SOURCE"], or ["OBSERVE_TARGE#UNSPECIFIED"].
    The list of possible intent and subintent names (see :ref:`scan intents`) is derived from the
    respective ASDM enumerations.
    """

    long_name: Attr[str] | None = "Scan name"
    """ Long-form name to use for axis. Should be ``"Scan name"``."""


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class ReceptorLabelArrayBase:
    """Base of the receptor_label coordinates: the names of the receptors of
    an antenna."""

    data: Data[ReceptorLabel, str]
