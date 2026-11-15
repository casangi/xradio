"""
Schema building blocks shared between the measurement set and image schemas.

This module holds the quantity and measure data array schemas, unit literals,
allowed value vocabularies and casacore translation tables that are common to
``xradio.measurement_set.schema`` and ``xradio.image.schema``.
The definitions originate from the measurement set v4 schema and are
re-exported there for backwards compatibility. The dimension literals moved
on to :py:mod:`xradio.schema.dimensions`; this module re-exports them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import KW_ONLY
from typing import Literal

from xradio.schema.bases import (
    xarray_dataarray_schema,
)

# Dimensions, defined in xradio.schema.dimensions and re-exported here (where
# earlier releases defined them)
from xradio.schema.dimensions import (  # noqa: F401
    ZD,
    CartesianPosLabel,
    EllipsoidDirLabel,
    EllipsoidDisLabel,
    Frequency,
    Polarization,
    SkyDirLabel,
    SkyDisLabel,
    Time,
)
from xradio.schema.typing import Attr, Data

# Types of quantity and measures
Quantity = Literal["quantity"]
SkyCoord = Literal["sky_coord"]
SpectralCoord = Literal["spectral_coord"]
Location = Literal["location"]
Doppler = Literal["doppler"]

# Units of quantities and measures
UnitsSeconds = Literal["s"]
UnitsHertz = Literal["Hz"]
UnitsMeters = Literal["m"]
UnitsRadians = Literal["rad"]
UnitsMetersPerSecond = Literal["m/s"]

UnitsOfSkyCoordInMetersOrRadians = Literal["m", "rad"]
UnitsOfLocationInMetersOrRadians = Literal[
    "m",
    "rad",
]
UnitsOfDopplerShift = Literal["ratio", "m/s"]


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class QuantityArrayBase:
    """
    Base of the quantity arrays: a value without dimensions and its units.
    Each quantity array schema derives from it and fixes the units.
    """

    data: Data[ZD, float]

    units: Attr[str]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInSecondsArray(QuantityArrayBase):
    """
    Quantity with units of seconds
    """

    units: Attr[UnitsSeconds]


@xarray_dataarray_schema
class QuantityInHertzArray(QuantityArrayBase):
    """
    Quantity with units of Hertz
    """

    units: Attr[UnitsHertz]


@xarray_dataarray_schema
class QuantityInMetersArray(QuantityArrayBase):
    """
    Quantity with units of meters
    """

    units: Attr[UnitsMeters]


@xarray_dataarray_schema
class QuantityInMetersPerSecondArray(QuantityArrayBase):
    """
    Quantity with units of meters per second
    """

    units: Attr[UnitsMetersPerSecond]


@xarray_dataarray_schema
class QuantityInRadiansArray(QuantityArrayBase):
    """
    Quantity with units of radians
    """

    units: Attr[UnitsRadians]


AllowedTimeScales = Literal["tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc"]


AllowedTimeFormats = Literal["unix", "mjd", "cxcsec", "gps"]

#: casacore ``MEpoch`` reference -> astropy time scale. Sidereal references
#: (LAST, LMST, GMST1, GAST) have no astropy scale and are absent.
CASACORE_EPOCH_REF_TO_SCALE = {
    "UTC": "utc",
    "TAI": "tai",
    "IAT": "tai",
    "TT": "tt",
    "TDT": "tt",
    "ET": "tt",
    "TDB": "tdb",
    "TCB": "tcb",
    "TCG": "tcg",
    "UT1": "ut1",
    "UT": "ut1",
}


@xarray_dataarray_schema
class TimeArray:
    """
    Representation of a time measure.

    :py:class:`astropy.time.Time` serves as the reference implementation.
    Data can be converted as follows::

        astropy.time.Time(data * astropy.units.Unit(attrs['units'][0]),
                          format=attrs['format'], scale=attrs['scale'])

    All formats that express time as floating point numbers since an epoch
    are permissible, so at present the realistic options are:

    * ``mjd`` (from 1858-11-17 00:00:00 UTC)
    * ``unix`` (from 1970-01-01 00:00:00 UTC)
    * ``unix_tai`` (from 1970-01-01 00:00:00 TAI)
    * ``cxcsec`` (from 1998-01-01 00:00:00 TT)
    * ``gps`` (from 1980-01-06 00:00:00 UTC)

    """

    data: Data[ZD, float]
    """Time since epoch, typically in seconds (see ``units``)."""

    type: Attr[Time] = "time"
    """ Array type. Should be ``"time"``. """
    units: Attr[UnitsSeconds] = "s"
    """ Units to associate with axis"""
    scale: Attr[AllowedTimeScales] = "utc"
    """
    Time scale of data. Must be one of ``('tai', 'tcb', 'tcg', 'tdb', 'tt', 'ut1', 'utc')``,
    see :py:class:`astropy.time.Time`
    """
    format: Attr[AllowedTimeFormats] = "unix"
    """Time representation and epoch, see :py:class:`~xradio.schema.measures.TimeArray`."""


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class TimeSamplingArrayBase:
    """
    Base of the data variables that hold time sampling data, such as the time
    centroids of the measurement set: the time measure attributes (see
    :py:class:`TimeArray`). Each schema adds the data and coordinates.
    """

    _: KW_ONLY

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """

    long_name: Attr[str] | None = "Time sampling data"
    units: Attr[UnitsSeconds] = "s"


# Taken from the list of astropy built-in frame classes:
# https://docs.astropy.org/en/stable/coordinates/index.html
AllowedSkyCoordFrames = Literal[
    "icrs",
    "fk5",
    "fk4",
    "fk4noterms",
    "galactic",
    "galactocentric",
    "supergalactic",
    "altaz",
    "hadec",
    "gcrs",
    "cirs",
    "itrs",
    "hcrs",
    "teme",
    "tete",
    "precessedgeocentric",
    "geocentricmeanecliptic",
    "barycentricmeanecliptic",
    "heliocentricmeanecliptic",
    "geocentrictrueecliptic",
    "barycentrictrueecliptic",
    "heliocentrictrueecliptic",
    "heliocentriceclipticiau76",
    "custombarycentricecliptic",
    "lsr",
    "lsrk",
    "lsrd",
    "galacticlsr",
]

#: casacore ``MDirection`` reference -> astropy frame, as the measurement set
#: converters translate them (see :py:class:`SkyCoordArray`).
CASACORE_DIRECTION_TO_FRAME = {
    "AZELGEO": "altaz",
    "ICRS": "icrs",
    "J2000": "fk5",
}

#: Equinox of the equatorial frames that need one, in the form the image
#: schema stores it (the equinox of casacore's J2000 and B1950 directions).
DEFAULT_EQUINOX = {"fk5": "j2000.0", "fk4": "b1950.0"}


@xarray_dataarray_schema
class SkyCoordArray:
    """Measures array for data variables that are sky coordinates, used in
    :py:class:`~xradio.measurement_set.schema.FieldSourceXds`"""

    data: Data[SkyDirLabel | SkyDisLabel, float]
    units: Attr[UnitsOfSkyCoordInMetersOrRadians]
    type: Attr[SkyCoord] = "sky_coord"
    frame: Attr[AllowedSkyCoordFrames] = "icrs"
    """
    Possible values are :py:class:`astropy.coordinates.SkyCoord` frames.

    Several casacore frames found in MSv2 are translated to ``astropy`` frames as follows:

    * ``AZELGEO`` => ``altaz``
    * ``J2000`` => ``fk5``
    * ``ICRS`` => ``icrs``

    From ``fixvis`` docs: ``clean`` and the ``im`` tool ignore the reference frame claimed
    by the UVW column (it is often mislabelled as ITRF when it is really FK5
    or J2000) and instead assume the (u, v, w)s are in the same frame as the phase
    tracking center. ``calcuvw`` does not yet force the UVW column and field centers
    to use the same reference frame!
    """


# Allowing both the casacore frames that are not translated to astropy (all
# in uppercase, casacore spelling) and the astropy frames (all in lowercase,
# taken from the list of SpectralCoord:
# https://docs.astropy.org/en/stable/coordinates/spectralcoord.html). Every
# casacore MFrequency frame maps to exactly one value: LSRK -> lsrk,
# LSRD -> lsrd and GEO -> gcrs; the others keep their casacore name.
AllowedSpectralCoordFrames = Literal[
    # casacore frames, not translated to astropy
    "REST",
    "BARY",
    "TOPO",
    "GALACTO",
    "LGROUP",
    "CMB",
    # astropy frames
    "gcrs",
    "icrs",
    "hcrs",
    "lsrk",
    "lsrd",
    "lsr",
]
"""
Spectral reference frames (``observer`` of spectral coordinates). The
lowercase values are astropy frame names (``gcrs``, ``icrs``, ``hcrs``,
``lsrk``, ``lsrd``, ``lsr``); the uppercase values are casacore frames that
are not translated to astropy and keep their casacore name (``REST``,
``BARY``, ``TOPO``, ``GALACTO``, ``LGROUP``, ``CMB``). The remaining casacore
frames are translated: ``LSRK`` is ``lsrk``, ``LSRD`` is ``lsrd`` and ``GEO``
is ``gcrs``. ``icrs``, ``hcrs`` and ``lsr`` have no casacore equivalent in
this vocabulary (astropy's ``lsr`` uses a different solar motion than
casacore's LSRK), so the image writers, which name spectral frames like
casacore and FITS do, cannot write images in these frames.
"""

#: casacore ``MFrequency`` reference frames, the canonical names held by the
#: ``frame`` attribute of the image frequency coordinate.
CASACORE_SPECTRAL_FRAMES = (
    "REST",
    "LSRK",
    "LSRD",
    "BARY",
    "GEO",
    "TOPO",
    "GALACTO",
    "LGROUP",
    "CMB",
)

#: casacore frame -> ``observer`` of spectral coordinates, following the shared
#: ``AllowedSpectralCoordFrames`` vocabulary: frames with an astropy
#: equivalent use the lowercase astropy name, the others keep the casacore name.
CASACORE_TO_OBSERVER = {
    "REST": "REST",
    "LSRK": "lsrk",
    "LSRD": "lsrd",
    "BARY": "BARY",
    "GEO": "gcrs",
    "TOPO": "TOPO",
    "GALACTO": "GALACTO",
    "LGROUP": "LGROUP",
    "CMB": "CMB",
}


@xarray_dataarray_schema
class SpectralCoordArray:
    """
    Measures array for data variables and attributes that are spectral coordinates.
    """

    data: Data[ZD, float]

    units: Attr[UnitsHertz] = "Hz"

    observer: Attr[AllowedSpectralCoordFrames] = "icrs"
    """
    Capitalized reference observers are from casacore. TOPO implies creating astropy earth_location.
    Astropy velocity reference frames are lowercase. Note that Astropy does not use the name 'TOPO' (telescope centric)
    rather it assumes if no velocity frame is given that this is the default.

    When converting from MSv2 and casacore frequency frames, the following translations from casacore to astropy
    frame names are applied: GEO=>gcrs, LSRK=>lsrk, LSRD=>lsrd. The other casacore frames (REST, BARY, TOPO,
    GALACTO, LGROUP, CMB) are not translated and keep their casacore name. The astropy frames icrs, hcrs and
    lsr are not translated to casacore frames (astropy's lsr uses a different solar motion than casacore's LSRK).
    """

    type: Attr[SpectralCoord] = "spectral_coord"


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class FrequencySamplingArrayBase:
    """
    Base of the data variables that hold frequency sampling data, such as the
    frequency centroids and effective channel widths of the measurement set:
    the spectral measure attributes (see :py:class:`SpectralCoordArray`).
    Each schema adds the data and coordinates.
    """

    _: KW_ONLY

    long_name: Attr[str] | None = "Frequency sampling data"
    units: Attr[UnitsHertz] = "Hz"
    observer: Attr[AllowedSpectralCoordFrames] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """


AllowedLocationFrames = Literal["ITRS", "Undefined"]


AllowedLocationCoordinateSystems = Literal[
    "geocentric",
    "planetcentric",
    "geodetic",
    "planetodetic",
    "orbital",
    "topocentric",
]


AllowedEllipsoid = Literal["GRS80", "WGS84", "WGS72"]


@xarray_dataarray_schema
class LocationArray:
    """
    Measure type used for example in antenna_xds/ANTENNA_POSITION, weather_xds/STATION_POSITION,
    field_and_source_xds(ephemeris)/OBSERVER_POSITION.

    Data dimensions can be CartesianPosLabel or EllipsoidDirLabel or EllipsoidDisLabel
    """

    data: Data[EllipsoidDirLabel | EllipsoidDisLabel | CartesianPosLabel, float]

    units: Attr[UnitsOfLocationInMetersOrRadians]
    """
    Units of the location coordinates (typically 'm' or 'rad').
    """

    frame: Attr[AllowedLocationFrames]
    """
    Reference frame. Can be ITRS (assumed for all Earth locations) or Undefined (used in non-Earth locations).
    """

    coordinate_system: Attr[AllowedLocationCoordinateSystems]
    """ Can be ``geocentric/planetcentric, geodetic/planetodetic, orbital`` """

    origin_object_name: Attr[str]
    """
    earth/sun/moon/etc.
    """

    ellipsoid: Attr[AllowedEllipsoid] | None
    """
    Ellipsoid used in geodetic Earth locations (with EllipsoidDirLabel and EllipsoidDisLabel coordinates)
    """

    type: Attr[Location] = "location"
    """ Measure type. Should be ``"location"``."""


AllowedDopplerTypes = Literal[
    "radio", "optical", "z", "ratio", "true", "relativistic", "beta", "gamma"
]


@xarray_dataarray_schema
class DopplerArray:
    """Doppler measure information for the frequency coordinate"""

    data: Data[ZD, float]

    type: Attr[Doppler] = "doppler"
    """ Coordinate type. Should be ``"doppler"``. """

    units: Attr[UnitsOfDopplerShift] = "m/s"
    """ Units to associate with axis, [ratio]/[m/s]"""

    doppler_type: Attr[AllowedDopplerTypes] = "radio"
    """
    Allowable values: radio, optical, z, ratio, true, relativistic, beta, gamma.
    Astropy only has radio and optical. Using casacore types: https://casadocs.readthedocs.io/en/stable/notebooks/memo-series.html?highlight=Spectral%20Frames#Spectral-Frames
    """


@xarray_dataarray_schema
class PolarizationArray:
    """
    Possible correlations that can be formed from polarised receptors. Possible
    values, taken from `Measures/Stokes.h
    <https://github.com/casacore/casacore/blob/5a8df94738bdc36be27e695d7b14fe949a1cc2df/measures/Measures/Stokes.h>`_:

    * ``I``, ``Q``, ``U``, ``V`` (standard stokes parameters)
    * ``RR``, ``RL``, ``LR``, ``LL`` (circular correlation products)
    * ``XX``, ``XY``, ``YX``, ``YY`` (linear correlation products)
    * ``RX``, ``RY``, ``LX``, ``LY``, ``XR``, ``XL``, ``YR``, ``YL`` (mixed correlation products)
    * ``PP``, ``PQ``, ``QP``, ``QQ`` (general quasi-orthogonal correlation products)
    * ``RCircular``, ``LCircular``, ``Linear`` (single dish polarization types)
    * ``Ptotal`` (polarized intensity: ``sqrt(Q²+U²+V²)``)
    * ``Plinear`` (linearly polarized intensity: ``sqrt(Q²+U²)``)
    * ``PFtotal`` (polarization fraction: ``Ptotal/I``)
    * ``PFlinear`` (linear polarization fraction: ``Plinear/I``)
    * ``Pangle`` (linear polarization angle: ``0.5 arctan(U/Q)`` in radians)

    """

    data: Data[Polarization, str]
    """ Polarization names. """
    long_name: Attr[str] | None = "Polarization"
    """ Long-form name to use for axis. Should be ``"Polarization"``"""
