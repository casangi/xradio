"""
Schema building blocks shared between the measurement set and image schemas.

This module holds the dimension literals, quantity and measure data array
schemas, unit literals and allowed value vocabularies that are common to
``xradio.measurement_set.schema`` and ``xradio.image.schema``.
The definitions originate from the measurement set v4 schema and are
re-exported there for backwards compatibility.
"""

from __future__ import annotations

from typing import Literal

from xradio.schema.bases import (
    xarray_dataarray_schema,
)
from xradio.schema.typing import Attr, Data

# Dimensions
Time = Literal["time"]
""" Observation time dimension """
Frequency = Literal["frequency"]
""" Frequency dimension """
Polarization = Literal["polarization"]
""" Polarization dimension """
SkyDirLabel = Literal["sky_dir_label"]
""" Coordinate labels of sky directions (typically shape 2 and 'ra', 'dec') """
SkyDisLabel = Literal["sky_dis_label"]
""" Coordinate labels of sky distance (typically shape 1 and 'dist') """
EllipsoidDirLabel = Literal["ellipsoid_dir_label"]
""" Coordinate labels of geodetic earth location data (typically shape 3 and 'lon', 'lat', 'height')"""
EllipsoidDisLabel = Literal["ellipsoid_dis_label"]
""" Coordinate label of geodetic earth height (typically shape 1 and 'dist')"""
CartesianPosLabel = Literal["cartesian_pos_label"]
""" Coordinate labels of geocentric earth location data (typically shape 3 and 'x', 'y', 'z')"""

# Represents "no dimension", i.e. used for coordinates and data variables with
# zero dimensions.
ZD = tuple[()]


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


@xarray_dataarray_schema
class QuantityInSecondsArray:
    """
    Quantity with units of seconds
    """

    data: Data[ZD, float]

    units: Attr[UnitsSeconds]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInHertzArray:
    """
    Quantity with units of Hertz
    """

    data: Data[ZD, float]

    units: Attr[UnitsHertz]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInMetersArray:
    """
    Quantity with units of meters
    """

    data: Data[ZD, float]

    units: Attr[UnitsMeters]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInMetersPerSecondArray:
    """
    Quantity with units of meters per second
    """

    data: Data[ZD, float]

    units: Attr[UnitsMetersPerSecond]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInRadiansArray:
    """
    Quantity with units of radians
    """

    data: Data[ZD, float]

    units: Attr[UnitsRadians]
    type: Attr[Quantity] = "quantity"


AllowedTimeScales = Literal["tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc"]


AllowedTimeFormats = Literal["unix", "mjd", "cxcsec", "gps"]


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


# For now allowing both some of the casacore frames (from "REST" to "TOPO" -
# all in uppercase) as well as the astropy frames (all in lowercase, taken
# from the list of SpectralCoord:
# https://docs.astropy.org/en/stable/coordinates/spectralcoord.html)
AllowedSpectralCoordFrames = Literal[
    "REST",
    # "LSRK" -> "lsrk",
    # "LSRD" -> "lsrd",
    "BARY",
    # "GEO", -> "gcrs"
    "TOPO",
    # astropy frames
    "gcrs",
    "icrs",
    "hcrs",
    "lsrk",
    "lsrd",
    "lsr",
]


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
    frame names are applied: GEO=>gcrs, LSRK=>lsrk, LSRD=>lsrd
    """

    type: Attr[SpectralCoord] = "spectral_coord"


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
    Ellipsoid used in geodetic Earth locations (with EllipsoidDirLabel and EllipsoidDirLabel coordinate)
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
