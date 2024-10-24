from __future__ import annotations

from typing import Literal, Optional, Union, List
from xradio.schema.bases import (
    xarray_dataset_schema,
    xarray_dataarray_schema,
    dict_schema,
)
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
import numpy

# Dimensions
Time = Literal["time"]
""" Observation time dimension """
TimeCal = Literal["time_cal"]
""" time dimension of system calibration (when not interpolated to main time)"""
TimeEphemeris = Literal["time_ephemeris"]
""" time dimension of ephemeris data (when not interpolated to main time) """
TimePhaseCal = Literal["time_phase_cal"]
""" Coordinate label for VLBI-specific phase cal time axis """
TimePointing = Literal["time_pointing"]
""" time dimension of pointing dataset (when not interpolated to main time) """
TimeWeather = Literal["time_weather"]
""" time dimension of weather dataset (when not interpolated to main time) """
AntennaName = Literal["antenna_name"]
""" Antenna name dimension """
StationName = Literal["station_name"]
""" Station identifier dimension """
ReceptorLabel = Literal["receptor_label"]
""" Receptor label dimension """
ToneLabel = Literal["tone_label"]
""" Tone label dimension """
BaselineId = Literal["baseline_id"]
""" Baseline ID dimension """
Frequency = Literal["frequency"]
""" Frequency dimension """
FrequencyCal = Literal["frequency_cal"]
""" Frequency dimension in the system calibration dataset """
Polarization = Literal["polarization"]
""" Polarization dimension """
UvwLabel = Literal["uvw_label"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """
SkyDirLabel = Literal["sky_dir_label"]
""" Coordinate labels of sky directions (typically shape 2 and 'ra', 'dec') """
LocalSkyDirLabel = Literal["local_sky_dir_label"]
""" Coordinate labels of local sky directions (typically shape 2 and 'az', 'alt') """
SphericalDirLabel = Literal["spherical_dir_label"]
""" Coordinate labels of spherical directions (shape 2 and 'lon', 'lat1' """
SkyPosLabel = Literal["sky_pos_label"]
""" Coordinate labels of sky positions (typically shape 3 and 'ra', 'dec', 'dist') """
SphericalPosLabel = Literal["spherical_pos_label"]
""" Coordinate labels of spherical positions (shape shape 3 and 'lon', 'lat1', 'dist2') """
EllipsoidPosLabel = Literal["ellipsoid_pos_label"]
""" Coordinate labels of geodetic earth location data (typically shape 3 and 'lon', 'lat', 'height')"""
CartesianPosLabel = Literal["cartesian_pos_label"]
""" Coordinate labels of geocentric earth location data (typically shape 3 and 'x', 'y', 'z')"""
nPolynomial = Literal["n_polynomial"]
""" For data that is represented as variable in time using Taylor expansion """
PolyTerm = Literal["poly_term"]
""" Polynomial term used in VLBI GAIN_CURVE """
LineLabel = Literal["line_label"]
""" Line labels (for line names and variables). """

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
UnitsSeconds = list[Literal["s"]]
UnitsHertz = list[Literal["Hz"]]
UnitsMeters = list[Literal["m"]]

UnitsOfSkyCoordInRadians = list[Literal["rad"], Literal["rad"]]
UnitsOfLocationInMetersOrRadians = Union[
    list[Literal["m"], Literal["m"], Literal["m"]],
    list[Literal["rad"], Literal["rad"], Literal["m"]],
]
UnitsOfPositionInRadians = list[Literal["rad"], Literal["rad"], Literal["m"]]
UnitsOfDopplerShift = Union[list[Literal["ratio"]], list[Literal["m/s"]]]

UnitsRadians = list[Literal["rad"]]
UnitsKelvin = list[Literal["K"]]
UnitsKelvinPerJansky = list[Literal["K/Jy"]]
UnitsMetersPerSecond = list[Literal["m/s"]]
UnitsPascal = list[Literal["Pa"]]  # hPa? (in MSv2)
UnitsPerSquareMeters = list[Literal["/m^2"]]


# Quantities


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
    Quantity with units of Hertz
    """

    data: Data[ZD, float]

    units: Attr[UnitsMeters]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInMetersPerSecondArray:
    """
    Quantity with units of Hertz
    """

    data: Data[ZD, float]

    units: Attr[UnitsMetersPerSecond]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInRadiansArray:
    """
    Quantity with units of Hertz
    """

    data: Data[ZD, float]

    units: Attr[UnitsRadians]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInKelvinArray:
    """
    Quantity with units of Kelvins
    """

    data: Data[ZD, float]

    units: Attr[UnitsKelvin]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInKelvinPerJanskyArray:
    """
    Quantity with units of K/Jy (sensitivity in gain curve)
    """

    data: Data[ZD, numpy.float64]

    units: Attr[UnitsKelvinPerJansky]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInPascalArray:
    """
    Quantity with units of Pa
    """

    data: Data[ZD, numpy.float64]

    units: Attr[UnitsPascal]
    type: Attr[Quantity] = "quantity"


@xarray_dataarray_schema
class QuantityInPerSquareMetersArray:
    """
    Quantity with units of /m^2
    """

    data: Data[ZD, numpy.float64]

    units: Attr[UnitsPerSquareMeters]
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
    units: Attr[UnitsSeconds] = ("s",)
    """ Units to associate with axis"""
    scale: Attr[AllowedTimeScales] = "utc"
    """
    Time scale of data. Must be one of ``(‘tai’, ‘tcb’, ‘tcg’, ‘tdb’, ‘tt’, ‘ut1’, ‘utc’)``,
    see :py:class:`astropy.time.Time`
    """
    format: Attr[AllowedTimeFormats] = "unix"
    """Time representation and epoch, see :py:class:`TimeArray`."""


# Taken from the list of astropy built-in frame classes: https://docs.astropy.org/en/stable/coordinates/index.html
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
    """Measures array for data variables that are sky coordinates, used in :py:class:`FieldSourceXds`"""

    data: Data[Union[SkyDirLabel, SkyPosLabel], float]

    type: Attr[SkyCoord] = "sky_coord"
    units: Attr[UnitsOfSkyCoordInRadians] = ("rad", "rad")
    frame: Attr[AllowedSkyCoordFrames] = ""
    """
    Possible values are astropy SkyCoord frames.
    Several casacore frames found in MSv2 are translated to astropy frames as follows: AZELGEO=>altaz, J2000=>fk5, ICRS=>icrs.
    From fixvis docs: clean and the im tool ignore the reference frame
    claimed by the UVW column (it is often mislabelled as ITRF when it is
    really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame
    as the phase tracking center. calcuvw does not yet force the UVW column and
    field centers to use the same reference frame! Blank = use the phase
    tracking frame of vis.
    """


@xarray_dataarray_schema
class LocalSkyCoordArray:
    """Measures array for the arrays that have coordinate local_sky_dir_label in :py:class:`PointingXds`"""

    data: Data[LocalSkyDirLabel, float]

    type: Attr[SkyCoord] = "sky_coord"
    units: Attr[UnitsOfSkyCoordInRadians] = ("rad", "rad")
    frame: Attr[AllowedSkyCoordFrames] = "fk5"
    """
    From fixvis docs: clean and the im tool ignore the reference frame claimed by the UVW column (it is often mislabelled
    as ITRF when it is really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame as the phase tracking
    center. calcuvw does not yet force the UVW column and field centers to use the same reference frame! Blank = use the
    phase tracking frame of vis.
    """


# Coordinates / Axes
@xarray_dataarray_schema
class TimeCoordArray:
    """Data model of the main dataset time axis. See also :py:class:`TimeArray`."""

    data: Data[Time, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``), see also see :py:class:`TimeArray`.
    """

    type: Attr[Time] = "time"
    """ Coordinate type. Should be ``"time"``. """

    units: Attr[UnitsSeconds] = ("s",)
    """ Units to associate with axis"""

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`TimeArray` """

    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`TimeArray`"""

    integration_time: Attr[QuantityInSecondsArray] = None
    """ The nominal sampling interval (ms v2). Units of seconds. """


@xarray_dataarray_schema
class TimeInterpolatedCoordArray:
    """
    Data model of a time axis when it is interpolated to match the time
    axis of the main dataset. This can be used in the system_calibration_xds,
    pointing_xds, weather_xds, field_and_source_info_xds, and phase_cal_xds
    when their respective time_cal, time_pointing, time_weather,
    time_ephemeris or time_phase_cal are interpolated to the main dataset
    time. See also :py:class:`TimeArray`.

    The only difference with respect to the main TimeCoordArray is the
    absence of the attribute integration_time
    """

    data: Data[Time, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``), see also see :py:class:`TimeArray`.
    """

    type: Attr[Time] = "time"
    """ Coordinate type. Should be ``"time"``. """

    units: Attr[UnitsSeconds] = ("s",)
    """ Units to associate with axis"""

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`TimeArray` """

    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`TimeArray`"""


@xarray_dataarray_schema
class TimeCalCoordArray:
    """Data model of 'time_cal' axis (time axis in system_calibration_xds
    subdataset when not interpolated to the main time axis. See also
    :py:class:`TimeCoordArray`."""

    data: Data[TimeCal, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``).
    """

    type: Attr[Time] = "time_cal"
    """ Coordinate type. Should be ``"time_cal"``. """

    units: Attr[UnitsSeconds] = ("s",)
    """ Units to associate with axis"""

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`TimeArray` """

    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`TimeArray`"""


@xarray_dataarray_schema
class TimePointingCoordArray:
    """Data model of the 'time_pointing' axis (time axis in pointing_xds
    subdataset when not interpolated to the main time axis. See also
    :py:class:`TimeCoordArray`."""

    data: Data[TimePointing, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``).
    """

    type: Attr[TimePointing] = "time_pointing"
    """ Coordinate type. Should be ``"time_pointing"``. """

    units: Attr[UnitsSeconds] = ("s",)
    """ Units to associate with axis"""

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`TimeArray` """

    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`TimeArray`"""


@xarray_dataarray_schema
class TimeEphemerisCoordArray:
    """Data model of the 'time_ephemeris' axis (time axis in the
    field_and_source_info_xds subdataset when not interpolated to the main
    time axis. See also :py:class:`TimeCoordArray`."""

    data: Data[TimeEphemeris, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``).
    """

    type: Attr[TimeEphemeris] = "time_ephemeris"
    """ Coordinate type. Should be ``"time_ephemeris"``. """

    units: Attr[UnitsSeconds] = ("s",)
    """ Units to associate with axis"""

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`TimeArray` """

    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`TimeArray`"""


@xarray_dataarray_schema
class TimeWeatherCoordArray:
    """Data model of the 'time_weather' axis (time axis in the weather_xds
    subdataset when not interpolated to the main time axis. See also
    :py:class:`TimeCoordArray`."""

    data: Data[TimeWeather, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``).
    """

    type: Attr[Time] = "time_weather"
    """ Coordinate type. Should be ``"time_weather"``. """

    units: Attr[UnitsSeconds] = ("s",)
    """ Units to associate with axis"""

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`TimeArray` """

    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`TimeArray`"""


# For now allowing both some of the casacore frames (from "REST" to "TOPO" - all in uppercase) as well as
# the astropy frames (all in lowercase, taken from the list of SpectralCoord:
# https://docs.astropy.org/en/stable/coordinates/spectralcoord.html)
AllowedSpectralCoordFrames = Literal[
    "REST",
    # "LSRK" -> "lsrk",
    # "LSRD" -> "lsrd",
    "BARY",
    "GEO",
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

    units: Attr[UnitsHertz] = ("Hz",)

    observer: Attr[AllowedSpectralCoordFrames] = "gcrs"
    """
    Capitalized reference observers are from casacore. TOPO implies creating astropy earth_location.
    Astropy velocity reference frames are lowercase. Note that Astropy does not use the name 'TOPO' (telescope centric)
    rather it assumes if no velocity frame is given that this is the default.
    """

    type: Attr[SpectralCoord] = "spectral_coord"


AllowedLocationFrames = Literal["ITRF", "GRS80", "WGS84", "WGS72", "Undefined"]


AllowedLocationCoordinateSystems = Literal[
    "geocentric",
    "planetcentric",
    "geodetic",
    "planetodetic",
    "orbital",
]


@xarray_dataarray_schema
class LocationArray:
    """
    Measure type used for example in antenna_xds/ANTENNA_POSITION, field_and_source_xds/OBSERVER_POSITION
    Data dimensions can be EllipsoidPosLabel or CartesianPosLabel
    """

    data: Data[Union[EllipsoidPosLabel, CartesianPosLabel], float]

    units: Attr[UnitsOfLocationInMetersOrRadians]
    """
    If the units are a list of strings then it must be the same length as
    the last dimension of the data array. This allows for having different
    units in the same data array,for example geodetic coordinates could use
    ``['rad','rad','m']``.
    """

    frame: Attr[AllowedLocationFrames]
    """
    Can be ITRF, GRS80, WGS84, WGS72, Undefined
    """

    coordinate_system: Attr[AllowedLocationCoordinateSystems]
    """ Can be ``geocentric/planetcentric, geodetic/planetodetic, orbital`` """

    origin_object_name: Attr[str]
    """
    earth/sun/moon/etc
    """

    type: Attr[Location] = "location"
    """ """


@xarray_dataarray_schema
class EllipsoidPosLocationArray:
    """
    Measure type used for example in field_and_source_xds/SUB_OBSERVER_POSITION, SUB_SOLAR_POSITION
    """

    data: Data[EllipsoidPosLabel, float]

    frame: Attr[AllowedLocationFrames]
    """
    Can be ITRF, GRS80, WGS84, WGS72
    """

    coordinate_system: Attr[AllowedLocationCoordinateSystems]
    """ Can be ``geocentric/planetcentric, geodetic/planetodetic, orbital`` """

    origin_object_name: Attr[str]
    """
    earth/sun/moon/etc
    """

    type: Attr[Location] = "location"
    """ """

    units: Attr[UnitsOfPositionInRadians] = ("rad", "rad", "m")
    """
    If the units are a list of strings then it must be the same length as
    the last dimension of the data array. This allows for having different
    units in the same data array,for example geodetic coordinates could use
    ``['rad','rad','m']``.
    """


@xarray_dataarray_schema
class BaselineArray:
    """Model of the baseline_id coordinate in the main dataset (interferometric data, :py:class:`VisibiiltyXds`)"""

    data: Data[BaselineId, Union[numpy.int64, numpy.int32]]
    """Unique id for each baseline."""
    long_name: Optional[Attr[str]] = "Baseline ID"


@xarray_dataarray_schema
class BaselineAntennaNameArray:
    """Array of antenna_name by baseline_id, as used in main_xds and main_sd_xds
    (antenna_name by baseline_id dim"""

    data: Data[BaselineId, str]
    """Unique id for each baseline."""
    long_name: Optional[Attr[str]] = "Antenna name by baseline_id"


@xarray_dataarray_schema
class AntennaNameArray:
    """
    Model of the antenna_name coordinate, used in the main dataset (single dish data, :py:class:`VisibiiltyXds`)
    and several sub-datasets such as antenna_xds, pointing_xds, weather_xds, system_calibration_xds, gain_curve_xds, etc.
    """

    data: Data[AntennaName, str]
    """Unique name for each antenna(_station)."""
    long_name: Optional[Attr[str]] = "Antenna name"


AllowedDopplerTypes = Literal[
    "radio", "optical", "z", "ratio", "true", "relativistic", "beta", "gamma"
]


@xarray_dataarray_schema
class DopplerArray:
    """Doppler measure information for the frequency coordinate"""

    data: Data[ZD, numpy.float64]

    type: Attr[Doppler] = "doppler"
    """ Coordinate type. Should be ``"spectral_coord"``. """

    units: Attr[UnitsOfDopplerShift] = ("m/s",)
    """ Units to associate with axis, [ratio]/[m/s]"""

    doppler_type: Attr[AllowedDopplerTypes] = "radio"
    """
    Allowable values: radio, optical, z, ratio, true, relativistic, beta, gamma.
    Astropy only has radio and optical. Using casacore types: https://casadocs.readthedocs.io/en/stable/notebooks/memo-series.html?highlight=Spectral%20Frames#Spectral-Frames
    """


@xarray_dataarray_schema
class FrequencyArray:
    """Frequency coordinate in the main dataset."""

    data: Data[Frequency, float]
    """ Time, expressed in SI seconds since the epoch. """
    spectral_window_name: Attr[str]
    """ Name associated with spectral window. """
    frequency_group_name: Optional[Attr[str]]
    """ Name associated with frequency group - needed for multi-band VLBI fringe-fitting."""
    reference_frequency: Attr[SpectralCoordArray]
    """ A frequency representative of the spectral window, usually the sky
    frequency corresponding to the DC edge of the baseband. Used by the calibration
    system if a ﬁxed scaling frequency is required or in algorithms to identify the
    observing band. """
    channel_width: Attr[
        QuantityInHertzArray
    ]  # Not SpectralCoord, as it is a difference
    """ The nominal channel bandwidth. Same units as data array (see units key). """
    doppler: Optional[Attr[DopplerArray]]
    """ Doppler tracking information """

    type: Attr[SpectralCoord] = "spectral_coord"
    """ Coordinate type. Should be ``"spectral_coord"``. """
    long_name: Optional[Attr[str]] = "Frequency"
    """ Long-form name to use for axis"""
    units: Attr[UnitsHertz] = ("Hz",)
    """ Units to associate with axis"""
    observer: Attr[AllowedSpectralCoordFrames] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """


@xarray_dataarray_schema
class FrequencyCalArray:
    """The frequency_cal coordinate of the system calibration dataset. It has
    only measures data, as opposed to the frequency array of the main dataset."""

    data: Data[FrequencyCal, float]
    """ Time, expressed in SI seconds since the epoch. """

    type: Attr[SpectralCoord] = "spectral_coord"
    units: Attr[UnitsHertz] = ("Hz",)
    """ Units to associate with axis"""

    observer: Attr[AllowedSpectralCoordFrames] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
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
    long_name: Optional[Attr[str]] = "Polarization"
    """ Long-form name to use for axis. Should be ``"Polarization"``"""


@xarray_dataarray_schema
class UvwLabelArray:
    """
    Coordinate axis to make up ``("u", "v", "w")`` tuple, see :py:class:`UvwArray`.
    """

    data: Data[UvwLabel, str] = ("u", "v", "w")
    """Should be ``('u','v','w')``, used by :py:class:`UvwArray`"""
    long_name: Optional[Attr[str]] = "U/V/W label"
    """ Long-form name to use for axis. Should be ``"U/V/W label"``"""


# Data variables
@xarray_dataarray_schema
class FlagArray:
    """
    An array of Boolean values with the same shape as `VISIBILITY`,
    representing the cumulative flags applying to this data matrix. Data are
    flagged bad if the ``FLAG`` array element is ``True``.
    """

    data: Data[
        Union[
            tuple[Time, BaselineId, Frequency, Polarization],
            tuple[Time, BaselineId, Frequency],
            tuple[Time, BaselineId],
            tuple[Time, AntennaName, Frequency, Polarization],  # SD
        ],
        bool,
    ]
    time: Coordof[TimeCoordArray]
    baseline_id: Optional[Coordof[BaselineArray]]  # Only IF
    antenna_name: Optional[Coordof[AntennaNameArray]]  # Only SD
    frequency: Coordof[FrequencyArray]
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Visibility flags"

    allow_mutiple_versions: Optional[Attr[bool]] = True


@xarray_dataarray_schema
class WeightArray:
    """
    The weight for each channel, with the same shape as the associated
    :py:class:`VisibilityArray`, as assigned by the correlator or processor.

    Weight spectrum in ms v2 is renamed weight. Should be calculated as
    1/sigma^2 (sigma rms noise).
    """

    data: Data[
        Union[
            tuple[Time, BaselineId, Frequency, Polarization],
            tuple[Time, BaselineId, Frequency],
            tuple[Time, BaselineId],
            tuple[Time, AntennaName, Frequency, Polarization],  # SD
        ],
        Union[numpy.float16, numpy.float32, numpy.float64],
    ]
    """Visibility weights"""
    time: Coordof[TimeCoordArray]
    baseline_id: Optional[Coordof[BaselineArray]]  # Only IF
    antenna_name: Optional[Coordof[AntennaNameArray]]  # Only SD
    frequency: Coordof[FrequencyArray] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Visibility weights"

    allow_mutiple_versions: Optional[Attr[bool]] = True


# J2000=>fk5 is used most often. icrs is used less often. Both fk5 and icrs are also borrowed from the field center (to fix
# ITRF=>J2000). APP has only been seen in WSRT datasets.
AllowedUvwFrames = Literal[
    "fk5",
    "icrs",
    "APP",  # "apparent geocentric position", used in WSRT datasets
]


@xarray_dataarray_schema
class UvwArray:
    """
    Coordinates for the baseline from ``baseline_antenna2_id`` to
    ``baseline_antenna1_id``, i.e. the baseline is equal to the difference
    ``POSITION2 - POSITION1``. The UVW given are for the ``TIME_CENTROID``, and
    correspond in general to the reference type for the
    ``field_info.phase_dir``.

    The baseline direction should be: ``W`` towards source direction; ``V`` in
    plane through source and system's pole; ``U`` in direction of increasing
    longitude coordinate.  So citing
    http://casa.nrao.edu/Memos/CoordConvention.pdf: Consider an XYZ Celestial
    coordinate system centered at the location of the interferometer, with
    :math:`X` towards the East, :math:`Z` towards the NCP and :math:`Y` to
    complete a right-handed system. The UVW coordinate system is then defined
    by the hour-angle and declination of the phase-reference direction such
    that

    #. when the direction of observation is the NCP (`ha=0,dec=90`),
       the UVW coordinates are aligned with XYZ,
    #. V, W and the NCP are always on a Great circle,
    #. when W is on the local meridian, U points East
    #. when the direction of observation is at zero declination, an
       hour-angle of -6 hours makes W point due East.

    This definition also determines the sign of the phase of ``VISIBILITY``.

    """

    data: Data[
        Union[tuple[Time, BaselineId, UvwLabel]],
        Union[
            numpy.float16,
            numpy.float32,
            numpy.float64,
        ],
    ]
    """Baseline coordinates from ``baseline_antenna2_id`` to ``baseline_antenna1_id``"""
    time: Coordof[TimeCoordArray]
    baseline_id: Optional[Coordof[BaselineArray]]
    uvw_label: Coordof[UvwLabelArray] = ("u", "v", "w")

    long_name: Optional[Attr[str]] = "Baseline coordinates"
    """ Long-form name to use for axis. Should be ``"Baseline coordinates``"""

    type: Attr[Literal["uvw"]] = "uvw"
    frame: Attr[AllowedUvwFrames] = "icrs"
    """ To be defined in astropy (see for example https://github.com/astropy/astropy/issues/7766) """
    units: Attr[UnitsMeters] = ("m",)

    allow_mutiple_versions: Optional[Attr[bool]] = True


@xarray_dataarray_schema
class TimeSamplingArray:
    """
    Model of arrays of measures used in the main dataset for data variables such as TIME_CENTROID and
    TIME_CENTROID_EXTRA_PRECISION.
    """

    data: Data[
        Union[
            tuple[Time, BaselineId],
            tuple[Time, AntennaName],  # SD
        ],
        float,
    ]

    time: Coordof[TimeCoordArray]
    baseline_id: Optional[Coordof[BaselineArray]]  # Only IF
    antenna_name: Optional[Coordof[AntennaNameArray]]  # Only SD

    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[AllowedTimeFormats] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """

    long_name: Optional[Attr[str]] = "Time sampling data"
    units: Attr[UnitsSeconds] = ("s",)


@xarray_dataarray_schema
class FreqSamplingArray:
    """
    Model of frequency related data variables of the main dataset, such as EFFECTIV_CHANNEL_WIDTH and FREQUENCY_CENTROID.
    """

    data: Data[
        Union[
            tuple[Time, BaselineId, Frequency, Polarization],
            tuple[Time, BaselineId, Frequency],
            tuple[Time, Frequency],
            tuple[Frequency],
        ],
        float,
    ]
    """
    Data about frequency sampling, such as centroid or integration
    time. Concrete function depends on concrete data array within
    :py:class:`VisibilityXds` or :py:class:`SpectrumXds`.
    """
    frequency: Coordof[FrequencyArray]
    time: Optional[Coordof[TimeCoordArray]] = None
    baseline_id: Optional[Coordof[BaselineArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Frequency sampling data"
    units: Attr[UnitsHertz] = ("Hz",)
    observer: Attr[AllowedSpectralCoordFrames] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """


# Define FieldAndSourceXds dataset already here, as it is needed in the
# definition of VisibilityArray
@xarray_dataset_schema
class FieldSourceXds:
    """
    Field positions for each source.

    Defines a field position on the sky. For interferometers, this is the correlated field position.
    For single dishes, this is the nominal pointing direction.
    """

    source_name: Optional[Coord[Union[ZD, Time], str]]
    """ Source name. """
    field_name: Optional[Coord[Union[ZD, Time], str]]
    """Field name."""

    time: Optional[Coordof[TimeInterpolatedCoordArray]]
    """Midpoint of time for which this set of parameters is accurate. Labeled 'time' when interpolated to main time """
    time_ephemeris: Optional[Coordof[TimeEphemerisCoordArray]]
    """Midpoint of time for which this set of parameters is accurate. Labeled 'time_ephemeris' when not interpolating to main time """

    line_label: Optional[Coord[LineLabel, str]]
    """ Line labels (for line names and variables). """

    line_names: Optional[
        Coord[
            Union[
                tuple[LineLabel],
                tuple[Time, LineLabel],
                tuple[TimeEphemeris, LineLabel],
            ],
            str,
        ]
    ]
    """ Line names (e.g. v=1, J=1-0, SiO). """

    FIELD_PHASE_CENTER: Optional[
        Data[Union[ZD, tuple[Time], tuple[TimeEphemeris]], SkyCoordArray]
    ]
    """
    Offset from the SOURCE_DIRECTION that gives the direction of phase
    center for which the fringes have been stopped-that is a point source in
    this direction will produce a constant measured phase (page 2 of
    https://articles.adsabs.harvard.edu/pdf/1999ASPC..180...79F). For
    conversion from MSv2, frame refers column keywords by default. If frame
    varies with field, it refers DelayDir_Ref column instead.
    """

    FIELD_REFERENCE_CENTER: Optional[
        Data[Union[ZD, tuple[Time], tuple[TimeEphemeris]], SkyCoordArray]
    ]
    """
    Used in single-dish to record the associated reference direction if positionswitching
    been applied. For conversion from MSv2, frame refers column keywords by default. If
    frame varies with field, it refers DelayDir_Ref column instead.
    """

    SOURCE_LOCATION: Optional[
        Data[
            Union[
                ZD,
                tuple[Time],
                tuple[TimeEphemeris],
            ],
            SkyCoordArray,
        ]
    ]
    """
    CASA Table Cols: RA,DEC,Rho."Astrometric RA and Dec and Geocentric
    distance with respect to the observer’s location (Geocentric). "Adjusted
    for light-time aberration only. With respect to the reference plane and
    equinox of the chosen system (ICRF or FK4/B1950). If the FK4/B1950 frame
    output is selected, elliptic aberration terms are added. Astrometric RA/DEC
    is generally used when comparing or reducing data against a star catalog."
    https://ssd.jpl.nasa.gov/horizons/manual.html : 1. Astrometric RA & DEC
    """

    LINE_REST_FREQUENCY: Optional[
        Data[
            Union[
                tuple[LineLabel],
                tuple[Time, LineLabel],
                tuple[TimeEphemeris, LineLabel],
            ],
            SpectralCoordArray,
        ]
    ]
    """ Rest frequencies for the transitions. """

    LINE_SYSTEMIC_VELOCITY: Optional[
        Data[
            Union[
                tuple[LineLabel],
                tuple[Time, LineLabel],
                tuple[TimeEphemeris, LineLabel],
            ],
            QuantityInMetersPerSecondArray,
        ]
    ]
    """ Systemic velocity at reference """

    SOURCE_RADIAL_VELOCITY: Optional[
        Data[
            Union[ZD, tuple[Time], tuple[TimeEphemeris]], QuantityInMetersPerSecondArray
        ]
    ]
    """ CASA Table Cols: RadVel. Geocentric distance rate """

    NORTH_POLE_POSITION_ANGLE: Optional[
        Data[Union[ZD, tuple[Time], tuple[TimeEphemeris]], QuantityInRadiansArray]
    ]
    """ CASA Table cols: NP_ang, "Targets' apparent north-pole position angle (counter-clockwise with respect to direction of true-of-date reference-frame north pole) and angular distance from the sub-observer point (center of disc) at print time. A negative distance indicates the north-pole is on the hidden hemisphere." https://ssd.jpl.nasa.gov/horizons/manual.html : 17. North pole position angle & distance from disc center. """

    NORTH_POLE_ANGULAR_DISTANCE: Optional[
        Data[Union[ZD, tuple[Time], tuple[TimeEphemeris]], QuantityInRadiansArray]
    ]
    """ CASA Table cols: NP_dist, "Targets' apparent north-pole position angle (counter-clockwise with respect to direction of true-of date reference-frame north pole) and angular distance from the sub-observer point (center of disc) at print time. A negative distance indicates the north-pole is on the hidden hemisphere."https://ssd.jpl.nasa.gov/horizons/manual.html : 17. North pole position angle & distance from disc center. """

    SUB_OBSERVER_DIRECTION: Optional[
        Data[
            Union[
                ZD,
                tuple[Time],
                tuple[TimeEphemeris],
            ],
            EllipsoidPosLocationArray,
        ]
    ]
    """ CASA Table cols: DiskLong, DiskLat. "Apparent planetodetic longitude and latitude of the center of the target disc seen by the OBSERVER at print-time. This is not exactly the same as the "nearest point" for a non-spherical target shape (since the center of the disc might not be the point closest to the observer), but is generally very close if not a very irregular body shape. The IAU2009 rotation models are used except for Earth and MOON, which use higher-precision models. For the gas giants Jupiter, Saturn, Uranus and Neptune, IAU2009 longitude is based on the "System III" prime meridian rotation angle of the magnetic field. By contrast, pole direction (thus latitude) is relative to the body dynamical equator. There can be an offset between the magnetic pole and the dynamical pole of rotation. Down-leg light travel-time from target to observer is taken into account. Latitude is the angle between the equatorial plane and perpendicular to the reference ellipsoid of the body and body oblateness thereby included. The reference ellipsoid is an oblate spheroid with a single flatness coefficient in which the y-axis body radius is taken to be the same value as the x-axis radius. Whether longitude is positive to the east or west for the target will be indicated at the end of the output ephemeris." https://ssd.jpl.nasa.gov/horizons/manual.html : 14. Observer sub-longitude & sub-latitude """

    SUB_SOLAR_POSITION: Optional[
        Data[
            Union[
                ZD,
                tuple[Time],
                tuple[TimeEphemeris],
            ],
            EllipsoidPosLocationArray,
        ]
    ]
    """ CASA Table cols: Sl_lon, Sl_lat, r. "Heliocentric distance along with "Apparent sub-solar longitude and latitude of the Sun on the target. The apparent planetodetic longitude and latitude of the center of the target disc as seen from the Sun, as seen by the observer at print-time.  This is _NOT_ exactly the same as the "sub-solar" (nearest) point for a non-spherical target shape (since the center of the disc seen from the Sun might not be the closest point to the Sun), but is very close if not a highly irregular body shape.  Light travel-time from Sun to target and from target to observer is taken into account.  Latitude is the angle between the equatorial plane and the line perpendicular to the reference ellipsoid of the body. The reference ellipsoid is an oblate spheroid with a single flatness coefficient in which the y-axis body radius is taken to be the same value as the x-axis radius. Uses IAU2009 rotation models except for Earth and Moon, which uses a higher precision models. Values for Jupiter, Saturn, Uranus and Neptune are Set III, referring to rotation of their magnetic fields.  Whether longitude is positive to the east or west for the target will be indicated at the end of the output ephemeris." https://ssd.jpl.nasa.gov/horizons/manual.html : 15. Solar sub-longitude & sub-latitude  """

    HELIOCENTRIC_RADIAL_VELOCITY: Optional[
        Data[
            Union[ZD, tuple[Time], tuple[TimeEphemeris]], QuantityInMetersPerSecondArray
        ]
    ]
    """ CASA Table cols: rdot."The Sun's apparent range-rate relative to the target, as seen by the observer. A positive "rdot" means the target was moving away from the Sun, negative indicates movement toward the Sun." https://ssd.jpl.nasa.gov/horizons/manual.html : 19. Solar range & range-rate (relative to target) """

    OBSERVER_PHASE_ANGLE: Optional[
        Data[Union[ZD, tuple[Time], tuple[TimeEphemeris]], QuantityInRadiansArray]
    ]
    """ CASA Table cols: phang.""phi" is the true PHASE ANGLE at the observers' location at print time. "PAB-LON" and "PAB-LAT" are the FK4/B1950 or ICRF/J2000 ecliptic longitude and latitude of the phase angle bisector direction; the outward directed angle bisecting the arc created by the apparent vector from Sun to target center and the astrometric vector from observer to target center. For an otherwise uniform ellipsoid, the time when its long-axis is perpendicular to the PAB direction approximately corresponds to lightcurve maximum (or maximum brightness) of the body. PAB is discussed in Harris et al., Icarus 57, 251-258 (1984)." https://ssd.jpl.nasa.gov/horizons/manual.html : Phase angle and bisector """

    OBSERVER_POSITION: Optional[Data[ZD, LocationArray]]
    """ Observer location. """

    # --- Attributes ---
    doppler_shift_velocity: Optional[Attr[UnitsOfDopplerShift]]
    """ Velocity definition of the Doppler shift, e.g., RADIO or OPTICAL velocity in m/s """

    source_model_url: Optional[Attr[str]]
    """URL to access source model"""
    ephemeris_name: Optional[Attr[str]]
    """The name of the ephemeris. For example DE430.

    This can be used with Astropy solar_system_ephemeris.set('DE430'), see
    https://docs.astropy.org/en/stable/coordinates/solarsystem.html.
    """
    is_ephemeris: Attr[bool] = False

    type: Attr[Literal["field_and_source"]] = "field_and_source"
    """
    Type of dataset.
    """

    # --- Optional coordinates ---
    sky_dir_label: Optional[Coord[SkyDirLabel, str]] = ("ra", "dec")
    """ Coordinate labels of sky directions (typically shape 2 and 'ra', 'dec') """
    sky_pos_label: Optional[Coord[SkyPosLabel, str]] = ("ra", "dec", "dist")
    """ Coordinate lables of sky positions (typically shape 3 and 'ra', 'dec', 'dist') """
    ellipsoid_pos_label: Optional[Coord[EllipsoidPosLabel, str]] = (
        "lon",
        "lat",
        "height",
    )
    """ Coordinate labels of geodetic earth location data (typically shape 3 and 'lon', 'lat', 'height')"""
    cartesian_pos_label: Optional[Coord[CartesianPosLabel, str]] = ("x", "y", "z")
    """ Coordinate labels of geocentric earth location data (typically shape 3 and 'x', 'y', 'z')"""


@xarray_dataarray_schema
class SpectrumArray:
    """Definition of xr.DataArray for SPECTRUM data (single dish)"""

    data: Data[
        tuple[Time, AntennaName, Frequency, Polarization],
        Union[numpy.float64, numpy.float32, numpy.float16],
    ]

    time: Coordof[TimeCoordArray]
    antenna_name: Coordof[AntennaNameArray]
    frequency: Coordof[FrequencyArray]
    polarization: Coordof[PolarizationArray]

    field_and_source_xds: Attr[FieldSourceXds]
    long_name: Optional[Attr[str]] = "Spectrum values"
    """ Long-form name to use for axis. Should be ``"Spectrum values"``"""
    units: Attr[list[str]] = ("Jy",)


@xarray_dataarray_schema
class VisibilityArray:
    """Visibility data array in main dataset (interferometric data, :py:class:`VisibiiltyXds`)"""

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization],
        Union[numpy.complex64, numpy.complex128],
    ]

    time: Coordof[TimeCoordArray]
    baseline_id: Coordof[BaselineArray]
    polarization: Coordof[PolarizationArray]
    frequency: Coordof[FrequencyArray]

    field_and_source_xds: Attr[FieldSourceXds]
    long_name: Optional[Attr[str]] = "Visibility values"
    """ Long-form name to use for axis. Should be ``"Visibility values"``"""
    units: Attr[list[str]] = ("Jy",)

    allow_mutiple_versions: Optional[Attr[bool]] = True


# Info dicts


@dict_schema
class PartitionInfoDict:
    # spectral_window_id: missing / remove for good?
    spectral_window_name: str
    """ Spectral window Name """
    # field_id: missing / probably remove for good?
    field_name: list[str]
    """ List of all field names """
    polarization_setup: list[str]
    """ List of polrization bases. """
    scan_number: list[int]
    """ List of scan numbers. """
    source_name: list[str]
    """ List of source names. """
    # source_id: mising / remove for good?
    intents: list[str]
    """ Infromation in obs_mode column of MSv2 State table. """
    taql: Optional[str]
    """ The taql query used if converted from MSv2. """
    line_name: list[str]
    """ Spectral line names """


@dict_schema
class ObservationInfoDict:
    observer: list
    """List of observer names."""
    project: str
    """Project Code/Project_UID"""
    release_date: str
    """Project release date. This is the date on which the data may become
    public. Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""
    execution_block_id: Optional[str]
    """ ASDM: Indicates the position of the execution block in the project
    (sequential numbering starting at 1).  """
    execution_block_number: Optional[int]
    """ASDM: Indicates the position of the execution block in the project
    (sequential numbering starting at 1)."""
    execution_block_UID: Optional[str]
    """ASDM: The archive’s UID of the execution block."""
    session_reference: Optional[str]
    """ASDM: The observing session reference."""
    observing_script: Optional[str]
    """ASDM: The text of the observation script."""
    observing_script_UID: Optional[str]
    """ASDM: A reference to the Entity which contains the observing script."""
    observing_log: Optional[str]
    """ASDM: Logs of the observation during this execu- tion block."""


@dict_schema
class ProcessorInfoDict:
    type: str
    """Processor type; reserved keywords include (”CORRELATOR” -
    interferometric correlator; ”SPECTROMETER” - single-dish correlator;
    ”RADIOMETER” - generic detector/integrator)."""
    sub_type: str
    """Processor sub-type, e.g. ”GBT” or ”JIVE”."""


# Data Sets


@xarray_dataset_schema
class AntennaXds:
    """
    Antenna dataset: global antenna properties for each antenna.
    """

    # Coordinates
    antenna_name: Coordof[AntennaNameArray]
    """ Antenna name """
    station: Coord[AntennaName, str]
    """ Name of the station pad (relevant to arrays with moving antennas). """
    mount: Coord[AntennaName, str]
    """ Mount type of the antenna. Reserved keywords include: ”EQUATORIAL” - equatorial mount;
    ”ALT-AZ” - azimuth-elevation mount;
    "ALT-AZ+ROTATOR"  alt-az mount with feed rotator; introduced for ASKAP dishes;
    "ALT-AZ+NASMYTH-R": Nasmyth mount with receivers at the right-hand side of the cabin. Many high-frequency antennas used for VLBI have such a mount typel;
    "ALT-AZ+NASMYTH-L:: Nasmyth mount with receivers at the left-hand side of the cabin.
    ”X-Y” - x-y mount;
    ”SPACE-HALCA” - specific orientation model."""
    telescope_name: Coord[AntennaName, str]
    """ Useful when data is combined from mutiple arrays for example ACA + ALMA. """
    receptor_label: Coord[ReceptorLabel, str]
    """ Names of receptors """
    polarization_type: Coord[tuple[AntennaName, ReceptorLabel], str]
    """ Polarization type to which each receptor responds (e.g. ”R”,”L”,”X” or ”Y”).
    This is the receptor polarization type as recorded in the final correlated data (e.g. ”RR”); i.e.
    as measured after all polarization combiners. ['X','Y'], ['R','L'] """
    cartesian_pos_label: Optional[Coord[CartesianPosLabel, str]]
    """ (x,y,z) - either cartesian or ellipsoid """
    ellipsoid_pos_label: Optional[Coord[EllipsoidPosLabel, str]]
    """ (lon, lat, dist) - either cartesian or ellipsoid"""

    # Data variables
    ANTENNA_POSITION: Data[tuple[AntennaName], LocationArray]
    """
    In a right-handed frame, X towards the intersection of the equator and
    the Greenwich meridian, Z towards the pole.
    """
    ANTENNA_DISH_DIAMETER: Optional[Data[tuple[AntennaName], QuantityInMetersArray]]
    """
    The diameter of the main reflector (or the largest dimension for non-circular apertures).
    """
    ANTENNA_EFFECTIVE_DISH_DIAMETER: Optional[
        Data[tuple[AntennaName], QuantityInMetersArray]
    ]
    """ Effective dish diameter used in computing beam model (such as airy disk). """

    ANTENNA_BLOCKAGE: Optional[Data[tuple[AntennaName], QuantityInMetersArray]]
    """
    Blockage caused by secondary reflector used in computing beam model (such as airy disk).
    """

    # TODO: setting BEAM_OFFSET and RECEPTOR_ANGLE as optional for now, as it
    # is not present in some datasets (example: test_alma_ephemris_mosaic)
    ANTENNA_RECEPTOR_ANGLE: Optional[
        Data[tuple[AntennaName, ReceptorLabel], QuantityInRadiansArray]
    ]
    """
    Polarization reference angle. Converts into parallactic angle in the sky domain.
    """
    ANTENNA_FOCUS_LENGTH: Optional[Data[tuple[AntennaName], QuantityInMetersArray]]
    """
    Focus length. As defined along the optical axis of the antenna.
    """

    # Attributes
    overall_telescope_name: Optional[Attr[str]]
    """
    The name of the collection of arrays and dishes that were used for the observation.
    In many instances this will only be a single array or dish. An example of a
    telescope consistening of mutiple arrays and dishes is the EHT. The coordinate
    telescope_name will give the names of the constituent arrays and dishes. From
    MSv2 observation table.
    """
    relocatable_antennas: Optional[Attr[bool]]
    """ Can the antennas be moved (ALMA, VLA, NOEMA) """
    type: Attr[Literal["antenna"]] = "antenna"
    """
    Type of dataset. Expected to be ``antenna``
    """


@xarray_dataset_schema
class GainCurveXds:
    """
    Gain curve dataset. See See https://casacore.github.io/casacore-notes/265.pdf for a full description.
    """

    # Coordinates
    antenna_name: Coordof[AntennaNameArray]
    """ Antenna name """
    station: Coord[AntennaName, str]
    """ Name of the station pad (relevant to arrays with moving antennas). """
    mount: Coord[AntennaName, str]
    """ Mount type of the antenna. Reserved keywords include: ”EQUATORIAL” - equatorial mount;
    ”ALT-AZ” - azimuth-elevation mount;
    "ALT-AZ+ROTATOR"  alt-az mount with feed rotator; introduced for ASKAP dishes;
    "ALT-AZ+NASMYTH-R": Nasmyth mount with receivers at the right-hand side of the cabin. Many high-frequency antennas used for VLBI have such a mount typel;
    "ALT-AZ+NASMYTH-L:: Nasmyth mount with receivers at the left-hand side of the cabin.
    ”X-Y” - x-y mount;
    ”SPACE-HALCA” - specific orientation model."""
    telescope_name: Coord[AntennaName, str]
    """ Useful when data is combined from mutiple arrays for example ACA + ALMA. """
    receptor_label: Coord[ReceptorLabel, str]
    """ Names of receptors """
    polarization_type: Optional[Coord[tuple[AntennaName, ReceptorLabel], str]]
    """ Polarization type to which each receptor responds (e.g. ”R”,”L”,”X” or ”Y”).
    This is the receptor polarization type as recorded in the final correlated data (e.g. ”RR”); i.e.
    as measured after all polarization combiners. ['X','Y'], ['R','L'] """
    gain_curve_type: Optional[Coord[AntennaName, str]]
    """
    Gain curve type. Reserved keywords include:
    (”POWER(EL)” - Power as a function of elevation;
     ”POWER(ZA)” - Power as a function of zenith angle;
     ”VOLTAGE(EL)” - Voltage as a function of elevation;
     ”VOLTAGE(ZA)” - Voltage as a function of zenith angle). See https://casacore.github.io/casacore-notes/265.pdf
    """
    poly_term: Coord[PolyTerm, int]
    """Term orders in gain curve polynomial"""

    GAIN_CURVE: Data[tuple[AntennaName, PolyTerm, ReceptorLabel], numpy.float64]
    """ Coeﬃcients of the polynomial that describes the (power or voltage) gain.  """
    GAIN_CURVE_INTERVAL: Data[tuple[AntennaName], QuantityInSecondsArray]
    """ Time interval. """
    GAIN_CURVE_SENSITIVITY: Data[
        tuple[AntennaName, ReceptorLabel], QuantityInKelvinPerJanskyArray
    ]
    """ Sensitivity of the antenna expressed in K/Jy. This is what AIPS calls “DPFU”. """

    measured_date: Attr[str]
    """
    Date gain curve was measured. Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)
    """
    type: Attr[Literal["gain_curve"]] = "gain_curve"
    """
    Type of dataset. Expected to be ``gain_curve``
    """


@xarray_dataset_schema
class PhaseCalibrationXds:
    """
    Phase calibration dataset: signal chain phase calibration measurements.
    """

    # Coordinates
    antenna_name: Coordof[AntennaNameArray]
    """ Antenna name """
    station: Coord[AntennaName, str]
    """ Name of the station pad (relevant to arrays with moving antennas). """
    mount: Coord[AntennaName, str]
    """ Mount type of the antenna. Reserved keywords include: ”EQUATORIAL” - equatorial mount;
    ”ALT-AZ” - azimuth-elevation mount;
    "ALT-AZ+ROTATOR"  alt-az mount with feed rotator; introduced for ASKAP dishes;
    "ALT-AZ+NASMYTH-R": Nasmyth mount with receivers at the right-hand side of the cabin. Many high-frequency antennas used for VLBI have such a mount typel;
    "ALT-AZ+NASMYTH-L:: Nasmyth mount with receivers at the left-hand side of the cabin.
    ”X-Y” - x-y mount;
    ”SPACE-HALCA” - specific orientation model."""
    telescope_name: Coord[AntennaName, str]
    """ Useful when data is combined from mutiple arrays for example ACA + ALMA. """
    receptor_label: Coord[ReceptorLabel, str]
    """ Names of receptors """
    polarization_type: Optional[Coord[tuple[AntennaName, ReceptorLabel], str]]
    """ Polarization type to which each receptor responds (e.g. ”R”,”L”,”X” or ”Y”).
    This is the receptor polarization type as recorded in the final correlated data (e.g. ”RR”); i.e.
    as measured after all polarization combiners. ['X','Y'], ['R','L'] """
    time: Optional[Coordof[TimeInterpolatedCoordArray]]
    """ Time for VLBI phase cal"""
    time_phase_cal: Optional[Coord[TimePhaseCal, numpy.float64]]
    """ Time for VLBI phase cal"""
    tone_label: Optional[Coord[ToneLabel, str]]
    """
    Phase-cal tones that are measured. This number may vary by antenna, and may vary by spectral window as well, especially
    if spectral windows of varying widths are supported
    """

    PHASE_CAL: Data[
        Union[
            tuple[AntennaName, Time, ReceptorLabel, ToneLabel],
            tuple[AntennaName, TimePhaseCal, ReceptorLabel, ToneLabel],
        ],
        numpy.complex64,
    ]
    """
    Phase calibration measurements. These are provided as complex values that represent both the phase
and amplitude for a measured phase-cal tone. Measurements are provided as a two-dimensional array such that
separate measurements can be provided for each receptor of a feed (so separate values for each polarization)
for each of the measured tones. See https://casacore.github.io/casacore-notes/265.pdf
    """
    PHASE_CAL_CABLE_CAL: Data[
        Union[tuple[AntennaName, Time], tuple[AntennaName, TimePhaseCal]],
        QuantityInSecondsArray,
    ]
    """
    Cable calibration measurement. This is a measurement of the delay in the cable that provides the
reference signal to the receiver. There should be only a single reference signal per feed (even if that feed has
multiple receptors) so this is provided as a simple scalar. See https://casacore.github.io/casacore-notes/265.pdf
    """
    PHASE_CAL_INTERVAL: Data[
        Union[tuple[AntennaName, Time], tuple[AntennaName, TimePhaseCal]],
        QuantityInSecondsArray,
    ]
    """
    Time interval. See https://casacore.github.io/casacore-notes/265.pdf
    """
    PHASE_CAL_TONE_FREQUENCY: Data[
        Union[
            tuple[AntennaName, Time, ReceptorLabel, ToneLabel],
            tuple[AntennaName, TimePhaseCal, ReceptorLabel, ToneLabel],
        ],
        QuantityInHertzArray,
    ]
    """
    The sky frequencies of each measured phase-cal tone. See https://casacore.github.io/casacore-notes/265.pdf
    """

    type: Attr[Literal["phase_calibration"]] = "phase_calibration"
    """
    Type of dataset. Expected to be ``phase_calibration``
    """


@xarray_dataset_schema
class WeatherXds:
    """
    Weather dataset: station positions and time-dependent mean external atmosphere and weather information
    """

    # Coordinates
    station_name: Coord[StationName, str]
    """ Station identifier """
    time: Optional[Coordof[TimeInterpolatedCoordArray]]
    """ Mid-point of the time interval. Labeled 'time' when interpolated to main time axis """
    time_weather: Optional[Coordof[TimeWeatherCoordArray]]
    """ Mid-point of the time interval. Labeled 'time_cal' when not interpolated to main time axis """
    antenna_name: Optional[Coordof[AntennaNameArray]]
    """ Antenna identifier """
    ellipsoid_pos_label: Optional[Coord[EllipsoidPosLabel, str]] = (
        "lon",
        "lat",
        "height",
    )
    """ Coordinate labels of geodetic earth location data (typically shape 3 and 'lon', 'lat', 'height')"""
    cartesian_pos_label: Optional[Coord[CartesianPosLabel, str]] = ("x", "y", "z")
    """ Coordinate labels of geocentric earth location data (typically shape 3 and 'x', 'y', 'z')"""

    # Data variables (all optional)
    H2O: Optional[
        Data[
            Union[tuple[StationName, Time], tuple[StationName, TimeWeather]],
            QuantityInPerSquareMetersArray,
        ]
    ] = None
    """ Average column density of water """
    IONOS_ELECTRON: Optional[
        Data[
            Union[
                tuple[StationName, Time],
                tuple[StationName, TimeWeather],
            ],
            QuantityInPerSquareMetersArray,
        ]
    ] = None
    """ Average column density of electrons """
    PRESSURE: Optional[
        Data[
            Union[
                tuple[StationName, Time],
                tuple[StationName, TimeWeather],
            ],
            QuantityInPascalArray,
        ]
    ] = None
    """ Ambient atmospheric pressure """
    REL_HUMIDITY: Optional[
        Data[
            Union[
                tuple[StationName, Time],
                tuple[StationName, TimeWeather],
            ],
            numpy.float64,
        ]
    ] = None
    """ Ambient relative humidity """
    TEMPERATURE: Optional[
        Data[
            Union[
                tuple[StationName, Time],
                tuple[StationName, TimeWeather],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ Ambient air temperature for an antenna """
    DEW_POINT: Optional[
        Data[
            Union[
                tuple[StationName, Time],
                tuple[StationName, TimeWeather],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ Dew point """
    WIND_DIRECTION: Optional[
        Data[
            Union[
                tuple[StationName, Time],
                tuple[StationName, TimeWeather],
            ],
            QuantityInRadiansArray,
        ]
    ] = None
    """ Average wind direction """
    WIND_SPEED: Optional[
        Data[
            Union[
                tuple[StationName, Time],
                tuple[StationName, TimeWeather],
            ],
            QuantityInMetersPerSecondArray,
        ]
    ] = None
    """ Average wind speed """
    STATION_POSITION: Optional[Data[tuple[StationName], LocationArray]] = None
    """ Station position """

    # Attributes
    type: Attr[Literal["weather"]] = "weather"
    """
    Type of dataset.
    """


@xarray_dataset_schema
class PointingXds:
    """
    Pointing dataset: antenna pointing information.

    In the past the relationship and definition of the pointing infromation has not been clear. Here we attempt to clarify it by explaining the relationship between the ASDM, MSv2 and MSv4 pointing information.

    The following abreviations are used:

    - M2: Measurement Set version 2
    - M4: Measurement Set version 4
    - A : ASDM

    The following definitions come from the ASDM's `SDM Tables Short Description <https://drive.google.com/file/d/16a3g0GQxgcO7N_ZabfdtexQ8r2jRbYIS/view>`_ page 97-99:

    - A_encoder: The values measured from the antenna. They may be however affected by metrology, if applied. Note that for ALMA this column will contain positions obtained using the AZ POSN RSP and EL POSN RSP monitor points of the ACU and not the GET AZ ENC and GET EL ENC monitor points (as these do not include the metrology corrections). It is agreed that the the vendor pointing model will never be applied. AZELNOWAntenna.position
    - A_pointingDirection: This is the commanded direction of the antenna. It is obtained by adding the target and offset columns, and then applying the pointing model referenced by PointingModelId. The pointing model can be the composition of the absolute pointing model and of a local pointing model. In that case their coefficients will both be in the PointingModel table.
    - A_target: This is the field center direction (as given in the Field Table), possibly affected by the optional antenna-based sourceOffset. This column is in horizontal coordinates. AZELNOWAntenna.position
    - A_offset: Additional offsets in horizontal coordinates (usually meant for measuring the pointing corrections, mapping the antenna beam, ...). AZELNOWAntenna.positiontarget
    - A_sourceOffset : Optionally, the antenna-based mapping offsets in the field. These are in the equatorial system, and used, for instance, in on-the-fly mapping when the antennas are driven independently across the field.

    M2_DIRECTION = rotate(A_target,A_offset)   #A_target is rotated to by A_offset

    if withPointingCorrection : M2_DIRECTION = rotate(A_target,A_offset) + (A_encoder - A_pointingDirection)

    M2_TARGET = A_target
    M2_POINTING_OFFSET = A_offset
    M2_ENCODER = A_encoder

    It should be noted that definition of M2_direction is not consistent, it depends if withPointingCorrection is set to True or False (see the `importasdm documenation <https://casadocs.readthedocs.io/en/v6.2.0/api/tt/casatasks.data.importasdm.html#with-pointing-correction>`_  and `code <https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casatools/src/tools/sdm/sdm_cmpt.cc#2257>`_ for details).

    M4_DIRECTION = M2_DIRECTION (withPointingCorrection=True)
    M4_ENCODER = M2_ENCODER

    """

    antenna_name: Coordof[AntennaNameArray]
    """
    Antenna name, as specified by baseline_antenna1/2_name in visibility dataset
    """

    local_sky_dir_label: Coord[LocalSkyDirLabel, str]
    """
    Direction labels.
    """

    POINTING_BEAM: Data[
        Union[
            tuple[Time, AntennaName],
            tuple[TimePointing, AntennaName],
            tuple[Time, AntennaName, nPolynomial],
            tuple[TimePointing, AntennaName, nPolynomial],
        ],
        LocalSkyCoordArray,
    ]
    """
    The direction of the peak response of the beam and is equavalent to the MSv2 DIRECTION (M2_direction) with_pointing_correction=True, optionally expressed as polynomial coefficients.
    """

    time: Optional[Coordof[TimeInterpolatedCoordArray]] = None
    """
    Mid-point of the time interval for which the information in this row is
    valid. Required to use the same time measure reference as in visibility dataset.
    Labeled 'time' when interpolating to main time axis.
    """
    time_pointing: Optional[Coordof[TimePointingCoordArray]] = None
    """ Midpoint of time for which this set of parameters is accurate. Labeled
    'time_pointing' when not interpolating to main time axis """

    POINTING_DISH_MEASURED: Optional[
        Data[
            Union[
                tuple[Time, AntennaName],
                tuple[TimePointing, AntennaName],
            ],
            LocalSkyCoordArray,
        ]
    ] = None
    """
    The current encoder values on the primary axes of the mount type for
    the antenna. ENCODER in MSv2 (M2_encoder).
    """
    POINTING_OVER_THE_TOP: Optional[
        Data[Union[tuple[Time, AntennaName], tuple[TimePointing, AntennaName]], bool]
    ] = None
    """
    True if the antenna was driven to this position ”over the top” (az-el mount).
    """

    # Attributes
    type: Attr[Literal["pointing"]] = "pointing"
    """
    Type of dataset.
    """


@xarray_dataset_schema
class SystemCalibrationXds:
    """
    System calibration dataset: time- and frequency- variable calibration measurements for each antenna,
    as indexed on receptor
    """

    # Coordinates
    antenna_name: Coordof[AntennaNameArray]
    """ Antenna identifier """
    receptor_label: Coord[ReceptorLabel, numpy.int64]
    """  """
    time: Optional[Coordof[TimeInterpolatedCoordArray]] = None
    """ Midpoint of time for which this set of parameters is accurate. Labeled 'time' when interpolating to main time axis """
    time_cal: Optional[Coordof[TimeCalCoordArray]] = None
    """ Midpoint of time for which this set of parameters is accurate. Labeled 'time_cal' when not interpolating to main time axis """
    frequency: Optional[Coordof[FrequencyCalArray]] = None
    """  """
    frequency_cal: Optional[Coord[FrequencyCal, int]] = None
    """TODO: What is this?"""

    # Data variables (all optional)
    PHASE_DIFFERENCE: Optional[
        Data[
            Union[tuple[AntennaName, TimeCal], tuple[AntennaName, Time]],
            QuantityInRadiansArray,
        ]
    ] = None
    """ Phase difference between receptor 0 and receptor 1 """
    TCAL: Optional[
        Data[
            Union[
                tuple[AntennaName, TimeCal, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, TimeCal, ReceptorLabel, Frequency],
                tuple[AntennaName, TimeCal, ReceptorLabel],
                tuple[AntennaName, Time, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, Time, ReceptorLabel, Frequency],
                tuple[AntennaName, Time, ReceptorLabel],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ Calibration temp """
    TRX: Optional[
        Data[
            Union[
                tuple[AntennaName, TimeCal, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, TimeCal, ReceptorLabel, Frequency],
                tuple[AntennaName, TimeCal, ReceptorLabel],
                tuple[AntennaName, Time, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, Time, ReceptorLabel, Frequency],
                tuple[AntennaName, Time, ReceptorLabel],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ Receiver temperature """
    TSKY: Optional[
        Data[
            Union[
                tuple[AntennaName, TimeCal, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, TimeCal, ReceptorLabel, Frequency],
                tuple[AntennaName, TimeCal, ReceptorLabel],
                tuple[AntennaName, Time, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, Time, ReceptorLabel, Frequency],
                tuple[AntennaName, Time, ReceptorLabel],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ Sky temperature """
    TSYS: Optional[
        Data[
            Union[
                tuple[AntennaName, TimeCal, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, TimeCal, ReceptorLabel, Frequency],
                tuple[AntennaName, TimeCal, ReceptorLabel],
                tuple[AntennaName, Time, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, Time, ReceptorLabel, Frequency],
                tuple[AntennaName, Time, ReceptorLabel],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ System temperature """
    TANT: Optional[
        Data[
            Union[
                tuple[AntennaName, TimeCal, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, TimeCal, ReceptorLabel, Frequency],
                tuple[AntennaName, TimeCal, ReceptorLabel],
                tuple[AntennaName, Time, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, Time, ReceptorLabel, Frequency],
                tuple[AntennaName, Time, ReceptorLabel],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ Antenna temperature """
    TANT_SYS: Optional[
        Data[
            Union[
                tuple[AntennaName, TimeCal, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, TimeCal, ReceptorLabel, Frequency],
                tuple[AntennaName, TimeCal, ReceptorLabel],
                tuple[AntennaName, Time, ReceptorLabel, FrequencyCal],
                tuple[AntennaName, Time, ReceptorLabel, Frequency],
                tuple[AntennaName, Time, ReceptorLabel],
            ],
            QuantityInKelvinArray,
        ]
    ] = None
    """ TANT/TSYS """

    # Attributes
    type: Attr[Literal["system_calibration"]] = "system_calibration"
    """
    Type of dataset.
    """


@xarray_dataset_schema
class PhasedArrayXds:
    """Not specified. Not implemented."""

    pass


@xarray_dataset_schema
class DopplerXds:
    """Not specified. Not implemented."""

    pass


@xarray_dataset_schema
class VisibilityXds:
    """Main dataset for interferometric data"""

    # --- Required Coordinates ---
    time: Coordof[TimeCoordArray]
    """
    The time coordinate is the mid-point of the nominal sampling interval, as
    speciﬁed in the ``ms_v4.time.attrs['integration_time']`` (ms v2 interval).
    """
    baseline_id: Coordof[BaselineArray]
    """ Baseline ID """
    frequency: Coordof[FrequencyArray]
    """Center frequencies for each channel."""
    polarization: Coordof[PolarizationArray]
    """
    Labels for polarization types, e.g. ``['XX','XY','YX','YY']``, ``['RR','RL','LR','LL']``.
    """

    # --- Required data variables ---

    VISIBILITY: Dataof[VisibilityArray]
    """Complex visibilities, either simulated or measured by interferometer."""

    baseline_antenna1_name: Coordof[BaselineAntennaNameArray]
    """Antenna name for 1st antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_name``"""
    baseline_antenna2_name: Coordof[BaselineAntennaNameArray]
    """Antenna name for 2nd antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_name``"""

    # --- Required Attributes ---
    partition_info: Attr[PartitionInfoDict]
    observation_info: Attr[ObservationInfoDict]
    processor_info: Attr[ProcessorInfoDict]
    antenna_xds: Attr[AntennaXds]

    schema_version: Attr[str]
    """Semantic version of xradio data format"""
    creation_date: Attr[str]
    """Date visibility dataset was created . Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""

    type: Attr[Literal["visibility"]] = "visibility"
    """
    Dataset type
    """

    # --- Optional Coordinates ---
    polarization_mixed: Optional[Coord[tuple[BaselineId, Polarization], str]] = None
    """
    If the polarizations are not constant over baseline
    """
    uvw_label: Optional[Coordof[UvwLabelArray]] = None
    """ u,v,w """
    scan_number: Optional[Coord[Time, Union[numpy.int64, numpy.int32]]] = None
    """Arbitary scan number to identify data taken in the same logical scan."""

    # --- Optional data variables / arrays ---

    # VISIBILITY_CORRECTED: Optional[Dataof[VisibilityArray]] = None
    # VISIBILITY_MODEL: Optional[Dataof[VisibilityArray]] = None

    FLAG: Dataof[FlagArray] = None
    WEIGHT: Dataof[WeightArray] = None
    UVW: Dataof[UvwArray] = None
    EFFECTIVE_INTEGRATION_TIME: Optional[
        Data[
            Union[
                tuple[Time, BaselineId],
                tuple[Time, BaselineId, Frequency],
                tuple[Time, BaselineId, Frequency, Polarization],
            ],
            QuantityInSecondsArray,
        ]
    ] = None
    """
    The integration time, including the effects of missing data, in contrast to
    ``integration_time`` attribute of the ``time`` coordinate,
    see :py:class:`TimeArray`. (MS v2: ``exposure``).
    """
    TIME_CENTROID: Optional[Dataof[TimeSamplingArray]] = None
    """
    The time centroid of the visibility, includes the effects of missing data
    unlike the ``time`` coordinate, see :py:class:`TimeArray`.
    """
    TIME_CENTROID_EXTRA_PRECISION: Optional[Dataof[TimeSamplingArray]] = None
    """Additional precision for ``TIME_CENTROID``"""
    EFFECTIVE_CHANNEL_WIDTH: Optional[Dataof[FreqSamplingArray]] = None
    """The channel bandwidth that includes the effects of missing data."""
    FREQUENCY_CENTROID: Optional[Dataof[FreqSamplingArray]] = None
    """Includes the effects of missing data unlike ``frequency``."""

    # --- Optional Attributes ---
    pointing_xds: Optional[Attr[PointingXds]] = None
    system_calibration_xds: Optional[Attr[SystemCalibrationXds]] = None
    gain_curve_xds: Optional[Attr[GainCurveXds]] = None
    phase_calibration_xds: Optional[Attr[PhaseCalibrationXds]] = None
    weather_xds: Optional[Attr[WeatherXds]] = None
    phased_array_xds: Optional[Attr[PhasedArrayXds]] = None

    xradio_version: Optional[Attr[str]] = None
    """ Version of XRADIO used if converted from MSv2. """

    intent: Optional[Attr[str]] = None
    """Identifies the intention of the scan, such as to calibrate or observe a
    target. See :ref:`scan intents` for possible values.
    """
    data_description_id: Optional[Attr[str]] = None
    """
    The id assigned to this combination of spectral window and polarization setup.
    """


@xarray_dataset_schema
class SpectrumXds:
    """Main dataset for single dish data"""

    # --- Required Coordinates ---
    time: Coordof[TimeCoordArray]
    """
    The time coordinate is the mid-point of the nominal sampling interval, as
    speciﬁed in the ``ms_v4.time.attrs['integration_time']`` (ms v2 interval).
    """
    antenna_name: Coordof[AntennaNameArray]
    """ antenna_name """
    frequency: Coordof[FrequencyArray]
    """Center frequencies for each channel."""
    polarization: Coordof[PolarizationArray]
    """
    Labels for polarization types, e.g. ``['XX','XY','YX','YY']``, ``['RR','RL','LR','LL']``.
    """

    # --- Required data variables ---
    SPECTRUM: Dataof[SpectrumArray]
    """Single dish data, either simulated or measured by an antenna."""

    # --- Required Attributes ---
    partition_info: Attr[PartitionInfoDict]
    observation_info: Attr[ObservationInfoDict]
    processor_info: Attr[ProcessorInfoDict]
    antenna_xds: Attr[AntennaXds]

    schema_version: Attr[str]
    """Semantic version of xradio data format"""
    creation_date: Attr[str]
    """Date MSv4 was created . Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""

    type: Attr[Literal["spectrum"]] = "spectrum"
    """
    Dataset type
    """

    # --- Optional Coordinates ---
    polarization_mixed: Optional[Coord[tuple[AntennaName, Polarization], str]] = None
    """
    If the polarizations are not constant over baseline
    """
    scan_number: Optional[Coord[Time, Union[numpy.int64, numpy.int32]]] = None
    """Arbitary scan number to identify data taken in the same logical scan."""

    # SPECTRUM_CORRECTED: Optional[Dataof[SpectrumArray]] = None

    FLAG: Dataof[FlagArray] = None
    WEIGHT: Dataof[WeightArray] = None

    # --- Optional data variables / arrays ---
    EFFECTIVE_INTEGRATION_TIME: Optional[
        Data[
            Union[
                tuple[Time, AntennaName],
                tuple[Time, AntennaName, Frequency],
                tuple[Time, AntennaName, Frequency, Polarization],
            ],
            QuantityInSecondsArray,
        ]
    ] = None
    """
    The integration time, including the effects of missing data, in contrast to
    ``integration_time`` attribute of the ``time`` coordinate,
    see :py:class:`TimeArray`. (MS v2: ``exposure``).
    """
    TIME_CENTROID: Optional[Dataof[TimeSamplingArray]] = None
    """
    The time centroid of the visibility, includes the effects of missing data
    unlike the ``time`` coordinate, see :py:class:`TimeArray`.
    """
    TIME_CENTROID_EXTRA_PRECISION: Optional[Dataof[TimeSamplingArray]] = None
    """Additional precision for ``TIME_CENTROID``"""
    EFFECTIVE_CHANNEL_WIDTH: Optional[Dataof[FreqSamplingArray]] = None
    """The channel bandwidth that includes the effects of missing data."""
    FREQUENCY_CENTROID: Optional[Dataof[FreqSamplingArray]] = None
    """Includes the effects of missing data unlike ``frequency``."""

    # --- Optional Attributes ---
    pointing_xds: Optional[Attr[PointingXds]] = None
    system_calibration_xds: Optional[Attr[SystemCalibrationXds]] = None
    gain_curve_xds: Optional[Attr[GainCurveXds]] = None
    phase_calibration_xds: Optional[Attr[PhaseCalibrationXds]] = None
    weather_xds: Optional[Attr[WeatherXds]] = None
    phased_array_xds: Optional[Attr[PhasedArrayXds]] = None

    xradio_version: Optional[Attr[str]] = None
    """ Version of XRADIO used if converted from MSv2. """

    intent: Optional[Attr[str]] = None
    """Identifies the intention of the scan, such as to calibrate or observe a
    target. See :ref:`scan intents` for possible values.
    """
    data_description_id: Optional[Attr[str]] = None
    """
    The id assigned to this combination of spectral window and polarization setup.
    """
