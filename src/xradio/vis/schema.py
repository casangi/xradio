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
AntennaId = Literal["antenna_id"]
""" Antenna ID dimension """
StationId = Literal["station_id"]
""" Station ID dimension """
ReceptorId = Literal["receptor_id"]
""" Receptor ID dimension """
ReceptorName = Literal["receptor_label"]
""" Receptor name dimension """
BaselineId = Literal["baseline_id"]
""" Baseline ID dimension """
Frequency = Literal["frequency"]
""" Frequency dimension """
Polarization = Literal["polarization"]
""" Polarization dimension """
UvwLabel = Literal["uvw_label"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """
SkyDirLabel = Literal["sky_dir_label"]
""" Coordinate labels of sky directions (typically shape 2 and 'ra', 'dec') """
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
TimePolynomial = Literal["time_polynomial"]
""" For data that is represented as variable in time using Taylor expansion """
LineLabel = Literal["line_label"]
""" Line labels (for line names and variables). """

# Represents "no dimension", i.e. used for coordinates and data variables with
# zero dimensions.
ZD = tuple[()]

# Quantities


@xarray_dataarray_schema
class TimeArray:
    """
    Representation of a time quantity.

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

    scale: Attr[str] = "tai"
    """
    Time scale of data. Must be one of ``(‘tai’, ‘tcb’, ‘tcg’, ‘tdb’, ‘tt’, ‘ut1’, ‘utc’)``,
    see :py:class:`astropy.time.Time`
    """
    format: Attr[str] = "unix_tai"
    """Time representation and epoch, see :py:class:`TimeArray`."""

    type: Attr[str] = "time"
    units: Attr[list[str]] = ("s",)


@xarray_dataarray_schema
class SkyCoordArray:
    data: Data[Union[SkyDirLabel, SkyPosLabel], float]

    type: Attr[str] = "sky_coord"
    units: Attr[list[str]] = ("rad", "rad")
    frame: Attr[str] = ""
    """
    From fixvis docs: clean and the im tool ignore the reference frame
    claimed by the UVW column (it is often mislabelled as ITRF when it is
    really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame
    as the phase tracking center. calcuvw does not yet force the UVW column and
    field centers to use the same reference frame! Blank = use the phase
    tracking frame of vis.
    """


@xarray_dataarray_schema
class SkyCoordOffsetArray:
    data: Data[Union[SkyDirLabel, SkyPosLabel], float]

    type: Attr[str] = "sky_coord"
    units: Attr[list[str]] = ("rad", "rad")


@xarray_dataarray_schema
class QuantityArray:
    """
    Anonymous quantity, possibly with associated units

    Often used for distances / differences (integration time, channel width etcetera).
    """

    data: Data[ZD, float]

    units: Attr[list[str]]
    type: Attr[str] = "quantity"


# Coordinates / Axes
@xarray_dataarray_schema
class TimeCoordArray:
    """Data model of visibility time axis. See also :py:class:`TimeArray`."""

    data: Data[Time, float]
    """
    Time, expressed in seconds since the epoch (see ``scale`` &
    ``format``), see also see :py:class:`TimeArray`.
    """

    integration_time: Optional[Attr[QuantityArray]] = None
    """ The nominal sampling interval (ms v2). Units of seconds. """
    effective_integration_time: Optional[Attr[str]] = None
    """
    Name of data array that contains the integration time that includes
    the effects of missing data.
    """

    type: Attr[str] = "time"
    """ Coordinate type. Should be ``"time"``. """
    units: Attr[list[str]] = ("s",)
    """ Units to associate with axis"""
    scale: Attr[str] = "tai"
    """ Astropy time scales, see :py:class:`TimeArray` """
    format: Attr[str] = "unix"
    """ Astropy format, see :py:class:`TimeArray`"""
    long_name: Optional[Attr[str]] = "Observation Time"
    """ Long-form name to use for axis"""


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

    time: Optional[Coordof[TimeCoordArray]]
    """Midpoint of time for which this set of parameters is accurate"""

    line_label: Optional[Coord[LineLabel, str]]
    """ Line labels (for line names and variables). """

    line_names: Optional[Coord[Union[tuple[LineLabel], tuple[Time, LineLabel]], str]]
    """ Line names (e.g. v=1, J=1-0, SiO). """

    FIELD_PHASE_CENTER: Optional[Data[Union[ZD, Time], SkyCoordOffsetArray]]
    """
    Offset from the SOURCE_DIRECTION that gives the direction of phase
    center for which the fringes have been stopped-that is a point source in
    this direction will produce a constant measured phase (page 2 of
    https://articles.adsabs.harvard.edu/pdf/1999ASPC..180...79F). For
    conversion from MSv2, frame refers column keywords by default. If frame
    varies with field, it refers DelayDir_Ref column instead.
    """
    FIELD_DELAY_CENTER: Optional[
        Data[Union[tuple[SkyDirLabel], tuple[Time, SkyDirLabel]], numpy.float64]
    ]
    """
    Offset from the SOURCE_DIRECTION that gives the direction of delay
    center where coherence is maximized by inserting delay into one element of
    an interferometer to compensate for the geometrical and instrumental
    differential delay. (For conversion from MSv2, frame refers column keywords
    by default. If frame varies with field, it refers PhaseDir_Ref column
    instead.)
    """

    SOURCE_LOCATION: Optional[
        Data[
            Union[
                tuple[SkyPosLabel],
                tuple[Time, SkyPosLabel],
                tuple[SkyDirLabel],
                tuple[Time, SkyDirLabel],
            ],
            numpy.float64,
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
        Data[Union[tuple[LineLabel], tuple[Time, LineLabel]], numpy.float64]
    ]
    """ Rest frequencies for the transitions. """

    LINE_SYSTEMIC_VELOCITY: Optional[
        Data[Union[tuple[LineLabel], tuple[Time, LineLabel]], numpy.float64]
    ]
    """ Systemic velocity at reference """

    SOURCE_RADIAL_VELOCITY: Optional[Data[tuple[Time], numpy.float64]]
    """ CASA Table Cols: RadVel. Geocentric distance rate """

    NORTH_POLE_POSITION_ANGLE: Optional[Data[tuple[Time], numpy.float64]]
    """ CASA Table cols: NP_ang, "Targets' apparent north-pole position angle (counter-clockwise with respect to direction of true-of-date reference-frame north pole) and angular distance from the sub-observer point (center of disc) at print time. A negative distance indicates the north-pole is on the hidden hemisphere." https://ssd.jpl.nasa.gov/horizons/manual.html : 17. North pole position angle & distance from disc center. """

    NORTH_POLE_ANGULAR_DISTANCE: Optional[Data[tuple[Time], numpy.float64]]
    """ CASA Table cols: NP_dist, "Targets' apparent north-pole position angle (counter-clockwise with respect to direction of true-of date reference-frame north pole) and angular distance from the sub-observer point (center of disc) at print time. A negative distance indicates the north-pole is on the hidden hemisphere."https://ssd.jpl.nasa.gov/horizons/manual.html : 17. North pole position angle & distance from disc center. """

    SUB_OBSERVER_DIRECTION: Optional[
        Data[tuple[Time, SphericalDirLabel], numpy.float64]
    ]
    """ CASA Table cols: DiskLong, DiskLat. "Apparent planetodetic longitude and latitude of the center of the target disc seen by the OBSERVER at print-time. This is not exactly the same as the "nearest point" for a non-spherical target shape (since the center of the disc might not be the point closest to the observer), but is generally very close if not a very irregular body shape. The IAU2009 rotation models are used except for Earth and MOON, which use higher-precision models. For the gas giants Jupiter, Saturn, Uranus and Neptune, IAU2009 longitude is based on the "System III" prime meridian rotation angle of the magnetic field. By contrast, pole direction (thus latitude) is relative to the body dynamical equator. There can be an offset between the magnetic pole and the dynamical pole of rotation. Down-leg light travel-time from target to observer is taken into account. Latitude is the angle between the equatorial plane and perpendicular to the reference ellipsoid of the body and body oblateness thereby included. The reference ellipsoid is an oblate spheroid with a single flatness coefficient in which the y-axis body radius is taken to be the same value as the x-axis radius. Whether longitude is positive to the east or west for the target will be indicated at the end of the output ephemeris." https://ssd.jpl.nasa.gov/horizons/manual.html : 14. Observer sub-longitude & sub-latitude """

    SUB_SOLAR_POSITION: Optional[Data[tuple[Time, SphericalPosLabel], numpy.float64]]
    """ CASA Table cols: Sl_lon, Sl_lat, r. "Heliocentric distance along with "Apparent sub-solar longitude and latitude of the Sun on the target. The apparent planetodetic longitude and latitude of the center of the target disc as seen from the Sun, as seen by the observer at print-time.  This is _NOT_ exactly the same as the "sub-solar" (nearest) point for a non-spherical target shape (since the center of the disc seen from the Sun might not be the closest point to the Sun), but is very close if not a highly irregular body shape.  Light travel-time from Sun to target and from target to observer is taken into account.  Latitude is the angle between the equatorial plane and the line perpendicular to the reference ellipsoid of the body. The reference ellipsoid is an oblate spheroid with a single flatness coefficient in which the y-axis body radius is taken to be the same value as the x-axis radius. Uses IAU2009 rotation models except for Earth and Moon, which uses a higher precision models. Values for Jupiter, Saturn, Uranus and Neptune are Set III, referring to rotation of their magnetic fields.  Whether longitude is positive to the east or west for the target will be indicated at the end of the output ephemeris." https://ssd.jpl.nasa.gov/horizons/manual.html : 15. Solar sub-longitude & sub-latitude  """

    HELIOCENTRIC_RADIAL_VELOCITY: Optional[Data[tuple[Time], numpy.float64]]
    """ CASA Table cols: rdot."The Sun's apparent range-rate relative to the target, as seen by the observer. A positive "rdot" means the target was moving away from the Sun, negative indicates movement toward the Sun." https://ssd.jpl.nasa.gov/horizons/manual.html : 19. Solar range & range-rate (relative to target) """

    OBSERVER_PHASE_ANGLE: Optional[Data[tuple[Time], numpy.float64]]
    """ CASA Table cols: phang.""phi" is the true PHASE ANGLE at the observers' location at print time. "PAB-LON" and "PAB-LAT" are the FK4/B1950 or ICRF/J2000 ecliptic longitude and latitude of the phase angle bisector direction; the outward directed angle bisecting the arc created by the apparent vector from Sun to target center and the astrometric vector from observer to target center. For an otherwise uniform ellipsoid, the time when its long-axis is perpendicular to the PAB direction approximately corresponds to lightcurve maximum (or maximum brightness) of the body. PAB is discussed in Harris et al., Icarus 57, 251-258 (1984)." https://ssd.jpl.nasa.gov/horizons/manual.html : Phase angle and bisector """

    OBSERVER_POSITION: Optional[
        Data[Union[tuple[EllipsoidPosLabel], tuple[CartesianPosLabel]], numpy.float64]
    ]
    """ Observer location. """

    # --- Attributes ---
    DOPPLER_SHIFT_VELOCITY: Optional[Attr[numpy.float64]]
    """ Velocity definition of the Doppler shift, e.g., RADIO or OPTICAL velocity in m/s """

    source_model_url: Optional[Attr[str]]
    """URL to access source model"""
    ephemeris_name: Optional[Attr[str]]
    """The name of the ephemeris. For example DE430.

    This can be used with Astropy solar_system_ephemeris.set('DE430'), see
    https://docs.astropy.org/en/stable/coordinates/solarsystem.html.
    """
    is_ephemeris: Attr[bool] = False

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
class SpectralCoordArray:
    data: Data[ZD, float]

    frame: Attr[str] = "gcrs"
    """Astropy time scales."""

    type: Attr[str] = "frequency"
    units: Attr[list[str]] = ("Hz",)


@xarray_dataarray_schema
class EarthLocationArray:
    data: Data[CartesianPosLabel, float]

    ellipsoid: Attr[str]
    """
    ITRF makes use of GRS80 ellipsoid and WGS84 makes use of WGS84 ellipsoid
    """
    units: Attr[list[str]] = ("m", "m", "m")
    """
    If the units are a list of strings then it must be the same length as
    the last dimension of the data array. This allows for having different
    units in the same data array,for example geodetic coordinates could use
    ``['rad','rad','m']``.
    """


@dict_schema
class ObservationInfoDict:
    observer: list
    """List of observer names."""
    project: str
    """Project Code/Project_UID"""
    release_data: str
    """Project release date. This is the date on which the data may become
    public. Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""
    execution_block_id: Optional[str]
    """ ASDM: Indicates the position of the execution block in the project
    (sequential numbering starting at 1).  """
    execution_block_number: Optional[str]
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


@xarray_dataarray_schema
class BaselineArray:
    """TODO: documentation"""

    data: Data[BaselineId, Union[numpy.int64, numpy.int32]]
    """Unique id for each baseline."""
    long_name: Optional[Attr[str]] = "Baseline ID"


@xarray_dataarray_schema
class BaselineAntennaArray:
    data: Data[BaselineId, Union[numpy.int64, numpy.int32]]
    """
    Antenna id for an antenna in a baseline. Maps to ``attrs['antenna_xds'].antenna_id``
    in :py:class:`VisibilityXds`
    """
    long_name: Optional[Attr[str]] = "Baseline Antenna ID"


@xarray_dataset_schema
class DopplerXds:
    # TODO
    pass


@xarray_dataarray_schema
class FrequencyArray:
    """TODO: documentation"""

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
    channel_width: Attr[QuantityArray]  # Not SpectralCoord, as it is a difference
    """ The nominal channel bandwidth. Same units as data array (see units key). """
    doppler: Optional[Attr[DopplerXds]]
    """ Doppler tracking information """

    type: Attr[str] = "spectral_coord"
    """ Coordinate type. Should be ``"spectral_coord"``. """
    long_name: Optional[Attr[str]] = "Frequency"
    """ Long-form name to use for axis"""
    units: Attr[list[str]] = ("Hz",)
    """ Units to associate with axis"""
    frame: Attr[str] = "icrs"
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
class VisibilityArray:
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization],
        Union[numpy.complex64, numpy.complex128],
    ]

    time: Coord[tuple[()], TimeCoordArray]
    baseline_id: Coord[tuple[()], BaselineArray]
    polarization: Coord[tuple[()], PolarizationArray]
    frequency: Coord[tuple[()], FrequencyArray]
    time: Coord[ZD, TimeCoordArray]
    baseline_id: Coord[ZD, BaselineArray]
    polarization: Coord[ZD, PolarizationArray]
    frequency: Coord[ZD, FrequencyArray]

    field_and_source_xds: Attr[FieldSourceXds]
    long_name: Optional[Attr[str]] = "Visibility values"
    """ Long-form name to use for axis. Should be ``"Visibility values"``"""
    units: Attr[list[str]] = ("Jy",)


@xarray_dataarray_schema
class SpectrumArray:
    """Definition of xr.DataArray for SPECTRUM data (single dish)"""

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization],
        Union[numpy.float64, numpy.float32, numpy.float16],
    ]
    time: Coord[tuple[()], TimeCoordArray]

    # in the spreadsheet this is antenna_id:
    # antenna_id: Coord[AntennaId, Union[int, numpy.int32]]
    baseline_id: Coord[tuple[()], BaselineArray]

    polarization: Coord[tuple[()], PolarizationArray]
    frequency: Coord[tuple[()], FrequencyArray]
    field_and_source_xds: Attr[FieldSourceXds]
    long_name: Optional[Attr[str]] = "Spectrum values"
    """ Long-form name to use for axis. Should be ``"Spectrum values"``"""
    units: Attr[list[str]] = ("Jy",)


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
        ],
        bool,
    ]
    time: Coordof[TimeCoordArray]
    baseline_id: Coordof[BaselineArray]
    frequency: Coordof[FrequencyArray]
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Visibility flags"


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
        ],
        Union[numpy.float16, numpy.float32, numpy.float64],
    ]
    """Visibility weights"""
    time: Coordof[TimeCoordArray]
    baseline_id: Coordof[BaselineArray]
    frequency: Optional[Coordof[FrequencyArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Visibility weights"


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
        Union[
            tuple[Time, BaselineId, Frequency, Polarization, UvwLabel],
            tuple[Time, BaselineId, Frequency, UvwLabel],
            tuple[Time, BaselineId, UvwLabel],
        ],
        Union[
            numpy.float16,
            numpy.float32,
            numpy.float64,
        ],
    ]
    """Baseline coordinates from ``baseline_antenna2_id`` to ``baseline_antenna1_id``"""
    time: Coordof[TimeCoordArray]
    baseline_id: Coordof[BaselineArray]
    frequency: Optional[Coordof[FrequencyArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    uvw_label: Coordof[UvwLabelArray] = ("u", "v", "w")
    long_name: Optional[Attr[str]] = "Baseline coordinates"
    """ Long-form name to use for axis. Should be ``"Baseline coordinates``"""
    units: Attr[list[str]] = ("m",)


@xarray_dataarray_schema
class TimeSamplingArray:
    """TODO: documentation"""

    data: Data[
        Union[
            tuple[Time, BaselineId, Frequency, Polarization],
            tuple[Time, BaselineId, Frequency],
            tuple[Time, BaselineId],
        ],
        float,
    ]

    time: Coordof[TimeCoordArray]
    baseline_id: Coordof[BaselineArray]
    frequency: Optional[Coordof[FrequencyArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None

    scale: Attr[str] = "tai"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[str] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """

    long_name: Optional[Attr[str]] = "Time sampling data"
    units: Attr[list[str]] = ("s",)


@xarray_dataarray_schema
class FreqSamplingArray:
    """TODO: documentation"""

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
    :py:class:`VisibilityXds`.
    """
    frequency: Coordof[FrequencyArray]
    time: Optional[Coordof[TimeCoordArray]] = None
    baseline_id: Optional[Coordof[BaselineArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Frequency sampling data"
    units: Attr[list[str]] = ("Hz",)
    frame: Attr[str] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """


# Data Sets


@xarray_dataset_schema
class AntennaXdsEmptyWhileRevampedTODO:
    """Minimal placeholder so that it shows in the main xds
    docs. TODO: update AntennaXds once review moves on
    """

    antenna_name: Coord[Literal["antenna_name"], str]
    receptor_label: Optional[Coord[ReceptorName, str]]
    cartesian_pos_label: Coord[CartesianPosLabel, str]
    sky_dir_label: Optional[Coord[SkyDirLabel, str]]
    gain_curve_time: Optional[Coord[Literal["gain_curve_time"], numpy.float64]]
    poly_term: Optional[Coord[Literal["poly_term"], numpy.int64]]
    phase_cal_time: Optional[Coord[Literal["phase_cal_time"], numpy.float64]]
    tone_label: Optional[Coord[Literal["tone_label"], numpy.int64]]


@xarray_dataset_schema
class AntennaXds:
    # --- Coordinates ---
    antenna_id: Coord[AntennaId, Union[int, numpy.int32]]
    """Antenna ID"""
    name: Coord[AntennaId, str]

    """Antenna name."""
    station: Coord[AntennaId, str]
    """Name of the station pad (relevant to arrays with moving antennas)."""
    antenna_type: Optional[Coord[AntennaId, str]]
    """Antenna type.
    
    Reserved keywords include: (``GROUND-BASED`` - conventional
    antennas; ``SPACE-BASED`` - orbiting antennas; ``TRACKING-STN`` - tracking
    stations)."""
    mount: Coord[AntennaId, str]
    """Mount type of the antenna.

    Reserved keywords include: (``EQUATORIAL`` - equatorial mount; ``ALTAZ`` -
    azimuth-elevation mount; ``X-Y`` - x-y mount; ``SPACE-HALCA`` - specific
    orientation model.)"""
    observatory: Optional[Coord[AntennaId, str]]
    """Support for VLBI"""
    receptor_name: Optional[Coord[ReceptorName, str]]
    """Names of receptors"""
    cartesian_pos_label: Coord[CartesianPosLabel, str]
    """Coordinate dimension of earth location data (typically shape 3 and 'x', 'y', 'z')"""
    sky_dir_label: Optional[Coord[SkyDirLabel, str]]
    """Coordinate dimension of sky coordinate data (possibly shape 2 and 'RA', "Dec")"""

    # --- Data variables ---
    ANTENNA_POSITION: Data[AntennaId, EarthLocationArray]
    """
    In a right-handed frame, X towards the intersection of the equator and
    the Greenwich meridian, Z towards the pole.
    """
    ANTENNA_FEED_OFFSET: Data[tuple[AntennaId, CartesianPosLabel], QuantityArray]
    """
    Offset of feed relative to position (``Antenna_Table.offset + Feed_Table.position``).
    """
    ANTENNA_DISH_DIAMETER: Data[AntennaId, QuantityArray]
    """
    Nominal diameter of dish, as opposed to the effective diameter.
    """
    BEAM_OFFSET: Optional[Data[AntennaId, SkyCoordArray]]
    """
    Beam position offset, as defined on the sky but in the antenna
    reference frame.
    """
    RECEPTOR_ANGLE: Optional[Data[tuple[AntennaId, ReceptorName], QuantityArray]]
    """
    Polarization reference angle. Converts into parallactic angle in the sky domain.
    """
    FOCUS_LENGTH: Optional[Data[AntennaId, QuantityArray]]
    """
    Focus length. As defined along the optical axis of the antenna.
    """
    ARRAY_CENTER: Optional[Data[AntennaId, EarthLocationArray]]
    EFFECTIVE_DISH_DIAMETER: Optional[Data[AntennaId, QuantityArray]]

    # --- Attributes ---
    telescope_name: Optional[Attr[str]]
    """
    From MS v2 observation table
    """
    type: Attr[str] = "antenna"
    """
    Type of dataset. Expected to be ``antenna``
    """


@xarray_dataset_schema
class WeatherXds:
    """TODO: largely incomplete"""

    station_id: Coord[StationId, numpy.int64]
    """ Station identifier """
    time: Coord[Time, numpy.float64]
    """ Mid-point of the time interval """


@xarray_dataset_schema
class PointingXds:
    time: Coordof[TimeCoordArray]
    """
    Mid-point of the time interval for which the information in this row is
    valid.  Required to use the same time measure reference as in visibility dataset
    """
    antenna_id: Coord[AntennaId, Union[int, numpy.int32]]
    """
    Antenna identifier, as specified by baseline_antenna1/2_id in visibility dataset
    """
    sky_dir_label: Coord[SkyDirLabel, str]
    """
    Direction labels.
    """

    BEAM_POINTING: Data[
        Union[tuple[Time, AntennaId, TimePolynomial], tuple[Time, AntennaId]],
        SkyCoordArray,
    ]
    """
    Antenna pointing direction, optionally expressed as polynomial coefficients. DIRECTION in MSv3.
    """
    DISH_MEASURED_POINTING: Optional[Data[tuple[Time, AntennaId], SkyCoordArray]]
    """
    The current encoder values on the primary axes of the mount type for
    the antenna. ENCODER in MSv3.
    """
    OVER_THE_TOP: Optional[Data[tuple[Time, AntennaId], bool]]


@xarray_dataset_schema
class SpectralCoordXds:
    # TODO
    pass


@xarray_dataset_schema
class PhasedArrayXds:
    # TODO
    pass


@xarray_dataset_schema
class SystemCalibrationXds:
    """TODO: largely incomplete"""

    antenna_id: Coord[AntennaId, Union[int, numpy.int32]]
    """ Antenna identifier """
    time: Coord[Time, numpy.float64]
    """ Midpoint of time for which this set of parameters is accurate """
    receptor_id: Coord[ReceptorId, numpy.float64]
    """  """


@xarray_dataset_schema
class VisibilityXds:
    """TODO: documentation"""

    # --- Required Coordinates ---
    time: Coordof[TimeCoordArray]
    """
    The time coordinate is the mid-point of the nominal sampling interval, as
    speciﬁed in the ``ms_v4.time.attrs['integration_time']`` (ms v2 interval).
    """
    baseline_id: Coordof[BaselineArray]
    frequency: Coordof[FrequencyArray]
    """Center frequencies for each channel."""
    polarization: Coordof[PolarizationArray]
    """
    Labels for polarization types, e.g. ``['XX','XY','YX','YY']``, ``['RR','RL','LR','LL']``.
    """
    uvw_label: Optional[Coordof[UvwLabelArray]]

    # --- Required Attributes ---
    # TODO: on hold while antenna_xds is reviewed/ updated
    # antenna_xds: Attr[AntennaXds]
    # antenna_xds: Attr[AntennaXdsEmptyWhileRevampedTODO]

    # --- Optional Coordinates ---
    baseline_antenna1_id: Optional[Coordof[BaselineAntennaArray]] = None
    """Antenna id for 1st antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""
    baseline_antenna2_id: Optional[Coordof[BaselineAntennaArray]] = None
    """Antenna id for 2nd antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""
    scan_id: Optional[Coord[Time, int]] = None
    """Arbitary scan number to identify data taken in the same logical scan."""

    # --- Required data variables ---

    # --- Optional data variables / arrays ---

    # Either VISIBILITY (interferometry) or SPECTRUM (single-dish)
    VISIBILITY: Optional[Dataof[VisibilityArray]] = None
    """Complex visibilities, either simulated or measured by interferometer."""
    SPECTRUM: Optional[Dataof[SpectrumArray]] = None
    """Single dish data, either simulated or measured by an antenna."""

    VISIBILITY_CORRECTED: Optional[Dataof[VisibilityArray]] = None
    VISIBILITY_MODEL: Optional[Dataof[VisibilityArray]] = None
    SPECTRUM_CORRECTED: Optional[Dataof[SpectrumArray]] = None

    FLAG: Optional[Dataof[FlagArray]] = None
    WEIGHT: Optional[Dataof[WeightArray]] = None
    UVW: Optional[Dataof[UvwArray]] = None
    EFFECTIVE_INTEGRATION_TIME: Optional[
        Data[
            Union[
                tuple[Time, BaselineId],
                tuple[Time, BaselineId, Frequency],
                tuple[Time, BaselineId, Frequency, Polarization],
            ],
            QuantityArray,
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
    observation_info: Optional[Attr[ObservationInfoDict]] = None
    processor_info: Optional[Attr[ProcessorInfoDict]] = None
    weather_xds: Optional[Attr[WeatherXds]] = None
    pointing_xds: Optional[Attr[PointingXds]] = None
    phased_array_xds: Optional[Attr[PhasedArrayXds]] = None
    system_calibration_xds: Optional[Attr[SystemCalibrationXds]] = None

    version: Optional[Attr[str]] = None  # TODO:
    """Semantic version of xradio data format"""
    creation_date: Optional[Attr[str]] = None
    """Date visibility dataset was created . Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""
    intent: Optional[Attr[str]] = None
    """Identifies the intention of the scan, such as to calibrate or observe a
    target. See :ref:`scan intents` for possible values.
    """
    data_description_id: Optional[Attr[str]] = None
    """
    The id assigned to this combination of spectral window and polarization setup.
    """

    type: Attr[str] = "visibility"
    """
    Dataset type
    """
