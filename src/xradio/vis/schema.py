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
ReceptorName = Literal["receptor_name"]
""" Receptor name dimension """
BaselineId = Literal["baseline_id"]
""" Baseline ID dimension """
Frequency = Literal["frequency"]
""" Frequency dimension """
Polarization = Literal["polarization"]
""" Polarization dimension """
UvwLabel = Literal["uvw_label"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """
XyzLabel = Literal["xyz_label"]
""" Coordinate dimension of earth location data (typically shape 3 and 'x', 'y', 'z')"""
TimePolynomial = Literal["time_polynomial"]
""" For data that is represented as variable in time using Taylor expansion """
SkyCoordLabel = Literal["sky_coord_label"]
""" Unlabeled axis """


# Plain data class models
@dict_schema
class SourceInfoDict:
    # TODO
    pass


@xarray_dataarray_schema
class TimeArray:
    data: Data[Time, float]

    scale: Attr[str] = "tai"
    """Astropy time scales."""
    format: Attr[str] = "unix"
    """Seconds from 1970-01-01 00:00:00 UTC"""

    type: Attr[str] = "time"
    units: Attr[list] = ("s",)


@xarray_dataarray_schema
class SkyCoordArray:
    data: Data[SkyCoordLabel, float]

    type: Attr[str] = "sky_coord"
    units: Attr[list] = ("rad", "rad")
    frame: Attr[str] = ""
    """
    From fixvis docs: clean and the im tool ignore the reference frame
    claimed by the UVW column (it is often mislabelled as ITRF when it is
    really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame
    as the phase tracking center. calcuvw does not yet force the UVW column and
    field centers to use the same reference frame! Blank = use the phase
    tracking frame of vis.
    """


@dict_schema
class FieldInfoDict:
    """
    Field positions for each source.

    Defines a field position on the sky. For interferometers, this is the correlated field position.
    For single dishes, this is the nominal pointing direction.
    """

    name: str
    """Field name."""
    field_id: int
    """Field id"""
    code: str
    """Field code indicating special characteristics of the field; user specified."""
    time_reference: Optional[TimeArray]
    """
    Time reference for the directions and rates. When used in :py:class:`VisibilityXds` should match 
    the scale and format given for ``time`` (see :py:class:`TimeArray`).
    """
    delay_direction: SkyCoordArray


@xarray_dataarray_schema
class QuantityArray:
    """
    Anonymous quantity
    """

    data: Data[tuple[()], float]

    type: Attr[str]
    units: Attr[list]


@xarray_dataarray_schema
class SpectralCoordArray:
    data: Data[tuple[()], float]

    frame: Attr[str] = "gcrs"
    """Astropy time scales."""

    type: Attr[str] = "frequency"
    units: Attr[list] = ("Hz",)


@xarray_dataarray_schema
class EarthLocationArray:
    data: Data[XyzLabel, float]

    ellipsoid: Attr[str]
    """
    ITRF makes use of GRS80 ellipsoid and WGS84 makes use of WGS84 ellipsoid
    """
    units: Attr[list] = ("m", "m", "m")
    """
    If the units are a list of strings then it must be the same length as
    the last dimension of the data array. This allows for having different
    units in the same data array,for example geodetic coordinates could use
    ``['rad','rad','m']``.
    """


@dict_schema
class ObservationInfoDict:
    observer: List[str]
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


# Coordinates / Axes
@xarray_dataarray_schema
class TimeArray:
    """Data model of time axis"""

    data: Data[Time, float]
    """ Time, expressed in SI seconds since the epoch (see ``scale`` & ``format``). """

    integration_time: Attr[Optional[TimeArray]] = None
    """ The nominal sampling interval (ms v2). Units of seconds. """
    effective_integration_time: Attr[Optional[TimeArray]] = None
    """ Name of data array that contains the integration time that includes the effects of missing data. """

    type: Attr[str] = "time"
    """ Coordinate type. Should be ``"time"``. """
    units: Attr[list[str]] = ("s",)
    """ Units to associate with axis"""
    scale: Attr[str] = "tai"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[str] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """
    long_name: Optional[Attr[str]] = "Observation Time"
    """ Long-form name to use for axis"""


@xarray_dataarray_schema
class AntennaArray:
    data: Data[AntennaId, int]
    """
    Antenna id of an antenna. Maps to ``antenna_id``
    in :py:class:`AntennaXds`.
    """
    long_name: Optional[Attr[str]] = "Antenna ID"


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
    frequency_group_name: Attr[str]
    """ Name associated with frequency group - needed for multi-band VLBI fringe-fitting."""
    reference_frequency: Attr[SpectralCoordArray]
    """ A frequency representative of the spectral window, usually the sky
    frequency corresponding to the DC edge of the baseband. Used by the calibration
    system if a ﬁxed scaling frequency is required or in algorithms to identify the
    observing band. """
    channel_width: Attr[SpectralCoordArray]
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
    time: Coord[tuple[()], TimeArray]
    baseline_id: Coord[tuple[()], BaselineArray]
    frequency: Coord[tuple[()], FrequencyArray]
    polarization: Coord[tuple[()], PolarizationArray]
    field_info: Attr[FieldInfoDict]
    long_name: Optional[Attr[str]] = "Visibility values"
    """ Long-form name to use for axis. Should be ``"Visibility values"``"""
    units: Attr[list] = ("Jy",)


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
    time: Coordof[TimeArray]
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
    time: Coordof[TimeArray]
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
    time: Coordof[TimeArray]
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

    time: Coordof[TimeArray]
    baseline_id: Coordof[BaselineArray]
    frequency: Optional[Coordof[FrequencyArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None

    scale: Attr[str] = "tai"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[str] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """

    long_name: Optional[Attr[str]] = "Time sampling data"
    units: Attr[str] = "s"


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
    time: Optional[Coordof[TimeArray]] = None
    baseline_id: Optional[Coordof[BaselineArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Frequency sampling data"
    units: Attr[str] = "Hz"
    frame: Attr[str] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """


# Data Sets


@xarray_dataset_schema
class AntennaXds:
    # --- Coordinates ---
    antenna_id: Coordof[AntennaArray]
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
    xyz_label: Coord[XyzLabel, str]
    """Coordinate dimension of earth location data (typically shape 3 and 'x', 'y', 'z')"""
    sky_coord_label: Optional[Coord[SkyCoordLabel, str]]
    """Coordinate dimension of sky coordinate data (possibly shape 2 and 'RA', "Dec")"""

    # --- Data variables ---
    POSITION: Data[AntennaId, EarthLocationArray]
    """
    In a right-handed frame, X towards the intersection of the equator and
    the Greenwich meridian, Z towards the pole.
    """
    FEED_OFFSET: Data[tuple[AntennaId, XyzLabel], QuantityArray]
    """
    Offset of feed relative to position (``Antenna_Table.offset + Feed_Table.position``).
    """
    DISH_DIAMETER: Data[AntennaId, QuantityArray]
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
class PointingXds:
    time: Coordof[TimeArray]
    """
    Mid-point of the time interval for which the information in this row is
    valid.  Required to use the same time measure reference as in visibility dataset
    """
    antenna_id: Coordof[AntennaArray]
    """
    Antenna identifier, as specified by baseline_antenna1/2_id in visibility dataset
    """
    sky_coord_label: Coord[SkyCoordLabel, str]
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
class SourceXds:
    # TODO
    pass


@xarray_dataset_schema
class PhasedArrayXds:
    # TODO
    pass


@xarray_dataset_schema
class VisibilityXds:
    """TODO: documentation"""

    # --- Required Coordinates ---
    time: Coordof[TimeArray]
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

    # --- Required data variables ---
    VISIBILITY: Dataof[VisibilityArray]

    # --- Required Attributes ---
    antenna_xds: Attr[AntennaXds]

    # --- Optional Coordinates ---
    baseline_antenna1_id: Optional[Coordof[BaselineAntennaArray]] = None
    """Antenna id for 1st antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""
    baseline_antenna2_id: Optional[Coordof[BaselineAntennaArray]] = None
    """Antenna id for 2nd antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""
    scan_id: Optional[Coord[Time, int]] = None
    """Arbitary scan number to identify data taken in the same logical scan."""

    # --- Optional data variables / arrays ---
    """Complex visibilities, either simulated or measured by interferometer."""
    FLAG: Optional[Dataof[FlagArray]] = None
    WEIGHT: Optional[Dataof[WeightArray]] = None
    UVW: Optional[Dataof[UvwArray]] = None
    EFFECTIVE_INTEGRATION_TIME: Optional[Dataof[TimeSamplingArray]] = None
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
    source_xds: Optional[Attr[SourceXds]] = None
    pased_array_xds: Optional[Attr[PhasedArrayXds]] = None
    observation_info: Optional[Attr[ObservationInfoDict]] = None
    observation_info: Optional[Attr[ProcessorInfoDict]] = None

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
