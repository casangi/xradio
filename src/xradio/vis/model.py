from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union
from xradio.schema.bases import AsDataArray, AsDataset
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
import numpy

# Dimensions
Time = Literal["time"]
""" Observation time dimension """
AntennaId = Literal["antenna_id"]
""" Antenna ID dimension """
BaselineId = Literal["baseline_id"]
""" Baseline ID dimension """
Frequency = Literal["frequency"]
""" Frequency dimension """
Polarization = Literal["polarization"]
""" Polarization dimension """
UvwLabel = Literal["uvw_label"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """
DirectionLabel = Literal["direction_label"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """
TimePolynomial = Literal["time_polynomial"]
""" For data that is represented as variable in time using Taylor expansion """


# Plain data class models
@dataclass
class SourceInfoDict:
    # TODO
    pass


@dataclass
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
    time_reference: str
    """
    Time reference for the directions and rates. When used in :py:class:`VisibilityXds` should match 
    the scale and format given for ``time`` (see :py:class:`TimeAxis`).
    """
    delay_direction: SkyCoordDict

    # TODO
    pass


@dataclass
class QuantityDict:
    # TODO
    pass


@dataclass
class SkyCoordDict:
    # TODO
    pass


@dataclass(frozen=True)
class ObservationInfoDict:
    observer: List[str]
    """List of observer names."""
    project: str
    """Project Code/Project_UID"""
    release_data: str
    """Project release date. This is the date on which the data may become public."""
    execution_block_id: Optional[str]
    execution_block_number: Optional[str]
    execution_block_UID: Optional[str]
    session_reference: Optional[str]
    observing_script: Optional[str]


# Coordinates / Axes
@dataclass(frozen=True)
class TimeAxis(AsDataArray):
    """Data model of time axis"""

    data: Data[Time, float]
    """ Time, expressed in SI seconds since the epoch (see ``scale`` & ``format``). """

    integration_time: Attr[Optional[QuantityDict]] = None
    """ The nominal sampling interval (ms v2). Units of seconds. """
    effective_integration_time: Attr[Optional[QuantityDict]] = None
    """ Name of data array that contains the integration time that includes the effects of missing data. """

    long_name: Attr[str] = "Observation Time"
    """ Long-form name to use for axis"""
    type: Attr[str] = "time"
    """ Coordinate type. Should be ``"time"``. """
    units: Attr[tuple[str]] = ("s",)
    """ Units to associate with axis"""
    scale: Attr[str] = "tai"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[str] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """


@dataclass(frozen=True)
class AntennaAxis(AsDataArray):
    data: Data[AntennaId, int]
    """
    Antenna id of an antenna. Maps to ``antenna_id``
    in :py:class:`AntennaXds`.
    """
    long_name: Attr[str] = "Antenna ID"


@dataclass(frozen=True)
class BaselineAxis(AsDataArray):
    """TODO: documentation"""

    data: Data[BaselineId, int]
    """Unique id for each baseline."""
    long_name: Attr[str] = "Baseline ID"


@dataclass(frozen=True)
class BaselineAntennaAxis(AsDataArray):
    data: Data[BaselineId, int]
    """
    Antenna id for an antenna in a baseline. Maps to ``attrs['antenna_xds'].antenna_id``
    in :py:class:`VisibilityXds`
    """
    long_name: Attr[str] = "Baseline Antenna ID"


@dataclass(frozen=True)
class FrequencyAxis(AsDataArray):
    """TODO: documentation"""

    data: Data[Frequency, float]
    """ Time, expressed in SI seconds since the epoch. """
    spectral_window_name: Attr[str]
    """ Name associated with spectral window. """
    frequency_group_name: Attr[str]
    """ Name associated with frequency group - needed for multi-band VLBI fringe-fitting."""
    reference_frequency: Attr[QuantityDict]
    """ A frequency representative of the spectral window, usually the sky
    frequency corresponding to the DC edge of the baseband. Used by the calibration
    system if a ﬁxed scaling frequency is required or in algorithms to identify the
    observing band. """
    channel_width: Attr[QuantityDict]
    """ The nominal channel bandwidth. Same units as data array (see units key). """
    doppler: Optional[Attr[DopplerXds]]
    """ Doppler tracking information """

    type: Attr[str] = "spectral_coord"
    """ Coordinate type. Should be ``"spectral_coord"``. """
    long_name: Attr[str] = "Frequency"
    """ Long-form name to use for axis"""
    units: Attr[tuple[str]] = ("Hz",)
    """ Units to associate with axis"""
    frame: Attr[str] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """


@dataclass(frozen=True)
class PolarizationAxis(AsDataArray):
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
    type: Attr[str] = "polarization"
    """ Coordinate type. Should be ``"polarization"``. """
    long_name: Attr[str] = "Polarization"
    """ Long-form name to use for axis. Should be ``"Polarization"``"""


@dataclass(frozen=True)
class UvwLabelAxis(AsDataArray):
    """
    Coordinate axis to make up ``("u", "v", "w")`` tuple, see :py:class:`UvwArray`.
    """

    data: Data[UvwLabel, str] = ("u", "v", "w")
    """Should be ``('u','v','w')``, used by :py:class:`UvwArray`"""
    long_name: Attr[str] = "U/V/W label"
    """ Long-form name to use for axis. Should be ``"U/V/W label"``"""


# Data variables
@dataclass(frozen=True)
class VisibilityArray(AsDataArray):
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization],
        numpy.complex64 | numpy.complex128,
    ]
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Coordof[FrequencyAxis]
    polarization: Coordof[PolarizationAxis]
    delay_direction: Attr[SkyCoordDict]
    """ 
    Direction of delay center, i.e. what the coorelator originally phased the
    visibilities to.

    For conversion from MSv2, frame refers column
    keywords by default. If frame varies with field, it refers PhaseDir_Ref
    column instead.
    """
    phase_direction: Attr[SkyCoordDict]
    """
    Phase direction of visibilities, i.e. the sky direction from which flux
    would result in real-valued visibilities independent of baseline UVW.
    """
    long_name: Attr[str] = "Visibility values"
    """ Long-form name to use for axis. Should be ``"Visibility values"``"""
    units: Attr[tuple[str]] = ("Jy",)


@dataclass(frozen=True)
class FlagArray(AsDataArray):
    """
    An array of Boolean values with the same shape as `VISIBILITY`,
    representing the cumulative flags applying to this data matrix. Data are
    flagged bad if the ``FLAG`` array element is ``True``.
    """

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization]
        | tuple[Time, BaselineId, Frequency]
        | tuple[Time, BaselineId],
        bool,
    ]
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Coordof[FrequencyAxis]
    polarization: Optional[Coordof[PolarizationAxis]] = None
    long_name: Attr[str] = "Visibility flags"


@dataclass(frozen=True)
class WeightArray(AsDataArray):
    """
    The weight for each channel, with the same shape as the associated
    :py:class:`VisibilityArray`, as assigned by the correlator or processor.

    Weight spectrum in ms v2 is renamed weight. Should be calculated as
    1/sigma^2 (sigma rms noise).
    """

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization]
        | tuple[Time, BaselineId, Frequency]
        | tuple[Time, BaselineId],
        numpy.float16 | numpy.float32 | numpy.float64,
    ]
    """Visibility weights"""
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Optional[Coordof[FrequencyAxis]] = None
    polarization: Optional[Coordof[PolarizationAxis]] = None
    long_name: Attr[str] = "Visibility weights"


@dataclass(frozen=True)
class UvwArray(AsDataArray):
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
        tuple[Time, BaselineId, Frequency, Polarization, UvwLabel]
        | tuple[Time, BaselineId, Frequency, UvwLabel]
        | tuple[Time, BaselineId, UvwLabel],
        numpy.float16 | numpy.float32 | numpy.float64,
    ]
    """Baseline coordinates from ``baseline_antenna2_id`` to ``baseline_antenna1_id``"""
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Optional[Coordof[FrequencyAxis]] = None
    polarization: Optional[Coordof[PolarizationAxis]] = None
    uvw_label: Coordof[UvwLabelAxis] = ("u", "v", "w")
    long_name: Attr[str] = "Baseline coordinates"
    """ Long-form name to use for axis. Should be ``"Baseline coordinates``"""
    units: Attr[tuple[str]] = ("m",)


@dataclass(frozen=True)
class TimeSamplingArray(AsDataArray):
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization]
        | tuple[Time, BaselineId, Frequency]
        | tuple[Time, BaselineId],
        float,
    ]

    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Optional[Coordof[FrequencyAxis]] = None
    polarization: Optional[Coordof[PolarizationAxis]] = None

    scale: Attr[str] = "tai"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[str] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """

    long_name: Attr[str] = "Time sampling data"
    units: Attr[str] = "s"


@dataclass(frozen=True)
class FreqSamplingArray(AsDataArray):
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Frequency, Polarization]
        | tuple[Time, BaselineId, Frequency]
        | tuple[Time, Frequency]
        | tuple[Frequency],
        float,
    ]
    """
    Data about frequency sampling, such as centroid or integration
    time. Concrete function depends on concrete data array within
    :py:class:`VisibilityXds`.
    """
    frequency: Coordof[FrequencyAxis]
    time: Optional[Coordof[TimeAxis]] = None
    baseline_id: Optional[Coordof[BaselineAxis]] = None
    polarization: Optional[Coordof[PolarizationAxis]] = None
    long_name: Attr[str] = "Frequency sampling data"
    units: Attr[str] = "Hz"
    frame: Attr[str] = "icrs"
    """
    Astropy velocity reference frames (see :external:ref:`astropy-spectralcoord`).
    Note that Astropy does not use the name
    'topo' (telescope centric) velocity frame, rather it assumes if no velocity
    frame is given that this is the default.
    """


# Data Sets
@dataclass(frozen=True)
class VisibilityXds(AsDataset):
    """TODO: documentation"""

    # Required Coordinates
    time: Coordof[TimeAxis]
    """
    The time coordinate is the mid-point of the nominal sampling interval, as
    speciﬁed in the ``ms_v4.time.attrs['integration_time']`` (ms v2 interval).
    """
    baseline_id: Coordof[BaselineAxis]
    frequency: Coordof[FrequencyAxis]
    """Center frequencies for each channel."""
    polarization: Coordof[PolarizationAxis]
    """
    Labels for polarization types, e.g. ``['XX','XY','YX','YY']``, ``['RR','RL','LR','LL']``.
    """
    uvw_label: Optional[Coordof[UvwLabelAxis]]

    # Required data variables / arrays
    VISIBILITY: Dataof[VisibilityArray]

    # Required Attributes
    field_info: Attr[FieldInfoDict]
    """Values without any phase rotation. See data array (``VISIBILITY`` and ``UVW``) attribute phase center for phase rotated value. """
    antenna_xds: Attr[AntennaXds]

    # Optional Coordinates
    baseline_antenna1_id: Optional[Coordof[BaselineAntennaAxis]] = None
    """Antenna id for 1st antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""
    baseline_antenna2_id: Optional[Coordof[BaselineAntennaAxis]] = None
    """Antenna id for 2nd antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""
    scan_id: Optional[Coord[Time, int]] = None
    """Arbitary scan number to identify data taken in the same logical scan."""

    # Optional data variables / arrays
    """Complex visibilities, either simulated or measured by interferometer."""
    FLAG: Optional[Dataof[FlagArray]] = None
    WEIGHT: Optional[Dataof[WeightArray]] = None
    UVW: Optional[Dataof[UvwArray]] = None
    EFFECTIVE_INTEGRATION_TIME: Optional[Dataof[TimeSamplingArray]] = None
    """
    The integration time, including the effects of missing data, in contrast to
    ``integration_time`` attribute of the ``time`` coordinate,
    see :py:class:`TimeAxis`. (MS v2: ``exposure``).
    """
    TIME_CENTROID: Optional[Dataof[TimeSamplingArray]] = None
    """
    The time centroid of the visibility, includes the effects of missing data
    unlike the ``time`` coordinate, see :py:class:`TimeAxis`.
    """
    TIME_CENTROID_EXTRA_PRECISION: Optional[Dataof[TimeSamplingArray]] = None
    """Additional precision for ``TIME_CENTROID``"""
    EFFECTIVE_CHANNEL_WIDTH: Optional[Dataof[FreqSamplingArray]] = None
    """The channel bandwidth that includes the effects of missing data."""
    FREQUENCY_CENTROID: Optional[Dataof[FreqSamplingArray]] = None
    """Includes the effects of missing data unlike ``frequency``."""

    # Optional Attributes
    pointing_xds: Optional[Attr[PointingXds]] = None
    source_xds: Optional[Attr[SourceXds]] = None
    pased_array_xds: Optional[Attr[PhasedArrayXds]] = None
    observation_info: Optional[Attr[ObservationInfoDict]] = None


@dataclass(frozen=True)
class PointingXds(AsDataset):
    # TODO
    pass


@dataclass(frozen=True)
class SpectralCoordXds(AsDataset):
    # TODO
    pass


@dataclass(frozen=True)
class AntennaXds(AsDataset):
    # TODO
    pass


@dataclass(frozen=True)
class SourceXds(AsDataset):
    # TODO
    pass


@dataclass(frozen=True)
class PhasedArrayXds(AsDataset):
    # TODO
    pass


@dataclass(frozen=True)
class DopplerXds(AsDataset):
    # TODO
    pass
