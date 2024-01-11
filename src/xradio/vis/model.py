from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union
from xradio.schema.bases import AsDataArray, AsDataset
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
import numpy

# Dimensions
Time = Literal["Time"]
""" Observation time dimension """
BaselineId = Literal["BL"]
""" Baseline dimension """
Channel = Literal["Channel"]
""" Channel dimension """
Polarization = Literal["Polar"]
""" Polarization dimension """
UvwLabel = Literal["Uvw"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """

# Plain data class models
@dataclass
class SourceInfo:
    # TODO
    pass


@dataclass
class FieldInfo:
    # TODO
    pass


@dataclass
class Quantity:
    # TODO
    pass


# Coordinates / Axes
@dataclass(frozen=True)
class TimeAxis(AsDataArray):
    """Data model of time axis"""

    data: Data[Time, float]
    """ Time, expressed in SI seconds since the epoch (see ``scale`` & ``format``). """

    integration_time: Attr[Quantity]
    """ The nominal sampling interval (ms v2). Units of seconds. """
    effective_integration_time: Attr[Quantity]
    """ Name of data array that contains the integration time that includes the effects of missing data. """

    scale: Attr[str] = "tai"
    """ Astropy time scales, see :py:class:`astropy.time.Time` """
    format: Attr[str] = "unix"
    """ Astropy format, see :py:class:`astropy.time.Time`. Default seconds from 1970-01-01 00:00:00 UTC """

    type: Attr[str] = "time"
    """ Coordinate type. Should be ``time``. """
    long_name: Attr[str] = "Observation Time"
    """ Long-form name to use for axis"""
    units: Attr[tuple[str]] = ("s",)
    """ Units to associate with axis"""


@dataclass(frozen=True)
class BaselineAxis(AsDataArray):
    """TODO: documentation"""

    data: Data[BaselineId, int]
    long_name: Attr[str] = "Baseline ID"


@dataclass(frozen=True)
class BaselineAntennaAxis(AsDataArray):
    """TODO: documentation"""

    data: Data[BaselineId, int]
    long_name: Attr[str] = "Baseline Antenna ID"


@dataclass(frozen=True)
class FrequencyAxis(AsDataArray):
    """TODO: documentation"""

    data: Data[Channel, float]
    """ Time, expressed in SI seconds since the epoch. """
    spectral_window_name: Attr[str]
    """ Name associated with spectral window. """
    reference_frequency: Attr[Quantity]
    """ A frequency representative of the spectral window, usually the sky
    frequency corresponding to the DC edge of the baseband. Used by the calibration
    system if a ﬁxed scaling frequency is required or in algorithms to identify the
    observing band. """
    channel_width: Attr[Quantity]
    """ The nominal channel bandwidth. Same units as data array (see units key). """
    doppler: Optional[Attr[DopplerXds]]
    """ Doppler tracking information """

    type: Attr[str] = "spectral_coord"
    """ Coordinate type. Should be ``spectral_coord``. """
    long_name: Attr[str] = "Frequency"
    """ Long-form name to use for axis"""
    units: Attr[tuple[str]] = ("Hz",)
    """ Units to associate with axis"""


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
    """ Coordinate type. Should be ``polarization``. """
    long_name: Attr[str] = "Polarization"
    """ Long-form name to use for axis. Should be ``Polarization``"""


@dataclass(frozen=True)
class UvwLabelAxis(AsDataArray):
    """TODO: documentation"""

    data: Data[UvwLabel, str] = ("u", "v", "w")
    long_name: Attr[str] = "U/V/W label"


# Data variables
@dataclass(frozen=True)
class VisibilityArray(AsDataArray):
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Channel, Polarization],
        numpy.complex64 | numpy.complex128,
    ]
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Coordof[FrequencyAxis]
    polarization: Coordof[PolarizationAxis]
    long_name: Attr[str] = "Visibility"
    units: Attr[tuple[str]] = ("Jy",)


@dataclass(frozen=True)
class FlagArray(AsDataArray):
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Channel, Polarization]
        | tuple[Time, BaselineId, Channel]
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
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Channel, Polarization]
        | tuple[Time, BaselineId, Channel]
        | tuple[Time, BaselineId],
        numpy.float16 | numpy.float32 | numpy.float64,
    ]
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Optional[Coordof[FrequencyAxis]] = None
    polarization: Optional[Coordof[PolarizationAxis]] = None
    long_name: Attr[str] = "Visibility weights"


@dataclass(frozen=True)
class UvwArray(AsDataArray):
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Channel, Polarization, UvwLabel]
        | tuple[Time, BaselineId, Channel, UvwLabel]
        | tuple[Time, BaselineId, UvwLabel],
        numpy.float16 | numpy.float32 | numpy.float64,
    ]
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Optional[Coordof[FrequencyAxis]] = None
    polarization: Optional[Coordof[PolarizationAxis]] = None
    uvw_label: Coordof[UvwLabelAxis] = ("u", "v", "w")
    long_name: Attr[str] = "Visibility UVW coordinates"
    units: Attr[tuple[str]] = ("m",)


@dataclass(frozen=True)
class TimeSamplingArray(AsDataArray):
    """TODO: documentation"""

    data: Data[
        tuple[Time, BaselineId, Channel, Polarization]
        | tuple[Time, BaselineId, Channel]
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
        tuple[Time, BaselineId, Channel, Polarization]
        | tuple[Time, BaselineId, Channel]
        | tuple[Time, Channel]
        | tuple[Channel],
        float,
    ]
    time: Coordof[TimeAxis]
    baseline_id: Coordof[BaselineAxis]
    frequency: Optional[Coordof[FrequencyAxis]] = None
    polarization: Optional[Coordof[PolarizationAxis]] = None
    long_name: Attr[str] = "Frequency sampling data"
    units: Attr[str] = "Hz"


# Data Sets
@dataclass(frozen=True, kw_only=True)
class VisibilityXds(AsDataset):
    """TODO: documentation"""

    # Coordinates
    time: Coordof[TimeAxis]
    """The time coordinate is the mid-point of the nominal sampling interval, as
    speciﬁed in the ms_v4.time.attrs['integration_time'] (ms v2 interval). The
    EFFECTIVE_INTEGRATION_TIME data array (ms v2 exposure), in contrast to
    integration_time, deﬁnes the integration time which including the effects
    of missing data.  """
    baseline_id: Coordof[BaselineAxis]
    """Unique id for each baseline."""
    frequency: Coordof[FrequencyAxis]
    """Center frequencies for each channel."""
    polarization: Coordof[PolarizationAxis]
    """
    Labels for polarization types, e.g. ``['XX','XY','YX','YY']``, ``['RR','RL','LR','LL']``.
    """
    uvw_label: Optional[Coordof[UvwLabelAxis]] = None
    """Should be ``('u','v','w')``, used by ``UVW``"""
    baseline_antenna1_id: Optional[Coordof[BaselineAntennaAxis]] = None
    """Antenna id for 1st antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""
    baseline_antenna2_id: Optional[Coordof[BaselineAntennaAxis]] = None
    """Antenna id for 2nd antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_id``"""

    # Data variables / arrays
    VISIBILITY: Dataof[VisibilityArray]
    """Complex visibilities, either simulated or measured by interferometer."""
    FLAG: Optional[Dataof[FlagArray]] = None
    """VISIBILITY are ﬂagged bad if the FLAG array element is True."""
    WEIGHT: Optional[Dataof[WeightArray]] = None
    """weight spectrum in ms v2 is renamed weight. Should be calculated as
    1/sigma^2 (sigma rms noise)."""
    UVW: Optional[Dataof[UvwArray]] = None
    """Coordinates for the baseline from ``baseline_antenna2_id`` to
    ``baseline_antenna1_id``, i.e. the baseline is equal to the difference
    ```POSITION2 - POSITION1```. The UVW given are for the TIME_CENTROID, and
    correspond in general to the reference type for the
    field_info.phase_dir. The baseline direction should be : W towards source
    direction; V in plane through source and system's pole; U in direction of
    increasing longitude coordinate. This definition also determines the sign
    of the phase of VISIBILITY.
    """
    EFFECTIVE_INTEGRATION_TIME: Optional[Dataof[TimeSamplingArray]] = None
    """
    The integration time that includes the effects of missing data. (exposure
    in ms v2).
    """
    TIME_CENTROID: Optional[Dataof[TimeSamplingArray]] = None
    """Includes the effects of missing data unlike ms_v4.time."""
    TIME_CENTROID_EXTRA_PRECISION: Optional[Dataof[TimeSamplingArray]] = None
    """Additional precision for ``TIME_CENTROID``"""
    EFFECTIVE_CHANNEL_WIDTH: Optional[Dataof[FreqSamplingArray]] = None
    """The channel bandwidth that includes the effects of missing data."""
    FREQUENCY_CENTROID: Optional[Dataof[FreqSamplingArray]] = None
    """Includes the effects of missing data unlike ``frequency``."""

    # Attributes
    field_info: Attr[FieldInfo]
    """Values without any phase rotation. See data array (``VISIBILITY`` and ``UVW``) attribute phase center for phase rotated value. """
    source_info: Attr[SourceInfo]
    antenna_xds: Attr[AntennaXds]
    pointing_xds: Optional[Attr[PointingXds]] = None
    source_xds: Optional[Attr[SourceXds]] = None
    pased_array_xds: Optional[Attr[PhasedArrayXds]] = None


@dataclass(frozen=True)
class SpectralCoordXds(AsDataset):
    # TODO
    pass


@dataclass(frozen=True)
class AntennaXds(AsDataset):
    # TODO
    pass


@dataclass(frozen=True)
class PointingXds(AsDataset):
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
