from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union
from xarray_dataclasses import AsDataArray, AsDataset
from xarray_dataclasses import Attr, Coord, Coordof, Data, Dataof, Name
import numpy

# Dimensions
Time = Literal["time"]
""" Observation time dimension """
BaselineId = Literal["baseline_id"]
""" Baseline dimension """
Channel = Literal["channel"]
""" Channel dimension """
Polarization = Literal["polarization"]
""" Polarization dimension """
UvwLabel = Literal["uvw_label"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """


# Coordinates / Axes
@dataclass(frozen=True)
class TimeAxis(AsDataArray):
    """Data model of time axis"""

    data: Data[Time, float]
    """ Time, expressed in SI seconds since the epoch. """
    long_name: Attr[str] = "Observation Time"
    """ Long-form name to use for axis"""
    units: Attr[str] = "s"
    """ Unit to associate with axis"""
    epoch: Attr[str] = "TAI"
    """ TODO - this is likely not enough """


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
    spectral_coord: Attr[SpectralCoordXds]
    long_name: Attr[str] = "Frequency"
    units: Attr[str] = "Hz"


@dataclass(frozen=True)
class PolarizationAxis(AsDataArray):
    """TODO: documentation"""

    data: Data[Polarization, str]
    long_name: Attr[str] = "Polarization"


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
    units: Attr[str] = "Jy"


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
    units: Attr[str] = "m"


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
    baseline_id: Coordof[BaselineAxis]
    frequency: Coordof[FrequencyAxis]
    polarization: Coordof[PolarizationAxis]
    uvw_label: Optional[Coordof[UvwLabelAxis]] = None
    baseline_antenna1_id: Optional[Coordof[BaselineAntennaAxis]] = None
    baseline_antenna2_id: Optional[Coordof[BaselineAntennaAxis]] = None

    # Data variables / arrays
    VISIBILITY: Dataof[VisibilityArray]
    FLAG: Optional[Dataof[FlagArray]] = None
    WEIGHT: Optional[Dataof[WeightArray]] = None
    UVW: Optional[Dataof[UvwArray]] = None
    EFFECTIVE_INTEGRATION_TIME: Optional[Dataof[TimeSamplingArray]] = None
    TIME_CENTROID: Optional[Dataof[TimeSamplingArray]] = None
    TIME_CENTROID_EXTRA_PRECISION: Optional[Dataof[TimeSamplingArray]] = None
    EFFECTIVE_CHANNEL_WIDTH: Optional[Dataof[FreqSamplingArray]] = None
    FREQUENCY_CENTROID: Optional[Dataof[FreqSamplingArray]] = None

    # Attributes
    field_info: Attr[FieldInfo]
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


@dataclass
class SourceInfo:
    # TODO
    pass


@dataclass
class FieldInfo:
    # TODO
    pass
