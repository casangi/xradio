from __future__ import annotations

from typing import Literal, Optional, Union
from xradio.schema.bases import (
    xarray_dataset_schema,
    xarray_dataarray_schema,
    dict_schema,
)

from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof

from xradio.measurement_set.schema import (
    CreatorDict,
    AntennaNameArray,
    AntennaName,
    ScanArray,
    Time,
    TimeCoordArray,
    Frequency,
    FrequencyArray,
    ReceptorLabel,
)

import numpy

CalibrationParameterName = Literal["calibration_parameter_name"]
""" Dimension for names of calibration parameters"""


@xarray_dataarray_schema
class ReceptorLabelArray:
    """
    Model of the receptor_label coordinate upgraded to be plottable
    """

    data: Data[ReceptorLabel, str]


@xarray_dataarray_schema
class CalibrationParameterNameArray:
    """
    Model of the calibration_parameter_name coordinate used in the main dataset.
    """

    data: Data[CalibrationParameterName, str]
    """Name for each parameter."""


@xarray_dataarray_schema
class AntennaCalibrationParameterArray:
    """
    Calibration parameters for antennas; these can be real or complex
    """

    data: Data[
        Union[
            tuple[
                Time, AntennaName, Frequency, CalibrationParameterName, ReceptorLabel
            ],
            tuple[Time, BaselineId, Frequency, Polarization],
        ],
        Union[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    ]


@xarray_dataarray_schema
class BaselineCalibrationParameterArray:
    """
    Calibration parameters for antennas; these can be real or complex
    """

    data: Data[
        tuple[Time, BaselineId, Frequency, CalibrationParameterName, ReceptorLabel],
        Union[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    ]


@xarray_dataarray_schema
class ParameterErrorArray:
    """
    Calibration parameter errors; these must be real
    """

    data: Data[
        Union[
            tuple[
                Time, AntennaName, Frequency, CalibrationParameterName, ReceptorLabel
            ],
            tuple[Time, BaselineId, Frequency, Polarization],
        ],
        Union[numpy.float32, numpy.float64],
    ]


# Data variables
@xarray_dataarray_schema
class FlagArray:
    """
    An array of Boolean or integer values with the same shape as the
    calibration parameters (either baseline or antenna based),
    representing the cumulative flags applying to this data matrix.
    """

    data: Data[
        Union[
            tuple[
                Time, AntennaName, Frequency, CalibrationParameterName, ReceptorLabel
            ],
            tuple[Time, BaselineId, Frequency, CalibrationParameterName, Polarization],
        ],
        bool,
    ]
    """ Flag value.  Data is flagged as bad if the array element is
    ``True`` or nonzero."""
    time: Coordof[TimeCoordArray]
    baseline_id: Optional[Coordof[BaselineArray]]  # Only IF
    antenna_name: Optional[Coordof[AntennaNameArray]]  # Only SD
    frequency: Coordof[FrequencyArray]
    receptor_label: Optional[Coordof[ReceptorLabelArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    long_name: Optional[Attr[str]] = "Calibration flags"


@xarray_dataset_schema
class AntennaCalibrationXds:
    """Main dataset for antenna-based calibration data"""

    # --- Required Coordinates ---
    time: Coordof[TimeCoordArray]
    """
    The time coordinate is the mid-point of the solution interval used to solve for
    the calibration parameters.
    """
    frequency: Coordof[FrequencyArray]
    """Center frequencies for each frequency interval used in calibration. """

    calibration_parameter_name: Coordof[CalibrationParameterNameArray]
    """Calibration parameter name. """

    antenna_name: Coordof[AntennaNameArray]
    """Antenna name. Maps to ``attrs['antenna_xds'].antenna_name``. """

    receptor_label: Coordof[ReceptorLabelArray]
    """
    Labels for polarization receptor types, e.g. ``['X','Y']``, ``['R','L']``, ``['P','Q']``.
    """

    # --- Required data variables ---

    ANTENNA_CALIBRATION_PARAMETER: Dataof[AntennaCalibrationParameterArray]
    """Calibration parameters for single antennas"""

    PARAMETER_ERROR: Dataof[ParameterErrorArray]
    """Error estimates for calibration paramters."""

    FLAG: Dataof[FlagArray]

    # --- Required Attributes ---

    schema_version: Attr[str]
    """Semantic version of calibration xds data format."""
    creator: Attr[CreatorDict]
    """Creator information (software, version)."""
    creation_date: Attr[str]
    """Date calibration dataset was created. Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""

    """
    Dataset type
    """

    type: Attr[str]
    """The type of calibration data stored in this xds."""
    type_version: Attr[str]
    """A calibration-specific version number."""

    # --- Optional Coordinates ---

    # These are compulsory in the Measurement Set xds, so maybe they should be compulsory here too?
    field_name: Coordof[Coord[Time, str]]
    """Field name."""

    scan_name: CoordOf[ScanArray]
    """Scan name to identify data taken in the same logical scan."""

    # --- Optional data variables / arrays ---

    # FIXME: Add reference antenna and spectral_window_name.

    # --- Optional Attributes ---


@xarray_dataset_schema
class BaselineCorrectionXds:
    """Calibration dataset for baseline effects"""

    # --- Required Coordinates ---
    time: Coordof[TimeCoordArray]
    """
    The time coordinate is the reference time for the calibration parameters
    """
    baseline_id: Coordof[BaselineArray]
    """ Baseline ID """
    frequency: Coordof[FrequencyArray]
    """Center frequencies for each channel."""
    polarization: Coordof[PolarizationArray]
    """
    Labels for polarization types, e.g. ``['XX','XY','YX','YY']``, ``['RR','RL','LR','LL']``.
    """
    field_name: Coordof[Coord[Time, str]]
    """Field name."""
    scan_name: Coordof[ScanArray]
    """Scan name to identify data taken in the same logical scan"""

    # --- Required data variables ---

    BASELINE_CALIBRATION_PARAMETER: Dataof[BaselineCalibrationParameterArray]
    """Calibration parameters for baselines"""

    PARAMETER_ERROR: Dataof[ParameterErrorArray]
    """Error estimates for calibration paramters"""

    FLAGS: Dataof[FlagArray]

    baseline_antenna1_name: Coordof[BaselineAntennaNameArray]
    """Antenna name for 1st antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_name``"""
    baseline_antenna2_name: Coordof[BaselineAntennaNameArray]
    """Antenna name for 2nd antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_name``"""

    # --- Required Attributes ---

    schema_version: Attr[str]
    """Semantic version of calibration xds data format."""
    creator: Attr[CreatorDict]
    """Creator information (software, version)."""
    creation_date: Attr[str]
    """Date calibration dataset was created. Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""

    type: Attr[Literal["antenna", "baseline"]] = "baseline"
    """
    Dataset type
    """

    # --- Optional Coordinates ---
    polarization_mixed: Optional[Coord[tuple[BaselineId, Polarization], str]] = None
    """
    If the polarizations are not constant over baseline. For mixed polarizations one would
    use ['PP', 'PQ', 'QP', 'QQ'] as the polarization labels and then specify here the
    actual polarization basis for each baseline using labels from the set of all
    combinations of 'X', 'Y', 'R' and 'L'.
    """
    # --- Optional Attributes ---
