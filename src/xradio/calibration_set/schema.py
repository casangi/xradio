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
class CalibrationParameterArray:
    """
    Calibration parameters; these can be real or complex
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

    CALIBRATION_PARAMETER: Dataof[CalibrationParameterArray]
    """Calibration parameters"""

    PARAMETER_ERROR: Dataof[ParameterErrorArray]
    """Error estimates for calibration paramters."""

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


# Note that the AntennaXDS has a map from (antenna_name, receptor_label) to polarization_type
# Also receptor_label, which is a full-on *dimension* has labels 'pol_0' and 'pol_1',

# xds.antenna_xds
#     Dimensions:                 (time: 120, baseline_id: 55, frequency: 32,
#                                  polarization: 4, uvw_label: 3, antenna_name: 10,
#                                  cartesian_pos_label: 3, receptor_label: 2)
#     Coordinates:
#     [...]
#         polarization_type       (antenna_name, receptor_label) <U1 80B dask.array<chunksize=(10, 2), meta=np.ndarray>
#       * receptor_label          (receptor_label) <U5 40B 'pol_0' 'pol_1'


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
    """Scan name to identify data taken in the same logical scan."""

    # --- Required data variables ---

    CALIBRATION_PARAMETER: Dataof[CalibrationParameterArray]
    """Calibration parameters"""

    PARAMETER_ERROR: Dataof[ParameterErrorArray]
    """Error estimates for calibration paramters."""

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
