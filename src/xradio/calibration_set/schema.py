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
        tuple[Time, AntennaName, Frequency, ReceptorLabel],
        Union[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    ]


@xarray_dataarray_schema
class ParameterErrorArray:
    """
    Calibration parameter errors; these must be real
    """

    data: Data[
        tuple[Time, AntennaName, Frequency, ReceptorLabel],
        Union[numpy.float32, numpy.float64],
    ]


@xarray_dataset_schema
class CalibrationXds:
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

    scan_name: Coordof[ScanArray]
    """Scan name to identify data taken in the same logical scan."""

    # --- Required data variables ---

    CALIBRATION_PARAMETER: Dataof[CalibrationParameterArray]
    """Complex visibilities, either simulated or measured by interferometer."""

    PARAMETER_ERROR: Dataof[ParameterErrorArray]

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

    # --- Optional Coordinates ---

    receptor_label_mixed: Optional[Coord[tuple[AntennaName, ReceptorLabel], str]] = None
    """If the receptor_labels are not consistent across antennas, the
    receptor_labels ['P', 'Q'] should be used and then the actual
    receptors for each antenna should be specified here."""

    # --- Optional data variables / arrays ---

