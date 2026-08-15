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
    BaselineId,
    BaselineArray,
    BaselineAntennaNameArray,
    ScanArray,
    Time,
    TimeCoordArray,
    Frequency,
    FrequencyArray,
    ReceptorLabel,
    Polarization,
    PolarizationArray,
    TimeSamplingArray,
    EffectiveChannelWidthArray,
    FrequencyCentroidArray,
)


import numpy

Direction = Literal["direction"]


# Note: this isn't in measurement_set.schema but I'm pretty sure that's where it belongs
@xarray_dataarray_schema
class ReceptorLabelArray:
    """
    Model of the receptor_label coordinate upgraded to be plottable
    """

    data: Data[ReceptorLabel, str]


@xarray_dataarray_schema
class AntennaReceptorCalibrationParameterArray:
    """
    A baseclass for AntennaReceptorCalibrationParameterArrays; each
    subclass is allowed to be more specific about the Data type and
    required to specify its own units.
    """

    data: Data[
        tuple[Direction, Time, AntennaName, Frequency, ReceptorLabel],  # Direction,
        Union[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    ]

    direction: Coord[Direction, str]
    time: Coordof[TimeCoordArray]
    antenna_name: Coordof[AntennaNameArray]
    frequency: Coordof[FrequencyArray]
    receptor_label: Coordof[ReceptorLabelArray]


@xarray_dataarray_schema
class AntennaPolarizationCalibrationParameterArray:
    data: Data[
        tuple[Direction, Time, AntennaName, Frequency, Polarization],  # Direction,
        Union[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    ]

    direction: Coord[Direction, str]
    time: Coordof[TimeCoordArray]
    antenna_name: Coordof[AntennaNameArray]
    frequency: Coordof[FrequencyArray]
    receptor_label: Coordof[ReceptorLabelArray]


@xarray_dataarray_schema
class BaselineCalibrationParameterArray:
    """
    Calibration parameters for antennas; these can be real or complex
    """

    data: Data[
        tuple[Direction, Time, BaselineId, Frequency, Polarization],
        Union[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    ]
    direction: Coord[Direction, str]
    time: Coordof[TimeCoordArray]
    baseline_id: Coordof[BaselineArray]
    frequency: Coordof[FrequencyArray]
    polarization: Coordof[PolarizationArray]

    long_name: Optional[Attr[str]] = "Baseline Calibration Parameter"


############################################################
## We flesh out how this works for fringefit parameters.
##
## We state that all the parameter values are real, and they all derive
## their axes by inheritance from AntennaCalibrationParameterArray but
## each of them has a different unit.
##
## I am not thrilled that we override the Data type being careful to
## match the superclass dimensions and changing only the type, though.
##############################################################

##
## Note: We would much prefer to say:
##
## PhaseCal = AntennaCalibrationParameterArrayFactory(AntennaCalibrationParameterArray,
##                                                    long_name="Phase offset",
##                                                    unit="rad")
##
## And then actually inline that instead in the FringeJones class


@xarray_dataarray_schema
class PhaseCalibrationParameterArray(AntennaReceptorCalibrationParameterArray):
    """A phase-offset calibration parameter for use in (e.g.) fringe fit tables"""

    data: Data[
        tuple[Direction, Time, AntennaName, Frequency, ReceptorLabel], numpy.float64
    ]

    long_name: Optional[Attr[str]] = "Phase offset"
    units: Attr[UnitsRadians] = "rad"


@xarray_dataarray_schema
class DelayCalibrationParameterArray(AntennaReceptorCalibrationParameterArray):
    """A delay calibration parameter for use in (e.g.) fringe fit tables"""

    data: Data[
        tuple[Direction, Time, AntennaName, Frequency, ReceptorLabel], numpy.float64
    ]
    long_name: Optional[Attr[str]] = "Delay"
    units: Attr[UnitsSeconds] = "s"


@xarray_dataarray_schema
class RateCalibrationParameterArray(AntennaReceptorCalibrationParameterArray):
    """A dimensionless delay-rate calibration parameter for use in (e.g.) fringe fit tables"""

    data: Data[
        tuple[Direction, Time, AntennaName, Frequency, ReceptorLabel], numpy.float64
    ]
    long_name: Optional[Attr[str]] = "Rate"
    units: Attr[UnitsDimensionless] = "dimensionless"


## Every calibration parameter comes with an error. I would really
## prefer not to repeat the units; I hope we can mandate that they are
## implicitly those of the parameter itself without needed to enforce
## that at the schema level?
@xarray_dataarray_schema
class ParameterErrorArray:
    """
    Calibration parameter errors; these must be real
    """

    data: Data[
        Union[
            tuple[
                Direction,
                Time,
                AntennaName,
                Frequency,
                ReceptorLabel,
            ],
            tuple[
                Direction,
                Time,
                AntennaName,
                Frequency,
                Polarization,
            ],
            tuple[Direction, Time, BaselineId, Frequency, Polarization],
        ],
        Union[numpy.float32, numpy.float64],
    ]
    direction: Coord[Direction, str]
    time: Coordof[TimeCoordArray]
    antenna_name: Coordof[AntennaNameArray]
    baseline_id: Coordof[BaselineArray]
    frequency: Coordof[FrequencyArray]
    receptor_label: Coordof[ReceptorLabelArray]
    polarization: Coordof[PolarizationArray]


# We expect that a given CalibrationFlagArray will have either
# (a) both a baseline and a polarization dimension; OR
# (b) both an antenna_name and a receptor_label dimension


@xarray_dataarray_schema
class CalibrationFlagArray:
    """
    An array of Boolean or integer values with the same shape as the
    calibration parameters (either baseline or antenna based),
    representing the cumulative flags applying to this data matrix.
    """

    data: Data[
        Union[
            tuple[
                Direction,
                Time,
                AntennaName,
                Frequency,
                ReceptorLabel,
            ],
            tuple[
                Direction,
                Time,
                AntennaName,
                Frequency,
                Polarization,
            ],
            tuple[
                Direction,
                Time,
                BaselineId,
                Frequency,
                Polarization,
            ],
        ],
        bool,
    ]
    """ Flag value.  Data is flagged as bad if the array element is
    ``True`` or nonzero."""
    direction: Coord[Direction, str]
    time: Coordof[TimeCoordArray]
    antenna_name: Optional[Coordof[AntennaNameArray]]  # Only SD
    baseline_id: Optional[Coordof[BaselineArray]]  # Only IF
    frequency: Coordof[FrequencyArray]
    receptor_label: Optional[Coordof[ReceptorLabelArray]] = None
    polarization: Optional[Coordof[PolarizationArray]] = None
    #
    long_name: Optional[Attr[str]] = "Calibration flags"


@xarray_dataset_schema
class FringeJonesXds:
    """A class for the FringeJonesXds, as used in the fringefit task.

    Each calibration type will need to specify its own Xds in this way.
    """

    # --- Required data variables ---
    CALPARAM_PHASE: Dataof[PhaseCalibrationParameterArray]
    CALPARAM_DELAY: Dataof[DelayCalibrationParameterArray]
    CALPARAM_RATE: Dataof[RateCalibrationParameterArray]

    CALERROR_PHASE: Dataof[ParameterErrorArray]
    CALERROR_DELAY: Dataof[ParameterErrorArray]
    CALERROR_RATE: Dataof[ParameterErrorArray]

    SNR: Dataof[ParameterErrorArray]
    # --- Required Coordinates ---
    direction: Coord[Direction, str]
    time: Coordof[TimeCoordArray]
    """
    The time coordinate is the mid-point of the solution interval used to solve for
    the calibration parameters.
    """
    antenna_name: Coordof[AntennaNameArray]
    """Antenna name. Maps to ``attrs['antenna_xds'].antenna_name``. """

    frequency: Coordof[FrequencyArray]
    """Center frequencies for each frequency interval used in calibration. """

    receptor_label: Coordof[ReceptorLabelArray]
    """
    Labels for polarization receptor types, e.g. ``['X','Y']``, ``['R','L']``, ``['P','Q']``.
    """
    # --- Required Attributes ---

    schema_version: Attr[str]
    """Semantic version of calibration xds data format."""
    creator: Attr[CreatorDict]
    """Creator information (software, version)."""
    creation_date: Attr[str]
    """Date calibration dataset was created. Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""

    type_version: Attr[str]
    """A calibration-specific version number."""

    type: Attr[Literal["antenna_calibration", "baseline_calibration"]] = (
        "antenna_calibration"
    )
    """The type of calibration data stored in this xds."""

    # --- Optional Coordinates ---

    field_name: Optional[Coordof[Coord[Time, str]]] = None
    """Field name."""

    scan_name: Optional[Coordof[ScanArray]] = None
    """Scan name to identify data taken in the same logical scan."""

    # --- Optional data variables / arrays ---

    # Note to self: we need this stuff too, we were asked for it!
    TIME_CENTROID: Optional[Dataof[TimeSamplingArray]] = None
    """
    The time centroid of the visibility, includes the effects of missing data
    unlike the ``time`` coordinate, see :py:class:`~xradio.measurement_set.schema.TimeArray`.
    """
    TIME_CENTROID_EXTRA_PRECISION: Optional[Dataof[TimeSamplingArray]] = None
    """Additional precision for ``TIME_CENTROID``"""
    EFFECTIVE_CHANNEL_WIDTH: Optional[Dataof[EffectiveChannelWidthArray]] = None
    """The channel bandwidth that includes the effects of missing data."""
    FREQUENCY_CENTROID: Optional[Dataof[FrequencyCentroidArray]] = None
    """Includes the effects of missing data unlike ``frequency``."""

    # --- Optional Attributes ---

    # Note that we do not need a spectral_window attribute because the
    # FrequencyArray coordinate is already required to have one

    reference_antenna: Optional[Attr[str]] = None
    """The reference antenna (if any) used for this calibration."""


## The BaselineCalibrationXds is now obsolete, but I am not ready to
## delete it yet; I need a concrete BaslineCalibration example fleshed
## out before I let it go.

# @xarray_dataset_schema
# class BaselineCalibrationXds:
#     """Calibration dataset for baseline effects"""

#     # --- Required Coordinates ---
#     time: Coordof[TimeCoordArray]
#     """
#     The time coordinate is the reference time for the calibration parameters
#     """
#     baseline_id: Coordof[BaselineArray]
#     """ Baseline ID """
#     frequency: Coordof[FrequencyArray]
#     """Center frequencies for each channel."""
#     polarization: Coordof[PolarizationArray]
#     """
#     Labels for polarization types, e.g. ``['XX','XY','YX','YY']``, ``['RR','RL','LR','LL']``.
#     """
#     calibration_parameter_name: Coordof[CalibrationParameterNameArray]
#     """Calibration parameter name. """

#     # --- Required data variables ---

#     BASELINE_CALIBRATION_PARAMETER: Dataof[BaselineCalibrationParameterArray]
#     """Calibration parameters for baselines"""

#     PARAMETER_ERROR: Dataof[ParameterErrorArray]
#     """Error estimates for calibration paramters"""

#     FLAGS: Dataof[CalibrationFlagArray]

#     baseline_antenna1_name: Coordof[BaselineAntennaNameArray]
#     """Antenna name for 1st antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_name``"""

#     baseline_antenna2_name: Coordof[BaselineAntennaNameArray]
#     """Antenna name for 2nd antenna in baseline. Maps to ``attrs['antenna_xds'].antenna_name``"""

#     units: Dataof[CalibrationParameterUnitArray]

#     # --- Required Attributes ---

#     schema_version: Attr[str]
#     """Semantic version of calibration xds data format."""
#     creator: Attr[CreatorDict]
#     """Creator information (software, version)."""
#     creation_date: Attr[str]
#     """Date calibration dataset was created. Format: YYYY-MM-DDTHH:mm:ss.SSS (ISO 8601)"""

# type: Attr[Literal["antenna_calibration", "baseline_calibration"]] = (
#         "baseline_calibration"
#     )
#     """
#     Dataset type
#     """

#     # --- Optional Coordinates ---
#     polarization_mixed: Optional[Coord[tuple[BaselineId, Polarization], str]] = None
#     """
#     If the polarizations are not constant over baseline. For mixed polarizations one would
#     use ['PP', 'PQ', 'QP', 'QQ'] as the polarization labels and then specify here the
#     actual polarization basis for each baseline using labels from the set of all
#     combinations of 'X', 'Y', 'R' and 'L'.
#     """
#     field_name: Optional[Coordof[Coord[Time, str]]] = None
#     """Field name."""
#     scan_name: Optional[Coordof[ScanArray]] = None
#     """Scan name to identify data taken in the same logical scan"""

#     # --- Optional Attributes ---
