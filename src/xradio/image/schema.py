"""
Schema of the xradio image dataset.

The image dataset is an ``xarray.Dataset`` that holds one or more images
(sky images, point spread functions, primary beams, masks, gridded
visibilities, etc.) that share coordinates. The schema follows the same
conventions as the measurement set v4 schema
(``xradio.measurement_set.schema``):

* Data variables are UPPER_SNAKE case, coordinates and attributes are
  lower_snake case.
* All measures are stored as xarray compatible dictionaries (i.e. they can be
  converted using :py:meth:`xarray.DataArray.from_dict`).
* Data variables can appear in multiple versions distinguished by an
  underscore separated suffix, for example ``SKY``, ``SKY_DECONVOLVED``
  or ``SKY_MODEL`` (see ``allow_multiple_versions``).
* The ``data_groups`` dataset attribute maps logical roles (``sky``,
  ``flag``, ``point_spread_function``, ...) to concrete data variable names,
  grouping the variables that belong together.

Schema building blocks shared with the measurement set schema live in
:py:mod:`xradio.schema.measures`.

The image tutorial (``docs/source/image_data/tutorials/image.ipynb``)
demonstrates the image dataset and its schema checking.
"""

from __future__ import annotations

from typing import Literal

import numpy
import xarray

from xradio.image._util.conventions import canonical_polarization_order
from xradio.schema.bases import (
    dict_schema,
    xarray_dataarray_schema,
    xarray_dataset_schema,
)
from xradio.schema.check import (
    SchemaIssue,
    SchemaIssues,
    _matching_data_var_names,
    check_array,
    check_dataset,
    check_dict,
)
from xradio.schema.dataclass import (
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
    xarray_dataclass_to_dict_schema,
)
from xradio.schema.measures import (
    ZD,
    AllowedDopplerTypes,
    AllowedSkyCoordFrames,
    AllowedSpectralCoordFrames,
    AllowedTimeFormats,
    AllowedTimeScales,
    CartesianPosLabel,
    Doppler,
    EllipsoidDirLabel,
    EllipsoidDisLabel,
    Frequency,
    Location,
    Polarization,
    PolarizationArray,
    QuantityInHertzArray,
    SkyCoord,
    SkyDirLabel,
    SpectralCoord,
    SpectralCoordArray,
    Time,
    UnitsHertz,
    UnitsOfDopplerShift,
    UnitsOfLocationInMetersOrRadians,
    UnitsOfSkyCoordInMetersOrRadians,
    UnitsRadians,
)
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof

# Dimensions
L = Literal["l"]
""" Projection plane coordinate towards the east, measured from the reference
direction (the direction cosine l of AIPS Memo #27 for the SIN projection,
see :py:class:`LCoordArray`). """
M = Literal["m"]
""" Projection plane coordinate towards the north, measured from the reference
direction (the direction cosine m of AIPS Memo #27 for the SIN projection,
see :py:class:`MCoordArray`). """
U = Literal["u"]
""" Aperture plane dimension conjugate to l. """
V = Literal["v"]
""" Aperture plane dimension conjugate to m. """
BeamParamsLabel = Literal["beam_params_label"]
""" Coordinate labels of Gaussian beam fit parameters (shape 3 and 'major', 'minor', 'pa'). """

BEAM_PARAMS_LABELS = ("major", "minor", "pa")
""" Values of the ``beam_params_label`` coordinate, in this order. """

UnitsOfImageTime = Literal["d", "s"]
""" Units of time values in images. Typically days ('d') for MJD formatted times. """

# Sub image types of a sky image, derived from the casacore image type, see
# https://github.com/casacore/casacore/blob/dede86795b94ea5651d26a889fea8ced455bfd14/images/Images/ImageInfo.h#L93-L110
# The values are casacore's names without spaces (casacore's "Column Density"
# is "ColumnDensity"), in casacore's order. Readers translate casacore image
# types and FITS BTYPE values case insensitively, and writers translate back
# to casacore's spelling (see normalize_sub_type and sub_type_to_casacore in
# xradio.image._util.conventions); casacore's "Undefined" type gives no
# sub_type (and an image without a sub_type is written to CASA as Intensity).
AllowedSkyImageSubTypes = Literal[
    "Intensity",
    "Beam",
    "ColumnDensity",
    "DepolarizationRatio",
    "KineticTemperature",
    "MagneticField",
    "OpticalDepth",
    "RotationMeasure",
    "RotationalTemperature",
    "SpectralIndex",
    "Velocity",
    "VelocityDispersion",
]


@xarray_dataarray_schema
class TimeMeasureArray:
    """
    Time measure used in image attributes, for example the observation date
    (``obsdate``). See :py:class:`~xradio.schema.measures.TimeArray` for the
    astropy based conventions; images typically use days since the MJD epoch.
    """

    data: Data[ZD, float]
    """Time since epoch, typically in days (see ``units`` and ``format``)."""

    type: Attr[Time] = "time"
    """ Measure type. Should be ``"time"``. """
    units: Attr[UnitsOfImageTime] = "d"
    """ Units to associate with the time value. """
    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scale, see :py:class:`astropy.time.Time`. """
    format: Attr[AllowedTimeFormats] = "mjd"
    """ Astropy time format, typically ``"mjd"``. """


# Coordinates / Axes
@xarray_dataarray_schema
class TimeCoordArray:
    """Time coordinate of the image dataset (typically a single value, the observation date)."""

    data: Data[Time, float]
    """ Time since epoch, typically in days (see ``units`` and ``format``). """

    type: Attr[Time] = "time"
    """ Coordinate type. Should be ``"time"``. """
    units: Attr[UnitsOfImageTime] = "d"
    """ Units to associate with axis. """
    scale: Attr[AllowedTimeScales] = "utc"
    """ Astropy time scale, see :py:class:`astropy.time.Time`. """
    format: Attr[AllowedTimeFormats] = "mjd"
    """ Astropy time format, typically ``"mjd"``. """


@xarray_dataarray_schema
class FrequencyCoordArray:
    """Frequency coordinate of the image dataset. Uses the same spectral
    coordinate measures as the measurement set frequency axis
    (:py:class:`~xradio.measurement_set.schema.FrequencyArray`), without the
    spectral window attributes that only apply to visibility data."""

    data: Data[Frequency, float]
    """ Center frequencies for each channel. """

    rest_frequency: Attr[QuantityInHertzArray]
    """ Rest frequency of the spectral line stored with the image. """
    reference_frequency: Attr[SpectralCoordArray]
    """ A frequency representative of the image spectral axis. Its
    ``observer`` is the spectral reference frame of the image (``frame``) in
    the vocabulary of
    :py:data:`~xradio.schema.measures.AllowedSpectralCoordFrames`, for example
    ``"lsrk"`` for LSRK, ``"gcrs"`` for GEO and ``"BARY"`` for BARY. """
    frame: Attr[str]
    """ Spectral reference frame of the image, by its casacore name: one of
    ``"REST"``, ``"LSRK"``, ``"LSRD"``, ``"BARY"``, ``"GEO"``, ``"TOPO"``,
    ``"GALACTO"``, ``"LGROUP"`` or ``"CMB"``. Readers translate other names
    for the same frames, such as FITS ``SPECSYS`` values (``"BARYCENT"``,
    ``"SOURCE"``), to these; the CASA reader keeps casacore's
    ``"Undefined"``. Images that the ``make_empty_*`` factories make in an
    astropy frame without a casacore equivalent hold its name (``"icrs"``,
    ``"hcrs"`` or ``"lsr"``); the CASA and FITS writers cannot write those. """
    channel_width: Attr[QuantityInHertzArray] | None = None
    """ The nominal channel width, as in the measurement set frequency
    coordinate (:py:class:`~xradio.measurement_set.schema.FrequencyArray`).
    Readers fill it from the spectral increment of the image; writers use it
    as the increment of a single channel spectral axis, which the frequency
    values alone do not determine. """
    wave_units: Attr[str] | None = None
    """ Units to associate with the wavelength representation of the axis, for example ``"mm"``. """
    observer: Attr[AllowedSpectralCoordFrames] | None = None
    """ Spectral reference frame of the axis in the vocabulary of
    :py:data:`~xradio.schema.measures.AllowedSpectralCoordFrames`, the same
    value as the ``observer`` of ``reference_frequency`` (see
    :py:class:`~xradio.schema.measures.SpectralCoordArray`). """

    type: Attr[SpectralCoord] = "spectral_coord"
    """ Coordinate type. Should be ``"spectral_coord"``. """
    units: Attr[UnitsHertz] = "Hz"
    """ Units to associate with axis. """


@xarray_dataarray_schema
class VelocityCoordArray:
    """Velocity coordinate of the image dataset. A non-dimensional coordinate
    parallel to ``frequency``, derived from the frequency values and the rest
    frequency using the Doppler convention given by ``doppler_type``."""

    data: Data[Frequency, float]
    """ Velocity value for each frequency channel. """

    type: Attr[Doppler] = "doppler"
    """ Coordinate type. Should be ``"doppler"``. """
    units: Attr[UnitsOfDopplerShift] = "m/s"
    """ Units to associate with axis, [ratio]/[m/s]. """
    doppler_type: Attr[AllowedDopplerTypes] = "radio"
    """ Doppler convention used to compute the velocities, typically ``"radio"``. """


@xarray_dataarray_schema
class LCoordArray:
    """The l coordinate of the image dataset.

    l is the projection plane coordinate (the FITS WCS intermediate world
    coordinate) towards the east, measured from the reference direction:
    l = x*cdelt, where x is the pixel offset from the reference pixel and
    cdelt the pixel increment in radians. For the SIN projection without
    projection parameters, the usual projection of interferometric images, l
    is the direction cosine l of AIPS Memo #27 (Section III). For other
    projections l is neither the direction cosine nor the angular offset (for
    TAN they differ from l by 1% to 1.5% at 10 degrees from the reference
    direction); the ``projection`` of ``coordinate_system_info`` defines it.
    """

    data: Data[L, float]
    """ Projection plane coordinate towards the east, in radians (the
    dimensionless direction cosine for the SIN projection). """

    note: Attr[str] | None = None
    """ Explanatory note on the definition of l. """


@xarray_dataarray_schema
class MCoordArray:
    """The m coordinate of the image dataset.

    m is the projection plane coordinate (the FITS WCS intermediate world
    coordinate) towards the north, measured from the reference direction:
    m = y*cdelt, where y is the pixel offset from the reference pixel and
    cdelt the pixel increment in radians. For the SIN projection without
    projection parameters, the usual projection of interferometric images, m
    is the direction cosine m of AIPS Memo #27 (Section III). For other
    projections m is neither the direction cosine nor the angular offset; the
    ``projection`` of ``coordinate_system_info`` defines it.
    """

    data: Data[M, float]
    """ Projection plane coordinate towards the north, in radians (the
    dimensionless direction cosine for the SIN projection). """

    note: Attr[str] | None = None
    """ Explanatory note on the definition of m. """


@xarray_dataarray_schema
class UCoordArray:
    """The u coordinate of aperture plane data (conjugate to l).

    By convention u is in wavelengths (``units`` ``"lambda"``), the Fourier
    conjugate of the dimensionless l."""

    data: Data[U, float]
    """ u values, in wavelengths. """

    units: Attr[str] | None = None
    """ Units to associate with axis. The convention is ``"lambda"``
    (wavelengths). """
    crval: Attr[float] | None = None
    """ Reference value at the reference pixel. """
    cdelt: Attr[float] | None = None
    """ Increment per pixel. """
    type: Attr[str] | None = None
    """ Coordinate type. """


@xarray_dataarray_schema
class VCoordArray:
    """The v coordinate of aperture plane data (conjugate to m).

    By convention v is in wavelengths (``units`` ``"lambda"``), the Fourier
    conjugate of the dimensionless m."""

    data: Data[V, float]
    """ v values, in wavelengths. """

    units: Attr[str] | None = None
    """ Units to associate with axis. The convention is ``"lambda"``
    (wavelengths). """
    crval: Attr[float] | None = None
    """ Reference value at the reference pixel. """
    cdelt: Attr[float] | None = None
    """ Increment per pixel. """
    type: Attr[str] | None = None
    """ Coordinate type. """


@xarray_dataarray_schema
class BeamParamsLabelCoordArray:
    """Coordinate axis to make up the ``("major", "minor", "pa")`` tuple of
    Gaussian beam fit parameters, see :py:class:`BeamFitParamsArray`."""

    data: Data[BeamParamsLabel, str] = BEAM_PARAMS_LABELS
    """Should be ``('major', 'minor', 'pa')``, in this order."""


# Measures used in image attributes
@xarray_dataarray_schema
class SkyDirectionArray:
    """
    Sky coordinate measure used in image attributes (the reference direction
    of the coordinate system and the pointing center). Like
    :py:class:`~xradio.schema.measures.SkyCoordArray`, with an optional
    equinox for equatorial frames such as fk5.
    """

    data: Data[SkyDirLabel, float]
    """ Sky direction values ('ra', 'dec' or 'lon', 'lat'), in radians. """

    units: Attr[UnitsOfSkyCoordInMetersOrRadians]
    """ Units of the direction values. """
    type: Attr[SkyCoord] = "sky_coord"
    """ Measure type. Should be ``"sky_coord"``. """
    frame: Attr[AllowedSkyCoordFrames] = "icrs"
    """ Astropy sky coordinate frame, for example ``"fk5"``. """
    equinox: Attr[str] | None = None
    """ Equinox of equatorial frames, for example ``"j2000.0"`` for fk5. """


@xarray_dataarray_schema
class TelescopeLocationArray:
    """
    Location measure of the telescope, stored in the ``telescope`` attribute
    of image data variables. Unlike
    :py:class:`~xradio.schema.measures.LocationArray` the frame is the native
    casacore telescope position frame (typically ``"ITRF"``). The readers
    store geocentric spherical coordinates (``coordinate_system``
    ``"geocentric"``), see :py:class:`TelescopeDict`.
    """

    data: Data[EllipsoidDirLabel | EllipsoidDisLabel | CartesianPosLabel, float]
    """ Location values ('lon', 'lat' in radians, or 'dist' in meters, or 'x', 'y', 'z' in meters). """

    units: Attr[UnitsOfLocationInMetersOrRadians]
    """ Units of the location coordinates (typically 'm' or 'rad'). """
    frame: Attr[str]
    """ Reference frame, for example ``"ITRF"``. """
    coordinate_system: Attr[str]
    """ Coordinate system, for example ``"geocentric"`` (spherical
    coordinates of the position vector: longitude, geocentric latitude and
    distance from the geocenter). """
    origin_object_name: Attr[str]
    """ earth/sun/moon/etc. """
    type: Attr[Location] = "location"
    """ Measure type. Should be ``"location"``. """


@xarray_dataarray_schema
class NativePoleDirectionArray:
    """
    Direction of the pole of the native projection coordinate system
    (combines the FITS keywords LONPOLE and LATPOLE in a single measure).
    Stored in the ``coordinate_system_info`` dataset attribute.
    """

    data: Data[EllipsoidDirLabel, float]
    """ Native pole direction ('lon', 'lat'), in radians. """

    frame: Attr[Literal["NATIVE_PROJECTION"]] = "NATIVE_PROJECTION"
    """ Frame label of the native projection coordinate system. """
    units: Attr[UnitsRadians] = "rad"
    """ Units to associate with the direction values. """
    type: Attr[Location] = "location"
    """ Measure type. Should be ``"location"``. """


# Info dicts
@dict_schema
class TelescopeDict:
    """Telescope information stored in the attributes of image data variables.

    The telescope position is stored in geocentric spherical ITRF
    coordinates (``coordinate_system`` ``"geocentric"``): ``direction`` holds
    the longitude and the geocentric latitude, ``distance`` the distance from
    the geocenter, so the Cartesian ITRF position is
    x = distance cos(lat) cos(lon), y = distance cos(lat) sin(lon) and
    z = distance sin(lat) (for example for astropy's
    ``EarthLocation.from_geocentric``). The geocentric latitude is not the
    geodetic latitude (for ALMA they differ by about 0.14 degrees)."""

    name: str
    """ Telescope name, for example 'ALMA'. """
    direction: TelescopeLocationArray | None
    """ Location measure holding the longitude and the geocentric latitude of
    the telescope ('lon', 'lat', in radians). """
    distance: TelescopeLocationArray | None
    """ Location measure holding the distance of the telescope from the
    geocenter ('dist', in meters). """


@dict_schema
class UserDict:
    """Free form dictionary of extra keywords carried along with an image, for
    example leftover FITS header cards or casacore miscinfo. No keys are
    required."""


@dict_schema
class CoordinateSystemInfoDict:
    """World coordinate system information of the image dataset. Present for
    images with a direction coordinate (sky images)."""

    reference_direction: SkyDirectionArray
    """ Sky coordinate measure of the reference direction (the direction at the reference pixel). """
    native_pole_direction: NativePoleDirectionArray
    """ Direction of the pole of the native projection coordinate system
    (latpole and lonpole in FITS combined in a single measure). """
    projection: str
    """ Sky projection, for example 'SIN' (see FITS-WCS paper II). """
    projection_parameters: list[float]
    """ Projection parameters (PVi_j in FITS-WCS). """
    pixel_coordinate_transformation_matrix: list[list[float]]
    """ Matrix relating pixel offsets to intermediate world coordinate offsets (PCi_j in FITS-WCS). """


@dict_schema
class DataGroupDict:
    """Defines a group of images. Keys are logical roles, values are the names
    of the data variables that fill the role for this group."""

    sky: str | None
    """ Image of the sky. Name of the sky variable, for example 'SKY'. Derived
    from the gridded visibilities. On plane tangential to celestial sphere.
    The variable's ``sub_type`` attribute records the physical quantity held
    by the image (Intensity, Beam, ColumnDensity, DepolarizationRatio,
    KineticTemperature, MagneticField, OpticalDepth, RotationMeasure,
    RotationalTemperature, SpectralIndex, Velocity, VelocityDispersion),
    derived from the casacore image type, see :py:class:`SkyArray`. """
    flag: str | None
    """ A boolean image defining any invalid pixels. Name of the sky pixels
    flags variable, for example 'FLAG_SKY'. For CASA images this is an
    internal mask. """
    point_spread_function: str | None
    """ The instrumental response or "dirty beam." Represents how a point
    source would appear given the uv-coverage and weighting scheme. Used by
    deconvolution algorithms to model and remove sidelobe artifacts.
    Determines the resolution of the image. Should be unity at the peak.
    Name of the point spread function variable of the group, for example
    'POINT_SPREAD_FUNCTION'. On plane tangential to celestial sphere. """
    primary_beam: str | None
    """ The effective antenna power pattern projected onto the sky, describing
    how sensitivity falls off from the pointing center. Values range from 1.0
    at center to ~0 at the edges. Name of the primary beam variable of the
    group, for example 'PRIMARY_BEAM'. On plane tangential to celestial
    sphere. """
    mask: str | None
    """ A boolean image defining the region(s) where the deconvolution
    algorithm is allowed to place clean components. Name of the deconvolution
    mask variable of the group, for example 'MASK'. On plane tangential to
    celestial sphere. """
    beam_fit_params_sky: str | None
    """ The Gaussian fit of the resolution element. Note that this would be
    the same as the beam_fit_params_point_spread_function except when the sky
    image is convolved to a common beam. Name of the beam fit parameters
    variable that applies to the sky variable of the group, for example
    'BEAM_FIT_PARAMS_SKY'. """
    beam_fit_params_point_spread_function: str | None
    """ The Gaussian fit to the peak of the point_spread_function. Name of the
    beam fit parameters variable that applies to the point spread function of
    the group, for example 'BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION'. """
    visibility: str | None
    """ Name of the visibility variable of the group, for example 'VISIBILITY'. The gridded visibilities used to create the images using a Fourier transform. On aperture plane."""
    visibility_normalization: str | None
    """ The aggregate of the image weights and used to normalize the gridded
    visibilities. Name of the variable, for example
    'VISIBILITY_NORMALIZATION'. """
    uv_sampling: str | None
    """ Name of the uv sampling variable of the group, for example 'UV_SAMPLING'. The gridded weights used to create the point spread function using a Fourier transform. On aperture plane."""
    uv_sampling_normalization: str | None
    """ Normalization factor for the gridded weights. This is the sum of weights and the sensitivity can be calculated using 1/sqrt(uv_sampling_normalization)."""
    aperture: str | None
    """ Name of the aperture variable of the group, for example 'APERTURE'. On aperture plane. The aperture is the Fourier transform of the primary beam."""
    aperture_normalization: str | None
    """ Normalization factor for the aperture data.  """
    description: str | None
    """ String description. More details about the data group. """
    date: str | None
    """ Date created. Creation date-time, in ISO 8601 format:
    'YYYY-MM-DDTHH:mm:ss.SSS'. """


@dict_schema
class DataGroupsDict:
    """Dictionary of image data group dictionaries. Groups can have arbitrary
    names, for example 'base', 'deconvolved', 'dirty', 'model' or 'residual'."""

    base: DataGroupDict | None
    """ The default data group, present when a single sky image is opened. """


# Data variables
@xarray_dataarray_schema
class SkyArray:
    """Image of the sky. Derived from the gridded visibilities, on a plane
    tangential to the celestial sphere. Examples of versions of this variable
    (each belonging to its own data group) are ``SKY``, ``SKY_DECONVOLVED``,
    ``SKY_MODEL`` and ``SKY_RESIDUAL``.

    The optional ``sub_type`` attribute records the physical quantity held by
    the image, derived from the `casacore image type
    <https://github.com/casacore/casacore/blob/dede86795b94ea5651d26a889fea8ced455bfd14/images/Images/ImageInfo.h#L93-L110>`_
    (without changing the image ``type``): Intensity, Beam, ColumnDensity,
    DepolarizationRatio, KineticTemperature, MagneticField, OpticalDepth,
    RotationMeasure, RotationalTemperature, SpectralIndex, Velocity or
    VelocityDispersion."""

    data: Data[
        tuple[Time, Frequency, Polarization, L, M],
        numpy.float32 | numpy.float64,
    ]
    """ Sky brightness (see ``units``, typically Jy/beam or Jy/pixel). """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    l: Coordof[LCoordArray]  # noqa: E741
    m: Coordof[MCoordArray]

    velocity: Coordof[VelocityCoordArray] | None = None
    right_ascension: Coord[tuple[L, M], float] | None = None
    """ Right ascension of each pixel, in radians. """
    declination: Coord[tuple[L, M], float] | None = None
    """ Declination of each pixel, in radians. """

    type: Attr[Literal["sky"]] = "sky"
    """ Image type. Should be ``"sky"``. """
    units: Attr[str] | None = None
    """ Brightness units, for example 'Jy/beam' or 'Jy/pixel'. """
    telescope: Attr[TelescopeDict] | None = None
    """ Telescope name and location. """
    observer: Attr[str] | None = None
    """ Name of the observer. """
    obsdate: Attr[TimeMeasureArray] | None = None
    """ Observation date, as a time measure. """
    pointing_center: Attr[SkyDirectionArray] | None = None
    """ Pointing center of the observation, as a sky coordinate measure. """
    object_name: Attr[str] | None = None
    """ Name of the observed object, for example '3c286'. """
    user: Attr[UserDict] | None = None
    """ Free form dictionary of extra keywords. """
    description: Attr[str] | None = None
    """ Description of the image. """
    beam_fit_params: Attr[str] | None = None
    """ Name of the beam fit parameters variable that applies to this image,
    for example 'BEAM_FIT_PARAMS_SKY'. """
    flag: Attr[str] | None = None
    """ Name of the flag variable that applies to this image, for example 'FLAG_SKY'. """
    sub_type: Attr[AllowedSkyImageSubTypes] | None = None
    """ Sub image type, derived from the casacore image type or FITS BTYPE
    (casacore's name with spaces removed, so 'Column Density' becomes
    'ColumnDensity'). casacore's 'Undefined' image type, like a missing or
    unknown BTYPE, gives no sub_type, and an image without a sub_type is
    written to CASA with casacore's default type, Intensity. """
    allow_multiple_versions: Attr[bool] | None = True


@xarray_dataarray_schema
class FlagArray:
    """A boolean image defining any invalid pixels (``True`` means invalid).
    For CASA images this is derived from the (inverted) internal mask.
    Versions of this variable are named after the image they apply to and
    have its dimensions: for example ``FLAG_SKY`` or ``FLAG_SKY_RESIDUAL`` on
    the sky plane ``(l, m)``, ``FLAG_APERTURE`` on the aperture plane
    ``(u, v)``, and ``FLAG_VISIBILITY_NORMALIZATION`` with one value per
    plane."""

    data: Data[
        tuple[Time, Frequency, Polarization, L, M]
        | tuple[Time, Frequency, Polarization, U, V]
        | tuple[Time, Frequency, Polarization],
        bool,
    ]
    """ Pixel flags, ``True`` means the pixel is invalid. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    l: Coordof[LCoordArray] | None = None  # noqa: E741
    m: Coordof[MCoordArray] | None = None
    u: Coordof[UCoordArray] | None = None
    v: Coordof[VCoordArray] | None = None

    type: Attr[Literal["flag"]] = "flag"
    """ Image type. Should be ``"flag"``. """
    allow_multiple_versions: Attr[bool] | None = True


@xarray_dataarray_schema
class BeamFitParamsArray:
    """Parameters of a Gaussian beam fit, per plane (time, frequency,
    polarization). The last dimension holds the major axis, minor axis and
    position angle, all in radians. Versions of this variable are named after
    the image they apply to:

    * ``BEAM_FIT_PARAMS_SKY``: The Gaussian fit of the resolution element.
      Note that this would be the same as the
      ``beam_fit_params_point_spread_function`` except when the sky image is
      convolved to a common beam.
    * ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION``: The Gaussian fit to the peak
      of the point_spread_function."""

    data: Data[tuple[Time, Frequency, Polarization, BeamParamsLabel], float]
    """ Gaussian fit parameters ('major', 'minor', 'pa'), in radians. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    beam_params_label: Coordof[BeamParamsLabelCoordArray]

    type: Attr[str]
    """ Image type, for example ``"beam_fit_params_sky"`` (matches the image
    the parameters apply to). """
    units: Attr[UnitsRadians] = "rad"
    """ Units of the fit parameters. """
    allow_multiple_versions: Attr[bool] | None = True


@xarray_dataarray_schema
class MaskArray:
    """A boolean image defining the region(s) where the deconvolution
    algorithm is allowed to place clean components (nonzero or ``True`` means
    selected). Loaders may deliver the mask as a float valued image."""

    data: Data[
        tuple[Time, Frequency, Polarization, L, M],
        numpy.float32 | numpy.float64 | bool,
    ]
    """ Mask values, nonzero/``True`` means the pixel is selected. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    l: Coordof[LCoordArray]  # noqa: E741
    m: Coordof[MCoordArray]

    type: Attr[str] = "mask"
    """ Image type. Should be ``"mask"``, or ``"mask_<suffix>"`` for versioned
    variables (the loader stamps the lower case variable name). """
    units: Attr[str] | None = None
    telescope: Attr[TelescopeDict] | None = None
    observer: Attr[str] | None = None
    obsdate: Attr[TimeMeasureArray] | None = None
    pointing_center: Attr[SkyDirectionArray] | None = None
    object_name: Attr[str] | None = None
    user: Attr[UserDict] | None = None
    description: Attr[str] | None = None
    sub_type: Attr[str] | None = None
    """ Sub image type, derived from the casacore image type, if known. """
    allow_multiple_versions: Attr[bool] | None = True


@xarray_dataarray_schema
class PrimaryBeamArray:
    """The effective antenna power pattern projected onto the sky, describing
    how sensitivity falls off from the pointing center. Values range from 1.0
    at center to ~0 at the edges."""

    data: Data[
        tuple[Time, Frequency, Polarization, L, M],
        numpy.float32 | numpy.float64,
    ]
    """ Primary beam response. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    l: Coordof[LCoordArray]  # noqa: E741
    m: Coordof[MCoordArray]

    type: Attr[str] = "primary_beam"
    """ Image type. Should be ``"primary_beam"``, or ``"primary_beam_<suffix>"``
    for versioned variables (the loader stamps the lower case variable name). """
    units: Attr[str] | None = None
    telescope: Attr[TelescopeDict] | None = None
    observer: Attr[str] | None = None
    obsdate: Attr[TimeMeasureArray] | None = None
    pointing_center: Attr[SkyDirectionArray] | None = None
    object_name: Attr[str] | None = None
    user: Attr[UserDict] | None = None
    description: Attr[str] | None = None
    sub_type: Attr[str] | None = None
    """ Sub image type, derived from the casacore image type, if known. """
    allow_multiple_versions: Attr[bool] | None = True


@xarray_dataarray_schema
class PointSpreadFunctionArray:
    """The instrumental response or "dirty beam." Represents how a point
    source would appear given the uv-coverage and weighting scheme. Used by
    deconvolution algorithms to model and remove sidelobe artifacts.
    Determines the resolution of the image. Should be unity at the peak."""

    data: Data[
        tuple[Time, Frequency, Polarization, L, M],
        numpy.float32 | numpy.float64,
    ]
    """ Point spread function. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    l: Coordof[LCoordArray]  # noqa: E741
    m: Coordof[MCoordArray]

    type: Attr[str] = "point_spread_function"
    """ Image type. Should be ``"point_spread_function"``, or
    ``"point_spread_function_<suffix>"`` for versioned variables (the loader
    stamps the lower case variable name). """
    units: Attr[str] | None = None
    telescope: Attr[TelescopeDict] | None = None
    observer: Attr[str] | None = None
    obsdate: Attr[TimeMeasureArray] | None = None
    pointing_center: Attr[SkyDirectionArray] | None = None
    object_name: Attr[str] | None = None
    user: Attr[UserDict] | None = None
    description: Attr[str] | None = None
    beam_fit_params: Attr[str] | None = None
    """ Name of the beam fit parameters variable that applies to this image,
    for example 'BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION'. """
    sub_type: Attr[str] | None = None
    """ Sub image type, derived from the casacore image type, if known. """
    allow_multiple_versions: Attr[bool] | None = True


@xarray_dataarray_schema
class VisibilityNormalizationArray:
    """The aggregate of the image weights (the sum of gridded weights per
    plane) and used to normalize the gridded visibilities. Loaded from the
    CASA sum of weights ('sumwt') image."""

    data: Data[
        tuple[Time, Frequency, Polarization],
        numpy.float32 | numpy.float64,
    ]
    """ Sum of weights per image plane. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]

    type: Attr[Literal["visibility_normalization"]] = "visibility_normalization"
    """ Image type. Should be ``"visibility_normalization"``. """
    units: Attr[str] | None = None
    telescope: Attr[TelescopeDict] | None = None
    observer: Attr[str] | None = None
    obsdate: Attr[TimeMeasureArray] | None = None
    pointing_center: Attr[SkyDirectionArray] | None = None
    object_name: Attr[str] | None = None
    user: Attr[UserDict] | None = None
    description: Attr[str] | None = None
    sub_type: Attr[str] | None = None
    """ Sub image type, derived from the casacore image type, if known. """


@xarray_dataarray_schema
class VisibilityArray:
    """Gridded visibilities, on the aperture plane. The sky image is created
    from these using a Fourier transform. Only used internally and for
    debugging."""

    data: Data[
        tuple[Time, Frequency, Polarization, U, V],
        numpy.complex64 | numpy.complex128,
    ]
    """ Gridded visibility values. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    u: Coordof[UCoordArray]
    v: Coordof[VCoordArray]

    type: Attr[Literal["visibility"]] = "visibility"
    """ Image type. Should be ``"visibility"``. """
    units: Attr[str] | None = None
    telescope: Attr[TelescopeDict] | None = None
    observer: Attr[str] | None = None
    obsdate: Attr[TimeMeasureArray] | None = None
    pointing_center: Attr[SkyDirectionArray] | None = None
    object_name: Attr[str] | None = None
    user: Attr[UserDict] | None = None
    description: Attr[str] | None = None
    sub_type: Attr[str] | None = None
    """ Sub image type, derived from the casacore image type, if known. """


@xarray_dataarray_schema
class UvSamplingArray:
    """Gridded weights, on the aperture plane. The point spread function is
    created from these using a Fourier transform. Only used internally and for
    debugging."""

    data: Data[
        tuple[Time, Frequency, Polarization, U, V],
        numpy.float32 | numpy.float64 | numpy.complex64 | numpy.complex128,
    ]
    """ Gridded weight values. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    u: Coordof[UCoordArray]
    v: Coordof[VCoordArray]

    type: Attr[Literal["uv_sampling"]] = "uv_sampling"
    """ Image type. Should be ``"uv_sampling"``. """
    units: Attr[str] | None = None
    telescope: Attr[TelescopeDict] | None = None
    observer: Attr[str] | None = None
    obsdate: Attr[TimeMeasureArray] | None = None
    pointing_center: Attr[SkyDirectionArray] | None = None
    object_name: Attr[str] | None = None
    user: Attr[UserDict] | None = None
    description: Attr[str] | None = None
    sub_type: Attr[str] | None = None
    """ Sub image type, derived from the casacore image type, if known. """


@xarray_dataarray_schema
class UvSamplingNormalizationArray:
    """Normalization used by the uv sampling. This is the sum of weights; the
    sensitivity can be calculated using 1/sqrt(uv_sampling_normalization)."""

    data: Data[
        tuple[Time, Frequency, Polarization],
        numpy.float32 | numpy.float64,
    ]
    """ Sum of weights per plane. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]

    type: Attr[Literal["uv_sampling_normalization"]] = "uv_sampling_normalization"
    """ Image type. Should be ``"uv_sampling_normalization"``. """
    units: Attr[str] | None = None


@xarray_dataarray_schema
class ApertureArray:
    """Gridded weighted apertures, on the aperture plane. Used to calculate
    the primary beam (the aperture is the Fourier transform of the primary
    beam). Only used internally and for debugging."""

    data: Data[
        tuple[Time, Frequency, Polarization, U, V],
        numpy.complex64 | numpy.complex128,
    ]
    """ Gridded aperture values. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]
    u: Coordof[UCoordArray]
    v: Coordof[VCoordArray]

    type: Attr[Literal["aperture"]] = "aperture"
    """ Image type. Should be ``"aperture"``. """
    units: Attr[str] | None = None
    telescope: Attr[TelescopeDict] | None = None
    observer: Attr[str] | None = None
    obsdate: Attr[TimeMeasureArray] | None = None
    pointing_center: Attr[SkyDirectionArray] | None = None
    object_name: Attr[str] | None = None
    user: Attr[UserDict] | None = None
    description: Attr[str] | None = None
    sub_type: Attr[str] | None = None
    """ Sub image type, derived from the casacore image type, if known. """


@xarray_dataarray_schema
class ApertureNormalizationArray:
    """Normalization used by the aperture data."""

    data: Data[
        tuple[Time, Frequency, Polarization],
        numpy.float32 | numpy.float64,
    ]
    """ Aperture normalization per plane. """

    time: Coordof[TimeCoordArray]
    frequency: Coordof[FrequencyCoordArray]
    polarization: Coordof[PolarizationArray]

    type: Attr[Literal["aperture_normalization"]] = "aperture_normalization"
    """ Image type. Should be ``"aperture_normalization"``. """
    units: Attr[str] | None = None


# Data Sets
@xarray_dataset_schema
class ImageXds:
    """Image dataset.

    Holds one or more images that share coordinates. Data variables can
    appear in multiple versions distinguished by an underscore separated
    suffix (for example ``SKY``, ``SKY_DECONVOLVED``, ``SKY_MODEL``,
    ``FLAG_SKY``, ``BEAM_FIT_PARAMS_SKY``). The ``data_groups`` attribute
    groups the variables that belong together and maps logical roles to
    concrete variable names.

    Sky plane images are defined on the ``(l, m)`` projection plane
    dimensions (the direction cosines for the SIN projection, see
    :py:class:`LCoordArray`), aperture plane data on the conjugate ``(u, v)``
    dimensions.
    """

    # --- Required Coordinates ---
    time: Coordof[TimeCoordArray]
    """ Observation time (typically a single value, the observation date). """
    frequency: Coordof[FrequencyCoordArray]
    """ Center frequencies for each channel. """
    polarization: Coordof[PolarizationArray]
    """ Labels for polarization types, e.g. ``['I', 'Q', 'U', 'V']``, in the
    order of the casacore ``Stokes`` enumeration (``I, Q, U, V``,
    ``RR, RL, LR, LL``, ``XX, XY, YX, YY``), so that the correlations of a
    pair of feeds map directly onto 2x2 Jones matrices. The readers return
    this order: FITS files cannot store the correlations in it, so the FITS
    writer reorders the planes in the file and the reader restores the order,
    and CASA images or zarr stores that hold another order are reordered when
    they are read. The ``make_empty_*`` factories require this order and
    :py:func:`check_image` reports any other. """

    # --- Required Attributes ---
    data_groups: Attr[DataGroupsDict]
    """ Defines groups of image variables that belong together, mapping
    logical roles (sky, flag, point_spread_function, ...) to data variable
    names. """

    # --- Optional Coordinates ---
    velocity: Coordof[VelocityCoordArray] | None = None
    """ Velocity of each frequency channel (non-dimensional coordinate parallel to ``frequency``). """
    l: Coordof[LCoordArray] | None = None  # noqa: E741
    """ Projection plane coordinate towards the east (sky plane images; the
    direction cosine for the SIN projection). """
    m: Coordof[MCoordArray] | None = None
    """ Projection plane coordinate towards the north (sky plane images; the
    direction cosine for the SIN projection). """
    u: Coordof[UCoordArray] | None = None
    """ Aperture plane coordinate conjugate to l. """
    v: Coordof[VCoordArray] | None = None
    """ Aperture plane coordinate conjugate to m. """
    beam_params_label: Coordof[BeamParamsLabelCoordArray] | None = None
    """ Labels of the Gaussian beam fit parameters ('major', 'minor', 'pa'). """
    right_ascension: Coord[tuple[L, M], float] | None = None
    """ Right ascension of each pixel, in radians (only if sky coordinates were computed). """
    declination: Coord[tuple[L, M], float] | None = None
    """ Declination of each pixel, in radians (only if sky coordinates were computed). """
    galactic_longitude: Coord[tuple[L, M], float] | None = None
    """ Galactic longitude of each pixel, in radians (images in the galactic frame). """
    galactic_latitude: Coord[tuple[L, M], float] | None = None
    """ Galactic latitude of each pixel, in radians (images in the galactic frame). """

    # --- Optional data variables / arrays ---
    SKY: Dataof[SkyArray] | None = None
    """ Sky image (multiple versions allowed, e.g. ``SKY_DECONVOLVED``). """
    FLAG: Dataof[FlagArray] | None = None
    """ Pixel flags (multiple versions allowed, e.g. ``FLAG_SKY``). """
    BEAM_FIT_PARAMS: Dataof[BeamFitParamsArray] | None = None
    """ Gaussian beam fit parameters (multiple versions allowed, e.g. ``BEAM_FIT_PARAMS_SKY``). """
    MASK: Dataof[MaskArray] | None = None
    """ Deconvolution mask. """
    PRIMARY_BEAM: Dataof[PrimaryBeamArray] | None = None
    """ Primary beam response. """
    POINT_SPREAD_FUNCTION: Dataof[PointSpreadFunctionArray] | None = None
    """ Point spread function. """
    VISIBILITY_NORMALIZATION: Dataof[VisibilityNormalizationArray] | None = None
    """ Sum of weights used to normalize the gridded visibilities. """
    VISIBILITY: Dataof[VisibilityArray] | None = None
    """ Gridded visibilities (aperture plane). """
    UV_SAMPLING: Dataof[UvSamplingArray] | None = None
    """ Gridded weights (aperture plane). """
    UV_SAMPLING_NORMALIZATION: Dataof[UvSamplingNormalizationArray] | None = None
    """ Normalization of the gridded weights. """
    APERTURE: Dataof[ApertureArray] | None = None
    """ Gridded weighted apertures (aperture plane). """
    APERTURE_NORMALIZATION: Dataof[ApertureNormalizationArray] | None = None
    """ Normalization of the aperture data. """

    # --- Optional Attributes ---
    coordinate_system_info: Attr[CoordinateSystemInfoDict] | None = None
    """ World coordinate system information (present for sky images). """

    type: Attr[Literal["image_dataset"]] = "image_dataset"
    """ Dataset type. """


# Mapping of data group roles to the array schemas their variables must
# conform to. The remaining DataGroupDict keys ("description", "date") do not
# reference data variables.
DATA_GROUP_ROLE_SCHEMAS = {
    "sky": SkyArray,
    "flag": FlagArray,
    "point_spread_function": PointSpreadFunctionArray,
    "primary_beam": PrimaryBeamArray,
    "mask": MaskArray,
    "beam_fit_params_sky": BeamFitParamsArray,
    "beam_fit_params_point_spread_function": BeamFitParamsArray,
    "visibility": VisibilityArray,
    "visibility_normalization": VisibilityNormalizationArray,
    "uv_sampling": UvSamplingArray,
    "uv_sampling_normalization": UvSamplingNormalizationArray,
    "aperture": ApertureArray,
    "aperture_normalization": ApertureNormalizationArray,
}


# Attributes of image variables that name another data variable, with the
# array schema the named variable must conform to, by the array schema of the
# image variables that carry them.
_REFERENCE_ATTRIBUTE_SCHEMAS = {
    SkyArray: {"flag": FlagArray, "beam_fit_params": BeamFitParamsArray},
    PointSpreadFunctionArray: {"beam_fit_params": BeamFitParamsArray},
}


def check_image(image_xds: xarray.Dataset) -> SchemaIssues:
    """Check an image dataset against the image schema.

    Runs :py:func:`xradio.schema.check.check_dataset` with the
    :py:class:`ImageXds` schema, which checks the data variables whose names
    match a data variable of the schema (for example ``SKY_RESIDUAL`` as a
    version of ``SKY``), and adds the checks it cannot do:

    * every data group is a dictionary that conforms to
      :py:class:`DataGroupDict`, and has no unknown roles;
    * every data variable referenced by a data group role exists and
      conforms to the array schema of its role (see
      ``DATA_GROUP_ROLE_SCHEMAS``), whatever its name;
    * the ``flag`` and ``beam_fit_params`` attributes of sky images, and the
      ``beam_fit_params`` attribute of point spread functions, name existing
      data variables that conform to :py:class:`FlagArray` and
      :py:class:`BeamFitParamsArray`;
    * the ``beam_params_label`` coordinate holds ``'major', 'minor', 'pa'``
      in this order, the order of the beam fit parameters in the schema (the
      writers select them by label, but other software indexes them by
      position);
    * the polarization labels are in canonical (casacore ``Stokes``, Jones
      matrix) order, see :py:class:`ImageXds`.

    A defect of a data variable is reported once, under the name of that
    variable, also when several data groups reference it. A defect of a
    coordinate is reported for the dataset and for each data variable that
    has the coordinate.

    The check covers structure and vocabularies (dimensions, dtypes and
    attribute types and values). Apart from the beam parameter labels and the
    order of the polarization labels it does not check coordinate values, nor
    the consistency of attributes with each
    other, and a conforming dataset is not necessarily writable in every
    format: the CASA and FITS writers raise
    for example for frequency axes that are not uniformly spaced, for
    spectral frames without a casacore equivalent (``icrs``, ``hcrs`` and
    ``lsr``) and, for FITS, for polarization sets that cannot form a FITS
    ``STOKES`` axis.

    :param image_xds: Image dataset to check
    :returns: List of schema issues found (empty if the dataset conforms)
    """

    issues = check_dataset(image_xds, ImageXds)

    # Names of the data variables checked so far against each array schema
    # (by schema name), so that check_image checks every variable once per
    # schema: check_dataset has already checked the variables whose names
    # match a data variable of the dataset schema
    checked = {}
    for data_var_schema in xarray_dataclass_to_dataset_schema(ImageXds).data_vars:
        checked.setdefault(data_var_schema.schema_name, set()).update(
            _matching_data_var_names(image_xds.data_vars, data_var_schema)
        )

    data_groups = image_xds.attrs.get("data_groups")
    # Missing or wrong-typed data_groups has already been reported by
    # check_dataset
    if isinstance(data_groups, dict):
        issues += _check_data_groups(image_xds, data_groups, checked)

    issues += _check_reference_attributes(image_xds, checked)
    issues += _check_beam_params_labels(image_xds)
    issues += _check_polarization_order(image_xds)
    return issues


def _check_referenced_variable(
    image_xds: xarray.Dataset, variable_name: str, array_schema: type, checked: dict
) -> SchemaIssues:
    """Check an existing data variable against an array schema, unless it
    has already been checked against that schema.

    :param image_xds: Image dataset
    :param variable_name: Name of a data variable of ``image_xds``
    :param array_schema: Array schema class the variable must conform to
    :param checked: Names of the variables checked so far, by schema name
        (updated)
    :returns: Schema issues found
    """

    names = checked.setdefault(
        xarray_dataclass_to_array_schema(array_schema).schema_name, set()
    )
    if variable_name in names:
        return SchemaIssues()
    names.add(variable_name)
    return check_array(image_xds[variable_name], array_schema).at_path(
        "data_vars", variable_name
    )


def _check_data_groups(
    image_xds: xarray.Dataset, data_groups: dict, checked: dict
) -> SchemaIssues:
    """Check the data groups of an image dataset and the variables their
    roles reference.

    :param image_xds: Image dataset
    :param data_groups: The ``data_groups`` attribute of ``image_xds``
    :param checked: Names of the variables checked so far, by schema name
        (updated)
    :returns: Schema issues found
    """

    issues = SchemaIssues()
    data_group_keys = {
        attr.name for attr in xarray_dataclass_to_dict_schema(DataGroupDict).attributes
    }

    for group_name, group in data_groups.items():
        if not isinstance(group, dict):
            issues += SchemaIssues(
                [
                    SchemaIssue(
                        path=[("attrs", "data_groups")],
                        message=f"Data group '{group_name}' is not a dictionary!",
                        found=type(group),
                        expected=[dict],
                    )
                ]
            )
            continue

        # The "base" group is already validated against DataGroupDict by
        # check_dataset (through the DataGroupsDict attribute schema); only
        # check the other groups here to avoid duplicate issues.
        if group_name != "base":
            issues += check_dict(group, DataGroupDict).at_path(
                "attrs", f"data_groups['{group_name}']"
            )

        for role, variable_name in group.items():
            if role not in data_group_keys:
                issues += SchemaIssues(
                    [
                        SchemaIssue(
                            path=[("attrs", f"data_groups['{group_name}']")],
                            message=f"Unknown data group role '{role}'!",
                            found=role,
                            expected=sorted(data_group_keys),
                        )
                    ]
                )
                continue
            role_schema = DATA_GROUP_ROLE_SCHEMAS.get(role)
            if role_schema is None or not isinstance(variable_name, str):
                continue
            if variable_name not in image_xds.data_vars:
                issues += SchemaIssues(
                    [
                        SchemaIssue(
                            path=[("data_vars", variable_name)],
                            message=f"Data group '{group_name}' role '{role}' "
                            "references a missing data variable!",
                            found=None,
                            expected=[variable_name],
                        )
                    ]
                )
                continue
            issues += _check_referenced_variable(
                image_xds, variable_name, role_schema, checked
            )

    return issues


def _check_reference_attributes(
    image_xds: xarray.Dataset, checked: dict
) -> SchemaIssues:
    """Check that the attributes of image variables that name other data
    variables (see ``_REFERENCE_ATTRIBUTE_SCHEMAS``) name existing variables
    that conform to the schema of the reference.

    :param image_xds: Image dataset
    :param checked: Names of the variables checked so far, by schema name;
        the images carrying the attributes are the variables checked against
        the image's array schema (updated)
    :returns: Schema issues found
    """

    issues = SchemaIssues()
    for image_schema, references in _REFERENCE_ATTRIBUTE_SCHEMAS.items():
        images = checked.get(
            xarray_dataclass_to_array_schema(image_schema).schema_name, set()
        )
        for image_name in [name for name in image_xds.data_vars if name in images]:
            for attr_name, array_schema in references.items():
                variable_name = image_xds[image_name].attrs.get(attr_name)
                # A value of the wrong type has been reported by the array check
                if not isinstance(variable_name, str):
                    continue
                if variable_name not in image_xds.data_vars:
                    issues.add(
                        SchemaIssue(
                            path=[("data_vars", image_name), ("attrs", attr_name)],
                            message=f"References data variable '{variable_name}', "
                            "which does not exist!",
                            found=variable_name,
                        )
                    )
                    continue
                issues += _check_referenced_variable(
                    image_xds, variable_name, array_schema, checked
                )
    return issues


def _check_beam_params_labels(image_xds: xarray.Dataset) -> SchemaIssues:
    """Check the values of the ``beam_params_label`` coordinate.

    :param image_xds: Image dataset
    :returns: Schema issues found
    """

    if "beam_params_label" not in image_xds.coords:
        return SchemaIssues()
    labels = [str(label) for label in image_xds.coords["beam_params_label"].values]
    if tuple(labels) == BEAM_PARAMS_LABELS:
        return SchemaIssues()
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("coords", "beam_params_label")],
                message="Beam fit parameter labels must be 'major', 'minor', 'pa' "
                "in this order!",
                found=labels,
                expected=[list(BEAM_PARAMS_LABELS)],
            )
        ]
    )


def _check_polarization_order(image_xds: xarray.Dataset) -> SchemaIssues:
    """Check that the polarization labels are in canonical order.

    :param image_xds: Image dataset
    :returns: Schema issues found
    """

    if "polarization" not in image_xds.coords:
        return SchemaIssues()
    labels = [str(label) for label in image_xds.coords["polarization"].values]
    order = canonical_polarization_order(labels)
    if order == list(range(len(labels))):
        return SchemaIssues()
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("coords", "polarization")],
                message="Polarization labels must be in canonical (casacore "
                "Stokes, Jones matrix) order, for example RR, RL, LR, LL!",
                found=labels,
                expected=[[labels[i] for i in order]],
            )
        ]
    )
