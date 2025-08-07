from __future__ import annotations

from typing import Literal, Optional, Union
from xradio.schema.bases import (
    xarray_dataset_schema,
    xarray_dataarray_schema,
    dict_schema,
)
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
from xradio.measurement_set.schema import (
    ZD,
    Time,
    TimeCoordArray,
    Frequency,
    FrequencyArray,
    SkyCoordArray,
    UnitsDimensionless,
    Polarization,
    PolarizationArray,
    SkyDirLabel,
    SkyPosLabel,
    QuantityInRadiansArray,
    QuantityInMetersArray
)
import numpy
import dataclasses


# https://docs.google.com/spreadsheets/d/1WW0Gl6z85cJVPgtdgW4dxucurHFa06OKGjgoK8OREFA

IMAGE_SCHEMA_VERSION = "4.0.-9999"

LCoord = Literal["l"]
MCoord = Literal["m"]
UCoord = Literal["u"]
VCoord = Literal["v"]
LMLabelIn = Literal["lm_in"]
LMLabelOut = Literal["lm_out"]
BeamParam = Literal["beam_param"]
UnitsJansky = Literal["Jy"]

@xarray_dataarray_schema
class CosineArray:
    """
    Directional cosine coordinate.
    """

    data: Data[ZD, numpy.float64]

    units: Attr[UnitsDimensionless] = ""

@xarray_dataarray_schema
class ApertureCoordArray:
    """
    Directional cosine coordinate.
    """

    data: Data[ZD, numpy.float64]

    units: Attr[UnitsDimensionless] = ""

    # Likely not right, should be in lambda (wavelengths?)
    reference_value: Attr[QuantityInMetersArray]
    """
    World reference value. Note crpix purposefully omitted because
    crpix cannot be reliably updated when selecting regions/subimages
    using standard xarray selection methods. Use a world2pix function
    if crpix is required in other computations.
    """

# TODO: Jy/beam? rad?!
UnitsImage = Literal[["Jy/beam"], ["Jy/pixel"], ["rad"], [""]]

@xarray_dataarray_schema
class ImageArray:
    """
    Astronomical image - mapping of sky coordinates to intensities
    """

    data: Data[
        Union[
            tuple[Time, Frequency, Polarization, LCoord, MCoord]
        ],
        Union[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
    ]
    
    image_type: Optional[Attr[str]] = None
    """type of image. eg, 'Intensity', 'spix', "mask", "beam" etc, can be blank"""

    active_mask: Optional[Attr[str]] = None
    """ Default mask that should be used by processing applications
    (would be interesting to allow an expresion here for eg boolean
    operations on multiple masks, eg ANDing two masks together, but
    that's probably best left to individual package implementations) """

    units: Attr[UnitsImage] = ("Jy",)


@xarray_dataarray_schema
class ApertureArray:
    """
    Aperture image - mapping of u/v coordinates to intensities
    """

    data: Data[
        Union[
            tuple[Time, Frequency, Polarization, UCoord, VCoord]
        ],
        Union[numpy.complex64, numpy.complex128]
    ]

    # "attributes are generally the same as for sky images"
    
    image_type: Optional[Attr[str]] = None
    """type of image. eg, 'Intensity', 'spix', "mask", "beam" etc, can be blank"""

    active_mask: Optional[Attr[str]] = None
    """ Default mask that should be used by processing applications
    (would be interesting to allow an expresion here for eg boolean
    operations on multiple masks, eg ANDing two masks together, but
    that's probably best left to individual package implementations) """

    units: Attr[UnitsImage] = ("Jy",)

@xarray_dataarray_schema
class LinearTransformArray:
    """
    Matrix describing linear transform
    Directional cosine coordinate.
    """

    data: Data[tuple[LMLabelOut, LMLabelIn], numpy.float64]

    lm_in: Coord[LMLabelIn, str] = ('l', 'm')
    lm_out: Coord[LMLabelIn, str] = ("l'", "m'")

    units: Attr[UnitsImage] = ("",)

AllowedProjections = Literal[
    "AZP", "TAN", "SIN", "STG", "ARC",
    "ZPN", "ZEA", "AIR", "CYP", "CAR", "MER", "CEA", "COP", "COD",
    "COE", "COO", "BON", "PCO", "SFL", "PAR", "AIT", "MOL", "CSC",
    "QSC", "TSC", "SZP", "HPX"
]

@xarray_dataarray_schema
class BeamParamArray:
    """
    Coordinate axis to make up ``("major", "minor", "pa")`` tuple
    """

    data: Data[BeamParam, str] = ("major", "minor", "pa")
    """Should be ``('major','minor','pa')``"""
    long_name: Optional[Attr[str]] = "Beam parameter label"
    """ Long-form name to use for axis. Should be ``"Beam parameter label"``"""

@dict_schema
class DirectionDict:
    latpole: QuantityInRadiansArray
    """Latitude of pole for reference frame, in radians"""
    lonpol: QuantityInRadiansArray
    """Longitude of pole for reference frame, in radians"""
    projection: AllowedProjections
    """Direction coordinate projection, eg SIN"""
    reference: SkyCoordArray
    """Reference world coordinate for direction (essentially crval +
    unit). Note that crpix is purposefully excluded; see notes for l
    and m above)"""
    pc: LinearTransformArray = ((1.0, 0.0), (0.0, 1.0))
    """Matrix describing linear transform"""
    projection_parameters: list[float] = (0.0, 0.0)
    """ Array describing projection, number of elements depends on the projection"""

@dataclasses.dataclass(frozen=True)
class BaseImageXds:

    # --- Coordinates ---
    
    time: Coordof[TimeCoordArray]
    """
    Normally one or a small number of planes. If unity, value
    should be the same as in the image coordinate system obsdate
    """
    
    frequency: Coordof[FrequencyArray]
    """
    frequency -> chan mapping
    """

    polarization: Coordof[PolarizationArray]

    beam_param: Coordof[BeamParamArray]

    # --- Attributes ---

    reference_frequency: Attr[FrequencyArray]
    """ TODO: Document """

    # Note that this would *not* actually share the Frequency
    # dimension. Okay, as it is an attribute?
    rest_frequencies: Attr[FrequencyArray]
    """ List of relevant rest frequencies. At a minimum will include frequency.rest_frequency"""
    # Not actually an array, just a naked quantity
    rest_frequency: Attr[FrequencyArray]
    """ Frequency used for velocity conversion. Must be in the frequency.rest_frequencies list. """

    single_beam: Optional[Attr[bool]]
    """
    Indicates if there is a single, global beam, i.e.
    ``BEAM`` is the same for all time steps, frequencies and
    polarisations.
    """

    # TODO: History?
    
    # --- Data variables ---

    IMAGE_CENTER: Optional[Data[tuple[Time], SkyCoordArray]]
    """
    Pointing center information

    Identifies the on-sky direction of the center of the image
    """
    
    velocity: Optional[Coord[Frequency, numpy.float64]]
    """ velocity, optional, allows for direct chan -> velocity or freq -> velocity mapping """
    
    sky_dir_label: Optional[Coord[SkyDirLabel, str]] = None
    """ Coordinate labels of sky directions (typically shape 2 and 'ra', 'dec') """
    
    sky_pos_label: Optional[Coord[SkyPosLabel, str]] = None
    """ Coordinate lables of sky positions (typically shape 3 and 'ra', 'dec', 'dist') """

    
@xarray_dataset_schema
class AstroImageXds(BaseImageXds):

    l: Coord[LCoord, CosineArray]
    """l direction cosine. Increases into direction of right ascension (RA) axis in image centre, but is not a longitude."""
    m: Coord[MCoord, CosineArray]
    """m direction cosine. Increases into direction of declination (Dec) axis in image centre, but is not a latitude."""

    # --- Data variables ---

    SKY: Optional[Dataof[ImageArray]]
    SKY_MODEL: Optional[Dataof[ImageArray]]
    PSF: Optional[Dataof[ImageArray]]
    RESIDUAL: Optional[Dataof[ImageArray]]
    
    MASK: Optional[Dataof[ImageArray]]
    """
    Image mask. Use names like ``MASK_[name]`` if there are multiple
    masks.

    Expected to use mask convention inverse of CASA6, True=good,
    False=bad.

    Synthesized beam data vars have coordinates time,
    polarization, frequency, and beam_param. In the case of a single,
    global beam, the values are repeated for each (time, polarization,
    frequency) tuple.  """

    BEAM: Optional[Data[tuple[Time, Polarization, Frequency, BeamParam], float]]
    """
    Synthesized beam parameters (minor and major axis as well as
    position angle).
    
    In the case of a single, global beam, the values are repeated
    for each (time, polarization, frequency) tuple.
    """

@xarray_dataset_schema
class ApertureImageXds(BaseImageXds):

    u: Coord[UCoord, ApertureCoordArray]
    """For Fourier images, u coordinate"""
    v: Coord[VCoord, ApertureCoordArray]
    """For Fourier images, v coordinate"""

    # --- Data variables ---

    APERTURE: Dataof[ApertureArray]
