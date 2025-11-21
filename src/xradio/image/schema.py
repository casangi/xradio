from __future__ import annotations

from typing import Literal, Optional, Union
from xradio.schema.bases import (
    xarray_dataset_schema,
    xarray_dataarray_schema,
    dict_schema,
)


@dict_schema
class DataGroupDict:
    """Defines a group of images."""

    sky: Optional[str]
    """ Name of the sky variable, for example 'SKY'. Derived from the gridded visibilities. On plane tangential to celestial sphere. """
    mask_sky: Optional[str]
    """ Name of the sky mask variable, for example 'MASK_SKY'. For CASA images this is an internal mask. """
    model: Optional[str]
    """ Name of the model variable, for example 'MODEL'. On plane tangential to celestial sphere. """
    residual: Optional[str]
    """ Name of the residual variable of the group, for example 'RESIDUAL'. residual = sky - model. On plane tangential to celestial sphere. """
    mask_residual: Optional[str]
    """ Name of the residual mask variable, for example 'MASK_RESIDUAL'. For CASA images this is an internal mask. """
    point_spread_function: Optional[str]
    """ Name of the point spread function variable of the group, for example 'POINT_SPREAD_FUNCTION'. On plane tangential to celestial sphere. """
    primary_beam: Optional[str]
    """ Name of the primary beam variable of the group, for example 'PRIMARY_BEAM'. On plane tangential to celestial sphere. """
    mask: Optional[str]
    """ Name of the mask variable of the group, for example 'MASK'. On plane tangential to celestial sphere. """
    visibility: Optional[str]
    """ Name of the visibility variable of the group, for example 'VISIBILITY'. The gridded visibilities used to create the images using a Fourier transform. On aperture plane."""
    visibility_normalization: Optional[str]
    """ Normalization factor for the gridded visibility data. """
    uv_sampling: Optional[str]
    """ Name of the uv sampling variable of the group, for example 'UV_SAMPLING'. The gridded weights used to create the point spread function using a Fourier transform. On aperture plane."""
    uv_sampling_normalization: Optional[str]
    """ Normalization factor for the gridded weights. This is the sum of weights and the sensitivity can be calculated using 1/sqrt(uv_sampling_normalization)."""
    aperture: Optional[str]
    """ Name of the aperture variable of the group, for example 'APERTURE'. On aperture plane. The aperture is the Fourier transform of the primary beam."""
    aperture_normalization: Optional[str]
    """ Normalization factor for the aperture data.  """
    beam_fit_parameters: Optional[str]
    """ Name of the beam fit parameters variable of the group, for example 'BEAM_FIT_PARAMETERS'. """
    description: str
    """ More details about the data group. """
    date: str
    """ Creation date-time, in ISO 8601 format: 'YYYY-MM-DDTHH:mm:ss.SSS'. """
