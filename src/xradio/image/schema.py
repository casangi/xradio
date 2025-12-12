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
    flag_sky: Optional[str]
    """ Name of the sky pixels flags variable, for example 'FLAG_SKY'. For CASA images this is an internal mask. """
    model: Optional[str]
    """ Name of the model variable, for example 'MODEL'. On plane tangential to celestial sphere. """
    flag_model: Optional[str]
    """ Name of the model pixels flags variable, for example 'FLAG_MODEL'. For CASA images this is an internal mask. """
    residual: Optional[str]
    """ Name of the residual variable of the group, for example 'RESIDUAL'. residual = sky - model. On plane tangential to celestial sphere. """
    flag_residual: Optional[str]
    """ Name of the residual pixels flags variable, for example 'FLAG_RESIDUAL'. For CASA images this is an internal mask. """
    point_spread_function: Optional[str]
    """ Name of the point spread function variable of the group, for example 'POINT_SPREAD_FUNCTION'. On plane tangential to celestial sphere. """
    flag_point_spread_function: Optional[str]
    """ Name of the point spread function pixels flags variable, for example 'FLAG_POINT_SPREAD_FUNCTION'. For CASA images this is an internal mask. """
    primary_beam: Optional[str]
    """ Name of the primary beam variable of the group, for example 'PRIMARY_BEAM'. On plane tangential to celestial sphere. """
    flag_primary_beam: Optional[str]
    """ Name of the primary beam pixels flags variable, for example 'FLAG_PRIMARY_BEAM'. For CASA images this is an internal mask. """
    mask_deconvolve: Optional[str]
    """ Name of the deconvolution mask variable of the group, for example 'MASK_DECONVOLVE'. On plane tangential to celestial sphere. """
    beam_fit_params: Optional[str]
    """ Name of the beam fit parameters variable of the group, for example 'BEAM_FIT_PARAMETERS'. That applies to the sky, residual images and the point spread function if present. """
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
    description: str
    """ More details about the data group. """
    date: str
    """ Creation date-time, in ISO 8601 format: 'YYYY-MM-DDTHH:mm:ss.SSS'. """
