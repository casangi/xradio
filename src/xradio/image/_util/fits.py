import logging
import os
from typing import Union
import xarray as xr
from ._fits.xds_from_fits import _fits_image_to_xds

def _read_fits_image(
    infile:str, chunks:dict, verbose:bool=False
) -> xr.Dataset:
    img_full_path = os.path.expanduser(infile)
    xds = _fits_image_to_xds(img_full_path, chunks, verbose)
    return xds
