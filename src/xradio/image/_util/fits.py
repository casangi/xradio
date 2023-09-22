import logging
import os
from typing import Union
import xarray as xr
from ._fits.fits_to_xds import __fits_image_to_xds_metadata

def __read_fits_image(
    infile:str, chunks:Union[list, dict], masks:bool=True,
    verbose:bool=False
) -> xr.Dataset:
    img_full_path = os.path.expanduser(infile)
    attrs = __fits_image_to_xds_metadata(img_full_path, verbose)
    # return attrs
    """
    xds = ret['xds']
    dimorder = __get_xds_dim_order(ret['sphr_dims'])
    xds = __add_sky_or_apeture(
        xds, __read_image_array(img_full_path, chunks, verbose=verbose),
        dimorder, img_full_path, ret['sphr_dims']
    )
    if masks:
        mymasks = __get_mask_names(img_full_path)
        for m in mymasks:
            ary = __read_image_array(img_full_path, chunks, mask=m, verbose=verbose)
            xds = __add_mask(xds, m, ary, dimorder)
    xds.attrs = __casa_image_to_xds_attrs(img_full_path, history)
    mb = __multibeam_array(xds, img_full_path)
    if mb is not None:
        xds['beam'] = mb
    xds = __add_coord_attrs(xds, ret['icoords'], ret['dir_axes'])
    return xds
    """


