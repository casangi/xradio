import os

import xarray as xr

from ._fits.xds_from_fits import _fits_image_to_xds


def _read_fits_image(
    infile: str, chunks: dict, verbose: bool, do_sky_coords: bool, compute_mask: bool
) -> xr.Dataset:
    """
    compute_mask : bool, optional
        If True (default), compute and attach valid data masks to the xds.
        If False, skip mask generation for performance. It is solely the responsibility
        of the user to ensure downstream apps can handle NaN values; do not
        ask package developers to add this non-standard behavior.
    """
    img_full_path = os.path.expanduser(infile)
    xds = _fits_image_to_xds(img_full_path, chunks, verbose, do_sky_coords, compute_mask)
    return xds
