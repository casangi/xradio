import os

import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio._utils.schema import get_data_group_keys
from xradio.image._util._fits.xds_to_fits import _xds_to_fits_image


def _xds_to_multiple_fits_images(xds: xr.Dataset, image_store_name: str) -> None:
    """Disentangle an xradio image dataset into multiple FITS images based on
    the data_groups attribute, mirroring the CASA writer. One FITS file is
    written per image type found in the data groups; flags are applied as NaN
    pixels (the FITS convention) and beam fit parameters are written as
    BMAJ/BMIN/BPA header cards (single beam) or a CASA style BEAMS binary
    table (per plane beams).

    Parameters
    ----------
    xds : xr.Dataset
        The xradio image dataset containing one or more images.
    image_store_name : str
        The base name or path for storing the output FITS images. If only one
        image is written, it will be named image_store_name, else the images
        will be named image_store_name.<image_type> where <image_type> is
        sky, point_spread_function, etc.
    """

    data_vars_name_set = set(xds.data_vars.keys())

    data_group_keys = list(get_data_group_keys(schema_name="image").keys())
    internal_image_types_to_exclude = [
        "flag",
        "beam_fit_params_sky",
        "beam_fit_params_point_spread_function",
    ]
    n_image_written = 0
    last_image_written = ""
    for data_group in xds.attrs["data_groups"].keys():
        for image_type in data_group_keys:
            if (image_type in xds.attrs["data_groups"][data_group]) and (
                image_type not in internal_image_types_to_exclude
            ):
                image_name = xds.attrs["data_groups"][data_group][image_type]
                if image_name in data_vars_name_set:
                    if "l" not in xds[image_name].dims:
                        xradio_logger().warning(
                            f"Not writing {image_name} to FITS: only sky "
                            "plane images (with l and m dimensions) are "
                            "supported"
                        )
                        data_vars_name_set.remove(image_name)
                        continue
                    image_to_write_xds = xr.Dataset()
                    image_to_write_xds.attrs = xds.attrs.copy()
                    image_to_write_xds["SKY"] = xds[image_name]

                    beam_fit_params_role = {
                        "sky": "beam_fit_params_sky",
                        "point_spread_function": (
                            "beam_fit_params_point_spread_function"
                        ),
                    }.get(image_type)
                    if (
                        beam_fit_params_role is not None
                        and beam_fit_params_role in xds.attrs["data_groups"][data_group]
                    ):
                        beam_fit_params_name = xds.attrs["data_groups"][data_group][
                            beam_fit_params_role
                        ]
                        image_to_write_xds["BEAM_FIT_PARAMS"] = xds[
                            beam_fit_params_name
                        ]

                    if (
                        image_type == "sky"
                        and "flag" in xds.attrs["data_groups"][data_group]
                    ):
                        flag_name = xds.attrs["data_groups"][data_group]["flag"]
                        image_to_write_xds["FLAG"] = xds[flag_name]

                    outname = image_store_name + "." + image_type
                    _xds_to_fits_image(image_to_write_xds, outname)
                    if not os.path.exists(outname):
                        raise OSError(f"Failed to write FITS image {outname}")
                    n_image_written += 1
                    last_image_written = outname
                    data_vars_name_set.remove(image_name)
    if n_image_written == 0:
        raise ValueError("No valid image types found in xds to write to FITS images.")
    if n_image_written == 1:
        # rename the single written image to what the user requested
        os.rename(last_image_written, image_store_name)
