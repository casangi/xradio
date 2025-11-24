#################################
# Helper File
#
# Not exposed in API
#
#################################
import os
import warnings
from typing import Union

import xarray as xr
import re

from xradio._utils.schema import get_data_group_keys

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from ._casacore.common import _open_image_ro
from ._casacore.xds_from_casacore import (
    _add_mask,
    _add_sky_or_aperture,
    _casa_image_to_xds_attrs,
    _casa_image_to_xds_coords,
    _get_mask_names,
    _get_persistent_block,
    _get_starts_shapes_slices,
    _get_transpose_list,
    _get_beam,
    _read_image_array,
)
from ._casacore.xds_to_casacore import (
    _coord_dict_from_xds,
    _history_from_xds,
    _imageinfo_dict_from_xds,
    _write_casa_data,
)
from .common import _aperture_or_sky, _get_xds_dim_order, _dask_arrayize_dv

warnings.filterwarnings("ignore", category=FutureWarning)


def _load_casa_image_block(
    infile: str, block_des: dict, do_sky_coords, image_type: str
) -> xr.Dataset:
    image_full_path = os.path.expanduser(infile)
    with _open_image_ro(image_full_path) as casa_image:
        coords = casa_image.coordinates()
        cshape = casa_image.shape()
    ret = _casa_image_to_xds_coords(image_full_path, False, do_sky_coords)
    xds = ret["xds"].isel(block_des)
    nchan = ret["xds"].sizes["frequency"]
    npol = ret["xds"].sizes["polarization"]
    starts, shapes, slices = _get_starts_shapes_slices(block_des, coords, cshape)
    dimorder = _get_xds_dim_order(ret["sphr_dims"])
    transpose_list, new_axes = _get_transpose_list(coords)
    block = _get_persistent_block(
        image_full_path, shapes, starts, dimorder, transpose_list, new_axes
    )
    xds = _add_sky_or_aperture(
        xds, block, dimorder, image_full_path, ret["sphr_dims"], False, image_type
    )
    mymasks = _get_mask_names(image_full_path)
    for m in mymasks:
        full_path = os.sep.join([image_full_path, m])
        block = _get_persistent_block(
            full_path, shapes, starts, dimorder, transpose_list, new_axes
        )
        # data vars are all caps by convention
        mask_name = re.sub(r"\bMASK(\d+)\b", r"MASK_\1", m.upper())
        xds = _add_mask(xds, mask_name, block, dimorder)
    xds.attrs = _casa_image_to_xds_attrs(image_full_path)
    beam = _get_beam(image_full_path, nchan, npol, False)

    if beam is not None:
        selectors = {
            k: block_des[k]
            for k in ("time", "frequency", "polarization")
            if k in block_des
        }
        xds["BEAM_FIT_PARAMS"] = beam.isel(selectors)
    return xds


def _open_casa_image(
    infile: str,
    chunks: Union[list, dict],
    verbose: bool,
    do_sky_coords: bool,
    masks: bool = True,
    history: bool = False,
    image_type: str = "SKY",
) -> xr.Dataset:
    img_full_path = os.path.expanduser(infile)
    ret = _casa_image_to_xds_coords(img_full_path, verbose, do_sky_coords)
    xds = ret["xds"]
    dimorder = _get_xds_dim_order(ret["sphr_dims"])
    xds = _add_sky_or_aperture(
        xds,
        _read_image_array(img_full_path, chunks, verbose=verbose),
        dimorder,
        img_full_path,
        ret["sphr_dims"],
        history,
        image_type=image_type,
    )
    if masks:
        mymasks = _get_mask_names(img_full_path)
        for m in mymasks:
            ary = _read_image_array(img_full_path, chunks, mask=m, verbose=verbose)
            # data var names are all caps by convention
            mask_name = re.sub(r"\bMASK(\d+)\b", r"MASK_\1", m.upper())
            xds = _add_mask(xds, mask_name, ary, dimorder)
    xds.attrs = _casa_image_to_xds_attrs(img_full_path)
    beam = _get_beam(
        img_full_path, xds.sizes["frequency"], xds.sizes["polarization"], True
    )
    if beam is not None:
        xds["BEAM_FIT_PARAMS"] = beam
    # xds = _add_coord_attrs(xds, ret["icoords"], ret["dir_axes"])
    xds = _dask_arrayize_dv(xds)
    return xds


def _xds_to_multiple_casa_images(xds: xr.Dataset, image_store_name: str) -> None:
    """Function disentagles xradio xr.Dataset into multiple casa images based on data_groups attribute.
    An xr.Dataset may contain multiple images (sky, residual, psf, etc) stored under different data variables sharing common coordinates.
    An addtional complication is that CASA images allow for internal masks and beam fit parameters to be stored alongside the main image data so these also need to be handled.
    This function creates separate casa images for each image type found in the data_groups attribute of the xr.Dataset.

    Parameters
    ----------
    xds : xr.Dataset
        The xradio xr.Dataset containing multiple images and associated data.
    image_store_name : str
        The base name or path for storing the output CASA images.
    """

    data_group_keys = list(get_data_group_keys(schema_name="image").keys())
    internal_image_types_to_exclude = ["mask_sky", "mask_residual", "beam_fit_params"]

    for data_group in xds.attrs["data_groups"].keys():
        for image_type in data_group_keys:
            if (image_type in xds.attrs["data_groups"][data_group]) and (
                image_type not in internal_image_types_to_exclude
            ):
                image_name = xds.attrs["data_groups"][data_group][image_type]
                if image_name in xds.data_vars:
                    image_to_write_xds = xr.Dataset()
                    image_to_write_xds.attrs = xds.attrs.copy()

                    # This code handles adding internal masks and beam fit params if they exist.
                    if image_type == "sky":
                        if "beam_fit_params" in xds.attrs["data_groups"][data_group]:
                            beam_fit_params_name = xds.attrs["data_groups"][data_group][
                                "beam_fit_params"
                            ]
                            image_to_write_xds["BEAM_FIT_PARAMS"] = xds[
                                beam_fit_params_name
                            ]

                        if "mask_sky" in xds.attrs["data_groups"][data_group]:
                            mask_sky_name = xds.attrs["data_groups"][data_group][
                                "mask_sky"
                            ]
                            image_to_write_xds["MASK_0"] = xds[mask_sky_name]

                    if image_type == "residual":
                        if "beam_fit_params" in xds.attrs["data_groups"][data_group]:
                            beam_fit_params_name = xds.attrs["data_groups"][data_group][
                                "beam_fit_params"
                            ]
                            image_to_write_xds["BEAM_FIT_PARAMS"] = xds[
                                beam_fit_params_name
                            ]

                        if "mask_residual" in xds.attrs["data_groups"][data_group]:
                            mask_sky_name = xds.attrs["data_groups"][data_group][
                                "mask_sky"
                            ]
                            image_to_write_xds["MASK_0"] = xds[mask_sky_name]

                    if image_type == "point_spread_function":
                        if "beam_fit_params" in xds.attrs["data_groups"][data_group]:
                            beam_fit_params_name = xds.attrs["data_groups"][data_group][
                                "beam_fit_params"
                            ]
                            image_to_write_xds["BEAM_FIT_PARAMS"] = xds[
                                beam_fit_params_name
                            ]

                    image_to_write_xds["SKY"] = xds[
                        image_name
                    ]  # Setting everying to sky for writing purposes.

                    _xds_to_casa_image(
                        image_to_write_xds, image_store_name + "." + image_type
                    )


def _xds_to_casa_image(xds: xr.Dataset, image_store_name: str) -> None:
    image_full_path = os.path.expanduser(image_store_name)
    _write_casa_data(xds, image_full_path)
    # create coordinates
    ap_sky = _aperture_or_sky(xds)
    coord = _coord_dict_from_xds(xds)
    ii = _imageinfo_dict_from_xds(xds)
    units = xds[ap_sky].attrs["units"] if "units" in xds[ap_sky].attrs else None
    miscinfo = (
        xds.attrs["user"]
        if "user" in xds.attrs and len(xds.attrs["user"]) > 0
        else None
    )
    tb = tables.table(
        image_full_path,
        readonly=False,
        lockoptions={"option": "permanentwait"},
        ack=False,
    )

    tb.putkeyword("coords", coord)
    tb.putkeyword("imageinfo", ii)
    if units:
        tb.putkeyword("units", units)
    if miscinfo:
        tb.putkeyword("miscinfo", miscinfo)
    tb.done()
    # history
    _history_from_xds(xds, image_full_path)
