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

from casacore import tables
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
    _multibeam_array,
    _read_image_array,
)
from ._casacore.xds_to_casacore import (
    _coord_dict_from_xds,
    _history_from_xds,
    _imageinfo_dict_from_xds,
    _write_casa_data,
)
from .common import _get_xds_dim_order, _dask_arrayize_dv

warnings.filterwarnings("ignore", category=FutureWarning)


def _load_casa_image_block(infile: str, block_des: dict, do_sky_coords) -> xr.Dataset:
    image_full_path = os.path.expanduser(infile)
    with _open_image_ro(image_full_path) as casa_image:
        coords = casa_image.coordinates()
        cshape = casa_image.shape()
    ret = _casa_image_to_xds_coords(image_full_path, False, do_sky_coords)
    xds = ret["xds"].isel(block_des)
    starts, shapes, slices = _get_starts_shapes_slices(block_des, coords, cshape)
    dimorder = _get_xds_dim_order(ret["sphr_dims"])
    transpose_list, new_axes = _get_transpose_list(coords)
    block = _get_persistent_block(
        image_full_path, shapes, starts, dimorder, transpose_list, new_axes
    )
    xds = _add_sky_or_aperture(xds, block, dimorder, image_full_path, ret["sphr_dims"])
    mymasks = _get_mask_names(image_full_path)
    for m in mymasks:
        full_path = os.sep.join([image_full_path, m])
        block = _get_persistent_block(
            full_path, shapes, starts, dimorder, transpose_list, new_axes
        )
        xds = _add_mask(xds, m, block, dimorder)
    xds.attrs = _casa_image_to_xds_attrs(image_full_path, True)
    mb = _multibeam_array(xds, image_full_path, False)
    if mb is not None:
        selectors = {}
        for k in ("time", "polarization", "frequency"):
            if k in block_des:
                selectors[k] = block_des[k]
        xds["beam"] = mb.isel(selectors)
    return xds


def _read_casa_image(
    infile: str,
    chunks: Union[list, dict],
    verbose: bool,
    do_sky_coords: bool,
    masks: bool = True,
    history: bool = True,
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
    )
    if masks:
        mymasks = _get_mask_names(img_full_path)
        for m in mymasks:
            ary = _read_image_array(img_full_path, chunks, mask=m, verbose=verbose)
            xds = _add_mask(xds, m, ary, dimorder)
    xds.attrs = _casa_image_to_xds_attrs(img_full_path, history)
    mb = _multibeam_array(xds, img_full_path, True)
    if mb is not None:
        xds["beam"] = mb
    # xds = _add_coord_attrs(xds, ret["icoords"], ret["dir_axes"])
    xds = _dask_arrayize_dv(xds)
    return xds


def _xds_to_casa_image(xds: xr.Dataset, imagename: str) -> None:
    image_full_path = os.path.expanduser(imagename)
    _write_casa_data(xds, image_full_path)
    # create coordinates
    ap_sky = "sky" if "sky" in xds.data_vars else "aperture"
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
