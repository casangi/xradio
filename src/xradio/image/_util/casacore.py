#################################
# Helper File
#
# Not exposed in API
#
#################################
import numbers
import os
import re
import warnings

import dask.array as da
import numpy as np
import xarray as xr

from xradio._utils.logging import xradio_logger

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables


from xradio.image._util._casacore.common import _beam_fit_params, _open_image_ro
from xradio.image._util._casacore.xds_from_casacore import (
    _add_mask,
    _add_sky_or_aperture,
    _casa_image_to_xds_attrs,
    _casa_image_to_xds_coords,
    _get_beam,
    _get_mask_names,
    _get_persistent_block,
    _get_starts_shapes_slices,
    _get_transpose_list,
    _read_image_array,
)
from xradio.image._util._casacore.xds_to_casacore import (
    _coord_dict_from_xds,
    _history_from_xds,
    _image_variable,
    _imageinfo_dict_from_xds,
    _write_casa_data,
)
from xradio.image._util._write_plan import (
    ImageOutput,
    ImageWritePlan,
    naming_the_variable,
    plan_image_outputs,
)
from xradio.image._util.common import (
    _dask_arrayize_dv,
    _get_xds_dim_order,
    _to_canonical_polarization_order,
)
from xradio.image._util.conventions import canonical_polarization_order

warnings.filterwarnings("ignore", category=FutureWarning)


def _squeeze_if_needed(ary: da, image_type: str) -> da:
    if image_type.upper() == "VISIBILITY_NORMALIZATION":
        shape = ary.shape
        if len(shape) != 5:
            raise ValueError(
                "VISIBILITY_NORMALIZATION casa image must be 5D before squeezing. "
                f"Found shape {shape}"
            )
        if shape[3] != 1 or shape[4] != 1:
            raise ValueError(
                "VISIBILITY_NORMALIZATION casa image must have l and m of length 1. "
                f"Found {(shape[3], shape[4])}"
            )
        ary = ary.squeeze(axis=(3, 4))
    return ary


def _get_casa_image_metadata(infile: str, do_sky_coords: bool, image_type: str) -> dict:
    image_full_path = os.path.expanduser(infile)
    with _open_image_ro(image_full_path) as casa_image:
        coords = casa_image.coordinates()
        cshape = casa_image.shape()
    ret = _casa_image_to_xds_coords(image_full_path, False, do_sky_coords, image_type)
    xds = ret["xds"]
    sphr_dims = ret["sphr_dims"]
    nchan = ret["xds"].sizes["frequency"]
    npol = ret["xds"].sizes["polarization"]
    dimorder = _get_xds_dim_order(ret["sphr_dims"], image_type)
    metadata = {
        "coords": coords,
        "cshape": cshape,
        "image_full_path": image_full_path,
        "nchan": nchan,
        "npol": npol,
        "dimorder": dimorder,
        # "xds_attrs": _casa_image_to_xds_attrs(image_full_path),
        "sphr_dims": sphr_dims,
        "xds": xds,
    }
    return metadata


def _load_casa_image_block(
    infile: str, block_des: dict, do_sky_coords: bool, image_type: str
) -> xr.Dataset:
    md = _get_casa_image_metadata(infile, do_sky_coords, image_type)
    # The dataset's polarization axis is in canonical order (see
    # _to_canonical_polarization_order), which the selection indexes: for an
    # image stored in another order, read every polarization, reorder, then
    # select
    pol_selection = None
    stored_pols = md["xds"].polarization.values
    if "polarization" in block_des and canonical_polarization_order(
        stored_pols
    ) != list(range(len(stored_pols))):
        pol_selection = block_des["polarization"]
        block_des = {k: v for k, v in block_des.items() if k != "polarization"}
    coords = md["coords"]
    cshape = md["cshape"]
    dimorder = md["dimorder"]
    sphr_dims = md["sphr_dims"]
    nchan = md["nchan"]
    npol = md["npol"]
    xds = md["xds"].isel(block_des)
    image_full_path = md["image_full_path"]
    starts, shapes, slices = _get_starts_shapes_slices(block_des, coords, cshape)
    transpose_list, new_axes = _get_transpose_list(coords)
    block = _get_persistent_block(
        image_full_path, shapes, starts, transpose_list, new_axes
    )
    block = _squeeze_if_needed(block, image_type)
    xds = _add_sky_or_aperture(
        xds, block, dimorder, image_full_path, sphr_dims, False, image_type
    )
    mymasks = _get_mask_names(image_full_path)
    for m in mymasks:
        full_path = os.sep.join([image_full_path, m])
        block = _get_persistent_block(
            full_path, shapes, starts, transpose_list, new_axes
        )
        block = _squeeze_if_needed(block, image_type)
        # data vars are all caps by convention
        mask_name = re.sub(r"\bMASK(\d+)\b", r"MASK_\1", m.upper())
        xds = _add_mask(xds, mask_name, block, dimorder)
    xds.attrs = _casa_image_to_xds_attrs(image_full_path)
    beam = _get_beam(image_full_path, nchan, npol, False, image_type)

    if beam is not None:
        selectors = {
            k: block_des[k]
            for k in ("time", "frequency", "polarization")
            if k in block_des
        }
        xds["BEAM_FIT_PARAMS_" + image_type.upper()] = beam.isel(selectors)
        xds["BEAM_FIT_PARAMS_" + image_type.upper()].attrs["type"] = (
            "beam_fit_params_" + image_type.lower()
        )
        xds[image_type.upper()].attrs[_beam_fit_params] = (
            "BEAM_FIT_PARAMS_" + image_type.upper()
        )
    xds = _to_canonical_polarization_order(xds)
    if pol_selection is not None:
        xds = xds.isel(polarization=pol_selection)
    return xds


def _open_casa_image(
    infile: str,
    chunks: list | dict,
    verbose: bool,
    do_sky_coords: bool,
    masks: bool = True,
    history: bool = False,
    image_type: str = "SKY",
) -> xr.Dataset:
    md = _get_casa_image_metadata(infile, do_sky_coords, image_type)
    xds = md["xds"]
    dimorder = md["dimorder"]
    sphr_dims = md["sphr_dims"]
    img_full_path = md["image_full_path"]
    ary = _read_image_array(img_full_path, chunks, verbose=verbose)
    ary = _squeeze_if_needed(ary, image_type)
    xds = _add_sky_or_aperture(
        xds,
        ary,
        dimorder,
        img_full_path,
        sphr_dims,
        history,
        image_type,
    )
    if masks:
        mymasks = _get_mask_names(img_full_path)
        for m in mymasks:
            ary = _read_image_array(img_full_path, chunks, mask=m, verbose=verbose)
            # masks have the image's shape, so squeeze them like the image
            ary = _squeeze_if_needed(ary, image_type)
            # data var names are all caps by convention
            mask_name = re.sub(r"\bMASK(\d+)\b", r"MASK_\1", m.upper())
            xds = _add_mask(xds, mask_name, ary, dimorder)
    xds.attrs = _casa_image_to_xds_attrs(img_full_path)
    beam = _get_beam(
        img_full_path,
        xds.sizes["frequency"],
        xds.sizes["polarization"],
        True,
        image_type,
    )
    if beam is not None:
        xds["BEAM_FIT_PARAMS_" + image_type.upper()] = beam
        xds["BEAM_FIT_PARAMS_" + image_type.upper()].attrs["type"] = (
            "beam_fit_params_" + image_type.lower()
        )
        xds[image_type.upper()].attrs[_beam_fit_params] = (
            "BEAM_FIT_PARAMS_" + image_type.upper()
        )

    # xds = _add_coord_attrs(xds, ret["icoords"], ret["dir_axes"])
    xds = _dask_arrayize_dv(xds)

    # images converted from FITS (importfits) can hold another order
    return _to_canonical_polarization_order(xds)


def _casa_image_xds(xds: xr.Dataset, output: ImageOutput) -> xr.Dataset:
    """Return the single image dataset the CASA writer writes for one planned
    output.

    The image becomes SKY (sky plane) or APERTURE (aperture plane); the flags
    of a sky image become its MASK_0 (default) mask and its beam fit
    parameters BEAM_FIT_PARAMS. An image whose ``type`` attribute is a mask
    or flag type (a deconvolution mask) gets the type of its plane instead,
    so that it is not registered as a mask of itself, and the image's
    ``flag`` attribute is derived from the data group only.
    """
    name = "SKY" if output.plane == "sky" else "APERTURE"
    image = xds[output.variable].copy(deep=False)
    attrs = dict(image.attrs)
    if attrs.get("type") in ("mask", "flag"):
        attrs["type"] = output.plane
    attrs.pop("flag", None)
    image.attrs = attrs
    image_xds = xr.Dataset(attrs=xds.attrs.copy())
    image_xds[name] = image
    if output.flag is not None:
        image_xds["MASK_0"] = xds[output.flag]
        image_xds[name].attrs["flag"] = "MASK_0"
    if output.beam_fit_params is not None:
        image_xds["BEAM_FIT_PARAMS"] = xds[output.beam_fit_params]
    return image_xds


def _xds_to_multiple_casa_images(
    xds: xr.Dataset, image_store_name: str, plan: ImageWritePlan | None = None
) -> None:
    """Function disentagles xradio xr.Dataset into multiple casa images based on data_groups attribute.
    An xr.Dataset may contain multiple images (sky, residual, psf, etc) stored under different data variables sharing common coordinates.
    An addtional complication is that CASA images allow for internal masks and beam fit parameters to be stored alongside the main image data so these also need to be handled.
    This function creates one casa image per data variable that is an image of some data group
    (see :func:`xradio.image._util._write_plan.plan_image_outputs`): sky plane images (l and m
    dimensions) and aperture plane images (u and v dimensions, such as the aperture, visibility
    and uv sampling images). Images with neither (the normalization images) are skipped with a
    warning. Every image is validated, and its metadata computed, before any image is written.

    Parameters
    ----------
    xds : xr.Dataset
        The xradio xr.Dataset containing multiple images and associated data.
    image_store_name : str
        The base name or path for storing the output CASA images.
        If only one image is written, it will be named image_store_name, else the images
        will be named image_store_name.<g1>...<gn>.<role>, where g1 to gn are the data groups
        whose <role> (sky, point_spread_function, primary_beam, etc.) refers to the image.
        Used only when ``plan`` is None.
    plan : ImageWritePlan, optional
        The planned outputs, as computed by ``write_image`` (whose output paths are then used
        instead of names derived from image_store_name).
    """
    if plan is None:
        plan = plan_image_outputs(xds, image_store_name, "casa")
    for message in plan.warnings:
        xradio_logger().warning(message)
    images = [(output, _casa_image_xds(xds, output)) for output in plan.outputs]
    # validate every image and compute its metadata before writing any pixels
    keywords = []
    for output, image_xds in images:
        with _naming_the_variable(output):
            keywords.append(_casa_image_keywords(image_xds))
    for (output, image_xds), image_keywords in zip(images, keywords, strict=True):
        with _naming_the_variable(output):
            _write_casa_image(image_xds, output.path, image_keywords)
        if not os.path.exists(output.path):
            raise OSError(f"Failed to write CASA image {output.path}")


def _naming_the_variable(output: ImageOutput):
    """Errors raised while writing a planned output name its data variable
    (the single image dataset calls the image SKY or APERTURE)."""
    return naming_the_variable(
        output.variable, "CASA", ("SKY", "APERTURE", "the image")
    )


# Range of the integers casacore keyword records hold (Int64)
_INT64 = np.iinfo(np.int64)


def _casa_keyword_value_ok(value) -> bool:
    """Whether casacore can store a value in a table keyword record: strings,
    booleans, real or complex numbers (integers within Int64), and arrays and
    lists of them, but no times or durations (numpy datetime64 or
    timedelta64), which python-casacore rejects and casatools drops."""
    if isinstance(value, np.datetime64 | np.timedelta64):
        return False
    if isinstance(value, numbers.Integral) and not isinstance(value, bool | np.bool_):
        return _INT64.min <= int(value) <= _INT64.max
    if isinstance(value, str | bool | numbers.Number | np.generic):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "u" and value.size:
            return int(value.max()) <= _INT64.max
        return value.dtype != object and value.dtype.kind not in "mMV"
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _casa_keyword_value_ok(v) for k, v in value.items()
        )
    if isinstance(value, list | tuple):
        return all(isinstance(v, str) for v in value) or all(
            isinstance(v, bool | numbers.Number | np.generic)
            and _casa_keyword_value_ok(v)
            for v in value
        )
    return False


def _miscinfo_from_xds(xds: xr.Dataset) -> dict:
    """Return the casacore miscinfo record of an image: the image variable's
    ``user`` attribute (where the readers store it), merged over any
    dataset level ``user`` attribute. Values casacore cannot store (None,
    mixed lists, arbitrary objects) are dropped with a warning."""
    ap_sky = _image_variable(xds)
    miscinfo = {}
    for user in (xds.attrs.get("user"), xds[ap_sky].attrs.get("user")):
        if isinstance(user, dict):
            miscinfo.update(user)
    dropped = [
        key
        for key, value in miscinfo.items()
        if not isinstance(key, str) or not _casa_keyword_value_ok(value)
    ]
    if dropped:
        xradio_logger().warning(
            f"Not writing user keywords {dropped} to the CASA image miscinfo: "
            "casacore cannot store their values"
        )
    return {key: value for key, value in miscinfo.items() if key not in dropped}


def _casa_image_keywords(xds: xr.Dataset) -> dict:
    """Compute the table keywords of a CASA image (coordinate system, image
    info, brightness units and miscinfo) from a single image dataset. Errors
    in the metadata are raised here, before any pixel is written."""
    sky_ap = _image_variable(xds)
    n_time = xds[sky_ap].sizes.get("time")
    if n_time != 1:
        raise RuntimeError(
            "XDS can only be converted if it has exactly one time plane "
            f"(found {n_time or 'no time axis'})"
        )
    keywords = {
        "coords": _coord_dict_from_xds(xds),
        "imageinfo": _imageinfo_dict_from_xds(xds),
    }
    units = xds[sky_ap].attrs.get("units")
    if units:
        keywords["units"] = units
    miscinfo = _miscinfo_from_xds(xds)
    if miscinfo:
        keywords["miscinfo"] = miscinfo
    return keywords


def _write_casa_image(xds: xr.Dataset, image_full_path: str, keywords: dict) -> None:
    """Write the pixels and masks of a single image dataset, then its table
    keywords (from :func:`_casa_image_keywords`) and history."""
    _write_casa_data(xds, image_full_path)
    tb = tables.table(
        image_full_path,
        readonly=False,
        lockoptions={"option": "permanentwait"},
        ack=False,
    )
    try:
        for name in ("coords", "imageinfo", "units", "miscinfo"):
            if name in keywords:
                tb.putkeyword(name, keywords[name])
    finally:
        tb.done()
    # history
    _history_from_xds(xds, image_full_path)


def _xds_to_casa_image(xds: xr.Dataset, image_store_name: str) -> None:
    image_full_path = os.path.expanduser(image_store_name)
    # metadata first, so that an error in it leaves no image behind
    keywords = _casa_image_keywords(xds)
    _write_casa_image(xds, image_full_path, keywords)
