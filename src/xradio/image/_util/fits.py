import os
import re

import xarray as xr
from astropy.io import fits

from xradio._utils.logging import xradio_logger
from xradio._utils.xarray_helpers import get_data_group_name
from xradio.image._util._fits.xds_to_fits import (
    _celestial_image_header,
    _fits_image_header,
    _image_label,
    _xds_to_fits_image,
)
from xradio.image._util._write_plan import (
    ImageOutput,
    ImageWritePlan,
    _image_roles,
    naming_the_variable,
    plan_image_outputs,
)


def _fits_image_xds(xds: xr.Dataset, output: ImageOutput) -> xr.Dataset:
    """Return the single image dataset the FITS writer writes for one
    planned output: the image as SKY, with its flags as FLAG and its beam fit
    parameters as BEAM_FIT_PARAMS."""
    image_xds = xr.Dataset(attrs=xds.attrs.copy())
    image_xds["SKY"] = xds[output.variable]
    if output.flag is not None:
        image_xds["FLAG"] = xds[output.flag]
    if output.beam_fit_params is not None:
        image_xds["BEAM_FIT_PARAMS"] = xds[output.beam_fit_params]
    return image_xds


def _naming_the_variable(image_xds: xr.Dataset, output: ImageOutput):
    """Errors raised while writing a planned output name its data variable
    (the single image dataset calls the image SKY, and the FITS writer's
    messages name it by its type)."""
    return naming_the_variable(
        output.variable, "FITS", (_image_label(image_xds["SKY"]), "SKY")
    )


def _prepared_fits_image(image_xds: xr.Dataset, output: ImageOutput) -> tuple:
    """Build and validate the FITS header of one planned output, without
    reading its pixels (see :func:`_fits_image_header`). Errors name the
    data variable."""
    with _naming_the_variable(image_xds, output):
        return _fits_image_header(image_xds)


def _xds_to_multiple_fits_images(
    xds: xr.Dataset, image_store_name: str, plan: ImageWritePlan | None = None
) -> None:
    """Disentangle an xradio image dataset into multiple FITS images based on
    the data_groups attribute, mirroring the CASA writer. One FITS file is
    written per data variable that is an image of some data group (see
    :func:`xradio.image._util._write_plan.plan_image_outputs`); flags are
    applied as NaN pixels (the FITS convention) and beam fit parameters are
    written as BMAJ/BMIN/BPA header cards (single beam) or a CASA style BEAMS
    binary table (per plane beams). Images without l and m dimensions are
    skipped with a warning. The header of every image is built and validated
    before any file is written.

    Parameters
    ----------
    xds : xr.Dataset
        The xradio image dataset containing one or more images.
    image_store_name : str
        The base name or path for storing the output FITS images. If only one
        image is written, it is named image_store_name, else the images are
        named <stem>.<g1>...<gn>.<role>[.fits], where g1 to gn are the data
        groups whose <role> refers to the image and a .fits extension of
        image_store_name stays last. Used only when ``plan`` is None.
    plan : ImageWritePlan, optional
        The planned outputs, as computed by ``write_image`` (whose output
        paths are then used instead of names derived from image_store_name).
    """
    if plan is None:
        plan = plan_image_outputs(xds, image_store_name, "fits")
    for message in plan.warnings:
        xradio_logger().warning(message)
    images = [(output, _fits_image_xds(xds, output)) for output in plan.outputs]
    # build and validate every header before writing any image; the headers
    # are then reused, so each is built (and its warnings logged) once
    prepared = [_prepared_fits_image(image_xds, output) for output, image_xds in images]
    for (output, image_xds), header in zip(images, prepared, strict=True):
        with _naming_the_variable(image_xds, output):
            _xds_to_fits_image(image_xds, output.path, prepared=header)
        if not os.path.exists(output.path):
            raise OSError(f"Failed to write FITS image {output.path}")


def _planned_fits_header(
    xds: xr.Dataset,
    data_group_name: str | None = None,
    role: str = "sky",
    time: int = 0,
    celestial: bool = False,
) -> fits.Header:
    """Return the FITS header ``write_image`` writes for one image.

    The header is built as :func:`_xds_to_multiple_fits_images` builds it
    for ``xds.isel(time=[time])``: the image of ``role`` in the data group
    ``data_group_name`` is the planned output of its data variable (see
    :func:`xradio.image._util._write_plan.plan_image_outputs`), with the
    flags and beam fit parameters that the plan writes with it. Like
    ``write_image``, which writes no file unless it can build the header of
    every planned image, it raises if the header of any planned image
    cannot be built; the celestial header needs only the plan and the
    direction axes of the image. No pixels are read.

    Parameters
    ----------
    xds : xr.Dataset
        The image dataset, with a ``data_groups`` attribute.
    data_group_name : str, optional
        Name of the data group of the image; by default ``"base"``, or the
        first data group if there is no ``"base"``.
    role : str, optional
        Data group role of the image, by default ``"sky"``.
    time : int, optional
        Index of the time plane (negative values count from the end), by
        default 0.
    celestial : bool, optional
        If True, return only the header of the direction axes and the
        observation date (see :func:`_celestial_image_header`), which needs
        neither the polarization nor the frequency axis to be representable
        in FITS. By default False: the complete primary header.

    Returns
    -------
    astropy.io.fits.Header
        A new header: the primary header (without the BEAMS extension of
        per plane beams), or with ``celestial`` the celestial header.

    Raises
    ------
    ValueError
        If the dataset has no such data group or image, or the image is not
        written to FITS, or (like the FITS writer) FITS cannot hold its
        direction axes or, unless ``celestial``, the image or another
        planned image.
    RuntimeError
        Like the FITS writer, for example for an unsupported direction frame.
    IndexError
        If ``time`` is out of range.
    """
    groups = xds.attrs.get("data_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            "The image dataset has no data_groups attribute naming its images"
        )
    group = get_data_group_name(xds, data_group_name)
    if group not in groups:
        raise ValueError(
            f"The image dataset has no data group {group!r}; its data groups "
            f"are {list(groups)}"
        )
    image_roles = _image_roles()
    if role not in image_roles:
        raise ValueError(
            f"{role!r} is not an image role of a data group; the image roles "
            f"are {image_roles}"
        )
    variable = groups[group].get(role)
    if variable is None:
        raise ValueError(f"The data group {group!r} has no {role!r} image")
    # only the variables matter, not the output path
    plan = plan_image_outputs(xds, "image.fits", "fits")
    output = next((o for o in plan.outputs if o.variable == variable), None)
    if output is None:
        # the plan's reasons for not writing it (the variable is missing, or
        # has no l and m dimensions)
        named = re.compile(rf"\b{re.escape(str(variable))}\b")
        reasons = [message for message in plan.warnings if named.search(message)]
        detail = f": {'; '.join(reasons)}" if reasons else ""
        raise ValueError(
            f"{variable} (the {role!r} image of data group {group!r}) is not "
            f"written to FITS{detail}"
        )
    plane = xds.isel(time=[time]) if "time" in xds.dims else xds
    image_xds = _fits_image_xds(plane, output)
    if celestial:
        with _naming_the_variable(image_xds, output):
            header = _celestial_image_header(image_xds)
    else:
        header, _, _ = _prepared_fits_image(image_xds, output)
        # write_image writes no file unless every header can be built; this
        # image's own errors come first
        for other in plan.outputs:
            if other is not output:
                _prepared_fits_image(_fits_image_xds(plane, other), other)
    # a plain astropy header: the writer's subclass only rewords its errors
    return fits.Header(header)
