import os

import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio.image._util._fits.xds_to_fits import (
    _fits_image_header,
    _image_label,
    _xds_to_fits_image,
)
from xradio.image._util._write_plan import (
    ImageOutput,
    ImageWritePlan,
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
