#################################
#
# Public interface
#
#################################
import warnings
from typing import List, Union
import copy
import numpy as np
import os
import shutil
import toolviper.utils.logger as logger
import xarray as xr

from xradio.image._util.image_factory import (
    _make_empty_aperture_image,
    _make_empty_lmuv_image,
    _make_empty_sky_image,
)
from xradio.image._util.zarr import (
    _load_image_from_zarr_no_dask,
    _xds_from_zarr,
    _xds_to_zarr,
)
from xradio.image._util._fits.xds_from_fits import _fits_image_to_xds

warnings.filterwarnings("ignore", category=FutureWarning)


def detect_image_type(store):
    import os

    if isinstance(store, str):
        if os.path.isfile(store):
            store_type = "fits"
        elif os.path.isdir(store):
            if "table.info" in os.listdir(store):
                store_type = "casa"
            elif ".zattrs" in os.listdir(store):
                store_type = "zarr"
            else:
                logger.error("Unknown directory structure.")
        else:
            logger.error("Path does not exist.")
    else:
        store_type = "zarr"

    if store_type == "fits" or store_type == "casa":
        if "image" in store.lower():
            image_type = "SKY"
        elif "im" in store.lower():
            image_type = "SKY"
        elif "psf" in store.lower():
            image_type = "POINT_SPREAD_FUNCTION"
        elif "model" in store.lower():
            image_type = "MODEL"
        elif "residual" in store.lower():
            image_type = "RESIDUAL"
        elif "mask" in store.lower():
            image_type = "MASK"
        elif "pb" in store.lower():
            image_type = "PRIMARY_BEAM"
        elif "aperture" in store.lower():
            image_type = "APERTURE"
        elif "visibility" in store.lower():
            image_type = "VISIBILITY"
        elif "sumwt" in store.lower():
            image_type = "VISIBILITY_NORMALIZATION"
        else:
            image_type = "IMAGE"
            logger.warning("Could not determine image type from filename.")
    elif store_type == "zarr":
        image_type = "ALL"

    return store_type, image_type


def open_image(
    store: Union[str, dict],
    chunks: dict = {},
    verbose: bool = False,
    do_sky_coords: bool = True,
    selection: dict = {},
    compute_mask: bool = True,
) -> xr.Dataset:
    """
    Convert CASA, FITS, or zarr image to xradio image xds format
    ngCASA image spec is located at
    https://docs.google.com/spreadsheets/d/1WW0Gl6z85cJVPgtdgW4dxucurHFa06OKGjgoK8OREFA/edit#gid=1719181934

    Notes on FITS compatibility and memory mapping:

    This function relies on Astropy's `memmap=True` to avoid loading full image data into memory.
    However, not all FITS files support memory-mapped reads.

    ⚠️ The following FITS types are incompatible with memory mapping:

    1. Compressed images (`CompImageHDU`)
        = Workaround: decompress the FITS using tools like `funpack`, `cfitsio`,
          or Astropy's `.scale()`/`.copy()` workflows
    2. Some scaled images (using BSCALE/BZERO headers)
        ✅ Supported:
            - Files with no BSCALE/BZERO headers (or BSCALE=1.0 and BZERO=0.0)
            - Uncompressed, unscaled primary HDUs
        ⚠️ Unsupported: Files with BSCALE ≠ 1.0 or BZERO ≠ 0.0
            - These require data rescaling in memory, which disables lazy access
            - Attempting to slice such arrays forces eager read of the full dataset
            - Workaround: remove scaling with Astropy's
                `HDU.data = HDU.data * BSCALE + BZERO` and save a new file

    These cases will raise `RuntimeError` to prevent silent eager loads that can exhaust memory.

    If you encounter such an error, consider preprocessing the file to make it memory-mappable.

    Parameters
    ----------
    store : str
        Path to the input CASA image
    :chunks : dict
        The desired dask chunk size. Only applicable for casacore and fits images.
        Supported optional keys are 'l', 'm', 'frequency', 'polarization', and 'time'.
        The supported values are positive integers, indicating the length of a chunk
        on that particular axis. If a key is missing, then the associated chunk length
        along that axis is equal to the number of pixels along that axis. For zarr
        images, this parameter is ignored and the chunk size used to store the arrays
        in the zarr image is used. 'l' represents the longitude like dimension, and 'm'
        represents the latitude like dimension. For apeature images, 'u' may be used in
        place of 'l', and 'v' in place of 'm'.
    verbose : bool
        emit debugging messages? Default is False.
    do_sky_coords : bool
        Compute SkyCoord at each pixel and add spherical (sky) dimensions as non-dimensional
        coordinates in the returned xr.Dataset. Only applies to CASA and FITS images; zarr
        images will have these coordinates added if they were saved with the zarr dataset,
        and if zarr image didn't have these coordinates when it was written, the resulting
        xr.Dataset will not.
    selection : dict
        The selection of data to return, supported keys are time,
        polarization, frequency, l (or u if aperture image), m (or v if aperture
        image) a missing key indicates to return the entire axis length for that
        dimension. Values can be non-negative integers or slices. Slicing
        behaves as numpy slicing does, that is the start pixel is included in
        the selection, and the end pixel is not. An empty dictionary (the
        default) indicates that the entire image should be returned. Currently
        only supported for images stored in zarr format.
     compute_mask : bool, optional
        If True (default), compute and attach valid data masks when converting from FITS to xds.
        If False, skip mask computation entirely. This may improve performance if the mask
        is not required for subsequent processing. It may, however, result in unpredictable behavior
        for applications that are not designed to handle missing data. It is the user's responsibility,
        not the software's, to ensure that the mask is computed if it is necessary. Currently only
        implemented for FITS images.
    Returns
    -------
    xarray.Dataset
    """

    if isinstance(store, str):
        store_list = [store]  # So can iterate over it.
    else:
        store_list = store

    img_xds = xr.Dataset()
    generic_image_counter = 0
    for store in store_list:
        store_type, image_type = detect_image_type(store)
        if store_type == "casa":
            from ._util.casacore import _open_casa_image

            xds = _open_casa_image(store, chunks, verbose, do_sky_coords)
            xds.attrs["type"] = image_type.lower()
        elif store_type == "fits":
            xds = _fits_image_to_xds(
                store, chunks, verbose, do_sky_coords, compute_mask
            )
            xds.attrs["type"] = image_type.lower()
        elif store_type == "zarr":
            xds = _xds_from_zarr(store, {"dv": "dask", "coords": "numpy"}, selection)
        else:
            raise RuntimeError(
                f"Unrecognized image format for path {store}. Supported types are CASA, FITS, and zarr.\n"
            )

        if image_type == "ALL":
            img_xds = img_xds.merge(
                xds, compat="override", 
                combine_attrs="no_conflicts",
                #combine_attrs="override"
            )
        elif image_type == "IMAGE":
            img_xds.attrs = img_xds.attrs | xds.attrs
            img_xds["IMAGE_" + str(generic_image_counter)] = xds["SKY"]
            generic_image_counter += 1

            # Need to figure out how to handle fits
            if "BEAM_FIT_PARAMS" in xds:
                img_xds["BEAM_FIT_PARAMS"] = xds["BEAM_FIT_PARAMS"]

            if "MASK_0" in xds:
                img_xds["MASK_0"] = xds["MASK_0"]

            if "beam_param" in xds:
                img_xds = img_xds.assign_coords(beam_param=xds["beam_param"])

        else:
            img_xds.attrs = img_xds.attrs | xds.attrs
            if "SKY" in xds:
                img_xds[image_type] = xds["SKY"]

            if "APERTURE" in xds:
                img_xds["APERTURE"] = xds["APERTURE"]

            if "BEAM_FIT_PARAMS" in xds:
                img_xds["BEAM_FIT_PARAMS"] = xds["BEAM_FIT_PARAMS"]

            if "MASK_0" in xds:
                img_xds["MASK_0"] = xds["MASK_0"]

            if "beam_param" in xds:
                img_xds = img_xds.assign_coords(beam_param=xds["beam_param"])
    img_xds.attrs["type"] = "image"
    return img_xds


def load_image(store: str, block_des: dict = None, do_sky_coords=True) -> xr.Dataset:
    """
    Load an image or portion of an image (subimage) into memory with data variables
    being converted from dask to numpy arrays and coordinate arrays being converted
    from dask arrays to numpy arrays. If already a numpy array, that data variable
    or coordinate is left unaltered.

    Parameters
    ----------
    store : str
        Path to the input image, currently CASA and zarr images are supported
    block_des : dict
        The description of data to return, supported keys are time,
        polarization, frequency, l (or u if aperture image), m (or v if aperture
        image) a missing key indicates to return the entire axis length for that
        dimension. Values can be non-negative integers or slices. Slicing
        behaves as numpy slicing does, that is the start pixel is included in
        the selection, and the end pixel is not. An empty dictionary (the
        default) indicates that the entire image should be returned. The returned
        dataset will have data variables stored as numpy, not dask, arrays.
        TODO I'd really like to rename this parameter "selection"
    do_sky_coords : bool
        Compute SkyCoord at each pixel and add spherical (sky) dimensions as non-dimensional
        coordinates in the returned xr.Dataset. Only applies to CASA and FITS images; zarr
        images will have these coordinates added if they were saved with the zarr dataset,
        and if zarr image didn't have these coordinates when it was written, the resulting
        xr.Dataset will not.
    Returns
    -------
    xarray.Dataset
    """

    if isinstance(store, str):
        store_list = [store]  # So can iterate over it.
    else:
        store_list = store

    if block_des is None:
        block_des = {}

    selection = copy.deepcopy(block_des) if block_des else block_des
    if selection:
        for k, v in selection.items():
            if type(v) == int:
                selection[k] = slice(v, v + 1)

    img_xds = xr.Dataset()
    generic_image_counter = 0
    for store in store_list:
        store_type, image_type = detect_image_type(store)

        if store_type == "casa":
            from ._util.casacore import _load_casa_image_block

            xds = _load_casa_image_block(store, selection, do_sky_coords)
            xds.attrs["type"] = image_type.lower()
        elif store_type == "zarr":
            xds = _load_image_from_zarr_no_dask(store, block_des)
            xds.attrs["type"] = image_type.lower()
        else:
            raise RuntimeError(
                f"Unrecognized image format for path {store}. Supported types are CASA, and zarr.\n"
            )

        if image_type == "ALL":
            img_xds = img_xds.merge(
                xds, compat="override", combine_attrs="no_conflicts"
            )
        elif image_type == "IMAGE":
            img_xds.attrs = img_xds.attrs | xds.attrs
            img_xds["IMAGE_" + str(generic_image_counter)] = xds["SKY"]
            generic_image_counter += 1

            # Need to figure out how to handle fits
            if "BEAM_FIT_PARAMS" in xds:
                img_xds["BEAM_FIT_PARAMS"] = xds["BEAM_FIT_PARAMS"]

            if "MASK_0" in xds:
                img_xds["MASK_0"] = xds["MASK_0"]

            if "beam_param" in xds:
                img_xds = img_xds.assign_coords(beam_param=xds["beam_param"])

        else:
            img_xds.attrs = img_xds.attrs | xds.attrs
            if "SKY" in xds:
                img_xds[image_type] = xds["SKY"]

            if "APERTURE" in xds:
                img_xds["APERTURE"] = xds["APERTURE"]

            if "BEAM_FIT_PARAMS" in xds:
                img_xds["BEAM_FIT_PARAMS"] = xds["BEAM_FIT_PARAMS"]

            if "MASK_0" in xds:
                img_xds["MASK_0"] = xds["MASK_0"]

            if "beam_param" in xds:
                img_xds = img_xds.assign_coords(beam_param=xds["beam_param"])

    img_xds.attrs["type"] = "image"
    return img_xds


def write_image(
    xds: xr.Dataset, imagename: str, out_format: str = "casa", overwrite: bool = False
) -> None:
    """
    Convert an xds image to CASA or zarr image.
    xds : xarray.Dataset
        XDS to convert
    imagename : str
        Path to output image
    out_format : str
        Format of output image, currently "casa" and "zarr" are supported
    overwrite : bool
        If True, overwrite existing image. Default is False.
    Returns
    -------
    None
    """
    if os.path.exists(imagename):
        if overwrite:
            logger.warning(
                f"Because overwrite=True, removing existing path {imagename}"
            )
            if os.path.isdir(imagename):
                shutil.rmtree(imagename)
            else:
                os.remove(imagename)
        else:
            raise FileExistsError(
                f"Path {imagename} already exists. Set overwrite=True to remove it."
            )
    my_format = out_format.lower()
    if my_format == "casa":
        from ._util.casacore import _xds_to_casa_image

        _xds_to_casa_image(xds, imagename)
    elif my_format == "zarr":
        _xds_to_zarr(xds, imagename)
    else:
        raise ValueError(
            f"Writing to format {out_format} is not supported. "
            'out_format must be either "casa" or "zarr".'
        )


def make_empty_sky_image(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
    chan_coords: Union[list, np.ndarray],
    pol_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
    direction_reference: str = "fK5",
    projection: str = "SIN",
    spectral_reference: str = "lsrk",
    do_sky_coords: bool = True,
) -> xr.Dataset:
    """
    Create an image xarray.Dataset with only coordinates (no datavariables).
    The image dimensionality is time, frequency, polarization, l, m

    Parameters
    ----------
    phase_center : array of float, length = 2, units = rad
        Image phase center.
    image_size : array of int, length = 2
        Number of x and y axis pixels in image.
    cell_size : array of float, length = 2, units = rad
        Cell size of x and y axis pixels in image.
    chan_coords : list or np.ndarray
        The center frequency in Hz of each image channel.
    pol_coords : list or np.ndarray
        The polarization code for each image polarization.
    time_coords : list or np.ndarray
        The time for each temporal plane in MJD.
    direction_reference : str, default = 'fk5'
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
    do_sky_coords : bool
        If True, compute SkyCoord at each pixel and add spherical (sky) dimensions as
        non-dimensional coordinates in the returned xr.Dataset.
    Returns
    -------
    xarray.Dataset
    """
    return _make_empty_sky_image(
        phase_center,
        image_size,
        cell_size,
        chan_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
        do_sky_coords,
    )


def make_empty_aperture_image(
    phase_center: Union[List[float], np.ndarray],
    image_size: Union[List[int], np.ndarray],
    sky_image_cell_size: Union[List[float], np.ndarray],
    chan_coords: Union[List[float], np.ndarray],
    pol_coords: Union[List[str], np.ndarray],
    time_coords: Union[List[float], np.ndarray],
    direction_reference: str = "fk5",
    projection: str = "SIN",
    spectral_reference: str = "lsrk",
) -> xr.Dataset:
    """
    Create an aperture (uv) mage xarray.Dataset with only coordinates (no datavariables).
    The image dimensionality is time, frequency, polarization, u, v

    Parameters
    ----------
    phase_center : array of float, length = 2, units = rad
        Image phase center.
    image_size : array of int, length = 2
        Number of x and y axis pixels in image.
    sky_image_cell_size : array of float, length = 2, units = rad
        Cell size of x and y axis pixels in sky image, used to get cell size in uv image
    chan_coords : list or np.ndarray
        The center frequency in Hz of each image channel.
    pol_coords : list or np.ndarray
        The polarization code for each image polarization.
    time_coords : list or np.ndarray
        The time for each temporal plane in MJD.
    direction_reference : str, default = 'fk5'
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
    Returns
    -------
    xarray.Dataset
    """
    return _make_empty_aperture_image(
        phase_center,
        image_size,
        sky_image_cell_size,
        chan_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
    )


def make_empty_lmuv_image(
    phase_center: Union[List[float], np.ndarray],
    image_size: Union[List[int], np.ndarray],
    sky_image_cell_size: Union[List[float], np.ndarray],
    chan_coords: Union[List[float], np.ndarray],
    pol_coords: Union[List[float], np.ndarray],
    time_coords: Union[List[float], np.ndarray],
    direction_reference: str = "fk5",
    projection: str = "SIN",
    spectral_reference: str = "lsrk",
    do_sky_coords: bool = True,
) -> xr.Dataset:
    """
    Create an image xarray.Dataset with only coordinates (no datavariables).
    The image dimensionality is time, frequency, polarization, l, m, u, v

    Parameters
    ----------
    phase_center : array of float, length = 2, units = rad
        Image phase center.
    image_size : array of int, length = 2
        Number of x and y axis pixels in image.
    sky_image_cell_size : array of float, length = 2, units = rad
        Cell size of sky image. The cell size of the u,v image will be computed from
        1/(image_size * sky_image_cell_size)
    chan_coords : list or np.ndarray
        The center frequency in Hz of each image channel.
    pol_coords : list or np.ndarray
        The polarization code for each image polarization.
    time_coords : list or np.ndarray
        The time for each temporal plane in MJD.
    direction_reference : str, default = 'fk5'
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
    do_sky_coords : bool
        If True, compute SkyCoord at each pixel and add spherical (sky) dimensions as
        non-dimensional coordinates in the returned xr.Dataset.
    Returns
    -------
    xarray.Dataset
    """
    return _make_empty_lmuv_image(
        phase_center,
        image_size,
        sky_image_cell_size,
        chan_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
        do_sky_coords,
    )
