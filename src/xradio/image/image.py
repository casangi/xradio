#################################
#
# Public interface
#
#################################
import os

import numpy as np
import xarray as xr

from xradio.image._util._fits.xds_from_fits import _fits_image_to_xds
from xradio.image._util._write_plan import (
    plan_image_outputs,
    write_outputs_atomically,
)
from xradio.image._util.image_factory import (
    _make_empty_aperture_image,
    _make_empty_lmuv_image,
    _make_empty_sky_image,
    create_image_xds_from_store,
)
from xradio.image._util.zarr import (
    _xds_from_zarr,
    _xds_to_zarr,
)

# warnings.filterwarnings("ignore", category=FutureWarning)


def _casa_image_reader_unavailable(exc: ImportError):
    """Stand-in for the CASA image readers when neither python-casacore nor
    casatools can be imported: it fails when a CASA image is read, so FITS and
    zarr images open without those packages."""

    def read_casa_image(store, **kwargs):
        raise ModuleNotFoundError(
            f"Reading the CASA image {store} needs python-casacore or casatools: {exc}"
        ) from exc

    return read_casa_image


def _int_selections_to_slices(selection: dict | None) -> dict:
    """Return a copy of an image selection with integer values replaced by
    length-1 slices, so that a selected dimension is kept (as ``load_image``
    always did). A negative integer counts from the end, as in numpy."""

    def one_pixel(index: int) -> slice:
        return slice(index, index + 1 or None)

    return {
        key: (
            one_pixel(int(value))
            if isinstance(value, int | np.integer)
            and not isinstance(value, bool | np.bool_)
            else value
        )
        for key, value in (selection or {}).items()
    }


def open_image(
    store: str | dict | list,
    chunks: dict | None = None,
    verbose: bool = False,
    do_sky_coords: bool = True,
    selection: dict | None = None,
    compute_mask: bool = True,
) -> xr.Dataset:
    """
    Open CASA, FITS or zarr images lazily as an xradio image dataset.
    The ngCASA image spec is located at
    https://docs.google.com/spreadsheets/d/1WW0Gl6z85cJVPgtdgW4dxucurHFa06OKGjgoK8OREFA/edit#gid=1719181934

    Supported formats:

    * CASA images (casacore tables), read with python-casacore or casatools;
    * FITS images, read with astropy only (python-casacore and casatools are
      not needed);
    * zarr stores written by :func:`write_image` (``out_format="zarr"``),
      which hold a whole image dataset. Stores written by xradio 1.2.3 and
      earlier are upgraded to the current schema conventions on read.

    Several CASA or FITS images are combined into one dataset by passing a
    dict that maps image types (data group roles such as ``"sky"``,
    ``"point_spread_function"``, ``"primary_beam"`` or ``"mask"``, and
    ``"sky_<group>"`` for the sky image of another data group) to paths, or a
    list of paths whose image types are detected from their names: tclean
    product names such as ``target.psf`` or ``target.residual.fits``, and the
    outputs of a :func:`write_image` call for a dataset with a single data
    group (``out.base.sky``, ``out.base.point_spread_function``, ...). The
    outputs of a :func:`write_image` call for several data groups all end in
    their role, so several of them are sky images
    (``out.deconvolved.sky``, ``out.model.sky``, ...): open them with a dict
    keyed by image type, for example ``{"sky": "out.deconvolved.sky",
    "sky_model": "out.model.sky", "point_spread_function":
    "out.deconvolved.model.point_spread_function"}``, or one data group at a
    time. Paths may start with ``~``, which is expanded to the home
    directory.

    Notes on FITS compatibility and memory mapping:

    This function relies on Astropy's ``memmap=True`` to avoid loading full
    image data into memory. However, not all FITS files support memory-mapped
    reads. The following FITS types are incompatible with memory mapping:

    1. Compressed images (``CompImageHDU``). Workaround: decompress the FITS
       file with tools like ``funpack``, ``cfitsio``, or Astropy's
       ``.scale()``/``.copy()`` workflows.
    2. Scaled images, with BSCALE != 1.0 or BZERO != 0.0 (files without
       BSCALE/BZERO cards, or with BSCALE=1.0 and BZERO=0.0, are supported).
       These require data rescaling in memory, which disables lazy access:
       slicing such arrays forces an eager read of the full dataset.
       Workaround: remove the scaling with Astropy
       (``HDU.data = HDU.data * BSCALE + BZERO``) and save a new file.

    These cases raise ``RuntimeError`` to prevent silent eager loads that can
    exhaust memory. If you encounter such an error, consider preprocessing the
    file to make it memory-mappable.

    The polarization axis is always returned in canonical (casacore
    ``Stokes``, Jones matrix) order (``I, Q, U, V``; ``RR, RL, LR, LL``;
    ``XX, XY, YX, YY``). FITS cannot store the correlation products in that
    order, so the FITS writer may reorder the planes in the file; the reader
    restores the canonical order, so a FITS file (or a CASA image converted
    from one) with the axis ``RR, LL, RL, LR`` opens as ``RR, RL, LR, LL``,
    every variable with a polarization dimension reordered the same way. CASA
    and FITS images without a spectral axis open with casacore's default
    single channel, and images without a Stokes axis with the polarization
    ``["I"]``.

    Parameters
    ----------
    store : str, dict or list
        Path to the input image, a dict mapping image types (data group
        roles) to paths, or a list of paths.
    chunks : dict
        The desired dask chunk size. Only applicable for casacore and fits images.
        Supported optional keys are 'l', 'm', 'frequency', 'polarization', and 'time'.
        The supported values are positive integers, indicating the length of a chunk
        on that particular axis. If a key is missing, then the associated chunk length
        along that axis is equal to the number of pixels along that axis. For zarr
        images, this parameter is ignored and the chunk size used to store the arrays
        in the zarr image is used. 'l' represents the longitude like dimension, and 'm'
        represents the latitude like dimension. For aperture images, 'u' may be used in
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
        the selection, and the end pixel is not. An integer selects a single
        pixel and keeps its dimension (with length 1), as in
        :func:`load_image`. An empty dictionary (the default) indicates that
        the entire image should be returned. Currently only supported for
        images stored in zarr format.
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
    # python-casacore and casatools are optional: FITS images are read with
    # astropy and zarr images with zarr, so only a CASA image needs them.
    try:
        from xradio.image._util.casacore import _open_casa_image
    except ImportError as exc:
        _open_casa_image = _casa_image_reader_unavailable(exc)

    if chunks is None:
        chunks = {}
    selection = _int_selections_to_slices(selection)

    img_xds = create_image_xds_from_store(
        store,
        _open_casa_image,
        {"chunks": chunks, "verbose": verbose, "do_sky_coords": do_sky_coords},
        _fits_image_to_xds,
        {
            "chunks": chunks,
            "verbose": verbose,
            "do_sky_coords": do_sky_coords,
            "compute_mask": compute_mask,
        },
        _xds_from_zarr,
        {"output": {"dv": "dask", "coords": "numpy"}, "selection": selection},
    )

    return img_xds


def load_image(store: str, block_des: dict = None, do_sky_coords=True) -> xr.Dataset:
    """
    Load an image or portion of an image (subimage) into memory with data variables
    being converted from dask to numpy arrays and coordinate arrays being converted
    from dask arrays to numpy arrays. If already a numpy array, that data variable
    or coordinate is left unaltered.

    CASA images and zarr stores are supported (FITS images are opened with
    :func:`open_image`). As in :func:`open_image`, zarr stores written by
    xradio 1.2.3 and earlier are upgraded to the current schema conventions,
    and the polarization axis is returned in canonical (Jones matrix) order.

    Parameters
    ----------
    store : str, dict or list
        Path to the input image (``~`` is expanded to the home directory), a
        dict mapping image types (data group roles such as ``"sky"`` or
        ``"point_spread_function"``) to the paths of CASA images, or a list of
        paths whose image types are detected from their names (see
        :func:`open_image`).
    block_des : dict
        The description of data to return, supported keys are time,
        polarization, frequency, l (or u if aperture image), m (or v if aperture
        image) a missing key indicates to return the entire axis length for that
        dimension. Values can be non-negative integers or slices. Slicing
        behaves as numpy slicing does, that is the start pixel is included in
        the selection, and the end pixel is not. An integer selects a single
        pixel and keeps its dimension (with length 1). An empty dictionary (the
        default) indicates that the entire image should be returned. The returned
        dataset will have data variables stored as numpy, not dask, arrays.
        Polarization indices refer to the canonical order of the returned
        dataset (for zarr stores, to the order of the store).
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
    selection = _int_selections_to_slices(block_des)

    try:
        from xradio.image._util.casacore import _load_casa_image_block
    except ImportError as exc:
        _load_casa_image_block = _casa_image_reader_unavailable(exc)

    img_xds = create_image_xds_from_store(
        store,
        _load_casa_image_block,
        {"block_des": selection, "do_sky_coords": do_sky_coords},
        None,
        {},
        _xds_from_zarr,
        {"output": {"dv": "dask", "coords": "numpy"}, "selection": selection},
    )
    return img_xds


def write_image(
    xds: xr.Dataset, imagename: str, out_format: str = "casa", overwrite: bool = False
) -> None:
    """
    Write an xradio image dataset as CASA, FITS or zarr images.

    TODO: I think the user should be permitted to specify data groups to write.

    Supported formats:

    * ``"casa"`` (the default) writes casacore image tables and needs
      python-casacore or casatools. Sky plane images (l and m dimensions) and
      aperture plane images (u and v dimensions) are written; the
      normalization images, which have neither, are skipped with a warning.
      A sky image's flags become its default mask and its beam fit
      parameters its restoring beam(s).
    * ``"fits"`` writes FITS files with astropy only. Only sky plane images
      are written; other images are skipped with a warning. Flagged pixels
      are written as NaN, and the polarization planes may be reordered in the
      file so that they form a FITS STOKES axis (:func:`open_image` restores
      the canonical order).
    * ``"zarr"`` writes the whole dataset, with all its data variables and
      data groups, to one zarr store.

    Output names: CASA and FITS images hold one image each, so every data
    variable that is an image of some data group (its ``sky``,
    ``point_spread_function``, ``primary_beam``, ``mask``, ... role) is
    written once, to its own output; flags and beam fit parameters are written
    with their image. When a single image is written, it is named
    ``imagename``. When several are written, each is named
    ``<imagename>.<g1>.<g2>...<gn>.<role>``, where ``g1`` to ``gn`` are, in
    data group order, all data groups whose ``<role>`` refers to that
    variable, for example ``out.base.sky`` and
    ``out.base.point_spread_function``, or ``out.dirty.residual.primary_beam``
    for a primary beam shared by the data groups ``dirty`` and ``residual``.
    For FITS, a ``.fits`` extension of ``imagename`` (in any case) stays
    last: ``img.fits`` gives ``img.base.sky.fits``.

    Overwriting: all output paths are determined before anything is written.
    With ``overwrite=False``, FileExistsError is raised if any of them exists,
    and nothing is written. The outputs are written into a temporary
    directory next to ``imagename`` and moved into place only when all of
    them have been written; existing outputs are then replaced (only with
    ``overwrite=True``). Other files, such as outputs of earlier writes with
    other names, are left alone. If writing fails, the temporary directory
    is removed and nothing at the output paths changes, so a dataset opened
    lazily from ``imagename`` can be written back to the same path.

    Parameters
    ----------
    xds : xarray.Dataset
        The image dataset to write.
    imagename : str
        Path of the output image, or the base name of the outputs when several
        images are written (see above). ``~`` is expanded to the home
        directory, and missing parent directories are created. Only local
        paths are supported (not URLs such as ``s3://...``).
    out_format : str
        Format of the output: ``"casa"`` (default), ``"fits"`` or ``"zarr"``
        (in any case).
    overwrite : bool
        If True, replace existing outputs. Default is False.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If an output exists and ``overwrite`` is False.
    ValueError
        If ``out_format`` is not supported, ``imagename`` is a URL, or the
        dataset holds no image the format can store.
    """
    imagename = os.fspath(imagename)
    if isinstance(imagename, bytes):
        imagename = os.fsdecode(imagename)
    if "://" in imagename:
        # the outputs are staged and moved into place on the local file system
        raise ValueError(
            f"Cannot write {imagename!r}: write_image writes to local paths "
            "only, not to URLs"
        )
    imagename = os.path.expanduser(imagename)
    if len(imagename) > 1:
        imagename = imagename.rstrip(os.sep)
    if os.path.basename(imagename) in ("", ".", ".."):
        raise ValueError(f"{imagename!r} is not a valid image name")
    my_format = out_format.lower()
    if my_format == "casa":
        try:
            from xradio.image._util.casacore import _xds_to_multiple_casa_images
        except ImportError as exc:
            raise ModuleNotFoundError(
                f"Writing CASA images needs python-casacore or casatools: {exc}"
            ) from exc
        write_images = _xds_to_multiple_casa_images
    elif my_format == "fits":
        from xradio.image._util.fits import _xds_to_multiple_fits_images

        write_images = _xds_to_multiple_fits_images
    elif my_format != "zarr":
        raise ValueError(
            f"Writing to format {out_format} is not supported. "
            'out_format must be either "casa", "fits" or "zarr".'
        )

    if my_format == "zarr":
        paths = [imagename]

        def write(directory: str) -> None:
            _xds_to_zarr(xds, os.path.join(directory, os.path.basename(imagename)))

    else:
        plan = plan_image_outputs(xds, imagename, my_format)
        paths = plan.paths

        def write(directory: str) -> None:
            write_images(
                xds,
                os.path.join(directory, os.path.basename(imagename)),
                plan=plan.in_directory(directory),
            )

    write_outputs_atomically(paths, overwrite, write)


def make_empty_sky_image(
    phase_center: list | np.ndarray,
    image_size: list | np.ndarray,
    cell_size: list | np.ndarray,
    frequency_coords: list | np.ndarray,
    pol_coords: list | np.ndarray,
    time_coords: list | np.ndarray,
    direction_reference: str = "fK5",
    projection: str = "SIN",
    spectral_reference: str = "lsrk",
    do_sky_coords: bool = False,
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
    frequency_coords : list or np.ndarray
        The center frequency in Hz of each image channel.
    pol_coords : list or np.ndarray
        The polarization label of each image polarization, in canonical
        (casacore ``Stokes``, Jones matrix) order, for example
        ``["I", "Q", "U", "V"]``, ``["RR", "RL", "LR", "LL"]`` or
        ``["XX", "YY"]``.
    time_coords : float, list, np.ndarray, str, datetime or astropy.time.Time
        The time of each temporal plane: numbers are MJD (UTC) days;
        ``datetime64`` values, ISO time strings, ``datetime.datetime`` objects
        and astropy ``Time`` objects are converted to MJD (UTC) days.
    direction_reference : str, default = 'fk5'
        Direction reference frame: ``'fk5'`` (equinox J2000), ``'fk4'``
        (equinox B1950), ``'icrs'`` or ``'galactic'``.
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
        Spectral reference frame: a casacore frame (``'LSRK'``, ``'LSRD'``,
        ``'BARY'``, ``'GEO'``, ``'TOPO'``, ``'REST'``, ``'GALACTO'``,
        ``'LGROUP'``, ``'CMB'``), a FITS ``SPECSYS`` value (``'BARYCENT'``,
        ...) or a schema observer (``'lsrk'``, ``'gcrs'``, ...), in any case.
        The astropy frames ``'icrs'``, ``'hcrs'`` and ``'lsr'``, which have no
        casacore equivalent, are accepted too, but images in them cannot be
        written to CASA or FITS.
    do_sky_coords : bool
        If True, compute SkyCoord at each pixel and add spherical (sky) dimensions as
        non-dimensional coordinates in the returned xr.Dataset.
    Returns
    -------
    xarray.Dataset

    Raises
    ------
    TypeError
        If ``time_coords`` are neither numbers nor times (for example
        ``timedelta64`` durations or booleans).
    ValueError
        If ``pol_coords`` are not in canonical order, or
        ``spectral_reference`` is not a spectral reference frame.
    """
    return _make_empty_sky_image(
        phase_center,
        image_size,
        cell_size,
        frequency_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
        do_sky_coords,
    )


def make_empty_aperture_image(
    phase_center: list[float] | np.ndarray,
    image_size: list[int] | np.ndarray,
    sky_image_cell_size: list[float] | np.ndarray,
    frequency_coords: list[float] | np.ndarray,
    pol_coords: list[str] | np.ndarray,
    time_coords: list[float] | np.ndarray,
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
    frequency_coords : list or np.ndarray
        The center frequency in Hz of each image channel.
    pol_coords : list or np.ndarray
        The polarization label of each image polarization, in canonical
        (casacore ``Stokes``, Jones matrix) order, for example
        ``["I", "Q", "U", "V"]``, ``["RR", "RL", "LR", "LL"]`` or
        ``["XX", "YY"]``.
    time_coords : float, list, np.ndarray, str, datetime or astropy.time.Time
        The time of each temporal plane: numbers are MJD (UTC) days;
        ``datetime64`` values, ISO time strings, ``datetime.datetime`` objects
        and astropy ``Time`` objects are converted to MJD (UTC) days.
    direction_reference : str, default = 'fk5'
        Direction reference frame: ``'fk5'`` (equinox J2000), ``'fk4'``
        (equinox B1950), ``'icrs'`` or ``'galactic'``.
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
        Spectral reference frame: a casacore frame (``'LSRK'``, ``'LSRD'``,
        ``'BARY'``, ``'GEO'``, ``'TOPO'``, ``'REST'``, ``'GALACTO'``,
        ``'LGROUP'``, ``'CMB'``), a FITS ``SPECSYS`` value (``'BARYCENT'``,
        ...) or a schema observer (``'lsrk'``, ``'gcrs'``, ...), in any case.
        The astropy frames ``'icrs'``, ``'hcrs'`` and ``'lsr'``, which have no
        casacore equivalent, are accepted too, but images in them cannot be
        written to CASA or FITS.
    Returns
    -------
    xarray.Dataset

    Raises
    ------
    TypeError
        If ``time_coords`` are neither numbers nor times.
    ValueError
        If ``pol_coords`` are not in canonical order, or
        ``spectral_reference`` is not a spectral reference frame.
    """
    return _make_empty_aperture_image(
        phase_center,
        image_size,
        sky_image_cell_size,
        frequency_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
    )


def make_empty_lmuv_image(
    phase_center: list[float] | np.ndarray,
    image_size: list[int] | np.ndarray,
    sky_image_cell_size: list[float] | np.ndarray,
    frequency_coords: list[float] | np.ndarray,
    pol_coords: list[float] | np.ndarray,
    time_coords: list[float] | np.ndarray,
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
    frequency_coords : list or np.ndarray
        The center frequency in Hz of each image channel.
    pol_coords : list or np.ndarray
        The polarization label of each image polarization, in canonical
        (casacore ``Stokes``, Jones matrix) order, for example
        ``["I", "Q", "U", "V"]``, ``["RR", "RL", "LR", "LL"]`` or
        ``["XX", "YY"]``.
    time_coords : float, list, np.ndarray, str, datetime or astropy.time.Time
        The time of each temporal plane: numbers are MJD (UTC) days;
        ``datetime64`` values, ISO time strings, ``datetime.datetime`` objects
        and astropy ``Time`` objects are converted to MJD (UTC) days.
    direction_reference : str, default = 'fk5'
        Direction reference frame: ``'fk5'`` (equinox J2000), ``'fk4'``
        (equinox B1950), ``'icrs'`` or ``'galactic'``.
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
        Spectral reference frame: a casacore frame (``'LSRK'``, ``'LSRD'``,
        ``'BARY'``, ``'GEO'``, ``'TOPO'``, ``'REST'``, ``'GALACTO'``,
        ``'LGROUP'``, ``'CMB'``), a FITS ``SPECSYS`` value (``'BARYCENT'``,
        ...) or a schema observer (``'lsrk'``, ``'gcrs'``, ...), in any case.
        The astropy frames ``'icrs'``, ``'hcrs'`` and ``'lsr'``, which have no
        casacore equivalent, are accepted too, but images in them cannot be
        written to CASA or FITS.
    do_sky_coords : bool
        If True, compute SkyCoord at each pixel and add spherical (sky) dimensions as
        non-dimensional coordinates in the returned xr.Dataset.
    Returns
    -------
    xarray.Dataset

    Raises
    ------
    TypeError
        If ``time_coords`` are neither numbers nor times.
    ValueError
        If ``pol_coords`` are not in canonical order, or
        ``spectral_reference`` is not a spectral reference frame.
    """
    return _make_empty_lmuv_image(
        phase_center,
        image_size,
        sky_image_cell_size,
        frequency_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
        do_sky_coords,
    )
