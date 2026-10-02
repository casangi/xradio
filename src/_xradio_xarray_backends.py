"""
xarray backend entry points of xradio for CASA and FITS images.

xarray loads every installed backend entry point on each ``xr.open_dataset``,
``xr.open_mfdataset`` and ``xr.open_datatree`` call made without an explicit
engine, and on ``xr.backends.list_engines()`` (xarray older than 2026.01 also
for explicit string engines such as ``"zarr"``). This module is therefore
light: it imports only :mod:`os` and ``xarray.backends``, and it is a top level
module of the xradio distribution rather than a module of the ``xradio``
package, so that loading the entry points does not run ``xradio/__init__.py``
(which imports zarr and sets zarr warning filters). The image readers
(:mod:`xradio.image` with dask, astropy and casacore) are imported only when an
image is actually opened. The implementation, with the documentation of the
parameters, is in :mod:`xradio.image.backends`, which re-exports the entry
point classes.
"""

import os

from xarray.backends import BackendEntrypoint

__all__ = ["CasaImageBackendEntrypoint", "FitsImageBackendEntrypoint"]

_URL = "https://xradio.readthedocs.io/en/latest/image_data/schema.html"

#: File name suffixes claimed by the FITS engine when no engine is given.
FITS_SUFFIXES = (".fits", ".fit", ".fts")


def _local_path(filename_or_obj) -> str | None:
    """Return a str or os.PathLike as a local path (with ``~`` expanded), or
    None for anything else (file objects, bytes, data stores)."""
    if not isinstance(filename_or_obj, str | os.PathLike):
        return None
    path = os.fspath(filename_or_obj)
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    return os.path.expanduser(path)


def _is_casa_image(path: str) -> bool:
    """True if ``path`` is a casacore image table (a directory whose
    ``table.info`` starts with ``Type = Image``)."""
    if not os.path.isdir(path):
        return False
    try:
        with open(os.path.join(path, "table.info"), "rb") as info:
            first_line = info.readline(256)
    except PermissionError:
        raise
    except OSError:
        return False
    return first_line.strip() == b"Type = Image"


def _astropy_fits():
    """The astropy.io.fits module, or None when astropy is not installed."""
    try:
        from astropy.io import fits
    except ImportError:
        return None
    return fits


def _is_fits_image(path: str) -> bool:
    """True if ``path`` is a FITS file whose primary HDU holds an image, the
    HDU the FITS image reader reads (random groups (UVFITS) and files with
    images in extension HDUs only are not claimed). False without astropy."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fits_file:
            if not fits_file.read(9) == b"SIMPLE  =":
                return False
    except PermissionError:
        raise
    except OSError:
        return False
    fits = _astropy_fits()
    if fits is None:
        return False
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
                primary = hdus[0]
                return (
                    isinstance(primary, fits.PrimaryHDU)
                    and not isinstance(primary, fits.GroupsHDU)
                    and bool(primary.header.get("NAXIS") or 0)
                )
    except PermissionError:
        raise
    except Exception:
        # Not a valid FITS file (or a damaged one)
        return False


def _is_zarr_store(path: str) -> bool:
    return os.path.isdir(path) and any(
        os.path.exists(os.path.join(path, name))
        for name in ("zarr.json", ".zgroup", ".zattrs", ".zarray")
    )


def _require_path(filename_or_obj, engine: str) -> str:
    """Return the local path to open, raising for objects and missing paths."""
    path = _local_path(filename_or_obj)
    if path is None:
        raise TypeError(
            f"The {engine} engine opens images from a local path (str or "
            f"os.PathLike), not from {type(filename_or_obj).__name__} objects."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: {path!r}")
    return path


def _wrong_format_message(path: str, engine: str) -> str | None:
    """Describe what ``path`` is when it is an image or store that another
    reader opens, else None."""
    if _is_casa_image(path):
        found, use = "a CASA image", "engine='xradio_casa_image'"
    elif _is_fits_image(path):
        found, use = "a FITS image", "engine='xradio_fits_image'"
    elif _is_zarr_store(path):
        found, use = "a zarr store", "xradio.image.open_image or xr.open_zarr"
    else:
        return None
    return f"{path!r} is {found}, not what the {engine} engine opens; use {use}."


class CasaImageBackendEntrypoint(BackendEntrypoint):
    """Open a CASA image (casacore image table) as an xradio image dataset.

    See :func:`xradio.image.backends.open_image_dataset` for the parameters.
    """

    description = "Open CASA images (casacore image tables) as xradio image datasets"
    url = _URL
    open_dataset_parameters = (
        "filename_or_obj",
        "drop_variables",
        "image_type",
        "image_chunks",
        "do_sky_coords",
        "verbose",
    )

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        image_type=None,
        image_chunks=None,
        do_sky_coords=True,
        verbose=False,
    ):
        path = _require_path(filename_or_obj, "xradio_casa_image")
        if not _is_casa_image(path):
            raise ValueError(
                _wrong_format_message(path, "xradio_casa_image")
                or f"{path!r} is not a CASA image (a casacore table directory "
                "whose table.info has 'Type = Image')."
            )
        from xradio.image.backends import open_image_dataset

        return open_image_dataset(
            path,
            drop_variables=drop_variables,
            image_type=image_type,
            image_chunks=image_chunks,
            do_sky_coords=do_sky_coords,
            verbose=verbose,
        )

    def guess_can_open(self, filename_or_obj) -> bool:
        path = _local_path(filename_or_obj)
        return path is not None and _is_casa_image(path)


class FitsImageBackendEntrypoint(BackendEntrypoint):
    """Open a FITS image as an xradio image dataset.

    See :func:`xradio.image.backends.open_image_dataset` for the parameters.
    """

    description = "Open FITS images as xradio image datasets"
    url = _URL
    open_dataset_parameters = (
        "filename_or_obj",
        "drop_variables",
        "image_type",
        "image_chunks",
        "do_sky_coords",
        "compute_mask",
        "verbose",
    )

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        image_type=None,
        image_chunks=None,
        do_sky_coords=True,
        compute_mask=True,
        verbose=False,
    ):
        path = _require_path(filename_or_obj, "xradio_fits_image")
        if _astropy_fits() is None:
            raise ModuleNotFoundError(
                "The xradio_fits_image engine reads FITS images with astropy, "
                "which is not installed (pip install astropy, or install xradio "
                "with one of its extras such as xradio[zarr])."
            )
        if not _is_fits_image(path):
            raise ValueError(
                _wrong_format_message(path, "xradio_fits_image")
                or f"{path!r} is not a FITS image the xradio_fits_image engine "
                "opens (a FITS file with an image in its primary HDU; random "
                "groups (UVFITS), table-only files and files with images in "
                "extension HDUs only are not)."
            )
        from xradio.image.backends import open_image_dataset

        return open_image_dataset(
            path,
            drop_variables=drop_variables,
            image_type=image_type,
            image_chunks=image_chunks,
            do_sky_coords=do_sky_coords,
            verbose=verbose,
            compute_mask=compute_mask,
        )

    def guess_can_open(self, filename_or_obj) -> bool:
        path = _local_path(filename_or_obj)
        return (
            path is not None
            and path.lower().endswith(FITS_SUFFIXES)
            and _is_fits_image(path)
        )
