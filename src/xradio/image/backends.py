"""
Xarray backend entrypoints for opening CASA and FITS images as xradio image
datasets, see https://docs.xarray.dev/en/latest/api/backends.html.

With xradio installed, images can be opened directly with xarray::

    import xarray as xr

    xds = xr.open_dataset("my_image.im", engine="xradio_casa_image")
    xds = xr.open_dataset("my_image.fits", engine="xradio_fits_image")

The backends are registered under the ``xradio_casa_image`` and
``xradio_fits_image`` engine names through the ``xarray.backends`` entry
points in pyproject.toml. The returned datasets conform to the image schema
(:py:class:`xradio.image.schema.ImageXds`) and hold lazy, dask backed data
variables.
"""

import os

from xarray.backends import BackendEntrypoint

__all__ = ["CasaImageBackendEntrypoint", "FitsImageBackendEntrypoint"]


class CasaImageBackendEntrypoint(BackendEntrypoint):
    """Open a CASA image (casacore image table) as an xradio image dataset."""

    description = "Open CASA images (casacore image tables) as xradio image datasets"
    url = "https://xradio.readthedocs.io/en/latest/image_data/schema.html"
    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "image_chunks",
        "do_sky_coords",
        "verbose",
    ]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        image_chunks=None,
        do_sky_coords=True,
        verbose=False,
    ):
        from xradio.image import open_image

        xds = open_image(
            str(filename_or_obj),
            chunks=image_chunks,
            verbose=verbose,
            do_sky_coords=do_sky_coords,
        )
        if drop_variables is not None:
            xds = xds.drop_vars(list(drop_variables), errors="ignore")
        return xds

    def guess_can_open(self, filename_or_obj):
        try:
            path = str(filename_or_obj)
        except TypeError:
            return False
        table_info = os.path.join(path, "table.info")
        if os.path.isdir(path) and os.path.isfile(table_info):
            with open(table_info) as info_file:
                return "Image" in info_file.readline()
        return False


class FitsImageBackendEntrypoint(BackendEntrypoint):
    """Open a FITS image as an xradio image dataset."""

    description = "Open FITS images as xradio image datasets"
    url = "https://xradio.readthedocs.io/en/latest/image_data/schema.html"
    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "image_chunks",
        "do_sky_coords",
        "compute_mask",
        "verbose",
    ]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        image_chunks=None,
        do_sky_coords=True,
        compute_mask=True,
        verbose=False,
    ):
        from xradio.image import open_image

        xds = open_image(
            str(filename_or_obj),
            chunks=image_chunks,
            verbose=verbose,
            do_sky_coords=do_sky_coords,
            compute_mask=compute_mask,
        )
        if drop_variables is not None:
            xds = xds.drop_vars(list(drop_variables), errors="ignore")
        return xds

    def guess_can_open(self, filename_or_obj):
        try:
            path = str(filename_or_obj)
        except TypeError:
            return False
        return path.lower().endswith(".fits")
