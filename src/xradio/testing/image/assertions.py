"""Image-specific assertion and comparison helpers.

All functions raise ``AssertionError`` on failure and are framework-agnostic,
so they work equally in pytest, unittest, and ASV benchmarks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr


def normalize_image_coords_for_compare(
    coords: dict,
    factor: float = 180 * 60 / np.pi,
    direction_key: str = "direction0",
    spectral_key: str = "spectral2",
    direction_units: list[str] | None = None,
    vel_unit: str = "km/s",
) -> None:
    """Normalise a CASA image coordinate dict for round-trip comparison.

    When an image is written from an ``xr.Dataset`` and re-opened by
    casacore, direction coordinate values are stored in radians whereas
    the original CASA image stores them in arcminutes.  This function
    converts the direction entries in *coords* by multiplying by *factor*
    (default: rad → arcmin) and sets the spectral velocity unit so the
    two dicts can be compared with
    :func:`~xradio.testing.assert_attrs_dicts_equal`.

    Modifies *coords* **in place**.

    Parameters
    ----------
    coords : dict
        Coordinate dict returned by ``casacore.images.image.info()["coordinates"]``
        or ``casacore.tables.table.getkeywords()["coords"]``.
    factor : float, optional
        Multiplicative scale applied to ``cdelt`` and ``crval`` of the
        direction sub-dict.  Defaults to ``180 * 60 / π`` (radians → arcminutes).
    direction_key : str, optional
        Key of the direction coordinate entry in *coords*.
        Defaults to ``"direction0"``.
    spectral_key : str, optional
        Key of the spectral coordinate entry in *coords*.
        Defaults to ``"spectral2"``.
    direction_units : list of str or None, optional
        Unit strings written into the direction sub-dict after scaling.
        Defaults to ``["'", "'"]`` (arcminutes) when *None*.
    vel_unit : str, optional
        Velocity unit string written into ``coords[spectral_key]["velUnit"]``.
        Defaults to ``"km/s"``.
    """
    if direction_units is None:
        direction_units = ["'", "'"]
    direction = coords[direction_key]
    direction["cdelt"] *= factor
    direction["crval"] *= factor
    direction["units"] = direction_units
    coords[spectral_key]["velUnit"] = vel_unit


def assert_image_block_equal(
    xds: xr.Dataset,
    output_path: str,
    selection: Dict[str, slice],
    zarr: bool = False,
    do_sky_coords: bool = True,
) -> None:
    """Write an image, reload a spatial block, and assert equality with the
    corresponding slice of the original dataset.

    Workflow
    --------
    1. Write *xds* to *output_path*.
    2. Load the region specified by *selection* from the written image via
       :func:`~xradio.image.load_image`.
    3. Compute the equivalent slice of *xds* with ``isel``.
    4. Assert equality using
       :func:`~xradio.testing.assert_xarray_datasets_equal`.

    Parameters
    ----------
    xds : xr.Dataset
        Full image dataset to write and slice.  Augment the dataset with
        any extra data variables (e.g. ``BEAM_FIT_PARAMS``) *before* calling
        this function if you want them included in the comparison.
    output_path : str
        Destination path for the written image.  The path is overwritten if
        it already exists.
    selection : dict of str to slice
        Mapping of dimension name to ``slice`` that defines the block to load
        and compare.  Every slice end must not exceed the corresponding
        dimension size in *xds*.
    zarr : bool, optional
        If *True* write in zarr format; otherwise write as a CASA image.
        Defaults to *False*.
    do_sky_coords : bool, optional
        Forwarded to :func:`~xradio.image.load_image` as ``do_sky_coords``.
        Defaults to *True*.

    Raises
    ------
    ValueError
        If any slice in *selection* exceeds the size of the corresponding
        dimension in *xds*.
    """
    from xradio.image import load_image, write_image
    from xradio.testing import assert_xarray_datasets_equal

    bad_dims = []
    for dim, slc in selection.items():
        size = xds.sizes.get(dim, 0)
        stop = slc.stop if slc.stop is not None else size
        if stop > size:
            bad_dims.append(f"{dim}: slice stop {stop} > size {size}")
    if bad_dims:
        raise ValueError(
            "assert_image_block_equal: selection exceeds dataset dimensions — "
            + ", ".join(bad_dims)
        )

    write_image(xds, output_path, out_format="zarr" if zarr else "casa", overwrite=True)

    loaded = load_image(output_path, selection, do_sky_coords=do_sky_coords)
    true_xds = xds.isel(**selection)
    assert_xarray_datasets_equal(loaded, true_xds)
