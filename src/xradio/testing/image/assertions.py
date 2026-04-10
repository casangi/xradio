"""Image-specific assertion and comparison helpers.

All functions raise ``AssertionError`` on failure and are framework-agnostic,
so they work equally in pytest, unittest, and ASV benchmarks.
"""

from __future__ import annotations

import numpy as np


def normalize_image_coords_for_compare(
    coords: dict,
    factor: float = 180 * 60 / np.pi,
) -> None:
    """Normalise a CASA image coordinate dict for round-trip comparison.

    When an image is written from an ``xr.Dataset`` and re-opened by
    casacore, direction coordinate values are stored in radians whereas
    the original CASA image stores them in arcminutes.  This function
    converts the direction entries in *coords* by multiplying by *factor*
    (default: rad → arcmin) and sets the spectral velocity unit to
    ``"km/s"`` so the two dicts can be compared with
    :func:`~xradio.testing.assert_attrs_dicts_equal`.

    Modifies *coords* **in place**.

    Parameters
    ----------
    coords : dict
        Coordinate dict returned by ``casacore.images.image.info()["coordinates"]``
        or ``casacore.tables.table.getkeywords()["coords"]``.
    factor : float, optional
        Multiplicative scale applied to ``cdelt`` and ``crval`` of the
        ``direction0`` sub-dict.  Defaults to ``180 * 60 / π``
        (radians → arcminutes).
    """
    direction = coords["direction0"]
    direction["cdelt"] *= factor
    direction["crval"] *= factor
    direction["units"] = ["'", "'"]
    coords["spectral2"]["velUnit"] = "km/s"


def assert_image_block_equal(
    xds,
    output_path: str,
    zarr: bool = False,
) -> None:
    """Write an image with a synthetic beam array, reload a spatial block, and
    assert equality with the expected slice.

    Workflow
    --------
    1. Attach a ``BEAM_FIT_PARAMS`` variable (built by
       :func:`~xradio.testing.image.generators.make_beam_fit_params`) to a
       copy of *xds*.
    2. Write the augmented dataset to *output_path*.
    3. Load a fixed spatial block
       ``{l: 2–10, m: 3–15, polarization: 0–1, frequency: 0–4}`` from the
       written image.
    4. Assert that the loaded block equals the corresponding slice of the
       written dataset using
       :func:`~xradio.testing.assert_xarray_datasets_equal`.

    Parameters
    ----------
    xds : xr.Dataset
        Full image dataset to augment and write.
    output_path : str
        Destination path for the written image.
    zarr : bool, optional
        If *True* write in zarr format; otherwise write as a CASA image.
    """
    from xradio.image import load_image, write_image
    from xradio.testing import assert_xarray_datasets_equal
    from xradio.testing.image.generators import make_beam_fit_params

    bfp = make_beam_fit_params(xds)
    bfp.attrs["units"] = "rad"
    xds_with_beam = xds.assign(BEAM_FIT_PARAMS=bfp)

    write_image(
        xds_with_beam,
        output_path,
        out_format="zarr" if zarr else "casa",
        overwrite=True,
    )

    loaded = load_image(
        output_path,
        {
            "l": slice(2, 10),
            "m": slice(3, 15),
            "polarization": slice(0, 1),
            "frequency": slice(0, 4),
        },
        do_sky_coords=True,
    )
    true_xds = xds_with_beam.isel(
        polarization=slice(0, 1),
        frequency=slice(0, 4),
        l=slice(2, 10),
        m=slice(3, 15),
    )
    assert_xarray_datasets_equal(loaded, true_xds)
