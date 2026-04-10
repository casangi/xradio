"""Test-data generators for image tests.

All functions are framework-agnostic and can be used in pytest, ASV
benchmarks, or any other harness that imports ``xradio.testing.image``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def make_beam_fit_params(xds: xr.Dataset) -> xr.DataArray:
    """Build a ``BEAM_FIT_PARAMS`` DataArray from an image Dataset.

    Creates a synthetic beam-parameter array whose shape is derived from
    the *time*, *frequency*, and *polarization* dimensions of *xds*.
    The first time–channel–polarization entry is set to 2.0; all others
    are 1.0.

    Parameters
    ----------
    xds : xr.Dataset
        An open image dataset with ``time``, ``frequency``, and
        ``polarization`` dimensions.

    Returns
    -------
    xr.DataArray
        Array with dims ``["time", "frequency", "polarization",
        "beam_params_label"]`` and coords inherited from *xds*.
    """
    shape = (
        xds.sizes["time"],
        xds.sizes["frequency"],
        xds.sizes["polarization"],
        3,
    )
    ary = np.ones(shape, dtype=np.float32)
    ary[0, 2, 0, :] = 2.0
    return xr.DataArray(
        data=ary,
        dims=["time", "frequency", "polarization", "beam_params_label"],
        coords={
            "time": xds.time,
            "frequency": xds.frequency,
            "polarization": xds.polarization,
            "beam_params_label": ["major", "minor", "pa"],
        },
    )


def create_empty_test_image(factory, do_sky_coords=None) -> xr.Dataset:
    """Call a ``make_empty_*`` factory with canonical test arguments.

    Provides a single set of standard test coordinates so every empty-image
    factory can be exercised with the same call.

    Parameters
    ----------
    factory : callable
        One of ``make_empty_sky_image``, ``make_empty_aperture_image``, or
        ``make_empty_lmuv_image``.
    do_sky_coords : bool or None, optional
        Forwarded as ``do_sky_coords`` keyword argument when not *None*.

    Returns
    -------
    xr.Dataset
        The empty image dataset produced by *factory*.
    """
    args = [
        [0.2, -0.5],  # phase_center
        [10, 10],  # image_size
        [np.pi / 180 / 60, np.pi / 180 / 60],  # cell_size
        [1.412e9, 1.413e9],  # frequency
        ["I", "Q", "U"],  # polarization
        [54000.1],  # time
    ]
    kwargs = {} if do_sky_coords is None else {"do_sky_coords": do_sky_coords}
    return factory(*args, **kwargs)


def scale_data_for_int16(data: np.ndarray) -> np.ndarray:
    """Scale a float array to the int16 range for FITS BSCALE/BZERO testing.

    Replaces NaNs with zero, clips to ``[-32768, 32767]``, and casts to
    ``int16``.

    Parameters
    ----------
    data : np.ndarray
        Input floating-point array.

    Returns
    -------
    np.ndarray
        A new array of dtype ``int16``.
    """
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, -32768, 32767)
    return data.astype(np.int16)


def create_bzero_bscale_fits(
    outname: str, source_fits: str, bzero: float, bscale: float
) -> None:
    """Write a FITS file with explicit BSCALE/BZERO headers for guard testing.

    Reads pixel data from *source_fits*, scales it to the int16 range via
    :func:`scale_data_for_int16`, and writes a new FITS primary HDU to
    *outname* with the given BSCALE and BZERO header keywords.

    Parameters
    ----------
    outname : str
        Destination FITS file path.
    source_fits : str
        Source FITS file whose pixel data is used as the basis.
    bzero : float
        Value written to the ``BZERO`` header keyword.
    bscale : float
        Value written to the ``BSCALE`` header keyword.
    """
    from astropy.io import fits

    with fits.open(source_fits) as hdulist:
        data = scale_data_for_int16(hdulist[0].data)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["BSCALE"] = bscale
    hdu.header["BZERO"] = bzero
    hdu.writeto(outname, overwrite=True)
