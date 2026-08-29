import numpy as np
import xarray as xr
from astropy.io import fits
from astropy.time import Time

from xradio._utils.logging import xradio_logger
from xradio.image._util.common import _compute_sky_reference_pixel
from xradio.measurement_set._utils._utils.stokes_types import stokes_types

_RAD_TO_DEG = 180.0 / np.pi
_RAD_TO_ARCSEC = _RAD_TO_DEG * 3600.0

# Frames the FITS reader can round trip (it only supports RA-/DEC- CTYPEs)
_EQUATORIAL_FRAMES = ("fk4", "fk5", "icrs")


def _equinox_to_fits(equinox: str) -> float:
    """Convert an equinox string such as 'j2000.0' or 'b1950.0' to the FITS
    EQUINOX value (a year number)."""
    eq = equinox.lower().lstrip("jb")
    return float(eq)


def _direction_header_cards(xds: xr.Dataset, header: fits.Header) -> None:
    csys = xds.attrs.get("coordinate_system_info")
    if not csys:
        raise RuntimeError(
            "Writing to FITS requires the coordinate_system_info dataset "
            "attribute (only sky plane images are supported)"
        )
    ref_dir = csys["reference_direction"]
    frame = ref_dir["attrs"]["frame"]
    if frame.lower() not in _EQUATORIAL_FRAMES:
        raise RuntimeError(
            f"Writing to FITS is only supported for equatorial direction "
            f"reference frames {_EQUATORIAL_FRAMES}, got '{frame}'"
        )
    projection = csys["projection"]
    header["CTYPE1"] = "RA---" + projection
    header["CTYPE2"] = "DEC--" + projection
    header["CRVAL1"] = ref_dir["data"][0] * _RAD_TO_DEG
    header["CRVAL2"] = ref_dir["data"][1] * _RAD_TO_DEG
    crpix = _compute_sky_reference_pixel(xds)
    header["CRPIX1"] = crpix[0] + 1
    header["CRPIX2"] = crpix[1] + 1
    l_vals = xds.l.values
    m_vals = xds.m.values
    header["CDELT1"] = float(l_vals[1] - l_vals[0]) * _RAD_TO_DEG
    header["CDELT2"] = float(m_vals[1] - m_vals[0]) * _RAD_TO_DEG
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["RADESYS"] = frame.upper()
    equinox = ref_dir["attrs"].get("equinox")
    if frame.lower() != "icrs" and equinox is not None:
        header["EQUINOX"] = _equinox_to_fits(equinox)
    pole = csys["native_pole_direction"]["data"]
    header["LONPOLE"] = pole[0] * _RAD_TO_DEG
    header["LATPOLE"] = pole[1] * _RAD_TO_DEG
    pc = csys["pixel_coordinate_transformation_matrix"]
    for i in (0, 1):
        for j in (0, 1):
            header[f"PC{i + 1}_{j + 1}"] = float(pc[i][j])
    header["PC3_3"] = 1.0
    header["PC4_4"] = 1.0


def _stokes_header_cards(xds: xr.Dataset, header: fits.Header) -> None:
    labels = list(xds.polarization.values)
    stokes_indices = {label: index for index, label in stokes_types.items()}
    indices = [stokes_indices[label] for label in labels]
    for prev, cur in zip(indices[:-1], indices[1:], strict=False):
        if cur != prev + 1:
            raise RuntimeError(
                f"Cannot write polarizations {labels} to FITS: the STOKES "
                "axis requires consecutive casacore stokes types"
            )
    header["CTYPE3"] = "STOKES"
    header["CRVAL3"] = float(indices[0])
    header["CDELT3"] = 1.0
    header["CRPIX3"] = 1.0
    header["CUNIT3"] = ""


def _frequency_header_cards(xds: xr.Dataset, header: fits.Header) -> None:
    freq = xds.frequency.values
    freq_attrs = xds.frequency.attrs
    crval = float(freq_attrs["reference_frequency"]["data"])
    cdelt = float(freq[1] - freq[0]) if len(freq) > 1 else 1000.0
    header["CTYPE4"] = "FREQ"
    header["CRVAL4"] = crval
    header["CDELT4"] = cdelt
    header["CRPIX4"] = (crval - float(freq[0])) / cdelt + 1
    header["CUNIT4"] = "Hz"
    header["SPECSYS"] = freq_attrs.get("frame", "LSRK")
    header["RESTFRQ"] = float(freq_attrs["rest_frequency"]["data"])


def _misc_header_cards(image: xr.DataArray, xds: xr.Dataset, header) -> None:
    attrs = image.attrs
    if attrs.get("units"):
        header["BUNIT"] = attrs["units"]
    if attrs.get("sub_type"):
        header["BTYPE"] = attrs["sub_type"]
    if attrs.get("object_name"):
        header["OBJECT"] = attrs["object_name"]
    if attrs.get("observer"):
        header["OBSERVER"] = attrs["observer"]
    telescope = attrs.get("telescope") or {}
    header["TELESCOP"] = telescope.get("name", "UNKNOWN")
    direction = telescope.get("direction")
    distance = telescope.get("distance")
    if direction is not None and distance is not None:
        lon, lat = direction["data"]
        r = distance["data"][0]
        header["OBSGEO-X"] = r * np.cos(lat) * np.cos(lon)
        header["OBSGEO-Y"] = r * np.cos(lat) * np.sin(lon)
        header["OBSGEO-Z"] = r * np.sin(lat)
    time_attrs = xds.time.attrs
    obstime = Time(
        float(xds.time.values[0]),
        format=time_attrs.get("format", "mjd"),
        scale=time_attrs.get("scale", "utc"),
    )
    header["DATE-OBS"] = obstime.isot
    header["TIMESYS"] = time_attrs.get("scale", "utc").upper()
    for key, value in (attrs.get("user") or {}).items():
        keyword = key.upper()
        if len(keyword) > 8 or keyword in header:
            continue
        try:
            header[keyword] = value
        except (ValueError, TypeError):
            xradio_logger().warning(
                f"Could not write user keyword {keyword} to the FITS header"
            )


def _beam_hdu_or_cards(xds: xr.Dataset, header: fits.Header) -> fits.BinTableHDU | None:
    """Write a single beam as BMAJ/BMIN/BPA header cards, or per plane beams
    as a CASA style BEAMS binary table extension."""
    if "BEAM_FIT_PARAMS" not in xds.data_vars:
        return None
    beams = np.asarray(xds["BEAM_FIT_PARAMS"].values)  # (time, chan, pol, 3), rad
    if np.allclose(beams, beams[0, 0, 0]):
        header["BMAJ"] = beams[0, 0, 0, 0] * _RAD_TO_DEG
        header["BMIN"] = beams[0, 0, 0, 1] * _RAD_TO_DEG
        header["BPA"] = beams[0, 0, 0, 2] * _RAD_TO_DEG
        return None
    header["CASAMBM"] = True
    nchan = beams.shape[1]
    npol = beams.shape[2]
    chans = np.repeat(np.arange(nchan), npol)
    pols = np.tile(np.arange(npol), nchan)
    beam_rows = beams[0, chans, pols]
    columns = fits.ColDefs(
        [
            fits.Column(
                name="BMAJ",
                format="E",
                unit="arcsec",
                array=beam_rows[:, 0] * _RAD_TO_ARCSEC,
            ),
            fits.Column(
                name="BMIN",
                format="E",
                unit="arcsec",
                array=beam_rows[:, 1] * _RAD_TO_ARCSEC,
            ),
            fits.Column(
                name="BPA", format="E", unit="deg", array=beam_rows[:, 2] * _RAD_TO_DEG
            ),
            fits.Column(name="CHAN", format="J", array=chans),
            fits.Column(name="POL", format="J", array=pols),
        ]
    )
    beams_hdu = fits.BinTableHDU.from_columns(columns, name="BEAMS")
    beams_hdu.header["NCHAN"] = nchan
    beams_hdu.header["NPOL"] = npol
    return beams_hdu


def _xds_to_fits_image(xds: xr.Dataset, image_store_name: str) -> None:
    """Write a single image dataset (data variable SKY, with optional FLAG and
    BEAM_FIT_PARAMS variables) to a FITS file that the FITS reader can round
    trip. Flagged pixels are written as NaN, following FITS convention."""
    if xds.sizes.get("time", 1) != 1:
        raise RuntimeError(
            "XDS can only be converted to FITS if it has exactly one time plane"
        )
    if "l" not in xds.dims or "m" not in xds.dims:
        raise RuntimeError(
            "Writing to FITS is only supported for sky plane images (with l "
            "and m dimensions)"
        )
    image = xds["SKY"]
    data = image.isel(time=0).transpose("frequency", "polarization", "m", "l").values
    if data.dtype not in (np.float32, np.float64):
        xradio_logger().warning(
            f"Converting image data of type {data.dtype} to float32 for FITS"
        )
        data = data.astype(np.float32)
    if "FLAG" in xds.data_vars:
        flag = (
            xds["FLAG"].isel(time=0).transpose("frequency", "polarization", "m", "l")
        ).values
        data = np.where(flag, np.nan, data)

    header = fits.Header()
    _direction_header_cards(xds, header)
    _stokes_header_cards(xds, header)
    _frequency_header_cards(xds, header)
    beams_hdu = _beam_hdu_or_cards(xds, header)
    _misc_header_cards(image, xds, header)

    primary = fits.PrimaryHDU(data=data)
    primary.header.update(header)
    hdus = [primary] if beams_hdu is None else [primary, beams_hdu]
    # overwrite=True: like the CASA writer, several data groups can share the
    # same image type and the last one written wins
    fits.HDUList(hdus).writeto(image_store_name, overwrite=True)
