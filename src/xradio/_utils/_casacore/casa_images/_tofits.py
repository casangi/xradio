"""Write a CASA paged image (read via casacoretables) to a FITS file.

This reimplements the slice of ``casacore.images.image.tofits`` that xradio's
test-suite exercises: a primary HDU with a WCS header derived from the image's
coordinate system, plus -- for multi-beam images -- a ``BEAMS`` binary-table
extension in the layout casacore writes and xradio's FITS reader expects. The
WCS is built with astropy (the sanctioned coordinate library).
"""

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time

# casacore direction reference frame -> (FITS RADESYS, EQUINOX or None).
_RADESYS_MAP = {
    "J2000": ("FK5", 2000.0),
    "B1950": ("FK4", 1950.0),
    "ICRS": ("ICRS", None),
    "GALACTIC": ("GALACTIC", None),
}

# casacore direction-axis unit strings -> astropy.
_UNIT_MAP = {
    "'": u.arcmin,
    "arcmin": u.arcmin,
    "''": u.arcsec,
    '"': u.arcsec,
    "arcsec": u.arcsec,
    "deg": u.deg,
    "rad": u.rad,
}

_PREFIXES = ("direction", "spectral", "stokes", "linear")


def _unit(unit_str):
    if unit_str in _UNIT_MAP:
        return _UNIT_MAP[unit_str]
    return u.Unit(unit_str)


def _coord_for_index(coords: dict, n: int):
    """Return (prefix, record) for coordinate number ``n`` (e.g. 'spectral1')."""
    for prefix in _PREFIXES:
        key = f"{prefix}{n}"
        if key in coords:
            return prefix, coords[key]
    return None, None


def _axis_wcs_from_coords(coords: dict, ndim: int) -> dict:
    """Map each casacore pixel axis -> dict(ctype, crval, cdelt, crpix, cunit).

    Uses the ``pixelmapN`` records, so any axis ordering is handled.
    """
    axes = {}
    n = 0
    while f"pixelmap{n}" in coords:
        pmap = [int(a) for a in np.atleast_1d(coords[f"pixelmap{n}"])]
        prefix, rec = _coord_for_index(coords, n)
        if prefix == "direction":
            units = rec.get("units", ["deg", "deg"])
            proj = rec.get("projection", "SIN")
            crval = np.asarray(rec["crval"], dtype=float)
            cdelt = np.asarray(rec["cdelt"], dtype=float)
            crpix = np.asarray(rec["crpix"], dtype=float)
            for comp, pix_ax in enumerate(pmap):
                u_i = _unit(units[comp])
                ctype = ("RA---" if comp == 0 else "DEC--") + proj
                axes[pix_ax] = {
                    "ctype": ctype,
                    "crval": float((crval[comp] * u_i).to(u.deg).value),
                    "cdelt": float((cdelt[comp] * u_i).to(u.deg).value),
                    "crpix": float(crpix[comp]) + 1.0,
                    "cunit": "deg",
                }
        elif prefix == "spectral":
            wcs = rec["wcs"]
            axes[pmap[0]] = {
                "ctype": "FREQ",
                "crval": float(wcs["crval"]),
                "cdelt": float(wcs["cdelt"]),
                "crpix": float(wcs["crpix"]) + 1.0,
                "cunit": rec.get("unit", "Hz"),
            }
        elif prefix == "stokes":
            crval = np.atleast_1d(np.asarray(rec["crval"], dtype=float))
            cdelt = np.atleast_1d(np.asarray(rec["cdelt"], dtype=float))
            crpix = np.atleast_1d(np.asarray(rec["crpix"], dtype=float))
            axes[pmap[0]] = {
                "ctype": "STOKES",
                "crval": float(crval[0]),
                "cdelt": float(cdelt[0]),
                "crpix": float(crpix[0]) + 1.0,
                "cunit": "",
            }
        elif prefix == "linear":
            units = rec.get("units", ["", ""])
            crval = np.asarray(rec["crval"], dtype=float)
            cdelt = np.asarray(rec["cdelt"], dtype=float)
            crpix = np.asarray(rec["crpix"], dtype=float)
            for comp, pix_ax in enumerate(pmap):
                axes[pix_ax] = {
                    "ctype": rec.get("axes", ["", ""])[comp][:8],
                    "crval": float(crval[comp]),
                    "cdelt": float(cdelt[comp]),
                    "crpix": float(crpix[comp]) + 1.0,
                    "cunit": units[comp] if comp < len(units) else "",
                }
        n += 1
    return axes


def _build_header(coords: dict, ndim: int, unit: str, multibeam: bool) -> fits.Header:
    hdr = fits.Header()
    axes = _axis_wcs_from_coords(coords, ndim)
    for pix_ax in range(ndim):
        info = axes.get(pix_ax)
        if info is None:
            continue
        i = pix_ax + 1  # FITS axes are 1-based
        hdr[f"CTYPE{i}"] = info["ctype"]
        hdr[f"CRVAL{i}"] = info["crval"]
        hdr[f"CDELT{i}"] = info["cdelt"]
        hdr[f"CRPIX{i}"] = info["crpix"]
        hdr[f"CUNIT{i}"] = info["cunit"]
    if unit:
        hdr["BUNIT"] = unit

    # Direction frame, native pole and PC matrix.
    for key in coords:
        if key.startswith("direction"):
            _add_direction_keywords(hdr, coords[key], pixel_axes_of(coords, key))
            break
    for key in coords:
        if key.startswith("spectral"):
            spec = coords[key]
            hdr["RESTFRQ"] = float(spec.get("restfreq", 0.0))
            hdr["SPECSYS"] = spec.get("system", "LSRK")
            break

    # Observation metadata.
    if "telescope" in coords:
        hdr["TELESCOP"] = coords["telescope"]
    if "observer" in coords:
        hdr["OBSERVER"] = coords["observer"]
    if "obsdate" in coords:
        obsdate = coords["obsdate"]
        scale = obsdate.get("refer", "UTC")
        mjd = float(obsdate["m0"]["value"])
        try:
            hdr["DATE-OBS"] = Time(mjd, format="mjd", scale=scale.lower()).isot
        except Exception:
            hdr["DATE-OBS"] = Time(mjd, format="mjd").isot
        hdr["TIMESYS"] = scale

    if multibeam:
        hdr["CASAMBM"] = True
    return hdr


def pixel_axes_of(coords: dict, dir_key: str):
    """Return the casacore pixel axes occupied by coordinate ``dir_key``."""
    n = int("".join(ch for ch in dir_key if ch.isdigit()))
    return [int(a) for a in np.atleast_1d(coords[f"pixelmap{n}"])]


def _add_direction_keywords(hdr: fits.Header, direction: dict, dir_axes) -> None:
    system = direction.get("system", "J2000")
    radesys, equinox = _RADESYS_MAP.get(system, (system, None))
    hdr["RADESYS"] = radesys
    if equinox is not None:
        hdr["EQUINOX"] = equinox
    # latpole/longpole are filled (in degrees) by the image coords post-processor.
    hdr["LONPOLE"] = float(direction.get("longpole", 180.0))
    hdr["LATPOLE"] = float(direction.get("latpole", 0.0))
    pc = np.asarray(direction.get("pc", [[1.0, 0.0], [0.0, 1.0]]), dtype=float)
    for i in (0, 1):
        for j in (0, 1):
            hdr[f"PC{dir_axes[i] + 1}_{dir_axes[j] + 1}"] = float(pc[i][j])


def _beams_hdu(perplanebeams: dict) -> fits.BinTableHDU:
    nchan = int(perplanebeams["nChannels"])
    npol = int(perplanebeams["nStokes"])
    bmaj, bmin, bpa, chans, pols = [], [], [], [], []
    for pol in range(npol):
        for chan in range(nchan):
            beam = perplanebeams[f"*{pol * nchan + chan}"]
            maj = beam["major"]
            minr = beam["minor"]
            pa = beam.get("positionangle", beam.get("pa"))
            bmaj.append(float((maj["value"] * u.Unit(maj["unit"])).to(u.arcsec).value))
            bmin.append(
                float((minr["value"] * u.Unit(minr["unit"])).to(u.arcsec).value)
            )
            bpa.append(float((pa["value"] * u.Unit(pa["unit"])).to(u.deg).value))
            chans.append(chan)
            pols.append(pol)
    cols = fits.ColDefs(
        [
            fits.Column(name="BMAJ", format="E", unit="arcsec", array=np.array(bmaj)),
            fits.Column(name="BMIN", format="E", unit="arcsec", array=np.array(bmin)),
            fits.Column(name="BPA", format="E", unit="deg", array=np.array(bpa)),
            fits.Column(name="CHAN", format="J", array=np.array(chans, dtype=np.int32)),
            fits.Column(name="POL", format="J", array=np.array(pols, dtype=np.int32)),
        ]
    )
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["EXTNAME"] = "BEAMS"
    hdu.header["NCHAN"] = nchan
    hdu.header["NPOL"] = npol
    return hdu


def to_fits(img, filename: str, overwrite: bool = True) -> None:
    """Write ``img`` (a casa_images.image) to ``filename`` in FITS format."""
    info = img.info()
    coords = info["coordinates"]
    imageinfo = info["imageinfo"]
    data = np.asarray(img.getdata())
    ndim = data.ndim
    multibeam = "perplanebeams" in imageinfo

    header = _build_header(coords, ndim, info.get("unit", ""), multibeam)
    # FITS pixels are written as float32 for a single-precision image.
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    hdus = [fits.PrimaryHDU(data=data, header=header)]

    if multibeam:
        hdus.append(_beams_hdu(imageinfo["perplanebeams"]))
    elif "restoringbeam" in imageinfo:
        beam = imageinfo["restoringbeam"]
        maj = beam["major"]
        minr = beam["minor"]
        pa = beam.get("positionangle", beam.get("pa"))
        hdus[0].header["BMAJ"] = float(
            (maj["value"] * u.Unit(maj["unit"])).to(u.deg).value
        )
        hdus[0].header["BMIN"] = float(
            (minr["value"] * u.Unit(minr["unit"])).to(u.deg).value
        )
        hdus[0].header["BPA"] = float(
            (pa["value"] * u.Unit(pa["unit"])).to(u.deg).value
        )

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
