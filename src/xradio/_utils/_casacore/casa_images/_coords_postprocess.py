"""Post-process a raw CASA ``coords`` keyword to match python-casacore's form.

python-casacore returns the coordinate-system record via casacore's C++ layer,
which reconstructs a full ``CoordinateSystem`` (wcslib) and re-serializes it.
That recomputes a few fields that are *not* stored verbatim in the on-disk
``coords`` table keyword -- most importantly the direction coordinate's
``latpole`` / ``longpole`` (the native pole direction), which wcslib derives
from the reference value and projection.

Since we read the raw keyword directly via tables, we replicate that single
piece of post-processing here using astropy's WCS (which wraps the same wcslib),
so the record xradio consumes carries the same pole values python-casacore would
have produced.
"""

import numpy as np
from astropy import units as u
from astropy.wcs import WCS

# casacore direction-axis unit strings -> astropy units.
_UNIT_MAP = {
    "'": u.arcmin,
    "arcmin": u.arcmin,
    "''": u.arcsec,
    '"': u.arcsec,
    "arcsec": u.arcsec,
    "deg": u.deg,
    "rad": u.rad,
}


def _to_deg(value, unit_str):
    unit = _UNIT_MAP.get(unit_str, u.deg)
    return float((value * unit).to(u.deg).value)


def _fill_direction_poles(direction: dict) -> None:
    """Fill ``latpole``/``longpole`` (and float-ify ``crval``) in place via wcslib."""
    units = direction.get("units", ["deg", "deg"])
    crval = np.asarray(direction["crval"], dtype=float)
    cdelt = np.asarray(direction["cdelt"], dtype=float)
    crpix = np.asarray(direction["crpix"], dtype=float)
    proj = direction.get("projection", "SIN")

    crval_deg = [_to_deg(crval[0], units[0]), _to_deg(crval[1], units[1])]
    cdelt_deg = [_to_deg(cdelt[0], units[0]), _to_deg(cdelt[1], units[1])]

    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
    w.wcs.crval = crval_deg
    # astropy/FITS pixel indices are 1-based; casacore crpix is 0-based.
    w.wcs.crpix = [crpix[0] + 1.0, crpix[1] + 1.0]
    w.wcs.cdelt = cdelt_deg
    w.wcs.set()

    direction["latpole"] = float(w.wcs.latpole)
    direction["longpole"] = float(w.wcs.lonpole)
    # casacore reports crval as float; the keyword may store it as int.
    direction["crval"] = crval


_COORD_PREFIXES = ("direction", "spectral", "stokes", "linear", "tabular")


def postprocess_coords(coords: dict) -> dict:
    """Normalize a raw ``coords`` record toward python-casacore's reported form.

    * fills each direction coordinate's ``latpole``/``longpole`` via wcslib, and
    * drops the runtime-only ``_axes_sizes`` / ``_image_axes`` entries.

    The on-disk ``coords`` keyword carries ``_axes_sizes``/``_image_axes`` for
    some images (those written by xradio/CASA at runtime) but not others, while
    python-casacore reconstructs them on every read. xradio itself never reads
    them, so we drop them to give a single, consistent record shape regardless
    of how the image was produced. The record returned by ``getkeyword`` is a
    fresh copy, so mutating it here does not touch the on-disk table.
    """
    for key, value in coords.items():
        if not isinstance(value, dict):
            continue
        if key.startswith("direction") and {"crval", "cdelt", "crpix"} <= set(value):
            try:
                _fill_direction_poles(value)
            except Exception:
                # Leave the raw stored values untouched on any failure.
                pass
        if key.startswith(_COORD_PREFIXES):
            value.pop("_axes_sizes", None)
            value.pop("_image_axes", None)
        if key.startswith("spectral"):
            # Default frequency-conversion layer that casacore reconstructs on
            # read; present in some on-disk keywords but not in images written by
            # xradio. Not consumed by xradio, so drop it for a consistent shape.
            value.pop("conversion", None)
    return coords
