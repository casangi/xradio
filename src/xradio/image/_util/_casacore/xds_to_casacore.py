import os
import shutil
import time
from contextlib import ExitStack

import numpy as np
import xarray as xr
from astropy import units as apu

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from xradio._utils._casacore.tables import open_table_rw
from xradio._utils.logging import xradio_logger
from xradio.image._util import conventions
from xradio.image._util._blocks import RegionReader
from xradio.image._util._casacore.common import (
    _create_new_image,
    _image_flag,
    _object_name,
    _pointing_center,
)
from xradio.image._util.common import (
    _compute_sky_reference_pixel,
    _doppler_types,
    _get_unit,
    _linear_axis_reference_pixel,
    _time_coord_to_astropy,
)
from xradio.measurement_set._utils._utils.stokes_types import stokes_types

# Direction systems (casacore MDirection references) a CASA image can be
# written in, with the axis names casacore gives to their direction axes
_DIRECTION_AXES = {
    "ICRS": ["Right Ascension", "Declination"],
    "J2000": ["Right Ascension", "Declination"],
    "B1950": ["Right Ascension", "Declination"],
    "GALACTIC": ["Longitude", "Latitude"],
}

# Velocity doppler_type aliases of the casacore MDoppler types in _doppler_types
# (casacore: OPTICAL is Z, RELATIVISTIC is BETA)
_DOPPLER_TYPE_ALIASES = {"optical": "z", "true": "beta", "relativistic": "beta"}

# casacore epoch references without an astropy time scale; their time values
# are written as they are (e.g. casacore's unset observation date, 0 d LAST)
_SIDEREAL_EPOCH_REFS = ("LAST", "LMST", "GMST1", "GMST", "GAST", "UT2")

# casacore's spectral frame of images without a known frame, which casacore
# images can hold (the CASA reader keeps it) but no other format names
_UNDEFINED_SPECTRAL_FRAME = "Undefined"

# Polarization labels a casacore Stokes coordinate accepts
_CASACORE_STOKES = frozenset(stokes_types.values())

# All planes of an image share one beam (written as casacore's single
# restoring beam rather than per plane beams) when their beam parameters agree
# within this relative tolerance, as in the FITS writer; beams that agree to
# float32 precision (as stored in FITS BEAMS tables) count as one beam
_BEAM_RTOL = 1e-5

# The pixel pass computes blocks that are unions of whole chunks of the image
# and of its flag variable, so that every chunk is computed once. When the two
# chunk grids are so misaligned that a block would hold more than this many
# image chunks, the pass uses the image's chunk grid instead (flag chunks that
# span several image chunks are then computed once per image chunk), which
# bounds the memory use
_MAX_IMAGE_CHUNKS_PER_BLOCK = 8


def _first(value):
    """Return the first element of a list or tuple, else the value itself
    (units are sometimes stored as one element lists)."""
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return value


def _required(mapping: dict, key: str, where: str):
    """Return ``mapping[key]``, raising a ValueError naming the missing key."""
    if key not in mapping:
        raise ValueError(
            f"Cannot write the image to CASA: {where} has no '{key}' entry"
        )
    return mapping[key]


def _image_variable(xds: xr.Dataset) -> str:
    """Return the name of the variable written as the image pixels."""
    for name in ("SKY", "APERTURE"):
        if name in xds.data_vars:
            return name
    raise ValueError(
        "Cannot write the image to CASA: the dataset has no SKY or APERTURE "
        f"variable (it has {list(xds.data_vars)})"
    )


def _to_hz(value, units, what: str):
    """Convert a frequency value (or array) in units to Hz."""
    if units == "Hz":
        return value
    try:
        return (np.asarray(value) * apu.Unit(units)).to_value(apu.Hz)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot write the image to CASA: the {what} units {units!r} are not "
            "frequency units"
        ) from None


def _quantity_value_and_units(quantity, default_units: str) -> tuple[float, str]:
    """Return the value and units of a quantity dict (or of a bare number,
    which is taken to be in default_units)."""
    if isinstance(quantity, dict):
        value = quantity.get("data", quantity.get("value"))
        units = (quantity.get("attrs") or {}).get("units") or quantity.get("units")
    else:
        value, units = quantity, None
    return float(np.ravel(value)[0]), _first(units) or default_units


def _parse_equinox(equinox) -> tuple[str, float]:
    """
    Parse an equinox into its kind and year.

    Parameters
    ----------
    equinox : str or float
        For example ``"j2000.0"``, ``"B1950"`` or ``2000.0``.

    Returns
    -------
    tuple of (str, float)
        ``"j"`` (Julian) or ``"b"`` (Besselian), and the year. A bare year is
        Besselian before 1984 and Julian from 1984 on (the FITS convention).

    Raises
    ------
    ValueError
        If the equinox cannot be parsed.
    """
    text = str(equinox).strip().lower()
    kind = ""
    if text[:1] in ("j", "b"):
        kind, text = text[0], text[1:]
    try:
        year = float(text)
    except ValueError:
        raise ValueError(f"Cannot parse the equinox {equinox!r}") from None
    if not kind:
        kind = "b" if year < 1984.0 else "j"
    return kind, year


def _casacore_direction_system(reference_direction: dict) -> str:
    """
    Return the casacore direction system of an image's reference direction.

    The ``frame`` attribute is the authority (the equinox is optional);
    datasets without a frame fall back to the equinox, which is what older
    datasets carried.

    Parameters
    ----------
    reference_direction : dict
        The ``coordinate_system_info["reference_direction"]`` sky coordinate
        measure.

    Returns
    -------
    str
        ``"ICRS"``, ``"J2000"``, ``"B1950"`` or ``"GALACTIC"``.

    Raises
    ------
    ValueError
        If the frame (and equinox) has no casacore direction system that CASA
        images support.
    """
    attrs = reference_direction.get("attrs") or {}
    frame = attrs.get("frame")
    equinox = attrs.get("equinox") or None
    if not frame:
        if equinox is None:
            frame = "icrs"
        else:
            frame = "fk4" if _parse_equinox(equinox)[0] == "b" else "fk5"
    frame = str(frame).strip().lower()
    if frame == "icrs":
        # an equinox is meaningless for ICRS and is ignored
        return "ICRS"
    if frame == "galactic":
        return "GALACTIC"
    if frame in ("fk5", "fk4"):
        system, kind, year = (
            ("J2000", "j", 2000.0) if frame == "fk5" else ("B1950", "b", 1950.0)
        )
        if equinox is None or _parse_equinox(equinox) == (kind, year):
            return system
        raise ValueError(
            f"Cannot write a sky image in the {frame} frame with equinox "
            f"{equinox!r} to CASA: casacore supports {frame} only with equinox "
            f"{kind.upper()}{year:.0f} (direction system {system})"
        )
    raise ValueError(
        f"Cannot write a sky image in the {frame!r} direction frame to CASA: "
        "supported frames are icrs, fk5 (equinox J2000), fk4 (equinox B1950) "
        "and galactic"
    )


def _axis_increment_from_other(cdelt: dict, axis: str, other: str, sign) -> float:
    """Increment of a single pixel axis: the other axis' increment, with the
    given sign (None keeps the other axis' sign). Any value keeps the world
    coordinate of the single pixel; it only sets the pixel's extent."""
    if cdelt[other] is None:
        raise ValueError(
            f"Cannot write a 1 x 1 pixel image to CASA: the pixel increments "
            f"of the {axis} and {other} axes cannot be determined"
        )
    if sign is None:
        return cdelt[other]
    return sign * abs(cdelt[other])


def _cdelt_attr(coord: xr.DataArray, default_units: str) -> float | None:
    """The increment held by a coordinate's 'cdelt' attribute (a number in
    default_units or a quantity dict), None without the attribute."""
    cdelt = coord.attrs.get("cdelt")
    if cdelt is None:
        return None
    value, units = _quantity_value_and_units(cdelt, default_units)
    value = (value * apu.Unit(_get_unit(units))).to_value(apu.Unit(default_units))
    if not np.isfinite(value) or value == 0:
        raise ValueError(
            f"Cannot write the image to CASA: the {coord.name} coordinate's "
            f"cdelt attribute must be a non-zero finite increment, got {cdelt!r}"
        )
    return float(value)


def _sky_increment_and_reference_pixel(xds: xr.Dataset) -> tuple:
    """
    Compute the increments and reference pixels of the l and m axes.

    The reference pixel is where l (m) is 0, extrapolated when it lies outside
    the grid (an image cut out away from its reference direction).

    Parameters
    ----------
    xds : xr.Dataset
        Image dataset with l and m coordinates.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``cdelt`` and (0-based) ``crpix`` for (l, m), in radians and pixels.

    Raises
    ------
    ValueError
        If l or m is not uniformly spaced, or the image is 1 x 1 pixels without
        'cdelt' attributes on l and m.
    """
    cdelt = {}
    for c in ("l", "m"):
        cdelt[c] = conventions.linear_axis_increment(xds[c].values, c)
        if cdelt[c] is None:
            # a single value does not define the increment
            cdelt[c] = _cdelt_attr(xds[c], "rad")
    # Without a cdelt attribute a single pixel axis takes the other axis'
    # pixel size, with the usual signs (l decreases and m increases with the
    # pixel index)
    for c, other, sign in (("l", "m", -1.0), ("m", "l", 1.0)):
        if cdelt[c] is None:
            cdelt[c] = _axis_increment_from_other(cdelt, c, other, sign)
    increments = [cdelt["l"], cdelt["m"]]
    crpix = np.asarray(_compute_sky_reference_pixel(xds, cdelt=increments), float)
    return np.array(increments), crpix


def _compute_direction_dict(xds: xr.Dataset) -> dict:
    """
    Given xds metadata, compute the direction dict that is valid
    for a CASA image coordinate system
    """
    if "coordinate_system_info" not in xds.attrs:
        raise ValueError(
            "Writing a sky image to CASA requires the coordinate_system_info "
            "dataset attribute"
        )
    xds_dir = xds.attrs["coordinate_system_info"]
    where = "coordinate_system_info"
    ref_dir = _required(xds_dir, "reference_direction", where)
    system = _casacore_direction_system(ref_dir)
    ref_attrs = ref_dir.get("attrs") or {}
    units = ref_attrs.get("units", "rad")
    if isinstance(units, str):
        units = [units, units]
    crval = [
        (float(value) * apu.Unit(_get_unit(unit))).to_value(apu.rad)
        for value, unit in zip(
            _required(ref_dir, "data", "reference_direction"), units, strict=True
        )
    ]
    cdelt, crpix = _sky_increment_and_reference_pixel(xds)
    pole = _required(xds_dir, "native_pole_direction", where)
    pole_units = _get_unit(_first((pole.get("attrs") or {}).get("units", "rad")))
    direction = {}
    direction["_axes_sizes"] = np.array(
        [xds.sizes[dim] for dim in ("l", "m")], dtype=np.int32
    )
    direction["_image_axes"] = np.array([2, 3], dtype=np.int32)
    direction["system"] = system
    direction["projection"] = _required(xds_dir, "projection", where)
    direction["projection_parameters"] = np.asarray(
        _required(xds_dir, "projection_parameters", where), dtype=float
    )
    # crval and cdelt (from l and m) are both written in radians
    direction["units"] = ["rad", "rad"]
    direction["crval"] = np.array(crval)
    direction["cdelt"] = cdelt
    direction["crpix"] = crpix
    direction["pc"] = np.array(
        _required(xds_dir, "pixel_coordinate_transformation_matrix", where),
        dtype=float,
    )
    direction["axes"] = list(_DIRECTION_AXES[system])
    direction["conversionSystem"] = system
    for i, s in enumerate(["longpole", "latpole"]):
        # longpole, latpole are numerical values in degrees in casa images
        direction[s] = float(
            (float(pole["data"][i]) * apu.Unit(pole_units)).to_value(apu.deg)
        )
    return direction


def _linear_axis(xds: xr.Dataset, name: str) -> dict:
    """
    Describe a u or v axis by its reference value, increment and units.

    The ``crval``, ``cdelt`` and ``units`` attributes of the coordinate are
    used when present (also when nested in a quantity dict, as the factories
    write them); a missing increment, or one that does not match the
    coordinate values, is derived from the values, and a missing reference
    value is 0 (the uv origin).

    Parameters
    ----------
    xds : xr.Dataset
        Image dataset with u and v coordinates.
    name : str
        ``"u"`` or ``"v"``.

    Returns
    -------
    dict
        ``values``, ``crval``, ``cdelt`` (None for a single pixel axis without
        a ``cdelt`` attribute) and ``units``.
    """
    coord = xds.coords[name]
    attrs = coord.attrs
    values = np.asarray(coord.values, dtype=float)
    nested = attrs.get("attrs") if isinstance(attrs.get("attrs"), dict) else {}
    units = _first(attrs.get("units")) or _first(nested.get("units")) or "lambda"
    if attrs.get("crval") is not None:
        crval = float(attrs["crval"])
    elif attrs.get("data") is not None:
        crval = float(np.ravel(attrs["data"])[0])
    else:
        crval = 0.0
    cdelt_attr = attrs.get("cdelt")
    cdelt_attr = float(cdelt_attr) if cdelt_attr else None
    cdelt = conventions.linear_axis_increment(values, name)
    if cdelt is None:
        cdelt = cdelt_attr
    elif cdelt_attr is not None and np.isclose(
        cdelt_attr, cdelt, rtol=conventions.LINEAR_AXIS_TOLERANCE, atol=0.0
    ):
        # keep the exact increment of the source image
        cdelt = cdelt_attr
    return {"values": values, "crval": crval, "cdelt": cdelt, "units": units}


def _compute_linear_dict(xds: xr.Dataset) -> dict:
    axes = {name: _linear_axis(xds, name) for name in ("u", "v")}
    cdelt = {name: axes[name]["cdelt"] for name in ("u", "v")}
    for name, other in (("u", "v"), ("v", "u")):
        if cdelt[name] is None:
            cdelt[name] = _axis_increment_from_other(cdelt, name, other, None)
    linear = {}
    linear["crval"] = np.array([axes[name]["crval"] for name in ("u", "v")])
    linear["cdelt"] = np.array([cdelt["u"], cdelt["v"]])
    linear["axes"] = ["UU", "VV"]
    linear["units"] = [axes[name]["units"] for name in ("u", "v")]
    # The reference pixel is where the axis takes its reference value,
    # extrapolated when that lies outside the grid (np.interp would clamp it)
    linear["crpix"] = np.array(
        [
            _linear_axis_reference_pixel(
                axes[name]["values"], axes[name]["crval"], cdelt[name]
            )
            for name in ("u", "v")
        ],
        dtype=np.float64,
    )
    linear["pc"] = np.array([[1.0, 0.0], [0.0, 1.0]])
    return linear


def _casacore_spectral_system(freq_attrs: dict) -> str:
    """
    Return the casacore spectral reference frame of the frequency axis.

    The ``frame`` attribute of the frequency coordinate is the authority; the
    ``reference_frequency`` observer is the fallback (for datasets without a
    frame). FITS ``SPECSYS`` and schema observer names are translated (for
    example ``"BARYCENT"`` and ``"gcrs"``).

    Parameters
    ----------
    freq_attrs : dict
        Attributes of the frequency coordinate.

    Returns
    -------
    str
        The casacore frame, one of
        :data:`~xradio.image._util.conventions.CASACORE_SPECTRAL_FRAMES`, or
        ``"Undefined"`` for an image whose frame is casacore's undefined
        frame (as the CASA reader keeps it).

    Raises
    ------
    ValueError
        If there is no frame, or it has no casacore equivalent (for example
        the astropy only frames ``icrs``, ``hcrs`` and ``lsr``).
    """
    ref_attrs = (freq_attrs.get("reference_frequency") or {}).get("attrs") or {}
    frame = freq_attrs.get("frame")
    observer = ref_attrs.get("observer") or freq_attrs.get("observer")
    name = frame or observer
    if not name:
        raise ValueError(
            "Cannot write the image to CASA: the frequency coordinate has no "
            "spectral reference frame (its 'frame' attribute)"
        )
    if str(name).strip().lower() == _UNDEFINED_SPECTRAL_FRAME.lower():
        return _UNDEFINED_SPECTRAL_FRAME
    try:
        system = conventions.normalize_spectral_frame(name)
    except ValueError as exc:
        raise ValueError(f"Cannot write the image to CASA: {exc}") from None
    if frame and observer:
        try:
            observer_system = conventions.normalize_spectral_frame(observer)
        except ValueError:
            observer_system = None
        if observer_system != system:
            xradio_logger().warning(
                f"The frequency coordinate's frame {frame!r} and the "
                f"reference_frequency observer {observer!r} disagree; writing "
                f"the spectral reference frame {system} given by the frame"
            )
    return system


def _casacore_velocity_type(doppler_type) -> int:
    """
    Return the casacore MDoppler type (the ``velType``) of a doppler_type.

    Parameters
    ----------
    doppler_type : str
        ``radio``, ``z`` or ``optical``, ``ratio``, ``beta``, ``true`` or
        ``relativistic``, ``gamma`` (any case).

    Returns
    -------
    int
        RADIO 0, Z 1, RATIO 2, BETA 3, GAMMA 4.

    Raises
    ------
    ValueError
        For other doppler types.
    """
    name = str(doppler_type).strip().lower()
    name = _DOPPLER_TYPE_ALIASES.get(name, name)
    if name not in _doppler_types:
        supported = sorted(set(_doppler_types) | set(_DOPPLER_TYPE_ALIASES))
        raise ValueError(
            f"Cannot write the image to CASA: velocity doppler_type "
            f"{doppler_type!r} has no casacore equivalent (supported: "
            f"{', '.join(supported)})"
        )
    return _doppler_types.index(name)


def _velocity_type_and_unit(xds: xr.Dataset) -> tuple[int, str]:
    """Return the casacore velocity type and unit, radio and m/s when the
    dataset has no velocity coordinate (or its attributes are missing)."""
    attrs = xds.coords["velocity"].attrs if "velocity" in xds.coords else {}
    vel_type = _casacore_velocity_type(attrs.get("doppler_type") or "radio")
    vel_unit = _first(attrs.get("units")) or "m/s"
    try:
        apu.Unit(vel_unit).to(apu.m / apu.s)
    except (ValueError, TypeError):
        # e.g. 'ratio': casacore needs a velocity unit
        vel_unit = "m/s"
    return vel_type, vel_unit


def _single_channel_width(freq_attrs: dict) -> float:
    """Increment (Hz) of a single channel spectral axis: the frequency
    coordinate's channel_width, else the conventional fallback width."""
    width = freq_attrs.get("channel_width")
    if width is not None:
        value, units = _quantity_value_and_units(width, "Hz")
        value = float(_to_hz(value, units, "channel_width"))
        if np.isfinite(value) and value != 0:
            return value
    return conventions.SINGLE_CHANNEL_WIDTH_FALLBACK_HZ


def _compute_spectral_dict(xds: xr.Dataset) -> dict:
    """
    Given xds metadata, compute the spectral dict that is valid
    for a CASA image coordinate system
    """
    freq_attrs = xds.frequency.attrs
    reference = freq_attrs.get("reference_frequency") or {}
    # The spectral axis is written in Hz (the units of the image schema)
    freq_units = _first(freq_attrs.get("units")) or "Hz"
    values = _to_hz(
        np.asarray(xds.frequency.values, dtype=float), freq_units, "frequency"
    )
    spec = {}
    spec["_axes_sizes"] = np.array([xds.sizes["frequency"]], dtype=np.int32)
    spec["_image_axes"] = np.array([0], dtype=np.int32)
    spec["formatUnit"] = ""
    spec["name"] = "Frequency"
    # spec["nativeType"] = _native_types.index(xds.frequency.attrs["native_type"])
    # FREQ
    spec["nativeType"] = 0
    if "rest_frequency" in freq_attrs:
        rest, rest_units = _quantity_value_and_units(freq_attrs["rest_frequency"], "Hz")
        spec["restfreq"] = float(_to_hz(rest, rest_units, "rest_frequency"))
    else:
        xradio_logger().warning(
            "The frequency coordinate has no rest_frequency; writing a CASA "
            "image without a rest frequency"
        )
        spec["restfreq"] = 0.0
    spec["restfreqs"] = np.array([spec["restfreq"]])
    spec["system"] = _casacore_spectral_system(freq_attrs)
    spec["unit"] = "Hz"
    spec["velType"], spec["velUnit"] = _velocity_type_and_unit(xds)
    spec["version"] = 2
    spec["waveUnit"] = _first(freq_attrs.get("wave_units")) or "mm"
    wcs = {}
    wcs["ctype"] = "FREQ"
    wcs["pc"] = 1.0
    if "data" in reference:
        crval, crval_units = _quantity_value_and_units(reference, freq_units)
        wcs["crval"] = float(_to_hz(crval, crval_units, "reference_frequency"))
    else:
        wcs["crval"] = float(values[0])
    # raises for a frequency axis that is not uniformly spaced (a CASA image
    # describes it by one increment)
    cdelt = conventions.linear_axis_increment(values, "frequency")
    if cdelt is None:
        cdelt = _single_channel_width(freq_attrs)
    wcs["cdelt"] = cdelt
    wcs["crpix"] = float((wcs["crval"] - values[0]) / wcs["cdelt"])
    spec["wcs"] = wcs
    return spec


def _stokes_labels(xds: xr.Dataset) -> list[str]:
    """Return the polarization labels, raising for labels casacore does not
    know (casacore would only fail when the written image is opened)."""
    labels = [str(p) for p in xds.polarization.values]
    unknown = [p for p in labels if p not in _CASACORE_STOKES]
    if unknown:
        raise ValueError(
            f"Cannot write the image to CASA: polarizations {unknown} are not "
            f"casacore Stokes types"
        )
    return labels


def _obsdate_from_xds(xds: xr.Dataset) -> dict:
    """
    Return the casacore observation date (an MEpoch record) of the image.

    The time value is interpreted through its ``units``, ``format`` and
    ``scale`` attributes and written as MJD days in that scale.

    Raises
    ------
    ValueError
        If the time cannot be interpreted or its scale is not a casacore epoch
        reference.
    """
    time_coord = xds.coords["time"]
    attrs = time_coord.attrs
    value = np.ravel(time_coord.values)[0]
    scale = str(attrs.get("scale") or "utc").strip()
    refer = scale.upper()
    astropy_scale = conventions.CASACORE_EPOCH_REF_TO_SCALE.get(refer)
    if astropy_scale is not None:
        try:
            # datetime64 values are converted, not taken as numbers
            mjd = float(
                _time_coord_to_astropy(value, {**attrs, "scale": astropy_scale}).mjd
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot write the image to CASA: the time {value} (units "
                f"{attrs.get('units')!r}, format {attrs.get('format')!r}, scale "
                f"{scale!r}) cannot be converted to an observation date: {exc}"
            ) from exc
        return {"refer": refer, "type": "epoch", "m0": {"unit": "d", "value": mjd}}
    if refer in _SIDEREAL_EPOCH_REFS and np.asarray(value).dtype.kind in "iuf":
        # no astropy time scale: written as it is
        return {
            "refer": refer,
            "type": "epoch",
            "m0": {"unit": _first(attrs.get("units")) or "d", "value": float(value)},
        }
    raise ValueError(
        f"Cannot write the image to CASA: time scale {scale!r} is not a casacore "
        f"epoch reference"
    )


def _coord_dict_from_xds(xds: xr.Dataset) -> dict:
    coord = {}
    sky_ap = _image_variable(xds)
    tel = xds[sky_ap].attrs.get("telescope") or {}
    if "name" in tel:
        coord["telescope"] = tel["name"]
    # casacore positions have three components: the direction and the distance
    # (both optional in the schema)
    if "direction" in tel and "distance" in tel:
        xds_telloc = tel["direction"]
        telloc = {}
        telloc["refer"] = xds_telloc["attrs"]["frame"]
        if telloc["refer"] == "GRS80":
            telloc["refer"] = "ITRF"
        for i in range(2):
            telloc[f"m{i}"] = {
                "unit": xds_telloc["attrs"]["units"],
                "value": xds_telloc["data"][i],
            }
        telloc[f"m{2}"] = {
            "unit": tel["distance"]["attrs"]["units"],
            "value": tel["distance"]["data"][0],
        }

        telloc["type"] = "position"
        coord["telescopeposition"] = telloc

    # if "location" in tel:
    #     xds_telloc = tel["location"]
    #     telloc = {}
    #     telloc["refer"] = xds_telloc["attrs"]["frame"]
    #     if telloc["refer"] == "GRS80":
    #         telloc["refer"] = "ITRF"
    #     for i in range(3):
    #         telloc[f"m{i}"] = {
    #             "unit": xds_telloc["attrs"]["units"],
    #             "value": xds_telloc["data"][i],
    #         }
    #     telloc["type"] = "position"
    #     coord["telescopeposition"] = telloc
    if xds[sky_ap].attrs.get("observer") is not None:
        coord["observer"] = xds[sky_ap].attrs["observer"]
    coord["obsdate"] = _obsdate_from_xds(xds)
    if _pointing_center in xds[sky_ap].attrs:
        pointing_center = xds[sky_ap].attrs[_pointing_center]
        pc_units = (pointing_center.get("attrs") or {}).get("units", "rad")
        if isinstance(pc_units, str):
            pc_units = [pc_units, pc_units]
        coord["pointingcenter"] = {
            "initial": True,
            # casacore holds the pointing center in radians
            "value": np.array(
                [
                    (float(value) * apu.Unit(_get_unit(unit))).to_value(apu.rad)
                    for value, unit in zip(
                        pointing_center["data"], pc_units, strict=True
                    )
                ]
            ),
        }
    if _image_dims(xds[sky_ap], sky_ap)[-1] == "l":
        coord["direction0"] = _compute_direction_dict(xds)
    else:
        coord["linear0"] = _compute_linear_dict(xds)
    coord["stokes1"] = {
        "_axes_sizes": np.array([xds.sizes["polarization"]], dtype=np.int32),
        "_image_axes": np.array([1], dtype=np.int32),
        "axes": ["Stokes"],
        "cdelt": np.array([1.0]),
        "crpix": np.array([0.0]),
        "crval": np.array([1.0]),
        "pc": np.array([[1.0]]),
        "stokes": _stokes_labels(xds),
    }
    coord["spectral2"] = _compute_spectral_dict(xds)
    coord["pixelmap0"] = np.array([0, 1], dtype=np.int32)
    coord["pixelmap1"] = np.array([2], dtype=np.int32)
    coord["pixelmap2"] = np.array([3], dtype=np.int32)
    coord["pixelreplace0"] = np.array([0.0, 0.0])
    coord["pixelreplace1"] = np.array([0.0])
    coord["pixelreplace2"] = np.array([0.0])
    coord["worldmap0"] = np.array([0, 1], dtype=np.int32)
    coord["worldmap1"] = np.array([2], dtype=np.int32)
    coord["worldmap2"] = np.array([3], dtype=np.int32)
    # this probbably needs some verification
    coord["worldreplace0"] = [0.0, 0.0]
    coord["worldreplace1"] = np.atleast_1d(coord["stokes1"]["crval"])
    coord["worldreplace2"] = np.atleast_1d(coord["spectral2"]["wcs"]["crval"])

    return coord


def _history_from_xds(xds: xr.Dataset, image: str) -> None:
    """
    Write history from xds attributes to CASA image logtable.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset with potential history attribute (stored as dict).
    image : str
        Path to the CASA image.

    Notes
    -----
    History is stored in data variable attributes (e.g., SKY.attrs['history']),
    not in the main dataset attributes.
    """
    # Check if history exists in data variable attributes (SKY, APERTURE, etc.)
    history_dict = None
    for var_name in xds.data_vars:
        if "history" in xds[var_name].attrs:
            history_dict = xds[var_name].attrs["history"]
            break

    # Also check dataset-level attributes as a fallback
    if history_dict is None and "history" in xds.attrs:
        history_dict = xds.attrs["history"]

    if history_dict is None or not isinstance(history_dict, dict):
        return

    # Convert dict back to xr.Dataset to access data
    try:
        history_xds = xr.Dataset.from_dict(history_dict)
    except Exception:
        # If conversion fails, history dict may be malformed, skip writing
        return

    # Check for row in both dims and coords (it could be either depending on the xarray version)
    nrows = 0
    if "row" in history_xds.dims:
        nrows = history_xds.sizes["row"]
    elif "row" in history_xds.coords:
        nrows = len(history_xds.row)
    elif "row" in history_xds.data_vars:
        # row might also be a data variable
        nrows = len(history_xds.row)

    if nrows > 0:
        # TODO need to implement nrows == 0 case
        with open_table_rw(os.sep.join([image, "logtable"])) as tb:
            tb.addrows(nrows + 1)
            for c in ["TIME", "PRIORITY", "MESSAGE", "LOCATION", "OBJECT_ID"]:
                if c in history_xds.data_vars or c in history_xds.coords:
                    vals = history_xds[c].values
                    if c == "TIME":
                        k = time.time() + 40587 * 86400
                    elif c == "PRIORITY":
                        k = "INFO"
                    elif c == "MESSAGE":
                        k = (
                            "Wrote xds to "
                            + os.path.basename(image)
                            + " using cngi_io.xds_to_casa_image_2()"
                        )
                    elif c == "LOCATION":
                        k = "cngi_io.xds_to_casa_image_2"
                    elif c == "OBJECT_ID":
                        k = ""
                    vals = np.append(vals, k)
                    tb.putcol(c, vals)


def _beam_planes(xds: xr.Dataset) -> tuple[np.ndarray, str]:
    """
    Return the beams of every image plane and their units.

    Parameters
    ----------
    xds : xr.Dataset
        Image dataset with a ``BEAM_FIT_PARAMS`` variable.

    Returns
    -------
    tuple of (np.ndarray, str)
        Array of shape (frequency, polarization, 3) holding the major axis,
        minor axis and position angle, picked by their beam_params_label (not
        by position), and the units of the values.

    Raises
    ------
    ValueError
        If a beam parameter label is missing or the units are not angular.
    """
    beam = xds["BEAM_FIT_PARAMS"]
    label_dim = "beam_params_label"
    if label_dim in beam.coords:
        labels = [str(label) for label in beam.coords[label_dim].values]
    else:
        # without labels the values are in the schema order
        labels = ["major", "minor", "pa"]
    missing = [p for p in ("major", "minor", "pa") if p not in labels]
    if missing or label_dim not in beam.dims:
        raise ValueError(
            f"Cannot write the beams to CASA: the BEAM_FIT_PARAMS "
            f"{label_dim} {labels} lacks {missing or [label_dim]}"
        )
    if "time" in beam.dims:
        beam = beam.isel(time=0)
    for dim in ("frequency", "polarization"):
        if dim not in beam.dims:
            beam = beam.expand_dims({dim: xds.sizes[dim]})
    beam = beam.transpose("frequency", "polarization", label_dim)
    values = np.asarray(beam.values, dtype=float)
    values = values[..., [labels.index(p) for p in ("major", "minor", "pa")]]
    units = _first(beam.attrs.get("units")) or "rad"
    try:
        apu.Unit(_get_unit(units)).to(apu.rad)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot write the beams to CASA: BEAM_FIT_PARAMS units {units!r} "
            f"are not angular units"
        ) from None
    return values, units


def _check_beams(beams: np.ndarray) -> bool:
    """
    Check that casacore can open an image with these beams.

    casacore refuses to open an image with a null beam (a zero major or minor
    axis) or a beam whose major axis is shorter than its minor axis, also in
    a set of per plane beams.

    Parameters
    ----------
    beams : np.ndarray
        Beams of every plane, shape (frequency, polarization, 3).

    Returns
    -------
    bool
        False when every beam is zero (the image has no beam and none is
        written), True when every beam is valid.

    Raises
    ------
    ValueError
        If some beams are null or have their axes swapped.
    """
    major, minor = beams[..., 0], beams[..., 1]
    if not np.any(beams[..., :2]):
        return False
    null = (major == 0) | (minor == 0)
    if np.any(null):
        planes = [tuple(int(i) for i in index) for index in np.argwhere(null)]
        raise ValueError(
            "Cannot write the beams to CASA: the beams of the (frequency, "
            f"polarization) planes {planes[:5]}{' ...' if len(planes) > 5 else ''} "
            "have a zero axis while others do not, and casacore cannot open an "
            "image with null beams"
        )
    swapped = major < minor
    if np.any(swapped):
        planes = [tuple(int(i) for i in index) for index in np.argwhere(swapped)]
        raise ValueError(
            "Cannot write the beams to CASA: the major axis of the beams of the "
            f"(frequency, polarization) planes {planes[:5]}"
            f"{' ...' if len(planes) > 5 else ''} is shorter than their minor axis"
        )
    return True


def _is_single_beam(beams: np.ndarray, units: str) -> bool:
    """Whether every plane has the same beam (see _BEAM_RTOL)."""
    beams = beams * apu.Unit(_get_unit(units)).to(apu.rad)
    return bool(np.allclose(beams, beams[0, 0], rtol=_BEAM_RTOL, atol=0.0))


def _beam_record(beam: np.ndarray, units: str) -> dict:
    return {
        "major": {"unit": units, "value": float(beam[0])},
        "minor": {"unit": units, "value": float(beam[1])},
        "positionangle": {"unit": units, "value": float(beam[2])},
    }


def _imageinfo_dict_from_xds(xds: xr.Dataset) -> dict:
    ii = {}
    ap_sky = _image_variable(xds)
    # The casacore image type is stored in the sub_type attribute (the type
    # attribute holds the data group role, e.g. "sky"). casacore only
    # recognizes its own spelling (e.g. "Spectral Index") and turns anything
    # else into Intensity
    ii["imagetype"] = conventions.sub_type_to_casacore(
        xds[ap_sky].attrs.get("sub_type")
    )
    ii["objectname"] = xds[ap_sky].attrs.get(_object_name) or ""
    if "BEAM_FIT_PARAMS" in xds.data_vars:
        beams, units = _beam_planes(xds)
        if not _check_beams(beams):
            xradio_logger().info(
                "The beam fit parameters are all zero (no beam): writing the "
                "CASA image without a restoring beam"
            )
        elif _is_single_beam(beams, units):
            ii["restoringbeam"] = _beam_record(beams[0, 0], units)
        else:
            nchan, npol = beams.shape[:2]
            pp = {"nChannels": nchan, "nStokes": npol}
            for polarization in range(npol):
                for chan in range(nchan):
                    pp["*" + str(nchan * polarization + chan)] = _beam_record(
                        beams[chan, polarization], units
                    )
            ii["perplanebeams"] = pp
    return ii


def _iter_chunk_slices(chunk_bounds: tuple):
    """Yield (blc, slices) for every chunk of a 4-d chunk bounds spec."""
    loc0 = 0
    for i0 in chunk_bounds[0]:
        loc1 = 0
        for i1 in chunk_bounds[1]:
            loc2 = 0
            for i2 in chunk_bounds[2]:
                loc3 = 0
                for i3 in chunk_bounds[3]:
                    yield (
                        (loc0, loc1, loc2, loc3),
                        (
                            slice(loc0, loc0 + i0),
                            slice(loc1, loc1 + i1),
                            slice(loc2, loc2 + i2),
                            slice(loc3, loc3 + i3),
                        ),
                    )
                    loc3 += i3
                loc2 += i2
            loc1 += i1
        loc0 += i0


def _block_bounds(*arrays) -> tuple:
    """
    Compute the blocks in which the pixel passes read their arrays.

    Every block is a union of whole chunks of each dask backed array, so that
    reading the arrays block by block computes every chunk exactly once.
    numpy backed arrays do not constrain the blocks.

    Parameters
    ----------
    *arrays : xr.DataArray or None
        Arrays of the same shape (None entries are ignored).

    Returns
    -------
    tuple of tuple of int
        Block sizes along each axis, for :func:`_iter_chunk_slices`.
    """
    arrays = [a for a in arrays if a is not None]
    shape = arrays[0].shape
    grids = [a.chunks for a in arrays if a.chunks is not None]
    bounds = []
    for axis, size in enumerate(shape):
        edges = {0, size}
        if grids:
            edges = set.intersection(
                *(set(np.cumsum((0,) + tuple(g[axis])).tolist()) for g in grids)
            )
        edges = sorted(edges)
        bounds.append(
            tuple(int(b - a) for a, b in zip(edges[:-1], edges[1:], strict=False))
        )
    bounds = tuple(bounds)
    if len(grids) > 1 and arrays[0].chunks is not None:
        image_chunk = np.prod([max(c) for c in arrays[0].chunks])
        largest_block = np.prod([max(b) for b in bounds])
        if largest_block > _MAX_IMAGE_CHUNKS_PER_BLOCK * image_chunk:
            return tuple(tuple(c) for c in arrays[0].chunks)
    return bounds


def _iter_blocks(*arrays):
    """
    Yield the blocks of the pixel passes with their values.

    Parameters
    ----------
    *arrays : xr.DataArray or None
        Arrays of the same shape (None entries yield None), in the order of
        the CASA image axes reversed.

    Yields
    ------
    blc : tuple of int
        Bottom left corner of the block (in the arrays' axis order).
    values : list of np.ndarray or None
        The values of each array in the block. The blocks are computed in
        batches with the active dask scheduler, from the tasks they need only
        (see :class:`xradio.image._util._blocks.RegionReader`), so that work
        the arrays share is done once and a pass costs in proportion to the
        number of chunks.
    """
    regions = [slices for _, slices in _iter_chunk_slices(_block_bounds(*arrays))]
    for region, values in RegionReader(arrays).regions(regions):
        yield tuple(r.start for r in region), values


def _put_block(tb, values: np.ndarray, blc: tuple) -> None:
    # casacore needs native byte order (python-casacore refuses byte swapped
    # arrays, casatools writes them as they are, i.e. byte swapped) and
    # contiguous data (FITS backed data is big endian)
    if not values.dtype.isnative or not values.flags.c_contiguous:
        values = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("="))
    tb.putcellslice(
        tb.colnames()[0],
        0,
        values,
        blc,
        tuple(np.array(blc) + np.array(values.shape) - 1),
    )


def _image_dims(xda: xr.DataArray, name: str) -> tuple:
    """
    Return the dims of an image in the order of the CASA image axes reversed.

    Parameters
    ----------
    xda : xr.DataArray
        Image (or mask) variable with time, frequency, polarization and either
        l and m (sky images) or u and v (aperture images) dims.
    name : str
        Variable name for the error message.

    Returns
    -------
    tuple of str
        ``("frequency", "polarization", "m", "l")`` or
        ``("frequency", "polarization", "v", "u")``.

    Raises
    ------
    ValueError
        If the variable lacks the dims of a CASA image.
    """
    for dims in (
        ("frequency", "polarization", "m", "l"),
        ("frequency", "polarization", "v", "u"),
    ):
        if set(dims) | {"time"} == set(xda.dims):
            return dims
    raise ValueError(
        f"Cannot write {name} with dims {xda.dims} to a CASA image: it needs the "
        "dims (time, frequency, polarization, l, m) or (time, frequency, "
        "polarization, u, v)"
    )


def _casa_pixel_dtype(xda: xr.DataArray, name: str) -> np.dtype:
    """
    Return the (native byte order) pixel type a variable is written with.

    casacore images hold float32, float64, complex64 or complex128 pixels, so
    bool images (masks) are written as float32 (1.0 and 0.0) and integer
    images as float32 or, for more than 16 bits, float64.

    Raises
    ------
    TypeError
        For other data types.
    """
    dtype = xda.dtype
    if dtype.kind == "b" or (dtype.kind in "iu" and dtype.itemsize <= 2):
        return np.dtype(np.float32)
    if dtype.kind in "iu":
        return np.dtype(np.float64)
    if dtype.kind == "f":
        return np.dtype(np.float32 if dtype.itemsize <= 4 else np.float64)
    if dtype.kind == "c":
        return np.dtype(np.complex64 if dtype.itemsize <= 8 else np.complex128)
    raise TypeError(
        f"Cannot write {name} of data type {dtype} to a CASA image: casacore "
        "images hold float or complex pixels"
    )


def _flag_variable(xds: xr.Dataset, sky_ap: str) -> str:
    """Return the name of the image's flag variable ("" for none), raising
    when the image's flag attribute names a variable the dataset lacks."""
    flag = xds[sky_ap].attrs.get(_image_flag) or ""
    if flag and flag not in xds.data_vars:
        raise ValueError(
            f"Cannot write {sky_ap} to a CASA image: its '{_image_flag}' "
            f"attribute names the variable {flag!r}, which the dataset does not "
            "have (remove the attribute, or add the variable and its 'flag' "
            "role in the data group)"
        )
    return flag


def _mask_record(image_full_path: str, name: str, casa_image_shape) -> dict:
    """Return the entry of the image's "masks" keyword for mask table name."""
    return {
        "box": {
            "blc": np.array([1.0, 1.0, 1.0, 1.0]),
            "comment": "",
            "isRegion": 1,
            "name": "LCBox",
            "oneRel": True,
            "shape": np.array(casa_image_shape),
            "trc": np.array(casa_image_shape),
        },
        "comment": "",
        "isRegion": 1,
        "name": "LCPagedMask",
        "mask": f"Table: {os.sep.join([image_full_path, name])}",
    }


def _nan_mask_names(masks: list, flag: str) -> tuple[str, str]:
    """Reserve the names of the masks derived from nan pixels: the nan mask
    and, if there is a flag, the nans-or-flag mask."""
    nans_mask_name = "mask_xds_nans"
    i = 0
    while nans_mask_name in masks:
        nans_mask_name = f"mask_xds_nans{i}"
        i += 1
    nans_or_flag_mask_name = ""
    if flag:
        base = f"mask_xds_nans_or_{flag}"
        nans_or_flag_mask_name = base
        i = 0
        while (
            nans_or_flag_mask_name in masks or nans_or_flag_mask_name == nans_mask_name
        ):
            nans_or_flag_mask_name = f"{base}{i}"
            i += 1
    return nans_mask_name, nans_or_flag_mask_name


def _copy_table(source: str, target: str) -> None:
    tb = tables.table(source, ack=False)
    try:
        tb.copy(target, deep=True, valuecopy=True)
    finally:
        tb.close()


def _write_casa_data(xds: xr.Dataset, image_full_path: str) -> None:
    """
    Create a CASA image and write its pixels and masks.

    All metadata is validated before any table is created, so an image the
    CASA format cannot represent raises without leaving anything behind. The
    pixels, the flag and the masks derived from nan pixels are then written in
    one pass in which every chunk of (dask backed) data is computed once.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset of one image: a SKY or APERTURE variable, with optional
        BEAM_FIT_PARAMS and mask or flag variables (the image's ``flag``
        attribute names its flag variable).
    image_full_path : str
        Path of the CASA image to create.
    """
    sky_ap = _image_variable(xds)
    image = xds[sky_ap]
    if image.sizes.get("time") != 1:
        raise RuntimeError("XDS can only be converted if it has exactly one time plane")
    trans_coords = _image_dims(image, sky_ap)
    pixel_dtype = _casa_pixel_dtype(image, sky_ap)
    flag = _flag_variable(xds, sky_ap)
    # Variables written as internal masks of the image: the flag and any other
    # mask or flag variables, but never the variable written as the image
    # itself (e.g. a deconvolution mask written as the pixels of <name>.mask)
    masks = [
        m
        for m in xds.data_vars
        if m != sky_ap and xds[m].attrs.get("type") in ("mask", "flag")
    ]
    if flag and flag not in masks:
        masks.append(flag)
    for m in masks:
        if set(xds[m].dims) != set(image.dims):
            raise ValueError(
                f"Cannot write {m} as a mask of {sky_ap}: its dims {xds[m].dims} "
                f"differ from the image dims {image.dims}"
            )
    # Raises for metadata the CASA image cannot represent, before any table is
    # created (the coordinates and image info are written once the pixels are)
    _coord_dict_from_xds(xds)
    _imageinfo_dict_from_xds(xds)

    image_t = image.isel(time=0).transpose(*trans_coords)
    casa_image_shape = image_t.shape[::-1]
    flag_t = xds[flag].isel(time=0).transpose(*trans_coords) if flag else None
    # only floating point pixels can be nan
    track_nans = image.dtype.kind in "fc"
    nans_mask_name, nans_or_flag_mask_name = _nan_mask_names(masks, flag)

    # Create the image with a mask table that serves as the donor for any
    # other mask tables: the flag mask if there is a flag, else the nan mask
    # (which is removed again below if it turns out not to be needed)
    creation_mask = flag if flag else nans_mask_name
    _write_initial_image(
        xds, image_full_path, creation_mask, casa_image_shape[::-1], pixel_dtype
    )
    nan_tables = []
    if track_nans:
        nan_tables = [nans_mask_name] + ([nans_or_flag_mask_name] if flag else [])

    # Single fused pass over the (possibly dask backed) data: the image (and
    # its flag) is read in blocks of whole chunks, so that every chunk is
    # computed once, and each block of pixels and flag mask is written right
    # away. The nan masks are written from the first block that holds a nan
    # on (images without nans pay nothing for them), after filling in the
    # blocks written before it, which hold no nans
    do_mask_nans = False
    nans_tb = None
    nans_or_tb = None
    written_without_nans = []
    with ExitStack() as stack:

        def open_rw(name=None):
            path = (
                image_full_path
                if name is None
                else os.sep.join([image_full_path, name])
            )
            return stack.enter_context(open_table_rw(path))

        image_tb = open_rw()
        flag_tb = open_rw(flag) if flag else None
        for blc, (values, flag_values) in _iter_blocks(image_t, flag_t):
            pixels = values.astype(pixel_dtype, copy=False)
            _put_block(image_tb, pixels, blc)
            not_flagged = None
            if flag_tb is not None:
                # casacore masks are True for good pixels
                not_flagged = np.logical_not(flag_values)
                _put_block(flag_tb, not_flagged, blc)
            if not track_nans:
                continue
            nan = np.isnan(pixels)
            if nans_tb is None:
                if not nan.any():
                    written_without_nans.append((blc, pixels.shape))
                    continue
                if flag:
                    # Copies of the flag mask: the nans-or-flag mask then
                    # holds its final values in the blocks written so far
                    flag_tb.flush()
                    for name in nan_tables:
                        _copy_table(
                            os.sep.join([image_full_path, flag]),
                            os.sep.join([image_full_path, name]),
                        )
                    nans_or_tb = open_rw(nans_or_flag_mask_name)
                nans_tb = open_rw(nans_mask_name)
                for done_blc, done_shape in written_without_nans:
                    _put_block(nans_tb, np.ones(done_shape, dtype=bool), done_blc)
                written_without_nans = []
            not_nan = np.logical_not(nan)
            _put_block(nans_tb, not_nan, blc)
            if nans_or_tb is None:
                do_mask_nans = do_mask_nans or bool(nan.any())
            else:
                _put_block(nans_or_tb, np.logical_and(not_nan, not_flagged), blc)
                # nans that are already flagged do not need an extra mask
                do_mask_nans = do_mask_nans or bool(
                    np.logical_and(nan, not_flagged).any()
                )
    nans_written = nans_tb is not None

    # Any mask variables other than the flag (each is a single pass over its
    # own chunk grid)
    for m in masks:
        if m != flag:
            _write_pixels(m, creation_mask, image_full_path, xds)

    default_mask = flag
    if do_mask_nans:
        masks.extend(nan_tables)
        default_mask = nan_tables[-1]
    else:
        # The nan masks are not needed: remove them (and, without a flag, the
        # mask the image was created with, restoring the maskless state)
        if flag:
            removed = nan_tables if nans_written else []
        else:
            removed = [nans_mask_name]
        for name in removed:
            shutil.rmtree(os.sep.join([image_full_path, name]))
        if not flag:
            with open_table_rw(image_full_path) as tb:
                tb.removekeyword("masks")
                tb.removekeyword("Image_defaultmask")

    if masks:
        # each entry is its own record pointing at its own table
        masks_rec = {
            name: _mask_record(image_full_path, name, casa_image_shape)
            for name in masks
        }
        with open_table_rw(image_full_path) as tb:
            tb.putkeyword("masks", masks_rec)
            tb.putkeyword("Image_defaultmask", default_mask)


def _write_initial_image(
    xds: xr.Dataset,
    imagename: str,
    maskname: str,
    image_shape: tuple,
    dtype: np.dtype | None = None,
) -> None:
    if not maskname:
        maskname = ""
    if dtype is None:
        sky_ap = _image_variable(xds)
        dtype = _casa_pixel_dtype(xds[sky_ap], sky_ap)
    # only the type of value matters (it selects the casacore image pixel
    # type); a numpy scalar keeps single precision complex as Complex (a
    # python complex would give DComplex)
    value = "default" if dtype == np.float32 else np.zeros((), dtype=dtype)[()]
    image_full_path = os.path.expanduser(imagename)
    with _create_new_image(
        image_full_path, mask=maskname, shape=image_shape, value=value
    ):
        # just create the image, don't do anything with it
        pass


def _write_pixels(
    v: str,
    active_mask: str,
    image_full_path: str,
    xds: xr.Dataset,
    value: xr.DataArray = None,
) -> None:
    """
    Write a variable's pixels, block by block of its own chunk grid.

    Parameters
    ----------
    v : str
        Variable name. SKY and APERTURE are written as the image pixels; any
        other variable is a flag (True means bad) written, inverted, as the
        internal mask table of that name.
    active_mask : str
        Existing mask table of the image that is copied to create the mask
        table of a new mask.
    image_full_path : str
        Path of the CASA image.
    xds : xr.Dataset
        Dataset holding the variable.
    value : xr.DataArray, optional
        Array to write when v is not a variable of xds.
    """
    flip = v not in ("SKY", "APERTURE")
    if flip:
        filename = os.sep.join([image_full_path, v])
        if not os.path.exists(filename):
            _copy_table(os.sep.join([image_full_path, active_mask]), filename)
    else:
        filename = image_full_path
    arr = xds[v] if v in xds.data_vars else value
    arr_t = arr.isel(time=0).transpose(*_image_dims(arr, v))
    pixel_dtype = None if flip else _casa_pixel_dtype(arr, v)
    with open_table_rw(filename) as tb:
        for blc, (block,) in _iter_blocks(arr_t):
            block = np.logical_not(block) if flip else block.astype(pixel_dtype)
            _put_block(tb, block, blc)
