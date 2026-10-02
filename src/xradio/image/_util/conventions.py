"""
Naming conventions shared by the image readers, writers and factories.

Each convention has one translation table here, so that the CASA and FITS
readers and writers and the ``make_empty_*`` factories translate names the
same way:

* spectral reference frames: casacore names (held by the frequency coordinate's
  ``frame`` attribute), FITS ``SPECSYS`` values and the schema's
  ``reference_frequency`` observer vocabulary,
* polarization labels and FITS ``STOKES`` axis codes,
* casacore image types (the sky image ``sub_type`` attribute and FITS
  ``BTYPE``),
* time coordinate values and their ``units`` / ``format`` / ``scale`` attributes.
"""

from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------
# Spectral reference frames
# ---------------------------------------------------------------------------

#: casacore ``MFrequency`` reference frames, the canonical names held by the
#: frequency coordinate's ``frame`` attribute.
CASACORE_SPECTRAL_FRAMES = (
    "REST",
    "LSRK",
    "LSRD",
    "BARY",
    "GEO",
    "TOPO",
    "GALACTO",
    "LGROUP",
    "CMB",
)

#: casacore frame -> FITS ``SPECSYS`` (FITS WCS Paper III, as written by
#: casacore's own FITS export).
CASACORE_TO_FITS_SPECSYS = {
    "REST": "SOURCE",
    "LSRK": "LSRK",
    "LSRD": "LSRD",
    "BARY": "BARYCENT",
    "GEO": "GEOCENTR",
    "TOPO": "TOPOCENT",
    "GALACTO": "GALACTOC",
    "LGROUP": "LOCALGRP",
    "CMB": "CMBDIPOL",
}

#: casacore frame -> ``reference_frequency`` observer, following the shared
#: ``AllowedSpectralCoordFrames`` vocabulary: frames with an astropy
#: equivalent use the lowercase astropy name, the others keep the casacore name.
CASACORE_TO_OBSERVER = {
    "REST": "REST",
    "LSRK": "lsrk",
    "LSRD": "lsrd",
    "BARY": "BARY",
    "GEO": "gcrs",
    "TOPO": "TOPO",
    "GALACTO": "GALACTO",
    "LGROUP": "LGROUP",
    "CMB": "CMB",
}

# Lowercase alias -> casacore frame: casacore names, FITS SPECSYS values and
# observer names, in any case. HELIOCEN is treated as barycentric, as casacore
# does when reading FITS.
_SPECTRAL_FRAME_ALIASES = {
    **{frame.lower(): frame for frame in CASACORE_SPECTRAL_FRAMES},
    **{specsys.lower(): frame for frame, specsys in CASACORE_TO_FITS_SPECSYS.items()},
    **{observer.lower(): frame for frame, observer in CASACORE_TO_OBSERVER.items()},
    "heliocen": "BARY",
}


def normalize_spectral_frame(name: str) -> str:
    """Return the casacore spectral reference frame for a frame name.

    Parameters
    ----------
    name : str
        A casacore frame (``"BARY"``), a FITS ``SPECSYS`` value
        (``"BARYCENT"``) or a ``reference_frequency`` observer (``"lsrk"``,
        ``"gcrs"``), in any case.

    Returns
    -------
    str
        The casacore frame name, one of :data:`CASACORE_SPECTRAL_FRAMES`.

    Raises
    ------
    ValueError
        If the frame has no casacore equivalent, for example the astropy-only
        observers ``"icrs"``, ``"hcrs"`` and ``"lsr"``.
    """
    try:
        return _SPECTRAL_FRAME_ALIASES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"Spectral reference frame {name!r} has no casacore equivalent; "
            f"supported frames are {', '.join(CASACORE_SPECTRAL_FRAMES)} (or "
            f"their FITS SPECSYS names {', '.join(CASACORE_TO_FITS_SPECSYS.values())})"
        ) from None


def spectral_frame_to_observer(name: str) -> str:
    """Return the ``reference_frequency`` observer for a spectral frame name.

    Parameters
    ----------
    name : str
        Any name accepted by :func:`normalize_spectral_frame`.

    Returns
    -------
    str
        The observer, a value of the schema's ``AllowedSpectralCoordFrames``.
    """
    return CASACORE_TO_OBSERVER[normalize_spectral_frame(name)]


def spectral_frame_to_fits_specsys(name: str) -> str:
    """Return the FITS ``SPECSYS`` value for a spectral frame name.

    Parameters
    ----------
    name : str
        Any name accepted by :func:`normalize_spectral_frame`.

    Returns
    -------
    str
        The FITS WCS Paper III ``SPECSYS`` value.
    """
    return CASACORE_TO_FITS_SPECSYS[normalize_spectral_frame(name)]


# ---------------------------------------------------------------------------
# Polarization labels and FITS STOKES codes
# ---------------------------------------------------------------------------

#: Polarization label -> FITS ``STOKES`` axis code (FITS standard, Greisen and
#: Calabretta 2002, Table 7). These differ from the casacore ``Stokes`` enum
#: for the correlation products.
FITS_STOKES_CODES = {
    "I": 1,
    "Q": 2,
    "U": 3,
    "V": 4,
    "RR": -1,
    "LL": -2,
    "RL": -3,
    "LR": -4,
    "XX": -5,
    "YY": -6,
    "XY": -7,
    "YX": -8,
}

#: FITS ``STOKES`` axis code -> polarization label.
FITS_STOKES_LABELS = {code: label for label, code in FITS_STOKES_CODES.items()}

#: Canonical in-memory polarization order: the casacore ``Stokes`` enum order,
#: which keeps the correlations of a pair of feeds in Jones (2x2 matrix) order,
#: ``RR, RL, LR, LL`` and ``XX, XY, YX, YY``, after ``I, Q, U, V``.
CANONICAL_POLARIZATION_ORDER = (
    "I",
    "Q",
    "U",
    "V",
    "RR",
    "RL",
    "LR",
    "LL",
    "XX",
    "XY",
    "YX",
    "YY",
    "RX",
    "RY",
    "LX",
    "LY",
    "XR",
    "XL",
    "YR",
    "YL",
    "PP",
    "PQ",
    "QP",
    "QQ",
)

_CANONICAL_POLARIZATION_INDEX = {
    label: index for index, label in enumerate(CANONICAL_POLARIZATION_ORDER)
}


def canonical_polarization_order(labels) -> list[int]:
    """Return the permutation that puts polarization labels in canonical order.

    Images keep their polarization axis in the casacore ``Stokes`` order
    (:data:`CANONICAL_POLARIZATION_ORDER`), so that the correlations of a feed
    pair are in Jones matrix order (``RR, RL, LR, LL``; ``XX, XY, YX, YY``).
    FITS cannot store that order for correlations (see
    :func:`fits_stokes_axis`), so FITS readers use this permutation to restore
    it, reordering every variable with a polarization dimension the same way.

    Parameters
    ----------
    labels : sequence of str
        Polarization labels in their current order.

    Returns
    -------
    list of int
        ``order`` such that ``labels[order]`` is in canonical order (the
        identity when the labels already are). Labels outside
        :data:`CANONICAL_POLARIZATION_ORDER` keep their relative order after
        the known ones.
    """
    labels = [str(label) for label in labels]
    unknown = len(CANONICAL_POLARIZATION_ORDER)
    return sorted(
        range(len(labels)),
        key=lambda i: (_CANONICAL_POLARIZATION_INDEX.get(labels[i], unknown), i),
    )


def fits_stokes_axis(labels) -> tuple[list[int], int, int]:
    """Lay out polarization labels as a linear FITS ``STOKES`` axis.

    A FITS axis is linear (``CRVAL3 + CDELT3 * (pixel - CRPIX3)``), so the
    codes of the planes must form an arithmetic progression. The canonical
    correlation orders, for example ``[RR, RL, LR, LL]`` (codes -1, -3, -4,
    -2), do not, but they do once sorted (``[RR, LL, RL, LR]``), so FITS
    writers reorder the planes when that is needed, and FITS readers restore
    the canonical order with :func:`canonical_polarization_order`.

    Parameters
    ----------
    labels : sequence of str
        Polarization labels, in the order of the image's polarization axis.

    Returns
    -------
    order : list of int
        Permutation of the polarization axis such that ``labels[order]`` is in
        FITS axis order (the identity when no reordering is needed).
    crval : int
        FITS code of the first plane in ``order``.
    cdelt : int
        Code increment between consecutive planes (1 for a single plane).

    Raises
    ------
    ValueError
        If a label has no FITS code, or no order of the codes is an
        arithmetic progression (for example ``[I, Q, V]``).
    """
    labels = [str(label) for label in labels]
    unknown = [label for label in labels if label not in FITS_STOKES_CODES]
    if unknown:
        raise ValueError(
            f"Cannot write polarizations {labels} to FITS: {unknown} have no "
            f"FITS STOKES code (supported: {', '.join(FITS_STOKES_CODES)})"
        )
    codes = [FITS_STOKES_CODES[label] for label in labels]
    if len(codes) == 1:
        return [0], codes[0], 1

    def _is_progression(values):
        step = values[1] - values[0]
        return step != 0 and all(
            b - a == step for a, b in zip(values, values[1:], strict=False)
        )

    if _is_progression(codes):
        return list(range(len(codes))), codes[0], codes[1] - codes[0]
    # Stokes ascending (1, 2, ...) and correlations descending (-1, -2, ...),
    # as casacore's FITS export lays them out
    for descending in (codes[0] < 0, codes[0] >= 0):
        order = sorted(range(len(codes)), key=codes.__getitem__, reverse=descending)
        ordered = [codes[i] for i in order]
        if _is_progression(ordered):
            return order, ordered[0], ordered[1] - ordered[0]
    raise ValueError(
        f"Cannot write polarizations {labels} to FITS: their STOKES codes "
        f"{codes} cannot be laid out as a linear axis in any order"
    )


def fits_stokes_labels(crval: float, cdelt: float, crpix: float, n: int) -> list[str]:
    """Return the polarization labels of a FITS ``STOKES`` axis.

    Parameters
    ----------
    crval, cdelt, crpix : float
        The axis' ``CRVAL``, ``CDELT`` and (1-based) ``CRPIX`` values.
    n : int
        Number of pixels along the axis.

    Returns
    -------
    list of str
        One label per pixel.

    Raises
    ------
    ValueError
        If a pixel's code is not a FITS Stokes or correlation code (for
        example the AIPS polarized-intensity codes 5 to 8).
    """
    labels = []
    for pixel in range(1, n + 1):
        code = int(round(crval + cdelt * (pixel - crpix)))
        if code not in FITS_STOKES_LABELS:
            raise ValueError(
                f"FITS STOKES axis value {code} (pixel {pixel}) is not a "
                f"supported Stokes or correlation code ({sorted(FITS_STOKES_LABELS)})"
            )
        labels.append(FITS_STOKES_LABELS[code])
    return labels


# ---------------------------------------------------------------------------
# casacore image types (sky image sub_type, FITS BTYPE)
# ---------------------------------------------------------------------------

#: casacore ``ImageInfo`` image types, in casacore's spelling.
CASACORE_IMAGE_TYPES = (
    "Undefined",
    "Intensity",
    "Beam",
    "Column Density",
    "Depolarization Ratio",
    "Kinetic Temperature",
    "Magnetic Field",
    "Optical Depth",
    "Rotation Measure",
    "Rotational Temperature",
    "Spectral Index",
    "Velocity",
    "Velocity Dispersion",
)


def _image_type_key(name) -> str:
    return re.sub(r"[\s_\-]", "", str(name)).lower()


_IMAGE_TYPE_BY_KEY = {_image_type_key(name): name for name in CASACORE_IMAGE_TYPES}


def normalize_sub_type(native) -> str | None:
    """Return the schema ``sub_type`` for a casacore image type or FITS BTYPE.

    Matching is case insensitive and ignores spaces, underscores and hyphens,
    as casacore's own parsing does, so ``"Spectral Index"``,
    ``"spectral_index"`` and ``"SPECTRALINDEX"`` all give ``"SpectralIndex"``.

    Parameters
    ----------
    native : str or None
        The casacore ``imagetype`` or FITS ``BTYPE`` value.

    Returns
    -------
    str or None
        The casacore type name without spaces, or ``None`` when the value is
        empty, ``"Undefined"`` or not a casacore image type.
    """
    if not native:
        return None
    name = _IMAGE_TYPE_BY_KEY.get(_image_type_key(native))
    if name is None or name == "Undefined":
        return None
    return name.replace(" ", "")


def sub_type_to_casacore(sub_type) -> str:
    """Return casacore's spelling of a schema ``sub_type``.

    casacore only recognizes its own spelling (``"Spectral Index"``) and maps
    any other string to ``"Intensity"``, so writers must translate back.

    Parameters
    ----------
    sub_type : str or None
        The sky image ``sub_type`` attribute.

    Returns
    -------
    str
        The casacore image type (with spaces), or ``""`` when ``sub_type`` is
        empty or unknown.
    """
    if not sub_type:
        return ""
    return _IMAGE_TYPE_BY_KEY.get(_image_type_key(sub_type), "")


# ---------------------------------------------------------------------------
# Linear axes
# ---------------------------------------------------------------------------

#: Increment written for a single-channel image whose frequency coordinate has
#: no ``channel_width`` attribute (the CASA writer's long-standing default; the
#: FITS writer uses it too so that both writers agree).
SINGLE_CHANNEL_WIDTH_FALLBACK_HZ = 1.8e9

#: Largest deviation from a linear axis, as a fraction of the increment, that
#: writers accept. Axes derived from an optical velocity axis deviate by about
#: 5e-5 of a channel for typical cubes; gaps between spectral windows or a
#: channel subset deviate by whole channels.
LINEAR_AXIS_TOLERANCE = 1e-3


def linear_axis_increment(values, name: str) -> float | None:
    """Return the increment of a linear coordinate axis.

    CASA and FITS images describe each axis by a reference value and a
    constant increment, so a writer must refuse a coordinate that is not
    uniformly spaced instead of writing wrong world coordinates.

    Parameters
    ----------
    values : array_like
        Coordinate values along the axis.
    name : str
        Axis name for the error message.

    Returns
    -------
    float or None
        ``(values[-1] - values[0]) / (n - 1)``, or ``None`` for an axis with a
        single value.

    Raises
    ------
    ValueError
        If the values deviate from a linear axis by more than
        :data:`LINEAR_AXIS_TOLERANCE` increments, or are constant.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    step = (values[-1] - values[0]) / (values.size - 1)
    if step == 0:
        raise ValueError(f"The {name} coordinate is constant; it is not a linear axis")
    deviation = np.abs(values - (values[0] + step * np.arange(values.size))).max()
    if deviation > LINEAR_AXIS_TOLERANCE * abs(step):
        raise ValueError(
            f"The {name} coordinate is not uniformly spaced: its values deviate "
            f"from a linear axis with increment {step:g} by up to {deviation:g} "
            f"({deviation / abs(step):.3g} increments). CASA and FITS images "
            f"need a linear {name} axis; write a uniformly spaced subset (for "
            f"example one spectral window at a time) instead."
        )
    return float(step)


# ---------------------------------------------------------------------------
# Time coordinates
# ---------------------------------------------------------------------------

#: casacore ``MEpoch`` reference -> astropy time scale. Sidereal references
#: (LAST, LMST, GMST1, GAST) have no astropy scale and are absent.
CASACORE_EPOCH_REF_TO_SCALE = {
    "UTC": "utc",
    "TAI": "tai",
    "IAT": "tai",
    "TT": "tt",
    "TDT": "tt",
    "ET": "tt",
    "TDB": "tdb",
    "TCB": "tcb",
    "TCG": "tcg",
    "UT1": "ut1",
    "UT": "ut1",
}


def time_values_to_astropy(values, attrs: dict):
    """Interpret time values through their ``units``, ``format`` and ``scale``.

    The values are taken as a quantity in ``units`` and then read in
    ``format``, as the schema's time measure documentation prescribes, so for
    example seconds with format ``"mjd"`` and days with format ``"unix"`` are
    both converted correctly.

    Parameters
    ----------
    values : float or array_like
        Time values.
    attrs : dict
        Time measure attributes with ``units`` (default ``"d"``), ``format``
        (default ``"mjd"``) and ``scale`` (default ``"utc"``).

    Returns
    -------
    astropy.time.Time
        The times.

    Raises
    ------
    ValueError
        If the values cannot be interpreted with the given attributes.
    """
    from astropy import units as u
    from astropy.time import Time

    units = attrs.get("units") or "d"
    if isinstance(units, list | tuple):
        units = units[0]
    time_format = (attrs.get("format") or "mjd").lower()
    scale = (attrs.get("scale") or "utc").lower()
    quantity = np.asarray(values, dtype=float) * u.Unit(units)
    return Time(quantity, format=time_format, scale=scale)
