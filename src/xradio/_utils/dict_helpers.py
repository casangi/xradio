from xradio._utils.list_and_array import to_python_type


def make_quantity(value, units: str, dims: list | None = None) -> dict:
    """
    Create a serialized quantity dictionary.

    Parameters
    ----------
    value : numeric or array of numerics
        Quantity value.
    units : str
        Quantity units.
    dims : list or None, default=None
        Dimension labels associated with the quantity data.

    Returns
    -------
    dict
        Quantity dictionary containing ``data``, ``dims``, and ``attrs``.
    """
    normalized_dims = [] if dims is None else list(dims)
    return {
        "data": to_python_type(value),
        "dims": normalized_dims,
        "attrs": make_quantity_attrs(units),
    }


def ensure_units_are_consistent(units):
    """
    Normalize unit input to a single unit string.

    Parameters
    ----------
    units : str or Sequence[str]
        Either a single unit string or a sequence of unit strings that must all
        be identical.

    Returns
    -------
    str
        The normalized unit string.
    """
    if isinstance(units, str):
        return units

    if len(units) == 0:
        raise ValueError("Units are empty; expected at least one unit value.")

    u0 = units[0]
    for u in units:
        if u0 != u:
            raise ValueError(f"Units are not consistent: {u0} != {u}.")
    return u0


def make_quantity_attrs(units: str) -> dict:
    """
    Create attrs for a serialized quantity.

    Parameters
    ----------
    units : str
        Quantity units.

    Returns
    -------
    dict
        Quantity attrs dictionary with ``units`` and ``type`` keys.
    """
    return {"units": ensure_units_are_consistent(units), "type": "quantity"}


def make_spectral_coord_reference_dict(
    value: float, units: str, observer: str = "lsrk"
) -> dict:
    """
    Create a serialized spectral-coordinate measure dictionary.

    Parameters
    ----------
    value : numeric or array of numerics
        Spectral coordinate value.
    units : str
        Spectral coordinate units.
    observer : str, default="lsrk"
        Spectral reference frame.

    Returns
    -------
    dict
        Spectral-coordinate measure dictionary with attrs and scalar data.
    """
    u = ensure_units_are_consistent(units)
    return {
        "attrs": make_spectral_coord_measure_attrs(
            u,
            observer.lower() if observer not in ["TOPO", "BARY", "REST"] else observer,
        ),
        "data": to_python_type(value),
        "dims": [],
    }


def make_spectral_coord_measure_attrs(units: str, observer: str = "lsrk") -> dict:
    """
    Create attrs for a serialized spectral-coordinate measure.

    Parameters
    ----------
    units : str or list[str]
        Spectral coordinate units.
    observer : str, default="lsrk"
        Spectral reference frame.

    Returns
    -------
    dict
        Attrs dictionary for a spectral-coordinate measure.
    """
    u = ensure_units_are_consistent(units)
    return {"units": u, "observer": observer, "type": "spectral_coord"}


def _default_sky_axis_labels(frame: str) -> tuple[str, str]:
    """
    Choose default sky-axis labels for a direction frame.

    Parameters
    ----------
    frame : str
        Direction frame name.

    Returns
    -------
    tuple[str, str]
        Default axis labels for the provided frame.
    """
    return ("lon", "lat") if frame.lower() == "galactic" else ("ra", "dec")


def make_skycoord_dict(
    data: list[float],
    units: str,
    frame: str,
    axis_labels: tuple[str, str] | None = None,
) -> dict:
    """
    Build a serialized sky-coordinate measure dictionary.

    Parameters
    ----------
    data : list[float]
        Two-element direction coordinate value in the requested frame.
    units : str
        Units for both coordinate values.
    frame : str
        Direction reference frame name.
    axis_labels : tuple[str, str] or None, default=None
        Axis labels stored in the ``sky_dir_label`` coordinate. If not given,
        labels are derived from ``frame`` (for example, Galactic uses ``lon/lat``).

    Returns
    -------
    dict
        Dictionary with ``attrs``, ``data``, ``dims``, and ``coords`` fields
        representing a sky-coordinate measure.
    """
    labels = _default_sky_axis_labels(frame) if axis_labels is None else axis_labels
    if len(labels) != 2:
        raise ValueError(
            f"axis_labels must contain exactly two values, got {len(labels)}."
        )
    return {
        "attrs": {
            "frame": frame.lower(),
            "type": "sky_coord",
            "units": ensure_units_are_consistent(units),
        },
        "data": to_python_type(data),
        "dims": "sky_dir_label",
        "coords": {
            "sky_dir_label": {
                "data": list(labels),
                "dims": "sky_dir_label",
            }
        },
    }


def make_direction_location_dict(data: list[float], units: str, frame: str) -> dict:
    """
    Build a serialized two-axis location dictionary.

    Parameters
    ----------
    data : list[float]
        Two-element longitude/latitude value.
    units : str
        Units for both values.
    frame : str
        Location frame name.

    Returns
    -------
    dict
        Dictionary with ``attrs``, ``data``, ``dims``, and ``coords`` fields
        representing a location.
    """
    return {
        "attrs": {
            "frame": frame.upper(),
            "type": "location",
            "units": ensure_units_are_consistent(units),
        },
        "data": to_python_type(data),
        "dims": "ellipsoid_dir_label",
        "coords": {
            "ellipsoid_dir_label": {
                "data": ["lon", "lat"],
                "dims": "ellipsoid_dir_label",
            }
        },
    }


def make_time_measure_attrs(units="s", scale="utc", time_format="mjd") -> dict:
    """
    Create attrs for a serialized time measure.

    Parameters
    ----------
    units : str, default="s"
        Time units.
    scale : str, default="utc"
        Time scale name.
    time_format : str, default="mjd"
        Time format name.

    Returns
    -------
    dict
        Time measure attrs containing units, scale, format, and type.
    """
    u = ensure_units_are_consistent(units)
    return {
        "units": u,
        "scale": scale.lower(),
        "format": time_format.lower(),
        "type": "time",
    }


def make_time_measure_dict(data, units="s", scale="utc", time_format="mjd") -> dict:
    """
    Create a serialized time measure dictionary.

    Parameters
    ----------
    data : numeric or array-like
        Time value.
    units : str, default="s"
        Time units.
    scale : str, default="utc"
        Time scale.
    time_format : str, default="mjd"
        Time format.

    Returns
    -------
    dict
        Time measure dictionary containing attrs and scalar data.
    """
    x = {}
    x["attrs"] = make_time_measure_attrs(units, scale, time_format)
    x["data"] = to_python_type(data)
    x["dims"] = []
    return x


def make_time_coord_attrs(units="s", scale="utc", time_format="mjd") -> dict:
    """
    Create coordinate attrs for time coordinates.

    Parameters
    ----------
    units : str, default="s"
        Time units.
    scale : str, default="utc"
        Time scale.
    time_format : str, default="mjd"
        Time format.

    Returns
    -------
    dict
        Time attrs dictionary without the ``type`` key, suitable for coords.
    """
    x = make_time_measure_attrs(units, scale.lower(), time_format.lower())
    del x["type"]
    return x


def _casacore_q_to_xradio_q(q: dict) -> dict:
    """
    Recursively convert casacore quantity dictionaries to xradio quantities.

    Parameters
    ----------
    q : dict
        Casacore-style quantity dictionary, potentially nested.

    Returns
    -------
    dict
        Converted xradio quantity dictionary.
    """
    if isinstance(q, dict):
        if "value" in q and "unit" in q:
            return make_quantity(q["value"], q["unit"])
        else:
            p = {}
            for k in q:
                p[k] = _casacore_q_to_xradio_q(q[k])
            return p
    else:
        raise ValueError(f"Cannot convert {q} to xradio quantity")
