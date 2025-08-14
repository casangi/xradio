def make_quantity(value, units: str, dims: list = []) -> dict:
    """
    create a quantity dictionary given value and units

    Parameters
    ----------
    value : numeric or array of numerics
        Quantity value
    units: str
        Quantity units

    Returns
    -------
    dict
    """
    return {"data": value, "dims": dims, "attrs": make_quantity_attrs(units)}


def ensure_units_are_consistent(units):
    if isinstance(units, str):
        return units
    else:
        u0 = units[0]
        for u in units:
            assert u0 == u, f"Units are not consistent: {u0} != {u}. "
        return u0


def make_quantity_attrs(units: str) -> dict:
    """
    Creates the dict of attributes of a quantity

    Parameters
    ----------
    units: str
        Quantity units

    Returns
    -------
    dict
    """
    return {"units": ensure_units_are_consistent(units), "type": "quantity"}


def make_spectral_coord_reference_dict(
    value: float, units: str, observer: str = "lsrk"
) -> dict:
    """
    creates a spectral_coord measure dict given the value, units, and observer

    Parameters
    ----------
    value : numeric or array of numerics
        measure value
    units : str
        measure units
    observer :
        observer reference frame

    Returns
    -------
    dict
    """
    u = ensure_units_are_consistent(units)
    return {
        "attrs": make_spectral_coord_measure_attrs(
            u,
            observer.lower() if observer not in ["TOPO", "BARY", "REST"] else observer,
        ),
        "data": value,
        "dims": [],
    }


def make_spectral_coord_measure_attrs(units: str, observer: str = "lsrk") -> dict:
    """
    Creates a spectral_coord measure attrs dict given units and observer

    Parameters
    ----------
    units: str or list of str
        Spectral coordinate units
    observer: str
        Spectral reference frame
    Returns
    -------
    dict
        Attrs dict for a spectral_coord measure
    """
    u = ensure_units_are_consistent(units)
    return {"units": u, "observer": observer, "type": "spectral_coord"}


def make_skycoord_dict(data: list[float], units: str, frame: str) -> dict:
    return {
        "attrs": {
            "frame": frame.lower(),
            "type": "sky_coord",
            "units": ensure_units_are_consistent(units),
        },
        "data": data,
        "dims": ["l", "m"],
    }


def make_time_measure_attrs(units="s", scale="utc", time_format="mjd") -> dict:
    u = ensure_units_are_consistent(units)
    return {"units": u, "scale": scale, "format": time_format, "type": "time"}


def make_time_measure_dict(data, units="s", scale="utc", time_format="mjd") -> dict:
    """
    create a time measure dictionary given value and units
    Parameters
    ----------
    value : numeric or array of numerics
        Time value
    units: str
        Time units
    scale: str
        Time scale
    time_format: str
        Time format
    Returns
    -------
    dict
    """
    x = {}
    x["attrs"] = make_time_measure_attrs(units, scale, time_format)
    x["data"] = data
    x["dims"] = []
    return x


def make_time_coord_attrs(units="s", scale="utc", time_format="mjd") -> dict:
    """
    create a time measure dictionary given value and units
    Parameters
    ----------
    value : numeric or array of numerics
        Time value
    units: str
        Time units
    scale: str
        Time scale
    time_format: str
        Time format
    Returns
    -------
    dict
    """
    x = make_time_measure_attrs(units, scale, time_format)
    del x["type"]
    return x


def _casacore_q_to_xradio_q(q: dict) -> dict:
    """
    Convert a casacore quantity to an xradio quantity
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
