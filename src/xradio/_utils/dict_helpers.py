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
    u = units if isinstance(units, list) else [units]
    return {"data": value, "dims": dims, "attrs": {"units": u, "type": "quantity"}}


def make_frequency_reference_dict(
    value: float, units: str, observer: str = "lsrk"
) -> dict:
    u = units if isinstance(units, list) else [units]
    return {
        "attrs": {"units": u, "observer": observer.lower(), "type": "frequency"},
        "data": value,
        "dims": [],
    }


def make_skycoord_dict(data: list[float], units: list[str], frame: str) -> dict:
    return {
        "attrs": {
            "frame": frame.lower(),
            "type": "sky_coord",
            "units": units,
        },
        "data": data,
        "dims": ["l", "m"],
    }


def make_time_measure_attrs(units=["s"], scale="utc", time_format="mjd") -> dict:
    u = units if isinstance(units, list) else [units]
    return {"units": u, "scale": scale, "format": time_format, "type": "time"}


def make_time_coord_attrs(units=["s"], scale="utc", time_format="mjd") -> dict:
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
