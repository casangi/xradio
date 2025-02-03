def make_quantity(value, units: str) -> dict:
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
    return {"value": value, "units": units, "type": "quantity"}

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


