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
