# TODO: move to top-level dict-helpers or related place
def make_sky_coord_measure_attrs(units: str, frame: str) -> dict:
    """
    Create a dictionary of sky coordinate measure attributes.
    Parameters
    ----------
    units : str or list
        Units for sky coordinate measure. Can be a single string or list of strings.
    frame : str
        Reference frame for sky coordinate measure.
    Returns
    -------
    dict
        Dictionary containing the measure attributes with the following keys:
        - units: list of units
        - frame: reference frame
        - type: fixed to "sky_coord"
    Examples
    --------
    >>> make_sky_coord_measure_attrs("rad", "ICRS")
    {'units': 'rad', 'frame': 'ICRS', 'type': 'sky_coord'}
    """

    sky_coord_measure_attrs = {"units": units, "frame": frame, "type": "sky_coord"}
    return sky_coord_measure_attrs
