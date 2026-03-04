from xradio._utils.dict_helpers import make_skycoord_dict


def test_make_skycoord_dict_defaults_to_lon_lat_for_galactic_frame():
    """
    Verify Galactic frames default to lon/lat sky-dir labels.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This test validates default axis-label behavior.
    """
    got = make_skycoord_dict(data=[1.5, 0.2], units="rad", frame="galactic")
    assert got["coords"]["sky_dir_label"]["data"] == ["lon", "lat"]


def test_make_skycoord_dict_preserves_explicit_axis_labels():
    """
    Verify explicit axis labels override frame-derived defaults.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This test validates explicit axis-label overrides.
    """
    got = make_skycoord_dict(
        data=[1.5, 0.2],
        units="rad",
        frame="galactic",
        axis_labels=("x", "y"),
    )
    assert got["coords"]["sky_dir_label"]["data"] == ["x", "y"]
