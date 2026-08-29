import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set._utils._msv2.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio.measurement_set.schema import FieldSourceXds
from xradio.schema.check import check_dataset


def test_create_field_and_source_xds_empty(ms_empty_required):
    with pytest.raises(AttributeError, match="no attribute"):
        _field_and_source_xds = create_field_and_source_xds(
            ms_empty_required.fname,
            np.array(0),
            0,
            np.arange(0, 100),
            False,
            (0, 1e10),
            False,
        )


def test_create_field_and_source_xds_minimal_wrong_field_ids(ms_empty_required):
    with pytest.raises(AttributeError, match="no attribute"):
        _field_and_source_xds = create_field_and_source_xds(
            ms_empty_required.fname,
            np.arange(0, 100),
            0,
            np.arange(0, 100),
            False,
            (0, 1e10),
            False,
        )


def get_expected_field_xds_type(descr):
    if descr["params"]["misbehave"]:
        return "field_and_source"
    else:
        return "field_and_source_ephemeris"


def test_create_field_and_source_xds_minimal(ms_minimal_required):
    field_and_source_xds, source_id, num_lines, field_names = (
        create_field_and_source_xds(
            ms_minimal_required.fname,
            np.arange(0, 1),
            0,
            np.arange(0, 1),
            False,
            (0, 1e10),
            True,
        )
    )

    assert source_id == [0]
    assert num_lines == 3
    assert field_names == np.array(["NGC3031_0"])
    check_dataset(field_and_source_xds, FieldSourceXds)
    assert field_and_source_xds.attrs["type"] == get_expected_field_xds_type(
        ms_minimal_required.descr
    )


def test_create_field_and_source_xds_misbehaved(ms_minimal_misbehaved):
    field_and_source_xds, source_id, num_lines, field_names = (
        create_field_and_source_xds(
            ms_minimal_misbehaved.fname,
            np.arange(0, 1),
            0,
            np.arange(0, 1),
            False,
            (0, 1e10),
            True,
        )
    )

    assert source_id == [0]
    assert num_lines == 0
    assert field_names == np.array(["NGC3031_0"])
    check_dataset(field_and_source_xds, FieldSourceXds)
    assert field_and_source_xds.attrs["type"] == get_expected_field_xds_type(
        ms_minimal_misbehaved.descr
    )


def test_create_field_and_source_xds_without_opt(ms_minimal_without_opt):
    field_and_source_xds, source_id, num_lines, field_names = (
        create_field_and_source_xds(
            ms_minimal_without_opt.fname,
            np.arange(0, 1),
            0,
            np.arange(0, 1),
            False,
            (0, 1e10),
            True,
        )
    )

    assert source_id == [0]
    assert num_lines == 0
    assert field_names == np.array(["NGC3031_0"])
    check_dataset(field_and_source_xds, FieldSourceXds)
    assert field_and_source_xds.attrs["type"] == get_expected_field_xds_type(
        ms_minimal_without_opt.descr
    )


def test_pad_missing_sources():
    from xradio.measurement_set._utils._msv2.create_field_and_source_xds import (
        pad_missing_sources,
    )

    # Prepare minimum needed for padding of source_ids
    some_string = "some_string"
    source_xds = xr.Dataset(
        data_vars={
            "VAR1": (["SOURCE_ID"], [some_string]),
        },
        coords={
            "SOURCE_ID": ("SOURCE_ID", [0]),
        },
        attrs={"other": {"msv2": {}}},
    )
    unique_source_ids = np.array([0, 3])
    res = pad_missing_sources(source_xds, unique_source_ids)
    assert "SOURCE_ID" in res.dims
    assert all(res.SOURCE_ID.values == unique_source_ids)
    assert all(res.VAR1 == [some_string, "Unknown"])


def test_ephemeris_padded_when_not_interpolating(tmp_path):
    """With ephemeris_interpolate=False the ephemeris is selected by the
    observation time range, extended by one tabulation step before and one
    after (where available), so that enough points are preserved for
    downstream (for example spline) interpolation. Regression test for issue
    #603, where the ephemeris was trimmed to the two tabulation points
    bracketing the observation."""
    import shutil
    from pathlib import Path

    from xradio.measurement_set.schema import FieldSourceEphemerisXds
    from xradio.testing.measurement_set.msv2_io import gen_minimal_ms, gen_subt_ephem

    nrows = 27
    dmjd = 0.0138889
    seconds_per_day = 86400
    fname, _spec = gen_minimal_ms(str(tmp_path / "test_msv2_ephem_trim.ms"))
    # replace the default single row ephemeris with a longer tabulation
    shutil.rmtree(Path(fname) / "FIELD" / "EPHEM0_FIELDNAME.tab")
    gen_subt_ephem(fname, nrows=nrows)

    def make_field_and_source(window_mjd, ephemeris_interpolate):
        """Run create_field_and_source_xds for an observation window given in
        MJD tabulation steps (casacore epoch is MJD days in seconds)."""
        time_min_max = (
            (50000 + window_mjd[0] * dmjd) * seconds_per_day,
            (50000 + window_mjd[1] * dmjd) * seconds_per_day,
        )
        field_times = np.linspace(time_min_max[0], time_min_max[1], 3)
        field_id = np.zeros(len(field_times), dtype=int)
        field_and_source_xds, _source_id, _num_lines, _field_names = (
            create_field_and_source_xds(
                fname,
                field_id,
                0,
                field_times,
                False,
                time_min_max,
                ephemeris_interpolate=ephemeris_interpolate,
            )
        )
        return field_and_source_xds

    # observation between tabulation points 10 and 11: the bracketing points
    # plus one step on either side must be preserved (points 9 to 12)
    xds = make_field_and_source((10.3, 10.6), ephemeris_interpolate=False)
    assert xds.attrs["type"] == "field_and_source_ephemeris"
    assert xds.sizes["time_ephemeris"] == 4
    assert xds["SOURCE_DIRECTION"].dims == ("time_ephemeris", "sky_dir_label")
    expected_ra = np.deg2rad(230.334 + np.arange(9, 13) * 0.015)
    np.testing.assert_allclose(
        xds["SOURCE_DIRECTION"].values[:, 0], expected_ra, rtol=1e-12
    )
    issues = check_dataset(xds, FieldSourceEphemerisXds)
    assert not issues, f"Schema check failed: {issues}"

    # at the edges of the tabulation there is no previous / next step to add
    xds = make_field_and_source((0.3, 0.6), ephemeris_interpolate=False)
    assert xds.sizes["time_ephemeris"] == 3  # points 0, 1 and 2
    xds = make_field_and_source((25.3, 25.6), ephemeris_interpolate=False)
    assert xds.sizes["time_ephemeris"] == 3  # points 24, 25 and 26

    # with ephemeris_interpolate=True the data is interpolated onto the main
    # time axis instead (bracketing points only, no padding needed)
    interp_xds = make_field_and_source((10.3, 10.6), ephemeris_interpolate=True)
    assert "time_ephemeris" not in interp_xds.dims
    assert interp_xds.sizes["time"] == 3
