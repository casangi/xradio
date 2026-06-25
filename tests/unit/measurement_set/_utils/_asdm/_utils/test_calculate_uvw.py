import numpy as np

import xarray as xr

import pytest


def test_calculate_uvw_fail():
    from xradio.measurement_set._utils._asdm._utils.calculate_uvw import calculate_uvw

    with pytest.raises(KeyError, match="antenna_name"):
        _uvw = calculate_uvw(
            None,
            xr.DataArray(),
            xr.DataArray(),
            xr.DataArray(),
            xr.DataArray(),
            xr.DataArray(),
        )


def test_calculate_uvw_few_antennas():
    from xradio.measurement_set._utils._asdm._utils.calculate_uvw import calculate_uvw

    key = None
    time = xr.DataArray(
        data=np.array([0, 10, 15, 17]) + 5e9,
        dims="time",
        coords={"time": "time"},
        attrs={"type": "time", "units": "s", "scale": "tai", "format": "unix"},
    )
    phase_center_ra_dec = xr.DataArray(
        data=[[0.11, 0.15]],
        dims=["field_name", "sky_dir_label"],
        coords={
            "field_name": ["dummy_field"],
            "sky_dir_label": ["ra", "dec"],
        },
    )
    antenna_position = xr.DataArray(
        data=[[100, 200, -300], [50, 100, 100]],
        dims=[
            "antenna_name",
            "ellipsoid_dir_label",
        ],
        coords={
            "antenna_name": ["DA01", "DV01"],
            "ellipsoid_dir_label": ["x", "y", "z"],
        },
    )
    baseline_antenna1_name = xr.DataArray(
        data=["DA01", "DV01"],
    )
    baseline_antenna2_name = xr.DataArray(
        data=["DV01", "DA01"],
    )
    uvw = calculate_uvw(
        key,
        time,
        baseline_antenna1_name,
        baseline_antenna2_name,
        antenna_position,
        phase_center_ra_dec,
    )
    expected_shape = (
        time.sizes["time"],
        antenna_position.sizes["antenna_name"],
        3,
    )

    assert isinstance(uvw, np.ndarray)
    assert uvw.shape == expected_shape
    assert uvw.dtype == "float64"
    assert np.allclose(
        uvw,
        [
            [
                [80.01513005, -407.20133056, 16.87173233],
                [-80.01513005, 407.20133056, -16.87173233],
            ],
            [
                [80.07211667, -407.19251244, 16.81409959],
                [-80.07211667, 407.19251244, -16.81409959],
            ],
            [
                [80.10059403, -407.18810103, 16.78526782],
                [-80.10059403, 407.18810103, -16.78526782],
            ],
            [
                [80.111982, -407.18633603, 16.77373224],
                [-80.111982, 407.18633603, -16.77373224],
            ],
        ],
    )
