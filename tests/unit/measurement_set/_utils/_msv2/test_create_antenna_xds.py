import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set._utils._msv2.create_antenna_xds import (
    create_antenna_xds,
    create_gain_curve_xds,
)
from xradio.measurement_set.schema import AntennaXds, GainCurveXds
from xradio.schema.check import check_dataset


def test_create_antenna_xds_empty(ms_empty_required):
    with pytest.raises(KeyError, match="No variable named"):
        _antenna_xds = create_antenna_xds(
            ms_empty_required.fname,
            0,
            np.arange(0, 11),
            list(range(0, 2)),
            "ALMA",
            xr.DataArray(),
        )


def test_create_antenna_xds_minimal_wrong_antenna_ids(ms_minimal_required):
    with pytest.raises(ValueError, match="conflicting sizes for dimension"):
        _antenna_xds = create_antenna_xds(
            ms_minimal_required.fname,
            0,
            np.arange(0, 10),
            [],  # np.arange(0, 2),
            "ALMA",
            xr.DataArray(),
        )


def test_create_antenna_xds_minimal_wrong_feed_ids(ms_minimal_required):
    with pytest.raises(RuntimeError, match="FEED_ID"):
        _antenna_xds = create_antenna_xds(
            ms_minimal_required.fname,
            0,
            np.arange(0, 5),
            np.arange(0, 0),
            "ALMA",
            xr.DataArray(),
        )


def test_create_antenna_xds_minimal(ms_minimal_required):
    antenna_xds = create_antenna_xds(
        ms_minimal_required.fname,
        0,
        np.arange(0, 5),
        np.arange(0, 2),
        "ALMA",
        xr.DataArray(),
    )

    check_dataset(antenna_xds, AntennaXds)
    assert len(antenna_xds.coords["antenna_name"]) == len(
        ms_minimal_required.descr["ANTENNA"]
    )


def test_create_antenna_xds_minimal_other_telescope(ms_minimal_required):
    antenna_xds = create_antenna_xds(
        ms_minimal_required.fname,
        0,
        np.arange(0, 5),
        np.arange(0, 2),
        "test_telescope",
        xr.DataArray(),
    )

    check_dataset(antenna_xds, AntennaXds)
    assert len(antenna_xds.coords["antenna_name"]) == len(
        ms_minimal_required.descr["ANTENNA"]
    )


def test_create_antenna_xds_misbehaved(ms_minimal_misbehaved):
    antenna_xds = create_antenna_xds(
        ms_minimal_misbehaved.fname,
        0,
        np.arange(0, 5),
        np.arange(0, 2),
        "ALMA",
        partition_polarization=xr.DataArray(["XX", "YY"]),
    )

    check_dataset(antenna_xds, AntennaXds)
    assert len(antenna_xds.coords["antenna_name"]) == len(
        ms_minimal_misbehaved.descr["ANTENNA"]
    )


def test_create_antenna_xds_without_opt(ms_minimal_without_opt):
    antenna_xds = create_antenna_xds(
        ms_minimal_without_opt.fname,
        0,
        np.arange(0, 5),
        np.arange(0, 2),
        "ALMA",
        partition_polarization=xr.DataArray(["XX", "YY"]),
    )

    check_dataset(antenna_xds, AntennaXds)
    assert len(antenna_xds.coords["antenna_name"]) == len(
        ms_minimal_without_opt.descr["ANTENNA"]
    )


def test_create_gain_curve_xds_minimal(ms_minimal_required):
    # The generated GAIN_CURVE has one row per antenna, all at the same (valid)
    # epoch. TIME used to be written as 1e12 into an int32 column, which
    # overflowed and gave platform/compiler-dependent, even per-row different,
    # values (osx-arm64 casacore 3.8.2).
    antenna_xds = create_antenna_xds(
        ms_minimal_required.fname,
        0,
        np.arange(0, 5),
        np.arange(0, 2),
        "ALMA",
        xr.DataArray(),
    )
    gain_curve_xds = create_gain_curve_xds(ms_minimal_required.fname, 0, antenna_xds)

    check_dataset(gain_curve_xds, GainCurveXds)
    assert gain_curve_xds.attrs["measured_date"].startswith("2025-05-01T01:01")
