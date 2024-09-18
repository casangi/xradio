from collections import namedtuple
from contextlib import nullcontext as no_raises
import numpy as np
import pytest
import xarray as xr

import xradio.correlated_data._utils._ms.conversion as conv


minxds = namedtuple("minxds", "data_vars coords sizes")
xds_main = minxds(
    {"VISIBILITY": np.empty((1), dtype=np.complex64)},
    {"baseline_id"},
    {
        "time": 220,
        "baseline_id": 55,
        "frequency": 3890,
        "polarization": 2,
    },
)
xds_main_sd = minxds(
    {"SPECTRUM": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 1200,
        "antenna_name": 11,
        "frequency": 1800,
        "polarization": 2,
    },
)
xds_main_sd_bogus = minxds(
    {"SPECTRUM": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 1200,
        "bogus": 11,
        "frequency": 1800,
        "polarization": 2,
    },
)
xds_pointing = minxds(
    {"BEAM_POINTING": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 10220,
        "antenna_name": 55,
        "direction": 2,
    },
)
xds_pointing_small = minxds(
    {"BEAM_POINTING": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 102,
        "antenna_name": 12,
        "direction": 2,
    },
)


@pytest.mark.parametrize(
    "input_chunksize, xds_type, xds, expected_chunksize, expected_error",
    [
        ({}, "main", None, {}, no_raises()),
        ({}, "pointing", None, {}, no_raises()),
        (
            {"time": 10, "baseline_id": 5, "frequency": 100, "polarization": 2},
            "main",
            xds_main,
            {"time": 10, "baseline_id": 5, "frequency": 100, "polarization": 2},
            no_raises(),
        ),
        (
            {"time": 10, "foo": 3},
            "main",
            xds_main,
            {},
            pytest.raises(ValueError, match="foo"),
        ),
        (
            0.02,
            "main",
            xds_main,
            {"time": 70, "baseline_id": 16, "frequency": 1198, "polarization": 2},
            no_raises(),
        ),
        (
            0.01,
            "main",
            xds_main_sd,
            {"time": 408, "antenna_name": 3, "frequency": 548, "polarization": 2},
            no_raises(),
        ),
        (
            0.01,
            "main",
            xds_main_sd_bogus,
            None,
            pytest.raises(KeyError, match="antenna_name"),
        ),
        (
            0.0002,
            "pointing",
            xds_pointing,
            {"time": 1677, "antenna_name": 8, "direction": 2},
            no_raises(),
        ),
    ],
)
def test_parse_chunksize(
    input_chunksize, xds_type, xds, expected_chunksize, expected_error
):
    # parse_chunksize checks the input chunksize (if dict), or auto-calculates the
    # sizes (if given as numeric memory value). The calculations are better tested for the
    # lower level functions
    with expected_error:
        assert (
            conv.parse_chunksize(input_chunksize, xds_type, xds) == expected_chunksize
        )


@pytest.mark.parametrize(
    "chunksize, xds_type, expectation",
    [
        ({}, "main", no_raises()),
        ({}, "pointing", no_raises()),
        (
            {"baseline_id": 1, "frequency": 2, "polarization": 3, "time": 4},
            "main",
            no_raises(),
        ),
        (
            {"baseline_id": 1, "frequency": 2, "polarization": 3, "time": 4},
            "pointing",
            pytest.raises(ValueError, match="baseline_id"),
        ),
        ({"foo": "a"}, "main", pytest.raises(ValueError, match="foo")),
        ({"foo": 1}, "pointing", pytest.raises(ValueError, match="foo")),
    ],
)
def test_check_chunksize(chunksize, xds_type, expectation):
    with expectation:
        conv.check_chunksize(chunksize, xds_type)


@pytest.mark.parametrize(
    "pseudo_xds, xds_type, expected_chunksize, expected_error",
    [
        (
            xds_main,
            "main",
            {"baseline_id": 28, "frequency": 2048, "polarization": 2, "time": 117},
            no_raises(),
        ),
        (
            xds_main,
            "bar_wrong",
            {"baseline_id": 28, "frequency": 2048, "polarization": 2, "time": 117},
            pytest.raises(RuntimeError),
        ),
    ],
)
def test_mem_chunksize_to_dict(
    pseudo_xds, xds_type, expected_chunksize, expected_error
):
    # mem_chunksize_to_dict relies on mem_chunksize_to_dict_main_*,
    # mem_chunksize_to_dict_pointing*, etc. which are better tested below
    with expected_error:
        assert (
            conv.mem_chunksize_to_dict(0.1, xds_type, pseudo_xds) == expected_chunksize
        )


@pytest.mark.parametrize(
    "mem_size, pseudo_xds, expected_chunksize, expected_error",
    [
        (
            1e-9,  # not enough even for one data point / all pols
            xds_main,
            {"baseline_id": 28, "frequency": 2048, "polarization": 2, "time": 117},
            pytest.raises(RuntimeError, match="memory bound"),
        ),
        (
            0.9,  # enough to hold all in mem
            xds_main,
            {"time": 220, "baseline_id": 55, "frequency": 3890, "polarization": 2},
            no_raises(),
        ),
        (
            0.01,
            xds_main_sd,
            {"antenna_name": 3, "frequency": 548, "polarization": 2, "time": 408},
            no_raises(),
        ),
        (
            0.01,
            xds_main_sd_bogus,
            None,
            pytest.raises(KeyError, match="antenna_name"),
        ),
    ],
)
def test_mem_chunksize_to_dict_main(
    mem_size, pseudo_xds, expected_chunksize, expected_error
):
    with expected_error:
        assert (
            conv.mem_chunksize_to_dict_main(mem_size, pseudo_xds) == expected_chunksize
        )


@pytest.mark.parametrize(
    "mem_size, dim_sizes, expected_chunksize",
    [
        (
            0.001,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
            {"time": 50, "baseline_id": 4, "frequency": 223, "polarization": 3},
        ),
        (
            0.02,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
            {"time": 124, "baseline_id": 12, "frequency": 601, "polarization": 3},
        ),
        (
            0.03,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
            {"time": 140, "baseline_id": 14, "frequency": 684, "polarization": 3},
        ),
        (
            0.05,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 4},
            {"time": 151, "baseline_id": 15, "frequency": 740, "polarization": 4},
        ),
    ],
)
def test_mem_chunksize_to_dict_main_balanced(mem_size, dim_sizes, expected_chunksize):
    res = conv.mem_chunksize_to_dict_main_balanced(
        mem_size, dim_sizes, "baseline_id", 8
    )
    assert res == pytest.approx(expected_chunksize)


@pytest.mark.parametrize(
    "mem_size, dim_sizes, expected_chunksize",
    [
        (
            0.1,
            xds_pointing,
            {"time": 10220, "antenna_name": 55, "direction": 2},
        ),
        (
            0.5,
            xds_pointing_small,
            xds_pointing_small.sizes,
        ),
        (0.5, minxds({}, {}, {}), {}),
    ],
)
def test_mem_chunksize_to_dict_pointing(mem_size, dim_sizes, expected_chunksize):
    res = conv.mem_chunksize_to_dict_pointing(mem_size, dim_sizes)
    assert res == pytest.approx(expected_chunksize)


def test_itemsize_spec():
    assert conv.itemsize_spec(xds_main) == 8


def itemsize_pointing_spec():
    assert conv.itemsize_spec(xds_pointing) == 8


def test_calc_used_gb():
    res = conv.calc_used_gb(
        {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
        "baseline_id",
        8,
    )
    assert res == pytest.approx(0.0938773)
