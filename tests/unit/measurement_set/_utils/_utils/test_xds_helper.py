from contextlib import nullcontext as does_not_raise
import pytest

import xarray


def test_xds_packager_mxds(ant_xds_min):
    from xradio.measurement_set._utils._utils.xds_helper import vis_xds_packager_mxds
    from xarray.core.utils import Frozen

    addt = "addition"
    subts = [("antenna", ant_xds_min)]
    parts = {9: 0}
    res = vis_xds_packager_mxds(parts, subts, addt)
    assert res.data_vars == {}
    assert res.sizes == Frozen({})
    assert res.attrs["metainfo"] == [("antenna", ant_xds_min)]
    assert res.attrs["partitions"] == parts


def test_make_global_coords_none():
    from xradio.measurement_set._utils._utils.xds_helper import make_global_coords

    with pytest.raises(KeyError, match="metainfo"):
        res = make_global_coords(xarray.Dataset())
        assert res


def test_make_global_coords_min(ms_minimal_required):
    from xradio.measurement_set._utils._utils.xds_helper import make_global_coords
    from xradio.measurement_set._utils._utils.xds_helper import vis_xds_packager_mxds

    # from xradio.measurement_set._utils.msv2 import read_ms

    # TODO: fixture forf an xds (main) + a cds fixture
    # cds = read_ms(ms_minimal_required.fname, partition_scheme="intent")
    # xds = list(cds.partitions.values())[0]
    # mxds = xarray.Dataset(attrs={"metainfo": cds.metainfo, "partitions": {}})
    mxds = xarray.Dataset(attrs={"metainfo": {}, "partitions": {}})

    with pytest.raises(
        ValueError, match="different number of dimensions on data and dims"
    ):
        res = make_global_coords(mxds)
        assert res


def test_expand_xds_empty():
    from xradio.measurement_set._utils._utils.xds_helper import expand_xds

    empty = xarray.Dataset()
    with pytest.raises(AttributeError, match="object has no attribute"):
        res = expand_xds(empty)
        assert "baseline" in res.data_vars


def test_expand_xds_ddi_min(main_xds_flat_min):
    from xradio.measurement_set._utils._utils.xds_helper import expand_xds

    res = expand_xds(main_xds_flat_min)
    assert res
    assert all([coord in res.coords for coord in ["baseline", "time"]])


def test_flatten_xds_empty():
    from xradio.measurement_set._utils._utils.xds_helper import flatten_xds

    empty = xarray.Dataset()
    res = flatten_xds(empty)
    assert xarray.Dataset.equals(res, empty)


def test_flatten_xds_main_min(main_xds_min):
    from xradio.measurement_set._utils._utils.xds_helper import flatten_xds

    res = flatten_xds(main_xds_min)
    assert all(
        [dim in res.dims for dim in ["row", "uvw_coords", "freq", "pol", "antenna_id"]]
    )
    assert all([dim not in res.dims for dim in ["time", "baseline"]])


def test_flatten_then_expand_xds_main_min(main_xds_min):
    from xradio.measurement_set._utils._utils.xds_helper import flatten_xds
    from xradio.measurement_set._utils._utils.xds_helper import expand_xds

    res = flatten_xds(main_xds_min)
    # flatten removes baseline, time from dims but they stay in vars
    res = res.drop_vars("baseline")
    res_expanded = expand_xds(res)

    assert all(
        [
            dim in res_expanded.dims
            for dim in ["time", "baseline", "uvw_coords", "freq", "pol", "antenna_id"]
        ]
    )


BYTES_TO_GB = 1024 * 1024 * 1204


@pytest.mark.parametrize(
    "mem_avail, shape, elem_size, col_name, expected_res",
    [
        (6 * BYTES_TO_GB, (100, 28, 200, 2), 4, "any name", 100),
        (4 * BYTES_TO_GB, (1000, 28, 4000, 4), 4, "other name", 1000),
        (8 * BYTES_TO_GB, (3600, 28, 10000, 4), 8, "name", 901),
        (8 * BYTES_TO_GB, (10000, 300, 20000, 2), 8, "name", 84),
        (4 * BYTES_TO_GB, (10000, 946, 20000, 4), 8, "name", 6),
    ],
)
def test_calc_otimal_ms_chunk_shape(
    mem_avail, shape, elem_size, col_name, expected_res
):
    import numbers
    from xradio.measurement_set._utils._utils.xds_helper import (
        calc_optimal_ms_chunk_shape,
    )

    res = calc_optimal_ms_chunk_shape(mem_avail, shape, elem_size, col_name)
    assert res == expected_res


@pytest.mark.parametrize(
    "mem_avail, shape, elem_size, col_name, expected_raises",
    [
        (6 * BYTES_TO_GB, (100, 28, 200, 2), 4, "any name", does_not_raise()),
        (
            4 * BYTES_TO_GB,
            (100, 946, 200000, 4),
            8,
            "name",
            pytest.raises(RuntimeError),
        ),
    ],
)
def test_calc_otimal_ms_chunk_shape_raises(
    mem_avail, shape, elem_size, col_name, expected_raises
):
    import numbers
    from xradio.measurement_set._utils._utils.xds_helper import (
        calc_optimal_ms_chunk_shape,
    )

    with expected_raises:
        res = calc_optimal_ms_chunk_shape(mem_avail, shape, elem_size, col_name)
