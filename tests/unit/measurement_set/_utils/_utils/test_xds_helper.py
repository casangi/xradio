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
