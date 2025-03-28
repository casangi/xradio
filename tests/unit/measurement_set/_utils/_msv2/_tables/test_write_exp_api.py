from contextlib import nullcontext as does_not_raise
from pathlib import Path
import pytest
import xarray as xr


@pytest.mark.parametrize(
    "cols, expected_output",
    [
        (
            ["OBSERVATION_ID", "VIS", "FLAG"],
            {"OBSERVATION_ID": "OBSERVATION_ID", "DATA": "VIS", "FLAG": "FLAG"},
        ),
        (["antenna1_id", "feed1_id"], {"ANTENNA1": "antenna1_id", "FEED1": "feed1_id"}),
    ],
)
def test_cols_from_xds_to_ms(cols, expected_output):
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        cols_from_xds_to_ms,
    )

    assert cols_from_xds_to_ms(cols) == expected_output


def test_xds_packager_mxds(generic_antenna_xds_min):
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        vis_xds_packager_mxds,
    )
    from xarray.core.utils import Frozen

    addt = "addition"
    subts = [("antenna", generic_antenna_xds_min)]
    parts = {9: 0}
    res = vis_xds_packager_mxds(parts, subts, addt)
    assert res.data_vars == {}
    assert res.sizes == Frozen({})
    assert res.attrs["metainfo"] == [("antenna", generic_antenna_xds_min)]
    assert res.attrs["partitions"] == parts


def test_make_global_coords_none():
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        make_global_coords,
    )

    with pytest.raises(KeyError, match="metainfo"):
        res = make_global_coords(xr.Dataset())
        assert res


def test_make_global_coords_min(
    msv4_min_correlated_xds, antenna_xds_min, field_and_source_xds_min
):
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        make_global_coords,
    )
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        vis_xds_packager_mxds,
    )

    mxds = vis_xds_packager_mxds(
        {"part0": msv4_min_correlated_xds},
        {"antenna": antenna_xds_min, "field": field_and_source_xds_min},
        False,
    )

    res = make_global_coords(mxds)
    assert res
    assert all(
        [key in res for key in ["antenna_name", "antennas", "field_name", "fields"]]
    )


def test_flatten_xds_empty():
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import flatten_xds

    empty = xr.Dataset()
    res = flatten_xds(empty)
    assert xr.Dataset.equals(res, empty)


def test_flatten_xds_main_min(main_xds_min):
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import flatten_xds

    res = flatten_xds(main_xds_min)
    assert all([dim in res.dims for dim in ["row", "uvw_coords", "freq", "pol"]])
    assert all([dim not in res.dims for dim in ["time", "baseline"]])


def test_write_ms_with_cds():
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import write_ms

    with pytest.raises(IndexError, match="out of range"):
        write_ms(
            xr.Dataset(attrs={"partitions": {}, "metainfo": {}}), "test_out_vis.ms"
        )


def test_write_ms_empty():
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        vis_xds_packager_mxds,
        write_ms,
    )

    mxds = vis_xds_packager_mxds({}, {}, add_global_coords=True)
    with pytest.raises(IndexError, match="out of range"):
        write_ms(mxds, outfile="test_out_write_ms_empty.ms")


def test_write_ms_serial_empty():
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        vis_xds_packager_mxds,
        write_ms_serial,
    )

    mxds = vis_xds_packager_mxds({}, {}, add_global_coords=True)
    with pytest.raises(IndexError, match="out of range"):
        write_ms_serial(mxds, outfile="test_out_write_ms_empty.ms")


def test_write_ms_cds_min(cds_minimal_required, tmp_path):
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import write_ms

    with pytest.raises(KeyError, match="variable named"):
        outpath = str(Path(tmp_path, "test_write_cds_min_blah.ms"))
        write_ms(
            cds_minimal_required, outpath, subtables=True, modcols={"FLAG": "flag"}
        )


def test_write_ms_serial_cds_min(cds_minimal_required, tmp_path):
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        vis_xds_packager_mxds,
        write_ms_serial,
    )

    with pytest.raises(KeyError, match="variable named"):
        outpath = str(Path(tmp_path, "test_write_cds_min_blah.ms"))
        write_ms_serial(cds_minimal_required, outpath, subtables=True, verbose=True)


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
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
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
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        calc_optimal_ms_chunk_shape,
    )

    with expected_raises:
        res = calc_optimal_ms_chunk_shape(mem_avail, shape, elem_size, col_name)
