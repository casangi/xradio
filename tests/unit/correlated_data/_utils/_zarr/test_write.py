import pytest
import tempfile


def test_write_part_keys_empty():
    from xradio.correlated_data._utils._zarr.write import write_part_keys
    import xarray as xr

    with pytest.raises(ValueError, match="not enough values to unpack"):
        res = write_part_keys({}, "output_path", {})
        assert res


def test_write_part_keys_min(cds_minimal_required):
    from xradio.correlated_data._utils._zarr.write import write_part_keys

    res = write_part_keys(cds_minimal_required.partitions, "output_path", {})
    assert res == None


def test_write_metainfo_empty():
    from xradio.correlated_data._utils._zarr.write import write_metainfo
    import xarray as xr

    with tempfile.TemporaryDirectory() as outname:
        res = write_metainfo(outname, {})
        assert not res


def test_write_partitions_empty():
    from xradio.correlated_data._utils._zarr.write import write_partitions
    import xarray as xr

    with tempfile.TemporaryDirectory() as outname:
        res = write_partitions(outname, {})
        assert not res


def test_write_xds_to_zarr_empty():
    from xradio.correlated_data._utils._zarr.write import write_xds_to_zarr
    import xarray as xr

    with pytest.raises(KeyError, match="other"):
        res = write_xds_to_zarr(xr.Dataset(), "name", "output_path")
        assert not res


def test_write_xds_to_zarr_ant(ant_xds_min):
    from xradio.correlated_data._utils._zarr.write import write_xds_to_zarr
    import xarray as xr

    res = write_xds_to_zarr(ant_xds_min, "antenna", "output_path")
    assert not res


def test_write_xds_to_zarr_pol(pol_xds_min):
    from xradio.correlated_data._utils._zarr.write import write_xds_to_zarr
    import xarray as xr

    res = write_xds_to_zarr(pol_xds_min, "polarization", "output_path")
    assert not res


def test_write_xds_to_zarr_main_min(main_xds_min):
    from xradio.correlated_data._utils._zarr.write import write_xds_to_zarr
    import xarray as xr

    # with pytest.raises(TypeError, match="Invalid attribute"):
    res = write_xds_to_zarr(
        main_xds_min, "xds_main_test", "output_path", chunks_on_disk={"time": 20}
    )
    assert not res


def test_prepare_attrs_for_zarr_empty():
    from xradio.correlated_data._utils._zarr.write import prepare_attrs_for_zarr
    import xarray as xr

    with pytest.raises(KeyError, match="other"):
        res = prepare_attrs_for_zarr("empty/test", xr.Dataset())
        assert res


def test_prepare_attrs_for_zarr_main_min(main_xds_min):
    from xradio.correlated_data._utils._zarr.write import prepare_attrs_for_zarr
    import xarray as xr

    # TODO: very crude way of exercising corner-case filtering
    main_xds_min.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"]["TIME"][
        "keywords"
    ]["CHANNEL_SELECTION"] = {}
    res = prepare_attrs_for_zarr("xds_test", main_xds_min)
    assert res
