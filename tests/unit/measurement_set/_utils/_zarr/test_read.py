import pytest


def test_read_part_keys(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_part_keys

    with pytest.raises(FileNotFoundError, match="Unable to find group"):
        keys = read_part_keys(ms_as_zarr_min)
        assert keys


def test_read_subtables(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_subtables

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        subts = read_subtables(ms_as_zarr_min, asdm_subtables=True)
        assert subts


def teest_read_paritions(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_partitions

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        parts = read_partitions(ms_as_zarr_min, part_keys=[])
        assert parts


def test_read_xds(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_xds

    with pytest.raises(FileNotFoundError, match="Unable to find group"):
        cds = read_xds(ms_as_zarr_min)
        assert cds


def test_read_zarr(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_zarr

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        cds = read_zarr(ms_as_zarr_min)
        assert cds


def test__fix_dict_for_ms(main_xds_min):
    from xradio.measurement_set._utils._zarr.read import _fix_dict_for_ms
    import copy

    # Crude way of giving this function what it expects. Iterate.
    xds = copy.deepcopy(main_xds_min)
    xds.attrs["column_descriptions"] = xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    with pytest.raises(AttributeError, match="has no attribut"):
        res = _fix_dict_for_ms("xds_test", xds)
        assert res


def test__fix_dict_for_ms_spw(spw_xds_min):
    from xradio.measurement_set._utils._zarr.read import _fix_dict_for_ms
    from xradio.measurement_set._utils._zarr.write import prepare_attrs_for_zarr
    import copy

    # Crude way of giving this function what it expects. Iterate.
    preproc = prepare_attrs_for_zarr("spectral_window", copy.deepcopy(spw_xds_min))
    preproc.attrs["column_descriptions"] = preproc.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    # with pytest.raises(KeyError, match="0"):
    res = _fix_dict_for_ms("spectral_window", preproc)
    assert res


def test__fix_dict_for_ms_ant(ant_xds_min):
    from xradio.measurement_set._utils._zarr.read import _fix_dict_for_ms
    from xradio.measurement_set._utils._zarr.write import prepare_attrs_for_zarr

    # Crude way of giving this function what it expects. Iterate.
    preproc = prepare_attrs_for_zarr("antenna", ant_xds_min)
    preproc.attrs["column_descriptions"] = preproc.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    res = _fix_dict_for_ms("antenna", preproc)
    assert res


def test__fix_dict_for_ms_field(field_xds_min):
    from xradio.measurement_set._utils._zarr.read import _fix_dict_for_ms
    from xradio.measurement_set._utils._zarr.write import prepare_attrs_for_zarr

    # Crude way of giving this function what it expects. Iterate.
    preproc = prepare_attrs_for_zarr("field", field_xds_min)
    preproc.attrs["column_descriptions"] = preproc.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    res = _fix_dict_for_ms("field", preproc)
    assert res


def test__fix_dict_for_ms_feed(feed_xds_min):
    from xradio.measurement_set._utils._zarr.read import _fix_dict_for_ms
    from xradio.measurement_set._utils._zarr.write import prepare_attrs_for_zarr

    # Crude way of giving this function what it expects. Iterate.
    preproc = prepare_attrs_for_zarr("feed", feed_xds_min)
    preproc.attrs["column_descriptions"] = preproc.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    res = _fix_dict_for_ms("feed", preproc)
    assert res


def test__fix_dict_for_ms_observation(observation_xds_min):
    from xradio.measurement_set._utils._zarr.read import _fix_dict_for_ms
    from xradio.measurement_set._utils._zarr.write import prepare_attrs_for_zarr

    # Crude way of giving this function what it expects. Iterate.
    preproc = prepare_attrs_for_zarr("observation", observation_xds_min)
    preproc.attrs["column_descriptions"] = preproc.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    res = _fix_dict_for_ms("observation", preproc)
    assert res
