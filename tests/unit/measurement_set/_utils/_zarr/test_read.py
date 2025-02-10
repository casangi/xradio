import pytest


def test_read_part_keys(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_part_keys

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        keys = read_part_keys(ms_as_zarr_min)
        assert keys


def test_read_subtables(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_subtables

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        subts = read_subtables(ms_as_zarr_min, asdm_subtables=True)
        assert subts


def teest_read_partitions(ms_as_zarr_min):
    from xradio.measurement_set._utils._zarr.read import read_partitions

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        parts = read_partitions(ms_as_zarr_min, part_keys=[])
        assert parts
