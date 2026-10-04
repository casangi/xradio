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


def test_create_field_and_source_xds_observer_position_metres(
    ms_minimal_required, tmp_path
):
    """
    OBSERVER_POSITION (ephemeris fields) holds plain float64 metres, not an
    astropy Quantity: the dataset equals what zarr gives back (the values
    written are unchanged).
    """
    field_and_source_xds, *_ = create_field_and_source_xds(
        ms_minimal_required.fname,
        np.arange(0, 1),
        0,
        np.arange(0, 1),
        False,
        (0, 1e10),
        True,
    )
    position = field_and_source_xds["OBSERVER_POSITION"].variable
    assert type(position.data) is np.ndarray
    assert position.dtype == np.float64
    assert position.dims == ("cartesian_pos_label",)
    assert position.attrs["units"] == "m"
    # an ITRS position on the Earth's surface, in metres
    assert 6.3e6 < np.linalg.norm(position.values) < 6.4e6
    store = str(tmp_path / "field_and_source.zarr")
    field_and_source_xds.to_zarr(store)
    with xr.open_zarr(store) as back:
        xr.testing.assert_identical(back["OBSERVER_POSITION"].variable, position)


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
