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
        field_and_source_xds = create_field_and_source_xds(
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
        field_and_source_xds = create_field_and_source_xds(
            ms_empty_required.fname,
            np.arange(0, 100),
            0,
            np.arange(0, 100),
            False,
            (0, 1e10),
            False,
        )


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
