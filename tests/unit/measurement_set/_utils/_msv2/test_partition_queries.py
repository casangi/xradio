import numpy as np
import pytest
import shutil

import xarray as xr

from xradio.measurement_set._utils._msv2.partition_queries import create_partitions

expected_partition_axes = [
    "DATA_DESC_ID",
    "OBSERVATION_ID",
    "FIELD_ID",
    "SCAN_NUMBER",
    "STATE_ID",
    "SOURCE_ID",
    "OBS_MODE",
    "SUB_SCAN_NUMBER",
]


def test_create_partitions_ms_empty(ms_empty_required):
    parts = create_partitions(ms_empty_required.fname, [])
    assert parts == []


def test_create_paritions_ms_min(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, [])
    assert isinstance(parts, list)
    assert len(parts) == 4
    assert all([axis in part for axis in expected_partition_axes for part in parts])


def test_create_partitions_ms_min_with_field(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, ["FIELD_ID"])
    assert isinstance(parts, list)
    assert len(parts) == 4
    assert all([axis in part for axis in expected_partition_axes for part in parts])


def test_create_partitions_ms_min_with_antenna1(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, ["ANTENNA1"])
    assert isinstance(parts, list)
    assert len(parts) == 16
    assert all([axis in part for axis in expected_partition_axes for part in parts])


def test_create_partitions_ms_min_with_state_id(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, ["STATE_ID"])
    assert isinstance(parts, list)
    assert len(parts) == 4
    assert all([axis in part for axis in expected_partition_axes for part in parts])


def test_create_partitions_ms_min_with_all(ms_minimal_required):

    parts = create_partitions(
        ms_minimal_required.fname,
        [
            "FIELD_ID",
            "SCAN_NUMBER",
            "STATE_ID",
            "SOURCE_ID",
            "SUB_SCAN_NUMBER",
            "ANTENNA1",
        ],
    )
    assert isinstance(parts, list)
    assert len(parts) == 16
    assert all([axis in part for axis in expected_partition_axes for part in parts])


def test_create_partitions_ms_min_with_other(ms_minimal_required):

    with pytest.raises(IndexError, match="out of range"):
        parts = create_partitions(
            ms_minimal_required.fname,
            [
                "FIELD_ID",
                "SOURCE_ID",
                "DATA_DESC_ID",
            ],
        )
