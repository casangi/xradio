import pytest


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
    "EPHEMERIS_ID",
]


def check_expected_min_partitions(partitions: list[dict], expected_len: int = 4):
    """Checks consistency of the basic partitions of the
    ms_minimal_required test MS (1 per DDI)"""

    assert isinstance(partitions, list)
    assert len(partitions) == expected_len

    assert all(
        [axis in part for axis in expected_partition_axes for part in partitions]
    )

    assert all(
        isinstance(part[axis], list) and len(part[axis]) == 1
        for axis in expected_partition_axes
        for part in partitions
    )
    ddis = {part["DATA_DESC_ID"][0] for part in partitions}
    assert ddis == {0, 1, 2, 3}


def test_create_partitions_ms_empty(ms_empty_required):
    parts = create_partitions(ms_empty_required.fname, [])
    assert parts == []


def test_create_partitions_ms_min(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, [])
    assert isinstance(parts, list)
    assert len(parts) == 4
    assert all([axis in part for axis in expected_partition_axes for part in parts])


def test_create_partitions_ms_min_with_field(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, ["FIELD_ID"])
    check_expected_min_partitions(parts, 4)


def test_create_partitions_ms_min_with_antenna1(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, ["ANTENNA1"])
    check_expected_min_partitions(parts, 16)


def test_create_partitions_ms_min_with_state_id(ms_minimal_required):

    parts = create_partitions(ms_minimal_required.fname, ["STATE_ID"])
    check_expected_min_partitions(parts, 4)


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
    check_expected_min_partitions(parts, 16)


def test_create_partitions_ms_min_with_other(ms_minimal_required):

    parts = create_partitions(
        ms_minimal_required.fname,
        ["FIELD_ID", "SOURCE_ID", "DATA_DESC_ID"],
    )

    check_expected_min_partitions(parts, 4)
