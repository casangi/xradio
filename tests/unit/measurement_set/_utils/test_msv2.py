import pytest

from tests.unit.measurement_set.ms_test_utils.cds_checks import check_cds


@pytest.mark.uses_download
def test_read_alma_ms_default(ms_alma_antennae_north_split):
    """Read with default parameters ('intent' partitioning)"""
    from xradio.measurement_set._utils.msv2 import read_ms

    correlated_data = read_ms(ms_alma_antennae_north_split.fname)
    check_cds(correlated_data)


def test_read_ms_by_intent_empty_required(ms_empty_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    with pytest.raises(AttributeError, match="object has no attribute"):
        correlated_data = read_ms(ms_empty_required.fname, partition_scheme="intent")
        assert correlated_data.metainfo


def test_read_ms_by_intent_minimal(ms_minimal_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    correlated_data = read_ms(ms_minimal_required.fname, partition_scheme="intent")
    assert correlated_data.metainfo


def test_read_ms_by_intent_expand_raises(ms_minimal_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    cds = read_ms(ms_minimal_required.fname, partition_scheme="intent", expand=True)
    assert cds
