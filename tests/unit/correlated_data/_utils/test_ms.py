import pytest

from tests.unit.correlated_data.ms_test_utils.cds_checks import check_cds


@pytest.mark.uses_download
def test_read_alma_ms_default(ms_alma_antennae_north_split):
    """Read with default parameters ('intent' partitioning)"""
    from xradio.correlated_data._utils.ms import read_ms

    correlated_data = read_ms(ms_alma_antennae_north_split.fname)
    check_cds(correlated_data)


@pytest.mark.uses_download
def test_read_alma_ms_by_ddi(ms_alma_antennae_north_split):
    from xradio.correlated_data._utils.ms import read_ms

    scheme = "ddi"
    correlated_data = read_ms(ms_alma_antennae_north_split.fname, partition_scheme=scheme)
    check_cds(correlated_data, partition_scheme=scheme)


def test_read_ms_by_ddi_empty_required(ms_empty_required):
    from xradio.correlated_data._utils.ms import read_ms

    with pytest.raises(AttributeError, match="object has no attribute 'row'"):
        correlated_data = read_ms(ms_empty_required.fname, partition_scheme="ddi")
        assert correlated_data.metainfo


def test_read_ms_by_ddi_minimal(ms_minimal_required):
    from xradio.correlated_data._utils.ms import read_ms

    correlated_data = read_ms(ms_minimal_required.fname, partition_scheme="ddi")
    assert correlated_data.metainfo


def test_read_ms_by_intent_empty_required(ms_empty_required):
    from xradio.correlated_data._utils.ms import read_ms

    with pytest.raises(AttributeError, match="object has no attribute"):
        correlated_data = read_ms(ms_empty_required.fname, partition_scheme="intent")
        assert correlated_data.metainfo


def test_read_ms_by_intent_minimal(ms_minimal_required):
    from xradio.correlated_data._utils.ms import read_ms

    correlated_data = read_ms(ms_minimal_required.fname, partition_scheme="intent")
    assert correlated_data.metainfo


def test_read_ms_by_ddi_minimal_with_asdm(ms_minimal_required):
    from xradio.correlated_data._utils.ms import read_ms

    correlated_data = read_ms(
        ms_minimal_required.fname, asdm_subtables=True, partition_scheme="ddi"
    )
    assert correlated_data.metainfo


def test_read_ms_by_intent_expand_raises(ms_minimal_required):
    from xradio.correlated_data._utils.ms import read_ms

    cds = read_ms(ms_minimal_required.fname, partition_scheme="intent", expand=True)
    assert cds
