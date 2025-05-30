import pytest
import sys
from pathlib import Path # this needs to be cleaner
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent)) # this needs to be cleaner
from tests._testutils.cds_checks import check_cds # this needs to be cleaner


@pytest.mark.uses_download
def test_read_alma_ms_default(ms_alma_antennae_north_split):
    """Read with default parameters ('intent' partitioning)"""
    from xradio.measurement_set._utils.msv2 import read_ms

    correlated_data = read_ms(ms_alma_antennae_north_split.fname)
    check_cds(correlated_data)


@pytest.mark.uses_download
def test_read_alma_ms_by_ddi(ms_alma_antennae_north_split):
    from xradio.measurement_set._utils.msv2 import read_ms

    scheme = "ddi"
    correlated_data = read_ms(
        ms_alma_antennae_north_split.fname, partition_scheme=scheme
    )
    check_cds(correlated_data, partition_scheme=scheme)


def test_read_ms_by_ddi_empty_required(ms_empty_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    with pytest.raises(AttributeError, match="object has no attribute 'row'"):
        correlated_data = read_ms(ms_empty_required.fname, partition_scheme="ddi")
        assert correlated_data.metainfo


def test_read_ms_by_ddi_minimal(ms_minimal_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    correlated_data = read_ms(ms_minimal_required.fname, partition_scheme="ddi")
    assert correlated_data.metainfo


def test_read_ms_by_intent_empty_required(ms_empty_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    with pytest.raises(AttributeError, match="object has no attribute"):
        correlated_data = read_ms(ms_empty_required.fname, partition_scheme="intent")
        assert correlated_data.metainfo


def test_read_ms_by_intent_minimal(ms_minimal_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    correlated_data = read_ms(ms_minimal_required.fname, partition_scheme="intent")
    assert correlated_data.metainfo


def test_read_ms_by_ddi_minimal_with_asdm(ms_minimal_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    correlated_data = read_ms(
        ms_minimal_required.fname, asdm_subtables=True, partition_scheme="ddi"
    )
    assert correlated_data.metainfo


def test_read_ms_by_intent_expand_raises(ms_minimal_required):
    from xradio.measurement_set._utils.msv2 import read_ms

    cds = read_ms(ms_minimal_required.fname, partition_scheme="intent", expand=True)
    assert cds


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
