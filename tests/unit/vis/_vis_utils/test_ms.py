import pytest

from tests.unit.vis.ms_test_utils.cds_checks import check_cds


@pytest.mark.uses_download
def test_read_alma_ms_default(ms_alma_antennae_north_split):
    """Read with default parameters ('intent' partitioning)"""
    from xradio.vis._vis_utils.ms import read_ms

    vis = read_ms(ms_alma_antennae_north_split.fname)
    check_cds(vis)


@pytest.mark.uses_download
def test_read_alma_ms_by_ddi(ms_alma_antennae_north_split):
    from xradio.vis._vis_utils.ms import read_ms

    scheme = "ddi"
    vis = read_ms(ms_alma_antennae_north_split.fname, partition_scheme=scheme)
    check_cds(vis, partition_scheme=scheme)


def test_read_ms_by_ddi_empty_required(ms_empty_required):
    from xradio.vis._vis_utils.ms import read_ms

    with pytest.raises(AttributeError, match="object has no attribute 'row'"):
        vis = read_ms(ms_empty_required.fname, partition_scheme="ddi")
        assert vis.metainfo


def test_read_ms_by_ddi_minimal(ms_minimal_required):
    from xradio.vis._vis_utils.ms import read_ms

    vis = read_ms(ms_minimal_required.fname, partition_scheme="ddi")
    assert vis.metainfo


def test_read_ms_by_intent_empty_required(ms_empty_required):
    from xradio.vis._vis_utils.ms import read_ms

    with pytest.raises(AttributeError, match="object has no attribute"):
        vis = read_ms(ms_empty_required.fname, partition_scheme="intent")
        assert vis.metainfo


def test_read_ms_by_intent_minimal(ms_minimal_required):
    from xradio.vis._vis_utils.ms import read_ms

    vis = read_ms(ms_minimal_required.fname, partition_scheme="intent")
    assert vis.metainfo


def test_read_ms_by_ddi_minimal_with_asdm(ms_minimal_required):
    from xradio.vis._vis_utils.ms import read_ms

    vis = read_ms(
        ms_minimal_required.fname, asdm_subtables=True, partition_scheme="ddi"
    )
    assert vis.metainfo


def test_read_ms_by_intent_expand_raises(ms_minimal_required):
    from xradio.vis._vis_utils.ms import read_ms

    cds = read_ms(ms_minimal_required.fname, partition_scheme="intent", expand=True)
    assert cds
