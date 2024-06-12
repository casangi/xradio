import os
import pytest
import shutil

# using unittest.mock, we could add pytest_mock to dependencies
from unittest.mock import patch

from .ms_test_utils.cds_checks import check_cds


@patch("xradio.vis._vis_utils.ms.read_ms")
def test_read_vis_minimal_defaults(mock_read_ms, ms_minimal_required):
    from xradio.vis import read_vis

    vis = read_vis(ms_minimal_required.fname)
    assert mock_read_ms.called_once()
    assert mock_read_ms.call_args[0] == (
        ms_minimal_required.fname,
        True,
        False,
        "intent",
        None,
        False,
    )


def test_read_vis_minimal_raises_from_read_ms(ms_minimal_required):
    from xradio.vis import read_vis

    with patch("xradio.vis._vis_utils.ms.read_ms", side_effect=RuntimeError):
        with pytest.raises(RuntimeError):
            vis = read_vis(ms_minimal_required.fname)


@patch("xradio.vis._vis_utils.ms.read_ms")
def test_read_vis_minimal_by_ddi(mock_read_ms, ms_minimal_required):
    from xradio.vis import read_vis

    scheme = "ddi"
    vis = read_vis(ms_minimal_required.fname, partition_scheme=scheme)
    assert mock_read_ms.called_once()
    assert mock_read_ms.call_args[0] == (
        ms_minimal_required.fname,
        True,
        False,
        "ddi",
        None,
        False,
    )


def test_read_nonexisting_ms():
    from xradio.vis import read_vis

    with pytest.raises(ValueError, match="invalid input"):
        vis = read_vis("non-existing-input-vis-tmp.ms")
        assert vis


def test_read_empty_ms_required(ms_empty_required):
    from xradio.vis import read_vis

    with pytest.raises(
        AttributeError, match="object has no attribute 'spectral_window_id'"
    ):
        vis = read_vis(ms_empty_required.fname)
        assert vis.metainfo


def test_read_ms_required(ms_minimal_required):
    from xradio.vis import read_vis

    vis = read_vis(ms_minimal_required.fname)
    assert vis.metainfo
    check_cds(vis)


def test_read_empty_ms_complete(ms_empty_complete):
    from xradio.vis import read_vis

    with pytest.raises(
        AttributeError, match="object has no attribute 'spectral_window_id'"
    ):
        vis = read_vis(ms_empty_complete.fname)
        assert vis.metainfo


def test_read_vis_zarr(vis_zarr_empty):
    from xradio.vis import read_vis

    with pytest.raises(
        ValueError, match="invalid input filename to read_generic_table"
    ):
        vis = read_vis(vis_zarr_empty, asdm_subtables=True)
        assert vis


# TODO: same but now with "ms_minimal_required"  - same with all tests that use ms_empty_required/complete => ms_minimal_required/ms_minimal_complete
def test_load_vis_block_empty_ms_required(ms_empty_required):
    from xradio.vis import load_vis_block

    chunk = {
        "time": slice(0, 8),
        "baseline": slice(0, 10),
        "freq": slice(0, 40),
        "pol": slice(0, 2),
    }

    vis = load_vis_block(ms_empty_required.fname, chunk, (0, 0, "intent"))
    assert vis
    assert vis.partitions
    assert vis.metainfo == {}


def test_load_vis_block_empty_ms_required_all_times(ms_empty_required):
    from xradio.vis import load_vis_block

    chunk = {
        # "time": slice(0, 8),
        "baseline": slice(0, 10),
        "freq": slice(0, 40),
        "pol": slice(0, 2),
    }

    vis = load_vis_block(ms_empty_required.fname, chunk, (0, 0, "intent"))
    assert vis
    assert vis.partitions
    assert vis.metainfo == {}


def test_load_vis_block_empty_ms_complete(ms_empty_complete):
    from xradio.vis import load_vis_block

    chunk = {
        "time": slice(0, 8),
        "baseline": slice(0, 10),
        "freq": slice(0, 40),
        "pol": slice(0, 2),
    }

    vis = load_vis_block(ms_empty_complete.fname, chunk, (0, 1, "other intent"))
    assert vis
    assert vis.partitions
    assert vis.metainfo == {}


def test_load_vis_block_ms_min(ms_minimal_required):
    from xradio.vis import load_vis_block

    chunk = {
        "time": slice(0, 8),
        "baseline": slice(0, 4),
        "freq": slice(0, 10),
        "pol": slice(0, 2),
    }

    vis = load_vis_block(
        ms_minimal_required.fname, chunk, (0, 0, "scan_intent#subscan_intent")
    )
    assert vis
    assert vis.partitions
    assert vis.metainfo == {}


@pytest.mark.uses_download
def test_load_chunk_ms_alma(ms_alma_antennae_north_split):
    from xradio.vis import load_vis_block

    chunk = {
        "time": slice(0, 8),
        "baseline": slice(0, 4),
        "freq": slice(0, 10),
        "pol": slice(0, 2),
    }

    vis = load_vis_block(
        ms_alma_antennae_north_split.fname, chunk, (0, 0, "scan_intent#subscan_intent")
    )
    check_cds(vis, subtables=[], chunks=True)


def clear_output(path):
    if os.path.lexists(path):
        shutil.rmtree(path)


@patch("xradio.vis._vis_utils.zarr.write_vis")
def test_write_vis_empty(mock_write_vis):
    from xradio.vis import write_vis
    from xradio.vis._vis_utils._utils.cds import CASAVisSet
    import xarray

    cds = CASAVisSet({}, {(0, 0, "intent"): xarray.Dataset()}, "empty vis set")
    outname = "test_vis_empty.zarr"
    write_vis(cds, outname, out_format="zarr")
    assert mock_write_vis.called_once()
    assert mock_write_vis.call_args[0] == (cds, outname, None, None)


@patch("xradio.vis._vis_utils.zarr.write_vis")
def test_write_vis_unsupported(mock_write_vis):
    """unsuported out_format"""
    from xradio.vis import write_vis

    with pytest.raises(ValueError, match="Unsupported output format"):
        write_vis({}, out_format="asdm", outpath="uns.asdm")

    assert not mock_write_vis.called
