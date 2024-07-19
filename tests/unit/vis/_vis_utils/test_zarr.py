import os, shutil, tempfile
from pathlib import Path
import pytest


def clear_output(path):
    if os.path.lexists(path):
        shutil.rmtree(path)


def test_zarr_write_vis_empty(tmp_path):
    from xradio.vis._vis_utils._utils.cds import CASAVisSet
    from xradio.vis._vis_utils.zarr import write_vis
    import xarray

    cds = CASAVisSet(
        {}, {(0, 0, "intent#subintent"): xarray.Dataset()}, "empty vis set"
    )
    outname = str(Path(tmp_path, "test_vis_empty.zarr"))
    with pytest.raises(KeyError, match="other"):
        clear_output(outname)
        write_vis(cds, outname)


def test_zarr_write_exists():
    from xradio.vis._vis_utils.zarr import write_vis

    with tempfile.TemporaryDirectory() as outname:
        with pytest.raises(ValueError, match="already exists"):
            res = write_vis({}, outname)


def test_zarr_write_ms_minimal(cds_minimal_required, tmp_path):
    from xradio.vis._vis_utils.zarr import write_vis
    import copy

    outname = Path(tmp_path, "test_vis_min_write.zarr")
    clear_output(outname)
    cds = copy.deepcopy(cds_minimal_required)
    write_vis(cds, outname)

    from xradio.vis._vis_utils.zarr import is_zarr_vis, read_vis

    res = is_zarr_vis(outname)
    assert res
    # TODO: move into a fixture/similar
    # @pytest.mark.depends(on=["test_zarr_write_ms_minimal"])
    cds = read_vis(outname)


def test_zarr_write_vis_attrs(tmp_path):
    """write to zarr (TODO: revisit)"""
    from xradio.vis._vis_utils._utils.cds import CASAVisSet
    from xradio.vis._vis_utils.zarr import write_vis
    import xarray

    empty_part = xarray.Dataset(
        attrs={"other": {"msv2": {"ctds_attrs": {"column_descriptions": {}}}}}
    )
    cds = CASAVisSet({}, {(0, 0, "intent"): empty_part}, "empty vis set")
    outname = str(Path(tmp_path, "test_vis_empty.zarr"))
    with pytest.raises(KeyError, match="UVW"):
        clear_output(outname)
        write_vis(cds, outname)
