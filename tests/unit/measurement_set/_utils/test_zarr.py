import os, shutil, tempfile
from pathlib import Path
import pytest


def clear_output(path):
    if os.path.lexists(path):
        shutil.rmtree(path)


def test_zarr_write_cor_empty(tmp_path):
    from xradio.measurement_set._utils._utils.cds import CASAVisSet
    from xradio.measurement_set._utils.zarr import write_cor
    import xarray

    cds = CASAVisSet(
        {}, {(0, 0, "intent#subintent"): xarray.Dataset()}, "empty vis set"
    )
    outname = str(Path(tmp_path, "test_cor_empty.zarr"))
    with pytest.raises(KeyError, match="other"):
        clear_output(outname)
        write_cor(cds, outname)


def test_zarr_write_exists():
    from xradio.measurement_set._utils.zarr import write_cor

    with tempfile.TemporaryDirectory() as outname:
        with pytest.raises(ValueError, match="already exists"):
            res = write_cor({}, outname)


def test_zarr_write_ms_minimal(cds_minimal_required, tmp_path):
    from xradio.measurement_set._utils.zarr import write_cor
    import copy

    outname = Path(tmp_path, "test_cor_min_write.zarr")
    clear_output(outname)
    cds = copy.deepcopy(cds_minimal_required)
    with pytest.raises(TypeError, match="Invalid attribute in Dataset.attrs"):
        write_cor(cds, outname)

        from xradio.measurement_set._utils.zarr import is_zarr_cor, read_cor

        res = is_zarr_cor(outname)
        assert res
        # TODO: move into a fixture/similar
        # @pytest.mark.depends(on=["test_zarr_write_ms_minimal"])
        cds = read_cor(outname)


def test_zarr_write_cor_attrs(tmp_path):
    """write to zarr (TODO: revisit)"""
    from xradio.measurement_set._utils._utils.cds import CASAVisSet
    from xradio.measurement_set._utils.zarr import write_cor
    import xarray

    empty_part = xarray.Dataset(
        attrs={"other": {"msv2": {"ctds_attrs": {"column_descriptions": {}}}}}
    )
    cds = CASAVisSet({}, {(0, 0, "intent"): empty_part}, "empty vis set")
    outname = str(Path(tmp_path, "test_cor_empty.zarr"))
    with pytest.raises(KeyError, match="UVW"):
        clear_output(outname)
        write_cor(cds, outname)
