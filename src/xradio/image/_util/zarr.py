from ._zarr.xds_to_zarr import _write_zarr
from ._zarr.xds_from_zarr import _read_zarr
import numpy as np
import os
import xarray as xr
from ..._utils.zarr.common import _load_no_dask_zarr


def _xds_to_zarr(xds: xr.Dataset, zarr_store: str):
    _write_zarr(xds, zarr_store)


def _xds_from_zarr(
    zarr_store: str, output: dict = {}, selection: dict = {}
) -> xr.Dataset:
    # supported key/values in output are:
    # "dv"
    #    what data variables should be returned as.
    #    "numpy": numpy arrays
    #    "dask": dask arrays
    # "coords"
    #    what coordinates should be returned as
    #    "numpy": numpy arrays
    return _read_zarr(zarr_store, output, selection)


def _load_image_from_zarr_no_dask(zarr_file: str, selection: dict) -> xr.Dataset:
    image_xds = _load_no_dask_zarr(zarr_file, selection)
    for h in ["HISTORY", "_attrs_xds_history"]:
        history = os.sep.join([zarr_file, h])
        if os.path.isdir(history):
            image_xds.attrs["history"] = _load_no_dask_zarr(history)
            break
    _iter_dict(image_xds.attrs)
    return image_xds


def _iter_dict(d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            keys = v.keys()
            if (
                len(keys) == 3
                and "_dtype" in keys
                and "_type" in keys
                and "_value" in keys
            ):
                d[k] = np.array(v["_value"], dtype=v["_dtype"])
            else:
                _iter_dict(v)
