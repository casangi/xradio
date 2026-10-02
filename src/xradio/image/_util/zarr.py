import os

import numpy as np
import xarray as xr

from xradio._utils.zarr.common import _open_dataset
from xradio.image._util._zarr.xds_from_zarr import _read_zarr
from xradio.image._util._zarr.xds_to_zarr import _write_zarr
from xradio.image._util.common import _to_canonical_polarization_order
from xradio.image._util.legacy import upgrade_legacy_image_attrs


def _xds_to_zarr(xds: xr.Dataset, zarr_store: str):
    _write_zarr(xds, zarr_store)


def _xds_from_zarr(
    zarr_store: str, output: dict | None = None, selection: dict | None = None
) -> xr.Dataset:
    # supported key/values in output are:
    # "dv"
    #    what data variables should be returned as.
    #    "numpy": numpy arrays
    #    "dask": dask arrays
    # "coords"
    #    what coordinates should be returned as
    #    "numpy": numpy arrays
    # Stores written by xradio <= 1.2.3 are upgraded to the current image
    # schema conventions (see upgrade_legacy_image_attrs), and the
    # polarization axis of a store written in another order is put in
    # canonical order, as the CASA and FITS readers return it (a selection
    # indexes the order of the store)
    if output is None:
        output = {}
    if selection is None:
        selection = {}
    return _to_canonical_polarization_order(
        upgrade_legacy_image_attrs(_read_zarr(zarr_store, output, selection))
    )


def _load_image_from_zarr_no_dask(zarr_file: str, selection: dict) -> xr.Dataset:
    # At the moment image module does not support S3 file system. file_system=os is hardcoded.
    image_xds = _open_dataset(
        store=zarr_file, file_system=os, xds_isel=selection, load=True
    )
    for h in ["HISTORY", "_attrs_xds_history"]:
        history = os.sep.join([zarr_file, h])
        if os.path.isdir(history):
            image_xds.attrs["history"] = _open_dataset(history, load=True)
            break
    _iter_dict(image_xds.attrs)
    return _to_canonical_polarization_order(upgrade_legacy_image_attrs(image_xds))


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
