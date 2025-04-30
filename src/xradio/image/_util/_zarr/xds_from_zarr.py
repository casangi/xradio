import copy
import dask.array as da

# import toolviper.utils.logger as logger
import numpy as np
import os
import xarray as xr
import s3fs
from .common import _np_types, _top_level_sub_xds
from ..common import _coords_to_numpy, _dask_arrayize_dv, _numpy_arrayize_dv
from xradio._utils.zarr.common import _get_file_system_and_items


def _read_zarr(
    zarr_store: str, id_dict: dict, selection: dict = {}
) -> (xr.Dataset, bool):
    # supported key/values in id_dict are:
    # "dv"
    #    what data variables should be returned as.
    #    "numpy": numpy arrays
    #    "dask": dask arrays
    # "coords"
    #    what coords should be returned as
    #    "numpy": numpy arrays
    # it's easiest just to copy the object rather than figuring out
    # if it should be copied at some point or not
    # xds = xr.open_zarr(zarr_store).isel(selection).copy(deep=True)
    xds = xr.open_zarr(zarr_store).isel(selection)
    do_dask = False
    do_numpy = False
    do_np_coords = False
    if "dv" in id_dict:
        dv = id_dict["dv"]
        if dv in ["dask", "numpy"]:
            do_dask = dv == "dask"
            do_numpy = not do_dask
        else:
            raise ValueError(
                f"Unsupported value {id_dict[dv]} for id_dict[dv]. "
                "Supported values are 'dask' and 'numpy'"
            )
    if "coords" in id_dict:
        c = id_dict["coords"]
        if c == "numpy":
            do_np_coords = True
        else:
            raise ValueError(
                f"Unexpected value {c} for id_dict[coords]. "
                "The supported value is 'numpy'"
            )
    # do not pass selection, because that is only for the top level data vars
    xds = _decode(xds, zarr_store, id_dict)
    if do_np_coords:
        xds = _coords_to_numpy(xds)
    if do_dask:
        xds = _dask_arrayize_dv(xds)
    elif do_numpy:
        xds = _numpy_arrayize_dv(xds)
    return xds


def _decode(xds: xr.Dataset, zarr_store: str, id_dict: dict) -> xr.Dataset:
    xds.attrs = _decode_dict(xds.attrs, "")
    _decode_sub_xdses(xds, zarr_store, id_dict)
    return xds


def _decode_dict(my_dict: dict, top_key: str) -> dict:
    # Decodes numpy arrays
    my_dict
    for k, v in my_dict.items():
        if isinstance(v, dict):
            if (
                "_type" in v
                and v["_type"] == "numpy.ndarray"
                and "_value" in v
                and "_dtype" in v
            ):
                my_dict[k] = np.array(v["_value"], dtype=_np_types[v["_dtype"]])
            else:
                z = os.sep.join([top_key, k]) if top_key else k
                my_dict[k] = _decode_dict(v, z)
    return my_dict


def _decode_sub_xdses(xarrayObj, top_dir: str, id_dict: dict) -> None:
    # FIXME this also needs to support S3
    # determine immediate subdirs of zarr_store
    entries = os.scandir(top_dir)
    for d in entries:
        path = os.path.join(top_dir, d.name)
        if os.path.isdir(path):
            if d.name.startswith(_top_level_sub_xds):
                ky = d.name[len(_top_level_sub_xds) :]
                xarrayObj.attrs[ky] = _read_zarr(path, id_dict)
                # TODO if attrs that are xdses have attrs that are xdses ...
            else:
                # descend into the directory
                _decode_sub_xdses(xarrayObj[d.name], path, id_dict)


"""
def _decode_sub_xdses(zarr_store: str, id_dict: dict) -> dict:
    sub_xdses = {}
    fs, store_contents = _get_file_system_and_items(zarr_store)
    if isinstance(fs, s3fs.core.S3FileSystem):
        # could we just use the items as returned from the helper function..?
        store_tree = fs.walk(zarr_store, topdown=True)
        # Q: what is prepend_s3 used for? In this version it is defined but not used.
        prepend_s3 = "s3://"
    else:
        store_tree = os.walk(zarr_store, topdown=True)
        prepend_s3 = ""
    for root, dirs, files in store_tree:
        relpath = os.path.relpath(root, zarr_store)
        print("rpath", relpath)
        for d in dirs:
            if d.startswith(_top_level_sub_xds):
                xds = _read_zarr(os.sep.join([root, d]), id_dict)
                # for k, v in xds.data_vars.items():
                #    xds = xds.drop_vars([k]).assign({k: v.compute()})
                ky = d[len(_top_level_sub_xds) :]
                sub_xdses[ky] = xds
    print(f"Sub xdses: {sub_xdses.keys()}")
    print("return")
    return sub_xdses
"""
