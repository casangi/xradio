import toolviper.utils.logger as logger, numcodecs, os, time, warnings
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Union

import xarray as xr
import zarr


def write_part_keys(
    partitions: Dict[Any, xr.Dataset], outpath: str, compressor: numcodecs.abc.Codec
) -> None:
    """
    Writes an xds with the partition keys.

    Parameters
    ----------
    partitions : Dict[Any, xr.Dataset]
        partitions from a cds
    outpath : str
        path to write a cds
    compressor : numcodecs.abc.Codec
        compressor used for the partition keys variable

    Returns
    -------

    """

    spw_ids, pol_setup_ids, intents = map(list, zip(*partitions.keys()))
    part_keys = xr.Dataset(
        data_vars={
            "spw_ids": spw_ids,
            "pol_setup_ids": pol_setup_ids,
            "intents": intents,
        }
    )

    encoding = dict(zip(list(part_keys.data_vars), cycle([{"compressor": compressor}])))

    out_path = Path(outpath, "partition_keys")
    xr.Dataset.to_zarr(
        part_keys,
        store=out_path,
        mode="w",
        encoding=encoding,
        consolidated=False,
    )
    zarr.consolidate_metadata(out_path)


def write_metainfo(
    outpath: str,
    metainfo: Dict[str, xr.Dataset],
    chunks_on_disk: Union[Dict, None] = None,
    compressor: Union[numcodecs.abc.Codec, None] = None,
    consolidated: bool = True,
) -> None:
    """
    Write all metainfo subtables from a cds to zarr storage

    Parameters
    ----------
    outpath : str

    metainfo : Dict[str, xr.Dataset]:

    chunks_on_disk : Union[Dict, None] (Default value = None)

    compressor : Union[numcodecs.abc.Codec, None) (Default value = None)

    consolidated : bool (Default value = True)


    Returns
    -------

    """
    metadir = Path(outpath, "metainfo")
    os.mkdir(metadir)
    for key, xds in metainfo.items():
        xds_outpath = Path(metadir, key)
        logger.debug(f"Saving metainfo xds {key} into {xds_outpath}")
        write_xds_to_zarr(
            xds, key, xds_outpath, chunks_on_disk, compressor, consolidated=True
        )


def write_partitions(
    outpath: str,
    partitions: Dict[str, xr.Dataset],
    chunks_on_disk: Union[Dict, None] = None,
    compressor: Union[numcodecs.abc.Codec, None] = None,
    consolidated: bool = True,
) -> None:
    """
    Write all data partitions metainfo from a cds to zarr storage

    Parameters
    ----------
    outpath : str :

    partitions : Dict[str, xr.Dataset]

    chunks_on_disk : Union[Dict, None] (Default value = None)

    compressor : Union[numcodecs.abc.Codec, None] (Default value = True)

    consolidated: bool (Default value = True)


    Returns
    -------

    """

    partdir = Path(outpath, "partitions")
    os.mkdir(partdir)
    for cnt, (key, xds) in enumerate(partitions.items()):
        xds_name = f"xds_{cnt}"
        xds_outpath = Path(partdir, str(xds_name))
        logger.debug(f"Saving partition xds {key} into {xds_outpath}")
        write_xds_to_zarr(
            xds, xds_name, xds_outpath, chunks_on_disk, compressor, consolidated=True
        )


def write_xds_to_zarr(
    xds: xr.Dataset,
    name: str,
    outpath: str,
    chunks_on_disk: Union[Dict, None] = None,
    compressor: Union[numcodecs.abc.Codec, None] = None,
    consolidated: bool = True,
    graph_name: str = "write_xds_to_zarr",
) -> None:
    """
    Write one xr dataset from a cds (either metainfo or a partition).

    Parameters
    ----------
    xds : xr.Dataset
        cds (sub)dataset
    name : str
        dataset name (example subtable name, or xds{i})
    outpath: str :

    chunks_on_disk : Union[Dict, None] (Default value = None)

    compressor : Union[numcodecs.abc.Codec, None] (Default value = None)

    consolidated : bool (Default value = True)

    graph_name : str
        the time taken to execute the graph and save the
        dataset is measured and saved as an attribute in the zarr file.
        The graph_name is the label for this timing information.

    Returns
    -------

    """

    xds_for_disk = xds
    if chunks_on_disk is not None:
        xds_for_disk = xds_for_disk.chunk(chunks=chunks_on_disk)

    xds_for_disk = prepare_attrs_for_zarr(name, xds_for_disk)

    if name.startswith("xds"):
        # Do not write replicated/interpolated pointing (keep only original
        # pointing subtable
        pointing_vars = [
            var for var in xds_for_disk.data_vars if var.startswith("pointing_")
        ]
        xds_for_disk = xds_for_disk.drop_vars(pointing_vars)
        # https://github.com/pydata/xarray/issues/2300
        # Could: compute / xds_for_disk[var] = xds_for_disk[var].chunk(chunks_on_disk)

    for var in xds_for_disk.data_vars:
        # Need a dequantify/quantify-kind of pair of functions for units?
        if (
            xds_for_disk.data_vars[var].dtype == "datetime64[ns]"
            and "units" in xds_for_disk.data_vars[var].attrs
        ):
            xds_for_disk.data_vars[var].attrs.pop("units")

    for coord in xds_for_disk.coords:
        if (
            xds_for_disk.coords[coord].dtype == "datetime64[ns]"
            and "units" in xds_for_disk.coords[coord].attrs
        ):
            xds_for_disk.coords[coord].attrs.pop("units")

    # Create compression encoding for each datavariable
    encoding = dict(
        zip(list(xds_for_disk.data_vars), cycle([{"compressor": compressor}]))
    )

    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in cast")
        xr.Dataset.to_zarr(
            xds_for_disk,
            store=outpath,
            mode="w",
            encoding=encoding,
            consolidated=consolidated,
        )
    time_to_calc_and_store = time.time() - start
    logger.debug(
        f"time to store and execute graph ({graph_name}) for {name}: {time_to_calc_and_store}"
    )

    # Add timing information?
    # dataset_group = zarr.open_group(xds_outpath, mode='a')
    # dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store

    if consolidated:
        zarr.consolidate_metadata(outpath)


def prepare_attrs_for_zarr(name: str, xds: xr.Dataset) -> xr.Dataset:
    """
    Deal with types that cannot be serialized as they are in the
    cds/xds (ndarray etc.)

    Parameters
    ----------
    name : str

    xds : xr.Dataset


    Returns
    -------

    """
    ctds_attrs = xds.attrs["other"]["msv2"]["ctds_attrs"]
    col_descrs = ctds_attrs["column_descriptions"]

    # Present in some old vla/alma MSs (model_0, model_1, model_2...
    # vla/ngc5921.applycal.ms, alma/uid___A002_X71a45c_X1d24.ms.split): Just drop.
    for attr in list(ctds_attrs):
        if attr.startswith("model_"):
            ctds_attrs.pop(attr)

    # Seen for example in vla/ngc5921_statwt_ref_test_algorithm_sep_corr_no_fitspw.ms. Just drop.
    data_cols = ["DATA", "CORRECTED_DATA", "MODEL_DATA"]
    for col in data_cols:
        if col in col_descrs and "CHANNEL_SELECTION" in col_descrs[col]["keywords"]:
            col_descrs[col]["keywords"].pop("CHANNEL_SELECTION")

    if "xds" in name:
        col_descrs["UVW"]["shape"] = ",".join(map(str, col_descrs["UVW"]["shape"]))

    if "spectral_window" == name:
        for col in ["CHAN_FREQ", "REF_FREQUENCY"]:
            measinfo = col_descrs[col]["keywords"]["MEASINFO"]
            if "TabRefCodes" in measinfo:
                measinfo["TabRefCodes"] = ",".join(
                    map(
                        str,
                        measinfo["TabRefCodes"],
                    )
                )

    if "antenna" == name:
        for col in ["OFFSET", "POSITION"]:
            if col in col_descrs:
                col_descrs[col]["shape"] = ",".join(map(str, col_descrs[col]["shape"]))
        # ARRAY_POSITION present in keywords of some ALMA-SD datasets
        # example: almasd/expected.ms
        for kw in ["ARRAY_POSITION"]:
            if kw in col_descrs["POSITION"]["keywords"]:
                col_descrs["POSITION"]["keywords"][kw] = ",".join(
                    map(str, col_descrs["POSITION"]["keywords"][kw])
                )

    if "ephemerides" == name:
        if "radii" in ctds_attrs:
            ctds_attrs["radii"]["value"] = ",".join(
                map(str, ctds_attrs["radii"]["value"])
            )

    if "feed" == name:
        col_descrs["POSITION"]["shape"] = ",".join(
            map(str, col_descrs["POSITION"]["shape"])
        )

    if "field" == name:
        for col in ["DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR"]:
            # These shapes are not present in many MSs
            if col in col_descrs and "shape" in col_descrs[col]:
                col_descrs[col]["shape"] = ",".join(map(str, col_descrs[col]["shape"]))
            if "TabRefCodes" in col_descrs[col]["keywords"]["MEASINFO"]:
                col_descrs[col]["keywords"]["MEASINFO"]["TabRefCodes"] = ",".join(
                    map(
                        str,
                        col_descrs[col]["keywords"]["MEASINFO"]["TabRefCodes"],
                    )
                )

    if "observation" == name:
        col_descrs["TIME_RANGE"]["shape"] = ",".join(
            map(str, col_descrs["TIME_RANGE"]["shape"])
        )

    if "source" == name:
        # Note several of these cols are optional and/or only
        # populated with arrays sometimes!
        for col in [
            "DIRECTION",
            "PROPER_MOTION",
            "POSITION",
            "TRANSITION",
            "REST_FREQUENCY",
            "SYSVEL",
        ]:
            if col in col_descrs and "shape" in col_descrs[col]:
                col_descrs[col]["shape"] = ",".join(map(str, col_descrs[col]["shape"]))

    if "weather" == name:
        # Non-std col
        for col in ["NS_WX_STATION_POSITION"]:
            if col in col_descrs and "shape" in col_descrs[col]:
                col_descrs[col]["shape"] = ",".join(map(str, col_descrs[col]["shape"]))

    return xds
