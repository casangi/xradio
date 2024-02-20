from pathlib import Path
from typing import Dict, Tuple

import xarray as xr
import numpy as np


from .msv2_msv3 import ignore_msv2_cols
from .partition_queries import make_partition_ids_by_ddi_scan
from .subtables import subt_rename_ids
from ._tables.load_main_table import load_expanded_main_table_chunk
from ._tables.read import read_generic_table, make_freq_attrs
from ._tables.read_subtables import read_delayed_pointing_table
from .._utils.partition_attrs import add_partition_attrs
from .._utils.xds_helper import make_coords
from xradio.vis._vis_utils._ms.optimised_functions import unique_1d


def read_spw_ddi_ant_pol(inpath: str) -> Tuple[xr.Dataset]:
    """
    Reads the four metainfo subtables needed to load data chunks into xdss.

    :param inpath: MS path (main table)
    :return: tuple with antenna, ddi, spw, and polarization setup subtables info
    """
    spw_xds = read_generic_table(
        inpath,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    )
    ddi_xds = read_generic_table(inpath, "DATA_DESCRIPTION")
    ant_xds = read_generic_table(
        inpath, "ANTENNA", rename_ids=subt_rename_ids["ANTENNA"]
    )
    pol_xds = read_generic_table(
        inpath, "POLARIZATION", rename_ids=subt_rename_ids["POLARIZATION"]
    )
    return ant_xds, ddi_xds, spw_xds, pol_xds


def load_main_chunk(
    infile: str, chunk: Dict[str, slice]
) -> Dict[Tuple[int, int], xr.Dataset]:
    """Loads a chunk of visibility data. For every DDI, a separate
    dataset is produced.
    This is very loosely equivalent to the
    partitions.read_*_partitions functions, but in a load (not lazy)
    fashion and with an implicit single partition wrt. anything but
    DDIs.
    Metainfo (sub)tables) are not loaded, and the result is one or more
    Xarray datasets. It produces one dataset per DDI found within the
    chunk slice of time/baseline.

    :param infile: MS path (main table)
    :param chunk: specification of chunk to load

    :return: dictionary of chunk datasets (keys are spw and pol_setup IDs)
    """

    chunk_dims = ["time", "baseline", "freq", "pol"]
    if not all(key in chunk_dims for key in chunk):
        raise ValueError(f"chunks dict has unknown keys. Accepted ones: {chunk_dims}")

    ant_xds, ddi_xds, spw_xds, pol_xds = read_spw_ddi_ant_pol(infile)

    # TODO: constrain this better/ properly
    data_desc_id, scan_number, state_id = make_partition_ids_by_ddi_scan(infile, False)

    all_xdss = {}
    data_desc_id = unique_1d(data_desc_id)
    for ddi in data_desc_id:
        xds, part_ids, attrs = load_expanded_main_table_chunk(
            infile, ddi, chunk, ignore_msv2_cols=ignore_msv2_cols
        )

        coords = make_coords(xds, ddi, (ant_xds, ddi_xds, spw_xds, pol_xds))
        xds = xds.assign_coords(coords)
        xds = add_partition_attrs(xds, ddi, ddi_xds, part_ids, other_attrs={})

        # freq dim needs to pull its units/measure info from the SPW subtable
        spw_id = xds.attrs["partition_ids"]["spw_id"]
        xds.freq.attrs.update(make_freq_attrs(spw_xds, spw_id))
        pol_setup_id = ddi_xds.polarization_id.values[ddi]

        chunk_ddi_key = (spw_id, pol_setup_id)
        all_xdss[chunk_ddi_key] = xds

    chunk_xdss = finalize_chunks(infile, all_xdss, chunk)

    return chunk_xdss


def finalize_chunks(
    infile: str, chunks: Dict[str, xr.Dataset], chunk_spec: Dict[str, slice]
) -> Dict[Tuple[int, int], xr.Dataset]:
    """
    Adds pointing variables to a dictionary of chunk xdss. This is
    intended to be added after reading chunks from an MS main table.

    :param infile: MS path (main table)
    :param chunks: chunk xdss
    :param chunk_spec: specification of chunk to load

    :return: dictionary of chunk xdss where every xds now has pointing
    data variables
    """
    pnt_name = "POINTING"
    pnt_path = Path(infile, pnt_name)
    if "time" in chunk_spec:
        time_slice = chunk_spec["time"]
    else:
        time_slice = None
    pnt_xds = read_delayed_pointing_table(
        str(pnt_path),
        rename_ids=subt_rename_ids.get(pnt_name, None),
        time_slice=time_slice,
    )
    pnt_xds = pnt_xds.compute()

    pnt_chunks = {
        key: finalize_chunk_xds(infile, xds, pnt_xds)
        for _idx, (key, xds) in enumerate(chunks.items())
    }

    return pnt_chunks


def finalize_chunk_xds(infile: str, chunk_xds: xr.Dataset, pointing_xds) -> xr.Dataset:
    """
    Adds pointing variables to one chunk xds.

    :param infile: MS path (main table)
    :param xds_chunk: chunks xds
    :param pointing_xds: pointing (sub)table xds

    :return: chunk xds with pointing data variables interpolated form
    the pointing (sub)table
    """

    interp_pnt = pointing_xds.interp(time=chunk_xds.time, method="nearest")

    for var in interp_pnt.data_vars:
        chunk_xds[f"pointing_{var}"] = interp_pnt[var]

    return chunk_xds
