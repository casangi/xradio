from pathlib import Path
from typing import Dict, Tuple

import xarray as xr


from .subtables import subt_rename_ids
from ._tables.read import load_generic_table
from ._tables.read_subtables import read_delayed_pointing_table


def read_spw_ddi_ant_pol(inpath: str) -> Tuple[xr.Dataset]:
    """
    Reads the four metainfo subtables needed to load data chunks into xdss.

    Parameters
    ----------
    inpath : str
        MS path (main table)

    Returns
    -------
    Tuple[xr.Dataset]
        tuple with antenna, ddi, spw, and polarization setup subtables info
    """
    spw_xds = load_generic_table(
        inpath,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    )
    ddi_xds = load_generic_table(inpath, "DATA_DESCRIPTION")
    ant_xds = load_generic_table(
        inpath, "ANTENNA", rename_ids=subt_rename_ids["ANTENNA"]
    )
    pol_xds = load_generic_table(
        inpath, "POLARIZATION", rename_ids=subt_rename_ids["POLARIZATION"]
    )
    return ant_xds, ddi_xds, spw_xds, pol_xds


def finalize_chunks(
    infile: str, chunks: Dict[str, xr.Dataset], chunk_spec: Dict[str, slice]
) -> Dict[Tuple[int, int], xr.Dataset]:
    """
    Adds pointing variables to a dictionary of chunk xdss. This is
    intended to be added after reading chunks from an MS main table.

    Parameters
    ----------
    infile : str
        MS path (main table)
    chunks : Dict[str, xr.Dataset]
        chunk xdss
    chunk_spec : Dict[str, slice]
        specification of chunk to load

    Returns
    -------
    Dict[Tuple[int, int], xr.Dataset]
        dictionary of chunk xdss where every xds now has pointing
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

    if "time" not in pnt_xds.dims:
        return xr.Dataset()

    pnt_xds = pnt_xds.compute()

    pnt_chunks = {
        key: finalize_chunk_xds(infile, xds, pnt_xds)
        for _idx, (key, xds) in enumerate(chunks.items())
    }

    return pnt_chunks


def finalize_chunk_xds(
    infile: str, chunk_xds: xr.Dataset, pointing_xds: xr.Dataset
) -> xr.Dataset:
    """
    Adds pointing variables to one chunk xds.

    Parameters
    ----------
    infile : str
        MS path (main table)
    xds_chunk : xr.Dataset
        chunks xds
    pointing_xds : xr.Dataset
        pointing (sub)table xds

    Returns
    -------
    xr.Dataset
        chunk xds with pointing data variables interpolated form
        the pointing (sub)table
    """

    interp_pnt = pointing_xds.interp(time=chunk_xds.time, method="nearest")

    for var in interp_pnt.data_vars:
        chunk_xds[f"pointing_{var}"] = interp_pnt[var]

    return chunk_xds
