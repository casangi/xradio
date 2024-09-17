import os

import toolviper.utils.logger as logger

from pathlib import Path
from typing import Dict, List

import xarray as xr

from ._tables.read import load_generic_table, table_exists
from ._tables.read_subtables import read_ephemerides, read_delayed_pointing_table


subt_rename_ids = {
    "ANTENNA": {"row": "antenna_id", "dim_1": "xyz"},
    "FEED": {"dim_1": "xyz", "dim_2": "receptor", "dim_3": "receptor2"},
    "FIELD": {"row": "field_id", "dim_1": "poly_id", "dim_2": "ra/dec"},
    "FREQ_OFFSET": {"antenna1": "antenna1_id", "antenna2": "antenna2_id"},
    "OBSERVATION": {"row": "observation_id", "dim_1": "start/end"},
    "POINTING": {"dim_1": "n_polynomial", "dim_3": "dir"},
    "POLARIZATION": {"row": "pol_setup_id", "dim_2": "product_id"},
    "PROCESSOR": {"row": "processor_id"},
    "SPECTRAL_WINDOW": {"row": "spectral_window_id", "dim_1": "chan"},
    "SOURCE": {"dim_1": "ra/dec", "dim_2": "line"},
    "STATE": {"row": "state_id"},
    "SYSCAL": {"dim_1": "frequency", "dim_2": "receptor"},
    # Would make sense for non-std "WS_NX_STATION_POSITION"
    "WEATHER": {"dim_1": "xyz"},
}


def read_ms_subtables(
    infile: str, done_subt: List[str], asdm_subtables: bool = False
) -> Dict[str, xr.Dataset]:
    """
    Read MSv2 subtables (main table keywords) as xr.Dataset

    Parameters
    ----------
    infile : str
        input MeasurementSet path
    done_subt : List[str]
        Subtables that were already read, to skip them
    asdm_subtables : bool (Default value = False)
        Whether to also read ASDM_* subtables

    Returns
    -------
    Dict[str, xr.Dataset]
        dict of xarray datasets read from subtables (metadata tables)

    """
    ignore_msv2_cols_subt = ["FLAG_CMD", "FLAG_ROW", "BEAM_ID"]
    skip_tables = ["SORTED_TABLE", "FLAG_CMD"] + done_subt
    stbl_list = sorted(
        [
            tname
            for tname in os.listdir(infile)
            if (tname not in skip_tables)
            and (os.path.isdir(os.path.join(infile, tname)))
            and (table_exists(os.path.join(infile, tname)))
        ]
    )

    subtables = {}
    for _ii, subt_name in enumerate(stbl_list):
        if not asdm_subtables and subt_name.startswith("ASDM_"):
            logger.debug(f"skipping ASDM_ subtable {subt_name}...")
            continue
        else:
            logger.debug(f"reading subtable {subt_name}...")

        if subt_name == "POINTING":
            subt_path = Path(infile, subt_name)
            rename_ids = {"dim_2": "n_polynomial", "dim_3": "dir"}
            xds = read_delayed_pointing_table(str(subt_path), rename_ids=rename_ids)
        else:
            xds = load_generic_table(
                infile,
                subt_name,
                timecols=["TIME"],
                ignore=ignore_msv2_cols_subt,
                rename_ids=subt_rename_ids.get(subt_name, None),
            )

        if len(xds.sizes) != 0:
            subtables[subt_name.lower()] = xds

    if "field" in subtables:
        ephem_xds = read_ephemerides(infile)
        if ephem_xds:
            subtables["ephemerides"] = ephem_xds

    return subtables


def add_pointing_to_partition(
    xds_part: xr.Dataset, xds_pointing: xr.Dataset
) -> xr.Dataset:
    """
    Take pointing variables from a (delayed) pointing dataset and
    transfer them to a main table partition dataset (interpolating into
    the destination time axis)

    Parameters
    ----------
    xds_part : xr.Dataset
        a partition/sub-xds of the main table
    xds_pointing : xr.Dataset
        the xds read from the pointing subtable

    Returns
    -------
    xr.Dataset
        partition xds with pointing variables added/interpolated from the
        pointing_xds into its time axis

    """
    interp_xds = xds_pointing.interp(time=xds_part.time, method="nearest")
    for var in interp_xds.data_vars:
        xds_part[f"pointing_{var}"] = interp_xds[var]

    return xds_part
