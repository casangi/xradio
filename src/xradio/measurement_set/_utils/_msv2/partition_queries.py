import itertools
import toolviper.utils.logger as logger

import numpy as np

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from ._tables.read import table_exists


def enumerated_product(*args):
    yield from zip(
        itertools.product(*(range(len(x)) for x in args)), itertools.product(*args)
    )


def create_partitions(in_file: str, partition_scheme: list) -> list[dict]:
    """Create a list of dictionaries with the partition information.

    Parameters
    ----------
    in_file: str
        Input MSv2 file path.
    partition_scheme:  list
        A MS v4 can only contain a single data description (spectral window and polarization setup), and observation mode. Consequently, the MS v2 is partitioned when converting to MS v4.
        In addition to data description and polarization setup a finer partitioning is possible by specifying a list of partitioning keys. Any combination of the following keys are possible:
        "FIELD_ID", "SCAN_NUMBER", "STATE_ID", "SOURCE_ID", "SUB_SCAN_NUMBER", "ANTENNA1".
        For mosaics where the phase center is rapidly changing (such as VLA on the fly mosaics)  partition_scheme should be set to an empty list []. By default, ["FIELD_ID"].
    Returns
    -------
    list
        list of dictionaries with the partition information.
    """
    # vla_otf (bool, optional):  The partioning of VLA OTF (on the fly) mosaics needs a special partitioning scheme. Defaults to False.

    # Create partition table
    import pandas as pd
    import os

    partition_scheme = ["DATA_DESC_ID", "OBS_MODE", "OBSERVATION_ID"] + partition_scheme

    # Open MSv2 tables and add columns to partition table (par_df):
    par_df = pd.DataFrame()
    main_tb = tables.table(
        in_file, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )
    par_df["DATA_DESC_ID"] = main_tb.getcol("DATA_DESC_ID")
    par_df["FIELD_ID"] = main_tb.getcol("FIELD_ID")
    par_df["SCAN_NUMBER"] = main_tb.getcol("SCAN_NUMBER")
    par_df["STATE_ID"] = main_tb.getcol("STATE_ID")
    par_df["OBSERVATION_ID"] = main_tb.getcol("OBSERVATION_ID")
    par_df["ANTENNA1"] = main_tb.getcol("ANTENNA1")
    par_df = par_df.drop_duplicates()

    field_tb = tables.table(
        os.path.join(in_file, "FIELD"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    # if vla_otf:
    #     par_df["FIELD_NAME"] = np.array(field_tb.getcol("NAME"))[par_df["FIELD_ID"]]

    # Get source ids if available from source table.
    if table_exists(os.path.join(os.path.join(in_file, "SOURCE"))):
        source_tb = tables.table(
            os.path.join(in_file, "SOURCE"),
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )
        if source_tb.nrows() != 0:
            par_df["SOURCE_ID"] = field_tb.getcol("SOURCE_ID")[par_df["FIELD_ID"]]
            # if vla_otf:
            #     par_df["SOURCE_NAME"] = np.array(source_tb.getcol("NAME"))[
            #         par_df["SOURCE_ID"]
            #     ]

    # Get intents and subscan numbers if available from state table.
    if table_exists(os.path.join(in_file, "STATE")):
        state_tb = tables.table(
            os.path.join(in_file, "STATE"),
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )
        if state_tb.nrows() != 0:
            # print('state_tb',state_tb.nrows(),state_tb)
            par_df["OBS_MODE"] = np.array(state_tb.getcol("OBS_MODE"))[
                par_df["STATE_ID"]
            ]
            par_df["SUB_SCAN_NUMBER"] = state_tb.getcol("SUB_SCAN")[par_df["STATE_ID"]]
        else:
            par_df.drop(["STATE_ID"], axis=1)

    # Check if all partition scheme criteria are present in the partition table.
    partition_scheme_updated = []
    partition_criteria = {}
    for par in partition_scheme:
        if par in par_df.columns:
            partition_criteria[par] = par_df[par].unique()
            partition_scheme_updated.append(par)
    logger.info(f"Partition scheme that will be used: {partition_scheme_updated}")

    # Make all possible combinations of the partition criteria.
    enumerated_partitions = enumerated_product(*list(partition_criteria.values()))

    # print('par_df',par_df)

    # Create a list of dictionaries with the partition information. This will be used to query the MSv2 main table.
    partitions = []
    partition_axis_names = [
        "DATA_DESC_ID",
        "OBSERVATION_ID",
        "FIELD_ID",
        "SCAN_NUMBER",
        "STATE_ID",
        "SOURCE_ID",
        "OBS_MODE",
        "SUB_SCAN_NUMBER",
    ]
    if "ANTENNA1" in partition_scheme:
        partition_axis_names.append("ANTENNA1")

    for idx, pair in enumerated_partitions:
        query = ""
        for i, par in enumerate(partition_scheme_updated):
            if isinstance(pair[i], str):
                query = query + f'{par} == "{pair[i]}" and '
            else:
                query = query + f"{par} == {pair[i]} and "
        query = query[:-4]  # remove last and
        sub_par_df = par_df.query(query).drop_duplicates()

        if sub_par_df.shape[0] != 0:
            partition_info = {}

            # FIELD_NAME	SOURCE_NAME
            for col_name in partition_axis_names:
                if col_name in sub_par_df.columns:
                    partition_info[col_name] = sub_par_df[col_name].unique()
                else:
                    partition_info[col_name] = [None]

            partitions.append(partition_info)

    return partitions
