import time

import numpy as np
import pandas as pd

import pyasdm
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)


def create_partitions(sdm: pyasdm.ASDM, partition_scheme: list[str]) -> list[dict]:

    def time_asdm_to_unix(times):
        # beware: ArrayTime sues TAI, not UTC scale -> look for UTCTime class
        # This should be fine, as the first leap second was in 1972.30.06?
        #
        # Also, these functions produce tai-referenced values:
        # time_values = [((asdm_interval.toFITS()) for asdm_interval in main_df["time"].values]

        MJD_TO_UNIX_TIME_DELTA = 3_506_716_800
        time_values = [
            (asdm_interval.get() - MJD_TO_UNIX_TIME_DELTA * 1e9) / 1e9
            for asdm_interval in main_df["time"].values
        ]

        return pd.to_datetime(time_values, unit="s")

    sdm_main_attrs = [
        "time",  # here for now to keep an eye on it
        "fieldId",
        "configDescriptionId",
        "scanNumber",
        "subscanNumber",
        "stateId",
        "dataUID",  # Here to see it, not partition idx
        "execBlockId",  # Here to see it, not partition idx
    ]
    main_df = exp_asdm_table_to_df(sdm, "Main", sdm_main_attrs)
    # It is a key, but in principle not a partition axis. Adding time for convenience.
    main_df["time"] = time_asdm_to_unix(main_df["time"].values)

    # assume one stateId / regardless of antenna
    do_single_state_id = True
    if do_single_state_id:
        main_df["stateId"] = main_df["stateId"].apply(lambda val: val[0])

    do_prints = True

    if do_prints:
        with pd.option_context(
            "display.max_rows", 400
        ):  # , 'display.max_columns', None):
            print(f"* {main_df=}")

    sdm_config_description_attrs = [
        "configDescriptionId",
        "dataDescriptionId",
    ]  # , "processorId"]
    config_description_df = exp_asdm_table_to_df(
        sdm, "ConfigDescription", sdm_config_description_attrs
    )
    # Explode the list in ConfigDescription/dataDescriptionId
    config_description_df = config_description_df.explode(
        "dataDescriptionId", ignore_index=True
    )
    # the explode changes the type of dataDescriptionId to object (of np.int64 items...)
    config_description_df["dataDescriptionId"] = config_description_df[
        "dataDescriptionId"
    ].astype(int)
    if do_prints:
        print(f"* {config_description_df=}")

    sdm_dd_attrs = ["dataDescriptionId", "spectralWindowId", "polOrHoloId"]
    data_description_df = exp_asdm_table_to_df(sdm, "DataDescription", sdm_dd_attrs)
    if do_prints:
        print(f"* {data_description_df=}")

    sdm_field_attrs = ["fieldId", "sourceId"]
    field_df = exp_asdm_table_to_df(sdm, "Field", sdm_field_attrs)
    if do_prints:
        print(f"* {field_df=}")

    sdm_scan_attrs = ["execBlockId", "scanNumber", "scanIntent"]
    scan_df = exp_asdm_table_to_df(sdm, "Scan", sdm_scan_attrs)
    if do_prints:
        print(f"* {scan_df=}")

    # replace scan_intents (list of str) by an id
    unique_scan_intents, unique_intents_inverse = np.unique(
        scan_df["scanIntent"], return_inverse=True
    )
    scan_df["scanIntent"] = unique_intents_inverse

    # Starting with Main+ConfigDescription will usually prune possibilities
    # down to the rows actually in Main (for example there are few rows/scans
    # left in main but the DataDescription and ConfigDescription still have
    # lots of SPWs/Pol-Setups.
    partitioning_df = pd.merge(main_df, config_description_df, on="configDescriptionId")
    if do_prints:
        print(f" * Initial merge, Main+CD: {partitioning_df=}")
    partitioning_df = pd.merge(
        partitioning_df, data_description_df, on="dataDescriptionId"
    )
    if do_prints:
        print(f" * AFTER DD merge: {partitioning_df=}")

    # Starting from DataDescription+ConfigDescription, then main
    # partitioning_df = pd.merge(
    #     config_description_df, data_description_df, on="dataDescriptionId"
    # )
    # if do_prints:
    #     print(f" * Initial merge, CD+DD: {partitioning_df=}")
    # partitioning_df = pd.merge(main_df, partitioning_df, on="configDescriptionId")
    # if do_prints:
    #     print(f" * AFTER Main merge: {partitioning_df=}")

    partitioning_df = pd.merge(partitioning_df, field_df, on="fieldId")
    if do_prints:
        print(f" * AFTER Field merge: {partitioning_df=}")
    partitioning_df = pd.merge(
        partitioning_df, scan_df, on=["scanNumber", "execBlockId"], suffixes=("", "_y")
    )
    if do_prints:
        with pd.option_context(
            "display.max_rows", 400
        ):  # , 'display.max_columns', None):
            print(f" * AFTER Scan (all) merges: {partitioning_df=}")

    potential_partitions = len(partitioning_df)
    print(f" **** {partitioning_df.columns=}")
    scheme_cols = ["execBlockId", "dataDescriptionId", "scanIntent"] + partition_scheme
    print(f"{scheme_cols=}")

    # possible check: would a full drop_duplicates() drop anything? It shouldn't
    # partition_df = partitioning_df.drop_duplicates()
    # print(f" *** {len(partition_df)=} after full drop duplicates!!! / of {potential_partitions}")

    partition_df = partitioning_df.drop_duplicates(subset=scheme_cols).reset_index(
        drop=True
    )
    print(f" \n\n\n***** AFTER drop_duplicates with subset: {partition_df=}")
    partition_df = partition_df.drop(
        columns=set(partition_df.columns.to_list()) - set(scheme_cols)
    )

    with pd.option_context("display.max_rows", 150):  # , 'display.max_columns', None):
        print(f" * FINAL: {partition_df=}")
    print(
        f" {len(partition_df)} out of {potential_partitions} potential partitions are found in dataset"
    )

    example_idx = min(90, partition_df.shape[0] - 1)
    all_cols_example_idx = partitioning_df.loc[partition_df.iloc[example_idx]]
    # print(f" Example {example_idx}-th partition (only explicit cols): {partition_df.iloc[example_idx].to_dict()=}, \n    all cols: {all_cols_example_idx=}")
    print(
        f" Example {example_idx}-th partition (only explicit cols): {partition_df.iloc[example_idx].to_dict()=}, \n    all cols: {all_cols_example_idx=}"
    )

    start = time.perf_counter()
    do_groups = True  # False
    if do_groups:
        partitions = finalize_partitions_groupby(
            partitioning_df, partition_df.columns.to_list(), unique_scan_intents
        )
    else:
        partitions = finalize_partitions(
            partitioning_df, partition_df, unique_scan_intents
        )

    end = time.perf_counter()
    elapsed = end - start
    print(f"Time taken: {elapsed:.6f} seconds")
    return partitions


def finalize_partitions_groupby(
    partitioning_df: pd.DataFrame,
    # partition_df: pd.DataFrame,
    partition_columns: list[str],
    unique_scan_intents: np.ndarray,
) -> list[dict]:
    """
    Produces a list of partitions, with every partition defined as a dict.
    One entry for every potentially partitioning ID/number/etc. column, with
    values set to an array including all the IDs/numbers/etc. of that column
    in the partition.

    partitioning_df: frame with all partitioning columns
    partition_df: frame with only the 'partition_scheme' columns left, where
       every row defines one partition base on those columns.
    """

    partition_groups = partitioning_df.groupby(partition_columns)

    print(f"{len(partition_groups)=}, while {partitioning_df.shape[0]=}")
    # if these two lens above, the partitions are a 1-row df => series

    if len(partition_groups) == partitioning_df.shape[0]:
        # Special case when partitioning all rows (scanNumber and subscanNumber
        # included).
        # Would need to pass something like orient="records" to to_dict()
        # but the API is not the same between pd.DataFrame and pd.Series
        # So when we are left with a 1-row frame (which will be seen by
        # apply as a Series, use this DataFrame global to_dict:
        partitions_list = partitioning_df.to_dict(orient="records")
    else:
        partitions_list = [
            group.apply(lambda col: col.unique(), axis=0).to_dict()
            for _name, group in partition_groups
        ]

    return partitions_list
