import gzip
import itertools
import os
import pickle
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.read import table_exists
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    group_row_runs,
    runs_to_rows,
)

# Keys of a partition description holding the partition's MAIN row membership, as
# runs of consecutive rows: run i is rows [starts[i], starts[i] + lengths[i]).
MAIN_ROW_STARTS_KEY = "main_row_starts"
MAIN_ROW_LENGTHS_KEY = "main_row_lengths"

# Partition description keys that select MAIN rows, in the order used by
# conversion.create_taql_query_where (which also lists STATE_ID twice). For
# ANTENNA1, ANTENNA2 must be in the same list (autocorrelations only).
MAIN_ROW_SELECTION_KEYS = (
    "DATA_DESC_ID",
    "OBSERVATION_ID",
    "STATE_ID",
    "FIELD_ID",
    "SCAN_NUMBER",
    "ANTENNA1",
)


def enumerated_product(*args):
    yield from zip(
        itertools.product(*(range(len(x)) for x in args)),
        itertools.product(*args),
        strict=False,
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
        list of dictionaries with the partition information. Besides the
        partition axes (a list of the unique values of every axis, or [None]
        when the axis is not available), every dictionary holds the MAIN rows
        of the partition as runs of consecutive rows (the same rows, in the
        same ascending order, that conversion.create_taql_query_where selects):

        - "main_row_starts": np.ndarray (int64), first row of every run.
        - "main_row_lengths": np.ndarray (int64), number of rows of every run.

        The rows are computed with one vectorized group-by over all MAIN rows.
        Like the TaQL selection they include the ANTENNA1 rule (with
        "ANTENNA1" in the scheme, a partition only holds the rows whose
        ANTENNA2 equals its ANTENNA1, i.e. autocorrelations), and rows with
        STATE_ID=-1 are grouped with the last STATE row (numpy negative
        indexing into STATE, as in the partition axes).
    """

    ### Test new implementation without
    # Always start with these (if available); then extend with user scheme.
    partition_scheme = [
        "DATA_DESC_ID",
        "OBS_MODE",
        "OBSERVATION_ID",
        "EPHEMERIS_ID",
    ] + list(partition_scheme)

    # partition_scheme = ["DATA_DESC_ID", "OBS_MODE"] + list(
    #     partition_scheme
    # )

    t0 = time.time()
    # --------- Load base columns from MAIN table ----------
    main_tb = tables.table(
        in_file, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )

    # Build minimal DF once. Pull only columns we may need.
    # Add columns here if you expect to aggregate them per-partition.
    base_cols = {
        "DATA_DESC_ID": main_tb.getcol("DATA_DESC_ID"),
        "FIELD_ID": main_tb.getcol("FIELD_ID"),
        "SCAN_NUMBER": main_tb.getcol("SCAN_NUMBER"),
        "STATE_ID": main_tb.getcol("STATE_ID"),
        "OBSERVATION_ID": main_tb.getcol("OBSERVATION_ID"),
        "ANTENNA1": main_tb.getcol("ANTENNA1"),
    }
    # ANTENNA2 is only needed for the row membership of ANTENNA1 partitions
    antenna2 = main_tb.getcol("ANTENNA2") if "ANTENNA1" in partition_scheme else None

    # Unique combinations of the key columns, in order of first appearance and
    # with the index labels of their first MAIN row (as drop_duplicates() gives
    # them), plus the combination of every MAIN row, used for row membership.
    row_key, first_rows = _factorize_rows(list(base_cols.values()))
    par_df = pd.DataFrame(
        {name: col[first_rows] for name, col in base_cols.items()},
        index=first_rows,
    )
    xradio_logger().debug(
        f"Loaded MAIN columns in {time.time() - t0:.2f}s "
        f"({len(par_df):,} unique MAIN rows)"
    )

    # --------- Optional SOURCE/STATE derived columns ----------
    # SOURCE_ID (via FIELD table)
    t1 = time.time()
    source_id_added = False
    field_tb = tables.table(
        os.path.join(in_file, "FIELD"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    if table_exists(os.path.join(in_file, "SOURCE")):
        source_tb = tables.table(
            os.path.join(in_file, "SOURCE"),
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )
        if source_tb.nrows() != 0:
            # Map SOURCE_ID via FIELD_ID
            field_source = np.asarray(field_tb.getcol("SOURCE_ID"))
            par_df["SOURCE_ID"] = field_source[par_df["FIELD_ID"]]
            source_id_added = True
    xradio_logger().debug(
        f"SOURCE processing in {time.time() - t1:.2f}s "
        f"(added SOURCE_ID={source_id_added})"
    )

    if "EPHEMERIS_ID" in field_tb.colnames():
        ephemeris_id_added = False
        if field_tb.nrows() != 0:
            # Map EPHEMERIS_ID via FIELD_ID
            field_ephemeris = np.asarray(field_tb.getcol("EPHEMERIS_ID"))
            par_df["EPHEMERIS_ID"] = field_ephemeris[par_df["FIELD_ID"]]
            ephemeris_id_added = True
        xradio_logger().debug(
            f"EPHEMERIS processing in {time.time() - t1:.2f}s "
            f"(added EPHEMERIS_ID={ephemeris_id_added})"
        )

    # OBS_MODE & SUB_SCAN_NUMBER (via STATE table)
    t2 = time.time()
    obs_mode_added = False
    sub_scan_added = False
    if table_exists(os.path.join(in_file, "STATE")):
        state_tb = tables.table(
            os.path.join(in_file, "STATE"),
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )
        if state_tb.nrows() != 0:
            state_obs_mode = np.asarray(state_tb.getcol("OBS_MODE"))
            state_sub_scan = np.asarray(state_tb.getcol("SUB_SCAN"))
            # Index by STATE_ID into STATE columns
            par_df["OBS_MODE"] = state_obs_mode[par_df["STATE_ID"]]
            par_df["SUB_SCAN_NUMBER"] = state_sub_scan[par_df["STATE_ID"]]
            obs_mode_added = True
            sub_scan_added = True
        else:
            # If STATE empty, drop STATE_ID (it cannot partition anything)
            if "STATE_ID" in par_df.columns:
                par_df.drop(columns=["STATE_ID"], inplace=True)

            if "SUB_SCAN_NUMBER" in par_df.columns:
                par_df.drop(columns=["SUB_SCAN_NUMBER"], inplace=True)

    xradio_logger().debug(
        f"STATE processing in {time.time() - t2:.2f}s "
        f"(OBS_MODE={obs_mode_added}, SUB_SCAN_NUMBER={sub_scan_added})"
    )

    # --------- Decide which partition keys are actually available ----------
    t3 = time.time()
    partition_scheme_updated = [k for k in partition_scheme if k in par_df.columns]
    xradio_logger().info(f"Updated partition scheme used: {partition_scheme_updated}")

    # If none of the requested keys exist, there is a single partition of "everything"
    if not partition_scheme_updated:
        partition_scheme_updated = []

    # These are the axes we report per partition (present => aggregate unique values)
    partition_axis_names = [
        "DATA_DESC_ID",
        "OBSERVATION_ID",
        "FIELD_ID",
        "SCAN_NUMBER",
        "STATE_ID",
        "SOURCE_ID",
        "OBS_MODE",
        "SUB_SCAN_NUMBER",
        "EPHEMERIS_ID",
    ]
    # Only include ANTENNA1 if user asked for it (keeps output size down)
    if "ANTENNA1" in partition_scheme:
        partition_axis_names.append("ANTENNA1")

    # --------- Group only by realized partitions (no Cartesian product!) ----------
    # observed=True speeds up if categorical; here it’s harmless. sort=False keeps source order.
    if partition_scheme_updated:
        grp = par_df.groupby(partition_scheme_updated, sort=False, observed=False)
        groups_iter = grp
    else:
        # Single group: everything
        groups_iter = [(None, par_df)]

    partitions = []
    # Partition index of every unique key combination (-1: in no partition)
    key_partition = np.full(len(first_rows), -1, dtype=np.int64)
    # Fast aggregation: use NumPy for uniques to avoid pandas overhead in the tight loop.
    for _, gdf in groups_iter:
        part = {}
        for name in partition_axis_names:
            if name in gdf.columns:
                # Return Python lists to match your prior structure (can be np.ndarray if preferred)
                part[name] = np.unique(gdf[name].to_numpy()).tolist()
            else:
                part[name] = [None]
        # gdf.index holds the first MAIN row of each of the group's combinations
        key_partition[row_key[gdf.index.to_numpy()]] = len(partitions)
        partitions.append(part)

    xradio_logger().debug(
        f"Partition build in {time.time() - t3:.2f}s; total {len(partitions):,} partitions"
    )

    # --------- MAIN row membership of every partition (row runs) ----------
    t4 = time.time()
    row_partition = key_partition[row_key]
    del row_key
    if antenna2 is not None:
        # create_taql_query_where adds "ANTENNA2 IN [<the partition's ANTENNA1>]".
        # ANTENNA1 is a partition key, so the partition's ANTENNA1 list is the
        # row's own ANTENNA1.
        row_partition[antenna2 != base_cols["ANTENNA1"]] = -1
    for part, (starts, lengths) in zip(
        partitions, group_row_runs(row_partition, len(partitions)), strict=True
    ):
        part[MAIN_ROW_STARTS_KEY] = starts
        part[MAIN_ROW_LENGTHS_KEY] = lengths
    xradio_logger().debug(
        f"Partition MAIN row runs in {time.time() - t4:.2f}s "
        f"({int(np.count_nonzero(row_partition >= 0)):,} of {len(row_partition):,} "
        "MAIN rows in a partition)"
    )
    xradio_logger().debug(f"Total create_partitions time: {time.time() - t0:.2f}s")

    # # with gzip.open("partition_original_small.pkl.gz", "wb") as f:
    # #     pickle.dump(partitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    # #partitions[1]["DATA_DESC_ID"] = [999]  # make a change to test comparison
    # #org_partitions = load_dict_list("partition_original_small.pkl.gz")
    # org_partitions = load_dict_list("partition_original.pkl.gz")

    return partitions


def _factorize_rows(columns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Number the distinct value combinations of several equal-length columns.

    Parameters
    ----------
    columns : list[np.ndarray]
        1-D columns (one value per row).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(row_key, first_rows)``: the combination number of every row (int64,
        numbered in order of first appearance) and the first row of every
        combination (ascending, i.e. the rows ``drop_duplicates()`` keeps).
    """
    nrows = len(columns[0]) if columns else 0
    if nrows == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    row_key = None
    for col in columns:
        # pd.factorize numbers values in order of first appearance
        codes, uniques = pd.factorize(col)
        if row_key is None:
            row_key = codes.astype(np.int64, copy=False)
        else:
            # < nrows**2 (re-factorized at every step), no int64 overflow
            row_key, _ = pd.factorize(row_key * len(uniques) + codes)
            row_key = row_key.astype(np.int64, copy=False)
    # Numbered in order of first appearance: a row starts a new combination
    # exactly when its number is above all the numbers before it.
    is_first = np.empty(nrows, dtype=bool)
    is_first[0] = True
    is_first[1:] = row_key[1:] > np.maximum.accumulate(row_key)[:-1]
    return row_key, np.flatnonzero(is_first)


def partition_main_rows(main_tb: tables.table, partition_info: dict) -> np.ndarray:
    """
    The MAIN row numbers of a partition, ascending: the rows that
    conversion.create_taql_query_where(partition_info) selects.

    Uses the row runs that create_partitions stores in the partition
    description. A description without them (e.g. built by hand) gets its rows
    from a numpy twin of the TaQL selection, which reads the key columns of
    the whole MAIN table once.

    Parameters
    ----------
    main_tb : tables.table
        The opened MAIN table (base table).
    partition_info : dict
        Partition description (as produced by create_partitions).

    Returns
    -------
    np.ndarray
        int64 MAIN row numbers of the partition.
    """
    if MAIN_ROW_STARTS_KEY in partition_info and MAIN_ROW_LENGTHS_KEY in partition_info:
        rows = runs_to_rows(
            partition_info[MAIN_ROW_STARTS_KEY], partition_info[MAIN_ROW_LENGTHS_KEY]
        )
        if rows.size and (rows[-1] >= main_tb.nrows() or rows[0] < 0):
            raise ValueError(
                f"The partition MAIN rows [{rows[0]}, {rows[-1]}] do not fit the MAIN "
                f"table of {main_tb.nrows()} rows (stale partition description?)"
            )
        return rows

    return select_main_rows(main_tb, partition_info)


def select_main_rows(main_tb: tables.table, partition_info: dict) -> np.ndarray:
    """
    numpy twin of the TaQL selection of conversion.create_taql_query_where:
    the rows whose MAIN_ROW_SELECTION_KEYS values are in the partition's value
    lists (keys missing or [None] are skipped; for ANTENNA1, ANTENNA2 must be in
    the same list).

    Parameters
    ----------
    main_tb : tables.table
        The opened MAIN table (base table).
    partition_info : dict
        Partition description.

    Returns
    -------
    np.ndarray
        int64 MAIN row numbers selected, ascending.
    """
    nrows = main_tb.nrows()

    def read_int_col(name: str) -> np.ndarray:
        # scalar int columns are never undefined: a whole-column read is safe
        values = np.empty(nrows, dtype=np.int32)
        if nrows:
            main_tb.getcolnp(name, values)
        return values

    mask = None
    for key in MAIN_ROW_SELECTION_KEYS:
        values = partition_info.get(key)
        if values is None or values[0] is None:
            continue
        key_mask = np.isin(read_int_col(key), np.asarray(values))
        if key == "ANTENNA1":
            key_mask &= np.isin(read_int_col("ANTENNA2"), np.asarray(values))
        mask = key_mask if mask is None else (mask & key_mask)

    if mask is None:
        return np.arange(nrows, dtype=np.int64)
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def save_dict_list(filename: str, data: list[dict[str, Any]]) -> None:
    """
    Save a list of dictionaries containing NumPy arrays (or other objects)
    to a compressed pickle file.
    """
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict_list(filename: str) -> list[dict[str, Any]]:
    """
    Load a list of dictionaries containing NumPy arrays (or other objects)
    from a compressed pickle file.
    """
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def dict_list_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """
    Compare two lists of dictionaries to ensure they are exactly the same.
    NumPy arrays are compared with array_equal, other objects with ==.
    """
    if len(a) != len(b):
        return False

    for d1, d2 in zip(a, b, strict=False):
        if d1.keys() != d2.keys():
            return False
        for k in d1:
            v1, v2 = d1[k], d2[k]
            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
    return True


def _to_python_scalar(x: Any) -> Any:
    """Convert NumPy scalars to Python scalars; leave others unchanged."""
    if isinstance(x, np.generic):
        return x.item()
    return x


def _to_hashable_value_list(v: Any) -> tuple[Any, ...]:
    """
    Normalize a dict value (often list/np.ndarray) into a sorted, hashable tuple.
    - Accepts list/tuple/np.ndarray/scalars/None.
    - Treats None as a value.
    - Sorts with a stable key that stringifies items to avoid dtype hiccups.
    """
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if v is None or isinstance(v, str | bytes):
        # Treat a bare scalar as a single-element collection for consistency.
        v = [v]
    elif not isinstance(v, list | tuple):
        v = [v]

    py_vals = [_to_python_scalar(x) for x in v]
    # Sort by (type name, repr) to keep mixed types stable if present
    return tuple(sorted(py_vals, key=lambda x: (type(x).__name__, repr(x))))


def _canon_partition(
    d: Mapping[str, Any], ignore_keys: Iterable[str] = ()
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """
    Canonicalize a partition dict into a hashable, order-insensitive representation.
    - Drops keys in ignore_keys.
    - Converts each value collection to a sorted tuple.
    - Sorts keys.
    """
    ign: set[str] = set(ignore_keys)
    items = []
    for k, v in d.items():
        if k in ign:
            continue
        items.append((k, _to_hashable_value_list(v)))
    items.sort(key=lambda kv: kv[0])
    return tuple(items)


def compare_partitions_subset(
    new_partitions: list[dict[str, Any]],
    original_partitions: list[dict[str, Any]],
    ignore_keys: Iterable[str] = (),
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Check that every partition in `new_partitions` also appears in `original_partitions`,
    ignoring ordering (of partitions and of values within each key).

    Parameters
    ----------
    new_partitions : list of dict
        Partitions produced by the optimized/new code.
    original_partitions : list of dict
        Partitions produced by the original code (the reference).
    ignore_keys : iterable of str, optional
        Keys to ignore when comparing partitions (e.g., timestamps or debug fields).

    Returns
    -------
    (ok, missing)
        ok : bool
            True if every new partition is found in the original set.
        missing : list of dict
            The list of partitions (from `new_partitions`) that were NOT found in `original_partitions`,
            useful for debugging diffs.
    """
    orig_set = {_canon_partition(p, ignore_keys) for p in original_partitions}
    missing = []
    for p in new_partitions:
        cp = _canon_partition(p, ignore_keys)
        if cp not in orig_set:
            missing.append(p)
    return (len(missing) == 0, missing)
