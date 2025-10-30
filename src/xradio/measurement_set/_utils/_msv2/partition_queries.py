import itertools
import time
import toolviper.utils.logger as logger
import os
import pandas as pd

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


import pickle, gzip


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
    par_df = pd.DataFrame(base_cols).drop_duplicates()
    logger.debug(
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
    logger.debug(
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
        logger.debug(
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

    logger.debug(
        f"STATE processing in {time.time() - t2:.2f}s "
        f"(OBS_MODE={obs_mode_added}, SUB_SCAN_NUMBER={sub_scan_added})"
    )

    # --------- Decide which partition keys are actually available ----------
    t3 = time.time()
    partition_scheme_updated = [k for k in partition_scheme if k in par_df.columns]
    logger.info(f"Updated partition scheme used: {partition_scheme_updated}")

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
    # Fast aggregation: use NumPy for uniques to avoid pandas overhead in the tight loop.
    for _, gdf in groups_iter:
        part = {}
        for name in partition_axis_names:
            if name in gdf.columns:
                # Return Python lists to match your prior structure (can be np.ndarray if preferred)
                part[name] = np.unique(gdf[name].to_numpy()).tolist()
            else:
                part[name] = [None]
        partitions.append(part)

    logger.debug(
        f"Partition build in {time.time() - t3:.2f}s; total {len(partitions):,} partitions"
    )
    logger.debug(f"Total create_partitions time: {time.time() - t0:.2f}s")

    # # with gzip.open("partition_original_small.pkl.gz", "wb") as f:
    # #     pickle.dump(partitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    # #partitions[1]["DATA_DESC_ID"] = [999]  # make a change to test comparison
    # #org_partitions = load_dict_list("partition_original_small.pkl.gz")
    # org_partitions = load_dict_list("partition_original.pkl.gz")

    return partitions


from typing import Any, List, Dict


def save_dict_list(filename: str, data: List[Dict[str, Any]]) -> None:
    """
    Save a list of dictionaries containing NumPy arrays (or other objects)
    to a compressed pickle file.
    """
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict_list(filename: str) -> List[Dict[str, Any]]:
    """
    Load a list of dictionaries containing NumPy arrays (or other objects)
    from a compressed pickle file.
    """
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def dict_list_equal(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    """
    Compare two lists of dictionaries to ensure they are exactly the same.
    NumPy arrays are compared with array_equal, other objects with ==.
    """
    if len(a) != len(b):
        return False

    for d1, d2 in zip(a, b):
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


from typing import Iterable, Mapping, Tuple, List, Dict, Any, Set
import numpy as np


def _to_python_scalar(x: Any) -> Any:
    """Convert NumPy scalars to Python scalars; leave others unchanged."""
    if isinstance(x, np.generic):
        return x.item()
    return x


def _to_hashable_value_list(v: Any) -> Tuple[Any, ...]:
    """
    Normalize a dict value (often list/np.ndarray) into a sorted, hashable tuple.
    - Accepts list/tuple/np.ndarray/scalars/None.
    - Treats None as a value.
    - Sorts with a stable key that stringifies items to avoid dtype hiccups.
    """
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if v is None or isinstance(v, (str, bytes)):
        # Treat a bare scalar as a single-element collection for consistency.
        v = [v]
    elif not isinstance(v, (list, tuple)):
        v = [v]

    py_vals = [_to_python_scalar(x) for x in v]
    # Sort by (type name, repr) to keep mixed types stable if present
    return tuple(sorted(py_vals, key=lambda x: (type(x).__name__, repr(x))))


def _canon_partition(
    d: Mapping[str, Any], ignore_keys: Iterable[str] = ()
) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
    """
    Canonicalize a partition dict into a hashable, order-insensitive representation.
    - Drops keys in ignore_keys.
    - Converts each value collection to a sorted tuple.
    - Sorts keys.
    """
    ign: Set[str] = set(ignore_keys)
    items = []
    for k, v in d.items():
        if k in ign:
            continue
        items.append((k, _to_hashable_value_list(v)))
    items.sort(key=lambda kv: kv[0])
    return tuple(items)


def compare_partitions_subset(
    new_partitions: List[Dict[str, Any]],
    original_partitions: List[Dict[str, Any]],
    ignore_keys: Iterable[str] = (),
) -> Tuple[bool, List[Dict[str, Any]]]:
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
