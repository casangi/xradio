import gzip
import hashlib
import itertools
import operator
import os
import pickle
import time
from collections.abc import Iterable, Mapping, Sequence
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
    group_row_runs_flat,
    runs_to_rows,
)

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
        list of dictionaries with the partition information: the partition
        axes, each a list of the unique values of the axis (or [None] when the
        axis is not available). See create_partitions_with_main_rows for the
        MAIN rows of the partitions.
    """
    partitions, _ = _create_partitions(in_file, partition_scheme, with_main_rows=False)
    return partitions


def create_partitions_with_main_rows(
    in_file: str, partition_scheme: list
) -> tuple[list[dict], "MainRowRuns"]:
    """
    create_partitions, plus the MAIN rows of every partition, computed with
    one vectorized group-by over all MAIN rows.

    The rows are the ones (in the same ascending order) that
    conversion.create_taql_query_where selects for each partition. Like that
    TaQL selection they include the ANTENNA1 rule (with "ANTENNA1" in the
    scheme, a partition only holds the rows whose ANTENNA2 equals its
    ANTENNA1, i.e. autocorrelations), and rows with STATE_ID=-1 are grouped
    with the last STATE row (numpy negative indexing into STATE, as in the
    partition axes).

    Parameters
    ----------
    in_file : str
        Input MSv2 file path.
    partition_scheme : list
        As in create_partitions.

    Returns
    -------
    tuple[list[dict], MainRowRuns]
        The partition descriptions (exactly as create_partitions returns
        them) and their MAIN rows, ``main_row_runs[i]`` for ``partitions[i]``
        (see MainRowRuns and partition_main_rows).
    """
    return _create_partitions(in_file, partition_scheme, with_main_rows=True)


def _create_partitions(
    in_file: str, partition_scheme: list, with_main_rows: bool
) -> tuple[list[dict], "MainRowRuns | None"]:
    """create_partitions[_with_main_rows]: the row runs only if with_main_rows."""

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
    main_nrows = main_tb.nrows()
    # ANTENNA2 is only needed for the row membership of ANTENNA1 partitions
    antenna2 = (
        main_tb.getcol("ANTENNA2")
        if with_main_rows and "ANTENNA1" in partition_scheme
        else None
    )

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

    main_row_runs = None
    if with_main_rows:
        # --------- MAIN row membership of every partition (row runs) ----------
        t4 = time.time()
        row_partition = key_partition[row_key]
        del row_key
        if antenna2 is not None:
            # create_taql_query_where adds "ANTENNA2 IN [<the partition's
            # ANTENNA1>]". ANTENNA1 is a partition key, so the partition's
            # ANTENNA1 list is the row's own ANTENNA1.
            row_partition[antenna2 != base_cols["ANTENNA1"]] = -1
        n_partition_rows = int(np.count_nonzero(row_partition >= 0))
        run_starts, run_lengths, bounds = group_row_runs_flat(
            row_partition, len(partitions)
        )
        del row_partition
        main_row_runs = MainRowRuns(
            run_starts,
            run_lengths,
            bounds,
            _stack_digests([partition_selection_digest(part) for part in partitions]),
            main_nrows,
        )
        xradio_logger().debug(
            f"Partition MAIN row runs in {time.time() - t4:.2f}s "
            f"({n_partition_rows:,} of {main_nrows:,} MAIN rows in a partition)"
        )
    xradio_logger().debug(f"Total create_partitions time: {time.time() - t0:.2f}s")

    # # with gzip.open("partition_original_small.pkl.gz", "wb") as f:
    # #     pickle.dump(partitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    # #partitions[1]["DATA_DESC_ID"] = [999]  # make a change to test comparison
    # #org_partitions = load_dict_list("partition_original_small.pkl.gz")
    # org_partitions = load_dict_list("partition_original.pkl.gz")

    return partitions, main_row_runs


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


# Size (bytes) of partition_selection_digest
SELECTION_DIGEST_SIZE = 16


def partition_selection_digest(partition_info: Mapping[str, Any]) -> bytes | None:
    """
    A digest of the MAIN row selection of a partition description: the values
    of MAIN_ROW_SELECTION_KEYS that conversion.create_taql_query_where turns
    into its WHERE (keys missing or [None] are skipped, as there; the order
    and repetition of values in a list do not matter, as for TaQL IN).

    Parameters
    ----------
    partition_info : Mapping[str, Any]
        Partition description.

    Returns
    -------
    bytes | None
        SELECTION_DIGEST_SIZE bytes, or None if a list holds values that are
        not integers (no digest: such a description never matches row runs).
    """
    digest = hashlib.blake2b(digest_size=SELECTION_DIGEST_SIZE)
    for key in MAIN_ROW_SELECTION_KEYS:
        values = partition_info.get(key)
        if values is None or len(values) == 0 or values[0] is None:
            digest.update(b"|")
            continue
        try:
            unique_values = sorted({operator.index(value) for value in values})
        except TypeError:
            return None
        digest.update(f"|{key}={unique_values}".encode())
    return digest.digest()


def _stack_digests(digests: list[bytes | None]) -> np.ndarray:
    """Digests as rows of a (n, SELECTION_DIGEST_SIZE) uint8 array (None: zeros)."""
    zeros = bytes(SELECTION_DIGEST_SIZE)
    return np.frombuffer(
        b"".join(zeros if d is None else d for d in digests), dtype=np.uint8
    ).reshape(len(digests), SELECTION_DIGEST_SIZE)


class MainRowRuns:
    """
    MAIN rows of a list of partitions (create_partitions_with_main_rows), as
    runs of consecutive rows: run ``j`` holds the rows ``starts[j] ..
    starts[j] + lengths[j] - 1``, and the runs of partition ``i`` are
    ``bounds[i]:bounds[i + 1]``. Stored in a few flat arrays (16 bytes per
    run, 24 per partition), indexed per partition with ``runs[i]``.

    Every partition also keeps a digest of its row selection
    (partition_selection_digest) and the number of MAIN rows the runs were
    computed for: the runs are only used for a partition description with
    the same selection, on a MAIN table of the same size (partition_main_rows).

    Parameters
    ----------
    starts, lengths : np.ndarray
        First row and number of rows of every run (int64).
    bounds : np.ndarray
        Run range of every partition (int64, n_partitions + 1 values).
    digests : np.ndarray
        Selection digest of every partition (uint8, shape (n_partitions,
        SELECTION_DIGEST_SIZE)).
    main_nrows : int
        Rows of the MAIN table the runs were computed from.
    """

    __slots__ = ("starts", "lengths", "bounds", "digests", "main_nrows")

    def __init__(
        self,
        starts: np.ndarray,
        lengths: np.ndarray,
        bounds: np.ndarray,
        digests: np.ndarray,
        main_nrows: int,
    ):
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.bounds = np.asarray(bounds, dtype=np.int64)
        self.digests = np.asarray(digests, dtype=np.uint8).reshape(
            -1, SELECTION_DIGEST_SIZE
        )
        self.main_nrows = int(main_nrows)
        if self.starts.shape != self.lengths.shape:
            raise ValueError("Run starts and lengths differ in shape")
        if self.bounds.size != self.digests.shape[0] + 1:
            raise ValueError("One run range and one digest per partition expected")

    def __len__(self) -> int:
        return int(self.bounds.size - 1)

    def __getitem__(self, index: int) -> "PartitionMainRows":
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Partition {index} out of range ({len(self)})")
        return PartitionMainRows(self, index)

    @property
    def nbytes(self) -> int:
        """Memory held by the arrays."""
        return (
            self.starts.nbytes
            + self.lengths.nbytes
            + self.bounds.nbytes
            + self.digests.nbytes
        )

    def subset(self, indices: Sequence[int]) -> "MainRowRuns":
        """The runs of the partitions ``indices`` (copies), in that order."""
        indices = [operator.index(i) for i in indices]
        lo, hi = self.bounds[indices], self.bounds[np.asarray(indices) + 1]
        run_idx = (
            np.concatenate([np.arange(a, b) for a, b in zip(lo, hi, strict=True)])
            if indices
            else np.empty(0, dtype=np.int64)
        )
        return MainRowRuns(
            self.starts[run_idx],
            self.lengths[run_idx],
            np.concatenate(([0], np.cumsum(hi - lo))),
            self.digests[indices],
            self.main_nrows,
        )


class PartitionMainRows:
    """
    The MAIN rows of one partition of a MainRowRuns (``main_row_runs[i]``).

    A light reference to the shared runs; pickled (e.g. into a dask task) it
    carries only this partition's runs.
    """

    __slots__ = ("_runs", "_index")

    def __init__(self, runs: MainRowRuns, index: int):
        self._runs = runs
        self._index = index

    def __reduce__(self):
        return (PartitionMainRows, (self._runs.subset([self._index]), 0))

    def _bounds(self) -> tuple[int, int]:
        return (
            int(self._runs.bounds[self._index]),
            int(self._runs.bounds[self._index + 1]),
        )

    @property
    def starts(self) -> np.ndarray:
        """First row of every run (int64)."""
        lo, hi = self._bounds()
        return self._runs.starts[lo:hi]

    @property
    def lengths(self) -> np.ndarray:
        """Number of rows of every run (int64)."""
        lo, hi = self._bounds()
        return self._runs.lengths[lo:hi]

    @property
    def digest(self) -> bytes:
        """Selection digest of the partition description the runs belong to."""
        return self._runs.digests[self._index].tobytes()

    @property
    def main_nrows(self) -> int:
        """Rows of the MAIN table the runs were computed from."""
        return self._runs.main_nrows

    def rows(self) -> np.ndarray:
        """The MAIN row numbers (int64, ascending)."""
        return runs_to_rows(self.starts, self.lengths)

    def matches(self, partition_info: Mapping[str, Any], main_nrows: int) -> bool:
        """
        Whether the runs are those of ``partition_info`` (same row selection)
        on a MAIN table of ``main_nrows`` rows.
        """
        return main_nrows == self.main_nrows and (
            partition_selection_digest(partition_info) == self.digest
        )


def partition_main_rows(
    main_tb: tables.table,
    partition_info: dict,
    main_row_runs: PartitionMainRows | None = None,
) -> np.ndarray:
    """
    The MAIN row numbers of a partition, ascending: the rows that
    conversion.create_taql_query_where(partition_info) selects.

    Uses ``main_row_runs`` (create_partitions_with_main_rows) when they were
    computed for the same row selection (partition_selection_digest) on a
    MAIN table of the same size. Otherwise (no runs, a description changed
    after create_partitions_with_main_rows, e.g. a subset of its FIELD_IDs,
    or another MAIN table) the rows come from a numpy twin of the TaQL
    selection, which reads the key columns of the whole MAIN table once.

    Parameters
    ----------
    main_tb : tables.table
        The opened MAIN table (base table).
    partition_info : dict
        Partition description (as produced by create_partitions).
    main_row_runs : PartitionMainRows | None, optional
        The MAIN rows computed with the partition descriptions.

    Returns
    -------
    np.ndarray
        int64 MAIN row numbers of the partition.
    """
    if main_row_runs is not None:
        if main_row_runs.matches(partition_info, main_tb.nrows()):
            return main_row_runs.rows()
        xradio_logger().debug(
            "The MAIN row runs do not belong to this partition description (or "
            "MAIN table): selecting its rows from the key columns instead"
        )
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
