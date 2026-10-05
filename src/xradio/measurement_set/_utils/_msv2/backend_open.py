"""
An MSv2 as a lazy processing set DataTree: the driver of the MSv2 xarray
backend (engine ``xradio_msv2``, ``_xradio_xarray_backends.MSv2BackendEntrypoint``).

The processing set is that of ``convert_msv2_to_processing_set`` (the same
partitions, MSv4 names, variables, values, attributes and encodings), with
the main data variables read lazily from the MS (see backend_partition.py and
backend_arrays.py). The partitions are computed on the first open and
stored in the MS (its XRADIO_PARTITIONS sub-table), then read from it or
from a per-process memo while the MS is unchanged (see partition_cache.py).
"""

import copy
import os
import time
import traceback
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import xarray as xr

from xradio._utils._casacore.tables import casatools_serialized
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
    SubtableCache,
    subtable_cache_supported,
)
from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
    read_table_lock,
    resync_unless_write_locked,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.backend_arrays import (
    GRID_KEY_COLUMNS,
    keys_token,
)
from xradio.measurement_set._utils._msv2.backend_errors import (
    MainRowsChangedError,
    MSv2ChangedError,
    PartitionCacheWarning,
    StalePartitionsError,
)
from xradio.measurement_set._utils._msv2.backend_partition import (
    RowCheck,
    open_partition,
)
from xradio.measurement_set._utils._msv2.conversion import msv4_name
from xradio.measurement_set._utils._msv2.partition_cache import (
    MAIN_NOT_FLUSHED,
    PARTITIONS_MEMO,
    PartitionsResult,
    compute_in_memory,
    load_or_create_partitions,
    resolve_partition_cache_mode,
)
from xradio.measurement_set._utils._msv2.partition_queries import (
    partition_key_maps,
    validate_partition_scheme,
)

# The values of on_partition_error
ON_PARTITION_ERROR = ("skip", "raise")
# Opening more partitions than this warns (each costs about 0.05-0.2 s to
# build, plus 3-12 ms and 70-200 kB per node in xarray)
LARGE_TREE_PARTITIONS = 1000


def open_msv2_tree(
    in_file: str | os.PathLike,
    *,
    drop_variables: str | Iterable[str] | None = None,
    partition_scheme: Iterable[str] | None = None,
    partition_filter: Callable[[dict[str, Any]], bool] | None = None,
    main_chunksize: dict | float | None = None,
    with_pointing: bool = True,
    pointing_chunksize: dict | float | None = None,
    pointing_interpolate: bool = False,
    ephemeris_interpolate: bool = False,
    phase_cal_interpolate: bool = False,
    sys_cal_interpolate: bool = False,
    partition_cache: str | None = None,
    on_partition_error: str = "skip",
    skip_columns: str | Iterable[str] | None = None,
) -> xr.DataTree:
    """
    Open an MSv2 as a processing set DataTree of lazy MSv4s (see
    :func:`xradio.measurement_set.open_msv2` for the parameters).

    Returns
    -------
    xr.DataTree
        The processing set: a root of type "processing_set" with one MSv4
        node per opened partition.
    """
    start = time.perf_counter()
    # absolute: the lazy arrays read the MS by path (also after a chdir)
    path = os.path.abspath(os.path.expanduser(os.fspath(in_file)))
    scheme = validate_partition_scheme(partition_scheme)
    mode = resolve_partition_cache_mode(partition_cache)
    if on_partition_error not in ON_PARTITION_ERROR:
        raise ValueError(
            f"on_partition_error must be one of {list(ON_PARTITION_ERROR)}, got "
            f"{on_partition_error!r}"
        )
    drop_variables = _check_names("drop_variables", drop_variables)
    skip_columns = _check_names("skip_columns", skip_columns)
    if partition_filter is not None and not callable(partition_filter):
        raise TypeError(
            f"partition_filter must be a callable, got {type(partition_filter)}"
        )
    import xradio.measurement_set  # noqa: F401  (the xr_ps / xr_ms accessors)

    build_options = {
        "main_chunksize": main_chunksize,
        "with_pointing": with_pointing,
        "pointing_chunksize": pointing_chunksize,
        "pointing_interpolate": pointing_interpolate,
        "ephemeris_interpolate": ephemeris_interpolate,
        "phase_cal_interpolate": phase_cal_interpolate,
        "sys_cal_interpolate": sys_cal_interpolate,
        "unreadable_columns": frozenset(skip_columns or ()),
    }
    for attempt in (1, 2):
        # (a MAIN table that this process holds open with another number of
        # rows than its files is re-synchronized with them first, unless the
        # process holds its write lock: then its rows are those of the
        # process, and the partitions are computed from them)
        if _check_main_is_current(path):
            result = compute_in_memory(path, scheme, MAIN_NOT_FLUSHED)
        else:
            result = load_or_create_partitions(path, scheme, mode)
        selected = _select(path, result.partitions, partition_filter)
        if attempt == 1:
            _warn_large_tree(path, len(selected))
        built = time.perf_counter()
        # (taken before the builds read the keys of the rows)
        with casatools_serialized():
            build_options["keys_token"] = keys_token(path, GRID_KEY_COLUMNS)
        try:
            _check_main_is_current(path, result.main_nrows)
            # partitions from the cache are checked against their rows
            verify = None
            if result.source in ("stored", "memo"):
                # (reads FIELD, SOURCE and STATE: casatools tables are used by
                # one thread at a time, a no-op with python-casacore)
                with casatools_serialized():
                    verify = RowCheck(partition_key_maps(path), scheme)
            tree = _build_tree(
                path,
                result,
                selected,
                build_options,
                drop_variables,
                on_partition_error,
                verify,
            )
            break
        except StalePartitionsError as exc:
            if attempt == 2:
                raise MSv2ChangedError(
                    f"{path} changed while it was opened ({exc}); open it again"
                ) from exc
            if result.source == "fresh" or isinstance(exc, MainRowsChangedError):
                # (a change made while the MS was opened, not a cache defect)
                xradio_logger().info(
                    f"{path} changed while it was opened ({exc}): opening it again"
                )
            else:
                warnings.warn(
                    f"The partition cache of {path} did not describe its rows "
                    f"although its staleness checks passed ({exc}); the partitions "
                    "are computed again. Please report this.",
                    PartitionCacheWarning,
                    stacklevel=4,
                )
            PARTITIONS_MEMO.discard_path(path)
            mode = "rebuild" if mode in ("auto", "rebuild") else "off"
    end = time.perf_counter()
    xradio_logger().info(
        f"Opened {path} with the xradio_msv2 engine: {len(tree.children)} MSv4s of "
        f"{len(selected)} selected partitions ({len(result.partitions)} in all, "
        f"partition_scheme {scheme}, partitions {result.status}) in "
        f"{end - start:.2f} s (partitions {built - start:.2f} s, MSv4s "
        f"{end - built:.2f} s)"
    )
    return tree


def _check_main_is_current(path: str, expected_nrows: int | None = None) -> bool:
    """
    Check that this process sees the MAIN table as it is on disk (or as the
    process itself changed it), and has as many rows as the partitions were
    computed for.

    casacore shares one table object per table in a process: if this
    process holds MAIN open (e.g. the user's handle), a new open gets that
    object, whose number of rows does not follow rows that other processes
    added. It is re-synchronized when it differs from the lock file, unless
    the process holds MAIN's write lock: a re-synchronization would drop
    the changes of the process that are not flushed yet (e.g. rows that the
    user's writable handle added), so the rows the process sees are used
    then, as the converter would read them.

    Parameters
    ----------
    path : str
        Path of the MS.
    expected_nrows : int | None, optional
        The MAIN rows of the partitions.

    Returns
    -------
    bool
        Whether the rows of MAIN in this process are not those on disk (the
        process holds MAIN's write lock, with another number of rows): the
        partitions must then be computed from them, without the partition
        cache (whose fingerprint is that of the files).

    Raises
    ------
    MSv2ChangedError
        If MAIN cannot be re-synchronized with its files.
    MainRowsChangedError
        If MAIN does not have ``expected_nrows`` rows.
    """
    lock = read_table_lock(path)
    on_disk = lock.nrrow if lock is not None and lock.lock_ok else None
    own_rows = False
    with casatools_serialized(), open_table_ro(path) as main_tb:
        nrows = main_tb.nrows()
        if on_disk is not None and nrows != on_disk:
            try:
                own_rows = not resync_unless_write_locked(main_tb)
            except RuntimeError as exc:
                raise MSv2ChangedError(
                    f"The MAIN table of {path} changed and cannot be re-read: {exc}"
                ) from exc
            nrows = main_tb.nrows()
    if on_disk is not None and nrows != on_disk and not own_rows:
        raise MSv2ChangedError(
            f"The MAIN table of {path} has {nrows} rows in this process and "
            f"{on_disk} on disk"
        )
    if expected_nrows is not None and nrows != expected_nrows:
        raise MainRowsChangedError(
            f"the MAIN table has {nrows} rows, the partitions are of {expected_nrows}"
        )
    return own_rows


def _check_names(option: str, names) -> list[str] | None:
    """drop_variables / skip_columns as a list of names (a str is one
    name)."""
    if names is None:
        return None
    if isinstance(names, str):
        return [names]
    try:
        listed = list(names)
    except TypeError:
        raise TypeError(
            f"{option} must be a str or an iterable of str, got {names!r}"
        ) from None
    if not all(isinstance(name, str) for name in listed):
        raise TypeError(f"{option} must hold names (str), got {listed!r}")
    return listed


def _select(
    path: str,
    partitions: list[dict],
    partition_filter: Callable[[dict[str, Any]], bool] | None,
) -> list[tuple[str, int]]:
    """
    The partitions to open, as (MSv4 id, partition index): those that
    ``partition_filter`` (called with a copy of every description) selects,
    numbered as the converter numbers them (zero-padded ids of the selected
    partitions). As in the converter, an MS without MAIN rows (no
    partitions) gives none, an empty processing set, and a filter that
    selects none raises.
    """
    indices = list(range(len(partitions)))
    if partition_filter is not None:
        indices = [
            idx for idx in indices if partition_filter(copy.deepcopy(partitions[idx]))
        ]
        if not indices:
            raise RuntimeError("No partitions selected by partition_filter")
    if not indices:
        xradio_logger().info(f"{path} has no MAIN rows: an empty processing set")
        return []
    width = len(str(len(indices) - 1))
    return [(f"{ms_v4_id:0>{width}}", idx) for ms_v4_id, idx in enumerate(indices)]


def _warn_large_tree(path: str, n_selected: int) -> None:
    if n_selected > LARGE_TREE_PARTITIONS:
        warnings.warn(
            f"Opening {n_selected} partitions of {path}: every partition costs "
            "about 0.05-0.2 s to build plus 3-12 ms and 70-200 kB per DataTree node. "
            "partition_scheme=[] or a partition_filter opens fewer.",
            UserWarning,
            stacklevel=4,
        )


def _describe(partition: dict) -> str:
    keys = ("DATA_DESC_ID", "OBSERVATION_ID", "FIELD_ID", "SCAN_NUMBER", "STATE_ID")
    return ", ".join(f"{key} {partition.get(key)}" for key in keys if key in partition)


def _build_tree(
    path: str,
    result: PartitionsResult,
    selected: list[tuple[str, int]],
    build_options: dict,
    drop_variables: list[str] | None,
    on_partition_error: str,
    verify: RowCheck | None = None,
) -> xr.DataTree:
    """
    The processing set root and one MSv4 node per selected partition with
    MAIN rows (a partition without rows consumes its id and has no node, as
    in the converter). A partition that fails to open is left out (logged
    with its traceback and warned once) with on_partition_error="skip", and
    raises with "raise"; if no partition could be opened but some failed, a
    RuntimeError is raised. With ``verify``, every partition's description
    is checked against its rows: StalePartitionsError (whatever
    on_partition_error).
    """
    root = xr.DataTree()
    root.attrs["type"] = "processing_set"
    # Sub-table data read once and shared by the partitions (python-casacore)
    subtable_cache = (
        SubtableCache(n_partitions=len(selected))
        if subtable_cache_supported()
        else None
    )
    failed: list[tuple[str, BaseException]] = []
    try:
        for ms_v4_id, idx in selected:
            partition = result.partitions[idx]
            name = msv4_name(path, ms_v4_id)
            xradio_logger().debug(
                f"Opening partition {idx} ({_describe(partition)}) as {name}"
            )
            try:
                node = open_partition(
                    path,
                    partition,
                    result.runs[idx],
                    node_name=name,
                    drop_variables=drop_variables,
                    subtable_cache=subtable_cache,
                    verify=verify,
                    **build_options,
                )
            except StalePartitionsError:
                raise
            except Exception as exc:
                what = f"Partition {idx} ({name}) of {path} could not be opened"
                if on_partition_error == "raise":
                    raise RuntimeError(f"{what}: {type(exc).__name__}: {exc}") from exc
                xradio_logger().error(
                    f"{what}, it is left out: {exc!r}\n{traceback.format_exc()}"
                )
                warnings.warn(
                    f"{what} and is left out of the processing set: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=4,
                )
                failed.append((name, exc))
                continue
            if node is not None:
                root[name] = node
    finally:
        # also after a failure: a kept traceback must not keep the cache alive
        if subtable_cache is not None:
            subtable_cache.clear()
    if failed and not root.children:
        raise RuntimeError(
            f"None of the {len(selected)} selected partitions of {path} could be "
            f"opened (on_partition_error='skip'); the first error: "
            f"{type(failed[0][1]).__name__}: {failed[0][1]}"
        ) from failed[0][1]
    return root
