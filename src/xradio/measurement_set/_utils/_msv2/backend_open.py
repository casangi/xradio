"""
An MSv2 as a lazy processing set DataTree: the driver of the MSv2 xarray
backend (engine ``xradio_msv2``, ``_xradio_xarray_backends.MSv2BackendEntrypoint``).

The processing set is that of ``convert_msv2_to_processing_set`` (the same
partitions, MSv4 names, variables, values, attributes and encodings), with
the main data variables read lazily from the MS (see backend_partition.py and
backend_arrays.py). The partitions are computed on open (or taken from the
per-process memo, see partition_cache.py).
"""

import copy
import os
import time
import traceback
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
    SubtableCache,
    subtable_cache_supported,
)
from xradio.measurement_set._utils._msv2.backend_partition import open_partition
from xradio.measurement_set._utils._msv2.conversion import msv4_name
from xradio.measurement_set._utils._msv2.partition_cache import (
    PartitionsResult,
    load_or_create_partitions,
    resolve_partition_cache_mode,
)
from xradio.measurement_set._utils._msv2.partition_queries import (
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
    drop_variables = _check_drop_variables(drop_variables)
    if partition_filter is not None and not callable(partition_filter):
        raise TypeError(
            f"partition_filter must be a callable, got {type(partition_filter)}"
        )
    import xradio.measurement_set  # noqa: F401  (the xr_ps / xr_ms accessors)

    result = load_or_create_partitions(path, scheme, mode)
    selected = _select(path, result.partitions, partition_filter)
    _warn_large_tree(path, len(selected))
    build_options = {
        "main_chunksize": main_chunksize,
        "with_pointing": with_pointing,
        "pointing_chunksize": pointing_chunksize,
        "pointing_interpolate": pointing_interpolate,
        "ephemeris_interpolate": ephemeris_interpolate,
        "phase_cal_interpolate": phase_cal_interpolate,
        "sys_cal_interpolate": sys_cal_interpolate,
    }
    built = time.perf_counter()
    tree = _build_tree(
        path, result, selected, build_options, drop_variables, on_partition_error
    )
    end = time.perf_counter()
    xradio_logger().info(
        f"Opened {path} with the xradio_msv2 engine: {len(tree.children)} MSv4s of "
        f"{len(selected)} selected partitions ({len(result.partitions)} in all, "
        f"partition_scheme {scheme}, partitions {result.status}) in "
        f"{end - start:.2f} s (partitions {built - start:.2f} s, MSv4s "
        f"{end - built:.2f} s)"
    )
    return tree


def _check_drop_variables(drop_variables) -> list[str] | None:
    """drop_variables as a list of names (a str is one name)."""
    if drop_variables is None:
        return None
    if isinstance(drop_variables, str):
        return [drop_variables]
    try:
        names = list(drop_variables)
    except TypeError:
        raise TypeError(
            f"drop_variables must be a str or an iterable of str, got {drop_variables!r}"
        ) from None
    if not all(isinstance(name, str) for name in names):
        raise TypeError(f"drop_variables must hold names (str), got {names!r}")
    return names


def _select(
    path: str,
    partitions: list[dict],
    partition_filter: Callable[[dict[str, Any]], bool] | None,
) -> list[tuple[str, int]]:
    """
    The partitions to open, as (MSv4 id, partition index): those that
    ``partition_filter`` (called with a copy of every description) selects,
    numbered as the converter numbers them (zero-padded ids of the selected
    partitions).
    """
    if not partitions:
        raise RuntimeError(f"No partitions to open in {path} (no MAIN rows)")
    indices = list(range(len(partitions)))
    if partition_filter is not None:
        indices = [
            idx for idx in indices if partition_filter(copy.deepcopy(partitions[idx]))
        ]
        if not indices:
            raise RuntimeError("No partitions selected by partition_filter")
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
) -> xr.DataTree:
    """
    The processing set root and one MSv4 node per selected partition with
    MAIN rows (a partition without rows consumes its id and has no node, as
    in the converter). A partition that fails to open is left out (logged
    with its traceback and warned once) with on_partition_error="skip", and
    raises with "raise"; if no partition could be opened but some failed, a
    RuntimeError is raised.
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
                    **build_options,
                )
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
