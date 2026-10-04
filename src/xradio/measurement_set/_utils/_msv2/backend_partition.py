"""
One partition of an MSv2 as a lazy MSv4 DataTree node, for the MSv2 xarray
backend (engine ``xradio_msv2``).

The MSv4 is built by the converter's own code (``conversion.build_partition``:
main xds, attributes and every sub-dataset, exactly as
``convert_msv2_to_processing_set`` builds it), with the data variables read
from MAIN columns left as placeholders. Those are then replaced by lazily
indexed arrays (``backend_arrays``) that read the MAIN table when they are
indexed. Opening reads no MAIN data column, and the MAIN table is closed
before the node is returned.
"""

import copy
from collections.abc import Iterable

import xarray as xr

from xradio._utils._casacore.tables import casatools_serialized
from xradio._utils.xarray_helpers import (
    remove_variable_references,
    remove_variables_from_data_groups,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import SubtableCache
from xradio.measurement_set._utils._msv2.backend_arrays import (
    MSv2MainColumnArray,
    OnesArray,
    PartitionIndex,
)
from xradio.measurement_set._utils._msv2.conversion import build_partition
from xradio.measurement_set._utils._msv2.partition_queries import PartitionMainRows


def open_partition(
    in_file: str,
    partition_info: dict,
    main_row_runs: PartitionMainRows | None = None,
    *,
    node_name: str = "",
    drop_variables: str | Iterable[str] | None = None,
    subtable_cache: SubtableCache | None = None,
    main_chunksize: dict | float | None = None,
    with_pointing: bool = True,
    pointing_chunksize: dict | float | None = None,
    pointing_interpolate: bool = False,
    ephemeris_interpolate: bool = False,
    phase_cal_interpolate: bool = False,
    sys_cal_interpolate: bool = False,
) -> xr.DataTree | None:
    """
    The MSv4 of one partition of an MSv2, with lazy main data variables.

    The node equals what ``convert_msv2_to_processing_set`` writes for the
    partition (opened with ``open_processing_set``), with the same encoding
    (chunks, compressors: ``to_zarr`` writes the converter's MSv4) plus
    ``preferred_chunks`` (see ``_set_preferred_chunks``).

    Parameters
    ----------
    in_file : str
        Absolute path of the MSv2 (the lazy arrays read it by this path).
    partition_info : dict
        The partition description (create_partitions).
    main_row_runs : PartitionMainRows | None, optional
        The partition's MAIN rows (create_partitions_with_main_rows).
    node_name : str, optional
        Name of the MSv4 node (for the messages of failed reads).
    drop_variables : str | Iterable[str] | None, optional
        Variables to leave out, from every node of the MSv4 (a str is one
        name; names not in a node are ignored); the data group roles that
        name them are removed too.
    subtable_cache : SubtableCache | None, optional
        Sub-table data shared by the partitions of one open.
    main_chunksize, with_pointing, pointing_chunksize, pointing_interpolate,
    ephemeris_interpolate, phase_cal_interpolate, sys_cal_interpolate :
        As for ``convert_msv2_to_processing_set``.

    Returns
    -------
    xr.DataTree | None
        The MSv4, or None if the partition has no MAIN rows.
    """
    # casatools tables are used by one thread at a time (a no-op with
    # python-casacore)
    with (
        casatools_serialized(),
        build_partition(
            in_file,
            partition_info,
            main_chunksize=main_chunksize,
            with_pointing=with_pointing,
            pointing_chunksize=pointing_chunksize,
            pointing_interpolate=pointing_interpolate,
            ephemeris_interpolate=ephemeris_interpolate,
            phase_cal_interpolate=phase_cal_interpolate,
            sys_cal_interpolate=sys_cal_interpolate,
            parallel_mode="none",
            subtable_cache=subtable_cache,
            main_row_runs=main_row_runs,
            defer_main_columns=True,
        ) as built,
    ):
        if built is None:
            return None
        index = PartitionIndex.seed(in_file, built)
        ms_xdt, deferred = built.ms_xdt, built.deferred
        reverse_frequency = built.reverse_frequency
    # The MAIN table is closed here: nothing below refers to it.

    xds = ms_xdt.to_dataset(inherit=False)
    lazy_variables = {}
    for name, spec in deferred.items():
        if name not in xds.data_vars:  # e.g. UVW, dropped for single dish
            continue
        var = xds.variables[name]
        if spec.col is None:
            array = OnesArray(var.shape, var.dtype)
        else:
            array = MSv2MainColumnArray.from_spec(
                index, spec, var.shape, var.dtype, node=node_name
            )
        lazy = xr.Variable(
            var.dims,
            xr.core.indexing.LazilyIndexedArray(array),
            attrs=copy.deepcopy(var.attrs),
        )
        # The arrays are in the channel order of the MSv2: a decreasing
        # frequency axis is reversed lazily (the arrays then see positive-step
        # slices). Variables constant along frequency (WEIGHT repeated from the
        # WEIGHT column, the WEIGHT=1 fallback) are not reversed, as in the
        # converter (stream_write); either way gives the same values.
        if (
            reverse_frequency
            and "frequency" in var.dims
            and not spec.frequency_constant
        ):
            lazy = lazy.isel(frequency=slice(None, None, -1))
        lazy.encoding = dict(var.encoding)
        lazy_variables[name] = lazy
    # (variables replaced in place: the order of the data variables is kept)
    xds = xds.assign(lazy_variables)
    _check_no_placeholder_left(xds)
    _set_preferred_chunks(xds)
    ms_xdt.dataset = xds
    if "pointing_xds" in ms_xdt.children:
        pointing_xds = ms_xdt["pointing_xds"].to_dataset(inherit=False)
        _set_preferred_chunks(pointing_xds)
        ms_xdt["pointing_xds"].dataset = pointing_xds
    if drop_variables is not None:
        _drop_variables(ms_xdt, drop_variables)
    return ms_xdt


def _is_placeholder(var: xr.Variable) -> bool:
    """Whether a variable is (a view of) a placeholder of the converter's
    streamed write (stream_write.deferred_placeholder)."""
    if var.chunks is None:  # not a dask array (var.data would load a lazy one)
        return False
    graph = getattr(var.data, "__dask_graph__", lambda: None)()
    if graph is None:
        return False
    return any(
        str(key[0] if isinstance(key, tuple) else key).startswith("deferred-")
        for key in graph
    )


def _check_no_placeholder_left(xds: xr.Dataset) -> None:
    """Raise if a placeholder survived (a variable of the build that has no
    lazy array): placeholders raise when computed."""
    left = sorted(
        str(name) for name, var in xds.variables.items() if _is_placeholder(var)
    )
    if left:
        raise RuntimeError(
            f"The data variables {left} of the MSv4 have no lazy array (placeholders "
            "of the build left)"
        )


def _set_preferred_chunks(xds: xr.Dataset) -> None:
    """
    Set ``encoding["preferred_chunks"]`` (in place), which ``chunks={}``
    uses as dask chunks:

    - variables with zarr chunks in their encoding (the data variables:
      the converter's main_chunksize / default_main_chunksize, or
      pointing_chunksize): those chunks, so that the dask chunks are the
      chunks of the converted MSv4;
    - the other non-index variables along time (coordinates such as
      scan_name and field_name): the time chunk of the data variables, so
      that every variable of the dataset has the same chunks along time
      (``Dataset.chunks`` is defined).
    """
    time_chunk = None
    for name, var in xds.variables.items():
        chunks = var.encoding.get("chunks")
        if chunks is None or name in xds.indexes:
            continue
        var.encoding["preferred_chunks"] = dict(zip(var.dims, chunks, strict=True))
        if "time" in var.dims and time_chunk is None:
            time_chunk = int(chunks[var.dims.index("time")])
    if time_chunk is None:
        return
    for name, var in xds.variables.items():
        if (
            name not in xds.indexes
            and "time" in var.dims
            and "preferred_chunks" not in var.encoding
        ):
            var.encoding["preferred_chunks"] = {"time": time_chunk}


def _drop_variables(ms_xdt: xr.DataTree, drop_variables: str | Iterable[str]) -> None:
    """
    Leave ``drop_variables`` out of every node of an MSv4 (in place), as
    xarray's backends do (a str is one name, names not in a node are
    ignored), with the data group roles and variable attributes that refer
    to them.
    """
    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    drop_variables = list(drop_variables)
    for node in ms_xdt.subtree:
        xds = node.to_dataset(inherit=False)
        names = [name for name in drop_variables if name in xds.variables]
        if not names:
            continue
        xds = xds.drop_vars(names)
        remove_variables_from_data_groups(xds, names)
        remove_variable_references(xds, names)
        node.dataset = xds
