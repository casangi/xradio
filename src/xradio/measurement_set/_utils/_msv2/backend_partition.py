"""
One partition of an MSv2 as a lazy MSv4 DataTree node, for the MSv2 xarray
backend (engine ``xradio_msv2``).

The MSv4 is built by the converter's own code (``conversion.build_partition``:
main xds, attributes and every sub-dataset, exactly as
``convert_msv2_to_processing_set`` builds it), with the data variables read
from MAIN columns left as placeholders. Those are then replaced by lazily
indexed arrays (``backend_arrays``) that read the MAIN table when they are
indexed. Opening reads no MAIN data column, and the MAIN table is closed
before the node is returned. The data variables of the pointing_xds are
lazy too (``backend_pointing``): opening reads the POINTING index columns
(TIME, ANTENNA_ID), not its values.
"""

import copy
import dataclasses
import functools
from collections.abc import Iterable, Sequence

import numpy as np
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
    partition_grouping,
)
from xradio.measurement_set._utils._msv2.backend_errors import StalePartitionsError
from xradio.measurement_set._utils._msv2.backend_pointing import (
    DeferredPointingVariable,
    PointingBuild,
    deferred_pointing_generic_xds,
    lazy_pointing_xds,
    rebuilt_pointing_xds,
)
from xradio.measurement_set._utils._msv2.conversion import build_partition
from xradio.measurement_set._utils._msv2.partition_queries import (
    MANDATORY_PARTITION_KEYS,
    PARTITION_MAIN_KEY_COLUMNS,
    PartitionKeyMaps,
    PartitionMainRows,
    describe_partition_rows,
)


@dataclasses.dataclass(frozen=True)
class RowCheck:
    """
    What verify_partition_rows needs to check a partition description taken
    from the partition cache against the partition's MAIN rows.

    Attributes
    ----------
    key_maps : PartitionKeyMaps
        partition_key_maps of the MS (read once per open).
    partition_scheme : Sequence[str]
        The partition scheme.
    """

    key_maps: PartitionKeyMaps
    partition_scheme: Sequence[str]


def verify_partition_rows(check: RowCheck, partition_info: dict, main_rows) -> None:
    """
    Check that a partition description from the partition cache describes
    the MAIN rows of the partition: their key columns (DATA_DESC_ID,
    OBSERVATION_ID, FIELD_ID, SCAN_NUMBER, STATE_ID; ANTENNA1 and ANTENNA2
    for ANTENNA1 schemes), and the keys derived from them, described as
    create_partitions describes a partition (describe_partition_rows).

    The descriptions must be equal. For ANTENNA1 schemes, whose partitions
    hold the autocorrelations of the rows with their keys (and are described
    by all of these rows), the grouping keys must be equal, the other values
    of the rows must be in the description, and every row must be an
    autocorrelation.

    With a scheme without ANTENNA1 and no partition filter, the cache's runs
    are disjoint and cover MAIN (the row checks of partition_cache), so
    descriptions that pass this check for every partition are those that
    create_partitions_with_main_rows gives for the MS now.

    Parameters
    ----------
    check : RowCheck
        The key maps and scheme.
    partition_info : dict
        The partition description.
    main_rows : MainTableRows
        The partition's rows (``BuiltPartition.main_rows``).

    Raises
    ------
    StalePartitionsError
        At the first difference.
    """
    scheme = list(check.partition_scheme)
    antenna1 = "ANTENNA1" in scheme
    names = [
        name for name in PARTITION_MAIN_KEY_COLUMNS if name != "ANTENNA1" or antenna1
    ]
    if antenna1:
        names.append("ANTENNA2")
    columns = {name: np.asarray(main_rows.getcol(name)) for name in names}
    if antenna1 and np.any(columns["ANTENNA1"] != columns["ANTENNA2"]):
        raise StalePartitionsError(
            "an ANTENNA1 partition has rows that are not autocorrelations"
        )
    described = describe_partition_rows(columns, check.key_maps, scheme)
    if list(described) != list(partition_info):
        raise StalePartitionsError(
            f"the description has the keys {list(partition_info)}, the rows "
            f"{list(described)}"
        )
    grouping = set(MANDATORY_PARTITION_KEYS) | set(scheme)
    for key, values in described.items():
        expected = partition_info[key]
        if values == expected:
            continue
        if antenna1 and key not in grouping and set(values) <= set(expected):
            continue
        raise StalePartitionsError(
            f"{key} {expected} in the description, {values} in the rows"
        )


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
    verify: RowCheck | None = None,
    lazy_pointing: bool = True,
    unreadable_columns: frozenset[str] = frozenset(),
    keys_token: str | None = None,
    partition_scheme: Sequence[str] = (),
    pointing_cache_mode: str | None = None,
) -> xr.DataTree | None:
    """
    The MSv4 of one partition of an MSv2, with lazy main data variables (and
    pointing_xds data variables, see backend_pointing.py).

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
    verify : RowCheck | None, optional
        Check the description against the partition's rows
        (verify_partition_rows): for partitions from the partition cache.
    lazy_pointing : bool, optional
        True (default): the data variables of the pointing_xds are read when
        indexed (with pointing_interpolate: read here, as by the converter);
        False: they are read here, as by the converter.
    unreadable_columns : frozenset[str], optional
        MAIN columns built as the converter builds a partition whose read of
        them failed (left out; WEIGHT_SPECTRUM: WEIGHT used instead).
    keys_token : str | None, optional
        ``backend_arrays.keys_token`` of the MS, taken before the partitions
        were computed (None: every read checks the partition against the MS).
    partition_scheme : Sequence[str], optional
        The partition scheme: with the mandatory partition keys, the keys
        that select the partition's rows when a read checks them
        (``backend_arrays.partition_grouping``).
    pointing_cache_mode : str | None, optional
        The ``partition_cache`` mode of the open, for the cell shapes of the
        POINTING table stored in the MS (``backend_pointing.open_pointing_index``;
        None: neither used nor stored).

    Returns
    -------
    xr.DataTree | None
        The MSv4, or None if the partition has no MAIN rows.

    Raises
    ------
    StalePartitionsError
        If ``verify`` finds that the description does not describe the rows.
    """
    # descriptions of the lazy pointing_xds data variables, by name, and the
    # partition's time range and antennas (as the loader got them)
    pointing_specs: dict[str, DeferredPointingVariable] = {}
    pointing_context: dict = {}
    pointing_loader = (
        functools.partial(
            deferred_pointing_generic_xds,
            specs=pointing_specs,
            context=pointing_context,
            cache_mode=pointing_cache_mode,
        )
        if lazy_pointing
        else None
    )
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
            unreadable_columns=frozenset(unreadable_columns),
            defer_main_columns=True,
            pointing_generic_loader=pointing_loader,
        ) as built,
    ):
        if built is None:
            return None
        if verify is not None:
            verify_partition_rows(verify, partition_info, built.main_rows)
        index = PartitionIndex.seed(
            in_file,
            built,
            keys_token,
            partition_grouping(partition_info, tuple(partition_scheme)),
            tuple(partition_scheme),
        )
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
            array = OnesArray(var.shape, var.dtype, index)
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
        if pointing_specs:
            pointing_xds = lazy_pointing_xds(
                pointing_xds, pointing_specs, node_name, partition=index
            )
        elif pointing_context and pointing_xds.data_vars:
            # built here by the converter's code (a POINTING table the lazy
            # reads cannot describe: values not kept), or described from the
            # cell shapes (placeholders: a partition with fewer than 1,000
            # POINTING rows, whose cells of other shapes the converter pads):
            # built again by the converter's code when read
            pointing_xds = _rebuilt_pointing(
                in_file, ms_xdt, pointing_xds, pointing_context, node_name, index
            )
        _check_no_placeholder_left(pointing_xds)
        _set_preferred_chunks(pointing_xds)
        ms_xdt["pointing_xds"].dataset = pointing_xds
    if drop_variables is not None:
        _drop_variables(ms_xdt, drop_variables)
    return ms_xdt


def _rebuilt_pointing(
    in_file: str,
    ms_xdt: xr.DataTree,
    pointing_xds: xr.Dataset,
    context: dict,
    node_name: str,
    partition: PartitionIndex | None = None,
) -> xr.Dataset:
    """The pointing_xds of a partition (built at open by the converter's
    code, or placeholders described from the cell shapes of its POINTING
    rows) with data variables that build it by the converter's code when
    read (rebuilt_pointing_xds); as it is if the antennas of the build
    cannot be told (kept eager; placeholders are then left, which
    _check_no_placeholder_left reports)."""
    antenna_ids = context["antenna_ids"]
    names = (
        ms_xdt["antenna_xds"].to_dataset(inherit=False)["antenna_name"].values
        if "antenna_xds" in ms_xdt.children
        else []
    )
    if len(names) != len(antenna_ids):
        return pointing_xds
    build = PointingBuild(
        in_file,
        context["time_min_max"],
        antenna_ids,
        tuple(str(name) for name in names),
    )
    return rebuilt_pointing_xds(pointing_xds, build, node_name, partition)


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
