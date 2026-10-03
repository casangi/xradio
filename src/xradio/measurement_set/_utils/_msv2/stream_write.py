"""
Streamed write of the large MAIN-derived data variables of an MSv4.

Without streaming, ``convert_and_write_partition`` reads every MAIN column of a
partition into a dense (time, baseline, ...) grid and keeps all of them in memory
until one ``DataTree.to_zarr`` call writes the MSv4. With streaming:

1. The MSv4 is built as before (coordinates, attributes, small variables and
   sub-datasets), but the data variables that come from MAIN columns
   (VISIBILITY*, SPECTRUM, WEIGHT, FLAG, UVW, TIME_CENTROID,
   EFFECTIVE_INTEGRATION_TIME and the WEIGHT=1 fallback) are lazy placeholders
   (``deferred_main_column``, ``deferred_ones``). Whether a column can be read is
   decided here, before anything is written, with the rule of the read path:
   the column is skipped (WEIGHT_SPECTRUM falls back to WEIGHT) if any cell of
   the partition is undefined or has a shape other than the first cell's
   (``check_partition_cells``).
2. ``DataTree.to_zarr(compute=False)`` writes all metadata (with the encoding,
   chunks and compressor of the non-streamed path), the numpy variables and the
   consolidated metadata. The placeholders are never computed.
3. ``write_deferred_variables`` fills the placeholders one variable at a time,
   in batches of whole zarr chunks along time: each batch is read with one
   ascending pass over its rows (``read_rows_to_grid``), converted as in the
   non-streamed path (TIME_CENTROID epoch, WEIGHT repeated along frequency,
   reversed frequency axis) and written as a chunk-aligned region, so every
   zarr chunk is encoded once, from the same values: the chunk files are
   byte-identical to the non-streamed path.

Batch size: as many whole time chunks as fit ``XRADIO_MSV2_STREAM_BATCH_MB``
(uncompressed, at least one chunk); a variable smaller than that is one batch.
Reading by time batches costs one extra row run per batch on time-ordered
partitions. When the partition's rows of a time batch are scattered over many
runs (baseline-major or interleaved row order), batching would multiply the
number of reads (and tile re-reads): such a variable is read in one pass if it
fits ``FRAGMENTED_BATCH_FACTOR`` batches, otherwise in batches of that size
(``choose_time_batches``).

If reading or writing fails after a variable has been partly written, the
variable is removed from the store (and the consolidated metadata rewritten)
and the exception is raised: the MSv4 is then incomplete, never silently
half-written (the non-streamed path would have dropped a variable whose read
failed, which cannot be reproduced once the metadata is written).
"""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import dask
import dask.array as da
import numpy as np
import xarray as xr

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from xradio._utils.list_and_array import get_pad_value
from xradio._utils.logging import xradio_logger
from xradio._utils.zarr.config import ZARR_FORMAT
from xradio.measurement_set._utils._msv2._tables.read import (
    _partition_cell_shape_and_dtype,
)
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    DEFAULT_MAX_TMP_BYTES,
    FRAGMENTED_RUNS,
    MainTableRows,
    TimeChunkRows,
    column_dtype,
    make_row_grid_plan,
    read_rows_to_grid,
    rows_to_runs,
)

# TEMPORARY, EXPLORATION ONLY (remove before merging): "1" (default) writes the
# MAIN data variables with the streamed write, "0" reads them all into memory
# and writes the MSv4 with one to_zarr call (the previous path), for A/B
# benchmarks. The streamed write needs the row read path
# (XRADIO_MSV2_MAIN_READ=rows) and parallel_mode "none" or "partition".
STREAM_WRITE_ENV_VAR = "XRADIO_MSV2_STREAM_WRITE"
# TEMPORARY, EXPLORATION ONLY (remove before merging): target size of a batch
# (MiB, uncompressed), to sweep the batch size in benchmarks.
STREAM_BATCH_MB_ENV_VAR = "XRADIO_MSV2_STREAM_BATCH_MB"
DEFAULT_STREAM_BATCH_MB = 128
# Small-read guard: time batching is "fragmenting" when it reads more than this
# many times the row runs of a one-pass read (plus one run per batch boundary).
FRAGMENTED_RUNS_RATIO = 2.0
# A fragmenting variable is read in one pass if it is at most this many target
# batches, otherwise in batches of that size.
FRAGMENTED_BATCH_FACTOR = 8
# Rows per getcolshapestring call of the cell shape scan.
SHAPE_SCAN_ROWS = 2**16

# casacore ColumnDesc option bit of arrays stored in the row (fixed shape, always
# defined)
_DIRECT_OPTION = 1


def get_stream_write_mode() -> bool:
    """
    TEMPORARY, EXPLORATION ONLY: whether the streamed write of the MAIN data
    variables is selected (environment variable XRADIO_MSV2_STREAM_WRITE, "1"
    by default, or "0").

    Returns
    -------
    bool
        True for the streamed write.
    """
    value = os.environ.get(STREAM_WRITE_ENV_VAR, "").strip() or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{STREAM_WRITE_ENV_VAR}={value!r} is not '0' or '1'")
    return value == "1"


def get_stream_batch_bytes() -> int:
    """
    TEMPORARY, EXPLORATION ONLY: target batch size of the streamed write, from
    the environment variable XRADIO_MSV2_STREAM_BATCH_MB (MiB, default
    DEFAULT_STREAM_BATCH_MB).

    Returns
    -------
    int
        Target batch size in bytes (uncompressed).
    """
    value = os.environ.get(STREAM_BATCH_MB_ENV_VAR, "").strip()
    try:
        mib = float(value) if value else float(DEFAULT_STREAM_BATCH_MB)
    except ValueError:
        mib = float("nan")
    if not mib > 0:
        raise ValueError(f"{STREAM_BATCH_MB_ENV_VAR}={value!r} is not a size > 0")
    return max(1, int(mib * 2**20))


class ColumnNotReadableError(RuntimeError):
    """A column whose cells cannot all be read for a partition."""


def _check_shape_strings(shapes: list[str], expected: str, col: str) -> None:
    if len(set(shapes)) > 1 or (shapes and shapes[0] != expected):
        raise ColumnNotReadableError(
            f"Column {col} has cells of a shape other than {expected} in the partition"
        )


def _scan_cell_shapes(
    table: tables.table, col: str, rows: np.ndarray, expected: str
) -> None:
    """
    Compare the shape of every cell of ``rows`` with ``expected`` (shape
    strings), with one getcolshapestring call per run of rows, or per batch of
    rows read through ``selectrows`` when a batch has many runs. Raises
    ColumnNotReadableError for an undefined cell or another shape.
    """
    table_nrows = table.nrows()
    for b0 in range(0, rows.size, SHAPE_SCAN_ROWS):
        batch = rows[b0 : b0 + SHAPE_SCAN_ROWS]
        starts, lengths = rows_to_runs(batch)
        try:
            if starts.size > FRAGMENTED_RUNS:
                ref = table.selectrows(batch)
                try:
                    _check_shape_strings(ref.getcolshapestring(col), expected, col)
                finally:
                    ref.close()
                continue
            for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
                pieces = [(start, length)]
                if start == 0 and length == table_nrows and length > 1:
                    # never a call over the whole column (see read_rows._read_run)
                    pieces = [(0, length - 1), (length - 1, 1)]
                for row0, nrow in pieces:
                    _check_shape_strings(
                        table.getcolshapestring(col, row0, nrow), expected, col
                    )
        except RuntimeError as exc:
            if isinstance(exc, ColumnNotReadableError):
                raise
            raise ColumnNotReadableError(
                f"Column {col} has undefined cells in the partition: {exc}"
            ) from exc


def check_partition_cells(table: tables.table, col: str, rows: np.ndarray) -> str:
    """
    Check, without reading any data, that every cell of a column can be read
    for the rows of a partition: every cell defined and of the shape of the
    first one. These are the cells for which the row read path
    (``read_col_conversion_rows``) succeeds; it raises (and the converter skips
    the column) otherwise. Decided in O(1) from the storage manager where it
    can be:

    - scalar columns, TiledColumnStMan columns and arrays stored in the row
      ("direct") are always defined with one shape;
    - for a TiledShapeStMan column, if its hypercubes hold every row of the
      table and all have the cell shape of the partition's first cell.

    Otherwise (other storage managers, or hypercubes that do not hold every row
    or have several cell shapes), the cell shapes of the partition rows are
    compared (``getcolshapestring``, no data read).

    Parameters
    ----------
    table : tables.table
        Base MAIN table.
    col : str
        Column name.
    rows : np.ndarray
        MAIN rows of the partition (strictly increasing, not empty).

    Returns
    -------
    str
        How it was decided (for logging).

    Raises
    ------
    ColumnNotReadableError
        If a cell of the partition is undefined or has another shape.
    """
    if table.isscalarcol(col):
        return "scalar column"
    rows = np.asarray(rows, dtype=np.int64)
    try:
        expected = table.getcolshapestring(col, int(rows[0]), 1)[0]
    except RuntimeError as exc:
        raise ColumnNotReadableError(
            f"Column {col}: the first cell of the partition is undefined"
        ) from exc
    option = int(table.getcoldesc(col).get("option", 0))
    dminfo = table.getdminfo(col)
    dm_type = dminfo.get("TYPE", "")
    if dm_type == "TiledColumnStMan":
        return "TiledColumnStMan (fixed shape)"
    if dm_type == "TiledShapeStMan":
        cubes = list(dminfo.get("SPEC", {}).get("HYPERCUBES", {}).values())
        rows_in_cubes = sum(
            int(np.asarray(cube["CubeShape"])[-1])
            for cube in cubes
            if np.asarray(cube.get("CubeShape", [])).size
        )
        # cube cell shapes are in Fortran order, the shape strings in numpy order
        cell_shapes = {
            str(list(np.asarray(cube["CellShape"]).tolist()[::-1]))
            if "CellShape" in cube
            else None
            for cube in cubes
        }
        if rows_in_cubes >= table.nrows() and (
            cell_shapes == {expected} or (len(cubes) == 1 and None in cell_shapes)
        ):
            return "TiledShapeStMan, all rows in hypercubes of one cell shape"
    elif option & _DIRECT_OPTION:
        return f"{dm_type} direct array (fixed shape)"
    _scan_cell_shapes(table, col, rows, expected)
    return f"{dm_type}: cell shapes of {rows.size} rows compared"


def _deferred_block_computed(name: str) -> None:
    raise RuntimeError(
        f"The deferred data variable {name} was computed: it is written by "
        "write_deferred_variables, after the MSv4 metadata"
    )


def deferred_placeholder(name: str, shape: tuple[int, ...], dtype) -> da.Array:
    """
    A lazy (dask) array of a given shape and dtype that stands for a data
    variable written later (one chunk; it raises if it is ever computed).

    Parameters
    ----------
    name : str
        Data variable name (for the error message).
    shape : tuple[int, ...]
        Shape.
    dtype : np.dtype
        dtype.

    Returns
    -------
    da.Array
        The placeholder.
    """
    block = dask.delayed(_deferred_block_computed, pure=False)(name)
    return da.from_delayed(block, shape=tuple(int(n) for n in shape), dtype=dtype)


@dataclass
class DeferredVariable:
    """
    A data variable of the main xds whose values are written after the MSv4
    metadata, by ``write_deferred_variables``.

    Attributes
    ----------
    name : str
        Data variable name.
    col : str | None
        MAIN column it is read from, None for the WEIGHT=1 fallback.
    grid_dtype : np.dtype
        dtype of the (time, baseline) grid the column is read into (the dtype
        of the partition's first cell, as in the row read path).
    cell_shape : tuple[int, ...]
        Cell shape of the column (numpy order).
    transform : Callable | None
        Conversion applied to every batch after reading it (the same as in the
        non-streamed path: TIME_CENTROID epoch, WEIGHT repeated along
        frequency), None for none.
    how : str
        How the readability of the column was decided (for logging).
    """

    name: str
    col: str | None
    grid_dtype: np.dtype
    cell_shape: tuple[int, ...] = ()
    transform: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False
    )
    how: str = ""


def deferred_main_column(
    main_rows: MainTableRows,
    col: str,
    datavar_name: str,
    time_baseline_shape: tuple[int, int],
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[da.Array, DeferredVariable]:
    """
    Placeholder and description of a data variable read from a MAIN column,
    after checking that the column can be read for the partition. Raises
    exactly where the row read path raises (first cell undefined, a value type
    that cannot be read in place) or would raise while reading (undefined
    cells, other cell shapes, see ``check_partition_cells``).

    Parameters
    ----------
    main_rows : MainTableRows
        MAIN table and the partition rows.
    col : str
        Column name.
    datavar_name : str
        Name of the data variable.
    time_baseline_shape : tuple[int, int]
        (n_times, n_baselines) of the partition.
    transform : Callable | None, optional
        Conversion applied after reading (see ``DeferredVariable``). Applied to
        one cell here to give the shape and dtype of the data variable.

    Returns
    -------
    tuple[da.Array, DeferredVariable]
        The placeholder (shape and dtype of the data variable) and the
        description used to write it.
    """
    table = main_rows.table
    cell_shape, grid_dtype = _partition_cell_shape_and_dtype(main_rows, col)
    column_dtype(table, col)  # value types the row reads cannot read raise there
    start = time.perf_counter()
    how = check_partition_cells(table, col, main_rows.rows)
    xradio_logger().debug(
        f"Column {col} readable for the partition ({how}, "
        f"{time.perf_counter() - start:.3f} s)"
    )
    probe = np.zeros((1, 1) + tuple(cell_shape), dtype=grid_dtype)
    if transform is not None:
        probe = transform(probe)
    shape = tuple(int(n) for n in time_baseline_shape) + probe.shape[2:]
    spec = DeferredVariable(
        name=datavar_name,
        col=col,
        grid_dtype=np.dtype(grid_dtype),
        cell_shape=tuple(cell_shape),
        transform=transform,
        how=how,
    )
    return deferred_placeholder(datavar_name, shape, probe.dtype), spec


def deferred_ones(
    name: str, shape: tuple[int, ...]
) -> tuple[da.Array, DeferredVariable]:
    """
    Placeholder and description of the WEIGHT=1 fallback (float64, the shape of
    VISIBILITY / SPECTRUM), written in batches like the column variables.

    Parameters
    ----------
    name : str
        Data variable name.
    shape : tuple[int, ...]
        Shape.

    Returns
    -------
    tuple[da.Array, DeferredVariable]
        Placeholder and description.
    """
    dtype = np.dtype(np.float64)
    spec = DeferredVariable(name=name, col=None, grid_dtype=dtype, how="ones")
    return deferred_placeholder(name, shape, dtype), spec


def time_batches(
    n_times: int, time_chunk: int, bytes_per_time: int, target_bytes: int
) -> tuple[int, ...]:
    """
    Split the time axis into batches of whole zarr chunks: as many chunks per
    batch as fit ``target_bytes`` (at least one), the last batch holding the
    remaining (possibly partial) chunk(s).

    Parameters
    ----------
    n_times : int
        Length of the time axis.
    time_chunk : int
        zarr chunk length along time.
    bytes_per_time : int
        Bytes of one time step of the variable (uncompressed).
    target_bytes : int
        Target batch size.

    Returns
    -------
    tuple[int, ...]
        Number of times of every batch (sum ``n_times``).
    """
    n_times = int(n_times)
    if n_times <= 0:
        return ()
    time_chunk = max(1, min(int(time_chunk), n_times))
    chunk_bytes = max(1, time_chunk * int(bytes_per_time))
    batch_times = max(1, int(target_bytes) // chunk_bytes) * time_chunk
    if batch_times >= n_times:
        return (n_times,)
    n_full, rest = divmod(n_times, batch_times)
    return (batch_times,) * n_full + ((rest,) if rest else ())


def count_row_runs(rows: np.ndarray) -> int:
    """Number of runs of consecutive rows of ascending ``rows``."""
    rows = np.asarray(rows)
    if rows.size == 0:
        return 0
    return int(np.count_nonzero(np.diff(rows) != 1)) + 1


@dataclass
class BatchChoice:
    """
    Time batches of one variable and the small-read guard figures.

    Attributes
    ----------
    batches : tuple[int, ...]
        Number of times of every batch.
    chunk_rows : TimeChunkRows | None
        Rows of every batch (None if nothing is read).
    runs_whole : int
        Row runs of a one-pass read of the partition.
    runs_batched : int
        Row runs of the batched read.
    guard : str
        The choice made ("one batch", "time", "fragmented: ...").
    """

    batches: tuple[int, ...]
    chunk_rows: TimeChunkRows | None
    runs_whole: int
    runs_batched: int
    guard: str


def choose_time_batches(
    main_rows: MainTableRows,
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    n_baselines: int,
    n_times: int,
    time_chunk: int,
    bytes_per_time: int,
    target_bytes: int,
    runs_whole: int,
    cache: dict | None = None,
) -> BatchChoice:
    """
    Time batches for reading one variable, with the small-read guard.

    Batches of whole zarr chunks of about ``target_bytes`` are used unless they
    fragment the reads: when the rows of the batches form more than
    ``FRAGMENTED_RUNS_RATIO`` times the row runs of a one-pass read (plus one
    per batch boundary). Every run is at least one read call and starts a new
    tile-cache window, so for baseline-major or interleaved row orders every
    batch would re-read the tiles of the whole partition. Then the variable is
    read in one pass if it fits ``FRAGMENTED_BATCH_FACTOR`` target batches,
    otherwise in batches of that size.

    Parameters
    ----------
    main_rows : MainTableRows
        MAIN table and the partition rows.
    tidxs, bidxs : np.ndarray
        Time and baseline index of every partition row.
    n_baselines : int
        Number of baselines of the grid.
    n_times : int
        Number of times.
    time_chunk : int
        zarr chunk length along time.
    bytes_per_time : int
        Bytes of one time step of the variable.
    target_bytes : int
        Target batch size.
    runs_whole : int
        Row runs of the partition (one-pass read).
    cache : dict | None, optional
        Holds the TimeChunkRows of the last batching (reused by the next
        variable with the same batches).

    Returns
    -------
    BatchChoice
        Batches, their rows and the guard figures.
    """

    def batch_rows(batches: tuple[int, ...]) -> TimeChunkRows:
        if cache is not None and cache.get("batches") == batches:
            return cache["chunk_rows"]
        if cache is not None:
            cache.clear()  # release the previous one first
        chunk_rows = TimeChunkRows(main_rows.rows, tidxs, bidxs, batches, n_baselines)
        if cache is not None:
            cache.update(batches=batches, chunk_rows=chunk_rows)
        return chunk_rows

    batches = time_batches(n_times, time_chunk, bytes_per_time, target_bytes)
    chunk_rows = batch_rows(batches)
    if len(batches) == 1:
        return BatchChoice(batches, chunk_rows, runs_whole, runs_whole, "one batch")
    runs_batched = chunk_rows.n_runs()
    if runs_batched <= FRAGMENTED_RUNS_RATIO * (runs_whole + len(batches) - 1):
        return BatchChoice(batches, chunk_rows, runs_whole, runs_batched, "time")
    large = time_batches(
        n_times, time_chunk, bytes_per_time, FRAGMENTED_BATCH_FACTOR * target_bytes
    )
    xradio_logger().debug(
        f"Time batches of {batches[0]} times would read {runs_batched} row runs "
        f"instead of {runs_whole}: reading in {len(large)} batch(es) of up to "
        f"{large[0]} times instead"
    )
    chunk_rows = batch_rows(large)
    runs_large = runs_whole if len(large) == 1 else chunk_rows.n_runs()
    guard = "fragmented: one pass" if len(large) == 1 else "fragmented: large batches"
    return BatchChoice(large, chunk_rows, runs_whole, runs_large, guard)


def _reverse_axis_in_place(
    values: np.ndarray, axis: int, max_tmp_bytes: int = DEFAULT_MAX_TMP_BYTES
) -> None:
    """Reverse ``values`` along ``axis`` (not 0) in place, slab by slab along
    axis 0, with a temporary of at most ``max_tmp_bytes`` (or one slab)."""
    if values.shape[0] == 0:
        return
    step = max(1, int(max_tmp_bytes) // max(1, values[0].nbytes))
    for t0 in range(0, values.shape[0], step):
        slab = values[t0 : t0 + step]
        slab[...] = np.flip(slab, axis=axis).copy()


def _read_batch(
    main_rows: MainTableRows,
    spec: DeferredVariable,
    chunk_rows: TimeChunkRows,
    k: int,
    batch_shape: tuple[int, ...],
    n_baselines: int,
    stats: dict,
) -> np.ndarray:
    """Values of batch ``k`` of a deferred variable (before any reversal)."""
    n_times = batch_shape[0]
    if spec.col is None:
        return np.ones(batch_shape, dtype=np.float64)
    rows, gidx = chunk_rows.chunk(k)
    plan = make_row_grid_plan(rows, gidx, n_times * n_baselines)
    del rows, gidx
    grid_shape = (n_times, n_baselines) + spec.cell_shape
    if plan.grid_is_full:
        grid = np.empty(grid_shape, dtype=spec.grid_dtype)
    else:
        grid = np.full(
            grid_shape, get_pad_value(spec.grid_dtype), dtype=spec.grid_dtype
        )
    read_rows_to_grid(
        main_rows.table,
        spec.col,
        plan,
        grid,
        max_elems=main_rows.max_elems,
        stats=stats,
    )
    del plan
    if spec.transform is not None:
        grid = spec.transform(grid)
    return grid


def _discard_variable(group, store_path: str, name: str) -> None:
    """Remove a partly written array and rewrite the consolidated metadata."""
    import zarr

    try:
        del group[name]
    except Exception as exc:  # best effort, the original error is raised
        xradio_logger().error(f"Could not remove {name} from {store_path}: {exc}")
    try:
        zarr.consolidate_metadata(store_path, zarr_format=ZARR_FORMAT)
    except Exception as exc:
        xradio_logger().error(
            f"Could not rewrite the consolidated metadata of {store_path}: {exc}"
        )


def write_deferred_variables(
    store_path: str,
    xds: xr.Dataset,
    deferred: dict[str, DeferredVariable],
    main_rows: MainTableRows,
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    n_baselines: int,
    reverse_frequency: bool,
    target_bytes: int | None = None,
) -> dict:
    """
    Write the values of the deferred data variables of a main xds whose MSv4
    (metadata and all other variables) is already in ``store_path``: one
    variable at a time, in batches of whole zarr chunks along time.

    Parameters
    ----------
    store_path : str
        The MSv4 zarr group (as written by ``DataTree.to_zarr(compute=False)``).
    xds : xr.Dataset
        The main xds that was written (dims of the variables; only its data
        variables that are in ``deferred`` are written).
    deferred : dict[str, DeferredVariable]
        The deferred data variables, by name.
    main_rows : MainTableRows
        MAIN table and the partition rows.
    tidxs, bidxs : np.ndarray
        Time and baseline index of every partition row.
    n_baselines : int
        Number of baselines (antennas for single dish) of the grid.
    reverse_frequency : bool
        Whether the frequency axis of the xds was reversed (decreasing channel
        frequencies), so every batch is reversed along frequency.
    target_bytes : int | None, optional
        Target batch size (uncompressed), by default get_stream_batch_bytes().

    Returns
    -------
    dict
        Statistics, per variable and in total (also logged at debug level).
    """
    import zarr

    if target_bytes is None:
        target_bytes = get_stream_batch_bytes()
    start_all = time.perf_counter()
    group = zarr.open_group(
        store_path, mode="r+", zarr_format=ZARR_FORMAT, use_consolidated=False
    )
    runs_whole = count_row_runs(main_rows.rows)
    cache: dict = {}
    var_stats = {}
    for name in xds.data_vars:
        spec = deferred.get(name)
        if spec is None:
            continue
        stats: dict = {}
        start = time.perf_counter()
        try:
            arr = group[name]
            n_times = int(arr.shape[0])
            bytes_per_time = int(np.prod(arr.shape[1:], dtype=np.int64)) * int(
                arr.dtype.itemsize
            )
            if spec.col is None:  # nothing to read
                choice = BatchChoice(
                    time_batches(
                        n_times, int(arr.chunks[0]), bytes_per_time, target_bytes
                    ),
                    None,
                    0,
                    0,
                    "no reads",
                )
            else:
                choice = choose_time_batches(
                    main_rows,
                    tidxs,
                    bidxs,
                    n_baselines,
                    n_times,
                    int(arr.chunks[0]),
                    bytes_per_time,
                    target_bytes,
                    runs_whole,
                    cache,
                )
            dims = xds[name].dims
            if dims[0] != "time":
                raise RuntimeError(f"{name} has dims {dims}, time is not the first")
            freq_axis = (
                dims.index("frequency")
                if reverse_frequency and spec.col is not None and "frequency" in dims
                else None
            )
            t0 = 0
            read_seconds = 0.0
            for k, batch_times in enumerate(choice.batches):
                batch_shape = (batch_times,) + tuple(arr.shape[1:])
                start_read = time.perf_counter()
                values = _read_batch(
                    main_rows,
                    spec,
                    choice.chunk_rows,
                    k,
                    batch_shape,
                    n_baselines,
                    stats,
                )
                if freq_axis is not None:
                    _reverse_axis_in_place(values, freq_axis)
                read_seconds += time.perf_counter() - start_read
                if values.shape != batch_shape or values.dtype != arr.dtype:
                    raise RuntimeError(
                        f"Batch of {name} has shape {values.shape} and dtype "
                        f"{values.dtype}, the zarr array needs {batch_shape} "
                        f"{arr.dtype}"
                    )
                arr[t0 : t0 + batch_times] = values
                del values
                t0 += batch_times
        except Exception:
            xradio_logger().error(
                f"Writing the data variable {name} of {store_path} failed: the "
                "variable is removed, the MSv4 is incomplete"
            )
            _discard_variable(group, store_path, name)
            raise
        rows_read = stats.get("direct_rows", 0) + stats.get("scatter_rows", 0)
        calls = stats.get("calls", 0)
        var_stats[name] = {
            "col": spec.col,
            "bytes": n_times * bytes_per_time,
            "time_chunk": int(arr.chunks[0]),
            "batches": len(choice.batches),
            "batch_times": choice.batches[0] if choice.batches else 0,
            "guard": choice.guard,
            "runs_whole": choice.runs_whole,
            "runs_batched": choice.runs_batched,
            "calls": calls,
            "selectrows_calls": stats.get("selectrows_calls", 0),
            "direct_rows": stats.get("direct_rows", 0),
            "scatter_rows": stats.get("scatter_rows", 0),
            "rows_per_call": rows_read / calls if calls else 0.0,
            "read_s": round(read_seconds, 4),
            "total_s": round(time.perf_counter() - start, 4),
            "readable": spec.how,
        }
        xradio_logger().debug(
            f"Streamed write of {name}: {json.dumps(var_stats[name], default=str)}"
        )
    cache.clear()
    summary = {
        "store": store_path,
        "target_bytes": int(target_bytes),
        "variables": len(var_stats),
        "batches": sum(v["batches"] for v in var_stats.values()),
        "calls": sum(v["calls"] for v in var_stats.values()),
        "rows_read": sum(
            v["direct_rows"] + v["scatter_rows"] for v in var_stats.values()
        ),
        "seconds": round(time.perf_counter() - start_all, 4),
    }
    summary["rows_per_call"] = (
        summary["rows_read"] / summary["calls"] if summary["calls"] else 0.0
    )
    xradio_logger().debug(f"Streamed write of the partition: {json.dumps(summary)}")
    return {"summary": summary, "variables": var_stats}
