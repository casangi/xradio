"""
Streamed write of the large MAIN-derived data variables of an MSv4.

Without streaming, ``convert_and_write_partition`` reads every MAIN column of a
partition into a dense (time, baseline, ...) grid and keeps all of them in memory
until one ``DataTree.to_zarr`` call writes the MSv4. With streaming:

1. The MSv4 is built as before (coordinates, attributes, small variables and
   sub-datasets), but the data variables that come from MAIN columns
   (VISIBILITY*, SPECTRUM, WEIGHT, FLAG, UVW, TIME_CENTROID,
   EFFECTIVE_INTEGRATION_TIME and the WEIGHT=1 fallback) are lazy placeholders
   (``deferred_main_column``, ``deferred_ones``). A column is skipped
   (WEIGHT_SPECTRUM falls back to WEIGHT) when the storage manager tells,
   without reading data, that a cell of the partition is undefined or of a
   shape other than the first cell's (``check_partition_cells``).
2. ``DataTree.to_zarr(compute=False, consolidated=False)`` writes all metadata
   (with the encoding, chunks and compressor of the non-streamed path) and the
   numpy variables, but not the consolidated metadata of the MSv4
   (``drop_consolidated_metadata`` also removes one left by an earlier write
   in mode "a"): ``consolidate_msv4`` writes it after the last value of the
   deferred variables. An MSv4 whose streamed write was interrupted (hard
   kill) is so marked as incomplete, as one whose to_zarr was interrupted:
   opening it warns (xarray falls back to the non-consolidated metadata,
   whose unwritten chunks read as fill values) or fails (``consolidated=True``),
   and a processing set whose conversion was interrupted does not list it (its
   root is consolidated after the last partition).
   The placeholders are never computed. The streamed
   write writes the values to the zarr arrays directly, bypassing xarray's
   encoding, so it is used only where that encoding leaves the values
   unchanged: the encoding xradio sets (chunks, compressors) and no CF coding
   (``deferred_encoding_problems``, before writing; otherwise the partition is
   read whole and written by to_zarr), and the zarr metadata declared for the
   deferred variables must match the values (dtype, shape, chunks, no filters,
   no CF coding attributes but a NaN _FillValue: ``deferred_array_problems``,
   before writing any value; otherwise the partition is converted again
   without the streamed write).
3. ``write_deferred_variables`` fills the placeholders one variable at a time,
   in batches of whole zarr chunks along time: each batch is read with one
   ascending pass over its rows (``read_time_chunk``), converted as in the
   non-streamed path (TIME_CENTROID epoch, WEIGHT repeated along frequency,
   reversed frequency axis) and written as chunk-aligned regions, so every
   zarr chunk is encoded once, from the same values: the chunk files are
   byte-identical to the non-streamed path.

Batch size: as many whole time chunks as fit ``STREAM_BATCH_BYTES``
(uncompressed, at least one chunk); a variable smaller than that is one batch.
A partition whose data variables together are at most
``IN_MEMORY_BATCH_FRACTION`` of a batch is not streamed: they are read whole
(``read_deferred_variables``) and the MSv4 is written by one to_zarr, as in the
non-streamed path (its peak, up to about twice the data, stays below that of one
streamed batch; the streamed write costs a few ms per variable).
On time-ordered partitions a time batch costs about one extra read call and one
re-read tile per batch boundary. When the rows of the time batches are spread
over the partition (baseline-major or interleaved row orders), batching would
multiply the read calls or the tiles read: the small-read guard
(``choose_time_batches``) then reads the variable in one pass if it fits
``FRAGMENTED_BATCH_FACTOR`` batches, otherwise in batches of that size.

Columns whose cells the storage manager cannot vouch for (StandardStMan /
IncrementalStMan indirect arrays, reference or concatenated tables) are checked
by the read itself, as in the non-streamed path. If the read of any column
fails after the metadata was written, ``write_deferred_variables`` raises
``DeferredReadError``; the caller removes the MSv4 and converts the partition
again without that column, which gives the result of the non-streamed path
(that skips a column whose read fails); the retry overwrites the MSv4 that the
failed attempt created (persistence mode "w-" becomes "w"). Any other failure
of the fill removes the MSv4 (``discard_msv4``, local paths and URLs) and is
raised: no MSv4 with missing or partly written data variables is left behind
(after a hard kill, one without consolidated metadata, see above).
"""

import base64
import json
import os
import posixpath
import shutil
import struct
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import dask.array as da
import numpy as np
import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio._utils.zarr.config import ZARR_FORMAT
from xradio.measurement_set._utils._msv2._tables.read import (
    _partition_cell_shape_and_dtype,
)
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    MainTableRows,
    TimeChunkRows,
    check_partition_cells,
    column_dtype,
    column_row_window,
    count_row_runs,
    count_row_windows,
    read_grid,
    read_time_chunk,
)

# Target size of a batch (bytes, uncompressed): the memory one variable takes
# while it is written (plus the zarr chunk encoding).
STREAM_BATCH_BYTES = 128 * 2**20
# Small-read guard. Time batches are kept only if
# - their row runs (read calls) are at most FRAGMENTED_RUNS_RATIO times those of
#   a one-pass read (plus one per batch boundary), and
# - the row windows (tiles) they load exceed those of a one-pass read by at most
#   FRAGMENTED_EXTRA_READS of them, or FRAGMENTED_BOUNDARY_WINDOWS per batch
#   boundary, whichever is more.
FRAGMENTED_RUNS_RATIO = 2.0
FRAGMENTED_EXTRA_READS = 0.25
FRAGMENTED_BOUNDARY_WINDOWS = 2
# A fragmenting variable is read in one pass if it is at most this many target
# batches, otherwise in batches of that size.
FRAGMENTED_BATCH_FACTOR = 8
# A partition whose data variables are at most this fraction of the target batch
# together is read whole and written by to_zarr (not streamed).
IN_MEMORY_BATCH_FRACTION = 0.25


class DeferredReadError(RuntimeError):
    """
    Reading the column of a deferred data variable failed after the MSv4
    metadata was written. The non-streamed path skips a column whose read
    fails; the caller reproduces that by converting the partition again
    without the column.

    Attributes
    ----------
    name : str
        Data variable name.
    col : str
        MAIN column.
    """

    def __init__(self, name: str, col: str, store_path: str):
        super().__init__(
            f"Reading column {col} for the data variable {name} of {store_path} failed"
        )
        self.name = name
        self.col = col


def _deferred_block_computed(name: str) -> None:
    raise RuntimeError(
        f"The deferred data variable {name} was computed: it is written by "
        "write_deferred_variables, after the MSv4 metadata"
    )


def deferred_placeholder(name: str, shape: tuple[int, ...], dtype) -> da.Array:
    """
    A lazy (dask) array of a given shape and dtype that stands for a data
    variable written later (one chunk; it raises if it is ever computed, so
    that nothing reads a whole variable into memory by accident).

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
    shape = tuple(int(n) for n in shape)
    # a one-task graph, built directly (da.from_delayed costs ~10x more)
    graph_name = f"deferred-{name}-{uuid.uuid4().hex}"
    key = (graph_name,) + (0,) * len(shape)
    return da.Array(
        {key: (_deferred_block_computed, name)},
        graph_name,
        chunks=tuple((n,) for n in shape),
        dtype=dtype,
    )


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
        of the partition's first cell, as in read_col_conversion_numpy).
    cell_shape : tuple[int, ...]
        Cell shape of the column (numpy order).
    transform : Callable | None
        Conversion applied to every batch after reading it (the same as in the
        non-streamed path: TIME_CENTROID epoch, WEIGHT repeated along
        frequency), None for none.
    how : str
        How the readability of the column was decided (for logging).
    verified : bool
        Whether the storage manager vouched for every cell of the partition
        (False: only the read can tell, see ``check_partition_cells``).
    window_rows : int
        Table rows per tile (or nominal window) of the column, for the
        small-read guard (``column_row_window``).
    window : str
        "tile" or "nominal" (how ``window_rows`` was found).
    frequency_constant : bool
        The values do not change along frequency (WEIGHT repeated from the
        WEIGHT column, the WEIGHT=1 fallback): no reversal needed.
    """

    name: str
    col: str | None
    grid_dtype: np.dtype
    cell_shape: tuple[int, ...] = ()
    transform: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False
    )
    how: str = ""
    verified: bool = True
    window_rows: int = 1
    window: str = ""
    frequency_constant: bool = False


def deferred_main_column(
    main_rows: MainTableRows,
    col: str,
    datavar_name: str,
    time_baseline_shape: tuple[int, int],
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    frequency_constant: bool = False,
) -> tuple[da.Array, DeferredVariable]:
    """
    Placeholder and description of a data variable read from a MAIN column.
    Raises where read_col_conversion_numpy raises before reading (first cell
    undefined, a value type that cannot be read in place) and where the
    storage manager tells that the read would raise (undefined cells, other
    cell shapes, see ``check_partition_cells``).

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
    frequency_constant : bool, optional
        The converted values do not change along frequency.

    Returns
    -------
    tuple[da.Array, DeferredVariable]
        The placeholder (shape and dtype of the data variable) and the
        description used to write it.
    """
    table = main_rows.table
    cell_shape, grid_dtype = _partition_cell_shape_and_dtype(main_rows, col)
    stored_dtype = column_dtype(table, col)  # types the row reads cannot read raise
    start = time.perf_counter()
    storage = main_rows.column_storage(col)
    check = check_partition_cells(table, col, main_rows.rows, storage)
    cell_bytes = int(np.prod(cell_shape, dtype=np.int64)) * stored_dtype.itemsize
    window_rows, window = column_row_window(
        storage, table.nrows(), cell_shape, cell_bytes
    )
    xradio_logger().debug(
        f"Column {col} for the partition: {check.how}; {window} window of "
        f"{window_rows} rows ({time.perf_counter() - start:.3f} s)"
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
        how=check.how,
        verified=check.verified,
        window_rows=window_rows,
        window=window,
        frequency_constant=frequency_constant,
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
    spec = DeferredVariable(
        name=name, col=None, grid_dtype=dtype, how="ones", frequency_constant=True
    )
    return deferred_placeholder(name, shape, dtype), spec


# Encoding keys of the deferred data variables (set by add_encoding) that act
# through the zarr array itself (chunk grid, compressors), so the batch writer,
# which writes through the same zarr arrays, applies them as to_zarr does.
STREAMED_ENCODING_KEYS = frozenset({"chunks", "compressors", "preferred_chunks"})
# CF encoding keys / attributes with which xarray transforms the values of a
# variable before handing them to zarr (masking, packing, dtype, time units).
# The streamed write writes the values to the zarr arrays directly, so a
# deferred variable with any of them is not streamed. Every such
# transformation is recorded in the stored metadata (the reader needs it to
# decode the values): the dtype or one of these attributes.
CF_VALUE_CODING_KEYS = frozenset(
    {
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "dtype",
        "calendar",
        "_Unsigned",
    }
)
# dtype kinds the streamed write writes (bool, integers, floats, complex)
STREAMED_DTYPE_KINDS = "biufc"


class DeferredEncodingError(RuntimeError):
    """
    The zarr metadata written for the deferred data variables declares an
    encoding that the streamed write does not apply (see
    ``deferred_array_problems``). Raised before any value is written; the
    caller converts the partition again without the streamed write.
    """


def check_deferred_variables(
    xds: xr.Dataset, deferred: dict[str, DeferredVariable]
) -> None:
    """
    Check, before the MSv4 is written, that the streamed write writes every
    variable exactly once: every lazy (placeholder) data variable of the main
    xds has a deferred description (otherwise its zarr array would get metadata
    but no chunks and read back as fill values), and every deferred variable
    of the xds is still a placeholder. The encodings are checked by
    ``deferred_encoding_problems`` and ``deferred_array_problems``.

    Parameters
    ----------
    xds : xr.Dataset
        The main xds about to be written.
    deferred : dict[str, DeferredVariable]
        The deferred data variables, by name (some may have been dropped from
        the xds, e.g. UVW of single dish).

    Raises
    ------
    RuntimeError
        If any of these does not hold.
    """
    for name, var in xds.data_vars.items():
        if isinstance(var.data, da.Array) and name not in deferred:
            raise RuntimeError(
                f"The data variable {name} is lazy but has no deferred description: "
                "the streamed write would leave it unwritten"
            )
    for name in deferred:
        if name in xds.data_vars and not isinstance(xds[name].data, da.Array):
            raise RuntimeError(
                f"The deferred data variable {name} holds values: they would be "
                "written twice"
            )


def deferred_encoding_problems(
    xds: xr.Dataset, deferred: dict[str, DeferredVariable]
) -> list[str]:
    """
    Why the deferred data variables of the xds cannot be streamed, from their
    dtype, attributes and encoding (checked before anything is written): the
    streamed write writes the values straight to the zarr arrays, so it gives
    what to_zarr writes only if xarray's encoding leaves the values unchanged.
    That holds for the encoding xradio sets (``add_encoding``: chunks and
    compressors, applied by the zarr arrays) on bool, integer, float and
    complex values; any other encoding key, or a CF coding key
    (``CF_VALUE_CODING_KEYS``: _FillValue, scale_factor, dtype, ...) in the
    attributes or the encoding, is reported. ``deferred_array_problems``
    checks the zarr metadata that to_zarr then declares.

    Parameters
    ----------
    xds : xr.Dataset
        The main xds about to be written.
    deferred : dict[str, DeferredVariable]
        The deferred data variables, by name.

    Returns
    -------
    list[str]
        One message per problem (empty: the variables can be streamed).
    """
    problems = []
    for name in deferred:
        if name not in xds.data_vars:
            continue
        var = xds[name].variable
        if var.dtype.kind not in STREAMED_DTYPE_KINDS or not var.dtype.isnative:
            problems.append(f"{name}: dtype {var.dtype}")
        other = sorted(set(var.encoding) - STREAMED_ENCODING_KEYS)
        if other:
            problems.append(f"{name}: encoding {other}")
        coded = sorted(CF_VALUE_CODING_KEYS & set(var.attrs))
        if coded:
            problems.append(f"{name}: attributes {coded}")
    return problems


def _stored_fill_value_is_nan(value) -> bool:
    """
    Whether a stored ``_FillValue`` attribute is NaN (masking NaN with NaN
    leaves the values unchanged): a number, or a string as zarr format 3
    stores xarray's floating fill values (base64 of a little-endian float64,
    ``"AAAAAAAA+H8="`` for NaN), or "NaN".
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float | np.integer | np.floating):
        return bool(np.isnan(value))
    if isinstance(value, str):
        if value == "NaN":
            return True
        try:
            raw = base64.b64decode(value, validate=True)
        except ValueError:
            return False
        if len(raw) in (4, 8):
            fmt = "<f" if len(raw) == 4 else "<d"
            return bool(np.isnan(struct.unpack(fmt, raw)[0]))
    return False


def deferred_array_problems(
    group, xds: xr.Dataset, deferred: dict[str, DeferredVariable]
) -> list[str]:
    """
    Why the zarr arrays that ``DataTree.to_zarr(compute=False)`` declared for
    the deferred data variables do not hold what the batch writer writes (it
    writes the values, of the placeholder's dtype, with the arrays' own codecs
    in chunk-aligned regions): another dtype, shape or chunk shape than the
    xds declares, array-to-array codecs (filters) or a serializer other than
    the plain bytes codec, or CF coding attributes that record a transformation
    of the values (scale_factor, add_offset, a non-NaN _FillValue, ...). Only
    public zarr metadata is used.

    Parameters
    ----------
    group : zarr.Group
        The MSv4 zarr group (as written by ``DataTree.to_zarr(compute=False)``).
    xds : xr.Dataset
        The main xds that was written.
    deferred : dict[str, DeferredVariable]
        The deferred data variables, by name.

    Returns
    -------
    list[str]
        One message per problem (empty: the batch writer writes exactly what
        to_zarr would).
    """
    problems = []
    for name in deferred:
        if name not in xds.data_vars:
            continue
        var = xds[name].variable
        try:
            arr = group[name]
            dtype = np.dtype(arr.dtype)
            shape, chunks = tuple(arr.shape), tuple(arr.chunks)
            filters, serializer = tuple(arr.filters), arr.serializer
            attrs = dict(arr.attrs)
        except Exception as exc:  # anything unexpected: do not stream
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if dtype != var.dtype:
            problems.append(f"{name}: zarr dtype {dtype}, values {var.dtype}")
        if shape != var.shape:
            problems.append(f"{name}: zarr shape {shape}, values {var.shape}")
        expected_chunks = tuple(int(n) for n in var.encoding.get("chunks", shape))
        if chunks != expected_chunks:
            problems.append(f"{name}: zarr chunks {chunks}, encoding {expected_chunks}")
        if filters:
            problems.append(f"{name}: filters {filters}")
        if type(serializer).__name__ != "BytesCodec":
            problems.append(f"{name}: serializer {serializer}")
        coded = sorted(CF_VALUE_CODING_KEYS & set(attrs))
        if "_FillValue" in coded and dtype.kind in "fc":
            # xarray's default for floats: masking NaN with NaN changes nothing
            fill = attrs["_FillValue"]
            parts = fill if isinstance(fill, list | tuple) else [fill]
            if parts and all(_stored_fill_value_is_nan(part) for part in parts):
                coded.remove("_FillValue")
        if coded:
            problems.append(
                f"{name}: attributes {{{', '.join(f'{k}: {attrs[k]!r}' for k in coded)}}}"
            )
    return problems


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


@dataclass
class BatchChoice:
    """
    Time batches of one variable and the small-read guard figures.

    Attributes
    ----------
    batches : tuple[int, ...]
        Number of times of every batch.
    chunk_rows : TimeChunkRows | None
        Rows of every batch; None for one batch (the partition's grid plan is
        used) or if nothing is read.
    runs_whole : int
        Row runs of a one-pass read of the partition.
    runs_batched : int
        Row runs of the batched read.
    guard : str
        The choice made ("one batch", "time", "fragmented: ...", "no reads").
    windows_whole : int | None
        Row windows (tiles) of a one-pass read (None if not computed).
    windows_batched : int | None
        Row windows (tiles) of the batched read.
    reason : str
        Why the time batches were not kept ("" if they were).
    """

    batches: tuple[int, ...]
    chunk_rows: TimeChunkRows | None
    runs_whole: int
    runs_batched: int
    guard: str
    windows_whole: int | None = None
    windows_batched: int | None = None
    reason: str = ""


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
    *,
    window_rows: int = 1,
    cache: dict | None = None,
) -> BatchChoice:
    """
    Time batches for reading one variable, with the small-read guard.

    Batches of whole zarr chunks of about ``target_bytes`` are used unless they
    fragment the reads, compared with one ascending pass over the partition:

    - read calls: the row runs of the batches exceed ``FRAGMENTED_RUNS_RATIO``
      times those of one pass (plus one per batch boundary), as for
      baseline-major rows: every run is read by a call of its own or as a
      piece of a selectrows call (see ``read_rows_to_grid``);
    - tiles: the row windows of ``window_rows`` rows (the column's tiles) that
      the batches load exceed those of one pass by more than
      ``FRAGMENTED_EXTRA_READS`` of them and ``FRAGMENTED_BOUNDARY_WINDOWS`` per
      batch boundary. Each batch loads the tiles its rows fall in (the tile
      cache keeps about one row-slab), so batches whose rows are spread over
      the same tiles re-read them, even when the calls stay few (rows of
      several partitions interleaved row by row, shuffled rows).

    A fragmenting variable is read in one pass if it fits
    ``FRAGMENTED_BATCH_FACTOR`` target batches, otherwise in batches of that
    size.

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
    window_rows : int, optional
        Table rows per tile of the column (``column_row_window``).
    cache : dict | None, optional
        Holds the TimeChunkRows of the last batching (reused by the next
        variable with the same batches) and the one-pass window counts.

    Returns
    -------
    BatchChoice
        Batches, their rows and the guard figures.
    """
    cache = {} if cache is None else cache

    def batch_rows(batches: tuple[int, ...]) -> TimeChunkRows:
        last = cache.get("batching")
        if last is not None and last[0] == batches:
            return last[1]
        cache.pop("batching", None)  # release the previous one first
        chunk_rows = TimeChunkRows(main_rows.rows, tidxs, bidxs, batches, n_baselines)
        cache["batching"] = (batches, chunk_rows)
        return chunk_rows

    def whole_windows() -> int:
        counts = cache.setdefault("windows_whole", {})
        if window_rows not in counts:
            counts[window_rows] = count_row_windows(main_rows.rows, window_rows)
        return counts[window_rows]

    batches = time_batches(n_times, time_chunk, bytes_per_time, target_bytes)
    if len(batches) <= 1:
        return BatchChoice(batches, None, runs_whole, runs_whole, "one batch")
    n_bounds = len(batches) - 1
    chunk_rows = batch_rows(batches)
    runs_batched = chunk_rows.n_runs()
    windows_whole = whole_windows()
    windows_batched = chunk_rows.n_windows(window_rows)
    reasons = []
    if runs_batched > FRAGMENTED_RUNS_RATIO * (runs_whole + n_bounds):
        reasons.append(f"{runs_batched} row runs instead of {runs_whole}")
    if windows_batched > windows_whole + max(
        FRAGMENTED_EXTRA_READS * windows_whole, FRAGMENTED_BOUNDARY_WINDOWS * n_bounds
    ):
        reasons.append(
            f"{windows_batched} tiles of {window_rows} rows read instead of "
            f"{windows_whole}"
        )
    if not reasons:
        return BatchChoice(
            batches,
            chunk_rows,
            runs_whole,
            runs_batched,
            "time",
            windows_whole,
            windows_batched,
        )
    reason = ", ".join(reasons)
    large = time_batches(
        n_times, time_chunk, bytes_per_time, FRAGMENTED_BATCH_FACTOR * target_bytes
    )
    xradio_logger().debug(
        f"Time batches of {batches[0]} times would read {reason}: reading in "
        f"{len(large)} batch(es) of up to {large[0]} times instead"
    )
    if len(large) == 1:
        cache.pop("batching", None)
        return BatchChoice(
            large,
            None,
            runs_whole,
            runs_whole,
            "fragmented: one pass",
            windows_whole,
            windows_whole,
            reason,
        )
    chunk_rows = batch_rows(large)
    return BatchChoice(
        large,
        chunk_rows,
        runs_whole,
        chunk_rows.n_runs(),
        "fragmented: large batches",
        windows_whole,
        chunk_rows.n_windows(window_rows),
        reason,
    )


def _read_batch(
    main_rows: MainTableRows,
    spec: DeferredVariable,
    choice: BatchChoice,
    k: int,
    batch_shape: tuple[int, ...],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    freq_axis: int | None,
    stats: dict,
) -> np.ndarray:
    """Values of batch ``k`` of a deferred variable, converted and reversed
    along frequency as the xds."""
    if spec.col is None:
        return np.ones(batch_shape, dtype=np.float64)
    reverse_axis = None if spec.frequency_constant else freq_axis
    if choice.chunk_rows is None:
        # The whole partition: the partition's grid plan, shared by every
        # variable read in one batch
        grid_shape = tuple(batch_shape[:2])
        plan = main_rows.grid_plan(tidxs, bidxs, grid_shape)
        return read_grid(
            main_rows.table,
            spec.col,
            plan,
            grid_shape + spec.cell_shape,
            spec.grid_dtype,
            main_rows.max_elems,
            spec.transform,
            reverse_axis,
            stats,
        )
    return read_time_chunk(
        main_rows.table,
        spec.col,
        choice.chunk_rows,
        k,
        spec.cell_shape,
        spec.grid_dtype,
        main_rows.max_elems,
        spec.transform,
        reverse_axis,
        stats,
    )


def deferred_bytes(xds: xr.Dataset, deferred: dict[str, DeferredVariable]) -> int:
    """Bytes (uncompressed) of the deferred data variables of the xds."""
    return sum(int(xds[name].nbytes) for name in deferred if name in xds.data_vars)


def fits_in_memory(
    xds: xr.Dataset, deferred: dict[str, DeferredVariable], target_bytes: int
) -> bool:
    """
    Whether the deferred data variables of a partition are small enough to be
    read whole (``read_deferred_variables``) instead of streamed: at most
    ``IN_MEMORY_BATCH_FRACTION`` of the target batch together.

    Parameters
    ----------
    xds : xr.Dataset
        The main xds.
    deferred : dict[str, DeferredVariable]
        The deferred data variables, by name.
    target_bytes : int
        Target batch size of the streamed write.

    Returns
    -------
    bool
        True to read the partition whole.
    """
    return deferred_bytes(xds, deferred) <= IN_MEMORY_BATCH_FRACTION * target_bytes


def _ordered_names(xds: xr.Dataset, deferred: dict[str, DeferredVariable]) -> list:
    """The deferred data variables of the xds, those whose cells only the read
    can verify first (if the read decides against one, less is lost)."""
    return sorted(
        (name for name in xds.data_vars if name in deferred),
        key=lambda name: deferred[name].verified,
    )


def _frequency_axis(xds: xr.Dataset, name: str, reverse_frequency: bool) -> int | None:
    dims = xds[name].dims
    if dims[0] != "time":
        raise RuntimeError(f"{name} has dims {dims}: time must be the first dimension")
    return (
        dims.index("frequency") if reverse_frequency and "frequency" in dims else None
    )


def read_deferred_variables(
    xds: xr.Dataset,
    deferred: dict[str, DeferredVariable],
    main_rows: MainTableRows,
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    reverse_frequency: bool,
) -> dict:
    """
    Read every deferred data variable of the xds whole and put its values in
    place of its placeholder (attributes and encoding are kept), for a small
    partition (``fits_in_memory``): the MSv4 is then written by to_zarr as in
    the non-streamed path (the same values), without the per-variable cost of
    the streamed write.

    Parameters
    ----------
    xds : xr.Dataset
        The main xds (its deferred variables are replaced in place).
    deferred : dict[str, DeferredVariable]
        The deferred data variables, by name.
    main_rows : MainTableRows
        MAIN table and the partition rows.
    tidxs, bidxs : np.ndarray
        Time and baseline index of every partition row.
    reverse_frequency : bool
        Whether the frequency axis of the xds was reversed.

    Returns
    -------
    dict
        Statistics, per variable and in total (also logged at debug level).

    Raises
    ------
    DeferredReadError
        If reading (or converting) a column fails: the caller converts the
        partition again without it.
    """
    start_all = time.perf_counter()
    var_stats = {}
    try:
        for name in _ordered_names(xds, deferred):
            spec, var = deferred[name], xds.variables[name]
            start, stats = time.perf_counter(), {}
            choice = BatchChoice((var.shape[0],), None, 0, 0, "one batch")
            freq_axis = _frequency_axis(xds, name, reverse_frequency)
            try:
                values = _read_batch(
                    main_rows,
                    spec,
                    choice,
                    0,
                    var.shape,
                    tidxs,
                    bidxs,
                    freq_axis,
                    stats,
                )
            except Exception as exc:
                if spec.col is None:
                    raise
                raise DeferredReadError(name, spec.col, "the main xds") from exc
            if values.shape != var.shape or values.dtype != var.dtype:
                raise RuntimeError(
                    f"{name} read with shape {values.shape} and dtype {values.dtype}, "
                    f"the placeholder has {var.shape} {var.dtype}"
                )
            var.data = values  # in place: order, attrs and encoding kept
            time_chunk = int(var.encoding.get("chunks", var.shape)[0])
            var_stats[name] = _variable_stats(
                spec, choice, int(values.nbytes), time_chunk, stats, start
            )
            var_stats[name]["read_s"] = var_stats[name]["total_s"]
            del values
    finally:
        main_rows.release_plans()
    summary = _summary(var_stats, "the main xds", None, start_all, in_memory=True)
    xradio_logger().debug(f"Read the partition in memory: {json.dumps(summary)}")
    return {"summary": summary, "variables": var_stats}


def _variable_stats(
    spec: DeferredVariable,
    choice: BatchChoice,
    nbytes: int,
    time_chunk: int,
    stats: dict,
    start: float,
) -> dict:
    """Statistics of one variable (logged and returned)."""
    rows_read = stats.get("direct_rows", 0) + stats.get("scatter_rows", 0)
    calls = stats.get("calls", 0)
    return {
        "col": spec.col,
        "bytes": nbytes,
        "time_chunk": time_chunk,
        "batches": len(choice.batches),
        "batch_times": choice.batches[0] if choice.batches else 0,
        "guard": choice.guard,
        "guard_reason": choice.reason,
        "runs_whole": choice.runs_whole,
        "runs_batched": choice.runs_batched,
        "window_rows": spec.window_rows,
        "window": spec.window,
        "windows_whole": choice.windows_whole,
        "windows_batched": choice.windows_batched,
        "calls": calls,
        "selectrows_calls": stats.get("selectrows_calls", 0),
        "direct_rows": stats.get("direct_rows", 0),
        "scatter_rows": stats.get("scatter_rows", 0),
        "gap_rows": stats.get("gap_rows", 0),
        "rows_per_call": rows_read / calls if calls else 0.0,
        "read_s": 0.0,
        "total_s": round(time.perf_counter() - start, 4),
        "readable": spec.how,
        "verified": spec.verified,
    }


def _summary(
    var_stats: dict,
    store: str,
    target_bytes: int | None,
    start_all: float,
    in_memory: bool,
) -> dict:
    summary = {
        "store": store,
        "in_memory": in_memory,
        "target_bytes": target_bytes,
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
    return summary


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
    variable at a time, in batches of whole zarr chunks along time, each
    written in pieces of at most ``target_bytes`` (whole chunks). Variables
    whose cells only the read can verify are written first.

    The caller checks ``check_deferred_variables`` and
    ``deferred_encoding_problems`` before writing the metadata; the zarr
    metadata is checked here (``deferred_array_problems``) before any value is
    written. Nothing is cleaned up on failure (see ``discard_msv4``).

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
        Target batch size (uncompressed), by default STREAM_BATCH_BYTES.

    Returns
    -------
    dict
        Statistics, per variable and in total (also logged at debug level).

    Raises
    ------
    DeferredEncodingError
        If the zarr metadata declares an encoding that the streamed write does
        not apply (nothing written yet): the caller converts the partition
        again without the streamed write.
    DeferredReadError
        If reading (or converting) a column fails: the caller converts the
        partition again without it.
    """
    import zarr

    if target_bytes is None:
        target_bytes = STREAM_BATCH_BYTES
    start_all = time.perf_counter()
    # (opening only the arrays written is cheaper than parsing the consolidated
    # metadata of the whole MSv4)
    group = zarr.open_group(
        store_path, mode="r+", zarr_format=ZARR_FORMAT, use_consolidated=False
    )
    problems = deferred_array_problems(group, xds, deferred)
    if problems:
        raise DeferredEncodingError(
            f"The zarr metadata of {store_path} declares an encoding that the "
            f"streamed write does not apply: {'; '.join(problems)}"
        )
    runs_whole = count_row_runs(main_rows.rows)
    cache: dict = {}
    var_stats = {}
    try:
        for name in _ordered_names(xds, deferred):
            spec = deferred[name]
            stats: dict = {}
            start = time.perf_counter()
            arr = group[name]
            n_times = int(arr.shape[0])
            time_chunk = int(arr.chunks[0])
            bytes_per_time = int(np.prod(arr.shape[1:], dtype=np.int64)) * int(
                arr.dtype.itemsize
            )
            if spec.col is None:  # nothing to read
                choice = BatchChoice(
                    time_batches(n_times, time_chunk, bytes_per_time, target_bytes),
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
                    time_chunk,
                    bytes_per_time,
                    target_bytes,
                    runs_whole,
                    window_rows=spec.window_rows,
                    cache=cache,
                )
            if tuple(arr.shape) != xds[name].shape:
                raise RuntimeError(
                    f"{name} has shape {xds[name].shape}, its zarr array {arr.shape}"
                )
            freq_axis = _frequency_axis(xds, name, reverse_frequency)
            t0 = 0
            read_seconds = 0.0
            for k, batch_times in enumerate(choice.batches):
                batch_shape = (batch_times,) + tuple(arr.shape[1:])
                start_read = time.perf_counter()
                try:
                    values = _read_batch(
                        main_rows,
                        spec,
                        choice,
                        k,
                        batch_shape,
                        tidxs,
                        bidxs,
                        freq_axis,
                        stats,
                    )
                except Exception as exc:
                    if spec.col is None:
                        raise
                    raise DeferredReadError(name, spec.col, store_path) from exc
                read_seconds += time.perf_counter() - start_read
                if values.shape != batch_shape or values.dtype != arr.dtype:
                    raise RuntimeError(
                        f"Batch of {name} has shape {values.shape} and dtype "
                        f"{values.dtype}, the zarr array needs {batch_shape} "
                        f"{arr.dtype}"
                    )
                # pieces of whole chunks of at most the target: bounds the
                # encoding buffers of a large (fragmented) batch
                p0 = 0
                for n in time_batches(
                    batch_times, time_chunk, bytes_per_time, target_bytes
                ):
                    arr[t0 + p0 : t0 + p0 + n] = values[p0 : p0 + n]
                    p0 += n
                del values
                t0 += batch_times
            var_stats[name] = _variable_stats(
                spec, choice, n_times * bytes_per_time, time_chunk, stats, start
            )
            var_stats[name]["read_s"] = round(read_seconds, 4)
            xradio_logger().debug(
                f"Streamed write of {name}: {json.dumps(var_stats[name], default=str)}"
            )
    finally:
        cache.clear()
        main_rows.release_plans()
    summary = _summary(var_stats, store_path, int(target_bytes), start_all, False)
    xradio_logger().debug(f"Streamed write of the partition: {json.dumps(summary)}")
    return {"summary": summary, "variables": var_stats}


def _is_url(store_path: str) -> bool:
    """Whether a store path is a URL (zarr opens it through fsspec)."""
    return "://" in str(store_path)


def _url_fs(store_path: str):
    """The fsspec file system and path of a store URL, as zarr opens it."""
    import fsspec

    return fsspec.core.url_to_fs(str(store_path))


def msv4_members(store_path: str) -> set[str] | None:
    """
    Entries of an MSv4 store (its zarr.json and the directories of its
    members), local path or URL (fsspec).

    Parameters
    ----------
    store_path : str
        The MSv4 zarr group.

    Returns
    -------
    set[str] | None
        The entry names, None if there is no MSv4 there.
    """
    if not _is_url(store_path):
        return set(os.listdir(store_path)) if os.path.isdir(store_path) else None
    fs, path = _url_fs(store_path)
    if not fs.exists(path):
        return None
    return {
        posixpath.basename(entry.rstrip("/")) for entry in fs.ls(path, detail=False)
    }


def drop_consolidated_metadata(store_path: str) -> None:
    """
    Remove the consolidated metadata of an MSv4 (from the zarr.json of its
    root group) before its deferred data variables are written, so that the
    MSv4 opens as incomplete until ``consolidate_msv4`` (see the module
    docstring). ``to_zarr(consolidated=False)`` writes none, and xarray drops
    one left by an earlier write in mode "a"; this does not depend on it.

    Parameters
    ----------
    store_path : str
        The MSv4 zarr group.
    """
    import zarr

    # Opened without the consolidated metadata: writing the group metadata
    # (the same attributes) leaves it out
    group = zarr.open_group(
        store_path, mode="r+", zarr_format=ZARR_FORMAT, use_consolidated=False
    )
    group.update_attributes({})


def consolidate_msv4(store_path: str) -> None:
    """
    Write the consolidated metadata of an MSv4 (as to_zarr does) once all its
    deferred data variables are written: from then on it opens as complete.

    Parameters
    ----------
    store_path : str
        The MSv4 zarr group.
    """
    import zarr

    zarr.consolidate_metadata(store_path, zarr_format=ZARR_FORMAT)


def discard_msv4(
    store_path: str,
    deferred_names,
    remove_store: bool,
    members_before: set[str] | None = None,
) -> str:
    """
    Remove what a failed streamed write left in an MSv4 store, so that no MSv4
    whose data variables are missing chunks (and read back as fill values) is
    left behind. Never raises (errors are logged).

    Parameters
    ----------
    store_path : str
        The MSv4 zarr group (local path or URL).
    deferred_names : Iterable[str]
        Names of the deferred data variables.
    remove_store : bool
        Whether this conversion wrote the whole MSv4 (persistence mode "w" or
        "w-", or no MSv4 there before): then it is removed, a local path with
        shutil, a URL through its fsspec file system (the one zarr writes
        through). Otherwise (mode "a" on an existing MSv4), or if that fails,
        the deferred arrays and the members this conversion added are removed;
        the MSv4 is left without consolidated metadata (as incomplete, see
        ``drop_consolidated_metadata``).
    members_before : set[str] | None, optional
        Entries of the MSv4 directory before this conversion wrote it
        (``msv4_members``; None: not known).

    Returns
    -------
    str
        What was done (for logging).
    """
    import zarr

    if remove_store:
        try:
            if _is_url(store_path):
                fs, path = _url_fs(store_path)
                if not fs.exists(path):
                    return "MSv4 not written"
                fs.rm(path, recursive=True)
            else:
                shutil.rmtree(store_path, ignore_errors=False)
            return "MSv4 removed"
        except FileNotFoundError:
            return "MSv4 not written"
        except Exception as exc:
            xradio_logger().error(
                f"Could not remove {store_path}: {exc}; removing its deferred "
                "data variables"
            )
    removed = []
    try:
        group = zarr.open_group(
            store_path, mode="r+", zarr_format=ZARR_FORMAT, use_consolidated=False
        )
        for name in sorted(group.keys()):
            added = members_before is not None and name not in members_before
            if name in deferred_names or added:
                try:
                    del group[name]
                    removed.append(name)
                except Exception as exc:
                    xradio_logger().error(
                        f"Could not remove {name} from {store_path}: {exc}"
                    )
        # (a consolidated metadata would still list the removed arrays)
        drop_consolidated_metadata(store_path)
    except Exception as exc:
        xradio_logger().error(f"Could not clean up {store_path}: {exc}")
    return f"removed {removed} from the MSv4 (left without consolidated metadata)"
