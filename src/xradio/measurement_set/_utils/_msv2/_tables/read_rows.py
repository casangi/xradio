"""
TaQL-free reads of MSv2 MAIN-table rows (python-casacore).

The MAIN rows of a partition are selected once, in numpy (see
``partition_queries.create_partitions``). Every column of the partition is then
read from the *base* table with bounded ``getcolnp`` / ``getcolslicenp`` calls:

- straight into the dense (time, baseline[, chan, pol]) grid where a run of
  consecutive MAIN rows maps to consecutive grid cells (no temporary at all), or
- through a bounded temporary plus a numpy scatter for everything else.

Rules that every read here follows (each one avoids a measured python-casacore
trap):

- Rows are passed in ascending order. Unsorted rows defeat the tile cache of the
  tiled storage managers (64x-512x more bytes read).
- Every ``get*np`` call is capped at ``max_elems`` elements (default 2**26, at
  most 2**29). Larger reads can silently leave rows unfilled
  (python-casacore issue #130).
- Buffers are C-contiguous, aligned, writeable, native-endian and of the exact
  column dtype. A different dtype makes casacore allocate two hidden full-size
  temporaries, and an unaligned buffer is silently left unfilled.
- A call never covers the whole column (start row 0, all rows): that call takes
  casacore's whole-column path, which crashes (SIGSEGV) on a TiledShapeStMan
  column with undefined cells, where partial reads raise an exception instead.
- Tables are opened in the process that reads them (no table opened before a
  ``fork()`` is used in the child).
"""

import dataclasses
from typing import Any

import numpy as np

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

# Default and maximum number of elements read by one getcolnp/getcolslicenp call
# (python-casacore #130: keep every call at 2**29 elements or fewer).
DEFAULT_MAX_ELEMS = 2**26
MAX_ELEMS_LIMIT = 2**29
# Default bound of the temporary buffer used for rows that cannot be read
# straight into the grid. It is the only memory the reads add on top of the
# grids (plus index arrays); a larger buffer means fewer (selectrows) calls.
DEFAULT_MAX_TMP_BYTES = 16 * 1024 * 1024
# A batch of rows with more runs than this is read with one selectrows() + one
# get*np call (casacore merges the runs in C++) instead of one call per run.
FRAGMENTED_RUNS = 64
# Maximum number of rows in one selectrows() reference table (8 bytes per row).
MAX_SELECTROWS_ROWS = 2**20
# Segments of consecutive rows (and grid cells) shorter than this are read through
# the temporary + scatter path (batched) rather than with one call each.
MIN_DIRECT_ROWS = 16

# casacore column value type -> numpy dtype that python-casacore uses for it
CASACORE_TO_NUMPY_DTYPE = {
    "boolean": np.dtype(np.bool_),
    "int": np.dtype(np.int32),
    "float": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "complex": np.dtype(np.complex64),
    "dcomplex": np.dtype(np.complex128),
}


def rows_to_runs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compress row numbers into runs of consecutive rows.

    Parameters
    ----------
    rows : np.ndarray
        1-D array of row numbers, strictly increasing.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(starts, lengths)``, both int64: run ``i`` holds the rows
        ``starts[i], ..., starts[i] + lengths[i] - 1``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    run_offsets = np.flatnonzero(np.diff(rows) != 1) + 1
    run_offsets = np.concatenate(([0], run_offsets))
    lengths = np.diff(np.concatenate((run_offsets, [rows.size])))
    return rows[run_offsets], lengths.astype(np.int64)


def runs_to_rows(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    Expand runs of consecutive rows into row numbers (inverse of ``rows_to_runs``).

    Parameters
    ----------
    starts : np.ndarray
        First row of every run.
    lengths : np.ndarray
        Number of rows of every run.

    Returns
    -------
    np.ndarray
        int64 row numbers, in run order.
    """
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if starts.shape != lengths.shape:
        raise ValueError(
            f"Run starts and lengths differ in shape: {starts.shape} vs {lengths.shape}"
        )
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    # offset of every run's first row in the output, subtracted from a running index
    offsets = np.cumsum(lengths) - lengths
    return np.repeat(starts - offsets, lengths) + np.arange(total, dtype=np.int64)


def group_row_runs(
    row_group: np.ndarray, n_groups: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Runs of consecutive rows of every group, computed for all groups at once.

    Parameters
    ----------
    row_group : np.ndarray
        Group index of every row (``row_group[row]``), or -1 for a row that
        belongs to no group.
    n_groups : int
        Number of groups (group indices are ``0 .. n_groups-1``).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        For every group, ``(starts, lengths)`` of its rows as in
        ``rows_to_runs`` (ascending row order).
    """
    run_starts, run_lengths, bounds = group_row_runs_flat(row_group, n_groups)
    return [
        (run_starts[lo:hi].copy(), run_lengths[lo:hi].copy())
        for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True)
    ]


def group_row_runs_flat(
    row_group: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``group_row_runs`` as three flat arrays (no per-group arrays).

    Parameters
    ----------
    row_group : np.ndarray
        Group index of every row (``row_group[row]``), or -1 for a row that
        belongs to no group.
    n_groups : int
        Number of groups (group indices are ``0 .. n_groups-1``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(run_starts, run_lengths, bounds)`` (int64): the runs of group ``g``
        are ``run_starts[bounds[g]:bounds[g + 1]]`` (ascending) with the
        corresponding ``run_lengths``.
    """
    row_group = np.asarray(row_group)
    empty = np.empty(0, dtype=np.int64)
    if row_group.size == 0:
        return empty, empty.copy(), np.zeros(n_groups + 1, dtype=np.int64)

    # Row numbers sorted by group; a stable sort keeps them ascending in a group.
    order = np.argsort(row_group, kind="stable")
    sorted_group = row_group[order]
    first_in_group = int(np.searchsorted(sorted_group, 0))
    rows = order[first_in_group:].astype(np.int64, copy=False)
    sorted_group = sorted_group[first_in_group:]
    del order
    if rows.size == 0:
        return empty, empty.copy(), np.zeros(n_groups + 1, dtype=np.int64)

    new_run = np.empty(rows.size, dtype=bool)
    new_run[0] = True
    new_run[1:] = (sorted_group[1:] != sorted_group[:-1]) | (np.diff(rows) != 1)
    run_offsets = np.flatnonzero(new_run)
    del new_run
    run_starts = rows[run_offsets]
    run_lengths = np.diff(np.concatenate((run_offsets, [rows.size]))).astype(np.int64)
    run_group = sorted_group[run_offsets]
    bounds = np.searchsorted(run_group, np.arange(n_groups + 1)).astype(np.int64)
    return run_starts, run_lengths, bounds


def column_dtype(table: tables.table, col: str) -> np.dtype:
    """
    The numpy dtype of a column's values as stored (the dtype python-casacore
    returns from ``getcol`` and the only buffer dtype it fills in place).

    Parameters
    ----------
    table : tables.table
        Table with the column.
    col : str
        Column name.

    Returns
    -------
    np.dtype
        dtype for the column's value type.

    Raises
    ------
    TypeError
        If the column's value type cannot be read into a numpy buffer.
    """
    value_type = table.getcoldesc(col)["valueType"]
    try:
        return CASACORE_TO_NUMPY_DTYPE[value_type]
    except KeyError:
        raise TypeError(
            f"Column {col} has value type {value_type!r}, which cannot be read into "
            "a numpy buffer"
        ) from None


def _check_max_elems(max_elems: int) -> int:
    max_elems = int(max_elems)
    if not 1 <= max_elems <= MAX_ELEMS_LIMIT:
        raise ValueError(
            f"max_elems must be between 1 and {MAX_ELEMS_LIMIT} (python-casacore "
            f"issue #130), got {max_elems}"
        )
    return max_elems


def _check_buffer(table: tables.table, col: str, buf: np.ndarray) -> None:
    flags = buf.flags
    if not (flags.c_contiguous and flags.aligned and flags.writeable):
        raise ValueError(
            f"The buffer for column {col} must be C-contiguous, aligned and "
            f"writeable (python-casacore fills unaligned buffers silently not at "
            f"all), got flags:\n{flags}"
        )
    expected = column_dtype(table, col)
    if buf.dtype != expected or not buf.dtype.isnative:
        raise TypeError(
            f"The buffer for column {col} has dtype {buf.dtype}, the column needs "
            f"{expected} (native byte order); any other dtype is converted through "
            "hidden full-size temporaries"
        )


def _check_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError(f"rows must be 1-D, got shape {rows.shape}")
    if rows.size > 1 and not np.all(rows[1:] > rows[:-1]):
        raise ValueError(
            "rows must be strictly increasing (read sorted rows and scatter them "
            "afterwards: unsorted rows defeat the casacore tile cache)"
        )
    return rows


def _cell_slicer(
    cell_ndim: int, chan: slice | None, pol: slice | None
) -> tuple[list[int], list[int]] | None:
    """blc/trc (numpy axis order, inclusive trc, -1 = whole axis) for getcolslicenp."""
    if chan is None and pol is None:
        return None
    if cell_ndim != 2:
        raise ValueError(
            f"A channel/polarization range needs a 2-D (chan, pol) cell, the cell has "
            f"{cell_ndim} dimensions"
        )
    blc, trc = [], []
    for name, slc in (("chan", chan), ("pol", pol)):
        if slc is None:
            blc.append(-1)
            trc.append(-1)
            continue
        if (
            slc.step not in (None, 1)
            or slc.start is None
            or slc.stop is None
            or not 0 <= slc.start < slc.stop
        ):
            raise ValueError(
                f"{name} must be a slice(start, stop) with 0 <= start < stop and "
                f"step 1, got {slc}"
            )
        blc.append(int(slc.start))
        trc.append(int(slc.stop) - 1)
    return blc, trc


def _read_run(
    table: tables.table,
    col: str,
    start_row: int,
    nrow: int,
    out: np.ndarray,
    slicer: tuple[list[int], list[int]] | None,
    table_nrows: int,
    rows_per_call: int,
    stats: dict[str, int],
) -> None:
    """Read the consecutive rows [start_row, start_row + nrow) into out[:nrow]."""
    done = 0
    while done < nrow:
        n = min(rows_per_call, nrow - done)
        row0 = start_row + done
        dst = out[done : done + n]
        if row0 == 0 and n == table_nrows:
            # Never request the whole column: casacore's whole-column path does not
            # check for undefined cells (SIGSEGV on TiledShapeStMan columns).
            if n > 1:
                n -= 1
                dst = out[done : done + n]
            else:
                ref = table.selectrows([0])
                try:
                    if slicer is None:
                        ref.getcolnp(col, dst)
                    else:
                        ref.getcolslicenp(col, dst, slicer[0], slicer[1], [])
                finally:
                    ref.close()
                stats["calls"] = stats.get("calls", 0) + 1
                stats["selectrows_calls"] = stats.get("selectrows_calls", 0) + 1
                done += n
                continue
        if slicer is None:
            table.getcolnp(col, dst, row0, n)
        else:
            table.getcolslicenp(col, dst, slicer[0], slicer[1], [], row0, n)
        stats["calls"] = stats.get("calls", 0) + 1
        done += n


def _read_rows_unchecked(
    table: tables.table,
    col: str,
    rows: np.ndarray,
    out: np.ndarray,
    slicer: tuple[list[int], list[int]] | None,
    table_nrows: int,
    rows_per_call: int,
    stats: dict[str, int],
) -> None:
    """read_rows without the argument checks (rows sorted, buffer checked)."""
    batch_rows = min(rows_per_call, MAX_SELECTROWS_ROWS)
    for b0 in range(0, rows.size, batch_rows):
        batch = rows[b0 : b0 + batch_rows]
        dst = out[b0 : b0 + batch.size]
        starts, lengths = rows_to_runs(batch)
        if starts.size <= FRAGMENTED_RUNS:
            offset = 0
            for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
                _read_run(
                    table,
                    col,
                    start,
                    length,
                    dst[offset : offset + length],
                    slicer,
                    table_nrows,
                    rows_per_call,
                    stats,
                )
                offset += length
        else:
            # Fragmented rows: one reference table, casacore merges the runs.
            ref = table.selectrows(batch)
            try:
                if slicer is None:
                    ref.getcolnp(col, dst)
                else:
                    ref.getcolslicenp(col, dst, slicer[0], slicer[1], [])
            finally:
                ref.close()
            stats["calls"] = stats.get("calls", 0) + 1
            stats["selectrows_calls"] = stats.get("selectrows_calls", 0) + 1


def read_rows(
    table: tables.table,
    col: str,
    rows: np.ndarray,
    out: np.ndarray,
    chan: slice | None = None,
    pol: slice | None = None,
    max_elems: int = DEFAULT_MAX_ELEMS,
    stats: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Read the cells of ``rows`` of one column into ``out`` (``out[i]`` = cell of
    ``rows[i]``), optionally a channel / polarization range of every cell.

    Contiguous runs of rows are read with one ``getcolnp`` / ``getcolslicenp``
    call each on the base table. When a batch of rows is fragmented into many
    runs, it is read with one ``selectrows`` reference table and one call
    instead. No call reads more than ``max_elems`` elements.

    Parameters
    ----------
    table : tables.table
        Base table (not a TaQL selection) holding the column.
    col : str
        Column name.
    rows : np.ndarray
        Row numbers of ``table``, strictly increasing.
    out : np.ndarray
        Output buffer of shape ``(len(rows),) + cell_shape`` (with the channel
        / polarization range applied), C-contiguous, aligned, writeable, with
        exactly the column dtype (see ``column_dtype``).
    chan : slice | None, optional
        Channel range ``slice(start, stop)`` to read (2-D cells only), by
        default all channels. A range that does not cover whole tiles makes
        casacore size the tile cache for the whole window (up to the whole
        hypercube); bound it first with ``table.setmaxcachesize(col, MiB)``.
    pol : slice | None, optional
        Polarization range ``slice(start, stop)`` to read (2-D cells only), by
        default all polarizations (same tile-cache caveat as ``chan``).
    max_elems : int, optional
        Maximum number of elements per casacore call, at most 2**29 (a call
        always reads at least one row).
    stats : dict[str, int] | None, optional
        Dict to accumulate call counters into ("calls", "selectrows_calls").

    Returns
    -------
    dict[str, int]
        The call counters (``stats``, if given).
    """
    stats = {} if stats is None else stats
    rows = _check_rows(rows)
    max_elems = _check_max_elems(max_elems)
    if out.shape[:1] != (rows.size,):
        raise ValueError(
            f"out has {out.shape[:1]} rows along its first axis, expected {rows.size}"
        )
    if rows.size == 0:
        return stats
    _check_buffer(table, col, out)
    slicer = _cell_slicer(out.ndim - 1, chan, pol)
    table_nrows = table.nrows()
    if rows[-1] >= table_nrows or rows[0] < 0:
        raise IndexError(
            f"Rows [{rows[0]}, {rows[-1]}] out of range for a table of {table_nrows} rows"
        )
    cell_elems = int(np.prod(out.shape[1:], dtype=np.int64)) or 1
    rows_per_call = max(1, max_elems // cell_elems)
    _read_rows_unchecked(
        table, col, rows, out, slicer, table_nrows, rows_per_call, stats
    )
    return stats


def read_column_rows(
    table: tables.table,
    col: str,
    rows: np.ndarray,
    max_elems: int = DEFAULT_MAX_ELEMS,
) -> np.ndarray:
    """
    Read one column for a set of rows into a new array of the column dtype, as
    ``getcol`` on a selection of those rows would return it (but without
    casacore's full-size intermediate copy). Meant for scalar and small columns.

    Parameters
    ----------
    table : tables.table
        Base table holding the column.
    col : str
        Column name.
    rows : np.ndarray
        Row numbers of ``table``, strictly increasing.
    max_elems : int, optional
        Maximum number of elements per casacore call.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(rows),) + cell_shape`` (cell shape of the first
        row; rows of a different shape raise).
    """
    rows = _check_rows(rows)
    if table.isscalarcol(col):
        cell_shape: tuple[int, ...] = ()
    elif rows.size:
        cell_shape = parse_shape_string(
            table.getcolshapestring(col, int(rows[0]), 1)[0]
        )
    else:
        cell_shape = ()
    try:
        dtype = column_dtype(table, col)
    except TypeError:
        # e.g. string columns (no in-place reads): python-casacore getcol per
        # run, in calls of at most max_elems elements
        parts = [
            np.asarray(part)
            for part in getcol_chunks(table, col, rows, cell_shape, max_elems)
        ]
        return np.concatenate(parts) if parts else np.empty((0,) + cell_shape)
    out = np.empty((rows.size,) + cell_shape, dtype=dtype)
    read_rows(table, col, rows, out, max_elems=max_elems)
    return out


def getcol_chunks(
    table: tables.table,
    col: str,
    rows: np.ndarray,
    cell_shape: tuple[int, ...],
    max_elems: int = DEFAULT_MAX_ELEMS,
) -> list[Any]:
    """
    python-casacore ``getcol`` results for sorted ``rows`` of a column, one per
    call of at most ``max_elems`` elements (cells of ``cell_shape``) over a run
    of consecutive rows, in row order. For value types that cannot be read in
    place (strings, short, uchar, ...); no call covers the whole column.

    Parameters
    ----------
    table : tables.table
        Table holding the column.
    col : str
        Column name.
    rows : np.ndarray
        Row numbers of ``table``, strictly increasing.
    cell_shape : tuple[int, ...]
        Cell shape, to bound the calls.
    max_elems : int, optional
        Maximum number of elements per call.

    Returns
    -------
    list[Any]
        What ``getcol`` returned for every call (arrays, or lists for strings).
    """
    rows = _check_rows(rows)
    max_elems = _check_max_elems(max_elems)
    rows_per_call = max(1, max_elems // (int(np.prod(cell_shape, dtype=np.int64)) or 1))
    table_nrows = table.nrows()
    parts = []
    starts, lengths = rows_to_runs(rows)
    for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
        done = 0
        while done < length:
            n = min(rows_per_call, length - done)
            if start + done == 0 and n == table_nrows:
                # never the whole column (see _read_run)
                if n > 1:
                    n -= 1
                else:
                    ref = table.selectrows([0])
                    try:
                        parts.append(ref.getcol(col))
                    finally:
                        ref.close()
                    done += 1
                    continue
            parts.append(table.getcol(col, start + done, n))
            done += n
    return parts


def parse_shape_string(shape_string: str) -> tuple[int, ...]:
    """
    Parse a casacore cell shape string as returned by ``getcolshapestring``
    (e.g. "[4, 2]", already in numpy axis order) into a shape tuple.
    """
    return tuple(int(dim) for dim in shape_string.strip("[]").split(", "))


@dataclasses.dataclass(frozen=True)
class RowGridPlan:
    """
    How the rows of a partition map onto the cells of a dense (time, baseline)
    grid, worked out once and reused for every column.

    Attributes
    ----------
    rows : np.ndarray
        MAIN row numbers of the partition (int64, strictly increasing).
    gidx : np.ndarray
        Flat grid cell (``time_index * n_baselines + baseline_index``) of every
        row (int64).
    ncells : int
        Number of grid cells (n_times * n_baselines).
    direct_offsets : np.ndarray
        Offsets (into ``rows``) of the segments read straight into the grid:
        both the row number and the grid cell advance by one along a segment.
    direct_lengths : np.ndarray
        Number of rows of every direct segment.
    scatter_idx : np.ndarray
        Offsets (into ``rows``) of all other rows, ascending. They are read into
        a bounded temporary and scattered into the grid.
    grid_is_full : bool
        Whether every grid cell receives at least one row (no padding needed).
    n_duplicate_rows : int
        Number of rows whose cell also receives another row. Those rows are
        always scattered, in row order, so the last row wins (as with a numpy
        fancy-index assignment of all rows).
    """

    rows: np.ndarray
    gidx: np.ndarray
    ncells: int
    direct_offsets: np.ndarray
    direct_lengths: np.ndarray
    scatter_idx: np.ndarray
    grid_is_full: bool
    n_duplicate_rows: int


def make_row_grid_plan(
    rows: np.ndarray,
    gidx: np.ndarray,
    ncells: int,
    min_direct_rows: int = MIN_DIRECT_ROWS,
) -> RowGridPlan:
    """
    Work out which rows of a partition can be read straight into the grid.

    Parameters
    ----------
    rows : np.ndarray
        MAIN row numbers, strictly increasing.
    gidx : np.ndarray
        Flat grid cell of every row, ``0 <= gidx < ncells``.
    ncells : int
        Number of cells of the grid.
    min_direct_rows : int, optional
        Shortest segment read straight into the grid. Shorter segments go
        through the (batched) temporary + scatter path.

    Returns
    -------
    RowGridPlan
        The plan, see ``RowGridPlan``.
    """
    rows = _check_rows(rows)
    gidx = np.asarray(gidx, dtype=np.int64)
    if gidx.shape != rows.shape:
        raise ValueError(f"rows and gidx differ in shape: {rows.shape} vs {gidx.shape}")
    nrows = rows.size
    if nrows and (gidx.min() < 0 or gidx.max() >= ncells):
        raise IndexError(f"Grid indices out of range for a grid of {ncells} cells")

    covered = np.zeros(ncells, dtype=bool)
    covered[gidx] = True
    n_covered = int(np.count_nonzero(covered))
    del covered
    grid_is_full = n_covered == ncells

    is_dup = None
    n_duplicate_rows = 0
    if n_covered < nrows:
        order = np.argsort(gidx, kind="stable")
        same = gidx[order[1:]] == gidx[order[:-1]]
        dup_sorted = np.zeros(nrows, dtype=bool)
        dup_sorted[1:] |= same
        dup_sorted[:-1] |= same
        is_dup = np.empty(nrows, dtype=bool)
        is_dup[order] = dup_sorted
        n_duplicate_rows = int(np.count_nonzero(is_dup))
        del order, same, dup_sorted

    if nrows == 0:
        empty = np.empty(0, dtype=np.int64)
        return RowGridPlan(rows, gidx, ncells, empty, empty, empty, grid_is_full, 0)

    breaks = (np.diff(rows) != 1) | (np.diff(gidx) != 1)
    if is_dup is not None:
        # isolate every duplicated row in its own (scattered) segment
        breaks |= is_dup[1:] | is_dup[:-1]
    seg_offsets = np.concatenate(([0], np.flatnonzero(breaks) + 1)).astype(np.int64)
    seg_lengths = np.diff(np.concatenate((seg_offsets, [nrows]))).astype(np.int64)
    direct = seg_lengths >= max(1, int(min_direct_rows))
    if is_dup is not None:
        direct &= ~is_dup[seg_offsets]
    scatter_idx = np.flatnonzero(~np.repeat(direct, seg_lengths)).astype(np.int64)

    return RowGridPlan(
        rows=rows,
        gidx=gidx,
        ncells=int(ncells),
        direct_offsets=seg_offsets[direct],
        direct_lengths=seg_lengths[direct],
        scatter_idx=scatter_idx,
        grid_is_full=grid_is_full,
        n_duplicate_rows=n_duplicate_rows,
    )


def read_rows_to_grid(
    table: tables.table,
    col: str,
    plan: RowGridPlan,
    grid: np.ndarray,
    chan: slice | None = None,
    pol: slice | None = None,
    max_elems: int = DEFAULT_MAX_ELEMS,
    max_tmp_bytes: int = DEFAULT_MAX_TMP_BYTES,
    stats: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Read one column of the rows of ``plan`` into a dense grid.

    Cells that receive no row keep their previous value (the caller pre-fills
    the grid with the pad value if ``plan.grid_is_full`` is False).

    The table is read in one ascending pass over the rows: the direct segments
    and the scattered rows between them are read in row order (the scattered
    rows collect in the temporary, which is scattered into the grid whenever
    it is full). Reading all direct segments first and the scattered rows
    afterwards would read every tile that holds both kinds of rows twice: the
    tile cache of the tiled storage managers keeps about one row-slab of tiles.

    Parameters
    ----------
    table : tables.table
        Base table holding the column.
    col : str
        Column name.
    plan : RowGridPlan
        Rows of the partition and their grid cells (``make_row_grid_plan``).
    grid : np.ndarray
        C-contiguous output of shape ``(n_times, n_baselines) + cell_shape``
        with ``n_times * n_baselines == plan.ncells`` and the channel /
        polarization range applied to the cell shape. Rows are read straight
        into it only if its dtype is the column dtype; otherwise all rows go
        through the temporary (of the column dtype) and are cast by the scatter.
    chan : slice | None, optional
        Channel range to read (2-D cells only).
    pol : slice | None, optional
        Polarization range to read (2-D cells only).
    max_elems : int, optional
        Maximum number of elements per casacore call, at most 2**29.
    max_tmp_bytes : int, optional
        Maximum size of the temporary buffer of the scatter path (at least one
        row is always read at a time).
    stats : dict[str, int] | None, optional
        Dict to accumulate counters into: "calls", "selectrows_calls",
        "direct_rows", "scatter_rows", "max_tmp_bytes".

    Returns
    -------
    dict[str, int]
        The counters (``stats``, if given).
    """
    stats = {} if stats is None else stats
    max_elems = _check_max_elems(max_elems)
    if grid.ndim < 2 or grid.shape[0] * grid.shape[1] != plan.ncells:
        raise ValueError(
            f"Grid of shape {grid.shape} does not have the {plan.ncells} (time, "
            "baseline) cells of the plan"
        )
    if not (grid.flags.c_contiguous and grid.flags.writeable):
        raise ValueError(
            f"The grid for column {col} must be C-contiguous and writeable"
        )
    nrows = plan.rows.size
    if nrows == 0:
        return stats

    col_dt = column_dtype(table, col)
    cell_shape = grid.shape[2:]
    slicer = _cell_slicer(len(cell_shape), chan, pol)
    table_nrows = table.nrows()
    if plan.rows[-1] >= table_nrows:
        raise IndexError(
            f"Row {plan.rows[-1]} out of range for a table of {table_nrows} rows"
        )
    cell_elems = int(np.prod(cell_shape, dtype=np.int64)) or 1
    rows_per_call = max(1, max_elems // cell_elems)
    flat = grid.reshape((plan.ncells,) + cell_shape)

    direct_ok = grid.dtype == col_dt and grid.dtype.isnative and grid.flags.aligned
    if direct_ok:
        direct_offsets = plan.direct_offsets.tolist()
        direct_lengths = plan.direct_lengths.tolist()
        scatter_idx = plan.scatter_idx
    else:
        direct_offsets, direct_lengths = [], []
        scatter_idx = np.arange(nrows, dtype=np.int64)

    tmp_full = None
    if scatter_idx.size:
        row_bytes = cell_elems * col_dt.itemsize
        tmp_rows = max(1, min(rows_per_call, int(max_tmp_bytes) // row_bytes))
        tmp_rows = min(tmp_rows, scatter_idx.size)
        tmp_full = np.empty((tmp_rows,) + cell_shape, dtype=col_dt)
        stats["max_tmp_bytes"] = max(stats.get("max_tmp_bytes", 0), tmp_full.nbytes)

    # Rows are offsets into plan.rows (ascending row numbers). The scattered rows
    # before direct segment k are scatter_idx[:scatter_before[k]].
    scatter_before = np.searchsorted(scatter_idx, direct_offsets).tolist()
    scatter_before.append(int(scatter_idx.size))
    tmp_start = 0  # scatter_idx position of the first row held in the temporary
    tmp_fill = 0  # rows held in the temporary

    def flush_tmp() -> None:
        nonlocal tmp_start, tmp_fill
        idx = scatter_idx[tmp_start : tmp_start + tmp_fill]
        # numpy assigns in index order: of duplicated cells the last row wins
        flat[plan.gidx[idx]] = tmp_full[:tmp_fill]
        stats["scatter_rows"] = stats.get("scatter_rows", 0) + tmp_fill
        tmp_start += tmp_fill
        tmp_fill = 0

    scatter_pos = 0
    for k, scatter_end in enumerate(scatter_before):
        # the scattered rows before direct segment k (or after the last one)
        while scatter_pos < scatter_end:
            n = min(scatter_end - scatter_pos, tmp_full.shape[0] - tmp_fill)
            idx = scatter_idx[scatter_pos : scatter_pos + n]
            _read_rows_unchecked(
                table,
                col,
                plan.rows[idx],
                tmp_full[tmp_fill : tmp_fill + n],
                slicer,
                table_nrows,
                rows_per_call,
                stats,
            )
            scatter_pos += n
            tmp_fill += n
            if tmp_fill == tmp_full.shape[0]:
                flush_tmp()
        if k < len(direct_offsets):
            offset, length = direct_offsets[k], direct_lengths[k]
            g0 = int(plan.gidx[offset])
            _read_run(
                table,
                col,
                int(plan.rows[offset]),
                length,
                flat[g0 : g0 + length],
                slicer,
                table_nrows,
                rows_per_call,
                stats,
            )
            stats["direct_rows"] = stats.get("direct_rows", 0) + length
    if tmp_fill:
        flush_tmp()
    return stats


class TimeChunkRows:
    """
    The rows of a partition grouped by chunks of times (the blocks of the lazy
    columns of parallel_mode="time"), computed once per partition and shared,
    not copied, by every block of every column.

    It holds references to the partition's row numbers and time / baseline
    indices (alive for the whole partition anyway) plus, only when the rows are
    not already ordered by time chunk, a permutation (8 bytes per row). Every
    block computes its own rows and grid cells from these when it runs, so
    the dask graph holds no per-block or per-column copies of the row indices.

    Parameters
    ----------
    rows : np.ndarray
        MAIN row numbers of the partition (strictly increasing).
    tidxs : np.ndarray
        Time index of every partition row.
    bidxs : np.ndarray
        Baseline index of every partition row.
    time_chunks : tuple[int, ...]
        Number of times of every chunk (as dask chunks along time).
    num_baselines : int
        Number of baselines of the grid.
    """

    __slots__ = (
        "rows",
        "tidxs",
        "bidxs",
        "order",
        "row_bounds",
        "time_bounds",
        "num_baselines",
    )

    def __init__(
        self,
        rows: np.ndarray,
        tidxs: np.ndarray,
        bidxs: np.ndarray,
        time_chunks: tuple[int, ...],
        num_baselines: int,
    ):
        if len(tidxs) != len(rows) or len(bidxs) != len(rows):
            raise ValueError(
                f"Got {len(tidxs)} time and {len(bidxs)} baseline indices for "
                f"{len(rows)} partition rows"
            )
        self.rows = rows
        self.tidxs = tidxs
        self.bidxs = bidxs
        self.num_baselines = int(num_baselines)
        self.time_bounds = np.cumsum(
            (0,) + tuple(int(n) for n in time_chunks), dtype=np.int64
        )
        n_chunks = len(time_chunks)
        chunk_of_row = np.searchsorted(self.time_bounds, tidxs, side="right") - 1
        if chunk_of_row.size < 2 or bool(np.all(chunk_of_row[1:] >= chunk_of_row[:-1])):
            # rows already ordered by time chunk (e.g. time-ordered MSs)
            self.order = None
        else:
            # a stable sort keeps the rows of a chunk ascending
            self.order = np.argsort(chunk_of_row, kind="stable")
            chunk_of_row = chunk_of_row[self.order]
        self.row_bounds = np.searchsorted(chunk_of_row, np.arange(n_chunks + 1))

    @property
    def n_chunks(self) -> int:
        """Number of time chunks."""
        return int(self.time_bounds.size - 1)

    def n_runs(self) -> int:
        """
        Number of runs of consecutive rows when the chunks are read one after
        the other (the runs of every chunk's ascending rows, summed): a measure
        of the read calls (and tile-cache restarts) of a chunk-by-chunk read.

        Returns
        -------
        int
            Sum over the chunks of the runs of consecutive rows.
        """
        n_rows = int(self.row_bounds[-1]) if self.row_bounds.size else 0
        if n_rows == 0:
            return 0
        rows = self.rows if self.order is None else self.rows[self.order]
        breaks = np.diff(np.asarray(rows, dtype=np.int64)) != 1
        del rows
        # a new chunk always starts a new run
        inner = self.row_bounds[1:-1]
        inner = inner[(inner > 0) & (inner < n_rows)]
        breaks[inner - 1] = True
        return int(np.count_nonzero(breaks)) + 1

    def chunk(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        The rows of time chunk ``k`` and their cells in the chunk's grid.

        Parameters
        ----------
        k : int
            Chunk index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(rows, gidx)``: MAIN row numbers (ascending) and flat cells
            ``(time_index - first time of the chunk) * num_baselines +
            baseline_index`` (both int64).
        """
        lo, hi = int(self.row_bounds[k]), int(self.row_bounds[k + 1])
        idx = slice(lo, hi) if self.order is None else self.order[lo:hi]
        rows = np.asarray(self.rows[idx], dtype=np.int64)
        gidx = (
            np.asarray(self.tidxs[idx], dtype=np.int64) - int(self.time_bounds[k])
        ) * self.num_baselines + np.asarray(self.bidxs[idx], dtype=np.int64)
        return rows, gidx


class MainTableRows:
    """
    A partition of the MSv2 MAIN table: the base table plus the partition's row
    numbers, read without TaQL.

    It provides the subset of the casacore table API that the converter uses on
    its TaQL partition selections (``nrows``, ``colnames``, ``getcol``,
    ``getcell``, ``iscelldefined``, ``isscalarcol``, ``rownumbers``), with row
    numbers relative to the partition, so the same converter code runs on both.
    Columns are read with ``read_rows`` (bounded, in place, no TaQL).

    Parameters
    ----------
    table : tables.table
        The opened MAIN table. It is owned by the caller (``close`` does not
        close it).
    rows : np.ndarray
        MAIN row numbers of the partition, strictly increasing.
    max_elems : int, optional
        Maximum number of elements per casacore call.
    """

    def __init__(
        self,
        table: tables.table,
        rows: np.ndarray,
        max_elems: int = DEFAULT_MAX_ELEMS,
    ):
        rows = _check_rows(rows)
        if rows.size and (rows[0] < 0 or rows[-1] >= table.nrows()):
            raise IndexError(
                f"Partition rows [{rows[0]}, {rows[-1]}] out of range for a MAIN "
                f"table of {table.nrows()} rows"
            )
        self._table = table
        self._rows = rows
        self._max_elems = _check_max_elems(max_elems)
        # Plans cached for the index arrays they were computed from. The arrays
        # are kept and compared by identity ("is"): an id() alone can be reused
        # by a new array once the old one is freed.
        self._grid_plan_key: tuple[Any, ...] | None = None
        self._grid_plan: RowGridPlan | None = None
        self._time_chunk_key: tuple[Any, ...] | None = None
        self._time_chunk_rows: TimeChunkRows | None = None
        self._time_chunk_delayed: Any = None

    @property
    def table(self) -> tables.table:
        """The base MAIN table."""
        return self._table

    @property
    def rows(self) -> np.ndarray:
        """MAIN row numbers of the partition (int64, strictly increasing)."""
        return self._rows

    @property
    def max_elems(self) -> int:
        """Maximum number of elements per casacore call."""
        return self._max_elems

    def name(self) -> str:
        """Path of the MAIN table."""
        return self._table.name()

    def nrows(self) -> int:
        """Number of rows of the partition."""
        return int(self._rows.size)

    def colnames(self) -> list[str]:
        """Column names of the MAIN table."""
        return self._table.colnames()

    def isscalarcol(self, col: str) -> bool:
        """Whether ``col`` is a scalar column."""
        return self._table.isscalarcol(col)

    def iscelldefined(self, col: str, rownr: int) -> bool:
        """Whether the cell of partition row ``rownr`` is defined."""
        return self._table.iscelldefined(col, int(self._rows[rownr]))

    def getcell(self, col: str, rownr: int) -> Any:
        """The cell of partition row ``rownr`` (as ``tables.table.getcell``)."""
        return self._table.getcell(col, int(self._rows[rownr]))

    def getcol(self, col: str, startrow: int = 0, nrow: int = -1) -> np.ndarray:
        """
        Column values of partition rows ``startrow .. startrow + nrow - 1``
        (all rows from ``startrow`` if ``nrow < 0``), as ``tables.table.getcol``.
        """
        stop = None if nrow < 0 else startrow + nrow
        return read_column_rows(
            self._table, col, self._rows[startrow:stop], max_elems=self._max_elems
        )

    def rownumbers(self) -> list[int]:
        """MAIN row numbers of the partition (as ``tables.table.rownumbers``)."""
        return self._rows.tolist()

    def close(self) -> None:
        """Release cached state. The MAIN table is closed by its owner."""
        self.release_plans()

    def release_plans(self) -> None:
        """
        Drop the cached grid plan and time-chunk rows (index arrays of 8-24
        bytes per row), e.g. once every column of the partition has been read.
        """
        self._grid_plan = None
        self._grid_plan_key = None
        self._time_chunk_rows = None
        self._time_chunk_key = None
        self._time_chunk_delayed = None

    @staticmethod
    def _same_key(cached: tuple[Any, ...] | None, key: tuple[Any, ...]) -> bool:
        # arrays by identity (the cached key holds them, so their ids cannot be
        # reused), everything else by value
        return (
            cached is not None
            and len(cached) == len(key)
            and all(
                a is b
                if isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
                else a == b
                for a, b in zip(cached, key, strict=True)
            )
        )

    def grid_plan(
        self,
        tidxs: np.ndarray,
        bidxs: np.ndarray,
        time_baseline_shape: tuple[int, int],
    ) -> RowGridPlan:
        """
        The ``RowGridPlan`` of this partition for the given time / baseline
        indices of its rows (computed on first use, then reused for every column
        read with the same index arrays). The arrays must not be modified in
        place while the plan is cached (see ``release_plans``).

        Parameters
        ----------
        tidxs : np.ndarray
            Time index of every partition row.
        bidxs : np.ndarray
            Baseline index of every partition row.
        time_baseline_shape : tuple[int, int]
            (n_times, n_baselines) of the grid.

        Returns
        -------
        RowGridPlan
            Plan of the partition rows.
        """
        key = (tidxs, bidxs, tuple(int(n) for n in time_baseline_shape))
        if self._grid_plan is None or not self._same_key(self._grid_plan_key, key):
            if len(tidxs) != self._rows.size or len(bidxs) != self._rows.size:
                raise ValueError(
                    f"Got {len(tidxs)} time and {len(bidxs)} baseline indices for "
                    f"{self._rows.size} partition rows"
                )
            nbaselines = int(time_baseline_shape[1])
            gidx = np.asarray(tidxs, dtype=np.int64) * nbaselines + np.asarray(
                bidxs, dtype=np.int64
            )
            self._grid_plan = make_row_grid_plan(
                self._rows, gidx, int(time_baseline_shape[0]) * nbaselines
            )
            self._grid_plan_key = key
        return self._grid_plan

    def time_chunk_rows(
        self,
        tidxs: np.ndarray,
        bidxs: np.ndarray,
        time_chunks: tuple[int, ...],
        num_baselines: int,
    ) -> TimeChunkRows:
        """
        The ``TimeChunkRows`` of this partition for the given time / baseline
        indices and time chunks (computed on first use, then shared by every
        column read with the same arguments; same caching rule as
        ``grid_plan``).

        Parameters
        ----------
        tidxs : np.ndarray
            Time index of every partition row.
        bidxs : np.ndarray
            Baseline index of every partition row.
        time_chunks : tuple[int, ...]
            Number of times of every chunk.
        num_baselines : int
            Number of baselines of the grid.

        Returns
        -------
        TimeChunkRows
            Rows of every time chunk.
        """
        key = (
            tidxs,
            bidxs,
            tuple(int(n) for n in time_chunks),
            int(num_baselines),
        )
        if self._time_chunk_rows is None or not self._same_key(
            self._time_chunk_key, key
        ):
            self._time_chunk_rows = TimeChunkRows(
                self._rows, tidxs, bidxs, time_chunks, num_baselines
            )
            self._time_chunk_key = key
            self._time_chunk_delayed = None
        return self._time_chunk_rows

    def time_chunk_rows_delayed(self, chunk_rows: TimeChunkRows) -> Any:
        """
        A dask Delayed of ``chunk_rows`` (as returned by ``time_chunk_rows``),
        created once, so that the blocks of every column depend on one graph key
        (the threaded scheduler passes the object itself to every block, a
        distributed scheduler sends it once per worker).

        Parameters
        ----------
        chunk_rows : TimeChunkRows
            The cached time-chunk rows of this partition.

        Returns
        -------
        dask.delayed.Delayed
            Delayed whose value is ``chunk_rows``.
        """
        if chunk_rows is not self._time_chunk_rows:
            raise ValueError("chunk_rows is not the cached TimeChunkRows")
        if self._time_chunk_delayed is None:
            import dask

            self._time_chunk_delayed = dask.delayed(
                chunk_rows, pure=False, traverse=False
            )
        return self._time_chunk_delayed
