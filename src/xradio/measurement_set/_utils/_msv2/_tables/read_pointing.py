"""
POINTING read once per conversion, selected per partition in numpy.

Without a sub-table cache, every partition reads the whole POINTING TIME column
(to project its time range), runs a TaQL selection over POINTING, loads all its
columns for the selected rows and pivots them to (time, antenna) with xarray.
Partitions overlap in time, so the table is read many times over (11.2x for the
VLASS test MS).

With a sub-table cache (``subtable_cache.py``) the columns the pointing_xds is
built from are read once, sorted by TIME and shared by the partitions. Every
partition then selects its rows (a TIME range and a list of antennas) with
``searchsorted`` + ``isin`` and pivots them with numpy. The resulting generic
dataset is identical, for every variable the pointing_xds is built from, to the
one ``load_generic_table`` + ``redimension_ms_subtable`` produce:

- the same rows: the TaQL ``TIME >= lo AND TIME <= hi`` (with lo/hi as the
  decimal strings TaQL parses) and ``ANTENNA_ID IN [...]``;
- the same pivot: unique sorted TIME (Unix seconds) and ANTENNA_ID, the first
  row (lowest row number) of duplicated (TIME, ANTENNA_ID) pairs, and, when
  (time, antenna) cells are missing, xarray's fill value of the promoted dtype
  cast back to the column dtype (e.g. NaN for floats, True for booleans);
- the same dimension names (they depend on the shapes of all the POINTING
  columns, see ``generic_dims``).

Memory: TIME, ANTENNA_ID and the row numbers (16 bytes per POINTING row) are
kept for the whole conversion, if they fit in ``POINTING_MAX_CACHED_INDEX_BYTES``
(otherwise the table is not cached). The data columns (33 bytes per row for
DIRECTION, ENCODER and OVER_THE_TOP) are kept too if they fit in
``POINTING_MAX_CACHED_DATA_BYTES``; otherwise every partition reads its own
rows of them (bounded ``getcolnp`` reads of ascending rows, no TaQL). The
columns are only read once a second partition of the process needs them, unless
the cache is known to be shared (see ``SubtableCache``).

Tables where that cannot be guaranteed (cells of varying shape or undefined,
unusual column types, zero-size dimensions, ...) are not cached: the caller
then falls back to the per-partition reads.
"""

import copy
import dataclasses
import os

import numpy as np
import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.read import (
    add_units_measures,
    convert_casacore_time,
    extract_table_attributes,
    find_loadable_cols,
    is_nested_ms,
    project_min_max_sorted,
    projection_tolerance,
)
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    CASACORE_TO_NUMPY_DTYPE,
    parse_shape_string,
    read_column_rows,
    read_rows,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
    active_subtable_cache,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.subtables import subt_rename_ids

try:
    # xarray's promotion rules for the fill value of missing (time, antenna)
    # cells when unstacking (Variable._unstack_once)
    from xarray.core.dtypes import maybe_promote
except ImportError:  # pragma: no cover - only if xarray moves it
    maybe_promote = None

POINTING_TABLE = "POINTING"
# Value types of scalar data columns for which tables.row() (used by the
# uncached reads of partitions with fewer than 1000 POINTING rows) and getcol()
# give the same dtype.
_SCALAR_VALUE_TYPES = ("boolean", "double")
# Elements read per call when checking that a column has cells of one shape
_CHECK_CHUNK_ELEMS = 2**20
# Data columns larger than this (all of them, whole table) are not kept in
# memory: every partition then reads its rows of them from the table.
POINTING_MAX_CACHED_DATA_BYTES = 256 * 1024 * 1024
# Tables whose TIME, ANTENNA_ID and row numbers (16 bytes per row) are larger
# than this are not cached at all (per-partition reads).
POINTING_MAX_CACHED_INDEX_BYTES = 256 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class PointingColumns:
    """
    The POINTING columns a pointing_xds is built from, sorted by TIME.

    Attributes
    ----------
    time : np.ndarray
        TIME (casacore seconds), ascending (stable sort of the table rows).
    tolerance : np.float64
        projection_tolerance() of ``time``.
    antenna_id : np.ndarray
        ANTENNA_ID of the rows, in the same order.
    row : np.ndarray
        POINTING row number of the rows, in the same order.
    data_columns : tuple[str, ...]
        Data columns (for example DIRECTION, ENCODER, OVER_THE_TOP), in table
        column order.
    data : dict[str, np.ndarray] | None
        Values of the data columns, rows in the same order as ``time``, or None
        if they are read per partition (larger than
        POINTING_MAX_CACHED_DATA_BYTES).
    data_dims : dict[str, tuple[str, ...]]
        Names of the cell dimensions of every data column, as
        load_generic_table() names them.
    var_attrs : dict[str, dict]
        Attributes load_generic_table() gives the data columns and TIME /
        ANTENNA_ID (units, measures).
    table_attrs : dict
        extract_table_attributes() of POINTING.
    bad_cols : list[str]
        Columns load_generic_table() does not load.
    table_path : str
        Path of the POINTING table.
    """

    time: np.ndarray
    tolerance: np.float64
    antenna_id: np.ndarray
    row: np.ndarray
    data_columns: tuple[str, ...]
    data: dict[str, np.ndarray] | None
    data_dims: dict[str, tuple[str, ...]]
    var_attrs: dict[str, dict]
    table_attrs: dict
    bad_cols: list[str]
    table_path: str

    @property
    def nbytes(self) -> int:
        """Memory held by the arrays."""
        return (
            self.time.nbytes
            + self.antenna_id.nbytes
            + self.row.nbytes
            + sum(values.nbytes for values in (self.data or {}).values())
        )

    def read_data(self, col: str, idx: np.ndarray) -> np.ndarray:
        """
        Values of data column ``col`` for the rows ``idx`` (unique indices into
        the sorted arrays), from memory or read from the table.
        """
        if self.data is not None:
            return self.data[col][idx]
        rows = self.row[idx].astype(np.int64)
        order = np.argsort(rows)
        with open_table_ro(self.table_path) as tb_tool:
            sorted_values = read_column_rows(tb_tool, col, rows[order])
        values = np.empty_like(sorted_values)
        values[order] = sorted_values
        return values


def uniform_cell_shape(tb_tool, col: str, nrows: int) -> tuple[int, ...] | None:
    """
    The cell shape of an array column if every cell is defined and has that
    shape (so that a getcol() of any rows succeeds), else None. Reads the
    column in bounded chunks and discards the data.

    Parameters
    ----------
    tb_tool : tables.table
        Base table.
    col : str
        Column name.
    nrows : int
        Rows of the table (> 0).

    Returns
    -------
    tuple[int, ...] | None
        The cell shape (numpy axis order, () for a scalar column), or None.
    """
    if tb_tool.isscalarcol(col):
        return ()
    dtype = CASACORE_TO_NUMPY_DTYPE.get(tb_tool.getcoldesc(col)["valueType"])
    if dtype is None:
        return None
    try:
        cell_shape = parse_shape_string(tb_tool.getcolshapestring(col, 0, 1)[0])
        rows_per_call = max(1, _CHECK_CHUNK_ELEMS // max(1, int(np.prod(cell_shape))))
        buf = np.empty((min(rows_per_call, nrows),) + cell_shape, dtype=dtype)
        for start in range(0, nrows, rows_per_call):
            nrow = min(rows_per_call, nrows - start)
            # raises on an undefined cell or a cell of another shape
            read_rows(tb_tool, col, np.arange(start, start + nrow), buf[:nrow])
    except MemoryError:
        raise
    except Exception:
        return None
    return cell_shape


def generic_dims(
    columns: list[tuple[str, bool, tuple[int, ...]]], nrows: int
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """
    The dimension names load_generic_table() gives the variables of POINTING,
    from the shapes of all its loadable columns.

    load_generic_table() names the dimensions after their position and size
    (``dim_<i>_<size>``), renames them in order of first appearance in the
    dataset (row, dim_1, dim_2, ...) and then applies subt_rename_ids. So the
    name of a dimension of one column depends on the shapes of the columns
    before it. This builds the same dataset from zero-strided placeholders.

    Parameters
    ----------
    columns : list[tuple[str, bool, tuple[int, ...]]]
        (column name, is a coordinate, cell shape) of every loaded column, in
        load order.
    nrows : int
        Number of rows (any positive number gives the same names).

    Returns
    -------
    tuple[dict[str, tuple[str, ...]], dict[str, int]]
        Dimension names of every variable (row dimension first) and the
        dataset sizes.
    """
    mcoords, mvars = {}, {}
    placeholder = np.zeros((), dtype=np.int8)
    for col, is_coord, cell_shape in columns:
        shape = (nrows,) + cell_shape
        array_data = xr.DataArray(
            np.broadcast_to(placeholder, shape),
            dims=[f"dim_{di}_{ds}" for di, ds in enumerate(shape)],
        )
        if is_coord:
            mcoords[col] = array_data
        else:
            mvars[col] = array_data
    xds = xr.Dataset(mvars, coords=mcoords)
    dims = ["row"] + [f"dim_{i}" for i in range(1, 20)]
    xds = xds.rename({dv: dims[di] for di, dv in enumerate(xds.sizes)})
    rename_ids = {
        k: v for k, v in subt_rename_ids[POINTING_TABLE].items() if k in xds.sizes
    }
    xds = xds.rename_dims(rename_ids)
    return {name: xds[name].dims for name in xds.variables}, dict(xds.sizes)


def read_pointing_columns(
    table_path: str, data_columns: tuple[str, ...]
) -> PointingColumns | None:
    """
    Reads TIME, ANTENNA_ID and the given data columns of a POINTING table once,
    sorted by TIME, if per-partition selections of them are guaranteed to give
    the same pointing data as the uncached reads.

    Parameters
    ----------
    table_path : str
        Path of the POINTING table.
    data_columns : tuple[str, ...]
        Columns the pointing_xds is built from (columns not in the table are
        ignored). DIRECTION must be in the table.

    Returns
    -------
    PointingColumns | None
        The columns, or None if the table is not suitable (missing, empty, cells
        of varying shape, too large, ...): the per-partition reads are then
        used. A MemoryError is raised (not stored as "not suitable").
    """
    try:
        return _read_pointing_columns(table_path, data_columns)
    except MemoryError:
        raise
    except Exception as exc:
        xradio_logger().debug(f"Not caching {table_path}: {exc}")
        return None


def _not_cached(table_path: str, reason: str) -> None:
    xradio_logger().debug(f"Not caching {table_path}: {reason}")


def _read_pointing_columns(
    table_path: str, data_columns: tuple[str, ...]
) -> PointingColumns | None:
    if maybe_promote is None:
        return _not_cached(table_path, "xarray.core.dtypes.maybe_promote missing")
    if not os.path.isdir(table_path):
        return _not_cached(table_path, "no table")
    table_attrs = extract_table_attributes(table_path)
    if is_nested_ms({"other": {"msv2": {"ctds_attrs": table_attrs}}}):
        return _not_cached(table_path, "looks like a MeasurementSet main table")

    with open_table_ro(table_path) as tb_tool:
        nrows = tb_tool.nrows()
        if nrows == 0:
            return _not_cached(table_path, "no rows")
        row_dtype = np.dtype(np.int32 if nrows < 2**31 else np.int64)
        index_bytes = nrows * (8 + 4 + row_dtype.itemsize)  # TIME, ANTENNA_ID, row
        if index_bytes > POINTING_MAX_CACHED_INDEX_BYTES:
            return _not_cached(table_path, f"{nrows} rows: index of {index_bytes} B")
        col_types = find_loadable_cols(tb_tool, [])
        # columns of load_generic_table's "select *, !~p/SOURCE_MODEL/"
        colnames = [col for col in tb_tool.colnames() if col != "SOURCE_MODEL"]
        if col_types.get("TIME") != "double" or col_types.get("ANTENNA_ID") != "int":
            return _not_cached(table_path, "no double TIME / int ANTENNA_ID column")
        if "DIRECTION" not in col_types:
            return _not_cached(table_path, "no DIRECTION column")

        all_rows = np.arange(nrows)
        loaded, columns, data_cells = {}, [], {}
        for col, col_type in col_types.items():
            is_coord = col.endswith("_ID") or col == "TIME"
            is_key = col in ("TIME", "ANTENNA_ID")
            is_data = col in data_columns and not is_coord
            if tb_tool.isscalarcol(col):
                if is_data and col_type not in _SCALAR_VALUE_TYPES:
                    return _not_cached(table_path, f"{col} is a {col_type} scalar")
                if is_key:
                    loaded[col] = read_column_rows(tb_tool, col, all_rows)
                cell_shape = ()
            elif is_key:
                return _not_cached(table_path, f"{col} is not a scalar column")
            elif is_data:
                if col_type not in CASACORE_TO_NUMPY_DTYPE:
                    return _not_cached(table_path, f"{col} is a {col_type} array")
                # shape of the first cell: all cells are checked below
                cell_shape = parse_shape_string(tb_tool.getcolshapestring(col, 0, 1)[0])
            else:
                cell_shape = uniform_cell_shape(tb_tool, col, nrows)
                if cell_shape is None:
                    return _not_cached(table_path, f"{col} cells vary in shape")
            if is_data:
                data_cells[col] = (CASACORE_TO_NUMPY_DTYPE[col_type], cell_shape)
            columns.append((col, is_coord, cell_shape))

        data_cols = tuple(data_cells)
        data_bytes = nrows * sum(
            dtype.itemsize * int(np.prod(cell_shape))
            for dtype, cell_shape in data_cells.values()
        )
        in_memory = data_bytes <= POINTING_MAX_CACHED_DATA_BYTES
        for col in data_cols:
            if in_memory:
                try:  # raises if a cell has another shape (or is undefined)
                    loaded[col] = read_column_rows(tb_tool, col, all_rows)
                except MemoryError:
                    raise
                except Exception as exc:
                    return _not_cached(table_path, f"{col} cells vary: {exc}")
            elif uniform_cell_shape(tb_tool, col, nrows) != data_cells[col][1]:
                return _not_cached(table_path, f"{col} cells vary in shape")

    var_dims, sizes = generic_dims(columns, nrows)
    data_dims = {col: var_dims[col][1:] for col in data_cols}
    data_sizes = {}
    for col in data_cols:
        data_sizes.update(zip(data_dims[col], data_cells[col][1], strict=True))
    sizes.pop("row")
    if data_sizes != sizes or 0 in sizes.values():
        # the variables not cached have dimensions of their own (or the cells are
        # empty), which the per-partition reads would see
        return _not_cached(table_path, f"dimensions {sizes} vs {data_sizes}")

    # attributes as load_generic_table() sets them
    var_attrs = {}
    for group in (data_cols, ["TIME", "ANTENNA_ID"]):
        placeholders = {col: xr.DataArray(np.zeros(1)) for col in group}
        add_units_measures(placeholders, table_attrs)
        var_attrs.update({col: placeholders[col].attrs for col in group})

    time = loaded.pop("TIME")
    if not np.isfinite(time).all():
        # the TaQL time range of the per-partition reads would compare with NaN
        return _not_cached(table_path, "TIME values that are not finite")
    # the sort order doubles as the row numbers (in their smallest dtype)
    order = np.argsort(time, kind="stable").astype(row_dtype, copy=False)
    time = time[order]
    tolerance = projection_tolerance(time)
    pointing_columns = PointingColumns(
        time=time,
        tolerance=tolerance,
        antenna_id=loaded.pop("ANTENNA_ID")[order],
        row=order,
        data_columns=data_cols,
        data={col: loaded.pop(col)[order] for col in data_cols} if in_memory else None,
        data_dims=data_dims,
        var_attrs=var_attrs,
        table_attrs=table_attrs,
        bad_cols=list(np.setdiff1d(colnames, list(col_types))),
        table_path=table_path,
    )
    xradio_logger().debug(
        f"Cached {nrows} rows of {table_path} ({', '.join(data_cols)} "
        f"{'in memory' if in_memory else 'read per partition'}): "
        f"{pointing_columns.nbytes / 2**20:.1f} MiB"
    )
    return pointing_columns


def select_pointing_rows(
    pointing_columns: PointingColumns,
    time_min_max: tuple[np.float64, np.float64],
    antenna_ids: np.ndarray,
) -> np.ndarray:
    """
    The rows (indices into the sorted columns) the uncached read of a partition
    selects: TIME in the projected time range and ANTENNA_ID in antenna_ids.

    Parameters
    ----------
    pointing_columns : PointingColumns
        The cached columns.
    time_min_max : tuple[np.float64, np.float64]
        Min/max time of the partition (casacore seconds).
    antenna_ids : np.ndarray
        Antennas of the partition.

    Returns
    -------
    np.ndarray
        Ascending indices into the arrays of ``pointing_columns``.
    """
    range_min, range_max = time_min_max
    lo, hi = project_min_max_sorted(
        range_min, range_max, pointing_columns.time, pointing_columns.tolerance
    )
    # The uncached read compares TIME with the decimal strings of lo/hi in TaQL
    lo, hi = float(f"{lo}"), float(f"{hi}")
    i0 = int(np.searchsorted(pointing_columns.time, lo, side="left"))
    i1 = int(np.searchsorted(pointing_columns.time, hi, side="right"))
    keep = np.isin(pointing_columns.antenna_id[i0:i1], np.asarray(antenna_ids))
    return i0 + np.flatnonzero(keep)


def pivot_time_antenna(
    values: np.ndarray, tcode: np.ndarray, acode: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """
    Scatter rows into a (time, antenna, ...) grid like xarray's unstack does
    after drop_duplicates (redimension_ms_subtable()).

    Parameters
    ----------
    values : np.ndarray
        One row per unique (time, antenna) cell.
    tcode, acode : np.ndarray
        Time and antenna index of every row.
    shape : tuple[int, int]
        Number of times and antennas.

    Returns
    -------
    np.ndarray
        Grid of shape ``shape + values.shape[1:]`` and of the dtype of
        ``values``. Missing cells hold xarray's fill value for the promoted
        dtype, cast back to the dtype of ``values``.
    """
    grid_shape = shape + values.shape[1:]
    if shape[0] * shape[1] > values.shape[0]:
        dtype, fill_value = maybe_promote(values.dtype)
        grid = np.full(grid_shape, fill_value, dtype=dtype)
        grid[tcode, acode] = values
        if dtype != values.dtype:
            with np.errstate(invalid="ignore"):
                grid = grid.astype(values.dtype)
    else:
        grid = np.empty(grid_shape, dtype=values.dtype)
        grid[tcode, acode] = values
    return grid


def pointing_generic_xds(
    pointing_columns: PointingColumns,
    time_min_max: tuple[np.float64, np.float64],
    antenna_ids: np.ndarray,
) -> xr.Dataset:
    """
    The generic pointing dataset of a partition (as load_generic_table() reads
    it with the partition's time range and antennas, for the cached columns).

    Parameters
    ----------
    pointing_columns : PointingColumns
        The cached columns.
    time_min_max : tuple[np.float64, np.float64]
        Min/max time of the partition (casacore seconds).
    antenna_ids : np.ndarray
        Antennas of the partition.

    Returns
    -------
    xr.Dataset
        Data variables on (TIME, ANTENNA_ID, ...), or an empty dataset if no
        row is selected.
    """
    sel = select_pointing_rows(pointing_columns, time_min_max, antenna_ids)
    if sel.size == 0:
        return xr.Dataset()

    time = convert_casacore_time(pointing_columns.time[sel], False)
    utime, tcode = np.unique(time, return_inverse=True)
    uant, acode = np.unique(pointing_columns.antenna_id[sel], return_inverse=True)
    key = tcode.astype(np.int64) * uant.size + acode
    # first row (lowest row number) of every (time, antenna) pair
    order = np.lexsort((pointing_columns.row[sel], key))
    key_sorted = key[order]
    first = order[np.r_[True, key_sorted[1:] != key_sorted[:-1]]]
    tcode_first, acode_first = np.divmod(key[first], uant.size)
    rows_first = sel[first]

    var_attrs = pointing_columns.var_attrs
    data_vars = {}
    for col in pointing_columns.data_columns:
        grid = pivot_time_antenna(
            pointing_columns.read_data(col, rows_first),
            tcode_first,
            acode_first,
            (utime.size, uant.size),
        )
        data_vars[col] = xr.Variable(
            ("TIME", "ANTENNA_ID") + pointing_columns.data_dims[col],
            grid,
            attrs=copy.deepcopy(var_attrs[col]),
        )
    coords = {
        "TIME": xr.Variable("TIME", utime, attrs=copy.deepcopy(var_attrs["TIME"])),
        "ANTENNA_ID": xr.Variable(
            "ANTENNA_ID", uant, attrs=copy.deepcopy(var_attrs["ANTENNA_ID"])
        ),
    }
    attrs = {
        "other": {
            "msv2": {
                "ctds_attrs": copy.deepcopy(pointing_columns.table_attrs),
                "bad_cols": list(pointing_columns.bad_cols),
            }
        }
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def load_cached_pointing_generic_xds(
    in_file: str,
    time_min_max: tuple[np.float64, np.float64] | None,
    antenna_ids: np.ndarray,
    data_columns: tuple[str, ...],
) -> xr.Dataset | None:
    """
    The generic pointing dataset of a partition from the POINTING columns
    cached in the active sub-table cache (read on first use).

    Parameters
    ----------
    in_file : str
        Input MS path.
    time_min_max : tuple[np.float64, np.float64] | None
        Min/max time of the partition (casacore seconds).
    antenna_ids : np.ndarray
        Antennas of the partition.
    data_columns : tuple[str, ...]
        POINTING columns the pointing_xds is built from.

    Returns
    -------
    xr.Dataset | None
        The generic dataset (empty if no POINTING row is selected), or None if
        no cache is active or the table is not cached: the caller then reads
        POINTING itself.
    """
    subtable_cache = active_subtable_cache()
    if subtable_cache is None or time_min_max is None or len(antenna_ids) == 0:
        return None
    table_path = os.path.join(in_file, POINTING_TABLE)
    pointing_columns = subtable_cache.get_or_build(
        ("pointing_columns", table_path, tuple(data_columns)),
        lambda: read_pointing_columns(table_path, tuple(data_columns)),
        amortized=True,
    )
    if pointing_columns is None:
        subtable_cache.count("pointing_uncached")
        return None
    subtable_cache.count("pointing_cached")
    return pointing_generic_xds(pointing_columns, time_min_max, antenna_ids)
