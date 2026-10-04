"""
Shared definitions of the casatools tests in this directory: the test MSs and
cases, the check that xradio reads MSv2 with casatools, and the
codec-independent fingerprints that the tests compare with the reference
values computed with python-casacore (``reference_python_casacore.json``,
written by ``make_reference.py``).

Only the standard library, numpy, xarray and pytest are imported at module
level: every test workflow imports this module (also ``--doctest-modules``),
with python-casacore or with casatools.
"""

import hashlib
import importlib
import importlib.util
import json
import pathlib
from typing import Any

import numpy as np
import pytest

# The test MSs are downloaded here, as in the stakeholder tests, so that the
# casatools workflow (which runs both) downloads every MS once.
MS_DIR = pathlib.Path("/tmp/test")
REFERENCE_FILE = pathlib.Path(__file__).with_name("reference_python_casacore.json")
# Hex digits kept of every sha256 digest (64 bits: enough to tell values apart,
# and keeps the reference file small).
DIGEST_HEX = 16

VLASS = "VLASS3.2.sb45755730.eb46170641.60480.16266136574.split.v6.ms"
ALMA = "ALMA_uid___A002_X1003af4_X75a3.split.avg.ms"
VLBI = "global_vlbi_gg084b_reduced.ms"
SD_STANDARD = "sdimaging.ms"
SD_NO_WEIGHT = "uid___A002_Xe3a5fd_Xe38e.small.ms"
LOFAR = "small_lofar.ms"
NGEHT = "ngEHT_E17A10.0.bin0000.source0000_split.ms"

# What the MSs hold (MAIN table) that the cases cover:
# - VLASS: TiledShapeStMan DATA/FLAG/WEIGHT/SIGMA, 4 SPWs, 20 partitions
#   (an on-the-fly mosaic of many fields), POINTING
# - ALMA: TiledShapeStMan, cells of 3 shapes (1, 4 and 7 channels) in one
#   column, one SPW of decreasing frequencies (the frequency axis is reversed),
#   ephemeris fields, 18 partitions
# - VLBI: WEIGHT_SPECTRUM and SIGMA_SPECTRUM (TiledShapeStMan), gain curve and
#   system calibration sub-tables, 2 partitions ([]) / 4 (["FIELD_ID"])
# - SD_STANDARD: single dish, every MAIN array column in StandardStMan
#   (FLOAT_DATA), POINTING
# - SD_NO_WEIGHT: single dish whose WEIGHT and SIGMA cells are all undefined
#   (no WEIGHT_SPECTRUM: WEIGHT=1), FLOAT_DATA cells of 2 shapes, decreasing
#   frequencies, 20 partitions
# - LOFAR: WEIGHT/SIGMA in IncrementalStMan, DATA/FLAG/WEIGHT_SPECTRUM in
#   TiledColumnStMan, one partition
# - NGEHT: TiledShapeStMan DATA/FLAG cells of 2 shapes
# FLAG_CATEGORY has only undefined cells in all of them but SD_STANDARD.

# Conversions: the MS, partition scheme and the options that change the
# values (none so far: no *_interpolate option, the interpolations do not
# depend on the casacore bindings and numpy.interp may round differently on
# other platforms). Every case is converted with each of its "variants",
# options that change only how the data is read and written (chunks, streamed
# write batches, parallel_mode): all give the values of the reference.
CONVERSION_CASES: dict[str, dict[str, Any]] = {
    "vlass": {"ms": VLASS, "partition_scheme": [], "options": {}},
    "alma": {"ms": ALMA, "partition_scheme": [], "options": {}},
    "vlbi": {"ms": VLBI, "partition_scheme": [], "options": {}},
    "vlbi_field_id": {"ms": VLBI, "partition_scheme": ["FIELD_ID"], "options": {}},
    "sd_standard": {"ms": SD_STANDARD, "partition_scheme": [], "options": {}},
    "sd_no_weight": {"ms": SD_NO_WEIGHT, "partition_scheme": [], "options": {}},
    "lofar": {"ms": LOFAR, "partition_scheme": [], "options": {}},
}
# The options of every conversion (unless a case sets them)
CONVERSION_DEFAULTS = {"with_pointing": True, "persistence_mode": "w"}

# variant name -> options; "stream_batch_bytes" sets
# stream_write.STREAM_BATCH_BYTES (small batches: the streamed write in several
# batches also for these small partitions)
CONVERSION_VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "time_chunks": {"main_chunksize": {"time": 20}},
    "gib_chunks_small_batches": {
        "main_chunksize": 0.0005,
        "stream_batch_bytes": 256 * 1024,
    },
    "lazy_time_reads": {"main_chunksize": {"time": 2}, "parallel_mode": "time"},
}
CASE_VARIANTS: dict[str, list[str]] = {
    "vlass": ["default"],
    "alma": ["default", "time_chunks"],
    "vlbi": ["default", "gib_chunks_small_batches"],
    "vlbi_field_id": ["default"],
    "sd_standard": ["default", "time_chunks"],
    "sd_no_weight": ["default"],
    "lofar": ["default", "lazy_time_reads"],
}

# Partitions, their MAIN rows and the row reads of the MAIN columns: the MS
# and partition scheme, and whether the columns are read (otherwise only the
# partitions and their rows are compared).
READ_CASES: dict[str, dict[str, Any]] = {
    "vlass": {"ms": VLASS, "partition_scheme": [], "read_columns": False},
    "alma": {"ms": ALMA, "partition_scheme": [], "read_columns": True},
    "alma_field_id": {
        "ms": ALMA,
        "partition_scheme": ["FIELD_ID"],
        "read_columns": False,
    },
    "vlbi": {"ms": VLBI, "partition_scheme": [], "read_columns": False},
    "vlbi_field_id": {
        "ms": VLBI,
        "partition_scheme": ["FIELD_ID"],
        "read_columns": True,
    },
    "sd_standard": {
        "ms": SD_STANDARD,
        "partition_scheme": ["ANTENNA1"],
        "read_columns": True,
    },
    "sd_no_weight": {
        "ms": SD_NO_WEIGHT,
        "partition_scheme": [],
        "read_columns": True,
    },
    "lofar": {"ms": LOFAR, "partition_scheme": [], "read_columns": True},
    "ngeht": {"ms": NGEHT, "partition_scheme": [], "read_columns": False},
}
# The self-consistency reads of every column (read_main_columns) use at most
# this many partition rows
SUBSET_ROWS = 2000
# The fields of a column record of read_main_columns
COLUMN_FIELDS = ("check", "window", "values", "grid")


SHIM_MODULE = "xradio._utils._casacore.casacore_from_casatools"


def skip_unless_casatools_backend() -> None:
    """
    Skip the calling test module (call it at module level) where xradio does
    not read MSv2 with casatools, and only there: casatools is not installed,
    or python-casacore is importable (xradio uses it whenever it is). The
    Linux and macOS test workflows (python-casacore) skip these tests; the
    casatools workflow runs them.

    Otherwise xradio's casatools shim (SHIM_MODULE, which configures casaconfig
    and imports casatools) is imported, and an import error is raised, not a
    skip: with casatools installed and no python-casacore, a shim or casatools
    that cannot be imported is an error of the casatools workflow.
    """
    if importlib.util.find_spec("casatools") is None:
        pytest.skip(
            "casatools is not installed: these tests need casatools without "
            "python-casacore (the casatools test workflow)",
            allow_module_level=True,
        )
    try:
        from casacore import tables  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip(
            "python-casacore is installed, so xradio reads MSv2 with "
            "python-casacore: these tests need casatools without "
            "python-casacore (the casatools test workflow)",
            allow_module_level=True,
        )
    importlib.import_module(SHIM_MODULE)


def ms_path(ms_name: str, folder: pathlib.Path = MS_DIR) -> pathlib.Path:
    """Download (once) a test MS into ``folder`` and return its path."""
    from xradio.testing.measurement_set.io import download_measurement_set

    return download_measurement_set(ms_name, folder)


def load_reference() -> dict:
    """The reference values (python-casacore)."""
    with open(REFERENCE_FILE) as ref_file:
        return json.load(ref_file)


# --- digests ------------------------------------------------------------------


def digest(data: bytes | str) -> str:
    """Truncated sha256 hex digest."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:DIGEST_HEX]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys (numpy values as Python values)."""
    return json.dumps(obj, sort_keys=True, default=_json_default)


def without_run_specific(obj: Any, parent: str | None = None) -> Any:
    """
    ``obj`` (attrs) without what differs from run to run or between
    environments: creation dates and the xradio version of the creator.
    """
    if isinstance(obj, dict):
        return {
            key: without_run_specific(value, key)
            for key, value in obj.items()
            if key not in ("creation_date", "date")
            and not (parent == "creator" and key == "version")
        }
    if isinstance(obj, list | tuple):
        return [without_run_specific(value, parent) for value in obj]
    return obj


def attrs_digest(attrs: dict) -> str:
    """Digest of attributes, without dates and versions."""
    return digest(canonical_json(without_run_specific(attrs)))


# Numeric values are hashed as these 64-bit types (exact for every narrower
# type, NaN payloads included), so that the same values stored with another
# width (see WIDENED_DTYPES) have the same digest; the dtypes are compared
# separately.
_HASHED_AS = {"f": np.float64, "c": np.complex128, "i": np.int64, "u": np.uint64}
# dtype of a value read with python-casacore -> the dtype casatools returns for
# it: casatools returns the values of Float, Complex and Int columns as
# float64, complex128 and int64 (getcol, getcell), python-casacore as float32,
# complex64 and int32.
WIDENED_DTYPES = {"float32": "float64", "complex64": "complex128", "int32": "int64"}


def array_digest(values: Any) -> str:
    """
    ``dtype|shape|digest`` of an array's values: the digest of the bytes of
    the values as 64-bit numbers (``_HASHED_AS``, NaN payloads included) or,
    for strings and objects, of their JSON. The dtype of string and object
    arrays is given as "str" (numpy fixed-width or object strings for the same
    values, depending on the versions).
    """
    arr = np.asarray(values)
    shape = "x".join(str(n) for n in arr.shape)
    if arr.dtype.kind in "OUS":
        payload = canonical_json(arr.tolist()).encode()
        dtype = "str"
    else:
        hashed_as = _HASHED_AS.get(arr.dtype.kind, arr.dtype)
        payload = np.ascontiguousarray(arr, dtype=hashed_as).tobytes()
        dtype = arr.dtype.name
    return f"{dtype}|{shape}|{digest(payload)}"


def same_up_to_widening(expected: str, actual: str) -> bool:
    """
    Whether two records of "|"-separated fields (array_digest,
    variable_record) are the same but for dtypes that casatools returns wider
    (WIDENED_DTYPES: the values are the same, see array_digest).
    """
    exp_fields, act_fields = expected.split("|"), actual.split("|")
    return len(exp_fields) == len(act_fields) and all(
        exp == act or WIDENED_DTYPES.get(exp) == act
        for exp, act in zip(exp_fields, act_fields, strict=True)
    )


# --- processing sets --------------------------------------------------------------

# Values as stored (no CF decoding but the coordinates)
OPEN_STORED_VALUES = {
    "engine": "zarr",
    "mask_and_scale": False,
    "decode_times": False,
    "decode_timedelta": False,
}


# Variables whose values come from libm or from interpolation, whose last bits
# may differ between platforms (the reference is computed on Linux x86-64, the
# casatools workflow also runs on macOS arm64, whose libm and compilers, e.g.
# fused multiply-adds, may round differently): their values are kept in the
# fingerprint ("approx") and compared with a tolerance of APPROX_RTOL relative
# to the largest magnitude of the variable (so that components that are 0 in
# the reference compare as well); their variable_record covers everything else
# (dtype, dims, shape, attributes) bit for bit, as all the other variables.
# See approx_variables:
# - OBSERVER_POSITION (LIBM_VARIABLES): astropy's EarthLocation (ERFA
#   trigonometry);
# - the EPHEMERIS_VARIABLES of an ephemeris field_and_source_xds that are on
#   the main time axis ("time"): interpolated from the ephemeris table to the
#   MSv4 times (create_field_and_source_xds: interpolate_to_time, xarray's
#   interp). FIELD_*_CENTER_DIRECTION and FIELD_*_CENTER_DISTANCE always are;
#   the others only with ephemeris_interpolate=True (otherwise they are on
#   "time_ephemeris", the values of the ephemeris table, bit for bit). The
#   other variables on "time" (LINE_*, from the SOURCE table) are not
#   interpolated.
LIBM_VARIABLES = ("OBSERVER_POSITION",)
EPHEMERIS_TYPE = "field_and_source_ephemeris"
# The variables that extract_ephemeris_info computes from the ephemeris table
EPHEMERIS_VARIABLES = (
    "FIELD_PHASE_CENTER_DIRECTION",
    "FIELD_PHASE_CENTER_DISTANCE",
    "FIELD_REFERENCE_CENTER_DIRECTION",
    "FIELD_REFERENCE_CENTER_DISTANCE",
    "HELIOCENTRIC_RADIAL_VELOCITY",
    "NORTH_POLE_ANGULAR_DISTANCE",
    "NORTH_POLE_POSITION_ANGLE",
    "OBSERVER_PHASE_ANGLE",
    "SOURCE_DIRECTION",
    "SOURCE_DISTANCE",
    "SOURCE_RADIAL_VELOCITY",
    "SUB_OBSERVER_DIRECTION",
    "SUB_SOLAR_DIRECTION",
    "SUB_SOLAR_DISTANCE",
)
APPROX_RTOL = 1e-12


def approx_variables(ds: Any) -> list[str]:
    """The variables of a dataset whose values are compared with a tolerance
    (LIBM_VARIABLES, and the EPHEMERIS_VARIABLES interpolated to the MSv4
    times)."""
    names = {str(name) for name in LIBM_VARIABLES if name in ds.variables}
    if ds.attrs.get("type") == EPHEMERIS_TYPE:
        names |= {
            name
            for name in EPHEMERIS_VARIABLES
            if name in ds.variables and "time" in ds.variables[name].dims
        }
    return sorted(names)


def variable_record(var: Any, approx: bool = False) -> str:
    """
    ``dtype|digest`` of a variable: the dtype, and the digest of its dims,
    shape, values (array_digest: numbers as 64-bit values, so the digest does
    not depend on the width of the dtype; not the values if ``approx``) and
    attributes.
    """
    dtype, shape, values = array_digest(var.values).split("|")
    payload = [
        [str(dim) for dim in var.dims],
        shape if approx else f"{shape}|{values}",
        attrs_digest(var.attrs),
    ]
    return f"{dtype}|{digest(canonical_json(payload))}"


def node_fingerprint(ds: Any) -> dict:
    """
    Fingerprint of one node (dataset) of a processing set: "vars", the
    variable_record of every variable (coordinates included), "meta", the
    digest of the coordinate names and the dataset attributes, and "approx",
    the float64 values of the approx_variables (if any).
    """
    approx_names = approx_variables(ds)
    meta = [sorted(str(name) for name in ds.coords), without_run_specific(ds.attrs)]
    fingerprint = {
        "meta": digest(canonical_json(meta)),
        "vars": {
            str(name): variable_record(var, str(name) in approx_names)
            for name, var in ds.variables.items()
        },
    }
    if approx_names:
        fingerprint["approx"] = {
            name: np.asarray(ds.variables[name].values, dtype=np.float64).tolist()
            for name in approx_names
        }
    return fingerprint


def approx_equal(expected: Any, actual: Any, rtol: float = APPROX_RTOL) -> bool:
    """
    Whether two arrays of float values are equal up to ``rtol`` times the
    largest finite magnitude of ``expected`` (NaNs at the same places).
    """
    exp_arr = np.asarray(expected, dtype=np.float64)
    act_arr = np.asarray(actual, dtype=np.float64)
    if exp_arr.shape != act_arr.shape:
        return False
    finite = np.abs(exp_arr[np.isfinite(exp_arr)])
    scale = float(finite.max()) if finite.size else 0.0
    return bool(
        np.allclose(act_arr, exp_arr, rtol=rtol, atol=rtol * scale, equal_nan=True)
    )


def _approx_differences(where: str, expected: dict, actual: dict) -> list[str]:
    diffs = []
    for name, exp_values in sorted(expected.items()):
        act_values = actual.get(name)
        if act_values is None:
            diffs.append(f"{where}/{name}: values expected, none kept")
        elif not approx_equal(exp_values, act_values):
            exp_arr = np.asarray(exp_values, dtype=np.float64)
            act_arr = np.asarray(act_values, dtype=np.float64)
            if exp_arr.shape != act_arr.shape:
                detail = f"shape {act_arr.shape}, expected {exp_arr.shape}"
            else:
                detail = (
                    "largest difference "
                    f"{np.nanmax(np.abs(act_arr - exp_arr), initial=0.0)!r}, "
                    f"largest value {np.nanmax(np.abs(exp_arr), initial=0.0)!r}, "
                    f"NaNs {int(np.isnan(act_arr).sum())} "
                    f"(expected {int(np.isnan(exp_arr).sum())})"
                )
            diffs.append(f"{where}/{name}: values differ: {detail}")
    for name in sorted(set(actual) - set(expected)):
        diffs.append(f"{where}/{name}: values kept, none expected")
    return diffs


def processing_set_fingerprint(ps_path: str | pathlib.Path) -> dict:
    """
    Fingerprint of a processing set, independent of chunks and codecs:

    - "msv4": MSv4 name -> {node path in the MSv4 ("." for the MSv4 itself)
      -> node id}
    - "nodes": node id -> node_fingerprint (one entry for the nodes that are
      the same in several MSv4s, e.g. antenna_xds)
    - "attrs": digest of the attributes of the processing set
    """
    import xarray as xr

    tree = xr.open_datatree(str(ps_path), **OPEN_STORED_VALUES)
    msv4, nodes = {}, {}
    for name, msv4_node in sorted(tree.children.items()):
        paths = {}
        for node in msv4_node.subtree:
            fingerprint = node_fingerprint(node.to_dataset(inherit=False))
            node_id = digest(canonical_json(fingerprint))
            nodes[node_id] = fingerprint
            paths[node.relative_to(msv4_node)] = node_id
        msv4[name] = paths
    attrs = attrs_digest(tree.attrs)
    tree.close()
    return {"msv4": msv4, "nodes": nodes, "attrs": attrs}


def fingerprint_differences(
    expected: dict, actual: dict, widened: list[str] | None = None
) -> list[str]:
    """
    What differs between two processing_set_fingerprint results (one line
    per difference, for assertion messages). If ``widened`` is a list, the
    variables that differ only by a dtype that casatools returns wider
    (``same_up_to_widening``: same values) are appended to it instead.
    """
    diffs = []
    if expected["attrs"] != actual["attrs"]:
        diffs.append("processing set attrs")
    if sorted(expected["msv4"]) != sorted(actual["msv4"]):
        diffs.append(
            f"MSv4s: expected {sorted(expected['msv4'])}, got {sorted(actual['msv4'])}"
        )
    for name in sorted(set(expected["msv4"]) & set(actual["msv4"])):
        exp_paths, act_paths = expected["msv4"][name], actual["msv4"][name]
        if sorted(exp_paths) != sorted(act_paths):
            diffs.append(
                f"{name}: nodes {sorted(exp_paths)} expected, got {sorted(act_paths)}"
            )
        for path in sorted(set(exp_paths) & set(act_paths)):
            if exp_paths[path] == act_paths[path]:
                continue
            exp = expected["nodes"][exp_paths[path]]
            act = actual["nodes"][act_paths[path]]
            where = f"{name}/{path}"
            if exp["meta"] != act["meta"]:
                diffs.append(f"{where}: coordinate names or attrs")
            for var in sorted(set(exp["vars"]) | set(act["vars"])):
                exp_var = exp["vars"].get(var)
                act_var = act["vars"].get(var)
                if exp_var == act_var:
                    continue
                if (
                    widened is not None
                    and exp_var is not None
                    and act_var is not None
                    and same_up_to_widening(exp_var, act_var)
                ):
                    exp_dtype, act_dtype = exp_var.split("|")[0], act_var.split("|")[0]
                    widened.append(f"{where}/{var}: {exp_dtype} -> {act_dtype}")
                else:
                    diffs.append(f"{where}/{var}: expected {exp_var}, got {act_var}")
            diffs += _approx_differences(
                where, exp.get("approx", {}), act.get("approx", {})
            )
    return diffs


# --- partitions and MAIN row reads ------------------------------------------------


def _stable_message(exc: Exception) -> str:
    """The part of a ColumnNotReadableError message that does not quote the
    casacore bindings' own error text."""
    return str(exc).split(" in the partition")[0]


def _cell_shape(table: Any, col: str, row: int) -> tuple[int, ...]:
    from xradio.measurement_set._utils._msv2._tables.read_rows import (
        parse_shape_string,
    )

    if table.isscalarcol(col):
        return ()
    return parse_shape_string(table.getcolshapestring(col, row, 1)[0])


def _read_column(table, col, rows, cell_shape, dtype, **kwargs) -> np.ndarray:
    from xradio.measurement_set._utils._msv2._tables.read_rows import read_rows

    out = np.empty((rows.size,) + tuple(cell_shape), dtype=dtype)
    read_rows(table, col, rows, out, **kwargs)
    return out


def _check_row_subsets(table, col, rows, values, cell_shape, dtype) -> None:
    """
    Reads of a subset of the partition rows (every k-th row: one call per
    row) with calls of at most 3 cells, and of channel / polarization ranges
    of 2-D cells, give the values of the whole read.
    """
    step = max(1, rows.size // SUBSET_ROWS)
    sub = rows[::step][:SUBSET_ROWS]
    expected = values[::step][:SUBSET_ROWS]
    cell_elems = int(np.prod(cell_shape, dtype=np.int64)) or 1
    got = _read_column(table, col, sub, cell_shape, dtype, max_elems=3 * cell_elems)
    np.testing.assert_array_equal(got, expected, err_msg=col)
    if len(cell_shape) == 2:
        n_chan, n_pol = cell_shape
        chan = slice(n_chan // 3, max(n_chan // 3 + 1, n_chan - 1))
        pol = slice(n_pol - 1, n_pol)
        for c, p in ((chan, None), (None, pol), (chan, pol)):
            shape = (
                len(range(n_chan)[c]) if c else n_chan,
                len(range(n_pol)[p]) if p else n_pol,
            )
            got = _read_column(table, col, sub, shape, dtype, chan=c, pol=p)
            np.testing.assert_array_equal(
                got, expected[:, c or slice(None), p or slice(None)], err_msg=col
            )


def _check_time_chunks(table, col, rows, tidxs, bidxs, grid) -> None:
    """The grid read in chunks of times (read_time_chunk, the read of the lazy
    and streamed paths) is the grid read whole."""
    from xradio.measurement_set._utils._msv2._tables.read_rows import (
        TimeChunkRows,
        read_time_chunk,
    )

    n_times, n_baselines = grid.shape[:2]
    size = max(1, n_times // 3)
    chunks = tuple(min(size, n_times - t) for t in range(0, n_times, size))
    chunk_rows = TimeChunkRows(rows, tidxs, bidxs, chunks, n_baselines)
    parts = [
        read_time_chunk(table, col, chunk_rows, k, grid.shape[2:], grid.dtype)
        for k in range(len(chunks))
    ]
    np.testing.assert_array_equal(np.concatenate(parts), grid, err_msg=col)


# Scalar MAIN columns read (besides every array column): those the converter
# reads for the data variables and the time / baseline indices
READ_SCALAR_COLUMNS = (
    "ANTENNA1",
    "ANTENNA2",
    "EXPOSURE",
    "FLAG_ROW",
    "INTERVAL",
    "TIME",
    "TIME_CENTROID",
)


def _try(func, *args, **kwargs) -> tuple[Any, str | None]:
    """(result, None), or (None, "raises <type>") if func raises (not an
    AssertionError)."""
    try:
        return func(*args, **kwargs), None
    except AssertionError:
        raise
    except Exception as exc:
        return None, f"raises {type(exc).__name__}"


def read_main_columns(table: Any, rows: np.ndarray) -> dict[str, list[str]]:
    """
    For every MAIN array column and READ_SCALAR_COLUMNS, on the rows of one
    partition, a record ``[check, window, values, grid]`` (COLUMN_FIELDS):

    - check: check_partition_cells (is every cell defined, of one shape?);
    - window: column_row_window (rows per tile / storage unit);
    - values: array_digest of read_rows of all the partition rows (into a
      buffer of the column dtype);
    - grid: array_digest of the (time, baseline, ...) grid of the converter
      (read_col_conversion_numpy, of the dtype of the first cell).

    A step that raises gives "raises <exception type>" ("-" if not done).
    Reads of row
    subsets, channel / polarization ranges and time chunks are checked
    against the whole reads (``_check_row_subsets``, ``_check_time_chunks``).
    """
    from xradio.measurement_set._utils._msv2._tables.read import (
        read_col_conversion_numpy,
    )
    from xradio.measurement_set._utils._msv2._tables.read_rows import (
        ColumnNotReadableError,
        MainTableRows,
        check_partition_cells,
        column_dtype,
        column_row_window,
    )
    from xradio.measurement_set._utils._msv2.conversion import (
        calc_indx_for_row_split,
    )

    main_rows = MainTableRows(table, rows)
    tidxs, bidxs, ant1, _, utimes = calc_indx_for_row_split(main_rows)
    time_baseline_shape = (len(utimes), len(ant1))
    columns = [
        col
        for col in sorted(table.colnames())
        if col in READ_SCALAR_COLUMNS or not table.isscalarcol(col)
    ]
    result = {}
    for col in columns:
        storage = main_rows.column_storage(col)
        try:
            check = check_partition_cells(table, col, rows, storage)
            verified = "verified" if check.verified else "unverified"
            record = [f"{verified}: {check.how}"]
        except ColumnNotReadableError as exc:
            record = [f"not readable: {_stable_message(exc)}"]
        result[col] = record
        try:
            cell_shape = _cell_shape(table, col, int(rows[0]))
            dtype = column_dtype(table, col)
        except Exception as exc:
            record += ["-", f"raises {type(exc).__name__}", "-"]
            continue
        cell_bytes = int(np.prod(cell_shape, dtype=np.int64)) * dtype.itemsize
        window_rows, window = column_row_window(
            storage, table.nrows(), cell_shape, cell_bytes
        )
        record.append(f"{window_rows} rows/{window}")
        values, error = _try(_read_column, table, col, rows, cell_shape, dtype)
        record.append(error or array_digest(values))
        if values is not None:
            _check_row_subsets(table, col, rows, values, cell_shape, dtype)
        del values
        grid, error = _try(
            read_col_conversion_numpy, main_rows, col, time_baseline_shape, tidxs, bidxs
        )
        record.append(error or array_digest(grid))
        if grid is not None:
            _check_time_chunks(table, col, rows, tidxs, bidxs, grid)
        del grid
    main_rows.close()
    return result


def main_table_fingerprint(
    in_file: str | pathlib.Path, partition_scheme: list, read_columns: bool
) -> dict:
    """
    Partitions and MAIN rows of an MS, and the row reads of its MAIN columns:

    - "partitions": digest of the partition descriptions
      (create_partitions_with_main_rows, the same as create_partitions),
      "n_partitions";
    - "rows": ``n|digest`` of the MAIN rows of every partition. The rows of
      the row runs, of select_main_rows (numpy selection from the key columns)
      and of partition_main_rows with and without the runs are checked to be
      the same;
    - "columns" (if ``read_columns``): partition index -> read_main_columns,
      for the first partition of every DATA_DESC_ID (cell shapes and
      frequencies change with the DDI).
    """
    from xradio._utils._casacore.tables import open_table_ro
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions,
        create_partitions_with_main_rows,
        partition_main_rows,
        select_main_rows,
    )

    partitions, runs = create_partitions_with_main_rows(str(in_file), partition_scheme)
    assert partitions == create_partitions(str(in_file), partition_scheme)
    result = {
        "partitions": digest(canonical_json(partitions)),
        "n_partitions": len(partitions),
        "rows": [],
    }
    read_idxs = {}
    for idx, part in enumerate(partitions):
        read_idxs.setdefault(tuple(part["DATA_DESC_ID"]), idx)
    if read_columns:
        result["columns"] = {}
    with open_table_ro(str(in_file)) as table:
        assert runs.main_nrows == table.nrows()
        for idx, part in enumerate(partitions):
            rows = runs[idx].rows()
            for other in (
                select_main_rows(table, part),
                partition_main_rows(table, part),
                partition_main_rows(table, part, runs[idx]),
            ):
                np.testing.assert_array_equal(other, rows)
            result["rows"].append(f"{rows.size}|{digest(rows.astype('<i8').tobytes())}")
            if read_columns and idx in read_idxs.values():
                result["columns"][str(idx)] = read_main_columns(table, rows)
    return result


# --- conversions ------------------------------------------------------------------


class ConversionSpy:
    """
    Records, while active (``with``), what a conversion did: the
    ``unreadable_columns`` of every attempt of a partition conversion (a
    second attempt follows a column that only the read found unreadable), the
    statistics of every streamed write (write_deferred_variables) and every
    partition read whole (read_deferred_variables), and the number of
    sub-table caches created. Patches module attributes (no pytest needed).
    """

    def __init__(self) -> None:
        self.attempts: list[list[str]] = []
        self.streamed: list[dict] = []
        self.read_whole: list[dict] = []
        self.subtable_caches = 0
        self._saved: list[tuple[Any, str, Any]] = []

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __enter__(self) -> "ConversionSpy":
        from xradio.measurement_set._utils._msv2 import conversion
        from xradio.measurement_set._utils._msv2._tables import subtable_cache

        convert_partition = conversion._convert_and_write_partition
        write_streamed = conversion.write_deferred_variables
        read_whole = conversion.read_deferred_variables
        cache_init = subtable_cache.SubtableCache.__init__

        def attempt(*args, unreadable_columns=frozenset(), **kwargs):
            self.attempts.append(sorted(unreadable_columns))
            return convert_partition(
                *args, unreadable_columns=unreadable_columns, **kwargs
            )

        def streamed(*args, **kwargs):
            stats = write_streamed(*args, **kwargs)
            self.streamed.append(stats)
            return stats

        def whole(*args, **kwargs):
            stats = read_whole(*args, **kwargs)
            self.read_whole.append(stats)
            return stats

        def cache(cache_self, *args, **kwargs):
            self.subtable_caches += 1
            cache_init(cache_self, *args, **kwargs)

        self._patch(conversion, "_convert_and_write_partition", attempt)
        self._patch(conversion, "write_deferred_variables", streamed)
        self._patch(conversion, "read_deferred_variables", whole)
        self._patch(subtable_cache.SubtableCache, "__init__", cache)
        return self

    def __exit__(self, *exc_info) -> None:
        while self._saved:
            owner, name, value = self._saved.pop()
            setattr(owner, name, value)


def convert(
    case: str, variant: str, out_file: str | pathlib.Path, folder=MS_DIR
) -> pathlib.Path:
    """
    Convert the MS of a conversion case with the options of a variant
    (stream_write.STREAM_BATCH_BYTES set for the conversion if the variant
    sets "stream_batch_bytes"). Returns the processing set path.
    """
    from xradio.measurement_set import convert_msv2_to_processing_set
    from xradio.measurement_set._utils._msv2 import stream_write

    spec = CONVERSION_CASES[case]
    options = dict(CONVERSION_DEFAULTS) | spec["options"]
    options |= CONVERSION_VARIANTS[variant]
    batch_bytes = options.pop("stream_batch_bytes", None)
    saved = stream_write.STREAM_BATCH_BYTES
    if batch_bytes is not None:
        stream_write.STREAM_BATCH_BYTES = batch_bytes
    try:
        convert_msv2_to_processing_set(
            in_file=str(ms_path(spec["ms"], folder)),
            out_file=str(out_file),
            partition_scheme=spec["partition_scheme"],
            **options,
        )
    finally:
        stream_write.STREAM_BATCH_BYTES = saved
    return pathlib.Path(out_file)
