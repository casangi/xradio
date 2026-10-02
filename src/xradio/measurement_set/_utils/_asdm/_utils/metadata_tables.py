"""Conversion of ASDM (pyasdm) metadata tables to Python values and pandas DataFrames.

Converting a pyasdm table calls one Python getter per row and column, which is slow
for big tables. The ASDM backend asks for the same whole tables (Main, Field,
Source, ConfigDescription, ...) once for every partition, so
:func:`exp_asdm_table_to_df` keeps a cache of the converted columns of every
table, per ASDM object. The cache:

- is keyed by the ASDM object itself, through weak references, so it never keeps
  an ASDM alive, and the entries of an ASDM go away when that ASDM is garbage
  collected. Different ASDM objects (including copies of one another) never share
  entries;
- is checked against the table object and its rows (number of rows, identity of
  the first and last rows) on every call, so replacing a table or adding rows
  invalidates it. Changing attributes of rows already in a table, after the table
  has been converted, is not detected: call :func:`clear_asdm_table_cache` after
  such changes;
- hands out copies: every call builds a new DataFrame, with copies of the mutable
  values (lists, numpy arrays, pyasdm quantities such as ArrayTime) it holds.
"""

import copy
import threading
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pyasdm

# TODO: also convert 1-D lists of Angle, Length, PolarizationType and 2-D lists of
#  Length, etc. Callers currently convert those values themselves (.get(), str()).

# Scalar enumerations returned as their name (str)
_SCALAR_ENUMERATIONS_AS_NAME = (
    pyasdm.enumerations.BasebandName,
    pyasdm.enumerations.ProcessorType,
    pyasdm.enumerations.ProcessorSubType,
    pyasdm.enumerations.SpectralResolutionType,
)

# Enumerations converted to a list of names when found in a 1-D list
_LIST_ENUMERATIONS_AS_NAMES = (
    pyasdm.enumerations.StokesParameter,
    pyasdm.enumerations.ScanIntent,
)


def _upper_first(col_string: str) -> str:
    return col_string[0].upper() + col_string[1:]


def _convert_asdm_value(value: Any) -> Any:
    """Convert one value returned by a pyasdm row getter (see :func:`load_asdm_col`).

    Parameters
    ----------
    value : Any
        Value as returned by a ``get<Column>()`` method of a pyasdm row.

    Returns
    -------
    Any
        The converted value, or ``value`` itself when there is no conversion for its
        type.
    """
    if isinstance(value, pyasdm.types.Tag):
        # As an int (value.getTag() would give the string form)
        return value.getTagValue()
    if isinstance(value, pyasdm.types.EntityRef):
        return value.getEntityId()
    if type(value) is pyasdm.types.Frequency:
        return value.get()
    if type(value) in _SCALAR_ENUMERATIONS_AS_NAME:
        # this could/would also include FrequencyReferenceCode, etc. enumerations
        return value.getName()

    if not isinstance(value, list) or len(value) == 0:
        return value

    first = value[0]
    if isinstance(first, pyasdm.types.Tag):
        return np.array([item_val.getTagValue() for item_val in value])
    if type(first) in _LIST_ENUMERATIONS_AS_NAMES:
        return [item_val.getName() for item_val in value]
    if type(first) is pyasdm.types.Speed:
        # example: Source/sysVel
        return [item_val.get() for item_val in value]
    if isinstance(first, list) and len(first) > 0:
        if isinstance(first[0], pyasdm.enumerations.PolarizationType):
            # 2-D list of PolarizationType, for example Polarization/corrProduct
            # ([numCorr][2]): keep the nesting
            return [[item_val.getName() for item_val in inner] for inner in value]
        if isinstance(first[0], pyasdm.types.Angle):
            # 2-D list of Angle, as seen in Field (directions) and Pointing
            return [[item_val.get() for item_val in inner] for inner in value]

    return value


def load_asdm_col(sdm_table: Any, col_name: str, *, allow_absent: bool = False) -> list:
    """Load a column from an ASDM table into a list.

    This function extracts values from a specified column in an ASDM (ALMA Science
    Data Model) table, handling various ASDM-specific data types and converting
    them to Python native types.

    Parameters
    ----------
    sdm_table : pyasdm table
        A table of an ASDM loaded using pyasdm (for example ``asdm.getMain()``, a
        ``pyasdm.MainTable``).
    col_name : str
        Name of the column (attribute) to extract from the table, as in the ASDM
        (for example "scanNumber").
    allow_absent : bool, default False
        How to handle rows where an optional attribute is absent
        (``is<Column>Exists()`` is False). If False, the pyasdm getter raises
        ValueError for those rows. If True, the value for those rows is None.
        Mandatory attributes are always read.

    Returns
    -------
    list
        One value per row of the table, converted to appropriate Python types.

    Raises
    ------
    AttributeError
        If the table has no such column.
    ValueError
        If an optional attribute is absent in some row and ``allow_absent`` is
        False (raised by pyasdm).

    Notes
    -----
    Handles special ASDM types including:

    - Tags: converted to integer values
    - EntityRefs: converted to entity IDs (str)
    - Arrays of Tags: converted to numpy arrays of integers
    - 1-D arrays of StokesParameter and ScanIntent enumerations: lists of names
    - Scalar BasebandName, ProcessorType, ProcessorSubType, SpectralResolutionType
      enumerations: names (str)
    - 2-D arrays of PolarizationType: nested lists of names (same shape, for
      example [numCorr][2] for Polarization/corrProduct)
    - 2-D arrays of Angle: nested lists of floats (radians)
    - Frequency objects: float values (Hz)
    - 1-D arrays of Speed: lists of floats (m/s)

    Other values are returned as given by pyasdm.

    Examples
    --------
    >>> import pyasdm
    >>> asdm = pyasdm.ASDM()
    >>> asdm.setFromFile("uid___X02_X1")
    >>> scan_numbers = load_asdm_col(asdm.getMain(), "scanNumber")
    >>> print(scan_numbers)
    [1, 2, 3, 4]
    """
    rows = sdm_table.get()
    attr_name = _upper_first(col_name)
    get_col_function_name = f"get{attr_name}"
    is_exists_function_name = f"is{attr_name}Exists"
    col_values = []
    for row in rows:
        if allow_absent:
            # Only optional attributes have an is<Column>Exists() method
            is_exists_function = getattr(row, is_exists_function_name, None)
            if is_exists_function is not None and not is_exists_function():
                col_values.append(None)
                continue
        get_col_function = getattr(row, get_col_function_name)
        col_values.append(_convert_asdm_value(get_col_function()))

    return col_values


# Values that can be shared between the cache and the DataFrames handed out
_IMMUTABLE_TYPES = (type(None), bool, int, float, complex, str, bytes, np.generic)

# pyasdm quantity types that make a copy when constructed from an instance (this is
# how the pyasdm getters copy them). Much faster than copy.deepcopy.
_COPY_CONSTRUCTIBLE_TYPES = frozenset(
    {
        pyasdm.types.Angle,
        pyasdm.types.AngularRate,
        pyasdm.types.ArrayTime,
        pyasdm.types.ArrayTimeInterval,
        pyasdm.types.Flux,
        pyasdm.types.Frequency,
        pyasdm.types.Humidity,
        pyasdm.types.Interval,
        pyasdm.types.Length,
        pyasdm.types.Pressure,
        pyasdm.types.Speed,
        pyasdm.types.Temperature,
    }
)


def _is_pyasdm_enumeration(value: Any) -> bool:
    # pyasdm enumeration values have no setters
    return type(value).__module__.startswith("pyasdm.enumerations.")


def _is_immutable(value: Any) -> bool:
    return isinstance(value, _IMMUTABLE_TYPES) or _is_pyasdm_enumeration(value)


def _copy_value(value: Any) -> Any:
    """Copy a (converted) table value so that changing the copy cannot change the cache."""
    if isinstance(value, _IMMUTABLE_TYPES):
        return value
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    if type(value) in _COPY_CONSTRUCTIBLE_TYPES:
        return type(value)(value)
    if _is_pyasdm_enumeration(value):
        return value
    return copy.deepcopy(value)


def _make_column_copier(values: list) -> Callable[[list], list]:
    """Choose, once per cached column, how to copy its values."""
    if all(_is_immutable(value) for value in values):
        return list

    value_types = {type(value) for value in values}
    if len(value_types) == 1:
        (value_type,) = value_types
        if value_type is np.ndarray:
            return lambda column_values: [value.copy() for value in column_values]
        if value_type in _COPY_CONSTRUCTIBLE_TYPES:
            return lambda column_values: [value_type(value) for value in column_values]

    return lambda column_values: [_copy_value(value) for value in column_values]


class _CachedColumn:
    """Converted values of one table column."""

    __slots__ = ("_values", "_copier")

    def __init__(self, values: list):
        self._values = values
        self._copier = _make_column_copier(values)

    def copy_values(self) -> list:
        return self._copier(self._values)


class _CachedTable:
    """Converted columns of one table, valid while the table and its rows are the same."""

    __slots__ = (
        "_table_ref",
        "_num_rows",
        "_first_row_ref",
        "_last_row_ref",
        "columns",
    )

    def __init__(self, table: Any, rows: list):
        # Weak references only: table and rows reference the ASDM (container)
        self._table_ref = weakref.ref(table)
        self._num_rows = len(rows)
        self._first_row_ref = weakref.ref(rows[0]) if rows else None
        self._last_row_ref = weakref.ref(rows[-1]) if rows else None
        self.columns: dict[tuple[str, bool], _CachedColumn] = {}

    def is_valid_for(self, table: Any, rows: list) -> bool:
        if self._table_ref() is not table or len(rows) != self._num_rows:
            return False
        if not rows:
            return True
        return self._first_row_ref() is rows[0] and self._last_row_ref() is rows[-1]


# ASDM object -> {table name: _CachedTable}
_asdm_table_cache: "weakref.WeakKeyDictionary[Any, dict[str, _CachedTable]]" = (
    weakref.WeakKeyDictionary()
)
_asdm_table_cache_lock = threading.Lock()


def _get_cached_table(
    sdm: Any, table_name: str, table: Any, rows: Any
) -> _CachedTable | None:
    """Find (or create) the valid cache entry for a table, None if it cannot be cached."""
    if not isinstance(rows, list):
        return None

    with _asdm_table_cache_lock:
        try:
            tables = _asdm_table_cache.get(sdm)
            if tables is None:
                tables = {}
                _asdm_table_cache[sdm] = tables
        except TypeError:
            # sdm is not hashable or cannot be weakly referenced
            return None

        cached_table = tables.get(table_name)
        if cached_table is None or not cached_table.is_valid_for(table, rows):
            try:
                cached_table = _CachedTable(table, rows)
            except TypeError:
                # table or rows cannot be weakly referenced
                return None
            tables[table_name] = cached_table

        return cached_table


def clear_asdm_table_cache(sdm: Any = None) -> None:
    """Drop the cached table conversions of :func:`exp_asdm_table_to_df`.

    Only needed after changing attributes of rows of a table that has already been
    converted (adding rows or replacing tables is detected).

    Parameters
    ----------
    sdm : pyasdm.ASDM, optional
        ASDM whose cached tables are dropped. If None (default), the cache is
        emptied for all ASDMs.
    """
    with _asdm_table_cache_lock:
        if sdm is None:
            _asdm_table_cache.clear()
        else:
            try:
                _asdm_table_cache.pop(sdm, None)
            except TypeError:
                pass


def exp_asdm_table_to_df(
    sdm: pyasdm.ASDM,
    table_name: str,
    col_names: list[str],
    *,
    allow_absent: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Convert an ASDM table to a pandas DataFrame.

    This function extracts specified columns from an ASDM table and converts them into
    a pandas DataFrame format (see :func:`load_asdm_col` for the conversion of
    values).

    The converted columns are cached per ASDM object, table and column (see the
    module docstring), so converting the same table again is cheap. Every call
    returns a new DataFrame that the caller can modify freely.

    Parameters
    ----------
    sdm : pyasdm.ASDM
        The ASDM object containing the table to be converted
    table_name : str
        Name of the table to extract from the ASDM (without 'Table' suffix)
    col_names : list[str]
        List of column names to extract from the table
    allow_absent : bool, default False
        If True, rows where an optional attribute is absent give None for that
        column (which pandas turns into NaN in numeric columns) instead of raising
        ValueError.
    use_cache : bool, default True
        If False, convert the table without using or filling the cache (for
        example for a very large table that is read only once).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the specified columns from the ASDM table, one row per
        table row, in table order.

    Raises
    ------
    AttributeError
        If the ASDM has no such table, or the table has no such column.
    ValueError
        If an optional attribute is absent in some row and ``allow_absent`` is
        False (raised by pyasdm).

    Examples
    --------
    >>> df = exp_asdm_table_to_df(sdm, "ExecBlock", ["startTime", "endTime"])
    """

    get_table_name = f"get{table_name}"
    get_table_function = getattr(sdm, get_table_name)
    table = get_table_function()

    cached_table = (
        _get_cached_table(sdm, table_name, table, table.get()) if use_cache else None
    )

    col_values = {}
    for col in col_names:
        if cached_table is None:
            col_values[col] = load_asdm_col(table, col, allow_absent=allow_absent)
            continue

        key = (col, allow_absent)
        cached_column = cached_table.columns.get(key)
        if cached_column is None:
            cached_column = _CachedColumn(
                load_asdm_col(table, col, allow_absent=allow_absent)
            )
            cached_table.columns[key] = cached_column
        col_values[col] = cached_column.copy_values()

    df = pd.DataFrame(col_values)
    return df
