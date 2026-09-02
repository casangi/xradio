"""
python-casacore ``tables``-compatible module implemented on top of arcae.

This module emulates the subset of the python-casacore ``casacore.tables``
API that XRADIO uses, backed by `arcae <https://github.com/ska-sa/arcae>`_
(Arrow-based casacore table bindings with self-contained wheels for Linux
and macOS).

arcae exposes a deliberately small surface (indexed ``getcol``/``putcol``,
``tabledesc``/``getcoldesc``/``getdminfo``, ``addrows``/``addcols`` and TaQL
via ``Table.from_taql``).  Everything else python-casacore offers is emulated
here, using two building blocks:

* **Table keywords** are read from ``tabledesc()["_keywords_"]`` (arcae
  serializes the full table descriptor, keywords included, through
  casacore's own ``JsonOut`` at full float precision).
* **TaQL statements** cover the operations arcae has no API for: reading and
  writing slices of tiled columns (arcae's ``getcol`` cannot read
  ``TiledCellStMan`` columns, but TaQL slice expressions can), creating
  tables (``CREATE TABLE``), copying tables (``GIVING ... AS PLAIN``) and
  dropping columns.  Statements always operate on the already-open handle
  (``$1`` plus arcae's ``tables=[...]`` argument) so no second lock is taken.
* **Keyword writes** go through casacore's JSON parser: the keyword value is
  attached as a column keyword of a temporary column (``addcols`` accepts
  keywords in the column descriptor), promoted to a table keyword with
  ``ALTER TABLE ... COPY KEYWORD`` and the temporary column dropped.  This
  supports arbitrary field names (e.g. the ``*0`` fields of per-plane beam
  records) which TaQL record literals cannot express.

Caveat: casacore errors raised inside ``Table.from_taql`` abort the process
(arcae does not convert them to Python exceptions), so callers of ``_taql``
validate their input first.
"""

import os
import re
import sys

import arcae
import numpy as np
from arcae.lib import arrow_tables as _at

# ---------------------------------------------------------------------------
# type maps
# ---------------------------------------------------------------------------

# casacore column valueType -> numpy dtype
_VALUETYPE_TO_DTYPE = {
    "boolean": np.dtype(bool),
    "bool": np.dtype(bool),
    "uchar": np.dtype(np.uint8),
    "short": np.dtype(np.int16),
    "ushort": np.dtype(np.uint16),
    "int": np.dtype(np.int32),
    "integer": np.dtype(np.int32),
    "uint": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
    "float": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "complex": np.dtype(np.complex64),
    "dcomplex": np.dtype(np.complex128),
}

# numpy dtype kind/itemsize -> casacore column valueType (for column creation)
_DTYPE_TO_VALUETYPE = {
    np.dtype(bool): "boolean",
    np.dtype(np.uint8): "uchar",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "ushort",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.int64): "int64",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.complex64): "complex",
    np.dtype(np.complex128): "dcomplex",
}


def _json_safe(value):
    """Convert a python/numpy value into a JSON-serializable equivalent that
    casacore's JsonParser converts back to a sensible Record field."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quote_path(path):
    """Quote a filesystem path for use as a TaQL table name."""
    if '"' in path or "\n" in path:
        raise ValueError(f"Unsupported character in table path: {path!r}")
    return f'"{path}"'


def _resolve_subtable_path(tablename):
    """Resolve python-casacore's ``maintable::SUBTABLE`` path syntax."""
    if "::" in tablename:
        main, _, sub = tablename.partition("::")
        return os.path.join(main, sub)
    return tablename


def _normalize_lockoptions(lockoptions):
    if lockoptions in (None, "default"):
        return "auto"
    return lockoptions


def _slice_spec(blc, trc, ndim):
    """Build a TaQL (1-based, inclusive, Fortran-order) slice expression from
    C-order blc/trc (both inclusive, python-casacore convention)."""
    blc = [int(b) for b in np.atleast_1d(blc)]
    trc = [int(t) for t in np.atleast_1d(trc)]
    if len(blc) != ndim or len(trc) != ndim:
        raise ValueError(f"blc/trc must have {ndim} entries, got blc={blc}, trc={trc}")
    # reverse C-order -> Fortran order and convert to 1-based inclusive
    parts = [f"{b + 1}:{t + 1}" for b, t in zip(blc[::-1], trc[::-1], strict=True)]
    return "[" + ",".join(parts) + "]"


# ---------------------------------------------------------------------------
# the table class
# ---------------------------------------------------------------------------

_KEYWORD_HELPER_COLUMN = "_XRADIO_KEYWORD_HELPER_"


class table:
    """python-casacore-compatible ``table`` on top of an arcae Table."""

    def __init__(
        self,
        tablename=None,
        tabledesc=None,
        nrow=0,
        readonly=True,
        lockoptions="default",
        ack=False,
        dminfo=None,
        _arcae_table=None,
        _name=None,
    ):
        self._closed = False
        if _arcae_table is not None:
            self._t = _arcae_table
            self._name = _name or ""
            return
        tablename = _resolve_subtable_path(os.path.expanduser(str(tablename)))
        self._name = tablename
        if tabledesc is not None:
            self._t = _create_table(tablename, tabledesc, nrow, dminfo)
        else:
            try:
                self._t = _at.Table.from_filename(
                    tablename,
                    readonly=readonly,
                    lockoptions=_normalize_lockoptions(lockoptions),
                )
            except Exception as exc:
                # python-casacore raises RuntimeError for unopenable tables
                raise RuntimeError(str(exc)) from exc

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._t.close()

    # python-casacore alias
    done = close

    def flush(self, recursive=False):
        pass

    def unlock(self):
        pass

    def lock(self, write=True, nattempts=0):
        pass

    def name(self):
        try:
            return self._t.name()
        except Exception:
            return self._name

    # -- taql helpers -------------------------------------------------------

    def _taql(self, command):
        """Run a TaQL command with ``$1`` bound to this table."""
        return _at.Table.from_taql(command, tables=[self._t])

    def taql(self, command, locals=None, globals=None):
        return taql(command, _depth=2)

    # -- structure ----------------------------------------------------------

    def nrows(self):
        return self._t.nrow()

    def ncols(self):
        return self._t.ncolumns()

    def colnames(self):
        return self._t.columns()

    def getcoldesc(self, columnname):
        return self._t.getcoldesc(columnname)

    def getdesc(self, actual=True):
        return self._t.tabledesc()

    def getdminfo(self, columnname=None):
        dminfo = self._t.getdminfo()
        if columnname is None:
            return dminfo
        for group in dminfo.values():
            if columnname in group.get("COLUMNS", []):
                return group
        raise RuntimeError(f"Column {columnname} not found in dminfo")

    def coldatatype(self, columnname):
        return self.getcoldesc(columnname)["valueType"]

    def _col_dtype(self, columnname):
        return _VALUETYPE_TO_DTYPE.get(self.coldatatype(columnname))

    def isscalarcol(self, columnname):
        desc = self.getcoldesc(columnname)
        return "ndim" not in desc and "shape" not in desc

    def isvarcol(self, columnname):
        desc = self.getcoldesc(columnname)
        return "ndim" in desc and "shape" not in desc

    def getcolshapestring(self, columnname, startrow=0, nrow=-1, rowincr=1):
        if self.isscalarcol(columnname):
            raise RuntimeError(
                f"Column {columnname} is a scalar column and has no shape"
            )
        nrows = self.nrows()
        if nrow < 0:
            nrow = nrows - startrow
        shapes = self._t.row_shapes(
            columnname, index=(slice(startrow, startrow + nrow),)
        ).to_pylist()
        # arcae row_shapes are already C-order, like python-casacore's strings
        return [
            "[" + ", ".join(str(s) for s in shape) + "]" if shape else "[]"
            for shape in shapes
        ]

    def iscelldefined(self, columnname, rownr):
        desc = self.getcoldesc(columnname)
        if "ndim" not in desc and "shape" not in desc:
            return True  # scalar cells are always defined
        try:
            shape = self._t.row_shapes(columnname, index=([rownr],)).to_pylist()[0]
        except Exception:
            return True
        return shape is not None

    # -- keywords -----------------------------------------------------------

    def _keywords(self):
        return self._t.tabledesc().get("_keywords_", {})

    def getkeywords(self):
        return {k: _from_json_keyword(v) for k, v in self._keywords().items()}

    def keywordnames(self):
        return list(self._keywords().keys())

    fieldnames = keywordnames

    def getkeyword(self, keyword):
        keywords = self._keywords()
        if keyword not in keywords:
            raise RuntimeError(f"Table keyword {keyword} does not exist")
        return _from_json_keyword(keywords[keyword])

    def getcolkeywords(self, columnname):
        keywords = self.getcoldesc(columnname).get("keywords", {})
        return {k: _from_json_keyword(v) for k, v in keywords.items()}

    def putkeyword(self, keyword, value, makesubrecord=False):
        if isinstance(value, str) and value.startswith("Table: "):
            # A genuine table-reference keyword may already have been created
            # (e.g. by arcae's MS factory when the subtable was made). Do not
            # degrade it to a plain string.
            if keyword in self._keywords():
                return
        self._put_keywords({keyword: value})

    def putkeywords(self, keywords):
        self._put_keywords(dict(keywords))

    def putcolkeyword(self, columnname, keyword, value):
        self._put_keywords({keyword: value}, column=columnname)

    def _put_keywords(self, keywords, column=None):
        """Write keywords with ``ALTER TABLE ... SET KEYWORD`` literals.

        TaQL record literals support typed scalars/arrays (including 2-d and
        empty arrays) and arbitrarily nested records, but record field names
        must be valid identifiers. Sub-records with other field names (e.g.
        the ``*0`` fields of per-plane beam records) are written as empty
        placeholders first and then filled in through casacore's JSON parser
        (:meth:`_put_keyword_json`).
        """
        if not keywords:
            return
        target_prefix = f"{column}::" if column else ""
        deferred = []
        assignments = []
        for name, value in keywords.items():
            literal = _taql_keyword_literal(value, deferred, (name,))
            assignments.append(f"{target_prefix}{name} = {literal}")
        self._taql("ALTER TABLE $1 SET KEYWORD " + ", ".join(assignments)).close()
        for path, value in deferred:
            self._put_keyword_json(target_prefix + ".".join(path), value)

    def _put_keyword_json(self, keyword_path, value):
        """Write one keyword through casacore's JSON parser.

        The value is attached as a column keyword of a temporary column
        (arcae ``addcols`` accepts keywords in the column descriptor, parsed
        by casacore's JsonParser, which supports arbitrary record field
        names but neither multi-dimensional nor empty arrays), then promoted
        with ``ALTER TABLE ... COPY KEYWORD`` and the temporary column is
        dropped.
        """
        self._t.addcols(
            {
                _KEYWORD_HELPER_COLUMN: {
                    "valueType": "boolean",
                    "option": 0,
                    "maxlen": 0,
                    "comment": "",
                    "keywords": {"kw": _json_safe(value)},
                }
            },
            None,
        )
        try:
            self._taql(
                f"ALTER TABLE $1 COPY KEYWORD "
                f"{keyword_path} = {_KEYWORD_HELPER_COLUMN}::kw"
            ).close()
        finally:
            self._taql(f"ALTER TABLE $1 DROP COLUMN {_KEYWORD_HELPER_COLUMN}").close()

    def removekeyword(self, keyword):
        self._taql(f"ALTER TABLE $1 DELETE KEYWORD {keyword}").close()

    def removecolkeyword(self, columnname, keyword):
        self._taql(f"ALTER TABLE $1 DELETE KEYWORD {columnname}::{keyword}").close()

    # -- table info ---------------------------------------------------------

    def _info_path(self):
        return os.path.join(self.name(), "table.info")

    def info(self):
        # table.info layout: "Type = X\nSubType = Y\n\n<readme lines>"
        info = {"type": "", "subType": "", "readme": ""}
        try:
            with open(self._info_path()) as f:
                lines = f.read().split("\n")
        except OSError:
            return info
        if lines and lines[0].startswith("Type = "):
            info["type"] = lines[0][len("Type = ") :]
            lines = lines[1:]
        if lines and lines[0].startswith("SubType = "):
            info["subType"] = lines[0][len("SubType = ") :]
            lines = lines[1:]
        if lines and lines[0] == "":
            lines = lines[1:]
        readme = "\n".join(lines).rstrip("\n")
        info["readme"] = readme + "\n" if readme else ""
        return info

    def putinfo(self, info):
        current = self.info()
        current.update(info)
        readme = current.get("readme", "")
        if readme and not readme.endswith("\n"):
            readme += "\n"
        with open(self._info_path(), "w") as f:
            f.write(f"Type = {current.get('type', '')}\n")
            f.write(f"SubType = {current.get('subType', '')}\n")
            f.write("\n")
            f.write(readme)

    # -- reading ------------------------------------------------------------

    def _row_index(self, startrow, nrow):
        if startrow == 0 and (nrow is None or nrow < 0):
            return None
        stop = self.nrows() if (nrow is None or nrow < 0) else startrow + nrow
        return (slice(int(startrow), int(stop)),)

    def _convert_read(self, data, columnname):
        dtype = self._col_dtype(columnname)
        if dtype is not None and data.dtype != dtype:
            if dtype == np.dtype(bool):
                data = data.astype(bool)
            elif np.can_cast(data.dtype, dtype, casting="same_kind"):
                data = data.astype(dtype)
        elif data.dtype == object:
            # string columns come back as object arrays of str
            data = data.astype(str)
        return data

    def getcol(self, columnname, startrow=0, nrow=-1, rowincr=1):
        if rowincr != 1:
            raise NotImplementedError("rowincr != 1 is not supported")
        data = self._t.getcol(columnname, index=self._row_index(startrow, nrow))
        return self._convert_read(data, columnname)

    def getcolnp(self, columnname, nparray, startrow=0, nrow=-1, rowincr=1):
        data = self.getcol(columnname, startrow, nrow, rowincr)
        nparray[...] = data
        return nparray

    def getcell(self, columnname, rownr):
        if self.coldatatype(columnname) == "record":
            raise RuntimeError(f"Cannot read record-valued column {columnname}")
        data = self._t.getcol(columnname, index=([int(rownr)],))
        data = self._convert_read(data, columnname)
        cell = data[0]
        if np.ndim(cell) == 0:
            return cell.item() if isinstance(cell, np.generic) else cell
        return cell

    def getcellslice(self, columnname, rownr, blc, trc, inc=[]):  # noqa: B006
        if inc not in ([], None) and any(int(i) != 1 for i in np.atleast_1d(inc)):
            raise NotImplementedError("inc != 1 is not supported")
        desc = self.getcoldesc(columnname)
        ndim = int(desc.get("ndim", len(np.atleast_1d(blc))))
        spec = _slice_spec(blc, trc, ndim)
        rownr = int(rownr)
        query = f"SELECT {columnname}{spec} AS DATA FROM $1 LIMIT {rownr}:{rownr + 1}"
        result = self._taql(query)
        try:
            data = result.getcol("DATA")
        finally:
            result.close()
        # TaQL expressions promote to double/dcomplex; restore column dtype
        dtype = self._col_dtype(columnname)
        if dtype is not None and data.dtype != dtype:
            data = data.astype(dtype)
        return data[0]

    def getcolslice(self, columnname, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):  # noqa: B006
        if rowincr != 1:
            raise NotImplementedError("rowincr != 1 is not supported")
        if inc not in ([], None) and any(int(i) != 1 for i in np.atleast_1d(inc)):
            raise NotImplementedError("inc != 1 is not supported")
        blc = np.atleast_1d(blc)
        trc = np.atleast_1d(trc)
        # blc/trc are C-order inclusive; build per-dimension slices after the
        # row dimension (arcae indexes are C-order too)
        dims = tuple(slice(int(b), int(t) + 1) for b, t in zip(blc, trc, strict=True))
        row_index = self._row_index(startrow, nrow)
        if row_index is None:
            row_index = (slice(0, self.nrows()),)
        data = self._t.getcol(columnname, index=row_index + dims)
        return self._convert_read(data, columnname)

    def col(self, columnname):
        return tablecolumn(self, columnname)

    def row(self, columnnames=[], exclude=False):  # noqa: B006
        return tablerow(self, columnnames, exclude)

    def rownumbers(self, table=None):
        result = self._taql("SELECT rowid() AS XRADIO_ROWID FROM $1")
        try:
            return result.getcol("XRADIO_ROWID")
        finally:
            result.close()

    def iter(self, columnnames, order="", sort=True):
        if isinstance(columnnames, list | tuple):
            if len(columnnames) != 1:
                raise NotImplementedError("iter over multiple columns")
            columnname = columnnames[0]
        else:
            columnname = columnnames
        values = self.getcol(columnname)
        if sort:
            base = f"SELECT FROM $1 ORDERBY {columnname}"
            values = np.sort(values, kind="stable")
        else:
            base = "SELECT FROM $1"
        if len(values) == 0:
            return
        boundaries = np.nonzero(values[1:] != values[:-1])[0] + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(values)]))
        for start, stop in zip(starts, stops, strict=True):
            sub = self._taql(f"{base} LIMIT {start}:{stop}")
            yield table(_arcae_table=sub, _name=self._name)

    def query(
        self, query="", name="", sortlist="", columns="", limit=0, offset=0, style=None
    ):
        command = "SELECT"
        if columns:
            command += f" {columns}"
        command += " FROM $1"
        if query:
            command += f" WHERE {query}"
        if sortlist:
            command += f" ORDERBY {sortlist}"
        if limit > 0:
            command += f" LIMIT {limit}"
        if offset > 0:
            command += f" OFFSET {offset}"
        return table(_arcae_table=self._taql(command), _name=self._name)

    # -- writing ------------------------------------------------------------

    def addrows(self, nrows=1):
        self._t.addrows(int(nrows))

    def addcols(self, desc, dminfo={}, addtoparent=True):  # noqa: B006
        coldescs, deferred_keywords = _prepare_coldescs_for_json(
            _json_safe(_flatten_coldescs(desc))
        )
        self._t.addcols(coldescs, _json_safe(dminfo) or None)
        _write_deferred_empty_keywords(self, deferred_keywords)

    def removecols(self, columnnames):
        if isinstance(columnnames, str):
            columnnames = [columnnames]
        cols = ", ".join(columnnames)
        self._taql(f"ALTER TABLE $1 DROP COLUMN {cols}").close()

    def _convert_write(self, columnname, value, expect_cell=False):
        """Coerce a python/numpy value to the column dtype, adding a leading
        row axis when ``expect_cell``."""
        dtype = self._col_dtype(columnname)
        data = np.asarray(value)
        if data.dtype.kind in ("U", "S", "O"):
            data = data.astype(object) if data.dtype.kind != "O" else data
        elif dtype is not None and data.dtype != dtype:
            data = data.astype(dtype)
        if expect_cell:
            data = data[np.newaxis, ...]
        return data

    def _is_tiled(self, columnname):
        return (
            self.getcoldesc(columnname).get("dataManagerType", "").startswith("Tiled")
        )

    def putcol(self, columnname, value, startrow=0, nrow=-1, rowincr=1):
        if rowincr != 1:
            raise NotImplementedError("rowincr != 1 is not supported")
        data = np.asarray(value)
        desc = self.getcoldesc(columnname)
        cell_ndim = int(desc.get("ndim", 0))
        if data.ndim == cell_ndim:
            # a single cell (or scalar) value: broadcast over the target rows
            nrows = self.nrows() - startrow if nrow < 0 else nrow
            data = np.broadcast_to(data, (nrows,) + data.shape)
        data = self._convert_write(columnname, data)
        if data.shape[0] == 0:
            return  # nothing to write (and arrow cannot type empty arrays)
        if self._is_tiled(columnname):
            # arcae cannot write to tiled columns; go through TaQL
            self._put_rows_via_taql(columnname, np.ascontiguousarray(data), startrow)
            return
        index = self._row_index(startrow, nrow)
        self._t.putcol(columnname, data, index=index)

    def putcell(self, columnname, rownr, value):
        data = self._convert_write(columnname, value, expect_cell=True)
        if self._is_tiled(columnname):
            self._put_rows_via_taql(columnname, np.ascontiguousarray(data), int(rownr))
            return
        self._t.putcol(columnname, data, index=([int(rownr)],))

    def putcellslice(self, columnname, rownr, value, blc, trc, inc=[]):  # noqa: B006
        if inc not in ([], None) and any(int(i) != 1 for i in np.atleast_1d(inc)):
            raise NotImplementedError("inc != 1 is not supported")
        desc = self.getcoldesc(columnname)
        ndim = int(desc.get("ndim", len(np.atleast_1d(blc))))
        value = np.asarray(value)
        dtype = self._col_dtype(columnname)
        if dtype is not None and value.dtype != dtype:
            value = value.astype(dtype)
        spec = _slice_spec(blc, trc, ndim)
        rownr = int(rownr)
        self._update_from_helper(
            columnname, value[np.newaxis, ...], rownr, slice_spec=spec
        )

    def _put_rows_via_taql(self, columnname, data, startrow):
        """Write whole cells of ``data`` (rows first) starting at ``startrow``
        with a TaQL update — used for tiled columns, which arcae cannot
        write to directly."""
        self._update_from_helper(columnname, data, startrow)

    def _update_from_helper(
        self,
        columnname: str,
        data: np.ndarray,
        startrow: int | float,
        slice_spec: str = "",
    ) -> None:
        """UPDATE rows [startrow, startrow+len(data)) of ``columnname`` (with
        an optional cell slice) from a temporary helper table holding
        ``data``."""
        import shutil
        import tempfile

        data = np.asarray(data)
        nrows = data.shape[0]
        valuetype = _DTYPE_TO_VALUETYPE.get(data.dtype, "double")
        if data.dtype.kind in ("U", "S", "O"):
            raise NotImplementedError("TaQL-based writes of string cells")
        cell_shape_f = list(data.shape[1:][::-1])
        helper_dir = tempfile.mkdtemp(prefix="xradio_arcae_put_")
        helper_path = os.path.join(helper_dir, "helper.tbl")
        try:
            desc = {
                "XRADIO_PUT_VALUE": {
                    "valueType": valuetype,
                    "option": 0,
                    "maxlen": 0,
                    "comment": "",
                }
            }
            if data.ndim > 1:
                desc["XRADIO_PUT_VALUE"]["ndim"] = int(data.ndim - 1)
                desc["XRADIO_PUT_VALUE"]["shape"] = cell_shape_f
            helper = _create_table(helper_path, desc, nrow=nrows, dminfo=None)
            helper.putcol("XRADIO_PUT_VALUE", data)
            helper.close()
            startrow = int(startrow)
            # TaQL comma-join requires equal row counts; create a RefTable as a
            # write-through view of target rows to match the helper size.
            ref_path: str = os.path.join(helper_dir, "ref.tbl")
            self._taql(
                f"SELECT FROM $1 "
                f"LIMIT {startrow}:{startrow + nrows} "
                f"GIVING {_quote_path(ref_path)}"
            ).close()
            self._taql(
                f"UPDATE {_quote_path(ref_path)}, "
                f"{_quote_path(helper_path)} t2 "
                f"SET {columnname}{slice_spec} = t2.XRADIO_PUT_VALUE"
            ).close()
        finally:
            shutil.rmtree(helper_dir, ignore_errors=True)

    def copy(
        self,
        newtablename,
        deep=False,
        valuecopy=False,
        dminfo={},  # noqa: B006
        endian="aipsrc",
    ):
        newtablename = os.path.expanduser(newtablename)
        self._taql(
            f"SELECT FROM $1 GIVING {_quote_path(newtablename)} AS PLAIN"
        ).close()
        return table(newtablename, readonly=False, ack=False)


class tablecolumn:
    """Minimal python-casacore ``tablecolumn`` emulation."""

    def __init__(self, tab, columnname):
        self._table = tab
        self._column = columnname

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._table.getcell(self._column, key)
        raise NotImplementedError("tablecolumn only supports integer indexing")


class tablerow:
    """Minimal python-casacore ``tablerow`` emulation."""

    def __init__(self, tab, columnnames=[], exclude=False):  # noqa: B006
        self._table = tab
        if isinstance(columnnames, str):
            columnnames = [columnnames]
        allcols = tab.colnames()
        if columnnames:
            if exclude:
                cols = [c for c in allcols if c not in columnnames]
            else:
                cols = [c for c in allcols if c in columnnames]
        else:
            cols = list(allcols)
        # record-valued columns cannot be read
        self._columns = [c for c in cols if tab.coldatatype(c) != "record"]

    def __len__(self):
        return self._table.nrows()

    def _get_rows(self, start, stop):
        nrows = self._table.nrows()
        start, stop, _ = slice(start, stop).indices(nrows)
        rows = [{} for _ in range(max(stop - start, 0))]
        if not rows:
            return rows
        for col in self._columns:
            values = self._read_column(col, start, stop)
            for row, value in zip(rows, values, strict=True):
                row[col] = value
        return rows

    @staticmethod
    def _undefined_cell(dtype, desc=None):
        # python-casacore returns an empty (size-0) array for undefined
        # variable-shaped array cells in a table row, and a shaped
        # (zero-filled) array for undefined fixed-shape cells
        shape = (desc or {}).get("shape")
        if shape is not None and len(shape) > 0:
            shape = [int(s) for s in shape]
            if dtype is None:  # string column
                return np.full(shape, "", dtype=object)
            return np.zeros(shape, dtype=dtype)
        if dtype is None:
            # undefined variable string cells surface as python-casacore's
            # dict representation of empty (non-1-d) string arrays
            return {
                "shape": np.array([0], dtype=np.int32),
                "array": np.array([], dtype=str),
            }
        return np.array([], dtype=dtype)

    def _read_column(self, col, start, stop):
        desc = self._table.getcoldesc(col)
        dtype = _VALUETYPE_TO_DTYPE.get(desc.get("valueType"))
        try:
            arrow_table = self._table._t.to_arrow(
                index=(slice(start, stop),), columns=[col]
            )
        except Exception:
            # e.g. "Shape derivation of null is not supported" when only
            # some cells are defined: fall back to per-cell reads
            return self._read_column_per_cell(col, start, stop, dtype, desc)
        if col not in arrow_table.column_names:
            # arcae drops columns it cannot read (e.g. all cells undefined)
            return [self._undefined_cell(dtype, desc) for _ in range(stop - start)]
        arrow = arrow_table.column(col)
        is_scalar = "ndim" not in desc and "shape" not in desc
        is_complex = dtype is not None and dtype.kind == "c"
        values = []
        for item in arrow.to_pylist():
            if item is None:
                value = self._undefined_cell(dtype, desc)
            elif is_scalar:
                if is_complex:
                    value = complex(item[0], item[1])
                else:
                    value = bool(item) if dtype == np.dtype(bool) else item
            else:
                value = np.asarray(item)
                if is_complex:
                    # arrow encodes complex values as a trailing [real, imag]
                    # dimension
                    value = (value[..., 0] + 1j * value[..., 1]).astype(dtype)
                elif dtype is not None and value.dtype != dtype and value.size:
                    value = value.astype(dtype)
            values.append(value)
        return values

    def _read_column_per_cell(self, col, start, stop, dtype, desc=None):
        values = []
        for rownr in range(start, stop):
            if not self._table.iscelldefined(col, rownr):
                values.append(self._undefined_cell(dtype, desc))
            else:
                values.append(self._table.getcell(col, rownr))
        return values

    def __getitem__(self, key):
        if isinstance(key, int):
            nrows = self._table.nrows()
            if key < 0:
                key += nrows
            return self._get_rows(key, key + 1)[0]
        if isinstance(key, slice):
            return self._get_rows(key.start, key.stop)
        raise TypeError(f"Invalid tablerow index {key!r}")


# ---------------------------------------------------------------------------
# module-level functions
# ---------------------------------------------------------------------------

_DOLLAR_NAME = re.compile(r"\$([A-Za-z_][A-Za-z_0-9]*)")

# TaQL constructs that casacore rejects with a parse error. They must be
# caught in Python because arcae's from_taql aborts the process on casacore
# exceptions instead of raising.
_EMPTY_IN_LIST = re.compile(r"\bIN\s*\[\s*\]", re.IGNORECASE)


def _precheck_taql(command):
    if _EMPTY_IN_LIST.search(command):
        raise RuntimeError(f"Error in TaQL command: '{command}'\n  empty IN list")


def taql(command, style="Python", tables=[], globals={}, locals={}, _depth=1):  # noqa: B006
    """Execute a TaQL command.

    ``$name`` variables referring to open :class:`table` objects in the
    caller's namespace are substituted (like python-casacore does). arcae
    supports at most one table argument.
    """

    def _lookup(varname):
        if varname in locals:
            return locals[varname]
        if varname in globals:
            return globals[varname]
        # like python-casacore's getvariable(): walk the whole call stack
        frame = sys._getframe(_depth)
        while frame is not None:
            if varname in frame.f_locals:
                return frame.f_locals[varname]
            frame = frame.f_back
        return None

    bound = {}

    def _substitute(match):
        varname = match.group(1)
        value = _lookup(varname)
        if isinstance(value, table):
            if varname not in bound:
                if bound:
                    raise NotImplementedError(
                        "TaQL with more than one table variable is not "
                        "supported by the arcae backend"
                    )
                bound[varname] = value
            return "$1"
        # like python-casacore's substitute(): expand plain python values
        if isinstance(value, str):
            return _taql_string(value)
        if isinstance(value, bool | np.bool_ | int | float | complex | np.generic):
            return _taql_scalar_body(value)
        if isinstance(value, list | tuple | np.ndarray):
            arr = np.asarray(value)
            if arr.dtype != object:
                return _taql_array_body(arr)
        return match.group(0)

    command = _DOLLAR_NAME.sub(_substitute, command)
    _precheck_taql(command)
    arcae_tables = [t._t for t in bound.values()]
    result = _at.Table.from_taql(command, tables=arcae_tables or None)
    return table(_arcae_table=result)


def tableexists(tablename):
    tablename = _resolve_subtable_path(os.path.expanduser(str(tablename)))
    return os.path.isdir(tablename) and os.path.exists(
        os.path.join(tablename, "table.dat")
    )


def tableiswritable(tablename):
    return tableexists(tablename) and os.access(
        os.path.join(tablename, "table.dat"), os.W_OK
    )


# ---------------------------------------------------------------------------
# keyword serialization (python value -> TaQL literal)
# ---------------------------------------------------------------------------

_VALID_TAQL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# words the TaQL lexer treats specially even in record-field position
_TAQL_RESERVED = {
    "and",
    "or",
    "xor",
    "not",
    "in",
    "like",
    "between",
    "exists",
    "t",
    "f",
    "all",
    "distinct",
    "select",
    "from",
    "where",
    "orderby",
    "limit",
    "offset",
    "giving",
    "update",
    "set",
    "insert",
    "delete",
    "count",
    "having",
    "groupby",
    "join",
    "on",
    "using",
    "as",
}


def _is_taql_fieldname(name):
    return bool(_VALID_TAQL_NAME.match(name)) and name.lower() not in _TAQL_RESERVED


def _taql_string(value):
    """Quote a string as a TaQL literal (concatenation of quoted segments —
    TaQL has no escape sequences inside strings)."""
    # NUL bytes (casacore fixed-length string padding) would terminate the
    # C string inside the TaQL parser
    value = value.replace("\x00", "")
    if value == "":
        return "''"
    segments = []
    for part in re.split(r"(')", value):
        if part == "'":
            segments.append('"\'"')
        elif part:
            segments.append(f"'{part}'")
    return "".join(segments)


_NDARRAY_TYPE_CODE = {
    "b": "B",
    "i": "I4",
    "u": "I4",
    "f": None,  # doubles are TaQL's native float type
    "c": "C8",
    "U": "S",
    "S": "S",
}


def _taql_array_body(arr):
    if arr.ndim == 0:
        return _taql_scalar_body(arr.item())
    return "[" + ",".join(_taql_array_body(sub) for sub in arr) + "]"


def _taql_scalar_body(value):
    if isinstance(value, bool | np.bool_):
        return "T" if value else "F"
    if isinstance(value, complex | np.complexfloating):
        value = complex(value)
        return f"{value.real!r}{value.imag:+}i"
    if isinstance(value, float | np.floating):
        text = repr(float(value))
        return text
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, str):
        return _taql_string(value)
    raise TypeError(f"Cannot express {type(value)} as a TaQL literal")


def _taql_keyword_literal(value, deferred, path):
    """Serialize a keyword value into a TaQL literal.

    Dict subtrees with field names TaQL cannot express are emitted as an
    empty record placeholder and appended to ``deferred`` as
    ``(path, value)`` for a follow-up JSON-based write.
    """
    if isinstance(value, dict):
        if not value:
            return "[=]"
        if not all(_is_taql_fieldname(str(k)) for k in value):
            deferred.append((path, value))
            return "[=]"
        fields = ", ".join(
            f"{k}={_taql_keyword_literal(v, deferred, path + (str(k),))}"
            for k, v in value.items()
        )
        return f"[{fields}]"
    if isinstance(value, str):
        if "\n" in value or "\r" in value:
            deferred.append((path, value))
            return "''"
        return _taql_string(value)
    if isinstance(value, list | tuple | np.ndarray):
        arr = np.asarray(value)
        if arr.dtype == object or arr.dtype.kind == "V":
            raise TypeError(f"Cannot express keyword value {value!r} in TaQL")
        code = _NDARRAY_TYPE_CODE.get(arr.dtype.kind)
        if arr.dtype == np.dtype(np.float32):
            code = "R4"
        if arr.size == 0:
            return f"[] AS {code or 'R8'}"
        if arr.dtype.kind in ("U", "S") and any(
            "\n" in s or "\r" in s for s in arr.ravel().astype(str)
        ):
            deferred.append((path, value))
            return "[=]"
        body = _taql_array_body(arr)
        return f"{body} AS {code}" if code else body
    if isinstance(value, bool | np.bool_):
        return "T" if value else "F"
    if isinstance(value, int | np.integer):
        return f"{int(value)} AS I4"
    return _taql_scalar_body(value)


# ---------------------------------------------------------------------------
# keyword JSON -> python conversion
# ---------------------------------------------------------------------------


def _from_json_keyword(value):
    """Convert a JSON-decoded keyword value into python-casacore-like types:
    numeric lists become numpy arrays, records stay dicts, strings stay
    strings (including ``Table: ...`` subtable references). NUL padding that
    casacore fixed-length strings can carry is stripped."""
    if isinstance(value, dict):
        return {k: _from_json_keyword(v) for k, v in value.items()}
    if isinstance(value, list):
        arr = np.asarray(value)
        if arr.dtype.kind in ("i", "u", "f", "c", "b") and arr.dtype != object:
            return arr
        return [_from_json_keyword(v) for v in value]
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


# ---------------------------------------------------------------------------
# table creation
# ---------------------------------------------------------------------------


def _flatten_coldescs(tabledesc):
    """Strip non-column entries from a table descriptor."""
    return {
        name: desc
        for name, desc in tabledesc.items()
        if not name.startswith("_") and isinstance(desc, dict)
    }


def _strip_empty_arrays(value, deferred, path):
    """Remove empty arrays from a JSON-bound structure.

    casacore's JsonParser (used by arcae for table descriptors) cannot parse
    empty JSON arrays. Removed keyword entries are recorded in ``deferred``
    as ``(path, value)`` so they can be written back with TaQL afterwards.
    """
    if isinstance(value, dict):
        out = {}
        for key, sub in value.items():
            if isinstance(sub, list | tuple | np.ndarray) and len(sub) == 0:
                deferred.append((path + (str(key),), sub))
            else:
                out[key] = _strip_empty_arrays(sub, deferred, path + (str(key),))
        return out
    return value


def _prepare_coldescs_for_json(coldescs):
    """Make column descriptors safe for casacore's JSON parser.

    Returns the cleaned descriptors plus a list of ``(column, keyword_path,
    value)`` entries for empty-array column keywords that must be written
    back with TaQL after the columns exist.
    """
    cleaned = {}
    deferred_keywords = []
    for name, desc in coldescs.items():
        desc = dict(desc)
        if (
            isinstance(desc.get("shape"), list | tuple | np.ndarray)
            and len(desc["shape"]) == 0
        ):
            del desc["shape"]
        deferred = []
        desc["keywords"] = _strip_empty_arrays(desc.get("keywords", {}), deferred, ())
        cleaned[name] = desc
        for kw_path, value in deferred:
            deferred_keywords.append((name, ".".join(kw_path), value))
    return cleaned, deferred_keywords


def _write_deferred_empty_keywords(tab, deferred_keywords):
    """Write empty-array keywords stripped by :func:`_prepare_coldescs_for_json`.

    The element type is not recoverable from an empty JSON array; string is
    the common case for such keywords in MS descriptors (e.g. ``CATEGORY``).
    """
    for column, kw_path, value in deferred_keywords:
        dtype = getattr(value, "dtype", None)
        code = "S"
        if dtype is not None and dtype.kind in ("i", "u"):
            code = "I4"
        elif dtype is not None and dtype.kind == "f":
            code = "R8"
        tab._taql(
            f"ALTER TABLE $1 SET KEYWORD {column}::{kw_path} = [] AS {code}"
        ).close()


def _create_table(tablename, tabledesc, nrow, dminfo):
    """Create a new table from a python-casacore style table descriptor.

    The (empty) table is created with TaQL; columns — including their
    keywords — and data managers are then added through arcae ``addcols``,
    and table keywords are written afterwards.
    """
    if os.path.exists(tablename):
        raise RuntimeError(f"Table {tablename} already exists")
    coldescs = _flatten_coldescs(tabledesc)
    created = _at.Table.from_taql(
        f"CREATE TABLE {_quote_path(tablename)} () LIMIT {int(nrow)}"
    )
    wrapper = table(_arcae_table=created, _name=tablename)
    if coldescs:
        wrapper.addcols(coldescs, dminfo or {})
    keywords = tabledesc.get("_keywords_")
    if keywords:
        wrapper.putkeywords(keywords)
    return created


# ---------------------------------------------------------------------------
# MS creation utilities (python-casacore msutil/tableutil equivalents)
# ---------------------------------------------------------------------------


def required_ms_desc(tabletype="MAIN"):
    """MS descriptor with required columns (python-casacore compatible)."""
    return _at.ms_descriptor((tabletype or "MAIN").upper(), complete=False)


def complete_ms_desc(tabletype="MAIN"):
    """MS descriptor with the complete set of columns."""
    return _at.ms_descriptor((tabletype or "MAIN").upper(), complete=True)


def makescacoldesc(
    columnname,
    value,
    datamanagertype="",
    datamanagergroup="",
    options=0,
    maxlen=0,
    comment="",
    valuetype="",
    keywords={},  # noqa: B006
):
    """Create a description of a scalar column (python-casacore compatible)."""
    vtype = valuetype or _DTYPE_TO_VALUETYPE.get(np.asarray(value).dtype, "double")
    desc = {
        "valueType": vtype,
        "dataManagerType": datamanagertype,
        "dataManagerGroup": datamanagergroup,
        "option": options,
        "maxlen": maxlen,
        "comment": comment,
        "keywords": dict(keywords),
    }
    return {"name": columnname, "desc": desc}


def makearrcoldesc(
    columnname,
    value,
    ndim=0,
    shape=[],  # noqa: B006
    datamanagertype="",
    datamanagergroup="",
    options=0,
    maxlen=0,
    comment="",
    valuetype="",
    keywords={},  # noqa: B006
):
    """Create a description of an array column (python-casacore compatible)."""
    vtype = valuetype or _DTYPE_TO_VALUETYPE.get(np.asarray(value).dtype, "double")
    if shape is not None and len(shape) > 0 and ndim <= 0:
        ndim = len(shape)
    desc = {
        "valueType": vtype,
        "dataManagerType": datamanagertype,
        "dataManagerGroup": datamanagergroup,
        "ndim": int(ndim),
        "shape": [int(s) for s in (shape or [])],
        # like python-casacore: shapes in descriptors are C (numpy) order
        "_c_order": True,
        "option": options,
        "maxlen": maxlen,
        "comment": comment,
        "keywords": dict(keywords),
    }
    return {"name": columnname, "desc": desc}


def maketabdesc(descs=[]):  # noqa: B006
    """Combine column descriptions into a table description."""
    if isinstance(descs, dict):
        descs = [descs]
    return {d["name"]: dict(d["desc"]) for d in descs}


def makedminfo(tabledesc, group_spec=None):
    """Build a data manager info record from a table description.

    Columns sharing a ``dataManagerGroup`` are grouped under one data
    manager. ``group_spec`` optionally maps group names to a SPEC record
    (e.g. ``{"DataGroup": {"DEFAULTTILESHAPE": [4, 2, 16]}}``).
    """
    group_spec = group_spec or {}
    groups = {}
    for name, desc in _flatten_coldescs(tabledesc).items():
        dmtype = desc.get("dataManagerType") or "StandardStMan"
        dmgroup = desc.get("dataManagerGroup") or "StandardStMan"
        group = groups.setdefault(dmgroup, {"type": dmtype, "columns": []})
        if group["type"] != dmtype:
            raise ValueError(
                f"Mismatched dataManagerType '{dmtype}' for "
                f"dataManagerGroup '{dmgroup}' (was '{group['type']}')"
            )
        group["columns"].append(name)
    dminfo = {}
    for i, (dmgroup, group) in enumerate(groups.items()):
        dminfo[f"*{i + 1}"] = {
            "TYPE": group["type"],
            "NAME": dmgroup,
            "SEQNR": i,
            "COLUMNS": group["columns"],
            "SPEC": dict(group_spec.get(dmgroup, {})),
        }
    return dminfo


def _ms_from_descriptor(name, subtable, tabdesc, dminfo, result_name):
    deferred_keywords = []
    json_desc = None
    if tabdesc:
        json_desc, deferred_keywords = _prepare_coldescs_for_json(
            _json_safe(_flatten_coldescs(tabdesc))
        )
    arcae_table = _at.Table.ms_from_descriptor(
        name, subtable, json_desc, _json_safe(dminfo) if dminfo else None
    )
    wrapper = table(_arcae_table=arcae_table, _name=result_name)
    _write_deferred_empty_keywords(wrapper, deferred_keywords)
    return wrapper


def default_ms(name, tabdesc=None, dminfo=None):
    """Create a default MeasurementSet (python-casacore compatible)."""
    name = os.path.expanduser(name)
    return _ms_from_descriptor(name, "MAIN", tabdesc, dminfo, name)


def default_ms_subtable(subtable, name=None, tabdesc=None, dminfo=None):
    """Create a default MS subtable and link it into its parent MS."""
    subtable = subtable.upper()
    if name:
        name = os.path.expanduser(str(name))
        # python-casacore passes the full subtable path; arcae expects the
        # parent MS path and appends the subtable name itself
        parent, leaf = os.path.split(name.rstrip("/"))
        if leaf.upper() == subtable and parent:
            name = parent
    else:
        name = subtable.lower() + ".ms"
    return _ms_from_descriptor(
        name, subtable, tabdesc, dminfo, os.path.join(name, subtable)
    )


__all__ = [
    "table",
    "tablecolumn",
    "tablerow",
    "taql",
    "tableexists",
    "tableiswritable",
    "required_ms_desc",
    "complete_ms_desc",
    "makescacoldesc",
    "makearrcoldesc",
    "maketabdesc",
    "makedminfo",
    "default_ms",
    "default_ms_subtable",
]

# re-export for introspection convenience
arcae_version = getattr(arcae, "__version__", "unknown")
