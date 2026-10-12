import numpy as np
import pytest

pytest.importorskip("arcae")

from xradio._utils._casacore import casacore_from_arcae as tables  # noqa: E402


def test_getcolshapestring_of_runs_of_cells_of_different_shapes(tmp_path):
    desc = tables.maketabdesc(
        [tables.makearrcoldesc("V", 0.0, ndim=2, valuetype="double")]
    )
    path = str(tmp_path / "runs.tbl")
    shapes = [(2, 3), (2, 3), (4, 1), (2, 3), (2, 3), (2, 3), (10, 12)]
    with tables.table(path, desc, nrow=len(shapes), readonly=False) as tb:
        for row, shape in enumerate(shapes):
            tb.putcell("V", row, np.zeros(shape))
    expected = [str(list(shape)) for shape in shapes]
    with tables.table(path) as tb:
        assert tb.getcolshapestring("V") == expected
        assert tb.getcolshapestring("V", 1, 4) == expected[1:5]
        assert tb.getcolshapestring("V", 6, 1) == expected[6:]


def test_column_descriptions_of_a_read_only_table(tmp_path):
    desc = tables.maketabdesc(
        [
            tables.makescacoldesc("S", 0.0, valuetype="double"),
            tables.makearrcoldesc("V", 0.0, ndim=1, valuetype="float"),
        ]
    )
    path = str(tmp_path / "ro.tbl")
    tables.table(path, desc, nrow=1, readonly=False).close()
    with tables.table(path) as tb:
        coldesc = tb.getcoldesc("V")
        coldesc["valueType"] = "int"
        del coldesc["ndim"]
        assert tb.getcoldesc("V")["valueType"] == "float"
        assert tb.coldatatype("V") == "float"
        assert not tb.isscalarcol("V") and tb.isvarcol("V")
        assert tb.isscalarcol("S") and tb.coldatatype("S") == "double"
        with tb.query("ROWID() < 1") as selection:
            coldesc = selection.getcoldesc("V")
            coldesc["valueType"] = "int"
            assert selection.coldatatype("V") == "float"
            assert not selection.isscalarcol("V") and selection.isvarcol("V")


def test_column_descriptions_of_a_writable_table_follow_its_columns(tmp_path):
    desc = tables.maketabdesc([tables.makescacoldesc("S", 0.0, valuetype="double")])
    with tables.table(str(tmp_path / "rw.tbl"), desc, nrow=1, readonly=False) as tb:
        assert tb.coldatatype("S") == "double" and tb.isscalarcol("S")
        tb.removecols("S")
        tb.addcols(tables.makearrcoldesc("S", 0, ndim=1, valuetype="int"))
        assert tb.coldatatype("S") == "int"
        assert tb.getcoldesc("S")["valueType"] == "int"
        assert not tb.isscalarcol("S") and tb.isvarcol("S")
