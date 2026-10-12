"""
Read-only tables of xradio._utils._casacore.tables: the number of table
instances (ninstances, from the argument or XRADIO_ARCAE_NINSTANCES) and the
cache of open tables. The backend tables are replaced by a recorder, except in
the last test.
"""

import os
import types

import numpy as np
import pytest

from xradio._utils._casacore import tables as table_utils


class _Table:
    def __init__(self, infile, **kwargs):
        self.infile = infile
        self.kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def opened(monkeypatch):
    """The tables opened through a recording backend."""
    tables = []

    def table(infile, **kwargs):
        tables.append(_Table(infile, **kwargs))
        return tables[-1]

    monkeypatch.setattr(table_utils, "tables", types.SimpleNamespace(table=table))
    monkeypatch.setattr(table_utils, "_TABLE_CACHE", {})
    monkeypatch.delenv("XRADIO_ARCAE_NINSTANCES", raising=False)
    return tables


@pytest.mark.parametrize(
    "env, ninstances, expected",
    [
        (None, None, 1),
        ("4", None, 4),
        ("4", 0, 4),
        ("4", 2, 2),
        ("0", None, 1),
        ("-3", None, 1),
        ("many", None, 1),
        (None, 3, 3),
    ],
)
def test_open_table_ro_ninstances(monkeypatch, opened, env, ninstances, expected):
    if env is not None:
        monkeypatch.setenv("XRADIO_ARCAE_NINSTANCES", env)
    with table_utils.open_table_ro("a.tbl", ninstances=ninstances) as tb:
        assert not tb.closed
    assert opened == [tb] and tb.closed
    assert tb.infile == "a.tbl"
    assert tb.kwargs.pop("ninstances", 1) == expected
    assert tb.kwargs == {
        "readonly": True,
        "lockoptions": {"option": "usernoread"},
        "ack": False,
    }


def test_backend_without_ninstances(monkeypatch, opened):
    """A backend table without ninstances (python-casacore) is opened without it."""
    recorder = table_utils.tables.table

    def table(infile, **kwargs):
        if "ninstances" in kwargs:
            raise TypeError("unexpected keyword argument 'ninstances'")
        return recorder(infile, **kwargs)

    monkeypatch.setattr(table_utils.tables, "table", table)
    with table_utils.open_table_ro("a.tbl", ninstances=3) as tb:
        assert "ninstances" not in tb.kwargs and tb.kwargs["readonly"]
    assert opened == [tb]


def test_get_table_ro_caches_by_path_and_ninstances(monkeypatch, tmp_path, opened):
    monkeypatch.chdir(tmp_path)
    path = os.path.join(os.getcwd(), "a.tbl")
    tb = table_utils.get_table_ro("a.tbl")
    assert tb.infile == path and "ninstances" not in tb.kwargs
    assert table_utils.get_table_ro(path) is tb
    monkeypatch.setenv("XRADIO_ARCAE_NINSTANCES", "2")
    pooled = table_utils.get_table_ro(path)
    assert pooled is not tb and pooled.kwargs["ninstances"] == 2
    assert table_utils.get_table_ro(path, ninstances=2) is pooled
    assert table_utils.get_table_ro(path, ninstances=1) is tb
    assert opened == [tb, pooled] and not tb.closed and not pooled.closed

    table_utils.clear_table_cache()
    assert tb.closed and pooled.closed
    assert table_utils.get_table_ro(path, ninstances=1) is not tb
    assert len(opened) == 3


def test_clear_table_cache_closes_the_others_when_a_close_fails(monkeypatch):
    class Failing:
        def close(self):
            raise RuntimeError("table is already closed")

    first, last = _Table("first.tbl"), _Table("last.tbl")
    cache = {
        ("first.tbl", 1): first,
        ("failing.tbl", 1): Failing(),
        ("closeless.tbl", 1): object(),
        ("last.tbl", 1): last,
    }
    monkeypatch.setattr(table_utils, "_TABLE_CACHE", cache)
    table_utils.clear_table_cache()
    assert first.closed and last.closed
    assert cache == {}


def test_tables_with_several_instances_read_the_table(monkeypatch, tmp_path):
    """With the backend in use, tables opened with ninstances 2 (arcae pools
    two table instances; python-casacore has one) read the table's values."""
    from xradio._utils._casacore.backend import makescacoldesc, maketabdesc

    monkeypatch.setattr(table_utils, "_TABLE_CACHE", {})
    path = str(tmp_path / "values.tbl")
    desc = maketabdesc([makescacoldesc("V", 0.0, valuetype="double")])
    tb = table_utils.tables.table(path, desc, nrow=4, readonly=False)
    tb.putcol("V", np.arange(4.0))
    tb.close()
    with table_utils.open_table_ro(path, ninstances=2) as tb:
        np.testing.assert_array_equal(tb.getcol("V"), np.arange(4.0))
    cached = table_utils.get_table_ro(path, ninstances=2)
    np.testing.assert_array_equal(cached.getcol("V", 1, 2), [1.0, 2.0])
    table_utils.clear_table_cache()
