"""
xradio reads MSv2 with casatools where python-casacore is not installed (the
casatools test workflow): through the casatools shim
(xradio._utils._casacore.casacore_from_casatools), with getcol reads (no
in-place reads) and without the sub-table cache.

Skipped where casatools is not installed or python-casacore is (the Linux and
macOS workflows), see reference.skip_unless_casatools_backend.
"""

import sys

import pytest

from tests.casatools import reference as ref

ref.skip_unless_casatools_backend()

SHIM = ref.SHIM_MODULE


def test_xradio_reads_msv2_with_casatools():
    from xradio._utils._casacore import tables as table_utils
    from xradio.measurement_set._utils._msv2 import partition_queries
    from xradio.measurement_set._utils._msv2._tables import (
        read,
        read_main_table,
        read_rows,
    )

    for module in (table_utils, partition_queries, read, read_main_table, read_rows):
        assert module.tables.__name__ == SHIM, module.__name__
    assert not read_rows.backend_has_in_place_reads()


def test_casatools_tables_read_with_getcol():
    """The MAIN table is a casatools table, without python-casacore's in-place
    reads (getcolnp, getcolslicenp): the row reads use getcol (and no
    selectrows, which casatools tables have)."""
    import casatools

    from xradio._utils._casacore.tables import open_table_ro
    from xradio.measurement_set._utils._msv2._tables.read_rows import (
        has_in_place_reads,
    )

    with open_table_ro(str(ref.ms_path(ref.LOFAR))) as main_tb:
        assert isinstance(main_tb, casatools.table)
        assert type(main_tb).__module__ == SHIM
        assert not has_in_place_reads(main_tb)
        assert not hasattr(main_tb, "getcolnp")
        assert not hasattr(main_tb, "getcolslicenp")


def test_subtable_cache_not_used():
    """Sub-tables are read per partition: no sub-table cache, also when one is
    passed (the conversions check that none is created)."""
    from xradio.measurement_set._utils._msv2 import conversion
    from xradio.measurement_set._utils._msv2._tables import subtable_cache as sc

    assert not sc.subtable_cache_supported()
    assert sc.resolve_subtable_cache(sc.SubtableCache()) is None
    assert sc.resolve_subtable_cache(None) is None
    assert conversion.resolve_subtable_cache(None) is None


def test_import_error_of_the_shim_is_not_skipped(monkeypatch):
    """
    With casatools installed and python-casacore not, an error importing
    xradio's casatools shim (or casatools) is raised by
    skip_unless_casatools_backend, not turned into a skip: a casatools that
    cannot be imported fails the casatools workflow.
    """
    monkeypatch.setitem(sys.modules, ref.SHIM_MODULE, None)  # its import raises
    try:
        ref.skip_unless_casatools_backend()
    except ImportError:
        pass
    except pytest.skip.Exception as exc:
        pytest.fail(f"skipped instead of raising the import error: {exc}")
    else:
        pytest.fail("no import error")
