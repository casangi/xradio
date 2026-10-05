"""
The ``xradio_msv2`` engine against the converter on downloaded MSs (the slow
tier of test_open_msv2_equivalence.py): a copy of each MS is converted with
``convert_msv2_to_processing_set`` and opened with
``xr.open_datatree(copy, engine=MSv2BackendEntrypoint, ...)``, which must give
the converted processing set opened with ``open_processing_set``.

The MSs cover reversed frequencies, rows in decreasing time order (a sorted
copy), many partitions (FIELD_ID and SCAN_NUMBER), VLBI sub-datasets (gain
curve, phase cal, system calibration), single dish, ephemeris fields,
POINTING (lazy pointing_xds), phased arrays and WEIGHT_SPECTRUM.

Slow (about 7 minutes, and the MSs are downloaded into ``MS_DIR``, which the
stakeholder and casatools tests share): marked ``slow`` and run only with the
environment variable ``XRADIO_SLOW_TESTS=1``.
"""

import os
import pathlib
import shutil

import numpy as np
import pytest
import xarray as xr

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set import convert_msv2_to_processing_set, open_processing_set
from xradio.measurement_set._utils._msv2 import partition_cache
from xradio.measurement_set._utils._msv2._tables import read_rows
from xradio.measurement_set._utils._msv2.partition_cache import PARTITIONS_MEMO
from xradio.testing.measurement_set.equivalence import (
    assert_nodes_identical,
    assert_processing_sets_equivalent,
)
from xradio.testing.measurement_set.io import download_measurement_set

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("XRADIO_SLOW_TESTS") != "1",
        reason="slow: set XRADIO_SLOW_TESTS=1 to run (downloads the test MSs)",
    ),
]

ENGINE = MSv2BackendEntrypoint
# The MSs are downloaded here (once), as in the stakeholder and casatools
# tests; the tests open and convert copies (opening may store partitions)
MS_DIR = pathlib.Path("/tmp/test")

VLASS = "VLASS3.2.sb45755730.eb46170641.60480.16266136574.split.v6.ms"
ALMA = "ALMA_uid___A002_X1003af4_X75a3.split.avg.ms"


def _every_1000th_field(partition):
    return partition["FIELD_ID"][0] % 1000 == 0


# case: (MS, options of the converter and of the engine, the copy: "same",
# "time_descending" (rows), or "odd_direction" (a POINTING DIRECTION cell of
# another shape))
CASES = {
    # reversed frequencies
    "antennae": ("Antennae_North.cal.lsrk.split.ms", {}, "same"),
    "antennae_time_descending_time7": (
        "Antennae_North.cal.lsrk.split.ms",
        {"main_chunksize": {"time": 7}},
        "time_descending",
    ),
    # 72 partitions
    "snr_field_scan_time4": (
        "SNR_G55_10s.split.ms",
        {
            "partition_scheme": ["FIELD_ID", "SCAN_NUMBER"],
            "main_chunksize": {"time": 4},
        },
        "same",
    ),
    # gain curve, phase cal, system calibration
    "vlba": ("VLBA_TL016B_split.ms", {}, "same"),
    # single dish, reversed frequencies, ephemeris, POINTING
    "alma_sd_time7": (
        "uid___A002_X1015532_X1926f.small.ms",
        {"main_chunksize": {"time": 7}},
        "same",
    ),
    # single dish in StandardStMan, POINTING
    "sdimaging_time7": ("sdimaging.ms", {"main_chunksize": {"time": 7}}, "same"),
    "venus_ephemeris_time5": (
        "venus_ephem_test.ms",
        {"main_chunksize": {"time": 5}},
        "same",
    ),
    # phased array
    "ska_low": ("ska_low_sim_18s.ms", {}, "same"),
    # WEIGHT_SPECTRUM
    "gmrt": ("gmrt.ms", {}, "same"),
    # an on-the-fly mosaic: fragmented partitions, 550,831 POINTING rows
    "vlass": (VLASS, {}, "same"),
    # cells of 3 shapes in one column, ephemeris fields
    "alma": (ALMA, {}, "same"),
    # a DIRECTION cell of another shape, as the converter reads it: without
    # the pointing_xds where it falls (sdimaging), without POINTING_BEAM in the
    # 12 partitions whose times cover it (VLASS)
    "sdimaging_odd_direction": ("sdimaging.ms", {}, "odd_direction"),
    "vlass_odd_direction": (VLASS, {}, "odd_direction"),
    # 48,824 partitions, of which the filter selects 52
    "vlass_field_filtered": (
        VLASS,
        {"partition_scheme": ["FIELD_ID"], "partition_filter": _every_1000th_field},
        "same",
    ),
}


def _copy_ms(name: str, rows: str, target: pathlib.Path) -> str:
    """A copy of a downloaded MS (``rows``: "same"; "time_descending": a
    deep copy with the rows sorted by decreasing TIME, ANTENNA1, ANTENNA2;
    "odd_direction": the POINTING DIRECTION cell of the middle row with two
    polynomial terms, the others have one)."""
    source = download_measurement_set(name, MS_DIR)
    target.mkdir()
    msname = str(target / name)
    from casacore import tables

    if rows in ("same", "odd_direction"):
        shutil.copytree(source, msname, symlinks=True)
        if rows == "odd_direction":
            pointing = os.path.join(msname, "POINTING")
            with tables.table(pointing, readonly=False, ack=False) as table:
                row = table.nrows() // 2
                cell = table.getcell("DIRECTION", row)
                table.putcell("DIRECTION", row, np.vstack([cell, cell * 0 + 1e-6]))
        return msname

    with tables.table(str(source), ack=False) as main_tb:
        with main_tb.sort("TIME desc, ANTENNA1 desc, ANTENNA2 desc") as rows_tb:
            rows_tb.copy(msname, deep=True).close()
    with tables.table(msname, ack=False) as main_tb:
        times = main_tb.getcol("TIME")
    assert times[0] > times[-1] and (times[:-1] >= times[1:]).all()
    return msname


@pytest.fixture
def grid_reads(monkeypatch):
    """The number of MAIN grid reads (read_rows_to_grid calls)."""
    reads = []
    read_rows_to_grid = read_rows.read_rows_to_grid

    def spy(table, col, plan, *args, **kwargs):
        reads.append(col)
        return read_rows_to_grid(table, col, plan, *args, **kwargs)

    monkeypatch.setattr(read_rows, "read_rows_to_grid", spy)
    return reads


@pytest.mark.parametrize("case", list(CASES))
def test_engine_equals_the_converted_processing_set(case, tmp_path, grid_reads):
    """
    On a copy of the MS, the engine's processing set equals the converted
    one: with chunks={} against array_backend="dask" (cold: the partitions
    are computed and stored in the copy; nothing of MAIN's data is read at
    open), and with chunks=None against array_backend="xarray" (warm: the
    stored partitions, checked against their rows). Same nodes, identical
    datasets (dates aside), the converter's dask chunks of the main data
    variables, lazy selections and accessors.
    """
    name, options, rows = CASES[case]
    msname = _copy_ms(name, rows, tmp_path / "ms")
    reference = str(tmp_path / "reference.ps.zarr")
    try:
        convert_msv2_to_processing_set(
            msname, reference, persistence_mode="w", **options
        )
        grid_reads.clear()
        partition_cache.clear_partition_memo()
        cold = xr.open_datatree(
            msname, engine=ENGINE, chunks={}, partition_cache="auto", **options
        )
        assert grid_reads == []
        assert PARTITIONS_MEMO.stats["computed"] == 1
        assert os.path.isdir(os.path.join(msname, partition_cache.SUBTABLE_NAME))
        assert_processing_sets_equivalent(
            cold, open_processing_set(reference, array_backend="dask")
        )
        del cold

        partition_cache.clear_partition_memo()
        warm = xr.open_datatree(
            msname, engine=ENGINE, chunks=None, partition_cache="auto", **options
        )
        assert PARTITIONS_MEMO.stats["stored hits"] == 1
        assert PARTITIONS_MEMO.stats["computed"] == 0
        assert_nodes_identical(
            warm, open_processing_set(reference, array_backend="xarray")
        )
    finally:
        partition_cache.clear_partition_memo()
        shutil.rmtree(reference, ignore_errors=True)
        shutil.rmtree(tmp_path / "ms", ignore_errors=True)
