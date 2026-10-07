"""
The ``xradio_msv2`` xarray engine on the stakeholder MSs: the processing set
it opens (``xr.open_datatree(ms, engine=MSv2BackendEntrypoint, chunks={})``)
equals the one ``convert_msv2_to_processing_set`` writes with the options of
the stakeholder conversions (test_measure_set_stakeholder.py), node by node,
and passes the same accessor, time and schema checks.

The downloaded MSs (in ``MS_DIR``, shared with the other stakeholder tests)
are only read: they are opened with ``partition_cache="off"``. The partition
cache ("auto": the partitions are stored in the MS on the first open) is
used on a copy.
"""

import os
import pathlib
import shutil

import pytest
import xarray as xr
from toolviper.utils.data import download

from tests.stakeholder.test_measure_set_stakeholder import (
    base_check_ms_accessor,
    base_check_ps_accessor,
    base_check_time_secondary_datasets,
    check_expected_datasets_presence,
)
from xradio.measurement_set import (
    convert_msv2_to_processing_set,
    open_processing_set,
)
from xradio.schema.check import check_datatree
from xradio.testing.measurement_set.equivalence import (
    assert_nodes_identical,
    assert_processing_sets_equivalent,
)

# The MSs are downloaded here, as by the other stakeholder tests
MS_DIR = pathlib.Path("/tmp/test")

# The options of the stakeholder conversions that shape the processing set
# (download_and_convert_msv2_to_processing_set)
OPTIONS = {
    "main_chunksize": 0.01,
    "pointing_chunksize": 0.00001,
    "pointing_interpolate": True,
    "ephemeris_interpolate": True,
}

# MS: (partition schemes, expected sub-datasets), as in the stakeholder tests
CASES = {
    "Antennae_North.cal.lsrk.split.ms": ([[], ["FIELD_ID"]], {"antenna", "weather"}),
    "small_lofar.ms": ([[], ["FIELD_ID"]], {"antenna", "pointing"}),
    "VLBA_TL016B_split.ms": (
        [[], ["FIELD_ID"]],
        {
            "antenna",
            "gain_curve",
            "system_calibration",
            "phase_calibration",
            "weather_xds",
        },
    ),
    "venus_ephem_test.ms": ([[], ["FIELD_ID"]], {"antenna", "weather"}),
    "sdimaging.ms": (
        [[]],
        {"antenna", "pointing", "system_calibration", "weather"},
    ),
    "ska_low_sim_18s.ms": ([[]], {"antenna", "phased_array"}),
}


def _engine():
    from _xradio_xarray_backends import MSv2BackendEntrypoint

    return MSv2BackendEntrypoint


def _file_states(path: pathlib.Path) -> dict:
    """(size, mtime_ns) of every file under ``path`` but the table.lock
    files."""
    states = {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename != "table.lock":
                stat = os.stat(os.path.join(dirpath, filename))
                states[os.path.join(dirpath, filename)] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )
    return states


@pytest.mark.parametrize("ms_name", list(CASES))
def test_engine_equals_the_converted_processing_set(
    ms_name: str, tmp_path: pathlib.Path
):
    from xradio._utils._casacore.tables import uses_casatools
    from xradio.measurement_set._utils._msv2 import partition_cache

    partition_schemes, expected_secondary_xds = CASES[ms_name]
    download(file=ms_name, folder=str(MS_DIR))
    ms_path = MS_DIR / ms_name
    before = _file_states(ms_path)
    for partition_scheme in partition_schemes:
        ps_name = tmp_path / (ms_name[:-3] + ".ps.zarr")
        convert_msv2_to_processing_set(
            in_file=str(ms_path),
            out_file=str(ps_name),
            partition_scheme=partition_scheme,
            persistence_mode="w",
            **OPTIONS,
        )
        reference = open_processing_set(str(ps_name))

        engine = xr.open_datatree(
            str(ms_path),
            engine=_engine(),
            chunks={},
            partition_cache="off",
            partition_scheme=partition_scheme,
            **OPTIONS,
        )
        assert_processing_sets_equivalent(engine, reference)
        assert len(check_datatree(engine)) == 0

        loaded = xr.open_datatree(
            str(ms_path),
            engine=_engine(),
            partition_cache="off",
            partition_scheme=partition_scheme,
            **OPTIONS,
        ).load()
        base_check_ps_accessor(engine, loaded)
        base_check_ms_accessor(loaded)
        base_check_time_secondary_datasets(loaded)
        check_expected_datasets_presence(loaded, expected_secondary_xds)
        shutil.rmtree(ps_name)
    assert _file_states(ms_path) == before

    # The partition cache, on a copy: the first open stores the partitions,
    # the next open (in a new process: the memo cleared) reads them
    copy = tmp_path / "copy" / ms_name
    shutil.copytree(ms_path, copy, symlinks=True)
    ps_name = tmp_path / (ms_name[:-3] + ".ps.zarr")
    convert_msv2_to_processing_set(
        in_file=str(copy), out_file=str(ps_name), persistence_mode="w", **OPTIONS
    )
    reference = open_processing_set(str(ps_name))
    for _ in range(2):
        partition_cache.clear_partition_memo()
        engine = xr.open_datatree(
            str(copy), engine=_engine(), chunks={}, partition_cache="auto", **OPTIONS
        )
        assert_nodes_identical(engine, reference)
    stored = os.path.isdir(copy / partition_cache.SUBTABLE_NAME)
    assert stored != uses_casatools()  # (never stored with casatools)
    shutil.rmtree(ps_name)
    shutil.rmtree(copy)
