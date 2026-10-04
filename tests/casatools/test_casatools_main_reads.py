"""
Partitions, their MAIN rows and the row reads of the MAIN columns, on real
casatools tables of downloaded test MSs, compared with the values that
python-casacore gives (reference_python_casacore.json, make_reference.py):

- create_partitions / create_partitions_with_main_rows: the same partitions
  and MAIN rows; select_main_rows and partition_main_rows (with and without
  the row runs) select the same rows;
- read_rows of all the rows of a partition: the same values; reads of row
  subsets (one getcol per row, calls of at most 3 cells), of channel /
  polarization ranges and of time chunks match the whole reads;
- the (time, baseline, ...) grids of the converter (read_col_conversion_numpy):
  the same values;
- check_partition_cells (the column decisions before reading): a column whose
  first cell is undefined (WEIGHT and SIGMA of uid___A002_Xe3a5fd_Xe38e,
  FLAG_CATEGORY) is not readable, as with python-casacore;
- the lazy reads of parallel_mode="time" computed in dask's threads give the
  grids of the whole reads.

Known differences with casatools:

- casatools returns Float, Complex and Int cells as float64, complex128 and
  int64: the grids, whose dtype is that of the first cell, are wider (same
  values, as on main); read_rows reads into buffers of the column dtype (same
  dtypes).
- casatools tables have no ``partnames``: the storage of a column is not known
  (``column_storage`` describes the error), so check_partition_cells verifies
  nothing beyond the first cell and the read decides; the read window of the
  small-read guard is the nominal one instead of the tile.

Skipped where python-casacore is installed (the Linux and macOS workflows).
"""

import pytest

from tests.casatools import reference as ref

ref.skip_unless_casatools_backend()

STORAGE_UNKNOWN = "unverified: storage unknown ("


@pytest.fixture(scope="module")
def reference() -> dict:
    return ref.load_reference()


def column_record_problems(where: str, expected: list, actual: list) -> list[str]:
    """
    The differences of a read_main_columns record from the python-casacore
    one that are not the known casatools differences (see the module
    docstring).
    """
    fields = dict(
        zip(ref.COLUMN_FIELDS, zip(expected, actual, strict=True), strict=True)
    )
    problems = []
    exp_check, act_check = fields["check"]
    storage_unknown = act_check.startswith(STORAGE_UNKNOWN)
    if exp_check != act_check and not (
        storage_unknown and not exp_check.startswith("not readable")
    ):
        problems.append(f"{where}: check {act_check!r}, expected {exp_check!r}")
    exp_window, act_window = fields["window"]
    if exp_window != act_window and not (
        storage_unknown and act_window.endswith("/nominal")
    ):
        problems.append(f"{where}: window {act_window}, expected {exp_window}")
    if fields["values"][0] != fields["values"][1]:
        problems.append(
            f"{where}: values {fields['values'][1]}, expected {fields['values'][0]}"
        )
    if not ref.same_up_to_widening(*fields["grid"]):
        problems.append(
            f"{where}: grid {fields['grid'][1]}, expected {fields['grid'][0]}"
        )
    return problems


@pytest.mark.parametrize("case", list(ref.READ_CASES))
def test_partitions_rows_and_column_reads(case, reference):
    spec = ref.READ_CASES[case]
    expected = reference["reads"][case]
    actual = ref.main_table_fingerprint(
        ref.ms_path(spec["ms"]), spec["partition_scheme"], spec["read_columns"]
    )
    assert actual["n_partitions"] == expected["n_partitions"]
    assert actual["partitions"] == expected["partitions"]
    assert actual["rows"] == expected["rows"]
    if not spec["read_columns"]:
        return
    assert sorted(actual["columns"]) == sorted(expected["columns"])
    problems = []
    for idx, exp_columns in sorted(expected["columns"].items()):
        act_columns = actual["columns"][idx]
        assert sorted(act_columns) == sorted(exp_columns), f"partition {idx}"
        for col, exp_record in sorted(exp_columns.items()):
            problems += column_record_problems(
                f"partition {idx} {col}", exp_record, act_columns[col]
            )
    assert not problems, "\n".join(problems)


def test_undefined_cells_not_readable(reference):
    """
    WEIGHT and SIGMA of uid___A002_Xe3a5fd_Xe38e have no defined cell (the
    MS has no WEIGHT_SPECTRUM: its conversion uses WEIGHT=1), nor has
    FLAG_CATEGORY: check_partition_cells decides, for every partition, that
    they cannot be read, as with python-casacore.
    """
    from xradio._utils._casacore.tables import open_table_ro
    from xradio.measurement_set._utils._msv2._tables.read_rows import (
        ColumnNotReadableError,
        check_partition_cells,
    )
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    in_file = str(ref.ms_path(ref.SD_NO_WEIGHT))
    partitions, runs = create_partitions_with_main_rows(in_file, [])
    columns = ("WEIGHT", "SIGMA", "FLAG_CATEGORY")
    with open_table_ro(in_file) as main_tb:
        assert "WEIGHT_SPECTRUM" not in main_tb.colnames()
        for idx in range(len(partitions)):
            rows = runs[idx].rows()
            for col in columns:
                assert not main_tb.iscelldefined(col, int(rows[0]))
                with pytest.raises(ColumnNotReadableError, match="first cell"):
                    check_partition_cells(main_tb, col, rows)
    for record in reference["reads"]["sd_no_weight"]["columns"].values():
        for col in columns:
            assert record[col][0].startswith("not readable: "), col


@pytest.mark.parametrize("ms_name", [ref.LOFAR, ref.SD_STANDARD])
def test_lazy_time_chunk_reads_in_threads(ms_name):
    """
    The lazy reads of parallel_mode="time" (read_col_conversion_dask, about 16
    blocks per column) computed by dask's threaded scheduler give the grid of
    read_col_conversion_numpy, every time: casatools tables are read by one
    thread at a time (concurrent reads crashed, hung or returned wrong
    values).
    """
    import dask
    import numpy as np

    from xradio._utils._casacore.tables import open_table_ro
    from xradio.measurement_set._utils._msv2._tables.read import (
        read_col_conversion_dask,
        read_col_conversion_numpy,
    )
    from xradio.measurement_set._utils._msv2._tables.read_rows import MainTableRows
    from xradio.measurement_set._utils._msv2.conversion import (
        calc_indx_for_row_split,
    )
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    in_file = str(ref.ms_path(ms_name))
    _, runs = create_partitions_with_main_rows(in_file, [])
    with open_table_ro(in_file) as main_tb:
        main_rows = MainTableRows(main_tb, runs[0].rows())
        tidxs, bidxs, ant1, _, utimes = calc_indx_for_row_split(main_rows)
        shape = (len(utimes), len(ant1))
        columns = [
            col
            for col in ("DATA", "FLOAT_DATA", "FLAG", "WEIGHT", "UVW")
            if col in main_tb.colnames()
        ]
        expected = {
            col: read_col_conversion_numpy(main_rows, col, shape, tidxs, bidxs)
            for col in columns
        }
        time_chunksize = max(1, shape[0] // 16)
        lazy = [
            read_col_conversion_dask(
                main_rows, col, shape, tidxs, bidxs, time_chunksize
            )
            for col in columns
        ]
        for _ in range(3):
            with dask.config.set(scheduler="threads", num_workers=8):
                computed = dask.compute(*lazy)
            for col, values in zip(columns, computed, strict=True):
                np.testing.assert_array_equal(values, expected[col], err_msg=col)
