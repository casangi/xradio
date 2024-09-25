import pytest
from pathlib import Path

import numpy as np


@pytest.mark.parametrize(
    "datetimes, expected_result",
    [
        (np.array([0]), np.array([3.5067168e09])),
        (
            np.array([-1, 10, 1_000_000_000_000, 1_000_000_000_000_000]),
            np.array([3.5067168e09, 3.5067168e09, 3.5067178e09, 3.5077168e09]),
        ),
    ],
)
def test_revert_time(datetimes, expected_result):
    from xradio.correlated_data._utils._ms._tables.write import revert_time

    assert all(revert_time(datetimes) == expected_result)


@pytest.mark.parametrize(
    "nptype, expected_result",
    [
        ("int64", "int"),
        ("int32", "int"),
        ("bool", "bool"),
        ("float32", "float"),
        ("float64", "double"),
        ("datetime64[ns]", "double"),
        ("complex64", "complex"),
        ("complex128", "dcomplex"),
        ("<U34", "string"),
        (None, "bad"),
        ("", "bad"),
        ("foo", "bad"),
    ],
)
def test_type_converter(nptype, expected_result):
    from xradio.correlated_data._utils._ms._tables.write import type_converter

    assert type_converter(nptype) == expected_result


def test_create_table_pol(pol_xds_min, tmp_path):
    from xradio.correlated_data._utils._ms._tables.write import create_table

    outtab = str(Path(tmp_path, "test_create_table_pol_out.tab"))
    create_table(outfile=outtab, xds=pol_xds_min, max_rows=100, generic=True)


def test_create_table_ant_with_col(ant_xds_min, tmp_path):
    """Writes sub-list of columns"""
    from xradio.correlated_data._utils._ms._tables.write import create_table

    outtab = str(Path(tmp_path, "test_create_table_ant_out.tab"))
    create_table(
        outfile=outtab,
        xds=ant_xds_min,
        max_rows=100,
        cols={"NAME": "name", "POSITION": "position", "DISH_DIAMETER": "dish_diameter"},
        generic=True,
    )


def test_create_table_with_infile(main_xds_min, ms_minimal_required, tmp_path):
    """Uses the 'infile' param to provide a source of subtables to be copied over"""
    from xradio.correlated_data._utils._ms._tables.write import create_table

    outtab = str(Path(tmp_path, "test_create_table_main_with_infile.tab"))
    create_table(
        outfile=outtab,
        xds=main_xds_min,
        infile=ms_minimal_required.fname,
        max_rows=10,
        cols={"TIME": "time", "SCAN_NUMBER": "scan_number"},
        generic=False,
    )


def test_write_generic_table_ant(ant_xds_min, tmp_path):
    from xradio.correlated_data._utils._ms._tables.write import write_generic_table

    dirname = Path(tmp_path, "test_write_generic_table.ant")
    write_generic_table(ant_xds_min, outfile=dirname, subtable="")


def test_write_generic_table_ant_named(
    ant_xds_min,
    ms_minimal_for_writes,
):
    """giving subtable name which will require the presence of a parent main table"""
    from xradio.correlated_data._utils._ms._tables.write import write_generic_table

    write_generic_table(
        ant_xds_min, outfile=ms_minimal_for_writes.fname, subtable="antenna"
    )


def test_write_generic_table_pol(pol_xds_min, tmp_path):
    from xradio.correlated_data._utils._ms._tables.write import write_generic_table

    dirname = Path(tmp_path, "test_write_generic_table.pol")
    write_generic_table(pol_xds_min, outfile=dirname, subtable="")
