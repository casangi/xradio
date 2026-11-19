"""Unit tests for reading the Zernike coefficient CSV tables (``xradio.primary_beam.read_zpc_csv``)."""

import numpy as np
import pytest

from xradio.primary_beam import check_zpc, read_zpc_csv

VLA = {"telescope_name": "VLA", "dish_diameter": 25.0, "max_rad_1GHz": 0.0149}


def write_csv(path, rows, header="#stokes,freq,ind,real,imag,eta"):
    lines = [header] if header else []
    lines += [",".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def table(codes=(5, 6, 7, 8), frequencies=(2052.0, 2948.0), n_coefficient=3):
    rows = []
    for frequency in frequencies:
        for code in codes:
            for index in range(n_coefficient):
                real = frequency + 10 * code + index
                rows.append((code, frequency, index, real, -real / 2, 1.0 + index / 10))
    return rows


def test_circular_table(tmp_path):
    zpc_xds = read_zpc_csv(write_csv(tmp_path / "t.csv", table()), **VLA)
    assert dict(zpc_xds.sizes) == {"frequency": 2, "polarization": 4, "coefficient": 3}
    assert zpc_xds.polarization.values.tolist() == ["RR", "RL", "LR", "LL"]
    np.testing.assert_array_equal(zpc_xds.frequency.values, [2.052e9, 2.948e9])
    assert (
        zpc_xds.ZPC.sel(frequency=2.948e9, polarization="LR").values[2]
        == 2948.0 + 72 - 1474.0j - 36j
    )
    np.testing.assert_array_equal(zpc_xds.ETA.values[1, 3], [1.0, 1.1, 1.2])
    assert zpc_xds.attrs["zernike_basis"] == "sirius"
    assert zpc_xds.attrs["origin"]["file"] == "t.csv"
    assert not check_zpc(zpc_xds)


def test_linear_table_without_eta_or_header(tmp_path):
    rows = [row[:5] for row in table(codes=(9, 10, 11, 12))]
    zpc_xds = read_zpc_csv(write_csv(tmp_path / "t.csv", rows, header=None), **VLA)
    assert zpc_xds.polarization.values.tolist() == ["XX", "XY", "YX", "YY"]
    assert np.all(zpc_xds.ETA.values == 1.0)
    assert not check_zpc(zpc_xds)


def test_spaced_header_and_optional_attributes(tmp_path):
    rows = table(codes=(5, 8))
    path = write_csv(
        tmp_path / "t.csv", rows, header="  # stokes , freq , ind , real , imag , eta"
    )
    zpc_xds = read_zpc_csv(path, antenna_type="DV", blockage_diameter=0.75, **VLA)
    assert zpc_xds.polarization.values.tolist() == ["RR", "LL"]
    assert zpc_xds.attrs["antenna_type"] == "DV"
    assert zpc_xds.attrs["blockage_diameter"] == 0.75
    assert not check_zpc(zpc_xds)


def test_missing_polynomials_are_zero(tmp_path):
    rows = [row for row in table(n_coefficient=4) if row[2] != 2]
    zpc_xds = read_zpc_csv(write_csv(tmp_path / "t.csv", rows), **VLA)
    assert zpc_xds.sizes["coefficient"] == 4
    assert np.all(zpc_xds.ZPC.values[:, :, 2] == 0)
    assert np.all(zpc_xds.ETA.values[:, :, 2] == 1)


def test_equal_etax_and_etay(tmp_path):
    rows = [row + (row[5],) for row in table()]
    path = write_csv(
        tmp_path / "t.csv", rows, header="#stokes,freq,ind,real,imag,etax,etay"
    )
    zpc_xds = read_zpc_csv(path, **VLA)
    np.testing.assert_array_equal(zpc_xds.ETA.values[0, 0], [1.0, 1.1, 1.2])


@pytest.mark.parametrize(
    "rows, header, message",
    [
        (
            [row + (row[5] + 0.1,) for row in table()],
            "#stokes,freq,ind,real,imag,etax,etay",
            "different etax and etay",
        ),
        (
            [row[:5] + (1.0,) for row in table()],
            "#stokes,freq,ind,real,imag,etax",
            "only one of the etax",
        ),
        (table(codes=(5, 6, 9)), "#stokes,freq,ind,real,imag,eta", "one feed basis"),
        (table(codes=(1,)), "#stokes,freq,ind,real,imag,eta", "one feed basis"),
        (table() + table()[:1], "#stokes,freq,ind,real,imag,eta", "more than one row"),
        ([row[:4] for row in table()], "#stokes,freq,ind,real", "no column imag"),
        (
            [row[:5] for row in table()],
            "#stokes,freq,ind,real,imag,eta",
            "names 6 columns",
        ),
    ],
)
def test_invalid_tables(tmp_path, rows, header, message):
    with pytest.raises(ValueError, match=message):
        read_zpc_csv(write_csv(tmp_path / "t.csv", rows, header=header), **VLA)
