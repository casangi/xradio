"""Read the Zernike aperture coefficient tables of Sekhar et al. (2022) as a
Zernike polynomial coefficient dataset (``zpc_xds``)."""

from __future__ import annotations

import os

import numpy
import xarray

from xradio.measurement_set._utils._utils.stokes_types import stokes_types
from xradio.primary_beam.schema import ZPC_SCHEMA_VERSION

_COLUMNS = ("stokes", "freq", "ind", "real", "imag")
_CIRCULAR = (5, 6, 7, 8)
_LINEAR = (9, 10, 11, 12)


def read_zpc_csv(
    csv_name: str,
    telescope_name: str,
    dish_diameter: float,
    max_rad_1GHz: float,
    aperture_frame: str = "Not recorded in the coefficient table.",
    frequency_unit: float = 1.0e6,
    antenna_type: str | None = None,
    blockage_diameter: float | None = None,
) -> xarray.Dataset:
    """Read a Zernike aperture coefficient table in the CSV format of Sekhar et
    al. (2022, AJ 163, 87) as a ``zpc_xds``.

    The table has one row per frequency, correlation and polynomial, with the
    columns ``stokes, freq, ind, real, imag`` and optionally the aperture scale
    ``eta``, or ``etax`` and ``etay`` when they are equal. A header line starting
    with ``#`` names the columns; without one, they are taken in that order.

    * ``stokes`` is the casacore ``Stokes`` code of the correlation: 5 to 8 for
      ``RR, RL, LR, LL`` or 9 to 12 for ``XX, XY, YX, YY``.
    * ``freq`` is the frequency in units of ``frequency_unit`` Hz.
    * ``ind`` is the index of the polynomial in the ``sirius`` basis, see
      :py:data:`~xradio.primary_beam.schema.AllowedZernikeBases`.

    Polynomials missing from the table get a zero coefficient, and ``ETA`` is 1
    when the table has no aperture scale.

    :param csv_name: Path of the CSV file
    :param telescope_name: Name of the telescope, for example ``"VLA"``
    :param dish_diameter: Diameter of the dish, in metres
    :param max_rad_1GHz: Radius from the beam centre, in radians at 1 GHz, out
        to which beams computed from the model apply
    :param aperture_frame: Orientation of the aperture coordinates, see
        :py:class:`~xradio.primary_beam.schema.ZernikePolynomialCoefficientsXds`.
        The tables do not record it.
    :param frequency_unit: Hz per unit of the ``freq`` column; the published
        tables are in MHz
    :param antenna_type: Antenna type of a heterogeneous array that the table
        applies to
    :param blockage_diameter: Diameter of the central blockage, in metres
    :returns: Zernike polynomial coefficient dataset
    """
    names, rows = _read_table(csv_name)
    if names is None:
        if rows.shape[1] not in (5, 6):
            raise ValueError(
                f"{csv_name} has {rows.shape[1]} columns and no header; "
                "expected stokes, freq, ind, real, imag and optionally eta"
            )
        names = list(_COLUMNS) + ["eta"] * (rows.shape[1] == 6)
    if len(names) != rows.shape[1]:
        raise ValueError(
            f"{csv_name} names {len(names)} columns in its header but has {rows.shape[1]}"
        )
    missing = [name for name in _COLUMNS if name not in names]
    if missing:
        raise ValueError(f"{csv_name} has no column {', '.join(missing)}")
    column = {name: rows[:, names.index(name)] for name in names}

    codes = column["stokes"].astype(int)
    index = column["ind"].astype(int)
    if not (
        numpy.array_equal(codes, column["stokes"])
        and numpy.array_equal(index, column["ind"])
    ):
        raise ValueError(f"{csv_name} has non-integer stokes or ind values")
    pol_codes = numpy.unique(codes)
    if not (set(pol_codes) <= set(_CIRCULAR) or set(pol_codes) <= set(_LINEAR)):
        raise ValueError(
            f"{csv_name} has Stokes codes {pol_codes.tolist()}; expected those of one "
            f"feed basis, {list(_CIRCULAR)} or {list(_LINEAR)}"
        )
    if index.min() < 0:
        raise ValueError(f"{csv_name} has negative polynomial indices")

    frequency = numpy.unique(column["freq"])
    i_freq = numpy.searchsorted(frequency, column["freq"])
    i_pol = numpy.searchsorted(pol_codes, codes)
    n_coefficient = int(index.max()) + 1
    flat = numpy.ravel_multi_index(
        (i_freq, i_pol, index), (frequency.size, pol_codes.size, n_coefficient)
    )
    if numpy.unique(flat).size != flat.size:
        raise ValueError(
            f"{csv_name} has more than one row for a frequency, correlation and polynomial"
        )

    shape = (frequency.size, pol_codes.size, n_coefficient)
    zpc = numpy.zeros(shape, dtype=complex)
    zpc[i_freq, i_pol, index] = column["real"] + 1j * column["imag"]
    eta = numpy.ones(shape)
    if "eta" in column:
        eta[i_freq, i_pol, index] = column["eta"]
    elif "etax" in column or "etay" in column:
        if "etax" not in column or "etay" not in column:
            raise ValueError(f"{csv_name} has only one of the etax and etay columns")
        if not numpy.allclose(column["etax"], column["etay"]):
            raise ValueError(
                f"{csv_name} has different etax and etay; a zpc_xds has one aperture scale"
            )
        eta[i_freq, i_pol, index] = column["etax"]

    attrs = {
        "type": "zernike_polynomial_coefficients",
        "schema_version": ZPC_SCHEMA_VERSION,
        "zernike_basis": "sirius",
        "telescope_name": telescope_name,
        "dish_diameter": float(dish_diameter),
        "max_rad_1GHz": float(max_rad_1GHz),
        "aperture_frame": aperture_frame,
        "origin": {
            "file": os.path.basename(csv_name),
            "format": "Zernike aperture coefficient CSV of Sekhar et al. (2022, AJ 163, 87)",
        },
    }
    if antenna_type is not None:
        attrs["antenna_type"] = antenna_type
    if blockage_diameter is not None:
        attrs["blockage_diameter"] = float(blockage_diameter)
    dims = ("frequency", "polarization", "coefficient")
    return xarray.Dataset(
        {
            "ZPC": (dims, zpc, {"long_name": "Zernike polynomial coefficients"}),
            "ETA": (dims, eta, {"long_name": "Aperture scale"}),
        },
        coords={
            "frequency": (
                "frequency",
                frequency * frequency_unit,
                {"type": "spectral_coord", "units": "Hz", "long_name": "Frequency"},
            ),
            "polarization": [stokes_types[int(code)] for code in pol_codes],
            "coefficient": (
                "coefficient",
                numpy.arange(n_coefficient),
                {"long_name": "Zernike polynomial index"},
            ),
        },
        attrs=attrs,
    )


def _read_table(csv_name: str) -> tuple[list[str] | None, numpy.ndarray]:
    """Column names from the first comment line before the data that holds a
    comma (None without one), and the data rows."""
    names, values = None, []
    with open(csv_name) as csv_file:
        for line in csv_file:
            stripped = line.strip()
            if stripped.startswith("#"):
                if names is None and not values and "," in stripped:
                    names = [name.strip().lower() for name in stripped[1:].split(",")]
            elif stripped:
                values.append([float(value) for value in stripped.split(",")])
    if not values or len({len(row) for row in values}) != 1:
        raise ValueError(f"{csv_name} has no rows or rows of different lengths")
    return names, numpy.array(values)
