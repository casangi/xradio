"""
Schema of the xradio primary beam datasets.

The Zernike polynomial coefficient dataset (``zpc_xds``) models the complex
aperture illumination of an antenna, or of a group of identical antennas, for
each correlation of a feed pair, as the coefficients of a Zernike polynomial
expansion over the aperture, one set per frequency. The voltage beams follow
from the illumination by a Fourier transform, and the Mueller matrix from the
Kronecker product of the Jones matrices of the two antennas of a baseline. The
schema follows the conventions of the measurement set v4 schema
(``xradio.measurement_set.schema``) and reuses its spectral coordinate
measures and polarization labels:

* Data variables are UPPER_SNAKE case, coordinates and attributes are
  lower_snake case.
* The ``frequency`` coordinate uses the spectral coordinate measures of the
  measurement set frequency axis, the ``polarization`` coordinate the labels of
  :py:class:`~xradio.schema.measures.PolarizationArray`, in Jones matrix order.
* The ``schema_version`` dataset attribute holds the version of this schema
  that the dataset conforms to, ``ZPC_SCHEMA_VERSION`` for the datasets that
  xradio makes.

Schema building blocks shared with the measurement set schema live in
:py:mod:`xradio.schema.measures`.
"""

from __future__ import annotations

from typing import Literal

import numpy
import xarray

from xradio.image._util.conventions import canonical_polarization_order
from xradio.schema.bases import (
    dict_schema,
    xarray_dataarray_schema,
    xarray_dataset_schema,
)
from xradio.schema.check import SchemaIssue, SchemaIssues, check_dataset
from xradio.schema.measures import (
    AllowedSpectralCoordFrames,
    Frequency,
    Polarization,
    PolarizationArray,
    SpectralCoord,
    UnitsHertz,
)
from xradio.schema.typing import Attr, Coordof, Data, Dataof

ZPC_SCHEMA_VERSION = "0.0.1"
"""Version of the Zernike polynomial coefficient schema."""

Coefficient = Literal["coefficient"]
""" Zernike polynomial index dimension. """

AllowedZernikeBases = Literal["sirius"]
"""
Zernike polynomial bases that the coefficients refer to:

* ``sirius``: the 66 functions of ``zernike_basis`` in SIRIUS and AstroVIPER
  (``astroviper.processing_functions.simulation.zernike_polynomials``), the
  basis of the aperture coefficient tables of Sekhar et al. (2022, AJ 163,
  87). They are ordered by radial order ``n`` (0 to 10) and, within an order,
  by azimuthal order ``m`` from ``-n`` to ``n``, written on aperture
  coordinates normalised to the unit disc, and not normalised to unit power.
  63 of them are the Zernike polynomials of their order, with ``x`` and ``y``
  in the roles of the Cartesian ``Y`` and ``X``. The functions with indices
  23, 39 and 53 have additional terms, so coefficients apply to the functions
  as implemented.
"""


@dict_schema
class OriginDict:
    """Free form provenance of the coefficients: software, version and method
    that made them, and the data they were fitted to. No keys are required."""


@xarray_dataarray_schema
class FrequencyCoordArray:
    """Frequency coordinate of the Zernike polynomial coefficient dataset. Uses
    the same spectral coordinate measures as the measurement set frequency axis
    (:py:class:`~xradio.measurement_set.schema.FrequencyArray`), without the
    spectral window attributes that only apply to visibility data."""

    data: Data[Frequency, float]
    """ Frequencies at which the coefficients apply. """
    observer: Attr[AllowedSpectralCoordFrames] | None = None
    """ Spectral reference frame of the axis in the vocabulary of
    :py:data:`~xradio.schema.measures.AllowedSpectralCoordFrames`, for
    example ``"TOPO"``. """

    type: Attr[SpectralCoord] = "spectral_coord"
    """ Coordinate type. Should be ``"spectral_coord"``. """
    units: Attr[UnitsHertz] = "Hz"
    """ Units to associate with axis. """
    long_name: Attr[str] | None = "Frequency"
    """ Long-form name to use for axis. """


@xarray_dataarray_schema
class CoefficientCoordArray:
    """Index of the Zernike polynomial in the basis named by the
    ``zernike_basis`` dataset attribute."""

    data: Data[Coefficient, int]
    """ Polynomial index, from 0. """
    long_name: Attr[str] | None = "Zernike polynomial index"
    """ Long-form name to use for axis. """


@xarray_dataarray_schema
class ZpcArray:
    """Complex Zernike polynomial coefficients of the aperture illumination,
    for each frequency and correlation. The illumination of a correlation is
    ``sum_k ZPC[k] Z_k(x, y)`` inside the unit disc of the aperture
    coordinates ``(x, y)`` (see :py:class:`EtaArray`) and zero outside. The
    overall scale is arbitrary: beams are normalised by their peak."""

    data: Data[tuple[Frequency, Polarization, Coefficient], complex]
    long_name: Attr[str] | None = "Zernike polynomial coefficients"
    """ Long-form name to use for the data. """


@xarray_dataarray_schema
class EtaArray:
    """Scale of the aperture coordinates: the unit disc of the polynomials has
    a radius of ``ETA * dish_diameter / 2``. ``ETA`` is 1 when the polynomials
    span the dish. AstroVIPER uses one value per frequency (that of the first
    correlation and polynomial)."""

    data: Data[tuple[Frequency, Polarization, Coefficient], float]
    long_name: Attr[str] | None = "Aperture scale"
    """ Long-form name to use for the data. """


@xarray_dataset_schema
class ZernikePolynomialCoefficientsXds:
    """Zernike polynomial coefficient dataset (``zpc_xds``).

    Complex aperture illumination of an antenna or a group of identical
    antennas, as Zernike polynomial coefficients for each frequency and
    correlation of a feed pair. Datasets of different antenna types of a
    heterogeneous array (for example the ALMA DA and DV antennas) are separate
    datasets, distinguished by ``antenna_type``.
    """

    # --- Required Coordinates ---
    frequency: Coordof[FrequencyCoordArray]
    """ Frequencies at which the coefficients apply. """
    polarization: Coordof[PolarizationArray]
    """ Correlation labels, in the order of the casacore ``Stokes``
    enumeration, so that the four correlations of a feed pair map onto 2x2
    Jones matrices: ``RR, RL, LR, LL`` or ``XX, XY, YX, YY``. Models of the
    diagonal only hold the two parallel hands. """
    coefficient: Coordof[CoefficientCoordArray]
    """ Polynomial index. """

    # --- Required Data Variables ---
    ZPC: Dataof[ZpcArray]
    """ Complex Zernike polynomial coefficients. """
    ETA: Dataof[EtaArray]
    """ Scale of the aperture coordinates. """

    # --- Required Attributes ---
    schema_version: Attr[str]
    """ Semantic version (MAJOR.MINOR.PATCH) of the schema that the dataset
    conforms to, see ``xradio.primary_beam.schema.ZPC_SCHEMA_VERSION``. """
    telescope_name: Attr[str]
    """ Name of the telescope, for example ``"ALMA"`` or ``"VLA"``. """
    dish_diameter: Attr[float]
    """ Diameter of the dish, in metres. """
    max_rad_1GHz: Attr[float]
    """ Radius from the beam centre, in radians at 1 GHz (scaling as the
    inverse of frequency), out to which beams computed from the model apply. """
    aperture_frame: Attr[str]
    """ Orientation of the aperture coordinates ``(x, y)``: the beam
    directions that ``x`` and ``y`` are conjugate to, for example the
    direction cosines of the holography measurement in the antenna mount
    frame. Beams on the sky follow by rotation by the parallactic angle. """

    type: Attr[Literal["zernike_polynomial_coefficients"]] = (
        "zernike_polynomial_coefficients"
    )
    """ Dataset type. """
    zernike_basis: Attr[AllowedZernikeBases] = "sirius"
    """ Polynomial basis that the coefficients refer to, see
    :py:data:`AllowedZernikeBases`. """

    # --- Optional Attributes ---
    antenna_type: Attr[str] | None = None
    """ Antenna type of a heterogeneous array that the model applies to, for
    example ``"DA"`` or ``"DV"`` for ALMA. """
    blockage_diameter: Attr[float] | None = None
    """ Diameter of the central blockage, in metres. """
    origin: Attr[OriginDict] | None = None
    """ Provenance: software, version and method that made the coefficients,
    and the data they were fitted to. """


def check_zpc(zpc_xds: xarray.Dataset) -> SchemaIssues:
    """Check a Zernike polynomial coefficient dataset against the schema.

    Runs :py:func:`xradio.schema.check.check_dataset` with the
    :py:class:`ZernikePolynomialCoefficientsXds` schema and adds the checks it
    cannot do:

    * the polarization labels are in canonical (casacore ``Stokes``, Jones
      matrix) order, see :py:class:`ZernikePolynomialCoefficientsXds`;
    * the coefficient indices are ``0, 1, ...`` and, for the ``sirius`` basis,
      at most 66;
    * ``ETA`` is positive.

    :param zpc_xds: Zernike polynomial coefficient dataset to check
    :returns: List of schema issues found (empty if the dataset conforms)
    """
    issues = check_dataset(zpc_xds, ZernikePolynomialCoefficientsXds)
    if "polarization" in zpc_xds.coords:
        labels = [str(label) for label in zpc_xds.polarization.values]
        if canonical_polarization_order(labels) != list(range(len(labels))):
            issues.add(
                SchemaIssue(
                    path=[("coord", "polarization")],
                    message=f"Polarization labels {labels} are not in canonical (Jones matrix) order",
                )
            )
    if "coefficient" in zpc_xds.coords:
        index = numpy.asarray(zpc_xds.coefficient.values)
        if not numpy.array_equal(index, numpy.arange(index.size)):
            issues.add(
                SchemaIssue(
                    path=[("coord", "coefficient")],
                    message="Coefficient indices must be 0, 1, ...",
                )
            )
        if zpc_xds.attrs.get("zernike_basis") == "sirius" and index.size > 66:
            issues.add(
                SchemaIssue(
                    path=[("coord", "coefficient")],
                    message=f"The sirius basis has 66 polynomials, the dataset has {index.size} coefficients",
                )
            )
    if "ETA" in zpc_xds.data_vars and not bool(numpy.all(zpc_xds.ETA.values > 0)):
        issues.add(
            SchemaIssue(path=[("data_var", "ETA")], message="ETA must be positive")
        )
    return issues
