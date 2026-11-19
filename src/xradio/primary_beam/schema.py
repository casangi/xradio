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

AllowedZernikeBases = Literal["zernike_osa"]
"""
Zernike polynomial bases that the coefficients refer to:

* ``zernike_osa``: the Zernike polynomials ``R_n^|m|(rho) cos(m theta)`` for
  ``m >= 0`` and ``R_n^|m|(rho) sin(|m| theta)`` for ``m < 0``, not normalised
  to unit power, on the unit disc of the aperture coordinates, with ``theta``
  the position angle from ``+y`` towards ``+x`` (``x = rho sin theta``,
  ``y = rho cos theta``). Index ``j = (n (n + 2) + m) / 2`` (OSA/ANSI order)
  for radial orders ``n`` from 0 to 10, so ``j`` runs from 0 to 65. These are
  the functions of AstroVIPER's ``zernike_basis``
  (``astroviper.processing_functions.simulation.zernike_polynomials``) since
  casangi/astroviper#389.
"""

AllowedApertureFrames = Literal["antenna"]
"""
Frames of the aperture coordinates ``(x, y)`` and of the beams:

* ``antenna``: fixed to the antenna. ``+y`` is the direction of increasing
  elevation and ``+x`` that of decreasing azimuth, both as directions on the
  sky at the pointing centre. At parallactic angle ``q``, the position angle
  of the zenith from north through east, ``+y`` lies at position angle ``q``
  and ``+x`` at ``q + 90`` degrees, so they are north and east at ``q = 0``.
  The voltage beam of a correlation is
  ``V(l, m) = integral A(x, y) exp(+2 pi i a (x l + y m) / lambda) dx dy``,
  with ``A`` the illumination, ``a`` the radius of the unit disc (see
  :py:class:`EtaArray`) and ``(l, m)`` the direction cosines of the source
  from the pointing centre along ``+x`` and ``+y``.
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
    ``zernike_basis`` dataset attribute. A dataset holds any increasing
    subset of the basis; the polynomials it leaves out have zero
    coefficients."""

    data: Data[Coefficient, int]
    """ Polynomial index, increasing. """
    long_name: Attr[str] | None = "Zernike polynomial index"
    """ Long-form name to use for axis. """


@xarray_dataarray_schema
class ZpcArray:
    """Complex Zernike polynomial coefficients of the aperture illumination,
    for each frequency and correlation. The illumination of a correlation is
    ``sum_k ZPC[k] Z_k(x, y)``, over the indices ``k`` of the ``coefficient``
    coordinate, inside the unit disc of the aperture
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
    aperture_frame: Attr[AllowedApertureFrames]
    """ Frame of the aperture coordinates ``(x, y)`` and of the beams, see
    :py:data:`AllowedApertureFrames`. """

    type: Attr[Literal["zernike_polynomial_coefficients"]] = (
        "zernike_polynomial_coefficients"
    )
    """ Dataset type. """
    zernike_basis: Attr[AllowedZernikeBases] = "zernike_osa"
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
    * the coefficient indices increase from at least 0 and, for the
      ``zernike_osa`` basis, stay below 66;
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
        if index.size and (index[0] < 0 or numpy.any(numpy.diff(index) <= 0)):
            issues.add(
                SchemaIssue(
                    path=[("coord", "coefficient")],
                    message="Coefficient indices must increase from at least 0",
                )
            )
        if (
            zpc_xds.attrs.get("zernike_basis") == "zernike_osa"
            and index.size
            and index.max() > 65
        ):
            issues.add(
                SchemaIssue(
                    path=[("coord", "coefficient")],
                    message=f"The zernike_osa basis has indices 0 to 65, the dataset has {index.max()}",
                )
            )
    if "ETA" in zpc_xds.data_vars and not bool(numpy.all(zpc_xds.ETA.values > 0)):
        issues.add(
            SchemaIssue(path=[("data_var", "ETA")], message="ETA must be positive")
        )
    return issues
