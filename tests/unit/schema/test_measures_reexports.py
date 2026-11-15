"""The shared schema building blocks were moved out of
``xradio.measurement_set.schema`` (to ``xradio.schema.measures``, and the
dimensions on to ``xradio.schema.dimensions``), and the polarization and
casacore tables out of the image and measurement set utilities (to
``xradio.schema.polarization`` and ``xradio.schema.measures``). The old
import paths re-export them. These tests pin the re-exports: each name must
be importable from its old modules and be the same object.
"""

import pytest

import xradio.image._util.conventions as conventions
import xradio.measurement_set.schema as ms_schema
import xradio.schema.dimensions as dimensions
import xradio.schema.measures as measures
import xradio.schema.polarization as polarization
from xradio.measurement_set._utils._utils.stokes_types import stokes_types

# The names that earlier releases moved from xradio.measurement_set.schema to
# xradio.schema.measures
MOVED_TO_MEASURES = (
    "ZD",
    "AllowedDopplerTypes",
    "AllowedEllipsoid",
    "AllowedLocationCoordinateSystems",
    "AllowedLocationFrames",
    "AllowedSkyCoordFrames",
    "AllowedSpectralCoordFrames",
    "AllowedTimeFormats",
    "AllowedTimeScales",
    "CartesianPosLabel",
    "Doppler",
    "DopplerArray",
    "EllipsoidDirLabel",
    "EllipsoidDisLabel",
    "Frequency",
    "Location",
    "LocationArray",
    "Polarization",
    "PolarizationArray",
    "Quantity",
    "QuantityInHertzArray",
    "QuantityInMetersArray",
    "QuantityInMetersPerSecondArray",
    "QuantityInRadiansArray",
    "QuantityInSecondsArray",
    "SkyCoord",
    "SkyCoordArray",
    "SkyDirLabel",
    "SkyDisLabel",
    "SpectralCoord",
    "SpectralCoordArray",
    "Time",
    "TimeArray",
    "UnitsHertz",
    "UnitsMeters",
    "UnitsMetersPerSecond",
    "UnitsOfDopplerShift",
    "UnitsOfLocationInMetersOrRadians",
    "UnitsOfSkyCoordInMetersOrRadians",
    "UnitsRadians",
    "UnitsSeconds",
)

# The dimensions of xradio.schema.dimensions that xradio.schema.measures
# defined before
MEASURES_DIMENSIONS = (
    "ZD",
    "CartesianPosLabel",
    "EllipsoidDirLabel",
    "EllipsoidDisLabel",
    "Frequency",
    "Polarization",
    "SkyDirLabel",
    "SkyDisLabel",
    "Time",
)

# The dimensions of xradio.schema.dimensions that xradio.measurement_set.schema
# defined before
MS_DIMENSIONS = ("AntennaName", "BaselineId", "ReceptorLabel")

# The tables of xradio.schema.measures and xradio.schema.polarization that
# xradio.image._util.conventions defined before
CONVENTIONS_TABLES = (
    (measures, "CASACORE_SPECTRAL_FRAMES"),
    (measures, "CASACORE_TO_OBSERVER"),
    (measures, "CASACORE_EPOCH_REF_TO_SCALE"),
    (polarization, "CANONICAL_POLARIZATION_ORDER"),
    (polarization, "canonical_polarization_order"),
)


@pytest.mark.parametrize("name", MOVED_TO_MEASURES)
def test_reexported_from_measurement_set_schema(name):
    assert hasattr(ms_schema, name), (
        f"xradio.measurement_set.schema no longer re-exports {name} from "
        "xradio.schema.measures"
    )
    assert getattr(ms_schema, name) is getattr(measures, name)


@pytest.mark.parametrize("name", MEASURES_DIMENSIONS)
def test_dimension_reexported_from_measures(name):
    assert getattr(measures, name) is getattr(dimensions, name)


@pytest.mark.parametrize("name", MEASURES_DIMENSIONS + MS_DIMENSIONS)
def test_dimension_reexported_from_measurement_set_schema(name):
    assert getattr(ms_schema, name) is getattr(dimensions, name)


@pytest.mark.parametrize(("module", "name"), CONVENTIONS_TABLES)
def test_table_reexported_from_image_conventions(module, name):
    assert getattr(conventions, name) is getattr(module, name)


def test_polarization_reexports():
    assert polarization.PolarizationArray is measures.PolarizationArray
    assert stokes_types is polarization.CASACORE_STOKES_TYPES
