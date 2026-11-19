"""Consistency of the vocabularies and casacore tables shared by the schemas
(xradio.schema.polarization and xradio.schema.measures)."""

import typing

from xradio._utils.schema import casacore_to_msv4_measure_type
from xradio.schema.measures import (
    CASACORE_DIRECTION_TO_FRAME,
    CASACORE_EPOCH_REF_TO_SCALE,
    CASACORE_SPECTRAL_FRAMES,
    CASACORE_TO_OBSERVER,
    DEFAULT_EQUINOX,
    AllowedSkyCoordFrames,
    AllowedSpectralCoordFrames,
    AllowedTimeScales,
)
from xradio.schema.polarization import (
    CANONICAL_POLARIZATION_ORDER,
    CASACORE_STOKES_TYPES,
    CIRCULAR_CORRELATIONS,
    LINEAR_CORRELATIONS,
    canonical_polarization_order,
)


def test_canonical_order_is_the_casacore_stokes_order():
    assert CANONICAL_POLARIZATION_ORDER == tuple(
        CASACORE_STOKES_TYPES[code] for code in range(1, 25)
    )


def test_correlation_products_are_in_canonical_order():
    assert CIRCULAR_CORRELATIONS == tuple(
        CASACORE_STOKES_TYPES[code] for code in range(5, 9)
    )
    assert LINEAR_CORRELATIONS == tuple(
        CASACORE_STOKES_TYPES[code] for code in range(9, 13)
    )
    for labels in (CIRCULAR_CORRELATIONS, LINEAR_CORRELATIONS):
        assert canonical_polarization_order(labels) == [0, 1, 2, 3]


def test_casacore_tables_use_the_schema_vocabularies():
    assert tuple(CASACORE_TO_OBSERVER) == CASACORE_SPECTRAL_FRAMES
    assert set(CASACORE_TO_OBSERVER.values()) <= set(
        typing.get_args(AllowedSpectralCoordFrames)
    )
    assert set(CASACORE_EPOCH_REF_TO_SCALE.values()) <= set(
        typing.get_args(AllowedTimeScales)
    )
    sky_frames = set(typing.get_args(AllowedSkyCoordFrames))
    assert set(CASACORE_DIRECTION_TO_FRAME.values()) <= sky_frames
    assert set(DEFAULT_EQUINOX) <= sky_frames


def test_measurement_set_converter_uses_the_shared_tables():
    assert casacore_to_msv4_measure_type["frequency"]["Ref_map"] == CASACORE_TO_OBSERVER
    assert (
        casacore_to_msv4_measure_type["direction"]["Ref_map"]
        == CASACORE_DIRECTION_TO_FRAME
    )
