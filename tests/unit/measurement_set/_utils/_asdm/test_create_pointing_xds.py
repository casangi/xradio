import contextlib
import importlib.util
import io
import logging
import re
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pyasdm
import pytest
from astropy.coordinates import AltAz, SkyCoord

from xradio.measurement_set._utils._asdm import create_pointing_xds as pointing_module
from xradio.measurement_set._utils._asdm.create_pointing_xds import (
    PointingConversionError,
    create_pointing_xds,
)
from xradio.measurement_set.schema import PointingXds
from xradio.schema.check import check_dataset

MJD_TO_UNIX_NS = 3_506_716_800 * 10**9

# Rows of conftest.add_pointing_table (true, un-halved times)
FIXTURE_ROW_STARTS_NS = (5_137_194_858_648_000_000, 5_137_194_883_368_000_000)
FIXTURE_ROW_DURATION_NS = 24_240_000_000
FIXTURE_NUM_SAMPLE = 4
FIXTURE_TARGET = (-1.46, 1.14)
# offsets per antenna and row
FIXTURE_OFFSETS = {
    "CM01": ((0.1, 0.2), (0.2, 0.3)),
    "CM03": ((0.01, 0.02), (0.001, 0.002)),
}

# start of the custom test rows below: 2023-11-10T12:00:00 (ASDM ns)
T0_NS = 5_181_249_600_000_000_000
# tolerance on times: pyasdm <= 0.0.7 parses XML timeInterval with float
# arithmetic (up to ~0.5 us error)
TIME_ATOL_S = 2e-6


def to_unix(times_ns) -> np.ndarray:
    return (np.asarray(times_ns, dtype=np.int64) - MJD_TO_UNIX_NS) / 1e9


def uniform_sample_centers_ns(start_ns: int, duration_ns: int, num_sample: int):
    return [
        start_ns + ((2 * idx + 1) * duration_ns) // (2 * num_sample)
        for idx in range(num_sample)
    ]


def skyoffset_reference(target, offset) -> np.ndarray:
    """Independent reference for offsets applied to targets: astropy
    SkyOffsetFrame centered on the target, (lon, lat) = offset."""
    out = []
    for (az, alt), (d_az, d_alt) in zip(
        np.reshape(target, (-1, 2)), np.reshape(offset, (-1, 2)), strict=True
    ):
        origin = SkyCoord(az=az * u.rad, alt=alt * u.rad, frame=AltAz())
        point = SkyCoord(
            lon=d_az * u.rad, lat=d_alt * u.rad, frame=origin.skyoffset_frame()
        ).transform_to(AltAz())
        out.append([point.az.rad, point.alt.rad])
    return np.reshape(np.array(out), np.shape(target))


def assert_directions_close(actual, expected, atol=1e-10):
    """Compare (az, alt) arrays, azimuths modulo 2 pi."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    d_az = np.mod(actual[..., 0] - expected[..., 0] + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(d_az, 0.0, rtol=0, atol=atol)
    np.testing.assert_allclose(actual[..., 1], expected[..., 1], rtol=0, atol=atol)


def angles_xml(values) -> str:
    values = np.asarray(values, dtype=float).reshape(-1, 2)
    return f"2 {values.shape[0]} 2 " + " ".join(f"{val:.17g}" for val in values.ravel())


def pointing_row_xml(
    antenna_id: int,
    start_ns: int,
    duration_ns: int,
    encoder,
    target=None,
    offset=None,
    pointing_direction=None,
    use_polynomials: bool = False,
    sampled_intervals: list[tuple[int, int]] | None = None,
) -> str:
    """XML for a Pointing row. timeInterval/sampledTimeInterval values are
    (start, duration) here and written as (midpoint, duration) like in ASDMs."""
    encoder = np.asarray(encoder, dtype=float).reshape(-1, 2)
    num_sample = len(encoder)
    target = encoder if target is None else np.asarray(target, dtype=float)
    offset = np.zeros_like(target) if offset is None else np.asarray(offset)
    pointing_direction = encoder if pointing_direction is None else pointing_direction
    num_term = len(np.asarray(target).reshape(-1, 2))
    sampled = ""
    if sampled_intervals is not None:
        values = " ".join(
            f"{start + duration // 2} {duration}"
            for start, duration in sampled_intervals
        )
        sampled = (
            f"<sampledTimeInterval> 1 {len(sampled_intervals)} {values} "
            "</sampledTimeInterval>"
        )
    return f"""<row>
    <timeInterval> {start_ns + duration_ns // 2} {duration_ns} </timeInterval>
    <numSample> {num_sample} </numSample>
    <encoder> {angles_xml(encoder)} </encoder>
    <pointingTracking> true </pointingTracking>
    <usePolynomials> {"true" if use_polynomials else "false"} </usePolynomials>
    <timeOrigin> {start_ns} </timeOrigin> <numTerm> {num_term} </numTerm>
    <pointingDirection> {angles_xml(pointing_direction)} </pointingDirection>
    <target> {angles_xml(target)} </target>
    <offset> {angles_xml(offset)} </offset>
    {sampled}
    <antennaId> Antenna_{antenna_id} </antennaId> <pointingModelId> 0 </pointingModelId>
    </row>"""


def set_row_from_xml(row, xml: str):
    """
    ``row.setFromXML(xml)`` with boolean attributes (``usePolynomials``,
    ``pointingTracking``) set to their XML value: pyasdm parses XML booleans with
    ``bool(text)``, so "false" reads as True (see synthetic_asdm.set_row_from_xml).
    """
    name = "xradio_tests_asdm_synthetic_asdm"  # same module object as the conftest
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("synthetic_asdm.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].set_row_from_xml(row, xml)


def make_pointing_asdm(rows_xml: list[str], num_antenna: int = 2) -> pyasdm.ASDM:
    """In-memory ASDM with an Antenna table (names DA41, DA42, ...) and the
    given Pointing rows."""
    asdm = pyasdm.ASDM()
    antenna_table = asdm.getAntenna()
    for ant in range(num_antenna):
        antenna_row = pyasdm.AntennaRow(antenna_table)
        antenna_row.setFromXML(
            f"""<row><antennaId> Antenna_{ant} </antennaId><name>DA4{ant + 1}</name>
<antennaMake>AEM_12</antennaMake><antennaType>GROUND_BASED</antennaType>
<dishDiameter> 12.0 </dishDiameter><position> 1 3 0.0 0.0 7.0 </position>
<offset> 1 3 0.0 0.0 0.0 </offset><time> {T0_NS} </time>
<stationId> Station_{ant} </stationId></row>"""
        )
        antenna_table.add(antenna_row)
    add_pointing_rows(asdm, rows_xml)
    return asdm


def add_pointing_rows(asdm: pyasdm.ASDM, rows_xml: list[str]):
    pointing_table = asdm.getPointing()
    for row_xml in rows_xml:
        pointing_row = pyasdm.PointingRow(pointing_table)
        set_row_from_xml(pointing_row, row_xml)
        pointing_table.add(pointing_row)


def directions(num_sample: int, az0: float, alt0: float, step: float = 1e-3):
    idx = np.arange(num_sample)
    return np.stack([az0 + step * idx, alt0 - 0.5 * step * idx], axis=-1)


def read_asdm(path: str) -> pyasdm.ASDM:
    asdm = pyasdm.ASDM()
    with contextlib.redirect_stdout(io.StringIO()):
        asdm.setFromFile(path)
    return asdm


def write_binary_pointing_asdm(
    rows_xml: list[str], use_polynomials: list[bool], path: Path
) -> str:
    """Write an ASDM made of the given Pointing rows (binary Pointing table,
    the pyasdm default) with the given usePolynomials flags, set through the
    setter (overriding the value in the row XML)."""
    asdm = make_pointing_asdm(rows_xml)
    rows = asdm.getPointing().get()
    assert len(rows) == len(use_polynomials)
    for row, flag in zip(rows, use_polynomials, strict=True):
        row.setUsePolynomials(flag)
    with contextlib.redirect_stdout(io.StringIO()):
        asdm.toFile(str(path))
    assert "<BulkStoreRef" in (path / "Pointing.xml").read_text()
    return str(path)


def buggy_from_bin(eis):
    """ArrayTimeInterval.fromBin of pyasdm <= 0.0.7 (halves the start time)."""
    first = eis.readLong()
    second = eis.readLong()
    if pyasdm.types.ArrayTimeInterval.readStartTimeDurationInBin():
        return pyasdm.types.ArrayTimeInterval(first, second)
    return pyasdm.types.ArrayTimeInterval(int((first - second) / 2), second)


def fixed_from_bin(eis):
    """ArrayTimeInterval.fromBin as in the C++/Java implementations."""
    first = eis.readLong()
    second = eis.readLong()
    if pyasdm.types.ArrayTimeInterval.readStartTimeDurationInBin():
        return pyasdm.types.ArrayTimeInterval(first, second)
    return pyasdm.types.ArrayTimeInterval(first - second // 2, second)


# ---------------------------------------------------------------------------
# Empty / invalid inputs (F76, K11)
# ---------------------------------------------------------------------------


def test_create_pointing_xds_none():
    with pytest.raises(TypeError, match="NoneType"):
        create_pointing_xds(None)


@pytest.mark.parametrize(
    "asdm_fixture", ["asdm_empty", "asdm_with_spw_default", "asdm_with_spw_simple"]
)
def test_create_pointing_xds_without_pointing_rows_gives_none(asdm_fixture, request):
    asdm = request.getfixturevalue(asdm_fixture)
    assert create_pointing_xds(asdm) is None
    assert create_pointing_xds(asdm, time_range=(0.0, 2.0e9)) is None


@pytest.mark.parametrize(
    "time_range", [(10.0, 5.0), (1.0,), "foo", (None, 3.0), (float("nan"), 1.0)]
)
def test_create_pointing_xds_invalid_time_range(
    asdm_with_antenna_station_pointing, time_range
):
    with pytest.raises(ValueError, match="time_range"):
        create_pointing_xds(asdm_with_antenna_station_pointing, time_range=time_range)


# ---------------------------------------------------------------------------
# conftest Pointing table: schema and values (F08, F24, F38, F87, K1)
# ---------------------------------------------------------------------------


def test_create_pointing_xds_with_asdm_antenna_pointing(
    asdm_with_antenna_station_pointing,
):
    pointing_xds = create_pointing_xds(asdm_with_antenna_station_pointing)
    issues = check_dataset(pointing_xds, PointingXds)
    assert not issues, str(issues)

    assert pointing_xds.attrs["type"] == "pointing"
    assert dict(pointing_xds.sizes) == {
        "time_pointing": 8,
        "antenna_name": 2,
        "local_sky_dir_label": 2,
    }
    assert list(pointing_xds.antenna_name.values) == ["CM01", "CM03"]
    assert list(pointing_xds.local_sky_dir_label.values) == ["az", "alt"]
    assert set(pointing_xds.data_vars) == {"POINTING_BEAM", "POINTING_DISH_MEASURED"}
    for data_var in pointing_xds.data_vars.values():
        assert data_var.dims == ("time_pointing", "antenna_name", "local_sky_dir_label")
        assert data_var.dtype == np.float64
        assert data_var.attrs["frame"] == "altaz"
        assert data_var.attrs["units"] == "rad"


def test_create_pointing_xds_time_pointing_values(asdm_with_antenna_station_pointing):
    """time_pointing = center of every sample (start + (i + 1/2) * duration /
    numSample), unix seconds UTC, strictly increasing (F08, F87, K1)."""
    time_pointing = create_pointing_xds(
        asdm_with_antenna_station_pointing
    ).time_pointing
    expected_ns = [
        center
        for start in FIXTURE_ROW_STARTS_NS
        for center in uniform_sample_centers_ns(
            start, FIXTURE_ROW_DURATION_NS, FIXTURE_NUM_SAMPLE
        )
    ]
    np.testing.assert_allclose(
        time_pointing.values, to_unix(expected_ns), rtol=0, atol=TIME_ATOL_S
    )
    assert np.all(np.diff(time_pointing.values) > 0)
    assert time_pointing.dtype == np.float64
    assert time_pointing.attrs == {
        "type": "time_pointing",
        "units": "s",
        "scale": "utc",
        "format": "unix",
    }
    # first sample: row start 2021-09-01T06:34:18.648 + 3.03 s (independent of
    # the ASDM epoch constant; within TIME_ATOL_S, see its comment)
    first_unix = (
        np.datetime64("2021-09-01T06:34:21.678", "ns") - np.datetime64("1970-01-01")
    ) / np.timedelta64(1, "s")
    assert abs(time_pointing.values[0] - first_unix) < TIME_ATOL_S


def test_create_pointing_xds_beam_applies_offsets(asdm_with_antenna_station_pointing):
    """POINTING_BEAM = target rotated by offset (+ encoder - pointingDirection,
    zero in this table); POINTING_DISH_MEASURED = encoder (F24, F38, F87)."""
    pointing_xds = create_pointing_xds(asdm_with_antenna_station_pointing)
    np.testing.assert_array_equal(
        pointing_xds.POINTING_DISH_MEASURED.values,
        np.broadcast_to(FIXTURE_TARGET, (8, 2, 2)),
    )

    target = np.array(FIXTURE_TARGET)
    target_coord = SkyCoord(az=target[0] * u.rad, alt=target[1] * u.rad, frame=AltAz())
    for ant_name, row_offsets in FIXTURE_OFFSETS.items():
        beam = pointing_xds.POINTING_BEAM.sel(antenna_name=ant_name).values
        expected = np.concatenate(
            [
                np.broadcast_to(
                    skyoffset_reference(target, np.array(offset)),
                    (FIXTURE_NUM_SAMPLE, 2),
                )
                for offset in row_offsets
            ]
        )
        assert_directions_close(beam, expected)
        # azimuth in the same convention as the target ([-pi, pi] here)
        assert np.all(np.abs(beam[:, 0] - target[0]) < np.pi)
        # separation from the target = angular length of the offset
        separations = target_coord.separation(
            SkyCoord(az=beam[:, 0] * u.rad, alt=beam[:, 1] * u.rad, frame=AltAz())
        ).rad
        offsets = np.repeat(np.array(row_offsets), FIXTURE_NUM_SAMPLE, axis=0)
        expected_sep = np.arccos(np.cos(offsets[:, 0]) * np.cos(offsets[:, 1]))
        np.testing.assert_allclose(separations, expected_sep, rtol=0, atol=1e-10)
        assert np.all(separations > 0)


def test_create_pointing_xds_beam_with_pointing_correction():
    """With encoder != pointingDirection the correction (encoder -
    pointingDirection) is added to the rotated target (F38)."""
    num_sample = 3
    target = directions(num_sample, -2.0, 0.7)
    offset = np.array([[0.0, 0.0], [1e-3, -2e-3], [-5e-3, 4e-3]])
    pointing_direction = skyoffset_reference(target, offset)
    correction = np.array([[1e-5, -2e-5], [3e-5, 1e-5], [-4e-5, 0.0]])
    encoder = pointing_direction + correction
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(
                ant, T0_NS, 3_000_000_000, encoder, target, offset, pointing_direction
            )
            for ant in range(2)
        ]
    )
    pointing_xds = create_pointing_xds(asdm)
    for ant in range(2):
        beam = pointing_xds.POINTING_BEAM.isel(antenna_name=ant).values
        assert_directions_close(beam, pointing_direction + correction)
        np.testing.assert_allclose(
            pointing_xds.POINTING_DISH_MEASURED.isel(antenna_name=ant).values,
            encoder,
            rtol=0,
            atol=1e-15,
        )
    # zero offset row sample: beam = target + correction
    assert_directions_close(
        pointing_xds.POINTING_BEAM.isel(antenna_name=0, time_pointing=0).values,
        target[0] + correction[0],
    )


# ---------------------------------------------------------------------------
# Per-antenna time axes (F37)
# ---------------------------------------------------------------------------


def test_create_pointing_xds_antennas_with_different_rows():
    """time_pointing is the union of the antenna sample times; each antenna's
    samples are at their own times and NaN elsewhere (F37)."""
    duration = 4_000_000_000
    num_sample = 4
    enc_a0_r0 = directions(num_sample, 1.0, 0.5)
    enc_a0_r1 = directions(num_sample, 1.1, 0.4)
    # antenna 1 joins late: only the second row, half a sample later
    half_sample = duration // num_sample // 2
    enc_a1 = directions(num_sample, -1.0, 0.8)
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(0, T0_NS, duration, enc_a0_r0),
            pointing_row_xml(0, T0_NS + duration, duration, enc_a0_r1),
            pointing_row_xml(1, T0_NS + duration + half_sample, duration, enc_a1),
        ]
    )
    pointing_xds = create_pointing_xds(asdm)

    times_a0 = uniform_sample_centers_ns(
        T0_NS, duration, num_sample
    ) + uniform_sample_centers_ns(T0_NS + duration, duration, num_sample)
    times_a1 = uniform_sample_centers_ns(
        T0_NS + duration + half_sample, duration, num_sample
    )
    expected_times = np.unique(np.array(times_a0 + times_a1, dtype=np.int64))
    np.testing.assert_allclose(
        pointing_xds.time_pointing.values,
        to_unix(expected_times),
        rtol=0,
        atol=TIME_ATOL_S,
    )
    assert len(expected_times) == 12
    assert list(pointing_xds.antenna_name.values) == ["DA41", "DA42"]

    for ant, times, encoder in (
        (0, times_a0, np.concatenate([enc_a0_r0, enc_a0_r1])),
        (1, times_a1, enc_a1),
    ):
        measured = pointing_xds.POINTING_DISH_MEASURED.isel(antenna_name=ant).values
        beam = pointing_xds.POINTING_BEAM.isel(antenna_name=ant).values
        at = np.searchsorted(expected_times, times)
        np.testing.assert_array_equal(measured[at], encoder)
        assert_directions_close(beam[at], encoder)
        others = np.setdiff1d(np.arange(len(expected_times)), at)
        assert np.isnan(measured[others]).all()
        assert np.isnan(beam[others]).all()


def test_create_pointing_xds_antennas_with_different_num_sample():
    """Antennas sampled differently over the same interval (F37)."""
    duration = 2_000_000_000
    enc0 = directions(4, 0.5, 0.6)
    enc1 = directions(2, 0.7, 0.3)
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(0, T0_NS, duration, enc0),
            pointing_row_xml(1, T0_NS, duration, enc1),
        ]
    )
    pointing_xds = create_pointing_xds(asdm)
    times0 = uniform_sample_centers_ns(T0_NS, duration, 4)
    times1 = uniform_sample_centers_ns(T0_NS, duration, 2)
    expected_times = np.unique(np.array(times0 + times1, dtype=np.int64))
    np.testing.assert_allclose(
        pointing_xds.time_pointing.values,
        to_unix(expected_times),
        rtol=0,
        atol=TIME_ATOL_S,
    )
    measured = pointing_xds.POINTING_DISH_MEASURED.values
    np.testing.assert_array_equal(
        measured[np.searchsorted(expected_times, times0), 0], enc0
    )
    np.testing.assert_array_equal(
        measured[np.searchsorted(expected_times, times1), 1], enc1
    )
    assert np.isnan(measured[:, 1]).all(axis=-1).sum() == len(expected_times) - 2


# ALMA pointing sampling interval (48 ms)
ALMA_TE_NS = 48_000_000


def test_merge_sample_times_groups_near_equal_times():
    times = np.array(
        [1_000_007, 10, 0, 2_000_000, 5, 1_000_000, 10, 2_000_011], dtype=np.int64
    )
    axis, index = pointing_module._merge_sample_times(times, 10)
    # groups {0, 5, 10, 10}: mean 6.25 -> 6; {1_000_000, 1_000_007}: mean
    # 1_000_003.5 -> 1_000_004; 2_000_000 and 2_000_011 (11 ns apart) separate
    np.testing.assert_array_equal(axis, [6, 1_000_004, 2_000_000, 2_000_011])
    np.testing.assert_array_equal(index, [1, 0, 0, 2, 0, 1, 0, 3])
    assert axis.dtype == np.int64


def test_merge_sample_times_without_tolerance_is_unique():
    rng = np.random.default_rng(42)
    times = T0_NS + rng.integers(0, 50, size=200) * ALMA_TE_NS
    axis, index = pointing_module._merge_sample_times(times, 0)
    unique_times, unique_index = np.unique(times, return_inverse=True)
    np.testing.assert_array_equal(axis, unique_times)
    np.testing.assert_array_equal(index, unique_index)
    np.testing.assert_array_equal(axis[index], times)


@pytest.mark.parametrize(
    "offset_ns, merged",
    [(0, True), (1_500, True), (-4_000, True), (50_000, False), (1_000_000, False)],
)
def test_create_pointing_xds_merges_near_equal_sample_times(offset_ns, merged):
    """Sample times of different antennas a few us apart (pyasdm <= 0.0.7 time
    reading errors) are the same instant of time_pointing; samples further
    apart are kept separate."""
    num_sample0, late = 6, 2
    num_sample1 = num_sample0 - late
    start1 = T0_NS + late * ALMA_TE_NS + offset_ns
    enc0 = directions(num_sample0, 1.0, 0.8)
    enc1 = directions(num_sample1, 1.0 + late * 1e-3, 0.8 - late * 0.5e-3)
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(0, T0_NS, num_sample0 * ALMA_TE_NS, enc0),
            pointing_row_xml(1, start1, num_sample1 * ALMA_TE_NS, enc1),
        ]
    )
    pointing_xds = create_pointing_xds(asdm)
    times0 = np.array(
        uniform_sample_centers_ns(T0_NS, num_sample0 * ALMA_TE_NS, num_sample0)
    )
    times1 = np.array(
        uniform_sample_centers_ns(start1, num_sample1 * ALMA_TE_NS, num_sample1)
    )
    measured = pointing_xds.POINTING_DISH_MEASURED.values
    valid = ~np.isnan(measured).any(axis=-1)
    if merged:
        expected = times0.copy()
        # mean of the two times (no int64 overflow)
        expected[late:] = times0[late:] + (times1 - times0[late:]) // 2
        np.testing.assert_allclose(
            pointing_xds.time_pointing.values,
            to_unix(expected),
            rtol=0,
            atol=TIME_ATOL_S,
        )
        assert valid[:, 0].all()
        np.testing.assert_array_equal(valid[:, 1], np.arange(num_sample0) >= late)
        np.testing.assert_array_equal(measured[:, 0], enc0)
        np.testing.assert_array_equal(measured[late:, 1], enc1)
    else:
        expected = np.sort(np.concatenate([times0, times1]))
        np.testing.assert_allclose(
            pointing_xds.time_pointing.values,
            to_unix(expected),
            rtol=0,
            atol=TIME_ATOL_S,
        )
        assert not (valid[:, 0] & valid[:, 1]).any()


@pytest.mark.parametrize("num_sample0, late", [(200, 3), (201, 3), (200, 4)])
@pytest.mark.parametrize("from_bin", [buggy_from_bin, fixed_from_bin])
def test_create_pointing_xds_binary_antennas_share_sample_times(
    tmp_path, monkeypatch, num_sample0, late, from_bin
):
    """Two antennas sampling the same 48 ms instants, one joining late, in a
    binary Pointing table. With the pyasdm <= 0.0.7 fromBin the corrected row
    starts are off by up to ~0.5 us depending on each row's interval: the
    shared instants must still be single entries of time_pointing with both
    antennas' samples (previously the axis split, ~2x longer, no shared
    time)."""
    num_sample1 = num_sample0 - late
    start1 = T0_NS + late * ALMA_TE_NS
    enc0 = directions(num_sample0, 1.0, 0.8, step=1e-5)
    enc1 = directions(num_sample1, 1.0 + late * 1e-5, 0.8 - late * 0.5e-5, step=1e-5)
    path = write_binary_pointing_asdm(
        [
            pointing_row_xml(0, T0_NS, num_sample0 * ALMA_TE_NS, enc0),
            pointing_row_xml(1, start1, num_sample1 * ALMA_TE_NS, enc1),
        ],
        [False, False],
        tmp_path / "asdm",
    )
    monkeypatch.setattr(
        pyasdm.types.ArrayTimeInterval, "fromBin", staticmethod(from_bin)
    )
    asdm = read_asdm(path)
    assert pointing_module._pointing_times_need_frombin_fix(asdm) == (
        from_bin is buggy_from_bin
    )
    pointing_xds = create_pointing_xds(asdm)

    true_times = uniform_sample_centers_ns(T0_NS, num_sample0 * ALMA_TE_NS, num_sample0)
    assert pointing_xds.sizes["time_pointing"] == num_sample0
    np.testing.assert_allclose(
        pointing_xds.time_pointing.values,
        to_unix(true_times),
        rtol=0,
        atol=TIME_ATOL_S,
    )
    measured = pointing_xds.POINTING_DISH_MEASURED.values
    valid = ~np.isnan(measured).any(axis=-1)
    assert (valid[:, 0] & valid[:, 1]).sum() == num_sample1
    np.testing.assert_array_equal(measured[:, 0], enc0)
    np.testing.assert_array_equal(measured[late:, 1], enc1)
    assert np.isnan(measured[:late, 1]).all()


# ---------------------------------------------------------------------------
# sampledTimeInterval and polynomial rows (F77)
# ---------------------------------------------------------------------------


def test_create_pointing_xds_sampled_time_interval():
    """Irregular sampling: time_pointing comes from sampledTimeInterval (F77)."""
    duration = 24_000_000_000
    sample_duration = 48_000_000
    offsets_ns = [0, 1_000_000_000, 2_000_000_000, 20_000_000_000]
    sampled = [(T0_NS + offset, sample_duration) for offset in offsets_ns]
    encoder = directions(4, 2.0, 1.0)
    asdm = make_pointing_asdm(
        [pointing_row_xml(0, T0_NS, duration, encoder, sampled_intervals=sampled)],
        num_antenna=1,
    )
    pointing_xds = create_pointing_xds(asdm)
    expected = [start + sample_duration // 2 for start, _ in sampled]
    np.testing.assert_allclose(
        pointing_xds.time_pointing.values, to_unix(expected), rtol=0, atol=TIME_ATOL_S
    )
    np.testing.assert_allclose(
        np.diff(pointing_xds.time_pointing.values), [1.0, 1.0, 18.0], atol=1e-6
    )
    np.testing.assert_array_equal(pointing_xds.POINTING_DISH_MEASURED[:, 0], encoder)


def test_create_pointing_xds_single_term_directions_are_constant():
    """target/offset/pointingDirection given as a single term (numTerm=1) are
    constant over the row's samples (F77)."""
    encoder = directions(4, 1.0, 0.5)
    target = np.array([[1.0, 0.5]])
    offset = np.array([[2e-3, -1e-3]])
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(
                0,
                T0_NS,
                4_000_000_000,
                encoder,
                target,
                offset,
                pointing_direction=target,
                use_polynomials=True,
            )
        ],
        num_antenna=1,
    )
    pointing_xds = create_pointing_xds(asdm)
    expected = skyoffset_reference(target, offset)[0] + encoder - target[0]
    assert_directions_close(pointing_xds.POINTING_BEAM.values[:, 0], expected)


def test_create_pointing_xds_polynomial_rows_not_supported():
    """Polynomial expansions with several terms raise a clear error (F77)."""
    encoder = directions(4, 1.0, 0.5)
    coefficients = np.array([[1.0, 0.5], [1e-4, 2e-5], [0.0, 1e-7]])
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(
                0,
                T0_NS,
                4_000_000_000,
                encoder,
                coefficients,
                np.zeros_like(coefficients),
                coefficients,
                use_polynomials=True,
            )
        ],
        num_antenna=1,
    )
    with pytest.raises(NotImplementedError, match="polynomial"):
        create_pointing_xds(asdm)


POLY_ENCODER = directions(4, 1.0, 0.5)
# 4 polynomial coefficients per axis (numTerm == numSample == 4), not samples
POLY_COEFFICIENTS = np.array([[1.0, 0.5], [1e-4, 2e-5], [0.0, 1e-7], [0.0, 0.0]])


def polynomial_rows_xml(target_ant0) -> list[str]:
    """Antenna 0: target given by target_ant0, zero offset and
    pointingDirection (so POINTING_BEAM = target + encoder); antenna 1:
    ordinary sampled row."""
    target_ant0 = np.asarray(target_ant0, dtype=float)
    zeros = np.zeros_like(target_ant0)
    return [
        pointing_row_xml(
            0, T0_NS, 4_000_000_000, POLY_ENCODER, target_ant0, zeros, zeros
        ),
        pointing_row_xml(1, T0_NS, 4_000_000_000, POLY_ENCODER),
    ]


@pytest.mark.parametrize(
    "target_ant0, use_polynomials, expected",
    [
        # coefficients with numTerm == numSample: polynomial, not samples
        (POLY_COEFFICIENTS, True, NotImplementedError),
        # same values flagged as samples
        (POLY_COEFFICIENTS, False, POLY_COEFFICIENTS),
        # single term: constant over the row
        (POLY_COEFFICIENTS[:1], True, np.repeat(POLY_COEFFICIENTS[:1], 4, axis=0)),
        # 3 values for 4 samples, not flagged as polynomial: inconsistent
        (POLY_COEFFICIENTS[:3], False, PointingConversionError),
    ],
)
def test_create_pointing_xds_binary_table_use_polynomials(
    tmp_path, monkeypatch, target_ant0, use_polynomials, expected
):
    """In binary Pointing tables (usual for ALMA) usePolynomials is read
    correctly (readBool) and tells coefficients from samples, also when
    numTerm == numSample (F77)."""
    path = write_binary_pointing_asdm(
        polynomial_rows_xml(target_ant0), [use_polynomials, False], tmp_path / "asdm"
    )
    asdm = read_asdm(path)
    assert [row.getUsePolynomials() for row in asdm.getPointing().get()] == [
        use_polynomials,
        False,
    ]
    # whatever the pyasdm XML parser does, the binary flag is used
    monkeypatch.setattr(
        pointing_module, "_pyasdm_xml_reads_use_polynomials", lambda: False
    )
    assert pointing_module._use_polynomials_flag_reliable(asdm)
    if expected is NotImplementedError:
        with pytest.raises(NotImplementedError, match="polynomial"):
            create_pointing_xds(asdm)
    elif expected is PointingConversionError:
        with pytest.raises(PointingConversionError, match="usePolynomials is false"):
            create_pointing_xds(asdm)
    else:
        pointing_xds = create_pointing_xds(asdm)
        assert pointing_xds.sizes["time_pointing"] == 4
        beam = pointing_xds.POINTING_BEAM.values
        # antenna 0: target (samples, or the constant term) + encoder
        np.testing.assert_allclose(
            beam[:, 0], expected + POLY_ENCODER, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(beam[:, 1], POLY_ENCODER, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(
            pointing_xds.POINTING_DISH_MEASURED.values[:, 0], POLY_ENCODER
        )


@pytest.mark.parametrize("xml_flag_reliable", [True, False])
def test_create_pointing_xds_in_memory_use_polynomials(monkeypatch, xml_flag_reliable):
    """Rows not read from a binary table: usePolynomials is used only when the
    installed pyasdm parses XML booleans correctly. Otherwise only the number
    of values is used, and coefficients with numTerm == numSample cannot be
    told from samples (documented limitation) (F77)."""
    asdm = make_pointing_asdm(polynomial_rows_xml(POLY_COEFFICIENTS))
    rows = asdm.getPointing().get()
    rows[0].setUsePolynomials(True)
    rows[1].setUsePolynomials(False)
    monkeypatch.setattr(
        pointing_module, "_pyasdm_xml_reads_use_polynomials", lambda: xml_flag_reliable
    )
    assert pointing_module._use_polynomials_flag_reliable(asdm) == xml_flag_reliable
    if xml_flag_reliable:
        with pytest.raises(NotImplementedError, match="usePolynomials=True"):
            create_pointing_xds(asdm)
    else:
        pointing_xds = create_pointing_xds(asdm)
        assert pointing_xds.sizes["time_pointing"] == 4


@pytest.mark.parametrize("xml_flag_reliable", [True, False])
def test_create_pointing_xds_in_memory_inconsistent_row(monkeypatch, xml_flag_reliable):
    """3 target values for numSample=4 in a row flagged usePolynomials=false.
    With a reliable flag it is an inconsistent row (PointingConversionError).
    Otherwise it cannot be told from a polynomial (NotImplementedError), and
    the message does not show the flag, which pyasdm may have misparsed (it
    reads "false" as True)."""
    asdm = make_pointing_asdm(polynomial_rows_xml(POLY_COEFFICIENTS[:3]))
    rows = asdm.getPointing().get()
    rows[0].setUsePolynomials(False)
    monkeypatch.setattr(
        pointing_module, "_pyasdm_xml_reads_use_polynomials", lambda: xml_flag_reliable
    )
    if xml_flag_reliable:
        with pytest.raises(PointingConversionError, match="usePolynomials is false"):
            create_pointing_xds(asdm)
    else:
        rows[0].setUsePolynomials(True)  # as pyasdm <= 0.0.7 reads "false"
        with pytest.raises(NotImplementedError) as exc:
            create_pointing_xds(asdm)
        message = str(exc.value)
        assert "either a polynomial expansion" in message
        assert "or an inconsistent row" in message
        assert "usePolynomials=" not in message
    assert issubclass(PointingConversionError, ValueError)
    assert pointing_module.POINTING_CONVERSION_ERRORS == (
        PointingConversionError,
        NotImplementedError,
    )


def xml_bool_parsing_set_from_xml(correct: bool):
    """PointingRow.setFromXML parsing usePolynomials correctly, or with
    bool(text) like pyasdm <= 0.0.7 ("false" -> True)."""
    original = pyasdm.PointingRow.setFromXML

    def set_from_xml(self, xmlrow):
        original(self, xmlrow)
        text = re.search(r"<usePolynomials>\s*(\S+)\s*</usePolynomials>", xmlrow)
        value = text.group(1)
        self._usePolynomials = value.lower() == "true" if correct else bool(value)

    return set_from_xml


@pytest.mark.parametrize("correct", [True, False])
def test_pyasdm_xml_use_polynomials_probe(monkeypatch, correct):
    monkeypatch.setattr(
        pyasdm.PointingRow, "setFromXML", xml_bool_parsing_set_from_xml(correct)
    )
    assert pointing_module._pyasdm_xml_reads_use_polynomials() == correct
    # in-memory ASDM: reliable exactly when the XML parsing is
    asdm = make_pointing_asdm([pointing_row_xml(0, T0_NS, 10**9, POLY_ENCODER)])
    assert pointing_module._use_polynomials_flag_reliable(asdm) == correct


def test_create_pointing_xds_inconsistent_encoder():
    encoder = directions(4, 1.0, 0.5)
    row_xml = pointing_row_xml(0, T0_NS, 4_000_000_000, encoder).replace(
        "<numSample> 4 </numSample>", "<numSample> 5 </numSample>"
    )
    asdm = make_pointing_asdm([row_xml], num_antenna=1)
    with pytest.raises(PointingConversionError, match="numSample"):
        create_pointing_xds(asdm)


def test_create_pointing_xds_unknown_antenna():
    row_xml = pointing_row_xml(2, T0_NS, 4_000_000_000, directions(4, 1.0, 0.5))
    asdm = make_pointing_asdm([row_xml], num_antenna=1)
    with pytest.raises(PointingConversionError, match=r"antennaId\(s\) \[2\]"):
        create_pointing_xds(asdm)


# ---------------------------------------------------------------------------
# time_range / antenna selection and caching (F40, K11)
# ---------------------------------------------------------------------------


def test_create_pointing_xds_time_range(asdm_with_antenna_station_pointing):
    asdm = asdm_with_antenna_station_pointing
    full = create_pointing_xds(asdm)
    times = full.time_pointing.values

    # inclusive boundaries, exactly on samples
    part = create_pointing_xds(asdm, time_range=(times[2], times[5]))
    np.testing.assert_array_equal(part.time_pointing.values, times[2:6])
    np.testing.assert_array_equal(
        part.POINTING_BEAM.values, full.POINTING_BEAM.values[2:6]
    )
    assert part.time_pointing.attrs == full.time_pointing.attrs

    # range covering only the second row
    second_row_start = to_unix(FIXTURE_ROW_STARTS_NS[1])
    part = create_pointing_xds(asdm, time_range=(second_row_start, times[-1] + 100.0))
    np.testing.assert_array_equal(part.time_pointing.values, times[4:])

    # between two samples, before and after the table: nothing selected
    assert (
        create_pointing_xds(asdm, time_range=(times[0] + 0.1, times[1] - 0.1)) is None
    )
    assert create_pointing_xds(asdm, time_range=(1.0e9, 1.0e9 + 10.0)) is None
    assert create_pointing_xds(asdm, time_range=(times[-1] + 1, times[-1] + 2)) is None


def test_create_pointing_xds_antenna_names(asdm_with_antenna_station_pointing):
    asdm = asdm_with_antenna_station_pointing
    full = create_pointing_xds(asdm)
    part = create_pointing_xds(asdm, antenna_names=["CM03", "XX99"])
    assert list(part.antenna_name.values) == ["CM03"]
    np.testing.assert_array_equal(
        part.POINTING_BEAM.values, full.POINTING_BEAM.sel(antenna_name=["CM03"]).values
    )
    assert create_pointing_xds(asdm, antenna_names=["XX99"]) is None


def test_create_pointing_xds_time_range_drops_antennas_without_samples():
    duration = 2_000_000_000
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(0, T0_NS, duration, directions(2, 1.0, 0.5)),
            pointing_row_xml(
                1, T0_NS + 10 * duration, duration, directions(2, 2.0, 0.5)
            ),
        ]
    )
    part = create_pointing_xds(asdm, time_range=(to_unix(T0_NS), to_unix(T0_NS) + 2.0))
    assert list(part.antenna_name.values) == ["DA41"]
    assert part.sizes["time_pointing"] == 2
    assert not np.isnan(part.POINTING_BEAM.values).any()


def test_create_pointing_xds_converts_table_once(monkeypatch):
    """The whole-table conversion is cached per ASDM object; every call returns
    an independent copy; the cache is refreshed when rows are added (F40)."""
    duration = 2_000_000_000
    asdm = make_pointing_asdm(
        [
            pointing_row_xml(ant, T0_NS, duration, directions(4, 1.0, 0.5))
            for ant in (0, 1)
        ]
    )
    builds = []
    original_build = pointing_module._build_full_pointing_xds

    def counting_build(*args, **kwargs):
        builds.append(args[0])
        return original_build(*args, **kwargs)

    monkeypatch.setattr(pointing_module, "_build_full_pointing_xds", counting_build)

    first = create_pointing_xds(asdm)
    times = first.time_pointing.values
    second = create_pointing_xds(asdm, time_range=(times[1], times[2]))
    third = create_pointing_xds(asdm)
    assert len(builds) == 1
    assert second.sizes["time_pointing"] == 2
    assert not np.shares_memory(first.POINTING_BEAM.values, third.POINTING_BEAM.values)
    first.POINTING_BEAM.values[:] = 0.0
    np.testing.assert_array_equal(
        create_pointing_xds(asdm).POINTING_BEAM.values, third.POINTING_BEAM.values
    )
    assert len(builds) == 1

    # another ASDM object is converted separately
    other = make_pointing_asdm(
        [pointing_row_xml(0, T0_NS, duration, directions(4, 3.0, 0.2))]
    )
    assert list(create_pointing_xds(other).antenna_name.values) == ["DA41"]
    assert len(builds) == 2

    # new rows invalidate the cached conversion
    add_pointing_rows(
        asdm,
        [
            pointing_row_xml(ant, T0_NS + duration, duration, directions(4, 1.2, 0.4))
            for ant in (0, 1)
        ],
    )
    assert create_pointing_xds(asdm).sizes["time_pointing"] == 8
    assert len(builds) == 3


def test_create_pointing_xds_caches_conversion_errors(monkeypatch, caplog):
    """A Pointing table that cannot be converted is converted (and the error
    logged) only once per ASDM object: later calls (one per partition) raise
    the same error again, without traceback from the earlier calls. Adding
    rows converts the table again."""
    duration = 2_000_000_000
    bad_row = pointing_row_xml(1, T0_NS, duration, directions(4, 1.0, 0.5)).replace(
        "<numSample> 4 </numSample>", "<numSample> 5 </numSample>"
    )
    asdm = make_pointing_asdm(
        [pointing_row_xml(0, T0_NS, duration, directions(4, 1.0, 0.5)), bad_row]
    )
    builds = []
    original_build = pointing_module._build_full_pointing_xds

    def counting_build(*args, **kwargs):
        builds.append(args[0])
        return original_build(*args, **kwargs)

    monkeypatch.setattr(pointing_module, "_build_full_pointing_xds", counting_build)

    errors = []
    with caplog.at_level(logging.ERROR):
        for time_range in (None, (0.0, 2e9), None):
            with pytest.raises(PointingConversionError, match="numSample=5") as exc:
                create_pointing_xds(asdm, time_range=time_range)
            errors.append(exc.value)
    assert len(builds) == 1
    logged = [rec for rec in caplog.records if "cannot be converted" in rec.message]
    assert len(logged) == 1
    assert logged[0].levelno == logging.ERROR
    assert "PointingConversionError" in logged[0].message
    assert "numSample=5" in logged[0].message
    # new exception objects with the same message; the cached ones carry no
    # traceback (which would keep the conversion data alive)
    assert len({id(error) for error in errors}) == 3
    assert len({str(error) for error in errors}) == 1
    cached = pointing_module._pointing_cache[asdm]
    assert cached.xds is None
    assert cached.error.__traceback__ is None
    assert type(cached.error) is PointingConversionError

    add_pointing_rows(
        asdm,
        [pointing_row_xml(0, T0_NS + duration, duration, directions(4, 1.2, 0.4))],
    )
    with pytest.raises(PointingConversionError):
        create_pointing_xds(asdm)
    assert len(builds) == 2


def test_create_pointing_xds_does_not_use_deep_copying_getters(monkeypatch):
    """The pyasdm getters of the direction columns deep copy lists of Angle
    objects (~96% of the conversion time of big tables, F40)."""
    asdm = make_pointing_asdm(
        [pointing_row_xml(0, T0_NS, 2_000_000_000, directions(4, 1.0, 0.5))],
        num_antenna=1,
    )

    def fail(*_args):
        raise AssertionError("deep-copying getter used")

    for getter in ("getEncoder", "getTarget", "getOffset", "getPointingDirection"):
        monkeypatch.setattr(pyasdm.PointingRow, getter, fail)
    pointing_xds = create_pointing_xds(asdm)
    np.testing.assert_array_equal(
        pointing_xds.POINTING_DISH_MEASURED.values[:, 0], directions(4, 1.0, 0.5)
    )


# ---------------------------------------------------------------------------
# Binary Pointing tables and the pyasdm ArrayTimeInterval.fromBin bug (F36)
# ---------------------------------------------------------------------------


def test_pyasdm_frombin_probe(monkeypatch):
    monkeypatch.setattr(
        pyasdm.types.ArrayTimeInterval, "fromBin", staticmethod(buggy_from_bin)
    )
    assert pointing_module._pyasdm_frombin_halves_start()
    monkeypatch.setattr(
        pyasdm.types.ArrayTimeInterval, "fromBin", staticmethod(fixed_from_bin)
    )
    assert not pointing_module._pyasdm_frombin_halves_start()


@pytest.mark.parametrize(
    "start_ns, duration_ns",
    [
        (5_137_194_858_648_000_000, 24_240_000_000),
        (5_181_249_600_000_000_123, 48_000_001),
        (4_953_045_089_210_000_000, 1_008_000_000),
    ],
)
def test_undo_frombin_start_halving(start_ns, duration_ns):
    midpoint = start_ns + duration_ns // 2
    halved_start = int((midpoint - duration_ns) / 2)
    recovered = pointing_module._undo_frombin_start_halving(halved_start, duration_ns)
    assert abs(recovered - start_ns) <= 1024


def test_pointing_table_read_from_binary(tmp_path, asdm_with_antenna_station_pointing):
    class FakeASDM:
        def __init__(self, directory):
            self._directory = directory

        def getDirectory(self):
            return self._directory

    assert not pointing_module._pointing_table_read_from_binary(
        asdm_with_antenna_station_pointing
    )
    assert not pointing_module._pointing_table_read_from_binary(FakeASDM(None))

    bin_header = tmp_path / "bin_header"
    bin_header.mkdir()
    (bin_header / "Pointing.xml").write_text(
        '<?xml version="1.0"?><PointingTable><BulkStoreRef file_id="X" '
        'byteOrder="Big_Endian"/></PointingTable>'
    )
    assert pointing_module._pointing_table_read_from_binary(FakeASDM(str(bin_header)))

    xml_table = tmp_path / "xml_table"
    xml_table.mkdir()
    (xml_table / "Pointing.xml").write_text(
        "<PointingTable><row></row></PointingTable>"
    )
    (xml_table / "Pointing.bin").write_bytes(b"")
    assert not pointing_module._pointing_table_read_from_binary(
        FakeASDM(str(xml_table))
    )

    bin_only = tmp_path / "bin_only"
    bin_only.mkdir()
    (bin_only / "Pointing.bin").write_bytes(b"")
    assert pointing_module._pointing_table_read_from_binary(FakeASDM(str(bin_only)))


@pytest.fixture(scope="module")
def synth_pointing_bin(synthetic_asdm_module, make_synthetic_asdm):
    """Synthetic ASDM whose Pointing table is written as a binary table."""
    spec = synthetic_asdm_module.interferometric_spec(
        with_pointing=True, name="uid___A002_X1234_X9b1n"
    )
    spec.pointing_as_bin = True
    return make_synthetic_asdm(spec)


@pytest.mark.parametrize("from_bin", [None, buggy_from_bin, fixed_from_bin])
def test_create_pointing_xds_binary_pointing_table(
    synth_pointing_bin, from_bin, monkeypatch
):
    """Times of a binary Pointing table are right with the installed pyasdm,
    with a pyasdm whose fromBin halves the start times (corrected), and with a
    fixed pyasdm (not corrected) (F36)."""
    truth = synth_pointing_bin
    pointing_header = Path(truth.path, "Pointing.xml").read_text()
    assert "<BulkStoreRef" in pointing_header, "the Pointing table should be binary"
    if from_bin is not None:
        monkeypatch.setattr(
            pyasdm.types.ArrayTimeInterval, "fromBin", staticmethod(from_bin)
        )
    asdm = read_asdm(truth.path)
    if from_bin is buggy_from_bin:
        assert pointing_module._pointing_times_need_frombin_fix(asdm)
    elif from_bin is fixed_from_bin:
        assert not pointing_module._pointing_times_need_frombin_fix(asdm)
    pointing_xds = create_pointing_xds(asdm)
    np.testing.assert_allclose(
        pointing_xds.time_pointing.values,
        truth.pointing_times_unix,
        rtol=0,
        atol=TIME_ATOL_S,
    )
    assert list(map(str, pointing_xds.antenna_name.values)) == truth.antenna_names
    np.testing.assert_allclose(
        pointing_xds.POINTING_BEAM.values, truth.pointing_target, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        pointing_xds.POINTING_DISH_MEASURED.values,
        truth.pointing_encoder,
        rtol=0,
        atol=1e-12,
    )


def test_create_pointing_xds_xml_pointing_table_not_corrected(
    synth_interferometric_pointing, monkeypatch
):
    """An XML Pointing table is never corrected, even with a buggy fromBin."""
    monkeypatch.setattr(
        pyasdm.types.ArrayTimeInterval, "fromBin", staticmethod(buggy_from_bin)
    )
    truth = synth_interferometric_pointing
    pointing_xds = create_pointing_xds(read_asdm(truth.path))
    np.testing.assert_allclose(
        pointing_xds.time_pointing.values,
        truth.pointing_times_unix,
        rtol=0,
        atol=TIME_ATOL_S,
    )


def test_create_pointing_xds_binary_pointing_table_uncorrected_bug(
    synth_pointing_bin, monkeypatch
):
    """Without the correction, the buggy fromBin puts the binary Pointing
    times ~half the ASDM epoch earlier (decades off): the correction matters."""
    monkeypatch.setattr(
        pyasdm.types.ArrayTimeInterval, "fromBin", staticmethod(buggy_from_bin)
    )
    monkeypatch.setattr(
        pointing_module, "_pointing_times_need_frombin_fix", lambda asdm: False
    )
    truth = synth_pointing_bin
    pointing_xds = create_pointing_xds(read_asdm(truth.path))
    years_off = (truth.pointing_times_unix - pointing_xds.time_pointing.values) / (
        365.25 * 86400
    )
    assert np.all(years_off > 50)
