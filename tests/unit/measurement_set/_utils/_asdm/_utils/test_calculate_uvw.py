import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time

from xradio.measurement_set._utils._asdm._utils import calculate_uvw as uvw_module
from xradio.measurement_set._utils._asdm._utils.calculate_uvw import (
    calculate_uvw,
    check_uvw_inputs,
    phase_center_frame_to_astropy,
)
from xradio.measurement_set._utils._asdm._utils.field_source import (
    ASDM_DIRECTION_CODE_TO_FRAME,
)

ALMA_CENTER_ITRF = np.array([2225142.180268967, -5440307.370348562, -2481029.851873547])
VLA_CENTER_ITRF = np.array([-1601185.4, -5041977.5, 3554875.9])

# ALMA-like array with ~16 km baselines, used for the casacore reference below
CASACORE_POSITIONS = np.array(
    [
        ALMA_CENTER_ITRF,
        ALMA_CENTER_ITRF + np.array([9000.0, 12000.0, -5000.0]),
        ALMA_CENTER_ITRF + np.array([-3000.0, 8000.0, 1000.0]),
    ]
)
CASACORE_TIMES_UTC = [
    "2024-03-01T05:59:23",
    "2024-03-01T09:00:00",
    "2024-07-21T04:26:40",
]
CASACORE_BASELINES = [("A", "B"), ("A", "C"), ("B", "C")]
# UVW (POSITION1 - POSITION2) of CASACORE_BASELINES at CASACORE_TIMES_UTC for a J2000
# phase center, computed with python-casacore 3.8.1 (measures.to_uvw(as_baseline(...)),
# frame: ITRF positions + UTC epoch + J2000 direction), i.e. the CASA/importasdm
# convention. Independent reference, not a snapshot of xradio's implementation.
CASACORE_UVW_J2000 = {
    (1.0, -0.5): np.array(
        [
            [
                [13574.7185, 7456.9077, 3181.4378],
                [7221.0837, -3051.8034, -3541.5316],
                [-6353.6348, -10508.7111, -6722.9694],
            ],
            [
                [14068.7868, 1938.9342, -6950.5231],
                [1834.803, -4866.9371, -6851.7458],
                [-12233.9838, -6805.8713, 98.7773],
            ],
            [
                [-435.4039, -2779.966, -15558.9914],
                [-7325.5271, -2992.4447, -3373.7112],
                [-6890.1233, -212.4787, 12185.2802],
            ],
        ]
    ),
    (2.0, -1.2): np.array(
        [
            [
                [1976.229, 15671.4331, 707.605],
                [7747.8702, 3006.5611, 2220.6075],
                [5771.6412, -12664.8719, 1513.0025],
            ],
            [
                [11951.522, 10254.8073, -1414.231],
                [8014.6401, -3121.1318, -155.1782],
                [-3936.8819, -13375.9391, 1259.0529],
            ],
            [
                [12375.8746, -6069.5509, -7745.8556],
                [-259.3195, -8324.2309, -2154.0504],
                [-12635.1942, -2254.68, 5591.8052],
            ],
        ]
    ),
}


def make_time(values, units="s", time_format="unix", scale="utc") -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims="time",
        attrs={"type": "time", "units": units, "format": time_format, "scale": scale},
    )


def make_antenna_position(positions, names) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(positions, dtype=float),
        dims=["antenna_name", "cartesian_pos_label"],
        coords={"antenna_name": names, "cartesian_pos_label": ["x", "y", "z"]},
        attrs={"type": "location", "units": "m", "frame": "ITRS"},
    )


def make_baseline_names(pairs) -> tuple[xr.DataArray, xr.DataArray]:
    return (
        xr.DataArray([pair[0] for pair in pairs], dims="baseline_id"),
        xr.DataArray([pair[1] for pair in pairs], dims="baseline_id"),
    )


def make_field_direction(ra, dec, frame="icrs") -> xr.DataArray:
    return xr.DataArray(
        [[ra, dec]],
        dims=["field_name", "sky_dir_label"],
        coords={"field_name": ["field_0"], "sky_dir_label": ["ra", "dec"]},
        attrs={"type": "sky_coord", "units": "rad", "frame": frame},
    )


def make_time_direction(ra_dec, frame="icrs") -> xr.DataArray:
    return xr.DataArray(
        np.asarray(ra_dec, dtype=float),
        dims=["time", "sky_dir_label"],
        coords={"sky_dir_label": ["ra", "dec"]},
        attrs={"type": "sky_coord", "units": "rad", "frame": frame},
    )


def apparent_direction_itrs(unix_utc, ra_dec) -> np.ndarray:
    """(time, 3) unit vectors towards (ra, dec) [ICRS] in ITRS (geocentric,
    apparent: incl. annual aberration), independent reference for w."""
    ra_dec = np.broadcast_to(np.asarray(ra_dec, dtype=float), (len(unix_utc), 2))
    obstime = Time(unix_utc, format="unix", scale="utc")
    sky = SkyCoord(ra_dec[:, 0] * u.rad, ra_dec[:, 1] * u.rad, frame="icrs")
    xyz = sky.transform_to(ITRS(obstime=obstime)).cartesian.xyz.value.T
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


@pytest.fixture
def small_array():
    names = ["DA41", "DA42", "DV01", "PM03"]
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [312.25, 141.5, -96.75],
            [-205.5, 498.0, 251.25],
            [790, -310.5, 402],
        ]
    )
    positions = ALMA_CENTER_ITRF + offsets
    pairs = [("DA42", "DA41"), ("DV01", "DA41"), ("DV01", "DA42"), ("PM03", "DV01")]
    pairs += [(name, name) for name in names]
    unix_times = Time("2024-03-01T05:59:23", scale="utc").unix + np.arange(6) * 30.0
    return names, positions, pairs, unix_times


@pytest.mark.parametrize("ra_dec", [(1.0, -0.5), (2.0, -1.2)])
def test_calculate_uvw_matches_casacore(ra_dec):
    """UVW agree with casacore's J2000 UVW (CASA/importasdm convention) within 8 cm
    on ~16 km baselines (F64: the PR's apparent-frame u,v basis was 0.2-3 m off)."""
    times = Time(CASACORE_TIMES_UTC, scale="utc")
    antenna1, antenna2 = make_baseline_names(CASACORE_BASELINES)
    uvw = calculate_uvw(
        None,
        make_time(times.unix),
        antenna1,
        antenna2,
        make_antenna_position(CASACORE_POSITIONS, ["A", "B", "C"]),
        make_field_direction(*ra_dec, frame="fk5"),
    )
    assert uvw.shape == (3, 3, 3)
    assert uvw.dtype == np.float64
    np.testing.assert_allclose(uvw, CASACORE_UVW_J2000[ra_dec], rtol=0, atol=0.08)


@pytest.mark.parametrize(
    "center", [ALMA_CENTER_ITRF, VLA_CENTER_ITRF], ids=["alma", "vla"]
)
@pytest.mark.parametrize("ra_dec", [(0.3, 0.4), (4.0, -1.1)])
def test_calculate_uvw_length_and_w_independent_reference(center, ra_dec):
    """|UVW| equals the baseline length and w = (P1 - P2).s_apparent(t) (within
    1 mm on a 33 km baseline), for an ALMA and a VLA site: no hard-coded
    observatory (F63), times decoded with their own attrs (F23)."""
    names = ["A", "B", "C"]
    positions = center + np.array(
        [[0.0, 0.0, 0.0], [30000.0, -12000.0, 6000.0], [-500.0, 250.0, 125.0]]
    )
    pairs = [("A", "B"), ("C", "A"), ("B", "C"), ("B", "B")]
    unix_times = (
        Time("2025-01-10T00:00:00", scale="utc").unix + np.linspace(0, 6, 5) * 3600
    )
    antenna1, antenna2 = make_baseline_names(pairs)
    uvw = calculate_uvw(
        None,
        make_time(unix_times),
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
        make_field_direction(*ra_dec),
    )
    index = {name: idx for idx, name in enumerate(names)}
    baselines = np.array([positions[index[a]] - positions[index[b]] for a, b in pairs])
    assert uvw.shape == (len(unix_times), len(pairs), 3)
    np.testing.assert_allclose(
        np.linalg.norm(uvw, axis=-1),
        np.broadcast_to(np.linalg.norm(baselines, axis=-1), uvw.shape[:2]),
        rtol=0,
        atol=1e-6,
    )
    w_expected = apparent_direction_itrs(unix_times, ra_dec) @ baselines.T
    np.testing.assert_allclose(uvw[..., 2], w_expected, rtol=0, atol=1e-3)
    # auto-correlation
    np.testing.assert_array_equal(uvw[:, 3], 0.0)


def test_calculate_uvw_antisymmetric_and_v_points_north(small_array):
    """Swapping the antennas negates UVW; for a baseline along the celestial pole
    direction, u ~ 0 and v > 0 (u east, v north)."""
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs[:4])
    swapped1, swapped2 = make_baseline_names([(b, a) for a, b in pairs[:4]])
    args = (
        make_time(unix_times),
        make_antenna_position(positions, names),
        make_field_direction(1.0, -0.5),
    )
    uvw = calculate_uvw(None, args[0], antenna1, antenna2, *args[1:])
    uvw_swapped = calculate_uvw(None, args[0], swapped1, swapped2, *args[1:])
    np.testing.assert_allclose(uvw_swapped, -uvw, rtol=0, atol=1e-9)

    # Baseline along the Earth's rotation axis (ITRS z ~ celestial pole), source
    # at the equator: the baseline is perpendicular to the source and along v.
    polar = make_antenna_position(
        [ALMA_CENTER_ITRF, ALMA_CENTER_ITRF + [0.0, 0.0, 1000.0]], ["ref", "north"]
    )
    north1, north2 = make_baseline_names([("north", "ref")])
    uvw_polar = calculate_uvw(
        None, args[0], north1, north2, polar, make_field_direction(2.0, 0.0)
    )
    np.testing.assert_allclose(uvw_polar[..., 1], 1000.0, rtol=0, atol=0.5)
    np.testing.assert_allclose(uvw_polar[..., [0, 2]], 0.0, rtol=0, atol=10.0)


@pytest.mark.parametrize(
    "values_from_time, units, time_format, scale",
    [
        (lambda t: t.unix, "s", "unix", "utc"),
        (lambda t: t.utc.mjd, "d", "mjd", "utc"),
        (lambda t: t.utc.mjd * 86400.0, "s", "mjd", "utc"),
        (lambda t: t.tai.unix_tai, "s", "unix_tai", "tai"),
        (lambda t: t.tai.mjd, "d", "mjd", "tai"),
    ],
)
def test_calculate_uvw_time_from_attrs(
    small_array, values_from_time, units, time_format, scale
):
    """The time values are interpreted with their own units/format/scale attrs
    (F23, K1): the same instants in different representations give the same UVW."""
    names, positions, pairs, unix_times = small_array
    instants = Time(unix_times, format="unix", scale="utc")
    antenna1, antenna2 = make_baseline_names(pairs)
    common = (
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
        make_field_direction(1.0, -0.5),
    )
    reference = calculate_uvw(None, make_time(unix_times), *common)
    uvw = calculate_uvw(
        None,
        make_time(values_from_time(instants), units, time_format, scale),
        *common,
    )
    np.testing.assert_allclose(uvw, reference, rtol=0, atol=1e-5)


@pytest.mark.filterwarnings("ignore")  # the mislabelled instants are in year ~2134
def test_calculate_uvw_time_label_matters(small_array):
    """MJD-epoch seconds labelled 'unix' are a different instant (the PR's
    format='mjd' hard-coding made both give the same, wrong-for-unix, UVW)."""
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs[:4])
    common = (
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
        make_field_direction(1.0, -0.5),
    )
    mjd_seconds = Time(unix_times, format="unix", scale="utc").mjd * 86400.0
    as_mjd = calculate_uvw(None, make_time(mjd_seconds, "s", "mjd"), *common)
    as_unix = calculate_uvw(None, make_time(unix_times), *common)
    np.testing.assert_allclose(as_mjd, as_unix, rtol=0, atol=1e-5)
    mislabelled = calculate_uvw(None, make_time(mjd_seconds, "s", "unix"), *common)
    assert np.abs(mislabelled - as_unix).max() > 10.0


def test_calculate_uvw_per_time_phase_center(small_array):
    """A (time, sky_dir_label) phase center gives, for every time, the UVW of the
    direction of that time (K10)."""
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs)
    antenna_position = make_antenna_position(positions, names)
    ra_dec = np.array(
        [[1.0, -0.5], [1.0, -0.5], [1.01, -0.49], [3.0, 0.2], [3.0, 0.2], [5.0, -1.2]]
    )
    uvw = calculate_uvw(
        None,
        make_time(unix_times),
        antenna1,
        antenna2,
        antenna_position,
        make_time_direction(ra_dec),
    )
    for tidx, (ra, dec) in enumerate(ra_dec):
        expected = calculate_uvw(
            None,
            make_time(unix_times[tidx : tidx + 1]),
            antenna1,
            antenna2,
            antenna_position,
            make_field_direction(ra, dec),
        )
        np.testing.assert_allclose(uvw[tidx : tidx + 1], expected, rtol=0, atol=1e-9)
    assert np.abs(uvw[3] - uvw[2]).max() > 100.0


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), slice(None), slice(None)),
        (slice(2, 5), slice(1, 7), slice(0, 3)),
        (slice(0, 6, 2), slice(None, None, 3), slice(2, 3)),
        (slice(5, 6), slice(4, 5), slice(1, 2)),
        (slice(None, None, -1), slice(6, 0, -2), slice(None, None, -1)),
    ],
)
@pytest.mark.parametrize("per_time", [False, True])
def test_calculate_uvw_key(small_array, key, per_time):
    """A key (one slice per (time, baseline_id, uvw_label)) selects the same
    elements as numpy indexing of the full result (F22: the label key used to be
    applied to the time axis)."""
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs)
    if per_time:
        direction = make_time_direction(
            np.stack([np.linspace(1.0, 1.1, 6), np.linspace(-0.5, -0.4, 6)], axis=-1)
        )
    else:
        direction = make_field_direction(1.0, -0.5)
    args = (
        make_time(unix_times),
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
        direction,
    )
    full = calculate_uvw(None, *args)
    selected = calculate_uvw(key, *args)
    np.testing.assert_allclose(selected, full[key], rtol=0, atol=1e-9)
    assert selected.shape == full[key].shape


@pytest.mark.parametrize(
    "frame, astropy_frame",
    [
        ("fk5", "fk5"),
        ("J2000", "fk5"),
        ("fk4", "fk4"),
        ("B1950", "fk4"),
        ("galactic", "galactic"),
        ("supergalactic", "supergalactic"),
        ("ICRS", "icrs"),
    ],
)
def test_calculate_uvw_frame_attribute(small_array, frame, astropy_frame):
    """The phase center frame attribute is honoured: a direction in a celestial
    frame gives the UVW of the same direction converted to ICRS by astropy (and
    differs from the UVW of the same numbers labelled ICRS)."""
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs[:4])
    common = (
        make_time(unix_times),
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
    )
    lon, lat = 1.0, -0.5
    icrs = SkyCoord(lon * u.rad, lat * u.rad, frame=astropy_frame).icrs
    uvw_frame = calculate_uvw(None, *common, make_field_direction(lon, lat, frame))
    uvw_icrs = calculate_uvw(
        None,
        *common,
        make_field_direction(icrs.ra.to_value(u.rad), icrs.dec.to_value(u.rad), "icrs"),
    )
    np.testing.assert_allclose(uvw_frame, uvw_icrs, rtol=0, atol=1e-6)
    if astropy_frame not in ("icrs", "fk5"):  # FK5 J2000 ~ ICRS (tens of mas)
        uvw_icrs_label = calculate_uvw(
            None, *common, make_field_direction(lon, lat, "icrs")
        )
        assert np.abs(uvw_frame - uvw_icrs_label).max() > 1.0


@pytest.mark.parametrize("frame", ["hadec", "altaz", "itrs", "AZEL", "TOPO", "mars"])
def test_calculate_uvw_unsupported_frames(small_array, frame):
    """Phase centers in Earth-fixed / topocentric (or unknown) frames are
    rejected with NotImplementedError, by calculate_uvw and by check_uvw_inputs
    (the check made when a partition is opened, without calculating)."""
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs)
    args = (
        make_time(unix_times),
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
        make_field_direction(1.0, 0.5, frame),
    )
    match = f"frame '{frame.lower()}' is not supported"
    with pytest.raises(NotImplementedError, match=match):
        phase_center_frame_to_astropy(frame)
    with pytest.raises(NotImplementedError, match=match):
        check_uvw_inputs(*args)
    with pytest.raises(NotImplementedError, match=match):
        calculate_uvw(None, *args)


def test_phase_center_frames_cover_field_source_frames(small_array):
    """Every frame that field_source gives to field directions is either
    supported by the UVW calculation (UVW computed, with |UVW| = baseline length)
    or explicitly unsupported (rejected by check_uvw_inputs, that is when the
    partition is opened): the frame tables of field_source and calculate_uvw are
    consistent (a new frame in field_source must be handled here)."""
    supported = set(uvw_module._ASTROPY_DIRECTION_FRAMES)
    unsupported = set(uvw_module._UNSUPPORTED_DIRECTION_FRAMES)
    assert not supported & unsupported
    field_frames = set(ASDM_DIRECTION_CODE_TO_FRAME.values())
    assert field_frames <= supported | unsupported, field_frames - (
        supported | unsupported
    )
    # the supported frames convert to ICRS without obstime / location
    for frame in supported:
        astropy_frame = phase_center_frame_to_astropy(frame)
        icrs = SkyCoord(1.0 * u.rad, 0.5 * u.rad, frame=astropy_frame).icrs
        assert np.isfinite([icrs.ra.rad, icrs.dec.rad]).all()

    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs)
    common = (
        make_time(unix_times),
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
    )
    index = {name: idx for idx, name in enumerate(names)}
    lengths = np.linalg.norm(
        [positions[index[a]] - positions[index[b]] for a, b in pairs], axis=-1
    )
    for frame in sorted(field_frames):
        direction = make_field_direction(1.0, 0.5, frame)
        if frame in supported:
            check_uvw_inputs(*common, direction)
            uvw = calculate_uvw(None, *common, direction)
            np.testing.assert_allclose(
                np.linalg.norm(uvw, axis=-1),
                np.broadcast_to(lengths, uvw.shape[:2]),
                rtol=0,
                atol=1e-6,
            )
        else:
            with pytest.raises(NotImplementedError, match=frame):
                check_uvw_inputs(*common, direction)


def test_check_uvw_inputs_valid_does_not_calculate(small_array, monkeypatch):
    """check_uvw_inputs accepts valid inputs (one direction per time or per
    field) without calculating any UVW."""
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs)

    def fail(*args, **kwargs):
        raise AssertionError("check_uvw_inputs must not calculate UVW")

    monkeypatch.setattr(uvw_module, "_calculate_uvw_astropy", fail)
    monkeypatch.setattr(uvw_module, "_itrs_to_gcrs_matrices", fail)
    common = (
        make_time(unix_times),
        antenna1,
        antenna2,
        make_antenna_position(positions, names),
    )
    assert check_uvw_inputs(*common, make_field_direction(1.0, -0.5, "fk5")) is None
    per_time = make_time_direction([[1.0, -0.5]] * len(unix_times), "supergalactic")
    assert check_uvw_inputs(*common, per_time) is None
    # no times
    assert (
        check_uvw_inputs(make_time([]), *common[1:], make_field_direction(1.0, -0.5))
        is None
    )


def _bad_uvw_inputs(case, names, positions, pairs, unix_times):
    """(time, antenna1, antenna2, antenna_position, direction) with one problem."""
    antenna1, antenna2 = make_baseline_names(pairs)
    time = make_time(unix_times)
    antenna_position = make_antenna_position(positions, names)
    direction = make_field_direction(1.0, -0.5)
    if case == "unknown_antenna":
        antenna1, antenna2 = make_baseline_names([("DA41", "XX99")])
    elif case == "baseline_names_lengths":
        antenna2 = antenna2[:-1]
    elif case == "no_format":
        del time.attrs["format"]
    elif case == "no_scale":
        del time.attrs["scale"]
    elif case == "bad_format":
        time.attrs["format"] = "not_a_format"
    elif case == "bad_scale":
        time.attrs["scale"] = "not_a_scale"
    elif case == "two_fields":
        direction = xr.DataArray(
            [[1.0, -0.5], [1.1, -0.5]],
            dims=["field_name", "sky_dir_label"],
            coords={"field_name": ["a", "b"], "sky_dir_label": ["ra", "dec"]},
        )
    elif case == "num_times":
        direction = make_time_direction([[1.0, -0.5]] * (len(unix_times) - 1))
    elif case == "direction_dims":
        direction = direction.rename(field_name="row")
    elif case == "labels":
        direction = direction.assign_coords(sky_dir_label=["dec", "ra"])
    elif case == "units":
        direction.attrs["units"] = "m"
    elif case == "positions_shape":
        antenna_position = antenna_position.isel(cartesian_pos_label=slice(0, 2))
    elif case == "frame":
        direction.attrs["frame"] = "hadec"
    return time, antenna1, antenna2, antenna_position, direction


@pytest.mark.parametrize(
    "case, error, match",
    [
        ("unknown_antenna", ValueError, "XX99"),
        ("baseline_names_lengths", ValueError, "one name per baseline"),
        ("no_format", ValueError, "format"),
        ("no_scale", ValueError, "scale"),
        ("bad_format", ValueError, "not_a_format"),
        ("bad_scale", ValueError, "not_a_scale"),
        ("two_fields", ValueError, "exactly one field"),
        ("num_times", ValueError, "times"),
        ("direction_dims", ValueError, "dimension"),
        ("labels", ValueError, "sky_dir_label"),
        ("units", ValueError, "angles"),
        ("positions_shape", ValueError, "cartesian"),
        ("frame", NotImplementedError, "hadec"),
    ],
)
def test_check_uvw_inputs_errors_match_calculate_uvw(small_array, case, error, match):
    """check_uvw_inputs raises the error that calculate_uvw raises later, so an
    input problem is found when the lazy UVW array is created (partition open)."""
    args = _bad_uvw_inputs(case, *small_array)
    with pytest.raises(error, match=match):
        check_uvw_inputs(*args)
    with pytest.raises(error, match=match):
        calculate_uvw(None, *args)


def test_check_uvw_inputs_time_dims(small_array):
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs)
    time = make_time(unix_times).rename(time="row")
    with pytest.raises(ValueError, match="single dimension 'time'"):
        check_uvw_inputs(
            time,
            antenna1,
            antenna2,
            make_antenna_position(positions, names),
            make_field_direction(1.0, -0.5),
        )


def test_calculate_uvw_errors(small_array):
    names, positions, pairs, unix_times = small_array
    antenna1, antenna2 = make_baseline_names(pairs)
    antenna_position = make_antenna_position(positions, names)
    time = make_time(unix_times)
    direction = make_field_direction(1.0, -0.5)

    with pytest.raises(KeyError, match="antenna_name"):
        calculate_uvw(
            None,
            xr.DataArray(),
            xr.DataArray(),
            xr.DataArray(),
            xr.DataArray(),
            xr.DataArray(),
        )

    with pytest.raises(TypeError, match="one slice"):
        calculate_uvw(
            (0, slice(None), slice(None)),
            time,
            antenna1,
            antenna2,
            antenna_position,
            direction,
        )

    unknown1, unknown2 = make_baseline_names([("DA41", "XX99")])
    with pytest.raises(ValueError, match="XX99"):
        calculate_uvw(None, time, unknown1, unknown2, antenna_position, direction)

    no_format = time.copy()
    del no_format.attrs["format"]
    with pytest.raises(ValueError, match="format"):
        calculate_uvw(None, no_format, antenna1, antenna2, antenna_position, direction)

    two_fields = xr.DataArray(
        [[1.0, -0.5], [1.1, -0.5]],
        dims=["field_name", "sky_dir_label"],
        coords={"field_name": ["a", "b"], "sky_dir_label": ["ra", "dec"]},
    )
    with pytest.raises(ValueError, match="exactly one field"):
        calculate_uvw(None, time, antenna1, antenna2, antenna_position, two_fields)

    wrong_num_times = make_time_direction([[1.0, -0.5]] * 3)
    with pytest.raises(ValueError, match="times"):
        calculate_uvw(None, time, antenna1, antenna2, antenna_position, wrong_num_times)

    with pytest.raises(NotImplementedError, match="azel"):
        calculate_uvw(
            None,
            time,
            antenna1,
            antenna2,
            antenna_position,
            make_field_direction(1.0, 0.5, "AZEL"),
        )
