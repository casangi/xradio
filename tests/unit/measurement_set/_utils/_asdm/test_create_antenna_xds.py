import copy
from unittest import mock

import astropy.units as u
import numpy as np
import pandas as pd
import pyasdm
import pytest
import xarray as xr
from astropy.coordinates import EarthLocation

import xradio.measurement_set._utils._asdm.create_antenna_xds as antenna_module
from xradio.measurement_set._utils._asdm.create_antenna_xds import (
    create_antenna_xds,
    create_feed_xds,
    get_telescope_name,
)
from xradio.measurement_set.schema import AntennaXds
from xradio.schema.check import check_dataset

# Values of the conftest add_antenna_station_tables / add_feed_table fixtures
# (real ALMA ACA rows)
PAD_POSITIONS = {
    "CM01": np.array([2225077.740418, -5440126.559719, -2481521.871323]),
    "CM03": np.array([2225074.121158, -5440116.537775, -2481546.908647]),
}
ANTENNA_LOCAL_POSITIONS = {
    "CM01": np.array([-0.002052, -2.32e-4, 7.502983]),
    "CM03": np.array([-0.001862, 0.001218, 7.49941]),
}
FIXTURE_RECEPTOR_ANGLES = [-0.9346238144, 0.6361725124]


def assert_no_schema_issues(xds):
    issues = check_dataset(xds, AntennaXds)
    assert not issues, str(issues)


def warnings_logged(mock_logger) -> list[str]:
    return [str(call.args[0]) for call in mock_logger.warning.call_args_list]


@pytest.fixture
def mock_logger(monkeypatch):
    logger = mock.MagicMock()
    monkeypatch.setattr(antenna_module, "xradio_logger", lambda: logger)
    return logger


def feed_row_xml(
    antenna_id: int,
    spw_id: int,
    receptor_angles=(0.1, 0.2),
    polarization_types=("X", "Y"),
    feed_id: int = 0,
    time_mid: int = 7226686294548387903,
    duration: int = 3993371484612775807,
) -> str:
    num_receptor = len(polarization_types)
    angles = " ".join(repr(float(angle)) for angle in receptor_angles)
    zeros_beam = " ".join(["0.0"] * (2 * num_receptor))
    zeros_focus = " ".join(["0.0"] * (3 * num_receptor))
    zeros_resp = " ".join(["0.0"] * (num_receptor * num_receptor * 2))
    return f"""
  <row>
    <feedId> {feed_id} </feedId>
    <timeInterval> {time_mid} {duration} </timeInterval>
    <numReceptor> {num_receptor} </numReceptor>
    <beamOffset> 2 {num_receptor} 2 {zeros_beam} </beamOffset>
    <focusReference> 2 {num_receptor} 3 {zeros_focus} </focusReference>
    <polarizationTypes> 1 {num_receptor} {" ".join(polarization_types)} </polarizationTypes>
    <polResponse> 2 {num_receptor} {num_receptor} {zeros_resp} </polResponse>
    <receptorAngle> 1 {len(receptor_angles)} {angles} </receptorAngle>
    <antennaId> Antenna_{antenna_id} </antennaId>
    <receiverId> 1 {num_receptor} {" ".join(["0"] * num_receptor)} </receiverId>
    <spectralWindowId> SpectralWindow_{spw_id} </spectralWindowId>
  </row>
    """


def add_feed_rows(asdm, *rows_xml):
    """Add Feed rows without uniqueness check (as pyasdm does when loading
    tables from files), so that several rows per antenna/SPW are possible."""
    table = asdm.getFeed()
    for xml in rows_xml:
        row = pyasdm.FeedRow(table)
        row.setFromXML(xml)
        table.checkAndAdd(row, True)


@pytest.fixture
def asdm_antennas(asdm_with_execblock_antenna_station_feed):
    """CM01/CM03 ASDM with feed rows for SPW 0 only; tests add SPW 5 rows."""
    return copy.deepcopy(asdm_with_execblock_antenna_station_feed)


def test_create_antenna_xds_empty():
    with pytest.raises(AttributeError, match="has no attribute"):
        create_antenna_xds(None, 2, 0, xr.DataArray(["XX", "YY"]))


def test_create_antenna_xds_with_asdm_empty(asdm_empty):
    with pytest.raises(RuntimeError, match="Issue with telescopeName"):
        create_antenna_xds(asdm_empty, 0, 0, xr.DataArray(["XX", "YY"]))


def test_create_antenna_xds_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(RuntimeError, match="antennas found"):
        create_antenna_xds(asdm_with_spw_default, 3, 0, xr.DataArray(["XX", "YY"]))


def test_create_antenna_xds_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(RuntimeError, match="antennas found"):
        create_antenna_xds(asdm_with_spw_simple, 4, 0, xr.DataArray(["XX", "YY"]))


def test_create_antenna_xds_with_asdm_simple_7m_antennas(
    asdm_with_execblock_antenna_station_feed,
):
    antenna_xds = create_antenna_xds(
        asdm_with_execblock_antenna_station_feed, 2, 0, xr.DataArray(["XX", "YY"])
    )
    assert_no_schema_issues(antenna_xds)

    assert list(antenna_xds.antenna_name.values) == ["CM01", "CM03"]
    assert list(antenna_xds.station_name.values) == ["N602", "J503"]
    assert list(antenna_xds.mount.values) == ["ALT-AZ", "ALT-AZ"]
    assert list(antenna_xds.telescope_name.values) == ["ALMA", "ALMA"]
    assert antenna_xds.attrs["overall_telescope_name"] == "ALMA"
    assert antenna_xds.attrs["relocatable_antennas"] is True
    assert list(antenna_xds.cartesian_pos_label.values) == ["x", "y", "z"]
    np.testing.assert_array_equal(antenna_xds.ANTENNA_DISH_DIAMETER.values, [7.0, 7.0])
    assert antenna_xds.ANTENNA_DISH_DIAMETER.attrs["units"] == "m"
    assert antenna_xds.ANTENNA_POSITION.attrs["frame"] == "ITRS"
    assert antenna_xds.ANTENNA_POSITION.attrs["coordinate_system"] == "geocentric"

    # Feed: matched by antenna, both rows with the same values in the fixture
    assert list(antenna_xds.receptor_label.values) == ["pol_0", "pol_1"]
    assert antenna_xds.polarization_type.values.tolist() == [["X", "Y"], ["X", "Y"]]
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_RECEPTOR_ANGLE.values,
        [FIXTURE_RECEPTOR_ANGLES, FIXTURE_RECEPTOR_ANGLES],
    )
    assert antenna_xds.ANTENNA_RECEPTOR_ANGLE.attrs["units"] == "rad"
    # F68: Feed.focusReference (-99999 placeholder) is not a focus length
    assert "ANTENNA_FOCUS_LENGTH" not in antenna_xds

    # Feed info available: polarization products not needed
    antenna_xds_none = create_antenna_xds(
        asdm_with_execblock_antenna_station_feed, 2, 0, None
    )
    xr.testing.assert_identical(antenna_xds_none, antenna_xds)


def test_antenna_position_rotated_from_local_frame(
    asdm_with_execblock_antenna_station_feed,
):
    """F28: Antenna.position is a local (East, North, Up) vector relative to the
    pad (ALMA: ~7.5 m above the pad), rotated into ITRF before being added to
    the ITRF Station.position."""
    antenna_xds = create_antenna_xds(
        asdm_with_execblock_antenna_station_feed, 2, 0, None
    )
    positions = antenna_xds.ANTENNA_POSITION.transpose(
        "antenna_name", "cartesian_pos_label"
    )
    for name in ["CM01", "CM03"]:
        pad = PAD_POSITIONS[name]
        local = ANTENNA_LOCAL_POSITIONS[name]
        position = positions.sel(antenna_name=name).values
        diff = position - pad
        # a rotation preserves the length of the pad-relative vector
        np.testing.assert_allclose(
            np.linalg.norm(diff), np.linalg.norm(local), atol=1e-9
        )
        # the antenna is above its pad by the Up component (geodetic heights)
        pad_height = EarthLocation.from_geocentric(*pad, unit=u.m).height.to_value(u.m)
        ant_height = EarthLocation.from_geocentric(*position, unit=u.m).height.to_value(
            u.m
        )
        np.testing.assert_allclose(ant_height - pad_height, local[2], atol=1e-6)
        # the pad-relative vector points along the local vertical (geodetic
        # normal of the pad), not along the ITRF z axis
        pad_geodetic = EarthLocation.from_geocentric(*pad, unit=u.m)
        lat = pad_geodetic.lat.to_value(u.rad)
        lon = pad_geodetic.lon.to_value(u.rad)
        up = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        np.testing.assert_allclose(np.dot(diff, up), local[2], atol=1e-6)
        # unrotated sum (previous behaviour) would be ~12.5 m away
        assert np.linalg.norm(position - (pad + local)) > 12.0


def test_antenna_position_equals_station_for_zero_offsets(asdm_antennas):
    asdm = asdm_antennas
    for row in asdm.getAntenna().get():
        row.setPosition([pyasdm.types.Length(0.0)] * 3)
    antenna_xds = create_antenna_xds(asdm, 2, 0, None)
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_POSITION.transpose("antenna_name", ...).values,
        [PAD_POSITIONS["CM01"], PAD_POSITIONS["CM03"]],
    )


def test_antenna_offset_added_in_local_frame(
    asdm_with_execblock_antenna_station_feed, asdm_antennas, mock_logger
):
    """Antenna.offset is added as a pad-relative local (East, North, Up) vector,
    like Antenna.position."""
    reference_xds = create_antenna_xds(
        asdm_with_execblock_antenna_station_feed, 2, 0, None
    )
    # (rows modified before any table read of this ASDM copy, see the table
    # cache of exp_asdm_table_to_df)
    rows = asdm_antennas.getAntenna().get()
    # move the 7.5 m height of CM01 from Antenna.position into Antenna.offset
    rows[0].setPosition(
        [pyasdm.types.Length(val) for val in (-0.002052, -2.32e-4, 0.0)]
    )
    rows[0].setOffset([pyasdm.types.Length(val) for val in (0.0, 0.0, 7.502983)])
    antenna_xds = create_antenna_xds(asdm_antennas, 2, 0, None)
    np.testing.assert_allclose(
        antenna_xds.ANTENNA_POSITION.values,
        reference_xds.ANTENNA_POSITION.values,
        rtol=0,
        atol=1e-8,
    )
    assert any("Antenna.offset" in msg for msg in warnings_logged(mock_logger))


def test_create_antenna_xds_antenna_id_selection(
    asdm_with_execblock_antenna_station_feed,
):
    asdm = asdm_with_execblock_antenna_station_feed
    default_xds = create_antenna_xds(asdm, 2, 0, None)

    reversed_xds = create_antenna_xds(asdm, 2, 0, None, antenna_id=[1, 0])
    assert_no_schema_issues(reversed_xds)
    assert list(reversed_xds.antenna_name.values) == ["CM03", "CM01"]
    assert list(reversed_xds.station_name.values) == ["J503", "N602"]
    xr.testing.assert_identical(
        reversed_xds.sel(antenna_name=["CM01", "CM03"]), default_xds
    )

    subset_xds = create_antenna_xds(asdm, 1, 0, None, antenna_id=np.array([1]))
    assert list(subset_xds.antenna_name.values) == ["CM03"]
    xr.testing.assert_identical(subset_xds, default_xds.sel(antenna_name=["CM03"]))

    with pytest.raises(RuntimeError, match=r"antenna ids \[7\] were not found"):
        create_antenna_xds(asdm, 2, 0, None, antenna_id=[0, 7])
    with pytest.raises(RuntimeError, match="2 antenna ids were given"):
        create_antenna_xds(asdm, 3, 0, None, antenna_id=[0, 1])


def test_feed_rows_matched_by_antenna_id(asdm_antennas):
    """F67: Feed rows are matched to the antennas by antennaId, whatever their
    order in the Feed table."""
    asdm = asdm_antennas
    add_feed_rows(
        asdm,
        feed_row_xml(1, 5, receptor_angles=(-0.2, 1.2)),
        feed_row_xml(0, 5, receptor_angles=(-0.1, 1.1)),
    )
    antenna_xds = create_antenna_xds(asdm, 2, 5, None)
    assert_no_schema_issues(antenna_xds)
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_RECEPTOR_ANGLE.sel(antenna_name="CM01").values, [-0.1, 1.1]
    )
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_RECEPTOR_ANGLE.sel(antenna_name="CM03").values, [-0.2, 1.2]
    )


def test_feed_rows_duplicate_and_gap(asdm_antennas, mock_logger):
    """F67: one antenna with two rows (time intervals) and another antenna
    without row: the earliest row goes to its own antenna, the antenna without
    row gets NaN angles (no misassignment, no AlignmentError)."""
    asdm = asdm_antennas
    add_feed_rows(
        asdm,
        feed_row_xml(0, 5, receptor_angles=(-0.15, 1.15), time_mid=5230001552242000000),
        feed_row_xml(0, 5, receptor_angles=(-0.1, 1.1), time_mid=5230000552242000000),
    )
    antenna_xds = create_antenna_xds(asdm, 2, 5, None)
    assert_no_schema_issues(antenna_xds)
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_RECEPTOR_ANGLE.transpose("antenna_name", ...).values,
        [[-0.1, 1.1], [np.nan, np.nan]],
    )
    assert antenna_xds.polarization_type.values.tolist() == [["X", "Y"], ["X", "Y"]]
    messages = warnings_logged(mock_logger)
    assert any("several rows" in msg for msg in messages)
    assert any("['CM03'] have no Feed row" in msg for msg in messages)


def test_feed_rows_lowest_feed_id(asdm_antennas):
    asdm = asdm_antennas
    add_feed_rows(
        asdm,
        feed_row_xml(0, 5, receptor_angles=(-0.3, 1.3), feed_id=1),
        feed_row_xml(0, 5, receptor_angles=(-0.1, 1.1), feed_id=0),
        feed_row_xml(1, 5, receptor_angles=(-0.2, 1.2), feed_id=0),
    )
    antenna_xds = create_antenna_xds(asdm, 2, 5, None)
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_RECEPTOR_ANGLE.values, [[-0.1, 1.1], [-0.2, 1.2]]
    )


def test_feed_single_receptor(asdm_antennas):
    """F67: the number of receptors comes from Feed.numReceptor."""
    asdm = asdm_antennas
    add_feed_rows(
        asdm,
        feed_row_xml(0, 5, receptor_angles=(0.1,), polarization_types=("X",)),
        feed_row_xml(1, 5, receptor_angles=(0.2,), polarization_types=("X",)),
    )
    antenna_xds = create_antenna_xds(asdm, 2, 5, None)
    assert_no_schema_issues(antenna_xds)
    assert list(antenna_xds.receptor_label.values) == ["pol_0"]
    assert antenna_xds.polarization_type.values.tolist() == [["X"], ["X"]]
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_RECEPTOR_ANGLE.values, [[0.1], [0.2]]
    )


def test_feed_inconsistent_num_receptor_raises(asdm_antennas):
    asdm = asdm_antennas
    add_feed_rows(
        asdm,
        feed_row_xml(0, 5, receptor_angles=(0.1,), polarization_types=("X",)),
        feed_row_xml(1, 5),
    )
    with pytest.raises(RuntimeError, match="different numbers of receptors"):
        create_antenna_xds(asdm, 2, 5, None)


@pytest.mark.parametrize(
    "products, expected",
    [
        (["XX", "YY"], ["X", "Y"]),
        (["XX", "XY", "YX", "YY"], ["X", "Y"]),
        (["XX"], ["X"]),
        (["RR", "RL", "LR", "LL"], ["R", "L"]),
    ],
)
def test_no_feed_rows_receptors_from_correlation_products(
    asdm_with_execblock_antenna_station_feed, products, expected
):
    """F95: without Feed rows for the SPW (e.g. ALMA WVR), the receptor types
    are derived from the correlation products."""
    antenna_xds = create_antenna_xds(
        asdm_with_execblock_antenna_station_feed,
        2,
        3,
        xr.DataArray(products, dims="polarization"),
    )
    assert_no_schema_issues(antenna_xds)
    assert list(antenna_xds.receptor_label.values) == [
        f"pol_{idx}" for idx in range(len(expected))
    ]
    assert antenna_xds.polarization_type.values.tolist() == [expected, expected]
    assert "ANTENNA_RECEPTOR_ANGLE" not in antenna_xds


def test_no_feed_rows_and_no_products_raises(asdm_with_execblock_antenna_station_feed):
    with pytest.raises(RuntimeError, match="no correlation products"):
        create_antenna_xds(asdm_with_execblock_antenna_station_feed, 2, 3, None)


def test_create_antenna_xds_with_asdm_sd_with_execblock_antenna_station_feed(
    asdm_sd_with_execblock_antenna_station_feed, mock_logger
):
    asdm = asdm_sd_with_execblock_antenna_station_feed
    # SPW 0: no Feed rows
    antenna_xds = create_antenna_xds(asdm, 2, 0, xr.DataArray(["XX", "YY"]))
    assert_no_schema_issues(antenna_xds)
    assert list(antenna_xds.antenna_name.values) == ["DA41", "DA42"]
    assert list(antenna_xds.station_name.values) == ["S306", "S301"]
    np.testing.assert_array_equal(
        antenna_xds.ANTENNA_DISH_DIAMETER.values, [12.0, 12.0]
    )
    assert antenna_xds.polarization_type.values.tolist() == [["X", "Y"], ["X", "Y"]]
    assert "ANTENNA_RECEPTOR_ANGLE" not in antenna_xds

    # SPW 0 without Feed rows nor polarization products
    with pytest.raises(RuntimeError, match="no correlation products"):
        create_antenna_xds(asdm, 2, 0, None)

    # The fixture Feed table has a gap: SPW 44 only for Antenna_0 and SPW 45
    # only for Antenna_1 (previously an AlignmentError)
    xds_44 = create_antenna_xds(asdm, 2, 44, None)
    assert_no_schema_issues(xds_44)
    np.testing.assert_array_equal(
        xds_44.ANTENNA_RECEPTOR_ANGLE.values,
        [[-0.1745329252, 1.3962634016], [np.nan, np.nan]],
    )
    xds_45 = create_antenna_xds(asdm, 2, 45, None)
    np.testing.assert_array_equal(
        xds_45.ANTENNA_RECEPTOR_ANGLE.values,
        [[np.nan, np.nan], [-0.1745329252, 1.3962634016]],
    )
    assert xds_45.polarization_type.values.tolist() == [["X", "Y"], ["X", "Y"]]


def test_create_feed_xds_empty():
    with pytest.raises(KeyError, match="name_antenna"):
        create_feed_xds(None, pd.DataFrame(), 0, xr.DataArray(["XX"]))


def test_create_feed_xds_with_asdm_empty(asdm_empty):
    antenna_df = pd.DataFrame({"antennaId": [0, 1], "name_antenna": ["A1", "A2"]})
    feed_xds = create_feed_xds(asdm_empty, antenna_df, 0, xr.DataArray(["XX", "YY"]))
    assert list(feed_xds.antenna_name.values) == ["A1", "A2"]
    assert feed_xds.polarization_type.values.tolist() == [["X", "Y"], ["X", "Y"]]


def test_create_feed_xds_matches_antenna_df_order(
    asdm_with_execblock_antenna_station_feed,
):
    asdm = copy.deepcopy(asdm_with_execblock_antenna_station_feed)
    add_feed_rows(
        asdm,
        feed_row_xml(0, 5, receptor_angles=(-0.1, 1.1)),
        feed_row_xml(1, 5, receptor_angles=(-0.2, 1.2)),
    )
    antenna_df = pd.DataFrame({"antennaId": [1, 0], "name_antenna": ["CM03", "CM01"]})
    feed_xds = create_feed_xds(asdm, antenna_df, 5, None)
    assert list(feed_xds.antenna_name.values) == ["CM03", "CM01"]
    np.testing.assert_array_equal(
        feed_xds.ANTENNA_RECEPTOR_ANGLE.values, [[-0.2, 1.2], [-0.1, 1.1]]
    )


def test_get_telescope_name_asdm_default(asdm_with_spw_default):
    with pytest.raises(RuntimeError, match="Issue with telescopeName"):
        get_telescope_name(asdm_with_spw_default)


def test_get_telescope_name_asdm_with_spw_simple(asdm_with_execblock_spw_simple):
    name = get_telescope_name(asdm_with_execblock_spw_simple)
    assert name == "ALMA"
