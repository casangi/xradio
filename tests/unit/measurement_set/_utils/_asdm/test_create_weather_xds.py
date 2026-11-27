import pytest

from xradio.measurement_set._utils._asdm.create_weather_xds import (
    create_weather_xds,
)
from xradio.measurement_set.schema import WeatherXds
from xradio.schema.check import check_dataset


def test_create_weather_xds_empty():
    with pytest.raises(AttributeError, match="NoneType"):
        create_weather_xds(None)


expected_exception_msg = "Could not convert tuple"


def test_create_weather_xds_with_asdm_empty(asdm_empty):
    with pytest.raises(ValueError, match=expected_exception_msg):
        create_weather_xds(asdm_empty)


def test_create_weather_xds_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(ValueError, match=expected_exception_msg):
        create_weather_xds(asdm_with_spw_default)


def test_create_weather_xds_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(ValueError, match=expected_exception_msg):
        create_weather_xds(asdm_with_spw_simple)


def test_create_weather_xds_with_asdm_antenna_station_weather(
    asdm_with_antenna_station_weather,
):
    weather_xds = create_weather_xds(asdm_with_antenna_station_weather)
    issues = check_dataset(weather_xds, WeatherXds)
    assert not issues

    for coord in ["station_name", "time_weather", "cartesian_pos_label"]:
        assert coord in weather_xds.coords

    for data_var in [
        "STATION_POSITION",
        "PRESSURE",
        "REL_HUMIDITY",
        "TEMPERATURE",
        "WIND_DIRECTION",
        "WIND_SPEED",
        "DEW_POINT",
    ]:
        assert data_var in weather_xds.data_vars

    # The fixture contains two stations present in the Station table.
    assert set(weather_xds.coords["station_name"].values) == {
        "N602",
        "J503",
    }

    # Station positions come from the Station table.
    assert weather_xds["STATION_POSITION"].shape == (2, 3)
