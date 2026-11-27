import time

import numpy as np
import pandas as pd
import pyasdm
import xarray as xr

from xradio._utils.dict_helpers import make_quantity_attrs, make_time_measure_attrs
from xradio._utils.list_and_array import unique_1d
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.time import (
    convert_time_asdm_to_unix,
)


def create_weather_xds(asdm: pyasdm.ASDM) -> xr.Dataset:
    """
    Build an xarray Dataset with weather information extracted from an ASDM
    Weather (and Station) table.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        ASDM instance from which the Weather and Station tables are read.

    Returns
    -------
    xr.Dataset
        Dataset following the MSv4 ``weather`` schema, with station positions
        and (when available) time-dependent weather quantities.
    """
    time_start = time.time()

    xds = xr.Dataset(
        attrs={
            "type": "weather",
        },
    )

    station_df = _read_station_table(asdm)

    weather_df = _read_weather_table(asdm)

    weather_df = weather_df.merge(
        station_df[["stationId", "name"]],
        on="stationId",
        how="left",
    )
    weather_df = weather_df.rename(columns={"name": "station_name"})

    weather_times = _get_weather_times(weather_df)
    unique_times = unique_1d(weather_times).tolist()

    station_names = station_df["name"].astype(str).to_numpy()
    unique_stations = unique_1d(station_names).tolist()

    xds = _add_coordinates(xds, unique_stations, unique_times)

    xds = _add_station_position(xds, station_df)

    xds = _add_weather_variables(
        xds, weather_df, weather_times, unique_stations, unique_times
    )

    xradio_logger().debug(
        f"create_weather_xds() took {time.time() - time_start:0.2f} s"
    )

    return xds


def _read_station_table(asdm: pyasdm.ASDM) -> pd.DataFrame:

    sdm_station_attrs = ["stationId", "name", "position"]
    station_df = exp_asdm_table_to_df(asdm, "Station", sdm_station_attrs)

    return station_df


def _read_weather_table(asdm: pyasdm.ASDM) -> pd.DataFrame:
    sdm_weather_required_attrs = ["stationId", "timeInterval"]

    sdm_weather_optional_attrs = [
        "pressure",
        "relHumidity",
        "temperature",
        "windDirection",
        "windSpeed",
        "dewPoint",
    ]
    weather_df = exp_asdm_table_to_df(
        asdm,
        "Weather",
        sdm_weather_required_attrs + sdm_weather_optional_attrs,
        optional_per_row=True,
    )

    return weather_df


def _get_weather_times(weather_df: pd.DataFrame) -> np.ndarray:

    # Use mid-points from ASDM TimInterval objects as timestamps
    weather_midpoints = np.array(
        [interval.getMidPoint() for interval in weather_df["timeInterval"]]
    )
    weather_times = convert_time_asdm_to_unix(weather_midpoints)

    return weather_times


def _add_coordinates(
    xds: xr.Dataset, unique_stations: np.ndarray, unique_times: np.ndarray
) -> pd.DataFrame:

    # TODO: revise/settle the tai <=> utc/unix discrepancies together with other datasets (xradio/689)
    time_attrs = make_time_measure_attrs("s", "tai", time_format="unix")
    time_attrs["type"] = "time_weather"

    xds = xds.assign_coords(
        {
            "station_name": ("station_name", unique_stations),
            "time_weather": ("time_weather", unique_times, time_attrs),
            "cartesian_pos_label": ("cartesian_pos_label", ["x", "y", "z"]),
        }
    )

    return xds


def _add_station_position(xds: xr.Dataset, station_df: pd.DataFrame) -> xr.Dataset:

    station_positions = np.array(
        [
            [value.get() for value in position]
            for position in station_df["position"].to_numpy()
        ],
        dtype=float,
    )

    # ASDM fixes position to "The position of the station in geocentric frame
    # of reference (X, Y, Z coordinates). ITRF"
    location_attrs = {
        "type": "location",
        "units": "m",
        "frame": "ITRS",
        "coordinate_system": "geocentric",
        "origin_object_name": "earth",
    }
    xds["STATION_POSITION"] = (
        ["station_name", "cartesian_pos_label"],
        station_positions,
        location_attrs,
    )

    return xds


def _add_weather_variables(
    xds: xr.Dataset,
    weather_df: pd.DataFrame,
    weather_times: np.ndarray,
    unique_stations: np.ndarray,
    unique_times: np.ndarray,
) -> xr.Dataset:

    # Map each weather row to its station/time indices in the output arrays.
    station_names_by_index = {name: idx for idx, name in enumerate(unique_stations)}
    station_indices = weather_df["station_name"].map(station_names_by_index)
    times_by_index = {value: idx for idx, value in enumerate(unique_times)}
    time_indices = pd.Series(weather_times).map(times_by_index)
    valid_station_time = station_indices.notna() & time_indices.notna()

    # Weather quantities, mapped from ASDM attributes to MSv4 data variables.
    quantity_specs = {
        "pressure": ("PRESSURE", make_quantity_attrs("Pa")),
        "relHumidity": ("REL_HUMIDITY", {}),
        "temperature": ("TEMPERATURE", make_quantity_attrs("K")),
        "windDirection": ("WIND_DIRECTION", make_quantity_attrs("rad")),
        "windSpeed": ("WIND_SPEED", make_quantity_attrs("m/s")),
        "dewPoint": ("DEW_POINT", make_quantity_attrs("K")),
    }

    for asdm_attr_name, (msv4_var_name, attrs) in quantity_specs.items():
        values = np.full((len(unique_stations), len(unique_times)), np.nan, dtype=float)
        # Convert optional ASDM quantity objects to their numeric values.
        numeric_values = (
            weather_df[asdm_attr_name]
            .map(_asdm_quantity_to_numeric)
            .to_numpy(dtype=float)
        )
        valid_value = ~np.isnan(numeric_values)
        mask = valid_station_time.to_numpy() & valid_value

        if mask.any():
            sidx = station_indices[mask].astype(int).to_numpy()
            tidx = time_indices[mask].astype(int).to_numpy()
            values[sidx, tidx] = numeric_values[mask]

        xds[msv4_var_name] = (
            ["station_name", "time_weather"],
            values,
            attrs,
        )

    return xds


def _asdm_quantity_to_numeric(value):
    """Convert various ASDM quantity objects (Pressure, Temp)"""
    if hasattr(value, "get"):
        return value.get()
    return value
