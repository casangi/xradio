import time

import numpy as np
import pyasdm
import xarray as xr

from xradio._utils.dict_helpers import make_quantity_attrs, make_time_measure_attrs
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

    sdm_station_attrs = ["stationId", "name", "position"]
    station_df = exp_asdm_table_to_df(asdm, "Station", sdm_station_attrs)
    if station_df.empty:
        raise RuntimeError("The ASDM Station table seems empty")

    sdm_weather_attrs = [
        "stationId",
        "timeInterval",
        "pressure",
        "relHumidity",
        "temperature",
        "windDirection",
        "windSpeed",
        "dewPoint",
    ]
    weather_df = exp_asdm_table_to_df(
        asdm, "Weather", sdm_weather_attrs, optional_per_row=True
    )
    if weather_df.empty:
        raise RuntimeError("The ASDM Weather table seems empty")

    # Station name lookup and positions.
    station_lookup = dict(
        zip(station_df["stationId"], station_df["name"].astype(str), strict=True)
    )
    station_names = station_df["name"].astype(str).to_numpy()
    station_positions = np.array(
        [
            [value.get() for value in position]
            for position in station_df["position"].to_numpy()
        ],
        dtype=float,
    )

    # Weather rows are tagged with a station id; resolve to station names.
    weather_df["station_name"] = weather_df["stationId"].map(station_lookup)

    # Convert ASDM time intervals to Unix timestamps (mid points).
    weather_midpoints = np.array(
        [interval.getMidPoint() for interval in weather_df["timeInterval"]]
    )
    weather_times = convert_time_asdm_to_unix(weather_midpoints)
    weather_df["time_weather"] = weather_times

    unique_stations = list(dict.fromkeys(station_names.tolist()))
    unique_times = list(dict.fromkeys(weather_times.tolist()))
    station_index = {name: idx for idx, name in enumerate(unique_stations)}
    time_index = {value: idx for idx, value in enumerate(unique_times)}

    # Coordinate definitions.
    time_attrs = make_time_measure_attrs("s", "tai", time_format="unix")
    time_attrs["type"] = "time_weather"

    xds = xds.assign_coords(
        {
            "station_name": ("station_name", unique_stations),
            "time_weather": ("time_weather", unique_times, time_attrs),
            "cartesian_pos_label": ("cartesian_pos_label", ["x", "y", "z"]),
        }
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

    # Map each weather row to its station/time indices in the output arrays.
    station_indices = weather_df["station_name"].map(station_index)
    time_indices = weather_df["time_weather"].map(time_index)
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

    for asdm_name, (msv4_name, attrs) in quantity_specs.items():
        values = np.full((len(unique_stations), len(unique_times)), np.nan, dtype=float)
        # Convert optional ASDM quantity objects to their numeric values.
        numeric_values = (
            weather_df[asdm_name].map(_asdm_quantity_to_numeric).to_numpy(dtype=float)
        )
        valid_value = ~np.isnan(numeric_values)
        mask = valid_station_time.to_numpy() & valid_value

        if mask.any():
            si = station_indices[mask].astype(int).to_numpy()
            ti = time_indices[mask].astype(int).to_numpy()
            values[si, ti] = numeric_values[mask]

        xds[msv4_name] = (
            ["station_name", "time_weather"],
            values,
            attrs,
        )

    xradio_logger().debug(
        f"create_weather_xds() took {time.time() - time_start:0.2f} s"
    )

    return xds


def _asdm_quantity_to_numeric(value):
    """Convert various ASDM quantity objects (Pressure, Temp)"""
    if hasattr(value, "get"):
        return value.get()
    return value
