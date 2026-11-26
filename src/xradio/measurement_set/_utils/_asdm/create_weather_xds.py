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
    weather_df = weather_df.copy()
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

    # Helper to convert ASDM quantity objects (Pressure, Temperature, ...)
    # to their underlying numeric value.
    def _to_number(value):
        if hasattr(value, "get"):
            return value.get()
        return value

    # Weather quantities, mapped from ASDM attributes to MSv4 data variables.
    quantity_specs = {
        "pressure": ("PRESSURE", make_quantity_attrs("Pa")),
        "relHumidity": ("REL_HUMIDITY", {}),
        "temperature": ("TEMPERATURE", make_quantity_attrs("K")),
        "windDirection": ("WIND_DIRECTION", make_quantity_attrs("rad")),
        "windSpeed": ("WIND_SPEED", make_quantity_attrs("m/s")),
        "dewPoint": ("DEW_POINT", make_quantity_attrs("K")),
    }

    for source_name, (target_name, attrs) in quantity_specs.items():
        values = np.full((len(unique_stations), len(unique_times)), np.nan, dtype=float)
        for row in weather_df.itertuples(index=False):
            station_name = row.station_name
            if station_name not in station_index:
                continue
            si = station_index[station_name]
            ti = time_index[row.time_weather]
            raw = _to_number(getattr(row, source_name))
            if isinstance(raw, (float, np.floating)) and np.isnan(raw):
                continue
            values[si, ti] = float(raw)

        xds[target_name] = (
            ["station_name", "time_weather"],
            values,
            attrs,
        )

    xradio_logger().info(f"create_weather_xds() took {time.time() - time_start:0.2f} s")

    print(f" ****{xds=}")
    return xds
