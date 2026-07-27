import time

import numpy as np
import pandas as pd
import xarray as xr

import pyasdm

from xradio._utils.dict_helpers import make_time_measure_attrs
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.pointing_direction_rotation import (
    rotate_offset_to_target,
)
from xradio.measurement_set._utils._asdm._utils.sky_coord_dict_helper import (
    make_sky_coord_measure_attrs,
)
from xradio.measurement_set._utils._asdm._utils.time import convert_time_asdm_to_unix


def create_pointing_xds(asdm: pyasdm.ASDM) -> xr.Dataset:
    """
    Build an xarray Dataset with antenna pointing information extracted from
    an ASDM Pointing table.

    How the MSv4 data_vars are derived from the attributes of the ASDM pointing table:
    - MSv4/POINTING_BEAM = rotate(ASDM/target, ASDM/offset) + correction
         where correction = (ASDM/encoder - ASDM/pointingDirection) is applied when
         importasdm/with_pointing_correction=True.
    - MSV4/(POINTING_DISH_MEASURED) = ASDM/encoder
    - MSv4/(POINTING_OVER_THE_TOP) = ASDM/overTheTop (optional attribute apparently not
         present in usual ALMA ASDMs


    Parameters
    ----------
    asdm : pyasdm.ASDM
        ASDM instance from which the Pointing and Antenna tables are read.

    Returns
    -------
    xr.Dataset
        Dataset with pointing metadata and data variables derived from the ASDM
        pointing measurements, including the measured dish direction and the
        nominal direction information.
    """

    time_start = time.time()
    xds = xr.Dataset(
        attrs={
            "type": "pointing",
        },
    )

    sdm_antenna_attrs = ["antennaId", "name"]
    antenna_df = exp_asdm_table_to_df(asdm, "Antenna", sdm_antenna_attrs)

    # overTheTop is optional, and not present in common ALMA ASDMs.
    sdm_pointing_attrs = [
        "antennaId",
        "timeInterval",
        "numTerm",
        "numSample",
        "target",
        "offset",
        "encoder",
        "pointingDirection",
    ]
    pointing_df = exp_asdm_table_to_df(asdm, "Pointing", sdm_pointing_attrs)
    if pointing_df.empty:
        raise RuntimeError(
            "The ASDM Pointing table seems empty but it was requested to read it into pointing_xds"
        )

    antenna_groups = pointing_df.groupby("antennaId")

    coords = _make_pointing_coords(pointing_df, antenna_df, antenna_groups)
    xds = xds.assign_coords(coords)

    data_vars = _make_pointing_data_vars(
        xds.coords["time_pointing"].values, antenna_groups
    )
    xds = xds.assign(data_vars)
    xds = xds.transpose("time_pointing", "antenna_name", "local_sky_dir_label")

    xradio_logger().info(
        f"create_pointing_xds() took {time.time() - time_start:0.2f} s"
    )

    return xds


def _make_pointing_coords(
    pointing_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    antenna_groups: pd.api.typing.DataFrameGroupBy,
) -> dict:
    all_antenna_names = []
    time_interval_all_rows = {}
    num_samples = {}
    for antenna_id, group in antenna_groups:
        antenna_name = antenna_df.loc[
            antenna_df["antennaId"] == antenna_id, "name"
        ].values[0]
        all_antenna_names.append(antenna_name)
        time_interval_all_rows[antenna_name] = group["timeInterval"].values
        num_samples[antenna_name] = pointing_df.loc[
            pointing_df["antennaId"] == antenna_id, "numSample"
        ].values

    one_antenna_name = all_antenna_names[0]
    all_time_centers = _retrieve_all_time_centers(
        num_samples[one_antenna_name], time_interval_all_rows[one_antenna_name]
    )
    time_pointing_values = convert_time_asdm_to_unix(all_time_centers)
    time_attrs = make_time_measure_attrs("s", "tai", time_format="unix")
    time_pointing_coord = ("time_pointing", time_pointing_values, time_attrs)
    # Could take: antenna_df["name"].values.astype("str"))
    antenna_name_coord = ("antenna_name", all_antenna_names)
    coords = {
        "time_pointing": time_pointing_coord,
        "antenna_name": antenna_name_coord,
        "local_sky_dir_label": ["az", "alt"],
    }

    return coords


def _make_pointing_data_vars(
    time_pointing_values: np.ndarray, antenna_groups: pd.api.typing.DataFrameGroupBy
) -> dict[str, tuple]:
    # The direction arrays are broken into time/interval rows with a subset
    # of samples within that interval. Concatenate them:
    empty_direction = np.array([]).reshape(0, len(time_pointing_values), 2)
    direction_vars = {
        "target": empty_direction,
        "offset": empty_direction,
        "encoder": empty_direction,
        "pointingDirection": empty_direction,
    }
    for _antenna_id, group in antenna_groups:
        for dvar in direction_vars:
            # First concatenate all the rows for the antenna (along time_pointing coord)
            direction_var_antenna_values = np.concatenate(group[dvar].values)
            # Then concatenate this antenna to all the antennas (along antenna_name coord)
            direction_vars[dvar] = np.concatenate(
                (direction_vars[dvar], [direction_var_antenna_values])
            )

    rotated_target = rotate_offset_to_target(
        direction_vars["target"], direction_vars["offset"]
    )
    rotated_target = direction_vars["target"]
    corrected_rotated_target = (
        rotated_target + direction_vars["encoder"] - direction_vars["pointingDirection"]
    )

    # Using antenna/pointing order first, as it is closer to what we get from the ASDM/Pointing rows
    # time_antenna_dir_dims = ["time_pointing", "antenna_name", "local_sky_dir_label"]
    antenna_time_dir_dims = ["antenna_name", "time_pointing", "local_sky_dir_label"]
    direction_attrs = make_sky_coord_measure_attrs("rad", "altaz")
    data_vars = {
        "DIRECTION": (antenna_time_dir_dims, corrected_rotated_target, direction_attrs),
        "POINTING_DISH_MEASURED": (
            antenna_time_dir_dims,
            direction_vars["encoder"],
            direction_attrs,
        ),
        # "POINTING_OVER_THE_TOP": (antenna_time_dir_dims, over_the_top),
    }

    return data_vars


def _retrieve_all_time_centers(
    num_samples: np.ndarray,
    time_interval_all_rows: np.ndarray,
) -> np.ndarray:
    # This makes the time coord.
    # Definitions for coords/time_pointing:
    #  if no sampled timeInterval: getStart() + interval/2 + idx * (interval), (interval = getDuration() / numSamples)
    #  if sampled timeInterval: the per sample ASDM timeInterval getStart()+getDuration()/2
    rows_per_antenna = len(num_samples)
    total_time_len = sum(num_samples)
    all_time_centers = np.zeros((total_time_len), dtype="float64")
    index_time_centers = 0
    for row_idx in np.arange(0, rows_per_antenna):
        time_interval = time_interval_all_rows[row_idx]
        # For MSv4 only the centers are kept (no INTERVAL col as in MSv2)
        # ASDM always in ns
        row_num_samples = int(num_samples[row_idx])
        sample_interval = (time_interval.getDuration() / row_num_samples).get()
        first_center = ((time_interval.getStart() + int(sample_interval)) / 2).get()
        last_center = first_center + (row_num_samples) * sample_interval
        time_centers_from_row = np.arange(first_center, last_center, sample_interval)
        all_time_centers[index_time_centers : index_time_centers + row_num_samples] = (
            time_centers_from_row
        )
        index_time_centers += row_num_samples

    return all_time_centers
