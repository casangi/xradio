import time

import numpy as np
import xarray as xr

import pyasdm


from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)


def create_pointing_xds(asdm: pyasdm.ASDM) -> xr.Dataset:
    """
    Create an xarray Dataset with pointing data from an ASDM.

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

    # Definitions for coords/time_pointing,
    #  if no sampled timeInterval: getStart() + interval/2 + idx * (interval), (interval = getDuration() / numSamples)
    #  if sampled timeInterval: the per sample ASDM timeInterval getStart()+getDuration()/2
    num_samples = int(pointing_df["numSample"].values[0])
    time_interval_all_rows = pointing_df["timeInterval"]
    num_rows = len(time_interval_all_rows)
    all_time_centers = np.zeros((num_rows * num_samples), dtype="float64")
    all_row_idx = np.arange(0, num_rows)
    for row_idx in all_row_idx:
        time_interval = time_interval_all_rows[row_idx]
        # ASDM always in ns
        interval = (time_interval.getDuration() / num_samples).get()
        first_center = ((time_interval.getStart() + int(interval)) / 2).get()
        last_center = first_center + (num_samples) * interval
        time_centers_from_row = np.arange(first_center, last_center, interval)
        index_time_centers = row_idx * num_samples
        all_time_centers[index_time_centers : index_time_centers + num_samples] = (
            time_centers_from_row
        )

    time_pointing = ("time", all_time_centers)
    antenna_name = ("antenna_name", antenna_df["name"].values.astype("str"))
    xds = xds.assign_coords(
        {
            "time_pointing": time_pointing,
            "antenna_name": antenna_name,
            "local_sky_dirlabel": ["az", "alt"],
        }
    )

    for row_idx in all_row_idx:
        # How the MSv4 data_vars are derived from the attributes of the ASDM pointing table:
        # MSv4/POINTING_BEAM = rotate(ASDM/target, ASDM/offset) + correction
        #  where correction = (ASDM/encoder - ASDM/pointingDirection) is applied when
        target = pointing_df["target"]
        offset = pointing_df["offset"]
        # rotate_target_to_offset(target, offset)

        # MSV4/(POINTING_DISH_MEASURED) = ASDM/encoder
        encoder = pointing_df["encoder"]
        pointing_direction = pointing_df["pointingDirection"]

        # MSv4/(POINTING_OVER_THE_TOP) = ASDM/overTheTop (optional attribute apparently not present in usual ALMA ASDMs

    xradio_logger().info(
        f"create_pointing_xds() took {time.time() - time_start:0.2f} s"
    )

    return xds
