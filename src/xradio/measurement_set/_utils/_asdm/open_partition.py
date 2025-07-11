import datetime
import importlib

import numpy as np
import xarray as xr

import pyasdm

from xradio.measurement_set.schema import MSV4_SCHEMA_VERSION


def open_partition(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
) -> xr.DataTree:
    """
    TODO: opens a partition as an MSv4 DataTre

    Parameters
    ----------
    asdm:
        Input ASDM object
    partition_descr:
        description of partition IDs in a dictionary of "ID ASDM key/attribute" -> numeric IDs

    Returns
    -------
    xr.DataTree
        Datatree with MSv4 populated from the ASDM partition
    """
    msv4_xdt = xr.DataTree()

    msv4_xdt.ds = open_correlated_xds(asdm, partition_descr)

    msv4_xdt["/antenna_xds"] = xr.Dataset(attrs={"type": "antenna"})

    # TODO:

    # gain_curve_xds

    # phase_calibration_xds

    # system_calibration_xds

    # weather_xds

    # pointing_xds

    # phased_array_xds

    # field_and_source_xds

    # info_dicts

    return msv4_xdt


def open_correlated_xds(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
) -> xr.DataTree:
    xds = xr.Dataset(
        attrs={
            "schema_version": MSV4_SCHEMA_VERSION,
            "creator": {
                "software_name": "xradio",
                "version": importlib.metadata.version("xradio"),
            },
            "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "visibility",
        }
    )

    xds.assign_coords(create_coordinates(asdm, partition_descr))

    return xds


def create_coordinates(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
) -> dict:
    coords = {
        "time": None,
        "baseline_id": None,
        "frequency": None,
        "polarization": None,
        "field_name": None,
        "baseline_antenna1_id": None,
        "baseline_antenna2_id": None,
        "scan_name": None,
        "uvw_label": None,
    }

    return coords
