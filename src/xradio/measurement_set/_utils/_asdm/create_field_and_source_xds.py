import numpy as np
import pandas as pd
import xarray as xr

import pyasdm

from xradio.measurement_set._utils._asdm._utils.field_source import get_direction_codes
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio._utils.dict_helpers import make_quantity, make_spectral_coord_measure_attrs


def create_field_and_source_xds(
    asdm: pyasdm.ASDM,
    partition_descr: dict,
    spectral_window_id: int,
    is_single_dish: bool,
) -> xr.Dataset:
    """
    Create an xarray Dataset containing field and source information from an ASDM.
    This function extracts field and source information from an ASDM and creates an xarray
    Dataset with coordinates and variables describing the field position, source direction,
    and spectral line information if available.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object to extract data from
    partition_descr : dict
        Dictionary containing partition description with at least a 'fieldId' key
    spectral_window_id : int
        ID of the spectral window to filter source information
    is_single_dish : bool
        Flag indicating if data is from single dish observations. Affects which center
        direction variable name is used.

    Returns
    -------
    xr.Dataset
        Dataset containing field and source information with the following structure:
        - Coordinates:
            - sky_dir_label: ['ra', 'dec']
            - field_name: field names as strings
            - source_name: source names as strings
            - line_label, line_name: (optional) spectral line information if available
        - Data variables:
            - FIELD_REFERENCE_CENTER or FIELD_PHASE_CENTER: field center coordinates
            - SOURCE_DIRECTION: source direction coordinates
            - LINE_REST_FREQUENCY: (optional) rest frequencies for spectral lines
            - LINE_SYSTEMIC_VELOCITY: (optional) systemic velocities for spectral lines
        - Attributes:
            - type: 'field_and_source'
            - is_ephemeris: boolean flag for ephemeris sources

    Raises
    ------
    RuntimeError
        If source_id or source_name are not unique for the given field
    """

    xds = xr.Dataset(attrs={"type": "field_and_source"})

    field_id = partition_descr["fieldId"]

    # TODO: sourceId is an opt attr
    sdm_field_attrs = ["fieldId", "fieldName", "referenceDir", "sourceId"]
    field_df = exp_asdm_table_to_df(asdm, "Field", sdm_field_attrs)

    field_df = field_df.loc[field_df["fieldId"].isin(partition_descr["fieldId"])]

    field_name = field_df["fieldName"].unique().astype("str")
    field_coords = {
        "sky_dir_label": ["ra", "dec"],
        "field_name": ("field_name", field_name),
    }
    xds = xds.assign_coords(field_coords)

    # TODO: to split in _DIRECTION/_DISTANCE
    if is_single_dish:
        center_dv = "FIELD_REFERENCE_CENTER"
    else:
        center_dv = "FIELD_PHASE_CENTER"

    # ignore the polynomial dimension
    ref_dir = field_df["referenceDir"].values[0][0]
    xds[center_dv] = (
        ["field_name", "sky_dir_label"],
        [[ref_dir[0].get(), ref_dir[1].get()]],
    )
    xds.data_vars[center_dv].attrs.update(
        make_sky_coord_measure_attrs(["rad", "rad"], "fk5")
    )

    line_info_available = True
    sdm_source_required_attrs = [
        "sourceId",
        "timeInterval",
        "spectralWindowId",
        "direction",
        "sourceName",
        # "directionCode",  it is optional, get it via a helper func
    ]
    sdm_source_optional_attrs = [
        "numLines",
        "transition",
        "restFrequency",
        "sysVel",
    ]
    try:
        source_df = exp_asdm_table_to_df(
            asdm, "Source", sdm_source_required_attrs + sdm_source_optional_attrs
        )
    except NameError:
        source_df = exp_asdm_table_to_df(asdm, "Source", sdm_source_required_attrs)
        line_info_available = False

    source_id = field_df["sourceId"].unique()
    if len(source_id) != 1:
        raise RuntimeError(f"{source_id=} not unique!")
    source_id_int = source_id[0]
    source_df = source_df.loc[
        (source_df["spectralWindowId"] == spectral_window_id)
        & (source_df["sourceId"].isin(np.array(source_id)))
    ]

    source_name = source_df["sourceName"].unique().astype("str")
    if len(source_name) != 1:
        raise RuntimeError(f"{source_name=} not unique!")
    source_coords = {
        "source_name": ("field_name", source_name),
    }
    xds = xds.assign_coords(source_coords)

    # TODO: to split in _DIRECTION/_DISTANCE
    source_direction = source_df["direction"].values[0]
    source_key = (
        source_df["sourceId"].values[0],
        source_df["timeInterval"].values[0],
        source_df["spectralWindowId"].values[0],
    )
    dir_code = get_direction_codes(asdm, source_key)
    xds["SOURCE_DIRECTION"] = (
        ["field_name", "sky_dir_label"],
        [[source_direction[0].get(), source_direction[1].get()]],
        make_sky_coord_measure_attrs(["rad", "rad"], dir_code),
    )

    if line_info_available:
        # TODO: fix this when some sources have it and some others don't
        rest_freq = source_df["restFrequency"].values
        xds["LINE_REST_FREQUENCY"] = (
            "field_name",
            rest_freq,
            make_spectral_coord_measure_attrs("Hz", observer="TOPO"),
        )
        xds["LINE_SYSTEMIC_VELOCITY"] = ("field_name", make_quantity("m/s"))

        line_name = source_df["transition"].values
        line_label = [f"line_{idx}" for idx in np.arange(line_name)]
        line_coords = {
            "line_label": line_label,
            "line_name": ("line_label", line_name),
        }
        xds = xds.assign_coords(line_coords)

    # TODO: ephem
    is_ephemeris = False
    xds.attrs.update({"is_ephemeris": is_ephemeris})

    return xds


# TODO: to move to dict-helpers or related place
def make_sky_coord_measure_attrs(units: str, frame: str) -> dict:
    """
    Create a dictionary of sky coordinate measure attributes.
    Parameters
    ----------
    units : str or list
        Units for sky coordinate measure. Can be a single string or list of strings.
    frame : str
        Reference frame for sky coordinate measure.
    Returns
    -------
    dict
        Dictionary containing the measure attributes with the following keys:
        - units: list of units
        - frame: reference frame
        - type: fixed to "sky_coord"
    Examples
    --------
    >>> make_sky_coord_measure_attrs("rad", "ICRS")
    {'units': ['rad'], 'frame': 'ICRS', 'type': 'sky_coord'}
    """

    unt = units if isinstance(units, list) else [units]
    sky_coord_measure_attrs = {"units": unt, "frame": frame, "type": "sky_coord"}
    return sky_coord_measure_attrs
