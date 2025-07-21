import datetime
import importlib
import itertools

import numpy as np
import xarray as xr

import pyasdm

from xradio.measurement_set.schema import MSV4_SCHEMA_VERSION
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.spectral_window import (
    ensure_spw_name_conforms,
    get_chan_width,
    get_reference_frame,
    get_spw_frequency_centers,
)
from xradio.measurement_set._utils._asdm._utils.time import convert_time_asdm_to_unix
from xradio.measurement_set._utils._asdm.create_antenna_xds import create_antenna_xds
from xradio.measurement_set._utils._asdm.create_info_dicts import create_info_dicts
from xradio._utils.dict_helpers import (
    make_quantity,
    make_spectral_coord_measure_attrs,
    make_spectral_coord_reference_dict,
    make_time_measure_attrs,
)


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

    correlated_xds, num_antenna = open_correlated_xds(asdm, partition_descr)
    msv4_xdt.ds = correlated_xds

    msv4_xdt["/antenna_xds"] = create_antenna_xds(
        asdm, num_antenna, partition_descr, correlated_xds.polarization
    )

    # TODO:

    # gain_curve_xds

    # phase_calibration_xds

    # system_calibration_xds

    # weather_xds

    # pointing_xds

    # phased_array_xds

    # field_and_source_xds
    msv4_xdt["/field_and_source_base_xds"] = xr.Dataset(
        attrs={"type": "field_and_source"}
    )

    # info_dicts

    return msv4_xdt


def open_correlated_xds(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
) -> tuple[xr.DataTree, int]:
    datetime_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    xds = xr.Dataset(
        attrs={
            "schema_version": MSV4_SCHEMA_VERSION,
            "creator": {
                "software_name": "xradio",
                "version": importlib.metadata.version("xradio"),
            },
            "creation_date": datetime_now,
            "type": "visibility",
        }
    )

    is_single_dish = False
    coords, coord_attrs, num_antenna = create_coordinates(
        asdm, partition_descr, is_single_dish
    )
    xds = xds.assign_coords(coords)
    # print(f" =================== {xds.coords=}")
    for coord_name in coords:
        if coord_name in coord_attrs:
            xds.coords[coord_name].attrs = coord_attrs[coord_name]

    xds = xds.assign(create_data_vars(xds))

    data_group_base = {
        "correlated_data": "VISIBILITY",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "field_and_source": "field_and_source_base_xds",
        "description": "Base data group derived from data in ASDM BDFs",
        "date": datetime_now,
    }

    info_dicts = create_info_dicts(asdm, xds)
    xds.attrs.update(info_dicts)

    xds.attrs.update({"data_groups": {"base": data_group_base}})

    return xds, num_antenna


def create_data_vars(xds: xr.Dataset) -> dict[str, tuple]:
    data_vars = {}

    data_vars["VISIBILITY"] = (
        ["time", "baseline_id", "frequency", "polarization"],
        np.ones(
            (
                xds.sizes["time"],
                xds.sizes["baseline_id"],
                xds.sizes["frequency"],
                xds.sizes["polarization"],
            ),
            dtype="complex128",
        ),
        {
            "type": "quantity",
            "units": [""],  # Do the ASDM/BDFs give anything?
            "field_and_source_xds": None,
        },
    )

    data_vars["WEIGHT"] = (
        ["time", "baseline_id", "frequency", "polarization"],
        np.ones(
            (
                xds.sizes["time"],
                xds.sizes["baseline_id"],
                xds.sizes["frequency"],
                xds.sizes["polarization"],
            ),
            dtype="float64",
        ),
    )

    data_vars["FLAG"] = (
        ["time", "baseline_id", "frequency", "polarization"],
        np.ones(
            (
                xds.sizes["time"],
                xds.sizes["baseline_id"],
                xds.sizes["frequency"],
                xds.sizes["polarization"],
            ),
            dtype="bool",
        ),
    )

    data_vars["UVW"] = (
        ["time", "baseline_id", "uvw_label"],
        np.ones(
            (
                xds.sizes["time"],
                xds.sizes["baseline_id"],
                xds.sizes["uvw_label"],
            ),
            dtype="float64",
        ),
        {"type": "uvw", "frame": "icrs", "units": ["m"]},
    )

    return data_vars


def create_coordinates(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, np.ndarray],
    is_single_dish: bool = False,
) -> tuple[dict, dict]:
    """
    TODO

    Parameters
    ----------
    asdm:
        Input ASDM object
    partition_descr:
        description of partition IDs in a dictionary of "ID ASDM key/attribute" -> numeric IDs

    Returns
    -------
    dict
        coordinates dict ready to be added with 'assign_coords'
    """
    coords = {}
    attrs = {}

    # Closest to time is the Scan/startTime,endTime,etc. but time values will probably will be
    # read from the subscans BDFs?
    # This is for now very incomplete. Subscan/startTime,numIntegrations,etc.
    sdm_scan_attrs = ["execBlockId", "scanNumber", "startTime", "endTime", "numSubscan"]
    scan_df = exp_asdm_table_to_df(asdm, "Scan", sdm_scan_attrs)
    scans_metadata = scan_df.loc[
        scan_df["scanNumber"].isin(partition_descr["scanNumber"])
    ]
    time_centers = convert_time_asdm_to_unix(scans_metadata["startTime"].values)
    coords["time"] = (["time"], time_centers)
    attrs["time"] = make_time_measure_attrs("s", "tai", time_format="unix")
    attrs["time"].update(
        {"dims": ["time"], "integration_time": make_quantity(0.0, "s")}
    )
    scan_numbers = np.array(scans_metadata["scanNumber"]).astype(str)
    coords["scan_name"] = (["time"], scan_numbers)

    # baselines...
    sdm_main_attrs = ["time", "configDescriptionId", "fieldId", "numAntenna"]
    main_df = exp_asdm_table_to_df(asdm, "Main", sdm_main_attrs)
    configurations = main_df.loc[
        main_df["configDescriptionId"].isin(partition_descr["configDescriptionId"])
        & main_df["fieldId"].isin(partition_descr["fieldId"])
    ]
    num_antenna = configurations["numAntenna"].max()
    num_baselines = num_antenna * (num_antenna - 1) / 2
    # This might turn out too simplistic. We'll have to check how the baselines (auto-corrs and
    # cross-corrs) are read from the BDFs, and other factors.
    baseline_antenna1_id, baseline_antenna2_id = zip(
        *itertools.combinations_with_replacement(np.arange(num_antenna), 2)
    )
    # TODO: get proper names
    coords["baseline_antenna1_name"] = (
        ["baseline_id"],
        list([str(idx) for idx in baseline_antenna1_id]),
    )
    coords["baseline_antenna2_name"] = (
        ["baseline_id"],
        list([str(idx) for idx in baseline_antenna2_id]),
    )
    coords["baseline_id"] = np.arange(len(baseline_antenna1_id))

    # From dataDescriptionId get SPW and polarization IDs
    dd_id = partition_descr["dataDescriptionId"][0]
    sdm_dd_attrs = ["dataDescriptionId", "spectralWindowId", "polOrHoloId"]
    data_description_df = exp_asdm_table_to_df(asdm, "DataDescription", sdm_dd_attrs)
    data_description = data_description_df.loc[
        data_description_df["dataDescriptionId"] == dd_id
    ]
    spw_id = data_description["spectralWindowId"].values[0]
    pol_setup_id = data_description["polOrHoloId"].values[0]

    # polarization coord
    sdm_polarization_attrs = ["polarizationId", "numCorr", "corrType"]
    polarization_df = exp_asdm_table_to_df(asdm, "Polarization", sdm_polarization_attrs)
    polarization_metadata = polarization_df.loc[
        polarization_df["polarizationId"] == pol_setup_id
    ]
    num_corr = polarization_metadata["numCorr"].values[0]
    polarization_setup = polarization_metadata["corrType"].values[0][:num_corr]
    coords["polarization"] = polarization_setup

    # frequency coord
    sdm_spw_attrs = [
        "spectralWindowId",
        "name",
        "numChan",
        "refFreq",
    ]
    # These are optional attrs of the ASDM table, better dealt with via util functions that
    # check for their presence and alternatives: "chanFreqStart", "chanFreqStep", "chanFreqArray",
    # "chanWidthArray", "effectiveBwArray", "measFreqRef".
    spw_df = exp_asdm_table_to_df(asdm, "SpectralWindow", sdm_spw_attrs)
    spectral_window = spw_df.loc[spw_df["spectralWindowId"] == spw_id]
    spw_name = spectral_window["name"].values
    num_chan = spectral_window["numChan"].values[0]
    frequency_centers = get_spw_frequency_centers(asdm, spw_id, num_chan)

    coords["frequency"] = (["frequency"], [freq for freq in frequency_centers])
    attrs["frequency"] = make_spectral_coord_measure_attrs("Hz", observer="TOPO")
    # Other keys of the frequency coord
    frequency_other_attrs = {
        "frame": get_reference_frame(asdm, spw_id),
        "spectral_window_name": ensure_spw_name_conforms(spw_name, spw_id),
        "reference_frequency": make_spectral_coord_reference_dict(
            spectral_window["refFreq"].values[0], "Hz", "TOPO"
        ),
        "channel_width": make_quantity(get_chan_width(asdm, spw_id), "Hz"),
    }
    attrs["frequency"].update(frequency_other_attrs)

    # field_name will be created from field_and_source_xds?
    sdm_field_attrs = ["fieldId", "fieldName"]
    field_df = exp_asdm_table_to_df(asdm, "Field", sdm_field_attrs)
    fields = field_df.loc[field_df["fieldId"].isin(partition_descr["fieldId"])][
        "fieldName"
    ].values
    coords["field_name"] = (["time"], np.repeat(fields, len(time_centers)).astype(str))

    if not is_single_dish:
        coords["uvw_label"] = np.array(["u", "v", "w"])

    return coords, attrs, num_antenna
