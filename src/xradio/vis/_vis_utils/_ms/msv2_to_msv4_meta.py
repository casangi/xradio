col_to_data_variable_names = {
    "FLOAT_DATA": "SPECTRUM",
    "DATA": "VISIBILITY",
    "CORRECTED_DATA": "VISIBILITY_CORRECTED",
    "WEIGHT_SPECTRUM": "WEIGHT",
    "WEIGHT": "WEIGHT",
    "FLAG": "FLAG",
    "UVW": "UVW",
    "TIME_CENTROID": "TIME_CENTROID",
    "EXPOSURE": "EFFECTIVE_INTEGRATION_TIME",
}
col_dims = {
    "DATA": ("time", "baseline_id", "frequency", "polarization"),
    "CORRECTED_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT_SPECTRUM": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT": ("time", "baseline_id", "frequency", "polarization"),
    "FLAG": ("time", "baseline_id", "frequency", "polarization"),
    "UVW": ("time", "baseline_id", "uvw_label"),
    "TIME_CENTROID": ("time", "baseline_id"),
    "EXPOSURE": ("time", "baseline_id"),
    "FLOAT_DATA": ("time", "baseline_id", "frequency", "polarization"),
}
col_to_coord_names = {
    "TIME": "time",
    "ANTENNA1": "baseline_ant1_id",
    "ANTENNA2": "baseline_ant2_id",
}

casacore_to_msv4_measure_type = {
    "quanta": {"type": "quantity", "Ref": None},
    "direction": {"type": "sky_coord", "Ref": "frame"},
    "epoch": {"type": "time", "Ref": "scale"},
    "frequency": {"type": "spectral_coord", "Ref": "frame"},
    "position": {"type": "earth_location", "Ref": "ellipsoid"},
    "uvw": {"type": "uvw", "Ref": "frame"},
}

casacore_to_msv4_ref = {"J2000": "FK5", "ITRF": "GRS80"}

casa_frequency_frames = [
    "REST",
    "LSRK",
    "LSRD",
    "BARY",
    "GEO",
    "TOPO",
    "GALACTO",
    "LGROUP",
    "CMB",
    "Undefined",
]

casa_frequency_frames_codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 64]


def column_description_casacore_to_msv4_measure(
    casacore_column_description, ref_code=None, time_format="unix"
):
    import numpy as np

    msv4_measure = {}
    if "MEASINFO" in casacore_column_description["keywords"]:
        msv4_measure["type"] = casacore_to_msv4_measure_type[
            casacore_column_description["keywords"]["MEASINFO"]["type"]
        ]["type"]
        msv4_measure["units"] = list(
            casacore_column_description["keywords"]["QuantumUnits"]
        )

        if "TabRefCodes" in casacore_column_description["keywords"]["MEASINFO"]:
            ref_index = np.where(
                casacore_column_description["keywords"]["MEASINFO"]["TabRefCodes"]
                == ref_code
            )[0][0]
            casa_ref = casacore_column_description["keywords"]["MEASINFO"][
                "TabRefTypes"
            ][ref_index]
        else:
            if "Ref" in casacore_column_description["keywords"]["MEASINFO"]:
                casa_ref = casacore_column_description["keywords"]["MEASINFO"]["Ref"]
            elif (
                casacore_column_description["keywords"]["MEASINFO"]["type"]
                == "frequency"
            ):
                # Some MSv2 don't have the "TabRefCodes".
                ref_index = np.where(casa_frequency_frames_codes == ref_code)[0][0]
                casa_ref = casa_frequency_frames[ref_index]

        if casa_ref in casacore_to_msv4_ref:
            casa_ref = casacore_to_msv4_ref[casa_ref]
        msv4_measure[
            casacore_to_msv4_measure_type[
                casacore_column_description["keywords"]["MEASINFO"]["type"]
            ]["Ref"]
        ] = casa_ref

        if msv4_measure["type"] == "time":
            msv4_measure["format"] = "unix"
    return msv4_measure


def create_attribute_metadata(col, main_column_descriptions):
    attrs_metadata = column_description_casacore_to_msv4_measure(
        main_column_descriptions[col]
    )
    if col in ["DATA", "CORRECTED_DATA", "WEIGHT"]:
        if not attrs_metadata:
            attrs_metadata["type"] = "quanta"
            attrs_metadata["units"] = ["unkown"]

    return attrs_metadata
