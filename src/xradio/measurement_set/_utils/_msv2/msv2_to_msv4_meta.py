from xradio._utils.schema import column_description_casacore_to_msv4_measure

col_to_data_variable_names = {
    "FLOAT_DATA": "SPECTRUM",
    "DATA": "VISIBILITY",
    "CORRECTED_DATA": "VISIBILITY_CORRECTED",
    "MODEL_DATA": "VISIBILITY_MODEL",
    "WEIGHT_SPECTRUM": "WEIGHT",
    "WEIGHT": "WEIGHT",
    "FLAG": "FLAG",
    "UVW": "UVW",
    "TIME_CENTROID": "TIME_CENTROID",
    "EXPOSURE": "EFFECTIVE_INTEGRATION_TIME",
}
# With CORRECTED_DATA, the MSv2 weights are those of CORRECTED_DATA and the
# weights of DATA are 1/SIGMA^2
col_to_data_variable_names_corrected = col_to_data_variable_names | {
    "WEIGHT_SPECTRUM": "WEIGHT_CORRECTED",
    "WEIGHT": "WEIGHT_CORRECTED",
    "SIGMA_SPECTRUM": "WEIGHT",
    "SIGMA": "WEIGHT",
}
# The column read instead of a spectrum column whose read fails
spectrum_fallback_columns = {"WEIGHT_SPECTRUM": "WEIGHT", "SIGMA_SPECTRUM": "SIGMA"}
# The MSv2 weight column whose attributes a weight from SIGMA gets
sigma_attrs_columns = {"SIGMA_SPECTRUM": "WEIGHT_SPECTRUM", "SIGMA": "WEIGHT"}
col_dims = {
    "DATA": ("time", "baseline_id", "frequency", "polarization"),
    "CORRECTED_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "MODEL_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT_SPECTRUM": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT": ("time", "baseline_id", "frequency", "polarization"),
    "SIGMA_SPECTRUM": ("time", "baseline_id", "frequency", "polarization"),
    "SIGMA": ("time", "baseline_id", "frequency", "polarization"),
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


def create_attribute_metadata(col, main_column_descriptions):
    if col in sigma_attrs_columns:
        col = sigma_attrs_columns[col]
        if col not in main_column_descriptions:
            col = "WEIGHT"
    attrs_metadata = column_description_casacore_to_msv4_measure(
        main_column_descriptions[col]
    )
    if col in ["DATA", "CORRECTED_DATA", "WEIGHT"]:
        if not attrs_metadata:
            attrs_metadata["type"] = "quanta"
            attrs_metadata["units"] = "unkown"

    return attrs_metadata
