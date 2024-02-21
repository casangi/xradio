from .msv2_to_msv4_meta import column_description_casacore_to_msv4_measure
from .subtables import subt_rename_ids
from ._tables.read import read_generic_table


def create_field_info(in_file, field_id):
    field_xds = read_generic_table(
        in_file,
        "FIELD",
        rename_ids=subt_rename_ids["FIELD"],
    ).sel(field_id=field_id)
    # https://stackoverflow.com/questions/53195684/how-to-navigate-a-dict-by-list-of-keys

    field_column_description = field_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    # ['DELAY_DIR', 'PHASE_DIR', 'REFERENCE_DIR', 'CODE', 'FLAG_ROW', 'NAME', 'NUM_POLY', 'SOURCE_ID', 'TIME']

    msv4_measure = column_description_casacore_to_msv4_measure(
        field_column_description["REFERENCE_DIR"],
        ref_code=getattr(field_xds.get("refdir_ref"), "data", None),
    )
    delay_dir = {
        "dims": "",
        "data": list(field_xds["delay_dir"].data[0, :]),
        "attrs": msv4_measure,
    }

    msv4_measure = column_description_casacore_to_msv4_measure(
        field_column_description["PHASE_DIR"],
        ref_code=getattr(field_xds.get("phasedir_ref"), "data", None),
    )
    phase_dir = {
        "dims": "",
        "data": list(field_xds["phase_dir"].data[0, :]),
        "attrs": msv4_measure,
    }

    msv4_measure = column_description_casacore_to_msv4_measure(
        field_column_description["DELAY_DIR"],
        ref_code=getattr(field_xds.get("delaydir_ref"), "data", None),
    )
    reference_dir = {
        "dims": "",
        "data": list(field_xds["delay_dir"].data[0, :]),
        "attrs": msv4_measure,
    }

    field_info = {
        "name": str(field_xds["name"].data),
        "code": str(field_xds["code"].data),
        "delay_direction": delay_dir,
        "phase_direction": phase_dir,
        "reference_direction": reference_dir,
        "field_id": field_id,
    }
    # xds.attrs["field_info"] = field_info

    return field_info
