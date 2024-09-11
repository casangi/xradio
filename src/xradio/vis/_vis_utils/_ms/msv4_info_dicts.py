from xradio.vis._vis_utils._ms.subtables import subt_rename_ids
from ._tables.read import load_generic_table


def create_processor_info(in_file: str, processor_id: int):
    """
    Makes a dict with the processor info extracted from the PROCESSOR subtable.

    Returns:
    --------
    processor_info: dict
        processor description
    processor_id: int
        processor ID for one MSv4 dataset
    """

    generic_processor_xds = load_generic_table(
        in_file,
        "PROCESSOR",
        rename_ids=subt_rename_ids["PROCESSOR"],
        taql_where=f" where ROWID() = {processor_id}",
    )

    processor_info = {
        "type": generic_processor_xds["TYPE"].values[0],
        "sub_type": generic_processor_xds["SUB_TYPE"].values[0],
    }

    return processor_info
