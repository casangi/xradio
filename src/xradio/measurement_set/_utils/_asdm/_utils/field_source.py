import pandas as pd
import pyasdm

from xradio._utils.schema import casacore_to_msv4_measure_type


def retrieve_field_direction_frame(field_df: pd.DataFrame, field_id: int) -> str:
    try:
        direction_code = field_df.loc[field_df["fieldId"] == field_id][
            "directionCode"
        ].item()
    except AttributeError as _exc:
        direction_code = ""

    # directionCode doc: "The common reference code for directions in the row, if not J2000."
    if direction_code and direction_code != "NaN":
        # Seen so far: ICRS, J2000, which are in this map:
        direction_frame = casacore_to_msv4_measure_type["direction"]["Ref_map"].get(
            direction_code, direction_code.lower()
        )
    else:
        direction_frame = "fk5"

    return direction_frame


def get_source_direction_code(asdm: pyasdm.ASDM, source_key: tuple) -> "str":
    """
    Get the direction code from a source in an ASDM dataset.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object.
    source_key : tuple
        A tuple containing (source_id, time_interval, spectral_window_id) that identifies
        a specific source in the ASDM.

    Returns
    -------
    str
        The direction code of the source. Returns 'fk5' if no direction code exists.

    Notes
    -----
    The direction code specifies the reference frame of the source coordinates
    (e.g., 'fk5', 'icrs', etc.).
    """

    source_id = int(source_key[0])
    # ignoring interval:
    # time_interval = source_key[1]
    spectral_window_id = int(source_key[2])
    key = (
        source_id,
        source_key[1],
        pyasdm.types.Tag(f"SpectralWindow_{spectral_window_id}"),
    )

    source_tbl = asdm.getSource()
    source_row = source_tbl.getRowByKey(*key)
    if source_row.isDirectionCodeExists():
        direction_code = source_row.getDirectionCode().getName().lower()
        if direction_code == "j2000":
            direction_code = "fk5"
    else:
        direction_code = "fk5"

    return direction_code


def retrieve_source_frequency_frame(source_df: pd.DataFrame) -> str:
    try:
        frequency_ref_code = source_df["frequencyRefCode"].item()
    except AttributeError as _exc:
        frequency_ref_code = ""

    # frequencyRefCode doc: "The reference code for frequency (if not LSRK)"
    if frequency_ref_code and frequency_ref_code != "NaN":
        frequency_frame = casacore_to_msv4_measure_type["frequency"]["Ref_map"].get(
            frequency_ref_code, frequency_ref_code.lower()
        )
    else:
        frequency_frame = "lsrk"

    return frequency_frame
